# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fnmatch
import json
import logging
import math
import pathlib
from copy import deepcopy

import numpy as np
from checkpoint_policy import CheckpointPolicy, restore_policy
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from gymnasium.wrappers.vector import DictInfoToList
from ray.rllib.utils.spaces.space_utils import batch, unbatch
from ray.train import Checkpoint
from ray.tune import Callback, registry
from ray.tune.experiment import Trial
from superdex.physics.viewer.utils import AnimationWriter
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

# Sidecar file written next to each checkpoint's videos, recording the authoritative
# training iteration/timestep for that checkpoint. visualize_training_history.py reads
# it to label the training-history montage instead of guessing from file mtimes. Keep
# the name in sync with `_LABEL_SIDECAR` in apps/rllib/visualize_training_history.py.
VIDEO_LABEL_SIDECAR = "video_labels.json"

# TensorBoard is a convenient preview surface, not the authoritative video artifact.
# Keeping its payload bounded avoids copying and re-encoding several gigabytes for a
# 1,000-frame 1280x720 rollout. The full-resolution, full-frame-rate MP4 written into
# the checkpoint directory is unchanged.
_TENSORBOARD_VIDEO_MAX_FRAMES = 300
_TENSORBOARD_VIDEO_MAX_SIDE = 480


def _coerce_int(value: object) -> int | None:
    """
    Coerces a training-result value to int, returning None when the value is missing
    or non-numeric. `trial.last_result` may hold None or unexpected types (dict.get
    returns None, not the default, when a key exists with a None value), so this keeps
    sidecar writing best-effort instead of raising TypeError/ValueError out of
    on_checkpoint. None is distinct from a legitimate 0 so callers can omit an
    incomplete sidecar rather than record a fake baseline.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            parsed = float(value)
            return int(parsed) if math.isfinite(parsed) else None
        except (ValueError, OverflowError):
            return None
    return None


def _prepare_tensorboard_video(
    animation: list[np.ndarray], fps: int
) -> tuple[np.ndarray, float]:
    """Returns a bounded RGB preview and an approximately duration-matched FPS."""

    preview_frame_count = min(len(animation), _TENSORBOARD_VIDEO_MAX_FRAMES)
    frame_indices = np.linspace(
        0, len(animation) - 1, preview_frame_count, dtype=np.int64
    )
    height, width = animation[0].shape[:2]
    spatial_stride = max(1, math.ceil(max(height, width) / _TENSORBOARD_VIDEO_MAX_SIDE))

    frames = []
    for frame_index in frame_indices:
        frame = animation[frame_index]
        if frame.ndim != 3:
            raise ValueError("TensorBoard video frames must use HWC layout.")
        preview = frame[::spatial_stride, ::spatial_stride]
        if preview.shape[-1] == 4:
            preview = preview[..., :3]
        frames.append(preview)

    # Stack (T, H, W, C) and transpose to (1, T, C, H, W) for add_video.
    video = np.transpose(np.stack(frames), (0, 3, 1, 2))[np.newaxis, ...]
    preview_fps = (
        fps
        if len(animation) == 1
        else fps * (preview_frame_count - 1) / (len(animation) - 1)
    )
    return video, preview_fps


########################################################################################


class CheckpointVideoGeneratorCallback(Callback):
    """
    A Ray Tune callback that generates videos at checkpoints during training. This
    callback creates videos of agent behavior at each checkpoint by:

    1. Creating an environment in a separate process using AsyncVectorEnv.
    2. Loading the policy from the checkpoint.
    3. Running the requested rollouts sequentially through that environment.
    4. Saving the resulting animations as MP4 files.
    """

    def __init__(
        self,
        count: int = 1,
        fmt: str = "mp4",
        fps: int = 30,
        pattern: str | list[str] | None = None,
        log_to_tensorboard: bool = True,
        tb_log_every: int = 1,
    ):
        """
        Initializes the callback. The callback will generate ``count`` videos per
        checkpoint for each trial whose ID matches the provided pattern. If no pattern
        is provided, all trials will generate videos. The fmt and fps parameters specify
        the format and frames per second of the generated videos.
        """

        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}.")
        self.envs = {}
        self.count = count
        self.fmt = fmt
        self.fps = fps
        self.patterns = [pattern] if isinstance(pattern, str) else pattern

        # TensorBoard video logging state.
        if tb_log_every < 1:
            raise ValueError(f"tb_log_every must be >= 1, got {tb_log_every}.")
        self.log_to_tensorboard = log_to_tensorboard
        self.tb_log_every = tb_log_every
        self.writers = {}
        self._checkpoint_counts = {}

    def on_trial_start(self, iteration: int, trials: list[Trial], trial: Trial, **info):
        """
        Called when a trial starts. Creates environments for video generation if the
        trial's environment passes the filter criteria.
        """

        # Skip this trial if its environment name doesn't match any of the patterns.
        trial_id = trial.trial_id
        if self.patterns and not any(
            fnmatch.fnmatch(trial_id, p) for p in self.patterns
        ):
            return

        # Retrieve the environment configuration from the trial. Use this to spawn the
        # environment that will be used for video generation. Note that the current
        # renderer does not support multiple contexts per process. Therefore, rendering
        # must occur in a separate process. We rely on Gymnasium's AsyncVectorEnv to
        # spawn a worker process to run the environment.

        # TODO: Figure out the proper way to do this with Ray's internals
        # (either build an Algorithm object with the trial description, or generate an
        # EnvRunnerGroup with the remote workers, and manually load & sync RLmodule from
        # checkpoints, etc.).
        env_name = trial.config["env"]
        env_cfg = deepcopy(trial.config["env_config"])
        env_cfg["render_mode"] = "rgb_array"

        # NOTE: Low-level access to Tune's registry.
        env_creator = registry._global_registry.get(registry.ENV_CREATOR, env_name)
        # Tune calls this again when a trial is restarted. Reset all per-trial state
        # best-effort so stale resources cannot block or distort the recovered run.
        self._cleanup_trial(trial_id, terminate=True)
        # Use "spawn" so each render worker is a fresh process. The Ray Tune driver is
        # multi-threaded (Ray + torch); forking it and then initializing the OpenGL/EGL
        # renderer in the child segfaults. Spawned children start clean.
        vector_env = AsyncVectorEnv(
            [lambda: env_creator(env_cfg)],
            daemon=True,
            context="spawn",
            autoreset_mode=AutoresetMode.DISABLED,
        )
        self.envs[trial_id] = DictInfoToList(vector_env)

    def _cleanup_trial(self, trial_id: str, *, terminate: bool = False) -> None:
        """Best-effort releases all resources and state held for a trial."""

        self._checkpoint_counts.pop(trial_id, None)
        env = self.envs.pop(trial_id, None)
        writer = self.writers.pop(trial_id, None)
        if env is not None:
            try:
                env.close(terminate=terminate)
            except Exception:
                logger.exception(
                    f"Failed to close render environment for trial {trial_id} "
                    f"with terminate={terminate}."
                )
        if writer is not None:
            try:
                writer.close()
            except Exception:
                logger.exception(
                    f"Failed to close TensorBoard writer for trial {trial_id}."
                )

    def on_trial_complete(
        self, iteration: int, trials: list["Trial"], trial: "Trial", **info
    ):
        """
        Called when a trial completes. Closes and cleans up the environments and writer
        created for video generation.
        """

        self._cleanup_trial(trial.trial_id)

    def on_trial_error(
        self, iteration: int, trials: list["Trial"], trial: "Trial", **info
    ):
        """
        Called when a trial errors. Cleans up the same resources as on completion so a
        failed trial does not leak render workers or event writers.
        """

        self._cleanup_trial(trial.trial_id)

    def on_experiment_end(self, trials: list["Trial"], **info):
        """
        Called when the experiment ends. Releases any resources still held for trials
        that never reached a complete/error callback.
        """

        for trial_id in (
            set(self.envs) | set(self.writers) | set(self._checkpoint_counts)
        ):
            self._cleanup_trial(trial_id)

    def on_checkpoint(
        self,
        iteration: int,
        trials: list[Trial],
        trial: Trial,
        checkpoint: Checkpoint,
        **info,
    ):
        """
        Called when a checkpoint is created. Generates videos by loading the policy from
        the checkpoint and running it on the environments until completion.
        """

        # Skip this trial if it does not have an environment for video generation.
        if trial.trial_id not in self.envs:
            return

        # Restore the policy together with its connector pipelines, so that the video
        # reflects the same observation normalization and action scaling as training.
        checkpoint_path = pathlib.Path(checkpoint.path)
        policy = restore_policy(checkpoint_path)

        # Roll the policy out and write the resulting animations to disk.
        animations = self._render_rollout(self.envs[trial.trial_id], policy)
        writer = AnimationWriter(checkpoint_path, self.fps, self.fmt)
        for index, animation in enumerate(animations):
            writer.add(animation)
            writer.write(f"video_{index:03d}")
        writer.flush()

        # Ray runs on_checkpoint before the checkpoint is registered, so neither the
        # label sidecar nor TensorBoard logging may raise and block the checkpoint from
        # being saved; both helpers handle their own failures best-effort.
        self._write_label_sidecar(trial, checkpoint_path)
        if self.log_to_tensorboard:
            self._safe_log_videos_to_tensorboard(trial, animations, checkpoint_path)

        logger.info(f"Checkpoint videos generated for trial {trial.trial_id}")

    def _render_rollout(
        self, env: AsyncVectorEnv, policy: CheckpointPolicy
    ) -> list[list[np.ndarray]]:
        """
        Runs the requested rollouts sequentially and returns one RGB-frame list per
        rollout. A single vector worker is reused to keep renderer process isolation
        without creating multiple graphics contexts.
        """

        animations = []
        for _ in range(self.count):
            obs, infos = env.reset()
            frames = [env.render()[0]]
            episode = policy.new_episode(
                unbatch(obs)[0],
                info=infos[0],
                observation_space=env.single_observation_space,
                action_space=env.single_action_space,
            )

            while True:
                raw_actions, env_actions, extra_outs = policy.compute_actions(
                    [episode], explore=False
                )
                obs, reward, terminated, truncated, infos = env.step(batch(env_actions))
                frames.append(env.render()[0])
                episode.add_env_step(
                    unbatch(obs)[0],
                    raw_actions[0],
                    reward[0],
                    infos=infos[0],
                    terminated=terminated[0],
                    truncated=truncated[0],
                    extra_model_outputs={k: v[0] for k, v in extra_outs.items()},
                )
                if terminated[0] or truncated[0]:
                    break

            animations.append(frames)
        return animations

    def _write_label_sidecar(self, trial: Trial, checkpoint_path: pathlib.Path) -> None:
        """
        Writes a small JSON sidecar next to the checkpoint videos recording the
        authoritative training iteration and timestep from `trial.last_result`. The
        training-history tool reads this to label the montage, avoiding unreliable
        mtime-based guessing. Best-effort: logs and returns on failure so checkpoint
        handling continues.
        """

        last_result = getattr(trial, "last_result", None) or {}
        iteration = _coerce_int(last_result.get("training_iteration"))
        timestep = _coerce_int(last_result.get("num_env_steps_sampled_lifetime"))
        # Skip the sidecar unless both metrics are available: a partially populated
        # sidecar would be read back as an authoritative `iter=0, timestep=0` label,
        # suppressing the result.json/progress.csv fallback in the history tool.
        if iteration is None or timestep is None:
            logger.warning(
                f"Skipping video label sidecar for trial {trial.trial_id} at "
                f"{checkpoint_path}: training_iteration={iteration}, "
                f"num_env_steps_sampled_lifetime={timestep}."
            )
            return

        label = {
            "training_iteration": iteration,
            "num_env_steps_sampled_lifetime": timestep,
        }
        try:
            with (checkpoint_path / VIDEO_LABEL_SIDECAR).open("w") as handle:
                json.dump(label, handle)
        except OSError:
            logger.exception(
                f"Failed to write video label sidecar for trial {trial.trial_id} "
                f"at {checkpoint_path}."
            )

    def _safe_log_videos_to_tensorboard(
        self,
        trial: Trial,
        animations: list[list[np.ndarray]],
        checkpoint_path: pathlib.Path,
    ) -> None:
        """
        Best-effort wrapper around `_log_videos_to_tensorboard`. A SummaryWriter,
        MoviePy, or add_video failure must not propagate out of on_checkpoint and
        prevent the checkpoint from being registered, so failures are logged and the
        bad writer is dropped/closed so a fresh one is created next checkpoint.
        """

        try:
            self._log_videos_to_tensorboard(trial, animations, checkpoint_path)
        except Exception:
            logger.exception(
                f"Failed to log checkpoint videos to TensorBoard for trial "
                f"{trial.trial_id} at {checkpoint_path}; dropping the writer and "
                "continuing checkpoint handling."
            )
            writer = self.writers.pop(trial.trial_id, None)
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    logger.exception(
                        f"Failed to close TensorBoard writer for trial "
                        f"{trial.trial_id}."
                    )

    def _log_videos_to_tensorboard(
        self,
        trial: Trial,
        animations: list[list[np.ndarray]],
        checkpoint_path: pathlib.Path,
    ) -> None:
        """
        Logs the given rollout animations to TensorBoard as video summaries. A
        SummaryWriter is created lazily per trial, writing into the Tune-managed trial
        directory so the videos share the same TensorBoard run as the scalar metrics.
        The first checkpoint is always logged, then every `tb_log_every` checkpoints.
        Requires the `moviepy` package to encode the summaries; if it is missing,
        TensorBoard silently drops them.
        """

        # Throttle logging to every `tb_log_every` checkpoints. The counter is
        # zero-based so the first checkpoint (the pre-training baseline) is always
        # logged, and every `tb_log_every`-th checkpoint thereafter.
        trial_id = trial.trial_id
        count = self._checkpoint_counts.get(trial_id, 0)
        self._checkpoint_counts[trial_id] = count + 1
        if count % self.tb_log_every != 0:
            return

        # Lazily create a writer pointing at the Tune-managed trial directory, where
        # Ray Tune also writes its scalar event file, so the videos share the same
        # TensorBoard run. Fall back to the checkpoint's parent directory if the trial
        # directory is unavailable.
        writer = self.writers.get(trial_id)
        if writer is None:
            trial_dir = getattr(trial, "local_path", None) or getattr(
                trial, "logdir", None
            )
            logdir = str(trial_dir) if trial_dir else str(checkpoint_path.parent)
            writer = SummaryWriter(logdir=logdir)
            self.writers[trial_id] = writer

        # Align the video step with the scalar metrics' x-axis. Ray/Tune's TensorBoard
        # logger prefers `timesteps_total`, then `num_env_steps_sampled_lifetime`,
        # falling back to `training_iteration`; mirror that order so the videos land on
        # the same x-axis as the scalars they accompany. Explicit None checks (not
        # `or`) keep a legitimate step of 0 at the pre-training baseline.
        last_result = getattr(trial, "last_result", None) or {}
        step = 0
        for key in (
            "timesteps_total",
            "num_env_steps_sampled_lifetime",
            "training_iteration",
        ):
            value = last_result.get(key)
            if value is not None:
                step = int(value)
                break

        # Log each rollout separately: episodes may have different lengths and thus
        # cannot be batched into a single (N, T, C, H, W) tensor.
        for index, animation in enumerate(animations):
            if len(animation) == 0:
                continue
            video, preview_fps = _prepare_tensorboard_video(animation, self.fps)
            writer.add_video(
                f"rollout/video_{index:03d}",
                video,
                global_step=step,
                fps=preview_fps,
            )
        writer.flush()
