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

"""
Restores a trained policy from an RLlib checkpoint for standalone inference.

A trained policy is more than the network weights: RLlib wraps the RLModule in two
connector pipelines that are part of the checkpoint. The env-to-module pipeline holds
observation preprocessing (notably the ``MeanStdFilter`` whose running mean/std are
*learned* during training), and the module-to-env pipeline samples actions from the
module's own distribution and rescales them into the environment's action space.

Restoring only the RLModule and hand-rolling those steps silently diverges from
training: the network sees unnormalized observations, and actions reach the
environment on the wrong scale. This module restores all three components together so
inference reproduces what training did.
"""

import logging
import pathlib

import numpy as np
from ray.rllib.connectors.env_to_module import EnvToModulePipeline, MeanStdFilter
from ray.rllib.connectors.module_to_env import ModuleToEnvPipeline
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import unbatch

torch, _ = try_import_torch()
logger = logging.getLogger(__name__)

# Layout of an RLlib checkpoint directory. Kept as literals rather than importing
# Ray's COMPONENT_* constants so the paths stay readable alongside the docs.
_RL_MODULE_SUBPATH = ("learner_group", "learner", "rl_module")
_ENV_RUNNER = "env_runner"
_ENV_TO_MODULE = "env_to_module_connector"
_MODULE_TO_ENV = "module_to_env_connector"

########################################################################################


class CheckpointPolicy:
    """
    A trained policy restored from a checkpoint, together with its connector pipelines.

    Use :meth:`new_episode` to start an episode and :meth:`compute_actions` to step it.
    Callers own the environment loop; this class only owns the observation -> action
    mapping and reproduces the transformations that were applied during training.
    """

    def __init__(
        self,
        rl_module: RLModule,
        env_to_module: EnvToModulePipeline | None,
        module_to_env: ModuleToEnvPipeline | None,
    ):
        self.rl_module = rl_module
        self.env_to_module = env_to_module
        self.module_to_env = module_to_env

        # Both pipelines are restored together or not at all, so the degraded path is
        # all-or-nothing. See `restore_policy` for when this happens.
        self._use_connectors = env_to_module is not None and module_to_env is not None

    @staticmethod
    def new_episode(
        obs, *, info: dict, observation_space, action_space
    ) -> SingleAgentEpisode:
        """
        Starts an episode seeded with a freshly reset observation and info dictionary.

        Pass the *single-environment* spaces: for a vectorized environment these are
        ``single_observation_space`` / ``single_action_space``, not the batched ones.
        """

        episode = SingleAgentEpisode(
            observation_space=observation_space,
            action_space=action_space,
        )
        episode.add_env_reset(observation=obs, infos=info)
        return episode

    def compute_actions(
        self, episodes: list[SingleAgentEpisode], *, explore: bool
    ) -> tuple[list, list, dict]:
        """
        Maps the latest observation of each episode to raw and environment actions.

        Returns per-episode policy-space actions, per-episode environment-ready actions,
        and the remaining module-to-env outputs. Callers step with the environment
        actions but record the raw actions and outputs in ``SingleAgentEpisode``.
        """

        if self._use_connectors:
            return self._compute_actions_with_connectors(episodes, explore=explore)
        return self._compute_actions_degraded(episodes, explore=explore)

    def _compute_actions_with_connectors(
        self, episodes: list[SingleAgentEpisode], *, explore: bool
    ) -> tuple[list, list, dict]:
        # `shared_data` carries state between the two pipelines within a single step
        # (for example the time-rank flag added for stateful modules), so it must be
        # created fresh per step and passed to both.
        shared_data = {}
        batch = self.env_to_module(
            episodes=episodes,
            rl_module=self.rl_module,
            explore=explore,
            shared_data=shared_data,
        )
        forward = (
            self.rl_module.forward_exploration
            if explore
            else self.rl_module.forward_inference
        )
        to_env = self.module_to_env(
            batch=forward(batch),
            episodes=episodes,
            rl_module=self.rl_module,
            explore=explore,
            shared_data=shared_data,
        )

        # `NormalizeAndClipActions` writes the environment-ready action to a separate
        # column, leaving the raw sampled action in ACTIONS. Mirror RLlib's own
        # EnvRunner and prefer the former, falling back when no rescaling applied.
        batched_actions = to_env.pop(Columns.ACTIONS)
        batched_actions_for_env = to_env.pop(Columns.ACTIONS_FOR_ENV, batched_actions)
        return unbatch(batched_actions), unbatch(batched_actions_for_env), to_env

    def _compute_actions_degraded(
        self, episodes: list[SingleAgentEpisode], *, explore: bool
    ) -> tuple[list, list, dict]:
        """
        Fallback used when a checkpoint has no stored connector state.

        Observations reach the module unprocessed and actions are not rescaled, so
        results only match training for policies trained without observation
        normalization on an environment whose action space is already [-1, 1].
        """

        obs = np.stack([episode.get_observations(-1) for episode in episodes])
        forward = (
            self.rl_module.forward_exploration
            if explore
            else self.rl_module.forward_inference
        )
        out = forward({Columns.OBS: torch.from_numpy(obs).to(torch.float32)})

        # Ask the module which distribution it was built with rather than assuming one:
        # PPO uses a diagonal Gaussian, SAC a squashed Gaussian, discrete spaces a
        # categorical. Mirrors RLlib's `GetActions` connector.
        dist_cls = (
            self.rl_module.get_exploration_action_dist_cls()
            if explore
            else self.rl_module.get_inference_action_dist_cls()
        )
        dist = dist_cls.from_logits(out[Columns.ACTION_DIST_INPUTS])
        if not explore:
            dist = dist.to_deterministic()
        actions = unbatch(convert_to_numpy(dist.sample()))
        return actions, actions, {}


########################################################################################


def restore_policy(
    checkpoint_path: pathlib.Path, module_id: str = DEFAULT_MODULE_ID
) -> CheckpointPolicy:
    """
    Restores the RLModule and both connector pipelines from a checkpoint directory.

    If the checkpoint predates connector checkpointing or was written with connector
    state excluded, no connector pair is available. Rather than fail, this warns loudly
    and returns a policy that feeds observations to the module unprocessed - enough to
    keep such checkpoints loadable, while making a suspect evaluation traceable.
    """

    rl_module = RLModule.from_checkpoint(
        checkpoint_path.joinpath(*_RL_MODULE_SUBPATH, module_id)
    )

    connector_roots = (checkpoint_path / _ENV_RUNNER, checkpoint_path)
    connector_paths = [
        (root / _ENV_TO_MODULE, root / _MODULE_TO_ENV) for root in connector_roots
    ]
    complete_pair = next(
        (paths for paths in connector_paths if all(path.is_dir() for path in paths)),
        None,
    )
    if complete_pair is None:
        connector_candidates = [path for paths in connector_paths for path in paths]
        present = [path for path in connector_candidates if path.exists()]
        if present:
            missing = [path for path in connector_candidates if not path.exists()]
            present_paths = ", ".join(str(path) for path in present)
            missing_paths = ", ".join(str(path) for path in missing)
            raise ValueError(
                f"Incomplete connector state in checkpoint {checkpoint_path}. "
                f"Present components: {present_paths}. "
                f"Missing components: {missing_paths}."
            )

        logger.warning(
            "No connector state found in checkpoint %s. Checked the env-runner and "
            "root-level checkpoint layouts. Falling back to feeding raw observations "
            "to the policy and sending actions to the environment unscaled. If this "
            "policy was trained with `normalize_observations`, or on an environment "
            "whose action space is not [-1, 1], the resulting returns and videos will "
            "NOT reflect the trained policy.",
            checkpoint_path,
        )
        return CheckpointPolicy(rl_module, None, None)

    env_to_module_path, module_to_env_path = complete_pair

    env_to_module = EnvToModulePipeline.from_checkpoint(env_to_module_path)
    module_to_env = ModuleToEnvPipeline.from_checkpoint(module_to_env_path)

    # Observation filters keep adapting their running statistics on every call by
    # default, which is right while training but wrong here: it would drift the policy's
    # preprocessing away from the checkpointed statistics and make a rollout depend on
    # how many steps preceded it. Freeze them so a checkpoint evaluates reproducibly.
    frozen = 0
    for connector in env_to_module.connectors:
        if isinstance(connector, MeanStdFilter):
            connector._update_stats = False
            frozen += 1
    if frozen:
        logger.debug("Froze running statistics on %d observation filter(s).", frozen)

    # The connectors derive their per-space state (the filter's shape, the action
    # space struct used for rescaling) from the spaces recorded in the pipeline. A
    # pipeline saved without them restores into a half-initialized state that fails
    # later inside the rollout, so surface it here instead.
    if env_to_module.input_observation_space is None:
        raise ValueError(
            f"Restored env-to-module pipeline from {env_to_module_path} has no "
            "observation space; cannot normalize observations."
        )
    if module_to_env.input_action_space is None:
        raise ValueError(
            f"Restored module-to-env pipeline from {module_to_env_path} has no "
            "action space; cannot rescale actions."
        )

    return CheckpointPolicy(rl_module, env_to_module, module_to_env)
