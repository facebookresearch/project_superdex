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

"""This scripts demonstrates the application of the RLlib library to train a policy for
the sample environments. For this, we rely on the tune.run and experiments API to define
and train simultaneously multiple environments.

The set of trainable environments is discovered automatically: any environment (base or
config variant) that ships a ``<env_module>[_<variant>].train.json`` recipe next to its
module is trainable, and its RLlib configuration is built from that file. There is no
hard-coded list of environments, so environments absent from a build are simply not
trainable -- no per-environment gating is required.

A recipe holds only training settings. Environment configuration belongs in a gym config
variant (``<env_module>_<variant>.json``), so that every configuration that gets trained
is also a nameable, runnable environment rather than a setup that exists only inside a
recipe.

Recipes are written against RLlib's new API stack, so settings are expressed as their
new-stack equivalents: observation normalization is requested with
``normalize_observations``, which installs a ``MeanStdFilter`` connector.

PPO is the recommended workflow. SAC is available as an experimental RLlib baseline.
Its bundled settings and stopping thresholds have not been tuned or validated for the
included environments, and per-environment ``ppo`` overrides are ignored.
"""

import argparse
import fnmatch
import pathlib
import warnings
from typing import Any

import ray
import ray.tune
from callbacks import CheckpointVideoGeneratorCallback, LogRewardAndInfoCallbacks
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.rllib.connectors.env_to_module import MeanStdFilter
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.train import CheckpointConfig
from ray.tune.experiment import Experiment, Trial
from ray.tune.utils.log import Verbosity
from superdex.lab.gym.utils.env_discovery import (
    EnvEntry,
    get_env_entries,
    load_entry_config,
)
from superdex.lab.gym.utils.train_cfg import TrainCfg
from superdex.physics.viewer import VIEWER_AVAILABLE
from utils import register_envs

########################################################################################

# Filename ``kind`` of the per-env training recipe (``<env_module>[_<variant>].train.json``).
TRAIN_CONFIG_KIND = "train"


def discover_trainable_envs() -> dict[str, tuple[EnvEntry, dict[str, Any]]]:
    """Discover environments that ship a ``.train.json`` recipe.

    Returns a mapping of env short name -> (entry, parsed train recipe). Config variants
    are trainable too, each from its own ``<module>_<variant>.train.json``; environments
    without a recipe are not trainable. Test-only variants are excluded: they are
    degenerate configurations kept as crash checks, not tasks.
    """
    trainable: dict[str, tuple[EnvEntry, dict[str, Any]]] = {}
    for entry in get_env_entries():
        if entry.test_only:
            continue
        recipe = load_entry_config(entry, TRAIN_CONFIG_KIND)
        if recipe:
            trainable[entry.short_name] = (entry, recipe)
    return trainable


def train_samples(train_cfg: TrainCfg):
    """
    Entrypoint to the sample training script. Selects, generates and runs the samples
    according to the supplied training configuration.
    """

    # Register environments (both with Gymnasium and the Ray Tune registry).
    register_envs(import_sample_envs=True)

    # Validate configuration.
    # Limit the number of environment runners to the number of available CPUs.
    # Note that we subtract the learners to leave CPUs for them.
    resources = ray.cluster_resources()
    num_cpus = int(resources["CPU"])

    # The learners must leave at least one CPU for an environment runner; otherwise the
    # runner cap below would be <= 0. Fail fast with a clear message instead of silently
    # producing a confusing downstream failure.
    if train_cfg.num_learners >= num_cpus:
        raise ValueError(
            f"Number of learners ({train_cfg.num_learners}) must be smaller than the "
            f"number of available CPUs ({num_cpus}) to leave room for at least one "
            f"environment runner."
        )

    if train_cfg.num_env_runners + train_cfg.num_learners >= num_cpus:
        capped_num_env_runners = max(1, num_cpus - train_cfg.num_learners)
        print(
            f"WARNING: Combined number of environment runners "
            f"({train_cfg.num_env_runners}) and learners ({train_cfg.num_learners}) "
            f"reaches or exceeds the number of available CPUs ({num_cpus}). Limiting "
            f"environment runners to {capped_num_env_runners}..."
        )
        train_cfg.num_env_runners = capped_num_env_runners

    # Discover trainable environments from their train.json recipes.
    trainable = discover_trainable_envs()

    # Determine samples to run.
    train_names = fnmatch.filter(trainable.keys(), train_cfg.pattern)
    if len(train_names) == 0:
        print("No samples to train, exitting...")
        exit(-1)

    # Generate Tune callbacks.
    callbacks = []
    if train_cfg.video_on_checkpoint:
        if not VIEWER_AVAILABLE:
            warnings.warn(
                "Renderer for SuperDex Gym environments is not available. The training "
                "script will fail on checkpoint if a SuperDex Gym environment is used. "
                "Disabling video generation on checkpoint to avoid this failure.",
                stacklevel=2,
            )
        else:
            callbacks.append(CheckpointVideoGeneratorCallback())

    # Run training for the selected samples.
    train_experiments = [
        build_experiment(train_cfg, *trainable[name]) for name in train_names
    ]
    ray.tune.run(
        train_experiments,
        verbose=Verbosity.V1_EXPERIMENT,
        callbacks=callbacks,
    )


########################################################################################
# Experiment construction from train.json recipes
########################################################################################


def build_experiment(
    train_cfg: TrainCfg, entry: EnvEntry, recipe: dict[str, Any]
) -> Experiment:
    """Build an RLlib experiment for ``entry`` from its ``train.json`` recipe."""

    if "env_config" in recipe:
        # Env configuration is a property of the environment, not of a training run: a
        # recipe-only override silently trains a configuration that is not discoverable,
        # runnable, or smoke-tested anywhere else.
        raise ValueError(
            f"Training recipe for '{entry.short_name}' contains an 'env_config' section. "
            "Environment configuration must live in a gym config variant "
            "('<env_module>_<variant>.json'); point the recipe at that variant by naming "
            "it '<env_module>_<variant>.train.json' instead."
        )

    supported_keys = {"description", "normalize_observations", "stop_criteria", "ppo"}
    unknown_keys = set(recipe) - supported_keys
    if unknown_keys:
        # Raise rather than ignore, matching the section guards below: a misspelled
        # 'stop_criteria' would otherwise leave the recipe trainable with no stop
        # condition, which only shows up as a run that never ends.
        raise ValueError(
            f"Unknown keys in the training recipe for '{entry.short_name}': "
            f"{', '.join(sorted(unknown_keys))}. "
            f"Supported keys: {', '.join(sorted(supported_keys))}."
        )

    # Start from the default configuration for the selected algorithm.
    cfg = default_config(train_cfg)

    # Observation normalization is algorithm-agnostic, so it is applied before the
    # algorithm-specific overrides below. RLlib calls the factory as
    # ``(env, spaces, device)`` and builds it once per EnvRunner, so each runner owns one
    # filter whose running statistics are checkpointed with that runner's state.
    if recipe.get("normalize_observations", False):
        cfg.env_runners(
            env_to_module_connector=lambda env, spaces, device: MeanStdFilter()
        )

    # Apply PPO-specific overrides (if any). SAC uses the defaults for every env.
    if train_cfg.algorithm == "PPO":
        _apply_ppo_overrides(cfg, recipe.get("ppo", {}), train_cfg)

    # The env config comes from the entry's config variant, spelled out explicitly rather
    # than left to the registered spec defaults: `run_inference.py` rebuilds the env from
    # the `env_config` recorded in the checkpoint's params.json, and it replaces the spec's
    # own `cfg` when doing so. Profiling is a runtime toggle rather than task
    # configuration, so it is injected from the train config; it lives in the base
    # MochiEnv, so it applies to every environment.
    env_config = {
        **entry.cfg_kwargs,
        "profile": train_cfg.profile,
        "dump_timings_to_info": train_cfg.profile,
    }
    cfg.environment(entry.env_id, env_config=env_config)

    return make_experiment(train_cfg, entry.short_name, cfg, _stop_criteria(recipe))


def _apply_ppo_overrides(
    cfg: PPOConfig, ppo: dict[str, Any], train_cfg: TrainCfg
) -> None:
    """Apply the ``ppo`` section of a train recipe onto a PPO config in place."""
    if not ppo:
        return

    supported_keys = {
        "env_runners",
        "training",
        "train_batch_size_per_runner",
    }
    unknown_keys = set(ppo) - supported_keys
    if unknown_keys:
        # Raise rather than ignore, matching _stop_criteria below: a silently dropped
        # override leaves training running on defaults with no indication why.
        raise ValueError(
            f"Unknown ppo override keys: {', '.join(sorted(unknown_keys))}. "
            f"Supported keys: {', '.join(sorted(supported_keys))}."
        )

    if "env_runners" in ppo:
        cfg.env_runners(**ppo["env_runners"])

    training = dict(ppo.get("training", {}))
    per_runner = ppo.get("train_batch_size_per_runner")
    if per_runner is not None:
        if "train_batch_size" in training:
            # The same setting expressed two ways; applying one would silently discard
            # the other, which is the failure mode the unknown-key check above prevents.
            raise ValueError(
                "Training recipe sets both 'ppo.training.train_batch_size' and "
                "'ppo.train_batch_size_per_runner', which configure the same value. "
                "Use the per-runner form to scale with --num_env_runners, or the "
                "fixed form to pin an absolute batch size."
            )
        # ``train_batch_size_per_runner`` scales with the number of runners at runtime.
        training["train_batch_size"] = per_runner * train_cfg.num_env_runners
    if training:
        cfg.training(**training)


def _stop_criteria(recipe: dict[str, Any]) -> dict[str, float]:
    """Translate a recipe's human-readable ``stop_criteria`` into RLlib metric keys."""
    criteria = recipe.get("stop_criteria", {})
    supported_keys = {
        "episode_return_mean",
        "num_env_steps_sampled_lifetime",
    }
    unknown_keys = set(criteria) - supported_keys
    if unknown_keys:
        raise ValueError(
            f"Unknown stop criteria: {', '.join(sorted(unknown_keys))}. "
            f"Supported criteria: {', '.join(sorted(supported_keys))}."
        )

    stop: dict[str, float] = {}
    if "episode_return_mean" in criteria:
        stop[f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}"] = criteria[
            "episode_return_mean"
        ]
    if "num_env_steps_sampled_lifetime" in criteria:
        stop[f"{NUM_ENV_STEPS_SAMPLED_LIFETIME}"] = criteria[
            "num_env_steps_sampled_lifetime"
        ]
    return stop


########################################################################################
# Auxiliary functions for defining the experiments
########################################################################################


def make_experiment(
    train_cfg: TrainCfg,
    name: str,
    cfg: PPOConfig | SACConfig,
    stop_criteria: dict[str, float],
) -> Experiment:
    """
    Builds an experiment with the provided name, configuration and stop criteria.
    """

    # Provide a more descriptive name for the trials rather than the standard one.
    def trial_name_generator(trial: Trial) -> str:
        return f"{name}_{trial.trial_id}"

    checkpoint_config = CheckpointConfig(
        checkpoint_frequency=train_cfg.checkpoint_freq,
        checkpoint_at_end=True,
    )

    storage_path = None
    if train_cfg.output_path is not None:
        storage_path = str(train_cfg.output_path.resolve())

    # Cap the number of training iterations when requested, without mutating the shared
    # stop-criteria constant.
    stop = dict(stop_criteria)
    if train_cfg.max_iterations > 0:
        stop["training_iteration"] = train_cfg.max_iterations

    # Generate experiment instance with our custom settings.
    # NOTE: "name" must be the same for all experiments - Otherwise Tune will only write
    # the results for a single experiment. Possibly a bug in Tune?
    return Experiment(
        name=train_cfg.algorithm,
        run=train_cfg.algorithm,
        config=cfg.to_dict(),
        stop=stop,
        trial_name_creator=trial_name_generator,
        trial_dirname_creator=trial_name_generator,
        checkpoint_config=checkpoint_config,
        storage_path=storage_path,
    )


def default_config(train_cfg: TrainCfg) -> PPOConfig | SACConfig:
    """
    Returns the default configuration for the selected algorithm.
    """
    match train_cfg.algorithm:
        case "PPO":
            return default_ppo_config(train_cfg)
        case "SAC":
            return default_sac_config(train_cfg)
        case _:
            raise ValueError(f"Unknown algorithm: {train_cfg.algorithm}")


def default_ppo_config(train_cfg: TrainCfg) -> PPOConfig:
    """
    Default PPO configuration for the samples. Adapted from RLlib's tuned examples.
    https://github.com/ray-project/ray/tree/master/rllib/examples/algorithms/ppo
    """

    cfg = PPOConfig()
    cfg.env_runners(
        num_env_runners=train_cfg.num_env_runners,
        rollout_fragment_length=512,
    )
    cfg.learners(
        num_learners=train_cfg.num_learners,
        num_gpus_per_learner=0,
    )
    cfg.callbacks(LogRewardAndInfoCallbacks)
    cfg.training(
        lambda_=0.95,
        lr=0.0003,
        minibatch_size=4096,
        num_epochs=15,
        train_batch_size=32 * 512,
        vf_loss_coeff=0.01,
    )
    cfg.rl_module(
        model_config={
            "fcnet_activation": "tanh",
            "fcnet_hiddens": [64, 64],
            "vf_share_layers": True,
        }
    )
    cfg.reporting(
        metrics_num_episodes_for_smoothing=5,
        min_sample_timesteps_per_iteration=1000,
    )
    cfg.evaluation(
        evaluation_config={"explore": False},
        evaluation_duration=1,
        evaluation_duration_unit="episodes",
        evaluation_interval=1,
        evaluation_num_env_runners=1,
        evaluation_parallel_to_training=True,
    )
    return cfg


def default_sac_config(train_cfg: TrainCfg) -> SACConfig:
    """
    Shared experimental SAC baseline adapted from RLlib examples. These settings have
    not been tuned or validated for the SuperDex Gym environments.
    https://github.com/ray-project/ray/tree/master/rllib/examples/algorithms/sac
    """

    cfg = SACConfig()
    cfg.env_runners(
        num_env_runners=train_cfg.num_env_runners,
        rollout_fragment_length=1,
    )
    cfg.learners(
        num_learners=train_cfg.num_learners,
        num_gpus_per_learner=0,
    )
    cfg.callbacks(DefaultCallbacks)
    cfg.training(
        initial_alpha=0.1,
        actor_lr=3e-5,
        critic_lr=3e-4,
        alpha_lr=1e-4,
        target_entropy="auto",
        n_step=1,
        tau=0.005,
        train_batch_size=256,
        target_network_update_freq=1,
        replay_buffer_config={
            "type": "EpisodeReplayBuffer",
            "capacity": int(1e5),
            "batch_size_B": 256,
            "batch_length_T": 1,
        },
        num_steps_sampled_before_learning_starts=1024,
    )
    cfg.reporting(
        metrics_num_episodes_for_smoothing=5,
        min_sample_timesteps_per_iteration=1000,
    )
    cfg.evaluation(
        evaluation_config={"explore": False},
        evaluation_duration="auto",
        evaluation_interval=1,
        evaluation_num_env_runners=1,
        evaluation_parallel_to_training=True,
    )
    return cfg


########################################################################################

if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Train SuperDex Gym sample environments with RLlib. PPO is the "
        "recommended workflow. SAC is experimental, and its bundled settings and "
        "stopping thresholds have not been tuned or validated for these environments."
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "SAC"],
        help="Training algorithm. PPO is recommended. SAC is experimental and ignores "
        "any overrides under the 'ppo' key in the environment's training recipe.",
    )
    parser.add_argument(
        "-n",
        "--num_env_runners",
        type=int,
        default=32,
        help="Number of environment runners for parallel experience collection",
    )
    parser.add_argument(
        "-nl",
        "--num_learners",
        type=int,
        default=1,
        help="Number of learners for distributed training",
    )
    parser.add_argument(
        "-cf",
        "--checkpoint_freq",
        type=int,
        default=10,
        help="Frequency of checkpoint generation (in training iterations)",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="*",
        help="Pattern to select which samples to train (use '*' for all)",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=pathlib.Path,
        default=None,
        help="Output path for training results (defaults to ~/ray_results/)",
    )
    parser.add_argument(
        "-vid",
        "--video_on_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate video for each checkpoint (off by default, since encoding adds "
        "per-checkpoint overhead)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable environment profiling and dump timing information to the info dict",
    )
    parser.add_argument(
        "-mi",
        "--max_iterations",
        type=int,
        default=0,
        help="Maximum number of training iterations per experiment (<=0 means no limit)",
    )
    args = parser.parse_args()

    ray.init()

    # Run samples training.
    train_cfg = TrainCfg(
        algorithm=args.algorithm,
        num_env_runners=args.num_env_runners,
        num_learners=args.num_learners,
        checkpoint_freq=args.checkpoint_freq,
        pattern=args.pattern,
        output_path=args.output_path,
        video_on_checkpoint=args.video_on_checkpoint,
        profile=args.profile,
        max_iterations=args.max_iterations,
    )
    train_samples(train_cfg)
