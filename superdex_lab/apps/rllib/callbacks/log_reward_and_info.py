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

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID

########################################################################################


class LogRewardAndInfoCallbacks(DefaultCallbacks):
    """
    A RLlib callback class that will automatically log the rewards and info values for
    each episode. This is useful for a finer-grained analysis of the training process.
    """

    def on_episode_end(
        self,
        *,
        episode: EpisodeType | EpisodeV2,
        env_runner: EnvRunner = None,
        metrics_logger: MetricsLogger | None = None,
        env: gym.Env | None = None,
        env_index: int,
        rl_module: RLModule | None = None,
        worker: EnvRunner | None = None,
        base_env: BaseEnv | None = None,
        policies: dict[PolicyID, Policy] | None = None,
        **kwargs,
    ):
        """
        Upon the end of an episode, log the rewards and info values for that episode.
        """

        # NOTE: This criterion is simplistic, but good enough for now. Change as needed.
        def is_reducible(val) -> bool:
            return not isinstance(val, str)

        # Aggregate all info values per key.
        aggregate_info = {}
        for info in episode.get_infos():
            for key, value in info.items():
                if is_reducible(value):
                    buffer = aggregate_info.get(key, None)
                    if buffer is None:
                        aggregate_info[key] = buffer = [value]
                    else:
                        buffer.append(value)

        # Log the mean and stdev of each info value.
        for k, v in aggregate_info.items():
            metrics_logger.log_value(f"info_{k}_mean", np.mean(v), clear_on_reduce=True)
            metrics_logger.log_value(f"info_{k}_stdev", np.std(v), clear_on_reduce=True)

        # Log the mean and stdev rewards.
        rewards = episode.get_rewards()
        metrics_logger.log_value("reward_mean", np.mean(rewards), clear_on_reduce=True)
        metrics_logger.log_value("reward_stdev", np.std(rewards), clear_on_reduce=True)
