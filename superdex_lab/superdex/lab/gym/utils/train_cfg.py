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

"""Shared training configuration for the SuperDex Gym sample-training entrypoints.

Kept separate from the RLlib training script so it can be imported without pulling in
``ray`` and the rest of the training stack.
"""

from __future__ import annotations

import pathlib
from typing import Literal

from superdex.physics.utils.configclasses import configclass


@configclass
class TrainCfg:
    """
    Configuration for the train function. These settings are used to a) determine which
    samples to train and b) setting some shared user-configurable training parameters
    (such as checkpoint frequency and number of environment runners). PPO is the
    recommended workflow. SAC is an experimental RLlib baseline whose bundled settings
    and stopping thresholds have not been tuned or validated for the included
    environments. Per-environment ``ppo`` overrides are ignored when using SAC.
    """

    num_env_runners: int
    """Number of environment runners. Determines the number of concurrent environments
    generating experiences to feed the training algorithm."""
    num_learners: int
    """Number of learners for distributed training."""
    checkpoint_freq: int
    """Frequency at which checkpoints are generated, in terms of training iterations."""
    pattern: str
    """Pattern determining which samples to train. Use "*" to train all of them."""
    output_path: pathlib.Path | None
    """Output path to the training results. If not provided, the default path will be
    used instead (~/ray_results/)."""
    video_on_checkpoint: bool
    """If True, a video will be generated for each checkpoint."""
    algorithm: Literal["PPO", "SAC"]
    """Algorithm to use for training. PPO is recommended. SAC is experimental and
    ignores any overrides under the ``ppo`` recipe key."""
    profile: bool
    """If True, enables environment profiling and dumps timing information to the info
    dict. Applies to all environments (profiling is implemented in the base MochiEnv)."""
    max_iterations: int
    """Maximum number of training iterations per experiment. If <= 0, no iteration limit
    is applied (training stops only on the sample's reward / env-steps criteria)."""
