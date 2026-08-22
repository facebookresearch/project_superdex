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

import logging
from collections.abc import Callable, Mapping
from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
from ray import tune
from superdex.lab.gym.utils.env_discovery import register_all_envs

logger = logging.getLogger(__name__)

########################################################################################


def _superdex_env_creator(
    env_spec: EnvSpec,
) -> Callable[[Mapping[str, Any]], gym.Env]:
    """Create an RLlib factory that preserves a SuperDex spec's config defaults."""

    def create(env_config: Mapping[str, Any]) -> gym.Env:
        merged_cfg = dict(env_spec.kwargs.get("cfg", {}))
        merged_cfg.update(env_config)
        return env_spec.make(cfg=merged_cfg)

    return create


def register_envs(import_sample_envs: bool = True):
    """
    Registers the Gymnasium environments in the Ray Tune registry. While RLlib supports
    the instantiation of Gymnasium environments, it does not correctly handle the
    configuration of these environments. This function registers the environments with
    a wrapper that provides the configuration as keyword arguments to the constructor.
    Optionally, this function also discovers and registers the SuperDex Gym sample
    environments with Gymnasium. Necessary for training the benchmarks.
    """

    if import_sample_envs:
        register_all_envs()

    for id, env_spec in gym.envs.registration.registry.items():
        # SuperDex Gym environments receive a single configuration mapping. Preserve
        # variant defaults registered on the Gymnasium spec while allowing Ray overrides.
        if id.startswith("superdex_gym/"):
            tune.register_env(id, _superdex_env_creator(env_spec))

        # General Gymnasium environments. Configuration is typically provided as keyword
        # arguments to the constructor, rather than with a configuration dictionary.
        # This is not supported by RLlib, so we need to wrap the instantiation of these
        # environments to unpack the configuration to keyword arguments.
        # See https://github.com/ray-project/ray/issues/8570.
        # TODO: Remove if RLlib ever introduces a way to provide ctor args.
        else:
            tune.register_env(id, lambda cfg, env_spec=env_spec: env_spec.make(**cfg))
