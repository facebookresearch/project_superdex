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

from __future__ import annotations

import unittest

import superdex.lab.gym as gym_namespace
import superdex.physics as physics
from superdex.lab.gym.envs.benchmarks.cartpole_env import CartPoleEnv, CartPoleEnvCfg


class SuperdexLabTest(unittest.TestCase):
    def tearDown(self) -> None:
        if physics.is_initialized():
            physics.shutdown()

    def test_canonical_namespace_exports_env_symbols(self) -> None:
        self.assertIs(gym_namespace.MochiEnv, gym_namespace.envs.MochiEnv)

    def test_cart_pole_steps_once(self) -> None:
        with CartPoleEnv(CartPoleEnvCfg()) as env:
            reset_result = env.reset()
            step_result = env.step(env.action_space.sample())

        self.assertIsNotNone(reset_result.observation)
        self.assertIsNotNone(step_result.observation)
