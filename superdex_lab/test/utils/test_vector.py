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

import pathlib
import sys

# Prepend path to root directory to allow importing the test package.
TEST_ROOT_PATH = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(TEST_ROOT_PATH))

########################################################################################

import unittest

from superdex.lab.gym.utils.vector import HybridVectorEnv
from test.mock_environments import BasicMockEnv

########################################################################################


class TestHybridVectorEnv(unittest.TestCase):
    """Test the HybridVectorEnv class."""

    def test_basics(self):
        # Test successful initialization with valid parameters.
        env_fns = [lambda local_idx=idx: BasicMockEnv(local_idx) for idx in range(8)]
        with HybridVectorEnv(env_fns, num_envs_per_worker=2) as env:
            assert env.num_envs == 8
            assert env.num_workers == 4
            assert env.num_envs_per_worker == 2
            assert env.observation_space.shape == (8, 4)
            assert env.action_space.shape == (8, 2)

            result = env.reset()
            assert result.observation.shape == env.observation_space.shape

            actions = env.action_space.sample()
            result = env.step(actions)
            assert result.observation.shape == env.observation_space.shape
            assert len(result.reward) == 8
            assert len(result.terminated) == 8
            assert len(result.truncated) == 8
            for v in result.info.values():
                assert len(v) == 8

    def test_reset_seeds(self):
        # Test reset providing no seeds, a single seed, or per-environment seed.
        env_fns = [lambda local_idx=idx: BasicMockEnv(local_idx) for idx in range(4)]
        with HybridVectorEnv(env_fns, num_envs_per_worker=2) as env:
            env.reset()  # No seed - ok
            env.reset(seed=42)  # Single seed - ok
            env.reset(seed=[42, 43, 44, 45])  # List of seeds - ok
            with self.assertRaises(ValueError):
                env.reset(seed=[42, 43])  # Wrong length

    def test_reset_mask_not_supported(self):
        # Test that reset_mask option raises an error.
        env_fns = [lambda: BasicMockEnv(0)]
        with HybridVectorEnv(env_fns, num_envs_per_worker=1) as env:
            with self.assertRaises(ValueError):
                env.reset(options={"reset_mask": [True, False, True, False]})

    def test_only_spawn_is_allowed(self):
        # Test that fork option raises an error.
        env_fns = [lambda: BasicMockEnv(0)]
        with self.assertWarnsRegex(RuntimeWarning, "spawn"):
            with HybridVectorEnv(env_fns, num_envs_per_worker=1, context="fork"):
                pass


########################################################################################

if __name__ == "__main__":
    unittest.main()
