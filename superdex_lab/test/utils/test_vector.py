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

import unittest

from gymnasium.vector import AutoresetMode
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

    def test_call_flattens_results_from_every_worker(self) -> None:
        env_fns = [lambda local_idx=idx: BasicMockEnv(local_idx) for idx in range(8)]
        with HybridVectorEnv(env_fns, num_envs_per_worker=2) as env:
            self.assertEqual(tuple(range(8)), env.call("get_env_id"))

    def test_reset_seeds(self):
        # Test reset providing no seeds, a single seed, or per-environment seed.
        env_fns = [lambda local_idx=idx: BasicMockEnv(local_idx) for idx in range(4)]
        with HybridVectorEnv(env_fns, num_envs_per_worker=2) as env:
            env.reset()  # No seed - ok
            env.reset(seed=42)  # Single seed - ok
            env.reset(seed=[42, 43, 44, 45])  # List of seeds - ok
            with self.assertRaises(ValueError):
                env.reset(seed=[42, 43])  # Wrong length

    def test_scalar_seed_expands_to_globally_unique_streams(self):
        # A scalar seed must expand to distinct per-env seeds across every worker, not
        # overlap between adjacent workers (outer + inner double offset).
        env_fns = [lambda local_idx=idx: BasicMockEnv(local_idx) for idx in range(6)]
        with HybridVectorEnv(env_fns, num_envs_per_worker=2) as env:
            env.reset(seed=100)
            seeds = env.call("get_last_seed")
            self.assertEqual(len(seeds), 6)
            self.assertNotIn(None, seeds)
            self.assertEqual(len(set(seeds)), 6)

    def test_published_autoreset_mode_matches_inner_mode(self):
        # The wrapper must advertise the inner autoreset mode, not the outer env's
        # DISABLED mode used to hand autoreset to the inner SyncVectorEnvs.
        env_fns = [lambda local_idx=idx: BasicMockEnv(local_idx) for idx in range(4)]
        with HybridVectorEnv(env_fns, num_envs_per_worker=2) as env:
            self.assertEqual(env.metadata["autoreset_mode"], AutoresetMode.NEXT_STEP)
        with HybridVectorEnv(
            env_fns, num_envs_per_worker=2, autoreset_mode=AutoresetMode.SAME_STEP
        ) as env:
            self.assertEqual(env.metadata["autoreset_mode"], AutoresetMode.SAME_STEP)

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
