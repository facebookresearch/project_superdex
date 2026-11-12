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
from unittest import mock

import gymnasium as gym
from apps.rllib import utils as rllib_utils
from apps.rllib.checkpoint_env import (
    make_checkpoint_env,
    prepare_checkpoint_env_config,
    resolve_checkpoint_env_spec,
)
from apps.rllib.utils import _superdex_env_creator, register_envs
from gymnasium.envs.registration import EnvSpec
from superdex.lab.gym.registration import register_envs as register_gym_envs
from test.envs.registry_test_utils import restore_gym_registry, snapshot_gym_registry


class RegistryAndCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        snapshot = snapshot_gym_registry()
        self.addCleanup(restore_gym_registry, snapshot)

    def test_ray_registration_registers_only_superdex_specs(
        self,
    ) -> None:
        gym.registry.clear()
        lookalike = EnvSpec("other/Lookalike-v0", "example.module:Environment")
        gym.registry[lookalike.id] = lookalike
        captured: dict[str, object] = {}
        sentinel = object()

        def capture(env_id: str, creator: object) -> None:
            if env_id == "superdex_gym/CartPole-v0":
                self.assertIn(env_id, gym.registry)
            captured[env_id] = creator

        with (
            mock.patch.object(rllib_utils.tune, "register_env", side_effect=capture),
            mock.patch.object(
                rllib_utils, "_superdex_env_creator", return_value=sentinel
            ) as superdex_creator,
        ):
            register_envs()

        self.assertIs(sentinel, captured["superdex_gym/CartPole-v0"])
        # A spec outside the superdex_gym namespace gets no Tune creator: register_envs()
        # only exposes SuperDex-namespaced specs to Tune.
        self.assertNotIn(lookalike.id, captured)
        self.assertIn(
            gym.spec("superdex_gym/CartPole-v0"),
            (call.args[0] for call in superdex_creator.call_args_list),
        )

    def test_ray_creator_forwards_env_config_to_spec_make(self) -> None:
        overrides = {"replace": {"user": True}, "added": [2]}
        spec = mock.Mock(id="superdex_gym/Test-v0")
        sentinel = object()
        spec.make.return_value = sentinel

        created = _superdex_env_creator(spec)(overrides)

        self.assertIs(sentinel, created)
        spec.make.assert_called_once_with(replace={"user": True}, added=[2])

    def test_checkpoint_rebuilds_canonical_and_historical_base_ids(self) -> None:
        register_gym_envs()
        env_id = "superdex_gym/Ant-v0"
        spec = gym.spec(env_id)
        persisted = {
            "include_contact_in_observation": False,
            "use_rotation_vector": True,
        }
        expected_env = object()

        with mock.patch.object(spec, "make", return_value=expected_env) as make:
            actual_env = make_checkpoint_env(env_id, persisted)

        self.assertIs(expected_env, actual_env)
        make.assert_called_once_with(
            include_contact_in_observation=False,
            use_rotation_vector=True,
        )
        self.assertEqual(
            {
                "include_contact_in_observation": False,
                "use_rotation_vector": True,
            },
            persisted,
        )

    def test_checkpoint_video_config_validates_id_and_isolates_trial_config(
        self,
    ) -> None:
        register_gym_envs()
        trial_config = {"nested": {"values": [1]}, "render_mode": "human"}

        _, prepared = prepare_checkpoint_env_config(
            "superdex_gym/CartPole-v0",
            trial_config,
            render_mode="rgb_array",
        )
        prepared["nested"]["values"].append(2)

        self.assertEqual("rgb_array", prepared["render_mode"])
        self.assertEqual(
            {"nested": {"values": [1]}, "render_mode": "human"},
            trial_config,
        )
        with self.assertRaisesRegex(ValueError, "not a registered canonical"):
            prepare_checkpoint_env_config(
                "superdex_gym/CartPole-v99",
                trial_config,
                render_mode="rgb_array",
            )

    def test_checkpoint_rejects_unsupported_legacy_identifiers(self) -> None:
        register_gym_envs()
        # The legacy mochi_gym namespace is no longer special-cased; it surfaces the
        # standard Gymnasium lookup error like any other unregistered namespace.
        with self.assertRaises(gym.error.Error):
            resolve_checkpoint_env_spec("mochi_gym/CartPole-v0")
        with self.assertRaisesRegex(ValueError, "not a registered canonical"):
            resolve_checkpoint_env_spec("superdex_gym/CartPole")
        with self.assertRaisesRegex(ValueError, "not a registered canonical"):
            resolve_checkpoint_env_spec("cart_pole")
        with self.assertRaisesRegex(ValueError, "not a registered canonical"):
            resolve_checkpoint_env_spec("superdex_gym/CartPole-v99")

    def test_checkpoint_preserves_ordinary_gymnasium_keyword_configuration(
        self,
    ) -> None:
        env_id = "external/Test-v0"
        spec = EnvSpec(env_id, "example.module:Environment")
        gym.registry[env_id] = spec
        config = {"nested": [1]}
        expected_env = object()

        with mock.patch.object(spec, "make", return_value=expected_env) as make:
            actual_env = make_checkpoint_env(env_id, config)

        self.assertIs(expected_env, actual_env)
        make.assert_called_once_with(nested=[1])
        self.assertEqual({"nested": [1]}, config)


if __name__ == "__main__":
    unittest.main()
