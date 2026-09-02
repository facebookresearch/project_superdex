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

"""Auto-generated scene-sharing and batched-stepping tests for every discovered env.

These validate base ``MochiEnv`` behavior across every environment discovered on the
filesystem, so environments absent from a build are simply not covered -- no
per-environment gating is required:

- ``test_scene_sharing_<env>``: the scene manager shares one scene between instances of
  the same env class when ``use_shared_scenes`` is set, and gives each instance its own
  scene otherwise. Applies to every env.
- ``test_batched_stepping_<env>``: stepping two shared-scene envs in alternating batches
  reproduces a single independent reference env exactly (deterministic shared-scene state
  capture/restore). This compares the ``agent_pose`` observation, so envs whose
  observation has no ``agent_pose`` are skipped.
"""

import unittest
from typing import Callable

import superdex.physics as sdp
from numpy.testing import assert_array_equal
from superdex.lab.gym.utils.env_discovery import discover_envs, EnvEntry

########################################################################################

_NUM_STEPS_TOTAL = 100  # Total number of steps to perform.
_NUM_STEPS_BATCH = 5  # Number of steps to perform in each batch.
_SEED = 42  # Random seed for reproducibility.

# Observation key compared by the batched-stepping test; envs without it are skipped.
_POSE_KEY = "agent_pose"


class TestSceneSharing(unittest.TestCase):
    """Scene-sharing and batched-stepping tests, generated per discovered env."""

    def tearDown(self) -> None:
        # Force the Mochi context to be shut down after each test.
        if sdp.is_initialized():
            sdp.shutdown()


def _make_scene_sharing_test(entry: EnvEntry) -> Callable[[TestSceneSharing], None]:
    def test(self: TestSceneSharing) -> None:
        # Shared scenes: both envs share the same scene via the scene manager.
        shared_cfg = entry.cfg_cls(**{**entry.cfg_kwargs, "use_shared_scenes": True})
        with entry.env_cls(shared_cfg) as env_1, entry.env_cls(shared_cfg) as env_2:
            self.assertIsNotNone(env_1._scene_manager)
            self.assertIsNotNone(env_2._scene_manager)
            self.assertIs(env_1._scene, env_2._scene)

        # Independent scenes: no scene manager, distinct scenes.
        indep_cfg = entry.cfg_cls(**{**entry.cfg_kwargs, "use_shared_scenes": False})
        with entry.env_cls(indep_cfg) as env_1, entry.env_cls(indep_cfg) as env_2:
            self.assertIsNone(env_1._scene_manager)
            self.assertIsNone(env_2._scene_manager)
            self.assertIsNot(env_1._scene, env_2._scene)

    test.__doc__ = f"Scene sharing for {entry.env_id}."
    return test


def _make_batched_stepping_test(entry: EnvEntry) -> Callable[[TestSceneSharing], None]:
    def test(self: TestSceneSharing) -> None:
        # Reference: a single independent env yields the ground-truth trajectory.
        ref_cfg = entry.cfg_cls(
            **{**entry.cfg_kwargs, "num_worker_threads": 0, "use_shared_scenes": False}
        )
        with entry.env_cls(ref_cfg) as ref_env:
            ref_env.reset(seed=_SEED)

            # The batched-stepping check compares the agent pose; skip envs without it.
            if _POSE_KEY not in ref_env.get_last_step().observation:
                self.skipTest(f"{entry.env_id} has no '{_POSE_KEY}' observation.")

            ref_env.action_space.seed(_SEED)
            ref_actions = []
            for _ in range(_NUM_STEPS_TOTAL):
                action = ref_env.action_space.sample()
                ref_actions.append(action)
                ref_env.step(action)
            ref_final_pose = ref_env.get_last_step().observation[_POSE_KEY]

        # Two shared-scene envs stepped in alternating batches must reproduce the
        # reference exactly, and end in an identical physics state.
        shared_cfg = entry.cfg_cls(
            **{**entry.cfg_kwargs, "num_worker_threads": 0, "use_shared_scenes": True}
        )
        with entry.env_cls(shared_cfg) as env_1, entry.env_cls(shared_cfg) as env_2:
            env_1.reset(seed=_SEED)
            env_2.reset(seed=_SEED)
            for i in range(0, _NUM_STEPS_TOTAL, _NUM_STEPS_BATCH):
                for action in ref_actions[i : i + _NUM_STEPS_BATCH]:
                    env_1.step(action)
                for action in ref_actions[i : i + _NUM_STEPS_BATCH]:
                    env_2.step(action)

            final_pose_1 = env_1.get_last_step().observation[_POSE_KEY]
            final_pose_2 = env_2.get_last_step().observation[_POSE_KEY]
            assert_array_equal(
                final_pose_1,
                ref_final_pose,
                err_msg=f"{entry.env_id}: env_1 batched pose diverged from reference.",
            )
            assert_array_equal(
                final_pose_2,
                ref_final_pose,
                err_msg=f"{entry.env_id}: env_2 batched pose diverged from reference.",
            )

            # env_1 and env_2 share the same physics.Scene, so their states are comparable.
            self.assertTrue(
                env_1._scene.is_equal_state(
                    env_1._state_snapshot, env_2._state_snapshot
                ),
                msg=f"{entry.env_id}: env_1 and env_2 state snapshots diverged.",
            )

    test.__doc__ = f"Shared-scene batched stepping for {entry.env_id}."
    return test


def _make_survives_creator_close_test(
    entry: EnvEntry,
) -> Callable[[TestSceneSharing], None]:
    def test(self: TestSceneSharing) -> None:
        # Closing the env that built the shared scene must not tear down resources the
        # surviving sibling still needs: scene-owned resources (e.g. a bot's articulated
        # actor) are released via SceneManager cleanup callbacks only once the reference
        # count reaches zero. A multi-item `with` unwinds in reverse, so the creator is
        # always closed last there -- this closes it first on purpose.
        shared_cfg = entry.cfg_cls(**{**entry.cfg_kwargs, "use_shared_scenes": True})
        creator = entry.env_cls(shared_cfg)
        sibling = entry.env_cls(shared_cfg)
        try:
            self.assertIs(creator._scene, sibling._scene)
            creator.close()
            sibling.reset()
            sibling.step(sibling.action_space.sample())
        finally:
            creator.close()
            sibling.close()

    test.__doc__ = f"Shared scene survives creator close for {entry.env_id}."
    return test


def _register_generated_tests() -> None:
    # Base envs only, deliberately: these exercise base `MochiEnv` behavior, which a
    # config variant does not change, and each entry costs three physics-instantiating
    # tests. Running them per variant would multiply this slow suite for no extra signal.
    for entry in discover_envs():
        if entry.variant is not None:
            continue
        setattr(
            TestSceneSharing,
            f"test_scene_sharing_{entry.short_name}",
            _make_scene_sharing_test(entry),
        )
        setattr(
            TestSceneSharing,
            f"test_batched_stepping_{entry.short_name}",
            _make_batched_stepping_test(entry),
        )
        setattr(
            TestSceneSharing,
            f"test_shared_scene_survives_creator_close_{entry.short_name}",
            _make_survives_creator_close_test(entry),
        )


_register_generated_tests()

########################################################################################

if __name__ == "__main__":
    unittest.main()
