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
from unittest.mock import Mock, patch

import superdex.physics as sdp
from superdex.lab.gym.envs.scene_manager import (
    destroy_scene_with_cleanup,
    SceneCleanupError,
    SceneData,
    SceneManager,
)

########################################################################################


class TestSceneManager(unittest.TestCase):
    """Test cases for SceneManager class."""

    def setUp(self):
        SceneManager._instances.clear()

    def tearDown(self):
        SceneManager._instances.clear()

    def test_singleton_behavior(self):
        # Test that SceneManager follows singleton pattern per environment class.

        manager1 = SceneManager.get_instance("TestEnv1")
        manager2 = SceneManager.get_instance("TestEnv1")
        manager3 = SceneManager.get_instance("TestEnv2")
        assert manager1 == manager2
        assert manager1 != manager3

    def test_initial_state(self):
        # Test that a new SceneManager starts with empty state.

        manager = SceneManager.get_instance("TestEnv")
        assert manager.scene_count == 0
        assert manager.scene_info == {}
        assert "test_scene" not in manager

    @patch.object(sdp, "destroy_scene")
    @patch.object(sdp, "is_initialized", return_value=True)
    def test_register_and_release_scene(self, mock_is_initialized, mock_destroy_scene):
        # Test successful scene registration.

        mock_scene = Mock()
        mock_agent = Mock()
        mock_initial_state = Mock()
        mock_scene.capture_state.return_value = mock_initial_state

        # Register scene.
        manager = SceneManager.get_instance("TestEnv")
        scene_data = manager.register_scene("test_scene", mock_scene, mock_agent)

        # Verify scene was registered correctly.
        assert isinstance(scene_data, SceneData)
        assert scene_data.scene == mock_scene
        assert scene_data.agent == mock_agent
        assert scene_data.initial_state is mock_initial_state
        assert scene_data.ref_count == 1

        # Verify manager state.
        assert manager.scene_count == 1
        assert manager.scene_info == {"test_scene": 1}
        assert "test_scene" in manager

        # Release the scene, verify cleanup.
        manager.release_scene("test_scene")
        assert manager.scene_count == 0
        assert manager.scene_info == {}
        assert "test_scene" not in manager
        mock_is_initialized.assert_called_once_with()
        mock_destroy_scene.assert_called_once_with(mock_scene)

    def test_register_scene_duplicate_name(self):
        # Test that registering a scene with duplicate name raises ValueError.

        mock_scene1 = Mock()
        mock_scene2 = Mock()
        mock_agent1 = Mock()
        mock_agent2 = Mock()
        # Register first scene
        manager = SceneManager.get_instance("TestEnv")
        manager.register_scene("test_scene", mock_scene1, mock_agent1)

        # Try to register second scene with same name
        with self.assertRaises(ValueError):
            manager.register_scene("test_scene", mock_scene2, mock_agent2)

    def test_try_find_scene_success(self):
        # Test successful scene acquisition.

        mock_scene = Mock()
        mock_agent = Mock()
        # Register scene
        manager = SceneManager.get_instance("TestEnv")
        scene_data = manager.register_scene("test_scene", mock_scene, mock_agent)

        # Find scene
        found_scene_data = manager.try_find_scene("test_scene")

        # Verify acquisition
        assert found_scene_data is scene_data
        assert found_scene_data.ref_count == 2
        assert manager.scene_info == {"test_scene": 2}

    @patch.object(sdp, "destroy_scene")
    @patch.object(sdp, "is_initialized", return_value=True)
    def test_release_scene_decrement_ref_count(
        self, mock_is_initialized, mock_destroy_scene
    ):
        # Test that releasing scene decrements reference count.

        mock_scene = Mock()
        mock_agent = Mock()
        mock_initial_state = Mock()
        mock_scene.capture_state.return_value = mock_initial_state
        cleanup_order = []
        cleanup = Mock(side_effect=lambda: cleanup_order.append("cleanup"))
        mock_destroy_scene.side_effect = lambda scene: cleanup_order.append(
            "destroy_scene"
        )

        # Register and acquire scene multiple times.
        manager = SceneManager.get_instance("TestEnv")
        scene_data = manager.register_scene(
            "test_scene", mock_scene, mock_agent, cleanup_callbacks=[cleanup]
        )
        manager.try_find_scene("test_scene")
        manager.try_find_scene("test_scene")
        assert manager.scene_info["test_scene"] == 3

        # Release scene once.
        manager.release_scene("test_scene")
        assert manager.scene_info["test_scene"] == 2
        assert manager.scene_count == 1
        cleanup.assert_not_called()

        # Release scene again.
        manager.release_scene("test_scene")
        assert manager.scene_info["test_scene"] == 1
        assert manager.scene_count == 1
        cleanup.assert_not_called()

        # Release final reference.
        manager.release_scene("test_scene")
        assert manager.scene_count == 0
        assert manager.scene_info == {}
        mock_scene.release_state.assert_called_once_with(scene_data.initial_state)
        cleanup.assert_called_once_with()
        mock_is_initialized.assert_called_once_with()
        mock_destroy_scene.assert_called_once_with(mock_scene)
        assert cleanup_order == ["cleanup", "destroy_scene"]

    def test_release_scene_nonexistent(self):
        # Test that releasing non-existent scene raises an error.

        manager = SceneManager.get_instance("TestEnv")
        with self.assertRaises(ValueError):
            manager.release_scene("nonexistent_scene")

    def test_multiple_managers_independent(self):
        # Test that different environment managers are independent.

        mock_scene1 = Mock()
        mock_scene2 = Mock()
        mock_agent1 = Mock()
        mock_agent2 = Mock()
        # Register scenes in different managers.
        manager1 = SceneManager.get_instance("Env1")
        manager1.register_scene("scene1", mock_scene1, mock_agent1)
        manager2 = SceneManager.get_instance("Env2")
        manager2.register_scene("scene2", mock_scene2, mock_agent2)

        # Verify independence.
        assert manager1.scene_count == 1
        assert manager2.scene_count == 1
        assert "scene1" in manager1
        assert "scene1" not in manager2
        assert "scene2" in manager2
        assert "scene2" not in manager1


class TestDestroySceneWithCleanup(unittest.TestCase):
    """Tests for the scene-teardown ordering contract.

    Cleanup callbacks release objects that outlive the scene and keep referencing its
    actors (e.g. bots owned by the process-global ``RoboticsContext``), so the scene may only
    be destroyed once every callback has succeeded.
    """

    @patch.object(sdp, "destroy_scene")
    def test_cleanup_failure_keeps_scene_alive(self, mock_destroy_scene):
        mock_scene = Mock()
        cleanup_error = RuntimeError("bot destruction failed")
        failing_cleanup = Mock(side_effect=cleanup_error)

        with self.assertRaises(SceneCleanupError) as context:
            destroy_scene_with_cleanup(mock_scene, None, [failing_cleanup])

        # The scene must survive: freeing it under a still-live owner is unrecoverable.
        mock_destroy_scene.assert_not_called()
        assert context.exception.__cause__ is cleanup_error

    @patch.object(sdp, "destroy_scene")
    def test_every_cleanup_runs_before_the_failure_is_reported(
        self, mock_destroy_scene
    ):
        mock_scene = Mock()
        first_error = RuntimeError("first failure")
        failing_cleanup = Mock(side_effect=first_error)
        later_cleanup = Mock()

        # Callbacks run in reverse registration order, so `failing_cleanup` runs last.
        with self.assertRaises(SceneCleanupError) as context:
            destroy_scene_with_cleanup(
                mock_scene, None, [failing_cleanup, later_cleanup]
            )

        later_cleanup.assert_called_once_with()
        failing_cleanup.assert_called_once_with()
        mock_destroy_scene.assert_not_called()
        assert context.exception.__cause__ is first_error

    @patch.object(sdp, "destroy_scene")
    def test_release_state_failure_still_destroys_scene(self, mock_destroy_scene):
        # A release_state failure does not imply an outside owner, so it must not block
        # destruction -- but it is still reported.
        mock_scene = Mock()
        release_error = RuntimeError("release_state failed")
        mock_scene.release_state.side_effect = release_error
        cleanup = Mock()

        with self.assertRaises(RuntimeError) as context:
            destroy_scene_with_cleanup(mock_scene, Mock(), [cleanup])

        cleanup.assert_called_once_with()
        mock_destroy_scene.assert_called_once_with(mock_scene)
        assert context.exception is release_error


########################################################################################

if __name__ == "__main__":
    unittest.main()
