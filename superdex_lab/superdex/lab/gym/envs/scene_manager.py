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

"""Scene management utilities for SuperDex Gym environments.

This module provides a singleton SceneManager that enables multiple environment
instances to share the same Mochi scene when they have identical configurations,
significantly reducing memory usage through reference counting and automatic cleanup.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import superdex.physics as physics
from superdex.physics.utils import render_model_registry

########################################################################################


class SceneCleanupError(RuntimeError):
    """Raised when scene-bound cleanup failed and the scene was therefore left alive.

    Seeing this means an object that outlives the scene (e.g. a bot owned by the
    process-global ``RoboticsContext``) could not be released and still references the
    scene's actors. Callers unwinding their own error path must let it propagate
    untouched and must **not** destroy the scene themselves.
    """


def destroy_scene_with_cleanup(
    scene: physics.Scene,
    initial_state: physics.StateHandle | None,
    cleanup_callbacks: list[Callable[[], None]],
) -> None:
    """Releases all scene-owned resources, then destroys the scene.

    If a cleanup callback fails the scene is deliberately *not* destroyed and
    :class:`SceneCleanupError` is raised instead: those callbacks release objects that
    outlive this scope and keep referencing the scene's actors (e.g. bots owned by the
    process-global ``RoboticsContext``). Leaking the scene is recoverable; destroying it out
    from under a live owner is not.

    A ``release_state`` failure does not imply an outside owner, so it is reported but
    does not block destruction.
    """
    first_error = None
    if initial_state is not None:
        try:
            scene.release_state(initial_state)
        except Exception as error:
            first_error = error
    cleanup_error = None
    for cleanup in reversed(cleanup_callbacks):
        try:
            cleanup()
        except Exception as error:
            cleanup_error = cleanup_error or error
    if cleanup_error is not None:
        raise SceneCleanupError(
            "Scene-bound cleanup failed; the scene was intentionally left alive because "
            "an owner that outlives it may still reference its actors."
        ) from cleanup_error
    # Drop any visual render models still keyed against this scene. Handle values are
    # recycled after destruction, so a stale entry would otherwise be inherited by a
    # future scene and render an unrelated GLB.
    render_model_registry.clear_scene(scene)
    try:
        physics.destroy_scene(scene)
    except Exception as error:
        first_error = first_error or error
    if first_error is not None:
        raise first_error


@dataclasses.dataclass
class SceneData:
    """Container for shared scene data."""

    scene: physics.Scene
    """Pointer to the shared Mochi scene."""
    initial_state: physics.StateHandle
    """Initial state of the scene upon initialization."""
    agent: physics.Actor | None
    """Pointer to the agent actor in the scene."""
    ref_count: int
    """Reference count for this scene. Used to manage cleanup."""
    cleanup_callbacks: list[Callable[[], None]]
    """Callbacks that release resources owned by the scene before its destruction."""


class SceneManager:
    """
    Global singleton manager for shared Mochi scenes. This manager allows multiple
    environment instances to share the same scene when they have identical
    configurations, reducing memory usage significantly.
    """

    ####################################################################################
    # Member variables
    ####################################################################################

    # Static members
    _instances: dict[str, SceneManager] = {}

    # Private members
    _scene: dict[str, SceneData]

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(self):
        """Private constructor. Use get_instance() to obtain the singleton."""
        self._scenes = {}

    ####################################################################################
    # Methods
    ####################################################################################

    @classmethod
    def get_instance(cls, env_class_name: str) -> SceneManager:
        """Get the singleton SceneManager instance for the given environment class."""
        instance = cls._instances.get(env_class_name, None)
        if instance is None:
            cls._instances[env_class_name] = instance = cls()
        return instance

    def register_scene(
        self,
        scene_name: str,
        scene: physics.Scene,
        agent: physics.Actor,
        cleanup_callbacks: list[Callable[[], None]] | None = None,
    ) -> SceneData:
        """
        Register a new scene with the manager for shared use across environment
        instances.

        When you call this method, the scene manager takes full ownership of both the
        scene and agent you provide. This means you should not destroy them yourself -
        the manager will handle their lifecycle automatically. The scene becomes
        available for sharing among multiple environment instances, which dramatically
        reduces memory usage when stepping many sequential environments with identical
        configuration.

        Upon registration, the manager immediately captures a snapshot of the scene's
        initial state. This snapshot can be used to reset environment scenes back to
        their starting configuration.

        The method returns a SceneData container that holds the scene, agent, initial
        state, and a reference count.

        If you try to register a scene with a name that already exists, the method will
        raise a ValueError to prevent accidental overwrites. Other environments can then
        access this shared scene using try_find_scene().

        When environments are done with the scene, they should call release_scene() to
        decrement the reference count. Once the reference count reaches zero, the scene
        will be automatically cleaned up.
        """

        if scene_name in self._scenes:
            raise ValueError(f"Scene with name {scene_name} already exists.")

        # Take initial state snapshot
        initial_state = scene.capture_state()

        # Store newly registered scene.
        scene_data = SceneData(
            scene=scene,
            initial_state=initial_state,
            agent=agent,
            ref_count=1,
            cleanup_callbacks=list(cleanup_callbacks or ()),
        )
        self._scenes[scene_name] = scene_data
        return scene_data

    def try_find_scene(self, scene_name: str) -> SceneData | None:
        """Get an existing shared scene matching the given name. If it does not exist,
        then this function will return None. You must call release_scene() when you are
        done using the scene."""

        scene_data = self._scenes.get(scene_name, None)
        if scene_data is not None:
            scene_data.ref_count += 1
        return scene_data

    def release_scene(self, scene_name: str):
        """Decrease reference count and cleanup if no longer used."""

        scene_data = self._scenes.get(scene_name, None)
        if scene_data is None:
            raise ValueError(
                f"Could not find scene with name {scene_name}. Either it was not "
                "registered with `register_scene`, or it was already released."
            )

        scene_data.ref_count -= 1
        if scene_data.ref_count <= 0:
            assert physics.is_initialized()
            del self._scenes[scene_name]
            cleanup_callbacks = scene_data.cleanup_callbacks
            scene_data.cleanup_callbacks = []
            destroy_scene_with_cleanup(
                scene_data.scene, scene_data.initial_state, cleanup_callbacks
            )

    @property
    def scene_count(self) -> int:
        """Get the number of currently managed scenes."""
        return len(self._scenes)

    @property
    def scene_info(self) -> dict[str, int]:
        """Get information about all managed scenes and their reference counts."""
        return {key: data.ref_count for key, data in self._scenes.items()}

    def __contains__(self, scene_name: str) -> bool:
        """Check if the manager contains a scene with the given name."""
        return scene_name in self._scenes
