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

"""
A collection of helper functions for SuperDex Physics scene manipulation.
"""

from __future__ import annotations

import fnmatch
import pathlib
from typing import TYPE_CHECKING

import superdex.physics as sdp
from superdex.physics.paths import get_assets_root

if TYPE_CHECKING:
    import superdex.robotics as sdr  # @manual


def _strip_win_extended_prefix(path_str: str) -> str:
    r"""Drop a Windows ``\\?\`` extended-length prefix.

    ``pathlib.Path.resolve()`` returns a ``\\?\``-prefixed path on Windows/EdenFS.
    Windows does NOT apply ``/``->``\`` normalization inside a ``\\?\`` path, so
    when the native prefab loader joins a ``.mochi_scene`` manifest's forward-slash
    relative link paths (e.g. ``generated_assets/foo.mochi.h5``) onto such a root,
    the result is an unopenable mixed-separator path. Stripping the prefix restores
    normal separator handling. No-op when the prefix is absent (non-Windows, or
    already-plain paths). Caveat: a genuinely >260-char path then relies on OS
    long-path support.
    """
    return path_str[4:] if path_str.startswith("\\\\?\\") else path_str


def find_actor(scene: sdp.Scene, pattern: str) -> sdp.Actor:
    """Returns an actor whose name matches the given pattern. The pattern is specified
    as a case-sensitive string that may contain shell-style wildcards, such as * and ?.
    For example, "my_actor*" will match any actor whose name starts with "my_actor".
    If no actor matches the pattern, or there are multiple candidate actors, an error is
    raised."""

    if scene is None:
        raise ValueError("There is no scene.")

    found_actor: sdp.Actor | None = None

    def check_if_matching_actor(actor: sdp.Actor) -> None:
        nonlocal found_actor
        if fnmatch.fnmatchcase(actor.get_name(), pattern):
            if found_actor is not None:
                raise ValueError(
                    f'Scene contains multiple actors matching pattern "{pattern}".'
                )
            found_actor = actor

    scene.for_each_actor(check_if_matching_actor)
    if found_actor is None:
        raise ValueError(f'No actor found matching pattern "{pattern}".')
    return found_actor


def create_ground_plane(
    scene: sdp.Scene,
    normal: sdp.Real3 | None = None,
    offset: float = 0.0,
    contact_params: sdp.ContactParams | None = None,
) -> sdp.Actor:
    """Creates a ground plane actor in the scene. By default, the plane is initialized
    pointing at +Y, and an offset of 0. If you want a different plane, you can specify
    the normal and offset. Unless you specify contact parameters, the default values
    will be used.
    """

    # Initialize the plane shape.
    normal = normal or sdp.Real3(0.0, 1.0, 0.0)
    shape = sdp.create_plane_shape(normal, offset)

    # Create the ground plane actor.
    actor = scene.create_rigid_actor(
        is_static=True,
        layer="Environment",
        name="StaticPlane",
        shape=shape,
        collider_type=sdp.ColliderType.PLANE,
        contact=contact_params if contact_params is not None else sdp.ContactParams(),
    )
    if actor is None:
        raise RuntimeError("Failed to create ground plane actor")
    return actor


def load_bot_scene(
    path: str,
    bots_ctx: sdr.RoboticsContext,
) -> sdr.BotScene:
    """Load a .mochi_bot_scene file and instantiate all scene objects.

    Creates the physics scene from the base_scene, instantiates all bots with
    placement and initial pose, and creates controllers via the provided
    RoboticsContext.

    Args:
        path: File path to the .mochi_bot_scene file.
        bots_ctx: RoboticsContext for controller lifetime management. The caller
            owns this — create before loading, destroy after all scenes are
            closed.

    Returns:
        The loaded BotScene, which owns the physics scene and provides named
        lookup for bots and spawnable prefabs.
    """
    if not sdp.is_initialized():
        raise RuntimeError(
            "SuperDex Physics is not initialized. Call superdex.physics.initialize() first."
        )
    import superdex.robotics as sdr  # @manual

    return sdr.load_bot_scene(path, bots_ctx)


def load_bot_scene_prefab(path: str):  # noqa: ANN201
    """Parse a .mochi_bot_scene file into a BotScenePrefab (no scene instantiation).

    Unlike load_bot_scene, this only deserializes and validates the file; it does
    not require SuperDex Physics to be initialized and creates no physics scene. Useful for
    reading scene metadata such as spawnable prefab names and their
    unreal_blueprint_name.

    Args:
        path: File path to the .mochi_bot_scene file.

    Returns:
        The parsed BotScenePrefab.
    """
    import superdex.robotics as sdr  # @manual

    return sdr.load_bot_scene_prefab_from_file(path)


def load_bot_task_prefab(path: str):  # noqa: ANN201
    """Parse a .mochi_bot_task file into a BotTaskPrefab.

    Deserializes and validates the task file (spawn names non-empty/unique,
    prefab names non-empty). Does not require SuperDex Physics to be initialized.

    Args:
        path: File path to the .mochi_bot_task file.

    Returns:
        The parsed BotTaskPrefab, whose `spawns` declare the named object
        instances (name, prefab_name, translation, rotation).
    """
    import superdex.robotics as sdr  # @manual

    return sdr.load_bot_task_prefab_from_file(path)


def create_scene_from_prefab(
    prefab: str | sdp.prefab.ScenePrefab,
    root_dir: str | pathlib.Path | None = None,
    params: sdp.prefab.PrefabParams | None = None,
    scene_name: str = "",
) -> sdp.Scene:
    """
    Create a SuperDex Physics scene and initialize it using a ScenePrefab object or
    prefab file path.

    Args:
        prefab: Either a file path to a prefab file (e.g., "samples/my_scene.mochi_scene")
            or a ScenePrefab object. Referenced model files and nested prefabs will be loaded
            as necessary. Relative file paths will be resolved using the specified root_dir.
        root_dir: The root directory for resolving relative paths. ``None`` (default) or an
            empty string selects the assets directory (SUPERDEX_ASSETS_PATH environment
            variable). A ``pathlib.Path`` is always taken literally, including
            ``Path("")``, which pathlib normalizes to ``Path(".")``.
            Used for resolving the prefab file path (when prefab is a str) and for
            loading any nested prefabs and shape files referenced within the prefab.
        params: Optional PrefabParams to customize prefab loading. If None, default
            parameters are used.
        scene_name: Optional name for the created scene. If empty, the scene will
            have no name.

    Returns:
        The created SuperDex Physics scene with the prefab loaded into it.

    Example:
        .. code-block:: python

            import superdex.physics as mochi
            from superdex.physics.utils.scene_helpers import create_scene_from_prefab

            # Example 1: Load prefab from file with relative path
            scene = create_scene_from_prefab("samples/my_scene.mochi_scene")

            # Example 2: Load, modify, then create scene
            p = mochi.prefab.shallow_load_from_file("samples/my_scene.mochi_scene")
            p.scene.gravity = [0.0, -5.0, 0.0]
            scene = create_scene_from_prefab(p, root_dir="/custom/path/to/assets")
    """
    if params is None:
        params = sdp.prefab.PrefabParams()

    # Decided on the argument as given: `pathlib.Path` defines no `__bool__`, so every Path
    # is truthy and a truthiness test would send `Path("")` -- normalized to `Path(".")` --
    # to the CWD rather than the assets root.
    if root_dir is None or root_dir == "":
        root_path = get_assets_root()
    else:
        root_path = pathlib.Path(root_dir)

    # Initialize an empty scene.
    scene = sdp.create_scene(scene_name)
    if scene is None:
        raise RuntimeError("Failed to create scene")

    # Load the prefab.
    if isinstance(prefab, str):
        # Interpret the file path as an assets-directory-relative path unless
        # it is formatted like an absolute path.
        path = pathlib.Path(prefab)
        if not path.is_absolute():
            path = root_path / path
        loaded_prefab = sdp.prefab.load_from_file(
            _strip_win_extended_prefix(str(path)),
            _strip_win_extended_prefix(str(root_path)),
        )
    elif isinstance(prefab, sdp.prefab.ScenePrefab):
        # Ensure that all referenced prefabs and model files have been loaded,
        # in case the caller created the ScenePrefab object procedurally or loaded
        # it via shallow_load_from_file.
        sdp.prefab.ensure_fully_loaded(
            prefab=prefab, root_path=_strip_win_extended_prefix(str(root_path))
        )
        loaded_prefab = prefab
    else:
        raise TypeError(
            f"prefab must be a str (file path) or mochi.prefab.ScenePrefab, got {type(prefab).__name__}"
        )

    # Add prefab to scene.
    sdp.prefab.add_to_scene(prefab=loaded_prefab, scene=scene, params=params)

    return scene
