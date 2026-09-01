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
A neutral registry mapping scene actors to external visual render models (.glb).

This module is the bridge between code that *creates* actors backed by a visual mesh
(e.g. the superdex bot-creation helper) and the default viewer that *renders* them.
The write side registers an actor's render model; the read side (the viewer) looks it
up and swaps in a GLB renderer instead of drawing the physics surface mesh.

It intentionally imports nothing beyond the standard library (superdex.physics is
referenced only for type annotations, which are deferred via ``from __future__ import
annotations``), so that populating the registry during headless training pulls in no
rendering dependencies (polyscope, trimesh).
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    import superdex.physics

########################################################################################


@dataclasses.dataclass(frozen=True)
class RenderModelEntry:
    """Visual render-model override for a single actor."""

    glb_path: str
    """Absolute path to the ``.glb`` visual mesh."""
    local_transform: superdex.physics.TransformRT
    """Transform from the render-mesh frame to the actor's root frame."""
    scale: superdex.physics.Real3
    """Per-axis scale to bake into the render-mesh vertices."""


# Keyed by scene handle value, then by actor handle value. Using handle values (plain
# integers) rather than the handle objects keeps lookups cheap and avoids relying on
# handle object identity/hashing.
_REGISTRY: dict[int, dict[int, RenderModelEntry]] = {}

########################################################################################


def register(
    scene: superdex.physics.Scene,
    actor_handle: superdex.physics.ActorHandle,
    glb_path: str,
    local_transform: superdex.physics.TransformRT,
    scale: superdex.physics.Real3,
) -> None:
    """Registers a visual render model for the given actor in the given scene."""
    scene_key = scene.get_handle().value
    _REGISTRY.setdefault(scene_key, {})[actor_handle.value] = RenderModelEntry(
        glb_path=glb_path,
        local_transform=local_transform,
        scale=scale,
    )


def get(
    scene_handle: superdex.physics.SceneHandle,
    actor_handle: superdex.physics.ActorHandle,
) -> RenderModelEntry | None:
    """Returns the render model registered for the actor, or None if there is none.

    Takes handle objects rather than their ``.value`` integers so that the read side
    matches the write side and no caller has to unwrap handles itself.
    """
    scene_entries = _REGISTRY.get(scene_handle.value)
    if scene_entries is None:
        return None
    return scene_entries.get(actor_handle.value)


def unregister_actors(
    scene: superdex.physics.Scene, actor_handles: Iterable[superdex.physics.ActorHandle]
) -> None:
    """Removes the render models registered for the given actors in the given scene."""
    scene_key = scene.get_handle().value
    scene_entries = _REGISTRY.get(scene_key)
    if scene_entries is None:
        return
    for actor_handle in actor_handles:
        scene_entries.pop(actor_handle.value, None)
    if not scene_entries:
        del _REGISTRY[scene_key]


def clear_scene(scene: superdex.physics.Scene) -> None:
    """Removes all render models registered for the given scene."""
    _REGISTRY.pop(scene.get_handle().value, None)
