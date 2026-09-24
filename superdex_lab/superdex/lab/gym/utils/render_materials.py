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

"""Material colors for the default (Polyscope) viewer.

The viewer draws a registered ``.glb`` render model's geometry but not its materials,
so every link gets an arbitrary color. :func:`apply_render_model_colors` reads each
model's base colors (per material, or the mean of a color texture) and paints them onto
the viewer's mesh through a small palette texture, so a white-and-grey FR3 looks like
one. Nothing here affects physics.
"""

from __future__ import annotations

import functools

import numpy as np
import numpy.typing as npt
import superdex.physics as physics
import trimesh
from superdex.physics.utils import render_model_registry

_FALLBACK_RGB = (0.8, 0.8, 0.8)


@functools.lru_cache(maxsize=256)
def glb_vertex_colors(glb_path: str) -> npt.NDArray[np.float32]:
    """Per-vertex RGB in [0, 1] for a ``.glb``, in the vertex order the viewer uses
    (``trimesh.load(path, force="mesh")``, which concatenates the scene's meshes)."""
    loaded = trimesh.load(glb_path)
    meshes = loaded.dump() if isinstance(loaded, trimesh.Scene) else [loaded]
    colors = [
        np.broadcast_to(_mesh_color(m), (len(m.vertices), 3))
        if not _has_vertex_colors(m)
        else np.asarray(m.visual.vertex_colors)[:, :3] / 255.0
        for m in meshes
    ]
    return np.concatenate(colors).astype(np.float32)


def _has_vertex_colors(mesh: trimesh.Trimesh) -> bool:
    visual = mesh.visual
    return isinstance(visual, trimesh.visual.ColorVisuals) and visual.kind == "vertex"


def _mesh_color(mesh: trimesh.Trimesh) -> npt.NDArray[np.float64]:
    material = getattr(mesh.visual, "material", None)
    if material is None:
        return np.asarray(_FALLBACK_RGB)
    if not hasattr(material, "baseColorFactor"):  # non-PBR material
        return np.asarray(material.main_color[:3]) / 255.0
    factor = material.baseColorFactor  # trimesh stores it as uint8 RGBA
    # glTF: a missing base color factor means white.
    rgb = np.ones(3) if factor is None else np.asarray(factor[:3]) / 255.0
    texture = material.baseColorTexture
    if texture is not None:
        pixels = np.asarray(texture.convert("RGB"), dtype=np.float64).reshape(-1, 3)
        rgb = rgb * pixels.mean(axis=0) / 255.0
    return rgb


def paint_mesh(renderer, vertex_colors: npt.ArrayLike) -> bool:
    """Color a viewer mesh per vertex through a palette texture. Returns False (and
    leaves the mesh alone) if the color count does not match its vertices."""
    colors = np.asarray(vertex_colors, dtype=np.float32)
    if colors.shape != (len(renderer.get_local_coordinates()), 3):
        return False
    palette, index = np.unique(np.round(colors, 3), axis=0, return_inverse=True)
    if len(palette) == 1:
        renderer.set_front_face_color(palette[0])
        return True
    # One texel per palette color; each vertex samples the middle of its texel.
    uv = np.column_stack(
        [(index.ravel() + 0.5) / len(palette), np.full(len(index), 0.5)]
    )
    renderer.set_texture_coordinates(uv)
    renderer.set_texture(palette[None, :, :])
    renderer.set_texture_filter(type(renderer).TextureFilter.NEAREST)
    return True


def apply_render_model_colors(
    viewer, scene: physics.Scene, colors: dict[str, tuple[float, float, float]] = None
) -> None:
    """Paint every actor that has a registered ``.glb`` with that model's colors, and
    actors named in ``colors`` (e.g. ``{"table": (0.55, 0.4, 0.26)}``) with a flat color."""
    colors = colors or {}
    for actor in viewer.get_actors():
        renderer = viewer.get_actor_renderer(actor)
        if renderer is None or not hasattr(renderer, "set_front_face_color"):
            continue
        name = actor.get_name()
        if name in colors:
            renderer.set_front_face_color(colors[name])
            continue
        entry = render_model_registry.get(scene.get_handle(), actor.get_handle())
        if entry is not None:
            paint_mesh(renderer, glb_vertex_colors(entry.glb_path))
