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

"""Visual mesh loading for textured Rerun visualization.

Loads COLLADA (.dae), glTF (.glb), and STL mesh files and extracts
per-material vertex colors for use with rr.Mesh3D. Adapted from
Dalton Moore's exo visualization work (D103430444).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Default fallback color when no material is found (light gray).
_DEFAULT_COLOR = np.array([180, 180, 180, 255], dtype=np.uint8)


def _ensure_collada_imports() -> None:
    """Force-import pycollada submodules that Meta's lazy-import breaks.

    Without these explicit imports, trimesh's isinstance checks against
    collada.polylist.Polylist etc. fail because the submodules haven't
    been loaded as attributes on the collada package.
    """
    try:
        import collada.lineset  # noqa: F401
        import collada.polylist  # noqa: F401
        import collada.triangleset  # noqa: F401
    except ImportError:
        pass


def _color_from_array(raw: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """Normalize a color array (float [0,1] or int [0,255]) to uint8."""
    arr = raw.ravel()
    if arr.max() <= 1.0:
        arr = (arr * 255).clip(0, 255)
    return arr.astype(np.uint8)


def _extract_base_color_factor(visual: object) -> npt.NDArray[np.uint8] | None:
    """Try to extract baseColorFactor from a trimesh visual's material."""
    try:
        mat = visual.material  # type: ignore[union-attr]
        if hasattr(mat, "baseColorFactor") and mat.baseColorFactor is not None:
            return _color_from_array(np.asarray(mat.baseColorFactor, dtype=np.float64))
    except Exception:
        logger.debug("Could not extract baseColorFactor", exc_info=True)
    return None


def _extract_main_color(visual: object) -> npt.NDArray[np.uint8] | None:
    """Try to extract main_color from a trimesh visual."""
    try:
        return _color_from_array(
            np.asarray(visual.main_color, dtype=np.float64)  # type: ignore[union-attr]
        )
    except Exception:
        logger.debug("Could not extract main_color", exc_info=True)
    return None


def _material_rgba(
    mesh: "trimesh.Trimesh",  # noqa: F821
    alpha: int = 255,
) -> npt.NDArray[np.uint8]:
    """Extract a single RGBA color from a trimesh mesh's PBR material.

    Checks baseColorFactor first (where URDF-exported COLLADA files
    encode part colors), then falls back to main_color, then to a
    default gray.

    Returns:
        (4,) uint8 array with RGBA values in [0, 255].
    """
    visual = mesh.visual
    rgba = None

    if visual is not None:
        rgba = _extract_base_color_factor(visual)
        if rgba is None:
            rgba = _extract_main_color(visual)

    if rgba is None:
        rgba = _DEFAULT_COLOR.copy()

    if len(rgba) == 3:
        rgba = np.append(rgba, alpha)
    else:
        rgba[3] = alpha

    return rgba.astype(np.uint8)


def _resolve_trimesh_list(
    file_path: Path,
) -> list | None:
    """Load a mesh file and return a list of trimesh.Trimesh objects, or None."""
    import trimesh

    _ensure_collada_imports()

    if not file_path.exists():
        return None

    try:
        loaded = trimesh.load(str(file_path))
    except Exception:
        logger.warning("Failed to load mesh: %s", file_path, exc_info=True)
        return None

    if isinstance(loaded, trimesh.Trimesh):
        return [loaded]
    if isinstance(loaded, trimesh.Scene):
        # dump() bakes visual_scene node transforms into vertex coordinates.
        # Critical for FR3 DAEs which use a 0.001 scale matrix.
        try:
            return list(loaded.dump())
        except Exception:
            logger.warning(
                "trimesh Scene.dump() failed for %s; falling back to raw "
                "geometry, vertex coordinates may be at the wrong scale",
                file_path,
                exc_info=True,
            )
            return list(loaded.geometry.values())

    logger.warning("Unexpected trimesh type %s for %s", type(loaded), file_path)
    return None


def _concatenate_submeshes(
    meshes: list,
) -> dict[str, npt.NDArray] | None:
    """Concatenate trimesh sub-meshes into a single vertex/face/normal/color dict."""
    import trimesh

    all_verts = []
    all_faces = []
    all_normals = []
    all_colors = []
    vert_offset = 0

    for mesh in meshes:
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue

        v = np.asarray(mesh.vertices, dtype=np.float32)
        f = np.asarray(mesh.faces, dtype=np.int32) + vert_offset
        n = np.asarray(mesh.vertex_normals, dtype=np.float32)
        rgba = _material_rgba(mesh)
        c = np.tile(rgba, (len(v), 1))

        all_verts.append(v)
        all_faces.append(f)
        all_normals.append(n)
        all_colors.append(c)
        vert_offset += len(v)

    if not all_verts:
        return None

    return {
        "vertices": np.concatenate(all_verts, axis=0),
        "faces": np.concatenate(all_faces, axis=0),
        "vertex_normals": np.concatenate(all_normals, axis=0),
        "vertex_colors": np.concatenate(all_colors, axis=0),
    }


def load_textured_mesh(
    file_path: str | Path,
) -> dict[str, npt.NDArray] | None:
    """Load a mesh file and extract per-material vertex colors.

    Handles multi-material COLLADA/glTF files by concatenating sub-meshes
    and tiling each sub-mesh's baseColorFactor across its vertices.

    Args:
        file_path: Path to a .dae, .glb, .stl, or .obj mesh file.

    Returns:
        Dict with keys 'vertices', 'faces', 'vertex_normals', 'vertex_colors'
        (all numpy arrays ready for rr.Mesh3D), or None on failure.
    """
    meshes = _resolve_trimesh_list(Path(file_path))
    if meshes is None or not meshes:
        return None
    return _concatenate_submeshes(meshes)


# Patterns to extract the bone name from a full Mochi prefab actor name.
# Format 1 (slash-separated): "...___Articulation/fr3_link0" -> "fr3_link0"
# Format 2 (underscore-separated): "...___articulation_fr3_link0" -> "fr3_link0"
_BONE_PATTERNS = [
    re.compile(r"[Aa]rticulation/([^/]+)$"),
    re.compile(r"___articulation_(.+)$", re.IGNORECASE),
]


def extract_bone_name(actor_name: str) -> str | None:
    """Extract the bone name from a Mochi articulated actor name.

    Args:
        actor_name: Full Mochi actor name, e.g.
            "BP_Robot_.../Articulation/fr3_link0" (slash format) or
            "bp_robot_...___articulation_fr3_link0" (underscore format)

    Returns:
        Bone name like "fr3_link0" or "dg5f_link_palm", or None if
        the name doesn't match any known pattern.
    """
    for pattern in _BONE_PATTERNS:
        m = pattern.search(actor_name)
        if m:
            return m.group(1)
    return None


def resolve_visual_mesh(
    bone_name: str,
    mesh_assets_dir: str | Path,
    hand: str = "right",
) -> Path | None:
    """Map a Mochi bone name to a visual mesh file path.

    Handles naming conventions for FR3 arm links and DG5F hand links.

    Args:
        bone_name: Bone name extracted via extract_bone_name(), e.g.
            "fr3_link0", "dg5f_link_palm", "dg5f_link_1_1".
        mesh_assets_dir: Root directory containing franka_description/
            and dg_description/ subdirectories with visual meshes.
        hand: "right" or "left" for DG5F hand side.

    Returns:
        Path to the visual mesh file (DAE/GLB/STL), or None if not found.
    """
    mesh_assets_dir = Path(mesh_assets_dir)

    # FR3 arm links: fr3_link{N} -> franka_description/.../visual/link{N}.dae
    fr3_match = re.match(r"fr3_link(\d+)$", bone_name)
    if fr3_match:
        link_num = fr3_match.group(1)
        dae_path = (
            mesh_assets_dir
            / "franka_description"
            / "meshes"
            / "robot_arms"
            / "fr3"
            / "visual"
            / f"link{link_num}.dae"
        )
        if dae_path.exists():
            return dae_path
        return None

    # DG5F hand links: dg5f_link_{part} -> dg_description/.../visual/{prefix}_dg_{part}.dae
    dg5f_match = re.match(r"dg5f_link_(.+)$", bone_name)
    if dg5f_match:
        part = dg5f_match.group(1)
        prefix = "rl" if hand == "right" else "ll"
        dae_path = (
            mesh_assets_dir
            / "dg_description"
            / "meshes"
            / f"dg5f_{hand}"
            / "visual"
            / f"{prefix}_dg_{part}.dae"
        )
        if dae_path.exists():
            return dae_path
        return None

    logger.debug("No visual mesh mapping for bone: %s", bone_name)
    return None
