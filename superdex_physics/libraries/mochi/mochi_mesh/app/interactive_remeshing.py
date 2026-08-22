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

"""
Interactive surface remeshing tool.

Loads a mesh from a file, visualizes original and remeshed surfaces side-by-side,
and provides an interactive UI for tweaking remeshing parameters and re-running.

Run with --help for the full list of options.
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np
from mochi.mesh import (
    compute_mesh_statistics,
    MeshStatistics,
    reconstruct_surface_from_sdf,
    remesh_surface,
    RemeshMethod,
    SurfaceRemeshingParams,
)
from mochi.utils.transformations import make_transform
from mochi.viewer import Viewer, ViewerCfg
from mochi.viewer.backend import polyscope_imgui as psim
from mochi.viewer.renderers.mesh_renderer import MeshRenderer
from mochi.viewer.ui.widgets import tooltip_on_hover


@dataclass
class RemeshingState:
    """Mutable state shared between UI callbacks and the render loop."""

    original_vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    original_faces: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.int32)
    )
    remeshed_vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    remeshed_faces: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.int32)
    )
    params: SurfaceRemeshingParams = field(default_factory=SurfaceRemeshingParams)
    original_stats: MeshStatistics | None = None
    remeshed_stats: MeshStatistics | None = None
    needs_remesh: bool = False
    needs_load: bool = False
    keep_original_coordinates: bool = False
    export_remesh_obj: bool = False
    edge_swapping: bool = True
    edge_swap_threshold: float = 0.001
    mesh_path: str = ""
    output_path: str | None = None
    status_message: str = ""
    has_mesh: bool = False

    # SDF state
    has_sdf: bool = False
    sdf_status_message: str = ""
    sdf_dims: np.ndarray | None = None
    sdf_bounds_min: np.ndarray | None = None
    sdf_bounds_max: np.ndarray | None = None
    sdf_neg_bounds_min: np.ndarray | None = None
    sdf_neg_bounds_max: np.ndarray | None = None
    sdf_values: np.ndarray | None = None
    # Optional parent-from-grid transform fields preserved across load/save round-trip.
    sdf_scale: np.ndarray | None = None
    sdf_rotation: np.ndarray | None = None
    sdf_translation: np.ndarray | None = None
    recon_vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    recon_faces: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.int32)
    )
    recon_stats: MeshStatistics | None = None
    show_recon: bool = False
    show_grid: bool = False
    needs_update_recon_view: bool = False
    needs_update_grid_view: bool = False

    # SDF baking defaults (overridable via CLI args)
    initial_sdf_resolution_mode: int = 0
    initial_sdf_resolution_delta: float = 0.25
    initial_sdf_boundary_padding_dist: float = 0.005
    initial_sdf_min_grid_resolution: int = 6

    # Display settings
    edge_width_scale: float = 0.5
    needs_update_edge_width: bool = False

    # Pending params JSON to apply to the UI on the next frame
    _pending_params: dict | None = field(default=None, repr=False)


def _clear_sdf_state(state: RemeshingState) -> None:
    """Reset all SDF-related state."""
    state.has_sdf = False
    state.sdf_status_message = ""
    state.sdf_dims = None
    state.sdf_bounds_min = None
    state.sdf_bounds_max = None
    state.sdf_neg_bounds_min = None
    state.sdf_neg_bounds_max = None
    state.sdf_values = None
    state.sdf_scale = None
    state.sdf_rotation = None
    state.sdf_translation = None
    state.recon_vertices = np.zeros((0, 3))
    state.recon_faces = np.zeros((0, 3), dtype=np.int32)
    state.recon_stats = None
    state.show_recon = False
    state.show_grid = False
    state.needs_update_recon_view = True
    state.needs_update_grid_view = True


_PARAM_JSON_KEYS = [
    "remesh_method",
    "edge_size",
    "detect_features",
    "relative_to_mesh_size",
    "alpha_wrap_relative_alpha",
    "alpha_wrap_relative_offset",
    "smoothing_iterations",
    "relaxation_steps_per_iteration",
    "tangential_relaxation_iterations",
    "angle_smoothing_iterations",
    "sharp_feature_angle",
    "protect_constraints",
    "relax_constraints",
    "use_adaptive_sizing",
    "adaptive_sizing_tolerance",
    "min_edge_size_factor",
    "max_edge_size_factor",
    "repair_mesh",
    "target_vertex_count",
    "acvd_gradation_factor",
    "facet_angle_bound",
    "facet_distance_bound",
    "sdf_resolution_mode",
    "sdf_resolution_delta",
    "sdf_boundary_padding_dist",
    "sdf_min_grid_resolution",
    "mesh_output_path",
    "keep_original_coordinates",
    "edge_swapping",
    "edge_swap_threshold",
]

_SDF_JSON_KEY_TO_STATE_ATTR = {
    "sdf_resolution_mode": "initial_sdf_resolution_mode",
    "sdf_resolution_delta": "initial_sdf_resolution_delta",
    "sdf_boundary_padding_dist": "initial_sdf_boundary_padding_dist",
    "sdf_min_grid_resolution": "initial_sdf_min_grid_resolution",
}


def _convert_params_to_display_units(data: dict) -> dict:
    """Convert internal param values to display units for JSON storage."""
    result = dict(data)
    relative = result.get("relative_to_mesh_size", False)
    if "edge_size" in result:
        result["edge_size"] = result["edge_size"] * (100.0 if relative else 1000.0)
    sdf_mode = result.get("sdf_resolution_mode", 0)
    if "sdf_resolution_delta" in result:
        result["sdf_resolution_delta"] = result["sdf_resolution_delta"] * (
            1000.0 if sdf_mode == 6 else 100.0
        )
    if "sdf_boundary_padding_dist" in result:
        result["sdf_boundary_padding_dist"] = (
            result["sdf_boundary_padding_dist"] * 1000.0
        )
    if "facet_distance_bound" in result:
        result["facet_distance_bound"] = result["facet_distance_bound"] * 1000.0
    if "edge_swap_threshold" in result:
        result["edge_swap_threshold"] = result["edge_swap_threshold"] * 1000.0
    return result


def _convert_params_from_display_units(data: dict) -> dict:
    """Convert display-unit param values back to internal units from JSON."""
    result = dict(data)
    relative = result.get("relative_to_mesh_size", False)
    if "edge_size" in result:
        result["edge_size"] = result["edge_size"] / (100.0 if relative else 1000.0)
    sdf_mode = result.get("sdf_resolution_mode", 0)
    if "sdf_resolution_delta" in result:
        result["sdf_resolution_delta"] = result["sdf_resolution_delta"] / (
            1000.0 if sdf_mode == 6 else 100.0
        )
    if "sdf_boundary_padding_dist" in result:
        result["sdf_boundary_padding_dist"] = (
            result["sdf_boundary_padding_dist"] / 1000.0
        )
    if "facet_distance_bound" in result:
        result["facet_distance_bound"] = result["facet_distance_bound"] / 1000.0
    if "edge_swap_threshold" in result:
        result["edge_swap_threshold"] = result["edge_swap_threshold"] / 1000.0
    return result


def _get_params_json_path(mesh_path: str) -> str:
    """Return the params JSON path adjacent to a mesh file."""
    return mesh_path + ".mochiMeshApp.json"


def _save_params_json(mesh_path: str, ui: dict) -> None:
    """Save current UI params to a JSON file adjacent to the mesh."""
    if not mesh_path:
        return
    json_path = _get_params_json_path(mesh_path)
    data = {k: ui[k] for k in _PARAM_JSON_KEYS if k in ui}
    data = _convert_params_to_display_units(data)
    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
    except Exception:
        pass


def _load_params_json(mesh_path: str) -> dict | None:
    """Load params from a JSON file adjacent to the mesh."""
    if not mesh_path:
        return None
    json_path = _get_params_json_path(mesh_path)
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path) as f:
            return json.load(f)
    except Exception:
        return None


def _apply_params_dict_to_ui(ui: dict, data: dict) -> None:
    """Apply loaded params dict values to the UI dict."""
    converted = _convert_params_from_display_units(data)
    for k in _PARAM_JSON_KEYS:
        if k in converted and k in ui:
            ui[k] = converted[k]


def _apply_params_json_to_params_and_state(
    data: dict, p: SurfaceRemeshingParams, state: RemeshingState
) -> None:
    """Apply loaded params dict to a SurfaceRemeshingParams and state (for CLI load)."""
    converted = _convert_params_from_display_units(data)
    if "remesh_method" in converted:
        p.method = RemeshMethod(converted["remesh_method"])
    for k in _PARAM_JSON_KEYS:
        if k == "remesh_method":
            continue
        if k in _SDF_JSON_KEY_TO_STATE_ATTR:
            if k in converted:
                setattr(state, _SDF_JSON_KEY_TO_STATE_ATTR[k], converted[k])
        elif k == "mesh_output_path":
            if k in converted and converted[k]:
                state.output_path = converted[k]
        elif k == "keep_original_coordinates":
            if k in converted:
                state.keep_original_coordinates = converted[k]
        elif k == "edge_swapping":
            if k in converted:
                state.edge_swapping = converted[k]
        elif k == "edge_swap_threshold":
            if k in converted:
                state.edge_swap_threshold = converted[k]
        elif k in converted and hasattr(p, k):
            setattr(p, k, converted[k])


def _closest_point_on_triangle(
    point: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> float:
    """Squared distance from point to the closest point on triangle (v0, v1, v2)."""
    edge0 = v1 - v0
    edge1 = v2 - v0
    v0_to_point = point - v0

    a = float(np.dot(edge0, edge0))
    b = float(np.dot(edge0, edge1))
    c = float(np.dot(edge1, edge1))
    d = float(np.dot(edge0, v0_to_point))
    e = float(np.dot(edge1, v0_to_point))

    det = a * c - b * b
    if det < 1e-30:
        d0 = float(np.sum((point - v0) ** 2))
        d1 = float(np.sum((point - v1) ** 2))
        d2 = float(np.sum((point - v2) ** 2))
        return min(d0, d1, d2)

    s = b * e - c * d
    t = b * d - a * e

    if s + t <= det:
        if s < 0:
            if t < 0:
                if d < 0:
                    s = min(max(-d / a, 0.0), 1.0) if a > 0 else 0.0
                    t = 0.0
                else:
                    s = 0.0
                    t = min(max(-e / c, 0.0), 1.0) if c > 0 else 0.0
            else:
                s = 0.0
                t = min(max(-e / c, 0.0), 1.0) if c > 0 else 0.0
        elif t < 0:
            t = 0.0
            s = min(max(-d / a, 0.0), 1.0) if a > 0 else 0.0
        else:
            inv_det = 1.0 / det
            s *= inv_det
            t *= inv_det
    else:
        if s < 0:
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                s = min(max(numer / denom, 0.0), 1.0) if denom > 0 else 0.0
                t = 1.0 - s
            else:
                s = 0.0
                t = min(max(-e / c, 0.0), 1.0) if c > 0 else 0.0
        elif t < 0:
            tmp0 = b + e
            tmp1 = a + d
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                t = min(max(numer / denom, 0.0), 1.0) if denom > 0 else 0.0
                s = 1.0 - t
            else:
                t = 0.0
                s = min(max(-d / a, 0.0), 1.0) if a > 0 else 0.0
        else:
            numer = (c + e) - (b + d)
            if numer <= 0:
                s = 0.0
            else:
                denom = a - 2.0 * b + c
                s = min(numer / denom, 1.0) if denom > 0 else 0.0
            t = 1.0 - s

    closest = v0 + s * edge0 + t * edge1
    return float(np.sum((point - closest) ** 2))


def _exact_point_to_mesh_dist_sq(
    point: np.ndarray,
    orig_vertices: np.ndarray,
    orig_faces: np.ndarray,
    vtx_to_faces: list[list[int]],
    tree,
    k: int = 8,
) -> float:
    """Exact squared distance from point to the original mesh surface."""
    _, nearest_indices = tree.query(point.astype(np.float64), k=k)
    if isinstance(nearest_indices, (int, np.integer)):
        nearest_indices = [nearest_indices]
    n_orig = len(orig_vertices)
    candidate_faces: set[int] = set()
    for vi in nearest_indices:
        if vi < n_orig:
            candidate_faces.update(vtx_to_faces[vi])

    if not candidate_faces:
        return float(tree.query(point.astype(np.float64))[0]) ** 2

    min_dist_sq = float("inf")
    for fi in candidate_faces:
        tri = orig_faces[fi]
        d_sq = _closest_point_on_triangle(
            point, orig_vertices[tri[0]], orig_vertices[tri[1]], orig_vertices[tri[2]]
        )
        if d_sq < min_dist_sq:
            min_dist_sq = d_sq
    return min_dist_sq


def _build_surface_sample_tree(orig_vertices: np.ndarray, orig_faces: np.ndarray):
    """Build a KD-tree from densely sampled points on the original mesh surface.

    Samples vertices, edge midpoints, and face centroids to approximate the
    surface for fast distance queries.
    """
    from scipy.spatial import cKDTree

    samples = [orig_vertices]

    orig_edges: set[tuple[int, int]] = set()
    for face in orig_faces:
        for j in range(3):
            a, b = int(face[j]), int(face[(j + 1) % 3])
            orig_edges.add((min(a, b), max(a, b)))
    if orig_edges:
        oe = np.array(list(orig_edges))
        samples.append((orig_vertices[oe[:, 0]] + orig_vertices[oe[:, 1]]) * 0.5)

    samples.append(
        (
            orig_vertices[orig_faces[:, 0]]
            + orig_vertices[orig_faces[:, 1]]
            + orig_vertices[orig_faces[:, 2]]
        )
        / 3.0
    )

    return cKDTree(np.concatenate(samples, axis=0).astype(np.float64))


def _build_edge_adjacency(
    faces: np.ndarray,
) -> tuple[dict[tuple[int, int], list[int]], list[tuple[tuple[int, int], int, int]]]:
    """Build edge-to-triangle adjacency and return interior edges."""
    edge_to_tris: dict[tuple[int, int], list[int]] = {}
    for fi in range(len(faces)):
        f = faces[fi]
        for j in range(3):
            e = (
                min(int(f[j]), int(f[(j + 1) % 3])),
                max(int(f[j]), int(f[(j + 1) % 3])),
            )
            edge_to_tris.setdefault(e, []).append(fi)

    interior_edges = [
        (e, tris[0], tris[1]) for e, tris in edge_to_tris.items() if len(tris) == 2
    ]
    return edge_to_tris, interior_edges


_STALE = "STALE"


def _try_swap_edge(
    vertices: np.ndarray,
    faces: np.ndarray,
    edge: tuple[int, int],
    fi0: int,
    fi1: int,
    edge_to_tris: dict[tuple[int, int], list[int]],
    reason_out: list | None = None,
) -> bool:
    """Attempt an edge swap, returning True if the swap was applied.

    Updates edge_to_tris incrementally on success. Appends _STALE to reason_out
    if the edge no longer exists in both faces (stale pre-computed data).
    """
    tri0 = faces[fi0]
    tri1 = faces[fi1]
    shared = set(edge)

    if not (
        shared <= {int(tri0[0]), int(tri0[1]), int(tri0[2])}
        and shared <= {int(tri1[0]), int(tri1[1]), int(tri1[2])}
    ):
        if reason_out is not None:
            reason_out.append(_STALE)
        return False

    opp0 = a_local = b_local = -1
    for i in range(3):
        if int(tri0[i]) not in shared:
            opp0 = int(tri0[i])
            a_local = int(tri0[(i + 1) % 3])
            b_local = int(tri0[(i + 2) % 3])
            break

    opp1 = -1
    for i in range(3):
        if int(tri1[i]) not in shared:
            opp1 = int(tri1[i])
            break

    if opp0 == -1 or opp1 == -1 or opp0 == opp1:
        return False

    alt_edge_key = (min(opp0, opp1), max(opp0, opp1))
    if alt_edge_key in edge_to_tris:
        return False

    new_tri0 = np.array([opp0, a_local, opp1], dtype=faces.dtype)
    new_tri1 = np.array([opp1, b_local, opp0], dtype=faces.dtype)

    def _tri_normal(t):
        return np.cross(
            vertices[t[1]] - vertices[t[0]],
            vertices[t[2]] - vertices[t[0]],
        )

    n_avg = _tri_normal(tri0) + _tri_normal(tri1)
    n_new0 = _tri_normal(new_tri0)
    n_new1 = _tri_normal(new_tri1)
    if float(np.dot(n_avg, n_new0)) <= 0:
        return False
    if float(np.dot(n_avg, n_new1)) <= 0:
        return False

    def _max_edge_len_sq(t):
        e0 = vertices[t[1]] - vertices[t[0]]
        e1 = vertices[t[2]] - vertices[t[1]]
        e2 = vertices[t[0]] - vertices[t[2]]
        return max(np.dot(e0, e0), np.dot(e1, e1), np.dot(e2, e2))

    if float(np.linalg.norm(n_new0)) < float(_max_edge_len_sq(new_tri0)) * 0.01:
        return False
    if float(np.linalg.norm(n_new1)) < float(_max_edge_len_sq(new_tri1)) * 0.01:
        return False

    for fi in [fi0, fi1]:
        old_f = faces[fi]
        for j in range(3):
            ek = (
                min(int(old_f[j]), int(old_f[(j + 1) % 3])),
                max(int(old_f[j]), int(old_f[(j + 1) % 3])),
            )
            lst = edge_to_tris.get(ek)
            if lst is not None and fi in lst:
                lst.remove(fi)
                if not lst:
                    del edge_to_tris[ek]

    faces[fi0] = new_tri0
    faces[fi1] = new_tri1

    for fi in [fi0, fi1]:
        new_f = faces[fi]
        for j in range(3):
            ek = (
                min(int(new_f[j]), int(new_f[(j + 1) % 3])),
                max(int(new_f[j]), int(new_f[(j + 1) % 3])),
            )
            edge_to_tris.setdefault(ek, []).append(fi)

    return True


def _edge_swap_post_process(
    vertices: np.ndarray,
    faces: np.ndarray,
    orig_vertices: np.ndarray,
    orig_faces: np.ndarray,
    num_planar_passes: int = 3,
    num_surface_passes: int = 3,
    surface_distance_threshold: float = 0.001,
    coplanarity_threshold: float = 0.999,
) -> tuple[np.ndarray, int]:
    """Swap edges to better match the original surface and improve triangle quality.

    Phase 2a (planar): for coplanar triangle pairs, swaps to the shorter
    diagonal to fix degenerate/obtuse pairs from isotropic remeshing.

    Phase 2b (surface): for non-coplanar pairs whose edge midpoint is farther
    than surface_distance_threshold from the original surface, swaps if the
    alternate diagonal's midpoint is closer (using exact point-to-triangle
    distance).

    Returns the modified faces array and total number of swaps performed.
    """
    if len(faces) == 0 or len(orig_faces) == 0:
        return faces, 0

    faces = faces.copy()

    sample_tree = _build_surface_sample_tree(orig_vertices, orig_faces)

    from scipy.spatial import cKDTree

    orig_vtx_tree = cKDTree(orig_vertices.astype(np.float64))
    orig_vtx_to_faces: list[list[int]] = [[] for _ in range(len(orig_vertices))]
    for fi in range(len(orig_faces)):
        for vi in orig_faces[fi]:
            orig_vtx_to_faces[int(vi)].append(fi)

    total_swapped = 0

    # Phase 2a: planar edge optimization (vectorized candidate selection)
    print("  Phase 2a: planar edge optimization...")
    for _pass_idx in range(num_planar_passes):
        edge_to_tris, interior_edges = _build_edge_adjacency(faces)
        if not interior_edges:
            break

        edge_a = np.array([e[0] for e, _, _ in interior_edges], dtype=np.int64)
        edge_b = np.array([e[1] for e, _, _ in interior_edges], dtype=np.int64)
        fi0_arr = np.array([fi0 for _, fi0, _ in interior_edges], dtype=np.int64)
        fi1_arr = np.array([fi1 for _, _, fi1 in interior_edges], dtype=np.int64)

        tri0_sum = (
            faces[fi0_arr, 0].astype(np.int64) + faces[fi0_arr, 1] + faces[fi0_arr, 2]
        )
        tri1_sum = (
            faces[fi1_arr, 0].astype(np.int64) + faces[fi1_arr, 1] + faces[fi1_arr, 2]
        )
        opp0_arr = tri0_sum - edge_a - edge_b
        opp1_arr = tri1_sum - edge_a - edge_b

        valid = (opp0_arr >= 0) & (opp1_arr >= 0) & (opp0_arr != opp1_arr)

        edge_lens_sq = np.sum((vertices[edge_b] - vertices[edge_a]) ** 2, axis=1)
        alt_lens_sq = np.sum((vertices[opp1_arr] - vertices[opp0_arr]) ** 2, axis=1)

        e0_t0 = vertices[faces[fi0_arr, 1]] - vertices[faces[fi0_arr, 0]]
        e1_t0 = vertices[faces[fi0_arr, 2]] - vertices[faces[fi0_arr, 0]]
        n0 = np.cross(e0_t0, e1_t0)
        e0_t1 = vertices[faces[fi1_arr, 1]] - vertices[faces[fi1_arr, 0]]
        e1_t1 = vertices[faces[fi1_arr, 2]] - vertices[faces[fi1_arr, 0]]
        n1 = np.cross(e0_t1, e1_t1)

        len_n0 = np.linalg.norm(n0, axis=1)
        len_n1 = np.linalg.norm(n1, axis=1)
        max_n0 = np.linalg.norm(e0_t0, axis=1) * np.linalg.norm(e1_t0, axis=1)
        max_n1 = np.linalg.norm(e0_t1, axis=1) * np.linalg.norm(e1_t1, axis=1)

        degenerate = (len_n0 < max_n0 * 1e-3) | (len_n1 < max_n1 * 1e-3)
        safe = (len_n0 > 1e-30) & (len_n1 > 1e-30)
        dots = np.sum(n0 * n1, axis=1)
        cos_angle = np.where(safe, dots / (len_n0 * len_n1), 0.0)
        is_coplanar = degenerate | (safe & (cos_angle > coplanarity_threshold))

        candidate_mask = valid & is_coplanar & (alt_lens_sq < edge_lens_sq)
        candidate_indices = np.where(candidate_mask)[0]
        ratios = edge_lens_sq[candidate_indices] / np.maximum(
            alt_lens_sq[candidate_indices], 1e-30
        )
        candidate_indices = candidate_indices[np.argsort(-ratios)]

        swapped = 0
        modified_tris: set[int] = set()
        had_skips = False

        for ci in candidate_indices:
            fi0 = int(fi0_arr[ci])
            fi1 = int(fi1_arr[ci])
            if fi0 in modified_tris or fi1 in modified_tris:
                had_skips = True
                continue
            edge = (int(edge_a[ci]), int(edge_b[ci]))
            if _try_swap_edge(vertices, faces, edge, fi0, fi1, edge_to_tris):
                modified_tris.add(fi0)
                modified_tris.add(fi1)
                swapped += 1

        total_swapped += swapped
        if swapped == 0 and not had_skips:
            break

    # Phase 2b: surface-distance swaps for non-coplanar pairs
    print(f"  Phase 2a complete: {total_swapped} swaps")
    print(
        f"  Phase 2b: surface-distance swaps (threshold={surface_distance_threshold * 1000:.3f}mm)..."
    )
    phase2b_swapped = 0
    threshold_sq = surface_distance_threshold**2

    for _pass_idx in range(num_surface_passes):
        edge_to_tris, interior_edges = _build_edge_adjacency(faces)
        if not interior_edges:
            break

        edge_a = np.array([e[0] for e, _, _ in interior_edges], dtype=np.int64)
        edge_b = np.array([e[1] for e, _, _ in interior_edges], dtype=np.int64)
        fi0_arr = np.array([fi0 for _, fi0, _ in interior_edges], dtype=np.int64)
        fi1_arr = np.array([fi1 for _, _, fi1 in interior_edges], dtype=np.int64)

        midpoints = (vertices[edge_a] + vertices[edge_b]) * 0.5
        dists, _ = sample_tree.query(midpoints.astype(np.float64))
        dists_sq = dists**2

        above_threshold = dists_sq >= threshold_sq
        if not np.any(above_threshold):
            break

        # Pre-filter: skip coplanar pairs (vectorized)
        e0_t0 = vertices[faces[fi0_arr, 1]] - vertices[faces[fi0_arr, 0]]
        e1_t0 = vertices[faces[fi0_arr, 2]] - vertices[faces[fi0_arr, 0]]
        n0 = np.cross(e0_t0, e1_t0)
        e0_t1 = vertices[faces[fi1_arr, 1]] - vertices[faces[fi1_arr, 0]]
        e1_t1 = vertices[faces[fi1_arr, 2]] - vertices[faces[fi1_arr, 0]]
        n1 = np.cross(e0_t1, e1_t1)
        len_n0 = np.linalg.norm(n0, axis=1)
        len_n1 = np.linalg.norm(n1, axis=1)
        safe = (len_n0 > 1e-30) & (len_n1 > 1e-30)
        dots = np.sum(n0 * n1, axis=1)
        cos_angle = np.where(safe, dots / (len_n0 * len_n1), 0.0)
        not_coplanar = ~(safe & (cos_angle > coplanarity_threshold))

        candidate_mask = above_threshold & not_coplanar
        candidate_indices = np.where(candidate_mask)[0]
        candidate_indices = candidate_indices[np.argsort(-dists_sq[candidate_indices])]

        tri0_sum = (
            faces[fi0_arr, 0].astype(np.int64) + faces[fi0_arr, 1] + faces[fi0_arr, 2]
        )
        tri1_sum = (
            faces[fi1_arr, 0].astype(np.int64) + faces[fi1_arr, 1] + faces[fi1_arr, 2]
        )
        opp0_arr = tri0_sum - edge_a - edge_b
        opp1_arr = tri1_sum - edge_a - edge_b

        swapped = 0
        modified_tris: set[int] = set()
        had_skips = False

        for ci in candidate_indices:
            fi0 = int(fi0_arr[ci])
            fi1 = int(fi1_arr[ci])
            if fi0 in modified_tris or fi1 in modified_tris:
                had_skips = True
                continue

            opp0 = int(opp0_arr[ci])
            opp1 = int(opp1_arr[ci])
            if opp0 < 0 or opp1 < 0 or opp0 == opp1:
                continue

            alt_edge_key = (min(opp0, opp1), max(opp0, opp1))
            if alt_edge_key in edge_to_tris:
                continue

            cur_mid = midpoints[ci]
            cur_dist_sq = _exact_point_to_mesh_dist_sq(
                cur_mid,
                orig_vertices,
                orig_faces,
                orig_vtx_to_faces,
                orig_vtx_tree,
            )
            alt_mid = (vertices[opp0] + vertices[opp1]) * 0.5
            alt_dist_sq = _exact_point_to_mesh_dist_sq(
                alt_mid,
                orig_vertices,
                orig_faces,
                orig_vtx_to_faces,
                orig_vtx_tree,
            )

            if alt_dist_sq >= cur_dist_sq:
                continue

            edge = (int(edge_a[ci]), int(edge_b[ci]))
            if _try_swap_edge(vertices, faces, edge, fi0, fi1, edge_to_tris):
                modified_tris.add(fi0)
                modified_tris.add(fi1)
                swapped += 1

        phase2b_swapped += swapped
        total_swapped += swapped
        if swapped == 0 and not had_skips:
            break

    print(f"  Phase 2b complete: {phase2b_swapped} swaps")
    return faces, total_swapped


def run_remesh(state: RemeshingState) -> None:
    """Execute remeshing with current parameters and update statistics."""
    state.status_message = "Remeshing..."
    p = state.params

    saved_tangential = p.tangential_relaxation_iterations
    saved_angle_smoothing = p.angle_smoothing_iterations

    # Phase 1: primary remesh without relaxation
    print("Phase 1: primary remesh (isotropic remeshing, no relaxation)...")
    p.tangential_relaxation_iterations = 0
    p.angle_smoothing_iterations = 0
    state.remeshed_vertices, state.remeshed_faces = remesh_surface(
        state.original_vertices, state.original_faces, p
    )
    p.tangential_relaxation_iterations = saved_tangential
    p.angle_smoothing_iterations = saved_angle_smoothing

    # Phase 2: edge swap post-process on unrelaxed mesh
    num_swaps = 0
    if state.edge_swapping:
        print("Phase 2: edge swap post-process...")
        state.remeshed_faces, num_swaps = _edge_swap_post_process(
            state.remeshed_vertices,
            state.remeshed_faces,
            state.original_vertices,
            state.original_faces,
            surface_distance_threshold=state.edge_swap_threshold,
        )

    # Phase 3: tangential relaxation on improved connectivity
    if saved_tangential > 0 or saved_angle_smoothing > 0:
        print("Phase 3: tangential relaxation...")
        relax_params = SurfaceRemeshingParams()
        relax_params.method = RemeshMethod(0)
        relax_params.smoothing_iterations = 0
        relax_params.tangential_relaxation_iterations = saved_tangential
        relax_params.angle_smoothing_iterations = saved_angle_smoothing
        relax_params.repair_mesh = False
        relax_params.detect_features = p.detect_features
        relax_params.sharp_feature_angle = p.sharp_feature_angle
        relax_params.edge_size = p.edge_size
        relax_params.relative_to_mesh_size = p.relative_to_mesh_size
        relax_params.protect_constraints = p.protect_constraints
        relax_params.relax_constraints = p.relax_constraints
        state.remeshed_vertices, state.remeshed_faces = remesh_surface(
            state.remeshed_vertices, state.remeshed_faces, relax_params
        )

    print("All phases complete. Computing statistics...")

    state.original_stats = compute_mesh_statistics(
        state.original_vertices, state.original_faces
    )
    state.remeshed_stats = compute_mesh_statistics(
        state.remeshed_vertices,
        state.remeshed_faces,
        state.original_vertices,
        state.original_faces,
    )
    swap_msg = f" ({num_swaps} edge swaps)" if num_swaps > 0 else ""
    state.status_message = (
        f"Done: {state.remeshed_stats.num_vertices} verts, "
        f"{state.remeshed_stats.num_faces} faces{swap_msg}"
    )

    if state.export_remesh_obj and state.mesh_path:
        obj_path = os.path.splitext(state.mesh_path)[0] + "_remeshed.obj"
        _save_obj(obj_path, state.remeshed_vertices, state.remeshed_faces)
        print(f"Exported remeshed OBJ: {obj_path}")
        state.status_message += f" | Exported {obj_path}"


def _build_model_data(state: RemeshingState):
    """Construct a mochi ModelData from the current remeshed mesh."""
    import mochi  # @manual

    coords = state.remeshed_vertices.astype(np.float32).flatten()
    conn = state.remeshed_faces.astype(np.int32).flatten()
    mesh_data = mochi.MeshData(
        nodes_per_element=3, coordinates=coords, connectivity=conn
    )
    return mochi.ModelData(mesh=mesh_data)


def _ensure_h5_extension(path: str) -> str:
    """Ensure the path has an .h5 extension."""
    root, ext = os.path.splitext(path)
    if ext.lower() != ".h5":
        return root + ".h5"
    return path


def _extract_sdf_arrays(
    sdf,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Extract SDF grid data into numpy arrays.

    Args:
        sdf: A mochi GridSdfData object.

    Returns:
        Tuple of (dims, values, bounds_min, bounds_max,
        neg_bounds_min, neg_bounds_max, scale, rotation, translation) as numpy
        arrays. The trailing scale/rotation/translation entries are ``None`` when
        the source SDF does not carry an optional parent-from-grid transform.
    """
    scale = np.array(sdf.scale, dtype=np.float32) if sdf.scale is not None else None
    rotation = (
        np.array(sdf.rotation, dtype=np.float32) if sdf.rotation is not None else None
    )
    translation = (
        np.array(sdf.translation, dtype=np.float32)
        if sdf.translation is not None
        else None
    )
    return (
        np.array(sdf.dims, dtype=np.int32),
        np.array(sdf.values, dtype=np.float32),
        np.array(sdf.bounds.min, dtype=np.float32),
        np.array(sdf.bounds.max, dtype=np.float32),
        np.array(sdf.negative_value_bounds.min, dtype=np.float32),
        np.array(sdf.negative_value_bounds.max, dtype=np.float32),
        scale,
        rotation,
        translation,
    )


def save_mesh(state: RemeshingState) -> None:
    """Export the remeshed mesh to an H5 file using mochi's model API."""
    import mochi  # @manual

    if state.output_path is None:
        return
    path = _ensure_h5_extension(state.output_path)
    model = _build_model_data(state)
    mochi.model.save_to_file(model, path, mochi.FileFormat.H5)
    state.status_message = f"Saved to {path}"


def _save_obj(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a triangle mesh to an OBJ file."""
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def _fmt(v: float, decimals: int = 4) -> str:
    """Format a float, using scientific notation if too small for fixed-point."""
    if v != 0.0 and abs(v) < 10 ** (-decimals):
        return f"{v:.2e}"
    return f"{v:.{decimals}f}"


def _format_stats(stats: MeshStatistics) -> list[str]:
    """Format mesh statistics as a list of display strings."""
    el = stats.edge_lengths
    a = stats.angles
    lines = [
        f"Vertices: {stats.num_vertices}   Faces: {stats.num_faces}",
        f"Watertight: {stats.is_closed}",
        f"Edge len: mean={_fmt(el.mean)} std={_fmt(el.standard_deviation)}",
        f"          min={_fmt(el.min)} max={_fmt(el.max)}",
        f"Angles:   mean={_fmt(a.mean, 1)} std={_fmt(a.standard_deviation, 1)}",
        f"          min={_fmt(a.min, 1)} max={_fmt(a.max, 1)}",
    ]
    if stats.hausdorff_distance >= 0:
        lines.append(f"Hausdorff: {_fmt(stats.hausdorff_distance, 6)}")
    return lines


def _imgui_update(ui: dict, key: str, widget_fn, *args):
    """Call an imgui widget and update the ui dict if the value changed.

    Widget is called as widget_fn(args[0], ui[key], *args[1:]) so that the
    current value is always the second positional argument (matching the
    convention used by InputFloat, InputInt, Checkbox, and Combo).
    """
    changed, val = widget_fn(args[0], ui[key], *args[1:])
    if changed:
        ui[key] = val


def _build_criteria_section(ui: dict) -> None:
    """Draw the Mesh Criteria collapsing header."""
    if not psim.CollapsingHeader(
        "Mesh Criteria", flags=psim.ImGuiTreeNodeFlags_DefaultOpen
    ):
        return
    _imgui_update(
        ui,
        "remesh_method",
        psim.Combo,
        "Remesh Method",
        _REMESH_METHOD_NAMES,
    )
    tooltip_on_hover(
        "None: skip primary remeshing, only run post-processing.\n"
        "Alpha Wrap: alpha-wrapping for watertight output.\n"
        "ACVD: approximate centroidal Voronoi diagram remeshing.\n"
        "Surface Delaunay: surface Delaunay triangulation."
    )
    if ui["relative_to_mesh_size"]:
        display_val = ui["edge_size"] * 100.0
        changed, new_val = psim.InputFloat(
            "Edge Size (Percentage)", display_val, 0.0, 0.0, "%.6f"
        )
        if changed:
            ui["edge_size"] = new_val / 100.0
    else:
        display_val = ui["edge_size"] * 1000.0
        changed, new_val = psim.InputFloat(
            "Edge Size (Millimeters)", display_val, 0.0, 0.0, "%.6f"
        )
        if changed:
            ui["edge_size"] = new_val / 1000.0
    tooltip_on_hover("Target edge length for isotropic remeshing.")
    _imgui_update(ui, "detect_features", psim.Checkbox, "Detect Features")
    tooltip_on_hover("Detect sharp edges using the dihedral angle threshold below.")
    prev_relative = ui["relative_to_mesh_size"]
    _imgui_update(ui, "relative_to_mesh_size", psim.Checkbox, "Relative to Mesh Size")
    tooltip_on_hover("Interpret edge size as a fraction of the mesh bounding box.")
    if ui["relative_to_mesh_size"] != prev_relative:
        if ui["relative_to_mesh_size"]:
            ui["edge_size"] *= 10.0
        else:
            ui["edge_size"] /= 10.0
    if ui["remesh_method"] == 1:  # AlphaWrap
        _imgui_update(
            ui,
            "alpha_wrap_relative_alpha",
            psim.InputFloat,
            "Alpha (x edge_size)",
            0.0,
            0.0,
            "%.6f",
        )
        tooltip_on_hover("Alpha parameter as a multiple of edge size.")
        _imgui_update(
            ui,
            "alpha_wrap_relative_offset",
            psim.InputFloat,
            "Offset (x edge_size, 0=auto)",
            0.0,
            0.0,
            "%.6f",
        )
        tooltip_on_hover("Offset distance as a multiple of edge size. 0 = automatic.")
    if ui["remesh_method"] == 2:  # ACVD
        _imgui_update(
            ui, "target_vertex_count", psim.InputInt, "Target Vertex Count (0=auto)"
        )
        tooltip_on_hover("Target number of vertices. 0 = auto from edge size.")
        _imgui_update(
            ui,
            "acvd_gradation_factor",
            psim.InputFloat,
            "Gradation Factor",
            0.0,
            0.0,
            "%.6f",
        )
        tooltip_on_hover("Controls vertex density variation. 0 = uniform.")
    if ui["remesh_method"] == 3:  # SurfaceDelaunay
        _imgui_update(
            ui,
            "facet_angle_bound",
            psim.InputFloat,
            "Min Facet Angle [deg]",
            0.0,
            0.0,
            "%.6f",
        )
        tooltip_on_hover("Lower bound on triangle angles in the output mesh.")
        facet_dist_display = ui["facet_distance_bound"] * 1000.0
        changed, new_val = psim.InputFloat(
            "Facet Distance mm (0=auto)", facet_dist_display, 0.0, 0.0, "%.6f"
        )
        if changed:
            ui["facet_distance_bound"] = new_val / 1000.0
        tooltip_on_hover(
            "Max distance between facet centers and the surface. 0 = auto."
        )


def _build_postprocessing_section(ui: dict) -> None:
    """Draw the Post-Processing collapsing header."""
    if not psim.CollapsingHeader(
        "Post-Processing", flags=psim.ImGuiTreeNodeFlags_DefaultOpen
    ):
        return
    _imgui_update(
        ui,
        "sharp_feature_angle",
        psim.InputFloat,
        "Sharp Feature Angle (Degrees)",
        0.0,
        0.0,
        "%.6f",
    )
    tooltip_on_hover("Dihedral angle threshold for detecting sharp feature edges.")
    _imgui_update(ui, "smoothing_iterations", psim.InputInt, "Smoothing Iterations")
    tooltip_on_hover(
        "Number of isotropic remeshing iterations.\n"
        "Each iteration splits long edges, collapses short edges,\n"
        "flips edges, and relocates vertices toward uniform edge length."
    )
    _imgui_update(
        ui, "relaxation_steps_per_iteration", psim.InputInt, "Relaxation Steps/Iter"
    )
    tooltip_on_hover(
        "Tangential vertex relocation steps per isotropic remeshing iteration.\n"
        "Smooths vertex positions within each remeshing pass."
    )
    _imgui_update(ui, "protect_constraints", psim.Checkbox, "Protect Constraints")
    tooltip_on_hover("Hard-protect detected feature edges during remeshing.")
    _imgui_update(ui, "relax_constraints", psim.Checkbox, "Relax Constraints")
    tooltip_on_hover("Allow feature vertices to slide along feature polylines.")
    _imgui_update(ui, "repair_mesh", psim.Checkbox, "Repair Mesh")
    tooltip_on_hover(
        "Remove degenerate faces, stitch borders, fix self-intersections.\n"
        "Runs after isotropic remeshing, before edge swapping."
    )
    changed, val = psim.Checkbox("Edge Swapping", ui["edge_swapping"])
    if changed:
        ui["edge_swapping"] = val
    tooltip_on_hover(
        "Swap edges to improve triangle quality.\n"
        "Phase 2a: swap to shorter diagonal on coplanar pairs.\n"
        "Phase 2b: swap non-coplanar edges to better match original surface."
    )
    if ui["edge_swapping"]:
        edge_thresh_display = ui["edge_swap_threshold"] * 1000.0
        changed, new_val = psim.InputFloat(
            "Edge Swap Threshold (Millimeters)", edge_thresh_display, 0.0, 0.0, "%.6f"
        )
        if changed:
            ui["edge_swap_threshold"] = max(new_val / 1000.0, 0.0)
        tooltip_on_hover(
            "Minimum edge midpoint distance from original surface\n"
            "to consider a non-coplanar edge swap (Phase 2b only)."
        )
    _imgui_update(
        ui,
        "tangential_relaxation_iterations",
        psim.InputInt,
        "Tangential Relaxation Iters",
    )
    tooltip_on_hover(
        "Standalone tangential relaxation passes after edge swapping.\n"
        "Moves vertices along the surface to improve triangle quality\n"
        "without changing connectivity. Constrained by feature edges."
    )
    _imgui_update(
        ui, "angle_smoothing_iterations", psim.InputInt, "Angle Smoothing Iters"
    )
    tooltip_on_hover(
        "Legacy angle-based smoothing. Only runs when\n"
        "Tangential Relaxation Iters is 0."
    )


def _build_adaptive_section(ui: dict) -> None:
    """Draw the Adaptive Sizing collapsing header."""
    if not psim.CollapsingHeader("Adaptive Sizing"):
        return
    _imgui_update(ui, "use_adaptive_sizing", psim.Checkbox, "Use Adaptive Sizing")
    tooltip_on_hover(
        "Use curvature-based adaptive sizing field.\n"
        "Produces smaller triangles in high-curvature regions."
    )
    if ui["use_adaptive_sizing"]:
        _imgui_update(
            ui,
            "adaptive_sizing_tolerance",
            psim.InputFloat,
            "Tolerance",
            0.0,
            0.0,
            "%.6f",
        )
        tooltip_on_hover("Error tolerance for the adaptive sizing field.")
        _imgui_update(
            ui,
            "min_edge_size_factor",
            psim.InputFloat,
            "Min Edge Factor",
            0.0,
            0.0,
            "%.6f",
        )
        tooltip_on_hover("Minimum edge length as a fraction of edge size.")
        _imgui_update(
            ui,
            "max_edge_size_factor",
            psim.InputFloat,
            "Max Edge Factor",
            0.0,
            0.0,
            "%.6f",
        )
        tooltip_on_hover("Maximum edge length as a fraction of edge size.")


def _sync_ui_to_params(ui: dict, p: SurfaceRemeshingParams) -> None:
    """Copy UI dict values back into the pybind11 params object."""
    p.edge_size = ui["edge_size"]
    p.detect_features = ui["detect_features"]
    p.relative_to_mesh_size = ui["relative_to_mesh_size"]
    p.alpha_wrap_relative_alpha = ui["alpha_wrap_relative_alpha"]
    p.alpha_wrap_relative_offset = ui["alpha_wrap_relative_offset"]
    p.smoothing_iterations = ui["smoothing_iterations"]
    p.angle_smoothing_iterations = ui["angle_smoothing_iterations"]
    p.sharp_feature_angle = ui["sharp_feature_angle"]
    p.protect_constraints = ui["protect_constraints"]
    p.relax_constraints = ui["relax_constraints"]
    p.use_adaptive_sizing = ui["use_adaptive_sizing"]
    p.adaptive_sizing_tolerance = ui["adaptive_sizing_tolerance"]
    p.min_edge_size_factor = ui["min_edge_size_factor"]
    p.max_edge_size_factor = ui["max_edge_size_factor"]
    p.method = RemeshMethod(ui["remesh_method"])
    p.relaxation_steps_per_iteration = ui["relaxation_steps_per_iteration"]
    p.tangential_relaxation_iterations = ui["tangential_relaxation_iterations"]
    p.repair_mesh = ui["repair_mesh"]
    p.target_vertex_count = ui["target_vertex_count"]
    p.acvd_gradation_factor = ui["acvd_gradation_factor"]
    p.facet_angle_bound = ui["facet_angle_bound"]
    p.facet_distance_bound = ui["facet_distance_bound"]


_SDF_RESOLUTION_MODE_NAMES = [
    "Largest Axis",
    "Smallest Axis",
    "Mean Axis",
    "Largest Edge",
    "Smallest Edge",
    "Mean Edge",
    "Explicit",
]

_REMESH_METHOD_NAMES = ["None", "Alpha Wrap", "ACVD", "Surface Delaunay"]


def _sync_ui_to_sdf_params(ui: dict):
    """Build a GridSdfParams from UI dict values."""
    import mochi  # @manual

    modes = [
        mochi.GridSdfResolutionMode.LARGEST_AXIS,
        mochi.GridSdfResolutionMode.SMALLEST_AXIS,
        mochi.GridSdfResolutionMode.MEAN_AXIS,
        mochi.GridSdfResolutionMode.LARGEST_EDGE,
        mochi.GridSdfResolutionMode.SMALLEST_EDGE,
        mochi.GridSdfResolutionMode.MEAN_EDGE,
        mochi.GridSdfResolutionMode.EXPLICIT,
    ]
    delta = ui["sdf_resolution_delta"]
    min_res = ui["sdf_min_grid_resolution"]
    return mochi.GridSdfParams(
        resolution_mode=modes[ui["sdf_resolution_mode"]],
        resolution_delta=[delta, delta, delta],
        boundary_padding_dist=ui["sdf_boundary_padding_dist"],
        min_grid_resolution=[min_res, min_res, min_res],
    )


def _build_grid_wireframe(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    dims: np.ndarray,
    max_edges: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate wireframe nodes and edges for a 3D Cartesian grid.

    Each voxel edge is a separate line segment so the internal grid structure is
    visible. If the total edge count exceeds max_edges, falls back to the 12-edge
    bounding box.
    """
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    total_edges = (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1)

    if total_edges > max_edges or nx < 2 or ny < 2 or nz < 2:
        bmin, bmax = bounds_min, bounds_max
        nodes = np.array(
            [
                [bmin[0], bmin[1], bmin[2]],
                [bmax[0], bmin[1], bmin[2]],
                [bmax[0], bmax[1], bmin[2]],
                [bmin[0], bmax[1], bmin[2]],
                [bmin[0], bmin[1], bmax[2]],
                [bmax[0], bmin[1], bmax[2]],
                [bmax[0], bmax[1], bmax[2]],
                [bmin[0], bmax[1], bmax[2]],
            ],
            dtype=np.float32,
        )
        edges = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ],
            dtype=np.int32,
        )
        return nodes, edges

    # Build grid nodes using meshgrid
    xs = np.linspace(float(bounds_min[0]), float(bounds_max[0]), nx)
    ys = np.linspace(float(bounds_min[1]), float(bounds_max[1]), ny)
    zs = np.linspace(float(bounds_min[2]), float(bounds_max[2]), nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

    # Node index for grid point (i,j,k)
    idx = np.arange(nx * ny * nz, dtype=np.int32).reshape(nx, ny, nz)

    # Edges along each axis: connect adjacent nodes
    ex = np.stack([idx[:-1, :, :].ravel(), idx[1:, :, :].ravel()], axis=1)
    ey = np.stack([idx[:, :-1, :].ravel(), idx[:, 1:, :].ravel()], axis=1)
    ez = np.stack([idx[:, :, :-1].ravel(), idx[:, :, 1:].ravel()], axis=1)
    edges = np.concatenate([ex, ey, ez], axis=0).astype(np.int32)

    return nodes, edges


def _bake_sdf(state: RemeshingState, ui: dict) -> None:
    """Bake an SDF from the remeshed mesh, reconstruct surface, and measure quality."""
    import mochi  # @manual

    if not state.has_mesh or state.remeshed_vertices.shape[0] == 0:
        state.sdf_status_message = "No remeshed mesh available."
        return

    model = _build_model_data(state)
    sdf_params = _sync_ui_to_sdf_params(ui)
    state.sdf_status_message = "Baking SDF..."
    try:
        mochi.model.bake_sdf(model, params=sdf_params)
    except Exception as e:
        # Bake failed: keep prior SDF state intact so user does not lose context.
        state.sdf_status_message = f"SDF baking failed: {e}"
        return

    # Bake succeeded: clear prior SDF state before applying the new one.
    _clear_sdf_state(state)
    if model.sdf is not None:
        (
            dims,
            values,
            bounds_min,
            bounds_max,
            neg_min,
            neg_max,
            scale,
            rotation,
            translation,
        ) = _extract_sdf_arrays(model.sdf)
        _apply_sdf_data(
            state,
            dims,
            values,
            bounds_min,
            bounds_max,
            neg_min,
            neg_max,
            "Baked SDF",
            scale,
            rotation,
            translation,
        )
        state.show_grid = True
        state.show_recon = True
    else:
        state.sdf_status_message = "SDF baking produced no output."


def _save_sdf_model(state: RemeshingState, ui: dict) -> None:
    """Save the remeshed mesh with cached baked SDF to an H5 file."""
    import mochi  # @manual

    if state.output_path is None:
        state.sdf_status_message = "No output path specified."
        return

    path = _ensure_h5_extension(state.output_path)
    model = _build_model_data(state)
    if (
        state.sdf_dims is not None
        and state.sdf_values is not None
        and state.sdf_bounds_min is not None
        and state.sdf_bounds_max is not None
        and state.sdf_neg_bounds_min is not None
        and state.sdf_neg_bounds_max is not None
    ):
        sdf_kwargs: dict = {
            "dims": list(state.sdf_dims),
            "values": list(state.sdf_values),
            "bounds": mochi.Aabb(
                list(state.sdf_bounds_min), list(state.sdf_bounds_max)
            ),
            "negative_value_bounds": mochi.Aabb(
                list(state.sdf_neg_bounds_min), list(state.sdf_neg_bounds_max)
            ),
        }
        # Preserve optional parent-from-grid transforms across the round-trip.
        if state.sdf_scale is not None:
            sdf_kwargs["scale"] = list(state.sdf_scale)
        if state.sdf_rotation is not None:
            sdf_kwargs["rotation"] = mochi.Quaternion(list(state.sdf_rotation))
        if state.sdf_translation is not None:
            sdf_kwargs["translation"] = list(state.sdf_translation)
        model.sdf = mochi.GridSdfData(**sdf_kwargs)
    else:
        sdf_params = _sync_ui_to_sdf_params(ui)
        try:
            mochi.model.bake_sdf(model, params=sdf_params)
        except Exception as e:
            state.sdf_status_message = f"SDF baking failed: {e}"
            return
    try:
        mochi.model.save_to_file(model, path, mochi.FileFormat.H5)
    except Exception as e:
        state.sdf_status_message = f"Save failed: {e}"
        return
    state.status_message = f"Saved to {path}"


def _build_sdf_section(state: RemeshingState, ui: dict) -> None:  # noqa: C901
    """Draw the SDF Baking collapsing header."""
    if not psim.CollapsingHeader(
        "SDF Baking", flags=psim.ImGuiTreeNodeFlags_DefaultOpen
    ):
        return

    prev_sdf_mode = ui["sdf_resolution_mode"]
    _imgui_update(
        ui,
        "sdf_resolution_mode",
        psim.Combo,
        "Resolution Mode",
        _SDF_RESOLUTION_MODE_NAMES,
    )
    tooltip_on_hover("How voxel size is determined relative to the mesh.")
    was_explicit = prev_sdf_mode == 6
    is_explicit = ui["sdf_resolution_mode"] == 6
    if was_explicit != is_explicit:
        if is_explicit:
            ui["sdf_resolution_delta"] /= 10.0
        else:
            ui["sdf_resolution_delta"] *= 10.0
    if is_explicit:
        display_val = ui["sdf_resolution_delta"] * 1000.0
        changed, new_val = psim.InputFloat(
            "Resolution (Millimeters)", display_val, 0.0, 0.0, "%.6f"
        )
        if changed:
            ui["sdf_resolution_delta"] = new_val / 1000.0
    else:
        display_val = ui["sdf_resolution_delta"] * 100.0
        changed, new_val = psim.InputFloat(
            "Resolution (Percentage)", display_val, 0.0, 0.0, "%.6f"
        )
        if changed:
            ui["sdf_resolution_delta"] = new_val / 100.0
    tooltip_on_hover("SDF voxel size. Smaller = higher resolution, more memory.")
    boundary_display = ui["sdf_boundary_padding_dist"] * 1000.0
    changed, new_val = psim.InputFloat(
        "Boundary Padding Dist (Millimeters)", boundary_display, 0.0, 0.0, "%.6f"
    )
    if changed:
        ui["sdf_boundary_padding_dist"] = new_val / 1000.0
    tooltip_on_hover("Extra padding around the mesh bounding box for the SDF grid.")
    _imgui_update(ui, "sdf_min_grid_resolution", psim.InputInt, "Min Grid Resolution")
    tooltip_on_hover("Minimum number of voxels along each axis.")

    if state.has_mesh and psim.Button("Bake SDF"):
        _save_params_json(state.mesh_path, ui)
        _bake_sdf(state, ui)

    if state.has_sdf:
        changed, val = psim.Checkbox("Show SDF Grid", state.show_grid)
        if changed:
            state.show_grid = val
            state.needs_update_grid_view = True
        psim.SameLine()
        changed, val = psim.Checkbox("Show Reconstructed Mesh", state.show_recon)
        if changed:
            state.show_recon = val
            state.needs_update_recon_view = True

    if state.sdf_status_message:
        psim.TextWrapped(state.sdf_status_message)

    if state.recon_stats is not None:
        psim.Separator()
        psim.TextWrapped("Reconstructed Mesh (from SDF):")
        for line in _format_stats(state.recon_stats):
            psim.BulletText(line)


def _build_load_section(state: RemeshingState, ui: dict) -> None:
    """Draw the mesh file path input and Load button."""
    if psim.CollapsingHeader("Input Mesh", flags=psim.ImGuiTreeNodeFlags_DefaultOpen):
        changed, val = psim.InputText("Mesh Path", ui["mesh_path"])
        if changed:
            ui["mesh_path"] = val.strip().strip('"').strip("'")
            state.mesh_path = ui["mesh_path"]
        changed, val = psim.Checkbox(
            "Keep Original Coordinates", ui["keep_original_coordinates"]
        )
        if changed:
            ui["keep_original_coordinates"] = val
            state.keep_original_coordinates = val
        tooltip_on_hover(
            "When unchecked, applies a +90 deg Z rotation to correct\n"
            "the obj generator's -90 deg coordinate system offset."
        )
        if psim.Button("Load"):
            # Load is read-only: it must never write the params JSON. Saving
            # here overwrote the target file's .mochiMeshApp.json with the
            # current (stale) UI values before it was read back in -- because
            # the Mesh Path InputText above has already reassigned
            # state.mesh_path to the new file. Param edits are persisted by the
            # Remesh / Save / Bake actions and on app exit instead.
            state.mesh_path = ui["mesh_path"].strip().strip('"').strip("'")
            state.needs_load = True


def build_remeshing_ui(state: RemeshingState):  # noqa: C901
    """Return a builder function for the interactive remeshing UI tab."""
    p = state.params

    ui = {
        "mesh_path": state.mesh_path,
        "edge_size": p.edge_size,
        "detect_features": p.detect_features,
        "relative_to_mesh_size": p.relative_to_mesh_size,
        "alpha_wrap_relative_alpha": p.alpha_wrap_relative_alpha,
        "alpha_wrap_relative_offset": p.alpha_wrap_relative_offset,
        "smoothing_iterations": p.smoothing_iterations,
        "angle_smoothing_iterations": p.angle_smoothing_iterations,
        "sharp_feature_angle": p.sharp_feature_angle,
        "protect_constraints": p.protect_constraints,
        "relax_constraints": p.relax_constraints,
        "use_adaptive_sizing": p.use_adaptive_sizing,
        "adaptive_sizing_tolerance": p.adaptive_sizing_tolerance,
        "min_edge_size_factor": p.min_edge_size_factor,
        "max_edge_size_factor": p.max_edge_size_factor,
        "remesh_method": int(p.method.value),
        "tangential_relaxation_iterations": p.tangential_relaxation_iterations,
        "relaxation_steps_per_iteration": p.relaxation_steps_per_iteration,
        "repair_mesh": p.repair_mesh,
        "target_vertex_count": p.target_vertex_count,
        "acvd_gradation_factor": p.acvd_gradation_factor,
        "facet_angle_bound": p.facet_angle_bound,
        "facet_distance_bound": p.facet_distance_bound,
        "mesh_output_path": state.output_path or state.mesh_path or "",
        "sdf_resolution_mode": state.initial_sdf_resolution_mode,
        "sdf_resolution_delta": state.initial_sdf_resolution_delta,
        "sdf_boundary_padding_dist": state.initial_sdf_boundary_padding_dist,
        "sdf_min_grid_resolution": state.initial_sdf_min_grid_resolution,
        "keep_original_coordinates": state.keep_original_coordinates,
        "export_remesh_obj": state.export_remesh_obj,
        "edge_swapping": state.edge_swapping,
        "edge_swap_threshold": state.edge_swap_threshold,
        "edge_width_scale": state.edge_width_scale,
    }
    state._ui = ui

    def builder():
        if state._pending_params is not None:
            _apply_params_dict_to_ui(ui, state._pending_params)
            _sync_ui_to_params(ui, p)
            # Restore state-level fields that are not part of the params object.
            # In particular the output path: the UI dict carries it, but
            # save_mesh()/_save_sdf_model() read it from state.output_path, so
            # an in-app load must propagate it there (matching the CLI load via
            # _apply_params_json_to_params_and_state). Empty path leaves the
            # existing output path untouched.
            loaded_output_path = ui.get("mesh_output_path", "").strip()
            if loaded_output_path:
                state.output_path = loaded_output_path
            state.keep_original_coordinates = ui["keep_original_coordinates"]
            state.edge_swapping = ui["edge_swapping"]
            state.edge_swap_threshold = ui["edge_swap_threshold"]
            # Propagate the SDF fields too: save_mesh()/_save_sdf_model() read these from
            # state.initial_sdf_*, not from the ui dict, so an in-app load must push them back the
            # same way the CLI path does via _apply_params_json_to_params_and_state.
            for json_key, state_attr in _SDF_JSON_KEY_TO_STATE_ATTR.items():
                if json_key in ui:
                    setattr(state, state_attr, ui[json_key])
            state._pending_params = None

        _build_load_section(state, ui)

        psim.PushItemWidth(psim.GetContentRegionAvail()[0] * 0.35)
        _build_criteria_section(ui)
        _build_postprocessing_section(ui)
        _build_adaptive_section(ui)

        psim.Separator()
        if psim.Button("Remesh") and state.has_mesh:
            _sync_ui_to_params(ui, p)
            state.edge_swapping = ui["edge_swapping"]
            state.edge_swap_threshold = ui["edge_swap_threshold"]
            _save_params_json(state.mesh_path, ui)
            state.needs_remesh = True
        psim.SameLine()
        changed, val = psim.Checkbox("Export OBJ", ui["export_remesh_obj"])
        if changed:
            ui["export_remesh_obj"] = val
            state.export_remesh_obj = val
        _build_sdf_section(state, ui)
        psim.PopItemWidth()

        if state.has_mesh:
            if psim.CollapsingHeader(
                "Output", flags=psim.ImGuiTreeNodeFlags_DefaultOpen
            ):
                changed, val = psim.InputText(
                    "Output Path", ui.get("mesh_output_path", "")
                )
                if changed:
                    ui["mesh_output_path"] = val.strip().strip('"').strip("'")
                if state.has_sdf:
                    psim.TextWrapped("Will save mesh + baked SDF.")
                if psim.Button("Save"):
                    path = ui.get("mesh_output_path", "").strip().strip('"').strip("'")
                    if path:
                        state.output_path = path
                    elif state.output_path is None:
                        state.output_path = state.mesh_path
                    if state.has_sdf:
                        _save_sdf_model(state, ui)
                    else:
                        save_mesh(state)
                    _save_params_json(state.mesh_path, ui)

        if state.status_message:
            psim.TextWrapped(state.status_message)

        if psim.CollapsingHeader("Original Mesh") and state.original_stats is not None:
            for line in _format_stats(state.original_stats):
                psim.BulletText(line)

        if psim.CollapsingHeader("Remeshed Mesh") and state.remeshed_stats is not None:
            for line in _format_stats(state.remeshed_stats):
                psim.BulletText(line)

    return builder


def _build_params_dict(state: RemeshingState) -> dict:
    """Build a params dict from state suitable for _save_params_json."""
    p = state.params
    return {
        "remesh_method": int(p.method.value),
        "edge_size": p.edge_size,
        "detect_features": p.detect_features,
        "relative_to_mesh_size": p.relative_to_mesh_size,
        "alpha_wrap_relative_alpha": p.alpha_wrap_relative_alpha,
        "alpha_wrap_relative_offset": p.alpha_wrap_relative_offset,
        "smoothing_iterations": p.smoothing_iterations,
        "relaxation_steps_per_iteration": p.relaxation_steps_per_iteration,
        "tangential_relaxation_iterations": p.tangential_relaxation_iterations,
        "angle_smoothing_iterations": p.angle_smoothing_iterations,
        "sharp_feature_angle": p.sharp_feature_angle,
        "protect_constraints": p.protect_constraints,
        "relax_constraints": p.relax_constraints,
        "use_adaptive_sizing": p.use_adaptive_sizing,
        "adaptive_sizing_tolerance": p.adaptive_sizing_tolerance,
        "min_edge_size_factor": p.min_edge_size_factor,
        "max_edge_size_factor": p.max_edge_size_factor,
        "repair_mesh": p.repair_mesh,
        "target_vertex_count": p.target_vertex_count,
        "acvd_gradation_factor": p.acvd_gradation_factor,
        "facet_angle_bound": p.facet_angle_bound,
        "facet_distance_bound": p.facet_distance_bound,
        "sdf_resolution_mode": state.initial_sdf_resolution_mode,
        "sdf_resolution_delta": state.initial_sdf_resolution_delta,
        "sdf_boundary_padding_dist": state.initial_sdf_boundary_padding_dist,
        "sdf_min_grid_resolution": state.initial_sdf_min_grid_resolution,
        "mesh_output_path": state.output_path or "",
        "keep_original_coordinates": state.keep_original_coordinates,
        "edge_swapping": state.edge_swapping,
        "edge_swap_threshold": state.edge_swap_threshold,
    }


def build_settings_ui(state: RemeshingState):
    """Return a builder function for the Settings tab entries."""
    ui = {"edge_width_scale": state.edge_width_scale}

    def builder():
        psim.TextDisabled("MESH DISPLAY")
        changed, val = psim.InputFloat(
            "Edge Width Scale", ui["edge_width_scale"], 0.0, 0.0, "%.2f"
        )
        if changed:
            ui["edge_width_scale"] = max(val, 0.01)
            state.edge_width_scale = ui["edge_width_scale"]
            state.needs_update_edge_width = True

    return builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive surface remeshing with viewer visualization."
    )
    parser.add_argument(
        "mesh_path",
        nargs="?",
        default=None,
        help="Path to input mesh file (.obj, .stl, .ply, .off, etc.)",
    )
    parser.add_argument(
        "--output", "-o", help="Output path for saving the remeshed mesh"
    )
    parser.add_argument("--edge-size", type=float, help="Target edge length")
    parser.add_argument(
        "--no-detect-features",
        action="store_true",
        help="Disable sharp feature detection",
    )
    parser.add_argument(
        "--absolute-sizes",
        action="store_true",
        help="Use absolute sizes instead of relative to mesh",
    )
    parser.add_argument(
        "--smoothing-iterations",
        type=int,
        help="Isotropic remeshing iterations (0 to disable)",
    )
    parser.add_argument(
        "--angle-smoothing-iterations",
        type=int,
        help="Angle smoothing iterations (0 to disable)",
    )
    parser.add_argument(
        "--relaxation-steps-per-iteration",
        type=int,
        help="Tangential relaxation steps per remeshing iteration",
    )
    parser.add_argument(
        "--tangential-relaxation-iterations",
        type=int,
        help="Standalone tangential relaxation iterations after remeshing",
    )
    parser.add_argument(
        "--sharp-feature-angle",
        type=float,
        help="Dihedral angle for feature edges [deg]",
    )
    parser.add_argument(
        "--alpha-wrap-alpha",
        type=float,
        help="Alpha wrap alpha as multiple of edge_size (default: 1.0)",
    )
    parser.add_argument(
        "--alpha-wrap-offset",
        type=float,
        help="Alpha wrap offset as multiple of edge_size (0=auto, default: 0)",
    )
    parser.add_argument(
        "--protect-constraints",
        action="store_true",
        help="Hard-protect sharp edges during remeshing",
    )
    parser.add_argument(
        "--relax-constraints",
        action="store_true",
        help="Allow feature vertices to slide along polylines",
    )
    parser.add_argument(
        "--adaptive-sizing",
        action="store_true",
        help="Use curvature-based adaptive sizing field",
    )
    parser.add_argument(
        "--adaptive-tolerance",
        type=float,
        help="Error tolerance for adaptive sizing",
    )
    parser.add_argument(
        "--method",
        choices=["none", "alpha-wrap", "acvd", "surface-delaunay"],
        help="Remeshing method (default: alpha-wrap)",
    )
    parser.add_argument(
        "--target-vertex-count",
        type=int,
        help="ACVD target vertex count (0=auto from edge_size, default: 0)",
    )
    parser.add_argument(
        "--acvd-gradation-factor",
        type=float,
        help="ACVD gradation factor (0=uniform, default: 1.5)",
    )
    parser.add_argument(
        "--no-repair-mesh",
        action="store_true",
        help="Disable mesh repair after remeshing",
    )
    parser.add_argument(
        "--min-edge-size-factor",
        type=float,
        help="Min edge length as fraction of edge_size (default: 0.25)",
    )
    parser.add_argument(
        "--max-edge-size-factor",
        type=float,
        help="Max edge length as fraction of edge_size (default: 2.0)",
    )
    parser.add_argument(
        "--bake-sdf",
        action="store_true",
        help="Bake SDF after remeshing and include it in the output file",
    )
    parser.add_argument(
        "--sdf-resolution-mode",
        choices=[
            "largest-axis",
            "smallest-axis",
            "mean-axis",
            "largest-edge",
            "smallest-edge",
            "mean-edge",
            "explicit",
        ],
        help="SDF resolution mode (default: largest-axis)",
    )
    parser.add_argument(
        "--sdf-resolution-delta",
        type=float,
        help="SDF resolution delta (voxel size)",
    )
    parser.add_argument(
        "--sdf-boundary-padding-dist",
        type=float,
        help="SDF boundary padding distance",
    )
    parser.add_argument(
        "--sdf-min-grid-resolution",
        type=int,
        help="SDF minimum grid resolution per axis",
    )
    parser.add_argument(
        "--keep-original-coordinates",
        action="store_true",
        help="Do not apply the +90 deg Z rotation that corrects the obj generator's coordinate system",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without viewer (remesh, optionally bake SDF, save, and exit)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Remesh and bake SDF with CLI params, then open viewer for review. "
        "Overrides any saved params JSON.",
    )
    return parser.parse_args()


def _apply_args_to_params(
    args: argparse.Namespace, params: SurfaceRemeshingParams
) -> None:
    """Apply parsed CLI arguments to remeshing params."""
    _ARG_TO_PARAM = {
        "edge_size": "edge_size",
        "smoothing_iterations": "smoothing_iterations",
        "angle_smoothing_iterations": "angle_smoothing_iterations",
        "sharp_feature_angle": "sharp_feature_angle",
        "alpha_wrap_alpha": "alpha_wrap_relative_alpha",
        "alpha_wrap_offset": "alpha_wrap_relative_offset",
        "adaptive_tolerance": "adaptive_sizing_tolerance",
        "target_vertex_count": "target_vertex_count",
        "acvd_gradation_factor": "acvd_gradation_factor",
        "relaxation_steps_per_iteration": "relaxation_steps_per_iteration",
        "tangential_relaxation_iterations": "tangential_relaxation_iterations",
        "min_edge_size_factor": "min_edge_size_factor",
        "max_edge_size_factor": "max_edge_size_factor",
    }
    for arg_name, param_name in _ARG_TO_PARAM.items():
        val = getattr(args, arg_name)
        if val is not None:
            setattr(params, param_name, val)

    if args.no_detect_features:
        params.detect_features = False
    if args.absolute_sizes:
        params.relative_to_mesh_size = False
    if args.no_repair_mesh:
        params.repair_mesh = False
    for flag in ("protect_constraints", "relax_constraints"):
        if getattr(args, flag):
            setattr(params, flag, True)
    if args.adaptive_sizing:
        params.use_adaptive_sizing = True
    if hasattr(args, "method") and args.method is not None:
        _METHOD_MAP = {
            "none": RemeshMethod.NONE,
            "alpha-wrap": RemeshMethod.ALPHA_WRAP,
            "acvd": RemeshMethod.ACVD,
            "surface-delaunay": RemeshMethod.SURFACE_DELAUNAY,
        }
        params.method = _METHOD_MAP[args.method]


def _build_sdf_params_from_args(args: argparse.Namespace):
    import mochi  # @manual

    sdf_params = mochi.GridSdfParams()
    if args.sdf_resolution_mode is not None:
        _MODE_MAP = {
            "largest-axis": mochi.GridSdfResolutionMode.LARGEST_AXIS,
            "smallest-axis": mochi.GridSdfResolutionMode.SMALLEST_AXIS,
            "mean-axis": mochi.GridSdfResolutionMode.MEAN_AXIS,
            "largest-edge": mochi.GridSdfResolutionMode.LARGEST_EDGE,
            "smallest-edge": mochi.GridSdfResolutionMode.SMALLEST_EDGE,
            "mean-edge": mochi.GridSdfResolutionMode.MEAN_EDGE,
            "explicit": mochi.GridSdfResolutionMode.EXPLICIT,
        }
        sdf_params.resolution_mode = _MODE_MAP[args.sdf_resolution_mode]
    if args.sdf_resolution_delta is not None:
        d = args.sdf_resolution_delta
        sdf_params.resolution_delta = [d, d, d]
    if args.sdf_boundary_padding_dist is not None:
        sdf_params.boundary_padding_dist = args.sdf_boundary_padding_dist
    if args.sdf_min_grid_resolution is not None:
        r = args.sdf_min_grid_resolution
        sdf_params.min_grid_resolution = [r, r, r]
    return sdf_params


_SDF_MODE_NAME_TO_INDEX = {
    "largest-axis": 0,
    "smallest-axis": 1,
    "mean-axis": 2,
    "largest-edge": 3,
    "smallest-edge": 4,
    "mean-edge": 5,
    "explicit": 6,
}


def _apply_sdf_args_to_state(args: argparse.Namespace, state: RemeshingState) -> None:
    """Apply parsed CLI SDF arguments to the state's initial SDF defaults."""
    if args.sdf_resolution_mode is not None:
        state.initial_sdf_resolution_mode = _SDF_MODE_NAME_TO_INDEX[
            args.sdf_resolution_mode
        ]
    if args.sdf_resolution_delta is not None:
        state.initial_sdf_resolution_delta = args.sdf_resolution_delta
    if (
        hasattr(args, "sdf_boundary_padding_dist")
        and args.sdf_boundary_padding_dist is not None
    ):
        state.initial_sdf_boundary_padding_dist = args.sdf_boundary_padding_dist
    if (
        hasattr(args, "sdf_min_grid_resolution")
        and args.sdf_min_grid_resolution is not None
    ):
        state.initial_sdf_min_grid_resolution = args.sdf_min_grid_resolution


def _setup_viewer(state: RemeshingState):
    """Compute bounding box metrics for viewer layout."""
    bbox_size = state.original_vertices.max(axis=0) - state.original_vertices.min(
        axis=0
    )
    offset = float(bbox_size.max()) * 0.7
    edge_radius = float(np.linalg.norm(bbox_size)) * 0.001 * state.edge_width_scale
    center = (
        state.original_vertices.max(axis=0) + state.original_vertices.min(axis=0)
    ) / 2.0
    return offset, edge_radius, center, bbox_size


def _configure_mesh(mesh, offset_x: float, edge_radius: float) -> None:
    """Apply standard transform/style to a mesh renderer."""
    mesh.set_transform(make_transform([offset_x, 0, 0], [0, 0, 0]))
    mesh.set_smooth_shading(False)
    mesh.set_enable_edges(True)
    mesh.set_edge_radius(edge_radius)
    mesh.set_material(MeshRenderer.Material.CLAY)


def _load_mesh_via_mochi(
    path: str,
) -> dict | None:
    """Try loading a model with mochi's loader.

    Returns a dict with keys 'vertices', 'faces', and optionally 'sdf' (a dict with
    'dims', 'values', 'bounds_min', 'bounds_max'), or None on failure.
    """
    import mochi  # @manual

    try:
        model = mochi.model.load_from_file(path)
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to load mesh from %s: %s", path, e)
        return None
    if model.mesh is None:
        return None
    num_nodes = model.mesh.get_num_nodes()
    npe = model.mesh.nodes_per_element
    vertices = np.array(model.mesh.coordinates, dtype=np.float32).reshape(num_nodes, 3)
    faces = np.array(model.mesh.connectivity, dtype=np.int32).reshape(-1, npe)

    result: dict = {"vertices": vertices, "faces": faces}

    if model.sdf is not None:
        (
            dims,
            values,
            bounds_min,
            bounds_max,
            neg_min,
            neg_max,
            scale,
            rotation,
            translation,
        ) = _extract_sdf_arrays(model.sdf)
        result["sdf"] = {
            "dims": dims,
            "values": values,
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "neg_bounds_min": neg_min,
            "neg_bounds_max": neg_max,
            "scale": scale,
            "rotation": rotation,
            "translation": translation,
        }

    return result


def _apply_sdf_data(
    state: RemeshingState,
    dims: np.ndarray,
    values: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    neg_bounds_min: np.ndarray,
    neg_bounds_max: np.ndarray,
    source_label: str,
    scale: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> None:
    """Reconstruct isosurface from SDF data and update state."""
    state.sdf_dims = dims
    state.sdf_bounds_min = bounds_min
    state.sdf_bounds_max = bounds_max
    state.sdf_neg_bounds_min = neg_bounds_min
    state.sdf_neg_bounds_max = neg_bounds_max
    state.sdf_values = values
    state.sdf_scale = scale
    state.sdf_rotation = rotation
    state.sdf_translation = translation

    try:
        state.recon_vertices, state.recon_faces = reconstruct_surface_from_sdf(
            dims, values, bounds_min, bounds_max
        )
        state.recon_stats = compute_mesh_statistics(
            state.recon_vertices,
            state.recon_faces,
            state.remeshed_vertices,
            state.remeshed_faces,
        )
    except Exception as e:
        state.sdf_status_message = f"SDF reconstruction failed: {e}"
        return

    state.has_sdf = True
    # NOTE: Do not auto-enable show_grid / show_recon here. Toggling them on
    # automatically clobbers user preferences set during the prior session.
    # Users can re-enable via the UI checkboxes after baking/loading.
    state.needs_update_recon_view = True
    state.needs_update_grid_view = True

    hausdorff_msg = ""
    if state.recon_stats is not None and state.recon_stats.hausdorff_distance >= 0:
        hausdorff_msg = f" | Hausdorff: {state.recon_stats.hausdorff_distance:.6f}"
    state.sdf_status_message = (
        f"{source_label} | Grid: {dims[0]}x{dims[1]}x{dims[2]}{hausdorff_msg}"
    )


def load_mesh_from_file(path: str, state: RemeshingState) -> bool:
    """Load a mesh file into the state. Returns True on success."""
    if not path or not path.strip():
        state.status_message = "No file path specified."
        return False

    result = _load_mesh_via_mochi(path)
    if result is None:
        state.status_message = f"Failed to load mesh from {path}"
        return False

    vertices, faces = result["vertices"], result["faces"]
    if not state.keep_original_coordinates:
        rotated = np.empty_like(vertices)
        rotated[:, 0] = -vertices[:, 1]
        rotated[:, 1] = vertices[:, 0]
        rotated[:, 2] = vertices[:, 2]
        vertices = rotated
    state.original_vertices = vertices
    state.original_faces = faces
    num_verts = vertices.shape[0]
    num_faces = faces.shape[0]
    state.mesh_path = path
    state.has_mesh = True

    # For .h5 files with SDF, the loaded mesh is already the final mesh.
    # Set remeshed = original so SDF section works immediately.
    has_loaded_sdf = "sdf" in result
    if has_loaded_sdf:
        state.remeshed_vertices = vertices.copy()
        state.remeshed_faces = faces.copy()
        state.original_stats = compute_mesh_statistics(vertices, faces)
        state.remeshed_stats = state.original_stats
        sdf = result["sdf"]
        _apply_sdf_data(
            state,
            sdf["dims"],
            sdf["values"],
            sdf["bounds_min"],
            sdf["bounds_max"],
            sdf["neg_bounds_min"],
            sdf["neg_bounds_max"],
            "Loaded SDF",
            sdf.get("scale"),
            sdf.get("rotation"),
            sdf.get("translation"),
        )
        state.show_recon = True
        state.status_message = (
            f"Loaded {path} : {num_verts} verts, {num_faces} faces (with SDF)"
        )
    else:
        _clear_sdf_state(state)
        state.status_message = f"Loaded {path} - {num_verts} verts, {num_faces} faces"

    return True


def _update_viewer_meshes(
    viewer: "Viewer", state: RemeshingState, has_reconstruction: bool = False
) -> tuple[float, float]:
    """Add/replace Original and Remeshed meshes in the viewer and fit the camera."""
    offset, edge_radius, center, bbox_size = _setup_viewer(state)

    original = viewer.add_mesh(
        "Original", state.original_vertices, state.original_faces
    )
    _configure_mesh(original, -offset, edge_radius)

    remeshed = viewer.add_mesh(
        "Remeshed", state.remeshed_vertices, state.remeshed_faces
    )
    _configure_mesh(remeshed, offset, edge_radius)

    rightmost_x = offset * 3 if has_reconstruction else offset
    scene_center_x = float(center[0]) + (-offset + rightmost_x) / 2
    scene_span = rightmost_x - (-offset) + float(bbox_size.max())
    cam_dist = max(float(bbox_size.max()) * 2.5, scene_span * 1.5)

    viewer.set_camera_view(
        look_from=[
            scene_center_x,
            float(center[1]) - cam_dist,
            float(center[2]) + cam_dist * 0.6,
        ],
        look_at=[scene_center_x, float(center[1]), float(center[2])],
    )

    # Set length scale proportional to scene span for comfortable zoom speed.
    import polyscope as ps  # @manual

    ps.set_length_scale(scene_span * 1.5)

    return offset, edge_radius


def main() -> None:
    import mochi  # @manual

    args = parse_args()

    if args.preview:
        args.bake_sdf = True

    state = RemeshingState()
    state.output_path = args.output
    state.keep_original_coordinates = args.keep_original_coordinates
    _apply_args_to_params(args, state.params)
    _apply_sdf_args_to_state(args, state)

    offset = 0.0
    edge_radius = 0.001

    if args.mesh_path is not None:
        if not load_mesh_from_file(args.mesh_path, state):
            raise ValueError(state.status_message)
        if not state.has_sdf:
            if args.headless or args.preview:
                try:
                    run_remesh(state)
                except Exception as e:
                    if args.headless:
                        raise
                    state.status_message = f"Remeshing failed: {e}"
                    state.remeshed_vertices = state.original_vertices.copy()
                    state.remeshed_faces = state.original_faces.copy()
                if args.bake_sdf:
                    model = _build_model_data(state)
                    sdf_params = _build_sdf_params_from_args(args)
                    mochi.model.bake_sdf(model, params=sdf_params)
                    if model.sdf is not None:
                        (
                            dims,
                            values,
                            bounds_min,
                            bounds_max,
                            neg_min,
                            neg_max,
                            scale,
                            rotation,
                            translation,
                        ) = _extract_sdf_arrays(model.sdf)
                        _apply_sdf_data(
                            state,
                            dims,
                            values,
                            bounds_min,
                            bounds_max,
                            neg_min,
                            neg_max,
                            "Baked SDF",
                            scale,
                            rotation,
                            translation,
                        )
                        state.show_recon = True
            else:
                loaded_params = _load_params_json(state.mesh_path)
                if loaded_params is not None:
                    _apply_params_json_to_params_and_state(
                        loaded_params, state.params, state
                    )
                state.remeshed_vertices = state.original_vertices.copy()
                state.remeshed_faces = state.original_faces.copy()
                state.original_stats = compute_mesh_statistics(
                    state.original_vertices, state.original_faces
                )
        if state.output_path is None:
            state.output_path = args.mesh_path

    # Headless mode: remesh, optionally bake SDF, save, and exit.
    if args.headless:
        if not state.has_mesh:
            raise ValueError("--headless requires a mesh_path argument.")
        if state.output_path is None:
            raise ValueError("--headless requires --output or a mesh_path.")

        path = _ensure_h5_extension(state.output_path)
        model = _build_model_data(state)

        # Attach loaded SDF data to the model (if present and not re-baking)
        if not args.bake_sdf and state.has_sdf and state.sdf_dims is not None:
            sdf_kwargs: dict = {
                "dims": list(state.sdf_dims),
                "values": list(state.sdf_values),
                "bounds": mochi.Aabb(
                    list(state.sdf_bounds_min), list(state.sdf_bounds_max)
                ),
                "negative_value_bounds": mochi.Aabb(
                    list(state.sdf_neg_bounds_min), list(state.sdf_neg_bounds_max)
                ),
            }
            # Preserve optional parent-from-grid transforms across the round-trip.
            if state.sdf_scale is not None:
                sdf_kwargs["scale"] = list(state.sdf_scale)
            if state.sdf_rotation is not None:
                sdf_kwargs["rotation"] = mochi.Quaternion(list(state.sdf_rotation))
            if state.sdf_translation is not None:
                sdf_kwargs["translation"] = list(state.sdf_translation)
            model.sdf = mochi.GridSdfData(**sdf_kwargs)

        if args.bake_sdf:
            sdf_params = _build_sdf_params_from_args(args)
            mochi.model.bake_sdf(model, params=sdf_params)
            if model.sdf is not None:
                (
                    dims,
                    values,
                    bounds_min,
                    bounds_max,
                    neg_min,
                    neg_max,
                    scale,
                    rotation,
                    translation,
                ) = _extract_sdf_arrays(model.sdf)
                _apply_sdf_data(
                    state,
                    dims,
                    values,
                    bounds_min,
                    bounds_max,
                    neg_min,
                    neg_max,
                    "Baked SDF",
                    scale,
                    rotation,
                    translation,
                )
                if state.has_sdf and state.recon_stats is not None:
                    print(
                        f"SDF Grid: {dims[0]}x{dims[1]}x{dims[2]} | "
                        f"Recon: {state.recon_stats.num_vertices} verts, "
                        f"{state.recon_stats.num_faces} faces | "
                        f"Watertight: {state.recon_stats.is_closed} | "
                        f"Hausdorff: {state.recon_stats.hausdorff_distance:.6f}"
                    )
                else:
                    print(state.sdf_status_message)

        mochi.model.save_to_file(model, path, mochi.FileFormat.H5)
        print(
            f"Saved {state.remeshed_stats.num_vertices} verts, "
            f"{state.remeshed_stats.num_faces} faces to {path}"
        )
        _save_params_json(state.mesh_path, _build_params_dict(state))
        return

    config = ViewerCfg()
    with Viewer(config) as viewer:
        if state.has_mesh:
            offset, edge_radius = _update_viewer_meshes(
                viewer, state, has_reconstruction=state.has_sdf
            )
        else:
            viewer.set_camera_view(
                look_from=[0.0, -5.0, 3.0],
                look_at=[0.0, 0.0, 0.0],
            )

        viewer.add_ui_tab("Remeshing", build_remeshing_ui(state))
        viewer.add_settings_builder(build_settings_ui(state))
        viewer.set_active_tab("Remeshing")

        while not viewer.user_requested_close():
            if state.needs_load:
                state.needs_load = False
                if load_mesh_from_file(state.mesh_path, state):
                    if not state.has_sdf:
                        _clear_sdf_state(state)
                        state.remeshed_vertices = state.original_vertices.copy()
                        state.remeshed_faces = state.original_faces.copy()
                        state.remeshed_stats = None
                        try:
                            state.original_stats = compute_mesh_statistics(
                                state.original_vertices, state.original_faces
                            )
                        except Exception:
                            pass
                    state._pending_params = _load_params_json(state.mesh_path)
                    offset, edge_radius = _update_viewer_meshes(
                        viewer, state, has_reconstruction=state.has_sdf
                    )
                    viewer.remove_mesh("Reconstructed")
                    viewer.remove_curve_network("SDF Grid")

            if state.needs_remesh:
                state.needs_remesh = False
                _clear_sdf_state(state)
                try:
                    run_remesh(state)
                except Exception as e:
                    state.status_message = f"Remeshing failed: {e}"
                else:
                    viewer.add_mesh(
                        "Remeshed", state.remeshed_vertices, state.remeshed_faces
                    )
                    remeshed = viewer.get_mesh("Remeshed")
                    _configure_mesh(remeshed, offset, edge_radius)
                    viewer.remove_mesh("Reconstructed")
                    viewer.remove_curve_network("SDF Grid")

            if state.needs_update_grid_view:
                state.needs_update_grid_view = False
                if (
                    state.show_grid
                    and state.sdf_dims is not None
                    and state.sdf_bounds_min is not None
                    and state.sdf_bounds_max is not None
                ):
                    nodes, edges = _build_grid_wireframe(
                        state.sdf_bounds_min, state.sdf_bounds_max, state.sdf_dims
                    )
                    # Offset grid nodes to align with the Remeshed mesh position
                    nodes[:, 0] += offset
                    grid_radius = edge_radius * 0.3
                    viewer.add_curve_network(
                        "SDF Grid",
                        nodes,
                        edges,
                        radius=grid_radius,
                        color=[0.4, 0.4, 0.8],
                        transparency=0.6,
                    )
                else:
                    viewer.remove_curve_network("SDF Grid")

            if state.needs_update_edge_width:
                state.needs_update_edge_width = False
                if state.has_mesh:
                    _, edge_radius, _, _ = _setup_viewer(state)
                    original = viewer.get_mesh("Original")
                    if original is not None:
                        original.set_edge_radius(edge_radius)
                    remeshed = viewer.get_mesh("Remeshed")
                    if remeshed is not None:
                        remeshed.set_edge_radius(edge_radius)
                    recon = viewer.get_mesh("Reconstructed")
                    if recon is not None:
                        recon.set_edge_radius(edge_radius)
                    if state.show_grid:
                        state.needs_update_grid_view = True

            if state.needs_update_recon_view:
                state.needs_update_recon_view = False
                if state.show_recon and state.recon_vertices.shape[0] > 0:
                    recon = viewer.add_mesh(
                        "Reconstructed",
                        state.recon_vertices,
                        state.recon_faces,
                    )
                    _configure_mesh(recon, offset * 3, edge_radius)
                    recon.set_front_face_color([0.8, 0.4, 0.2])
                else:
                    viewer.remove_mesh("Reconstructed")

            viewer.render()

        if state.has_mesh and state.mesh_path and hasattr(state, "_ui"):
            _save_params_json(state.mesh_path, state._ui)


if __name__ == "__main__":
    main()
