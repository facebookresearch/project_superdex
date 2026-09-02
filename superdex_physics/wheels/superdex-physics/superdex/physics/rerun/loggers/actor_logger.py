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

import logging
import math
from collections import defaultdict

import numpy as np
import numpy.typing as npt
import rerun as rr
import superdex.physics as sdp
from superdex.physics import Actor, ActorType
from superdex.physics.rerun.loggers.base import Logger
from superdex.physics.utils.coordinate_systems import CoordinateTransform
from superdex.physics.utils.decorators import override_from
from superdex.physics.utils.transformations import (
    apply_linear_map,
    make_transform,
    transformrt_to_numpy,
)

logger = logging.getLogger(__name__)


########################################################################################


def _compute_face_normals_array(
    vertices: npt.NDArray[np.float32],
    faces: npt.NDArray[np.int32],
) -> npt.NDArray[np.float32]:
    """Compute normalized per-face normals. Returns (F, 3) float32 array."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.maximum(norms, 1e-12)
    return fn.astype(np.float32)


def _build_edge_adjacency(
    faces: npt.NDArray[np.int32],
) -> dict[tuple[int, int], list[int]]:
    """Build a map from each edge (sorted vertex pair) to adjacent face indices."""
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in range(len(faces)):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            va, vb = int(faces[fi, a]), int(faces[fi, b])
            key = (min(va, vb), max(va, vb))
            edge_to_faces[key].append(fi)
    return edge_to_faces


def _build_smoothing_groups(
    num_faces: int,
    face_normals: npt.NDArray[np.float32],
    edge_adj: dict[tuple[int, int], list[int]],
    cos_threshold: float,
) -> npt.NDArray[np.int32]:
    """Flood-fill faces into smoothing groups connected by smooth edges.

    An edge is smooth when the dot product of its two adjacent face normals
    is >= cos_threshold. Boundary and non-manifold edges are always sharp.

    Returns:
        face_to_group: (F,) int32 array mapping each face to its group id.
    """
    smooth_adj: list[list[int]] = [[] for _ in range(num_faces)]
    for face_list in edge_adj.values():
        if len(face_list) != 2:
            continue
        fa, fb = face_list
        if np.dot(face_normals[fa], face_normals[fb]) >= cos_threshold:
            smooth_adj[fa].append(fb)
            smooth_adj[fb].append(fa)

    face_to_group = np.full(num_faces, -1, dtype=np.int32)
    group_id = 0
    for start in range(num_faces):
        if face_to_group[start] >= 0:
            continue
        stack = [start]
        face_to_group[start] = group_id
        while stack:
            fi = stack.pop()
            for nb in smooth_adj[fi]:
                if face_to_group[nb] < 0:
                    face_to_group[nb] = group_id
                    stack.append(nb)
        group_id += 1
    return face_to_group


def compute_face_normals(
    vertices: npt.NDArray[np.float32],
    faces: npt.NDArray[np.int32],
    angle_threshold_deg: float = 30.0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Compute auto-smooth normals with angle-based edge splitting.

    Edges where the dihedral angle between adjacent faces exceeds the
    threshold are treated as sharp (vertices are split). Smooth edges share
    angle-weighted averaged normals. This gives crisp hard edges on boxes
    while preserving smooth shading on curved surfaces.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) triangle indices into vertices.
        angle_threshold_deg: Dihedral angle threshold in degrees. Edges with
            angles above this are sharp. Default 30 (matches Blender).

    Returns:
        Tuple of (new_vertices, new_faces, new_normals).
    """
    face_normals = _compute_face_normals_array(vertices, faces)
    edge_adj = _build_edge_adjacency(faces)
    cos_threshold = math.cos(math.radians(angle_threshold_deg))
    face_to_group = _build_smoothing_groups(
        len(faces), face_normals, edge_adj, cos_threshold
    )

    # For each original vertex, collect which smoothing groups reference it
    # and which faces in each group use it.  Store the corner index (0/1/2)
    # so we can look up the angle weight.
    vert_group_faces: defaultdict[int, defaultdict[int, list[tuple[int, int]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for fi in range(len(faces)):
        g = int(face_to_group[fi])
        for corner in range(3):
            vi = int(faces[fi, corner])
            vert_group_faces[vi][g].append((fi, corner))

    # Pre-compute the angle at each corner of each face for angle weighting.
    # corner_angles[fi, c] = angle at corner c of face fi.
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edges = np.stack([v1 - v0, v2 - v1, v0 - v2], axis=1)  # (F, 3, 3)
    corner_angles = np.zeros((len(faces), 3), dtype=np.float32)
    for c in range(3):
        e_a = -edges[:, (c + 2) % 3]  # edge into this corner
        e_b = edges[:, c]  # edge out of this corner
        len_a = np.linalg.norm(e_a, axis=1, keepdims=True)
        len_b = np.linalg.norm(e_b, axis=1, keepdims=True)
        cos_ang = np.sum(e_a * e_b, axis=1) / np.maximum(
            (len_a * len_b).squeeze(), 1e-12
        )
        corner_angles[:, c] = np.arccos(np.clip(cos_ang, -1.0, 1.0))

    # Build output arrays: one vertex per (original_vertex, smoothing_group).
    new_verts_list: list[npt.NDArray[np.float32]] = []
    new_normals_list: list[npt.NDArray[np.float32]] = []
    new_faces = faces.copy()
    new_idx = 0

    # Map (original_vertex, group) -> new vertex index
    split_map: dict[tuple[int, int], int] = {}

    for vi in sorted(vert_group_faces.keys()):
        for g, fi_corner_list in vert_group_faces[vi].items():
            split_map[(vi, g)] = new_idx
            new_verts_list.append(vertices[vi])
            avg = np.zeros(3, dtype=np.float64)
            for fi, corner in fi_corner_list:
                avg += corner_angles[fi, corner] * face_normals[fi]
            length = np.linalg.norm(avg)
            avg = avg / max(length, 1e-12)
            new_normals_list.append(avg.astype(np.float32))
            new_idx += 1

    # Remap face indices
    for fi in range(len(faces)):
        g = int(face_to_group[fi])
        for corner in range(3):
            vi = int(faces[fi, corner])
            new_faces[fi, corner] = split_map[(vi, g)]

    out_verts = np.array(new_verts_list, dtype=np.float32)
    out_normals = np.array(new_normals_list, dtype=np.float32)
    return out_verts, new_faces, out_normals


_compute_face_normals = compute_face_normals


def extract_actor_mesh(
    actor: Actor,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Extract vertices, faces, and normals from a Mochi actor.

    Args:
        actor: A Mochi actor with a surface mesh.

    Returns:
        Tuple of (vertices, faces, normals) where vertices is (V, 3),
        faces is (F, 3), and normals is (V, 3), all in local space.
    """
    actor.register_query_and_compute(sdp.QueryType.SURFACE_NODE_POSITIONS)
    vertices = np.asarray(
        actor.get_surface_mesh_node_positions_local(), dtype=np.float32
    ).reshape(-1, 3)
    faces = np.asarray(actor.get_surface_mesh().connectivity, dtype=np.int32).reshape(
        -1, 3
    )
    actor.register_query_and_compute(sdp.QueryType.SURFACE_NODE_NORMALS)
    normals = np.asarray(
        actor.get_surface_mesh_node_normals_local(), dtype=np.float32
    ).reshape(-1, 3)
    return vertices, faces, normals


class ActorLogger(Logger):
    """
    Logger for Mochi actors to rerun.

    This class handles logging actor mesh geometry and transforms to rerun.
    For rigid actors, it logs the initial mesh and updates the transform.
    For deformable actors (soft), it updates the vertex positions each frame.
    """

    ####################################################################################
    # Class Variables
    ####################################################################################

    # Color palette for actors (matching polyscope-style distinct colors).
    # These are visually distinguishable colors for different actors.
    _COLOR_PALETTE: list[tuple[float, float, float]] = [
        (0.11, 0.388, 0.894),  # Blue
        (1.0, 0.5, 0.0),  # Orange
        (0.2, 0.7, 0.2),  # Green
        (0.8, 0.2, 0.2),  # Red
        (0.6, 0.4, 0.8),  # Purple
        (0.4, 0.8, 0.8),  # Cyan
        (0.8, 0.8, 0.2),  # Yellow
        (0.8, 0.4, 0.6),  # Pink
        (0.4, 0.6, 0.4),  # Sage
        (0.6, 0.6, 0.8),  # Lavender
    ]
    _color_index: int = 0

    ####################################################################################
    # Members
    ####################################################################################

    _actor: Actor
    _mesh_logged: bool
    _vertices_cache: npt.NDArray[np.float32] | None
    _faces_cache: npt.NDArray[np.int32] | None
    _color: tuple[float, float, float]
    _visual_mesh: dict[str, npt.NDArray] | None
    _mesh_assets_dir: str | None
    _warned_no_root_transform: bool

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(
        self,
        actor: Actor,
        coordinate_transform: CoordinateTransform,
        entity_path_prefix: str = "world/actors",
        mesh_assets_dir: str | None = None,
    ):
        """Initialize the actor logger.

        Args:
            actor: The Mochi actor to log.
            coordinate_transform: Converter for coordinate system transformations.
            entity_path_prefix: Prefix for the entity path in the rerun hierarchy.
            mesh_assets_dir: Optional directory with visual mesh files (DAE/GLB).
                When provided, loads textured visual meshes instead of physics meshes.
        """
        assert not actor.get_surface_mesh().is_empty()
        self._actor = actor
        self._mesh_logged = False
        self._vertices_cache = None
        self._faces_cache = None
        self._mesh_assets_dir = mesh_assets_dir
        self._visual_mesh = None
        self._warned_no_root_transform = False

        # Assign a color from the palette (cycling through colors).
        self._color = ActorLogger._COLOR_PALETTE[
            ActorLogger._color_index % len(ActorLogger._COLOR_PALETTE)
        ]
        ActorLogger._color_index += 1

        # Try to load a visual mesh if mesh_assets_dir is configured.
        # Only for rigid actors — deformable actors update vertex positions
        # each frame, and the visual mesh has different topology.
        if mesh_assets_dir is not None and actor.get_type() != ActorType.SOFT:
            self._visual_mesh = self._try_load_visual_mesh()

        unique_name = f"{actor.get_name()}_h{actor.get_handle().value}"
        super().__init__(
            entity_path=f"{entity_path_prefix}/{unique_name}",
            coordinate_transform=coordinate_transform,
        )

    ####################################################################################
    # Logging
    ####################################################################################

    @override_from(Logger)
    def log(self, static: bool = False) -> None:
        """Log the actor mesh and transform to rerun.

        For rigid actors, the mesh geometry is logged once as static (timeless)
        data since it doesn't change - only the transform is updated each frame.
        For deformable actors (soft), the mesh is re-logged each frame with
        updated vertex positions.

        When a visual mesh is available (loaded from DAE/GLB), it replaces the
        physics mesh for rendering. Visual meshes include per-material vertex
        colors extracted from the source files.

        Args:
            static: If True, log as static (timeless) data.
        """
        # Get the change of basis matrix.
        source_to_target = self._coordinate_transform.source_to_target

        # Get current vertex positions in local space.
        vertices_local = self._get_actor_coordinates()

        is_rigid = self._actor.get_type() != ActorType.SOFT

        if not self._mesh_logged:
            if self._visual_mesh is not None:
                self._log_visual_mesh(source_to_target, static or is_rigid)
            else:
                self._log_physics_mesh(
                    source_to_target, vertices_local, is_rigid, static
                )
            self._mesh_logged = True

            if is_rigid:
                transform = self._get_actor_transform()
                if transform is not None:
                    transform_target = source_to_target @ transform
                    rr.log(
                        self._entity_path,
                        rr.Transform3D(
                            translation=transform_target[:3, 3],
                            mat3x3=transform_target[:3, :3],
                        ),
                    )
        else:
            if not is_rigid:
                # Deformable actors: re-log vertex positions each frame.
                # Visual meshes are never loaded for deformable actors.
                vertices_target = apply_linear_map(
                    source_to_target[:3, :3], vertices_local
                )
                normals_local = self._get_actor_normals()
                normals_target = apply_linear_map(
                    source_to_target[:3, :3], normals_local
                )
                rr.log(
                    self._entity_path,
                    rr.Mesh3D(
                        vertex_positions=vertices_target,
                        triangle_indices=self._faces_cache,
                        vertex_normals=normals_target,
                        albedo_factor=self._color,
                    ),
                )
            else:
                transform = self._get_actor_transform()
                if transform is not None:
                    transform_target = source_to_target @ transform
                    rr.log(
                        self._entity_path,
                        rr.Transform3D(
                            translation=transform_target[:3, 3],
                            mat3x3=transform_target[:3, :3],
                        ),
                    )

    def _log_visual_mesh(
        self,
        source_to_target: npt.NDArray,
        log_static: bool,
    ) -> None:
        """Log the visual mesh (from DAE/GLB) with per-material vertex colors."""
        vm = self._visual_mesh
        assert vm is not None

        vertices = apply_linear_map(source_to_target[:3, :3], vm["vertices"])
        normals = apply_linear_map(source_to_target[:3, :3], vm["vertex_normals"])
        faces = vm["faces"]

        self._faces_cache = faces

        rr.log(
            self._entity_path,
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
                vertex_normals=normals,
                vertex_colors=vm["vertex_colors"],
            ),
            static=log_static,
        )

    def _log_physics_mesh(
        self,
        source_to_target: npt.NDArray,
        vertices_local: npt.NDArray[np.float32],
        is_rigid: bool,
        static: bool,
    ) -> None:
        """Log the physics mesh (from Mochi actor) with palette color."""
        faces = self._get_actor_faces()
        vertices_target = apply_linear_map(source_to_target[:3, :3], vertices_local)

        if is_rigid:
            vertices_target, faces, normals_target = _compute_face_normals(
                vertices_target, faces
            )
        else:
            normals_local = self._get_actor_normals()
            normals_target = apply_linear_map(source_to_target[:3, :3], normals_local)

        self._faces_cache = faces
        log_static = static or is_rigid

        rr.log(
            self._entity_path,
            rr.Mesh3D(
                vertex_positions=vertices_target,
                triangle_indices=faces,
                vertex_normals=normals_target,
                albedo_factor=self._color,
            ),
            static=log_static,
        )

    @override_from(Logger)
    def clear(self) -> None:
        """Clear/remove this entity from rerun."""
        rr.log(self._entity_path, rr.Clear(recursive=True))
        self._mesh_logged = False
        self._vertices_cache = None
        self._faces_cache = None

    ####################################################################################
    # Actor geometry and topology
    ####################################################################################

    def _get_actor_transform(self) -> npt.NDArray[float] | None:
        """Gets the rigid actor's transform."""
        if not self._actor.has_root_transform():
            # No world pose to log; skip rather than silently logging identity
            # (which would plant the mesh at the world origin). Warn once so the
            # gap is visible — it usually means the logger was pointed at the
            # wrong scene (e.g. a parallel scene that never received poses).
            if not self._warned_no_root_transform:
                logger.warning(
                    "ActorLogger: actor %r has a mesh but no root transform; "
                    "skipping its transform (mesh will not be positioned).",
                    self._actor.get_name(),
                )
                self._warned_no_root_transform = True
            return None
        pos, rotvec = transformrt_to_numpy(self._actor.get_root_transform())
        return make_transform(pos, rotvec)

    def _get_actor_coordinates(self) -> npt.NDArray[np.float32]:
        """Gets the Actor's surface mesh coordinates in local space."""
        self._actor.register_query_and_compute(sdp.QueryType.SURFACE_NODE_POSITIONS)
        return np.asarray(
            self._actor.get_surface_mesh_node_positions_local(), dtype=np.float32
        ).reshape(-1, 3)

    def _get_actor_faces(self) -> npt.NDArray[np.int32]:
        """Gets the Actor's surface mesh connectivity."""
        return np.asarray(
            self._actor.get_surface_mesh().connectivity, dtype=np.int32
        ).reshape(-1, 3)

    def _get_actor_normals(self) -> npt.NDArray[np.float32]:
        """Gets the Actor's surface mesh node normals in local space."""
        self._actor.register_query_and_compute(sdp.QueryType.SURFACE_NODE_NORMALS)
        return np.asarray(
            self._actor.get_surface_mesh_node_normals_local(), dtype=np.float32
        ).reshape(-1, 3)

    ####################################################################################
    # Visual mesh loading
    ####################################################################################

    def _try_load_visual_mesh(self) -> dict[str, npt.NDArray] | None:
        """Try to load a visual mesh file for this actor.

        Uses the actor's name to determine the bone name, then resolves
        it to a DAE/GLB file in the mesh_assets_dir.
        """
        mesh_assets_dir = self._mesh_assets_dir
        if mesh_assets_dir is None:
            return None

        from superdex.physics.rerun.mesh_assets import (
            extract_bone_name,
            load_textured_mesh,
            resolve_visual_mesh,
        )

        actor_name = self._actor.get_name()
        bone_name = extract_bone_name(actor_name)
        if bone_name is None:
            return None

        mesh_path = resolve_visual_mesh(bone_name, mesh_assets_dir)
        if mesh_path is None:
            return None

        logger.info("Loading visual mesh for %s: %s", bone_name, mesh_path)
        try:
            return load_textured_mesh(mesh_path)
        except Exception:
            logger.warning(
                "Visual mesh load failed for %s (%s); falling back to physics mesh",
                bone_name,
                mesh_path,
                exc_info=True,
            )
            return None

    ####################################################################################
    # Actor properties
    ####################################################################################

    def get_actor(self) -> Actor:
        """Returns the actor."""
        return self._actor
