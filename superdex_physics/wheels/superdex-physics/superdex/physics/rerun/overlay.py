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
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import cast, NamedTuple

import numpy as np
import numpy.typing as npt
import rerun as rr
import superdex.physics as sdp
from scipy.spatial.transform import Rotation
from superdex.physics import Actor, ActorType, Scene
from superdex.physics.rerun.loggers.actor_logger import (
    compute_face_normals,
    extract_actor_mesh,
)
from superdex.physics.utils.configclasses import configclass
from superdex.physics.utils.coordinate_systems import CoordinateTransform
from superdex.physics.utils.transformations import (
    apply_linear_map,
    make_transform,
    transformrt_to_numpy,
)

logger = logging.getLogger(__name__)

########################################################################################


class _RigidActor(NamedTuple):
    entity_path: str
    actor: Actor


@dataclass
class _ArticulatedOverlay:
    actor: Actor
    root_entity_path: str
    # Per-link entity paths. Links upstream of the anchor are None.
    entity_paths: list[str | None] = field(default_factory=list)
    # Per-link static mount transforms (inverse, for update_from_scene).
    mount_inv: list[np.ndarray | None] = field(default_factory=list)
    # Per-link geometry-frame joint→child-link offset (``jointFromChildLink``),
    # composed after the joint rotation in set_articulated_pose so the leaf
    # node lands on the link geometry frame (where its mesh is authored).
    child_offset: list[np.ndarray | None] = field(default_factory=list)
    # Parent indices for the subtree links.
    parents: list[int] = field(default_factory=list)
    # Anchor link index; its descendants are driven by joint angles, while the
    # anchor itself is positioned via the per-frame ``anchor_transform`` arg.
    anchor_index: int = 0


@configclass
class OverlayCfg:
    """Configuration for a ghost/overlay visualization."""

    name: str = "overlay"
    """Used in entity paths (e.g., 'observed', 'ghost')."""

    color: tuple[float, ...] = (1.0, 1.0, 1.0, 0.5)
    """RGBA in 0.0–1.0; alpha optional."""

    include_actors: list[str] | None = None
    """fnmatch patterns to include. Cosmetic for articulated links."""

    exclude_actors: list[str] | None = None
    """fnmatch patterns to exclude. Cosmetic for articulated links."""

    flat_shading: bool = True
    """Per-face normals (crisp) vs smooth vertex normals."""

    anchors: dict[str, str] = field(default_factory=dict)
    """Per-articulated-actor anchor link name. With an entry, the entity tree
    starts at that link; everything upstream is omitted. Supply the anchor's
    world pose per frame via ``set_articulated_pose(anchor_transform=...)``.
    Example: ``{"Robot": "Robot/fr3_link8"}``."""


########################################################################################


def _transformrt_to_4x4(tf: sdp.TransformRT) -> np.ndarray:
    pos, rotvec = transformrt_to_numpy(tf)
    return make_transform(pos, rotvec).astype(np.float64)


def _log_transform(path: str, tf: np.ndarray, *, static: bool = False) -> None:
    rr.log(
        path,
        rr.Transform3D(translation=tf[:3, 3], mat3x3=tf[:3, :3]),
        static=static,
    )


def rotation_to_matrix(
    *,
    quat: npt.ArrayLike | None = None,
    rotvec: npt.ArrayLike | None = None,
    rot6d: npt.ArrayLike | None = None,
    mat: npt.ArrayLike | None = None,
) -> np.ndarray:
    """Resolve one rotation representation (xyzw quat / rotvec / 6D / 3×3 mat) to 3×3."""
    provided = sum(x is not None for x in (quat, rotvec, rot6d, mat))
    if provided != 1:
        raise ValueError(
            f"Exactly one of quat/rotvec/rot6d/mat must be provided (got {provided})."
        )
    if quat is not None:
        return Rotation.from_quat(np.asarray(quat, dtype=np.float64)).as_matrix()
    if rotvec is not None:
        return Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    if rot6d is not None:
        r6 = np.asarray(rot6d, dtype=np.float64).reshape(6)
        r1 = r6[:3] / (np.linalg.norm(r6[:3]) + 1e-12)
        r2 = r6[3:6] - np.dot(r6[3:6], r1) * r1
        r2 = r2 / (np.linalg.norm(r2) + 1e-12)
        return np.stack([r1, r2, np.cross(r1, r2)], axis=0)
    return np.asarray(mat, dtype=np.float64).reshape(3, 3)


def pose_to_transform(
    pos: npt.ArrayLike,
    *,
    quat: npt.ArrayLike | None = None,
    rotvec: npt.ArrayLike | None = None,
    rot6d: npt.ArrayLike | None = None,
    mat: npt.ArrayLike | None = None,
) -> np.ndarray:
    """Build a 4×4 transform from a position + rotation (any supported rep).

    Public helper for apps that need to compose into a non-world frame, e.g.
    ``world_from_actor @ pose_to_transform(ee_pos_base, rot6d=ee_rot6d_base)``.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation_to_matrix(quat=quat, rotvec=rotvec, rot6d=rot6d, mat=mat)
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


########################################################################################


class OverlayLogger:
    """Logs ghost/overlay meshes and per-frame transforms to Rerun.

    Articulated actors are logged as a hierarchical entity tree rooted at an
    *anchor link* (the base by default, or any mid-chain link via
    ``OverlayCfg.anchors``). Joint rotations are logged per frame; Rerun's
    transform propagation handles FK. Mid-chain anchoring is how to overlay
    just a hand positioned by EE pose data — no arm joints, no flange-offset
    composition required.

    Rigid actors use flat entity paths with world-space transforms.

    All static meshes and transforms are logged during construction. Created
    via ``RerunLogger.create_overlay()`` or directly.
    """

    def __init__(
        self,
        cfg: OverlayCfg,
        scene: Scene,
        coordinate_transform: CoordinateTransform,
        entity_path_prefix: str = "",
    ) -> None:
        self._cfg = cfg
        self._scene = scene
        self._coordinate_transform = coordinate_transform
        self._entity_path_prefix = entity_path_prefix
        self._actors: dict[str, _RigidActor] = {}
        self._articulated: dict[str, _ArticulatedOverlay] = {}
        self._init_and_log(scene)

    # ─── Construction + static logging (one-time) ────────────────────────

    def _init_and_log(self, scene: Scene) -> None:
        """Discover actors, log static meshes/transforms."""
        s2t = self._coordinate_transform.source_to_target
        art_actors: list[Actor] = []

        def find_articulated(a: Actor) -> None:
            if a.get_type() == ActorType.ARTICULATED and self._matches(a.get_name()):
                art_actors.append(a)

        scene.for_each_actor(find_articulated)

        articulated_link_names: set[str] = set()
        for actor in art_actors:
            result = self._build_and_log_articulated(scene, actor, s2t)
            if result is None:
                continue
            overlay, link_names = result
            self._articulated[actor.get_name()] = overlay
            articulated_link_names.update(link_names)

        def gather_rigid(a: Actor) -> None:
            if a.get_surface_mesh().is_empty():
                return
            name = a.get_name()
            if name in articulated_link_names or a.is_nested_link_actor():
                return
            if not self._matches(name):
                return
            path = self._make_rigid_path(a)
            self._actors[name] = _RigidActor(path, a)
            self._log_mesh(path, a)
            if a.has_root_transform():
                pos, rotvec = transformrt_to_numpy(a.get_root_transform())
                _log_transform(path, s2t @ make_transform(pos, rotvec), static=False)
            else:
                # No world pose to log: the mesh is logged but stays at the world
                # origin until the caller pushes a transform per frame
                # (set_rigid_pose / update_from_scene). Warn so the gap is
                # visible — it usually means the overlay was built from the wrong
                # scene (e.g. a parallel scene whose actors never received poses).
                logger.warning(
                    "Overlay %r: rigid actor %r has a mesh but no root "
                    "transform; it will render at the world origin until "
                    "positioned per frame.",
                    self._cfg.name,
                    name,
                )

        scene.for_each_actor(gather_rigid)

        logger.info(
            f"Overlay '{self._cfg.name}': "
            f"{len(self._articulated)} articulated, {len(self._actors)} rigid"
        )

    def _build_and_log_articulated(
        self, scene: Scene, actor: Actor, s2t: np.ndarray
    ) -> tuple[_ArticulatedOverlay, set[str]] | None:
        """Build entity hierarchy, log mounts/meshes.

        If ``cfg.anchors`` specifies an anchor link for this actor, the tree
        starts at that link; everything upstream is omitted. Cosmetic
        filtering (``include/exclude_actors``) hides meshes but does NOT
        remove links from the kinematic chain.

        The hierarchy uses geometry-frame joint transforms from
        ``get_articulated_shape_info`` (``parent_link_from_joint`` /
        ``joint_from_child_link``), as needed for link surface meshes, which
        are authored in the geometry frame. Rest-pose FK is
        ``rootFromLink[i] = rootFromLink[parent] @ parent_link_from_joint[i]
        @ jointTx[i] @ joint_from_child_link[i]``; the static mount holds
        ``parent_link_from_joint[i]`` and the per-frame leaf transform holds
        ``jointTx[i] @ joint_from_child_link[i]``.
        """
        actor_name = actor.get_name()
        shape_info = actor.get_articulated_shape_info()
        link_handles = list(actor.get_nested_link_actors())
        parents = list(shape_info.parents)
        link_names_list = list(shape_info.link_names)
        # Geometry-frame joint transforms (no center-of-mass offset baked in).
        # Sized numLinks + numCycles; tree-link entries occupy [0, numLinks).
        parent_link_from_joint = list(shape_info.parent_link_from_joint)
        joint_from_child_link = list(shape_info.joint_from_child_link)
        root_from_links_at_rest = list(shape_info.root_from_links_at_rest)
        num_links = len(link_handles)
        if num_links == 0:
            return None

        anchor_index = _resolve_anchor_index(
            actor_name, link_names_list, self._cfg.anchors.get(actor_name)
        )
        in_subtree = _mark_subtree(parents, anchor_index)

        layer = actor.get_contact_layer()
        parts: list[str] = []
        if self._entity_path_prefix:
            parts.append(self._entity_path_prefix)
        parts.extend(["world", self._cfg.name, str(layer), actor_name])
        root_entity = "/".join(parts)

        link_actors = [scene.get_actor(h) for h in link_handles]

        # pyre-fixme[9]: pyre infers `list[None]` from `[None] * N`; the
        # annotation widens it to the value type actually used below.
        entity_paths: list[str | None] = [None] * num_links
        for i in range(num_links):
            if not in_subtree[i]:
                continue
            safe = link_names_list[i].replace("/", "_")
            parent_path = root_entity if i == anchor_index else entity_paths[parents[i]]
            entity_paths[i] = f"{parent_path}/{safe}_mount/{safe}"

        # Default anchor world pose: its geometry-frame rest pose. Per-frame
        # ``set_articulated_pose(anchor_transform=...)`` overrides this.
        actor_root = (
            _transformrt_to_4x4(actor.get_root_transform())
            if actor.has_root_transform()
            else np.eye(4, dtype=np.float64)
        )
        _log_transform(
            root_entity,
            s2t
            @ actor_root
            @ _transformrt_to_4x4(root_from_links_at_rest[anchor_index]),
            static=False,
        )

        logged_names: set[str] = set()
        # pyre-fixme[9]
        mount_inv_list: list[np.ndarray | None] = [None] * num_links
        # pyre-fixme[9]
        child_offset_list: list[np.ndarray | None] = [None] * num_links
        for i in range(num_links):
            if not in_subtree[i]:
                continue
            if i == anchor_index:
                mount_tf = np.eye(4, dtype=np.float64)
                child_tf = np.eye(4, dtype=np.float64)
            else:
                mount_tf = _transformrt_to_4x4(parent_link_from_joint[i])
                child_tf = _transformrt_to_4x4(joint_from_child_link[i])
            mount_path = entity_paths[i].rsplit("/", 1)[0]  # type: ignore[union-attr]
            _log_transform(mount_path, mount_tf, static=True)
            mount_inv_list[i] = np.linalg.inv(mount_tf)
            child_offset_list[i] = child_tf
            # Rest-pose leaf transform (joint at identity). Per-frame updates
            # override this via set_articulated_pose / update_from_scene.
            _log_transform(cast(str, entity_paths[i]), child_tf, static=False)

            link_actor = link_actors[i]
            if link_actor is None:
                continue
            if (
                self._matches(link_actor.get_name())
                and not link_actor.get_surface_mesh().is_empty()
            ):
                self._log_mesh(entity_paths[i], link_actor)  # type: ignore[arg-type]
            logged_names.add(link_actor.get_name())

        return (
            _ArticulatedOverlay(
                actor=actor,
                root_entity_path=root_entity,
                entity_paths=entity_paths,
                mount_inv=mount_inv_list,
                child_offset=child_offset_list,
                parents=parents,
                anchor_index=anchor_index,
            ),
            logged_names,
        )

    def _log_mesh(self, entity_path: str, actor: Actor) -> None:
        """Extract mesh from actor and log to Rerun as static geometry."""
        s2t = self._coordinate_transform.source_to_target
        vertices, faces, normals = extract_actor_mesh(actor)
        verts_t = apply_linear_map(s2t[:3, :3], vertices)
        if self._cfg.flat_shading:
            verts_t, faces, normals_t = compute_face_normals(verts_t, faces)
        else:
            normals_t = apply_linear_map(s2t[:3, :3], normals)
        rr.log(
            entity_path,
            rr.Mesh3D(
                vertex_positions=verts_t,
                triangle_indices=faces,
                vertex_normals=normals_t,
                albedo_factor=self._cfg.color,
            ),
            static=True,
        )

    def _matches(self, name: str) -> bool:
        inc = self._cfg.include_actors
        if inc is not None and not any(fnmatch(name, p) for p in inc):
            return False
        exc = self._cfg.exclude_actors
        return exc is None or not any(fnmatch(name, p) for p in exc)

    def _make_rigid_path(self, actor: Actor) -> str:
        parts: list[str] = []
        if self._entity_path_prefix:
            parts.append(self._entity_path_prefix)
        parts.extend(
            [
                "world",
                self._cfg.name,
                str(actor.get_contact_layer()),
                f"{actor.get_name()}_h{actor.get_handle().value}",
            ]
        )
        return "/".join(parts)

    # ─── Per-frame updates ────────────────────────────────────────────────

    def set_transform(
        self, actor_name: str, transform: npt.NDArray[np.floating]
    ) -> None:
        """Set a rigid actor's world-space transform from a 4×4 matrix."""
        oa = self._actors.get(actor_name)
        if oa is None:
            return
        s2t = self._coordinate_transform.source_to_target
        _log_transform(oa.entity_path, s2t @ np.asarray(transform))

    def set_rigid_pose(
        self,
        actor_name: str,
        pos: npt.ArrayLike,
        *,
        quat: npt.ArrayLike | None = None,
        rotvec: npt.ArrayLike | None = None,
        rot6d: npt.ArrayLike | None = None,
        mat: npt.ArrayLike | None = None,
    ) -> None:
        """Set a rigid actor's world-space pose from pos + one rotation rep."""
        if actor_name not in self._actors:
            return
        self.set_transform(
            actor_name,
            pose_to_transform(pos, quat=quat, rotvec=rotvec, rot6d=rot6d, mat=mat),
        )

    def update_from_scene(self, scene: Scene | None = None) -> None:
        """Update all overlay transforms from live scene state.

        Syncs rigid actor world poses and articulated link transforms.
        For articulated actors, each link's world-space transform is queried
        directly (bypassing FK recomposition) to avoid drift from rest-pose
        mount approximations.
        """
        scene = scene or self._scene
        scene_actors: dict[str, Actor] = {}
        scene.for_each_actor(lambda a: scene_actors.__setitem__(a.get_name(), a))
        s2t = self._coordinate_transform.source_to_target
        for oa in self._actors.values():
            sa = scene_actors.get(oa.actor.get_name())
            if sa is not None and sa.has_root_transform():
                pos, rotvec = transformrt_to_numpy(sa.get_root_transform())
                _log_transform(oa.entity_path, s2t @ make_transform(pos, rotvec))
        for art in self._articulated.values():
            sa = scene_actors.get(art.actor.get_name())
            if sa is None:
                continue
            self._update_articulated_from_scene(scene, sa, art, s2t)

    def _update_articulated_from_scene(
        self,
        scene: Scene,
        sa: Actor,
        art: _ArticulatedOverlay,
        s2t: np.ndarray,
    ) -> None:
        """Sync a single articulated overlay from its scene actor."""
        link_handles = list(sa.get_nested_link_actors())
        # pyre-fixme[9]: pyre infers `list[None]` from `[None] * N`.
        link_worlds: list[np.ndarray | None] = [None] * len(link_handles)
        for i, h in enumerate(link_handles):
            if art.entity_paths[i] is None:
                continue
            la = scene.get_actor(h)
            if la is not None and la.has_root_transform():
                pos, rotvec = transformrt_to_numpy(la.get_root_transform())
                link_worlds[i] = s2t @ make_transform(pos, rotvec)
        anchor = art.anchor_index
        anchor_world = link_worlds[anchor]
        anchor_mount_inv = art.mount_inv[anchor]
        if anchor_world is not None and anchor_mount_inv is not None:
            _log_transform(art.root_entity_path, anchor_world @ anchor_mount_inv)
        for i in range(len(link_handles)):
            path = art.entity_paths[i]
            link_world_i = link_worlds[i]
            if path is None or link_world_i is None:
                continue
            if i == anchor:
                _log_transform(path, np.eye(4, dtype=np.float64))
                continue
            p = art.parents[i]
            parent_world = link_worlds[p]
            mount_inv = art.mount_inv[i]
            if parent_world is None or mount_inv is None:
                continue
            local_tf = mount_inv @ np.linalg.inv(parent_world) @ link_world_i
            _log_transform(path, local_tf)

    def set_articulated_pose(
        self,
        actor_name: str,
        joint_angles: npt.NDArray[np.floating],
        *,
        anchor_transform: npt.ArrayLike | None = None,
    ) -> None:
        """Update an articulated overlay's joints and (optionally) its anchor pose.

        ``anchor_transform`` is the world-space 4×4 of the overlay's anchor link
        (the base, or the mid-chain link declared in ``OverlayCfg.anchors``).
        When omitted, the previously logged anchor pose is reused. Joint angles
        index into the actor's full DOF vector; only joints downstream of the
        anchor are applied (slots above the anchor are ignored).
        """
        art = self._articulated.get(actor_name)
        if art is None:
            return

        s2t = self._coordinate_transform.source_to_target
        if anchor_transform is not None:
            _log_transform(
                art.root_entity_path,
                s2t @ np.asarray(anchor_transform, dtype=np.float64).reshape(4, 4),
            )

        q = np.asarray(joint_angles, dtype=np.float64)
        shape_info = art.actor.get_articulated_shape_info()
        for i, dof_info in enumerate(shape_info.dof_info):
            path = art.entity_paths[i]
            if path is None or i == art.anchor_index:
                continue
            n_dofs = dof_info.get_size()
            if n_dofs == 0:
                continue
            offset = dof_info.offset
            dofs = q[offset : offset + n_dofs]
            T = np.eye(4, dtype=np.float64)
            joint_type = shape_info.joint_types[i]
            if joint_type == sdp.ArticulatedJointType.REVOLUTE:
                axis = np.array(shape_info.joint_axes[i], dtype=np.float64)
                T[:3, :3] = Rotation.from_rotvec(float(dofs[0]) * axis).as_matrix()
            elif joint_type == sdp.ArticulatedJointType.PRISMATIC:
                T[:3, 3] = float(dofs[0]) * np.array(
                    shape_info.joint_axes[i], dtype=np.float64
                )
            elif joint_type == sdp.ArticulatedJointType.SPHERICAL:
                T[:3, :3] = Rotation.from_rotvec(dofs[:3]).as_matrix()
            elif joint_type == sdp.ArticulatedJointType.FREE:
                T[:3, :3] = Rotation.from_rotvec(dofs[3:6]).as_matrix()
                T[:3, 3] = dofs[:3]
            # Compose the joint motion with the geometry-frame child-link
            # offset so the leaf node lands on the link geometry frame.
            child = art.child_offset[i]
            _log_transform(path, T if child is None else T @ child)

    # ─── Queries ──────────────────────────────────────────────────────────

    def get_actor_names(self) -> list[str]:
        """Names of all rigid actors in this overlay."""
        return list(self._actors)

    def get_articulated_names(self) -> list[str]:
        """Names of all articulated actors in this overlay."""
        return list(self._articulated)

    def get_entity_path(self, actor_name: str) -> str | None:
        """Rerun entity path for ``actor_name``, or None if unknown."""
        oa = self._actors.get(actor_name)
        if oa is not None:
            return oa.entity_path
        art = self._articulated.get(actor_name)
        return art.root_entity_path if art is not None else None

    def clear(self) -> None:
        """Clear all overlay entities from Rerun."""
        for oa in self._actors.values():
            rr.log(oa.entity_path, rr.Clear(recursive=True))
        for art in self._articulated.values():
            rr.log(art.root_entity_path, rr.Clear(recursive=True))


########################################################################################
# Module-private helpers


def _resolve_anchor_index(
    actor_name: str, link_names: list[str], anchor_link: str | None
) -> int:
    """Index of the anchor link (base by default).

    Accepts bare (``fr3_link8``) or prefixed (``Robot/fr3_link8``) names —
    Mochi's ``shape_info.link_names`` uses bare names, but link actors carry
    the prefixed form, so apps typically think in prefixed terms.
    """
    if anchor_link is None:
        return 0
    for cand in (anchor_link, anchor_link.removeprefix(f"{actor_name}/")):
        if cand in link_names:
            return link_names.index(cand)
    raise ValueError(
        f"Anchor link {anchor_link!r} not found in {actor_name!r}; "
        f"available: {link_names}"
    )


def _mark_subtree(parents: list[int], root_index: int) -> list[bool]:
    """Mark ``root_index`` and all descendants in a parent-indexed tree."""
    n = len(parents)
    in_subtree = [False] * n
    in_subtree[root_index] = True
    # Parents have lower indices by construction → single forward pass suffices.
    for i in range(root_index + 1, n):
        p = parents[i]
        if 0 <= p < n and in_subtree[p]:
            in_subtree[i] = True
    return in_subtree
