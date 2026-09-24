#!/usr/bin/env python3
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

# PEP 723 inline metadata: `uv run tools/build_revo2_assets.py` resolves these into a cached
# environment. A plain `python tools/build_revo2_assets.py` ignores them.
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "superdex-physics==1.0.0",
#     "numpy",
#     "scipy",
#     "scikit-image",
#     "trimesh",
#     "manifold3d",
#     "rtree",
# ]
# ///

"""Build the BrainCo Revo2 SuperDex bot assets from the upstream URDF package.

    uv run tools/build_revo2_assets.py                         # clone the pinned upstream
    uv run tools/build_revo2_assets.py --source <revo2_description checkout>

Run from the repository root. Writes ``assets/bots/hands/revo2/{left,right}/``:

- ``collision/<link>_collision.mochi.h5`` -- watertight collider with a baked grid SDF.
  The upstream STLs are open, multi-shell CAD exports (and the palm is a hollow housing),
  so every collider is voxel-remeshed into a closed solid, smoothed, and simplified with a
  manifold-preserving decimator before its SDF is baked offline. Baking offline is what
  keeps bot creation fast; the runtime URDF importer would spend minutes baking at load.
- ``render/<link>_render.glb`` -- visual mesh (Y-up glTF, as SuperDex expects).
- ``revo2_<side>.superdex_bot`` -- kinematics, dynamics, the five mimic couplings as rigid
  linear transmissions, rest-pose contact overrides, and a ``TACTILE_PAD`` sensor on every
  fingertip ``*_touch_link``, modeled on the Revo2 Touch (capacitive) sensor.

The build is deterministic for a given upstream commit, so rerunning it after an upstream
bump regenerates every file and the diff shows what changed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import manifold3d
import numpy as np
import scipy.ndimage as ndi
import superdex.physics as physics
import trimesh
from scipy.spatial.transform import Rotation
from skimage import measure
from skimage.morphology import convex_hull_image

UPSTREAM_URL = "https://github.com/BrainCoTech/revo2_description.git"
UPSTREAM_COMMIT = "92cc697c7fa691db59404ce52344f3969a5ef7a6"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "assets" / "bots" / "hands" / "revo2"

SIDES = ("right", "left")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# Registered by superdex.lab.sensors.tactile; a build without it skips the sensor with a
# warning and still loads the bot.
TACTILE_SENSOR_TYPE = "TACTILE_PAD"
# Revo2 Touch (capacitive) fingertip sensor: one sensing element per finger reporting a
# 3D force (normal, tangential, direction) and proximity. Specs: 0-25 N, 0.1 N
# resolution, 0-1 cm proximity (BrainCo; see assets/bots/hands/revo2/README.md).
TACTILE_PARAMS = {
    "rows": 1,
    "cols": 1,
    "saturation": 25.0,
    "resolution": 0.1,
    "proximity_range": 0.01,
}

# Silicone fingertip pads grip; the housing is hard plastic.
PAD_FRICTION = 1.0

# Contact penalty ramp for hand-scale geometry. The engine default ramps the penalty in
# over ~10 mm, which is thicker than the 3-5 mm pads and lets fingers sink ~4 mm into
# objects; 0.5 mm keeps grasp penetration around 0.5-1.3 mm.
HAND_CONTACT = {
    "penaltySmoothingHalfDistance": 0.0005,
    "penaltyThresholdDefault": 0.0005,
}
# One contact sample per triangle on the housing links (the default is three); the pads
# keep the default density because it sets the tactile map's spatial resolution.
HOUSING_BOUNDARY_ELEMENT = "P1Q1"

# Small armature and damping keep the ~10 g finger links well conditioned. The pose
# controller supplies the servo stiffness/damping; these are passive drivetrain terms.
FINGER_JOINT_ARMATURE = 2.0e-5  # [kg m^2]
FINGER_JOINT_FRICTION = {"coulomb": 0.002, "viscous": 0.005}

# Mimic couplings are rigid (gear/linkage), so the transmission may push and pull.
MIMIC_STIFFNESS = 50.0  # [N m / rad]
MIMIC_DAMPING = 0.05  # [N m s / rad]

# glTF is Y-up; SuperDex link frames are Z-up. (x, y, z) -> (x, z, -y).
Z_UP_TO_Y_UP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
)

HOUSING_RGBA = (0.16, 0.16, 0.18, 1.0)
PAD_RGBA = (0.80, 0.82, 0.85, 1.0)


@dataclass(frozen=True)
class MeshRecipe:
    """Resolution settings for one collider, in meters."""

    voxel_pitch: float
    simplify_tolerance: float
    sdf_voxel: float


RECIPE_PALM = MeshRecipe(
    voxel_pitch=0.5e-3, simplify_tolerance=0.3e-3, sdf_voxel=0.8e-3
)
RECIPE_THUMB_BASE = MeshRecipe(0.3e-3, 0.12e-3, 0.5e-3)
RECIPE_PHALANX = MeshRecipe(0.25e-3, 0.1e-3, 0.4e-3)
RECIPE_PAD = MeshRecipe(0.2e-3, 0.08e-3, 0.3e-3)

# Below this height the palm housing is an open shell (the thumb mechanism sits in the
# opening). Each slice is closed with its 2D convex hull so the palm is a solid. Above it
# the four knuckle housings are separate closed loops and are only hole-filled.
PALM_SOLID_BELOW_Z = 0.062  # [m]

SDF_PADDING = 0.005  # [m]


@dataclass
class UrdfLink:
    name: str
    mass: float = 0.0
    com: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    inertia: list[float] = field(default_factory=lambda: [0.0] * 6)
    visuals: list[str] = field(default_factory=list)
    collisions: list[str] = field(default_factory=list)


@dataclass
class UrdfJoint:
    name: str
    type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray | None = None
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    mimic: tuple[str, float, float] | None = None


def parse_urdf(path: Path) -> tuple[dict[str, UrdfLink], list[UrdfJoint]]:
    root = ET.parse(path).getroot()
    links: dict[str, UrdfLink] = {}
    for el in root.findall("link"):
        link = UrdfLink(name=el.get("name"))
        inertial = el.find("inertial")
        if inertial is not None:
            origin = inertial.find("origin")
            if any(float(v) != 0.0 for v in origin.get("rpy", "0 0 0").split()):
                raise ValueError(
                    f"{link.name}: rotated inertial frames are not supported"
                )
            link.com = [float(v) for v in origin.get("xyz").split()]
            link.mass = float(inertial.find("mass").get("value"))
            i = inertial.find("inertia")
            link.inertia = [
                float(i.get(k)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
            ]
        for tag, dest in (("visual", link.visuals), ("collision", link.collisions)):
            for g in el.findall(tag):
                origin = g.find("origin")
                if origin is not None and any(
                    float(v) != 0.0
                    for v in origin.get("xyz").split() + origin.get("rpy").split()
                ):
                    raise ValueError(
                        f"{link.name}: offset {tag} geometry is not supported"
                    )
                dest.append(g.find("geometry/mesh").get("filename").split("/")[-1])
        links[link.name] = link

    joints: list[UrdfJoint] = []
    for el in root.findall("joint"):
        origin = el.find("origin")
        joint = UrdfJoint(
            name=el.get("name"),
            type=el.get("type"),
            parent=el.find("parent").get("link"),
            child=el.find("child").get("link"),
            xyz=np.array([float(v) for v in origin.get("xyz").split()]),
            rpy=np.array([float(v) for v in origin.get("rpy").split()]),
        )
        if joint.type == "revolute":
            joint.axis = np.array(
                [float(v) for v in el.find("axis").get("xyz").split()]
            )
            limit = el.find("limit")
            joint.lower = float(limit.get("lower"))
            joint.upper = float(limit.get("upper"))
            joint.effort = float(limit.get("effort"))
            mimic = el.find("mimic")
            if mimic is not None:
                joint.mimic = (
                    mimic.get("joint"),
                    float(mimic.get("multiplier", "1")),
                    float(mimic.get("offset", "0")),
                )
        elif joint.type != "fixed":
            raise ValueError(f"{joint.name}: unsupported joint type {joint.type}")
        joints.append(joint)
    return links, joints


def natural_key(name: str) -> list:
    """Sort key matching SuperDex's alphanumeric link ordering."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def superdex_link_order(
    links: dict[str, UrdfLink], joints: list[UrdfJoint]
) -> list[str]:
    """DFS pre-order with alphanumerically sorted children, i.e. the order SuperDex sorts
    bots into on load. Authoring in this order keeps file indices equal to runtime ones."""
    children: dict[str, list[str]] = {name: [] for name in links}
    child_names = set()
    for j in joints:
        children[j.parent].append(j.child)
        child_names.add(j.child)
    (root,) = [n for n in links if n not in child_names]
    order, stack = [], [root]
    while stack:
        name = stack.pop()
        order.append(name)
        stack.extend(sorted(children[name], key=natural_key, reverse=True))
    return order


def joint_transform(joint: UrdfJoint) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler(
        "xyz", joint.rpy
    ).as_matrix()  # URDF: fixed-axis XYZ
    T[:3, 3] = joint.xyz
    return T


# ------------------------------------------------------------------------------------
# Meshes
# ------------------------------------------------------------------------------------


def load_mesh(mesh_dir: Path, filename: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_dir / filename, force="mesh")
    mesh.merge_vertices()
    return mesh


def voxelize_solid(
    meshes: list[trimesh.Trimesh], pitch: float, solid_below_z: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Occupancy grid of the union of ``meshes`` with enclosed cavities filled."""
    combined = trimesh.util.concatenate(meshes)
    vox = combined.voxelized(pitch)
    occupancy = vox.matrix.copy()
    transform = vox.transform.copy()
    if solid_below_z is not None:
        # Close each open horizontal section of the palm housing with its convex hull.
        z_of_layer = transform[2, 3] + pitch * np.arange(occupancy.shape[2])
        for k in np.nonzero(z_of_layer < solid_below_z)[0]:
            if occupancy[:, :, k].any():
                occupancy[:, :, k] |= convex_hull_image(occupancy[:, :, k])
    occupancy = ndi.binary_fill_holes(occupancy)
    return occupancy, transform


def remesh_watertight(
    meshes: list[trimesh.Trimesh],
    recipe: MeshRecipe,
    solid_below_z: float | None = None,
) -> trimesh.Trimesh:
    """Closed, manifold, simplified surface enclosing the union of ``meshes``."""
    occupancy, transform = voxelize_solid(meshes, recipe.voxel_pitch, solid_below_z)
    # Smoothing the occupancy before contouring removes voxel stair-steps and the
    # diagonal-touch configurations that would make the contour non-manifold.
    field_ = ndi.gaussian_filter(np.pad(occupancy, 1).astype(np.float32), 0.6)
    verts, faces, _, _ = measure.marching_cubes(field_, 0.5)
    verts = trimesh.transform_points(verts - 1.0, transform)
    contour = trimesh.Trimesh(verts, faces)
    contour.fix_normals()
    solid = manifold3d.Manifold(
        manifold3d.Mesh(
            vert_properties=contour.vertices.astype(np.float32),
            tri_verts=contour.faces.astype(np.uint32),
        )
    )
    if solid.status() != manifold3d.Error.NoError:
        raise RuntimeError(f"contour is not manifold: {solid.status()}")
    simplified = solid.simplify(recipe.simplify_tolerance).to_mesh()
    # process=False: merging coincident vertices across a thin wall would turn the
    # manifold output non-manifold.
    out = trimesh.Trimesh(
        np.asarray(simplified.vert_properties)[:, :3].astype(np.float64),
        np.asarray(simplified.tri_verts),
        process=False,
    )
    out.fix_normals()
    out.vertices = relax_slivers(out, max_step=0.02 * recipe.voxel_pitch)
    if not (out.is_watertight and out.is_volume):
        raise RuntimeError("remeshed collider is not a closed volume")
    if len(sliver_faces(out.vertices, out.faces)):
        raise RuntimeError("remeshed collider still has degenerate triangles")
    return out


def sliver_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Near-zero-area triangles, measured at the engine's single precision.

    Mochi integrates a collider's volume over its surface triangles, so a single
    degenerate one makes the volume NaN and the link cannot be created."""
    tri = vertices.astype(np.float32).astype(np.float64)[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    longest = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1)
    return np.nonzero((area < 1e-13) | (area / longest**2 < 1e-4))[0]


def relax_slivers(
    mesh: trimesh.Trimesh, max_step: float, max_iters: int = 200
) -> np.ndarray:
    """Nudge the vertices of degenerate triangles toward their one-ring centroid.

    The decimator occasionally leaves collinear triangles. Moving vertices (by at most
    ``max_step`` per iteration) fixes them without changing connectivity, so the mesh
    stays manifold; the total displacement is tens of micrometers."""
    vertices = mesh.vertices.copy()
    neighbors = mesh.vertex_neighbors
    for _ in range(max_iters):
        bad = sliver_faces(vertices, mesh.faces)
        if len(bad) == 0:
            break
        for i in np.unique(mesh.faces[bad]):
            delta = vertices[neighbors[i]].mean(axis=0) - vertices[i]
            dist = np.linalg.norm(delta)
            if dist > 0:
                vertices[i] += delta * min(1.0, max_step / dist)
    return vertices


def write_collider(mesh: trimesh.Trimesh, recipe: MeshRecipe, path: Path) -> None:
    model = physics.ModelData()
    model.mesh = physics.MeshData(
        nodes_per_element=3,
        coordinates=mesh.vertices.ravel(),
        connectivity=mesh.faces.ravel(),
    )
    physics.model.bake_sdf(
        model,
        physics.GridSdfParams(
            resolution_mode=physics.GridSdfResolutionMode.EXPLICIT,
            resolution_delta=[recipe.sdf_voxel] * 3,
            boundary_padding_dist=SDF_PADDING,
            min_grid_resolution=[6, 6, 6],
        ),
    )
    physics.model.validate(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    physics.model.save_to_file(model, str(path), physics.FileFormat.H5)


def write_render_glb(
    meshes: list[trimesh.Trimesh], rgba: tuple[float, ...], name: str, path: Path
) -> None:
    mesh = trimesh.util.concatenate(meshes)
    mesh.merge_vertices()
    mesh.apply_transform(Z_UP_TO_Y_UP)
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name=f"M_Revo2_{name}",
            baseColorFactor=[int(round(255 * c)) for c in rgba],
            metallicFactor=0.0,
            roughnessFactor=0.6,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Scene({name: mesh}).export(str(path), file_type="glb")


def recipe_for(link_name: str) -> MeshRecipe:
    if link_name.endswith("base_link"):
        return RECIPE_PALM
    if link_name.endswith("touch_link"):
        return RECIPE_PAD
    if "thumb_metacarpal" in link_name or "thumb_proximal" in link_name:
        return RECIPE_THUMB_BASE
    return RECIPE_PHALANX


# ------------------------------------------------------------------------------------
# Tactile pad frame
# ------------------------------------------------------------------------------------


def pad_sensor_frame(
    pad: trimesh.Trimesh, distal: trimesh.Trimesh, pad_from_distal: np.ndarray
) -> dict:
    """Sensor frame on a fingertip pad, in the pad link's frame.

    z is the pad's outward surface normal (the thinnest principal axis, pointing away
    from the finger), y runs along the finger toward the tip (the longest principal
    axis, pointing away from the distal joint), and the origin is the center of the
    pad's footprint in that plane. ``size`` is the footprint extent along (x, y).
    """
    verts = pad.vertices
    centroid = verts.mean(axis=0)
    _, _, vt = np.linalg.svd(verts - centroid, full_matrices=False)
    along, _, normal = vt  # descending variance
    finger_center = trimesh.transform_points(distal.vertices, pad_from_distal).mean(
        axis=0
    )
    distal_joint = pad_from_distal[:3, 3]
    if np.dot(normal, centroid - finger_center) < 0:
        normal = -normal
    if np.dot(along, centroid - distal_joint) < 0:
        along = -along
    across = np.cross(along, normal)
    rotation = np.column_stack([across, along, normal])  # pad_from_sensor
    local = (verts - centroid) @ rotation
    lo, hi = local.min(axis=0), local.max(axis=0)
    mid = 0.5 * (lo + hi)
    origin = centroid + rotation @ np.array(
        [mid[0], mid[1], hi[2]]
    )  # on the pad surface
    return {
        "translation": origin.tolist(),
        "rotation": Rotation.from_matrix(rotation).as_quat().tolist(),  # [x, y, z, w]
        "size": (hi[:2] - lo[:2]).tolist(),
    }


# ------------------------------------------------------------------------------------
# Bot file
# ------------------------------------------------------------------------------------


def rounded(values, digits: int = 9) -> list[float]:
    return [float(round(float(v), digits)) + 0.0 for v in values]


def transform_json(T: np.ndarray) -> dict:
    out = {}
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    if quat[3] < 0:
        quat = -quat
    if not np.allclose(quat, [0, 0, 0, 1], atol=1e-12):
        out["rotation"] = rounded(quat)
    if not np.allclose(T[:3, 3], 0.0, atol=1e-12):
        out["translation"] = rounded(T[:3, 3])
    return out


def finger_of(name: str) -> str | None:
    return next((f for f in FINGERS if f"_{f}_" in name), None)


def load_collider(path: Path) -> trimesh.Trimesh:
    model = physics.model.load_from_file(str(path))
    return trimesh.Trimesh(
        np.asarray(model.mesh.coordinates, dtype=np.float64).reshape(-1, 3),
        np.asarray(model.mesh.connectivity).reshape(-1, 3),
        process=False,
    )


def build_side(
    side: str, source: Path, out_dir: Path, reuse_colliders: bool = False
) -> None:
    links, joints = parse_urdf(source / "urdf" / f"revo2_{side}_hand.urdf")
    mesh_dir = source / "meshes" / f"revo2_{side}_hand"
    joint_by_child = {j.child: j for j in joints}
    order = superdex_link_order(links, joints)
    index_of = {name: i for i, name in enumerate(order)}
    side_dir = out_dir / side

    # --- Colliders and render models --------------------------------------------------
    colliders: dict[str, trimesh.Trimesh] = {}
    for name in order:
        link = links[name]
        if not link.collisions:  # *_tip_link: a massless fingertip frame
            continue
        collider_path = side_dir / "collision" / f"{name}_collision.mochi.h5"
        if reuse_colliders:
            colliders[name] = load_collider(collider_path)
            continue
        recipe = recipe_for(name)
        sources = [load_mesh(mesh_dir, f) for f in link.collisions]
        solid_below = None
        if name.endswith("base_link"):
            # The palm cover is visual-only upstream, but it closes the housing.
            sources += [
                load_mesh(mesh_dir, f) for f in link.visuals if f not in link.collisions
            ]
            solid_below = PALM_SOLID_BELOW_Z
        collider = remesh_watertight(sources, recipe, solid_below_z=solid_below)
        colliders[name] = collider
        write_collider(collider, recipe, collider_path)
        rgba = PAD_RGBA if name.endswith("touch_link") else HOUSING_RGBA
        write_render_glb(
            [load_mesh(mesh_dir, f) for f in link.visuals],
            rgba,
            name,
            side_dir / "render" / f"{name}_render.glb",
        )
        print(
            f"  {name}: {len(collider.faces)} collider faces, "
            f"volume {collider.volume * 1e6:.2f} cm^3"
        )

    # --- Links ----------------------------------------------------------------------
    bot_links = []
    for name in order:
        link = links[name]
        entry: dict = {
            "centerOfMass": rounded(link.com),
            "mass": float(link.mass),
            "momentOfInertia": rounded(link.inertia, 15),
            "name": name,
        }
        if name != order[0]:
            entry["parentLink"] = index_of[joint_by_child[name].parent]
        if name in colliders:
            entry["renderModel"] = f"render/{name}_render.glb"
            entry["shape"] = f"collision/{name}_collision.mochi.h5"
            entry["contact"] = dict(HAND_CONTACT)
            if not name.endswith("touch_link"):
                entry["boundaryElementType"] = HOUSING_BOUNDARY_ELEMENT
        if name.endswith("touch_link"):
            finger = finger_of(name)
            distal = f"{side}_{finger}_distal_link"
            pad_from_distal = np.linalg.inv(joint_transform(joint_by_child[name]))
            frame = pad_sensor_frame(
                colliders[name], colliders[distal], pad_from_distal
            )
            entry["contact"]["coulombFrictionCoefficient"] = PAD_FRICTION
            entry["sensors"] = [
                {
                    "type": TACTILE_SENSOR_TYPE,
                    "name": f"{finger}_tactile",
                    "parentFromSensor": {
                        "rotation": rounded(frame["rotation"]),
                        "translation": rounded(frame["translation"]),
                    },
                    "params": json.dumps(
                        {
                            "pad_from_sensor": {
                                "rotation": rounded(frame["rotation"]),
                                "translation": rounded(frame["translation"]),
                            },
                            "size": rounded(frame["size"]),
                            **TACTILE_PARAMS,
                        },
                        sort_keys=True,
                    ),
                }
            ]
        bot_links.append(entry)

    # --- Joints ---------------------------------------------------------------------
    bot_joints = []
    for name in order:
        if name == order[0]:
            bot_joints.append({"name": "world_joint", "type": "Free"})
            continue
        j = joint_by_child[name]
        entry = {
            "name": j.name,
            "parentLinkFromJoint": transform_json(joint_transform(j)),
        }
        if j.type == "fixed":
            entry.update(type="Hard", minLimit=[0, 0, 0], maxLimit=[0, 0, 0])
        else:
            axis = j.axis / np.linalg.norm(j.axis)
            entry.update(
                type="Revolute",
                axis=rounded(axis),
                minLimit=rounded(axis * j.lower),
                maxLimit=rounded(axis * j.upper),
                effortLimit=j.effort,
                inertia=FINGER_JOINT_ARMATURE,
                friction=dict(FINGER_JOINT_FRICTION),
            )
        bot_joints.append(entry)

    # --- Mimic couplings as rigid linear transmissions ------------------------------
    joint_index = {j["name"]: i for i, j in enumerate(bot_joints)}
    transmissions = []
    for j in joints:
        if j.mimic is None:
            continue
        leader, multiplier, offset = j.mimic
        if offset != 0.0:
            raise ValueError(f"{j.name}: mimic offsets are not supported")
        # s = multiplier * q_leader - q_follower is held at zero.
        transmissions.append(
            {
                "allowCompressiveForce": True,
                "damping": MIMIC_DAMPING,
                "jointAxisDisps": [0.0, 0.0],
                "jointCoefficients": [multiplier, -1.0],
                "jointIndices": [joint_index[leader], joint_index[j.name]],
                "name": f"{j.name}_mimic",
                "stiffness": MIMIC_STIFFNESS,
            }
        )

    # --- Contact overrides for pairs that touch at rest ------------------------------
    overrides = rest_pose_overlaps(order, links, joint_by_child, colliders)

    num_dofs = sum(1 for j in bot_joints if j["type"] == "Revolute")
    bot = {
        "contactOverrides": [
            {"enable": False, "linkA": a, "linkB": b} for a, b in overrides
        ],
        "defaultPose": [0.0] * num_dofs,
        "joints": bot_joints,
        "linearTransmissions": transmissions,
        "links": bot_links,
        "name": f"revo2_{side}",
    }
    bot_path = side_dir / f"revo2_{side}.superdex_bot"
    bot_path.write_text(json.dumps(bot, indent=2, sort_keys=True) + "\n")
    print(
        f"  wrote {bot_path.relative_to(REPO_ROOT)} ({num_dofs} DOFs, "
        f"{len(transmissions)} mimic transmissions, {len(overrides)} contact overrides)"
    )


def rest_pose_overlaps(order, links, joint_by_child, colliders, margin: float = 0.5e-3):
    """Link pairs whose colliders overlap (or nearly touch) in the zero pose, which the
    default parent/child rule does not already exclude. Their contact is disabled so the
    hand does not push itself apart at spawn."""
    world_from = {order[0]: np.eye(4)}
    for name in order[1:]:
        j = joint_by_child[name]
        world_from[name] = world_from[j.parent] @ joint_transform(j)

    def excluded_by_default(a: str, b: str) -> bool:
        return any(
            name in joint_by_child and joint_by_child[name].parent == other
            for name, other in ((a, b), (b, a))
        )

    placed = {}
    for name, mesh in colliders.items():
        m = mesh.copy()
        m.apply_transform(world_from[name])
        placed[name] = m
    names = [n for n in order if n in placed]
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if excluded_by_default(a, b):
                continue
            ma, mb = placed[a], placed[b]
            lo = np.maximum(ma.bounds[0], mb.bounds[0]) - margin
            hi = np.minimum(ma.bounds[1], mb.bounds[1]) + margin
            if np.any(lo > hi):
                continue
            samples = mb.sample(4000, seed=0)
            inside = (samples >= lo).all(axis=1) & (samples <= hi).all(axis=1)
            if not inside.any():
                continue
            depth = trimesh.proximity.signed_distance(ma, samples[inside])
            if depth.max() > -margin:
                pairs.append((a, b))
    return pairs


MOUNT_OUTPUT = REPO_ROOT / "assets" / "bots" / "arm_hand_combos" / "fr3_v2_revo2"
MOUNT_NAME = "fr3_revo2_mount"
# Generic adapter plate: FR3 flange (ISO 9409-1-31.5, ~63 mm face) tapering to the
# Revo2's ~40 mm wrist face. BrainCo does not publish its adapter, so this is a simple
# aluminum frustum of plausible size and mass, not a replica.
MOUNT_FLANGE_RADIUS = 0.0315  # [m]
MOUNT_WRIST_RADIUS = 0.022  # [m]
MOUNT_LENGTH = 0.015  # [m]
MOUNT_DENSITY = 2700.0  # [kg/m^3] aluminum
MOUNT_RGBA = (0.62, 0.64, 0.67, 1.0)
MOUNT_RECIPE = MeshRecipe(
    voxel_pitch=0.5e-3, simplify_tolerance=0.1e-3, sdf_voxel=1.0e-3
)


def build_fr3_mount(out_dir: Path) -> dict:
    """Adapter collider and render model; returns its link inertial parameters in the
    mount frame (origin on the flange face, +z toward the hand)."""
    profile = np.array(
        [
            [0.0, 0.0],
            [MOUNT_FLANGE_RADIUS, 0.0],
            [MOUNT_FLANGE_RADIUS, 0.004],
            [MOUNT_WRIST_RADIUS, MOUNT_LENGTH],
            [0.0, MOUNT_LENGTH],
        ]
    )
    mount = trimesh.creation.revolve(profile, sections=96)
    mount.fix_normals()
    if not (mount.is_watertight and mount.is_volume):
        raise RuntimeError("mount mesh is not a closed volume")
    mount.density = MOUNT_DENSITY
    write_collider(
        mount, MOUNT_RECIPE, out_dir / "collision" / f"{MOUNT_NAME}_collision.mochi.h5"
    )
    write_render_glb(
        [mount], MOUNT_RGBA, MOUNT_NAME, out_dir / "render" / f"{MOUNT_NAME}_render.glb"
    )
    inertia = mount.moment_inertia  # about the center of mass, mount axes
    return {
        "centerOfMass": rounded(mount.center_mass),
        "mass": float(round(mount.mass, 6)),
        "momentOfInertia": rounded(
            [
                inertia[0, 0],
                inertia[0, 1],
                inertia[0, 2],
                inertia[1, 1],
                inertia[1, 2],
                inertia[2, 2],
            ],
            12,
        ),
        "length": MOUNT_LENGTH,
    }


def fetch_upstream(dest: Path) -> Path:
    subprocess.run(["git", "clone", "--quiet", UPSTREAM_URL, str(dest)], check=True)
    subprocess.run(
        ["git", "-C", str(dest), "checkout", "--quiet", UPSTREAM_COMMIT], check=True
    )
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, help="revo2_description checkout")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sides", nargs="+", choices=SIDES, default=list(SIDES))
    parser.add_argument(
        "--mount-only", action="store_true", help="only rebuild the FR3 adapter mount"
    )
    parser.add_argument(
        "--reuse-colliders",
        action="store_true",
        help="keep the existing colliders and render models; only regenerate the bot files",
    )
    args = parser.parse_args()

    physics.initialize(num_worker_threads=0)
    if not args.reuse_colliders:
        print("Building FR3 mount...")
        print(
            f"  mount inertial parameters: {json.dumps(build_fr3_mount(MOUNT_OUTPUT))}"
        )
    if args.mount_only:
        physics.shutdown()
        return
    with tempfile.TemporaryDirectory() as tmp:
        source = args.source or fetch_upstream(Path(tmp) / "revo2_description")
        for side in args.sides:
            print(f"Building revo2 {side}...")
            if not args.reuse_colliders:
                shutil.rmtree(args.output / side, ignore_errors=True)
            build_side(side, source, args.output, reuse_colliders=args.reuse_colliders)
    physics.shutdown()


if __name__ == "__main__":
    main()
