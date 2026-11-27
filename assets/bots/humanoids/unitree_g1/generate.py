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

"""Writes the Unitree G1 (29 DoF, rev 1.0) bot and its joint controller from unitree_rl_mjlab.

Source: unitree_rl_mjlab @1425b15 (Apache-2.0), ``src/assets/robots/unitree_g1/xmls/g1.xml`` and
``g1_constants.py``; tarball sha256 186d70765121c2aa356798cc9c51c0e29d0027891b72fce62856b3111c79ad4d.
Run from this directory: ``python3 generate.py --xml <path to g1.xml>``. It reads the MJCF as XML
text (nothing of the upstream repo runs) and rewrites ``unitree_g1.superdex_bot``,
``unitree_g1_box_feet.superdex_bot`` and ``shapes/``, and the LAB_JOINT_ACTUATORS init args of
``tasks/g1_velocity_flat/g1_velocity.superdex_scene``. As in g1.xml, the bot's joints carry no armature or
effort limit: the actuators set them.

Each MJCF body becomes a link, its joint the link's joint (the pelvis's free joint the root). Mochi
has no capsule collider and one shape per link, so each collision capsule becomes spheres on
massless Hard-jointed child links: a foot capsule its two end spheres (on a plane, MuJoCo's capsule
contact is those two), a body capsule spheres no more than one radius apart. The box-feet bot
replaces each foot's 7 capsules by one box around them.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCENE_DIR = _HERE.parents[2] / "tasks/g1_velocity_flat"
"""The G1 scene's folder: the generator writes its actuators."""
_TAG = ""
"""Shape paths are relative to the bot file."""

# g1_constants.py: rotor inertias and gears per motor, reflected through a two-stage planetary.
_NATURAL_FREQ = 10 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0


def _reflected(
    rotor: tuple[float, float, float], gears: tuple[float, float, float]
) -> float:
    """mjlab's reflected_inertia_from_two_stage_planetary."""
    return rotor[0] * (gears[1] * gears[2]) ** 2 + rotor[1] * gears[2] ** 2 + rotor[2]


_ARM_5020 = _reflected((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16))
_ARM_7520_14 = _reflected((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22))
_ARM_7520_22 = _reflected((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))
_ARM_4010 = _reflected((0.068e-4, 0.0, 0.0), (1, 5, 5))

_ACTUATORS: list[tuple[tuple[str, ...], float, float]] = [
    # (joint name patterns, armature, effort limit); waist roll/pitch and ankles: 2 x 5020.
    (
        (
            r".*_elbow_joint",
            r".*_shoulder_pitch_joint",
            r".*_shoulder_roll_joint",
            r".*_shoulder_yaw_joint",
            r".*_wrist_roll_joint",
        ),
        _ARM_5020,
        25.0,
    ),
    (
        (r".*_hip_pitch_joint", r".*_hip_yaw_joint", r"waist_yaw_joint"),
        _ARM_7520_14,
        88.0,
    ),
    ((r".*_hip_roll_joint", r".*_knee_joint"), _ARM_7520_22, 139.0),
    ((r".*_wrist_pitch_joint", r".*_wrist_yaw_joint"), _ARM_4010, 5.0),
    ((r"waist_pitch_joint", r"waist_roll_joint"), 2 * _ARM_5020, 50.0),
    ((r".*_ankle_pitch_joint", r".*_ankle_roll_joint"), 2 * _ARM_5020, 50.0),
]

_ACTUATOR_NAMES = (
    "g1_5020",
    "g1_7520_14",
    "g1_7520_22",
    "g1_4010",
    "g1_waist",
    "g1_ankle",
)
"""g1_constants.py's G1_ACTUATOR_5020, _7520_14, _7520_22, _4010, _WAIST and _ANKLE."""

# HOME_KEYFRAME (pelvis at z 0.8); joints not named are 0.
_HOME = {
    r".*_hip_pitch_joint": -0.1,
    r".*_knee_joint": 0.3,
    r".*_ankle_pitch_joint": -0.2,
    r".*_shoulder_pitch_joint": 0.35,
    r".*_elbow_joint": 0.87,
    r"left_shoulder_roll_joint": 0.18,
    r"right_shoulder_roll_joint": -0.18,
}
HOME_HEIGHT = 0.8

_FOOT_GEOM = re.compile(r"^(left|right)_foot[1-7]_collision$")
_SITES = ("imu_in_pelvis", "imu_in_torso", "left_foot", "right_foot")
_LIMIT_STIFFNESS = 1000.0
"""N m/rad: about 10x the stiffest PD (knee/hip roll kp 99), so a limit holds within ~0.1 rad of a
full-effort push. MuJoCo's limits are soft constraints with no single stiffness to copy."""


def _floats(text: str | None, default: list[float]) -> list[float]:
    return [float(v) for v in text.split()] if text else list(default)


def _quat_xyzw(wxyz: list[float]) -> list[float]:
    w, x, y, z = wxyz
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return [x / n, y / n, z / n, w / n]


def _rotation_matrix(wxyz: list[float]) -> list[list[float]]:
    x, y, z, w = _quat_xyzw(wxyz)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _inertia(inertial: ET.Element) -> list[float]:
    """The inertia tensor at the COM in body axes, [ixx, ixy, ixz, iyy, iyz, izz] (tensor entries)."""
    if inertial.get("fullinertia"):
        ixx, iyy, izz, ixy, ixz, iyz = _floats(inertial.get("fullinertia"), [])
        return [ixx, ixy, ixz, iyy, iyz, izz]
    d = _floats(inertial.get("diaginertia"), [])
    r = _rotation_matrix(_floats(inertial.get("quat"), [1, 0, 0, 0]))
    t = [
        [sum(r[i][k] * d[k] * r[j][k] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]
    return [t[0][0], t[0][1], t[0][2], t[1][1], t[1][2], t[2][2]]


def _actuator(joint: str) -> tuple[float, float]:
    for patterns, armature, effort in _ACTUATORS:
        if any(re.fullmatch(p, joint) for p in patterns):
            return armature, effort
    raise ValueError(f"no G1 actuator for joint {joint}")


def gains(joint: str) -> tuple[float, float, float]:
    """mjlab's BuiltinPositionActuator stiffness, damping and effort limit for `joint`."""
    armature, effort = _actuator(joint)
    return (
        armature * _NATURAL_FREQ**2,
        2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ,
        effort,
    )


def _spheres(geom: ET.Element) -> list[tuple[list[float], float]]:
    """Sphere centers (in the body frame) and radius standing in for a collision geom."""
    radius = _floats(geom.get("size"), [0.01])[0]
    kind = geom.get("type", "capsule")
    if kind == "sphere":
        return [(_floats(geom.get("pos"), [0, 0, 0]), radius)]
    assert kind == "capsule" and geom.get("fromto"), geom.attrib
    p = _floats(geom.get("fromto"), [])
    a, b = p[:3], p[3:]
    if _FOOT_GEOM.match(geom.get("name", "")):
        return [(a, radius), (b, radius)]
    length = math.dist(a, b)
    n = max(1, math.ceil(length / radius))
    return [
        ([a[i] + (b[i] - a[i]) * k / n for i in range(3)], radius) for k in range(n + 1)
    ]


def _shape_name(radius: float) -> str:
    return f"shapes/sphere_r{round(radius * 1000)}mm.mochi.json"


_FOOT_FRICTION = 0.36
"""A foot collider's Coulomb friction: Mochi takes the geometric mean of the two actors', and the G1
ground's is 1, so the foot-ground friction is sqrt(0.36) = 0.6, g1_constants.py's foot friction."""
_COLLIDER_MASS = 0.001
"""kg: a link with a shape must have mass. Each collider link takes this much from its parent link,
which keeps the parent's and colliders' total mass and center of mass (see _take_mass)."""


def _massless(
    name: str, parent: int, translation: list[float], mass: float = 0.0
) -> tuple[dict, dict]:
    inertia = 0.4 * mass * 0.01**2
    link = {
        "name": name,
        "parentLink": parent,
        "mass": mass,
        "centerOfMass": [0, 0, 0],
        "momentOfInertia": [inertia, 0, 0, inertia, 0, inertia],
    }
    joint = {
        "name": f"{name}_joint",
        "type": "Hard",
        "parentLinkFromJoint": {"translation": translation},
    }
    return link, joint


def _add_colliders_and_sites(
    links: list[dict],
    joints: list[dict],
    extras: list[tuple[int, ET.Element]],
    feet: str,
    body_colliders: bool,
    radii: set[float],
) -> None:
    """Appends the sites and collider links after the kinematic tree, so the 30 bodies keep MJCF
    order, and adds the sphere radii used to `radii`."""
    box_done: set[int] = set()
    for parent, element in extras:
        name = element.get("name", "")
        if element.tag == "site":
            if name in _SITES:
                link, joint = _massless(
                    name, parent, _floats(element.get("pos"), [0, 0, 0])
                )
                links.append(link)
                joints.append(joint)
            continue
        if not name.endswith("_collision"):
            continue
        is_foot = bool(_FOOT_GEOM.match(name))
        if not is_foot and not body_colliders:
            continue
        if is_foot and feet == "box":
            if parent in box_done:
                continue
            box_done.add(parent)
            side = name.split("_")[0]
            link, joint = _massless(
                f"{side}_foot_box", parent, list(_FOOT_BOX_CENTER), _COLLIDER_MASS
            )
            link.update(
                shape=_TAG + "shapes/foot_box.mochi.json",
                colliderType="Box",
                layer="G1Foot",
                contact={"coulombFrictionCoefficient": _FOOT_FRICTION},
            )
            links.append(link)
            joints.append(joint)
            continue
        for k, (center, radius) in enumerate(_spheres(element)):
            radii.add(radius)
            link, joint = _massless(
                f"{name[: -len('_collision')]}_s{k}", parent, center, _COLLIDER_MASS
            )
            link.update(
                shape=_TAG + _shape_name(radius),
                colliderType="Sphere",
                layer="G1Foot" if is_foot else "G1Body",
            )
            if is_foot:
                link["contact"] = {"coulombFrictionCoefficient": _FOOT_FRICTION}
            links.append(link)
            joints.append(joint)


def build_bot(
    xml: Path, feet: str, body_colliders: bool
) -> tuple[dict, list[str], set[float]]:
    """The bot, its 29 actuated joints in bot order, and the sphere radii it uses."""
    root = ET.parse(xml).getroot()
    pelvis = root.find("./worldbody/body")
    assert pelvis is not None and pelvis.get("name") == "pelvis"
    links: list[dict] = []
    joints: list[dict] = []
    actuated: list[str] = []
    extras: list[tuple[int, ET.Element]] = []
    radii: set[float] = set()

    def visit(body: ET.Element, parent: int) -> None:
        index = len(links)
        inertial = body.find("inertial")
        assert inertial is not None, body.get("name")
        links.append(
            {
                "name": body.get("name"),
                **({"parentLink": parent} if parent >= 0 else {}),
                "mass": float(inertial.get("mass")),
                "centerOfMass": _floats(inertial.get("pos"), [0, 0, 0]),
                "momentOfInertia": _inertia(inertial),
            }
        )
        frame: dict[str, list[float]] = {}
        if parent >= 0:
            frame["translation"] = _floats(body.get("pos"), [0, 0, 0])
            if body.get("quat"):
                frame["rotation"] = _quat_xyzw(_floats(body.get("quat"), []))
        mj_joint = body.find("joint")
        if body.find("freejoint") is not None:
            joints.append({"name": body.find("freejoint").get("name"), "type": "Free"})
        elif mj_joint is not None:
            name = mj_joint.get("name")
            axis = _floats(mj_joint.get("axis"), [0, 0, 1])
            low, high = _floats(mj_joint.get("range"), [])
            _actuator(name)  # every joint has an actuator
            joints.append(
                {
                    "name": name,
                    "type": "Revolute",
                    "axis": axis,
                    "parentLinkFromJoint": frame,
                    "minLimit": [low * a for a in axis],
                    "maxLimit": [high * a for a in axis],
                    "limitStiffness": _LIMIT_STIFFNESS,
                }
            )
            actuated.append(name)
        else:
            joints.append(
                {
                    "name": f"{body.get('name')}_joint",
                    "type": "Hard",
                    "parentLinkFromJoint": frame,
                }
            )
        for child in body:
            if child.tag in ("geom", "site"):
                extras.append((index, child))
        for child in body.findall("body"):
            visit(child, index)

    visit(pelvis, -1)

    _add_colliders_and_sites(links, joints, extras, feet, body_colliders, radii)
    _take_mass(links, joints)
    bot = {
        "name": "unitree_g1_29dof" + ("_box_feet" if feet == "box" else ""),
        "defaultPose": [home_position(j) for j in actuated],
        "joints": joints,
        "links": links,
        "contactOverrides": [],
    }
    return bot, actuated, radii


def _take_mass(links: list[dict], joints: list[dict]) -> None:
    """Moves each collider link's mass out of its parent link, shifting the parent's center of mass
    so that their total mass and center of mass stay g1.xml's. The parent's inertia is kept (the
    colliders' 1 g each changes a foot's inertia by under 0.1%)."""
    for link, joint in zip(links, joints):
        mass = link["mass"]
        if joint["type"] != "Hard" or "shape" not in link or mass == 0:
            continue
        parent = links[link["parentLink"]]
        at = joint["parentLinkFromJoint"]["translation"]
        rest = parent["mass"] - mass
        parent["centerOfMass"] = [
            (parent["mass"] * c - mass * a) / rest
            for c, a in zip(parent["centerOfMass"], at, strict=True)
        ]
        parent["mass"] = rest


def home_position(joint: str) -> float:
    for pattern, value in _HOME.items():
        if re.fullmatch(pattern, joint):
            return value
    return 0.0


# The 7 foot capsules (radius 0.01) span x [-0.054, 0.132], y [-0.026, 0.026], z -0.025 in the ankle
# roll link: the box covers them with the radius added, its bottom at the capsules' bottom.
_FOOT_BOX_HALF = (0.103, 0.036, 0.01)
_FOOT_BOX_CENTER = (0.039, 0.0, -0.025)


def _icosphere(level: int) -> tuple[list[list[float]], list[tuple[int, int, int]]]:
    t = (1 + math.sqrt(5)) / 2
    verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]  # fmt: skip
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
        (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
        (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    verts = [[c / math.sqrt(sum(x * x for x in v)) for c in v] for v in verts]
    for _ in range(level):
        cache: dict[tuple[int, int], int] = {}

        def mid(a: int, b: int, cache: dict[tuple[int, int], int] = cache) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = [(verts[a][i] + verts[b][i]) / 2 for i in range(3)]
                n = math.sqrt(sum(x * x for x in m))
                verts.append([x / n for x in m])
                cache[key] = len(verts) - 1
            return cache[key]

        new = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    return verts, faces


def _signed_volume(p: list[list[float]], tet: list[int]) -> float:
    a, b, c, d = (p[i] for i in tet)
    u = [b[i] - a[i] for i in range(3)]
    v = [c[i] - a[i] for i in range(3)]
    w = [d[i] - a[i] for i in range(3)]
    return (
        u[0] * (v[1] * w[2] - v[2] * w[1])
        - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
    ) / 6


def _tet_mesh(points: list[list[float]], tets: list[list[int]]) -> dict:
    fixed = []
    for tet in tets:
        if _signed_volume(points, tet) < 0:
            tet = [tet[0], tet[2], tet[1], tet[3]]
        fixed.append(tet)
    return {
        "element_size": 4,
        "coordinates": [round(c, 9) for p in points for c in p],
        "connectivity": [i for t in fixed for i in t],
    }


def sphere_mesh(radius: float, level: int = 2) -> dict:
    """A tet mesh of a sphere: an icosphere surface (level 2: 320 faces, ~0.3 r edges) fanned to its center."""
    verts, faces = _icosphere(level)
    points = [[0.0, 0.0, 0.0]] + [[radius * c for c in v] for v in verts]
    return _tet_mesh(points, [[0, a + 1, b + 1, c + 1] for a, b, c in faces])


def box_mesh(half: tuple[float, float, float], cells: tuple[int, int, int]) -> dict:
    """A structured tet mesh of a box centered at the origin, 6 tets per cell (Kuhn split)."""
    nx, ny, nz = cells
    points = [
        [
            -half[0] + 2 * half[0] * i / nx,
            -half[1] + 2 * half[1] * j / ny,
            -half[2] + 2 * half[2] * k / nz,
        ]
        for k in range(nz + 1)
        for j in range(ny + 1)
        for i in range(nx + 1)
    ]

    def node(i: int, j: int, k: int) -> int:
        return (k * (ny + 1) + j) * (nx + 1) + i

    tets = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = [
                    node(i + (b & 1), j + (b >> 1 & 1), k + (b >> 2 & 1))
                    for b in range(8)
                ]
                for path in ((1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6)):
                    tets.append([c[0], c[path[0]], c[path[0] | path[1]], c[7]])
    return _tet_mesh(points, tets)


def actuators(actuated: list[str]) -> dict:
    """LAB_JOINT_ACTUATORS init args: g1_constants.py's 6 BuiltinPositionActuatorCfg as mjlab-style
    `position` actuators (implicit damping), each over its joint patterns with one value for all."""
    specs = []
    for patterns, armature, effort in _ACTUATORS:
        specs.append(
            {
                "name": _ACTUATOR_NAMES[len(specs)],
                "type": "position",
                "joints": list(patterns),
                "stiffness": [armature * _NATURAL_FREQ**2],
                "damping": [2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ],
                "effortLimit": [effort],
                "armature": [armature],
            }
        )
    return {"joints": actuated, "actuators": specs}


def action_scale(actuated: list[str]) -> list[float]:
    """G1_ACTION_SCALE: 0.25 effort limit / stiffness per joint."""
    return [0.25 * gains(j)[2] / gains(j)[0] for j in actuated]


def _write(path: Path, content: dict, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=sort_keys) + "\n")


def _write_scene_actuators(init_args: dict) -> None:
    """Writes the actuators into each G1 bot's LAB_JOINT_ACTUATORS entry of the G1 scene, inline:
    a scene entry's init args file would resolve against the process's directory, not the scene's."""
    path = _SCENE_DIR / "g1_velocity.superdex_scene"
    scene = json.loads(path.read_text())
    for bot in scene["bots"]:
        for entry in bot["controllers"]:
            if entry["type"] == "LAB_JOINT_ACTUATORS":
                entry["initArgs"] = init_args
    _write(path, scene, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--xml", type=Path, required=True, help="unitree_rl_mjlab's g1.xml"
    )
    parser.add_argument("--out", type=Path, default=_HERE)
    parser.add_argument(
        "--body-colliders", action="store_true", help="also the 19 body capsules"
    )
    args = parser.parse_args()
    radii: set[float] = set()
    for feet, name in (("spheres", "unitree_g1"), ("box", "unitree_g1_box_feet")):
        bot, actuated, used = build_bot(args.xml, feet, args.body_colliders)
        radii |= used
        _write(args.out / f"{name}.superdex_bot", bot)
    for radius in sorted(radii):
        _write(args.out / _shape_name(radius), sphere_mesh(radius), sort_keys=False)
    _write(
        args.out / "shapes/foot_box.mochi.json",
        box_mesh(_FOOT_BOX_HALF, (20, 7, 1)),
        sort_keys=False,
    )
    _write_scene_actuators(actuators(actuated))
    print("joints:", json.dumps(actuated))
    print("action scale:", json.dumps([round(s, 6) for s in action_scale(actuated)]))


if __name__ == "__main__":
    main()
