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

"""Resuming a recorded pick-up from any moment keeps the object in the hand.

Each scenario lowers an open hand over an object resting on a table, closes it,
lifts the object, and moves it through a sequence of motions, under backward Euler.
Resumes from frames across the episode must keep the object, and an object restored
from the wrong frame must be lost.
"""

from __future__ import annotations

import gc
import math
import os
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import superdex.physics as sdp
import superdex.recorder as recorder
import superdex.robotics as sdr
from superdex.physics.paths import resolve_asset
from superdex.recorder.reader import EpisodeReader

# Set to a directory to keep a Rerun recording of every run, named by scenario and
# resume frame.
RECORDINGS_DIR_ENV = "SUPERDEX_RESUME_CAPABILITY_RECORDINGS"

DT = 0.005
APPROACH_END = 40
CLOSE_END = 70
LIFT_START = 90
LIFT_END = 190
APPROACH_HEIGHT = 0.08
LIFT_HEIGHT = 0.12
# Each motion is followed by this many frames holding still.
HOLD_FRAMES = 40
MOTION_FRAMES = {"swing": 80, "fig8": 240, "wrist8": 240, "shake": 200}
# Every scenario also resumes while the hand comes down, closes, and lifts.
EARLY_RESUMES = (20, 55, 140)
# A restore bug that the check must catch: the object restored from this many frames
# before the hand.
STALE_FRAMES = 20
# The object is lost once it moves this far in the palm frame.
MAX_DRIFT = 0.03
MIN_CONTACT_FORCE = 0.1
TABLE_CLEARANCE = 0.003

ARM_TRACKING = (1000.0, 100.0, -1.0)
SPHERE_SHAPE = "prefabs/sphere/collision/sphere.mochi.h5"
SPHERE_SIZE = (0.03, 0.03, 0.03)
ROD_SHAPE = "prefabs/internal/simple_shapes/cylinder/collision/cylinder.mochi.h5"
ROD_SIZE = (0.03, 0.03, 0.25)
BOX_TRIANGLES = (
    (0, 2, 1),
    (1, 2, 3),
    (4, 5, 6),
    (5, 7, 6),
    (0, 1, 5),
    (0, 5, 4),
    (2, 6, 3),
    (3, 6, 7),
    (0, 4, 2),
    (2, 4, 6),
    (1, 3, 5),
    (3, 7, 5),
)
CONTACT = {
    "penalty_coefficient": 1e7,
    "penalty_threshold_default": 0.002,
    "coulomb_friction_coefficient": 1.0,
    "viscous_friction_coefficient": 2.0,
}


@dataclass(frozen=True)
class Embodiment:
    bot: str
    palm: str
    # Stiffness, damping and saturation of the hand joints. When `driven` is set, the
    # other hand joints are passive.
    finger_tracking: tuple[float, float, float] = (20.0, 0.5, -1.0)
    driven: tuple[str, ...] = ()


DG5F = Embodiment(
    "bots/arm_hand_combos/fr3_dg5f_short/right/fr3_dg5f_short_right.superdex_bot",
    "dg5f_link_palm",
)
WUJI2 = Embodiment(
    "bots/arm_hand_combos/fr3_v2_wuji2_beta1/right/fr3_v2_wuji2_beta1_right.superdex_bot",
    "link_01_wrist",
)
# The 2F-85 linkage closes through soft springs, so only the knuckles are driven, with
# a capped torque that keeps the loops closed.
ROBOTIQ_2F85 = Embodiment(
    "bots/arm_hand_combos/fr3_v2_2f_85/fr3_v2_2f_85.superdex_bot",
    "2f_85_base_link",
    finger_tracking=(100.0, 2.0, 0.04),
    driven=("2f_85_left_knuckle_link", "2f_85_right_knuckle_link"),
)


@dataclass(frozen=True)
class Wrap:
    """Stand the object in front of the palm, for a hand whose fingers point sideways.

    It goes `normal` out from the palm toward where the fingers close, and `along`
    toward the fingertips. The direction the fingers close is measured by bending
    `knuckle`, which moves `fingertip`.
    """

    fingertip: str
    knuckle: str
    normal: float
    along: float


@dataclass(frozen=True)
class Grasp:
    embodiment: Embodiment
    # Arm joints, by pose index, that turn the hand for this grasp.
    arm: Mapping[int, float]
    # Hand joint targets by link name. The hand comes down in the spawn pose updated by
    # `open`, and closes to `closed`.
    open: Mapping[str, float]
    closed: Mapping[str, float]
    # Fingers, named by a part of their link names, that must carry the object once it
    # is lifted, and fingers that must not touch it.
    contacts: tuple[str, ...]
    idle: tuple[str, ...] = ()
    # The object rests under the point midway between the first of these links and
    # the rest in the closed pose, turned so that its first axis spans the grip.
    aim: tuple[str, ...] = ()
    wrap: Wrap | None = None


def dg5f_pose(*fingers: Sequence[float]) -> dict[str, float]:
    """Joint targets for the thumb, index, middle, ring and pinky."""
    return {
        f"dg5f_link_{finger}_{joint}": value
        for finger, values in enumerate(fingers, start=1)
        for joint, value in enumerate(values, start=1)
    }


def wuji2_pose(**fingers: Sequence[float]) -> dict[str, float]:
    links = {
        "thumb": ("02_thumb_proximal", "03_thumb_proximal_abd")
        + ("04_thumb_middle", "05_thumb_distal"),
        "index": ("09_index_finger_proximal", "10_index_finger_proximal_abd")
        + ("11_index_finger_middle", "12_index_finger_distal"),
        "middle": ("16_middle_finger_proximal", "17_middle_finger_proximal_abd")
        + ("18_middle_finger_middle", "19_middle_finger_distal"),
        "ring": ("23_ring_finger_proximal", "24_ring_finger_proximal_abd")
        + ("25_ring_finger_middle", "26_ring_finger_distal"),
        "pinky": ("30_pinky_proximal", "31_pinky_proximal_abd")
        + ("32_pinky_middle", "33_pinky_distal"),
    }
    return {
        f"link_{link}": value
        for finger, values in fingers.items()
        for link, value in zip(links[finger], values)
    }


def robotiq_pose(angle: float) -> dict[str, float]:
    joints = ("inner_knuckle", "knuckle", "finger_tip")
    return {
        f"2f_85_{side}_{joint}_link": sign * angle
        for side, sides in (("left", (1, 1, -1)), ("right", (-1, -1, 1)))
        for joint, sign in zip(joints, sides)
    }


DG5F_FINGERS = tuple(f"dg5f_link_{finger}_" for finger in range(1, 6))
DG5F_TIPS = tuple(f"{finger}tip" for finger in DG5F_FINGERS)
# Palm down, fingers pointing forward and down.
DG5F_PALM_DOWN = {5: 3.14, 6: 0.0}
DG5F_RING_PINKY_STRAIGHT = ((0.035, 0, 0, 0), (0.087, 0.105, 0, 0))
DG5F_PINCH = Grasp(
    DG5F,
    arm=DG5F_PALM_DOWN,
    open=dg5f_pose(
        (-0.38, -1.49, 0.2, 0.87),
        (0.05, 1.34, 0.01, 0),
        (-0.035, 0, 0, 0),
        *DG5F_RING_PINKY_STRAIGHT,
    ),
    # The thumb's last joint is driven past its limit, which holds it straight.
    closed=dg5f_pose(
        (0.81, -0.98, 0.59, -0.4),
        (-0.06, 1.55, 0, 0.04),
        (-0.035, 0, 0, 0),
        *DG5F_RING_PINKY_STRAIGHT,
    ),
    contacts=DG5F_FINGERS[:2],
    idle=("dg5f_link_palm", *DG5F_FINGERS[2:]),
    aim=DG5F_TIPS[:2],
)
DG5F_TRIPOD = Grasp(
    DG5F,
    arm=DG5F_PALM_DOWN,
    open=dg5f_pose(
        (-0.3, -1.52, 0.19, 0.21),
        (0.05, 1.27, 0, 0.62),
        (0.36, 1.1, 0.59, 0.36),
        *DG5F_RING_PINKY_STRAIGHT,
    ),
    closed=dg5f_pose(
        (-0.17, -1.62, 0.59, 0),
        (-0.06, 1.7, 0.01, 0),
        (-0.11, 1.75, 0.15, 0),
        *DG5F_RING_PINKY_STRAIGHT,
    ),
    contacts=DG5F_FINGERS[:3],
    idle=("dg5f_link_palm", *DG5F_FINGERS[3:]),
    aim=DG5F_TIPS[:3],
)
# Palm facing the object, fingers pointing forward.
DG5F_POWER = Grasp(
    DG5F,
    arm={5: 3.14, 6: -1.57},
    open={},
    closed=dg5f_pose(
        (0.5, -1.2, 0.9, 0.9),
        (-0.1, 1.0, 1.0, 1.0),
        (-0.03, 1.0, 1.0, 1.0),
        (0.03, 1.0, 1.0, 1.0),
        (0.087, 0.08, 1.0, 1.0),
    ),
    contacts=("dg5f_link_palm", DG5F_FINGERS[0], DG5F_FINGERS[1], DG5F_FINGERS[3]),
    wrap=Wrap("dg5f_link_3_tip", "dg5f_link_3_2", normal=0.045, along=0.045),
)
WUJI2_FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
WUJI2_PADS = tuple(
    f"link_{pad}_pad"
    for pad in ("07_thumb", "14_index_finger", "21_middle_finger")
    + ("28_ring_finger", "35_pinky")
)
WUJI2_PALM_DOWN = {5: 2.8, 6: 0.0}
WUJI2_PINCH = Grasp(
    WUJI2,
    arm=WUJI2_PALM_DOWN,
    open=wuji2_pose(thumb=(1.24, -0.78, -0.89, 0.09), index=(1.26, -0.23, 0.03, 0.65)),
    closed=wuji2_pose(
        thumb=(1.25, -0.48, -0.77, 0.13), index=(1.57, -0.32, -0.01, 0.45)
    ),
    contacts=WUJI2_FINGERS[:2],
    idle=("link_01_wrist", *WUJI2_FINGERS[2:]),
    aim=WUJI2_PADS[:2],
)
WUJI2_TRIPOD = Grasp(
    WUJI2,
    arm=WUJI2_PALM_DOWN,
    open=wuji2_pose(
        thumb=(0.97, -1.06, 0.42, -0.7),
        index=(1.15, -0.32, 0.18, 0.9),
        middle=(1.18, -0.37, 0.12, 0.85),
    ),
    closed=wuji2_pose(
        thumb=(0.89, -0.79, 0.63, -0.83),
        index=(1.57, -0.29, -0.06, 0.38),
        middle=(1.57, -0.28, -0.04, 0.39),
    ),
    contacts=WUJI2_FINGERS[:3],
    idle=("link_01_wrist", *WUJI2_FINGERS[3:]),
    aim=WUJI2_PADS[:3],
)
WUJI2_WRAP = (1.2, 0, 1.3, 0.8)
WUJI2_POWER = Grasp(
    WUJI2,
    arm={5: 3.5, 6: 1.57},
    open={},
    closed=wuji2_pose(
        thumb=(1.0, -0.4, 0.45, 0.35),
        index=WUJI2_WRAP,
        middle=WUJI2_WRAP,
        ring=WUJI2_WRAP,
        pinky=WUJI2_WRAP,
    ),
    contacts=("link_01_wrist", *WUJI2_FINGERS[:3]),
    wrap=Wrap(
        "link_22_middle_finger_tip",
        "link_16_middle_finger_proximal",
        normal=0.04,
        along=0.05,
    ),
)
ROBOTIQ_PARALLEL = Grasp(
    ROBOTIQ_2F85,
    arm={},
    open={},
    closed=robotiq_pose(0.8),
    contacts=("left_finger_tip", "right_finger_tip"),
    idle=("inner_knuckle", "base_link"),
    aim=("2f_85_left_finger_tip_link", "2f_85_right_finger_tip_link"),
)


@dataclass(frozen=True)
class Pickup:
    kind: str
    size: tuple[float, float, float]
    # Without a mass, the density is 200 kg/m^3.
    mass: float | None = None
    # A rod lies down unless it stands.
    upright: bool = False

    def create_shape(self) -> tuple[sdp.ShapeHandle, sdp.ColliderType]:
        if self.kind == "box":
            sx, sy, sz = (s / 2 for s in self.size)
            coordinates = [
                c
                for z in (-sz, sz)
                for y in (-sy, sy)
                for x in (-sx, sx)
                for c in (x, y, z)
            ]
            shape = sdp.create_tri_mesh_shape(
                coordinates=coordinates,
                connectivity=[i for triangle in BOX_TRIANGLES for i in triangle],
            )
            return shape, sdp.ColliderType.BOX
        path, native = (
            (SPHERE_SHAPE, SPHERE_SIZE)
            if self.kind == "sphere"
            else (ROD_SHAPE, ROD_SIZE)
        )
        scale = [s / n for s, n in zip(self.size, native)]
        return (
            sdp.load_shape_from_file(str(resolve_asset(path)), scale),
            sdp.ColliderType.AUTO,
        )

    def height(self) -> float:
        return self.size[0] if self.kind == "rod" and not self.upright else self.size[2]


TALL_BLOCK = Pickup("box", (0.045, 0.055, 0.12))
CUBE_25 = Pickup("box", (0.025, 0.025, 0.025), mass=0.03)
CUBE_35 = Pickup("box", (0.035, 0.035, 0.035), mass=0.03)
SPHERE_25 = Pickup("sphere", (0.025, 0.025, 0.025), mass=0.02)
SPHERE_30 = Pickup("sphere", (0.03, 0.03, 0.03), mass=0.03)
ROD_18 = Pickup("rod", (0.018, 0.018, 0.15), mass=0.04)
UPRIGHT_ROD_30 = Pickup("rod", (0.03, 0.03, 0.15), mass=0.05, upright=True)


@dataclass(frozen=True)
class Scenario:
    grasp: Grasp
    pickup: Pickup
    motions: tuple[str, ...]
    # Check that an object restored from before the hand is lost during the swing.
    stale_check: bool = False


SCENARIOS = {
    "dg5f_power_block": Scenario(
        DG5F_POWER, TALL_BLOCK, ("swing", "wrist8"), stale_check=True
    ),
    "dg5f_pinch_cube": Scenario(DG5F_PINCH, CUBE_25, ("fig8", "wrist8")),
    "dg5f_pinch_sphere": Scenario(DG5F_PINCH, SPHERE_25, ("swing", "fig8")),
    "dg5f_tripod_cube": Scenario(DG5F_TRIPOD, CUBE_25, ("swing", "shake")),
    "dg5f_tripod_sphere": Scenario(DG5F_TRIPOD, SPHERE_30, ("fig8", "wrist8")),
    "wuji2_pinch_cube": Scenario(WUJI2_PINCH, CUBE_25, ("fig8", "wrist8")),
    "wuji2_pinch_sphere": Scenario(
        WUJI2_PINCH, SPHERE_25, ("swing", "wrist8"), stale_check=True
    ),
    "wuji2_tripod_sphere": Scenario(WUJI2_TRIPOD, SPHERE_30, ("swing", "fig8")),
    "wuji2_power_rod": Scenario(WUJI2_POWER, UPRIGHT_ROD_30, ("swing", "shake")),
    "robotiq_parallel_cube": Scenario(
        ROBOTIQ_PARALLEL,
        CUBE_35,
        ("swing", "fig8", "wrist8", "shake"),
        stale_check=True,
    ),
    "robotiq_parallel_rod": Scenario(ROBOTIQ_PARALLEL, ROD_18, ("swing", "wrist8")),
    "robotiq_parallel_sphere": Scenario(ROBOTIQ_PARALLEL, SPHERE_30, ("fig8", "shake")),
}


def ease(alpha: float) -> float:
    alpha = min(1.0, max(0.0, alpha))
    return 0.5 - 0.5 * math.cos(math.pi * alpha)


def add_motion(name: str, t: float, palm: list[float], arm: list[float]) -> None:
    """Offset the palm's position and orientation, or the arm joints, `t` seconds
    into a motion.

    The swing turns the base and stays there. The others are smooth loops that end
    where they started: a figure 8 of the palm with its orientation fixed, a figure 8
    of tilts and twist about the palm, and a vertical shake.
    """
    if name == "swing":
        arm[0] += 1.5 * ease(t / 0.4)
        return
    duration = MOTION_FRAMES[name] * DT
    ramp = 0.15 if name == "shake" else 0.2
    t = min(t, duration)
    envelope = ease(t / ramp) * ease((duration - t) / ramp)
    w = 2 * math.pi / 1.2
    if name == "fig8":
        palm[0] += 0.05 * math.sin(2 * w * t) * envelope
        palm[1] += 0.1 * math.sin(w * t) * envelope
    elif name == "wrist8":
        palm[3] += 0.35 * math.sin(w * t) * envelope
        palm[4] += 0.25 * math.sin(2 * w * t) * envelope
        palm[5] += 0.5 * math.cos(w * t) * envelope
    else:
        palm[2] += 0.02 * math.sin(8 * math.pi * t) * envelope


def unit(v: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(c * c for c in v))
    return [c / norm for c in v]


def magnitude(v: sdp.Real3) -> float:
    return math.sqrt(sum(c * c for c in v.tolist()))


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    rows = [row + [value] for row, value in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(n):
            if r != col:
                f = rows[r][col] / rows[col][col]
                rows[r] = [x - f * y for x, y in zip(rows[r], rows[col])]
    return [rows[i][n] / rows[i][i] for i in range(n)]


def distance_to_box(point: Sequence[float], box: sdp.Aabb) -> float:
    low, high = box.min.tolist(), box.max.tolist()
    return math.dist(
        point, [min(max(p, lo), hi) for p, lo, hi in zip(point, low, high)]
    )


class GraspScene:
    """A hand on an FR3 arm and an object on a table, built the same way every time."""

    def __init__(self, scenario: Scenario) -> None:
        grasp = scenario.grasp
        self.grasp = grasp
        self.robotics_context = sdr.create_context()
        scene = sdp.create_scene("resume_capability")
        scene.set_gravity([0, 0, -9.81])
        solver_params = scene.get_solver_params()
        solver_params.integration_method = sdp.IntegrationMethod.BACKWARD_EULER
        scene.set_solver_params(solver_params)
        prefab = sdr.load_bot_prefab_from_file(str(resolve_asset(grasp.embodiment.bot)))
        bot = sdr.create_bot(scene, prefab, self.robotics_context)
        actor = bot.get_articulated_actor() if bot is not None else None
        if bot is None or actor is None:
            raise RuntimeError(f"{grasp.embodiment.bot} was not loaded.")
        self.scene = scene
        self.actor = actor
        self.links: dict[str, sdp.Actor] = {}
        for handle in actor.get_nested_link_actors():
            link = scene.get_actor(handle)
            if link is not None:
                self.links[link.get_name().rsplit("/", 1)[-1]] = link
        self.hand = [
            link
            for name, link in self.links.items()
            if name != "base" and not name.startswith("fr3_")
        ]
        self.palm = self.links[grasp.embodiment.palm]
        dofs, tracking = self.joint_tracking()
        self.open_pose = self.joint_positions()
        for index, value in grasp.arm.items():
            self.open_pose[index] = value
        for name, value in grasp.open.items():
            self.open_pose[dofs[name]] = value
        self.closed_pose = self.open_pose.copy()
        for name, value in grasp.closed.items():
            self.closed_pose[dofs[name]] = value
        self.palm_basis = self.palm_jacobian_inverse()
        self.pickup, self.table = self.create_pickup(scenario.pickup)
        self.segments: list[tuple[str, int]] = []
        start = LIFT_END
        for motion in scenario.motions:
            self.segments.append((motion, start))
            start += MOTION_FRAMES[motion] + HOLD_FRAMES
        self.num_frames = start
        actor.set_articulated_pose_from_joints(self.target(0))
        self.root = actor.get_root_transform()
        controller = bot.create_controller("MOCHI_ARTICULATED_POSE")
        if not isinstance(controller, sdr.ControllerMochiArticulatedPose):
            raise RuntimeError("Failed to create the articulated pose controller.")
        controller.set_params(
            sdr.ControllerMochiArticulatedPoseParams(
                pose_controller_params=sdp.PoseControllerParams(
                    joint_tracking=sdp.DynamicArrayPoseTrackingParams(tracking)
                )
            )
        )
        controller.initialize(True)
        self.controller = controller

    def joint_tracking(self) -> tuple[dict[str, int], list[sdp.PoseTrackingParams]]:
        """The pose index of every link with a moving joint, and per-link tracking."""
        embodiment = self.grasp.embodiment
        info = self.actor.get_articulated_shape_info()
        names = [str(name) for name in info.link_names]
        dofs: dict[str, int] = {}
        tracking = []
        for name, joint in zip(names, info.joint_types):
            if joint == sdp.ArticulatedJointType.HARD:
                tracking.append(sdp.PoseTrackingParams())
                continue
            dofs[name] = len(dofs)
            if name.startswith("fr3_"):
                gains = ARM_TRACKING
            elif not embodiment.driven or name in embodiment.driven:
                gains = embodiment.finger_tracking
            else:
                gains = (0.0, 0.0, -1.0)
            stiffness, damping, saturation = gains
            tracking.append(
                sdp.PoseTrackingParams(
                    stiffness=stiffness, damping=damping, saturation=saturation
                )
            )
        return dofs, tracking

    def joint_positions(self) -> list[float]:
        positions = sdp.DynamicArrayReal(self.actor.get_num_dofs())
        self.actor.get_articulated_pose(positions)
        return list(positions)

    def palm_frame(self, pose: list[float]) -> sdp.TransformRT:
        self.actor.set_articulated_pose_from_joints(pose)
        return self.palm.get_center_of_mass_transform()

    def palm_jacobian_inverse(self) -> list[list[float]]:
        """Arm joint offsets per unit of palm translation along x, y and z, and of
        rotation about them, at the pose where the hand closes."""
        step = 1e-4
        start = self.palm_frame(self.open_pose)
        columns = []
        for joint in range(7):
            pose = self.open_pose.copy()
            pose[joint] += step
            frame = self.palm_frame(pose)
            moved = frame.translation.tolist()
            turned = (frame * start.inverse()).rotation.to_rotation_vector().tolist()
            columns.append(
                [(m - s) / step for m, s in zip(moved, start.translation.tolist())]
                + [r / step for r in turned]
            )
        jjt = [[sum(c[r] * c[s] for c in columns) for s in range(6)] for r in range(6)]
        basis = []
        for axis in range(6):
            y = solve(jjt, [1.0 if r == axis else 0.0 for r in range(6)])
            basis.append([sum(c[r] * y[r] for r in range(6)) for c in columns])
        return basis

    def lowest_hand_point(self) -> float:
        lowest = math.inf
        for pose in (self.open_pose, self.closed_pose):
            self.actor.set_articulated_pose_from_joints(pose)
            lowest = min(
                [lowest] + [link.get_aabb_world().min.tolist()[2] for link in self.hand]
            )
        return lowest

    def placement(self, pickup: Pickup) -> tuple[list[float], float]:
        """Where the object stands under the hand, and its heading."""
        grasp = self.grasp
        if grasp.wrap is not None:
            wrap = grasp.wrap
            palm = self.palm_frame(self.open_pose).translation.tolist()
            tip = self.links[wrap.fingertip].get_center_of_mass_transform()
            bent = self.open_pose.copy()
            bent[self.knuckle_index(wrap.knuckle)] += 0.05
            self.actor.set_articulated_pose_from_joints(bent)
            moved = self.links[wrap.fingertip].get_center_of_mass_transform()
            along = unit(
                [tip.translation[0] - palm[0], tip.translation[1] - palm[1], 0]
            )
            closing = [
                m - t
                for m, t in zip(moved.translation.tolist(), tip.translation.tolist())
            ]
            dot = closing[0] * along[0] + closing[1] * along[1]
            normal = unit([closing[0] - dot * along[0], closing[1] - dot * along[1], 0])
            center = [
                p + wrap.normal * n + wrap.along * a
                for p, n, a in zip(palm, normal, along)
            ]
            return center, math.atan2(normal[1], normal[0])
        self.actor.set_articulated_pose_from_joints(self.closed_pose)
        points = [
            self.links[name].get_center_of_mass_transform().translation.tolist()
            for name in grasp.aim
        ]
        first = points[0]
        rest = [sum(c) / len(points[1:]) for c in zip(*points[1:])]
        center = [(a + b) / 2 for a, b in zip(first, rest)]
        return center, math.atan2(first[1] - rest[1], first[0] - rest[0])

    def knuckle_index(self, name: str) -> int:
        info = self.actor.get_articulated_shape_info()
        index = 0
        for link, joint in zip(info.link_names, info.joint_types):
            if str(link) == name:
                return index
            if joint != sdp.ArticulatedJointType.HARD:
                index += 1
        raise KeyError(name)

    def create_pickup(self, pickup: Pickup) -> tuple[sdp.Actor, sdp.Actor]:
        """Rest the object on a table just below the lowest point of the hand."""
        center, heading = self.placement(pickup)
        table_top = self.lowest_hand_point() - TABLE_CLEARANCE
        center[2] = table_top + pickup.height() / 2
        rotation = sdp.Quaternion.rotation_z(heading)
        if pickup.kind == "rod" and not pickup.upright:
            axis = [math.cos(heading), math.sin(heading), 0.0]
            rotation = sdp.Quaternion.from_axis_angle(axis, math.pi / 2) * rotation
        shape, collider = pickup.create_shape()
        bounds = sdp.get_shape_aabb(shape)
        middle = [
            (h + lo) / 2 for h, lo in zip(bounds.max.tolist(), bounds.min.tolist())
        ]
        rotated_middle = (
            sdp.TransformRT(rotation=rotation) * sdp.TransformRT(translation=middle)
        ).translation.tolist()
        pickup_actor = self.scene.create_rigid_actor(
            name="Pickup",
            shape=shape,
            world_from_local=sdp.TransformRT(
                rotation=rotation,
                translation=[c - m for c, m in zip(center, rotated_middle)],
            ),
            collider_type=collider,
            contact=sdp.ContactParams(**CONTACT),
            mass=pickup.mass,
            density=None if pickup.mass is not None else 200.0,
        )
        table = self.scene.create_rigid_actor(
            name="Table",
            shape=Pickup("box", (0.3, 0.3, 0.02)).create_shape()[0],
            world_from_local=sdp.TransformRT(
                translation=[center[0], center[1], table_top - 0.01]
            ),
            collider_type=sdp.ColliderType.BOX,
            is_static=True,
            contact=sdp.ContactParams(**CONTACT),
        )
        if pickup_actor is None or table is None:
            raise RuntimeError("Failed to create the pickup and table.")
        pickup_actor.register_query(sdp.QueryType.CONTACT_POINTS)
        pickup_actor.register_query(sdp.QueryType.TOTAL_CONTACT_FORCE)
        return pickup_actor, table

    def target(self, frame: int) -> list[float]:
        """Come down, close the hand, lift the object, then run the motions."""
        if frame <= APPROACH_END:
            target = self.open_pose.copy()
        else:
            alpha = min(1.0, (frame - APPROACH_END) / (CLOSE_END - APPROACH_END))
            target = [
                (1 - alpha) * a + alpha * b
                for a, b in zip(self.open_pose, self.closed_pose)
            ]
        palm = [0.0] * 6
        palm[2] += APPROACH_HEIGHT * (1 - ease(frame / APPROACH_END))
        palm[2] += LIFT_HEIGHT * ease((frame - LIFT_START) / (LIFT_END - LIFT_START))
        arm = [0.0] * 7
        for motion, start in self.segments:
            if frame > start:
                add_motion(motion, (frame - start) * DT, palm, arm)
        for joint in range(7):
            target[joint] += arm[joint] + sum(
                offset * column[joint] for offset, column in zip(palm, self.palm_basis)
            )
        return target

    def step(self, frame: int) -> None:
        self.controller.compute_output(
            sdr.ControllerMochiArticulatedPoseObsv(),
            sdr.ControllerMochiArticulatedPoseTarget(
                world_from_root=self.root, pose_dofs=self.target(frame)
            ),
        )
        self.scene.step(DT)

    def pickup_in_palm(self) -> list[float]:
        return (
            self.palm.get_center_of_mass_transform().inverse()
            * self.pickup.get_center_of_mass_transform()
        ).translation.tolist()

    def contact_forces(self) -> dict[str, float]:
        """Contact force on the object from each link and the table. A hand's skin is
        part of the articulated actor, so its contacts count for the nearest link."""
        forces = {
            name: magnitude(self.pickup.get_contact_force_from_actor_world(link))
            for name, link in self.links.items()
        }
        forces["table"] = magnitude(
            self.pickup.get_contact_force_from_actor_world(self.table)
        )
        skin = self.actor.get_handle()
        for point in self.pickup.get_contact_points_world():
            if skin not in (point.actor_a, point.actor_b):
                continue
            position = (point.pos_a if point.actor_a == skin else point.pos_b).tolist()
            nearest = min(
                self.links,
                key=lambda name: distance_to_box(
                    position, self.links[name].get_aabb_world()
                ),
            )
            forces[nearest] += magnitude(point.force)
        return forces

    def segment_end(self, frame: int) -> int:
        """The end of the hold after the motion that `frame` is in."""
        for motion, start in reversed(self.segments):
            if frame > start:
                return start + MOTION_FRAMES[motion] + HOLD_FRAMES
        return LIFT_END + HOLD_FRAMES

    def resume_frames(self) -> list[int]:
        frames = list(EARLY_RESUMES)
        for motion, start in self.segments:
            frames += [
                start + MOTION_FRAMES[motion] // 4,
                start + MOTION_FRAMES[motion] // 2,
            ]
        return frames


class Recording:
    def __init__(self, scene: GraspScene, directory: Path, first_frame: int) -> None:
        self.scene = scene
        self.episode = recorder.Recorder(
            recorder.RecorderConfig(directory)
        ).begin_episode()
        self.physics = recorder.PhysicsRecorder(self.episode, scene.scene)
        self.record(first_frame)

    def record(self, frame: int) -> None:
        self.physics.record(frame, self.scene.scene.get_total_simulation_time())


@contextmanager
def recording(
    scene: GraspScene, directory: Path, first_frame: int, *, always: bool = False
) -> Iterator[Recording | None]:
    """Record a run that the restorer reads, or that is kept for viewing."""
    if not (always or os.environ.get(RECORDINGS_DIR_ENV)):
        yield None
        return
    run = Recording(scene, directory, first_frame)
    yield run
    run.episode.commit_sync()


def keep_recording(run: Recording | None, name: str) -> None:
    kept = os.environ.get(RECORDINGS_DIR_ENV)
    if run is not None and kept:
        Path(kept).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run.episode.output_path, Path(kept) / f"{name}.rrd")


class ResumeCapabilityTest(unittest.TestCase):
    def setUp(self) -> None:
        sdp.initialize(num_worker_threads=0)
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.directory = Path(temp_dir.name)

    def tearDown(self) -> None:
        gc.collect()
        sdp.shutdown()

    def test_dg5f_power_block(self) -> None:
        self.check("dg5f_power_block")

    def test_dg5f_pinch_cube(self) -> None:
        self.check("dg5f_pinch_cube")

    def test_dg5f_pinch_sphere(self) -> None:
        self.check("dg5f_pinch_sphere")

    def test_dg5f_tripod_cube(self) -> None:
        self.check("dg5f_tripod_cube")

    def test_dg5f_tripod_sphere(self) -> None:
        self.check("dg5f_tripod_sphere")

    def test_wuji2_pinch_cube(self) -> None:
        self.check("wuji2_pinch_cube")

    def test_wuji2_pinch_sphere(self) -> None:
        self.check("wuji2_pinch_sphere")

    def test_wuji2_tripod_sphere(self) -> None:
        self.check("wuji2_tripod_sphere")

    def test_wuji2_power_rod(self) -> None:
        self.check("wuji2_power_rod")

    def test_robotiq_parallel_cube(self) -> None:
        self.check("robotiq_parallel_cube")

    def test_robotiq_parallel_rod(self) -> None:
        self.check("robotiq_parallel_rod")

    def test_robotiq_parallel_sphere(self) -> None:
        self.check("robotiq_parallel_sphere")

    def check(self, name: str) -> None:
        scenario = SCENARIOS[name]
        live = GraspScene(scenario)
        reference: list[float] = []
        live_drift = 0.0
        with recording(live, self.directory / "live", 0, always=True) as run:
            assert run is not None
            for frame in range(1, live.num_frames + 1):
                live.step(frame)
                run.record(frame)
                if frame == LIFT_END:
                    reference = live.pickup_in_palm()
                    self.assert_grasp(live)
                if frame >= LIFT_END:
                    drift = math.dist(live.pickup_in_palm(), reference)
                    live_drift = max(live_drift, drift)
        keep_recording(run, f"{name}_live")
        self.assertLessEqual(live_drift, MAX_DRIFT, "the live run lost it")

        restored = GraspScene(scenario)
        restorer = recorder.EpisodeRestorer(
            EpisodeReader(run.episode.output_path), restored.scene
        )
        for start in live.resume_frames():
            with self.subTest(resume_from=start):
                restorer.restore_frame(start)
                lost = self.resume(restored, reference, start, name)
                self.assertIsNone(lost, f"resuming from {start} lost it at {lost}")

        if scenario.stale_check:
            # The check must be able to fail: an object restored from before the hand
            # is lost when the resume starts during the swing.
            swing = next(start for motion, start in live.segments if motion == "swing")
            start = swing + MOTION_FRAMES["swing"] // 4
            restorer.restore_frame(start - STALE_FRAMES)
            pickup = restored.pickup
            stale_transform = pickup.get_center_of_mass_transform()
            stale_linear = pickup.get_linear_velocity().tolist()
            stale_angular = pickup.get_angular_velocity().tolist()
            restorer.restore_frame(start)
            pickup.set_center_of_mass_transform(stale_transform)
            pickup.set_velocity(stale_linear, stale_angular)
            lost = self.resume(restored, reference, start, f"{name}_stale")
            self.assertIsNotNone(lost, "a resume with a stale object kept it")

    def assert_grasp(self, scene: GraspScene) -> None:
        forces = scene.contact_forces()

        def finger_force(finger: str) -> float:
            return sum(force for name, force in forces.items() if finger in name)

        for finger in scene.grasp.contacts:
            force = finger_force(finger)
            self.assertGreater(force, MIN_CONTACT_FORCE, f"{finger} isn't holding it")
        for finger in (*scene.grasp.idle, "table"):
            force = finger_force(finger)
            self.assertLess(force, MIN_CONTACT_FORCE, f"{finger} is touching it")

    def resume(
        self, scene: GraspScene, reference: list[float], start: int, label: str
    ) -> int | None:
        """Follow the episode from `start` to the end of its motion; return the first
        frame the object is lost."""
        lost = None
        with recording(scene, self.directory / f"{label}_{start}", start) as run:
            for frame in range(start + 1, scene.segment_end(start) + 1):
                scene.step(frame)
                if run is not None:
                    run.record(frame)
                if lost is None and frame >= LIFT_END:
                    if math.dist(scene.pickup_in_palm(), reference) > MAX_DRIFT:
                        lost = frame
        outcome = "held" if lost is None else f"lost_at_{lost}"
        keep_recording(run, f"{label}_from_{start}_{outcome}")
        return lost
