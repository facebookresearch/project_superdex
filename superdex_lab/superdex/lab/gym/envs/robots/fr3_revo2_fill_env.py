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

"""Hold a paper cup while it is filled with water: FR3 + BrainCo Revo2 with fingertip
tactile sensing.

The hand starts around an empty paper cup in a side (power) grasp: palm against the cup,
thumb up and opposed on the far side. The policy commands only the hand. The arm follows a
fixed script: it holds still while the hand closes, lifts the cup, and then holds it,
swaying gently, while water is poured in. Water is simulated as mass (with the matching
center of mass and inertia) that grows at a random rate to a random fill level, and it
is drawn in the viewer as a pour and a rising water level.

The difficulty is the grip force. An empty 15 g cup needs almost none, while a full one
(about 0.35 kg) slides out of a light grip once the arm moves. Squeezing a paper cup too
hard crushes it, and the policy is never told how much water there is. It has to read the
load from its fingertips: the tangential force a Revo2 Touch sensor reports grows with
the weight, and the ratio of tangential to normal force rises as a finger nears slipping.

Scripted baselines (``apps/envs/run_fr3_revo2_fill.py``, 12 episodes each, default
config): a fixed light grip holds none (4 drops); a fixed firm grip crushes the empty cup
every time; a fixed medium grip holds 10; a simple tactile rule holds 11 with about 20 %
less squeeze. A learned policy should beat all of them.

Action (in [-1, 1]):

- ``hand`` (6): absolute targets for the Revo2's actuated joints (thumb metacarpal, thumb
  flexion, index, middle, ring, pinky). With ``lock_thumb_opposition`` (the default) the
  metacarpal stays fully opposed and its action is ignored.

Observation:

- ``hand_qpos``, ``hand_qvel``, ``hand_target`` (6 each).
- ``tactile`` (5 x 5): per fingertip, normal force, tangential force, tangential direction
  as (cos, sin), and proximity, as in ``Fr3Revo2Env``.
- ``phase`` (3): one-hot of the arm script (closing, lifting, holding).
- ``fill`` (1, only with ``observe_fill``): the fill level, for ablations.
- ``object_in_hand`` (7, only with ``observe_object_pose``): the cup's pose in the hand
  frame (position, quaternion), as a camera would provide it.

Episode outcome (``info``): the cup is **crushed** when a fingertip presses harder than
``crush_tip_force`` or the hand's total squeeze exceeds ``crush_squeeze``; it is
**dropped** when it slides more than ``drop_distance`` in the hand. Both end the episode.
An episode that runs to the end without either, with the cup still within
``success_slip`` of where it sat in the hand when the lift ended, is a success.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import superdex.physics as physics
from scipy.spatial.transform import Rotation
from superdex.lab.gym.envs import (
    ActionSpace,
    Info,
    MochiEnv,
    ObservationSpace,
    RewardTerms,
    StructuredAction,
    StructuredObservation,
)
from superdex.lab.gym.envs.robots.fr3_revo2_env import (
    _SCENE_HANDLES,
    FINGERS,
    Fr3Revo2Env,
    Fr3Revo2EnvCfg,
)
from superdex.physics.utils.configclasses import configclass
from superdex.physics.utils.decorators import override_from
from superdex.physics.utils.transformations import make_transform, transformrt_to_numpy

# Arm joints [rad] for the side grasp: the hand's palm faces +x, thumb up, around a cup
# standing at (0.55, 0) m, with the cup's axis 53 mm in front of the palm and 95 mm along
# the fingers, 60 mm above the table. Every hand link then clears the cup by >= 3.7 mm (the
# opposed thumb by 6 mm) while the palm stays close enough to carry part of the load.
# Lifted is the same grasp 0.12 m higher; moving linearly in joint space between the two
# stays within 6 mm of a vertical lift. Solved with inverse kinematics; >= 0.67 rad from
# every joint limit.
ARM_GRASP = {
    "right": (1.1385, 1.1123, -0.7304, -1.9383, -0.6477, 1.8483, 1.7356),
    "left": (-0.7673, 0.9275, 0.3521, -1.9014, 0.9616, 1.6162, -0.3484),
}
ARM_LIFTED = {
    "right": (1.2663, 0.8494, -0.887, -2.0895, -0.7663, 1.7837, 1.6433),
    "left": (-0.823, 0.6276, 0.4089, -2.0541, 1.0367, 1.5281, -0.2383),
}

# Paper cup interior (from its collision mesh): inner bottom height, and the inner wall
# radius r(z) = CUP_INNER_RADIUS + CUP_INNER_TAPER * z [m], up to the brim.
CUP_INNER_BOTTOM = 0.0051
CUP_INNER_RADIUS = 0.0271
CUP_INNER_TAPER = 0.136
CUP_BRIM = 0.1105
WATER_DENSITY = 1000.0  # [kg/m^3]
WATER_RGB = (0.16, 0.45, 0.85)
WATER_OPACITY = 0.85
WATER_INSET = 0.0006  # [m], the water surface drawn just inside the paper wall
STREAM_RADIUS = 0.003  # [m], the pour drawn falling along the cup's axis
STREAM_TOP = 0.30  # [m] above the cup's base
_RING = 48  # sides of the drawn water and stream


def _cup_inner_radius(z: np.ndarray) -> np.ndarray:
    return CUP_INNER_RADIUS + CUP_INNER_TAPER * z


def _water_tables(num: int = 400) -> dict[str, np.ndarray]:
    """Water volume, height, center of mass and inertia as the cup fills, by slices."""
    z = np.linspace(CUP_INNER_BOTTOM, CUP_BRIM, num + 1)
    mid = 0.5 * (z[1:] + z[:-1])
    r = _cup_inner_radius(mid)
    dm = WATER_DENSITY * np.pi * r**2 * np.diff(z)
    mass = np.concatenate([[0.0], np.cumsum(dm)])
    com = np.concatenate([[CUP_INNER_BOTTOM], np.cumsum(dm * mid) / mass[1:]])
    izz = np.concatenate([[0.0], np.cumsum(0.5 * dm * r**2)])
    # Discs: r^2 / 4 about their own diameter, then shifted to the water's center.
    ixx_origin = np.concatenate([[0.0], np.cumsum(dm * (r**2 / 4 + mid**2))])
    ixx = np.maximum(ixx_origin - mass * com**2, 0.0)
    return {"level": z, "mass": mass, "com": com, "ixx": ixx, "izz": izz}


WATER = _water_tables()
CUP_CAPACITY = float(WATER["mass"][-1]) / WATER_DENSITY  # [m^3], to the brim


def _frustum(z0: float, z1: float, r0: float, r1: float) -> np.ndarray:
    """Vertices of a closed frustum along z, for :func:`_frustum_faces`."""
    angle = np.linspace(0, 2 * np.pi, _RING, endpoint=False)
    ring = np.stack([np.cos(angle), np.sin(angle), np.zeros(_RING)], axis=1)
    return np.vstack(
        [ring * r0 + [0, 0, z0], ring * r1 + [0, 0, z1], [[0, 0, z0]], [[0, 0, z1]]]
    ).astype(np.float32)


def _frustum_faces() -> np.ndarray:
    n, faces = _RING, []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i]]
        faces += [[2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return np.asarray(faces)


def _tensor(real6) -> np.ndarray:
    ixx, ixy, ixz, iyy, iyz, izz = (float(v) for v in real6)
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


def _shift(mass: float, offset: np.ndarray) -> np.ndarray:
    """Parallel-axis term moving an inertia tensor from the center of mass by offset."""
    return mass * (float(offset @ offset) * np.eye(3) - np.outer(offset, offset))


@configclass
class Fr3Revo2FillEnvCfg(Fr3Revo2EnvCfg):
    """Options for the hold-while-filling environment."""

    steps_per_episode: int = 250  # 10 s at 25 Hz
    object_name: str = "paper_cup"
    object_xy_noise: float = 0.002
    """The hand starts around the cup with >= 3.7 mm of clearance, so the cup's placement
    noise stays below that (<= 2.8 mm diagonally)."""

    # Arm script [s]: hold while the hand closes, then lift, then hold.
    close_duration: float = 1.2
    lift_duration: float = 1.2
    arm_shake_amplitude: float = 0.03
    """Amplitude of a random sinusoidal sway of the arm joints while holding [rad], as
    when a cup is carried while it fills. Standing still, the tapered cup wedges in the
    hand and almost any grip holds it; swaying adds loads that grow with the water's
    mass, so the grip has to follow the fill."""
    arm_shake_frequency: tuple[float, float] = (0.5, 2.0)
    """Range of the wobble frequency [Hz]."""

    # Filling, randomized per episode.
    fill_start: tuple[float, float] = (2.8, 3.6)
    """When pouring starts [s]."""
    fill_duration: tuple[float, float] = (3.0, 5.0)
    """How long pouring takes [s]."""
    fill_fraction: tuple[float, float] = (0.2, 0.9)
    """Final fill, as a fraction of the cup's brim volume (0.41 l)."""
    friction: tuple[float, float] = (0.5, 0.9)
    """Coulomb friction between the cup and the hand (a wet cup is slipperier)."""
    friction_falloff_velocity: float = 1e-4
    """Sliding speed below which friction is smoothed out [m/s]. The engine's default
    (1 cm/s) lets a held cup creep down by millimeters per second; a paper cup held
    within its friction cone does not move."""

    lock_thumb_opposition: bool = True
    """Keep the thumb metacarpal fully opposed (its action is ignored)."""
    observe_fill: bool = False
    """Observe the fill level (privileged; for ablations)."""
    observe_object_pose: bool = False
    """Observe the cup's pose in the hand frame, as vision would provide."""
    observe_tactile: bool = True
    """Observe the fingertip sensors (disable for the no-touch ablation)."""

    # Outcome thresholds.
    crush_tip_force: float = 6.0
    """A single fingertip pressing harder than this dents the cup wall [N]."""
    crush_squeeze: float = 25.0
    """Total normal force from the hand on the cup that crushes it [N]."""
    crush_time_constant: float = 0.1
    """Crushing takes sustained force: both limits apply to the forces low-pass
    filtered with this time constant [s], so a brief impact does not count."""
    drop_distance: float = 0.03
    """Slide of the cup in the hand since it closed that counts as a drop [m]."""
    success_slip: float = 0.005
    """Largest slide in the hand while holding (after the lift, as the water pours in)
    still counted as a secure hold at the end [m]."""

    # Reward weights.
    hold_weight: float = 0.1
    grip_weight: float = 0.1
    slip_weight: float = 20.0
    squeeze_weight: float = 0.2
    failure_penalty: float = 10.0
    success_bonus: float = 5.0
    action_rate_weight: float = 0.01


@dataclass
class _Water:
    """Water state of one env (the cup may be shared between envs)."""

    start: float = 3.0  # [s]
    duration: float = 4.0  # [s]
    target_mass: float = 0.0  # [kg]
    mass: float = 0.0  # [kg]


# Per scene: the render-only actors drawing the water in the cup and the pour.
_WATER_ACTORS: dict[str, tuple[physics.Actor, physics.Actor]] = {}


class Fr3Revo2FillEnv(Fr3Revo2Env):
    """Hold a paper cup steady, without crushing it, while water is poured in."""

    SCENE_PREFIX = "fr3_revo2_fill"

    def __init__(self, cfg: Fr3Revo2FillEnvCfg | dict[str, Any]):
        if not isinstance(cfg, Fr3Revo2FillEnvCfg):
            cfg = Fr3Revo2FillEnvCfg(**cfg)
        if cfg.object_name != "paper_cup":
            raise ValueError("Fr3Revo2FillEnv fills the paper cup")
        super().__init__(cfg)
        self._cfg: Fr3Revo2FillEnvCfg = cfg

        obj = self._object
        self._cup_mass = float(obj.get_mass())
        self._cup_com = np.asarray(obj.get_rigid_center_of_mass_local(), dtype=float)
        self._cup_inertia = _tensor(obj.get_rigid_moment_of_inertia_local())
        self._cup_contact = obj.get_contact_params()
        self._cup_contact.friction_falloff_vel = cfg.friction_falloff_velocity
        self._hand_links = {
            h.value
            for h in self._agent.get_nested_link_actors()
            if f"/{cfg.hand_side}_" in self._scene.get_actor(h).get_name()
        }
        if self._scene_key not in _WATER_ACTORS:
            raise RuntimeError("the fill scene has no water actors")
        self._water_actors = _WATER_ACTORS[self._scene_key]
        self._water = _Water()
        self._friction = float(np.mean(cfg.friction))
        self._slip_ref: tuple[np.ndarray, np.ndarray] | None = None
        self._slip = 0.0
        self._slip_speed = 0.0
        self._crush_load = np.zeros(2)
        self._shake = (np.zeros(7), np.zeros(7), np.zeros(7))
        lo, hi = self._dof_min[self._hand_dofs], self._dof_max[self._hand_dofs]
        self._hand_lo, self._hand_hi = lo, hi
        self._prev_hand_action = np.zeros(6)

        # Only the hand is actuated by the policy.
        self._setup_action_space(hand=ActionSpace(-1.0, 1.0, (6,), dtype=np.float32))
        observation_space = {
            "hand_qpos": ObservationSpace(-np.inf, np.inf, (6,), dtype=np.float32),
            "hand_qvel": ObservationSpace(-np.inf, np.inf, (6,), dtype=np.float32),
            "hand_target": ObservationSpace(-1.0, 1.0, (6,), dtype=np.float32),
            "phase": ObservationSpace(0.0, 1.0, (3,), dtype=np.float32),
        }
        if cfg.observe_tactile:
            observation_space["tactile"] = ObservationSpace(
                -np.inf, np.inf, (len(FINGERS) * 5,), dtype=np.float32
            )
        if cfg.observe_fill:
            observation_space["fill"] = ObservationSpace(0.0, 1.0, (1,), np.float32)
        if cfg.observe_object_pose:
            observation_space["object_in_hand"] = ObservationSpace(
                -np.inf, np.inf, (7,), dtype=np.float32
            )
        self._setup_observation_space(**observation_space)

        if self._renderer:
            self._renderer.set_camera_view(
                look_from=[0.95, -0.55, 0.35], look_at=[0.5, 0.0, 0.1]
            )

    ####################################################################################
    # Scene
    ####################################################################################

    @classmethod
    def arm_home(cls, side: str) -> tuple[float, ...]:
        return ARM_GRASP[side]

    def _init_scene(self, cfg: Fr3Revo2FillEnvCfg):
        super()._init_scene(cfg)
        # Start with the thumb already opposed: swinging it in would sweep it through
        # the cup. The fingers start open.
        metacarpal = self._hand_dofs[0]
        self._initial_pose[metacarpal] = self._dof_max[metacarpal]
        for follower, leader, multiplier in self._mimic:
            self._initial_pose[follower] = multiplier * self._initial_pose[leader]

    @classmethod
    def _build_scene(cls, scene_key: str, cfg: Fr3Revo2FillEnvCfg):
        scene, agent, cleanups = super()._build_scene(scene_key, cfg)
        # The cup's contacts give the hand's squeeze.
        _SCENE_HANDLES[scene_key].obj.register_query(physics.QueryType.CONTACT_POINTS)
        # Water is only drawn: actors without colliders, whose meshes and poses the
        # renderer takes from the cup and its fill level each frame.
        actors = []
        for name in ("water", "water_pour"):
            model = physics.ModelData()
            model.mesh = physics.MeshData(
                nodes_per_element=3,
                coordinates=_frustum(0.0, 0.05, 0.02, 0.02).ravel(),
                connectivity=_frustum_faces().ravel(),
            )
            actors.append(
                scene.create_rigid_actor(
                    name=name,
                    shape=physics.create_model_shape(model),
                    is_static=True,
                    collider_type=physics.ColliderType.NONE,
                )
            )
        _WATER_ACTORS[scene_key] = tuple(actors)

        def cleanup():
            _WATER_ACTORS.pop(scene_key, None)

        return scene, agent, [cleanup, *cleanups]

    @override_from(MochiEnv)
    def _reset_scene(self):
        cfg = self._cfg
        super()._reset_scene()
        rng = self.np_random
        self._water = _Water(
            start=rng.uniform(*cfg.fill_start),
            duration=rng.uniform(*cfg.fill_duration),
            target_mass=WATER_DENSITY * CUP_CAPACITY * rng.uniform(*cfg.fill_fraction),
        )
        self._friction = rng.uniform(*cfg.friction)
        self._slip_ref = None
        self._slip = 0.0
        self._slip_speed = 0.0
        self._crush_load = np.zeros(2)  # filtered (max fingertip force, squeeze)
        self._prev_hand_action = np.zeros(6)
        freq = rng.uniform(*cfg.arm_shake_frequency, 7)
        self._shake = (
            cfg.arm_shake_amplitude * rng.uniform(0.5, 1.0, 7),
            2 * np.pi * freq,
            rng.uniform(0, 2 * np.pi, 7),
        )
        self._apply_cup_properties()

    def _apply_cup_properties(self) -> None:
        """This env's water mass and friction onto the (possibly shared) cup."""
        water = self._water.mass
        cup_mass = self._cup_mass + water
        k = np.clip(water / WATER["mass"][-1], 0.0, 1.0) * (len(WATER["mass"]) - 1)
        water_com = np.array(
            [0.0, 0.0, np.interp(k, np.arange(len(WATER["com"])), WATER["com"])]
        )
        com = (self._cup_mass * self._cup_com + water * water_com) / cup_mass
        inertia = self._cup_inertia + _shift(self._cup_mass, self._cup_com - com)
        if water > 0.0:
            ixx = np.interp(k, np.arange(len(WATER["ixx"])), WATER["ixx"])
            izz = np.interp(k, np.arange(len(WATER["izz"])), WATER["izz"])
            inertia = (
                inertia + np.diag([ixx, ixx, izz]) + _shift(water, water_com - com)
            )
        obj = self._object
        if abs(obj.get_mass() - cup_mass) > 1e-7:
            # The engine keeps the center of mass where it was, so moving it moves the
            # cup: put the cup back, with the velocity of its new center of mass.
            pose = obj.get_root_transform()
            old_com = np.asarray(obj.get_rigid_center_of_mass_local())
            v = np.asarray(obj.get_linear_velocity())
            w = np.asarray(obj.get_angular_velocity())
            obj.set_inertia_properties(
                cup_mass,
                com.tolist(),
                [
                    inertia[0, 0],
                    inertia[0, 1],
                    inertia[0, 2],
                    inertia[1, 1],
                    inertia[1, 2],
                    inertia[2, 2],
                ],
            )
            obj.set_root_transform(pose)
            R = Rotation.from_quat(list(pose.rotation)).as_matrix()
            obj.set_velocity(
                (v + np.cross(w, R @ (com - old_com))).tolist(), w.tolist()
            )
        contact = self._cup_contact
        current = obj.get_contact_params()
        if current.coulomb_friction_coefficient != np.float32(
            self._friction
        ) or current.friction_falloff_vel != np.float32(contact.friction_falloff_vel):
            contact.coulomb_friction_coefficient = self._friction
            obj.set_contact_params(contact)

    ####################################################################################
    # Stepping
    ####################################################################################

    def time(self) -> float:
        """Time since the start of the episode [s]."""
        return self._step_count * self._control_dt

    def phase(self) -> int:
        """Arm script phase: 0 closing, 1 lifting, 2 holding."""
        t, cfg = self.time(), self._cfg
        if t < cfg.close_duration:
            return 0
        return 1 if t < cfg.close_duration + cfg.lift_duration else 2

    def water_mass(self) -> float:
        return self._water.mass

    def fill_level(self) -> float:
        """Water volume as a fraction of the brim volume."""
        return self._water.mass / (WATER_DENSITY * CUP_CAPACITY)

    def _arm_script(self) -> np.ndarray:
        cfg, side = self._cfg, self._cfg.hand_side
        s = np.clip((self.time() - cfg.close_duration) / cfg.lift_duration, 0.0, 1.0)
        s = s * s * (3 - 2 * s)  # smoothstep
        q = np.asarray(ARM_GRASP[side]) + s * (
            np.asarray(ARM_LIFTED[side]) - np.asarray(ARM_GRASP[side])
        )
        if self.phase() == 2:
            amplitude, omega, phase = self._shake
            t = self.time() - cfg.close_duration - cfg.lift_duration
            q = q + amplitude * np.sin(omega * t + phase) * min(1.0, t)
        return q

    @override_from(MochiEnv)
    def _simulate(self, action: StructuredAction):
        cfg = self._cfg
        hand = np.clip(np.asarray(action["hand"], dtype=np.float64), -1.0, 1.0)
        self._action_rate = float(np.sum((hand - self._prev_hand_action) ** 2))
        self._prev_hand_action = hand.copy()
        goal = self._goal
        goal[self._arm_dofs] = self._arm_script()
        lo, hi = self._hand_lo, self._hand_hi
        goal[self._hand_dofs] = lo + 0.5 * (hand + 1.0) * (hi - lo)
        if cfg.lock_thumb_opposition:
            goal[self._hand_dofs[0]] = hi[0]
        np.clip(goal, self._dof_min, self._dof_max, out=goal)

        # Pour: the water mass ramps up linearly.
        w = self._water
        fraction = np.clip((self.time() - w.start) / w.duration, 0.0, 1.0)
        w.mass = fraction * w.target_mass
        self._substep = 0
        MochiEnv._simulate(self, action)

    @override_from(MochiEnv)
    def _apply_action(self, action: StructuredAction):
        if self._substep == 0:
            # After the scene state is restored: re-apply this env's cup properties.
            self._apply_cup_properties()
        self._substep += 1
        dt = self._sim_dt
        # The arm script is smooth and slow; no rate limit beyond the hand's.
        self._target[self._arm_dofs] = self._goal[self._arm_dofs]
        delta = self._goal[self._hand_dofs] - self._target[self._hand_dofs]
        max_step = self._hand_speed * dt
        self._target[self._hand_dofs] += np.clip(delta, -max_step, max_step)
        self._apply_targets()

    ####################################################################################
    # Observation, reward, termination
    ####################################################################################

    def _object_in_hand(self) -> tuple[np.ndarray, np.ndarray]:
        Tw = self._wrist.get_root_transform()
        Rw = Rotation.from_quat(list(Tw.rotation)).as_matrix()
        To = self._object.get_root_transform()
        Ro = Rotation.from_quat(list(To.rotation)).as_matrix()
        pos = Rw.T @ (np.asarray(To.translation) - np.asarray(Tw.translation))
        return pos, Rw.T @ Ro

    def squeeze(self) -> float:
        """Total normal force the hand presses on the cup with [N] (ground truth)."""
        total, cup = 0.0, self._object.get_handle().value
        try:
            contacts = self._object.get_contact_points_world()
        except physics.Error:  # no step since the query was registered (first reset)
            return 0.0
        for cp in contacts:
            a, b = cp.actor_a.value, cp.actor_b.value
            other = b if a == cup else a
            if other not in self._hand_links:
                continue
            f, n = cp.force.tolist(), cp.normal.tolist()
            total += abs(f[0] * n[0] + f[1] * n[1] + f[2] * n[2])
        return total

    @override_from(MochiEnv)
    def _make_observation(self) -> tuple[StructuredObservation, Info]:
        cfg = self._cfg
        base_obs, base_info = super()._make_observation()
        # Noise-free fingertip forces decide crushing; the policy sees the noisy ones.
        true_normals = {f: self._readings[f].normal_force for f in FINGERS}

        pos, rot = self._object_in_hand()
        phase = self.phase()
        if phase == 0:
            self._slip_ref = (pos, rot)
        if phase < 2:
            self._hold_ref = pos
        ref_pos, ref_rot = self._slip_ref
        slip = float(np.linalg.norm(pos - ref_pos))
        hold_slip = float(np.linalg.norm(pos - self._hold_ref))
        self._slip_speed = abs(slip - self._slip) / self._control_dt
        self._slip = slip
        tilt = float(np.degrees(Rotation.from_matrix(ref_rot.T @ rot).magnitude()))
        squeeze = self.squeeze()

        lo, hi = self._hand_lo, self._hand_hi
        target = 2.0 * (self._target[self._hand_dofs] - lo) / (hi - lo) - 1.0
        obs = {
            "hand_qpos": base_obs["hand_qpos"],
            "hand_qvel": base_obs["hand_qvel"],
            "hand_target": target,
            "phase": np.eye(3)[phase],
        }
        if cfg.observe_tactile:
            obs["tactile"] = base_obs["tactile"]
        if cfg.observe_fill:
            obs["fill"] = [self.fill_level()]
        if cfg.observe_object_pose:
            obs["object_in_hand"] = np.concatenate(
                [pos, Rotation.from_matrix(rot).as_quat()]
            )
        obs = {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}

        weight = (self._cup_mass + self._water.mass) * 9.81
        load = np.array([max(true_normals.values()), squeeze])
        alpha = self._control_dt / (cfg.crush_time_constant + self._control_dt)
        self._crush_load += alpha * (load - self._crush_load)
        crushed = bool(
            self._crush_load[0] > cfg.crush_tip_force
            or self._crush_load[1] > cfg.crush_squeeze
        )
        dropped = bool(phase > 0 and slip > cfg.drop_distance)
        # The last step of an episode that neither crushed nor dropped the cup.
        success = bool(
            not (crushed or dropped)
            and self._step_count == self._steps_per_episode
            and hold_slip <= cfg.success_slip
        )
        info = {
            **{k: base_info[k] for k in ("object_height", "fingers_in_contact")},
            "tactile_normal_force": base_info["tactile_normal_force"],
            "tactile_tangential_force": dict(
                zip(
                    FINGERS, base_obs["tactile"].reshape(len(FINGERS), 5)[:, 1].tolist()
                )
            ),
            "tactile_direction": dict(
                zip(
                    FINGERS,
                    base_obs["tactile"].reshape(len(FINGERS), 5)[:, 2:4].tolist(),
                )
            ),
            "tactile_proximity": base_info["tactile_proximity"],
            # Gravity's direction in each pad's plane, from the hand's kinematics.
            "tactile_gravity_direction": {
                f: self._pad_gravity_direction(f) for f in FINGERS
            },
            "phase": phase,
            "fill_level": self.fill_level(),
            "water_mass": self._water.mass,
            "friction": self._friction,
            "slip": slip,
            "hold_slip": hold_slip,
            "slip_speed": self._slip_speed,
            "tilt_deg": tilt,
            "squeeze": squeeze,
            "max_tip_force": max(true_normals.values()),
            "weight": weight,
            # Squeeze in units of the least that friction needs to carry the weight.
            "grip_ratio": squeeze * self._friction / weight,
            "crushed": crushed,
            "dropped": dropped,
            "is_success": success,
        }
        return obs, info

    def _pad_gravity_direction(self, finger: str) -> list[float]:
        R, _ = self._sensors[finger].world_from_sensor()
        down = R.T @ np.array([0.0, 0.0, -1.0])
        return (down[:2] / max(np.linalg.norm(down[:2]), 1e-9)).tolist()

    @override_from(MochiEnv)
    def _compute_reward_terms(
        self, action: StructuredAction, observation: StructuredObservation, info: Info
    ) -> RewardTerms:
        cfg = self._cfg
        failed = info["crushed"] or info["dropped"]
        holding = info["phase"] > 0 and not failed
        touching = info["fingers_in_contact"]
        grip = min(len(touching), 3) / 3.0 if "thumb" in touching else 0.0
        return {
            "hold": cfg.hold_weight * holding,
            "grip": cfg.grip_weight * grip * (info["phase"] == 0),
            "slip": -cfg.slip_weight * info["slip_speed"] * (info["phase"] > 0),
            "squeeze": -cfg.squeeze_weight
            * min(info["squeeze"] / cfg.crush_squeeze, 1.0),
            "failure": -cfg.failure_penalty * failed,
            "success": cfg.success_bonus * info["is_success"],
            "action_rate": -cfg.action_rate_weight * self._action_rate,
        }

    @override_from(MochiEnv)
    def _check_stop_criteria(
        self,
        observation: StructuredObservation,
        action: StructuredAction,
        reward: RewardTerms,
        info: Info,
    ):
        MochiEnv._check_stop_criteria(
            self, observation=observation, action=action, reward=reward, info=info
        )
        if info["crushed"]:
            info["terminated_reason"] = "Crushed"
            self._terminated = True
        elif info["dropped"]:
            info["terminated_reason"] = "Dropped"
            self._terminated = True

    ####################################################################################
    # Rendering
    ####################################################################################

    @override_from(MochiEnv)
    def _reset_renderer(self):
        super()._reset_renderer()
        cup = self._object

        # Draw the water in the cup's frame rather than at the (static) actors' pose.
        def cup_transform():
            pos, rotvec = transformrt_to_numpy(cup.get_root_transform())
            return make_transform(pos, rotvec)

        for renderer in self._water_renderers():
            renderer.set_front_face_color(WATER_RGB)
            renderer.set_back_face_policy(type(renderer).BackFacePolicy.IDENTICAL)
            renderer.set_smooth_shading(True)
            renderer.set_transparency(WATER_OPACITY)
            renderer._get_actor_transform = cup_transform

    @override_from(MochiEnv)
    def _update_renderer(self):
        super()._update_renderer()
        renderers = self._water_renderers()
        if not renderers:
            return
        water, pour = renderers
        mass = self._water.mass
        k = np.clip(mass / WATER["mass"][-1], 0.0, 1.0) * (len(WATER["mass"]) - 1)
        top = float(np.interp(k, np.arange(len(WATER["level"])), WATER["level"]))
        w = self._water
        pouring = w.start <= self.time() < w.start + w.duration
        for renderer, visible in ((water, mass > 1e-4), (pour, pouring)):
            # Enabling the renderer turns on all of its parts; keep only the surface.
            renderer.set_enabled(visible)
            renderer.set_enable_edges(False)
            renderer.set_enable_nodes(False)
            renderer.set_enable_axes(False)
        if mass > 1e-4:
            water.set_local_coordinates(
                _frustum(
                    CUP_INNER_BOTTOM,
                    top,
                    _cup_inner_radius(CUP_INNER_BOTTOM) - WATER_INSET,
                    _cup_inner_radius(top) - WATER_INSET,
                )
            )
        if pouring:
            pour.set_local_coordinates(
                _frustum(
                    max(top, CUP_INNER_BOTTOM),
                    STREAM_TOP,
                    STREAM_RADIUS,
                    STREAM_RADIUS,
                )
            )

    def _water_renderers(self) -> list:
        if self._renderer is None:
            return []
        renderers = [self._renderer.get_actor_renderer(a) for a in self._water_actors]
        return [] if None in renderers else renderers
