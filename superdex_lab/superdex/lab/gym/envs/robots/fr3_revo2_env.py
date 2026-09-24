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

"""Franka Research 3 with a BrainCo Revo2 hand and fingertip tactile sensing.

Task: grasp a box (can-sized by default) from the table and lift it. The robot is the
``arm_hand_combos/fr3_v2_revo2`` bot, with a ``TACTILE_PAD`` taxel-array sensor on each
of the five fingertip pads.

Control is joint-space through one implicit pose controller (solved inside the physics
step, so it holds high gains stably). Robot links carry no gravity, standing in for the
FR3's gravity compensation and the Revo2's position servos.

Action (all in [-1, 1]):

- ``arm`` (7): joint-target increments, scaled by ``arm_joint_step`` [rad] per step.
- ``hand`` (6): absolute targets for the Revo2's actuated joints (thumb metacarpal,
  thumb flexion, index, middle, ring, pinky), mapped onto each joint's range. The five
  distal joints follow through the hand's mimic transmissions. Targets move no faster
  than the Revo2's rated joint speeds.

Observation:

- ``arm_qpos``, ``arm_qvel`` (7 each); ``hand_qpos``, ``hand_qvel`` (6 each, actuated).
- ``grasp_point`` (3): the point between the fingers and thumb, world frame [m].
- ``object_pos`` (3), ``object_quat`` (4, [x, y, z, w]), ``object_to_grasp`` (3).
- ``tactile`` (5 x 7): per finger (thumb, index, middle, ring, pinky) the pad's net
  force in its sensor frame (3), normal and shear magnitude (2), and center of
  pressure (2). See :class:`superdex.lab.sensors.TactileReading`.
- ``taxels`` (256): the five pads' taxel maps (8 x 8 thumb, 8 x 6 fingers), flattened
  in finger order (only when ``include_taxels``).

The arm starts palm down with the fingers pointing sideways, the grasp point hovering
0.2 m above the object, and the hand open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import superdex.physics as physics
import superdex.robotics as robotics
import trimesh
from scipy.spatial.transform import Rotation
from superdex.lab.gym.envs import (
    ActionSpace,
    Info,
    MochiEnv,
    MochiEnvCfg,
    ObservationSpace,
    RewardTerms,
    StructuredAction,
    StructuredObservation,
)
from superdex.lab.gym.utils import mochi_helpers
from superdex.lab.gym.utils.bot_loading import load_bot_prefab
from superdex.lab.sensors import (
    TactilePadSensor,
    find_tactile_pad_sensors,
    register_tactile_pad_sensor,
)
from superdex.physics.utils.configclasses import configclass
from superdex.physics.utils.coordinate_systems import CoordinateSystem
from superdex.physics.utils.decorators import override_from

COMBO_BOT = "bots/arm_hand_combos/fr3_v2_revo2/fr3_v2_revo2_{side}.superdex_bot"
ARM_POSE_CONTROLLER = "bots/arms/fr3_v2/control/fr3_v2_pose.superdex_controller"
ARM_BOT = "bots/arms/fr3_v2/fr3_v2.superdex_bot"

ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# The Revo2's six actuators, in action order ({side} is "left" or "right").
HAND_ACTUATED_JOINTS = (
    "{side}_thumb_metacarpal_joint",
    "{side}_thumb_proximal_joint",
    "{side}_index_proximal_joint",
    "{side}_middle_proximal_joint",
    "{side}_ring_proximal_joint",
    "{side}_pinky_proximal_joint",
)
# Rated joint speeds from the Revo2 URDF [rad/s], in action order.
HAND_JOINT_SPEEDS = (2.6175, 2.5303, 2.2685, 2.2685, 2.2685, 2.2685)

# Home: palm down, fingers pointing sideways (-y for the right hand, +y for the left),
# grasp point hovering 0.2 m above (0.55, 0) m. Solved once with IK; of the palm-down
# poses this is the best conditioned (>= 0.9 rad from every joint limit here and at the
# table), whereas fingers-forward sits within 0.15 rad of a limit.
ARM_HOME = {
    "right": (1.0872, 0.7094, -0.7974, -1.9568, -0.9421, 1.6616, 0.0043),
    "left": (-0.7027, 0.565, 0.3502, -1.9335, 1.1427, 1.4588, 1.4699),
}

# Point between the curled fingers and the opposed thumb, in the hand base frame [m]
# (+x is the palm normal, +z runs along the fingers).
GRASP_POINT_IN_HAND = (0.045, 0.0, 0.085)

# Implicit PD gains of the Revo2 actuators [N m/rad, N m s/rad]; the output saturates
# at each joint's rated effort.
HAND_STIFFNESS = 3.0
HAND_DAMPING = 0.03

# Object contact matches the hand's (see tools/build_revo2_assets.py).
OBJECT_CONTACT = dict(
    penalty_smoothing_half_distance=0.0005, penalty_threshold_default=0.0005
)


@configclass
class Fr3Revo2EnvCfg(MochiEnvCfg):
    """Options for the FR3 + Revo2 lift environment."""

    control_frequency: int = 25
    simulation_frequency: int = 250
    steps_per_episode: int = 250
    # SuperDex robots are X-forward, Y-left, Z-up; the viewer's default is Y-up.
    render_coordinate_system: CoordinateSystem | str | None = CoordinateSystem(
        right="-y", up="+z", forward="+x"
    )

    hand_side: str = "right"
    """Which Revo2 is mounted: ``"right"`` or ``"left"``."""
    include_taxels: bool = True
    """Whether to add the flattened taxel maps to the observation."""
    tactile_noise_std: float = 0.0
    """Gaussian noise on each taxel [N], drawn from the env's seeded generator (the
    sensors themselves are shared between envs that share a scene, so they stay
    noise-free). Readings are clipped at zero; the summary features are noise-free."""
    arm_joint_step: float = 0.04
    """Largest arm joint-target change per control step [rad]."""
    hand_speed_scale: float = 1.0
    """Fraction of the Revo2's rated joint speeds the hand targets may move at."""

    object_extents: tuple[float, float, float] = (0.045, 0.045, 0.09)
    """Box size (x, y, z) [m]. Can-sized by default: a palm-down grasp needs the object
    to stand taller than the opposed thumb reaches below the palm (~4 cm)."""
    object_mass: float = 0.1
    """Mass of the object [kg]."""
    object_friction: float = 0.8
    """Coulomb friction coefficient of the object and the table."""
    object_xy: tuple[float, float] = (0.55, 0.0)
    """Nominal object position on the table [m]."""
    object_xy_noise: float = 0.02
    """Half-width of the uniform randomization of the object position [m]."""
    arm_pose_noise: float = 0.0
    """Half-width of the uniform randomization of the arm's initial joints [rad]."""

    lift_height: float = 0.1
    """Lift above the resting height that counts as success [m]."""
    contact_force_threshold: float = 0.05
    """Pad normal force above which a finger counts as touching [N]."""
    reach_weight: float = 1.0
    contact_weight: float = 0.25
    lift_weight: float = 5.0
    success_bonus: float = 2.0
    action_weight: float = 0.01
    terminate_on_success: bool = False


@dataclass
class _SceneHandles:
    """Objects created with a scene, shared by every env that shares it."""

    bot: robotics.Bot
    controller: robotics.ControllerMochiArticulatedPose
    sensors: dict[str, TactilePadSensor]
    obj: physics.Actor


# Scene factories run inside MochiEnv._load_scene, which only returns the scene and the
# agent; the rest of what a factory builds is handed over through here.
_SCENE_HANDLES: dict[str, _SceneHandles] = {}


class Fr3Revo2Env(MochiEnv):
    """Franka Research 3 + BrainCo Revo2 with fingertip tactile sensing."""

    def __init__(self, cfg: Fr3Revo2EnvCfg | dict[str, Any]):
        if not isinstance(cfg, Fr3Revo2EnvCfg):
            cfg = Fr3Revo2EnvCfg(**cfg)
        if cfg.hand_side not in ("right", "left"):
            raise ValueError(
                f"hand_side must be 'right' or 'left', got {cfg.hand_side!r}"
            )
        super().__init__(cfg)
        self._cfg = cfg
        self._init_scene(cfg)

        num_taxels = sum(int(np.prod(self._sensors[f].shape)) for f in FINGERS)
        observation_space = {
            "arm_qpos": ObservationSpace(-np.inf, np.inf, (7,), dtype=np.float32),
            "arm_qvel": ObservationSpace(-np.inf, np.inf, (7,), dtype=np.float32),
            "hand_qpos": ObservationSpace(-np.inf, np.inf, (6,), dtype=np.float32),
            "hand_qvel": ObservationSpace(-np.inf, np.inf, (6,), dtype=np.float32),
            "grasp_point": ObservationSpace(-np.inf, np.inf, (3,), dtype=np.float32),
            "object_pos": ObservationSpace(-np.inf, np.inf, (3,), dtype=np.float32),
            "object_quat": ObservationSpace(-1.0, 1.0, (4,), dtype=np.float32),
            "object_to_grasp": ObservationSpace(
                -np.inf, np.inf, (3,), dtype=np.float32
            ),
            "tactile": ObservationSpace(
                -np.inf, np.inf, (len(FINGERS) * 7,), dtype=np.float32
            ),
        }
        if cfg.include_taxels:
            observation_space["taxels"] = ObservationSpace(
                0.0, np.inf, (num_taxels,), dtype=np.float32
            )
        self._setup_observation_space(**observation_space)
        self._setup_action_space(
            arm=ActionSpace(-1.0, 1.0, (7,), dtype=np.float32),
            hand=ActionSpace(-1.0, 1.0, (6,), dtype=np.float32),
        )

        if self._renderer:
            self._renderer.set_camera_view(
                look_from=[1.3, -0.9, 0.8], look_at=[0.45, 0, 0.2]
            )

    ####################################################################################
    # Scene
    ####################################################################################

    def _init_scene(self, cfg: Fr3Revo2EnvCfg):
        scene_key = (
            f"fr3_revo2_{cfg.hand_side}_{cfg.object_extents}_{cfg.object_mass}_"
            f"{cfg.object_friction}"
        )
        self._load_scene(scene_key, lambda: self._build_scene(scene_key, cfg))
        handles = _SCENE_HANDLES[scene_key]
        self._bot = handles.bot
        self._controller = handles.controller
        self._sensors = handles.sensors
        self._object = handles.obj

        # DOF bookkeeping. Moving joints are all 1-DOF revolute, in joint order.
        prefab = self._bot.get_bot_prefab()
        dof_names = [
            j.name
            for j in prefab.joints
            if j.type == physics.ArticulatedJointType.REVOLUTE
        ]
        side = cfg.hand_side
        self._arm_dofs = np.array([dof_names.index(n) for n in ARM_JOINTS])
        self._hand_dofs = np.array(
            [dof_names.index(n.format(side=side)) for n in HAND_ACTUATED_JOINTS]
        )
        joint_to_dof = {
            i: dof_names.index(j.name)
            for i, j in enumerate(prefab.joints)
            if j.type == physics.ArticulatedJointType.REVOLUTE
        }
        # Followers of the mimic couplings: target = multiplier * leader target.
        self._mimic = [
            (
                joint_to_dof[t.joint_indices[1]],
                joint_to_dof[t.joint_indices[0]],
                float(t.joint_coefficients[0]),
            )
            for t in prefab.linear_transmissions
        ]
        num_dofs = self._agent.get_num_dofs()
        limits = mochi_helpers.get_articulated_dof_limits(self._agent)
        self._dof_min, self._dof_max = limits[:, 0], limits[:, 1]
        self._hand_speed = np.asarray(HAND_JOINT_SPEEDS) * cfg.hand_speed_scale

        wrist_name = f"{side}_base_link"
        self._wrist = next(
            self._scene.get_actor(h)
            for h in self._agent.get_nested_link_actors()
            if self._scene.get_actor(h).get_name().endswith("/" + wrist_name)
        )

        self._initial_pose = np.zeros(num_dofs, dtype=np.float32)
        self._initial_pose[self._arm_dofs] = ARM_HOME[side]
        self._initial_velocity = np.zeros(num_dofs, dtype=np.float32)
        self._target = self._initial_pose.astype(np.float64)
        self._goal = self._target.copy()
        self._pose_target = robotics.ControllerMochiArticulatedPoseTarget()
        self._pose_target.world_from_root = self._agent.get_root_transform()
        self._pose_obsv = robotics.ControllerMochiArticulatedPoseObsv()
        self._object_rest_z = cfg.object_extents[2] / 2

    @staticmethod
    def _build_scene(scene_key: str, cfg: Fr3Revo2EnvCfg):
        scene = physics.create_scene(scene_key)
        scene.set_gravity([0.0, 0.0, -9.81])
        context = robotics.create_context()
        register_tactile_pad_sensor(context)

        prefab = load_bot_prefab(
            mochi_helpers.resolve_bot_asset(COMBO_BOT.format(side=cfg.hand_side))
        )
        for i in range(len(prefab.links)):
            prefab.links[i].has_gravity = False
        pose = list(prefab.default_pose)
        pose[:7] = ARM_HOME[cfg.hand_side]
        prefab.default_pose = pose
        bot = mochi_helpers.create_bot(scene, prefab, context)
        try:
            controller = Fr3Revo2Env._create_controller(bot)
            sensors = find_tactile_pad_sensors(bot, context)
            sensors = {name.removesuffix("_tactile"): s for name, s in sensors.items()}
            missing = set(FINGERS) - set(sensors)
            if missing:
                raise RuntimeError(f"missing tactile sensors: {sorted(missing)}")

            ground = physics.ContactParams(
                coulomb_friction_coefficient=cfg.object_friction
            )
            mochi_helpers.create_ground_plane(
                scene, physics.Real3(0, 0, 1), 0.0, ground
            )
            obj = Fr3Revo2Env._create_object(scene, cfg)
        except Exception:
            mochi_helpers.destroy_bot(scene, bot)
            raise

        _SCENE_HANDLES[scene_key] = _SceneHandles(bot, controller, sensors, obj)

        def cleanup():
            _SCENE_HANDLES.pop(scene_key, None)
            mochi_helpers.destroy_bot(scene, bot)

        return scene, bot.get_articulated_actor(), [cleanup]

    @staticmethod
    def _create_controller(
        bot: robotics.Bot,
    ) -> robotics.ControllerMochiArticulatedPose:
        """One implicit pose controller: the FR3's tuned gains on the arm, servo gains on
        the Revo2's actuated joints, none on the mimic followers or fixed joints."""
        arm_params = robotics.ControllerMochiArticulatedPoseParams.load_from_file(
            str(mochi_helpers.resolve_bot_asset(ARM_POSE_CONTROLLER))
        ).pose_controller_params
        arm_names = [
            j.name
            for j in robotics.load_bot_prefab_from_file(
                str(mochi_helpers.resolve_bot_asset(ARM_BOT))
            ).joints
        ]
        arm_gains = dict(zip(arm_names, arm_params.joint_tracking, strict=True))

        prefab = bot.get_bot_prefab()
        followers = {t.joint_indices[1] for t in prefab.linear_transmissions}
        tracking = []
        for i, joint in enumerate(prefab.joints):
            gains = physics.PoseTrackingParams()
            if joint.name in arm_gains:
                gains = arm_gains[joint.name]
            elif (
                joint.type == physics.ArticulatedJointType.REVOLUTE
                and i not in followers
            ):
                gains.stiffness = HAND_STIFFNESS
                gains.damping = HAND_DAMPING
                gains.saturation = joint.effort_limit
            tracking.append(gains)
        pose_params = physics.PoseControllerParams()
        pose_params.joint_tracking = tracking
        params = robotics.ControllerMochiArticulatedPoseParams()
        params.pose_controller_params = pose_params

        controller = bot.create_controller("MOCHI_ARTICULATED_POSE")
        controller.set_params(params)
        controller.initialize(True)
        return controller

    @staticmethod
    def _create_object(scene: physics.Scene, cfg: Fr3Revo2EnvCfg) -> physics.Actor:
        # Subdivided so the faces carry enough contact samples for the fingertips.
        box = trimesh.creation.box(cfg.object_extents)
        while box.edges_unique_length.max() > 0.006:
            box = box.subdivide()
        shape = physics.create_mesh_shape(
            physics.MeshData(
                nodes_per_element=3,
                coordinates=box.vertices.ravel(),
                connectivity=box.faces.ravel(),
            )
        )
        return scene.create_rigid_actor(
            name="object",
            shape=shape,
            is_static=False,
            mass=cfg.object_mass,
            contact=physics.ContactParams(
                coulomb_friction_coefficient=cfg.object_friction, **OBJECT_CONTACT
            ),
            sdf=physics.GridSdfParams(
                resolution_mode=physics.GridSdfResolutionMode.EXPLICIT,
                resolution_delta=[0.001] * 3,
                boundary_padding_dist=0.005,
                min_grid_resolution=[6, 6, 6],
            ),
            world_from_local=physics.TransformRT(
                physics.Quaternion.identity(),
                [cfg.object_xy[0], cfg.object_xy[1], cfg.object_extents[2] / 2],
            ),
        )

    @override_from(MochiEnv)
    def _reset_scene(self):
        cfg = self._cfg
        self._initial_pose[self._arm_dofs] = np.asarray(
            ARM_HOME[cfg.hand_side]
        ) + self.np_random.uniform(-cfg.arm_pose_noise, cfg.arm_pose_noise, 7)
        super()._reset_scene()

        xy = np.asarray(cfg.object_xy) + self.np_random.uniform(
            -cfg.object_xy_noise, cfg.object_xy_noise, 2
        )
        yaw = self.np_random.uniform(-np.pi / 4, np.pi / 4)
        self._object.set_root_transform(
            physics.TransformRT(
                physics.Quaternion.rotation_z(yaw), [xy[0], xy[1], self._object_rest_z]
            )
        )
        self._object.set_velocity([0, 0, 0], [0, 0, 0])

        self._target = self._initial_pose.astype(np.float64)
        self._goal = self._target.copy()
        self._apply_targets()
        for sensor in self._sensors.values():
            sensor.reset()

    ####################################################################################
    # Stepping
    ####################################################################################

    @override_from(MochiEnv)
    def _simulate(self, action: StructuredAction):
        # Turn the action into this control step's goal; _apply_action then moves the
        # controller target toward it every substep, within the joint speed limits.
        arm = np.asarray(action["arm"], dtype=np.float64)
        hand = np.asarray(action["hand"], dtype=np.float64)
        goal = self._goal
        goal[self._arm_dofs] = (
            self._target[self._arm_dofs] + arm * self._cfg.arm_joint_step
        )
        lo, hi = self._dof_min[self._hand_dofs], self._dof_max[self._hand_dofs]
        goal[self._hand_dofs] = lo + 0.5 * (hand + 1.0) * (hi - lo)
        np.clip(goal, self._dof_min, self._dof_max, out=goal)
        super()._simulate(action)

    @override_from(MochiEnv)
    def _apply_action(self, action: StructuredAction):
        dt = self._sim_dt
        arm_rate = self._cfg.arm_joint_step * self._control_frequency
        for dofs, max_step in (
            (self._arm_dofs, arm_rate * dt),
            (self._hand_dofs, self._hand_speed * dt),
        ):
            delta = self._goal[dofs] - self._target[dofs]
            self._target[dofs] += np.clip(delta, -max_step, max_step)
        self._apply_targets()

    def _apply_targets(self):
        for follower, leader, multiplier in self._mimic:
            self._target[follower] = multiplier * self._target[leader]
        self._pose_target.pose_dofs = self._target.astype(np.float32)
        self._controller.compute_output(self._pose_obsv, self._pose_target)

    ####################################################################################
    # Observation, reward, termination
    ####################################################################################

    def arm_target(self) -> np.ndarray:
        """The arm's current joint-position target [rad], which ``arm`` actions move."""
        return self._target[self._arm_dofs].copy()

    def grasp_point(self) -> np.ndarray:
        """The point between the curled fingers and the thumb, world frame [m]."""
        T = self._wrist.get_root_transform()
        R = Rotation.from_quat(list(T.rotation)).as_matrix()
        return R @ np.asarray(GRASP_POINT_IN_HAND) + np.asarray(T.translation)

    @override_from(MochiEnv)
    def _make_observation(self) -> tuple[StructuredObservation, Info]:
        pose = mochi_helpers.get_articulated_pose(self._agent)
        vel = mochi_helpers.get_articulated_joint_velocities(self._agent)
        T = self._object.get_root_transform()
        object_pos = np.asarray(T.translation, dtype=np.float64)
        grasp = self.grasp_point()

        readings = {
            f: self._sensors[f].compute_signal(self._control_dt) for f in FINGERS
        }
        touching = [
            f
            for f in FINGERS
            if readings[f].normal_force > self._cfg.contact_force_threshold
        ]
        obs = {
            "arm_qpos": pose[self._arm_dofs],
            "arm_qvel": vel[self._arm_dofs],
            "hand_qpos": pose[self._hand_dofs],
            "hand_qvel": vel[self._hand_dofs],
            "grasp_point": grasp,
            "object_pos": object_pos,
            "object_quat": np.asarray(list(T.rotation)),
            "object_to_grasp": grasp - object_pos,
            "tactile": np.concatenate([readings[f].to_vector() for f in FINGERS]),
        }
        if self._cfg.include_taxels:
            taxels = np.concatenate([readings[f].taxels.ravel() for f in FINGERS])
            if self._cfg.tactile_noise_std > 0:
                noise = self.np_random.normal(
                    0.0, self._cfg.tactile_noise_std, taxels.shape
                )
                taxels = np.maximum(taxels + noise, 0.0)
            obs["taxels"] = taxels
        obs = {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}

        height = object_pos[2] - self._object_rest_z
        info = {
            "object_height": float(height),
            "grasp_distance": float(np.linalg.norm(grasp - object_pos)),
            "fingers_in_contact": touching,
            "tactile_normal_force": {f: readings[f].normal_force for f in FINGERS},
            "is_success": bool(
                height >= self._cfg.lift_height
                and "thumb" in touching
                and len(touching) >= 2
            ),
        }
        return obs, info

    @override_from(MochiEnv)
    def _compute_reward_terms(
        self, action: StructuredAction, observation: StructuredObservation, info: Info
    ) -> RewardTerms:
        cfg = self._cfg
        touching = info["fingers_in_contact"]
        # An opposed grasp needs the thumb and at least one finger.
        grip = min(len(touching), 3) / 3.0 if "thumb" in touching else 0.0
        lift = (
            np.clip(info["object_height"] / cfg.lift_height, 0.0, 1.0)
            if touching
            else 0.0
        )
        effort = sum(float(np.dot(a, a)) for a in action.values())
        return {
            "reach": -cfg.reach_weight * info["grasp_distance"],
            "grip": cfg.contact_weight * grip,
            "lift": cfg.lift_weight * float(lift),
            "success": cfg.success_bonus * info["is_success"],
            "action": -cfg.action_weight * effort,
        }

    @override_from(MochiEnv)
    def _check_stop_criteria(
        self,
        observation: StructuredObservation,
        action: StructuredAction,
        reward: RewardTerms,
        info: Info,
    ):
        super()._check_stop_criteria(
            observation=observation, action=action, reward=reward, info=info
        )
        object_xy = observation["object_pos"][:2]
        if (
            info["object_height"] < -0.05
            or np.linalg.norm(object_xy - self._cfg.object_xy) > 0.3
        ):
            info["terminated_reason"] = "Object lost"
            self._terminated = True
        elif self._cfg.terminate_on_success and info["is_success"]:
            info["terminated_reason"] = "Success"
            self._terminated = True
