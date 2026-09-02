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

from typing import Any

import numpy as np
import superdex.physics as sdp
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
from superdex.physics.paths import get_assets_root
from superdex.physics.utils.configclasses import configclass
from superdex.physics.utils.decorators import override_from
from superdex.physics.utils.transformations import rotvec_to_quat

#######################################################################################


@configclass
class AntEnvCfg(MochiEnvCfg):
    """Options for the Ant environment."""

    # Default environment options
    control_frequency: int = 20
    simulation_frequency: int = 100
    steps_per_episode: int = 1000
    reset_noise_scale: float = 0.1

    # Custom environment options
    forward_reward_weight: float = 1
    """Weight of the forward reward term."""
    control_cost_weight: float = 0.5
    """Weight of the control cost term."""
    contact_cost_weight: float = 5e-4
    """Weight of the contact cost term."""
    healthy_reward: float = 1.0
    """Reward given to healthy agents."""
    terminate_when_unhealthy: bool = True
    """If `True`, issue a `terminated` signal if unhealthy."""
    healthy_y_range: tuple[float, float] = (0.2, 1.0)
    """The ant is considered healthy if the y-coordinate of the torso is in this range."""
    contact_force_range: tuple[float, float] = (-1.0, 1.0)
    """Contact forces are clipped to this range in the computation of *contact_cost*."""
    exclude_current_positions_from_observation: bool = True
    """Whether or not to omit the x- and z-coordinates from observations. Excluding the
    position can serve as an inductive bias to induce position-agnostic behavior in
    policies."""
    include_contact_in_observation: bool = True
    """Whether to include *cfrc_ext* elements in the observations."""
    use_rotation_vector: bool = False
    """Whether to use rotation vectors to represent rotations in the observation."""

    # Other tweaks
    use_damping: bool = True
    """If True, damping is applied to the Ant's joints."""
    use_gravity: bool = True
    """If True, gravity is applied to the Ant."""
    use_low_friction: bool = False
    """If True, the ground plane has low friction."""
    use_high_friction: bool = False
    """If True, the ground plane has high friction."""
    init_dist_from_ground: float = 0.55
    """Initial distance from the ground."""
    init_ankle_angle: float = np.radians(57.30)
    """Initial angle of the ankle joints."""


#######################################################################################


class AntEnv(MochiEnv):
    """Ant environment."""

    ####################################################################################
    # Member variables
    ####################################################################################

    # Private members.
    _forward_reward_weight: float
    _control_cost_weight: float
    _contact_cost_weight: float
    _healthy_reward: float
    _terminate_when_unhealthy: bool
    _healthy_y_range: tuple[float, float]
    _contact_force_range: tuple[float, float]
    _exclude_current_positions_from_observation: bool
    _include_contact_in_observation: bool
    _use_rotation_vector: bool
    _contact_actors: list[sdp.Actor]

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(self, cfg: AntEnvCfg | dict[str, Any]):
        """Constructor for the Ant environment."""
        if not isinstance(cfg, AntEnvCfg):
            cfg = AntEnvCfg(**cfg)

        # Initialize the base environment.
        super().__init__(cfg)

        # Initialize the Ant environment.
        self._control_cost_weight = cfg.control_cost_weight
        self._contact_cost_weight = cfg.contact_cost_weight
        self._forward_reward_weight = cfg.forward_reward_weight
        self._healthy_reward = cfg.healthy_reward
        self._terminate_when_unhealthy = cfg.terminate_when_unhealthy
        self._healthy_y_range = cfg.healthy_y_range
        self._contact_force_range = cfg.contact_force_range
        self._exclude_current_positions_from_observation = (
            cfg.exclude_current_positions_from_observation
        )
        self._include_contact_in_observation = cfg.include_contact_in_observation
        self._use_rotation_vector = cfg.use_rotation_vector

        # Initialize the Mochi scene.
        self._init_scene(cfg)

        # Setup the observation and action spaces.
        num_dofs = 14
        num_bodies = 9
        num_control_joints = num_bodies - 1
        num_pose_elems = (
            num_dofs
            - 2 * self._exclude_current_positions_from_observation
            + (not self._use_rotation_vector)
        )
        num_vel_elems = num_dofs
        control_limit = 1.0

        # Setup observation space.
        observation_space = {
            # Position values of the robot’s body parts.
            "pose": ObservationSpace(
                -np.inf, np.inf, (num_pose_elems,), dtype=np.float32
            ),
            # The velocities of these individual body parts (their derivatives).
            "vel": ObservationSpace(
                -np.inf, np.inf, (num_vel_elems,), dtype=np.float32
            ),
        }
        # Contact forces and torques exerted on the body parts.
        # Only present if include_contact_in_observation is True.
        if self._include_contact_in_observation:
            num_contact_elems = num_bodies * 6
            observation_space["contact_forces"] = ObservationSpace(
                -np.inf, np.inf, (num_contact_elems,), dtype=np.float32
            )
        self._setup_observation_space(**observation_space)

        # Setup action space.
        self._setup_action_space(
            # Torques applied at the hinge joints.
            control=ActionSpace(
                -control_limit, control_limit, (num_control_joints,), dtype=np.float32
            ),
        )

        # Setup preferred renderer settings.
        if self._renderer:
            self._renderer.set_camera_view(look_from=[-2, 3, 2], look_at=[0, 1, 0])
            self._renderer.set_enable_follow_camera(True)
            self._renderer.set_follow_camera_smoothness(0.8)
            self._renderer.set_compute_automatic_distance(True)
            self._renderer.frame_scene()

    ####################################################################################
    # Functions handling Mochi scene loading and resetting.
    ####################################################################################

    def _init_scene(self, cfg: AntEnvCfg):
        # Perform scene creation. This is done only once, for the first environment in
        # this process (if scene sharing is enabled). Subsequent environments will reuse
        # the same scene, as long as the configuration is the same.
        def scene_builder():
            # Load the Ant prefab.
            assets_root = get_assets_root()
            prefab_path = assets_root / "benchmarks" / "ant" / "ant.mochi_prefab"
            prefab = sdp.prefab.shallow_load_from_file(str(prefab_path))

            # Add scene-level settings to the prefab
            prefab.scene = sdp.prefab.SceneParams(
                solver=sdp.SolverParams(
                    linear_solver=sdp.LinearSolverParams(abs_tol=1e-10)
                )
            )

            # Tweak the prefab according to the given config.
            if not cfg.use_damping:
                for joint in prefab.actors.articulated[0].joints:
                    joint.friction.viscous = 0.0
            if not cfg.use_gravity:
                prefab.scene.gravity = [0, 0, 0]
            if cfg.use_low_friction:
                for link in prefab.actors.articulated[0].links:
                    link.contact = sdp.ContactParams(coulomb_friction_coefficient=0.1)
            if cfg.use_high_friction:
                for link in prefab.actors.articulated[0].links:
                    link.contact = sdp.ContactParams(coulomb_friction_coefficient=2.0)

            # Initialize scene from prefab.
            prefab_params = mochi_helpers.PrefabParams()
            prefab_params.agent_actor_name = "Ant"
            prefab_params.add_ground_plane = cfg.use_gravity
            return mochi_helpers.init_prefab_scene(prefab, prefab_params)

        uid_fields = (
            cfg.use_damping,
            cfg.use_gravity,
            cfg.use_low_friction,
            cfg.use_high_friction,
        )
        self._load_scene(f"scene_{hash(uid_fields)}", scene_builder)

        # Define initial pose and velocity.
        num_dofs = self._agent.get_num_dofs()
        self._initial_pose = np.zeros(num_dofs, dtype=np.float32)
        self._initial_pose[1] = cfg.init_dist_from_ground
        self._initial_pose[7::2] = cfg.init_ankle_angle
        self._initial_velocity = np.zeros(num_dofs, dtype=np.float32)

        # Retrieve the actor handles from which we will sample contact forces.
        # Enable contact queries for them.
        self._contact_actors = []
        for link_handle in self._agent.get_nested_link_actors():
            link_actor = self._scene.get_actor(link_handle)
            link_actor.register_query(sdp.QueryType.TOTAL_CONTACT_FORCE)
            self._contact_actors.append(link_actor)

    ####################################################################################
    # Functions handling environment initialization and resetting.
    ####################################################################################

    @override_from(MochiEnv)
    def _apply_action(self, action: StructuredAction):
        start_dof_idx = 6
        num_control_joints = 8
        dofs = np.arange(start_dof_idx, start_dof_idx + num_control_joints)
        # The policy emits a normalized control in [-1, 1] per actuated joint;
        # force_scale maps it to the joint force applied to each controlled DoF.
        force_scale = 150
        control = action["control"]
        forces = control * force_scale

        self._agent.set_external_forces_on_dofs(dofs, forces)

    @override_from(MochiEnv)
    def _make_observation(self) -> tuple[StructuredObservation, Info]:
        # Construct the pose observation.
        pose = mochi_helpers.get_articulated_pose(self._agent)
        x_position, y_position, z_position = pose[0:3]
        if not self._use_rotation_vector:
            pose = np.concatenate([pose[:3], rotvec_to_quat(pose[3:6]), pose[6:]])
        if self._exclude_current_positions_from_observation:
            pose = np.concatenate([pose[1:2], pose[3:]])

        # Construct the velocity observation.
        velocity = mochi_helpers.get_articulated_joint_velocities(self._agent)
        x_velocity, _, z_velocity = velocity[0:3]

        # Construct contact forces observation.
        unclipped_contact = np.concatenate(
            [
                mochi_helpers.get_contact_force_and_torque_world(actor)
                for actor in self._contact_actors
            ]
        )
        contact = np.clip(unclipped_contact, *self._contact_force_range)

        # Determine if the Ant is healthy.
        min_y_position, max_y_position = self._healthy_y_range
        is_healthy = min_y_position <= y_position <= max_y_position

        # Fill in the observation and info.
        obs = {"pose": pose, "vel": velocity}
        if self._include_contact_in_observation:
            obs["contact_forces"] = contact

        contact_forces = contact.reshape(-1, 6)[:, 0:3]
        contact_torques = contact.reshape(-1, 6)[:, 3:6]
        contact_forces_norms = np.linalg.norm(contact_forces, axis=1)
        contact_torques_norms = np.linalg.norm(contact_torques, axis=1)
        unclipped_contact_forces = unclipped_contact.reshape(-1, 6)[:, 0:3]
        unclipped_contact_torques = unclipped_contact.reshape(-1, 6)[:, 3:6]
        unclipped_contact_forces_norms = np.linalg.norm(
            unclipped_contact_forces, axis=1
        )
        unclipped_contact_torques_norms = np.linalg.norm(
            unclipped_contact_torques, axis=1
        )

        info = {
            "x_position": x_position,
            "z_position": z_position,
            "distance_from_origin": np.sqrt(x_position**2 + z_position**2),
            "x_velocity": x_velocity,
            "z_velocity": z_velocity,
            "is_healthy": is_healthy,
            "sqr_sum_of_contact_forces": np.dot(contact, contact),
            "contact_forces_min": np.min(contact_forces_norms),
            "contact_forces_max": np.max(contact_forces_norms),
            "contact_forces_mean": np.mean(contact_forces_norms),
            "contact_torques_min": np.min(contact_torques_norms),
            "contact_torques_max": np.max(contact_torques_norms),
            "contact_torques_mean": np.mean(contact_torques_norms),
            "unclipped_contact_forces_min": np.min(unclipped_contact_forces_norms),
            "unclipped_contact_forces_max": np.max(unclipped_contact_forces_norms),
            "unclipped_contact_forces_mean": np.mean(unclipped_contact_forces_norms),
            "unclipped_contact_torques_min": np.min(unclipped_contact_torques_norms),
            "unclipped_contact_torques_max": np.max(unclipped_contact_torques_norms),
            "unclipped_contact_torques_mean": np.mean(unclipped_contact_torques_norms),
        }
        return obs, info

    @override_from(MochiEnv)
    def _compute_reward_terms(
        self, action: StructuredAction, observation: StructuredObservation, info: Info
    ) -> RewardTerms:
        # Compute forward reward.
        forward_reward = info["x_velocity"] * self._forward_reward_weight

        # Compute healthy reward.
        healthy_reward = self._healthy_reward if info["is_healthy"] else 0

        # Compute control cost.
        control = action["control"]
        control_cost = -self._control_cost_weight * np.dot(control, control)

        # Compute contact cost.
        contact_cost = -self._contact_cost_weight * info["sqr_sum_of_contact_forces"]

        # Fill in the reward terms.
        return {
            "forward": forward_reward,
            "ctrl": control_cost,
            "contact": contact_cost,
            "survive": healthy_reward,
        }

    @override_from(MochiEnv)
    def _check_stop_criteria(
        self,
        observation: StructuredObservation,
        action: StructuredAction,
        reward: RewardTerms,
        info: Info,
    ):
        # Check episode truncation if the step count has been reached.
        super()._check_stop_criteria(
            action=action,
            observation=observation,
            reward=reward,
            info=info,
        )

        # Terminate episode if unhealthy.
        if self._terminate_when_unhealthy and not info["is_healthy"]:
            info["terminated_reason"] = "Unhealthy"
            self._terminated = True


#######################################################################################
