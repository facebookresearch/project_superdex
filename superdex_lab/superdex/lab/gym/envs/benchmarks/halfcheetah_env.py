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

#######################################################################################


@configclass
class HalfCheetahEnvCfg(MochiEnvCfg):
    """Options for the HalfCheetah environment."""

    # Default environment options
    control_frequency: int = 20
    simulation_frequency: int = 100
    steps_per_episode: int = 1000
    reset_noise_scale: float = 0.1

    # Custom environment options
    forward_reward_weight: float = 1
    """Weight of the forward reward term."""
    control_cost_weight: float = 0.1
    """Weight of the control cost term."""
    exclude_current_position_from_observation: bool = True
    """Whether or not to omit the x-coordinate from observations. Excluding the
    position can serve as an inductive bias to induce position-agnostic behavior in
    policies."""

    # Other tweaks
    use_gravity: bool = True
    """If True, gravity is used."""
    use_rest_springs: bool = True
    """If True, rest springs are applied to the Cheetah's joints."""


#######################################################################################


class HalfCheetahEnv(MochiEnv):
    """HalfCheetah environment."""

    ####################################################################################
    # Member variables
    ####################################################################################

    # Private members.
    _forward_reward_weight: float
    _control_cost_weight: float
    _exclude_current_position_from_observation: bool

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(self, cfg: HalfCheetahEnvCfg | dict[str, Any]):
        """Constructor for the HalfCheetah environment."""
        if not isinstance(cfg, HalfCheetahEnvCfg):
            cfg = HalfCheetahEnvCfg(**cfg)

        # Initialize the base environment.
        super().__init__(cfg)

        # Initialize the HalfCheetah environment.
        self._forward_reward_weight = cfg.forward_reward_weight
        self._control_cost_weight = cfg.control_cost_weight
        self._exclude_current_position_from_observation = (
            cfg.exclude_current_position_from_observation
        )

        # Initialize the Mochi scene.
        self._init_scene(cfg)

        # Setup the observation and action spaces.
        num_dofs = 9
        num_control_joints = 6
        num_pose_elems = num_dofs - self._exclude_current_position_from_observation
        num_vel_elems = num_dofs
        control_limit = 1

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
            self._renderer.set_camera_view(look_from=[0, 1.5, 3], look_at=[0, 1, 0])
            self._renderer.set_enable_follow_camera(True)
            self._renderer.set_follow_camera_smoothness(0.8)
            self._renderer.set_compute_automatic_distance(True)
            self._renderer.frame_scene()

    ####################################################################################
    # Functions handling Mochi scene loading and resetting.
    ####################################################################################

    def _init_scene(self, cfg: HalfCheetahEnvCfg):
        # Perform scene creation. This is done only once, for the first environment in
        # this process (if scene sharing is enabled). Subsequent environments will reuse
        # the same scene, as long as the configuration is the same.
        def scene_builder():
            # Load the original Half Cheetah scene prefab.
            assets_root = get_assets_root()
            prefab_path = (
                assets_root / "benchmarks" / "half_cheetah" / "half_cheetah.mochi_scene"
            )
            prefab = sdp.prefab.shallow_load_from_file(str(prefab_path))

            # Tweak the prefab according to the supplied configuration.
            if not cfg.use_gravity:
                prefab.scene.gravity = [0, 0, 0]
            if not cfg.use_rest_springs:
                prefab.controllers = []

            # Initialize scene from prefab.
            prefab_params = mochi_helpers.PrefabParams()
            prefab_params.agent_actor_name = "HalfCheetah"
            prefab_params.add_ground_plane = cfg.use_gravity
            return mochi_helpers.init_prefab_scene(prefab, prefab_params)

        uid_fields = (cfg.use_gravity, cfg.use_rest_springs)
        self._load_scene(f"scene_{hash(uid_fields)}", scene_builder)

        # Retrieve the initial pose and velocity of the agent.
        self._initial_pose = mochi_helpers.get_articulated_pose(self._agent)
        self._initial_velocity = mochi_helpers.get_articulated_joint_velocities(
            self._agent
        )

    ####################################################################################
    # Functions handling environment initialization and resetting.
    ####################################################################################

    @override_from(MochiEnv)
    def _apply_action(self, action: StructuredAction):
        start_dof_idx = 3
        dofs = np.arange(start_dof_idx, start_dof_idx + 6)
        # The policy emits a normalized control in [-1, 1] per actuated joint;
        # force_scale maps it to the per-joint force applied to each controlled DoF.
        force_scale = np.array([120, 90, 60, 120, 60, 30])
        control = action["control"]
        forces = control * force_scale

        self._agent.set_external_forces_on_dofs(dofs, forces)

    @override_from(MochiEnv)
    def _make_observation(self) -> tuple[StructuredObservation, Info]:
        # Construct the pose observation.
        pose = mochi_helpers.get_articulated_pose(self._agent)
        x_position = pose[0]
        if self._exclude_current_position_from_observation:
            pose = pose[1:]

        # Construct the velocity observation.
        velocity = mochi_helpers.get_articulated_joint_velocities(self._agent)
        x_velocity = velocity[0]

        # Fill in the observation and info.
        obs = {"pose": pose, "vel": velocity}
        info = {"x_position": x_position, "x_velocity": x_velocity}
        return obs, info

    @override_from(MochiEnv)
    def _compute_reward_terms(
        self, action: StructuredAction, observation: StructuredObservation, info: Info
    ) -> RewardTerms:
        # Compute forward reward.
        forward_reward = info["x_velocity"] * self._forward_reward_weight

        # Compute control cost.
        control = action["control"]
        control_cost = -self._control_cost_weight * np.dot(control, control)

        # Fill in the reward terms.
        return {"forward": forward_reward, "ctrl": control_cost}


#######################################################################################
