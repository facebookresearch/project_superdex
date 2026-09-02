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
class CartPoleEnvCfg(MochiEnvCfg):
    """Options for the Cartpole environment."""

    # Default environment options
    control_frequency: int = 25
    simulation_frequency: int = 50
    steps_per_episode: int = 1000
    reset_noise_scale: float = 0.1

    # Custom environment options
    render_control: bool = True
    """If True, renders the control signal driving the cart."""
    actuate_on_pole: bool = False
    """If True, the control signal will exert a force on the pole instead of on the cart."""

    # Other tweaks
    use_damping: bool = True
    """If True, damping is applied to the cart and the pole."""
    use_gravity: bool = True
    """If True, gravity is applied to the cart and the pole."""
    free_pole: bool = False
    """If True, joint limits are removed from the pole."""


#######################################################################################


class CartPoleEnv(MochiEnv):
    """Cartpole environment."""

    ####################################################################################
    # Member variables
    ####################################################################################

    # Private members.
    _render_control: bool
    _actuate_on_pole: bool

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(self, cfg: CartPoleEnvCfg | dict[str, Any]):
        """Constructor for the Cartpole environment."""
        if not isinstance(cfg, CartPoleEnvCfg):
            cfg = CartPoleEnvCfg(**cfg)

        # Initialize the base environment.
        super().__init__(cfg)

        # Initialize the CartPole environment.
        self._render_control = cfg.render_control
        self._actuate_on_pole = cfg.actuate_on_pole

        # Initialize the Mochi scene.
        self._init_scene(cfg)

        # Setup the observation and action spaces.
        self._setup_observation_space(
            # Position of the cart along the linear surface. (m)
            position=ObservationSpace(-np.inf, np.inf, (1,), dtype=np.float32),
            # Vertical angle of the pole on the cart. (rad)
            vertical_ang=ObservationSpace(-np.inf, np.inf, (1,), dtype=np.float32),
            # Linear velocity of the cart. (m/s)
            linear_vel=ObservationSpace(-np.inf, np.inf, (1,), dtype=np.float32),
            # Angular velocity of the pole on the cart. (rad/s)
            angular_vel=ObservationSpace(-np.inf, np.inf, (1,), dtype=np.float32),
        )
        self._setup_action_space(
            # Control applied on the cart.
            control=ActionSpace(-3.0, 3.0, (1,), dtype=np.float32),
        )

        # Setup preferred renderer settings.
        if self._renderer:
            self._renderer.set_camera_view(look_from=[0, 0, 8], look_at=[0, 0, 0])
            self._renderer.add_grid(
                "Reference",
                size=6,
                center=np.array([0, 0, -0.5]),
                axes="xy",
                double_sided=True,
            )

    ####################################################################################
    # Functions handling Mochi scene loading and resetting.
    ####################################################################################

    def _init_scene(self, cfg: CartPoleEnvCfg):
        # Perform scene creation. This is done only once, for the first environment in
        # this process (if scene sharing is enabled). Subsequent environments will reuse
        # the same scene, as long as the configuration is the same.
        def scene_builder():
            # Load the original CartPole scene prefab.
            assets_root = get_assets_root()
            prefab_path = (
                assets_root / "benchmarks" / "cart_pole" / "cart_pole.mochi_scene"
            )
            prefab = sdp.prefab.shallow_load_from_file(str(prefab_path))

            # Tweak the prefab according to the supplied configuration.
            if not cfg.use_gravity:
                prefab.scene.gravity = [0, 0, 0]
            if not cfg.use_damping:
                for joint in prefab.actors.articulated[0].joints:
                    joint.friction.viscous = 0.0
            if cfg.free_pole:
                prefab.actors.articulated[0].joints[1].min_limit = None
                prefab.actors.articulated[0].joints[1].max_limit = None

            # Initialize scene from prefab.
            prefab_params = mochi_helpers.PrefabParams()
            prefab_params.agent_actor_name = "CartPole"
            prefab_params.add_ground_plane = False
            return mochi_helpers.init_prefab_scene(prefab, prefab_params)

        uid_fields = (cfg.use_damping, cfg.use_gravity, cfg.free_pole)
        self._load_scene(f"scene_{hash(uid_fields)}", scene_builder)

        # Retrieve initial pose and velocity of the agent.
        self._initial_pose = mochi_helpers.get_articulated_pose(self._agent)
        self._initial_velocity = mochi_helpers.get_articulated_joint_velocities(
            self._agent
        )

    ####################################################################################
    # Functions handling environment initialization and resetting.
    ####################################################################################

    @override_from(MochiEnv)
    def _apply_action(self, action: StructuredAction):
        # force_scale maps the action-space control to the force applied to the
        # actuated DoF.
        force_scale = 100
        control = action["control"].item()
        force = control * force_scale
        index = int(self._actuate_on_pole)
        self._agent.set_external_forces_on_dofs([index], [force])

    @override_from(MochiEnv)
    def _make_observation(self) -> tuple[StructuredObservation, Info]:
        # Get the current pose of the CartPole.
        pose = mochi_helpers.get_articulated_pose(self._agent)
        position, vertical_ang = pose

        # Compute the velocity of the CartPole.
        velocity = mochi_helpers.get_articulated_joint_velocities(self._agent)
        linear_vel, angular_vel = velocity

        # Determine if the pole is upright.
        is_upright = np.abs(vertical_ang) <= 0.2

        # Fill in the observation and info.
        obs = {
            "position": position,
            "vertical_ang": vertical_ang,
            "linear_vel": linear_vel,
            "angular_vel": angular_vel,
        }
        info = {
            "reward_survive": is_upright,
        }
        return obs, info

    @override_from(MochiEnv)
    def _compute_reward_terms(
        self, action: StructuredAction, observation: StructuredObservation, info: Info
    ) -> RewardTerms:
        # A constant reward of 1 is given for every step in which the pole is upright.
        is_upright = info["reward_survive"]
        return {"upright_reward": 1.0 if is_upright else 0.0}

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

        # Check episode termination if the pole angle has exceeded the threshold.
        if not info["reward_survive"]:
            info["terminated_reason"] = "Pole angle exceeded threshold"
            self._terminated = True


#######################################################################################
