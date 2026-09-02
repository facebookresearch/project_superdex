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

"""
Defines an environment capable of hand interaction simulation, and all the necessary
data structures supporting it. The environment is built on top of the gymnasium library,
and relies on the SuperDex Gym library to simulate the interaction of actors with
their environment.

See :class:`MochiEnv` for a detailed description of the environment, its API, and how to
extend it.
"""

from __future__ import annotations

import abc
import enum
import json
import logging
import os
import warnings
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import superdex.physics as sdp
from gymnasium import Env, spaces
from gymnasium.error import ClosedEnvironmentError
from superdex.lab.gym.envs.scene_manager import destroy_scene_with_cleanup, SceneManager
from superdex.lab.gym.envs.types import (
    Action,
    ActionSpace,
    ActionSpaceStructure,
    Info,
    Observation,
    ObservationSpace,
    ObservationSpaceStructure,
    ResetResult,
    RewardTerms,
    StepResult,
    StructuredAction,
    StructuredObservation,
    StructuredStepResult,
)
from superdex.lab.gym.utils import mochi_helpers
from superdex.physics.utils.configclasses import configclass
from superdex.physics.utils.coordinate_systems import CoordinateSystem
from superdex.physics.utils.decorators import override_from
from superdex.physics.utils.deprecation import deprecated
from superdex.physics.utils.json import ExtendedJSONEncoder
from superdex.physics.utils.logging import forward_mochi_logs_to_logger
from superdex.physics.utils.profiling import Profiler
from superdex.physics.viewer import RenderFrame, Viewer, ViewerCfg
from superdex.physics.viewer.backend import polyscope_imgui as psim
from superdex.physics.viewer.ui import widgets

logger = logging.getLogger(__name__)

#######################################################################################


VALID_RENDER_MODES: tuple[str | None, ...] = (None, "human", "rgb_array")
"""Valid render mode values accepted by MochiEnv.

- ``None``: No rendering.
- ``"human"``: Render the environment in a human-readable format (Polyscope viewer).
- ``"rgb_array"``: Render the environment as an RGB array.
"""


class RenderMode(enum.Enum):
    """
    .. deprecated::
        Use plain strings (``None``, ``"human"``, ``"rgb_array"``) instead.
        This enum is kept only for backward compatibility and will be removed
        in a future release.
    """

    NONE = None
    HUMAN = "human"
    RGB_ARRAY = "rgb_array"


#######################################################################################


@configclass
class MochiEnvCfg:
    """
    Options for the Mochi Environment. Inherit from this class to define additional
    options for your environment.
    """

    # Environment options
    control_frequency: int
    """Control frequency in Hz."""
    simulation_frequency: int
    """Simulation frequency in Hz. Must be a multiple of the control frequency."""
    steps_per_episode: int = -1
    """Number of steps per episode. Set to -1 for no limit."""
    reset_noise_scale: float = 0.0
    """Scale of random perturbations of initial position and velocity."""
    num_worker_threads: int = 0
    """Number of worker threads allocated by Mochi to parallelize the solver. The value
    used here is highly dependent on the particularities of your hardware, environment
    and training settings. If num_worker_threads is set to -1, the number of threads
    will be automatically determined by Mochi based on the number of available cores.
    If num_worker_threads is set to 0, no multithreading will be used."""
    use_shared_scenes: bool = True
    """If True, signals the environments to try use scene sharing. This is a performance
    optimization that can be used to reduce the memory footprint of the environment if
    the scenes are identical across environments. Note this makes a strong assumption,
    that is that environments instanced in the same process will be stepped
    sequentially. This is the case for batched execution of environments with
    HybridVectorEnv, but it is not guaranteed for other vectorization strategies."""

    # Rendering options
    render_mode: str | None = None
    """Render mode. Use ``None`` (no rendering), ``"human"`` (Polyscope viewer),
    or ``"rgb_array"`` (offscreen rendering)."""
    render_size: tuple[int, int] | None = None
    """Width and height of the renderer window. If None, the default size will be used."""
    render_coordinate_system: CoordinateSystem | str | None = None
    """Coordinate system to use for visualization. If None, the default coordinate system
    (Polyscope convention: right-handed, Y-up, -Z-forward) will be used. You can specify
    a custom coordinate system using CoordinateSystem class or use named presets like
    "unity", "unreal", etc."""
    start_paused: bool = False
    """If the environment is being rendered in human rendering mode, start the viewer
    with the simulation paused."""
    # Other options
    profile: bool = False
    """If True, wall-clock timings for different aspects of the environment will be
    measured and logged into the profiler object."""
    dump_timings_to_info: bool = True
    """If True, the timings recorded by the profiler object will be appended to the
    info object. Requires `profile` to be set to True."""

    def __post_init__(self):
        if isinstance(self.render_mode, RenderMode):
            warnings.warn(
                "RenderMode enum is deprecated; pass plain strings "
                '(None, "human", "rgb_array") instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            self.render_mode = self.render_mode.value
        if self.render_mode not in VALID_RENDER_MODES:
            raise ValueError(
                f"Invalid render_mode {self.render_mode!r}. "
                f"Must be one of {VALID_RENDER_MODES}."
            )


#######################################################################################


class MochiEnv(abc.ABC, Env[ObservationSpace, ActionSpace]):
    """
    Base environment class for Mochi, built on top of Gymnasium's Env. This class
    provides the basic functionality for interacting with a Mochi scene, and it is meant
    to be extended by derived environments to support specific use cases. Basic
    environment configuration is provided through the MochiEnvCfg class, which can be
    used to configure the control and simulation frequencies, the number of steps per
    episode, and the render mode.

    Derived environments must implement scene initialization in the constructor and/or
    _reset_scene(). For scene loading, derived environments should use the _load_scene()
    method, which handles scene creation, sharing optimization, and proper resource
    management. This method takes a scene name and factory function, automatically
    managing scene lifecycle and enabling memory-efficient scene sharing across
    environment instances.

    Derived environments must override the following methods to provide the necessary
    functionality:

    - `_reset_scene`: Resets the Mochi scene. This function is called each time the
    environment is reset (i.e. at the start of the episode). Here the user must (re)load
    the desired Mochi scene. To enable curriculum-style learning, this function can be
    be used to load different scenes depending on the current episode.

    - `_make_observation`: Samples the environment and returns a structured observation
    and additional info about the environment. The output structured observation is
    assumed to be a dictionary with the same keys as the observation space structure.

    - `_compute_reward_terms`: Computes the reward terms given the applied action,
    environment observation and any additional information about the current state of
    the environment.

    - `_apply_action`: Applies the given action to the environment. The input action is
    assumed to be a dictionary with the same keys as the action space structure.

    In addition to the above, there are several other functions that can be overridden
    to fine-tune the behavior of the environment:

    - `_check_stop_criteria`: Checks whether the episode should be truncated or
    terminated. The default stop criteria is to terminate the episode after a fixed
    number of steps, but derived environments can override this function to introduce
    additional criteria (e.g. based on the environment observation or the applied
    action).

    - `_reset_renderer`: Resets the renderer. This function is called each time the
    environment is reset (i.e. at the start of the episode). Derived environments can
    override this function to tweak the renderer (e.g. to add additional visuals aiding
    the understanding of the environment state). It is guaranteed that this function
    will be called only if the renderer is available, and after the scene has been
    loaded.

    - `_update_renderer`: Updates the renderer. This function is called each time the
    environment is stepped (i.e. at the end of the control step). Derived environments
    can override this function to tweak the renderer (e.g. to add additional visuals
    aiding the understanding of the environment state, like force vectors). It is
    guaranteed that this function will be called only if the renderer is available, and
    after the environment has been stepped.

    - `_init_ui`: Initializes the user interface. This function is called only once at
    the start of the environment. Derived environments can override this function to
    register additional user interface tabs. It is guaranteed that this function will
    be called only if the renderer is available.
    """

    ####################################################################################
    # Member variables
    ####################################################################################

    # Static members
    metadata = {
        "render_modes": VALID_RENDER_MODES,
    }

    # Protected members
    _scene: sdp.Scene
    _scene_name: str | None
    _scene_manager: SceneManager | None
    _agent: sdp.Actor
    _initial_state_snapshot: sdp.StateHandle
    _state_snapshot: sdp.StateHandle
    _control_frequency: int
    _control_dt: float
    _simulation_frequency: int
    _sim_substeps: int
    _sim_dt: float
    _steps_per_episode: int
    _step_count: int
    _episode: int
    _truncated: bool
    _terminated: bool
    _observation_space_structure: ObservationSpaceStructure
    _action_space_structure: ActionSpaceStructure
    _renderer: Viewer | None
    _render_size: tuple[int, int] | None
    _render_start_paused: bool
    _render_scene_dirty: bool
    _reset_noise_scale: float
    _initial_pose: npt.NDArray[float]
    _initial_velocity: npt.NDArray[float]
    _dummy_step: float
    _last_observation: StructuredObservation
    _last_action: StructuredAction
    _last_reward: RewardTerms
    _last_info: Info
    _is_closed: bool
    _profiler: Profiler
    _dump_timings_to_info: bool

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(self, cfg: MochiEnvCfg | dict[str, Any]):
        """
        Initializes the environment according to the given configuration. Note that
        environments derived from this class must manually specify the observation and
        action spaces.
        """

        # Validate the given configuration.
        # fmt: off
        if cfg.control_frequency <= 0:
            raise ValueError("Control frequency must be positive.")
        if cfg.simulation_frequency <= 0:
            raise ValueError("Simulation frequency must be positive.")
        if cfg.control_frequency > cfg.simulation_frequency:
            raise ValueError("Simulation frequency must be greater or equal than control frequency.")
        if cfg.simulation_frequency % cfg.control_frequency != 0:
            raise ValueError("Simulation frequency must be an integer multiple of control frequency.")
        if cfg.render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode: {cfg.render_mode}.")
        if cfg.reset_noise_scale < 0:
            raise ValueError("Reset noise scale must not be negative.")
        # fmt: on

        # Mochi-scene-related variables
        self._scene = None
        self._scene_name = None
        self._scene_manager = None
        self._agent = None
        self._initial_state_snapshot = None
        self._state_snapshot = None
        self._scene_cleanup_callbacks: list[Callable[[], None]] = []
        self._renderer = None

        # Gym-related variables
        self._control_frequency = cfg.control_frequency
        self._control_dt = 1.0 / cfg.control_frequency
        self._simulation_frequency = cfg.simulation_frequency
        self._sim_substeps = int(cfg.simulation_frequency // cfg.control_frequency)
        self._sim_dt = 1.0 / cfg.simulation_frequency
        self._steps_per_episode = cfg.steps_per_episode
        self._reset_noise_scale = cfg.reset_noise_scale
        self._dump_timings_to_info = cfg.dump_timings_to_info
        self._step_count = 0
        self._episode = 0
        self._truncated = False
        self._terminated = False
        self._observation_space_structure = None
        self._action_space_structure = None
        self._initial_pose = None
        self._initial_velocity = None
        self._dummy_step = 1e-5
        self._last_observation = None
        self._last_action = None
        self._last_reward = None
        self._last_info = None
        self._is_closed = False

        # Initialize profiler.
        self._profiler = Profiler()
        self._profiler.enabled = cfg.profile

        # Initialize Mochi Physics context (only the first time).
        self._init_context(cfg)

        # Initialize the scene manager.
        if cfg.use_shared_scenes:
            self._scene_manager = SceneManager.get_instance(self.__class__.__name__)

        # Initialize renderer.
        self.render_mode = cfg.render_mode
        self._render_size = cfg.render_size
        self._render_coordinate_system = cfg.render_coordinate_system
        self._render_start_paused = cfg.start_paused
        self._render_scene_dirty = False
        if self.render_mode is not None:
            self._init_renderer()

    def __del__(self):
        """Destructor for the environment, automatically calls `close` to release the
        Mochi scene and any other allocated resources.

        Best-effort fallback for consumers that don't call ``close()`` explicitly.
        ``close()`` skips only the Mochi scene teardown when the context has
        already been finalized (interpreter shutdown, after ``physics.shutdown()``);
        the renderer teardown is Mochi-context-independent and always runs.
        """
        self.close()

    ####################################################################################
    # API from Gymnasium's Env
    ####################################################################################

    @override_from(Env)
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> ResetResult:
        """
        Resets the environment and returns an initial observation.
        """

        if self._is_closed:
            raise ClosedEnvironmentError()

        with self._profiler.enter("reset"):
            # Ensure observation and action spaces were set during initialization.
            assert self._observation_space_structure is not None
            assert self._action_space_structure is not None

            # We need the following line to seed self.np_random.
            super().reset(seed=seed, options=options)

            # Increment episode count and reset step count and truncated/terminated state.
            self._episode += 1
            self._step_count = 0
            self._truncated = False
            self._terminated = False

            # Reset Mochi scene (calls the derived class).
            with self._profiler.enter("reset_scene"):
                self._reset_scene()

            # Perform a very small step to initialize step-dependent data (e.g. contacts)
            self._check_valid_scene()
            self._scene.step(self._dummy_step)

            # Store the current state of the scene.
            if self._state_snapshot is not None:
                # Release first, in case it wasn't released previously. Redundant release is OK.
                self._scene.release_state(self._state_snapshot)
                self._state_snapshot = self._scene.capture_state()

            # Generate initial observation.
            struct_observation, info = self._make_observation()
            self._last_observation = struct_observation
            self._last_action = None
            self._last_reward = None
            self._last_info = info

            # Reset the renderer. Given the scene may have changed, flag that the
            # renderer scene is dirty and needs to be reset.
            if self._renderer:
                self._render_scene_dirty = True
                if self.render_mode == "human":
                    self.render()

        # Append profiled section to info.
        if self._profiler.enabled and self._dump_timings_to_info:
            info.update(self._profiler["reset"].to_dict("timings"))

        # Return reset results.
        return ResetResult(
            observation=self.to_observation(struct_observation),
            info=info,
        )

    @override_from(Env)
    def step(self, action: Action) -> StepResult:
        """
        Runs one control timestep and returns the observation, reward, termination, and
        truncation flags, as well as any additional information about the current state
        of the environment.
        """

        if self._is_closed:
            raise ClosedEnvironmentError()

        with self._profiler.enter("step"):
            # Project the given action to the boundaries defined in the action space.
            action = np.clip(action, self.action_space.low, self.action_space.high)

            # Run the simulation applying the given action.
            struct_action = self.to_structured_action(action=action)
            with self._profiler.enter("simulate"):
                self._simulate(action=struct_action)
            self._step_count += 1

            # Compute observation and reward, check for stop criteria.
            with self._profiler.enter("make_observation"):
                struct_observation, info = self._make_observation()
            with self._profiler.enter("compute_reward_terms"):
                reward_terms = self._compute_reward_terms(
                    action=struct_action,
                    observation=struct_observation,
                    info=info,
                )
            with self._profiler.enter("check_stop_criteria"):
                self._check_stop_criteria(
                    action=struct_action,
                    observation=struct_observation,
                    reward=reward_terms,
                    info=info,
                )

            # Store last observation, action, reward and info.
            self._last_observation = struct_observation
            self._last_action = struct_action
            self._last_reward = reward_terms
            self._last_info = info

            # Update the renderer.
            if self.render_mode == "human":
                self.render()

        # Append profiled section to info.
        if self._profiler.enabled and self._dump_timings_to_info:
            info.update(self._profiler["step"].to_dict("timings"))

        # Return step results.
        return StepResult(
            observation=self.to_observation(struct_observation),
            reward=sum(reward_terms.values()),
            terminated=self._terminated,
            truncated=self._truncated,
            info={**info, **{f"reward_{k}": v for k, v in reward_terms.items()}},
        )

    @override_from(Env)
    def render(self) -> RenderFrame | None:
        """
        Returns the current render frame. If the environment is not being rendered, this
        function will return None.
        """

        if self._is_closed:
            raise ClosedEnvironmentError()

        with self._profiler.enter("render"):
            # Exit if no renderer available.
            if self._renderer is None:
                return None

            # Reset renderer if the scene is dirty (i.e. the environment has been reset
            # at least once since the last render).
            if self._render_scene_dirty:
                with self._profiler.enter("reset"):
                    self._renderer.set_scene(self._scene)
                    self._renderer.snap_follow_camera()
                    self._reset_renderer()
                self._render_scene_dirty = False

            # With the scene reset, we can now render the current frame.
            with self._profiler.enter("update"):
                self._update_renderer()
                return self._renderer.render()

    @override_from(Env)
    def close(self):
        """
        Closes the environment, shutting down the Mochi scene and releasing any
        resources allocated by the environment. Calling close on a already closed
        environment has no effect and won't raise an error.
        """

        # Shut down the renderer independently of the physics context.
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        # Destroy the Mochi scene. Skip when the context has already been
        # finalized (e.g. GC firing __del__ at interpreter shutdown, after
        # physics.shutdown()); destroy_scene requires a live context and would
        # otherwise raise.
        if self._scene is not None and sdp.is_initialized():
            self._release_scene()

        # Flag as closed.
        self._is_closed = True

    def is_closed(self) -> bool:
        """Returns whether the environment is closed (i.e. `close` was called)."""
        return self._is_closed

    ####################################################################################
    # Functions handling environment initialization and resetting.
    ####################################################################################

    def _init_context(self, cfg: MochiEnvCfg):
        """Initializes the Mochi context. This is done only once, for the first
        environment in this process."""

        # Initialize the Mochi context if not already initialized.
        if not sdp.is_initialized():
            logger.info(
                f"Initializing Mochi context on process {os.getpid()} with "
                f"{cfg.num_worker_threads} worker threads..."
            )
            sdp.initialize(num_worker_threads=cfg.num_worker_threads)
            logger.info("Mochi context initialized.")

            forward_mochi_logs_to_logger(logger)
            logger.info("Forwarded Mochi logs to Python logger.")

        # Enable the file cache so that each model is only loaded once and shared
        # between actors and scenes.
        sdp.enable_file_cache(True)

    def _load_scene(
        self,
        scene_name: str,
        scene_factory: Callable[
            [],
            tuple[sdp.Scene, sdp.Actor]
            | tuple[sdp.Scene, sdp.Actor, list[Callable[[], None]]],
        ],
    ):
        """
        Loads a Mochi scene with the environment and handles scene sharing optimization.

        This method creates or retrieves a scene based on the environment's scene
        sharing configuration. When scene sharing is enabled, multiple environment
        instances can reuse the same scene to reduce memory usage. The scene_name serves
        as a unique identifier for lookup and registration purposes. If a scene with the
        given name does not exist, a new one is created and registered through the
        given scene factory. The scene factory is a callable that returns a tuple
        containing a new Scene and the selected agent Actor.

        Scene sharing works by assuming environments are stepped sequentially, which is
        true for batched execution like HybridVectorEnv but may not hold for other
        vectorization approaches. When sharing is enabled, the SceneManager handles
        the initial state snapshot to ensure consistent reset behavior across all
        environment instances using the shared scene.

        Note that if a scene has already been loaded prior to calling this method, it
        will automatically be released. Any dangling references to the previous scene
        should be handled by the derived environment class.
        """

        perform_initialization_step = False

        # Release the previous scene if any.
        if self._scene is not None:
            self._release_scene()

        # If scene sharing is disabled for the environment, always create a new scene.
        if self._scene_manager is None:
            assert self._initial_state_snapshot is None

            def adopt(scene, agent, cleanup_callbacks):
                # Capture first: on failure the scene is torn down and the environment
                # must not be left holding a reference to it.
                self._initial_state_snapshot = scene.capture_state()
                self._scene = scene
                self._agent = agent
                self._scene_cleanup_callbacks = cleanup_callbacks

            self._create_scene(scene_name, scene_factory, adopt)
            perform_initialization_step = True

        # Otherwise, find the scene in the scene manager and store references to shared
        # objects. If not found, create the scene and register it.
        else:
            scene_manager = self._scene_manager
            self._scene_name = scene_name
            scene_data = scene_manager.try_find_scene(scene_name)
            if scene_data is None:
                scene_data = self._create_scene(
                    scene_name,
                    scene_factory,
                    lambda scene, agent, cleanups: scene_manager.register_scene(
                        scene_name, scene, agent, cleanups
                    ),
                )
                perform_initialization_step = True
                logger.debug("Scene created and registered to scene.")
            else:
                logger.debug(f'Found shared scene "{scene_name}", reusing it.')
            self._scene = scene_data.scene
            self._agent = scene_data.agent
            self._initial_state_snapshot = scene_data.initial_state
            self._state_snapshot = sdp.StateHandle()
            self._scene.restore_state(
                handle=self._initial_state_snapshot, release_immediately=False
            )

        # Perform dummy step to force the scene initialization to complete.
        # Only needed if the scene was just created.
        if perform_initialization_step:
            self._scene.step(self._dummy_step)
            self._scene.restore_state(
                handle=self._initial_state_snapshot, release_immediately=False
            )

    def _create_scene(self, scene_name, scene_factory, adopt):
        """Builds and validates a scene, then hands it to ``adopt`` for ownership.

        ``adopt`` takes ``(scene, agent, cleanup_callbacks)`` and returns whatever the
        caller needs. If the factory result is malformed, or ``adopt`` raises, the
        partially constructed scene is torn down before the error propagates.
        """
        logger.debug(f'Creating scene "{scene_name}"...')
        result = scene_factory()
        if len(result) == 2:
            scene, agent = result
            cleanup_callbacks = []
        elif len(result) == 3:
            scene, agent, cleanup_callbacks = result
        else:
            raise ValueError(
                "Scene factory must return (scene, agent) or "
                "(scene, agent, cleanup_callbacks)."
            )
        cleanup_callbacks = list(cleanup_callbacks)
        logger.debug("Scene created.")
        try:
            if not isinstance(scene, sdp.Scene):
                raise TypeError("Scene factory must return a valid Mochi scene.")
            if not isinstance(agent, sdp.Actor):
                raise TypeError("Scene factory must return a valid Mochi agent.")
            return adopt(scene, agent, cleanup_callbacks)
        except Exception:
            try:
                destroy_scene_with_cleanup(scene, None, cleanup_callbacks)
            except Exception:
                logger.exception("Failed to clean up scene after construction error.")
            raise

    def _release_scene(self):
        """
        Releases the current Mochi scene. This function must be called whenever the
        environment wants to release the scene (e.g. when closing the environment, or
        prior to loading a new one).
        """

        # Exit if no scene to release.
        if self._scene is None:
            raise RuntimeError("No scene to release")

        # Try destroy the current scene.
        # If using a shared scene through scene manager, release the scene instead.
        if self._scene_name is not None:
            # NOTE: Do not release _initial_state_snapshot since it might be shared.
            if self._state_snapshot is not None:
                self._scene.release_state(self._state_snapshot)
            scene_name = self._scene_name
            self._scene = None
            self._agent = None
            self._initial_state_snapshot = None
            self._state_snapshot = None
            self._scene_cleanup_callbacks = []
            self._scene_name = None
            self._scene_manager.release_scene(scene_name)
        else:
            scene = self._scene
            initial_state_snapshot = self._initial_state_snapshot
            cleanup_callbacks = self._scene_cleanup_callbacks
            self._scene = None
            self._agent = None
            self._initial_state_snapshot = None
            self._state_snapshot = None
            self._scene_cleanup_callbacks = []

            destroy_scene_with_cleanup(scene, initial_state_snapshot, cleanup_callbacks)

    def _reset_scene(self):
        """
        Resets the Mochi scene. This function is called each time the environment is
        reset (i.e. at the start of the episode). Derived environments can override
        this function to (re)load the desired Mochi scene or modify the default
        behavior.
        """

        self._check_valid_scene()

        # Warn the user if the environment is using scene sharing but the current scene
        # is not registered in the scene manager. TODO: Deprecate once we enforce users
        # to handle their scenes through the scene manager.
        if self._scene_manager is not None and self._scene_name is None:
            warnings.warn(
                "The environment has scene sharing enabled, but the current scene is "
                "not registered in the scene manager. In the future, this path will be "
                "marked as deprecated. Please update the environment to handle scene "
                "loading through `_load_scene`. or disable scene sharing to suppress "
                "this warning.",
                category=RuntimeWarning,
                stacklevel=3,
            )

        # Reset all objects in the scene (including the agent) to their original pose.
        self._scene.restore_state(
            handle=self._initial_state_snapshot, release_immediately=False
        )

        # Initialize the pose and velocity of the agent, possibly with noise.
        if self._initial_pose is None:
            raise RuntimeError("Derived classes must initialize the initial pose")
        if self._initial_velocity is None:
            raise RuntimeError("Derived classes must initialize the initial velocity")

        min_pose_limits, max_pose_limits = mochi_helpers.get_articulated_dof_limits(
            self._agent
        ).T
        pose = np.copy(self._initial_pose)
        pose += self._reset_noise_scale * self.np_random.uniform(
            -1.0, 1.0, size=pose.shape
        )
        pose = np.clip(pose, min_pose_limits, max_pose_limits)
        self._agent.set_articulated_pose_from_joints(pose)

        vel = np.copy(self._initial_velocity)
        vel += self._reset_noise_scale * self.np_random.standard_normal(size=vel.shape)
        self._agent.set_articulated_joint_velocities(vel)

    ####################################################################################
    # Functions handling time stepping and sampling of the environment.
    ####################################################################################

    def _simulate(self, action: StructuredAction):
        """
        Runs the simulation by a number of steps (determined by the simulation and
        control frequencies) applying the given action. If you need to perform any
        action before or after the simulation, you can override this function.
        """

        if self._state_snapshot is not None:
            with self._profiler.enter("restore_state"):
                self._scene.restore_state(
                    handle=self._state_snapshot, release_immediately=True
                )

        for _ in range(self._sim_substeps):
            self._check_valid_scene()
            self._apply_action(action)
            with self._profiler.enter("step_scene"):
                self._scene.step(self._sim_dt)

            if self._profiler.enabled:
                stats = self._scene.get_performance_stats()
                total_step = stats.total_step_duration_sec
                solve_step = stats.solve_step_duration_sec
                pre_callbacks = stats.pre_step_callbacks_duration_sec
                post_callbacks = stats.post_step_callbacks_duration_sec
                recording_step = stats.recording_step_duration_sec
                self._profiler.record("cpp_total_step", 1e3 * total_step)
                self._profiler.record("cpp_solve_step", 1e3 * solve_step)
                self._profiler.record("cpp_pre_callbacks", 1e3 * pre_callbacks)
                self._profiler.record("cpp_post_step_callbacks", 1e3 * post_callbacks)
                self._profiler.record("cpp_recording_step", 1e3 * recording_step)

        if self._state_snapshot is not None:
            with self._profiler.enter("capture_state"):
                self._state_snapshot = self._scene.capture_state()

    @abc.abstractmethod
    def _apply_action(self, action: StructuredAction):
        """
        Applies the given action to the environment. The input action is assumed to be a
        dictionary with the same keys as the action space structure.
        """
        pass

    @abc.abstractmethod
    def _make_observation(self) -> tuple[StructuredObservation, Info]:
        """
        Samples the environment and returns a structured observation and additional info
        about the environment. The output structured observation is assumed to be a
        dictionary with the same keys as the observation space structure.
        """
        pass

    @abc.abstractmethod
    def _compute_reward_terms(
        self, action: StructuredAction, observation: StructuredObservation, info: Info
    ) -> RewardTerms:
        """
        Computes the reward terms given the applied action, environment observation
        and any additional information about the current state of the environment.
        """
        pass

    def _check_stop_criteria(
        self,
        action: StructuredAction,
        observation: StructuredObservation,
        reward: RewardTerms,
        info: Info,
    ):
        """
        Checks whether the episode should be truncated or terminated.
        """

        # Check episode truncation if the step count has been reached.
        if self._step_count == self._steps_per_episode:
            info["truncated_reason"] = "Steps per episode reached"
            self._truncated = True

    def get_control_frequency(self) -> int:
        """Returns the rate at which the environment is controlled (i.e. the rate at
        which the policy is sampled). This must be a multiple of the simulation
        frequency."""
        return self._control_frequency

    def get_simulation_frequency(self) -> int:
        """Returns the rate at which the environment is simulated. This must be a
        multiple of the control frequency."""
        return self._simulation_frequency

    def get_control_timestep(self) -> float:
        """Returns the control timestep (i.e. the inverse of the control frequency)."""
        return self._control_dt

    def get_simulation_timestep(self) -> float:
        """Returns the simulation timestep (i.e. the inverse of the simulation frequency)"""
        return self._sim_dt

    def get_control_to_simulation_ratio(self) -> int:
        """Returns the ratio between the control and simulation frequencies. This
        represents the number of simulation steps per control step."""
        return self._sim_substeps

    def get_step_count(self) -> int:
        """Returns the current step. This is the number of control steps since the
        environment was reset."""
        return self._step_count

    def get_steps_per_episode(self) -> int:
        """Returns the number of steps per episode. By default, the episode is truncated
        when the step count reaches this value. To override this behavior, refer to the
        check_stop_criteria method."""
        return self._steps_per_episode

    def get_episode(self) -> int:
        """Returns the current episode count since the environment was created."""
        return self._episode

    def get_last_step(self) -> StructuredStepResult:
        """Returns the results generated from the last call to the step or reset
        methods. Note that, unlike `step` and `reset`, the returned results are provided
        in structured form. If you need the results in flattened form, you can use the
        `to_observation` and `to_action` methods to convert them."""
        return StructuredStepResult(
            action=self._last_action,
            observation=self._last_observation,
            reward=self._last_reward,
            terminated=self._terminated,
            truncated=self._truncated,
            info=self._last_info,
        )

    ####################################################################################
    # Functions customizing the rendering of the environment.
    ####################################################################################

    def _init_renderer(self):
        """
        Initializes the renderer. This function is called only once at the start of the
        environment. Derived environments can override this function to use a different
        renderer (e.g., UnrealCV for Unreal Engine rendering).
        """
        logger.info("Initializing renderer...")

        cfg = ViewerCfg(
            size=self._render_size,
            coordinate_system=self._render_coordinate_system,
            offscreen=self.render_mode == "rgb_array",
        )
        self._renderer = Viewer(cfg)
        self._renderer.set_paused(self._render_start_paused)
        self._init_ui()

    def _reset_renderer(self):
        """
        Function called whenever the renderer is reset. Depending on the render mode,
        this can happen at the start of the environment (human rendering mode), or
        whenever a render frame is requested (RGB array mode). Derived environments can
        override this function to customize the renderer after reset (e.g. to introduce
        auxiliary visualizations). It is guaranteed that this function will be called
        only if the renderer is available, and that the renderer's reset method has been
        called prior to this function.
        """
        pass

    def _update_renderer(self):
        """
        Function called whenever the renderer is updated. Depending on the render mode,
        this can happen at the end of a step (human rendering mode), or whenever a
        render frame is requested (RGB array mode). Derived environments can override
        this function to customize the renderer after a step (e.g. to introduce auxiliary
        visualizations, such as force vectors). It is guaranteed that this function will
        be called only if the renderer is available, and that the renderer's render
        method will be called after this function.
        """
        pass

    def get_renderer(self) -> Viewer | None:
        """Returns the object handling environment rendering. If the environment is not
        being rendered, None will be returned instead."""
        return self._renderer

    ####################################################################################
    # Functions customizing the renderer UI.
    ####################################################################################

    def _init_ui(self):
        """
        Initializes a custom UI tab for the environment. This function is called only if
        the renderer is available, and after the renderer has been initialized. Derived
        environments can override this function to prevent the default environment UI
        from being displayed, or to add additional UI tabs.
        """
        self._renderer.add_ui_tab("Environment", self._build_environment_panel)

    def _build_environment_panel(self):
        """
        Builds the default environment UI panel. This function is called only if the
        renderer is available, and after the renderer has been initialized. Derived
        environments can override this function to customize the default environment UI
        panel.
        """
        # Show the current episode and step count.
        psim.TextDisabled("ENVIRONMENT")
        # - Episode count.
        psim.InputInt("Episode", self._episode, flags=psim.ImGuiInputTextFlags_ReadOnly)
        # - Step count.
        progress = self._step_count / self._steps_per_episode
        psim.ProgressBar(progress, (0, 0))
        psim.SameLine()
        psim.Text(f"{self._step_count}/{self._steps_per_episode}")
        # - Simulation frequency.
        psim.InputFloat(
            "Simulation Freq.",
            self._simulation_frequency,
            flags=psim.ImGuiInputTextFlags_ReadOnly,
        )
        # - Control frequency.
        psim.InputFloat(
            "Control Freq.",
            self._control_frequency,
            flags=psim.ImGuiInputTextFlags_ReadOnly,
        )
        # - Simulation to control ratio.
        psim.InputInt(
            "Ratio",
            self._sim_substeps,
            flags=psim.ImGuiInputTextFlags_ReadOnly,
        )
        widgets.vertical_block_spacing()

        # Build UI for the last observation, action, reward and info.
        psim.TextDisabled("LAST STEP")
        if psim.CollapsingHeader("Observation"):
            if self._last_observation is None:
                psim.TextDisabled("No observation available")
            else:
                psim.PushID("Observation")
                for key, values in self._last_observation.items():
                    lower_limits = self._observation_space_structure[key].low
                    upper_limits = self._observation_space_structure[key].high
                    widgets.ndarray_inspector(
                        key, values, lower_limits, upper_limits, read_only=True
                    )
                psim.PopID()
        if psim.CollapsingHeader("Action"):
            if self._last_action is None:
                psim.TextDisabled("No action available")
            else:
                psim.PushID("Action")
                for key, values in self._last_action.items():
                    lower_limits = self._action_space_structure[key].low
                    upper_limits = self._action_space_structure[key].high
                    widgets.ndarray_inspector(
                        key, values, lower_limits, upper_limits, read_only=True
                    )
                psim.PopID()
        if psim.CollapsingHeader("Reward"):
            if self._last_action is None:
                psim.TextDisabled("No action available")
            else:
                psim.PushID("Reward")
                widgets.dict_inspector(self._last_reward, read_only=True)
                psim.PopID()
        if psim.CollapsingHeader("Info"):
            if self._last_info is None:
                psim.TextDisabled("No info available")
            else:
                psim.PushID("Info")
                widgets.dict_inspector(self._last_info, read_only=True)
                psim.PopID()
        if psim.Button("Copy to Clipboard"):
            contents = {
                "observations": self._last_observation,
                "actions": self._last_action,
                "rewards": self._last_reward,
                "info": self._last_info,
            }
            psim.SetClipboardText(json.dumps(contents, cls=ExtendedJSONEncoder))

    ####################################################################################
    # Helper functions converting between structured observations/actions to flattened
    # observations/actions. Note these are internally used by the base environment, and
    # derived environments do not need to use them, unless the step method is overridden.
    ####################################################################################

    def _setup_observation_space(self, **structure: ObservationSpace):
        """
        Sets up the observation space of the environment. This function should be called
        only on environment initialization.
        """
        self._observation_space_structure = ObservationSpaceStructure(structure)
        self.observation_space = spaces.flatten_space(self._observation_space_structure)

    def _setup_action_space(self, **structure: ActionSpace):
        """
        Sets up the action space of the environment. This function should be called only
        on environment initialization.
        """
        self._action_space_structure = ActionSpaceStructure(structure)
        self.action_space = spaces.flatten_space(self._action_space_structure)

    def to_observation(self, obs: StructuredObservation) -> Observation:
        """
        Converts the given structured observation to a flattened observation. The input
        observation is assumed to be a dictionary with the same keys as the observation
        space structure.
        """
        flattened_obs = spaces.flatten(self._observation_space_structure, obs)
        assert flattened_obs.shape == self.observation_space.shape
        return flattened_obs

    def to_structured_observation(self, obs: Observation) -> StructuredObservation:
        """
        Converts the given flattened observation to a structured observation. The output
        observation is assumed to be a dictionary with the same keys as the observation
        space structure.
        """
        return spaces.unflatten(self._observation_space_structure, obs)

    def to_action(self, action: StructuredAction) -> Action:
        """
        Converts the given structured action to a flattened action. The input action is
        assumed to be a dictionary with the same keys as the action space structure.
        """
        flattened_action = spaces.flatten(self._action_space_structure, action)
        assert flattened_action.shape == self.action_space.shape
        return flattened_action

    def to_structured_action(self, action: Action) -> StructuredAction:
        """
        Converts the given flattened action to a structured action. The output action is
        assumed to be a dictionary with the same keys as the action space structure.
        """
        return spaces.unflatten(self._action_space_structure, action)

    def get_observation_space(self) -> ObservationSpace:
        """
        Returns the observation space of the environment. Note calling this method
        is equivalent to accessing the observation_space attribute directly. We provide
        this method for consistency, and also convenience when using Ray remote workers,
        which do not support accessing actor attributes directly.
        """
        return self.observation_space

    def get_observation_space_structure(self) -> ObservationSpaceStructure:
        """Returns the observation space structure of the environment."""
        return self._observation_space_structure

    def get_action_space(self) -> ActionSpace:
        """
        Returns the action space of the environment. Note calling this method is
        equivalent to accessing the action_space attribute directly. We provide this
        method for consistency, and also convenience when using Ray remote workers, which
        do not support accessing actor attributes directly.
        """
        return self.action_space

    def get_action_space_structure(self) -> ActionSpaceStructure:
        """
        Returns the action space structure of the environment.
        """
        return self._action_space_structure

    ####################################################################################
    # Private methods and helpers
    ####################################################################################

    def _check_valid_scene(self):
        if self._scene is None:
            raise RuntimeError("There is no scene to run.")

    @deprecated(
        "`_post_init_scene` is deprecated and will be removed in a future version. "
        "Post-scene initialization is now handled internally by the environment "
        "during `_load_scene`. Please update your code accordingly."
    )
    def _post_init_scene(self):
        pass

    ####################################################################################
    # Other properties and attributes
    ####################################################################################

    def get_profiler(self) -> Profiler:
        """Returns the profiler measuring the step times."""
        return self._profiler
