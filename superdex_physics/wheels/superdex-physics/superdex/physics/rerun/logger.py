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
from pathlib import Path
from typing import TYPE_CHECKING

import rerun as rr
from superdex.physics import Actor, ActorHandle, Scene, SceneHandle
from superdex.physics.rerun.logger_cfg import RerunLoggerCfg

if TYPE_CHECKING:
    from superdex.physics.rerun.overlay import OverlayLogger
from superdex.physics.utils.coordinate_systems import (
    COORDINATE_SYSTEMS,
    CoordinateSystem,
    CoordinateTransform,
    DEFAULT_COORDINATE_SYSTEM,
)

logger = logging.getLogger(__name__)

########################################################################################


class RerunLogger:
    """
    Logs Mochi scenes and actors to rerun for visualization.

    Unlike the Viewer class which uses polyscope for interactive rendering,
    RerunLogger sends data to the rerun SDK for logging and visualization.
    This enables:
    - Recording simulations to .rrd files for later playback
    - Remote visualization via gRPC connection
    - Timeline scrubbing and temporal analysis
    - Integration with other rerun-based pipelines
    """

    ####################################################################################
    # Members
    ####################################################################################

    _cfg: RerunLoggerCfg
    _coordinate_system: CoordinateSystem
    _coordinate_transform: CoordinateTransform
    _scene: Scene | None
    _scene_handle: SceneHandle | None
    _actor_loggers: dict
    _debug_draw_logger: object | None
    _frame_count: int
    _sim_time_origin: float  # Scene clock at the last reset_scene(); see log_frame
    _sim_time: float  # Episode-relative sim time of the last logged frame
    _spawned: bool  # Whether a local viewer has already been spawned

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(self, cfg: RerunLoggerCfg | None = None) -> None:
        """Initialize the RerunLogger.

        Args:
            cfg: Configuration for the logger. If None, defaults will be used.
        """
        if cfg is None:
            cfg = RerunLoggerCfg()

        # Resolve coordinate system.
        # Rerun uses the same coordinate system as polyscope (RUB: right-handed, Y-up).
        target_system = DEFAULT_COORDINATE_SYSTEM
        coordinate_system = cfg.coordinate_system
        if coordinate_system is None:
            coordinate_system = DEFAULT_COORDINATE_SYSTEM
        elif isinstance(coordinate_system, str):
            if coordinate_system not in COORDINATE_SYSTEMS:
                available_presets = ", ".join(f'"{cs}"' for cs in COORDINATE_SYSTEMS)
                raise ValueError(
                    f'Invalid coordinate system "{coordinate_system}". '
                    "You must provide a valid coordinate system object, or a "
                    "valid coordinate system preset, or None to use the default "
                    f"coordinate system. Available presets are: {available_presets}."
                )
            coordinate_system = COORDINATE_SYSTEMS[coordinate_system]

        self._cfg = cfg
        self._coordinate_system = coordinate_system
        self._coordinate_transform = CoordinateTransform(
            coordinate_system, target_system
        )
        self._scene = None
        self._scene_handle = None
        self._actor_loggers = {}
        self._debug_draw_logger = None
        self._frame_count = 0
        self._sim_time_origin = 0.0
        self._sim_time = 0.0
        self._spawned = False

        # Initialize rerun recording.
        self._init_rerun()

    def __del__(self) -> None:
        """Closes the logger."""
        self.close()

    ####################################################################################
    # Rerun initialization
    ####################################################################################

    def _init_rerun(self) -> None:
        """Create a global recording via ``rr.init()`` and attach configured sinks."""
        logger.info("Initializing rerun backend...")
        logger.info(f"Rerun version: {rr.__version__}")

        rr.init(
            application_id=self._cfg.application_id,
            recording_id=self._cfg.recording_id,
        )
        if self._cfg.recording_name is not None:
            rr.send_recording_name(self._cfg.recording_name)
        self._attach_sinks()

    def _attach_sinks(self, save_path: str | Path | None = None) -> None:
        """Attach save/connect/spawn sinks to the current global recording.

        A ``RecordingStream`` supports multiple sinks, so a viewer stream and a
        file save are attached together in a single ``rr.set_sinks`` call (a
        "tee").  Issuing ``rr.save`` and ``rr.connect_grpc`` sequentially would
        instead leave only the last sink active, silently dropping either the
        file or the viewer stream.

        Args:
            save_path: If provided, overrides ``cfg.save_path`` for this call.
        """
        save_path = save_path if save_path is not None else self._cfg.save_path
        save_path = Path(save_path) if save_path is not None else None

        # Combine the viewer (connect/spawn) and save (file) sinks into one tee
        # so neither replaces the other.
        sinks: list = []
        if self._cfg.spawn:
            # Launch the local viewer only once; later recordings (e.g. new
            # per-episode recordings) stream to the already-spawned viewer via
            # its default gRPC sink rather than spawning another process.
            if not self._spawned:
                try:
                    rr.spawn()
                    self._spawned = True
                    logger.info("Spawned local rerun viewer")
                except Exception as e:
                    logger.warning(f"Failed to spawn rerun viewer: {e}")
                    return
                # rr.spawn() already attached a viewer sink to the current
                # recording; only a file sink (if any) needs teeing in.
                if save_path is not None:
                    rr.set_sinks(rr.GrpcSink(), rr.FileSink(str(save_path)))
                    logger.info(f"Saving recording to {save_path}")
                return
            sinks.append(rr.GrpcSink())
            logger.info("Streaming to spawned rerun viewer")
        elif self._cfg.connect:
            addr = self._cfg.connect_addr or "127.0.0.1:9876"
            sinks.append(rr.GrpcSink(url=f"rerun+http://{addr}/proxy"))
            logger.info(f"Streaming to rerun viewer via gRPC at {addr}")
        if save_path is not None:
            sinks.append(rr.FileSink(str(save_path)))
            logger.info(f"Saving recording to {save_path}")

        if sinks:
            try:
                rr.set_sinks(*sinks)
            except Exception as e:
                logger.warning(f"Failed to attach rerun sinks: {e}")

    ####################################################################################
    # Scene management
    ####################################################################################

    def get_scene(self) -> Scene | None:
        """Returns the Mochi scene associated with the logger."""
        return self._scene

    def get_sim_time(self) -> float:
        """Returns the sim_time [s] of the most recently logged frame.

        Measured from the scene clock captured by the last :meth:`reset_scene`,
        or from 0 for a logger that never reset. 0.0 until the first frame of
        an episode is logged. Lets a caller logging its own entities stamp them
        on the same ``sim_time`` as the scene pose they describe, instead of
        re-deriving a value that can drift from the timeline.
        """
        return self._sim_time

    def set_scene(self, scene: Scene | None) -> None:
        """Sets the Mochi scene associated with the logger.

        Args:
            scene: The Mochi scene to log, or None to clear.
        """
        # Perform updates only if the given scene is different.
        handle = None if scene is None else scene.get_handle()
        if handle == self._scene_handle:
            return

        # Update Mochi scene instance.
        self._scene = scene
        self._scene_handle = handle
        # The origin belongs to the scene it was captured from; carrying it
        # over to a different scene would stamp that scene's clock against a
        # foreign reference, going negative whenever the new scene is younger.
        self._sim_time_origin = 0.0
        self._sim_time = 0.0

        # Clear existing loggers if no scene was given.
        if scene is None:
            self._clear_actor_loggers()
            self._debug_draw_logger = None
            return

        # Create new loggers for the scene.
        self._update_actor_loggers()
        if self._cfg.log_debug_draw:
            self._create_debug_draw_logger()

    def reset_scene(
        self,
        scene: Scene | None = None,
        *,
        new_recording: bool = False,
        recording_id: str | None = None,
        save_path: str | Path | None = None,
    ) -> "rr.RecordingStream | None":
        """Reset the logger for a new episode or scene.

        Resets the frame counter, clears actor loggers, logs static geometry,
        and recreates actor loggers for the current or provided scene.

        Args:
            scene: Optional Mochi scene to set. If provided, replaces the
                current scene. If None, reuses the existing scene.
            new_recording: If True, create a fresh Rerun recording via
                ``rr.RecordingStream(make_default=True)``, installing it as
                the global recording.
                This gives the new episode an independent timeline.
                Sinks (connect/spawn/save) are attached automatically
                from the logger's configuration via ``_attach_sinks``.
            recording_id: Optional recording ID for the new recording
                (only used when *new_recording* is True).  Appears as the
                recording name in the viewer's recording selector.
            save_path: Optional .rrd save path for this recording.
                Overrides ``cfg.save_path`` for this call only.

        Returns:
            The new ``rr.RecordingStream`` if *new_recording* is True,
            otherwise None.
        """
        self._frame_count = 0

        # Clear existing actor loggers so they get recreated.
        # Note: We don't call clear() to avoid logging rr.Clear commands
        # that erase previously logged data from the viewer timeline.
        self._actor_loggers.clear()
        self._debug_draw_logger = None

        # Optionally create a new recording for an independent timeline.
        rec = None
        if new_recording:
            rec = rr.RecordingStream(
                application_id=self._cfg.application_id,
                recording_id=recording_id,
                make_default=True,
            )
            if self._cfg.recording_name is not None:
                rr.send_recording_name(self._cfg.recording_name, recording=rec)
            self._attach_sinks(save_path=save_path)

        # Set or update scene if provided.
        if scene is not None:
            self._scene = scene
            self._scene_handle = scene.get_handle()

        # Log static geometry (after scene is set so future changes can use it).
        self.log_static_geometry()

        # Recreate actor loggers.
        self._update_actor_loggers()
        if self._cfg.log_debug_draw:
            self._create_debug_draw_logger()

        # Anchor this episode's sim_time at the scene's current clock, which
        # already includes any settling stepped before the episode is recorded.
        # Must be captured after any restore_state, which rewinds the clock.
        self._sim_time_origin = (
            self._scene.get_total_simulation_time() if self._scene is not None else 0.0
        )
        self._sim_time = 0.0

        # Seed the timeline at sim_time=0 with the scene's current pose so every
        # actor has a valid transform from the very start of the recording.
        # Otherwise a caller that steps the scene before its first log_frame()
        # leaves the [0, first_frame) interval with no transform -- actors
        # render at identity (the world origin) until the first frame. Does not
        # advance the frame counter, so the caller's first log_frame(0) still
        # lands on frame 0.
        if self._scene is not None and self._cfg.use_timeline:
            rr.set_time("frame", sequence=0)
            rr.set_time("sim_time", duration=0.0)
            self._emit_actor_frame()

        return rec

    ####################################################################################
    # Logging methods
    ####################################################################################

    def log_frame(self, frame_idx: int | None = None) -> None:
        """Log the current scene state as a frame.

        The timestamp is automatically derived from the scene's total simulation time.
        If a frame_idx is provided, it will be used for the frame sequence timeline.
        Otherwise, an internal frame counter is used.

        Args:
            frame_idx: Optional frame index for timeline. If None, uses internal counter.
        """
        scene = self._scene
        if scene is None:
            return

        # Tracked even when the timeline is off, so get_sim_time() stays
        # meaningful for callers stamping their own entities.
        self._sim_time = scene.get_total_simulation_time() - self._sim_time_origin

        # Set timeline if enabled.
        if self._cfg.use_timeline:
            if frame_idx is None:
                frame_idx = self._frame_count
            rr.set_time("frame", sequence=frame_idx)
            rr.set_time("sim_time", duration=self._sim_time)

        # Update actor loggers, then emit their state (+ debug draw).
        self._update_actor_loggers()
        self._emit_actor_frame()

        self._frame_count += 1

    def log_scene_snapshot(self, *, frame_idx: int, sim_time: float) -> None:
        """Log the current scene pose without advancing the frame counter."""
        if self._scene is None:
            return

        self._sim_time = sim_time
        if self._cfg.use_timeline:
            rr.set_time("frame", sequence=frame_idx)
            rr.set_time("sim_time", duration=sim_time)

        self._update_actor_loggers()
        self._emit_actor_frame()

    def _emit_actor_frame(self) -> None:
        """Emit every actor's current state (and debug draw) at the current
        timeline position.

        Assumes the actor loggers are already up to date (call
        ``_update_actor_loggers`` first) and that the caller has positioned the
        timeline. Does not advance the internal frame counter.
        """
        for actor_logger in self._actor_loggers.values():
            actor_logger.log()

        if self._cfg.log_debug_draw and self._debug_draw_logger is not None:
            from superdex.physics.rerun.loggers.debug_draw_logger import DebugDrawLogger

            assert isinstance(self._debug_draw_logger, DebugDrawLogger)
            self._debug_draw_logger.log()

    def log_static_geometry(self) -> None:
        """Log static scene geometry (grids, reference frames, etc.)."""
        # Log world coordinate frame.
        rr.log(
            "world",
            rr.ViewCoordinates.RIGHT_HAND_Y_UP,
            static=True,
        )

    ####################################################################################
    # Actor logger management
    ####################################################################################

    def _update_actor_loggers(self) -> None:
        """Update actor loggers, creating new ones for new actors and removing
        loggers for removed actors."""
        if self._scene is None:
            return

        # Import here to avoid circular imports.
        from superdex.physics.rerun.loggers.actor_logger import ActorLogger

        # Gather all actors from the scene.
        current_actors: dict[ActorHandle, Actor] = {}

        def gather_actor(actor: Actor):
            handle = actor.get_handle()
            current_actors[handle] = actor

        self._scene.for_each_actor(gather_actor)

        # Determine which actors to add and remove.
        current_handles = set(current_actors.keys())
        existing_handles = set(self._actor_loggers.keys())
        handles_to_add = current_handles - existing_handles
        handles_to_remove = existing_handles - current_handles

        # Remove loggers for removed actors.
        for handle in handles_to_remove:
            self._actor_loggers[handle].clear()
            del self._actor_loggers[handle]

        # Create loggers for new actors.
        for handle in handles_to_add:
            actor = current_actors[handle]
            if not actor.get_surface_mesh().is_empty():
                layer = actor.get_contact_layer()
                self._actor_loggers[handle] = ActorLogger(
                    actor=actor,
                    coordinate_transform=self._coordinate_transform,
                    entity_path_prefix=f"world/{layer}",
                    mesh_assets_dir=self._cfg.mesh_assets_dir,
                )

    def _clear_actor_loggers(self) -> None:
        """Remove all actor loggers from the internal dictionary.

        Uses dict.clear() to empty the dictionary. Does NOT call each
        ActorLogger's clear() method, which would emit rr.Clear commands
        that erase previously logged data from the viewer timeline.
        """
        self._actor_loggers.clear()

    def _create_debug_draw_logger(self) -> None:
        """Create the debug draw logger."""
        if self._scene is None:
            return

        from superdex.physics.rerun.loggers.debug_draw_logger import DebugDrawLogger

        self._debug_draw_logger = DebugDrawLogger(
            scene=self._scene,
            coordinate_transform=self._coordinate_transform,
        )

    ####################################################################################
    # Overlay support
    ####################################################################################

    def create_overlay(
        self,
        name: str = "overlay",
        color: tuple[float, ...] = (1.0, 1.0, 1.0, 0.5),
        include_actors: list[str] | None = None,
        exclude_actors: list[str] | None = None,
        scene: Scene | None = None,
        flat_shading: bool = True,
        anchors: dict[str, str] | None = None,
        *,
        setup_frame: int = 0,
        setup_sim_time: float = 0.0,
    ) -> OverlayLogger:
        """Create a ghost/overlay visualization of the current scene.

        Clones mesh geometry from scene actors under a separate entity
        path prefix with configurable appearance. The returned
        OverlayLogger can then be driven with per-frame transforms from
        external data.

        Args:
            name: Overlay name for entity paths (e.g., 'observed').
            color: RGBA color tuple (0.0-1.0). Default: white at 50% opacity.
            include_actors: Actor/link name patterns to include (fnmatch).
                Cosmetic only for articulated links (mesh hidden, chain intact).
            exclude_actors: Actor/link name patterns to exclude (fnmatch).
                Cosmetic only for articulated links (see ``include_actors``).
            scene: Scene to clone from. Defaults to self._scene.
            flat_shading: Use per-face normals for crisp shading.
            anchors: Per-articulated-actor anchor link name. The overlay's
                entity tree starts at the anchor; upstream links are omitted.
                Supply the anchor's world pose per frame via
                ``OverlayLogger.set_articulated_pose``.
            setup_frame: Frame timeline value for construction-time poses.
            setup_sim_time: Simulation-time value [s] for construction-time poses.

        Returns:
            OverlayLogger ready for per-frame transform updates.

        Raises:
            ValueError: If no scene is set and none is provided.
        """
        from superdex.physics.rerun.overlay import OverlayCfg, OverlayLogger

        target_scene = scene or self._scene
        if target_scene is None:
            raise ValueError(
                "No scene available. Call set_scene() first or pass a scene argument."
            )

        # pyre-fixme[28]: pyre doesn't trace kwargs through the `@configclass`
        # decorator. The fields are valid dataclass kw-only arguments.
        cfg = OverlayCfg(
            name=name,
            color=color,
            include_actors=include_actors,
            exclude_actors=exclude_actors,
            flat_shading=flat_shading,
            anchors=anchors or {},
        )
        # OverlayLogger logs static structure + initial (dynamic) poses during
        # construction. Anchor the frame/sim_time cursor first so those
        # initial poses land on the real timelines rather than only log_time
        # (log_frame advances them from here). axis_length is logged static and
        # is unaffected by the cursor.
        if self._cfg.use_timeline:
            rr.set_time("frame", sequence=setup_frame)
            rr.set_time("sim_time", duration=setup_sim_time)
        return OverlayLogger(
            cfg=cfg,
            scene=target_scene,
            coordinate_transform=self._coordinate_transform,
        )

    ####################################################################################
    # Lifecycle management
    ####################################################################################

    def close(self) -> None:
        """Close the logger and finalize any recordings.

        Calling close on an already closed logger has no effect.
        """
        # Clear all loggers.
        self._clear_actor_loggers()
        self._debug_draw_logger = None
        self._scene = None
        self._scene_handle = None

    ####################################################################################
    # Context manager support
    ####################################################################################

    def __enter__(self) -> "RerunLogger":
        """Support with-statement for the logger."""
        return self

    def __exit__(self, *ignored) -> bool:
        """Support with-statement for the logger, closing at the end."""
        self.close()
        return False
