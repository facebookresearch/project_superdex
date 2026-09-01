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
UnrealCV-based viewer for SuperDex Lab gym environments.

This module provides a viewer implementation that uses UnrealCV to render mochi
simulations in Unreal Engine. It follows the same interface patterns as the
Polyscope-based Viewer class.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from fnmatch import fnmatchcase
from io import BytesIO
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from superdex.physics import Actor, ActorHandle, ActorType, Scene
from superdex.physics.utils.coordinate_systems import (
    COORDINATE_SYSTEMS,
    CoordinateSystem,
    CoordinateTransform,
    DEFAULT_COORDINATE_SYSTEM,
)
from superdex.physics.utils.profiling import Profiler
from superdex.physics.viewer.utils.aabb import AABB

from .unrealcv_client import UnrealCVClient
from .unrealcv_image import bgr_to_rgb
from .unrealcv_viewer_cfg import CaptureFormat, CaptureMode, UnrealCVViewerCfg
from .unrealcv_viewer_state import UnrealCVActorState, UnrealCVViewerState
from .updaters.unrealcv_actor_updater import UnrealCVActorUpdater
from .updaters.unrealcv_articulated_actor_updater import UnrealCVArticulatedActorUpdater
from .updaters.unrealcv_updater import UnrealCVUpdater

logger = logging.getLogger(__name__)

########################################################################################

RenderFrame = npt.NDArray[np.uint8]
"""A render frame, consists of an RGB image."""


class UnrealCVViewer:
    """
    Viewer implementation using UnrealCV to render mochi scenes in Unreal Engine.

    This viewer connects to a running Unreal Engine instance with the UnrealCV
    plugin and synchronizes mochi simulation state (actor transforms, camera
    position) with the UE scene.

    Requirements:
        - Unreal Engine project with UnrealCV plugin installed and running
        - Actors pre-placed in the UE level with names matching mochi actors
        - For deformable meshes: custom UnrealCV command handler in UE
    """

    ####################################################################################
    # Members
    ####################################################################################

    _cfg: UnrealCVViewerCfg
    _client: UnrealCVClient
    _state: UnrealCVViewerState

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(self, cfg: UnrealCVViewerCfg, profiler: Profiler | None = None):
        """
        Initialize the UnrealCV viewer and connect to Unreal Engine.

        Args:
            cfg: Configuration for the viewer.
            profiler: Optional Profiler instance for hierarchical timing. When
                provided, render() and _build_actor_commands() record their
                timings into this profiler instead of printing to stdout.

        Raises:
            RuntimeError: If connection to UnrealCV server fails.
        """
        self._cfg = cfg
        self._profiler = profiler

        # Resolve coordinate system
        unreal_system = COORDINATE_SYSTEMS["unreal"]
        source_system = cfg.coordinate_system
        if source_system is None:
            source_system = DEFAULT_COORDINATE_SYSTEM
        elif isinstance(source_system, str):
            if source_system not in COORDINATE_SYSTEMS:
                available_presets = ", ".join(f'"{cs}"' for cs in COORDINATE_SYSTEMS)
                raise ValueError(
                    f'Invalid coordinate system "{source_system}". '
                    f"Available presets are: {available_presets}."
                )
            source_system = COORDINATE_SYSTEMS[source_system]

        # Initialize state
        self._state = UnrealCVViewerState(
            offscreen=cfg.offscreen,
            paused=cfg.start_paused,
            coordinate_system=source_system,
            coordinate_transform=CoordinateTransform(source_system, unreal_system),
            meters_to_cm=cfg.meters_to_cm,
            capture_format=cfg.capture_format,
            capture_mode=cfg.capture_mode,
        )
        self._state.camera.camera_id = cfg.camera_id

        # Create and connect the UnrealCV client
        logger.info(f"Connecting to UnrealCV server at {cfg.host}:{cfg.port}...")
        self._client = UnrealCVClient(
            cfg.host,
            cfg.port,
            cfg.connection_timeout,
            camera_mapping=cfg.camera_mapping,
        )

        # Connect without resolving cameras yet (allows spawning blueprints first)
        if not self._client.connect(cfg.retry_attempts, resolve_cameras=False):
            raise RuntimeError(
                f"Failed to connect to UnrealCV server at {cfg.host}:{cfg.port}. "
                "Make sure Unreal Engine is running with the UnrealCV plugin enabled."
            )

        self._state.connected = True
        logger.info("Connected to UnrealCV server.")

        # Camera resolution and size configuration is now deferred to resolve_cameras()

        # Profiling state
        self._last_render_time: float | None = None
        self._frame_count: int = 0

        # Background thread for pipelined get_img: submits get_img commands at the
        # end of a frame so the server can start processing them immediately.
        # At the start of the next frame we just await the future.
        self._get_img_executor: ThreadPoolExecutor | None = None
        self._get_img_future: Future | None = None
        self._get_img_cam_order: list[int] = []

    def __del__(self):
        """Clean up on destruction."""
        self.close()

    def _set_uniform_camera_sizes(self, width: int, height: int) -> None:
        """Set a uniform size for all cameras in the client's mapping.

        This is a legacy helper for when cfg.size is used instead of
        per-camera dimensions in camera_mapping.
        """
        camera_ids = list(set(self._client._obs_key_to_camera_id.values()))
        if not camera_ids:
            return
        results = self._client.set_all_camera_sizes(camera_ids, width, height)
        for camera_id, success in zip(camera_ids, results):
            if success:
                logger.info(f"Set camera {camera_id} size to {width}x{height}")
            else:
                logger.warning(
                    f"Failed to set camera {camera_id} size to {width}x{height}"
                )

    def resolve_cameras(self) -> None:
        """
        Resolve camera name mapping after actors have been spawned.

        This method should be called after blueprint spawning is complete to
        ensure all cameras exist in the UE scene before querying them.

        Sets up:
        1. Camera name-to-index mapping (queries UE via vget /cameras)
        2. Camera capture sizes (either per-camera or uniform)

        Raises:
            ValueError: If any camera name in the mapping is not found in the scene.
        """
        # Resolve pending camera mapping
        self._client.resolve_camera_mapping()

        # Configure capture sizes for all mapped cameras
        camera_mapping = self._cfg.camera_mapping
        if camera_mapping is not None:
            self._client.set_mapped_camera_sizes(camera_mapping)
            self._client.set_mapped_camera_crops(camera_mapping)
        elif self._cfg.size is not None:
            # Legacy path: use uniform size for all cameras
            size = self._cfg.size  # pyre-ignore[16]: size is not None here
            self._set_uniform_camera_sizes(size[0], size[1])

        logger.info("Camera mapping resolved and capture projections configured")

    ####################################################################################
    # Viewer Management Methods
    ####################################################################################

    def render(  # noqa: C901
        self,
        camera_ids: list[int] | None = None,
        camera_names: list[str] | None = None,
    ) -> RenderFrame | dict[int, RenderFrame] | dict[str, RenderFrame] | None:
        """
        Render the current frame.

        The capture timing depends on the configured ``capture_mode``:

        **PIPELINED** (default) — 1-frame-lag pipeline: gets the *previous*
        frame's captures at the start, updates actors, then starts *this*
        frame's captures.  On the first call ``None`` is returned.

        **SYNCHRONIZED** — same-frame capture: updates actors, starts
        captures, then immediately gets them in a second batch.  Every call
        returns the current simulation state at the cost of higher per-frame
        latency.

        Args:
            camera_ids: Optional list of camera IDs to capture images from.
                If provided, capture commands for all cameras are included in
                the batch and a dict mapping camera_id -> image is returned.
                If None, falls back to the legacy single-camera offscreen
                behavior using self._state.camera.camera_id.
            camera_names: Optional list of observation-key names (e.g.
                ``["image.left", "image.right"]``).  Requires
                :meth:`UnrealCVClient.set_camera_name_mapping` to have been
                called on the client.  When provided, the names are resolved
                to camera IDs internally and results are returned as a dict
                mapping observation key name -> image.  Mutually exclusive
                with *camera_ids*.

        Returns:
            If camera_names is provided: dict mapping obs-key name to RGB
                image arrays (only includes cameras that decoded successfully).
            If camera_ids is provided: dict mapping camera_id to RGB image
                arrays (only includes cameras that decoded successfully).
            If neither is provided: single RGB image array if in offscreen
                mode, None otherwise.
        """
        pipelined = self._state.capture_mode == CaptureMode.PIPELINED
        p = self._profiler
        frame_start = time.perf_counter()

        # Resolve camera_names to camera_ids if provided
        name_to_id: dict[str, int] | None = None
        if camera_names is not None:
            name_to_id = {}
            resolved_ids: list[int] = []
            for name in camera_names:
                cam_id = self._client.obs_key_to_camera_id(name)
                if cam_id is not None:
                    name_to_id[name] = cam_id
                    if cam_id not in resolved_ids:
                        resolved_ids.append(cam_id)
                else:
                    logger.warning(
                        f"Camera name '{name}' not found in client mapping, skipping"
                    )
            camera_ids = resolved_ids if resolved_ids else None

        # --- 1a. Await previous frame's decoded captures (pipelined only) ---
        get_img_cam_order: list[int] = []
        num_get_img_cmds = 0
        if p is not None:
            p.enter("network")
        captured_frames: dict[int, RenderFrame] = {}
        if pipelined and self._get_img_future is not None:
            captured_frames = self._get_img_future.result()
            get_img_cam_order = self._get_img_cam_order
            num_get_img_cmds = len(get_img_cam_order)
            self._get_img_future = None
            self._get_img_cam_order = []
        if p is not None:
            p.exit()

        commands: list[str] = []

        # --- 1b. Actor transform commands for this frame ---
        if p is not None:
            p.enter("build_actor_cmds")
        self._state.requires_update = True
        if self._state.requires_update and self._state.scene.instance is not None:
            self._build_actor_commands(commands)
            self._update_scene_bounds()
            self._state.requires_update = False
        if p is not None:
            p.exit()

        # --- 1c. Async capture start commands for this frame ---
        target_cam_ids: list[int] = []
        if camera_ids is not None:
            target_cam_ids = list(camera_ids)
        elif self._state.offscreen:
            target_cam_ids = [self._state.camera.camera_id]

        for cam_id in target_cam_ids:
            commands.append(UnrealCVClient.build_start_lit_async_command(cam_id))

        # --- 1d. Synchronized: append get commands for this frame's captures ---
        get_img_start_index = 0
        if not pipelined and target_cam_ids:
            get_img_start_index = len(commands)
            for cam_id in target_cam_ids:
                commands.append(
                    UnrealCVClient.build_get_lit_latest_command(
                        cam_id, self._state.capture_format.value
                    )
                )
            get_img_cam_order = target_cam_ids
            num_get_img_cmds = len(target_cam_ids)

        # --- 2. Send batch(es) ---
        if p is not None:
            p.enter("network")
        get_img_responses: list = []
        if pipelined:
            # Fire-and-forget actor transforms + capture starts.
            if commands:
                self._client._request_batch(commands, use_async=True)
            # Submit get for this frame's captures in a background thread so
            # the server starts processing them immediately.  We'll await the
            # result at the top of the *next* render() call.
            if target_cam_ids:
                get_img_cmds = [
                    UnrealCVClient.build_get_lit_latest_command(
                        cam_id, self._state.capture_format.value
                    )
                    for cam_id in target_cam_ids
                ]
                if self._get_img_executor is None:
                    self._get_img_executor = ThreadPoolExecutor(max_workers=1)
                self._get_img_cam_order = list(target_cam_ids)
                self._get_img_future = self._get_img_executor.submit(  # pyre-ignore[16]
                    self._get_img_and_decode, get_img_cmds, list(target_cam_ids)
                )
        else:
            # Synchronized: single blocking batch for everything.
            responses: list[str | None] = []
            if commands:
                batch_result = self._client._request_batch(commands)
                if isinstance(batch_result, tuple):
                    responses = batch_result[0]
                else:
                    responses = batch_result
            if num_get_img_cmds > 0:
                get_img_responses = responses[
                    get_img_start_index : get_img_start_index + num_get_img_cmds
                ]
        if p is not None:
            p.exit()

        # --- 3. Decode / collect results ---
        if p is not None:
            p.enter("decode")
        result: RenderFrame | dict[int, RenderFrame] | None = None

        if pipelined:
            # Pipelined: captured_frames were already decoded in the
            # background thread — just pick the right return shape.
            if captured_frames:
                if camera_ids is not None:
                    result = captured_frames
                else:
                    first_cam = get_img_cam_order[0] if get_img_cam_order else None
                    result = captured_frames.get(first_cam) if first_cam else None
        elif num_get_img_cmds > 0:
            # Synchronized: decode inline.
            captured_frames_sync: dict[int, RenderFrame] = {}
            for i, cam_id in enumerate(get_img_cam_order):
                resp = get_img_responses[i] if i < len(get_img_responses) else None
                if isinstance(resp, str) and resp == "pending":
                    pass
                elif resp is not None:
                    frame = self._decode_capture_response(resp)
                    if frame is not None:
                        captured_frames_sync[cam_id] = frame

            if camera_ids is not None:
                result = captured_frames_sync if captured_frames_sync else None
            elif captured_frames_sync:
                first_cam = get_img_cam_order[0]
                result = captured_frames_sync.get(first_cam)

        if p is not None:
            p.exit()

        self._frame_count += 1
        self._last_render_time = frame_start

        # If camera_names were used, remap int-keyed results to name-keyed results
        if name_to_id is not None and isinstance(result, dict):
            id_to_name = {cam_id: name for name, cam_id in name_to_id.items()}
            result = {
                id_to_name[cam_id]: frame
                for cam_id, frame in result.items()
                if cam_id in id_to_name
            }

        return result

    def _decode_capture_response(
        self, response: str | bytes | None
    ) -> RenderFrame | None:
        """Decode a capture response from the batch into a numpy RGB array.

        Supports both npy (numpy binary) and png formats based on the
        viewer's configured capture_format.
        """
        if response is None:
            return None
        try:
            if self._state.capture_format == CaptureFormat.PNG:
                from PIL import Image

                img = Image.open(BytesIO(response)).convert("RGB")
                return np.array(img)
            else:
                arr = np.load(BytesIO(response))
                if arr.ndim == 3 and arr.shape[2] in (3, 4):
                    return bgr_to_rgb(arr)
                return arr
        except Exception as e:
            logger.error(f"Failed to decode captured image: {e}")
            return None

    def _get_img_and_decode(
        self, get_img_cmds: list[str], cam_order: list[int]
    ) -> dict[int, RenderFrame]:
        """Send get commands and decode the responses into RGB frames.

        Designed to run in a background thread so that network I/O and image
        decoding overlap with the next simulation step.
        """
        responses = self._client._request_batch(get_img_cmds)
        if isinstance(responses, tuple):
            responses = responses[0]

        frames: dict[int, RenderFrame] = {}
        for i, cam_id in enumerate(cam_order):
            resp = responses[i] if i < len(responses) else None
            if isinstance(resp, str) and resp == "pending":
                continue
            if resp is not None:
                frame = self._decode_capture_response(resp)
                if frame is not None:
                    frames[cam_id] = frame
        return frames

    def close(self):
        """
        Close the viewer and disconnect from Unreal Engine.

        Calling close on an already closed viewer has no effect.
        """
        if not self._state.connected:
            return

        # Drain any in-flight get_img future before tearing down the connection.
        if self._get_img_future is not None:
            try:
                self._get_img_future.result(timeout=5.0)
            except Exception:
                pass
            self._get_img_future = None
        if self._get_img_executor is not None:
            self._get_img_executor.shutdown(wait=False)
            self._get_img_executor = None

        # Hide all actors
        self._remove_actors()

        # Disconnect from UE
        self._client.disconnect()
        self._state.connected = False
        logger.info("UnrealCV viewer closed.")

    def take_screenshot(self) -> RenderFrame | None:
        """
        Capture a screenshot from the UE camera.

        Returns:
            RGB image as numpy array, or None if capture failed.
        """
        return self._client.capture_image(
            self._state.camera.camera_id,
            "lit",
            self._state.capture_format.value,
        )

    def user_requested_close(self) -> bool:
        """
        Check if the user requested to close the viewer.

        Note: UnrealCV doesn't have a built-in way to detect this, so this
        always returns False. The viewer should be closed programmatically.
        """
        return self._state.close_requested

    ####################################################################################
    # Scene Management
    ####################################################################################

    def get_scene(self) -> Scene | None:
        """Returns the mochi scene associated with the viewer."""
        return self._state.scene.instance

    def set_scene(self, scene: Scene | None):
        """
        Set the mochi scene to render.

        Args:
            scene: The mochi scene to render, or None to clear.
        """
        # Only update if the scene has changed
        handle = None if scene is None else scene.get_handle()
        if handle == self._state.scene.handle:
            return

        # Update scene reference
        self._state.scene.instance = scene
        self._state.scene.handle = handle

        # If no scene, clear actors
        if scene is None:
            self._remove_actors()
            return

        # Update actors and send initial transforms
        commands = self._build_actor_commands()
        self._update_scene_bounds()
        if commands:
            self._client._request_batch(commands, use_async=True)

    def is_paused(self) -> bool:
        """Returns the paused state of the viewer."""
        return self._state.paused

    def set_paused(self, paused: bool):
        """Sets the paused state of the viewer."""
        self._state.paused = paused

    def get_scene_bounds(self) -> AABB:
        """Returns the scene bounding box."""
        return self._state.scene.aabb

    def trigger_scene_reset(self) -> bool:
        """
        Trigger a scene reset event in Unreal.

        This broadcasts the JitterMaterial event to all actors that implement
        the JitterMaterial function (UFUNCTION). Use this when resetting the
        environment to randomize materials or other visual properties.

        Returns:
            True if successful, False otherwise.
        """
        return self._client.trigger_scene_reset()

    def _update_scene_bounds(self):
        """Update the scene bounding box from actor AABBs."""
        # Gather actor AABBs
        actor_aabbs = [
            actor.updater.get_aabb()
            for actor in self._state.scene.actors.values()
            if actor.updater is not None
        ]

        # Compute scene AABB
        if len(actor_aabbs) > 0:
            self._state.scene.aabb.compute_from_aabbs(actor_aabbs)

    ####################################################################################
    # Actor Management
    ####################################################################################

    def get_actors(self) -> list[Actor]:
        """Returns the list of actors being rendered."""
        return [actor.instance for actor in self._state.scene.actors.values()]

    def get_actor_updaters(self) -> list[UnrealCVUpdater]:
        """Returns the list of actor updaters."""
        return [
            actor.updater
            for actor in self._state.scene.actors.values()
            if actor.updater is not None
        ]

    def get_actor_updater(self, actor: Actor) -> UnrealCVUpdater | None:
        """
        Get the updater for a specific actor.

        Args:
            actor: The mochi actor.

        Returns:
            The updater, or None if not found.
        """
        actor_handle = actor.get_handle()
        actor_state = self._state.scene.actors.get(actor_handle, None)
        return None if actor_state is None else actor_state.updater

    def set_excluded_actors(self, excluded_actors: list[str]):
        """
        Set actors to exclude from rendering.

        Args:
            excluded_actors: List of actor names or wildcard patterns.
        """
        self._state.scene.excluded_actors = set(excluded_actors)
        commands = self._build_actor_commands()
        if commands:
            self._client._request_batch(commands, use_async=True)

    def _is_valid_actor(self, name: str) -> bool:
        """Check if an actor should be rendered (not excluded)."""
        return not any(
            fnmatchcase(name, pattern) for pattern in self._state.scene.excluded_actors
        )

    def _gather_valid_actors_from_scene(self) -> dict[ActorHandle, Actor]:
        """Get all valid actors from the mochi scene."""
        actors = {}

        def gather_actor(actor: Actor):
            name = actor.get_name()
            if self._is_valid_actor(name):
                handle = actor.get_handle()
                actors[handle] = actor

        if self._state.scene.instance is not None:
            self._state.scene.instance.for_each_actor(gather_actor)

        return actors

    def _map_actor_name(self, mochi_name: str) -> str:
        """
        Map a mochi actor name to a UE actor name.

        Args:
            mochi_name: The name of the actor in mochi.

        Returns:
            The corresponding name in Unreal Engine.
        """
        if self._cfg.actor_name_mapping is not None:
            return self._cfg.actor_name_mapping.get(mochi_name, mochi_name)
        return mochi_name

    def _get_articulated_actor_mapping(
        self, mochi_actor_name: str
    ) -> tuple[str, dict[str, str] | None] | None:
        """
        Get the mapping for an articulated actor to a UE skeletal mesh.

        Args:
            mochi_actor_name: The name of the mochi articulated actor.

        Returns:
            Tuple of (ue_actor_name, link_to_bone_mapping) if mapping exists,
            None otherwise. link_to_bone_mapping may be None if not specified.
        """
        if self._cfg.articulated_actor_mapping is not None:
            mapping = self._cfg.articulated_actor_mapping.get(mochi_actor_name)
            if mapping is not None:
                ue_actor = mapping.get("ue_actor")
                if ue_actor is not None:
                    link_to_bone = mapping.get("link_to_bone")
                    return (ue_actor, link_to_bone)
        return None

    def _get_articulated_actor_hack_fixup_root_xform(
        self, mochi_actor_name: str
    ) -> bool:
        """
        Check if root transform fixup is enabled for an articulated actor.

        Args:
            mochi_actor_name: The name of the mochi articulated actor.

        Returns:
            True if hack_fixup_root_xform is enabled, False otherwise.
        """
        if self._cfg.articulated_actor_mapping is not None:
            mapping = self._cfg.articulated_actor_mapping.get(mochi_actor_name)
            if mapping is not None:
                return mapping.get("hack_fixup_root_xform", False)
        return False

    def _build_actor_commands(  # noqa: C901
        self, commands: list[str] | None = None
    ) -> list[str]:
        """
        Build all actor update commands without sending anything.

        When a ``Profiler`` was supplied to the viewer, the sub-phases
        (gather, remove, collect, add_new, cmd_build) are recorded as nested
        profiler sections instead of being tracked in a flat timing dict.

        Args:
            commands: Optional list to append commands into.  When ``None``
                a fresh list is created and returned.

        Returns:
            The list of generated command strings.
        """
        if commands is None:
            commands = []
        p = self._profiler

        # Get valid actors from mochi
        mochi_actors = self._gather_valid_actors_from_scene()

        # Determine actors to add/remove
        actor_handles = set(mochi_actors)
        existing_handles = set(self._state.scene.actors.keys())
        handles_to_add = actor_handles - existing_handles
        handles_to_remove = existing_handles - actor_handles

        # Remove old actors
        for handle in handles_to_remove:
            actor_state = self._state.scene.actors[handle]
            if actor_state.updater is not None:
                actor_state.updater.remove()
            del self._state.scene.actors[handle]

        # Collect transform data from all existing actors
        if p is not None:
            p.enter("collect")
        object_transforms: list[
            tuple[str, npt.NDArray[np.floating] | None, npt.NDArray[np.floating] | None]
        ] = []
        # Type for bone transforms: actor_name -> { "root_transform": tuple, "bones": list }
        bone_transforms_by_actor: dict[str, Any] = {}

        for actor_state in self._state.scene.actors.values():
            if actor_state.updater is not None:
                if isinstance(actor_state.updater, UnrealCVArticulatedActorUpdater):
                    actor_root_info, articulated_bone_transforms = (
                        actor_state.updater.get_all_bone_transform_data()
                    )
                    actor_name, root_pos, root_quat = actor_root_info
                    if actor_name not in bone_transforms_by_actor:
                        bone_transforms_by_actor[actor_name] = {
                            "root_transform": (root_pos, root_quat),
                            "bones": [],
                        }
                    for (
                        _,
                        bone_name,
                        position,
                        rotation,
                    ) in articulated_bone_transforms:
                        bone_transforms_by_actor[actor_name]["bones"].append(
                            (bone_name, position, rotation)
                        )
                elif isinstance(actor_state.updater, UnrealCVActorUpdater):
                    transform_data = actor_state.updater.get_transform_data()
                    if transform_data is not None:
                        ue_actor_name, position_ue, rotation_ue = transform_data
                        object_transforms.append(
                            (ue_actor_name, position_ue, rotation_ue)
                        )
        if p is not None:
            p.exit()

        # Add new actors
        new_object_transforms: list[
            tuple[str, npt.NDArray[np.floating] | None, npt.NDArray[np.floating] | None]
        ] = []
        # Type for bone transforms: actor_name -> { "root_transform": tuple, "bones": list }
        new_bone_transforms_by_actor: dict[str, Any] = {}

        actors_handled_by_articulated_updater: set[ActorHandle] = set()

        # First pass: Process ARTICULATED actors
        for handle in handles_to_add:
            instance = mochi_actors[handle]
            mochi_actor_name = instance.get_name()

            if instance.get_type() != ActorType.ARTICULATED:
                continue

            articulated_mapping = self._get_articulated_actor_mapping(mochi_actor_name)
            if articulated_mapping is None:
                continue

            ue_actor, link_to_bone = articulated_mapping
            logger.info(
                f"Articulated actor '{mochi_actor_name}' maps to skeletal mesh "
                f"'{ue_actor}' using joint-based bone rotations."
            )
            self._client.set_object_visibility(ue_actor, True)
            # Query the actual UE skeleton bone names so the updater can
            # handle hierarchy mismatches between mochi and UE.
            ue_bone_names = self._client.get_bone_names(ue_actor)
            if ue_bone_names is not None:
                logger.debug(
                    f"UE skeleton for '{ue_actor}' has {len(ue_bone_names)} bones"
                )
            # Check if hack_fixup_root_xform is enabled in the config
            hack_fixup_root_xform = self._get_articulated_actor_hack_fixup_root_xform(
                mochi_actor_name
            )
            # scene.instance is checked to be non-None before this code path is reached
            assert self._state.scene.instance is not None
            updater = UnrealCVArticulatedActorUpdater(
                actor=instance,
                ue_skeletal_mesh_actor_name=ue_actor,
                client=self._client,
                coordinate_transform=self._state.coordinate_transform,
                link_to_bone_mapping=link_to_bone,
                meters_to_cm=self._state.meters_to_cm,
                ue_bone_names=ue_bone_names,
                hack_fixup_root_xform=hack_fixup_root_xform,
            )
            for link_handle in instance.get_nested_link_actors():
                actors_handled_by_articulated_updater.add(link_handle)
            actor_root_info, articulated_bone_transforms = (
                updater.get_all_bone_transform_data()
            )
            actor_name, root_pos, root_quat = actor_root_info
            if actor_name not in new_bone_transforms_by_actor:
                new_bone_transforms_by_actor[actor_name] = {
                    "root_transform": (root_pos, root_quat),
                    "bones": [],
                }
            for (
                _,
                bone_name,
                position,
                rotation,
            ) in articulated_bone_transforms:
                new_bone_transforms_by_actor[actor_name]["bones"].append(
                    (bone_name, position, rotation)
                )

            self._state.scene.actors[handle] = UnrealCVActorState(
                handle=handle,
                instance=instance,
                ue_actor_name=ue_actor,
                updater=updater,
            )

        # Second pass: Process all other actors
        for handle in handles_to_add:
            instance = mochi_actors[handle]
            mochi_actor_name = instance.get_name()

            if handle in self._state.scene.actors:
                continue

            if handle in actors_handled_by_articulated_updater:
                logger.debug(
                    f"Skipping actor '{mochi_actor_name}' - handled by articulated updater"
                )
                self._state.scene.actors[handle] = UnrealCVActorState(
                    handle=handle,
                    instance=instance,
                    ue_actor_name=self._map_actor_name(mochi_actor_name),
                    updater=None,
                )
                continue

            updater = None

            if not instance.get_surface_mesh().is_empty():
                ue_name = self._map_actor_name(mochi_actor_name)
                if self._client.object_exists(ue_name):
                    logger.info(
                        f"Found actor '{ue_name}' in UE scene, mapped to '{mochi_actor_name}'"
                    )
                    self._client.set_object_visibility(ue_name, True)
                    updater = UnrealCVActorUpdater(
                        actor=instance,
                        ue_actor_name=ue_name,
                        client=self._client,
                        coordinate_transform=self._state.coordinate_transform,
                        meters_to_cm=self._state.meters_to_cm,
                    )
                    transform_data = updater.get_transform_data()
                    if transform_data is not None:
                        ue_actor_name, pos_ue, rot_ue = transform_data
                        new_object_transforms.append((ue_actor_name, pos_ue, rot_ue))
                else:
                    logger.warning(
                        f"Actor '{ue_name}' not found in UE scene. "
                        f"Skipping rendering for mochi actor '{mochi_actor_name}'."
                    )

            self._state.scene.actors[handle] = UnrealCVActorState(
                handle=handle,
                instance=instance,
                ue_actor_name=self._map_actor_name(instance.get_name()),
                updater=updater,
            )

        # Sort actors by name
        if handles_to_add:
            sorted_pairs = sorted(
                self._state.scene.actors.items(),
                key=lambda kv: kv[1].instance.get_name(),
            )
            self._state.scene.actors = dict(sorted_pairs)

        # Build all commands from collected transform data

        # Object transform commands (existing + new)
        all_object_transforms = object_transforms + new_object_transforms
        if all_object_transforms:
            commands.extend(
                UnrealCVClient.build_object_transform_commands(
                    all_object_transforms  # pyre-ignore[6]
                )
            )

        # Bone transform commands (existing + new)
        all_bone_actors = dict(bone_transforms_by_actor)
        for actor_name, actor_data in new_bone_transforms_by_actor.items():
            if actor_name in all_bone_actors:
                all_bone_actors[actor_name]["bones"].extend(actor_data["bones"])
            else:
                all_bone_actors[actor_name] = actor_data

        for actor_name, actor_data in all_bone_actors.items():
            root_pos, root_quat = actor_data["root_transform"]
            bone_list = actor_data["bones"]
            cmd = UnrealCVClient.build_poseable_xforms_command(
                actor_name, bone_list, root_pos, root_quat
            )
            if cmd is not None:
                commands.append(cmd)

        return commands

    def _build_camera_commands(self) -> list[str]:
        """
        Build camera update commands without sending.

        Runs the same camera logic as _update_camera() but returns
        command strings instead of sending them.

        Used for driving the main view camera (framing, etc) but not scene cameras.

        Returns:
            List of command strings for camera location/rotation.
        """
        # Check for deferred framing
        if self._state.camera.frame_camera_on_next_update:
            if not self._state.scene.aabb.is_empty:
                self.frame_scene(self._state.camera.frame_camera_direction)
            self._state.camera.frame_camera_on_next_update = False

        # Update follow camera if tracking is enabled
        if self._state.camera.use_follow_camera:
            self._update_follow_camera()

        # Compute camera UE position and rotation (without sending)
        camera_pos_ue, camera_rot_ue = self._get_camera_ue_state()

        commands = [
            UnrealCVClient.build_camera_location_command(
                self._state.camera.camera_id, camera_pos_ue
            ),
            UnrealCVClient.build_camera_rotation_command(
                self._state.camera.camera_id, camera_rot_ue
            ),
        ]
        return commands

    def _get_camera_ue_state(
        self,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """
        Get the current camera position and rotation in UE coordinates.

        Computes the UE position/rotation from the smoothed camera state
        without sending any network commands.

        Returns:
            Tuple of (position_ue, rotation_ue).
        """
        # Camera position
        look_from = self._state.camera.smoothed_position
        if look_from is None:
            look_from = np.zeros(3)

        # Camera look-at target
        look_at = self._state.camera.smoothed_target_position
        if look_at is None:
            look_at = np.array([0, 0, 0])

        position_ue, rotation_ue = self._compute_camera_view(look_from, look_at)
        return position_ue, rotation_ue

    def _remove_actors(self):
        """Hide all actors in UE."""
        for actor_state in self._state.scene.actors.values():
            if actor_state.updater is not None:
                actor_state.updater.remove()
        self._state.scene.actors.clear()
        self._state.scene.selected_actor = None
        self._state.scene.aabb.set_empty()

    ####################################################################################
    # Camera Management
    ####################################################################################

    def set_camera_view(
        self,
        look_from: npt.NDArray[np.floating],
        look_at: npt.NDArray[np.floating],
        up_dir: npt.NDArray[np.floating] | None = None,
        fly_to: bool = False,
    ):
        """
        Set the camera to look from a position at a target.

        This computes the UE position/rotation and sends them immediately.
        For batched rendering, use _compute_camera_view() + _build_camera_commands().

        Args:
            look_from: Camera position in mochi coordinates.
            look_at: Target position to look at in mochi coordinates.
            up_dir: Up direction vector. If None, uses coordinate system up.
            fly_to: Ignored (no animation support in UnrealCV).
        """
        position_ue, rotation_ue = self._compute_camera_view(look_from, look_at, up_dir)
        self._client.set_camera_location(self._state.camera.camera_id, position_ue)
        self._client.set_camera_rotation(self._state.camera.camera_id, rotation_ue)

    def _compute_camera_view(
        self,
        look_from: npt.NDArray[np.floating],
        look_at: npt.NDArray[np.floating],
        up_dir: npt.NDArray[np.floating] | None = None,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """
        Compute camera UE position and rotation without sending.

        Updates the cached rotation state so get_camera_look_dir() can
        work without a network round-trip.

        Returns:
            Tuple of (position_ue, rotation_ue).
        """
        look_from = np.asarray(look_from, dtype=np.float64)
        look_at = np.asarray(look_at, dtype=np.float64)

        look_dir = look_at - look_from
        look_dir = look_dir / np.linalg.norm(look_dir)

        if up_dir is None:
            up_dir = self._state.coordinate_system.up.to_vector()
        else:
            up_dir = np.asarray(up_dir, dtype=np.float64)

        position_ue = self._convert_position_to_ue(look_from)
        rotation_ue = self._compute_camera_rotation(look_dir, up_dir)

        # Cache locally so get_camera_look_dir() doesn't need a GET
        self._state.camera.cached_rotation_ue = rotation_ue

        return position_ue, rotation_ue

    def get_camera_position(self) -> npt.NDArray[np.floating]:
        """
        Get the current camera position.

        Returns:
            Camera position in mochi coordinates.
        """
        pos_ue = self._client.get_camera_location(self._state.camera.camera_id)
        if pos_ue is None:
            return np.zeros(3)

        # Convert back to mochi coordinates
        transform_inv = self._state.coordinate_transform.target_to_source[:3, :3]
        position = transform_inv @ pos_ue / self._state.meters_to_cm
        return position

    def get_camera_look_dir(self) -> npt.NDArray[np.floating]:
        """
        Get the current camera look direction.

        Returns:
            Look direction vector in mochi coordinates.
        """
        rot_ue = getattr(self._state.camera, "cached_rotation_ue", None)
        if rot_ue is None:
            rot_ue = self._client.get_camera_rotation(self._state.camera.camera_id)
        if rot_ue is None:
            return np.array([0, 0, -1])

        # Convert UE rotation to look direction
        pitch, yaw, roll = rot_ue
        rot = Rotation.from_euler("ZYX", [yaw, pitch, roll], degrees=True)
        look_dir_ue = rot.apply(np.array([1, 0, 0]))  # UE forward is +X

        # Convert to mochi coordinates
        transform_inv = self._state.coordinate_transform.target_to_source[:3, :3]
        look_dir = transform_inv @ look_dir_ue
        return look_dir / np.linalg.norm(look_dir)

    def is_follow_camera_enabled(self) -> bool:
        """Returns whether follow camera is enabled."""
        return self._state.camera.use_follow_camera

    def set_enable_follow_camera(self, enabled: bool):
        """Enable or disable follow camera mode."""
        self._state.camera.use_follow_camera = enabled

    def set_follow_camera_smoothness(self, smoothing: float):
        """Set the smoothness of follow camera movement."""
        self._state.camera.smoothing = np.clip(smoothing, 0.0, 1.0)

    def get_follow_camera_smoothness(self) -> float:
        """Get the follow camera smoothness."""
        return self._state.camera.smoothing

    def set_compute_automatic_distance(self, enabled: bool):
        """Enable automatic camera distance computation."""
        self._state.camera.automatic_distance = enabled

    def is_automatic_distance_enabled(self) -> bool:
        """Check if automatic distance is enabled."""
        return self._state.camera.automatic_distance

    def get_coordinate_system(self) -> CoordinateSystem:
        """Get the source coordinate system."""
        return self._state.coordinate_system

    def snap_follow_camera(self):
        """Immediately snap camera to follow target without smoothing."""
        if not self._state.camera.use_follow_camera:
            return

        if self._state.scene.aabb.is_empty:
            return

        look_dir = self.get_camera_look_dir()
        distance = self._state.camera.distance
        self.frame_scene_from_direction(look_dir, distance)

    def frame_scene(
        self,
        look_dir: npt.NDArray[np.floating] | None = None,
        fly_to: bool = False,
    ):
        """
        Frame the scene so all actors are visible.

        Args:
            look_dir: Direction to look from. If None, uses current direction.
            fly_to: Ignored (no animation support).
        """
        if self._state.scene.aabb.is_empty:
            self._state.camera.frame_camera_on_next_update = True
            self._state.camera.frame_camera_direction = look_dir
            return

        if look_dir is None:
            look_dir = self.get_camera_look_dir()

        # Compute distance to frame scene
        radius = np.linalg.norm(self._state.scene.aabb.extents) / 2
        distance = 2.0 * radius * self._state.meters_to_cm  # Convert to cm

        self.frame_scene_from_direction(look_dir, distance)

    def frame_scene_from_direction(
        self,
        look_dir: npt.NDArray[np.floating],
        distance: float,
        fly_to: bool = False,
    ):
        """
        Frame the scene from a specific direction and distance.

        Updates the camera state but does NOT send network commands.
        The commands will be included in the next batch via _build_camera_commands.

        Args:
            look_dir: Direction to look from.
            distance: Distance from the scene center.
            fly_to: Ignored (no animation support).
        """
        look_at = self._state.scene.aabb.center
        look_from = look_at - (distance / self._state.meters_to_cm) * look_dir

        # Update smoothed camera state (commands built later by _build_camera_commands)
        self._state.camera.smoothed_position = look_from
        self._state.camera.smoothed_target_position = look_at
        self._state.camera.distance = distance
        self._state.camera.smoothed_distance = distance

        # Pre-compute UE rotation so get_camera_look_dir() works from cache
        self._compute_camera_view(look_from, look_at)

    def _update_camera(self):
        """Update camera each frame."""
        # Check for deferred framing
        if self._state.camera.frame_camera_on_next_update:
            if not self._state.scene.aabb.is_empty:
                self.frame_scene(self._state.camera.frame_camera_direction)
                self._state.camera.frame_camera_on_next_update = False
                self._state.camera.frame_camera_direction = None

        # Update follow camera
        if self._state.camera.use_follow_camera:
            self._update_follow_camera()

    def _update_follow_camera(self):
        if self._state.scene.aabb.is_empty:
            return

        def smooth(current, target):
            alpha = self._state.camera.smoothing
            return alpha * current + (1 - alpha) * target

        # Smooth target position
        if self._state.camera.smoothed_target_position is None:
            self._state.camera.smoothed_target_position = self._state.scene.aabb.center
        else:
            self._state.camera.smoothed_target_position = smooth(
                self._state.camera.smoothed_target_position,
                self._state.scene.aabb.center,
            )

        # Compute distance
        if self._state.camera.automatic_distance:
            radius = np.linalg.norm(self._state.scene.aabb.extents) / 2
            distance = 2.0 * radius * self._state.meters_to_cm
            self._state.camera.distance = distance
            self._state.camera.smoothed_distance = smooth(
                self._state.camera.smoothed_distance, distance
            )
        else:
            self._state.camera.smoothed_distance = self._state.camera.distance

        # Update camera position state (no network I/O)
        look_dir = self.get_camera_look_dir()
        look_at = self._state.camera.smoothed_target_position
        look_from = (
            look_at
            - (self._state.camera.smoothed_distance / self._state.meters_to_cm)
            * look_dir
        )
        self._state.camera.smoothed_position = look_from

    ####################################################################################
    # Coordinate Conversion Helpers
    ####################################################################################

    def _convert_position_to_ue(
        self, position: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Convert a position from mochi to UE coordinates."""
        transform = self._state.coordinate_transform.source_to_target[:3, :3]
        position_ue = transform @ position * self._state.meters_to_cm
        return position_ue

    def _compute_camera_rotation(
        self,
        look_dir: npt.NDArray[np.floating],
        up_dir: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """
        Compute camera rotation in UE coordinates.

        Args:
            look_dir: Look direction in mochi coordinates.
            up_dir: Up direction in mochi coordinates.

        Returns:
            Rotation as [pitch, yaw, roll] in degrees for UE.
        """
        # Transform directions to UE coordinates
        transform = self._state.coordinate_transform.source_to_target[:3, :3]
        look_dir_ue = transform @ look_dir
        up_dir_ue = transform @ up_dir

        # Normalize
        look_dir_ue = look_dir_ue / np.linalg.norm(look_dir_ue)
        up_dir_ue = up_dir_ue / np.linalg.norm(up_dir_ue)

        # Build rotation matrix
        # UE camera: +X is forward, +Z is up
        right_ue = np.cross(look_dir_ue, up_dir_ue)
        right_ue = right_ue / np.linalg.norm(right_ue)
        up_ue = np.cross(right_ue, look_dir_ue)

        rot_matrix = np.column_stack([look_dir_ue, right_ue, up_ue])

        # Convert to Euler angles
        try:
            rot = Rotation.from_matrix(rot_matrix)
            euler = rot.as_euler("ZYX", degrees=True)
            return np.array([euler[1], euler[0], euler[2]])  # pitch, yaw, roll
        except Exception:
            return np.zeros(3)

    ####################################################################################
    # Context Manager Support
    ####################################################################################

    def __enter__(self) -> "UnrealCVViewer":
        """Support with-statement for the viewer."""
        return self

    def __exit__(self, *ignored) -> bool:
        """Close the viewer at the end of the with-statement."""
        self.close()
        return False  # Propagate exceptions
