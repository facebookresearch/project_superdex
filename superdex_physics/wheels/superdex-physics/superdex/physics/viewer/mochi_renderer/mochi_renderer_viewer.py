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

"""MochiRenderer-based viewer for SuperDex Lab gym environments.

This module provides a client-side viewer for the standalone ``mochi_viewer`` TCP
server. It implements only a subset of the Polyscope ``Viewer`` interface.
"""

from __future__ import annotations

import logging
import math
import struct

import numpy as np
import numpy.typing as npt
import superdex.physics as mochi
from superdex.physics import Actor, ActorHandle, ActorType, Scene

from .mochi_renderer_client import CommandEntry, MochiRendererClient
from .mochi_renderer_viewer_cfg import MochiRendererViewerCfg

logger = logging.getLogger(__name__)

########################################################################################

RenderFrame = npt.NDArray[np.uint8]
"""A render frame, consists of an RGBA image."""


class MochiRendererViewer:
    """Client-side viewer for the standalone ``mochi_viewer`` TCP server.

    This class implements a subset of the Polyscope ``Viewer`` interface. It connects
    to a running TCP server and synchronizes Mochi simulation state (actor meshes,
    transforms, camera) with the renderer.
    """

    ####################################################################################
    # Members
    ####################################################################################

    _cfg: MochiRendererViewerCfg
    _client: MochiRendererClient
    _scene: Scene | None
    _paused: bool
    _actor_handles: dict[ActorHandle, _ActorInfo]

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(self, cfg: MochiRendererViewerCfg) -> None:
        """
        Initialize the MochiRenderer viewer and connect to the server.

        Args:
            cfg: Configuration for the viewer.

        Raises:
            RuntimeError: If connection to the mochi_renderer server fails.
        """
        self._cfg = cfg
        self._scene = None
        self._paused = False
        self._actor_handles = {}

        # Create and connect the TCP client
        logger.info(f"Connecting to mochi_renderer at {cfg.host}:{cfg.port}...")
        self._client = MochiRendererClient(cfg.host, cfg.port)
        if not self._client.connect(cfg.connection_timeout, cfg.retry_attempts):
            raise RuntimeError(
                f"Failed to connect to mochi_renderer at {cfg.host}:{cfg.port}. "
                "Make sure the mochi_renderer server is running."
            )
        logger.info("Connected to mochi_renderer.")

        # Create default camera
        w, h = cfg.size
        commands = [
            CommandEntry(text=f"vset /camera/{cfg.camera_name}/create {w} {h}"),
        ]

        # Create additional named cameras from config
        cameras = cfg.cameras
        if cameras is not None:
            for cam_name, cam_cfg in cameras.items():
                commands.append(
                    CommandEntry(
                        text=f"vset /camera/{cam_name}/create {cam_cfg.width} {cam_cfg.height}"
                    )
                )

        self._client.request_batch(commands)

        # Set initial pose for configured cameras (prefer transform over lookat)
        if cameras is not None:
            camera_commands: list[CommandEntry] = []
            for cam_name, cam_cfg in cameras.items():
                if cam_cfg.position is not None and cam_cfg.rotation is not None:
                    p = cam_cfg.position
                    r = cam_cfg.rotation
                    camera_commands.append(
                        CommandEntry(
                            text=(
                                f"vset /camera/{cam_name}/transform "
                                f"{p[0]} {p[1]} {p[2]} "
                                f"{r[0]} {r[1]} {r[2]} {r[3]}"
                            )
                        )
                    )
                elif cam_cfg.look_from is not None and cam_cfg.look_at is not None:
                    ef = cam_cfg.look_from
                    et = cam_cfg.look_at
                    camera_commands.append(
                        CommandEntry(
                            text=(
                                f"vset /camera/{cam_name}/lookat "
                                f"{ef[0]} {ef[1]} {ef[2]} {et[0]} {et[1]} {et[2]}"
                            )
                        )
                    )
                horizontal_fov_deg = cam_cfg.horizontal_fov_deg
                if horizontal_fov_deg is not None:
                    aspect = cam_cfg.width / cam_cfg.height
                    hfov_rad = math.radians(horizontal_fov_deg)
                    vfov_deg = math.degrees(
                        2.0 * math.atan(math.tan(hfov_rad / 2.0) / aspect)
                    )
                    camera_commands.append(
                        CommandEntry(text=f"vset /camera/{cam_name}/fov {vfov_deg}")
                    )
            if camera_commands:
                self._client.request_batch(camera_commands)

        # Load IBL environment map if configured
        if cfg.environment_ibl:
            logger.info(f"Loading IBL: {cfg.environment_ibl}")
            ibl_commands = [
                CommandEntry(text=f"vset /scene/ibl {cfg.environment_ibl}"),
                CommandEntry(
                    text=f"vset /scene/skybox_visible {'1' if cfg.skybox_visible else '0'}"
                ),
            ]
            self._client.request_batch(ibl_commands)
        else:
            logger.debug("No IBL configured (environment_ibl is None)")

        # NOTE: environment glTF is loaded in set_scene(), not here.
        # Loading it here would cause a redundant load since set_scene()
        # destroys and re-creates it anyway.

    ####################################################################################
    # Viewer management methods
    ####################################################################################

    def render(
        self,
        camera_names: list[str] | None = None,
    ) -> RenderFrame | dict[str, RenderFrame] | None:
        """
        Render the current frame.

        Updates all actor transforms / mesh data and captures images.

        Args:
            camera_names: If provided, capture from all listed cameras and
                return a dict mapping camera name to RGBA image. If None,
                capture from the default camera only.

        Returns:
            When camera_names is None: RGBA image as a numpy uint8 array with
                shape (H, W, 4), or None if no scene is set or capture failed.
            When camera_names is provided: dict mapping camera name to RGBA
                image array. Cameras that fail to capture are omitted.
        """
        if self._scene is None:
            return {} if camera_names is not None else None

        commands: list[CommandEntry] = []
        self._sync_actors(self._scene, commands)

        if camera_names is not None:
            for cam_name in camera_names:
                commands.append(CommandEntry(text=f"vget /camera/{cam_name}/lit"))
        else:
            commands.append(
                CommandEntry(text=f"vget /camera/{self._cfg.camera_name}/lit")
            )

        responses = self._client.request_batch(commands)

        if not responses:
            return {} if camera_names is not None else None

        return self._decode_capture(responses, camera_names)

    def _sync_actors(self, scene: Scene, commands: list[CommandEntry]) -> None:
        """Detect added/removed/updated actors and append commands."""
        current_handles: dict[ActorHandle, Actor] = {}

        def _gather(actor: Actor) -> None:
            if not actor.get_surface_mesh().is_empty():
                current_handles[actor.get_handle()] = actor

        scene.for_each_actor(_gather)

        current_set = set(current_handles.keys())
        known_set = set(self._actor_handles.keys())

        for handle in known_set - current_set:
            info = self._actor_handles.pop(handle)
            commands.append(CommandEntry(text=f"vset /object/{info.name}/destroy"))

        for handle in current_set - known_set:
            commands.extend(self._create_actor_mesh(current_handles[handle]))

        for handle, info in self._actor_handles.items():
            actor = current_handles.get(handle)
            if actor is not None:
                commands.extend(self._update_actor(actor, info))

    def _decode_capture(
        self,
        responses: list,
        camera_names: list[str] | None,
    ) -> RenderFrame | dict[str, RenderFrame] | None:
        """Decode image responses from a render batch."""
        if camera_names is not None:
            num_cameras = len(camera_names)
            image_responses = responses[-num_cameras:]
            result: dict[str, RenderFrame] = {}
            for cam_name, resp in zip(camera_names, image_responses):
                if resp.type == 1:
                    frame = self._decode_image_response(resp.data)
                    if frame is not None:
                        result[cam_name] = frame
            return result

        image_resp = responses[-1]
        if image_resp.type != 1:
            return None
        return self._decode_image_response(image_resp.data)

    def close(self) -> None:
        """
        Close the viewer and disconnect from the server.

        Calling close on an already closed viewer has no effect.
        """
        if not self._client.is_connected():
            return

        # Destroy all known actors and environment glTF
        commands: list[CommandEntry] = []
        for info in self._actor_handles.values():
            commands.append(CommandEntry(text=f"vset /object/{info.name}/destroy"))
        if self._cfg.environment_gltf:
            commands.append(CommandEntry(text="vset /object/__environment__/destroy"))
        if commands:
            try:
                self._client.request_batch(commands)
            except Exception:
                pass

        self._actor_handles.clear()
        self._client.disconnect()
        logger.info("MochiRenderer viewer closed.")

    def set_paused(self, paused: bool) -> None:
        """Sets the paused state of the viewer."""
        self._paused = paused

    def is_paused(self) -> bool:
        """Returns the paused state of the viewer."""
        return self._paused

    def snap_follow_camera(self) -> None:
        """No-op for the MochiRenderer viewer."""
        pass

    def set_camera_view(
        self,
        look_from: list[float] | npt.NDArray | None = None,
        look_at: list[float] | npt.NDArray | None = None,
        **kwargs,
    ) -> None:
        """Set the camera position and target via a look-at command."""
        if look_from is None or look_at is None:
            return
        ef = [float(v) for v in look_from]
        et = [float(v) for v in look_at]
        self._client.request(
            f"vset /camera/{self._cfg.camera_name}/lookat "
            f"{ef[0]} {ef[1]} {ef[2]} {et[0]} {et[1]} {et[2]}"
        )

    def add_grid(self, name: str, **kwargs) -> None:
        """No-op — the mochi_renderer server does not support grids."""
        pass

    def set_enable_follow_camera(self, enabled: bool) -> None:
        """No-op for the MochiRenderer viewer."""
        pass

    def create_camera(self, name: str, width: int, height: int) -> None:
        """Create a named camera on the renderer server.

        Args:
            name: Camera name.
            width: Image width in pixels.
            height: Image height in pixels.
        """
        self._client.request(f"vset /camera/{name}/create {width} {height}")

    def set_camera_transform(
        self,
        name: str,
        position: list[float] | npt.NDArray,
        rotation: list[float] | npt.NDArray,
    ) -> None:
        """Set a named camera's pose via position and orientation.

        Args:
            name: Camera name.
            position: Camera position (x, y, z).
            rotation: Camera orientation as an XYZW quaternion.
        """
        p = [float(v) for v in position]
        r = [float(v) for v in rotation]
        self._client.request(
            f"vset /camera/{name}/transform "
            f"{p[0]} {p[1]} {p[2]} {r[0]} {r[1]} {r[2]} {r[3]}"
        )

    def set_camera_lookat(
        self,
        name: str,
        look_from: list[float] | npt.NDArray,
        look_at: list[float] | npt.NDArray,
    ) -> None:
        """Set a named camera's position and target.

        Args:
            name: Camera name.
            look_from: Eye position (x, y, z).
            look_at: Target position (x, y, z).
        """
        ef = [float(v) for v in look_from]
        et = [float(v) for v in look_at]
        self._client.request(
            f"vset /camera/{name}/lookat "
            f"{ef[0]} {ef[1]} {ef[2]} {et[0]} {et[1]} {et[2]}"
        )

    ####################################################################################
    # Scene management
    ####################################################################################

    def get_scene(self) -> Scene | None:
        """Returns the mochi scene associated with the viewer."""
        return self._scene

    def set_scene(self, scene: Scene | None) -> None:
        """
        Set the mochi scene to render.

        Iterates all actors in the scene, extracts meshes, and sends
        mesh creation commands to the renderer. Also loads the
        environment glTF if configured.

        Args:
            scene: The mochi scene to render, or None to clear.
        """
        # Clean up previous scene actors and environment
        cleanup_commands: list[CommandEntry] = []
        if self._actor_handles:
            for info in self._actor_handles.values():
                cleanup_commands.append(
                    CommandEntry(text=f"vset /object/{info.name}/destroy")
                )
            self._actor_handles.clear()
        if self._cfg.environment_gltf:
            logger.info("[env_gltf] destroying environment gltf")
            cleanup_commands.append(
                CommandEntry(text="vset /object/__environment__/destroy")
            )
        if cleanup_commands:
            try:
                self._client.request_batch(cleanup_commands)
            except Exception:
                pass

        self._scene = scene

        if scene is None:
            return

        # Iterate actors and create meshes
        commands: list[CommandEntry] = []

        # Load environment glTF if configured
        if self._cfg.environment_gltf:
            commands.append(
                CommandEntry(
                    text=f"vset /object/__environment__/gltf {self._cfg.environment_gltf}"
                )
            )

        def _create(actor: Actor) -> None:
            if not actor.get_surface_mesh().is_empty():
                cmds = self._create_actor_mesh(actor)
                commands.extend(cmds)

        scene.for_each_actor(_create)

        if commands:
            self._client.request_batch(commands)

    ####################################################################################
    # Actor helpers
    ####################################################################################

    def _create_actor_mesh(self, actor: Actor) -> list[CommandEntry]:
        """
        Extract mesh data from an actor and build mesh creation commands.

        If the actor has a glTF mapping in ``cfg.actor_gltfs``, a glTF load
        command is sent instead of the raw physics mesh.

        Returns:
            List of CommandEntry objects for mesh creation + initial transform.
        """
        name = actor.get_name()
        actor_type = actor.get_type()

        # Check if this actor has a glTF override
        gltf_path = self._cfg.actor_gltfs.get(name) if self._cfg.actor_gltfs else None

        commands: list[CommandEntry] = []

        if gltf_path is not None:
            # Use glTF mesh instead of physics mesh
            commands.append(CommandEntry(text=f"vset /object/{name}/gltf {gltf_path}"))

            # Send initial transform
            xform_cmd = self._build_xform_command(actor, name)
            if xform_cmd is not None:
                commands.append(xform_cmd)

            # Track this actor
            self._actor_handles[actor.get_handle()] = _ActorInfo(
                name=name,
                actor_type=actor_type,
                num_verts=0,
            )
        else:
            # Use physics mesh (original path)
            is_dynamic = actor_type in (
                ActorType.SOFT,
                ActorType.SHELL,
            )

            # Get mesh data
            actor.register_query_and_compute(mochi.QueryType.SURFACE_NODE_POSITIONS)
            positions = np.asarray(
                actor.get_surface_mesh_node_positions_local(), dtype=np.float32
            ).ravel()

            indices = np.asarray(
                actor.get_surface_mesh().connectivity, dtype=np.int32
            ).ravel()

            actor.register_query_and_compute(mochi.QueryType.SURFACE_NODE_NORMALS)
            normals = np.asarray(
                actor.get_surface_mesh_node_normals_local(), dtype=np.float32
            ).ravel()

            num_verts = len(positions) // 3
            num_indices = len(indices)

            # Build binary payload: [positions] [normals] [indices]
            binary_data = positions.tobytes() + normals.tobytes() + indices.tobytes()

            commands.append(
                CommandEntry(
                    text=(
                        f"vset /object/{name}/mesh "
                        f"{num_verts} {num_indices} {int(is_dynamic)}"
                    ),
                    binary_data=binary_data,
                )
            )

            # Send initial transform
            xform_cmd = self._build_xform_command(actor, name)
            if xform_cmd is not None:
                commands.append(xform_cmd)

            # Show the object
            commands.append(CommandEntry(text=f"vset /object/{name}/show"))

            # Track this actor
            self._actor_handles[actor.get_handle()] = _ActorInfo(
                name=name,
                actor_type=actor_type,
                num_verts=num_verts,
            )

        return commands

    def _update_actor(self, actor: Actor, info: _ActorInfo) -> list[CommandEntry]:
        """
        Build update commands for an existing actor.

        For rigid/articulated actors: sends transform update.
        For soft FEM actors: sends updated mesh vertex data + transform.
        """
        commands: list[CommandEntry] = []

        if info.actor_type in (ActorType.SOFT, ActorType.SHELL):
            # Re-extract vertex positions and normals
            actor.register_query_and_compute(mochi.QueryType.SURFACE_NODE_POSITIONS)
            positions = np.asarray(
                actor.get_surface_mesh_node_positions_local(), dtype=np.float32
            ).ravel()

            actor.register_query_and_compute(mochi.QueryType.SURFACE_NODE_NORMALS)
            normals = np.asarray(
                actor.get_surface_mesh_node_normals_local(), dtype=np.float32
            ).ravel()

            num_verts = len(positions) // 3
            binary_data = positions.tobytes() + normals.tobytes()

            commands.append(
                CommandEntry(
                    text=f"vset /object/{info.name}/update_mesh {num_verts}",
                    binary_data=binary_data,
                )
            )

        # Always update transform
        xform_cmd = self._build_xform_command(actor, info.name)
        if xform_cmd is not None:
            commands.append(xform_cmd)

        return commands

    @staticmethod
    def _build_xform_command(actor: Actor, name: str) -> CommandEntry | None:
        """Build a transform command for an actor, or None if it has no transform.

        The server handles Y-up → Z-up correction for glTF objects.
        """
        if not actor.has_root_transform():
            return None

        xform = actor.get_root_transform()
        pos = np.asarray(xform.translation, dtype=np.float64)
        quat = np.asarray(xform.rotation, dtype=np.float64)

        return CommandEntry(
            text=(
                f"vset /object/{name}/xform "
                f"{pos[0]} {pos[1]} {pos[2]} "
                f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"
            )
        )

    @staticmethod
    def _decode_image_response(data: bytes) -> RenderFrame | None:
        """
        Decode an image response from the server.

        Image format: [uint32 width] [uint32 height] [uint32 channels] [RGBA pixels]
        """
        if len(data) < 12:
            return None

        width, height, channels = struct.unpack_from("<III", data, 0)
        pixel_data = data[12:]

        expected_size = width * height * channels
        if len(pixel_data) < expected_size:
            logger.warning(
                f"Image response truncated: expected {expected_size} bytes, "
                f"got {len(pixel_data)}"
            )
            return None

        image = np.frombuffer(pixel_data[:expected_size], dtype=np.uint8)
        image = image.reshape((height, width, channels))
        return image

    ####################################################################################
    # Context manager support
    ####################################################################################

    def __enter__(self) -> MochiRendererViewer:
        """Support with-statement for the viewer."""
        return self

    def __exit__(self, *ignored) -> bool:
        """Close the viewer at the end of the with-statement."""
        self.close()
        return False  # Propagate exceptions


########################################################################################
# Internal helper
########################################################################################


class _ActorInfo:
    """Bookkeeping for a tracked actor."""

    __slots__ = ("name", "actor_type", "num_verts")

    def __init__(self, name: str, actor_type: ActorType, num_verts: int) -> None:
        self.name = name
        self.actor_type = actor_type
        self.num_verts = num_verts
