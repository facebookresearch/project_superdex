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
from fnmatch import fnmatchcase
from typing import Callable, List, Union

import numpy as np
import numpy.typing as npt
from superdex.physics import Actor, ActorHandle, Scene
from superdex.physics.utils import render_model_registry
from superdex.physics.utils.coordinate_systems import (
    COORDINATE_SYSTEMS,
    CoordinateSystem,
    CoordinateTransform,
    DEFAULT_COORDINATE_SYSTEM,
)
from superdex.physics.utils.transformations import apply_linear_map
from superdex.physics.viewer.backend import (
    polyscope as ps,
    POLYSCOPE_AVAILABLE,
    polyscope_imgui as psim,
    POLYSCOPE_VERSION_GE_2_5_0,
)
from superdex.physics.viewer.logging_handler import LoggingHandler
from superdex.physics.viewer.renderers.actor_renderer import ActorRenderer
from superdex.physics.viewer.renderers.curve_network_renderer import (
    CurveNetworkRenderer,
)
from superdex.physics.viewer.renderers.debug_draw_renderer import DebugDrawRenderer
from superdex.physics.viewer.renderers.glb_actor_renderer import GlbActorRenderer
from superdex.physics.viewer.renderers.grid_renderer import Axes, GridRenderer, Styles
from superdex.physics.viewer.renderers.mesh_renderer import MeshRenderer
from superdex.physics.viewer.renderers.point_cloud_renderer import PointCloudRenderer
from superdex.physics.viewer.renderers.static_plane_renderer import StaticPlaneRenderer
from superdex.physics.viewer.ui import styling
from superdex.physics.viewer.ui.logger_window import build_logger_window
from superdex.physics.viewer.ui.navigation_gizmo import build_navigation_gizmo
from superdex.physics.viewer.ui.sidebar_window import build_sidebar_window
from superdex.physics.viewer.ui.simulation_controls_window import (
    build_simulation_controls_window,
)
from superdex.physics.viewer.utils.aabb import AABB
from superdex.physics.viewer.viewer_cfg import ViewerCfg
from superdex.physics.viewer.viewer_state import ActorState, PlotState, ViewerState

logger = logging.getLogger(__name__)

########################################################################################

VIEWER_AVAILABLE = POLYSCOPE_AVAILABLE and POLYSCOPE_VERSION_GE_2_5_0
"""Whether the viewer is available or not."""

########################################################################################

RenderFrame = npt.NDArray[np.uint8]
"""A Polyscope render frame, consists of an RGB image."""

########################################################################################

_DISPLAY_STATE_ACCESSORS: tuple[tuple[str, str], ...] = (
    ("is_surface_enabled", "set_enable_surface"),
    ("are_edges_enabled", "set_enable_edges"),
    ("are_nodes_enabled", "set_enable_nodes"),
    ("are_axes_enabled", "set_enable_axes"),
    ("get_front_face_color", "set_front_face_color"),
    ("get_back_face_color", "set_back_face_color"),
    ("get_back_face_policy", "set_back_face_policy"),
    ("get_edge_color", "set_edge_color"),
    ("get_edge_width", "set_edge_width"),
    ("get_edge_radius", "set_edge_radius"),
    ("get_node_color", "set_node_color"),
    ("get_node_radius", "set_node_radius"),
    ("get_transparency", "set_transparency"),
    ("get_material", "set_material"),
    ("get_smooth_shading", "set_smooth_shading"),
)
"""(getter, setter) pairs of the @ref MeshRenderer display properties the Scene UI lets
the user edit. They live on the renderer, so swapping a renderer would otherwise reset
them to defaults."""


def capture_display_state(renderer: object) -> dict[str, object]:
    """Snapshots the user-editable display properties of a mesh-backed renderer."""
    if not isinstance(renderer, MeshRenderer):
        return {}
    return {
        setter: getattr(renderer, getter)()
        for getter, setter in _DISPLAY_STATE_ACCESSORS
    }


def apply_display_state(renderer: object, state: dict[str, object]) -> None:
    """Restores a snapshot taken by @ref capture_display_state onto a new renderer."""
    if not isinstance(renderer, MeshRenderer):
        return
    for setter, value in state.items():
        getattr(renderer, setter)(value)


class Viewer:
    """
    Implements a viewer that uses Polyscope to render SuperDex Physics scenes. This viewer is
    intended to be used for debugging and visualization purposes.
    """

    ####################################################################################
    # Members
    ####################################################################################

    # Private members.
    _state: ViewerState

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(self, cfg: ViewerCfg) -> None:
        # Check if it's even possible to initialize the renderer.
        if not VIEWER_AVAILABLE:
            raise RuntimeError(
                "Failed to initialize the renderer. Polyscope is not installed in the "
                "current environment, or it's an incompatible version. Please install "
                "Polyscope >= 2.5.0 to enable the renderer."
            )

        # Validate input configuration.
        # - Resolve coordinate system.
        polyscope_system = COORDINATE_SYSTEMS["polyscope"]
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

        # Initialize Polyscope.
        if ps.is_initialized():
            raise RuntimeError(
                "Polyscope is already initialized, either by the user or from "
                "previous Viewer instances. This is not supported. Please make "
                "sure to call Viewer.close() on any previous Viewer instances "
                "before creating a new one. If you need to use multiple renderers, "
                "please use separate processes for each one of them."
            )

        # Log viewer initialization.
        logger.info("Initializing viewer backend...")
        ps.set_allow_headless_backends(True)
        ps.set_use_prefs_file(False)
        ps.init(cfg.backend)
        ps.remove_all_groups()
        ps.remove_all_structures()
        ps.clear_user_callback()
        ps.set_always_redraw(True)
        ps.set_automatically_compute_scene_extents(False)
        ps.set_ground_plane_mode("none")
        ps.set_length_scale(8)
        ps.set_SSAA_factor(2)
        ps.set_build_default_gui_panels(False)
        ps.set_build_gui(False)
        ps.set_frame_tick_limit_fps_mode("ignore_limits")
        if not (cfg.offscreen or ps.is_headless()):
            ps.set_user_callback(self._update_ui)
        if cfg.size is not None:
            ps.set_window_size(*cfg.size)
        psim.GetIO().IniFilename = None
        logger.info("Viewer backend initialized.")

        # Initialize internal state.
        self._state = ViewerState(
            offscreen=cfg.offscreen,
            paused=cfg.start_paused,
            coordinate_system=coordinate_system,
            coordinate_transform=CoordinateTransform(
                coordinate_system, polyscope_system
            ),
        )

        # Create helpers groups.
        # We'll use these to selectively hide the helpers from the structure list.
        helpers_group = ps.create_group("Helpers")
        helpers_group.set_show_child_details(False)
        helpers_group.set_hide_descendants_from_structure_lists(True)
        self._state.helpers.group = helpers_group

        # Create debug draw helper.
        assert self._state.coordinate_transform is not None, (
            "Coordinate transform must be set"
        )
        self._state.helpers.debug_draw = DebugDrawRenderer(
            self._state.coordinate_transform
        )

        # Initialize logging handler (only for non-offscreen mode).
        if not cfg.offscreen:
            handler = LoggingHandler(max_messages=100)
            logging.getLogger().addHandler(handler)
            self._state.logging.handler = handler

    def __del__(self) -> None:
        """Closes the renderer."""
        self.close()

    ####################################################################################
    # Viewer management methods
    ####################################################################################

    def render(self) -> RenderFrame | None:
        """
        Renders the current environment. Called after each physics step.
        """

        # Perform initial prepare for render.
        self._state.requires_update = True
        self._prepare_for_render()

        # If rendering offscreen, return the screenshot.
        if self._state.offscreen:
            return self.take_screenshot()

        # Otherwise, tick the Polyscope window.
        # If paused, hijack main loop until the user unpauses in the UI, or presses on
        # the window close button. Simply tick Polyscope until the user presses the play
        # button. The only exception is if the user has requested to advance one frame.
        # In that case, we liberate the main loop and let it run until the next render
        # call.
        ps.frame_tick()
        close_requested = ps.window_requests_close()
        while self._state.paused and not self._state.step_once and not close_requested:
            self._prepare_for_render()
            ps.frame_tick()
            close_requested = ps.window_requests_close()
        self._state.step_once = False
        self._state.close_requested = close_requested

    def _handle_reset(self) -> None:
        """Handles a reset request by restoring the scene to its initial state."""
        if (
            self._state.scene.instance is not None
            and self._state.scene.initial_state is not None
        ):
            self._state.scene.instance.restore_state(
                self._state.scene.initial_state, release_immediately=False
            )
            self._state.requires_update = True
            self.snap_follow_camera()

    def close(self) -> None:
        """
        Closes the renderer. Called when the environment is closed. Any resources
        allocated by the renderer should be released at this point. Calling close on a
        already closed renderer has no effect and won't raise an error.
        """

        # Remove logging handler.
        handler = self._state.logging.handler
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            self._state.logging.handler = None

        # Destroy renderers.
        self._remove_actors()
        self._remove_grids()
        self._remove_meshes()
        self._remove_point_clouds()
        self._remove_curve_networks()
        self.get_debug_draw_renderer().remove()

        # Shutdown Polyscope.
        if ps.is_initialized():
            ps.remove_all_groups()
            ps.remove_all_structures()
            ps.clear_user_callback()
            ps.shutdown()

    def take_screenshot(self) -> RenderFrame:
        """Takes a screenshot of the current window."""
        return ps.screenshot_to_buffer()

    def user_requested_close(self) -> bool:
        """Returns whether the user requested to close the window."""
        return self._state.close_requested

    def _prepare_for_render(self) -> None:
        """
        Helper function to prepare for rendering the next frame. This might be called
        once per render call, or multiple times if Polyscope is to hijack the main
        loop and wait for the user to press the play button.
        """

        # Handle reset request if pending.
        if self._state.reset_requested:
            self._handle_reset()
            self._state.reset_requested = False

        if self._state.render_models_dirty:
            self._rebuild_render_model_actors()
            self._state.render_models_dirty = False
            # Force an actor update and bounds recompute this frame, since the GLB and
            # collision meshes have different extents.
            self._state.requires_update = True

        if self._state.requires_update:
            # Update actors only if a scene is available.
            # Validate the scene handle is in sync with its corresponding instance.
            if self._state.scene.instance is not None:
                assert (
                    self._state.scene.handle == self._state.scene.instance.get_handle()
                )
                self._update_actors()
                self._state.debug_requires_update = True

            # Update helpers.
            self._update_grids()
            self._update_meshes()
            self._update_point_clouds()
            self._update_curve_networks()

            # Update scene bounds.
            self._update_scene_bounds()
            self._state.requires_update = False

        if self._state.debug_requires_update:
            self.get_debug_draw_renderer().update()
            self._state.debug_requires_update = False

        # Update camera.
        self._update_camera()

    ####################################################################################
    # Grid helpers methods
    ####################################################################################

    def add_grid(
        self,
        name: str,
        size: float = np.inf,
        center: npt.NDArray[float] | None = None,
        period: float = 1,
        axes: Axes = "xz",
        style: Styles = "checker",
        color_1: npt.ArrayLike | tuple[float, float, float] = (0.9, 0.9, 0.9),
        color_2: npt.ArrayLike | tuple[float, float, float] | tuple[int, int, int] = (
            1,
            1,
            1,
        ),
        double_sided: bool = False,
    ) -> GridRenderer:
        """
        Adds a reference grid to the scene. The grid is a surface mesh consisting of a
        quad of the given size and centered at the given center. The grid is
        parameterized such that the grid lines are spaced by the given period distance.
        The axes parameter controls the direction in which the grid is laid out. The
        default is "xz", which means the grid is laid out in the xz plane. Other options
        are "xy" and "yz". If a reference grid with the same name was previously added,
        it will replaced with the newly created one.
        """

        # If no center is provided, set to the origin.
        center = np.copy(center) if center is not None else np.zeros(3)

        # Try retrieve the existing grid.
        grid = self._state.helpers.grids.get(name, None)
        if grid is not None:
            grid.set_size(size)
            grid.set_center(center)
            grid.set_period(period)
            grid.set_axes(axes)
            grid.set_style(style)
            grid.set_color_1(color_1)
            grid.set_color_2(color_2)
            grid.set_double_sided(double_sided)
            return grid

        # Otherwise, create a new grid. Place the generated render structure in the
        # helpers group so it's not shown in Polyscope's structure list.
        assert self._state.coordinate_transform is not None
        grid = GridRenderer(
            name=name,
            coordinate_transform=self._state.coordinate_transform,
            size=size,
            center=center,
            axes=axes,
            period=period,
            style=style,
            color_1=np.asarray(color_1, dtype=float),
            color_2=np.asarray(color_2, dtype=float),
            double_sided=double_sided,
        )
        self._state.helpers.grids[name] = grid
        if self._state.helpers.group is not None:
            self._state.helpers.group.add_child_structure(grid.get_render_structure())
        return grid

    def get_grids(self) -> list[GridRenderer]:
        """Returns the list of reference grids."""
        return list(self._state.helpers.grids.values())

    def get_grid(self, name: str) -> GridRenderer | None:
        """Returns the reference grid with the given name. None if it doesn't exist."""
        return self._state.helpers.grids.get(name, None)

    def remove_grid(self, grid: str | GridRenderer) -> None:
        """Removes the reference grid with the given name or instance."""
        if isinstance(grid, GridRenderer):
            grid = grid.get_name()
        if grid in self._state.helpers.grids:
            grid_renderer = self._state.helpers.grids[grid]
            grid_renderer.remove()
            del self._state.helpers.grids[grid]

    def _update_grids(self) -> None:
        for grid_renderer in self._state.helpers.grids.values():
            grid_renderer.update()

    def _remove_grids(self) -> None:
        for grid_renderer in self._state.helpers.grids.values():
            grid_renderer.remove()
        self._state.helpers.grids = {}

    ####################################################################################
    # Mesh helpers methods
    ####################################################################################

    def add_mesh(  # noqa: C901
        self,
        name: str,
        coordinates: npt.NDArray[float],
        faces: npt.NDArray[int],
        transform: npt.NDArray[float] | None = None,
        texture_coordinates: npt.NDArray[float] | None = None,
        front_face_color: npt.NDArray[float] | npt.ArrayLike | None = None,
        back_face_color: npt.NDArray[float] | npt.ArrayLike | None = None,
        back_face_policy: MeshRenderer.BackFacePolicy | None = None,
        transparency: float | None = None,
        edge_color: npt.NDArray[float] | npt.ArrayLike | None = None,
        edge_width: float | None = None,
        edge_radius: float | None = None,
        node_color: npt.NDArray[float] | npt.ArrayLike | None = None,
        node_radius: float | None = None,
        smooth_shading: bool | None = None,
        material: MeshRenderer.Material | None = None,
    ) -> MeshRenderer:
        """
        Adds a helper mesh to the scene. Helper meshes are useful for visualizing
        custom geometric objects such as bounding boxes, reference objects, or debug
        visualizations. The mesh is defined by vertex coordinates and face indices.
        If a helper mesh with the same name was previously added, it will be replaced
        with the newly created one.

        Args:
            name: Unique name identifier for the mesh.
            coordinates: Nx3 array of vertex coordinates in local space.
            faces: Mx3 array of triangle face indices.
            transform: 4x4 transformation matrix from local to world space. Defaults to identity.
            texture_coordinates: Nx2 array of texture coordinates. Defaults to None.
            front_face_color: RGB color for front faces. Defaults to polyscope default.
            back_face_color: RGB color for back faces. Defaults to front face color.
            back_face_policy: Policy for rendering back faces. Defaults to polyscope default.
            transparency: Transparency level (0=opaque, 1=fully transparent). Defaults to opaque.
            edge_color: RGB color for edges. Defaults to darker front face color.
            edge_width: Width of edges in surface rendering. Defaults to 0.01.
            edge_radius: Radius of edges in edge rendering. Defaults to 0.00025.
            node_color: RGB color for nodes. Defaults to darker front face color.
            node_radius: Radius of nodes. Defaults to 0.00125.
            material: Material for the surface mesh. Defaults to Clay.

        Returns:
            The created or updated MeshRenderer instance.
        """

        # Try retrieve the existing mesh.
        mesh = self._state.helpers.meshes.get(name, None)
        if mesh is not None:
            # Fall back to identity transform if not supplied.
            if transform is None:
                transform = np.eye(4)
            # Update the mesh.
            mesh.replace_geometry(coordinates, faces, transform, texture_coordinates)
            # Update the mesh properties.
            if front_face_color is not None:
                mesh.set_front_face_color(front_face_color)
            if back_face_color is not None:
                mesh.set_back_face_color(back_face_color)
            if back_face_policy is not None:
                mesh.set_back_face_policy(back_face_policy)
            if transparency is not None:
                mesh.set_transparency(transparency)
            if edge_color is not None:
                mesh.set_edge_color(edge_color)
            if edge_width is not None:
                mesh.set_edge_width(edge_width)
            if edge_radius is not None:
                mesh.set_edge_radius(edge_radius)
            if node_color is not None:
                mesh.set_node_color(node_color)
            if node_radius is not None:
                mesh.set_node_radius(node_radius)
            if smooth_shading is not None:
                mesh.set_smooth_shading(smooth_shading)
            if material is not None:
                mesh.set_material(material)
            return mesh

        # Otherwise, create a new mesh. Place the generated render structure in the
        # helpers group so it's not shown in Polyscope's structure list.
        assert self._state.coordinate_transform is not None
        mesh = MeshRenderer(
            name=name,
            coordinates=coordinates,
            coordinate_transform=self._state.coordinate_transform,
            faces=faces,
            transform=transform,
            texture_coordinates=texture_coordinates,
            front_face_color=front_face_color,
            back_face_color=back_face_color,
            back_face_policy=back_face_policy,
            transparency=transparency,
            edge_color=edge_color,
            edge_width=edge_width,
            edge_radius=edge_radius,
            node_color=node_color,
            node_radius=node_radius,
            smooth_shading=smooth_shading,
            material=material,
        )
        self._state.helpers.meshes[name] = mesh
        if self._state.helpers.group is not None:
            self._state.helpers.group.add_child_group(mesh._group)
        return mesh

    def get_meshes(self) -> list[MeshRenderer]:
        """Returns the list of helper meshes."""
        return list(self._state.helpers.meshes.values())

    def get_mesh(self, name: str) -> MeshRenderer | None:
        """Returns the helper mesh with the given name. None if it doesn't exist."""
        return self._state.helpers.meshes.get(name, None)

    def remove_mesh(self, mesh: str | MeshRenderer) -> None:
        """Removes the helper mesh with the given name or instance."""
        if isinstance(mesh, MeshRenderer):
            mesh = mesh.get_name()
        if mesh in self._state.helpers.meshes:
            mesh_renderer = self._state.helpers.meshes[mesh]
            mesh_renderer.remove()
            del self._state.helpers.meshes[mesh]

    def _update_meshes(self) -> None:
        """Updates all helper meshes. Called during prepare for render."""
        for mesh_renderer in self._state.helpers.meshes.values():
            mesh_renderer.update()

    def _remove_meshes(self) -> None:
        """Removes all helper meshes. Called during viewer close."""
        for mesh_renderer in self._state.helpers.meshes.values():
            mesh_renderer.remove()
        self._state.helpers.meshes = {}

    ####################################################################################
    # Point cloud helpers methods
    ####################################################################################

    def add_point_cloud(
        self,
        name: str,
        coordinates: npt.NDArray[float] | npt.ArrayLike,
        radius: float | None = None,
        radii: npt.NDArray[float] | npt.ArrayLike | None = None,
        color: npt.NDArray[float] | npt.ArrayLike | None = None,
        colors: npt.NDArray[float] | npt.ArrayLike | None = None,
        transparency: float | None = None,
    ) -> PointCloudRenderer:
        """
        Adds a helper point cloud to the scene. Point clouds are useful for visualizing
        scattered 3D data such as sampled points, particle systems, or debug markers.
        If a helper point cloud with the same name was previously added, it will be
        replaced with the newly created one.

        Args:
            name: Unique name identifier for the point cloud.
            coordinates: Nx3 array of point coordinates in global space.
            radius: Global radius for all points. Defaults to 0.01. Ignored if
                per-point radii are specified.
            radii: Optional N-element array of per-point radii. If provided,
                overrides the global radius.
            color: RGB color for all points as a 3-element array. Defaults to
                Polyscope default. Ignored if per-point colors are specified.
            colors: Optional Nx3 array of per-point RGB colors. If provided,
                overrides the global color.
            transparency: Transparency level (0=opaque, 1=fully transparent).
                Defaults to opaque.

        Returns:
            The created or updated PointCloudRenderer instance.
        """

        # Try retrieve the existing point cloud.
        point_cloud = self._state.helpers.point_clouds.get(name, None)
        if point_cloud is not None:
            # Update the point cloud geometry.
            point_cloud.set_points(coordinates, radii, colors)
            # Update the point cloud properties.
            if radius is not None:
                point_cloud.set_radius(radius)
            if color is not None:
                point_cloud.set_color(color)
            if transparency is not None:
                point_cloud.set_transparency(transparency)
            return point_cloud

        # Otherwise, create a new point cloud. Place the generated render structure in
        # the helpers group so it's not shown in Polyscope's structure list.
        assert self._state.coordinate_transform is not None
        point_cloud = PointCloudRenderer(
            name=name,
            coordinates=coordinates,
            coordinate_transform=self._state.coordinate_transform,
            radius=radius,
            radii=radii,
            color=color,
            colors=colors,
            transparency=transparency,
        )
        self._state.helpers.point_clouds[name] = point_cloud
        assert self._state.helpers.group is not None
        self._state.helpers.group.add_child_structure(
            point_cloud.get_render_structure()
        )
        return point_cloud

    def get_point_clouds(self) -> list[PointCloudRenderer]:
        """Returns the list of helper point clouds."""
        return list(self._state.helpers.point_clouds.values())

    def get_point_cloud(self, name: str) -> PointCloudRenderer | None:
        """Returns the helper point cloud with the given name. None if it doesn't exist."""
        return self._state.helpers.point_clouds.get(name, None)

    def remove_point_cloud(self, point_cloud: str | PointCloudRenderer) -> None:
        """Removes the helper point cloud with the given name or instance."""
        if isinstance(point_cloud, PointCloudRenderer):
            point_cloud = point_cloud.get_name()
        if point_cloud in self._state.helpers.point_clouds:
            point_cloud_renderer = self._state.helpers.point_clouds[point_cloud]
            point_cloud_renderer.remove()
            del self._state.helpers.point_clouds[point_cloud]

    def _update_point_clouds(self) -> None:
        """Updates all helper point clouds. Called during prepare for render."""
        for point_cloud_renderer in self._state.helpers.point_clouds.values():
            point_cloud_renderer.update()

    def _remove_point_clouds(self) -> None:
        """Removes all helper point clouds. Called during viewer close."""
        for point_cloud_renderer in self._state.helpers.point_clouds.values():
            point_cloud_renderer.remove()
        self._state.helpers.point_clouds = {}

    ####################################################################################
    # Curve network helpers methods
    ####################################################################################

    def add_curve_network(
        self,
        name: str,
        nodes: npt.NDArray[float] | npt.ArrayLike,
        edges: npt.NDArray[int] | npt.ArrayLike,
        radius: float | None = None,
        node_radii: npt.NDArray[float] | npt.ArrayLike | None = None,
        edge_radii: npt.NDArray[float] | npt.ArrayLike | None = None,
        color: npt.NDArray[float] | npt.ArrayLike | None = None,
        node_colors: npt.NDArray[float] | npt.ArrayLike | None = None,
        edge_colors: npt.NDArray[float] | npt.ArrayLike | None = None,
        transparency: float | None = None,
    ) -> CurveNetworkRenderer:
        """
        Adds a helper curve network to the scene. Curve networks are useful for
        visualizing line segments, graphs, or skeletal structures.
        If a helper curve network with the same name was previously added, it will be
        replaced with the newly created one.

        Args:
            name: Unique name identifier for the curve network.
            nodes: Nx3 array of node coordinates in global space.
            edges: Mx2 array of edge indices, where each row specifies the indices
                of the two nodes that form an edge.
            radius: Global radius for all edges. Defaults to 0.005. Ignored if
                per-node/per-edge radii are specified.
            node_radii: Optional N-element array of per-node radii. If provided,
                overrides the global radius for nodes.
            edge_radii: Optional M-element array of per-edge radii. If provided,
                overrides the global radius for edges.
            color: RGB color for all elements as a 3-element array. Defaults to
                Polyscope default. Ignored if per-node/per-edge colors are specified.
            node_colors: Optional Nx3 array of per-node RGB colors. If provided,
                overrides the global color for nodes.
            edge_colors: Optional Mx3 array of per-edge RGB colors. If provided,
                overrides the global color for edges.
            transparency: Transparency level (0=opaque, 1=fully transparent).
                Defaults to opaque.

        Returns:
            The created or updated CurveNetworkRenderer instance.
        """

        # Try retrieve the existing curve network.
        curve_network = self._state.helpers.curve_networks.get(name, None)
        if curve_network is not None:
            # Update the curve network geometry.
            curve_network.set_curve_network(
                nodes, edges, node_radii, edge_radii, node_colors, edge_colors
            )
            # Update the curve network properties.
            if radius is not None:
                curve_network.set_radius(radius)
            if color is not None:
                curve_network.set_color(color)
            if transparency is not None:
                curve_network.set_transparency(transparency)
            return curve_network

        # Otherwise, create a new curve network. Place the generated render structure in
        # the helpers group so it's not shown in Polyscope's structure list.
        assert self._state.coordinate_transform is not None
        curve_network = CurveNetworkRenderer(
            name=name,
            nodes=nodes,
            edges=edges,
            coordinate_transform=self._state.coordinate_transform,
            radius=radius,
            node_radii=node_radii,
            edge_radii=edge_radii,
            color=color,
            node_colors=node_colors,
            edge_colors=edge_colors,
            transparency=transparency,
        )
        self._state.helpers.curve_networks[name] = curve_network
        assert self._state.helpers.group is not None
        self._state.helpers.group.add_child_structure(
            curve_network.get_render_structure()
        )
        return curve_network

    def get_curve_networks(self) -> list[CurveNetworkRenderer]:
        """Returns the list of helper curve networks."""
        return list(self._state.helpers.curve_networks.values())

    def get_curve_network(self, name: str) -> CurveNetworkRenderer | None:
        """Returns the helper curve network with the given name. None if it doesn't exist."""
        return self._state.helpers.curve_networks.get(name, None)

    def remove_curve_network(self, curve_network: str | CurveNetworkRenderer):
        """Removes the helper curve network with the given name or instance."""
        if isinstance(curve_network, CurveNetworkRenderer):
            curve_network = curve_network.get_name()
        if curve_network in self._state.helpers.curve_networks:
            curve_network_renderer = self._state.helpers.curve_networks[curve_network]
            curve_network_renderer.remove()
            del self._state.helpers.curve_networks[curve_network]

    def _update_curve_networks(self):
        """Updates all helper curve networks. Called during prepare for render."""
        for curve_network_renderer in self._state.helpers.curve_networks.values():
            curve_network_renderer.update()

    def _remove_curve_networks(self):
        """Removes all helper curve networks. Called during viewer close."""
        for curve_network_renderer in self._state.helpers.curve_networks.values():
            curve_network_renderer.remove()
        self._state.helpers.curve_networks = {}

    def add_debug_lines(
        self,
        name: str,
        positions: List[tuple[float, float, float]],
        colors: List[tuple[int, int, int, int]] | None = None,
        radius: float = 0.002,
    ) -> CurveNetworkRenderer | None:
        """
        Adds debug line segments to the viewer using the curve network renderer.

        This is a convenience method for visualizing line segments from debug draw data,
        such as MPC trajectory visualization. The positions list contains pairs of points
        that define line segments (positions[0]-positions[1], positions[2]-positions[3], etc.).

        Args:
            name: Unique name identifier for the debug lines.
            positions: List of 3D positions as tuples. Each consecutive pair of positions
                defines a line segment. Must have an even number of positions.
            colors: Optional list of RGBA colors (0-255 range) for each position.
                If provided, colors will be normalized to [0, 1] range.
                If None, a default color will be used.
            radius: Radius of the line segments. Defaults to 0.002.

        Returns:
            The created CurveNetworkRenderer instance, or None if no valid line segments.
        """
        if len(positions) < 2:
            # Remove existing curve network if it exists
            if name in self._state.helpers.curve_networks:
                self.remove_curve_network(name)
            return None

        # Convert positions to nodes array
        nodes = np.array(positions, dtype=np.float32)
        num_nodes = nodes.shape[0]

        # Create edges connecting consecutive pairs of positions
        # positions[0]-positions[1], positions[2]-positions[3], etc.
        num_edges = num_nodes // 2
        edges = np.zeros((num_edges, 2), dtype=np.int32)
        for i in range(num_edges):
            edges[i, 0] = i * 2
            edges[i, 1] = i * 2 + 1

        # Process colors if provided
        node_colors = None
        if colors is not None and len(colors) >= num_nodes:
            # Convert from 0-255 RGBA to 0-1 RGB
            node_colors = np.array(
                [
                    [c[0] / 255.0, c[1] / 255.0, c[2] / 255.0]
                    for c in colors[:num_nodes]
                ],
                dtype=np.float32,
            )

        # Create or update the curve network
        return self.add_curve_network(
            name=name,
            nodes=nodes,
            edges=edges,
            radius=radius,
            node_colors=node_colors,
        )

    ####################################################################################
    # Scene management
    ####################################################################################

    def get_scene(self) -> Scene | None:
        """Returns the SuperDex Physics scene associated with the renderer."""
        return self._state.scene.instance

    def set_scene(self, scene: Scene | None) -> None:
        """Sets the SuperDex Physics scene associated with the renderer."""

        # Perform updates only if the given scene is different.
        handle = None if scene is None else scene.get_handle()
        if handle == self._state.scene.handle:
            return

        # Clear the previous initial state. We don't call release_state() here because
        # the previous scene may have been destroyed externally (e.g., via a context
        # manager). The state will be cleaned up when the scene itself is destroyed.
        self._state.scene.initial_state = None

        # Update SuperDex Physics scene instance.
        self._state.scene.instance = scene
        self._state.scene.handle = handle

        # If no scene was given, remove all actors and the debug draw.
        # Fall back to dummy AABB and scene scale.
        if scene is None:
            self._remove_actors()
            self.get_debug_draw_renderer().remove()
            return

        # Capture initial state for reset functionality.
        self._state.scene.initial_state = scene.capture_state()

        # Reset renderers.
        self._update_actors()
        self._update_scene_bounds()
        self.get_debug_draw_renderer().reset(scene)

    def is_paused(self) -> bool:
        """Returns the paused state of the renderer."""
        return self._state.paused

    def set_paused(self, paused: bool) -> None:
        """Sets the paused state of the renderer."""
        self._state.paused = paused

    def get_scene_bounds(self) -> AABB:
        """Returns the scene bounds."""
        return self._state.scene.aabb

    def get_debug_draw_renderer(self) -> DebugDrawRenderer:
        """Returns the debug draw renderer for customizing debug visualization."""
        assert self._state.helpers.debug_draw is not None
        return self._state.helpers.debug_draw

    def _update_scene_bounds(self) -> None:
        """Updates the scene bounds."""

        # Gather renderable actors AABBs. Exclude static plane actors (e.g. the ground
        # plane): their AABB is a single point pinned at the plane offset (near the
        # origin), which would otherwise drag the scene-bounds center toward the origin
        # and cause the follow camera to lag behind actors that move away from it.
        actor_aabbs = [
            actor.renderer.get_aabb()
            for actor in self._state.scene.actors.values()
            if actor.renderer is not None
            and not isinstance(actor.renderer, StaticPlaneRenderer)
        ]

        # Use this to recompute the scene AABB.
        # Update Polyscope's AABB instead of letting it recompute so it can effectively
        # ignore helpers such as the reference grid.
        if actor_aabbs:
            self._state.scene.aabb.compute_from_aabbs(actor_aabbs)
            min_point = self._state.scene.aabb.min
            max_point = self._state.scene.aabb.max
            assert self._state.coordinate_transform is not None
            source_to_ps = self._state.coordinate_transform.source_to_target
            min_point_ps = apply_linear_map(source_to_ps[:3, :3], min_point)
            max_point_ps = apply_linear_map(source_to_ps[:3, :3], max_point)
            ps.set_bounding_box(
                low=np.minimum(min_point_ps, max_point_ps),
                high=np.maximum(min_point_ps, max_point_ps),
            )
        else:
            # Clear bounds from the previous scene or actor set. Polyscope requires an
            # explicit dummy box because automatic scene-extent computation is disabled.
            self._state.scene.aabb.set_empty()
            ps.set_bounding_box(low=np.zeros(3), high=np.zeros(3))

    ####################################################################################
    # Actor management
    ####################################################################################

    def get_actors(self) -> list[Actor]:
        """Returns the list of actors in the scene that are not excluded."""
        return [actor.instance for actor in self._state.scene.actors.values()]

    def get_actor_renderers(
        self,
    ) -> list[Union[ActorRenderer, StaticPlaneRenderer]]:
        """Returns the list of actors renderers in the scene. Note that the size of this
        list does not necessarily match the number of actors in the scene. Some actors
        may not be renderable."""
        return [
            a.renderer
            for a in self._state.scene.actors.values()
            if a.renderer is not None
        ]

    def get_actor_renderer(
        self, actor: Actor
    ) -> Union[ActorRenderer, StaticPlaneRenderer, None]:
        """Returns the actor renderer from the given instance. None if it doesn't exist,
        or the actor is not renderable."""
        actor_handle = actor.get_handle()
        actor_state = self._state.scene.actors.get(actor_handle, None)
        return None if actor_state is None else actor_state.renderer

    def set_excluded_actors(self, excluded_actors: list[str]) -> None:
        """Sets the list of actors to exclude from rendering. Excluded actors can be
        specified via their names, or through Unix shell-style wildcard patterns. This
        function triggers an internal reset, causing the actor renderers to be recreated
        with the new list of excluded actors."""
        self._state.scene.excluded_actors = set(excluded_actors)
        self._update_actors()

    def _is_valid_actor(self, name: str) -> bool:
        """Returns whether the given actor is valid for rendering. An actor is valid if
        its name does not match any of the exclusion patterns."""
        return not any(
            fnmatchcase(name, pattern) for pattern in self._state.scene.excluded_actors
        )

    def _is_plane_actor(self, actor: Actor) -> bool:
        """Returns whether the given actor is a plane shape.

        A plane shape is identified by having an AABB with at least one axis
        having infinite bounds, which is unique to plane collision shapes.
        Other implicit shapes (sphere, box, capsule) have finite AABBs.
        """
        try:
            if not actor.get_surface_mesh().is_empty() or not actor.is_static():
                return False
            aabb = actor.get_aabb_world()
            # Plane shapes should have infinite bounds in directions perpendicular to the normal
            has_infinite_bound = (
                not np.isfinite(aabb.min[0])
                or not np.isfinite(aabb.max[0])
                or not np.isfinite(aabb.min[1])
                or not np.isfinite(aabb.max[1])
                or not np.isfinite(aabb.min[2])
                or not np.isfinite(aabb.max[2])
            )
            return has_infinite_bound
        except Exception:
            return False

    def _gather_valid_actors_from_scene(self) -> dict[ActorHandle, Actor]:
        """Returns a dictionary of valid actors from the current scene, keyed by actor handle."""
        actors = {}

        def gather_actor(actor: Actor):
            name = actor.get_name()
            if self._is_valid_actor(name):
                handle = actor.get_handle()
                actors[handle] = actor

        assert self._state.scene.instance is not None
        self._state.scene.instance.for_each_actor(gather_actor)
        return actors

    def _create_glb_renderer(
        self, handle: ActorHandle, instance: Actor
    ) -> GlbActorRenderer | None:
        """Builds a GLB renderer for the actor if a visual render model is registered.

        Returns None if no render model is registered for this actor, or if loading the
        registered GLB fails (in which case the caller falls back to the physics mesh).
        """
        if not self._state.show_visual_mesh:
            return None
        if self._state.scene.handle is None:
            return None
        entry = render_model_registry.get(self._state.scene.handle, handle)
        if entry is None:
            return None
        assert self._state.coordinate_transform is not None
        try:
            return GlbActorRenderer(
                instance,
                entry.glb_path,
                entry.local_transform,
                entry.scale,
                self._state.coordinate_transform,
            )
        except Exception:
            logger.warning(
                "Failed to load GLB render model '%s' for actor '%s'; falling back "
                "to the physics surface mesh.",
                entry.glb_path,
                instance.get_name(),
                exc_info=True,
            )
            return None

    def _create_actor_renderer(
        self, handle: ActorHandle, instance: Actor
    ) -> ActorRenderer | StaticPlaneRenderer | None:
        """Creates the preferred renderer for an actor, with physics fallbacks."""
        renderer = self._create_glb_renderer(handle, instance)
        if renderer is not None:
            return renderer

        coordinate_transform = self._state.coordinate_transform
        assert coordinate_transform is not None
        if not instance.get_surface_mesh().is_empty():
            return ActorRenderer(instance, coordinate_transform)
        if self._is_plane_actor(instance):
            return StaticPlaneRenderer(instance, coordinate_transform)
        return None

    def _rebuild_render_model_actors(self) -> None:
        """Rebuilds renderers for actors that have a registered visual render model, so
        that a change to show_visual_mesh takes effect on already-created actors."""
        scene_handle = self._state.scene.handle
        if scene_handle is None:
            return
        for handle, actor_state in self._state.scene.actors.items():
            entry = render_model_registry.get(scene_handle, handle)
            if entry is None:
                continue
            instance = actor_state.instance
            previous = actor_state.renderer
            if (
                not self._state.show_visual_mesh
                and previous is not None
                and instance.get_surface_mesh().is_empty()
                and not self._is_plane_actor(instance)
            ):
                # A link that ships only a visual mesh has no physics geometry to fall
                # back to; keep drawing the GLB rather than making the actor vanish.
                continue
            # The replacement registers the same Polyscope structure name, so the old
            # renderer has to go first; carry its display state across the swap.
            display_state = capture_display_state(previous)
            if previous is not None:
                previous.remove()
            renderer = self._create_actor_renderer(handle, instance)
            apply_display_state(renderer, display_state)
            actor_state.renderer = renderer

    def _update_actors(self) -> None:
        """
        Updates the actors in the scene. If the scene has changed (i.e. a new scene was
        set, or an actor was added or removed), this function will create or destroy
        actor renderers as necessary.
        """

        # Retrieve all the valid actors in the SuperDex Physics scene.
        mochi_actors = self._gather_valid_actors_from_scene()

        # Determine actors to create and destroy.
        actor_handles = set(mochi_actors)
        existing_actor_handles = set(self._state.scene.actors.keys())
        actor_handles_to_add = actor_handles - existing_actor_handles
        actor_handles_to_remove = existing_actor_handles - actor_handles

        # Remove destroyed actors.
        for handle in actor_handles_to_remove:
            actor = self._state.scene.actors[handle]
            if actor.renderer is not None:
                actor.renderer.remove()
            del self._state.scene.actors[handle]

        # If the selected actor is no longer in the scene, reset selection.
        if self._state.scene.selected_actor in actor_handles_to_remove:
            self._state.scene.selected_actor = None

        # Update existing actors.
        # Do this before creating new ones to prevent double updating.
        for actor in self._state.scene.actors.values():
            renderer = actor.renderer
            if renderer is not None:
                renderer.set_show_axes_at_com(self._state.use_com_transform)
                renderer.update()

        # Create new actors.
        for handle in actor_handles_to_add:
            instance = mochi_actors[handle]
            renderer = self._create_actor_renderer(handle, instance)
            self._state.scene.actors[handle] = ActorState(handle, instance, renderer)

        # Finally, if actors were added, sort the internal state to ensure they're
        # kept alphabetically sorted in the UI.
        if actor_handles_to_add:
            sorted_pairs = sorted(
                self._state.scene.actors.items(),
                key=lambda kv: kv[1].instance.get_name(),
            )
            self._state.scene.actors = dict(sorted_pairs)

    def _remove_actors(self) -> None:
        """Removes all actor renderers from the scene."""
        for actor in self._state.scene.actors.values():
            if actor.renderer is not None:
                actor.renderer.remove()
        self._state.scene.actors.clear()
        self._state.scene.selected_actor = None
        self._state.scene.aabb.set_empty()

    ####################################################################################
    # Camera management
    ####################################################################################

    def set_camera_view(
        self,
        look_from: npt.NDArray[float] | npt.ArrayLike,
        look_at: npt.NDArray[float] | npt.ArrayLike,
        up_dir: npt.NDArray[float] | npt.ArrayLike | None = None,
        fly_to: bool = False,
    ) -> None:
        """
        Sets the camera position and orientation such that the camera is looking at the
        given target. The camera is positioned at the given look_from position, and the
        camera is oriented such that the given up_dir is pointing up. If up_dir is not
        specified, the up direction from the configured coordinate system will be used.
        """

        # Cancel fly-to if the camera is on follow mode. Otherwise, the camera will
        # not animate to the new position.
        if self._state.camera.use_follow_camera:
            fly_to = False

        # Determine look-from and look-at coordinates.
        assert self._state.coordinate_transform is not None
        source_to_ps = self._state.coordinate_transform.source_to_target
        look_from = np.asarray(look_from, dtype=np.float32)
        look_from_ps = source_to_ps[:3, :3] @ look_from
        look_at = np.asarray(look_at, dtype=np.float32)
        look_at_ps = source_to_ps[:3, :3] @ look_at

        # Determine up dir. If none provided, use the coordinate system up dir.
        # If the up dir is not orthogonal to the look dir, use the system right dir as
        # up dir.
        if up_dir is None:
            look_dir = look_at - look_from
            look_dir /= np.linalg.norm(look_dir)
            up_dir = self.get_coordinate_system().up.to_vector()
            if np.abs(np.dot(up_dir, look_dir)) > 0.99:
                up_dir = self.get_coordinate_system().right.to_vector()
        else:
            up_dir = np.asarray(up_dir, dtype=np.float32)
        up_dir_ps = source_to_ps[:3, :3] @ up_dir

        # Polyscope turntable camera sometimes bugs out if the current camera and the
        # new camera are orthogonal - Setting navigation style to free fixes this.
        current_navigation_style = ps.get_navigation_style()
        ps.set_navigation_style("free")

        # Set camera view center and look direction.
        ps.set_view_center(look_at_ps)
        ps.look_at_dir(look_from_ps, look_at_ps, up_dir_ps, fly_to=fly_to)

        # Restore user navigation style.
        ps.set_navigation_style(current_navigation_style)

    def get_camera_position(self) -> npt.NDArray[float]:
        """Returns the current camera position."""
        assert self._state.coordinate_transform is not None
        ps_to_source = self._state.coordinate_transform.target_to_source
        position_ps = ps.get_view_camera_parameters().get_position()
        position = ps_to_source[:3, :3] @ position_ps
        return np.asarray(position, dtype=np.float32)

    def get_camera_up_dir(self) -> npt.NDArray[float]:
        """Returns the current camera up direction."""
        assert self._state.coordinate_transform is not None
        ps_to_source = self._state.coordinate_transform.target_to_source
        up_ps = ps.get_view_camera_parameters().get_up_dir()
        up = ps_to_source[:3, :3] @ up_ps
        return np.asarray(up, dtype=np.float32)

    def get_camera_look_dir(self) -> npt.NDArray[float]:
        """Returns the current camera look direction."""
        assert self._state.coordinate_transform is not None
        ps_to_source = self._state.coordinate_transform.target_to_source
        look_dir_ps = ps.get_view_camera_parameters().get_look_dir()
        look_dir = ps_to_source[:3, :3] @ look_dir_ps
        return np.asarray(look_dir, dtype=np.float32)

    def set_camera_view_center(
        self, view_center: npt.NDArray[float] | npt.ArrayLike
    ) -> None:
        """Sets the view center of the camera. This changes the point around which the
        turntable camera rotates. The view center is expressed in world space, and is
        updated whenever a camera view is set through `set_camera_view`. Users can
        alter the view center to focus on particular areas of interest within the viewer
        by holding ctrl+shift (cmd+shift on macOS) and clicking in the scene."""
        assert self._state.coordinate_transform is not None
        source_to_ps = self._state.coordinate_transform.source_to_target
        view_center = np.asarray(view_center, dtype=np.float32)
        view_center_ps = source_to_ps[:3, :3] @ view_center
        ps.set_view_center(view_center_ps)

    def get_camera_view_center(self) -> npt.NDArray[float]:
        """Returns the view center of the camera. This is the point around which the
        turntable camera rotates. The view center is expressed in world space."""
        assert self._state.coordinate_transform is not None
        ps_to_source = self._state.coordinate_transform.target_to_source
        view_center_ps = ps.get_view_center()
        view_center = ps_to_source[:3, :3] @ view_center_ps
        return np.asarray(view_center, dtype=np.float32)

    def is_follow_camera_enabled(self) -> bool:
        """Returns whether the follow camera is enabled."""
        return self._state.camera.use_follow_camera

    def set_enable_follow_camera(self, enabled: bool) -> None:
        """Enables the follow camera, which automatically centers the camera on the
        rendered actors. The follow camera is disabled by default."""
        self._state.camera.use_follow_camera = enabled

    def set_follow_camera_smoothness(self, smoothing: float) -> None:
        """Sets the smoothness of the follow camera. This helps reduce the jittering
        caused by fast-moving actors. The smoothness is a value between 0 and 1, where
        0 means no smoothing, and 1 means full smoothing."""
        self._state.camera.smoothing = smoothing

    def get_follow_camera_smoothness(self) -> float:
        """Returns the smoothness of the follow camera."""
        return self._state.camera.smoothing

    def set_compute_automatic_distance(self, enabled: bool) -> None:
        """Sets whether or not to use automatic distance to the actors. If enabled, the
        camera is positioned at a distance in which the entire scene is framed. Only
        used when follow camera is enabled."""
        self._state.camera.automatic_distance = enabled

    def is_automatic_distance_enabled(self) -> bool:
        """Returns whether or not automatic distance to the actors is enabled."""
        return self._state.camera.automatic_distance

    def get_coordinate_system(self) -> CoordinateSystem:
        """Returns the coordinate system used for visualization. Note the coordinate
        system can only be specified at initialization time, and cannot be changed
        afterwards."""
        assert self._state.coordinate_system is not None
        return self._state.coordinate_system

    def snap_follow_camera(self) -> None:
        """Resets the follow camera. This function should be called in instances where
        the scene is reset (e.g. when switching scenes). This will immediately snap the
        camera to the current scene bounds without smoothing."""

        if not self._state.camera.use_follow_camera:
            return
        view_distance = self._state.camera.distance
        if view_distance > 0:
            look_dir = ps.get_view_camera_parameters().get_look_dir()
            self.frame_scene_from_direction(look_dir, view_distance)

    def frame_scene(
        self, look_dir: npt.NDArray[float] | None = None, fly_to: bool = False
    ) -> None:
        """Frames the scene, centering the camera on the scene's AABB. The scene's AABB
        is computed from the actors that are not excluded from rendering. Note that if
        not scene is loaded in the SuperDex Physics scene, this function will defer the framing to
        the next update in which a scene is loaded."""

        # If the AABB has not yet been computed (i.e. no render actors have been yet
        # created), defer framing to the next render call.
        if self._state.scene.aabb.is_empty:
            self._state.camera.frame_camera_on_next_update = True
            self._state.camera.frame_camera_direction = look_dir
            self._state.camera.frame_fly_to = fly_to
            return

        # Use current look direction if not specified.
        if look_dir is None:
            look_dir = self.get_camera_look_dir()

        # Get current view parameters.
        camera_params = ps.get_view_camera_parameters()
        aspect = camera_params.get_aspect()
        half_fov = np.deg2rad(camera_params.get_fov_vertical_deg()) / 2

        # Compute follow distance based on the radius of the enclosing AABB.
        # Use the maximum half fov (horizontal or vertical, depends on aspect ratio)
        # to compute the distance.
        radius = np.linalg.norm(self._state.scene.aabb.extents) / 2
        max_half_fov = np.maximum(half_fov, half_fov / aspect)
        distance = 1.5 * radius / np.tan(max_half_fov)

        # Update camera.
        self.frame_scene_from_direction(look_dir, distance, fly_to)

    def frame_scene_from_direction(
        self, look_dir: npt.NDArray[float], distance: float, fly_to: bool = False
    ) -> None:
        """Frames the scene from the given direction. The scene's AABB is computed from
        the actors that are not excluded from rendering. The direction and distance
        parameters control the position of the camera."""

        # Update camera position.
        look_at = self._state.scene.aabb.center
        look_from = look_at - distance * look_dir
        self.set_camera_view(look_from, look_at, fly_to=fly_to)

        # Snap the smoothed target position for the follow camera. Only used if enabled.
        # Also record the last view distance to actors. We'll use this to properly
        # reposition the follow camera when resetting the scene.
        self._state.camera.smoothed_target_position = look_at
        self._state.camera.distance = distance
        self._state.camera.smoothed_distance = distance

    def _compute_frame_distance(self) -> float:
        """Computes the distance from the camera to the scene's AABB. This is used to
        determine the distance of the follow camera.
        """
        # Get current view parameters.
        camera_params = ps.get_view_camera_parameters()
        aspect = camera_params.get_aspect()
        half_fov = np.deg2rad(camera_params.get_fov_vertical_deg()) / 2

        # Compute distance based on the radius of the enclosing AABB.
        # Use the maximum half fov (horizontal or vertical, depends on aspect ratio)
        # to compute the distance.
        radius = np.linalg.norm(self._state.scene.aabb.extents) / 2
        max_half_fov = np.maximum(half_fov, half_fov / aspect)
        return 1.5 * radius / np.tan(max_half_fov)

    def _update_camera(self) -> None:
        """Updates the camera. This function is called every frame."""
        if self._state.camera.frame_camera_on_next_update:
            self.frame_scene(
                self._state.camera.frame_camera_direction,
                self._state.camera.frame_fly_to,
            )
            self._state.camera.frame_camera_on_next_update = False
            self._state.camera.frame_camera_direction = None
            self._state.camera.frame_fly_to = False
        if self._state.camera.use_follow_camera:
            self._update_follow_camera()

    def _update_follow_camera(self) -> None:
        """Updates the follow camera. This function is called every frame while the
        follow camera is enabled."""
        # Ignore follow camera while holding down the shift button.
        # This will allow the user to move around the object even in this mode.
        if psim.IsKeyDown(psim.ImGuiKey_ModShift):
            return

        # Get current view parameters.
        camera_params = ps.get_view_camera_parameters()
        position_ps = camera_params.get_position()
        look_dir_ps = camera_params.get_look_dir()
        up_dir_ps = camera_params.get_up_dir()

        def smooth(current, target):
            # TODO: Use viewer delta time for frame-rate independent smoothing.
            alpha = self._state.camera.smoothing
            return alpha * current + (1 - alpha) * target

        # Update target position.
        assert self._state.coordinate_transform is not None
        source_to_ps = self._state.coordinate_transform.source_to_target
        self._state.camera.smoothed_target_position = smooth(
            self._state.camera.smoothed_target_position, self._state.scene.aabb.center
        )
        smoothed_target_position_ps = (
            source_to_ps[:3, :3] @ self._state.camera.smoothed_target_position
        )

        # Determine distance from target. Smooth only if automatic distance is enabled.
        # Otherwise, changes in distance done by the user (i.e. from mouse wheel) will
        # result in jittering. Store the computed distance to actors to properly
        # reposition the follow camera when resetting the scene.
        if self._state.camera.automatic_distance:
            distance = self._compute_frame_distance()
            self._state.camera.distance = distance
            self._state.camera.smoothed_distance = smooth(
                self._state.camera.smoothed_distance, self._state.camera.distance
            )
        else:
            view_center_ps = ps.get_view_center()
            distance = np.linalg.norm(position_ps - view_center_ps)
            self._state.camera.distance = distance
            self._state.camera.smoothed_distance = distance

        # Update camera.
        look_at_ps = smoothed_target_position_ps
        look_from_ps = look_at_ps - self._state.camera.smoothed_distance * look_dir_ps
        ps.look_at_dir(look_from_ps, look_at_ps, up_dir_ps, fly_to=False)
        ps.set_view_center(look_at_ps)

    ####################################################################################
    # Plotting
    ####################################################################################

    def add_plot(self, plot: PlotState):
        """Add a user-defined 2D line segment plot to the viewer. User can optionally
        define the name, legend, and axis information for the plot. Different plots with
        the same name are grouped and rendered in the same plot. Note that for each plot
        group, only one of them can have axis information defined."""

        if self._state.plots is None:
            self._state.plots = []
        self._state.plots.append(plot)

    def get_plot(self, name: str, legend: str | None = None) -> List[PlotState]:
        """Return all the plots with the given name (and optionally legend)."""
        if self._state.plots is None:
            return []
        return [
            plot for plot in self._state.plots if self.match_plot(plot, name, legend)
        ]

    def remove_plot(self, name: str, legend: str | None = None):
        if self._state.plots is None:
            return
        """Remove all plots with the given name (and optionally legend)."""
        self._state.plots = [
            plot
            for plot in self._state.plots
            if not self.match_plot(plot, name, legend)
        ]

    def match_plot(self, plot, name: str, legend: str | None = None) -> bool:
        """Return True if a plot with given name (and optionally legend) exists."""
        if legend is None and plot.name == name:
            return True
        elif legend is not None and plot.name == name and plot.legend == legend:
            return True
        return False

    ####################################################################################
    # UI Management
    ####################################################################################

    def add_ui_tab(self, name: str, builder: Callable[[], None]) -> None:
        """Adds a user-defined UI tab. The builder function is called every frame to
        build the UI. The name must be unique. If a tab with the same name already
        exists, it will be replaced with the newly created one."""

        self._state.ui.user_tabs[name] = builder

    def add_settings_builder(self, builder: Callable[[], None]) -> None:
        """Adds a user-defined builder appended to the Settings tab."""
        self._state.ui.user_settings_builders.append(builder)

    def set_active_tab(self, name: str) -> None:
        """Selects the given UI tab on the next frame."""
        self._state.ui.active_tab = name

    def remove_ui_tab(self, name: str, error_if_absent: bool = True) -> None:
        """Removes a user-defined UI tab. The name must be unique. If a tab with the
        same name does not exist, it will be ignored. If error_if_absent is True, an
        exception will be raised if the tab does not exist."""

        if name not in self._state.ui.user_tabs:
            if error_if_absent:
                raise ValueError(f"UI tab with name '{name}' does not exist.")
            return
        del self._state.ui.user_tabs[name]

    def _update_ui(self) -> None:
        """Builds the UI. This function is called every frame if non-offscreen."""

        # Setup UI theme.
        # Polyscope versions prior to 2.5.0 retained the modified style across frames.
        # This is no longer the case, so we must reapply the style every frame.
        styling.push_imgui_theme()

        # Retrieve window size.
        width, height = ps.get_window_size()
        self._state.ui.window_width = width
        self._state.ui.window_height = height

        # Build the UI windows.
        build_sidebar_window(self._state)
        build_simulation_controls_window(self._state)
        build_navigation_gizmo(self._state)
        build_logger_window(self._state)

        # Show the Polyscope UI if the user requested it.
        if self._state.ui.show_polyscope_ui:
            ps.build_polyscope_gui()
        if self._state.ui.show_structs_ui:
            ps.build_structure_gui()

        # Cleanup UI theme.
        styling.pop_imgui_theme()

    ####################################################################################
    # Logging Management
    ####################################################################################

    def show_logger_window(self, show: bool = True) -> None:
        """Shows or hides the logger window."""
        self._state.ui.show_logger_window = show

    def is_logger_window_shown(self) -> bool:
        """Returns whether the logger window is currently shown."""
        return self._state.ui.show_logger_window

    def clear_logs(self) -> None:
        """Clears all stored log messages."""
        if self._state.logging.handler is not None:
            self._state.logging.handler.clear()

    ####################################################################################
    # Other operators
    ####################################################################################

    def __enter__(self) -> Viewer:
        """Support with-statement for the renderer."""
        return self

    def __exit__(self, *ignored) -> bool:
        """Support with-statement for the renderer, closing the renderer at the end of
        the statement."""
        self.close()
        return False  # Propagate exceptions.
