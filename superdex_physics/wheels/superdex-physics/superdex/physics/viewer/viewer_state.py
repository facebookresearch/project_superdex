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

import dataclasses
from typing import Callable, List, Tuple, Union

import numpy.typing as npt
from superdex.physics import Actor, ActorHandle, Scene, SceneHandle, StateHandle
from superdex.physics.utils.coordinate_systems import (
    CoordinateSystem,
    CoordinateTransform,
)
from superdex.physics.viewer.backend import polyscope as ps
from superdex.physics.viewer.logging_handler import LoggingHandler, LogLevel
from superdex.physics.viewer.renderers.actor_renderer import ActorRenderer
from superdex.physics.viewer.renderers.curve_network_renderer import (
    CurveNetworkRenderer,
)
from superdex.physics.viewer.renderers.debug_draw_renderer import DebugDrawRenderer
from superdex.physics.viewer.renderers.grid_renderer import GridRenderer
from superdex.physics.viewer.renderers.mesh_renderer import MeshRenderer
from superdex.physics.viewer.renderers.point_cloud_renderer import PointCloudRenderer
from superdex.physics.viewer.renderers.static_plane_renderer import StaticPlaneRenderer
from superdex.physics.viewer.utils.aabb import AABB

########################################################################################


@dataclasses.dataclass
class ActorState:
    """Structure holding the state of an actor in the scene."""

    handle: ActorHandle
    """Unique handle of the actor."""
    instance: Actor
    """Pointer to the actor."""
    renderer: Union[ActorRenderer, StaticPlaneRenderer, None] = None
    """Associated renderer, if any."""


@dataclasses.dataclass
class CameraState:
    use_follow_camera: bool = False
    """If True, the camera will follow the scene actors."""
    frame_camera_on_next_update: bool = False
    """If True, the camera will frame the scene on the next frame."""
    frame_camera_direction: npt.NDArray[float] | None = None
    """Direction of the camera when framing the scene. Used if
    frame_camera_on_next_update is True."""
    frame_fly_to: bool = False
    """If True, the camera will fly to the target position."""
    automatic_distance: bool = False
    """If True, the camera will automatically compute the distance framing the scene."""
    smoothing: float = 0.4
    """Smoothing factor for the camera movement."""
    smoothed_target_position: npt.NDArray[float] | None = None
    """Smoothed target position for the camera."""
    distance: float = 0.0
    """Last view distance for the camera when following the actors."""
    smoothed_distance: float = 0.0
    """Smoothed view distance for the camera when following the actors."""


@dataclasses.dataclass
class SceneState:
    """Structure holding the state of the scene."""

    instance: Scene | None = None
    """Pointer to the SuperDex Physics scene."""
    handle: SceneHandle | None = None
    """Handle to the SuperDex Physics scene."""
    actors: dict[ActorHandle, ActorState] = dataclasses.field(default_factory=dict)
    """Actors present in the scene, keyed by unique actor handle."""
    aabb: AABB = dataclasses.field(default_factory=AABB.empty)
    """Bounding box of the scene."""
    selected_actor: ActorHandle | None = None
    """Handle of currently selected actor."""
    excluded_actors: set[str] = dataclasses.field(default_factory=set)
    """Actors that should be excluded from the scene."""
    initial_state: StateHandle | None = None
    """Initial state snapshot captured when scene is set, used for reset."""


@dataclasses.dataclass
class HelpersState:
    """Structure holding the state of the helpers."""

    group: ps.Group | None = None
    """Group holding the helpers."""
    grids: dict[str, GridRenderer] = dataclasses.field(default_factory=dict)
    """Grids present in the scene."""
    meshes: dict[str, MeshRenderer] = dataclasses.field(default_factory=dict)
    """Helper meshes present in the scene."""
    point_clouds: dict[str, PointCloudRenderer] = dataclasses.field(
        default_factory=dict
    )
    """Point clouds present in the scene."""
    curve_networks: dict[str, CurveNetworkRenderer] = dataclasses.field(
        default_factory=dict
    )
    """Curve networks present in the scene."""
    debug_draw: DebugDrawRenderer | None = None
    """Renderer for debug drawing."""


@dataclasses.dataclass
class UiState:
    """Structure holding the state of the UI."""

    window_width: int = 0
    """Width of the window."""
    window_height: int = 0
    """Height of the window."""
    sidebar_width: int = 0
    """Width of the sidebar."""
    logger_height: int = 0
    """Height of the logger window."""
    show_logger_window: bool = True
    """If True, the logger window will be shown."""
    show_polyscope_ui: bool = False
    """If True, the built-in Polyscope UI will be shown."""
    show_structs_ui: bool = False
    """If True, the built-in Polyscope structured UI will be shown."""
    user_tabs: dict[str, Callable[[], None]] = dataclasses.field(default_factory=dict)
    """User-defined UI tabs."""
    user_settings_builders: list[Callable[[], None]] = dataclasses.field(
        default_factory=list
    )
    """User-defined builders appended to the Settings tab."""
    active_tab: str | None = None
    """Tab to select on the next frame (consumed after use)."""


@dataclasses.dataclass
class PlotAxisInfo:
    """Structure holding the axis information of a plot."""

    limit: Tuple[float, float] | None = None
    """Range of data to show."""

    name: str | None = None
    """Name and unit of the data along this axis"""


@dataclasses.dataclass
class PlotState:
    """Structure holding the state of a plot."""

    name: str | None = None
    """Name of the plot."""

    legend: str | None = None
    """If this is a grouped plot, this is the legend for this plot."""

    x: npt.NDArray[float] | None = None
    """Data for x-axis."""

    y: npt.NDArray[float] | None = None
    """Data for y-axis."""

    lower: npt.NDArray[float] | None = None
    """Data for y-axis shaded plot lower bound."""

    upper: npt.NDArray[float] | None = None
    """Data for y-axis shaded plot upper bound."""

    shaded_legend_name: str | None = None
    """Custom name for the shaded plot."""

    x_axis_info: PlotAxisInfo | None = None
    """Information of the x-axis data."""

    y_axis_info: PlotAxisInfo | None = None
    """Information of the y-axis data."""


@dataclasses.dataclass
class LoggingState:
    """Structure holding the state of the logging system."""

    handler: LoggingHandler | None = None
    """The logging handler that captures log messages."""
    filter_level: LogLevel = LogLevel.ALL
    """Current filter level for displaying logs."""
    auto_scroll: bool = True
    """If True, the log window will auto-scroll to show newest messages."""
    word_wrap: bool = False
    """If True, the log messages will be word-wrapped."""
    search_text: str = ""
    """Text to search for in the logs."""


@dataclasses.dataclass
class ViewerState:
    """Structure holding the state of the viewer. This state object is passed to the
    different UI objects to allow them to access and modify the state of the viewer."""

    # General state.
    paused: bool = False
    """Whether the simulation is paused."""
    step_once: bool = False
    """If True, the simulation must step once and then pause."""
    reset_requested: bool = False
    """If True, the simulation reset has been requested via the UI."""
    requires_update: bool = False
    """If True, the renderer must update the scene actors."""
    debug_requires_update: bool = False
    """If True, the renderer must update the debug draw."""
    coordinate_system: CoordinateSystem | None = None
    """Coordinate system to use for visualization."""
    coordinate_transform: CoordinateTransform | None = None
    """Transform converting between the coordinate system and Polyscope."""
    offscreen: bool = False
    """Whether the rendering should occur offscreen."""
    close_requested: bool = False
    """If True, the user requested to close the viewer."""

    # Scene state.
    scene: SceneState = dataclasses.field(default_factory=SceneState)
    """State of the scene."""
    helpers: HelpersState = dataclasses.field(default_factory=HelpersState)
    """State of the helpers."""
    camera: CameraState = dataclasses.field(default_factory=CameraState)
    """State of the camera."""

    # UI state.
    ui: UiState = dataclasses.field(default_factory=UiState)
    """State of the UI."""
    logging: LoggingState = dataclasses.field(default_factory=LoggingState)
    """State of the logging system."""

    # Plot state.
    plots: List[PlotState] | None = None
    """State of the plotting system."""

    # Other visualization state.
    use_com_transform: bool = True
    """If True, all relevant components (transform inspector, frame of reference axes)
    will use the CoM transform instead of the root transform. Valid only for dynamic
    rigid actors."""

    show_visual_mesh: bool = True
    """If True, actors with a registered visual render model (.glb) are drawn with that
    mesh; if False, they fall back to their physics (collision) surface mesh. Only
    affects actors that have a registered render model (e.g. bots)."""

    render_models_dirty: bool = False
    """If True, renderers for actors with a registered visual render model must be
    rebuilt to reflect a change in show_visual_mesh."""
