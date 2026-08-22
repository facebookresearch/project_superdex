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

import numpy as np
from superdex.physics.utils.coordinate_systems import Axis
from superdex.physics.viewer.backend import polyscope as ps, polyscope_imgui as psim
from superdex.physics.viewer.ui import styling
from superdex.physics.viewer.ui.styling import (
    adjust_color,
    color_to_u32,
    lerp_color,
    replace_color_alpha,
)
from superdex.physics.viewer.viewer_state import ViewerState

########################################################################################
# Navigation Gizmo Constants
########################################################################################

NAVIGATION_SIZE = 100
"""Size of the navigation gizmo window in pixels (width and height)."""
NAVIGATION_PADDING = 3
"""Internal padding around the gizmo content in pixels."""
NAVIGATION_BACKGROUND_COLOR = color_to_u32([0, 0, 0, 0.1])
"""Background color of the circular gizmo (semi-transparent black)."""
NAVIGATION_LINE_THICKNESS = 2.5
"""Thickness of the axis lines in pixels."""
NAVIGATION_LABEL_SIZE = 20
"""Size of the axis label buttons in pixels (width and height)."""
NAVIGATION_LABEL_SCALE = 0.9
"""Font scale factor for axis labels."""
NAVIGATION_LABEL_BORDER = 2
"""Border thickness around axis label buttons in pixels."""
NAVIGATION_AXES = [
    (Axis.POS_X, "X##X"),
    (Axis.POS_Y, "Y##Y"),
    (Axis.POS_Z, "Z##Z"),
    (Axis.NEG_X, "##-X"),  # No visible label
    (Axis.NEG_Y, "##-Y"),  # No visible label
    (Axis.NEG_Z, "##-Z"),  # No visible label
]
"""List of axes to display, with their Axis enum and label text. Ordered as: +X, +Y,
+Z, -X, -Y, -Z for consistent rendering."""
NAVIGATION_TEXT_COLOR = color_to_u32([0, 0, 0])
"""Color of axis label text in normal state (black)."""
NAVIGATION_HOVER_COLOR = color_to_u32([1.0, 1.0, 1.0])
"""Color of the axes when hovered (white)."""
NAVIGATION_HOVER_WEIGHT = 0.5
"""Weight of hover color blending for the axes."""
NAVIGATION_NEG_DARKEN_FACTOR = 0.75
"""Darkening factor for negative axis colors."""
NAVIGATION_MINIMUM_ALPHA = 0.1
"""Minimum alpha value for axis colors, used to attenuate back-facing axes."""

########################################################################################


def build_navigation_gizmo(state: ViewerState) -> None:
    """
    Builds an interactive 3D orientation gizmo showing the current camera view.

    The gizmo displays a circular widget in the upper-right corner of the viewport
    with colored axes (X, Y, Z) extending from the center. Each axis shows both
    positive and negative directions, with:
    - Positive axes: Bright colors with solid lines
    - Negative axes: Darker, semi-transparent colors without lines

    Users can click on any axis label to reorient the camera to look along that
    direction, making it easy to snap to standard orthogonal views.

    Args:
        state: The viewer state containing camera, UI dimensions, and coordinate
               system transform information.
    """

    # Get current camera frame from Polyscope and transform to user coordinate system.
    # The camera view matrix (transposed) gives us the camera's right/up/forward vectors.
    # We transform these from Polyscope's coordinate system to the user's system.
    camera_axes_ps = ps.get_camera_view_matrix().T[:3, :3]
    assert state.coordinate_transform is not None
    target_to_ps = state.coordinate_transform.target_to_source[:3, :3]

    # Compute positive and negative axis directions in user coordinate system
    positive_axes = target_to_ps @ camera_axes_ps
    negative_axes = target_to_ps @ -camera_axes_ps

    # Position gizmo window in the upper-right corner of the viewport.
    # Account for sidebar width and apply standard padding.
    gizmo_x = (
        state.ui.window_width
        - state.ui.sidebar_width
        - NAVIGATION_SIZE
        - styling.WINDOW_DISTANCE_FROM_EDGE
        - styling.BLOCK_SPACING
    )
    gizmo_y = styling.WINDOW_DISTANCE_FROM_EDGE
    gizmo_position = (gizmo_x, gizmo_y)

    # Configure window flags for a minimal, fixed-position overlay.
    # The gizmo is interactive (buttons are clickable) but cannot be moved or resized.
    window_flags = (
        psim.ImGuiWindowFlags_NoTitleBar  # No title bar
        | psim.ImGuiWindowFlags_NoScrollbar  # No scrollbars
        | psim.ImGuiWindowFlags_NoMove  # Cannot be moved
        | psim.ImGuiWindowFlags_NoResize  # Cannot be resized
        | psim.ImGuiWindowFlags_NoCollapse  # Cannot be collapsed
        | psim.ImGuiWindowFlags_NoBackground  # Transparent window background
        | psim.ImGuiWindowFlags_NoNav  # No keyboard navigation
    )

    # Set window position and size
    psim.SetNextWindowPos(gizmo_position, psim.ImGuiCond_Always)
    psim.SetNextWindowSize((NAVIGATION_SIZE, NAVIGATION_SIZE), psim.ImGuiCond_Always)

    if psim.Begin("##NavigationGizmo", True, flags=window_flags):
        # Calculate the center and radius of the gizmo circle.
        gizmo_radius = 0.5 * NAVIGATION_SIZE - NAVIGATION_PADDING

        # Combine positive and negative axes into a single array for processing
        all_axes = np.vstack((positive_axes, negative_axes))  # Shape: (6, 3)

        # Project 3D axes to 2D screen coordinates (X, Y).
        # Flip Y coordinate to match screen space (down is positive).
        axes_2d = all_axes[:, :2] * (1, -1)

        # Sort axes by depth (Z coordinate) to render back-to-front.
        # This ensures closer axes are drawn on top of farther ones.
        depth_sorted_indices = np.argsort(all_axes[:, 2])

        # Calculate screen positions for axis endpoints and label centers.
        gizmo_center_screen = np.asarray(gizmo_position) + 0.5 * NAVIGATION_SIZE
        axis_endpoint_radius = gizmo_radius - NAVIGATION_LABEL_SIZE / 2
        axis_endpoints_screen = gizmo_center_screen + axis_endpoint_radius * axes_2d

        # Get the ImGui draw list for custom drawing (lines, circles, etc.).
        draw_list = psim.GetWindowDrawList()

        # Draw the circular background for the gizmo.
        if psim.IsWindowHovered():
            draw_list.AddCircleFilled(
                gizmo_center_screen, gizmo_radius, NAVIGATION_BACKGROUND_COLOR
            )

        # Configure button styles for axis labels: remove padding, make buttons
        # circular, add border thickness, scale down font size...
        psim.PushStyleVar(psim.ImGuiStyleVar_FramePadding, (0, 0))
        psim.PushStyleVar(psim.ImGuiStyleVar_FrameRounding, NAVIGATION_LABEL_SIZE)
        psim.PushStyleVar(psim.ImGuiStyleVar_FrameBorderSize, NAVIGATION_LABEL_BORDER)
        psim.SetWindowFontScale(NAVIGATION_LABEL_SCALE)

        # Render axes in depth-sorted order (back to front)
        for axis_index in depth_sorted_indices:
            # Determine if this is a positive axis (index 0-2) or negative (index 3-5).
            is_positive_axis = axis_index < 3
            axis_depth = all_axes[axis_index, 2]

            # Get the screen position for this axis endpoint.
            axis_endpoint_screen = axis_endpoints_screen[axis_index]

            # Create an invisible box to detect hover state.
            # This allows us to detect hover before rendering the button.
            psim.SetCursorScreenPos(
                tuple(axis_endpoint_screen - NAVIGATION_LABEL_SIZE / 2)
            )
            psim.Dummy((NAVIGATION_LABEL_SIZE, NAVIGATION_LABEL_SIZE))
            is_axis_hovered = psim.IsItemHovered()

            # Get the axis enum and label text.
            axis_enum, axis_label = NAVIGATION_AXES[axis_index]

            # Configure button colors based on hover state and axis direction.
            alpha = float(
                1.0 + axis_depth + NAVIGATION_MINIMUM_ALPHA * (1.0 - axis_depth)
            )
            label_text_color = color_to_u32(
                replace_color_alpha(NAVIGATION_TEXT_COLOR, alpha)
            )
            axis_color = styling.AXES_COLORS[axis_index % 3]
            axis_color_u32 = color_to_u32(replace_color_alpha(axis_color, alpha))
            border_color_u32 = axis_color_u32
            if not is_positive_axis:
                axis_color_u32 = color_to_u32(
                    adjust_color(axis_color, factor=NAVIGATION_NEG_DARKEN_FACTOR)
                )
            if is_axis_hovered:
                axis_color_u32 = color_to_u32(
                    lerp_color(
                        axis_color_u32,
                        NAVIGATION_HOVER_COLOR,
                        NAVIGATION_HOVER_WEIGHT,
                    )
                )
                border_color_u32 = axis_color_u32

            # Push all button style colors (text, border, and button states).
            psim.PushStyleColor(psim.ImGuiCol_Text, label_text_color)
            psim.PushStyleColor(psim.ImGuiCol_Border, border_color_u32)
            psim.PushStyleColor(psim.ImGuiCol_Button, axis_color_u32)
            psim.PushStyleColor(psim.ImGuiCol_ButtonActive, axis_color_u32)
            psim.PushStyleColor(psim.ImGuiCol_ButtonHovered, axis_color_u32)

            # Draw line from center to endpoint (only for positive axes).
            if is_positive_axis:
                draw_list.AddLine(
                    gizmo_center_screen,
                    axis_endpoint_screen,
                    axis_color_u32,
                    NAVIGATION_LINE_THICKNESS,
                )

            # Position and render the axis label button.
            psim.SetCursorScreenPos(
                tuple(axis_endpoint_screen - NAVIGATION_LABEL_SIZE / 2)
            )

            # Create clickable button for camera reorientation.
            # The button ID includes the label to ensure uniqueness
            if psim.Button(axis_label, (NAVIGATION_LABEL_SIZE, NAVIGATION_LABEL_SIZE)):
                # Reorient camera to look along the selected axis direction
                state.camera.frame_camera_on_next_update = True
                state.camera.frame_camera_direction = -axis_enum.to_vector()
                state.camera.frame_fly_to = True

            # Pop the style colors we pushed for this button.
            psim.PopStyleColor(5)

        # Pop the style variables we pushed for all buttons.
        psim.PopStyleVar(3)

    psim.End()
