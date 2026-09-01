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
import numpy.typing as npt
from superdex.physics.viewer.backend import POLYSCOPE_AVAILABLE, polyscope_imgui as psim

########################################################################################

Color = npt.NDArray[float]
"""A color represented as a numpy array with RGB or RGBA float values in [0, 1]."""

ColorU32 = int
"""A 32-bit packed color in ABGR format (alpha in highest byte, red in lowest byte)."""

ColorLike = Color | ColorU32
"""A color that can be either a numpy array (RGB/RGBA) or a packed U32 integer (ABGR)."""


def make_color_rgb(r: float, g: float, b: float) -> Color:
    """Create an RGB color from float components in [0, 1].

    Args:
        r: Red component (0.0 to 1.0)
        g: Green component (0.0 to 1.0)
        b: Blue component (0.0 to 1.0)

    Returns:
        RGB color as numpy array [r, g, b]
    """
    return np.array([r, g, b])


def make_color_rgba(r: float, g: float, b: float, a: float = 1.0) -> Color:
    """Create an RGBA color from float components in [0, 1].

    Args:
        r: Red component (0.0 to 1.0)
        g: Green component (0.0 to 1.0)
        b: Blue component (0.0 to 1.0)
        a: Alpha/opacity component (0.0 to 1.0), defaults to 1.0 (fully opaque)

    Returns:
        RGBA color as numpy array [r, g, b, a]
    """
    return np.array([r, g, b, a])


def make_color_rgb_from_u8(r: int, g: int, b: int) -> Color:
    """Create an RGB color from 8-bit integer components.

    Args:
        r: Red component (0 to 255)
        g: Green component (0 to 255)
        b: Blue component (0 to 255)

    Returns:
        RGB color as numpy array with float values [0, 1]
    """
    return np.array([r / 255.0, g / 255.0, b / 255.0])


def make_color_rgba_from_u8(r: int, g: int, b: int, a: int = 255) -> Color:
    """Create an RGBA color from 8-bit integer components.

    Args:
        r: Red component (0 to 255)
        g: Green component (0 to 255)
        b: Blue component (0 to 255)
        a: Alpha/opacity component (0 to 255), defaults to 255 (fully opaque)

    Returns:
        RGBA color as numpy array with float values [0, 1]
    """
    return np.array([r / 255.0, g / 255.0, b / 255.0, a / 255.0])


def color_to_u32(color: ColorLike | list[float]) -> ColorU32:
    """Convert a color to a packed ABGR integer.

    Args:
        color: RGB or RGBA color with float components in [0, 1], or 32-bit packed color

    Returns:
        32-bit packed color in ABGR format (alpha in highest byte, red in lowest)
    """
    if isinstance(color, int):
        return color
    assert len(color) == 3 or len(color) == 4
    r = np.clip(int(color[0] * 255), 0, 255)
    g = np.clip(int(color[1] * 255), 0, 255)
    b = np.clip(int(color[2] * 255), 0, 255)
    a = 255 if len(color) == 3 else np.clip(int(color[3] * 255), 0, 255)
    packed = (a << 24) | (b << 16) | (g << 8) | r  # dtype=np.int32
    packed = int(np.uint32(packed))
    return packed


def u32_to_color(color_u32: ColorU32) -> Color:
    """Convert a packed ABGR integer to an RGBA color.

    Args:
        color_u32: 32-bit packed color in ABGR format

    Returns:
        RGBA color as numpy array with float values [0, 1]
    """
    a = (color_u32 >> 24) & 0xFF
    b = (color_u32 >> 16) & 0xFF
    g = (color_u32 >> 8) & 0xFF
    r = color_u32 & 0xFF
    return make_color_rgba_from_u8(r, g, b, a)


def replace_color_alpha(color: ColorLike, alpha: float) -> ColorLike:
    """Replace the alpha channel of a color.

    Args:
        color: Input color (RGB/RGBA array or ColorU32)
        alpha: New alpha value in [0, 1]

    Returns:
        Color with updated alpha channel (same format as input)
    """
    if isinstance(color, int):
        alpha_u8 = np.clip(int(255 * alpha), 0, 255)
        packed = (alpha_u8 << 24) | (color & 0x00FFFFFF)  # dtype=np.int32
        return int(np.uint32(packed))
    return np.array([*np.asarray(color)[:3], alpha])


def adjust_color(
    color: ColorLike, factor: float = 1.0, offset: float = 0.0
) -> ColorLike:
    """Adjust color brightness and offset.

    Applies the transformation: new_color = color * factor + offset

    Args:
        color: Input color (RGB/RGBA array or ColorU32)
        factor: Multiplicative factor (default 1.0). Values > 1.0 brighten, < 1.0 darken
        offset: Additive offset (default 0.0) applied after multiplication

    Returns:
        Adjusted color (same format as input)
    """
    if isinstance(color, int):
        color = u32_to_color(color)
        color[0:3] = np.clip(color[0:3] * factor + offset, 0.0, 1.0)
        return color_to_u32(color)
    color = np.copy(color)
    color[0:3] = np.clip(color[0:3] * factor + offset, 0.0, 1.0)
    return color


def lerp_color(color_1: ColorLike, color_2: ColorLike, t: float) -> ColorLike:
    """Linear interpolation between two colors.

    Args:
        color_1: First color
        color_2: Second color
        t: Interpolation factor (0.0 = color_1, 1.0 = color_2)

    Returns:
        Interpolated color
    """

    return_as_u32 = isinstance(color_1, int) or isinstance(color_2, int)
    if isinstance(color_1, int):
        color_1 = u32_to_color(color_1)
    if isinstance(color_2, int):
        color_2 = u32_to_color(color_2)
    color = np.asarray(color_1) * (1.0 - t) + np.asarray(color_2) * t
    return color_to_u32(color) if return_as_u32 else color


########################################################################################

# General styling constants.
WINDOW_ROUNDING = 8.0
"""Rounding of window borders."""
WINDOW_DISTANCE_FROM_EDGE = 10.0
"""Distance from edge of screen to place window."""
BLOCK_SPACING = 8.0
"""Space between blocks of UI elements."""

# Common axes coloring constants.
AXES_COLORS = (
    make_color_rgb_from_u8(245, 51, 82),  # X axis.
    make_color_rgb_from_u8(135, 214, 2),  # Y axis.
    make_color_rgb_from_u8(41, 140, 245),  # Z axis.
)

# UI theming constants.
# These dictionaries are only populated when polyscope is available, since they
# reference polyscope_imgui constants that don't exist when polyscope fails to import.
if POLYSCOPE_AVAILABLE:
    THEME_STYLE_VARS = {
        psim.ImGuiStyleVar_WindowRounding: WINDOW_ROUNDING,
    }
    """Style variables to set for the UI theme."""
    THEME_COLORS = {
        psim.ImGuiCol_Text: color_to_u32([0.0, 0.0, 0.0, 1.0]),
        psim.ImGuiCol_TextDisabled: color_to_u32([0.60, 0.60, 0.60, 1.0]),
        psim.ImGuiCol_WindowBg: color_to_u32([0.93, 0.93, 0.93, 1.0]),
        psim.ImGuiCol_ChildBg: color_to_u32([1.0, 1.0, 1.0, 0.25]),
        psim.ImGuiCol_PopupBg: color_to_u32([1.0, 1.0, 1.0, 0.98]),
        psim.ImGuiCol_Border: color_to_u32([0.0, 0.0, 0.0, 0.30]),
        psim.ImGuiCol_BorderShadow: color_to_u32([0.0, 0.0, 0.0, 0.0]),
        psim.ImGuiCol_FrameBg: color_to_u32([1.0, 1.0, 1.0, 1.0]),
        psim.ImGuiCol_FrameBgHovered: color_to_u32([0.25, 0.58, 0.97, 0.40]),
        psim.ImGuiCol_FrameBgActive: color_to_u32([0.25, 0.58, 0.97, 0.67]),
        psim.ImGuiCol_TitleBg: color_to_u32([0.95, 0.95, 0.95, 1.0]),
        psim.ImGuiCol_TitleBgActive: color_to_u32([0.81, 0.81, 0.81, 1.0]),
        psim.ImGuiCol_TitleBgCollapsed: color_to_u32([1.0, 1.0, 1.0, 0.50]),
        psim.ImGuiCol_MenuBarBg: color_to_u32([0.85, 0.85, 0.85, 1.0]),
        psim.ImGuiCol_ScrollbarBg: color_to_u32([0.97, 0.97, 0.97, 0.52]),
        psim.ImGuiCol_ScrollbarGrab: color_to_u32([0.68, 0.68, 0.68, 0.80]),
        psim.ImGuiCol_ScrollbarGrabHovered: color_to_u32([0.48, 0.48, 0.48, 0.80]),
        psim.ImGuiCol_ScrollbarGrabActive: color_to_u32([0.48, 0.48, 0.48, 1.0]),
        psim.ImGuiCol_CheckMark: color_to_u32([0.25, 0.58, 0.98, 1.0]),
        psim.ImGuiCol_SliderGrab: color_to_u32([0.23, 0.51, 0.87, 1.0]),
        psim.ImGuiCol_SliderGrabActive: color_to_u32([0.25, 0.58, 0.98, 1.0]),
        psim.ImGuiCol_Button: color_to_u32([0.25, 0.58, 0.97, 0.40]),
        psim.ImGuiCol_ButtonHovered: color_to_u32([0.25, 0.58, 0.97, 1.0]),
        psim.ImGuiCol_ButtonActive: color_to_u32([0.05, 0.52, 0.97, 1.0]),
        psim.ImGuiCol_Header: color_to_u32([0.25, 0.58, 0.97, 0.31]),
        psim.ImGuiCol_HeaderHovered: color_to_u32([0.25, 0.58, 0.97, 0.80]),
        psim.ImGuiCol_HeaderActive: color_to_u32([0.25, 0.58, 0.97, 1.0]),
        psim.ImGuiCol_Separator: color_to_u32([0.38, 0.38, 0.38, 0.62]),
        psim.ImGuiCol_SeparatorHovered: color_to_u32([0.13, 0.43, 0.80, 0.77]),
        psim.ImGuiCol_SeparatorActive: color_to_u32([0.13, 0.43, 0.80, 1.0]),
        psim.ImGuiCol_ResizeGrip: color_to_u32([0.34, 0.34, 0.34, 0.17]),
        psim.ImGuiCol_ResizeGripHovered: color_to_u32([0.25, 0.58, 0.97, 0.67]),
        psim.ImGuiCol_ResizeGripActive: color_to_u32([0.25, 0.58, 0.97, 0.94]),
        psim.ImGuiCol_Tab: color_to_u32([0.76, 0.79, 0.83, 0.93]),
        psim.ImGuiCol_TabHovered: color_to_u32([0.25, 0.58, 0.97, 0.80]),
        psim.ImGuiCol_TabActive: color_to_u32([0.59, 0.72, 0.88, 1.0]),
        psim.ImGuiCol_TabUnfocused: color_to_u32([0.91, 0.92, 0.93, 0.98]),
        psim.ImGuiCol_TabUnfocusedActive: color_to_u32([0.74, 0.81, 0.91, 1.0]),
        psim.ImGuiCol_PlotLines: color_to_u32([0.38, 0.38, 0.38, 1.0]),
        psim.ImGuiCol_PlotLinesHovered: color_to_u32([1.0, 0.42, 0.34, 1.0]),
        psim.ImGuiCol_PlotHistogram: color_to_u32([0.89, 0.69, 0.0, 1.0]),
        psim.ImGuiCol_PlotHistogramHovered: color_to_u32([1.0, 0.44, 0.0, 1.0]),
        psim.ImGuiCol_TextSelectedBg: color_to_u32([0.25, 0.58, 0.97, 0.34]),
        psim.ImGuiCol_DragDropTarget: color_to_u32([0.25, 0.58, 0.97, 0.94]),
        psim.ImGuiCol_NavHighlight: color_to_u32([0.25, 0.58, 0.97, 0.80]),
        psim.ImGuiCol_NavWindowingHighlight: color_to_u32([0.69, 0.69, 0.69, 0.69]),
        psim.ImGuiCol_NavWindowingDimBg: color_to_u32([0.20, 0.20, 0.20, 0.20]),
        psim.ImGuiCol_ModalWindowDimBg: color_to_u32([0.20, 0.20, 0.20, 0.34]),
    }
    """Colors to set for the UI theme."""
else:
    THEME_STYLE_VARS = {}
    """Style variables to set for the UI theme (empty when polyscope unavailable)."""
    THEME_COLORS = {}
    """Colors to set for the UI theme (empty when polyscope unavailable)."""

########################################################################################


def push_imgui_theme() -> None:
    """Pushes the specified theme onto the stack."""
    for key, value in THEME_STYLE_VARS.items():
        psim.PushStyleVar(key, value)
    for key, value in THEME_COLORS.items():
        psim.PushStyleColor(key, value)


def pop_imgui_theme() -> None:
    """Pops the specified theme from the stack."""
    psim.PopStyleVar(len(THEME_STYLE_VARS))
    psim.PopStyleColor(len(THEME_COLORS))
