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

import numpy as np
from superdex.physics.viewer.backend import polyscope_imgui as psim
from superdex.physics.viewer.logging_handler import LogLevel
from superdex.physics.viewer.ui import styling
from superdex.physics.viewer.ui.navigation_gizmo import NAVIGATION_SIZE
from superdex.physics.viewer.viewer_state import ViewerState

########################################################################################

LOGGER_WINDOW_MIN_HEIGHT = 80
"""Minimum height of the logger window in pixels."""
LOGGER_FONT_SCALE = 0.9
"""Font scale multiplier for logger text. Reduces font size for more compact display."""
LOGGER_TIMESTAMP_WIDTH = 80
"""Width in pixels for the timestamp column."""
LOGGER_LEVEL_WIDTH = 90
"""Width in pixels for the log level column."""
LOGGER_LOCATION_WIDTH = 180
"""Width in pixels for the file location column (filename:lineno)."""
LOGGER_FOOTER_HEIGHT = 20
"""Height in pixels for the footer (message count)."""
LOGGER_FOOTER_FONT_SCALE = 0.75
"""Font scale multiplier for the footer text."""
LOGGER_COLORS = {
    LogLevel.CRITICAL: (0.64, 0.28, 0.64, 1.0),
    LogLevel.ERROR: (0.93, 0.11, 0.14, 1.0),
    LogLevel.WARNING: (1.0, 0.50, 0.15, 1.0),
    LogLevel.INFO: (0.44, 0.57, 0.75, 1.0),
}
"""Color mapping for log levels in RGBA format (red, green, blue, alpha)."""
LOGGER_DEFAULT_COLOR = (0.63, 0.63, 0.63, 1.0)
"""Default text color for log levels not explicitly defined (e.g., DEBUG)."""

########################################################################################


def build_logger_window(state: ViewerState) -> None:  # noqa: C901
    """
    Build and render the logger window UI.

    This window displays log messages captured by the logging handler in a scrollable
    list with filtering and control options. It is designed to be attached to the
    SuperDex Physics Viewer UI and provide a convenient way to view and manage logs.

    Args:
        state: The viewer state containing logging configuration and handler.
    """

    # Window layout:
    # ┌─────────────────────────────────────────────────────────────────────────────┐
    # │ Logger                                                                      │
    # ├─────────────────────────────────────────────────────────────────────────────┤
    # │ [Clear] [Filters] [Search text] [Auto-scroll] [Word-wrap] <- Header Controls│
    # ├───────────┬────────────┬─────────────────────────────────┬──────────────────┤
    # │ Timestamp │ Level      │ Message                         │ Location         │
    # ├───────────┼────────────┼─────────────────────────────────┼──────────────────┤
    # │ 12:34:56  │ INFO       │ Application started             │ main.py:42       │
    # │ 12:34:57  │ WARNING    │ Config file not found           │ config.py:15     │
    # │ 12:34:58  │ ERROR      │ Connection failed               │ network.py:89    │
    # │    ...    │    ...     │          ...                    │      ...         │
    # │           │            │                                 │                  │
    # │           │            │   (Scrollable Area)             │                  │
    # │           │            │                                 │                  │
    # ├───────────┴────────────┴─────────────────────────────────┴──────────────────┤
    # │                                                           3/10 <- Footer    │
    # └─────────────────────────────────────────────────────────────────────────────┘

    # Calculate available window dimensions.
    # This take into consideration the sidebar and navigation gizmo.
    # I wish Polyscope was on the Docking branch of ImGui so we could simply rely on
    # that for layout and positioning :/
    available_width = (
        state.ui.window_width
        - state.ui.sidebar_width
        - styling.BLOCK_SPACING
        - 2 * styling.WINDOW_DISTANCE_FROM_EDGE
    )
    available_height = (
        state.ui.window_height
        - 2 * styling.WINDOW_DISTANCE_FROM_EDGE
        - styling.BLOCK_SPACING
        - NAVIGATION_SIZE
    )

    # Configure window position and size constraints.
    psim.SetNextWindowCollapsed(True, psim.ImGuiCond_Once)
    psim.SetNextWindowSizeConstraints(
        (available_width, LOGGER_WINDOW_MIN_HEIGHT),
        (available_width, available_height),
    )
    psim.SetNextWindowPos(
        (
            styling.WINDOW_DISTANCE_FROM_EDGE,
            state.ui.window_height
            - state.ui.logger_height
            - styling.WINDOW_DISTANCE_FROM_EDGE,
        ),
        psim.ImGuiCond_Always,
    )

    # Begin window and track its state.
    psim.Begin("Logger", True)
    state.ui.show_logger_window = not psim.IsWindowCollapsed()
    _, logger_height = psim.GetWindowSize()
    state.ui.logger_height = int(logger_height)
    if not state.ui.show_logger_window:
        psim.End()
        return

    # Header Controls
    # Clear button removes all messages from the buffer
    if psim.Button("Clear"):
        if state.logging.handler is not None:
            state.logging.handler.clear()

    # Filters button opens log level selection popup
    psim.SameLine()
    if psim.Button("Filters"):
        psim.OpenPopup("FiltersPopup")

    if psim.BeginPopup("FiltersPopup"):
        # Display checkboxes for each individual log level
        # (excluding NONE and ALL which are handled by quick-select buttons).
        log_levels = (
            LogLevel.DEBUG,
            LogLevel.INFO,
            LogLevel.WARNING,
            LogLevel.ERROR,
            LogLevel.CRITICAL,
        )

        for level in log_levels:
            is_enabled = bool(state.logging.filter_level & level)
            level_name = level.name if level.name is not None else str(level)
            changed, new_value = psim.Checkbox(level_name, is_enabled)
            if changed:
                if new_value:
                    state.logging.filter_level |= level
                else:
                    state.logging.filter_level &= ~level

        psim.Separator()

        # Quick-select buttons for convenience.
        if psim.Button("All"):
            state.logging.filter_level = LogLevel.ALL
        psim.SameLine()
        if psim.Button("None"):
            state.logging.filter_level = LogLevel.NONE

        psim.EndPopup()

    # Search text input
    psim.SameLine()
    changed, new_search_text = psim.InputText("Search", state.logging.search_text)
    if changed:
        state.logging.search_text = new_search_text

    # Auto-scroll checkbox
    psim.SameLine()
    changed, new_auto_scroll = psim.Checkbox("Auto-scroll", state.logging.auto_scroll)
    if changed:
        state.logging.auto_scroll = new_auto_scroll

    # Word-wrap checkbox
    psim.SameLine()
    changed, new_word_wrap = psim.Checkbox("Word-wrap", state.logging.word_wrap)
    if changed:
        state.logging.word_wrap = new_word_wrap

    # Create a scrollable child window for the message list.
    # Height of -LOGGER_FOOTER_HEIGHT reserves space for the footer.
    displayed_count = 0
    total_count = 0

    if psim.BeginChild("##LogMessages", (0, -LOGGER_FOOTER_HEIGHT), True):
        handler = state.logging.handler
        if handler is not None:
            # Check for new messages before retrieving (retrieval clears the flag).
            has_new_messages = handler.has_new_messages
            messages = handler.messages
        else:
            has_new_messages = False
            messages = []
        total_count = len(messages)

        # Setup 4-column layout: [Timestamp | Level | Message | Location].
        available_width, _ = psim.GetContentRegionAvail()
        message_col_width = (
            available_width
            - LOGGER_TIMESTAMP_WIDTH
            - LOGGER_LEVEL_WIDTH
            - LOGGER_LOCATION_WIDTH
        )

        psim.Columns(4, border=False)
        psim.SetColumnWidth(0, LOGGER_TIMESTAMP_WIDTH)
        psim.SetColumnWidth(1, LOGGER_LEVEL_WIDTH)
        psim.SetColumnWidth(2, message_col_width)
        psim.SetColumnWidth(3, LOGGER_LOCATION_WIDTH)
        psim.SetWindowFontScale(LOGGER_FONT_SCALE)

        # Determine function for displaying messages.
        message_text_func = psim.TextWrapped if state.logging.word_wrap else psim.Text

        # Display filtered messages
        for msg in messages:
            if not (state.logging.filter_level & msg.level):
                continue
            if state.logging.search_text not in msg.message:
                continue

            # Get color for log level.
            color = LOGGER_COLORS.get(msg.level, LOGGER_DEFAULT_COLOR)

            # Render message columns.
            psim.TextDisabled(msg.timestamp)
            psim.NextColumn()
            psim.TextColored(color, msg.level.name)
            psim.NextColumn()
            message_text_func(msg.message)
            psim.NextColumn()
            psim.TextDisabled(f"{msg.filename}:{msg.lineno}")
            psim.NextColumn()

            # Show detailed location on hover.
            if psim.IsItemHovered():
                psim.BeginTooltip()
                psim.Text(f"[{msg.module}] {msg.pathname}:{msg.lineno}")
                psim.EndTooltip()

            displayed_count += 1

        # Auto-scroll to bottom when new messages arrive.
        if has_new_messages and state.logging.auto_scroll:
            psim.SetScrollHereY(1.0)

    psim.SetWindowFontScale(1.0)
    psim.EndChild()

    # Display message count (right-aligned)
    psim.SetWindowFontScale(LOGGER_FOOTER_FONT_SCALE)
    count_text = f"{displayed_count}/{total_count}"
    content_width, _ = psim.GetContentRegionAvail()
    text_width, _ = psim.CalcTextSize(count_text)
    psim.SetCursorPosX(np.maximum(0, content_width - text_width))
    psim.Text(count_text)
    psim.SetWindowFontScale(1.0)

    psim.End()
