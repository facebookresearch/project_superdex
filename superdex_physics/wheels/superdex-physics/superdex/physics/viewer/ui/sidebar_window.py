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

from superdex.physics.viewer.backend import polyscope_imgui as psim
from superdex.physics.viewer.ui import styling
from superdex.physics.viewer.ui.plot_panel import build_plot_panel
from superdex.physics.viewer.ui.scene_panel import build_scene_panel
from superdex.physics.viewer.ui.settings_panel import build_settings_panel
from superdex.physics.viewer.viewer_state import ViewerState

########################################################################################

SIDEBAR_INIT_WIDTH = 525
"""Initial width of the sidebar in pixels."""
SIDEBAR_MIN_WIDTH = 300
"""Minimum width of the sidebar in pixels."""

########################################################################################


def build_sidebar_window(state: ViewerState) -> None:
    flags = (
        psim.ImGuiWindowFlags_NoTitleBar
        | psim.ImGuiWindowFlags_NoMove
        | psim.ImGuiWindowFlags_NoCollapse
    )

    height = state.ui.window_height - 2 * styling.WINDOW_DISTANCE_FROM_EDGE
    psim.SetNextWindowSize(
        (SIDEBAR_INIT_WIDTH, height),
        psim.ImGuiCond_Once,
    )
    psim.SetNextWindowSizeConstraints(
        (SIDEBAR_MIN_WIDTH, height),
        (state.ui.window_width / 2, height),
    )

    if psim.Begin("##SidebarWindow", True, flags):
        state.ui.sidebar_width = int(psim.GetWindowWidth())
        sidebar_coords = (
            state.ui.window_width
            - state.ui.sidebar_width
            - styling.WINDOW_DISTANCE_FROM_EDGE,
            styling.WINDOW_DISTANCE_FROM_EDGE,
        )
        psim.SetWindowPos(sidebar_coords, psim.ImGuiCond_Always)
        if psim.BeginTabBar("##TabBar"):
            psim.PushID("BuiltIn")
            shown, _ = psim.BeginTabItem("Scene", True)
            if shown:
                build_scene_panel(state)
                psim.EndTabItem()
            shown, _ = psim.BeginTabItem("Settings", True)
            if shown:
                build_settings_panel(state)
                psim.EndTabItem()
            shown, _ = psim.BeginTabItem("Plots", True)
            if shown:
                build_plot_panel(state)
                psim.EndTabItem()
            psim.PopID()
            psim.PushID("User")
            for name, builder_fn in state.ui.user_tabs.items():
                flags = 0
                if state.ui.active_tab == name:
                    flags = psim.ImGuiTabItemFlags_SetSelected
                    state.ui.active_tab = None
                shown, _ = psim.BeginTabItem(name, True, flags)
                if shown:
                    builder_fn()
                    psim.EndTabItem()
            psim.PopID()
            psim.EndTabBar()
    psim.End()
