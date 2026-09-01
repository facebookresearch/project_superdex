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
from superdex.physics.viewer.viewer_state import ViewerState

########################################################################################

CONTROLS_WIDTH = 260
"""Width of the simulation controls window in pixels."""

CONTROLS_HEIGHT = 40
"""Height of the simulation controls window in pixels."""

########################################################################################


def build_simulation_controls_window(state: ViewerState) -> None:
    """Builds the simulation controls window. This window will appear on the top of the
    screen and will contain buttons to control the simulation."""

    # Setup window properties.
    # The window cannot be resized, moved, collapsed. Also appears with no title bar.
    flags = (
        psim.ImGuiWindowFlags_NoResize
        | psim.ImGuiWindowFlags_NoTitleBar
        | psim.ImGuiWindowFlags_NoMove
        | psim.ImGuiWindowFlags_NoCollapse
    )

    window_size = (CONTROLS_WIDTH, CONTROLS_HEIGHT)
    window_coords = (
        (state.ui.window_width - CONTROLS_WIDTH) / 2,
        styling.WINDOW_DISTANCE_FROM_EDGE,
    )
    psim.SetNextWindowSize(window_size, psim.ImGuiCond_Always)
    psim.SetNextWindowPos(window_coords, psim.ImGuiCond_Always)

    # Build the window.
    if psim.Begin("##SimControl", True, flags):
        width, height = psim.GetContentRegionAvail()
        button_size = (width / 3 - 4.0, height)

        # Resume/Pause button.
        toggle_state = psim.Button("Resume" if state.paused else "Pause", button_size)

        # Step once button.
        psim.SameLine()
        step_once = psim.Button("Step", button_size)

        # Reset button.
        psim.SameLine()
        reset_pressed = psim.Button("Reset", button_size)

        # Update the simulation state.
        if toggle_state:
            state.paused = not state.paused
        if step_once:
            state.step_once = True
        if reset_pressed:
            state.reset_requested = True
    psim.End()
