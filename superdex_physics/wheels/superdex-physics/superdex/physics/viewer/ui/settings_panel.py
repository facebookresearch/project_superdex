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
from superdex.physics.viewer.ui import widgets
from superdex.physics.viewer.viewer_state import ViewerState

########################################################################################


def build_settings_panel(state: ViewerState) -> None:
    build_camera_settings(state)
    build_rendering_settings(state)
    build_debug_draw_settings(state)
    build_ui_settings(state)
    for builder_fn in state.ui.user_settings_builders:
        builder_fn()


def build_camera_settings(state: ViewerState) -> None:
    # Build camera settings UI.
    psim.TextDisabled("CAMERA SETTINGS")
    # - Use follow camera.
    _, state.camera.use_follow_camera = psim.Checkbox(
        "Use follow camera", state.camera.use_follow_camera
    )
    psim.BeginDisabled(not state.camera.use_follow_camera)
    # - Automatic distance.
    _, state.camera.automatic_distance = psim.Checkbox(
        "Automatic distance", state.camera.automatic_distance
    )
    # - Smoothing.
    _, state.camera.smoothing = psim.SliderFloat(
        "Smoothing", state.camera.smoothing, 0.0, 1.0
    )
    psim.EndDisabled()
    # - Frame All
    if psim.Button("Frame all"):
        state.camera.frame_camera_on_next_update = True
    widgets.vertical_block_spacing()


def build_rendering_settings(state: ViewerState) -> None:
    # Build rendering settings UI.
    psim.TextDisabled("RENDERING")
    # - Choose visual (.glb) mesh vs physics (collision) mesh.
    changed, state.show_visual_mesh = psim.Checkbox(
        "Show visual mesh", state.show_visual_mesh
    )
    widgets.tooltip_on_hover(
        "Draw registered visual meshes (.glb) instead of the collision mesh. "
        "Only affects actors that ship a visual render model (e.g. bots)."
    )
    if changed:
        state.render_models_dirty = True
    widgets.vertical_block_spacing()


def build_debug_draw_settings(state: ViewerState) -> None:
    scene_instance = state.scene.instance
    if scene_instance is None:
        return

    # Build debug draw settings UI.
    psim.TextDisabled("DEBUG DRAW")

    # - Enable debug draw.
    debug_draw = scene_instance.get_debug_draw()
    is_enabled = debug_draw.is_enabled()
    changed, is_enabled = psim.Checkbox("Enable debug draw", is_enabled)
    if changed:
        debug_draw.enable(is_enabled)
        state.debug_requires_update = True

    # - Debug features.
    if psim.CollapsingHeader("Features"):
        num_features = debug_draw.get_num_features()
        for i in range(num_features):
            feature_name = debug_draw.get_feature_name(i)
            feature_desc = debug_draw.get_feature_description(i)
            feature_is_enabled = debug_draw.is_feature_enabled(i)
            changed, feature_is_enabled = psim.Checkbox(
                feature_name, feature_is_enabled
            )
            widgets.tooltip_on_hover(feature_desc)
            if changed:
                debug_draw.enable_feature(i, feature_is_enabled)
                state.debug_requires_update = True

    # - Debug rendering.
    if psim.CollapsingHeader("Rendering"):
        debug_draw_renderer = state.helpers.debug_draw
        if debug_draw_renderer is not None:
            widgets.scalar_property(
                "Line Radius",
                debug_draw_renderer.get_line_radius,
                debug_draw_renderer.set_line_radius,
                format="%f",
            )
            widgets.slider_property(
                "Sphere Radius Scale",
                debug_draw_renderer.get_sphere_radius_scale,
                debug_draw_renderer.set_sphere_radius_scale,
                v_min=0.0,
                v_max=5.0,
            )
    widgets.vertical_block_spacing()


def build_ui_settings(state: ViewerState) -> None:
    psim.TextDisabled("UI SETTINGS")
    # Build polyscope settings UI.
    _, state.ui.show_polyscope_ui = psim.Checkbox(
        "Show Polyscope UI", state.ui.show_polyscope_ui
    )
    _, state.ui.show_structs_ui = psim.Checkbox(
        "Show Structures UI", state.ui.show_structs_ui
    )
