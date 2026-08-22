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

import json

import numpy as np
import numpy.typing as npt
from superdex.physics import Actor, ActorType, DynamicArrayReal, DynamicArrayReal2
from superdex.physics.utils.transformations import transformrt_to_numpy
from superdex.physics.viewer.backend import polyscope_imgui as psim
from superdex.physics.viewer.renderers.mesh_renderer import MeshRenderer
from superdex.physics.viewer.ui import widgets
from superdex.physics.viewer.viewer_state import ActorState, ViewerState

########################################################################################

SPLIT_RATIO = 0.33
"""Amount of space dedicated to the scene inspector."""
TYPE_COL_SIZE = 100.0
"""Width of the column containing the actor type."""
VISIBILITY_COL_SIZE = 20.0
"""Width of the column containing the actor visibility controls."""

########################################################################################


def _get_articulated_pose(actor: Actor) -> npt.NDArray[float]:
    """Gets the pose of an articulated actor."""
    buffer = DynamicArrayReal(actor.get_num_dofs())
    actor.get_articulated_pose(buffer)
    return np.array(buffer)


def _get_articulated_joint_velocities(actor: Actor) -> npt.NDArray[float]:
    """Gets the velocity of an articulated actor."""
    buffer = DynamicArrayReal(actor.get_num_dofs())
    actor.get_articulated_joint_velocities(buffer)
    return np.array(buffer)


def _get_articulated_dof_limits(actor: Actor) -> npt.NDArray[float]:
    """Gets the joint limits of an articulated actor."""
    buffer = DynamicArrayReal2(actor.get_num_dofs())
    actor.get_articulated_dof_limits(buffer)
    return np.array(buffer)


def get_type_display_name(actor_type: ActorType, is_static: bool) -> str:
    """Returns the display name of the given actor type."""

    type_name = actor_type.name
    if is_static:
        type_name += " (static)"
    return type_name


def build_scene_panel(state: ViewerState) -> None:
    """
    Constructs the scene panel UI, which includes the actor list and properties.
    """

    _, avail_height = psim.GetContentRegionAvail()

    # Build actor list UI.
    psim.TextDisabled("SCENE ACTORS")
    if psim.BeginChild("##ActorList", (0, SPLIT_RATIO * avail_height)):
        build_actor_list(state)
    psim.EndChild()
    widgets.vertical_block_spacing()

    # Build actor properties UI.
    psim.TextDisabled("ACTOR PROPERTIES")
    if psim.BeginChild("##ActorProperties", flags=psim.ImGuiWindowFlags_NoBackground):
        build_actor_properties(state)
    psim.EndChild()


def build_actor_list(state: ViewerState) -> None:  # noqa: C901
    """
    Constructs the actor list UI, displaying each actor's name, type, and visibility
    controls. The user can select an actor to inspect its properties.
    """

    # Setup the list frame.
    # psim.BeginChildFrame(psim.GetID("SceneInspector"), (0, 0))
    psim.BeginChild("##SceneInspector", border=True, size=(0, 0))
    psim.PushStyleVar(psim.ImGuiStyleVar_FramePadding, (0, 0))
    psim.PushStyleVar(psim.ImGuiStyleVar_ItemSpacing, (0, 4))
    region_min = psim.GetWindowContentRegionMin()
    region_max = psim.GetWindowContentRegionMax()
    width = region_max[0] - region_min[0]
    psim.Columns(6, border=False)
    psim.SetColumnWidth(0, width - TYPE_COL_SIZE - 4 * VISIBILITY_COL_SIZE)
    psim.SetColumnWidth(1, TYPE_COL_SIZE)
    psim.SetColumnWidth(2, VISIBILITY_COL_SIZE)
    psim.SetColumnWidth(3, VISIBILITY_COL_SIZE)
    psim.SetColumnWidth(4, VISIBILITY_COL_SIZE)
    psim.SetColumnWidth(5, VISIBILITY_COL_SIZE)

    # Draw header row.
    header_row_height = psim.GetCursorPosY()
    psim.Text("Name")
    psim.NextColumn()
    psim.Text("Type")
    psim.NextColumn()
    psim.NextColumn()
    psim.NextColumn()
    psim.NextColumn()
    psim.NextColumn()
    psim.Separator()

    # Draw the list of actors
    num_surfaces_shown = 0
    num_edges_shown = 0
    num_nodes_shown = 0
    num_axes_shown = 0
    num_renderers = 0
    for actor in state.scene.actors.values():
        # Actor name (selectable).
        actor_name = actor.instance.get_name()
        # Mask actor handle to lower 32 bits since ImGUI PushID expects a 32-bit integer
        psim.PushID(actor.handle.value & 0xFFFFFFFF)
        is_selected = actor.handle == state.scene.selected_actor
        selected_changed, is_selected = psim.Selectable(actor_name, is_selected)
        psim.NextColumn()
        if selected_changed:
            state.scene.selected_actor = actor.handle if is_selected else None

        # Actor type.
        is_static = actor.instance.is_static()
        psim.Text(get_type_display_name(actor.instance.get_type(), is_static))
        psim.NextColumn()

        # Visibility controls (if actor is renderable).
        renderer = actor.renderer
        if renderer is None:
            psim.NextColumn()
            psim.NextColumn()
            psim.NextColumn()
            psim.NextColumn()
            psim.PopID()
            continue
        show_surface = renderer.is_surface_enabled()
        show_edges = renderer.are_edges_enabled()
        show_nodes = renderer.are_nodes_enabled()
        show_axes = renderer.are_axes_enabled()
        show_surface_changed, show_surface = psim.Checkbox("##S", show_surface)
        psim.NextColumn()
        show_edges_changed, show_edges = psim.Checkbox("##E", show_edges)
        psim.NextColumn()
        show_nodes_changed, show_nodes = psim.Checkbox("##N", show_nodes)
        psim.NextColumn()
        show_axes_changed, show_axes = psim.Checkbox("##A", show_axes)
        psim.NextColumn()

        # If visibility controls were changed, update the actor renderer.
        if show_surface_changed:
            renderer.set_enable_surface(show_surface)
        if show_edges_changed:
            renderer.set_enable_edges(show_edges)
        if show_nodes_changed:
            renderer.set_enable_nodes(show_nodes)
        if show_axes_changed:
            renderer.set_enable_axes(show_axes)
        num_surfaces_shown += show_surface
        num_edges_shown += show_edges
        num_nodes_shown += show_nodes
        num_axes_shown += show_axes
        num_renderers += 1
        psim.PopID()

    # Draw the all-actors visibility controls. This allows the user to quickly
    # toggle certain visibility features for all the actors at once.
    def determine_checkbox_state(num_enabled: int) -> int:
        # 0: all disabled, 1: some enabled, 2: all enabled
        return 2 * int(num_enabled == num_renderers) + int(num_enabled > 0)

    all_surface_state = determine_checkbox_state(num_surfaces_shown)
    all_edges_state = determine_checkbox_state(num_edges_shown)
    all_nodes_state = determine_checkbox_state(num_nodes_shown)
    all_axes_state = determine_checkbox_state(num_axes_shown)

    psim.NextColumn()
    psim.NextColumn()
    psim.SetCursorPosY(header_row_height)
    toggle_surfaces, _ = psim.CheckboxFlags("##AS", all_surface_state, 3)
    psim.NextColumn()
    psim.SetCursorPosY(header_row_height)
    toggle_edges, _ = psim.CheckboxFlags("##AE", all_edges_state, 3)
    psim.NextColumn()
    psim.SetCursorPosY(header_row_height)
    toggle_nodes, _ = psim.CheckboxFlags("##AN", all_nodes_state, 3)
    psim.NextColumn()
    psim.SetCursorPosY(header_row_height)
    toggle_axes, _ = psim.CheckboxFlags("##AA", all_axes_state, 3)
    psim.NextColumn()

    # If the all-actors visibility controls were changed, update actor renderers.
    if any((toggle_surfaces, toggle_edges, toggle_nodes, toggle_axes)):
        for actor in state.scene.actors.values():
            renderer = actor.renderer
            if renderer is None:
                continue
            if toggle_surfaces:
                renderer.set_enable_surface(all_surface_state != 3)
            if toggle_edges:
                renderer.set_enable_edges(all_edges_state != 3)
            if toggle_nodes:
                renderer.set_enable_nodes(all_nodes_state != 3)
            if toggle_axes:
                renderer.set_enable_axes(all_axes_state != 3)
    psim.PopStyleVar(2)
    # psim.EndChildFrame()
    psim.EndChild()


def build_actor_properties(state: ViewerState) -> None:
    """
    Constructs the actor properties UI, displaying properties of the selected actor.
    """

    # Retrieve the selected actor. If there is no selected actor, do nothing.
    selected_handle = state.scene.selected_actor
    actor = None
    if selected_handle is not None:
        actor = state.scene.actors.get(selected_handle, None)
    if actor is None:
        psim.Text("Select an actor to inspect its properties.")
        return

    # Build UI for common actor properties.
    flags = psim.ImGuiInputTextFlags_ReadOnly
    is_static = actor.instance.is_static()
    actor_type = actor.instance.get_type()
    psim.InputText("Name", actor.instance.get_name(), flags)
    # Mask actor handle to lower 32 bits since ImGUI expects a 32-bit integer
    psim.InputText("Handle", f"0x{actor.handle.value:016x}", flags=flags)
    psim.InputText("Type", get_type_display_name(actor_type, is_static), flags)

    # Build specific UI for the selected actor type.
    if actor_type == ActorType.RIGID:
        build_rigid_actor_inspector(state, actor)
    elif actor_type == ActorType.SOFT:
        build_soft_actor_inspector(state, actor)
    elif actor_type == ActorType.ARTICULATED:
        build_articulated_actor_inspector(state, actor)

    # Build renderer UI if it has an associated renderer.
    if actor.renderer is not None and isinstance(actor.renderer, MeshRenderer):
        build_mesh_renderer_inspector(actor.renderer)


def build_rigid_actor_inspector(state: ViewerState, actor: ActorState) -> None:
    """
    Constructs the UI for inspecting properties of a rigid actor.
    """

    psim.PushID("RigidActorInspector")
    # TODO: Show generalities of the rigid body. E.g. density, mass, inertia, etc.
    # Build transform UI.
    if psim.CollapsingHeader("Transform"):
        # If the rigid is not static, allow the user to toggle between root and CoM
        # transforms.
        is_static = actor.instance.is_static()
        if not is_static:
            if psim.RadioButton("Root Transform", not state.use_com_transform):
                state.use_com_transform = False
                state.requires_update = True
            psim.SameLine()
            if psim.RadioButton("CoM Transform", state.use_com_transform):
                state.use_com_transform = True
                state.requires_update = True

        # Retrieve the correct transform.
        translation, rotation = transformrt_to_numpy(
            actor.instance.get_center_of_mass_transform()
            if not is_static and state.use_com_transform
            else actor.instance.get_root_transform()
        )

        build_transform_widget(translation, rotation)
    psim.PopID()


def build_soft_actor_inspector(state: ViewerState, actor: ActorState) -> None:
    """
    Constructs the UI for inspecting properties of a soft actor.
    """

    psim.PushID("SoftActorInspector")
    if psim.CollapsingHeader("Transform"):
        psim.TextDisabled("Note: Soft actor transforms are approximate.")
        translation, rotation = transformrt_to_numpy(
            actor.instance.get_root_transform()
        )
        build_transform_widget(translation, rotation)
    psim.PopID()


def build_articulated_actor_inspector(  # noqa: C901
    state: ViewerState, actor: ActorState
) -> None:
    """
    Constructs the UI for inspecting properties of an articulated actor.
    """

    psim.PushID("ArticulatedActorInspector")

    # Show generalities of the articulated body.
    flags = psim.ImGuiInputTextFlags_ReadOnly
    shape_info = actor.instance.get_articulated_shape_info()
    num_dofs = actor.instance.get_num_dofs()
    num_joints = len(shape_info.joint_types)
    psim.InputInt("Nb. of DoFs", num_dofs, flags=flags)
    psim.InputInt("Nb. of Joints", num_joints, flags=flags)

    # Retrieve agent joint names and dofs.
    joint_names = list(shape_info.joint_names)
    dof_info = list(shape_info.dof_info)

    # Helper to build the pose and velocity inspectors.
    def iterate_all_dofs():
        for name, info in zip(joint_names, dof_info):
            offset, size = info.offset, info.get_size()
            for index in range(offset, offset + size):
                label = f"[{index}] {name}"
                if size > 1:
                    label += f" ({index - offset})"
                yield index, label

    # Show pose inspector UI.
    # Note we only allow to edit the pose even when the simulation is paused.
    if psim.CollapsingHeader("Pose"):
        pose = _get_articulated_pose(actor.instance)
        lower_limits, upper_limits = _get_articulated_dof_limits(actor.instance).T
        needs_update = False

        psim.PushID("Pose")
        psim.BeginDisabled(not state.paused)
        for index, label in iterate_all_dofs():
            ddof_dt = pose[index]
            dof_limits = (lower_limits[index], upper_limits[index])
            if np.isfinite(dof_limits).all():
                changed, ddof_dt = psim.SliderFloat(
                    label, float(ddof_dt), float(dof_limits[0]), float(dof_limits[1])
                )
            else:
                changed, ddof_dt = psim.InputFloat(label, float(ddof_dt))
            if changed:
                pose[index] = ddof_dt
                needs_update = True
        psim.EndDisabled()
        copy_to_clipboard = psim.Button("Copy to Clipboard")
        psim.Spacing()
        psim.PopID()

        if needs_update:
            state.requires_update = True
            actor.instance.set_articulated_pose_from_joints(pose.tolist())

        if copy_to_clipboard:
            contents = {"pose": pose.tolist()}
            psim.SetClipboardText(json.dumps(contents))

    # Show velocity inspector UI.
    # NOTE: We do not allow to edit the velocity.
    if psim.CollapsingHeader("Velocity"):
        velocity = _get_articulated_joint_velocities(actor.instance)
        psim.PushID("Velocity")
        psim.BeginDisabled(True)
        for index, label in iterate_all_dofs():
            psim.InputFloat(label, float(velocity[index]))
        psim.EndDisabled()
        copy_to_clipboard = psim.Button("Copy to Clipboard")
        psim.Spacing()
        psim.PopID()

        if copy_to_clipboard:
            contents = {"velocity": velocity.tolist()}
            psim.SetClipboardText(json.dumps(contents))

    psim.PopID()


def build_mesh_renderer_inspector(renderer: MeshRenderer) -> None:
    """
    Constructs the UI for inspecting properties of a mesh renderer.
    """

    psim.PushID("MeshRendererInspector")
    if psim.CollapsingHeader("Rendering"):
        # Common rendering properties.
        widgets.slider_property(
            "Transparency",
            renderer.get_transparency,
            renderer.set_transparency,
            v_min=0.0,
            v_max=1.0,
        )
        psim.Spacing()

        # Surface rendering properties.
        widgets.rgb_color_property(
            "Front Face Color",
            renderer.get_front_face_color,
            renderer.set_front_face_color,
        )
        widgets.rgb_color_property(
            "Back Face Color",
            renderer.get_back_face_color,
            renderer.set_back_face_color,
        )
        widgets.enum_property(
            "Back Face Policy",
            renderer.get_back_face_policy,
            renderer.set_back_face_policy,
            MeshRenderer.BackFacePolicy,
        )
        widgets.enum_property(
            "Material",
            renderer.get_material,
            renderer.set_material,
            MeshRenderer.Material,
        )
        widgets.bool_property(
            "Smooth Shading",
            renderer.get_smooth_shading,
            renderer.set_smooth_shading,
        )
        psim.Spacing()

        # Edge rendering properties.
        widgets.rgb_color_property(
            "Edge Color",
            renderer.get_edge_color,
            renderer.set_edge_color,
        )
        widgets.slider_property(
            "Edge Width",
            renderer.get_edge_width,
            renderer.set_edge_width,
            v_min=0.0,
            v_max=1.0,
        )
        widgets.scalar_property(
            "Edge Radius",
            renderer.get_edge_radius,
            renderer.set_edge_radius,
        )
        psim.Spacing()

        # Node rendering properties.
        widgets.rgb_color_property(
            "Node Color",
            renderer.get_node_color,
            renderer.set_node_color,
        )
        widgets.scalar_property(
            "Node Radius",
            renderer.get_node_radius,
            renderer.set_node_radius,
        )
        psim.Spacing()

        # Texture rendering properties.
        widgets.enum_property(
            "Texture Origin",
            renderer.get_texture_origin,
            renderer.set_texture_origin,
            MeshRenderer.TextureOrigin,
        )
        widgets.enum_property(
            "Texture Filter",
            renderer.get_texture_filter,
            renderer.set_texture_filter,
            MeshRenderer.TextureFilter,
        )
        widgets.enum_property(
            "Texture Colormap",
            renderer.get_texture_colormap,
            renderer.set_texture_colormap,
            MeshRenderer.Colormap,
        )
        psim.Spacing()

        # Force update the renderer if dirty. Prevents from updating the entire scene.
        if renderer.is_dirty():
            renderer.update()

    psim.PopID()


def build_transform_widget(
    translation: npt.NDArray[float], rotation: npt.NDArray[float]
) -> None:
    flags = psim.ImGuiInputTextFlags_ReadOnly
    psim.BeginDisabled(True)
    psim.InputFloat3("Translation", translation.tolist(), flags=flags)
    psim.InputFloat3("Rotation", rotation.tolist(), flags=flags)
    psim.EndDisabled()
    copy_to_clipboard = psim.Button("Copy to Clipboard")
    psim.Spacing()

    # Copy transform to clipboard in JSON format.
    if copy_to_clipboard:
        contents = {
            "translation": translation.tolist(),
            "rotation": rotation.tolist(),
        }
        psim.SetClipboardText(json.dumps(contents))
