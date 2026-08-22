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

"""Example: Compare rod, spatial-tendon, and linear-transmission tendon models.

See website/docs/examples/tendons_rods/tendon_model_fidelity.md.
"""

import math
from dataclasses import dataclass

import numpy as np
import superdex.physics as physics
from superdex.physics import Actor, ActorHandle, Scene
from superdex.physics.paths import resolve_asset, resolve_asset_root

TENDON_ARTICULATION_PREFAB = "samples/tendon_comparison_articulation.mochi_scene"

SLIDE_AMPLITUDE = 0.15  # Peak slider pull [m]
DRIVE_TIME_SCALE = 1.0  # [s]
TIME_STEP = 1.0 / 60.0  # [s]

# Contact stiffness shared by the rod and the links it contacts. The default is chosen
# for general object manipulation; contact between the narrow tendon and the sharp
# eyelet geometry concentrates force over a small region, so a custom value is used.
PENALTY_COEFFICIENT = 1e11  # [Pa/m]

ROD_RADIUS = 0.5e-2  # [m]
ROD_DENSITY = 1e3  # [kg/m^3]
ROD_YOUNGS_MODULUS = 1e8  # [Pa]
ROD_SHEAR_MODULUS = 1e8  # [Pa]
ROD_NUM_ELEMENTS = 128
ROD_NUM_CROSS_SECTION_SEGMENTS = 8
ROD_AREA = math.pi * ROD_RADIUS**2  # [m^2]

BONE_LINK_NAMES = ("Bone0", "Bone1", "Bone2", "Bone3")
EYELET_LINK_NAMES = ("Eyelet0", "Eyelet1", "Eyelet2", "Eyelet3")
HINGE_JOINT_NAMES = ("BoneHinge1", "BoneHinge2", "BoneHinge3")
ROOT_JOINT_NAME = "RootFree"
SLIDER_LINK_NAME = "Slider"
SLIDER_JOINT_NAME = "SliderPrismatic"

# The low-fidelity model assumes a constant moment arm rather than deriving one from
# the articulation geometry.
LINEAR_MOMENT_ARM = 0.05  # [m]

CLONE_Z_OFFSET = 0.5  # [m]
SCENE_X_TRANSLATION = -0.5  # [m]
SCENE_Z_TRANSLATION = -1.0  # [m]


@dataclass
class TendonArticulation:
    """Shared articulated assembly reusable across transmission models."""

    actor: Actor
    link_indices: dict[str, int]
    joint_indices: dict[str, int]
    slider_link: ActorHandle
    last_eyelet_link: ActorHandle
    bc_dof_indices: np.ndarray  # free-root DoFs + slider DoF
    bc_root_values: list[float]
    rod_start_root: np.ndarray
    rod_end_root: np.ndarray
    rest_length: float


def _joint_dof_indices(
    shape_info: physics.ArticulatedShapeInfo, joint_index: int
) -> list[int]:
    """Return the flattened DoF indices belonging to one joint."""
    dof_info = shape_info.dof_info[joint_index]
    return list(range(dof_info.offset, dof_info.offset + dof_info.get_size()))


def _actuator_stiffness(rest_length: float) -> float:
    """Match a length actuator to the rod's axial structural stiffness EA/L."""
    return ROD_YOUNGS_MODULUS * ROD_AREA / rest_length


def create_tendon_articulation(
    prefab: physics.prefab.ScenePrefab,
    scene: Scene,
    translation: list[float],
    name: str,
) -> TendonArticulation:
    """Instantiate and resolve the shared tendon articulation by semantic names."""
    result = physics.prefab.add_to_scene(
        prefab=prefab,
        scene=scene,
        params=physics.prefab.PrefabParams(
            name=name,
            translation=translation,
            apply_scene_settings=False,
        ),
    )
    articulated_actors = result.filter(physics.ActorType.ARTICULATED)
    assert len(articulated_actors) == 1, (
        "Tendon articulation prefab must create exactly one articulated actor."
    )
    actor = articulated_actors[0]

    shape_info = actor.get_articulated_shape_info()
    link_indices = {
        link_name: index for index, link_name in enumerate(shape_info.link_names)
    }
    joint_indices = {
        joint_name: index for index, joint_name in enumerate(shape_info.joint_names)
    }

    required_links = {
        *BONE_LINK_NAMES,
        *EYELET_LINK_NAMES,
        SLIDER_LINK_NAME,
    }
    required_joints = {
        ROOT_JOINT_NAME,
        *HINGE_JOINT_NAMES,
        *(f"EyeletWeld{i}" for i in range(len(EYELET_LINK_NAMES))),
        SLIDER_JOINT_NAME,
    }
    missing_links = required_links - link_indices.keys()
    missing_joints = required_joints - joint_indices.keys()
    assert not missing_links, (
        f"Tendon articulation prefab is missing links: {missing_links}"
    )
    assert not missing_joints, (
        f"Tendon articulation prefab is missing joints: {missing_joints}"
    )

    root_dof_indices = _joint_dof_indices(shape_info, joint_indices[ROOT_JOINT_NAME])
    slider_dof_indices = _joint_dof_indices(
        shape_info, joint_indices[SLIDER_JOINT_NAME]
    )
    assert len(root_dof_indices) == 6, "RootFree must provide six DoFs."
    assert len(slider_dof_indices) == 1, "SliderPrismatic must provide one DoF."

    root_values = physics.DynamicArrayReal(len(root_dof_indices))
    actor.get_dof_values(np.array(root_dof_indices, dtype=np.int32), root_values)
    bc_root_values = list(root_values)
    bc_dof_indices = np.array(root_dof_indices + slider_dof_indices, dtype=np.int32)
    actor.add_boundary_condition_dofs_world(bc_dof_indices, bc_root_values + [0.0])

    slider_link_index = link_indices[SLIDER_LINK_NAME]
    last_eyelet_index = link_indices[EYELET_LINK_NAMES[-1]]
    rod_start_root = np.array(
        shape_info.root_from_links_at_rest[slider_link_index].translation,
        dtype=np.float32,
    )
    rod_end_root = np.array(
        shape_info.root_from_links_at_rest[last_eyelet_index].translation,
        dtype=np.float32,
    )
    rest_length = float(np.linalg.norm(rod_end_root - rod_start_root))
    assert rest_length > 0.0, "Tendon attachment frames must be distinct."

    link_actors = actor.get_nested_link_actors()
    return TendonArticulation(
        actor=actor,
        link_indices=link_indices,
        joint_indices=joint_indices,
        slider_link=link_actors[slider_link_index],
        last_eyelet_link=link_actors[last_eyelet_index],
        bc_dof_indices=bc_dof_indices,
        bc_root_values=bc_root_values,
        rod_start_root=rod_start_root,
        rod_end_root=rod_end_root,
        rest_length=rest_length,
    )


def add_rod_tendon(scene: Scene, art: TendonArticulation) -> Actor:
    """Build a straight elastic rod between the prefab's attachment frames."""
    world_from_root = art.actor.get_root_transform()
    world_from_start = world_from_root * physics.TransformRT(
        translation=art.rod_start_root
    )
    world_from_end = world_from_root * physics.TransformRT(translation=art.rod_end_root)
    start_world = np.array(world_from_start.translation, dtype=np.float32)
    end_world = np.array(world_from_end.translation, dtype=np.float32)

    num_nodes = ROD_NUM_ELEMENTS + 1
    t = np.linspace(0.0, 1.0, num_nodes, dtype=np.float32)[:, None]
    nodes = start_world + t * (end_world - start_world)
    element_frame_axes = np.tile([0.0, 1.0, 0.0], (ROD_NUM_ELEMENTS, 1)).astype(
        np.float32
    )

    # Build a polyline simulation mesh plus a tubular visual mesh (with embedding data)
    # so the rod can use visual-mesh contact.
    model = physics.experimental.generate_tubular_rod_model_data(
        nodes=nodes,
        element_frame_axes=element_frame_axes,
        radius=ROD_RADIUS,
        num_cross_section_segments=ROD_NUM_CROSS_SECTION_SEGMENTS,
        is_closed_loop=False,
    )
    shape = physics.create_model_shape(model)

    polar_moment_of_inertia = 0.5 * math.pi * ROD_RADIUS**4
    second_moment_of_area = 0.25 * math.pi * ROD_RADIUS**4
    torsion_constant = 0.5 * math.pi * ROD_RADIUS**4
    material = physics.experimental.RodMaterialParams(
        linear_density=ROD_DENSITY * ROD_AREA,
        linear_rotational_inertia=ROD_DENSITY * polar_moment_of_inertia,
        axial_stiffness=ROD_AREA * ROD_YOUNGS_MODULUS,
        torsional_stiffness=ROD_SHEAR_MODULUS * torsion_constant,
        flexural_stiffness=[
            ROD_YOUNGS_MODULUS * second_moment_of_area,
            ROD_YOUNGS_MODULUS * second_moment_of_area,
        ],
    )

    rod = physics.experimental.create_rod_actor(
        scene,
        physics.experimental.RodActorParams(
            name="TendonRod",
            shape=shape,
            world_from_local=physics.TransformRT(),
            material=material,
            layer="Tendon",
            contact=physics.ContactParams(penalty_coefficient=PENALTY_COEFFICIENT),
            use_visual_mesh_contact=True,
        ),
    )

    # Attach both rod endpoints to the semantic local origins authored in the prefab.
    last_node = num_nodes - 1
    scene.create_deformable_node_to_rigid_constraint(
        deformable_actor=rod.get_handle(),
        rigid_actor=art.slider_link,
        deformable_node_index=0,
        rigid_local_pos=[0.0, 0.0, 0.0],
        fix_to_deformable_pos=False,
    )
    scene.create_deformable_node_to_rigid_constraint(
        deformable_actor=rod.get_handle(),
        rigid_actor=art.last_eyelet_link,
        deformable_node_index=last_node,
        rigid_local_pos=[0.0, 0.0, 0.0],
        fix_to_deformable_pos=False,
    )

    return rod


def drive_slider(art: TendonArticulation, sim_time: float) -> None:
    """Kinematically drive the slider with a smooth raised-cosine pull each step."""
    slider_dof = SLIDE_AMPLITUDE * 0.5 * (1.0 - math.cos(sim_time / DRIVE_TIME_SCALE))
    art.actor.clear_boundary_conditions()
    art.actor.add_boundary_condition_dofs_world(
        art.bc_dof_indices, art.bc_root_values + [slider_dof]
    )


def add_spatial_tendon(art: TendonArticulation) -> int:
    """Route a spatial tendon through the prefab's semantic attachment frames."""
    route_names = (SLIDER_LINK_NAME, *EYELET_LINK_NAMES)
    elements = [
        physics.RoutingElement(
            type=physics.RoutingElementType.WAYPOINT,
            index=art.link_indices[link_name],
            local_position=[0.0, 0.0, 0.0],
        )
        for link_name in route_names
    ]
    tendon_index = physics.experimental.add_spatial_tendon(
        art.actor,
        physics.experimental.SpatialTendonParams(routing_elements=elements),
    )
    physics.experimental.attach_displacement_control_actuator(
        art.actor,
        tendon_index,
        physics.experimental.DisplacementControlActuatorParams(
            stiffness=_actuator_stiffness(art.rest_length), target_displacement=0.0
        ),
    )
    return tendon_index


def add_linear_transmission(art: TendonArticulation) -> int:
    """Couple the named slider and hinge joints using a fixed moment arm."""
    joint_indices = [
        art.joint_indices[SLIDER_JOINT_NAME],
        *(art.joint_indices[name] for name in HINGE_JOINT_NAMES),
    ]
    joint_coefficients = [1.0] + [-LINEAR_MOMENT_ARM] * len(HINGE_JOINT_NAMES)

    tendon_index = physics.experimental.add_linear_transmission(
        art.actor,
        physics.experimental.LinearTransmissionParams(
            joint_indices=joint_indices,
            joint_coefficients=joint_coefficients,
        ),
    )
    physics.experimental.attach_displacement_control_actuator(
        art.actor,
        tendon_index,
        physics.experimental.DisplacementControlActuatorParams(
            stiffness=_actuator_stiffness(art.rest_length), target_displacement=0.0
        ),
    )
    return tendon_index


def create_tendon_comparison_simulation() -> tuple[Scene, list[TendonArticulation]]:
    """Build the three side-by-side tendon-fidelity copies and configure the solver.

    Note: physics.initialize() must be called before this function.
    """
    scene = physics.create_scene("Tendon Fidelity Comparison Scene")

    prefab = physics.prefab.load_from_file(
        prefab_path=str(resolve_asset(TENDON_ARTICULATION_PREFAB)),
        root_path=str(resolve_asset_root(TENDON_ARTICULATION_PREFAB)),
    )

    art_rod = create_tendon_articulation(
        prefab,
        scene,
        [SCENE_X_TRANSLATION, 0.0, SCENE_Z_TRANSLATION],
        "RodTendonArticulation",
    )
    add_rod_tendon(scene, art_rod)

    art_spatial = create_tendon_articulation(
        prefab,
        scene,
        [
            SCENE_X_TRANSLATION,
            0.0,
            SCENE_Z_TRANSLATION + CLONE_Z_OFFSET,
        ],
        "SpatialTendonArticulation",
    )
    add_spatial_tendon(art_spatial)

    art_linear = create_tendon_articulation(
        prefab,
        scene,
        [
            SCENE_X_TRANSLATION,
            0.0,
            SCENE_Z_TRANSLATION + 2.0 * CLONE_Z_OFFSET,
        ],
        "LinearTransmissionArticulation",
    )
    add_linear_transmission(art_linear)

    solver_params = scene.get_solver_params()
    solver_params.non_linear_solver.line_search_type = (
        physics.LineSearchType.WOLFE_STRONG
    )
    scene.set_solver_params(solver_params)

    return scene, [art_rod, art_spatial, art_linear]


def main():
    """Build the three tendon-fidelity copies and run the interactive comparison."""
    physics.initialize(num_worker_threads=0)

    scene, articulations = create_tendon_comparison_simulation()

    # Enable debug draw for the rod polyline, spatial tendon, and linear transmission.
    debug_draw = scene.get_debug_draw()
    debug_draw.enable(True)
    debug_draw.enable_feature(debug_draw.find_feature("Rod Actor Polyline"), True)
    debug_draw.enable_feature(debug_draw.find_feature("Spatial Tendon"), True)
    debug_draw.enable_feature(
        debug_draw.find_feature("Linear Transmission Terms"), True
    )

    if not physics.debugger.attach():
        physics.shutdown()
        return

    while physics.debugger.is_attached():
        sim_time = scene.get_total_simulation_time()
        for art in articulations:
            drive_slider(art, sim_time)
        scene.step(TIME_STEP)

    physics.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
