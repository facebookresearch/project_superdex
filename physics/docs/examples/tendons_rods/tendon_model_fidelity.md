---
title: Choosing a Tendon Model
sidebar_position: 1
---

# Choosing a Tendon Model

:::caution Experimental API
Rod actors and transmissions are part of the **experimental** API. Their interfaces may change without notice.
:::

This example compares three ways to model the same tendon-driven articulated mechanism: an elastic rod, a waypoint-routed spatial tendon, and a fixed-coefficient linear transmission. The example places all three models side by side and drives them with the same prescribed slider motion.

**Source**: `examples/example_tendon_comparison.py`

For the underlying formulations and parameter references, see [Transmissions](../../concepts/transmissions.mdx) and [Rod Actors](../../concepts/actors/rod.mdx).

## Choose a Fidelity Level

| Model | What it captures | Relative cost | Use it when |
|---|---|---|---|
| Rod | A deformable centerline, cross-section motion, contact within each eyelet, and potentially frictional routing | Highest | Contact, bending, torsion, or routing details affect the behavior of interest |
| Spatial tendon | Pose-dependent moment arms derived from waypoint geometry | Moderate | You need a practical tendon model whose routing changes with pose |
| Linear transmission | Fixed moment arms encoded as constant joint coefficients | Lowest | Moment-arm variation is minor over the mechanism's operating range |

A spatial tendon is the recommended default for most tendon-actuation problems: it retains pose-dependent moment arms without adding the rod's many deformable degrees of freedom.

A spatial tendon may also mix waypoint and linear-joint routing elements. A linear-joint term can approximate limited wrapping cases in which the tendon follows a constant-radius path coaxial with a revolute joint. This is not a general wrapping or contact model, and the example on this page uses waypoints only.

:::warning Qualitative comparison
The three models in this example have **not** been calibrated to match tension, tip displacement, or any other response quantity. Their side-by-side motion illustrates differences in model structure and fidelity, not quantitative agreement.
:::

## Shared Articulation and Slider Motion

Each model uses a copy of the same articulated prefab. The free root and prismatic slider degrees of freedom are prescribed with world-space boundary conditions. The slider value is updated before every simulation step by clearing the previous conditions and installing new values:

```python
def drive_slider(art: TendonArticulation, sim_time: float) -> None:
    slider_dof = SLIDE_AMPLITUDE * 0.5 * (1.0 - math.cos(sim_time / DRIVE_TIME_SCALE))
    art.actor.clear_boundary_conditions()
    art.actor.add_boundary_condition_dofs_world(
        art.bc_dof_indices, art.bc_root_values + [slider_dof]
    )
```

This keeps the root fixed while prescribing a smooth raised-cosine pull at the slider.

## High Fidelity: Elastic Rod

The rod begins as a straight polyline between the slider and distal eyelet attachment frames. `generate_tubular_rod_model_data()` adds a tubular visual surface and the embedding data that relates it to the simulated centerline:

```python
model = physics.experimental.generate_tubular_rod_model_data(
    nodes=nodes,
    element_frame_axes=element_frame_axes,
    radius=ROD_RADIUS,
    num_cross_section_segments=ROD_NUM_CROSS_SECTION_SEGMENTS,
    is_closed_loop=False,
)
shape = physics.create_model_shape(model)
```

If the generated model should be validated or serialized instead of registered directly, follow the [Authoring Assets](../../authoring_scenes/authoring_assets.mdx) workflow.

The rod material parameters, omitted from these snippets, follow the homogeneous circular-cross-section formulas described in [Rod Actors: Parameter Selection](../../concepts/actors/rod.mdx#parameter-selection).

```python
rod = physics.experimental.create_rod_actor(
    scene,
    physics.experimental.RodActorParams(
        name="TendonRod",
        shape=shape,
        material=material,
        layer="Tendon",
        contact=physics.ContactParams(penalty_coefficient=PENALTY_COEFFICIENT),
        use_visual_mesh_contact=True,
    ),
)
```

With `use_visual_mesh_contact=True`, contact samples are placed on the tubular visual surface instead of on the rod centerline, and their forces are mapped to the rod degrees of freedom through the skinning Jacobian. This lets the tendon interact with the interior geometry of each eyelet across its finite cross-section.

The example also uses a custom penalty coefficient because contact between the narrow tendon and sharp eyelet geometry concentrates force over a small region. This regime falls outside the general object-manipulation scenarios for which the default penalty coefficient is selected.

The rod representation supports frictional routing, but the shared prefab at `assets/samples/tendon_comparison_articulation.mochi_scene` sets both Coulomb and viscous friction coefficients on every eyelet to zero. This isolates contact routing without introducing frictional losses that the transmission-based models do not resolve.

The first and last rod nodes are attached to nested rigid links of the articulation:

```python
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
    deformable_node_index=num_nodes - 1,
    rigid_local_pos=[0.0, 0.0, 0.0],
    fix_to_deformable_pos=False,
)
```

These constraints pull the proximal endpoint with the slider and pin the distal endpoint to the final eyelet link. The intervening rod is free to deform and find a contact-constrained path through the guides.

### Choosing the Rod Resolution

The element size of the rod polyline must be small enough to resolve both concentrated contact with eyelets and the sharp bending that results from it. Since these two solution features are causally related, their requirements are similar. The rod spans 0.8 m between its attachment frames, so `ROD_NUM_ELEMENTS = 128` produces 6.25 mm elements against a 20 mm eyelet length, placing about three elements inside every eyelet.

## Medium Fidelity: Spatial Tendon

The spatial tendon replaces the deformable rod with a sequence of points fixed in articulation-link frames:

```python
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
        stiffness=_actuator_stiffness(art.rest_length),
        target_displacement=0.0,
    ),
)
```

Straight segments connect consecutive waypoints. As the links move, the segment directions change, so the tendon Jacobian and joint moment arms change with the articulation pose. Unlike the rod, this model does not resolve contact or deformation between routing elements. Aggregate dissipative effects can instead be approximated with actuator damping.

## Low Fidelity: Linear Transmission

The linear transmission directly couples the slider coordinate and the three hinge angles. Its coefficients are constant:

```python
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
        stiffness=_actuator_stiffness(art.rest_length),
        target_displacement=0.0,
    ),
)
```

The slider coefficient is dimensionless, while each revolute-joint coefficient has units of length and represents an assumed signed moment arm. This model avoids routing geometry entirely, so its suitability depends on whether constant moment arms are a reasonable approximation.

## Matching Nominal Stiffness

Both reduced models use a displacement-control actuator with stiffness matched to the rod's nominal axial structural stiffness:

```python
def _actuator_stiffness(rest_length: float) -> float:
    return ROD_YOUNGS_MODULUS * ROD_AREA / rest_length
```

For a straight rod of length $L$, area $A$, and Young's modulus $E$, the small-strain axial force-displacement stiffness is $EA/L$. This gives the models a common nominal axial scale only; it is not a quantitative calibration. The reduced models' displacement-control actuators also clamp compressive force by default, whereas the elastic rod resists compression.

## Running

Calling [`physics.debugger.attach()`](pathname:///generated/api/v1.0/python/api/debugger.html#superdex.physics.debugger.attach) launches or focuses the SuperDex Physics Debugger for visualization. See [Inspecting Scenes](../../debugging_scenes.md) for debugger connection, navigation, and playback controls. Execute the example from the SuperDex Physics root directory:

```bash
cd <path_to_superdex_physics>
uv run --no-project examples/example_tendon_comparison.py
```

When run, three copies of the same four-bone chain will appear side by side, offset by 0.5 m along $z$. All three are driven by the same raised-cosine slider pull, so they curl and relax together on a shared cycle. What differs is the tendon rendered between the guides.

Each of the three tendon models is drawn by a different debug-draw system, and these systems are off by default. The example enables debug draw and then activates the three it needs by name:

```python
debug_draw = scene.get_debug_draw()
debug_draw.enable(True)
debug_draw.enable_feature(debug_draw.find_feature("Rod Actor Polyline"), True)
debug_draw.enable_feature(debug_draw.find_feature("Spatial Tendon"), True)
debug_draw.enable_feature(
    debug_draw.find_feature("Linear Transmission Terms"), True
)
```

Without these calls the articulations and eyelets still render, but the tendons themselves are invisible. The reduced models have no surface geometry at all, and the rod would show only its tubular visual mesh (if toggled in the debugger) rather than the simulated centerline.

The rod renders as a deforming tube surrounding the simulated centerline polyline, and can be seen to fully resolve dynamic contact with the eyelets. The spatial tendon is depicted as a series of line segments between waypoints at the slider and inside each eyelet. The linear transmission has no associated geometry, and is represented abstractly by symbolic gizmos rendered at the joints, connected by a polyline that indicates mathematical coupling only, not the physical tendon path.
