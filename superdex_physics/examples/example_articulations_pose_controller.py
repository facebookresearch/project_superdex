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

"""Example: Articulation pose controllers

A double pendulum on a rail walks through three pose-controller regimes in one
continuous simulation. A hybrid controller first damps the moving pendulum toward
its seeded rest target. Joint-only tracking then pulls the rail back and drives it
forward while swinging the upper hinge to kick the ball. Finally, link-only
Cartesian tracking sweeps the ``LowerArm`` end effector around a horizontal circle.

This programmatic twin loads
``assets/samples/articulations_double_pendulum_on_rail.mochi_scene`` and adds the
controller in code; the declarative twin is
``assets/samples/articulations_pose_controller.mochi_scene``.

The live readout relates each active target to the current joint/link state and the
controller force or torque produced at every degree of freedom. Keep the debugger
attached to watch all three phases; the circular phase continues until the debugger
detaches.
"""

from __future__ import annotations

import math

import numpy as np
import superdex.physics as physics
from superdex.physics import Actor, Scene
from superdex.physics.paths import resolve_asset, resolve_asset_root

np_real = np.float64 if physics.uses_double_precision() else np.float32

ARTICULATION_NAME = "DoublePendulumOnRail"
NUM_DOFS = 5
NUM_LINKS = 4
RAIL_DOF = 0  # prismatic rail position [m]
UPPER_HINGE_DOF = 1  # revolute upper-hinge angle [rad]

# Controller arrays are link-indexed; entry i controls link i and its inbound joint:
#   0: RailHousing / CeilingWeld (Hard)      -> no DoFs
#   1: Cart        / Rail        (Prismatic) -> DoF 0
#   2: UpperArm    / UpperSwing  (Revolute)  -> DoF 1
#   3: LowerArm    / LowerSwing  (Spherical) -> DoFs 2-4
CART_LINK = 1
UPPER_ARM_LINK = 2
END_EFFECTOR_LINK = 3
SEED_JOINT_VELOCITIES = [0.3, 4.2, 0.0, 0.0, 0.0]  # rail [m/s]; rotations [rad/s]

HYBRID = "hybrid"
JOINT_ONLY = "joint only"
LINK_ONLY = "link only"

# Joint-tracking gains are per-DoF-type: the rail is prismatic (linear units),
# the upper hinge revolute (angular units), so they cannot share one gain pair.
RAIL_JOINT_STIFFNESS = 125.0  # [N/m]
RAIL_JOINT_DAMPING = 8.8  # [N*s/m]
HINGE_JOINT_STIFFNESS = 31.25  # [N*m/rad]
HINGE_JOINT_DAMPING = 2.2  # [N*m*s/rad]
END_EFFECTOR_POSITION_STIFFNESS = 75.0  # [N/m]
END_EFFECTOR_POSITION_DAMPING = 5.3  # [N*s/m]
END_EFFECTOR_ROTATION_STIFFNESS = 1.56  # [N*m/rad]
END_EFFECTOR_ROTATION_DAMPING = 0.22  # [N*m*s/rad]

TIME_STEP = 1.0 / 60.0  # [s]
HOLD_DURATION = 3.0  # [s]
KICK_BACK_DURATION = 2.0  # [s]
KICK_FORWARD_DURATION = 2.0  # [s]
CIRCLE_START_TIME = HOLD_DURATION + KICK_BACK_DURATION + KICK_FORWARD_DURATION  # [s]
CIRCLE_RAMP_DURATION = 2.0  # [s]
REPORT_INTERVAL = 1.0  # [s]

RAIL_BACK_OFFSET = -0.19  # [m]
RAIL_TARGET = 0.19  # [m]
HINGE_TARGET = -1.2  # [rad]
CIRCLE_RADIUS = 0.10  # [m]
CIRCLE_PERIOD = 4.0  # [s]


def _add_hybrid_controller(articulation: Actor) -> None:
    """Add a controller with joint and end-effector tracking."""
    params = physics.PoseControllerParams(NUM_LINKS)
    params.joint_tracking[CART_LINK] = physics.PoseTrackingParams(
        stiffness=RAIL_JOINT_STIFFNESS,
        damping=RAIL_JOINT_DAMPING,
    )
    params.joint_tracking[UPPER_ARM_LINK] = physics.PoseTrackingParams(
        stiffness=HINGE_JOINT_STIFFNESS,
        damping=HINGE_JOINT_DAMPING,
    )
    params.link_pos_tracking[END_EFFECTOR_LINK] = physics.PoseTrackingParams(
        stiffness=END_EFFECTOR_POSITION_STIFFNESS,
        damping=END_EFFECTOR_POSITION_DAMPING,
    )
    params.link_rot_tracking[END_EFFECTOR_LINK] = physics.PoseTrackingParams(
        stiffness=END_EFFECTOR_ROTATION_STIFFNESS,
        damping=END_EFFECTOR_ROTATION_DAMPING,
    )
    articulation.add_articulated_pose_controller(params)


def _set_hybrid_controller(articulation: Actor) -> None:
    """Enable joint and end-effector tracking."""
    params = physics.PoseControllerParams(NUM_LINKS)
    params.joint_tracking[CART_LINK] = physics.PoseTrackingParams(
        stiffness=RAIL_JOINT_STIFFNESS,
        damping=RAIL_JOINT_DAMPING,
    )
    params.joint_tracking[UPPER_ARM_LINK] = physics.PoseTrackingParams(
        stiffness=HINGE_JOINT_STIFFNESS,
        damping=HINGE_JOINT_DAMPING,
    )
    params.link_pos_tracking[END_EFFECTOR_LINK] = physics.PoseTrackingParams(
        stiffness=END_EFFECTOR_POSITION_STIFFNESS,
        damping=END_EFFECTOR_POSITION_DAMPING,
    )
    params.link_rot_tracking[END_EFFECTOR_LINK] = physics.PoseTrackingParams(
        stiffness=END_EFFECTOR_ROTATION_STIFFNESS,
        damping=END_EFFECTOR_ROTATION_DAMPING,
    )
    articulation.set_articulated_pose_controller_params(params)


def _set_joint_only_controller(articulation: Actor) -> None:
    """Enable joint tracking and disable link tracking."""
    params = physics.PoseControllerParams(NUM_LINKS)
    params.joint_tracking[CART_LINK] = physics.PoseTrackingParams(
        stiffness=RAIL_JOINT_STIFFNESS,
        damping=RAIL_JOINT_DAMPING,
    )
    params.joint_tracking[UPPER_ARM_LINK] = physics.PoseTrackingParams(
        stiffness=HINGE_JOINT_STIFFNESS,
        damping=HINGE_JOINT_DAMPING,
    )
    articulation.set_articulated_pose_controller_params(params)


def _set_link_only_controller(articulation: Actor) -> None:
    """Enable end-effector tracking and disable joint tracking."""
    params = physics.PoseControllerParams(NUM_LINKS)
    params.link_pos_tracking[END_EFFECTOR_LINK] = physics.PoseTrackingParams(
        stiffness=END_EFFECTOR_POSITION_STIFFNESS,
        damping=END_EFFECTOR_POSITION_DAMPING,
    )
    params.link_rot_tracking[END_EFFECTOR_LINK] = physics.PoseTrackingParams(
        stiffness=END_EFFECTOR_ROTATION_STIFFNESS,
        damping=END_EFFECTOR_ROTATION_DAMPING,
    )
    articulation.set_articulated_pose_controller_params(params)


def build_scene() -> tuple[Scene, Actor]:
    """Load the plain scene and add the hybrid joint/link pose controller."""
    scene = physics.create_scene("Articulations Pose Controller Scene")
    result = physics.prefab.add_to_scene(
        prefab_path=str(
            resolve_asset("samples/articulations_double_pendulum_on_rail.mochi_scene")
        ),
        root_path=str(
            resolve_asset_root(
                "samples/articulations_double_pendulum_on_rail.mochi_scene"
            )
        ),
        scene=scene,
    )
    articulation = next(
        actor
        for actor in result.filter(physics.ActorType.ARTICULATED)
        if actor.get_name() == ARTICULATION_NAME
    )

    _add_hybrid_controller(articulation)
    return scene, articulation


def _get_pose(articulation: Actor) -> np.ndarray:
    pose = np.zeros(articulation.get_num_dofs(), dtype=np_real)
    articulation.get_articulated_pose(out_pose=pose)
    return pose


def _get_controller_params(articulation: Actor) -> physics.PoseControllerParams:
    params = physics.PoseControllerParams(NUM_LINKS)
    articulation.get_articulated_pose_controller_params(out_params=params)
    return params


def _format_gains(gains: physics.PoseTrackingParams) -> str:
    if gains.stiffness == 0.0 and gains.damping == 0.0:
        return "off"
    return f"Kp={gains.stiffness:g}, Kd={gains.damping:g}"


def print_controller_setup(articulation: Actor) -> None:
    """Print each link's joint- and Cartesian-tracking gains."""
    params = _get_controller_params(articulation)
    shape_info = articulation.get_articulated_shape_info()

    print(
        f"pose controller: present={articulation.has_articulated_pose_controller()}, "
        f"num_dofs={articulation.get_num_dofs()}, "
        f"num_links={len(shape_info.link_names)}"
    )
    for link in range(NUM_LINKS):
        print(
            f"  {shape_info.link_names[link]} / {shape_info.joint_names[link]}: "
            f"joint [{_format_gains(params.joint_tracking[link])}], "
            f"position [{_format_gains(params.link_pos_tracking[link])}], "
            f"rotation [{_format_gains(params.link_rot_tracking[link])}]"
        )


def reset_to_initial_state(articulation: Actor) -> np.ndarray:
    """Restore the prefab's starting state and return its rest target."""
    rest_target = np.zeros(NUM_DOFS, dtype=np_real)
    articulation.set_articulated_pose_from_joints(pose=rest_target)
    articulation.set_articulated_joint_velocities(
        velocities=np.array(SEED_JOINT_VELOCITIES, dtype=np_real)
    )
    articulation.reset_articulated_target_pose(pose=rest_target)
    return rest_target


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _phase_at(sim_time: float) -> tuple[str, str]:
    if sim_time < HOLD_DURATION:
        return "hybrid hold", HYBRID
    if sim_time < CIRCLE_START_TIME:
        return "joint-control kick", JOINT_ONLY
    return "link-control circle", LINK_ONLY


def _joint_kick_target(sim_time: float, rest_target: np.ndarray) -> np.ndarray:
    target = rest_target.copy()
    kick_time = sim_time - HOLD_DURATION
    if kick_time <= 0.0:
        return target

    if kick_time < KICK_BACK_DURATION:
        target[RAIL_DOF] += RAIL_BACK_OFFSET * _smoothstep(
            kick_time / KICK_BACK_DURATION
        )
        return target

    forward_fraction = _smoothstep(
        (kick_time - KICK_BACK_DURATION) / KICK_FORWARD_DURATION
    )
    target[RAIL_DOF] += (
        RAIL_BACK_OFFSET + (RAIL_TARGET - RAIL_BACK_OFFSET) * forward_fraction
    )
    target[UPPER_HINGE_DOF] += HINGE_TARGET * forward_fraction
    return target


def _circle_link_target(
    elapsed: float, center_transforms: physics.DynamicArrayTransformRT
) -> physics.DynamicArrayTransformRT:
    center_transform = center_transforms[END_EFFECTOR_LINK]
    radius = CIRCLE_RADIUS * _smoothstep(elapsed / CIRCLE_RAMP_DURATION)
    angle = 2.0 * math.pi * elapsed / CIRCLE_PERIOD
    center = center_transform.translation
    position = [
        center[0] + radius * math.cos(angle),
        center[1],
        center[2] + radius * math.sin(angle),
    ]

    targets = physics.DynamicArrayTransformRT(list(center_transforms))
    targets[END_EFFECTOR_LINK] = physics.TransformRT(
        center_transform.rotation, position
    )
    return targets


def _report(
    articulation: Actor,
    sim_time: float,
    phase_name: str,
    mode: str,
    joint_target: np.ndarray | None,
    link_targets: physics.DynamicArrayTransformRT | None,
) -> None:
    pose = _get_pose(articulation)
    link_transforms = physics.DynamicArrayTransformRT(NUM_LINKS)
    articulation.get_articulated_link_transforms(out_world_from_links=link_transforms)
    end_effector_position = link_transforms[END_EFFECTOR_LINK].translation
    controller_generalized_forces = articulation.get_articulated_controller_force()

    print(f"t={sim_time:.1f}s | phase={phase_name} | mode={mode}")
    if mode == LINK_ONLY:
        assert link_targets is not None
        target_position = link_targets[END_EFFECTOR_LINK].translation
        print(
            "  active target: end-effector world position="
            f"{[round(float(value), 3) for value in target_position]} m"
        )
    else:
        assert joint_target is not None
        print(
            f"  active target: rail={joint_target[RAIL_DOF]:.3f} m, "
            f"upper hinge={joint_target[UPPER_HINGE_DOF]:.3f} rad"
        )

    print(
        f"  current: rail={pose[RAIL_DOF]:.3f} m, "
        f"upper hinge={pose[UPPER_HINGE_DOF]:.3f} rad, "
        "end-effector world position="
        f"{[round(float(value), 3) for value in end_effector_position]} m"
    )
    generalized_force_readout = [
        f"DoF 0={float(controller_generalized_forces[0]):.3f} N"
    ]
    generalized_force_readout.extend(
        f"DoF {dof}={float(controller_generalized_forces[dof]):.3f} N*m"
        for dof in range(1, articulation.get_num_dofs())
    )
    print(f"  controller force/torque: {', '.join(generalized_force_readout)}")


def run_interactive(scene: Scene, articulation: Actor, rest_target: np.ndarray) -> None:
    """Run the three controller phases while the debugger remains attached."""
    if not physics.debugger.attach():
        return

    sim_time = 0.0
    next_report = REPORT_INTERVAL
    old_mode = HYBRID
    circle_center_transforms: physics.DynamicArrayTransformRT | None = None

    def detect_debugger_reset(step_info: physics.StepInfo) -> None:
        nonlocal next_report
        nonlocal old_mode
        nonlocal circle_center_transforms
        if step_info.scene.get_total_simulation_time() < sim_time:
            _set_hybrid_controller(articulation)
            next_report = REPORT_INTERVAL
            old_mode = HYBRID
            circle_center_transforms = None
            print("Debugger reset detected; restarting hybrid hold.")

    reset_callback = scene.register_pre_step_callback(
        "Detect pose-controller tutorial reset", detect_debugger_reset
    )
    query = articulation.register_query(physics.QueryType.ARTICULATED_CONTROLLER_FORCE)

    while physics.debugger.is_attached():
        phase_name, mode = _phase_at(sim_time)
        if mode != old_mode:
            if mode == JOINT_ONLY:
                _set_joint_only_controller(articulation)
            else:
                _set_link_only_controller(articulation)
            print(f"t={sim_time:.1f}s: entering {phase_name}; controller mode={mode}")

        joint_target: np.ndarray | None = None
        link_targets: physics.DynamicArrayTransformRT | None = None
        if mode == HYBRID:
            joint_target = rest_target
        elif mode == JOINT_ONLY:
            joint_target = _joint_kick_target(sim_time, rest_target)
            articulation.set_articulated_target_pose(pose=joint_target)
        else:  # LINK_ONLY
            if mode != old_mode:
                # Preserve a valid world-space target for every link at handoff.
                circle_center_transforms = physics.DynamicArrayTransformRT(NUM_LINKS)
                articulation.get_articulated_link_transforms(
                    out_world_from_links=circle_center_transforms
                )
            assert circle_center_transforms is not None
            circle_elapsed = sim_time - CIRCLE_START_TIME
            link_targets = _circle_link_target(circle_elapsed, circle_center_transforms)
            if mode != old_mode:
                # Reset clears inferred target velocity at the mode boundary.
                articulation.reset_articulated_target_link_transforms(
                    world_from_targets=link_targets
                )
            else:
                articulation.set_articulated_target_link_transforms(
                    world_from_targets=link_targets
                )

        old_mode = mode
        scene.step(TIME_STEP)
        sim_time = scene.get_total_simulation_time()

        if sim_time >= next_report:
            _report(
                articulation,
                sim_time,
                phase_name,
                mode,
                joint_target,
                link_targets,
            )
            next_report += REPORT_INTERVAL

    scene.cancel_callback(reset_callback)
    articulation.cancel_query(query)


def main() -> None:
    """Run the interactive articulation pose-controller tutorial."""
    physics.initialize(num_worker_threads=0)
    scene, articulation = build_scene()

    print_controller_setup(articulation)
    rest_target = reset_to_initial_state(articulation)
    run_interactive(scene, articulation, rest_target)

    physics.destroy_scene(scene)
    physics.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
