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

"""Example: Cross-Thread Capture/Restore

A parent-owned scene captures one checkpoint. Two worker-owned compatible scenes
restore its immutable bytes and run independent rollouts of the same kick played
back at different speeds.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from threading import Event

import numpy as np
import superdex.physics as sdp
from superdex.physics import Actor, Scene
from superdex.physics.paths import resolve_asset, resolve_asset_root

np_real = np.float64 if sdp.uses_double_precision() else np.float32

ARTICULATION_NAME = "DoublePendulumOnRail"
ARTICULATION_PREFAB = "samples/articulations_double_pendulum_on_rail.mochi_scene"
NUM_DOFS = 5
NUM_LINKS = 4
ZERO_TARGET = np.zeros(NUM_DOFS, dtype=np_real)
RAIL_DOF = 0  # prismatic rail position [m]
UPPER_HINGE_DOF = 1  # revolute upper-hinge angle [rad]

# Controller arrays are link-indexed; these entries control the rail and upper
# hinge through their inbound joints.
CART_LINK = 1
UPPER_ARM_LINK = 2
LOWER_ARM_LINK = 3

CART_STIFFNESS = 250.0  # [N/m]
CART_DAMPING = 17.7  # [N*s/m]
HINGE_STIFFNESS = 31.25  # [N*m/rad]
HINGE_DAMPING = 2.2  # [N*m*s/rad]
LOWER_SWING_DAMPING = 0.44  # [N*m*s/rad]

TIME_STEP = 1.0 / 60.0  # [s]
KICK_BACK_DURATION = 3.0  # [s]
KICK_FORWARD_DURATION = 2.0  # [s]
MOTION_DURATION = KICK_BACK_DURATION + KICK_FORWARD_DURATION  # [s]
ROLLOUT_STEPS = round(MOTION_DURATION / TIME_STEP)

# Candidate control signals: the nominal kick and the same kick at double speed.
NOMINAL_TIME_SCALE = 1.0
FAST_TIME_SCALE = 2.0

RAIL_BACK_OFFSET = -0.19  # [m]
RAIL_TARGET = 0.19  # [m]
HINGE_TARGET = -1.2  # [rad]


def create_simulation(name: str) -> tuple[Scene, Actor]:
    """Create one scene containing the controlled pendulum and ball."""

    scene = sdp.create_scene(name)
    result = sdp.prefab.add_to_scene(
        prefab_path=str(resolve_asset(ARTICULATION_PREFAB)),
        root_path=str(resolve_asset_root(ARTICULATION_PREFAB)),
        scene=scene,
    )

    # Add a controller and zero the initial pose and velocity.
    articulation = next(
        actor
        for actor in result.filter(sdp.ActorType.ARTICULATED)
        if actor.get_name() == ARTICULATION_NAME
    )
    params = sdp.PoseControllerParams(NUM_LINKS)
    params.joint_tracking[CART_LINK] = sdp.PoseTrackingParams(
        stiffness=CART_STIFFNESS,
        damping=CART_DAMPING,
    )
    params.joint_tracking[UPPER_ARM_LINK] = sdp.PoseTrackingParams(
        stiffness=HINGE_STIFFNESS,
        damping=HINGE_DAMPING,
    )
    params.joint_tracking[LOWER_ARM_LINK] = sdp.PoseTrackingParams(
        stiffness=0.0,
        damping=LOWER_SWING_DAMPING,
    )
    articulation.add_articulated_pose_controller(params)
    articulation.set_articulated_pose_from_joints(pose=ZERO_TARGET)
    articulation.set_articulated_joint_velocities(velocities=ZERO_TARGET)
    articulation.reset_articulated_target_pose(pose=ZERO_TARGET)

    return scene, articulation


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def joint_kick_target(
    sim_time: float, time_scale: float = NOMINAL_TIME_SCALE
) -> np.ndarray:
    """Return the joint target for the kick trajectory at ``sim_time``.

    ``time_scale`` plays the same nominal trajectory back faster or slower. Once
    the scaled time passes ``MOTION_DURATION`` the target holds its final pose.
    """
    kick_time = sim_time * time_scale
    target = ZERO_TARGET.copy()
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


def _repeat_rollout(
    name: str,
    state_bytes: bytes,
    time_scale: float,
    stop: Event,
) -> None:
    """Own one scene and repeat its restored rollout until disconnection."""
    scene, articulation = create_simulation(name)
    scene.restore_state_from_bytes(state_bytes)
    while not stop.is_set() and sdp.debugger.is_attached():
        for _ in range(ROLLOUT_STEPS):
            joint_target = joint_kick_target(
                scene.get_total_simulation_time(), time_scale
            )
            articulation.set_articulated_target_pose(pose=joint_target)
            scene.step(TIME_STEP)
        scene.restore_state_from_bytes(state_bytes)
    sdp.destroy_scene(scene)


def main() -> None:
    """Fan one parent checkpoint out to two replaying worker scenes."""
    sdp.initialize(num_worker_threads=0)
    source_scene, _ = create_simulation("Cross-Thread Capture/Restore Source")
    state_buffer = sdp.DynamicArrayUint8()
    source_scene.capture_state_to_bytes(state_buffer)
    state_bytes = bytes(state_buffer)

    if not sdp.debugger.attach():
        raise RuntimeError("Mochi Debugger did not attach; see the logged error.")

    stop = Event()
    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            workers = (
                executor.submit(
                    _repeat_rollout,
                    "Cross-Thread Capture/Restore Rollout 1",
                    state_bytes,
                    NOMINAL_TIME_SCALE,
                    stop,
                ),
                executor.submit(
                    _repeat_rollout,
                    "Cross-Thread Capture/Restore Rollout 2",
                    state_bytes,
                    FAST_TIME_SCALE,
                    stop,
                ),
            )
            while sdp.debugger.is_attached():
                source_scene.update_debugger()
                done, _ = wait(workers, timeout=TIME_STEP)
                for worker in done:
                    worker.result()
                if done:
                    break
            stop.set()
            for worker in workers:
                worker.result()
        except BaseException:
            # A sibling may be parked inside a paused scene.step(). Only stopping
            # the server releases it, so the executor join below cannot hang.
            stop.set()
            sdp.get_debug_server().stop()
            raise

    sdp.destroy_scene(source_scene)
    sdp.shutdown()

    print("Simulation complete.")


if __name__ == "__main__":
    main()
