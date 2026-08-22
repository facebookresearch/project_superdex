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

"""Capture and repeatedly restore the initial state of one scene."""

from __future__ import annotations

import math

import superdex.physics as physics
from superdex.physics import Actor, Scene
from superdex.physics.paths import resolve_asset

TIME_STEP = 1.0 / 60.0  # [s]
RESTORE_INTERVAL = 3.0  # [s]
DUCK_SCALE = 0.5
DROP_HEIGHT = 3.0 * DUCK_SCALE  # [m]
INITIAL_ROTATION_ANGLE = math.pi / 2  # [rad]


def create_state_capture_simulation() -> tuple[Scene, Actor]:
    """Create a scene with a rigid duck falling onto a static plane."""
    scene = physics.create_scene("State Capture Scene")

    plane_shape = physics.create_plane_shape(normal=[0, 1, 0], distance=0)
    scene.create_rigid_actor(
        name="ground",
        shape=plane_shape,
        is_static=True,
    )

    duck_shape = physics.load_shape_from_file(
        file_path=str(resolve_asset("duck/duck_coarse.mochi.h5")),
        bake_scale=[DUCK_SCALE] * 3,
    )
    duck = scene.create_rigid_actor(
        name="duck",
        shape=duck_shape,
        world_from_local=physics.TransformRT(
            rotation=physics.Quaternion.rotation_x(INITIAL_ROTATION_ANGLE),
            translation=[0, DROP_HEIGHT, 0],
        ),
    )

    return scene, duck


def main() -> None:
    """Run a falling duck and restore its initial state every three seconds."""
    physics.initialize(num_worker_threads=0)
    try:
        scene, _ = create_state_capture_simulation()

        # Capture only after the scene topology is complete. Keeping this handle alive
        # lets every restore return to the same initial state.
        initial_state = scene.capture_state()

        if not physics.debugger.attach():
            return

        while physics.debugger.is_attached():
            scene.step(TIME_STEP)
            if scene.get_total_simulation_time() >= RESTORE_INTERVAL:
                scene.restore_state(initial_state, release_immediately=False)
    finally:
        physics.shutdown()

    print("Simulation complete.")


if __name__ == "__main__":
    main()
