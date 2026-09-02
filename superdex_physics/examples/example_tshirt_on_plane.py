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

"""Example: T-shirt on Plane

Simulates a t-shirt cloth mesh with self-contact falling onto a static ground
plane, modeled as an experimental shell actor.

It demonstrates:
- Loading a triangle-mesh garment from a file
- Building shell material parameters from 3D isotropic elasticity and a thickness
- Creating a shell actor with point-cloud self-contact
"""

import superdex.physics as sdp
from superdex.physics import Actor, Scene
from superdex.physics.paths import resolve_asset


YOUNGS_MODULUS = 1e5  # [Pa]
POISSON_RATIO = 0.25
DENSITY = 1e3  # [kg/m^3]
THICKNESS = 2e-3  # [m]
INITIAL_X_TRANSLATION = -0.5  # [m]
INITIAL_HEIGHT = 0.1  # [m]
CONTACT_RADIUS = 1.5e-2  # [m]
TIME_STEP = 1.0 / 60.0  # [s]


def create_tshirt_on_plane_simulation() -> tuple[Scene, Actor, Actor]:
    """Create a t-shirt-on-plane shell physics simulation.

    Note: physics.initialize() must be called before this function.

    Returns:
        tuple: (scene, shell_actor, rigid_plane_actor)
    """
    # Create scene
    scene = sdp.create_scene("T-shirt on Plane Scene")

    # Configure the solver for interactive shell contact.
    solver_params = scene.get_solver_params()
    solver_params.non_linear_solver.max_iter = 2
    solver_params.non_linear_solver.line_search_type = sdp.LineSearchType.WOLFE_STRONG
    solver_params.experimental_eval.implicit_normal_force_for_dissipation = True
    scene.set_solver_params(solver_params)

    # Static ground plane at y = 0.
    plane_shape = sdp.create_plane_shape(normal=[0, 1, 0], distance=0)
    rigid_plane_actor = scene.create_rigid_actor(
        name="ground", shape=plane_shape, is_static=True
    )

    # Load the t-shirt mesh.
    shape_path = str(resolve_asset("garments/tshirt_visual_subdiv_2.mochi.h5"))
    shape = sdp.load_shape_from_file(shape_path)

    # Convert familiar 3D material properties into thickness-integrated shell
    # stiffnesses and an areal density.
    material = sdp.experimental.shell_material_params_from3d_isotropic(
        YOUNGS_MODULUS, POISSON_RATIO, DENSITY, THICKNESS
    )

    # Create the shell actor with point-cloud self-contact.
    shell_params = sdp.experimental.ShellActorParams(
        name="T-shirt",
        shape=shape,
        material=material,
        world_from_local=sdp.TransformRT(
            translation=[INITIAL_X_TRANSLATION, INITIAL_HEIGHT, 0]
        ),
    )
    shell_params.point_cloud_collider.radius = CONTACT_RADIUS
    shell_params.point_cloud_collider.self_contact = True

    shell_actor = sdp.experimental.create_shell_actor(scene, shell_params)

    return scene, shell_actor, rigid_plane_actor


def main():
    """Main function that runs the interactive t-shirt-on-plane simulation."""
    # Use a default worker-thread count based on the CPU hardware.
    sdp.initialize(num_worker_threads=-1)

    # Create simulation
    scene, _, _ = create_tshirt_on_plane_simulation()

    # Launch and attach the remote debugger for visualization and interaction.
    if not sdp.debugger.attach():
        sdp.shutdown()
        return

    # Simulate until the debugger detaches
    while sdp.debugger.is_attached():
        scene.step(TIME_STEP)

    # Shut down SuperDex Physics.
    sdp.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
