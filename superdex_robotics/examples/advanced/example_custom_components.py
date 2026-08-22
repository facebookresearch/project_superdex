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

"""Example: custom controller, sensor, and actuator components on a .superdex_bot.

This example defines custom controller, sensor, and actuator classes, then connects
them in a closed sensing-control-actuation loop on a 2-DOF planar arm:
``MyVelocityForceController`` (task space, outputs per-joint velocity),
``MyContactForceSensor`` (contact force on the end-effector), and
``MyVelocityServoActuator`` (per-joint PI velocity servo, outputs torque)::

    trajectory --> [P, task space] --> joint velocity --> [PI, per joint] --> torque
                         ^                                                      |
                         +------ contact force <-- MyContactForceSensor <-------+

The sensor and both actuators are declared on their links in the ``.superdex_bot``
and built by ``create_bot()``, so the script only registers the type names and
looks the components up. Controllers are not directly associated with
``.superdex_bot`` files, so the controller is created in code. Each component
matches its C++ contract: a factory taking ``(actor, param_args)`` -- controllers
take only ``actor`` -- plus a required ``reset()``. The framework calls only
``reset()``; everything else is our own API, driven from the simulation loop.

X stays under velocity control throughout. Z switches to force control on contact
to regulate a constant downward push, and returns to velocity control on liftoff
or lost contact -- see ``MyVelocityForceController`` for the two control laws.

Usage:
    cd <path_to_superdex_robotics>
    python3 examples/advanced/example_custom_components.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import superdex.physics as physics
import superdex.robotics as robotics
from superdex.physics.paths import get_assets_root, resolve_asset

# The build's `real` type. Arrays handed to or filled by the engine must match it,
# or the bindings convert element by element -- and reject out-parameters outright.
np_real = np.float64 if physics.uses_double_precision() else np.float32

# Index into the bot's nested link actors: base_link, link_1, link_2, ball.
END_EFFECTOR_LINK_INDEX = 3

# Component names as declared in the .superdex_bot; these must match. The actuator
# order is DOF order, which run_simulation() relies on when it pairs each servo
# with the controller's per-joint output.
SENSOR_NAME = "force_sensor"
ACTUATOR_NAMES = ("servo_1", "servo_2")

CONTROL_RATE_HZ = 250.0
DUCK_LAMP_ASSET = "prefabs/duck_lamp/duck_lamp_recumbent.mochi_prefab"


def get_default_bot_path() -> str:
    """Resolve path to the default 2-DOF example bot .superdex_bot file."""
    return str(resolve_asset("bots/fun/example_bot_2dof/example_bot_2dof.superdex_bot"))


def get_link_actor(bot_actor, link_index: int):
    """Resolve one of a bot's nested link actors by index."""
    return bot_actor.get_scene().get_actor(
        bot_actor.get_nested_link_actors()[link_index]
    )


def resolve_link_dof(link_actor):
    """Find the articulation a link belongs to and the DOF its joint drives."""
    scene = link_actor.get_scene()
    bot_actor = scene.get_actor(link_actor.get_articulated_actor())
    handles = list(bot_actor.get_nested_link_actors())
    link_index = handles.index(link_actor.get_handle())
    dof_info = bot_actor.get_articulated_shape_info().dof_info[link_index]
    return bot_actor, dof_info.offset


class MyVelocityForceController:
    """Task-space controller with a switchable Z axis.

    Velocity mode applies a P law in task space::

        v_des = x_dot_target + k_p * (x_target - x)     [k_p in 1/s]

    Force mode replaces only the Z row with an admittance law::

        v_z = k_f * (f_z - f_z_target)                  [k_f in (m/s)/N]

    Either way the result is mapped to joint velocities by
    ``q_dot_des = pinv(J_eff) @ v_eff``, where ``J_eff`` is the planar X-Z slice
    of the translation Jacobian (2x2 for this arm) and ``v_eff`` the matching
    rows of ``v_des``. Output is per-joint velocity in rad/s for the servos.

    Force mode is entered when the measured contact force exceeds
    ``f_z_target``, and left when the target clears the measured height by
    ``exit_margin`` or contact is lost entirely. ``f_z_target`` is both the
    entry threshold and the regulated setpoint.
    """

    # The factory receives the robot actor, or None when created without one.
    def __init__(self, actor):
        self.actor = actor
        self.k_p: float = 5.0  # 1/s
        self.k_f: float = 0.05  # (m/s)/N
        self.f_z_target: float = 0.5  # N

        # Exit to velocity mode when the target rises this far above the end-effector.
        self.exit_margin: float = 0.005  # m

        # Discrete state, cleared by reset().
        self.force_mode: bool = False

    # Required by the framework. Clears internal state, leaving params alone.
    def reset(self) -> None:
        self.force_mode = False

    def get_current_observations(self) -> dict[str, np.ndarray]:
        """Sample the state the control law reads out of the simulation."""
        if self.actor is None:
            return None

        link_actor = get_link_actor(self.actor, END_EFFECTOR_LINK_INDEX)

        xforms = physics.DynamicArrayTransformRT(4)
        self.actor.get_articulated_link_transforms(xforms)
        x = np.array(xforms[END_EFFECTOR_LINK_INDEX].translation, dtype=np_real)

        # get_articulated_jacobian() is flat row-major [6, num_dofs] in world
        # frame: rows 0-2 linear, rows 3-5 angular. Keep the linear block.
        num_dofs = self.actor.get_num_dofs()
        jacobian = np.array(link_actor.get_articulated_jacobian(), dtype=np_real)
        J_pos = jacobian.reshape(6, num_dofs)[0:3, :]

        return {"x": x, "J_pos": J_pos}

    def compute_output(self, obsv: dict, target: dict) -> np.ndarray:
        """Map a task-space target to per-joint velocities in rad/s."""
        x_target = np.asarray(target["x"], dtype=np.float64)
        x_dot_target = np.asarray(target.get("x_dot", np.zeros(3)), dtype=np.float64)
        x = np.asarray(obsv["x"], dtype=np.float64)
        J_pos = np.asarray(obsv["J_pos"], dtype=np.float64)

        v_des = x_dot_target + self.k_p * (x_target - x)

        # Expects the caller to have added "f_contact", the world-frame contact
        # force on the end-effector, alongside the observations sampled above.
        f_z = float(np.asarray(obsv["f_contact"])[2])
        self._update_mode(f_z, float(x_target[2]), float(x[2]))
        if self.force_mode:
            v_des[2] = self.k_f * (f_z - self.f_z_target)

        # This arm moves in the X-Z plane and the Y Jacobian row is ~0, so the
        # planar slice keeps the mapping square. pinv degrades gracefully near
        # singularities and is the exact inverse elsewhere.
        J_eff = J_pos[[0, 2], :]
        q_dot_des = np.linalg.pinv(J_eff) @ v_des[[0, 2]]
        return np.asarray(q_dot_des).reshape(-1)

    def _update_mode(self, f_z: float, x_target_z: float, x_z: float) -> None:
        """Enter force mode on contact; leave it on liftoff or lost contact."""
        if not self.force_mode:
            self.force_mode = f_z > self.f_z_target
        elif x_target_z > x_z + self.exit_margin or f_z <= 0.0:
            self.force_mode = False


class MyContactForceSensor:
    """Contact-force sensor on a single link.

    Reports the total contact force applied *to* the link by everything touching
    it, in world frame -- so pressing down on a surface reads positive in Z. The
    query must be registered up front, and its results are only valid after a
    simulation step.
    """

    # The factory receives the link actor (None for an actor-less sensor) and the
    # param string, exactly as a C++ sensor's constructor does. This sensor has
    # nothing to configure, so it ignores param_args.
    def __init__(self, actor, param_args: str):
        self.actor = actor
        if actor is not None:
            actor.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)

    # Required by the framework, even with no state to clear.
    def reset(self) -> None:
        pass

    def compute_signal(self) -> np.ndarray:
        """World-frame contact force [N]. Call only after scene.step()."""
        return np.array(self.actor.get_contact_force_world(), dtype=np_real)


class MyVelocityServoActuator:
    """Per-joint velocity-mode PI servo, in physical units.

    An abstract servo with no register or PWM layer::

        effort = k_p * e + k_i * integral(e),  e = w_traj - w_present

    ``w_traj`` ramps toward the commanded velocity at ``profile_accel`` (0 for no
    smoothing), and the output is clamped to ``+/- effort_limit``. Anti-windup is
    by clamping: the integrator only commits when the output is unsaturated, or
    when the error would pull it back out of saturation.
    """

    # The factory receives the link actor -- an actuator always has one -- and
    # the param string, exactly as a C++ actuator's constructor does.
    def __init__(self, actor, param_args: str):
        assert actor is not None, "MyVelocityServoActuator requires a link actor"
        self.actor = actor
        self.bot_actor, self.dof_index = resolve_link_dof(actor)

        # The bot file supplies param_args as inline JSON; the framework also
        # allows a path to a JSON file, or empty to take the defaults below.
        text = param_args.strip()
        if not text:
            params = {}
        else:
            source = text if text.startswith("{") else Path(text).read_text()
            try:
                params = json.loads(source)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid param_args: {param_args!r}") from e

        self.k_p: float = float(params.get("k_p", 1.0))  # N*m/(rad/s)
        self.k_i: float = float(params.get("k_i", 0.3))  # N*m/rad
        self.effort_limit: float = float(params.get("effort_limit", 1.0))  # N*m
        self.profile_accel: float = float(params.get("profile_accel", 0.0))  # rad/s^2

        # Discrete state, cleared by reset().
        self._integral: float = 0.0
        self._w_traj: float = 0.0

    # Required by the framework. Clears internal state, leaving params alone.
    def reset(self) -> None:
        self._integral = 0.0
        self._w_traj = 0.0

    def get_current_observations(self) -> dict[str, float]:
        """Sample this joint's measured velocity from the simulation."""
        num_dofs = self.bot_actor.get_num_dofs()
        q_dot = np.zeros(num_dofs, dtype=np_real)
        self.bot_actor.get_articulated_joint_velocities(out_velocities=q_dot)
        return {
            "w_present": float(q_dot[self.dof_index]),
            "w_traj": float(self._w_traj),
        }

    def compute_effort(self, obsv: dict, w_goal: float, dt: float) -> float:
        """Joint torque [N*m] tracking ``w_goal`` [rad/s]."""
        w_present = float(obsv["w_present"])
        self._w_traj = self._ramp_toward(float(w_goal), dt)

        e = self._w_traj - w_present
        integral_next = self._integral + e * dt
        effort_unsat = self.k_p * e + self.k_i * integral_next
        effort = float(np.clip(effort_unsat, -self.effort_limit, self.effort_limit))

        # Clamping anti-windup: commit only if unsaturated, or if the error is
        # already driving the output back out of saturation.
        if effort == effort_unsat or e * effort_unsat < 0:
            self._integral = integral_next
        return effort

    def _ramp_toward(self, goal: float, dt: float) -> float:
        """Rate-limit the velocity setpoint to +/- profile_accel * dt."""
        if self.profile_accel <= 0:
            return goal
        max_step = self.profile_accel * dt
        return self._w_traj + float(np.clip(goal - self._w_traj, -max_step, max_step))


def compute_trajectory_target(t: float) -> dict[str, np.ndarray]:
    """Circle in the X-Z plane: center (-0.3, 0.15), radius 0.08 m, period 3 s."""
    center = np.array([-0.3, 0.0, 0.15])
    radius = 0.08
    omega = 2 * math.pi / 3.0
    theta = omega * t
    return {
        "x": center + radius * np.array([math.cos(theta), 0.0, math.sin(theta)]),
        "x_dot": radius * omega * np.array([-math.sin(theta), 0.0, math.cos(theta)]),
    }


def register_component_types(robotics_context) -> None:
    """Make the three Python classes the factories for their type names.

    Names are global to the context. This must run before ``create_bot()``,
    which constructs the sensor and actuators declared in the ``.superdex_bot``.
    """
    robotics.register_python_controller(
        robotics_context, "MY_VELOCITY_FORCE_CONTROLLER", MyVelocityForceController
    )
    robotics.register_python_sensor(
        robotics_context, "MY_CONTACT_FORCE_SENSOR", MyContactForceSensor
    )
    robotics.register_python_actuator(
        robotics_context, "MY_VELOCITY_SERVO_ACTUATOR", MyVelocityServoActuator
    )


def _one_handle(handles, name: str):
    """Unwrap a find_*_by_name() result that must contain exactly one component."""
    if len(handles) != 1:
        raise RuntimeError(
            f"expected exactly one component named {name!r}, found {len(handles)}"
        )
    return handles[0]


def run_simulation(scene, bot_actor, controller, sensor, actuators) -> None:
    """Drive the sensor -> controller -> actuators loop until the debugger quits."""
    time_step = 1.0 / CONTROL_RATE_HZ
    dof_indices = np.arange(bot_actor.get_num_dofs(), dtype=np.int32)
    contact_force = np.zeros(3, dtype=np_real)

    # Declare the coordinate convention so the debugger renders the scene the
    # right way up: SuperDex is X-forward, Y-left, Z-up. Must precede attach().
    physics.get_debug_server().set_coordinate_space(
        physics.CoordinateSpace(axes=physics.CoordinateSpaceAxes.FLU)
    )

    # The Physics Debugger is a separate desktop app for viewing and interacting
    # with the simulation. attach() returns False if it cannot connect.
    if not physics.debugger.attach():
        return

    while physics.debugger.is_attached():
        t = scene.get_total_simulation_time()

        obsv = controller.get_current_observations()
        obsv["f_contact"] = contact_force
        q_dot_des = controller.compute_output(obsv, compute_trajectory_target(t))

        efforts = np.array(
            [
                actuator.compute_effort(
                    actuator.get_current_observations(), w_goal, dt=time_step
                )
                for actuator, w_goal in zip(actuators, q_dot_des)
            ],
            dtype=np_real,
        )

        # This call replaces the entire external-force set, so every torque
        # contribution has to go through a single call.
        bot_actor.set_external_forces_on_dofs(
            dof_indices=dof_indices, force_values=efforts
        )
        scene.step(time_step)

        # Valid only now that the step has completed.
        contact_force = sensor.compute_signal()


def main() -> None:
    """Load the 2-DOF example bot and simulate it."""
    # Initialize the engine before creating scenes or actors.
    # num_worker_threads=0 runs single-threaded; -1 auto-selects.
    physics.initialize(num_worker_threads=0)

    # SuperDex robots are Z-up, so gravity points down -Z.
    scene = physics.create_scene("Custom Components Example")
    scene.set_gravity([0, 0, -9.81])

    # A .superdex_bot is a self-contained robot description. Loading it returns
    # a "prefab": a reusable template of the robot's links and joints.
    bot_prefab = robotics.load_bot_prefab_from_file(get_default_bot_path())

    # Instantiate the prefab as a live bot. This builds the articulated actor,
    # seeds its default pose, and constructs the sensor and actuators the bot
    # file declares on its links. The context tracks every bot and component.
    robotics_context = robotics.create_context()
    register_component_types(robotics_context)
    bot = robotics.create_bot(scene, bot_prefab, robotics_context)
    bot_actor = bot.get_articulated_actor()

    # Static ground plane for the robot to rest on (normal points up, +Z).
    plane_shape = physics.create_plane_shape(normal=[0, 0, 1], distance=0)
    scene.create_rigid_actor(name="ground", shape=plane_shape, is_static=True)

    # A static prop for the arm to touch.
    physics.prefab.add_to_scene(
        prefab_path=str(resolve_asset(DUCK_LAMP_ASSET)),
        root_path=str(get_assets_root()),
        scene=scene,
        params=physics.prefab.PrefabParams(name="duck_lamp"),
    )

    # Bot files cannot declare controllers yet, so this one is built by hand.
    # Controllers take the articulated actor and no params.
    handle = robotics_context.create_controller(
        "MY_VELOCITY_FORCE_CONTROLLER", None, bot_actor, "my_controller"
    )
    controller = robotics.get_python_controller(robotics_context, handle)

    # The sensor and actuators already exist -- create_bot() built them from the
    # .superdex_bot. Look them up by name, then unwrap to the Python objects.
    sensor = robotics.get_python_sensor(
        robotics_context,
        _one_handle(bot.find_sensors_by_name(SENSOR_NAME), SENSOR_NAME),
    )
    actuators = [
        robotics.get_python_actuator(
            robotics_context, _one_handle(bot.find_actuators_by_name(name), name)
        )
        for name in ACTUATOR_NAMES
    ]

    print(f"Robot: {bot_prefab.name}")
    print(f"  Links:  {len(bot_prefab.links)}")
    print(f"  Joints: {len(bot_prefab.joints)}")
    print(f"  DOFs:   {bot_actor.get_num_dofs()}")

    run_simulation(scene, bot_actor, controller, sensor, actuators)

    # Tear down: destroy the bot, then shut the engine down cleanly.
    robotics.destroy_bot(scene, bot)
    physics.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
