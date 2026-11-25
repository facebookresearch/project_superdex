---
title: OSC Control
sidebar_label: OSC Control
sidebar_position: 2
---

# OSC Control

A basic demonstration of the operational-space PD (OSC) controller. It loads an
FR3 v2 arm and drives the end-effector around a horizontal circle while keeping
the tool pointing straight down.

**Source**: `examples/control/example_osc_control.py`

## Key Concepts

OSC works in Cartesian task space: you give it a target end-effector pose and it
solves for the arm joint torques that move the end-effector there. Each step we
build a target pose (a point on the circle, tool pointing down), compute the OSC
torques, apply them to the articulated actor, and step the simulation.

### The controlled chain

The OSC controller acts on the chain of joints between two links on the FR3 arm.
A bot's link actors are named `"<bot_name>/<link_name>"`, so these are prefixed at
runtime with `bot.get_name()`:

```python
ARM_BASE_LINK = "fr3_link0"
ARM_EE_LINK = "fr3_link8"
```

### Initializing the engine and scene

Initialize the physics engine before creating scenes or actors.
`num_worker_threads=0` runs single-threaded; pass `-1` to auto-select. SuperDex
robots use a Z-up convention, so gravity points down the -Z axis:

```python
physics.initialize(num_worker_threads=0)

scene = physics.create_scene("OSC Control Example")
scene.set_gravity([0, 0, -9.81])

bot_prefab = robotics.load_bot_prefab_from_file(bot_path)
```

### Cheap gravity compensation

`BASIC_OSC_PD` is a pure task-space PD with no gravity term, so if the arm has to
fight gravity it sags off the target. Disabling gravity on every link before
spawning makes holding the default end-effector frame (nearly) load-free:

```python
for i in range(len(bot_prefab.links)):
    bot_prefab.links[i].has_gravity = False

robotics_context = robotics.create_context()
bot = robotics.create_bot(scene, bot_prefab, robotics_context)
bot_actor = bot.get_articulated_actor()

plane_shape = physics.create_plane_shape(normal=[0, 0, 1], distance=0)
scene.create_rigid_actor(name="ground", shape=plane_shape, is_static=True)

num_dofs = bot_actor.get_num_dofs()
all_dof_indices = np.arange(num_dofs, dtype=np.int32)
```

### The OSC controller

`create_controller` attaches the controller to the bot. `initialize()` resolves
the base and end-effector links by name and figures out which DOFs lie between
them (the arm joints):

```python
osc = bot.create_controller("BASIC_OSC_PD")
bot_name = bot.get_name()
osc.initialize(f"{bot_name}/{ARM_BASE_LINK}", f"{bot_name}/{ARM_EE_LINK}")
```

Task-space PD gains are set for position (`_p`) and rotation (`_r`). Error-magnitude
normalization (on by default) clamps how far the target may pull before the force
saturates, so the caps `max_translation_error` [m] and `max_rotation_error` [rad]
must be positive:

```python
osc_params = osc.get_params()
osc_params.kp_p = 900.0
osc_params.kd_p = 75.0
osc_params.kp_r = 30.0
osc_params.kd_r = 3.0
osc_params.max_translation_error = 0.05
osc_params.max_rotation_error = 0.4
osc_params.b_apply_max_osc_torque_normalization = True
osc.set_params(osc_params)
```

### The circle target

Read the arm's default frames once. The FR3 bot file has an intrinsically fixed
base (its root joint is a `HARD` weld by default), so `world_from_root` is
constant and can be captured once to convert world-frame targets into the root
frame OSC expects:

```python
obsv = osc.get_current_observations_from_mochi()
world_from_root = obsv.world_from_root
```

The circle lies in a horizontal plane (its normal is the world up-axis) 0.45 m
above the ground and 0.5 m in front of the robot base along +X. Keeping the
end-effector pointing straight down (its z-axis into the ground) is a 180-degree
rotation about world X, which flips local +Z to world -Z:

```python
root_pos = np.asarray(world_from_root.translation, dtype=float)
circle_center = np.array([root_pos[0] + 0.5, root_pos[1], 0.45])
circle_radius = 0.12
circle_period = 4.0

ee_down = physics.Quaternion.rotation_x(np.pi)
```

### The control loop

Declare the scene's coordinate convention so the debugger renders it the right
way up (FLU: X-forward, Y-left, Z-up). This must come before `attach()`, which
starts the server:

```python
time_step = 1.0 / 200.0

physics.get_debug_server().set_coordinate_space(
    physics.CoordinateSpace(axes=physics.CoordinateSpaceAxes.FLU)
)
```

Each step, build the target end-effector pose in the world frame (a point on the
circle, oriented so the EE z-axis points into the ground) and convert it into the
actor root frame OSC expects. OSC returns a full-length torque vector for the
actor, so it can be applied directly:

```python
if physics.debugger.attach():
    while physics.debugger.is_attached():
        theta = 2.0 * np.pi * scene.get_total_simulation_time() / circle_period
        world_from_target_ee = physics.TransformRT()
        world_from_target_ee.translation = [
            circle_center[0] + circle_radius * np.cos(theta),
            circle_center[1] + circle_radius * np.sin(theta),
            circle_center[2],
        ]
        world_from_target_ee.rotation = ee_down

        target_root_from_ee = world_from_root.inverse() * world_from_target_ee

        obsv = osc.get_current_observations_from_mochi()
        arm_tau = np.asarray(
            osc.compute_output(
                obsv,
                robotics.ControllerBasicOscPdTarget(
                    root_from_target_ee=target_root_from_ee
                ),
            ),
            dtype=np.float32,
        )
        bot_actor.set_external_forces_on_dofs(
            dof_indices=all_dof_indices,
            force_values=arm_tau,
        )
        scene.step(time_step)
```

### Teardown

Destroy the bot, then shut the engine down cleanly:

```python
robotics.destroy_bot(scene, bot)
physics.shutdown()
```

## Running

```bash
uv run python superdex_robotics/examples/control/example_osc_control.py
```
