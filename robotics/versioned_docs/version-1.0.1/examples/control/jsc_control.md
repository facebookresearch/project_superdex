---
title: JSC Control
sidebar_label: JSC Control
sidebar_position: 1
---

# JSC Control

A basic demonstration of the joint-space PD (JSC) controller. It loads a DG5F
hand, welds it to the world, and holds every joint at its default pose except the
four non-thumb knuckles, which sweep back and forth with a phase-shifted sine.

**Source**: `examples/control/example_jsc_control.py`

## Key Concepts

JSC works in joint space: you give it a target joint pose (one angle per DOF) and
it applies per-joint PD torques to reach it. Each step we build the target pose
(default everywhere, sinusoidal on the knuckles), compute the JSC torques, apply
them to the articulated actor, and step the simulation.

### The knuckle joints

Finger 1 is the thumb. For each of the other four fingers we drive its knuckle
(the metacarpophalangeal flexion joint):

```python
KNUCKLE_JOINTS = (
    "dg5f_joint_2_2",
    "dg5f_joint_3_2",
    "dg5f_joint_4_2",
    "dg5f_joint_5_3",
)
```

The build's `real` type sets the numpy dtype for poses. A pose handed to a
controller `Target` is copied into the `Target`'s own storage, so matching the
dtype here keeps that a straight copy rather than an element-by-element
conversion:

```python
np_real = np.float64 if physics.uses_double_precision() else np.float32
```

### Initializing the engine and scene

Initialize the physics engine before creating scenes or actors.
`num_worker_threads=0` runs single-threaded; pass `-1` to auto-select. SuperDex
robots use a Z-up convention, so gravity points down the -Z axis:

```python
physics.initialize(num_worker_threads=0)

scene = physics.create_scene("JSC Control Example")
scene.set_gravity([0, 0, -9.81])

bot_prefab = robotics.load_bot_prefab_from_file(bot_path)
```

The hand ships with a free (6-DoF) world joint at index 0. Making it a `HARD`
(0-DoF weld) joint fixes the hand rigidly to the world:

```python
bot_prefab.joints[0].type = physics.ArticulatedJointType.HARD
```

### Mapping joints to DOF indices

Map each actuated joint name to its DOF index. DOFs follow the prefab joint
order; after the weld above every remaining moving joint is a 1-DoF revolute
joint, so they get consecutive indices. The weld is also what lets these index
the actor arrays directly: a root joint's DOFs lead the actor's, and a welded
root has none, so nothing is skipped ahead of the first knuckle:

```python
joint_name_to_dof = {}
dof = 0
for i in range(len(bot_prefab.joints)):
    joint = bot_prefab.joints[i]
    if joint.type == physics.ArticulatedJointType.REVOLUTE:
        joint_name_to_dof[joint.name] = dof
        dof += 1
knuckle_dofs = [joint_name_to_dof[name] for name in KNUCKLE_JOINTS]
```

Instantiate the prefab as a live `Bot`. The robotics context tracks every bot and
controller you create:

```python
robotics_context = robotics.create_context()
bot = robotics.create_bot(scene, bot_prefab, robotics_context)
bot_actor = bot.get_articulated_actor()

num_dofs = bot_actor.get_num_dofs()
all_dof_indices = np.arange(num_dofs, dtype=np.int32)
```

### The JSC controller

`create_controller` attaches the controller to the bot. JSC spans every DOF, so
its per-joint parameters are all the same length as the pose: the position gain
`kp` [Nm/rad], the damping gain `kd` [Nms/rad], the `saturation` torque clamp,
and a `deadband`:

```python
jsc = bot.create_controller("BASIC_JSC_PD")
jsc_params = robotics.ControllerBasicJscPdParams()
jsc_params.kp = np.full(num_dofs, 3.0, dtype=np.float32)
jsc_params.kd = np.full(num_dofs, 0.2, dtype=np.float32)
jsc_params.saturation = np.full(num_dofs, 2.0, dtype=np.float32)
jsc_params.deadband = np.zeros(num_dofs, dtype=np.float32)
jsc.set_params(jsc_params)
```

### Building the hold pose and the sweep

Read the hand's default joint pose. `get_articulated_pose` fills a
`DynamicArrayReal`, which matches the engine's float precision; copy it to a numpy
array to build per-step targets:

```python
default_pose = physics.DynamicArrayReal(num_dofs)
bot_actor.get_articulated_pose(default_pose)
hold_pose = np.array(default_pose, dtype=np_real)
```

Each knuckle oscillates between 0 and 60 degrees, with a 30-degree phase offset
between fingers and a 2 s period. The midpoint is 30 degrees so the swing spans
0..60 degrees:

```python
sweep_period = 2.0
sweep_mid = np.radians(30.0)
sweep_amplitude = np.radians(30.0)
finger_phase_offset = np.radians(30.0)
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

Each step, start from the default pose and drive the four knuckles with a
phase-shifted sine, then read this step's robot state off the simulation, supply
the control period (which the harvester cannot know), compute the torques, apply
them, and step:

```python
if physics.debugger.attach():
    while physics.debugger.is_attached():
        t = scene.get_total_simulation_time()
        target_pose = np.array(hold_pose, dtype=np_real)
        for finger, knuckle_dof in enumerate(knuckle_dofs):
            target_pose[knuckle_dof] = sweep_mid + sweep_amplitude * np.sin(
                2.0 * np.pi * t / sweep_period + finger * finger_phase_offset
            )

        obsv = jsc.get_current_observations_from_mochi()
        obsv.dt = time_step
        tau = np.asarray(
            jsc.compute_output(
                obsv,
                robotics.ControllerBasicJscPdTarget(target_pose=target_pose),
            ),
            dtype=np.float32,
        )
        bot_actor.set_external_forces_on_dofs(
            dof_indices=all_dof_indices,
            force_values=tau,
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
uv run python superdex_robotics/examples/control/example_jsc_control.py
```
