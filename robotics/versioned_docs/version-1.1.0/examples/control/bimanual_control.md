---
title: Bimanual Control
sidebar_label: Bimanual Control
sidebar_position: 5
---

# Bimanual Control

Runs one operational-space PD (OSC) controller per arm on an OpenArm V2.0: a
fixed torso with a 7-DoF arm and a two-finger gripper on each side. Each gripper
traces a vertical circle around its own starting position, and the two circles are
swept in opposite directions, so the hands mirror each other. Meanwhile a
joint-space PD (JSC) pinches all four fingers open and closed, five times per arm
revolution.

**Source**: `examples/control/example_bimanual_control.py`

## Key Concepts

Driving two arms needs no coordination machinery: the two OSC chains are disjoint,
one arm each, so every controller gets its own target and its own torque vector
and the vectors simply add. The JSC spans the whole actor, so only its finger
entries are kept.

### Sides, links, and finger joints

Each OSC controller acts on the chain of joints between a base and an
end-effector link. The base is the arm's mount on the torso and the end-effector
is the gripper's root, so the chain spans exactly that arm's seven revolute
joints. A bot's link actors are named `"<bot_name>/<link_name>"`, so these are
prefixed at runtime. The two gripper finger joints per side are the only DOFs
outside the two OSC chains, and each pinch sweeps a fraction of finger travel so
it stops just short of the hard stops:

```python
np_real = np.float64 if physics.uses_double_precision() else np.float32

SIDES = ("left", "right")
ARM_BASE_LINK = "openarm_{side}_base_link"
ARM_EE_LINK = "openarm_{side}_ee_base_link"
FINGER_JOINT_TOKEN = "finger_joint"
FINGER_TRAVEL_FRACTION = 0.9
```

### Initializing the engine and scene

Initialize the physics engine before creating scenes or actors.
`num_worker_threads=0` runs single-threaded; pass `-1` to auto-select. SuperDex
robots use a Z-up convention, so gravity points down the -Z axis:

```python
physics.initialize(num_worker_threads=0)

scene = physics.create_scene("Bimanual Control Example")
scene.set_gravity([0, 0, -9.81])

bot_prefab = robotics.load_bot_prefab_from_file(bot_path)
```

Cheap gravity compensation: neither controller has a gravity term, so the arms
would otherwise sag off their targets. Disable gravity on every link before
spawning:

```python
for i in range(len(bot_prefab.links)):
    bot_prefab.links[i].has_gravity = False
```

### Finding the finger DOFs

Find the finger DOFs and how far each one may travel. This numbers them in bot DOF
space: the prefab's joint order, skipping the root joint, and every moving joint
on this bot is a 1-DoF revolute joint, so the indices run consecutively. Joint
limits are stored per axis, so we read the component the joint actually rotates
about:

```python
finger_dofs = []
finger_travel = []
dof = 0
for i in range(len(bot_prefab.joints)):
    joint = bot_prefab.joints[i]
    if joint.type != physics.ArticulatedJointType.REVOLUTE:
        continue
    if FINGER_JOINT_TOKEN in joint.name:
        axis = int(np.argmax(np.abs(np.asarray(joint.axis, dtype=float))))
        finger_dofs.append(dof)
        finger_travel.append(
            max(
                abs(float(joint.min_limit[axis])), abs(float(joint.max_limit[axis]))
            )
        )
    dof += 1
finger_travel = np.array(finger_travel, dtype=np.float32)

robotics_context = robotics.create_context()
bot = robotics.create_bot(scene, bot_prefab, robotics_context)
bot_actor = bot.get_articulated_actor()

plane_shape = physics.create_plane_shape(normal=[0, 0, 1], distance=0)
scene.create_rigid_actor(name="ground", shape=plane_shape, is_static=True)

num_dofs = bot_actor.get_num_dofs()
all_dof_indices = np.arange(num_dofs, dtype=np.int32)
```

The loop above numbered the fingers in bot DOF space, which never includes the
root joint's DOFs. Everything from here on indexes the actor, where the root's
DOFs come first, so shift the indices across that gap. This torso is welded to the
world and contributes none, but reading the count off the actor keeps the mapping
correct for a bot on a free-floating base, which contributes six:

```python
num_root_dofs = bot_actor.get_articulated_shape_info().dof_info[0].get_size()
finger_dof_indices = np.array(finger_dofs, dtype=np.int32) + num_root_dofs
```

### One OSC controller per arm

`initialize()` resolves the base and end-effector links by name and figures out
which DOFs lie between them. The two chains are disjoint, so the two controllers
never contend for a DOF:

```python
bot_name = bot.get_name()
arm_oscs = []
for side in SIDES:
    osc = bot.create_controller("BASIC_OSC_PD")
    osc.initialize(
        f"{bot_name}/{ARM_BASE_LINK.format(side=side)}",
        f"{bot_name}/{ARM_EE_LINK.format(side=side)}",
    )
    osc_params = osc.get_params()
    osc_params.kp_p = 900.0
    osc_params.kd_p = 75.0
    osc_params.kp_r = 5.0
    osc_params.kd_r = 0.05
    osc_params.max_translation_error = 0.05
    osc_params.max_rotation_error = 0.4
    osc_params.b_apply_max_osc_torque_normalization = True
    osc.set_params(osc_params)
    arm_oscs.append(osc)
```

### The JSC controller pinching the fingers

JSC has no notion of a sub-chain: its gains, its target pose, and its output are
all sized to the full actor. Only its finger entries are harvested below:

```python
jsc = bot.create_controller("BASIC_JSC_PD")
jsc_params = robotics.ControllerBasicJscPdParams()
jsc_params.kp = np.full(num_dofs, 0.5, dtype=np.float32)
jsc_params.kd = np.full(num_dofs, 0.005, dtype=np.float32)
jsc_params.saturation = np.full(num_dofs, 0.5, dtype=np.float32)
jsc_params.deadband = np.zeros(num_dofs, dtype=np.float32)
jsc.set_params(jsc_params)

default_pose = physics.DynamicArrayReal(num_dofs)
bot_actor.get_articulated_pose(default_pose)
hold_pose = np.array(default_pose, dtype=np_real)
```

Each finger swings between fully closed (0) and its open limit. The limits record
only the magnitude of the travel, so the direction comes from the pose the gripper
spawned in: both fingers of a hand share a sign there, which is what closes them
symmetrically onto each other:

```python
finger_open = (
    np.sign(hold_pose[finger_dof_indices]) * FINGER_TRAVEL_FRACTION * finger_travel
)
```

### The circle targets

The torso is welded to the world, so `world_from_root` is constant and can be
captured once to convert world-frame targets into the root frame OSC expects. Each
circle is centered on its own gripper's spawn pose and the orientation is held at
its spawn value, so the arms only have to translate:

```python
start_obsv = [osc.get_current_observations_from_mochi() for osc in arm_oscs]
root_from_world = start_obsv[0].world_from_root.inverse()

ee_start_positions = [
    np.asarray(obsv.world_from_ee_link.translation, dtype=float)
    for obsv in start_obsv
]
ee_start_rotations = [obsv.world_from_ee_link.rotation for obsv in start_obsv]
```

The circles lie in the vertical plane spanned by world Y (across) and Z (up),
centered on each gripper's spawn position. Sweeping the two arms in opposite
directions mirrors them: the hands rise and fall together while swinging apart and
back together. The pinch runs five times per arm revolution:

```python
circle_radius = 0.12
circle_period = 4.0
sweep_directions = (1.0, -1.0)

pinch_period = 0.8
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

Each step, sweep the fingers closed → open → closed with the JSC (its target is
full length, so start from the spawn pose and overwrite the fingers), then keep
only the JSC's finger entries since the arm DOFs belong to the OSCs. Each OSC
output is also full length but already zero outside its own arm chain, so the
three contributions just add up:

```python
if physics.debugger.attach():
    while physics.debugger.is_attached():
        t = scene.get_total_simulation_time()

        target_pose = np.array(hold_pose, dtype=np_real)
        target_pose[finger_dof_indices] = (
            0.5 * finger_open * (1.0 - np.cos(2.0 * np.pi * t / pinch_period))
        )

        jsc_obsv = jsc.get_current_observations_from_mochi()
        jsc_obsv.dt = time_step
        jsc_tau = np.asarray(
            jsc.compute_output(
                jsc_obsv,
                robotics.ControllerBasicJscPdTarget(target_pose=target_pose),
            ),
            dtype=np.float32,
        )
        tau = np.zeros(num_dofs, dtype=np.float32)
        tau[finger_dof_indices] = jsc_tau[finger_dof_indices]

        for osc, start_pos, start_rot, direction in zip(
            arm_oscs, ee_start_positions, ee_start_rotations, sweep_directions
        ):
            theta = direction * 2.0 * np.pi * t / circle_period
            world_from_target_ee = physics.TransformRT(
                rotation=start_rot,
                translation=[
                    start_pos[0],
                    start_pos[1] + circle_radius * np.sin(theta),
                    start_pos[2] + circle_radius * np.cos(theta),
                ],
            )
            tau += np.asarray(
                osc.compute_output(
                    osc.get_current_observations_from_mochi(),
                    robotics.ControllerBasicOscPdTarget(
                        root_from_target_ee=root_from_world * world_from_target_ee
                    ),
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
uv run python superdex_robotics/examples/control/example_bimanual_control.py
```
