---
title: OSC + JSC Control
sidebar_label: OSC + JSC Control
sidebar_position: 4
---

# OSC + JSC Control

Runs the OSC and JSC examples at the same time on one arm-hand combo: an FR3 arm
with a DG5F short-seed hand at its tip. OSC tracks a planar circle with the wrist
while JSC waves the hand's knuckles.

**Source**: `examples/control/example_osc_jsc_control.py`

## Key Concepts

Two controllers on one actor means two torque vectors on one actor. Both
controllers return a vector sized to the *whole* actor, so combining them is a
sum, but only if each contributes zero on the DOFs it does not own. OSC does that
for free: it zeros every DOF outside the base-to-end-effector chain. JSC does not;
it spans every DOF, so we zero the arm entries of its output by hand before
summing. Each step we build both targets, compute both torques, mask and sum
them, apply the result, and step the simulation.

### Links, joints, and DOF sets

OSC acts on the chain of joints between two links; `fr3_link8` is the arm's tool
flange, where the hand is attached. The arm's joints all share a prefix, which is
how we tell arm DOFs (owned by OSC) from hand DOFs (owned by JSC). Finger 1 is
the thumb, so the non-thumb knuckles are the other four fingers:

```python
np_real = np.float64 if physics.uses_double_precision() else np.float32

ARM_BASE_LINK = "fr3_link0"
ARM_EE_LINK = "fr3_link8"
ARM_JOINT_PREFIX = "fr3_joint"
KNUCKLE_JOINTS = (
    "dg5f_joint_2_2",
    "dg5f_joint_3_2",
    "dg5f_joint_4_2",
    "dg5f_joint_5_3",
)
```

### Initializing the engine and scene

Initialize the physics engine before creating scenes or actors.
`num_worker_threads=0` runs single-threaded; pass `-1` to auto-select. SuperDex
robots use a Z-up convention, so gravity points down the -Z axis:

```python
physics.initialize(num_worker_threads=0)

scene = physics.create_scene("OSC + JSC Control Example")
scene.set_gravity([0, 0, -9.81])

bot_prefab = robotics.load_bot_prefab_from_file(bot_path)
```

Cheap gravity compensation: neither `BASIC_OSC_PD` nor `BASIC_JSC_PD` has a
gravity term, so without it both the arm and the fingers would sag off their
targets. Disable gravity on every link before spawning:

```python
for i in range(len(bot_prefab.links)):
    bot_prefab.links[i].has_gravity = False
```

### Splitting the DOFs

Split the DOFs into the two disjoint sets the controllers own. This numbers them
in bot DOF space: the prefab's joint order, skipping the root joint, and every
moving joint on this bot is a 1-DoF revolute joint, so the indices run
consecutively:

```python
arm_dofs = []
joint_name_to_dof = {}
dof = 0
for i in range(len(bot_prefab.joints)):
    joint = bot_prefab.joints[i]
    if joint.type != physics.ArticulatedJointType.REVOLUTE:
        continue
    joint_name_to_dof[joint.name] = dof
    if joint.name.startswith(ARM_JOINT_PREFIX):
        arm_dofs.append(dof)
    dof += 1

robotics_context = robotics.create_context()
bot = robotics.create_bot(scene, bot_prefab, robotics_context)
bot_actor = bot.get_articulated_actor()

plane_shape = physics.create_plane_shape(normal=[0, 0, 1], distance=0)
scene.create_rigid_actor(name="ground", shape=plane_shape, is_static=True)

num_dofs = bot_actor.get_num_dofs()
all_dof_indices = np.arange(num_dofs, dtype=np.int32)
```

The loop above numbered the joints in bot DOF space, which never includes the root
joint's DOFs. Everything from here on indexes the actor, where the root's DOFs
come first, so shift the indices across that gap. This arm is welded to the world
and contributes none, but reading the count off the actor keeps the mapping
correct for a bot on a free-floating base, which contributes six:

```python
num_root_dofs = bot_actor.get_articulated_shape_info().dof_info[0].get_size()
arm_dof_indices = np.array(arm_dofs, dtype=np.int32) + num_root_dofs
knuckle_dofs = [joint_name_to_dof[name] + num_root_dofs for name in KNUCKLE_JOINTS]
```

### The OSC controller on the arm

`initialize()` resolves the base and end-effector links by name and figures out
which DOFs lie between them, i.e. the arm joints:

```python
osc = bot.create_controller("BASIC_OSC_PD")
bot_name = bot.get_name()
osc.initialize(f"{bot_name}/{ARM_BASE_LINK}", f"{bot_name}/{ARM_EE_LINK}")

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

### The JSC controller on the hand

JSC has no notion of a sub-chain: its gains, its target pose, and its output are
all sized to the full actor, arm DOFs included. The default pose is the JSC hold
target; only the knuckles move off it:

```python
jsc = bot.create_controller("BASIC_JSC_PD")
jsc_params = robotics.ControllerBasicJscPdParams()
jsc_params.kp = np.full(num_dofs, 3.0, dtype=np.float32)
jsc_params.kd = np.full(num_dofs, 0.2, dtype=np.float32)
jsc_params.saturation = np.full(num_dofs, 2.0, dtype=np.float32)
jsc_params.deadband = np.zeros(num_dofs, dtype=np.float32)
jsc.set_params(jsc_params)

default_pose = physics.DynamicArrayReal(num_dofs)
bot_actor.get_articulated_pose(default_pose)
hold_pose = np.array(default_pose, dtype=np_real)
```

### The targets

The FR3 base is an intrinsically fixed `HARD` weld, so `world_from_root` is
constant and can be captured once to convert world-frame targets into the root
frame OSC expects. The circle lies in a horizontal plane 0.45 m above the ground
and 0.5 m in front of the robot base along +X. Keeping the hand pointing straight
down is a 180-degree rotation about world X. Each knuckle oscillates between 0 and
60 degrees, with a 30-degree phase offset between fingers and a 2 s period:

```python
world_from_root = osc.get_current_observations_from_mochi().world_from_root

root_pos = np.asarray(world_from_root.translation, dtype=float)
circle_center = np.array([root_pos[0] + 0.5, root_pos[1], 0.45])
circle_radius = 0.12
circle_period = 4.0

ee_down = physics.Quaternion.rotation_x(np.pi)

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

Each step, build the OSC target (a point on the circle, hand oriented into the
ground, expressed in the actor root frame) and the JSC target (the default pose
everywhere, with the four knuckles driven by a phase-shifted sine). Each
controller reads its own observations off the simulation; the JSC additionally
needs the control period, which cannot be harvested. Both torque vectors span the
whole actor: OSC already zeros everything outside its arm chain, but JSC does
not, so its arm entries are zeroed by hand; with the two now disjoint, the
combined torque is just their sum:

```python
if physics.debugger.attach():
    while physics.debugger.is_attached():
        t = scene.get_total_simulation_time()

        theta = 2.0 * np.pi * t / circle_period
        world_from_target_ee = physics.TransformRT()
        world_from_target_ee.translation = [
            circle_center[0] + circle_radius * np.cos(theta),
            circle_center[1] + circle_radius * np.sin(theta),
            circle_center[2],
        ]
        world_from_target_ee.rotation = ee_down
        target_root_from_ee = world_from_root.inverse() * world_from_target_ee

        target_pose = np.array(hold_pose, dtype=np_real)
        for finger, knuckle_dof in enumerate(knuckle_dofs):
            target_pose[knuckle_dof] = sweep_mid + sweep_amplitude * np.sin(
                2.0 * np.pi * t / sweep_period + finger * finger_phase_offset
            )

        osc_obsv = osc.get_current_observations_from_mochi()
        arm_tau = np.array(
            osc.compute_output(
                osc_obsv,
                robotics.ControllerBasicOscPdTarget(
                    root_from_target_ee=target_root_from_ee
                ),
            ),
            dtype=np.float32,
        )
        jsc_obsv = jsc.get_current_observations_from_mochi()
        jsc_obsv.dt = time_step
        hand_tau = np.array(
            jsc.compute_output(
                jsc_obsv,
                robotics.ControllerBasicJscPdTarget(target_pose=target_pose),
            ),
            dtype=np.float32,
        )

        hand_tau[arm_dof_indices] = 0.0
        bot_actor.set_external_forces_on_dofs(
            dof_indices=all_dof_indices,
            force_values=arm_tau + hand_tau,
        )
        scene.step(time_step)
```

The controllers return read-only views onto their internal buffers, so `np.array`
(a copy) is used here rather than `np.asarray`.

### Teardown

Destroy the bot, then shut the engine down cleanly:

```python
robotics.destroy_bot(scene, bot)
physics.shutdown()
```

## Running

```bash
uv run python superdex_robotics/examples/control/example_osc_jsc_control.py
```
