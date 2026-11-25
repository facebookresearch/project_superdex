---
title: IK + Pose Control
sidebar_label: IK + Pose Control
sidebar_position: 3
---

# IK + Pose Control

Drives an FR3 v2 arm around the same horizontal circle as the OSC example, tool
pointing straight down, but reaches it a different way: inverse kinematics picks
the joint angles and Mochi's articulated pose controller holds the arm there.

**Source**: `examples/control/example_ik_pose_control.py`

## Key Concepts

Mochi's IK solver runs on its own scene, so the arm is loaded twice: once into
the simulated scene and once into a kinematic twin that only the solver touches.
Each step we put position and rotation targets on the twin's end-effector, solve
for a joint configuration that reaches them, and hand that configuration to the
pose controller as its target. Only the simulated scene is ever stepped.

### Targets and weights

The IK targets are placed on the FR3's tool flange. A bot's link actors are named
`"<bot_name>/<link_name>"`. The pose controller loads per-joint stiffness and
damping tuned for this arm and shipped alongside it:

```python
ARM_EE_LINK = "fr3_link8"
POSE_CONTROLLER_PARAMS = "bots/arms/fr3_v2/control/fr3_v2_pose.superdex_controller"
```

The objective weights for the two IK targets are in [N/m] and [Nm/rad]. The
solver stops once the objective gradient falls below its absolute tolerance, so
the weights have to be large enough that a millimetre of error still registers;
their relative magnitude sets the position/rotation tradeoff when both cannot be
met. Keeping the end-effector pointing straight down is a half turn about world X
(IK rotation targets are rotation vectors, axis × angle), which flips local +Z to
world -Z:

```python
IK_POSITION_WEIGHT = 1.0e4
IK_ROTATION_WEIGHT = 1.0e2
EE_DOWN_ROTATION_VECTOR = [np.pi, 0.0, 0.0]
```

### Loading the arm

Nothing in this example compensates for gravity, so switching it off on every
link is what lets the arm hold exactly the configuration IK asks for. The arm is
loaded through a helper because it is loaded twice, once per scene:

```python
def create_arm(
    scene: physics.Scene, bot_path: str, robotics_context: robotics.RoboticsContext
) -> robotics.Bot:
    bot_prefab = robotics.load_bot_prefab_from_file(bot_path)
    for i in range(len(bot_prefab.links)):
        bot_prefab.links[i].has_gravity = False
    return robotics.create_bot(scene, bot_prefab, robotics_context)
```

### Initializing the simulated scene

Initialize the physics engine before creating scenes or actors.
`num_worker_threads=0` runs single-threaded; pass `-1` to auto-select. SuperDex
robots use a Z-up convention, so gravity points down the -Z axis:

```python
physics.initialize(num_worker_threads=0)

scene = physics.create_scene("IK + Pose Control Example")
scene.set_gravity([0, 0, -9.81])

robotics_context = robotics.create_context()
bot = create_arm(scene, bot_path, robotics_context)
bot_actor = bot.get_articulated_actor()

plane_shape = physics.create_plane_shape(normal=[0, 0, 1], distance=0)
scene.create_rigid_actor(name="ground", shape=plane_shape, is_static=True)
```

### The IK solver

The solver takes ownership of the scene it is given and reshapes it for
quasistatic solving, so it gets a scene of its own holding nothing but a second
copy of the arm. Resolve the end-effector link before handing the scene over;
targets are addressed by link actor handle:

```python
ik_scene = physics.create_scene("IK Solver Scene")
ik_bot = create_arm(ik_scene, bot_path, robotics_context)
ik_actor = ik_bot.get_articulated_actor()
ik_ee_handle = next(
    handle
    for handle in ik_actor.get_nested_link_actors()
    if ik_scene.get_actor(handle).get_name().endswith(f"/{ARM_EE_LINK}")
)
ik_solver = physics.experimental.create_ik_solver(ik_scene)
```

Create the two targets once and keep the position one: the solver hands back a
constraint whose target is updated in place each step. The rotation target never
changes, so it is set up and left alone. A scratch buffer reads the solved
configuration back out:

```python
ik_position_target = ik_solver.create_position_target(
    ik_ee_handle,
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    IK_POSITION_WEIGHT,
)
ik_solver.create_rotation_target(
    ik_ee_handle,
    [0.0, 0.0, 0.0],
    EE_DOWN_ROTATION_VECTOR,
    IK_ROTATION_WEIGHT,
)

ik_pose = physics.DynamicArrayReal(ik_actor.get_num_dofs())
```

### The pose controller

The pose controller is an implicit PD: instead of handing torques back for you to
apply as external forces, its per-joint stiffness and damping become part of the
system the solver resolves during the step. Being solved rather than applied, it
stays stable at gains an explicit PD could not hold, so `compute_output` applies
the control itself and returns nothing. An articulation may only carry one pose
controller, and installing it on the actor is what `initialize()` does, so the
params are set first:

```python
pose_controller = bot.create_controller("MOCHI_ARTICULATED_POSE")
pose_controller.set_params(
    robotics.ControllerMochiArticulatedPoseParams.load_from_file(
        str(resolve_asset(POSE_CONTROLLER_PARAMS))
    )
)
pose_controller.initialize(True)
```

The target pairs a root transform with the non-root joint DOFs. The FR3 base is
welded, so the root transform is constant and only the DOFs change:

```python
pose_obsv = robotics.ControllerMochiArticulatedPoseObsv()
pose_target = robotics.ControllerMochiArticulatedPoseTarget()
pose_target.world_from_root = bot_actor.get_root_transform()
```

### The circle target

IK targets are given in the world frame, so no conversion is needed, unlike OSC,
which wants them in the actor root frame. The circle lies in a horizontal plane
0.45 m above the ground and 0.5 m in front of the robot base along +X:

```python
root_position = np.asarray(bot_actor.get_root_transform().translation, dtype=float)

circle_center = np.array([root_position[0] + 0.5, root_position[1], 0.45])
circle_radius = 0.12
circle_period = 4.0
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

This example has two scenes, and the debugger starts every scene paused and plays
only the selected one: solving IK steps the IK scene, so nothing moves until both
are running. Press play on the IK scene, then switch to this example's scene and
press play again. Unchecking **Start Paused** when connecting brings both scenes
up already running.

Each step, place the position target on the circle and solve. The solution lands
in the twin's joint pose, which becomes the pose controller's target for the
simulated arm:

```python
if physics.debugger.attach():
    while physics.debugger.is_attached():
        theta = 2.0 * np.pi * scene.get_total_simulation_time() / circle_period
        ik_position_target.set_target_position(
            [
                circle_center[0] + circle_radius * np.cos(theta),
                circle_center[1] + circle_radius * np.sin(theta),
                circle_center[2],
            ]
        )
        ik_solver.solve_ik()

        ik_actor.get_articulated_pose(ik_pose)
        pose_target.pose_dofs = ik_pose
        pose_controller.compute_output(pose_obsv, pose_target)

        scene.step(time_step)
```

### Teardown

Release the IK targets and the twin, then destroy the solver (which destroys the
scene it owns), then the simulated bot and the engine:

```python
ik_solver.clear_position_target(ik_ee_handle)
ik_solver.clear_rotation_target(ik_ee_handle)
robotics.destroy_bot(ik_scene, ik_bot)
physics.experimental.destroy_ik_solver(ik_solver)
robotics.destroy_bot(scene, bot)
physics.shutdown()
```

## Running

```bash
uv run python superdex_robotics/examples/control/example_ik_pose_control.py
```
