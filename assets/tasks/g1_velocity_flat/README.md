# Unitree G1 velocity tracking on flat ground

After unitree_rl_mjlab @1425b15 (Apache-2.0) `Unitree-G1-Flat`. The bot and where its constants come from:
`bots/humanoids/unitree_g1/README.md`.

| File | What |
|---|---|
| `g1_velocity.superdex_scene` | The G1 at mjlab's home pose (pelvis z 0.8) with g1_constants.py's actuators as mjlab-style position actuators (`LAB_JOINT_ACTUATORS`, written by the bot's `generate.py`); slot `robot` offers `g1` (sphere feet) and `g1_box_feet` |
| `scene/g1_flat.mochi_scene` | A ground plane, gravity 9.81 (mjlab's), the G1 collider layers touching only the ground, a 1 mm contact smoothing half-band on the plane |
| `g1_velocity.superdex_task` | No task objects, 200 Hz, 20 s |
| `g1_velocity_flat.superdex_lab` | Written by `generate_labs.py`: mjlab's MDP without the angular momentum penalty, the torso COM DR and the self-collision penalty; feet-only contact |
| `g1_velocity_flat_mjlab.superdex_lab` | The same plus the angular momentum penalty and the torso COM DR. All of mjlab's MDP except its collisions: upstream's `FULL_COLLISION` (body-ground and self contact) and the self-collision penalty are missing; only the feet touch the ground (see the bot README's Layers) |
| `g1_velocity_flat_mjlab_full_collision.superdex_lab` | `g1_velocity_flat_mjlab` on `g1_velocity_full_collision.superdex_scene`: upstream's FULL_COLLISION (body-ground and self contact; see the bot README). The self-collision penalty is not in it yet |
| `g1_velocity_flat_mjlab_full_collision_clip.superdex_lab` | The clip lab (23 s schedule) on the full-collision scene |
| `g1_velocity_full_collision.superdex_scene`, `scene/g1_flat_full_collision.mochi_scene` | Written by the bot's `generate.py`: the full-collision G1 bot as `g1`, on the ground plane without the G1 layer filter |
| `g1_velocity_flat_mjlab_clip.superdex_lab` | `g1_velocity_flat_mjlab` through a fixed 23 s command schedule (stand, walk, turn, arc, sidestep), for clips |
| `g1_velocity_flat_mjlab_clip_rules.superdex_lab` | The same through the overnight run's pre-registered clip commands: vx 0.5, vx 1.0, vy 0.5 m/s, yaw rate 0.5 rad/s, 5 s each |
| `g1_velocity_flat_mjlab_eval.superdex_lab` | `g1_velocity_flat_mjlab` plus records of the command and the pelvis velocities, for tracking errors |
| `g1_velocity_flat.ppo.superdex_learn` | mjlab's rl_cfg.py runner, for both labs |
| `test/g1_stand.superdex_lab` | The home pose held with zero actions under mjlab's gains |
| `test/g1_stand_stiff.superdex_lab` | The same with every actuator 30x stiffer and more damped |

## The stand test (`superdex_lab/tasks/locomotion:locomotion_test`)

- **mjlab's gains do not hold the G1 up**, in MuJoCo as here: at the home pose the PD is softer than gravity's pull, and
  with zero action the robot falls forward within 1.5 s. `superdex_lab/tasks/locomotion/tools/mujoco_stand.py` runs the same model in
  MuJoCo (g1.xml, g1_constants.py's actuators, feet-only collision, friction 0.6, 200 Hz). SuperDex Lab's fall follows
  it: at 0.2, 0.6, 1.0 and 1.2 s the pelvis is within 2 cm and 2° of MuJoCo's (x 0.019/0.136/0.349/0.539 m, z
  0.779/0.758/0.675/0.548 m, tilt 2.6/1.6/18.1/42.6°), with the ground carrying m g while both feet are flat. In MuJoCo,
  2x, 3x and 5x gains fall too (and 10x diverges at 200 Hz).
- **With 30x gains it stands for 10 s** (both foot models): drift under 1 mm, tilt 0.3°, pelvis speed RMS 0.7 mm/s,
  ground load m g (327.1 N, spread 0.1 N), no fault; the sphere feet sink 0.8 mm.

## License

After unitree_rl_mjlab @1425b15 (Apache License 2.0; `LICENSE` here is its license file,
unchanged): the labs, learn file and scene port its `Unitree-G1-Flat` task configuration
(velocity_env_cfg.py, config/g1/env_cfgs.py, rl_cfg.py). The scene's geometry is a ground plane;
the robot and its license notes: `bots/humanoids/unitree_g1/README.md`.
