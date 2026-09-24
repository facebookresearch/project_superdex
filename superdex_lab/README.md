# SuperDex Lab

[SuperDex Lab](https://projectsuperdex.com/lab/) provides Gymnasium-based
environment and benchmarking surfaces for Project SuperDex.

Provides the `superdex.lab` package: reinforcement-learning environments, reference
benchmark tasks, and the vectorized runners built on `superdex-robotics`.

```bash
pip install superdex-lab
```

The benchmark assets are not included with `pip install superdex-lab`. Clone the
[Project SuperDex repository](https://github.com/facebookresearch/project_superdex)
and set `SUPERDEX_ASSETS_PATH=<path-to-project_superdex>/assets` before running a
benchmark environment. See the
[setup guide](https://projectsuperdex.com/lab/docs/superdex_gym/setup/) for complete
instructions.

See the [repository README](https://github.com/facebookresearch/project_superdex#readme)
for the full list of SuperDex distributions.

## FR3 + BrainCo Revo2 with tactile sensing

`superdex_gym/Fr3Revo2-v0` is a grasp-and-lift task: a Franka Research 3 on a desk,
carrying a BrainCo Revo2 hand, lifts a paper cup. Each fingertip has a Revo2 Touch-style
sensor (`superdex.lab.sensors.tactile`): normal force, tangential force and direction,
and proximity, at 0-25 N with 0.1 N resolution and 0-1 cm proximity range. Variants:
`fr3_revo2_left`, `fr3_revo2_box` (a box instead of the cup), `fr3_revo2_randomized`
(sensor noise and wider randomization). It runs on the published wheels plus this
source tree:

```bash
uv venv && uv pip install superdex
uv pip install --no-sources -e superdex_lab   # the env is newer than the superdex-lab wheel
export SUPERDEX_ASSETS_PATH=$PWD/assets

uv run --no-project superdex_lab/apps/envs/run_fr3_revo2_grasp.py --render  # scripted lift
uv run --no-project superdex_lab/apps/envs/run_sample.py fr3_revo2          # random actions
```

```python
import gymnasium as gym
from superdex.lab.gym.utils.env_discovery import register_all_envs

register_all_envs()
env = gym.make("superdex_gym/Fr3Revo2-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
info["tactile_normal_force"]  # {"thumb": ..., "index": ..., ...} [N]
info["tactile_proximity"]     # 0 (nothing within 1 cm) .. 1 (touching)
```

Actions are 7 arm joint-target increments and 6 absolute targets for the Revo2's actuated
joints. Observations include joint states, the object pose, and per fingertip
`[normal, tangential, cos(direction), sin(direction), proximity]`. See the module
docstring of `superdex/lab/gym/envs/robots/fr3_revo2_env.py` for the full layout, and
`assets/bots/hands/revo2/README.md` for the hand and sensor model.

The scripted demo closes each finger fast until proximity senses the cup, then regulates
each fingertip's normal force to 3 N, the kind of tactile grip the Revo2 Touch is built
for. It is a baseline, not a tuned policy (about 5/5 lifts per hand on its seeds).

In the default viewer, render models are drawn with their base material colors; heavy
grasping contact simulates at roughly 4-5x slower than real time on one core (free
motion runs faster than real time).
