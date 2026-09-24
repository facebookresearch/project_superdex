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

"""Hold a paper cup while it is filled with water: scripted grip baselines.

Three hand-written grips drive ``Fr3Revo2FillEnv`` (the arm follows the env's own
script: close, lift, hold while water is poured in):

- ``light``: every fingertip holds a fixed 0.4 N. Fine for an empty cup; a full cup
  slides out.
- ``firm``: every fingertip holds a fixed 3 N. Never slips, but the cup is pressed into
  the opposed thumb hard enough to crush it, even while it is empty.
- ``tactile``: reads the load from the fingertips' tangential (shear) forces, which grow
  with the water's weight, and squeezes only as hard as that load needs (see
  :class:`TactileGrip`), the way the Revo2 Touch sensors let a real hand.

These are baselines to beat, not policies: the env is meant for reinforcement learning.

Usage:
    python run_fr3_revo2_fill.py [--grip tactile|light|firm|all] [--episodes 3] [--render]
"""

from __future__ import annotations

import argparse

import numpy as np
from superdex.lab.gym.envs.robots.fr3_revo2_env import FINGERS
from superdex.lab.gym.envs.robots.fr3_revo2_fill_env import (
    Fr3Revo2FillEnv,
    Fr3Revo2FillEnvCfg,
)

# Hand targets in action units, in action order: thumb metacarpal (kept opposed by the
# env), thumb flexion, index, middle, ring, pinky.
OPEN_HAND = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
CLOSED_HAND = np.array([1.0, 0.9, 0.9, 0.9, 0.9, 0.9])
GRIP_FINGERS = (None, "thumb", "index", "middle", "ring", "pinky")

FAST_CLOSE = 0.08  # action units per step, nothing sensed
SLOW_CLOSE = 0.02  # action units per step, object within proximity range
GAIN = 0.01  # action units per step, per newton of force error
MAX_STEP = 0.02  # action units per step, once touching

GRIPS = {"light": 0.4, "firm": 3.0}


class TactileGrip:
    """Squeeze only as hard as the water's weight needs, read from the fingertips.

    Each pad's shear, projected on gravity's direction in the pad's plane (known from the
    hand's kinematics), is the share of the cup's weight that pad carries. Their sum when
    the hand has just closed is the grasp's own offset; what it gains afterwards is the
    water. Friction must carry that load, so the fingertips' total normal force is kept
    at load / 0.4 (a friction coefficient below any the cup has, for margin). The opposed
    thumb sits at its joint limit and presses back as hard as the four fingers push the
    cup into it (about 3 of their 4 shares), so each finger's target is the total / 7.
    Squeezing harder also adds about 0.8 N of shear along gravity per newton of finger
    target (the cup's tapered wall), which is not load and is subtracted.
    """

    FRICTION = 0.4  # assumed; the cup's is 0.5-0.9
    SHARES = 7.0  # fingertip total normal force per unit of finger target
    SQUEEZE_SHEAR = 0.8  # shear along gravity per newton of finger target
    FLOOR = 0.3  # [N]
    CEILING = 1.6  # [N]
    RISE, FALL = 0.2, 0.01  # largest target change per step [N]

    def __init__(self):
        self.offset: float | None = None
        self.grip = self.FLOOR

    @staticmethod
    def load(info: dict) -> float:
        """Summed shear of the pads along gravity [N]."""
        total = 0.0
        for f in FINGERS:
            c, s = info["tactile_direction"][f]
            gx, gy = info["tactile_gravity_direction"][f]
            total += info["tactile_tangential_force"][f] * (c * gx + s * gy)
        return total

    def targets(self, info: dict) -> dict[str, float]:
        if info["phase"] == 0:  # closing: hold the floor, record the grasp's own offset
            self.offset = self.load(info)
        else:
            squeeze_shear = self.SQUEEZE_SHEAR * (self.grip - self.FLOOR)
            water = max(0.0, self.load(info) - self.offset - squeeze_shear)
            want = self.FLOOR + water / (self.FRICTION * self.SHARES)
            want = min(want, self.CEILING)
            self.grip += np.clip(want - self.grip, -self.FALL, self.RISE)
        return dict.fromkeys(FINGERS, self.grip)


def force_targets(grip, info: dict) -> dict[str, float]:
    """Normal force each fingertip should hold [N]."""
    if isinstance(grip, TactileGrip):
        return grip.targets(info)
    return dict.fromkeys(FINGERS, GRIPS[grip])


def regulate(hand: np.ndarray, targets: dict[str, float], info: dict) -> np.ndarray:
    """One step of per-finger normal-force control on the hand command."""
    hand = hand.copy()
    for j, finger in enumerate(GRIP_FINGERS):
        if finger is None:
            continue
        force = info["tactile_normal_force"][finger]
        if force > 0.0:
            step = GAIN * (targets[finger] - force)
            hand[j] += np.clip(step, -MAX_STEP, MAX_STEP)
        else:
            near = info["tactile_proximity"][finger] > 0.5
            hand[j] += SLOW_CLOSE if near else FAST_CLOSE
    return np.clip(hand, OPEN_HAND, CLOSED_HAND)


def run_episode(env: Fr3Revo2FillEnv, grip: str, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    controller = TactileGrip() if grip == "tactile" else grip
    hand = OPEN_HAND.copy()
    squeeze, tips = [], []
    while True:
        hand = regulate(hand, force_targets(controller, info), info)
        obs, reward, terminated, truncated, info = env.step(
            env.to_action({"hand": hand.astype(np.float32)})
        )
        squeeze.append(info["squeeze"])
        tips.append(sum(info["tactile_normal_force"].values()))
        if terminated or truncated:
            break
    outcome = info.get("terminated_reason") or (
        "held" if info["is_success"] else "slid"
    )
    return {
        "outcome": outcome,
        "time": env.time(),
        "water": info["water_mass"],
        "friction": info["friction"],
        "slip": info["slip"],
        "hold_slip": info["hold_slip"],
        "mean_squeeze": float(np.mean(squeeze)),
        "max_squeeze": float(np.max(squeeze)),
        "mean_tip_sum": float(np.mean(tips)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--grip", choices=("tactile", "light", "firm", "all"), default="all"
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--render", action="store_true", help="open the viewer")
    args = parser.parse_args()

    env = Fr3Revo2FillEnv(
        Fr3Revo2FillEnvCfg(
            hand_side=args.side,
            hand_action="absolute",  # the grips below command targets directly
            render_mode="human" if args.render else None,
        )
    )
    grips = ("light", "firm", "tactile") if args.grip == "all" else (args.grip,)
    print(
        f"{'grip':8s} {'seed':>4s} {'outcome':>8s} {'at [s]':>6s} {'water [g]':>9s} "
        f"{'friction':>8s} {'slip while filling [mm]':>23s} {'squeeze mean/max [N]':>20s}"
    )
    for grip in grips:
        held = 0
        for seed in range(args.episodes):
            r = run_episode(env, grip, seed)
            held += r["outcome"] == "held"
            print(
                f"{grip:8s} {seed:4d} {r['outcome']:>8s} {r['time']:6.1f} "
                f"{1000 * r['water']:9.0f} {r['friction']:8.2f} "
                f"{1000 * r['hold_slip']:23.1f} "
                f"{r['mean_squeeze']:10.1f}/{r['max_squeeze']:4.1f}"
            )
        print(f"{grip:8s} held {held}/{args.episodes}\n")
    env.close()


if __name__ == "__main__":
    main()
