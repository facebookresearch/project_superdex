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

"""Writes the G1 velocity labs and learn file from unitree_rl_mjlab's Unitree-G1-Flat (Apache-2.0,
@1425b15): src/tasks/velocity/velocity_env_cfg.py, config/g1/env_cfgs.py (the flat config: the rough
one without terrain, height scan and terrain curriculum) and config/g1/rl_cfg.py.

Run from this directory: ``python3 generate_labs.py``. It needs the bot written first
(``bots/humanoids/unitree_g1/generate.py``): it reads the bot's joint order.

- ``g1_velocity_flat.superdex_lab``: mjlab's MDP with what SuperDex Lab supports today; it leaves
  out the whole-body angular momentum penalty, the torso center-of-mass DR and the self-collision
  penalty (see README.md).
- ``g1_velocity_flat_mjlab.superdex_lab``: the same plus the angular momentum penalty and the torso
  center-of-mass DR: all of mjlab's MDP but the self-collision penalty.
- ``g1_velocity_flat.ppo.superdex_learn``: rl_cfg.py's runner, for both labs.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BOT = _HERE.parents[1] / "bots/humanoids/unitree_g1/unitree_g1.superdex_bot"

_COMMAND_STAGE_STEP = 5000 * 24
"""velocity_env_cfg.py's commands_vel curriculum: the second stage at common step 5000 x 24."""

# env_cfgs.py's variable_posture std per joint pattern, by speed regime.
_STD_STANDING = {r".*": 0.05}
_STD_WALKING = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
}
_STD_RUNNING = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
}

# g1_constants.py's actuators: (patterns, armature, effort limit); G1_ACTION_SCALE = 0.25 effort / kp.
_FREQ = 10 * 2.0 * 3.1415926535


def _reflected(
    rotor: tuple[float, float, float], gears: tuple[float, float, float]
) -> float:
    return rotor[0] * (gears[1] * gears[2]) ** 2 + rotor[1] * gears[2] ** 2 + rotor[2]


_A5020 = _reflected((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16))
_ACTUATORS = [
    (
        (r".*_elbow_joint", r".*_shoulder_.*_joint", r".*_wrist_roll_joint"),
        _A5020,
        25.0,
    ),
    (
        (r".*_hip_pitch_joint", r".*_hip_yaw_joint", r"waist_yaw_joint"),
        _reflected((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22)),
        88.0,
    ),
    (
        (r".*_hip_roll_joint", r".*_knee_joint"),
        _reflected((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5)),
        139.0,
    ),
    (
        (r".*_wrist_pitch_joint", r".*_wrist_yaw_joint"),
        _reflected((0.068e-4, 0, 0), (1, 5, 5)),
        5.0,
    ),
    (
        (r"waist_pitch_joint", r"waist_roll_joint", r".*_ankle_.*_joint"),
        2 * _A5020,
        50.0,
    ),
]


def _match(table: dict[str, float], joint: str) -> float:
    """mjlab's resolve_matching_names_values: the value of the pattern that matches."""
    values = [v for p, v in table.items() if re.fullmatch(p, joint)]
    if len(values) != 1:
        raise ValueError(f"{joint} matches {len(values)} patterns")
    return values[0]


def _action_scale(joint: str) -> float:
    for patterns, armature, effort in _ACTUATORS:
        if any(re.fullmatch(p, joint) for p in patterns):
            return 0.25 * effort / (armature * _FREQ**2)
    raise ValueError(joint)


def _noisy(name: str, feature: str, noise: float, **extra: object) -> dict:
    return {
        "name": name,
        "feature": feature,
        "noise": "Uniform",
        "noiseLow": -noise,
        "noiseHigh": noise,
        **extra,
    }


def flat_lab(joints: list[str]) -> dict:  # noqa: C901
    gravity_scale = [1.0 / 9.81]
    actor = [
        _noisy("base_ang_vel", "pelvis.angular_velocity_from_base", 0.2),
        _noisy("projected_gravity", "gravity_from_base", 0.05, scale=gravity_scale),
        {"name": "command", "feature": "twist.command"},
        {"name": "phase", "feature": "twist.phase"},
        _noisy("joint_pos", "body.joints.from_initial", 0.01),
        _noisy("joint_vel", "body.joint_velocities", 1.5),
        {"name": "actions", "feature": "action"},
    ]
    critic = [
        {k: v for k, v in o.items() if not k.startswith("noise")} for o in actor
    ] + [
        {"name": "base_lin_vel", "feature": "pelvis.linear_velocity_from_base"},
        {"name": "foot_height", "feature": "feet.height"},
        {"name": "foot_air_time", "feature": "feet.air_time"},
        {"name": "foot_contact", "feature": "feet.found"},
        {"name": "foot_contact_forces", "feature": "feet.force_log1p"},
    ]
    foot = {"command": "twist", "contact": "feet", "commandThreshold": 0.1}
    joints_param = {"joints": "body.joints.from_initial"}
    rewards = [
        _reward("track_linear_velocity", 1.0, command="twist",
                velocity="pelvis.linear_velocity_from_base", std=math.sqrt(0.25)),
        _reward("track_angular_velocity", 1.0, command="twist",
                velocity="pelvis.angular_velocity_from_base", std=math.sqrt(0.5)),
        _reward("body_orientation_l2", -1.0, kind="projected_gravity_xy",
                rotation="torso_link.rotation_in_task_root.quat"),
        _reward("pose", 1.0, kind="variable_posture", command="twist", **joints_param,
                stdStanding=[_match(_STD_STANDING, j) for j in joints],
                stdWalking=[_match(_STD_WALKING, j) for j in joints],
                stdRunning=[_match(_STD_RUNNING, j) for j in joints],
                walkingThreshold=0.1, runningThreshold=1.5),
        _reward("body_ang_vel", -0.05, kind="body_angular_velocity_xy",
                velocity="torso_link.angular_velocity_from_base",
                baseRotation="pelvis.rotation_in_task_root.quat"),
        _reward("is_terminated", -200.0),
        _reward("joint_acc_l2", -2.5e-7, velocities="body.joint_velocities"),
        _reward("joint_pos_limits", -10.0, joints="body"),
        _reward("action_rate_l2", -0.05),
        _reward("foot_gait", 0.5, kind="feet_gait", period=0.6, offset=[0.0, 0.5],
                threshold=0.56, **foot),
        _reward("foot_clearance", -1.0, kind="feet_clearance", targetHeight=0.10, **foot),
        _reward("foot_slip", -0.25, kind="feet_slip", **foot),
        _reward("soft_landing", -1e-3, **foot),
        _reward("stand_still", -1.0, command="twist", **joints_param, commandThreshold=0.1),
    ]  # fmt: skip
    command_globals = {
        # (name, first stage, second stage): commands_vel's lin_vel_x and lin_vel_y stages.
        "twist_vx_lo": (-0.5, -1.0),
        "twist_vx_hi": (1.0, 2.0),
        "twist_vy_lo": (-0.5, -1.0),
        "twist_vy_hi": (0.5, 1.0),
    }
    lab = {
        "metadata": {
            "name": "Unitree G1 velocity (flat)",
            "description": (
                "unitree_rl_mjlab's Unitree-G1-Flat (Apache-2.0, @1425b15) on SuperDex Lab, "
                "written by generate_labs.py: 200 Hz physics, 50 Hz policy, mjlab's actuators, "
                "observations, twist command and curriculum, events, terms and termination, "
                "without the angular momentum penalty, the torso COM DR and self collision "
                "(README.md)"
            ),
        },
        "scene": "g1_velocity.superdex_scene",
        "task": "g1_velocity.superdex_task",
        "slots": {
            "robot": {
                "bot": "g1",
                "gravityCompensated": False,
                "baseLink": "pelvis",
                "softLimitFactor": 0.9,
            }
        },
        "controlHz": 50,
        "physicsHz": 200,
        "episodeS": 20.0,
        "randomInitialStep": True,
        "settleSteps": 0,
        "actions": [
            {"controller": "body", "scale": [_action_scale(j) for j in joints]}
        ],
        "observations": actor,
        "critic": critic,
        "commands": [
            {
                "name": "twist",
                "kind": "twist_command",
                "params": {
                    "base": "pelvis",
                    "linVelX": [-1.0, 2.0],
                    "linVelY": [-1.0, 1.0],
                    "angVelZ": [-1.0, 1.0],
                    "heading": [-math.pi, math.pi],
                    "resamplingS": [3.0, 8.0],
                    "standingShare": 0.05,
                    "headingShare": 1.0,
                    "headingCommand": True,
                    "headingStiffness": 0.5,
                    "minNorm": 0.1,
                    "phasePeriodS": 0.6,
                    "linVelXGlobals": ["twist_vx_lo", "twist_vx_hi"],
                    "linVelYGlobals": ["twist_vy_lo", "twist_vy_hi"],
                },
            },
            {
                "name": "feet",
                "kind": "foot_contact",
                "params": {
                    "links": ["links:left_foot?_s?", "links:right_foot?_s?"],
                    "sites": ["left_foot", "right_foot"],
                    "ground": "StaticPlane",
                    "base": "pelvis",
                    "forceThreshold": 1.0,
                },
            },
        ],
        "events": [
            {
                "name": "reset_base_position",
                "kind": "randomize_field",
                "params": {
                    "field": "bot.root_position",
                    "target": "bot/g1",
                    "distribution": ["uniform_box", [-0.5, -0.5], [0.5, 0.5]],
                    "operation": "add",
                    "axes": [0, 1],
                    "temporal": "reset",
                },
            },
            {
                "name": "reset_base_yaw",
                "kind": "randomize_field",
                "params": {
                    "field": "bot.root_rotation",
                    "target": "bot/g1",
                    "distribution": [
                        "uniform_box",
                        [0.0, 0.0, -3.14],
                        [0.0, 0.0, 3.14],
                    ],
                    "operation": "add",
                    "temporal": "reset",
                },
            },
            {
                "name": "push_robot",
                "kind": "push_robot",
                "params": {
                    "intervalS": [5.0, 6.0],
                    "linear": [[-0.5, 0.5], [-0.5, 0.5], [-0.4, 0.4]],
                    "angular": [[-0.52, 0.52], [-0.52, 0.52], [-0.78, 0.78]],
                },
            },
            {
                # The foot-ground friction is sqrt(foot x ground) and the ground's is 1: a foot
                # friction u² gives the pair u, drawn uniformly as mjlab's geom_friction (shared).
                "name": "foot_friction",
                "kind": "randomize_field",
                "params": {
                    "field": "contact.coulombFrictionCoefficient",
                    "target": "bot/g1/links/*_foot?_s?",
                    "distribution": ["uniform", 0.3, 1.6, {"power": 2}],
                    "operation": "abs",
                    "temporal": "startup",
                },
            },
            {
                "name": "encoder_bias",
                "kind": "randomize_field",
                "params": {
                    "field": "encoder.offset",
                    "target": "bot/g1/joints/{left,right,waist}_*",
                    "distribution": ["uniform_vec", -0.015, 0.015, len(joints)],
                    "operation": "abs",
                    "temporal": "startup",
                },
            },
        ],
        "terminations": [
            {
                # bad_orientation at 70°: gravity's z in pelvis axes above -g cos 70°.
                "name": "fell_over",
                "feature": "gravity_from_base",
                "compare": "Above",
                "thresholds": [1e9, 1e9, -9.81 * math.cos(math.radians(70.0))],
            }
        ],
        "rewards": rewards,
        "curricula": [
            {
                "name": name,
                "kind": "reward_curriculum",
                "global": name,
                "params": {
                    "stages": [
                        {"step": 0, "weight": first},
                        {"step": _COMMAND_STAGE_STEP, "weight": second},
                    ]
                },
            }
            for name, (first, second) in command_globals.items()
        ],
        "globals": [
            {"name": name, "default": first, "eval": first}
            for name, (first, _) in command_globals.items()
        ],
        "variants": {
            "eval": {"randomInitialStep": False},
            "play": {
                "randomInitialStep": False,
                "dropNoise": True,
                "dropStartupDr": True,
            },
        },
    }
    return lab


def mjlab_lab(joints: list[str]) -> dict:
    """flat_lab plus the two terms of mjlab's MDP it lacked: the whole-body angular momentum
    penalty (|L|², weight -0.025, L about the robot's center of mass as MuJoCo's subtreeangmom at
    the pelvis) and the torso's center-of-mass DR (U(-0.05, 0.05) m per axis at startup). Self
    collision (-1) is still left out."""
    lab = flat_lab(joints)
    lab["metadata"] = {
        "name": "Unitree G1 velocity (flat), mjlab parity",
        "description": (
            "g1_velocity_flat plus mjlab's angular momentum penalty and torso COM DR, written by "
            "generate_labs.py; all of Unitree-G1-Flat's MDP but its self-collision penalty "
            "(README.md)"
        ),
    }
    rewards = lab["rewards"]
    at = next(i for i, r in enumerate(rewards) if r["name"] == "body_ang_vel") + 1
    rewards.insert(
        at,
        _reward(
            "angular_momentum",
            -0.025,
            kind="squared_norm",
            feature="links:*.angular_momentum",
        ),
    )
    lab["events"].append(
        {
            "name": "base_com",
            "kind": "randomize_field",
            "params": {
                "field": "body.com_offset",
                "target": "bot/g1/links/torso_link",
                "distribution": ["uniform_box", [-0.05] * 3, [0.05] * 3],
                "operation": "add",
                "temporal": "startup",
            },
        }
    )
    return lab


_CLIP_SCHEDULE = [
    # [t (s), vx, vy, wz]: stand, walk, walk faster, turn in place, arc, sidestep, stand.
    [0.0, 0.0, 0.0, 0.0],
    [2.0, 0.5, 0.0, 0.0],
    [6.0, 1.0, 0.0, 0.0],
    [10.0, 0.0, 0.0, 0.8],
    [14.0, 0.5, 0.0, 0.5],
    [17.0, 0.0, 0.4, 0.0],
    [20.0, 0.0, 0.0, 0.0],
]


def clip_lab(joints: list[str]) -> dict:
    """g1_velocity_flat_mjlab for clips and fixed-command looks: one 23 s episode through
    _CLIP_SCHEDULE (twist_command's schedule: no draws, heading control or standing envs), no pushes
    and no base pose draw, from the episode's start; its `play` variant drops the observation noise
    and the startup DR."""
    lab = {
        "metadata": {
            "name": "Unitree G1 velocity (flat), clip schedule",
            "description": (
                "g1_velocity_flat_mjlab with fixed commands for clips (stand, walk 0.5 and "
                "1.0 m/s, turn in place, arc, sidestep, stand), written by generate_labs.py"
            ),
        },
        "base": "g1_velocity_flat_mjlab.superdex_lab",
        "episodeS": _CLIP_SCHEDULE[-1][0] + 3.0,
        "randomInitialStep": False,
        # The schedule reads no command-range globals, so neither they nor their curricula stay.
        "globals": [],
        "curricula": [],
    }
    mjlab = mjlab_lab(joints)
    twist, feet = mjlab["commands"]
    lab["commands"] = [
        {"name": "twist", "kind": "twist_command",
         "params": {"base": "pelvis", "schedule": _CLIP_SCHEDULE}},
        feet,
    ]  # fmt: skip
    lab["events"] = [
        e for e in mjlab["events"]
        if e["name"] not in ("push_robot", "reset_base_position", "reset_base_yaw")
    ]  # fmt: skip
    return lab


_RULES_CLIP_SCHEDULE = [
    # The overnight run's pre-registered clip: vx 0.5, vx 1.0, vy 0.5 m/s, yaw rate 0.5 rad/s.
    [0.0, 0.5, 0.0, 0.0],
    [5.0, 1.0, 0.0, 0.0],
    [10.0, 0.0, 0.5, 0.0],
    [15.0, 0.0, 0.0, 0.5],
]


def rules_clip_lab(joints: list[str]) -> dict:
    """clip_lab through _RULES_CLIP_SCHEDULE, 5 s per command, 20 s."""
    lab = clip_lab(joints)
    lab["metadata"] = {
        "name": "Unitree G1 velocity (flat), registered clip commands",
        "description": (
            "g1_velocity_flat_mjlab_clip with four fixed commands, 5 s each (vx 0.5, vx 1.0, "
            "vy 0.5 m/s, yaw rate 0.5 rad/s), written by generate_labs.py"
        ),
    }
    lab["episodeS"] = 20.0
    lab["commands"][0]["params"]["schedule"] = _RULES_CLIP_SCHEDULE
    return lab


def eval_lab() -> dict:
    """g1_velocity_flat_mjlab plus records of the command and the base velocities, for tracking
    errors read off `superdex_lab_cli step --features`."""
    features = [
        "twist.command",
        "pelvis.linear_velocity_from_base",
        "pelvis.angular_velocity_from_base",
    ]
    return {
        "metadata": {
            "name": "Unitree G1 velocity (flat), tracking-error records",
            "description": (
                "g1_velocity_flat_mjlab with records of the commanded twist and the pelvis "
                "velocities, written by generate_labs.py"
            ),
        },
        "base": "g1_velocity_flat_mjlab.superdex_lab",
        "records": [{"name": f, "feature": f} for f in features],
    }


def _reward(
    name: str, weight: float, kind: str | None = None, **params: object
) -> dict:
    entry: dict[str, object] = {"name": name, "kind": kind or name, "weight": weight}
    if params:
        entry["params"] = params
    return entry


def learn() -> dict:
    """rl_cfg.py's unitree_g1_ppo_runner_cfg in the .superdex_learn schema (as yam_lift_cube's)."""
    return {
        "num_steps_per_env": 24,
        "max_iterations": 10001,
        "save_interval": 100,
        "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
        "actor": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms.PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "normalize_advantage_per_mini_batch": False,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "optimizer": "adam",
            "rnd_cfg": None,
            "schedule": "adaptive",
            "share_cnn_encoders": False,
            "symmetry_cfg": None,
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
    }


def _write(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n")


def main() -> None:
    bot = json.loads(_BOT.read_text())
    joints = [j["name"] for j in bot["joints"] if j["type"] == "Revolute"]
    assert len(joints) == 29, len(joints)
    _write(_HERE / "g1_velocity_flat.superdex_lab", flat_lab(joints))
    _write(_HERE / "g1_velocity_flat_mjlab.superdex_lab", mjlab_lab(joints))
    _write(_HERE / "g1_velocity_flat_mjlab_clip.superdex_lab", clip_lab(joints))
    _write(
        _HERE / "g1_velocity_flat_mjlab_clip_rules.superdex_lab", rules_clip_lab(joints)
    )
    _write(_HERE / "g1_velocity_flat_mjlab_eval.superdex_lab", eval_lab())
    _write(_HERE / "g1_velocity_flat.ppo.superdex_learn", learn())


if __name__ == "__main__":
    main()
