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

"""The G1 stand in MuJoCo, as the reference for SuperDex Lab's G1 stand test: unitree_rl_mjlab's
g1.xml with g1_constants.py's actuators (position actuators: kp, kv, force range; joint armature),
feet-only collision (FEET_ONLY_COLLISION), a plane, mjlab's 200 Hz solver options, the home pose
held with zero action for 10 s. Reads the MJCF as data (visual meshes stripped); runs nothing of
the upstream repo.

    buck2 run fbsource//arvr/projects/superdex/superdex_lab/tasks/locomotion/tools:mujoco_stand -- --xml <g1.xml>
"""

import argparse
import math
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

_FREQ = 10 * 2.0 * 3.1415926535
_ZETA = 2.0


def _refl(rotor, gears):
    return rotor[0] * (gears[1] * gears[2]) ** 2 + rotor[1] * gears[2] ** 2 + rotor[2]


_A5020 = _refl((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16))
_ACTUATORS = [
    ((r".*_elbow_joint", r".*_shoulder_.*_joint", r".*_wrist_roll_joint"), _A5020, 25.0),
    ((r".*_hip_pitch_joint", r".*_hip_yaw_joint", r"waist_yaw_joint"),
     _refl((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22)), 88.0),
    ((r".*_hip_roll_joint", r".*_knee_joint"), _refl((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5)), 139.0),
    ((r".*_wrist_pitch_joint", r".*_wrist_yaw_joint"), _refl((0.068e-4, 0, 0), (1, 5, 5)), 5.0),
    ((r"waist_pitch_joint", r"waist_roll_joint", r".*_ankle_.*_joint"), 2 * _A5020, 50.0),
]  # fmt: skip
_HOME = {
    r".*_hip_pitch_joint": -0.1,
    r".*_knee_joint": 0.3,
    r".*_ankle_pitch_joint": -0.2,
    r".*_shoulder_pitch_joint": 0.35,
    r".*_elbow_joint": 0.87,
    r"left_shoulder_roll_joint": 0.18,
    r"right_shoulder_roll_joint": -0.18,
}


def _match(table, name):
    for patterns, *rest in table:
        if any(re.fullmatch(p, name) for p in patterns):
            return rest
    raise ValueError(name)


def build(xml_path: str, feet_friction: float, gain_scale: float) -> str:
    root = ET.parse(xml_path).getroot()
    root.remove(root.find("asset"))
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and child.get("class") == "visual":
                parent.remove(child)
            elif child.tag == "geom" and "mesh" in child.attrib:
                parent.remove(child)
    for geom in root.iter("geom"):
        name = geom.get("name", "")
        if re.fullmatch(r"(left|right)_foot[1-7]_collision", name):
            geom.set("condim", "3")
            geom.set("friction", f"{feet_friction}")
            geom.set("contype", "0")
            geom.set("conaffinity", "1")
            geom.set("priority", "1")
        elif name.endswith("_collision"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
    ET.SubElement(root, "option", timestep="0.005", iterations="10", ls_iterations="20")
    world = root.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.05",
        contype="1",
        conaffinity="1",
    )
    actuator = ET.SubElement(root, "actuator")
    for joint in root.iter("joint"):
        name = joint.get("name")
        armature, effort = _match(_ACTUATORS, name)
        joint.set("armature", f"{armature}")
        ET.SubElement(
            actuator,
            "position",
            name=name,
            joint=name,
            kp=f"{gain_scale * armature * _FREQ**2}",
            kv=f"{gain_scale * 2 * _ZETA * armature * _FREQ}",
            forcerange=f"{-effort} {effort}",
            forcelimited="true",
        )
    return ET.tostring(root, encoding="unicode")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--friction", type=float, default=0.6)
    parser.add_argument(
        "--gain-scale", type=float, default=1.0, help="kp and kv times this"
    )
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_string(
        build(args.xml, args.friction, args.gain_scale)
    )
    data = mujoco.MjData(model)
    data.qpos[:3] = [0, 0, 0.8]
    data.qpos[3:7] = [1, 0, 0, 0]
    home = []
    for i in range(model.nu):
        name = model.actuator(i).name
        q = next((v for p, v in _HOME.items() if re.fullmatch(p, name)), 0.0)
        data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]] = q
        home.append(q)
    data.ctrl[:] = home
    mujoco.mj_forward(model, data)
    steps = int(args.seconds / model.opt.timestep)
    jadr = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    for k in range(steps):
        mujoco.mj_step(model, data)
        if k % 40 == 39:
            w, x, y, z = data.qpos[3:7]
            up = 1 - 2 * (x * x + y * y)  # z of the pelvis z axis
            err = np.abs(data.qpos[jadr] - np.array(home)).max()
            load = sum(_normal(model, data, c) for c in range(data.ncon))
            print(
                f"t {(k + 1) * model.opt.timestep:.2f} pelvis {data.qpos[0]:.3f} {data.qpos[1]:.3f} "
                f"{data.qpos[2]:.3f} tilt {math.degrees(math.acos(max(-1, min(1, up)))):.1f} "
                f"jerr {err:.3f} load {load:.1f} ncon {data.ncon}"
            )


def _normal(model, data, c) -> float:
    force = np.zeros(6)
    mujoco.mj_contactForce(model, data, c, force)
    return float(force[0])


if __name__ == "__main__":
    main()
