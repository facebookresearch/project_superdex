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

"""Bot loading that keeps the joint couplings of attached sub-bots.

``superdex-robotics`` 1.0.0 merges an ``AttachBot`` child's links, joints, contact
overrides and cycles into the combo, but drops its ``linearTransmissions``, so a hand's
coupled (mimic) finger joints come loose once the hand is mounted on an arm. The engine
fix lives in ``superdex_robotics/src/utils/bot_utils.cpp``; until a release carries it,
:func:`load_bot_prefab` re-attaches the missing transmissions by joint name. It is a
no-op on a build that already merges them.
"""

from __future__ import annotations

import json
from pathlib import Path

import superdex.robotics as robotics

_ROOT_MARKER = ".superdex_root"


def load_bot_prefab(path: str | Path) -> robotics.BotPrefab:
    """Load a ``.superdex_bot`` like ``robotics.load_bot_prefab_from_file``, restoring
    the linear transmissions of every attached sub-bot (recursively)."""
    path = Path(path).resolve()
    prefab = robotics.load_bot_prefab_from_file(str(path))
    raw = json.loads(path.read_text())
    if "base" not in raw:  # a plain bot has nothing attached
        return prefab

    joint_index = {joint.name: i for i, joint in enumerate(prefab.joints)}
    present = {t.name for t in prefab.linear_transmissions}
    transmissions = list(prefab.linear_transmissions)
    for attach in _attach_bot_mods(raw):
        prefix = attach.get("prefix", "")
        child = load_bot_prefab(_resolve_bot_path(attach["path"], path))
        for transmission in child.linear_transmissions:
            name = prefix + transmission.name
            if name in present:
                continue
            names = [prefix + child.joints[i].name for i in transmission.joint_indices]
            if not all(n in joint_index for n in names):
                raise RuntimeError(
                    f"cannot re-attach transmission '{name}' of {path.name}"
                )
            transmission.name = name
            transmission.joint_indices = [joint_index[n] for n in names]
            transmissions.append(transmission)
            present.add(name)
    prefab.linear_transmissions = transmissions
    return prefab


def _attach_bot_mods(raw: dict) -> list[dict]:
    mods = []
    for modification in raw.get("modifications", []):
        attach = modification.get("AttachBot")
        if attach is not None and attach.get("enabled", True):
            mods.append(attach)
    return mods


def _resolve_bot_path(reference: str, referrer: Path) -> Path:
    """Resolve a bot reference: ``//`` is the folder holding ``.superdex_root``."""
    if reference.startswith("//"):
        for folder in referrer.parents:
            if (folder / _ROOT_MARKER).is_file():
                return folder / reference[2:]
        raise FileNotFoundError(
            f"no {_ROOT_MARKER} above {referrer} to resolve {reference}"
        )
    candidate = Path(reference)
    return candidate if candidate.is_absolute() else referrer.parent / candidate
