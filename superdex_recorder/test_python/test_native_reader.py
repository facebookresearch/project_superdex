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

"""The native episode reader must read what Rerun's own reader reads.

This is the guard on Rerun upgrades: the native reader decodes Rerun's file format itself,
so a format change shows up here as a mismatch or a rejected file.
"""

from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import rerun.experimental as rr_experimental
import superdex.physics as sdp
import superdex.recorder as recorder
import superdex.robotics as sdr
from superdex.physics.paths import resolve_asset
from superdex.physics.utils.scene_helpers import load_bot_scene
from superdex.recorder import _extension
from superdex.recorder.reader import EpisodeReader

# The Rerun wheel does not expose its experimental API to static type checkers.
_rr_experimental: Any = rr_experimental

BOT_SCENE_PATH = resolve_asset(
    "bots/internal/tests/bot_scene/test_bot_scene.mochi_bot_scene"
)
ROBOT_NAME = "t3_left"
NUM_FRAMES = 6
UNTRACKED_FRAME = 3
TIME_STEP = 0.005
# A wrist with a one-segment thumb and a two-segment index finger.
KEYPOINT_NAMES = ["wrist", "thumb_tip", "index_knuckle", "index_tip"]
KEYPOINT_PARENTS = [-1, 0, 0, 2]
KEYPOINTS = [[0.0, 0.0, 0.0], [0.03, 0.02, 0.0], [0.0, 0.04, 0.0], [0.0, 0.08, 0.01]]


def value_kind(column_type: pa.DataType) -> str:
    """Classify a component column by its element type, looking through XYZ triples."""
    value_type = column_type.value_type
    if pa.types.is_fixed_size_list(value_type):
        value_type = value_type.value_type
    if pa.types.is_string(value_type):
        return "text"
    return "number" if pa.types.is_floating(value_type) else "other"


def stored_components(path: Path) -> dict[tuple[str, str, bool], str]:
    """Map every (entity path, component, is static) that Rerun's reader sees to its kind."""
    components = {}
    for chunk in _rr_experimental.RrdReader(path).stream():
        for field in chunk.to_record_batch().schema:
            if (field.metadata or {}).get(b"rerun:kind") == b"data":
                key = (chunk.entity_path.lstrip("/"), field.name, chunk.is_static)
                components[key] = value_kind(field.type)
    return components


def flatten(value: Any) -> list[Any]:
    """Flatten nested lists, such as a row of XYZ triples, into one list."""
    if isinstance(value, list):
        return [item for element in value for item in flatten(element)]
    return [value]


class NativeEpisodeReaderTest(unittest.TestCase):
    def setUp(self) -> None:
        sdp.initialize(num_worker_threads=0)
        self.addCleanup(sdp.shutdown)
        self.addCleanup(gc.collect)
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def record_robot_episode(self) -> Path:
        """Record every kind of data the recorder writes: physics states and contacts,
        robot controls, a robot shadow, and a skeleton shadow that loses tracking."""
        robotics_context = sdr.create_context()
        bot_scene = load_bot_scene(str(BOT_SCENE_PATH), robotics_context)
        scene = bot_scene.get_scene()
        bot = bot_scene.get_bot(ROBOT_NAME)
        assert scene is not None and bot is not None
        robot = bot.get_articulated_actor()
        assert robot is not None
        metadata = recorder.EpisodeMetadata(properties={"/": "slash", "build": "abc"})
        config = recorder.RecorderConfig(self.directory)
        episode = recorder.Recorder(config).begin_episode(metadata)
        physics_config = recorder.PhysicsRecorderConfig()
        physics_config.record_contact_forces = True
        physics = recorder.PhysicsRecorder(episode, scene, physics_config)
        robotics = recorder.RoboticsRecorder(episode, bot)
        command = recorder.RobotShadow(episode, bot, "command")
        hand = recorder.SkeletonShadow(
            episode, bot, "tracked_hand", KEYPOINT_NAMES, KEYPOINT_PARENTS
        )
        real_dtype = np.asarray(scene.get_gravity()).dtype
        pose = np.zeros(robot.get_num_dofs(), dtype=real_dtype)
        keypoints = np.array(KEYPOINTS, dtype=real_dtype)
        for frame in range(NUM_FRAMES + 1):
            if frame > 0:
                scene.step(TIME_STEP)
            time_sec = scene.get_total_simulation_time()
            physics.record(frame, time_sec)
            robotics.record(frame, time_sec)
            robot.get_articulated_pose(pose)
            command.record(frame, time_sec, robot.get_root_transform(), pose)
            if frame == UNTRACKED_FRAME:
                hand.record_untracked(frame, time_sec)
            else:
                hand.record(frame, time_sec, (keypoints + 0.001 * frame).reshape(-1))
        episode.commit_sync()
        # Mochi objects must be released before Mochi shuts down.
        del physics, robotics, command, hand, robot, bot, scene, bot_scene
        del robotics_context
        return episode.output_path

    def test_reads_robot_episodes_like_rerun(self) -> None:
        path = self.record_robot_episode()
        native = _extension._EpisodeReader(path)
        components = stored_components(path)
        # Rerun's reader orders each component's samples by frame, keeping the recorded
        # order at one frame, as the native reader does.
        samples = EpisodeReader(path).component_samples_many(
            sorted({component for _, component, _ in components})
        )

        self.assertEqual(
            {(c.entity_path, c.component, c.is_static) for c in native.components()},
            set(components),
        )
        # The native reader reads text and float components. The episode has both.
        self.assertLessEqual({"text", "number"}, set(components.values()))
        for (entity_path, component, is_static), kind in components.items():
            if kind == "other":
                continue
            with self.subTest(entity_path=entity_path, component=component):
                rows = [
                    sample
                    for sample in samples[component]
                    if sample.entity_path.lstrip("/") == entity_path
                    and sample.is_static == is_static
                ]
                expected = [flatten(row.value) for row in rows]
                if kind == "text":
                    actual = [native.read_static_texts(entity_path, component)]
                    self.assertEqual(actual, expected[-1:])
                elif is_static:
                    actual = [native.read_static_scalars(entity_path, component)]
                    self.assertEqual(actual, expected[-1:])
                else:
                    read = native.read_scalars(entity_path, component)
                    self.assertEqual(
                        read.frames, [row.timelines["frame"] for row in rows]
                    )
                    self.assertEqual(
                        read.simulation_times_ns,
                        [row.timelines["sim_time"] for row in rows],
                    )
                    offsets = list(zip(read.offsets, read.offsets[1:]))
                    actual = [read.values[begin:end] for begin, end in offsets]
                    self.assertEqual(actual, expected)
