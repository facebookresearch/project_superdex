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

"""The frame_bank command line packs recorded episodes into a bank and describes it."""

from __future__ import annotations

import contextlib
import gc
import io
import tempfile
import unittest
from pathlib import Path

import superdex.physics as sdp
import superdex.recorder as recorder
import superdex.robotics as sdr
from superdex.physics.paths import resolve_asset
from superdex.recorder import frame_bank

BOT_PATH = "bots/arms/fr3/fr3.superdex_bot"
TIME_STEP = 0.01


def run_cli(arguments: list[str]) -> str:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        frame_bank.main(arguments)
    return output.getvalue()


class FrameBankTest(unittest.TestCase):
    def setUp(self) -> None:
        sdp.initialize(num_worker_threads=0)
        self.addCleanup(sdp.shutdown)
        self.addCleanup(gc.collect)
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def record_episode(self, run: str, num_frames: int) -> Path:
        """Record an FR3 arm under gravity for `num_frames` frames."""
        context = sdr.create_context()
        scene = sdp.create_scene("frame bank")
        prefab = sdr.load_bot_prefab_from_file(str(resolve_asset(BOT_PATH)))
        bot = sdr.create_bot(scene, prefab, context)
        assert bot is not None
        config = recorder.RecorderConfig(self.directory / run)
        episode = recorder.Recorder(config).begin_episode()
        physics = recorder.PhysicsRecorder(episode, scene)
        robotics = recorder.RoboticsRecorder(episode, bot)
        for frame in range(num_frames):
            if frame > 0:
                scene.step(TIME_STEP)
            time_sec = scene.get_total_simulation_time()
            physics.record(frame, time_sec)
            robotics.record(frame, time_sec)
        episode.commit_sync()
        # Mochi objects must be released before Mochi shuts down.
        del physics, robotics
        sdr.destroy_bot(scene, bot)
        sdp.destroy_scene(scene)
        del bot, context
        return episode.output_path

    def test_packs_recorded_episodes_and_describes_the_bank(self) -> None:
        later = self.record_episode("run_b", 3)
        earlier = self.record_episode("run_a", 5)
        bank = self.directory / "bank.superdex_framebank"

        packed = run_cli(["pack", "--out", str(bank), str(later), str(earlier)])

        self.assertEqual(packed, run_cli(["info", str(bank)]))
        self.assertRegex(
            packed, r"bank.superdex_framebank: 2 episodes, 8 frames, fp(32|64)"
        )
        self.assertRegex(packed, r"actor data/physics/actors/[^:]+: articulated")

    def test_packs_in_order_of_the_recordings_paths(self) -> None:
        recordings = [
            str(self.record_episode(run, 2 + len(run))) for run in ("a", "bb")
        ]
        first = self.directory / "first.superdex_framebank"
        second = self.directory / "second.superdex_framebank"

        run_cli(["pack", "--out", str(first), *recordings])
        run_cli(["pack", "--out", str(second), *reversed(recordings)])

        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_refuses_files_that_are_not_banks(self) -> None:
        path = self.directory / "other.superdex_framebank"
        path.write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")

        with self.assertRaisesRegex(sdp.Error, "is not a frame bank file"):
            frame_bank.info(path)
