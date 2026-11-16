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

from __future__ import annotations

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
from rrd_video.recording import Clock, Recording, RecordingError
from rrd_video.tests import synthetic_recordings as synthetic
from rrd_video.timing import make_schedule, playback_rate, window_frames


def clock(times_s: list[float], first_frame: int = 0) -> Clock:
    frames = np.arange(first_frame, first_frame + len(times_s), dtype=np.int64)
    return Clock(frames, np.round(np.array(times_s) * 1e9).astype(np.int64))


class TimingTest(unittest.TestCase):
    def test_constant_rate_plays_every_frame_at_the_recorded_rate(self) -> None:
        schedule = make_schedule(clock([i / 60 for i in range(100)]))
        self.assertEqual(schedule.frames, list(range(100)))
        self.assertEqual(schedule.fps, Fraction(60))
        self.assertTrue(schedule.uniform)

    def test_rates_above_240_hz_are_subsampled_by_an_integer(self) -> None:
        schedule = make_schedule(clock([i / 600 for i in range(601)]))
        self.assertEqual(schedule.fps, Fraction(200))
        self.assertEqual(schedule.subsample, 3)
        self.assertEqual(schedule.frames, list(range(0, 601, 3)))

    def test_a_lower_fps_skips_recorded_frames(self) -> None:
        recorded = clock([i / 120 for i in range(121)])
        self.assertEqual(make_schedule(recorded, fps=30).frames, list(range(0, 121, 4)))

    def test_variable_steps_play_at_60_fps_showing_the_nearest_frame(self) -> None:
        # Frames at 0, 10, 30, 35, 40, 70 ms: the 60 fps grid (0, 16.7, 33.3, 50 ms)
        # shows the nearest recorded frame.
        schedule = make_schedule(clock([0.0, 0.01, 0.03, 0.035, 0.04, 0.07]))
        self.assertFalse(schedule.uniform)
        self.assertEqual(schedule.fps, Fraction(60))
        self.assertEqual(schedule.frames, [0, 1, 3, 4, 5])

    def test_millisecond_rounded_times_and_held_frames_keep_the_native_rate(
        self,
    ) -> None:
        rounded = [round(i / 60, 3) for i in range(120)]
        self.assertEqual(make_schedule(clock(rounded)).fps, Fraction(60))
        doubled = [i / 100 for i in range(200)]
        for i in (20, 60, 100, 140, 180):
            doubled[i:] = [t + 0.01 for t in doubled[i:]]  # Five dropped frames.
        schedule = make_schedule(clock(doubled))
        self.assertEqual((schedule.fps, schedule.irregular_steps), (Fraction(100), 0))

    def test_frames_repeat_evenly_above_the_recorded_rate(self) -> None:
        for hz, fps in ((30, 60), (60, 120), (15, 60), (48, 240)):
            for rounding in (np.round, np.floor):  # sim_time in whole nanoseconds
                times = rounding(np.arange(hz * 4 + 1) * 1e9 / hz).astype(np.int64)
                recorded = Clock(np.arange(len(times), dtype=np.int64), times)
                frames = make_schedule(recorded, fps=fps).frames
                holds = np.unique(frames, return_counts=True)[1][1:-1]  # Whole holds.
                with self.subTest(hz=hz, fps=fps, rounding=rounding.__name__):
                    self.assertEqual(set(holds.tolist()), {fps // hz})

    def test_any_frame_rate_plays_in_real_time(self) -> None:
        recorded = clock([i / 100 for i in range(201)])  # 2 s at 100 Hz.
        at_60 = make_schedule(recorded, fps=60)
        self.assertEqual((len(at_60.frames), at_60.subsample), (121, 0))
        self.assertEqual(at_60.frames[60], 100)  # 1 s in, the frame recorded at 1 s.
        at_200 = make_schedule(recorded, fps=200)
        self.assertEqual(
            at_200.frames[:5], [0, 0, 1, 1, 2]
        )  # Ties go to the earlier frame.

    def test_fps_is_capped_and_subsample_only_when_whole(self) -> None:
        recorded = clock([i / 600 for i in range(600)])
        with self.assertRaises(RecordingError):
            make_schedule(recorded, fps=600)
        self.assertEqual(make_schedule(recorded, fps=120).subsample, 5)
        self.assertEqual(make_schedule(recorded, fps=7).subsample, 0)

    def test_one_irregular_step_keeps_the_native_rate(self) -> None:
        times = [i / 240 for i in range(500)]
        times[200:] = [t + 0.5 / 240 for t in times[200:]]
        schedule = make_schedule(clock(times))
        self.assertTrue(schedule.uniform)
        self.assertEqual((schedule.fps, schedule.irregular_steps), (Fraction(240), 1))

    def test_window_is_relative_to_the_first_frame(self) -> None:
        recorded = clock([0.5 + i / 50 for i in range(100)], first_frame=7)
        schedule = make_schedule(recorded, start_s=1.0, end_s=1.5)
        self.assertEqual(schedule.frames, list(range(57, 83)))
        with self.assertRaises(RecordingError):
            make_schedule(recorded, start_s=5.0)
        with self.assertRaisesRegex(RecordingError, "No recorded frame"):
            window_frames(recorded, start_s=1.005, end_s=1.01)  # Between two frames.

    def test_duration_is_one_period_longer_than_the_recorded_span(self) -> None:
        schedule = make_schedule(clock([i / 60 for i in range(61)]))
        self.assertAlmostEqual(schedule.duration_s, 61 / 60)

    def test_rates_snap_to_integers_only_when_close(self) -> None:
        self.assertEqual(playback_rate(1e9 / 16666667), Fraction(60))
        self.assertEqual(playback_rate(1 / 0.016667), Fraction(60))
        self.assertEqual(playback_rate(29.97), Fraction(2997, 100))


class RecordingTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)

    def sparse_scene(self, frames_entity: bool = True) -> Recording:
        # The cube is logged only at frames 0, 3 and 8; Rerun holds the latest value.
        positions = [None] * 10
        positions[0], positions[3], positions[8] = (0, 0, 0), (1, 0, 0), (2, 0, 0)
        cube = synthetic.Body("cube", (0.1, 0.1, 0.1), (200, 0, 0), positions)
        floor = synthetic.Body(
            "floor", (1, 1, 0.01), (0, 200, 0), [(0, 0, -0.1)], static=True
        )
        scene = synthetic.Scene(
            [cube, floor],
            hz=100,
            properties={"task": "slide"},
            frames_entity=frames_entity,
        )
        return Recording(synthetic.write(self.dir / "sparse.rrd", scene))

    def translations_at(self, recording: Recording, frame: int) -> list[object]:
        values = []
        for chunk in recording.chunks_at(frame):
            batch = chunk.to_record_batch()
            if (
                chunk.entity_path == "/world/cube"
                and "Transform3D:translation" in batch.schema.names
            ):
                values += batch.column("Transform3D:translation").to_pylist()
        return values

    def test_clock_and_properties(self) -> None:
        recording = self.sparse_scene()
        self.assertEqual(recording.clock.frames.tolist(), list(range(10)))
        self.assertEqual(recording.clock.times_ns[4], 40_000_000)
        self.assertEqual(recording.properties, {"task": "slide"})

    def test_cut_holds_only_the_latest_row(self) -> None:
        recording = self.sparse_scene()
        self.assertEqual(self.translations_at(recording, 2), [[[0.0, 0.0, 0.0]]])
        self.assertEqual(self.translations_at(recording, 7), [[[1.0, 0.0, 0.0]]])
        self.assertEqual(self.translations_at(recording, 9), [[[2.0, 0.0, 0.0]]])

    def test_cut_keeps_static_meshes_and_drops_data_entities(self) -> None:
        paths = {c.entity_path for c in self.sparse_scene().chunks_at(5)}
        self.assertIn("/world/floor", paths)
        self.assertIn("/data/physics/frames", paths)
        self.assertNotIn("/data/physics/actors/cube", paths)

    def test_clock_without_frames_entity_uses_every_timeline_pair(self) -> None:
        recording = self.sparse_scene(frames_entity=False)
        self.assertEqual(recording.clock.frames.tolist(), [0, 3, 8])

    def test_recording_without_frame_timeline_is_refused(self) -> None:
        path = self.dir / "other.rrd"
        rec = synthetic.rr.RecordingStream("other", recording_id="other")
        rec.save(str(path))
        schema = {"superdex.episode.schema_name": "superdex.episode"}
        rec.log("/episode", synthetic.rr.AnyValues(**schema), static=True)
        rec.set_time("step", sequence=0)
        rec.log("/world/p", synthetic.rr.Points3D([[0, 0, 0]]))
        rec.disconnect()
        with self.assertRaisesRegex(RecordingError, "no frame and sim_time timelines"):
            Recording(path)

    def test_recording_without_episode_schema_is_refused(self) -> None:
        path = self.dir / "plain.rrd"
        rec = synthetic.rr.RecordingStream("plain", recording_id="plain")
        rec.save(str(path))
        rec.set_time("frame", sequence=0)
        rec.set_time("sim_time", duration=0.0)
        rec.log("/world/p", synthetic.rr.Points3D([[0, 0, 0]]))
        rec.disconnect()
        with self.assertRaisesRegex(RecordingError, "no /episode schema"):
            Recording(path)

    def test_clear_events_and_named_frames_are_refused(self) -> None:
        for name, entity, archetype in (
            ("clear", "/world/cube", synthetic.rr.Clear(recursive=False)),
            ("named", "/world/cube", synthetic.rr.CoordinateFrame("cube_frame")),
        ):
            path = synthetic.write(
                self.dir / f"{name}.rrd",
                synthetic.sliding_box_scene(frames=3),
                extra=[(entity, archetype)],
            )
            with (
                self.subTest(name),
                self.assertRaisesRegex(RecordingError, "not supported"),
            ):
                Recording(path)

    def test_truncated_recording_is_refused(self) -> None:
        path = synthetic.write(
            self.dir / "whole.rrd", synthetic.sliding_box_scene(frames=10)
        )
        cut = self.dir / "cut.rrd"
        cut.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
        with self.assertRaisesRegex(RecordingError, "no footer"):
            Recording(cut)
