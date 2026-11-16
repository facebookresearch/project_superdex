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

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rrd_video.auto_camera import parse_up, Pick, pick_camera, subject_in_view
from rrd_video.camera import (
    CameraError,
    CameraPose,
    load_registry,
    RegistryEntry,
    save_entry,
    task_key,
)
from rrd_video.cli import run
from rrd_video.geometry import read_geometry
from rrd_video.recording import Recording
from rrd_video.tests import synthetic_recordings as synthetic


class CameraPoseTest(unittest.TestCase):
    def test_parse_and_round_trip(self) -> None:
        pose = CameraPose.parse("eye=1,2,3 target=0,0,0.5 up=0,1,0")
        self.assertEqual(pose, CameraPose((1, 2, 3), (0, 0, 0.5), (0, 1, 0)))
        self.assertEqual(
            CameraPose.from_json(json.loads(json.dumps(pose.to_json()))), pose
        )
        self.assertEqual(CameraPose.parse("eye=1,2,3 target=0,0,0").up, (0, 0, 1))

    def test_bad_specs_are_refused(self) -> None:
        specs = (
            "eye=1,2,3",
            "eye=1,2 target=0,0,0",
            "eye=1,2,3 target=0,0,0 fov=40",
            "eye=1,2,3 target=0,0,0 extra",
            "eye=1,1,1 target=1,1,1",  # No view direction.
            "eye=0,0,2 target=0,0,0",  # Looking along the default up.
            "eye=nan,0,0 target=0,0,0",
        )
        with self.assertRaises(CameraError):
            CameraPose.from_json({"eye": "123", "target": [0, 0, 0]})
        for spec in specs:
            with self.subTest(spec=spec), self.assertRaises(CameraError):
                CameraPose.parse(spec)

    def test_up_axis_names(self) -> None:
        self.assertEqual(parse_up("-y").tolist(), [0, -1, 0])
        with self.assertRaises(ValueError):
            parse_up("w")


class RegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)

    def key(self, properties: dict[str, str]) -> str:
        scene = synthetic.sliding_box_scene(frames=3)
        scene.properties = properties
        return task_key(Recording(synthetic.write(self.dir / "k.rrd", scene)))

    def test_most_specific_property_wins_and_paths_are_normalized(self) -> None:
        lab = "/home/u/project_superdex/assets/tasks/a/b.superdex_lab"
        self.assertEqual(self.key({"env": lab, "task": "lift"}), "task=lift")
        self.assertEqual(self.key({"env": lab}), "env=tasks/a/b.superdex_lab")
        self.assertEqual(
            self.key({"env": "/home/u/cpp_eval/env/wuji_eval.json"}),
            "env=wuji_eval.json",
        )
        self.assertEqual(
            self.key({"scene.task.filename": "C:\\Users\\u\\t3.mochi_bot_task"}),
            "scene.task.filename=t3.mochi_bot_task",
        )

    def test_without_properties_the_key_is_the_actor_names(self) -> None:
        self.assertEqual(self.key({}), "actors=cube+floor+wall")

    def test_saving_again_replaces_the_camera_of_that_task(self) -> None:
        cameras = self.dir / "cameras"
        save_entry(
            cameras,
            RegistryEntry("task=lift", CameraPose((1, 1, 1), (0, 0, 0)), "manual"),
        )
        save_entry(
            cameras,
            RegistryEntry("task=push", CameraPose((0, 1, 1), (0, 0, 0)), "manual"),
        )
        moved = CameraPose((2, 1, 1), (0, 0, 0))
        save_entry(cameras, RegistryEntry("task=lift", moved, "manual"))
        registry = load_registry(cameras)
        self.assertEqual(sorted(registry), ["task=lift", "task=push"])
        self.assertEqual(registry["task=lift"].pose, moved)

    def test_invalid_registry_file_is_named(self) -> None:
        cameras = self.dir / "bad"
        cameras.mkdir()
        for name, text in (
            ("broken.json", "{"),
            ("list.json", "[]"),
            ("keys.json", '{"schema": "superdex.rrd_video.camera/1", "key": ["a"]}'),
        ):
            (cameras / name).write_text(text)
            with self.subTest(name), self.assertRaisesRegex(CameraError, name):
                load_registry(cameras)
            (cameras / name).unlink()
        with self.assertRaises(CameraError):
            load_registry(self.dir / "missing")


class AutoCameraTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)

    def pick(
        self, scene: synthetic.Scene, name: str
    ) -> tuple[np.ndarray, CameraPose, Recording, Pick]:
        recording = Recording(synthetic.write(self.dir / f"{name}.rrd", scene))
        frames = recording.clock.frames.tolist()
        pick = pick_camera(recording, read_geometry(recording), frames, 16 / 9)
        pose = pick.best.pose
        offset = np.subtract(pose.eye, pose.target)
        return offset / np.linalg.norm(offset), pose, recording, pick

    def test_camera_sees_the_moving_box_from_the_open_side(self) -> None:
        direction, _, _, pick = self.pick(
            synthetic.sliding_box_scene(up="z"), "slide_z"
        )
        subject = pick.subject
        self.assertEqual(subject.paths, ["/world/cube"])
        self.assertEqual(
            (subject.up.tolist(), subject.up_source), ([0, 0, 1], "flat static bodies")
        )
        self.assertGreater(direction[1], 0.3)  # The wall stands on -y.
        self.assertLess(abs(direction[0]), 0.6)  # Looking across the +x motion.
        self.assertGreater(direction[2], 0.2)

    def test_y_up_scene_gets_a_y_up_camera(self) -> None:
        direction, pose, _, _ = self.pick(
            synthetic.sliding_box_scene(up="y"), "slide_y"
        )
        self.assertEqual(pose.up, (0.0, 1.0, 0.0))
        self.assertLess(direction[2], -0.3)  # The wall stands on +z.
        self.assertGreater(direction[1], 0.2)

    def test_pick_is_deterministic_across_fresh_reads(self) -> None:
        poses = {
            self.pick(synthetic.sliding_box_scene(up="z"), f"det_{i}")[1]
            for i in range(5)
        }
        self.assertEqual(len(poses), 1)

    def test_a_body_falling_out_of_the_scene_does_not_pull_the_camera_away(
        self,
    ) -> None:
        scene = synthetic.sliding_box_scene(frames=30)
        fall = [(0.2, 0.2, 0.05)] * 20 + [
            (0.2, 0.2, 0.05 - 0.5 * (i + 1) ** 2) for i in range(10)
        ]
        scene.bodies.append(
            synthetic.Body("lost", (0.05, 0.05, 0.05), (200, 200, 60), fall)
        )
        _, pose, recording, pick = self.pick(scene, "fall")
        self.assertLess(np.linalg.norm(np.subtract(pose.eye, pose.target)), 5.0)
        frames = recording.clock.frames.tolist()
        shares = subject_in_view(
            read_geometry(recording), pick.subject, pose, frames, 16 / 9
        )
        self.assertEqual(min(shares[:20]), 1.0)
        self.assertLess(shares[-1], 1.0)  # The in-view check reports the fall.


class CameraChoiceTest(unittest.TestCase):
    """Which camera `camera show` (and so `render`) uses."""

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)

    def recording(self, name: str, task: str | None = None, offset: float = 0.0) -> str:
        scene = synthetic.sliding_box_scene(frames=12)
        scene.properties = {"task": task} if task else {}
        cube = scene.bodies[-1]
        cube.positions = [
            None if p is None else (p[0] + offset, p[1], p[2]) for p in cube.positions
        ]
        return str(synthetic.write(self.dir / name, scene))

    def show(self, *argv: str) -> list[dict[str, object]]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(run(["camera", "show", *argv]), 0)
        return json.loads(out.getvalue())["cameras"]

    def test_flag_then_registry_then_automatic(self) -> None:
        path = self.recording("a.rrd", task="slide")
        registry = self.dir / "cams"
        save_entry(
            registry,
            RegistryEntry("task=slide", CameraPose((1, 1, 1), (0, 0, 0)), "manual"),
        )
        self.assertEqual(self.show(path)[0]["source"], "auto")
        self.assertEqual(
            self.show(path, "--registry", str(registry))[0]["source"],
            "registry (manual)",
        )
        flag = ["--registry", str(registry), "--camera", "eye=2,0,1 target=0,0,0"]
        self.assertEqual(self.show(path, *flag)[0]["source"], "flag")
        self.assertEqual(
            self.show(path, "--registry", str(registry), "--camera", "auto")[0][
                "source"
            ],
            "auto",
        )

    def test_recordings_of_a_task_share_a_camera_unless_it_loses_the_subject(
        self,
    ) -> None:
        first, same = (
            self.recording("a.rrd", task="slide"),
            self.recording("b.rrd", task="slide"),
        )
        far = self.recording("c.rrd", task="slide", offset=3.0)
        cameras = self.show(first, same, far)
        self.assertEqual(cameras[1]["camera"], cameras[0]["camera"])
        self.assertTrue(str(cameras[2]["source"]).startswith("auto (own pick"))
        unnamed = self.show(
            self.recording("d.rrd"), self.recording("e.rrd", offset=3.0)
        )
        self.assertEqual([c["source"] for c in unnamed], ["auto", "auto"])

    def test_excluded_entities_are_not_framed(self) -> None:
        path = self.recording("a.rrd")
        self.assertNotEqual(
            self.show(path)[0]["camera"],
            self.show(path, "--exclude", "/world/cube")[0]["camera"],
        )
