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

"""End-to-end renders with headless Rerun on lavapipe."""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import rerun as rr  # @manual
import rerun.experimental as rre  # @manual
from PIL import Image
from rrd_video import encode, viewer
from rrd_video.camera import CameraPose
from rrd_video.cli import run
from rrd_video.recording import Recording
from rrd_video.tests import synthetic_recordings as synthetic

CUBE_COLOR = (220, 100, 80)
# rerun.experimental ships no type information.
_experimental: Any = rre


class FolderStorage:
    """A storage for 'test://' URIs, kept in a local folder."""

    scheme = "test://"

    def __init__(self, root: Path) -> None:
        self.root = root

    def download(self, uri: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / uri.removeprefix(self.scheme), destination)
        return destination

    def upload(self, source: Path, uri: str) -> None:
        target = self.root / uri.removeprefix(self.scheme)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def quiet_run(argv: list[str], storages: list[FolderStorage] | None = None) -> int:
    """`run`, without its JSON summary on stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        return run(argv, storages or [])


def cube_center(path: Path) -> np.ndarray:
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    mask = np.abs(pixels - np.array(CUBE_COLOR)).max(axis=2) < 60
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])


class RenderTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.toolchain = viewer.find_toolchain(self.dir / "viewer")

    def test_render_command_writes_a_checked_real_time_video(self) -> None:
        recording = synthetic.write(
            self.dir / "slide.rrd", synthetic.sliding_box_scene(frames=30)
        )
        out = self.dir / "out"
        command = [
            "render",
            str(recording),
            "--out",
            str(out),
            "--size",
            "320x180",
            "--workers",
            "4",
        ]
        self.assertEqual(quiet_run(command), 0)
        manifest = json.loads((out / "slide.video.json").read_text())
        self.assertEqual(
            (manifest["video"]["fps"], manifest["video"]["frames"]), ("60/1", 30)
        )
        self.assertAlmostEqual(manifest["video"]["duration_s"], 0.5)
        self.assertEqual(
            (manifest["camera"]["source"], manifest["checks"]["warnings"]), ("auto", [])
        )
        video = encode.probe(
            encode.find_ffmpeg(self.dir / "viewer"), out / "slide.webm"
        )
        self.assertEqual((video.frames, video.codec, video.fps), (30, "vp9", 60.0))
        self.assertIn("bt709", video.color)
        start, end = (
            cube_center(out / "slide.first.png"),
            cube_center(out / "slide.last.png"),
        )
        self.assertGreater(
            np.linalg.norm(end - start), 40
        )  # The box slides across the view.
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            self.assertEqual(run(["camera", "show", str(recording)]), 0)
        camera = json.loads(shown.getvalue())["cameras"][0]["camera"]
        self.assertEqual(camera["eye"], manifest["camera"]["eye"])  # What render used.

    def test_frames_above_the_recorded_rate_repeat_and_render_once(self) -> None:
        recording = synthetic.write(
            self.dir / "slow.rrd", synthetic.sliding_box_scene(frames=10)
        )
        out = self.dir / "out"
        command = [
            "render",
            str(recording),
            "--out",
            str(out),
            "--size",
            "320x180",
            "--fps",
            "120",
        ]
        self.assertEqual(quiet_run(command), 0)
        manifest = json.loads((out / "slow.video.json").read_text())
        self.assertEqual(
            (manifest["video"]["fps"], manifest["video"]["frames"]), ("120/1", 19)
        )
        self.assertEqual(manifest["timing"]["rendered_frames"], 10)

    def test_remote_locations_go_through_a_storage(self) -> None:
        recording = synthetic.write(
            self.dir / "slide.rrd", synthetic.sliding_box_scene(frames=4)
        )
        storage = FolderStorage(self.dir / "remote")
        (self.dir / "remote" / "in").mkdir(parents=True)
        shutil.copyfile(recording, self.dir / "remote" / "in" / "slide.rrd")
        self.assertEqual(
            quiet_run(
                ["render", "test://in/slide.rrd", "--size", "320x180"], [storage]
            ),
            1,
        )  # No --out.
        self.assertEqual(
            quiet_run(
                ["render", "other://in/slide.rrd", "--out", str(self.dir)], [storage]
            ),
            1,
        )
        command = [
            "render",
            "test://in/slide.rrd",
            "--out",
            "test://out",
            "--size",
            "320x180",
        ]
        self.assertEqual(quiet_run(command, [storage]), 0)
        self.assertTrue((self.dir / "remote" / "out" / "slide.webm").is_file())

    def test_a_failure_after_rendering_publishes_nothing(self) -> None:
        good = synthetic.write(
            self.dir / "a_slide.rrd", synthetic.sliding_box_scene(frames=3)
        )
        points = (
            self.dir / "b_points.rrd"
        )  # Renders nothing an automatic camera can frame.
        rec = rr.RecordingStream("superdex", recording_id="points")
        rec.save(str(points))
        rec.log(
            "/episode",
            rr.AnyValues(**{"superdex.episode.schema_name": "superdex.episode"}),
            static=True,
        )
        for frame in range(3):
            rec.set_time("frame", sequence=frame)
            rec.set_time("sim_time", duration=frame / 60)
            rec.log("/world/p", rr.Points3D([[0, 0, frame * 0.1]]))
        rec.disconnect()
        out = self.dir / "out"
        self.assertEqual(
            quiet_run(
                [
                    "render",
                    str(good),
                    str(points),
                    "--out",
                    str(out),
                    "--size",
                    "320x180",
                ]
            ),
            1,
        )
        self.assertFalse(out.exists())
        (out / "a_slide.last.png").mkdir(
            parents=True
        )  # A target that cannot be replaced.
        self.assertEqual(
            quiet_run(["render", str(good), "--out", str(out), "--size", "320x180"]), 1
        )
        self.assertEqual([p.name for p in out.iterdir()], ["a_slide.last.png"])

    def test_errors_exit_1_without_writing(self) -> None:
        recording = synthetic.write(
            self.dir / "slide.rrd", synthetic.sliding_box_scene(frames=3)
        )
        copy = self.dir / "other" / "slide.rrd"
        copy.parent.mkdir()
        copy.write_bytes(recording.read_bytes())
        out = self.dir / "out"
        self.assertEqual(
            quiet_run(["render", str(recording), str(copy), "--out", str(out)]), 1
        )
        truncated = self.dir / "truncated.rrd"
        truncated.write_bytes(recording.read_bytes()[: recording.stat().st_size // 2])
        both = [
            "render",
            str(recording),
            str(truncated),
            "--out",
            str(out),
            "--size",
            "320x180",
        ]
        self.assertEqual(
            quiet_run(both), 1
        )  # The truncated input is refused before any rendering.
        self.assertEqual(
            quiet_run(["render", str(self.dir / "missing.rrd"), "--out", str(out)]), 1
        )
        self.assertEqual(
            quiet_run(
                [
                    "render",
                    str(recording),
                    "--out",
                    str(out),
                    "--camera",
                    "eye=0,0,1 target=0,0,0",
                ]
            ),
            1,
        )
        self.assertFalse(out.exists())

    def test_cut_renders_like_the_whole_recording_up_to_the_frame(self) -> None:
        path = self.write_layered_recording()
        pose = CameraPose((0.4, -1.2, 0.8), (0.2, 0, 0))
        for frame in (2, 4):
            truncated = self.dir / f"truncated_{frame}.rrd"
            chunks = []
            for chunk in _experimental.RrdReader(path).stream():
                batch = chunk.to_record_batch()
                if chunk.is_static or "frame" not in batch.schema.names:
                    chunks.append(chunk)
                elif (
                    kept := batch.filter(pc.less_equal(batch.column("frame"), frame))
                ).num_rows:
                    chunks.append(
                        _experimental.Chunk.from_record_batch(kept).with_entity_path(
                            chunk.entity_path
                        )
                    )
            _experimental.ChunkStore.from_chunks(chunks).write_rrd(
                truncated, application_id="superdex", recording_id="t"
            )
            cut = self.render(path, pose, [frame])[0]
            whole = self.capture(truncated, pose, self.dir / f"whole_{frame}.png")
            np.testing.assert_array_equal(
                np.asarray(Image.open(cut)), np.asarray(Image.open(whole))
            )

    def test_empty_view_is_an_error(self) -> None:
        cube = synthetic.Body("cube", (0.05, 0.05, 0.05), CUBE_COLOR, [(0, 0, 0)] * 2)
        path = synthetic.write(self.dir / "away.rrd", synthetic.Scene([cube]))
        with self.assertRaises(viewer.RenderError):
            self.render(path, CameraPose((0, -1, 0), (0, -2, 0)), [1])

    def write_layered_recording(self) -> Path:
        """Sparse updates to two components, two rows at one frame, and a static transform
        that overrides temporal ones."""
        path = self.dir / "layered.rrd"
        rec = rr.RecordingStream("superdex", recording_id="layered")
        rec.save(str(path))
        rec.log(
            "/episode",
            rr.AnyValues(**{"superdex.episode.schema_name": "superdex.episode"}),
            static=True,
        )
        box = rr.Mesh3D(
            vertex_positions=synthetic.box_vertices((0.05, 0.05, 0.05)),
            triangle_indices=synthetic.BOX_TRIANGLES,
        )
        rec.log("/world/cube", box, static=True)
        rec.log("/world/pin", box, static=True)
        rec.log("/world/pin", rr.Transform3D(translation=[0.4, 0, 0.2]), static=True)
        for frame in range(5):
            rec.set_time("frame", sequence=frame)
            rec.set_time("sim_time", duration=frame / 60)
            rec.log(
                "/data/physics/frames",
                rr.AnyValues(**{"superdex.physics.step_dt_sec": 1 / 60}),
            )
            rec.log(
                "/world/pin", rr.Transform3D(translation=[0, 0, -1])
            )  # Hidden by the static one.
            if frame in (0, 3):
                rec.log("/world/cube", rr.Transform3D(translation=[0.1 * frame, 0, 0]))
            if frame == 3:
                rec.log(
                    "/world/cube", rr.Transform3D(translation=[0.4, 0, 0.1])
                )  # Same frame, later row.
            if frame == 2:
                rec.log(
                    "/world/cube",
                    rr.Mesh3D.from_fields(albedo_factor=[60, 160, 220, 255]),
                )
        rec.disconnect()
        return path

    def capture(self, recording: Path, pose: CameraPose, output: Path) -> Path:
        blueprint = self.dir / "view.rbl"
        viewer.write_blueprint(
            blueprint, "superdex", pose
        )  # Every test recording's application id.
        viewer.render_capture(
            self.toolchain, recording, blueprint, self.dir / "capture.png", 320, 180
        )
        viewer.crop_view(self.dir / "capture.png", 320, 180).save(output)
        return output

    def render(self, path: Path, pose: CameraPose, frames: list[int]) -> list[Path]:
        recording = Recording(path)
        outputs = []
        for frame in frames:
            recording.write_at(frame, self.dir / "cut.rrd")
            outputs.append(
                self.capture(self.dir / "cut.rrd", pose, self.dir / f"{frame}.png")
            )
        return outputs
