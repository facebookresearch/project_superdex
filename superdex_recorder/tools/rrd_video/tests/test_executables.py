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

"""The bundled Rerun CLI and ffmpeg when their packages are imported from a zip, as in a
self-extracting par."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
import zipfile
from importlib import resources
from pathlib import Path
from unittest import mock

from rrd_video import encode, viewer
from rrd_video.executables import packaged_executable

PACKAGE = "rrd_video_test_zipped_bins"
FAKE_RERUN = "#!/bin/sh\necho 'rerun-cli 0.33.0 (fake)'\n"
FFMPEG = "ffmpeg-linux-x86_64-v7.0.2"
FILES = {
    f"{PACKAGE}/__init__.py": "",
    f"{PACKAGE}/rerun_cli/__init__.py": "",
    f"{PACKAGE}/rerun_cli/rerun": FAKE_RERUN,
    f"{PACKAGE}/binaries/__init__.py": "",
    f"{PACKAGE}/binaries/{FFMPEG}": "ffmpeg bytes",
}
ZIPPED = {
    "rerun_cli": f"{PACKAGE}.rerun_cli",
    "imageio_ffmpeg.binaries": f"{PACKAGE}.binaries",
}


class ZippedPackageTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        archive = str(self.dir / "bins.zip")
        with zipfile.ZipFile(archive, "w") as z:
            for name, text in FILES.items():
                z.writestr(name, text)
        sys.path.insert(0, archive)
        self.addCleanup(self.forget, archive)
        files = resources.files
        patches = [
            mock.patch.object(
                resources, "files", lambda name: files(ZIPPED.get(name, name))
            ),
            mock.patch.dict(os.environ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        for key in ("RRD_VIDEO_RERUN_CLI", "RRD_VIDEO_FFMPEG", "RRD_VIDEO_LAVAPIPE"):
            os.environ.pop(key, None)
        self.work = self.dir / "work"

    def forget(self, archive: str) -> None:
        sys.path.remove(archive)
        for name in [m for m in sys.modules if m.split(".")[0] == PACKAGE]:
            del sys.modules[name]
        importlib.invalidate_caches()

    def test_a_zipped_executable_is_copied_into_the_work_dir(self) -> None:
        resource = resources.files(f"{PACKAGE}.rerun_cli").joinpath("rerun")
        self.assertNotIsInstance(resource, Path)

        path = packaged_executable(resource, self.work)

        self.assertEqual(path, self.work / "bin" / "rerun")
        self.assertEqual(path.read_text(), FAKE_RERUN)
        self.assertTrue(os.access(path, os.X_OK))

    def test_find_toolchain_runs_the_zipped_rerun_cli(self) -> None:
        toolchain = viewer.find_toolchain(self.work)

        self.assertEqual(toolchain.rerun_cli.parent, self.work / "bin")
        self.assertTrue(toolchain.rerun_cli.is_file())
        self.assertEqual(viewer.rerun_version(toolchain), "rerun-cli 0.33.0 (fake)")

    def test_find_ffmpeg_returns_a_copy_of_the_zipped_ffmpeg(self) -> None:
        path = encode.find_ffmpeg(self.work)

        self.assertEqual(path, self.work / "bin" / FFMPEG)
        self.assertEqual(path.read_text(), "ffmpeg bytes")


class InstalledExecutableTest(unittest.TestCase):
    def test_an_installed_executable_is_used_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            installed = Path(d) / "rerun"
            installed.write_text(FAKE_RERUN)
            installed.chmod(0o755)

            self.assertEqual(
                packaged_executable(installed, Path(d) / "work"), installed
            )
