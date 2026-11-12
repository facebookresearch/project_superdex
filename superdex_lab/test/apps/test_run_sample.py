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

import unittest
from unittest import mock

from apps.envs import run_sample


def _resolve(
    requested: str,
    video_recording: bool = False,
    binary: bool = True,
    interactive: bool = True,
) -> str | None:
    with (
        mock.patch.object(run_sample, "MOCHI_RENDERER_VIEWER_AVAILABLE", binary),
        mock.patch.object(
            run_sample, "is_interactive_viewer_available", return_value=interactive
        ),
    ):
        return run_sample._resolve_render_mode(requested, video_recording)


class TestResolveRenderMode(unittest.TestCase):
    def test_auto_opens_the_window_when_the_build_ships_it(self) -> None:
        self.assertEqual("human", _resolve("auto"))

    def test_auto_falls_back_to_headless_without_the_interactive_window(self) -> None:
        # Open-source builds strip the window; auto must not route into it.
        with self.assertWarnsRegex(UserWarning, "not supported in this build"):
            self.assertIsNone(_resolve("auto", interactive=False))

    def test_auto_records_offscreen_without_the_interactive_window(self) -> None:
        self.assertEqual(
            "rgb_array", _resolve("auto", video_recording=True, interactive=False)
        )

    def test_auto_falls_back_to_headless_without_the_binary(self) -> None:
        with self.assertWarnsRegex(UserWarning, "mochi_viewer_app was not found"):
            self.assertIsNone(_resolve("auto", binary=False))

    def test_explicit_human_without_the_interactive_window_exits(self) -> None:
        with self.assertRaisesRegex(SystemExit, "not supported in this build"):
            _resolve("human", interactive=False)

    def test_explicit_rgb_array_does_not_need_the_interactive_window(self) -> None:
        self.assertEqual("rgb_array", _resolve("rgb_array", interactive=False))
