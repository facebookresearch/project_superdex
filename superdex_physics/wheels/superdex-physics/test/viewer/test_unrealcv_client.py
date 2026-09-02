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

import unittest

import numpy as np
from superdex.physics.viewer.unrealcv.unrealcv_client import UnrealCVClient


class RecordingUnrealCVClient(UnrealCVClient):
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.batch_commands: list[list[str]] = []
        self.responses: dict[str, str | None] = {}
        self.batch_responses: list[str | None] = []

    def _request(self, command: str, use_async: bool = False) -> str | None:
        self.commands.append(command)
        return self.responses.get(command)

    def _request_batch(
        self, commands: list[str], use_async: bool = False, profile: bool = False
    ) -> list[str | None] | tuple[list[str | None], dict]:
        self.batch_commands.append(commands)
        if profile:
            return self.batch_responses, {}
        return self.batch_responses


class TestUnrealCVClientCameraCalibration(unittest.TestCase):
    def test_builds_exact_camera_calibration_commands(self) -> None:
        self.assertEqual(
            UnrealCVClient.build_camera_local_location_command(
                3, np.array([1.25, -2.5, 0.0])
            ),
            "vset /camera/3/local_location 1.25 -2.5 0",
        )
        self.assertEqual(
            UnrealCVClient.build_camera_local_rotation_command(
                3, np.array([10.0, -20.5, 30.25])
            ),
            "vset /camera/3/local_rotation 10 -20.5 30.25",
        )
        self.assertEqual(
            UnrealCVClient.build_camera_filmback_command(3, np.array([6.4, 4.8])),
            "vset /camera/3/filmback 6.4 4.8",
        )
        self.assertEqual(
            UnrealCVClient.build_camera_focal_length_command(3, 1.88),
            "vset /camera/3/focal_length 1.88",
        )

    def test_gets_typed_camera_calibration_values(self) -> None:
        client = RecordingUnrealCVClient()
        client.responses = {
            "vget /camera/4/local_location": "1.0 -2.5 3",
            "vget /camera/4/local_rotation": "10 20.5 -30",
            "vget /camera/4/filmback": "6.4 4.8",
            "vget /camera/4/focal_length": "1.88",
        }

        np.testing.assert_allclose(
            client.get_camera_local_location(4), np.array([1.0, -2.5, 3.0])
        )
        np.testing.assert_allclose(
            client.get_camera_local_rotation(4), np.array([10.0, 20.5, -30.0])
        )
        np.testing.assert_allclose(client.get_camera_filmback(4), np.array([6.4, 4.8]))
        self.assertEqual(client.get_camera_focal_length(4), 1.88)

    def test_gets_return_none_for_failure_responses(self) -> None:
        client = RecordingUnrealCVClient()
        client.responses = {
            "vget /camera/5/local_location": "error missing camera",
            "vget /camera/5/local_rotation": "10 20",
            "vget /camera/5/filmback": None,
            "vget /camera/5/focal_length": "not-a-float",
        }

        self.assertIsNone(client.get_camera_local_location(5))
        self.assertIsNone(client.get_camera_local_rotation(5))
        self.assertIsNone(client.get_camera_filmback(5))
        self.assertIsNone(client.get_camera_focal_length(5))

    def test_sets_return_false_for_failure_responses(self) -> None:
        client = RecordingUnrealCVClient()
        client.responses = {
            "vset /camera/6/local_location 1 2 3": "ok",
            "vset /camera/6/local_rotation 4 5 6": "error invalid rotation",
            "vset /camera/6/filmback 6.4 4.8": None,
            "vset /camera/6/focal_length 1.88": "OK",
        }

        self.assertTrue(client.set_camera_local_location(6, np.array([1.0, 2.0, 3.0])))
        self.assertFalse(client.set_camera_local_rotation(6, np.array([4.0, 5.0, 6.0])))
        self.assertFalse(client.set_camera_filmback(6, np.array([6.4, 4.8])))
        self.assertTrue(client.set_camera_focal_length(6, 1.88))

    def test_apply_camera_calibration_skips_omitted_fields(self) -> None:
        client = RecordingUnrealCVClient()
        client.batch_responses = ["ok", "ok"]

        result = client.apply_camera_calibration(
            7,
            local_location=np.array([1.0, 2.0, 3.0]),
            focal_length=1.88,
        )

        self.assertEqual(result, [True, True])
        self.assertEqual(
            client.batch_commands,
            [
                [
                    "vset /camera/7/local_location 1 2 3",
                    "vset /camera/7/focal_length 1.88",
                ]
            ],
        )

    def test_apply_camera_calibration_uses_deterministic_full_order(self) -> None:
        client = RecordingUnrealCVClient()
        client.batch_responses = ["ok", "error bad rotation", None, "OK"]

        result = client.apply_camera_calibration(
            8,
            focal_length=1.88,
            filmback=np.array([6.4, 4.8]),
            local_rotation=np.array([4.0, 5.0, 6.0]),
            local_location=np.array([1.0, 2.0, 3.0]),
        )

        self.assertEqual(result, [True, False, False, True])
        self.assertEqual(
            client.batch_commands,
            [
                [
                    "vset /camera/8/local_location 1 2 3",
                    "vset /camera/8/local_rotation 4 5 6",
                    "vset /camera/8/filmback 6.4 4.8",
                    "vset /camera/8/focal_length 1.88",
                ]
            ],
        )

    def test_apply_camera_calibration_without_fields_does_not_send_batch(self) -> None:
        client = RecordingUnrealCVClient()

        self.assertEqual(client.apply_camera_calibration(9), [])
        self.assertEqual(client.batch_commands, [])


if __name__ == "__main__":
    unittest.main()
