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

"""Tests for the ``TACTILE_PAD`` sensor in ``superdex.lab.sensors.tactile``."""

from __future__ import annotations

import json
import unittest

import numpy as np
import superdex.physics as physics
import trimesh
from superdex.lab.sensors.tactile import TactilePadParams, TactilePadSensor

GRAVITY = 9.81
WEIGHT_MASS = 0.1  # [kg]
WEIGHT_XY = (0.003, -0.004)  # [m], weight center in the sensor frame
PAD_SIZE = (0.02, 0.03, 0.01)  # [m]


def setUpModule() -> None:
    if not physics.is_initialized():
        physics.initialize(num_worker_threads=0)


def _mesh_shape(mesh: trimesh.Trimesh) -> physics.ShapeHandle:
    return physics.create_mesh_shape(
        physics.MeshData(
            nodes_per_element=3,
            coordinates=mesh.vertices.ravel(),
            connectivity=mesh.faces.ravel(),
        )
    )


def _box(extents) -> trimesh.Trimesh:
    return trimesh.creation.box(extents).subdivide().subdivide()


class TestTactilePadParams(unittest.TestCase):
    def test_empty_params_are_defaults(self) -> None:
        self.assertEqual(TactilePadParams.from_param_args(""), TactilePadParams())

    def test_inline_json(self) -> None:
        params = TactilePadParams.from_param_args(
            json.dumps(
                {
                    "rows": 4,
                    "cols": 3,
                    "size": [0.01, 0.02],
                    "pad_from_sensor": {"translation": [0, 0, 0.001]},
                    "noise_std": 0.01,
                }
            )
        )
        self.assertEqual((params.rows, params.cols), (4, 3))
        self.assertEqual(params.size, (0.01, 0.02))
        self.assertEqual(params.pad_from_sensor_translation, (0.0, 0.0, 0.001))
        self.assertEqual(params.pad_from_sensor_rotation, (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(params.noise_std, 0.01)

    def test_unknown_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TactilePadParams.from_param_args('{"taxels": 3}')

    def test_invalid_grid_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TactilePadParams.from_param_args('{"rows": 0}')


class TestTactilePadSensor(unittest.TestCase):
    """A weight resting on a sensorized pad, which itself rests on the ground."""

    def setUp(self) -> None:
        self.scene = physics.create_scene("TactilePadSensorTest")
        self.addCleanup(physics.destroy_scene, self.scene)
        self.scene.set_gravity([0, 0, -GRAVITY])
        self.scene.create_rigid_actor(
            name="ground",
            shape=physics.create_plane_shape([0, 0, 1], 0),
            is_static=True,
        )
        half_height = PAD_SIZE[2] / 2
        self.pad = self.scene.create_rigid_actor(
            name="pad",
            shape=_mesh_shape(_box(PAD_SIZE).subdivide()),
            is_static=False,
            mass=0.05,
            world_from_local=physics.TransformRT(
                physics.Quaternion.identity(), [0, 0, half_height]
            ),
        )
        self.weight = self.scene.create_rigid_actor(
            name="weight",
            shape=_mesh_shape(_box([0.008] * 3)),
            is_static=False,
            mass=WEIGHT_MASS,
            world_from_local=physics.TransformRT(
                physics.Quaternion.identity(),
                [WEIGHT_XY[0], WEIGHT_XY[1], PAD_SIZE[2] + 0.0042],
            ),
        )

    def _sensor(self, **overrides) -> TactilePadSensor:
        params = {
            "rows": 6,
            "cols": 4,
            "size": list(PAD_SIZE[:2]),
            # Sensor on the pad's top face, +z up.
            "pad_from_sensor": {"translation": [0, 0, PAD_SIZE[2] / 2]},
        }
        params.update(overrides)
        return TactilePadSensor(self.pad, json.dumps(params))

    def _settle(self, sensor: TactilePadSensor, steps: int = 400):
        dt = 1.0 / 500.0
        reading = None
        for _ in range(steps):
            self.scene.step(dt)
            reading = sensor.compute_signal(dt)
        return reading

    def test_reports_the_weight_only_from_the_sensing_face(self) -> None:
        reading = self._settle(self._sensor())
        weight = WEIGHT_MASS * GRAVITY
        # The pad's own ground contacts are on its back face and must not be sensed.
        self.assertAlmostEqual(reading.normal_force, weight, delta=0.03 * weight)
        self.assertLess(reading.tangential_force, 0.05 * weight)
        self.assertTrue(reading.in_contact)
        self.assertAlmostEqual(float(reading.taxels.sum()), weight, delta=0.1 * weight)

    def test_center_of_pressure_and_taxel_peak(self) -> None:
        reading = self._settle(self._sensor())
        np.testing.assert_allclose(reading.center_of_pressure, WEIGHT_XY, atol=1e-3)
        # 6 x 4 taxels of 5 x 5 mm over a 20 x 30 mm pad: (x=3, y=-4) mm is in the
        # taxel at column 2, row 2 (row grows along +y, column along +x).
        row, col = np.unravel_index(np.argmax(reading.taxels), reading.taxels.shape)
        self.assertEqual((row, col), (2, 2))
        self.assertEqual(reading.taxels.shape, (6, 4))
        self.assertTrue(np.all(reading.taxels >= 0))

    def test_no_load_without_contact(self) -> None:
        self.scene.destroy_actor(self.weight)
        reading = self._settle(self._sensor(), steps=50)
        self.assertFalse(reading.in_contact)
        self.assertEqual(reading.normal_force, 0.0)
        self.assertFalse(reading.taxels.any())

    def test_saturation_and_threshold(self) -> None:
        reading = self._settle(self._sensor(saturation=0.1, threshold=0.05))
        self.assertLessEqual(float(reading.taxels.max()), 0.1 + 1e-6)
        nonzero = reading.taxels[reading.taxels > 0]
        self.assertTrue(np.all(nonzero >= 0.05))

    def test_noise_is_reproducible_after_reset(self) -> None:
        sensor = self._sensor(noise_std=0.01, seed=7)
        self._settle(sensor, steps=10)
        sensor.reset()
        first = sensor.compute_signal().taxels
        sensor.reset()
        second = sensor.compute_signal().taxels
        np.testing.assert_array_equal(first, second)

    def test_time_constant_lags_the_load(self) -> None:
        dt = 1.0 / 500.0
        lagged = self._sensor(time_constant=0.05)
        instant = self._sensor()
        # Let the weight land, then compare the two responses on the first loaded step.
        for _ in range(400):
            self.scene.step(dt)
            fast = instant.compute_signal(dt)
            slow = lagged.compute_signal(dt)
            if fast.taxels.sum() > 0:
                break
        self.assertLess(slow.taxels.sum(), fast.taxels.sum())


if __name__ == "__main__":
    unittest.main()
