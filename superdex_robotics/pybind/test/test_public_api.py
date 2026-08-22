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

from test.conftest import bots, mochi, MochiTestBase


class PublicApiTest(MochiTestBase):
    def test_public_controller_params(self) -> None:
        params = bots.ControllerBasicJscPdParams(kp=[1.0], kd=[0.5])
        self.assertEqual(params.kp[0], 1.0)
        self.assertEqual(params.kd[0], 0.5)

    def test_default_handle_is_invalid(self) -> None:
        self.assertFalse(bots.RoboticsHandle().is_valid())

    def test_bot_prefab_component_surface(self) -> None:
        joint = bots.BotJointPrefab()
        self.assertEqual(bots.EFFORT_UNBOUNDED, -1.0)
        self.assertEqual(joint.effort_limit, bots.EFFORT_UNBOUNDED)

        sensor = bots.BotSensorPrefab(
            type=bots.CameraSensor.type_name(),
            name="camera",
            parent_from_sensor=mochi.TransformRT([1.0, 2.0, 3.0]),
            params="camera.json",
        )
        actuator = bots.BotActuatorPrefab(
            type="TEST_ACTUATOR",
            name="motor",
            params="actuator.json",
        )
        link = bots.BotLinkPrefab(sensors=[sensor], actuators=[actuator])

        self.assertEqual(str(link.sensors[0].name), "camera")
        self.assertEqual(str(link.actuators[0].name), "motor")
        self.assertFalse(hasattr(sensor, "_legacy_type_name"))
        self.assertFalse(hasattr(sensor, "_legacy_params_file"))

    def test_create_sensor_uses_empty_default_params(self) -> None:
        context = bots.create_context()
        scene = mochi.create_scene("Default sensor params")
        actor = scene.create_rigid_actor(
            shape=mochi.create_plane_shape(normal=[0, 1, 0], distance=0),
            is_static=True,
        )
        handle = bots.SensorHandle()
        try:
            handle = context.create_sensor(bots.CameraSensor.type_name(), actor)
            self.assertTrue(context.is_valid_sensor(handle))
        finally:
            if context.is_valid_sensor(handle):
                context.destroy_sensor(handle)
            mochi.destroy_scene(scene)
