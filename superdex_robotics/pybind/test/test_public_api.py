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

from test.conftest import MochiTestBase, sdp, sdr


class PublicApiTest(MochiTestBase):
    def test_public_controller_params(self) -> None:
        params = sdr.ControllerBasicJscPdParams(kp=[1.0], kd=[0.5])
        self.assertEqual(params.kp[0], 1.0)
        self.assertEqual(params.kd[0], 0.5)

    def test_default_handle_is_invalid(self) -> None:
        self.assertFalse(sdr.RoboticsHandle().is_valid())

    def test_bot_prefab_component_surface(self) -> None:
        joint = sdr.BotJointPrefab()
        self.assertEqual(sdr.EFFORT_UNBOUNDED, -1.0)
        self.assertEqual(joint.effort_limit, sdr.EFFORT_UNBOUNDED)

        sensor = sdr.BotSensorPrefab(
            type=sdr.CameraSensor.type_name(),
            name="camera",
            parent_from_sensor=sdp.TransformRT([1.0, 2.0, 3.0]),
            params="camera.json",
        )
        actuator = sdr.BotActuatorPrefab(
            type="TEST_ACTUATOR",
            name="motor",
            params="actuator.json",
        )
        link = sdr.BotLinkPrefab(sensors=[sensor], actuators=[actuator])

        self.assertEqual(str(link.sensors[0].name), "camera")
        self.assertEqual(str(link.actuators[0].name), "motor")
        self.assertFalse(hasattr(sensor, "_legacy_type_name"))
        self.assertFalse(hasattr(sensor, "_legacy_params_file"))

    def test_create_sensor_uses_empty_default_params(self) -> None:
        context = sdr.create_context()
        scene = sdp.create_scene("Default sensor params")
        actor = scene.create_rigid_actor(
            shape=sdp.create_plane_shape(normal=[0, 1, 0], distance=0),
            is_static=True,
        )
        handle = sdr.SensorHandle()
        try:
            handle = context.create_sensor(sdr.CameraSensor.type_name(), actor)
            self.assertTrue(context.is_valid_sensor(handle))
        finally:
            if context.is_valid_sensor(handle):
                context.destroy_sensor(handle)
            sdp.destroy_scene(scene)
