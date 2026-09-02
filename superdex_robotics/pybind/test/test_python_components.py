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

"""End-to-end tests for the pure-Python sensor and actuator paths.

The controller counterpart lives in test/internal/test_python_controller.py; these cover the
other two component kinds -- registered with register_python_sensor / register_python_actuator,
created directly, created on a bot's actor, and (for sensors) declared at scene level in a
.mochi_bot_scene so the scene loader instantiates them by type string.

The bots context is a process-wide singleton, so every test registers its own type names and
destroys what it creates.
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
import weakref

from test.conftest import (
    bot_tag_path,
    MochiTestBase,
    requires_hdf5,
    requires_internal,
    sdp,
    sdr,
    write_assets_tag_root_marker,
)


class PyTestSensor:
    """A sensor implemented purely in Python.

    The framework calls reset(); everything else (compute_signal here) is this sensor's own API,
    driven from Python on the instance returned by get_python_sensor.
    """

    def __init__(self, actor: object, param_args: str) -> None:
        self.actor = actor
        self.param_args = param_args
        self.reset_calls = 0
        self.samples: list[float] = []

    def compute_signal(self, value: float) -> float:
        self.samples.append(value)
        return value * 2.0

    def reset(self) -> None:
        self.reset_calls += 1
        self.samples.clear()


class PyTestActuator:
    """An actuator implemented purely in Python (same contract as PyTestSensor)."""

    def __init__(self, actor: object, param_args: str) -> None:
        self.actor = actor
        self.param_args = param_args
        self.reset_calls = 0
        self.applied: list[float] = []

    def apply(self, effort: float) -> None:
        self.applied.append(effort)

    def reset(self) -> None:
        self.reset_calls += 1
        self.applied.clear()


class PurePythonSensorTest(MochiTestBase):
    def test_register_create_drive_and_destroy_without_an_actor(self) -> None:
        sensor_type = "PY_TEST_SENSOR_STANDALONE"
        bots_ctx = sdr.create_context()
        sdr.register_python_sensor(bots_ctx, sensor_type, PyTestSensor)
        self.assertTrue(bots_ctx.is_sensor_type_registered(sensor_type))

        # No actor at all: the fusion-sensor case, which needs no scene.
        handle = bots_ctx.create_sensor(sensor_type, None, "fusion", '{"rate": 30}')
        self.assertTrue(handle.is_valid())

        sensor = sdr.get_python_sensor(bots_ctx, handle)
        self.assertIsInstance(sensor, PyTestSensor)
        self.assertIsNone(sensor.actor)
        # param_args reaches the Python factory verbatim -- a sensor's only config channel.
        self.assertEqual(sensor.param_args, '{"rate": 30}')

        # The sensor's own API is driven straight on the Python instance.
        self.assertEqual(sensor.compute_signal(1.5), 3.0)

        # Framework Reset() reaches the Python override through the C++ base pointer.
        base = bots_ctx.get_sensor(handle)
        assert base is not None
        base.reset()
        self.assertEqual(sensor.reset_calls, 1)
        self.assertEqual(sensor.samples, [])
        del base

        # Destroying the sensor releases the C++ reference to the Python instance.
        ref = weakref.ref(sensor)
        del sensor
        bots_ctx.destroy_sensor(handle)
        gc.collect()
        self.assertIsNone(ref())
        self.assertFalse(bots_ctx.is_valid_sensor(handle))

    def test_several_python_sensor_types_coexist(self) -> None:
        first_type = "PY_TEST_SENSOR_A"
        second_type = "PY_TEST_SENSOR_B"
        bots_ctx = sdr.create_context()

        class OtherPyTestSensor(PyTestSensor):
            pass

        sdr.register_python_sensor(bots_ctx, first_type, PyTestSensor)
        sdr.register_python_sensor(bots_ctx, second_type, OtherPyTestSensor)

        first = bots_ctx.create_sensor(first_type, None, "a", "")
        second = bots_ctx.create_sensor(second_type, None, "b", "")
        try:
            self.assertIsInstance(sdr.get_python_sensor(bots_ctx, first), PyTestSensor)
            self.assertIsInstance(
                sdr.get_python_sensor(bots_ctx, second), OtherPyTestSensor
            )
            # Each instance is distinct, and the type string routes to the right factory.
            found = bots_ctx.find_sensors_by_type(second_type)
            self.assertEqual([h.value for h in found], [second.value])
        finally:
            bots_ctx.destroy_sensor(first)
            bots_ctx.destroy_sensor(second)

    def test_factory_exception_surfaces_as_an_error(self) -> None:
        sensor_type = "PY_TEST_SENSOR_RAISES"
        bots_ctx = sdr.create_context()

        def failing_factory(actor: object, param_args: str) -> PyTestSensor:
            raise ValueError("no sensor for you")

        sdr.register_python_sensor(bots_ctx, sensor_type, failing_factory)
        # The ValueError is logged and reported as a mochi error, not propagated: asserting
        # on that specific type is what proves the factory ran and the error path converted
        # it, rather than the call failing for some unrelated reason.
        with self.assertRaises(sdp.Error):
            bots_ctx.create_sensor(sensor_type, None, "doomed", "")

    def test_sensor_without_reset_is_rejected(self) -> None:
        sensor_type = "PY_TEST_SENSOR_NO_RESET"
        bots_ctx = sdr.create_context()

        class NoResetSensor:
            def __init__(self, actor: object, param_args: str) -> None:
                self.actor = actor

        sdr.register_python_sensor(bots_ctx, sensor_type, NoResetSensor)
        # Every component must implement reset(); the omission is caught at creation, not
        # silently ignored until someone resets between episodes and nothing happens.
        with self.assertRaises(sdp.Error):
            bots_ctx.create_sensor(sensor_type, None, "no_reset", "")

    def test_get_python_sensor_none_for_invalid_handle(self) -> None:
        bots_ctx = sdr.create_context()
        self.assertIsNone(sdr.get_python_sensor(bots_ctx, sdr.SensorHandle()))

    def test_get_python_actuator_none_for_invalid_handle(self) -> None:
        bots_ctx = sdr.create_context()
        self.assertIsNone(sdr.get_python_actuator(bots_ctx, sdr.ActuatorHandle()))

    def test_get_python_sensor_none_for_cpp_sensor(self) -> None:
        bots_ctx = sdr.create_context()
        handle = bots_ctx.create_sensor(
            sdr.CameraSensor.type_name(), None, "cpp_cam", ""
        )
        try:
            self.assertIsNone(sdr.get_python_sensor(bots_ctx, handle))
        finally:
            bots_ctx.destroy_sensor(handle)


class PurePythonComponentsOnABotTest(MochiTestBase):
    """Python sensors and actuators attached to a bot, and loaded from a bot scene."""

    @requires_internal
    @requires_hdf5
    def test_python_sensor_and_actuator_on_a_bot(self) -> None:
        sensor_type = "PY_TEST_SENSOR_ON_BOT"
        actuator_type = "PY_TEST_ACTUATOR_ON_BOT"
        bots_ctx = sdr.create_context()
        sdr.register_python_sensor(bots_ctx, sensor_type, PyTestSensor)
        sdr.register_python_actuator(bots_ctx, actuator_type, PyTestActuator)
        self.assertTrue(bots_ctx.is_actuator_type_registered(actuator_type))

        with tempfile.TemporaryDirectory() as tmp:
            write_assets_tag_root_marker(tmp)
            with open(os.path.join(tmp, "empty.mochi_scene"), "w") as f:
                f.write('{ "scene": { "description": "empty" } }')
            scene_file = os.path.join(tmp, "bot.mochi_bot_scene")
            with open(scene_file, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "metadata": {"name": "PyComponents", "version": "1.0"},
                            "scene": {"baseScene": "empty.mochi_scene"},
                            "bots": [
                                {
                                    "name": "robot",
                                    "path": bot_tag_path(
                                        "bots/arms/fr3/fr3.superdex_bot"
                                    ),
                                    "controllers": [],
                                }
                            ],
                        }
                    )
                )

            bot_scene = sdr.load_bot_scene(scene_file, bots_ctx)
            bot = bot_scene.get_bot("robot")
            self.assertIsNotNone(bot)
            actor = bot.get_articulated_actor()
            self.assertIsNotNone(actor)

            # Created on one of the bot's actors, so the context infers the bot as owner and both
            # components show up in the bot's own handle lists.
            sensor_handle = bots_ctx.create_sensor(sensor_type, actor, "py_sensor", "")
            actuator_handle = bots_ctx.create_actuator(
                actuator_type, actor, "py_actuator", "spec.json"
            )
            self.assertIn(
                sensor_handle.value, [h.value for h in bot.get_sensor_handles()]
            )
            self.assertIn(
                actuator_handle.value, [h.value for h in bot.get_actuator_handles()]
            )

            sensor = sdr.get_python_sensor(bots_ctx, sensor_handle)
            actuator = sdr.get_python_actuator(bots_ctx, actuator_handle)
            self.assertIsInstance(sensor, PyTestSensor)
            self.assertIsInstance(actuator, PyTestActuator)
            # Both were handed the live actor they are attached to.
            self.assertIsNotNone(sensor.actor)
            self.assertIsNotNone(actuator.actor)
            self.assertEqual(actuator.param_args, "spec.json")

            # They survive simulation steps and stay drivable from Python.
            scene = bot_scene.get_scene()
            for step in range(3):
                sensor.compute_signal(float(step))
                actuator.apply(float(step))
                scene.step(0.001)
            self.assertEqual(len(sensor.samples), 3)
            self.assertEqual(len(actuator.applied), 3)

            bots_ctx.get_actuator(actuator_handle).reset()
            self.assertEqual(actuator.reset_calls, 1)

            # Teardown: destroying the bot cascades to the components it owns, dropping the C++
            # references to the Python instances.
            sensor_ref = weakref.ref(sensor)
            actuator_ref = weakref.ref(actuator)
            del sensor, actuator
            del bot, actor, scene
            del bot_scene
            gc.collect()
            self.assertIsNone(sensor_ref())
            self.assertIsNone(actuator_ref())

    @requires_internal
    @requires_hdf5
    def test_scene_level_python_sensor_is_instantiated_by_the_loader(self) -> None:
        sensor_type = "PY_TEST_SENSOR_SCENE_LEVEL"
        bots_ctx = sdr.create_context()
        sdr.register_python_sensor(bots_ctx, sensor_type, PyTestSensor)

        with tempfile.TemporaryDirectory() as tmp:
            write_assets_tag_root_marker(tmp)
            with open(os.path.join(tmp, "empty.mochi_scene"), "w") as f:
                f.write('{ "scene": { "description": "empty" } }')
            scene_file = os.path.join(tmp, "scene_sensor.mochi_bot_scene")
            with open(scene_file, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "metadata": {"name": "PySceneSensor", "version": "1.0"},
                            "scene": {"baseScene": "empty.mochi_scene"},
                            "bots": [
                                {
                                    "name": "robot",
                                    "path": bot_tag_path(
                                        "bots/arms/fr3/fr3.superdex_bot"
                                    ),
                                    "controllers": [],
                                }
                            ],
                            "sensors": [
                                {
                                    "type": sensor_type,
                                    "name": "scene_py_sensor",
                                    "params": '{"fov": 90}',
                                    "translation": [1.0, 2.0, 3.0],
                                    "rotation": [0.0, 0.0, 0.0, 1.0],
                                }
                            ],
                        }
                    )
                )

            bot_scene = sdr.load_bot_scene(scene_file, bots_ctx)

            # The loader built the Python sensor from its type string alone.
            found = bots_ctx.find_sensors_by_name("scene_py_sensor")
            self.assertEqual(len(found), 1)
            sensor = sdr.get_python_sensor(bots_ctx, found[0])
            self.assertIsInstance(sensor, PyTestSensor)
            self.assertEqual(sensor.param_args, '{"fov": 90}')
            # A scene-level sensor has no link actor and belongs to no bot.
            self.assertIsNone(sensor.actor)

            ref = weakref.ref(sensor)
            del sensor
            del bot_scene
            gc.collect()
            self.assertIsNone(ref())
