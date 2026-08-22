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

"""Tests for the component API shared by controllers, sensors and actuators.

These cover the ComponentBase surface every component inherits (is_valid, get_actor,
get_owning_bot, get_name, get_type_name) plus the per-kind additions -- SensorBase's pose
accessors and ControllerBase's configure_from_scene_entry.
"""

from __future__ import annotations

import json
import os
import tempfile

from test.conftest import (
    bot_tag_path,
    bots,
    mochi,
    MochiTestBase,
    requires_hdf5,
    requires_internal,
    write_assets_tag_root_marker,
)


def _make_scene_json(bot_path: str) -> str:
    """A scene with one bot carrying a joint-space PD controller."""
    return json.dumps(
        {
            "metadata": {"name": "ComponentApi", "version": "1.0"},
            "scene": {"baseScene": "empty.mochi_scene"},
            "bots": [
                {
                    "name": "robot",
                    "path": bot_path,
                    "controllers": [
                        {"type": bots.ControllerBasicJscPd.type_name(), "name": "jsc"}
                    ],
                }
            ],
        }
    )


class ComponentIdentityTest(MochiTestBase):
    """The identity every component reports, on a component that needs no scene."""

    def test_actor_less_sensor_reports_its_identity(self) -> None:
        bots_ctx = bots.create_context()
        handle = bots_ctx.create_sensor(
            bots.CameraSensor.type_name(), None, "fixed_cam", ""
        )
        try:
            sensor = bots_ctx.get_sensor(handle)
            assert sensor is not None
            self.assertTrue(sensor.is_valid())
            self.assertEqual(sensor.get_name(), "fixed_cam")
            self.assertEqual(sensor.get_type_name(), bots.CameraSensor.type_name())
            # Created with no link actor, so it is bound to nothing and owned by no bot.
            self.assertIsNone(sensor.get_actor())
            self.assertIsNone(sensor.get_owning_bot())
        finally:
            bots_ctx.destroy_sensor(handle)

    def test_sensor_created_without_a_name_reports_an_empty_one(self) -> None:
        bots_ctx = bots.create_context()
        handle = bots_ctx.create_sensor(bots.CameraSensor.type_name(), None, "", "")
        try:
            sensor = bots_ctx.get_sensor(handle)
            assert sensor is not None
            self.assertEqual(sensor.get_name(), "")
        finally:
            bots_ctx.destroy_sensor(handle)


class SensorPoseTest(MochiTestBase):
    def test_parent_pose_round_trips_and_resolves_to_world(self) -> None:
        bots_ctx = bots.create_context()
        handle = bots_ctx.create_sensor(bots.CameraSensor.type_name(), None, "cam", "")
        try:
            sensor = bots_ctx.get_sensor(handle)
            assert sensor is not None
            pose = mochi.TransformRT([1.0, 2.0, 3.0])
            sensor.set_parent_from_sensor(pose)
            self.assertEqual(sensor.get_parent_from_sensor(), pose)
            # With no actor the scene root is the parent, so world == parent_from_sensor.
            self.assertEqual(sensor.get_world_transform(), pose)
        finally:
            bots_ctx.destroy_sensor(handle)


class ComponentOnABotTest(MochiTestBase):
    """The same accessors on components that are attached to a bot."""

    @requires_internal
    @requires_hdf5
    def test_bot_owned_components_report_their_actor_and_owner(self) -> None:
        bots_ctx = bots.create_context()
        with tempfile.TemporaryDirectory() as tmp:
            write_assets_tag_root_marker(tmp)
            with open(os.path.join(tmp, "empty.mochi_scene"), "w") as f:
                f.write('{ "scene": { "description": "empty" } }')
            scene_file = os.path.join(tmp, "components.mochi_bot_scene")
            with open(scene_file, "w") as f:
                f.write(
                    _make_scene_json(bot_tag_path("bots/arms/fr3/fr3.superdex_bot"))
                )

            bot_scene = bots.load_bot_scene(scene_file, bots_ctx)
            bot = bot_scene.get_bot("robot")

            controller = bots_ctx.get_controller(
                bot_scene.get_controller_handle("robot/jsc")
            )
            assert controller is not None
            self.assertEqual(controller.get_name(), "jsc")
            self.assertEqual(
                controller.get_type_name(), bots.ControllerBasicJscPd.type_name()
            )
            self.assertIsNotNone(controller.get_actor())
            # The owning bot is inferred from the actor the controller was built on.
            owner = controller.get_owning_bot()
            assert owner is not None
            self.assertEqual(owner.get_name(), bot.get_name())
            # A sensor on a bot link resolves its world pose through the link's actor, so
            # it is offset from the pose expressed in the link frame.
            sensor_handle = bots_ctx.create_sensor(
                bots.CameraSensor.type_name(), bot.get_articulated_actor(), "wrist", ""
            )
            try:
                sensor = bots_ctx.get_sensor(sensor_handle)
                assert sensor is not None
                sensor.set_parent_from_sensor(mochi.TransformRT([0.0, 0.0, 0.5]))
                self.assertIsNotNone(sensor.get_actor())
                sensor_owner = sensor.get_owning_bot()
                assert sensor_owner is not None
                self.assertEqual(sensor_owner.get_name(), bot.get_name())
                self.assertIsInstance(sensor.get_world_transform(), mochi.TransformRT)
            finally:
                bots_ctx.destroy_sensor(sensor_handle)

    @requires_internal
    @requires_hdf5
    def test_configure_from_scene_entry_applies_params(self) -> None:
        bots_ctx = bots.create_context()
        with tempfile.TemporaryDirectory() as tmp:
            write_assets_tag_root_marker(tmp)
            with open(os.path.join(tmp, "empty.mochi_scene"), "w") as f:
                f.write('{ "scene": { "description": "empty" } }')
            scene_file = os.path.join(tmp, "configure.mochi_bot_scene")
            with open(scene_file, "w") as f:
                f.write(
                    _make_scene_json(bot_tag_path("bots/arms/fr3/fr3.superdex_bot"))
                )

            bot_scene = bots.load_bot_scene(scene_file, bots_ctx)
            controller = bots_ctx.get_controller(
                bot_scene.get_controller_handle("robot/jsc")
            )
            assert controller is not None

            # The hook the scene loader uses is callable directly: empty args mean
            # "use defaults", which every controller must accept.
            controller.configure_from_scene_entry("", "")
            self.assertIsNotNone(controller.get_params())
