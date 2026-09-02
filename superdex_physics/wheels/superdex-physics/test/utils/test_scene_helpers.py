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

import json
import os
import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import patch

import superdex.physics as sdp
from superdex.physics.paths import get_assets_root
from superdex.physics.utils.scene_helpers import (
    create_scene_from_prefab,
    load_bot_task_prefab,
)

CHILD_PREFAB_JSON = """{
    "actors": {
        "rigid": [{
            "name": "ChildActor",
            "shape": "duck/duck_coarse_mesh.mochi.json"
        }]
    }
}"""

PARENT_PREFAB_JSON = """{
    "actors": {
        "rigid": [{
            "name": "ParentActor",
            "shape": "cube/cube_minimal.mochi.json"
        }]
    },
    "prefabs": [{
        "name": "child",
        "path": "./child.mochi_scene"
    }]
}"""


class CreateSceneFromPrefabTest(unittest.TestCase):
    def setUp(self) -> None:
        sdp.initialize(0)
        self._temp_dir = tempfile.mkdtemp(prefix="mochi_scene_helpers_test_")
        assets_root = get_assets_root()
        for shape_dir in ("duck", "cube"):
            os.symlink(
                os.path.join(str(assets_root), shape_dir),
                os.path.join(self._temp_dir, shape_dir),
            )
        child_path = os.path.join(self._temp_dir, "child.mochi_scene")
        with open(child_path, "w") as f:
            f.write(CHILD_PREFAB_JSON)
        self._parent_path = os.path.join(self._temp_dir, "parent.mochi_scene")
        with open(self._parent_path, "w") as f:
            f.write(PARENT_PREFAB_JSON)

    def tearDown(self) -> None:
        shutil.rmtree(self._temp_dir, ignore_errors=True)
        sdp.shutdown()

    def _create_and_verify(
        self,
        prefab,
        expected_names=("ParentActor", "child/ChildActor"),
        **kwargs,
    ) -> sdp.Scene:
        kwargs.setdefault("root_dir", self._temp_dir)
        scene = create_scene_from_prefab(prefab, **kwargs)
        self.assertIsNotNone(scene)
        self.assertIsInstance(scene, sdp.Scene)
        self.assertEqual(len(expected_names), scene.get_num_actors())
        names: list[str] = []
        scene.for_each_actor(lambda a: names.append(a.get_name()))
        for name in expected_names:
            self.assertIn(name, names)
        return scene

    def test_create_from_file_path(self) -> None:
        # Relative path resolved against root_dir
        scene = self._create_and_verify("parent.mochi_scene")
        sdp.destroy_scene(scene)

        # Absolute path used as-is
        scene = self._create_and_verify(self._parent_path)
        sdp.destroy_scene(scene)

        # No root_dir — falls back to the assets root
        with patch(
            "superdex.physics.utils.scene_helpers.get_assets_root",
            return_value=pathlib.Path(self._temp_dir),
        ):
            scene = self._create_and_verify("parent.mochi_scene", root_dir="")
            sdp.destroy_scene(scene)

        # Custom scene_name
        scene = self._create_and_verify("parent.mochi_scene", scene_name="my_scene")
        sdp.destroy_scene(scene)

        # Custom PrefabParams
        params = sdp.prefab.PrefabParams()
        params.name = "custom"
        scene = self._create_and_verify(
            "parent.mochi_scene",
            expected_names=("custom/ParentActor", "custom/child/ChildActor"),
            params=params,
        )
        sdp.destroy_scene(scene)

    def test_create_from_scene_prefab(self) -> None:
        # Fully loaded ScenePrefab
        prefab = sdp.prefab.load_from_file(self._parent_path, self._temp_dir)
        scene = self._create_and_verify(prefab)
        sdp.destroy_scene(scene)

        # Shallow loaded ScenePrefab (ensure_fully_loaded called internally)
        prefab = sdp.prefab.shallow_load_from_file(self._parent_path)
        scene = self._create_and_verify(prefab)
        sdp.destroy_scene(scene)


class LoadBotTaskPrefabTest(unittest.TestCase):
    """Tests for `load_bot_task_prefab` — parse-only, no Mochi init or scene build."""

    def setUp(self) -> None:
        self._temp_dir = tempfile.mkdtemp(prefix="mochi_load_bot_task_prefab_test_")

    def tearDown(self) -> None:
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def _write(self, filename: str, payload: dict) -> str:
        path = os.path.join(self._temp_dir, filename)
        with open(path, "w") as f:
            f.write(json.dumps(payload))
        return path

    def test_spawns_round_trip(self) -> None:
        path = self._write(
            "two_spawns.mochi_bot_task",
            {
                "metadata": {"name": "Tiny Task", "version": "1.0"},
                "spawns": [
                    {
                        "name": "Foam",
                        "prefabName": "Foam",
                        "translation": [0.0, 0.0, 0.05],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                    },
                    {
                        "name": "m8_knob",
                        "prefabName": "M8_Knob",
                        "parent": "Foam",
                        "translation": [0.1, 0.2, 0.25],
                        "rotation": [0.7071068, 0.0, 0.0, 0.7071068],
                    },
                ],
            },
        )

        params = load_bot_task_prefab(path)

        self.assertEqual(str(params.metadata.name), "Tiny Task")
        self.assertEqual(len(params.spawns), 2)

        self.assertEqual(str(params.spawns[0].name), "Foam")
        self.assertEqual(str(params.spawns[0].prefab_name), "Foam")
        self.assertTrue(str(params.spawns[0].parent) in ("", "task_root"))
        self.assertAlmostEqual(
            params.spawns[0].parent_from_spawn.translation[2], 0.05, places=5
        )

        self.assertEqual(str(params.spawns[1].name), "m8_knob")
        self.assertEqual(str(params.spawns[1].prefab_name), "M8_Knob")
        self.assertEqual(str(params.spawns[1].parent), "Foam")
        self.assertAlmostEqual(
            params.spawns[1].parent_from_spawn.translation[0], 0.1, places=5
        )
        self.assertAlmostEqual(
            params.spawns[1].parent_from_spawn.rotation[0], 0.7071068, places=5
        )
        self.assertAlmostEqual(
            params.spawns[1].parent_from_spawn.rotation[3], 0.7071068, places=5
        )

    def test_empty_name_raises(self) -> None:
        path = self._write(
            "empty_name.mochi_bot_task",
            {"spawns": [{"name": "", "prefabName": "Foam"}]},
        )
        with self.assertRaises(sdp.Error):  # pyre-ignore[16]
            load_bot_task_prefab(path)

    def test_task_root_name_raises(self) -> None:
        path = self._write(
            "task_root_name.mochi_bot_task",
            {"spawns": [{"name": "task_root", "prefabName": "Foam"}]},
        )
        with self.assertRaises(sdp.Error):  # pyre-ignore[16]
            load_bot_task_prefab(path)

    def test_duplicate_name_raises(self) -> None:
        path = self._write(
            "duplicate_name.mochi_bot_task",
            {
                "spawns": [
                    {"name": "obj", "prefabName": "Foam"},
                    {"name": "obj", "prefabName": "M8_Knob"},
                ]
            },
        )
        with self.assertRaises(sdp.Error):  # pyre-ignore[16]
            load_bot_task_prefab(path)
