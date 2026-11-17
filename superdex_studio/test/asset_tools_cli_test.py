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
import subprocess
import tempfile
import unittest
from pathlib import Path


class AssetToolsCliTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="asset-tools-cli-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "output"
        self.urdf = self.root / "source.urdf"
        self.scene = self.root / "scene.mochi_scene"

    def prepare(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [os.environ["PREPARE_URDF_ASSET"], str(self.urdf), str(self.output)],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def read_scene(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [os.environ["READ_ASSET_ZERO_STEP"], str(self.scene)],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def assert_output_empty(self) -> None:
        if self.output.exists():
            self.assertEqual(list(self.output.iterdir()), [])

    def write_valid_urdf(self) -> None:
        self.urdf.write_text(
            '<robot name="test"><link name="base">'
            '<visual><geometry><box size="1 1 1"/></geometry></visual>'
            '<collision><geometry><box size="1 1 1"/></geometry></collision>'
            "</link></robot>"
        )

    def test_missing_urdf_does_not_block_retry(self) -> None:
        for existing in (False, True):
            with self.subTest(existing=existing):
                self.output = self.root / f"output-{existing}"
                if existing:
                    self.output.mkdir()
                result = self.prepare()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Failed to load", result.stderr)
                self.assert_output_empty()
                self.assertEqual(self.output.exists(), existing)
                self.write_valid_urdf()
                result = self.prepare()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((self.output / "asset.superdex_bot").is_file())
                self.urdf.unlink()

    def test_malformed_urdf_does_not_create_output(self) -> None:
        self.urdf.write_text("<robot")
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())
        self.assertNotEqual(result.stderr, "")

    def test_missing_mesh_keeps_import_warnings_and_output_empty(self) -> None:
        self.urdf.write_text(
            '<robot name="test"><link name="base">'
            "<visual/>"
            '<collision><geometry><mesh filename="missing.obj"/></geometry></collision>'
            "</link></robot>"
        )
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing mesh", result.stderr)
        self.assertIn("Import warning:", result.stderr)
        self.assertIn("no geometry", result.stderr)
        self.assertFalse(self.output.exists())

    def test_unsafe_link_name_does_not_create_output(self) -> None:
        self.urdf.write_text('<robot name="test"><link name="../escape"/></robot>')
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root / "escape.glb").exists())

    def test_case_colliding_link_names_do_not_create_output(self) -> None:
        self.urdf.write_text(
            '<robot name="test"><link name="Finger">'
            '<visual><geometry><box size="1 1 1"/></geometry></visual>'
            '</link><link name="finger">'
            '<visual><geometry><sphere radius="0.5"/></geometry></visual>'
            '</link><joint name="weld" type="fixed">'
            '<parent link="Finger"/><child link="finger"/></joint></robot>'
        )
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("distinct ignoring case", result.stderr)
        self.assertFalse(self.output.exists())
        repeated = self.prepare()
        self.assertEqual(repeated.stderr, result.stderr)
        self.assertFalse(self.output.exists())

    def test_late_render_failure_cleans_owned_files_and_retains_warnings(self) -> None:
        (self.root / "broken.obj").write_text("this is not a mesh")
        self.urdf.write_text(
            '<robot name="test"><link name="first">'
            '<visual><geometry><box size="1 1 1"/></geometry></visual>'
            '<collision><geometry><box size="1 1 1"/></geometry></collision>'
            '</link><link name="second">'
            '<visual><geometry><mesh filename="broken.obj"/></geometry></visual>'
            '</link><joint name="weld" type="fixed">'
            '<parent link="first"/><child link="second"/></joint></robot>'
        )
        for existing in (False, True):
            with self.subTest(existing=existing):
                self.output = self.root / f"late-output-{existing}"
                if existing:
                    self.output.mkdir()
                result = self.prepare()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Render conversion failed for second", result.stderr)
                self.assertIn("Import warning:", result.stderr)
                self.assertIn("broken.obj", result.stderr)
                self.assert_output_empty()
                self.assertEqual(self.output.exists(), existing)
                repeated = self.prepare()
                self.assertNotEqual(repeated.returncode, 0)
                self.assertEqual(repeated.stderr, result.stderr)
                self.assert_output_empty()
        self.write_valid_urdf()
        result = self.prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        bot = json.loads((self.output / "asset.superdex_bot").read_text())
        self.assertEqual(len(bot["links"]), 1)
        self.assertTrue((self.output / "render" / "base.glb").is_file())
        recipe_file = self.output / "recipes" / "base.StudioProcessing.json"
        self.assertTrue(recipe_file.is_file())
        self.assertFalse((self.output / "collision" / "base.mochi.h5").exists())
        self.assertIn("run the emitted collision recipes", result.stdout)

    def test_existing_output_data_is_preserved(self) -> None:
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_bytes(b"user data\x00")
        self.write_valid_urdf()
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_bytes(), b"user data\x00")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_output_file_is_preserved(self) -> None:
        self.output.write_bytes(b"user file")
        self.write_valid_urdf()
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output directory must be empty", result.stderr)
        self.assertEqual(self.output.read_bytes(), b"user file")

    def test_missing_or_malformed_scene_emits_only_stderr(self) -> None:
        for contents in (None, "{bad"):
            with self.subTest(contents=contents):
                if contents is not None:
                    self.scene.write_text(contents)
                result = self.read_scene()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn("prefab", result.stderr)

    def test_native_warning_does_not_pollute_json(self) -> None:
        self.scene.write_text(
            json.dumps(
                {
                    "actors": {
                        "articulated": [
                            {
                                "name": "massless",
                                "joints": [{"type": "Free"}],
                                "links": [
                                    {"name": "base", "colliderType": "None", "mass": 2}
                                ],
                            }
                        ]
                    }
                }
            )
        )
        result = self.read_scene()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertIn("Warning:", result.stderr)
        self.assertIn("massless", result.stderr)
        self.assertEqual(report["articulations"][0]["links"][0]["mass_kg"], 0)

    def write_seven_dof_scene(self, name: str, link_name: str) -> None:
        link = {
            "name": link_name,
            "colliderType": "None",
            "mass": 2,
            "centerOfMass": [0, 0, 0],
            "momentOfInertia": [1, 0, 0, 2, 0, 3],
        }
        actor = {
            "name": name,
            "translation": [1, 2, 3],
            "joints": [
                {"name": "root", "type": "Free"},
                {
                    "name": "hinge",
                    "type": "Revolute",
                    "axis": [0, 0, 1],
                    "minLimit": [0, 0, -0.4],
                    "maxLimit": [0, 0, 0.7],
                },
            ],
            "links": [link, dict(link, name="child", parentLink=0)],
            "jointPose": [0] * 6 + [0.25],
            "jointVelocities": [0] * 6 + [0.5],
        }
        self.scene.write_text(json.dumps({"actors": {"articulated": [actor]}}))

    def test_escaped_names_and_seven_dof_native_state_are_preserved(self) -> None:
        name = 'robot"\\\n\t'
        link_name = 'base"\n\t'
        inertia = [1, 0, 0, 2, 0, 3]
        pose = [0] * 6 + [0.25]
        velocity = [0] * 6 + [0.5]
        self.write_seven_dof_scene(name, link_name)
        result = self.read_scene()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["positive_steps"], 0)
        self.assertIn(report["precision_bytes"], (4, 8))
        actor = report["articulations"][0]
        self.assertEqual(actor["name"], name)
        self.assertEqual(actor["num_dofs"], 7)
        self.assertFalse(actor["has_pose_controller"])
        self.assertEqual(actor["q"], pose)
        self.assertEqual(actor["qd"], velocity)
        self.assertEqual(actor["external_forces"], [0] * 7)
        self.assertEqual(actor["limits"][:6], [["-inf", "+inf"]] * 6)
        self.assertAlmostEqual(actor["limits"][6][0], -0.4, places=6)
        self.assertAlmostEqual(actor["limits"][6][1], 0.7, places=6)
        base = actor["links"][0]
        self.assertEqual(base["name"], name + "/" + link_name)
        self.assertEqual(base["mass_kg"], 2)
        self.assertEqual(base["inertia_com_kg_m2_real6"], inertia)
        self.assertEqual(base["com_local_m"], [0, 0, 0])
        self.assertEqual(base["world_translation"], [1, 2, 3])
        self.assertFalse(base["is_static"])

    def test_zero_dof_static_articulation_remains_readable(self) -> None:
        self.scene.write_text(
            json.dumps(
                {
                    "actors": {
                        "articulated": [
                            {
                                "name": "static",
                                "joints": [{"type": "Hard"}],
                                "links": [{"name": "base", "colliderType": "None"}],
                            }
                        ]
                    }
                }
            )
        )
        result = self.read_scene()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        actor = report["articulations"][0]
        self.assertEqual(actor["num_dofs"], 0)
        self.assertEqual(actor["q"], [])
        self.assertEqual(actor["qd"], [])
        self.assertTrue(actor["links"][0]["is_static"])
