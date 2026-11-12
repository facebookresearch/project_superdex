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

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from superdex.lab.rllib.recipe_manifest import (
    load_recipe_manifest,
    load_recipes,
    PUBLIC_RECIPE_MANIFEST,
)

EXPECTED_PUBLIC_RECIPE_ASSOCIATIONS = (
    ("superdex_gym/AntNoContact-v0", "train"),
    ("superdex_gym/CartPole-v0", "train"),
    ("superdex_gym/HalfCheetah-v0", "train"),
)


def _write_manifest(
    directory: Path,
    *,
    slug: str,
    relative_path: str | None = None,
) -> Path:
    record: dict[str, object] = {"slug": slug}
    if relative_path is not None:
        record["recipes"] = {"train": relative_path}
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps({"superdex_gym/Test-v0": record}), encoding="utf-8"
    )
    return manifest_path


class RecipeManifestTest(unittest.TestCase):
    def test_public_associations_match_frozen_baseline(self) -> None:
        recipes = load_recipes("train")

        self.assertEqual(
            EXPECTED_PUBLIC_RECIPE_ASSOCIATIONS,
            tuple((env_id, recipe.kind) for env_id, recipe in recipes.items()),
        )
        self.assertEqual(
            ("ant_no_contact", "cart_pole", "half_cheetah"),
            tuple(recipe.slug for recipe in recipes.values()),
        )

    def test_variant_recipe_does_not_fall_back_to_base_or_siblings(self) -> None:
        recipes = load_recipes("train")

        self.assertIn("superdex_gym/AntNoContact-v0", recipes)
        self.assertNotIn("superdex_gym/Ant-v0", recipes)
        self.assertNotIn("superdex_gym/AntFullObservation-v0", recipes)
        self.assertNotIn("superdex_gym/CartPoleActuateOnPole-v0", recipes)

    def test_loaded_recipe_data_is_fresh(self) -> None:
        first = load_recipes("train")
        first["superdex_gym/HalfCheetah-v0"].config["ppo"]["training"]["gamma"] = 0.0

        second = load_recipes("train")

        self.assertEqual(
            0.99,
            second["superdex_gym/HalfCheetah-v0"].config["ppo"]["training"]["gamma"],
        )

    def test_recipes_never_contain_environment_configuration(self) -> None:
        for recipe in load_recipes("train").values():
            with self.subTest(env_id=recipe.env_id):
                self.assertNotIn("env_config", recipe.config)

    def test_manifest_rejects_recipe_path_escape(self) -> None:
        for name, relative_path in (
            ("posix_parent", "../private.json"),
            ("windows_parent", "..\\private.json"),
            ("posix_absolute", "/private.json"),
            ("windows_absolute", "C:\\private.json"),
            ("windows_drive_relative", "C:private.json"),
            ("windows_unc", "\\\\server\\share\\private.json"),
        ):
            with self.subTest(name):
                with tempfile.TemporaryDirectory() as temporary:
                    manifest_path = _write_manifest(
                        Path(temporary), slug="test", relative_path=relative_path
                    )

                    with self.assertRaisesRegex(ValueError, "must stay below"):
                        load_recipe_manifest(manifest_path, "train")

    def test_manifest_loads_resolved_nested_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_directory = Path(temporary)
            recipe_path = manifest_directory / "nested" / "train.json"
            recipe_path.parent.mkdir()
            recipe_path.write_text('{"value": 7}', encoding="utf-8")
            manifest_path = _write_manifest(
                manifest_directory,
                slug="test",
                relative_path="nested/train.json",
            )

            recipes = load_recipe_manifest(manifest_path, "train")

            self.assertEqual({"value": 7}, recipes["superdex_gym/Test-v0"].config)

    def _create_symlink_or_skip(
        self,
        link: Path,
        target: Path,
        *,
        target_is_directory: bool,
    ) -> None:
        try:
            link.symlink_to(target, target_is_directory=target_is_directory)
        except OSError as error:
            if os.name == "nt" and getattr(error, "winerror", None) == 1314:
                self.skipTest(
                    "Windows symbolic-link privilege is unavailable (winerror 1314)."
                )
            raise

    def test_manifest_rejects_directory_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_directory = root / "manifest"
            outside_directory = root / "outside"
            manifest_directory.mkdir()
            outside_directory.mkdir()
            (outside_directory / "train.json").write_text("{}", encoding="utf-8")
            link = manifest_directory / "linked"
            self._create_symlink_or_skip(
                link, outside_directory, target_is_directory=True
            )
            manifest_path = _write_manifest(
                manifest_directory,
                slug="test",
                relative_path="linked/train.json",
            )

            with self.assertRaisesRegex(ValueError, "must stay below"):
                load_recipe_manifest(manifest_path, "train")

    def test_manifest_rejects_file_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_directory = root / "manifest"
            manifest_directory.mkdir()
            outside_recipe = root / "outside.json"
            outside_recipe.write_text("{}", encoding="utf-8")
            link = manifest_directory / "train.json"
            self._create_symlink_or_skip(
                link, outside_recipe, target_is_directory=False
            )
            manifest_path = _write_manifest(
                manifest_directory,
                slug="test",
                relative_path="train.json",
            )

            with self.assertRaisesRegex(ValueError, "must stay below"):
                load_recipe_manifest(manifest_path, "train")

    @unittest.skipUnless(os.name == "nt", "Windows junctions are Windows-specific")
    def test_manifest_rejects_directory_junction_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_directory = root / "manifest"
            outside_directory = root / "outside"
            manifest_directory.mkdir()
            outside_directory.mkdir()
            (outside_directory / "train.json").write_text("{}", encoding="utf-8")
            junction = manifest_directory / "linked"
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(junction), str(outside_directory)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                0,
                result.returncode,
                f"junction creation failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            try:
                manifest_path = _write_manifest(
                    manifest_directory,
                    slug="test",
                    relative_path="linked/train.json",
                )
                with self.assertRaisesRegex(ValueError, "must stay below"):
                    load_recipe_manifest(manifest_path, "train")
            finally:
                if junction.exists():
                    junction.rmdir()

    def test_manifest_and_recipe_paths_are_explicit_files(self) -> None:
        self.assertTrue(PUBLIC_RECIPE_MANIFEST.is_file())
        for recipe in load_recipes("train").values():
            self.assertNotIn("/", recipe.slug)
            self.assertNotIn("\\", recipe.slug)

    def test_manifest_rejects_unsafe_slug(self) -> None:
        for name, slug in (
            ("dot", "."),
            ("dot_dot", ".."),
            ("posix_separator", "nested/path"),
            ("windows_separator", "nested\\path"),
            ("posix_absolute", "/absolute"),
            ("windows_absolute", "C:\\absolute"),
            ("windows_drive_relative", "C:drive"),
            ("windows_unc", "\\\\server\\share"),
            ("less_than", "bad<name"),
            ("greater_than", "bad>name"),
            ("colon", "bad:name"),
            ("quote", 'bad"name'),
            ("pipe", "bad|name"),
            ("question", "bad?name"),
            ("asterisk", "bad*name"),
            ("control_nul", "bad\x00name"),
            ("control_unit_separator", "bad\x1fname"),
            ("trailing_dot", "trailing."),
            ("trailing_space", "trailing "),
            ("reserved_con", "CON"),
            ("reserved_mixed_case", "PrN"),
            ("reserved_aux_extension", "aux.json"),
            ("reserved_nul_extension", "NUL.txt"),
            ("reserved_com_low", "COM1"),
            ("reserved_com_high_extension", "com9.log"),
            ("reserved_lpt_low", "LPT1"),
            ("reserved_lpt_high_extension", "lPt9.json"),
            ("reserved_com_superscript_one", "COM\u00b9"),
            ("reserved_com_superscript_two_extension", "com\u00b2.log"),
            ("reserved_com_superscript_three", "CoM\u00b3"),
            ("reserved_lpt_superscript_one_extension", "LPT\u00b9.json"),
            ("reserved_lpt_superscript_two", "lpt\u00b2"),
            ("reserved_lpt_superscript_three_extension", "LpT\u00b3.txt"),
        ):
            with self.subTest(name):
                with tempfile.TemporaryDirectory() as temporary:
                    manifest_path = _write_manifest(Path(temporary), slug=slug)

                    with self.assertRaisesRegex(ValueError, "filesystem-safe"):
                        load_recipe_manifest(manifest_path, "train")

    def test_manifest_rejects_empty_slug(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = _write_manifest(Path(temporary), slug="")

            with self.assertRaisesRegex(TypeError, "non-empty slug"):
                load_recipe_manifest(manifest_path, "train")

    def test_manifest_accepts_portable_slug(self) -> None:
        for name, slug in (
            ("snake_case", "ant_no_contact"),
            ("hyphen", "cart-pole"),
            ("ordinary_extension", "experiment.v2"),
            ("reserved_prefix", "console"),
            ("non_reserved_com", "com10"),
            ("non_reserved_lpt", "lpt0"),
        ):
            with self.subTest(name):
                with tempfile.TemporaryDirectory() as temporary:
                    manifest_path = _write_manifest(Path(temporary), slug=slug)

                    recipes = load_recipe_manifest(manifest_path, "train")

                    self.assertEqual({}, recipes)
