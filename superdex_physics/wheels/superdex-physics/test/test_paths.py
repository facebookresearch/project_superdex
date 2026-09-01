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

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .module_stand_ins import load_physics_module

# Loaded by path so this module does not depend on a sibling test having already
# registered the `superdex.physics.*` stand-ins. `paths` imports from `environment`.
load_physics_module("environment")
paths = load_physics_module("paths")


def _physics_module_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "superdex_physics"
        / "wheels"
        / "superdex-physics"
        / "superdex"
        / "physics"
        / "paths.py"
    )


class PathsTest(unittest.TestCase):
    def test_primary_env_wins_over_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as canonical_dir:
            with tempfile.TemporaryDirectory() as legacy_dir:
                with patch.dict(
                    os.environ,
                    {
                        paths.ASSETS_PATH_ENV_VAR: canonical_dir,
                        paths.LEGACY_ASSETS_PATH_ENV_VAR: legacy_dir,
                    },
                    clear=True,
                ):
                    self.assertEqual(
                        Path(canonical_dir).resolve(),
                        paths._find_assets_path(),
                    )

    def test_primary_env_wins_over_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as canonical_dir:
            with tempfile.TemporaryDirectory() as source_root:
                module_path = _physics_module_path(Path(source_root))
                module_path.parent.mkdir(parents=True)
                module_path.write_text("# stub\n")
                (Path(source_root) / "assets").mkdir()

                with patch.dict(
                    os.environ,
                    {paths.ASSETS_PATH_ENV_VAR: canonical_dir},
                    clear=True,
                ):
                    with patch.object(paths, "__file__", str(module_path)):
                        self.assertEqual(
                            Path(canonical_dir).resolve(),
                            paths._find_assets_path(),
                        )

    def test_source_tree_fallback_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as source_root:
            repo_root = Path(source_root)
            assets_root = repo_root / "assets"
            module_path = _physics_module_path(repo_root)
            module_path.parent.mkdir(parents=True)
            module_path.write_text("# stub\n")
            assets_root.mkdir()

            with patch.dict(os.environ, {}, clear=True):
                with patch.object(paths, "__file__", str(module_path)):
                    self.assertEqual(
                        assets_root.resolve(), paths._source_tree_assets_root()
                    )
                    self.assertEqual(assets_root.resolve(), paths._find_assets_path())

    def test_source_tree_fallback_skips_native_physics_assets(self) -> None:
        with tempfile.TemporaryDirectory() as source_root:
            repo_root = Path(source_root)
            assets_root = repo_root / "assets"
            physics_assets = repo_root / "superdex_physics" / "assets"
            module_path = _physics_module_path(repo_root)
            module_path.parent.mkdir(parents=True)
            module_path.write_text("# stub\n")
            assets_root.mkdir()
            physics_assets.mkdir(parents=True)

            with patch.object(paths, "__file__", str(module_path)):
                self.assertEqual(
                    assets_root.resolve(), paths._source_tree_assets_root()
                )

    def test_resolve_asset_uses_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as assets_dir:
            asset_path = Path(assets_dir) / "custom" / "example.asset"
            asset_path.parent.mkdir(parents=True)
            asset_path.write_text("placeholder")

            with patch.dict(
                os.environ,
                {paths.ASSETS_PATH_ENV_VAR: assets_dir},
                clear=True,
            ):
                self.assertEqual(
                    asset_path.resolve(),
                    paths.resolve_asset("custom/example.asset"),
                )

    def test_resolve_asset_uses_source_tree_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as source_root:
            repo_root = Path(source_root)
            assets_root = repo_root / "assets"
            asset_path = assets_root / "custom" / "example.asset"
            module_path = _physics_module_path(repo_root)
            module_path.parent.mkdir(parents=True)
            module_path.write_text("# stub\n")
            asset_path.parent.mkdir(parents=True)
            asset_path.write_text("placeholder")

            with patch.dict(os.environ, {}, clear=True):
                with patch.object(paths, "__file__", str(module_path)):
                    self.assertEqual(
                        asset_path.resolve(),
                        paths.resolve_asset("custom/example.asset"),
                    )

    def test_empty_env_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as source_root:
            repo_root = Path(source_root)
            assets_root = repo_root / "assets"
            module_path = _physics_module_path(repo_root)
            module_path.parent.mkdir(parents=True)
            module_path.write_text("# stub\n")
            assets_root.mkdir()

            with patch.dict(
                os.environ,
                {
                    paths.ASSETS_PATH_ENV_VAR: "",
                    paths.LEGACY_ASSETS_PATH_ENV_VAR: "",
                },
                clear=True,
            ):
                with patch.object(paths, "__file__", str(module_path)):
                    self.assertEqual(assets_root.resolve(), paths.get_assets_root())

    def test_missing_assets_return_none(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(paths, "_source_tree_assets_root", return_value=None):
                self.assertIsNone(paths._find_assets_path())

    def test_missing_assets_raise_clear_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(paths, "_source_tree_assets_root", return_value=None):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "Set SUPERDEX_ASSETS_PATH",
                ):
                    paths.get_assets_root()
