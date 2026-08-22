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

import importlib.machinery
import os
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from .module_stand_ins import ensure_physics_package, load_physics_module

ensure_physics_package()
load_physics_module("environment")
loader = load_physics_module("loader")

_PHYSICS_PAYLOAD = loader.NativePayload(
    subpackage="physics",
    distribution="superdex-physics",
    fp64_package="superdex_physics_fp64",
)
_ROBOTICS_PAYLOAD = loader.NativePayload(
    subpackage="robotics",
    distribution="superdex-robotics",
    fp64_package="superdex_robotics_fp64",
)


def _package_spec(name: str, package_dir: Path) -> importlib.machinery.ModuleSpec:
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(package_dir)]
    return spec


def _find_spec_for(installed: dict[str, Path]) -> object:
    """Stand in for ``importlib.util.find_spec`` over a fixed set of installed packages."""

    def find_spec(name: str) -> importlib.machinery.ModuleSpec | None:
        package_dir = installed.get(name)
        return None if package_dir is None else _package_spec(name, package_dir)

    return find_spec


class LoaderTest(unittest.TestCase):
    def test_packaged_native_dir_uses_the_flat_base_payload_for_single_precision(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "superdex" / "physics"
            native_dir = package_dir / "_native"
            native_dir.mkdir(parents=True)

            with (
                patch.object(loader, "__file__", str(package_dir / "loader.py")),
                patch.object(loader, "USE_DOUBLE_PRECISION", False),
                patch.object(loader, "PRECISION_NAME", "single"),
            ):
                self.assertEqual(
                    native_dir,
                    loader._packaged_native_dir(_PHYSICS_PAYLOAD),
                )

    def test_packaged_native_dir_uses_the_flat_robotics_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "superdex" / "physics"
            native_dir = Path(temp_dir) / "superdex" / "robotics" / "_native"
            native_dir.mkdir(parents=True)

            with (
                patch.object(loader, "__file__", str(package_dir / "loader.py")),
                patch.object(loader, "USE_DOUBLE_PRECISION", False),
                patch.object(loader, "PRECISION_NAME", "single"),
            ):
                self.assertEqual(
                    native_dir,
                    loader._packaged_native_dir(_ROBOTICS_PAYLOAD),
                )

    def test_packaged_native_dir_uses_the_fp64_sibling_package_for_physics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "superdex" / "physics"
            (package_dir / "_native").mkdir(parents=True)
            fp64_package_dir = Path(temp_dir) / "superdex_physics_fp64"
            fp64_native_dir = fp64_package_dir / "_native"
            fp64_native_dir.mkdir(parents=True)

            with (
                patch.object(loader, "__file__", str(package_dir / "loader.py")),
                patch.object(loader, "USE_DOUBLE_PRECISION", True),
                patch.object(loader, "PRECISION_NAME", "double"),
                patch.object(
                    loader.importlib.util,
                    "find_spec",
                    _find_spec_for({"superdex_physics_fp64": fp64_package_dir}),
                ),
            ):
                self.assertEqual(
                    fp64_native_dir,
                    loader._packaged_native_dir(_PHYSICS_PAYLOAD),
                )

    def test_packaged_native_dir_uses_the_fp64_sibling_package_for_robotics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "superdex" / "physics"
            fp64_package_dir = Path(temp_dir) / "superdex_robotics_fp64"
            fp64_native_dir = fp64_package_dir / "_native"
            fp64_native_dir.mkdir(parents=True)

            with (
                patch.object(loader, "__file__", str(package_dir / "loader.py")),
                patch.object(loader, "USE_DOUBLE_PRECISION", True),
                patch.object(loader, "PRECISION_NAME", "double"),
                patch.object(
                    loader.importlib.util,
                    "find_spec",
                    _find_spec_for({"superdex_robotics_fp64": fp64_package_dir}),
                ),
            ):
                self.assertEqual(
                    fp64_native_dir,
                    loader._packaged_native_dir(_ROBOTICS_PAYLOAD),
                )

    def test_packaged_native_dir_is_absent_without_the_fp64_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "superdex" / "physics"
            (package_dir / "_native").mkdir(parents=True)

            with (
                patch.object(loader, "__file__", str(package_dir / "loader.py")),
                patch.object(loader, "USE_DOUBLE_PRECISION", True),
                patch.object(loader, "PRECISION_NAME", "double"),
                patch.object(loader.importlib.util, "find_spec", _find_spec_for({})),
                # Pinned at an empty tree: a real fp64 install in the interpreter's
                # site-packages would otherwise answer this lookup.
                patch.object(
                    loader, "site_packages_root", lambda: Path(temp_dir) / "absent"
                ),
            ):
                self.assertIsNone(loader._packaged_native_dir(_PHYSICS_PAYLOAD))

    def test_packaged_native_dir_falls_back_to_site_packages_when_editable(
        self,
    ) -> None:
        """An editable install splits the facade from its payload: `loader.py` resolves
        into the source tree, while CMake installs the payload under site-packages."""

        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "src" / "superdex" / "physics"
            source_dir.mkdir(parents=True)
            site_packages = Path(temp_dir) / "site-packages"
            native_dir = site_packages / "superdex" / "physics" / "_native"
            native_dir.mkdir(parents=True)

            with (
                patch.object(loader, "__file__", str(source_dir / "loader.py")),
                patch.object(loader, "USE_DOUBLE_PRECISION", False),
                patch.object(loader, "PRECISION_NAME", "single"),
                patch.object(loader, "site_packages_root", lambda: site_packages),
            ):
                self.assertEqual(
                    native_dir,
                    loader._packaged_native_dir(_PHYSICS_PAYLOAD),
                )

    def test_packaged_native_dir_falls_back_to_site_packages_for_fp64(self) -> None:
        """The fp64 spec points at the source tree under an editable install too."""

        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "src" / "superdex" / "physics"
            source_dir.mkdir(parents=True)
            fp64_source_dir = Path(temp_dir) / "src" / "superdex_physics_fp64"
            fp64_source_dir.mkdir(parents=True)
            site_packages = Path(temp_dir) / "site-packages"
            native_dir = site_packages / "superdex_physics_fp64" / "_native"
            native_dir.mkdir(parents=True)

            with (
                patch.object(loader, "__file__", str(source_dir / "loader.py")),
                patch.object(loader, "USE_DOUBLE_PRECISION", True),
                patch.object(loader, "PRECISION_NAME", "double"),
                patch.object(
                    loader.importlib.util,
                    "find_spec",
                    _find_spec_for({"superdex_physics_fp64": fp64_source_dir}),
                ),
                patch.object(loader, "site_packages_root", lambda: site_packages),
            ):
                self.assertEqual(
                    native_dir,
                    loader._packaged_native_dir(_PHYSICS_PAYLOAD),
                )

    def test_packaged_native_dir_prefers_the_facade_sibling_over_site_packages(
        self,
    ) -> None:
        """A wheel install must not be diverted by a stray site-packages candidate."""

        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "superdex" / "physics"
            native_dir = package_dir / "_native"
            native_dir.mkdir(parents=True)
            site_packages = Path(temp_dir) / "site-packages"
            (site_packages / "superdex" / "physics" / "_native").mkdir(parents=True)

            with (
                patch.object(loader, "__file__", str(package_dir / "loader.py")),
                patch.object(loader, "USE_DOUBLE_PRECISION", False),
                patch.object(loader, "PRECISION_NAME", "single"),
                patch.object(loader, "site_packages_root", lambda: site_packages),
            ):
                self.assertEqual(
                    native_dir,
                    loader._packaged_native_dir(_PHYSICS_PAYLOAD),
                )

    def test_ensure_packaged_native_search_path_prepends_native_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "superdex" / "physics"
            native_dir = package_dir / "_native"
            native_dir.mkdir(parents=True)

            with (
                patch.object(loader, "_packaged_native_dir", return_value=native_dir),
                patch.object(loader.sys, "path", ["existing"]),
                patch.object(loader, "_PACKAGED_NATIVE_SEARCH_PATHS", set()),
            ):
                self.assertEqual(
                    native_dir,
                    loader._ensure_packaged_native_search_path(_PHYSICS_PAYLOAD),
                )
                self.assertEqual([str(native_dir), "existing"], loader.sys.path)

    def test_ensure_packaged_native_search_path_retains_windows_dll_handle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "superdex" / "physics"
            native_dir = package_dir / "_native"
            native_dir.mkdir(parents=True)
            handle = object()

            with (
                patch.object(loader, "_packaged_native_dir", return_value=native_dir),
                patch.object(loader.sys, "platform", "win32"),
                patch.object(loader.sys, "path", []),
                patch.object(loader, "_PACKAGED_NATIVE_SEARCH_PATHS", set()),
                patch.object(loader, "_PACKAGED_NATIVE_DLL_DIR_HANDLES", []),
                patch.object(
                    loader.os,
                    "add_dll_directory",
                    return_value=handle,
                    create=True,
                ) as add_dll_directory,
            ):
                loader._ensure_packaged_native_search_path(_PHYSICS_PAYLOAD)

                add_dll_directory.assert_called_once_with(str(native_dir))
                self.assertEqual([handle], loader._PACKAGED_NATIVE_DLL_DIR_HANDLES)

    def test_primary_precision_env_wins_over_legacy(self) -> None:
        with patch.dict(
            os.environ,
            {
                loader.PRECISION_ENV_VAR: "single",
                loader.LEGACY_PRECISION_ENV_VAR: "double",
            },
            clear=True,
        ):
            self.assertFalse(loader._get_use_double_precision())
            self.assertEqual("single", os.environ[loader.PRECISION_ENV_VAR])
            self.assertEqual("single", os.environ[loader.LEGACY_PRECISION_ENV_VAR])

    def test_legacy_precision_env_is_accepted(self) -> None:
        with patch.dict(
            os.environ,
            {loader.LEGACY_PRECISION_ENV_VAR: "float64"},
            clear=True,
        ):
            self.assertTrue(loader._get_use_double_precision())
            self.assertEqual("double", os.environ[loader.PRECISION_ENV_VAR])
            self.assertEqual("double", os.environ[loader.LEGACY_PRECISION_ENV_VAR])

    def test_unknown_precision_is_rejected(self) -> None:
        with (
            patch.dict(os.environ, {loader.PRECISION_ENV_VAR: "half"}, clear=True),
            self.assertRaisesRegex(ImportError, "Unknown SuperDex precision: 'half'"),
        ):
            loader._get_use_double_precision()

    def test_module_is_present_uses_module_spec(self) -> None:
        spec = importlib.machinery.ModuleSpec(
            "runtime_module", loader=None, origin="runtime_module.pyd"
        )
        with patch.object(loader.importlib.util, "find_spec", return_value=spec):
            self.assertTrue(
                loader.module_is_present("runtime_module", payload=_PHYSICS_PAYLOAD)
            )

    def test_import_module_reports_missing_runtime_module(self) -> None:
        error = ModuleNotFoundError(name="runtime_module")
        with (
            patch.object(loader, "USE_DOUBLE_PRECISION", False),
            patch.object(loader.importlib, "import_module", side_effect=error),
            self.assertRaisesRegex(
                loader.NativeModuleNotFoundError,
                "Could not import native module 'runtime_module'",
            ) as context,
        ):
            loader.import_module("runtime_module", payload=_PHYSICS_PAYLOAD)

        self.assertIs(context.exception.__cause__, error)

    def test_import_module_names_the_double_extra_when_fp64_is_not_installed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "superdex" / "physics"
            (package_dir / "_native").mkdir(parents=True)
            error = ModuleNotFoundError(name="mochi_physics_double")

            with (
                patch.object(loader, "__file__", str(package_dir / "loader.py")),
                patch.object(loader, "PRECISION_NAME", "double"),
                patch.object(loader, "USE_DOUBLE_PRECISION", True),
                patch.object(
                    loader,
                    "site_packages_root",
                    return_value=Path(temp_dir) / "site-packages",
                ),
                patch.object(loader.importlib.util, "find_spec", _find_spec_for({})),
                patch.object(loader.importlib, "import_module", side_effect=error),
                self.assertRaisesRegex(
                    loader.NativeModuleNotFoundError,
                    r"pip install 'superdex-physics\[double\]'",
                ) as context,
            ):
                loader.import_module("mochi_physics", payload=_PHYSICS_PAYLOAD)

            self.assertIn("installed: single", str(context.exception))
            self.assertIs(context.exception.__cause__, error)

    def test_import_module_names_the_robotics_double_extra(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "superdex" / "physics"
            package_dir.mkdir(parents=True)
            (Path(temp_dir) / "superdex" / "robotics" / "_native").mkdir(parents=True)
            error = ModuleNotFoundError(name="superdex_robotics_double")

            with (
                patch.object(loader, "__file__", str(package_dir / "loader.py")),
                patch.object(loader, "PRECISION_NAME", "double"),
                patch.object(loader, "USE_DOUBLE_PRECISION", True),
                patch.object(
                    loader,
                    "site_packages_root",
                    return_value=Path(temp_dir) / "site-packages",
                ),
                patch.object(loader.importlib.util, "find_spec", _find_spec_for({})),
                patch.object(loader.importlib, "import_module", side_effect=error),
                self.assertRaisesRegex(
                    loader.NativeModuleNotFoundError,
                    r"pip install 'superdex-robotics\[double\]'",
                ),
            ):
                loader.import_module("superdex_robotics", payload=_ROBOTICS_PAYLOAD)

    def test_import_module_omits_install_guidance_outside_a_packaged_layout(
        self,
    ) -> None:
        # A source checkout packages neither precision, so naming a pip extra would be
        # misleading advice.
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "superdex" / "physics"
            package_dir.mkdir(parents=True)
            error = ModuleNotFoundError(name="mochi_physics_double")

            with (
                patch.object(loader, "__file__", str(package_dir / "loader.py")),
                patch.object(loader, "PRECISION_NAME", "double"),
                patch.object(loader, "USE_DOUBLE_PRECISION", True),
                patch.object(loader.importlib.util, "find_spec", _find_spec_for({})),
                patch.object(loader.importlib, "import_module", side_effect=error),
                self.assertRaises(loader.NativeModuleNotFoundError) as context,
            ):
                loader.import_module("mochi_physics", payload=_PHYSICS_PAYLOAD)

            self.assertNotIn("pip install", str(context.exception))

    def test_import_module_ignores_disabled_source_build_helper(self) -> None:
        runtime_module = ModuleType("runtime_module")
        runtime_module.__file__ = "runtime_module.pyd"

        with (
            patch.object(loader, "USE_DOUBLE_PRECISION", False),
            patch.object(loader, "_try_get_source_build_tools", return_value=None),
            patch.object(
                loader.importlib, "import_module", return_value=runtime_module
            ),
        ):
            self.assertIs(
                runtime_module,
                loader.import_module(
                    "runtime_module",
                    payload=_PHYSICS_PAYLOAD,
                    allow_source_build=True,
                ),
            )

    def test_forward_module_copies_public_symbols(self) -> None:
        source = ModuleType("native_module", "Native module documentation.")
        source.public_value = 42
        source._private_value = 7

        namespace: dict[str, object] = {"__doc__": "Facade documentation."}
        public_names = loader.forward_module(source, namespace)

        self.assertEqual(["public_value"], public_names)
        self.assertEqual(42, namespace["public_value"])
        self.assertEqual(7, namespace["_private_value"])
        self.assertEqual(
            "Facade documentation.\n\nNative module documentation.",
            namespace["__doc__"],
        )
