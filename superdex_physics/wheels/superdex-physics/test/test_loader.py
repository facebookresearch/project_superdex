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

import gc
import importlib.machinery
import json
import os
import sysconfig
import tempfile
import unittest
import weakref
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


class IsaFallbackTest(unittest.TestCase):
    def test_default_uses_only_matching_precision_runtimes(self) -> None:
        cases = (
            ((), "avx2"),
            (("avx2",), "avx2"),
            (("avx512",), "avx512"),
            (("avx2", "avx512"), "avx2"),
            (("stub_avx2", "avx512"), "avx512"),
            (("stub_avx512",), "avx2"),
        )
        for double in (False, True):
            for layout in ("importable", "packaged", "bundle"):
                for available, expected in cases:
                    with self.subTest(
                        double=double, layout=layout, available=available
                    ):
                        self._check_default(double, layout, available, expected)

    def _check_default(
        self, double: bool, layout: str, available: tuple[str, ...], expected: str
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = root / "superdex" / "physics"
            site = root / "site-packages"
            with (
                patch.object(loader, "__file__", str(package / "loader.py")),
                patch.object(loader, "site_packages_root", return_value=site),
                patch.object(loader, "USE_DOUBLE_PRECISION", double),
                patch.object(loader.platform, "machine", return_value="x86_64"),
                patch.object(loader.sys, "path", []),
                patch.dict(os.environ, {}, clear=True),
                patch.dict(
                    loader.sys.modules,
                    {"mochi.loader": None, "superdex.physics.loader": None},
                ),
            ):
                for suffix in ("", "_double", "_avx512", "_double_avx512"):
                    loader.sys.modules.pop("mochi_physics" + suffix, None)
                if layout == "importable":
                    directory = root / "extensions"
                    loader.sys.path.append(str(directory))
                elif layout == "packaged":
                    directory = (
                        site / "superdex_physics_fp64" / "_native"
                        if double
                        else package / "_native"
                    )
                else:
                    directory = root / "mochi_native"
                # Other-precision runtimes must not influence the selection.
                variants = [(isa, not double) for isa in ("avx2", "avx512")]
                variants += [(isa, double) for isa in available]
                for variant, precision in variants:
                    isa = variant.removeprefix("stub_")
                    destination = directory / isa if layout == "bundle" else directory
                    destination.mkdir(parents=True, exist_ok=True)
                    name = "mochi_physics" + ("_double" if precision else "")
                    name += "_avx512" if isa == "avx512" else ""
                    if variant.startswith("stub_"):
                        (destination / name).mkdir()
                    else:
                        (destination / (name + ".py")).write_text(
                            "raise AssertionError('ISA discovery must not import runtimes')\n"
                        )
                self.assertEqual(loader._resolve_isa(), expected)
                self.assertEqual(dict(os.environ), {})

    def test_explicit_isa_does_not_fall_back_to_available_runtime(self) -> None:
        for variable in ("SUPERDEX_ISA", "MOCHI_ISA"):
            with (
                self.subTest(variable=variable),
                patch.object(loader.platform, "machine", return_value="x86_64"),
                patch.dict(os.environ, {variable: "avx2"}, clear=True),
                patch.dict(
                    loader.sys.modules,
                    {"mochi.loader": None, "superdex.physics.loader": None},
                ),
                patch.object(loader.importlib.util, "find_spec") as find_spec,
            ):
                self.assertEqual(loader._resolve_isa(), "avx2")
                find_spec.assert_not_called()


class LoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The import_module tests patch importlib.import_module, which sysconfig's lazy
        # first load goes through; a failed first load breaks sysconfig for the process.
        sysconfig.get_config_vars()

    def setUp(self):
        isa_patch = patch.object(loader, "ISA", "avx2")
        isa_patch.start()
        self.addCleanup(isa_patch.stop)

    def test_namespaced_bundle_import_uses_the_selected_image(self) -> None:
        name = "superdex_renderer.mochi._mochi_paths"
        images = {
            "avx2": "avx2/libsuperdex_native.so",
            "avx512": "avx512/libsuperdex_native_avx512.so",
        }
        precision = "double" if loader.USE_DOUBLE_PRECISION else "single"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = root / "mochi_native"
            bundle.mkdir()
            manifest = bundle / "manifest.json"
            manifest.write_text(json.dumps({"precision": precision, "images": images}))
            file_patch = patch.object(
                loader, "__file__", str(root / "superdex" / "physics" / "loader.py")
            )
            for isa, image in images.items():
                with (
                    self.subTest(isa=isa),
                    file_patch,
                    patch.object(loader, "ISA", isa),
                    patch.object(
                        importlib.machinery.ExtensionFileLoader,
                        "create_module",
                        return_value=ModuleType(name),
                    ),
                    patch.object(
                        importlib.machinery.ExtensionFileLoader, "exec_module"
                    ) as execute_module,
                ):
                    module = loader.import_bundled_module(name)
                    self.assertEqual(name, module.__spec__.name)
                    self.assertEqual(str(bundle / image), module.__spec__.origin)
                    execute_module.assert_called_once_with(module)
            other = "single" if precision == "double" else "double"
            manifest.write_text(json.dumps({"precision": other, "images": images}))
            with file_patch, self.assertRaisesRegex(ImportError, "precision"):
                loader.import_bundled_module(name)

    def test_packaged_native_dir_uses_the_flat_base_payload_for_fp32(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Resolve the base: loader resolves __file__, which canonicalizes
            # symlinks such as /var -> /private/var on macOS.
            base = Path(temp_dir).resolve()
            package_dir = base / "superdex" / "physics"
            native_dir = package_dir / "_native"
            native_dir.mkdir(parents=True)

            with (
                patch.object(loader, "__file__", str(package_dir / "loader.py")),
                patch.object(loader, "USE_DOUBLE_PRECISION", False),
                patch.object(loader, "PRECISION_NAME", "fp32"),
            ):
                self.assertEqual(
                    native_dir,
                    loader._packaged_native_dir(_PHYSICS_PAYLOAD),
                )

    def test_packaged_native_dir_uses_the_flat_robotics_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Resolve the base: loader resolves __file__, which canonicalizes
            # symlinks such as /var -> /private/var on macOS.
            base = Path(temp_dir).resolve()
            package_dir = base / "superdex" / "physics"
            native_dir = base / "superdex" / "robotics" / "_native"
            native_dir.mkdir(parents=True)

            with (
                patch.object(loader, "__file__", str(package_dir / "loader.py")),
                patch.object(loader, "USE_DOUBLE_PRECISION", False),
                patch.object(loader, "PRECISION_NAME", "fp32"),
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
                patch.object(loader, "PRECISION_NAME", "fp64"),
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
                patch.object(loader, "PRECISION_NAME", "fp64"),
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
                patch.object(loader, "PRECISION_NAME", "fp64"),
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
                patch.object(loader, "PRECISION_NAME", "fp32"),
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
                patch.object(loader, "PRECISION_NAME", "fp64"),
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
            # Resolve the base: loader resolves __file__, which canonicalizes
            # symlinks such as /var -> /private/var on macOS.
            base = Path(temp_dir).resolve()
            package_dir = base / "superdex" / "physics"
            native_dir = package_dir / "_native"
            native_dir.mkdir(parents=True)
            site_packages = base / "site-packages"
            (site_packages / "superdex" / "physics" / "_native").mkdir(parents=True)

            with (
                patch.object(loader, "__file__", str(package_dir / "loader.py")),
                patch.object(loader, "USE_DOUBLE_PRECISION", False),
                patch.object(loader, "PRECISION_NAME", "fp32"),
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
                loader.PRECISION_ENV_VAR: "fp32",
                loader.LEGACY_PRECISION_ENV_VAR: "double",
            },
            clear=True,
        ):
            self.assertFalse(loader._get_use_double_precision())
            self.assertEqual("fp32", os.environ[loader.PRECISION_ENV_VAR])
            self.assertEqual("fp32", os.environ[loader.LEGACY_PRECISION_ENV_VAR])

    def test_precision_env_accepts_canonical_and_legacy_values(self) -> None:
        for value, expected, canonical in (
            ("fp32", False, "fp32"),
            ("fp64", True, "fp64"),
            ("single", False, "fp32"),
            ("float", False, "fp32"),
            ("double", True, "fp64"),
        ):
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ,
                    {loader.PRECISION_ENV_VAR: value},
                    clear=True,
                ),
            ):
                self.assertEqual(expected, loader._get_use_double_precision())
                self.assertEqual(canonical, os.environ[loader.PRECISION_ENV_VAR])

    def test_legacy_precision_env_is_accepted(self) -> None:
        with patch.dict(
            os.environ,
            {loader.LEGACY_PRECISION_ENV_VAR: "float64"},
            clear=True,
        ):
            self.assertTrue(loader._get_use_double_precision())
            self.assertEqual("fp64", os.environ[loader.PRECISION_ENV_VAR])
            self.assertEqual("fp64", os.environ[loader.LEGACY_PRECISION_ENV_VAR])

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
            self.assertTrue(loader._runtime_is_present("runtime_module"))

    def test_import_module_reports_missing_runtime_module(self) -> None:
        error = ModuleNotFoundError(name="runtime_module")
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(loader, "USE_DOUBLE_PRECISION", False),
            patch.object(
                loader,
                "site_packages_root",
                return_value=Path(temp_dir) / "site-packages",
            ),
            patch.object(loader.importlib.util, "find_spec", _find_spec_for({})),
            patch.object(loader.importlib, "import_module", side_effect=error),
            self.assertRaisesRegex(
                loader.NativeModuleNotFoundError,
                "Could not import native module 'runtime_module'",
            ) as context,
        ):
            loader.import_module("runtime_module", payload=_PHYSICS_PAYLOAD)

        self.assertIs(context.exception.__cause__, error)

    def test_import_module_names_the_fp64_extra_when_fp64_is_not_installed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "superdex" / "physics"
            (package_dir / "_native").mkdir(parents=True)
            error = ModuleNotFoundError(name="mochi_physics_double")

            with (
                patch.object(loader, "__file__", str(package_dir / "loader.py")),
                patch.object(loader, "PRECISION_NAME", "fp64"),
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
                    r"pip install 'superdex-physics\[fp64\]'",
                ) as context,
            ):
                loader.import_module("mochi_physics", payload=_PHYSICS_PAYLOAD)

            self.assertIn("installed: fp32", str(context.exception))
            self.assertIs(context.exception.__cause__, error)

    def test_avx512_omits_baseline_wheel_install_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = root / "superdex" / "physics"
            (package / "_native").mkdir(parents=True)
            with (
                patch.object(loader, "__file__", str(package / "loader.py")),
                patch.object(
                    loader, "site_packages_root", return_value=root / "absent"
                ),
                patch.object(loader, "USE_DOUBLE_PRECISION", True),
                patch.object(loader, "PRECISION_NAME", "fp64"),
                patch.object(loader.importlib.util, "find_spec", _find_spec_for({})),
            ):
                for isa in ("avx2", "avx512"):
                    with self.subTest(isa=isa), patch.object(loader, "ISA", isa):
                        guidance = loader._missing_payload_guidance(_PHYSICS_PAYLOAD)
                        if isa == "avx2":
                            self.assertIn(
                                "pip install 'superdex-physics[fp64]'", guidance
                            )
                        else:
                            self.assertEqual(guidance, "")

    def test_import_module_names_the_robotics_fp64_extra(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "superdex" / "physics"
            package_dir.mkdir(parents=True)
            (Path(temp_dir) / "superdex" / "robotics" / "_native").mkdir(parents=True)
            error = ModuleNotFoundError(name="superdex_robotics_double")

            with (
                patch.object(loader, "__file__", str(package_dir / "loader.py")),
                patch.object(loader, "PRECISION_NAME", "fp64"),
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
                    r"pip install 'superdex-robotics\[fp64\]'",
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
                patch.object(loader, "PRECISION_NAME", "fp64"),
                patch.object(loader, "USE_DOUBLE_PRECISION", True),
                patch.object(
                    loader,
                    "site_packages_root",
                    return_value=Path(temp_dir) / "site-packages",
                ),
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
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(loader, "USE_DOUBLE_PRECISION", False),
            patch.object(
                loader,
                "site_packages_root",
                return_value=Path(temp_dir) / "site-packages",
            ),
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

    def test_resolve_isa(self) -> None:
        cases = [
            ("x86_64", {}, "avx2", None),
            ("x86_64", {"SUPERDEX_ISA": ""}, "avx2", None),
            ("x86_64", {"SUPERDEX_ISA": " AVX2 "}, "avx2", None),
            ("x86_64", {"SUPERDEX_ISA": " AVX512 "}, "avx512", None),
            ("AMD64", {}, "avx2", None),
            ("AMD64", {"SUPERDEX_ISA": ""}, "avx2", None),
            ("arm64", {}, "neon", None),
            ("AARCH64", {}, "neon", None),
            ("AARCH64", {"SUPERDEX_ISA": ""}, "neon", None),
            ("arm64", {"SUPERDEX_ISA": "avx512"}, None, "target architecture"),
            ("arm64", {"SUPERDEX_ISA": "avx2"}, None, "target architecture"),
            ("x86_64", {"SUPERDEX_ISA": "neon"}, None, "target architecture"),
            ("x86_64", {"SUPERDEX_ISA": "other"}, None, "Unknown SUPERDEX_ISA"),
            (
                "x86_64",
                {"SUPERDEX_ISA": " AVX512 ", "MOCHI_ISA": "avx2"},
                "avx512",
                None,
            ),
            ("x86_64", {"SUPERDEX_ISA": "", "MOCHI_ISA": "avx512"}, "avx2", None),
            ("x86_64", {"MOCHI_ISA": "avx512"}, "avx512", None),
            ("x86_64", {"SUPERDEX_ISA": "avx512"}, "avx512", None),
        ]
        cases.extend(
            (
                machine,
                {} if value is None else {"SUPERDEX_ISA": value},
                None,
                "target architecture",
            )
            for machine in ("riscv64", "i686", "armv7l")
            for value in (None, "", "avx2", "avx512", "neon")
        )
        for machine, env, expected, error in cases:
            with (
                self.subTest(machine=machine, env=env),
                patch.object(loader.platform, "machine", return_value=machine),
                patch.object(loader, "_physics_runtime_is_present", return_value=False),
                patch.dict(os.environ, env, clear=True),
                patch.dict(
                    loader.sys.modules,
                    {"mochi.loader": None, "superdex.physics.loader": None},
                ),
            ):
                if error is None:
                    self.assertEqual(loader._resolve_isa(), expected)
                else:
                    with self.assertRaisesRegex(ImportError, error):
                        loader._resolve_isa()
                self.assertEqual(dict(os.environ), env)

    def test_resolve_isa_without_machine_name(self) -> None:
        cases = [
            ("win32", "win-amd64", "avx2"),
            ("win32", "win-arm64", "neon"),
            ("win32", "win32", None),
            ("linux", "linux-x86_64", None),
        ]
        for sys_platform, interpreter_platform, expected in cases:
            with (
                self.subTest(platform=sys_platform, interpreter=interpreter_platform),
                patch.object(loader.platform, "machine", return_value=""),
                patch.object(loader.sys, "platform", sys_platform),
                patch.object(
                    loader.sysconfig, "get_platform", return_value=interpreter_platform
                ),
                patch.object(loader, "_physics_runtime_is_present", return_value=False),
                patch.dict(os.environ, {}, clear=True),
                patch.dict(
                    loader.sys.modules,
                    {"mochi.loader": None, "superdex.physics.loader": None},
                ),
            ):
                if expected is None:
                    with self.assertRaisesRegex(ImportError, "target architecture"):
                        loader._resolve_isa()
                else:
                    self.assertEqual(loader._resolve_isa(), expected)

    def test_precision_before_isa_suffix(self) -> None:
        for isa in ("avx2", "avx512", "neon"):
            for double in (False, True):
                with (
                    self.subTest(isa=isa, double=double),
                    patch.object(loader, "ISA", isa),
                    patch.object(loader, "USE_DOUBLE_PRECISION", double),
                ):
                    expected = (
                        "native"
                        + ("_double" if double else "")
                        + ("_avx512" if isa == "avx512" else "")
                    )
                    self.assertEqual(
                        loader.selected_native_module_name("native"), expected
                    )

    def test_cached_static_native_module_remains_present(self) -> None:
        module = ModuleType("static_native")
        with patch.dict(loader.sys.modules, {"static_native": module}):
            self.assertTrue(loader._runtime_is_present("static_native"))
            module.__path__ = []
            self.assertFalse(loader._runtime_is_present("static_native"))

    def test_mismatch_is_symmetric(self) -> None:
        for double in (False, True):
            for isa in ("avx2", "avx512"):
                selected = (
                    "native"
                    + ("_double" if double else "")
                    + ("_avx512" if isa == "avx512" else "")
                )
                other = (
                    "native"
                    + ("_double" if not double else "")
                    + ("_avx512" if isa != "avx512" else "")
                )
                spec = importlib.machinery.ModuleSpec(
                    other, loader=None, origin=other + ".so"
                )
                with (
                    self.subTest(selected=selected),
                    patch.object(loader, "USE_DOUBLE_PRECISION", double),
                    patch.object(loader, "ISA", isa),
                    patch.object(
                        loader.importlib,
                        "import_module",
                        side_effect=ModuleNotFoundError(name=selected),
                    ),
                    patch.object(
                        loader.importlib.util,
                        "find_spec",
                        side_effect=lambda name, other=other, spec=spec: spec
                        if name == other
                        else None,
                    ),
                    self.assertRaises(loader.NativeVariantUnavailableError) as caught,
                ):
                    loader.import_module("native")
                message = str(caught.exception)
                precision = "fp64" if double else "fp32"
                self.assertIn(f"SUPERDEX_PRECISION={precision}", message)
                self.assertIn(f"SUPERDEX_ISA={isa}", message)
                self.assertIn(f"Installed variants: {other}", message)
                self.assertIn(
                    "Select an installed variant using SUPERDEX_PRECISION and SUPERDEX_ISA",
                    message,
                )
                self.assertNotIn("Current wheels", message)

    def test_initializer_missing_its_own_name_is_propagated(self) -> None:
        error = ModuleNotFoundError(name="runtime_module")
        spec = importlib.machinery.ModuleSpec(
            "runtime_module", loader=None, origin="runtime_module.so"
        )
        with (
            patch.object(loader, "USE_DOUBLE_PRECISION", False),
            patch.object(loader.importlib, "import_module", side_effect=error),
            patch.object(loader.importlib.util, "find_spec", return_value=spec),
            self.assertRaises(ModuleNotFoundError) as caught,
        ):
            loader.import_module("runtime_module")
        self.assertIs(caught.exception, error)

    def test_lazy_import_preserves_variant_mismatch(self) -> None:
        error = loader.NativeVariantUnavailableError("wrong ISA/precision")
        resolve = loader.lazy_import_resolver(
            lazy_imports={"mesh": "superdex.physics.mesh"},
            namespace={},
            module_name="superdex.physics",
        )
        with (
            patch.object(loader.importlib, "import_module", side_effect=error),
            self.assertRaises(loader.NativeVariantUnavailableError) as caught,
        ):
            resolve("mesh")
        self.assertIs(caught.exception, error)

    def test_optional_absence_and_initialization_failure(self) -> None:
        with (
            patch.object(
                loader.importlib,
                "import_module",
                side_effect=ModuleNotFoundError(name="native"),
            ),
            patch.object(loader.importlib.util, "find_spec", return_value=None),
            patch.object(loader, "USE_DOUBLE_PRECISION", False),
        ):
            self.assertIsNone(loader.import_optional_module("native"))
        error = ImportError("undefined symbol")
        with (
            patch.object(loader.importlib, "import_module", side_effect=error),
            self.assertRaises(ImportError) as caught,
        ):
            loader.import_optional_module("native")
        self.assertIs(caught.exception, error)

    def test_optional_absence_releases_importing_frames(self) -> None:
        class Local:
            pass

        def import_with_local(local: Local) -> ModuleType | None:
            return loader.import_optional_module("superdex_loader_test_absent")

        local = Local()
        local_ref = weakref.ref(local)
        # Without the cyclic collector, only plain reference counting can free `local`.
        gc.disable()
        try:
            self.assertIsNone(import_with_local(local))
            del local
            self.assertIsNone(local_ref())
        finally:
            gc.enable()

    def test_optional_mismatch_finds_unselected_payload_outside_sys_path(self) -> None:
        for variant in ("native_double", "native_double_avx512"):
            with (
                self.subTest(variant=variant),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                root = Path(temp_dir)
                facade = root / "superdex" / "physics"
                facade.mkdir(parents=True)
                payload = root / "superdex_physics_fp64"
                native = payload / "_native"
                native.mkdir(parents=True)
                (native / (variant + ".py")).write_text(
                    "raise AssertionError('Optional discovery must not import another runtime')\n"
                )
                search_path = list(loader.sys.path)
                error = ModuleNotFoundError(name="native")
                with (
                    patch.object(loader, "USE_DOUBLE_PRECISION", False),
                    patch.object(loader, "__file__", str(facade / "loader.py")),
                    patch.object(
                        loader, "site_packages_root", return_value=root / "absent"
                    ),
                    patch.object(loader.sys, "path", search_path),
                    patch.object(
                        loader.importlib.util,
                        "find_spec",
                        _find_spec_for({"superdex_physics_fp64": payload}),
                    ),
                    patch.object(
                        loader.importlib, "import_module", side_effect=error
                    ) as imports,
                    self.assertRaises(loader.NativeVariantUnavailableError) as caught,
                ):
                    self.assertNotIn(str(native), loader.sys.path)
                    loader.import_optional_module("native", payload=_PHYSICS_PAYLOAD)
                self.assertIn(variant, str(caught.exception))
                self.assertIs(caught.exception.__cause__, error)
                self.assertNotIn(str(native), search_path)
                imports.assert_called_once_with("native")

    def test_optional_mismatch_raises(self) -> None:
        spec = importlib.machinery.ModuleSpec(
            "native_avx512", loader=None, origin="native_avx512.so"
        )
        with (
            patch.object(loader, "USE_DOUBLE_PRECISION", False),
            patch.object(
                loader.importlib,
                "import_module",
                side_effect=ModuleNotFoundError(name="native"),
            ),
            patch.object(
                loader.importlib.util,
                "find_spec",
                side_effect=lambda name: spec if name == "native_avx512" else None,
            ),
            self.assertRaises(loader.NativeVariantUnavailableError),
        ):
            loader.import_optional_module("native")

    def test_bundle_preparation_adds_only_selected_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for isa in ("avx2", "avx512"):
                (root / "mochi_native" / isa).mkdir(parents=True)
            for isa in ("avx2", "avx512"):
                with (
                    patch.object(
                        loader,
                        "__file__",
                        str(root / "superdex" / "physics" / "loader.py"),
                    ),
                    patch.object(loader, "ISA", isa),
                    patch.object(loader.sys, "path", []),
                ):
                    loader._prepare_native_bundle()
                    self.assertEqual(
                        [str(root / "mochi_native" / isa)], loader.sys.path
                    )
