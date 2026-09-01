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

import importlib
import os
import sys
import unittest
from types import ModuleType

# Set by the Buck test target, where the native extensions are always built. Outside Buck
# they may be absent, and the smoke test skips rather than passing vacuously.
_REQUIRE_NATIVE_ENV_VAR = "SUPERDEX_REQUIRE_NATIVE"


def _is_missing_native_module(error: ImportError) -> bool:
    return (
        error.__class__.__name__ == "NativeModuleNotFoundError"
        and error.__class__.__module__ == "superdex.physics.loader"
    )


def _drop_superdex_test_stand_ins() -> None:
    """Evict by-path stand-ins that sibling test modules registered under the real names.

    Those entries never executed `superdex/physics/__init__.py`, so importing through them
    would check a half-built package instead of the installed one.
    """
    for name in [
        name
        for name in sys.modules
        if name == "superdex" or name.startswith("superdex.")
    ]:
        del sys.modules[name]


def _import_superdex_physics_modules() -> tuple[
    ModuleType, ModuleType, ModuleType, ModuleType
]:
    _drop_superdex_test_stand_ins()
    try:
        physics = importlib.import_module("superdex.physics")
        debugger = importlib.import_module("superdex.physics.debugger")
        mesh = importlib.import_module("superdex.physics.mesh")
        viewer = importlib.import_module("superdex.physics.viewer")
    except ImportError as error:
        if _is_missing_native_module(error) and not os.environ.get(
            _REQUIRE_NATIVE_ENV_VAR
        ):
            raise unittest.SkipTest(
                f"SuperDex physics smoke test requires built native modules: {error}"
            ) from error
        raise

    return physics, debugger, mesh, viewer


class SuperdexPhysicsImportTest(unittest.TestCase):
    def test_import_smoke(self) -> None:
        physics, debugger, mesh, viewer = _import_superdex_physics_modules()

        self.assertTrue(hasattr(physics, "Actor"))
        self.assertTrue(hasattr(mesh, "remesh_surface"))
        self.assertTrue(hasattr(viewer, "Viewer"))
        self.assertTrue(hasattr(debugger, "attach"))

    def test_lazy_submodules_are_available_from_physics(self) -> None:
        physics, debugger, mesh, viewer = _import_superdex_physics_modules()

        self.assertIs(physics.debugger, debugger)
        self.assertIs(physics.mesh, mesh)
        self.assertIs(physics.viewer, viewer)
