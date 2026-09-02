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

"""Load `superdex.physics` modules by path, without the native extensions.

Importing `superdex.physics` normally runs its `__init__.py`, which loads the compiled
extension. Tests that only exercise pure-Python modules register stand-ins under the real
names instead; those shadow the installed package for the rest of the process, so a later
test must register its own or drop them first -- see `test_superdex_physics.py`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_MODULE_PATH = Path(__file__).resolve()
_PAR_ROOT = next(
    (parent for parent in _MODULE_PATH.parents if parent.suffix == ".par"), None
)
SUPERDEX_ROOT: Path = (
    _PAR_ROOT / "superdex"
    if _PAR_ROOT is not None
    else _MODULE_PATH.parents[1] / "superdex"
)
PHYSICS_ROOT: Path = SUPERDEX_ROOT / "physics"


def ensure_test_package(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__dict__["__path__"] = [str(path)]
        module.__dict__["__package__"] = name
        sys.modules[name] = module

        parent_name, _, child_name = name.rpartition(".")
        if parent_name:
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child_name, module)
    return module


def load_test_module(qualified_name: str, path: Path) -> ModuleType:
    try:
        source = path.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        loader = globals().get("__loader__")
        if loader is None or not hasattr(loader, "get_data"):
            raise
        archive_path = Path("superdex") / path.relative_to(SUPERDEX_ROOT)
        source = loader.get_data(archive_path.as_posix())

    spec = importlib.util.spec_from_loader(
        qualified_name, loader=None, origin=str(path)
    )
    if spec is None:
        raise RuntimeError(f"Could not create module spec for {qualified_name!r}")

    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[qualified_name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)

    parent_name, _, child_name = qualified_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, child_name, module)
    return module


def ensure_physics_package() -> None:
    """Register the `superdex` and `superdex.physics` package stand-ins."""
    ensure_test_package("superdex", SUPERDEX_ROOT)
    ensure_test_package("superdex.physics", PHYSICS_ROOT)


def load_physics_module(name: str) -> ModuleType:
    """Load one `superdex.physics` submodule by path, registering the packages first."""
    ensure_physics_package()
    return load_test_module(f"superdex.physics.{name}", PHYSICS_ROOT / f"{name}.py")
