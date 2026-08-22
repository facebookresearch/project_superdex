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

"""Console-script entry point for SuperDex Studio.

Running the ``superdex_studio`` binary in ``_native/`` directly works just as well; the
application resolves everything relative to its own executable, not the working directory.

This shim only reports a clearer failure when that binary is missing, and sets
``SUPERDEX_MESH_CLI_PATH`` when it can find the optional mesh helper through the import system.
It stays quiet when it cannot: the application runs its own search, which succeeds in layouts
this one cannot see, and warns if both come up empty.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

EXECUTABLE_NAME = (
    "superdex_studio.exe" if sys.platform == "win32" else "superdex_studio"
)

# Kept in step with `kCliExecutableName` in the SuperDex Physics mesh client.
MESH_CLI_PACKAGE = "superdex_mesh_cli"

# `__package__` is None when this file is run as a plain script rather than imported.
_PACKAGE = __package__ or "superdex_studio"
MESH_CLI_EXECUTABLE_NAME = (
    "superdex_mesh_cli.exe" if sys.platform == "win32" else "superdex_mesh_cli"
)


def _platlib(package: str) -> Path:
    return Path(sysconfig.get_paths()["platlib"]) / package / "_native"


def _native_roots() -> list[Path]:
    """Where this distribution's payload may live, most authoritative first.

    A wheel install puts it next to this module; an editable install leaves this module in
    the source tree and the CMake-installed payload under site-packages.
    """

    return [
        Path(__file__).resolve().parent / "_native",
        _platlib(_PACKAGE),
    ]


def _mesh_cli_roots() -> list[Path]:
    """Where the separately-distributed mesh helper may live.

    Located through the import system rather than assumed adjacent, since pip is free to
    place another distribution elsewhere.
    """

    roots: list[Path] = []
    try:
        spec = importlib.util.find_spec(MESH_CLI_PACKAGE)
    except (ImportError, ValueError):
        # ValueError: the name is in sys.modules with a cleared __spec__.
        spec = None
    if spec is not None and spec.submodule_search_locations:
        roots.append(Path(next(iter(spec.submodule_search_locations))) / "_native")
    roots.append(_platlib(MESH_CLI_PACKAGE))
    return roots


def _first_file(roots: list[Path], name: str) -> Path | None:
    for root in roots:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def find_executable() -> Path | None:
    return _first_file(_native_roots(), EXECUTABLE_NAME)


def find_mesh_cli() -> Path | None:
    return _first_file(_mesh_cli_roots(), MESH_CLI_EXECUTABLE_NAME)


def _child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    # Never override an explicit choice: pointing at a locally built helper is the reason
    # the variable exists.
    if environment.get("SUPERDEX_MESH_CLI_PATH"):
        return environment
    mesh_cli = find_mesh_cli()
    if mesh_cli is not None:
        environment["SUPERDEX_MESH_CLI_PATH"] = str(mesh_cli)
    return environment


def main() -> int:
    executable = find_executable()
    if executable is None:
        searched = "\n  ".join(str(root) for root in _native_roots())
        raise SystemExit(
            f"superdex-studio: no {EXECUTABLE_NAME} in this install.\n"
            f"Looked in:\n  {searched}\n"
            "Reinstall the distribution, or run the executable directly if you built it "
            "from source."
        )

    environment = _child_environment()
    arguments = [str(executable), *sys.argv[1:]]
    if sys.platform == "win32":
        # A gui-script runs under pythonw.exe, so waiting would keep that interpreter
        # resident for the window's lifetime.
        subprocess.Popen(
            arguments,
            env=environment,
            creationflags=subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        return 0

    # Replace this process, so the application owns the terminal's signals and exit status.
    os.execve(str(executable), arguments, environment)


if __name__ == "__main__":
    sys.exit(main())
