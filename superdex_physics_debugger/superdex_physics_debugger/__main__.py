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

"""Console-script entry point for the SuperDex physics debugger.

Running the ``mochi_debugger`` binary in ``_native/`` directly works just as well; this
shim only gives the command a name of its own.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

EXECUTABLE_NAME = "mochi_debugger.exe" if sys.platform == "win32" else "mochi_debugger"

# `__package__` is None when this file is run as a plain script rather than imported.
_PACKAGE = __package__ or "superdex_physics_debugger"


def _native_roots() -> list[Path]:
    """Where the native payload may live, most authoritative first.

    A wheel install puts it next to this module; an editable install leaves this module in
    the source tree and the CMake-installed payload under site-packages.
    """

    return [
        Path(__file__).resolve().parent / "_native",
        Path(sysconfig.get_paths()["platlib"]) / _PACKAGE / "_native",
    ]


def find_executable() -> Path | None:
    for root in _native_roots():
        candidate = root / EXECUTABLE_NAME
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    executable = find_executable()
    if executable is None:
        searched = "\n  ".join(str(root) for root in _native_roots())
        raise SystemExit(
            f"superdex-physics-debugger: no {EXECUTABLE_NAME} in this install.\n"
            f"Looked in:\n  {searched}\n"
            "Reinstall the distribution, or run the executable directly if you built it "
            "from source."
        )

    arguments = [str(executable), *sys.argv[1:]]
    if sys.platform == "win32":
        # A gui-script runs under pythonw.exe, so waiting would keep that interpreter
        # resident for the window's lifetime.
        subprocess.Popen(
            arguments,
            creationflags=subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        return 0

    # Replace this process, so the debugger owns the terminal's signals and exit status.
    os.execv(str(executable), arguments)


if __name__ == "__main__":
    sys.exit(main())
