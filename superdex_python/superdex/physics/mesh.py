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

"""SuperDex Physics mesh-processing operations.

Provides surface remeshing, mesh statistics, and SDF isosurface reconstruction.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

from superdex.physics._native_payload import PHYSICS_NATIVE_PAYLOAD
from superdex.physics.loader import forward_module, import_module, native_roots

########################################################################################

# The GPL-licensed geometry helper ships as its own distribution so nothing here links
# CGAL. It is optional: operations that marshal to it error cleanly when it is absent.
_MESH_CLI_PACKAGE = "superdex_mesh_cli"


def _mesh_cli_executable_name() -> str:
    # Kept in step with `kCliExecutableName` in
    # superdex_physics/libraries/mochi/mochi_mesh/src/mesh_cli_client.cpp.
    return (
        "superdex_mesh_cli.exe"
        if platform.system() == "Windows"
        else "superdex_mesh_cli"
    )


def _configure_packaged_mesh_cli_path() -> None:
    # Never override an explicit choice; pointing at a local build is why the var exists.
    if os.environ.get("SUPERDEX_MESH_CLI_PATH"):
        return
    name = _mesh_cli_executable_name()
    for root in native_roots(_MESH_CLI_PACKAGE):
        helper_path = root / name
        if helper_path.is_file():
            os.environ["SUPERDEX_MESH_CLI_PATH"] = str(helper_path.resolve())
            return


_configure_packaged_mesh_cli_path()

# Load the bindings for the selected precision. Forward all public symbols.
_extension = import_module(
    "mochi_mesh",
    payload=PHYSICS_NATIVE_PAYLOAD,
    allow_source_build=True,
)
__all__ = forward_module(_extension, globals())
