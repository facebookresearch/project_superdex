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

from types import ModuleType
from typing import TYPE_CHECKING

from superdex.physics._native_payload import PHYSICS_NATIVE_PAYLOAD
from superdex.physics.loader import (
    forward_module,
    import_module,
    lazy_import_resolver,
    PRECISION_NAME,
    USE_DOUBLE_PRECISION,
)

########################################################################################


def _validate_physics_precision(module: ModuleType) -> None:
    """Reject Physics bindings that do not match the process precision selection."""
    reported_precision = module.uses_double_precision()
    if reported_precision != USE_DOUBLE_PRECISION:
        module_path = getattr(module, "__file__", None)
        raise ImportError(
            f"Precision mismatch: selected precision={PRECISION_NAME!r}, "
            f"module={module.__name__!r}, module_path={module_path!r}, "
            f"reported uses_double_precision()={reported_precision!r}."
        )


########################################################################################

# Load the bindings for the selected precision.
_extension = import_module(
    "mochi_physics",
    payload=PHYSICS_NATIVE_PAYLOAD,
    allow_source_build=True,
)
_validate_physics_precision(_extension)

# Forward all public symbols from the extension module.
__all__ = forward_module(_extension, globals())

_LAZY_IMPORTS = {
    "debugger": "superdex.physics.debugger",
    "mesh": "superdex.physics.mesh",
    "rerun": "superdex.physics.rerun",
    "utils": "superdex.physics.utils",
    "viewer": "superdex.physics.viewer",
}

__getattr__ = lazy_import_resolver(
    lazy_imports=_LAZY_IMPORTS,
    namespace=globals(),
    module_name=__name__,
)

__all__ += list(_LAZY_IMPORTS)

# Generated canonical stubs describe the shared public API for static type checking.
if TYPE_CHECKING:
    from mochi_physics import *  # noqa F401,F403
