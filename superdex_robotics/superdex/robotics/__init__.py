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

from typing import TYPE_CHECKING

# Importing from `superdex.physics` initializes the shared pybind core and the physics
# types before the robotics extension that depends on them is imported.
from superdex.physics.loader import (
    forward_module,
    import_module,
    module_is_present,
    NativePayload,
)

########################################################################################

# Declared here rather than in the loader: this is the only facade that imports the
# robotics payload.
_ROBOTICS_NATIVE_PAYLOAD = NativePayload(
    subpackage="robotics",
    distribution="superdex-robotics",
    fp64_package="superdex_robotics_fp64",
)

_extension = import_module(
    "superdex_robotics",
    payload=_ROBOTICS_NATIVE_PAYLOAD,
    allow_source_build=True,
)
# The extension registers its public API on the `bots` submodule; forwarding the
# top-level module instead would export a silently incomplete API.
_public_module = getattr(_extension, "bots", None)
if _public_module is None:
    raise ImportError(
        f"{_extension.__name__!r} does not expose the expected 'bots' submodule."
    )

__all__ = forward_module(_public_module, globals())

# Register torch-dependent sensor types if available. Its docstring is appended to the
# robotics docstring, and it may not silently shadow any symbol already forwarded.
if module_is_present(
    "superdex_robotics_torch_sensors",
    payload=_ROBOTICS_NATIVE_PAYLOAD,
):
    _torch_extension = import_module(
        "superdex_robotics_torch_sensors",
        payload=_ROBOTICS_NATIVE_PAYLOAD,
    )
    __all__ += forward_module(_torch_extension, globals(), reserved_names=__all__)

# Generated canonical stubs describe the shared public API for static type checking.
if TYPE_CHECKING:
    from superdex_robotics.bots import *  # noqa F401,F403
