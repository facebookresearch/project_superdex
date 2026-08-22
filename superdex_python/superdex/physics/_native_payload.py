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

# Kept out of `__init__.py` so importing the mesh facade does not re-enter
# `superdex/physics/__init__.py` and load the physics extension as a side effect.

from __future__ import annotations

from superdex.physics.loader import NativePayload

PHYSICS_NATIVE_PAYLOAD = NativePayload(
    subpackage="physics",
    distribution="superdex-physics",
    fp64_package="superdex_physics_fp64",
)
