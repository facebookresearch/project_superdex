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

# Mochi Physics Remote Debugger Protocol (shared by client and server)
mochi_add_interface(mochi_physics_dbg)
target_include_directories(mochi_physics_dbg INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Depend on mochi_physics public headers (not the library).
target_link_libraries(mochi_physics_dbg INTERFACE mochi_core mochi_physics_headers)
