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

# Passes CMAKE_POLICY_VERSION_MINIMUM on to Arrow's mimalloc build, which CMake 4 otherwise
# rejects. The Rerun SDK 0.33 does the same with `git apply`, which silently does nothing when the
# build directory is inside a git checkout. Later Rerun SDKs fix that; delete this with the upgrade.
set(toolchain "${ARROW_SOURCE_DIR}/cpp/cmake_modules/ThirdpartyToolchain.cmake")
file(READ "${toolchain}" contents)
set(anchor [=[
      -DMI_BUILD_TESTS=OFF]=])
set(patched [=[
      -DMI_BUILD_TESTS=OFF
      "-DCMAKE_POLICY_VERSION_MINIMUM=${CMAKE_POLICY_VERSION_MINIMUM}"]=])

string(FIND "${contents}" "${patched}" patched_position)
if(NOT patched_position EQUAL -1)
    return()
endif()

string(FIND "${contents}" "${anchor}" anchor_position)
if(anchor_position EQUAL -1)
    message(FATAL_ERROR "Arrow's mimalloc arguments have changed; update the mimalloc policy patch.")
endif()

string(REPLACE "${anchor}" "${patched}" contents "${contents}")
file(WRITE "${toolchain}" "${contents}")
