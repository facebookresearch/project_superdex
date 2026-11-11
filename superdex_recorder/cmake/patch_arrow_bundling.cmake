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

set(build_utils "${ARROW_SOURCE_DIR}/cpp/cmake_modules/BuildUtils.cmake")
file(READ "${build_utils}" contents)
set(original [=[elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|GNU|Intel|IntelLLVM)$")]=])
set(patched [=[elseif(NOT MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|GNU|Intel|IntelLLVM)$")]=])

string(FIND "${contents}" "${patched}" patched_position)
if(NOT patched_position EQUAL -1)
    return()
endif()

string(FIND "${contents}" "${original}" original_position)
if(original_position EQUAL -1)
    message(FATAL_ERROR "Arrow's archive bundling condition has changed; update the clang-cl patch.")
endif()

string(REPLACE "${original}" "${patched}" contents "${contents}")
file(WRITE "${build_utils}" "${contents}")
