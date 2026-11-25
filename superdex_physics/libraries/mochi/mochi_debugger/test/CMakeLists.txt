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

# Keep this target explicit: mochi_add_test(mochi_debugger_lib) would rename it
# to mochi_debugger_lib_test, but the CMake runner invokes mochi_debugger_test.
mochi_add_test_executable(mochi_debugger_test "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(mochi_debugger_test PRIVATE ${PROJECT_SOURCE_DIR})

target_link_libraries(mochi_debugger_test
    PRIVATE
    mochi_core
    mochi_debugger_lib
    mochi_physics_dbg
    EnTT)

# Unit tests always statically link mochi_physics, even when MOCHI_BUILD_SHARED.
if (MOCHI_BUILD_SHARED)
    target_link_libraries(mochi_debugger_test PRIVATE mochi_physics_static)
else ()
    target_link_libraries(mochi_debugger_test PRIVATE mochi_physics)
endif ()

if (TARGET imgui)
    mochi_add_test_executable(mochi_debugger_gui_test
        "${CMAKE_CURRENT_SOURCE_DIR}/../test_support/gui")
    target_sources(mochi_debugger_gui_test PRIVATE
        ../app/gui/command_input.cpp)
    target_include_directories(mochi_debugger_gui_test PRIVATE ../app/gui)
    target_link_libraries(mochi_debugger_gui_test PRIVATE mochi_core mochi_debugger_lib imgui gtest_main)
endif ()
