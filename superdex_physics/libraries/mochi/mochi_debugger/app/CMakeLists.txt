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

mochi_add_executable(mochi_debugger "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(mochi_debugger
        PRIVATE
        mochi_core
        mochi_renderer
        mochi_debugger_lib)
target_include_directories(mochi_debugger
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}")

target_compile_definitions(mochi_debugger
        PRIVATE
        FILAMENT_DISABLE_MATOPT=0
        IMGUI_DEFINE_MATH_OPERATORS)

if (MOCHI_USE_DOUBLE_PRECISION)
    set_target_properties(mochi_debugger PROPERTIES OUTPUT_NAME "mochi_debugger_double")
else ()
    set_target_properties(mochi_debugger PROPERTIES OUTPUT_NAME "mochi_debugger")
endif ()

if (APPLE)
    target_sources(mochi_debugger
            PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/viewport/render_target_metal.mm")
    target_link_libraries(mochi_debugger
            PRIVATE
            imguios_metal)
else ()
    target_link_libraries(mochi_debugger
            PRIVATE
            imguios)
endif ()

mochi_set_gui_subsystem(mochi_debugger)

#-----------------------------------------------------------------------------
# Installation
#-----------------------------------------------------------------------------

mochi_install_runtime_dependencies(mochi_debugger COMPONENT mochi_debugger)
