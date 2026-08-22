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

pybind11_add_module(mochi_mesh_pybind mesh_tools_pybind.cpp)
target_link_libraries(mochi_mesh_pybind PUBLIC mochi_mesh)
target_include_directories(mochi_mesh_pybind PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
mochi_set_compiler_options(mochi_mesh_pybind)
mochi_set_output_directories(mochi_mesh_pybind)
mochi_set_pybind_rpath(mochi_mesh_pybind)

# Output name must match the PyInit_ symbol selected by the precision build flag, so both
# variants can be bundled side by side in one wheel.
if (MOCHI_USE_DOUBLE_PRECISION)
    set(mochi_mesh_pybind_output_name "mochi_mesh_double")
else ()
    set(mochi_mesh_pybind_output_name "mochi_mesh")
endif ()
set_target_properties(mochi_mesh_pybind PROPERTIES
    PREFIX ""
    OUTPUT_NAME "${mochi_mesh_pybind_output_name}"
)
if (WIN32)
    set_target_properties(mochi_mesh_pybind PROPERTIES SUFFIX ".pyd")
else ()
    set_target_properties(mochi_mesh_pybind PROPERTIES SUFFIX ".so")
endif ()

install(TARGETS mochi_mesh_pybind DESTINATION . COMPONENT mochi_mesh_pybind)

if (MOCHI_BUILD_SHARED)
    install(TARGETS mochi_mesh DESTINATION . COMPONENT mochi_mesh_pybind)
endif ()
