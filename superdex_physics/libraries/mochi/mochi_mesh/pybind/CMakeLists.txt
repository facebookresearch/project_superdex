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

if (SKBUILD_SABI_VERSION)
    nanobind_add_module(mochi_mesh_pybind STABLE_ABI mesh_tools_pybind.cpp)
else ()
    nanobind_add_module(mochi_mesh_pybind mesh_tools_pybind.cpp)
endif ()
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
mochi_set_variant_output_name(mochi_mesh_pybind "mochi_mesh")
set_target_properties(mochi_mesh_pybind PROPERTIES PREFIX "")
if (WIN32)
    set_target_properties(mochi_mesh_pybind PROPERTIES SUFFIX ".pyd")
elseif (NOT SKBUILD_SABI_VERSION)
    set_target_properties(mochi_mesh_pybind PROPERTIES SUFFIX ".so")
endif ()

install(TARGETS mochi_mesh_pybind DESTINATION . COMPONENT mochi_mesh_pybind)

if (MOCHI_BUILD_SHARED)
    mochi_set_pybind_rpath(mochi_mesh)
    install(TARGETS mochi_mesh
        RUNTIME DESTINATION . COMPONENT mochi_mesh_pybind
        LIBRARY DESTINATION . COMPONENT mochi_mesh_pybind
    )
endif ()
