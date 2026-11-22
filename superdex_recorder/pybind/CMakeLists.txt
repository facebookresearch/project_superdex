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
    nanobind_add_module(superdex_recorder_pybind STABLE_ABI NB_DOMAIN mochi)
else ()
    nanobind_add_module(superdex_recorder_pybind NB_DOMAIN mochi)
endif ()
target_sources(superdex_recorder_pybind PRIVATE src/superdex_recorder_pybind.cpp)
target_link_libraries(
    superdex_recorder_pybind
    PUBLIC superdex_recorder mochi_physics mochi_pybind_core
)
target_include_directories(superdex_recorder_pybind PRIVATE ${SUPERDEX_MOCHI_SOURCE_DIR})
mochi_set_compiler_options(superdex_recorder_pybind)
mochi_set_output_directories(superdex_recorder_pybind)
mochi_set_pybind_rpath(superdex_recorder_pybind)

mochi_set_variant_output_name(superdex_recorder_pybind "superdex_recorder")
set_target_properties(superdex_recorder_pybind PROPERTIES PREFIX "")
if (WIN32)
    set_target_properties(superdex_recorder_pybind PROPERTIES SUFFIX ".pyd")
elseif (NOT SKBUILD_SABI_VERSION)
    set_target_properties(superdex_recorder_pybind PROPERTIES SUFFIX ".so")
endif ()

install(TARGETS superdex_recorder_pybind DESTINATION . COMPONENT superdex_recorder_pybind)
