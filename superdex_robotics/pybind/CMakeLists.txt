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

# CMake target is `superdex_robotics_pybind` (with `_pybind` suffix) so it does not collide
# with the `superdex_robotics` C++ library target; the extension file is named via
# OUTPUT_NAME below so it imports as `superdex_robotics` (or `superdex_robotics_double`).
pybind11_add_module(superdex_robotics_pybind)
file(GLOB_RECURSE superdex_robotics_pybind_srcs src/*.cpp)
list(FILTER superdex_robotics_pybind_srcs EXCLUDE REGEX "/internal/")
target_sources(superdex_robotics_pybind PRIVATE ${superdex_robotics_pybind_srcs})
# Link the bots library (`superdex_robotics`) plus the physics *library* target
# `mochi_physics` (not the mochi_physics_pybind extension MODULE) for shared types, and the
# shared `mochi_pybind_core` library (defined in the physics pybind CMakeLists) for the single
# g_context + exported MochiErrorException.
target_link_libraries(superdex_robotics_pybind PUBLIC superdex_robotics mochi_physics mochi_pybind_core)
# src/ for the bots pybind's own headers; the mochi root so the generated TUs can resolve
# <mochi_physics/pybind/src/...> helper headers. SUPERDEX_MOCHI_SOURCE_DIR is the mochi root that
# this checkout resolved.
target_include_directories(superdex_robotics_pybind PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${SUPERDEX_MOCHI_SOURCE_DIR}
)
mochi_set_compiler_options(superdex_robotics_pybind)
mochi_set_output_directories(superdex_robotics_pybind)
# Resolve the sibling shared libs (libsuperdex_robotics, libmochi_physics) from the
# extension's own dir.
mochi_set_pybind_rpath(superdex_robotics_pybind)
target_compile_definitions(superdex_robotics_pybind PRIVATE "MOCHI_USE_REFLECTION=1")

# Output as superdex_robotics.so (Linux/Mac) or superdex_robotics.pyd (Windows). The output
# name must match the PyInit_ symbol selected by SUPERDEX_ROBOTICS_MODULE_NAME in
# pybind_include.h, which is superdex_robotics_double for double-precision builds.
if (MOCHI_USE_DOUBLE_PRECISION)
    set(superdex_robotics_pybind_output_name "superdex_robotics_double")
else ()
    set(superdex_robotics_pybind_output_name "superdex_robotics")
endif ()
set_target_properties(superdex_robotics_pybind PROPERTIES
    PREFIX ""
    OUTPUT_NAME "${superdex_robotics_pybind_output_name}"
    # Same collision as mochi_physics_pybind: the superdex_robotics library target emits
    # superdex_robotics.dll into this same bin/, so without a rename both linker PDBs would be
    # superdex_robotics.pdb and one would overwrite the other.
    PDB_NAME "${superdex_robotics_pybind_output_name}.pyd"
)
if(WIN32)
    set_target_properties(superdex_robotics_pybind PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(superdex_robotics_pybind PROPERTIES SUFFIX ".so")
endif()

install(TARGETS superdex_robotics_pybind DESTINATION . COMPONENT superdex_robotics_pybind)

# Install the shared bots library next to the extension so its $ORIGIN/@loader_path
# RPATH resolves it at import time. libmochi_physics is installed by the physics
# pybind CMakeLists (always added alongside this one under MOCHI_USE_PYBIND). Only
# needed for shared builds; under a static build the lib is absorbed into the extension.
if (MOCHI_BUILD_SHARED)
    install(TARGETS superdex_robotics DESTINATION . COMPONENT superdex_robotics_pybind)
endif ()
