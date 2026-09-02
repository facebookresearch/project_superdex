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

# The CMake target is named mochi_physics_pybind (mirroring superdex_robotics_pybind) to
# avoid colliding with the mochi_physics *library* target created in
# ../CMakeLists.txt. The extension is still emitted as mochi_physics.{so,pyd} via OUTPUT_NAME.
pybind11_add_module(mochi_physics_pybind)
file(GLOB_RECURSE mochi_physics_srcs src/*.cpp)
# Drop all internal-only sources (hand-written src/internal/ and generated src/generated/internal/).
list(FILTER mochi_physics_srcs EXCLUDE REGEX "/internal/")
target_sources(mochi_physics_pybind PRIVATE ${mochi_physics_srcs})

# Shared "pybind core" library: owns the single process-wide instance of the bindings-layer
# context state (g_context, the MochiErrorException RTTI, the context lifecycle, and the ordered
# teardown registry). Both extensions (physics, bots) dynamically link it, so they share one
# g_context and one exported MochiErrorException — the CMake analogue of the Buck
# core:pybind_core shared library. Built SHARED unconditionally: its whole purpose is to be a
# single shared instance.
add_library(mochi_pybind_core SHARED core/pybind_core.cpp)
mochi_set_hidden_symbol_visibility(mochi_pybind_core)
target_link_libraries(mochi_pybind_core PUBLIC mochi_physics pybind11::module)
target_include_directories(mochi_pybind_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/core)
mochi_set_compiler_options(mochi_pybind_core)
mochi_set_output_directories(mochi_pybind_core)
# Resolve the sibling shared libs (libmochi_physics) from the library's own dir.
mochi_set_pybind_rpath(mochi_pybind_core)
target_compile_definitions(mochi_pybind_core PRIVATE "MOCHI_USE_REFLECTION=1")
# Same reason as mochi_physics: the two precisions ship in separate wheels and are resolved
# across wheel directories by SONAME, so they must not share one.
if (MOCHI_USE_DOUBLE_PRECISION)
    set_target_properties(mochi_pybind_core PROPERTIES OUTPUT_NAME "mochi_pybind_core_double")
endif ()

target_link_libraries(mochi_physics_pybind PUBLIC mochi_physics mochi_pybind_core)
# Restricted diffsim header, used by pybind_include.h and the generated bindings. Linked PRIVATE
# (non-exported).
target_link_libraries(mochi_physics_pybind PRIVATE mochi_diffsim_headers)
# src/ for the pybind helper headers; the mochi root (../..) so pybind_include.h can resolve
# <mochi_physics/pybind/core/pybind_core.h>.
target_include_directories(mochi_physics_pybind PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
mochi_set_compiler_options(mochi_physics_pybind)
mochi_set_output_directories(mochi_physics_pybind)
# Resolve the sibling shared libs (libmochi_physics, libmochi_pybind_core) from the
# extension's own dir.
mochi_set_pybind_rpath(mochi_physics_pybind)
target_compile_definitions(mochi_physics_pybind PRIVATE "MOCHI_USE_REFLECTION=1")

# Output name must match MOCHI_PHYSICS_MODULE_NAME's PyInit_ symbol (pybind_helpers.h).
if (MOCHI_USE_DOUBLE_PRECISION)
    set(mochi_physics_pybind_output_name "mochi_physics_double")
else ()
    set(mochi_physics_pybind_output_name "mochi_physics")
endif ()
set_target_properties(mochi_physics_pybind PROPERTIES
    PREFIX ""
    OUTPUT_NAME "${mochi_physics_pybind_output_name}"
    # MSVC names the PDB after OUTPUT_NAME, and the mochi_physics *library* target emits into the
    # same bin/ directory. Both would write mochi_physics.pdb, and the loser silently loses its
    # symbols. Only the PDB is renamed; the wheel repair exclusions match on OUTPUT_NAME.
    PDB_NAME "${mochi_physics_pybind_output_name}.pyd"
)
if(WIN32)
    set_target_properties(mochi_physics_pybind PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(mochi_physics_pybind PROPERTIES SUFFIX ".so")
endif()

# Bundle the extension's own shared libs (mochi_physics, mochi_pybind_core, marl) plus any
# genuinely external runtime deps (HDF5, zlib) so the wheel is self-contained.
# RUNTIME_DEPENDENCY_SET only catches external deps, not same-project targets, so
# mochi_physics/marl must be installed explicitly. Only needed when MOCHI_BUILD_SHARED
# produces separate .so/.dlls -- a static build absorbs everything into the extension.
set(mochi_physics_pybind_install_targets mochi_physics_pybind mochi_pybind_core)
if (MOCHI_BUILD_SHARED)
    list(APPEND mochi_physics_pybind_install_targets mochi_physics marl)
    # mochi_physics also needs RPATH to find bundled deps like libhdf5 alongside it.
    mochi_set_pybind_rpath(mochi_physics)
endif ()

install(TARGETS ${mochi_physics_pybind_install_targets}
    RUNTIME_DEPENDENCY_SET mochi_physics_pybind_deps
    RUNTIME DESTINATION . COMPONENT mochi_physics_pybind
    LIBRARY DESTINATION . COMPONENT mochi_physics_pybind
    # No ARCHIVE: .lib/.a files are link-time only, not needed in the wheel.
)
install(RUNTIME_DEPENDENCY_SET mochi_physics_pybind_deps
    DESTINATION .
    COMPONENT mochi_physics_pybind
    DIRECTORIES $<TARGET_FILE_DIR:mochi_physics_pybind>
    # Skip OS-provided libraries and the Python runtime -- the target environment always has
    # its own. Anchored to path roots so HDF5/zlib (nested under a lib/ dir) aren't excluded too.
    PRE_EXCLUDE_REGEXES "api-ms-.*" "ext-ms-.*" "python3.*\\.dll" "libpython3.*\\.(so|dylib).*"
    POST_EXCLUDE_REGEXES ".*system32.*" "^/lib/.*" "^/lib64/.*" "^/usr/lib/.*" "^/usr/lib64/.*" ".*/System/Library/.*"
)
