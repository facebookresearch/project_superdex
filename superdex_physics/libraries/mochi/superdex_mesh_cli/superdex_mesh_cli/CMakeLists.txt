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

# superdex_mesh_cli is the separately-shipped, GPL-licensed helper executable -- the ONLY Mochi
# target that links CGAL. It holds the relocated CGAL geometry algorithms and is invoked by the
# (CGAL-free) shipping libraries over a pipe. This is a standalone build, kept out of the default
# aggregate so the core libraries never link CGAL (see MOCHI_BUILD_MESH_CLI in the top CMakeLists).

file(GLOB cli_srcs "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(superdex_mesh_cli ${cli_srcs})

target_include_directories(superdex_mesh_cli PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

# The helper deliberately links no *compiled* Mochi library. On MSVC it pins the release DLL CRT to
# match the always-Release OCCT toolkits (see the OCCT block in the parent CMakeLists), while the
# Mochi libraries follow the parent build's configuration -- mixing CRTs inside one binary is an
# LNK2038 error. The wire-encoding sources are therefore compiled straight into the helper rather
# than linked from mochi_mesh_protocol_private; its include directories are still consumed from the
# target so the two sides cannot drift. Everything else the helper uses (protocol_public, CGAL,
# Boost, Eigen) is header-only and so carries no CRT of its own.
get_target_property(_protocol_private_srcs mochi_mesh_protocol_private SOURCES)
target_sources(superdex_mesh_cli PRIVATE ${_protocol_private_srcs})
target_include_directories(
    superdex_mesh_cli
    PRIVATE $<TARGET_PROPERTY:mochi_mesh_protocol_private,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(
    superdex_mesh_cli
    PRIVATE
    mochi_mesh_protocol_public
    CGAL::CGAL)

# Pinned independently of the parent build's configuration: this is a standalone executable that
# shares no compiled code with Mochi, so it is free to use the CRT its own dependencies need.
set_property(TARGET superdex_mesh_cli PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

# Enable CGAL's Eigen integration (adaptive sizing / ACVD remeshing) only when Eigen is available;
# those code paths are otherwise compiled out via MOCHI_USE_EIGEN. Eigen3::Eigen is only defined
# when MOCHI_USE_EIGEN is on, so the link has to be guarded along with the definitions.
if (MOCHI_USE_EIGEN)
    target_compile_definitions(superdex_mesh_cli PRIVATE "CGAL_EIGEN3_ENABLED" "MOCHI_USE_EIGEN=1")
    target_link_libraries(superdex_mesh_cli PRIVATE Eigen3::Eigen)
else ()
    target_compile_definitions(superdex_mesh_cli PRIVATE "MOCHI_USE_EIGEN=0")
endif ()

# OCCT (STEP reading + tessellation) is linked only when its toolkits were built (see the OCCT
# block in the parent CMakeLists). Otherwise the STEP path compiles out via MOCHI_USE_OCCT=0 and
# those ops return a clean "unsupported" error.
if (TARGET TKDESTEP)
    target_compile_definitions(superdex_mesh_cli PRIVATE "MOCHI_USE_OCCT=1")
    target_include_directories(superdex_mesh_cli SYSTEM PRIVATE "${MOCHI_OCCT_INCLUDE_DIR}")
    target_include_directories(superdex_mesh_cli SYSTEM PRIVATE "${MOCHI_RAPIDJSON_INCLUDE_DIR}")
    # MOCHI_OCCT_LIBRARIES is a flat list of imported static archives that declare no
    # interdependencies, and the OCCT toolkits reference each other cyclically (TKXCAF -> TKernel,
    # TKV3d -> TKHLR, ...). GNU ld scans archives once, left to right, and would report undefined
    # references for any toolkit whose provider appears earlier on the line; RESCAN wraps the set in
    # --start-group/--end-group so it rescans until the symbols settle. Apple's ld and MSVC's
    # link.exe resolve static archives order-independently, and CMake defines RESCAN only for the
    # linkers that need it -- hence the guard rather than an unconditional LINK_GROUP.
    if (CMAKE_LINK_GROUP_USING_RESCAN_SUPPORTED)
        target_link_libraries(
            superdex_mesh_cli PRIVATE "$<LINK_GROUP:RESCAN,${MOCHI_OCCT_LIBRARIES}>")
    else ()
        target_link_libraries(superdex_mesh_cli PRIVATE ${MOCHI_OCCT_LIBRARIES})
    endif ()
    target_link_libraries(superdex_mesh_cli PRIVATE ${MOCHI_OCCT_SYSTEM_LIBRARIES})
else ()
    target_compile_definitions(superdex_mesh_cli PRIVATE "MOCHI_USE_OCCT=0")
endif ()

mochi_set_compiler_options(superdex_mesh_cli)
mochi_set_output_directories(superdex_mesh_cli)
set_property(TARGET superdex_mesh_cli PROPERTY CXX_STANDARD 20)

mochi_install_runtime_dependencies(superdex_mesh_cli COMPONENT superdex_mesh_cli)
