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

# Public wire-protocol types shared with the superdex_mesh_cli helper: the parameter structs
# (SurfaceRemeshingParams, StepTessellationParams, ...) plus their enum members. Both mochi_mesh
# and the CLI helper include this header, so the on-wire memory image cannot silently drift.
# Header-only; no Mochi dependencies, no encoding logic -- see mochi_mesh_protocol_private
# for the framing / (de)serialization primitives.
add_library(mochi_mesh_protocol_public INTERFACE)
target_include_directories(
    mochi_mesh_protocol_public
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/protocol/public")

# Private wire-protocol encoding shared with the superdex_mesh_cli helper: frame magic, protocol
# version, GeometryOp, MeshData, and the PayloadReader/PayloadWriter (de)serialization primitives.
# Both mochi_mesh and the CLI helper link this so they encode the wire identically. Kept separate
# from mochi_mesh_protocol_public so consumers that only need the parameter type
# definitions do not pay for the encoding headers.
add_library(
    mochi_mesh_protocol_private STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/protocol/private/src/mochi_mesh_cli_encoding.cpp")
set_property(TARGET mochi_mesh_protocol_private PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET mochi_mesh_protocol_private PROPERTY CXX_STANDARD 20)
target_include_directories(
    mochi_mesh_protocol_private
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/protocol/private")
target_link_libraries(mochi_mesh_protocol_private PUBLIC mochi_mesh_protocol_public)

if (MOCHI_BUILD_TESTS)
    file(GLOB mochi_mesh_protocol_test_srcs
        "${CMAKE_CURRENT_SOURCE_DIR}/protocol/private/test/*.cpp")
    add_executable(
        mochi_mesh_protocol_test
        ${mochi_mesh_protocol_test_srcs})
    set_property(TARGET mochi_mesh_protocol_test PROPERTY CXX_STANDARD 20)
    target_link_libraries(
        mochi_mesh_protocol_test
        PRIVATE mochi_mesh_protocol_private GTest::gtest_main)
    mochi_set_output_directories(mochi_mesh_protocol_test)
endif ()

mochi_add_library(mochi_mesh)

# mochi_mesh marshals its heavier mesh operations to the superdex_mesh_cli helper over a pipe (see
# mesh_cli_client). It depends only on mochi_core and the wire-protocol targets above; it never
# links CGAL itself.
target_link_libraries(mochi_mesh
    PUBLIC
    mochi_core
    mochi_mesh_protocol_public)
target_link_libraries(mochi_mesh
    PRIVATE
    mochi_mesh_protocol_private)

# Allow the library's own sources to include their internal src headers (e.g. mesh_cli_client.h).
target_include_directories(mochi_mesh PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

# mochi_mesh has no DLL export annotations, so export all symbols when built as a shared library.
if (MOCHI_BUILD_SHARED)
    set_target_properties(mochi_mesh PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif ()

# The auto-generated test (added by mochi_add_library when MOCHI_BUILD_TESTS is on) includes the
# internal mesh_cli_client.h and the private wire encoding header directly. The mochi_mesh_test
# spawns the CLI helper at runtime, but the add_dependencies() on that helper lives in the mochi
# root CMakeLists.txt since the helper is defined by a sibling
# add_subdirectory that runs after this file.
if (MOCHI_BUILD_TESTS AND TARGET mochi_mesh_test)
    target_include_directories(
        mochi_mesh_test
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/test"
    )
    target_link_libraries(mochi_mesh_test PRIVATE mochi_mesh_protocol_private)
endif ()

if (MOCHI_BUILD_TESTS)
    add_executable(
        mochi_mesh_cli_concurrency_helper
        "${CMAKE_CURRENT_SOURCE_DIR}/test_tools/concurrency/mesh_cli_concurrency_helper.cpp")
    set_property(TARGET mochi_mesh_cli_concurrency_helper PROPERTY CXX_STANDARD 20)
    target_link_libraries(
        mochi_mesh_cli_concurrency_helper
        PRIVATE mochi_core mochi_mesh_protocol_private)
    mochi_set_compiler_options(mochi_mesh_cli_concurrency_helper)
    mochi_set_output_directories(mochi_mesh_cli_concurrency_helper)

    add_executable(
        mochi_mesh_cli_concurrency_test
        "${CMAKE_CURRENT_SOURCE_DIR}/test_tools/concurrency/mesh_cli_client_concurrency_test.cpp")
    set_property(TARGET mochi_mesh_cli_concurrency_test PROPERTY CXX_STANDARD 20)
    target_include_directories(
        mochi_mesh_cli_concurrency_test
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        mochi_mesh_cli_concurrency_test
        PRIVATE mochi_mesh mochi_mesh_protocol_private GTest::gtest_main)
    mochi_set_compiler_options(mochi_mesh_cli_concurrency_test)
    mochi_set_output_directories(mochi_mesh_cli_concurrency_test)
    add_dependencies(
        mochi_mesh_cli_concurrency_test
        mochi_mesh_cli_concurrency_helper)
    add_test(
        NAME mochi_mesh_cli_concurrency_test
        COMMAND mochi_mesh_cli_concurrency_test
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:mochi_mesh_cli_concurrency_test>")
    set_property(
        TEST mochi_mesh_cli_concurrency_test
        APPEND PROPERTY ENVIRONMENT
        "SUPERDEX_MESH_CLI_PATH=$<TARGET_FILE:mochi_mesh_cli_concurrency_helper>")
endif ()

if (MOCHI_USE_PYBIND AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/pybind")
    add_subdirectory(pybind)
endif ()
