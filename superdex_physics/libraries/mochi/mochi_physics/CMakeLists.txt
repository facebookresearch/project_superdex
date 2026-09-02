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

#-----------------------------------------------------------------------------
# Public header-only INTERFACE target
#-----------------------------------------------------------------------------
add_library(mochi_physics_headers INTERFACE)
target_include_directories(mochi_physics_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

#-----------------------------------------------------------------------------
# Remote debugger support (shared by client and server)
#-----------------------------------------------------------------------------
add_subdirectory(dbg)

#-----------------------------------------------------------------------------
# Restricted (internal) headers: mochi_diffsim_headers
#-----------------------------------------------------------------------------
add_library(mochi_diffsim_headers INTERFACE)
target_include_directories(mochi_diffsim_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/private/diffsim/include>)

#-----------------------------------------------------------------------------
# Main library: mochi_physics
# Builds as SHARED (DLL) when MOCHI_BUILD_SHARED=ON, STATIC otherwise.
# External consumers (hello_world, samples, pybind, etc.) always link to this.
#-----------------------------------------------------------------------------
mochi_add_library(mochi_physics)

# The two precisions ship as separate wheels that can share a sys.path, and are resolved by
# SONAME, so they must not share one. Otherwise the loader binds whichever was dlopen'ed first,
# and a double-precision extension silently runs against float symbols. No SOVERSION is set
# anywhere, so the SONAME follows OUTPUT_NAME.
if (MOCHI_USE_DOUBLE_PRECISION)
    set_target_properties(mochi_physics PROPERTIES OUTPUT_NAME "mochi_physics_double")
endif ()

target_link_libraries(mochi_physics
    PUBLIC
    mochi_core
    mochi_physics_headers
    PRIVATE
    marl
    EnTT
    mochi_physics_dbg
    mochi_diffsim_headers
)

if (MOCHI_BUILD_SHARED)
    target_compile_definitions(mochi_physics
        PUBLIC "MOCHI_PHYSICS_DYNAMICALLY_LINKED=1"
        PRIVATE "MOCHI_PHYSICS_EXPORTS=1"
    )
else ()
    target_compile_definitions(mochi_physics
        PUBLIC "MOCHI_PHYSICS_DYNAMICALLY_LINKED=0"
    )
endif ()

#-----------------------------------------------------------------------------
# Additional static library for tests and benchmarks in shared builds
#
# Tests and benchmarks link mochi_physics_static so that they can call internal functions without
# making those symbols visible to the dynamic linker. On Windows, this also avoids EnTT type ID
# issues across DLL boundaries.
#-----------------------------------------------------------------------------
if (MOCHI_BUILD_SHARED AND (MOCHI_BUILD_TESTS OR MOCHI_BUILD_BENCHMARKS))
    get_target_property(_SOURCES mochi_physics SOURCES)
    get_target_property(_INCLUDES mochi_physics INCLUDE_DIRECTORIES)

    add_library(mochi_physics_static STATIC ${_SOURCES})
    target_include_directories(mochi_physics_static PRIVATE ${_INCLUDES})
    target_include_directories(mochi_physics_static PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/mochi_physics/include>)
    target_link_libraries(mochi_physics_static PUBLIC mochi_core PRIVATE marl EnTT mochi_physics_dbg mochi_diffsim_headers)
    target_compile_definitions(mochi_physics_static
        PUBLIC "MOCHI_PHYSICS_DYNAMICALLY_LINKED=0"
        PRIVATE "MOCHI_USE_REFLECTION=1")
    mochi_set_compiler_options(mochi_physics_static)
endif ()

#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------
if (MOCHI_BUILD_TESTS)
    target_include_directories(mochi_physics_test PRIVATE ${PROJECT_SOURCE_DIR})

    if (TARGET mochi_physics_static)
        # Re-link tests against static library (DLL still exists for other consumers)
        mochi_remove_link_library(mochi_physics_test mochi_physics)
        target_link_libraries(mochi_physics_test PRIVATE mochi_physics_static)
    endif ()

    target_link_libraries(mochi_physics_test PRIVATE marl EnTT)

    #-------------------------------------------------------------------------
    # Differentiable-simulation tests (mochi_physics_differentiable_test)
    #
    # Exercises the restricted <mochi_physics/diffsim/...> headers, so it lives
    # in its own executable, separate from mochi_physics_test.
    # mochi_add_test only models one <library>_test from test/. This second
    # target combines private/diffsim/test with the shared scene fixture below.
    #-------------------------------------------------------------------------
    mochi_add_test_executable(mochi_physics_differentiable_test
        ${CMAKE_CURRENT_SOURCE_DIR}/private/diffsim/test)

    # Compile the shared gtest scene fixture directly (no standalone fixture lib).
    target_sources(mochi_physics_differentiable_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/test/mochi_physics_test_fixture.cpp)

    target_include_directories(mochi_physics_differentiable_test PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/test
        ${CMAKE_CURRENT_SOURCE_DIR}/private/diffsim/test)

    target_link_libraries(mochi_physics_differentiable_test PRIVATE
        mochi_physics mochi_diffsim_headers marl EnTT)

    if (TARGET mochi_physics_static)
        # Re-link tests against static library (DLL still exists for other consumers)
        mochi_remove_link_library(mochi_physics_differentiable_test mochi_physics)
        target_link_libraries(mochi_physics_differentiable_test PRIVATE mochi_physics_static)
    endif ()
endif ()

#-----------------------------------------------------------------------------
# Benchmarks
#-----------------------------------------------------------------------------
if (MOCHI_BUILD_BENCHMARKS)
    target_include_directories(mochi_physics_benchmark PRIVATE ${PROJECT_SOURCE_DIR})

    if (TARGET mochi_physics_static)
        # Re-link benchmarks against static library (DLL still exists for other consumers)
        mochi_remove_link_library(mochi_physics_benchmark mochi_physics)
        target_link_libraries(mochi_physics_benchmark PRIVATE mochi_physics_static)
    endif ()

    target_link_libraries(mochi_physics_benchmark PRIVATE marl EnTT)
endif ()

if (MOCHI_USE_PYBIND)
    add_subdirectory(pybind)
endif ()
