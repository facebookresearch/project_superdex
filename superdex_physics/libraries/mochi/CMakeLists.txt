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

cmake_policy(VERSION 3.25)
cmake_minimum_required(VERSION 3.25)

# Apple's linker rescans static archives, so a library repeated on the link line is redundant and
# is diagnosed as a duplicate-libraries warning. CMP0156 NEW de-duplicates link-line entries, so
# only one copy of each library is passed and the warning goes away as a consequence.
if (POLICY CMP0156)
    cmake_policy(SET CMP0156 NEW)
endif ()

# Select default compiler. Must come before "project(mochi)".
include("${CMAKE_CURRENT_LIST_DIR}/cmake/compiler.cmake")

#-----------------------------------------------------------------------------
# Root Project Setup
#-----------------------------------------------------------------------------

project(mochi)

# Include build-wide settings that must be applied by the top level project.
# No-op if an outer project already included it.
include("${CMAKE_CURRENT_LIST_DIR}/cmake/root_project_setup.cmake")

include("cmake/paths.cmake")
include("cmake/utils.cmake")

#-----------------------------------------------------------------------------
# Mochi Settings
#-----------------------------------------------------------------------------

include("cmake/settings.cmake")

# Local user settings can go in "cmake/settings.local.cmake", if you create one.
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/settings.local.cmake")
    include(cmake/settings.local.cmake)
endif ()

string(TOUPPER "${CMAKE_BUILD_TYPE}" _mochi_build_type_upper)
if (
    MOCHI_WARN_IF_NOT_OPTIMIZED
    AND NOT CMAKE_CONFIGURATION_TYPES
    AND _mochi_build_type_upper STREQUAL "DEBUG"
)
    message(WARNING
        "********************************************************************************\n"
        "This is a Debug build, which disables compiler optimizations.\n"
        "This will greatly reduce performance.\n"
        "To enable optimizations, configure CMake with -DCMAKE_BUILD_TYPE=Release.\n"
        "To silence this warning, use -DMOCHI_WARN_IF_NOT_OPTIMIZED=OFF.\n"
        "********************************************************************************"
    )
endif()
unset(_mochi_build_type_upper)

#-----------------------------------------------------------------------------
# Derived Settings
#-----------------------------------------------------------------------------

if (MOCHI_BUILD_ALL)
    set(MOCHI_BUILD_BENCHMARKS ON CACHE BOOL "" FORCE)
    set(MOCHI_BUILD_DEBUGGER ON CACHE BOOL "" FORCE)
    set(MOCHI_BUILD_IMGUIOS ON CACHE BOOL "" FORCE)
    set(MOCHI_BUILD_TESTS ON CACHE BOOL "" FORCE)
    set(MOCHI_BUILD_RENDERER ON CACHE BOOL "" FORCE)
    set(MOCHI_BUILD_MESH_CLI ON CACHE BOOL "" FORCE)
    set(MOCHI_USE_PYBIND ON CACHE BOOL "" FORCE) # Required for pybind targets to build
endif ()

if (MOCHI_BUILD_DEBUGGER)
    set(MOCHI_BUILD_IMGUIOS ON CACHE BOOL "" FORCE)
    set(MOCHI_BUILD_RENDERER ON CACHE BOOL "" FORCE)
endif ()

# Make tinyfiledialog available to all ImGui based apps.
if (MOCHI_BUILD_IMGUIOS)
    set(MOCHI_USE_TINYFILEDIALOG ON CACHE BOOL "" FORCE)
endif ()

#-----------------------------------------------------------------------------
# Third Party Dependencies
#-----------------------------------------------------------------------------

if (MOCHI_BUILD_BENCHMARKS AND MOCHI_BENCHMARK_FIND_PACKAGE)
    find_package(benchmark 1.6.1 REQUIRED)
endif ()

if (MOCHI_FIND_ENTT_PACKAGE)
    find_package(EnTT 3.9.0 REQUIRED)
endif ()

if (MOCHI_BUILD_TESTS AND MOCHI_GTEST_FIND_PACKAGE)
    find_package(GTest 1.14.0 REQUIRED)
endif ()

if (MOCHI_USE_HDF5 OR MOCHI_BUILD_RENDERER)
    if (MOCHI_ZLIB_FIND_PACKAGE)
        find_package(ZLIB 1.3.1 REQUIRED)
    endif ()
endif ()

if (MOCHI_USE_HDF5)
    if (MOCHI_HDF5_FIND_PACKAGE)
        find_package(HDF5 1.14.4.3 COMPONENTS C HL CXX REQUIRED)
    endif ()
endif ()

if (MOCHI_USE_PYBIND OR MOCHI_BUILD_RENDERER)
    # Components needed:
    #   Interpreter: runs bin_to_c.py (mochi_renderer + Filament resource wiring)
    #   Development.Module: provides Python.h and the extension-module link rules (pybind11)
    #
    # Not plain Development: it implies Development.Embed, which requires libpython. manylinux
    # CPython ships none, so requiring Embed fails configure on the wheel-build platform.
    set(_mochi_python3_components Interpreter)
    if (MOCHI_USE_PYBIND)
        list(APPEND _mochi_python3_components Development.Module)
    endif ()
    # Pass the same version range pybind11's later find_package() uses so the
    # first (sticky) lookup picks a compatible interpreter. Without this CMake
    # may resolve to an older system Python (e.g. /usr/bin/python3.9), which
    # then fails pybind11's 3.12.x check.
    find_package(Python3 3.12...3.12.99 REQUIRED COMPONENTS ${_mochi_python3_components})
    unset(_mochi_python3_components)
endif ()

if (MOCHI_USE_PYBIND AND MOCHI_PYBIND_FIND_PACKAGE)
    find_package(pybind11 2.13.6 REQUIRED)
endif ()

if (MOCHI_USE_TRACY)
    set(TRACY_VERSION 0.12.1)
    if (MOCHI_TRACY_FIND_PACKAGE)
        find_package(tracy ${TRACY_VERSION} REQUIRED)
    endif ()
endif ()

add_subdirectory(third_party)

#-----------------------------------------------------------------------------
# Testing
#-----------------------------------------------------------------------------

# Must precede the subdirectories that call add_test(), and deliberately follows
# third_party so vendored projects do not inherit it. A parent project embedding
# Mochi needs its own enable_testing() to run ctest from its build root.
if (MOCHI_BUILD_TESTS)
    enable_testing()

    # Asset root for the test executables registered in this directory and below.
    # Overridable, but never CACHE: a sibling project's tests keep their assets
    # elsewhere and must not inherit Mochi's. Left unset in the OSS export, which
    # ships no assets/.
    if (NOT MOCHI_TEST_ASSETS_DIR AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets")
        set(MOCHI_TEST_ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")
    endif ()
endif ()

#-----------------------------------------------------------------------------
# imguios (first-party library)
#-----------------------------------------------------------------------------

# imguios is consumed by mochi_debugger; skip otherwise.
if (MOCHI_BUILD_IMGUIOS)
    message(STATUS "[Mochi] Adding imguios at '${MOCHI_IMGUIOS_SOURCE_DIR}'")
    set(IMGUIOS_BUILD_EXAMPLE OFF)
    if (APPLE)
        set(IMGUIOS_BUILD_METAL ON)
    endif ()
    add_subdirectory("${MOCHI_IMGUIOS_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/imguios")
    mochi_target_warnings_not_errors(imguios)
    mochi_set_compiler_options(imguios)
    set_target_properties(imguios PROPERTIES FOLDER "libraries")
endif ()

#-----------------------------------------------------------------------------
# simple_reflection (first-party library)
#-----------------------------------------------------------------------------

message(STATUS "[Mochi] Adding simple_reflection at '${MOCHI_SIMPLE_REFLECTION_SOURCE_DIR}'")
set(SR_BUILD_TESTS OFF)
set(SR_USE_NLOHMANN_JSON OFF)
set(SR_PICOJSON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/picojson")
add_subdirectory("${MOCHI_SIMPLE_REFLECTION_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/simple_reflection")
mochi_target_warnings_not_errors(simple_reflection)
mochi_set_compiler_options(simple_reflection)
set_target_properties(simple_reflection PROPERTIES FOLDER "libraries")

#-----------------------------------------------------------------------------
# Mochi Projects
#-----------------------------------------------------------------------------

# Always included:
add_subdirectory(mochi_core)
add_subdirectory(mochi_physics)

# mochi_mesh owns the wire-protocol targets both sides link, so it must be configured first.
if (MOCHI_BUILD_MESH_CLI)
    set(MOCHI_BUILD_MESH_LIB ON CACHE BOOL "" FORCE)
endif ()

if (MOCHI_BUILD_MESH_LIB)
    add_subdirectory(mochi_mesh)
endif ()

if (MOCHI_BUILD_MESH_CLI)
    add_subdirectory(superdex_mesh_cli)
    # mochi_mesh_test spawns the CLI helper at runtime; wire the build dependency here since the
    # helper target is not defined until superdex_mesh_cli's add_subdirectory above has run.
    if (MOCHI_BUILD_TESTS AND TARGET mochi_mesh_test)
        add_dependencies(mochi_mesh_test superdex_mesh_cli)
    endif ()
endif ()

if (MOCHI_BUILD_RENDERER)
    add_subdirectory(mochi_renderer)
endif ()

if (MOCHI_BUILD_DEBUGGER)
    add_subdirectory(mochi_debugger)
endif ()

#-----------------------------------------------------------------------------
# Allow experimental test apps that are not checked in.
#-----------------------------------------------------------------------------

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/execs.local.cmake")
    include(cmake/execs.local.cmake)
endif ()
