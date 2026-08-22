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

# Select default compiler. Must come before "project(superdex_physics)"
include("${CMAKE_CURRENT_LIST_DIR}/cmake/compiler.cmake")

#-----------------------------------------------------------------------------
# Root Project Setup
#-----------------------------------------------------------------------------

project(superdex_physics)

include("cmake/paths.cmake")

# Include build-wide settings that must be applied by the top level project.
# No-op if an outer project already included it.
include("cmake/root_project_setup.cmake")

#-----------------------------------------------------------------------------
# SuperDex Physics Settings
#-----------------------------------------------------------------------------

include("cmake/settings.cmake")

#-----------------------------------------------------------------------------
# Derived Settings
#-----------------------------------------------------------------------------

if (SUPERDEX_PHYSICS_BUILD_ALL)
    set(SUPERDEX_PHYSICS_BUILD_BENCHMARKS ON CACHE BOOL "" FORCE)
    set(SUPERDEX_PHYSICS_BUILD_TESTS ON CACHE BOOL "" FORCE)
    set(MOCHI_BUILD_ALL ON CACHE BOOL "" FORCE)
endif ()

if (SUPERDEX_PHYSICS_BUILD_BENCHMARKS)
    set(MOCHI_BUILD_BENCHMARKS ON CACHE BOOL "" FORCE)
endif ()

if (SUPERDEX_PHYSICS_BUILD_TESTS)
    set(MOCHI_BUILD_TESTS ON CACHE BOOL "" FORCE)
endif ()

#-----------------------------------------------------------------------------
# Mochi (physics simulation backend)
#-----------------------------------------------------------------------------

# Set path variables for mochi. It will configure imguios and simple_reflection.
set(MOCHI_IMGUIOS_SOURCE_DIR "${SUPERDEX_IMGUIOS_SOURCE_DIR}" CACHE PATH "Source directory for building imguios" FORCE)
set(MOCHI_SIMPLE_REFLECTION_SOURCE_DIR "${SUPERDEX_SIMPLE_REFLECTION_SOURCE_DIR}" CACHE PATH "Source directory for building simple_reflection" FORCE)

add_subdirectory("${SUPERDEX_MOCHI_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/mochi")

#-----------------------------------------------------------------------------
# superdex_physics (header-only interface wrapper for mochi_physics)
#-----------------------------------------------------------------------------

mochi_add_interface(superdex_physics)
target_include_directories(superdex_physics INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(superdex_physics INTERFACE mochi_physics)
