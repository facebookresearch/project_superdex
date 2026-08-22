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

# Select default compiler. Must come before "project(superdex)"
include("${CMAKE_CURRENT_LIST_DIR}/superdex_physics/cmake/compiler.cmake")

#-----------------------------------------------------------------------------
# Root Project Setup
#-----------------------------------------------------------------------------

project(superdex)

# Include build-wide settings that must be applied by the top level project.
include("superdex_physics/cmake/root_project_setup.cmake")

#-----------------------------------------------------------------------------
# SuperDex Settings
#-----------------------------------------------------------------------------

include("cmake/settings.cmake")

#-----------------------------------------------------------------------------
# Derived Settings
#-----------------------------------------------------------------------------

if (MOCHI_INTERNAL)
    message(FATAL_ERROR "[Superdex] Internal builds are not supported with CMake.")
endif ()
set(MOCHI_INTERNAL OFF CACHE BOOL "Only external builds are supported with CMake." FORCE)

if (SUPERDEX_BUILD_ALL)
    set(SUPERDEX_BUILD_ROBOTICS ON CACHE BOOL "" FORCE)
    set(SUPERDEX_BUILD_STUDIO ON CACHE BOOL "" FORCE)
    set(SUPERDEX_PHYSICS_BUILD_ALL ON CACHE BOOL "" FORCE)
endif ()

if (SUPERDEX_BUILD_STUDIO)
    # Needed to build the studio: it links all four.
    set(SUPERDEX_BUILD_ROBOTICS ON CACHE BOOL "" FORCE)
    set(MOCHI_BUILD_RENDERER ON CACHE BOOL "" FORCE)
    set(MOCHI_BUILD_MESH_LIB ON CACHE BOOL "" FORCE)
    set(MOCHI_BUILD_IMGUIOS ON CACHE BOOL "" FORCE)

    # Needed to use the studio, not to build it. On by default, but no FORCE: the wheels ship the
    # mesh helper and the debugger as separate distributions and have to be able to decline them.
    #
    # These run before Mochi's own `option()` calls, so they create the cache entries unless `-D`
    # already did, which is what makes the override work.
    set(MOCHI_BUILD_DEBUGGER ON CACHE BOOL "")
    set(MOCHI_BUILD_MESH_CLI ON CACHE BOOL "")

    # superdex_studio uses tinyfiledialog directly.
    set(MOCHI_USE_TINYFILEDIALOG ON CACHE BOOL "" FORCE)
endif ()

#-----------------------------------------------------------------------------
# Testing
#-----------------------------------------------------------------------------

# CTest only writes a testfile at the build root when testing is enabled here,
# in the top-level source directory. Mochi enabling it for its own subtree is
# not enough: without this, `ctest` from a Superdex build root finds nothing.
if (MOCHI_BUILD_TESTS)
    enable_testing()
endif ()

#-----------------------------------------------------------------------------
# Superdex Projects
#-----------------------------------------------------------------------------

# Always add superdex_physics
add_subdirectory(superdex_physics)

if (SUPERDEX_BUILD_ROBOTICS)
    add_subdirectory(superdex_robotics)
endif ()

if (SUPERDEX_BUILD_STUDIO)
    add_subdirectory(superdex_studio)
endif ()
