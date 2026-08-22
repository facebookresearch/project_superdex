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

# Include this from the top level CMake project to ensure that global settings
# apply correctly to all recursive projects.

include_guard(DIRECTORY)

#-----------------------------------------------------------------------------
# Default build configuration
#-----------------------------------------------------------------------------

# Multi-config generators select the configuration when building.
if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    message(STATUS "[Mochi] Setting CMAKE_BUILD_TYPE=Release by default")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif ()

#-----------------------------------------------------------------------------
# Language Standard
#-----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#-----------------------------------------------------------------------------
# Apple: Objective-C / Objective-C++
#-----------------------------------------------------------------------------

# Enable compilers for .m and .mm files. This must precede the compiler check, which
# validates every enabled language.
if (APPLE)
    enable_language(OBJC)
    enable_language(OBJCXX)
endif ()

#-----------------------------------------------------------------------------
# Check Compiler Support
#-----------------------------------------------------------------------------

include("${CMAKE_CURRENT_LIST_DIR}/compiler.cmake")
_mochi_check_compiler_support()
_mochi_define_compiler_variables()

#-----------------------------------------------------------------------------
# Detect Target Features
#-----------------------------------------------------------------------------

# On Windows with Ninja generator, CMAKE_SYSTEM_PROCESSOR may be empty.
if (WIN32 AND (NOT CMAKE_SYSTEM_PROCESSOR OR CMAKE_SYSTEM_PROCESSOR STREQUAL ""))
    set(CMAKE_SYSTEM_PROCESSOR "AMD64")
    message(STATUS "[Mochi] CMAKE_SYSTEM_PROCESSOR was empty. Defaulting to: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64|armv|aarch64")
    message(STATUS "[Mochi] Target Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    set(MOCHI_ARCH_ARM true)
    set(MOCHI_ARCH_X64 false)
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64|amd64|AMD64")
    message(STATUS "[Mochi] Target Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    set(MOCHI_ARCH_ARM false)
    set(MOCHI_ARCH_X64 true)
else ()
    message(FATAL_ERROR "[Mochi] UNKNOWN ARCHITECTURE: ${CMAKE_SYSTEM_PROCESSOR}")
endif ()

#-----------------------------------------------------------------------------
# Apple: Objective-C / Objective-C++
#-----------------------------------------------------------------------------

# Several subtrees compile Objective-C / Objective-C++ sources without enabling
# those languages themselves: GLFW (via imguios) has Cocoa .m files, imgui_metal
# has imgui_impl_metal.mm, and superdex_studio has render_target_metal.mm.
# Filament enables OBJC/OBJCXX only in its own scope, so the compile rules
# (CMAKE_OBJC_COMPILE_OBJECT / CMAKE_OBJCXX_COMPILE_OBJECT) are missing
# elsewhere. Enable both at the root so every subdirectory inherits them.
if (APPLE)
    enable_language(OBJC)
    enable_language(OBJCXX)

    set(CMAKE_OBJCXX_STANDARD 20)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif ()

#-----------------------------------------------------------------------------
# MSVC: global compiler options (including third-party libraries built from source)
#-----------------------------------------------------------------------------

if (MSVC)
    # /MP only helps the Visual Studio generator, which hands many sources to a
    # single cl.exe. Under Ninja (used by CI) every source is already its own cl.exe
    # invocation, so /MP just oversubscribes the host: each of the N parallel jobs
    # spawns its own sub-compiler pool, spiking memory and temp/PDB pressure and
    # producing flaky failures such as "C1083: Cannot open compiler generated file".
    if (CMAKE_GENERATOR MATCHES "Visual Studio")
        # Multi-processor compilation. Guard to C/CXX so the flag is not forwarded
        # to the MASM assembler (ml64), which rejects cl-style options with
        # "A4018: invalid command-line option" (e.g. for Filament's bluegl .S file).
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/MP>)
    endif ()

    # _ITERATOR_DEBUG_LEVEL must match across every object linked together or MSVC fails with LNK2038.
    # Filament (third-party) builds with iterator debugging off: it forces _ITERATOR_DEBUG_LEVEL=0 in
    # Debug and inherits the MSVC default of 0 in optimized configs. Match that across all targets.
    # To enable iterator debugging instead, we would also have to modify Filament's CMake configuration.
    add_compile_definitions($<$<COMPILE_LANGUAGE:C,CXX>:_ITERATOR_DEBUG_LEVEL=0>)
endif ()
