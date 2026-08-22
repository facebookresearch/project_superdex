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

include_guard(GLOBAL)

#-----------------------------------------------------------------------------
# Link Time Optimizations (disabled by default)
#-----------------------------------------------------------------------------

if (NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)
endif ()

# Also set per-config defaults for multi-config generators (e.g. MSVC)
if (NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION_DEBUG)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_DEBUG OFF CACHE BOOL "" FORCE)
endif ()
if (NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF CACHE BOOL "" FORCE)
endif ()
if (NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO OFF CACHE BOOL "" FORCE)
endif ()
if (NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL OFF CACHE BOOL "" FORCE)
endif ()

#-----------------------------------------------------------------------------
# What library type to build:
#-----------------------------------------------------------------------------

option(MOCHI_BUILD_SHARED "Build mochi subprojects as shared/dynamic libraries (else static)" ON)

#-----------------------------------------------------------------------------
# What targets to build:
#-----------------------------------------------------------------------------

option(MOCHI_BUILD_BENCHMARKS "Build mochi benchmark applications" OFF)
option(MOCHI_BUILD_IMGUIOS "Build imguios (first-party dependency)" OFF)
option(MOCHI_BUILD_DEBUGGER "Build mochi debugger application" ON)
option(MOCHI_BUILD_TESTS "Build mochi unit test applications" OFF)
option(MOCHI_BUILD_RENDERER "Build mochi_renderer subproject (requires Filament)" OFF)
option(MOCHI_BUILD_MESH_LIB "Build the mochi_mesh library" OFF)
option(MOCHI_BUILD_MESH_CLI "Build the GPL-isolated superdex_mesh_cli helper executable" OFF)

option(MOCHI_BUILD_ALL "Build all mochi targets" OFF)

#-----------------------------------------------------------------------------
# Optional Features:
#-----------------------------------------------------------------------------

option(MOCHI_USE_AVX512 "Compile with AVX512 instructions. Requires a compatible CPU." OFF)
option(MOCHI_USE_CCACHE "Let third-party builds wrap the compiler in ccache if one is found" OFF)
option(MOCHI_USE_DOUBLE_PRECISION "Use 64-bit double-precision floating point" OFF)
option(MOCHI_USE_EIGEN "Compile with Eigen to enable some experimental features" ON)
option(MOCHI_USE_HDF5 "Compile with HDF5 to enable use of the H5 file format" ON)
option(MOCHI_USE_PYBIND "Compile with pybind11 to enable Python bindings for select subprojects" ON)
option(MOCHI_USE_TINYFILEDIALOG "Compile with tinyfiledialog for file dialog support in ImGui applications" OFF)
option(MOCHI_USE_TRACY "Compile with Tracy to enable profiling" OFF)
option(MOCHI_BUILD_TRACY_GUI "Build the tracy-profiler GUI alongside Mochi" OFF)
option(MOCHI_DOWNLOAD_TRACY_GUI "Download tracy-profiler GUI from GitHub" OFF)
option(MOCHI_WARN_IF_NOT_OPTIMIZED "Print a warning if compiler optimizations are disabled, to prevent mistakes" ON)

if (MOCHI_INTERNAL)
    message(FATAL_ERROR "[Mochi] Internal builds are not supported with CMake.")
endif ()
set(MOCHI_INTERNAL OFF CACHE BOOL "Only external builds are supported with CMake." FORCE)

#-----------------------------------------------------------------------------
# Compiler Settings
#-----------------------------------------------------------------------------

include(CheckCXXCompilerFlag)

set(MOCHI_COMPILE_OPTIONS)

# Settings shared by Clang and GCC (but not ClangCL)
if ((_mochi_compiler_clang AND NOT _mochi_compiler_clang_cl) OR _mochi_compiler_gcc)
    list(APPEND MOCHI_COMPILE_OPTIONS
        -ffunction-sections
        -fdata-sections
        -fstack-protector-strong
        -Wno-attributes
        $<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-fsized-deallocation>
        $<$<NOT:$<CONFIG:Debug>>:-fno-omit-frame-pointer>
        $<$<NOT:$<CONFIG:Debug>>:-mno-omit-leaf-frame-pointer>
    )

    # Floating-point
    list(APPEND MOCHI_COMPILE_OPTIONS
        $<$<CONFIG:Debug>:-ffp-contract=on>
        $<$<NOT:$<CONFIG:Debug>>:-ffp-contract=fast>

        "$<$<CONFIG:Debug>:-ftrapping-math>"
        "$<$<NOT:$<CONFIG:Debug>>:-fno-trapping-math>"
    )

    # Architecture flags
    if (MOCHI_ARCH_X64)
        list(APPEND MOCHI_COMPILE_OPTIONS
            -mbmi2
            -mlzcnt
            -mpclmul
            -mfma
            -mf16c)
        if (MOCHI_USE_AVX512)
            list(APPEND MOCHI_COMPILE_OPTIONS -mavx512f)
        else ()
            list(APPEND MOCHI_COMPILE_OPTIONS -mavx2)
        endif ()
    endif ()

    if (MOCHI_BUILD_SHARED)
        list(APPEND MOCHI_COMPILE_OPTIONS
            "$<$<COMPILE_LANGUAGE:C,CXX,OBJC,OBJCXX>:-fPIC>"
        )
    endif ()

    if (NOT APPLE)
        list(APPEND MOCHI_COMPILE_OPTIONS -fdebug-types-section)
    endif ()

    # If Interprocedural Optimization (i.e. link time optimization) is enabled, then use parallel LTRANS jobs
    # with an automatic number of threads. This applies to ALL targets within the "mochi" directory, including
    # third-party libraries.
    set(CMAKE_C_COMPILE_OPTIONS_IPO -flto=auto)
    set(CMAKE_CXX_COMPILE_OPTIONS_IPO -flto=auto)
    if (APPLE)
        set(CMAKE_OBJC_COMPILE_OPTIONS_IPO -flto=auto)
        set(CMAKE_OBJCXX_COMPILE_OPTIONS_IPO -flto=auto)
    endif ()
endif ()

# Settings unique to Clang (not GCC, not ClangCL)
if (_mochi_compiler_clang AND NOT _mochi_compiler_clang_cl)
    list(APPEND MOCHI_COMPILE_OPTIONS -ftrivial-auto-var-init=zero)

    # Mochi uses C++ designated initializers with optional fields (unspecified fields use defaults).
    # Clang versions that support -Wmissing-designated-field-initializers warn on this pattern, which
    # -Werror promotes to a build error. Probe before adding the C++-only -Wno-* flag because older
    # Clang versions may not support it.
    check_cxx_compiler_flag(
        "-Wno-missing-designated-field-initializers"
        MOCHI_HAS_WNO_MISSING_DESIGNATED_FIELD_INITIALIZERS)
    if (MOCHI_HAS_WNO_MISSING_DESIGNATED_FIELD_INITIALIZERS)
        list(APPEND MOCHI_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:-Wno-missing-designated-field-initializers>")
    endif ()
endif ()

# Settings shared by MSVC and ClangCL
if (_mochi_compiler_msvc OR _mochi_compiler_clang_cl)
    list(APPEND MOCHI_COMPILE_OPTIONS
        "/bigobj"
        "/GS"
        "/Gy"
        "/Gw"
        "$<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>"

        # NOTE: Unreal Engine builds require "/Zp8", but do not enable it globally here. Mixing code built
        # with different packing settings can cause ABI mismatches and undefined behavior. Supporting this
        # requires carefully controlling third-party library settings and further investigation.
    )

    if (MOCHI_ARCH_X64)
        if (MOCHI_USE_AVX512)
            list(APPEND MOCHI_COMPILE_OPTIONS /arch:AVX512)
        else ()
            list(APPEND MOCHI_COMPILE_OPTIONS /arch:AVX2)
        endif ()
    endif ()
endif ()

# Settings unique to ClangCL
if (_mochi_compiler_clang_cl)
    list(APPEND MOCHI_COMPILE_OPTIONS
        "/clang:-ftrivial-auto-var-init=zero"
        "$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:/clang:-fsized-deallocation>"
    )

    # The MSVC CRT marks getenv/strncpy/etc. deprecated. The arvr Buck toolchain turns this
    # category off outright (and additionally defines _CRT_SECURE_NO_WARNINGS), so match it.
    list(APPEND MOCHI_COMPILE_OPTIONS
        "/clang:-Wno-deprecated-declarations"
    )

    # MSVC's /fp: switches do not map onto clang's floating-point model: clang-cl's /fp:fast
    # implies -ffast-math, hence -ffinite-math-only, which makes +/-infinity undefined
    # behavior (what -Wnan-infinity-disabled reports). Mochi requires working infinities, and
    # Buck's opt mode only enables FMA contraction, so forward the individual clang options.
    list(APPEND MOCHI_COMPILE_OPTIONS
        "$<$<CONFIG:Debug>:/clang:-ffp-contract=on>"
        "$<$<NOT:$<CONFIG:Debug>>:/clang:-ffp-contract=fast>"

        "$<$<CONFIG:Debug>:/clang:-ftrapping-math>"
        "$<$<NOT:$<CONFIG:Debug>>:/clang:-fno-trapping-math>"
    )
endif ()

# Settings unique to MSVC (not ClangCL)
if (_mochi_compiler_msvc)
    list(APPEND MOCHI_COMPILE_OPTIONS
        "$<$<CONFIG:Debug>:/fp:precise>"
        "$<$<NOT:$<CONFIG:Debug>>:/fp:fast>"

        "$<$<CONFIG:Debug>:/fp:except>"
        "$<$<NOT:$<CONFIG:Debug>>:/fp:except->"
    )
endif ()
