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
# Select Default Compiler
#-----------------------------------------------------------------------------

# Newest first, unversioned last: `apt install clang-17` installs only /usr/bin/clang-17, so probing
# the bare name alone would miss every distribution package.
set(_mochi_clang_suffixes "-30" "-29" "-28" "-27" "-26" "-25" "-24" "-23" "-22" "-21" "-20" "-19" "-18" "-17" "")

# Leave toolchain and cross builds, and any explicit compiler choice, to their own configuration.
if (NOT CMAKE_TOOLCHAIN_FILE AND NOT CMAKE_SYSTEM_NAME AND
    NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED CMAKE_CXX_COMPILER AND
    NOT DEFINED ENV{CC} AND NOT DEFINED ENV{CXX})
    if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        foreach (_mochi_suffix IN LISTS _mochi_clang_suffixes)
            unset(_mochi_clang)
            unset(_mochi_clangxx)
            find_program(_mochi_clang "clang${_mochi_suffix}" NO_CACHE)
            find_program(_mochi_clangxx "clang++${_mochi_suffix}" NO_CACHE)
            if (_mochi_clang AND _mochi_clangxx)
                break ()
            endif ()
        endforeach ()
        if (_mochi_clang AND _mochi_clangxx)
            set(CMAKE_C_COMPILER "${_mochi_clang}" CACHE FILEPATH "C compiler")
            set(CMAKE_CXX_COMPILER "${_mochi_clangxx}" CACHE FILEPATH "C++ compiler")
            message(STATUS "[Mochi] Selecting Clang C/C++ compiler: ${_mochi_clang}, ${_mochi_clangxx}")
        endif ()
        unset(_mochi_clang)
        unset(_mochi_clangxx)
        unset(_mochi_suffix)
    elseif (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows" AND NOT CMAKE_GENERATOR MATCHES "^Visual Studio")
        find_program(_mochi_clang_cl clang-cl NO_CACHE)
        if (_mochi_clang_cl)
            set(CMAKE_C_COMPILER "${_mochi_clang_cl}" CACHE FILEPATH "C compiler")
            set(CMAKE_CXX_COMPILER "${_mochi_clang_cl}" CACHE FILEPATH "C++ compiler")
            message(STATUS "[Mochi] Selecting ClangCL C/C++ compiler: ${_mochi_clang_cl}")
        endif ()
        unset(_mochi_clang_cl)
    endif ()
endif ()

unset(_mochi_clang_suffixes)

#-----------------------------------------------------------------------------
# Check Compiler Support
#-----------------------------------------------------------------------------

option(MOCHI_SKIP_COMPILER_CHECK "Skip the compiler check to allow a different C/C++ compiler (use at your own risk)" OFF)

set(_mochi_bypass "Other compilers are not regularly tested. They may fail or degrade performance.\nPass `-DMOCHI_SKIP_COMPILER_CHECK=ON` to bypass this check (use at your own risk).")

set(_mochi_stale "If this build directory previously used another compiler, delete its CMakeCache.txt and configure again.")

# Minimum MSVC toolset, as major.minor or major.minor.build. 14.30 is the first Visual Studio 2022
# toolset (platform toolset v143); Visual Studio 2019 ends at 14.29. Raise the minor or build to
# require a specific VS2022 servicing update.
set(_mochi_msvc_toolset_min "14.30")

# AppleClang 21 ships with Xcode 26; its version series is independent of upstream Clang.
set(_mochi_apple_clang_min "21")

function (_mochi_check_language_compiler lang)
    set(_id "${CMAKE_${lang}_COMPILER_ID}")
    set(_version "${CMAKE_${lang}_COMPILER_VERSION}")
    set(_detected "Detected ${lang} compiler: ${_id} ${_version} (${CMAKE_${lang}_COMPILER})")

    if (APPLE)
        if (NOT ((_id STREQUAL "AppleClang" AND NOT _version VERSION_LESS ${_mochi_apple_clang_min}) OR
                 (_id STREQUAL "Clang" AND NOT _version VERSION_LESS 17)))
            message(FATAL_ERROR
                "[Mochi] Supported ${lang} compilers: AppleClang ${_mochi_apple_clang_min} or newer, or upstream Clang 17 or newer.\n"
                "${_detected}\n"
                "Install or update Apple Clang with: xcode-select --install\n"
                "Or install upstream Clang with: brew install llvm\n"
                "explicitly: CC=$(brew --prefix llvm)/bin/clang CXX=$(brew --prefix llvm)/bin/clang++ cmake ...\n"
                "${_mochi_stale}\n"
                "${_mochi_bypass}")
        endif ()
    elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if (NOT _id STREQUAL "Clang" OR _version VERSION_LESS 17)
            message(FATAL_ERROR
                "[Mochi] Supported ${lang} compiler: Clang 17 or newer.\n"
                "${_detected}\n"
                "Install Clang 17 or newer with your package manager:\n"
                "  Ubuntu/Debian: sudo apt install clang-17\n"
                "  Fedora: sudo dnf install clang\n"
                "  Arch Linux: sudo pacman -S clang\n"
                "Clang is then selected automatically when it is on PATH.\n"
                "${_mochi_stale}\n"
                "${_mochi_bypass}")
        endif ()
    elseif (WIN32)
        # Only the compiler is checked here. CMAKE_<LANG>_SIMULATE_VERSION is clang's
        # `-fms-compatibility-version`, which clang infers from whatever MSVC installation it happens to
        # discover and replaces with a hardcoded default when it finds none, so it cannot be used to
        # require MSVC 2022. The toolset is checked separately by _mochi_check_msvc_toolset().
        #
        # FRONTEND_VARIANT is what separates clang-cl.exe from a clang.exe targeting the MSVC ABI. Both
        # report Clang and simulate MSVC, so neither the id nor SIMULATE_ID tells them apart, but only
        # clang-cl accepts MSVC-style flags -- and this build's dependencies pass plenty. HDF5 probes its
        # entire configuration with `/D...` in CMAKE_REQUIRED_FLAGS; under a GNU frontend clang reads
        # those as input filenames, so every probe fails as a missing file and HDF5 generates an
        # H5pubconf.h claiming the platform has no <sys/types.h> and no sizeof(int). Nothing reports it
        # until a compile reaches `off_t`, hundreds of targets later.
        if (NOT _id STREQUAL "Clang" OR _version VERSION_LESS 17 OR
            NOT CMAKE_${lang}_SIMULATE_ID STREQUAL "MSVC" OR
            NOT CMAKE_${lang}_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
            message(FATAL_ERROR
                "[Mochi] Supported ${lang} compiler: clang-cl 17 or newer in MSVC-compatible mode.\n"
                "${_detected}\n"
                "Command-line frontend: ${CMAKE_${lang}_COMPILER_FRONTEND_VARIANT} (must be MSVC). clang.exe and\n"
                "clang-cl.exe are the same compiler behind different drivers, and only clang-cl.exe accepts the\n"
                "MSVC-style flags this build and its dependencies pass.\n"
                "Install the required command-line build tools with:\n"
                "  winget install Microsoft.VisualStudio.2022.BuildTools --override \"--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Llvm.Clang\"\n"
                "For Visual Studio generators, add argument \"-T ClangCL\" or set environment variable CMAKE_GENERATOR_TOOLSET=ClangCL before configuring.\n"
                "For other generators, ensure clang-cl.exe is on PATH. It will be selected automatically -- but note\n"
                "that setting CC, CXX, CMAKE_C_COMPILER or CMAKE_CXX_COMPILER turns that selection off, and an\n"
                "explicit `clang` is then used exactly as given.\n"
                "${_mochi_stale}\n"
                "${_mochi_bypass}")
        endif ()
    else ()
        message(FATAL_ERROR "[Mochi] Unknown platform: ${CMAKE_SYSTEM_NAME}")
    endif ()
endfunction ()

# clang-cl compiles against the Visual Studio C++ headers and libraries, so a supported clang-cl can
# still be paired with an unsupported Visual Studio. The compiler cannot report which one: _MSC_VER,
# and hence CMAKE_<LANG>_SIMULATE_VERSION, comes from `-fms-compatibility-version`, which clang infers
# from whatever MSVC installation it discovers (VCToolsInstallDir, VCINSTALLDIR, a PATH scan, vswhere,
# the registry) and replaces with a hardcoded default when it finds none. <crtversion.h> is
# authoritative: it ships with the VC toolset and not with the Windows SDK, so it reports the version
# of the headers this build will actually use.
function (_mochi_check_msvc_toolset)
    if (NOT DEFINED MOCHI_MSVC_TOOLSET_VERSION)
        try_compile(_compiled
            SOURCE_FROM_CONTENT mochi_msvc_toolset_probe.cpp
[[#include <crtversion.h>
#define MOCHI_STR2(x) #x
#define MOCHI_STR(x) MOCHI_STR2(x)
#pragma message("MOCHI_MSVC_TOOLSET=" MOCHI_STR(_VC_CRT_MAJOR_VERSION) "." MOCHI_STR(_VC_CRT_MINOR_VERSION) "." MOCHI_STR(_VC_CRT_BUILD_VERSION))
int main() { return 0; }
]]
            OUTPUT_VARIABLE _output)

        if (NOT _compiled OR NOT _output MATCHES "MOCHI_MSVC_TOOLSET=([0-9]+\\.[0-9]+\\.[0-9]+)")
            message(FATAL_ERROR
                "[Mochi] Could not determine the MSVC toolset version.\n"
                "A probe including <crtversion.h> did not compile, or did not report a version. That header "
                "ships with the Visual Studio C++ toolset, so this usually means no MSVC toolset is on the "
                "include path.\n"
                "INCLUDE=$ENV{INCLUDE}\n"
                "${_mochi_stale}\n"
                "${_mochi_bypass}")
        endif ()

        set(MOCHI_MSVC_TOOLSET_VERSION "${CMAKE_MATCH_1}" CACHE INTERNAL "Detected MSVC toolset version")
        message(STATUS "[Mochi] MSVC toolset: ${MOCHI_MSVC_TOOLSET_VERSION}")
    endif ()

    if ("${MOCHI_MSVC_TOOLSET_VERSION}" VERSION_LESS "${_mochi_msvc_toolset_min}")
        message(FATAL_ERROR
            "[Mochi] Supported MSVC toolset: ${_mochi_msvc_toolset_min} or newer (Visual Studio 2022).\n"
            "Detected MSVC toolset: ${MOCHI_MSVC_TOOLSET_VERSION}\n"
            "Install the required command-line build tools with:\n"
            "  winget install Microsoft.VisualStudio.2022.BuildTools --override \"--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Llvm.Clang\"\n"
            "${_mochi_stale}\n"
            "${_mochi_bypass}")
    endif ()
endfunction ()

function (_mochi_check_compiler_support)
    if (MOCHI_SKIP_COMPILER_CHECK)
        return ()
    endif ()

    get_property(_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
    if (NOT CMAKE_C_COMPILER_LOADED OR NOT CMAKE_CXX_COMPILER_LOADED)
        message(FATAL_ERROR
            "_mochi_check_compiler_support() must be called after a project() that enables both C and C++.\n"
            "Enabled languages: ${_enabled_languages}")
    endif ()

    _mochi_check_language_compiler(C)
    _mochi_check_language_compiler(CXX)

    # Apple builds compile .m/.mm sources, for which CMake selects compilers independently of C/CXX.
    foreach (_objc_lang IN ITEMS OBJC OBJCXX)
        if ("${_objc_lang}" IN_LIST _enabled_languages)
            _mochi_check_language_compiler(${_objc_lang})
        endif ()
    endforeach ()

    if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "${CMAKE_CXX_COMPILER_ID}" OR
        NOT "${CMAKE_C_COMPILER_VERSION}" VERSION_EQUAL "${CMAKE_CXX_COMPILER_VERSION}")
        message(FATAL_ERROR
            "[Mochi] The C and C++ compilers must be the same compiler and version.\n"
            "Detected C compiler:   ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION} (${CMAKE_C_COMPILER})\n"
            "Detected C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} (${CMAKE_CXX_COMPILER})\n"
            "Set both CMAKE_C_COMPILER and CMAKE_CXX_COMPILER, or neither to let Mochi select a matching pair.\n"
            "${_mochi_bypass}")
    endif ()

    if (WIN32)
        _mochi_check_msvc_toolset()
    endif ()
endfunction ()

#-----------------------------------------------------------------------------
# Compiler Identification
#-----------------------------------------------------------------------------

macro (_mochi_define_compiler_variables)
    set(_mochi_compiler_clang FALSE)        # upstream Clang or AppleClang, under either driver
    set(_mochi_compiler_clang_cl FALSE)     # ClangCL used with MSVC
    set(_mochi_compiler_apple_clang FALSE)  # Xcode's Clang; its version series is its own
    set(_mochi_compiler_gcc FALSE)          # gcc
    set(_mochi_compiler_msvc FALSE)         # cl.exe

    if (CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
        set(_mochi_compiler_clang TRUE)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
            set(_mochi_compiler_apple_clang TRUE)
        endif ()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(_mochi_compiler_gcc TRUE)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(_mochi_compiler_msvc TRUE)
    endif ()

    if (_mochi_compiler_clang AND MSVC)
        set(_mochi_compiler_clang_cl TRUE)
    endif ()
endmacro ()
