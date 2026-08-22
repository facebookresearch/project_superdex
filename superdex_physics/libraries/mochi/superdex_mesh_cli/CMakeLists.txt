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

# CGAL and Boost are header-only for our usage.
set(_mochi_cgal_dir "${MOCHI_CGAL_SOURCE_DIR}")
set(_mochi_boost_include_dir "${MOCHI_BOOST_SOURCE_DIR}")

if (NOT MOCHI_CGAL_FIND_PACKAGE AND NOT TARGET CGAL::CGAL)
    if (NOT EXISTS "${_mochi_cgal_dir}")
        message(STATUS "[Mochi] Fetching CGAL 6.1.1 headers via FetchContent")
        include(FetchContent)
        FetchContent_Declare(
            cgal
            URL "https://github.com/CGAL/cgal/releases/download/v6.1.1/CGAL-6.1.1-library.tar.xz"
            URL_HASH SHA256=37e9fffe48a83209b070e1914c6aa0a7bae8076749712ab78b53245e176e0e0e
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            # Header-only usage: extract but do not add CGAL's own CMake project.
            SOURCE_SUBDIR "do-not-configure"
        )
        FetchContent_MakeAvailable(cgal)
        set(_mochi_cgal_dir "${cgal_SOURCE_DIR}")
    endif ()
    if (NOT EXISTS "${_mochi_boost_include_dir}")
        message(STATUS "[Mochi] Fetching Boost 1.77.0 headers via FetchContent")
        include(FetchContent)
        FetchContent_Declare(
            boost
            URL "https://archives.boost.io/release/1.77.0/source/boost_1_77_0.tar.gz"
            URL_HASH SHA256=5347464af5b14ac54bb945dc68f1dd7c56f0dad7262816b956138fc53bcc0131
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            SOURCE_SUBDIR "do-not-configure"
        )
        FetchContent_MakeAvailable(boost)
        set(_mochi_boost_include_dir "${boost_SOURCE_DIR}")
    endif ()
endif ()

if (MOCHI_CGAL_FIND_PACKAGE)
    find_package(CGAL REQUIRED)
elseif (NOT TARGET CGAL::CGAL)
    file(GLOB CGAL_PACKAGE_DIRS "${_mochi_cgal_dir}/CGAL/*/include")
    if (EXISTS "${_mochi_cgal_dir}/include/CGAL")
        list(APPEND CGAL_PACKAGE_DIRS "${_mochi_cgal_dir}/include")
    endif ()
    if (NOT CGAL_PACKAGE_DIRS)
        message(FATAL_ERROR "[Mochi] CGAL headers not found at: ${_mochi_cgal_dir}")
    endif ()
    message(STATUS "[Mochi] Including CGAL (header-only) from: ${_mochi_cgal_dir}")
    if (NOT TARGET Boost::headers)
        add_library(boost_headers INTERFACE)
        add_library(Boost::headers ALIAS boost_headers)
        target_include_directories(boost_headers SYSTEM INTERFACE "${_mochi_boost_include_dir}")
    endif ()
    add_library(CGAL INTERFACE)
    add_library(CGAL::CGAL ALIAS CGAL)
    target_include_directories(CGAL SYSTEM INTERFACE ${CGAL_PACKAGE_DIRS})
    target_compile_definitions(CGAL INTERFACE
        CGAL_DISABLE_GMP=1
        CGAL_NO_GMP=1
        CGAL_NO_MPFR=1
        CGAL_CORE_USE_BOOST_BACKEND=1
        # The vendored Boost's container_hash uses std::unary_function, removed in C++17. Newer libc++
        # drops the symbol entirely, so tell Boost to use its own hash_base instead.
        BOOST_NO_CXX98_FUNCTION_BASE=1)
    target_link_libraries(CGAL INTERFACE Boost::headers)
endif ()

# ---------------------------------------------------------------------------
# RapidJSON -- header-only, required by OCCT's glTF writer (RWGltf_CafWriter).
# Resolved from MOCHI_RAPIDJSON_SOURCE_DIR, or downloaded when absent.
# ---------------------------------------------------------------------------
set(MOCHI_RAPIDJSON_INCLUDE_DIR "${MOCHI_RAPIDJSON_SOURCE_DIR}")

if (NOT EXISTS "${MOCHI_RAPIDJSON_INCLUDE_DIR}")
    include(FetchContent)
    message(STATUS "[Mochi] Fetching RapidJSON 1.1.0 headers via FetchContent")
    FetchContent_Declare(
        rapidjson
        URL "https://github.com/Tencent/rapidjson/archive/7484e06c589873e1ed80382d262087e4fa80fb63.tar.gz"
        URL_HASH SHA256=fc22de09b56c68bf4e0463e33352f0d7622eb9500ba93af453b7d2d66b5d6be9
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR "do-not-configure"
    )
    FetchContent_MakeAvailable(rapidjson)
    set(MOCHI_RAPIDJSON_INCLUDE_DIR "${rapidjson_SOURCE_DIR}/include")
endif ()

# ---------------------------------------------------------------------------
# OpenCASCADE (OCCT) -- STEP file reading + tessellation.
#
# OCCT's CMake assumes it is the top-level project (it resolves adm/cmake via
# CMAKE_SOURCE_DIR), so it cannot be add_subdirectory'd like the header-only deps
# above. Build it as an ExternalProject (its own CMake run, pinned tarball from
# GitHub) and consume the installed toolkits via imported targets.
# ---------------------------------------------------------------------------
option(MOCHI_MESH_CLI_WITH_OCCT "Build superdex_mesh_cli with OpenCASCADE (OCCT) STEP support" ON)

if (MOCHI_MESH_CLI_WITH_OCCT AND NOT TARGET TKDESTEP)
    include(ExternalProject)

    # Keep the (expensive) OCCT build + download outside the main build tree so it
    # survives cache wipes; fall back to the build dir when unset.
    if (DEFINED FETCHCONTENT_BASE_DIR)
        set(_occt_base "${FETCHCONTENT_BASE_DIR}/occt-ep")
    else ()
        set(_occt_base "${CMAKE_BINARY_DIR}/occt-ep")
    endif ()
    set(_occt_install "${_occt_base}/install")
    set(MOCHI_OCCT_INCLUDE_DIR "${_occt_install}/include/opencascade")

    # Static linking needs the full toolkit dependency closure of the STEP +
    # meshing path on the link line.
    set(MOCHI_OCCT_LIBRARIES
        TKDESTEP TKDE TKXSBase TKMesh TKBO TKShHealing TKTopAlgo TKPrim TKGeomAlgo
        TKBRep TKGeomBase TKG3d TKG2d TKMath TKernel
        TKXCAF TKVCAF TKV3d TKService TKCAF TKLCAF TKCDF TKHLR
        TKRWMesh TKDEGLTF TKDEOBJ TKDESTL)

    # Static import libraries the link step consumes -- declared as ExternalProject
    # byproducts so Ninja tracks them.
    set(_occt_byproducts "")
    foreach (_tk IN LISTS MOCHI_OCCT_LIBRARIES)
        list(APPEND _occt_byproducts
            "${_occt_install}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${_tk}${CMAKE_STATIC_LIBRARY_SUFFIX}")
    endforeach ()

    set(_occt_patch_script "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_occt.cmake")
    set(_occt_touch_clang_format OFF)
    if (MOCHI_OCCT_SOURCE_DIR AND EXISTS "${MOCHI_OCCT_SOURCE_DIR}/CMakeLists.txt")
        message(STATUS "[Mochi] Building OpenCASCADE (OCCT) from: ${MOCHI_OCCT_SOURCE_DIR}")
        set(_occt_staged "${_occt_base}/occt-src")
        set(_occt_touch_clang_format ON)
        set(_occt_src_args
            SOURCE_DIR "${_occt_staged}"
            DOWNLOAD_COMMAND "${CMAKE_COMMAND}" -E copy_directory "${MOCHI_OCCT_SOURCE_DIR}" "${_occt_staged}")
    else ()
        message(STATUS "[Mochi] Fetching + building OpenCASCADE (OCCT) 7.9.3 via ExternalProject")
        set(_occt_src_args
            URL "https://github.com/Open-Cascade-SAS/OCCT/archive/refs/tags/V7_9_3.tar.gz"
            URL_HASH SHA256=5ecf094ec6b12d5413dfb851d8c3590c354058aee556e32e408bdfbf8c357d57
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    endif ()

    # Forward this build's toolchain / target-platform settings to the OCCT sub-build.
    set(_occt_toolchain_args "")
    if (CMAKE_TOOLCHAIN_FILE)
        list(APPEND _occt_toolchain_args "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")
    elseif (NOT CMAKE_GENERATOR MATCHES "Visual Studio")
        # The Visual Studio generators select the compiler via -T/-A (not CMAKE_CXX_COMPILER), where
        # forwarding a compiler path conflicts; only forward it for single-compiler generators
        # (Ninja / Makefiles, which CI uses).
        if (CMAKE_C_COMPILER)
            list(APPEND _occt_toolchain_args "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
        endif ()
        if (CMAKE_CXX_COMPILER)
            list(APPEND _occt_toolchain_args "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
        endif ()
    endif ()
    if (APPLE)
        if (CMAKE_OSX_ARCHITECTURES)
            list(APPEND _occt_toolchain_args "-DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}")
        endif ()
        if (CMAKE_OSX_DEPLOYMENT_TARGET)
            list(APPEND _occt_toolchain_args "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
        endif ()
        if (CMAKE_OSX_SYSROOT)
            list(APPEND _occt_toolchain_args "-DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}")
        endif ()
    endif ()

    # OCCT is compiled third-party code. Keep its diagnostics out of first-party
    # warning-clean builds without weakening warning flags on Mochi targets.
    #
    # MSVC (including clang-cl) is deliberately excluded: OCCT's occt_defs_flags.cmake
    # unconditionally forces /W4, rewriting any /W[0-4] already present in CMAKE_CXX_FLAGS and
    # appending /W4 otherwise, so nothing passed via CMAKE_<LANG>_FLAGS_INIT survives. Passing
    # a lowercase /w there would only add a per-TU "D9025: overriding '/w' with '/W4'". MSVC
    # consumers are still covered by the INTERFACE_SYSTEM_INCLUDE_DIRECTORIES set on the
    # imported targets below. -w on gcc/clang is order-independent, so it does win there.
    set(_occt_warning_args "")
    if (NOT MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(_occt_warning_args
            "-DCMAKE_C_FLAGS_INIT=-w"
            "-DCMAKE_CXX_FLAGS_INIT=-w")
    endif ()

    include(ProcessorCount)
    ProcessorCount(_occt_build_jobs)
    if (NOT _occt_build_jobs)
        set(_occt_build_jobs 1)
    endif ()

    ExternalProject_Add(
        occt_external
        ${_occt_src_args}
        PREFIX "${_occt_base}"
        INSTALL_DIR "${_occt_install}"
        PATCH_COMMAND
            "${CMAKE_COMMAND}"
            -DOCCT_SOURCE_DIR=<SOURCE_DIR>
            -DMOCHI_OCCT_TOUCH_CLANG_FORMAT=${_occt_touch_clang_format}
            -P "${_occt_patch_script}"
        CMAKE_ARGS
            # OCCT is always built Release regardless of the parent config.
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_SUPPRESS_DEVELOPER_WARNINGS=ON
            # Every object in one binary must agree on the MSVC runtime library or the link fails
            # with LNK2038.
            -DCMAKE_POLICY_DEFAULT_CMP0091=NEW
            -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL
            ${_occt_toolchain_args}
            ${_occt_warning_args}
            -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
            -DCMAKE_INSTALL_PREFIX=${_occt_install}
            -DINSTALL_DIR_LAYOUT=Unix
            -DINSTALL_DIR_WITH_VERSION=OFF
            -DINSTALL_DIR_INCLUDE=include/opencascade
            -DINSTALL_DIR_LIB=lib
            -DINSTALL_DIR_BIN=bin
            -DBUILD_LIBRARY_TYPE=Static
            # OCCT ships a precompiled header per toolkit but defaults them off; they roughly
            # halve the sub-build's compile time.
            -DBUILD_USE_PCH=ON
            -DBUILD_MODULE_FoundationClasses=OFF
            -DBUILD_MODULE_ModelingData=OFF
            -DBUILD_MODULE_ModelingAlgorithms=OFF
            -DBUILD_MODULE_DataExchange=OFF
            -DBUILD_MODULE_Visualization=OFF
            -DBUILD_MODULE_ApplicationFramework=OFF
            -DBUILD_MODULE_DETools=OFF
            -DBUILD_MODULE_Draw=OFF
            "-DBUILD_ADDITIONAL_TOOLKITS=TKDESTEP TKMesh TKBO TKShHealing TKTopAlgo TKGeomBase TKG3d TKBRep TKMath TKernel TKXCAF TKLCAF TKCAF TKRWMesh TKDEGLTF TKDEOBJ TKDESTL"
            -DBUILD_DOC_Overview=OFF
            -DINSTALL_TEST_CASES=OFF
            -DUSE_FREETYPE=OFF
            -DUSE_TK=OFF
            -DUSE_TCL=OFF
            -DUSE_TBB=OFF
            -DUSE_VTK=OFF
            -DUSE_RAPIDJSON=ON
            "-D3RDPARTY_RAPIDJSON_INCLUDE_DIR=${MOCHI_RAPIDJSON_INCLUDE_DIR}"
            -DUSE_DRACO=OFF
            -DUSE_FREEIMAGE=OFF
            -DUSE_FFMPEG=OFF
            -DUSE_OPENVR=OFF
            -DUSE_XLIB=OFF
        BUILD_COMMAND
            "${CMAKE_COMMAND}" -E env --unset=MAKEFLAGS
            "${CMAKE_COMMAND}" --build <BINARY_DIR> --config Release --parallel "${_occt_build_jobs}"
        INSTALL_COMMAND "${CMAKE_COMMAND}" --install <BINARY_DIR> --config Release
        # This allows OCCT build logs to appear during compilation
        USES_TERMINAL_BUILD TRUE
        BUILD_BYPRODUCTS ${_occt_byproducts}
    )

    # Imported targets' INTERFACE include dir must exist at generate time.
    file(MAKE_DIRECTORY "${MOCHI_OCCT_INCLUDE_DIR}")

    foreach (_tk IN LISTS MOCHI_OCCT_LIBRARIES)
        add_library(${_tk} STATIC IMPORTED GLOBAL)
        set_target_properties(${_tk} PROPERTIES
            IMPORTED_LOCATION "${_occt_install}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${_tk}${CMAKE_STATIC_LIBRARY_SUFFIX}"
            INTERFACE_INCLUDE_DIRECTORIES "${MOCHI_OCCT_INCLUDE_DIR}"
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${MOCHI_OCCT_INCLUDE_DIR}")
        add_dependencies(${_tk} occt_external)
    endforeach ()

    # Platform libraries the OCCT toolkits call into
    find_package(Threads REQUIRED)
    if (WIN32)
        set(MOCHI_OCCT_SYSTEM_LIBRARIES advapi32 gdi32 user32 shell32 wsock32 psapi winmm)
    elseif (APPLE)
        set(MOCHI_OCCT_SYSTEM_LIBRARIES "-framework AppKit" "-framework IOKit" objc)
    else ()
        set(MOCHI_OCCT_SYSTEM_LIBRARIES ${CMAKE_DL_LIBS})
    endif ()
    list(APPEND MOCHI_OCCT_SYSTEM_LIBRARIES Threads::Threads)
endif ()

add_subdirectory(superdex_mesh_cli)
