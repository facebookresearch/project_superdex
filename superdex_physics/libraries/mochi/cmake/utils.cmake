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
# mochi_set_output_directories
#-----------------------------------------------------------------------------

function (mochi_set_output_directories target_name)
    set_target_properties(${target_name} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${target_name}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction ()

#-----------------------------------------------------------------------------
# mochi_set_sibling_library_rpath
#-----------------------------------------------------------------------------

# Set the RPATH of a target so it resolves sibling Mochi shared libraries from the directory
# the target itself lives in. All Mochi .so/.dylib files are co-located in ${CMAKE_BINARY_DIR}/bin
# (build tree) and installed next to each other (install tree), so a single self-relative RPATH
# covers both. Windows has no RPATH concept; DLLs are located via PATH / the executable's own
# directory, so this is a no-op there.
function (mochi_set_sibling_library_rpath target_name)
    if (APPLE)
        set_target_properties(${target_name} PROPERTIES
            BUILD_RPATH "@loader_path"
            INSTALL_RPATH "@loader_path"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    elseif (UNIX)
        set_target_properties(${target_name} PROPERTIES
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif ()
endfunction ()

#-----------------------------------------------------------------------------
# mochi_set_pybind_rpath
#-----------------------------------------------------------------------------

function (mochi_set_pybind_rpath target_name)
    mochi_set_sibling_library_rpath(${target_name})
endfunction ()

#-----------------------------------------------------------------------------
# mochi_set_gui_subsystem
#-----------------------------------------------------------------------------

# Pairs with mochi::AttachParentConsole() (mochi_core/utils/console.h) at the top of main(),
# because a GUI-subsystem process gets no console and discards stdout/stderr. No-op off Windows.
function (mochi_set_gui_subsystem target_name)
    set_target_properties(${target_name} PROPERTIES WIN32_EXECUTABLE TRUE)
    if (MSVC)
        # WIN32_EXECUTABLE links /SUBSYSTEM:WINDOWS, whose default entry point is WinMain. These
        # apps have an ordinary main(), so point the linker back at the CRT startup that calls it.
        target_link_options(${target_name} PRIVATE "/ENTRY:mainCRTStartup")
    endif ()
endfunction ()

#-----------------------------------------------------------------------------
# mochi_collect_shared_link_libraries
#-----------------------------------------------------------------------------

# Collect every SHARED library target reachable from ${target_name}'s link graph into ${out_var}.
# Static and interface libraries are traversed but not collected.
function (mochi_collect_shared_link_libraries target_name out_var)
    set(_collected "")
    set(_visited "")
    set(_pending "${target_name}")

    while (_pending)
        list(POP_FRONT _pending _current)

        # PRIVATE dependencies of a static library are recorded in INTERFACE_LINK_LIBRARIES wrapped
        # in $<LINK_ONLY:...>; they are real runtime dependencies, so unwrap rather than skip.
        string(REGEX REPLACE "^\\$<LINK_ONLY:(.+)>$" "\\1" _current "${_current}")

        # Conditional entries such as $<$<NOT:$<PLATFORM_ID:Windows>>:dl> cannot be evaluated until
        # generate time. Keep every identifier they mention that names a target and discard the
        # condition: an extra library in the payload is dead weight, a missing one is a crash.
        if (_current MATCHES "\\$<")
            string(REGEX REPLACE "[$<>:,]" ";" _tokens "${_current}")
            foreach (_token IN LISTS _tokens)
                if (_token AND TARGET "${_token}")
                    list(APPEND _pending "${_token}")
                endif ()
            endforeach ()
            continue ()
        endif ()

        if (_current IN_LIST _visited)
            continue ()
        endif ()
        list(APPEND _visited "${_current}")

        if (NOT TARGET "${_current}")
            continue ()
        endif ()

        get_target_property(_aliased "${_current}" ALIASED_TARGET)
        if (_aliased)
            list(APPEND _pending "${_aliased}")
            continue ()
        endif ()

        # IMPORTED targets come from outside this build, and install(TARGETS) rejects them, so
        # skip before collecting rather than after.
        get_target_property(_imported "${_current}" IMPORTED)
        if (_imported)
            continue ()
        endif ()

        get_target_property(_type "${_current}" TYPE)
        if (_type STREQUAL "SHARED_LIBRARY")
            list(APPEND _collected "${_current}")
        endif ()

        foreach (_property IN ITEMS LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
            get_target_property(_dependencies "${_current}" ${_property})
            if (_dependencies)
                list(APPEND _pending ${_dependencies})
            endif ()
        endforeach ()
    endwhile ()

    list(REMOVE_DUPLICATES _collected)
    set(${out_var} "${_collected}" PARENT_SCOPE)
endfunction ()

#-----------------------------------------------------------------------------
# mochi_install_runtime_dependencies
#-----------------------------------------------------------------------------

# Install an executable together with everything it needs to run, flat, under one component.
#
#   mochi_install_runtime_dependencies(<target> COMPONENT <component>)
#
# The flat layout is what the sibling RPATH, Windows' next-to-the-exe DLL search, and
# FindMeshCliPath()'s adjacency probe all rely on.
#
# install(RUNTIME_DEPENDENCY_SET) resolves only dependencies outside this build, so targets built
# here are named explicitly.
function (mochi_install_runtime_dependencies target_name)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "COMPONENT" "")
    if (NOT arg_COMPONENT)
        message(FATAL_ERROR "[Mochi] mochi_install_runtime_dependencies(${target_name}) requires COMPONENT.")
    endif ()

    set(_install_targets ${target_name})
    mochi_set_sibling_library_rpath(${target_name})
    if (MOCHI_BUILD_SHARED)
        # A static build absorbs all of this into the executable.
        mochi_collect_shared_link_libraries(${target_name} _shared_libraries)
        list(APPEND _install_targets ${_shared_libraries})
        foreach (_library IN LISTS _shared_libraries)
            mochi_set_sibling_library_rpath(${_library})
        endforeach ()
    endif ()

    set(_dependency_set "${target_name}_runtime_deps")
    install(TARGETS ${_install_targets}
        RUNTIME_DEPENDENCY_SET ${_dependency_set}
        RUNTIME DESTINATION . COMPONENT ${arg_COMPONENT}
        LIBRARY DESTINATION . COMPONENT ${arg_COMPONENT}
        BUNDLE DESTINATION . COMPONENT ${arg_COMPONENT}
        # No ARCHIVE: .lib/.a files are link-time only.
    )
    install(RUNTIME_DEPENDENCY_SET ${_dependency_set}
        DESTINATION .
        COMPONENT ${arg_COMPONENT}
        DIRECTORIES $<TARGET_FILE_DIR:${target_name}>
        # Skip OS-provided libraries: shipping the graphics stack is wrong.
        PRE_EXCLUDE_REGEXES "api-ms-.*" "ext-ms-.*"
        # The Visual C++ Redistributable is not OS-provided, whatever its location suggests, so
        # it is lifted back out of system32 -- POST_INCLUDE beats POST_EXCLUDE. The UCRT is a
        # Windows component and stays excluded, but nothing on a clean Windows install supplies
        # msvcp140*.dll, and nothing supplies vcruntime140*.dll either: CPython ships that one
        # beside python.exe, which covers an extension module loaded into the interpreter but
        # not an executable installed here, since that runs as its own process and searches its
        # own directory. msvcp140.dll imports it, so omitting it would ship a runtime that
        # cannot load.
        #
        # The whole family, matched the same way `check_wheels.py` polices it -- keep the two in
        # step. Breadth costs nothing, because a dependency is only copied when something
        # actually imports it. The digit after the prefix is load-bearing: it keeps out
        # msvcrt.dll and msvcp_win.dll, which are genuine Windows components.
        POST_INCLUDE_REGEXES "(concrt|mfc|msvcp|msvcr|vcamp|vccorlib|vcomp|vcruntime)[0-9].*\\.dll$"
        POST_EXCLUDE_REGEXES ".*system32.*" "^/lib/.*" "^/lib64/.*" "^/usr/lib/.*" "^/usr/lib64/.*" ".*/System/Library/.*"
    )
endfunction ()

#-----------------------------------------------------------------------------
# mochi_set_compiler_options
#-----------------------------------------------------------------------------

function (mochi_set_compiler_options target_name)

    # Mochi targets should compile with C++20. This is "PRIVATE" so users
    # are not forced to do the same.
    target_compile_features(${target_name} PRIVATE cxx_std_20)

    # TODO: Many of these settings should probably be PRIVATE so we don't forced
    # external users to build their targets with the same settings. For now, at
    # least some of them need to be public (example "/arch:AVX2" is currently
    # required for the public API in mochi_physics.h)
    target_compile_options(${target_name}
            PUBLIC
            ${MOCHI_COMPILE_OPTIONS}
    )

endfunction ()

#-----------------------------------------------------------------------------
# mochi_add_interface
#-----------------------------------------------------------------------------

# Add a header-only Mochi library with its public C++ standard requirement.
function (mochi_add_interface lib_name)
    add_library(${lib_name} INTERFACE)
    target_compile_features(${lib_name} INTERFACE cxx_std_17)
endfunction ()

#-----------------------------------------------------------------------------
# mochi_add_executable
#-----------------------------------------------------------------------------

# Add a mochi executable target (e.g. benchmark, unit test, ...)
function (mochi_add_executable target_name target_dir)
    add_executable(${target_name})

    file(GLOB_RECURSE target_srcs "${target_dir}/*.cpp" "${target_dir}/*.c" "${target_dir}/*.h")

    # Internal-only sources live under an internal/ directory.
    list(FILTER target_srcs EXCLUDE REGEX "/internal/")

    target_sources(${target_name} PRIVATE ${target_srcs})

    mochi_set_compiler_options(${target_name})
    mochi_set_output_directories(${target_name})

    # Enable reflection for src files within this executable.
    target_compile_definitions(${target_name} PRIVATE "MOCHI_USE_REFLECTION=1")
endfunction ()

#-----------------------------------------------------------------------------
# mochi_add_test_executable
#-----------------------------------------------------------------------------

# Add and register a test executable with CTest. TARGET_FILE_DIR rather than the
# output directory mochi_set_output_directories names supports multi-config
# generators, which append the configuration to that path.
#
# MOCHI_TEST_ASSETS_DIR feeds MOCHI_ASSETS_PATH, which path::FindAssetsDirectory()
# checks before walking up from the working directory -- and that directory is
# outside the source tree for the usual `cmake -S . -B <dir>`, so none of the
# other fallbacks reach the test data.
function (mochi_add_test_executable target_name target_dir)
    mochi_add_executable(${target_name} "${target_dir}")
    target_link_libraries(${target_name} PRIVATE gtest)

    add_test(
        NAME ${target_name}
        COMMAND ${target_name}
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:${target_name}>"
    )
    if (MOCHI_TEST_ASSETS_DIR)
        set_tests_properties(${target_name} PROPERTIES
            ENVIRONMENT "MOCHI_ASSETS_PATH=${MOCHI_TEST_ASSETS_DIR}")
    endif ()
endfunction ()

#-----------------------------------------------------------------------------
# mochi_add_test
#-----------------------------------------------------------------------------

# Add the test executable target corresponding to a mochi library
function (mochi_add_test lib_name)
    set(target_name "${lib_name}_test")
    mochi_add_test_executable(
        ${target_name}
        "${CMAKE_CURRENT_SOURCE_DIR}/test")
    target_link_libraries(${target_name} PRIVATE ${lib_name})
endfunction ()

#-----------------------------------------------------------------------------
# mochi_add_benchmark
#-----------------------------------------------------------------------------

# Add the benchmark executable target corresponding to a mochi library
function (mochi_add_benchmark lib_name)
    set(target_name "${lib_name}_benchmark")
    set(target_dir "${CMAKE_CURRENT_SOURCE_DIR}/benchmark")
    mochi_add_executable(${target_name} ${target_dir})
    target_link_libraries(${target_name} PRIVATE ${lib_name} benchmark)
endfunction ()

#-----------------------------------------------------------------------------
# mochi_target_warnings_not_errors
#-----------------------------------------------------------------------------

# Do not treat warnings as errors for the specified target
function (mochi_target_warnings_not_errors target_name)
    if (MSVC)
        target_compile_options(${target_name} PRIVATE /WX-)
    else ()
        target_compile_options(${target_name} PRIVATE -Wno-error)
    endif ()
endfunction ()

#-----------------------------------------------------------------------------
# mochi_target_ignore_all_warnings
#-----------------------------------------------------------------------------

# Disable all warnings for the specified target
function (mochi_target_ignore_all_warnings target_name)
    if (MSVC)
        # Strip any warning-level flags already on the target to prevent MSVC warnings
        # like, "D9025: overriding '/W3' with '/w'".
        #
        # We use /W0 rather than /w on purpose. CMake's Visual Studio generator
        # recognizes /W0../W4 and folds them into the structured
        # <WarningLevel> element (here TurnOffAllWarnings), removing the flag
        # from AdditionalOptions. It does NOT recognize lowercase /w.
        get_target_property(_opts ${target_name} COMPILE_OPTIONS)
        if (_opts)
            set(_filtered)
            foreach (_o IN LISTS _opts)
                if (NOT _o MATCHES "^/[Ww][0-4]?$")
                    list(APPEND _filtered "${_o}")
                endif ()
            endforeach ()
            set_target_properties(${target_name} PROPERTIES COMPILE_OPTIONS "${_filtered}")
        endif ()
        target_compile_options(${target_name} PRIVATE /W0)
    else ()
        target_compile_options(${target_name} PRIVATE -w)
    endif ()
endfunction ()

#-----------------------------------------------------------------------------
# mochi_set_hidden_symbol_visibility
#-----------------------------------------------------------------------------

function (mochi_set_hidden_symbol_visibility target_name)
    if ((_mochi_compiler_clang AND NOT _mochi_compiler_clang_cl) OR _mochi_compiler_gcc)
        set_target_properties(${target_name} PROPERTIES
            C_VISIBILITY_PRESET hidden
            CXX_VISIBILITY_PRESET hidden)

        if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
            target_compile_options(${target_name} PRIVATE -fno-semantic-interposition)
        endif ()
    endif ()
endfunction ()

#-----------------------------------------------------------------------------
# mochi_remove_link_library
#-----------------------------------------------------------------------------

function (mochi_remove_link_library target_name library_name)
    get_property(link_libraries_set TARGET "${target_name}" PROPERTY LINK_LIBRARIES SET)
    if (link_libraries_set)
        get_target_property(link_libraries "${target_name}" LINK_LIBRARIES)
        list(REMOVE_ITEM link_libraries "${library_name}")
        set_property(TARGET "${target_name}" PROPERTY LINK_LIBRARIES "${link_libraries}")
    endif ()
endfunction ()

#-----------------------------------------------------------------------------
# function (_mochi_add_library_impl lib_name lib_type)
#-----------------------------------------------------------------------------

# Used to implement mochi_add_library and mochi_add_static_library
# Adds the library target with all of its source files and headers.
# The caller can set other target properties after calling this function.
function (_mochi_add_library_impl lib_name lib_type)
    set(lib_dir "${CMAKE_CURRENT_SOURCE_DIR}")
    file(GLOB_RECURSE lib_srcs "${lib_dir}/src/*.cpp" "${lib_dir}/src/*.c" "${lib_dir}/src/*.h")
    file(GLOB_RECURSE lib_incs "${lib_dir}/include/*.h")

    # Internal-only source lives under any internal/ directory.
    list(FILTER lib_srcs EXCLUDE REGEX "/internal/")
    list(FILTER lib_incs EXCLUDE REGEX "/internal/")

    # Add Visual Studio debugger files (natvis and natstepfilter) for MSVC
    if (MSVC)
        file(GLOB vs_files "${lib_dir}/vs/*.natvis" "${lib_dir}/vs/*.natstepfilter")
    else ()
        set(vs_files "")
    endif ()

    add_library(${lib_name} ${lib_type})

    target_sources(${lib_name}
        PRIVATE
        ${lib_srcs}
        PUBLIC FILE_SET HEADERS
        BASE_DIRS ${lib_dir}/include/ ${lib_dir}/vs/
        FILES ${lib_incs} ${vs_files}
    )

    target_include_directories(${lib_name} PRIVATE ${lib_dir}/src)

    mochi_set_compiler_options(${lib_name})
    mochi_set_output_directories(${lib_name})

    # Enable reflection for PRIVATE src files within this library.
    target_compile_definitions(${lib_name} PRIVATE "MOCHI_USE_REFLECTION=1")

    # Add the corresponding test executable automatically
    if (MOCHI_BUILD_TESTS)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test")
            mochi_add_test(${lib_name})
        endif()
    endif ()

    # Add the corresponding benchmark executable automatically
    if (MOCHI_BUILD_BENCHMARKS)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmark")
            mochi_add_benchmark(${lib_name})
        endif()
    endif ()

endfunction ()

#-----------------------------------------------------------------------------
# mochi_add_library
#-----------------------------------------------------------------------------

# Add a mochi library target that is either STATIC or SHARED, depending on MOCHI_BUILD_SHARED
function (mochi_add_library lib_name)
    if (MOCHI_BUILD_SHARED)
        set(lib_type SHARED)
    else ()
        set(lib_type STATIC)
    endif ()
    _mochi_add_library_impl(${lib_name} ${lib_type})

    # When building a shared library, hide all symbols by default. This avoids dynamic linking
    # overhead within internal library code. The public API functions must be exported
    # explicitly with macros (e.g. MOCHI_API).
    if (MOCHI_BUILD_SHARED)
        mochi_set_hidden_symbol_visibility(${lib_name})
    endif ()
endfunction ()

#-----------------------------------------------------------------------------
# mochi_add_static_library
#-----------------------------------------------------------------------------

# Add a mochi library target that must be a STATIC library (currently only mochi_core).
function (mochi_add_static_library lib_name)
    _mochi_add_library_impl(${lib_name} STATIC)

    # Set POSITION_INDEPENDENT_CODE if this static library will be used in combination
    # with Mochi shared libraries.
    if (MOCHI_BUILD_SHARED)
        set_target_properties(
            ${lib_name}
            PROPERTIES
            POSITION_INDEPENDENT_CODE ${MOCHI_BUILD_SHARED}
        )
    endif ()
endfunction ()
