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

include(FetchContent)

# Fix for git path on Windows in case FetchContent is used to download files from github.
if ((CMAKE_SYSTEM_NAME STREQUAL "Windows") AND (NOT GIT_EXECUTABLE))
    if (EXISTS "C:/Program Files/Git/cmd/git.exe")
        set(GIT_EXECUTABLE "C:/Program Files/Git/cmd/git.exe")
    endif ()
endif ()


#-----------------------------------------------------------------------------
# Accelerate
#-----------------------------------------------------------------------------

# Import the "Accelerate" library if the target is an Apple platform.
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    find_library(ACCELERATE_LIBRARY Accelerate)
    if (NOT ACCELERATE_LIBRARY)
        message(WARNING "Accelerate library not found. Apples matrix acceleration hardware will not be used.")
    endif ()
endif ()

#-----------------------------------------------------------------------------
# benchmark
#-----------------------------------------------------------------------------

if (MOCHI_BUILD_BENCHMARKS)

    # benchmark options
    set(BENCHMARK_ENABLE_TESTING OFF)
    set(BENCHMARK_INSTALL_DOCS OFF)

    # Find package or build from source
    if (MOCHI_BENCHMARK_FIND_PACKAGE)
        if (benchmark_FOUND)
            message(STATUS "[Mochi] Found benchmark")
        endif ()
    else ()
        message(STATUS "[Mochi] Including Benchmark")
        FetchContent_Declare(benchmark SOURCE_DIR ${MOCHI_BENCHMARK_SOURCE_DIR})
        FetchContent_MakeAvailable(benchmark)
        mochi_target_ignore_all_warnings(benchmark)
        mochi_target_ignore_all_warnings(benchmark_main)
        set_target_properties(benchmark PROPERTIES FOLDER "third_party")
        set_target_properties(benchmark_main PROPERTIES FOLDER "third_party")
    endif ()
endif ()

#-----------------------------------------------------------------------------
# Marl
#-----------------------------------------------------------------------------

message(STATUS "[Mochi] Including Marl")

# Marl options
set(MARL_BUILD_SHARED ${MOCHI_BUILD_SHARED})

# Always build marl from source because it has been modified for Mochi.
add_subdirectory(marl/marl)

mochi_target_ignore_all_warnings(marl)
mochi_set_output_directories(marl) # The the dynamic library can be easily found
set_target_properties(marl PROPERTIES FOLDER "third_party")

if (WIN32)
    target_compile_definitions(marl PRIVATE NOMINMAX)
endif ()

#-----------------------------------------------------------------------------
# Eigen (header-only)
#-----------------------------------------------------------------------------

if (MOCHI_USE_EIGEN)
    message(STATUS "[Mochi] Including Eigen")
    add_library(eigen INTERFACE)
    add_library(Eigen3::Eigen ALIAS eigen)
    target_include_directories(eigen INTERFACE "${MOCHI_EIGEN_SOURCE_DIR}")
    set_target_properties(eigen PROPERTIES FOLDER "third_party")
endif ()

#-----------------------------------------------------------------------------
# EnTT
#-----------------------------------------------------------------------------

# EnTT options
set(ENTT_INCLUDE_AOB OFF)
set(ENTT_INCLUDE_HEADERS ON)
if (MSVC)
    set(ENTT_INCLUDE_NATVIS ON)
endif ()

# Find package or build from source
if (MOCHI_FIND_ENTT_PACKAGE)
    if (EnTT_FOUND)
        message(STATUS "[Mochi] Found EnTT")
    endif ()
else ()
    message(STATUS "[Mochi] Including EnTT")
    FetchContent_Declare(EnTT SOURCE_DIR ${MOCHI_ENTT_SOURCE_DIR})
    FetchContent_MakeAvailable(EnTT)

    if(MSVC)
        # CMake normally would not generate EnTT.vcxproj because it is an INTERFACE target
        # that does not produce any build artifacts. We want to generate it anyway because
        # it is nice to have the headers and natvis files in the IDE.
        file(GLOB_RECURSE entt_natvis_files "${EnTT_SOURCE_DIR}/natvis/entt/*.natvis")
        file(GLOB_RECURSE entt_include_files "${EnTT_SOURCE_DIR}/src/entt/*.h" "${EnTT_SOURCE_DIR}/src/entt/*.hpp")
        target_sources(EnTT
            PRIVATE
            ${entt_natvis_files}
            ${entt_include_files}
        )
    endif()

    set_target_properties(EnTT PROPERTIES FOLDER "third_party")
endif ()

# ENTT_STANDARD_CPP must be defined to force EnTT to generate type hashes in a way that is portable across multiple compilation/linking units.
target_compile_definitions(
    EnTT
    INTERFACE
    "-DENTT_STANDARD_CPP"
)

#-----------------------------------------------------------------------------
# GTest
#-----------------------------------------------------------------------------

if (MOCHI_BUILD_TESTS)

    # GTest options
    set(BUILD_GMOCK OFF)

    # Find package or build from source
    if (MOCHI_GTEST_FIND_PACKAGE)
        if (GTest_FOUND)
            message(STATUS "[Mochi] Found GTest")
        endif ()
    else ()
        message(STATUS "[Mochi] Including GTest")
        FetchContent_Declare(GTest SOURCE_DIR ${MOCHI_GTEST_SOURCE_DIR})
        FetchContent_MakeAvailable(GTest)
        mochi_target_ignore_all_warnings(gtest)
        mochi_target_ignore_all_warnings(gtest_main)
        set_target_properties(gtest PROPERTIES FOLDER "third_party")
        set_target_properties(gtest_main PROPERTIES FOLDER "third_party")
    endif ()

endif ()

#-----------------------------------------------------------------------------
# ZLIB
#-----------------------------------------------------------------------------

# HDF5 and Filament both require ZLIB. ZLIB must be imported before HDF5.
if (MOCHI_USE_HDF5 OR MOCHI_BUILD_RENDERER)

    # ZLIB options
    set(ZLIB_BUILD_EXAMPLES OFF)
    set(ZLIB_BUILD_MINIZIP OFF)
    set(ZLIB_BUILD_TESTING OFF)

    # Find package or build from source
    if (MOCHI_ZLIB_FIND_PACKAGE)
        if (ZLIB_FOUND)
            message(STATUS "[Mochi] Found ZLIB")
        endif ()
    else ()
        message(STATUS "[Mochi] Including ZLIB")

        # The zlib shared target is excluded from the build below, but its install rule for
        # `libz.so.1.3.1` survives, so an unscoped `cmake --install` fails on a file that was never
        # produced. Nothing consumes an installed zlib, so drop the rules.
        set(SKIP_INSTALL_ALL ON)

        FetchContent_Declare(ZLIB SOURCE_DIR ${MOCHI_ZLIB_SOURCE_DIR})
        FetchContent_MakeAvailable(ZLIB)
        # FetchContent_GetProperties(ZLIB)
        # if(NOT zlib_POPULATED)
        #     message(STATUS POPULATING)
        #     FetchContent_Populate(ZLIB)
        # endif()

        # Exclude zlib shared library from build - we only use zlibstatic.
        # This avoids the broken RC compiler issue on Windows with Ninja.
        # The zlib target still exists for CMake logic, but won't be built.
        if(TARGET zlib)
            set_target_properties(zlib PROPERTIES EXCLUDE_FROM_ALL TRUE)
        endif()

        mochi_target_ignore_all_warnings(zlibstatic)
        set_target_properties(
            zlibstatic
            PROPERTIES
            POSITION_INDEPENDENT_CODE ${MOCHI_BUILD_SHARED}
        )

        # Set variables so that hdf5 can find everything, as if zlib was installed.
        set(ZLIB_FOUND ON)
        set(ZLIB_DIR "${MOCHI_ZLIB_SOURCE_DIR}")
        set(ZLIB_INCLUDE_DIRS "${MOCHI_ZLIB_SOURCE_DIR}")
        set(ZLIB_INCLUDE_DIR_GEN "${zlib_BINARY_DIR}")

        set_target_properties(zlibstatic PROPERTIES FOLDER "third_party")

        # Scoped to zlib: the name is generic, and HDF5 is added below.
        unset(SKIP_INSTALL_ALL)
    endif ()

endif ()

#-----------------------------------------------------------------------------
# HDF5
#-----------------------------------------------------------------------------

if (MOCHI_USE_HDF5)

    # HDF5 options
    set(HDF5_EXTERNALLY_CONFIGURED ON)
    set(ZLIB_USE_EXTERNAL ON)
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_STATIC_LIBS ON)
    set(HDF_PACKAGE_NAMESPACE HDF5)
    set(HDF5_ENABLE_Z_LIB_SUPPORT ON)
    set(HDF5_ENABLE_SZIP_SUPPORT OFF)
    set(HDF5_ENABLE_ALL_WARNINGS OFF)
    set(HDF5_BUILD_CPP_LIB ON)
    set(HDF5_BUILD_HL_LIB ON)
    set(HDF5_BUILD_DOC OFF)
    set(HDF5_BUILD_EXAMPLES OFF)
    set(HDF5_BUILD_FORTRAN OFF)
    set(HDF5_BUILD_GENERATORS OFF)
    set(HDF5_BUILD_HL_GIF_TOOLS OFF)
    set(HDF5_BUILD_JAVA OFF)
    set(HDF5_BUILD_PARALLEL_TOOLS OFF)
    set(HDF5_BUILD_STATIC_TOOLS OFF)
    set(HDF5_BUILD_TOOLS OFF)
    set(HDF5_BUILD_UTILS OFF)
    set(BUILD_TESTING OFF)

    # Find package or build from source
    if (MOCHI_HDF5_FIND_PACKAGE)
        if (HDF5_FOUND)
            message(STATUS "[Mochi] Found HDF5")
        endif ()
    else ()
        message(STATUS "[Mochi] Including HDF5")
        FetchContent_Declare(HDF5 SOURCE_DIR ${MOCHI_HDF5_SOURCE_DIR})
        FetchContent_MakeAvailable(HDF5)

        mochi_target_ignore_all_warnings(hdf5_cpp-static)
        mochi_target_ignore_all_warnings(hdf5_hl_cpp-static)
        mochi_target_ignore_all_warnings(hdf5_hl-static)
        mochi_target_ignore_all_warnings(hdf5-static)

        # MSVC doesn't ignore warning C4005 (macro redefinition) even with argument "/w"
        if (MSVC)
            target_compile_options(hdf5_hl-static PRIVATE "/wd4005")
            target_compile_options(hdf5-static PRIVATE "/wd4005")
        endif ()

        if (NOT MOCHI_ZLIB_FIND_PACKAGE)
            # hdf5-static needs this additional include directory when ZLIB is built from source.
            target_include_directories(hdf5-static PRIVATE "${ZLIB_INCLUDE_DIR_GEN}")
        endif ()

        set_target_properties(hdf5_cpp-static PROPERTIES FOLDER "third_party")
        set_target_properties(hdf5_hl_cpp-static PROPERTIES FOLDER "third_party")
        set_target_properties(hdf5_hl-static PROPERTIES FOLDER "third_party")
        set_target_properties(hdf5-static PROPERTIES FOLDER "third_party")
    endif ()

    # Combine hdf5 targets into a single target name that other libraries can take a dependency on.
    add_library(hdf5_targets INTERFACE)
    target_link_libraries(hdf5_targets INTERFACE hdf5-static hdf5_cpp-static hdf5_hl_cpp-static)
    add_library(HDF5::HDF5 ALIAS hdf5_targets)

    # The HDF5 archive reaches every Mochi shared library through mochi_core, so a process can end up
    # with several .so files each carrying a copy of it. ELF's flat symbol namespace binds all copies of
    # a given global to whichever .so the loader saw first, while each .so still runs its own static
    # initializers. HDF5 aborts due to redundant initialization.
    #
    # The solution is to exclude HDF5 symbols from each shared library's dynamic symbol table. Each
    # library then gets its own private copy and its own initialization. This is safe only as long as
    # no `hid_t` or `H5::` object crosses a Mochi library boundary. mochi_core's HDF5 wrappers hand
    # out such objects, so they are declared MOCHI_HDF5_LOCAL to keep each library bound to its own
    # copy -- see mochi_core/include/mochi_core/utils/hdf5_utils.h.
    #
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Applies to every consumer that gets its own dynamic symbol table: SHARED_LIBRARY and
        # MODULE_LIBRARY, the latter being what pybind11_add_module() produces for the Python
        # extensions.
        set(_mochi_hdf5_dynamic_image "$<OR:$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>,$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>>")
        foreach (_mochi_hdf5_target hdf5-static hdf5_cpp-static hdf5_hl-static hdf5_hl_cpp-static)
            if (TARGET ${_mochi_hdf5_target})
                target_link_options(hdf5_targets INTERFACE
                    "$<${_mochi_hdf5_dynamic_image}:-Wl,--exclude-libs,$<TARGET_FILE_NAME:${_mochi_hdf5_target}>>")
            endif ()
        endforeach ()
        unset(_mochi_hdf5_dynamic_image)
    endif ()
endif ()

#-----------------------------------------------------------------------------
# xxhash (header-only)
#-----------------------------------------------------------------------------

add_library(xxhash INTERFACE)
target_include_directories(xxhash INTERFACE "xxhash")
set_target_properties(xxhash PROPERTIES FOLDER "third_party")

#-----------------------------------------------------------------------------
# miniz (zip archive read/write)
#-----------------------------------------------------------------------------

add_library(miniz STATIC "miniz/miniz.c" "miniz/miniz.h")
target_include_directories(miniz PUBLIC "miniz")
set_target_properties(miniz PROPERTIES
    FOLDER "third_party"
    POSITION_INDEPENDENT_CODE ${MOCHI_BUILD_SHARED}
)
mochi_target_ignore_all_warnings(miniz)

#-----------------------------------------------------------------------------
# tinyfiledialogs
#-----------------------------------------------------------------------------

if (MOCHI_USE_TINYFILEDIALOG)
    add_library(tinyfiledialogs STATIC "tinyfiledialogs/tinyfiledialogs.c" "tinyfiledialogs/tinyfiledialogs.h")
    target_include_directories(tinyfiledialogs PUBLIC "tinyfiledialogs")
    set_target_properties(tinyfiledialogs PROPERTIES
        FOLDER "third_party"
        POSITION_INDEPENDENT_CODE ${MOCHI_BUILD_SHARED}
    )
    mochi_target_ignore_all_warnings(tinyfiledialogs)

    # Windows requires comdlg32 and ole32 for native file dialogs
    if (WIN32)
        target_link_libraries(tinyfiledialogs PRIVATE comdlg32 ole32)
    endif ()

    if (UNIX AND NOT APPLE)
        # tinyfiledialogs on linux requires one of these helper programs to be installed.
        find_program(MOCHI_FILE_DIALOG_HELPER NAMES zenity kdialog yad matedialog qarma)
        if (NOT MOCHI_FILE_DIALOG_HELPER)
            message(WARNING
                "[Mochi] No file dialog helper found; file/folder browsing will not work unless you install one.\n"
                "  Fedora/RHEL:   sudo dnf install zenity\n"
                "  Ubuntu/Debian: sudo apt-get install zenity")
        endif ()
    endif ()
endif ()

#-----------------------------------------------------------------------------
# picojson
#-----------------------------------------------------------------------------

add_library(picojson INTERFACE)
target_sources(picojson INTERFACE "picojson/picojson/picojson.h")
target_include_directories(picojson INTERFACE "picojson")
set_target_properties(picojson PROPERTIES FOLDER "third_party")

#-----------------------------------------------------------------------------
# tiny_obj_loader (OBJ parser) -- permissive mesh I/O for mochi_core
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# tinyxml2 (XML parser) -- COLLADA (.dae) reading for mochi_renderer
#-----------------------------------------------------------------------------

# Mochi-owned vendored copy under mochi/third_party (MOCHI_TINYXML2_SOURCE_DIR),
# built like tiny_obj_loader.
add_library(tinyxml2 STATIC "${MOCHI_TINYXML2_SOURCE_DIR}/tinyxml2.cpp" "${MOCHI_TINYXML2_SOURCE_DIR}/tinyxml2.h")
target_include_directories(tinyxml2 PUBLIC "${MOCHI_TINYXML2_SOURCE_DIR}")
set_target_properties(tinyxml2 PROPERTIES
    FOLDER "third_party"
    POSITION_INDEPENDENT_CODE ${MOCHI_BUILD_SHARED}
)
mochi_target_ignore_all_warnings(tinyxml2)

#-----------------------------------------------------------------------------
# tiny_obj_loader (OBJ parser) -- permissive mesh I/O for mochi_core
#-----------------------------------------------------------------------------

# tiny_obj_loader ships a single .cc implementation file alongside its header.
add_library(tiny_obj_loader STATIC "${MOCHI_TINYOBJ_SOURCE_DIR}/tiny_obj_loader.cc")
target_include_directories(tiny_obj_loader PUBLIC "${MOCHI_TINYOBJ_SOURCE_DIR}")
# Parse OBJ coordinates at double precision; mochi_core casts down to `real` for single-precision
# builds. PUBLIC so `real_t` matches in both this library and mochi_core's view of attrib_t.
target_compile_definitions(tiny_obj_loader PUBLIC TINYOBJLOADER_USE_DOUBLE)
set_target_properties(tiny_obj_loader PROPERTIES
    FOLDER "third_party"
    POSITION_INDEPENDENT_CODE ${MOCHI_BUILD_SHARED}
)
mochi_target_ignore_all_warnings(tiny_obj_loader)

#-----------------------------------------------------------------------------
# happly (PLY parser, header-only) -- permissive mesh I/O for mochi_core
#-----------------------------------------------------------------------------

# happly is header-only; #include <happly/happly.h> resolves against MOCHI_HAPPLY_SOURCE_DIR.
add_library(happly INTERFACE)
target_include_directories(happly INTERFACE "${MOCHI_HAPPLY_SOURCE_DIR}")
set_target_properties(happly PROPERTIES FOLDER "third_party")

#-----------------------------------------------------------------------------
# pybind11
#-----------------------------------------------------------------------------

if (MOCHI_USE_PYBIND)
    # pybind11 requires Python development headers (Python.h) to build.
    # Development.Module, not Development: the latter implies Development.Embed, which needs a
    # libpython that manylinux CPython does not ship.
    find_package(Python3 3.12...3.12.99 COMPONENTS Interpreter Development.Module QUIET)

    if (Python3_FOUND AND Python3_Development.Module_FOUND)
        message(STATUS "[Mochi] Found Python3 (${Python3_VERSION}) with development headers")
        message(STATUS "[Mochi]   Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
    else ()
        # Provide helpful error messages for different platforms
        if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
            message(FATAL_ERROR
                "[Mochi] Failed to find Python3 development headers (Python.h).\n"
                "  pybind11 requires Python development headers to build.\n"
                "  To fix this on Fedora/RHEL, run: sudo dnf install python3-devel\n"
                "  To fix this on Ubuntu/Debian, run: sudo apt install python3-dev\n"
                "  Alternatively, disable pybind support with: -DMOCHI_USE_PYBIND=OFF"
            )
        elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            message(FATAL_ERROR
                "[Mochi] Failed to find Python3 development headers (Python.h).\n"
                "  pybind11 requires Python development headers to build.\n"
                "  On macOS, install Python via Homebrew: brew install python3\n"
                "  Or use pyenv/conda which includes development headers.\n"
                "  Alternatively, disable pybind support with: -DMOCHI_USE_PYBIND=OFF"
            )
        elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
            message(FATAL_ERROR
                "[Mochi] Failed to find Python3 development headers (Python.h).\n"
                "  pybind11 requires Python development headers to build.\n"
                "  On Windows, install Python from python.org (includes dev headers).\n"
                "  Make sure to check 'Add Python to PATH' during installation.\n"
                "  Alternatively, disable pybind support with: -DMOCHI_USE_PYBIND=OFF"
            )
        else ()
            message(FATAL_ERROR
                "[Mochi] Failed to find Python3 development headers (Python.h).\n"
                "  pybind11 requires Python development headers to build.\n"
                "  Please install Python3 with development headers for your platform.\n"
                "  Alternatively, disable pybind support with: -DMOCHI_USE_PYBIND=OFF"
            )
        endif ()
    endif ()

    # Set hints for pybind11's Python search to use the Python3 we just found.
    # pybind11 uses the legacy FindPythonInterp/FindPythonLibs modules which look
    # for PYTHON_EXECUTABLE, PYTHON_INCLUDE_DIR, and PYTHON_LIBRARY variables.
    set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "Python interpreter" FORCE)
    set(PYTHON_INCLUDE_DIR "${Python3_INCLUDE_DIRS}" CACHE PATH "Python include directory" FORCE)

    # Find the Python import library.
    #
    # Windows-only: pybind11::module links ${PYTHON_LIBRARIES} solely on Windows and
    # Cygwin. macOS resolves the symbols with -undefined dynamic_lookup and Linux leaves
    # them to the interpreter that loads the module, so neither platform reads this
    # variable -- which is also why Development.Module leaves Python3's library variables
    # unset there. Searching on those platforms would only produce a spurious warning.
    if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
        if (Python3_LIBRARY)
            set(PYTHON_LIBRARY "${Python3_LIBRARY}" CACHE FILEPATH "Python library" FORCE)
        elseif (Python3_LIBRARIES)
            list(GET Python3_LIBRARIES 0 _first_lib)
            set(PYTHON_LIBRARY "${_first_lib}" CACHE FILEPATH "Python library" FORCE)
        else ()
            # Development.Module normally sets one of the above on Windows, so reaching
            # here means an unusual install. Search the standard layout as a last resort:
            # <prefix>/include/Python.h beside <prefix>/libs/pythonXY.lib.
            get_filename_component(_python_prefix "${Python3_INCLUDE_DIRS}" DIRECTORY)
            set(_python_lib_name "python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}")
            set(_python_lib_dirs "${_python_prefix}/libs" "${_python_prefix}/lib")

            find_library(PYTHON_LIBRARY
                NAMES ${_python_lib_name}
                PATHS ${_python_lib_dirs}
                NO_DEFAULT_PATH
            )
            if (PYTHON_LIBRARY)
                message(STATUS "[Mochi]   Python3_LIBRARY: ${PYTHON_LIBRARY}")
            else ()
                message(WARNING
                    "[Mochi] Could not find the Python import library '${_python_lib_name}.lib' for\n"
                    "  ${Python3_EXECUTABLE} (${Python3_VERSION}). pybind11 must link it on Windows,\n"
                    "  so the build will likely fail. To fix, either:\n"
                    "    - Install the Python ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR} development files from python.org, or\n"
                    "    - Point CMake at such an install: -DPython3_EXECUTABLE=<path\\to\\python.exe>, or\n"
                    "    - Name the import library directly: -DPYTHON_LIBRARY=<path\\to\\${_python_lib_name}.lib>, or\n"
                    "    - Disable Python bindings: -DMOCHI_USE_PYBIND=OFF"
                )
            endif ()
        endif ()
    endif ()

    if (MOCHI_PYBIND_FIND_PACKAGE)
        if (pybind11_FOUND)
            message(STATUS "[Mochi] Found pybind11")
        endif ()
    else ()
        message(STATUS "[Mochi] Including pybind11")
        FetchContent_Declare(pybind11 SOURCE_DIR ${MOCHI_PYBIND_SOURCE_DIR})
        # Suppress CMake warning: "Compatibility with CMake < 3.10 will be removed from a future version"
        if (DEFINED CACHE{CMAKE_WARN_DEPRECATED})
            set(_mochi_warn_deprecated "$CACHE{CMAKE_WARN_DEPRECATED}")
        endif ()
        set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(pybind11)
        if (DEFINED _mochi_warn_deprecated)
            set(CMAKE_WARN_DEPRECATED "${_mochi_warn_deprecated}" CACHE BOOL "" FORCE)
            unset(_mochi_warn_deprecated)
        else ()
            unset(CMAKE_WARN_DEPRECATED CACHE)
        endif ()
    endif ()
endif ()

#-----------------------------------------------------------------------------
# Tracy
#-----------------------------------------------------------------------------

if (MOCHI_USE_TRACY)
    # Find package or build from source
    if (MOCHI_TRACY_FIND_PACKAGE)
        if (tracy_FOUND)
            message(STATUS "[Mochi] Found Tracy")
        endif ()
    else ()
        # If Mochi is using shared libraries, then Tracy needs to build as a shared library as well.
        # Otherwise, each dynamic library would have a separate Tracy Client (at least on Windows).
        if (${MOCHI_BUILD_SHARED})
          set(TRACY_STATIC OFF CACHE BOOL "" FORCE)
        else ()
          set(TRACY_STATIC ON CACHE BOOL "" FORCE)
        endif ()

        # Configure Tracy with the ability to build the GUI
        set(TRACY_BUILD_GUI ON CACHE BOOL "" FORCE)

        message(STATUS "[Mochi] Including Tracy v${TRACY_VERSION} from GitHub")
        FetchContent_Declare(
            tracy
            GIT_REPOSITORY https://github.com/wolfpld/tracy.git
            GIT_TAG "v${TRACY_VERSION}"
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
        )
        FetchContent_MakeAvailable(tracy)
        set_target_properties(
            TracyClient
            PROPERTIES
            POSITION_INDEPENDENT_CODE ${MOCHI_BUILD_SHARED}
            FOLDER "third_party"
            WINDOWS_EXPORT_ALL_SYMBOLS ${MOCHI_BUILD_SHARED}
        )
        mochi_target_ignore_all_warnings(TracyClient)
        mochi_set_output_directories(TracyClient)
        target_compile_definitions(
            TracyClient
            PUBLIC
            "-DTRACY_ENABLE=1"
            "-DTRACY_FIBERS=1"
            "-DTRACY_ON_DEMAND"
        )
        if (${MOCHI_BUILD_SHARED})
            target_compile_definitions(
                TracyClient
                PRIVATE
                "-DTRACY_EXPORTS=1"
                PUBLIC
                "-DTRACY_DELAYED_INIT=1"
                "-DTRACY_MANUAL_LIFETIME=1"
                "-DTRACY_IMPORTS=1"
            )
        endif ()
    endif ()

    # Optionally build the Tracy GUI as well
    if (MOCHI_BUILD_TRACY_GUI)
        message(STATUS "[Mochi] Including the Tracy GUI")
        include(ExternalProject)

        # Build the GUI with X11 support on Linux.
        set(TRACY_LINUX_ARGS "")
        set(TRACY_FILE_SELECTOR_ARGS "")
        if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
            set(TRACY_LINUX_ARGS "-DGLFW_BUILD_WAYLAND=OFF;-DGLFW_BUILD_X11=ON")

            # Check if dbus-1 is available (required for native file dialogs on Linux)
            find_package(PkgConfig QUIET)
            if (PKG_CONFIG_FOUND)
                pkg_check_modules(DBUS QUIET dbus-1)
                if (DBUS_FOUND)
                    message(STATUS "[Mochi] dbus-1 found - Tracy GUI will have native file dialog support")
                else()
                    message(STATUS "[Mochi] dbus-1 not found - Tracy GUI will build without file dialog support")
                    message(STATUS "[Mochi] To enable file dialogs, install dbus development package:")
                    message(STATUS "[Mochi]   Fedora/RHEL: sudo dnf install dbus-devel")
                    message(STATUS "[Mochi]   Ubuntu/Debian: sudo apt install libdbus-1-dev")
                    set(TRACY_FILE_SELECTOR_ARGS "-DNO_FILESELECTOR=ON;-DCMAKE_CXX_FLAGS=-DTRACY_NO_FILESELECTOR")
                endif()
            endif()
        endif()

        ExternalProject_Add(tracy_gui
            SOURCE_DIR ${tracy_SOURCE_DIR}/profiler
            BINARY_DIR ${CMAKE_BINARY_DIR}/bin
            CMAKE_ARGS -DLEGACY=1 -DCMAKE_BUILD_TYPE=Release ${TRACY_LINUX_ARGS} ${TRACY_FILE_SELECTOR_ARGS}
            BUILD_ALWAYS 1
            BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release
            INSTALL_COMMAND "" # no install necessary
        )
    elseif (MOCHI_DOWNLOAD_TRACY_GUI)
        if (WIN32)
            message(STATUS "[Mochi] Downloading the Tracy GUI to ${CMAKE_BINARY_DIR}/bin.")
            include(ExternalProject)
            set(TRACY_GUI_URL "https://github.com/wolfpld/tracy/releases/download/v${TRACY_VERSION}/windows-${TRACY_VERSION}.zip")
            ExternalProject_Add(TracyGUI
                URL ${TRACY_GUI_URL}
                CONFIGURE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND "" # NO NEED TO BUILD OR INSTALL
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
                SOURCE_DIR "${CMAKE_BINARY_DIR}/bin"
            )
        else ()
            message(STATUS "[Mochi] Automatic download of the Tracy GUI is only supported on Windows. To build the Tracy GUI automatically, set MOCHI_BUILD_TRACY_GUI.")
        endif ()
    endif ()
endif ()

#-----------------------------------------------------------------------------
# Filament
#-----------------------------------------------------------------------------

if (MOCHI_BUILD_RENDERER)

    message(STATUS "[Mochi] Including Filament from: ${MOCHI_FILAMENT_SOURCE_DIR}")

    # Pre-set Filament options BEFORE add_subdirectory().
    set(FILAMENT_SKIP_SDL2 ON CACHE BOOL "" FORCE)
    set(FILAMENT_SKIP_SAMPLES ON CACHE BOOL "" FORCE)
    set(FILAMENT_BUILD_FILAMAT OFF CACHE BOOL "" FORCE)
    set(FILAMENT_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(FILAMENT_ENABLE_MATDBG OFF CACHE BOOL "" FORCE)
    set(FILAMENT_ENABLE_FGVIEWER OFF CACHE BOOL "" FORCE)
    set(FILAMENT_ENABLE_PERFETTO OFF CACHE BOOL "" FORCE)
    set(FILAMENT_SUPPORTS_WEBGPU OFF CACHE BOOL "" FORCE)
    set(FILAMENT_SUPPORTS_WEBP_TEXTURES OFF CACHE BOOL "" FORCE)
    # We have patched out matc/resgen calls, but turn off the prebuilt-executables
    # branch defensively to keep Filament from trying to include ImportExecutables*.cmake.
    set(FILAMENT_IMPORT_PREBUILT_EXECUTABLES OFF CACHE BOOL "" FORCE)

    # Filament's top-level CMakeLists does an unconditional
    # `find_program(CCACHE_PROGRAM ccache)` and, if it hits, routes every
    # Filament compile through a generated launch-c/launch-cxx wrapper. A
    # broken or unsupported ccache on PATH therefore fails the build with no
    # way to opt out (on Meta devservers /usr/local/bin/ccache is the retired
    # fbpkg build, which aborts every compile with "Error 837"). Seeding the
    # cache entry short-circuits find_program, leaving the compiler unwrapped.
    if (MOCHI_USE_CCACHE)
        # Clear a forced-empty entry left by a previous OFF configure of this
        # build tree, otherwise find_program treats it as already resolved and
        # never searches PATH. An explicit non-empty user path is preserved.
        if (DEFINED CACHE{CCACHE_PROGRAM} AND NOT CCACHE_PROGRAM)
            unset(CCACHE_PROGRAM CACHE)
        endif ()
    else ()
        set(CCACHE_PROGRAM "" CACHE FILEPATH "Disabled by MOCHI_USE_CCACHE=OFF" FORCE)
    endif ()

    # Filament's tier-1 supported compilers are Clang and MSVC. It gates GCC
    # support behind FILAMENT_ENABLE_EXPERIMENTAL_GCC_SUPPORT.
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set(FILAMENT_ENABLE_EXPERIMENTAL_GCC_SUPPORT ON CACHE BOOL "" FORCE)
    endif ()

    # Filament defaults USE_STATIC_LIBCXX to ON on Linux, which forces
    # `-static-libgcc -static-libstdc++` into every link line. Mochi (and many
    # build environments, including our Linux CI) does not ship the static
    # libstdc++ archive, so the link of shared targets like
    # libmochi_renderer.so fails with "cannot find -lstdc++". Mochi targets
    # dynamic system stdlibs, so disable the static-stdlib path entirely.
    set(USE_STATIC_LIBCXX OFF CACHE BOOL "" FORCE)

    # Filament unconditionally does add_compile_options(-fno-rtti) unless
    # FILAMENT_ENABLE_RTTI is ON. MSVC's cl/ml64 don't understand the
    # GCC/Clang spelling and warn "D9002: ignoring unknown option '-fno-rtti'"
    # (and ml64 fails outright on bluegl's .S file). RTTI is enabled by default
    # under MSVC anyway, so turning Filament's RTTI option ON for MSVC simply
    # skips the bogus flag without changing behavior.
    if (MSVC)
        set(FILAMENT_ENABLE_RTTI ON CACHE BOOL "" FORCE)

        # Filament defaults USE_STATIC_CRT to ON on Windows, which forces /MT
        # (static CRT) into every Filament object. Mochi targets use the MSVC
        # default /MD (dynamic CRT), so linking Filament's static libs into
        # libmochi_renderer.dll fails with LNK2038 "RuntimeLibrary mismatch
        # (MT_StaticRelease vs MD_DynamicRelease)". Match Mochi's dynamic CRT.
        set(USE_STATIC_CRT OFF CACHE BOOL "" FORCE)

        # bluegl assembles one MASM source, and CMake resolves an assembler it was
        # not given by bare name. `ml64` is MSVC's, on PATH only inside a Developer
        # Command Prompt; when it is absent CMake reports "ASM_MASM compiler
        # identification is unknown", keeps the bare name, and the build dies much
        # later on `CreateProcess failed`. `llvm-ml` sits beside the clang-cl these
        # builds already require, so preferring it keeps the toolchain in one family
        # and leaves nothing to go looking for a Visual Studio layout. A cl.exe build
        # is in that prompt already and keeps ml64. `-m64` because llvm-ml, unlike
        # ml64, defaults to 32-bit output.
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(CMAKE_ASM_MASM_COMPILER "llvm-ml" CACHE FILEPATH "" FORCE)
            set(CMAKE_ASM_MASM_FLAGS "-m64" CACHE STRING "" FORCE)
        endif ()
    endif ()

    # Filament uses bin_to_c.py from its tools/ directory.
    set(MOCHI_FILAMENT_TOOLS_DIR "${MOCHI_FILAMENT_SOURCE_DIR}/tools")

    # libpng/tnt and tinyexr/tnt link to a 'z' target. Map it to whichever ZLIB
    # target Mochi configured so usage requirements such as include paths propagate.
    if (NOT TARGET z)
        if (TARGET zlibstatic)
            add_library(z ALIAS zlibstatic)
        else ()
            add_library(z ALIAS ZLIB::ZLIB)
        endif ()
    endif ()

    # Filament's bundled external/zstd was trimmed from this tree, but uberz
    # (a transitive gltfio dependency) includes <zstd.h> and links a `zstd`
    # target. basisu also prefers an external `zstd` target over building its
    # own bundled copy (avoiding duplicate-symbol conflicts). Provide a `zstd`
    # target built from the zstd amalgamation that still ships inside basisu.
    # This must be defined BEFORE add_subdirectory() so both uberz and basisu
    # resolve to it.
    if (NOT TARGET zstd)
        set(_mochi_zstd_dir "${MOCHI_FILAMENT_SOURCE_DIR}/third_party/basisu/zstd")
        add_library(zstd STATIC "${_mochi_zstd_dir}/zstd.c")
        target_include_directories(zstd PUBLIC "${_mochi_zstd_dir}")
        set_target_properties(zstd PROPERTIES
            FOLDER "third_party/filament"
            POSITION_INDEPENDENT_CODE ON
        )
        mochi_target_ignore_all_warnings(zstd)
        unset(_mochi_zstd_dir)
    endif ()

    # Filament's bundled external/getopt was trimmed from this tree. On
    # platforms without a system getopt (Windows), filament's `utils` library
    # does `target_link_libraries(utils PUBLIC getopt)`, so without a `getopt`
    # target CMake emits a bogus `getopt.lib` link input (LNK1181). The trimmed
    # command-line tools were the only consumers that actually called getopt;
    # `utils` itself only re-exports the declarations from its getopt.h. Provide
    # an empty INTERFACE `getopt` target so the link reference resolves to
    # nothing. Must be defined BEFORE add_subdirectory().
    if (NOT TARGET getopt)
        add_library(getopt INTERFACE)
    endif ()

    # Suppress all warnings from Filament and its dependencies.
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(-fpermissive) # Work-around for non-standard-compliant Filament code.
        add_compile_options(-w)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-w)
    endif ()

    # Keep Filament's intermediate build tree shallow to reduce path length issues on Windows.
    set(_mochi_filament_binary_dir "${CMAKE_BINARY_DIR}/_deps/filament")
    add_subdirectory(
        ${MOCHI_FILAMENT_SOURCE_DIR}
        ${_mochi_filament_binary_dir}
        EXCLUDE_FROM_ALL
    )

    # ---------------------------------------------------------------------
    # Give Filament back its infinities under clang-cl.
    #
    # Filament compiles itself with `/fp:fast`. Under cl.exe that means what it
    # looks like: fast arithmetic, +/-infinity still working. clang-cl maps the
    # same switch onto -ffast-math, which implies -ffinite-math-only and makes
    # infinity undefined behavior. Filament's own non-MSVC branches pass
    # -fno-finite-math-only next to -ffast-math for exactly this reason; the MSVC
    # branches have no equivalent, because under MSVC they need none.
    #
    # Filament seeds bounding boxes with +/-infinity, so left alone this compiles
    # clean and renders nothing -- and says nothing, because these same targets are
    # built `-w` a few lines above.
    #
    # Has to come after add_subdirectory(): clang takes the last floating-point
    # option on the command line, and a target's own `/fp:fast` sorts after
    # anything set through CMAKE_CXX_FLAGS or add_compile_options.
    #
    # Only finite-math is restored. Reassociation and signed zeros stay on, because
    # Filament runs with those on Linux and macOS too -- the goal is to match its
    # other platforms, not to opt out of what Filament asked for.
    # ---------------------------------------------------------------------
    if (MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # `filament` carries the bounding-box math this exists to protect. Missing it
        # means Filament was re-vendored under different target names and the list
        # below is stale -- stop, rather than produce a build that renders nothing.
        if (NOT TARGET filament)
            message(FATAL_ERROR
                "[Mochi] Filament target `filament` not found, so the /fp:fast list "
                "below is out of date. Re-derive it from Filament's CMakeLists: every "
                "target that sets /fp:fast under `if (MSVC)` needs to be here.")
        endif ()

        # Every Filament target that sets /fp:fast. C and CXX only -- bluegl assembles
        # a MASM source, and the assembler errors on a compiler flag it does not
        # recognize, the same way it does on Filament's `-fno-rtti` handled above.
        foreach (target IN ITEMS
                filament
                backend
                math
                geometry
                image
                ibl
                ibl-lite
                filament-iblprefilter
                filament-generatePrefilterMipmap
                mikktspace
                uberzlib)
            if (TARGET ${target})
                target_compile_options(${target} PRIVATE
                    $<$<COMPILE_LANGUAGE:C,CXX>:/clang:-fno-finite-math-only>)
            endif ()
        endforeach ()
    endif ()

    # ---------------------------------------------------------------------
    # Replace Filament's matc/resgen-driven material pipeline with a tiny
    # bin_to_c.py-based wiring on the pre-built blobs that ship under each
    # library's src/buck_generated/. The corresponding upstream blocks in
    # libs/iblprefilter/CMakeLists.txt, libs/gltfio/CMakeLists.txt, and
    # filament/CMakeLists.txt are commented out (the last two via an
    # `if (FALSE) # [Mochi] ...` wrapper) so this is the only place those
    # resources are produced.
    # ---------------------------------------------------------------------

    # Helper: stage a single pre-built <name>.h + run bin_to_c.py on
    # <name>.bin at CONFIGURE TIME, then expose the generated .c as a
    # separate static library that is linked into <target>. Putting the .c
    # in its own library keeps it from inheriting <target>'s per-target
    # C++-only compile options (e.g. -Wextra-semi, -Wnon-virtual-dtor)
    # which produce noisy warnings on C sources.
    # Generating at configure time (rather than via add_custom_command)
    # sidesteps the Makefile generator's limitation that custom-command
    # rules registered in one directory aren't always visible to consumer
    # targets in another. The pre-built blobs only change when Filament is
    # upgraded, so the loss of incremental regeneration is acceptable.
    function (_mochi_filament_wire_prebuilt_resource target prebuilt_dir name symbol)
        get_target_property(_bin_dir ${target} BINARY_DIR)
        set(_resource_dir "${_bin_dir}/generated/resources")
        file(MAKE_DIRECTORY "${_resource_dir}")

        configure_file(
            "${prebuilt_dir}/${name}.h"
            "${_resource_dir}/${name}.h"
            COPYONLY
        )

        set(_c "${_resource_dir}/${name}.c")
        execute_process(
            COMMAND "${Python3_EXECUTABLE}"
                "${MOCHI_FILAMENT_TOOLS_DIR}/bin_to_c.py"
                "${prebuilt_dir}/${name}.bin"
                "${_c}"
                "${symbol}"
            RESULT_VARIABLE _rc
        )
        if (NOT _rc EQUAL 0)
            message(FATAL_ERROR "[Mochi] bin_to_c.py failed for ${name} (rc=${_rc})")
        endif ()

        set(_lib "_mochi_filament_${target}_${name}_resource")
        add_library(${_lib} STATIC "${_c}")
        set_target_properties(${_lib} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            FOLDER "third_party/filament"
        )
        mochi_target_ignore_all_warnings(${_lib})
        target_link_libraries(${target} PRIVATE ${_lib})
    endfunction ()

    # iblprefilter: a single materials package.
    if (TARGET filament-iblprefilter)
        _mochi_filament_wire_prebuilt_resource(
            filament-iblprefilter
            "${MOCHI_FILAMENT_SOURCE_DIR}/libs/iblprefilter/src/buck_generated/generated/resources"
            iblprefilter_materials
            IBLPREFILTER_MATERIALS
        )
    endif ()

    # filament (main library): 13 pre-built resource packages, one per
    # src/materials/<subdir> plus the DFG LUT and the top-level `materials`
    # package. Symbol = uppercase of the package name (per bin_to_c.py).
    if (TARGET filament)
        set(_filament_resource_packages
            bloom
            colorGrading
            dfg
            dof
            evsm
            flare
            fog
            fsr
            fxaa
            materials
            sgsr
            ssao
            taa
        )
        set(_filament_prebuilt_dir "${MOCHI_FILAMENT_SOURCE_DIR}/filament/src/buck_generated/generated/resources")
        foreach (_pkg IN LISTS _filament_resource_packages)
            string(TOUPPER "${_pkg}" _sym)
            _mochi_filament_wire_prebuilt_resource(
                filament
                "${_filament_prebuilt_dir}"
                "${_pkg}"
                "${_sym}"
            )
        endforeach ()
        unset(_filament_resource_packages)
        unset(_filament_prebuilt_dir)
    endif ()

    # gltfio's `uberarchive` target. gltfio/CMakeLists.txt's own
    # `add_library(uberarchive ...)` is commented out via [Mochi] markers,
    # so we define it here. mochi_renderer includes <materials/uberarchive.h>
    # and links against the UBERARCHIVE_*_DATA symbols defined in
    # uberarchive.c, so the public include dir must place materials/ at the
    # root of the include search.
    if (NOT TARGET uberarchive)
        set(_uberarchive_bin_dir "${_mochi_filament_binary_dir}/libs/gltfio")
        set(_uberarchive_resource_dir "${_uberarchive_bin_dir}/materials")
        file(MAKE_DIRECTORY "${_uberarchive_resource_dir}")
        set(_uberarchive_prebuilt_dir "${MOCHI_FILAMENT_SOURCE_DIR}/libs/gltfio/src/buck_generated/generated/materials")
        configure_file(
            "${_uberarchive_prebuilt_dir}/uberarchive.h"
            "${_uberarchive_resource_dir}/uberarchive.h"
            COPYONLY
        )
        set(_uberarchive_c "${_uberarchive_resource_dir}/uberarchive.c")
        execute_process(
            COMMAND "${Python3_EXECUTABLE}"
                "${MOCHI_FILAMENT_TOOLS_DIR}/bin_to_c.py"
                "${_uberarchive_prebuilt_dir}/uberarchive.bin"
                "${_uberarchive_c}"
                UBERARCHIVE
            RESULT_VARIABLE _rc
        )
        if (NOT _rc EQUAL 0)
            message(FATAL_ERROR "[Mochi] bin_to_c.py failed for uberarchive (rc=${_rc})")
        endif ()
        add_library(uberarchive STATIC "${_uberarchive_c}")
        target_include_directories(uberarchive PUBLIC "${_uberarchive_bin_dir}")
        set_target_properties(uberarchive PROPERTIES
            FOLDER "third_party/filament"
            POSITION_INDEPENDENT_CODE ON
        )
        mochi_target_ignore_all_warnings(uberarchive)
        unset(_uberarchive_bin_dir)
        unset(_uberarchive_resource_dir)
        unset(_uberarchive_prebuilt_dir)
        unset(_uberarchive_c)
    endif ()

    # Folder-tag and warning-relax every Filament target Mochi may build.
    # Each call is guarded with if (TARGET ...) since the exact target set depends
    # on the platform and on FILAMENT_SUPPORTS_* options.
    set(_mochi_filament_targets
        filament
        gltfio gltfio_core
        camutils
        filameshio
        filament-iblprefilter
        image
        imageio
        imageio-lite
        ktxreader
        math mathio
        utils
        filabridge
        filaflat
        geometry
        cgltf
        meshoptimizer
        mikktspace
        smol-v
        SPIRV-Headers
        stb
        basis_encoder basis_transcoder basisu
        bluegl
        bluevk
        vkmemalloc
        png
        tinyexr
        dracodec
        tsl
        uberzlib
    )
    foreach (_t IN LISTS _mochi_filament_targets)
        if (TARGET ${_t})
            set_target_properties(${_t} PROPERTIES FOLDER "third_party/filament")
            # Skip INTERFACE targets (PRIVATE compile options not allowed).
            get_target_property(_type ${_t} TYPE)
            if (NOT _type STREQUAL "INTERFACE_LIBRARY")
                # Mochi users should not see third-party warnings.
                mochi_target_ignore_all_warnings(${_t})
            endif ()
        endif ()
    endforeach ()

    # Recursively collect all targets in Filament's source-dir subtree so
    # we can post-process them without enumerating every sub-library.
    function (_mochi_filament_collect_targets dir out_var)
        get_property(_targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
        get_property(_subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
        set(_result ${_targets})
        foreach (_s IN LISTS _subdirs)
            _mochi_filament_collect_targets("${_s}" _sub)
            list(APPEND _result ${_sub})
        endforeach ()
        set(${out_var} ${_result} PARENT_SCOPE)
    endfunction ()

    _mochi_filament_collect_targets("${MOCHI_FILAMENT_SOURCE_DIR}" _all_filament_targets)

    foreach (_t IN LISTS _all_filament_targets)
        get_target_property(_type ${_t} TYPE)
        if (_type STREQUAL "INTERFACE_LIBRARY")
            continue ()
        endif ()

        # GCC rejects `-fstrict-vtable-pointers` (Clang-only) as a hard
        # error; strip it from every Filament target's COMPILE_OPTIONS.
        # The flag is typically embedded in a `$<$<CONFIG:Release>:...>`
        # generator expression alongside other flags, so we rewrite each
        # entry rather than using list(REMOVE_ITEM ...).
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            get_target_property(_opts ${_t} COMPILE_OPTIONS)
            if (_opts)
                set(_new_opts)
                foreach (_opt IN LISTS _opts)
                    # Strip either bare or list-embedded occurrences.
                    string(REPLACE ";-fstrict-vtable-pointers" "" _opt "${_opt}")
                    string(REPLACE "-fstrict-vtable-pointers;" "" _opt "${_opt}")
                    string(REPLACE "-fstrict-vtable-pointers" "" _opt "${_opt}")
                    if (NOT _opt STREQUAL "")
                        list(APPEND _new_opts "${_opt}")
                    endif ()
                endforeach ()
                set_target_properties(${_t} PROPERTIES COMPILE_OPTIONS "${_new_opts}")
            endif ()
        endif ()

        # Filament applies MSVC cl-only flags (e.g. /MT, /MTd) via unguarded
        # add_compile_options() at its directory scope; these get snapshotted
        # into each target's COMPILE_OPTIONS and are then forwarded to the MASM
        # assembler (ml64) for bluegl's .S file, which rejects them with
        # "A4018: invalid command-line option". Re-scope every per-target
        # compile option to C/CXX only (they are meaningless to the assembler)
        # so ml64 receives none of them. Entries already carrying a
        # COMPILE_LANGUAGE guard are left untouched, as are warning-level flags
        # (/w, /Wn): CMake folds those into the C/C++-only <WarningLevel>
        # element (which never reaches MASM), and wrapping them in a genex would
        # prevent that folding and reintroduce "D9025: overriding '/W3'".
        if (MSVC)
            get_target_property(_opts ${_t} COMPILE_OPTIONS)
            if (_opts)
                set(_new_opts)
                foreach (_opt IN LISTS _opts)
                    if (_opt MATCHES "COMPILE_LANGUAGE" OR _opt MATCHES "^/[Ww][0-4]?$")
                        list(APPEND _new_opts "${_opt}")
                    else ()
                        list(APPEND _new_opts "$<$<COMPILE_LANGUAGE:C,CXX>:${_opt}>")
                    endif ()
                endforeach ()
                set_target_properties(${_t} PROPERTIES COMPILE_OPTIONS "${_new_opts}")
                unset(_new_opts)
            endif ()
        endif ()

        # mochi_renderer is built as a SHARED lib when MOCHI_BUILD_SHARED is
        # ON. Any Filament static lib it transitively links must be PIC.
        if (_type STREQUAL "STATIC_LIBRARY")
            set_target_properties(${_t} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif ()

        # Mochi users should not see warnings from Filament headers when
        # compiling their own code (e.g. mochi_renderer). Mark Filament's
        # exported include directories as SYSTEM so consumer compilations
        # silence diagnostics originating from those headers.
        get_target_property(_inc_dirs ${_t} INTERFACE_INCLUDE_DIRECTORIES)
        if (_inc_dirs)
            set_target_properties(${_t}
                PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_inc_dirs}"
            )
        endif ()

        # Silence warnings for every Filament-subtree target. The
        # directory-level add_compile_options(-w) above only catches
        # compile flags inherited from the directory scope; per-target
        # warning flags set by Filament can re-enable noise, so apply -w
        # PRIVATE here too as a belt-and-suspenders.
        mochi_target_ignore_all_warnings(${_t})
    endforeach ()

    # Filament headers contain constructs that GCC rejects without
    # -fpermissive (e.g. the self-naming
    # `using PoolAllocator = PoolAllocator<...>` in utils/Allocator.h).
    # Filament's own translation units already pick up the flag via the
    # directory-level add_compile_options(-fpermissive) above, but every
    # consumer that #includes Filament headers (mochi_renderer,
    # mochi_renderer_test, ...) needs it too. Add it as
    # an INTERFACE compile option on every Filament target so it propagates
    # automatically through target_link_libraries. INTERFACE works for
    # INTERFACE_LIBRARY targets too, so this loop has no early skip.
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        foreach (_t IN LISTS _all_filament_targets)
            target_compile_options(${_t} INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:-fpermissive>
            )
        endforeach ()
    endif ()

    unset(_all_filament_targets)
    unset(_mochi_filament_targets)
    unset(_mochi_filament_binary_dir)

endif ()
