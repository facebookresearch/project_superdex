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

# Rerun forwards the toolchain file to Arrow's separate CMake configuration.
if(WIN32 AND NOT CMAKE_TOOLCHAIN_FILE)
    if(IS_ABSOLUTE "${CMAKE_RC_COMPILER}" AND EXISTS "${CMAKE_RC_COMPILER}")
        set(_superdex_arrow_rc "${CMAKE_RC_COMPILER}")
    else()
        get_filename_component(_superdex_compiler_dir "${CMAKE_C_COMPILER}" DIRECTORY)
        find_program(_superdex_arrow_rc NAMES llvm-rc rc
            HINTS "${_superdex_compiler_dir}" NO_CACHE REQUIRED)
    endif()
    configure_file(
        "${CMAKE_CURRENT_LIST_DIR}/cmake/arrow_toolchain.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/arrow_toolchain.cmake"
        @ONLY
    )
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_BINARY_DIR}/arrow_toolchain.cmake")
    set(_superdex_generated_arrow_toolchain ON)
endif()

FetchContent_Declare(
    rerun_sdk
    URL https://github.com/rerun-io/rerun/releases/download/0.33.0/rerun_cpp_sdk.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    FIND_PACKAGE_ARGS 0.33.0 EXACT CONFIG
)
if(WIN32)
    set(_superdex_original_rerun_project_include "${CMAKE_PROJECT_rerun_sdk_INCLUDE}")
    set(CMAKE_PROJECT_rerun_sdk_INCLUDE "${CMAKE_CURRENT_LIST_DIR}/cmake/configure_rerun.cmake")
endif()
FetchContent_MakeAvailable(rerun_sdk)
if(WIN32)
    set(CMAKE_PROJECT_rerun_sdk_INCLUDE "${_superdex_original_rerun_project_include}")
endif()

# CMake's cached system configuration can skip a newly supplied toolchain file.
if(_superdex_generated_arrow_toolchain AND TARGET arrow_cpp)
    block()
        ExternalProject_Get_Property(arrow_cpp BINARY_DIR)
        if(EXISTS "${BINARY_DIR}/CMakeCache.txt")
            load_cache("${BINARY_DIR}" READ_WITH_PREFIX _arrow_cache_
                CMAKE_C_COMPILER CMAKE_CXX_COMPILER CMAKE_RC_COMPILER)
            if("${_arrow_cache_CMAKE_C_COMPILER}" MATCHES "-NOTFOUND$" OR
                "${_arrow_cache_CMAKE_CXX_COMPILER}" MATCHES "-NOTFOUND$" OR
                (DEFINED _arrow_cache_CMAKE_RC_COMPILER AND
                    NOT "${_arrow_cache_CMAKE_RC_COMPILER}" STREQUAL "${_superdex_arrow_rc}"))
                message(STATUS "[SuperDex] Resetting Arrow's stale toolchain cache")
                file(REMOVE "${BINARY_DIR}/CMakeCache.txt")
                file(REMOVE_RECURSE "${BINARY_DIR}/CMakeFiles")
            endif()
        endif()
    endblock()
endif()

# The episode reader decompresses Rerun's LZ4 blocks with the reference decoder, lib/lz4.c.
FetchContent_Declare(
    lz4
    URL https://github.com/lz4/lz4/archive/refs/tags/v1.9.4.tar.gz
    URL_HASH SHA256=0b0e3aa07c8c063ddf40b082bdf7e37a1562bda40a0ff5272957f3e987e0e54b
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    # No CMakeLists.txt here, so the sources are fetched without adding lz4's own project.
    SOURCE_SUBDIR lib
)
FetchContent_MakeAvailable(lz4)
add_library(superdex_recorder_lz4 STATIC "${lz4_SOURCE_DIR}/lib/lz4.c")
target_include_directories(superdex_recorder_lz4 SYSTEM PUBLIC "${lz4_SOURCE_DIR}/lib")
set_target_properties(superdex_recorder_lz4 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(superdex_recorder_core STATIC
    src/link_kinematics.cpp
    src/series.cpp
    src/recorder.cpp
    src/source_recorder.cpp
    include/superdex_recorder/link_kinematics.h
    include/superdex_recorder/recorder.h
    include/superdex_recorder/series.h
    include/superdex_recorder/source_recorder.h
)
target_include_directories(superdex_recorder_core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(superdex_recorder_core PUBLIC mochi_core rerun_sdk)
if (WIN32)
    target_link_libraries(superdex_recorder_core PRIVATE bcrypt)
endif ()
target_compile_features(superdex_recorder_core PUBLIC cxx_std_17)
mochi_set_compiler_options(superdex_recorder_core)
set_target_properties(superdex_recorder_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(superdex_recorder_physics STATIC
    src/contact_collector.cpp
    src/episode_restorer.cpp
    src/physics_recorder.cpp
    include/superdex_recorder/contact_collector.h
    include/superdex_recorder/episode_restorer.h
    include/superdex_recorder/physics_recorder.h
)
target_include_directories(superdex_recorder_physics PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(superdex_recorder_physics PUBLIC superdex_recorder_core superdex_recorder_reader mochi_physics)
target_compile_features(superdex_recorder_physics PUBLIC cxx_std_17)
mochi_set_compiler_options(superdex_recorder_physics)
set_target_properties(superdex_recorder_physics PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Reads episodes without Rerun: it needs only the Arrow library that the Rerun SDK builds.
add_library(superdex_recorder_reader STATIC
    src/episode_reader.cpp
    src/rrd_chunks.cpp
    src/rrd_chunks.h
    include/superdex_recorder/episode_reader.h
    include/superdex_recorder/formatted_error.h
)
target_include_directories(superdex_recorder_reader PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(superdex_recorder_reader PUBLIC mochi_core PRIVATE rerun_arrow_target superdex_recorder_lz4)
target_compile_features(superdex_recorder_reader PUBLIC cxx_std_17)
mochi_set_compiler_options(superdex_recorder_reader)
set_target_properties(superdex_recorder_reader PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(superdex_recorder INTERFACE)
target_sources(superdex_recorder INTERFACE include/superdex_recorder/superdex_recorder.h)
target_include_directories(superdex_recorder INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(superdex_recorder INTERFACE superdex_recorder_physics superdex_recorder_reader)

if (TARGET superdex_robotics)
    add_library(superdex_recorder_robotics STATIC
        src/robot_shadow.cpp
        src/robotics_recorder.cpp
        src/skeleton_shadow.cpp
        include/superdex_recorder/robot_shadow.h
        include/superdex_recorder/robotics_recorder.h
        include/superdex_recorder/skeleton_shadow.h
    )
    target_include_directories(superdex_recorder_robotics PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    target_link_libraries(superdex_recorder_robotics PUBLIC superdex_recorder_core superdex_robotics)
    target_compile_features(superdex_recorder_robotics PUBLIC cxx_std_17)
    mochi_set_compiler_options(superdex_recorder_robotics)
    set_target_properties(superdex_recorder_robotics PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_compile_definitions(superdex_recorder INTERFACE SUPERDEX_RECORDER_WITH_ROBOTICS=1)
    target_link_libraries(superdex_recorder INTERFACE superdex_recorder_robotics)
endif ()

# One shared library with the whole recorder, for the Unreal SuperDexRecorder module (it links
# superdex_recorder_shared.dll on Win64). Mirrors the Buck target of the same name. Windows only:
# that is the only platform the Unreal module links it on.
if (WIN32 AND MOCHI_BUILD_SHARED AND TARGET superdex_recorder_robotics)
    add_library(superdex_recorder_shared SHARED
        src/contact_collector.cpp
        src/link_kinematics.cpp
        src/physics_recorder.cpp
        src/recorder.cpp
        src/robot_shadow.cpp
        src/robotics_recorder.cpp
        src/series.cpp
        src/skeleton_shadow.cpp
        src/source_recorder.cpp
    )
    target_include_directories(superdex_recorder_shared PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    target_link_libraries(superdex_recorder_shared
        PUBLIC mochi_core mochi_physics superdex_robotics
        PRIVATE rerun_sdk bcrypt
    )
    target_compile_definitions(superdex_recorder_shared
        PUBLIC SUPERDEX_RECORDER_SHARED=1 SUPERDEX_RECORDER_WITH_ROBOTICS=1
        PRIVATE SUPERDEX_RECORDER_EXPORTS=1
    )
    target_compile_features(superdex_recorder_shared PUBLIC cxx_std_17)
    mochi_set_compiler_options(superdex_recorder_shared)
    mochi_set_output_directories(superdex_recorder_shared)
endif ()

if (MOCHI_BUILD_TESTS)
    add_executable(superdex_recorder_core_test test/superdex_recorder_test.cpp)
    target_link_libraries(superdex_recorder_core_test PRIVATE superdex_recorder_core gtest_main)
    mochi_set_compiler_options(superdex_recorder_core_test)
    mochi_set_output_directories(superdex_recorder_core_test)
    add_test(NAME superdex_recorder_core_test COMMAND superdex_recorder_core_test)

    add_executable(superdex_recorder_physics_test test/physics_recorder_test.cpp)
    target_link_libraries(
        superdex_recorder_physics_test
        PRIVATE superdex_recorder_physics superdex_physics gtest_main
    )
    mochi_set_compiler_options(superdex_recorder_physics_test)
    mochi_set_output_directories(superdex_recorder_physics_test)
    add_test(NAME superdex_recorder_physics_test COMMAND superdex_recorder_physics_test)

    add_executable(superdex_recorder_restorer_test test/episode_restorer_test.cpp)
    target_link_libraries(
        superdex_recorder_restorer_test
        PRIVATE superdex_recorder_physics superdex_physics gtest_main
    )
    mochi_set_compiler_options(superdex_recorder_restorer_test)
    mochi_set_output_directories(superdex_recorder_restorer_test)
    add_test(NAME superdex_recorder_restorer_test COMMAND superdex_recorder_restorer_test)

    add_executable(superdex_recorder_reader_test test/episode_reader_test.cpp)
    target_link_libraries(
        superdex_recorder_reader_test
        PRIVATE superdex_recorder_core superdex_recorder_reader gtest_main
    )
    mochi_set_compiler_options(superdex_recorder_reader_test)
    mochi_set_output_directories(superdex_recorder_reader_test)
    add_test(NAME superdex_recorder_reader_test COMMAND superdex_recorder_reader_test)

    if (TARGET superdex_recorder_robotics)
        add_executable(superdex_recorder_robotics_test test/robotics_recorder_test.cpp)
        target_link_libraries(
            superdex_recorder_robotics_test
            PRIVATE superdex_recorder_robotics gtest_main
        )
        mochi_set_compiler_options(superdex_recorder_robotics_test)
        mochi_set_output_directories(superdex_recorder_robotics_test)
        add_test(NAME superdex_recorder_robotics_test COMMAND superdex_recorder_robotics_test)
    endif ()
endif ()

if (MOCHI_USE_PYBIND AND TARGET superdex_recorder_robotics)
    add_subdirectory(pybind)
endif ()
