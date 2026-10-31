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

FetchContent_Declare(
    rerun_sdk
    URL https://github.com/rerun-io/rerun/releases/download/0.33.0/rerun_cpp_sdk.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    FIND_PACKAGE_ARGS 0.33.0 EXACT CONFIG
)
FetchContent_MakeAvailable(rerun_sdk)

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
    include/superdex_recorder/link_kinematics.h
    include/superdex_recorder/recorder.h
    include/superdex_recorder/series.h
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
