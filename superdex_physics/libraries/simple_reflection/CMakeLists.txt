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

project(simple_reflection)

#-----------------------------------------------------------------------------
# Settings
#-----------------------------------------------------------------------------

# Set third-party paths if building from fbsource (not github)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/settings.fb.cmake")
    include(cmake/settings.fb.cmake)
endif ()

include(cmake/settings.cmake)

#-----------------------------------------------------------------------------
# Third Party Dependencies
#-----------------------------------------------------------------------------

if (SR_BUILD_TESTS)
    if (NOT SR_BUILD_GTEST_FROM_SOURCE)
        find_package(GTest 1.14.0 REQUIRED)
    endif ()
endif ()

add_subdirectory(third-party)

#-----------------------------------------------------------------------------
# simple_reflection library
#-----------------------------------------------------------------------------

add_library(simple_reflection STATIC)

target_sources(simple_reflection
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simple_reflection.cpp
    PUBLIC FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include/
    FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/simple_reflection/simple_reflection.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/simple_reflection/simple_reflection_inl.h
)

target_link_libraries(simple_reflection PRIVATE cityhash PUBLIC picojson)
target_compile_features(simple_reflection PRIVATE cxx_std_17)

if (SR_USE_NLOHMANN_JSON)
    target_compile_definitions(simple_reflection PUBLIC "SR_USE_NLOHMANN_JSON=1")
    target_link_libraries(simple_reflection PUBLIC nlohmann_json)
else ()
    target_compile_definitions(simple_reflection PUBLIC "SR_USE_NLOHMANN_JSON=0")
endif ()

#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------

if (SR_BUILD_TESTS)
    add_executable(simple_reflection_test test/simple_reflection_test.cpp test/type_definitions.cpp)
    target_link_libraries(simple_reflection_test PRIVATE simple_reflection gtest_main)
    target_compile_features(simple_reflection_test PRIVATE cxx_std_17)
endif ()
