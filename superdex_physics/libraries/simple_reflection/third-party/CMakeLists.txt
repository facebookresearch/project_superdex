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

#-----------------------------------------------------------------------------
# cityhash
#-----------------------------------------------------------------------------

add_subdirectory(cityhash)

# cityhash is vendored: its warnings are not actionable, and consumers should not see them.
if (MSVC)
    target_compile_options(cityhash PRIVATE /W0)
else ()
    target_compile_options(cityhash PRIVATE -w)
endif ()

#-----------------------------------------------------------------------------
# GTest
#-----------------------------------------------------------------------------

if (SR_BUILD_TESTS AND NOT TARGET gtest AND SR_BUILD_GTEST_FROM_SOURCE)
    set(BUILD_GMOCK OFF)
    FetchContent_Declare(GTest SOURCE_DIR ${SR_GTEST_SOURCE_DIR})
    FetchContent_MakeAvailable(GTest)

    # Work-around for the fact that gtest-port.cc is missing a required include for <thread>.
    # This is a bug in version 1.14.0 of GTest.
    if(MSVC)
        target_compile_options(gtest PRIVATE /FI"thread")
        target_compile_options(gtest_main PRIVATE /FI"thread")
    else()
        target_compile_options(gtest PRIVATE -include thread)
        target_compile_options(gtest_main PRIVATE -include thread)
    endif()

endif ()

#-----------------------------------------------------------------------------
# nlohmann_json
#-----------------------------------------------------------------------------

if (SR_USE_NLOHMANN_JSON AND NOT TARGET nlohmann_json)
    # Store original warning state
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS_BEFORE ${CMAKE_SUPPRESS_DEVELOPER_WARNINGS})

    # Disable warnings without CACHE to avoid persistence
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON)

    FetchContent_Declare(nlohmann_json SOURCE_DIR ${SR_NLOHMANN_JSON_SOURCE_DIR})
    FetchContent_MakeAvailable(nlohmann_json)

    # Restore original warning state
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ${CMAKE_SUPPRESS_DEVELOPER_WARNINGS_BEFORE})
endif ()

#-----------------------------------------------------------------------------
# picojson
#-----------------------------------------------------------------------------

# Optionally build picojson from source
if (NOT TARGET picojson)
    add_library(picojson INTERFACE)
    target_sources(picojson INTERFACE "${SR_PICOJSON_SOURCE_DIR}/picojson/picojson.h")
    target_include_directories(picojson INTERFACE "${SR_PICOJSON_SOURCE_DIR}")
endif ()
