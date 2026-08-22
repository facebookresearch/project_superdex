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

project(imguios)

#-----------------------------------------------------------------------------
# Options
#-----------------------------------------------------------------------------

option(IMGUIOS_BUILD_EXAMPLE "Build example app" OFF)
option(IMGUIOS_BUILD_METAL "Build Metal backend (macOS only)" OFF)

#-----------------------------------------------------------------------------
# Dependencies
#-----------------------------------------------------------------------------

add_subdirectory(third-party)

#-----------------------------------------------------------------------------
# imguios library
#-----------------------------------------------------------------------------

add_library(imguios STATIC)

#file(GLOB_RECURSE lib_srcs "${lib_dir}/src/*.cpp" "${lib_dir}/src/*.c" "${lib_dir}/src/*.h")
#file(GLOB_RECURSE lib_incs "${lib_dir}/include/*.h")

target_sources(imguios
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/application.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/application_mcp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/common.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/imguios_window.cpp
        PUBLIC FILE_SET HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include/
        FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/imguios/application.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/imguios/common.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/imguios/gpu_selector.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/imguios/imguios.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/imguios/imguios_window.h
)

target_compile_features(imguios PRIVATE cxx_std_20)
target_include_directories(imguios PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(imguios PUBLIC imgui implot glad glfw imguios_fonts)

# On Linux, imguios_window.cpp calls XSetErrorHandler directly (non-fatal X11 error handler
# for MIT-SHM BadMatch on shared GLX contexts). This requires linking against X11.
if (UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)
    target_link_libraries(imguios PRIVATE X11::X11)
endif()

#-----------------------------------------------------------------------------
# imguios_metal library (macOS only)
#-----------------------------------------------------------------------------

if (APPLE AND IMGUIOS_BUILD_METAL)
    add_library(imguios_metal STATIC)

    target_sources(imguios_metal
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/application_metal.mm
            ${CMAKE_CURRENT_SOURCE_DIR}/src/application_mcp.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/common.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/imguios_window.cpp
            PUBLIC FILE_SET HEADERS
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include/
            FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/include/imguios/application.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/imguios/common.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/imguios/gpu_selector.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/imguios/imguios.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/imguios/imguios_window.h
    )

    target_compile_features(imguios_metal PRIVATE cxx_std_20)
    target_include_directories(imguios_metal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(imguios_metal PUBLIC IMGUIOS_USE_METAL=1)
    target_link_libraries(imguios_metal PUBLIC imgui_metal implot glfw imguios_fonts)
    target_link_libraries(imguios_metal PUBLIC "-framework Metal" "-framework QuartzCore")
endif()

#-----------------------------------------------------------------------------
# imguios_example_epp
#-----------------------------------------------------------------------------

if (IMGUIOS_BUILD_EXAMPLE)
    add_subdirectory(example)
endif()
