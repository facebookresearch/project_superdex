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

# Puts zenoh-pico and nanopb's C runtime into this plugin's modules, so Unreal can build it:
#
#   cmake -S <this directory> -B <build directory>
#
# Configuring is enough; see README.md. Also provides TeleopNodeZenoh::teleop_msgs (teleop.proto's
# nanopb output and teleop_codec.h) and zenohpico::lib for code outside Unreal, e.g. on the robot.

cmake_policy(VERSION 3.25)
cmake_minimum_required(VERSION 3.25)

project(TeleopNodeZenoh LANGUAGES C CXX)

option(TELEOP_NODE_ZENOH_PLACE_SOURCES "Place zenoh-pico and nanopb into the plugin's modules" ON)

include("${CMAKE_CURRENT_LIST_DIR}/cmake/paths.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/dependencies.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/stage_sources.cmake")

if (TELEOP_NODE_ZENOH_PLACE_SOURCES)
    teleop_node_zenoh_stage_zenoh_pico(
        SOURCE_DIR "${TELEOP_NODE_ZENOH_ZENOH_PICO_SOURCE_DIR}"
        CONFIG_H "${TELEOP_NODE_ZENOH_ZENOH_PICO_CONFIG_H}"
        DEST "${CMAKE_CURRENT_SOURCE_DIR}/Source/TeleopNodeZenoh/Private/ZenohPico"
    )
    teleop_node_zenoh_stage_nanopb(
        SOURCE_DIR "${TELEOP_NODE_ZENOH_NANOPB_SOURCE_DIR}"
        DEST "${CMAKE_CURRENT_SOURCE_DIR}/Source/TeleopNodeZenoh/Private/Nanopb"
    )
endif ()

#-----------------------------------------------------------------------------
# TeleopNodeZenoh::teleop_msgs: the teleop messages and their nanopb codec
#-----------------------------------------------------------------------------

set(_teleop_msgs_dir "${CMAKE_CURRENT_SOURCE_DIR}/Source/TeleopNodeZenoh/Private/TeleopMsgs")
add_library(teleop_node_zenoh_msgs STATIC
    "${_teleop_msgs_dir}/generated/teleop.pb.c"
    "${TELEOP_NODE_ZENOH_NANOPB_SOURCE_DIR}/pb_common.c"
    "${TELEOP_NODE_ZENOH_NANOPB_SOURCE_DIR}/pb_decode.c"
    "${TELEOP_NODE_ZENOH_NANOPB_SOURCE_DIR}/pb_encode.c"
)
add_library(TeleopNodeZenoh::teleop_msgs ALIAS teleop_node_zenoh_msgs)
target_include_directories(teleop_node_zenoh_msgs PUBLIC
    "${_teleop_msgs_dir}"
    "${_teleop_msgs_dir}/generated"
)
target_include_directories(teleop_node_zenoh_msgs SYSTEM PUBLIC "${TELEOP_NODE_ZENOH_NANOPB_SOURCE_DIR}")
# For teleop_codec.h.
target_compile_features(teleop_node_zenoh_msgs INTERFACE cxx_std_17)
unset(_teleop_msgs_dir)
