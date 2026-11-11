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

include(FetchContent)

#-----------------------------------------------------------------------------
# zenoh-pico (transport)
#
# Added with its own CMake project, which configures zenoh-pico/config.h and provides zenohpico::lib.
# Its settings are scoped to this function so they don't leak into the including project.
#-----------------------------------------------------------------------------

function(_teleop_node_zenoh_add_zenoh_pico)
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_EXAMPLES OFF)
    set(BUILD_TESTING OFF)
    set(BUILD_TOOLS OFF)
    set(BUILD_INTEGRATION OFF)
    set(PACKAGING OFF)
    # Publishers and subscribers in the same session must see each other, e.g. an operator and a
    # robot in one process.
    set(Z_FEATURE_LOCAL_SUBSCRIBER 1)
    # zenoh-pico requires only CMake 3.14, which would leave CMP0126 OLD and let its
    # set(... CACHE ...) defaults discard the values above.
    set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)

    if (SUPERDEX_ZENOH_PICO_SOURCE_DIR)
        message(STATUS "[TeleopNodeZenoh] Using zenoh-pico from: ${SUPERDEX_ZENOH_PICO_SOURCE_DIR}")
        FetchContent_Declare(zenohpico SOURCE_DIR "${SUPERDEX_ZENOH_PICO_SOURCE_DIR}")
    else ()
        message(STATUS "[TeleopNodeZenoh] Fetching zenoh-pico ${SUPERDEX_ZENOH_PICO_GIT_TAG}")
        find_package(Git REQUIRED)
        FetchContent_Declare(
            zenohpico
            GIT_REPOSITORY "${SUPERDEX_ZENOH_PICO_GIT_REPOSITORY}"
            GIT_TAG "${SUPERDEX_ZENOH_PICO_GIT_TAG}"
            PATCH_COMMAND
                "${CMAKE_COMMAND}"
                -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
                -DPATCH_DIR=${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../ThirdParty/zenoh-pico/patches
                -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/apply_patches.cmake"
        )
    endif ()
    FetchContent_MakeAvailable(zenohpico)
    set(TELEOP_NODE_ZENOH_ZENOH_PICO_SOURCE_DIR "${zenohpico_SOURCE_DIR}" PARENT_SCOPE)
    set(TELEOP_NODE_ZENOH_ZENOH_PICO_CONFIG_H "${zenohpico_BINARY_DIR}/include/zenoh-pico/config.h" PARENT_SCOPE)
endfunction()

_teleop_node_zenoh_add_zenoh_pico()

#-----------------------------------------------------------------------------
# nanopb (protobuf encoding)
#
# Only the C runtime is used: teleop.pb.c/.h are checked in, so neither nanopb's generator nor its
# CMake project is needed.
#-----------------------------------------------------------------------------

set(TELEOP_NODE_ZENOH_NANOPB_SOURCE_DIR "${SUPERDEX_NANOPB_SOURCE_DIR}")
if (NOT TELEOP_NODE_ZENOH_NANOPB_SOURCE_DIR)
    message(STATUS "[TeleopNodeZenoh] Fetching nanopb ${SUPERDEX_NANOPB_GIT_TAG}")
    FetchContent_Declare(
        nanopb
        GIT_REPOSITORY "${SUPERDEX_NANOPB_GIT_REPOSITORY}"
        GIT_TAG "${SUPERDEX_NANOPB_GIT_TAG}"
        # Download only; don't add nanopb's own CMake project.
        SOURCE_SUBDIR "do-not-configure"
    )
    FetchContent_MakeAvailable(nanopb)
    set(TELEOP_NODE_ZENOH_NANOPB_SOURCE_DIR "${nanopb_SOURCE_DIR}")
else ()
    message(STATUS "[TeleopNodeZenoh] Using nanopb from: ${TELEOP_NODE_ZENOH_NANOPB_SOURCE_DIR}")
endif ()
if (NOT EXISTS "${TELEOP_NODE_ZENOH_NANOPB_SOURCE_DIR}/pb_encode.c")
    message(FATAL_ERROR "[TeleopNodeZenoh] nanopb runtime not found at: ${TELEOP_NODE_ZENOH_NANOPB_SOURCE_DIR}")
endif ()
