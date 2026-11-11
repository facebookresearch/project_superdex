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

# If "paths.fb.cmake" exists, then import it first
if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/paths.fb.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/paths.fb.cmake")
endif ()

#-----------------------------------------------------------------------------
# Third Party Dependencies:
#
#   Each dependency is downloaded at a pinned commit unless its SUPERDEX_XXX_SOURCE_DIR points at a
#   local copy of that version. To use a mirror, override SUPERDEX_XXX_GIT_REPOSITORY.
#
#-----------------------------------------------------------------------------

set(SUPERDEX_ZENOH_PICO_SOURCE_DIR "" CACHE PATH
    "zenoh-pico 1.10.1 source tree to use instead of downloading it; the patches in ThirdParty/zenoh-pico/patches must already be applied")
set(SUPERDEX_ZENOH_PICO_GIT_REPOSITORY "https://github.com/eclipse-zenoh/zenoh-pico.git" CACHE STRING
    "Git repository zenoh-pico is downloaded from")
set(SUPERDEX_ZENOH_PICO_GIT_TAG "e1ab223a28aaebb5dec1e70d98eab152332f777a" CACHE STRING
    "zenoh-pico commit to download (1.10.1)")

set(SUPERDEX_NANOPB_SOURCE_DIR "" CACHE PATH
    "nanopb 0.4.9.1 source tree (holding pb.h and pb_*.c) to use instead of downloading it")
set(SUPERDEX_NANOPB_GIT_REPOSITORY "https://github.com/nanopb/nanopb.git" CACHE STRING
    "Git repository nanopb is downloaded from")
set(SUPERDEX_NANOPB_GIT_TAG "cad3c18ef15a663e30e3e43e3a752b66378adec1" CACHE STRING
    "nanopb commit to download (0.4.9.1)")
