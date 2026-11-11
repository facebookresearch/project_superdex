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

foreach (_include IN LISTS _superdex_rerun_project_include)
    include("${_include}")
endforeach ()

function(_superdex_patch_arrow_libtool)
    if (TARGET arrow_cpp)
        include(ExternalProject)
        ExternalProject_Get_Property(arrow_cpp SOURCE_DIR)
        ExternalProject_Add_Step(arrow_cpp superdex_apple_libtool_compat
            COMMAND "${CMAKE_COMMAND}" "-DARROW_SOURCE_DIR=${SOURCE_DIR}"
                -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patch_arrow_libtool.cmake"
            DEPENDEES patch
            DEPENDERS configure
        )
    endif ()
endfunction()

cmake_language(DEFER CALL _superdex_patch_arrow_libtool)
