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

if(_superdex_original_rerun_project_include)
    include("${_superdex_original_rerun_project_include}")
endif()

function(_superdex_add_arrow_bundling_patch)
    if(TARGET arrow_cpp)
        ExternalProject_Add_Step(arrow_cpp superdex_clang_cl_bundle_patch
            COMMAND "${CMAKE_COMMAND}" "-DARROW_SOURCE_DIR=<SOURCE_DIR>"
                -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patch_arrow_bundling.cmake"
            DEPENDS "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patch_arrow_bundling.cmake"
            DEPENDEES patch
            DEPENDERS configure
        )
    endif()
endfunction()

cmake_language(DEFER CALL _superdex_add_arrow_bundling_patch)
