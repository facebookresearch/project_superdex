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

# Applies every *.patch in PATCH_DIR, in name order, to the source tree in the current directory.
# Run as a FetchContent PATCH_COMMAND, which can run again on an already patched tree, so patches
# that are already applied are skipped.
#
#   cmake -DGIT_EXECUTABLE=<git> -DPATCH_DIR=<dir> -P apply_patches.cmake

file(GLOB _patches "${PATCH_DIR}/*.patch")
list(SORT _patches)
foreach (_patch IN LISTS _patches)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${_patch}"
        RESULT_VARIABLE _already_applied
        OUTPUT_QUIET ERROR_QUIET)
    if (_already_applied EQUAL 0)
        continue ()
    endif ()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --whitespace=nowarn "${_patch}"
        RESULT_VARIABLE _result)
    if (NOT _result EQUAL 0)
        message(FATAL_ERROR "Failed to apply ${_patch}")
    endif ()
    message(STATUS "Applied ${_patch}")
endforeach ()
