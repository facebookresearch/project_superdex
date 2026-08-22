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

if (NOT DEFINED OCCT_SOURCE_DIR)
    message(FATAL_ERROR "OCCT_SOURCE_DIR must be set")
endif ()

if (MOCHI_OCCT_TOUCH_CLANG_FORMAT)
    file(TOUCH "${OCCT_SOURCE_DIR}/.clang-format")
endif ()

if (WIN32)
    set(_occt_toolkit_cmake "${OCCT_SOURCE_DIR}/adm/cmake/occt_toolkit.cmake")
    file(READ "${_occt_toolkit_cmake}" _occt_toolkit_contents)
    string(REPLACE
        "if (MSVC)\n  set (USED_RCFILE"
        "# Modified by SuperDex on 2026-08-17\nif (MSVC AND BUILD_SHARED_LIBS)\n  set (USED_RCFILE"
        _occt_toolkit_contents
        "${_occt_toolkit_contents}")
    file(WRITE "${_occt_toolkit_cmake}" "${_occt_toolkit_contents}")

    # OCCT sets /EHa (async exception handling) instead of MSVC's default /EHsc.
    # Under clang-cl this causes extremely slow compiles because every instruction
    # becomes an exception-handling point, blowing up LLVM's WinEHPrepare pass.
    #
    # The cost is that an access violation is no longer converted to an OCCT Standard_Failure;
    # it kills the process.
    set(_occt_defs_flags_cmake "${OCCT_SOURCE_DIR}/adm/cmake/occt_defs_flags.cmake")
    file(READ "${_occt_defs_flags_cmake}" _occt_defs_flags_contents)
    string(REPLACE
        "string (REGEX REPLACE \"EHsc\" \"EHa\" CMAKE_CXX_FLAGS \"\${CMAKE_CXX_FLAGS}\")"
        "# Modified by SuperDex on 2026-08-17: keep /EHsc; /EHa is pathologically slow under clang-cl."
        _occt_defs_flags_contents
        "${_occt_defs_flags_contents}")
    string(REPLACE
        "set (CMAKE_CXX_FLAGS \"\${CMAKE_CXX_FLAGS} /EHa\")"
        "set (CMAKE_CXX_FLAGS \"\${CMAKE_CXX_FLAGS} /EHsc\")"
        _occt_defs_flags_contents
        "${_occt_defs_flags_contents}")
    file(WRITE "${_occt_defs_flags_cmake}" "${_occt_defs_flags_contents}")
endif ()
