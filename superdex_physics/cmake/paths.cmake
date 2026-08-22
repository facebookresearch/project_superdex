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
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/paths.fb.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/paths.fb.cmake")
endif()

#-----------------------------------------------------------------------------
# imguios
#-----------------------------------------------------------------------------

cmake_path(SET _superdex_imguios_default NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../libraries/imguios")
set(SUPERDEX_IMGUIOS_SOURCE_DIR "${_superdex_imguios_default}" CACHE PATH "Source directory for building imguios")
unset(_superdex_imguios_default)

#-----------------------------------------------------------------------------
# mochi
#-----------------------------------------------------------------------------

cmake_path(SET _superdex_mochi_default NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../libraries/mochi")
set(SUPERDEX_MOCHI_SOURCE_DIR "${_superdex_mochi_default}" CACHE PATH "Source directory for building mochi")
unset(_superdex_mochi_default)


#-----------------------------------------------------------------------------
# simple_reflection
#-----------------------------------------------------------------------------

cmake_path(SET _superdex_simple_reflection_default NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../libraries/simple_reflection")
set(SUPERDEX_SIMPLE_REFLECTION_SOURCE_DIR "${_superdex_simple_reflection_default}" CACHE PATH "Source directory for building simple_reflection")
unset(_superdex_simple_reflection_default)
