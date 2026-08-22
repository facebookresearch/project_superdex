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
# imguios
#-----------------------------------------------------------------------------

cmake_path(SET _mochi_imguios_default NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../utils/imguios")
set(MOCHI_IMGUIOS_SOURCE_DIR "${_mochi_imguios_default}" CACHE PATH "Source directory for building imguios")
unset(_mochi_imguios_default)

#-----------------------------------------------------------------------------
# simple_reflection
#-----------------------------------------------------------------------------

cmake_path(SET _mochi_simple_reflection_default NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../utils/simple_reflection")
set(MOCHI_SIMPLE_REFLECTION_SOURCE_DIR "${_mochi_simple_reflection_default}" CACHE PATH "Source directory for building simple_reflection")
unset(_mochi_simple_reflection_default)

#-----------------------------------------------------------------------------
# Third Party Dependencies:
#
#   Mochi includes a copy of most of the third-party libraries that it uses.
#   By default, those libraries will be built form the provided source.
#
#   If you want to build from source in a different location, then override the corresponding
#   MOCHI_XXX_SOURCE_DIR variable (replace XXX with the library name).
#
#   If you want to use find_package to locate a pre-installed copy of that library, then
#   override the corresponding MOCHI_XXX_FIND_PACKAGE option (replace XXX with the library name).
#
#-----------------------------------------------------------------------------

cmake_path(SET _mochi_third_party NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../third_party")

option(MOCHI_BENCHMARK_FIND_PACKAGE "Use find_package for benchmark, rather than building from source" OFF)
set(MOCHI_BENCHMARK_SOURCE_DIR "${_mochi_third_party}/benchmark" CACHE STRING "Source directory for building benchmark")

set(MOCHI_EIGEN_SOURCE_DIR "${_mochi_third_party}/eigen" CACHE STRING "Source directory containing the Eigen/ header folder")

option(MOCHI_ENTT_FIND_PACKAGE "Use find_package for EnTT, rather than building from source" OFF)
set(MOCHI_ENTT_SOURCE_DIR "${_mochi_third_party}/entt" CACHE STRING "Source directory for building EnTT")

option(MOCHI_GTEST_FIND_PACKAGE "Use find_package for GTest, rather than building from source" OFF)
set(MOCHI_GTEST_SOURCE_DIR "${_mochi_third_party}/googletest" CACHE STRING "Source directory for building GTest")

option(MOCHI_HDF5_FIND_PACKAGE "Use find_package for HDF5, rather than building from source" OFF)
set(MOCHI_HDF5_SOURCE_DIR "${_mochi_third_party}/hdf5" CACHE STRING "Source directory for building HDF5")

# Permissive mesh-file parser used by mochi_core's OBJ loader.
set(MOCHI_TINYOBJ_SOURCE_DIR "${_mochi_third_party}/tiny_obj_loader" CACHE STRING "Source directory containing tiny_obj_loader.{h,cc}")
set(MOCHI_HAPPLY_SOURCE_DIR "${_mochi_third_party}/happly" CACHE STRING "Source directory containing the happly/ header folder")

# tinyxml2 (COLLADA/.dae parsing for mochi_renderer). Vendored under mochi/third_party
# (like tiny_obj_loader / happly) so the build is self-contained.
set(MOCHI_TINYXML2_SOURCE_DIR "${_mochi_third_party}/tinyxml2" CACHE STRING "Source directory containing tinyxml2.{h,cpp}")

option(MOCHI_PYBIND_FIND_PACKAGE "Use find_package for pybind11, rather than building from source" OFF)
set(MOCHI_PYBIND_SOURCE_DIR "${_mochi_third_party}/pybind11" CACHE STRING "Source directory for building pybind11")

option(MOCHI_TRACY_FIND_PACKAGE "Use find_package for Tracy, rather than building from source" OFF)

set(MOCHI_FILAMENT_SOURCE_DIR "${_mochi_third_party}/filament" CACHE STRING "Source directory for building Filament")

option(MOCHI_ZLIB_FIND_PACKAGE "Use find_package for ZLIB, rather than building from source" OFF)
set(MOCHI_ZLIB_SOURCE_DIR "${_mochi_third_party}/zlib" CACHE STRING "Source directory for building ZLIB")

unset(_mochi_third_party)

#-----------------------------------------------------------------------------
# superdex_mesh_cli third party
#-----------------------------------------------------------------------------

option(MOCHI_CGAL_FIND_PACKAGE "Use find_package for CGAL, rather than building from source" OFF)
set(MOCHI_CGAL_SOURCE_DIR "" CACHE PATH "Source directory containing the CGAL headers")

set(MOCHI_BOOST_SOURCE_DIR "" CACHE PATH "Source directory containing the boost/ header folder")

set(MOCHI_RAPIDJSON_SOURCE_DIR "" CACHE PATH "Source directory containing the rapidjson/ header folder")

set(MOCHI_OCCT_SOURCE_DIR "" CACHE PATH "Source directory for building OpenCASCADE (OCCT)")
