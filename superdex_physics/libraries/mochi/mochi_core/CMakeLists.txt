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

mochi_add_static_library(mochi_core)

target_compile_definitions(mochi_core PUBLIC "MOCHI_CORE_STATIC_LINKING=1")
target_compile_definitions(mochi_core PUBLIC "MOCHI_INTERNAL=0")

if (MOCHI_WARN_IF_NOT_OPTIMIZED)
    target_compile_definitions(mochi_core PRIVATE "MOCHI_WARN_IF_NOT_OPTIMIZED=1")
endif ()

target_link_libraries(mochi_core PUBLIC marl)
target_link_libraries(mochi_core PUBLIC picojson simple_reflection)

# Permissive mesh-file parser: OBJ via tiny_obj_loader.
target_link_libraries(mochi_core PUBLIC tiny_obj_loader)

# Permissive mesh-file parser: PLY via happly (header-only).
target_link_libraries(mochi_core PUBLIC happly)

if (ACCELERATE_LIBRARY)
    target_compile_definitions(mochi_core PUBLIC "MOCHI_USE_MATH_ACCELERATION=1")
    target_link_libraries(mochi_core PUBLIC ${ACCELERATE_LIBRARY})
endif ()

if (MOCHI_USE_EIGEN)
    target_compile_definitions(mochi_core PUBLIC "MOCHI_USE_EIGEN=1")
    target_link_libraries(mochi_core PUBLIC Eigen3::Eigen)
endif ()

if (MOCHI_USE_HDF5)
    target_compile_definitions(mochi_core PUBLIC "MOCHI_USE_HDF5=1")
    if (MOCHI_BUILD_SHARED AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_definitions(mochi_core PUBLIC "MOCHI_HDF5_LOCAL_VISIBILITY=1")
    endif ()
    target_link_libraries(mochi_core PUBLIC HDF5::HDF5 zlibstatic)
    add_dependencies(mochi_core zlibstatic)
endif ()

if (MOCHI_USE_TRACY)
    target_compile_definitions(mochi_core PUBLIC "MOCHI_USE_TRACY=1" "MOCHI_PROFILE_ENABLE=1")
    target_link_libraries(mochi_core PUBLIC TracyClient)
endif ()

if (WIN32)
    target_link_libraries(mochi_core PUBLIC ws2_32)
endif ()

if (MOCHI_USE_DOUBLE_PRECISION)
    target_compile_definitions(mochi_core PUBLIC "MOCHI_USE_DOUBLE_PRECISION=1")
endif ()
