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

# The robotics tests want the SuperDex assets root containing bots/ and test/.
# Mochi's own assets/ is the wrong root for this target.
set(MOCHI_TEST_ASSETS_DIR "${PROJECT_SOURCE_DIR}/assets")

mochi_add_library(superdex_robotics)

target_link_libraries(superdex_robotics
    PUBLIC
    mochi_core
    mochi_physics
    superdex_physics
    PRIVATE
    xxhash
    miniz
    tinyxml2
)

if (MOCHI_BUILD_SHARED)
    target_compile_definitions(superdex_robotics
        PUBLIC "MOCHI_PHYSICS_DYNAMICALLY_LINKED=1"
        PRIVATE "MOCHI_PHYSICS_EXPORTS=1"
    )
else ()
    target_compile_definitions(superdex_robotics
        PUBLIC "MOCHI_PHYSICS_DYNAMICALLY_LINKED=0"
    )
endif ()

# The bots pybind extension (mochi.bots) lives here (superdex_robotics/pybind), co-located
# with the bots C++ library it binds. It is added when building with pybind; the EXISTS
# guard keeps a standalone superdex_robotics build robust if the pybind/ dir is absent.
if (MOCHI_USE_PYBIND AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/pybind")
    add_subdirectory(pybind)
endif()
