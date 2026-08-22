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

add_executable(imguios_example_app)

target_sources(imguios_example_app
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/imguios_example_app.cpp
)

target_compile_features(imguios_example_app PRIVATE cxx_std_17)
target_link_libraries(imguios_example_app PRIVATE imguios)

#-----------------------------------------------------------------------------
# imguios_example_app_metal (macOS only)
#-----------------------------------------------------------------------------

if (APPLE AND IMGUIOS_BUILD_METAL)
    add_executable(imguios_example_app_metal)

    target_sources(imguios_example_app_metal
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/imguios_example_app.cpp
    )

    target_compile_features(imguios_example_app_metal PRIVATE cxx_std_17)
    target_compile_definitions(imguios_example_app_metal PRIVATE IMGUIOS_USE_METAL=1)
    target_link_libraries(imguios_example_app_metal PRIVATE imguios_metal)
endif()
