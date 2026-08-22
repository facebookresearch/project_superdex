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

#-----------------------------------------------------------------------------
# What targets to build:
#-----------------------------------------------------------------------------

option(SR_BUILD_TESTS OFF "Build unit tests")

#-----------------------------------------------------------------------------
# Optional Features:
#-----------------------------------------------------------------------------

option(SR_USE_NLOHMANN_JSON "Include support for JSON serialization with nlohmann_json (third-party)" OFF)

#-----------------------------------------------------------------------------
# Third party source locations must be specified:
#
#   SR_GTEST_SOURCE_DIR (used if SR_BUILD_GTEST_FROM_SOURCE is ON)
#   SR_PICOJSON_SOURCE_DIR
#   SR_NLOHMANN_JSON_SOURCE_DIR
#-----------------------------------------------------------------------------

option(SR_BUILD_GTEST_FROM_SOURCE "Build googletest from source located at SR_GTEST_SOURCE_DIR. Else, we will try to find it." OFF)
