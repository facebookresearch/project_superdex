/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <string_view>

namespace superdex::lab::locomotion {

/// twist_command's outputs.
inline constexpr std::string_view kCommandOutput = "command";
inline constexpr std::string_view kPhaseOutput = "phase";

/// foot_contact's outputs, one value per foot each.
inline constexpr std::string_view kFoundOutput = "found";
inline constexpr std::string_view kForceOutput = "force";
inline constexpr std::string_view kForceLog1pOutput = "force_log1p";
inline constexpr std::string_view kAirTimeOutput = "air_time";
inline constexpr std::string_view kContactTimeOutput = "contact_time";
inline constexpr std::string_view kLastAirTimeOutput = "last_air_time";
inline constexpr std::string_view kFirstContactOutput = "first_contact";
inline constexpr std::string_view kHeightOutput = "height";
inline constexpr std::string_view kSpeedXyOutput = "speed_xy";

} // namespace superdex::lab::locomotion
