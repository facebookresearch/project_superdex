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

#include <superdex_robotics/sensors/sensor_base.h>

#include <string_view>

namespace superdex::robotics {

/* @brief Which side of a sensor/environment interaction the reported force or wrench acts on. */
enum class InteractionWrenchSense {
  EnvironmentOnSensor,
  SensorOnEnvironment,
};

/* @brief Interaction sense of a force or wrench observation. */
struct InteractionWrenchMeta {
  InteractionWrenchSense sense = InteractionWrenchSense::EnvironmentOnSensor;
};

template <>
struct SensorMetadataKey<InteractionWrenchMeta> {
  static constexpr std::string_view Key = "wrench.interaction_sense.v1";
};

} // namespace superdex::robotics

MOCHI_ENUM_BEGIN(superdex::robotics::InteractionWrenchSense)
MOCHI_ENUM_ITEM(EnvironmentOnSensor)
MOCHI_ENUM_ITEM(SensorOnEnvironment)
MOCHI_ENUM_END()
