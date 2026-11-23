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

#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>

namespace superdex::robotics {

/* @brief Source that publishes the values a C++ function writes at every step of its graph, the
 * reset step included. Scene and bot files cannot declare it. */
class MOCHI_API SensorFunction final : public SensorBase {
 public:
  /* @brief Writes one sample's values into @p out and returns the sample they derive from, such as
   * another node's Latest(), whose capture step and time the publication keeps, or nullopt for
   * values captured at this step. Nothing publishes while the returned sample has no capture. */
  using Read = std::function<std::optional<SensorSample>(Span<float> out)>;

  static constexpr std::string_view TypeName() {
    return "SENSOR_FUNCTION";
  }

  /* @brief A source publishing in @p format, which must be valid, the values @p read writes. A
   * mirror of another node can pass that node's GetFormat(). */
  SensorFunction(SensorFormat format, Read read, superdex::Error& error);
  ~SensorFunction() override;

  /* @brief The format of @p width plain values: kind "Values" and one "value" axis. */
  [[nodiscard]] static SensorFormat ValuesFormat(int32_t width);

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  [[nodiscard]] SensorStepResult Step(StepTime const& now, superdex::Error& error) override;

 private:
  [[nodiscard]] SensorFormat InitSensor(SensorScope const& scope, superdex::Error& error) override;
  void ResetSensor() override;

  SensorFormat _outputFormat;
  Read _read;
  DynamicArray<float> _values;
};

} // namespace superdex::robotics
