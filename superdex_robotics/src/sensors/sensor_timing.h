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

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace superdex::robotics::detail {

inline int64_t SensorPeriodIndex(double timeSeconds, real sampleRateHz) {
  double const scaled = timeSeconds * static_cast<double>(sampleRateHz);
  double const nearest = std::round(scaled);
  double const roundoffTolerance =
      8.0 * static_cast<double>(std::numeric_limits<real>::epsilon()) * (1.0 + std::abs(scaled));
  return static_cast<int64_t>(
      std::abs(scaled - nearest) <= roundoffTolerance ? nearest : std::floor(scaled));
}

inline std::optional<int64_t> SensorCapturePeriod(
    StepTime const& now,
    real sampleRateHz,
    std::optional<int64_t> lastPeriod,
    std::optional<double>& epochTime) {
  if (now.dt <= 0.0) {
    return std::nullopt;
  }
  if (sampleRateHz > 0_r && !epochTime) {
    epochTime = now.t;
  }
  int64_t const period =
      sampleRateHz > 0_r ? SensorPeriodIndex(now.t - *epochTime, sampleRateHz) : now.step;
  if (sampleRateHz > 0_r && lastPeriod == period) {
    return std::nullopt;
  }
  return period;
}

} // namespace superdex::robotics::detail
