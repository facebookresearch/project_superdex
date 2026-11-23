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

#include <superdex_robotics/sensors/sensor_format.h>

#include <limits>

using namespace mochi;
using namespace superdex::robotics;

std::size_t superdex::robotics::GetSensorFormatElementCount(
    SensorFormat const& format,
    Error& error) {
  MOCHI_ERROR_RETURN(error, 0);
  if (format.kind.empty()) {
    MOCHI_ERROR_SET(error, "Sensor output kind must not be empty");
    return 0;
  }
  if (format.shape.empty()) {
    MOCHI_ERROR_SET(error, "Sensor output shape must not be empty");
    return 0;
  }
  if (format.axes.size() != format.shape.size()) {
    MOCHI_ERROR_SET(error, "Sensor output axes must match the shape rank");
    return 0;
  }

  std::size_t elementCount = 1;
  bool hasUnits = false;
  for (std::size_t i = 0; i < format.shape.size(); ++i) {
    SensorDimension const dimension = format.shape[i];
    if (dimension <= 0) {
      MOCHI_ERROR_SET(error, "Sensor output dimensions must be positive");
      return 0;
    }
    auto const size = static_cast<std::size_t>(dimension);
    if (elementCount > std::numeric_limits<std::size_t>::max() / size) {
      MOCHI_ERROR_SET(error, "Sensor output element count overflows size_t");
      return 0;
    }
    elementCount *= size;

    SensorAxis const& axis = format.axes[i];
    if (axis.name.empty()) {
      MOCHI_ERROR_SET(error, "Sensor output axis names must not be empty");
      return 0;
    }
    if (!axis.labels.empty() && axis.labels.size() != size) {
      MOCHI_ERROR_SET(error, "Sensor output axis labels must match their dimension");
      return 0;
    }
    if (!axis.units.empty()) {
      if (hasUnits || (axis.units.size() != 1 && axis.units.size() != size)) {
        MOCHI_ERROR_SET(error, "Sensor output units must belong to one compatible axis");
        return 0;
      }
      hasUnits = true;
    }
  }
  return elementCount;
}
