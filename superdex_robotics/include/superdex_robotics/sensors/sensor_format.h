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

#include <superdex_physics.h>

#include <cstdint>

namespace superdex::robotics {

enum class SensorDType {
  Float32,
};

using SensorDimension = int32_t;

/* @brief Meaning of one tensor dimension.
 * Labels are empty or contain one entry per element. Units are empty, contain one broadcast unit,
 * or contain one unit per element. At most one axis in a format declares units. */
struct SensorAxis {
  DynamicString name;
  DynamicArray<DynamicString> labels;
  DynamicArray<DynamicString> units;

  bool operator==(SensorAxis const& rhs) const = default;
};

/* @brief Immutable shape and meaning of one sensor's numeric output. */
struct SensorFormat {
  DynamicString kind;
  DynamicArray<SensorDimension> shape;
  SensorDType dtype = SensorDType::Float32;
  DynamicArray<SensorAxis> axes;

  bool operator==(SensorFormat const& rhs) const = default;
};

/* @brief Owning flat row-major tensor used at copying and serialization boundaries. */
using SensorTensor = DynamicArray<float>;

} // namespace superdex::robotics
