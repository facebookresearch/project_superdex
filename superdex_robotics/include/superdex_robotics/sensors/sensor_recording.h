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

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace superdex::robotics {

struct SensorSample;

struct SensorRecordingInt64Values {
  std::string_view name;
  Span<int64_t const> values;
};

struct SensorRecordingFloat64Values {
  std::string_view name;
  Span<double const> values;
};

/* @brief Backend-neutral destination for one sensor's constants and timestamped samples.
 * Concrete sensors describe only their own optional fields. The sink owns storage, encoding and
 * persistence, so a recorder does not need to include or identify concrete sensor classes. */
class MOCHI_API SensorRecordingSink {
 public:
  SensorRecordingSink() = default;
  virtual ~SensorRecordingSink();

  MOCHI_DECLARE_MOVE_ONLY(SensorRecordingSink)

  virtual void DeclareMetadataType(std::string_view typeId, superdex::Error& error) = 0;

  virtual void WriteConstantText(
      std::string_view relativePath,
      std::string_view name,
      std::string_view value,
      superdex::Error& error) = 0;
  virtual void WriteConstantTexts(
      std::string_view relativePath,
      std::string_view name,
      Span<DynamicString const> values,
      superdex::Error& error) = 0;
  virtual void WriteConstantScalars(
      std::string_view relativePath,
      std::string_view name,
      Span<double const> values,
      superdex::Error& error) = 0;
  virtual void WriteConstantInt32s(
      std::string_view relativePath,
      std::string_view name,
      Span<int32_t const> values,
      superdex::Error& error) = 0;
  virtual void WriteConstantReal3s(
      std::string_view relativePath,
      std::string_view name,
      Span<Real3 const> values,
      superdex::Error& error) = 0;

  virtual void
  DeclareInt64SampleField(std::string_view name, std::size_t count, superdex::Error& error) = 0;
  virtual void
  DeclareFloat64SampleField(std::string_view name, std::size_t count, superdex::Error& error) = 0;

  void WriteSample(SensorSample const& sample, superdex::Error& error);
  virtual void WriteSampleWithFields(
      SensorSample const& sample,
      Span<SensorRecordingInt64Values const> int64Fields,
      Span<SensorRecordingFloat64Values const> float64Fields,
      superdex::Error& error) = 0;
};

} // namespace superdex::robotics
