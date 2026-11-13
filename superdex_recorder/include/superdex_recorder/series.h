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

#include <superdex_recorder/recorder.h>

#include <mochi_core/utils/error.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>

#include <rerun.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace superdex::recorder {

/* @brief A timestamped set of columns buffered until its episode commits. */
class SUPERDEX_RECORDER_API Series {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(Series)

 public:
  virtual ~Series();

  [[nodiscard]] size_t GetNumSamples() const;

 protected:
  explicit Series(std::string_view entityPath);

  /* Checks that time follows the previous append's time and makes it the latest, without storing
   * a sample. */
  void AdvanceTime(SampleTime time, mochi::Error& error);
  void AppendTime(SampleTime time, mochi::Error& error);

 private:
  friend class Episode;

  void Flush(rerun::RecordingStream& recording, mochi::Error& error);
  virtual void BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error) = 0;

  std::string _entityPath;
  std::vector<int64_t> _frames;
  std::vector<double> _simulationTimesSec;
  std::vector<double> _wallTimesSec; // Empty when the samples have no wall time.
  std::optional<SampleTime> _latestTime; // May be newer than the last stored time.
  bool _sealed = false; // Rejects appends once its episode commits or is discarded.
};

/* @brief Stores one or more variable-length scalar fields per sample. */
class SUPERDEX_RECORDER_API ScalarSeries final : public Series {
 private:
  // Values keep the scene's precision. Rerun's scalar component is always 64-bit, so
  // single-precision builds store a 32-bit datatype instead.
  using Value = std::conditional_t<
      std::is_same_v<mochi::real, float>,
      rerun::datatypes::Float32,
      rerun::components::Scalar>;

  struct Column {
    ComponentField field;
    std::vector<uint32_t> lengths;
    std::vector<Value> values;
  };

 public:
  ScalarSeries(std::string_view entityPath, std::vector<ComponentField> const& fields);

  void Append(
      SampleTime time,
      mochi::Span<mochi::Span<mochi::real const> const> values,
      mochi::Error& error);

 private:
  void BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error) override;

  std::vector<Column> _columns;
};

/* @brief Stores one translation and quaternion per sample. A transform equal to the previous
 * sample is not stored, because readers take the latest sample at or before a time. */
class SUPERDEX_RECORDER_API TransformSeries final : public Series {
 public:
  explicit TransformSeries(std::string_view entityPath);

  void Append(
      SampleTime time,
      std::array<mochi::real, 3> const& translation,
      std::array<mochi::real, 4> const& quaternionXyzw,
      mochi::Error& error);
  void Append(SampleTime time, mochi::TransformRT const& transform, mochi::Error& error);

 private:
  void BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error) override;

  std::vector<rerun::components::Translation3D> _translations;
  std::vector<rerun::components::RotationQuat> _rotations;
  // Translation then quaternion, as stored.
  std::optional<std::array<float, 7>> _lastTransform;
};

/* @brief Stores a variable-length batch of world-space 3D arrows per sample. */
class SUPERDEX_RECORDER_API Arrow3DSeries final : public Series {
 public:
  explicit Arrow3DSeries(std::string_view entityPath);

  void Append(
      SampleTime time,
      mochi::Span<mochi::Real3 const> origins,
      mochi::Span<mochi::Real3 const> vectors,
      mochi::Error& error);

 private:
  void BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error) override;

  std::vector<uint32_t> _lengths;
  std::vector<rerun::components::Position3D> _origins;
  std::vector<rerun::components::Vector3D> _vectors;
};

/* @brief Stores a variable-length batch of world-space 3D points per sample. A batch equal to the
 * previous sample is not stored, because readers take the latest sample at or before a time. */
class SUPERDEX_RECORDER_API Points3DSeries final : public Series {
 public:
  explicit Points3DSeries(std::string_view entityPath);

  void Append(SampleTime time, mochi::Span<mochi::Real3 const> positions, mochi::Error& error);

 private:
  void BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error) override;

  std::vector<uint32_t> _lengths;
  std::vector<rerun::components::Position3D> _positions;
};

/* @brief Stores a triangle mesh whose vertices move but whose triangles do not, such as the surface
 * of a soft actor. Each sample holds every vertex's position and normal; the triangles are written
 * once, by Episode::CreateDeformableMeshSeries. A sample equal to the previous one is not stored,
 * because readers take the latest sample at or before a time. */
class SUPERDEX_RECORDER_API DeformableMeshSeries final : public Series {
 public:
  DeformableMeshSeries(std::string_view entityPath, size_t numVertices);

  /* Takes XYZ triples, one per vertex. */
  void Append(
      SampleTime time,
      mochi::Span<mochi::real const> vertexPositions,
      mochi::Span<mochi::real const> vertexNormals,
      mochi::Error& error);

 private:
  void BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error) override;

  size_t _numVertices;
  std::vector<rerun::components::Position3D> _positions;
  std::vector<rerun::components::Vector3D> _normals;
};

class SUPERDEX_RECORDER_API TextSeries final : public Series {
 public:
  TextSeries(std::string_view entityPath, ComponentField field);

  /* Records the texts of one frame. An empty sample records the frame with no text. */
  void Append(SampleTime time, mochi::Span<std::string const> texts, mochi::Error& error);

 private:
  void BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error) override;

  ComponentField _field;
  std::vector<uint32_t> _lengths;
  std::vector<rerun::components::Text> _texts;
};

/* @brief Stores one double per sample as Rerun's Scalars archetype, which the Rerun viewer plots by
 * default. */
class SUPERDEX_RECORDER_API PlotSeries final : public Series {
 public:
  explicit PlotSeries(std::string_view entityPath);

  void Append(SampleTime time, double value, mochi::Error& error);

 private:
  void BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error) override;

  std::vector<rerun::components::Scalar> _values;
};

} // namespace superdex::recorder
