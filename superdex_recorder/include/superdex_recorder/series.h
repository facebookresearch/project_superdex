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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace superdex::recorder {

namespace detail {

/* Values a series has buffered, which its episode's writer thread encodes as one chunk. */
struct SeriesRows {
  virtual ~SeriesRows();
  virtual void BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error)
      const = 0;
};

} // namespace detail

/* @brief A timestamped set of columns, written in chunks by its episode's writer thread. */
class SUPERDEX_RECORDER_API Series {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(Series)

 public:
  virtual ~Series();

  /* Includes samples already submitted to the writer. */
  [[nodiscard]] size_t GetNumSamples() const;

 protected:
  explicit Series(std::string_view entityPath);

  /* Checks that time follows the previous append's time and makes it the latest, without storing
   * a sample. */
  void AdvanceTime(SampleTime time, mochi::Error& error);
  /* Stores a sample's time; the subclass then stores its numBytes of values. */
  void AppendTime(SampleTime time, size_t numBytes, mochi::Error& error);

 private:
  friend class Episode;

  /* Moves out the buffered values, leaving the series empty. */
  virtual std::unique_ptr<detail::SeriesRows> TakeRows() = 0;
  /* Queues the buffered samples on the writer thread as one RRD chunk. */
  void SubmitChunk();

  std::string _entityPath;
  std::vector<int64_t> _frames;
  std::vector<double> _simulationTimesSec;
  std::vector<double> _wallTimesSec; // Empty when the samples have no wall time.
  size_t _bufferedBytes = 0;
  size_t _numSamples = 0;
  detail::ChunkWriter* _writer = nullptr;
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
  struct Rows final : detail::SeriesRows {
    void BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error)
        const override;

    std::vector<Column> columns;
  };

 public:
  ScalarSeries(std::string_view entityPath, std::vector<ComponentField> const& fields);

  void Append(
      SampleTime time,
      mochi::Span<mochi::Span<mochi::real const> const> values,
      mochi::Error& error);

 private:
  std::unique_ptr<detail::SeriesRows> TakeRows() override;

  std::unique_ptr<Rows> _rows = std::make_unique<Rows>();
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
  struct Rows final : detail::SeriesRows {
    void BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error)
        const override;

    std::vector<rerun::components::Translation3D> translations;
    std::vector<rerun::components::RotationQuat> rotations;
  };

  std::unique_ptr<detail::SeriesRows> TakeRows() override;

  std::unique_ptr<Rows> _rows = std::make_unique<Rows>();
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
  struct Rows final : detail::SeriesRows {
    void BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error)
        const override;

    std::vector<uint32_t> lengths;
    std::vector<rerun::components::Position3D> origins;
    std::vector<rerun::components::Vector3D> vectors;
  };

  std::unique_ptr<detail::SeriesRows> TakeRows() override;

  std::unique_ptr<Rows> _rows = std::make_unique<Rows>();
};

/* @brief Stores a variable-length batch of world-space 3D points per sample. A batch equal to the
 * previous sample is not stored, because readers take the latest sample at or before a time. */
class SUPERDEX_RECORDER_API Points3DSeries final : public Series {
 public:
  explicit Points3DSeries(std::string_view entityPath);

  void Append(SampleTime time, mochi::Span<mochi::Real3 const> positions, mochi::Error& error);

 private:
  struct Rows final : detail::SeriesRows {
    void BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error)
        const override;

    std::vector<uint32_t> lengths;
    std::vector<rerun::components::Position3D> positions;
  };

  std::unique_ptr<detail::SeriesRows> TakeRows() override;

  std::unique_ptr<Rows> _rows = std::make_unique<Rows>();
  std::vector<rerun::components::Position3D> _lastPositions;
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
  struct Rows final : detail::SeriesRows {
    void BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error)
        const override;

    std::vector<uint32_t> lengths;
    std::vector<rerun::components::Position3D> positions;
    std::vector<rerun::components::Vector3D> normals;
  };

  std::unique_ptr<detail::SeriesRows> TakeRows() override;

  size_t _numVertices;
  std::unique_ptr<Rows> _rows = std::make_unique<Rows>();
  std::vector<rerun::components::Position3D> _lastPositions;
  std::vector<rerun::components::Vector3D> _lastNormals;
};

class SUPERDEX_RECORDER_API TextSeries final : public Series {
 public:
  TextSeries(std::string_view entityPath, ComponentField field);

  /* Records the texts of one frame. An empty sample records the frame with no text. */
  void Append(SampleTime time, mochi::Span<std::string const> texts, mochi::Error& error);

 private:
  struct Rows final : detail::SeriesRows {
    void BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error)
        const override;

    ComponentField field;
    std::vector<uint32_t> lengths;
    std::vector<rerun::components::Text> texts;
  };

  std::unique_ptr<detail::SeriesRows> TakeRows() override;

  std::unique_ptr<Rows> _rows = std::make_unique<Rows>();
};

/* @brief Stores one double per sample as Rerun's Scalars archetype, which the Rerun viewer plots by
 * default. */
class SUPERDEX_RECORDER_API PlotSeries final : public Series {
 public:
  explicit PlotSeries(std::string_view entityPath);

  void Append(SampleTime time, double value, mochi::Error& error);

 private:
  struct Rows final : detail::SeriesRows {
    void BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error)
        const override;

    std::vector<rerun::components::Scalar> values;
  };

  std::unique_ptr<detail::SeriesRows> TakeRows() override;

  std::unique_ptr<Rows> _rows = std::make_unique<Rows>();
};

} // namespace superdex::recorder
