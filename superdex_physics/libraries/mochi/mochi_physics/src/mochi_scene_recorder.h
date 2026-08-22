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

#include "mochi_common_components.h"
#include "mochi_ecs.h"

#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/group_rw.h>

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace mochi {

/**********************************************************************************************
  SceneRecorder captures data from a mochi::Scene and records it to disk using a GroupWriter
  interface. The GroupWriter determines the file format (example: see CreateGroupWriterHDF5).
*/
class SceneRecorder final {
 public:
  ~SceneRecorder();
  SceneRecorder(SceneRecorder const&) = delete;
  SceneRecorder& operator=(SceneRecorder const&) = delete;
  SceneRecorder(SceneRecorder&&) = delete;
  SceneRecorder& operator=(SceneRecorder&&) = delete;

  // Construct a SceneRecorder. It will take ownership of the provided RecordWriter.
  SceneRecorder(
      std::unique_ptr<GroupWriter> writer,
      entt::registry& registry,
      RecordingParams const& params = {});

  // Call before a simulation step
  void OnStepBegin(double timeStepSec, Real3 stepGravity);

  // Call at the end of a simulation step
  void OnStepEnd();

  // Return true if no errors have been encountered. Equivalent to GetError().IsOK().
  bool IsOK() const {
    return _error.IsOK();
  }

  // If the recording failed, then this is the reason.
  Error const& GetError() const {
    return _error;
  }

 private:
  [[nodiscard]] GroupWriter::ScopeGuard EnterNewEvent(std::string_view eventType);
  void CheckForErrors();
  void AddCreateActorEvents();
  void AddDestroyActorEvents();
  void AddStepEvent();

  // ECS Callbacks
  void OnCreateActor(entt::registry& reg, entt::entity e);
  void OnDestroyActor(entt::registry& reg, entt::entity e);

  entt::registry& _registry;
  std::unique_ptr<GroupWriter> _writer;
  std::unique_ptr<GroupWriter::ScopeGuard> _eventsGroup;
  std::vector<entt::entity> _createdActors;
  std::vector<entt::entity> _destroyedActors;
  Error _error;
  double _timeTotal = 0.0;
  double _timeStep = 0.0;
  Real3 _gravity = {};
  uint64_t _eventCounter = 0;
  bool _hasReportedError = false;
  RecordingParams _params;
  std::vector<std::pair<entt::entity, QueryHandle>> _queries;
};

/**********************************************************************************************
  ECS Components
*/

/**
  CRecordingData
  - Each actor will have this component when recording is active.
  - Stores arbitrary attributes and datasets which should recorded.
  - Attributes must be single values or 1D arrays. Datasets can have more dimensions.
  - Entries will be flushed and cleared at the end of the simulation step.
*/
struct CRecordingData : public NoCopy {
  // Expand this list as needed
  enum Type { Double, Float, Int };

  // Helper template returns the Type enum
  template <typename T>
  static Type TypeOf() {
    if constexpr (std::is_same_v<T, double>) {
      return Type::Double;
    } else if constexpr (std::is_same_v<T, float>) {
      return Type::Float;
    } else if constexpr (std::is_same_v<T, int>) {
      return Type::Int;
    } else {
      static_assert(std::is_void_v<T>, "Unsupported type");
    }
  };

  // Data for a single attribute or dataset
  struct Entry {
    Type type = Type::Float; // Value type stored in the data (required)
    bool isAttribute = false; // If true, save as an attribute instead of a dataset
    std::vector<size_t> dims; // Dimensions of the data (required for datasets)

    // Data stored as a flat array of bytes (required). The last dimension should be contiguous just
    // like a C array (e.g. real[D0][D1][D2]). This is called row-major in the 2D matrix case.
    std::vector<std::byte> data;
  };

  std::map<std::string, Entry> entries;
  RecordingParams params = {};

  CRecordingData() = default;
  inline CRecordingData(RecordingParams params) : params(params) {}
};

// This tag on the global ctx when recording is active.
struct TagSceneRecordingEnabled {};

// Global context component stores the current recording parameters
using CRecordingParams = RecordingParams;

/**********************************************************************************************
Helper Functions
*/

// Add an attribute to CRecordingData using a Span<Scalar const>
template <typename Scalar>
void RecordAttribute(std::string_view name, Span<Scalar const> values, CRecordingData& outData) {
  auto const* bytesBegin = reinterpret_cast<std::byte const*>(&values[0]);
  std::byte const* bytesEnd = bytesBegin + values.size() * sizeof(Scalar);
  auto& outEntry = outData.entries[std::string{name}];
  outEntry.isAttribute = true;
  outEntry.type = CRecordingData::TypeOf<Scalar>();
  outEntry.dims.resize(1);
  outEntry.dims[0] = isize(values);
  outEntry.data.assign(bytesBegin, bytesEnd);
}

// Add an n-dimensional dataset to CRecordingData using a Span<Scalar const>.
// Values in the last dimension should be stored contiguously (row major in the case of a matrix)
template <typename Scalar>
void RecordDataset(
    std::string_view name,
    Span<int const> dims,
    Span<Scalar const> values,
    CRecordingData& outData) {
  MOCHI_ASSERT(!values.empty(), "Empty datasets are not allowed");
  auto const* bytesBegin = reinterpret_cast<std::byte const*>(&values[0]);
  std::byte const* bytesEnd = bytesBegin + values.size() * sizeof(Scalar);
  auto& outEntry = outData.entries[std::string{name}];
  MOCHI_ASSERT(
      outEntry.data.empty(),
      "RecordDataset called twice with the same key in a single step. Dataset keys must be unique per step.");
  outEntry.isAttribute = false;
  outEntry.type = CRecordingData::TypeOf<Scalar>();
  outEntry.dims.assign(dims.begin(), dims.end());
  outEntry.data.assign(bytesBegin, bytesEnd);
}

template <typename Scalar, int kRows>
void RecordDataset(
    std::string_view name,
    ColumnVectorView<Scalar const, kRows> view,
    CRecordingData& outData) {
  int dims[] = {view.Rows()};
  RecordDataset(name, MakeSpan(dims), view.GetConstSpan(), outData);
}

template <typename Scalar>
void RecordEmptySparseMatrixCSR(
    std::string_view name,
    int numRows,
    int numCols,
    CRecordingData& outData) {
  std::string dsName{name};

  // H5 does not allow empty datasets. Represent empty CSR matrices with a dummy value and
  // numNonZeros=0 metadata.
  auto data = static_cast<Scalar>(0);
  auto const* bytesBegin = reinterpret_cast<std::byte const*>(&data);
  std::byte const* bytesEnd = bytesBegin + sizeof(Scalar);

  // Add dataset
  auto& entry = outData.entries[dsName];
  entry.isAttribute = false;
  entry.type = CRecordingData::TypeOf<Scalar>();
  entry.dims = {1};
  entry.data.assign(bytesBegin, bytesEnd);

  // Add attributes to the dataset (denoted by a forward slash in the name)
  int numNonZeros = 0;
  RecordAttribute(dsName + "/numRows", Span<int const>{&numRows, 1}, outData);
  RecordAttribute(dsName + "/numCols", Span<int const>{&numCols, 1}, outData);
  RecordAttribute(dsName + "/numNonZeros", Span<int const>{&numNonZeros, 1}, outData);
}

// Add a dataset to CRecordingData using a sparse matrix in CSR format. The non-zero values will be
// stored in the dataset itself. The dimensions, offsets, and indices will be stored as attributes
// on the dataset.
template <typename Scalar>
void RecordDatasetSparseMatrixCSR(
    std::string_view name,
    int numRows,
    int numCols,
    Span<int const> rowOffsets,
    Span<int const> colIndices,
    Span<Scalar const> nonZeroValues,
    CRecordingData& outData) {
  if (isize(nonZeroValues) == 0) {
    RecordEmptySparseMatrixCSR<Scalar>(name, numRows, numCols, outData);
  } else {
    MOCHI_ASSERT(isize(nonZeroValues) > 0, "Empty dataset not allowed");
    MOCHI_ASSERT(isize(rowOffsets) == (numRows + 1), "Expected offsets for each row plus one");
    MOCHI_ASSERT(isize(colIndices) == isize(nonZeroValues), "Expected 1 col index per value");
    int const numNonZeros = isize(nonZeroValues);
    auto const* bytesBegin = reinterpret_cast<std::byte const*>(&nonZeroValues[0]);
    std::byte const* bytesEnd = bytesBegin + numNonZeros * sizeof(Scalar);
    std::string dsName{name};

    // Add dataset
    auto& entry = outData.entries[dsName];
    entry.isAttribute = false;
    entry.type = CRecordingData::TypeOf<Scalar>();
    entry.dims.assign(&numNonZeros, &numNonZeros + 1);
    entry.data.assign(bytesBegin, bytesEnd);

    // Add attributes to the dataset (denoted by a forward slash in the name)
    RecordAttribute(dsName + "/numRows", Span<int const>{&numRows, 1}, outData);
    RecordAttribute(dsName + "/numCols", Span<int const>{&numCols, 1}, outData);
    RecordAttribute(dsName + "/numNonZeros", Span<int const>{&numNonZeros, 1}, outData);

    // The row offsets and column indices are potentially too large for attributes in HDF5 format
    // (64Kb limit), so we save them as separate datasets.
    int rowOffsetsDims[1] = {numRows + 1};
    int colIndicesDims[1] = {numNonZeros};
    RecordDataset(dsName + "_rowOffsets", rowOffsetsDims, rowOffsets, outData);
    RecordDataset(dsName + "_colIndices", colIndicesDims, colIndices, outData);
  }
}

// Writes a transform to recording data by writing the translation and rotations separately
// to the attributes named by translationName and rotationName. The rotation is stored as a
// 3x3 matrix.
void WriteTransformAttributes(
    std::string const& translationName,
    std::string const& rotationName,
    TransformRT const& transform,
    CRecordingData& data);

namespace scene_recorder {
void InitializeOnce(entt::registry& reg);
}

} // namespace mochi
