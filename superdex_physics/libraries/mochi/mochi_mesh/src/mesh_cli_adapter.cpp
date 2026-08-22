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

#include "mesh_cli_adapter.h"

#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/span_utils.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

using namespace mochi;
using namespace mochi::mesh;

namespace {

Real3 ToReal3FromWire(std::array<double, 3> const& values) {
  return Real3{
      static_cast<real>(values[0]), static_cast<real>(values[1]), static_cast<real>(values[2])};
}

// Shape and finiteness rules the wire representation must satisfy in both directions.
//
// negativeValueBounds is only required to be finite, not populated: it is carried through verbatim,
// and callers such as ReconstructSurfaceFromSdf have no reason to set it. Finiteness is still
// enforced so a corrupt peer cannot inject NaN into a GridSdfData.
void ValidateCliScalarField(cli::ScalarField3d const& field, Error& error) {
  MOCHI_ERROR_RETURN(error);

  uint64_t sampleCount = 1;
  for (int axis = 0; axis < 3; ++axis) {
    MOCHI_ERROR_IF(field.dims[axis] < 2, error, "SDF dimensions must all be at least two.");
    MOCHI_ERROR_RETURN(error);
    auto const dimension = static_cast<uint64_t>(field.dims[axis]);
    MOCHI_ERROR_IF(
        sampleCount > std::numeric_limits<uint64_t>::max() / dimension,
        error,
        "SDF dimensions overflow the protocol sample count.");
    MOCHI_ERROR_RETURN(error);
    sampleCount *= dimension;
  }
  MOCHI_ERROR_IF(
      sampleCount != field.values.size(), error, "SDF dimensions do not match the value count.");
  MOCHI_ERROR_IF(
      !std::all_of(
          field.values.begin(), field.values.end(), [](double value) { return IsFinite(value); }),
      error,
      "SDF values must be finite.");
  MOCHI_ERROR_RETURN(error);

  for (int axis = 0; axis < 3; ++axis) {
    MOCHI_ERROR_IF(
        !IsFinite(field.boundsMin[axis]) || !IsFinite(field.boundsMax[axis]) ||
            field.boundsMin[axis] >= field.boundsMax[axis],
        error,
        "SDF bounds must be finite and non-degenerate.");
    MOCHI_ERROR_IF(
        !IsFinite(field.negativeValueBoundsMin[axis]) ||
            !IsFinite(field.negativeValueBoundsMax[axis]),
        error,
        "SDF negative-value bounds must be finite.");
  }
}

} // namespace

cli::MeshData cli_adapter::ToCliMeshData(MeshDataView const& mesh, Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  // Don't convert an invalid model
  ModelDataView model;
  model.mesh = mesh;
  model::Validate(model, error);
  MOCHI_ERROR_RETURN(error, {});

  cli::MeshData outMesh;
  outMesh.nodesPerElement = mesh.nodesPerElement;
  outMesh.coordinates.resize(mesh.coordinates.size());
  StaticCast(MakeConstSpan(mesh.coordinates), MakeSpan(outMesh.coordinates));
  outMesh.connectivity.resize(mesh.connectivity.size());
  StaticCast(MakeConstSpan(mesh.connectivity), MakeSpan(outMesh.connectivity));

  return outMesh;
}

MeshData cli_adapter::FromCliMeshData(cli::MeshData const& mesh, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MeshData outMesh;
  outMesh.nodesPerElement = mesh.nodesPerElement;
  outMesh.coordinates.resize_noinit(mesh.coordinates.size());
  StaticCast(MakeConstSpan(mesh.coordinates), MakeSpan(outMesh.coordinates));
  outMesh.connectivity.resize_noinit(mesh.connectivity.size());
  StaticCast(MakeConstSpan(mesh.connectivity), MakeSpan(outMesh.connectivity));

  // Don't accept an invalid mesh
  ModelDataView model;
  model.mesh = MeshDataView{outMesh};
  model::Validate(model, error);
  MOCHI_ERROR_RETURN(error, {});

  return outMesh;
}

cli::ScalarField3d cli_adapter::ToCliScalarField(GridSdfDataView const& grid, Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  auto const boundsMin = grid.bounds.GetMin();
  auto const boundsMax = grid.bounds.GetMax();
  auto const negativeBoundsMin = grid.negativeValueBounds.GetMin();
  auto const negativeBoundsMax = grid.negativeValueBounds.GetMax();

  cli::ScalarField3d outField;
  for (int axis = 0; axis < 3; ++axis) {
    outField.dims[axis] = static_cast<int32_t>(grid.dims[axis]);
    outField.boundsMin[axis] = static_cast<double>(boundsMin[axis]);
    outField.boundsMax[axis] = static_cast<double>(boundsMax[axis]);
    outField.negativeValueBoundsMin[axis] = static_cast<double>(negativeBoundsMin[axis]);
    outField.negativeValueBoundsMax[axis] = static_cast<double>(negativeBoundsMax[axis]);
  }
  outField.values.resize(grid.values.size());
  StaticCast(grid.values, MakeSpan(outField.values));

  // Don't send an ill-formed field across the process boundary.
  ValidateCliScalarField(outField, error);
  MOCHI_ERROR_RETURN(error, {});

  return outField;
}

GridSdfData cli_adapter::FromCliScalarField(cli::ScalarField3d const& field, Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  // Don't accept an ill-formed field; it crossed a process boundary.
  ValidateCliScalarField(field, error);
  MOCHI_ERROR_RETURN(error, {});

  GridSdfData outGrid;
  for (int axis = 0; axis < 3; ++axis) {
    outGrid.dims[axis] = field.dims[axis];
  }
  outGrid.values.resize_noinit(field.values.size());
  StaticCast(MakeConstSpan(field.values), MakeSpan(outGrid.values));
  outGrid.bounds = Aabb(ToReal3FromWire(field.boundsMin), ToReal3FromWire(field.boundsMax));
  outGrid.negativeValueBounds = Aabb(
      ToReal3FromWire(field.negativeValueBoundsMin), ToReal3FromWire(field.negativeValueBoundsMax));
  return outGrid;
}
