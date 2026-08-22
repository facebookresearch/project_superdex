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

#include <mochi_core/geometry/grid_sdf.h>

#include <mochi_core/contact/contact_utils.h>
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/profile.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>

namespace mochi {

static Real3
ComputeAxisBasedResolution(TriangularMesh const* mesh, GridSdfParams const& params, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_PROFILE_SCOPE();

  // Compute axis size
  auto const& aabb = mesh->GetAabb();
  Real3 range = aabb.GetMax() - aabb.GetMin();
  real axisSize = 0_r;

  switch (params.resolutionMode) {
    case GridSdfResolutionMode::LargestAxis:
      axisSize = Max(range);
      break;

    case GridSdfResolutionMode::SmallestAxis:
      axisSize = Min(range);
      break;

    case GridSdfResolutionMode::MeanAxis:
      axisSize = Mean(range);
      break;

    default:
      MOCHI_ASSERT_VERBOSE(false, "Unreachable code");
  }

  MOCHI_ERROR_IF(
      !IsFinite(axisSize) || (axisSize <= 0_r),
      error,
      "Mesh AABB side length must be finite and positive for SDF computation.");

  // Compute delta
  return params.resolutionDelta * axisSize;
}

static Real3
ComputeEdgeBasedResolution(TriangularMesh const* mesh, GridSdfParams const& params, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_PROFILE_SCOPE();

  // Compute edge size
  auto nodes = mesh->GetNodeCoordinates();
  auto edges = mesh->GetEdges();
  MOCHI_ERROR_IF(edges.empty(), error, "Empty triangle mesh cannot be used to compute an SDF.");
  MOCHI_ERROR_RETURN(error, {});
  real edgeSize = 0_r;

  switch (params.resolutionMode) {
    case GridSdfResolutionMode::LargestEdge:
      edgeSize = std::numeric_limits<real>::min();
      for (int i = 0; i < mesh->GetNumEdges(); ++i) {
        edgeSize = Max(edgeSize, Norm(nodes[edges[i][0]] - nodes[edges[i][1]]));
      }
      break;

    case GridSdfResolutionMode::SmallestEdge:
      edgeSize = std::numeric_limits<real>::max();
      for (int i = 0; i < mesh->GetNumEdges(); ++i) {
        edgeSize = Min(edgeSize, Norm(nodes[edges[i][0]] - nodes[edges[i][1]]));
      }
      break;

    case GridSdfResolutionMode::MeanEdge:
      edgeSize = 0_r;
      for (int i = 0; i < mesh->GetNumEdges(); ++i) {
        edgeSize += Norm(nodes[edges[i][0]] - nodes[edges[i][1]]);
      }
      edgeSize /= isize(edges);
      break;

    default:
      MOCHI_ASSERT_VERBOSE(false, "Unreachable code");
  }

  MOCHI_ERROR_IF(
      !IsFinite(edgeSize) || (edgeSize <= 0_r),
      error,
      "Mesh edge length must be finite and non-zero to compute an SDF.");

  // Compute delta
  return params.resolutionDelta * edgeSize;
}

GridSdf::GridSdf(
    std::shared_ptr<TriangularMesh const> const& mesh,
    GridSdfParams const& params,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_PROFILE_SCOPE();
  Initialize(mesh, params, error);
}

GridSdf::GridSdf(
    std::shared_ptr<DenseGrid3D<real> const> const& grid,
    VMatrix4x4r const& actorFromGrid)
    : _distanceGrid(grid) {
  MOCHI_ASSERT(_distanceGrid != nullptr);
  MOCHI_ASSERT(
      (actorFromGrid[3] == Vec4r{0_r, 0_r, 0_r, 1_r}),
      "GridSdf has invalid transformation matrix. It may only contain scale, rotation (or mirroring), and translation.");

  // Compute scale of the first 3 basis vectors
  auto actorFromGridCols = Transpose4x4(actorFromGrid);
  auto scaleVec =
      Sqrt(Sqr(actorFromGridCols[0]) + Sqr(actorFromGridCols[1]) + Sqr(actorFromGridCols[2]));
  MOCHI_ASSERT(
      AllTrue<3>(scaleVec > std::numeric_limits<real>::epsilon()),
      "GridSdf transformation matrix has degenerate scale");

  // Store uniform scale (X, Y, and Z scale should be equal)
  _actorFromGridScale = scaleVec[0];

  // Remove scale from the matrix
  auto actorFromGridNoScale = Dot4x4(actorFromGrid, VDiagonalMatrix<4>(1_r / scaleVec));

  // Store the 3x3 grid-from-actor rotation matrix (transpose of a rotation matrix is its inverse)
  // The 4th component of each SIMD vector should be ignored, but clamp it to zero just in case.
  _gridFromActorRotation = Transpose3x3(actorFromGridNoScale);
  _gridFromActorRotation[0] = ToSimdDirection(_gridFromActorRotation[0]);
  _gridFromActorRotation[1] = ToSimdDirection(_gridFromActorRotation[1]);
  _gridFromActorRotation[2] = ToSimdDirection(_gridFromActorRotation[2]);

  // Store the full grid-from-actor transform
  _actorFromGridMatT = Transpose4x4(actorFromGrid);
  _gridFromActorMatT = InvertTransformationTransposed(_actorFromGridMatT);

  // Transform the collider bounds
  _colliderBoundsInActorSpace =
      TransformShape(actorFromGrid, _distanceGrid->GetNegativeValueBounds());
}

GridSdf GridSdf::Create(GridSdfData&& data) {
  auto grid = std::make_unique<DenseGrid3D<real>>(
      data.dims, data.bounds, data.negativeValueBounds, std::move(data.values));
  Real3 scale = data.scale.value_or(Real3{1_r, 1_r, 1_r});
  Quaternion rotation = data.rotation.value_or(Quaternion::Identity());
  Real3 translation = data.translation.value_or(Real3{});
  auto rt = TransformRT{rotation, translation};
  VMatrix4x4r actorFromGrid = Dot4x4(ToVMatrix4x4(rt), VDiagonalMatrix<4>(ToSimd(scale, 1_r)));
  return GridSdf{std::move(grid), actorFromGrid};
}

void GridSdf::Initialize(
    std::shared_ptr<TriangularMesh const> const& mesh,
    GridSdfParams const& params,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_PROFILE_SCOPE();

  MOCHI_ERROR_IF_NOT(mesh, error, "Must provide a triangle mesh");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.boundaryPaddingDist) && (params.boundaryPaddingDist >= 0_r),
      error,
      "Boundary padding distance must be finite and non-negative.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.resolutionDelta) && (params.resolutionDelta[0] > 0_r) &&
          (params.resolutionDelta[1] > 0_r) && (params.resolutionDelta[2] > 0_r),
      error,
      "Resolution delta must be finite and positive.");
  MOCHI_ERROR_RETURN(error);

  // Determine delta
  Real3 delta{};
  switch (params.resolutionMode) {
    case GridSdfResolutionMode::LargestAxis:
    case GridSdfResolutionMode::SmallestAxis:
    case GridSdfResolutionMode::MeanAxis:
      delta = ComputeAxisBasedResolution(mesh.get(), params, error);
      break;

    case GridSdfResolutionMode::LargestEdge:
    case GridSdfResolutionMode::SmallestEdge:
    case GridSdfResolutionMode::MeanEdge:
      delta = ComputeEdgeBasedResolution(mesh.get(), params, error);
      break;

    case GridSdfResolutionMode::Explicit:
      delta = params.resolutionDelta;
      break;

    default:
      MOCHI_ERROR_SET(error, "Invalid GridSdfResolutionMode");
  }
  MOCHI_ERROR_RETURN(error);

  // Grid bounds
  Aabb colliderBounds = mesh->GetAabb();
  Aabb gridBounds = ExpandShape(colliderBounds, params.boundaryPaddingDist);

  // Make sure the grid bounds are strictly larger than the mesh abounds.
  Real3 epsilon = gridBounds.GetSize() * std::numeric_limits<real>::epsilon();
  gridBounds = Aabb{gridBounds.GetMin() - epsilon, gridBounds.GetMax() + epsilon};

  // Grid resolution
  auto gridSize = gridBounds.GetSize();
  auto cellDims64 = StaticCast<NdArray<int64_t, 3>>(Ceil(gridSize / delta));

  // Sanity check in case of excessively small delta
  for (int i = 0; i < 3; ++i) {
    MOCHI_ERROR_IF(
        cellDims64[i] >= std::numeric_limits<int>::max(),
        error,
        "Computed SDF grid resolution would be too large. Your mesh may have degenerate bounds or degenerate "
        "triangles. See GridSdfParams for resolution options.");
  }
  MOCHI_ERROR_IF(
      static_cast<double>(cellDims64[0]) * cellDims64[1] * cellDims64[2] >= 1e9,
      error,
      "Computed SDF resolution would include more than a billion cells. Your mesh may have degenerate bounds "
      "or degenerate triangles. See GridSdfParams for resolution options.");
  MOCHI_ERROR_RETURN(error);

  auto cellDims = StaticCast<Int3>(cellDims64);
  for (int i = 0; i < 3; ++i) {
    MOCHI_ASSERT_VERBOSE(
        (gridSize[i] / static_cast<real>(cellDims[i])) <= delta[i],
        "This algorithm should result in a cell size no larger than the delta computed based on GridSdfParams.");
  }

  // Every grid needs at least 1 cell (voxel), but that isn't enough for good results.
  // In fact, mochi_contact.cpp will log a warning if there are less than 6 cells per axis.
  for (int i = 0; i < 3; ++i) {
    cellDims[i] = Max(cellDims[i], Max(1, params.minGridResolution[i]));
  }

  // If there are N cells in a particular direction, then the scalar field will have (N + 1) values.
  auto gridDims = cellDims + Int3{1, 1, 1};

  // Initialize dense grid
  auto newGrid = std::make_shared<DenseGrid3D<real>>(gridDims, gridBounds, colliderBounds);

  // WARNING: This is very slow. It could be made faster if that's important enough.
  _isMeshClosed = InitializeBruteForce(mesh, *newGrid);

  // Same bounds as the grid because transform is identity
  _colliderBoundsInActorSpace = colliderBounds;

  // Store pointer-to-const
  _distanceGrid = newGrid;
}

bool GridSdf::InitializeBruteForce(
    std::shared_ptr<TriangularMesh const> const& mesh,
    DenseGrid3D<real>& outDistanceGrid) {
  MOCHI_PROFILE_SCOPE();

  Int3 const gridDims = outDistanceGrid.GetDimensions();
  std::vector<Real3> samples;

  // Generate a list of sample points
  for (int x = 0; x < gridDims[0]; ++x) {
    for (int y = 0; y < gridDims[1]; ++y) {
      for (int z = 0; z < gridDims[2]; ++z) {
        samples.push_back(outDistanceGrid.GetPointOf(Int3{x, y, z}));
      }
    }
  }

  // Use MeshCollider to find the signed distance and closest face
  MeshCollider meshCollider(mesh);
  meshCollider.Initialize();
  DynamicArray<int> indices;
  DynamicArray<Real3> contacts;
  SdfInfo sdf;
  bool isSdfGradUnitary = {};
  ContactDetectionParams params;
  params.tolerance = std::numeric_limits<real>::infinity(); // Include all points
  params.useAccelerationStructures = true;
  mochi::FindPointContactsParallel(
      samples, &meshCollider, params, TransformRT{}, indices, contacts, sdf, isSdfGradUnitary);
  MOCHI_ASSERT(contacts.size() == samples.size());

  // Fill in the grid
  int i = 0;
  for (int x = 0; x < gridDims[0]; ++x) {
    for (int y = 0; y < gridDims[1]; ++y) {
      for (int z = 0; z < gridDims[2]; ++z) {
        Int3 indexXYZ{x, y, z};
        outDistanceGrid(indexXYZ) = sdf.val[i];
        ++i;
      }
    }
  }

  return meshCollider.IsMeshClosed();
}

template <class SamplerT>
void GridSdf::FindPointContactsImpl(
    Span<Real3 const> points,
    TransformRT const& pointsFromActor,
    ContactDetectionParams const& params,
    DynamicArray<int>& outIndices,
    DynamicArray<Real3>& outContacts,
    SdfInfo& outSdf) const {
  //
  // About coordinate spaces:
  //  - Point-space is what we call the coordinate space of the input points.
  //  - Actor-space is the local space of the owning actor (called the collider).
  //      The caller provides a TransformRT to convert between point-space and actor-space.
  //  - Grid-space is the local space in which the DenseGrid3D was computed. This may be different
  //      from actor-space if the SDF was computed offline and a transform was applied on load. In
  //      that case, the GridSdf will store transformation matrices to convert between actor-space
  //      and grid-space.
  //
  MOCHI_ASSERT_VERBOSE(outIndices.empty(), "Expected empty contact detection result.");
  MOCHI_ASSERT_VERBOSE(outContacts.empty(), "Expected empty contact detection result.");
  MOCHI_ASSERT_VERBOSE(outSdf.empty(), "Expected empty contact detection result.");
  int const numPoints = isize(points);
  real const toleranceInGridSpace = params.tolerance / _actorFromGridScale;
  bool hasReserved = false;

  // 4x4 matrices used to transform points between point-space and grid-space.
  // They are transposed because DotVecMat(vec, matT) is faster than DotMatVec(mat, vec).
  auto const actorFromPointsMatT = ToVMatrix4x4Transpose(Invert(pointsFromActor));
  auto const gridFromPointsMatT = Dot4x4(actorFromPointsMatT, _gridFromActorMatT);
  auto const& actorFromGridRotT = _gridFromActorRotation;

  // Transform the SDF's AABB into point-space. This gives us a quick way to reject points that are
  // outside the volume (often the majority) before we transform them into SDF-space. If the
  // requested tolerance is infinite, set these bounds to infinity to avoid any culling.
  Aabb boundsInGridSpace{-kInf3, kInf3};
  Aabb boundsInPointSpace{-kInf3, kInf3};
  if (IsFinite(toleranceInGridSpace)) {
    auto const pointsFromGridT = InvertTransformationTransposed(gridFromPointsMatT);
    boundsInGridSpace = ExpandShape(_distanceGrid->GetNegativeValueBounds(), toleranceInGridSpace);
    boundsInPointSpace = TransformShape_Transposed(pointsFromGridT, boundsInGridSpace);
  }

  static constexpr int kMaxBatchSize = 64 * Simd<real>::kSize;

  // We will perform a batch of signed distance calculations when this fills up
  int sdBatchSize = 0;
  Real3 sdPoints[kMaxBatchSize + 1] MOCHI_NO_INIT; // +1 for SIMD padding
  real distances[kMaxBatchSize] MOCHI_NO_INIT;
  int sdIndices[kMaxBatchSize] MOCHI_NO_INIT;

  // We will perform a batch of gradient calculations when this fills up
  int gradBatchSize = 0;
  Real3 gradPoints[kMaxBatchSize + 1] MOCHI_NO_INIT; // +1 for SIMD padding
  real gradDistances[kMaxBatchSize] MOCHI_NO_INIT;
  int gradIndices[kMaxBatchSize] MOCHI_NO_INIT;

  SamplerT sampler;

  auto flushGradBatch = [&]() {
    if (!hasReserved) {
      // Reserve memory the first time we know we have points to add.
      // Reserve the max size so we don't have to allocate again.
      outIndices.reserve(numPoints);
      outContacts.reserve(numPoints);
      outSdf.reserve(numPoints);
      hasReserved = true;
    }
    Real3 gradients[kMaxBatchSize + 1] MOCHI_NO_INIT; // +1 for SIMD padding
    sampler.Gradient(
        *_distanceGrid, Span{&gradPoints[0], gradBatchSize}, Span{&gradients[0], gradBatchSize});
    for (int i = 0; i < gradBatchSize; ++i) {
      int pointIndex = gradIndices[i];
      // The caller expects results in actor-space.
      Vec4r point = DotVecMat4x4(ToSimd(gradPoints[i], 1_r), _actorFromGridMatT);
      // Rotate the gradient vector into actor-space using DotVecMat3x3 with the transpose of the
      // rotation matrix on the right (the transpose is the inverse in this case).
      Vec4r grad = DotVecMat3x3(Load<Vec4r>(gradients[i].data()), actorFromGridRotT);
      real sd = gradDistances[i] * _actorFromGridScale;

      // Output the result
      outIndices.push_back(pointIndex);
      outContacts.push_back(ToReal3(point));
      outSdf.push_back(sd, ToReal3(grad));
    }
    gradBatchSize = 0;
  };

  auto flushSdBatch = [&]() {
    // Transpose points
    sampler(*_distanceGrid, Span{&sdPoints[0], sdBatchSize}, Span{&distances[0], sdBatchSize});
    for (int i = 0; i < sdBatchSize; ++i) {
      if (distances[i] <= toleranceInGridSpace) {
        MOCHI_ASSERT_VERBOSE(gradBatchSize < std::size(gradPoints));
        gradPoints[gradBatchSize] = sdPoints[i];
        gradIndices[gradBatchSize] = sdIndices[i];
        gradDistances[gradBatchSize] = distances[i];
        gradBatchSize++;
        if (gradBatchSize == kMaxBatchSize)
          MOCHI_UNLIKELY {
            flushGradBatch();
          }
      }
    }
    sdBatchSize = 0;
  };

  auto processPoint = [&](Vec4r pt, int index) {
    Vec4r ptInGridSpace = DotVecMat4x4(ToSimdPoint(pt), gridFromPointsMatT);
    if (ContainsPoint(boundsInGridSpace, ptInGridSpace)) {
      Store(sdPoints[sdBatchSize].data(), ptInGridSpace);
      sdIndices[sdBatchSize] = index;
      sdBatchSize++;
      if (sdBatchSize == kMaxBatchSize)
        MOCHI_UNLIKELY {
          flushSdBatch();
        }
    }
  };

  // Iterate 3 points at a time. Stop before the last point so we can use full-size SIMD loads.
  // Cull points using boundsInPointSpace first (faster than transforming them to SDF-space).
  int i = 0;
  for (i = 0; i + 3 < numPoints; i += 3) {
    auto pt0 = Load<Vec4r>(points[i + 0].data());
    auto pt1 = Load<Vec4r>(points[i + 1].data());
    auto pt2 = Load<Vec4r>(points[i + 2].data());
    auto in0 = VContainsPoint(boundsInPointSpace, pt0);
    auto in1 = VContainsPoint(boundsInPointSpace, pt1);
    auto in2 = VContainsPoint(boundsInPointSpace, pt2);
    if (!AllTrue<3>(in0 | in1 | in2)) {
      continue; // All of these points are outside
    }
    if (AllTrue<3>(in0)) {
      processPoint(pt0, i + 0);
    }
    if (AllTrue<3>(in1)) {
      processPoint(pt1, i + 1);
    }
    if (AllTrue<3>(in2)) {
      processPoint(pt2, i + 2);
    }
  }
  for (; i < numPoints; ++i) {
    auto pt = Load<3, Vec4r>(points[i].data());
    if (ContainsPoint(boundsInPointSpace, pt)) {
      processPoint(pt, i);
    }
  }

  if (sdBatchSize > 0) {
    flushSdBatch();
  }

  if (gradBatchSize > 0) {
    flushGradBatch();
  }
}

void GridSdf::FindPointContacts(
    Span<Real3 const> points,
    TransformRT const& pointsFromActor,
    ContactDetectionParams const& params,
    DynamicArray<int>& outIndices,
    DynamicArray<Real3>& outContacts,
    SdfInfo& outSdf,
    bool& outIsSdfGradUnitary) const {
  MOCHI_PROFILE_SCOPE();

  outIsSdfGradUnitary = false; // Grid SDF gradient may not be unitary.

  // If the grid bounds are at least as large as the collider bounds + tolerance, then
  // we can use the faster method of sampling interior points.
  auto colliderBounds = _distanceGrid->GetNegativeValueBounds();
  auto gridBounds = _distanceGrid->GetBounds();
  real const minGridPadding =
      Min(Min(colliderBounds.GetMin() - gridBounds.GetMin()),
          Min(gridBounds.GetMax() - colliderBounds.GetMax()));
  MOCHI_ASSERT_VERBOSE(
      IsFinite(minGridPadding) && minGridPadding >= 0_r,
      "Grid SDF negative-value bounds must be finite and contained within grid bounds.");
  real const toleranceInGridSpace = params.tolerance / _actorFromGridScale;
  if (minGridPadding >= toleranceInGridSpace) {
    FindPointContactsImpl<TrilinearSdfGridInteriorSampler<real>>(
        points, pointsFromActor, params, outIndices, outContacts, outSdf);
  } else {
    FindPointContactsImpl<TrilinearSdfGridUpperBoundSampler<real>>(
        points, pointsFromActor, params, outIndices, outContacts, outSdf);
  }
}

void GridSdf::GetGridSdfData(GridSdfData& outData) const {
  outData = {};
  outData.dims = _distanceGrid->GetDimensions();
  outData.values = _distanceGrid->GetData();
  outData.bounds = _distanceGrid->GetBounds();
  outData.negativeValueBounds = GetAabb(_distanceGrid->GetNegativeValueBounds());

  VMatrix4x4r actorFromGrid = Transpose4x4(_actorFromGridMatT);
  auto [scale, rt] = DecomposeMatrixTransform(actorFromGrid);
  if (!NearEqual(scale, Real3{1_r, 1_r, 1_r})) {
    outData.scale = scale;
  }
  if (!EquivalentRotation(rt.GetRotation(), Quaternion::Identity())) {
    outData.rotation = rt.GetRotation();
  }
  if (!NearEqual(rt.GetTranslation(), Real3{})) {
    outData.translation = rt.GetTranslation();
  }
}

} // namespace mochi
