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

#include <mochi_core/geometry/any_shape.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/grid_sdf_params.h>
#include <mochi_core/geometry/sdf.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/vmatrix.h>

#include <memory>

namespace mochi {

// Forwards
struct GridSdfData;

/*
  Dense grid real field containing the SDF of a triangular mesh. The size and resolution of the
  scalar field is configured through GridSdfParams.
*/
class GridSdf final : public Sdf {
 public:
  // Creates an empty SDF. Requires a call to Initialize.
  ~GridSdf() override = default;

  // Creates a GridSdf of the specified triangular mesh. See GridSdfParams.
  explicit GridSdf(
      std::shared_ptr<TriangularMesh const> const& mesh,
      GridSdfParams const& params,
      Error& error);

  // Create a GridSdf by reference counting an existing DenseGrid3D, and applying a 3D
  // transformation relative to the local space of the owning actor.
  GridSdf(std::shared_ptr<DenseGrid3D<real> const> const& grid, VMatrix4x4r const& actorFromGrid);

  // Create a GridSdf from serializable data.
  static GridSdf Create(GridSdfData&& data);

  // Get a copy of the data for use with ModelData (and related utilities).
  void GetGridSdfData(GridSdfData& outData) const;

  // ScalarField interface:
  [[nodiscard]] AnyShape GetColliderBounds() const override {
    return _colliderBoundsInActorSpace;
  }

  void FindPointContacts(
      Span<Real3 const> points,
      TransformRT const& pointsFromActor,
      ContactDetectionParams const& params,
      DynamicArray<int>& outIndices,
      DynamicArray<Real3>& outContacts,
      SdfInfo& outSdf,
      bool& outIsSdfGradUnitary) const override;

  [[nodiscard]] DenseGrid3D<real> const& GetDistanceGrid() const {
    MOCHI_ASSERT_VERBOSE(_distanceGrid != nullptr);
    return *_distanceGrid;
  }

  /// Number of grid cells per axis (X, Y, Z). In grids computed from a mesh, it reflects the
  /// resolution covering the mesh bounds (excluding extra boundary layers, if any).
  [[nodiscard]] Int3 GetCellResolution() const {
    MOCHI_ASSERT_VERBOSE(_distanceGrid != nullptr);
    return _distanceGrid->GetDimensions() - Int3{1, 1, 1};
  }

  /// True if the mesh used to compute the SDF was topologically closed.
  [[nodiscard]] bool IsMeshClosed() const {
    return _isMeshClosed;
  }

  [[nodiscard]] auto const& GetGridFromActorTranspose() const {
    return _gridFromActorMatT;
  }

  [[nodiscard]] real GetActorFromGridScale() const {
    return _actorFromGridScale;
  }

 private:
  // Computes the SDF grid from a triangular mesh.
  void Initialize(
      std::shared_ptr<TriangularMesh const> const& mesh,
      GridSdfParams const& params,
      Error& error);

  // Populates the distance grid and returns whether the mesh is topologically closed.
  static bool InitializeBruteForce(
      std::shared_ptr<TriangularMesh const> const& mesh,
      DenseGrid3D<real>& outDistanceGrid);

  template <class SamplerT>
  void FindPointContactsImpl(
      Span<Real3 const> points,
      TransformRT const& pointsFromActor,
      ContactDetectionParams const& params,
      DynamicArray<int>& outIndices,
      DynamicArray<Real3>& outContacts,
      SdfInfo& outSdf) const;

  // Uniform scale of the grid, relative to the actor's coordinate space.
  real _actorFromGridScale = 1_r;

  // 3x3 matrix which rotates vectors from actor-space to grid-space. To rotate from grid-space to
  // actor-space, simply transpose this matrix, or call DotVecMat3x3 such that the matrix is on the
  // right. This matrix has UNIT SCALE.
  VMatrix3x3r _gridFromActorRotation = VEye<3>();

  // 4x4 matrices which scale, rotate, and translate points from actor-space to grid-space, and vice
  // versa. These matrices may NOT have unit scale.
  VMatrix4x4r _gridFromActorMatT = VEye<4>();
  VMatrix4x4r _actorFromGridMatT = VEye<4>();

  // Bounds of the mesh boundary (where signed distance is zero) in actor-space.
  Aabb _colliderBoundsInActorSpace;

  // Reference counted distance grid (possibly shared with other GridSdf instances)
  std::shared_ptr<DenseGrid3D<real> const> _distanceGrid;

  // True if the mesh used to compute the SDF was topologically closed.
  bool _isMeshClosed = true;
};

} // namespace mochi
