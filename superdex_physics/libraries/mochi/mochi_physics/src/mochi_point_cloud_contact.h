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
#include "mochi_contact.h"
#include "mochi_discretization_components.h"

#include <mochi_core/contact/contact_utils.h>
#include <mochi_core/geometry/spatial_hash_table.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/overload_visitor.h>

#include <variant>

namespace mochi {

// A lightweight pseudo-element that makes the nodal collider representation look like a degenerate
// quadrature case: one "element" per mesh node, with exactly 1 DOF and 1 quadrature point whose
// basis function value is 1.0.
struct NodalColliderElement {
  static constexpr int kSpaceDim = 3;
  static constexpr int kNumDofs = 1;
  static constexpr int kNumQuadPoints = 1;
  // clang-format off
  static inline NdArray<real, 1, 1> const kBasisEvaluated = [] { NdArray<real, 1, 1> b{}; b[0][0] = 1_r; return b; }();
  // clang-format on

  int const elementIndex;
  Span<Real3 const> const coordinates;
  // For the nodal case, the "element" is a single node, so the connectivity is trivial.
  struct NodalConnectivity {
    int nodeIndex;
    NdArray<int, 1> operator[](int) const {
      return {nodeIndex};
    }
  };
  NodalConnectivity const connectivity;

  NdArray<real, 1, kSpaceDim> mapEvaluated;
  NdArray<real, 1> quadWeights;

  NodalColliderElement(
      int elementIndex,
      Span<Real3 const> coordinates,
      int globalNodeIndex,
      real weight)
      : elementIndex(elementIndex), coordinates(coordinates), connectivity{globalNodeIndex} {
    for (int d = 0; d < kSpaceDim; ++d) {
      mapEvaluated[0][d] = coordinates[globalNodeIndex][d];
    }
    quadWeights[0] = weight;
  }
};

using NodalColliderDiscretization = FemDiscretization<NodalColliderElement>;

// Collider-side point cloud discretization that supports nodal, surface (quadrature-based), and
// segment (1D curve) representations. Uses a multi-level dispatch: check for
// NodalColliderDiscretization first, then CFemSegmentDiscretization, then fall through to
// CFemSurfaceDiscretization::Visit() for the FEM surface element types. The dispatch happens once
// per algorithm function call (not in hot loops), so the cost is negligible.
struct CColliderPointCloudDiscretization {
  using variant_t = std::
      variant<NodalColliderDiscretization, CFemSurfaceDiscretization, CFemSegmentDiscretization>;
  variant_t value;

  // Number of DoFs per node in the displacement array. Must be set explicitly by the caller at
  // construction time (use kSpaceDim3 = 3 for spatial-only DoF layouts — shell, surface, and
  // nodal colliders — and fem::kNumRodFields = 4 for rods, whose nodes carry 3 displacement +
  // 1 twist DoF and where only the spatial components participate in position interpolation).
  int dofsPerNode{};

  explicit CColliderPointCloudDiscretization(NodalColliderDiscretization&& disc, int dofsPerNode)
      : value(std::move(disc)), dofsPerNode(dofsPerNode) {}
  explicit CColliderPointCloudDiscretization(CFemSurfaceDiscretization&& disc, int dofsPerNode)
      : value(std::move(disc)), dofsPerNode(dofsPerNode) {}
  explicit CColliderPointCloudDiscretization(CFemSegmentDiscretization&& disc, int dofsPerNode)
      : value(std::move(disc)), dofsPerNode(dofsPerNode) {}

  // Dispatches to the visitor with a concrete FemDiscretization<ElementT>. For the nodal case,
  // the visitor receives FemDiscretization<NodalColliderElement>. For the segment case, it
  // dispatches through CFemSegmentDiscretization::Visit(). For the surface case, it dispatches
  // through CFemSurfaceDiscretization::Visit() to resolve the concrete Pk2DElement type.
  template <typename Visitor>
  auto VisitCollider(Visitor&& vis) const {
    return std::visit(
        OverloadVisitor{
            [&](NodalColliderDiscretization const& nodal) { return vis(nodal); },
            [&](CFemSurfaceDiscretization const& fem) {
              return fem.Visit(std::forward<Visitor>(vis));
            },
            [&](CFemSegmentDiscretization const& segment) {
              return segment.Visit(std::forward<Visitor>(vis));
            }},
        value);
  }

  int GetNumColliderPoints() const {
    return VisitCollider([](auto const& disc) { return disc.GetNumQuadPoints(); });
  }
};

namespace details {
// Interpolates the current position of a collider point from the element's reference-configuration
// map and the nodal displacements.
template <typename DiscretizationT, typename ElementT>
MOCHI_FORCE_INLINE Real3 InterpolateColliderPointPosition(
    ElementT const& element,
    int elementIndex,
    int localQuadIndex,
    ColumnVectorView<real const> displacements,
    int dofsPerNode) {
  int constexpr kNumEleNodes = DiscretizationT::kNumEleNodes;
  Real3 position = element.mapEvaluated[localQuadIndex];
  for (int k = 0; k < kNumEleNodes; ++k) {
    int const globalNodeIndex = element.connectivity[elementIndex][k];
    int const dofOffset = globalNodeIndex * dofsPerNode;
    real const basisVal = DiscretizationT::ElementT::kBasisEvaluated[localQuadIndex][k];
    position += basisVal *
        Real3{displacements[dofOffset], displacements[dofOffset + 1], displacements[dofOffset + 2]};
  }
  return position;
}
} // namespace details

// Wrapper for the colliding-side discretization in a point-cloud collision pair. For shell this
// is CFemSurfaceDiscretization; for rods it is CFemSegmentDiscretization. Both types share a
// duck-typed Visit() interface, so we dispatch through std::variant of pointers. Currently
// consumed only on the self-contact path of @ref ComputePointsToColliderPoints.
struct CollidingPointCloudDiscretization {
  std::variant<CFemSurfaceDiscretization const*, CFemSegmentDiscretization const*> value;

  template <typename Visitor>
  auto Visit(Visitor&& vis) const {
    return std::visit(
        OverloadVisitor{
            [&](CFemSurfaceDiscretization const* surf) {
              return surf->Visit(std::forward<Visitor>(vis));
            },
            [&](CFemSegmentDiscretization const* seg) {
              return seg->Visit(std::forward<Visitor>(vis));
            }},
        value);
  }
};
/// Validates shared @ref experimental::PointCloudColliderParams fields.
void ValidatePointCloudColliderParams(
    experimental::PointCloudColliderParams const& params,
    Error& error);

SpatialHashTable CreateSpatialHashTable(
    experimental::PointCloudColliderParams const& params,
    CColliderPointCloudDiscretization const& colliderDiscretization,
    real contactThreshold);

DynamicArray<DynamicArray<int>> ComputePointsToColliderPoints(
    experimental::PointCloudColliderParams const& pointCloudColliderParams,
    CColliderPointCloudDiscretization const& colliderDiscretization,
    ColumnVectorView<real const> colliderDisplacements,
    TransformRT const& worldFromCollider,
    CollidingPointCloudDiscretization const& collidingDiscretization,
    bool selfContact,
    Span<Real3 const> collidingPointPositions,
    Span<int const> collidingPointSampleIndices,
    TransformRT const& worldFromColliding,
    SpatialHashTable const& colliderHashTable,
    real contactThreshold);

// Allocates and populates pointIndices and colliderPointIndices based on the secondary culling
// done by ComputePointsToColliderPoints.
void ComputePointCloudContactIndices(
    DynamicArray<DynamicArray<int>> const& pointsToColliderPoints,
    DynamicArray<int>& outPointIndices,
    DynamicArray<int>& outColliderPointIndices);

// Populates the remaining ContactDetectionResult fields using point indices of the form provided
// by ComputePointCloudContactIndices.
void ComputePointCloudContactDetectionFields(
    CColliderPointCloudDiscretization const& colliderDiscretization,
    ColumnVectorView<real const> colliderDisplacements,
    TransformRT const& worldFromCollider,
    Span<Real3 const> collidingPointPositions,
    TransformRT const& worldFromColliding,
    Span<int const> pointIndices,
    Span<int const> colliderPointIndices,
    real radius,
    DynamicArray<Real3>& outPosColliding,
    SdfInfo* outSdfInfo,
    int* outNdofs,
    DynamicArray<real>* outColliderIntegrationWeights,
    DynamicArray<VMatrix3x3r>* outJacColliderFromWorld,
    DynamicArray<ColliderJacDofs>* outJacWorldFromDofs);

/// Assembles nodal weights from any FemDiscretization by accumulating basis-weighted quadrature.
/// Works for surface, segment, or any element type with the standard FemDiscretization interface.
template <typename ElementT>
DynamicArray<real> InitializeNodalWeights(FemDiscretization<ElementT> const& femDiscretization) {
  int constexpr kNumNodes = FemDiscretization<ElementT>::kNumEleNodes;
  int constexpr kNumQuadPoints = FemDiscretization<ElementT>::kNumQuads;
  int const numElements = isize(femDiscretization.femElements);
  MOCHI_ASSERT(numElements > 0, "Attempting to compute nodal weights with no elements");
  int const numNodes = isize(femDiscretization.femElements[0].coordinates);
  DynamicArray<real> nodalWeights(numNodes, 0_r);
  for (int elementIndex = 0; elementIndex < numElements; ++elementIndex) {
    auto const& element = femDiscretization.femElements[elementIndex];
    for (int localNodeIndex = 0; localNodeIndex < kNumNodes; ++localNodeIndex) {
      int const globalNodeIndex = element.connectivity[elementIndex][localNodeIndex];
      for (int quadPointIndex = 0; quadPointIndex < kNumQuadPoints; ++quadPointIndex) {
        real const basisEval = ElementT::kBasisEvaluated[quadPointIndex][localNodeIndex];
        real const quadPointWeight = element.quadWeights[quadPointIndex];
        nodalWeights[globalNodeIndex] += basisEval * quadPointWeight;
      }
    }
  }
  return nodalWeights;
}

using CSpatialHashTable = SpatialHashTable;
struct CNodalWeights {
  CNodalWeights(DynamicArray<real> valuesIn) : values(std::move(valuesIn)) {}
  DynamicArray<real> values;
};

void UpdateSpatialHashTable(
    ecs::Included<TagUsePointCloudContact>,
    CColliderPointCloudDiscretization const& colliderDiscretization,
    CFinalDisplacementRef<TimeStep::Current> const& colliderDisplacementsRef,
    CSpatialHashTable& outHashTable);

namespace point_cloud_contact {
void InitializeOnce(entt::registry& reg);
} // namespace point_cloud_contact

} // namespace mochi
