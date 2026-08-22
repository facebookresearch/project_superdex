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

#include "mochi_point_cloud_contact.h"

#include <mochi_core/contact/contact_utils.h>
#include <mochi_core/geometry/geometry_utils.h>

#include <limits>

namespace mochi {

void ValidatePointCloudColliderParams(
    experimental::PointCloudColliderParams const& params,
    Error& error) {
  MOCHI_ERROR_IF_NOT(params.radius > 0_r, error, "Point-cloud collider radius must be positive.");
  MOCHI_ERROR_IF_NOT(
      !params.selfContact || params.selfContactExclusionRatio > 1_r,
      error,
      "Point-cloud self-contact exclusion ratio must be > 1.");
  MOCHI_ERROR_IF_NOT(
      params.spatialHashLoadFactor > 0_r, error, "Hash table load factor must be positive.");
}

SpatialHashTable CreateSpatialHashTable(
    experimental::PointCloudColliderParams const& params,
    CColliderPointCloudDiscretization const& colliderDiscretization,
    real contactThreshold) {
  int const numColliderPoints = colliderDiscretization.GetNumColliderPoints();
  MOCHI_ASSERT(numColliderPoints > 0, "Collider discretization must have at least one point");
  auto const minNumBinsDouble = static_cast<double>(
      Ceil(static_cast<real>(numColliderPoints) / params.spatialHashLoadFactor));
  MOCHI_ASSERT(
      IsFinite(minNumBinsDouble) && minNumBinsDouble >= 1.0 &&
          minNumBinsDouble <= static_cast<double>(std::numeric_limits<int>::max()),
      "Point-cloud spatial hash load factor produced an invalid bin count.");
  auto const minNumBins = static_cast<int>(minNumBinsDouble);
  return {params.radius + contactThreshold, numColliderPoints, minNumBins};
}

void UpdateSpatialHashTable(
    ecs::Included<TagUsePointCloudContact>,
    CColliderPointCloudDiscretization const& colliderDiscretization,
    CFinalDisplacementRef<TimeStep::Current> const& colliderDisplacementsRef,
    CSpatialHashTable& outHashTable) {
  int const dofsPerNode = colliderDiscretization.dofsPerNode;
  MOCHI_ASSERT_VERBOSE(
      dofsPerNode > 0,
      "CColliderPointCloudDiscretization::dofsPerNode was not set (zero sentinel)");
  colliderDiscretization.VisitCollider([&](auto const& disc) {
    using DiscretizationT = std::decay_t<decltype(disc)>;
    static_assert(DiscretizationT::kSpaceDim == kSpaceDim3, "Invalid spatial dimension");
    int constexpr kNumEleNodes = DiscretizationT::kNumEleNodes;
    int const numElements = isize(disc.femElements);
    MOCHI_ASSERT(numElements > 0, "Collider discretization must have at least one element");
    ColumnVectorView<real const> colliderDisplacements = colliderDisplacementsRef.value;

    MOCHI_ASSERT_VERBOSE(
        outHashTable.GetCapacity() == numElements * DiscretizationT::kNumQuads,
        "Inconsistent hash table capacity");
    outHashTable.Reset();

    for (int elementIndex = 0; elementIndex < numElements; ++elementIndex) {
      auto const& element = disc.femElements[elementIndex];
      Real3 nodeDisplacements[kNumEleNodes];
      for (int k = 0; k < kNumEleNodes; ++k) {
        int const globalNodeIndex = element.connectivity[elementIndex][k];
        int const dofOffset = globalNodeIndex * dofsPerNode;
        nodeDisplacements[k] = {
            colliderDisplacements[dofOffset],
            colliderDisplacements[dofOffset + 1],
            colliderDisplacements[dofOffset + 2]};
      }
      for (int q = 0; q < DiscretizationT::kNumQuads; ++q) {
        int const colliderPointIndex = elementIndex * DiscretizationT::kNumQuads + q;
        Real3 position = element.mapEvaluated[q];
        for (int k = 0; k < kNumEleNodes; ++k) {
          real const basisVal = DiscretizationT::ElementT::kBasisEvaluated[q][k];
          position += basisVal * nodeDisplacements[k];
        }
        outHashTable.AddPoint(colliderPointIndex, position);
      }
    }
  });
}

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
    real contactThreshold) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(
      colliderHashTable.GetNumPoints() == colliderDiscretization.GetNumColliderPoints(),
      "Point-cloud collider hash table must be populated before collision queries.");

  int const numCollidingPoints = isize(collidingPointPositions);
  int constexpr kColliderPointsPerPointEstimate = 8;
  DynamicArray<DynamicArray<int>> pointsToColliderPoints(numCollidingPoints);

  real const contactRange = pointCloudColliderParams.radius + contactThreshold;
  real const contactRangeSquared = Sqr(contactRange);
  real const selfContactExclusionRangeSquared =
      Sqr(pointCloudColliderParams.radius * pointCloudColliderParams.selfContactExclusionRatio +
          contactThreshold);
  Aabb const queryBounds = ExpandShape(colliderHashTable.GetOccupiedPointBounds(), contactRange);

  // Transformation from colliding local coordinates to collider local coordinates.
  TransformRT const colliderFromColliding = Invert(worldFromCollider) * worldFromColliding;
  VMatrix4x4r const colliderFromCollidingT = ToVMatrix4x4Transpose(colliderFromColliding);

  int const dofsPerNode = colliderDiscretization.dofsPerNode;
  MOCHI_ASSERT_VERBOSE(
      dofsPerNode > 0,
      "CColliderPointCloudDiscretization::dofsPerNode was not set (zero sentinel)");

  // Resolve the collider discretization type once, outside the parallel loop.
  colliderDiscretization.VisitCollider([&](auto const& disc) {
    using DiscretizationT = std::decay_t<decltype(disc)>;

    // Safe to parallelize: each colliding point writes to a separate row of
    // pointsToColliderPoints, and the hash table is read-only.
    int constexpr kMinPerTask = 256;
    ParallelForN(
        "PointCloudCollisionDetection",
        numCollidingPoints,
        kMinPerTask,
        [&](int collidingPointIndex) {
          DynamicArray<int>& colliderPointIndices = pointsToColliderPoints[collidingPointIndex];

          Vec4r const collidingPointPositionCollidingFrame =
              Load<kSpaceDim3, Vec4r>(&(collidingPointPositions[collidingPointIndex][0]));
          Vec4r const collidingPointPositionColliderFrame = DotVecMat4x4(
              ToSimdPoint(collidingPointPositionCollidingFrame), colliderFromCollidingT);

          if (!ContainsPoint(queryBounds, collidingPointPositionColliderFrame)) {
            return;
          }

          Vec4r collidingPointReferencePositionColliderFrame MOCHI_NO_INIT;
          if (selfContact) {
            collidingDiscretization.Visit([&](auto const& collidingDisc) {
              using CollidingDiscretizationT = std::decay_t<decltype(collidingDisc)>;
              int const sampleIndex = collidingPointSampleIndices.empty()
                  ? collidingPointIndex
                  : collidingPointSampleIndices[collidingPointIndex];
              int const collidingElementIndex = sampleIndex / CollidingDiscretizationT::kNumQuads;
              int const collidingLocalQuadPointIndex =
                  sampleIndex % CollidingDiscretizationT::kNumQuads;
              Vec4r const collidingReferencePositionCollidingFrame =
                  ToSimd(collidingDisc.femElements[collidingElementIndex]
                             .mapEvaluated[collidingLocalQuadPointIndex]);
              collidingPointReferencePositionColliderFrame = DotVecMat4x4(
                  ToSimdPoint(collidingReferencePositionCollidingFrame), colliderFromCollidingT);
            });
          }

          colliderHashTable.IteratePointsNearPosition(
              ToReal3(collidingPointPositionColliderFrame), [&](int colliderPointIndex) {
                int const elementIndex = colliderPointIndex / DiscretizationT::kNumQuads;
                int const localQuadIndex = colliderPointIndex % DiscretizationT::kNumQuads;
                auto const& element = disc.femElements[elementIndex];

                // Self-contact exclusion using reference collider point position.
                if (selfContact) {
                  Vec4r const colliderRefPos = ToSimd(element.mapEvaluated[localQuadIndex]);
                  Vec4r const referenceDisplacement =
                      colliderRefPos - collidingPointReferencePositionColliderFrame;
                  if (NormSqr<3>(referenceDisplacement) < selfContactExclusionRangeSquared) {
                    return;
                  }
                }

                // Interpolate the current collider point position.
                Real3 const colliderPointPos =
                    details::InterpolateColliderPointPosition<DiscretizationT>(
                        element, elementIndex, localQuadIndex, colliderDisplacements, dofsPerNode);
                Vec4r const colliderPointPositionColliderFrame =
                    Load<kSpaceDim3, Vec4r>(&colliderPointPos[0]);

                // Distance culling.
                if (NormSqr<3>(
                        collidingPointPositionColliderFrame - colliderPointPositionColliderFrame) >=
                    contactRangeSquared) {
                  return;
                }

                if (colliderPointIndices.empty()) {
                  colliderPointIndices.reserve(kColliderPointsPerPointEstimate);
                }
                colliderPointIndices.push_back(colliderPointIndex);
              });
        });
  });
  return pointsToColliderPoints;
}

void ComputePointCloudContactIndices(
    DynamicArray<DynamicArray<int>> const& pointsToColliderPoints,
    DynamicArray<int>& outPointIndices,
    DynamicArray<int>& outColliderPointIndices) {
  MOCHI_PROFILE_SCOPE();

  int const numCollidingPoints = isize(pointsToColliderPoints);

  int totalContacts = 0;
  for (int collidingPointIndex = 0; collidingPointIndex < numCollidingPoints;
       ++collidingPointIndex) {
    totalContacts += isize(pointsToColliderPoints[collidingPointIndex]);
  }

  outPointIndices.resize_noinit(totalContacts);
  outColliderPointIndices.resize_noinit(totalContacts);

  if (totalContacts == 0) {
    return;
  }

  int contactIndex = 0;
  for (int collidingPointIndex = 0; collidingPointIndex < numCollidingPoints;
       ++collidingPointIndex) {
    for (int const colliderPointIndex : pointsToColliderPoints[collidingPointIndex]) {
      outPointIndices[contactIndex] = collidingPointIndex;
      outColliderPointIndices[contactIndex] = colliderPointIndex;
      ++contactIndex;
    }
  }
}

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
    DynamicArray<ColliderJacDofs>* outJacWorldFromDofs) {
  MOCHI_PROFILE_SCOPE();
  int const totalContacts = isize(pointIndices);
  if (totalContacts == 0) {
    return;
  }

  int const dofsPerNode = colliderDiscretization.dofsPerNode;
  MOCHI_ASSERT_VERBOSE(
      dofsPerNode > 0,
      "CColliderPointCloudDiscretization::dofsPerNode was not set (zero sentinel)");
  colliderDiscretization.VisitCollider([&](auto const& disc) {
    using DiscretizationT = std::decay_t<decltype(disc)>;
    static_assert(DiscretizationT::kSpaceDim == kSpaceDim3, "Invalid spatial dimension");
    int constexpr kNumEleNodes = DiscretizationT::kNumEleNodes;

    // Each collider point has kNumEleNodes * kSpaceDim3 spatial DoFs.
    if (outNdofs != nullptr) {
      *outNdofs = kSpaceDim3 * kNumEleNodes;
    }

    int const numElements = isize(disc.femElements);
    if (numElements == 0)
      MOCHI_UNLIKELY {
        return;
      }

    // Populate collider integration weights from element quadrature weights.
    if (outColliderIntegrationWeights != nullptr) {
      outColliderIntegrationWeights->resize_noinit(totalContacts);
      for (int contactIndex = 0; contactIndex < totalContacts; ++contactIndex) {
        int const cpIndex = colliderPointIndices[contactIndex];
        int const elementIndex = cpIndex / DiscretizationT::kNumQuads;
        int const localQuadIndex = cpIndex % DiscretizationT::kNumQuads;
        (*outColliderIntegrationWeights)[contactIndex] =
            disc.femElements[elementIndex].quadWeights[localQuadIndex];
      }
    }

    // Transformation from colliding local coordinates to collider local coordinates.
    VMatrix4x4r const colliderFromCollidingT =
        ToVMatrix4x4Transpose(Invert(worldFromCollider) * worldFromColliding);
    VMatrix3x3r const colliderRotT = ToVMatrix3x3Transpose(worldFromCollider.GetRotation());

    outPosColliding.resize_noinit(totalContacts);
    if (outSdfInfo != nullptr) {
      outSdfInfo->resize_noinit(totalContacts);
    }
    if (outJacWorldFromDofs != nullptr) {
      outJacWorldFromDofs->resize_noinit(totalContacts);
    }
    if (outJacColliderFromWorld != nullptr) {
      outJacColliderFromWorld->resize_noinit(1);
      (*outJacColliderFromWorld)[0] = colliderRotT;
    }

    int constexpr kMinPerTask = 256;
    ParallelForN(
        "PointCloudContactDetectionFields", totalContacts, kMinPerTask, [&](int contactIndex) {
          int const collidingPointIndex = pointIndices[contactIndex];
          int const cpIndex = colliderPointIndices[contactIndex];
          int const elementIndex = cpIndex / DiscretizationT::kNumQuads;
          int const localQuadIndex = cpIndex % DiscretizationT::kNumQuads;
          auto const& element = disc.femElements[elementIndex];

          Vec4r const collidingPointPosition =
              Load<kSpaceDim3, Vec4r>(&(collidingPointPositions[collidingPointIndex][0]));

          // Interpolate collider point position from element nodes.
          Real3 const colliderPointPos = details::InterpolateColliderPointPosition<DiscretizationT>(
              element, elementIndex, localQuadIndex, colliderDisplacements, dofsPerNode);
          Vec4r const colliderPointPosition = Load<kSpaceDim3, Vec4r>(&colliderPointPos[0]);

          // Transform the colliding point to the collider's local frame.
          Vec4r const collidingPositionInCollider =
              DotVecMat4x4(ToSimdPoint(collidingPointPosition), colliderFromCollidingT) -
              colliderPointPosition;

          Store<kSpaceDim3>(&(outPosColliding[contactIndex][0]), collidingPositionInCollider);

          if (outSdfInfo != nullptr) {
            real const physicalDistance = Norm<3>(collidingPositionInCollider);
            outSdfInfo->val[contactIndex] = physicalDistance - radius;
            Vec4r sdfGrad =
                collidingPositionInCollider / (physicalDistance + std::numeric_limits<real>::min());
            Store<kSpaceDim3>(&(outSdfInfo->grad[contactIndex][0]), sdfGrad);
          }

          // Populate Jacobians weighted by basis functions.
          if (outJacWorldFromDofs != nullptr) {
            ColliderJacDofs& jacWorldFromDofs = (*outJacWorldFromDofs)[contactIndex];
            for (int k = 0; k < DiscretizationT::kNumEleNodes; ++k) {
              real const basisVal = DiscretizationT::ElementT::kBasisEvaluated[localQuadIndex][k];
              int const globalNodeIndex = element.connectivity[elementIndex][k];
              for (int d = 0; d < kSpaceDim3; ++d) {
                jacWorldFromDofs.jac[kSpaceDim3 * k + d] = basisVal * colliderRotT[d];
                jacWorldFromDofs.inds[kSpaceDim3 * k + d] = globalNodeIndex * dofsPerNode + d;
              }
            }
          }
        });
  });
}

namespace point_cloud_contact {
void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CPointCloudColliderParams>(reg);
  ecs::RegisterComponent<CSpatialHashTable>(reg);
  ecs::RegisterComponent<CNodalWeights>(reg);
  ecs::RegisterComponent<CColliderPointCloudDiscretization>(reg);
  ecs::RegisterComponent<TagUsePointCloudContact>(reg);
}
} // namespace point_cloud_contact
} // namespace mochi
