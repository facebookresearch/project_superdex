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

#include <mochi_core/contact/contact_utils.h>
#include <mochi_core/geometry/tetrahedral_map.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/vmatrix.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>

using namespace mochi;

// Apply collider integration weights from a ContactDetectionResult to a CollisionResponseResult.
// This scales the energy, force, and dforce arrays by the corresponding collider weights.
// If colliderIntegrationWeights is empty, this function is a no-op.
static void ApplyColliderWeightsToCollisionResponse(
    ContactDetectionResult const& contactQuery,
    Interval<int> pointRange,
    CollisionResponseResult& outResponse) {
  if (contactQuery.colliderIntegrationWeights.empty()) {
    return;
  }
  bool const scaleEnergy = !outResponse.energy.empty();
  bool const scaleForce = !outResponse.force.empty();
  bool const scaleDForce = !outResponse.dforce.empty();
  for (auto s : pointRange) {
    real colliderWeight = contactQuery.colliderIntegrationWeights[s];
    if (scaleEnergy) {
      outResponse.energy[s] *= colliderWeight;
    }
    if (scaleForce) {
      outResponse.force[s] *= colliderWeight;
    }
    if (scaleDForce) {
      outResponse.dforce[s] *= colliderWeight;
    }
  }
}

template <GradTarget kGradTarget>
void mochi::ComputeCollisionResponseRange(
    Interval<int> pointRange,
    ContactDetectionResult const& contactQuery,
    ContactParams const& params,
    ContactEvalConfig const& config,
    real dtStage,
    bool assemEnergy,
    bool assemForce,
    bool assemDForce,
    CollisionResponseResult& outResponse) {
  MOCHI_PROFILE_SCOPE();

  // Outputs should already be the correct size
  MOCHI_ASSERT(!assemEnergy || (outResponse.energy.size() == isize(contactQuery.sampleIndices)));
  MOCHI_ASSERT(!assemForce || (outResponse.force.size() == isize(contactQuery.sampleIndices)));
  MOCHI_ASSERT(!assemDForce || (outResponse.dforce.size() == isize(contactQuery.sampleIndices)));

  MOCHI_ASSERT_VERBOSE(
      !config.explicitNormals ||
          contactQuery.sdfInfo.size() == contactQuery.sdfInfoStageStart.size(),
      "Explicit normals require stage-start sdf info");
  auto const& sdfInfoImplicit = contactQuery.sdfInfo;
  auto const& sdfInfoExplicit = contactQuery.sdfInfoStageStart;

  MOCHI_ASSERT_VERBOSE(
      !config.validCollidingNormals ||
          contactQuery.sdfInfo.size() == contactQuery.normalColliding.size(),
      "Valid colliding normals require normalColliding");

  // Compute the contact response of all contact samples, in batches to improve performance.
  constexpr int kBatchSize = Min(2 * Simd<real>::kSize, kCollResponseMaxBatchSize);
  for (int s = pointRange.Min(); s <= pointRange.Max(); s += kBatchSize) {
    int const numPoints = Min(pointRange.Max() - s + 1, kBatchSize);

#define MOCHI_COMPUTE_BATCH_COLLISION_FORCE_DFORCE(N)                                          \
  ComputeBatchCollisionForceDForce<N, kGradTarget>(                                            \
      assemEnergy ? MakeSpan(outResponse.energy).subspan(s, N) : Span<double>{},               \
      assemForce ? MakeSpan(outResponse.force).subspan(s, N) : Span<Real3>{},                  \
      assemDForce ? MakeSpan(outResponse.dforce).subspan(s, N) : Span<VMatrix3x3r>{},          \
      MakeConstSpan(sdfInfoImplicit.val).subspan(s, N),                                        \
      MakeConstSpan(sdfInfoImplicit.grad).subspan(s, N),                                       \
      config.explicitNormals ? MakeConstSpan(sdfInfoExplicit.val).subspan(s, N)                \
                             : Span<real const>{},                                             \
      config.explicitNormals ? MakeConstSpan(sdfInfoExplicit.grad).subspan(s, N)               \
                             : Span<Real3 const>{},                                            \
      config.validCollidingNormals ? MakeConstSpan(contactQuery.normalColliding).subspan(s, N) \
                                   : Span<Real3 const>{},                                      \
      MakeConstSpan(contactQuery.posColliding).subspan(s, N),                                  \
      MakeConstSpan(contactQuery.posCollidingStageStart).subspan(s, N),                        \
      params,                                                                                  \
      config,                                                                                  \
      dtStage,                                                                                 \
      assemEnergy,                                                                             \
      assemForce,                                                                              \
      assemDForce,                                                                             \
      contactQuery.isSdfGradUnitary);

    // Compute contact response
    switch (numPoints) {
      case 1:
        MOCHI_COMPUTE_BATCH_COLLISION_FORCE_DFORCE(1);
        break;
      case 2:
        MOCHI_COMPUTE_BATCH_COLLISION_FORCE_DFORCE(2);
        break;
      case 3:
        MOCHI_COMPUTE_BATCH_COLLISION_FORCE_DFORCE(3);
        break;
      case 4:
        MOCHI_COMPUTE_BATCH_COLLISION_FORCE_DFORCE(4);
        break;
      case 5:
        MOCHI_COMPUTE_BATCH_COLLISION_FORCE_DFORCE(5);
        break;
      case 6:
        MOCHI_COMPUTE_BATCH_COLLISION_FORCE_DFORCE(6);
        break;
      case 7:
        MOCHI_COMPUTE_BATCH_COLLISION_FORCE_DFORCE(7);
        break;
      case 8:
        MOCHI_COMPUTE_BATCH_COLLISION_FORCE_DFORCE(8);
        break;
      default:
        static_assert(kCollResponseMaxBatchSize == 8, "Please update this switch statement");
        MOCHI_ASSERT(false, "Unsupported batch size (%d).", numPoints);
    }
  }

#undef MOCHI_COMPUTE_BATCH_COLLISION_FORCE_DFORCE

  // Apply collider integration weights (e.g., for shell/shell contact).
  ApplyColliderWeightsToCollisionResponse(contactQuery, pointRange, outResponse);
}

#define MOCHI_SPECIALIZE_COMPUTE_COLLISION_RESPONSE_RANGE(kGradTarget) \
  template void mochi::ComputeCollisionResponseRange<kGradTarget>(     \
      Interval<int>,                                                   \
      ContactDetectionResult const&,                                   \
      ContactParams const&,                                            \
      ContactEvalConfig const&,                                        \
      real,                                                            \
      bool,                                                            \
      bool,                                                            \
      bool,                                                            \
      CollisionResponseResult&);
MOCHI_SPECIALIZE_COMPUTE_COLLISION_RESPONSE_RANGE(GradTarget::Current);
MOCHI_SPECIALIZE_COMPUTE_COLLISION_RESPONSE_RANGE(GradTarget::Previous);
#undef MOCHI_SPECIALIZE_COMPUTE_COLLISION_RESPONSE_RANGE

template <typename Bv>
int MeshColliderBvh<Bv>::FindClosestFaceBruteForce(Vec4r position, real& outDistSqr) const {
  auto const& faces = _mesh->GetElementConnectivity();
  int closestFace = 0;
  outDistSqr = std::numeric_limits<real>::infinity();
  for (size_t f = 0; f < faces.size(); ++f) {
    // Get face geometry
    Int3 const& face = faces[f];
    Vec4r A = ToSimd(_nodeCoordinates[face[0]]);
    Vec4r B = ToSimd(_nodeCoordinates[face[1]]);
    Vec4r C = ToSimd(_nodeCoordinates[face[2]]);

    // NOTE: Consider the option of this function returning also position/normal
    VDistanceSignParams unused;
    real currentDist2 = Get0(VDistancePointTriangleSqr(position, A, B, C, unused));
    if (currentDist2 < outDistSqr) {
      outDistSqr = currentDist2;
      closestFace = (int)f;
    }
  }
  return closestFace;
}

template <typename Bv>
bool MeshColliderBvh<Bv>::QueryPoint(
    Vec4r position,
    ContactDetectionParams const& params,
    Vec4r& outPos,
    real& outSdf,
    Vec4r& outSdfGrad) const {
  // Find closest element to point in collider.
  // Disregard both sign and parameterization at this point - We will need to recover them
  // afterwards when testing the specific scenario we're dealing with.
  real closestDist2 = std::numeric_limits<real>::infinity();
  int closestFace = params.useAccelerationStructures
      ? _bvhTree->VFindClosest(position, &closestDist2)
      : FindClosestFaceBruteForce(position, closestDist2);

  // Now that we know the closest face, get the signed distance (squared).
  // TODO: This code is inefficient, because it calls the distance query for this triangle twice.
  Int3 const& face = _mesh->GetElementConnectivity()[closestFace];
  Vec4r A = ToSimd(_nodeCoordinates[face[0]]);
  Vec4r B = ToSimd(_nodeCoordinates[face[1]]);
  Vec4r C = ToSimd(_nodeCoordinates[face[2]]);
  auto const& face2HalfEdges = _halfEdge.face2half;
  auto const& halfEdges = _halfEdge.halfEdges;
  HalfEdge const& he = halfEdges[face2HalfEdges[closestFace]];
  int edgeAB = he.edge;
  int edgeBC = halfEdges[he.next].edge;
  int edgeCA = halfEdges[halfEdges[he.next].next].edge;
  VDistanceSignParams signParams;
  signParams.normalA = ToSimd(_nodeNormals[face[0]]);
  signParams.normalB = ToSimd(_nodeNormals[face[1]]);
  signParams.normalC = ToSimd(_nodeNormals[face[2]]);
  signParams.normalAB = ToSimd(_edgeNormals[edgeAB]);
  signParams.normalBC = ToSimd(_edgeNormals[edgeBC]);
  signParams.normalCA = ToSimd(_edgeNormals[edgeCA]);
  signParams.computeSign = true;
  Vec4r par;
  [[maybe_unused]] real closestDist2_again =
      Get0(VDistancePointTriangleSqr(position, A, B, C, signParams, &par));
  MOCHI_ASSERT_VERBOSE(
      closestDist2_again == closestDist2, "This math should be deterministic to full precision");

  // Determine signed distance (negative inside the mesh). Note that the point can be outside the
  // mesh even though it is BELOW the plane of the triangle. This often happens when the closest
  // point is an edge or a node of the triangle. In those cases, VDistancePointTriangle uses the
  // pre-computed edge normal or node normal to determine whether the node is inside or outside
  // the volume which is enclosed by the mesh.
  real distance = signParams.outSign * std::sqrt(closestDist2);

  // If there is contact, add to results.
  if (distance <= params.tolerance) {
    // Get the closest point on the triangle
    Vec4r projPos = Broadcast<0>(par) * A + Broadcast<1>(par) * B + Broadcast<2>(par) * C;

    // Compute the collision normal. If the distance is larger than kEps, use (position - projPos)
    // to compute the normal and flip the sign with outSign if position is inside the mesh. If the
    // distance is too small, look up the face normal, or edge normal, or node normal instead.
    // This isn't very efficient, but hopefully most of the nodes are not within +/- kEps of the
    // surface most of the time.
    Vec4r normal;
    constexpr real kEps = 1e-5_r;
    if (std::abs(distance) >= kEps) {
      normal = signParams.outSign * Normalize<3>(position - projPos);
    } else {
      Vec4r parNonZero = (par > kEps);
      if (AllTrue<3>(parNonZero)) {
        // The closes point is somewhere on the triangle. Use the triangle normal.
        normal = ToSimd(_elementNormals[closestFace]);
      } else {
        auto bA = IsTrue<0>(parNonZero); // par[0] > kEps
        auto bB = IsTrue<1>(parNonZero); // par[1] > kEps
        auto bC = IsTrue<2>(parNonZero); // par[2] > kEps
        if (bA && bB) {
          // Closest point is on the AB edge
          normal = ToSimd(_edgeNormals[edgeAB]);
        } else if (bB && bC) {
          // Closest point is on the BC edge
          normal = ToSimd(_edgeNormals[edgeBC]);
        } else if (bC && bA) {
          // closest point is on the CA edge
          normal = ToSimd(_edgeNormals[edgeCA]);
        } else if (bA) {
          normal = ToSimd(_nodeNormals[face[0]]); // Closest point is node A
        } else if (bB) {
          normal = ToSimd(_nodeNormals[face[1]]); // Closest point is node B
        } else if (bC) {
          normal = ToSimd(_nodeNormals[face[2]]); // Closest point is node C
        } else
          MOCHI_UNLIKELY {
            MOCHI_ASSERT(false, "Bad parametric coordinates");
            normal = {};
          }
      }
    }

    // Copy result
    outSdf = distance;
    outPos = position;
    outSdfGrad = normal;
    return true;
  } else {
    return false;
  }
}

/*************************************************************************************************/

// Implementation of FindPointContactsT for a plane collider. Transform the collider to the space of
// the points. Then transform active collisions to the collider's space.
template <>
void mochi::FindPointContactsT<Plane>(
    Span<Real3 const> points,
    Plane const* collider,
    ContactDetectionParams const& params,
    TransformRT const& pointsFromCollider,
    DynamicArray<int>& outIndices,
    DynamicArray<Real3>& outContacts,
    SdfInfo& outSdf,
    bool& outIsSdfGradUnitary) {
  MOCHI_PROFILE_SCOPE();
  outIsSdfGradUnitary = true;

  // Transform collider shape to colliding space.
  auto const colliderTransformed = TransformShape(pointsFromCollider, *collider);

  // Use native SIMD size except on ARM where 2x that size has been shown to improve performance.
  // Minimum size is 4 because Simd<int, N> must also be supported.
  constexpr int N = Max(4, MOCHI_ARCH_ARM_NEON ? (2 * Simd<real>::kSize) : Simd<real>::kSize);
  static_assert(Simd<real, N>::kIsSupported && Simd<int, N>::kIsSupported);
  using V = Simd<real, N>; // Empirically selected batch size
  using V3 = NdArray<V, 3>; // 3xN
  using V3x3 = NdArray<V, 3, 3>; // 3x3xN

  auto const colliderFromPoints = Invert(pointsFromCollider);
  auto const matT = ToVMatrix4x4Transpose(colliderFromPoints);
  V3x3 const rotT = Broadcast3x3<V>(matT);
  V3 const trans = Broadcast3<V>(matT[3]);
  Vec4r planeNormal = colliderTransformed.VGetNormal(); // Broadcast3<V>(planeNormal) happens inside
                                                        // the loop to reduce register pressure.
  real const planeDist = colliderTransformed.GetDistanceFromOrigin();
  real const maxDist = planeDist + params.tolerance;
  size_t const numPoints = points.size();
  auto const sequence = Sequence<Simd<int, V::kSize>>(); // {0, 1, 2, ...}

  int constexpr kBufferFlushSize = 64; // Flush the buffer when it grows to at least this size
  alignas(V) real bufPoints[3][kBufferFlushSize + V::kSize] MOCHI_NO_INIT; // {Xs, Ys, Zs}

  size_t iSrc = 0; // Input index
  size_t iDst = 0; // Output index
  size_t iBuf = 0; // Buffer index

  auto flush = [&]() {
    auto offset = iDst - iBuf;
    for (size_t j = 0; j < iBuf; j += V::kSize) {
      V3 pt{Load<V>(&bufPoints[0][j]), Load<V>(&bufPoints[1][j]), Load<V>(&bufPoints[2][j])};
      pt = DotVecMat(pt, rotT) + trans;
      StoreTransposed(&outContacts[offset + j][0], pt);
    }
    iBuf = 0;
  };

  auto process = [&](V3 const& pt, auto hitMask) {
    V sd = Dot(Broadcast3<V>(planeNormal), pt);
    V hit = sd <= maxDist;
    if constexpr (IsSimd<decltype(hitMask)>) {
      // The final call might be a partial batch. This may ignore some of the hits.
      hit &= ReinterpretCast<V>(hitMask);
    }
    if (!AnyTrue(hit)) {
      return;
    }

    sd -= planeDist;

    // Select just the points that are in contact.
    // Write indices and distances directly to the output.
    StoreSelected(&outIndices[iDst], hit, sequence + static_cast<int>(iSrc));
    iDst += StoreSelected(&outSdf.val[iDst], hit, sd);

    // Write positions to a temporary buffer
    StoreSelected(&bufPoints[0][iBuf], hit, pt[0]);
    StoreSelected(&bufPoints[1][iBuf], hit, pt[1]);
    iBuf += StoreSelected(&bufPoints[2][iBuf], hit, pt[2]);

    // Flush the batch after it grows large enough.
    if (iBuf >= kBufferFlushSize) {
      flush();
    }
  };

  // Allocate output buffers with padding at the end to allow batch writes.
  outIndices.resize_noinit(numPoints + V::kSize);
  outContacts.resize_noinit(numPoints + V::kSize);
  outSdf.resize_noinit(numPoints + V::kSize);

  // Process points V::kSize at a time
  V3 pt;
  for (; iSrc + V::kSize <= numPoints; iSrc += V::kSize) {
    LoadTransposed(&points[iSrc][0], pt);
    process(pt, false);
  }

  // Process any trailing points
  auto numRemaining = static_cast<int>(numPoints - iSrc);
  if (numRemaining) {
    Real3 temp[V::kSize] MOCHI_NO_INIT;
    std::copy(points.begin() + iSrc, points.end(), temp);
    LoadTransposed(&temp[0][0], pt);
    using I = std::conditional_t<(sizeof(real) == 4), int, int64_t>; // Integer same size as real
    using VI = Simd<I, V::kSize>;
    auto hitMask = StaticCast<VI>(sequence) < numRemaining; // Only consider numRemaining points.
    process(pt, hitMask);
  }
  flush();

  // Resize down to correct output size
  outIndices.resize(iDst);
  outContacts.resize(iDst);
  outSdf.resize(iDst);

  // sdfGrad is the same for every point.
  Fill(MakeSpan(outSdf.grad), collider->GetNormal());
}

// Implementation of FindPointContactsT for a sphere collider. Transform the collider to the space
// of the points. Then transform active collisions to the collider's space.
template <>
void mochi::FindPointContactsT<Sphere>(
    Span<Real3 const> points,
    Sphere const* collider,
    ContactDetectionParams const& params,
    TransformRT const& pointsFromCollider,
    DynamicArray<int>& outIndices,
    DynamicArray<Real3>& outContacts,
    SdfInfo& outSdf,
    bool& outIsSdfGradUnitary) {
  MOCHI_PROFILE_SCOPE();
  outIsSdfGradUnitary = true;

  // Transform collider shape to colliding space.
  auto const colliderTransformed = TransformShape(pointsFromCollider, *collider);

  // Use native SIMD size except on ARM where 2x that size has been shown to improve performance.
  // Minimum size is 4 because Simd<int, N> must also be supported.
  constexpr int N = Max(4, MOCHI_ARCH_ARM_NEON ? (2 * Simd<real>::kSize) : Simd<real>::kSize);
  static_assert(Simd<real, N>::kIsSupported && Simd<int, N>::kIsSupported);
  using V = Simd<real, N>; // Empirically selected batch size
  using V3 = NdArray<V, 3>; // 3xN
  using V3x3 = NdArray<V, 3, 3>; // 3x3xN

  auto const colliderFromPoints = Invert(pointsFromCollider);
  auto const matT = ToVMatrix4x4Transpose(colliderFromPoints);
  V3x3 const rotT = Broadcast3x3<V>(matT);
  V3 const trans = Broadcast3<V>(matT[3]);
  Vec4r const sphereCenter = colliderTransformed.VGetCenter();
  Vec4r const sphereCenterInOutputSpace = DotVecMat4x4(sphereCenter, matT);
  real const radius = colliderTransformed.GetRadius();
  real const radiusPlusToleranceSqr = Sqr(radius + params.tolerance);
  size_t const numPoints = points.size();
  auto const sequence = Sequence<Simd<int, V::kSize>>(); // {0, 1, 2, ...}
  real constexpr kDistanceEpsilon = 10_r * std::numeric_limits<real>::epsilon();

  int constexpr kBufferFlushSize = 64; // Flush the buffer when it grows to at least this size
  alignas(V) real bufPoints[3][kBufferFlushSize + V::kSize] MOCHI_NO_INIT; // {Xs, Ys, Zs}

  size_t iSrc = 0; // Input index
  size_t iDst = 0; // Output index
  size_t iBuf = 0; // Buffer index

  auto flush = [&]() {
    auto offset = iDst - iBuf;
    for (size_t j = 0; j < iBuf; j += V::kSize) {
      V3 pt{Load<V>(&bufPoints[0][j]), Load<V>(&bufPoints[1][j]), Load<V>(&bufPoints[2][j])};
      pt = DotVecMat(pt, rotT) + trans; // Transform to output space
      StoreTransposed(&outContacts[offset + j][0], pt);

      V3 rayFromCenter = pt - Broadcast3<V>(sphereCenterInOutputSpace); // Ray in output space
      V distFromCenter = Norm(rayFromCenter);
      V sd = distFromCenter - radius;
      Store(&outSdf.val[offset + j], sd);

      V3 gsd = rayFromCenter * (1_r / distFromCenter); // Normalize
      // If pt is extremely close to the center, then gsd might not be unitary.
      // In this case, we arbitrarily choose {1, 0, 0} as the gradient.
      V nearCenter = (distFromCenter < kDistanceEpsilon);
      gsd[0] = Select(nearCenter, V{1_r}, gsd[0]);
      gsd[1] = Select(nearCenter, V{0_r}, gsd[1]);
      gsd[2] = Select(nearCenter, V{0_r}, gsd[2]);
      StoreTransposed(&outSdf.grad[offset + j][0], gsd);
    }
    iBuf = 0;
  };

  auto process = [&](V3 const& pt, auto hitMask) {
    V3 rayFromCenter = pt - Broadcast3<V>(sphereCenter);
    V distSqr = NormSqr(rayFromCenter);
    V hit = (distSqr <= radiusPlusToleranceSqr);
    if constexpr (IsSimd<decltype(hitMask)>) {
      // The final call might be a partial batch. This may ignore some of the hits.
      hit &= ReinterpretCast<V>(hitMask);
    }
    if (!AnyTrue(hit)) {
      return;
    }

    // Select just the points that are in contact. Write indices directly to the output.
    iDst += StoreSelected(&outIndices[iDst], hit, sequence + static_cast<int>(iSrc));

    // Write positions to a temporary buffer
    StoreSelected(&bufPoints[0][iBuf], hit, pt[0]);
    StoreSelected(&bufPoints[1][iBuf], hit, pt[1]);
    iBuf += StoreSelected(&bufPoints[2][iBuf], hit, pt[2]);

    // Flush the batch after it grows large enough.
    if (iBuf >= kBufferFlushSize) {
      flush();
    }
  };

  // Allocate output buffers with padding at the end to allow batch writes.
  outIndices.resize_noinit(numPoints + V::kSize);
  outContacts.resize_noinit(numPoints + V::kSize);
  outSdf.resize_noinit(numPoints + V::kSize);

  // Process points V::kSize at a time
  V3 pt;
  for (; iSrc + V::kSize <= numPoints; iSrc += V::kSize) {
    LoadTransposed(&points[iSrc][0], pt);
    process(pt, false);
  }

  // Process any trailing points
  auto numRemaining = static_cast<int>(numPoints - iSrc);
  if (numRemaining) {
    Real3 temp[V::kSize] MOCHI_NO_INIT;
    std::copy(points.begin() + iSrc, points.end(), temp);
    LoadTransposed(&temp[0][0], pt);
    using I = std::conditional_t<(sizeof(real) == 4), int, int64_t>; // Integer same size as real
    using VI = Simd<I, V::kSize>;
    auto hitMask = StaticCast<VI>(sequence) < numRemaining; // Only consider numRemaining points.
    process(pt, hitMask);
  }
  flush();

  // Resize down to correct output size
  outIndices.resize(iDst);
  outContacts.resize(iDst);
  outSdf.resize(iDst);
}

// Implementation of FindPointContactsT for an OBB collider. Transform the collider to the space
// of the points. Then transform active collisions to the collider's space.
template <>
void mochi::FindPointContactsT<Obb>(
    Span<Real3 const> points,
    Obb const* collider,
    ContactDetectionParams const& params,
    TransformRT const& pointsFromCollider,
    DynamicArray<int>& outIndices,
    DynamicArray<Real3>& outContacts,
    SdfInfo& outSdf,
    bool& outIsSdfGradUnitary) {
  MOCHI_PROFILE_SCOPE();
  outIsSdfGradUnitary = true;

  // Transform collider shape to colliding space.
  auto const colliderTransformed = TransformShape(pointsFromCollider, *collider);

  // Use native SIMD size except on ARM where 2x that size has been shown to improve performance.
  // Minimum size is 4 because Simd<int, N> must also be supported.
  constexpr int N = Max(4, MOCHI_ARCH_ARM_NEON ? (2 * Simd<real>::kSize) : Simd<real>::kSize);

  using V = Simd<real, N>; // Empirically selected batch size
  using V3 = NdArray<V, 3>; // 3xN
  using V3x3 = NdArray<V, 3, 3>; // 3x3xN

  // Compute transformation to the box local space
  auto const pointsFromBox = colliderTransformed.GetTransform();
  auto const pointsFromBoxMatT = ToVMatrix4x4Transpose(pointsFromBox);
  V3x3 const boxFromPointsRot = Broadcast3x3<V>(pointsFromBoxMatT); // 3x3 transpose is the inverse
  V3 const boxFromPointsTrans = Broadcast3<V>(-pointsFromBoxMatT[3]); // Inverse translation

  // Compute transformation from box local space to the output space.
  auto const colliderFromPoints = Invert(pointsFromCollider);
  auto const colliderFromBoxMat =
      Dot4x4(ToVMatrix4x4(colliderFromPoints), Transpose4x4(pointsFromBoxMatT));
  V3x3 const colliderFromBoxRot = Broadcast3x3<V>(colliderFromBoxMat);
  V3 const colliderFromBoxTrans{
      V{colliderFromBoxMat[0][3]},
      V{colliderFromBoxMat[1][3]},
      V{colliderFromBoxMat[2][3]}}; // 4th col

  // Tolerance used to avoid divide-by-near-zero when normalizing the gradient vector.
  auto constexpr kDistanceEpsilon = 10_r * std::numeric_limits<real>::epsilon();

  // Buffer for temporary storage of points.
  int constexpr kBufferFlushSize = 64; // Flush the buffer when it grows to at least this size
  alignas(V) real bufPoints[3][kBufferFlushSize + V::kSize] MOCHI_NO_INIT; // {Xs, Ys, Zs}

  // {0, 1, 2, 3, ...}
  auto const sequence = Sequence<Simd<int, N>>();

  size_t iSrc = 0; // Input index
  size_t iDst = 0; // Output index
  size_t iBuf = 0; // Buffer index

  auto flush = [&]() {
    auto offset = iDst - iBuf;
    for (size_t j = 0; j < iBuf; j += V::kSize) {
      V3 localPos{Load<V>(&bufPoints[0][j]), Load<V>(&bufPoints[1][j]), Load<V>(&bufPoints[2][j])};

      // First, assume localPos is outside the box. Clamp it to the surface.
      V3 vHalfExt = Broadcast3<V>(colliderTransformed.VGetHalfExtents());
      V3 clampPos{
          Clamp(localPos[0], -vHalfExt[0], vHalfExt[0]),
          Clamp(localPos[1], -vHalfExt[1], vHalfExt[1]),
          Clamp(localPos[2], -vHalfExt[2], vHalfExt[2])};

      // Then, find the closest face, assuming localPos is inside the box.
      V3 faceDist = vHalfExt - Abs(localPos);
      V minFaceDist = Min(faceDist[0], faceDist[1], faceDist[2]);
      V3 isClosestFace MOCHI_NO_INIT;
      isClosestFace[0] = VEqual(minFaceDist, faceDist[0]);
      isClosestFace[1] = VEqual(minFaceDist, faceDist[1]) & ~isClosestFace[0];
      isClosestFace[2] = VEqual(minFaceDist, faceDist[2]) & ~isClosestFace[0] & ~isClosestFace[1];
      // Select the unit normal of the closest face
      V3 projNormal{
          Select(isClosestFace[0], Sign(localPos[0]), V::Zero()),
          Select(isClosestFace[1], Sign(localPos[1]), V::Zero()),
          Select(isClosestFace[2], Sign(localPos[2]), V::Zero())};
      // Project position onto the closest face.
      V3 projPos{
          Select(isClosestFace[0], projNormal[0] * vHalfExt[0], localPos[0]),
          Select(isClosestFace[1], projNormal[1] * vHalfExt[1], localPos[1]),
          Select(isClosestFace[2], projNormal[2] * vHalfExt[2], localPos[2])};

      // Select the surface position from one of the above calculations.
      V isInside = (minFaceDist >= 0_r);
      V3 surfacePos{
          Select(isInside, projPos[0], clampPos[0]),
          Select(isInside, projPos[1], clampPos[1]),
          Select(isInside, projPos[2], clampPos[2])};

      // Compute signed distance
      V3 surfaceDelta = localPos - surfacePos;
      V sd = Norm(surfaceDelta);
      sd = Select(isInside, -sd, sd);
      Store(&outSdf.val[offset + j], sd);

      // Use the projected normal if the point is on or very close to the surface of the box.
      // Otherwise, use surfaceDelta to compute the normal.
      V sdInv = 1_r / sd;
      V3 normal{
          Select(sd < kDistanceEpsilon, projNormal[0], surfaceDelta[0] * sdInv),
          Select(sd < kDistanceEpsilon, projNormal[1], surfaceDelta[1] * sdInv),
          Select(sd < kDistanceEpsilon, projNormal[2], surfaceDelta[2] * sdInv)};

      // Apply the output rotation to the normal
      normal = DotMatVec(colliderFromBoxRot, normal);
      StoreTransposed(&outSdf.grad[offset + j][0], normal);

      // Apply the output transform to the position
      V3 outPos = DotMatVec(colliderFromBoxRot, localPos) + colliderFromBoxTrans;
      StoreTransposed(&outContacts[offset + j][0], outPos);
    }
    iBuf = 0;
  };

  Vec4r halfExtPlusTol = colliderTransformed.VGetHalfExtents() + params.tolerance;
  auto process = [&](V3 const& pt) {
    // Transform to the local space of the box.
    V3 localPos = DotMatVec(boxFromPointsRot, pt + boxFromPointsTrans);
    // In local-space, the box is an AABB with center at the origin. Points will be included in the
    // output if they are within that AABB, after padding by the contact tolerance. Points at the
    // extreme corners may return a signed distance greater than tolerance, but that is considered
    // to be OK.
    V hit = //
        (Abs(localPos[0]) <= Get<0>(halfExtPlusTol)) & //
        (Abs(localPos[1]) <= Get<1>(halfExtPlusTol)) & //
        (Abs(localPos[2]) <= Get<2>(halfExtPlusTol));
    if (!AnyTrue(hit)) {
      return;
    }

    // Select just the points with contact. Write the corresponding indices to the output.
    iDst += StoreSelected(&outIndices[iDst], hit, sequence + static_cast<int>(iSrc));

    // Write local-space positions to a temporary buffer
    StoreSelected(&bufPoints[0][iBuf], hit, localPos[0]);
    StoreSelected(&bufPoints[1][iBuf], hit, localPos[1]);
    iBuf += StoreSelected(&bufPoints[2][iBuf], hit, localPos[2]);

    // Flush the buffer after it grows large enough.
    if (iBuf >= kBufferFlushSize) {
      flush();
    }
  };

  // Allocate output buffers with padding at the end to allow batch writes.
  size_t const numPoints = points.size();
  outIndices.resize_noinit(numPoints + V::kSize);
  outContacts.resize_noinit(numPoints + V::kSize);
  outSdf.resize_noinit(numPoints + V::kSize);

  // Process points V::kSize at a time
  V3 pt;
  for (; iSrc + V::kSize <= numPoints; iSrc += V::kSize) {
    LoadTransposed(&points[iSrc][0], pt);
    process(pt);
  }

  // Process any trailing points
  auto numRemaining = static_cast<int>(numPoints - iSrc);
  if (numRemaining) {
    // Load remaining points into one last SIMD batch
    Real3 temp[V::kSize] MOCHI_NO_INIT;
    std::copy(points.begin() + iSrc, points.end(), temp);
    LoadTransposed(&temp[0][0], pt);
    // Fill unused points with QNaN so they fail the AABB test.
    using I = std::conditional_t<(sizeof(real) == 4), int, int64_t>;
    using VI = Simd<I, V::kSize>;
    auto mask = StaticCast<VI>(sequence) < static_cast<I>(numRemaining);
    pt[0] = Select(mask, pt[0], V{std::numeric_limits<real>::quiet_NaN()});
    pt[1] = Select(mask, pt[1], V{std::numeric_limits<real>::quiet_NaN()});
    pt[2] = Select(mask, pt[2], V{std::numeric_limits<real>::quiet_NaN()});
    process(pt);
  }
  flush();

  // Resize down to correct output size
  outIndices.resize(iDst);
  outContacts.resize(iDst);
  outSdf.resize(iDst);
}

// Implementation of FindPointContactsT for a mesh collider. Transform the points to the collider
// space. Then output active collisions in the collider's space.
template <>
void mochi::FindPointContactsT<MeshCollider>(
    Span<Real3 const> points,
    MeshCollider const* collider,
    ContactDetectionParams const& params,
    TransformRT const& pointsFromCollider,
    DynamicArray<int>& outIndices,
    DynamicArray<Real3>& outContacts,
    SdfInfo& outSdf,
    bool& outIsSdfGradUnitary) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(collider, "MeshCollider is null");
  MOCHI_ASSERT_VERBOSE(collider->IsInitialized(), "MeshCollider is not initialized");
  MOCHI_ASSERT_VERBOSE(outIndices.empty(), "Expected empty contact detection result.");

  auto const colliderFromPoints = Invert(pointsFromCollider);

  // Query MeshCollider for each point
  outIsSdfGradUnitary = true;
  for (int i = 0; i < isize(points); ++i) {
    Vec4r posColliding = {};
    real sdf = {};
    Vec4r sdfGrad = {};
    Vec4r pointInCollider = colliderFromPoints.TransformPoint(ToSimd(points[i], 1_r));
    if (collider->QueryPoint(pointInCollider, params, posColliding, sdf, sdfGrad)) {
      outIndices.push_back(i);
      outContacts.push_back(ToReal3(posColliding));
      outSdf.push_back(sdf, ToReal3(sdfGrad));
    }
  }
}

// Implementation of FindPointContactsT for a GridSdf collider. Transform the points to the collider
// space. Then output active collisions in the collider's space.
template <>
void mochi::FindPointContactsT<GridSdf>(
    Span<Real3 const> points,
    GridSdf const* collider,
    ContactDetectionParams const& params,
    TransformRT const& pointsFromCollider,
    DynamicArray<int>& outIndices,
    DynamicArray<Real3>& outContacts,
    SdfInfo& outSdf,
    bool& outIsSdfGradUnitary) {
  collider->FindPointContacts(
      points, pointsFromCollider, params, outIndices, outContacts, outSdf, outIsSdfGradUnitary);
}

void mochi::FindPointContactsMapped(
    Span<Real3 const> points,
    MappedSdfCollider const* collider,
    ContactDetectionParams const& params,
    TransformRT const& pointsFromCollider,
    AnyShape const& bounds,
    DynamicArray<int>& outIndices,
    DynamicArray<Real3>& outContacts,
    SdfInfo* outSdf,
    bool& outIsSdfGradUnitary,
    int* outNDofs,
    DynamicArray<VMatrix3x3r>* outMapJac,
    DynamicArray<ColliderJacDofs>* outDofsJac) {
  // Cull points based on collider bounds. The culled points are in collider space.
  auto const colliderFromPoints = Invert(pointsFromCollider);
  DynamicArray<Real3> pointsCulled(points.size());
  DynamicArray<int> indsCulled(points.size());
  auto numPointsCulled = static_cast<size_t>(
      FindPointsInAnyShape(bounds, points, colliderFromPoints, pointsCulled, indsCulled));

  // Map the points. If SDF data is required, use temporary storage for the result.
  DynamicArray<Real3> pointsMapped;
  DynamicArray<int> indsMapped;
  collider->mapping->MapPoints(
      Span{pointsCulled.data(), numPointsCulled},
      Span{indsCulled.data(), numPointsCulled},
      nullptr,
      outSdf ? pointsMapped : outContacts,
      outSdf ? indsMapped : outIndices,
      outMapJac,
      outDofsJac);

  // Run the query if SDF data is required
  if (outSdf) {
    MOCHI_ASSERT(collider->shape);
    auto const* gridSdf = dynamic_cast<GridSdf const*>(collider->shape);
    MOCHI_ASSERT(gridSdf, "Only grid SDF is supported.");
    // Pass an identity transform because the points are already in the collider space.
    // TODO: Try to leverage identity transforms in GridSdf.
    FindPointContactsT(
        pointsMapped,
        gridSdf,
        params,
        TransformRT{},
        outIndices,
        outContacts,
        *outSdf,
        outIsSdfGradUnitary);

    // Reindex the result
    BaseMap::ReindexResult(indsMapped, outIndices, outMapJac, outDofsJac);
  }

  // The SDF might not be unitary after the mapping
  outIsSdfGradUnitary = false;

  // Define the number of DoFs of the mapping if needed
  if (outNDofs) {
    *outNDofs = collider->mapping->GetNumDofs();
  }
}

void mochi::ContactJac::CompressIndices() {
  if (nContacts == 0) {
    // Nothing to do
    groupsInitialized = true;
    return;
  }

  int numRows = hasSharedDoFs ? 1 : nContacts;
  // If the DoFs are not shared, do not attempt to create DoF groups larger than 3. In a soft
  // object, only groups of 3 indices share the same sparsity pattern.
  bool forceTriplets = !hasSharedDoFs;

  // Prepare index groups. Create a temporary data structure that stores index groups for all
  // contacts.
  MOCHI_FILO_STACK_ALLOCATOR(tempAlloc, 16 * 1024); // Probably more than enough
  DynamicArray<IndexGroups> indGroupsAll(&tempAlloc);
  indGroupsAll.reserve(numRows);
  for (int i = 0; i < numRows; i++) {
    indGroupsAll.emplace_back(CreateIndexGroups(Inds(i), forceTriplets, &tempAlloc));
  }

  // Create final storage. Use the maximum group count across all contacts, since different
  // contacts may have different numbers of index groups (e.g., rod visual mesh contact where
  // skinning connectivity varies per contact). Per-contact group counts are stored so that
  // IndGroups(i) returns only the valid entries for each contact.
  int maxGroups = 0;
  for (int i = 0; i < numRows; i++) {
    maxGroups = Max(maxGroups, isize(indGroupsAll[i]));
  }
  _data.indGroups.clear();
  _data.indGroups.resize_noinit(numRows * maxGroups);
  _data.indGroupCounts.resize_noinit(numRows);
  _indGroups.Reset(_data.indGroups.data(), numRows, maxGroups);
  for (int i = 0; i < numRows; i++) {
    _data.indGroupCounts[i] = isize(indGroupsAll[i]);
    std::copy(indGroupsAll[i].begin(), indGroupsAll[i].end(), _indGroups.Row(i).Data());
  }

  // The index groups are properly initialized
  groupsInitialized = true;
}
