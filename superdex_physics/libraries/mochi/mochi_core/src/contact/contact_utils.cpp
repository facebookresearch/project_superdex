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
#include <optional>

using namespace mochi;

// Number of contact points whose collision response is computed together.
constexpr int kCollResponseBatchSize = Min(2 * Simd<real>::kSize, kCollResponseMaxBatchSize);

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
  for (int s = pointRange.Min(); s <= pointRange.Max(); s += kCollResponseBatchSize) {
    int const numPoints = Min(pointRange.Max() - s + 1, kCollResponseBatchSize);

    // Discard unreachable batch sizes, so the batch kernels get no dead call sites.
#define MOCHI_BATCH_CASE(N)                                                                        \
  case N:                                                                                          \
    if constexpr ((N) <= kCollResponseBatchSize) {                                                 \
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
          contactQuery.anchorOffset.empty()                                                        \
              ? Span<Real3 const>{}                                                                \
              : MakeConstSpan(contactQuery.anchorOffset).subspan(s, N),                            \
          params,                                                                                  \
          config,                                                                                  \
          dtStage,                                                                                 \
          assemEnergy,                                                                             \
          assemForce,                                                                              \
          assemDForce,                                                                             \
          contactQuery.isSdfGradUnitary,                                                           \
          contactQuery.isNormalCollidingUnitary);                                                  \
    }                                                                                              \
    break;

    // Compute contact response
    // clang-format off
    switch (numPoints) {
      MOCHI_BATCH_CASE(1) MOCHI_BATCH_CASE(2) MOCHI_BATCH_CASE(3) MOCHI_BATCH_CASE(4)
      MOCHI_BATCH_CASE(5) MOCHI_BATCH_CASE(6) MOCHI_BATCH_CASE(7) MOCHI_BATCH_CASE(8)
      default:
        static_assert(kCollResponseMaxBatchSize == 8, "Please update this switch statement");
        MOCHI_ASSERT(false, "Unsupported batch size (%d).", numPoints);
    }
    // clang-format on
  }

#undef MOCHI_BATCH_CASE

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

namespace {

// HSumEach reduces several lane vectors together, in fewer instructions than separate HSum calls.
// It sums all lanes, so partial batches, with padding lanes, use HSum.
template <int kBatchSize>
constexpr bool kUseHSumEach =
    kBatchSize == BatchReal<kBatchSize>::kSize && kHasHSumEach<BatchReal<kBatchSize>>;

// Sum over the lanes of a batch of 3-vectors or 3x3 matrices.
template <int kBatchSize>
MOCHI_FORCE_INLINE Vec4r SumLanes3(BatchReal3<kBatchSize> const& v) {
  if constexpr (kUseHSumEach<kBatchSize>) {
    return HSumEach(v[0], v[1], v[2], BatchReal<kBatchSize>{});
  } else {
    return {HSum<kBatchSize>(v[0]), HSum<kBatchSize>(v[1]), HSum<kBatchSize>(v[2]), 0_r};
  }
}

template <int kBatchSize>
MOCHI_FORCE_INLINE VMatrix3x3r SumLanes3x3(BatchReal3x3<kBatchSize> const& m) {
  return {SumLanes3<kBatchSize>(m[0]), SumLanes3<kBatchSize>(m[1]), SumLanes3<kBatchSize>(m[2])};
}

// Sum over the lanes of a batch of symmetric 3x3 matrices, reading only the upper triangle.
template <int kBatchSize>
MOCHI_FORCE_INLINE VMatrix3x3r SumLanesSym3x3(BatchReal3x3<kBatchSize> const& m) {
  if constexpr (kUseHSumEach<kBatchSize>) {
    // Reducing the off-diagonal entries twice avoids the shuffles to build the rows from 6 sums.
    return {
        SumLanes3<kBatchSize>({m[0][0], m[0][1], m[0][2]}),
        SumLanes3<kBatchSize>({m[0][1], m[1][1], m[1][2]}),
        SumLanes3<kBatchSize>({m[0][2], m[1][2], m[2][2]})};
  } else {
    real const m00 = HSum<kBatchSize>(m[0][0]);
    real const m01 = HSum<kBatchSize>(m[0][1]);
    real const m02 = HSum<kBatchSize>(m[0][2]);
    real const m11 = HSum<kBatchSize>(m[1][1]);
    real const m12 = HSum<kBatchSize>(m[1][2]);
    real const m22 = HSum<kBatchSize>(m[2][2]);
    return {Vec4r{m00, m01, m02, 0_r}, Vec4r{m01, m11, m12, 0_r}, Vec4r{m02, m12, m22, 0_r}};
  }
}

// S(r) m. The columns of S(r) m are r x (the columns of m).
template <int kBatchSize>
MOCHI_FORCE_INLINE BatchReal3x3<kBatchSize> SkewTimes(
    BatchReal3<kBatchSize> const& r,
    BatchReal3x3<kBatchSize> const& m) {
  return {r[1] * m[2] - r[2] * m[1], r[2] * m[0] - r[0] * m[2], r[0] * m[1] - r[1] * m[0]};
}

// m S(r). The rows of m S(r) are (the rows of m) x r.
template <int kBatchSize>
MOCHI_FORCE_INLINE BatchReal3x3<kBatchSize> TimesSkew(
    BatchReal3x3<kBatchSize> const& m,
    BatchReal3<kBatchSize> const& r) {
  return {Cross(m[0], r), Cross(m[1], r), Cross(m[2], r)};
}

// Compute the collision response of the batch of kBatchSize contact points starting at 'first',
// store its forces in outForce if not empty, and gather the weights. If computeLeverArms, also
// compute the lever arms from 'origin' to the contact points at the time step of kGradTarget.
template <int kBatchSize, GradTarget kGradTarget>
MOCHI_FORCE_INLINE void ComputeBatchSumsPrologue(
    int first,
    ContactDetectionResult const& contactQuery,
    ContactParams const& params,
    ContactEvalConfig const& config,
    real dtStage,
    Span<real const> weights,
    Vec4r const& origin,
    bool assemEnergy,
    bool assemForce,
    bool assemDForce,
    bool computeLeverArms,
    Span<Real3> outForce,
    BatchCollisionResponse<kBatchSize, kGradTarget>& outResponse,
    BatchReal<kBatchSize>& outW,
    BatchReal3<kBatchSize>& outLeverArms) {
  using V = BatchReal<kBatchSize>;
  ComputeBatchCollisionResponse<kBatchSize, kGradTarget>(
      outResponse,
      MakeConstSpan(contactQuery.sdfInfo.val).subspan(first, kBatchSize),
      MakeConstSpan(contactQuery.sdfInfo.grad).subspan(first, kBatchSize),
      config.explicitNormals
          ? MakeConstSpan(contactQuery.sdfInfoStageStart.val).subspan(first, kBatchSize)
          : Span<real const>{},
      config.explicitNormals
          ? MakeConstSpan(contactQuery.sdfInfoStageStart.grad).subspan(first, kBatchSize)
          : Span<Real3 const>{},
      config.validCollidingNormals
          ? MakeConstSpan(contactQuery.normalColliding).subspan(first, kBatchSize)
          : Span<Real3 const>{},
      MakeConstSpan(contactQuery.posColliding).subspan(first, kBatchSize),
      MakeConstSpan(contactQuery.posCollidingStageStart).subspan(first, kBatchSize),
      contactQuery.anchorOffset.empty()
          ? Span<Real3 const>{}
          : MakeConstSpan(contactQuery.anchorOffset).subspan(first, kBatchSize),
      params,
      config,
      dtStage,
      assemEnergy,
      assemForce,
      assemDForce,
      contactQuery.isSdfGradUnitary,
      contactQuery.isNormalCollidingUnitary);

  if (!outForce.empty()) {
    StoreTransposed<kBatchSize>(&outForce[first][0], outResponse.force);
  }

  real sampleWeights[kBatchSize] MOCHI_NO_INIT;
  for (int i = 0; i < kBatchSize; ++i) {
    sampleWeights[i] = weights[contactQuery.sampleIndices[first + i]];
  }
  outW = Load<kBatchSize, V>(sampleWeights);

  if (computeLeverArms) {
    auto const& positions = (kGradTarget == GradTarget::Current)
        ? contactQuery.posColliding
        : contactQuery.posCollidingStageStart;
    LoadTransposed<kBatchSize>(&positions[first][0], outLeverArms);
    outLeverArms -= Broadcast3<V>(origin);
  }
}

// Partial RigidContactSums of batches of kBatchSize contact points, one per SIMD lane: lane i sums
// the ith point of every batch. The lane sums are carried across all the batches of a range and
// reduced over the lanes once at the end. This is cheap because a rigid slice has a single pair, so
// the lane sums have a fixed size (~2 KB for 8 double lanes) that stays in L1, and their zeroing
// and final reduction are amortized over all the batches. See AddBatchSkinnedContactSums for the
// skinned case.
//
// TODO: Explore selecting the force derivatives at compile time (a kDForce template parameter), so
// that the residual path carries no derivative sums, and storing the symmetric dForce and
// skewDForceSkew as BatchSymMatrix3x3. See D122431619.
template <int kBatchSize>
struct RigidContactLaneSums {
  BatchDouble<kBatchSize> energy = {};
  BatchReal3<kBatchSize> force = {};
  BatchReal3<kBatchSize> torque = {};
  BatchReal3x3<kBatchSize> dForce;
  BatchReal3x3<kBatchSize> skewDForce;
  BatchReal3x3<kBatchSize> skewDForceSkew;

  // Zero the derivatives only if requested. Declare instances MOCHI_NO_INIT, otherwise the build
  // may zero them anyway. Zeroing the energy, force and torque unconditionally is cheaper than
  // branching on their flags.
  explicit RigidContactLaneSums(bool assemDForce) {
    if (assemDForce) {
      dForce = {};
      skewDForce = {};
      skewDForceSkew = {};
    }
  }

  // Add the requested quantities of the collision response of a batch of contact points, with
  // weights w and lever arms s.
  template <GradTarget kGradTarget>
  void Add(
      BatchCollisionResponse<kBatchSize, kGradTarget> const& response,
      BatchReal<kBatchSize> const& w,
      BatchReal3<kBatchSize> const& s,
      bool assemEnergy,
      bool assemForce,
      bool assemDForce) {
    if (assemEnergy) {
      energy += StaticCast<BatchDouble<kBatchSize>>(w) * response.energy;
    }
    if (assemForce) {
      BatchReal3<kBatchSize> const wf = w * response.force;
      force += wf;
      torque += Cross(s, wf);
    }
    MOCHI_ASSERT_VERBOSE(
        kGradTarget == GradTarget::Current || !assemDForce,
        "The force derivative sums only support GradTarget::Current.");
    if constexpr (kGradTarget == GradTarget::Current) {
      if (assemDForce) {
        BatchReal3x3<kBatchSize> const wd = w * SymToFull3x3(response.dForce);
        BatchReal3x3<kBatchSize> const sd = SkewTimes<kBatchSize>(s, wd);
        dForce += wd;
        skewDForce += sd;
        skewDForceSkew += TimesSkew<kBatchSize>(sd, s);
      }
    }
  }

  // Add the sums over the lanes of the requested quantities to outSums.
  void AddTo(RigidContactSums& outSums, bool assemEnergy, bool assemForce, bool assemDForce) const {
    if (assemEnergy) {
      outSums.energy += HSum<kBatchSize>(energy);
    }
    if (assemForce) {
      outSums.force += SumLanes3<kBatchSize>(force);
      outSums.torque += SumLanes3<kBatchSize>(torque);
    }
    if (assemDForce) {
      outSums.dForce += SumLanes3x3<kBatchSize>(dForce);
      outSums.skewDForce += SumLanes3x3<kBatchSize>(skewDForce);
      outSums.skewDForceSkew += SumLanes3x3<kBatchSize>(skewDForceSkew);
    }
  }
};

} // namespace

// Add the collision response of the batch of kBatchSize contact points starting at 'first' to
// outLaneSums, and store its forces in outForce if not empty. See ComputeRigidContactSums.
template <int kBatchSize, GradTarget kGradTarget>
static void AddBatchRigidContactSums(
    int first,
    ContactDetectionResult const& contactQuery,
    ContactParams const& params,
    ContactEvalConfig const& config,
    real dtStage,
    Span<real const> weights,
    Vec4r const& origin,
    bool assemEnergy,
    bool assemForce,
    bool assemDForce,
    Span<Real3> outForce,
    RigidContactLaneSums<kBatchSize>& outLaneSums) {
  BatchCollisionResponse<kBatchSize, kGradTarget> response MOCHI_NO_INIT;
  BatchReal<kBatchSize> w MOCHI_NO_INIT;
  BatchReal3<kBatchSize> leverArms MOCHI_NO_INIT;
  ComputeBatchSumsPrologue<kBatchSize, kGradTarget>(
      first,
      contactQuery,
      params,
      config,
      dtStage,
      weights,
      origin,
      assemEnergy,
      assemForce,
      assemDForce,
      /*computeLeverArms*/ assemForce || assemDForce,
      outForce,
      response,
      w,
      leverArms);

  outLaneSums.Add(response, w, leverArms, assemEnergy, assemForce, assemDForce);
}

template <GradTarget kGradTarget>
RigidContactSums mochi::ComputeRigidContactSums(
    Interval<int> pointRange,
    ContactDetectionResult const& contactQuery,
    ContactParams const& params,
    ContactEvalConfig const& config,
    real dtStage,
    Span<real const> weights,
    Vec4r origin,
    bool assemEnergy,
    bool assemForce,
    bool assemDForce,
    Span<Real3> outForce) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(pointRange.Valid());
  MOCHI_ASSERT_VERBOSE(
      pointRange.Min() >= 0 && pointRange.Max() < isize(contactQuery.sampleIndices));
  MOCHI_ASSERT_VERBOSE(
      !config.explicitNormals ||
          contactQuery.sdfInfo.size() == contactQuery.sdfInfoStageStart.size(),
      "Explicit normals require stage-start SDF info.");
  MOCHI_ASSERT_VERBOSE(
      !config.validCollidingNormals ||
          contactQuery.sdfInfo.size() == contactQuery.normalColliding.size(),
      "Valid colliding normals require normalColliding.");
  MOCHI_ASSERT_VERBOSE(
      outForce.empty() || (assemForce && isize(outForce) == isize(contactQuery.sampleIndices)),
      "outForce requires assemForce and must be indexed like the contact points.");
  MOCHI_ASSERT_VERBOSE(
      contactQuery.colliderIntegrationWeights.empty(),
      "Rigid contact sums do not support collider integration weights.");
  static_assert(kGradTarget == GradTarget::Current || kGradTarget == GradTarget::Previous);
  MOCHI_ASSERT_VERBOSE(
      kGradTarget == GradTarget::Current || !assemDForce,
      "The force derivative sums only support GradTarget::Current.");

  // Full batches accumulate into the same lane sums, which are reduced once. The remaining points
  // form a narrower batch rather than a masked full batch: a full batch would read past pointRange,
  // and a narrower batch may use a narrower SIMD type, which is cheaper to compute and reduce. For
  // the same reason, ranges shorter than a full batch skip the full-batch lane sums.
  RigidContactSums sums;
  int s = pointRange.Min();
  if (pointRange.Size() >= kCollResponseBatchSize) {
    RigidContactLaneSums<kCollResponseBatchSize> laneSums MOCHI_NO_INIT(assemDForce);
    for (; s <= pointRange.Max() - kCollResponseBatchSize + 1; s += kCollResponseBatchSize) {
      AddBatchRigidContactSums<kCollResponseBatchSize, kGradTarget>(
          s,
          contactQuery,
          params,
          config,
          dtStage,
          weights,
          origin,
          assemEnergy,
          assemForce,
          assemDForce,
          outForce,
          laneSums);
    }
    laneSums.AddTo(sums, assemEnergy, assemForce, assemDForce);
  }

  // The remaining points, fewer than a full batch, are summed over their own lanes.
#define MOCHI_BATCH_CASE(N)                                             \
  case N:                                                               \
    if constexpr ((N) < kCollResponseBatchSize) {                       \
      RigidContactLaneSums<N> lastBatchSums MOCHI_NO_INIT(assemDForce); \
      AddBatchRigidContactSums<N, kGradTarget>(                         \
          s,                                                            \
          contactQuery,                                                 \
          params,                                                       \
          config,                                                       \
          dtStage,                                                      \
          weights,                                                      \
          origin,                                                       \
          assemEnergy,                                                  \
          assemForce,                                                   \
          assemDForce,                                                  \
          outForce,                                                     \
          lastBatchSums);                                               \
      lastBatchSums.AddTo(sums, assemEnergy, assemForce, assemDForce);  \
    }                                                                   \
    break;

  int const numPoints = pointRange.Max() - s + 1;
  // clang-format off
  switch (numPoints) {
    case 0:
      break;
    MOCHI_BATCH_CASE(1) MOCHI_BATCH_CASE(2) MOCHI_BATCH_CASE(3) MOCHI_BATCH_CASE(4)
    MOCHI_BATCH_CASE(5) MOCHI_BATCH_CASE(6) MOCHI_BATCH_CASE(7)
    default:
      static_assert(kCollResponseMaxBatchSize == 8, "Please update this switch statement");
      MOCHI_ASSERT_VERBOSE(false, "Unsupported batch size (%d).", numPoints);
  }

#undef MOCHI_BATCH_CASE
  // clang-format on
  return sums;
}

#define MOCHI_SPECIALIZE_COMPUTE_RIGID_CONTACT_SUMS(kGradTarget)         \
  template RigidContactSums mochi::ComputeRigidContactSums<kGradTarget>( \
      Interval<int>,                                                     \
      ContactDetectionResult const&,                                     \
      ContactParams const&,                                              \
      ContactEvalConfig const&,                                          \
      real,                                                              \
      Span<real const>,                                                  \
      Vec4r,                                                             \
      bool,                                                              \
      bool,                                                              \
      bool,                                                              \
      Span<Real3>);
MOCHI_SPECIALIZE_COMPUTE_RIGID_CONTACT_SUMS(GradTarget::Current);
MOCHI_SPECIALIZE_COMPUTE_RIGID_CONTACT_SUMS(GradTarget::Previous);
#undef MOCHI_SPECIALIZE_COMPUTE_RIGID_CONTACT_SUMS

namespace {

// Per-slice quantities of a batch of contact points, reused across batches.
template <int kBatchSize>
struct SkinnedContactBatchSlices {
  DynamicArray<BatchReal<kBatchSize>> scales;
  DynamicArray<BatchReal3<kBatchSize>> leverArms;
  DynamicArray<BatchReal3x3<kBatchSize>> skewWD; // S(r) w D

  SkinnedContactBatchSlices(int numSlices, bool assemDForce, Allocator* allocator)
      : scales(allocator), leverArms(allocator), skewWD(allocator) {
    scales.resize_noinit(numSlices);
    leverArms.resize_noinit(numSlices);
    if (assemDForce) {
      skewWD.resize_noinit(numSlices);
    }
  }
};

// Add the sums of the batch of kBatchSize contact points starting at 'first' to outSums, reducing
// them over the lanes of the batch right away. Unlike RigidContactLaneSums, lane sums are not
// carried across batches: they would need four 3x3 matrices per slice pair, i.e. O(numSlices^2)
// storage
// (~2.3 KB per pair for 8 double lanes), which exceeds L1 for a handful of slices. Every batch
// would read and write all of it, and zeroing and reducing it would dominate each call, since
// skinned contact pairs typically have few points. Reducing every batch keeps the state in the
// scalar sums, O(numSlices^2) but kBatchSize times smaller, at the cost of a lane reduction per
// batch. See ComputeSkinnedContactSums.
template <int kBatchSize>
void AddBatchSkinnedContactSums(
    int first,
    ContactDetectionResult const& contactQuery,
    ContactParams const& params,
    ContactEvalConfig const& config,
    real dtStage,
    Span<real const> weights,
    SkinnedContactSlices const& slices,
    bool assemEnergy,
    bool assemForce,
    bool assemDForce,
    Span<Real3> outForce,
    SkinnedContactBatchSlices<kBatchSize>& batchSlices,
    SkinnedContactSums& outSums) {
  using V = BatchReal<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  using M3 = BatchReal3x3<kBatchSize>;
  GradTarget constexpr kGradTarget = GradTarget::Current;
  int const numBones = isize(slices.boneRotations);
  int const numSlices = slices.NumSlices();
  bool const hasRigidColliderSlice = slices.rigidColliderOrigin.has_value();

  BatchCollisionResponse<kBatchSize, kGradTarget> response MOCHI_NO_INIT;
  V w MOCHI_NO_INIT;
  V3 rigidColliderLeverArms MOCHI_NO_INIT;
  ComputeBatchSumsPrologue<kBatchSize, kGradTarget>(
      first,
      contactQuery,
      params,
      config,
      dtStage,
      weights,
      slices.rigidColliderOrigin.value_or(Vec4r{}),
      assemEnergy,
      assemForce,
      assemDForce,
      /*computeLeverArms*/ hasRigidColliderSlice && (assemForce || assemDForce),
      outForce,
      response,
      w,
      rigidColliderLeverArms);

  if (assemEnergy) {
    outSums.energy += HSum<kBatchSize>(StaticCast<BatchDouble<kBatchSize>>(w) * response.energy);
  }
  if (!assemForce && !assemDForce) {
    return;
  }

  // Gather (rho, s) of each bone and compute r = M rho. The rigid-collider slice has s = 1.
  auto& s = batchSlices.scales;
  auto& r = batchSlices.leverArms;
  int pointOffsets[kBatchSize] MOCHI_NO_INIT;
  for (int i = 0; i < kBatchSize; ++i) {
    int const sample = contactQuery.sampleIndices[first + i];
    pointOffsets[i] = slices.pointOffsets[sample];
    MOCHI_ASSERT_VERBOSE(
        slices.pointOffsets[sample + 1] - pointOffsets[i] == numBones,
        "Every sample must have one point per bone.");
  }
  for (int j = 0; j < numBones; ++j) {
    real gathered[4][kBatchSize] MOCHI_NO_INIT;
    for (int i = 0; i < kBatchSize; ++i) {
      Vec4r const& point = slices.points[pointOffsets[i] + j];
      gathered[0][i] = point[0];
      gathered[1][i] = point[1];
      gathered[2][i] = point[2];
      gathered[3][i] = point[3];
    }
    V3 const rho{
        Load<kBatchSize, V>(gathered[0]),
        Load<kBatchSize, V>(gathered[1]),
        Load<kBatchSize, V>(gathered[2])};
    s[j] = Load<kBatchSize, V>(gathered[3]);
    VMatrix3x3r const& m = slices.boneRotations[j];
    for (int c = 0; c < 3; ++c) {
      r[j][c] = m[c][0] * rho[0] + m[c][1] * rho[1] + m[c][2] * rho[2];
    }
  }
  if (hasRigidColliderSlice) {
    s[numBones] = V{1_r};
    r[numBones] = rigidColliderLeverArms;
  }

  if (assemForce) {
    V3 const wf = w * response.force;
    for (int a = 0; a < numSlices; ++a) {
      outSums.force[a] += SumLanes3<kBatchSize>(s[a] * wf);
      outSums.torque[a] += SumLanes3<kBatchSize>(Cross(r[a], wf));
    }
  }

  if (assemDForce) {
    auto& h = batchSlices.skewWD;
    M3 const wd = w * SymToFull3x3(response.dForce);
    for (int a = 0; a < numSlices; ++a) {
      h[a] = SkewTimes<kBatchSize>(r[a], wd);
    }
    // dForce is symmetric for every pair, and skewDForceSkew for diagonal pairs, where skewADForce
    // equals skewBDForce.
    for (int a = 0; a < numSlices; ++a) {
      int const aa = outSums.PairIndex(a, a);
      VMatrix3x3r const skewDForce = SumLanes3x3<kBatchSize>(s[a] * h[a]);
      outSums.dForce[aa] += SumLanesSym3x3<kBatchSize>((s[a] * s[a]) * wd);
      outSums.skewBDForce[aa] += skewDForce;
      outSums.skewADForce[aa] += skewDForce;
      outSums.skewDForceSkew[aa] += SumLanesSym3x3<kBatchSize>(TimesSkew<kBatchSize>(h[a], r[a]));
      for (int b = a + 1; b < numSlices; ++b) {
        int const ab = outSums.PairIndex(a, b);
        outSums.dForce[ab] += SumLanesSym3x3<kBatchSize>((s[a] * s[b]) * wd);
        outSums.skewBDForce[ab] += SumLanes3x3<kBatchSize>(s[a] * h[b]);
        outSums.skewADForce[ab] += SumLanes3x3<kBatchSize>(s[b] * h[a]);
        outSums.skewDForceSkew[ab] += SumLanes3x3<kBatchSize>(TimesSkew<kBatchSize>(h[a], r[b]));
      }
    }
  }
}

// Add the SkinnedContactSums of 'pointRange', whose size must be a multiple of kBatchSize, to
// outSums.
template <int kBatchSize>
void AddSkinnedContactSums(
    Interval<int> pointRange,
    ContactDetectionResult const& contactQuery,
    ContactParams const& params,
    ContactEvalConfig const& config,
    real dtStage,
    Span<real const> weights,
    SkinnedContactSlices const& slices,
    bool assemEnergy,
    bool assemForce,
    bool assemDForce,
    Span<Real3> outForce,
    Allocator* scratch,
    SkinnedContactSums& outSums) {
  MOCHI_ASSERT_VERBOSE(pointRange.Size() % kBatchSize == 0);
  SkinnedContactBatchSlices<kBatchSize> batchSlices(slices.NumSlices(), assemDForce, scratch);
  for (int first = pointRange.Min(); first <= pointRange.Max(); first += kBatchSize) {
    AddBatchSkinnedContactSums<kBatchSize>(
        first,
        contactQuery,
        params,
        config,
        dtStage,
        weights,
        slices,
        assemEnergy,
        assemForce,
        assemDForce,
        outForce,
        batchSlices,
        outSums);
  }
}

} // namespace

SkinnedContactSums mochi::ComputeSkinnedContactSums(
    Interval<int> pointRange,
    ContactDetectionResult const& contactQuery,
    ContactParams const& params,
    ContactEvalConfig const& config,
    real dtStage,
    Span<real const> weights,
    SkinnedContactSlices const& slices,
    bool assemEnergy,
    bool assemForce,
    bool assemDForce,
    Allocator* scratch,
    Span<Real3> outForce) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(pointRange.Valid());
  MOCHI_ASSERT_VERBOSE(
      pointRange.Min() >= 0 && pointRange.Max() < isize(contactQuery.sampleIndices));
  MOCHI_ASSERT_VERBOSE(
      !config.explicitNormals ||
          contactQuery.sdfInfo.size() == contactQuery.sdfInfoStageStart.size(),
      "Explicit normals require stage-start SDF info.");
  MOCHI_ASSERT_VERBOSE(
      !config.validCollidingNormals ||
          contactQuery.sdfInfo.size() == contactQuery.normalColliding.size(),
      "Valid colliding normals require normalColliding.");
  MOCHI_ASSERT_VERBOSE(
      outForce.empty() || (assemForce && isize(outForce) == isize(contactQuery.sampleIndices)),
      "outForce requires assemForce and must be indexed like the contact points.");
  MOCHI_ASSERT_VERBOSE(
      contactQuery.colliderIntegrationWeights.empty(),
      "Skinned contact sums do not support collider integration weights.");

  // Allocate the output before any scratch data, to keep the FILO order.
  SkinnedContactSums sums(slices.NumSlices(), assemForce, assemDForce, scratch);

  // Full batches, then the remaining points as a narrower batch, which may use a narrower SIMD type
  // (see ComputeRigidContactSums).
  auto const addSums = [&]<int kBatchSize>(Interval<int> range) {
    AddSkinnedContactSums<kBatchSize>(
        range,
        contactQuery,
        params,
        config,
        dtStage,
        weights,
        slices,
        assemEnergy,
        assemForce,
        assemDForce,
        outForce,
        scratch,
        sums);
  };

  int const numPoints = pointRange.Size();
  int const numRemaining = numPoints % kCollResponseBatchSize;
  Interval<int> const fullBatches{pointRange.Min(), pointRange.Min() + numPoints - numRemaining};
  Interval<int> const lastBatch{fullBatches.Max() + 1, pointRange.Max() + 1};
  if (fullBatches.Size() > 0) {
    addSums.template operator()<kCollResponseBatchSize>(fullBatches);
  }
#define MOCHI_BATCH_CASE(N)                       \
  case N:                                         \
    if constexpr ((N) < kCollResponseBatchSize) { \
      addSums.template operator()<N>(lastBatch);  \
    }                                             \
    break;
  // clang-format off
  switch (numRemaining) {
    case 0:
      break;
    MOCHI_BATCH_CASE(1) MOCHI_BATCH_CASE(2) MOCHI_BATCH_CASE(3) MOCHI_BATCH_CASE(4)
    MOCHI_BATCH_CASE(5) MOCHI_BATCH_CASE(6) MOCHI_BATCH_CASE(7)
    default:
      static_assert(kCollResponseMaxBatchSize == 8, "Please update this switch statement");
      MOCHI_ASSERT_VERBOSE(false, "Unsupported batch size (%d).", numRemaining);
  }
#undef MOCHI_BATCH_CASE
  // clang-format on
  return sums;
}

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
  auto const& halfEdges = _halfEdge.halfEdges;
  HalfEdge const& heAB = halfEdges[_halfEdge.face2half[closestFace]];
  HalfEdge const& heBC = halfEdges[heAB.next];
  HalfEdge const& heCA = halfEdges[heBC.next];
  VDistanceSignParams signParams;
  signParams.normalA = ToSimd(_nodeNormals[face[0]]);
  signParams.normalB = ToSimd(_nodeNormals[face[1]]);
  signParams.normalC = ToSimd(_nodeNormals[face[2]]);
  signParams.normalAB = ToSimd(_edgeNormals[heAB.edge]);
  signParams.normalBC = ToSimd(_edgeNormals[heBC.edge]);
  signParams.normalCA = ToSimd(_edgeNormals[heCA.edge]);
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
    // The gradient of the signed distance is the face normal inside the face. On an edge or a
    // vertex, it is the signed direction of the offset from the closest point, computed from the
    // feature's own vertices. Rounding perturbs the offset by about epsilon times the query point's
    // distance to those vertices, and its direction by that over the length of the offset, which
    // leaves the direction undefined at the feature. So the offset is blended towards the
    // pseudonormal that determined the sign, weighted by that perturbation against how far the
    // gradient can deviate from the pseudonormal: |n_face - n_ps| for either face of an edge, and
    // the largest over a vertex's faces, since its pseudonormal can match one of them. The sign
    // keeps the gradient within 90° of the pseudonormal, a deviation of √2, which the boundary of
    // an open mesh can reach, so boundary edges take √2 and boundary vertices at least √2. The
    // normal is then within rounding of the gradient of the feature found, the face normal on flat
    // features whose stored normals agree, and the pseudonormal where the computed offset
    // vanishes. It is continuous up to rounding wherever the exact gradient is, and across edges,
    // where it tends to the pseudonormal, but not at sharp vertices, where the perturbation shrinks
    // with the offset, so it follows the exact gradient, which has no limit there. A point on a
    // vertex gets the vertex's pseudonormal, whichever incident face the search returned, since
    // rounding can leave its barycentrics on an incident edge or inside the face.
    Vec4r const faceNormal = ToSimd(_elementNormals[closestFace]);
    Vec4r normal = faceNormal;
    int const vertex = Equal<3>(position, A) ? 0
        : Equal<3>(position, B)              ? 1
        : Equal<3>(position, C)              ? 2
                                             : -1;
    Vec4r const isPositive = par > SimdZero();
    if (vertex >= 0) {
      normal = ToSimd(_nodeNormals[face[vertex]]);
    } else if (!AllTrue<3>(isPositive)) {
      bool const bA = IsTrue<0>(isPositive);
      bool const bB = IsTrue<1>(isPositive);
      bool const bC = IsTrue<2>(isPositive);
      Vec4r pseudonormal MOCHI_NO_INIT;
      real deviation MOCHI_NO_INIT;
      Vec4r diff MOCHI_NO_INIT;
      real roundoff MOCHI_NO_INIT;
      if (int(bA) + int(bB) + int(bC) == 1) {
        int const node = face[bA ? 0 : bB ? 1 : 2];
        pseudonormal = ToSimd(_nodeNormals[node]);
        deviation = _nodeNormalDeviations[node];
        diff = position - ToSimd(_nodeCoordinates[node]);
        roundoff = std::numeric_limits<real>::epsilon() * HMax<3>(Abs(diff));
      } else {
        HalfEdge const& he = !bC ? heAB : !bA ? heBC : heCA;
        pseudonormal = ToSimd(_edgeNormals[he.edge]);
        deviation = he.pair == -1 ? kSqrt2 : Norm<3>(faceNormal - pseudonormal);
        Vec4r const v0 = !bC ? A : !bA ? B : C;
        Vec4r const v1 = !bC ? B : !bA ? C : A;
        Vec4r const edge = v1 - v0;
        Vec4r const p0 = position - v0;
        Vec4r const p1 = position - v1;
        Vec4r const nearer = NormSqr<3>(p1) < NormSqr<3>(p0) ? p1 : p0;
        diff = nearer - (Dot<3>(nearer, edge) / NormSqr<3>(edge)) * edge;
        roundoff = std::numeric_limits<real>::epsilon() * HMax<3>(Max(Abs(p0), Abs(p1)));
      }
      Vec4r const blend = (deviation * signParams.outSign) * diff + roundoff * pseudonormal;
      real const blendSqr = NormSqr<3>(blend);
      normal =
          blendSqr >= std::numeric_limits<real>::min() ? Normalize(blend, blendSqr) : pseudonormal;
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

// Call LoadTransposed for a partial batch without inlining. This is a work-around for a performance
// issue with the Clang compiler. When inlined, this function changes the register allocation of the
// whole function causing a significant performance regression to the hot loop even though this
// function is called outside the hot loop.
template <class V3>
MOCHI_NO_INLINE static V3 LoadTransposedNoInline(Real3 const* points, int count) {
  V3 pt MOCHI_NO_INIT;
  LoadTransposed(&points[0][0], pt, count);
  return pt;
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
      StoreTransposed(&outContacts[offset + j][0], TransformPointsBatch(pt, rotT, trans));
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
    pt = LoadTransposedNoInline<V3>(&points[iSrc], numRemaining);
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
      pt = TransformPointsBatch(pt, rotT, trans); // Transform to output space
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
    pt = LoadTransposedNoInline<V3>(&points[iSrc], numRemaining);
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
  // Transposed inverse rotation, which is the rotation itself.
  V3x3 const boxFromPointsRotT = Broadcast3x3<V>(Transpose4x4(pointsFromBoxMatT));
  V3 const boxFromPointsTrans = Broadcast3<V>(-pointsFromBoxMatT[3]); // Inverse translation

  // Compute transformation from box local space to the output space.
  auto const colliderFromPoints = Invert(pointsFromCollider);
  auto const colliderFromBoxMatT =
      Transpose4x4(Dot4x4(ToVMatrix4x4(colliderFromPoints), Transpose4x4(pointsFromBoxMatT)));
  V3x3 const colliderFromBoxRotT = Broadcast3x3<V>(colliderFromBoxMatT);
  V3 const colliderFromBoxTrans = Broadcast3<V>(colliderFromBoxMatT[3]);

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
      normal = RotateVectorsBatch(normal, colliderFromBoxRotT);
      StoreTransposed(&outSdf.grad[offset + j][0], normal);

      // Apply the output transform to the position
      V3 outPos = TransformPointsBatch(localPos, colliderFromBoxRotT, colliderFromBoxTrans);
      StoreTransposed(&outContacts[offset + j][0], outPos);
    }
    iBuf = 0;
  };

  Vec4r halfExtPlusTol = colliderTransformed.VGetHalfExtents() + params.tolerance;
  auto process = [&](V3 const& pt) MOCHI_FORCE_INLINE_LAMBDA {
    // Transform to the local space of the box.
    V3 localPos = RotateVectorsBatch(pt + boxFromPointsTrans, boxFromPointsRotT);
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
    LoadTransposed(&points[iSrc][0], pt, numRemaining);
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
  DynamicArray<Real3> pointsCulled;
  DynamicArray<int> indsCulled;
  pointsCulled.resize_noinit(points.size());
  indsCulled.resize_noinit(points.size());
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

void mochi::ContactJac::CoalesceColumnsByCachedMapping(
    ContactJacColumnCoalescingMap const& mapping,
    Span<int const> sampleIndices,
    int dofOffset) {
  MOCHI_ASSERT_VERBOSE(!hasSharedDoFs, "Coalescing requires per-contact DoF indices.");
  MOCHI_ASSERT_VERBOSE(!hasSharedJacs, "Coalescing requires per-contact Jacobians.");
  MOCHI_ASSERT_VERBOSE(_jacAux.empty(), "Coalescing does not support an auxiliary Jacobian.");
  MOCHI_ASSERT_VERBOSE(
      nDoFsInternal == nDoFsState, "Internal and state Jacobian widths must match.");
  MOCHI_ASSERT_VERBOSE(!groupsInitialized, "Index groups must not be initialized.");
  MOCHI_ASSERT_VERBOSE(isize(sampleIndices) == nContacts, "Expected one sample index per contact.");
  MOCHI_ASSERT_VERBOSE(mapping.sourceBlockCount > 0, "Source block count must be positive.");
  MOCHI_ASSERT_VERBOSE(mapping.samplesPerElement > 0, "Samples per element must be positive.");
  MOCHI_ASSERT_VERBOSE(!mapping.elementDofOffsets.empty(), "Element DoF offsets are missing.");
  MOCHI_ASSERT_VERBOSE(
      mapping.elementDofOffsets.front() == 0 &&
          mapping.elementDofOffsets.back() == isize(mapping.elementDofIndices),
      "Element DoF offsets are inconsistent.");
  int const numElements = mapping.NumElements();
  MOCHI_ASSERT_VERBOSE(
      isize(mapping.elementSourceContributionOffsets) == numElements * mapping.sourceBlockCount + 1,
      "Source contribution offsets are inconsistent.");
  MOCHI_ASSERT_VERBOSE(
      mapping.elementSourceContributionOffsets.front() == 0 &&
          mapping.elementSourceContributionOffsets.back() ==
              isize(mapping.elementSourceContributionDstColumns),
      "Source contribution offsets are inconsistent.");
  for (int i = 0; i < numElements; ++i) {
    MOCHI_ASSERT_VERBOSE(mapping.elementDofOffsets[i] <= mapping.elementDofOffsets[i + 1]);
  }
  for (int i = 0; i + 1 < isize(mapping.elementSourceContributionOffsets); ++i) {
    MOCHI_ASSERT_VERBOSE(
        mapping.elementSourceContributionOffsets[i] <=
        mapping.elementSourceContributionOffsets[i + 1]);
  }

  int const oldWidth = nDoFsInternal;
  MOCHI_ASSERT_VERBOSE(
      oldWidth % mapping.sourceBlockCount == 0,
      "Wide Jacobian width must be divisible by the source block count.");
  int const querySourceStride = oldWidth / mapping.sourceBlockCount;

  if (nContacts == 0) {
    _data.inds.resize_noinit(0);
    _data.jac.resize_noinit(0);
    nDoFsInternal = 0;
    nDoFsState = 0;
    _inds.Reset(_data.inds.data(), 0, 0);
    _jac.Reset(_data.jac.data(), kDofsPerNode, 0);
    return;
  }

  int newWidth = 0;
  for (int contact = 0; contact < nContacts; ++contact) {
    int const sampleIndex = sampleIndices[contact];
    MOCHI_ASSERT_VERBOSE(sampleIndex >= 0, "Sample indices must be nonnegative.");
    int const elementIndex = sampleIndex / mapping.samplesPerElement;
    MOCHI_ASSERT_VERBOSE(elementIndex >= 0 && elementIndex < numElements);

    auto const elementDofs = mapping.ElementDofIndices(elementIndex);
    MOCHI_ASSERT_VERBOSE(!elementDofs.empty(), "Selected elements must affect at least one DoF.");
    for (int i = 0; i < isize(elementDofs); ++i) {
      MOCHI_ASSERT_VERBOSE(elementDofs[i] >= 0, "Cached DoF indices must be nonnegative.");
      MOCHI_ASSERT_VERBOSE(
          i == 0 || elementDofs[i - 1] < elementDofs[i],
          "Cached element DoF indices must be strictly increasing.");
    }
    newWidth = Max(newWidth, isize(elementDofs));
    for (int sourceBlock = 0; sourceBlock < mapping.sourceBlockCount; ++sourceBlock) {
      auto const destinations = mapping.SourceContributionDstColumns(elementIndex, sourceBlock);
      MOCHI_ASSERT_VERBOSE(
          isize(destinations) <= querySourceStride,
          "A source block has more mapped contributions than the query stride.");
      for ([[maybe_unused]] int destination : destinations) {
        MOCHI_ASSERT_VERBOSE(destination >= 0 && destination < isize(elementDofs));
      }
    }
  }
  MOCHI_ASSERT_VERBOSE(newWidth <= oldWidth, "Cached coalescing cannot increase the width.");

  // Holds one contact's wide columns; wider Jacobians fall back to the heap.
  int constexpr kMaxStackColumns = 64;
  MOCHI_FILO_STACK_ALLOCATOR(tempAlloc, kMaxStackColumns * sizeof(Real3));
  DynamicArray<Real3> originalColumns(&tempAlloc);
  originalColumns.resize_noinit(oldWidth);
  for (int contact = 0; contact < nContacts; ++contact) {
    auto indices = Inds(contact);
    [[maybe_unused]] auto sourceIndices = indices;
    auto jac = Jac(contact);
    for (int column = 0; column < oldWidth; ++column) {
      originalColumns[column] = {jac(0, column), jac(1, column), jac(2, column)};
    }

    int const elementIndex = sampleIndices[contact] / mapping.samplesPerElement;
    auto const elementDofs = mapping.ElementDofIndices(elementIndex);
    indices = Span<int>(_data.inds.data() + contact * newWidth, newWidth);
    jac.Reset(_data.jac.data() + kDofsPerNode * contact * newWidth, kDofsPerNode, newWidth);
    jac.SetZero();

    for (int sourceBlock = 0; sourceBlock < mapping.sourceBlockCount; ++sourceBlock) {
      auto const destinations = mapping.SourceContributionDstColumns(elementIndex, sourceBlock);
      for (int localEntry = 0; localEntry < isize(destinations); ++localEntry) {
        int const sourceColumn = sourceBlock * querySourceStride + localEntry;
        int const destinationColumn = destinations[localEntry];
        MOCHI_ASSERT_VERBOSE(
            sourceIndices[sourceColumn] == elementDofs[destinationColumn] + dofOffset,
            "Cached coalescing map does not match the source Jacobian indices.");
        for (int component = 0; component < kDofsPerNode; ++component) {
          jac(component, destinationColumn) += originalColumns[sourceColumn][component];
        }
      }
    }

    for (int column = 0; column < isize(elementDofs); ++column) {
      long long const globalDof =
          static_cast<long long>(elementDofs[column]) + static_cast<long long>(dofOffset);
      MOCHI_ASSERT_VERBOSE(
          globalDof >= std::numeric_limits<int>::min() &&
              globalDof <= std::numeric_limits<int>::max(),
          "Global DoF index is not representable as int.");
      indices[column] = static_cast<int>(globalDof);
    }
    for (int column = isize(elementDofs); column < newWidth; ++column) {
      indices[column] = indices[0];
    }
  }

  _data.inds.resize_noinit(nContacts * newWidth);
  _data.jac.resize_noinit(kDofsPerNode * nContacts * newWidth);
  nDoFsInternal = newWidth;
  nDoFsState = newWidth;
  _inds.Reset(_data.inds.data(), nContacts, newWidth);
  _jac.Reset(_data.jac.data(), kDofsPerNode, nContacts * newWidth);
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

  // Prepare index groups. Create a temporary data structure that stores the index groups of each
  // row that differs from the previous one. Consecutive contacts often act on the same DoFs (e.g.,
  // samples of one surface element), so a repeated row reuses the previous row's groups. Until the
  // final storage is filled, indGroupCounts holds the index of each row's groups in indGroupsAll.
  MOCHI_FILO_STACK_ALLOCATOR(tempAlloc, 16 * 1024); // Probably more than enough
  DynamicArray<IndexGroups> indGroupsAll(&tempAlloc);
  indGroupsAll.reserve(numRows);
  _data.indGroupCounts.resize_noinit(numRows);
  int maxGroups = 0;
  for (int i = 0; i < numRows; i++) {
    if (i == 0 || Inds(i) != Inds(i - 1)) {
      auto groups = CreateIndexGroups(Inds(i), forceTriplets, &tempAlloc);
      maxGroups = Max(maxGroups, isize(groups));
      indGroupsAll.emplace_back(std::move(groups));
    }
    _data.indGroupCounts[i] = isize(indGroupsAll) - 1;
  }

  // Create final storage. Use the maximum group count across all contacts, since different
  // contacts may have different numbers of index groups (e.g., rod surface contact where
  // skinning connectivity varies per contact). Per-contact group counts are stored so that
  // IndGroups(i) returns only the valid entries for each contact.
  _data.indGroups.clear();
  _data.indGroups.resize_noinit(numRows * maxGroups);
  _indGroups.Reset(_data.indGroups.data(), numRows, maxGroups);
  for (int i = 0; i < numRows; i++) {
    auto const& groups = indGroupsAll[_data.indGroupCounts[i]];
    _data.indGroupCounts[i] = isize(groups);
    std::copy_n(groups.begin(), groups.size(), _indGroups.Row(i).Data());
  }

  // The index groups are properly initialized
  groupsInitialized = true;
}
