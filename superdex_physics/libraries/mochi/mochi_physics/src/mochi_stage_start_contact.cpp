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

#include "mochi_contact.h"

#include "mochi_deformable.h"
#include "mochi_point_cloud_contact.h"
#include "mochi_shell.h"

#include <mochi_core/contact/contact_correspondence.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_core/utils/vmatrix.h>

#include <entt/entt.hpp>

#include <functional>
#include <utility>

using namespace mochi;

template <ContactType kContactType>
void mochi::AddMissingStageStartCollisions(
    CActiveCollisions<kContactType, TimeStep::StageStart> const& stageStartCollisions,
    CActiveCollisions<kContactType, TimeStep::Current>& outCurrentCollisions) {
  // outCurrentCollisions is laid out as consecutive blocks of numPartitions entries per
  // colliderEntity (see CActiveCollisions::SetUp). We must preserve this invariant: when a
  // collider is in stageStartCollisions but missing from outCurrentCollisions, add ALL of
  // its partitions, including ones whose sampleIndices are empty. Otherwise SetUp on the
  // next step would observe size() % numPartitions != 0.

  // First pass: identify [first, last] (inclusive) ranges in stageStartCollisions that
  // belong to colliders missing from outCurrentCollisions and that have at least one
  // partition with contacts.
  // Each std::pair<int, int> is 8 bytes; 1024 bytes covers ~128 missing colliders before
  // the FILO falls back to heap pages.
  MOCHI_FILO_STACK_ALLOCATOR(filoAllocator, 128 * sizeof(std::pair<int, int>));
  DynamicArray<std::pair<int, int>> missingRanges(&filoAllocator);
  int totalMissing = 0;

  int s = 0;
  int c = 0;
  while (s < isize(stageStartCollisions)) {
    auto const colliderEntity = stageStartCollisions[s].colliderEntity;

    // Find the end of this collider's block in stageStartCollisions.
    int sEnd = s + 1;
    while (sEnd < isize(stageStartCollisions) &&
           stageStartCollisions[sEnd].colliderEntity == colliderEntity) {
      ++sEnd;
    }

    // Advance c past entries with smaller colliderEntity in outCurrentCollisions.
    while (c < isize(outCurrentCollisions) &&
           outCurrentCollisions[c].colliderEntity < colliderEntity) {
      ++c;
    }

    // If colliderEntity is present, skip its block in outCurrentCollisions.
    bool inCurrent = false;
    while (c < isize(outCurrentCollisions) &&
           outCurrentCollisions[c].colliderEntity == colliderEntity) {
      inCurrent = true;
      ++c;
    }
    if (!inCurrent) {
      // Missing collider. Add the block iff at least one partition has contacts.
      bool anyNonEmpty = false;
      for (int i = s; i < sEnd; ++i) {
        if (!stageStartCollisions[i].collisionResult.sampleIndices.empty()) {
          anyNonEmpty = true;
          break;
        }
      }
      if (anyNonEmpty) {
        missingRanges.emplace_back(s, sEnd - 1);
        totalMissing += sEnd - s;
      }
    }
    s = sEnd;
  }

  if (totalMissing == 0) {
    return;
  }

  // Second pass: merge missing blocks into current, back to front. Because totalMissing
  // entries are appended to the end first, we can shift in place without a temp buffer.
  int const originalSize = isize(outCurrentCollisions);
  int const newSize = originalSize + totalMissing;
  outCurrentCollisions.resize(newSize);

  c = originalSize - 1;
  int m = newSize - 1;
  for (int r = isize(missingRanges) - 1; r >= 0; --r) {
    auto const [rangeBegin, rangeEnd] = missingRanges[r];
    auto const colliderEntity = stageStartCollisions[rangeBegin].colliderEntity;

    // Shift current entries with greater colliderEntity to the right of the new block.
    while (c >= 0 && outCurrentCollisions[c].colliderEntity > colliderEntity) {
      outCurrentCollisions[m] = outCurrentCollisions[c];
      --c;
      --m;
    }

    // Insert all partitions of this collider, preserving their order.
    for (int i = rangeEnd; i >= rangeBegin; --i) {
      outCurrentCollisions[m].colliderEntity = colliderEntity;
      outCurrentCollisions[m].collisionResult = {};
      outCurrentCollisions[m].collisionResult.collidingPartitionId =
          stageStartCollisions[i].collisionResult.collidingPartitionId;
      outCurrentCollisions[m].collisionResult.isSdfGradUnitary =
          stageStartCollisions[i].collisionResult.isSdfGradUnitary;
      --m;
    }
  }
  MOCHI_ASSERT_VERBOSE(c == m, "Internal error: merge index mismatch");
}

// Explicit template instantiations
template void mochi::AddMissingStageStartCollisions<ContactType::Async>(
    CActiveCollisions<ContactType::Async, TimeStep::StageStart> const& stageStartCollisions,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& outCurrentCollisions);

template void mochi::AddMissingStageStartCollisions<ContactType::Sync>(
    CActiveCollisions<ContactType::Sync, TimeStep::StageStart> const& stageStartCollisions,
    CActiveCollisions<ContactType::Sync, TimeStep::Current>& outCurrentCollisions);

static void TransformToColliderSpaceMapped(
    ContactDetectionResult const& stageStart,
    Span<ContactCorrespondence::Pair const> missingSamples,
    Span<Real3> outPos) {
  for (int i = 0; i < missingSamples.size(); ++i) {
    auto const& contactIdx = missingSamples[i].second;
    outPos[i] = stageStart.posColliding[contactIdx];
  }
}

static void TransformToColliderSpacePointCloud(
    entt::registry const& reg,
    entt::entity collider,
    CContactSamples<TimeStep::Current> const& samples,
    ContactDetectionResult const& stageStart,
    VMatrix4x4r const& colliderFromCollidingT,
    Span<ContactCorrespondence::Pair const> missingSamples,
    Span<Real3> outPos) {
  auto const& colliderDiscretization = reg.get<CColliderPointCloudDiscretization const>(collider);
  auto const& disp = reg.get<CFinalDisplacementRef<TimeStep::Current> const>(collider).value;

  // The computation of collider-space positions for point-cloud colliders mimics the approach in
  // ComputePointCloudContactDetectionFields() in mochi_point_cloud_contact.cpp. Each collider
  // point defines a local reference system.
  colliderDiscretization.VisitCollider([&](auto const& disc) {
    using DiscretizationT = std::decay_t<decltype(disc)>;
    int constexpr kNumQuads = DiscretizationT::kNumQuads;

    for (int i = 0; i < missingSamples.size(); ++i) {
      auto const& [sampleIdx, contactIdx] = missingSamples[i];
      Vec4r const collidingSamplePos = ToSimd(samples.positions[sampleIdx], 1_r);
      int const cpIndex = stageStart.colliderFeatureIndices[contactIdx];
      int const elementIndex = cpIndex / kNumQuads;
      int const localQuadIndex = cpIndex % kNumQuads;
      auto const& element = disc.femElements[elementIndex];
      Real3 const colliderPointPos = details::InterpolateColliderPointPosition<DiscretizationT>(
          element, elementIndex, localQuadIndex, disp, colliderDiscretization.dofsPerNode);
      Vec4r const colliderPointPosition = Load<kSpaceDim3, Vec4r>(&colliderPointPos[0]);
      Vec4r const collidingInColliderFrame =
          DotVecMat4x4(collidingSamplePos, colliderFromCollidingT) - colliderPointPosition;
      outPos[i] = ToReal3(collidingInColliderFrame);
    }
  });
}

static void TransformToColliderSpaceRigid(
    CContactSamples<TimeStep::Current> const& samples,
    VMatrix4x4r const& colliderFromCollidingT,
    Span<ContactCorrespondence::Pair const> missingSamples,
    Span<Real3> outPos) {
  for (int i = 0; i < missingSamples.size(); ++i) {
    auto const& sampleIdx = missingSamples[i].first;
    Vec4r const collidingSamplePos = ToSimd(samples.positions[sampleIdx], 1_r);
    outPos[i] = ToReal3(DotVecMat4x4(collidingSamplePos, colliderFromCollidingT));
  }
}

template <ContactType kContactType>
static void AddPerCollisionMissingStageStartContacts(
    entt::registry const& reg,
    entt::entity collider,
    CContactSamples<TimeStep::Current> const& samples,
    TransformRT const& worldFromColliding,
    bool addPadding,
    ContactDetectionResult const& stageStart,
    ContactDetectionResult& current,
    CContactCorrespondence<kContactType>& correspondence) {
  // Identify stage-start contacts missing in current using correspondence lookup.
  Span<ContactCorrespondence::Pair const> missingSamples = correspondence.GetMissingSamples(
      stageStart.sampleIndices,
      stageStart.colliderFeatureIndices,
      current.sampleIndices,
      current.colliderFeatureIndices);
  if (missingSamples.empty()) {
    return;
  }

  // Identify collider type.
  bool const isMappedCollider =
      reg.try_get<CSdfMapping<TimeStep::Current> const>(collider) != nullptr;
  bool const isPointCloudCollider = reg.all_of<TagUsePointCloudContact>(collider);

  // Get the transform of the collider.
  auto const& worldFromCollider = GetRootTransform<TimeStep::Current>(reg, collider);
  auto const colliderFromColliding = Invert(worldFromCollider) * worldFromColliding;
  VMatrix4x4r const colliderFromCollidingT = ToVMatrix4x4Transpose(colliderFromColliding);

  // Transform missing samples to collider space.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 1024 * sizeof(Real3));
  DynamicArray<Real3> posCollidingMissing(&allocator);
  int const numMissing = isize(missingSamples);
  posCollidingMissing.resize_noinit(numMissing);
  if (isMappedCollider) {
    TransformToColliderSpaceMapped(stageStart, missingSamples, posCollidingMissing);
  } else if (isPointCloudCollider) {
    TransformToColliderSpacePointCloud(
        reg,
        collider,
        samples,
        stageStart,
        colliderFromCollidingT,
        missingSamples,
        posCollidingMissing);
  } else {
    TransformToColliderSpaceRigid(
        samples, colliderFromCollidingT, missingSamples, posCollidingMissing);
  }

  // Reserve space for appending missing stage-start contacts.
  // Mapped colliders have per-contact Jacobians; rigid and point-cloud colliders share a single
  // Jacobian. Point-cloud colliders have integration weights and feature indices; others do not.
  // Mapped and point-cloud colliders have jacWorldFromDofs; rigid colliders do not.
  int const newSize = isize(current.sampleIndices) + numMissing;
  current.sampleIndices.reserve(newSize);
  current.posColliding.reserve(newSize);
  current.sdfInfo.val.reserve(newSize);
  current.sdfInfo.grad.reserve(newSize);
  current.posCollidingStageStart.reserve(newSize);
  current.sdfInfoStageStart.val.reserve(newSize);
  current.sdfInfoStageStart.grad.reserve(newSize);
  if (isMappedCollider) {
    current.jacColliderFromWorld.reserve(newSize);
    current.jacColliderFromWorldStageStart.reserve(newSize);
  } else if (current.sampleIndices.empty()) {
    // Rigid or point-cloud collider with no current contacts - initialize shared Jacobian.
    current.jacColliderFromWorld.resize_noinit(1);
    current.jacColliderFromWorld[0] = ToVMatrix3x3Transpose(worldFromCollider.GetRotation());
    current.jacColliderFromWorldStageStart.resize_noinit(1);
    current.jacColliderFromWorldStageStart[0] = stageStart.jacColliderFromWorld[0];
  }
  if (isPointCloudCollider) {
    current.colliderIntegrationWeights.reserve(newSize);
    current.colliderFeatureIndices.reserve(newSize);
  }
  if (isMappedCollider || isPointCloudCollider) {
    current.jacWorldFromDofs.reserve(newSize);
  }

  // If the current collision is empty, initialize shared data.
  if (current.sampleIndices.empty()) {
    current.isSdfGradUnitary = stageStart.isSdfGradUnitary;
    current.ndofs = stageStart.ndofs;
  }

  // Append missing stage-start contacts.
  // As the point is not colliding, use the distance tolerance and the stage-start gradient.
  auto const& colliderContactParams = reg.get<CContactParams const>(collider);
  real const distanceTolerance = colliderContactParams.GetPenaltyThresholdDist(addPadding);
  for (int i = 0; i < numMissing; ++i) {
    auto const& [sampleIdx, contactIdx] = missingSamples[i];
    current.sampleIndices.push_back(sampleIdx);
    current.posColliding.push_back(posCollidingMissing[i]);
    current.sdfInfo.val.push_back(distanceTolerance);
    current.sdfInfo.grad.push_back(stageStart.sdfInfo.grad[contactIdx]);
    current.posCollidingStageStart.push_back(stageStart.posColliding[contactIdx]);
    current.sdfInfoStageStart.val.push_back(stageStart.sdfInfo.val[contactIdx]);
    current.sdfInfoStageStart.grad.push_back(stageStart.sdfInfo.grad[contactIdx]);
    if (isMappedCollider) {
      current.jacColliderFromWorld.push_back(stageStart.jacColliderFromWorld[contactIdx]);
      current.jacColliderFromWorldStageStart.push_back(stageStart.jacColliderFromWorld[contactIdx]);
    }
    if (isPointCloudCollider) {
      current.colliderIntegrationWeights.push_back(
          stageStart.colliderIntegrationWeights[contactIdx]);
      current.colliderFeatureIndices.push_back(stageStart.colliderFeatureIndices[contactIdx]);
    }
    if (isMappedCollider || isPointCloudCollider) {
      current.jacWorldFromDofs.push_back(stageStart.jacWorldFromDofs[contactIdx]);
    }
  }
}

template <ContactType kContactType>
static void AddMissingStageStartContacts(
    entt::registry const& reg,
    CContactSamples<TimeStep::Current> const& samples,
    TransformRT const& worldFromColliding,
    bool addPadding,
    CActiveCollisions<kContactType, TimeStep::StageStart> const& stageStartCollisions,
    CActiveCollisions<kContactType, TimeStep::Current>& outCurrentCollisions,
    CContactCorrespondence<kContactType>& correspondence) {
  int s = 0;
  for (int c = 0; c < isize(outCurrentCollisions); ++c) {
    while (s < isize(stageStartCollisions) && stageStartCollisions[s] < outCurrentCollisions[c]) {
      ++s;
    }
    if (s < isize(stageStartCollisions) && stageStartCollisions[s] == outCurrentCollisions[c]) {
      AddPerCollisionMissingStageStartContacts(
          reg,
          outCurrentCollisions[c].colliderEntity,
          samples,
          worldFromColliding,
          addPadding,
          stageStartCollisions[s].collisionResult,
          outCurrentCollisions[c].collisionResult,
          correspondence);
      ++s;
    }
  }
}

template <ContactType kContactType>
void mochi::AddStageStartCollisionDetection(
    entt::registry const& reg,
    entt::entity e,
    CColliderInfo const& colliderInfo,
    CContactSamples<TimeStep::Current> const& samples,
    CActiveCollisions<kContactType, TimeStep::StageStart> const& stageStartCollisions,
    CActiveCollisions<kContactType, TimeStep::Current>& outCurrentCollisions,
    CContactCorrespondence<kContactType>& correspondence) {
  bool const addPadding = ShouldAddPenaltyPadding(colliderInfo.type);
  auto const& worldFromColliding = GetRootTransform<TimeStep::Current>(reg, e);
  AddMissingStageStartCollisions(stageStartCollisions, outCurrentCollisions);
  AddMissingStageStartContacts(
      reg,
      samples,
      worldFromColliding,
      addPadding,
      stageStartCollisions,
      outCurrentCollisions,
      correspondence);
}

// Explicit template instantiations
template void mochi::AddStageStartCollisionDetection<ContactType::Async>(
    entt::registry const& reg,
    entt::entity e,
    CColliderInfo const& colliderInfo,
    CContactSamples<TimeStep::Current> const& samples,
    CActiveCollisions<ContactType::Async, TimeStep::StageStart> const& stageStartCollisions,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& outCurrentCollisions,
    CContactCorrespondence<ContactType::Async>& correspondence);

template void mochi::AddStageStartCollisionDetection<ContactType::Sync>(
    entt::registry const& reg,
    entt::entity e,
    CColliderInfo const& colliderInfo,
    CContactSamples<TimeStep::Current> const& samples,
    CActiveCollisions<ContactType::Sync, TimeStep::StageStart> const& stageStartCollisions,
    CActiveCollisions<ContactType::Sync, TimeStep::Current>& outCurrentCollisions,
    CContactCorrespondence<ContactType::Sync>& correspondence);
