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

#include "mochi_articulated_body.h"
#include "mochi_blended.h"
#include "mochi_contact_filter.h"
#include "mochi_deformable.h"
#include "mochi_ecs_utils.h"
#include "mochi_group.h"
#include "mochi_hyper_reduction.h"
#include "mochi_island.h"
#include "mochi_point_cloud_contact.h"
#include "mochi_rigid.h"
#include "mochi_rod.h"
#include "mochi_scene_recorder.h"
#include "mochi_shell.h"
#include "mochi_simulation.h"
#include "mochi_soft.h"
#include "mochi_soft_rom_systems.h"
#include "mochi_soft_skinned.h"

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/grid_sdf.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/sparsity_utils.h>

#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

using namespace mochi;
using namespace mochi::experimental;

/*************************************************************************************************/

// Maximum number of consecutive sample points to process together by ContactDResXYZ utilities if
// they all affect the same DoFs. Processing them together reduces the number of write operations
// from the dense matrix in which the local contact dresidual is accumulated to the global contact
// dresidual (expensive). The larger the max batch size, the larger the initialization overhead.
static constexpr int kMaxBatchSize = 6; // More than enough for default of 3 sample points per face.

template <bool kAllowFarSdfQuery>
static real GetFarSdfEvaluationDistance(
    ecs::PartialRegistry<CRequiresFarSdfEvaluation const> reg,
    entt::entity e) {
  if constexpr (kAllowFarSdfQuery) {
    auto const* farEval = reg.try_get<CRequiresFarSdfEvaluation const>(e);
    return farEval ? farEval->maxDistance : 0.0_r;
  } else {
    return 0.0_r;
  }
}

// Find all the entities that might collide with the colliding actor. Cull the ones that don't
// overlap the colliding's bounding volume at all. Outputs CPotentialColliders with a list of the
// entities that are worth considering for collision detection.
template <ContactType kContactType, TimeStep kTimeStep, bool kAllowFarSdfQuery = false>
static void UpdatePotentialColliders(
    entt::registry const& reg,
    entt::entity colliding,
    CBoundingVolume<TimeStep::Current> const& boundsColliding,
    CConservativePotentialColliders<kContactType> const& conservativeColliders,
    CContactLayer const& layerColliding,
    CIslandMemberInfo const& islandColliding,
    CPotentialColliders<kContactType>& potentialColliders) {
  MOCHI_PROFILE_SCOPE();

  bool constexpr kIsSync = kContactType == ContactType::Sync;

  // Discard previous colliders
  potentialColliders.clear();

  // Get colliding transform
  TransformRT const& worldFromColliding = GetRootTransform<kTimeStep>(reg, colliding);

  if constexpr (kAllowFarSdfQuery) {
    // (kAllowFarSdfQuery == true) requires is to consider contact with actors that are outside of
    // the normal bounding volume. This means we can't use CConservativePotentialColliders. Instead,
    // we have to check every other actor in the scene with O(N^2) cost.

    real farEvaluationDistance = GetFarSdfEvaluationDistance<kAllowFarSdfQuery>(reg, colliding);

    AnyShape const worldBoundsColliding = TransformShape(
        worldFromColliding, ExpandShape(boundsColliding.localShape, farEvaluationDistance));

    auto const& contactTable = reg.ctx<CContactFilterTable const>();

    // Iterate over all shapes with a collider
    reg.view<
           CColliderInfo const,
           CContactLayer const,
           CIslandMemberInfo const*,
           CBoundingVolumeFor<kContactType, kTimeStep> const,
           CContactParams const>()
        .each([&](entt::entity collider,
                  auto const& colliderInfo,
                  auto const& layerCollider,
                  auto const* islandCollider,
                  auto const& boundsCollider,
                  auto const& paramsCollider) {
          // Filter out self contact
          if (collider == colliding) {
            return;
          }

          // Filter out disabled colliders
          if (colliderInfo.type == ColliderType::None) {
            return;
          }

          // Check contact filter settings
          if (!contactTable.IsContactEnabled(
                  colliding, collider, layerColliding.id, layerCollider.id)) {
            return;
          }

          // For ContactType::Sync, only consider actors in the same island.
          // For ContactType::Async, only consider actors NOT in the same island.
          bool sameIsland = (islandCollider && (islandCollider->island == islandColliding.island));
          if (sameIsland != kIsSync) {
            return;
          }

          // Get root transform.
          TransformRT const& worldFromCollider = GetRootTransform<kTimeStep>(reg, collider);

          // Pad the bounds of the collider by the penalty threshold distance.
          // We use the penalty threshold of the collider, which is the one used for contact
          AnyShape worldBoundsCollider = TransformShape(
              worldFromCollider,
              ExpandColliderBoundsForContact(boundsCollider.localShape, paramsCollider));

          // Check for overlap
          if (HasOverlap(worldBoundsColliding, worldBoundsCollider)) {
            potentialColliders.emplace_back(collider);
          }
        });
  } else {
    // conservativeColliders lists actors that overlap the CConservativeStepBounds of the colliding
    // actor. Collider types and contact filtering have already been taken into account. We simply
    // have to find the subset of those actors which also overlap the colliding's tight fitting
    // bounds.
    AnyShape const worldBoundsColliding =
        TransformShape(worldFromColliding, boundsColliding.localShape);
    for (auto const& potentialColliderData : conservativeColliders) {
      auto collider = potentialColliderData.entity;

      MOCHI_ASSERT_VERBOSE(
          (reg.all_of<
              CBoundingVolumeFor<kContactType, kTimeStep>,
              CColliderInfo,
              CContactParams,
              CRootTransform>(collider)),
          "Entities listed by CConservativePotentialColliders should have all of these components.");

      // Colliders must not have type "none".
      MOCHI_ASSERT_VERBOSE(
          reg.get<CColliderInfo const>(collider).type != ColliderType::None,
          "CConservativePotentialColliders should only list entities with valid colliders.");

      // Get root transform.
      TransformRT const& worldFromCollider = GetRootTransform<kTimeStep>(reg, collider);

      // Get the world-space bounds of the collider and pad it by the penalty threshold distance
      // We use the penalty threshold of the collider, which is the one used for contact
      auto const& boundsCollider =
          reg.get<CBoundingVolumeFor<kContactType, kTimeStep> const>(collider);

      auto const& paramsCollider = reg.get<CContactParams const>(collider);
      AnyShape worldBoundsCollider = TransformShape(
          worldFromCollider,
          ExpandColliderBoundsForContact(boundsCollider.localShape, paramsCollider));

      // Check for overlap
      if (HasOverlap(worldBoundsColliding, worldBoundsCollider)) {
        potentialColliders.emplace_back(collider);
      }
    }
  }
}

// Note: outResult is in collider's frame.
template <class T>
static void QueryBasicShape(
    entt::registry const& reg,
    entt::entity collider,
    entt::entity /* colliding */,
    AnyShape const& /* colliderBoundsInCollider */,
    TransformRT const& worldFromColliding,
    TransformRT const& worldFromCollider,
    Span<Real3 const> positionsInColliding,
    Span<int const> /* indicesToQuery */,
    ContactDetectionParams const& params,
    Span<int const> /* colliderFeatureIndices */,
    DynamicArray<int>& outIndices,
    DynamicArray<int>* /* outColliderFeatureIndices */,
    DynamicArray<Real3>& outContacts,
    SdfInfo& outSdf,
    bool& outIsSdfGradUnitary,
    int* /* outNDofs */,
    DynamicArray<real>* /* outColliderIntegrationWeights */,
    DynamicArray<VMatrix3x3r>* /* outMapJac */,
    DynamicArray<ColliderJacDofs>* /* outDofsJac */) {
  // Run the query.
  auto const colliderShape = reg.get<T const>(collider).shape;
  auto const collidingFromCollider = Invert(worldFromColliding) * worldFromCollider;
  FindPointContactsParallel(
      positionsInColliding,
      &colliderShape,
      params,
      collidingFromCollider,
      outIndices,
      outContacts,
      outSdf,
      outIsSdfGradUnitary);
}

// Note: outResult is in collider's frame.
static void QueryMesh(
    entt::registry const& reg,
    entt::entity collider,
    entt::entity /* colliding */,
    AnyShape const& /* colliderBoundsInCollider */,
    TransformRT const& worldFromColliding,
    TransformRT const& worldFromCollider,
    Span<Real3 const> positionsInColliding,
    Span<int const> /* indicesToQuery */,
    ContactDetectionParams const& params,
    Span<int const> /* colliderFeatureIndices */,
    DynamicArray<int>& outIndices,
    DynamicArray<int>* /* outColliderFeatureIndices */,
    DynamicArray<Real3>& outContacts,
    SdfInfo& outSdf,
    bool& outIsSdfGradUnitary,
    int* /* outNDofs */,
    DynamicArray<real>* /* outColliderIntegrationWeights */,
    DynamicArray<VMatrix3x3r>* /* outMapJac */,
    DynamicArray<ColliderJacDofs>* /* outDofsJac */) {
  // Run the query
  auto const& meshCollider = reg.get<CMeshCollider const>(collider);
  auto const collidingFromCollider = Invert(worldFromColliding) * worldFromCollider;
  FindPointContactsParallel(
      positionsInColliding,
      static_cast<MeshCollider const*>(&meshCollider),
      params,
      collidingFromCollider,
      outIndices,
      outContacts,
      outSdf,
      outIsSdfGradUnitary);
}

// Note: outResult is in collider's frame.
template <TimeStep kTimeStep>
static void QuerySdf(
    entt::registry const& reg,
    entt::entity collider,
    entt::entity /* colliding */,
    AnyShape const& colliderBoundsInCollider,
    TransformRT const& worldFromColliding,
    TransformRT const& worldFromCollider,
    Span<Real3 const> positionsInColliding,
    Span<int const> /* indicesToQuery */,
    ContactDetectionParams const& params,
    Span<int const> /* colliderFeatureIndices */,
    DynamicArray<int>& outIndices,
    DynamicArray<int>* /* outColliderFeatureIndices */,
    DynamicArray<Real3>& outContacts,
    SdfInfo& outSdf,
    bool& outIsSdfGradUnitary,
    int* outNDofs,
    DynamicArray<real>* /* outColliderIntegrationWeights */,
    DynamicArray<VMatrix3x3r>* outMapJac,
    DynamicArray<ColliderJacDofs>* outDofsJac) {
  // The SDF frame may be different than the Collider frame, e.g. they may differ by the rigid
  // transform for ROM colliders.
  TransformRT worldFromSdf = worldFromCollider;
  if (reg.all_of<TagRomActor>(collider)) {
    if (auto rigidTransform = rom::GetRigidTransform<kTimeStep>(reg, collider)) {
      worldFromSdf *= rigidTransform.value();
    }
  }
  auto collidingFromSdf = Invert(worldFromColliding) * worldFromSdf;

  // Get Sdf collider
  auto const* sdf = reg.get<CSdfCollider>(collider).shape.get();

  // Check if there's a mapping
  auto const* map = static_cast<std::unique_ptr<BaseMap> const*>(
      reg.try_get<CSdfMapping<kTimeStep> const>(collider));
  if (map) {
    // Transform collider bounds to SDF space.
    auto sdfFromCollider = Invert(worldFromSdf) * worldFromCollider;
    auto colliderBoundsInSdf = TransformShape(sdfFromCollider, colliderBoundsInCollider);

    // Create a MappedSdfCollider
    MappedSdfCollider mappedCollider{.shape = sdf, .mapping = map->get()};

    // Run the query
    FindPointContactsMapped(
        positionsInColliding,
        &mappedCollider,
        params,
        collidingFromSdf,
        colliderBoundsInSdf,
        outIndices,
        outContacts,
        &outSdf,
        outIsSdfGradUnitary,
        outNDofs,
        outMapJac,
        outDofsJac);
  } else {
    // Run the query with a regular SDF.
    auto const* gridSdf = dynamic_cast<GridSdf const*>(sdf);
    MOCHI_ASSERT(gridSdf, "Only grid SDF is supported.");
    FindPointContactsT(
        positionsInColliding,
        gridSdf,
        params,
        collidingFromSdf,
        outIndices,
        outContacts,
        outSdf,
        outIsSdfGradUnitary);
  }
}

template <TimeStep kTimeStep>
static void QueryPointCloud(
    entt::registry const& reg,
    entt::entity collider,
    entt::entity colliding,
    AnyShape const& /* colliderBoundsInCollider */,
    TransformRT const& worldFromColliding,
    TransformRT const& worldFromCollider,
    Span<Real3 const> positionsInColliding,
    Span<int const> indicesToQuery,
    ContactDetectionParams const& /* params */,
    Span<int const> colliderFeatureIndices,
    DynamicArray<int>& outIndices,
    DynamicArray<int>* outColliderFeatureIndices,
    DynamicArray<Real3>& outContacts,
    SdfInfo& outSdf,
    bool& outIsSdfGradUnitary,
    int* outNDofs,
    DynamicArray<real>* outColliderIntegrationWeights,
    DynamicArray<VMatrix3x3r>* outMapJac,
    DynamicArray<ColliderJacDofs>* outDofsJac) {
  MOCHI_ASSERT(
      kTimeStep == TimeStep::Current || kTimeStep == TimeStep::StageStart,
      "QueryPointCloud assumes TimeStep::Current or TimeStep::StageStart");
  MOCHI_ASSERT_VERBOSE(
      colliderFeatureIndices.empty() == (outColliderFeatureIndices != nullptr),
      "Provide or request collider features, but not both or neither.");
  MOCHI_ASSERT_VERBOSE(
      reg.all_of<TagUsePointCloudContact>(collider),
      "QueryPointCloud must only be called for point-cloud colliders.");

  // SDF gradients are explicitly normalized for point-cloud colliders.
  outIsSdfGradUnitary = true;

  bool const selfContact = (colliding == collider);

  // Get collider-side data from registry.
  auto const& colliderDiscretization = reg.get<CColliderPointCloudDiscretization>(collider);
  ColumnVectorView<real const> colliderDisplacements =
      reg.get<CFinalDisplacementRef<kTimeStep>>(collider).value;
  CPointCloudColliderParams const& params = reg.get<CPointCloudColliderParams>(collider);
  ContactParams const& contactParams = reg.get<CContactParams const>(collider);

  if (colliderFeatureIndices.empty()) {
    SpatialHashTable const& colliderSpatialHashTable = reg.get<CSpatialHashTable>(collider);

    // The colliding-side discretization provides access to precomputed reference-configuration
    // positions of samples, used for self-contact exclusion.
    CollidingPointCloudDiscretization collidingDiscretization = [&] {
      if (auto const* surfDisc = reg.try_get<CFemSurfaceDiscretization>(collider)) {
        return CollidingPointCloudDiscretization{surfDisc};
      }
      auto const* segDisc = reg.try_get<CFemSegmentDiscretization>(collider);
      MOCHI_ASSERT_VERBOSE(
          segDisc != nullptr,
          "No surface or segment discretization found for point cloud collider.");
      return CollidingPointCloudDiscretization{segDisc};
    }();

    // Compute sample-to-collider-point connectivity (query collider hash table at colliding sample
    // positions)
    DynamicArray<DynamicArray<int>> samplesToColliderPoints = ComputePointsToColliderPoints(
        params,
        colliderDiscretization,
        colliderDisplacements,
        worldFromCollider,
        collidingDiscretization,
        selfContact,
        positionsInColliding,
        indicesToQuery,
        worldFromColliding,
        colliderSpatialHashTable,
        contactParams.GetPenaltyThresholdDist(true));

    // Compute sampleIndices and colliderFeatureIndices from the collision detection output.
    ComputePointCloudContactIndices(
        samplesToColliderPoints, outIndices, *outColliderFeatureIndices);
  } else {
    // Skip the secondary culling based on the spatial hash table, fill in outIndices with an
    // identity mapping, and use colliderFeatureIndices as input (computed in a previous pass).
    MOCHI_ASSERT_VERBOSE(
        isize(positionsInColliding) == isize(colliderFeatureIndices),
        "Size mismatch between colliding samples and collider points.");
    outIndices.resize_noinit(positionsInColliding.size());
    std::iota(outIndices.begin(), outIndices.end(), 0);
  }

  // Populate the remaining ContactDetectionResult fields using the computed indices.
  // Pass the appropriate colliderFeatureIndices: if output was computed, use it; otherwise use
  // input.
  Span<int const> colliderPointIndicesToUse = outColliderFeatureIndices != nullptr
      ? MakeConstSpan(*outColliderFeatureIndices)
      : colliderFeatureIndices;
  ComputePointCloudContactDetectionFields(
      colliderDiscretization,
      colliderDisplacements,
      worldFromCollider,
      positionsInColliding,
      worldFromColliding,
      outIndices,
      colliderPointIndicesToUse,
      params.radius,
      outContacts,
      &outSdf,
      outNDofs,
      outColliderIntegrationWeights,
      outMapJac,
      outDofsJac);
}

/*
  Get the query function pointer for full collision detection (points and distances)
*/
template <TimeStep kTimeStep>
static auto GetQueryFuncForPointsAndSdf(entt::registry const& reg, entt::entity collider) {
  auto const colliderType = reg.get<CColliderInfo const>(collider).type;
  auto queryFunc = &QueryBasicShape<CBoxCollider>;
  switch (colliderType) {
    case ColliderType::Sphere: {
      queryFunc = &QueryBasicShape<CSphereCollider>;
    } break;
    case ColliderType::Plane: {
      queryFunc = &QueryBasicShape<CPlaneCollider>;
    } break;
    case ColliderType::Mesh: {
      queryFunc = &QueryMesh;
    } break;
    case ColliderType::Sdf: {
      queryFunc = &QuerySdf<kTimeStep>;
    } break;
    case ColliderType::PointCloud: {
      queryFunc = &QueryPointCloud<kTimeStep>;
    } break;
    case ColliderType::Auto: {
      MOCHI_ASSERT(false, "ColliderType::Auto should have been resolved before reaching here.");
    } break;
    case ColliderType::None: {
      MOCHI_ASSERT(false, "Expected a collider.");
    } break;
    default:
      break;
  }
  static_assert(
      static_cast<int>(ColliderType::Count) == 8,
      "Please update the switch statement above if ColliderType enum changes");

  return queryFunc;
}

static void TransformToCollider(
    entt::registry const& /* reg */,
    entt::entity /* collider */,
    TransformRT const& worldFromColliding,
    TransformRT const& worldFromCollider,
    Span<Real3 const> positionsInColliding,
    Span<int const> /* colliderFeatureIndices */,
    DynamicArray<int>& outIndices,
    DynamicArray<Real3>& outContacts) {
  outIndices.resize_noinit(positionsInColliding.size());
  std::iota(outIndices.begin(), outIndices.end(), 0);
  outContacts.resize_noinit(positionsInColliding.size());
  auto const colliderFromColliding = Invert(worldFromCollider) * worldFromColliding;
  ArrayTransformPoints(MakeSpan(outContacts), positionsInColliding, colliderFromColliding);
}

template <TimeStep kTimeStep>
static void MapToCollider(
    entt::registry const& reg,
    entt::entity collider,
    TransformRT const& worldFromColliding,
    TransformRT const& worldFromCollider,
    Span<Real3 const> positionsInColliding,
    Span<int const> /* colliderFeatureIndices */,
    DynamicArray<int>& outIndices,
    DynamicArray<Real3>& outContacts) {
  // The SDF frame may be different than the Collider frame, e.g. they may differ by the rigid
  // transform for ROM colliders.
  TransformRT worldFromSdf = worldFromCollider;
  if (reg.all_of<TagRomActor>(collider)) {
    if (auto rigidTransform = rom::GetRigidTransform<kTimeStep>(reg, collider)) {
      worldFromSdf *= rigidTransform.value();
    }
  }
  auto collidingFromSdf = Invert(worldFromColliding) * worldFromSdf;

  // Get collider shape and mapping
  auto const& sdfCollider = reg.get<CSdfCollider>(collider);
  auto const& map = reg.get<CSdfMapping<kTimeStep> const>(collider);

  // Create a MappedSdfCollider
  MappedSdfCollider mappedCollider{.shape = sdfCollider.shape.get(), .mapping = map.get()};

  // Set infinite tolerance and collider bounds.
  ContactDetectionParams detectionParams;
  detectionParams.tolerance = kInf;
  Sphere colliderBounds(Real3{}, kInf);

  // Run the query
  bool isSdfGradUnitary{}; // unused
  FindPointContactsMapped(
      positionsInColliding,
      &mappedCollider,
      detectionParams,
      collidingFromSdf,
      colliderBounds,
      outIndices,
      outContacts,
      /*outSdf*/ nullptr,
      isSdfGradUnitary,
      /*outNDofs*/ nullptr,
      /*outMapJac*/ nullptr,
      /*outDofsJac*/ nullptr);
}

// Points-only query for point-cloud colliders. This is a lightweight wrapper that extracts collider
// data from the registry and delegates to ComputePointCloudContactDetectionFields with null
// optional outputs.
template <TimeStep kTimeStep>
static void QueryPointCloudPointsOnly(
    entt::registry const& reg,
    entt::entity collider,
    TransformRT const& worldFromColliding,
    TransformRT const& worldFromCollider,
    Span<Real3 const> positionsInColliding,
    Span<int const> colliderFeatureIndices,
    DynamicArray<int>& outIndices,
    DynamicArray<Real3>& outPosColliding) {
  int const totalContacts = isize(colliderFeatureIndices);
  MOCHI_ASSERT_VERBOSE(
      isize(positionsInColliding) == totalContacts,
      "Size mismatch between colliding samples and collider nodes.");
  if (totalContacts == 0) {
    return;
  }

  // Get collider-side data from registry.
  auto const& colliderDiscretization = reg.get<CColliderPointCloudDiscretization>(collider);
  ColumnVectorView<real const> colliderDisplacements =
      reg.get<CFinalDisplacementRef<kTimeStep>>(collider).value;

  // Populate outIndices with identity mapping.
  outIndices.resize_noinit(totalContacts);
  std::iota(outIndices.begin(), outIndices.end(), 0);

  // Call ComputePointCloudContactDetectionFields with null optional outputs.
  ComputePointCloudContactDetectionFields(
      colliderDiscretization,
      colliderDisplacements,
      worldFromCollider,
      positionsInColliding,
      worldFromColliding,
      outIndices,
      colliderFeatureIndices,
      /*radius=*/0_r,
      outPosColliding,
      /*outSdfInfo=*/nullptr,
      /*outNdofs=*/nullptr,
      /*outColliderIntegrationWeights=*/nullptr,
      /*outJacColliderFromWorld=*/nullptr,
      /*outJacWorldFromDofs=*/nullptr);
}

/*
  Get the query function pointer for partial collision detection (points only)
*/
template <TimeStep kTimeStep>
static auto GetQueryFuncForPointsOnly(entt::registry const& reg, entt::entity collider) {
  auto const colliderType = reg.get<CColliderInfo const>(collider).type;
  // MappedSdfCollider requires a special function. Point-cloud colliders use their own query.
  // Other colliders just use a transform function.
  if (colliderType == ColliderType::PointCloud) {
    return &QueryPointCloudPointsOnly<kTimeStep>;
  }
  auto const* map = reg.try_get<CSdfMapping<kTimeStep> const>(collider);
  return map ? &MapToCollider<kTimeStep> : &TransformToCollider;
}

/*
  Given resultPerCollider[0] that contains the aggregated contact detection results of all
  partitions, distribute them to the appropriate partitions.
  The function is templatized with TimeStep to distinguish behavior with TimeStep::Current and
  TimeStep::StageStart. In TimeStep::Current, both current and stage-start fields are populated. In
  TimeStep::StageStart, only current fields are populated (it's not necessary to populate the
  stage-start fields, because their content would be duplicated).
*/
template <TimeStep kTimeStep>
static void DistributeContactDetectionResultToPartitions(
    bool explicitNormals,
    Span<ContactDetectionResult*> resultPerCollider,
    CContactPartitions const& collisionPartitions) {
  static_assert(kTimeStep == TimeStep::Current || kTimeStep == TimeStep::StageStart);

  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(resultPerCollider.size() == collisionPartitions.size(), "Inconsistent sizes.");
  if (resultPerCollider.size() <= 1) {
    return;
  }

  // Use the first partition's result as the source.
  ContactDetectionResult& sourceResult = *resultPerCollider[0];

  // Set up data that is shared by all partitions
  MOCHI_ASSERT_VERBOSE(
      sourceResult.jacColliderFromWorld.size() == 1, "Deformable colliders are not supported");
  MOCHI_ASSERT_VERBOSE(
      sourceResult.jacColliderFromWorldStageStart.empty() ||
          (explicitNormals && (sourceResult.jacColliderFromWorldStageStart.size() == 1)),
      "Unexpected size");
  for (int p = 1; p < isize(resultPerCollider); p++) {
    resultPerCollider[p]->isSdfGradUnitary = sourceResult.isSdfGradUnitary;
    resultPerCollider[p]->jacColliderFromWorld.resize_noinit(1);
    resultPerCollider[p]->jacColliderFromWorld[0] = sourceResult.jacColliderFromWorld[0];
    if (kTimeStep == TimeStep::Current && explicitNormals) {
      resultPerCollider[p]->jacColliderFromWorldStageStart.resize_noinit(1);
      resultPerCollider[p]->jacColliderFromWorldStageStart[0] =
          sourceResult.jacColliderFromWorldStageStart[0];
    }
  }

  // Distribute results to their respective partitions.
  int firstPartitionCount = 0; // How many points belong to the first partition
  for (int i = 0; i < isize(sourceResult.sampleIndices); ++i) {
    int const sampleIdx = sourceResult.sampleIndices[i];
    int const partitionIdx = collisionPartitions.SampleIdxToPartitionIdx(sampleIdx);

    if (partitionIdx == 0) {
      // Move the point to the consolidated section at the beginning.
      if (i != firstPartitionCount) {
        sourceResult.sampleIndices[firstPartitionCount] = sampleIdx;

        // TODO: Copy arrays instead of individual points
        sourceResult.posColliding[firstPartitionCount] = sourceResult.posColliding[i];
        sourceResult.sdfInfo.grad[firstPartitionCount] = sourceResult.sdfInfo.grad[i];
        sourceResult.sdfInfo.val[firstPartitionCount] = sourceResult.sdfInfo.val[i];
        if constexpr (kTimeStep == TimeStep::Current) {
          sourceResult.posCollidingStageStart[firstPartitionCount] =
              sourceResult.posCollidingStageStart[i];
          if (explicitNormals) {
            sourceResult.sdfInfoStageStart.grad[firstPartitionCount] =
                sourceResult.sdfInfoStageStart.grad[i];
            sourceResult.sdfInfoStageStart.val[firstPartitionCount] =
                sourceResult.sdfInfoStageStart.val[i];
          }
        }
      }
      firstPartitionCount++;
    } else {
      auto& targetPartition = *resultPerCollider[partitionIdx];
      targetPartition.sampleIndices.push_back(sampleIdx);

      // TODO: Copy arrays instead of individual points
      targetPartition.posColliding.push_back(sourceResult.posColliding[i]);
      targetPartition.sdfInfo.grad.push_back(sourceResult.sdfInfo.grad[i]);
      targetPartition.sdfInfo.val.push_back(sourceResult.sdfInfo.val[i]);
      if constexpr (kTimeStep == TimeStep::Current) {
        targetPartition.posCollidingStageStart.push_back(sourceResult.posCollidingStageStart[i]);
        if (explicitNormals) {
          targetPartition.sdfInfoStageStart.push_back(
              sourceResult.sdfInfoStageStart.val[i], sourceResult.sdfInfoStageStart.grad[i]);
        }
      }
    }
  }

  sourceResult.sampleIndices.resize(firstPartitionCount);
  sourceResult.posColliding.resize(firstPartitionCount);
  sourceResult.sdfInfo.resize(firstPartitionCount);
  if constexpr (kTimeStep == TimeStep::Current) {
    sourceResult.posCollidingStageStart.resize(firstPartitionCount);
    if (explicitNormals) {
      sourceResult.sdfInfoStageStart.resize(firstPartitionCount);
    }
  }
}

/**
 * @brief Filters sample points that are potentially in contact to reduce collision detection
 * workload.
 *
 * @param reg
 * @param colliding
 * @param collider
 * @param contactSamples
 * @param collidingBounds
 * @param colliderBounds
 * @param collidingFromCollider
 * @param culledPositionsBuffer Output buffer for filtered positions (only populated if culling
 *                              occurs)
 * @param culledIndicesBuffer   Output buffer for filtered indices (only populated if culling
 *                              occurs)
 *
 * @return A pair containing:
 *   - Span of positions to query for collision detection
 *   - Span of indices mapping these points to sample point indices (empty means identity mapping)
 *
 * @note The returned spans may point to:
 *   - culledPositionsBuffer/culledIndicesBuffer (if culling is performed)
 *   - contactSamples.activePositions/activeIndices (with subsampling but no culling)
 *   - contactSamples.positions (neither subsampling nor culling)
 */
template <TimeStep kTimeStep, bool kAllowFarSdfQuery>
[[nodiscard]] static std::pair<Span<Real3 const>, Span<int const>> FilterSamplePointsForCollision(
    entt::registry const& reg,
    entt::entity colliding,
    entt::entity collider,
    ContactSamples const& contactSamples,
    AnyShape const& collidingBounds,
    AnyShape const& colliderBounds,
    TransformRT const& collidingFromCollider,
    DynamicArray<Real3>& culledPositionsBuffer,
    DynamicArray<int>& culledIndicesBuffer) {
  MOCHI_PROFILE_SCOPE();
  culledPositionsBuffer.clear();
  culledIndicesBuffer.clear();

  real const farSdfDistance = GetFarSdfEvaluationDistance<kAllowFarSdfQuery>(reg, colliding);
  AnyShape expandedColliderBoundsInCollidingSpace =
      ExpandShape(TransformShape(collidingFromCollider, colliderBounds), farSdfDistance);
  if (!HasOverlap(collidingBounds, expandedColliderBoundsInCollidingSpace)) {
    return {{}, {}};
  }

  auto const* activeBoundaryFaces = reg.try_get<CActiveBoundaryFaces const>(colliding);
  if (activeBoundaryFaces) {
    [[maybe_unused]] auto const numActiveSamples =
        activeBoundaryFaces->ViewIndices().size() * activeBoundaryFaces->NumQuadPerFace();
    MOCHI_ASSERT(
        (contactSamples.activePositions.size() == numActiveSamples) &&
            (contactSamples.activeIndices.size() == numActiveSamples),
        "Active sample points must have been computed before performing collision detection.");
  }

  Span<Real3 const> outPositionsToQuery = activeBoundaryFaces
      ? MakeConstSpan(contactSamples.activePositions)
      : MakeConstSpan(contactSamples.positions);
  Span<int const> outIndices =
      activeBoundaryFaces ? MakeConstSpan(contactSamples.activeIndices) : Span<int const>{};

  // Cull sample points by checking the BSH tree of the (active) sample points against the
  // collider's bounding volume. For now, this is the only culling strategy. Other strategies may be
  // introduced in the future.
  if (contactSamples.bsh && IsFinite(farSdfDistance)) {
    MOCHI_ASSERT(
        outPositionsToQuery.size() == contactSamples.bsh->NumSamplePoints(),
        "Inconsistent number of sample points. Is the BVH tree up-to-date with the number of active sample points?");

    AnyBoundingVolume colliderBvForCulling;
    bool shouldCull = true;
    auto const colliderType = reg.get<CColliderInfo const>(collider).type;
    switch (colliderType) {
      case ColliderType::Box:
      case ColliderType::Sphere:
      case ColliderType::Plane:
      case ColliderType::PointCloud: {
        // For Box, Sphere, Plane, and PointCloud colliders, cull against their BV. Box and Sphere
        // colliders are second-class citizens and could be optimized further:
        // - The BV of Box colliders is often OBB (tighter than AABB but with more expensive
        //   HadOverlap overloads). Using the AABB that bounds the OBB may be a better tradeoff.
        // - If the colliding's BV largely overlaps the collider's BV, it may be more efficient to
        //   skip culling altogether.
        // PointCloud colliders will do a secondary pass of culling within the query, using their
        // spatial hash tables.
        colliderBvForCulling = ToAnyBoundingVolume(expandedColliderBoundsInCollidingSpace);
      } break;
      case ColliderType::Sdf: {
        // For SDF colliders, cull against their SDF level set.
        // TODO(T225595100): Support culling with mapping and/or rigid transform.
        shouldCull = !reg.any_of<CSdfMapping<kTimeStep>>(collider) &&
            !rom::GetRigidTransform<kTimeStep>(reg, collider);
        if (shouldCull) {
          auto const* gridSdf =
              assert_cast<GridSdf const*>(reg.get<CSdfCollider>(collider).shape.get());
          auto const gridFromCollidingT = Dot4x4(
              ToVMatrix4x4Transpose(Invert(collidingFromCollider)),
              gridSdf->GetGridFromActorTranspose());
          auto const& contactParams = reg.get<CContactParams const>(collider);
          colliderBvForCulling = SdfBv{
              .gridSdf = gridSdf,
              .distanceThreshold = GetColliderPadding(contactParams) + farSdfDistance,
              .gridFromPointsT = gridFromCollidingT};
        }
      } break;
      case ColliderType::Mesh: {
        // TODO(T225595100): Support culling against Mesh colliders.
        shouldCull = false;
      } break;
      case ColliderType::Auto: {
        MOCHI_ASSERT(false, "ColliderType::Auto should have been resolved before reaching here.");
      } break;
      case ColliderType::None: {
        MOCHI_ASSERT(false, "Expected a collider.");
      } break;
      default:
        break;
    }
    static_assert(
        static_cast<int>(ColliderType::Count) == 8,
        "Please update the switch statement above if ColliderType enum changes");

    // Perform culling.
    if (shouldCull) {
      culledIndicesBuffer.reserve(outPositionsToQuery.size());
      contactSamples.bsh->FindIntersectingSamples(colliderBvForCulling, culledIndicesBuffer);

      // Store culled positions and make 'outPositionsToQuery' point to them.
      culledPositionsBuffer.reserve(culledIndicesBuffer.size());
      for (int idx : culledIndicesBuffer) {
        culledPositionsBuffer.push_back(outPositionsToQuery[idx]);
      }

      // Remap indices.
      if (activeBoundaryFaces) {
        for (int& idx : culledIndicesBuffer) {
          MOCHI_ASSERT_VERBOSE(idx >= 0 && idx < contactSamples.activeIndices.size());
          idx = contactSamples.activeIndices[idx];
        }
      }

      outPositionsToQuery = MakeConstSpan(culledPositionsBuffer);
      outIndices = MakeConstSpan(culledIndicesBuffer);
    }
  }

  return {outPositionsToQuery, outIndices};
}

/**
 * @brief Performs collision detection between a colliding entity (collidee) and a collider entity.
 *
 * @tparam kTimeStepCollider   Time slice for collider data (Current, StageStart or Previous)
 * @tparam kAllowFarSdfQuery   Allow SDF queries beyond the normal contact culling distance
 * @param reg                  Registry
 * @param colliding            Colliding entity (collidee)
 * @param contactSamples       Sample points on the colliding entity's surface
 * @param collisionPartitions  Optional partitioning of the colliding entity's surface
 * @param collidingBounds      Bounding shape of the colliding entity
 * @param worldFromColliding   Transform from colliding entity's local space to world space
 * @param colliderData         Data for the specific collider to check against
 * @param outResult            Output for collision detection results (one for all the partitions)
 *
 * @note If collision partitions are specified, the results are distributed across the partitions
 * based on which partition each sample point belongs to.
 */
template <ContactType kContactType, TimeStep kTimeStep, bool kAllowFarSdfQuery>
static void DetectCollisionsWithSingleCollider(
    entt::registry const& reg,
    entt::entity colliding,
    ContactSamples const& contactSamples,
    AnyShape const& collidingBounds,
    TransformRT const& worldFromColliding,
    PotentialColliderData& colliderData,
    ContactDetectionResult& outResult) {
  MOCHI_PROFILE_SCOPE();

  entt::entity collider = colliderData.entity;

  // Contact must be sync for dynamic colliders and async for static colliders
  [[maybe_unused]] bool constexpr kIsSync = kContactType == ContactType::Sync;
  MOCHI_ASSERT_VERBOSE(
      kIsSync != reg.all_of<TagStaticActor>(collider), "Wrong contact type for this collider");

  auto const& colliderContactParams = reg.get<CContactParams const>(collider);
  auto const& worldFromCollider = GetRootTransform<kTimeStep>(reg, collider);
  auto const& colliderBounds =
      reg.get<CBoundingVolumeFor<kContactType, kTimeStep> const>(collider).localShape;
  AnyShape expandedColliderBounds =
      ExpandColliderBoundsForContact(colliderBounds, colliderContactParams);
  auto collidingFromCollider = Invert(worldFromColliding) * worldFromCollider;

  // List of sample points to be tested.
  auto [positionsToQuery, indicesToQuery] =
      FilterSamplePointsForCollision<kTimeStep, kAllowFarSdfQuery>(
          reg,
          colliding,
          collider,
          contactSamples,
          collidingBounds,
          expandedColliderBounds,
          collidingFromCollider,
          outResult.culledPositionsBuffer,
          outResult.culledIndicesBuffer);

  if (positionsToQuery.empty()) {
    return;
  }

  // Set tolerance depending on penalty threshold distance. Add extra padding (if any) if this actor
  // has no collider or a point-cloud collider.
  ContactDetectionParams detectionParams;
  bool const addPadding = ShouldAddPenaltyPadding(reg.get<CColliderInfo const>(colliding).type);
  real const farSdfDistance = GetFarSdfEvaluationDistance<kAllowFarSdfQuery>(reg, colliding);
  detectionParams.tolerance =
      colliderContactParams.GetPenaltyThresholdDist(addPadding) + farSdfDistance;

  // Initialize outResult.jacColliderFromWorld assuming a rigid collider. This will be overwritten
  // for deformable colliders.
  outResult.jacColliderFromWorld.resize_noinit(1);
  outResult.jacColliderFromWorld[0] = ToVMatrix3x3Transpose(worldFromCollider.GetRotation());

  // Run the collision query.
  auto queryFunc = GetQueryFuncForPointsAndSdf<kTimeStep>(reg, collider);
  queryFunc(
      reg,
      collider,
      colliding,
      expandedColliderBounds,
      worldFromColliding,
      worldFromCollider,
      positionsToQuery,
      indicesToQuery,
      detectionParams,
      /*colliderFeatureIndices=*/{},
      outResult.sampleIndices,
      &outResult.colliderFeatureIndices,
      outResult.posColliding,
      outResult.sdfInfo,
      outResult.isSdfGradUnitary,
      &outResult.ndofs,
      &outResult.colliderIntegrationWeights,
      &outResult.jacColliderFromWorld,
      &outResult.jacWorldFromDofs);

  // If the collider is a ROM with a rigid transform layer, add the rigid Jacobians.
  if (!outResult.sampleIndices.empty() && /*isContactMapped*/ (outResult.ndofs > 0) &&
      reg.all_of<TagRomActor>(collider)) {
    rom::AddRigidContactJacobians(reg, collider, outResult);
  }

  // Reindex points in contact to sample point indices.
  if (!indicesToQuery.empty()) {
    for (int& idx : outResult.sampleIndices) {
      idx = indicesToQuery[idx];
    }
  }
}

static void EvalStageStartContactWithSingleCollider(
    entt::registry const& reg,
    entt::entity colliding,
    bool explicitNormals,
    ContactSamples const& contactSamples,
    TransformRT const& worldFromColliding,
    PotentialColliderData& colliderData,
    ContactDetectionResult& outResult) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(!outResult.sampleIndices.empty(), "Expected at least one collision.");

  // Get the transform of the collider.
  entt::entity collider = colliderData.entity;
  auto const& worldFromCollider = GetRootTransform<TimeStep::StageStart>(reg, collider);

  // List of sample points to be tested.
  auto& positionsInColliding = outResult.culledPositionsBuffer;
  positionsInColliding.clear();
  positionsInColliding.reserve(outResult.sampleIndices.size());
  for (auto const& i : outResult.sampleIndices) {
    positionsInColliding.push_back(contactSamples.positions[i]);
  }

  // Output container for indices. Do not pass outResult.sampleIndices, because the query expects
  // an empty container.
  auto& outIndices = outResult.culledIndicesBuffer;
  outIndices.clear();

  // If explicitNormals, evaluate contact points, SDF and Jacobians; otherwise evaluate contact
  // points only.
  if (explicitNormals) {
    // Initialize outResult.jacColliderFromWorldStageStart assuming a rigid collider. This will be
    // overwritten for deformable colliders.
    outResult.jacColliderFromWorldStageStart.resize_noinit(1);
    outResult.jacColliderFromWorldStageStart[0] =
        ToVMatrix3x3Transpose(worldFromCollider.GetRotation());

    // Set infinite tolerance and collider bounds.
    ContactDetectionParams detectionParams;
    detectionParams.tolerance = kInf;
    Sphere colliderBounds(Real3{}, kInf);
    bool isSdfGradUnitary{};
    auto queryFunc = GetQueryFuncForPointsAndSdf<TimeStep::StageStart>(reg, collider);
    queryFunc(
        reg,
        collider,
        colliding,
        colliderBounds,
        worldFromColliding,
        worldFromCollider,
        positionsInColliding,
        outResult.sampleIndices,
        detectionParams,
        outResult.colliderFeatureIndices,
        outIndices,
        /*outColliderFeatureIndices=*/nullptr,
        outResult.posCollidingStageStart,
        outResult.sdfInfoStageStart,
        isSdfGradUnitary,
        nullptr, /* outNdofs: registered in the detection pass, no need to update */
        nullptr, /* outColliderIntegrationWeights: registered in the detection pass */
        &outResult.jacColliderFromWorldStageStart,
        nullptr /* outDofsJac: registered in the detection pass, no need to update */);

    MOCHI_ASSERT_VERBOSE(
        isSdfGradUnitary == outResult.isSdfGradUnitary, "Inconsistent isSdfGradUnitary flags");
  } else {
    auto queryFunc = GetQueryFuncForPointsOnly<TimeStep::StageStart>(reg, collider);
    queryFunc(
        reg,
        collider,
        worldFromColliding,
        worldFromCollider,
        positionsInColliding,
        outResult.colliderFeatureIndices,
        outIndices,
        outResult.posCollidingStageStart);
  }

  // If needed, fill any missing results.
  if (outIndices.size() != outResult.sampleIndices.size()) {
    MOCHI_ASSERT_VERBOSE(outResult.ndofs > 0, "Only mapped colliders may miss results");
    MOCHI_ASSERT_VERBOSE(
        outResult.posCollidingStageStart.size() == outIndices.size(),
        "Positions and indices size mismatch");
    MOCHI_ASSERT_VERBOSE(
        !explicitNormals || (outResult.sdfInfoStageStart.val.size() == outIndices.size()),
        "Distances and indices size mismatch");
    MOCHI_ASSERT_VERBOSE(
        !explicitNormals || (outResult.sdfInfoStageStart.grad.size() == outIndices.size()),
        "Gradients and indices size mismatch");
    MOCHI_ASSERT_VERBOSE(
        !explicitNormals || (outResult.jacColliderFromWorldStageStart.size() == outIndices.size()),
        "Jacobians and indices size mismatch");
    // Reserve stack memory for 128 contacts involving position, distance, gradient and Jacobian.
    MOCHI_FILO_STACK_ALLOCATOR(
        allocator, 128 * (sizeof(real) + 2 * sizeof(Real3) + sizeof(VMatrix3x3r)));
    DynamicArray<Real3> posCollidingTemp(&allocator);
    DynamicArray<real> sdfValTemp(&allocator);
    DynamicArray<Real3> sdfGradTemp(&allocator);
    DynamicArray<VMatrix3x3r> jacColliderFromWorldTemp(&allocator);
    posCollidingTemp.resize_noinit(outResult.sampleIndices.size());
    if (explicitNormals) {
      sdfValTemp.resize_noinit(outResult.sampleIndices.size());
      sdfGradTemp.resize_noinit(outResult.sampleIndices.size());
      jacColliderFromWorldTemp.resize_noinit(outResult.sampleIndices.size());
    }
    for (int i = 0, j = 0; i < outResult.sampleIndices.size(); ++i) {
      if (j < isize(outIndices) && outIndices[j] == i) {
        // This contact is in the output data. Use it.
        posCollidingTemp[i] = outResult.posCollidingStageStart[j];
        if (explicitNormals) {
          sdfValTemp[i] = outResult.sdfInfoStageStart.val[j];
          sdfGradTemp[i] = outResult.sdfInfoStageStart.grad[j];
          jacColliderFromWorldTemp[i] = outResult.jacColliderFromWorldStageStart[j];
        }
        j++;
      } else {
        // This contact is missing. Use infinite distance and cached data, hence friction is
        // cancelled and positions and normals are the same as those at current time.
        posCollidingTemp[i] = outResult.posColliding[i];
        if (explicitNormals) {
          sdfValTemp[i] = kInf;
          sdfGradTemp[i] = outResult.sdfInfo.grad[i];
          jacColliderFromWorldTemp[i] = outResult.jacColliderFromWorld[i];
        }
      }
    }
    outResult.posCollidingStageStart = posCollidingTemp;
    if (explicitNormals) {
      outResult.sdfInfoStageStart.val = sdfValTemp;
      outResult.sdfInfoStageStart.grad = sdfGradTemp;
      outResult.jacColliderFromWorldStageStart = jacColliderFromWorldTemp;
    }
  }
  outResult.culledIndicesBuffer.clear(); // = outIndices
  outResult.culledPositionsBuffer.clear(); // = positionsInColliding
}

/**
 * @brief Finds sample points of an actor that are in contact with a collection of potential
 * colliders. For each potential collider, it performs collision detection in parallel and stores
 * the results in the output active collisions structure.
 *
 * @param reg                  Registry
 * @param colliding            Colliding entity (collidee)
 * @param boundingVolume       Bounding volume of the colliding entity
 * @param collisionSamples     Sample points on the colliding entity's surface
 * @param collisionPartitions  Optional partitioning of the colliding entity's surface
 * @param potentialColliders   List of potential colliders to check against
 * @param outActiveCollisions  Output for active collision results
 */
template <ContactType kContactType, TimeStep kTimeStep, bool kAllowFarSdfQuery = false>
static void DetectCollisionsWithPotentialColliders(
    entt::registry const& reg,
    entt::entity colliding,
    CBoundingVolume<TimeStep::Current> const& boundingVolume,
    CContactSamples<TimeStep::Current> const& samplesCurrent,
    CContactSamples<TimeStep::StageStart> const* samplesStageStartDeformable,
    CContactPartitions const* collisionPartitions,
    CPotentialColliders<kContactType>& potentialColliders,
    CActiveCollisions<kContactType, kTimeStep>& outActiveCollisions) {
  static_assert(
      kContactType == ContactType::Sync || kContactType == ContactType::Async,
      "Invalid contact type");
  MOCHI_PROFILE_SCOPE();

  int const numPartitions = collisionPartitions ? isize(*collisionPartitions) : 1;

  // Set up memory for collision points.
  outActiveCollisions.SetUp(potentialColliders, numPartitions);

  if (potentialColliders.empty()) {
    return;
  }

  // Prepare results for all colliders and all partitions, to allow parallelization.
  int const numPotentialColliders = isize(potentialColliders);
  MOCHI_FILO_STACK_ALLOCATOR(filoAllocator, 4096);

  DynamicArray<DynamicArray<ContactDetectionResult*>> resultPerCollider(&filoAllocator);
  resultPerCollider.reserve(numPotentialColliders);

  for (int c = 0; c < numPotentialColliders; ++c) {
    // Find active collisions corresponding to this potential collider.
    int idx = 0;
    for (; idx < isize(outActiveCollisions); idx += numPartitions) {
      if (outActiveCollisions[idx].colliderEntity == potentialColliders[c].entity) {
        break;
      }
    }
    MOCHI_ASSERT_VERBOSE(idx < isize(outActiveCollisions));

    resultPerCollider.emplace_back(&filoAllocator);
    auto& resultPerPartition = resultPerCollider.back();
    resultPerPartition.reserve(numPartitions);
    for (int p = 0; p < numPartitions; p++) {
      MOCHI_ASSERT_VERBOSE(
          outActiveCollisions[idx].colliderEntity == potentialColliders[c].entity,
          "Inconsistent collider entity.");
      resultPerPartition.push_back(&outActiveCollisions[idx++].collisionResult);
    }
  }

  MOCHI_ASSERT_VERBOSE(
      !samplesCurrent.positions.empty(),
      "Sample points should have been computed/updated BEFORE calling DetectCollisionsWithPotentialColliders.");
  if (samplesStageStartDeformable) {
    MOCHI_ASSERT_VERBOSE(
        !samplesStageStartDeformable->positions.empty(),
        "Sample points should have been computed/updated BEFORE calling DetectCollisionsWithPotentialColliders.");
  }

  // Get the transforms of the colliding actor.
  auto const& worldFromColliding = GetRootTransform<kTimeStep>(reg, colliding);
  [[maybe_unused]] auto const& worldFromCollidingStageStart =
      GetRootTransform<TimeStep::StageStart>(reg, colliding);

  // If the actor has a deforming surface, CContactSamples<TimeStep::StageStart> stores collision
  // sample positions at stage start. If the actor has a rigid surface,
  // CContactSamples<TimeStep::Current> stores the positions for all time slices.
  auto const& samplesTimeStep = (kTimeStep == TimeStep::Current || !samplesStageStartDeformable)
      ? static_cast<ContactSamples const&>(samplesCurrent)
      : static_cast<ContactSamples const&>(*samplesStageStartDeformable);
  [[maybe_unused]] auto const& samplesStageStart = samplesStageStartDeformable
      ? static_cast<ContactSamples const&>(*samplesStageStartDeformable)
      : static_cast<ContactSamples const&>(samplesCurrent);

  // If explicitNormals = true, we also need SDF distance and gradient evaluation at
  // TimeStep::StageStart.
  bool const explicitNormals = reg.ctx<CSimulationParams const>().experimentalEval.explicitNormals;

  ParallelForN("DetectCollisionsWithSingleCollider", numPotentialColliders, 1, [&](int c) {
    // Perform collision detection with this collider. Even if there are multiple partitions, write
    // the full result on the container for the first partition.
    DetectCollisionsWithSingleCollider<kContactType, kTimeStep, kAllowFarSdfQuery>(
        reg,
        colliding,
        samplesTimeStep,
        /*collidingBounds*/ boundingVolume.localShape,
        worldFromColliding,
        potentialColliders[c],
        *resultPerCollider[c][0]);
    if constexpr (kTimeStep == TimeStep::Current && !kAllowFarSdfQuery) {
      // For queries in TimeStep::Current and not kAllowFarSdfQuery, complete with data from
      // TimeStep::StageStart. Again, write the full result on the container for the first
      // partition.
      // WARNING: This code path cannot access CBoundingVolume<TimeStep::Current> or
      // CSpatialHashTable, because they are currently updated with TimeStep::Current, not
      // TimeStep::StageStart.
      if (!resultPerCollider[c][0]->sampleIndices.empty()) {
        EvalStageStartContactWithSingleCollider(
            reg,
            colliding,
            explicitNormals,
            samplesStageStart,
            worldFromCollidingStageStart,
            potentialColliders[c],
            *resultPerCollider[c][0]);
      }
    }
    // With multiple partitions, distribute the result to each partition's container.
    // TODO: Extend this function to handle contact between partitions and soft colliders.
    if (collisionPartitions && !resultPerCollider[c][0]->sampleIndices.empty()) {
      DistributeContactDetectionResultToPartitions<kTimeStep>(
          explicitNormals, resultPerCollider[c], *collisionPartitions);
    }
  });
}

template <ContactType kContactType, TimeStep kTimeStep, bool kAllowFarSdfQuery = false>
static void CollisionDetection(entt::registry& reg, entt::entity ent) {
  // Cull potential colliders using the updated Aabb and determine active contact components.
  ecs::TryInvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
      &UpdatePotentialColliders<kContactType, kTimeStep, kAllowFarSdfQuery>, reg, ent);

  // Perform collision detection (updates CActiveCollisions).
  ecs::TryInvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
      &DetectCollisionsWithPotentialColliders<kContactType, kTimeStep, kAllowFarSdfQuery>,
      reg,
      ent);
}

// Compute contact normals of a colliding actor
template <ContactType kContactType>
static void SetupActiveCollisionNormals(entt::registry& reg, entt::entity ent) {
  MOCHI_PROFILE_SCOPE();
  ecs::TryInvokeOnEntity(
      &deformable::SetupActiveCollisionNormals<kContactType, CFemBoundaryDiscretization>, reg, ent);
  ecs::TryInvokeOnEntity(
      &deformable::SetupActiveCollisionNormals<kContactType, CFemSurfaceDiscretization>, reg, ent);
  ecs::TryInvokeOnEntity(&rigid::SetupActiveCollisionNormals<kContactType>, reg, ent);
}

template <TimeStep kTimeStep>
static void InitCollidingJacobians(
    TaskSemaphore& sem,
    entt::registry& reg,
    CIslandDescendants const& descendants) {
  MOCHI_PROFILE_SCOPE();

  // Schedule all deforming actors in parallel
  ecs::ScheduleInvokeForEach(
      sem,
      "deformable::SetupCollidingJacobians<TagSoftActor, CFemBoundaryDiscretization>",
      &deformable::SetupCollidingJacobians<TagSoftActor, CFemBoundaryDiscretization>,
      reg,
      descendants.softActors);
  ecs::ScheduleInvokeForEach(
      sem,
      "deformable::SetupCollidingJacobians<TagShellActor, CFemSurfaceDiscretization>",
      &deformable::SetupCollidingJacobians<TagShellActor, CFemSurfaceDiscretization>,
      reg,
      descendants.shellActors);
  // Centerline Jacobians for rods without surface contact.
  ecs::ScheduleInvokeForEach(
      sem,
      "deformable::SetupCollidingJacobians<TagRodActor, CFemSegmentDiscretization>",
      &deformable::SetupCollidingJacobians<TagRodActor, CFemSegmentDiscretization>,
      reg,
      descendants.rodActors);
  // Jacobians for rods with surface contact.
  ecs::ScheduleInvokeForEach(
      sem,
      "rod::SetupSurfaceCollidingJacobians",
      &rod::SetupSurfaceCollidingJacobians,
      reg,
      descendants.rodActors);
  ecs::ScheduleInvokeForEach(
      sem,
      "rom::SetupCollidingJacobians",
      &rom::SetupCollidingJacobians,
      reg,
      descendants.softActors);
  ecs::ScheduleInvokeForEach(
      sem,
      "articulated::compound::SetupCollidingJacobians<CFemBoundaryDiscretization>",
      &articulated::compound::SetupCollidingJacobians<CFemBoundaryDiscretization>,
      reg,
      descendants.compoundActors);
  ecs::ScheduleInvokeForEach(
      sem,
      "articulated::compound::SetupCollidingJacobians<CFemSurfaceDiscretization>",
      &articulated::compound::SetupCollidingJacobians<CFemSurfaceDiscretization>,
      reg,
      descendants.compoundActors);
  ecs::ScheduleInvokeForEach(
      sem,
      "skinned::SetupCollidingJacobians",
      &skinned::SetupCollidingJacobians,
      reg,
      descendants.nestedSoftActors);
  ecs::ScheduleInvokeForEach(
      sem,
      "blended::SetupCollidingJacobians<CFemBoundaryDiscretization>",
      &blended::SetupCollidingJacobians<CFemBoundaryDiscretization>,
      reg,
      descendants.blendedActors);
  ecs::ScheduleInvokeForEach(
      sem,
      "blended::SetupCollidingJacobians<CFemSurfaceDiscretization>",
      &blended::SetupCollidingJacobians<CFemSurfaceDiscretization>,
      reg,
      descendants.blendedActors);

  // Schedule jointly all the rigid links in an articulated actor
  for (auto e : descendants.compoundActors) {
    if (reg.all_of<TagArticulatedActor>(e)) {
      Schedule(sem, "SetupCollidingJacobians_Articulated", [&reg, e]() {
        for (auto nested : reg.get<CGroupMembers const>(e).actors) {
          ecs::TryInvokeOnEntity(
              &articulated::rigid::SetupCollidingJacobians<kTimeStep>, reg, nested);
        }
      });
    }
  }

  // Rigid actors are pretty cheap. Do them on the calling thread.
  ecs::InvokeForEach(&rigid::SetupCollidingJacobians<kTimeStep>, reg, descendants.rigidActors);
}

// This system is only used for far SDF queries.
MOCHI_API void mochi::FarSdfCollisionDetection(
    ecs::Included<TagUseContact, CRequiresFarSdfEvaluation>,
    entt::registry& reg,
    entt::entity e) {
  // TODO: This system is only used for dynamic hyper-reduction. It computes the distance from each
  // sample point to all colliders, which is later used to compute the distance form each sample
  // point to the closest collider. Since only the distance to the closest collider is needed, it
  // could be optimized to skip computing the distance to colliders that are provable farther than
  // the closest collider computed so far. For BSH strategies, further optimizations may be possible
  // by exploiting already-computed distances from other sample points.
  constexpr bool kAllowFarSdfQuery = true;
  CollisionDetection<ContactType::Async, TimeStep::Current, kAllowFarSdfQuery>(reg, e);
  CollisionDetection<ContactType::Sync, TimeStep::Current, kAllowFarSdfQuery>(reg, e);
}

static void RegisterContactJacobians(entt::registry& reg, Span<entt::entity const> actors) {
  MOCHI_PROFILE_SCOPE();

  // At input, the contact Jacobians are from the previous assembly. Temporarily set the contact
  // type to None to signal that they may no longer be in contact.
  for (auto const& e : actors) {
    if (auto* collidingJacs = reg.try_get<CCollJacs<CollRole::Colliding>>(e)) {
      for (auto& collJac : *collidingJacs) {
        MOCHI_ASSERT_VERBOSE(collJac.type != ContactType::None, "Expected valid contact type.");
        collJac.type = ContactType::None;
        collJac.query = nullptr; // For safety
      }
    }
    if (auto* colliderJacs = reg.try_get<CCollJacs<CollRole::Collider>>(e)) {
      for (auto& collJac : *colliderJacs) {
        MOCHI_ASSERT_VERBOSE(collJac.type != ContactType::None, "Expected valid contact type.");
        collJac.type = ContactType::None;
        collJac.query = nullptr; // For safety
      }
    }
  }

  // Register the contact Jacobians for all the active collisions.
  for (auto const& e : actors) {
    auto* activeCollsAsync =
        reg.try_get<CActiveCollisions<ContactType::Async, TimeStep::Current>>(e);
    auto* activeCollsSync = reg.try_get<CActiveCollisions<ContactType::Sync, TimeStep::Current>>(e);
    if (!activeCollsAsync && !activeCollsSync) {
      continue; // No active collisions.
    }

    auto& collidingJacs = reg.get<CCollJacs<CollRole::Colliding>>(e);
    int const prevCollidingJacs = isize(collidingJacs);
    collidingJacs.reserve(
        prevCollidingJacs + (activeCollsAsync ? isize(*activeCollsAsync) : 0) +
        (activeCollsSync ? isize(*activeCollsSync) : 0));

    auto registerJacobians = [&](std::vector<ActiveCollision>& activeColls,
                                 ContactType const& contactType) {
      for (auto& activeColl : activeColls) {
        // Register colliding Jacobian (owned by this entity e).
        bool wasPreviouslyActive = false;
        for (int i = 0; i < prevCollidingJacs; ++i) {
          auto& collJac = collidingJacs[i];
          if ((collJac.otherEntity == activeColl.colliderEntity) &&
              (collJac.collidingPartitionId == activeColl.collisionResult.collidingPartitionId)) {
            wasPreviouslyActive = true;
            activeColl.collidingJacId = i;
            collJac.type = contactType;
            collJac.query = &activeColl.collisionResult;
            MOCHI_ASSERT_VERBOSE(
                collJac.bothRigid ==
                    (reg.all_of<TagRigidActor>(e) &&
                     reg.all_of<TagRigidActor>(activeColl.colliderEntity)),
                "Inconsistent 'bothRigid' flag.");
            break;
          }
        }
        if (!wasPreviouslyActive) {
          activeColl.collidingJacId = isize(collidingJacs);
          bool bothRigid =
              reg.all_of<TagRigidActor>(e) && reg.all_of<TagRigidActor>(activeColl.colliderEntity);
          collidingJacs.emplace_back(
              contactType,
              &activeColl.collisionResult,
              bothRigid,
              activeColl.colliderEntity,
              activeColl.collisionResult.collidingPartitionId);
        }
        if (contactType == ContactType::Sync) {
          // Register collider Jacobian (owned by the other entity).
          MOCHI_ASSERT_VERBOSE(
              Contains(actors, activeColl.colliderEntity), "Invalid collider entity.");
          auto& colliderJacs = reg.get<CCollJacs<CollRole::Collider>>(activeColl.colliderEntity);
          wasPreviouslyActive = false;
          for (int i = 0; i < isize(colliderJacs); ++i) {
            auto& collJac = colliderJacs[i];
            if ((collJac.otherEntity == e) &&
                (collJac.collidingPartitionId == activeColl.collisionResult.collidingPartitionId)) {
              // Recycle this JacData
              wasPreviouslyActive = true;
              activeColl.colliderJacId = i;
              collJac.type = contactType;
              collJac.query = &activeColl.collisionResult;
              MOCHI_ASSERT_VERBOSE(
                  collJac.bothRigid ==
                      (reg.all_of<TagRigidActor>(e) &&
                       reg.all_of<TagRigidActor>(activeColl.colliderEntity)),
                  "Inconsistent 'bothRigid' flag.");
              break;
            }
          }
          if (!wasPreviouslyActive) {
            activeColl.colliderJacId = isize(colliderJacs);
            bool bothRigid = reg.all_of<TagRigidActor>(e) &&
                reg.all_of<TagRigidActor>(activeColl.colliderEntity);
            colliderJacs.emplace_back(
                contactType,
                &activeColl.collisionResult,
                bothRigid,
                e,
                activeColl.collisionResult.collidingPartitionId);
          }
        }
      }
    };

    if (activeCollsAsync) {
      registerJacobians(*activeCollsAsync, ContactType::Async);
    }
    if (activeCollsSync) {
      registerJacobians(*activeCollsSync, ContactType::Sync);
    }
  }

  // Remove contact Jacobians that that were previously active (in contact) but are no longer
  // active.
  auto removeInactiveJacs = [](auto& jacs, auto& jacIdMapping) {
    jacIdMapping.resize(isize(jacs), -1);
    int writeIdx = 0;
    for (int readIdx = 0; readIdx < isize(jacs); ++readIdx) {
      if (jacs[readIdx].type != ContactType::None) {
        if (writeIdx != readIdx) {
          jacs[writeIdx] = std::move(jacs[readIdx]);
        }
        jacIdMapping[readIdx] = writeIdx;
        ++writeIdx;
      }
    }
    jacs.erase(jacs.begin() + writeIdx, jacs.end());
  };

  using JacIdArray = DynamicArray<int>;
  int const numActors = isize(actors);

  // Temporary storage
  MOCHI_FILO_STACK_ALLOCATOR(filoAllocator, 4096);

  // Mapping of entity -> JacIdArray
  DynamicArray<std::pair<entt::entity, JacIdArray>> colliderJacIdMappings(&filoAllocator);
  colliderJacIdMappings.reserve(numActors);

  for (int i = 0; i < numActors; ++i) {
    auto const e = actors[i];

    if (auto* colliderJacs = reg.try_get<CCollJacs<CollRole::Collider>>(e)) {
      colliderJacIdMappings.emplace_back(e, JacIdArray(&filoAllocator)); // Insert pair
      removeInactiveJacs(*colliderJacs, colliderJacIdMappings.back().second);
    }

    JacIdArray collidingJacIdMapping(&filoAllocator);

    if (auto* collidingJacs = reg.try_get<CCollJacs<CollRole::Colliding>>(e)) {
      removeInactiveJacs(*collidingJacs, collidingJacIdMapping);
    }

    // Update colliding Jacobian IDs.
    if (auto* activeCollsAsync =
            reg.try_get<CActiveCollisions<ContactType::Async, TimeStep::Current>>(e)) {
      for (auto& activeColl : *activeCollsAsync) {
        MOCHI_ASSERT_VERBOSE(collidingJacIdMapping[activeColl.collidingJacId] >= 0);
        activeColl.collidingJacId = collidingJacIdMapping[activeColl.collidingJacId];
      }
    }
    if (auto* activeCollsSync =
            reg.try_get<CActiveCollisions<ContactType::Sync, TimeStep::Current>>(e)) {
      for (auto& activeColl : *activeCollsSync) {
        MOCHI_ASSERT_VERBOSE(collidingJacIdMapping[activeColl.collidingJacId] >= 0);
        activeColl.collidingJacId = collidingJacIdMapping[activeColl.collidingJacId];
      }
    }
  }

  // Update collider Jacobian IDs now that colliderJacIdMappings has been computed for all actors.
  for (auto const& e : actors) {
    if (auto* activeCollsSync =
            reg.try_get<CActiveCollisions<ContactType::Sync, TimeStep::Current>>(e)) {
      for (auto& activeColl : *activeCollsSync) {
        // Linear search for the collider entity (the list is usually pretty short)
        [[maybe_unused]] bool foundIt = false;
        for (auto const& pair : colliderJacIdMappings) {
          if (pair.first == activeColl.colliderEntity) {
            auto remappedId = pair.second[activeColl.colliderJacId];
            MOCHI_ASSERT_VERBOSE(remappedId >= 0);
            activeColl.colliderJacId = remappedId;
            foundIt = true;
            break;
          }
        }
        MOCHI_ASSERT_VERBOSE(
            foundIt,
            "Expected to find collider entity in colliderJacIdMappings. The code above is broken.");
      }
    }
  }
}

template <TimeStep kTimeStep>
static void InitColliderJacobians(
    TaskSemaphore& sem,
    entt::registry& reg,
    CIslandDescendants const& descendants) {
  MOCHI_PROFILE_SCOPE();

  ecs::ScheduleInvokeForEach(
      sem,
      "deformable::SetupColliderJacobians (soft)",
      &deformable::SetupColliderJacobians,
      reg,
      descendants.softActors);

  ecs::ScheduleInvokeForEach(
      sem,
      "deformable::SetupColliderJacobians (shell)",
      &deformable::SetupColliderJacobians,
      reg,
      descendants.shellActors);

  ecs::ScheduleInvokeForEach(
      sem,
      "deformable::SetupColliderJacobians (rod)",
      &deformable::SetupColliderJacobians,
      reg,
      descendants.rodActors);

  ecs::ScheduleInvokeForEach(
      sem,
      "articulated::rigid::SetupColliderJacobians",
      &articulated::rigid::SetupColliderJacobians<kTimeStep>,
      reg,
      descendants.rigidActors);

  // Rigid actors are pretty cheap. Do them on the calling thread.
  ecs::InvokeForEach(&rigid::SetupColliderJacobians<kTimeStep>, reg, descendants.rigidActors);
}

namespace {
class ContactResAssembler {
 public:
  // Constructor
  ContactResAssembler(ContactJac const& jac, Allocator* allocator)
      : _jac(jac), _resTerm(jac.nDoFsInternal, allocator) {
    if (jac.hasSharedDoFs) {
      _resTerm.SetZero();
    }
  }

  // Compute and possibly assemble
  void ComputeTerm(int nContact, Real3 const& resContact, ColumnVectorView<real> outRes) {
    if (_jac.hasSharedDoFs) {
      // Compute and add for deferred assembly
      _resTerm += _jac.Jac(nContact).Transpose() * ColumnVectorView<real const, 3>(&resContact[0]);
    } else {
      // Compute and assemble
      _resTerm = _jac.Jac(nContact).Transpose() * ColumnVectorView<real const, 3>(&resContact[0]);
      AssembleTerm(nContact, _resTerm, outRes);
    }
  }

  // Deferred assemble
  void DeferredAssemble(ColumnVectorView<real> outRes) {
    if (_jac.hasSharedDoFs) {
      AssembleTerm(0, _jac.JacAux() ? _jac.JacAux().Transpose() * _resTerm : _resTerm, outRes);
    }
  }

 protected:
  // Actor contact jacobian
  ContactJac const& _jac;

  // Local result holder
  ColumnVector<real> _resTerm;

  // Assembly function
  void
  AssembleTerm(int nContact, ColumnVectorView<real const> resTerm, ColumnVectorView<real> outRes) {
    for (int i = 0; i < _jac.IndGroups(nContact).size(); i++) {
      IndexGroup const& inds = _jac.IndGroups(nContact)[i];
      outRes.Slice(inds.dst, inds.count) += resTerm.Slice(inds.src, inds.count);
    }
  }
};

// Given a span of local contact dresiduals dresContact and a contact Jacobian jac, it computes
// the products dresContact[i] * jac using column-major x column-major matrices, and stores the
// result in row-major format for efficient products down the pipeline.
class ContactDResTransformer {
 public:
  // Constructor
  ContactDResTransformer(ContactJac const& jac, Allocator* allocator) : _jac(jac) {
    for (int i = 0; i < kMaxBatchSize; ++i) {
      _dresTerm[i].Reset(3, jac.nDoFsInternal, allocator);
    }
  }

  // Get term
  RowMatrixView<real const, 3> Get(int idx) const {
    return _dresTerm[idx];
  }

  // Compute term
  void ComputeTerm(Span<int const> contactIndices, Span<Matrix<real, 3, 3> const> dresContact) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
    MOCHI_ASSERT_VERBOSE(!contactIndices.empty(), "Empty contact indices.");
    MOCHI_ASSERT_VERBOSE(contactIndices.size() <= kMaxBatchSize, "More indices than legal.");
    MOCHI_ASSERT_VERBOSE(contactIndices.size() == dresContact.size(), "Inconsistent sizes.");
    for (int i = 1; i < isize(contactIndices); ++i) {
      MOCHI_ASSERT_VERBOSE(
          _jac.Inds(contactIndices[i]) == _jac.Inds(contactIndices[0]), "Inconsistent indices.");
    }
#endif
    for (int i = 0; i < isize(contactIndices); ++i) {
      _dresTerm[i] = dresContact[i] * _jac.Jac(contactIndices[i]);
    }
  }

 protected:
  // Actor contact jacobian
  ContactJac const& _jac;

  // Local result holder
  std::array<RowMatrix<real, 3>, kMaxBatchSize> _dresTerm;
};

class ContactDResDiagAssembler {
 public:
  // Constructor
  ContactDResDiagAssembler(
      ContactJac const& jac,
      ContactDResTransformer const& dresJ,
      Allocator* allocator)
      : _jac(jac),
        _dresJ(dresJ),
        _shared(jac.hasSharedDoFs),
        _dresTerm(jac.nDoFsInternal, jac.nDoFsInternal, allocator),
        _allocator(allocator) {
    if (_shared) {
      _dresTerm.SetZero();
    }
  }

  // Compute and possibly assemble
  void ComputeTerm(Span<int const> contactIndices, AnyMatrixView<real>& outDRes) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
    MOCHI_ASSERT_VERBOSE(!contactIndices.empty(), "Empty contact indices.");
    MOCHI_ASSERT_VERBOSE(contactIndices.size() <= kMaxBatchSize, "More indices than legal.");
    for (int i = 1; i < isize(contactIndices); ++i) {
      MOCHI_ASSERT_VERBOSE(
          _jac.Inds(contactIndices[i]) == _jac.Inds(contactIndices[0]), "Inconsistent indices.");
    }
#endif
    // Products are computed in row-major times row-major format
    if (_shared) {
      // Compute and add for deferred assembly
      for (int i = 0; i < isize(contactIndices); ++i) {
        _dresTerm += _jac.Jac(contactIndices[i]).Transpose() * _dresJ.Get(i);
      }
    } else {
      // Compute and assemble
      int cIndex0 = contactIndices[0];
      _dresTerm = _jac.Jac(cIndex0).Transpose() * _dresJ.Get(0);
      for (int i = 1; i < isize(contactIndices); ++i) {
        _dresTerm += _jac.Jac(contactIndices[i]).Transpose() * _dresJ.Get(i);
      }
      MatAddSubBlocks(outDRes, _jac.IndGroups(cIndex0), _jac.IndGroups(cIndex0), _dresTerm);
    }
  }

  // Deferred assemble
  void DeferredAssemble(AnyMatrixView<real>& outDRes) {
    if (_shared) {
      if (_jac.JacAux()) {
        // These products may not be optimally efficient, but they should not happen as often as
        // those in ComputeTerm().
        Matrix<real> temp{_jac.JacAux().Transpose() * _dresTerm, _allocator};
        Matrix<real> dres{temp * _jac.JacAux(), _allocator};
        MatAddSubBlocks(outDRes, _jac.IndGroups(0), _jac.IndGroups(0), dres.Transpose());
      } else {
        MatAddSubBlocks(outDRes, _jac.IndGroups(0), _jac.IndGroups(0), _dresTerm);
      }
    }
  }

 protected:
  // Actor contact jacobian
  ContactJac const& _jac;

  // dresidual transformer
  ContactDResTransformer const& _dresJ;

  // Shared DoFs
  bool _shared;

  // Local result holder
  RowMatrix<real> _dresTerm;

  // Allocator for temporary data
  Allocator* _allocator = nullptr;
};

class ContactDResOffAssembler {
 public:
  // Constructor. If only one of the actor contact Jacobians has shared DoFs, make it actor B
  ContactDResOffAssembler(
      ContactJac const& jacA,
      ContactJac const& jacB,
      ContactDResTransformer const& dresJa,
      ContactDResTransformer const& dresJb,
      Allocator* allocator)
      : _switch(jacA.hasSharedDoFs && !jacB.hasSharedDoFs),
        _jacA(_switch ? jacB : jacA),
        _jacB(_switch ? jacA : jacB),
        _dresJa(_switch ? dresJb : dresJa),
        _dresJb(_switch ? dresJa : dresJb),
        _sharedA(_jacA.hasSharedDoFs),
        _sharedB(_jacB.hasSharedDoFs),
        _dresab(_jacA.nDoFsInternal, _jacB.nDoFsInternal, allocator),
        _dresabFull(_jacA.nDoFsState, _jacB.nDoFsState, allocator),
        _dresbaFull(_jacB.nDoFsState, _jacA.nDoFsState, allocator),
        _allocator(allocator) {
    if (_sharedA && _sharedB) {
      _dresab.SetZero();
    }
  }

  // Compute and possibly assemble
  void ComputeTerm(Span<int const> contactIndices, AnyMatrixView<real>& outDRes) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
    MOCHI_ASSERT_VERBOSE(!contactIndices.empty(), "Empty contact indices.");
    MOCHI_ASSERT_VERBOSE(contactIndices.size() <= kMaxBatchSize, "More indices than legal.");
    for (int i = 1; i < isize(contactIndices); ++i) {
      MOCHI_ASSERT_VERBOSE(
          _jacA.Inds(contactIndices[i]) == _jacA.Inds(contactIndices[0]), "Inconsistent indices.");
      MOCHI_ASSERT_VERBOSE(
          _jacB.Inds(contactIndices[i]) == _jacB.Inds(contactIndices[0]), "Inconsistent indices.");
    }
#endif
    // Products are computed in row-major times row-major format
    if (_sharedA && _sharedB) {
      // Compute and add for deferred assembly
      for (int i = 0; i < isize(contactIndices); ++i) {
        _dresab += _jacA.Jac(contactIndices[i]).Transpose() * _dresJb.Get(i);
      }
    } else {
      // Compute and assemble. Possibly multiply by auxiliary Jacobian
      int cIndex0 = contactIndices[0];
      if (_jacB.JacAux()) {
        _dresab = _jacA.Jac(cIndex0).Transpose() * _dresJb.Get(0);
        for (int i = 1; i < isize(contactIndices); ++i) {
          _dresab += _jacA.Jac(contactIndices[i]).Transpose() * _dresJb.Get(i);
        }
        _dresbaFull = _jacB.JacAux().Transpose() * _dresab.Transpose();
      } else {
        _dresbaFull = _jacB.Jac(cIndex0).Transpose() * _dresJa.Get(0);
        for (int i = 1; i < isize(contactIndices); ++i) {
          _dresbaFull += _jacB.Jac(contactIndices[i]).Transpose() * _dresJa.Get(i);
        }
      }
      _dresabFull = _dresbaFull.Transpose();
      MatAddSubBlocks(outDRes, _jacA.IndGroups(cIndex0), _jacB.IndGroups(cIndex0), _dresabFull);
      MatAddSubBlocks(outDRes, _jacB.IndGroups(cIndex0), _jacA.IndGroups(cIndex0), _dresbaFull);
    }
  }

  // Deferred assemble
  void DeferredAssemble(AnyMatrixView<real>& outDRes) {
    if (_sharedA && _sharedB) {
      // Possibly multiply by auxiliary Jacobians before assembly. These products may not be
      // optimally efficient, but they should not happen as often as those in ComputeTerm()
      RowMatrixView<real> dresbaView(_dresab.Transpose());
      std::optional<RowMatrix<real>> dresbaAux;
      if (_jacB.JacAux()) {
        dresbaAux.emplace(RowMatrix<real>(_jacB.nDoFsState, _jacA.nDoFsInternal, _allocator));
        dresbaAux = _jacB.JacAux().Transpose() * dresbaView;
        dresbaView.Reset(dresbaAux.value());
      }
      if (_jacA.JacAux()) {
        _dresbaFull = dresbaView * _jacA.JacAux();
        dresbaView.Reset(_dresbaFull);
      }
      _dresabFull = dresbaView.Transpose();
      MatAddSubBlocks(outDRes, _jacA.IndGroups(0), _jacB.IndGroups(0), _dresabFull);
      MatAddSubBlocks(outDRes, _jacB.IndGroups(0), _jacA.IndGroups(0), dresbaView);
    }
  }

 protected:
  // bool to switch objects internally. If only one of the actor contact Jacobians has shared DoFs,
  // then it's actor B
  bool _switch;

  // Actor contact jacobians
  ContactJac const& _jacA;
  ContactJac const& _jacB;

  // dresidual transformers
  ContactDResTransformer const& _dresJa;
  ContactDResTransformer const& _dresJb;

  // Shared DoFs
  bool _sharedA;
  bool _sharedB;

  // Local result holders
  Matrix<real> _dresab;
  RowMatrix<real> _dresabFull;
  RowMatrix<real> _dresbaFull;

  // Allocator for temporary data
  Allocator* _allocator = nullptr;
};

} // namespace

// Sync contact assembly between a pair of rigid actors (including articulated rigid). Faster than
// the generic assembly since (1) it operates with compile-time size matrices, and (2) it skips the
// products that involve the identity sub-blocks in the contact Jacobians.
// NOTES:
// - Only supported for sync contact. Async contact is assembled through rigid::EntityAssemble.
// - The 1st contact Jacobian must be the colliding Jacobian, and the 2nd must be the collider
//   Jacobian.
template <GradTarget kGradTarget>
static void AssembleCollisionResponseRange_SyncRigid(
    ContactAssemblyReg reg,
    entt::entity colliding,
    entt::entity collider,
    Interval<int> pointRange,
    ContactDetectionResult const& contactQuery,
    CollisionResponseResult const& collisionResponse,
    Span<real const> intWeights,
    Span<ContactJac const*> jacs,
    Allocator* filoAllocator,
    double* outObj,
    ColumnVectorView<real> outRes,
    AnyMatrixView<real> outDRes) {
  bool const assemObj = (outObj != nullptr);
  bool const assemRes = !outRes.empty();
  bool const assemDRes = (GetNumValues(outDRes) > 0);

  if (assemObj) {
    for (auto s : pointRange) {
      real weight = intWeights[contactQuery.sampleIndices[s]];
      *outObj += weight * collisionResponse.energy[s];
    }
  }

  if (!assemRes && !assemDRes) {
    return;
  }

  // Check assumptions on Jacobians.
  static_assert(
      (krylov::details::MatTraits<decltype(jacs[0]->JacAux())>::kMajorDir ==
       krylov::Direction::ColMajor),
      "Expected column-major storage"); // Assumed when taking views.
#if MOCHI_ASSERT_VERBOSE_ENABLED
  MOCHI_ASSERT_VERBOSE(
      pointRange.Size() > 0, "Expected the caller to early exit."); // Easy to enforce (static)
  MOCHI_ASSERT_VERBOSE(jacs.size() == 2 && jacs[0] && jacs[1], "Requires 2 contact Jacobians.");
  MOCHI_ASSERT_VERBOSE(jacs[0]->groupsInitialized && jacs[1]->groupsInitialized);
  MOCHI_ASSERT_VERBOSE(jacs[0]->hasSharedDoFs && jacs[1]->hasSharedDoFs);
  MOCHI_ASSERT_VERBOSE(
      (jacs[0]->JacAux() && jacs[0]->JacAux().Rows() == 6 && jacs[0]->JacAux().LeadDim() == 6) ||
      (jacs[0]->IndGroups(0).size() == 1 && jacs[0]->IndGroups(0)[0].count == 6));
  MOCHI_ASSERT_VERBOSE(
      (jacs[1]->JacAux() && jacs[1]->JacAux().Rows() == 6 && jacs[1]->JacAux().LeadDim() == 6) ||
      (jacs[1]->IndGroups(0).size() == 1 && jacs[1]->IndGroups(0)[0].count == 6));
#endif

  // Use the rigid state corresponding to the appropriate gradient target.
  TimeStep constexpr kTimeStep = GetTimeStep<kGradTarget>();

  // WARNING: If this implementation is modified, consider modifying also
  // AccumulateAllSyncRigidContactForceAdjoints, which is its dual for differentiability.

  // clang-format off
  // With p the collision point in world space, p' the collision point in collider's local space
  // and comB' the collider's com in local space:
  // Trans jac A (colliding): rotBT
  // Trans jac B (collider): -rotBT
  // Rot jac A: rotBT * sk(comA - p) = rotBT * sk(comA - comB) - sk(pB' - comB') * rotBT
  // Rot jac B: rotBT * sk(p - comB) = sk(pB' - comB') * rotBT
  // The only point-dependent term is sk(pB' - comB'). All other terms can be applied after res
  // and/or dres are accumulated. For convenience, the transpose Jacobians are:
  // Trans jacT A: rotB
  // Trans jacT B: -rotB
  // Rot jacT A: rotB * sk(pB' - comB') - sk(comA - comB) * rotB
  // Rot jacT B: - rotB * sk(pB' - comB')
  // clang-format on
  auto const& jacA = *jacs[0];
  auto const& jacB = *jacs[1];
  Vec4r comA = reg.template get<CRigidState<kTimeStep> const>(colliding).value.VGetTranslation();
  auto const& stateB = reg.template get<CRigidState<kTimeStep> const>(collider).value;
  auto [rotB, rotBT] = ToVMatrix3x3_WithTranspose(stateB.GetRotation());
  Vec4r comB = stateB.VGetTranslation();
  Vec4r comBLocal = reg.template get<CRigidBodyInertia const>(collider).GetCenterOfMassLocal();

  // NOTE: Memory used from filoAllocator for N DoFs is:
  //       (N + 5 * N * N) * sizeof(real)

  if (assemRes) {
    MOCHI_PROFILE_SCOPE_N("Residual Assembly");
    Vec4r res = {};
    Vec4r skPRes = {};
    for (auto s : pointRange) {
      real weight = intWeights[contactQuery.sampleIndices[s]];
      Vec4r collRes = -weight * ToSimd(collisionResponse.force[s]);
      res += collRes;
      auto posColliding = ToSimd(GetCollidingPosition<kTimeStep>(contactQuery, s));
      skPRes += Cross3(posColliding - comBLocal, collRes);
    }
    res = DotVecMat3x3(res, rotBT);
    skPRes = DotVecMat3x3(skPRes, rotBT);
    ColumnVector<real, 6> resA;
    ColumnVector<real, 6> resB;
    Store(resA.data(), res);
    Store<3>(resA.data() + 3, skPRes - Cross3(comA - comB, res));
    Store(resB.data(), -res);
    Store<3>(resB.data() + 3, -skPRes);

    // The contact Jacobians store articulated Jacobians in the case of articulated links, but are
    // not used for rigid actors.
    if (jacA.JacAux()) {
      MatrixView<real const, 6, krylov::kDynamic> auxJacA(jacA.JacAux().data(), 6, jacA.nDoFsState);
      ColumnVector<real> auxResA(auxJacA.Transpose() * resA, filoAllocator);
      auto const& inds = jacA.IndGroups(0);
      for (int i = 0; i < isize(inds); ++i) {
        outRes.Slice(inds[i].dst, inds[i].count) += auxResA.Slice(inds[i].src, inds[i].count);
      }
    } else {
      outRes.Slice<6>(jacA.IndGroups(0)[0].dst, 6) += resA;
    }
    if (jacB.JacAux()) {
      MatrixView<real const, 6, krylov::kDynamic> auxJacB(jacB.JacAux().data(), 6, jacB.nDoFsState);
      ColumnVector<real> auxResB(auxJacB.Transpose() * resB, filoAllocator);
      auto const& inds = jacB.IndGroups(0);
      for (int i = 0; i < isize(inds); ++i) {
        outRes.Slice(inds[i].dst, inds[i].count) += auxResB.Slice(inds[i].src, inds[i].count);
      }
    } else {
      outRes.Slice<6>(jacB.IndGroups(0)[0].dst, 6) += resB;
    }
  }

  if (assemDRes) {
    MOCHI_PROFILE_SCOPE_N("DResidual Assembly");
    // The DResidual is composed of 4 6x6 blocks: AA, BB, AB, BA. Each 6x6 block is composed of 4
    // 3x3 sub-blocks. Some of the 3x3 sub-blocks are shared across 6x6 blocks. The assembly is
    // performed on the 3x3 sub-blocks, which are stored into the 6x6 blocks at the end.
    VMatrix3x3r dres = {};
    VMatrix3x3r skPDres = {};
    VMatrix3x3r skPDresSkP = {};
    for (auto s : pointRange) {
      real weight = intWeights[contactQuery.sampleIndices[s]];
      VMatrix3x3r collDres = -weight * collisionResponse.dforce[s];
      dres += collDres;
      auto posColliding = ToSimd(GetCollidingPosition<kTimeStep>(contactQuery, s));
      VMatrix3x3r collSkP = Skew3(posColliding - comBLocal);
      VMatrix3x3r collSkPDres = Dot3x3(collSkP, collDres);
      skPDres += collSkPDres;
      skPDresSkP += Dot3x3(collSkPDres, collSkP);
    }
    dres = Dot3x3(rotB, Dot3x3(dres, rotBT));
    skPDres = Dot3x3(rotB, Dot3x3(skPDres, rotBT));
    VMatrix3x3r dresSkPNeg = Transpose3x3(skPDres);
    skPDresSkP = Dot3x3(rotB, Dot3x3(skPDresSkP, rotBT));
    VMatrix3x3r skCom = Skew3(comA - comB);
    VMatrix3x3r dresSkCom = Dot3x3(dres, skCom);
    VMatrix3x3r skComDresSkCom = Dot3x3(skCom, dresSkCom);
    VMatrix3x3r skComDresNeg = Transpose3x3(dresSkCom);
    VMatrix3x3r skPDresSkCom = Dot3x3(skPDres, skCom);
    VMatrix3x3r skComDresSkP = Transpose3x3(skPDresSkCom);

    // AA diagonal block.
    RowMatrix<real, 6, 6> aaBlock;
    aaBlock.Block<3, 3>(0, 0, 3, 3) = AsMatrixView(dres);
    aaBlock.Block<3, 3>(0, 3, 3, 3) = AsMatrixView(dresSkCom + dresSkPNeg);
    aaBlock.Block<3, 3>(3, 0, 3, 3) = aaBlock.Block<3, 3>(0, 3, 3, 3).Transpose();
    aaBlock.Block<3, 3>(3, 3, 3, 3) =
        AsMatrixView(skPDresSkCom + skComDresSkP - skPDresSkP - skComDresSkCom);

    // BB diagonal block.
    RowMatrix<real, 6, 6> bbBlock;
    bbBlock.Block<3, 3>(0, 0, 3, 3) = AsMatrixView(dres);
    bbBlock.Block<3, 3>(0, 3, 3, 3) = AsMatrixView(dresSkPNeg);
    bbBlock.Block<3, 3>(3, 0, 3, 3) = bbBlock.Block<3, 3>(0, 3, 3, 3).Transpose();
    bbBlock.Block<3, 3>(3, 3, 3, 3) = AsMatrixView(-skPDresSkP);

    // AB off-diagonal block.
    RowMatrix<real, 6, 6> abBlock;
    abBlock.Block<3, 3>(0, 0, 3, 3) = AsMatrixView(-dres);
    abBlock.Block<3, 3>(0, 3, 3, 3) = AsMatrixView(-dresSkPNeg);
    abBlock.Block<3, 3>(3, 0, 3, 3) = AsMatrixView(-skPDres - skComDresNeg);
    abBlock.Block<3, 3>(3, 3, 3, 3) = AsMatrixView(skPDresSkP - skComDresSkP);

    RowMatrixView<real> aaBlockView(aaBlock);
    RowMatrixView<real> bbBlockView(bbBlock);
    RowMatrixView<real> abBlockView(abBlock);

    // The auxiliary Jacobians are applied using the following storage directions for performance
    // reasons:
    // AA and BB diagonal blocks:
    // - Product #1: Col-Major = Row-Major x Row-Major
    // - Product #2: Col-Major = Col-Major x Col-Major -> Requires more FLOPs than #1 if nDoFs > 6
    // AB off-diagonal block:
    // - Product #1: Row-Major = Row-Major x Col-Major
    // - Product #2: Row-Major = Row-Major x Row-Major -> Requires more FLOPs than #1 if nDoFs > 6
    std::optional<RowMatrix<real>> bbBlockAux; // Declared in same order as potentially allocated.
    std::optional<RowMatrix<real>> abBlockAux1;
    std::optional<RowMatrix<real>> aaBlockAux;
    std::optional<RowMatrix<real>> abBlockAux2;
    if (jacB.JacAux()) {
      // BB diagonal block.
      MatrixView<real const, 6, krylov::kDynamic> auxJacB(jacB.JacAux().data(), 6, jacB.nDoFsState);
      bbBlockAux.emplace(RowMatrix<real>(jacB.nDoFsState, jacB.nDoFsState, filoAllocator));
      bbBlockAux->Transpose() =
          Matrix<real, krylov::kDynamic, 6>(auxJacB.Transpose() * bbBlock, filoAllocator) * auxJacB;
      bbBlockView.Reset(bbBlockAux.value());

      // AB off-diagonal block.
      abBlockAux1.emplace(RowMatrix<real>(6, jacB.nDoFsState, filoAllocator));
      abBlockAux1 = abBlock * auxJacB;
      abBlockView.Reset(abBlockAux1.value());
    }

    if (jacA.JacAux()) {
      // AA diagonal block.
      MatrixView<real const, 6, krylov::kDynamic> auxJacA(jacA.JacAux().data(), 6, jacA.nDoFsState);
      aaBlockAux.emplace(RowMatrix<real>(jacA.nDoFsState, jacA.nDoFsState, filoAllocator));
      aaBlockAux->Transpose() =
          Matrix<real, krylov::kDynamic, 6>(auxJacA.Transpose() * aaBlock, filoAllocator) * auxJacA;
      aaBlockView.Reset(aaBlockAux.value());

      // AB off-diagonal block.
      abBlockAux2.emplace(RowMatrix<real>(jacA.nDoFsState, abBlockView.Cols(), filoAllocator));
      abBlockAux2 = auxJacA.Transpose() * abBlockView;
      abBlockView.Reset(abBlockAux2.value());
    }

    MatAddSubBlocks(outDRes, jacA.IndGroups(0), jacA.IndGroups(0), aaBlockView);
    MatAddSubBlocks(outDRes, jacB.IndGroups(0), jacB.IndGroups(0), bbBlockView);
    MatAddSubBlocks(outDRes, jacA.IndGroups(0), jacB.IndGroups(0), abBlockView);
    MatAddSubBlocks(
        outDRes,
        jacB.IndGroups(0),
        jacA.IndGroups(0),
        RowMatrix<real>(abBlockView.Transpose(), filoAllocator));
  }
}

template <GradTarget kGradTarget>
static void AssembleCollisionResponseRange(
    ContactAssemblyReg reg,
    entt::entity colliding,
    entt::entity collider,
    Interval<int> pointRange,
    ContactDetectionResult const& contactQuery,
    CollisionResponseResult const& collisionResponse,
    Span<real const> intWeights,
    Span<ContactJac const*> jacs,
    Allocator* filoAllocator,
    double* outObj,
    ColumnVectorView<real> outRes,
    AnyMatrixView<real> outDRes,
    bool isSyncRigid) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(pointRange.Valid());
  MOCHI_ASSERT_VERBOSE(
      pointRange.Min() >= 0 && pointRange.Max() < isize(contactQuery.sampleIndices));

  if (pointRange.Size() == 0) {
    return;
  }
  if (isSyncRigid) {
    return AssembleCollisionResponseRange_SyncRigid<kGradTarget>(
        reg,
        colliding,
        collider,
        pointRange,
        contactQuery,
        collisionResponse,
        intWeights,
        jacs,
        filoAllocator,
        outObj,
        outRes,
        outDRes);
  }

  bool assemObj = (outObj != nullptr);
  bool assemRes = !outRes.empty();
  bool assemDRes = (GetNumValues(outDRes) > 0);

  if (assemRes || assemDRes) {
    for (auto const& jac : jacs) {
      MOCHI_ASSERT(jac, "Contact Jacobian must exist");
      MOCHI_ASSERT(jac->groupsInitialized, "Index groups not initialized");
    }
  }

  // Assemble objective
  if (assemObj) {
    for (auto s : pointRange) {
      real intWeight = intWeights[contactQuery.sampleIndices[s]];
      *outObj += intWeight * collisionResponse.energy[s];
    }
  }

  // Assemble residual
  if (assemRes) {
    DynamicArray<ContactResAssembler> resAssemblers(filoAllocator);
    resAssemblers.reserve(jacs.size());
    for (auto const& jac : jacs) {
      resAssemblers.emplace_back(*jac, filoAllocator);
    }
    {
      MOCHI_PROFILE_SCOPE_N("Residual Assembly Products");
      for (auto s : pointRange) {
        real intWeight = intWeights[contactQuery.sampleIndices[s]];
        // Transform force to DoFs.
        Real3 collRes = -intWeight * collisionResponse.force[s];
        for (auto& resAssembler : resAssemblers) {
          resAssembler.ComputeTerm(s, collRes, outRes);
        }
      }
    }
    {
      MOCHI_PROFILE_SCOPE_N("Residual Deferred Assembly");
      for (auto& resAssembler : resAssemblers) {
        resAssembler.DeferredAssemble(outRes);
      }
    }
  }

  // Assemble dresidual
  if (assemDRes) {
    DynamicArray<ContactDResTransformer> dresTransformers(filoAllocator);
    DynamicArray<ContactDResDiagAssembler> dresAssemblersii(filoAllocator);
    DynamicArray<ContactDResOffAssembler> dresAssemblersij(filoAllocator);
    dresTransformers.reserve(jacs.size());
    dresAssemblersii.reserve(jacs.size());
    dresAssemblersij.reserve(!jacs.empty() ? jacs.size() * (jacs.size() - 1) / 2 : 0);
    for (int i = 0; i < isize(jacs); i++) {
      dresTransformers.emplace_back(*jacs[i], filoAllocator);
      dresAssemblersii.emplace_back(*jacs[i], dresTransformers[i], filoAllocator);
      for (int j = 0; j < i; j++) {
        dresAssemblersij.emplace_back(
            *jacs[i], *jacs[j], dresTransformers[i], dresTransformers[j], filoAllocator);
      }
    }
    {
      // Assemble products. If consecutive sample points affect the same DoFs (up to kMaxBatchSize),
      // they are processed together by ContactDResXYZ utilities. This reduces the number of write
      // operations from the dense matrix in which the local contact dresidual is accumulated to the
      // global contact dresidual (expensive).
      MOCHI_PROFILE_SCOPE_N("Assembly products");
      std::array<Matrix<real, 3, 3>, kMaxBatchSize> collDResBuffer;
      std::array<int, kMaxBatchSize> contactIndicesBuffer{};
      int sBatchBegin = pointRange.Min();
      while (sBatchBegin <= pointRange.Max()) { // Note Max() is inclusive.
        int sCandidate = sBatchBegin;
        int numPointsInBatch = 0;
        bool pointSharesIndices = true;
        for (; sCandidate <= pointRange.Max() && numPointsInBatch < kMaxBatchSize &&
             pointSharesIndices;
             ++sCandidate) {
          if (sCandidate != sBatchBegin) {
            for (auto const& jac : jacs) {
              if (jac->Inds(sCandidate) != jac->Inds(sBatchBegin)) {
                pointSharesIndices = false;
                break;
              }
            }
          }

          if (pointSharesIndices) {
            contactIndicesBuffer[numPointsInBatch] = sCandidate;
            real intWeight = intWeights[contactQuery.sampleIndices[sCandidate]];
            collDResBuffer[numPointsInBatch] =
                Matrix<real, 3, 3>(-intWeight * AsMatrixView(collisionResponse.dforce[sCandidate]));
            ++numPointsInBatch;
          } else {
            break;
          }
        }

        if (numPointsInBatch > 0) {
          // Transform dforce to DoFs and possibly assemble
          auto contactIndicesInBatch = Span(contactIndicesBuffer.data(), numPointsInBatch);
          auto collDResInBatch = Span(collDResBuffer.data(), numPointsInBatch);
          for (auto& dresTransformer : dresTransformers) {
            dresTransformer.ComputeTerm(contactIndicesInBatch, collDResInBatch);
          }
          for (auto& dresAssemblerii : dresAssemblersii) {
            dresAssemblerii.ComputeTerm(contactIndicesInBatch, outDRes);
          }
          for (auto& dresAssemblerij : dresAssemblersij) {
            dresAssemblerij.ComputeTerm(contactIndicesInBatch, outDRes);
          }
        }
        sBatchBegin = sCandidate;
      }
    }
    {
      MOCHI_PROFILE_SCOPE_N("Deferred assembly");
      for (auto& dresAssemblerii : dresAssemblersii) {
        dresAssemblerii.DeferredAssemble(outDRes);
      }
      for (auto& dresAssemblerij : dresAssemblersij) {
        dresAssemblerij.DeferredAssemble(outDRes);
      }
    }

    // Destroy in reverse order to satisfy FILO allocator
    for (int i = isize(jacs) - 1; i >= 0; --i) {
      for (int j = i - 1; j >= 0; --j) {
        dresAssemblersij.pop_back();
      }
      dresAssemblersii.pop_back();
      dresTransformers.pop_back();
    }
  }
}

MOCHI_API void mochi::AssembleCollisionResponse(
    ContactAssemblyReg reg,
    entt::entity colliding,
    entt::entity collider,
    ContactDetectionResult const& contactQuery,
    CollisionResponseResult const& collisionResponse,
    Span<real const> intWeights,
    Span<ContactJac const*> jacs,
    Allocator* filoAllocator,
    double* outObj,
    ColumnVectorView<real> outRes,
    AnyMatrixView<real> outDRes,
    bool isSyncRigid) {
  int numContactPoints = isize(contactQuery.sampleIndices);
  AssembleCollisionResponseRange<GradTarget::Current>(
      reg,
      colliding,
      collider,
      Interval<int>{0, numContactPoints}, // full range
      contactQuery,
      collisionResponse,
      intWeights,
      jacs,
      filoAllocator,
      outObj,
      outRes,
      outDRes,
      isSyncRigid);
}

template <bool kUpdateOnlyActiveFaces, typename DiscretizationType, int kNumFields>
void mochi::UpdateCollisionSamplePositionsImpl(
    ColumnVectorView<real const> currSol,
    DiscretizationType const& boundaryDiscrVariant,
    CActiveBoundaryFaces const* activeBoundaryFaces,
    ContactSamples& outSamples) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(
      !kUpdateOnlyActiveFaces || activeBoundaryFaces,
      "Cannot update active faces without CActiveBoundaryFaces component.");

  boundaryDiscrVariant.Visit([&](auto const& boundaryDiscr) {
    using DiscretizationT = std::decay_t<decltype(boundaryDiscr)>;
    static int constexpr kSpaceDim = DiscretizationT::kSpaceDim;
    static int constexpr kNumQuadsPerFace = DiscretizationT::kNumQuads;
    static_assert(kSpaceDim == 3, "Invalid spatial dimensions");
    static_assert(kNumFields >= kSpaceDim, "Not enough fields to define displacement");
    int const numBoundaryFaces = isize(boundaryDiscr.femElements);
    int const numBoundaryFacesToUpdate =
        kUpdateOnlyActiveFaces ? isize(activeBoundaryFaces->ViewIndices()) : numBoundaryFaces;

    // Resize dynamic vectors.
    if constexpr (kUpdateOnlyActiveFaces) {
      auto numSamplesToUpdate = static_cast<size_t>(numBoundaryFacesToUpdate) * kNumQuadsPerFace;
      outSamples.activePositions.resize(numSamplesToUpdate);
      outSamples.activeIndices.resize(numSamplesToUpdate);
    } else {
      outSamples.activePositions.clear();
      outSamples.activeIndices.clear();
    }

    // For each element in the boundary discretization...
    static int constexpr kMinPerTask =
        600 / kNumQuadsPerFace; // At least 20 μs per task (assuming ~100 ns
                                // per element with 3 quadrature points).
    ParallelForN("CalcCollisionSample", numBoundaryFacesToUpdate, kMinPerTask, [&](int j) {
      // Fetch element and indices of the degrees of freedom.
      int const boundaryFaceIdx =
          kUpdateOnlyActiveFaces ? activeBoundaryFaces->ViewIndices()[j] : j;
      auto const& element = boundaryDiscr.femElements[boundaryFaceIdx];
      using ElementType = std::decay_t<decltype(element)>;
      int constexpr kNumNodesPerFace = ElementType::kNumNodes;
      int const boundaryFaceOffset = boundaryFaceIdx * kNumQuadsPerFace;
      int const activeOffset = j * kNumQuadsPerFace;
      auto const dofIndices = kNumFields * element.Nodes();
      auto const localNodes = element.LocalNodes();

      // Load the displacements for each node
      NdArray<Vec4r, kNumNodesPerFace> nodeDisplacements;
      for (int i = 0; i < kNumNodesPerFace; ++i) {
        nodeDisplacements[i] = Load<3, Vec4r>(&currSol[dofIndices[i]]);
      }

      // Store the element quad weights scaled (if applicable) by the boundary face weight.
      // These quad weights are copied below in outSamples.weights.
      real const thisFaceWeight =
          kUpdateOnlyActiveFaces ? activeBoundaryFaces->ViewWeights()[boundaryFaceIdx] : 1_r;
      auto const quadWeights = element.quadWeights * thisFaceWeight;

      // Interpolate displacement field at quadrature points.
      for (int q = 0; q < kNumQuadsPerFace; ++q) {
        int const sampleIndex = boundaryFaceOffset + q;

        // Compute local position
        Vec4r samplePos = ToSimd(element.mapEvaluated[q]);
        for (int i = 0; i < kNumNodesPerFace; ++i) {
          samplePos += nodeDisplacements[i] * element.basisEvaluated[q][localNodes[i]];
        }
        outSamples.positions[sampleIndex] = ToReal3(samplePos);

        // Store the quadrature weight
        outSamples.weights[sampleIndex] = quadWeights[q];

        if constexpr (kUpdateOnlyActiveFaces) {
          outSamples.activePositions[activeOffset + q] = ToReal3(samplePos);
          outSamples.activeIndices[activeOffset + q] = sampleIndex;
        }
      }
    });
  });

  if (outSamples.bsh) {
    outSamples.bsh->Refit();
  }
}

#define MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES_IMPL(activeFaces, discretization, numFields)     \
  template void mochi::UpdateCollisionSamplePositionsImpl<activeFaces, discretization, numFields>( \
      ColumnVectorView<real const>,                                                                \
      discretization const&,                                                                       \
      CActiveBoundaryFaces const*,                                                                 \
      ContactSamples&);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES_IMPL(false, CFemBoundaryDiscretization, 3);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES_IMPL(true, CFemBoundaryDiscretization, 3);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES_IMPL(false, CFemSurfaceDiscretization, 3);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES_IMPL(true, CFemSurfaceDiscretization, 3);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES_IMPL(false, CFemSegmentDiscretization, 4);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES_IMPL(true, CFemSegmentDiscretization, 4);
#undef MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES_IMPL

template <typename DiscretizationType, TimeStep kTimeStep, int kNumFields>
MOCHI_API void mochi::UpdateCollisionSamplePositions(
    ecs::RequiredTag<TagUseContact>,
    CFinalDisplacementRef<kTimeStep> const& currSol,
    DiscretizationType const& discretization,
    CActiveBoundaryFaces const* activeBoundaryFaces,
    CContactSamples<kTimeStep>& outSamples) {
  MOCHI_PROFILE_SCOPE();

  if (!activeBoundaryFaces) {
    UpdateCollisionSamplePositionsImpl</*kUpdateOnlyActiveFaces*/ false,
                                       DiscretizationType,
                                       kNumFields>(
        currSol.value, discretization, activeBoundaryFaces, outSamples);
  } else {
    UpdateCollisionSamplePositionsImpl</*kUpdateOnlyActiveFaces*/ true,
                                       DiscretizationType,
                                       kNumFields>(
        currSol.value, discretization, activeBoundaryFaces, outSamples);
  }
}

#define MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES(dicretization, timeStep, numFields) \
  template MOCHI_API void                                                             \
  mochi::UpdateCollisionSamplePositions<dicretization, timeStep, numFields>(          \
      ecs::RequiredTag<TagUseContact>,                                                \
      CFinalDisplacementRef<timeStep> const&,                                         \
      dicretization const&,                                                           \
      CActiveBoundaryFaces const*,                                                    \
      CContactSamples<timeStep>&);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES(CFemBoundaryDiscretization, TimeStep::Current, 3);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES(CFemBoundaryDiscretization, TimeStep::StageStart, 3);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES(CFemSurfaceDiscretization, TimeStep::Current, 3);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES(CFemSurfaceDiscretization, TimeStep::StageStart, 3);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES(CFemSegmentDiscretization, TimeStep::Current, 4);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES(CFemSegmentDiscretization, TimeStep::StageStart, 4);
#undef MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES

void mochi::UpdateQuerySdfSurface(
    CSdfCollider const& collider,
    CBoundingVolume<TimeStep::Current> const& bounds,
    CQuerySdfSurface& outQuery) {
  MOCHI_PROFILE_SCOPE();

  static constexpr int kMaxIterations = 5;
  static constexpr real kCloseEnough = 0.001_r; // Stop iterating when it gets at least this close
  static constexpr real kTooFarToConsider = 0.01_r; // Immediately cull points farther than this
  static constexpr int kMinSamplesPerSide = 2; // Min  resolution is 2x2x2 points (box corners).
  static constexpr int kMaxSamplesPerSide = 32; // Max resolution
  static constexpr real kMinSampleSpacing = 0.005_r; // Use fewer samples if they would be too close

  // Clear
  outQuery.positions.clear();
  outQuery.normals.clear();

  Aabb localAabb = GetAabb(bounds.localShape);
  Real3 localAabbSize = localAabb.GetSize();

  // Subdivide the local Aabb into a grid of points.
  auto numNodesPerSide = StaticCast<Int3>(localAabbSize / kMinSampleSpacing);
  for (int& num : numNodesPerSide) {
    num = Clamp(num, kMinSamplesPerSide, kMaxSamplesPerSide);
  }

  int numNodes = numNodesPerSide[0] * numNodesPerSide[1] * numNodesPerSide[2];
  std::vector<Real3> tempPositions(numNodes);
  Real3 corner = localAabb.GetMin();
  Real3 increment = localAabbSize / StaticCast<Real3>(numNodesPerSide - Int3{1, 1, 1});
  int idx = 0;
  Real3 pt;
  pt[0] = corner[0];
  for (int x = 0; x < numNodesPerSide[0]; ++x, pt[0] += increment[0]) {
    pt[1] = corner[1];
    for (int y = 0; y < numNodesPerSide[1]; ++y, pt[1] += increment[1]) {
      pt[2] = corner[2];
      for (int z = 0; z < numNodesPerSide[2]; ++z, pt[2] += increment[2]) {
        tempPositions[idx++] = pt;
      }
    }
  }

  ContactDetectionParams params;
  ContactDetectionResult result;

  // We don't just want points on the surface. We also want points some distance beyond the surface
  // because we can project them onto the surface and iterate from there.
  params.tolerance = kTooFarToConsider;

  // Iteratively move points toward the surface (where abs(sd) is zero)
  for (int i = 0; i < kMaxIterations; ++i) {
    // Calc signed distance for temp positions.
    FindPointContactsT(
        tempPositions,
        static_cast<GridSdf const*>(collider.shape.get()),
        params,
        TransformRT{},
        result.sampleIndices,
        result.posColliding,
        result.sdfInfo,
        result.isSdfGradUnitary);
    int const numResults = isize(result.sdfInfo);
    int redoCount = 0;
    for (int r = 0; r < numResults; ++r) {
      auto normal = Normalize(result.sdfInfo.grad[r]);
      auto distance = result.sdfInfo.val[r];
      auto posCollider = result.posColliding[r] - normal * distance;
      if (std::abs(distance) <= kCloseEnough) {
        // This point is close enough. Copy it to the query output
        outQuery.positions.push_back(posCollider);
        outQuery.normals.push_back(normal);
      } else {
        // Copy the revised position back into tempPositions for the next iteration.
        tempPositions[redoCount] = posCollider;
        redoCount++;
      }
    }

    // Trim tempPositions (gets smaller as values move to query.positions)
    tempPositions.resize(redoCount);
    if (tempPositions.empty()) {
      break; // done
    }

    // Reset for the next iteration
    result.Clear();
  }
}

void mochi::UpdateQueryContactSamples(
    CContactSamples<TimeStep::Current> const& samplePositions,
    CRootTransform const& rootTransform,
    CRecenteringParams const* recenterParams,
    ecs::OptionalTag<TagSoftActor> hasSoftActorTag,
    CQueryContactSamples& outQuery) {
  MOCHI_PROFILE_SCOPE();

  auto flatPositions = Flatten(Span<Real3 const>(samplePositions.positions));
  outQuery.contactSamples.assign(flatPositions.begin(), flatPositions.end());

  // Only for soft actors: if recentering is activated, correct last contacts
  if (hasSoftActorTag && recenterParams && recenterParams->useRecentering &&
      rootTransform.worldFromLocal != rootTransform.worldFromLocalPrev) {
    ArrayTransformPoints(
        Unflatten<Real3>(outQuery.contactSamples),
        Unflatten<Real3 const>(outQuery.contactSamples),
        Invert(rootTransform.worldFromLocal) * rootTransform.worldFromLocalPrev);
  }
}

struct SdfSample {
  real distance;
  Real3 worldPosition;
  Real3 distanceGrad;
};

template <ContactType kContactType>
static void AppendActiveSdfSamplesWorldSpace(
    CActiveCollisions<kContactType, TimeStep::Current> const& collisions,
    CContactSamples<TimeStep::Current> const& collidingSamples,
    TransformRT const& collidingTransform,
    std::unordered_map<int, SdfSample>& samples) {
  size_t const numColliders = collisions.size();
  if (numColliders > 0) {
    for (size_t iCollider = 0; iCollider < numColliders; ++iCollider) {
      auto const& collisionResults = collisions[iCollider].collisionResult;
      size_t const numContacts = collisionResults.posColliding.size();

      // Handle jacColliderFromWorld indexing: size 1 for rigid, per-contact for deformable
      int const dJac = collisionResults.jacColliderFromWorld.size() == 1 ? 0 : 1;
      for (size_t iColliderContact = 0, iJac = 0; iColliderContact < numContacts;
           ++iColliderContact, iJac += dJac) {
        auto const sampleIndex = collisionResults.sampleIndices[iColliderContact];

        // Fetch the Jacobian between collider and world space
        auto const& jacColliderFromWorld = collisionResults.jacColliderFromWorld[iJac];

        // Compute world position from colliding sample position
        Real3 const worldPosition =
            collidingTransform.TransformPoint(collidingSamples.positions[sampleIndex]);

        // Transform distance gradient from collider space to world space
        Real3 const distanceGrad = ToReal3(DotVecMat3x3(
            ToSimd(collisionResults.sdfInfo.grad[iColliderContact]), jacColliderFromWorld));

        auto sample = SdfSample{
            .distance = collisionResults.sdfInfo.val[iColliderContact],
            .worldPosition = worldPosition,
            .distanceGrad = distanceGrad};

        if (auto it = samples.find(sampleIndex); it != samples.end()) {
          // This sample index has already been processed, so we need to update the distance if
          // closer
          if (it->second.distance > sample.distance) {
            it->second = sample;
          }
        } else {
          // This sample index has not been processed yet, so we need to add it to the list
          samples.emplace_hint(it, sampleIndex, sample);
        }
      }
    }
  }
}

void mochi::UpdateQuerySdfDistances(
    CContactSamples<TimeStep::Current> const& collidingSamples,
    CRootTransform const& rootTransform,
    CActiveCollisions<ContactType::Async, TimeStep::Current> const* collisionsAsync,
    CActiveCollisions<ContactType::Sync, TimeStep::Current> const* collisionsSync,
    CRequiresFarSdfEvaluation const* farSdfEval,
    CConvergenceStatus const* convergenceStatus,
    CQuerySdfDistances& outQuerySdfDistances) {
  MOCHI_PROFILE_SCOPE();

  // Use the colliding actor's transform to convert positions to world space
  auto const& collidingTransform = rootTransform.worldFromLocal;

  // Compute the closest contact point for each sample index
  std::unordered_map<int, SdfSample> contactPointDetails;

  // If the solver diverged, contact/SDF data is stale (actor state was rolled back without
  // re-running assembly). Do not populate query.
  if (!convergenceStatus || convergenceStatus->stepStatus != ConvergenceStatus::Diverged) {
    if (collisionsAsync) {
      AppendActiveSdfSamplesWorldSpace(
          *collisionsAsync, collidingSamples, collidingTransform, contactPointDetails);
    }
    if (collisionsSync) {
      AppendActiveSdfSamplesWorldSpace(
          *collisionsSync, collidingSamples, collidingTransform, contactPointDetails);
    }
  }

  // Aggregate the results into the output query structure
  outQuerySdfDistances.Clear();
  outQuerySdfDistances.Reserve(contactPointDetails.size());

  for (auto const& [sampleIndex, deets] : contactPointDetails) {
    outQuerySdfDistances.sampleIndices.push_back(sampleIndex);
    outQuerySdfDistances.distances.push_back(deets.distance);
    outQuerySdfDistances.worldPositions.push_back(deets.worldPosition);
    outQuerySdfDistances.distanceGrads.push_back(deets.distanceGrad);
  }
  outQuerySdfDistances.isInitialized = true;

  if (farSdfEval) {
    outQuerySdfDistances.maxSdfFarDistanceEvaluation = farSdfEval->maxDistance;
  } else {
    outQuerySdfDistances.maxSdfFarDistanceEvaluation = 0.0_r;
  }
}

void mochi::UpdateQueryActiveContactsWorldSpace(
    entt::registry const& reg,
    entt::entity e,
    ecs::RequiredTag<TagQueryActiveContacts>,
    ecs::Excluded<CRequiresFarSdfEvaluation>,
    CActiveCollisions<ContactType::Async, TimeStep::Current> const& collisionsAsync,
    CActiveCollisions<ContactType::Sync, TimeStep::Current> const& collisionsSync,
    CCollJacs<CollRole::Collider> const* colliderJacs,
    CConvergenceStatus const* convergenceStatus,
    CQueryContactPoints* outQueryActiveContacts,
    CQueryNodeContactForces* outQueryNodeForces) {
  if (!outQueryActiveContacts && !outQueryNodeForces) {
    return;
  }

  MOCHI_PROFILE_SCOPE();

  // Clear outputs
  if (outQueryActiveContacts) {
    outQueryActiveContacts->contactPoints.clear();
  }
  if (outQueryNodeForces) {
    outQueryNodeForces->nodeContactForcesMap.clear();
  }

  // If the solver diverged, contact data is stale (actor state was rolled back without re-running
  // assembly). Do not populate queries.
  if (!convergenceStatus || convergenceStatus->stepStatus != ConvergenceStatus::Diverged) {
    SceneHandle sceneHandle = reg.ctx<CSceneHandle const>().value;

    auto appendActiveContactsForPair = [&](entt::entity colliding,
                                           entt::entity collider,
                                           ContactDetectionResult const& collisionResult,
                                           CQueryContactPoints* outActiveContacts,
                                           CQueryNodeContactForces* outNodeForces,
                                           ContactType contactType) {
      // Validate that the force data is available
      int numContacts = isize(collisionResult.posColliding);
      MOCHI_ASSERT(
          numContacts == collisionResult.forcePerUnitArea.size(),
          "Expected a force vector for every contact between this pair of entities");

      ActorHandle collidingActor = GetActorHandle(colliding, sceneHandle);
      ActorHandle colliderActor = GetActorHandle(collider, sceneHandle);

      // Fetch colliding actor data. For deformable actors, use the old root transform as it may
      // have been updated by recentering.
      auto const isDeformable = reg.all_of<TagDeformableActor>(colliding);
      auto const isRigid = reg.all_of<TagRigidActor>(colliding);
      auto const& root = reg.get<CRootTransform const>(colliding);
      auto const& transformCurrent = isDeformable ? root.worldFromLocalPrev : root.worldFromLocal;
      auto const jacCollidingFromWorld = ToVMatrix3x3Transpose(transformCurrent.GetRotation());
      auto const& transformStageStart = isRigid
          ? root.worldFromLocalStageStart
          : (isDeformable ? root.worldFromLocalPrev : root.worldFromLocal);
      auto const& samples = reg.get<CContactSamples<TimeStep::Current> const>(colliding);
      auto const* samplesStageStart =
          reg.try_get<CContactSamples<TimeStep::StageStart> const>(colliding);
      auto const* boundaryDiscretization = reg.try_get<CFemBoundaryDiscretization const>(colliding);
      auto const* surfaceDiscretization =
          reg.try_get<CFemSurfaceDiscretizationLite const>(colliding);

      // Fetch dtStage for velocity computation
      real const dtStage = reg.get<CTimeIntegratorState const>(colliding).dtStage;
      auto const isRodActor = reg.any_of<TagRodActor>(colliding);
      MOCHI_ASSERT(
          boundaryDiscretization || surfaceDiscretization || isRodActor,
          "Colliding actor must have a boundary or surface discretization unless it is a rod actor");
      MOCHI_ASSERT(
          !outNodeForces || !isRodActor, "Node contact forces not supported for rod actors");

      // Add contacts to the query data
      int const dJac = collisionResult.jacColliderFromWorld.size() == 1 ? 0 : 1;
      for (int iContact = 0, iJac = 0; iContact < numContacts; ++iContact, iJac += dJac) {
        int sampleIndex = collisionResult.sampleIndices[iContact];

        // Fetch the Jacobian between collider and world space
        auto const& jacColliderFromWorld = collisionResult.jacColliderFromWorld[iJac];

        // Fetch the integration weight
        real weight = samples.weights[sampleIndex];

        // Fetch the force and scale by the integration weight
        Real3 force = weight * collisionResult.forcePerUnitArea[iContact];

        // Transform the force to world space. For async contact on deformable actors, force is in
        // colliding space. Otherwise it is in collider space.
        force = ToReal3(DotVecMat3x3(
            ToSimd(force),
            (contactType == ContactType::Async && isDeformable) ? jacCollidingFromWorld
                                                                : jacColliderFromWorld));

        // Get the discretization info of the contact point (element, nodes and basis weights).
        // Consider two cases: the discretization is made with trace elements of a volume
        // discretization (e.g. for soft actors), or the discretization is made directly with a
        // surface discretization (e.g. for rigid actors).
        // Rod actors do not support these data, because they don't use a triangle-mesh
        // discretization.
        int elementIndex = {}; // Needed for CQueryContactPoints
        Int3 nodeIndices = {}; // Needed for CQueryNodeContactForces
        Real3 basisWeights = {}; // Needed for both
        if (boundaryDiscretization) {
          boundaryDiscretization->Visit([&](auto const& boundaryDiscretization) {
            if (outQueryActiveContacts) {
              using TraceElementT =
                  typename std::decay_t<decltype(boundaryDiscretization)>::ElementT;
              static int constexpr kNumQuadsPerFace = TraceElementT::kNumQuadPoints;
              elementIndex = sampleIndex / kNumQuadsPerFace;
            }
            if (outNodeForces) {
              nodeIndices = GetNodeInfo(MakeSpan(boundaryDiscretization.femElements), sampleIndex);
            }
            basisWeights = GetWeightInfo(MakeSpan(boundaryDiscretization.femElements), sampleIndex);
          });
        } else if (surfaceDiscretization) {
          surfaceDiscretization->Visit([&](auto const& surfaceDiscretization) {
            using FaceElementT = typename std::decay_t<decltype(surfaceDiscretization)>::ElementT;
            static int constexpr kNumQuadsPerFace = FaceElementT::kNumQuadPoints;
            elementIndex = sampleIndex / kNumQuadsPerFace;
            if (outNodeForces) {
              nodeIndices = surfaceDiscretization.connectivity[elementIndex];
            }
            int const quadPoint = sampleIndex % kNumQuadsPerFace;
            basisWeights = FaceElementT::kBasisEvaluated[quadPoint];
          });
        }

        // Add to the query data
        if (outActiveContacts) {
          auto const& sdfInfo = collisionResult.sdfInfo;
          auto const& posColliding = collisionResult.posColliding;
          auto const& posCollidingStageStart = collisionResult.posCollidingStageStart;

          // Much of the data in collisionResult is in collider space and cannot be used directly.
          // distance: use the value in sdfInfo.val
          real const distance = sdfInfo.val[iContact];
          // normal: Transform from collider space to world space and normalize.
          Real3 const normal = ToReal3(
              Normalize<3>(DotVecMat3x3(ToSimd(sdfInfo.grad[iContact]), jacColliderFromWorld)));
          // posA: Fetch the colliding sample position and transform to world space.
          Real3 const posA = transformCurrent.TransformPoint(samples.positions[sampleIndex]);
          // posB: Estimate based on posA, distance and normal
          Real3 const posB = posA - normal * distance;

          // pointVelocityA: velocity of the colliding point. Estimated by finite differences of
          // current and stage-start world positions.
          Real3 posAStageStart = samplesStageStart ? samplesStageStart->positions[sampleIndex]
                                                   : samples.positions[sampleIndex];
          posAStageStart = transformStageStart.TransformPoint(posAStageStart);
          Real3 const pointVelocityA = (posA - posAStageStart) / dtStage;
          // pointVelocityB: velocity of the collider point. Compute relative velocity in collider
          // space, transform to world space, and subtract from pointVelocityA.
          // Use jacWorldFromCollider = jacColliderFromWorld^T, which is accurate for rigid
          // colliders.
          Real3 velDiff = (posColliding[iContact] - posCollidingStageStart[iContact]) / dtStage;
          velDiff = ToReal3(DotVecMat3x3(ToSimd(velDiff), jacColliderFromWorld));
          Real3 const pointVelocityB = pointVelocityA - velDiff;

          outQueryActiveContacts->contactPoints.push_back(
              ContactPoint{
                  .actorA = collidingActor,
                  .actorB = colliderActor,
                  .distance = distance,
                  .posA = posA,
                  .posB = posB,
                  .normal = normal,
                  .force = force,
                  .pointVelocityA = pointVelocityA,
                  .pointVelocityB = pointVelocityB,
                  .sampleIndex = sampleIndex,
                  .intWeight = weight,
                  .elementIndex = elementIndex,
                  .parametricCoords = basisWeights});
        }
        if (outNodeForces) {
          outNodeForces->AddContactForce(force, nodeIndices, basisWeights);
        }
      }
    };

    // Add contacts from ContactType::Async.
    for (auto const& coll : collisionsAsync) {
      appendActiveContactsForPair(
          e,
          coll.colliderEntity,
          coll.collisionResult,
          outQueryActiveContacts,
          outQueryNodeForces,
          ContactType::Async);
    }

    // Add contacts from ContactType::Sync.
    for (auto const& coll : collisionsSync) {
      appendActiveContactsForPair(
          e,
          coll.colliderEntity,
          coll.collisionResult,
          outQueryActiveContacts,
          outQueryNodeForces,
          ContactType::Sync);
    }

    // Process contacts as collider entity. Not needed for CQueryNodeContactForces
    if (colliderJacs && outQueryActiveContacts) {
      for (auto const& jac : *colliderJacs) {
        // Entities that use "far sdf" queries are not compatible with contact point reporting.
        // If the other actor uses that feature, then do not attempt to report contacts
        // because we won't have the matching forcePerUnitArea data.
        if (reg.any_of<CRequiresFarSdfEvaluation>(jac.otherEntity)) {
          continue;
        }
        // Collider jacs are always from sync contact
        appendActiveContactsForPair(
            jac.otherEntity, e, *jac.query, outQueryActiveContacts, nullptr, ContactType::Sync);
      }
    }
  }

  // Indicate that the data is now ready to be queried
  if (outQueryActiveContacts) {
    outQueryActiveContacts->isInitialized = true;
  }
  if (outQueryNodeForces) {
    outQueryNodeForces->FinalizeContactForces();
  }
}

void mochi::UpdateQueryActorContactForces(
    entt::registry const& reg,
    entt::entity e,
    // System is incompatible with things that require a final SDF pass
    ecs::Excluded<CRequiresFarSdfEvaluation>,
    CContactSamples<TimeStep::Current> const& samples,
    CActiveCollisions<ContactType::Async, TimeStep::Current> const& collisionsAsync,
    CActiveCollisions<ContactType::Sync, TimeStep::Current> const& collisionsSync,
    CCollJacs<CollRole::Collider> const* colliderJacs,
    CRigidState<TimeStep::Current> const* rigidState,
    CConvergenceStatus const* convergenceStatus,
    ecs::OptionalTag<TagRigidActor> hasRigidActorTag,
    CQueryActorContactForces& outQueryActorForces) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(!hasRigidActorTag || rigidState, "Rigid actors must have a rigid state");

  // If the solver diverged, contact data is stale (actor state was rolled back without re-running
  // assembly). Clear query outputs and return.
  if (convergenceStatus && convergenceStatus->stepStatus == ConvergenceStatus::Diverged) {
    outQueryActorForces.Clear();
    outQueryActorForces.isInitialized = true;
    return;
  }

  // Clear output and reserve size
  int size = isize(collisionsAsync) + isize(collisionsSync);
  if (colliderJacs) {
    size += isize(*colliderJacs);
  }
  outQueryActorForces.Clear();
  outQueryActorForces.Reserve(size);

  auto appendActiveContactsForPair = [&](entt::entity otherEntity,
                                         ContactDetectionResult const& collisionResult,
                                         ContactType contactType,
                                         bool isColliderRole) {
    // Derive the colliding entity from the role of `e` in this contact pair.
    entt::entity const colliding = isColliderRole ? otherEntity : e;

    // Validate that the force data is available
    int numContacts = isize(collisionResult.posColliding);
    MOCHI_ASSERT(
        numContacts == collisionResult.forcePerUnitArea.size(),
        "Expected a force vector for every contact between this pair of entities");

    // Define the source entity's samples.
    auto const& collidingSamples =
        !isColliderRole ? samples : reg.get<CContactSamples<TimeStep::Current> const>(colliding);

    // Define the source entity's transform. For soft actors, note that the transform may have been
    // modified by recentering, so use worldFromLocalPrev.
    auto const isCollidingDeformable = reg.all_of<TagDeformableActor>(colliding);
    auto const& collidingRoot = reg.get<CRootTransform const>(colliding);
    auto const& worldFromColliding =
        isCollidingDeformable ? collidingRoot.worldFromLocalPrev : collidingRoot.worldFromLocal;
    VMatrix4x4r const worldFromCollidingMatT = ToVMatrix4x4Transpose(worldFromColliding);

    // Force and torque accumulators
    Vec4r totalForce = {};
    Vec4r totalTorque = {};

    // Define the stride for collider transforms
    int const stride = collisionResult.jacColliderFromWorld.size() > 1 ? 1 : 0;

    // Add contacts to the query data.
    for (int iContact = 0, iTransform = 0; iContact < numContacts;
         ++iContact, iTransform += stride) {
      int sampleIndex = collisionResult.sampleIndices[iContact];

      // Fetch the force and scale by the integration weight.
      Vec4r force = collidingSamples.weights[sampleIndex] *
          ToSimd(collisionResult.forcePerUnitArea[iContact]);

      // Transform the force to world space. For async contact on deformable actors, the force is in
      // local colliding space. Otherwise, the force is in local collider space.
      if (contactType == ContactType::Async && isCollidingDeformable) {
        force = DotVecMat3x3(force, worldFromCollidingMatT);
      } else {
        force = DotVecMat3x3(force, collisionResult.jacColliderFromWorld[iTransform]);
      }

      // Add to the accumulators
      totalForce += force;
      if (hasRigidActorTag) {
        Vec4r const pos = DotVecMat4x4(
            ToSimd(collidingSamples.positions[sampleIndex], 1_r), worldFromCollidingMatT);
        totalTorque += Cross3(pos - rigidState->value.VGetTranslation(), force);
      }
    }

    // Add to the query container, keyed by the other entity. When `e` is the colliding entity, it
    // receives the direct force; when `e` is the collider, it receives the reaction force.
    if (!isColliderRole) {
      outQueryActorForces.Add(otherEntity, totalForce, totalTorque);
    } else {
      outQueryActorForces.Add(otherEntity, -totalForce, -totalTorque);
    }
  };

  // Add contacts from ContactType::Async.
  for (auto const& coll : collisionsAsync) {
    appendActiveContactsForPair(
        coll.colliderEntity, coll.collisionResult, ContactType::Async, /*isColliderRole=*/false);
  }

  // Add contacts from ContactType::Sync.
  for (auto const& coll : collisionsSync) {
    appendActiveContactsForPair(
        coll.colliderEntity, coll.collisionResult, ContactType::Sync, /*isColliderRole=*/false);
  }

  // Process contacts as collider entity.
  if (colliderJacs) {
    for (auto const& jac : *colliderJacs) {
      // Entities that use "far sdf" queries are not compatible with contact point reporting.
      // If the other actor uses that feature, then do not attempt to report contacts
      // because we won't have the matching forcePerUnitArea data.
      if (reg.any_of<CRequiresFarSdfEvaluation>(jac.otherEntity)) {
        continue;
      }
      // Collider jacs are always from sync contact
      appendActiveContactsForPair(
          jac.otherEntity, *jac.query, ContactType::Sync, /*isColliderRole=*/true);
    }
  }

  // Indicate that the data is now ready to be queried
  outQueryActorForces.isInitialized = true;
}

// Take a Span of full DOF indices and appends block indices to the output vector.
// Example with (kBlockSize == 3): [0, 1, 2, 3, 4, 5, 6, 10, 14] --> [0, 1, 2, 3, 4]
template <int kBlockSize>
static void AppendBlockIndices(DynamicArray<int>& outBlockIndices, Span<int const> fullIndices) {
  if constexpr (kBlockSize == 1) {
    Append(outBlockIndices, fullIndices);
  } else {
    int outIdx = isize(outBlockIndices);
    int prevIdx = -1;
    outBlockIndices.resize(outBlockIndices.size() + fullIndices.size()); // worst case
    for (int i : fullIndices) {
      int bi = i / kBlockSize;
      if (bi != prevIdx) {
        outBlockIndices[outIdx++] = bi;
        prevIdx = bi;
      }
    }
    outBlockIndices.resize(outIdx);
  }
};

template <int kBlockSize, ContactType kContactType>
Graph<int, int> mochi::MakeContactGraph(
    entt::registry const& reg,
    Span<entt::entity const> actors) {
  MOCHI_PROFILE_SCOPE();
  using PtrT = int; // Just "Ptr" would alias a type in the global namespace on Mac/iOS
  DynamicArray<PtrT> ptr;
  DynamicArray<int> targetIndices;

  int const kJacsReserveSize = 2 * JacData::kMaxJacs; // 2x for collider & colliding
  int const kJacsReserveSizeInBytes = kJacsReserveSize * sizeof(ContactJac*);
  MOCHI_FILO_STACK_ALLOCATOR(filoAllocator, kJacsReserveSizeInBytes);

  DynamicArray<ContactJac const*> jacs(&filoAllocator);
  jacs.reserve(kJacsReserveSize);

  ptr.push_back(0);
  for (auto e : actors) {
    auto const* activeCollisions =
        reg.try_get<CActiveCollisions<kContactType, TimeStep::Current> const>(e);
    if (!activeCollisions) {
      continue; // No collisions
    }
    // Traverse colliding entities
    for (auto const& coll : *activeCollisions) {
      auto e2 = coll.colliderEntity;
      // Get contact Jacobians
      jacs.clear();
      reg.get<CCollJacs<CollRole::Colliding> const>(e)[coll.collidingJacId].GetJacs(jacs);
      if constexpr (kContactType == ContactType::Sync) {
        reg.get<CCollJacs<CollRole::Collider> const>(e2)[coll.colliderJacId].GetJacs(jacs);
      }
      for (int i0 = 0; i0 < jacs.size(); i0++) {
        ContactJac const& j0 = *jacs[i0];
        if constexpr (kContactType == ContactType::Async) {
          if (j0.hasSharedDoFs) {
            AppendBlockIndices<kBlockSize>(targetIndices, j0.Inds(0));
            ptr.push_back(static_cast<PtrT>(targetIndices.size()));
          } else {
            Span<int const> lastIndices0{};
            for (int i = 0; i < j0.nContacts; ++i) {
              auto indices0 = j0.Inds(i);
              if (indices0 != lastIndices0) {
                AppendBlockIndices<kBlockSize>(targetIndices, indices0);
                ptr.push_back(static_cast<PtrT>(targetIndices.size()));
                lastIndices0 = indices0;
              }
            }
          }
        }
        for (int i1 = i0 + 1; i1 < jacs.size(); i1++) {
          ContactJac const& j1 = *jacs[i1];
          MOCHI_ASSERT(j0.nContacts == j1.nContacts);
          if (j0.hasSharedDoFs && j1.hasSharedDoFs) {
            AppendBlockIndices<kBlockSize>(targetIndices, j0.Inds(0));
            AppendBlockIndices<kBlockSize>(targetIndices, j1.Inds(0));
            ptr.push_back(static_cast<PtrT>(targetIndices.size()));
          } else if (j0.hasSharedDoFs) {
            Span<int const> lastIndices1{};
            auto indices0 = j0.Inds(0);
            for (int i = 0; i < j0.nContacts; ++i) {
              auto indices1 = j1.Inds(i);
              if (indices1 != lastIndices1) {
                AppendBlockIndices<kBlockSize>(targetIndices, indices0);
                AppendBlockIndices<kBlockSize>(targetIndices, indices1);
                ptr.push_back(static_cast<PtrT>(targetIndices.size()));
                lastIndices1 = indices1;
              }
            }
          } else if (j1.hasSharedDoFs) {
            Span<int const> lastIndices0{};
            auto indices1 = j1.Inds(0);
            for (int i = 0; i < j1.nContacts; ++i) {
              auto indices0 = j0.Inds(i);
              if (indices0 != lastIndices0) {
                AppendBlockIndices<kBlockSize>(targetIndices, indices0);
                AppendBlockIndices<kBlockSize>(targetIndices, indices1);
                ptr.push_back(static_cast<PtrT>(targetIndices.size()));
                lastIndices0 = indices0;
              }
            }
          } else {
            Span<int const> lastIndices0{};
            Span<int const> lastIndices1{};
            for (int i = 0; i < j0.nContacts; ++i) {
              auto indices0 = j0.Inds(i);
              auto indices1 = j1.Inds(i);
              if (!((indices0 == lastIndices0) && (indices1 == lastIndices1))) {
                AppendBlockIndices<kBlockSize>(targetIndices, indices0);
                AppendBlockIndices<kBlockSize>(targetIndices, indices1);
                ptr.push_back(static_cast<PtrT>(targetIndices.size()));
                lastIndices0 = indices0;
                lastIndices1 = indices1;
              }
            }
          }
        }
      }
    }
  }

  auto cToN = Graph<int, int>{std::move(ptr), std::move(targetIndices)};
  return Traverse(Reverse(cToN), cToN).SortTargets();
}

// Templates externed in mochi_contact.h
template Graph<int, int> mochi::MakeContactGraph<1, ContactType::Async>(
    entt::registry const& reg,
    Span<entt::entity const> actors);
template Graph<int, int> mochi::MakeContactGraph<3, ContactType::Async>(
    entt::registry const& reg,
    Span<entt::entity const> actors);

template <GradTarget kGradTarget>
static void AssembleAllSyncContactPairs(
    entt::registry const& reg,
    ecs::PartialRegistry<CActiveCollisions<ContactType::Sync, TimeStep::Current>> regActiveColls,
    Span<entt::entity const> actors,
    Allocator* filoAllocator,
    double* outObj,
    ColumnVectorView<real> outRes,
    AnyMatrixView<real> outDRes,
    bool psdDRes,
    bool useFittedHessian) {
  MOCHI_PROFILE_SCOPE();
  bool const assemObj = (outObj != nullptr);
  bool const assemRes = !outRes.empty();
  bool const assemDRes = (GetNumValues(outDRes) > 0);

  // This function can be VERY expensive. The interaction matrix can have many thousands of non-zero
  // values, even for a single object contacting the hands. Most of the cost comes from from adding
  // the values into the SparseMatrix, not from computing those values.
  //
  // For the future:
  //  Ideally, we wouldn't assemble contact into a sparse matrix at all. The matrix outDRes is just
  //  intermediate storage. Ideally, we would store dense blocks of on-diagonal values, and dense
  //  blocks of off-diagonal values. The linear solve could use those blocks directly, or it could
  //  efficiently merge them into a global matrix.
  //
  // For now:
  //  This code subdivides the work. It is dumb. It doesn't take advantage of the fact that some
  //  contacts affect non-overlapping DOF ranges. It duplicates the entire dresidual value array for
  //  every task (except one). It adds them at the end, even though many of the values will be zero.
  //  This is inefficient, but still much faster than doing it all on one thread.

  struct ContactPair {
    entt::entity entity0 = {};
    ActiveCollision* collision = nullptr;
    Interval<int> jacRange;
    double costPerPoint = 0;
  };

  // Start by enumerating the contacting pairs and their ContactJacs
  DynamicArray<ContactJac const*> allJacs;
  DynamicArray<ContactPair> allPairs(filoAllocator);
  allJacs.reserve(2 * JacData::kMaxJacs * isize(actors) * isize(actors)); // worst case
  allPairs.reserve(isize(actors) * isize(actors)); // worst case
  for (auto entity0 : actors) {
    if (auto* activeCollisions =
            regActiveColls.try_get<CActiveCollisions<ContactType::Sync, TimeStep::Current>>(
                entity0)) {
      for (auto& coll : *activeCollisions) {
        int firstJac = isize(allJacs);
        if (assemRes || assemDRes) {
          // For each contact pair, colliding Jacobian(s) must go before collider Jacobian(s).
          entt::entity entity1 = coll.colliderEntity;
          reg.get<CCollJacs<CollRole::Colliding> const>(entity0)[coll.collidingJacId].GetJacs(
              allJacs);
          reg.get<CCollJacs<CollRole::Collider> const>(entity1)[coll.colliderJacId].GetJacs(
              allJacs);
        }
        ContactPair pair;
        pair.entity0 = entity0;
        pair.collision = &coll;
        pair.jacRange = {firstJac, isize(allJacs)};
        allPairs.emplace_back(pair);
      }
    }
  }
  if (allPairs.empty()) {
    return; // No contact
  }

  // Estimate the cost of each pair.
  double totalCost = 0;
  for (auto& cp : allPairs) {
    int numContacts = isize(cp.collision->collisionResult.sampleIndices);
    if (assemObj) {
      cp.costPerPoint += 1;
    }
    if (assemRes) { // linear cost
      for (int i : cp.jacRange) {
        double alpha = allJacs[i]->hasSharedDoFs ? 0 : 2.5;
        cp.costPerPoint += allJacs[i]->nDoFsInternal * (1 + alpha);
      }
    }
    if (assemDRes) { // quadratic cost
      for (int a : cp.jacRange) {
        double alpha = allJacs[a]->hasSharedDoFs ? 0 : 5;
        int nDofsInternalA = allJacs[a]->nDoFsInternal;
        int nDofsStateA = allJacs[a]->nDoFsState;
        cp.costPerPoint += nDofsInternalA * nDofsInternalA * (1 + alpha);
        for (int b = cp.jacRange.Min(); b < a; ++b) {
          double beta = allJacs[a]->hasSharedDoFs && allJacs[b]->hasSharedDoFs ? 1 : 0;
          int nDofsInternalB = allJacs[b]->nDoFsInternal;
          int nDofsStateB = allJacs[b]->nDoFsState;
          cp.costPerPoint += nDofsInternalA * nDofsInternalB * beta;
          cp.costPerPoint += nDofsStateA * nDofsStateB * (1 - beta + alpha);
        }
      }
    }
    totalCost += numContacts * cp.costPerPoint;
  }

  // Determine the target number of tasks. The actual number of tasks may be smaller than the
  // target, depending on how the cost is distributed across contact pairs. If we use too many
  // tasks, it will be diminishing returns because of the high cost of duplicating the value
  // arrays and adding them back together.
  // TODO: Retune parallelization. It seems to over-parallelize.
  int constexpr kTargetCostPerTask = 10000;
  int constexpr kMaxTasks = 12;
  int const maxTasks = Min(kMaxTasks, TaskScheduler::StaticGetNumOtherThreads() + 1);
  int const numTargetTasks = Min(static_cast<int>(Ceil(totalCost / kTargetCostPerTask)), maxTasks);

  // Work to be done for a ContactPair
  struct PairWork {
    PairWork(ContactPair* contactPair_, Interval<int> pointRange_)
        : contactPair(contactPair_), pointRange(pointRange_) {}
    ContactPair* contactPair = nullptr;
    Interval<int> pointRange;
    DynamicArray<Real3> outForce; // May be used for contact queries
  };

  // Work to be done by each task
  struct TaskWork {
    DynamicArray<PairWork> pairWork;
    double objective = 0.0;
    ColumnVector<real> resValues;
    ColumnVector<real> dresValues;
    double cost = 0;
  };

  // Now, split the work into tasks of roughly equal cost.
  DynamicArray<TaskWork> taskWork(filoAllocator);
  taskWork.reserve(kMaxTasks);
  double const targetCostPerTask = totalCost / numTargetTasks;
  for (auto& cp : allPairs) {
    Interval<int> pointsRemaining{0, isize(cp.collision->collisionResult.sampleIndices)};
    while (pointsRemaining.Size() > 0) {
      if (taskWork.empty() || taskWork.back().cost >= targetCostPerTask) {
        taskWork.push_back({});
      }
      auto& currentTaskWork = taskWork.back();
      double const taskRemainingCost = targetCostPerTask - currentTaskWork.cost;
      int const numPointsToAdd =
          Min(static_cast<int>(Ceil(taskRemainingCost / cp.costPerPoint)), pointsRemaining.Size());
      MOCHI_ASSERT(numPointsToAdd > 0 && numPointsToAdd <= pointsRemaining.Size());
      Interval<int> subRange{pointsRemaining.Min(), pointsRemaining.Min() + numPointsToAdd};
      currentTaskWork.pairWork.emplace_back(&cp, subRange);
      currentTaskWork.cost += numPointsToAdd * cp.costPerPoint;
      pointsRemaining.Min() += numPointsToAdd;
    }
  }

  // Do part of the work
  auto doTask = [&](int iTask) {
    auto& myTask = taskWork[iTask];
    double* taskObj = outObj;
    ColumnVectorView<real> taskRes = outRes;
    AnyMatrixView<real> taskDRes = outDRes;

    if (iTask == 0) {
      // The first task will assemble directly to the output residual/matrix
    } else {
      MOCHI_PROFILE_SCOPE_N("Init Task Data");
      // Other threads will allocate their own values arrays for res & dres
      if (assemObj) {
        taskObj = &myTask.objective;
      }
      if (assemRes) {
        myTask.resValues.Reset(outRes.size());
        taskRes.Reset(AsView(myTask.resValues));
        taskRes.SetZero();
      }
      if (assemDRes) {
        myTask.dresValues.Reset(ColumnVector<real>::Zero(GetNumValues(outDRes)));
        std::visit(
            OverloadVisitor{
                // Use emplace<> to ensure the constructor of the desired matrix view type (and not
                // the move assignment) is called.
                [&](MatrixView<real> dense) {
                  taskDRes.emplace<MatrixView<real>>(
                      myTask.dresValues.data(), dense.Rows(), dense.Cols());
                },
                [&](SparseMatrixView<real> sp) {
                  taskDRes.emplace<SparseMatrixView<real>>(
                      sp.Cols(), sp.Pointers(), sp.Indices(), myTask.dresValues.GetSpan());
                },
                [&](BlockSparseMatrixView<real, 3> bsp) {
                  taskDRes.emplace<BlockSparseMatrixView<real, 3>>(
                      bsp.Cols(), bsp.Pointers(), bsp.Indices(), myTask.dresValues.GetSpan());
                },
                [&](BlockSparseMatrixView<real, 4> bsp) {
                  taskDRes.emplace<BlockSparseMatrixView<real, 4>>(
                      bsp.Cols(), bsp.Pointers(), bsp.Indices(), myTask.dresValues.GetSpan());
                }},
            outDRes);
      }
    }

    // Preallocate memory for collision response. Use stack memory if possible. 32 KiB is enough in
    // most cases.
    MOCHI_FILO_STACK_ALLOCATOR(taskLocalAlloc, 32 * 1024);

    CollisionResponseResult response(&taskLocalAlloc);
    int maxContacts = 0;
    for (auto const& pairWork : taskWork[iTask].pairWork) {
      maxContacts =
          Max(isize(pairWork.contactPair->collision->collisionResult.sampleIndices), maxContacts);
    }
    response.ResizeNoInit(maxContacts, assemObj, assemRes, assemDRes);

    // Contact evaluation settings
    auto const& configScene = reg.ctx<CSimulationParams const>().experimentalEval;
    ContactEvalConfig configAllPairs{
        .psdDRes = psdDRes,
        .explicitNormals = configScene.explicitNormals,
        .fadeFriction = configScene.fadeFriction,
        .implicitNormalForceForDissipation = configScene.implicitNormalForceForDissipation,
        .useFittedHessian = useFittedHessian,
        .frictionModel = configScene.frictionModel};

    // Loop over all pairs.
    for (auto& pairWork : taskWork[iTask].pairWork) {
      auto const& pair = *pairWork.contactPair;
      auto entity0 = pair.entity0;
      auto entity1 = pair.collision->colliderEntity;
      int numContacts = isize(pair.collision->collisionResult.sampleIndices);
      real const dtStage = reg.get<CTimeIntegratorState const>(entity0).dtStage;
      auto const& samples = reg.get<CContactSamples<TimeStep::Current> const>(entity0);

      // Contact evaluation settings
      ContactEvalConfig configPair = configAllPairs;
      configPair.addPadding = ShouldAddPenaltyPadding(reg.get<CColliderInfo const>(entity0).type);
      configPair.validCollidingNormals = ValidCollidingNormals(reg, entity0);

      // Compute collision response
      auto const& query = pair.collision->collisionResult;
      auto contactParams = GetContactPairParams(reg, entity0, entity1);
      response.ResizeNoInit(numContacts, assemObj, assemRes, assemDRes);
      ComputeCollisionResponseRange<kGradTarget>(
          pairWork.pointRange,
          query,
          contactParams,
          configPair,
          dtStage,
          assemObj,
          assemRes,
          assemDRes,
          response);

      // Perform assembly
      auto jacs = (pair.jacRange.size() > 0)
          ? Span{&allJacs[pair.jacRange.Min()], pair.jacRange.size()}
          : Span<ContactJac const*>{};
      bool const isSyncRigid =
          reg.all_of<TagRigidActor>(entity0) && reg.all_of<TagRigidActor>(entity1);
      AssembleCollisionResponseRange<kGradTarget>(
          reg,
          entity0,
          entity1,
          pairWork.pointRange,
          query,
          response,
          samples.weights,
          jacs,
          &taskLocalAlloc,
          taskObj,
          taskRes,
          taskDRes,
          isSyncRigid);

      // Store forces if we need to return them.
      if (assemRes &&
          (reg.all_of<TagQueryActiveContacts>(entity0) ||
           reg.all_of<TagQueryActiveContacts>(entity1))) {
        pairWork.outForce.reserve(response.force.size());
        pairWork.outForce.clear();
        pairWork.outForce.resize(pairWork.pointRange.Min(), Real3{});
        pairWork.outForce.append(
            response.force.begin() + pairWork.pointRange.Min(),
            response.force.begin() + pairWork.pointRange.Max() + 1);
        pairWork.outForce.resize(response.force.size(), Real3{});
      }
    }
  };

  // GO!
  ParallelForN("AssemblePairs", isize(taskWork), 1, doTask);

  // Add the results
  {
    MOCHI_PROFILE_SCOPE_N("Accumulate");
    for (int i = 1; i < isize(taskWork); ++i) {
      if (assemObj) {
        *outObj += taskWork[i].objective;
      }
      if (assemRes) {
        outRes += taskWork[i].resValues;
      }
      if (assemDRes) {
        AsView(GetValues(outDRes)) += taskWork[i].dresValues;
      }
    }
  }

  // Store the forces in case they are needed for a later contact query
  if (assemRes) {
    MOCHI_PROFILE_SCOPE_N("Output Forces");
    for (auto& work : taskWork) {
      for (auto& pair : work.pairWork) {
        auto entity0 = pair.contactPair->entity0;
        auto entity1 = pair.contactPair->collision->colliderEntity;
        if (!reg.all_of<TagQueryActiveContacts>(entity0) &&
            !reg.all_of<TagQueryActiveContacts>(entity1)) {
          continue;
        }

        auto& query = pair.contactPair->collision->collisionResult;
        if (pair.pointRange.Min() == 0) {
          query.forcePerUnitArea = std::move(pair.outForce);
        } else {
          // This ContactPair must have been divided between tasks. Add the forces.
          MOCHI_ASSERT((query.forcePerUnitArea.size() == pair.outForce.size()));
          for (int i = 0; i < isize(query.forcePerUnitArea); ++i) {
            query.forcePerUnitArea[i] += pair.outForce[i];
          }
        }
      }
    }
  }
}

template <ContactType kContactType>
static bool ActorsHaveContact(entt::registry const& reg, Span<entt::entity const> actors) {
  MOCHI_PROFILE_SCOPE();
  for (auto e : actors) {
    if (auto const* activeCollisions =
            reg.try_get<CActiveCollisions<kContactType, TimeStep::Current> const>(e)) {
      for (auto const& col : *activeCollisions) {
        if (!col.collisionResult.Empty()) {
          return true;
        }
      }
    }
  }
  return false;
}

void mochi::AssembleIslandSyncContact(
    AssemblyParams const& params,
    bool useBlockSparse3x3,
    entt::registry& reg,
    CIslandDofInfo const& islandDofInfo,
    CIslandDescendants const& descendants,
    CIslandContactSnle& outContactSnle) {
  MOCHI_PROFILE_SCOPE();

  // If this is an input grad target, all we need to do is resize the residual to 0
  auto const gradTarget = params.gradTarget;
  if (gradTarget == GradTarget::CurrentInput || gradTarget == GradTarget::PreviousInput) {
    MOCHI_ASSERT_VERBOSE(
        !params.assemObj && params.assemRes && !params.assemDRes, "Invalid request");
    outContactSnle.residuals.resize(1);
    outContactSnle.residuals[0].second.Resize(0);
    return;
  }

  auto const& actors = descendants.actors;
  if (!ActorsHaveContact<ContactType::Sync>(reg, actors)) {
    outContactSnle.useInSolver = false;
    return;
  }

  outContactSnle.residuals.resize(1);
  outContactSnle.dresiduals.resize(1);
  auto& outContactResidual = outContactSnle.residuals[0].second;
  auto& outContactDResidual = outContactSnle.dresiduals[0].matrix;

  if (params.assemRes) {
    // Make sure the contact residual is the right size
    // TODO: A sparse vector would be more efficient.
    outContactResidual.Resize(islandDofInfo.dofsSize);
  }

  if (params.assemDRes) {
    // Build a dresidual matrix based on the DOFs that are actually affected by contact
    MOCHI_PROFILE_SCOPE_N("Make Contact Matrix");
    if (useBlockSparse3x3) {
      outContactDResidual =
          BlockSparseMatrix<real, 3>{MakeContactGraph<3, ContactType::Sync>(reg, actors)};
    } else {
      outContactDResidual = SparseMatrix<real>{MakeContactGraph<1, ContactType::Sync>(reg, actors)};
    }
    MOCHI_ASSERT_VERBOSE(
        GetNumValues(outContactDResidual) > 0,
        "If there was no sync contact in the island, then we shouldn't have gotten this far.");
  }

  outContactSnle.SetZero(params);
  outContactSnle.useInSolver = true;

  // Allocator for temporary memory throughout the assembly process.
  // Uses stack memory first, but can fall back on heap memory if needed.
  MOCHI_FILO_STACK_ALLOCATOR(filoAllocator, 128 * 1024);

  // Dispatch to the appropriate templatized assembly based on GradTarget.
  MOCHI_ASSERT_VERBOSE(
      params.gradTarget == GradTarget::Current || params.gradTarget == GradTarget::Previous,
      "Unsupported gradient target");
  auto assembleAllSyncContactPairsFn = params.gradTarget == GradTarget::Previous
      ? AssembleAllSyncContactPairs<GradTarget::Previous>
      : AssembleAllSyncContactPairs<GradTarget::Current>;
  assembleAllSyncContactPairsFn(
      reg,
      reg,
      actors,
      &filoAllocator,
      params.assemObj ? &outContactSnle.objective : nullptr,
      params.assemRes ? AsView(outContactResidual) : ColumnVectorView<real>{},
      params.assemDRes ? AsView(outContactDResidual) : AnyMatrixView<real>{},
      params.psdDRes,
      params.fittedSaturationHessian.contactFriction);
}

void mochi::AssembleAsyncSkinnedContact(
    AssemblyParams const& params,
    bool useBlockSparse3x3,
    ecs::RequiredTag<TagSkinnedContact>,
    ecs::OptionalTag<TagQueryActiveContacts> queryActiveContacts,
    entt::registry const& reg,
    entt::entity e,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& activeCollisions,
    CIslandMemberInfo const& islandMember,
    CTimeIntegratorState const& intState,
    CContactSamples<TimeStep::Current> const& samples,
    CColliderInfo const& colliderInfo,
    CCollJacs<CollRole::Colliding> const& collJacs,
    CSkinnedContactSnle& outContactSnle) {
  MOCHI_PROFILE_SCOPE();

  MOCHI_ASSERT_VERBOSE(params.gradTarget == GradTarget::Current, "Unexpected grad target");

  // Skip if there are no contacts at all
  bool hasContact = false;
  for (auto const& coll : activeCollisions) {
    if (!coll.collisionResult.Empty()) {
      hasContact = true;
      break;
    }
  }
  if (!hasContact) {
    outContactSnle.useInSolver = false;
    return;
  }

  // Set up the output SNLE
  outContactSnle.residuals.resize(1);
  outContactSnle.dresiduals.resize(1);
  auto& outContactResidual = outContactSnle.residuals[0].second;
  auto& outContactDResidual = outContactSnle.dresiduals[0].matrix;

  if (params.assemRes) {
    // TODO: Skinned contact should assemble directly into this entity's reduced CActorSnle, or to
    // another residual/dresidual of the same size. For now, it produces an interactionResidual
    // and interactionMatrix for the island's SNLE problem, so they need to be the size of the
    // island.
    auto const& islandDofInfo = reg.get<CIslandDofInfo>(islandMember.island);

    // Resize the contact residual if necessary
    if (outContactResidual.size() != islandDofInfo.dofsSize) {
      outContactResidual.Reset(islandDofInfo.dofsSize);
    }
  }

  if (params.assemDRes) {
    // Compute sparsity based on the DOFs that are actually affected by explicit contact.
    if (useBlockSparse3x3) {
      outContactDResidual = BlockSparseMatrix<real, 3>{
          MakeContactGraph<3, ContactType::Async>(reg, MakeSingletonConstSpan(e))};
    } else {
      outContactDResidual = SparseMatrix<real>{
          MakeContactGraph<1, ContactType::Async>(reg, MakeSingletonConstSpan(e))};
    }
    MOCHI_ASSERT(
        GetNumValues(outContactDResidual) > 0,
        "If there was no async contact with this actor, then we shouldn't have gotten this far.");
  }

  outContactSnle.SetZero(params);
  outContactSnle.useInSolver = true;

  // Configure common containers for all colliders
  MOCHI_FILO_STACK_ALLOCATOR(filoAllocator, 64 * 1024);
  CollisionResponseResult response(&filoAllocator);
  response.Reserve(activeCollisions, params.assemObj, params.assemRes, params.assemDRes);
  DynamicArray<ContactJac const*> jacs(&filoAllocator);
  jacs.reserve(JacData::kMaxJacs);
  ContactEvalConfig config{
      .psdDRes = params.psdDRes,
      .addPadding = ShouldAddPenaltyPadding(colliderInfo.type),
      .validCollidingNormals = ValidCollidingNormals(reg, e),
      .explicitNormals = simParams->experimentalEval.explicitNormals,
      .fadeFriction = simParams->experimentalEval.fadeFriction,
      .implicitNormalForceForDissipation =
          simParams->experimentalEval.implicitNormalForceForDissipation,
      .useFittedHessian = params.fittedSaturationHessian.contactFriction,
      .frictionModel = simParams->experimentalEval.frictionModel};

  // Traverse colliding entities and assemble to outContactSnle
  for (auto& coll : activeCollisions) {
    // Compute collision response
    auto const& query = coll.collisionResult;
    int const numPoints = isize(query.sampleIndices);
    response.ResizeNoInit(numPoints, params.assemObj, params.assemRes, params.assemDRes);
    auto contactParams = GetContactPairParams(reg, e, coll.colliderEntity);
    ComputeCollisionResponse<GradTarget::Current>(
        query,
        contactParams,
        config,
        intState.dtStage,
        params.assemObj,
        params.assemRes,
        params.assemDRes,
        response);

    // Get contact Jacobians
    jacs.clear();
    collJacs[coll.collidingJacId].GetJacs(jacs);

    // Perform assembly.
    AssembleCollisionResponse(
        reg,
        e,
        coll.colliderEntity,
        query,
        response,
        samples.weights,
        jacs,
        &filoAllocator,
        params.assemObj ? &outContactSnle.objective : nullptr,
        params.assemRes ? AsView(outContactResidual) : ColumnVectorView<real>{},
        params.assemDRes ? AsView(outContactDResidual) : AnyMatrixView<real>{},
        /*isSyncRigid*/ false);

    // Store the forces in case they are needed for a later contact point query
    if (params.assemRes && queryActiveContacts) {
      coll.collisionResult.forcePerUnitArea = response.force;
    }
  } // for (auto& coll : activeCollisions)
}

Aabb mochi::ExpandConservativeBoundsWithContactPadding(
    Aabb bounds,
    ecs::PartialRegistry<
        CContactParams const,
        CRequiresFarSdfEvaluation const,
        CPointCloudColliderParams const> reg,
    entt::entity e) {
  auto const* contactParams = reg.try_get<CContactParams const>(e);
  real farSdfDistance = GetFarSdfEvaluationDistance<true>(reg, e);

  // Pad with: (1) contact params (for colliders), (2) point-cloud collider radius (for
  // point-cloud colliders such as shell/rods), and (3) far-SDF distance (for collidees).
  if (contactParams) {
    bounds = ExpandColliderBoundsForContact(bounds, *contactParams);
  }

  if (auto const* pcParams = reg.try_get<CPointCloudColliderParams const>(e)) {
    bounds = ExpandShape(bounds, pcParams->radius);
  }

  return ExpandShape(bounds, farSdfDistance);
}

void mochi::CheckConservativeStepBounds(entt::registry const& reg, entt::entity e) {
  MOCHI_PROFILE_SCOPE();

  // This function only logs warnings.
  if (!IsLogChannelEnabled(mochi::LogChannel::Warning)) {
    return;
  }

  if (reg.all_of<TagIsland>(e)) {
    auto const& descendants = reg.get<CIslandDescendants const>(e).actors;
    for (entt::entity child : descendants) {
      CheckConservativeStepBounds(reg, child);
    }
    return;
  }

  auto const* csb = reg.try_get<CConservativeStepBounds const>(e);
  if (csb) {
    // Any actor with CConservativeStepBounds should also have these:
    auto const& root = reg.get<CRootTransform const>(e);
    auto const& bounds = reg.get<CBoundingVolume<TimeStep::Current>>(e);

    // Compute the current world bounds
    Aabb worldAabb = GetAabb(TransformShape(root.worldFromLocal, bounds.localShape));
    worldAabb = ExpandConservativeBoundsWithContactPadding(worldAabb, reg, e);

    bool inBounds = AllTrue<3>(
        (worldAabb.VGetMin() >= csb->worldAabb.VGetMin()) &
        (worldAabb.VGetMax() <= csb->worldAabb.VGetMax()));

    if (!inBounds) {
      [[maybe_unused]] real maxErrorDistance = HMax<3>(
          Max(csb->worldAabb.VGetMin() - worldAabb.VGetMin(),
              worldAabb.VGetMax() - csb->worldAabb.VGetMax()));

      [[maybe_unused]] auto const* actorInfo = reg.try_get<CActorInfo const>(e);

      // TODO: Use MOCHI_LOG_WARNING here so that it causes unit tests to fail.
      //       It isn't a critical failure yet because:
      //
      //        1) All our current samples and demos force actors into the same compound, and thus
      //           the same island. That island will not be dynamically split even if
      //           CConservativeStepBounds was wrong.
      //
      //        2) We still recompute bounds and CPotentialColliders for every actor every time we
      //           assemble contact. Thus, contact assembly will still work even if
      //           CConservativeStepBounds was wrong.
      //
      MOCHI_LOG(
          "[WARNING] Actor %u (\"%s\") has moved outside of the expected CConservativeStepBounds by a "
          "distance of %g. Island culling may be incorrect.",
          e,
          actorInfo ? actorInfo->name.c_str() : " no name",
          maxErrorDistance);
    }
  }
}

void mochi::LogSdfColliderDiagnostics(SdfCollider const& collider, std::string const& actorName) {
  auto const* gridSdf = dynamic_cast<GridSdf const*>(collider.shape.get());
  if (!gridSdf) {
    return;
  }
  auto const& res = gridSdf->GetCellResolution();
  if (Min(res) < 6) {
    MOCHI_LOG_WARNING(
        "Actor '%s': SDF grid has very low resolution (%d x %d x %d cells).\n"
        "  This may cause inaccurate collision detection.\n"
        "  - For simple shapes like boxes or spheres, consider using ColliderType::Box or ColliderType::Sphere instead.\n"
        "  - For complex shapes, consider increasing the SDF grid resolution (see GridSdfParams documentation for details).",
        actorName.c_str(),
        res[0],
        res[1],
        res[2]);
  }
  if (!gridSdf->IsMeshClosed()) {
    MOCHI_LOG_WARNING(
        "Actor '%s': The collider mesh is not topologically closed (not a closed surface).\n"
        "  The SDF sign may be incorrect, causing unreliable collision detection near the open boundaries.",
        actorName.c_str());
  }
}

void mochi::LogMeshColliderDiagnostics(MeshCollider const& collider, std::string const& actorName) {
  if (!collider.IsMeshClosed()) {
    MOCHI_LOG_WARNING(
        "Actor '%s': The collider mesh is not topologically closed (not a closed surface).\n"
        "  Collision detection may be unreliable near the open boundaries.",
        actorName.c_str());
  }
}

void mochi::WaitForPendingSdfCollider(
    mochi::CActorInfo const& actorInfo,
    mochi::CSdfColliderPending& pendingComp,
    mochi::CSdfCollider& finalComp) {
  pendingComp.gridSdfShape->GetGridSdfSemaphore().Wait();
  finalComp.shape = pendingComp.gridSdfShape->GetGridSdf();
  LogSdfColliderDiagnostics(finalComp, actorInfo.name);
}

// Global context component used in this cpp file to reduce re-allocation of temporary data storage.
namespace {
struct CTempPotentialColliderData {
  std::vector<entt::entity> staticColliders;
  std::vector<Aabb> staticBounds;
  std::vector<entt::entity> dynamicEntities;
  std::vector<Aabb> dynamicBounds;
  std::vector<uint8_t> dynamicHasCollider;
};
} // namespace

namespace mochi::contact {
void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CActiveCollisions<ContactType::Async, TimeStep::Current>>(reg);
  ecs::RegisterComponent<CActiveCollisions<ContactType::Async, TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CActiveCollisions<ContactType::Sync, TimeStep::Current>>(reg);
  ecs::RegisterComponent<CActiveCollisions<ContactType::Sync, TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CBoxCollider>(reg);
  ecs::RegisterComponent<CColliderInfo>(reg);
  ecs::RegisterComponent<CCollJacs<CollRole::Collider>>(reg);
  ecs::RegisterComponent<CCollJacs<CollRole::Colliding>>(reg);
  ecs::RegisterComponent<CContactPartitions>(reg);
  ecs::RegisterComponent<CContactSamples<TimeStep::Current>>(reg);
  ecs::RegisterComponent<CContactSamples<TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CContactCorrespondence<ContactType::Async>>(reg);
  ecs::RegisterComponent<CContactCorrespondence<ContactType::Sync>>(reg);
  ecs::RegisterComponent<CMeshCollider>(reg);
  ecs::RegisterComponent<CPlaneCollider>(reg);
  ecs::RegisterComponent<CDeformablePointAsyncCollisionsResponse>(reg);
  ecs::RegisterComponent<CConservativePotentialColliders<ContactType::Async>>(reg);
  ecs::RegisterComponent<CConservativePotentialColliders<ContactType::Sync>>(reg);
  ecs::RegisterComponent<CPotentialColliders<ContactType::Async>>(reg);
  ecs::RegisterComponent<CPotentialColliders<ContactType::Sync>>(reg);
  ecs::RegisterComponent<CPrevRigidVelocity>(reg);
  ecs::RegisterComponent<CRequiresFarSdfEvaluation>(reg);
  ecs::RegisterComponent<CSdfCollider>(reg);
  ecs::RegisterComponent<CSdfColliderPending>(reg);
  ecs::RegisterComponent<CSdfMapping<TimeStep::Current>>(reg);
  ecs::RegisterComponent<CSdfMapping<TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CSphereCollider>(reg);

  // Global Context
  reg.set<CContactFilterTable>();
  auto& data = reg.set<CTempPotentialColliderData>();
  int constexpr kReserveSize = 128; // Reserve memory for a modest number of actors up front
  data.staticColliders.reserve(kReserveSize);
  data.staticBounds.reserve(kReserveSize);
  data.dynamicEntities.reserve(kReserveSize);
  data.dynamicBounds.reserve(kReserveSize);
  data.dynamicHasCollider.reserve(kReserveSize);
}
} // namespace mochi::contact

void mochi::contact::UpdateConservativePotentialColliders(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();

  // Contact filtering
  auto const& contactTable = reg.ctx<CContactFilterTable const>();

  // Recycled memory storage
  auto& ctx = reg.ctx<CTempPotentialColliderData>();
  auto& staticColliders = ctx.staticColliders;
  auto& staticBounds = ctx.staticBounds;
  auto& dynamicEntities = ctx.dynamicEntities;
  auto& dynamicBounds = ctx.dynamicBounds;
  auto& dynamicHasCollider = ctx.dynamicHasCollider;
  staticColliders.clear();
  staticBounds.clear();
  dynamicEntities.clear();
  dynamicBounds.clear();
  dynamicHasCollider.clear();

  // Clear CConservativePotentialColliders for all actors
  reg.view<CConservativePotentialColliders<ContactType::Async>>().each(
      [](auto& colliders) { colliders.clear(); });
  reg.view<CConservativePotentialColliders<ContactType::Sync>>().each(
      [](auto& colliders) { colliders.clear(); });

  // Find all static colliders
  // TODO: These rarely change. We could cache this information and update it incrementally.
  for (auto&& [e, colliderInfo, root, bounds, contactParams] :
       reg.view<
              TagStaticActor,
              CColliderInfo const,
              CRootTransform const,
              CBoundingVolume<TimeStep::Previous> const,
              CContactParams const>()
           .each()) {
    if (colliderInfo.type != ColliderType::None) {
      staticColliders.push_back(e);
      staticBounds.push_back(ExpandColliderBoundsForContact(
          GetAabb(TransformShape(root.worldFromLocalPrev, bounds.localShape)), contactParams));
    }
  }

  // Find all dynamic actors that participate in contact (colliding and/or collider)
  for (auto&& [e, bounds, membership, layer] :
       reg.view<CConservativeStepBounds const, CIslandMemberInfo const, CContactLayer const>(
              entt::exclude_t<TagStaticActor>())
           .each()) {
    auto const* cinfo = reg.try_get<CColliderInfo>(e);
    // NOTE: Point-cloud colliders are excluded here, since they only collide with each other (and
    // potentially themselves). Their conservative potential colliders are updated separately.
    bool hasCollider =
        cinfo && (cinfo->type != ColliderType::None) && (cinfo->type != ColliderType::PointCloud);
    dynamicEntities.push_back(e);
    dynamicBounds.push_back(bounds.worldAabb);
    dynamicHasCollider.push_back(hasCollider ? 1 : 0);
  }

  // Find pairs of overlapping actors and update CConservativePotentialColliders
  CConservativePotentialColliders<ContactType::Async>* iAsync = nullptr;
  CConservativePotentialColliders<ContactType::Sync>* iSync = nullptr;
  int const numDynamic = isize(dynamicEntities);
  int const numStatic = isize(staticColliders);
  for (int i = 0; i < numDynamic; ++i) {
    entt::entity ie = dynamicEntities[i];
    auto iLayer = reg.get<CContactLayer>(ie).id;
    bool iUseContact = reg.all_of<TagUseContact>(ie);
    if (iUseContact) {
      iAsync = &reg.get<CConservativePotentialColliders<ContactType::Async>>(ie);
      iSync = &reg.get<CConservativePotentialColliders<ContactType::Sync>>(ie);

      // Find static actors that overlap entity ie.
      for (int j = 0; j < numStatic; ++j) {
        if (HasOverlap(dynamicBounds[i], staticBounds[j])) {
          entt::entity je = staticColliders[j];
          ContactLayerId jLayer = reg.get<CContactLayer>(je).id;
          if (contactTable.IsContactEnabled(ie, je, iLayer, jLayer)) {
            MOCHI_ASSERT_VERBOSE(iAsync != nullptr);
            iAsync->emplace_back(je); // NOLINT(facebook-hte-NullableDereference)
          }
        }
      }
    }

    // Iterate over all other dynamic actors, starting at (i + 1).
    // They may contact entity ie even if it doesn't contact them.
    for (int j = i + 1; j < numDynamic; ++j) {
      if (HasOverlap(dynamicBounds[i], dynamicBounds[j])) {
        entt::entity je = dynamicEntities[j];
        ContactLayerId jLayer = reg.get<CContactLayer>(je).id;

        // Consider i-vs-j
        if (iUseContact && dynamicHasCollider[j]) {
          if (contactTable.IsContactEnabled(ie, je, iLayer, jLayer)) {
            MOCHI_ASSERT_VERBOSE(iSync != nullptr);
            iSync->emplace_back(je); // NOLINT(facebook-hte-NullableDereference)
          }
        }

        // Consider j-vs-i
        if (reg.all_of<TagUseContact>(je) && dynamicHasCollider[i]) {
          if (contactTable.IsContactEnabled(je, ie, jLayer, iLayer)) {
            auto& jSync = reg.get<CConservativePotentialColliders<ContactType::Sync>>(je);
            jSync.emplace_back(ie);
          }
        }
      }
    }
  }

  // All point-cloud colliders with overlapping conservative step bounds are potential colliders
  // with each other (including themselves, if self-contact is active), and point-cloud colliders
  // with overlapping conservative step bounds must be grouped into the same island. Additionally,
  // point-cloud colliders cannot be colliders for any other type of actor. These potential collider
  // relationships are excluded from the above process and added separately here, to enforce the
  // specialized logic for point-cloud collisions.
  reg.view<TagUsePointCloudContact const, CConservativeStepBounds const>().each(
      [&](entt::entity collidingEntity, CConservativeStepBounds const& collidingStepBounds) {
        auto const& collidingLayer = reg.get<CContactLayer>(collidingEntity);
        auto const& collidingPointCloudColliderParams =
            reg.get<CPointCloudColliderParams>(collidingEntity);
        // Only sync contact is supported for point-cloud collisions.
        auto& potentialColliders =
            reg.get<CConservativePotentialColliders<ContactType::Sync>>(collidingEntity);
        // Point-cloud colliders with overlapping conservative step bounds are potential colliders
        // with each other (including themselves)
        reg.view<TagUsePointCloudContact const, CConservativeStepBounds const>().each(
            [&](entt::entity colliderEntity, CConservativeStepBounds const& colliderStepBounds) {
              auto const& colliderLayer = reg.get<CContactLayer>(colliderEntity);
              // Skip self-collisions if indicated by the point-cloud collider parameters.
              bool const isSelfContact = (colliderEntity == collidingEntity);
              if (isSelfContact && !collidingPointCloudColliderParams.selfContact) {
                return;
              }
              if (contactTable.IsContactEnabled(
                      collidingEntity, colliderEntity, collidingLayer.id, colliderLayer.id) &&
                  HasOverlap(collidingStepBounds.worldAabb, colliderStepBounds.worldAabb)) {
                potentialColliders.emplace_back(colliderEntity);
              }
            });
      });
}

void mochi::UpdateStageStartDataPipeline(
    entt::registry& reg,
    CIslandDescendants const& descendants) {
  // Update stage-start map for soft actors that have a mapped SDF.
  ecs::InvokeForEach(&soft::UpdateMap<TimeStep::StageStart>, reg, descendants.softActors);
  ecs::InvokeForEach(&rom::UpdateMap<TimeStep::StageStart>, reg, descendants.softActors);

  // Update stage-start contact samples of soft and compound actors (with a tet-mesh skin).
  // TODO: In most integration methods, the Newton solve is initialized with current state =
  // stage-start state. Then, the initial current contact samples match the stage-start contact
  // samples, so the values could just be copied over.
  TaskSemaphore sem;
  std::array<Span<entt::entity const>, 2> softAndCompoundActors = {
      descendants.softActors, descendants.compoundActors};
  for (auto actors : softAndCompoundActors) {
    ecs::ScheduleInvokeForEach(
        sem,
        "UpdateCollisionSamplePositions<CFemBoundaryDiscretization,TimeStep::StageStart>",
        UpdateCollisionSamplePositions<
            CFemBoundaryDiscretization,
            TimeStep::StageStart,
            kSpaceDim3>,
        reg,
        actors);
  }
  ecs::ScheduleInvokeForEach(
      sem,
      "rod::UpdateSurfaceContactPositions<TimeStep::StageStart>",
      rod::UpdateSurfaceContactPositions<TimeStep::StageStart>,
      reg,
      descendants.rodActors);
  // Update stage-start contact samples of shell and compound actors (with a tri-mesh skin).
  std::array<Span<entt::entity const>, 2> shellAndCompoundActors = {
      descendants.shellActors, descendants.compoundActors};
  for (auto actors : shellAndCompoundActors) {
    ecs::ScheduleInvokeForEach(
        sem,
        "UpdateCollisionSamplePositions<CFemSurfaceDiscretization,TimeStep::StageStart>",
        UpdateCollisionSamplePositions<CFemSurfaceDiscretization, TimeStep::StageStart, kSpaceDim3>,
        reg,
        actors);
  }
  ecs::ScheduleInvokeForEach(
      sem,
      "UpdateCollisionSamplePositions<CFemSegmentDiscretization,TimeStep::StageStart>",
      UpdateCollisionSamplePositions<
          CFemSegmentDiscretization,
          TimeStep::StageStart,
          (kSpaceDim3 + 1)>,
      reg,
      descendants.rodActors);
  sem.Wait();
}

template <TimeStep kTimeStep>
MOCHI_API void mochi::CollisionDetectionPipeline(
    entt::registry& reg,
    CIslandDescendants const& descendants) {
  MOCHI_PROFILE_SCOPE();
  TaskSemaphore sem;

  ////////////////////////////////////////////////////////////////////////////////
  // Pre-Collision Setup
  ////////////////////////////////////////////////////////////////////////////////

  // Update the local-space bounds of actors that deform
  {
    MOCHI_PROFILE_SCOPE_N("InvokeForEach UpdateBounds");
    // Soft and compound actors with a skinned tet mesh.
    std::array<Span<entt::entity const>, 2> softAndCompoundActors = {
        descendants.softActors, descendants.compoundActors};
    for (auto actors : softAndCompoundActors) {
      ecs::ScheduleInvokeForEach(
          sem, "soft::UpdateBounds", &soft::UpdateBounds<kTimeStep>, reg, actors);
    }
    // Compound actors with a skinned tri mesh.
    ecs::ScheduleInvokeForEach(
        sem,
        "articulated::compound::UpdateBounds",
        &articulated::compound::UpdateBounds<kTimeStep>,
        reg,
        descendants.compoundActors);
    ecs::ScheduleInvokeForEach(
        sem, "shell::UpdateBounds", &shell::UpdateBounds<kTimeStep>, reg, descendants.shellActors);
    // Schedule contact-skin bounds first, then centerline bounds.
    // ECS component matching ensures rod actors with TagRodSurfaceContact match
    // UpdateSurfaceContactBounds; those without match UpdateBounds.
    ecs::ScheduleInvokeForEach(
        sem,
        "rod::UpdateSurfaceContactBounds",
        &rod::UpdateSurfaceContactBounds<kTimeStep>,
        reg,
        descendants.rodActors);
    ecs::ScheduleInvokeForEach(
        sem, "rod::UpdateBounds", &rod::UpdateBounds<kTimeStep>, reg, descendants.rodActors);
  }

  // Update Mapped SDFs
  {
    MOCHI_PROFILE_SCOPE_N("InvokeForEach UpdateMap");
    ecs::ScheduleInvokeForEach(
        sem,
        "soft::UpdateMap<kTimeStep>",
        &soft::UpdateMap<kTimeStep>,
        reg,
        descendants.softActors);
    ecs::ScheduleInvokeForEach(
        sem, "rom::UpdateMap<kTimeStep>", &rom::UpdateMap<kTimeStep>, reg, descendants.softActors);
  }

  // Update spatial hash tables for point-cloud contact. This is a collider-side
  // update, and must complete before the loop over colliding actors below.
  ecs::ScheduleInvokeForEach(
      sem, "UpdateSpatialHashTable", &UpdateSpatialHashTable, reg, descendants.shellActors);
  ecs::ScheduleInvokeForEach(
      sem, "UpdateSpatialHashTable", &UpdateSpatialHashTable, reg, descendants.rodActors);

  // The above must complete before collision detection can start.
  sem.Wait();

  ////////////////////////////////////////////////////////////////////////////////
  // Per Actor Collision Detection & Response
  ////////////////////////////////////////////////////////////////////////////////
  [[maybe_unused]] bool const explicitNormals =
      reg.ctx<CSimulationParams const>().experimentalEval.explicitNormals;

  auto asyncCollisionAndResponse = [&](entt::entity e) {
    MOCHI_PROFILE_SCOPE_N("asyncCollisionAndResponse");
    CollisionDetection<ContactType::Async, kTimeStep>(reg, e);
    if constexpr (kTimeStep == TimeStep::Current) {
      if (explicitNormals) {
        ecs::TryInvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
            AddStageStartCollisionDetection<ContactType::Async>, reg, e);
      }
      SetupActiveCollisionNormals<ContactType::Async>(reg, e);
    }
    // Notify other systems (e.g. soft actor assembly) that async collision detection and response
    // has been completed.
    if (auto* semComp = reg.try_get<CActorAsyncContactSemaphore>(e)) {
      if (!semComp->asyncContactUpToDate->IsDone()) {
        semComp->asyncContactUpToDate->Done();
      }
    }
  };

  auto syncCollisionAndResponse = [&](entt::entity e) {
    MOCHI_PROFILE_SCOPE_N("syncCollisionAndResponse");
    CollisionDetection<ContactType::Sync, kTimeStep>(reg, e);
    if constexpr (kTimeStep == TimeStep::Current) {
      if (explicitNormals) {
        ecs::TryInvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
            AddStageStartCollisionDetection<ContactType::Sync>, reg, e);
      }
      SetupActiveCollisionNormals<ContactType::Sync>(reg, e);
    }
  };

  // Loop over colliding actors and find collisions with colliders.
  for (auto e : descendants.actors) {
    if (reg.all_of<TagUseContact>(e)) {
      if (ecs::CanInvokeOnEntity(
              UpdateCollisionSamplePositions<CFemBoundaryDiscretization, kTimeStep, kSpaceDim3>,
              reg,
              e)) {
        Schedule(sem, "UpdateCollisionSamplePositions<CFemBoundaryDiscretization>", [&, e, sem]() {
          // Compute the soft actor's sample positions before we schedule collision detection
          ecs::InvokeOnEntity(
              &UpdateCollisionSamplePositions<CFemBoundaryDiscretization, kTimeStep, kSpaceDim3>,
              reg,
              e);
          Schedule(sem, "AsyncCollisionAndResponse", [&, e]() { asyncCollisionAndResponse(e); });
          syncCollisionAndResponse(e); // Do the work on this thread
        });
      } else if (ecs::CanInvokeOnEntity(rod::UpdateSurfaceContactPositions<kTimeStep>, reg, e)) {
        Schedule(sem, "rod::UpdateSurfaceContactPositions", [&, e, sem]() {
          ecs::InvokeOnEntity(rod::UpdateSurfaceContactPositions<kTimeStep>, reg, e);
          Schedule(sem, "AsyncCollisionAndResponse", [&, e]() { asyncCollisionAndResponse(e); });
          syncCollisionAndResponse(e);
        });
      } else if (ecs::CanInvokeOnEntity(
                     UpdateCollisionSamplePositions<
                         CFemSurfaceDiscretization,
                         kTimeStep,
                         kSpaceDim3>,
                     reg,
                     e)) {
        Schedule(sem, "UpdateCollisionSamplePositions<CFemSurfaceDiscretization>", [&, e, sem]() {
          // Compute the shell actor's sample positions before we schedule collision detection
          ecs::InvokeOnEntity(
              &UpdateCollisionSamplePositions<CFemSurfaceDiscretization, kTimeStep, kSpaceDim3>,
              reg,
              e);
          Schedule(sem, "AsyncCollisionAndResponse", [&, e]() { asyncCollisionAndResponse(e); });
          syncCollisionAndResponse(e); // Do the work on this thread
        });
      } else if (ecs::CanInvokeOnEntity(
                     UpdateCollisionSamplePositions<
                         CFemSegmentDiscretization,
                         kTimeStep,
                         (kSpaceDim3 + 1)>,
                     reg,
                     e)) {
        Schedule(sem, "UpdateCollisionSamplePositions<CFemSegmentDiscretization>", [&, e, sem]() {
          // Compute the rod actor's sample positions before we schedule collision detection
          ecs::InvokeOnEntity(
              &UpdateCollisionSamplePositions<
                  CFemSegmentDiscretization,
                  kTimeStep,
                  (kSpaceDim3 + 1)>,
              reg,
              e);
          Schedule(sem, "AsyncCollisionAndResponse", [&, e]() { asyncCollisionAndResponse(e); });
          syncCollisionAndResponse(e); // Do the work on this thread
        });
      } else {
        Schedule(sem, "AsyncCollisionAndResponse", [&, e]() { asyncCollisionAndResponse(e); });
        Schedule(sem, "SyncCollisionAndResponse", [&, e]() { syncCollisionAndResponse(e); });
      }
    }
  }

  sem.Wait();
}

MOCHI_API void mochi::ContactJacobiansPipeline(
    entt::registry& reg,
    GradTarget gradTarget,
    CIslandDescendants const& descendants,
    TaskSemaphore const& updateJacobianSem) {
  MOCHI_PROFILE_SCOPE();
  TaskSemaphore sem;

  ////////////////////////////////////////////////////////////////////////////////
  // Post-Collisions
  ////////////////////////////////////////////////////////////////////////////////

  // Register contact Jacobians. This call cannot be parallelized, as it will write collider data.
  RegisterContactJacobians(reg, descendants.actors);

  // Wait for UpdateJacobiansSubpipeline (which has probably already finished), then call
  // InitCollidingJacobians and InitColliderJacobians.
  // TODO: It may be possible to start InitCollidingJacobians earlier.
  updateJacobianSem.Wait();

  // Schedule work to initialize collision jacobians. Dispatch to the appropriate templatized
  // implementations based on gradTarget.
  if (gradTarget == GradTarget::Current) {
    InitCollidingJacobians<TimeStep::Current>(sem, reg, descendants);
    InitColliderJacobians<TimeStep::Current>(sem, reg, descendants);
  } else {
    MOCHI_ASSERT_VERBOSE(gradTarget == GradTarget::Previous, "Unsupported gradient target");
    InitCollidingJacobians<TimeStep::StageStart>(sem, reg, descendants);
    InitColliderJacobians<TimeStep::StageStart>(sem, reg, descendants);
  }

  sem.Wait();
}

// Explicit instantiations for CollisionDetectionPipeline
template MOCHI_API void mochi::CollisionDetectionPipeline<TimeStep::Current>(
    entt::registry& reg,
    CIslandDescendants const& descendants);

template MOCHI_API void mochi::CollisionDetectionPipeline<TimeStep::StageStart>(
    entt::registry& reg,
    CIslandDescendants const& descendants);

/*************************************************************************************************/

// Rational-fit coefficients relating normal viscous damping to the effective coefficient of
// restitution: alpha = (1 - e) * (1 + kCorFitA*e) / (e * (1 + kCorFitB*e)), with alpha = c * v.
static constexpr real kCorFitA = 9_r / 2_r;
static constexpr real kCorFitB = 8_r / 3_r;

real mochi::experimental::CalibrateNormalViscousDampingCoefficient(
    real cor,
    real impactVelocity,
    Error& error) {
  MOCHI_ERROR_IF_NOT(
      IsFinite(cor) && (cor > 0_r) && (cor <= 1_r),
      error,
      "Invalid CoR for normal viscous damping calibration: cor must be finite and in (0, 1].");
  MOCHI_ERROR_IF_NOT(
      IsFinite(impactVelocity) && (impactVelocity > 0_r),
      error,
      "Invalid impact velocity for normal viscous damping calibration: impactVelocity must be "
      "finite and positive.");
  MOCHI_ERROR_RETURN(error, {});

  real const alpha = (1_r - cor) * (1_r + kCorFitA * cor) / (cor * (1_r + kCorFitB * cor));
  return alpha / impactVelocity;
}

real mochi::experimental::EffectiveCoefficientOfRestitution(
    real normalViscousDampingCoefficient,
    real impactVelocity,
    Error& error) {
  MOCHI_ERROR_IF_NOT(
      IsFinite(normalViscousDampingCoefficient) && (normalViscousDampingCoefficient >= 0_r),
      error,
      "Invalid normal viscous damping coefficient for effective CoR: must be finite and "
      "non-negative.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(impactVelocity) && (impactVelocity > 0_r),
      error,
      "Invalid impact velocity for effective CoR: impactVelocity must be finite and positive.");
  MOCHI_ERROR_RETURN(error, {});

  // Solve the positive root of (a + b*alpha) e^2 + (alpha + 1 - a) e - 1 = 0.
  real const alpha = normalViscousDampingCoefficient * impactVelocity;
  real const quadA = kCorFitA + kCorFitB * alpha;
  real const quadB = alpha + 1_r - kCorFitA;
  return 2_r / (quadB + Sqrt(Sqr(quadB) + 4_r * quadA));
}
