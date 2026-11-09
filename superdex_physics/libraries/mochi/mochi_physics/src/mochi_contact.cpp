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
#include "mochi_capture.h"
#include "mochi_contact_anchors.h"
#include "mochi_contact_filter.h"
#include "mochi_deformable.h"
#include "mochi_ecs_utils.h"
#include "mochi_group.h"
#include "mochi_hyper_reduction.h"
#include "mochi_island.h"
#include "mochi_linear_contact_skin.h"
#include "mochi_point_cloud_contact.h"
#include "mochi_rigid.h"
#include "mochi_rod.h"
#include "mochi_scene_recorder.h"
#include "mochi_shell.h"
#include "mochi_simulation.h"
#include "mochi_skinned_contact.h"
#include "mochi_soft.h"
#include "mochi_soft_rom_systems.h"
#include "mochi_soft_skinned.h"

#include <mochi_core/contact/dmap.h>
#include <mochi_core/geometry/batch_sphere.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/grid_sdf.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/sparsity_utils.h>
#include <mochi_core/utils/spmat_utils.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ranges>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

using namespace mochi;
using namespace mochi::experimental;

/*************************************************************************************************/

void mochi::InitializeContactSkinningColumnCoalescingMap(
    CFemSurfaceDiscretization const& surfaceDisc,
    CContactSkinningData& outSkinning) {
  MOCHI_PROFILE_SCOPE_N("InitializeContactSkinningColumnCoalescingMap");

  auto& mapping = outSkinning.columnCoalescingMap;
  mapping = {};
  surfaceDisc.Visit([&](auto const& discImpl) {
    using DiscT = std::decay_t<decltype(discImpl)>;
    using ElementT = typename DiscT::ElementT;
    static_assert(ElementT::kNumNodes == 3, "Contact skinning supports triangular surfaces.");

    int const numElements = isize(discImpl.femElements);
    mapping.sourceBlockCount = ElementT::kNumNodes;
    mapping.samplesPerElement = ElementT::kNumQuadPoints;
    MOCHI_ASSERT_VERBOSE(
        discImpl.GetNumQuadPoints() == numElements * mapping.samplesPerElement,
        "Unexpected contact-surface sample layout.");

    mapping.elementDofOffsets.reserve(numElements + 1);
    mapping.elementDofOffsets.push_back(0);
    mapping.elementSourceContributionOffsets.reserve(numElements * mapping.sourceBlockCount + 1);
    mapping.elementSourceContributionOffsets.push_back(0);

    DynamicArray<int> elementDofs;
    for (int elementIndex = 0; elementIndex < numElements; ++elementIndex) {
      auto const nodes = discImpl.femElements[elementIndex].Nodes();
      elementDofs.clear();
      for (int sourceBlock = 0; sourceBlock < mapping.sourceBlockCount; ++sourceBlock) {
        int const skinNode = nodes[sourceBlock];
        MOCHI_ASSERT_VERBOSE(
            skinNode >= 0 && skinNode < outSkinning.jacobian.Rows(),
            "Contact-surface node is outside the skinning Jacobian.");
        auto const rowDofs = outSkinning.jacobian.Indices(skinNode);
        for (int i = 0; i < isize(rowDofs); ++i) {
          MOCHI_ASSERT_VERBOSE(
              rowDofs[i] >= 0 && rowDofs[i] < outSkinning.jacobian.Cols(),
              "Skinning Jacobian DoF is out of range.");
          MOCHI_ASSERT_VERBOSE(
              i == 0 || rowDofs[i - 1] < rowDofs[i],
              "Skinning Jacobian rows must be sorted and duplicate-free.");
        }
        elementDofs.append(rowDofs);
      }

      std::sort(elementDofs.begin(), elementDofs.end());
      elementDofs.erase(std::unique(elementDofs.begin(), elementDofs.end()), elementDofs.end());
      MOCHI_ASSERT_VERBOSE(!elementDofs.empty(), "Contact-surface elements must affect a DoF.");
      mapping.elementDofIndices.append(elementDofs);
      mapping.elementDofOffsets.push_back(isize(mapping.elementDofIndices));

      for (int sourceBlock = 0; sourceBlock < mapping.sourceBlockCount; ++sourceBlock) {
        auto const rowDofs = outSkinning.jacobian.Indices(nodes[sourceBlock]);
        for (int dof : rowDofs) {
          auto* const it = std::lower_bound(elementDofs.begin(), elementDofs.end(), dof);
          MOCHI_ASSERT_VERBOSE(it != elementDofs.end() && *it == dof);
          mapping.elementSourceContributionDstColumns.push_back(
              static_cast<int>(it - elementDofs.begin()));
        }
        mapping.elementSourceContributionOffsets.push_back(
            isize(mapping.elementSourceContributionDstColumns));
      }
    }
  });
}

void mochi::InitializeLinearContactSkinningJacobian(
    LinearMeshEmbedding const& embedding,
    int numPhysicsNodes,
    Span<int const> skinNodeIndices,
    CContactSkinningData& outSkinning) {
  int const weightsPerNode = static_cast<int>(embedding.GetNumSkinningWeightsPerEntry());
  Span<int const> const indices = embedding.GetIndices();
  Span<real const> const weights = embedding.GetWeights();
  int const numEmbeddedNodes = isize(indices) / weightsPerNode;
  int const numSkinNodes = skinNodeIndices.empty() ? numEmbeddedNodes : isize(skinNodeIndices);

  DynamicArray<int> rowPointers(numSkinNodes + 1, 0);
  DynamicArray<int> columnIndices;
  DynamicArray<Real3> values;
  columnIndices.reserve(indices.size() * kSpaceDim3);
  values.reserve(indices.size() * kSpaceDim3);

  DynamicArray<std::pair<int, real>> influences;
  influences.reserve(weightsPerNode);
  for (int skinNode = 0; skinNode < numSkinNodes; ++skinNode) {
    int const embeddedNode = skinNodeIndices.empty() ? skinNode : skinNodeIndices[skinNode];
    MOCHI_ASSERT_VERBOSE(embeddedNode >= 0 && embeddedNode < numEmbeddedNodes);
    influences.clear();
    for (int influence = 0; influence < weightsPerNode; ++influence) {
      int const embeddingIndex = embeddedNode * weightsPerNode + influence;
      int const physicsNode = indices[embeddingIndex];
      MOCHI_ASSERT_VERBOSE(
          physicsNode >= 0 && physicsNode < numPhysicsNodes,
          "Contact-skin embedding index is outside the physics mesh.");
      influences.emplace_back(physicsNode, weights[embeddingIndex]);
    }
    std::sort(influences.begin(), influences.end());

    for (int influence = 0; influence < isize(influences);) {
      int const physicsNode = influences[influence].first;
      real combinedWeight = 0_r;
      do {
        combinedWeight += influences[influence].second;
        ++influence;
      } while (influence < isize(influences) && influences[influence].first == physicsNode);

      for (int component = 0; component < kSpaceDim3; ++component) {
        columnIndices.push_back(kSpaceDim3 * physicsNode + component);
        Real3 derivative{};
        derivative[component] = combinedWeight;
        values.push_back(derivative);
      }
    }
    rowPointers[skinNode + 1] = isize(columnIndices);
  }

  outSkinning.jacobian = SparseMatrix<Real3>(
      numPhysicsNodes * kSpaceDim3,
      std::move(rowPointers),
      std::move(columnIndices),
      std::move(values));
}

void mochi::SetupContactSkinCollidingJacobians(
    ecs::RequiredTag<TagUseDeformableContactSkin>,
    ecs::Excluded<CLinearContactSkinElementMap>,
    CFemSurfaceDiscretization const& surfaceDisc,
    CRootTransform const& transform,
    CDofOffset const& dofOffset,
    CContactSkinningData const& skinningData,
    CCollJacs<CollRole::Colliding>& outJacobians) {
  MOCHI_PROFILE_SCOPE();

  auto const skinJacView = AsConstView(skinningData.jacobian);
  dmap::DMapSparseSkinning sparseSkinning(0, dofOffset.dofsOffset, skinJacView);
  dmap::DMapRTConst dtransform(transform.worldFromLocal);

  surfaceDisc.Visit([&](auto const& discImpl) {
    using DiscT = std::decay_t<decltype(discImpl)>;
    using ElementT = typename DiscT::ElementT;
    using DQuad = dmap::DMapQuad<ElementT>;

    auto const& mapping = skinningData.columnCoalescingMap;
    MOCHI_ASSERT_VERBOSE(
        mapping.IsInitialized(), "Contact-skin coalescing map is not initialized.");
    MOCHI_ASSERT_VERBOSE(
        mapping.sourceBlockCount == ElementT::kNumNodes &&
            mapping.samplesPerElement == ElementT::kNumQuadPoints &&
            mapping.NumElements() == isize(discImpl.femElements),
        "Contact-skin coalescing map does not match the surface discretization.");

    ParallelForEach("SetupContactSkinCollidingJacobians Range", outJacobians, 1, [&](JacData& jac) {
      if (jac.query->sampleIndices.empty()) {
        return;
      }

      DQuad dquad(discImpl.femElements, jac.query->jacColliderFromWorld);
      dmap::DMap<DQuad, dmap::DMapRTConst, dmap::DMapSparseSkinning> dmap(
          &dquad, &dtransform, &sparseSkinning);

      auto jacs = jac.PrepareJacs(1);
      {
        MOCHI_PROFILE_SCOPE_N("GetJac");
        dmap.GetJac(jac.query->sampleIndices, jacs, jac.JacScratch());
      }
      {
        MOCHI_PROFILE_SCOPE_N("CoalesceColumnsByCachedMapping");
        jacs[0].CoalesceColumnsByCachedMapping(
            mapping, jac.query->sampleIndices, dofOffset.dofsOffset);
      }
      {
        MOCHI_PROFILE_SCOPE_N("CompressIndices");
        jacs[0].CompressIndices();
      }
    });
  });
}

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

namespace {
// The contact bounds of a non-static collider. Its island's CollisionDetectionPipeline computes
// them before collision detection, for that call only.
struct CColliderWorldBounds {
  AnyShape shape;
  // Whether some actor's CConservativePotentialColliders</*kIsSync*/ true> holds this collider.
  // Only then does collision detection read `shape`, so only then does its island compute it.
  bool isPotentialSyncCollider = false;
};
} // namespace

// A collider's world-space bounds, padded for contact.
[[nodiscard]] static AnyShape ComputeColliderWorldBounds(
    TransformRT const& worldFromCollider,
    CBoundingVolume const& bounds,
    CContactParams const& params) {
  return TransformShape(
      worldFromCollider, ExpandColliderBoundsForContact(bounds.localShape, params));
}

// Find all the entities that might collide with the colliding actor. Cull the ones that don't
// overlap the colliding's bounding volume at all. Outputs CPotentialColliders with a list of the
// entities that are worth considering for collision detection.
template <bool kIsSync, TimeStep kTimeStep, bool kAllowFarSdfQuery = false>
static void UpdatePotentialColliders(
    entt::registry const& reg,
    entt::entity colliding,
    CBoundingVolume const& boundsColliding,
    CConservativePotentialColliders<kIsSync> const& conservativeColliders,
    CContactLayer const& layerColliding,
    CIslandMemberInfo const& islandColliding,
    CPotentialColliders<kIsSync>& potentialColliders) {
  MOCHI_PROFILE_SCOPE();

  // Discard previous colliders
  potentialColliders.clear();

  // Without far SDF queries, the potential colliders are a subset of the conservative ones.
  if (!kAllowFarSdfQuery && conservativeColliders.empty()) {
    return;
  }

  // Get colliding transform
  TransformRT const& worldFromColliding = GetRootTransform<kTimeStep>(reg, colliding);

  // The colliding padding pads the colliding bounds, because the collider bounds are shared by all
  // colliding actors. Colliders that ignore it are culled conservatively.
  real const collidingPadding =
      reg.get<CContactParams const>(colliding).collidingPenaltyThresholdPadding;

  if constexpr (kAllowFarSdfQuery) {
    // (kAllowFarSdfQuery == true) requires is to consider contact with actors that are outside of
    // the normal bounding volume. This means we can't use CConservativePotentialColliders. Instead,
    // we have to check every other actor in the scene with O(N^2) cost.

    real farEvaluationDistance = GetFarSdfEvaluationDistance<kAllowFarSdfQuery>(reg, colliding);

    AnyShape const worldBoundsColliding = TransformShape(
        worldFromColliding,
        ExpandShape(boundsColliding.localShape, farEvaluationDistance + collidingPadding));

    auto const& contactTable = reg.ctx<CContactFilterTable const>();

    // Iterate over all shapes with a collider
    reg.view<
           CColliderInfo const,
           CContactLayer const,
           CIslandMemberInfo const*,
           CBoundingVolume const,
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

          // For sync contact, only consider actors in the same island.
          // For async contact, only consider actors NOT in the same island.
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
    AnyShape const worldBoundsColliding = TransformShape(
        worldFromColliding, ExpandShape(boundsColliding.localShape, collidingPadding));
#if MOCHI_ASSERT_VERBOSE_ENABLED
    for (entt::entity const collider : conservativeColliders) {
      MOCHI_ASSERT_VERBOSE(
          (reg.all_of<CBoundingVolume, CColliderInfo, CContactParams, CRootTransform>(collider)),
          "Entities listed by CConservativePotentialColliders should have all of these components.");

      // Colliders must not have type "none".
      MOCHI_ASSERT_VERBOSE(
          reg.get<CColliderInfo const>(collider).type != ColliderType::None,
          "CConservativePotentialColliders should only list entities with valid colliders.");

      MOCHI_ASSERT_VERBOSE(
          !kIsSync || (reg.get<CIslandMemberInfo const>(collider).island == islandColliding.island),
          "Sync colliders should be members of the colliding actor's island.");
      MOCHI_ASSERT_VERBOSE(
          !kIsSync ||
              (reg.all_of<CColliderWorldBounds>(collider) &&
               reg.get<CColliderWorldBounds const>(collider).isPotentialSyncCollider),
          "Sync colliders should be flagged, so that their island computes their bounds.");
    }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
    if constexpr (kIsSync) {
      // A bounding sphere test culls most of the distant Obbs before the exact test.
      Sphere const sphereColliding = GetBoundingSphere(worldBoundsColliding);
      // Precondition: this island's CollisionDetectionPipeline computed these bounds in this call.
      auto const& colliderWorldBounds = reg.storage<CColliderWorldBounds>();
      for (entt::entity const collider : conservativeColliders) {
        AnyShape const& worldBoundsCollider = colliderWorldBounds.get(collider).shape;
        bool const isCulled = std::holds_alternative<Obb>(worldBoundsCollider) &&
            IsObbOutsideSphere(std::get<Obb>(worldBoundsCollider), sphereColliding);
        if (!isCulled && HasOverlap(worldBoundsColliding, worldBoundsCollider)) {
          potentialColliders.emplace_back(collider);
        }
      }
    } else {
      // Async colliders are static, and other islands read them concurrently.
      auto const& rootTransforms = reg.storage<CRootTransform>();
      auto const& boundingVolumes = reg.storage<CBoundingVolume>();
      auto const& contactParams = reg.storage<CContactParams>();
      for (entt::entity const collider : conservativeColliders) {
        AnyShape const worldBoundsCollider = ComputeColliderWorldBounds(
            GetRootTransform<kTimeStep>(reg, collider, rootTransforms.get(collider)),
            boundingVolumes.get(collider),
            contactParams.get(collider));
        if (HasOverlap(worldBoundsColliding, worldBoundsCollider)) {
          potentialColliders.emplace_back(collider);
        }
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

// Note: outContacts and outSdf are in SDF space, in the reference configuration for mapped SDFs.
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

    // The map's Jacobians are in SDF space. Rotate them so that outMapJac is the derivative of the
    // reference position wrt the world position, and outDofsJac that of the world position wrt the
    // DoFs.
    VMatrix3x3r const rotSdfFromWorld = ToVMatrix3x3Transpose(worldFromSdf.GetRotation());
    if (outMapJac) {
      for (auto& jac : *outMapJac) {
        jac = Dot3x3(jac, rotSdfFromWorld);
      }
    }
    if (outDofsJac) {
      int const numDofs = mappedCollider.mapping->GetNumDofs();
      for (auto& dofsJac : *outDofsJac) {
        for (int j = 0; j < numDofs; ++j) {
          dofsJac.jac[j] = DotVecMat3x3(dofsJac.jac[j], rotSdfFromWorld);
        }
      }
    }
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

    // Find colliding-point/collider-point pairs within the contact range.
    ComputePointCloudContactIndices(
        params,
        colliderDiscretization,
        worldFromCollider,
        collidingDiscretization,
        selfContact,
        positionsInColliding,
        indicesToQuery,
        worldFromColliding,
        colliderSpatialHashTable,
        contactParams.GetPenaltyThresholdDist(true),
        outIndices,
        *outColliderFeatureIndices);
  } else {
    // Skip the secondary culling based on the spatial hash table, fill in outIndices with an
    // identity mapping, and use colliderFeatureIndices as input (computed in a previous pass).
    MOCHI_ASSERT_VERBOSE(
        isize(positionsInColliding) == isize(colliderFeatureIndices),
        "Size mismatch between colliding samples and collider points.");
    outIndices.resize_noinit(positionsInColliding.size());
    std::iota(outIndices.begin(), outIndices.end(), 0);
  }

  Span<int const> colliderPointIndicesToUse = outColliderFeatureIndices != nullptr
      ? MakeConstSpan(*outColliderFeatureIndices)
      : colliderFeatureIndices;

  // Compute contact fields for the selected point pairs.
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

// Not inlined, so equal transforms give bit-identical matrices at every call site.
MOCHI_NO_INLINE static VMatrix4x4r ComputeColliderFromCollidingMatT(
    TransformRT const& worldFromColliding,
    TransformRT const& worldFromCollider) {
  return ToVMatrix4x4Transpose(Invert(worldFromCollider) * worldFromColliding);
}

// Stage-start positions are the current positions minus each sample's displacement in the collider
// frame, A p - A0 p0 = (A - A0) p0 + R (p - p0) with A = [R t]. The displacement is formed from
// differences, so it is exactly zero for a stationary pair however the query rounded positions.
static void TransformToCollider(
    entt::registry const& reg,
    entt::entity collider,
    TransformRT const& worldFromColliding,
    TransformRT const& worldFromCollidingStageStart,
    TransformRT const& worldFromColliderStageStart,
    Span<Real3 const> samples,
    Span<Real3 const> samplesStageStart,
    Span<int const> sampleIndices,
    Span<int const> /* colliderFeatureIndices */,
    Span<Real3 const> posColliding,
    DynamicArray<Real3>& /* positionsBuffer */,
    DynamicArray<int>& /* outIndices */,
    DynamicArray<Real3>& outPosCollidingStageStart) {
  MOCHI_ASSERT_VERBOSE(
      (reg.any_of<TagRigidActor, TagStaticActor>(collider)),
      "TransformToCollider must only be called for rigid colliders.");
  MOCHI_ASSERT_VERBOSE(
      posColliding.size() == sampleIndices.size(), "Expected one current position per contact.");
  VMatrix4x4r const colliderFromCollidingMatT = ComputeColliderFromCollidingMatT(
      worldFromColliding, GetRootTransform<TimeStep::Current>(reg, collider));
  VMatrix4x4r const deltaMatT = colliderFromCollidingMatT -
      ComputeColliderFromCollidingMatT(worldFromCollidingStageStart, worldFromColliderStageStart);
  // Rigid colliding actors share one sample array between time slices, so their samples don't move.
  bool const samplesMove = samples.data() != samplesStageStart.data();
  int const numContacts = isize(sampleIndices);
  outPosCollidingStageStart.resize_noinit(numContacts);
  for (int i = 0; i < numContacts; ++i) {
    int const sampleIdx = sampleIndices[i];
    Real3 const& sampleStageStart = samplesStageStart[sampleIdx];
    Vec4r displacement = DotVecMat4x4(ToSimd(sampleStageStart, 1_r), deltaMatT);
    if (samplesMove) {
      displacement +=
          DotVecMat3x3(ToSimd(samples[sampleIdx] - sampleStageStart), colliderFromCollidingMatT);
    }
    outPosCollidingStageStart[i] = posColliding[i] - ToReal3(displacement);
  }
}

static void MapToCollider(
    entt::registry const& reg,
    entt::entity collider,
    TransformRT const& /* worldFromColliding */,
    TransformRT const& worldFromCollidingStageStart,
    TransformRT const& worldFromColliderStageStart,
    Span<Real3 const> /* samples */,
    Span<Real3 const> samplesStageStart,
    Span<int const> sampleIndices,
    Span<int const> /* colliderFeatureIndices */,
    Span<Real3 const> /* posColliding */,
    DynamicArray<Real3>& positionsBuffer,
    DynamicArray<int>& outIndices,
    DynamicArray<Real3>& outPosCollidingStageStart) {
  // The SDF frame may be different than the Collider frame, e.g. they may differ by the rigid
  // transform for ROM colliders.
  TransformRT worldFromSdf = worldFromColliderStageStart;
  if (reg.all_of<TagRomActor>(collider)) {
    if (auto rigidTransform = rom::GetRigidTransform<TimeStep::StageStart>(reg, collider)) {
      worldFromSdf *= rigidTransform.value();
    }
  }
  auto collidingFromSdf = Invert(worldFromCollidingStageStart) * worldFromSdf;

  // The query takes contiguous points.
  positionsBuffer.clear();
  positionsBuffer.reserve(sampleIndices.size());
  for (auto const& i : sampleIndices) {
    positionsBuffer.push_back(samplesStageStart[i]);
  }

  // Get collider shape and mapping
  auto const& sdfCollider = reg.get<CSdfCollider>(collider);
  auto const& map = reg.get<CSdfMapping<TimeStep::StageStart> const>(collider);

  // Create a MappedSdfCollider
  MappedSdfCollider mappedCollider{.shape = sdfCollider.shape.get(), .mapping = map.get()};

  // Set infinite tolerance and collider bounds.
  ContactDetectionParams detectionParams;
  detectionParams.tolerance = kInf;
  Sphere colliderBounds(Real3{}, kInf);

  // Run the query
  bool isSdfGradUnitary{}; // unused
  FindPointContactsMapped(
      positionsBuffer,
      &mappedCollider,
      detectionParams,
      collidingFromSdf,
      colliderBounds,
      outIndices,
      outPosCollidingStageStart,
      /*outSdf*/ nullptr,
      isSdfGradUnitary,
      /*outNDofs*/ nullptr,
      /*outMapJac*/ nullptr,
      /*outDofsJac*/ nullptr);
}

// Points-only query for point-cloud colliders. This is a lightweight wrapper that extracts collider
// data from the registry and delegates to ComputePointCloudContactDetectionFields with null
// optional outputs.
static void QueryPointCloudPointsOnly(
    entt::registry const& reg,
    entt::entity collider,
    TransformRT const& /* worldFromColliding */,
    TransformRT const& worldFromCollidingStageStart,
    TransformRT const& worldFromColliderStageStart,
    Span<Real3 const> /* samples */,
    Span<Real3 const> samplesStageStart,
    Span<int const> sampleIndices,
    Span<int const> colliderFeatureIndices,
    Span<Real3 const> /* posColliding */,
    DynamicArray<Real3>& /* positionsBuffer */,
    DynamicArray<int>& /* outIndices */,
    DynamicArray<Real3>& outPosCollidingStageStart) {
  int const totalContacts = isize(colliderFeatureIndices);
  MOCHI_ASSERT_VERBOSE(
      isize(sampleIndices) == totalContacts,
      "Size mismatch between colliding samples and collider nodes.");
  if (totalContacts == 0) {
    return;
  }

  // Get collider-side data from registry.
  auto const& colliderDiscretization = reg.get<CColliderPointCloudDiscretization>(collider);
  ColumnVectorView<real const> colliderDisplacements =
      reg.get<CFinalDisplacementRef<TimeStep::StageStart>>(collider).value;

  // Call ComputePointCloudContactDetectionFields with null optional outputs. It reads the samples
  // through sampleIndices, so they need not be gathered.
  ComputePointCloudContactDetectionFields(
      colliderDiscretization,
      colliderDisplacements,
      worldFromColliderStageStart,
      samplesStageStart,
      worldFromCollidingStageStart,
      sampleIndices,
      colliderFeatureIndices,
      /*radius=*/0_r,
      outPosCollidingStageStart,
      /*outSdfInfo=*/nullptr,
      /*outNdofs=*/nullptr,
      /*outColliderIntegrationWeights=*/nullptr,
      /*outJacColliderFromWorld=*/nullptr,
      /*outJacWorldFromDofs=*/nullptr);
}

/*
  Get the query function pointer for stage-start contact positions (points only). Unsuffixed
  arguments are current values.
*/
static auto GetQueryFuncForPointsOnly(entt::registry const& reg, entt::entity collider) {
  auto const colliderType = reg.get<CColliderInfo const>(collider).type;
  // MappedSdfCollider requires a special function. Point-cloud colliders use their own query.
  // Other colliders are rigid and just use a transform function.
  if (colliderType == ColliderType::PointCloud) {
    return &QueryPointCloudPointsOnly;
  }
  auto const* map = reg.try_get<CSdfMapping<TimeStep::StageStart> const>(collider);
  return map ? &MapToCollider : &TransformToCollider;
}

/*
  Distribute an aggregate narrow-phase result to the active partition results. The function is
  templatized with TimeStep to distinguish behavior with TimeStep::Current and TimeStep::StageStart.
  In TimeStep::Current, both current and stage-start fields are populated. In TimeStep::StageStart,
  only current fields are populated (it's not necessary to populate the stage-start fields, because
  their content would be duplicated).
*/
template <TimeStep kTimeStep>
static void DistributeContactDetectionResultToPartitions(
    bool explicitNormals,
    ContactDetectionResult& sourceResult,
    Span<ContactDetectionResult*> resultPerPartition,
    CContactPartitions const& collisionPartitions) {
  static_assert(kTimeStep == TimeStep::Current || kTimeStep == TimeStep::StageStart);

  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(collisionPartitions.size() > 1, "Expected multiple partitions.");
  MOCHI_ASSERT_VERBOSE(resultPerPartition.size() == collisionPartitions.size());
  // TODO: Distribute and truncate jacWorldFromDofs, colliderIntegrationWeights, and
  // colliderFeatureIndices to support soft colliders.
  MOCHI_ASSERT_VERBOSE(
      sourceResult.jacColliderFromWorld.size() == 1, "Deformable colliders are not supported");
  MOCHI_ASSERT_VERBOSE(
      sourceResult.jacColliderFromWorldStageStart.empty() ||
          (explicitNormals && (sourceResult.jacColliderFromWorldStageStart.size() == 1)),
      "Unexpected size");

  for (int p = 1; p < isize(resultPerPartition); ++p) {
    auto* targetPartition = resultPerPartition[p];
    if (!targetPartition) {
      continue;
    }
    targetPartition->isSdfGradUnitary = sourceResult.isSdfGradUnitary;
    targetPartition->isNormalCollidingUnitary = sourceResult.isNormalCollidingUnitary;
    targetPartition->jacColliderFromWorld.resize_noinit(1);
    targetPartition->jacColliderFromWorld[0] = sourceResult.jacColliderFromWorld[0];
    if constexpr (kTimeStep == TimeStep::Current) {
      if (explicitNormals) {
        targetPartition->jacColliderFromWorldStageStart.resize_noinit(1);
        targetPartition->jacColliderFromWorldStageStart[0] =
            sourceResult.jacColliderFromWorldStageStart[0];
      }
    }
  }

  // Keep partition zero in the aggregate result so its arrays can be moved into the active result.
  int firstPartitionCount = 0;
  for (int i = 0; i < isize(sourceResult.sampleIndices); ++i) {
    int const sampleIdx = sourceResult.sampleIndices[i];
    int const partitionIdx = collisionPartitions.SampleIdxToPartitionIdx(sampleIdx);

    if (partitionIdx == 0) {
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
      ++firstPartitionCount;
      continue;
    }

    auto* targetPartition = resultPerPartition[partitionIdx];
    MOCHI_ASSERT_VERBOSE(targetPartition, "Expected an active contact partition.");
    targetPartition->sampleIndices.push_back(sampleIdx);

    // TODO: Copy arrays instead of individual points
    targetPartition->posColliding.push_back(sourceResult.posColliding[i]);
    targetPartition->sdfInfo.grad.push_back(sourceResult.sdfInfo.grad[i]);
    targetPartition->sdfInfo.val.push_back(sourceResult.sdfInfo.val[i]);
    if constexpr (kTimeStep == TimeStep::Current) {
      targetPartition->posCollidingStageStart.push_back(sourceResult.posCollidingStageStart[i]);
      if (explicitNormals) {
        targetPartition->sdfInfoStageStart.push_back(
            sourceResult.sdfInfoStageStart.val[i], sourceResult.sdfInfoStageStart.grad[i]);
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
        outPositionsToQuery.size() == contactSamples.bsh->GetNumSamples(),
        "Inconsistent number of sample points. Is the SphereTree tree up-to-date with the number of active sample points?");

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
              .distanceThreshold = GetColliderPadding(contactParams) +
                  GetCollidingPadding(reg, colliding, collider) + farSdfDistance,
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

      std::visit(
          [&](auto const& bv) {
            contactSamples.bsh->FindIntersectingSamples(bv, culledIndicesBuffer);
          },
          colliderBvForCulling);

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
 * @tparam kTimeStep           Time slice for collider data (Current or StageStart)
 * @tparam kAllowFarSdfQuery   Allow SDF queries beyond the normal contact culling distance
 * @param reg                  Registry
 * @param colliding            Colliding entity (collidee)
 * @param contactSamples       Sample points on the colliding entity's surface
 * @param collisionPartitions  Optional partitioning of the colliding entity's surface
 * @param collidingBounds      Bounding shape of the colliding entity
 * @param worldFromColliding   Transform from colliding entity's local space to world space
 * @param collider             Collider entity to check against
 * @param outResult            Output for collision detection results (one for all the partitions)
 *
 * @note If collision partitions are specified, the results are distributed across the partitions
 * based on which partition each sample point belongs to.
 */
template <TimeStep kTimeStep, bool kAllowFarSdfQuery>
static void DetectCollisionsWithSingleCollider(
    entt::registry const& reg,
    entt::entity colliding,
    ContactSamples const& contactSamples,
    AnyShape const& collidingBounds,
    TransformRT const& worldFromColliding,
    entt::entity collider,
    ContactDetectionResult& outResult) {
  MOCHI_PROFILE_SCOPE();

  auto const& colliderContactParams = reg.get<CContactParams const>(collider);
  auto const& worldFromCollider = GetRootTransform<kTimeStep>(reg, collider);
  auto const& colliderBounds = reg.get<CBoundingVolume const>(collider).localShape;
  AnyShape expandedColliderBounds = ExpandShape(
      colliderBounds,
      GetColliderPadding(colliderContactParams) + GetCollidingPadding(reg, colliding, collider));
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

  // Set tolerance depending on the penalty threshold distance of the pair.
  ContactDetectionParams detectionParams;
  real const farSdfDistance = GetFarSdfEvaluationDistance<kAllowFarSdfQuery>(reg, colliding);
  detectionParams.tolerance =
      GetPairPenaltyThresholdDist(reg, colliding, collider) + farSdfDistance;

  // Initialize outResult.jacColliderFromWorld assuming a rigid collider. This will be overwritten
  // for deformable colliders.
  outResult.jacColliderFromWorld.resize_noinit(1);
  outResult.jacColliderFromWorld[0] = ToVMatrix3x3Transpose(worldFromCollider.GetRotation());

  // Only mapped colliders have Jacobians that aren't rotations.
  outResult.isNormalCollidingUnitary = !reg.all_of<CSdfMapping<kTimeStep>>(collider);

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
    rom::AddRigidContactJacobians(reg, collider, positionsToQuery, worldFromColliding, outResult);
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
    ContactSamples const& contactSamplesStageStart,
    TransformRT const& worldFromColliding,
    TransformRT const& worldFromCollidingStageStart,
    entt::entity collider,
    ContactDetectionResult& outResult) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(!outResult.sampleIndices.empty(), "Expected at least one collision.");

  // Get the transform of the collider.
  auto const& worldFromColliderStageStart = GetRootTransform<TimeStep::StageStart>(reg, collider);

  // Output container for indices. Do not pass outResult.sampleIndices, because the query expects
  // an empty container. Branches that output every contact may leave it empty.
  auto& outIndices = outResult.culledIndicesBuffer;
  outIndices.clear();

  // If explicitNormals, evaluate contact points, SDF and Jacobians; otherwise evaluate contact
  // points only.
  if (explicitNormals) {
    // List of sample points to be tested.
    auto& positionsInColliding = outResult.culledPositionsBuffer;
    positionsInColliding.clear();
    positionsInColliding.reserve(outResult.sampleIndices.size());
    for (auto const& i : outResult.sampleIndices) {
      positionsInColliding.push_back(contactSamplesStageStart.positions[i]);
    }

    // Initialize outResult.jacColliderFromWorldStageStart assuming a rigid collider. This will be
    // overwritten for deformable colliders.
    outResult.jacColliderFromWorldStageStart.resize_noinit(1);
    outResult.jacColliderFromWorldStageStart[0] =
        ToVMatrix3x3Transpose(worldFromColliderStageStart.GetRotation());

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
        worldFromCollidingStageStart,
        worldFromColliderStageStart,
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
    auto queryFunc = GetQueryFuncForPointsOnly(reg, collider);
    queryFunc(
        reg,
        collider,
        worldFromColliding,
        worldFromCollidingStageStart,
        worldFromColliderStageStart,
        contactSamples.positions,
        contactSamplesStageStart.positions,
        outResult.sampleIndices,
        outResult.colliderFeatureIndices,
        outResult.posColliding,
        outResult.culledPositionsBuffer,
        outIndices,
        outResult.posCollidingStageStart);
  }

  // If needed, fill any missing results. Queries that miss contacts list the ones they output in
  // outIndices.
  if (outResult.posCollidingStageStart.size() != outResult.sampleIndices.size()) {
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
template <bool kIsSync, TimeStep kTimeStep, bool kAllowFarSdfQuery = false>
static void DetectCollisionsWithPotentialColliders(
    entt::registry const& reg,
    entt::entity colliding,
    CBoundingVolume const& boundingVolume,
    CContactSamples<TimeStep::Current> const& samplesCurrent,
    CContactSamples<TimeStep::StageStart> const* samplesStageStartDeformable,
    CContactPartitions const* collisionPartitions,
    CPotentialColliders<kIsSync>& potentialColliders,
    CActiveCollisions<kIsSync, kTimeStep>& outActiveCollisions) {
  MOCHI_PROFILE_SCOPE();

  int const numPartitions = collisionPartitions ? isize(*collisionPartitions) : 1;
  MOCHI_ASSERT_VERBOSE(
      outActiveCollisions.GetNumPartitions() == numPartitions,
      "Active collision component does not match contact partition count.");

  // Set up memory for collision points.
  outActiveCollisions.SetUp(potentialColliders);

  if (potentialColliders.empty()) {
    return;
  }

// Contact must be sync for dynamic colliders and async for static colliders.
#if MOCHI_ASSERT_VERBOSE_ENABLED
  for (entt::entity const potentialCollider : potentialColliders) {
    MOCHI_ASSERT_VERBOSE(
        kIsSync != reg.all_of<TagStaticActor>(potentialCollider),
        "Wrong contact type for this collider");
  }
#endif

  // If the actor has a deforming surface, CContactSamples<TimeStep::StageStart> stores collision
  // sample positions at stage start. If the actor has a rigid surface,
  // CContactSamples<TimeStep::Current> stores the positions for all time slices.
  MOCHI_ASSERT_VERBOSE(
      !samplesCurrent.positions.empty(),
      "Sample points should have been computed/updated BEFORE calling DetectCollisionsWithPotentialColliders.");
  if (samplesStageStartDeformable) {
    MOCHI_ASSERT_VERBOSE(
        !samplesStageStartDeformable->positions.empty(),
        "Sample points should have been computed/updated BEFORE calling DetectCollisionsWithPotentialColliders.");
  }
  auto const& samplesTimeStep = (kTimeStep == TimeStep::Current || !samplesStageStartDeformable)
      ? static_cast<ContactSamples const&>(samplesCurrent)
      : static_cast<ContactSamples const&>(*samplesStageStartDeformable);
  [[maybe_unused]] auto const& samplesStageStart = samplesStageStartDeformable
      ? static_cast<ContactSamples const&>(*samplesStageStartDeformable)
      : static_cast<ContactSamples const&>(samplesCurrent);

  // Get the transforms of the colliding actor.
  auto const& worldFromColliding = GetRootTransform<kTimeStep>(reg, colliding);
  [[maybe_unused]] auto const& worldFromCollidingStageStart =
      GetRootTransform<TimeStep::StageStart>(reg, colliding);

  // If explicitNormals = true, we also need SDF distance and gradient evaluation at
  // TimeStep::StageStart.
  bool const explicitNormals = reg.ctx<CSimulationParams const>().experimentalEval.explicitNormals;

  // Prepare one narrow-phase output per collider. With one partition, this points directly to the
  // active result. With multiple partitions, it points to an aggregate result that is later
  // distributed among the active partition results.
  int const numPotentialColliders = isize(potentialColliders);
  MOCHI_FILO_STACK_ALLOCATOR(filoAllocator, 4096);
  DynamicArray<ContactDetectionResult*> resultPerCollider(&filoAllocator);
  resultPerCollider.reserve(numPotentialColliders);
  for (entt::entity const potentialCollider : potentialColliders) {
    resultPerCollider.push_back(&outActiveCollisions.GetNarrowPhaseResult(potentialCollider));
  }

  // In case of multiple partitions, prepare also a double-array of storage pointers. In the first
  // pass, they are used only as non-null activation markers.
  DynamicArray<DynamicArray<ContactDetectionResult*>> resultPerColliderPerPartition(&filoAllocator);
  if (numPartitions > 1) {
    MOCHI_ASSERT_VERBOSE(collisionPartitions, "Expected contact partitions.");
    resultPerColliderPerPartition.reserve(numPotentialColliders);
    for (int c = 0; c < numPotentialColliders; ++c) {
      resultPerColliderPerPartition.emplace_back(&filoAllocator);
      resultPerColliderPerPartition.back().resize(numPartitions, nullptr);
    }
  }

  // Narrow-phase collision detection with all colliders.
  ParallelForN("DetectCollisionsWithSingleCollider", numPotentialColliders, 1, [&](int c) {
    // Perform collision detection with this collider.
    DetectCollisionsWithSingleCollider<kTimeStep, kAllowFarSdfQuery>(
        reg,
        colliding,
        samplesTimeStep,
        /*collidingBounds*/ boundingVolume.localShape,
        worldFromColliding,
        potentialColliders[c],
        *resultPerCollider[c]);
    if constexpr (kTimeStep == TimeStep::Current && !kAllowFarSdfQuery) {
      // For queries in TimeStep::Current and not kAllowFarSdfQuery, complete with data from
      // TimeStep::StageStart.
      // WARNING: Do not use CBoundingVolume or CSpatialHashTable here. They contain
      // TimeStep::Current data, while this path evaluates TimeStep::StageStart.
      if (!resultPerCollider[c]->sampleIndices.empty()) {
        EvalStageStartContactWithSingleCollider(
            reg,
            colliding,
            explicitNormals,
            samplesTimeStep,
            samplesStageStart,
            worldFromColliding,
            worldFromCollidingStageStart,
            potentialColliders[c],
            *resultPerCollider[c]);
      }
    }

    // With multiple partitions, flag active collider-partition pairs. Use the aggregate pointer
    // only as a temporary non-null activation marker.
    if (numPartitions > 1) {
      auto& resultPerPartition = resultPerColliderPerPartition[c];
      int numActivePartitions = 0;
      for (int sampleIdx : resultPerCollider[c]->sampleIndices) {
        int const partitionIdx = collisionPartitions->SampleIdxToPartitionIdx(sampleIdx);
        if (!resultPerPartition[partitionIdx]) {
          resultPerPartition[partitionIdx] = resultPerCollider[c];
          if (++numActivePartitions == numPartitions) {
            break;
          }
        }
      }
    }
  });

  if (numPartitions == 1) {
    return;
  }

  // Connect the ContactDetectionResult pointers of active collider-partition pairs to their final
  // storage within the ActiveCollision array. This is done in three steps: (1) Allocate/reuse
  // memory for active pairs. (2) Sort pairs for efficient lookup. (3) Connect the pointers.
  for (int c = 0; c < numPotentialColliders; ++c) {
    for (int p = 0; p < numPartitions; ++p) {
      if (resultPerColliderPerPartition[c][p]) {
        outActiveCollisions.ActivatePartitionResult(potentialColliders[c], p);
      }
    }
  }
  outActiveCollisions.FinalizeActiveCollisions();
  for (int c = 0; c < numPotentialColliders; ++c) {
    for (int p = 0; p < numPartitions; ++p) {
      if (resultPerColliderPerPartition[c][p]) {
        resultPerColliderPerPartition[c][p] =
            &outActiveCollisions.GetActivePartitionResult(potentialColliders[c], p);
      }
    }
  }

  // Distribute partitions 1..N-1 to their active results and compact partition 0 in the aggregate.
  // Then move the partition 0 aggregate into its active result without copying its arrays; the
  // next SetUp restores the aggregate storage.
  ParallelForN(
      "DistributeContactDetectionResultToPartitions", numPotentialColliders, 1, [&](int c) {
        if (resultPerCollider[c]->sampleIndices.empty()) {
          return;
        }
        DistributeContactDetectionResultToPartitions<kTimeStep>(
            explicitNormals,
            *resultPerCollider[c],
            MakeSpan(resultPerColliderPerPartition[c]),
            *collisionPartitions);
        if (auto* const partitionZeroResult = resultPerColliderPerPartition[c][0]) {
          std::swap(*resultPerCollider[c], *partitionZeroResult);
        }
      });
}

template <bool kIsSync, TimeStep kTimeStep, bool kAllowFarSdfQuery = false>
static void CollisionDetection(entt::registry& reg, entt::entity ent) {
  // Cull potential colliders using the updated Aabb and determine active contact components.
  ecs::TryInvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
      &UpdatePotentialColliders<kIsSync, kTimeStep, kAllowFarSdfQuery>, reg, ent);

  // Perform collision detection (updates CActiveCollisions).
  ecs::TryInvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
      &DetectCollisionsWithPotentialColliders<kIsSync, kTimeStep, kAllowFarSdfQuery>, reg, ent);
}

// Compute contact normals of a colliding actor
template <bool kIsSync>
static void SetupActiveCollisionNormals(entt::registry& reg, entt::entity ent) {
  MOCHI_PROFILE_SCOPE();
  auto const* activeCollisions = reg.try_get<CActiveCollisions<kIsSync, TimeStep::Current>>(ent);
  if (!activeCollisions || !activeCollisions->HasContact()) {
    return;
  }
  ecs::TryInvokeOnEntity(&linear_contact_skin::SetupActiveCollisionNormals<kIsSync>, reg, ent);
  ecs::TryInvokeOnEntity(
      &deformable::SetupActiveCollisionNormals<kIsSync, CFemBoundaryDiscretization>, reg, ent);
  ecs::TryInvokeOnEntity(
      &deformable::SetupActiveCollisionNormals<kIsSync, CFemSurfaceDiscretization>, reg, ent);
  ecs::TryInvokeOnEntity(&rigid::SetupActiveCollisionNormals<kIsSync>, reg, ent);
}

template <class StorageT>
[[nodiscard]] static MOCHI_FORCE_INLINE bool HasCollJacs(
    StorageT const& collJacsStorage,
    entt::entity e) {
  auto const* collJacs = TryGet(collJacsStorage, e);
  return collJacs && !collJacs->empty();
}

template <class StorageT>
[[nodiscard]] static MOCHI_FORCE_INLINE bool AnyHasCollJacs(
    StorageT const& collJacsStorage,
    Span<entt::entity const> actors) {
  return std::ranges::any_of(
      actors, [&](entt::entity e) { return HasCollJacs(collJacsStorage, e); });
}

// Contact Jacobian setup systems only iterate CCollJacs, so actors without contact Jacobians have
// nothing to set up.
template <class StorageT, class SystemT>
static MOCHI_FORCE_INLINE void ScheduleForActorsWithCollJacs(
    TaskSemaphore& sem,
    std::string_view debugLabel,
    SystemT system,
    entt::registry& reg,
    StorageT const& collJacsStorage,
    Span<entt::entity const> actors) {
  for (entt::entity const e : actors) {
    if (HasCollJacs(collJacsStorage, e)) {
      ecs::TryScheduleInvokeOnEntity(sem, debugLabel, system, reg, e);
    }
  }
}

template <TimeStep kTimeStep>
static void InitCollidingJacobians(
    TaskSemaphore& sem,
    entt::registry& reg,
    CIslandDescendants const& descendants) {
  MOCHI_PROFILE_SCOPE();
  auto const& collidingJacsStorage = std::as_const(reg).storage<CCollJacs<CollRole::Colliding>>();

  // Schedule all deforming actors in parallel
  ScheduleForActorsWithCollJacs(
      sem,
      "deformable::SetupCollidingJacobians<TagSoftActor, CFemBoundaryDiscretization>",
      &deformable::SetupCollidingJacobians<TagSoftActor, CFemBoundaryDiscretization>,
      reg,
      collidingJacsStorage,
      descendants.softActors);
  ScheduleForActorsWithCollJacs(
      sem,
      "deformable::SetupCollidingJacobians<TagShellActor, CFemSurfaceDiscretization>",
      &deformable::SetupCollidingJacobians<TagShellActor, CFemSurfaceDiscretization>,
      reg,
      collidingJacsStorage,
      descendants.shellActors);
  // Centerline Jacobians for rods without surface contact.
  ScheduleForActorsWithCollJacs(
      sem,
      "deformable::SetupCollidingJacobians<TagRodActor, CFemSegmentDiscretization>",
      &deformable::SetupCollidingJacobians<TagRodActor, CFemSegmentDiscretization>,
      reg,
      collidingJacsStorage,
      descendants.rodActors);
  std::array<Span<entt::entity const>, 3> contactSkinActors = {
      descendants.softActors, descendants.shellActors, descendants.rodActors};
  for (auto actors : contactSkinActors) {
    ScheduleForActorsWithCollJacs(
        sem,
        "SetupContactSkinCollidingJacobians",
        &SetupContactSkinCollidingJacobians,
        reg,
        collidingJacsStorage,
        actors);
  }
  ScheduleForActorsWithCollJacs(
      sem,
      "rom::SetupCollidingJacobians",
      &rom::SetupCollidingJacobians,
      reg,
      collidingJacsStorage,
      descendants.softActors);
  ScheduleForActorsWithCollJacs(
      sem,
      "articulated::compound::SetupCollidingJacobians<CFemBoundaryDiscretization>",
      &articulated::compound::SetupCollidingJacobians<CFemBoundaryDiscretization>,
      reg,
      collidingJacsStorage,
      descendants.compoundActors);
  ScheduleForActorsWithCollJacs(
      sem,
      "articulated::compound::SetupCollidingJacobians<CFemSurfaceDiscretization>",
      &articulated::compound::SetupCollidingJacobians<CFemSurfaceDiscretization>,
      reg,
      collidingJacsStorage,
      descendants.compoundActors);
  ScheduleForActorsWithCollJacs(
      sem,
      "skinned::SetupCollidingJacobians",
      &skinned::SetupCollidingJacobians,
      reg,
      collidingJacsStorage,
      descendants.nestedSoftActors);
  ScheduleForActorsWithCollJacs(
      sem,
      "blended::SetupCollidingJacobians<CFemBoundaryDiscretization>",
      &blended::SetupCollidingJacobians<CFemBoundaryDiscretization>,
      reg,
      collidingJacsStorage,
      descendants.blendedActors);
  ScheduleForActorsWithCollJacs(
      sem,
      "blended::SetupCollidingJacobians<CFemSurfaceDiscretization>",
      &blended::SetupCollidingJacobians<CFemSurfaceDiscretization>,
      reg,
      collidingJacsStorage,
      descendants.blendedActors);

  // Schedule jointly all the rigid links in an articulated actor
  for (auto e : descendants.compoundActors) {
    if (reg.all_of<TagArticulatedActor>(e) &&
        AnyHasCollJacs(collidingJacsStorage, reg.get<CGroupMembers const>(e).actors)) {
      Schedule(sem, "SetupCollidingJacobians_Articulated", [&reg, &collidingJacsStorage, e]() {
        for (auto link : reg.get<CGroupMembers const>(e).actors) {
          if (HasCollJacs(collidingJacsStorage, link)) {
            ecs::TryInvokeOnEntity(
                &articulated::rigid::SetupCollidingJacobians<kTimeStep>, reg, link);
          }
        }
      });
    }
  }

  // Rigid actors are pretty cheap. Do them on the calling thread.
  for (auto e : descendants.rigidActors) {
    if (HasCollJacs(collidingJacsStorage, e)) {
      ecs::TryInvokeOnEntity(&rigid::SetupCollidingJacobians<kTimeStep>, reg, e);
    }
  }
}

// This system is only used for far SDF queries.
void mochi::FarSdfCollisionDetection(
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
  CollisionDetection</*kIsSync*/ false, TimeStep::Current, kAllowFarSdfQuery>(reg, e);
  CollisionDetection</*kIsSync*/ true, TimeStep::Current, kAllowFarSdfQuery>(reg, e);
}

// Whether the contact between a colliding actor's partition and a collider is assembled from
// contact sums, without per-contact Jacobian blocks: contact between rigid actors (including
// articulated links), and contact of articulated skins with rigid or static colliders. For blended
// skins, only partitions without a soft actor, whose slices are all skinning bones.
static bool UseContactSums(
    entt::registry const& reg,
    entt::entity colliding,
    entt::entity collider,
    int collidingPartitionId) {
  if (reg.all_of<TagRigidActor>(colliding)) {
    return reg.all_of<TagRigidActor>(collider);
  }
  if (reg.all_of<TagArticulatedActor>(colliding) &&
      reg.any_of<TagRigidActor, TagStaticActor>(collider)) {
    auto const& descriptors =
        reg.get<CContactPartitions const>(colliding)[collidingPartitionId].GetDofDescriptors();
    bool const hasSoftSlice =
        reg.all_of<TagBlendedActor>(colliding) && std::get<int>(descriptors[1]) >= 0;
    return !hasSoftSlice && !std::get<DynamicArray<int>>(descriptors[0]).empty();
  }
  return false;
}

static void RegisterContactJacobians(entt::registry& reg, Span<entt::entity const> actors) {
  MOCHI_PROFILE_SCOPE();

  auto& collidingJacsStorage = reg.storage<CCollJacs<CollRole::Colliding>>();
  auto& colliderJacsStorage = reg.storage<CCollJacs<CollRole::Collider>>();
  auto& activeCollsAsyncStorage =
      reg.storage<CActiveCollisions</*kIsSync*/ false, TimeStep::Current>>();
  auto& activeCollsSyncStorage =
      reg.storage<CActiveCollisions</*kIsSync*/ true, TimeStep::Current>>();

  // At input, the contact Jacobians are from the previous assembly. Invalidate them, since their
  // queries point to the previous collision results and they may no longer be in contact.
  for (auto const& e : actors) {
    if (auto* collidingJacs = TryGet(collidingJacsStorage, e)) {
      for (auto& collJac : *collidingJacs) {
        MOCHI_ASSERT_VERBOSE(collJac.query, "Expected a registered contact Jacobian.");
        collJac.Invalidate();
      }
    }
    if (auto* colliderJacs = TryGet(colliderJacsStorage, e)) {
      for (auto& collJac : *colliderJacs) {
        MOCHI_ASSERT_VERBOSE(collJac.query, "Expected a registered contact Jacobian.");
        collJac.Invalidate();
      }
    }
  }

  // Register the contact Jacobians for all the active collisions.
  for (auto const& e : actors) {
    auto* activeCollsAsync = TryGet(activeCollsAsyncStorage, e);
    auto* activeCollsSync = TryGet(activeCollsSyncStorage, e);
    if (!activeCollsAsync && !activeCollsSync) {
      continue; // No active collisions.
    }

    auto& collidingJacs = collidingJacsStorage.get(e);
    int const prevCollidingJacs = isize(collidingJacs);
    collidingJacs.reserve(
        prevCollidingJacs + (activeCollsAsync ? isize(*activeCollsAsync) : 0) +
        (activeCollsSync ? isize(*activeCollsSync) : 0));

    auto registerJacobians = [&](auto& activeColls, bool isSync) {
      for (auto& activeColl : activeColls) {
        int const partitionId = activeColl.collisionResult.collidingPartitionId;

        // Register colliding Jacobian (owned by this entity e).
        bool wasPreviouslyActive = false;
        for (int i = 0; i < prevCollidingJacs; ++i) {
          auto& collJac = collidingJacs[i];
          if ((collJac.otherEntity == activeColl.colliderEntity) &&
              (collJac.collidingPartitionId == partitionId)) {
            wasPreviouslyActive = true;
            activeColl.collidingJacId = i;
            collJac.isSync = isSync;
            collJac.query = &activeColl.collisionResult;
            MOCHI_ASSERT_VERBOSE(
                collJac.useContactSums ==
                    UseContactSums(reg, e, activeColl.colliderEntity, partitionId),
                "Inconsistent 'useContactSums' flag.");
            break;
          }
        }
        if (!wasPreviouslyActive) {
          activeColl.collidingJacId = isize(collidingJacs);
          collidingJacs.emplace_back(
              isSync,
              &activeColl.collisionResult,
              UseContactSums(reg, e, activeColl.colliderEntity, partitionId),
              activeColl.colliderEntity,
              partitionId);
        }
        if (isSync) {
          // Register collider Jacobian (owned by the other entity).
          MOCHI_ASSERT_VERBOSE(
              Contains(actors, activeColl.colliderEntity), "Invalid collider entity.");
          auto& colliderJacs = colliderJacsStorage.get(activeColl.colliderEntity);
          wasPreviouslyActive = false;
          for (int i = 0; i < isize(colliderJacs); ++i) {
            auto& collJac = colliderJacs[i];
            if ((collJac.otherEntity == e) && (collJac.collidingPartitionId == partitionId)) {
              // Recycle this JacData
              wasPreviouslyActive = true;
              activeColl.colliderJacId = i;
              collJac.isSync = isSync;
              collJac.query = &activeColl.collisionResult;
              MOCHI_ASSERT_VERBOSE(
                  collJac.useContactSums ==
                      UseContactSums(reg, e, activeColl.colliderEntity, partitionId),
                  "Inconsistent 'useContactSums' flag.");
              break;
            }
          }
          if (!wasPreviouslyActive) {
            activeColl.colliderJacId = isize(colliderJacs);
            colliderJacs.emplace_back(
                isSync,
                &activeColl.collisionResult,
                UseContactSums(reg, e, activeColl.colliderEntity, partitionId),
                e,
                partitionId);
          }
        }
      }
    };

    if (activeCollsAsync) {
      registerJacobians(*activeCollsAsync, /*isSync*/ false);
    }
    if (activeCollsSync) {
      registerJacobians(*activeCollsSync, /*isSync*/ true);
    }
  }

  // Remove contact Jacobians that were previously active (in contact) but are no longer active,
  // i.e., whose query was not registered again.
  auto removeInactiveJacs = [](auto& jacs, auto& jacIdMapping) {
    jacIdMapping.resize(isize(jacs), -1);
    int writeIdx = 0;
    for (int readIdx = 0; readIdx < isize(jacs); ++readIdx) {
      if (jacs[readIdx].query) {
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

    if (auto* colliderJacs = TryGet(colliderJacsStorage, e)) {
      colliderJacIdMappings.emplace_back(e, JacIdArray(&filoAllocator)); // Insert pair
      removeInactiveJacs(*colliderJacs, colliderJacIdMappings.back().second);
    }

    JacIdArray collidingJacIdMapping(&filoAllocator);

    if (auto* collidingJacs = TryGet(collidingJacsStorage, e)) {
      removeInactiveJacs(*collidingJacs, collidingJacIdMapping);
    }

    // Update colliding Jacobian IDs.
    if (auto* activeCollsAsync = TryGet(activeCollsAsyncStorage, e)) {
      for (auto& activeColl : *activeCollsAsync) {
        MOCHI_ASSERT_VERBOSE(collidingJacIdMapping[activeColl.collidingJacId] >= 0);
        activeColl.collidingJacId = collidingJacIdMapping[activeColl.collidingJacId];
      }
    }
    if (auto* activeCollsSync = TryGet(activeCollsSyncStorage, e)) {
      for (auto& activeColl : *activeCollsSync) {
        MOCHI_ASSERT_VERBOSE(collidingJacIdMapping[activeColl.collidingJacId] >= 0);
        activeColl.collidingJacId = collidingJacIdMapping[activeColl.collidingJacId];
      }
    }
  }

  // Update collider Jacobian IDs now that colliderJacIdMappings has been computed for all actors.
  for (auto const& e : actors) {
    if (auto* activeCollsSync = TryGet(activeCollsSyncStorage, e)) {
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
  auto const& colliderJacsStorage = std::as_const(reg).storage<CCollJacs<CollRole::Collider>>();

  ScheduleForActorsWithCollJacs(
      sem,
      "deformable::SetupColliderJacobians (soft)",
      &deformable::SetupColliderJacobians,
      reg,
      colliderJacsStorage,
      descendants.softActors);

  ScheduleForActorsWithCollJacs(
      sem,
      "deformable::SetupColliderJacobians (shell)",
      &deformable::SetupColliderJacobians,
      reg,
      colliderJacsStorage,
      descendants.shellActors);

  ScheduleForActorsWithCollJacs(
      sem,
      "deformable::SetupColliderJacobians (rod)",
      &deformable::SetupColliderJacobians,
      reg,
      colliderJacsStorage,
      descendants.rodActors);

  ScheduleForActorsWithCollJacs(
      sem,
      "articulated::rigid::SetupColliderJacobians",
      &articulated::rigid::SetupColliderJacobians<kTimeStep>,
      reg,
      colliderJacsStorage,
      descendants.articulatedLinkActors);

  // Rigid actors are pretty cheap. Do them on the calling thread.
  for (auto e : descendants.rigidActors) {
    if (HasCollJacs(colliderJacsStorage, e)) {
      ecs::TryInvokeOnEntity(&rigid::SetupColliderJacobians<kTimeStep>, reg, e);
    }
  }
}

// Whether the residual of a contact Jacobian slice is assembled in link space. This is the case for
// slices with an internal DoF offset (articulated links and skinning bones), whose residual is
// projected to joint space after assembly. Other slices assemble into the solver-space residual.
static bool HasLinkResidual(ContactJac const& jac) {
  return jac.internalDofOffset != kSentinelIndex;
}

// Link residual slot of slice 'a' of a contact pair. 'linkRes' holds 6 values per slice, and only
// the slots of the slices with link residuals are written.
template <typename T>
static ColumnVectorView<T, RigidSize::kDAll> LinkResidualSlot(ColumnVectorView<T> linkRes, int a) {
  return linkRes.template Slice<RigidSize::kDAll>(RigidSize::kDAll * a, RigidSize::kDAll);
}

// Adds the link residual slots of the slices in 'jacs' that have them to the link-space residual of
// their articulation, at their internal DoF offsets.
static void AddLinkResiduals(
    Span<ContactJac const*> jacs,
    ColumnVectorView<real const> linkRes,
    ArticulatedLinkResidual* linkResidual) {
  MOCHI_ASSERT_VERBOSE(linkRes.Rows() == RigidSize::kDAll * isize(jacs));
  for (int a = 0; a < isize(jacs); ++a) {
    if (!HasLinkResidual(*jacs[a])) {
      continue;
    }
    MOCHI_ASSERT_VERBOSE(linkResidual, "Expected a link-space residual destination.");
    linkResidual->AddLinkResidual(jacs[a]->internalDofOffset, LinkResidualSlot(linkRes, a));
  }
}

namespace {

// Appends the kBlockSize-block indices of the DoFs that an element maps to, in increasing order.
template <int kBlockSize>
void AppendLinearContactSkinElementIndices(
    DynamicArray<int>& outIndices,
    CLinearContactSkinElementMap const& elementMap,
    int elementIndex,
    int dofOffset) {
  static_assert(kBlockSize == 1 || kBlockSize == kSpaceDim3);
  int previousBlock = -1;
  for (int physicsNode : elementMap.PhysicsNodeIndices(elementIndex)) {
    int const firstDof = dofOffset + kSpaceDim3 * physicsNode;
    for (int component = 0; component < kSpaceDim3; ++component) {
      int const blockIndex = (firstDof + component) / kBlockSize;
      if (blockIndex != previousBlock) {
        outIndices.push_back(blockIndex);
        previousBlock = blockIndex;
      }
    }
  }
}

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

  // Deferred assemble. If the slice, with index 'a' in its contact pair, has a link residual, it is
  // written to its slot in linkRes.
  void DeferredAssemble(ColumnVectorView<real> outRes, ColumnVectorView<real> linkRes, int a) {
    MOCHI_ASSERT_VERBOSE(
        _jac.hasSharedDoFs || !HasLinkResidual(_jac), "Link residuals require shared DoFs.");
    if (!_jac.hasSharedDoFs) {
      return;
    }
    if (HasLinkResidual(_jac)) {
      LinkResidualSlot(linkRes, a) = _resTerm.Slice<RigidSize::kDAll>(0, RigidSize::kDAll);
    } else {
      AssembleTerm(0, _resTerm, outRes);
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

// View of a block with compile-time rows and cols, each of which may be krylov::kDynamic.
template <int kRows, int kCols, typename Scalar>
MOCHI_FORCE_INLINE RowMatrixView<Scalar, kRows, kCols> AsFixedSize(RowMatrixView<Scalar> block) {
  if constexpr (kRows != krylov::kDynamic) {
    MOCHI_ASSERT_VERBOSE(block.Rows() == kRows, "Inconsistent number of rows.");
  }
  if constexpr (kCols != krylov::kDynamic) {
    MOCHI_ASSERT_VERBOSE(block.Cols() == kCols, "Inconsistent number of columns.");
  }
  return {block.Data(), block.Rows(), block.Cols(), block.LeadDim()};
}

// Accumulates the dresidual blocks among the contact Jacobians with shared DoFs into a dense block
// over the union of their state DoFs, which the ContactDResWriter assembles with a single sparse
// scatter instead of one per pair of Jacobians. The union block is dense in the dresidual, since
// every pair of these Jacobians couples all their DoFs.
class SharedDResAccumulator {
 public:
  SharedDResAccumulator(Span<ContactJac const*> jacs, Allocator* allocator)
      : _dofs(CollectSharedDoFs(jacs, allocator)),
        _groups(CreateIndexGroups(MakeConstSpan(_dofs), /*forceTriplets*/ false, allocator)),
        _unionOffsets(allocator),
        _unionIndices(allocator) {
    _unionOffsets.resize_noinit(jacs.size() + 1);
    int numIndices = 0;
    for (int i = 0; i < isize(jacs); ++i) {
      _unionOffsets[i] = numIndices;
      if (jacs[i]->hasSharedDoFs) {
        numIndices += jacs[i]->nDoFsState;
      }
    }
    _unionOffsets[jacs.size()] = numIndices;
    _unionIndices.resize_noinit(numIndices);
    for (int i = 0; i < isize(jacs); ++i) {
      if (jacs[i]->hasSharedDoFs) {
        int* indices = _unionIndices.data() + _unionOffsets[i];
        for (int dof : jacs[i]->Inds(0)) {
          *indices++ =
              static_cast<int>(std::lower_bound(_dofs.begin(), _dofs.end(), dof) - _dofs.begin());
        }
      }
    }
    _dres.Reset(isize(_dofs), isize(_dofs), allocator);
    _dres.SetZero();
  }

  // Add the block between the state DoFs of the shared Jacobians rowJac and colJac and, if
  // kAddTranspose, its transpose between colJac and rowJac.
  template <bool kAddTranspose>
  void Add(int rowJac, int colJac, RowMatrixView<real const> block) {
    Span<int const> rows = UnionIndices(rowJac);
    Span<int const> cols = UnionIndices(colJac);
    MOCHI_ASSERT_VERBOSE(block.Rows() == isize(rows) && block.Cols() == isize(cols));
    for (int r = 0; r < isize(rows); ++r) {
      for (int c = 0; c < isize(cols); ++c) {
        real const value = block(r, c);
        _dres(rows[r], cols[c]) += value;
        if constexpr (kAddTranspose) {
          _dres(cols[c], rows[r]) += value;
        }
      }
    }
  }

  // Block between DoFs() and themselves.
  RowMatrixView<real const> Block() const {
    return _dres;
  }

  // Sorted union of the state DoFs of the shared Jacobians.
  Span<int const> DoFs() const {
    return MakeConstSpan(_dofs);
  }

  // Index groups of DoFs(), with sources indexing DoFs().
  Span<IndexGroup const> Groups() const {
    return MakeConstSpan(_groups);
  }

  // Index in DoFs() of each state DoF of the shared Jacobian jacIndex. Empty for Jacobians with
  // per-point DoFs.
  Span<int const> UnionIndices(int jacIndex) const {
    return {
        _unionIndices.data() + _unionOffsets[jacIndex],
        _unionIndices.data() + _unionOffsets[jacIndex + 1]};
  }

 private:
  static DynamicArray<int> CollectSharedDoFs(Span<ContactJac const*> jacs, Allocator* allocator) {
    int numDoFs = 0;
    for (auto const* jac : jacs) {
      if (jac->hasSharedDoFs) {
        numDoFs += jac->nDoFsState;
      }
    }
    DynamicArray<int> dofs(allocator);
    dofs.reserve(numDoFs);
    for (auto const* jac : jacs) {
      if (jac->hasSharedDoFs) {
        for (int dof : jac->Inds(0)) {
          dofs.push_back(dof);
        }
      }
    }
    std::sort(dofs.begin(), dofs.end());
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
    return dofs;
  }

  // Members are declared in allocation order to satisfy the FILO allocator.
  DynamicArray<int> _dofs;
  IndexGroups _groups;
  // Offset of each Jacobian in _unionIndices, followed by the total size.
  DynamicArray<int> _unionOffsets;
  // Concatenated UnionIndices() of the Jacobians.
  DynamicArray<int> _unionIndices;
  RowMatrix<real> _dres;
};

// Gathers the dresidual blocks of a batch of contact points between a contact Jacobian with
// per-point DoFs and the Jacobians in a SharedDResAccumulator, which the ContactDResWriter
// assembles with two sparse scatters per batch, instead of two per shared Jacobian.
class NonSharedDResStrip {
 public:
  NonSharedDResStrip(
      ContactJac const& jac,
      int jacIndex,
      SharedDResAccumulator const& shared,
      Allocator* allocator)
      : _jacIndex(jacIndex), _shared(shared) {
    MOCHI_ASSERT_VERBOSE(!jac.hasSharedDoFs && !jac.JacAux());
    _dres.Reset(isize(shared.DoFs()), jac.nDoFsState, allocator);
    _dres.SetZero();
  }

  // Index of the contact Jacobian in the pair.
  int JacIndex() const {
    return _jacIndex;
  }

  // Add the block between the state DoFs of the shared Jacobian sharedJac and the batch DoFs.
  void AddShared(int sharedJac, RowMatrixView<real const> block) {
    Span<int const> rows = _shared.UnionIndices(sharedJac);
    MOCHI_ASSERT_VERBOSE(block.Rows() == isize(rows) && block.Cols() == _dres.Cols());
    for (int r = 0; r < isize(rows); ++r) {
      _dres.Row(rows[r]) += block.Row(r);
    }
  }

  // Block between the shared DoFs (rows) and the batch DoFs (cols).
  RowMatrixView<real const> Block() const {
    return _dres;
  }

  // Reset the block for the next batch.
  void Clear() {
    _dres.SetZero();
  }

 private:
  int _jacIndex;
  SharedDResAccumulator const& _shared;
  RowMatrix<real> _dres;
};

// Assembles the dresidual blocks of a contact pair, given between the internal DoFs of its contact
// Jacobians. It maps them to state DoFs with the auxiliary Jacobians, if any. With groupShared, the
// blocks of all Jacobians with shared DoFs, which typically overlap (e.g. the bones of a skinned
// actor), are accumulated for a single scatter, and the blocks of each batch between each Jacobian
// with per-point DoFs and those are gathered in a strip. Other blocks are assembled directly.
class ContactDResWriter {
 public:
  ContactDResWriter(
      AnyMatrixView<real>& outDRes,
      Span<ContactJac const*> jacs,
      bool groupShared,
      Allocator* allocator)
      : _outDRes(outDRes), _jacs(jacs), _allocator(allocator), _strips(allocator) {
    if (groupShared) {
      _shared.emplace(jacs, allocator);
      // Each actor has at most one Jacobian with per-point DoFs.
      _strips.reserve(2);
      for (int i = 0; i < isize(jacs); ++i) {
        if (!jacs[i]->hasSharedDoFs) {
          MOCHI_ASSERT_VERBOSE(isize(_strips) < 2, "At most 2 Jacobians with per-point DoFs.");
          _strips.emplace_back(*jacs[i], i, _shared.value(), allocator);
        }
      }
    }
  }

  // Strips reference the accumulator, so the writer must not be copied or moved.
  MOCHI_DECLARE_NO_COPY_NO_MOVE(ContactDResWriter);

  // Assemble the blocks gathered during a batch of contact points that share the DoFs of
  // firstContact.
  void EndBatch(int firstContact) {
    for (auto& strip : _strips) {
      ContactJac const& jac = *_jacs[strip.JacIndex()];
#if MOCHI_ASSERT_VERBOSE_ENABLED
      for (int dof : jac.Inds(firstContact)) {
        MOCHI_ASSERT_VERBOSE(
            !std::binary_search(_shared->DoFs().begin(), _shared->DoFs().end(), dof),
            "Per-point DoFs must be disjoint from shared DoFs.");
      }
#endif
      WriteOffDiag(_shared->Groups(), jac.IndGroups(firstContact), strip.Block());
      strip.Clear();
    }
  }

  // Assemble the blocks accumulated over all the contact points.
  void Finish() {
    if (_shared) {
      MatAddSubBlocks(_outDRes, _shared->Groups(), _shared->Groups(), _shared->Block());
    }
  }

  // Add the diagonal block of Jacobian i, between its internal DoFs. The block must be symmetric.
  // contact selects the DoFs of per-point Jacobians.
  void AddDiag(int i, RowMatrixView<real const> block, int contact) {
    ContactJac const& jac = *_jacs[i];
    if (!jac.JacAux()) {
      AddDiagState(i, block, contact);
      return;
    }
    // These products may not be optimally efficient, but they should not happen as often as the
    // products of the assemblers.
    Matrix<real, krylov::kDynamic, RigidSize::kDAll> const temp(
        jac.JacAux().Transpose() * AsFixedSize<RigidSize::kDAll, RigidSize::kDAll>(block),
        _allocator);
    Matrix<real> const dres(temp * jac.JacAux(), _allocator);
    AddDiagState(i, dres.Transpose(), contact);
  }

  // Add the block between the internal DoFs of Jacobians a (rows) and b (cols), and its transpose
  // between b and a. If only one of them has shared DoFs, it must be a. contact selects the DoFs of
  // per-point Jacobians.
  void AddOffDiag(int a, int b, RowMatrixView<real const> blockAB, int contact) {
    ContactJac const& jacA = *_jacs[a];
    ContactJac const& jacB = *_jacs[b];
    MOCHI_ASSERT_VERBOSE(jacA.hasSharedDoFs || !jacB.hasSharedDoFs);
    if (jacA.JacAux() || jacB.JacAux()) {
      AddOffDiagState(a, b, ToStateDoFs(jacA, jacB, blockAB), contact);
      return;
    }
    AddOffDiagState(a, b, blockAB, contact);
  }

 private:
  // Add the diagonal block of Jacobian i, between its state DoFs.
  void AddDiagState(int i, RowMatrixView<real const> block, int contact) {
    if (_shared && _jacs[i]->hasSharedDoFs) {
      _shared->Add</*kAddTranspose*/ false>(i, i, block);
    } else {
      Span<IndexGroup const> groups = _jacs[i]->IndGroups(contact);
      MatAddSubBlocks(_outDRes, groups, groups, block);
    }
  }

  // Add the block between the state DoFs of Jacobians a (rows) and b (cols), and its transpose
  // between b and a.
  void AddOffDiagState(int a, int b, RowMatrixView<real const> blockAB, int contact) {
    if (_shared && _jacs[a]->hasSharedDoFs) {
      if (_jacs[b]->hasSharedDoFs) {
        _shared->Add</*kAddTranspose*/ true>(a, b, blockAB);
      } else {
        StripOf(b).AddShared(a, blockAB);
      }
    } else {
      WriteOffDiag(_jacs[a]->IndGroups(contact), _jacs[b]->IndGroups(contact), blockAB);
    }
  }

  // Add a block between the DoFs in groupsA (rows) and groupsB (cols), and its transpose.
  void WriteOffDiag(
      Span<IndexGroup const> groupsA,
      Span<IndexGroup const> groupsB,
      RowMatrixView<real const> blockAB) {
    RowMatrix<real> const blockBA(blockAB.Transpose(), _allocator);
    MatAddSubBlocks(_outDRes, groupsB, groupsA, blockBA);
    MatAddSubBlocks(_outDRes, groupsA, groupsB, blockAB);
  }

  // Map a block between the internal DoFs of Jacobians a (rows) and b (cols), at least one of which
  // has an auxiliary Jacobian, to their state DoFs.
  RowMatrix<real>
  ToStateDoFs(ContactJac const& jacA, ContactJac const& jacB, RowMatrixView<real const> blockAB) {
    MOCHI_ASSERT_VERBOSE(jacA.JacAux() || jacB.JacAux());
    RowMatrix<real> stateAB(
        jacA.JacAux() ? jacA.nDoFsState : blockAB.Rows(),
        jacB.JacAux() ? jacB.nDoFsState : blockAB.Cols(),
        _allocator);
    RowMatrixView<real> result = stateAB;
    // The internal DoFs of a Jacobian with an auxiliary Jacobian are its 6 rigid DoFs. If the other
    // Jacobian also has 6 internal DoFs (e.g. a rigid actor), the products are fixed-size.
    if (!jacB.JacAux()) {
      if (jacB.nDoFsInternal == RigidSize::kDAll) {
        auto fixedResult = AsFixedSize<krylov::kDynamic, RigidSize::kDAll>(result);
        fixedResult =
            jacA.JacAux().Transpose() * AsFixedSize<RigidSize::kDAll, RigidSize::kDAll>(blockAB);
      } else {
        result =
            jacA.JacAux().Transpose() * AsFixedSize<RigidSize::kDAll, krylov::kDynamic>(blockAB);
      }
    } else if (!jacA.JacAux()) {
      if (jacA.nDoFsInternal == RigidSize::kDAll) {
        auto fixedResult = AsFixedSize<RigidSize::kDAll, krylov::kDynamic>(result);
        fixedResult = AsFixedSize<RigidSize::kDAll, RigidSize::kDAll>(blockAB) * jacB.JacAux();
      } else {
        result = AsFixedSize<krylov::kDynamic, RigidSize::kDAll>(blockAB) * jacB.JacAux();
      }
    } else {
      RowMatrix<real, krylov::kDynamic, RigidSize::kDAll> const temp(
          jacA.JacAux().Transpose() * AsFixedSize<RigidSize::kDAll, RigidSize::kDAll>(blockAB),
          _allocator);
      result = temp * jacB.JacAux();
    }
    return stateAB;
  }

  // Strip of Jacobian i, which has per-point DoFs.
  NonSharedDResStrip& StripOf(int i) {
    MOCHI_ASSERT_VERBOSE(!_strips.empty());
    NonSharedDResStrip& strip = _strips[0].JacIndex() == i ? _strips[0] : _strips.back();
    MOCHI_ASSERT_VERBOSE(strip.JacIndex() == i);
    return strip;
  }

  AnyMatrixView<real>& _outDRes;
  Span<ContactJac const*> _jacs;
  Allocator* _allocator;
  // Members below are declared in allocation order to satisfy the FILO allocator.
  // Accumulator of the Jacobians with shared DoFs, and strips of the Jacobians with per-point DoFs,
  // if grouping.
  std::optional<SharedDResAccumulator> _shared;
  DynamicArray<NonSharedDResStrip> _strips;
};

// Computes the dresidual block between the internal DoFs of contact Jacobians B (rows) and A
// (cols), for batches of contact points, and passes it to a ContactDResWriter. With A == B, it
// computes the diagonal block. If both Jacobians have shared DoFs, the block is accumulated over
// all the batches and passed in DeferredAssemble(). Otherwise, it is passed after each batch.
class ContactDResAssembler {
 public:
  // If only one of the Jacobians has shared DoFs, it is made B.
  ContactDResAssembler(
      ContactJac const& jacA,
      ContactJac const& jacB,
      int jacIndexA,
      int jacIndexB,
      ContactDResTransformer const& dresJa,
      ContactDResTransformer const& dresJb,
      Allocator* allocator)
      : _switch(jacA.hasSharedDoFs && !jacB.hasSharedDoFs),
        _jacA(_switch ? jacB : jacA),
        _jacB(_switch ? jacA : jacB),
        _jacIndexA(_switch ? jacIndexB : jacIndexA),
        _jacIndexB(_switch ? jacIndexA : jacIndexB),
        _dresJa(_switch ? dresJb : dresJa),
        _accumulate(jacA.hasSharedDoFs && jacB.hasSharedDoFs),
        _dresba(_jacB.nDoFsInternal, _jacA.nDoFsInternal, allocator) {
    if (_accumulate) {
      _dresba.SetZero();
    }
  }

  // Compute and possibly assemble
  void ComputeTerm(Span<int const> contactIndices, ContactDResWriter& writer) {
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
    if (_accumulate) {
      // Compute and add for deferred assembly
      for (int i = 0; i < isize(contactIndices); ++i) {
        _dresba += _jacB.Jac(contactIndices[i]).Transpose() * _dresJa.Get(i);
      }
    } else {
      // Compute and assemble
      int cIndex0 = contactIndices[0];
      _dresba = _jacB.Jac(cIndex0).Transpose() * _dresJa.Get(0);
      for (int i = 1; i < isize(contactIndices); ++i) {
        _dresba += _jacB.Jac(contactIndices[i]).Transpose() * _dresJa.Get(i);
      }
      Write(writer, cIndex0);
    }
  }

  // Deferred assemble
  void DeferredAssemble(ContactDResWriter& writer) {
    if (_accumulate) {
      Write(writer, 0);
    }
  }

 private:
  void Write(ContactDResWriter& writer, int contact) {
    if (_jacIndexA == _jacIndexB) {
      writer.AddDiag(_jacIndexA, _dresba, contact);
    } else {
      writer.AddOffDiag(_jacIndexB, _jacIndexA, _dresba, contact);
    }
  }

  // bool to switch objects internally. If only one of the actor contact Jacobians has shared DoFs,
  // then it's actor B
  bool _switch;

  // Actor contact jacobians
  ContactJac const& _jacA;
  ContactJac const& _jacB;

  // Indices of the contact Jacobians in the pair
  int _jacIndexA;
  int _jacIndexB;

  // dresidual transformer of A
  ContactDResTransformer const& _dresJa;

  // Whether both Jacobians have shared DoFs, and the block is accumulated over all the batches
  bool _accumulate;

  // Local result holder, between the internal DoFs of B (rows) and A (cols)
  RowMatrix<real> _dresba;
};

} // namespace

// Writes the residual of contact Jacobian slice 'a', with 6 rigid internal DoFs, to its link
// residual slot, or adds it to the solver-space residual if it has none.
static void AddRigidSliceResidual(
    Span<ContactJac const*> jacs,
    int a,
    ColumnVector<real, RigidSize::kDAll> const& res,
    ColumnVectorView<real> outRes,
    ColumnVectorView<real> linkRes) {
  ContactJac const& jac = *jacs[a];
  MOCHI_ASSERT_VERBOSE(jac.hasSharedDoFs && jac.nDoFsInternal == RigidSize::kDAll);
  if (HasLinkResidual(jac)) {
    LinkResidualSlot(linkRes, a) = res;
    return;
  }
  MOCHI_ASSERT_VERBOSE(
      jac.IndGroups(0).size() == 1 && jac.IndGroups(0)[0].count == RigidSize::kDAll,
      "A rigid slice without link residual must map to 6 consecutive DoFs.");
  outRes.Slice<RigidSize::kDAll>(jac.IndGroups(0)[0].dst, RigidSize::kDAll) += res;
}

// Sync contact assembly between a pair of rigid actors (including articulated rigid), given the
// RigidContactSums of their contact points about the collider's center of mass. Faster than the
// generic assembly since (1) it operates with compile-time size matrices, and (2) it skips the
// products that involve the identity sub-blocks in the contact Jacobians.
// NOTES:
// - Only supported for sync contact. Async contact is assembled through rigid::EntityAssemble.
// - The 1st contact Jacobian must be the colliding Jacobian, and the 2nd must be the collider
//   Jacobian. Those with link residuals write them to their slots in linkRes.
template <GradTarget kGradTarget>
static void AssembleSyncRigidContactSums(
    ContactAssemblyReg reg,
    entt::entity colliding,
    entt::entity collider,
    RigidContactSums const& sums,
    Span<ContactJac const*> jacs,
    Allocator* filoAllocator,
    double* outObj,
    ColumnVectorView<real> outRes,
    ColumnVectorView<real> linkRes,
    AnyMatrixView<real> outDRes) {
  bool const assemObj = (outObj != nullptr);
  bool const assemRes = !outRes.empty() || !linkRes.empty();
  bool const assemDRes = (GetNumValues(outDRes) > 0);

  if (assemObj) {
    *outObj += sums.energy;
  }

  if (!assemRes && !assemDRes) {
    return;
  }

  // Check assumptions on Jacobians.
#if MOCHI_ASSERT_VERBOSE_ENABLED
  MOCHI_ASSERT_VERBOSE(jacs.size() == 2 && jacs[0] && jacs[1], "Requires 2 contact Jacobians.");
  MOCHI_ASSERT_VERBOSE(jacs[0]->groupsInitialized && jacs[1]->groupsInitialized);
  MOCHI_ASSERT_VERBOSE(jacs[0]->hasSharedDoFs && jacs[1]->hasSharedDoFs);
  MOCHI_ASSERT_VERBOSE(
      jacs[0]->JacAux() ||
      (jacs[0]->IndGroups(0).size() == 1 && jacs[0]->IndGroups(0)[0].count == 6));
  MOCHI_ASSERT_VERBOSE(
      jacs[1]->JacAux() ||
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
  // The only point-dependent term is sk(pB' - comB'), which the sums (with lever arms pB' - comB')
  // account for. All other terms are applied here. For convenience, the transpose Jacobians are:
  // Trans jacT A: rotB
  // Trans jacT B: -rotB
  // Rot jacT A: rotB * sk(pB' - comB') - sk(comA - comB) * rotB
  // Rot jacT B: - rotB * sk(pB' - comB')
  // clang-format on
  Vec4r comA = reg.template get<CRigidState<kTimeStep> const>(colliding).value.VGetTranslation();
  auto const& stateB = reg.template get<CRigidState<kTimeStep> const>(collider).value;
  auto [rotB, rotBT] = ToVMatrix3x3_WithTranspose(stateB.GetRotation());
  Vec4r comB = stateB.VGetTranslation();

  // NOTE: Memory used from filoAllocator for up to N DoFs per actor is at most
  //       2 * max(N, 6)^2 * sizeof(real), for the temporaries of the ContactDResWriter.

  if (assemRes) {
    MOCHI_PROFILE_SCOPE_N("Residual Assembly");
    Vec4r const res = DotVecMat3x3(-sums.force, rotBT);
    Vec4r const skPRes = DotVecMat3x3(-sums.torque, rotBT);
    ColumnVector<real, 6> resA;
    ColumnVector<real, 6> resB;
    Store(resA.data(), res);
    Store<3>(resA.data() + 3, skPRes - Cross3(comA - comB, res));
    Store(resB.data(), -res);
    Store<3>(resB.data() + 3, -skPRes);
    AddRigidSliceResidual(jacs, 0, resA, outRes, linkRes);
    AddRigidSliceResidual(jacs, 1, resB, outRes, linkRes);
  }

  if (assemDRes) {
    MOCHI_PROFILE_SCOPE_N("DResidual Assembly");
    // The DResidual is composed of 4 6x6 blocks: AA, BB, AB, BA. Each 6x6 block is composed of 4
    // 3x3 sub-blocks. Some of the 3x3 sub-blocks are shared across 6x6 blocks. The assembly is
    // performed on the 3x3 sub-blocks, which are stored into the 6x6 blocks at the end.
    VMatrix3x3r const dres = Dot3x3(rotB, Dot3x3(-sums.dForce, rotBT));
    VMatrix3x3r const skPDres = Dot3x3(rotB, Dot3x3(-sums.skewDForce, rotBT));
    VMatrix3x3r dresSkPNeg = Transpose3x3(skPDres);
    VMatrix3x3r const skPDresSkP = Dot3x3(rotB, Dot3x3(-sums.skewDForceSkew, rotBT));
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

    // The writer maps the 6x6 blocks to state DoFs with the auxiliary Jacobians, if any.
    ContactDResWriter writer(outDRes, jacs, /*groupShared*/ false, filoAllocator);
    writer.AddDiag(0, aaBlock, /*contact*/ 0);
    writer.AddDiag(1, bbBlock, /*contact*/ 0);
    writer.AddOffDiag(0, 1, abBlock, /*contact*/ 0);
  }
}

// Rigid-form slices of the bones of a skinned actor's partition, in the local space of the collider
// of 'query', followed by a rigid-collider slice if 'rigidColliderOrigin' is set (see
// SkinnedContactSlices). The bone rotations are stored in outBoneRotations.
static SkinnedContactSlices GetSkinnedContactSlices(
    entt::registry const& reg,
    entt::entity skinned,
    ContactDetectionResult const& query,
    std::optional<Vec4r> rigidColliderOrigin,
    DynamicArray<VMatrix3x3r>& outBoneRotations) {
  auto const& bonePoints = reg.get<CContactSampleBonePoints const>(skinned);
  auto const& partition = reg.get<CContactPartitions const>(skinned)[query.collidingPartitionId];
  auto const& boneIds = std::get<DynamicArray<int>>(partition.GetDofDescriptors()[0]);
  auto const& linkTransforms =
      reg.get<CArticulatedLinkTransforms<TimeStep::Current> const>(skinned);
  MOCHI_ASSERT_VERBOSE(query.jacColliderFromWorld.size() == 1, "Expected a rigid collider frame.");
  VMatrix3x3r const& colliderFromWorld = query.jacColliderFromWorld[0];
  outBoneRotations.reserve(isize(boneIds));
  for (int boneId : boneIds) {
    outBoneRotations.push_back(
        Dot3x3(colliderFromWorld, ToVMatrix3x3(linkTransforms[boneId].GetRotation())));
  }
  return {
      .pointOffsets = bonePoints.offsets,
      .points = bonePoints.points,
      .boneRotations = outBoneRotations,
      .rigidColliderOrigin = rigidColliderOrigin};
}

// Contact assembly of the rigid-form slices of a skinned actor's partition, given their sums (see
// SkinnedContactSums). 'jacs' holds the structure of the slices, in the order of the sums: the bone
// slices, then the rigid-collider slice if 'hasRigidColliderSlice', with sign -1. With Q the
// collider-from-world rotation:
// - residual of slice a: -sign_a diag(Q, Q)^T [force_a; torque_a]. Slices with link residuals write
//   it to their slots in linkRes, and others add it to their 6 rigid DoFs.
// - dresidual: sum_ab L_a^T B_ab L_b, with L_a = sign_a diag(Q, Q) JacAux_a, B_ab = [[-dForce,
//   -skewBDForce^T], [-skewADForce, skewDForceSkew]] of the pair (a, b), and B_ba = B_ab^T.
// Like the generic assembly, Q is applied to the 6x6 blocks rather than to JacAux, and the
// dresidual blocks are passed to a ContactDResWriter, which groups the slices for a single scatter
// if there are 2+ bone slices, since the links of an articulation share DoFs.
static void AssembleSkinnedContactSums(
    SkinnedContactSums const& sums,
    Span<ContactJac const*> jacs,
    bool hasRigidColliderSlice,
    VMatrix3x3r const& colliderFromWorld,
    Allocator* filoAllocator,
    double* outObj,
    ColumnVectorView<real> outRes,
    ColumnVectorView<real> linkRes,
    AnyMatrixView<real> outDRes) {
  bool const assemRes = !outRes.empty() || !linkRes.empty();
  bool const assemDRes = (GetNumValues(outDRes) > 0);
  if (outObj) {
    *outObj += sums.energy;
  }
  if (!assemRes && !assemDRes) {
    return;
  }
  int const numSlices = sums.numSlices;
  MOCHI_ASSERT_VERBOSE(isize(jacs) == numSlices, "Inconsistent contact Jacobian slices.");
  for ([[maybe_unused]] auto const* jac : jacs) {
    MOCHI_ASSERT_VERBOSE(jac->hasSharedDoFs && jac->nDoFsInternal == RigidSize::kDAll);
  }
  auto sign = [&](int a) { return (hasRigidColliderSlice && a == numSlices - 1) ? -1_r : 1_r; };

  if (assemRes) {
    MOCHI_PROFILE_SCOPE_N("Residual Assembly");
    for (int a = 0; a < numSlices; ++a) {
      // -sign_a diag(Q, Q)^T [force_a; torque_a]
      real const s = -sign(a);
      ColumnVector<real, RigidSize::kDAll> res;
      Store(res.data(), DotVecMat3x3(s * sums.force[a], colliderFromWorld));
      Store<3>(res.data() + 3, DotVecMat3x3(s * sums.torque[a], colliderFromWorld));
      AddRigidSliceResidual(jacs, a, res, outRes, linkRes);
    }
  }

  if (assemDRes) {
    MOCHI_PROFILE_SCOPE_N("DResidual Assembly");
    VMatrix3x3r const worldFromCollider = Transpose3x3(colliderFromWorld);
    auto toWorld = [&](VMatrix3x3r const& m) {
      return Dot3x3(worldFromCollider, Dot3x3(m, colliderFromWorld));
    };
    int const numBoneSlices = hasRigidColliderSlice ? numSlices - 1 : numSlices;
    ContactDResWriter writer(outDRes, jacs, /*groupShared*/ numBoneSlices >= 2, filoAllocator);
    for (int a = 0; a < numSlices; ++a) {
      for (int b = a; b < numSlices; ++b) {
        // D_ab = sign_a sign_b diag(Q, Q)^T B_ab diag(Q, Q)
        int const p = sums.PairIndex(a, b);
        real const s = sign(a) * sign(b);
        RowMatrix<real, 6, 6> block;
        block.Block<3, 3>(0, 0, 3, 3) = AsMatrixView(-s * toWorld(sums.dForce[p]));
        block.Block<3, 3>(0, 3, 3, 3) =
            AsMatrixView(-s * Transpose3x3(toWorld(sums.skewBDForce[p])));
        block.Block<3, 3>(3, 0, 3, 3) = AsMatrixView(-s * toWorld(sums.skewADForce[p]));
        block.Block<3, 3>(3, 3, 3, 3) = AsMatrixView(s * toWorld(sums.skewDForceSkew[p]));
        if (a == b) {
          writer.AddDiag(a, block, /*contact*/ 0);
        } else {
          writer.AddOffDiag(a, b, block, /*contact*/ 0);
        }
      }
    }
    writer.Finish();
  }
}

static void AssembleCollisionResponseRange(
    Interval<int> pointRange,
    ContactDetectionResult const& contactQuery,
    CollisionResponseResult const& collisionResponse,
    Span<real const> intWeights,
    Span<ContactJac const*> jacs,
    int numCollidingJacs, // The colliding Jacobians go first in jacs.
    Allocator* filoAllocator,
    double* outObj,
    ColumnVectorView<real> outRes,
    ColumnVectorView<real> linkRes, // 6 values per slice, for those with link residuals
    AnyMatrixView<real> outDRes) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(pointRange.Valid());
  MOCHI_ASSERT_VERBOSE(
      pointRange.Min() >= 0 && pointRange.Max() < isize(contactQuery.sampleIndices));

  if (pointRange.Size() == 0) {
    return;
  }

  bool assemObj = (outObj != nullptr);
  bool assemRes = !outRes.empty() || !linkRes.empty();
  bool assemDRes = (GetNumValues(outDRes) > 0);

  if (assemRes || assemDRes) {
    for (auto const* jac : jacs) {
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
    for (auto const* jac : jacs) {
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
      for (int a = 0; a < isize(resAssemblers); ++a) {
        resAssemblers[a].DeferredAssemble(outRes, linkRes, a);
      }
    }
  }

  // Assemble dresidual
  if (assemDRes) {
    // Constructed first to be destroyed last, as required by the FILO allocator. The Jacobians with
    // shared DoFs are grouped if the colliding actor has 2+ of them.
    auto const isShared = [](ContactJac const* jac) { return jac->hasSharedDoFs; };
    MOCHI_ASSERT_VERBOSE(numCollidingJacs >= 0 && numCollidingJacs <= isize(jacs));
    bool const groupShared =
        std::count_if(jacs.begin(), jacs.begin() + numCollidingJacs, isShared) >= 2;
    ContactDResWriter writer(outDRes, jacs, groupShared, filoAllocator);

    // One assembler for each pair of contact Jacobians, including each Jacobian with itself.
    DynamicArray<ContactDResTransformer> dresTransformers(filoAllocator);
    DynamicArray<ContactDResAssembler> dresAssemblers(filoAllocator);
    dresTransformers.reserve(jacs.size());
    dresAssemblers.reserve(jacs.size() * (jacs.size() + 1) / 2);
    for (int i = 0; i < isize(jacs); i++) {
      dresTransformers.emplace_back(*jacs[i], filoAllocator);
      for (int j = 0; j <= i; j++) {
        dresAssemblers.emplace_back(
            *jacs[i], *jacs[j], i, j, dresTransformers[i], dresTransformers[j], filoAllocator);
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
          for (auto& dresAssembler : dresAssemblers) {
            dresAssembler.ComputeTerm(contactIndicesInBatch, writer);
          }
          writer.EndBatch(contactIndicesInBatch[0]);
        }
        sBatchBegin = sCandidate;
      }
    }
    {
      MOCHI_PROFILE_SCOPE_N("Deferred assembly");
      for (auto& dresAssembler : dresAssemblers) {
        dresAssembler.DeferredAssemble(writer);
      }
      writer.Finish();
    }

    // Destroy in reverse order to satisfy FILO allocator
    for (int i = isize(jacs) - 1; i >= 0; --i) {
      for (int j = i; j >= 0; --j) {
        dresAssemblers.pop_back();
      }
      dresTransformers.pop_back();
    }
  }
}

void mochi::AssembleCollisionResponse(
    ContactDetectionResult const& contactQuery,
    CollisionResponseResult const& collisionResponse,
    Span<real const> intWeights,
    Span<ContactJac const*> jacs,
    int numCollidingJacs,
    Allocator* filoAllocator,
    double* outObj,
    ColumnVectorView<real> outRes,
    AnyMatrixView<real> outDRes,
    ColumnVectorView<real> outLinkRes) {
  int const numContactPoints = isize(contactQuery.sampleIndices);
  AssembleCollisionResponseRange(
      Interval<int>{0, numContactPoints}, // full range
      contactQuery,
      collisionResponse,
      intWeights,
      jacs,
      numCollidingJacs,
      filoAllocator,
      outObj,
      outRes,
      outLinkRes,
      outDRes);
}

template <
    bool kUpdateOnlyActiveFaces,
    bool kNodeValuesArePositions,
    typename DiscretizationType,
    int kNumFields>
static void UpdateCollisionSamplePositionsImplInternal(
    ColumnVectorView<real const> nodeValues,
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

      NdArray<Vec4r, kNumNodesPerFace> elementNodeValues;
      for (int i = 0; i < kNumNodesPerFace; ++i) {
        elementNodeValues[i] = Load<3, Vec4r>(&nodeValues[dofIndices[i]]);
      }

      // Store the element quad weights scaled (if applicable) by the boundary face weight.
      // These quad weights are copied below in outSamples.weights.
      real const thisFaceWeight =
          kUpdateOnlyActiveFaces ? activeBoundaryFaces->ViewWeights()[boundaryFaceIdx] : 1_r;
      auto const quadWeights = element.quadWeights * thisFaceWeight;

      // Interpolate the node values at quadrature points.
      for (int q = 0; q < kNumQuadsPerFace; ++q) {
        int const sampleIndex = boundaryFaceOffset + q;

        Vec4r samplePos = kNodeValuesArePositions ? Vec4r{} : ToSimd(element.mapEvaluated[q]);
        for (int i = 0; i < kNumNodesPerFace; ++i) {
          samplePos += elementNodeValues[i] * element.basisEvaluated[q][localNodes[i]];
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

  MOCHI_ASSERT(
      !outSamples.bsh.has_value(),
      "SphereOctTree does not currently support refitting. You could rebuild the tree from scratch, "
      "or add a Refit method, but the best solution probably involves a different data structure "
      "(e.g. an AABB tree).");
}

template <bool kUpdateOnlyActiveFaces, typename DiscretizationType, int kNumFields>
void mochi::UpdateCollisionSamplePositionsFromNodeDisplacements(
    ColumnVectorView<real const> currSol,
    DiscretizationType const& boundaryDiscrVariant,
    CActiveBoundaryFaces const* activeBoundaryFaces,
    ContactSamples& outSamples) {
  UpdateCollisionSamplePositionsImplInternal<
      kUpdateOnlyActiveFaces,
      /*kNodeValuesArePositions=*/false,
      DiscretizationType,
      kNumFields>(currSol, boundaryDiscrVariant, activeBoundaryFaces, outSamples);
}

#define MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES(activeFaces, discretization, numFields)          \
  template void mochi::                                                                            \
      UpdateCollisionSamplePositionsFromNodeDisplacements<activeFaces, discretization, numFields>( \
          ColumnVectorView<real const>,                                                            \
          discretization const&,                                                                   \
          CActiveBoundaryFaces const*,                                                             \
          ContactSamples&);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES(false, CFemBoundaryDiscretization, 3);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES(true, CFemBoundaryDiscretization, 3);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES(false, CFemSurfaceDiscretization, 3);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES(true, CFemSurfaceDiscretization, 3);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES(false, CFemSegmentDiscretization, 4);
MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES(true, CFemSegmentDiscretization, 4);
#undef MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES

void mochi::UpdateCollisionSamplePositionsFromNodePositions(
    Span<Real3 const> nodePositions,
    CFemSurfaceDiscretization const& surfaceDiscretization,
    ContactSamples& outSamples) {
  UpdateCollisionSamplePositionsImplInternal<
      /*kUpdateOnlyActiveFaces=*/false,
      /*kNodeValuesArePositions=*/true,
      CFemSurfaceDiscretization,
      kSpaceDim3>(AsConstView(Flatten(nodePositions)), surfaceDiscretization, nullptr, outSamples);
}

template <typename DiscretizationType, TimeStep kTimeStep, int kNumFields>
void mochi::UpdateCollisionSamplePositions(
    ecs::RequiredTag<TagUseContact>,
    ecs::Excluded<TagUseDeformableContactSkin>,
    CFinalDisplacementRef<kTimeStep> const& currSol,
    DiscretizationType const& discretization,
    CActiveBoundaryFaces const* activeBoundaryFaces,
    CContactSamples<kTimeStep>& outSamples) {
  MOCHI_PROFILE_SCOPE();

  if (!activeBoundaryFaces) {
    UpdateCollisionSamplePositionsFromNodeDisplacements<
        /*kUpdateOnlyActiveFaces*/ false,
        DiscretizationType,
        kNumFields>(currSol.value, discretization, activeBoundaryFaces, outSamples);
  } else {
    UpdateCollisionSamplePositionsFromNodeDisplacements<
        /*kUpdateOnlyActiveFaces*/ true,
        DiscretizationType,
        kNumFields>(currSol.value, discretization, activeBoundaryFaces, outSamples);
  }
}

#define MOCHI_SPECIALIZE_UPDATE_COLLISION_SAMPLES(dicretization, timeStep, numFields)      \
  template void mochi::UpdateCollisionSamplePositions<dicretization, timeStep, numFields>( \
      ecs::RequiredTag<TagUseContact>,                                                     \
      ecs::Excluded<TagUseDeformableContactSkin>,                                          \
      CFinalDisplacementRef<timeStep> const&,                                              \
      dicretization const&,                                                                \
      CActiveBoundaryFaces const*,                                                         \
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
    CBoundingVolume const& bounds,
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
    CQueryContactSamples& outQuery) {
  MOCHI_PROFILE_SCOPE();

  auto flatPositions = Flatten(Span<Real3 const>(samplePositions.positions));
  outQuery.contactSamples.assign(flatPositions.begin(), flatPositions.end());
}

struct SdfSample {
  real distance;
  Real3 worldPosition;
  Real3 distanceGrad;
};

template <bool kIsSync>
static void AppendActiveSdfSamplesWorldSpace(
    CActiveCollisions<kIsSync, TimeStep::Current> const& collisions,
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
    CActiveCollisions</*kIsSync*/ false, TimeStep::Current> const* collisionsAsync,
    CActiveCollisions</*kIsSync*/ true, TimeStep::Current> const* collisionsSync,
    CRequiresFarSdfEvaluation const* farSdfEval,
    CConvergenceStatus const* convergenceStatus,
    CQuerySdfDistances& outQuerySdfDistances) {
  MOCHI_PROFILE_SCOPE();

  // Compute the closest contact point for each sample index
  std::unordered_map<int, SdfSample> contactPointDetails;

  // If the solver diverged, contact/SDF data is stale (actor state was rolled back without
  // re-running assembly). Do not populate query.
  if (!convergenceStatus || convergenceStatus->stepStatus != ConvergenceStatus::Diverged) {
    if (collisionsAsync) {
      AppendActiveSdfSamplesWorldSpace(
          *collisionsAsync, collidingSamples, rootTransform.worldFromLocal, contactPointDetails);
    }
    if (collisionsSync) {
      AppendActiveSdfSamplesWorldSpace(
          *collisionsSync, collidingSamples, rootTransform.worldFromLocal, contactPointDetails);
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

// Async contact on deformable actors stores query forces in the colliding actor's local frame,
// except skinned contact, which stores them in the collider's local frame like all other contact.
static bool
IsQueryForceCollidingLocal(entt::registry const& reg, entt::entity colliding, bool isSync) {
  return !isSync && reg.all_of<TagDeformableActor>(colliding) &&
      !reg.all_of<TagSkinnedContact>(colliding);
}

// Like the collider's local frame, the colliding actor's local frame for query forces is that of
// the last solve, so that the forces stay fixed in world space if the actor is moved before the
// next step. A deformable actor's root transform does not change during a step, so that frame is
// the one in worldFromLocalPrev, which setters such as SetRootTransform leave unchanged.
static VMatrix3x3r GetQueryForceJacCollidingFromWorld(
    entt::registry const& reg,
    entt::entity colliding) {
  return ToVMatrix3x3Transpose(GetRootTransform<TimeStep::Previous>(reg, colliding).GetRotation());
}

void mochi::UpdateQueryActiveContactsWorldSpace(
    entt::registry const& reg,
    entt::entity e,
    ecs::RequiredTag<TagQueryActiveContacts>,
    ecs::Excluded<CRequiresFarSdfEvaluation>,
    CActiveCollisions</*kIsSync*/ false, TimeStep::Current> const& collisionsAsync,
    CActiveCollisions</*kIsSync*/ true, TimeStep::Current> const& collisionsSync,
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
                                           bool isSync) {
      // Validate that the force data is available
      int numContacts = isize(collisionResult.posColliding);
      MOCHI_ASSERT(
          numContacts == collisionResult.forcePerUnitArea.size(),
          "Expected a force vector for every contact between this pair of entities");

      ActorHandle collidingActor = GetActorHandle(colliding, sceneHandle);
      ActorHandle colliderActor = GetActorHandle(collider, sceneHandle);

      // Fetch colliding actor data
      auto const forceIsCollidingLocal = IsQueryForceCollidingLocal(reg, colliding, isSync);
      auto const jacCollidingFromWorld = GetQueryForceJacCollidingFromWorld(reg, colliding);
      auto const& transformCurrent = GetRootTransform<TimeStep::Current>(reg, colliding);
      auto const& transformStageStart = GetRootTransform<TimeStep::StageStart>(reg, colliding);
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

        // Transform the force to world space. See @ref IsQueryForceCollidingLocal.
        force = ToReal3(DotVecMat3x3(
            ToSimd(force), forceIsCollidingLocal ? jacCollidingFromWorld : jacColliderFromWorld));

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

    // Add contacts from async contact.
    for (auto const& coll : collisionsAsync) {
      appendActiveContactsForPair(
          e,
          coll.colliderEntity,
          coll.collisionResult,
          outQueryActiveContacts,
          outQueryNodeForces,
          /*isSync*/ false);
    }

    // Add contacts from sync contact.
    for (auto const& coll : collisionsSync) {
      appendActiveContactsForPair(
          e,
          coll.colliderEntity,
          coll.collisionResult,
          outQueryActiveContacts,
          outQueryNodeForces,
          /*isSync*/ true);
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
            jac.otherEntity, e, *jac.query, outQueryActiveContacts, nullptr, /*isSync*/ true);
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
    CActiveCollisions</*kIsSync*/ false, TimeStep::Current> const& collisionsAsync,
    CActiveCollisions</*kIsSync*/ true, TimeStep::Current> const& collisionsSync,
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
                                         bool isSync,
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

    // Define the source entity's transform.
    auto const forceIsCollidingLocal = IsQueryForceCollidingLocal(reg, colliding, isSync);
    auto const jacCollidingFromWorld = GetQueryForceJacCollidingFromWorld(reg, colliding);
    auto const& worldFromColliding = GetRootTransform<TimeStep::Current>(reg, colliding);
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

      // Transform the force to world space. See @ref IsQueryForceCollidingLocal.
      if (forceIsCollidingLocal) {
        force = DotVecMat3x3(force, jacCollidingFromWorld);
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

  // Add contacts from async contact.
  for (auto const& coll : collisionsAsync) {
    appendActiveContactsForPair(
        coll.colliderEntity, coll.collisionResult, /*isSync=*/false, /*isColliderRole=*/false);
  }

  // Add contacts from sync contact.
  for (auto const& coll : collisionsSync) {
    appendActiveContactsForPair(
        coll.colliderEntity, coll.collisionResult, /*isSync=*/true, /*isColliderRole=*/false);
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
          jac.otherEntity, *jac.query, /*isSync=*/true, /*isColliderRole=*/true);
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
    outBlockIndices.resize_noinit(outBlockIndices.size() + fullIndices.size()); // worst case
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

template <int kBlockSize, bool kIsSync>
Graph<int, int> mochi::MakeContactGraph(
    entt::registry const& reg,
    Span<entt::entity const> actors) {
  MOCHI_PROFILE_SCOPE();
  using PtrT = int; // Just "Ptr" would alias a type in the global namespace on Mac/iOS
  DynamicArray<PtrT> ptr;
  DynamicArray<int> targetIndices;

  ptr.push_back(0);
  for (auto e : actors) {
    auto const* activeCollisions =
        reg.try_get<CActiveCollisions<kIsSync, TimeStep::Current> const>(e);
    if (!activeCollisions) {
      continue; // No collisions
    }
    auto const* elementMap = reg.try_get<CLinearContactSkinElementMap const>(e);
    int const mappedDofOffset = elementMap ? reg.get<CDofOffset const>(e).dofsOffset : 0;

    // Traverse colliding entities
    for (auto const& coll : *activeCollisions) {
      [[maybe_unused]] JacData const* colliderJacData = nullptr;
      if constexpr (kIsSync) {
        colliderJacData =
            &reg.get<CCollJacs<CollRole::Collider> const>(coll.colliderEntity)[coll.colliderJacId];
      }

      if (elementMap) {
        auto const& query = coll.collisionResult;
        for (int contactIndex = 0; contactIndex < isize(query.sampleIndices); ++contactIndex) {
          int const elementIndex =
              query.sampleIndices[contactIndex] / elementMap->samplesPerElement;
          int const rowBegin = isize(targetIndices);
          AppendLinearContactSkinElementIndices<kBlockSize>(
              targetIndices, *elementMap, elementIndex, mappedDofOffset);
          if constexpr (kIsSync) {
            colliderJacData->ForEachActiveJac([&](ContactJac const& jac) {
              AppendBlockIndices<kBlockSize>(targetIndices, jac.Inds(contactIndex));
            });
            auto const rowIndices =
                std::ranges::subrange(targetIndices.begin() + rowBegin, targetIndices.end());
            std::ranges::sort(rowIndices);
            targetIndices.erase(std::ranges::unique(rowIndices).begin(), targetIndices.end());
          }
          // Drop a row that repeats the previous one, e.g. for consecutive contacts on one element.
          int const previousRowBegin = isize(ptr) > 1 ? ptr[isize(ptr) - 2] : rowBegin;
          auto* const rowStart = targetIndices.begin() + rowBegin;
          if (std::equal(
                  targetIndices.begin() + previousRowBegin,
                  rowStart,
                  rowStart,
                  targetIndices.end())) {
            targetIndices.resize(rowBegin);
          } else {
            ptr.push_back(static_cast<PtrT>(targetIndices.size()));
          }
        }
        continue;
      }

      // Lambda on active contact Jacobians of the pair. Async contact has only colliding Jacobians.
      auto const& collidingJacData =
          reg.get<CCollJacs<CollRole::Colliding> const>(e)[coll.collidingJacId];
      auto const forEachActiveJac = [&](auto const& fn) {
        collidingJacData.ForEachActiveJac(fn);
        if constexpr (kIsSync) {
          colliderJacData->ForEachActiveJac(fn);
        }
      };

      // Gather the number of contacts and whether all Jacobians share DoFs.
      int nContacts = 0;
      bool allSharedDoFs = true;
      forEachActiveJac([&](ContactJac const& jac) {
        MOCHI_ASSERT_VERBOSE(nContacts == 0 || jac.nContacts == nContacts);
        nContacts = jac.nContacts;
        allSharedDoFs = allSharedDoFs && jac.hasSharedDoFs;
      });

      // Add one row per group of consecutive contacts with identical DoFs, holding the DoFs of all
      // active slices. Traverse() below couples all the DoFs of a row.
      int const numRows = allSharedDoFs ? Min(nContacts, 1) : nContacts;
      for (int i = 0; i < numRows; ++i) {
        if (i > 0) {
          bool sameDofs = true;
          forEachActiveJac([&](ContactJac const& jac) {
            sameDofs = sameDofs && (jac.Inds(i) == jac.Inds(i - 1));
          });
          if (sameDofs) {
            continue;
          }
        }
        forEachActiveJac([&](ContactJac const& jac) {
          AppendBlockIndices<kBlockSize>(targetIndices, jac.Inds(i));
        });
        ptr.push_back(static_cast<PtrT>(targetIndices.size()));
      }
    }
  }

  auto cToN = Graph<int, int>{std::move(ptr), std::move(targetIndices)};
  return Traverse(Reverse(cToN), cToN).SortTargets();
}

// Template specialization used outside this file
template Graph<int, int> mochi::MakeContactGraph<1, /*kIsSync*/ true>(
    entt::registry const& reg,
    Span<entt::entity const> actors);

// Sync link-space contact residual of the articulation that owns the link residuals of 'actor' (the
// articulation of a link or a nested soft actor, or the actor itself), or null if none.
static CArticulatedSyncContactLinkResidual* TryGetSyncContactLinkResidual(
    entt::registry const& reg,
    ecs::PartialRegistry<CArticulatedSyncContactLinkResidual> regOut,
    entt::entity actor) {
  entt::entity articulation = entt::null;
  if (auto const* articulatedEntity = reg.try_get<CArticulatedEntity const>(actor)) {
    articulation = articulatedEntity->entity;
  } else if (auto const* composition = reg.try_get<CSkinnedComposition const>(actor)) {
    articulation = composition->articulated;
  } else if (reg.all_of<TagArticulatedActor>(actor)) {
    articulation = actor;
  }
  return articulation == entt::null
      ? nullptr
      : regOut.try_get<CArticulatedSyncContactLinkResidual>(articulation);
}

namespace {

struct ContactAssemblyPair {
  entt::entity collidingEntity = {};
  ActiveCollision* collision = nullptr;
  // The Jacobians of this pair in allJacs, contiguous because GetActiveJacs only appends. The
  // colliding Jacobians go first.
  Interval<int> jacRange;
  int numCollidingJacs = 0;
  CLinearContactSkinElementMap const* linearElementMap = nullptr;
  // Link-space residuals of the colliding and collider slices with link residuals, if any.
  ArticulatedLinkResidual* collidingLinkResidual = nullptr;
  ArticulatedLinkResidual* colliderLinkResidual = nullptr;
  // Set by the pair builders, since sync and async contact retain forces under different
  // conditions, and read again after the tasks.
  bool retainForces = false;
  double costPerPoint = 0;

  [[nodiscard]] bool UsesLinearElementMap() const {
    return linearElementMap != nullptr;
  }
};

} // namespace

ContactEvalConfig mochi::MakeContactEvalConfigAsync(
    ExperimentalEvalParams const& experimentalEval,
    ContactAssemblyReg reg,
    entt::entity colliding,
    bool psdDRes,
    bool useFittedHessian) {
  return ContactEvalConfig{
      .psdDRes = psdDRes,
      .addPadding = ShouldAddPenaltyPadding(reg.get<CColliderInfo const>(colliding).type),
      .validCollidingNormals = ValidCollidingNormals(reg, colliding),
      .explicitNormals = experimentalEval.explicitNormals,
      .fadeFriction = experimentalEval.fadeFriction,
      .implicitNormalForceForDissipation = experimentalEval.implicitNormalForceForDissipation,
      .useFittedHessian = useFittedHessian,
      .frictionModel = experimentalEval.frictionModel,
      .elastoPlasticFriction = experimentalEval.elastoPlasticFriction};
}

static ContactEvalConfig MakeContactEvalConfigPair(
    ExperimentalEvalParams const& experimentalEval,
    ContactAssemblyReg reg,
    entt::entity colliding,
    entt::entity collider,
    bool psdDRes,
    bool useFittedHessian) {
  ContactEvalConfig config =
      MakeContactEvalConfigAsync(experimentalEval, reg, colliding, psdDRes, useFittedHessian);
  config.elastoPlasticFriction = config.elastoPlasticFriction && HasFrictionAnchors(reg, collider);
  return config;
}

template <GradTarget kGradTarget>
static void AssembleContactPairRanges(
    entt::registry const& reg,
    bool isSync,
    Span<ContactAssemblyPair> allPairs,
    Span<ContactJac const*> allJacs,
    Allocator* filoAllocator,
    double* outObj,
    bool assemRes,
    ColumnVectorView<real> outRes, // Empty if no slice assembles into it
    AnyMatrixView<real> outDRes,
    bool psdDRes,
    bool useFittedHessian) {
  MOCHI_PROFILE_SCOPE();
  bool const assemObj = (outObj != nullptr);
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

  // Estimate the cost of each pair.
  double totalCost = 0;
  for (auto& pair : allPairs) {
    int const numContacts = isize(pair.collision->collisionResult.sampleIndices);
    pair.costPerPoint = 0;
    if (assemObj) {
      pair.costPerPoint += 1;
    }
    double const averageMappedDofs =
        pair.UsesLinearElementMap() ? pair.linearElementMap->AverageDofsPerElement() : 0.0;
    if (assemRes) { // linear cost
      pair.costPerPoint += averageMappedDofs;
      for (int i : pair.jacRange) {
        double const alpha = allJacs[i]->hasSharedDoFs ? 0 : 2.5;
        pair.costPerPoint += allJacs[i]->nDoFsInternal * (1 + alpha);
      }
    }
    if (assemDRes) { // quadratic cost
      pair.costPerPoint += averageMappedDofs * averageMappedDofs;
      for (int a : pair.jacRange) {
        double const alpha = allJacs[a]->hasSharedDoFs ? 0 : 5;
        int const nDofsInternalA = allJacs[a]->nDoFsInternal;
        int const nDofsStateA = allJacs[a]->nDoFsState;
        pair.costPerPoint += 2 * averageMappedDofs * nDofsStateA;
        pair.costPerPoint += static_cast<double>(nDofsInternalA) * nDofsInternalA * (1 + alpha);
        for (int b = pair.jacRange.Min(); b < a; ++b) {
          double const beta = allJacs[a]->hasSharedDoFs && allJacs[b]->hasSharedDoFs ? 1 : 0;
          int const nDofsInternalB = allJacs[b]->nDoFsInternal;
          int const nDofsStateB = allJacs[b]->nDoFsState;
          pair.costPerPoint += static_cast<double>(nDofsInternalA) * nDofsInternalB * beta;
          pair.costPerPoint += static_cast<double>(nDofsStateA) * nDofsStateB * (1 - beta + alpha);
        }
      }
    }
    // Pairs without contacts are never split below, so only their cost may be zero. The clamp
    // keeps the split, which divides by the cost per point, valid where assertions are disabled.
    if (numContacts > 0) {
      MOCHI_ASSERT_VERBOSE(pair.costPerPoint > 0, "Contact pair assembly cost must be positive.");
      pair.costPerPoint = Max(pair.costPerPoint, 1.0);
      totalCost += numContacts * pair.costPerPoint;
    }
  }
  if (totalCost == 0) {
    return;
  }

  // Determine the target number of tasks. The actual number of tasks may be smaller than the
  // target, depending on how the cost is distributed across contact pairs. If we use too many
  // tasks, it will be diminishing returns because of the high cost of duplicating the value
  // arrays and adding them back together.
  // TODO: Retune parallelization. It seems to over-parallelize.
  int constexpr kTargetCostPerTask = 10000;
  int constexpr kMaxTasks = 12;
  int const maxTasks = Min(kMaxTasks, TaskScheduler::StaticGetNumOtherThreads() + 1);
  int const numTargetTasks =
      static_cast<int>(Min<double>(Ceil(totalCost / kTargetCostPerTask), maxTasks));

  // Work to be done for a contact pair.
  struct PairWork {
    PairWork(ContactAssemblyPair* contactPair_, Interval<int> pointRange_)
        : contactPair(contactPair_), pointRange(pointRange_) {}
    ContactAssemblyPair* contactPair = nullptr;
    Interval<int> pointRange;
    int linkResOffset = 0; // Offset of the link residuals of the pair in linkResiduals
  };

  // Work to be done by each task.
  struct TaskWork {
    DynamicArray<PairWork> pairWork;
    double objective = 0.0;
    ColumnVector<real> resValues;
    ColumnVector<real> dresValues;
    double cost = 0;
  };

  // Split the work into tasks of roughly equal cost.
  DynamicArray<TaskWork> taskWork(filoAllocator);
  taskWork.reserve(kMaxTasks);
  double const targetCostPerTask = totalCost / numTargetTasks;
  for (auto& pair : allPairs) {
    Interval<int> pointsRemaining{0, isize(pair.collision->collisionResult.sampleIndices)};
    while (pointsRemaining.Size() > 0) {
      if (isize(taskWork) == kMaxTasks) {
        // Floating-point rounding can leave work after kMaxTasks. Do not grow taskWork:
        // DynamicArray::GrowCapacity allocates a new FILO block, then tries to free the previous
        // block. FiloAllocator only permits freeing the most recently allocated block.
        taskWork.back().pairWork.emplace_back(&pair, pointsRemaining);
        break;
      }
      if (taskWork.empty() || taskWork.back().cost >= targetCostPerTask) {
        taskWork.push_back({});
      }
      auto& currentTaskWork = taskWork.back();
      double const taskRemainingCost = targetCostPerTask - currentTaskWork.cost;
      int const numPointsToAdd = static_cast<int>(
          Min<double>(Ceil(taskRemainingCost / pair.costPerPoint), pointsRemaining.Size()));
      MOCHI_ASSERT(numPointsToAdd > 0 && numPointsToAdd <= pointsRemaining.Size());
      Interval<int> const subRange{pointsRemaining.Min(), pointsRemaining.Min() + numPointsToAdd};
      currentTaskWork.pairWork.emplace_back(&pair, subRange);
      currentTaskWork.cost += numPointsToAdd * pair.costPerPoint;
      pointsRemaining.Min() += numPointsToAdd;
    }
  }

  // Link residuals of all the work, with a contiguous range for each PairWork, with 6 values per
  // slice of its pair. A pair may be split between tasks, but each PairWork belongs to one task, so
  // the ranges are written concurrently.
  int linkResSize = 0;
  if (assemRes) {
    for (auto& task : taskWork) {
      for (auto& pairWork : task.pairWork) {
        pairWork.linkResOffset = linkResSize;
        linkResSize += RigidSize::kDAll * pairWork.contactPair->jacRange.size();
      }
    }
  }
  ColumnVector<real> linkResiduals(linkResSize, filoAllocator);

  // The point ranges of a pair are disjoint, so the tasks write the retained forces directly, and
  // together they cover every contact.
  if (assemRes) {
    for (auto const& pair : allPairs) {
      auto& query = pair.collision->collisionResult;
      if (pair.retainForces && !query.sampleIndices.empty()) {
        query.forcePerUnitArea.resize_noinit(query.sampleIndices.size());
      }
    }
  }

  auto doTask = [&](int iTask) {
    auto& myTask = taskWork[iTask];
    double* taskObj = outObj;
    ColumnVectorView<real> taskRes = outRes;
    AnyMatrixView<real> taskDRes = outDRes;

    if (iTask == 0) {
      // The first task assembles directly to the output residual and matrix.
    } else {
      MOCHI_PROFILE_SCOPE_N("Init Task Data");
      // Other threads allocate private values arrays for the residual and derivative.
      if (assemObj) {
        taskObj = &myTask.objective;
      }
      if (!outRes.empty()) {
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
    for (auto const& pairWork : myTask.pairWork) {
      maxContacts =
          Max(isize(pairWork.contactPair->collision->collisionResult.sampleIndices), maxContacts);
    }
    response.ResizeNoInit(maxContacts, assemObj, assemRes, assemDRes);

    auto const& simParams = reg.ctx<CSimulationParams const>();
    for (auto& pairWork : myTask.pairWork) {
      auto const& pair = *pairWork.contactPair;
      entt::entity const entity0 = pair.collidingEntity;
      entt::entity const entity1 = pair.collision->colliderEntity;
      bool const useContactSums =
          reg.get<CCollJacs<CollRole::Colliding> const>(entity0)[pair.collision->collidingJacId]
              .useContactSums;
      int const numContacts = isize(pair.collision->collisionResult.sampleIndices);
      real const dtStage = reg.get<CTimeIntegratorState const>(entity0).dtStage;
      auto const& samples = reg.get<CContactSamples<TimeStep::Current> const>(entity0);

      ContactEvalConfig const configPair = MakeContactEvalConfigPair(
          simParams.experimentalEval, reg, entity0, entity1, psdDRes, useFittedHessian);

      // Compute collision response and assemble it
      auto const& query = pair.collision->collisionResult;
      auto const contactParams = GetContactPairParams(reg, entity0, entity1);
      response.ResizeNoInit(numContacts, assemObj, assemRes, assemDRes);
      auto const jacs = (pair.jacRange.size() > 0)
          ? Span{&allJacs[pair.jacRange.Min()], pair.jacRange.size()}
          : Span<ContactJac const*>{};
      ColumnVectorView<real> const linkRes(
          linkResiduals.data() + pairWork.linkResOffset,
          assemRes ? RigidSize::kDAll * isize(jacs) : 0);
      bool const storeForces = assemRes && pair.retainForces;
      Span<Real3> const retainedForces =
          storeForces ? MakeSpan(pair.collision->collisionResult.forcePerUnitArea) : Span<Real3>{};
      if (useContactSums && reg.all_of<TagRigidActor>(entity0)) {
        // Between rigid actors, only sums of the per-point response are needed. Async contact never
        // reaches here; see the TagRigidActor assert in AssembleAsyncSkinnedContact.
        MOCHI_ASSERT_VERBOSE(isSync, "Rigid contact sums require sync contact.");
        Vec4r const comBLocal = (assemRes || assemDRes)
            ? reg.get<CRigidBodyInertia const>(entity1).GetCenterOfMassLocal()
            : Vec4r{};
        RigidContactSums const sums = ComputeRigidContactSums<kGradTarget>(
            pairWork.pointRange,
            query,
            contactParams,
            configPair,
            dtStage,
            samples.weights,
            comBLocal,
            assemObj,
            assemRes,
            assemDRes,
            storeForces ? MakeSpan(response.force) : Span<Real3>{});
        AssembleSyncRigidContactSums<kGradTarget>(
            reg,
            entity0,
            entity1,
            sums,
            jacs,
            &taskLocalAlloc,
            taskObj,
            taskRes,
            linkRes,
            taskDRes);
      } else if (useContactSums) {
        // A skinned actor against a rigid or static collider: sums over the bone slices and, for
        // sync contact, the rigid-collider slice about the collider's center of mass.
        MOCHI_ASSERT_VERBOSE(
            kGradTarget == GradTarget::Current,
            "Only contact between rigid actors supports differentiability.");
        std::optional<Vec4r> rigidColliderOrigin;
        if (isSync) {
          rigidColliderOrigin = (assemRes || assemDRes)
              ? reg.get<CRigidBodyInertia const>(entity1).GetCenterOfMassLocal()
              : Vec4r{};
        }
        DynamicArray<VMatrix3x3r> boneRotations(&taskLocalAlloc);
        SkinnedContactSlices const slices =
            GetSkinnedContactSlices(reg, entity0, query, rigidColliderOrigin, boneRotations);
        SkinnedContactSums const sums = ComputeSkinnedContactSums(
            pairWork.pointRange,
            query,
            contactParams,
            configPair,
            dtStage,
            samples.weights,
            slices,
            assemObj,
            assemRes,
            assemDRes,
            &taskLocalAlloc,
            storeForces ? MakeSpan(response.force) : Span<Real3>{});
        AssembleSkinnedContactSums(
            sums,
            jacs,
            slices.rigidColliderOrigin.has_value(),
            query.jacColliderFromWorld[0],
            &taskLocalAlloc,
            taskObj,
            taskRes,
            linkRes,
            taskDRes);
      } else {
        MOCHI_ASSERT_VERBOSE(
            kGradTarget == GradTarget::Current,
            "Only contact between rigid actors supports differentiability.");
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
        if (pair.UsesLinearElementMap()) {
          linear_contact_skin::AssembleContactResponseRange(
              query,
              *pair.linearElementMap,
              reg.get<CFemSurfaceDiscretization const>(entity0),
              reg.get<CRootTransform const>(entity0),
              samples.weights,
              reg.get<CDofOffset const>(entity0).dofsOffset,
              pairWork.pointRange,
              response,
              jacs,
              &taskLocalAlloc,
              {.outObj = taskObj,
               .outRes = taskRes,
               .outDRes = taskDRes,
               .params = {
                   .assemObj = assemObj,
                   .assemRes = assemRes,
                   .assemDRes = assemDRes,
                   .psdDRes = psdDRes}});
        }
        // Mapped pairs hold only collider Jacobians here, none for async, and their energy is
        // assembled above.
        if (!pair.UsesLinearElementMap() || !jacs.empty()) {
          AssembleCollisionResponseRange(
              pairWork.pointRange,
              query,
              response,
              samples.weights,
              jacs,
              pair.numCollidingJacs,
              &taskLocalAlloc,
              pair.UsesLinearElementMap() ? nullptr : taskObj,
              taskRes,
              linkRes,
              taskDRes);
        }
      }

      if (storeForces) {
        std::copy(
            response.force.begin() + pairWork.pointRange.Min(),
            response.force.begin() + pairWork.pointRange.Max() + 1,
            retainedForces.begin() + pairWork.pointRange.Min());
      }
    }
  };

  ParallelForN("AssemblePairs", isize(taskWork), 1, doTask);

  {
    MOCHI_PROFILE_SCOPE_N("Accumulate");
    for (int i = 1; i < isize(taskWork); ++i) {
      if (assemObj) {
        *outObj += taskWork[i].objective;
      }
      if (!outRes.empty()) {
        outRes += taskWork[i].resValues;
      }
      if (assemDRes) {
        AsView(GetValues(outDRes)) += taskWork[i].dresValues;
      }
    }
  }

  // Gather the link residuals into the link-space residuals of their articulations, in a fixed
  // order. Each is zeroed on its first gathered link residual, since it is inactive until then.
  if (assemRes) {
    for (auto const& task : taskWork) {
      for (auto const& pairWork : task.pairWork) {
        auto const& pair = *pairWork.contactPair;
        Span<ContactJac const*> const jacs{
            allJacs.data() + pair.jacRange.Min(), pair.jacRange.size()};
        if (std::none_of(
                jacs.begin(), jacs.end(), [](auto const* jac) { return HasLinkResidual(*jac); })) {
          continue;
        }
        int const numColliding = pair.numCollidingJacs;
        int const numCollider = isize(jacs) - numColliding;
        real* const linkRes = linkResiduals.data() + pairWork.linkResOffset;
        AddLinkResiduals(
            jacs.subspan(0, numColliding),
            ColumnVectorView<real>(linkRes, RigidSize::kDAll * numColliding),
            pair.collidingLinkResidual);
        AddLinkResiduals(
            jacs.subspan(numColliding),
            ColumnVectorView<real>(
                linkRes + RigidSize::kDAll * numColliding, RigidSize::kDAll * numCollider),
            pair.colliderLinkResidual);
      }
    }
  }
}

template <GradTarget kGradTarget>
static void AssembleAllSyncContactPairs(
    entt::registry const& reg,
    ecs::PartialRegistry<
        CActiveCollisions</*kIsSync*/ true, TimeStep::Current>,
        CArticulatedSyncContactLinkResidual> regOut,
    Span<entt::entity const> actors,
    Allocator* filoAllocator,
    double* outObj,
    bool assemRes,
    ColumnVectorView<real> outRes, // Empty if no slice assembles into it
    AnyMatrixView<real> outDRes,
    bool psdDRes,
    bool useFittedHessian) {
  MOCHI_PROFILE_SCOPE();
  bool const needsJacs = assemRes || GetNumValues(outDRes) > 0;

  // FILO-backed arrays must reserve their full capacities before they begin filling. The number of
  // contact Jacobians is an upper bound on the active ones.
  int numContactPairs = 0;
  int numJacs = 0;
  for (entt::entity const entity : actors) {
    if (auto const* activeCollisions =
            regOut.try_get<CActiveCollisions</*kIsSync*/ true, TimeStep::Current>>(entity)) {
      numContactPairs += isize(*activeCollisions);
      if (needsJacs) {
        auto const& collidingJacs = reg.get<CCollJacs<CollRole::Colliding> const>(entity);
        for (auto const& collision : *activeCollisions) {
          auto const& colliderJacs =
              reg.get<CCollJacs<CollRole::Collider> const>(collision.colliderEntity);
          numJacs += isize(collidingJacs[collision.collidingJacId].GetJacs());
          numJacs += isize(colliderJacs[collision.colliderJacId].GetJacs());
        }
      }
    }
  }
  if (numContactPairs == 0) {
    return;
  }

  DynamicArray<ContactJac const*> allJacs(filoAllocator);
  DynamicArray<ContactAssemblyPair> allPairs(filoAllocator);
  if (needsJacs) {
    allJacs.reserve(numJacs);
  }
  allPairs.reserve(numContactPairs);
  for (entt::entity const entity0 : actors) {
    if (auto* activeCollisions =
            regOut.try_get<CActiveCollisions</*kIsSync*/ true, TimeStep::Current>>(entity0)) {
      auto const* linearElementMap = reg.try_get<CLinearContactSkinElementMap const>(entity0);
      auto const& collidingJacs = reg.get<CCollJacs<CollRole::Colliding> const>(entity0);
      auto* const collidingLinkResidual =
          assemRes ? TryGetSyncContactLinkResidual(reg, regOut, entity0) : nullptr;
      bool const collidingRetainsForces = assemRes && reg.all_of<TagQueryActiveContacts>(entity0);
      for (auto& collision : *activeCollisions) {
        // Empty collisions get no work; see AssembleAsyncSkinnedContact.
        if (collision.collisionResult.Empty()) {
          continue;
        }
        int const firstJac = isize(allJacs);
        int numCollidingJacs = 0;
        entt::entity const entity1 = collision.colliderEntity;
        auto const& collidingJacData = collidingJacs[collision.collidingJacId];
        if (needsJacs) {
          // Mapped contact retains only collider-side Jacobians. Unmapped fallback pairs preserve
          // colliding before collider ordering.
          if (!linearElementMap) {
            collidingJacData.GetActiveJacs(allJacs);
          }
          numCollidingJacs = isize(allJacs) - firstJac;
          reg.get<CCollJacs<CollRole::Collider> const>(entity1)[collision.colliderJacId]
              .GetActiveJacs(allJacs);
        }
        allPairs.push_back({
            .collidingEntity = entity0,
            .collision = &collision,
            .jacRange = {firstJac, isize(allJacs)},
            .numCollidingJacs = numCollidingJacs,
            .linearElementMap = linearElementMap,
            .collidingLinkResidual = collidingLinkResidual,
            .colliderLinkResidual =
                assemRes ? TryGetSyncContactLinkResidual(reg, regOut, entity1) : nullptr,
            .retainForces =
                collidingRetainsForces || (assemRes && reg.all_of<TagQueryActiveContacts>(entity1)),
        });
      }
    }
  }

  AssembleContactPairRanges<kGradTarget>(
      reg,
      /*isSync*/ true,
      MakeSpan(allPairs),
      MakeSpan(allJacs),
      filoAllocator,
      outObj,
      assemRes,
      outRes,
      outDRes,
      psdDRes,
      useFittedHessian);
}

template <bool kIsSync>
static bool ActorsHaveContact(entt::registry const& reg, Span<entt::entity const> actors) {
  MOCHI_PROFILE_SCOPE();
  for (auto e : actors) {
    if (auto const* activeCollisions =
            reg.try_get<CActiveCollisions<kIsSync, TimeStep::Current> const>(e)) {
      for (auto const& col : *activeCollisions) {
        if (!col.collisionResult.Empty()) {
          return true;
        }
      }
    }
  }
  return false;
}

// Whether some contact Jacobian slice of 'actors' may assemble into the solver-space residual,
// i.e., some actor is neither an articulated link nor a non-blended articulated actor (whose slices
// all have link residuals). A blended actor may have soft slices.
static bool NeedResidual(entt::registry const& reg, Span<entt::entity const> actors) {
  return std::any_of(actors.begin(), actors.end(), [&](entt::entity e) {
    return !reg.any_of<TagArticulatedLinkActor, TagArticulatedActor>(e) ||
        reg.all_of<TagBlendedActor>(e);
  });
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
  if (!ActorsHaveContact</*kIsSync*/ true>(reg, actors)) {
    outContactSnle.useInSolver = false;
    return;
  }

  outContactSnle.dresiduals.resize(1);
  auto& outContactDResidual = outContactSnle.dresiduals[0].matrix;

  ColumnVectorView<real> outContactResidual; // Empty if not assembled or no actor assembles into it
  if (params.assemRes) {
    if (NeedResidual(reg, actors)) {
      // TODO: A sparse vector would be more efficient.
      outContactSnle.residuals.resize(1);
      outContactSnle.residuals[0].second.Resize(islandDofInfo.dofsSize);
      outContactResidual.Reset(outContactSnle.residuals[0].second);
    } else {
      outContactSnle.residuals.clear();
    }
  }

  if (params.assemDRes) {
    // Build a dresidual matrix based on the DOFs that are actually affected by contact
    MOCHI_PROFILE_SCOPE_N("Make Contact Matrix");
    if (useBlockSparse3x3) {
      outContactDResidual =
          BlockSparseMatrix<real, 3>{MakeContactGraph<3, /*kIsSync*/ true>(reg, actors)};
    } else {
      outContactDResidual = SparseMatrix<real>{MakeContactGraph<1, /*kIsSync*/ true>(reg, actors)};
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
  MOCHI_ASSERT(
      params.gradTarget == GradTarget::Current || !params.assemDRes,
      "The contact DResidual only supports GradTarget::Current.");
  auto assembleAllSyncContactPairsFn = params.gradTarget == GradTarget::Previous
      ? AssembleAllSyncContactPairs<GradTarget::Previous>
      : AssembleAllSyncContactPairs<GradTarget::Current>;
  assembleAllSyncContactPairsFn(
      reg,
      reg,
      actors,
      &filoAllocator,
      params.assemObj ? &outContactSnle.objective : nullptr,
      params.assemRes,
      outContactResidual,
      params.assemDRes ? AsView(outContactDResidual) : AnyMatrixView<real>{},
      params.psdDRes,
      params.fittedSaturationHessian.contactFriction);
}

// Returns the island-sized contact residual, empty if not assembled or if no slice may assemble
// into it, rather than into a link-space residual.
static ColumnVectorView<real> SetUpAsyncSkinnedContactSnle(
    AssemblyParams const& params,
    bool useBlockSparse3x3,
    entt::registry const& reg,
    entt::entity e,
    CIslandMemberInfo const& islandMember,
    CSkinnedContactSnle& outContactSnle) {
  outContactSnle.dresiduals.resize(1);
  auto& outContactDResidual = outContactSnle.dresiduals[0].matrix;

  ColumnVectorView<real> outContactResidual;
  if (params.assemRes) {
    if (NeedResidual(reg, MakeSingletonConstSpan(e))) {
      // TODO: Skinned contact should assemble directly into this entity's reduced CActorSnle, or
      // to another residual/dresidual of the same size. For now, it produces an
      // interactionResidual and interactionMatrix for the island's SNLE problem, so they need to
      // be the size of the island.
      auto const& islandDofInfo = reg.get<CIslandDofInfo>(islandMember.island);
      outContactSnle.residuals.resize(1);
      outContactSnle.residuals[0].second.Resize(islandDofInfo.dofsSize);
      outContactResidual.Reset(AsView(outContactSnle.residuals[0].second));
    } else {
      outContactSnle.residuals.clear();
    }
  }

  if (params.assemDRes) {
    // Compute sparsity based on the DOFs that are actually affected by explicit contact.
    if (useBlockSparse3x3) {
      outContactDResidual = BlockSparseMatrix<real, 3>{
          MakeContactGraph<3, /*kIsSync*/ false>(reg, MakeSingletonConstSpan(e))};
    } else {
      outContactDResidual = SparseMatrix<real>{
          MakeContactGraph<1, /*kIsSync*/ false>(reg, MakeSingletonConstSpan(e))};
    }
    MOCHI_ASSERT_VERBOSE(
        GetNumValues(outContactDResidual) > 0,
        "If there was no async contact with this actor, then we shouldn't have gotten this far.");
  }

  outContactSnle.SetZero(params);
  return outContactResidual;
}

void mochi::AssembleAsyncSkinnedContact(
    AssemblyParams const& params,
    bool useBlockSparse3x3,
    ecs::RequiredTag<TagSkinnedContact>,
    ecs::OptionalTag<TagQueryActiveContacts> queryActiveContacts,
    entt::registry const& reg,
    entt::entity e,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CActiveCollisions</*kIsSync*/ false, TimeStep::Current>& activeCollisions,
    CIslandMemberInfo const& islandMember,
    CTimeIntegratorState const& intState,
    CCollJacs<CollRole::Colliding> const& collJacs,
    CLinearContactSkinElementMap const* linearElementMap,
    CSkinnedContactSnle& outContactSnle,
    CArticulatedAsyncContactLinkResidual* outLinkResidual) {
  MOCHI_PROFILE_SCOPE();

  MOCHI_ASSERT_VERBOSE(params.gradTarget == GradTarget::Current, "Unexpected grad target");
  // TagSkinnedContact is only on articulated skins and on nested-soft, blended, and contact-skinned
  // deformable actors.
  MOCHI_ASSERT_VERBOSE(
      !reg.all_of<TagRigidActor>(e), "Rigid actors have no async skinned contact.");

  // Mapped contact reaches the DoFs through the linear element map rather than through colliding
  // Jacobians, so all of its collisions have solver DoFs.
  auto const hasSolverDoFs = [&](ActiveCollision const& collision) {
    return linearElementMap || collJacs[collision.collidingJacId].HasSolverDoFs();
  };

  // Skip if there are no contacts at all
  bool hasContact = false;
  bool hasContactWithDofs = false;
  bool hasContactWithoutDofs = false;
  for (auto const& coll : activeCollisions) {
    if (!coll.collisionResult.Empty()) {
      hasContact = true;
      bool const withDofs = hasSolverDoFs(coll);
      hasContactWithDofs |= withDofs;
      hasContactWithoutDofs |= !withDofs;
    }
  }
  outContactSnle.useInSolver = hasContactWithDofs;
  bool const needForces = params.assemRes && queryActiveContacts;
  if (!hasContact || (!hasContactWithDofs && !needForces)) {
    return;
  }

  // Set up the output SNLE if needed
  ColumnVectorView<real> outContactResidual; // Empty if not assembled or no slice assembles into it
  if (hasContactWithDofs) {
    outContactResidual.Reset(SetUpAsyncSkinnedContactSnle(
        params, useBlockSparse3x3, reg, e, islandMember, outContactSnle));
  }

  MOCHI_FILO_STACK_ALLOCATOR(filoAllocator, 64 * 1024);
  // Collisions without solver DoFs are evaluated only for contact-query forces.
  CollisionResponseResult forceOnlyResponse(&filoAllocator);
  ContactEvalConfig forceOnlyConfig;
  if (needForces && hasContactWithoutDofs) {
    forceOnlyResponse.Reserve(activeCollisions, false, true, false);
    forceOnlyConfig = MakeContactEvalConfigAsync(
        simParams->experimentalEval,
        reg,
        e,
        params.psdDRes,
        params.fittedSaturationHessian.contactFriction);
  }
  // FILO-backed arrays must reserve their full capacities before they begin filling. The number of
  // contact Jacobians is an upper bound on the active ones.
  bool const needsJacs = params.assemRes || params.assemDRes;
  DynamicArray<ContactJac const*> allJacs(&filoAllocator);
  DynamicArray<ContactAssemblyPair> allPairs(&filoAllocator);
  if (needsJacs) {
    int numJacs = 0;
    for (auto const& collision : activeCollisions) {
      numJacs += isize(collJacs[collision.collidingJacId].GetJacs());
    }
    allJacs.reserve(numJacs);
  }
  allPairs.reserve(activeCollisions.size());

  for (auto& collision : activeCollisions) {
    // An empty collision holds no forces to reset: CActiveCollisions::SetUp clears every result
    // each pass, and ContactDetectionResult::CheckSizes keeps forcePerUnitArea sized to the
    // contacts.
    if (collision.collisionResult.Empty()) {
      continue;
    }
    auto const& jacData = collJacs[collision.collidingJacId];
    if (!hasSolverDoFs(collision)) {
      if (needForces) {
        auto& query = collision.collisionResult;
        forceOnlyResponse.ResizeNoInit(isize(query.sampleIndices), false, true, false);
        ComputeCollisionResponse<GradTarget::Current>(
            query,
            GetContactPairParams(reg, e, collision.colliderEntity),
            forceOnlyConfig,
            intState.dtStage,
            false,
            true,
            false,
            forceOnlyResponse);
        query.forcePerUnitArea = forceOnlyResponse.force;
      }
      continue;
    }

    int const firstJac = isize(allJacs);
    if (needsJacs && !linearElementMap) {
      jacData.GetActiveJacs(allJacs);
    }
    allPairs.push_back({
        .collidingEntity = e,
        .collision = &collision,
        .jacRange = {firstJac, isize(allJacs)},
        .numCollidingJacs = isize(allJacs) - firstJac,
        .linearElementMap = linearElementMap,
        // Async pairs have no collider Jacobians, so colliderLinkResidual is never used.
        .collidingLinkResidual = outLinkResidual,
        .retainForces = static_cast<bool>(queryActiveContacts),
    });
  }

  if (!hasContactWithDofs) {
    return;
  }

  auto& outContactDResidual = outContactSnle.dresiduals[0].matrix;
  AssembleContactPairRanges<GradTarget::Current>(
      reg,
      /*isSync*/ false,
      MakeSpan(allPairs),
      MakeSpan(allJacs),
      &filoAllocator,
      params.assemObj ? &outContactSnle.objective : nullptr,
      params.assemRes,
      outContactResidual,
      params.assemDRes ? AsView(outContactDResidual) : AnyMatrixView<real>{},
      params.psdDRes,
      params.fittedSaturationHessian.contactFriction);
}

Aabb mochi::ExpandConservativeBoundsWithContactPadding(
    Aabb bounds,
    ecs::PartialRegistry<CContactParams const, CRequiresFarSdfEvaluation const> reg,
    entt::entity e) {
  auto const* contactParams = reg.try_get<CContactParams const>(e);
  real farSdfDistance = GetFarSdfEvaluationDistance<true>(reg, e);

  // Point-cloud collider radii are already included in the actors' local bounding volumes.
  // Bounds overlap if their distance is at most the sum of their paddings. Padding each actor by
  // the max of its collider and colliding paddings covers the threshold in both pair directions,
  // e.g., max(colliderA, collidingA) + max(colliderB, collidingB) >= colliderA + collidingB.
  // Without colliding padding, keep the collider padding even if it is negative.
  if (contactParams) {
    real const colliderPadding = GetColliderPadding(*contactParams);
    real const collidingPadding = contactParams->collidingPenaltyThresholdPadding;
    bounds = ExpandShape(
        bounds, collidingPadding > 0_r ? Max(colliderPadding, collidingPadding) : colliderPadding);
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
    auto const& bounds = reg.get<CBoundingVolume>(e);

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
  DynamicArray<std::pair<real, int>> dynamicSweepOrder; // (lower bound, index in dynamicBounds)
};
} // namespace

void mochi::AdvanceFrictionAnchorOffsets(
    ContactAssemblyReg reg,
    entt::entity e,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CConvergenceStatus const* convergence,
    CActiveCollisions</*kIsSync*/ false, TimeStep::Current>& asyncCollisions,
    CActiveCollisions</*kIsSync*/ true, TimeStep::Current>& syncCollisions) {
  if (convergence && convergence->stageStatus == ConvergenceStatus::Diverged) {
    return;
  }
  ContactEvalConfig const config = MakeContactEvalConfigAsync(
      simParams->experimentalEval, reg, e, /*psdDRes*/ false, /*useFittedHessian*/ false);
  std::array<Span<ActiveCollision>, 2> const allCollisions{
      MakeSpan(asyncCollisions), MakeSpan(syncCollisions)};
  for (auto const collisions : allCollisions) {
    for (auto& collision : collisions) {
      entt::entity const collider = collision.colliderEntity;
      if (!collision.collisionResult.Empty() && HasFrictionAnchors(reg, collider)) {
        AdvanceAnchorOffsets(
            collision.collisionResult, GetContactPairParams(reg, e, collider), config);
      }
    }
  }
}

namespace mochi::contact {
void EnableFrictionAnchorComponents(entt::registry& reg, bool enable) {
  // Every actor with contact samples has the components, or none does.
  if (enable == !reg.storage<CFrictionAnchors>().empty()) {
    return;
  }
  if (!enable) {
    reg.clear<CFrictionAnchors>();
    reg.clear<CFrictionAnchorIndex>();
    return;
  }
  for (auto const [e, samples] : reg.view<CContactSamples<TimeStep::Current> const>().each()) {
    EmplaceFrictionAnchors(reg, e, isize(samples.positions));
  }
}

void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CActiveCollisions</*kIsSync*/ false, TimeStep::Current>>(reg);
  ecs::RegisterComponent<CActiveCollisions</*kIsSync*/ false, TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CActiveCollisions</*kIsSync*/ true, TimeStep::Current>>(reg);
  ecs::RegisterComponent<CActiveCollisions</*kIsSync*/ true, TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CBoxCollider>(reg);
  ecs::RegisterComponent<CColliderInfo>(reg);
  ecs::RegisterComponent<CColliderWorldBounds>(reg);
  ecs::RegisterComponent<CCollJacs<CollRole::Collider>>(reg);
  ecs::RegisterComponent<CCollJacs<CollRole::Colliding>>(reg);
  ecs::RegisterComponent<CContactPairParamsOverrideTable>(reg);
  ecs::RegisterComponent<CContactPartitions>(reg);
  ecs::RegisterComponent<CContactSamples<TimeStep::Current>>(reg);
  ecs::RegisterComponent<CContactSamples<TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CFrictionAnchors>(reg);
  ecs::RegisterComponent<CFrictionAnchorIndex>(reg);
  ecs::RegisterComponent<CContactSkinningData>(reg);
  ecs::RegisterComponent<CLinearContactSkinElementMap>(reg);
  ecs::RegisterComponent<CDeformedContactSkinNodes>(reg);
  ecs::RegisterComponent<CContactCorrespondence</*kIsSync*/ false>>(reg);
  ecs::RegisterComponent<CContactCorrespondence</*kIsSync*/ true>>(reg);
  ecs::RegisterComponent<CMeshCollider>(reg);
  ecs::RegisterComponent<CPlaneCollider>(reg);
  ecs::RegisterComponent<CDeformablePointAsyncCollisionsResponse>(reg);
  ecs::RegisterComponent<CConservativePotentialColliders</*kIsSync*/ false>>(reg);
  ecs::RegisterComponent<CConservativePotentialColliders</*kIsSync*/ true>>(reg);
  ecs::RegisterComponent<CPotentialColliders</*kIsSync*/ false>>(reg);
  ecs::RegisterComponent<CPotentialColliders</*kIsSync*/ true>>(reg);
  ecs::RegisterComponent<CRequiresFarSdfEvaluation>(reg);
  ecs::RegisterComponent<CSdfCollider>(reg);
  ecs::RegisterComponent<CSdfColliderPending>(reg);
  ecs::RegisterComponent<CSdfMapping<TimeStep::Current>>(reg);
  ecs::RegisterComponent<CSdfMapping<TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CContactSampleBonePoints>(reg);
  ecs::RegisterComponent<CSphereCollider>(reg);

  // Global Context
  reg.set<CContactFilterTable>();
  reg.set<CContactPairParamsOverrideTable>();
  auto& data = reg.set<CTempPotentialColliderData>();
  int constexpr kReserveSize = 128; // Reserve memory for a modest number of actors up front
  data.staticColliders.reserve(kReserveSize);
  data.staticBounds.reserve(kReserveSize);
  data.dynamicEntities.reserve(kReserveSize);
  data.dynamicBounds.reserve(kReserveSize);
  data.dynamicHasCollider.reserve(kReserveSize);
  data.dynamicSweepOrder.reserve(kReserveSize);

  // Post-restore fixup: the friction anchor index is not captured.
  capture::RegisterPostRestoreSystem(&ReindexFrictionAnchors, reg);
}
} // namespace mochi::contact

size_t mochi::contact::SelectSweepAxis(Span<Aabb const> bounds) {
  // Accumulate offsets from the first finite center, in double, so that the variances don't cancel
  // far from the origin.
  Double3 origin{};
  Double3 sum{};
  Double3 sumSqr{};
  double count = 0.0;
  for (Aabb const& box : bounds) {
    // Non-finite boxes don't vote. Check them before taking the center: the center of infinite
    // (quasistatic) bounds computes -inf + inf, which raises an invalid-operation exception.
    if (!IsFinite(box.GetMin()) || !IsFinite(box.GetMax())) {
      continue;
    }
    auto const center = StaticCast<Double3>(box.GetCenter());
    if (count == 0.0) {
      origin = center;
    }
    Double3 const offset = center - origin;
    sum += offset;
    sumSqr += offset * offset;
    ++count;
  }
  // count^2 times the variances, all zero (selecting x) if no box is finite.
  return ArgMax(count * sumSqr - sum * sum);
}

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
  auto& dynamicSweepOrder = ctx.dynamicSweepOrder;
  staticColliders.clear();
  staticBounds.clear();
  dynamicEntities.clear();
  dynamicBounds.clear();
  dynamicHasCollider.clear();
  dynamicSweepOrder.clear();

  // Clear CConservativePotentialColliders for all actors
  reg.view<CConservativePotentialColliders</*kIsSync*/ false>>().each(
      [](auto& colliders) { colliders.clear(); });
  reg.view<CConservativePotentialColliders</*kIsSync*/ true>>().each(
      [](auto& colliders) { colliders.clear(); });

  auto& colliderWorldBounds = reg.storage<CColliderWorldBounds>();

  // Find all static colliders
  // TODO: These rarely change. We could cache this information and update it incrementally.
  for (auto&& [e, colliderInfo, root, bounds, contactParams] : reg.view<
                                                                      TagStaticActor,
                                                                      CColliderInfo const,
                                                                      CRootTransform const,
                                                                      CBoundingVolume const,
                                                                      CContactParams const>()
                                                                   .each()) {
    if (colliderInfo.type != ColliderType::None) {
      staticColliders.push_back(e);
      staticBounds.push_back(GetStaticColliderStepBounds(root, bounds, contactParams));
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
    // Islands set CColliderWorldBounds concurrently, so it is emplaced here. Unlike hasCollider,
    // this includes point-cloud colliders, which the point-cloud loop below adds to sync lists.
    if (cinfo && (cinfo->type != ColliderType::None) && !colliderWorldBounds.contains(e)) {
      colliderWorldBounds.emplace(e);
    }
  }

  // Find pairs of overlapping actors and update CConservativePotentialColliders
  auto const& layers = reg.storage<CContactLayer>();
  auto const& usesContact = reg.storage<TagUseContact>();
  auto& asyncColliders = reg.storage<CConservativePotentialColliders</*kIsSync*/ false>>();
  auto& syncColliders = reg.storage<CConservativePotentialColliders</*kIsSync*/ true>>();
  CConservativePotentialColliders</*kIsSync*/ false>* iAsync = nullptr;
  CConservativePotentialColliders</*kIsSync*/ true>* iSync = nullptr;
  int const numDynamic = isize(dynamicEntities);
  int const numStatic = isize(staticColliders);
  size_t const axis = SelectSweepAxis(dynamicBounds);

  // Sort the dynamic actors by lower bound on that axis, then by index so that the order does not
  // depend on the sort implementation. A NaN must not reach the comparator (undefined behavior),
  // and NaN bounds overlap nothing, so they sort last.
  for (int i = 0; i < numDynamic; ++i) {
    real const lower = dynamicBounds[i].GetMin()[axis];
    dynamicSweepOrder.emplace_back(std::isnan(lower) ? kInf : lower, i);
  }
  std::ranges::sort(dynamicSweepOrder);

  for (int iSweep = 0; iSweep < numDynamic; ++iSweep) {
    int const i = dynamicSweepOrder[iSweep].second;
    entt::entity ie = dynamicEntities[i];
    Aabb const iBounds = dynamicBounds[i];
    auto iLayer = layers.get(ie).id;
    bool iUseContact = usesContact.contains(ie);
    if (iUseContact) {
      iAsync = &asyncColliders.get(ie);
      iSync = &syncColliders.get(ie);

      // Find static actors that overlap entity ie.
      for (int j = 0; j < numStatic; ++j) {
        if (HasOverlap(iBounds, staticBounds[j])) {
          entt::entity je = staticColliders[j];
          ContactLayerId jLayer = layers.get(je).id;
          if (contactTable.IsContactEnabled(ie, je, iLayer, jLayer)) {
            MOCHI_ASSERT_VERBOSE(iAsync != nullptr);
            iAsync->emplace_back(je); // NOLINT(facebook-hte-NullableDereference)
          }
        }
      }
    }

    // Iterate over the later dynamic actors in the sweep that overlap entity ie on the sweep axis.
    // They may contact entity ie even if it doesn't contact them.
    real const iUpper = iBounds.GetMax()[axis];
    for (int jSweep = iSweep + 1; jSweep < numDynamic && dynamicSweepOrder[jSweep].first <= iUpper;
         ++jSweep) {
      int const j = dynamicSweepOrder[jSweep].second;
      if (HasOverlap(iBounds, dynamicBounds[j])) {
        entt::entity je = dynamicEntities[j];
        ContactLayerId jLayer = layers.get(je).id;

        // Consider i-vs-j
        if (iUseContact && dynamicHasCollider[j]) {
          if (contactTable.IsContactEnabled(ie, je, iLayer, jLayer)) {
            MOCHI_ASSERT_VERBOSE(iSync != nullptr);
            iSync->emplace_back(je); // NOLINT(facebook-hte-NullableDereference)
          }
        }

        // Consider j-vs-i
        if (usesContact.contains(je) && dynamicHasCollider[i]) {
          if (contactTable.IsContactEnabled(je, ie, jLayer, iLayer)) {
            auto& jSync = syncColliders.get(je);
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
            reg.get<CConservativePotentialColliders</*kIsSync*/ true>>(collidingEntity);
        // Point-cloud colliders with overlapping conservative step bounds are potential colliders
        // with each other (including themselves)
        reg.view<TagUsePointCloudContact const, CConservativeStepBounds const>().each(
            [&](entt::entity colliderEntity, CConservativeStepBounds const& colliderStepBounds) {
              // Skip self-collisions if indicated by the point-cloud collider parameters.
              bool const isSelfContact = (colliderEntity == collidingEntity);
              if (isSelfContact && !collidingPointCloudColliderParams.selfContact) {
                return;
              }
              if (!HasOverlap(collidingStepBounds.worldAabb, colliderStepBounds.worldAabb)) {
                return;
              }
              auto const& colliderLayer = reg.get<CContactLayer>(colliderEntity);
              if (contactTable.IsContactEnabled(
                      collidingEntity, colliderEntity, collidingLayer.id, colliderLayer.id)) {
                potentialColliders.emplace_back(colliderEntity);
              }
            });
      });

  FlagPotentialSyncColliders(reg);
}

void mochi::contact::FlagPotentialSyncColliders(entt::registry& reg) {
  auto& colliderWorldBounds = reg.storage<CColliderWorldBounds>();
  reg.view<CColliderWorldBounds>().each(
      [](auto& bounds) { bounds.isPotentialSyncCollider = false; });
  reg.view<CConservativePotentialColliders</*kIsSync*/ true> const>().each(
      [&](auto const& colliders) {
        for (entt::entity const collider : colliders) {
          MOCHI_ASSERT_VERBOSE(
              colliderWorldBounds.contains(collider),
              "Sync colliders should have CColliderWorldBounds, which the broadphase emplaces.");
          colliderWorldBounds.get(collider).isPotentialSyncCollider = true;
        }
      });
#if MOCHI_ASSERT_VERBOSE_ENABLED
  DynamicArray<entt::entity> potentialSyncColliders;
  reg.view<CConservativePotentialColliders</*kIsSync*/ true> const>().each(
      [&](auto const& colliders) { potentialSyncColliders.append(colliders); });
  std::ranges::sort(potentialSyncColliders);
  reg.view<CColliderWorldBounds const>().each([&](entt::entity e, auto const& bounds) {
    MOCHI_ASSERT_VERBOSE(
        bounds.isPotentialSyncCollider == std::ranges::binary_search(potentialSyncColliders, e),
        "isPotentialSyncCollider should flag exactly the colliders that sync lists hold.");
  });
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
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
  std::array<Span<entt::entity const>, 2> linearContactSkinActors = {
      descendants.softActors, descendants.shellActors};
  for (auto actors : linearContactSkinActors) {
    ecs::ScheduleInvokeForEach(
        sem,
        "linear_contact_skin::UpdatePositionsAndSamples<TimeStep::StageStart>",
        linear_contact_skin::UpdatePositionsAndSamples<TimeStep::StageStart>,
        reg,
        actors);
  }
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

// Computes the contact bounds of the island's potential sync colliders, for this collision
// detection call. Sync lists only hold members of their actor's island, so other islands never
// touch them.
template <TimeStep kTimeStep>
static void UpdateIslandColliderWorldBounds(entt::registry& reg, Span<entt::entity const> actors) {
  auto& colliderWorldBounds = reg.storage<CColliderWorldBounds>();
  auto const isPotentialSyncCollider = [&](entt::entity e) {
    auto const* worldBounds = TryGet(colliderWorldBounds, e);
    return worldBounds && worldBounds->isPotentialSyncCollider;
  };
  if (std::ranges::none_of(actors, isPotentialSyncCollider)) {
    return;
  }
  auto const& roots = reg.storage<CRootTransform>();
  auto const& boundingVolumes = reg.storage<CBoundingVolume>();
  auto const& contactParams = reg.storage<CContactParams>();
  for (entt::entity const e : actors) {
    if (isPotentialSyncCollider(e)) {
      colliderWorldBounds.get(e).shape = ComputeColliderWorldBounds(
          GetRootTransform<kTimeStep>(reg, e, roots.get(e)),
          boundingVolumes.get(e),
          contactParams.get(e));
    }
  }
}

template <TimeStep kTimeStep>
void mochi::CollisionDetectionPipeline(entt::registry& reg, CIslandDescendants const& descendants) {
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
    std::array<Span<entt::entity const>, 2> linearContactSkinActors = {
        descendants.softActors, descendants.shellActors};
    for (auto actors : linearContactSkinActors) {
      ecs::ScheduleInvokeForEach(
          sem,
          "linear_contact_skin::UpdateBounds",
          &linear_contact_skin::UpdateBounds<kTimeStep>,
          reg,
          actors);
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
    // ECS component matching ensures rod actors with TagUseDeformableContactSkin match
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
  std::array<Span<entt::entity const>, 2> shellAndRodActors = {
      descendants.shellActors, descendants.rodActors};
  for (auto actors : shellAndRodActors) {
    ecs::ScheduleInvokeForEach(
        sem, "UpdateSpatialHashTable", &UpdateSpatialHashTable<kTimeStep>, reg, actors);
  }

  // The above must complete before collision detection can start.
  sem.Wait();

  UpdateIslandColliderWorldBounds<kTimeStep>(reg, descendants.actors);

  ////////////////////////////////////////////////////////////////////////////////
  // Per Actor Collision Detection & Response
  ////////////////////////////////////////////////////////////////////////////////
  [[maybe_unused]] bool const explicitNormals =
      reg.ctx<CSimulationParams const>().experimentalEval.explicitNormals;
  [[maybe_unused]] bool const elastoPlasticFriction =
      reg.ctx<CSimulationParams const>().experimentalEval.elastoPlasticFriction;

  auto asyncCollisionAndResponse = [&](entt::entity e) {
    MOCHI_PROFILE_SCOPE_N("asyncCollisionAndResponse");
    CollisionDetection</*kIsSync*/ false, kTimeStep>(reg, e);
    if constexpr (kTimeStep == TimeStep::Current) {
      if (explicitNormals) {
        ecs::TryInvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
            AddStageStartCollisionDetection</*kIsSync*/ false>, reg, e);
      }
      SetupActiveCollisionNormals</*kIsSync*/ false>(reg, e);
      if (elastoPlasticFriction) {
        ecs::TryInvokeOnEntity(&FillFrictionAnchorOffsets</*kIsSync*/ false>, reg, e);
      }
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
    CollisionDetection</*kIsSync*/ true, kTimeStep>(reg, e);
    if constexpr (kTimeStep == TimeStep::Current) {
      if (explicitNormals) {
        ecs::TryInvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
            AddStageStartCollisionDetection</*kIsSync*/ true>, reg, e);
      }
      SetupActiveCollisionNormals</*kIsSync*/ true>(reg, e);
      if (elastoPlasticFriction) {
        ecs::TryInvokeOnEntity(&FillFrictionAnchorOffsets</*kIsSync*/ true>, reg, e);
      }
    }
  };

  // Loop over colliding actors and find collisions with colliders.
  auto const& usesContact = reg.storage<TagUseContact>();
  auto const& rigidActors = reg.storage<TagRigidActor>();
  for (auto e : descendants.actors) {
    if (usesContact.contains(e)) {
      // Rigid actors' contact samples are fixed in their local frame, so they have none to update.
      if (rigidActors.contains(e)) {
        Schedule(sem, "AsyncCollisionAndResponse", [&, e]() { asyncCollisionAndResponse(e); });
        Schedule(sem, "SyncCollisionAndResponse", [&, e]() { syncCollisionAndResponse(e); });
      } else if (ecs::CanInvokeOnEntity(
                     UpdateCollisionSamplePositions<
                         CFemBoundaryDiscretization,
                         kTimeStep,
                         kSpaceDim3>,
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
      } else if (ecs::CanInvokeOnEntity(linear_contact_skin::UpdateSamples<kTimeStep>, reg, e)) {
        Schedule(sem, "linear_contact_skin::UpdateSamples", [&, e, sem]() {
          ecs::InvokeOnEntity(linear_contact_skin::UpdateSamples<kTimeStep>, reg, e);
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
        MOCHI_ASSERT(false, "Contact actors must be rigid or update their contact samples.");
      }
    }
  }

  sem.Wait();
}

void mochi::ContactJacobiansPipeline(
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
template void mochi::CollisionDetectionPipeline<TimeStep::Current>(
    entt::registry& reg,
    CIslandDescendants const& descendants);

template void mochi::CollisionDetectionPipeline<TimeStep::StageStart>(
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
