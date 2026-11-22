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

#include "mochi_blended.h"

#include "mochi_common_components.h"
#include "mochi_compound.h"
#include "mochi_contact_filter.h"
#include "mochi_island.h"

#include <mochi_core/rom/rom_hyper_reduction.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/dskinning.h>
#include <mochi_core/utils/sparsity_utils.h>
#include <mochi_core/utils/task_scheduler.h>

#include <algorithm>
#include <array>
#include <iterator>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace mochi;
using namespace mochi::skinned;

// Possibly augment a soft actor's active boundary faces to support the nodes needed by a blended
// actor. The implementation works on first-come-first-serve-basis, i.e., it picks the first
// boundary face that has the required node.
static void AugmentSoftActorBoundaryDiscretization(
    entt::registry& reg,
    entt::entity soft,
    Span<int const> necessaryNodesSpan) {
  // If the soft actor has no hyper-reduction, we're done.
  auto* activeBdFacesSoft = reg.try_get<CActiveBoundaryFaces>(soft);
  if (!activeBdFacesSoft) {
    return;
  }

  // Prune the nodes that are already active on the soft actor
  std::vector<int> necessaryNodes(necessaryNodesSpan.begin(), necessaryNodesSpan.end());
  {
    auto const softActiveNodes = activeBdFacesSoft->ViewUniqueVolumeNodes();
    std::unordered_set<int> const set(softActiveNodes.begin(), softActiveNodes.end());
    auto& inds = necessaryNodes;
    auto end = std::remove_if(
        inds.begin(), inds.end(), [&set](int n) { return set.find(n) != set.end(); });
    inds.erase(end, inds.end());
  }

  auto const& femBoundaryDisc = reg.get<CFemBoundaryDiscretization const>(soft);
  femBoundaryDisc.Visit([&](auto const& femBoundaryDiscImpl) {
    auto const& tracesAll = femBoundaryDiscImpl.femElements;

    // Based on the necessary nodes, identify necessary traces.
    std::vector<int> newTracesSoft;
    newTracesSoft.reserve(necessaryNodes.size());
    std::unordered_set<int> set(necessaryNodes.begin(), necessaryNodes.end());
    for (int i = 0; i < tracesAll.size(); ++i) {
      for (auto node : tracesAll[i].Nodes()) {
        if (set.find(node) != set.end()) {
          set.erase(node);
          newTracesSoft.emplace_back(i);
        }
      }
    }

    // Merge the new and existing traces. Remove duplicates
    auto const inds = activeBdFacesSoft->ViewIndices();
    newTracesSoft.reserve(newTracesSoft.size() + inds.size());
    std::copy(inds.begin(), inds.end(), std::back_inserter(newTracesSoft));
    SortAndRemoveDuplicates(newTracesSoft);

    // Update active elements and unique nodes of the soft actor.
    activeBdFacesSoft->Recompute(newTracesSoft, MakeConstSpan(tracesAll));
    auto const& activeVolElements = reg.get<CActiveVolumeElements const>(soft);
    reg.get<CActiveUniqueNodes>(soft).Recompute(activeVolElements, *activeBdFacesSoft);
  });
}

static void InitBlendedMesh(
    entt::registry& reg,
    entt::entity e,
    ArticulatedSkinParams const& params,
    std::shared_ptr<Shape const> shape,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_PROFILE_SCOPE();

  // Create the surface discretization for contact
  int numCollidingSamples = InitDiscretizationSkinMesh(
      reg, e, *shape, params.boundarySubsampling, params.boundaryElementType, error);
  MOCHI_ERROR_RETURN(error);
  reg.emplace<CNodePositions>(e, reg.get<CArticulatedSkinningData const>(e).restCoords.Rows());

  // Emplace the blending data component.
  auto const& softActors = reg.get<CBlendedComposition const>(e).soft;
  auto& blending = reg.emplace<CBlendingData>(e);
  blending.reserve(softActors.size());

  // Add the blending data for each soft actor.
  auto const& blendingData = shape->GetMeshBlending();
  MOCHI_ERROR_IF_NOT(blendingData, error, "Blended skin shape carries no blending data.");
  MOCHI_ERROR_RETURN(error);
  for (auto soft : softActors) {
    auto const& sourceShapeName = reg.get<CActorInfo>(soft).name;
    // Soft actor name may be a slash-separated path. Get just the last part.
    auto const lastSlash = sourceShapeName.find_last_of('/');
    auto const nameHasSlash = (lastSlash != std::string::npos);
    auto const& nameToFind = nameHasSlash ? DynamicString{sourceShapeName.substr(lastSlash + 1)}
                                          : DynamicString{sourceShapeName};

    auto blendingDataTarget = blendingData->perSourceShapeData.find(nameToFind);
    MOCHI_ERROR_IF(
        blendingDataTarget == blendingData->perSourceShapeData.end(),
        error,
        "No blending data for this soft actor");
    MOCHI_ERROR_RETURN(error);
    int const numNodes = reg.get<CSimplicialMesh const>(soft).mesh->GetNumNodes();
    blending.emplace_back(blendingDataTarget->second.GetSourceBlendingData(numNodes));
  }

  // If needed, emplace the per-soft-actor active node component
  CBlendedActiveNodes* activeNodesBlended = nullptr;
  if (auto const* activeNodes = reg.try_get<CActiveUniqueNodes>(e)) {
    activeNodesBlended = &reg.emplace<CBlendedActiveNodes>(e);
    activeNodesBlended->reserve(softActors.size());

    // For each soft actor, find the intersection of active and blended nodes
    for (auto const& blendingSoft : blending) {
      auto const in = activeNodes->ViewIds();
      auto const& map = blendingSoft.mappingTargetToSource;
      std::vector<int> out(in.size());
      std::transform(in.begin(), in.end(), out.begin(), [&map](int i) { return map[i]; });
      out.erase(std::remove_if(out.begin(), out.end(), [](int i) { return i == -1; }), out.end());
      activeNodesBlended->emplace_back(std::move(out));
    }
  }

  // Possibly augment the soft-actor boundary discretizations with new active traces
  for (int i = 0; i < softActors.size(); i++) {
    // Identify the nodes needed by blending
    auto necessaryNodes = (activeNodesBlended != nullptr) ? MakeConstSpan((*activeNodesBlended)[i])
                                                          : MakeConstSpan(blending[i].nodesSource);
    AugmentSoftActorBoundaryDiscretization(reg, softActors[i], necessaryNodes);
  }

  // Initialize components for colliding-actor role.
  std::array<ContactPartitionStrategy, 2> strategies = {
      ContactPartitionStrategy::SharedBones, ContactPartitionStrategy::SoftActorId};
  InitCollidingSkinMesh(reg, e, e, *shape, strategies, params.contact, numCollidingSamples);
}

/*
 * System to update the blended displacements after skinning. Given rest position x0, elastic
 * displacement u, skinning transform T, and blending weight w, world position is:
 * x = w * T(x0 + u) + (1 - w) * T(x0)
 * The blending code below uses this expression, as articulated skinning and soft skinning are
 * already applied. Contact Jacobians, on the other hand, build on the equivalent expression:
 * x = T(x0 + w * u)
 */
template <TimeStep kStep, bool kForceUseAllNodes = false>
static void ResolveBlending(
    ecs::PartialRegistry<
        CArticulatedSkinningData const,
        CNodePositions const,
        CDisplacementSlice<real, kStep, DisplacementLayer::Skinned> const> reg,
    CBlendedComposition const& composition,
    CBlendingData const& blendingData,
    CArticulatedSkinningData const& skinningData,
    CActiveUniqueNodes const* activeNodes,
    CBlendedActiveNodes const* activeNodesBlended,
    CRootTransform const& rootTransform,
    CNodePositions& outPositions,
    CDisplacementSlice<real, kStep, DisplacementLayer::Skinned>& outDisplacements) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(
      (activeNodes == nullptr) == (activeNodesBlended == nullptr), "Inconsistent active nodes");

  bool const useActiveNodes = activeNodes && !kForceUseAllNodes;

  // Note that displacements are already initialized with the articulated displacements.
  // Articulation displacements are in local frame; soft displacements are in world frame.
  // Positions are pre-skinned, and must be initialized.
  auto disp3 = Unflatten<Real3>(MakeSpan(outDisplacements.value));
  auto const rest3 = Unflatten<Real3 const>(MakeConstSpan(skinningData.restCoords));
  auto const rootFromWorld = Invert(rootTransform.worldFromLocal);
  auto pos3 = Unflatten<Real3>(MakeSpan(outPositions.value));

  // Initialize pre-skinned positions, needed for contact Jacobians
  if (useActiveNodes) {
    for (int node : activeNodes->ViewIds()) {
      pos3[node] = rest3[node];
    }
  } else {
    outPositions.value = skinningData.restCoords;
  }

  // Iterate over nested soft actors and blend displacements and positions
  for (int s = 0; s < composition.soft.size(); s++) {
    entt::entity const soft = composition.soft[s];
    auto const& blending = blendingData[s];
    auto const& softDisp =
        reg.template get<CDisplacementSlice<real, kStep, DisplacementLayer::Skinned> const>(soft);
    auto const softDisp3 = Unflatten<Real3 const>(MakeConstSpan(softDisp.value));
    auto const& softSkinningData = reg.template get<CArticulatedSkinningData const>(soft);
    auto const softRest3 = Unflatten<Real3 const>(MakeConstSpan(softSkinningData.restCoords));
    auto const softPos3 =
        Unflatten<Real3 const>(MakeConstSpan(reg.template get<CNodePositions const>(soft).value));

    // Define the nodes to blend
    auto const nodes = useActiveNodes ? MakeConstSpan((*activeNodesBlended)[s])
                                      : MakeConstSpan(blending.nodesSource);
    for (int src : nodes) {
      // Convert soft position to local displacement. Blend displacement and pre-skinned position.
      int const dst = blending.mappingSourceToTarget[src];
      Real3 const softLocalDisp =
          rootFromWorld.TransformPoint(softRest3[src] + softDisp3[src]) - rest3[dst];
      real const weight = blending.weightsSource[src];
      disp3[dst] += weight * (softLocalDisp - disp3[dst]);
      pos3[dst] += weight * (softPos3[src] - pos3[dst]);
    }
  }
}

void blended::ResolveAllNodeBlendingDisplacementsPipeline(
    entt::registry& reg,
    Span<entt::entity const> entities) {
  MOCHI_PROFILE_SCOPE();
  ecs::InvokeForEach<ecs::policy::AllowReadWriteSameComponent>(
      &ResolveBlending<TimeStep::Current, /* kForceUseAllNodes */ true>, reg, entities);
}

void blended::UpdateBlendingVelocity(
    ecs::PartialRegistry<CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned> const>
        reg,
    CBlendedComposition const& composition,
    CBlendingData const& blendingData,
    CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned>& inOutVelocity) {
  // Velocity is initialized with world-space skinning velocity. Blend the soft actor contribution.
  auto blendedVel3 = Unflatten<Real3>(inOutVelocity.value.GetSpan());
  for (int s = 0; s < isize(composition.soft); ++s) {
    auto const soft = composition.soft[s];
    auto const softVel3 = Unflatten<Real3 const>(
        reg.get<CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned> const>(soft)
            .value.GetConstSpan());
    auto const& blending = blendingData[s];
    for (int src : blending.nodesSource) {
      int const dst = blending.mappingSourceToTarget[src];
      blendedVel3[dst] += blending.weightsSource[src] * (softVel3[src] - blendedVel3[dst]);
    }
  }
}

void blended::AttachNestedSoftActors(
    entt::registry& reg,
    entt::entity e,
    Span<ActorHandle const> softHandles) {
  MOCHI_ASSERT(reg.all_of<TagArticulatedActor const>(e), "Not an articulated actor.");

  auto const articulatedHandle = GetActorHandle(e, GetSceneHandle(reg));
  auto& composition = reg.emplace<CBlendedComposition>(e);
  composition.soft.reserve(softHandles.size());
  composition.softHandles.reserve(softHandles.size());
  for (auto handle : softHandles) {
    auto entity = GetEntity(reg, handle, ErrorAssert{});
    auto& skinnedComposition = reg.emplace<CSkinnedComposition>(entity);
    skinnedComposition.articulated = e;
    skinnedComposition.articulatedHandle = articulatedHandle;
    reg.emplace<TagNestedSoftActor>(entity);

    composition.soft.emplace_back(entity);
    composition.softHandles.emplace_back(handle);
  }
}

void blended::InitBlendedActor(
    entt::registry& reg,
    entt::entity e,
    ArticulatedActorParams const& params,
    std::shared_ptr<Shape const> blendedShapePtr,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_PROFILE_SCOPE();

  // Add blended actor tag
  reg.emplace<TagBlendedActor>(e);

  // Ensure this actor is automatically added to a compound
  reg.emplace<TagEnsureEntityInCompound>(e);

  if (!blendedShapePtr) {
    return; // Nothing else to do if there's no blended surface
  }

  // A blended surface is built from the skeleton's skin shape, so skin params are present.
  MOCHI_ASSERT_VERBOSE(params.skin.has_value(), "A blended surface requires skin params.");
  InitBlendedMesh(reg, e, *params.skin, blendedShapePtr, error);
  MOCHI_ERROR_RETURN(error);

  // Resolve blending so the blended skin displacements reflect the resolved articulated skin and
  // soft-skinned displacements.
  ResolveAllNodeBlendingDisplacementsPipeline(reg, MakeSingletonConstSpan(e));
}

void blended::PreStagePipeline(entt::registry& reg, Span<entt::entity const> entities) {
  if (entities.empty()) {
    return;
  }
  MOCHI_PROFILE_SCOPE();
  // Compute displacement at the start of the stage.
  ecs::InvokeForEach<ecs::policy::AllowReadWriteSameComponent>(
      &ResolveBlending<TimeStep::StageStart>, reg, entities);
}

void blended::PostLastStagePipeline(entt::registry& reg, Span<entt::entity const> entities) {
  if (entities.empty()) {
    return;
  }
  MOCHI_PROFILE_SCOPE();
  ResolveAllNodeBlendingDisplacementsPipeline(reg, entities);
}

void blended::UpdateDerivedStatePipeline(entt::registry& reg, Span<entt::entity const> entities) {
  if (entities.empty()) {
    return;
  }
  MOCHI_PROFILE_SCOPE();
  ecs::InvokeForEach<ecs::policy::AllowReadWriteSameComponent>(
      &ResolveBlending<TimeStep::Current>, reg, entities);
}

namespace mochi::blended {
void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CBlendingData>(reg);
  ecs::RegisterComponent<CBlendedActiveNodes>(reg);
  ecs::RegisterComponent<CBlendedComposition>(reg);
}
} // namespace mochi::blended
