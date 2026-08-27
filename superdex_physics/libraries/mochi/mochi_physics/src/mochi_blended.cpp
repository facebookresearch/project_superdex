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
#include "mochi_rom_jacobian.h"

#include <mochi_core/contact/dmap.h>
#include <mochi_core/rom/rom_hyper_reduction.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/container_utils.h>
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
#include <variant>
#include <vector>

using namespace mochi;
using namespace mochi::skinned;
using namespace mochi::dmap;

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
    blending.emplace_back(blendingDataTarget->second.GetSourceBlendingData<1>(numNodes));
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
      ContactPartitionStrategy::SkinningDofGroups, ContactPartitionStrategy::SoftActorId};
  InitCollidingSkinMesh(reg, e, e, *shape, strategies, params.contact, numCollidingSamples);
}

/*
 * System to update the blended positions after skinning.
 */
template <TimeStep kStep, bool kForceUseAllNodes = false>
static void ResolveBlending(
    ecs::PartialRegistry<CDisplacementSlice<real, kStep, DisplacementLayer::Skinned> const> reg,
    CBlendedComposition const& composition,
    CBlendingData const& blendingData,
    CActiveUniqueNodes const* activeNodes,
    CBlendedActiveNodes const* activeNodesBlended,
    CDisplacementSlice<real, kStep, DisplacementLayer::Skinned>& outDisplacements) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(
      (activeNodes == nullptr) == (activeNodesBlended == nullptr), "Inconsistent active nodes");

  // Note that displacements are already initialized with the articulated displacements
  auto disp3 = Unflatten<Real3>(MakeSpan(outDisplacements.value));

  // Iterate over soft skinned actors and blend displacements
  for (int s = 0; s < composition.soft.size(); s++) {
    auto const& blending = blendingData[s];
    auto const& softDisp =
        reg.template get<CDisplacementSlice<real, kStep, DisplacementLayer::Skinned> const>(
            composition.soft[s]);
    auto softDisp3 = Unflatten<Real3 const>(MakeConstSpan(softDisp.value));

    // Define the nodes to blend
    auto const nodes = (activeNodes && !kForceUseAllNodes) ? MakeConstSpan((*activeNodesBlended)[s])
                                                           : MakeConstSpan(blending.nodesSource);
    for (int src : nodes) {
      // Blend the displacements of the soft actor and the articulated actor.
      int const dst = blending.mappingSourceToTarget[src];
      disp3[dst] += blending.weightsSource[src] * (softDisp3[src] - disp3[dst]);
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

void blended::InitBlendedActor(
    entt::registry& reg,
    entt::entity e,
    ArticulatedActorParams const& params,
    Span<ActorHandle const> softHandles,
    std::shared_ptr<Shape const> blendedShapePtr,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_PROFILE_SCOPE();

  MOCHI_ASSERT(reg.all_of<TagArticulatedActor const>(e), "Not an articulated actor.");

  // Add composition component.
  auto& composition = reg.emplace<CBlendedComposition>(e);
  composition.soft.reserve(softHandles.size());
  composition.softHandles.reserve(softHandles.size());
  for (auto handle : softHandles) {
    auto entity = GetEntity(reg, handle, ErrorAssert{});
    MOCHI_ASSERT(reg.all_of<TagSoftSkinnedActor const>(entity), "Not a soft skinned actor.");
    composition.soft.emplace_back(entity);
    composition.softHandles.emplace_back(handle);
  }

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

/*
 * System to update the Jacobian of a blended mesh wrt joints.
 * The Jacobian wrt bones is not updated, as it is not used.
 */
static void ResolveJacobianDJoints(
    ecs::PartialRegistry<CArticulatedSkinningData const> reg,
    CBlendedComposition const& composition,
    CBlendingData const& blendingData,
    CActiveUniqueNodes const* activeNodes,
    CBlendedActiveNodes const* activeNodesBlended,
    CArticulatedSkinningData& outSkinningData) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(
      (activeNodes == nullptr) == (activeNodesBlended == nullptr), "Inconsistent active nodes");

  // Note that the Jacobian is already initialized with the articulated Jacobian
  auto result = AsView(outSkinningData.jacobianDJoints);

  // Iterate over soft skinned actors and blend Jacobians
  for (int s = 0; s < composition.soft.size(); s++) {
    auto const& blending = blendingData[s];
    auto const& softJac =
        reg.get<CArticulatedSkinningData const>(composition.soft[s]).jacobianDJoints;

    // Define the nodes to blend
    auto const nodes =
        activeNodes ? MakeConstSpan((*activeNodesBlended)[s]) : MakeConstSpan(blending.nodesSource);
    for (int src : nodes) {
      // Blend the Jacobians of the soft actor and the articulated actor.
      int const dst = blending.mappingSourceToTarget[src];
      result.MiddleRows<3>(3 * dst, 3) += blending.weightsSource[src] *
          (softJac.MiddleRows<3>(3 * src, 3) - result.MiddleRows<3>(3 * dst, 3));
    }
  }
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

void blended::UpdateJacobiansPipeline(entt::registry& reg, Span<entt::entity const> entities) {
  if (entities.empty()) {
    return;
  }
  MOCHI_PROFILE_SCOPE();
  ecs::ParallelInvokeForEach<ecs::policy::AllowReadWriteSameComponent>(
      "Compute blended Jacobian wrt skeleton", &ResolveJacobianDJoints, reg, entities);
}

template <typename DiscretizationT>
void blended::SetupCollidingJacobians(
    ecs::Included<TagBlendedActor>,
    ecs::PartialRegistry<CDofOffset const, rom::CRomJacobian const> reg,
    CBlendedComposition const& composition,
    CDofOffset const& dofOffset,
    DiscretizationT const& discretization,
    CBlendingData const& blendingData,
    CContactPartitions const& contactPartitions,
    CArticulatedSkinningData const& skinningInfo,
    CArticulatedLinkTransforms<TimeStep::Current> const& linkTransforms,
    CCollJacs<CollRole::Colliding>& outJacobians) {
  MOCHI_PROFILE_SCOPE();

  // Reserve stack memory for up to 256 elements.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(JacData*) * 256);
  auto jacobiansActive = outJacobians.GetPtrsNonEmpty(&allocator);
  if (jacobiansActive.empty()) {
    return;
  }

  // Prepare bone rotations
  std::vector<VMatrix3x3r> rotations(linkTransforms.size());
  auto const& transforms = skinningInfo.skinningTransform.GetParameterizations();
  for (int i = 0; i < rotations.size(); i++) {
    auto const& preTransform = transforms[i].preTransform.GetRotation();
    rotations[i] = ToVMatrix3x3(linkTransforms[i].GetRotation() * preTransform);
  }

  // Compute Jacobians
  ParallelForEach("SetupCollidingJacobianSoftSkinned", jacobiansActive, 1, [&](JacData* jacData) {
    discretization.Visit([&](auto const& discretizationImpl) {
      using DiscretizationImplT = std::decay_t<decltype(discretizationImpl)>;
      using DQuad = DMapQuad<typename DiscretizationImplT::ElementT>;
      using DSoft = DMap<DQuad, DMapSkinInput, DMapBlending, DMapSoft>;
      using DRom = DMap<DQuad, DMapSkinInput, DMapBlending, DMapRom>;
      using DNoSoft = DMap<DQuad, DMapSkinNoInput>;
      using DMapThis = std::variant<DSoft, DRom, DNoSoft>;
      auto const& skinningJacobian = skinningInfo.jacobianDJoints;
      auto const& skinningData = skinningInfo.skinningData;

      // Get the soft actor from the contact partition [possibly none].
      std::optional<entt::entity> soft;
      auto const& variantId =
          contactPartitions[jacData->query->collidingPartitionId].GetDofDescriptors()[1];
      int softId = std::get<int>(variantId);
      if (softId >= 0) {
        soft = composition.soft[softId];
      }

      // Prepare dmaps that depend on the soft actor
      std::optional<DMapSoft> dsoft;
      std::optional<DMapRom> drom;
      std::optional<DMapBlending> dblend;
      if (soft) {
        auto const& softDofOffset = reg.get<CDofOffset const>(*soft).dofsOffset;
        auto const* romJacobian = reg.try_get<rom::CRomJacobian const>(*soft);
        if (romJacobian) {
          drom.emplace(0, romJacobian->value, softDofOffset);
        } else {
          dsoft.emplace(0, softDofOffset);
        }
        dblend.emplace(blendingData[softId]);
      }

      // Prepare the skinning dmap
      std::optional<DMapSkinInput> dskinIn;
      std::optional<DMapSkinNoInput> dskinNoIn;
      auto const& dofsVariant =
          contactPartitions[jacData->query->collidingPartitionId].GetDofDescriptors()[0];
      auto dofs = MakeConstSpan(std::get<DynamicArray<int>>(dofsVariant));
      if (soft) {
        dskinIn.emplace(1, skinningJacobian, dofs, dofOffset.dofsOffset, skinningData, rotations);
      } else {
        dskinNoIn.emplace(1, skinningJacobian, dofs, dofOffset.dofsOffset);
      }

      // Prepare the quadrature dmap
      DQuad dquad(discretizationImpl.femElements, jacData->query->jacColliderFromWorld);

      // Create per-partition differentiable map
      DMapThis dmap = soft
          ? (drom ? DMapThis{DRom(&dquad, &dskinIn.value(), &dblend.value(), &drom.value())}
                  : DMapThis{DSoft(&dquad, &dskinIn.value(), &dblend.value(), &dsoft.value())})
          : DMapThis{DNoSoft(&dquad, &dskinNoIn.value())};

      // Compute Jacobian
      auto& jacs = *jacData->jacs;
      std::visit([&](auto const& dmap) { dmap.GetJac(jacData->query->sampleIndices, jacs); }, dmap);
      jacs[0].CompressIndices();
      jacs[1].CompressIndices();
    });
  });
}

#define MOCHI_BLENDED_SETUP_COLLIDING_JACOBIANS_INST(DISCRETIZATION_TYPE) \
  template void blended::SetupCollidingJacobians<DISCRETIZATION_TYPE>(    \
      ecs::Included<TagBlendedActor>,                                     \
      ecs::PartialRegistry<CDofOffset const, rom::CRomJacobian const>,    \
      CBlendedComposition const&,                                         \
      CDofOffset const&,                                                  \
      DISCRETIZATION_TYPE const&,                                         \
      CBlendingData const&,                                               \
      CContactPartitions const&,                                          \
      CArticulatedSkinningData const&,                                    \
      CArticulatedLinkTransforms<TimeStep::Current> const&,               \
      CCollJacs<CollRole::Colliding>&);
MOCHI_BLENDED_SETUP_COLLIDING_JACOBIANS_INST(CFemBoundaryDiscretization);
MOCHI_BLENDED_SETUP_COLLIDING_JACOBIANS_INST(CFemSurfaceDiscretization);

namespace mochi::blended {
void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CBlendingData>(reg);
  ecs::RegisterComponent<CBlendedActiveNodes>(reg);
  ecs::RegisterComponent<CBlendedComposition>(reg);
}
} // namespace mochi::blended
