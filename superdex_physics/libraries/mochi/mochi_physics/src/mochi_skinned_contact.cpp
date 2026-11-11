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

#include "mochi_skinned_contact.h"

#include "mochi_simulation.h"

#include <mochi_core/contact/contact_partition.h>
#include <mochi_core/contact/dmap.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/utils/dskinning.h>
#include <mochi_core/utils/task_scheduler.h>

#include <type_traits>
#include <variant>

using namespace mochi;
using namespace mochi::dmap;

// Stack size to gather the frames of all bones. Larger skeletons use the heap.
constexpr int kMaxStackBones = 64;
constexpr size_t kSkinningBonesStackSize =
    kMaxStackBones * sizeof(SkinningBoneFrame) + alignof(SkinningBoneFrame);

// Gather the frames of all bones, indexed by bone id.
static void GatherSkinningBones(
    CArticulatedLinkTransforms<TimeStep::Current> const& linkTransforms,
    DSkinningTransform const& skinningTransform,
    DynamicArray<SkinningBoneFrame>& outBones) {
  int const numBones = skinningTransform.GetBoneCount();
  MOCHI_ASSERT_VERBOSE(isize(linkTransforms) == numBones);
  outBones.reserve(numBones);
  for (int boneId = 0; boneId < numBones; ++boneId) {
    // Bone pre-transform rotated by the link: its translation is the offset from the link origin to
    // the reference origin, without subtracting world positions.
    TransformRT const rotatedPreTransform = TransformRT{linkTransforms[boneId].GetRotation()} *
        skinningTransform.GetBonePreTransform(boneId);
    outBones.emplace_back(
        SkinningBoneFrame{
            .worldFromReferenceRotationTranspose =
                ToVMatrix3x3Transpose(rotatedPreTransform.GetRotation()),
            .linkOriginToReferenceOrigin = rotatedPreTransform.VGetTranslation()});
  }
}

// Express the rigid Jacobian slices of the bones in 'boneIds', one slice per bone from slice 0, wrt
// the articulation DoFs through the Jacobians of their links. Then compress the indices of all
// slices.
static void FinishJacobians(
    ArticulatedJacobian const& jacobian,
    int articulatedDofOffset,
    Span<int const> boneIds,
    Span<ContactJac> jacs) {
  MOCHI_ASSERT_VERBOSE(isize(boneIds) <= isize(jacs));
  for (int slice = 0; slice < isize(boneIds); ++slice) {
    int const boneId = boneIds[slice];
    auto const linkJacobian = jacobian.Link(boneId);
    jacs[slice].SetJacAux(
        linkJacobian.block, articulatedDofOffset, linkJacobian.dofs, RigidSize::kDAll * boneId);
  }
  for (auto& jac : jacs) {
    jac.CompressIndices();
  }
}

// Write only the structure of the contact Jacobian slices of the bones in 'boneIds', one slice per
// bone, for assembly from contact sums.
static void SetupRigidStructureJacobians(
    int dofOffset,
    Span<int const> boneIds,
    Span<int const> sampleIndices,
    Span<ContactJac> outJacobians) {
  for (int slice = 0; slice < isize(boneIds); ++slice) {
    DMapRigidStructure const structureMap(slice, dofOffset);
    DMap<DMapRigidStructure> const dmap(&structureMap);
    dmap.GetJac(sampleIndices, outJacobians);
  }
}

template <typename DiscretizationT>
void articulated::compound::SetupCollidingJacobians(
    ecs::Included<TagArticulatedActor>,
    ecs::Excluded<TagBlendedActor>,
    DiscretizationT const& discretization,
    CDofOffset const& dofOffset,
    CArticulatedJacobian const& jacobian,
    CArticulatedLinkTransforms<TimeStep::Current> const& linkTransforms,
    CArticulatedSkinningData const& skinningData,
    CContactPartitions const& contactPartitions,
    CCollJacs<CollRole::Colliding>& outJacobians) {
  MOCHI_PROFILE_SCOPE();

  if (outJacobians.empty()) {
    return;
  }

  auto const& skinningTransform = skinningData.skinningTransform;
  MOCHI_FILO_STACK_ALLOCATOR(bonesAllocator, kSkinningBonesStackSize);
  DynamicArray<SkinningBoneFrame> bones(&bonesAllocator);
  GatherSkinningBones(linkTransforms, skinningTransform, bones);
  auto const restPositions = Unflatten<Real3 const>(MakeConstSpan(skinningData.restCoords));

  discretization.Visit([&](auto const& discretizationImpl) {
    using ElementT = typename std::decay_t<decltype(discretizationImpl)>::ElementT;
    using DQuad = DMapQuad<ElementT>;
    // Process all Jacobians in parallel
    ParallelForEach(
        "SetupCollidingJacobiansArticulatedCompound", outJacobians, 1, [&](JacData& jacData) {
          auto const& boneIds = std::get<DynamicArray<int>>(
              contactPartitions[jacData.query->collidingPartitionId].GetDofDescriptors()[0]);
          if (boneIds.empty()) {
            jacData.SetZeroDofJacobian();
            return;
          }
          auto jacs = jacData.PrepareJacs(isize(boneIds));
          if (jacData.useContactSums) {
            SetupRigidStructureJacobians(
                dofOffset.dofsOffset, boneIds, jacData.query->sampleIndices, jacs);
          } else {
            DQuad dquad(discretizationImpl.femElements, jacData.query->jacColliderFromWorld);
            DMapSkinning const skinningMap(
                0, skinningTransform, restPositions, bones, boneIds, dofOffset.dofsOffset);
            DMap<DQuad, DMapSkinning> const dmap(&dquad, &skinningMap);
            dmap.GetJac(jacData.query->sampleIndices, jacs, jacData.JacScratch());
          }
          FinishJacobians(jacobian.value, dofOffset.dofsOffset, boneIds, jacs);
        });
  });
}

#define MOCHI_ARTICULATED_COMPOUND_SETUP_COLLIDING_JACOBIANS_INST(DISCRETIZATION_TYPE) \
  template void articulated::compound::SetupCollidingJacobians<DISCRETIZATION_TYPE>(   \
      ecs::Included<TagArticulatedActor>,                                              \
      ecs::Excluded<TagBlendedActor>,                                                  \
      DISCRETIZATION_TYPE const&,                                                      \
      CDofOffset const&,                                                               \
      CArticulatedJacobian const&,                                                     \
      CArticulatedLinkTransforms<TimeStep::Current> const&,                            \
      CArticulatedSkinningData const&,                                                 \
      CContactPartitions const&,                                                       \
      CCollJacs<CollRole::Colliding>&);
MOCHI_ARTICULATED_COMPOUND_SETUP_COLLIDING_JACOBIANS_INST(CFemBoundaryDiscretization);
MOCHI_ARTICULATED_COMPOUND_SETUP_COLLIDING_JACOBIANS_INST(CFemSurfaceDiscretization);
#undef MOCHI_ARTICULATED_COMPOUND_SETUP_COLLIDING_JACOBIANS_INST

void skinned::SetupCollidingJacobians(
    ecs::Included<TagNestedSoftActor>,
    ecs::PartialRegistry<
        CDofOffset const,
        CArticulatedLinkTransforms<TimeStep::Current> const,
        CArticulatedJacobian const> reg,
    CSkinnedComposition const& composition,
    CDofOffset const& dofOffset,
    CContactPartitions const& contactPartitions,
    CFemBoundaryDiscretization const& discretization,
    CNodePositions const& positions,
    CArticulatedSkinningData const& skinningInfo,
    rom::CRomJacobian const* romJacobian,
    CCollJacs<CollRole::Colliding>& outJacobians) {
  MOCHI_PROFILE_SCOPE();

  if (outJacobians.empty()) {
    return;
  }

  // Prepare articulated-actor data, shared by all partitions
  auto articulated = composition.articulated;
  auto const& articulatedDofOffset = reg.get<CDofOffset const>(articulated).dofsOffset;
  auto const& linkTransforms =
      reg.get<CArticulatedLinkTransforms<TimeStep::Current> const>(articulated);
  auto const& jacobian = reg.get<CArticulatedJacobian const>(articulated).value;
  auto const& skinningTransform = skinningInfo.skinningTransform;
  MOCHI_FILO_STACK_ALLOCATOR(bonesAllocator, kSkinningBonesStackSize);
  DynamicArray<SkinningBoneFrame> bones(&bonesAllocator);
  GatherSkinningBones(linkTransforms, skinningTransform, bones);
  auto const nodePositions = Unflatten<Real3 const>(MakeConstSpan(positions.value));

  // Compute Jacobians
  ParallelForEach("SetupCollidingJacobianSoftSkinned", outJacobians, 1, [&](JacData& jacData) {
    discretization.Visit([&](auto const& discretizationImpl) {
      using DiscretizationT = std::decay_t<decltype(discretizationImpl)>;
      using DQuad = DMapQuad<typename DiscretizationT::ElementT>;
      DQuad dquad(discretizationImpl.femElements, jacData.query->jacColliderFromWorld);
      auto const& boneIds = std::get<DynamicArray<int>>(
          contactPartitions[jacData.query->collidingPartitionId].GetDofDescriptors()[0]);
      int const numBoneJacs = isize(boneIds);
      auto jacs = jacData.PrepareJacs(numBoneJacs + 1);
      DMapSkinning const skinningMap(
          0, skinningTransform, nodePositions, bones, boneIds, articulatedDofOffset);
      if (romJacobian) {
        DMapRom const drom(numBoneJacs, romJacobian->value, dofOffset.dofsOffset);
        DMap<DQuad, DMapSkinning, DMapRom> const dmap(&dquad, &skinningMap, &drom);
        dmap.GetJac(jacData.query->sampleIndices, jacs, jacData.JacScratch());
      } else {
        DMapSoft const dsoft(numBoneJacs, dofOffset.dofsOffset);
        DMap<DQuad, DMapSkinning, DMapSoft> const dmap(&dquad, &skinningMap, &dsoft);
        dmap.GetJac(jacData.query->sampleIndices, jacs, jacData.JacScratch());
      }
      FinishJacobians(jacobian, articulatedDofOffset, boneIds, jacs);
    });
  });
}

template <typename DiscretizationT>
void blended::SetupCollidingJacobians(
    ecs::Included<TagBlendedActor>,
    ecs::PartialRegistry<CDofOffset const, rom::CRomJacobian const> reg,
    CBlendedComposition const& composition,
    CDofOffset const& dofOffset,
    CArticulatedJacobian const& jacobian,
    DiscretizationT const& discretization,
    CBlendingData const& blendingData,
    CContactPartitions const& contactPartitions,
    CNodePositions const& positions,
    CArticulatedSkinningData const& skinningInfo,
    CArticulatedLinkTransforms<TimeStep::Current> const& linkTransforms,
    CCollJacs<CollRole::Colliding>& outJacobians) {
  MOCHI_PROFILE_SCOPE();

  if (outJacobians.empty()) {
    return;
  }

  auto const& skinningTransform = skinningInfo.skinningTransform;
  MOCHI_FILO_STACK_ALLOCATOR(bonesAllocator, kSkinningBonesStackSize);
  DynamicArray<SkinningBoneFrame> bones(&bonesAllocator);
  GatherSkinningBones(linkTransforms, skinningTransform, bones);
  auto const targetPositions = Unflatten<Real3 const>(MakeConstSpan(positions.value));

  // Compute Jacobians
  ParallelForEach("SetupCollidingJacobianBlended", outJacobians, 1, [&](JacData& jacData) {
    auto const& descriptors =
        contactPartitions[jacData.query->collidingPartitionId].GetDofDescriptors();
    auto const& boneIds = std::get<DynamicArray<int>>(descriptors[0]);
    int const softId = std::get<int>(descriptors[1]);
    if (boneIds.empty() && softId < 0) {
      jacData.SetZeroDofJacobian();
      return;
    }
    if (jacData.useContactSums) {
      // Only partitions without a soft actor are assembled from contact sums. Their nodes are not
      // blended, so their positions before skinning are the rest positions, as for articulated
      // skins.
      MOCHI_ASSERT_VERBOSE(softId < 0, "Contact sums do not support blended soft slices.");
      auto jacs = jacData.PrepareJacs(isize(boneIds));
      SetupRigidStructureJacobians(
          dofOffset.dofsOffset, boneIds, jacData.query->sampleIndices, jacs);
      FinishJacobians(jacobian.value, dofOffset.dofsOffset, boneIds, jacs);
      return;
    }

    discretization.Visit([&](auto const& discretizationImpl) {
      using DiscretizationImplT = std::decay_t<decltype(discretizationImpl)>;
      using DQuad = DMapQuad<typename DiscretizationImplT::ElementT>;
      DQuad dquad(discretizationImpl.femElements, jacData.query->jacColliderFromWorld);
      int const numBoneJacs = isize(boneIds);
      int const numJacs = numBoneJacs + (softId >= 0 ? 1 : 0);
      auto jacs = jacData.PrepareJacs(numJacs);
      DMapSkinning const skinningMap(
          0, skinningTransform, targetPositions, bones, boneIds, dofOffset.dofsOffset);
      if (softId < 0) {
        DMap<DQuad, DMapSkinning> const dmap(&dquad, &skinningMap);
        dmap.GetJac(jacData.query->sampleIndices, jacs, jacData.JacScratch());
      } else {
        entt::entity const soft = composition.soft[softId];
        DMapBlending const dblend(blendingData[softId]);
        int const softDofOffset = reg.get<CDofOffset const>(soft).dofsOffset;
        if (auto const* romJacobian = reg.try_get<rom::CRomJacobian const>(soft)) {
          DMapRom const drom(numBoneJacs, romJacobian->value, softDofOffset);
          DMap<DQuad, DMapSkinning, DMapBlending, DMapRom> const dmap(
              &dquad, &skinningMap, &dblend, &drom);
          dmap.GetJac(jacData.query->sampleIndices, jacs, jacData.JacScratch());
        } else {
          DMapSoft const dsoft(numBoneJacs, softDofOffset);
          DMap<DQuad, DMapSkinning, DMapBlending, DMapSoft> const dmap(
              &dquad, &skinningMap, &dblend, &dsoft);
          dmap.GetJac(jacData.query->sampleIndices, jacs, jacData.JacScratch());
        }
      }
      FinishJacobians(jacobian.value, dofOffset.dofsOffset, boneIds, jacs);
    });
  });
}

#define MOCHI_BLENDED_SETUP_COLLIDING_JACOBIANS_INST(DISCRETIZATION_TYPE) \
  template void blended::SetupCollidingJacobians<DISCRETIZATION_TYPE>(    \
      ecs::Included<TagBlendedActor>,                                     \
      ecs::PartialRegistry<CDofOffset const, rom::CRomJacobian const>,    \
      CBlendedComposition const&,                                         \
      CDofOffset const&,                                                  \
      CArticulatedJacobian const&,                                        \
      DISCRETIZATION_TYPE const&,                                         \
      CBlendingData const&,                                               \
      CContactPartitions const&,                                          \
      CNodePositions const&,                                              \
      CArticulatedSkinningData const&,                                    \
      CArticulatedLinkTransforms<TimeStep::Current> const&,               \
      CCollJacs<CollRole::Colliding>&);
MOCHI_BLENDED_SETUP_COLLIDING_JACOBIANS_INST(CFemBoundaryDiscretization);
MOCHI_BLENDED_SETUP_COLLIDING_JACOBIANS_INST(CFemSurfaceDiscretization);
#undef MOCHI_BLENDED_SETUP_COLLIDING_JACOBIANS_INST
