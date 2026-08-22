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

#include "mochi_soft_rom_systems.h"

#include "mochi_actor_convergence.h"
#include "mochi_deformable.h"
#include "mochi_discretization_functions.h"
#include "mochi_integration.h"
#include "mochi_scene_recorder.h"
#include "mochi_soft.h"
#include "mochi_soft_rom_linear_systems.h"
#include "mochi_soft_rom_neural_net_crom_systems.h"
#include "mochi_soft_rom_pivot.h"

#include <mochi_core/element_operations/element_assembler.h>
#include <mochi_core/linear_algebra/factor_kernels.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/linear_algebra/qr.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/materials/batched_materials.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include "mochi_hyper_reduction.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#if MOCHI_USE_EIGEN
#include <Eigen/Dense>
#endif

using namespace mochi;
using namespace mochi::experimental;
using namespace mochi::rom;

static constexpr real kDResRegularizationCoefficient = 1e-6_r;

#if MOCHI_USE_EIGEN
static constexpr real kQQTRegularizationCoefficient = 1e-8_r;
#endif

/***************************************************************************
  Assemble systems
*/

MOCHI_API void mochi::rom::AssembleFullToReduced(
    AssemblyParams const& params,
    ecs::Included<TagSoftActor, TagRomActor>,
    ecs::OptionalTag<TagSoftSkinnedActor> isSkinned,
    ecs::OptionalTag<TagUseGravity> hasGravityTag,
    ecs::OptionalTag<TagUseInertia> hasInertiaTag,
    ecs::OptionalTag<TagUseStress> hasStressTag,
    ecs::OptionalTag<TagUseContact> hasContactTag,
    CSkinnedEnergy const& skinnedEnergy,
    CRomProjectionStrategy const& projectionStrategy,
    CRomJacobian& jacobian,
    CActorSnle& actorSnle) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(actorSnle.UseReduced(), "Reduced SNLE must be enabled.");

  bool const hasGravity = hasGravityTag && !skinnedEnergy.gravity;
  bool const hasInertia = hasInertiaTag && !skinnedEnergy.inertia;
  bool const hasStress = hasStressTag && !skinnedEnergy.stress;
  bool const hasContact = hasContactTag && !isSkinned;
  if ( // Nothing to project.
      (!hasGravity && !hasInertia && !hasStress && !hasContact) ||
      // Projection is performed element-by-element. Full actor is not explicitly assembled.
      (projectionStrategy.value == RomProjectionStrategy::ElementLevelProjection)) {
    return;
  } else {
    MOCHI_ASSERT_VERBOSE(
        projectionStrategy.value == RomProjectionStrategy::ActorLevelProjection,
        "Unexpected ROM projection strategy.");
  }

  // Project the residual
  if (params.assemRes) {
    MOCHI_PROFILE_SCOPE_N("Project Residual");
    jacobian.ApplyTranspose(actorSnle.fullResidual, actorSnle.reducedResidual);
  }

  // Project the Jacobian
  if (params.assemDRes) {
    MOCHI_PROFILE_SCOPE_N("Project DResidual");
    using BSpMat = BlockSparseMatrix<real, 3>;
    MOCHI_ASSERT(std::holds_alternative<BSpMat>(actorSnle.fullDResidual));
    MOCHI_ASSERT(
        std::holds_alternative<Matrix<real>>(actorSnle.reducedDResidual),
        "Expected dense storage.");
    auto const& D = std::get<BSpMat>(actorSnle.fullDResidual);
    auto& JtDJ = std::get<Matrix<real>>(actorSnle.reducedDResidual);
    bool const hasDResSource = hasInertia || hasStress || hasContact;
    if (hasDResSource) {
      jacobian.Conjugate(AsConstView(D), AsView(JtDJ));
    } else {
      MOCHI_ASSERT_VERBOSE(D.Norm() == 0_r, "Expected zero DResidual.");
      JtDJ.SetZero();
    }

    // regularize the jacobian
    for (int i = 0; i < JtDJ.Rows(); i++) {
      JtDJ(i, i) += kDResRegularizationCoefficient;
    }
  }
}

MOCHI_API void mochi::rom::AssembleAndProjectBody(
    AssemblyParams const& params, // External parameter
    ecs::Included<TagSoftActor, TagRomActor>,
    ecs::CtxGlobal<CSceneGravity const> sceneGravity,
    ecs::OptionalTag<TagUseGravity> hasGravityTag,
    ecs::OptionalTag<TagUseInertia> hasInertiaTag,
    ecs::OptionalTag<TagUseStress> hasStressTag,
    [[maybe_unused]] ecs::OptionalTag<TagSoftSkinnedActor> isSkinned,
    CSkinnedEnergy const& skinnedEnergy,
    CRomProjectionStrategy const& projectionStrategy,
    CLocal2GlobalMap const& l2g,
    CFemVolumeDiscretizationP1Q1 const& femLowVolDisc,
    CFemVolumeDiscretizationP1Q4 const& femHighVolDisc,
    CRootTransform const& rootTransform,
    CSoftMaterialParams const& materialParams,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::Current> const& currDispl,
    CDisplacementSlice<real, TimeStep::StageStart> const& stageStartDispl,
    CVelocitySlice<real, TimeStep::StageStart> const& stageStartVel,
    CPerElementMassMatrix<CFemVolumeDiscretizationP1Q4> const& perElemMass,
    CRomJacobian const& jacobian,
    CActorSnle& outSnle,
    CActiveVolumeElements const* activeVolElems) {
  MOCHI_PROFILE_SCOPE();
  if (projectionStrategy.value != RomProjectionStrategy::ElementLevelProjection) {
    // Skip if not using element-level projection.
    return;
  }

  bool const hasGravity = hasGravityTag && !skinnedEnergy.gravity;
  bool const hasInertia = hasInertiaTag && !skinnedEnergy.inertia;
  bool const hasStress = hasStressTag && !skinnedEnergy.stress;

  // Skinned ROM actors do not currently support element-level projection. If that ever changes, the
  // "at least one unposed term" assert below must be revisited (a skinned actor with only posed
  // terms is a legal configuration).
  MOCHI_ASSERT_VERBOSE(
      !isSkinned, "Element-level projection is not supported for soft-skinned ROM actors.");
  MOCHI_ASSERT_VERBOSE(
      hasGravity || hasInertia || hasStress,
      "ROM actors with element-level projection must have at least one of inertia, gravity or stress enabled.");
  MOCHI_ASSERT_VERBOSE(outSnle.UseReduced(), "Reduced SNLE must be enabled.");
  MOCHI_ASSERT_VERBOSE(
      std::holds_alternative<Matrix<real>>(outSnle.reducedDResidual),
      "Expected dense reduced dresidual.");
  MOCHI_ASSERT_VERBOSE(
      jacobian.template TryGet<CRomJacobian::DenseT>(), "Expected dense ROM Jacobian.");

  // Clear SNLE data.
  auto& JtDJ = std::get<Matrix<real>>(outSnle.reducedDResidual);
  if (params.assemObj) {
    outSnle.objective = 0.0;
  }
  if (params.assemRes) {
    outSnle.reducedResidual.SetZero();
  }
  if (params.assemDRes) {
    JtDJ.SetZero();

    // Regularize the dresidual.
    for (int i = 0; i < JtDJ.Rows(); ++i) {
      JtDJ(i, i) = kDResRegularizationCoefficient;
    }
  }

  AssemblyParams bodyParams = params;
  bodyParams.assemDRes &= (hasInertia || hasStress); // Only inertia and stress contribute to DRes
  if (!bodyParams.assemObj && !bodyParams.assemRes && !bodyParams.assemDRes) {
    return;
  }

  auto const dampingScales =
      soft::ComputeSoftDampingScales(materialParams, intState.dtStage, hasInertia, hasStress);
  real const massScale = dampingScales.massScale;
  real const massDampingScale = dampingScales.massDampingScale;
  real const stiffnessDampingFactor = dampingScales.stiffnessDampingFactor;
  bool const includeStiffnessDampingGeometricTerm =
      dampingScales.includeStiffnessDampingGeometricTerm;
  // Perform assembly and projection.
  auto initEleDResFn = [&](auto&& eleDRes, int globalEleIdx) -> bool {
    // Initialize the element dresidual to the scaled element mass matrix (inertia + mass damping)
    // if applicable, and to zero otherwise.
    if (massScale != 0_r) {
      auto const& elementMass = perElemMass.values[globalEleIdx];
      int constexpr kNumElemDofs =
          CPerElementMassMatrix<CFemVolumeDiscretizationP1Q4>::kNumElemDofs;
      RowMatrixView<real const, kNumElemDofs, kNumElemDofs> matView(&elementMass[0][0]);

      static_assert(
          krylov::details::MatTraits<decltype(eleDRes)>::kMajorDir ==
              krylov::details::MatTraits<decltype(matView)>::kMajorDir,
          "Expected same storage direction for performance reasons.");

      eleDRes = massScale * matView;
      return true;
    } else {
      eleDRes.SetZero();
      return false;
    }
  };

  MOCHI_ASSERT_VERBOSE(
      !activeVolElems || !activeVolElems->empty(), "Active volume elements must not be empty.");
  AssemblyActiveSubset activeSubset = activeVolElems
      ? AssemblyActiveSubset{activeVolElems->ViewIndices(), activeVolElems->ViewIsActive()}
      : AssemblyActiveSubset{};

  Span<real const> activeVolWeights =
      activeVolElems ? activeVolElems->ViewWeights() : Span<real const>{};
  auto const gravity =
      ToReal3(rootTransform.worldFromLocal.TransformDirectionInverse(sceneGravity->accel));

  soft::details::VisitPerElementMaterialParams(
      materialParams, [&]<typename ParamsT>(auto const& perElem) {
        auto batchedConstitutive = materials::MakeBatchedConstitutiveResponse<ParamsT>(perElem);
        auto bodyOp = soft::MakeBatchedBodyOp(
            MakeConstSpan(femLowVolDisc.femElements),
            MakeConstSpan(femHighVolDisc.femElements),
            batchedConstitutive,
            materialParams.referenceMaterialStiffness,
            hasStress,
            hasGravity,
            hasInertia,
            gravity,
            materialParams.density,
            stageStartDispl.value.GetConstSpan(),
            stageStartVel.value.GetConstSpan(),
            intState.dtStage,
            massDampingScale,
            stiffnessDampingFactor,
            includeStiffnessDampingGeometricTerm,
            activeVolWeights);

        AssembleAndProjectObjResDRes<soft::SoftStencilElement>(
            l2g,
            bodyOp,
            activeSubset,
            initEleDResFn,
            AsConstView(currDispl.value),
            AsConstView(*jacobian.template TryGet<CRomJacobian::DenseT>()),
            outSnle.objective,
            AsView(outSnle.reducedResidual),
            AsView(JtDJ),
            bodyParams);
      });
}

void mochi::rom::AssembleAndProjectAsyncContact(
    AssemblyParams const& params, // External parameter
    entt::entity e,
    ecs::Included<TagSoftActor, TagRomActor, TagUseContact>,
    ecs::Excluded<TagSoftSkinnedActor>,
    ecs::OptionalTag<TagQueryActiveContacts> queryActiveContacts,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    ContactAssemblyReg reg,
    CRomProjectionStrategy const& projectionStrategy,
    CFemBoundaryDiscretization const& femBoundaryDisc,
    CBoundaryLocal2GlobalMap const& bdL2g,
    CContactSamples<TimeStep::Current> const& samples,
    CColliderInfo const& colliderInfo,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& collisions,
    CRootTransform const& rootTransform,
    CTimeIntegratorState const& intState,
    CRomJacobian const& jacobian,
    CDeformablePointAsyncCollisionsResponse& outResponse,
    CActorSnle& outSnle,
    CActiveBoundaryFaces const* activeBoundaryFaces) {
  MOCHI_PROFILE_SCOPE();
  if (projectionStrategy.value != RomProjectionStrategy::ElementLevelProjection) {
    return;
  }

  MOCHI_ASSERT_VERBOSE(outSnle.UseReduced(), "Reduced SNLE must be enabled.");
  MOCHI_ASSERT_VERBOSE(
      std::holds_alternative<Matrix<real>>(outSnle.reducedDResidual), "Expected dense dresidual.");
  MOCHI_ASSERT_VERBOSE(
      jacobian.template TryGet<CRomJacobian::DenseT>(), "Expected dense ROM Jacobian.");

  if (collisions.empty() || (activeBoundaryFaces && activeBoundaryFaces->empty())) {
    // No contacts or no active boundary faces.
    outResponse.Clear();
    return;
  }

  // Compute contact response.
  deformable::ComputeAsyncContactResponse<CFemBoundaryDiscretization, kSpaceDim3>(
      reg,
      e,
      simParams->experimentalEval,
      queryActiveContacts,
      femBoundaryDisc,
      samples,
      colliderInfo,
      collisions,
      intState,
      rootTransform,
      params,
      outResponse,
      activeBoundaryFaces ? activeBoundaryFaces->ViewIsActive() : Span<bool const>{});

  if (outResponse.Empty()) {
    // No contacts.
    return;
  }

  auto initEleDResFn = [](auto&& eleDRes, int /*globalEleIdx*/) -> bool {
    eleDRes.SetZero();
    return false;
  };

  femBoundaryDisc.Visit([&](auto const& disc) {
    using DiscT = std::decay_t<decltype(disc)>;
    using ElementT = typename DiscT::ElementT;

    AssemblyActiveSubset const bdActiveSubset = outResponse.ViewActiveContactElementSubset();

    Span<real const> bdFaceWeights =
        activeBoundaryFaces ? activeBoundaryFaces->ViewWeights() : Span<real const>{};
    auto boundaryOp = deformable::MakeBatchedBoundaryOp(
        MakeConstSpan(disc.femElements), outResponse, bdFaceWeights);

    AssembleAndProjectObjResDRes<ElementT>(
        bdL2g,
        boundaryOp,
        bdActiveSubset,
        initEleDResFn,
        AsConstView(*jacobian.template TryGet<CRomJacobian::DenseT>()),
        outSnle.objective,
        AsView(outSnle.reducedResidual),
        AsView(std::get<Matrix<real>>(outSnle.reducedDResidual)),
        params);
  });
}

/***************************************************************************
  Related to rigid transform
*/
template <TimeStep kTimeStep>
std::optional<TransformRT> mochi::rom::GetRigidTransform(
    RigidTransformRegistryExp reg,
    entt::entity entity) {
  // Confirm if this is a ROM actor
  if (!reg.all_of<TagRomActor const>(entity)) {
    return std::nullopt;
  }

  // Check if the actor has a rigid transform layer
  auto const* rigidState = reg.try_get<CRigidState<kTimeStep> const>(entity);
  if (!rigidState) {
    return std::nullopt;
  }

  // Check if there is pivot
  auto const* pivot = reg.try_get<CMeshPivot const>(entity);
  if (!pivot) {
    return rigidState->value;
  }

  // Add the pivot
  return Repivot(rigidState->value, -pivot->position);
}

template std::optional<TransformRT> mochi::rom::GetRigidTransform<TimeStep::Current>(
    RigidTransformRegistryExp,
    entt::entity);
template std::optional<TransformRT> mochi::rom::GetRigidTransform<TimeStep::StageStart>(
    RigidTransformRegistryExp,
    entt::entity);
template std::optional<TransformRT> mochi::rom::GetRigidTransform<TimeStep::Previous>(
    RigidTransformRegistryExp,
    entt::entity);

void mochi::rom::AddRigidContactJacobians(
    RigidTransformRegistryExp reg,
    entt::entity entity,
    ContactDetectionResult& outContact) {
  // Confirm if this is a ROM actor
  if (!reg.all_of<TagRomActor>(entity)) {
    return;
  }

  // Check if the actor has a rigid transform layer
  auto const* rigidStateComp = reg.try_get<CRigidState<TimeStep::Current> const>(entity);
  if (!rigidStateComp) {
    return;
  }
  auto const& rigidState = rigidStateComp->value;

  // Get the root transform and the pivot (if there is one)
  auto const& transformRoot = reg.get<CRootTransform const>(entity).worldFromLocal;
  auto const* pivot = reg.try_get<CMeshPivot const>(entity);

  // p = contact point in world coordinates
  // x = contact point in local coordinates
  // troot,Rroot = root transform of the actor
  // tlocal,Rlocal = rigid transform layer of the actor
  // v = pivot of the rigid transform layer (optional)

  // Transformation from local coords to world coords:
  // p = Rroot * (Rlocal * x + tlocal + v) + troot
  // Viceversa:
  // x = Rlocal^T * (Rroot^T * (p - troot) - tlocal - v)
  // xT = ((p - troot)^T * Rroot - (tLocal + v)^T) * Rlocal

  // Jacobians:
  // dp/dtLocal = Rroot
  // z = Rlocal * x = Rroot^T * (p - troot) - tlocal - v
  // dp/drLocal = Rroot * skew(-z)
  // dp/dtLocal^T = Rroot^T
  // dp/drLocal^T = skew(z) * Rroot^T

  // Prepare common data
  Vec4r troot = ToSimd(transformRoot.GetTranslation());
  std::pair<VMatrix3x3r, VMatrix3x3r> Rroot_T =
      ToVMatrix3x3_WithTranspose(transformRoot.GetRotation());
  Vec4r tlocalAndPivot = rigidState.VGetTranslation();
  if (pivot) {
    tlocalAndPivot += ToSimd(pivot->position);
  }
  VMatrix3x3r const& DpDtlocalT = Rroot_T.second;

  // Resize number of DoFs
  int ndofsOld = outContact.ndofs;
  outContact.ndofs += RigidSize::kDAll;
  MOCHI_ASSERT(
      ColliderJacDofs::kMaxDoFs >= outContact.ndofs,
      "ColliderJacDofs::kMaxDoFs is not sufficiently large");

  // Traverse all the contact samples and add rigid Jacobians
  for (int s = 0; s < isize(outContact.sampleIndices); ++s) {
    ColliderJacDofs& jacWorldFromDofs = outContact.jacWorldFromDofs[s];
    auto* inds = jacWorldFromDofs.inds.data();
    auto* ddef_ddofs = jacWorldFromDofs.jac.data();

    // Reset DoF indices
    std::iota(inds, inds + outContact.ndofs, 0);

    // Move soft Jacobian
    std::copy_backward(ddef_ddofs, ddef_ddofs + ndofsOld, ddef_ddofs + outContact.ndofs);

    // Compute rotated local contact point
    Vec4r p = ToSimd(outContact.posColliding[s]);
    Vec4r z = DotVecMat3x3(p - troot, Rroot_T.first) - tlocalAndPivot;

    // Compute rotation Jacobian
    VMatrix3x3r DpDRlocalT = Dot3x3(lie::DMultRotVecDRot(-z), Rroot_T.second);

    // Add translation and rotation Jacobians. Transposed for convenience of the copy.
    std::copy(DpDtlocalT.data(), DpDtlocalT.data() + RigidSize::kDTrans, ddef_ddofs);
    std::copy(
        DpDRlocalT.data(), DpDRlocalT.data() + RigidSize::kDRot, ddef_ddofs + RigidSize::kDTrans);
  }
}

/***************************************************************************
  Recording Systems
*/

void mochi::rom::RecordRomAmplitudes(
    ecs::RequiredTag<TagRomActor>,
    CRomModeAmplitudes const& amplitudes,
    CRecordingData& outData) {
  RecordDataset("romAmplitudes", AsConstView(amplitudes.value), outData);
}

void mochi::rom::RecordRomRigidTransform(
    ecs::RequiredTag<TagRomActor>,
    CRigidState<TimeStep::Current> const& rigid,
    CRecordingData& outData) {
  ColumnVector<real, RigidSize::kAll> raw;
  TransformToRawPose(rigid.value, raw);
  RecordDataset("romRigidTransform", AsConstView(raw), outData);
}

void mochi::rom::RecordingPipeline(entt::registry& reg, Span<entt::entity const> entities) {
  ecs::InvokeForEach(&RecordRomAmplitudes, reg, entities);
  ecs::InvokeForEach(&RecordRomRigidTransform, reg, entities);
}

/***************************************************************************
  Static functions needed by ROM pipelines
*/

static void ResolveDisplacementOnAllNodes(entt::registry& reg, Span<entt::entity const> entities) {
  ecs::InvokeForEach(&rom::linear::ResolveDisplacement</*kForceUseAllNodes*/ true>, reg, entities);
}

static void ResolveDisplacementAndJacobianOnActiveNodes(
    entt::registry& reg,
    Span<entt::entity const> entities) {
  ecs::InvokeForEach(
      &rom::linear::ResolveDisplacementAndJacobian</*kForceUseAllNodes*/ false>, reg, entities);
}

/**************************************************************************
  Pre-step
*/

static int CountActiveCollisionPoints(
    CActiveCollisions<ContactType::Async, TimeStep::Current> const& activeCollisionsAsync,
    CActiveCollisions<ContactType::Sync, TimeStep::Current> const& activeCollisionsSync) {
  int count = 0;
  for (auto const& collisions : activeCollisionsAsync) {
    count += isize(collisions.collisionResult.sampleIndices);
  }
  for (auto const& collisions : activeCollisionsSync) {
    count += isize(collisions.collisionResult.sampleIndices);
  }
  return count;
}

static void MoveSampleMeshToCache(
    CSampleMeshCaching& sampleMeshCache,
    CActiveVolumeElements& ave,
    CActiveBoundaryFaces& abf,
    CActiveUniqueNodes& an) {
  sampleMeshCache.activeVolumeElements = std::make_unique<CActiveVolumeElements>(std::move(ave));
  sampleMeshCache.activeBoundaryFaces = std::make_unique<CActiveBoundaryFaces>(std::move(abf));
  sampleMeshCache.activeNodes = std::make_unique<CActiveUniqueNodes>(std::move(an));
}

static void SoftMassUpdateRomFomSwitching(
    CActorSnle const& actorSnle,
    CSoftMaterialParams const& material,
    CFemVolumeDiscretizationP1Q4 const& femHighVolDisc,
    CLocal2GlobalMap const& l2g,
    CNodalBasedStructure const& nbs,
    CFullSparsityPattern const& sparsity,
    CMassMatrix& massMatrix,
    CPerElementMassMatrix<CFemVolumeDiscretizationP1Q4>& perElemMass,
    CLumpedMassMatrix& lumpedMass,
    CActiveVolumeElements* ave = nullptr) {
  soft::UpdateSoftMass(
      actorSnle,
      l2g,
      nbs,
      material,
      femHighVolDisc,
      sparsity,
      massMatrix,
      perElemMass,
      lumpedMass,
      ave);
}

/// @brief Update components to switch from ROM to FOM.
static void RomToFomImpl(entt::registry& reg, entt::entity e) {
  reg.remove<TagRomActor>(e);

  CActorDofInfo& dofInfo = reg.get<CActorDofInfo>(e);
  auto const& mesh = reg.get<CTetrahedralMesh const>(e);
  dofInfo.poseSize = mesh.mesh->GetNumNodes() * kSpaceDim3;
  dofInfo.dofsSize = dofInfo.poseSize;

  // Disable reduced SNLE.
  reg.get<CActorSnle>(e).DisableReduced();

  // Invalidate cached convergence weights.
  InvalidateActorConvergenceWeights(reg, e);

  if (reg.any_of<CActiveVolumeElements, CActiveBoundaryFaces, CActiveUniqueNodes>(e)) {
    MOCHI_ASSERT((reg.all_of<CActiveVolumeElements, CActiveBoundaryFaces, CActiveUniqueNodes>(e)));
    // First, move sample mesh components to the cache.
    mochi::ecs::InvokeOnEntity(&MoveSampleMeshToCache, reg, e);

    // Second, remove sample mesh components.
    reg.remove<CActiveVolumeElements, CActiveBoundaryFaces, CActiveUniqueNodes>(e);

    // Finally, update the mass matrix after the sample mesh has been removed.
    mochi::ecs::InvokeOnEntity(&SoftMassUpdateRomFomSwitching, reg, e);
  }
}

/// @brief Update components to switch from FROM to ROM.
static void FomToRomImpl(entt::registry& reg, entt::entity e) {
  reg.emplace<TagRomActor>(e);

  CActorDofInfo& dofInfo = reg.get<CActorDofInfo>(e);
  auto const& romProp = reg.get<CRomCommonProperties>(e);
  dofInfo.poseSize = romProp.value.reducedPoseDim;
  dofInfo.dofsSize = romProp.value.reducedDofsDim;

  reg.get<CActorSnle>(e).EnableReduced(
      Matrix<real>::Zero(romProp.value.reducedDofsDim, romProp.value.reducedDofsDim));

  // Invalidate cached convergence weights.
  InvalidateActorConvergenceWeights(reg, e);

  if (auto* sampleMeshToRestore = reg.try_get<CSampleMeshCaching>(e)) {
    // First, emplace sample mesh components.
    reg.emplace<CActiveVolumeElements>(e, std::move(*sampleMeshToRestore->activeVolumeElements));
    reg.emplace<CActiveBoundaryFaces>(e, std::move(*sampleMeshToRestore->activeBoundaryFaces));
    reg.emplace<CActiveUniqueNodes>(e, std::move(*sampleMeshToRestore->activeNodes));

    // Then, update the mass matrix after the sample mesh has been emplaced.
    mochi::ecs::InvokeOnEntity(&SoftMassUpdateRomFomSwitching, reg, e);
  }
}

namespace {
enum class SwitchDirection { RomToFom, FomToRom };
}

static bool ShouldSwitchModel(
    RomFomSwitchingTestOnlyParams const& params,
    entt::registry& reg,
    entt::entity /*e*/,
    SwitchDirection const& direction) {
  auto const& stepCounter = reg.ctx<CSceneStepCounter const>();
  return stepCounter.value ==
      ((direction == SwitchDirection::RomToFom) ? params.romToFomSwappingStep
                                                : params.fomToRomSwappingStep);
}

static bool ShouldSwitchModel(
    RomFomSwitchingContactInformedParams const& params,
    entt::registry& reg,
    entt::entity e,
    SwitchDirection const& direction) {
  MOCHI_ASSERT_VERBOSE(
      (reg.all_of<
          CActiveCollisions<ContactType::Async, TimeStep::Current>,
          CActiveCollisions<ContactType::Sync, TimeStep::Current>>(e)),
      "Missing required components.");

  auto const& activeCollisionsAsync =
      reg.get<CActiveCollisions<ContactType::Async, TimeStep::Current> const>(e);
  auto const& activeCollisionsSync =
      reg.get<CActiveCollisions<ContactType::Sync, TimeStep::Current> const>(e);
  int const count = CountActiveCollisionPoints(activeCollisionsAsync, activeCollisionsSync);

  switch (direction) {
    case SwitchDirection::RomToFom:
      return count >= params.numCollisionPtsThreshold;
    case SwitchDirection::FomToRom:
      return count < params.numCollisionPtsThreshold;
    default:
      MOCHI_ASSERT(false, "Unexpected switch direction.");
      return false;
  };
}

void mochi::rom::RomFomSwitchingPipeline(
    entt::registry& reg,
    entt::entity e,
    CRomFomSwitchingParams const& params) {
  MOCHI_PROFILE_SCOPE();

  if (reg.all_of<TagRomActor>(e)) {
    bool const shouldSwitch = std::visit(
        [&](auto const& p) -> bool {
          return ShouldSwitchModel(p, reg, e, SwitchDirection::RomToFom);
        },
        params.params);
    if (shouldSwitch) {
      RomToFomImpl(reg, e);
    }
  } else {
    bool const shouldSwitch = std::visit(
        [&](auto const& p) -> bool {
          return ShouldSwitchModel(p, reg, e, SwitchDirection::FomToRom);
        },
        params.params);
    if (shouldSwitch) {
      FomToRomImpl(reg, e);
    }
  }
}

// Propose a new basis based on the contact forces from the previous step.
static void ProposeNewLinearBasisContactForceInformedStrategy(
    ecs::RequiredTag<TagRomActor>,
    CRomAdaptiveBasisContactForceInformed const& strategy,
    CQueryNodeContactForces const* contactForces,
    CRootTransform const* rootTransform,
    CRigidState<TimeStep::Current> const& rigidTransform,
    CRomLinearBasis& linearBasis) {
  MOCHI_PROFILE_SCOPE();

  auto const& candidateBasis = strategy.candidateBasis;
  MOCHI_ASSERT(
      contactForces && rootTransform,
      "ContactForceInformedRomAdaptivityParams requires CQueryNodeContactForces and rootTransform.");
  MOCHI_ASSERT(
      !strategy.requiredBasis || (candidateBasis.Rows() == strategy.requiredBasis->Rows()),
      "Inconsistent number of rows.");

  // If there is no contact, avoid proposing a new basis.
  if (contactForces->nodeContactForces.empty()) {
    linearBasis.ShouldUseAlternativeBasis(false);
    return;
  }

  // Create the new basis matrix.
  int const numRows = candidateBasis.Rows();
  auto& proposedBasis = linearBasis.GetAlternativeMatrix();
  if (strategy.requiredBasis) {
    int const numRequiredModes = strategy.requiredBasis->Cols();
    proposedBasis.Resize(numRows, numRequiredModes + strategy.numAdaptiveBasis);
    proposedBasis.LeftCols(numRequiredModes) = *strategy.requiredBasis;
  } else {
    proposedBasis.Resize(numRows, strategy.numAdaptiveBasis);
  }

  // "Basis" = "Frame in which the basis in expressed"
  auto worldFromLocal = ToVMatrix4x4(rootTransform->worldFromLocal);
  auto localFromBasis = ToVMatrix4x4(rigidTransform.value);
  auto worldFromBasis = worldFromLocal * localFromBasis;

  // Adaptive modes selected from candidate modes based on the contact force.
  auto br = ColumnVector<real>::Zero(candidateBasis.Cols());
  for (auto const& it : contactForces->nodeContactForces) {
    int const nodeId = it.index;
    auto const forceWorld = it.force;
    Vec4r const forceBasis = DotVecMat3x3(ToSimd(forceWorld, 0_r), worldFromBasis);
    auto const phiRows =
        candidateBasis.template MiddleRows<kSpaceDim3>(nodeId * kSpaceDim3, kSpaceDim3);
    br += phiRows.Transpose() * AsColumnVectorView<kSpaceDim3>(forceBasis);
  }

  for (int i = 0; i < isize(br); ++i) {
    br[i] = -Abs(br[i]);
  }

  auto adaptiveBasis = proposedBasis.RightCols(strategy.numAdaptiveBasis);
  auto const orderedCols = ArgSort<int>(MakeConstSpan(br));
  for (int j = 0; j < strategy.numAdaptiveBasis; ++j) {
    adaptiveBasis.Col(j) = candidateBasis.Col(orderedCols[j]);
  }

  // Mark the alternative basis as ready to use but don't perform basis swapping. That's the
  // responsibility of a different system.
  linearBasis.ShouldUseAlternativeBasis(true);
}

static void ComputeNeuralAffineBasisViaEncoderProjection(
    CNeuralAffineRomStrategy& strategy,
    CDisplacementSlice<real, TimeStep::Current> const& currentDisplacements,
    CTetrahedralMesh const& mesh,
    CRigidState<TimeStep::Current> const& rigidTransform,
    CMeshPivot const& pivot,
    CAuxiliaryPositionsForRomRigidTransform& rigidTransformAuxPositions,
    RowMatrixView<real> proposedBasis) {
  MOCHI_PROFILE_SCOPE();

  int const outputDim = mesh.mesh->GetNumNodes() * 3;
  ColumnVectorView<real> latentState = strategy.latentState;
  ColumnVector<real> currentDisplacementsInBasis = currentDisplacements.value;
  rom::rigid_transform::TransformDisplacements(
      *mesh.mesh,
      /* activeNodes */ Span<int const>{}, // Resolve all nodes (not just the active ones)
      pivot.position,
      Invert(rigidTransform.value),
      rigidTransformAuxPositions.data,
      currentDisplacementsInBasis);

  auto const& standardization = strategy.neuralModel.meanAndStdevForInputStandardize.value();
  MOCHI_ASSERT(standardization.size() == 6);
  // Apply standardization: (x - mean) / std for each spatial dimension
  for (int i = 0; i < outputDim; ++i) {
    int const dim = i % kSpaceDim3; // Which spatial dimension (0, 1, or 2)
    real const meanVal = standardization(dim);
    real const stdVal = standardization(dim + kSpaceDim3);
    currentDisplacementsInBasis(i) = (currentDisplacementsInBasis(i) - meanVal) / stdVal;
  }

  strategy.neuralModel.encoder->Forward(currentDisplacementsInBasis, latentState);

  ColumnVector<real> cromDisplacement(outputDim);
  neural_net_crom::ResolveDisplacementAndJacobian(
      *mesh.mesh,
      /*activeNodes*/ {}, // Resolve all nodes (not just the active ones)
      strategy.neuralModel,
      strategy.decoderData,
      latentState,
      cromDisplacement,
      proposedBasis);
}

static void ComputeNeuralAffineBasisViaNonLinearLeastSquareProjection(
    CNeuralAffineRomStrategy& strategy,
    CDisplacementSlice<real, TimeStep::Current> const& currentDisplacements,
    CTetrahedralMesh const& mesh,
    CRigidState<TimeStep::Current> const& rigidTransform,
    CMeshPivot const& pivot,
    RowMatrixView<real> proposedBasis) {
  MOCHI_PROFILE_SCOPE();

  int const outputDim = mesh.mesh->GetNumNodes() * 3;
  int const qDim = strategy.neuralModel.decoder.InputDim() - 3;
  ColumnVectorView<real> latentState = strategy.latentState;

  // Implement optimization-based projection: min_p Σ||u - h(p, x_i)||²
  // Where:
  // - u is the current displacement (currentDisplacements->value)
  // - h is the decoder network (neural CROM model)
  // - p is the latent state we're solving for
  // - x_i are the spatial coordinates (spatialCoords)

  // Solver parameters.
  int constexpr kMaxIters = 100;
  real constexpr kRelReductionTol = 1e-5_r;
  int constexpr kLineSearchMaxIters = 10;
  real constexpr kLineSearchAlpha = 0.5_r;
  bool constexpr kVerbose = false;

  // Target displacement vector
  ColumnVectorView<real const> targetDisplacement = currentDisplacements.value;

  // Declare matrices outside the loop to avoid repeated allocation
  ColumnVector<real> cromDisplacement(outputDim); // u_{local}
  RowMatrix<real> jacobian(outputDim, qDim); // du_{local}/dp
  ColumnVector<real> r(outputDim);
  ColumnVector<real> dx(qDim);
  ColumnVector<real> latentState0(qDim);
  ColumnVector<real> newResidual(outputDim);

  // Create auxiliary buffers for ResolveDisplacementAndJacobianInLocal
  ColumnVector<real> displacementBuffer(outputDim); // temp buffer, not u_{basis}

  // Perform projection by solving a non-linear least squares problem.
  real residualNorm = std::numeric_limits<real>::max();
  bool needUpdate = false;
  for (int iter = 0; iter < kMaxIters; ++iter) {
    // Use the existing neural net CROM implementation for forward pass and Jacobian
    // computation
    neural_net_crom::ResolveDisplacementAndJacobian(
        *mesh.mesh,
        /*activeNodes*/ {}, // Resolve all nodes (not just the active ones)
        strategy.neuralModel,
        strategy.decoderData,
        latentState,
        cromDisplacement,
        jacobian,
        &rigidTransform.value,
        &pivot.position,
        displacementBuffer,
        proposedBasis);
    needUpdate = false;

    // Compute residual and loss
    real const prevResidualNorm = residualNorm;
    r = targetDisplacement - cromDisplacement;
    residualNorm = r.Norm();

    // Check convergence based on relative improvement
    if (residualNorm >= prevResidualNorm * (1.0_r - kRelReductionTol)) {
      if constexpr (kVerbose) {
        MOCHI_LOG(
            "Neural Affine ROM optimization converged after %d iterations with residual norm %f.",
            iter,
            residualNorm);
      }
      break;
    }

    // Solve for Gauss-Newton step using QR decomposition
    ThinQR<real> qr(jacobian);
    dx = qr.Q().Transpose() * r;
    kernel::BackSubstitutionInPlace(qr.R(), dx);

    // Backtracking line search
    latentState0 = latentState;
    real alpha = 1.0_r;
    bool lineSearchAccepted = false;
    int lsIter = 0;
    while (++lsIter <= kLineSearchMaxIters) {
      // Try step with current alpha
      latentState = latentState0 + alpha * dx;

      neural_net_crom::ResolveDisplacement(
          *mesh.mesh,
          {},
          strategy.neuralModel,
          strategy.decoderData,
          latentState,
          cromDisplacement,
          &rigidTransform.value,
          &pivot.position,
          displacementBuffer);
      newResidual = targetDisplacement - cromDisplacement;
      real newResidualNorm = newResidual.Norm();

      // Check for sufficient decrease
      if (newResidualNorm <= residualNorm) {
        lineSearchAccepted = true;
        needUpdate = true;
        break;
      } else {
        alpha *= kLineSearchAlpha;
      }
    }
    if (!lineSearchAccepted) {
      latentState = latentState0;
      if constexpr (kVerbose) {
        MOCHI_LOG(
            "Neural Affine ROM optimization stopped at iteration %d because line search failed with residual norm %f.",
            iter + 1,
            residualNorm);
      }
      break;
    }
  }

  if (needUpdate) {
    neural_net_crom::ResolveDisplacementAndJacobian(
        *mesh.mesh,
        /*activeNodes*/ {}, // Resolve all nodes (not just the active ones)
        strategy.neuralModel,
        strategy.decoderData,
        latentState,
        cromDisplacement,
        jacobian,
        &rigidTransform.value,
        &pivot.position,
        displacementBuffer,
        proposedBasis);
  }
}

#if MOCHI_USE_EIGEN
static void ComputeNeuralAffineBasisViaInterpolation(
    CNeuralAffineRomStrategy& strategy,
    CDisplacementSlice<real, TimeStep::Current> const& currentDisplacements,
    CTetrahedralMesh const& mesh,
    CRigidState<TimeStep::Current> const& rigidTransform,
    CMeshPivot const& pivot,
    CAuxiliaryPositionsForRomRigidTransform& rigidTransformAuxPositions,
    RowMatrixView<real> proposedBasis) {
  MOCHI_PROFILE_SCOPE();

  int const qDim = strategy.neuralModel.decoder.InputDim() - 3;
  int const outputDim = mesh.mesh->GetNumNodes() * 3;

  ColumnVectorView<real> latentState = strategy.latentState;
  ColumnVector<real> currentDisplacementsInBasis = currentDisplacements.value;
  rom::rigid_transform::TransformDisplacements(
      *mesh.mesh,
      /* activeNodes */ Span<int const>{}, // Resolve all nodes (not just the active ones)
      pivot.position,
      Invert(rigidTransform.value),
      rigidTransformAuxPositions.data,
      currentDisplacementsInBasis);

  auto const& standardization = strategy.neuralModel.meanAndStdevForInputStandardize.value();
  MOCHI_ASSERT(standardization.size() == 6);
  // Apply standardization: (x - mean) / std for each spatial dimension
  for (int i = 0; i < outputDim; ++i) {
    int const dim = i % kSpaceDim3; // Which spatial dimension (0, 1, or 2)
    real const meanVal = standardization(dim);
    real const stdVal = standardization(dim + kSpaceDim3);
    currentDisplacementsInBasis(i) = (currentDisplacementsInBasis(i) - meanVal) / stdVal;
  }

  strategy.neuralModel.encoder->Forward(currentDisplacementsInBasis, latentState);
  strategy.onlineBasisParams.onlineU.push_back(currentDisplacementsInBasis);
  strategy.onlineBasisParams.onlineQ.push_back(strategy.latentState);

  ColumnVector<real> cromDisplacement(outputDim);
  neural_net_crom::ResolveDisplacementAndJacobian(
      *mesh.mesh,
      /*activeNodes*/ {}, // Resolve all nodes (not just the active ones)
      strategy.neuralModel,
      strategy.decoderData,
      latentState,
      cromDisplacement,
      proposedBasis);

  if (strategy.onlineBasisParams.step < strategy.onlineBasisParams.dataDim) {
    strategy.onlineBasisParams.step += 1;
  } else {
    // Remove oldest elements from deques
    strategy.onlineBasisParams.onlineU.pop_front();
    strategy.onlineBasisParams.onlineQ.pop_front();

    int const dataDim = strategy.onlineBasisParams.dataDim;
    RowMatrix<real> dU(outputDim, dataDim);
    RowMatrix<real> Q(qDim, dataDim);
    for (int i = 0; i < dataDim; ++i) {
      dU.Col(i) = strategy.onlineBasisParams.onlineU[i];
      Q.Col(i) = strategy.onlineBasisParams.onlineQ[i];
    }
    RowMatrix<real> QQT = Q * Q.Transpose();

    // Add regularization to QQT before inversion to handle rank-deficient cases
    for (int i = 0; i < QQT.Rows(); ++i) {
      QQT(i, i) += kQQTRegularizationCoefficient;
    }

    RowMatrix<real> displacementBasis = dU * Q.Transpose() * Inverse(QQT);

    ThinQR<real> qr_old(displacementBasis);
    ThinQR<real> qr_new(proposedBasis);

    using EigenMatrixType = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
    using EigenVectorType = Eigen::Matrix<real, Eigen::Dynamic, 1>;
    int const m = qr_old.Q().Rows();
    int const n = qr_old.Q().Cols();

    EigenMatrixType Q_old(m, n);
    EigenMatrixType Q_new(m, n);

    // Copy data from Mochi matrices to Eigen matrices
    for (int row = 0; row < m; ++row) {
      for (int col = 0; col < n; ++col) {
        Q_old(row, col) = qr_old.Q()(row, col);
        Q_new(row, col) = qr_new.Q()(row, col);
      }
    }

    // Step 1: M = Q_old.T * Q_new
    EigenMatrixType M = Q_old.transpose() * Q_new;

    // Step 2: M = USV^T (SVD decomposition)
    Eigen::JacobiSVD<EigenMatrixType> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    EigenMatrixType const& U = svd.matrixU();
    EigenVectorType const& S = svd.singularValues();
    EigenMatrixType const& V = svd.matrixV();

    // Step 3: theta = arccos(S)
    EigenVectorType theta(n);
    for (int i = 0; i < n; ++i) {
      // Clamp singular values to [0, 1] to avoid numerical issues
      real clampedS = std::clamp(S(i), 0_r, 1_r);
      theta(i) = std::acos(clampedS);
    }

    // Step 4: P = Q_new*V - Q_old*U*cos(theta)
    EigenMatrixType P = Q_new * V - Q_old * U * S.asDiagonal();

    // Step 5: K = P * diag(sin^-1(theta))
    EigenMatrixType sinInvTheta_diag = EigenMatrixType::Zero(n, n);
    for (int i = 0; i < n; ++i) {
      real sinTheta = std::sin(theta(i));
      // Avoid division by zero for small angles
      if (std::abs(sinTheta) > kDefaultNearEqualEpsilon<real>) {
        sinInvTheta_diag(i, i) = 1.0_r / sinTheta;
      }
    }
    EigenMatrixType K = P * sinInvTheta_diag;

    // Step 6: Q_t = Q_old*U*cos(theta*t) + K*sin(theta*t)
    EigenMatrixType cosTheta_t_diag = EigenMatrixType::Zero(n, n);
    EigenMatrixType sinTheta_t_diag = EigenMatrixType::Zero(n, n);
    for (int i = 0; i < n; ++i) {
      cosTheta_t_diag(i, i) = std::cos(theta(i) * strategy.onlineBasisParams.t);
      sinTheta_t_diag(i, i) = std::sin(theta(i) * strategy.onlineBasisParams.t);
    }

    EigenMatrixType Q_t = Q_old * U * cosTheta_t_diag + K * sinTheta_t_diag;

    // Step 7: Q_t = qr(Q_t) (QR decomposition to orthonormalize)
    Eigen::HouseholderQR<EigenMatrixType> qr(Q_t);
    EigenMatrixType Q_t_orthonormal = qr.householderQ() * EigenMatrixType::Identity(m, n);

    // Copy the interpolated basis back to proposedBasis
    for (int row = 0; row < m; ++row) {
      for (int col = 0; col < n; ++col) {
        proposedBasis(row, col) = Q_t_orthonormal(row, col);
      }
    }
  }
}
#endif // MOCHI_USE_EIGEN

static void ProposeNewLinearBasisNeuralAffineStrategy(
    ecs::RequiredTag<TagRomActor>,
    CNeuralAffineRomStrategy& strategy,
    CDisplacementSlice<real, TimeStep::Current> const& currentDisplacements,
    CRomLinearBasis& linearBasis,
    CTetrahedralMesh const& mesh,
    CRigidState<TimeStep::Current> const& rigidTransform,
    CMeshPivot const& pivot,
    CAuxiliaryPositionsForRomRigidTransform& rigidTransformAuxPositions) {
  MOCHI_PROFILE_SCOPE();

  // Get problem dimensions
  int const qDim = strategy.neuralModel.decoder.InputDim() - 3;
  int const outputDim = mesh.mesh->GetNumNodes() * 3;

  // Set up the proposed basis matrix
  auto& proposedBasis = linearBasis.GetAlternativeMatrix();
  proposedBasis.Resize(outputDim, qDim);

  // Compute the Jacobian in Basis frame using the appropriate projection method
  if (strategy.method == NeuralAffineRomMethod::CromEncoder) {
    ComputeNeuralAffineBasisViaEncoderProjection(
        strategy,
        currentDisplacements,
        mesh,
        rigidTransform,
        pivot,
        rigidTransformAuxPositions,
        proposedBasis);
  } else if (strategy.method == NeuralAffineRomMethod::CromProjection) {
    ComputeNeuralAffineBasisViaNonLinearLeastSquareProjection(
        strategy, currentDisplacements, mesh, rigidTransform, pivot, proposedBasis);
  } else if (strategy.method == NeuralAffineRomMethod::Interpolation) {
#if MOCHI_USE_EIGEN
    ComputeNeuralAffineBasisViaInterpolation(
        strategy,
        currentDisplacements,
        mesh,
        rigidTransform,
        pivot,
        rigidTransformAuxPositions,
        proposedBasis);
#else
    MOCHI_ASSERT(
        false,
        "NeuralAffineRomMethod::Interpolation method is not supported without Eigen. To enable it, include Eigen in your build setup and define MOCHI_USE_EIGEN=1");
#endif
  } else {
    MOCHI_ASSERT(false, "Unknown projection method for neural affine ROM");
  }

  // Mark the alternative basis as ready to use
  linearBasis.ShouldUseAlternativeBasis(true);
}

static void SelectActiveBasis(
    ecs::RequiredTag<TagRomActor>,
    CRomModeAmplitudes& currentAmpl,
    CRomLinearBasis& linearBasis,
    CRomCommonProperties& currentRomProps,
    CActorDofInfo& dofInfo,
    CRomJacobian& romJac,
    CReducedSparsityPattern& reducedSpPattern,
    CActorSnle& actorSnle) {
  MOCHI_PROFILE_SCOPE();

  // Create the new ModelProperties.
  int const baseDimBeforeBasisSwap = currentRomProps.value.baseDim;
  rom::ModelProperties newProps = currentRomProps.value;

  // If there is a transform, do not break it.
  int const currentTransformStateSize =
      currentRomProps.value.reducedPoseDim - currentRomProps.value.baseDim;
  int const currentTransformDerivativeSize =
      currentRomProps.value.reducedDofsDim - currentRomProps.value.baseDim;

  // Update the basis.
  linearBasis.UpdateBasis();

  // Set the new base and reduced dimensions.
  int const numModes = linearBasis.NumModes();
  newProps.baseDim = numModes;
  newProps.reducedPoseDim = numModes + currentTransformStateSize;
  newProps.reducedDofsDim = numModes + currentTransformDerivativeSize;
  currentRomProps.value = newProps;

  // Resize the ROM components if the number of modes has changed.
  bool const mustResizeRomStorage = (baseDimBeforeBasisSwap != newProps.baseDim);
  if (mustResizeRomStorage) {
    int const reducedPoseDim = currentRomProps.value.reducedPoseDim;
    int const reducedDofsDim = currentRomProps.value.reducedDofsDim;
    int const outputDim = currentRomProps.value.outputDim;

    currentAmpl.value.Resize(currentRomProps.value.baseDim);
    dofInfo.poseSize = reducedPoseDim;
    dofInfo.dofsSize = reducedDofsDim;
    reducedSpPattern.graph = MakeDenseSparsityGraph(reducedDofsDim, reducedDofsDim);
    romJac.value = CRomJacobian::DenseT::Zero(outputDim, reducedDofsDim);
    actorSnle.reducedResidual.Reset(ColumnVector<real>::Zero(reducedDofsDim));
    MOCHI_ASSERT(
        std::holds_alternative<Matrix<real>>(actorSnle.reducedDResidual),
        "Expected dense reduced dresidual.");
    std::get<Matrix<real>>(actorSnle.reducedDResidual)
        .Reset(Matrix<real>::Zero(reducedDofsDim, reducedDofsDim));
  }
}

static void PreStepStateUpdate(
    ecs::Included<TagRomActor>,
    CDisplacementSlice<real, TimeStep::Current> const& currDispl,
    CDisplacementSlice<real, TimeStep::Previous>& prevDispl,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CVelocitySlice<real, TimeStep::Previous>& prevVel,
    CRigidState<TimeStep::Current> const& currRigidTransform,
    CRigidState<TimeStep::Previous>& prevRigidTransform,
    CRomVelocity<real, TimeStep::Current>* currRomVelocity = nullptr,
    CRomVelocity<real, TimeStep::Previous>* prevRomVelocity = nullptr) {
  MOCHI_ASSERT(
      (currRomVelocity != nullptr) == (prevRomVelocity != nullptr),
      "Either both or none of the ROM velocity slices should be provided.");

  prevRigidTransform.value = currRigidTransform.value;
  prevDispl.CopyFrom(currDispl);
  prevVel.CopyFrom(currVel);
  currVel.value.SetZero();

  if (currRomVelocity) {
    prevRomVelocity->value = currRomVelocity->value;
    currRomVelocity->value.SetZero();
  }
}

void mochi::rom::PreStepPipeline(entt::registry& reg) {
  // Systems to propose the basis for the current step. If a new basis is proposed, it is stored as
  // the "alternative" basis in CRomLinearBasis and marked as ready to use.
  ecs::InvokeForEachGlobal(&ProposeNewLinearBasisContactForceInformedStrategy, reg);
  ecs::InvokeForEachGlobal(&ProposeNewLinearBasisNeuralAffineStrategy, reg);

  // Swap the basis and resize the ROM components (if needed).
  ecs::InvokeForEachGlobal(&SelectActiveBasis, reg);

  // Shift the state from 'Current' to 'Previous'.
  ecs::InvokeForEachGlobal(&PreStepStateUpdate, reg);
}

/**************************************************************************
  pre first stage
  ************************************************************************/

void mochi::rom::EntityPreFirstStage(
    ecs::RequiredTag<TagRomActor>,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::Previous> const& prevDispl,
    CVelocitySlice<real, TimeStep::Previous> const& prevFomVel,
    CIntegrationDisplacementSlices& intDispls,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intFomVels) {
  // Use integration utilities to compute displacement and velocity at TimeTarget::StepStart.
  integration::ApplyTimeIntegrationStepStart(intState, intDispls, prevDispl, intDispls.stepStart);
  integration::ApplyTimeIntegrationStepStart(
      intState, intFomVels, prevFomVel, intFomVels.stepStart);
}

/**************************************************************************
  pre stage
  ************************************************************************/

static void ComputeFomStateAtStageStart(
    ecs::RequiredTag<TagRomActor>,
    CTimeIntegratorState const& intState,
    CIntegrationDisplacementSlices& intFomDispls,
    CDisplacementSlice<real, TimeStep::StageStart>& stageStartFomDispl,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intFomVels,
    CVelocitySlice<real, TimeStep::StageStart>& stageStartFomVel) {
  // Use integration utilities to compute displacement and velocity at TimeTarget::StageStart.
  integration::ApplyTimeIntegration<TimeTarget::StageStart>(
      intState, intFomDispls, stageStartFomDispl);
  integration::ApplyTimeIntegration<TimeTarget::StageStart>(intState, intFomVels, stageStartFomVel);
}

static void PreStagePrepareState(
    ecs::RequiredTag<TagRomActor>,
    ecs::OptionalTag<TagRomActorFixRigidTransformInSolve> isRigidTransformFixedInSolve,
    CRomModeAmplitudes& currentAmpl,
    CRomShiftVector& shift,
    CRigidState<TimeStep::Current>& currRigidTransform,
    CDisplacementSlice<real, TimeStep::StageStart> const& fomDispl,
    CLocal2GlobalMap const& l2g,
    CRigidTransformEvalPoint const& evalPoint,
    CTetrahedralMesh const& mesh,
    CMeshPivot const& pivot,
    CAuxiliaryPositionsForRomRigidTransform& rigidTransformAuxPositions,
    CRigidState<TimeStep::StageStart>* stageStartRigidState = nullptr) {
  MOCHI_ASSERT_VERBOSE(mesh.mesh, "Missing mesh.");

  // (1) Use the *stage start* FOM displ to evaluate the rigid transform.
  auto& referencePivotFromDeformedPivot = currRigidTransform.value;
  TransformRT referencePivotFromWorld(Quaternion::Identity(), -pivot.position);
  TransformRT worldFromDeformedPivot;
  soft::ComputeTransformAtEvalPoint(fomDispl.value, l2g, evalPoint, worldFromDeformedPivot);
  referencePivotFromDeformedPivot = referencePivotFromWorld * worldFromDeformedPivot;

  if (!isRigidTransformFixedInSolve) {
    MOCHI_ASSERT(stageStartRigidState, "Missing CRigidState<TimeStep::StageStart>.");
    // set the stage start RT equal to that
    stageStartRigidState->value = currRigidTransform.value;
  }

  // (2) Set the ROM affine shift equal to the stage start displacements *without* the rigid
  // transform contribution.
  shift.value = fomDispl.value;
  rom::rigid_transform::TransformDisplacements(
      *mesh.mesh,
      /* activeNodes */ Span<int const>{}, // Need to consider all nodes (not just the active ones)
      pivot.position,
      Invert(currRigidTransform.value),
      rigidTransformAuxPositions.data,
      shift.value);

  // (3) Since the affine shift was updated, we need to set the ROM amplitudes to zero so that the
  // reconstruction of the FOM displacements is consistent.
  currentAmpl.value.SetZero();
}

void mochi::rom::PreStagePipeline(entt::registry& reg, Span<entt::entity const> entities) {
  if (entities.empty()) {
    return;
  }
  MOCHI_PROFILE_SCOPE();
  // IMPORTANT: below the order matters because (c) depends on (b) which depends on (a)

  ecs::InvokeForEach(&ComputeFomStateAtStageStart, reg, entities); // (a)
  ecs::InvokeForEach(&PreStagePrepareState, reg, entities); // (b)
  ResolveDisplacementAndJacobianOnActiveNodes(reg, entities); // (c)
}

/**************************************************************************
  post new solution
  ************************************************************************/

static void EntitySetSolutionWrapper(
    ColumnVectorView<real const> solution,
    ecs::Included<TagSoftActor, TagRomActor>,
    ecs::OptionalTag<TagRomActorFixRigidTransformInSolve> isRigidTransformFixedInSolve,
    CDofOffset const& dofOffset,
    CRomCommonProperties const& props,
    CRomModeAmplitudes& currAmplitudes,
    CRigidState<TimeStep::Current>& rigidTransform) {
  auto solutionVec = solution.MiddleRows(dofOffset.poseOffset, props.value.reducedPoseDim);
  EntitySetSolution(solutionVec, {}, isRigidTransformFixedInSolve, currAmplitudes, rigidTransform);
}

static void EntitySetIncrementWrapper(
    ColumnVectorView<real const> reference,
    ColumnVectorView<real const> increment,
    ecs::Included<TagSoftActor, TagRomActor>,
    ecs::OptionalTag<TagRomActorFixRigidTransformInSolve> isRigidTransformFixedInSolve,
    CDofOffset const& dofOffset,
    CRomCommonProperties const& props,
    CRomModeAmplitudes& currAmplitudes,
    CRigidState<TimeStep::Current>& rigidTransform) {
  auto referenceVec = reference.MiddleRows(dofOffset.poseOffset, props.value.reducedPoseDim);
  auto incrementVec = increment.MiddleRows(dofOffset.dofsOffset, props.value.reducedDofsDim);
  EntitySetIncrement(
      referenceVec, incrementVec, {}, isRigidTransformFixedInSolve, currAmplitudes, rigidTransform);
}

MOCHI_API void mochi::rom::UpdateCurrentRomVelocity(
    ecs::Included<TagSoftActor, TagRomActor>,
    ecs::OptionalTag<TagRomActorFixRigidTransformInSolve> isRigidTransformFixedInSolve,
    CRomCommonProperties const& /*props*/,
    CTimeIntegratorState const& intState,
    CRomModeAmplitudes const& currAmplitudes,
    CRigidState<TimeStep::Current> const* /*currRigidTransform*/,
    CRigidState<TimeStep::StageStart> const* /*stageStartRigidTransform*/,
    CRomVelocity<real, TimeStep::Current>& outVelocity) {
  MOCHI_PROFILE_SCOPE();

  // FRIZZI: TODO: fix this for deep flow, which i cannot run yet

  if (isRigidTransformFixedInSolve) {
    outVelocity.value = currAmplitudes.value;
    outVelocity.value /= intState.dtStage;
  } else {
    MOCHI_ASSERT(false);
  }

  // ColumnVector<real> currReducedState(props.value.reducedDim);
  // ColumnVector<real> stageStartReducedState(props.value.reducedDim);

  // auto stageStartAmplitudes = ColumnVector<real>::Zero(props.value.reducedDim);

  //// Resolve the current and stage-start states and take finite differences in time
  // EntityGetSolution<TimeStep::Current>(
  //     currReducedState, {}, RTFixedDuringSolve, currAmplitudes, currRigidTransform);
  // EntityGetSolution<TimeStep::StageStart>(
  //     stageStartReducedState, {}, RTFixedDuringSolve, stageStartAmplitudes,
  //     stageStartRigidTransform);

  // outVelocity.value = (currReducedState - stageStartReducedState);
  // outVelocity.value /= intState.dtStage;
}

static void PostNewSolutionPipelineImpl(entt::registry& reg, Span<entt::entity const> entities) {
  // Only need to resolve on the active nodes.
  ResolveDisplacementAndJacobianOnActiveNodes(reg, entities);

  // Update the current velocity of the ROM in reduced coordinates
  ecs::InvokeForEach(&UpdateCurrentRomVelocity, reg, entities);
}

void mochi::rom::PostNewSolutionPipeline(
    entt::registry& reg,
    Span<entt::entity const> entities,
    ColumnVectorView<real const> solution) {
  if (entities.empty()) {
    return;
  }
  MOCHI_PROFILE_SCOPE();

  // Copy over results from the solution vector of the non-linear problem.
  ecs::InvokeForEach(&EntitySetSolutionWrapper, reg, entities, solution);

  PostNewSolutionPipelineImpl(reg, entities);
}

void mochi::rom::PostNewIncrementPipeline(
    entt::registry& reg,
    Span<entt::entity const> entities,
    ColumnVectorView<real const> reference,
    ColumnVectorView<real const> increment) {
  if (entities.empty()) {
    return;
  }
  MOCHI_PROFILE_SCOPE();

  // Copy over results from the solution vector of the non-linear problem.
  ecs::InvokeForEach(&EntitySetIncrementWrapper, reg, entities, reference, increment);

  PostNewSolutionPipelineImpl(reg, entities);
}

/**************************************************************************
  post stage and post last stage
  ************************************************************************/

static void ComputeCurrentFomVelocityAndPushCurrentStateToIntegrationStages(
    ecs::Included<TagRomActor>,
    CTimeIntegratorState const& intState,
    CIntegrationDisplacementSlices& intFomDispls,
    CDisplacementSlice<real, TimeStep::StageStart> const& stageStartDispl,
    CDisplacementSlice<real, TimeStep::Current> const& currDispl,
    CVelocitySlice<real, TimeStep::Current>& currFomVel,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intFomVels) {
  MOCHI_PROFILE_SCOPE();

  currFomVel.value = (currDispl.value - stageStartDispl.value) * (1_r / intState.dtStage);
  intFomDispls.stages[intState.currentStage].value = currDispl.value;
  intFomVels.stages[intState.currentStage].value = currFomVel.value;
}

void mochi::rom::PostStagePipeline(entt::registry& reg, Span<entt::entity const> entities) {
  if (entities.empty()) {
    return;
  }
  MOCHI_PROFILE_SCOPE();

  // Resolve displacement on the full mesh. This is required so that the FOM displacements and
  // velocities at StepEnd are correct in all nodes, which is in turn required for (at least)
  // rendering and multi-step time integration with dynamic hyper-reduction.
  ResolveDisplacementOnAllNodes(reg, entities);

  ecs::InvokeForEach(
      &ComputeCurrentFomVelocityAndPushCurrentStateToIntegrationStages, reg, entities);
}

void mochi::rom::EntityPostLastStage(
    ecs::RequiredTag<TagRomActor>,
    CTimeIntegratorState const& intState,
    CIntegrationDisplacementSlices& intFomDispls,
    CDisplacementSlice<real, TimeStep::Current>& currFomDispl,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intFomVels,
    CVelocitySlice<real, TimeStep::Current>& currFomVel) {
  // Use integration utilities to compute displacement and velocity at TimeTarget::StepEnd.
  integration::ApplyTimeIntegration<TimeTarget::StepEnd>(intState, intFomDispls, currFomDispl);
  integration::ApplyTimeIntegration<TimeTarget::StepEnd>(intState, intFomVels, currFomVel);
}

/**************************************************************************
  Swapping active elements
*/

static void WarnIfUnderdetermined(
    CRomCommonProperties const& romProps,
    CActiveUniqueNodes const& activeUniqueNodes) {
  if (activeUniqueNodes.Count() * kSpaceDim3 < romProps.value.reducedDofsDim) {
    MOCHI_LOG_WARNING(
        "Underdetermined ROM problem after mesh swapping: Number of active node DoFs < ROM DoFs.");
  }
}

static void ClosestDistanceBySamplePoint(
    CActiveCollisions<ContactType::Async, TimeStep::Current> const& activeCollisionsAsync,
    CActiveCollisions<ContactType::Sync, TimeStep::Current> const& activeCollisionsSync,
    int numSamples,
    DynamicArray<real>& outDistances) {
  MOCHI_PROFILE_SCOPE();
  outDistances.clear();
  outDistances.resize(numSamples, std::numeric_limits<real>::infinity());

  auto processActiveCollisionFunc = [&](ActiveCollision const& collisions) {
    for (int i = 0; i < isize(collisions.collisionResult.sampleIndices); ++i) {
      auto sampleIdx = collisions.collisionResult.sampleIndices[i];
      MOCHI_ASSERT_VERBOSE(sampleIdx >= 0 && sampleIdx < outDistances.size());
      real distance = collisions.collisionResult.sdfInfo.val[i];
      outDistances[sampleIdx] = Min(outDistances[sampleIdx], distance);
    }
  };
  for (auto const& collisions : activeCollisionsAsync) {
    processActiveCollisionFunc(collisions);
  }
  for (auto const& collisions : activeCollisionsSync) {
    processActiveCollisionFunc(collisions);
  }
}

static void SwapActiveElementsBshStrategy(
    ecs::RequiredTag<TagRomActor>,
    CRomCommonProperties const& romProps,
    CDynamicSampleMeshStrategy<DynamicSampleMeshBsh> const& strategy,
    CTetrahedralMesh const& tetMesh,
    CActiveVolumeElements const& activeVolElements,
    CActiveBoundaryFaces& activeBoundaryFaces,
    CActiveUniqueNodes& activeUniqueNodes,
    CFemBoundaryDiscretization const& femBoundaryDiscC,
    CDisplacementSlice<real, TimeStep::Current> const& displacements,
    CRootTransform const& transform,
    ecs::CtxGlobal<CSceneTime const> time,
    CActiveCollisions<ContactType::Async, TimeStep::Current> const& activeCollisionsAsync,
    CActiveCollisions<ContactType::Sync, TimeStep::Current> const& activeCollisionsSync,
    CRequiresFarSdfEvaluation const& farSdfEval,
    rom::hyper::CDynamicSampleMeshBshManager& manager) {
  MOCHI_PROFILE_SCOPE();
  // We need a one point quadrature element
  if (!femBoundaryDiscC.Is<CFemBoundaryDiscretizationP1Q1_1>()) {
    MOCHI_LOG_WARNING( // TODO(T224856535)
        "Dynamic BSH Hyper-Reduction: FEM boundary discretization is not P1Q1_1. "
        "This is incompatible with dynamic BSH-based hyper-reduction. "
        "Please pass P1Q1 as the desired boundary element to SoftActorParams "
        "upon actor creation. Dynamic BSH Hyper-Reduction is disabled.");
    return;
  }

  // Compute the distance to the closest collider for each sample point.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 2048 * (sizeof(real) + sizeof(Real3)));
  DynamicArray<real> distanceBySample(&allocator);
  ClosestDistanceBySamplePoint(
      activeCollisionsAsync,
      activeCollisionsSync,
      femBoundaryDiscC.GetNumQuadPoints(),
      distanceBySample);

  // Compute world space positions of the surface mesh points.
  DynamicArray<Real3> faceBarycenters = [&tetMesh, &displacements, &transform, &allocator]() {
    MOCHI_PROFILE_SCOPE_N("Compute Face Barycenters");
    auto boundaryFaces = tetMesh.mesh->GetBoundaryFacesConnectivity();
    auto coords = tetMesh.mesh->GetNodeCoordinates();

    auto positionAt = [&](int i) -> Real3 {
      Real3 point = coords[i] +
          Real3{
              displacements.value[3 * i + 0],
              displacements.value[3 * i + 1],
              displacements.value[3 * i + 2]};
      return transform.worldFromLocal.TransformPoint(point);
    };

    DynamicArray<Real3> result(&allocator);
    result.reserve(boundaryFaces.size());
    std::transform(
        boundaryFaces.begin(),
        boundaryFaces.end(),
        std::back_inserter(result),
        [&](Int3 const& face) -> Real3 {
          auto p1 = positionAt(face[0]);
          auto p2 = positionAt(face[1]);
          auto p3 = positionAt(face[2]);
          return (p1 + p2 + p3) / 3_r;
        });
    return result;
  }();

  // Update the SDF lower bound and the radii of the BSH.
  // NOTE: It may be possible to combine Update and ComputeActiveBoundaryFaces to avoid updating
  // the position of children that are provably inactive.
  manager.value.Update(
      faceBarycenters, // TODO(T224856535): Assumes one sample per boundary face.
      activeBoundaryFaces.ViewIndices(),
      distanceBySample,
      time->StepEndTime(),
      manager.maxColliderVelocity,
      farSdfEval.maxDistance);

  // Compute the new active boundary faces.
  auto const maxActiveSamples = static_cast<int>(
      Floor(strategy.strategy.maxSubsamplingDensity * femBoundaryDiscC.GetNumQuadPoints()));

  manager.value.ComputeActiveBoundaryFaces(
      maxActiveSamples, strategy.strategy.sampleActivationDistance);

  femBoundaryDiscC.Visit([&](auto const& femBoundaryDisc) {
    activeBoundaryFaces.Recompute(
        manager.value.ActiveBoundaryFaceIndices(),
        MakeConstSpan(femBoundaryDisc.femElements),
        manager.value.ActiveBoundaryFaceWeightMultipliers());
  });

  // recompute the active nodes
  activeUniqueNodes.Recompute(activeVolElements, activeBoundaryFaces);
  WarnIfUnderdetermined(romProps, activeUniqueNodes);
}

void mochi::rom::SwapActiveElements(entt::registry& reg, entt::entity e) {
  MOCHI_PROFILE_SCOPE();

  mochi::ecs::TryInvokeOnEntity(&SwapActiveElementsBshStrategy, reg, e);
}
