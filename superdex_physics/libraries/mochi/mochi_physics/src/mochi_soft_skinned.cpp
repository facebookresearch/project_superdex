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

#include "mochi_soft_skinned.h"

#include "mochi_common_components.h"
#include "mochi_contact_filter.h"
#include "mochi_integration.h"
#include "mochi_island.h"
#include "mochi_rom_jacobian.h"
#include "mochi_soft_rom_components.h"

#include <mochi_core/contact/dmap.h>
#include <mochi_core/rom/rom_hyper_reduction.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/sparsity_utils.h>
#include <mochi_core/utils/task_scheduler.h>

#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace mochi;
using namespace mochi::skinned;
using namespace mochi::dmap;

static constexpr int kSoftIdx = CSkinnedInteractionSnle::kSoftIdx;
static constexpr int kArticulatedIdx = CSkinnedInteractionSnle::kArticulatedIdx;

void skinned::EntityAssembleBody(
    AssemblyParams const& params, // external parameter
    TaskSemaphore softAssemblySem,
    TaskSemaphore artAssemblySem,
    entt::registry& reg,
    ecs::CtxGlobal<CSceneGravity const> sceneGravity,
    ecs::OptionalTag<TagUseGravity> hasGravityTag,
    ecs::OptionalTag<TagUseInertia> hasInertiaTag,
    ecs::OptionalTag<TagUseStress> hasStressTag,
    ecs::OptionalTag<TagRomActor> isRom,
    CSkinnedEnergy const& skinnedEnergy,
    CLocal2GlobalMap const& l2g,
    CNodalBasedStructure const& nbs,
    CFemVolumeDiscretizationP1Q1 const& femLowVolDisc,
    CFemVolumeDiscretizationP1Q4 const& femHighVolDisc,
    CSoftMaterialParams const& materialParams,
    CMassMatrix const& massMatrix,
    CDofOffset const& dofOffset,
    CActiveVolumeElements const* activeVolElems,
    CSkinnedComposition const& composition,
    CTimeIntegratorState const& intState,
    CArticulatedSkinningData const& skinningData,
    CSoftSkinnedJacobianDynamic const& jacobianSoft,
    CDisplacementSlice<real, TimeStep::Current, DisplacementLayer::Skinned> const& currDispl,
    CDisplacementSlice<real, TimeStep::StageStart, DisplacementLayer::Skinned> const&
        stageStartDispl,
    CVelocitySlice<real, TimeStep::StageStart, DisplacementLayer::Skinned> const* stageStartVel,
    [[maybe_unused]] CRomProjectionStrategy const* romProjectionStrategy,
    CSoftSkinnedAssemblyScratchMemory& scratchMemory,
    CActorSnle& outSoftSnle,
    CSoftSkinnedUnposedSnle& outSoftSkinnedSnle,
    CSkinnedInteractionSnle& outInteractionSnle) {
  MOCHI_PROFILE_SCOPE();

  bool const hasPosedGravity = hasGravityTag && skinnedEnergy.gravity;
  bool const hasPosedInertia = hasInertiaTag && skinnedEnergy.inertia;
  bool const hasPosedStress = hasStressTag && skinnedEnergy.stress;
  bool const hasUnposedInertia = hasInertiaTag && !skinnedEnergy.inertia;
  bool const hasUnposedStress = hasStressTag && !skinnedEnergy.stress;
  if (!hasPosedGravity && !hasPosedInertia && !hasPosedStress) {
    return;
  }

  MOCHI_ASSERT(!hasPosedInertia || stageStartVel, "Expected stage-start velocity.");
  MOCHI_ASSERT(!isRom || romProjectionStrategy, "Missing ROM projection strategy.");
  MOCHI_ASSERT(
      !isRom ||
          romProjectionStrategy->value == experimental::RomProjectionStrategy::ActorLevelProjection,
      "The dedicated assembly logic for soft skinned actors requires actor-level projection for ROMs.");

  // Perform the assembly of energy terms on the soft-actor level
  soft::AssembleBodyImpl(
      params,
      sceneGravity,
      hasPosedGravity,
      hasPosedInertia,
      hasPosedStress,
      l2g,
      nbs,
      femLowVolDisc,
      femHighVolDisc,
      TransformRT::Identity(),
      materialParams,
      intState,
      currDispl.value,
      stageStartDispl.value,
      stageStartVel ? stageStartVel->value : ColumnVector<real>{},
      massMatrix,
      outSoftSkinnedSnle,
      activeVolElems);

  // Transform the terms to the soft skinned actor
  auto softOffset = dofOffset.dofsOffset;
  auto articulated = composition.articulated;
  auto artOffset = reg.get<CDofOffset const>(articulated).dofsOffset;
  auto const& Jart = skinningData.jacobianDJoints;

  if (params.assemObj) {
    outInteractionSnle.objective = outSoftSkinnedSnle.objective;
  }

  if (params.assemRes) {
    MOCHI_PROFILE_SCOPE_N("Project soft skinned Residual");
    MOCHI_ASSERT(outInteractionSnle.residuals.size() == 2, "Expected 2 residuals.");

    // Transform the soft-actor residual by the skinning Jacobian
    outInteractionSnle.residuals[kSoftIdx].first = softOffset;
    auto& outSoftResidual = outInteractionSnle.residuals[kSoftIdx].second;
    jacobianSoft.TransposeMultiply(
        AsConstView(outSoftSkinnedSnle.fullResidual), AsView(outSoftResidual));

    outInteractionSnle.residuals[kArticulatedIdx].first = artOffset;
    auto& outArtResidual = outInteractionSnle.residuals[kArticulatedIdx].second;
    outArtResidual = Jart.Transpose() * outSoftSkinnedSnle.fullResidual;
  }

  if (params.assemDRes) {
    MOCHI_PROFILE_SCOPE_N("Project soft skinned DResidual");
    // Transform the soft actor dresidual by the skinning Jacobian.
    // Notation:
    //   Jsoft, Jart = Jacobians of the soft and articulated actors.
    //   A = dresidual of the soft actor.
    // Comments:
    // - There are 2 diagonal submatrices (JsoftT_A_Jsoft, JartT_A_Jart) and 2 off-diagonal
    //   submatrices (JsoftT_A_Jart, JartT_A_Jsoft). For performance reasons, the diagonal
    //   submatrices are assembled into the soft and articulated dresiduals, and the off-diagonal
    //   submatrices into the soft skinned dresidual.
    // - The off-diagonal submatrix JsoftT_A_Jart is computed as JsoftT * (A * Jart). This order of
    //   operations requires fewer FLOPs than (JsoftT * A) * Jart if artDofs < "average number of
    //   non-zeros per row in the FOM dresidual" (for FOMs) or if artDofs < romDofs (for ROMs).
    //   Otherwise, (JsoftT * A) * Jart would require fewer FLOPs.
    // - The off-diagonal submatrix JartT_A_Jsoft is computed as (JsoftT_A_Jart)^T.
    MOCHI_ASSERT_VERBOSE(
        (std::holds_alternative<BlockSparseMatrix<real, 3>>(outSoftSkinnedSnle.fullDResidual)),
        "Expected block sparse dresidual.");
    auto const& A = std::get<BlockSparseMatrix<real, 3>>(outSoftSkinnedSnle.fullDResidual);

    MOCHI_ASSERT(outInteractionSnle.dresiduals.size() == 2, "Expected 2 dresiduals.");
    MOCHI_ASSERT(
        (std::holds_alternative<Matrix<real>>(outInteractionSnle.dresiduals[kSoftIdx].matrix) &&
         std::holds_alternative<Matrix<real>>(
             outInteractionSnle.dresiduals[kArticulatedIdx].matrix)),
        "Expected dense storage.");
    outInteractionSnle.dresiduals[kSoftIdx].rowOffset = softOffset;
    outInteractionSnle.dresiduals[kSoftIdx].colOffset = artOffset;
    auto& JsoftT_A_Jart = std::get<Matrix<real>>(outInteractionSnle.dresiduals[kSoftIdx].matrix);
    outInteractionSnle.dresiduals[kArticulatedIdx].rowOffset = artOffset;
    outInteractionSnle.dresiduals[kArticulatedIdx].colOffset = softOffset;
    auto& JartT_A_Jsoft =
        std::get<Matrix<real>>(outInteractionSnle.dresiduals[kArticulatedIdx].matrix);

    TaskSemaphore sem;
    Schedule(sem, "SoftSkinned: Assemble Soft Matrix", [&]() {
      bool const addToDres = hasUnposedInertia || hasUnposedStress;
      if (!isRom) {
        auto& JsoftT_A_Jsoft = std::get<BlockSparseMatrix<real, 3>>(outSoftSnle.fullDResidual);
        if (addToDres) {
          softAssemblySem.Wait(); // Wait for soft assembly before modifying its dresidual.
          jacobianSoft.FrontTransposeAndBackMultiply<true>(A, JsoftT_A_Jsoft);
        } else {
          jacobianSoft.FrontTransposeAndBackMultiply<false>(A, JsoftT_A_Jsoft);
        }

      } else {
        MOCHI_ASSERT_VERBOSE(outSoftSnle.UseReduced(), "Reduced SNLE must be enabled.");
        auto& JsoftT_A_Jsoft = std::get<Matrix<real>>(outSoftSnle.reducedDResidual);
        if (addToDres) {
          softAssemblySem.Wait(); // Wait for soft assembly before modifying its dresidual.
          jacobianSoft.FrontTransposeAndBackMultiply<true>(
              A, scratchMemory.A_Jsoft, JsoftT_A_Jsoft);
        } else {
          jacobianSoft.FrontTransposeAndBackMultiply<false>(
              A, scratchMemory.A_Jsoft, JsoftT_A_Jsoft);
        }
      }
    });

    scratchMemory.A_Jart = A * Jart;
    Schedule(sem, "SoftSkinned: Assemble Interaction Matrices", [&]() mutable {
      // Store result in JartT_A_Jsoft.Transpose() (row-major) instead of JsoftT_A_Jart (col-major)
      // to improve write memory access.
      jacobianSoft.TransposeMultiply(AsConstView(scratchMemory.A_Jart), JartT_A_Jsoft.Transpose());
      JsoftT_A_Jart = JartT_A_Jsoft.Transpose(); // TODO: Could be parallelized.
    });

    {
      MOCHI_PROFILE_SCOPE_N("SoftSkinned: Assemble Articulated Matrix");
      MOCHI_ASSERT_VERBOSE(
          reg.get<CActorSnle>(articulated).UseReduced(), "Reduced SNLE must be enabled.");
      auto& JartT_A_Jart =
          std::get<Matrix<real>>(reg.get<CActorSnle>(articulated).reducedDResidual);
      artAssemblySem.Wait(); // Wait for articulated assembly before modifying its dresidual.
      JartT_A_Jart += Jart.Transpose() * scratchMemory.A_Jart;
    }

    sem.Wait();
  }
}

static void
InitSoftSkinnedMesh(entt::registry& reg, entt::entity e, bool useContact, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_PROFILE_SCOPE();

  auto articulated = reg.get<CSkinnedComposition const>(e).articulated;

  // Init skinning components
  auto shapePtr =
      std::dynamic_pointer_cast<TetrahedralMeshShape const>(reg.get<CShape const>(e).shape);
  InitSkinnedMesh(reg, e, articulated, shapePtr, error);
  MOCHI_ERROR_RETURN(error);

  // Emplace pre-skinning position component
  int const numDofs = 3 * shapePtr->GetMesh()->GetNumNodes();
  reg.emplace<CNodePositions>(e, numDofs);

  // If not colliding, we're done
  if (!useContact) {
    return;
  }

  MOCHI_ASSERT(
      reg.all_of<CFemBoundaryDiscretization>(e),
      "Soft skinned actor with contact must have a boundary discretization");
  int const numCollidingSamples = reg.get<CFemBoundaryDiscretization const>(e).GetNumQuadPoints();

  // Initialize components for colliding-actor role.
  auto strategy = ContactPartitionStrategy::SkinningDofGroups;
  auto const& contactParams = reg.get<CContactParams>(e);
  InitCollidingSkinMesh(
      reg,
      e,
      articulated,
      *shapePtr,
      MakeSingletonConstSpan(strategy),
      contactParams,
      numCollidingSamples);
}

static VariantJacobian::MatrixBlockDiag CreateSkinningJacobianRest(int nNodes) {
  VariantJacobian::MatrixBlockDiag result(3 * nNodes, 3);
  Matrix<real, 3, 3> eye;
  eye.SetIdentity();
  for (int i = 0; i < nNodes; i++) {
    result.Block<3, 3>(3 * i, 0, 3, 3) = eye;
  }
  return result;
}

// Emplace data to process energy terms.
static void
InitEnergyTerms(entt::registry& reg, entt::entity e, SoftSkinnedActorParams const& params) {
  MOCHI_PROFILE_SCOPE();

  // Get the size of the actor's mesh.
  auto const* tetMesh = reg.get<CTetrahedralMesh const>(e).mesh.get();
  int numNodes = tetMesh->GetNumNodes();

  // Velocity for inertial term
  if (params.hasInertia) {
    reg.emplace<CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned>>(
        e, 3 * numNodes);
    reg.emplace<CVelocitySlice<real, TimeStep::Previous, DisplacementLayer::Skinned>>(
        e, 3 * numNodes);
    reg.emplace<CVelocitySlice<real, TimeStep::StageStart, DisplacementLayer::Skinned>>(
        e, 3 * numNodes);
    reg.emplace<CIntegrationVelocitySlices<DisplacementLayer::Skinned>>(e, 3 * numNodes);
  }

  // Tags
  auto& skinnedEnergy = reg.get<CSkinnedEnergy>(e);
  if (params.hasGravity) {
    reg.emplace<TagUseGravity>(e);
    skinnedEnergy.gravity = true;
  }
  if (params.hasInertia) {
    reg.emplace<TagUseInertia>(e);
    skinnedEnergy.inertia = true;
  }
  if (params.hasStress) {
    reg.emplace<TagUseStress>(e);
    skinnedEnergy.stress = true;
    // Modify the displacement-slice reference for elasticity computations.
    auto const& dispCurr =
        reg.get<CDisplacementSlice<real, TimeStep::Current, DisplacementLayer::Skinned> const>(e);
    reg.emplace_or_replace<CStressDisplacementRef>(e, dispCurr.value);
  }

  // SNLE data for the soft actor body assembly.
  AnyMatrix<real> softDres = reg.get<CActorSnle const>(e).fullDResidual; // Owning copy.
  auto& softSnle = reg.emplace<CSoftSkinnedUnposedSnle>(e, std::move(softDres));
  softSnle.useInSolver = false;

  // SNLE data for the soft actor terms transformed to the full dofs.
  int softDofs = reg.get<CActorDofInfo const>(e).dofsSize;
  auto articulated = reg.get<CSkinnedComposition const>(e).articulated;
  int artDofs = reg.get<CActorDofInfo const>(articulated).dofsSize;
  auto& actorSnle = reg.emplace<CSkinnedInteractionSnle>(e);
  actorSnle.residuals.resize(2);
  actorSnle.residuals[kSoftIdx].second = ColumnVector<real>::Zero(softDofs);
  actorSnle.residuals[kArticulatedIdx].second = ColumnVector<real>::Zero(artDofs);

  // Offsets will be populated in skinned::EntityAssembleBody based on island composition.
  int constexpr kDummyOffset = 0;
  auto dresSoftArt = Matrix<real>::Zero(softDofs, artDofs);
  auto dresArtSoft = Matrix<real>::Zero(artDofs, softDofs);
  auto dresSoftArtView = AsConstView(dresSoftArt);
  auto dresArtSoftView = AsConstView(dresArtSoft);
  static_assert(
      kSoftIdx == 0 && kArticulatedIdx == 1, "Please update order of emplace_back's below");
  actorSnle.dresiduals.emplace_back(
      kDummyOffset,
      kDummyOffset,
      std::move(dresSoftArt), // Does not invalidate dresSoftArtView
      dresArtSoftView);
  actorSnle.dresiduals.emplace_back(
      kDummyOffset,
      kDummyOffset,
      std::move(dresArtSoft), // Does not invalidate dresArtSoftView
      dresSoftArtView);

  // Emplace Jacobians for the soft actor mesh.
  auto& softJacobianRest = reg.emplace<CSoftSkinnedJacobianRest>(e);
  softJacobianRest.value = CreateSkinningJacobianRest(numNodes);
  auto const* romJacobian = reg.try_get<rom::CRomJacobian const>(e);
  auto numRows = 3 * numNodes;
  auto numCols = romJacobian ? romJacobian->Cols() : 3;
  bool useBlockDiagForSoftJacobian = romJacobian == nullptr;
  auto const& softJacobianDynamic =
      reg.emplace<CSoftSkinnedJacobianDynamic>(e, useBlockDiagForSoftJacobian, numRows, numCols);

  // Emplace scratch memory for the soft actor assembly.
  auto& scratchMemory = reg.emplace<CSoftSkinnedAssemblyScratchMemory>(e);
  auto const& softSnleDResBsp = std::get<BlockSparseMatrix<real, 3>>(softSnle.fullDResidual);
  auto const& jacobianArt = reg.get<CArticulatedSkinningData const>(e).jacobianDJoints;
  scratchMemory.A_Jart.Resize(softSnleDResBsp.Rows(), jacobianArt.Cols());
  if (!softJacobianDynamic.IsBlockDiagonal()) {
    // A_Jsoft is only needed for ROMs.
    scratchMemory.A_Jsoft.Resize(softSnleDResBsp.Rows(), softJacobianDynamic.Cols());
  }
}

/*
 * System to update the displacements of a skinned surface.
 */
template <TimeStep kStep, bool kForceUseAllNodes = false>
static void ResolveSkinning(
    ecs::PartialRegistry<CArticulatedLinkTransforms<kStep> const> reg,
    CArticulatedSkinningData const& skinningData,
    CSkinnedComposition const& composition,
    CActiveUniqueNodes const* activeNodes,
    CDisplacementSlice<real, kStep> const& softDisp,
    CDisplacementSlice<real, kStep, DisplacementLayer::Skinned>& outDisplacements,
    CNodePositions& outPositions) {
  // Fetch link transforms
  auto const& linkTransforms =
      reg.template get<CArticulatedLinkTransforms<kStep> const>(composition.articulated);

  if (activeNodes && !kForceUseAllNodes) {
    auto rest3 = Unflatten<Real3 const>(MakeConstSpan(skinningData.restCoords));
    auto softDisp3 = Unflatten<Real3 const>(MakeConstSpan(softDisp.value));
    auto pos3 = Unflatten<Real3>(MakeSpan(outPositions.value));
    auto disp3 = Unflatten<Real3>(MakeSpan(outDisplacements.value));
    for (int node : activeNodes->ViewIds()) {
      pos3[node] = rest3[node] + softDisp3[node];
    }
    skinningData.skinningTransform.Transform(
        linkTransforms, outPositions.value, outDisplacements.value, activeNodes->ViewIds());
    for (int node : activeNodes->ViewIds()) {
      disp3[node] -= rest3[node];
    }
  } else {
    outPositions.value = skinningData.restCoords + softDisp.value;
    skinningData.skinningTransform.Transform(
        linkTransforms, outPositions.value, outDisplacements.value);
    outDisplacements.value -= skinningData.restCoords;
  }
}

void skinned::ResolveAllNodeSkinningDisplacementsPipeline(
    entt::registry& reg,
    Span<entt::entity const> entities) {
  MOCHI_PROFILE_SCOPE();
  ecs::InvokeForEach(
      &ResolveSkinning<TimeStep::Current, /* kForceUseAllNodes */ true>, reg, entities);
}

void skinned::InitSkinnedActor(
    entt::registry& reg,
    entt::entity e,
    SoftSkinnedActorParams const& params,
    bool useContact,
    ActorHandle articulatedHandle,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_PROFILE_SCOPE();

  MOCHI_ASSERT(reg.all_of<TagSoftActor const>(e), "Not a soft actor.");
  MOCHI_ASSERT(
      (reg.any_of<CDofPositionsBC, TagRomActorFixRigidTransformInSolve>(e)),
      "Soft skinned actor must have Dirichlet boundary conditions or it must be a ROM with no rigid dofs");
  MOCHI_ASSERT(
      !reg.all_of<CRecenteringParams>(e), "Soft skinned actor must have recentering disabled");

  MOCHI_ERROR_IF(
      reg.any_of<TagUseGravity>(e), error, "Soft skinned actor cannot use unposed gravity");
  MOCHI_ERROR_IF(
      reg.any_of<TagUseInertia>(e) && params.hasInertia,
      error,
      "Soft skinned actor cannot use both posed and unposed inertia");
  MOCHI_ERROR_IF(
      reg.any_of<TagUseStress>(e) && params.hasStress,
      error,
      "Soft skinned actor cannot use both posed and unposed stress");
  MOCHI_ERROR_RETURN(error);

  // Add composition component.
  auto& composition = reg.emplace<CSkinnedComposition>(e);
  auto articulated = GetEntity(reg, articulatedHandle, ErrorAssert{});
  MOCHI_ASSERT(reg.all_of<TagArticulatedActor const>(articulated), "Not an articulated actor.");
  composition.articulated = articulated;
  composition.articulatedHandle = articulatedHandle;

  // Add soft skinned actor tag
  reg.emplace<TagSoftSkinnedActor>(e);

  InitSoftSkinnedMesh(reg, e, useContact, error);
  MOCHI_ERROR_RETURN(error);

  // Initialize assembly of energy terms.
  if (params.hasGravity || params.hasInertia || params.hasStress) {
    InitEnergyTerms(reg, e, params);
  }

  // Resolve skinning so the skinned displacements reflect the actual skeleton pose rather than the
  // rest mesh.
  ResolveAllNodeSkinningDisplacementsPipeline(reg, MakeSingletonConstSpan(e));
}

static void ComputeCurrentVelocity(
    ecs::RequiredTag<TagSoftSkinnedActor>,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::StageStart, DisplacementLayer::Skinned> const&
        stageStartDispl,
    CDisplacementSlice<real, TimeStep::Current, DisplacementLayer::Skinned> const& currDispl,
    CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned>& currVel) {
  // Velocity is recovered via finite differences of the displacements at the beginning and at the
  // end of the stage.
  currVel.value = (currDispl.value - stageStartDispl.value) * (1_r / intState.dtStage);
}

template <int kNumDofsPerNode>
static void MultiplyJacobians(
    RowMatrixView<real const, krylov::kDynamic, kNumDofsPerNode> skinJacobian,
    rom::CRomJacobian const& romJacobian,
    RowMatrixView<real> result,
    Span<int const> activeNodes = {}) {
  MOCHI_PROFILE_SCOPE();
  static_assert(
      kNumDofsPerNode != krylov::kDynamic, "Implementation assumes compile-time number of columns");

  /*
   - romJacobian : [numRows x k]
   - skinJacobian: [numRows x 3]
   - result      : [numRows x k]

   When the activeNodes is empty, it means that ALL nodes are active so we must operate on ALL rows
   of skinJacobian, otherwise we need to use only those that correspond to the active nodes.
  */

  int const numRows = skinJacobian.Rows();
  MOCHI_ASSERT(
      (numRows % kNumDofsPerNode == 0) && result.Rows() == numRows,
      "Unexpected number of matrix rows");
  int const romJCols = romJacobian.Cols();
  MOCHI_ASSERT(result.Cols() == romJCols, "Unexpected number of matrix cols");

  // If there are active nodes, tailor things based on that, else we use all rows.
  int const numNodes = activeNodes.empty() ? numRows / kNumDofsPerNode : isize(activeNodes);

  // Compute the product in parallel. At least ~100000 FLOP per worker (50 μs @ 2 GFLOP/s).
  int const minNodesPerTask =
      Clamp(100000 / ((2 * kNumDofsPerNode - 1) * kNumDofsPerNode * romJCols), 1, numNodes);
  ParallelForRange(
      "MultiplyJacobians", 0, numNodes, minNodesPerTask, numNodes, [&](int loopBegin, int loopEnd) {
        romJacobian.Visit([&](auto const& J) {
          // Enforce J to be row-major so that the products below are "Row-Major = Row-Major x
          // Row-Major". Internally, they may be computed as the transposed product.
          static_assert(
              krylov::details::MatTraits<decltype(J)>::kMajorDir == krylov::Direction::RowMajor);

          for (int i = loopBegin; i < loopEnd; ++i) {
            // Suppose that we have a total of N nodes to act on, and that this task
            // is assigned the index range (p, p+1, p+2, p+3].
            // Given an integer, i, looping over (p, p+1, p+2, p+3],
            // we need to find the row, rowBegin, of skinJacobian that has
            // the block of data corresponding to the node identified by i.
            // The mapping from i -> node depends on whether we have active nodes or not.
            // - if ALL nodes are active, then the parallel_for tasks' ranges
            //   map 1:1 to the node range, so we have rowBegin = i*kNumDofsPerNode
            //
            // - if some nodes are active, then the parallel_for tasks' ranges
            //   map 1:1 to the *interval of active nodes*, so we must do
            //   rowBegin = activeNodes[i]*kNumDofsPerNode
            //
            int const rowBegin =
                activeNodes.empty() ? kNumDofsPerNode * i : kNumDofsPerNode * activeNodes[i];

            result.template MiddleRows<kNumDofsPerNode>(rowBegin, kNumDofsPerNode) =
                skinJacobian.template Block<kNumDofsPerNode, kNumDofsPerNode>(
                    rowBegin, 0, kNumDofsPerNode, kNumDofsPerNode) *
                J.template MiddleRows<kNumDofsPerNode>(rowBegin, kNumDofsPerNode);
          }
        });
      });
}

/*
 * System to update the Jacobian of a soft skinned mesh wrt bones and joints.
 */
static void ResolveJacobianDJoints(
    ecs::PartialRegistry<
        CArticulatedLinkTransforms<TimeStep::Current> const,
        CArticulatedJacobian const> reg,
    CSkinnedComposition const& composition,
    CNodePositions const& positions,
    CActiveUniqueNodes const* activeNodes,
    CArticulatedSkinningData& outSkinningData) {
  // Compute Jacobian DBones
  auto const& linkTransforms =
      reg.get<CArticulatedLinkTransforms<TimeStep::Current> const>(composition.articulated);
  articulated::compound::ResolveSkinningJacobianDBones(
      linkTransforms, positions.value, outSkinningData, activeNodes);

  // Compute Jacobian DJoints by chain rule with Jacobian DBones/DJoints
  auto const& articulatedJacobian = reg.get<CArticulatedJacobian const>(composition.articulated);
  static_assert(
      krylov::details::MatTraits<decltype(outSkinningData.jacobianDJoints)>::kMajorDir ==
      krylov::Direction::RowMajor); // Enforced for performance reasons.
  static_assert(
      krylov::details::MatTraits<decltype(articulatedJacobian.value)>::kMajorDir ==
      krylov::Direction::RowMajor); // Enforced for performance reasons.
  outSkinningData.jacobianDJoints = outSkinningData.jacobianDBones * articulatedJacobian.value;
}

/*
 * System to update the Jacobian of a soft skinned mesh wrt soft dofs.
 */
static void ResolveJacobianDSoft(
    ecs::PartialRegistry<CArticulatedLinkTransforms<TimeStep::Current> const> reg,
    CSkinnedComposition const& composition,
    CSoftSkinnedJacobianRest const& jacobianSoftRest,
    CArticulatedSkinningData const& skinningData,
    rom::CRomJacobian const* romJacobian,
    CActiveUniqueNodes const* activeNodes,
    CSoftSkinnedJacobianDynamic& outJacobianSoft) {
  // Compute Jacobian DSoft. Use an auxiliary matrix for the result if this is a ROM actor.
  auto articulated = composition.articulated;
  auto const& linkTransforms =
      reg.get<CArticulatedLinkTransforms<TimeStep::Current> const>(articulated);
  VariantJacobian::MatrixBlockDiagConstView inputView(jacobianSoftRest.value);
  VariantJacobian::MatrixBlockDiagView dsoftView;
  std::optional<VariantJacobian::MatrixBlockDiag> dsoftAux;
  if (romJacobian) {
    dsoftAux.emplace(jacobianSoftRest.value);
    dsoftView.Reset(dsoftAux.value());
  } else {
    dsoftView.Reset(outJacobianSoft.GetBlockDiagView());
  }

  if (activeNodes) {
    skinningData.skinningTransform.DTransform(
        linkTransforms, inputView, dsoftView, activeNodes->ViewIds());
  } else {
    skinningData.skinningTransform.DTransform(linkTransforms, inputView, dsoftView);
  }

  // If the actor is ROM, multiply DSoft by the ROM Jacobian
  if (romJacobian) {
    if (activeNodes) {
      MultiplyJacobians(
          AsConstView(dsoftView),
          *romJacobian,
          outJacobianSoft.GetDenseView(),
          activeNodes->ViewIds());
    } else {
      MultiplyJacobians(AsConstView(dsoftView), *romJacobian, outJacobianSoft.GetDenseView());
    }
  }
}

void skinned::EntityPreFirstStage(
    ecs::RequiredTag<TagSoftSkinnedActor>,
    CTimeIntegratorState const& intState,
    CVelocitySlice<real, TimeStep::Previous, DisplacementLayer::Skinned> const& prevVel,
    CIntegrationVelocitySlices<DisplacementLayer::Skinned>& intVels) {
  MOCHI_PROFILE_SCOPE();
  // Skinned velocities are treated as integration variables, which is akin to computing them via
  // finite differences.
  integration::ApplyTimeIntegrationStepStart(intState, intVels, prevVel, intVels.stepStart);
}

static void ComputeVelocityAtStageStart(
    ecs::RequiredTag<TagSoftSkinnedActor>,
    CTimeIntegratorState const& intState,
    CIntegrationVelocitySlices<DisplacementLayer::Skinned>& intVels,
    CVelocitySlice<real, TimeStep::StageStart, DisplacementLayer::Skinned>& stageStartVel) {
  // Skinned velocities are treated as integration variables, which is akin to computing them via
  // finite differences in each stage.
  integration::ApplyTimeIntegration<TimeTarget::StageStart>(intState, intVels, stageStartVel);
}

static void ComputeVelocityAtTimeStepEnd(
    ecs::RequiredTag<TagSoftSkinnedActor>,
    CTimeIntegratorState const& intState,
    CIntegrationVelocitySlices<DisplacementLayer::Skinned>& intVels,
    CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned>& currVel) {
  // Skinned velocities are treated as integration variables, which is akin to computing them via
  // finite differences in each stage.
  integration::ApplyTimeIntegration<TimeTarget::StepEnd>(intState, intVels, currVel);
}

void skinned::EntityIncrementStep(
    ecs::RequiredTag<TagSoftSkinnedActor>,
    CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned>& currVel,
    CVelocitySlice<real, TimeStep::Previous, DisplacementLayer::Skinned>& prevVel) {
  MOCHI_PROFILE_SCOPE();
  // Copy current velocity to previous velocity. Previous position and displacement components
  // neither exist nor need to be updated.
  prevVel.CopyFrom(currVel);
  currVel.value.SetZero(); // Effectively no-op. It will be recomputed before it's needed.
}

void skinned::PreStagePipeline(entt::registry& reg, Span<entt::entity const> entities) {
  if (entities.empty()) {
    return;
  }
  MOCHI_PROFILE_SCOPE();
  // Compute displacement and velocity at the start of the stage.
  ecs::InvokeForEach(&ResolveSkinning<TimeStep::StageStart>, reg, entities);
  ecs::InvokeForEach(&ComputeVelocityAtStageStart, reg, entities);
}

void skinned::EntityPostStage(
    ecs::RequiredTag<TagSoftSkinnedActor>,
    CConvergenceStatus const& convergence,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::Current, DisplacementLayer::Skinned>& currDispl,
    CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned>& currVel,
    CIntegrationVelocitySlices<DisplacementLayer::Skinned>& intVels) {
  // Skinned displacement and velocity are up-to-date. If the solver diverged, reset them to zero.
  if (convergence.stageStatus == ConvergenceStatus::Diverged) {
    currDispl.value.SetZero();
    currVel.value.SetZero();
  }

  // Push skinned velocity to the integration stages.
  intVels.stages[intState.currentStage].value = currVel.value;
}

void skinned::PostLastStagePipeline(entt::registry& reg, Span<entt::entity const> entities) {
  if (entities.empty()) {
    return;
  }
  MOCHI_PROFILE_SCOPE();
  ResolveAllNodeSkinningDisplacementsPipeline(reg, entities);
  ecs::InvokeForEach(&ComputeVelocityAtTimeStepEnd, reg, entities);
}

void skinned::UpdateDerivedStatePipeline(entt::registry& reg, Span<entt::entity const> entities) {
  if (entities.empty()) {
    return;
  }
  MOCHI_PROFILE_SCOPE();
  ecs::InvokeForEach(&ResolveSkinning<TimeStep::Current>, reg, entities);
  ecs::InvokeForEach(&ComputeCurrentVelocity, reg, entities);
}

void skinned::UpdateJacobiansPipeline(entt::registry& reg, Span<entt::entity const> entities) {
  if (entities.empty()) {
    return;
  }
  MOCHI_PROFILE_SCOPE();
  ecs::ParallelInvokeForEach(
      "Compute soft skinned Jacobian wrt skeleton", &ResolveJacobianDJoints, reg, entities);
  ecs::ParallelInvokeForEach(
      "Compute soft skinned Jacobian wrt soft", &ResolveJacobianDSoft, reg, entities);
}

void skinned::SetupCollidingJacobians(
    ecs::Included<TagSoftSkinnedActor>,
    ecs::PartialRegistry<CDofOffset const, CArticulatedLinkTransforms<TimeStep::Current> const> reg,
    CSkinnedComposition const& composition,
    CDofOffset const& dofOffset,
    CContactPartitions const& contactPartitions,
    CFemBoundaryDiscretization const& discretization,
    CArticulatedSkinningData const& skinningInfo,
    rom::CRomJacobian const* romJacobian,
    CCollJacs<CollRole::Colliding>& outJacobians) {
  MOCHI_PROFILE_SCOPE();

  // Reserve stack memory for up to 256 elements.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(JacData*) * 256);
  auto jacobiansActive = outJacobians.GetPtrsNonEmpty(&allocator);
  if (jacobiansActive.empty()) {
    return;
  }

  // Prepare articulated-actor data, shared by all partitions
  auto articulated = composition.articulated;
  auto const& articulatedDofOffset = reg.get<CDofOffset const>(articulated).dofsOffset;
  auto const& linkTransforms =
      reg.get<CArticulatedLinkTransforms<TimeStep::Current> const>(articulated);
  // Prepare bone rotations
  std::vector<VMatrix3x3r> rotations(linkTransforms.size());
  auto const& transforms = skinningInfo.skinningTransform.GetParameterizations();
  for (int i = 0; i < rotations.size(); i++) {
    auto const& preTransform = transforms[i].preTransform.GetRotation();
    rotations[i] = ToVMatrix3x3(linkTransforms[i].GetRotation() * preTransform);
  }

  // Prepare dmap that depends on the soft actor, shared by all partitions.
  std::optional<DMapSoft> dsoft;
  std::optional<DMapRom> drom;
  auto const softDofOffset = dofOffset.dofsOffset;
  if (romJacobian) {
    drom.emplace(0, romJacobian->value, softDofOffset);
  } else {
    dsoft.emplace(0, softDofOffset);
  }

  // Compute Jacobians
  ParallelForEach("SetupCollidingJacobianSoftSkinned", jacobiansActive, 1, [&](JacData* jacData) {
    discretization.Visit([&](auto const& discretizationImpl) {
      using DiscretizationT = std::decay_t<decltype(discretizationImpl)>;
      using DQuad = DMapQuad<typename DiscretizationT::ElementT>;
      using DSoft = DMap<DQuad, DMapSkinInput, DMapSoft>;
      using DRom = DMap<DQuad, DMapSkinInput, DMapRom>;
      using DMapThis = std::variant<DSoft, DRom>;
      auto const& skinningJacobian = skinningInfo.jacobianDJoints;
      auto const& skinningData = skinningInfo.skinningData;

      // Prepare the skinning dmap, specific to the partition
      auto const& dofsVariant =
          contactPartitions[jacData->query->collidingPartitionId].GetDofDescriptors()[0];
      auto dofs = MakeConstSpan(std::get<DynamicArray<int>>(dofsVariant));
      DMapSkinInput dskinning(
          1, skinningJacobian, dofs, articulatedDofOffset, skinningData, rotations);

      // Prepare the quadrature dmap
      DQuad dquad(discretizationImpl.femElements, jacData->query->jacColliderFromWorld);

      // Create per-partition differentiable map
      DMapThis dmap = drom ? DMapThis{DRom(&dquad, &dskinning, &drom.value())}
                           : DMapThis{DSoft(&dquad, &dskinning, &dsoft.value())};

      // Compute Jacobian
      auto& jacs = *jacData->jacs;
      std::visit([&](auto const& dmap) { dmap.GetJac(jacData->query->sampleIndices, jacs); }, dmap);
      jacs[0].CompressIndices();
      jacs[1].CompressIndices();
    });
  });
}

void skinned::InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CSkinnedComposition>(reg);
  ecs::RegisterComponent<CNodePositions>(reg);
  ecs::RegisterComponent<CSoftSkinnedAssemblyScratchMemory>(reg);
  ecs::RegisterComponent<CSoftSkinnedJacobianDynamic>(reg);
  ecs::RegisterComponent<CSoftSkinnedJacobianRest>(reg);
  ecs::RegisterComponent<CSoftAttachmentLinks>(reg);
}
