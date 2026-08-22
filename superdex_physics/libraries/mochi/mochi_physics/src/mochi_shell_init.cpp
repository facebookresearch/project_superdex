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

#include "mochi_shell_init.h"
#include "mochi_actor_convergence.h"
#include "mochi_common_components.h"
#include "mochi_contact.h"
#include "mochi_contact_filter.h"
#include "mochi_deformable.h"
#include "mochi_ecs_utils.h"
#include "mochi_island.h"
#include "mochi_point_cloud_contact.h"
#include "mochi_scene_recorder.h"
#include "mochi_shell.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"

#include <mochi_core/element_operations/element_assembler.h>
#include <mochi_core/element_operations/fem_inertia.h>
#include <mochi_core/geometry/deep_flow_map.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/tetrahedral_map.h>
#include <mochi_core/linear_algebra/utils/assembly.h>
#include <mochi_core/solvers/snle_problem.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/profile.h>

#include <memory>
#include <type_traits>
#include <utility>

#include <entt/entity/view.hpp>

using namespace mochi;
using namespace mochi::experimental;

static void UpdateMassMatrix(
    CActorSnle const& actorSnle,
    CLocal2GlobalMap const& l2g,
    CNodalBasedStructure const& nbs,
    real const& density,
    CFemSurfaceDiscretizationP1Q3 const& femHighDisc,
    CFullSparsityPattern const& sparsity,
    CMassMatrix& outMassMatrix,
    CPerElementMassMatrix<CFemSurfaceDiscretizationP1Q3>& outPerElemMass,
    CLumpedMassMatrix& outLumpedMass,
    CActiveVolumeElements const* activeVolElems = nullptr) {
  MOCHI_PROFILE_SCOPE();

  // Wrap the mass matrix values in a BlockSparseMatrixView for use below
  MOCHI_ASSERT_VERBOSE(
      (std::holds_alternative<BlockSparseMatrix<real, 3>>(actorSnle.fullDResidual)),
      "Expected block sparse actor matrix.");
  auto const& dresidual = std::get<BlockSparseMatrix<real, 3>>(actorSnle.fullDResidual);
  BlockSparseMatrixView<real, 3> outMassMatrixBSp(
      dresidual.BlockCols(), dresidual.Pointers(), dresidual.Indices(), outMassMatrix.values);

  // Initialize to zero
  outPerElemMass.values.resize_noinit(femHighDisc.femElements.size());
  deformable::SetZeroMassMatrix(outMassMatrixBSp, MakeSpan(outPerElemMass.values));

  MOCHI_ASSERT(
      !activeVolElems || !activeVolElems->empty(), "Active volume elements must not be empty.");
  Span<int const> activeVolIndices =
      activeVolElems ? activeVolElems->ViewIndices() : Span<int const>{};
  Span<real const> activeVolWeights =
      activeVolElems ? activeVolElems->ViewWeights() : Span<real const>{};

  // we need to first recompute the per-element mass matrices since this is used
  // below for assembling the global mass matrix
  fem::ComputeMassMatrixPerElement(
      MakeConstSpan(femHighDisc.femElements),
      density,
      MakeSpan(outPerElemMass.values),
      activeVolIndices,
      activeVolWeights);

  //
  // here we only request to assemble the DRes because we want
  // to assemble the mass matrix
  //
  AssemblyParams assemblyParams{
      .assemObj = false, .assemRes = false, .assemDRes = true, .psdDRes = false};

  AssemblyActiveSubset activeSubset = activeVolElems
      ? AssemblyActiveSubset{activeVolIndices, activeVolElems->ViewIsActive()}
      : AssemblyActiveSubset{};

  // Assemble the global mass matrix from precomputed per-element triangle mass matrices. The shell
  // assembler runs over the 6-node bending stencil, but inertia only uses the first 3 nodes, which
  // are the triangle vertices. The 9x9 triangle mass matrix is written into the top-left block of
  // the batched 18x18 element dresidual. The assembler then scatters it into the global matrix.
  auto const massMatrixOp = deformable::MakeAddMassMatrixToDResOp<shell::ShellStencilElement>(
      MakeConstSpan(outPerElemMass.values));
  AssembleObjResDRes<shell::ShellStencilElement>(
      l2g,
      nbs,
      massMatrixOp,
      AssemblyResults<real>{
          .outObj = nullptr, .outRes = {}, .outDRes = outMassMatrixBSp, .params = assemblyParams},
      activeSubset);

  // Compute lumped mass matrix.
  deformable::ComputeLumpedMassMatrix(sparsity, outMassMatrix, outLumpedMass);
}

static void EmplaceShellActorDiscretization(
    entt::registry& reg,
    entt::entity e,
    std::shared_ptr<TriangularMeshShape const> shape) {
  reg.emplace<CShape>(e, shape);
  reg.emplace<CTriangularMesh>(e, shape->GetMesh());
  reg.emplace<CSimplicialMesh>(e, shape->GetMesh());
  reg.emplace<CSurfaceMesh>(e, shape->GetMesh());

  if (shape->GetVisualMesh() && shape->GetVisualEmbedding()) {
    reg.emplace<CVisualMesh>(e, shape->GetVisualMesh(), shape->GetVisualEmbedding());
  } else if (shape->GetVisualMesh()) {
    MOCHI_LOG_WARNING(
        "Shell shape has a visual mesh but no embedding. The visual mesh will be ignored.");
  }
}

static void EmplaceShellActorContact(
    entt::registry& reg,
    entt::entity e,
    ShellActorParams const& params,
    std::shared_ptr<TriangularMeshShape const> shape,
    int numCollidingSamples,
    Error& error) {
  reg.emplace<CBoundingVolume<TimeStep::Current>>(e, shape->GetMesh()->GetObb());
  reg.emplace<CBoundingVolume<TimeStep::Previous>>(e, shape->GetMesh()->GetObb());

  ColliderType colliderType = params.colliderType;
  if (colliderType == ColliderType::Auto) {
    colliderType = ColliderType::PointCloud;
  }

  // Set collider type based on point-cloud contact settings.
  // ColliderType::PointCloud enables point-cloud contact, ColliderType::None disables it.
  // Other collider types are not supported.
  MOCHI_ERROR_IF(
      colliderType != ColliderType::PointCloud && colliderType != ColliderType::None,
      error,
      "Collider type not supported for shell actors.");
  MOCHI_ERROR_RETURN(error);
  auto& collider = reg.emplace<CColliderInfo>(e);
  collider.type = colliderType;

  reg.emplace<CContactParams>(e, params.contact);

  // Components to detect and compute contact against other actors
  deformable::EmplaceContactComponents(reg, e, numCollidingSamples);
  reg.emplace<CDeformablePointAsyncCollisionsResponse>(e);

  reg.emplace<CActorAsyncContactSemaphore>(e);
}

static void EmplaceShellShellContact(
    entt::registry& reg,
    entt::entity e,
    ShellActorParams const& params,
    TriangularMesh const& actorTriMesh,
    Error& error) {
  if (reg.get<CColliderInfo>(e).type == ColliderType::None) {
    return;
  }
  MOCHI_ERROR_IF_NOT(
      reg.get<CColliderInfo>(e).type == ColliderType::PointCloud,
      error,
      "Collider type not supported for shell actors.");
  MOCHI_ERROR_RETURN(error);

  reg.emplace<TagUsePointCloudContact>(e);
  reg.emplace<CCollJacs<CollRole::Collider>>(e);

  // Emplace point-cloud collider properties.
  ValidatePointCloudColliderParams(params.pointCloudCollider, error);
  MOCHI_ERROR_RETURN(error);
  auto& pcComponent = reg.emplace<CPointCloudColliderParams>(e, params.pointCloudCollider);
  pcComponent.integralDim = 2;
  auto const& nodalWeights = reg.emplace<CNodalWeights>(
      e, InitializeNodalWeights(reg.get<CFemSurfaceDiscretizationP1Q1>(e)));
  // Create collider surface discretization from colliderTriangleElementType.
  CColliderPointCloudDiscretization colliderDisc = [&]() -> CColliderPointCloudDiscretization {
    if (!params.pointCloudCollider.colliderTriangleElementType.has_value()) {
      NodalColliderDiscretization nodalDisc;
      int const numNodes = isize(actorTriMesh.GetNodeCoordinates());
      auto coordinates = actorTriMesh.GetNodeCoordinates();
      nodalDisc.femElements.reserve(numNodes);
      for (int i = 0; i < numNodes; ++i) {
        nodalDisc.femElements.emplace_back(i, coordinates, i, nodalWeights.values[i]);
      }
      return CColliderPointCloudDiscretization(std::move(nodalDisc), kSpaceDim3);
    }
    return CColliderPointCloudDiscretization(
        CFemSurfaceDiscretization::Create(
            *params.pointCloudCollider.colliderTriangleElementType, actorTriMesh),
        kSpaceDim3);
  }();
  auto& colliderDiscRef =
      reg.emplace<CColliderPointCloudDiscretization>(e, std::move(colliderDisc));
  reg.emplace<CSpatialHashTable>(
      e,
      CreateSpatialHashTable(
          params.pointCloudCollider,
          colliderDiscRef,
          params.contact.GetPenaltyThresholdDist(true)));
}

void mochi::InitShellActor(
    entt::registry& reg,
    entt::entity e,
    ShellActorParams const& params,
    std::shared_ptr<TriangularMeshShape const> shapePtr,
    Error& error) {
  ValidateContactParams(params.contact, error);
  shell::ValidateShellMaterialParams(params.material, error);
  MOCHI_ERROR_RETURN(error);

  // Identification
  reg.emplace<TagShellActor>(e);
  reg.emplace<TagDeformableActor>(e);
  reg.emplace<CActorInfo>(e, std::string(params.name), ActorType::Shell);
  reg.emplace<CDofOffset>(e);

  EmplaceContactLayer(reg, e, params.layer);

  reg.emplace<CRootTransform>(e, params.worldFromLocal);
  reg.emplace<CTimeIntegratorState>(e);
  reg.emplace<CConvergenceStatus>(e);

  // Material
  MOCHI_ERROR_RETURN(error);
  auto& material = reg.emplace<CShellMaterialParams>(e);
  shell::SetMaterialParams(params.material, material);

  if (params.hasGravity) {
    reg.emplace<TagUseGravity>(e);
  }

  EmplaceShellActorDiscretization(reg, e, shapePtr);
  reg.emplace<CDirichletBC<real>>(e);
  reg.emplace<CExternalForces>(e);

  TriangularMesh const& actorTriMesh = *shapePtr->GetMesh();
  CActorDofInfo& dofInfo = reg.emplace<CActorDofInfo>(e);
  dofInfo.poseSize = actorTriMesh.GetNumNodes() * 3;
  dofInfo.dofsSize = dofInfo.poseSize;

  // Kinematics data
  int const actorMeshDofs = actorTriMesh.GetNumNodes() * kSpaceDim3;
  auto& dispCurr = reg.emplace<CDisplacementSlice<real, TimeStep::Current>>(e, actorMeshDofs);
  reg.emplace<CDisplacementSlice<real, TimeStep::Previous>>(e, actorMeshDofs);
  auto& dispStart = reg.emplace<CDisplacementSlice<real, TimeStep::StageStart>>(e, actorMeshDofs);
  reg.emplace<CIntegrationDisplacementSlices>(e, actorMeshDofs);

  reg.emplace<CVelocitySlice<real, TimeStep::Current>>(e, actorMeshDofs);
  reg.emplace<CVelocitySlice<real, TimeStep::Previous>>(e, actorMeshDofs);
  reg.emplace<CVelocitySlice<real, TimeStep::StageStart>>(e, actorMeshDofs);
  reg.emplace<CIntegrationVelocitySlices<DisplacementLayer::Default>>(e, actorMeshDofs);

  // Set default displacement-slice references.
  reg.emplace<CFinalDisplacementRef<TimeStep::Current>>(e, dispCurr.value);
  reg.emplace<CFinalDisplacementRef<TimeStep::StageStart>>(e, dispStart.value);

  // Nodal based structure.
  auto const bendingNodalConnectivityAndStencil =
      actorTriMesh.GenerateBendingConnectivityAndStencil();
  auto const& bendingEToN = bendingNodalConnectivityAndStencil.first;
  auto const& bendingStencil = bendingNodalConnectivityAndStencil.second;

  // Build padded bending connectivity: every element has exactly kNumStencilNodes nodes. Missing
  // stencil positions are filled with the element's first node (node-0). This makes the NBS store
  // a uniform 6×6 sparse index block per element.
  //
  // The N-to-N graph is computed from the original (non-padded) bending connectivity and passed
  // explicitly to the padded NBS builder. This avoids duplicate entries that would arise from the
  // padding node appearing twice in a boundary element's node list.
  auto const& nbs = reg.emplace<CNodalBasedStructure>(
      e, BuildPaddedNodalBasedStructure<shell::kNumStencilNodes>(bendingEToN, bendingStencil));

  // L2G stores the raw bending connectivity plus stencil positions. Padded indices give the FEM
  // assembler a uniform shell::kNumStencilNodes * kSpaceDim3 stride. Physical sparsity is still
  // built from the raw connectivity.
  auto& l2g = reg.emplace<CLocal2GlobalMap>(e);
  l2g.InitializeFromElementNodeConnectivity(bendingEToN, kSpaceDim3);
  l2g.InitializeStencilIndices(bendingStencil);
  l2g.InitializePaddedIndices(shell::kNumStencilNodes * kSpaceDim3);

  auto& fullSparsity = reg.emplace<CFullSparsityPattern>(e, MakeSparsityGraph(l2g, actorMeshDofs));

  // DResidual Matrix
  BlockSparseMatrix<real, 3> actorDRes;
  {
    int const numRows = isize(fullSparsity.graph.GetPointers()) - 1;
    int const numCols = numRows; // Symmetrical
    DynamicArray<real> values(fullSparsity.graph.NumTargets());
    auto actorDResFullSparse = SparseMatrixView<real const>(
        numCols,
        fullSparsity.graph.GetPointers(),
        fullSparsity.graph.GetTargets(),
        MakeSpan(values));
    auto blockStructure = BlockedStructure<3>(actorDResFullSparse);
    actorDRes.Reset(
        blockStructure.nBlockCols,
        std::move(blockStructure.ptr),
        std::move(blockStructure.ndIndices),
        std::move(values));
  }

  // Actor SNLE data.
  auto const& actorSnle = reg.emplace<CActorSnle>(e, std::move(actorDRes));

  // Non-linear solver convergence weights (lazily initialized).
  reg.emplace<CActorConvergenceWeights>(e);

  // Surface discretizations
  auto* femLowSurfDisc = &reg.emplace<CFemSurfaceDiscretizationP1Q1>(e);
  auto* femHighSurfDisc = &reg.emplace<CFemSurfaceDiscretizationP1Q3>(e);
  {
    femLowSurfDisc->femElements.reserve(actorTriMesh.GetNumElements());
    femHighSurfDisc->femElements.reserve(actorTriMesh.GetNumElements());
    auto const meshCoords = actorTriMesh.GetNodeCoordinates();
    auto const meshConnec = actorTriMesh.GetElementConnectivity();
    int const meshNumEle = actorTriMesh.GetNumElements();
    for (int i = 0; i < meshNumEle; ++i) {
      femLowSurfDisc->femElements.emplace_back(i, meshCoords, meshConnec);
      femHighSurfDisc->femElements.emplace_back(i, meshCoords, meshConnec);
    }
  }
  auto const& contactDisc = reg.emplace<CFemSurfaceDiscretization>(
      e, CFemSurfaceDiscretization::Create(params.contactElementType, actorTriMesh));
  // FIXME: This is a workaround to accommodate the fact that some queries use only the "lite"
  // surface discretization designed for rigid actors. These should be unified behind a single
  // interface, so only one component is added.
  reg.emplace<CFemSurfaceDiscretizationLite>(
      e, CFemSurfaceDiscretizationLite::Create(params.contactElementType, actorTriMesh));

  // Codimensional contact assembly uses the shell surface triangles as the assembly elements:
  // L2G has fixed 3-node x 3-DoF stride, matching CFemSurfaceDiscretization. The NBS uses the same
  // triangle element order, while sparse indices are computed against the actor's full
  // bending-stencil sparsity.
  auto const& triConnectivity = actorTriMesh.GetElementConnectivity();
  auto& contactL2g = reg.emplace<CContactLocal2GlobalMap>(e);
  contactL2g.InitializeFromElementNodeConnectivity(triConnectivity, kSpaceDim3);
  reg.emplace<CContactNodalBasedStructure>(
      e, NodalBasedStructure(GraphFromRangeOfRanges<int, int>(triConnectivity), nbs.GetNToN()));

  int const numCollidingSamples = contactDisc.GetNumQuadPoints();
  EmplaceShellActorContact(reg, e, params, shapePtr, numCollidingSamples, error);
  MOCHI_ERROR_RETURN(error);

  EmplaceShellShellContact(reg, e, params, actorTriMesh, error);
  MOCHI_ERROR_RETURN(error);

  // Bounds used for collision detection.
  reg.emplace<CConservativeStepBounds>(e);

  // MassMatrix (for inertia)
  auto& massMatrix = reg.emplace<CMassMatrix>(e);
  massMatrix.values.resize(GetNumValues(actorSnle.fullDResidual), 0_r);

  // Lumped mass matrix. Computed in UpdateMassMatrix.
  auto& lumpedMass = reg.emplace<CLumpedMassMatrix>(e);

  // Per-element mass matrix
  // here, rather than hardwiring the template, we can deduce it so
  // that if that changes this does not have to be modified
  CPerElementMassMatrix<std::remove_pointer_t<decltype(femHighSurfDisc)>> perElemMass;
  perElemMass.values.resize(l2g.GetNumElements());

  UpdateMassMatrix(
      actorSnle,
      l2g,
      nbs,
      material.density,
      *femHighSurfDisc,
      fullSparsity,
      massMatrix,
      perElemMass,
      lumpedMass);
}
