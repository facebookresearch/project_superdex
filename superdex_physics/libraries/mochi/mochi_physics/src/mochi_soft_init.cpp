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

#include "mochi_soft_init.h"

#include "mochi_actor_convergence.h"
#include "mochi_common_components.h"
#include "mochi_contact.h"
#include "mochi_contact_filter.h"
#include "mochi_deformable.h"
#include "mochi_island.h"
#include "mochi_scene_recorder.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"

#include <mochi_core/element_operations/element_assembler.h>
#include <mochi_core/geometry/deep_flow_map.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/tetrahedral_map.h>
#include <mochi_core/materials/material_params_utils.h>
#include <mochi_core/solvers/snle_problem.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/profile.h>
#include <mochi_physics/mochi_physics_experimental.h>

#include <memory>
#include <type_traits>
#include <utility>

#include <entt/entity/view.hpp>

using namespace mochi;
using namespace mochi::experimental;

static void EmplaceSoftActorDiscretization(
    entt::registry& reg,
    entt::entity e,
    std::shared_ptr<TetrahedralMeshShape const> shape) {
  reg.emplace<CShape>(e, shape);
  reg.emplace<CTetrahedralMesh>(e, shape->GetMesh());
  reg.emplace<CSimplicialMesh>(e, shape->GetMesh());
  reg.emplace<CSurfaceMesh>(e, shape->GetMesh()->GetBoundaryMesh());

  if (shape->GetVisualMesh() && shape->GetVisualEmbedding()) {
    reg.emplace<CVisualMesh>(e, shape->GetVisualMesh(), shape->GetVisualEmbedding());
  }
}

static void EmplaceSoftActorContactSdf(
    entt::registry& reg,
    entt::entity e,
    std::shared_ptr<TetrahedralMeshShape const> shape,
    std::shared_ptr<DeepFlowShape const> flow,
    SoftActorParams const& params,
    ExperimentalSoftActorParams const& experimentalParams,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  auto const& surfMesh = shape->GetSurfaceMesh();

  ContactParams contactParams = params.contact;

  // If a deep flow is provided, create deep flow mappings
  if (flow) {
    reg.emplace<CSdfMapping<TimeStep::StageStart>>(
        e, CreateDeepFlowMap(flow->flow, contactParams.objScale, error));
    reg.emplace<CSdfMapping<TimeStep::Current>>(
        e, CreateDeepFlowMap(flow->flow, contactParams.objScale, error));

    reg.emplace<TagHasDeepFlowCollider>(e);

    // All deep flow collision queries need to be combined into a single batch to send to the GPU.
    // That means that all actors with deep flow colliders must be in a single island, together with
    // any other actors that might collide with them. The easiest way to accomplish this is to force
    // every actor into a single island.
    //
    // WARNING: Forming one large island can be very bad for CPU performance. We should revisit this
    // requirement if we every use Deep Flow for interactive experiences.
    //
    reg.emplace<TagForceSingleIsland>(e);
  }
  // Otherwise create tetrahedral mappings
  else {
    MOCHI_ERROR_IF(
        experimentalParams.rom, error, "Tetrahedral map SDF not supported for ROM actors.");
    MOCHI_ERROR_RETURN(error);
    reg.emplace<CSdfMapping<TimeStep::StageStart>>(
        e, std::make_unique<TetrahedralMap>(shape->GetMesh()));
    reg.emplace<CSdfMapping<TimeStep::Current>>(
        e, std::make_unique<TetrahedralMap>(shape->GetMesh()));

    // Contact with tetrahedral mappings works only with a distance of 0
    contactParams.penaltyThresholdDefault = 0_r;
    contactParams.penaltyThresholdExtraPadding = 0_r;
  }

  auto& contactParamsComponent = reg.emplace<CContactParams>(e, contactParams);

  // Create the SDF
  auto& colliderObj = reg.emplace<CSdfCollider>(e);
  GridSdfParams gridSdfParams = experimentalParams.sdf;
  // Define the padding based on the most conservative penalty threshold distance.
  gridSdfParams.boundaryPaddingDist =
      contactParamsComponent.GetPenaltyThresholdDist(/* addPadding */ true);
  colliderObj.shape = std::make_unique<GridSdf>(surfMesh, gridSdfParams, error);
  MOCHI_ERROR_RETURN(error);

  LogSdfColliderDiagnostics(colliderObj, reg.get<CActorInfo>(e).name);
}

static void EmplaceSoftActorContact(
    entt::registry& reg,
    entt::entity e,
    SoftActorParams const& params,
    ExperimentalSoftActorParams const& experimentalParams,
    bool useContact,
    int numCollidingSamples,
    std::shared_ptr<TetrahedralMeshShape const> shape,
    Error& error,
    std::shared_ptr<DeepFlowShape const> flow) {
  reg.emplace<CBoundingVolume<TimeStep::Current>>(e, shape->GetMesh()->GetObb());
  reg.emplace<CBoundingVolume<TimeStep::Previous>>(e, shape->GetMesh()->GetObb());
  auto& collider = reg.emplace<CColliderInfo>(e);

  static_assert(
      static_cast<int>(ColliderType::Count) == 8,
      "Please update the following switch statement if the ColliderType enum changes.");

  ColliderType resolvedColliderType = experimentalParams.colliderType;

  switch (resolvedColliderType) {
    case ColliderType::Sphere:
    case ColliderType::Box:
    case ColliderType::Mesh:
    case ColliderType::Plane:
    case ColliderType::PointCloud: {
      MOCHI_ERROR_SET(error, "Unsupported ColliderType.");
    } break;

    case ColliderType::Auto:
    case ColliderType::Sdf: {
      EmplaceSoftActorContactSdf(reg, e, shape, flow, params, experimentalParams, error);
      collider.type = ColliderType::Sdf;
    } break;

    case ColliderType::None: {
      // Other actors won't be able to collide with this one. That is legal. Contact might still
      // be possible via sync contact, as long as the other actor has a valid ColliderType.
      reg.emplace<CContactParams>(e, params.contact);
      collider.type = ColliderType::None;
    } break;

    default:
      MOCHI_ASSERT(false, "Invalid ColliderType");
      break;
  }

  // Components to detect and compute contact against other actors
  if (useContact) {
    deformable::EmplaceContactComponents(reg, e, numCollidingSamples);
    reg.emplace<CDeformablePointAsyncCollisionsResponse>(e);
  }
  if (collider.type != ColliderType::None) {
    reg.emplace<CCollJacs<CollRole::Collider>>(e);
  }

  reg.emplace<CActorAsyncContactSemaphore>(e);
}

static void EmplaceSoftActorRigidPivotAndRecentering(
    entt::registry& reg,
    entt::entity e,
    TetrahedralMesh const& solverMesh,
    CFemVolumeDiscretizationP1Q1 const& femLowVolDisc) {
  Real3 rigidPivotPos = {};
  int rigidPivotEleIdx = -1;

  bool pivotFound = solverMesh.GetRigidPivot(rigidPivotEleIdx, rigidPivotPos);
  MOCHI_ASSERT(pivotFound, "No recentering pivot; something went wrong");
  MOCHI_ASSERT(rigidPivotEleIdx >= 0);
  MOCHI_ASSERT(rigidPivotEleIdx < (int)femLowVolDisc.femElements.size());
  auto const& pivotElement = femLowVolDisc.femElements[rigidPivotEleIdx];
  auto const& rigidTransformEvalPoint =
      reg.emplace_or_replace<CRigidTransformEvalPoint>(e, pivotElement, rigidPivotPos);
  MOCHI_ASSERT(
      rigidTransformEvalPoint.IsValid(),
      "The EvalPoint should be valid because the element index is valid");
  reg.emplace_or_replace<CRigidTransformEval>(e);
  ecs::InvokeOnEntity(&soft::UpdateRigidTransformEval, reg, e);
  reg.emplace_or_replace<CMeshPivot>(e, rigidPivotPos);

  // Recentering
  reg.emplace_or_replace<CRecenteringParams>(e);
}

void mochi::InitSoftActor(
    entt::registry& reg,
    entt::entity e,
    SoftActorParams const& params,
    ExperimentalSoftActorParams const& experimentalParams,
    bool useContact,
    bool isNestedSoft,
    std::shared_ptr<TetrahedralMeshShape const> shapePtr,
    std::shared_ptr<DeepFlowShape const> flow,
    Error& error) {
  MOCHI_ERROR_IF(
      !isNestedSoft && !params.hasInertia && !params.hasGravity && !params.hasStress,
      error,
      "Soft actors must have at least one of inertia, gravity or stress enabled.");
  ValidateSoftMaterialParams(params.material, error);
  ValidateContactParams(params.contact, error);
  MOCHI_ERROR_RETURN(error);
  if (!params.hasInertia && params.material.massDampingCoefficient > 0_r) {
    MOCHI_LOG_WARNING("Nonzero soft mass damping inactive because hasInertia is false.");
  }
  if (!params.hasStress && params.material.stiffnessDampingCoefficient > 0_r) {
    MOCHI_LOG_WARNING("Nonzero soft stiffness damping inactive because hasStress is false.");
  }

  // Identification
  reg.emplace<TagSoftActor>(e);
  reg.emplace<TagDeformableActor>(e);
  reg.emplace<CActorInfo>(e, std::string(params.name), ActorType::Soft);
  reg.emplace_or_replace<CDofOffset>(e);

  reg.emplace<CSoftExportParams>(e).boundaryElementType = params.boundaryElementType;

  EmplaceContactLayer(reg, e, params.layer);

  reg.emplace<CRootTransform>(e, params.worldFromLocal);
  reg.emplace<CTimeIntegratorState>(e);
  reg.emplace<CConvergenceStatus>(e);

  auto& material = reg.emplace_or_replace<CSoftMaterialParams>(e);
  soft::SetMaterialParams(params.material, material);

  // WARNING: If the model file contained per-element material data for the same SoftMaterialType,
  // then we will use the parameters from the model file and ignore the parameters provided via
  // SoftActorParams::material. Is that what the caller wanted?
  // TODO: Improve the public API to make the behavior unambiguous.
  auto const& materialField = shapePtr->GetSoftMaterialParamsField();
  if (materialField && (materialField->type == params.material.type)) {
    soft::SetMaterialParamsField(
        materialField.get(), shapePtr->GetMesh()->GetNumElements(), material, error);
    MOCHI_ERROR_RETURN(error);
  }

  // energy terms
  if (params.hasInertia) {
    reg.emplace<TagUseInertia>(e);
  }
  if (params.hasStress) {
    reg.emplace<TagUseStress>(e);
  }
  if (params.hasGravity) {
    reg.emplace<TagUseGravity>(e);
  }
  // by default, no energy is evaluated on skinned positions
  reg.emplace<CSkinnedEnergy>(e);

  EmplaceSoftActorDiscretization(reg, e, shapePtr);
  reg.emplace<CDirichletBC<real>>(e);

  reg.emplace<CPrevRigidVelocity>(e);

  TetrahedralMesh const& actorTetMesh = *shapePtr->GetMesh();
  CActorDofInfo& dofInfo = reg.emplace<CActorDofInfo>(e);
  dofInfo.poseSize = actorTetMesh.GetNumNodes() * 3;
  dofInfo.dofsSize = dofInfo.poseSize;

  // Kinematics data
  int const actorMeshDofs = actorTetMesh.GetNumNodes() * 3;
  auto& dispCurr = reg.emplace<CDisplacementSlice<real, TimeStep::Current>>(e, actorMeshDofs);
  reg.emplace<CDisplacementSlice<real, TimeStep::Previous>>(e, actorMeshDofs);
  auto& dispStart = reg.emplace<CDisplacementSlice<real, TimeStep::StageStart>>(e, actorMeshDofs);
  reg.emplace<CIntegrationDisplacementSlices>(e, actorMeshDofs);

  reg.emplace<CVelocitySlice<real, TimeStep::Current>>(e, actorMeshDofs);
  reg.emplace<CVelocitySlice<real, TimeStep::Previous>>(e, actorMeshDofs);
  reg.emplace<CVelocitySlice<real, TimeStep::StageStart>>(e, actorMeshDofs);
  reg.emplace<CIntegrationVelocitySlices<DisplacementLayer::Default>>(e, actorMeshDofs);

  // Set default displacement-slice references. These may be overwritten if the actor is skinned.
  reg.emplace<CFinalDisplacementRef<TimeStep::Current>>(e, dispCurr.value);
  reg.emplace<CFinalDisplacementRef<TimeStep::StageStart>>(e, dispStart.value);
  reg.emplace<CStressDisplacementRef>(e, dispCurr.value);

  // Nodal based structure.
  auto const& nbs = reg.emplace<CNodalBasedStructure>(e, actorTetMesh.GetElementConnectivity());

  // L2G
  CLocal2GlobalMap* l2g = nullptr;
  tetrahedral::BarycentricBasisTetrahedra<1> basis;
  l2g = &reg.emplace<CLocal2GlobalMap>(e);
  l2g->InitializeFromMeshAndBasis(&actorTetMesh, basis, 3);

  // Sparsity pattern.
  auto& fullSparsity = reg.emplace<CFullSparsityPattern>(e, MakeSparsityGraph(*l2g, actorMeshDofs));

  // DResidual Matrix
  int const numRows = isize(fullSparsity.graph.GetPointers()) - 1;
  int const numCols = numRows; // Symmetrical
  DynamicArray<real> values(fullSparsity.graph.NumTargets());
  auto actorDResFullSparse = SparseMatrixView<real const>(
      numCols, fullSparsity.graph.GetPointers(), fullSparsity.graph.GetTargets(), MakeSpan(values));
  auto blockStructure = BlockedStructure<3>(actorDResFullSparse);
  BlockSparseMatrix<real, 3> actorDRes(
      blockStructure.nBlockCols,
      std::move(blockStructure.ptr),
      std::move(blockStructure.ndIndices),
      std::move(values));

  // Actor SNLE data.
  auto const& actorSnle = reg.emplace<CActorSnle>(e, std::move(actorDRes));

  // Non-linear solver convergence weights (lazily initialized).
  reg.emplace<CActorConvergenceWeights>(e);

  // Volume discretizations
  auto& femLowVolDisc = reg.emplace<CFemVolumeDiscretizationP1Q1>(e);
  auto& femHighVolDisc = reg.emplace<CFemVolumeDiscretizationP1Q4>(e);
  {
    femLowVolDisc.femElements.reserve(actorTetMesh.GetNumElements());
    femHighVolDisc.femElements.reserve(actorTetMesh.GetNumElements());
    auto const meshCoords = actorTetMesh.GetNodeCoordinates();
    auto const meshConnec = actorTetMesh.GetElementConnectivity();
    int const meshNumEle = actorTetMesh.GetNumElements();
    for (int i = 0; i < meshNumEle; ++i) {
      femLowVolDisc.femElements.emplace_back(
          i, meshCoords, meshConnec, tetrahedral::kTetrahedralQuadrature1);
      femHighVolDisc.femElements.emplace_back(
          i, meshCoords, meshConnec, tetrahedral::kTetrahedralQuadrature4);
    }
  }

  // Initialize recentering
  if (experimentalParams.useRecentering) {
    EmplaceSoftActorRigidPivotAndRecentering(reg, e, actorTetMesh, femLowVolDisc);
  }

  // Boundary discretization
  auto const& boundaryDisc = reg.emplace<CFemBoundaryDiscretization>(
      e,
      CFemBoundaryDiscretization::Create(actorTetMesh, femLowVolDisc, params.boundaryElementType));
  int const numCollidingSamples = boundaryDisc.GetNumQuadPoints();

  // Boundary-face nodal based structure and L2G.
  boundaryDisc.Visit([&](auto const& disc) {
    BoundaryAssemblyData bdData(
        MakeConstSpan(disc.femElements),
        actorTetMesh.GetElementConnectivity(),
        nbs.GetNToN(),
        /*numFields*/ kSpaceDim3);
    reg.emplace<CBoundaryNodalBasedStructure>(e, std::move(bdData.nbs));
    reg.emplace<CBoundaryLocal2GlobalMap>(e, std::move(bdData.l2g));
  });

  EmplaceSoftActorContact(
      reg, e, params, experimentalParams, useContact, numCollidingSamples, shapePtr, error, flow);

  // bounds
  reg.emplace<CConservativeStepBounds>(e);

  // MassMatrix (for inertia)
  auto& massMatrix = reg.emplace<CMassMatrix>(e);
  massMatrix.values.resize(GetNumValues(actorSnle.fullDResidual), 0_r);

  // Lumped mass matrix. Computed in UpdateSoftMass.
  auto& lumpedMass = reg.emplace<CLumpedMassMatrix>(e);

  // Per-element mass matrix
  // here, rather than hardwiring the template, we can deduce it so
  // that if that changes this does not have to be modified
  auto& perElemMass = reg.emplace<CPerElementMassMatrix<std::decay_t<decltype(femHighVolDisc)>>>(e);
  perElemMass.values.resize(l2g->GetNumElements());

  soft::UpdateSoftMass(
      actorSnle,
      *l2g,
      nbs,
      material,
      femHighVolDisc,
      fullSparsity,
      massMatrix,
      perElemMass,
      lumpedMass);
}
