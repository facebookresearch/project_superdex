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
#include "mochi_actor_preconditioner_policy.h"
#include "mochi_common_components.h"
#include "mochi_contact.h"
#include "mochi_contact_filter.h"
#include "mochi_deformable.h"
#include "mochi_ecs_utils.h"
#include "mochi_island.h"
#include "mochi_linear_contact_skin.h"
#include "mochi_point_cloud_contact.h"
#include "mochi_scene_recorder.h"
#include "mochi_shell.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"

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

#include <memory>
#include <utility>

#include <entt/entity/view.hpp>

using namespace mochi;
using namespace mochi::experimental;

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

  if (shape->GetContactSkin()) {
    reg.emplace<CContactSkinMesh>(e, shape->GetContactSkin(), shape->GetContactSkinEmbedding());
  }
}

static void EmplaceShellActorContact(
    entt::registry& reg,
    entt::entity e,
    ShellActorParams const& params,
    std::shared_ptr<TriangularMeshShape const> shape,
    int numCollidingSamples,
    Error& error) {
  reg.emplace<CBoundingVolume>(e, shape->GetMesh()->GetObb());

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
  deformable::EmplaceContactComponents(reg, e, numCollidingSamples, /*numContactPartitions*/ 1);
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
  ValidatePointCloudColliderParams(params.pointCloudCollider, params.contact, error);
  MOCHI_ERROR_RETURN(error);
  auto& pcComponent = reg.emplace<CPointCloudColliderParams>(e, params.pointCloudCollider);
  pcComponent.integralDim = 2;
  int const numNodes = actorTriMesh.GetNumNodes();
  auto const& nodalWeights = reg.emplace<CNodalWeights>(
      e, InitializeNodalWeights(reg.get<CFemSurfaceDiscretizationP1Q1>(e), numNodes));
  // Create collider surface discretization from colliderTriangleElementType.
  CColliderPointCloudDiscretization colliderDisc = [&]() -> CColliderPointCloudDiscretization {
    if (!params.pointCloudCollider.colliderTriangleElementType.has_value()) {
      NodalColliderDiscretization nodalDisc;
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
  auto const& shapeContactSkinMesh = shapePtr->GetContactSkin();
  MOCHI_ERROR_IF(
      params.useContactSkin && !shapeContactSkinMesh,
      error,
      "useContactSkin requires a shell shape with a contact skin.");
  MOCHI_ERROR_IF(
      shapePtr->GetMesh()->GetNumActiveNodes() != shapePtr->GetMesh()->GetNumNodes(),
      error,
      "Shell mesh must not contain nodes that are unreferenced by its triangles.");
  MOCHI_ERROR_RETURN(error);
  // Plane stress of any valid isotropic 3D material gives membraneLambda / membraneMu < 2. The
  // tolerance avoids warning on rounding error near the incompressible limit.
  real constexpr kMaxPlaneStressLameRatio = 2_r + 1e-4_r;
  if (params.material.membraneLambda > kMaxPlaneStressLameRatio * params.material.membraneMu) {
    MOCHI_LOG_WARNING(
        "Creating shell actor \"%s\" with membraneLambda / membraneMu = %g > 2, which exceeds the "
        "plane-stress limit of any valid isotropic 3D material. The model remains well-posed, but "
        "is not derivable from an isotropic 3D material.",
        params.name.c_str(),
        static_cast<double>(params.material.membraneLambda / params.material.membraneMu));
  }

  // Identification
  reg.emplace<TagShellActor>(e);
  reg.emplace<TagDeformableActor>(e);
  reg.emplace<CActorInfo>(e, std::string(params.name), ActorType::Shell);
  reg.emplace<CDofOffset>(e);

  EmplaceContactLayer(reg, e, params.layer);

  reg.emplace<CRootTransform>(e, params.worldFromLocal);
  reg.emplace<CTimeIntegratorState>(e);
  reg.emplace<CConvergenceStatus>(e);
  reg.emplace<CRecenteringParams>(e);
  reg.emplace<CRecenteringPivotNode>(
      e, deformable::FindRecenteringPivotNode(shapePtr->GetMesh()->GetNodeCoordinates()));

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
  // assembler a uniform shell::kNumStencilNodes * kSpaceDim3 stride.
  auto& l2g = reg.emplace<CLocal2GlobalMap>(e);
  l2g.InitializeFromElementNodeConnectivity(bendingEToN, kSpaceDim3);
  l2g.InitializeStencilIndices(bendingStencil);
  l2g.InitializePaddedIndices(shell::kNumStencilNodes * kSpaceDim3);

  // Actor SNLE data.
  reg.emplace<CActorSnle>(e, BlockSparseMatrix<real, 3>(actorTriMesh.GetNumNodes(), nbs.GetNToN()));

  // Non-linear solver convergence weights (lazily initialized).
  reg.emplace<CActorConvergenceWeights>(e);

  // Copy of the island's adaptive preconditioner selection (written by the island solve).
  reg.emplace<CActorPreconditionerPolicyCopy>(e);

  // Surface discretization
  auto& femSurfDisc = reg.emplace<CFemSurfaceDiscretizationP1Q1>(e);
  {
    femSurfDisc.femElements.reserve(actorTriMesh.GetNumElements());
    auto const meshCoords = actorTriMesh.GetNodeCoordinates();
    auto const meshConnec = actorTriMesh.GetElementConnectivity();
    int const meshNumEle = actorTriMesh.GetNumElements();
    for (int i = 0; i < meshNumEle; ++i) {
      femSurfDisc.femElements.emplace_back(i, meshCoords, meshConnec);
    }
  }
  TriangularMesh const& contactMesh = params.useContactSkin ? *shapeContactSkinMesh : actorTriMesh;
  auto const& contactDisc = reg.emplace<CFemSurfaceDiscretization>(
      e, CFemSurfaceDiscretization::Create(params.contactElementType, contactMesh));
  // FIXME: This is a workaround to accommodate the fact that some queries use only the "lite"
  // surface discretization designed for rigid actors. These should be unified behind a single
  // interface, so only one component is added.
  reg.emplace<CFemSurfaceDiscretizationLite>(
      e, CFemSurfaceDiscretizationLite::Create(params.contactElementType, contactMesh));

  if (params.useContactSkin) {
    linear_contact_skin::EmplaceComponents(
        reg,
        e,
        contactDisc,
        *shapeContactSkinMesh,
        *shapePtr->GetContactSkinEmbedding(),
        actorTriMesh.GetNodeCoordinates());
  } else {
    // Direct contact assembles shell triangles into the actor's body matrix.
    auto const& triConnectivity = actorTriMesh.GetElementConnectivity();
    auto& contactL2g = reg.emplace<CContactLocal2GlobalMap>(e);
    contactL2g.InitializeFromElementNodeConnectivity(triConnectivity, kSpaceDim3);
    reg.emplace<CContactNodalBasedStructure>(
        e, NodalBasedStructure(GraphFromRangeOfRanges<int, int>(triConnectivity), nbs.GetNToN()));
  }

  int const numCollidingSamples = contactDisc.GetNumQuadPoints();
  EmplaceShellActorContact(reg, e, params, shapePtr, numCollidingSamples, error);
  MOCHI_ERROR_RETURN(error);

  EmplaceShellShellContact(reg, e, params, actorTriMesh, error);
  MOCHI_ERROR_RETURN(error);

  if (params.useContactSkin) {
    ecs::InvokeOnEntity(linear_contact_skin::UpdateBounds<TimeStep::Current>, reg, e);
  } else {
    ecs::InvokeOnEntity(shell::UpdateBounds<TimeStep::Current>, reg, e);
  }

  // Bounds used for collision detection.
  reg.emplace<CConservativeStepBounds>(e);

  // Lumped mass, for the convergence weights.
  auto& lumpedMass = reg.emplace<CLumpedMassMatrix>(e);
  lumpedMass.values.resize_noinit(actorMeshDofs);
  fem::ComputeLumpedMass(
      MakeConstSpan(femSurfDisc.femElements),
      material.density,
      /*activeElementIndices*/ {},
      /*perElementExtraWeight*/ {},
      MakeSpan(lumpedMass.values));
}
