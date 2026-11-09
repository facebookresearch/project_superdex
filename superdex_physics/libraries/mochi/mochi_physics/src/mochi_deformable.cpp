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

#include "mochi_deformable.h"

#include "mochi_contact.h"
#include "mochi_contact_anchors.h"
#include "mochi_discretization_components.h"
#include "mochi_ecs_utils.h"
#include "mochi_rod.h"
#include "mochi_scene_recorder.h"
#include "mochi_soft.h"

#include <mochi_core/contact/dmap.h>
#include <mochi_core/element_operations/fem_traction.h>
#include <mochi_core/utils/array_utils.h>

using namespace mochi;
using namespace dmap;

// Update CDirichletBC (local space) based on CDofPositionsBC (if any)
static void UpdateDirichletBC(
    CRootTransform const& root,
    CSimplicialMesh const& mesh,
    CDofPositionsBC const* inWorldBC,
    CDirichletBC<real>& outLocalBC) {
  MOCHI_PROFILE_SCOPE();
  outLocalBC.Clear();

  // CDofPositionsBC only exists if the user set BCs through the public API
  if (!inWorldBC || inWorldBC->poseIndices.empty()) {
    return; // No fixed nodes
  }

  MOCHI_ASSERT(isize(inWorldBC->poseIndices) % 3 == 0, "Expected 3 DOFs per node");
  MOCHI_ASSERT(isize(inWorldBC->poseValues) == isize(inWorldBC->poseIndices));
  MOCHI_ASSERT(isize(inWorldBC->dofIndices) == isize(inWorldBC->poseIndices));
  MOCHI_ASSERT(isize(inWorldBC->colValueIndices) % 3 == 0);

  // Sanity check
#if MOCHI_ASSERT_VERBOSE_ENABLED
  int const numNodes = mesh.mesh->GetNumNodes();
  for (int i = 0; i < isize(inWorldBC->poseIndices); i += 3) {
    int iDof = inWorldBC->poseIndices[i];
    int iNode = iDof / 3;
    MOCHI_ASSERT_VERBOSE(iDof % 3 == 0, "Expected 3 DOF indices per node");
    MOCHI_ASSERT_VERBOSE(iNode < numNodes, "Index out-of-range");
  }
#endif

  // Copy BC DOF indices
  outLocalBC.poseIndices = inWorldBC->poseIndices;
  outLocalBC.dofIndices = inWorldBC->dofIndices;
  outLocalBC.colValueIndices = inWorldBC->colValueIndices;

  // DotVecMat(vec, matrixTranspose) is the fastest way to transform points.
  TransformRT localFromWorld = Invert(root.worldFromLocal);
  VMatrix4x4r worldToLocalT = ToVMatrix4x4Transpose(localFromWorld);

  // SNLE BC stores BC values as displacements, but we store BC positions.
  // Subtract our positions from the reference pose to update SNLE BC.
  outLocalBC.poseValues.resize_noinit(inWorldBC->poseIndices.size());
  Span<real const> refPositions = Flatten(mesh.mesh->GetNodeCoordinates());
  for (int i = 0; i < isize(inWorldBC->poseValues); i += 3) {
    Vec4r posWorld = ToSimdPoint(Load<3, Vec4r>(&inWorldBC->poseValues[i]));
    Vec4r posLocal = DotVecMat4x4(posWorld, worldToLocalT);
    Vec4r posRef = Load<3, Vec4r>(&refPositions[inWorldBC->poseIndices[i]]);
    Store<3>(&outLocalBC.poseValues[i], posLocal - posRef);
  }
}

void mochi::PreStepDeformableActorAsync(entt::registry& reg, entt::entity e) {
  MOCHI_PROFILE_SCOPE();
  // Called on an ASYNC WORKER TREAD to prepare a specific soft actor
  // for the simulation step. DO NOT ADD/REMOVE COMPONENTS AT THIS TIME.

  ecs::InvokeOnEntity(&UpdateDirichletBC, reg, e);
}

static real ComputeMaxContactSkinSpeed(
    CContactSkinningData const& contactSkinning,
    ColumnVectorView<real const> velocity) {
  auto const& jacobian = contactSkinning.jacobian;
  MOCHI_ASSERT_VERBOSE(
      jacobian.Cols() == velocity.Rows(),
      "Contact-skin Jacobian and velocity dimensions must match.");

  real maxSpeedSqr = 0_r;
  for (int row = 0; row < jacobian.Rows(); ++row) {
    Vec4r skinVelocity{};
    auto const indices = jacobian.Indices(row);
    auto const values = jacobian.Values(row);
    for (int i = 0; i < isize(indices); ++i) {
      skinVelocity += velocity[indices[i]] * ToSimd(values[i]);
    }
    maxSpeedSqr = Max(maxSpeedSqr, NormSqr<3>(skinVelocity));
  }
  return Sqrt(maxSpeedSqr);
}

void deformable::UpdateMaxGeometrySpeed(
    ecs::Included<TagDeformableActor>,
    ecs::Excluded<TagNestedSoftActor, TagRodActor>,
    CVelocitySlice<real, TimeStep::Current> const& velocity,
    CColliderInfo const& collider,
    CContactSkinningData const* contactSkinning,
    CConservativeStepBounds& outStepBounds) {
  // Match tight bounds: skin-only actors use the skin, while colliders also use the physics mesh.
  real const physicsMaxSpeed = (collider.type != ColliderType::None || !contactSkinning)
      ? MaxPackedVector3Norm<kSpaceDim3>(velocity.value.GetConstSpan())
      : 0_r;
  real const contactSkinMaxSpeed =
      contactSkinning ? ComputeMaxContactSkinSpeed(*contactSkinning, velocity.value) : 0_r;
  outStepBounds.maxGeometrySpeed = Max(physicsMaxSpeed, contactSkinMaxSpeed);
}

int deformable::FindRecenteringPivotNode(Span<Real3 const> referenceNodes) {
  MOCHI_ASSERT(!referenceNodes.empty());
  Real3 centroid{};
  for (Real3 const& node : referenceNodes) {
    centroid += node;
  }
  centroid = centroid * (1_r / static_cast<real>(referenceNodes.size()));
  int closest = 0;
  real closestDistSqr = NormSqr(referenceNodes[0] - centroid);
  for (int i = 1; i < isize(referenceNodes); ++i) {
    real const distSqr = NormSqr(referenceNodes[i] - centroid);
    if (distSqr < closestDistSqr) {
      closest = i;
      closestDistSqr = distSqr;
    }
  }
  return closest;
}

std::optional<Real3> deformable::TranslateRootForRecentering(
    CRecenteringParams const& params,
    Real3 const& pivotDisplacement,
    CRootTransform& root,
    CBoundingVolume* bounds) {
  if (!params.useRecentering || NearEqual(Real3{}, pivotDisplacement, params.translationEpsilon)) {
    return std::nullopt;
  }
  Real3 const oldTranslation = root.worldFromLocal.GetTranslation();
  root.worldFromLocal.SetTranslation(root.worldFromLocal.TransformPoint(pivotDisplacement));
  // Use the stored (rounded) translation change so that world-space positions are preserved,
  // rather than accumulating round-off with every recentering.
  Real3 const shift = -root.worldFromLocal.TransformDirectionInverse(
      root.worldFromLocal.GetTranslation() - oldTranslation);

  // The previous-step state is translated too, so move its root transform to keep it fixed in world
  // space.
  TransformRT const newLocalFromOldLocal{Quaternion::Identity(), shift};
  root.worldFromLocalPrev = root.worldFromLocalPrev * Invert(newLocalFromOldLocal);
  if (bounds) {
    bounds->localShape = TransformShape(newLocalFromOldLocal, bounds->localShape);
  }
  return shift;
}

template <bool kIsSync, typename DiscretizationT>
void deformable::SetupActiveCollisionNormals(
    ecs::Excluded<TagShellActor, TagRodActor, TagUseDeformableContactSkin>,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    DiscretizationT const& femDisc,
    CFinalDisplacementRef<TimeStep::Current> const& currentDispl,
    CFinalDisplacementRef<TimeStep::StageStart> const& stageStartDispl,
    CRootTransform const& transform,
    [[maybe_unused]] CContactSamples<TimeStep::Current> const& contactPositions,
    CActiveCollisions<kIsSync, TimeStep::Current>& activeCollisions) {
  MOCHI_PROFILE_SCOPE();
  static_assert(
      std::is_same_v<DiscretizationT, CFemBoundaryDiscretization> ||
          std::is_same_v<DiscretizationT, CFemSurfaceDiscretization>,
      "deformable::SetupActiveCollisionNormals: unsupported discretization type.");

  MOCHI_ASSERT_VERBOSE(
      !contactPositions.normals.has_value(),
      "Expected no pre-computed normals for a deformable actor");

  auto const explicitNormals = simParams->experimentalEval.explicitNormals;
  ColumnVectorView<real const> displForNormals =
      explicitNormals ? stageStartDispl.value : currentDispl.value;

  details::SetupActiveCollisionNormalsFromElements(
      explicitNormals,
      transform,
      femDisc,
      activeCollisions,
      [&](auto const& discretizationImpl, int elementIndex, int quadPointIndex) -> Vec4r {
        using DiscretizationImplT = std::decay_t<decltype(discretizationImpl)>;
        static int constexpr kNumEleNodes = DiscretizationImplT::kNumEleNodes;
        auto const& element = discretizationImpl.femElements[elementIndex];
        auto const baseElementDofIndices = kSpaceDim3 * element.GetBaseElement().Nodes();
        NdArray<Vec4r, kNumEleNodes> nodeCoords;
        for (int j = 0; j < kNumEleNodes; ++j) {
          if constexpr (std::is_same_v<DiscretizationT, CFemBoundaryDiscretization>) {
            nodeCoords[j] = ToSimd(element.nodesCrdsPhys[j]);
          } else {
            nodeCoords[j] = ToSimd(element.coordinates[element.Nodes()[j]]);
          }
          nodeCoords[j] += Load<3, Vec4r>(&displForNormals[baseElementDofIndices[j]]);
        }

        // Compute the surface normal in the colliding actor's local frame. Linear elements have a
        // constant normal across their quadrature points.
        Vec4r normalColliding;
        if constexpr (std::is_same_v<DiscretizationT, CFemBoundaryDiscretization>) {
          // Tetrahedral trace: evaluate the deformed normal from the tetrahedral map.
          Vec4r vmap;
          VMatrix3x3r vdmap;
          element.QuadraturePointEvaluateMap(quadPointIndex, nodeCoords, vmap, vdmap);
          real const det = Det3x3(vdmap);
          real unused = 0_r;
          element.QuadraturePointEvaluateWeightNormal(
              quadPointIndex, det, Invert3x3(vdmap, det), unused, normalColliding);
        } else {
          // Triangular surface: evaluate the deformed tangent map as in Pk2DElement::Initialize.
          using ElementImplT = std::decay_t<decltype(element)>;
          Vec4r tangent0{};
          Vec4r tangent1{};
          for (int f = 0; f < kNumEleNodes; ++f) {
            auto const& dBasis =
                ElementImplT::kBasisEvaluatedParametric.kDBasisEvaluated[quadPointIndex][f];
            tangent0 += nodeCoords[f] * dBasis[0];
            tangent1 += nodeCoords[f] * dBasis[1];
          }
          normalColliding = Normalize<3>(Cross3(tangent0, tangent1));
        }
        return normalColliding;
      });
}

#define MOCHI_SETUP_ACTIVE_COLLISIONS_KINEMATICS_INST(IS_SYNC, DISCRETIZATION_TYPE)    \
  template void deformable::SetupActiveCollisionNormals<IS_SYNC, DISCRETIZATION_TYPE>( \
      ecs::Excluded<TagShellActor, TagRodActor, TagUseDeformableContactSkin>,          \
      ecs::CtxGlobal<CSimulationParams const>,                                         \
      DISCRETIZATION_TYPE const&,                                                      \
      CFinalDisplacementRef<TimeStep::Current> const&,                                 \
      CFinalDisplacementRef<TimeStep::StageStart> const&,                              \
      CRootTransform const&,                                                           \
      CContactSamples<TimeStep::Current> const&,                                       \
      CActiveCollisions<IS_SYNC, TimeStep::Current>&);
MOCHI_SETUP_ACTIVE_COLLISIONS_KINEMATICS_INST(/*kIsSync*/ false, CFemBoundaryDiscretization);
MOCHI_SETUP_ACTIVE_COLLISIONS_KINEMATICS_INST(/*kIsSync*/ true, CFemBoundaryDiscretization);
MOCHI_SETUP_ACTIVE_COLLISIONS_KINEMATICS_INST(/*kIsSync*/ false, CFemSurfaceDiscretization);
MOCHI_SETUP_ACTIVE_COLLISIONS_KINEMATICS_INST(/*kIsSync*/ true, CFemSurfaceDiscretization);
#undef MOCHI_SETUP_ACTIVE_COLLISIONS_KINEMATICS_INST

template <typename ElementT>
static void ComputeAsyncContactResponseImpl(
    ContactAssemblyReg reg,
    entt::entity e,
    ecs::OptionalTag<TagQueryActiveContacts> queryActiveContacts,
    [[maybe_unused]] FemDiscretization<ElementT> const& discretization,
    TransformRT const& transform,
    [[maybe_unused]] CContactSamples<TimeStep::Current> const& samples,
    ContactEvalConfig const& config,
    real dtStage,
    CActiveCollisions</*kIsSync*/ false, TimeStep::Current>& collisions,
    CDeformablePointAsyncCollisionsResponse& outResponse,
    bool evalEner,
    bool evalGrad,
    bool evalHess,
    Span<bool const> allowedContactElementMask) {
  MOCHI_PROFILE_SCOPE();

  static int constexpr kSpaceDim = FemDiscretization<ElementT>::kSpaceDim;
  static_assert(kSpaceDim == 3, "Invalid spatial dimensions");

  int const numColliders = isize(collisions);

  // Sanitize sample points - Make sure we're not using the wrong discretization for this!
  int const numContactElements = isize(discretization.femElements);
  MOCHI_ASSERT_VERBOSE(
      isize(samples.positions) == numContactElements * FemDiscretization<ElementT>::kNumQuads,
      "Invalid number of sample points");
  MOCHI_ASSERT_VERBOSE(
      allowedContactElementMask.empty() || isize(allowedContactElementMask) == numContactElements,
      "Allowed contact element mask size must match the number of contact elements.");

  // Transformation from sample space to world space.
  auto const worldFromLocalRotation = ToVMatrix3x3(transform.GetRotation());

  // Reset contact response samples.
  outResponse.Reset(numContactElements, FemDiscretization<ElementT>::kNumQuads);

  // Preallocate memory for collision response. Use stack memory if possible. 32 KiB is enough in
  // most cases.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 32 * 1024);
  CollisionResponseResult collisionResponse(&allocator);
  collisionResponse.Reserve(collisions, evalEner, evalGrad, evalHess);

  // Compute collision response.
  for (int c = 0; c < numColliders; ++c) {
    auto contactParams = GetContactPairParams(reg, e, collisions[c].colliderEntity);
    ContactDetectionResult& contactQuery = collisions[c].collisionResult;
    if (contactQuery.sampleIndices.empty()) {
      continue;
    }

    int const numActiveSamples = isize(contactQuery.sampleIndices);
    collisionResponse.ResizeNoInit(numActiveSamples, evalEner, evalGrad, evalHess);

    // Add the transformation between collider and world space. The response is the gradient wrt the
    // current contact points, so map it back through the current collider frame even when explicit
    // normals take the friction plane from the stage-start frame.
    auto const& colliderFromWorld = contactQuery.jacColliderFromWorld;
    MOCHI_ASSERT(
        colliderFromWorld.size() == 1,
        "Collider transform must be shared by all sample points for async contact.");
    auto const colliderFromLocal = Dot3x3(colliderFromWorld[0], worldFromLocalRotation);

    ComputeCollisionResponse<GradTarget::Current>(
        contactQuery,
        contactParams,
        config,
        dtStage,
        evalEner,
        evalGrad,
        evalHess,
        collisionResponse);

    // Assemble
    for (size_t i = 0; i < numActiveSamples; ++i) {
      // Fetch the response and transform to sample space.
      double* outEnergy = evalEner ? &collisionResponse.energy[i] : nullptr;
      Real3* outForce = nullptr;
      if (evalGrad) {
        outForce = &collisionResponse.force[i];
        *outForce = TransformContactForceToCollidingLocal(colliderFromLocal, *outForce);
      }
      VMatrix3x3r* outDForce = nullptr;
      if (evalHess) {
        outDForce = &collisionResponse.dforce[i];
        *outDForce = TransformContactDForceToCollidingLocal(colliderFromLocal, *outDForce);
      }

      // Write energy response to buffer.
      int const sampleIndex = contactQuery.sampleIndices[i];
      outResponse.AddContactSampleResponse(sampleIndex, outEnergy, outForce, outDForce);
    }

    // Optionally store data for queries
    if (evalGrad && queryActiveContacts) {
      contactQuery.forcePerUnitArea = collisionResponse.force;
    }
  }

  outResponse.ValidateInvariants(allowedContactElementMask);
}

template <typename DiscretizationType, int kNumFields>
void deformable::ComputeAsyncContactResponse(
    ContactAssemblyReg reg,
    entt::entity e,
    ExperimentalEvalParams const& experimentalEval,
    ecs::OptionalTag<TagQueryActiveContacts> queryActiveContacts,
    DiscretizationType const& femBoundaryDisc,
    CContactSamples<TimeStep::Current> const& samples,
    CActiveCollisions</*kIsSync*/ false, TimeStep::Current>& collisions,
    CTimeIntegratorState const& intState,
    CRootTransform const& rootTransform,
    AssemblyParams const& params,
    CDeformablePointAsyncCollisionsResponse& outResponse,
    Span<bool const> allowedContactElementMask) {
  femBoundaryDisc.Visit([&](auto const& femBoundaryDiscImpl) {
    ContactEvalConfig const config = MakeContactEvalConfigAsync(
        experimentalEval, reg, e, params.psdDRes, params.fittedSaturationHessian.contactFriction);

    ComputeAsyncContactResponseImpl(
        reg,
        e,
        queryActiveContacts,
        femBoundaryDiscImpl,
        rootTransform.worldFromLocal,
        samples,
        config,
        intState.dtStage,
        collisions,
        outResponse,
        params.assemObj,
        params.assemRes,
        params.assemDRes,
        allowedContactElementMask);
  });
}

#define MOCHI_COMPUTE_ASYNC_CONTACT_RESPONSE_INST(DISCRETIZATION_TYPE, NUM_FIELDS)        \
  template void deformable::ComputeAsyncContactResponse<DISCRETIZATION_TYPE, NUM_FIELDS>( \
      ContactAssemblyReg,                                                                 \
      entt::entity,                                                                       \
      ExperimentalEvalParams const&,                                                      \
      ecs::OptionalTag<TagQueryActiveContacts>,                                           \
      DISCRETIZATION_TYPE const&,                                                         \
      CContactSamples<TimeStep::Current> const&,                                          \
      CActiveCollisions</*kIsSync*/ false, TimeStep::Current>&,                           \
      CTimeIntegratorState const&,                                                        \
      CRootTransform const&,                                                              \
      AssemblyParams const&,                                                              \
      CDeformablePointAsyncCollisionsResponse&,                                           \
      Span<bool const>);
MOCHI_COMPUTE_ASYNC_CONTACT_RESPONSE_INST(CFemBoundaryDiscretization, 3);
MOCHI_COMPUTE_ASYNC_CONTACT_RESPONSE_INST(CFemSurfaceDiscretization, 3);
MOCHI_COMPUTE_ASYNC_CONTACT_RESPONSE_INST(CFemSegmentDiscretization, 4);
#undef MOCHI_COMPUTE_ASYNC_CONTACT_RESPONSE_INST

template <typename ActorTag, typename DiscretizationType>
void deformable::SetupCollidingJacobians(
    ecs::Included<ActorTag>,
    ecs::Excluded<TagRomActor, TagNestedSoftActor, TagUseDeformableContactSkin>,
    DiscretizationType const& discretization,
    CRootTransform const& transform,
    CDofOffset const& dofOffset,
    CCollJacs<CollRole::Colliding>& outJacobians) {
  MOCHI_PROFILE_SCOPE();

  int constexpr kNumFields = std::is_same_v<ActorTag, TagRodActor> ? 4 : 3;

  discretization.Visit([&](auto const& discretizationImpl) {
    using DiscretizationT = std::decay_t<decltype(discretizationImpl)>;

    // Define shared differentiable maps
    DMapDeformable<kNumFields> dsoft(0, dofOffset.dofsOffset);
    DMapRTConst dtransform(transform.worldFromLocal);

    // Find all the Sync jacs
    MOCHI_FILO_STACK_ALLOCATOR(tempAlloc, 256 * sizeof(JacData*)); // Probably more than enough
    DynamicArray<JacData*> syncJacs(&tempAlloc);
    syncJacs.reserve(outJacobians.size());
    for (int i = 0; i < isize(outJacobians); ++i) {
      if (outJacobians[i].isSync) {
        syncJacs.push_back(&outJacobians[i]);
      }
    }

    // Compute Jacobians (in parallel if there are several)
    int constexpr kMinPerTask = 2;
    ParallelForEach(
        "deformable::SetupCollidingJacobians Range", syncJacs, kMinPerTask, [&](JacData* jac) {
          MOCHI_ASSERT_VERBOSE(jac->isSync);

          // Create differentiable map
          using DQuad = DMapQuad<typename DiscretizationT::ElementT>;
          DQuad dquad(discretizationImpl.femElements, jac->query->jacColliderFromWorld);
          DMap<DQuad, DMapRTConst, DMapDeformable<kNumFields>> dmap(&dquad, &dtransform, &dsoft);

          auto jacs = jac->PrepareJacs(1);
          dmap.GetJac(jac->query->sampleIndices, jacs, jac->JacScratch());
          jacs[0].CompressIndices();
        });
  });
}

#define MOCHI_SETUP_COLLIDING_JACOBIANS_INST(ACTOR_TAG, DISCRETIZATION_TYPE)         \
  template void deformable::SetupCollidingJacobians<ACTOR_TAG, DISCRETIZATION_TYPE>( \
      ecs::Included<ACTOR_TAG>,                                                      \
      ecs::Excluded<TagRomActor, TagNestedSoftActor, TagUseDeformableContactSkin>,   \
      DISCRETIZATION_TYPE const& discretization,                                     \
      CRootTransform const& transform,                                               \
      CDofOffset const& dofOffset,                                                   \
      CCollJacs<CollRole::Colliding>& outJacobians);
MOCHI_SETUP_COLLIDING_JACOBIANS_INST(TagSoftActor, CFemBoundaryDiscretization);
MOCHI_SETUP_COLLIDING_JACOBIANS_INST(TagShellActor, CFemSurfaceDiscretization);
MOCHI_SETUP_COLLIDING_JACOBIANS_INST(TagRodActor, CFemSegmentDiscretization);

void deformable::SetupColliderJacobians(
    [[maybe_unused]] ecs::OptionalTag<TagSoftActor> isSoftActor,
    [[maybe_unused]] ecs::OptionalTag<TagShellActor> isShellActor,
    [[maybe_unused]] ecs::OptionalTag<TagRodActor> isRodActor,
    ecs::OptionalTag<TagRomActor> isRomActor,
    CDofOffset const& dofOffset,
    CCollJacs<CollRole::Collider>& outJacobians) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(
      isSoftActor || isShellActor || isRodActor,
      "Invalid actor type for deformable::SetupColliderJacobians.");

  // Create differentiable map
  DMapInverse dinvmap(0, dofOffset.dofsOffset, isRomActor);
  DMap<DMapInverse> dmap(&dinvmap);

  // Compute Jacobians
  for (auto& jac : outJacobians) {
    dinvmap.SetData(jac.query);
    auto jacs = jac.PrepareJacs(1);
    dmap.GetJac({}, jacs);
    jacs[0].CompressIndices();
  }
}

void deformable::EmplaceContactComponents(
    entt::registry& reg,
    entt::entity e,
    int numCollidingSamples,
    int numContactPartitions) {
  reg.emplace<TagUseContact>(e);
  reg.emplace<CConservativePotentialColliders</*kIsSync*/ false>>(e);
  reg.emplace<CConservativePotentialColliders</*kIsSync*/ true>>(e);
  reg.emplace<CPotentialColliders</*kIsSync*/ false>>(e);
  reg.emplace<CPotentialColliders</*kIsSync*/ true>>(e);
  reg.emplace<CContactSamples<TimeStep::Current>>(e, numCollidingSamples);
  reg.emplace<CContactSamples<TimeStep::StageStart>>(e, numCollidingSamples);
  if (reg.ctx<CSimulationParams const>().experimentalEval.elastoPlasticFriction) {
    EmplaceFrictionAnchors(reg, e, numCollidingSamples);
  }
  reg.emplace<CContactCorrespondence</*kIsSync*/ false>>(e, numCollidingSamples);
  reg.emplace<CContactCorrespondence</*kIsSync*/ true>>(e, numCollidingSamples);
  reg.emplace<CActiveCollisions</*kIsSync*/ false, TimeStep::Current>>(e, numContactPartitions);
  reg.emplace<CActiveCollisions</*kIsSync*/ false, TimeStep::StageStart>>(e, numContactPartitions);
  reg.emplace<CActiveCollisions</*kIsSync*/ true, TimeStep::Current>>(e, numContactPartitions);
  reg.emplace<CActiveCollisions</*kIsSync*/ true, TimeStep::StageStart>>(e, numContactPartitions);
  reg.emplace<CCollJacs<CollRole::Colliding>>(e);
}

void mochi::deformable::RecordRigidTransformEval(
    CRigidTransformEval const& eval,
    CRecordingData& outData) {
  WriteTransformAttributes("pivotEvalTranslation", "pivotEvalRotation", eval.value, outData);
}

void mochi::deformable::RecordState(
    CDisplacementSlice<real, TimeStep::Current> const* disp,
    CVelocitySlice<real, TimeStep::Current> const& vel,
    CDisplacementSlice<real, TimeStep::Current, DisplacementLayer::Skinned> const* dispSkinned,
    CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned> const* velSkinned,
    CIntegrationVelocitySlices<DisplacementLayer::Skinned> const* integrationVelSkinned,
    CRodPose<TimeStep::Current> const* rodPose,
    [[maybe_unused]] ecs::OptionalTag<TagSoftActor> isSoft,
    [[maybe_unused]] ecs::OptionalTag<TagShellActor> isShell,
    ecs::OptionalTag<TagRodActor> isRod,
    CRecordingData& outData) {
  MOCHI_ASSERT_VERBOSE(
      (isSoft.hasTag + isShell.hasTag + isRod.hasTag) == 1,
      "Deformable recording requires exactly one deformable actor tag");

  // Get displacement data from either the displacement slice or the rod pose
  Span<real const> dispSpan;
  if (disp) {
    dispSpan = disp->value.GetConstSpan();
  } else if (rodPose) {
    dispSpan = rodPose->value.displacements.GetConstSpan();
  } else {
    MOCHI_ASSERT(false, "Expected either CDisplacementSlice or CRodPose");
    return;
  }

  int const dofsPerNode = isRod ? 4 : 3;
  MOCHI_ASSERT(
      isize(dispSpan) % dofsPerNode == 0, "Unexpected number of displacement values per node");
  int const numNodes = isize(dispSpan) / dofsPerNode;
  // Displacements of the default layer
  int const dims[2] = {numNodes, dofsPerNode};
  RecordDataset("displacement", dims, dispSpan, outData);

  // Velocities of the default layer
  MOCHI_ASSERT(
      vel.value.size() % dofsPerNode == 0, "Unexpected number of velocity values per node");
  MOCHI_ASSERT(isize(vel.value) == isize(dispSpan), "Displacement/velocity size mismatch");
  RecordDataset("velocity", dims, vel.value.GetConstSpan(), outData);

  // Displacements of the skinned layer
  if (dispSkinned) {
    MOCHI_ASSERT_VERBOSE(isSoft, "Skinned displacements are only supported for soft actors");
    MOCHI_ASSERT(dispSkinned->value.size() % 3 == 0, "Expected 3 values per node");
    MOCHI_ASSERT(isize(dispSkinned->value) == isize(dispSpan), "Size mismatch");
    RecordDataset("displacementSkinned", dims, dispSkinned->value.GetConstSpan(), outData);
  }

  // Velocities of the skinned layer (only if state)
  if (velSkinned && integrationVelSkinned) {
    MOCHI_ASSERT_VERBOSE(isSoft, "Skinned velocities are only supported for soft actors");
    MOCHI_ASSERT(velSkinned->value.size() % 3 == 0, "Expected 3 values per node");
    MOCHI_ASSERT(isize(velSkinned->value) == isize(dispSpan), "Size mismatch");
    RecordDataset("velocitySkinned", dims, velSkinned->value.GetConstSpan(), outData);
  }

  // Element frame axes (rod actors only — read from CRodPose<Current>)
  if (rodPose) {
    MOCHI_ASSERT_VERBOSE(isRod, "Rod pose is only supported for rod actors");
    int const numElements = isize(rodPose->value.frameAxes);
    MOCHI_ASSERT_VERBOSE(
        numElements == numNodes - 1 || numElements == numNodes,
        "Element/node number mismatch for rod actor");
    int const axesDims[2] = {numElements, 3};
    RecordDataset(
        "elementFrameAxes", axesDims, Flatten(MakeSpan(rodPose->value.frameAxes)), outData);
  }
}

void mochi::deformable::RecordingPipeline(entt::registry& reg, Span<entt::entity const> entities) {
  ecs::InvokeForEach(&RecordState, reg, entities);
  ecs::InvokeForEach(&RecordRigidTransformEval, reg, entities);
}
