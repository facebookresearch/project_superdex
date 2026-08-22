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

#include <mochi_physics/diffsim/mochi_diffsim.h>
#include <mochi_physics/src/mochi_articulated_body.h>
#include <mochi_physics/src/mochi_differentiable.h>
#include <mochi_physics/src/mochi_rigid.h>
#include <mochi_physics/src/mochi_solve.h>
#include <mochi_physics/src/mochi_step.h>
#include <mochi_physics/utils/mochi_prefab.h>
#include "mochi_autodiff.h"
#include "mochi_physics_test_fixture.h"

using namespace mochi;
using namespace mochi::solver;
using namespace mochi::experimental;
using namespace mochi::diffsim;
using namespace mochi::test;

#if MOCHI_USE_DOUBLE_PRECISION && MOCHI_USE_EIGEN
using namespace mochi::autodiff;
#define MOCHI_USE_DOUBLE_AND_EIGEN 1
#else
#define MOCHI_USE_DOUBLE_AND_EIGEN 0
#endif

// The scene assets used by these tests are not shipped externally.
#if MOCHI_USE_DOUBLE_AND_EIGEN && MOCHI_INTERNAL
#define MOCHI_USE_DOUBLE_EIGEN_AND_INTERNAL 1
#else
#define MOCHI_USE_DOUBLE_EIGEN_AND_INTERNAL 0
#endif

namespace {
Real3 RandomVector(auto& generator) {
  return {
      mochi::RandomUniformValue(generator, -1_r, 1_r),
      mochi::RandomUniformValue(generator, -1_r, 1_r),
      mochi::RandomUniformValue(generator, -1_r, 1_r),
  };
}

// *****************************************************************************
// Finite different API: In a future diff, these functions will appear in the mochi_physics source
// code for finite-difference state- and control-derivatives estimator used by the LQR controller.
// We copy paste them here so that we can compare the finite difference value with analytic values.
void RestoreKinematicStatePair(
    Scene* scene,
    double timeStepSec,
    StateHandle stateCurr,
    StateHandle stateOld,
    std::optional<std::function<void()>> callbackCurr,
    std::optional<std::function<void()>> callbackOld,
    Error& error) {
  entt::registry& reg = assert_cast<SceneImpl*>(scene)->GetRegistry();

  MOCHI_ERROR_IF(!stateCurr.IsValid(), error, "Current state must be valid")
  MOCHI_ERROR_RETURN(error);

  // If only the current state is valid, we just restore
  if (!stateOld.IsValid()) {
    scene->RestoreState(stateCurr, false, error);
    return;
  }
  // Otherwise, we have to recover state from two kinematic poses

  // TimeStep::Current (pose,vel) <- stateOld
  // We do not want velocity, that part of computation is wasted
  scene->RestoreState(stateOld, false, error);
  MOCHI_ERROR_RETURN(error);
  if (callbackOld && *callbackOld) {
    // Optionally modify the old pose
    (*callbackOld)();
  }

  // Advance time and pre-step ECS, so that:
  // TimeStep::Previous (pose,vel) <- TimeStep::Current (pose,vel)
  // We do not want velocity, that part of computation is wasted
  reg.ctx<CSceneTime>().Advance(timeStepSec);
  mochi::PreStepEcs(reg);

  // Save this parameter and restore later, we need to force creating a single island below
  auto savedUseSingleIsland = scene->GetForceSingleIsland();

  // TimeStep::StageStart (pose) <- TimeStep::Current (pose)
  scene->SetForceSingleIsland(true);
  island::PreStep(reg); // In case no island exists
  reg.view<CIslandDescendants const>().each(
      [&](entt::entity /*island*/, CIslandDescendants const& descendants) {
        auto const integrationParams =
            CreateIslandTimeIntegrationParams(reg, descendants, IntegrationMethod::BackwardEuler);
        SetTimeIntegratorState(reg, descendants.actors, integrationParams, 0);
        // Run PreFirstStage pipeline, but only for rigid and articulated body
        ecs::InvokeForEach(&mochi::rigid::EntityPreFirstStage, reg, descendants.rigidActors);
        if (!descendants.compoundActors.empty()) {
          ecs::InvokeForEach(
              &mochi::articulated::compound::EntityPreFirstStage, reg, descendants.compoundActors);
        }
        // Run PreStage pipeline, but only for rigid and articulated body
        ecs::InvokeForEach(&mochi::rigid::EntityPreStage, reg, descendants.rigidActors);
        if (!descendants.compoundActors.empty()) {
          mochi::articulated::compound::PreStagePipeline(reg, descendants.actors);
        }
      });

  // TimeStep::Current (pose,vel) <- stateCurr
  // We do not want velocity, that part of computation is wasted
  scene->RestoreState(stateCurr, false, error);
  MOCHI_ERROR_RETURN(error);
  if (callbackCurr && *callbackCurr) {
    // Optionally modify the old pose
    (*callbackCurr)();
  }

  // Call PostStageLocalPipeline, so that:
  // TimeStep::Current (vel) <- finite-difference(stateCurr, stateOld)
  scene->SetForceSingleIsland(true);
  island::PreStep(reg); // In case no island exists
  reg.view<CIslandDescendants const>().each(
      [&](entt::entity /*island*/, CIslandDescendants const& descendants) {
        // Run PreFirstStage pipeline, but only for rigid and articulated body
        auto const integrationParams =
            CreateIslandTimeIntegrationParams(reg, descendants, IntegrationMethod::BackwardEuler);
        SetTimeIntegratorState(reg, descendants.actors, integrationParams, 0);
        // This is because we need to populate CTimeIntegratorState.stages
        ecs::InvokeForEach(&mochi::rigid::EntityPreFirstStage, reg, descendants.rigidActors);
        if (!descendants.compoundActors.empty()) {
          ecs::InvokeForEach(
              &mochi::articulated::compound::EntityPreFirstStage, reg, descendants.compoundActors);
        }
        // Run PreStageLocal pipeline, but only for rigid and articulated body
        mochi::solver::PostStageLocalPipeline(reg, descendants);
      });

  // Restore is original island setting
  scene->SetForceSingleIsland(savedUseSingleIsland);
}

void GetPoseDelta(
    Scene* scene,
    Span<real const> prevPose,
    Span<real const> currPose,
    Span<real> delta,
    Error& error) {
  int offset = 0;
  MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(real) * 9 * 256);
  scene->ForEachActor([&](Actor* actor) {
    if (actor->GetType() == ActorType::Articulated) {
      // Extract the pose
      Span<real const> prevPoseActor(prevPose.data() + offset, actor->GetNumDofs());
      Span<real const> currPoseActor(currPose.data() + offset, actor->GetNumDofs());
      Span<real> deltaPoseActor(delta.data() + offset, actor->GetNumDofs());
      // Compute delta pose
      actor->ComputeArticulatedPoseDelta(prevPoseActor, currPoseActor, deltaPoseActor, error);
    } else if (actor->GetType() == ActorType::Rigid) {
      if (actor->IsNestedLinkActor() || actor->IsStatic()) {
        return;
      }
      // Extract the center of mass transform
      auto t = Load<Vec4r>(prevPose.data() + offset, 3);
      auto r = Load<Vec4r>(prevPose.data() + offset + 3, 3);
      auto t2 = Load<Vec4r>(currPose.data() + offset, 3);
      auto r2 = Load<Vec4r>(currPose.data() + offset + 3, 3);
      auto dt = t2 - t;
      auto dr =
          (Quaternion::FromRotationVector(r2) * Quaternion::FromRotationVector(r).GetConjugate())
              .VToRotationVector();
      // Assign
      ColumnVectorView<real>(delta.data() + offset, 3) = AsColumnVectorView<3>(dt);
      ColumnVectorView<real>(delta.data() + offset + 3, 3) = AsColumnVectorView<3>(dr);
    } else {
      MOCHI_ERROR_SET(error, "GetStepJacobian do not support non-articulated or rigid actors");
    }
    offset += actor->GetNumDofs();
  });
}

[[maybe_unused]]
int GetNumDof(Scene* scene, Error& error) {
  int numDof = 0;
  scene->ForEachActor([&](Actor const* actor) {
    if (actor->GetType() == ActorType::Articulated) {
      numDof += actor->GetNumDofs();
    } else if (actor->GetType() == ActorType::Rigid) {
      if (actor->IsNestedLinkActor() || actor->IsStatic()) {
        return;
      }
      numDof += actor->GetNumDofs();
    } else {
      MOCHI_ERROR_SET(error, "GetStepJacobian do not support non-articulated or rigid actors");
    }
  });
  return numDof;
}

[[maybe_unused]]
int GetOffsetDof(Scene* scene, Actor* actorRef, Error& error) {
  int numDof = 0, offDof = 0;
  scene->ForEachActor([&](Actor const* actor) {
    if (actor->GetType() == ActorType::Articulated) {
      if (actorRef == actor) {
        offDof = numDof;
      }
      numDof += actor->GetNumDofs();
    } else if (actor->GetType() == ActorType::Rigid) {
      if (actor->IsNestedLinkActor() || actor->IsStatic()) {
        return;
      }
      numDof += actor->GetNumDofs();
    } else {
      MOCHI_ERROR_SET(error, "GetStepJacobian do not support non-articulated or rigid actors");
    }
  });
  return offDof;
}

void GetPoseVector(Scene* scene, Span<real> pose, Error& error) {
  int offset = 0;
  scene->ForEachActor([&](Actor const* actor) {
    if (actor->GetType() == ActorType::Articulated) {
      actor->GetArticulatedPose(Span<real>(pose.data() + offset, actor->GetNumDofs()), error);
    } else if (actor->GetType() == ActorType::Rigid) {
      if (actor->IsNestedLinkActor() || actor->IsStatic()) {
        return;
      }
      auto trans = actor->GetCenterOfMassTransform(error);
      ColumnVectorView<real>(pose.data() + offset, 3) = AsConstView(trans.GetTranslation());
      ColumnVectorView<real>(pose.data() + offset + 3, 3) =
          AsConstView(trans.GetRotation().ToRotationVector());
    } else {
      MOCHI_ERROR_SET(error, "GetStepJacobian do not support non-articulated or rigid actors");
    }
    offset += actor->GetNumDofs();
  });
}

void SetPoseVector(Scene* scene, Span<real const> pose, Span<real const> deltaPose, Error& error) {
  MOCHI_ERROR_IF(
      !deltaPose.empty() && isize(deltaPose) != isize(pose),
      error,
      "deltaPose must be empty or has the same size as pose")
  MOCHI_ERROR_RETURN(error)
  MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(real) * 3 * 256);

  int offset = 0;
  scene->ForEachActor([&](Actor* actor) {
    if (actor->GetType() == ActorType::Articulated) {
      if (deltaPose.empty()) {
        actor->SetArticulatedPoseFromJoints(
            Span<real const>(pose.data() + offset, actor->GetNumDofs()), error);
      } else {
        ColumnVector<real> outPose(actor->GetNumDofs(), &allocator);
        actor->AddArticulatedDeltaToPose(pose, deltaPose, outPose, error);
        actor->SetArticulatedPoseFromJoints(outPose, error);
      }
    } else if (actor->GetType() == ActorType::Rigid) {
      if (actor->IsNestedLinkActor() || actor->IsStatic()) {
        return;
      }
      auto t = Load<Vec4r>(pose.data() + offset, 3);
      auto r = Load<Vec4r>(pose.data() + offset + 3, 3);
      TransformRT trans(Quaternion::FromRotationVector(r), t);
      if (!deltaPose.empty()) {
        auto dt = Load<Vec4r>(deltaPose.data() + offset, 3);
        auto dr = Load<Vec4r>(deltaPose.data() + offset + 3, 3);
        trans.SetTranslation(trans.VGetTranslation() + dt);
        trans.SetRotation(Quaternion::FromRotationVector(dr) * trans.GetRotation());
      }
      // We avoid synchornizing the history to only test partial derivatives with the current pose
      actor->SetCenterOfMassTransform(trans, error);
    } else {
      MOCHI_ERROR_SET(error, "GetStepJacobian do not support non-articulated or rigid actors");
    }
    offset += actor->GetNumDofs();
  });
}

void StepJacobianFiniteDifference(
    Scene* scene,
    double timeStepSec,
    double epsilon,
    bool useCentralDiff,
    StateHandle stateNew,
    StateHandle stateCurr,
    StateHandle stateOld,
    std::optional<std::function<void(int, real)>> shiftControl,
    MatrixView<real> jacControl,
    MatrixView<real> jacCurr,
    MatrixView<real> jacOld,
    Error& error) {
  // Count the total degree of freedom
  int numDOF = GetNumDof(scene, error);
  MOCHI_ERROR_RETURN(error);

  // Check the size of Jacobian
  MOCHI_ERROR_IF(
      !jacCurr.empty() && (jacCurr.Rows() != numDOF || jacCurr.Cols() != numDOF),
      error,
      "Incorrect size of current Jacobian matrix");
  MOCHI_ERROR_IF(
      !jacOld.empty() && (jacOld.Rows() != numDOF || jacOld.Cols() != numDOF),
      error,
      "Incorrect size of old Jacobian matrix");
  MOCHI_ERROR_RETURN(error);

  // Allocate space for pose and get the pose at stateNew
  MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(real) * 15 * 256);
  DynamicArray<real> refPose(numDOF, &allocator);
  DynamicArray<real> refPoseOld(numDOF, &allocator);
  DynamicArray<real> deltaPose(numDOF, &allocator);
  DynamicArray<real> deltaPoseP(numDOF, &allocator);
  DynamicArray<real> deltaPoseN(numDOF, &allocator);
  scene->RestoreState(stateNew, false, error);
  GetPoseVector(scene, refPose, error);
  MOCHI_ERROR_RETURN(error);

  // Compute state derivative
  // Do the same for jacCurr and jacOld
  real eps = 0_r;
  AsView(deltaPose).SetZero();
  std::optional<std::function<void()>> cbCurr, cbOld;
  for (MatrixView<real>* jac : {&jacCurr, &jacOld}) {
    for (int c = 0; c < jac->Cols(); c++) {
      auto shiftPose = [&] {
        GetPoseVector(scene, refPoseOld, error);
        deltaPose[c] = eps;
        SetPoseVector(scene, refPoseOld, deltaPose, error);
      };
      if (jac == &jacCurr) {
        cbCurr = shiftPose;
        cbOld = {};
      } else {
        cbCurr = {};
        cbOld = shiftPose;
      }
      // P
      eps = epsilon;
      RestoreKinematicStatePair(scene, timeStepSec, stateCurr, stateOld, cbCurr, cbOld, error);
      MOCHI_ERROR_RETURN(error);
      scene->Step(timeStepSec);
      GetPoseVector(scene, deltaPoseP, error);
      if (useCentralDiff) {
        // N
        eps = -epsilon;
        RestoreKinematicStatePair(scene, timeStepSec, stateCurr, stateOld, cbCurr, cbOld, error);
        MOCHI_ERROR_RETURN(error);
        scene->Step(timeStepSec);
        GetPoseVector(scene, deltaPoseN, error);
        GetPoseDelta(scene, deltaPoseN, deltaPoseP, jac->Col(c).GetSpan(), error);
      } else {
        GetPoseDelta(scene, refPose, deltaPoseP, jac->Col(c).GetSpan(), error);
      }
      MOCHI_ERROR_RETURN(error);
      deltaPose[c] = 0_r;
    }
    *jac /= (useCentralDiff ? 2_r : 1_r) * epsilon;
  }

  // Compute control derivative
  if (shiftControl && *shiftControl && !jacControl.empty()) {
    MOCHI_ERROR_IF(jacControl.Rows() != numDOF, error, "Invalid number of rows in jacControl");
    MOCHI_ERROR_IF(jacControl.Cols() > numDOF, error, "Invalid number of columns in jacControl");
    DynamicArray<real> control(jacControl.Cols(), &allocator);
    for (int c = 0; c < jacControl.Cols(); c++) {
      // P
      eps = epsilon;
      RestoreKinematicStatePair(scene, timeStepSec, stateCurr, stateOld, {}, {}, error);
      MOCHI_ERROR_RETURN(error);
      (*shiftControl)(c, eps);
      scene->Step(timeStepSec);
      GetPoseVector(scene, deltaPoseP, error);
      if (useCentralDiff) {
        // N
        eps = -epsilon;
        RestoreKinematicStatePair(scene, timeStepSec, stateCurr, stateOld, {}, {}, error);
        MOCHI_ERROR_RETURN(error);
        (*shiftControl)(c, eps);
        scene->Step(timeStepSec);
        GetPoseVector(scene, deltaPoseN, error);
        GetPoseDelta(scene, deltaPoseN, deltaPoseP, jacControl.Col(c).GetSpan(), error);
      } else {
        GetPoseDelta(scene, refPose, deltaPoseP, jacControl.Col(c).GetSpan(), error);
      }
      MOCHI_ERROR_RETURN(error);
      deltaPose[c] = 0_r;
    }
    jacControl /= useCentralDiff ? 2_r * epsilon : epsilon;
  }
}

// *****************************************************************************
// These functions test the (mixed) hessians entry by entry, which is convenient for debugging when
// we incorporate more things into forward propagation (e.g. contact and constraints). These
// functions will stay here and not incorporated into mochi_physics.
[[maybe_unused]]
void AssignResBlk(
    entt::registry& reg,
    CIslandMembers const& members,
    MatrixView<real> grad,
    MatrixView<real const> gradBlk) {
  for (int r = 0, rOff = 0; r < isize(members.actors); r++) {
    int rOffGlobal = reg.get<CSceneStateOffset const>(members.actors[r]).dofsOffset;
    int rDofs = reg.get<CActorDofInfo const>(members.actors[r]).dofsSize;
    grad.MiddleRows(rOffGlobal, rDofs) = gradBlk.MiddleRows(rOff, rDofs);
    rOff += rDofs;
  }
}

[[maybe_unused]]
void AssignDResBlk(
    entt::registry& reg,
    CIslandMembers const& members,
    MatrixView<real> dRes,
    MatrixView<real const> dResBlk) {
  for (int r = 0, rOff = 0; r < isize(members.actors); r++) {
    int rOffGlobal = reg.get<CSceneStateOffset const>(members.actors[r]).dofsOffset;
    int rDofs = reg.get<CActorDofInfo const>(members.actors[r]).dofsSize;
    for (int c = 0, cOff = 0; c < isize(members.actors); c++) {
      int cOffGlobal = reg.get<CSceneStateOffset const>(members.actors[c]).dofsOffset;
      int cDofs = reg.get<CActorDofInfo const>(members.actors[c]).dofsSize;
      dRes.Block(rOffGlobal, cOffGlobal, rDofs, cDofs) = dResBlk.Block(rOff, cOff, rDofs, cDofs);
      cOff += cDofs;
    }
    rOff += rDofs;
  }
}

template <GradTarget kGradTarget>
[[maybe_unused]]
void ComputeResDResIslandAsync(
    entt::registry& reg,
    entt::entity island,
    CIslandMembers const& members,
    CIslandDescendants const& descendants,
    real* merit,
    ColumnVectorView<real> res,
    MatrixView<real> dRes) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(!descendants.actors.empty(), "Empty islands should have been pruned");

  // General settings and SNLE forward problem for gradient assembly
  auto const& islandDofInfo = reg.get<CIslandDofInfo>(island);
  int const solutionSize = islandDofInfo.poseSize;
  int const dofsSize = islandDofInfo.dofsSize;

  // Initialize island assemble problem
  SnleProblemFunctions<real> functions;
  functions.assemble = [&](SnleProblem<real>& problem, AssemblyParams const& params) {
    solver::AssembleIslandPipeline(reg, island, params, problem);
  };
  SnleProblem<real> problem(dofsSize, solutionSize, std::move(functions));

  // We need to make sure that backward Euler is used, multi-stage assembly is not supported
  auto const& simParams = reg.ctx<CSimulationParams const>();
  auto const integrationParams =
      CreateIslandTimeIntegrationParams(reg, descendants, simParams.integrationMethod);
  MOCHI_ASSERT(integrationParams.numStages == 1, "Only one stage is supported for assembly");
  SetTimeIntegratorState(reg, descendants.actors, integrationParams, 0);

  // Set the stage-start state
  solver::PreFirstStageLocalPipeline(reg, descendants);
  solver::PreStageLocalPipeline(reg, descendants, problem);

  // Assemble merit
  if (merit) {
    AssemblyParams paramsRes = {
        .assemObj = true,
        .assemRes = false,
        .assemDRes = false,
        .psdDRes = false,
        .gradTarget = GradTarget::Current};
    problem.InvalidateCachedData();
    problem.UpdateObjResDRes(paramsRes);

    *merit += problem.GetObjective();
  }

  // Assemble residual
  if (!res.empty()) {
    MOCHI_ASSERT(kGradTarget == GradTarget::Current);
    AssemblyParams paramsRes = {
        .assemObj = false, .assemRes = true, .assemDRes = false, .psdDRes = false};
    problem.InvalidateCachedData();
    problem.UpdateObjResDRes(paramsRes);

    // Assign to the global vector
    AssignResBlk(reg, members, res, problem.GetResidual());
  }

  // Assemble dResidual
  if (!dRes.empty()) {
    MOCHI_ASSERT(kGradTarget == GradTarget::Current || kGradTarget == GradTarget::Previous);
    auto assemble = [&](SnleProblem<real>& problem, AssemblyParams const& params) {
      solver::AssembleIslandPipeline(reg, island, params, problem);
    };
    problem.SetAssemblyFunction(assemble);

    AssemblyParams paramsRes = {
        .assemObj = false,
        .assemRes = false,
        .assemDRes = true,
        .psdDRes = false,
        .gradTarget = kGradTarget};
    problem.InvalidateCachedData();
    problem.UpdateObjResDRes(paramsRes);

    auto dResBlk = ToMatrix(problem.GetDResidual());
    AssignDResBlk(reg, members, dRes, dResBlk);
  }

  if constexpr (kGradTarget == GradTarget::PreviousDelta) {
    MOCHI_ASSERT(res.empty() && dRes.empty());
    AssemblyParams paramsRes = {
        .assemObj = false,
        .assemRes = false,
        .assemDRes = true,
        .psdDRes = false,
        .gradTarget = kGradTarget};
    problem.InvalidateCachedData();
    problem.UpdateObjResDRes(paramsRes);

    // Compute d2f/dqk/dδ
    ecs::InvokeForEach(ComputeHqx, reg, members.actors, dofsSize, MakeConstSpan(members.actors));
  }
}

template <GradTarget kGradTarget>
[[maybe_unused]]
void ComputeResDRes(Scene* scene, real* merit, ColumnVectorView<real> res, MatrixView<real> dRes) {
  auto& reg = assert_cast<SceneImpl*>(scene)->GetRegistry();
  static_assert(
      kGradTarget == GradTarget::Current || kGradTarget == GradTarget::Previous ||
          kGradTarget == GradTarget::PreviousDelta,
      "Unsupported kGradTarget");

  // Compute scene state offset
  int dofOffset = 0;
  scene->ForEachActor([&](Actor* actor) {
    if (!actor->IsStatic()) {
      int const numDofs = actor->GetNumDofs();
      auto e = GetEntity(reg, actor->GetHandle(), ExpectOK{});
      reg.get<CSceneStateOffset>(e).dofsOffset = dofOffset;
      dofOffset += numDofs;
    }
  });

  // Initialize to zero
  if (merit) {
    *merit = 0_r;
  }
  if (!res.empty()) {
    res.SetZero();
  }
  if (!dRes.empty()) {
    dRes.SetZero();
  }

  reg.view<CIslandMembers const, CIslandDescendants const>().each(
      [&](entt::entity island,
          CIslandMembers const& members,
          CIslandDescendants const& descendants) {
        // The island pre-step operation is also needed for computing ResDRes
        PreStepIslandAsync(reg, descendants);
        // Now compute ResDRes each island
        ComputeResDResIslandAsync<kGradTarget>(reg, island, members, descendants, merit, res, dRes);
        // The island post-step operation is not needed
      });
}
// *****************************************************************************

template <typename MATRIX>
[[maybe_unused]]
void LogMatrix(std::string name, MATRIX const& mat) {
  MOCHI_LOG(
      "// *****************************************************************************%s",
      name.c_str())
  for (int r = 0; r < mat.Rows(); r++) {
    std::string row;
    for (int c = 0; c < mat.Cols(); c++) {
      row += " " + std::to_string(mat(r, c));
    }
    MOCHI_LOG("%s [Row %d]", row.c_str(), r)
  }
  MOCHI_LOG("// *****************************************************************************")
}

void CheckEntryConsistency(
    MatrixView<real const> matFD,
    MatrixView<real const> mat,
    int r,
    int c,
    real tol) {
  real maxNorm = std::max(std::fabs(matFD(r, c)), std::fabs(mat(r, c)));
  real diffNorm = std::fabs(matFD(r, c) - mat(r, c));
  real relNorm = diffNorm / std::max(1e-10_r, maxNorm);
  real absNorm = std::fabs(matFD(r, c) - mat(r, c));
  if (std::min(relNorm, absNorm) > tol) {
    MOCHI_LOG("Inconsistent value at (%d, %d) valFD=%f, val=%f", r, c, matFD(r, c), mat(r, c))
  }
  EXPECT_NEAR_TOL(std::min(relNorm, absNorm), 0_r, tol);
}

void ForewardConsistencyCheck(
    Scene* scene,
    StateHandle stateNew,
    StateHandle stateCurr,
    StateHandle stateOld,
    real dt,
    real epsilon,
    real relTol) {
  // We do not rely on the actor order of creation
  int numDof = 0;
  DynamicArray<int> numDofs;
  DynamicArray<Actor*> actors;
  scene->ForEachActor([&](Actor* a) {
    actors.emplace_back(a);
    numDofs.emplace_back(a->GetNumDofs());
    numDof += a->GetNumDofs();
  });

  // Run forward propagation
  Matrix<real> jacCurr, jacOld;
  jacCurr.Resize(numDof, numDof);
  jacOld.Resize(numDof, numDof);
  GetStepJacobian(scene, stateNew, stateCurr, stateOld, jacCurr, jacOld, test::ExpectOK{});

  // Run forward propagation using finite-difference
  Matrix<real> jacCurrFD, jacOldFD;
  jacCurrFD.Resize(numDof, numDof);
  jacOldFD.Resize(numDof, numDof);
  StepJacobianFiniteDifference(
      scene,
      dt,
      epsilon,
      true,
      stateNew,
      stateCurr,
      stateOld,
      {},
      {},
      jacCurrFD,
      jacOldFD,
      test::ExpectOK{});

  for (int r = 0; r < jacCurr.Rows(); r++) {
    for (int c = 0; c < jacCurr.Cols(); c++) {
      CheckEntryConsistency(jacCurrFD, jacCurr, r, c, relTol);
      CheckEntryConsistency(jacOldFD, jacOld, r, c, relTol);
    }
  }
}

class MochiStepJacobian : public MochiSceneTestBase {
 public:
  void SetUp() override {
    MochiSceneTestBase::SetUp();
    // Back-propagation requires backward Euler.
    test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
    MakeSceneDifferentiable(_scene, test::ExpectOK{});
  }

  void LoadScenePrefab(std::string_view prefabName) {
    // Load the scene from a prefab. Do not apply scene settings or they will conflict with those
    // needed for differentiability.
    prefab::PrefabParams params{.applySceneSettings = false};
    prefab::AddToScene(
        test::GetAssetPath(std::string("differentiability_test/") + std::string(prefabName)),
        test::GetAssetPath(""),
        _scene,
        params,
        test::ExpectOK{});

    // Use more strict parameters
    auto simParams = _scene->GetSolverParams();
    simParams.nonLinearSolver.lineSearchType = LineSearchType::Armijo;
    simParams.nonLinearSolver.maxIter = 15;
    simParams.linearSolver.solverType = LinearSolverType::LDLT;
    simParams.linearSolver.preconditionerType = PreconditionerType::None;
    _scene->SetSolverParams(simParams, test::ExpectOK{});
  }

  void TestStepJacobian(std::string const& sceneName) {
    // Create three cubes
    auto generator = mochi::RandomGenerator(20);
    LoadScenePrefab(sceneName);
    real dt = 0.01_r;
    real velSz = 100_r;
    _scene->Step(dt);

    // Randomly change rotation and translation to get current
    // Velocity is not used but we set to a large value to make sure it is not used
    std::array<StateHandle, 3> statePair; //[new,curr,prev]
    for (int d = 0; d < 2; d++) {
      _scene->ForEachActor([&](Actor* a) {
        if (a->IsStatic()) {
          return;
        }
        TransformRT trans = a->GetCenterOfMassTransform(test::ExpectOK{}) *
            TransformRT(Quaternion::FromRotationVector(RandomVector(generator)),
                        RandomVector(generator) * 0.01_r);
        a->SetCenterOfMassTransform(trans, test::ExpectOK{});
        a->SetVelocity(
            RandomVector(generator) * velSz, RandomVector(generator) * velSz, test::ExpectOK{});
      });
      statePair[d + 1] = _scene->CaptureState(test::ExpectOK{});
    }

    // We capture a new state that is derived from the two kinematic poses (stateOld, stateCurr)
    RestoreKinematicStatePair(_scene, dt, statePair[1], statePair[2], {}, {}, test::ExpectOK{});
    // Re-capture state since it's derived velocity has changed by recomputing using finite
    // difference
    _scene->ReleaseState(statePair[1]);
    statePair[1] = _scene->CaptureState(test::ExpectOK{});
    // Setup the state from two kinematic poses, run step to get a new state
    _scene->Step(dt);
    statePair[0] = _scene->CaptureState(test::ExpectOK{});
    ForewardConsistencyCheck(
        _scene, statePair[0], statePair[1], statePair[2], 1e-2_r, 1e-3_r, 1e-2_r);
  }
};

class MochiStepJacobianAutoDiff : public MochiStepJacobian {
 public:
  void TestHessianAutoDiff([[maybe_unused]] std::string const& sceneName) {
#if MOCHI_USE_DOUBLE_PRECISION && MOCHI_USE_EIGEN
    // Create three cubes
    auto generator = mochi::RandomGenerator(20);
    LoadScenePrefab(sceneName);
    real dt = 0.01_r;
    real velSz = 100_r;
    _scene->Step(dt);

    // Randomly change rotation and translation to get current
    // Velocity is not used but we set to a large value to make sure it is not used
    std::array<StateHandle, 3> statePair; //[new,curr,prev]
    for (int d = 2; d >= 0; d--) {
      _scene->ForEachActor([&](Actor* a) {
        if (a->IsStatic()) {
          return;
        }
        TransformRT trans = a->GetCenterOfMassTransform(test::ExpectOK{}) *
            TransformRT(Quaternion::FromRotationVector(RandomVector(generator)),
                        RandomVector(generator) * 0.01_r);
        a->SetCenterOfMassTransform(trans, test::ExpectOK{});
        a->SetVelocity(
            RandomVector(generator) * velSz, RandomVector(generator) * velSz, test::ExpectOK{});
      });
      statePair[d] = _scene->CaptureState(test::ExpectOK{});
      if (d < 2) {
        // We capture a new state that is derived from the two kinematic poses (stateOld, stateCurr)
        RestoreKinematicStatePair(
            _scene, dt, statePair[d], statePair[d + 1], {}, {}, test::ExpectOK{});
        // Re-capture state: it's derived velocity has changed by recomputing using finite
        // difference
        _scene->ReleaseState(statePair[d]);
        statePair[d] = _scene->CaptureState(test::ExpectOK{});
      }
    }

    // Use analytic method to compute res/dres
    real merit{};
    ColumnVector<real> res(GetNumDof(_scene, test::ExpectOK{}));
    Matrix<real> dResNew(GetNumDof(_scene, test::ExpectOK{}), GetNumDof(_scene, test::ExpectOK{}));
    Matrix<real> dResCurr(GetNumDof(_scene, test::ExpectOK{}), GetNumDof(_scene, test::ExpectOK{}));
    Matrix<real> dResOld(GetNumDof(_scene, test::ExpectOK{}), GetNumDof(_scene, test::ExpectOK{}));
    // Step 1 compute dres/d(q_k+1,q_k,Dx)
    assert_cast<SceneImpl*>(_scene)->RestoreStatePair(statePair[0], statePair[1], test::ExpectOK{});
    ComputeResDRes<GradTarget::Current>(_scene, &merit, res, dResNew);
    ComputeResDRes<GradTarget::Previous>(_scene, nullptr, {}, dResCurr);
    ComputeResDRes<GradTarget::PreviousDelta>(_scene, nullptr, {}, {});
    // Step 2 compute dDx/d(q_k,q_k-1)
    assert_cast<SceneImpl*>(_scene)->RestoreStatePair(statePair[1], statePair[2], test::ExpectOK{});
    StepJacobianShiftAndProject(assert_cast<SceneImpl*>(_scene)->GetRegistry(), dResCurr, dResOld);

    // Use autodiff to compute res/dres
    AutoDiffAssembly ad(_scene, statePair);
    ad.Assemble(dt);
    auto resAd = ad.GetRes();
    auto dResNewAd = ad.GetDRes(0);
    auto dResCurrAd = ad.GetDRes(1);
    auto dResOldAd = ad.GetDRes(2);
    EXPECT_NEAR_RTOL(merit, ad.GetMerit(), 1e-6_r);
    for (int r = 0; r < dResNew.Rows(); r++) {
      EXPECT_NEAR_RTOL(res[r], resAd[r], 1e-6_r);
    }
    for (int r = 0; r < dResNew.Rows(); r++) {
      for (int c = 0; c < dResNew.Cols(); c++) {
        EXPECT_NEAR_RTOL(dResNew(r, c), dResNewAd(r, c), 1e-6_r);
      }
    }
    for (int r = 0; r < dResNew.Rows(); r++) {
      for (int c = 0; c < dResNew.Cols(); c++) {
        EXPECT_NEAR_RTOL(dResCurr(r, c), dResCurrAd(r, c), 1e-6_r);
      }
    }
    for (int r = 0; r < dResNew.Rows(); r++) {
      for (int c = 0; c < dResNew.Cols(); c++) {
        EXPECT_NEAR_RTOL(dResOld(r, c), dResOldAd(r, c), 1e-6_r);
      }
    }
#endif
  }
};
} // namespace

TEST_IF_F(MOCHI_USE_DOUBLE_EIGEN_AND_INTERNAL, MochiStepJacobianAutoDiff, ThreeRigid) {
  TestHessianAutoDiff("three_rigid_cubes_free.mochi_scene");
}

TEST_IF_F(MOCHI_USE_DOUBLE_EIGEN_AND_INTERNAL, MochiStepJacobianAutoDiff, TwoRigidWithStatic) {
  TestHessianAutoDiff("two_rigid_cubes_free_one_static.mochi_scene");
}

// The scene assets used by these tests are not shipped externally.
TEST_IF_F(MOCHI_INTERNAL, MochiStepJacobian, ThreeRigid) {
  TestStepJacobian("three_rigid_cubes_free.mochi_scene");
}

TEST_IF_F(MOCHI_INTERNAL, MochiStepJacobian, TwoRigidWithStatic) {
  TestStepJacobian("two_rigid_cubes_free_one_static.mochi_scene");
}
