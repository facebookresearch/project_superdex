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

#include "mochi_ik.h"
#include "mochi_articulated_body.h"
#include "mochi_contact_pair_params.h"
#include "mochi_context.h"
#include "mochi_scene.h"

#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/utils/dynamic_array.h>

using namespace mochi;

IKSolverImpl::IKSolverImpl(Scene* scene, Error& error) {
  MOCHI_ERROR_IF(scene == nullptr, error, "Cannot create an IKSolver without a Scene");
  MOCHI_ERROR_RETURN(error);
  auto* const sceneImpl = assert_cast<SceneImpl*>(scene);

  // Check to ensure that we only have articulated or rigid actors
  sceneImpl->ForEachActor([&](Actor* actor) {
    MOCHI_ERROR_IF(
        actor->GetType() != ActorType::Articulated && actor->GetType() != ActorType::Rigid,
        error,
        "Cannot create an IKSolver with non-articulated or non-rigid actors");
  });
  MOCHI_ERROR_RETURN(error);

  // AsyncScene and IKSolver both destroy their owned Scene, so ownership must be exclusive.
  MOCHI_ERROR_IF_NOT(sceneImpl->TryClaimOwnership(), error, "Scene is already owned.");
  MOCHI_ERROR_RETURN(error);

  // Assign _scene only after error checks. ~IKSolverImpl destroys it.
  _scene = sceneImpl;

  // Use default solver params
  SetSolverParams(experimental::IKSolverParams{});

  // Ensure gravity is zero
  _scene->SetGravity(Real3{0_r, 0_r, 0_r});

  // Force use single island, so that collisions in another island will not be missed.
  _scene->SetForceSingleIsland(true);

  // Disable contact dissipation on all actors
  auto& reg = _scene->GetRegistry();
  reg.view<ContactParams>().each([](ContactParams& params) {
    params.coulombFrictionCoefficient = 0_r;
    params.viscousFrictionCoefficient = 0_r;
    params.normalViscousDampingCoefficient = 0_r;
  });
  reg.ctx<CContactPairParamsOverrideTable>().DisableDissipation();

  // Disable joint friction and inertia by zeroing them via the setters. The components are
  // always present on articulated actors, so zeroing (rather than removing them) keeps that
  // invariant while the assembly gates skip the now-zero forces. IK always supplies
  // validly-sized, finite, non-negative values, so a failure here is a bug, not user error —
  // hence ErrorAssert{}. (CActorInterface is file-local to mochi_scene.cpp, so the Actor*
  // comes from ForEachActor rather than a reg.view here.)
  ArticulatedJointFrictionParams zeroFriction{};
  zeroFriction.falloffVel = 0_r; // value-init keeps the struct's 1e-3 default; zero it too
  // Fixed FILO stack budget (a generous joint-count bound). The macro needs a compile-time
  // size, so it can't be numJoints; the DynamicArrays spill to heap if an actor exceeds it.
  constexpr size_t kJointStackBudget =
      256 * (sizeof(ArticulatedJointFrictionParams) + sizeof(real));
  _scene->ForEachActor([&](Actor* actor) {
    if (actor->GetType() != ActorType::Articulated) {
      return;
    }
    size_t const numJoints = actor->GetArticulatedJointFrictionParams(ErrorAssert{}).size();
    MOCHI_FILO_STACK_ALLOCATOR(allocator, kJointStackBudget);
    DynamicArray<ArticulatedJointFrictionParams> friction(numJoints, zeroFriction, &allocator);
    actor->SetArticulatedJointFrictionParams(friction, ErrorAssert{});
    DynamicArray<real> inertia(numJoints, 0_r, &allocator);
    actor->SetArticulatedJointInertiaParams(inertia, ErrorAssert{});
  });
}

IKSolverImpl::~IKSolverImpl() {
  if (!_scene) {
    return;
  }
  // Clear targets
  for (auto const& target : _positionTargets) {
    _scene->DestroyConstraint(target.second);
  }
  for (auto const& target : _rotationTargets) {
    _scene->DestroyConstraint(target.second);
  }
  _positionTargets.clear();
  _rotationTargets.clear();

  _scene->GetContext()->DestroyScene(_scene);
}

void IKSolverImpl::SetSolverParams(experimental::IKSolverParams const& ikParams) {
  // Override the series of solver parameters
  auto params = _scene->GetSolverParams();
  params.nonLinearSolver.maxIter = ikParams.maxIter;
  params.nonLinearSolver.verbosity = ikParams.verbosity;
  params.nonLinearSolver.absTol = ikParams.absTol;
  params.nonLinearSolver.relTol = ikParams.relTol;
  params.nonLinearSolver.maxElapsedTimeSeconds = ikParams.maxElapsedTimeSeconds;
  // We use the most relaxed condition of merit-based line search accept, to allow
  // rough gradient computation under low precisions
  params.nonLinearSolver.lineSearchType = LineSearchType::Armijo;
  params.nonLinearSolver.lineSearchMaxIter = ikParams.lineSearchMaxIter;
  _scene->SetSolverParams(params, ErrorAssert{});

  _positionErrorThres = ikParams.positionErrorThres;
  _rotationErrorThres = ikParams.rotationErrorThres;
}

experimental::IKSolverParams IKSolverImpl::GetSolverParams() const {
  auto const& params = _scene->GetSolverParams();

  experimental::IKSolverParams ikParams{
      .maxIter = params.nonLinearSolver.maxIter,
      .verbosity = params.nonLinearSolver.verbosity,
      .absTol = params.nonLinearSolver.absTol,
      .relTol = params.nonLinearSolver.relTol,
      .positionErrorThres = _positionErrorThres,
      .rotationErrorThres = _rotationErrorThres,
      .lineSearchMaxIter = params.nonLinearSolver.lineSearchMaxIter,
      .maxElapsedTimeSeconds = params.nonLinearSolver.maxElapsedTimeSeconds};
  return ikParams;
}

Constraint* IKSolverImpl::CreatePositionTarget(
    ActorHandle actor,
    Real3 localPosition,
    Real3 targetPosition,
    real weight,
    Error& error) {
  MOCHI_ERROR_IF(!actor.IsValid(), error, "Actor handle is invalid");
  MOCHI_ERROR_IF(weight < 0_r, error, "Weight must not be negative");
  MOCHI_ERROR_RETURN(error, {});

  RigidPivotPositionConstraintParams conParams;
  conParams.localPosition = localPosition;
  conParams.targetPosition = targetPosition;
  conParams.actor = actor;
  conParams.stiffness = weight;
  auto* con = _scene->CreateRigidPivotPositionConstraint(conParams, error);
  MOCHI_ERROR_RETURN(error, {});

  ClearPositionTarget(actor, ErrorAssert{});
  _positionTargets[actor] = con;
  return con;
}

void IKSolverImpl::ClearPositionTarget(ActorHandle actor, Error& error) {
  MOCHI_ERROR_IF(!actor.IsValid(), error, "Actor handle is invalid");
  MOCHI_ERROR_RETURN(error);

  auto it = _positionTargets.find(actor);
  if (it != _positionTargets.end()) {
    _scene->DestroyConstraint(it->second);
    _positionTargets.erase(it);
  }
}

Constraint* IKSolverImpl::CreateRotationTarget(
    ActorHandle actor,
    Real3 localRotation,
    Real3 targetRotation,
    real weight,
    Error& error) {
  MOCHI_ERROR_IF(!actor.IsValid(), error, "Actor handle is invalid");
  MOCHI_ERROR_IF(weight < 0_r, error, "Weight must not be negative");
  MOCHI_ERROR_RETURN(error, {});

  RigidPivotRotationConstraintParams conParams;
  conParams.localRotation = localRotation;
  conParams.targetRotation = targetRotation;
  conParams.actor = actor;
  conParams.stiffness = weight;
  auto* con = _scene->CreateRigidPivotRotationConstraint(conParams, error);
  MOCHI_ERROR_RETURN(error, {});

  ClearRotationTarget(actor, ErrorAssert{});
  _rotationTargets[actor] = con;
  return con;
}

void IKSolverImpl::ClearRotationTarget(ActorHandle actor, Error& error) {
  MOCHI_ERROR_IF(!actor.IsValid(), error, "Actor handle is invalid");
  MOCHI_ERROR_RETURN(error);

  auto it = _rotationTargets.find(actor);
  if (it != _rotationTargets.end()) {
    _scene->DestroyConstraint(it->second);
    _rotationTargets.erase(it);
  }
}

bool IKSolverImpl::SolveIK(Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(
      _positionTargets.empty() && _rotationTargets.empty(),
      error,
      "We cannot solve an IK problem with no targets");
  MOCHI_ERROR_RETURN(error, {});

  // Notify the scene to solve the optimization
  _scene->Step(std::numeric_limits<real>::infinity());

  bool isReachable = true;
  // Compute position target
  for (auto const& target : _positionTargets) {
    Real3 posError = Unflatten<Real3>(target.second->GetDeviation())[0];
    if (Max(Abs(posError)) > _positionErrorThres) {
      isReachable = false;
    }
  }
  // Compute rotation target
  for (auto const& target : _rotationTargets) {
    Real3 rotError = Unflatten<Real3>(target.second->GetDeviation())[0];
    if (Max(Abs(rotError)) > _rotationErrorThres) {
      isReachable = false;
    }
  }
  return isReachable;
}

namespace mochi::experimental {

IKSolver* CreateIKSolver(Scene* scene, Context* context, Error& error) {
  MOCHI_ERROR_IF(context == nullptr, error, "Invalid context.");
  MOCHI_ERROR_RETURN(error, nullptr);
  return assert_cast<ContextImpl*>(context)->CreateIKSolver(scene, error);
}

void DestroyIKSolver(IKSolver* solver, Context* context, Error& error) {
  MOCHI_ERROR_IF(context == nullptr, error, "Invalid context.");
  MOCHI_ERROR_RETURN(error);
  assert_cast<ContextImpl*>(context)->DestroyIKSolver(solver);
}

bool IsValidIKSolver(IKSolver const* solver, Context* context, Error& error) {
  MOCHI_ERROR_IF(context == nullptr, error, "Invalid context.");
  MOCHI_ERROR_RETURN(error, false);
  return assert_cast<ContextImpl*>(context)->IsValidIKSolver(solver);
}

} // namespace mochi::experimental
