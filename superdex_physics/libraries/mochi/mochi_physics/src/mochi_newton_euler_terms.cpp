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

#include "mochi_newton_euler_terms.h"
#include "mochi_articulated_body.h"
#include "mochi_context.h"
#include "mochi_ecs_utils.h"
#include "mochi_scene.h"
#include "mochi_solve.h"
#include "mochi_step.h"

#include <mochi_core/articulated_body/articulated_body_hessian.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/rigid_body_utils.h>
#include <mochi_core/utils/spmat_utils.h>
#include <mochi_physics/utils/mochi_prefab.h>

using namespace mochi;
using namespace mochi::articulated;
using namespace mochi::experimental;

NewtonEulerTermsImpl::NewtonEulerTermsImpl(Actor* robot, Error& error) {
  MOCHI_ERROR_IF(
      robot == nullptr || robot->GetType() != ActorType::Articulated,
      error,
      "Actor is null or not an articulated body");
  MOCHI_ERROR_RETURN(error);

  // Clone the actor into an isolated scene for Newton-Euler computation.
  _robot = CopyActorToSingletonScene(robot, robot->GetContext(), error);
  MOCHI_ERROR_RETURN(error);
  _scene = assert_cast<SceneImpl*>(_robot->GetScene());

  // Set solver configuration/settings
  experimental::EnableNewtonEulerInertia(_robot, true, ErrorAssert{});
}

NewtonEulerTermsImpl::~NewtonEulerTermsImpl() {
  if (_scene != nullptr) {
    _scene->GetContext()->DestroyScene(_scene);
  }
}

void NewtonEulerTermsImpl::Compute(
    real dt,
    Span<real const> q,
    Span<real const> dq,
    Span<real> outM,
    Span<real> outC,
    Span<real> outJ,
    Span<real> outJtF,
    Error& error) {
  auto& reg = _scene->GetRegistry();
  auto entity = GetEntity(reg, _robot->GetHandle(), ErrorAssert{});
  auto const& groupMembers = reg.get<CGroupMembers const>(entity);
  int numLinks = isize(groupMembers.actors);
  int numDofs = _robot->GetNumDofs();

  MOCHI_ERROR_IF(isize(q) != numDofs, error, "Incorrect size of joint position vector");
  MOCHI_ERROR_IF(isize(dq) != numDofs, error, "Incorrect size of joint velocity vector");
  MOCHI_ERROR_IF(
      !outJ.empty() && isize(outJ) != numLinks * numDofs * RigidSize::kDAll,
      error,
      "Incorrect size of the Jacobian matrix");
  MOCHI_ERROR_IF(
      isize(outC) != numDofs, error, "Incorrect size of the Centrifugal and Coriolis term");
  MOCHI_ERROR_IF(isize(outJtF) != numDofs, error, "Incorrect size of the external force term");
  MOCHI_ERROR_IF(isize(outM) != numDofs * numDofs, error, "Incorrect size of the mass matrix");
  MOCHI_ERROR_RETURN(error);

  // Set robot configuration
  _robot->SetArticulatedPoseFromJoints(q, error);
  MOCHI_ERROR_RETURN(error);
  _robot->SetArticulatedJointVelocities(dq, error);
  MOCHI_ERROR_RETURN(error);

  // Set gravity to zero and restore later
  auto gravitySaved = _scene->GetGravity();
  _scene->SetGravity({0_r, 0_r, 0_r});
  MOCHI_DEFER(_scene->SetGravity(gravitySaved));

  // Set integration method to backward Euler and restore later
  auto solverParamsSaved = _scene->GetSolverParams();
  auto solverParamsBE = solverParamsSaved;
  solverParamsBE.integrationMethod = IntegrationMethod::BackwardEuler;
  _scene->SetSolverParams(solverParamsBE, ErrorAssert{});
  MOCHI_DEFER(_scene->SetSolverParams(solverParamsSaved, ErrorAssert{}));

  // Jacobian
  auto const& artJacobian = reg.get<CArticulatedJacobian const>(entity).value;
  if (!outJ.empty()) {
    AsView(outJ) = ColumnVectorView<real const>(artJacobian.data(), isize(artJacobian));
  }

  // Mass matrix and Centrifugal and Coriolis force
  {
    // Store time step
    reg.ctx<CSceneTime>().Reset(0_r, dt, dt);

    // Shift state to the start of a timestep
    mochi::PreStepEcs(reg);

    // Extract result: f and DfDq
    ColumnVector<real> f(numDofs);
    Matrix<real> DfDq(numDofs, numDofs);
    reg.view<CIslandDescendants const>().each(
        [&](entt::entity island, CIslandDescendants const& descendants) {
          mochi::solver::AssembleIsland(reg, island, descendants, f, DfDq);
        });

    // Convert f to C(q,dq)
    MatrixView<real> mass(outM.data(), numDofs, numDofs);
    mass = DfDq * Sqr(dt);
    AsView(outC) = f + mass * AsConstView(dq) * (1_r / dt);
  }

  // External force
  {
    auto JtF = AsView(outJtF);
    JtF.SetZero();
    for (auto const& e : groupMembers.actors) {
      auto const& inertia = reg.get<CRigidBodyInertia const>(e);
      auto const& bodyDofOffset = reg.get<CDofOffset const>(e);

      Vec4r mg = inertia.GetMass() * ToSimd(gravitySaved);
      auto jacBody =
          artJacobian.MiddleRows<RigidSize::kDTrans>(bodyDofOffset.dofsOffset, RigidSize::kDTrans);
      JtF += jacBody.Transpose() * AsColumnVectorView<3>(mg);
    }
  }
}

// Free function implementations that delegate to ContextImpl

NewtonEulerTerms*
mochi::experimental::CreateNewtonEulerTerms(Actor* robot, Context* context, Error& error) {
  MOCHI_ERROR_IF(context == nullptr, error, "Invalid context");
  MOCHI_ERROR_RETURN(error, nullptr);
  return assert_cast<ContextImpl*>(context)->CreateNewtonEulerTerms(robot, error);
}

void mochi::experimental::DestroyNewtonEulerTerms(
    NewtonEulerTerms* newtonEulerTerms,
    Context* context,
    Error& error) {
  MOCHI_ERROR_IF(context == nullptr, error, "Invalid context");
  MOCHI_ERROR_RETURN(error);
  assert_cast<ContextImpl*>(context)->DestroyNewtonEulerTerms(newtonEulerTerms);
}

bool mochi::experimental::IsValidNewtonEulerTerms(
    NewtonEulerTerms const* newtonEulerTerms,
    Context* context,
    Error& error) {
  MOCHI_ERROR_IF(context == nullptr, error, "Invalid context");
  MOCHI_ERROR_RETURN(error, false);
  return assert_cast<ContextImpl*>(context)->IsValidNewtonEulerTerms(newtonEulerTerms);
}
