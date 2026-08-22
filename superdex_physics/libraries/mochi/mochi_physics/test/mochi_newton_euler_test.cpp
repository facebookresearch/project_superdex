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

#include "mochi_physics_test_fixture.h"

#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/ldlt.h>
#include <mochi_core/utils/string_utils.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <gtest/gtest.h>

#include <random>
#include <string>

using namespace mochi;
using namespace mochi::experimental;

#if MOCHI_USE_HDF5 && MOCHI_USE_DOUBLE_PRECISION && !MOCHI_DEBUG
#define MOCHI_HDF5_AND_DOUBLE_AND_NOT_DEBUG 1
#else
#define MOCHI_HDF5_AND_DOUBLE_AND_NOT_DEBUG 0
#endif

namespace {

class NewtonEulerTest : public test::MochiSceneTestBase {
 public:
  void SetUp() override {
    test::MochiSceneTestBase::SetUp();
    _scene->SetGravity({0_r, -9.81_r, 0_r});
  }

  void TearDown() override {
    if (_newtonEuler != nullptr) {
      experimental::DestroyNewtonEulerTerms(_newtonEuler, _mochiContext, test::ExpectOK{});
      _newtonEuler = nullptr;
    }
    test::MochiSceneTestBase::TearDown();
  }

 protected:
  static Actor* CreateFr3Actor(Context* mochiContext, Scene* scene) {
    // Load the prefab
    auto const prefabPath = test::GetAssetPath("franka_arm/fr3/fr3.mochi_prefab");
    auto const assetsDir = test::GetAssetPath("");
    auto scenePrefab = prefab::LoadFromFile(prefabPath, assetsDir, mochiContext, test::ExpectOK{});
    MOCHI_ASSERT(scenePrefab.actors.articulated.size() == 1, "Expected one articulated actor");
    auto& actorPrefab = scenePrefab.actors.articulated[0];

    // Fix the base joint
    actorPrefab.joints[0].type = ArticulatedJointType::Hard;

    for (auto& link : actorPrefab.links) {
      link.colliderType = ColliderType::None;
      link.layer = "Object";
      link.density = 1000_r;
      link.boundaryElementType = ActorBoundaryElementType::P1Q6;
    }

    // Remove pose-tracking controllers. NewtonEulerTerms uses prefab export/import to clone the
    // actor, which does not preserve pose-tracking constraints. Keeping them would cause M to
    // differ between the original and cloned scenes.
    scenePrefab.controllers.clear();

    // Create the actor
    auto rotation = Quaternion::RotationX(-90_r * kRadiansPerDegree);
    auto result = prefab::AddToScene(
        scenePrefab, scene, prefab::PrefabParams{.rotation = rotation}, test::ExpectOK{});
    auto actors = result.Filter(ActorType::Articulated);
    MOCHI_ASSERT(actors.size() == 1, "Expected one articulated actor");
    auto* actor = actors[0];

    scene->EnableLayerContactSymmetric("Object", "Object", false, test::ExpectOK{});

    EnableNewtonEulerInertia(actor, true, test::ExpectOK{});
    return actor;
  }

  struct StepResult {
    ColumnVector<real> q;
    ColumnVector<real> dq;
  };

  // Step using NewtonEulerTerms::Compute + manual LDLT solve (semi-implicit Euler).
  StepResult StepNewtonEuler(Actor* robot, Span<real const> q, Span<real const> dq, real dt) {
    int const numDofs = robot->GetNumDofs();

    Matrix<real> M(numDofs, numDofs);
    ColumnVector<real> C(numDofs);
    ColumnVector<real> JtF(numDofs);

    _newtonEuler->Compute(dt, q, dq, M, C, {}, JtF, test::ExpectOK{});

    int status = 0;
    LDLt<real> solver(M, status);
    MOCHI_ASSERT(status == 0, "LDLT factorization failed");

    // ddq = M^{-1} * (JtF - C)
    ColumnVector<real> ddq = JtF - C;
    solver.LeftSolveInPlace(ddq);

    // Semi-implicit Euler: dq1 = dq + ddq*dt
    StepResult result;
    result.dq = AsConstView(dq) + ddq * dt;

    // Integrate pose: q1 = q + dq1*dt
    result.q.Resize(numDofs);
    ColumnVector<real> dqdt = result.dq * dt;
    robot->AddArticulatedDeltaToPose(q, dqdt, result.q, test::ExpectOK{});
    return result;
  }

  // Step using Mochi's built-in Scene::Step with matched solver settings.
  static StepResult StepMochi(Scene* scene, Actor* robot, real dt) {
    auto solverParams = scene->GetSolverParams();
    solverParams.nonLinearSolver.maxIter = 1;
    solverParams.linearSolver.preconditionerType = PreconditionerType::None;
    solverParams.linearSolver.solverType = LinearSolverType::LDLT;
    solverParams.integrationMethod = IntegrationMethod::BackwardEuler;
    scene->SetSolverParams(solverParams, test::ExpectOK{});

    scene->Step(dt);

    int const numDofs = robot->GetNumDofs();
    StepResult result;
    result.q.Resize(numDofs);
    result.dq.Resize(numDofs);
    robot->GetArticulatedPose(result.q, test::ExpectOK{});
    robot->GetArticulatedJointVelocities(result.dq, test::ExpectOK{});
    return result;
  }

  NewtonEulerTerms* _newtonEuler = nullptr;
};

// Verify that stepping with NewtonEulerTerms + LDLT matches Mochi's built-in implicit integrator.
TEST_IF_F(MOCHI_HDF5_AND_DOUBLE_AND_NOT_DEBUG, NewtonEulerTest, CompareWithMochiStep) {
  auto* robot = CreateFr3Actor(_mochiContext, _scene);
  _newtonEuler = experimental::CreateNewtonEulerTerms(robot, _mochiContext, test::ExpectOK{});
  int const numDofs = robot->GetNumDofs();

  auto shapeInfo = robot->GetArticulatedShapeInfo(test::ExpectOK{});

  DynamicArray<Real2> dofLimits(numDofs);
  robot->GetArticulatedDofLimits(dofLimits, test::ExpectOK{});

  // Generate random joint positions within limits and small random velocities
  auto rng = RandomGenerator(42);
  DynamicArray<real> q(numDofs, 0_r);
  DynamicArray<real> dq(numDofs, 0_r);
  robot->GetArticulatedPose(q, test::ExpectOK{});

  for (int i = 0; i < isize(shapeInfo.jointTypes); ++i) {
    if (shapeInfo.jointTypes[i] == ArticulatedJointType::Revolute) {
      int const dofOffset = shapeInfo.dofInfo[i].offset;
      real const lo = dofLimits[dofOffset][0];
      real const hi = dofLimits[dofOffset][1];
      q[dofOffset] = RandomUniformValue<real>(rng, lo, hi);
      dq[dofOffset] = RandomUniformValue<real>(rng, -.5_r, .5_r);
    }
  }
  robot->SetArticulatedPoseFromJoints(q, test::ExpectOK{});
  robot->SetArticulatedJointVelocities(dq, test::ExpectOK{});

  // Simulation loop
  real dt = 1_r / 60_r, totalTime = 1_r;
  for (real t = 0; t < totalTime; t += dt) {
    robot->GetArticulatedPose(q, test::ExpectOK{});
    robot->GetArticulatedJointVelocities(dq, test::ExpectOK{});

    auto result1 = StepNewtonEuler(robot, q, dq, dt);
    auto result2 = StepMochi(_scene, robot, dt);

    robot->SetArticulatedPoseFromJoints(result1.q, test::ExpectOK{});
    robot->SetArticulatedJointVelocities(result1.dq, test::ExpectOK{});

    result1.q -= result2.q;
    EXPECT_NEAR_TOL(result1.q.Norm() / result2.q.Norm(), 0_r, 1e-4_r);

    result1.dq -= result2.dq;
    EXPECT_NEAR_TOL(result1.dq.Norm() / result2.dq.Norm(), 0_r, 1e-2_r);
  }
}
} // namespace
