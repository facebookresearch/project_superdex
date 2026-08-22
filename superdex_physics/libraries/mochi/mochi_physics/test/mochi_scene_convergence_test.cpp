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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_physics/mochi_physics.h>

#include <gtest/gtest.h>

using namespace mochi;

/**
 * @brief Tests for SolverStats::convergenceStatus (via Scene::GetSolverStats()).
 *
 * Verifies the scene-level aggregation of per-actor convergence statuses:
 * - None: no dynamic actors or no step called
 * - Converged: all dynamic actors converged
 * - Stopped: at least one actor stopped, none diverged
 * - Diverged: at least one actor diverged
 */
class SceneConvergenceStatus : public test::MochiSceneTestBase {
 protected:
  static constexpr real kDt = 0.01_r;

  ShapeHandle CreateSmallTetShape() {
    real constexpr kScale = 0.1_r;
    auto [c, t] = test::CreateMinimalTetMeshUnitCube(Real3{kScale, kScale, kScale});
    return _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(c)), Flatten(MakeSpan(t)), test::ExpectOK{});
  }

  Actor* CreateSoft(Real3 translation, bool hasGravity) {
    SoftActorParams p;
    p.shape = CreateSmallTetShape();
    p.worldFromLocal = TransformRT{translation};
    p.hasGravity = hasGravity;
    return _scene->CreateSoftActor(p, test::ExpectOK{});
  }

  Actor* CreateStaticRigid(Real3 translation) {
    RigidActorParams p;
    p.shape = CreateSmallTetShape();
    p.isStatic = true;
    p.colliderType = ColliderType::None;
    p.worldFromLocal = TransformRT{translation};
    return _scene->CreateRigidActor(p, test::ExpectOK{});
  }

  Actor* CreateDynamicRigid(Real3 translation, bool hasGravity) {
    RigidActorParams p;
    p.shape = CreateSmallTetShape();
    p.colliderType = ColliderType::None;
    p.worldFromLocal = TransformRT{translation};
    p.hasGravity = hasGravity;
    return _scene->CreateRigidActor(p, test::ExpectOK{});
  }

  // Zero tolerance: no actor converges → Stopped.
  void SetSolverForStopped() {
    auto p = _scene->GetSolverParams();
    p.nonLinearSolver.absTol = 0_r;
    p.nonLinearSolver.relTol = 0_r;
    p.nonLinearSolver.relStepTol = 0_r;
    _scene->SetSolverParams(p, test::ExpectOK{});
  }

  // Zero tolerance + tiny explosion threshold: triggers divergence on any nonzero residual.
  void SetSolverForDiverged() {
    auto p = _scene->GetSolverParams();
    p.nonLinearSolver.absTol = 0_r;
    p.nonLinearSolver.relTol = 0_r;
    p.nonLinearSolver.relStepTol = 0_r;
    p.nonLinearSolver.absDivTol = 1e-30_r;
    _scene->SetSolverParams(p, test::ExpectOK{});
  }
};

// Empty scene → None. Static-only → None. Add dynamics → Converged. Remove statics → Converged.
TEST_F(SceneConvergenceStatus, ConvergedLifecycle) {
  EXPECT_EQ(ConvergenceStatus::None, _scene->GetSolverStats().convergenceStatus);

  auto* s1 = CreateStaticRigid({0_r, 0_r, 0_r});
  auto* s2 = CreateStaticRigid({5_r, 0_r, 0_r});
  _scene->Step(kDt);
  EXPECT_EQ(ConvergenceStatus::None, _scene->GetSolverStats().convergenceStatus);

  CreateSoft({0_r, 2_r, 0_r}, /*hasGravity*/ false);
  CreateSoft({5_r, 2_r, 0_r}, /*hasGravity*/ false);
  _scene->Step(kDt);
  EXPECT_EQ(ConvergenceStatus::Converged, _scene->GetSolverStats().convergenceStatus);

  _scene->DestroyActor(s1);
  _scene->DestroyActor(s2);
  _scene->Step(kDt);
  EXPECT_EQ(ConvergenceStatus::Converged, _scene->GetSolverStats().convergenceStatus);
}

// Static-only → None. Add dynamics with zero tolerance → Stopped. Remove statics → Stopped.
TEST_F(SceneConvergenceStatus, StoppedLifecycle) {
  auto* s1 = CreateStaticRigid({0_r, 0_r, 0_r});
  auto* s2 = CreateStaticRigid({5_r, 0_r, 0_r});
  _scene->Step(kDt);
  EXPECT_EQ(ConvergenceStatus::None, _scene->GetSolverStats().convergenceStatus);

  CreateSoft({0_r, 2_r, 0_r}, /*hasGravity*/ true);
  CreateSoft({5_r, 2_r, 0_r}, /*hasGravity*/ true);
  SetSolverForStopped();
  _scene->Step(kDt);
  EXPECT_EQ(ConvergenceStatus::Stopped, _scene->GetSolverStats().convergenceStatus);

  _scene->DestroyActor(s1);
  _scene->DestroyActor(s2);
  _scene->Step(kDt);
  EXPECT_EQ(ConvergenceStatus::Stopped, _scene->GetSolverStats().convergenceStatus);
}

// Static-only → None. Trigger explosion control → Diverged. Remove statics → Diverged.
TEST_F(SceneConvergenceStatus, DivergedLifecycle) {
  test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);

  auto* s1 = CreateStaticRigid({0_r, 0_r, 0_r});
  auto* s2 = CreateStaticRigid({5_r, 0_r, 0_r});
  _scene->Step(kDt);
  EXPECT_EQ(ConvergenceStatus::None, _scene->GetSolverStats().convergenceStatus);

  CreateSoft({0_r, 2_r, 0_r}, /*hasGravity*/ true);
  CreateSoft({5_r, 2_r, 0_r}, /*hasGravity*/ true);
  SetSolverForDiverged();
  _scene->Step(kDt);
  EXPECT_EQ(ConvergenceStatus::Diverged, _scene->GetSolverStats().convergenceStatus);

  _scene->DestroyActor(s1);
  _scene->DestroyActor(s2);
  _scene->Step(kDt);
  EXPECT_EQ(ConvergenceStatus::Diverged, _scene->GetSolverStats().convergenceStatus);
}

// Two-island priority chain: Converged < Stopped < Diverged.
// actorA (no gravity) is always in equilibrium → Converged regardless of solver params.
// actorB (gravity) outcome is controlled by solver params: Converged → Stopped → Diverged.
// Actors are created in separate steps to guarantee separate islands (ColliderType::None).
TEST_F(SceneConvergenceStatus, WorstStatusDominates) {
  test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);

  auto const* actorA = CreateDynamicRigid({0_r, 0_r, 0_r}, /*hasGravity*/ false);
  _scene->Step(kDt);

  auto const* actorB = CreateDynamicRigid({100_r, 0_r, 0_r}, /*hasGravity*/ true);

  // Default params: both converge → scene: Converged
  _scene->Step(kDt);
  EXPECT_EQ(ConvergenceStatus::Converged, actorA->GetConvergenceStatus());
  EXPECT_EQ(ConvergenceStatus::Converged, actorB->GetConvergenceStatus());
  EXPECT_EQ(ConvergenceStatus::Converged, _scene->GetSolverStats().convergenceStatus);

  // Zero tolerance: actorA converges (zero residual), actorB stops → scene: Stopped
  SetSolverForStopped();
  _scene->Step(kDt);
  EXPECT_EQ(ConvergenceStatus::Converged, actorA->GetConvergenceStatus());
  EXPECT_EQ(ConvergenceStatus::Stopped, actorB->GetConvergenceStatus());
  EXPECT_EQ(ConvergenceStatus::Stopped, _scene->GetSolverStats().convergenceStatus);

  // Explosion control: actorA converges (zero residual), actorB diverges → scene: Diverged
  SetSolverForDiverged();
  _scene->Step(kDt);
  EXPECT_EQ(ConvergenceStatus::Converged, actorA->GetConvergenceStatus());
  EXPECT_EQ(ConvergenceStatus::Diverged, actorB->GetConvergenceStatus());
  EXPECT_EQ(ConvergenceStatus::Diverged, _scene->GetSolverStats().convergenceStatus);
}

// Normal actors that would converge plus explosion control → scene reports Diverged.
TEST_F(SceneConvergenceStatus, DivergedDominates) {
  test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);

  CreateSoft({0_r, 0_r, 0_r}, /*hasGravity*/ false);
  CreateSoft({5_r, 0_r, 0_r}, /*hasGravity*/ false);
  CreateSoft({20_r, 0_r, 0_r}, /*hasGravity*/ true);
  CreateSoft({25_r, 0_r, 0_r}, /*hasGravity*/ true);

  SetSolverForDiverged();
  _scene->Step(kDt);
  EXPECT_EQ(ConvergenceStatus::Diverged, _scene->GetSolverStats().convergenceStatus);
}

// Multi-stage integrators (e.g. DIRK22) run several solver stages per step. The reported status
// must be the worst across all stages, not the last stage's. A velocity perturbation makes the
// first stage diverge; the divergence handler then resets the actor to rest, so the (gravity-free)
// second stage starts from equilibrium and converges. The step must therefore report Diverged,
// and a subsequent clean step must reset the accumulated status back to Converged.
TEST_F(SceneConvergenceStatus, WorstStatusAcrossStages) {
  test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);

  // Use a 2-stage integrator so per-stage statuses can differ within a single step.
  auto params = _scene->GetSolverParams();
  params.integrationMethod = IntegrationMethod::DIRK22;
  _scene->SetSolverParams(params, test::ExpectOK{});

  auto* actor = CreateDynamicRigid({0_r, 0_r, 0_r}, /*hasGravity*/ false);

  // Perturb with a velocity so the first stage has a nonzero residual and diverges. The reset to
  // rest leaves the second stage at equilibrium, so it converges: worst across stages is Diverged,
  // last stage is Converged.
  actor->SetVelocity(Real3{10_r, 0_r, 0_r}, Real3{}, test::ExpectOK{});
  SetSolverForDiverged();
  _scene->Step(kDt);
  EXPECT_EQ(ConvergenceStatus::Diverged, actor->GetConvergenceStatus());
  EXPECT_EQ(ConvergenceStatus::Diverged, _scene->GetSolverStats().convergenceStatus);

  // The actor was reset to rest, so a clean step converges in every stage. This also verifies that
  // the accumulated step status is reset before the stage loop (otherwise it would stay Diverged).
  params.nonLinearSolver = {};
  _scene->SetSolverParams(params, test::ExpectOK{});
  _scene->Step(kDt);
  EXPECT_EQ(ConvergenceStatus::Converged, actor->GetConvergenceStatus());
  EXPECT_EQ(ConvergenceStatus::Converged, _scene->GetSolverStats().convergenceStatus);
}
