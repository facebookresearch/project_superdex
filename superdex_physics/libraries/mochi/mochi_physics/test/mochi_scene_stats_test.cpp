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
#include <mochi_physics/src/mochi_island.h>

#include <gtest/gtest.h>

using namespace mochi;

/***************************************************************************************************
  Test Fixture Class
*/
class MochiSceneStats : public test::MochiSceneTestBase {
 public:
  using BaseClass = test::MochiSceneTestBase;
  static constexpr double kTimeStep = 0.02;

  void SetUp() override {
    BaseClass::SetUp();
    // Enable evaluation of residual norm error
    auto solverParams = _scene->GetSolverParams();
    solverParams.experimentalEval.consistencyResNorm = true;
    _scene->SetSolverParams(solverParams, test::ExpectOK{});
  }

  void CreateDynamicRigidActor(Real3 position = {}) {
    real constexpr kScale = 0.1_r;
    auto&& [coordinates, connectivity] =
        test::CreateMinimalTetMeshUnitCube(Real3{kScale, kScale, kScale});
    auto shape = _scene->GetContext()->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});
    RigidActorParams params;
    params.shape = shape;
    params.colliderType = ColliderType::None;
    params.worldFromLocal.SetTranslation(position);
    _scene->CreateRigidActor(params, test::ExpectOK{});
  }
};

/***************************************************************************************************
  Test Cases
*/

// Test that GetPerformanceStats/GetSolverStats return valid initial values before any simulation
// steps
TEST_F(MochiSceneStats, InitialStats) {
  PerformanceStats performanceStats = _scene->GetPerformanceStats();
  SolverStats solverStats = _scene->GetSolverStats();

  EXPECT_EQ(performanceStats.totalStepDurationSec, 0.0);
  EXPECT_EQ(performanceStats.solveStepDurationSec, 0.0);
  EXPECT_EQ(performanceStats.preStepCallbacksDurationSec, 0.0);
  EXPECT_EQ(performanceStats.postStepCallbacksDurationSec, 0.0);
  EXPECT_EQ(performanceStats.recordingStepDurationSec, 0.0);
  EXPECT_EQ(solverStats.maxNonLinearIters, 0);
  EXPECT_EQ(solverStats.maxLineSearchIters, 0);
  EXPECT_EQ(solverStats.residualNorm, 0.0);

  auto debugStats = experimental::GetDebugStats(_scene, test::ExpectOK{});
  EXPECT_EQ(debugStats.maxResidualNormRelativeError, 0_r);
}

// Test that GetPerformanceStats/GetSolverStats return correct values after stepping with a dynamic
// actor
TEST_F(MochiSceneStats, StatsAfterSingleStep) {
  // Create a dynamic actor so the scene has something to simulate
  CreateDynamicRigidActor();

  _scene->Step(kTimeStep);
  PerformanceStats performanceStats = _scene->GetPerformanceStats();
  SolverStats solverStats = _scene->GetSolverStats();

  // Simulated time step should match the requested time step
  EXPECT_DOUBLE_EQ(_scene->GetLastTimeStep(), kTimeStep);

  // Durations should be positive.
  EXPECT_GT(performanceStats.totalStepDurationSec, 0.0);
  EXPECT_GT(performanceStats.solveStepDurationSec, 0.0);

  // After stepping with at least one dynamic actor, we expect at least one iteration
  EXPECT_GE(solverStats.maxNonLinearIters, 1);

  // Residual norm should be non-negative and finite
  EXPECT_GE(solverStats.residualNorm, 0.0);
  EXPECT_TRUE(IsFinite(solverStats.residualNorm));

  // Residual norm error should be positive and finite
  auto debugStats = experimental::GetDebugStats(_scene, test::ExpectOK{});
  EXPECT_GT(debugStats.maxResidualNormRelativeError, 0.0);
  EXPECT_TRUE(IsFinite(debugStats.maxResidualNormRelativeError));
}

// Test that stepping with zero time step doesn't produce invalid stats
TEST_F(MochiSceneStats, ZeroTimeStepStats) {
  CreateDynamicRigidActor();

  _scene->Step(0.0);
  SolverStats solverStats = _scene->GetSolverStats();

  // With zero time step, solver stats are reset but solver doesn't run
  EXPECT_EQ(solverStats.maxNonLinearIters, 0);
  EXPECT_EQ(solverStats.maxLineSearchIters, 0);
  EXPECT_EQ(solverStats.residualNorm, 0.0);

  auto debugStats = experimental::GetDebugStats(_scene, test::ExpectOK{});
  EXPECT_EQ(debugStats.maxResidualNormRelativeError, 0_r);
}
