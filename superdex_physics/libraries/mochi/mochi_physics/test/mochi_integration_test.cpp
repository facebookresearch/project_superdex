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

#include <mochi_physics/src/mochi_integration.h>

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/integration/integration_utils.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dtransform.h>
#include <mochi_core/utils/rodrigues_utils.h>
#include <mochi_core/utils/simd.h>

#include <gtest/gtest.h>

using namespace mochi;
using namespace mochi::integration;

static auto GetTestParams() {
  // Dummy multi-step, multi-stage scheme that exercises all codepaths.
  return TimeIntegratorParams(
      3,
      {{18_r / 11_r, -9_r / 11_r, 2_r / 11_r}},
      6_r / 11_r,
      2,
      {{0.25_r, 0_r}, {0.5_r, 0.25_r}},
      {{0.5_r, 0.5_r}},
      {{0.25_r, 0.75_r}});
}

template <
    typename ComputeTarget,
    typename Init,
    typename StageSolve,
    typename Emplace,
    typename Check,
    typename Previous,
    typename StageStart,
    typename Current,
    typename Integration>
static void EmulateTimeStep(
    ComputeTarget&& computeTarget,
    Init&& init,
    StageSolve&& stageSolve,
    Emplace&& emplace,
    Check&& check,
    Previous&& previous,
    StageStart&& stageStart,
    Current&& current,
    Integration&& integration,
    TimeIntegratorParams const& params,
    real dt) {
  CTimeIntegratorState intState = {};
  intState.numSteps = params.numSteps;
  intState.numStages = params.numStages;
  intState.alpha.Reset(params.alpha);
  intState.bTilde.Reset(params.bTilde);

  // Initialize previous solution.
  init(previous, integration);

  // Compute solution at the beginning of the step.
  computeTarget(intState, integration, previous, stageStart, current, TimeTarget::StepStart);

  // Perform time integration.
  for (int iStage = 0; iStage < params.numStages; ++iStage) {
    // Set stage params.
    intState.currentStage = iStage;
    intState.dtStage = params.A(iStage, iStage) * dt;
    intState.aTilde.Reset(
        params.A.Block(iStage, 0, 1, iStage) * params.Ainv.Block(0, 0, iStage, iStage));

    // Compute solution at the beginning of the stage.
    computeTarget(intState, integration, previous, stageStart, current, TimeTarget::StageStart);

    // Perform stage solve.
    stageSolve(stageStart, current);

    // Set solution at the end of the stage.
    emplace(integration, current, iStage);
  }

  // Compute solution at the end of the step.
  computeTarget(intState, integration, previous, stageStart, current, TimeTarget::StepEnd);

  // Perform checks.
  check(previous, current);

  // Check that the test was performed with an integration scheme that exercises all codepaths.
  EXPECT_GT(params.numSteps, 1);
  EXPECT_GT(params.numStages, 1);
  EXPECT_FALSE(NearEqual(params.beta, 1_r));
  for (int iStep = 0; iStep < params.numSteps; ++iStep) {
    EXPECT_FALSE(NearEqual(params.alpha(iStep), 0_r));
  }
  for (int iStage = 0; iStage < params.numStages; ++iStage) {
    EXPECT_FALSE(NearEqual(params.bTilde(iStage), 0_r));
    RowVector<real> aTilde =
        params.A.Block(iStage, 0, 1, iStage) * params.Ainv.Block(0, 0, iStage, iStage);
    for (int j = 0; j < iStage; ++j) {
      EXPECT_FALSE(NearEqual(aTilde(j), 0_r));
    }
  }
}

TEST(MochiIntegration, ConstantStep) {
  // Emulate a time step in which the solution at the end of each stage is the same as the solution
  // at the beginning of the stage, and check that the solution at the end of the step is the same
  // as the solution at the beginning of the step.

  auto params = GetTestParams();
  real const dt = 1e-2_r;

  auto computeTarget = [](auto const& intState,
                          auto& integration,
                          auto const& previous,
                          auto& stageStart,
                          auto& current,
                          TimeTarget const& targetTime) {
    if (targetTime == TimeTarget::StepStart) {
      ApplyTimeIntegrationStepStart(intState, integration, previous, integration.stepStart);
    } else if (targetTime == TimeTarget::StageStart) {
      ApplyTimeIntegration<TimeTarget::StageStart>(intState, integration, stageStart);
    } else if (targetTime == TimeTarget::StepEnd) {
      ApplyTimeIntegration<TimeTarget::StepEnd>(intState, integration, current);
    }
  };

  {
    int const numDofs = 100;
    auto init = [&](auto& previous, auto& integration) {
      previous.value.SetRandom(1, -1_r, 1_r);
      while (integration.prevSteps.size() < params.numSteps) {
        integration.prevSteps.emplace_back(previous.value);
      }
    };
    auto stageSolve = [](auto const& stageStart, auto& current) {
      current.value = stageStart.value;
    };
    auto emplace = [](auto& integration, auto const& current, int iStage) {
      integration.stages[iStage].value = current.value;
    };
    auto check = [](auto const& previous, auto const& current) {
      EXPECT_EQ(previous.value.size(), current.value.size());
      EXPECT_TRUE(test::NearEqualMatrices(previous.value, current.value));
    };

    EmulateTimeStep(
        computeTarget,
        init,
        stageSolve,
        emplace,
        check,
        CDisplacementSlice<real, TimeStep::Previous>(numDofs),
        CDisplacementSlice<real, TimeStep::StageStart>(numDofs),
        CDisplacementSlice<real, TimeStep::Current>(numDofs),
        CIntegrationDisplacementSlices(numDofs),
        params,
        dt);

    EmulateTimeStep(
        computeTarget,
        init,
        stageSolve,
        emplace,
        check,
        CVelocitySlice<real, TimeStep::Previous>(numDofs),
        CVelocitySlice<real, TimeStep::StageStart>(numDofs),
        CVelocitySlice<real, TimeStep::Current>(numDofs),
        CIntegrationVelocitySlices<DisplacementLayer::Default>(numDofs),
        params,
        dt);
  }

  {
    auto init = [&](auto& previous, auto& integration) {
      previous.value.SetVCom({0.2, -0.3, -0.1});
      previous.value.SetOmega({1.2, 0.3, -0.5});
      previous.value.UpdateVSymIfDirty(dt);
      while (integration.prevSteps.size() < params.numSteps) {
        integration.prevSteps.emplace_back(previous.value);
      }
    };
    auto stageSolve = [](auto const& stageStart, auto& current) {
      current.value = stageStart.value;
    };
    auto emplace = [](auto& integration, auto const& current, int iStage) {
      integration.stages[iStage].value = current.value;
    };
    auto check = [](auto const& previous, auto const& current) {
      EXPECT_NEAR_EQ(previous.value.GetVCom(), current.value.GetVCom());
      EXPECT_NEAR_EQ(previous.value.GetOmegaAndVSym().first, current.value.GetOmegaAndVSym().first);
    };

    EmulateTimeStep(
        computeTarget,
        init,
        stageSolve,
        emplace,
        check,
        CRigidVel<TimeStep::Previous>(),
        CRigidVel<TimeStep::StageStart>(),
        CRigidVel<TimeStep::Current>(),
        CIntegrationRigidVels(),
        params,
        dt);
  }

  {
    auto init = [&](auto& previous, auto& integration) {
      previous.value.SetTranslation(Real3{0.3_r, 0.1_r, -0.2_r});
      SetRotationVector(
          Vec4r{-0.3_r, -0.5_r, 0.2_r},
          previous.value); // Previous rotation vector's magnitude must be <= |pi|.
      while (integration.prevSteps.size() < params.numSteps) {
        integration.prevSteps.emplace_back(previous.value);
      }
    };
    auto stageSolve = [](auto const& stageStart, auto& current) {
      current.value = stageStart.value;
    };
    auto emplace = [](auto& integration, auto const& current, int iStage) {
      integration.stages[iStage].value = current.value;
    };
    auto check = [](auto const& previous, auto const& current) {
      EXPECT_NEAR_EQ(previous.value.GetTranslation(), current.value.GetTranslation());
      EXPECT_NEAR_EQ(GetRotationMatrix(previous.value), GetRotationMatrix(current.value));
    };

    EmulateTimeStep(
        computeTarget,
        init,
        stageSolve,
        emplace,
        check,
        CRigidState<TimeStep::Previous>(),
        CRigidState<TimeStep::StageStart>(),
        CRigidState<TimeStep::Current>(),
        CIntegrationRigidStates(),
        params,
        dt);
  }

  {
    DynamicArray<ArticulatedJointType> const jointTypes = {
        ArticulatedJointType::Free,
        ArticulatedJointType::Spherical,
        ArticulatedJointType::Revolute,
        ArticulatedJointType::Prismatic};
    auto const dofInfo = articulated::SetupJointDofs(jointTypes);
    auto const poseInfo = articulated::SetupJointPose(jointTypes);
    ArticulatedIntegrationMetadata const metadata{jointTypes, dofInfo, poseInfo};
    int const numDofs = RigidSize::kAll + RigidSize::kRot + 1 + 1;

    auto init = [&](auto& previous, auto& integration) {
      previous.value.SetRandom(1, -1_r, 1_r);
      articulated::NormalizeQuaternions(jointTypes, poseInfo, previous.value);
      while (integration.prevSteps.size() < params.numSteps) {
        integration.prevSteps.emplace_back(previous.value);
      }
    };
    auto stageSolve = [](auto const& stageStart, auto& current) {
      current.value = stageStart.value;
    };
    auto emplace = [](auto& integration, auto const& current, int iStage) {
      integration.stages[iStage].value = current.value;
    };
    auto check = [](auto const& previous, auto const& current) {
      EXPECT_TRUE(test::NearEqualMatrices(previous.value, current.value));
    };
    auto computeTargetArticulated = [&metadata](
                                        auto const& intState,
                                        auto& integration,
                                        auto const& previous,
                                        auto& stageStart,
                                        auto& current,
                                        TimeTarget const& targetTime) {
      if (targetTime == TimeTarget::StepStart) {
        ApplyTimeIntegrationStepStart(
            metadata, intState, integration, previous, integration.stepStart);
      } else if (targetTime == TimeTarget::StageStart) {
        ApplyTimeIntegration<TimeTarget::StageStart>(metadata, intState, integration, stageStart);
      } else if (targetTime == TimeTarget::StepEnd) {
        ApplyTimeIntegration<TimeTarget::StepEnd>(metadata, intState, integration, current);
      }
    };

    EmulateTimeStep(
        computeTargetArticulated,
        init,
        stageSolve,
        emplace,
        check,
        CArticulatedReducedPose<TimeStep::Previous>(numDofs),
        CArticulatedReducedPose<TimeStep::StageStart>(numDofs),
        CArticulatedReducedPose<TimeStep::Current>(numDofs),
        CIntegrationArticulatedReducedPose(numDofs),
        params,
        dt);
  }
}
