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

#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/src/mochi_articulated_body.h>
#include <mochi_physics/src/mochi_common_components.h>
#include <mochi_physics/src/mochi_integration.h>
#include <mochi_physics/src/mochi_island.h>
#include <mochi_physics/src/mochi_rod_pose.h>

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/integration/integration_utils.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dtransform.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/rodrigues_utils.h>
#include <mochi_core/utils/simd.h>

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <string>
#include <utility>
#include <variant>

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

namespace {
// A non-BDF method (or empty history) drives the "fallback" branch of
// ReconstructPreviousPoseFromHistoryOrExtrapolate / ReconstructPreviousVelocityFromHistoryOrCopy;
// the BDF methods drive the "history" branch.
constexpr IntegrationMethod kFallbackMethod = IntegrationMethod::DIRK22;
} // namespace

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
      ApplyTimeIntegrationStepStart(
          /*advanceStep*/ true, intState, integration, previous, integration.stepStart);
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
    int constexpr kNumJoints = 3;
    auto init = [&](auto& previous, auto& integration) {
      for (int i = 0; i < kNumJoints; ++i) {
        previous.value[i].SetVCom({0.2_r * i, -0.3_r * i, -0.1_r * i});
        previous.value[i].SetOmega({1.2_r * i, 0.3_r * i, -0.5_r * i});
        previous.value[i].UpdateVSymIfDirty(dt);
      }
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
      ASSERT_EQ(previous.value.size(), current.value.size());
      for (int i = 0; i < isize(previous.value); ++i) {
        EXPECT_NEAR_EQ(previous.value[i].GetVCom(), current.value[i].GetVCom());
        EXPECT_NEAR_EQ(
            previous.value[i].GetOmegaAndVSym().first, current.value[i].GetOmegaAndVSym().first);
        EXPECT_NEAR_EQ(
            previous.value[i].GetOmegaAndVSym().second, current.value[i].GetOmegaAndVSym().second);
      }
    };

    EmulateTimeStep(
        computeTarget,
        init,
        stageSolve,
        emplace,
        check,
        CArticulatedJointVels<TimeStep::Previous>(kNumJoints),
        CArticulatedJointVels<TimeStep::StageStart>(kNumJoints),
        CArticulatedJointVels<TimeStep::Current>(kNumJoints),
        CIntegrationArticulatedJointVels(kNumJoints),
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
    DynamicArray<Real3> const jointAxes = {{}, {}, {1_r, 0_r, 0_r}, {0_r, 1_r, 0_r}};
    auto const dofInfo = articulated::SetupJointDofs(jointTypes);
    auto const poseInfo = articulated::SetupJointPose(jointTypes);

    ArticulatedIntegrationMetadata const metadata{jointTypes, jointAxes, dofInfo, poseInfo};
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
            metadata,
            /*advanceStep*/ true,
            intState,
            integration,
            previous,
            integration.stepStart);
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

namespace {

// Rod mesh nodes shared by the rod cases (a short straight open polyline).
DynamicArray<Real3> MakeRodNodes(int numNodes) {
  DynamicArray<Real3> nodes(numNodes, Real3{});
  for (int i = 0; i < numNodes; ++i) {
    nodes[i] = Real3{static_cast<real>(i), 0_r, 0_r};
  }
  return nodes;
}

void ExpectRodPoseEq(RodPose const& actual, RodPose const& expected) {
  EXPECT_TRUE(test::NearEqualMatrices(actual.displacements, expected.displacements));
  ASSERT_EQ(isize(actual.frameAxes), isize(expected.frameAxes));
  for (int i = 0; i < isize(expected.frameAxes); ++i) {
    EXPECT_NEAR_EQ(actual.frameAxes[i], expected.frameAxes[i]);
  }
}

// Per-data-type "cases" driven by the typed tests below. Each case holds the containers under test
// and provides the per-type steps: SetupHistory/CheckHistory (history branch) and
// SetupFallback/CheckFallback (fallback branch). The invoke() step is generic and lives in
// InvokeReconstruct(): it calls ReconstructPreviousVelocityFromHistoryOrCopy() when
// kUsesSetPrevious is true, else the metadata overload of
// ReconstructPreviousPoseFromHistoryOrExtrapolate(). Members read by InvokeReconstruct:
// integration, curr, outPrev, and (for the ReconstructPreviousPoseFromHistoryOrExtrapolate path)
// metadata and derivative.

struct DisplacementCase {
  static constexpr char kName[] = "Displacement";
  static constexpr bool kUsesSetPrevious = false;
  static constexpr int kNumDofs = 8;

  std::monostate metadata{};
  CIntegrationDisplacementSlices integration{kNumDofs};
  CDisplacementSlice<real, TimeStep::Current> curr{kNumDofs};
  CVelocitySlice<real, TimeStep::Current> derivative{kNumDofs};
  CDisplacementSlice<real, TimeStep::Previous> outPrev{kNumDofs};
  ColumnVector<real> expected;

  void SetupHistory() {
    expected = ColumnVector<real>::Zero(kNumDofs);
    expected.SetRandom(1, -1_r, 1_r);
    integration.prevSteps.emplace_back(expected);
    curr.value.SetRandom(2, -5_r, 5_r); // junk that must be ignored
    derivative.value.SetRandom(3, -5_r, 5_r);
  }
  void CheckHistory() const {
    EXPECT_TRUE(test::NearEqualMatrices(outPrev.value, expected));
  }
  void SetupFallback(real /*dt*/) {
    curr.value.SetRandom(1, -1_r, 1_r);
    derivative.value.SetRandom(2, -1_r, 1_r);
  }
  void CheckFallback(real dt) const {
    // prev = curr - dt*vel  =>  advancing prev by +dt returns curr.
    ColumnVector<real> advanced = outPrev.value + dt * derivative.value;
    EXPECT_TRUE(test::NearEqualMatrices(advanced, curr.value));
  }
};

struct RigidStateCase {
  static constexpr char kName[] = "RigidState";
  static constexpr bool kUsesSetPrevious = false;

  std::monostate metadata{};
  CIntegrationRigidStates integration{};
  CRigidState<TimeStep::Current> curr{};
  CRigidVel<TimeStep::Current> derivative{};
  CRigidState<TimeStep::Previous> outPrev{};
  TransformRT expected;

  void SetupHistory() {
    expected.SetTranslation(Real3{0.3_r, 0.1_r, -0.2_r});
    SetRotationVector(Vec4r{-0.3_r, -0.5_r, 0.2_r}, expected);
    integration.prevSteps.emplace_back(expected);
    curr.value.SetTranslation(Real3{9_r, 9_r, 9_r}); // junk that must be ignored
    SetRotationVector(Vec4r{0.9_r, -0.1_r, 0.4_r}, curr.value);
    derivative.value.SetVCom({5_r, 5_r, 5_r});
    derivative.value.SetOmega({1_r, 1_r, 1_r});
  }
  void CheckHistory() const {
    EXPECT_NEAR_EQ(outPrev.value.GetTranslation(), expected.GetTranslation());
    EXPECT_NEAR_EQ(GetRotationMatrix(outPrev.value), GetRotationMatrix(expected));
  }
  void SetupFallback(real dt) {
    curr.value.SetTranslation(Real3{0.3_r, 0.1_r, -0.2_r});
    SetRotationVector(Vec4r{-0.3_r, -0.5_r, 0.2_r}, curr.value);
    derivative.value.SetVCom({0.2_r, -0.3_r, -0.1_r});
    // Externally authored velocity, rotating by a large finite step (|omega| * dt ~ 0.8).
    derivative.value.SetOmega({72_r, 18_r, -30_r});
    derivative.value.UpdateVSymIfDirty(dt);
  }
  void CheckFallback(real dt) const {
    // Time-stepping the reconstructed previous pose with the velocity yields the current pose.
    EXPECT_NEAR_EQ(
        ToReal3(outPrev.value.VGetTranslation() + dt * derivative.value.GetVCom()),
        curr.value.GetTranslation());
    VMatrix3x3r const advancedRot =
        derivative.value.EvalTimeSteppedRotation(ToVMatrix3x3(outPrev.value.GetRotation()), dt);
    EXPECT_NEAR_EQ(ToNdArray3x3(advancedRot), GetRotationMatrix(curr.value));

    // Finite-differencing the reconstructed and current poses reproduces the angular velocity.
    RigidBodyVel finiteDifference;
    finiteDifference.SetFromFiniteDifferencePose(outPrev.value, curr.value, dt);
    Real3 const omega = ToReal3(derivative.value.GetOmegaAndVSym().first);
    EXPECT_NEAR_TOL(ToReal3(finiteDifference.GetOmegaAndVSym().first), omega, 1e-4_r * Norm(omega));
  }
};

// Fallback with a velocity produced by a solved step, i.e., finite-differenced from the previous
// pose: the reconstruction recovers that previous pose.
struct RigidStateFromSolveCase {
  static constexpr char kName[] = "RigidStateFromSolve";
  static constexpr bool kUsesSetPrevious = false;

  std::monostate metadata{};
  CIntegrationRigidStates integration{};
  CRigidState<TimeStep::Current> curr{};
  CRigidVel<TimeStep::Current> derivative{};
  CRigidState<TimeStep::Previous> outPrev{};
  TransformRT expected;

  void SetupFallback(real dt) {
    expected.SetTranslation(Real3{0.3_r, 0.1_r, -0.2_r});
    SetRotationVector(Vec4r{-0.3_r, -0.5_r, 0.2_r}, expected);
    // Large finite step of 0.8 rad.
    curr.value.SetTranslation(Real3{0.31_r, 0.08_r, -0.19_r});
    curr.value.SetRotation(
        Quaternion::FromRotationVector(Vec4r{0.48_r, 0_r, -0.64_r}) * expected.GetRotation());
    derivative.value.SetFromFiniteDifferencePose(expected, curr.value, dt);
  }
  void CheckFallback(real /*dt*/) const {
    EXPECT_NEAR_EQ(outPrev.value.GetTranslation(), expected.GetTranslation());
    EXPECT_NEAR_EQ(GetRotationMatrix(outPrev.value), GetRotationMatrix(expected));
  }
};

struct ArticulatedCase {
  static constexpr char kName[] = "Articulated";
  static constexpr bool kUsesSetPrevious = false;

  DynamicArray<ArticulatedJointType> jointTypes = {
      ArticulatedJointType::Free,
      ArticulatedJointType::Spherical,
      ArticulatedJointType::Revolute,
      ArticulatedJointType::Prismatic};
  // Non-axis-aligned, so that the projection of joint velocities onto the axes is exercised.
  Real3 axis = Normalize(Real3{1_r, 2_r, 2_r});
  DynamicArray<Real3> jointAxes = {{}, {}, axis, axis};
  DynamicArray<ArticulatedDofInfo> dofInfo = articulated::SetupJointDofs(jointTypes);
  DynamicArray<ArticulatedPoseInfo> poseInfo = articulated::SetupJointPose(jointTypes);
  ArticulatedIntegrationMetadata metadata{jointTypes, jointAxes, dofInfo, poseInfo};

  int poseSize = articulated::GetReducedPoseSize(poseInfo);
  CIntegrationArticulatedReducedPose integration{poseSize};
  CArticulatedReducedPose<TimeStep::Current> curr{poseSize};
  CArticulatedJointVels<TimeStep::Current> derivative{isize(jointTypes)};
  CArticulatedReducedPose<TimeStep::Previous> outPrev{poseSize};
  ColumnVector<real> expected;

  void SetupHistory() {
    expected = ColumnVector<real>::Zero(poseSize);
    expected.SetRandom(1, -1_r, 1_r);
    articulated::NormalizeQuaternions(jointTypes, poseInfo, expected);
    integration.prevSteps.emplace_back(expected);
    curr.value.SetRandom(2, -5_r, 5_r); // junk that must be ignored
  }
  void CheckHistory() const {
    EXPECT_TRUE(test::NearEqualMatrices(outPrev.value, expected));
  }
  void SetupFallback(real dt) {
    curr.value.SetRandom(1, -1_r, 1_r);
    articulated::NormalizeQuaternions(jointTypes, poseInfo, curr.value);
    auto& vels = derivative.value;

    // Free joint: externally authored velocity, rotating by a large finite step.
    vels[0].SetVCom({0.4_r, -0.2_r, 0.3_r});
    vels[0].SetOmega({40_r, -50_r, 48_r});
    vels[0].UpdateVSymIfDirty(dt);

    // Spherical joint: velocity finite-differenced from a previous rotation 0.8 rad away.
    Quaternion const currRot = JointTransform(curr.value, 1).GetRotation();
    expectedSphericalRot = Quaternion::FromRotationVector(Vec4r{0_r, -0.48_r, 0.64_r}) * currRot;
    vels[1].SetFromFiniteDifferencePose(
        TransformRT{expectedSphericalRot}, TransformRT{currRot}, dt);

    // Revolute and prismatic joints: exact rates along the joint axes.
    vels[2].SetOmegaWithZeroVSym(kRevoluteRate * ToSimd(jointAxes[2]));
    vels[3].SetVCom(kPrismaticRate * ToSimd(jointAxes[3]));
  }
  void CheckFallback(real dt) const {
    // Free: time-stepping the reconstructed joint transform yields the current one.
    TransformRT const prevFree = JointTransform(outPrev.value, 0);
    TransformRT const currFree = JointTransform(curr.value, 0);
    auto const& freeVel = derivative.value[0];
    EXPECT_NEAR_EQ(
        ToReal3(prevFree.VGetTranslation() + dt * freeVel.GetVCom()), currFree.GetTranslation());
    EXPECT_NEAR_EQ(
        ToNdArray3x3(freeVel.EvalTimeSteppedRotation(ToVMatrix3x3(prevFree.GetRotation()), dt)),
        GetRotationMatrix(currFree));

    // Spherical: the previous rotation is recovered.
    EXPECT_NEAR_EQ(
        GetRotationMatrix(JointTransform(outPrev.value, 1)),
        GetRotationMatrix(TransformRT{expectedSphericalRot}));

    // Revolute and prismatic: coordinates are retracted linearly.
    int const revoluteOffset = poseInfo[2].offset;
    EXPECT_NEAR_EQ(outPrev.value[revoluteOffset], curr.value[revoluteOffset] - dt * kRevoluteRate);
    int const prismaticOffset = poseInfo[3].offset;
    EXPECT_NEAR_EQ(
        outPrev.value[prismaticOffset], curr.value[prismaticOffset] - dt * kPrismaticRate);
  }

 private:
  static constexpr real kRevoluteRate = 70_r;
  static constexpr real kPrismaticRate = 0.5_r;
  Quaternion expectedSphericalRot;

  TransformRT JointTransform(ColumnVectorView<real const> pose, int joint) const {
    return articulated::ComputeJointTransform(
        pose, jointTypes[joint], jointAxes[joint], poseInfo[joint]);
  }
};

struct RodCase {
  static constexpr char kName[] = "Rod";
  static constexpr bool kUsesSetPrevious = false;
  static constexpr int kNumNodes = 4;
  static constexpr bool kIsClosedLoop = false;

  DynamicArray<Real3> nodes = MakeRodNodes(kNumNodes);
  Span<Real3 const> metadata = MakeConstSpan(nodes);
  CIntegrationRodPoses integration{kNumNodes, kIsClosedLoop};
  CRodPose<TimeStep::Current> curr{kNumNodes, kIsClosedLoop};
  CVelocitySlice<real, TimeStep::Current> derivative{kNumNodes * fem::kNumRodFields};
  CRodPose<TimeStep::Previous> outPrev{kNumNodes, kIsClosedLoop};
  RodPose expected;

  void SetupHistory() {
    expected.displacements = ColumnVector<real>::Zero(kNumNodes * fem::kNumRodFields);
    expected.displacements.SetRandom(1, -0.1_r, 0.1_r);
    expected.frameAxes = DynamicArray<Real3>(isize(curr.value.frameAxes), Real3{0_r, 0_r, 1_r});
    integration.prevSteps.emplace_back(expected);
    curr.value.displacements.SetRandom(2, -5_r, 5_r); // junk that must be ignored
  }
  void CheckHistory() const {
    ExpectRodPoseEq(outPrev.value, expected);
  }
  void SetupFallback(real /*dt*/) {
    curr.value.displacements.SetRandom(1, -0.1_r, 0.1_r);
    for (auto& axis : curr.value.frameAxes) {
      axis = Real3{0_r, 0_r, 1_r};
    }
    // Every node translates and twists.
    for (int i = 0; i < kNumNodes; ++i) {
      int const offset = i * fem::kNumRodFields;
      derivative.value[offset] = 0.2_r;
      derivative.value[offset + 1] = -0.3_r;
      derivative.value[offset + 2] = 0.1_r;
      derivative.value[offset + fem::kRodThetaDofOffset] = 0.5_r;
    }
  }
  void CheckFallback(real dt) const {
    ColumnVector<real> const delta = dt * derivative.value;
    ColumnVector<real> const expectedDisplacements = curr.value.displacements - delta;
    EXPECT_TRUE(test::NearEqualMatrices(outPrev.value.displacements, expectedDisplacements));
    // Advancing the reconstructed pose by +dt with the same velocity returns the current pose,
    // including the frame axes that the twist rotates.
    RodPose advanced;
    advanced.displacements.Resize(isize(delta));
    advanced.frameAxes.resize_noinit(isize(curr.value.frameAxes));
    rod::ApplyLieDeltaToPose(
        metadata,
        outPrev.value.displacements,
        outPrev.value.frameAxes,
        delta,
        advanced.displacements,
        advanced.frameAxes);
    ExpectRodPoseEq(advanced, curr.value);
  }
};

struct VelocityCase {
  static constexpr char kName[] = "Velocity";
  static constexpr bool kUsesSetPrevious = true;
  static constexpr int kNumDofs = 8;

  CIntegrationVelocitySlices<DisplacementLayer::Default> integration{kNumDofs};
  CVelocitySlice<real, TimeStep::Current> curr{kNumDofs};
  CVelocitySlice<real, TimeStep::Previous> outPrev{kNumDofs};
  ColumnVector<real> expected;

  void SetupHistory() {
    expected = ColumnVector<real>::Zero(kNumDofs);
    expected.SetRandom(1, -1_r, 1_r);
    integration.prevSteps.emplace_back(expected);
    curr.value.SetRandom(2, -5_r, 5_r); // junk that must be ignored
  }
  void CheckHistory() const {
    EXPECT_TRUE(test::NearEqualMatrices(outPrev.value, expected));
  }
  void SetupFallback(real /*dt*/) {
    curr.value.SetRandom(1, -1_r, 1_r);
  }
  void CheckFallback(real /*dt*/) const {
    EXPECT_TRUE(test::NearEqualMatrices(outPrev.value, curr.value)); // fallback copies current
  }
};

struct RigidVelCase {
  static constexpr char kName[] = "RigidVel";
  static constexpr bool kUsesSetPrevious = true;

  CIntegrationRigidVels integration{};
  CRigidVel<TimeStep::Current> curr{};
  CRigidVel<TimeStep::Previous> outPrev{};
  RigidBodyVel expected;

  void SetupHistory() {
    expected.SetVCom({0.2_r, -0.3_r, -0.1_r});
    expected.SetOmega({1.2_r, 0.3_r, -0.5_r});
    integration.prevSteps.emplace_back(expected);
    curr.value.SetVCom({9_r, 9_r, 9_r}); // junk that must be ignored
    curr.value.SetOmega({9_r, 9_r, 9_r});
  }
  void CheckHistory() const {
    EXPECT_NEAR_EQ(outPrev.value.GetVCom(), expected.GetVCom());
    EXPECT_NEAR_EQ(outPrev.value.GetOmegaAndVSym().first, expected.GetOmegaAndVSym().first);
  }
  void SetupFallback(real /*dt*/) {
    curr.value.SetVCom({0.2_r, -0.3_r, -0.1_r});
    curr.value.SetOmega({1.2_r, 0.3_r, -0.5_r});
  }
  void CheckFallback(real /*dt*/) const {
    EXPECT_NEAR_EQ(outPrev.value.GetVCom(), curr.value.GetVCom());
    EXPECT_NEAR_EQ(outPrev.value.GetOmegaAndVSym().first, curr.value.GetOmegaAndVSym().first);
  }
};

// The generic invoke() step, shared by all cases and both branches.
template <typename CaseT>
void InvokeReconstruct(CaseT& c, IntegrationMethod method, CSceneTime const& time) {
  if constexpr (CaseT::kUsesSetPrevious) {
    ReconstructPreviousVelocityFromHistoryOrCopy(method, c.integration, c.curr, c.outPrev);
  } else {
    ReconstructPreviousPoseFromHistoryOrExtrapolate(
        c.metadata, method, time, c.integration, c.curr, c.derivative, c.outPrev);
  }
}

std::string MethodName(IntegrationMethod method) {
  if (method == IntegrationMethod::BackwardEuler) {
    return "BackwardEuler";
  }
  if (method == IntegrationMethod::BDF2) {
    return "BDF2";
  }
  if (method == IntegrationMethod::BDF3) {
    return "BDF3";
  }
  return "Method" + std::to_string(static_cast<int>(method));
}

template <IntegrationMethod kM>
struct MethodTag {
  static constexpr IntegrationMethod kMethod = kM;
};

// Pairs a data-type case with an integration method, so the history test varies over both.
template <typename CaseT, typename MethodTagT>
struct HistoryCase {
  using Case = CaseT;
  static constexpr IntegrationMethod kMethod = MethodTagT::kMethod;
};

struct HistoryCaseName {
  template <typename T>
  static std::string GetName(int) {
    return std::string(T::Case::kName) + "_" + MethodName(T::kMethod);
  }
};

struct CaseName {
  template <typename T>
  static std::string GetName(int) {
    return std::string(T::kName);
  }
};

// History-branch variations: every data type crossed with every BDF (history-reproducible) method.
using HistoryCases = ::testing::Types<
    HistoryCase<DisplacementCase, MethodTag<IntegrationMethod::BackwardEuler>>,
    HistoryCase<DisplacementCase, MethodTag<IntegrationMethod::BDF2>>,
    HistoryCase<DisplacementCase, MethodTag<IntegrationMethod::BDF3>>,
    HistoryCase<RigidStateCase, MethodTag<IntegrationMethod::BackwardEuler>>,
    HistoryCase<RigidStateCase, MethodTag<IntegrationMethod::BDF2>>,
    HistoryCase<RigidStateCase, MethodTag<IntegrationMethod::BDF3>>,
    HistoryCase<ArticulatedCase, MethodTag<IntegrationMethod::BackwardEuler>>,
    HistoryCase<ArticulatedCase, MethodTag<IntegrationMethod::BDF2>>,
    HistoryCase<ArticulatedCase, MethodTag<IntegrationMethod::BDF3>>,
    HistoryCase<RodCase, MethodTag<IntegrationMethod::BackwardEuler>>,
    HistoryCase<RodCase, MethodTag<IntegrationMethod::BDF2>>,
    HistoryCase<RodCase, MethodTag<IntegrationMethod::BDF3>>,
    HistoryCase<VelocityCase, MethodTag<IntegrationMethod::BackwardEuler>>,
    HistoryCase<VelocityCase, MethodTag<IntegrationMethod::BDF2>>,
    HistoryCase<VelocityCase, MethodTag<IntegrationMethod::BDF3>>,
    HistoryCase<RigidVelCase, MethodTag<IntegrationMethod::BackwardEuler>>,
    HistoryCase<RigidVelCase, MethodTag<IntegrationMethod::BDF2>>,
    HistoryCase<RigidVelCase, MethodTag<IntegrationMethod::BDF3>>>;

// Fallback-branch variations: every data type (fallback is method-agnostic once history is empty).
using FallbackCases = ::testing::Types<
    DisplacementCase,
    RigidStateCase,
    RigidStateFromSolveCase,
    ArticulatedCase,
    RodCase,
    VelocityCase,
    RigidVelCase>;

template <typename T>
class ReconstructHistoryTest : public ::testing::Test {};
TYPED_TEST_SUITE(ReconstructHistoryTest, HistoryCases, HistoryCaseName);

// History branch: with a BDF method and non-empty history,
// ReconstructPreviousPoseFromHistoryOrExtrapolate / ReconstructPreviousVelocityFromHistoryOrCopy
// copy prevSteps[0] exactly, ignoring curr / derivative.
TYPED_TEST(ReconstructHistoryTest, CopiesHistoryExactly) {
  CSceneTime time;
  typename TypeParam::Case c;
  c.SetupHistory();
  InvokeReconstruct(c, TypeParam::kMethod, time);
  c.CheckHistory();
}

template <typename T>
class ReconstructFallbackTest : public ::testing::Test {};
TYPED_TEST_SUITE(ReconstructFallbackTest, FallbackCases, CaseName);

// Fallback branch: with empty history, ReconstructPreviousPoseFromHistoryOrExtrapolate
// back-extrapolates by the finite step that the velocity describes, and
// ReconstructPreviousVelocityFromHistoryOrCopy copies current.
TYPED_TEST(ReconstructFallbackTest, ExtrapolatesFromCurrent) {
  CSceneTime time;
  real const dt = static_cast<real>(time.DeltaTime());
  TypeParam c;
  c.SetupFallback(dt);
  InvokeReconstruct(c, kFallbackMethod, time);
  c.CheckFallback(dt);
}

class IntegrationMethodTest : public test::MochiSceneTestBase,
                              public ::testing::WithParamInterface<int> {};

INSTANTIATE_TEST_SUITE_P(
    IntegrationMethods,
    IntegrationMethodTest,
    ::testing::Range(0, static_cast<int>(IntegrationMethod::Count)),
    [](::testing::TestParamInfo<int> const& info) {
      return std::string(SReflect::EnumToString(static_cast<IntegrationMethod>(info.param)));
    });

// A quasistatic step ends a motion at rest at the static equilibrium.
TEST_P(IntegrationMethodTest, QuasistaticStepEndsAtRestAtEquilibrium) {
  test::SetSceneIntegrationMethod(_scene, static_cast<IntegrationMethod>(GetParam()));
  _scene->SetGravity(Real3{});
  Real3 constexpr kVelocity{0.6_r, -0.4_r, 0.7_r};
  RigidActorParams params;
  params.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
  Actor* box = _scene->CreateRigidActor(params, test::ExpectOK{});
  RigidPivotPositionConstraintParams position;
  position.actor = box->GetHandle();
  _scene->CreateRigidPivotPositionConstraint(position, test::ExpectOK{});
  RigidPivotRotationConstraintParams rotation;
  rotation.actor = box->GetHandle();
  _scene->CreateRigidPivotRotationConstraint(rotation, test::ExpectOK{});
  box->SetVelocity(kVelocity, Real3{0.5_r, 0.2_r, 0.3_r}, test::ExpectOK{});

  // Deformables pinned so that their rest shape is their only static equilibrium: soft and shell
  // cubes on one face, and a rod at its first node, twist included, and its second node.
  SoftActorParams softParams;
  softParams.shape = params.shape;
  softParams.worldFromLocal.SetTranslation(Real3{2_r, 0_r, 0_r});
  Actor* soft = _scene->CreateSoftActor(softParams, test::ExpectOK{});
  auto const [triCoordinates, triConnectivity] = test::CreateMinimalTriMeshUnitCube();
  experimental::ShellActorParams shellParams;
  shellParams.shape = _mochiContext->CreateTriMeshShape(
      Flatten(MakeSpan(triCoordinates)), Flatten(MakeSpan(triConnectivity)), test::ExpectOK{});
  shellParams.worldFromLocal.SetTranslation(Real3{4_r, 0_r, 0_r});
  Actor* shell = experimental::CreateShellActor(_scene, shellParams, test::ExpectOK{});
  for (Actor* cube : {soft, shell}) {
    real const faceX = cube->GetRootTransform().GetTranslation()[0];
    experimental::ConstrainNodesByPosition(
        cube, [faceX](int, Real3 const& node) { return node[0] == faceX; }, test::ExpectOK{});
  }

  experimental::RodActorParams rodParams;
  rodParams.shape = experimental::CreatePolylineShape(
      _mochiContext,
      MakeRodNodes(3),
      DynamicArray<Real3>(2, Real3{0_r, 1_r, 0_r}),
      /*isClosedLoop=*/false,
      test::ExpectOK{});
  rodParams.worldFromLocal.SetTranslation(Real3{0_r, 3_r, 0_r});
  // Stiff enough for the solve to stop within kTol of the rest shape.
  rodParams.material.flexuralStiffness = Real2{1_r, 1_r};
  Actor* rod = experimental::CreateRodActor(_scene, rodParams, test::ExpectOK{});
  DynamicArray<int> const rodPinnedDofs = {0, 1, 2, 3, 4, 5, 6};
  DynamicArray<real> const rodPinnedValues = {0_r, 3_r, 0_r, 0_r, 1_r, 3_r, 0_r};
  rod->AddBoundaryConditionDofsWorld(
      MakeConstSpan(rodPinnedDofs), MakeConstSpan(rodPinnedValues), test::ExpectOK{});

  std::array const deformables = {
      std::pair{soft, kSpaceDim3},
      std::pair{shell, kSpaceDim3},
      std::pair{rod, fem::kNumRodFields}};
  DynamicArray<DynamicArray<Real3>> restNodes;
  for (auto const& [deformable, numFieldsPerNode] : deformables) {
    // Every node moves at kVelocity, without twisting.
    DynamicArray<real> velocities(deformable->GetNumDofs(), 0_r);
    for (int dof = 0; dof < isize(velocities); ++dof) {
      int const field = dof % numFieldsPerNode;
      velocities[dof] = field < kSpaceDim3 ? kVelocity[field] : 0_r;
    }
    deformable->SetNodeVelocitiesLocal(MakeConstSpan(velocities), test::ExpectOK{});
    restNodes.push_back(test::GetWorldNodePositions(*deformable, numFieldsPerNode));
  }

  // An articulated link on a spherical joint with inertia, which a pose controller holds at its
  // rest pose.
  ArticulatedActorParams articulatedParams;
  articulatedParams.worldFromRoot.SetTranslation(Real3{6_r, 0_r, 0_r});
  articulatedParams.joints.resize(1);
  articulatedParams.joints[0].type = ArticulatedJointType::Spherical;
  articulatedParams.joints[0].inertia = 0.1_r;
  articulatedParams.links.resize(1);
  articulatedParams.links[0].shape = params.shape;
  articulatedParams.links[0].colliderType = ColliderType::None;
  articulatedParams.jointVelocities = DynamicArray<real>{0.5_r, 0.2_r, 0.3_r};
  Actor* articulated = _scene->CreateArticulatedActor(articulatedParams, test::ExpectOK{});
  PoseControllerParams controllerParams;
  controllerParams.jointTracking = {{.stiffness = 1e3_r}};
  articulated->AddArticulatedPoseController(controllerParams, test::ExpectOK{});
  Actor const* link = _scene->GetActor(articulated->GetNestedLinkActors(test::ExpectOK{})[0]);
  TransformRT const linkRest = link->GetRootTransform();

  // Enough dynamic steps for every multi-step method to have a full history.
  for (int i = 0; i < kMaxIntegrationSteps; ++i) {
    _scene->Step(0.01);
  }

  {
    // Methods other than backward Euler warn, once per process, that it replaces them.
    auto suppressWarning = test::SuppressLogWarning();
    _scene->Step(std::numeric_limits<double>::infinity());
  }

  real constexpr kTol = 1e-4_r;
  auto const expectAtEquilibrium = [&] {
    EXPECT_NEAR_TOL(TransformRT::Identity(), box->GetRootTransform(), kTol);
    EXPECT_NEAR_TOL(linkRest, link->GetRootTransform(), kTol);
    for (int i = 0; i < isize(deformables); ++i) {
      auto const [deformable, numFieldsPerNode] = deformables[i];
      DynamicArray<Real3> const nodes = test::GetWorldNodePositions(*deformable, numFieldsPerNode);
      for (int n = 0; n < isize(nodes); ++n) {
        EXPECT_NEAR_TOL(restNodes[i][n], nodes[n], kTol);
      }
    }
  };
  expectAtEquilibrium();
  for (int i = 0; i < kMaxIntegrationSteps; ++i) {
    _scene->Step(0.01);
    expectAtEquilibrium();
  }
}

// An actor without inertia ends a step at its static equilibrium. Other islands keep the requested
// method.
TEST_P(IntegrationMethodTest, ActorWithoutInertiaEndsStepAtEquilibrium) {
  auto const method = static_cast<IntegrationMethod>(GetParam());
  test::SetSceneIntegrationMethod(_scene, method);
  // Two soft cubes hang from their constrained top faces, far enough apart to form two islands.
  // Only the first has no inertia.
  auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
  ModelData model;
  model.mesh.emplace();
  model.mesh->nodesPerElement = 4;
  model.mesh->coordinates = Flatten(MakeSpan(coordinates));
  model.mesh->connectivity = Flatten(MakeSpan(connectivity));
  model.constrainedNodes = DynamicArray<int>{2, 3, 6, 7};
  SoftActorParams params;
  params.shape = _mochiContext->CreateModelShape(model, test::ExpectOK{});
  params.hasInertia = false;
  params.material.neoHookean.youngsModulus = 1e6_r;
  Actor* cube = _scene->CreateSoftActor(params, test::ExpectOK{});
  cube->AddBoundaryConditionConstrainedNodesAtRest(test::ExpectOK{});
  params.hasInertia = true;
  params.worldFromLocal = TransformRT{Real3{3_r, 0_r, 0_r}};
  Actor* inertialCube = _scene->CreateSoftActor(params, test::ExpectOK{});
  inertialCube->AddBoundaryConditionConstrainedNodesAtRest(test::ExpectOK{});

  // Methods replaced in the first cube's island, or in the quasistatic step, warn once per process.
  auto suppressWarning = test::SuppressLogWarning();
  for (int i = 0; i < kMaxIntegrationSteps; ++i) {
    _scene->Step(0.01);
  }
  auto const numIslandStages = [this](Actor* actor) {
    auto const& reg = GetRegistry();
    entt::entity const island = reg.get<CIslandMemberInfo const>(GetEntity(actor)).island;
    return isize(reg.get<CIslandSolverStats const>(island).stages);
  };
  EXPECT_EQ(GetNumStages(GetMethodEndingAtLastStage(method)), numIslandStages(cube));
  EXPECT_EQ(GetNumStages(method), numIslandStages(inertialCube));
  DynamicArray<Real3> const afterStep = test::GetWorldNodePositions(*cube, kSpaceDim3);
  // A quasistatic step solves for the static equilibrium.
  _scene->Step(std::numeric_limits<double>::infinity());
  DynamicArray<Real3> const equilibrium = test::GetWorldNodePositions(*cube, kSpaceDim3);
  for (int i = 0; i < isize(equilibrium); ++i) {
    EXPECT_NEAR_TOL(equilibrium[i], afterStep[i], 1e-4_r);
  }
}

} // namespace
