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

#pragma once

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <superdex_robotics/controllers/controller_basic_jsc_pd.h>
#include <superdex_robotics/core/context.h>
#include <superdex_robotics/superdex_robotics.h>

#include <gtest/gtest.h>

#include <numeric>
#include <string_view>
#include <type_traits>
#include <vector>

/* The fixture of the controllers' float input tests: controller_float_input_test.cpp and
 * internal/controller_float_input_osc_test.cpp. */
namespace superdex::robotics::controller_float_input {

using mochi::operator""_r;

inline constexpr std::string_view kFr3Path = "bots/arms/fr3/fr3.superdex_bot";
inline constexpr std::string_view kOscInitArgs =
    R"({"baseLinkName": "fr3_link0", "eeLinkName": "fr3_link8"})";
inline constexpr mochi::real kToolOffset = 0.13_r; // [m], along the EE link's z
inline constexpr mochi::real kDt = 1_r / 60_r; // [s]
inline constexpr int kNumSteps = 30;

inline std::vector<mochi::real> ToVector(mochi::Span<mochi::real const> values) {
  return {values.begin(), values.end()};
}

inline std::vector<int> ToVector(mochi::Span<int const> values) {
  return {values.begin(), values.end()};
}

/* x y z, qx qy qz qw. */
inline std::vector<mochi::real> PoseValues(mochi::TransformRT const& pose) {
  mochi::Real3 const t = pose.GetTranslation();
  mochi::Quaternion const& q = pose.GetRotation();
  return {t[0], t[1], t[2], q.data[0], q.data[1], q.data[2], q.data[3]};
}

/* The typed path for @p controller's float input, then ComputeOutput: for BASIC_JSC_PD the live
 * observations with the input as the target joint positions, for a pose input the input as the
 * target pose. */
template <typename Controller>
mochi::Span<mochi::real const> TypedOutput(
    Controller& controller,
    std::vector<mochi::real> const& input) {
  typename Controller::Target target;
  if constexpr (std::is_same_v<Controller, ControllerBasicJscPd>) {
    typename Controller::Obsv obsv =
        controller.GetCurrentObservationsFromMochi(mochi::test::ExpectOK{});
    obsv.dt = kDt;
    target.targetPose.assign(input.begin(), input.end());
    return controller.ComputeOutput(obsv, target, mochi::test::ExpectOK{});
  } else {
    target.rootFromTargetEE = mochi::TransformRT{
        mochi::Quaternion(input[3], input[4], input[5], input[6]),
        mochi::Real3{input[0], input[1], input[2]}};
    return controller.ComputeOutput(
        controller.GetCurrentObservationsFromMochi(mochi::test::ExpectOK{}),
        target,
        mochi::test::ExpectOK{});
  }
}

/* An FR3 in a scene without gravity. Each case drives it with one controller's float input while
 * a twin of that controller, configured alike, computes the typed output for the same input. */
class ControllerFloatInputTest : public testing::Test {
 protected:
  void SetUp() override {
    _mochiContext = mochi::CreateContext(0);
    _scene = _mochiContext->CreateScene("ControllerFloatInputTest");
    _scene->SetGravity(mochi::Real3{0_r, 0_r, 0_r});
    _botsCtx = CreateRoboticsContext();
    BotPrefab const prefab =
        LoadBotPrefabFromFile(mochi::test::GetAssetPath(kFr3Path), mochi::test::ExpectOK{});
    _bot = CreateBot(_scene, prefab, _botsCtx, mochi::test::ExpectOK{});
    ASSERT_NE(_bot, nullptr);
    _robot = _bot->GetArticulatedActor();
  }

  void TearDown() override {
    DestroyRoboticsContext(_botsCtx);
    _mochiContext->DestroyScene(_scene);
    mochi::DestroyContext(_mochiContext);
  }

  template <typename Controller>
  Controller& Create(char const* name, std::string_view params, std::string_view initArgs) {
    auto* const controller = static_cast<Controller*>(
        _bot->CreateController(Controller::TypeName(), name, mochi::test::ExpectOK{}));
    controller->ConfigureFromSceneEntry(params, initArgs, mochi::test::ExpectOK{});
    return *controller;
  }

  /* One control step: @p controller's output for @p input must equal @p twin's typed output, bit
   * for bit, and moves the robot. */
  template <typename Controller>
  std::vector<mochi::real>
  StepBoth(Controller& controller, Controller& twin, std::vector<mochi::real> const& input) {
    std::vector<mochi::real> efforts(_robot->GetNumDofs(), 1_r);
    controller.ComputeOutputFromInput(input, kDt, efforts, mochi::test::ExpectOK{});
    EXPECT_EQ(efforts, ToVector(TypedOutput(twin, input)));
    std::vector<int> dofs(efforts.size());
    std::iota(dofs.begin(), dofs.end(), 0);
    std::vector<mochi::experimental::ControlType> const types(
        dofs.size(), mochi::experimental::ControlType::Force);
    mochi::experimental::SetArticulatedForceAndTargetPose(
        _robot, efforts, types, dofs, mochi::test::ExpectOK{});
    _scene->Step(kDt);
    return efforts;
  }

  /* Holds @p osc's current input, then steps it and @p twin toward a shifted pose. */
  template <typename Osc>
  void ExpectPoseInputIsTheTypedPath(Osc& osc, Osc& twin) {
    EXPECT_EQ(osc.GetInputSize(), 7);
    EXPECT_EQ(osc.GetInputKind(), ControllerInputKind::Pose);
    /* From fr3_link0 to fr3_link8 it drives all seven arm joints. */
    ControllerBase const& base = osc;
    EXPECT_EQ(ToVector(base.GetControlledDofs()), (std::vector<int>{0, 1, 2, 3, 4, 5, 6}));

    /* The current input is the tool pose root^-1 * EE link * EELinkFromEE, as stored. */
    std::vector<mochi::real> input(7);
    osc.GetCurrentInput(input, mochi::test::ExpectOK{});
    auto const obsv = osc.GetCurrentObservationsFromMochi(mochi::test::ExpectOK{});
    mochi::TransformRT const rootFromLink =
        mochi::Invert(obsv.worldFromRoot) * obsv.worldFromEELink;
    mochi::TransformRT const rootFromTool = rootFromLink * osc.GetParams().EELinkFromEE;
    EXPECT_EQ(input, PoseValues(rootFromTool));
    EXPECT_NEAR(
        mochi::Norm(rootFromTool.GetTranslation() - rootFromLink.GetTranslation()),
        kToolOffset,
        1e-5_r);

    std::vector<mochi::real> target = input;
    target[0] += 0.04_r;
    target[2] -= 0.03_r;
    for (int step = 0; step < kNumSteps; ++step) {
      StepBoth(osc, twin, target);
    }
    osc.GetCurrentInput(input, mochi::test::ExpectOK{});
    EXPECT_LT(mochi::Abs(input[0] - target[0]), 0.04_r);
  }

  mochi::Context* _mochiContext = nullptr;
  mochi::Scene* _scene = nullptr;
  RoboticsContext* _botsCtx = nullptr;
  Bot* _bot = nullptr;
  mochi::Actor* _robot = nullptr;
};

} // namespace superdex::robotics::controller_float_input
