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

#include "controller_float_input_test.h"

#include <superdex_robotics/controllers/controller_basic_osc_pd.h>
#include <superdex_robotics/controllers/controller_mochi_articulated_pose.h>

#include <gtest/gtest.h>

#include <numeric>
#include <string_view>
#include <vector>

using namespace mochi;
using namespace mochi::test;
using namespace superdex::robotics;
using namespace superdex::robotics::controller_float_input;

namespace {

/* Teleop pick_place_block's BASIC_OSC_PD, with its 0.13 m tool offset. */
constexpr std::string_view kBasicOscParams = R"({
    "EELinkFromEE": {"rotation": [0, 0, 0, 1], "translation": [0, 0, 0.13]},
    "Kp_p": 900, "Kd_p": 75, "Kp_r": 30, "Kd_r": 3,
    "maxTranslationError": 0.05, "maxRotationError": 0.4})";
/* BASIC_JSC_PD gains for @p numDofs DOFs, with a saturation and a deadband. */
ControllerBasicJscPdParams JscParams(int numDofs) {
  ControllerBasicJscPdParams params;
  params.Kp.resize(numDofs, 200_r);
  params.Kd.resize(numDofs, 20_r);
  params.saturation.resize(numDofs, 40_r);
  params.deadband.resize(numDofs, 1e-3_r);
  return params;
}

TEST_F(ControllerFloatInputTest, BasicJscPdTakesJointPositions) {
  int const numDofs = _robot->GetNumDofs();
  auto& jsc = Create<ControllerBasicJscPd>("jsc", "", "");
  auto& twin = Create<ControllerBasicJscPd>("twin", "", "");
  jsc.SetParams(JscParams(numDofs), ExpectOK{});
  twin.SetParams(JscParams(numDofs), ExpectOK{});
  EXPECT_EQ(jsc.GetInputSize(), numDofs);
  EXPECT_EQ(jsc.GetInputKind(), ControllerInputKind::JointPositions);
  std::vector<int> everyDof(numDofs);
  std::iota(everyDof.begin(), everyDof.end(), 0);
  EXPECT_EQ(ToVector(jsc.GetControlledDofs()), everyDof);

  /* The current input is the live joint positions; holding it at rest outputs no effort. */
  std::vector<real> current(numDofs);
  jsc.GetCurrentInput(current, ExpectOK{});
  std::vector<real> positions(numDofs);
  _robot->GetArticulatedPose(positions, ExpectOK{});
  EXPECT_EQ(current, positions);
  EXPECT_EQ(StepBoth(jsc, twin, current), std::vector<real>(numDofs, 0_r));

  /* Toward a shifted target, with its target-velocity feed-forward on the first step. */
  std::vector<real> target = current;
  for (int i = 0; i < numDofs; ++i) {
    target[i] += (i % 2 == 0 ? 0.2_r : -0.1_r);
  }
  for (int step = 0; step < kNumSteps; ++step) {
    StepBoth(jsc, twin, target);
  }
  _robot->GetArticulatedPose(positions, ExpectOK{});
  EXPECT_NE(positions, current);
}

/* Without an actor the input size comes from the gains, it names no DOFs, and there is nothing to
 * read or drive. */
TEST_F(ControllerFloatInputTest, BasicJscPdWithoutAnActorSizesFromItsGains) {
  ControllerHandle const handle = _botsCtx->CreateController(
      ControllerBasicJscPd::TypeName(), nullptr, nullptr, "detached", ExpectOK{});
  auto* const jsc = static_cast<ControllerBasicJscPd*>(_botsCtx->GetController(handle));
  jsc->SetParams(JscParams(3), ExpectOK{});
  EXPECT_EQ(jsc->GetInputSize(), 3);
  EXPECT_TRUE(jsc->GetControlledDofs().empty());
  std::vector<real> values(3);
  jsc->GetCurrentInput(values, ExpectNotOK{});
  jsc->ComputeOutputFromInput(values, kDt, values, ExpectNotOK{});
}

TEST_F(ControllerFloatInputTest, BasicOscPdTakesTheToolPose) {
  auto& osc = Create<ControllerBasicOscPd>("osc", kBasicOscParams, kOscInitArgs);
  auto& twin = Create<ControllerBasicOscPd>("twin", kBasicOscParams, kOscInitArgs);
  ExpectPoseInputIsTheTypedPath(osc, twin);
}

/* Wrong sizes, an OSC before Initialize (which still takes 7 values but names no DOFs) and a
 * controller not driven by floats (the defaults: size 0, kind None, no DOFs) are refused. */
TEST_F(ControllerFloatInputTest, Refusals) {
  int const numDofs = _robot->GetNumDofs();
  std::vector<real> pose(7);
  std::vector<real> six(6);
  std::vector<real> efforts(numDofs);
  std::vector<real> shortEfforts(numDofs - 1);
  auto& osc = Create<ControllerBasicOscPd>("osc", kBasicOscParams, kOscInitArgs);
  osc.GetCurrentInput(six, ExpectNotOK{});
  osc.ComputeOutputFromInput(six, kDt, efforts, ExpectNotOK{});
  osc.ComputeOutputFromInput(pose, kDt, shortEfforts, ExpectNotOK{});
  auto& jsc = Create<ControllerBasicJscPd>("jsc", "", "");
  jsc.SetParams(JscParams(numDofs), ExpectOK{});
  jsc.GetCurrentInput(shortEfforts, ExpectNotOK{});
  jsc.ComputeOutputFromInput(shortEfforts, kDt, efforts, ExpectNotOK{});

  auto* const uninitialized = static_cast<ControllerBasicOscPd*>(
      _bot->CreateController(ControllerBasicOscPd::TypeName(), "uninitialized", ExpectOK{}));
  EXPECT_EQ(uninitialized->GetInputSize(), 7);
  EXPECT_TRUE(uninitialized->GetControlledDofs().empty());
  uninitialized->GetCurrentInput(pose, ExpectNotOK{});
  uninitialized->ComputeOutputFromInput(pose, kDt, efforts, ExpectNotOK{});

  ControllerBase* const notFloats =
      _bot->CreateController(ControllerMochiArticulatedPose::TypeName(), "pose", ExpectOK{});
  EXPECT_EQ(notFloats->GetInputSize(), 0);
  EXPECT_EQ(notFloats->GetInputKind(), ControllerInputKind::None);
  EXPECT_TRUE(notFloats->GetControlledDofs().empty());
  notFloats->GetCurrentInput(pose, ExpectNotOK{});
  notFloats->ComputeOutputFromInput(pose, kDt, efforts, ExpectNotOK{});
}

} // namespace
