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

#include <gtest/gtest.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_physics/mochi_physics.h>
#include <superdex_robotics/core/loader.h>
#include <superdex_robotics/utils/bot_utils.h>
#include "simulation/ik_preview_session.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>

using namespace mochi;
using namespace mochi::test;
using namespace superdex::robotics;
using namespace superdex::studio;

namespace {

constexpr real kTolerance = 3e-3_r;
constexpr int kArm = 2;
constexpr int kTip = 4;

BotLinkPrefab Link(char const* name, int parent) {
  BotLinkPrefab link;
  link.name = name;
  link.parentLink = parent;
  link.mass = 1_r;
  link.centerOfMass = Real3{};
  link.momentOfInertia = Real6{1_r, 0_r, 0_r, 1_r, 0_r, 1_r};
  return link;
}

BotJointPrefab Joint(char const* name, ArticulatedJointType type, Real3 translation = {}) {
  BotJointPrefab joint;
  joint.name = name;
  joint.type = type;
  joint.axis =
      type == ArticulatedJointType::Prismatic ? Real3{1_r, 0_r, 0_r} : Real3{0_r, 0_r, 1_r};
  joint.parentLinkFromJoint = TransformRT{translation};
  return joint;
}

BotPrefab Robot(ArticulatedJointType root = ArticulatedJointType::Hard) {
  BotPrefab bot;
  bot.name = "studio_ik_preview";
  bot.worldFromRoot = TransformRT{Quaternion::RotationY(0.25_r), Real3{1_r, 2_r, 3_r}};
  bot.links = {
      Link("root", kIndexNone),
      Link("a_weld", 0),
      Link("b_arm", 0),
      Link("c_slide", kArm),
      Link("d_tip", 3)};
  bot.joints = {
      Joint("root_joint", root),
      Joint("weld_joint", ArticulatedJointType::Hard, {0_r, 0.2_r, 0_r}),
      Joint("arm_joint", ArticulatedJointType::Revolute, {0.3_r, 0_r, 0_r}),
      Joint("slide_joint", ArticulatedJointType::Prismatic, {0.5_r, 0_r, 0_r}),
      Joint("tip_joint", ArticulatedJointType::Hard, {0.2_r, 0_r, 0_r})};
  bot.defaultPose = {0.15_r, 0.1_r};
  RebuildBotData(bot, ExpectOK{});
  return bot;
}

BotPrefab CoupledRobot() {
  auto bot = Robot();
  bot.joints[3].type = ArticulatedJointType::Revolute;
  bot.joints[3].axis = Real3{0_r, 0_r, 1_r};
  RebuildBotData(bot, ExpectOK{});
  return bot;
}
BotIKJointCouplingConfig Coupling(real ratio = 0.5_r) {
  BotIKJointCouplingConfig coupling;
  coupling.jointA = "arm_joint";
  coupling.jointB = "slide_joint";
  coupling.ratio = ratio;
  return coupling;
}

DynamicArray<TransformRT> Expected(TransformRT const& base, real angle, real extension) {
  auto const arm = base * TransformRT{Quaternion::RotationZ(angle), Real3{0.3_r, 0_r, 0_r}};
  auto const slide = arm * TransformRT{Real3{0.5_r + extension, 0_r, 0_r}};
  return {
      base,
      base * TransformRT{Real3{0_r, 0.2_r, 0_r}},
      arm,
      slide,
      slide * TransformRT{Real3{0.2_r, 0_r, 0_r}}};
}

void ExpectLinks(Span<TransformRT const> actual, Span<TransformRT const> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (int i = 0; i < isize(actual); ++i) {
    SCOPED_TRACE(i);
    EXPECT_TRUE(NearEqual(actual[i], expected[i], kTolerance));
  }
}

BotIKConfig Config(real angle = 0.4_r, real extension = 0.3_r) {
  BotIKConfig spec;
  spec.outputBot = "test.superdex_bot";
  spec.createParams.solverParams.maxIter = 80;
  spec.createParams.solverParams.absTol = 1e-4_r;
  spec.createParams.solverParams.relTol = 1e-8_r;
  spec.createParams.solverParams.positionErrorThres = 2e-3_r;
  spec.createParams.solverParams.rotationErrorThres = 2e-3_r;
  spec.createParams.solverParams.maxElapsedTimeSeconds = 0;
  BotIKGoalConfig goal;
  goal.name = "tip";
  goal.outputBotLink = "d_tip";
  goal.positionStiffness = 1000_r;
  goal.rotationStiffness = 1000_r;
  SetIKGoalConstraintTransform(goal, Expected({}, angle, extension)[kTip]);
  spec.goals.push_back(goal);
  return spec;
}

struct CountingLoader : FileBotLoader {
  mutable int loads = 0;
  bool fail = false;
  ShapeHandle LoadShape(
      std::string_view,
      Real3 const&,
      TransformRT const&,
      Context* context,
      Error& error) const override {
    ++loads;
    if (fail) {
      MOCHI_ERROR_SET(error, "Deliberate loader failure");
      return {};
    }
    real const coordinates[] = {
        -0.4_r, -0.4_r, -0.4_r, 0.4_r, -0.4_r, -0.4_r, 0_r, 0.4_r, -0.4_r, 0_r, 0_r, 0.4_r};
    int const connectivity[] = {0, 1, 2, 3};
    return context->CreateTetMeshShape(coordinates, connectivity, error);
  }
};

class IKPreviewTest : public testing::Test {
 protected:
  void SetUp() override {
    _context = CreateContext(0);
    ASSERT_NE(_context, nullptr);
  }
  void TearDown() override {
    DestroyContext(_context);
  }
  Context* _context = nullptr;
  FileBotLoader _loader;
};

TEST_F(IKPreviewTest, OutputBotConfiguration_NativeScaleMatchesCreatedBot) {
  auto bot = Robot();
  bot.scale = 2_r;
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic)) << diagnostic;
  DynamicArray<real> const positions{0.4_r, -0.05_r};
  ASSERT_TRUE(session.SetOutputBotConfiguration(positions, diagnostic)) << diagnostic;
  auto seeded = bot;
  seeded.defaultPose = positions;
  Error error;
  std::unique_ptr<BotIK, decltype(&BotIK::Destroy)> reference{
      BotIK::Create(_context, seeded, BotIKCreateParams{}, _loader, error), &BotIK::Destroy};
  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  ASSERT_NE(reference, nullptr);
  ExpectLinks(session.GetOutputBotWorldFromLinks(), reference->GetWorldFromLinks());
}

TEST_F(IKPreviewTest, Prepare_CopiesNonidentityAuthoredSeedWithoutSolverOrOutputMutation) {
  auto bot = Robot();
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic)) << diagnostic;
  EXPECT_FALSE(session.HasSolver());
  EXPECT_EQ(session.GetWorldFrameFromOutputBaseFrame(), bot.worldFromRoot);
  EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, 0.15_r, 0.1_r));
  EXPECT_EQ(
      session.GetOutputBotLinkNames(),
      (DynamicArray<std::string>{"root", "a_weld", "b_arm", "c_slide", "d_tip"}));
  bot.defaultPose[0] = 0.9_r;
  EXPECT_NE(session.GetOutputBot().defaultPose, bot.defaultPose);
  EXPECT_NE(session.GetOutputBotJointPositions(), bot.defaultPose);
}

TEST_F(IKPreviewTest, Prepare_InvalidTopologyAndSeedPreservePreviousGeneration) {
  auto const bot = Robot();
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  for (int fault = 0; fault < 7; ++fault) {
    auto bad = bot;
    if (fault == 0) {
      bad.links[kTip].name = bad.links[kArm].name;
    }
    if (fault == 1) {
      bad.links[kTip].parentLink = kTip;
    }
    if (fault == 2) {
      bad.links[kTip]._index = 90;
    }
    if (fault == 3) {
      bad._numDofs = 90;
    }
    if (fault == 4) {
      bad.defaultPose.pop_back();
    }
    if (fault == 5) {
      bad.defaultPose[0] = std::numeric_limits<real>::quiet_NaN();
    }
    if (fault == 6) {
      bad.links[0]._childrenIndices.clear();
    }
    SCOPED_TRACE(fault);
    EXPECT_FALSE(session.Prepare(bad, diagnostic));
    EXPECT_FALSE(diagnostic.empty());
    EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
    EXPECT_EQ(session.GetWorldFrameFromOutputBaseFrame(), bot.worldFromRoot);
  }
}

TEST_F(IKPreviewTest, Step_LivePoseUpdatesWarmStartAndStopRestoresSeed) {
  auto const bot = Robot();
  auto spec = Config();
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  for (real const angle : {0.4_r, -0.2_r, 0.3_r}) {
    SetIKGoalConstraintTransform(spec.goals[0], Expected({}, angle, 0.3_r)[kTip]);
    ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
    EXPECT_TRUE(session.HasSolver());
    EXPECT_NEAR(session.GetOutputBotJointPositions()[0], angle, kTolerance);
    EXPECT_NEAR(session.GetOutputBotJointPositions()[1], 0.3_r, kTolerance);
    ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, angle, 0.3_r));
    EXPECT_EQ(session.GetWorldFrameFromOutputBaseFrame(), bot.worldFromRoot);
    EXPECT_GE(session.GetLastSolveMilliseconds(), 0);
  }
  auto const solved = session.GetOutputBotWorldFromLinks();
  spec.goals[0].enabled = false;
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  EXPECT_EQ(session.GetStatus(), "No Active Goals");
  EXPECT_EQ(session.GetLastSolveMilliseconds(), 0);
  EXPECT_EQ(session.GetOutputBotWorldFromLinks(), solved);
  session.Stop();
  EXPECT_FALSE(session.HasSolver());
  EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, 0.15_r, 0.1_r));
  spec.goals[0].enabled = true;
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  EXPECT_NEAR(session.GetOutputBotJointPositions()[0], 0.3_r, kTolerance);
  EXPECT_EQ(session.GetOutputBot().defaultPose, bot.defaultPose);
}

TEST_F(IKPreviewTest, Step_LinkRelativeFrameUpdatesGoalAttachment) {
  auto const bot = Robot();
  auto spec = Config();
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  for (real const angle : {0.4_r, -0.2_r, 0.3_r}) {
    auto& goal = spec.goals[0];
    goal.linkFrameFromConstraintFrame =
        TransformRT{Quaternion::RotationX(angle), Real3{0.1_r, angle * 0.1_r, 0.05_r}};
    SetIKGoalConstraintTransform(
        goal, Expected({}, angle, 0.3_r)[kTip] * goal.linkFrameFromConstraintFrame);
    ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
    EXPECT_EQ(session.GetStatus(), "Goals Reached");
    EXPECT_NEAR(session.GetOutputBotJointPositions()[0], angle, kTolerance);
    EXPECT_NEAR(session.GetOutputBotJointPositions()[1], 0.3_r, kTolerance);
    ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, angle, 0.3_r));
    EXPECT_TRUE(NearEqual(
        session.GetOutputBotWorldFromLinks()[kTip] * goal.linkFrameFromConstraintFrame,
        bot.worldFromRoot * IKGoalConstraintTransform(goal),
        kTolerance));
  }
}

TEST_F(IKPreviewTest, Step_ZeroStiffnessRemovalAndLinkRetargetClearOldConstraints) {
  auto const bot = Robot();
  auto spec = Config();
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  spec.goals[0].positionStiffness = 0_r;
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  EXPECT_NEAR(session.GetOutputBotJointPositions()[0], 0.4_r, kTolerance);
  EXPECT_NEAR(session.GetOutputBotJointPositions()[1], 0.1_r, kTolerance);
  spec.goals[0].rotationStiffness = 0_r;
  auto const previous = session.GetOutputBotJointPositions();
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  EXPECT_EQ(session.GetOutputBotJointPositions(), previous);
  EXPECT_EQ(session.GetStatus(), "No Active Goals");
  spec.goals[0].rotationStiffness = 2000_r;
  spec.goals[0].outputBotLink = "b_arm";
  SetIKGoalConstraintTransform(spec.goals[0], Expected({}, -0.25_r, 0.1_r)[kArm]);
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  EXPECT_NEAR(session.GetOutputBotJointPositions()[0], -0.25_r, kTolerance);
  spec.goals.clear();
  auto const pose = session.GetOutputBotJointPositions();
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  EXPECT_EQ(session.GetOutputBotJointPositions(), pose);
  EXPECT_EQ(session.GetStatus(), "No Active Goals");
  spec = Config(0.2_r, 0.35_r);
  spec.goals[0].rotationStiffness = 0_r;
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  EXPECT_LT(
      Norm(
          session.GetOutputBotWorldFromLinks()[kTip].GetTranslation() -
          Expected(bot.worldFromRoot, 0.2_r, 0.35_r)[kTip].GetTranslation()),
      2_r * kTolerance);
}

TEST_F(IKPreviewTest, Step_LiveStiffnessChangesConflictingGoalCompromise) {
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(Robot(), diagnostic));
  auto spec = Config(0.6_r);
  spec.goals[0].positionStiffness = 0_r;
  BotIKGoalConfig arm = spec.goals[0];
  arm.name = "arm";
  arm.outputBotLink = "b_arm";
  SetIKGoalConstraintTransform(arm, Expected({}, 0_r, 0.1_r)[kArm]);
  spec.goals.push_back(arm);
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  real const compromise = session.GetOutputBotJointPositions()[0];
  EXPECT_GT(compromise, 0.2_r);
  EXPECT_LT(compromise, 0.4_r);
  spec.goals[0].rotationStiffness *= 4_r;
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  EXPECT_GT(session.GetOutputBotJointPositions()[0], compromise + 0.1_r);
}

TEST_F(IKPreviewTest, Step_UnreachableBestEffortPublishesFiniteOutputAndAcceptsEdits) {
  auto spec = Config();
  spec.goals[0].baseOrInputLinkFrameFromConstraintFrame.SetTranslation(
      spec.goals[0].baseOrInputLinkFrameFromConstraintFrame.GetTranslation() +
      Real3{0_r, 0_r, 0.5_r});
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(Robot(), diagnostic));
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  EXPECT_TRUE(diagnostic.empty());
  EXPECT_NE(session.GetStatus().find("Best Effort"), std::string::npos);
  for (auto const& link : session.GetOutputBotWorldFromLinks()) {
    EXPECT_TRUE(IsFinite(link));
  }
  for (real const q : session.GetOutputBotJointPositions()) {
    EXPECT_TRUE(IsFinite(q));
  }
  EXPECT_GT(Abs(session.GetOutputBotJointPositions()[0] - 0.15_r), 0.05_r);
  spec = Config(-0.1_r, 0.2_r);
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  EXPECT_NEAR(session.GetOutputBotJointPositions()[0], -0.1_r, kTolerance);
}

TEST_F(IKPreviewTest, Step_FreeBaseMovesButGoalsRetainAuthoredAnchorAcrossTicks) {
  auto const bot = Robot(ArticulatedJointType::Free);
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  auto spec = Config();
  spec.createParams.fixedBase = false;
  spec.goals[0].outputBotLink = "root";
  TransformRT const baseGoal{Quaternion::RotationX(0.3_r), Real3{0.3_r, -0.2_r, 0.1_r}};
  SetIKGoalConstraintTransform(spec.goals[0], baseGoal);
  for (int tick = 0; tick < 3; ++tick) {
    ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
    EXPECT_TRUE(NearEqual(
        session.GetOutputBotWorldFromLinks()[0], bot.worldFromRoot * baseGoal, kTolerance));
    EXPECT_EQ(session.GetWorldFrameFromOutputBaseFrame(), bot.worldFromRoot);
    EXPECT_EQ(session.GetOutputBotJointPositions().size(), bot.defaultPose.size());
  }
  spec.createParams.fixedBase = true;
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  EXPECT_TRUE(NearEqual(session.GetOutputBotWorldFromLinks()[0], bot.worldFromRoot, kTolerance));
  EXPECT_NE(session.GetStatus().find("Best Effort"), std::string::npos);
  session.Stop();
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, 0.15_r, 0.1_r));
}

TEST_F(IKPreviewTest, Step_CreationOptionsRecreateButNumericalUpdatesKeepWarmStart) {
  auto bot = Robot(ArticulatedJointType::Free);
  for (auto& link : bot.links) {
    link.shapeFile = "synthetic";
  }
  CountingLoader loader;
  auto spec = Config();
  spec.createParams.excludeCollisionShapes = false;
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  int const shapes = _context->GetNumShapes();
  ASSERT_TRUE(session.Step(_context, loader, spec, diagnostic)) << diagnostic;
  EXPECT_EQ(loader.loads, isize(bot.links));
  auto const pose = session.GetOutputBotJointPositions();
  spec.goals[0].enabled = false;
  spec.createParams.solverParams.maxIter = 43;
  spec.createParams.solverParams.absTol = 2e-4_r;
  spec.createParams.solverParams.lineSearchMaxIter = 17;
  spec.createParams.solverParams.maxElapsedTimeSeconds = 0.005;
  ASSERT_TRUE(session.Step(_context, loader, spec, diagnostic));
  EXPECT_EQ(loader.loads, isize(bot.links));
  EXPECT_EQ(session.GetOutputBotJointPositions(), pose);
  spec.createParams.fixedBase = false;
  ASSERT_TRUE(session.Step(_context, loader, spec, diagnostic));
  EXPECT_EQ(loader.loads, 2 * isize(bot.links));
  EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
  spec.createParams.excludeCollisionShapes = true;
  ASSERT_TRUE(session.Step(_context, loader, spec, diagnostic));
  EXPECT_EQ(loader.loads, 2 * isize(bot.links));
  EXPECT_EQ(_context->GetNumShapes(), shapes);
  session.Stop();
  EXPECT_EQ(_context->GetNumShapes(), shapes);
}

TEST_F(IKPreviewTest, CanSolve_MissingDuplicateAndMalformedGoalsBlockWithoutDestroyingDisplay) {
  IKPreviewSession session;
  std::string diagnostic;
  auto spec = Config();
  EXPECT_FALSE(session.CanSolve(spec, diagnostic));
  ASSERT_TRUE(session.Prepare(Robot(), diagnostic));
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  auto const links = session.GetOutputBotWorldFromLinks();
  for (int fault = 0; fault < 3; ++fault) {
    auto bad = spec;
    if (fault == 0) {
      bad.goals[0].outputBotLink = "missing";
      bad.goals[0].enabled = false;
    }
    if (fault == 1) {
      bad.goals.push_back(BotIKGoalConfig{bad.goals[0]});
    }
    if (fault == 2) {
      bad.goals[0].rotationStiffness = -1_r;
    }
    EXPECT_FALSE(session.CanSolve(bad, diagnostic));
    EXPECT_FALSE(session.Step(_context, _loader, bad, diagnostic));
    EXPECT_FALSE(diagnostic.empty());
    EXPECT_EQ(session.GetOutputBotWorldFromLinks(), links);
  }
  EXPECT_FALSE(session.Step(nullptr, _loader, spec, diagnostic));
  EXPECT_EQ(session.GetOutputBotWorldFromLinks(), links);
  EXPECT_EQ(session.GetStatus(), diagnostic);
  spec = Config(-0.2_r, 0.25_r);
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  EXPECT_TRUE(diagnostic.empty());
  EXPECT_NEAR(session.GetOutputBotJointPositions()[0], -0.2_r, kTolerance);
  EXPECT_NEAR(session.GetOutputBotJointPositions()[1], 0.25_r, kTolerance);
}

TEST_F(IKPreviewTest, Prepare_RebindUsesLinkNamesAndNewAuthoredAnchorWithoutChangingGoals) {
  auto bot = Robot();
  auto spec = Config();
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  auto const document = SerializeStudioIKConfig(spec);
  bot.links[1].name = "z_weld";
  bot.worldFromRoot = TransformRT{Quaternion::RotationX(-0.4_r), Real3{-2_r, 1_r, 0_r}};
  RebuildBotData(bot, ExpectOK{});
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  EXPECT_FALSE(session.HasSolver());
  EXPECT_EQ(session.GetWorldFrameFromOutputBaseFrame(), bot.worldFromRoot);
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  int const tip = FindLinkIndexByName(bot, "d_tip");
  EXPECT_TRUE(NearEqual(
      session.GetOutputBotWorldFromLinks()[tip],
      bot.worldFromRoot * IKGoalConstraintTransform(spec.goals[0]),
      kTolerance));
  EXPECT_EQ(SerializeStudioIKConfig(spec), document);
}

TEST_F(IKPreviewTest, Lifecycle_LoaderFailureStopAndDestructionLeaveContextUsable) {
  auto bot = Robot();
  for (auto& link : bot.links) {
    link.shapeFile = "synthetic";
  }
  CountingLoader loader;
  auto spec = Config();
  spec.createParams.excludeCollisionShapes = false;
  std::string diagnostic;
  int const shapes = _context->GetNumShapes();
  for (int repeat = 0; repeat < 3; ++repeat) {
    IKPreviewSession session;
    ASSERT_TRUE(session.Prepare(bot, diagnostic));
    loader.fail = true;
    EXPECT_FALSE(session.Step(_context, loader, spec, diagnostic));
    EXPECT_FALSE(session.HasSolver());
    EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
    EXPECT_EQ(_context->GetNumShapes(), shapes);
    loader.fail = false;
    ASSERT_TRUE(session.Step(_context, loader, spec, diagnostic));
  }
  EXPECT_EQ(_context->GetNumShapes(), shapes);
  auto* scene = _context->CreateScene("still_usable");
  ASSERT_NE(scene, nullptr);
  _context->DestroyScene(scene);
}

TEST(IKSkinMappingTest, ComposedNamesReorderBonesAndPreserveWorldRootPlacement) {
  auto bot = Robot();
  auto const links = Expected(bot.worldFromRoot, 0.3_r, 0.2_r);
  bot._skinBoneLinks = {"d_tip", "root", "b_arm"};
  DynamicArray<TransformRT> bones;
  std::string diagnostic;
  ASSERT_TRUE(MapIKSkinTransforms(bot, links, bones, diagnostic));
  EXPECT_EQ(bones, (DynamicArray<TransformRT>{links[kTip], links[0], links[kArm]}));
  auto const previous = bones;
  bot._skinBoneLinks[0] = "missing";
  EXPECT_FALSE(MapIKSkinTransforms(bot, links, bones, diagnostic));
  EXPECT_EQ(bones, previous);
  EXPECT_FALSE(MapIKSkinTransforms(bot, {}, bones, diagnostic));
  EXPECT_EQ(bones, previous);
  bot._skinBoneLinks.clear();
  ASSERT_TRUE(MapIKSkinTransforms(bot, links, bones, diagnostic));
  EXPECT_EQ(bones, links);
  bot.links[kTip].name = bot.links[kArm].name;
  EXPECT_FALSE(MapIKSkinTransforms(bot, links, bones, diagnostic));
}

TEST_F(IKPreviewTest, Step_InitiallyEmptyGoalsKeepAuthoredSeedWithoutSolving) {
  auto const bot = Robot();
  auto spec = Config();
  spec.goals.clear();
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  auto const seedLinks = session.GetOutputBotWorldFromLinks();
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  EXPECT_EQ(session.GetStatus(), "No Active Goals");
  EXPECT_EQ(session.GetLastSolveMilliseconds(), 0);
  EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
  EXPECT_EQ(session.GetOutputBotWorldFromLinks(), seedLinks);
}

TEST_F(IKPreviewTest, Step_LimitedIterationsResumeSolvedConfiguration) {
  auto const bot = Robot();
  auto spec = Config(0.8_r, 0.4_r);
  spec.createParams.solverParams.maxIter = 1;
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  auto const first = session.GetOutputBotJointPositions();
  auto const goal = bot.worldFromRoot * IKGoalConstraintTransform(spec.goals[0]);
  real const firstError =
      Norm(session.GetOutputBotWorldFromLinks()[kTip].GetTranslation() - goal.GetTranslation());
  ASSERT_GT(firstError, kTolerance);
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  EXPECT_NE(session.GetOutputBotJointPositions(), first);
  EXPECT_LT(
      Norm(session.GetOutputBotWorldFromLinks()[kTip].GetTranslation() - goal.GetTranslation()),
      firstError);
}

TEST_F(IKPreviewTest, OutputBotConfiguration_ConfiguredFKAndEmptyDefaultPreserveAuthoredSkinData) {
  auto bot = Robot();
  bot.skin.emplace();
  bot.skin->shapeFile = "authored_skin";
  bot._skinBoneLinks = {"d_tip", "root", "b_arm"};
  auto const authored = SReflect::ToJsonString(bot, false);
  IKPreviewSession session;
  std::string diagnostic;
  DynamicArray<real> positions{-0.35_r, 0.25_r};
  EXPECT_FALSE(session.SetOutputBotConfiguration(positions, diagnostic));
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.SetOutputBotConfiguration(positions, diagnostic)) << diagnostic;
  EXPECT_TRUE(diagnostic.empty());
  EXPECT_FALSE(session.HasSolver());
  EXPECT_EQ(session.GetOutputBotJointPositions(), positions);
  auto const expected = Expected(bot.worldFromRoot, positions[0], positions[1]);
  ExpectLinks(session.GetOutputBotWorldFromLinks(), expected);
  DynamicArray<TransformRT> bones;
  ASSERT_TRUE(MapIKSkinTransforms(
      session.GetOutputBot(), session.GetOutputBotWorldFromLinks(), bones, diagnostic));
  ExpectLinks(bones, DynamicArray<TransformRT>{expected[kTip], expected[0], expected[kArm]});
  auto spec = Config();
  spec.outputBotConfiguration = positions;
  spec.goals[0].positionStiffness = 0_r;
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  EXPECT_NEAR(session.GetOutputBotJointPositions()[1], positions[1], kTolerance);
  EXPECT_EQ(SReflect::ToJsonString(session.GetOutputBot(), false), authored);
  EXPECT_EQ(SReflect::ToJsonString(bot, false), authored);
  auto const solvedPositions = session.GetOutputBotJointPositions();
  ASSERT_TRUE(session.SetOutputBotConfiguration(session.GetOutputBotJointPositions(), diagnostic));
  EXPECT_FALSE(session.HasSolver());
  EXPECT_EQ(session.GetOutputBotJointPositions(), solvedPositions);
  session.Stop();
  EXPECT_EQ(session.GetOutputBotJointPositions(), solvedPositions);
  ASSERT_TRUE(session.SetOutputBotConfiguration({}, diagnostic));
  EXPECT_FALSE(session.HasSolver());
  EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, 0.15_r, 0.1_r));
  EXPECT_EQ(SReflect::ToJsonString(session.GetOutputBot(), false), authored);
}

TEST_F(IKPreviewTest, OutputBotConfiguration_InvalidEditsPreserveSeedSolverAndSolvedDisplay) {
  auto const bot = Robot();
  auto spec = Config();
  spec.outputBotConfiguration = {-0.3_r, 0.2_r};
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.SetOutputBotConfiguration(spec.outputBotConfiguration, diagnostic));
  auto const seedLinks = session.GetOutputBotWorldFromLinks();
  DynamicArray<real> const invalidSeed{std::numeric_limits<real>::quiet_NaN(), 0.2_r};
  EXPECT_FALSE(session.SetOutputBotConfiguration(invalidSeed, diagnostic));
  EXPECT_FALSE(session.HasSolver());
  EXPECT_EQ(session.GetOutputBotJointPositions(), spec.outputBotConfiguration);
  EXPECT_EQ(session.GetOutputBotWorldFromLinks(), seedLinks);
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  auto const solvedLinks = session.GetOutputBotWorldFromLinks();
  auto const solvedPositions = session.GetOutputBotJointPositions();
  for (int fault = 0; fault < 4; ++fault) {
    SCOPED_TRACE(fault);
    auto bad = spec;
    if (fault == 0) {
      bad.outputBotConfiguration.pop_back();
    } else if (fault == 1) {
      bad.outputBotConfiguration.push_back(0_r);
    } else {
      bad.outputBotConfiguration[fault - 2] = fault == 2 ? std::numeric_limits<real>::quiet_NaN()
                                                         : std::numeric_limits<real>::infinity();
    }
    EXPECT_FALSE(session.SetOutputBotConfiguration(bad.outputBotConfiguration, diagnostic));
    EXPECT_FALSE(diagnostic.empty());
    EXPECT_FALSE(session.CanSolve(bad, diagnostic));
    EXPECT_FALSE(session.Step(_context, _loader, bad, diagnostic));
    EXPECT_TRUE(session.HasSolver());
    EXPECT_EQ(session.GetOutputBotJointPositions(), solvedPositions);
    EXPECT_EQ(session.GetOutputBotWorldFromLinks(), solvedLinks);
  }
  auto changed = spec;
  changed.outputBotConfiguration = {0.6_r, 0.35_r};
  EXPECT_FALSE(session.Step(nullptr, _loader, changed, diagnostic));
  EXPECT_TRUE(session.HasSolver());
  EXPECT_EQ(session.GetOutputBotJointPositions(), solvedPositions);
  EXPECT_EQ(session.GetOutputBotWorldFromLinks(), solvedLinks);
  changed.goals[0].outputBotLink = "missing";
  EXPECT_FALSE(session.Step(_context, _loader, changed, diagnostic));
  EXPECT_EQ(session.GetOutputBotWorldFromLinks(), solvedLinks);
  session.Stop();
  EXPECT_EQ(session.GetOutputBotJointPositions(), spec.outputBotConfiguration);
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, -0.3_r, 0.2_r));
}

TEST_F(IKPreviewTest, OutputBotConfiguration_FinitePositionsWithOverflowingWorldFKAreRejected) {
  auto bot = Robot();
  real const large = std::numeric_limits<real>::max() * 0.75_r;
  bot.worldFromRoot = TransformRT{Real3{large, 0_r, 0_r}};
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic)) << diagnostic;
  auto const previous = session.GetOutputBotWorldFromLinks();
  auto spec = Config();
  spec.goals.clear();
  spec.outputBotConfiguration = {0_r, large};
  EXPECT_FALSE(session.CanSolve(spec, diagnostic));
  EXPECT_FALSE(diagnostic.empty());
  EXPECT_FALSE(session.SetOutputBotConfiguration(spec.outputBotConfiguration, diagnostic));
  EXPECT_FALSE(session.Step(_context, _loader, spec, diagnostic));
  EXPECT_FALSE(session.HasSolver());
  EXPECT_EQ(session.GetOutputBotWorldFromLinks(), previous);
  EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
  session.Stop();
  EXPECT_EQ(session.GetOutputBotWorldFromLinks(), previous);
}

TEST_F(IKPreviewTest, OutputBotConfiguration_PreflightIsNonmutatingAndStepSynchronizesEdits) {
  auto const bot = Robot();
  auto spec = Config();
  spec.goals.clear();
  spec.outputBotConfiguration = {-0.2_r, 0.35_r};
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.CanSolve(spec, diagnostic));
  EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
  EXPECT_FALSE(session.HasSolver());
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  EXPECT_EQ(session.GetOutputBotJointPositions(), spec.outputBotConfiguration);
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, -0.2_r, 0.35_r));
  spec.outputBotConfiguration = {0.5_r, -0.1_r};
  ASSERT_TRUE(session.CanSolve(spec, diagnostic));
  EXPECT_EQ(session.GetOutputBotJointPositions(), (DynamicArray<real>{-0.2_r, 0.35_r}));
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, 0.5_r, -0.1_r));
  spec.outputBotConfiguration.clear();
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, 0.15_r, 0.1_r));
}

TEST_F(IKPreviewTest, OutputBotConfiguration_StopAndCreationChangesRestoreConfiguredSeed) {
  auto bot = Robot(ArticulatedJointType::Free);
  for (auto& link : bot.links) {
    link.shapeFile = "synthetic";
  }
  CountingLoader loader;
  auto spec = Config();
  spec.outputBotConfiguration = {-0.25_r, 0.2_r};
  spec.createParams.excludeCollisionShapes = false;
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.Step(_context, loader, spec, diagnostic));
  auto const solved = session.GetOutputBotJointPositions();
  EXPECT_NE(solved, spec.outputBotConfiguration);
  spec.goals[0].enabled = false;
  spec.createParams.solverParams.maxIter = 43;
  ASSERT_TRUE(session.Step(_context, loader, spec, diagnostic));
  EXPECT_EQ(loader.loads, isize(bot.links));
  EXPECT_EQ(session.GetOutputBotJointPositions(), solved);
  spec.createParams.fixedBase = false;
  ASSERT_TRUE(session.Step(_context, loader, spec, diagnostic));
  EXPECT_EQ(loader.loads, 2 * isize(bot.links));
  EXPECT_EQ(session.GetOutputBotJointPositions(), spec.outputBotConfiguration);
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, -0.25_r, 0.2_r));
  spec.createParams.excludeCollisionShapes = true;
  ASSERT_TRUE(session.Step(_context, loader, spec, diagnostic));
  EXPECT_EQ(loader.loads, 2 * isize(bot.links));
  spec.goals[0].enabled = true;
  ASSERT_TRUE(session.Step(_context, loader, spec, diagnostic));
  session.Stop();
  EXPECT_EQ(session.GetOutputBotJointPositions(), spec.outputBotConfiguration);
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, -0.25_r, 0.2_r));
  spec.goals[0].enabled = false;
  ASSERT_TRUE(session.Step(_context, loader, spec, diagnostic));
  EXPECT_EQ(session.GetOutputBotJointPositions(), spec.outputBotConfiguration);
  loader.fail = true;
  spec.createParams.excludeCollisionShapes = false;
  EXPECT_FALSE(session.Step(_context, loader, spec, diagnostic));
  EXPECT_FALSE(session.HasSolver());
  EXPECT_EQ(session.GetOutputBotJointPositions(), spec.outputBotConfiguration);
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, -0.25_r, 0.2_r));
}

TEST_F(IKPreviewTest, OutputBotConfiguration_ContextRefreshRestoresConfiguredSeed) {
  auto const bot = Robot();
  auto spec = Config();
  spec.outputBotConfiguration = {-0.25_r, 0.2_r};
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  EXPECT_NE(session.GetOutputBotJointPositions(), spec.outputBotConfiguration);
  session.Stop();
  DestroyContext(_context);
  _context = CreateContext(0);
  ASSERT_NE(_context, nullptr);
  spec.goals[0].enabled = false;
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  EXPECT_EQ(session.GetOutputBotJointPositions(), spec.outputBotConfiguration);
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, -0.25_r, 0.2_r));
  EXPECT_TRUE(session.HasSolver());
}

TEST_F(IKPreviewTest, OutputBotConfiguration_UnchangedSeedResumesLimitedIterationWarmStart) {
  auto const bot = Robot();
  auto spec = Config(0.8_r, 0.4_r);
  spec.outputBotConfiguration = {-0.25_r, 0.2_r};
  spec.createParams.solverParams.maxIter = 1;
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  auto const first = session.GetOutputBotJointPositions();
  auto const goal = bot.worldFromRoot * IKGoalConstraintTransform(spec.goals[0]);
  real const firstError =
      Norm(session.GetOutputBotWorldFromLinks()[kTip].GetTranslation() - goal.GetTranslation());
  ASSERT_GT(firstError, kTolerance);
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  EXPECT_NE(session.GetOutputBotJointPositions(), first);
  EXPECT_LT(
      Norm(session.GetOutputBotWorldFromLinks()[kTip].GetTranslation() - goal.GetTranslation()),
      firstError);
}

TEST_F(IKPreviewTest, OutputBotConfiguration_CreationSeedsHardAndFreeRootsWithEitherBaseOption) {
  for (auto const root : {ArticulatedJointType::Hard, ArticulatedJointType::Free}) {
    for (bool const fixedBase : {false, true}) {
      SCOPED_TRACE(static_cast<int>(root));
      SCOPED_TRACE(fixedBase);
      auto const bot = Robot(root);
      auto const authored = SReflect::ToJsonString(bot, false);
      auto spec = Config();
      spec.outputBotConfiguration = {-0.4_r, 0.35_r};
      spec.createParams.fixedBase = fixedBase;
      spec.goals[0].outputBotLink = "root";
      SetIKGoalConstraintTransform(spec.goals[0], TransformRT{});
      IKPreviewSession session;
      std::string diagnostic;
      ASSERT_TRUE(session.Prepare(bot, diagnostic));
      ASSERT_TRUE(session.SetOutputBotConfiguration(spec.outputBotConfiguration, diagnostic));
      // An active root goal publishes the actual solver pose, not just the session's FK display.
      ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
      ExpectLinks(
          session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, -0.4_r, 0.35_r));
      TransformRT const offset{Quaternion::RotationX(0.3_r), Real3{0.3_r, -0.2_r, 0.1_r}};
      SetIKGoalConstraintTransform(spec.goals[0], offset);
      ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
      auto const base = root == ArticulatedJointType::Free && !fixedBase
          ? bot.worldFromRoot * offset
          : bot.worldFromRoot;
      ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(base, -0.4_r, 0.35_r));
      ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
      ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(base, -0.4_r, 0.35_r));
      EXPECT_EQ(session.GetWorldFrameFromOutputBaseFrame(), bot.worldFromRoot);
      session.Stop();
      ExpectLinks(
          session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, -0.4_r, 0.35_r));
      EXPECT_EQ(SReflect::ToJsonString(session.GetOutputBot(), false), authored);
    }
  }
}

TEST_F(IKPreviewTest, OutputBotConfiguration_RebindResetsSeedAndRevalidatesAgainstNewOutput) {
  auto bot = Robot();
  auto spec = Config();
  spec.outputBotConfiguration = {-0.3_r, 0.2_r};
  spec.goals.clear();
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic));
  bot.worldFromRoot = TransformRT{Quaternion::RotationX(-0.4_r), Real3{-2_r, 1_r, 0_r}};
  bot.defaultPose = {0.25_r, 0.05_r};
  RebuildBotData(bot, ExpectOK{});
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  EXPECT_FALSE(session.HasSolver());
  EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
  ASSERT_TRUE(session.SetOutputBotConfiguration(spec.outputBotConfiguration, diagnostic));
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, -0.3_r, 0.2_r));
  bot.joints[3].type = ArticulatedJointType::Hard;
  RebuildBotData(bot, ExpectOK{});
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  EXPECT_FALSE(session.CanSolve(spec, diagnostic));
  EXPECT_FALSE(session.SetOutputBotConfiguration(spec.outputBotConfiguration, diagnostic));
  EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
  EXPECT_FALSE(session.HasSolver());
}

TEST_F(IKPreviewTest, Couplings_ReachableRotationUsesBothOutputJointsAndSignedRatio) {
  auto const bot = CoupledRobot();
  auto const authored = SReflect::ToJsonString(bot, false);
  for (real const ratio : {0.5_r, -0.5_r}) {
    SCOPED_TRACE(ratio);
    auto spec = Config(0.6_r);
    spec.goals[0].positionStiffness = 0_r;
    spec.createParams.solverParams.maxIter = 200;
    spec.jointCouplings = {Coupling(ratio)};
    auto const document = SerializeStudioIKConfig(spec);
    IKPreviewSession session;
    std::string diagnostic;
    ASSERT_TRUE(session.Prepare(bot, diagnostic));
    ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
    // Two revolute joints, one orientation goal: the reachable zero-residual solution has
    // qA+qB=0.6 and qB=ratio*qA. This is not a hard-guarantee test for conflicting goals.
    auto const q = session.GetOutputBotJointPositions();
    EXPECT_NEAR(q[0], 0.6_r / (1_r + ratio), 2_r * kTolerance);
    EXPECT_NEAR(q[1], ratio * q[0], 2_r * kTolerance);
    EXPECT_GT(Abs(q[0] - bot.defaultPose[0]), 0.05_r);
    EXPECT_GT(Abs(q[1] - bot.defaultPose[1]), 0.05_r);
    EXPECT_EQ(SerializeStudioIKConfig(spec), document);
    EXPECT_EQ(SReflect::ToJsonString(session.GetOutputBot(), false), authored);
    session.Stop();
    EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
  }
}

TEST_F(IKPreviewTest, Couplings_AllCreationFieldsRebuildWhileOrdinaryTicksStayWarm) {
  auto bot = CoupledRobot();
  for (auto& link : bot.links) {
    link.shapeFile = "synthetic";
  }
  CountingLoader loader;
  auto spec = Config(0.6_r);
  spec.goals[0].positionStiffness = 0_r;
  spec.createParams.excludeCollisionShapes = false;
  spec.jointCouplings = {Coupling()};
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  auto const shapes = _context->GetNumShapes();
  ASSERT_TRUE(session.Step(_context, loader, spec, diagnostic)) << diagnostic;
  EXPECT_EQ(loader.loads, isize(bot.links));
  auto const solved = session.GetOutputBotJointPositions();
  spec.goals[0].enabled = false;
  spec.createParams.solverParams.maxIter = 1;
  ASSERT_TRUE(session.Step(_context, loader, spec, diagnostic)) << diagnostic;
  EXPECT_EQ(loader.loads, isize(bot.links));
  EXPECT_EQ(session.GetOutputBotJointPositions(), solved);
  int generations = 1;
  auto recreate = [&] {
    ASSERT_TRUE(session.Step(_context, loader, spec, diagnostic)) << diagnostic;
    EXPECT_EQ(loader.loads, ++generations * isize(bot.links));
    EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
    EXPECT_EQ(session.GetStatus(), "No Active Goals");
    ASSERT_TRUE(session.Step(_context, loader, spec, diagnostic)) << diagnostic;
    EXPECT_EQ(loader.loads, generations * isize(bot.links));
  };
  spec.jointCouplings[0].ratio = -0.5_r;
  recreate();
  spec.jointCouplings[0].stiffness = 20_r;
  recreate();
  spec.jointCouplings[0].enabled = false;
  recreate();
  std::swap(spec.jointCouplings[0].jointA, spec.jointCouplings[0].jointB);
  recreate();
  spec.jointCouplings[0].enabled = true;
  spec.jointCouplings[0].stiffness = 0_r;
  recreate();
  spec.jointCouplings.clear();
  recreate();
  spec.jointCouplings = {Coupling()};
  recreate();
  session.Stop();
  EXPECT_EQ(_context->GetNumShapes(), shapes);
}

TEST_F(IKPreviewTest, Couplings_LimitedIterationsKeepWarmStartWhenDefinitionsUnchanged) {
  auto const bot = CoupledRobot();
  auto spec = Config();
  spec.jointCouplings = {Coupling()};
  spec.createParams.solverParams.maxIter = 1;
  DynamicArray<real> desired{0.8_r, 0.4_r};
  DynamicArray<TransformRT> desiredLinks;
  ComputeLinkTransformsFromPose(
      bot, desired, desiredLinks, LinkTransformSpace::RootFromParent, ExpectOK{});
  SetIKGoalConstraintTransform(spec.goals[0], desiredLinks[kTip]);
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  auto const first = session.GetOutputBotJointPositions();
  auto const goal = bot.worldFromRoot * IKGoalConstraintTransform(spec.goals[0]);
  auto const error =
      Norm(session.GetOutputBotWorldFromLinks()[kTip].GetTranslation() - goal.GetTranslation());
  ASSERT_GT(error, kTolerance);
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  EXPECT_NE(session.GetOutputBotJointPositions(), first);
  EXPECT_LT(
      Norm(session.GetOutputBotWorldFromLinks()[kTip].GetTranslation() - goal.GetTranslation()),
      error);
}

TEST_F(IKPreviewTest, Couplings_InvalidActiveBindingsPreserveSolverAndDisplay) {
  auto const bot = CoupledRobot();
  auto spec = Config(0.3_r);
  spec.goals[0].positionStiffness = 0_r;
  spec.jointCouplings = {Coupling()};
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  auto const positions = session.GetOutputBotJointPositions();
  auto const links = session.GetOutputBotWorldFromLinks();
  for (int fault = 0; fault < 5; ++fault) {
    SCOPED_TRACE(fault);
    auto bad = spec;
    if (fault == 0) {
      bad.jointCouplings[0].jointA = "missing";
    } else if (fault == 1) {
      bad.jointCouplings[0].jointA = "root_joint";
    } else if (fault == 2) {
      bad.jointCouplings[0].jointA = "weld_joint";
    } else if (fault == 3) {
      bad.jointCouplings[0].ratio = 0_r;
    } else {
      bad.jointCouplings.push_back(Coupling());
      std::swap(bad.jointCouplings[1].jointA, bad.jointCouplings[1].jointB);
    }
    EXPECT_FALSE(session.CanSolve(bad, diagnostic));
    EXPECT_FALSE(diagnostic.empty());
    EXPECT_FALSE(session.Step(_context, _loader, bad, diagnostic));
    EXPECT_TRUE(session.HasSolver());
    EXPECT_EQ(session.GetOutputBotJointPositions(), positions);
    EXPECT_EQ(session.GetOutputBotWorldFromLinks(), links);
  }
  IKPreviewSession mixed;
  ASSERT_TRUE(mixed.Prepare(Robot(), diagnostic));
  EXPECT_FALSE(
      mixed.CanSolve(spec, diagnostic)); // Output revolute/prismatic is not document-authorable.
}

TEST_F(IKPreviewTest, Couplings_InactiveStaleNamesAndNoGoalsPreserveBaselineMode) {
  auto const bot = CoupledRobot();
  auto spec = Config();
  spec.goals.clear();
  auto coupling = Coupling();
  coupling.jointA = "missing_a";
  coupling.jointB = "missing_b";
  coupling.enabled = false;
  spec.jointCouplings = {coupling};
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.CanSolve(spec, diagnostic)) << diagnostic;
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  EXPECT_EQ(session.GetStatus(), "No Active Goals");
  EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
  spec.jointCouplings[0].enabled = true;
  spec.jointCouplings[0].stiffness = 0_r;
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
  spec.jointCouplings[0].stiffness = 10_r;
  EXPECT_FALSE(session.CanSolve(spec, diagnostic));
  spec.jointCouplings = {Coupling()};
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  EXPECT_EQ(
      session.GetStatus(),
      "No Active Goals"); // Couplings do not introduce a standalone solve mode.
  EXPECT_EQ(session.GetLastSolveMilliseconds(), 0);
  EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
}

TEST_F(IKPreviewTest, Couplings_OutputOnlyWithManualAndRecordingInputBotOverrides) {
  auto const bot = CoupledRobot();
  auto const inputBot =
      Robot(); // Mixed joint kinds on the input bot must not affect output binding.
  auto spec = Config();
  spec.goals[0].positionStiffness = 0_r;
  spec.inputBot = "reference.superdex_bot";
  spec.inputBotConfiguration = {0.3_r, 0.2_r};
  spec.goals[0].inputBotLink = "d_tip";
  SetIKGoalConstraintTransform(spec.goals[0], TransformRT{});
  spec.jointCouplings = {Coupling()};
  auto const document = SerializeStudioIKConfig(spec);
  auto const inputBotSnapshot = SReflect::ToJsonString(inputBot, false);
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.SetInputBot(&inputBot, diagnostic));
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  EXPECT_NEAR(session.GetOutputBotJointPositions()[0], 0.2_r, 2_r * kTolerance);
  EXPECT_NEAR(session.GetOutputBotJointPositions()[1], 0.1_r, 2_r * kTolerance);
  // This borrowed transient pose is the same input path the RRD player supplies; no recording
  // goal coordinates or document seed are rewritten by output coupling solves.
  DynamicArray<real> inputBotPositions{0.6_r, 0.4_r};
  IKInputBotPose const pose{inputBotPositions, bot.worldFromRoot};
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic, &pose)) << diagnostic;
  EXPECT_NEAR(session.GetOutputBotJointPositions()[0], 0.4_r, 2_r * kTolerance);
  EXPECT_NEAR(session.GetOutputBotJointPositions()[1], 0.2_r, 2_r * kTolerance);
  EXPECT_EQ(SerializeStudioIKConfig(spec), document);
  EXPECT_EQ(session.GetOutputBot().defaultPose, bot.defaultPose);
  EXPECT_EQ(SReflect::ToJsonString(*session.GetInputBot(), false), inputBotSnapshot);
  session.Stop();
  EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
}

TEST_F(IKPreviewTest, Couplings_SerializedUndoRedoRecreatesOnlySolveStateAndRetainsAuthoring) {
  auto bot = CoupledRobot();
  for (auto& link : bot.links) {
    link.shapeFile = "synthetic";
  }
  auto spec = Config(0.3_r);
  spec.outputBotConfiguration = {0.05_r, 0.15_r};
  spec.createParams.excludeCollisionShapes = false;
  spec.goals[0].positionStiffness = 0_r;
  spec.inputBot = "reference.superdex_bot";
  spec.inputBotConfiguration = {0.2_r, 0.3_r};
  spec.jointCouplings = {Coupling()};
  auto const undo = SerializeStudioIKConfig(spec);
  auto changed = spec;
  changed.jointCouplings[0].ratio = -0.5_r;
  auto const redo = SerializeStudioIKConfig(changed);
  CountingLoader loader;
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.SetInputBot(&bot, diagnostic));
  ASSERT_TRUE(session.Step(_context, loader, spec, diagnostic)) << diagnostic;
  EXPECT_NE(session.GetOutputBotJointPositions(), spec.outputBotConfiguration);
  int generations = 1;
  for (auto const& json : {redo, undo, redo}) {
    BotIKConfig restored;
    ASSERT_TRUE(ParseIKConfig(json, restored, diagnostic)) << diagnostic;
    EXPECT_EQ(restored.outputBotConfiguration, spec.outputBotConfiguration);
    EXPECT_EQ(restored.inputBot, spec.inputBot);
    EXPECT_EQ(restored.inputBotConfiguration, spec.inputBotConfiguration);
    // Serialized undo restoration changes creation settings even for a direct Step caller.
    restored.goals[0].enabled = false;
    ASSERT_TRUE(session.Step(_context, loader, restored, diagnostic)) << diagnostic;
    EXPECT_EQ(loader.loads, ++generations * isize(bot.links));
    EXPECT_EQ(session.GetOutputBotJointPositions(), spec.outputBotConfiguration);
    EXPECT_EQ(session.GetInputBot()->defaultPose, bot.defaultPose);
  }
  session.Stop();
  EXPECT_EQ(session.GetOutputBotJointPositions(), spec.outputBotConfiguration);
}

// Opt-in measurement: no checkout paths or large asset runtime dependencies in ordinary CI.
// Reports solver-only timing, not frame/gizmo latency. Budget checks are not hard preemption.
TEST_F(IKPreviewTest, DISABLED_RealAssets_TenMillisecondBudgetSolverTimings) {
  char const* const assets = std::getenv("SUPERDEX_IK_TIMING_ASSETS");
  if (assets == nullptr || *assets == '\0') {
    GTEST_SKIP() << "Set SUPERDEX_IK_TIMING_ASSETS to the public SuperDex assets directory";
  }
  std::string diagnostic;
  struct Case {
    char const* relativePath;
    int expectedDofs;
    int expectedLeaves;
  };
  for (auto const& entry :
       {Case{"bots/arms/fr3/fr3.superdex_bot", 7, 1},
        Case{"bots/hands/allegro_v5/right/allegro_v5_right.superdex_bot", 16, 4}}) {
    auto const path = (std::filesystem::path(assets) / entry.relativePath).generic_string();
    SCOPED_TRACE(path);
    Error error;
    auto const bot = LoadBotPrefab(path, _loader, true, error);
    ASSERT_TRUE(error.IsOK()) << error.GetDescription();
    ASSERT_EQ(bot._numDofs, entry.expectedDofs);
    auto const leaves = FindLeafLinkIndices(bot);
    ASSERT_EQ(isize(leaves), entry.expectedLeaves);
    BotIKConfig spec;
    spec.outputBot = path;
    ASSERT_EQ(spec.createParams.solverParams.maxElapsedTimeSeconds, 0.01);
    ASSERT_TRUE(InitializeIKGoals(bot, spec, diagnostic)) << diagnostic;
    IKPreviewSession session;
    ASSERT_TRUE(session.Prepare(bot, diagnostic)) << diagnostic;
    DynamicArray<double> elapsed;
    int reached = 0;
    int overBudget = 0;
    constexpr int kWarmup = 10;
    constexpr int kSamples = 120;
    double coldMilliseconds = 0;
    for (int tick = 0; tick < kWarmup + kSamples; ++tick) {
      // Nearby FK-generated goals on every tick; the solver retains authored soft joint limits.
      auto desiredPose = bot.defaultPose;
      real const offset = 0.12_r * static_cast<real>(std::sin(0.35 * tick));
      for (int i = 0; i < isize(desiredPose); ++i) {
        desiredPose[i] += (i % 2 == 0 ? offset : -offset);
      }
      DynamicArray<TransformRT> desiredLinks;
      ComputeLinkTransformsFromPose(
          bot, desiredPose, desiredLinks, LinkTransformSpace::RootFromParent, error);
      ASSERT_TRUE(error.IsOK()) << error.GetDescription();
      for (int i = 0; i < isize(leaves); ++i) {
        SetIKGoalConstraintTransform(spec.goals[i], desiredLinks[leaves[i]]);
      }
      ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
      double const milliseconds = session.GetLastSolveMilliseconds();
      EXPECT_TRUE(std::isfinite(milliseconds));
      EXPECT_GE(milliseconds, 0);
      for (auto const& link : session.GetOutputBotWorldFromLinks()) {
        EXPECT_TRUE(IsFinite(link));
      }
      if (tick == 0) {
        coldMilliseconds = milliseconds;
      }
      if (tick >= kWarmup) {
        elapsed.push_back(milliseconds);
        reached += session.GetStatus() == "Goals Reached";
        overBudget += milliseconds > 10.0;
      }
    }
    std::sort(elapsed.begin(), elapsed.end());
    double const mean = std::accumulate(elapsed.begin(), elapsed.end(), 0.0) / elapsed.size();
    std::printf(
        "IK_TIMING asset=%s links=%zu dofs=%d goals=%zu samples=%d "
        "budget_ms=10 cold_solve_ms=%.3f mean_ms=%.3f median_ms=%.3f "
        "p95_ms=%.3f max_ms=%.3f reached=%d over_budget=%d\n",
        path.c_str(),
        bot.links.size(),
        bot._numDofs,
        spec.goals.size(),
        kSamples,
        coldMilliseconds,
        mean,
        elapsed[elapsed.size() / 2],
        elapsed[(elapsed.size() * 95 + 99) / 100 - 1],
        elapsed.back(),
        reached,
        overBudget);
  }
}

TEST_F(IKPreviewTest, InputBot_RootPlacementJointPoseAndOffsetsDrivePreview) {
  auto const bot = Robot();
  auto inputBot = Robot(ArticulatedJointType::Free);
  inputBot.links[kTip].name = "reference_tip";
  RebuildBotData(inputBot, ExpectOK{});
  auto spec = Config();
  spec.inputBot = "reference.superdex_bot";
  spec.inputBotConfiguration = {0.35_r, 0.2_r};
  spec.outputBaseFrameFromInputBaseFrame = TransformRT{Real3{0_r, 0_r, 0.1_r}};
  spec.goals[0].inputBotLink = "reference_tip";
  spec.goals[0].baseOrInputLinkFrameFromConstraintFrame = TransformRT{Real3{0_r, 0_r, -0.1_r}};
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  EXPECT_FALSE(session.CanSolve(spec, diagnostic));
  ASSERT_TRUE(session.SetInputBot(&inputBot, diagnostic));
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, 0.35_r, 0.2_r));
  spec.inputBotConfiguration = {0.5_r, 0.3_r};
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, 0.5_r, 0.3_r));
  auto const previous = session.GetOutputBotWorldFromLinks();
  spec.goals[0].inputBotLink = "missing";
  EXPECT_FALSE(session.Step(_context, _loader, spec, diagnostic));
  EXPECT_EQ(session.GetOutputBotWorldFromLinks(), previous);
  session.Stop();
  EXPECT_EQ(spec.inputBotConfiguration[0], 0.5_r);
}

TEST_F(IKPreviewTest, InputBot_UniformScaleIncludesOffsetsAndUpdatesLiveWithoutMutatingBots) {
  auto const bot = Robot();
  auto const inputBot = Robot(ArticulatedJointType::Free);
  auto spec = Config();
  spec.inputBot = "reference.superdex_bot";
  spec.inputBotConfiguration = {0.35_r, 0.2_r};
  auto& goal = spec.goals[0];
  goal.inputBotLink = "d_tip";
  goal.baseOrInputLinkFrameFromConstraintFrame =
      TransformRT{Quaternion::RotationX(0.2_r), Real3{0.1_r, 0_r, 0_r}};
  goal.linkFrameFromConstraintFrame = goal.baseOrInputLinkFrameFromConstraintFrame;
  auto const outputBotSnapshot = SReflect::ToJsonString(bot, false);
  auto const inputBotSnapshot = SReflect::ToJsonString(inputBot, false);
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.SetInputBot(&inputBot, diagnostic));
  for (real const scale : {0.5_r, 1_r, 2_r}) {
    SCOPED_TRACE(scale);
    spec.inputBotScale = scale;
    // Keep the arm joint origin fixed; only the rotated arm/slide/offset distances scale.
    spec.outputBaseFrameFromInputBaseFrame = TransformRT{Real3{0.3_r * (1_r - scale), 0_r, 0_r}};
    ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic)) << diagnostic;
    EXPECT_EQ(session.GetStatus(), "Goals Reached");
    auto const expected = Expected(bot.worldFromRoot, 0.35_r, scale - 0.8_r);
    ExpectLinks(session.GetOutputBotWorldFromLinks(), expected);
    EXPECT_TRUE(NearEqual(
        session.GetOutputBotWorldFromLinks()[kTip] * goal.linkFrameFromConstraintFrame,
        expected[kTip] * goal.linkFrameFromConstraintFrame,
        kTolerance));
  }
  auto const previous = session.GetOutputBotWorldFromLinks();
  spec.inputBotScale = 0_r;
  EXPECT_FALSE(session.CanSolve(spec, diagnostic));
  EXPECT_FALSE(session.Step(_context, _loader, spec, diagnostic));
  EXPECT_EQ(session.GetOutputBotWorldFromLinks(), previous);
  EXPECT_EQ(SReflect::ToJsonString(session.GetOutputBot(), false), outputBotSnapshot);
  EXPECT_EQ(SReflect::ToJsonString(*session.GetInputBot(), false), inputBotSnapshot);
  session.Stop();
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, 0.15_r, 0.1_r));
}

TEST_F(IKPreviewTest, InputBotOverride_DifferentInputBotDOFsDriveLiveIKWithoutChangingOutputSeed) {
  auto bot = Robot();
  for (auto& link : bot.links) {
    link.shapeFile = "synthetic";
  }
  auto inputBot = Robot(ArticulatedJointType::Free);
  inputBot.joints[kTip].type = ArticulatedJointType::Revolute;
  inputBot.defaultPose.push_back(0.2_r);
  RebuildBotData(inputBot, ExpectOK{});
  ASSERT_NE(bot._numDofs, inputBot._numDofs);
  auto spec = Config();
  spec.inputBot = "reference.superdex_bot";
  spec.inputBotConfiguration = {-0.1_r, 0.4_r, 0.3_r};
  spec.createParams.excludeCollisionShapes = false;
  spec.goals[0].outputBotLink = "b_arm";
  spec.goals[0].inputBotLink = "b_arm";
  SetIKGoalConstraintTransform(spec.goals[0], TransformRT{});
  auto const document = SerializeStudioIKConfig(spec);
  CountingLoader loader;
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.SetInputBot(&inputBot, diagnostic));
  for (real const angle : {0.4_r, -0.2_r, 0.3_r}) {
    DynamicArray<real> const inputBotPositions{angle, 0.8_r, -0.5_r};
    IKInputBotPose const pose{inputBotPositions, bot.worldFromRoot};
    ASSERT_TRUE(session.CanSolve(spec, diagnostic, &pose)) << diagnostic;
    ASSERT_TRUE(session.Step(_context, loader, spec, diagnostic, &pose)) << diagnostic;
    EXPECT_EQ(loader.loads, isize(bot.links));
    EXPECT_EQ(isize(session.GetOutputBotJointPositions()), bot._numDofs);
    EXPECT_NEAR(session.GetOutputBotJointPositions()[0], angle, kTolerance);
    EXPECT_NEAR(session.GetOutputBotJointPositions()[1], bot.defaultPose[1], kTolerance);
    EXPECT_EQ(session.GetOutputBot().defaultPose, bot.defaultPose);
    EXPECT_EQ(session.GetInputBot()->defaultPose, inputBot.defaultPose);
    EXPECT_EQ(SerializeStudioIKConfig(spec), document);
  }
  session.Stop();
  EXPECT_EQ(session.GetOutputBotJointPositions(), bot.defaultPose);
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, 0.15_r, 0.1_r));
}

TEST_F(
    IKPreviewTest,
    InputBotOverride_FloatingRootKeepsAuthoredAnchorAndUnscaledWristDisplacement) {
  auto const bot = Robot(ArticulatedJointType::Free);
  auto inputBot = Robot(ArticulatedJointType::Free);
  inputBot.scale = 2_r;
  auto spec = Config();
  spec.createParams.fixedBase = false;
  spec.outputBotConfiguration = {-0.25_r, 0.2_r};
  spec.inputBot = "reference.superdex_bot";
  spec.inputBotScale = 3_r;
  spec.goals[0].outputBotLink = "root";
  spec.goals[0].inputBotLink = "root";
  SetIKGoalConstraintTransform(spec.goals[0], TransformRT{});
  auto const document = SerializeStudioIKConfig(spec);
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.SetInputBot(&inputBot, diagnostic));
  DynamicArray<real> const inputBotPositions{0.8_r, 0.4_r};
  for (real const displacement : {0.2_r, 0.4_r, 0.2_r}) {
    TransformRT const root = bot.worldFromRoot * TransformRT{Real3{displacement, 0_r, 0_r}};
    IKInputBotPose const pose{inputBotPositions, root};
    ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic, &pose)) << diagnostic;
    EXPECT_TRUE(NearEqual(session.GetOutputBotWorldFromLinks()[0], root, kTolerance));
    EXPECT_EQ(session.GetWorldFrameFromOutputBaseFrame(), bot.worldFromRoot);
    EXPECT_EQ(session.GetOutputBotJointPositions(), spec.outputBotConfiguration);
    EXPECT_EQ(SerializeStudioIKConfig(spec), document);
  }
  spec.createParams.fixedBase = true;
  IKInputBotPose const pose{
      inputBotPositions, bot.worldFromRoot * TransformRT{Real3{0.2_r, 0_r, 0_r}}};
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic, &pose)) << diagnostic;
  EXPECT_TRUE(NearEqual(session.GetOutputBotWorldFromLinks()[0], bot.worldFromRoot, kTolerance));
  session.Stop();
  EXPECT_EQ(session.GetOutputBotJointPositions(), spec.outputBotConfiguration);
  ExpectLinks(session.GetOutputBotWorldFromLinks(), Expected(bot.worldFromRoot, -0.25_r, 0.2_r));
}

TEST_F(IKPreviewTest, InputBotOverride_InvalidInputsPreserveSolverDisplayAndDocument) {
  auto const bot = Robot();
  auto spec = Config();
  spec.inputBot = "reference.superdex_bot";
  spec.goals[0].inputBotLink = "d_tip";
  SetIKGoalConstraintTransform(spec.goals[0], TransformRT{});
  auto const document = SerializeStudioIKConfig(spec);
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(bot, diagnostic));
  ASSERT_TRUE(session.SetInputBot(&bot, diagnostic));
  DynamicArray<real> const validPositions{0.35_r, 0.2_r};
  IKInputBotPose const valid{validPositions, bot.worldFromRoot};
  ASSERT_TRUE(session.Step(_context, _loader, spec, diagnostic, &valid)) << diagnostic;
  auto const display = session.GetOutputBotWorldFromLinks();
  auto const positions = session.GetOutputBotJointPositions();
  for (int fault = 0; fault < 4; ++fault) {
    SCOPED_TRACE(fault);
    auto badPositions = validPositions;
    auto badRoot = bot.worldFromRoot;
    auto badConfig = spec;
    if (fault == 0) {
      badPositions.pop_back();
    } else if (fault == 1) {
      badPositions[0] = std::numeric_limits<real>::quiet_NaN();
    } else if (fault == 2) {
      badRoot.SetTranslation(Real3{std::numeric_limits<real>::quiet_NaN(), 0_r, 0_r});
    } else {
      badConfig.inputBot.clear();
    }
    IKInputBotPose const bad{badPositions, badRoot};
    EXPECT_FALSE(session.CanSolve(badConfig, diagnostic, &bad));
    EXPECT_FALSE(session.Step(_context, _loader, badConfig, diagnostic, &bad));
    EXPECT_FALSE(diagnostic.empty());
    EXPECT_TRUE(session.HasSolver());
    EXPECT_EQ(session.GetOutputBotWorldFromLinks(), display);
    EXPECT_EQ(session.GetOutputBotJointPositions(), positions);
    EXPECT_EQ(SerializeStudioIKConfig(spec), document);
  }
}

BotIKConfig SnapshotConfig() {
  auto config = Config();
  config.inputBot = "input.superdex_bot";
  config.outputBotConfiguration = {-0.2_r, 0.35_r};
  config.inputBotConfiguration = {0.1_r, 0.2_r};
  config.inputBotScale = 3_r;
  config.outputBaseFrameFromInputBaseFrame =
      TransformRT{Quaternion::RotationX(0.15_r), Real3{0.1_r, -0.2_r, 0.3_r}};
  config.goals[0].inputBotLink = "d_tip";
  config.goals[0].linkFrameFromConstraintFrame =
      TransformRT{Quaternion::RotationZ(-0.1_r), Real3{0.03_r, 0.04_r, 0.05_r}};
  auto coupling = Coupling(-0.5_r);
  coupling.enabled = false;
  config.jointCouplings = {coupling};
  return config;
}

TEST(IKInputPoseSnapshotTest, CopiesOnlyInputQAndRootWithoutScalingOrChangingModels) {
  auto inputBot = Robot();
  inputBot.scale = 2_r;
  auto const model = SReflect::ToJsonString(inputBot, false);
  auto const config = SnapshotConfig();
  auto const original = SerializeStudioIKConfig(config);
  TransformRT const outputAnchor{Quaternion::RotationY(0.3_r), Real3{1_r, 2_r, -3_r}};
  TransformRT const relative{Quaternion::RotationZ(0.4_r), Real3{0.5_r, -0.3_r, 0.2_r}};
  IKInputPoseSnapshot displayed{{0.45_r, 0.25_r}, outputAnchor * relative};
  BotIKConfig candidate;
  std::string diagnostic;
  ASSERT_TRUE(
      BuildIKInputPoseSnapshot(config, &inputBot, displayed, outputAnchor, candidate, diagnostic))
      << diagnostic;
  EXPECT_EQ(candidate.inputBotConfiguration, displayed.inputBotConfiguration);
  EXPECT_TRUE(NearEqual(candidate.outputBaseFrameFromInputBaseFrame, relative, 1e-5_r));
  EXPECT_EQ(candidate.inputBotScale, 3_r);
  EXPECT_EQ(SReflect::ToJsonString(inputBot, false), model);
  // Restoring the two permitted fields must restore the entire authored document.
  candidate.inputBotConfiguration = config.inputBotConfiguration;
  candidate.outputBaseFrameFromInputBaseFrame = config.outputBaseFrameFromInputBaseFrame;
  EXPECT_EQ(SerializeStudioIKConfig(candidate), original);
}

TEST(IKInputPoseSnapshotTest, MaterializesDisplayedDefaultAndRepeatedCaptureIsAnAuthoredNoop) {
  auto const inputBot = Robot();
  auto config = SnapshotConfig();
  config.inputBotConfiguration.clear();
  auto const outputAnchor = Robot().worldFromRoot;
  IKInputPoseSnapshot displayed{
      inputBot.defaultPose, outputAnchor * config.outputBaseFrameFromInputBaseFrame};
  BotIKConfig candidate;
  std::string diagnostic;
  ASSERT_TRUE(
      BuildIKInputPoseSnapshot(config, &inputBot, displayed, outputAnchor, candidate, diagnostic))
      << diagnostic;
  EXPECT_EQ(candidate.inputBotConfiguration, inputBot.defaultPose);
  EXPECT_EQ(candidate.outputBaseFrameFromInputBaseFrame, config.outputBaseFrameFromInputBaseFrame);
  auto const saved = SerializeStudioIKConfig(candidate);
  displayed.worldFrameFromInputBaseFrame =
      outputAnchor * candidate.outputBaseFrameFromInputBaseFrame;
  BotIKConfig repeated;
  ASSERT_TRUE(
      BuildIKInputPoseSnapshot(candidate, &inputBot, displayed, outputAnchor, repeated, diagnostic))
      << diagnostic;
  EXPECT_EQ(SerializeStudioIKConfig(repeated), saved);
}

TEST(IKInputPoseSnapshotTest, CandidateOwnsQAfterBorrowedRecordingSourceChanges) {
  auto const inputBot = Robot();
  auto const config = SnapshotConfig();
  auto const outputAnchor = Robot().worldFromRoot;
  IKInputPoseSnapshot displayed{{0.45_r, 0.25_r}, outputAnchor};
  BotIKConfig candidate;
  std::string diagnostic;
  ASSERT_TRUE(
      BuildIKInputPoseSnapshot(config, &inputBot, displayed, outputAnchor, candidate, diagnostic));
  displayed.inputBotConfiguration[0] = -0.8_r;
  displayed.inputBotConfiguration.clear();
  EXPECT_EQ(candidate.inputBotConfiguration, (DynamicArray<real>{0.45_r, 0.25_r}));
}

TEST(IKInputPoseSnapshotTest, InvalidOrIncompletePosePreservesCandidateAndDocument) {
  auto const inputBot = Robot();
  auto const config = SnapshotConfig();
  auto const original = SerializeStudioIKConfig(config);
  auto const outputAnchor = Robot().worldFromRoot;
  for (int fault = 0; fault < 12; ++fault) {
    SCOPED_TRACE(fault);
    auto badConfig = config;
    auto anchor = outputAnchor;
    IKInputPoseSnapshot displayed{{0.45_r, 0.25_r}, outputAnchor};
    auto const* model = &inputBot;
    if (fault == 0) {
      displayed.inputBotConfiguration.clear();
    }
    if (fault == 1) {
      displayed.inputBotConfiguration.push_back(0_r);
    }
    if (fault == 2) {
      displayed.inputBotConfiguration[0] = std::numeric_limits<real>::quiet_NaN();
    }
    if (fault == 3) {
      displayed.worldFrameFromInputBaseFrame.SetTranslation(
          Real3{std::numeric_limits<real>::infinity(), 0_r, 0_r});
    }
    if (fault == 4) {
      displayed.worldFrameFromInputBaseFrame.SetRotation(Quaternion{0_r, 0_r, 0_r, 2_r});
    }
    if (fault == 5) {
      anchor.SetRotation(Quaternion{0_r, 0_r, 0_r, 0_r});
    }
    if (fault == 6) {
      anchor.SetTranslation(Real3{std::numeric_limits<real>::quiet_NaN(), 0_r, 0_r});
    }
    if (fault == 7) {
      badConfig.inputBotScale = 0_r;
    }
    if (fault == 8) {
      badConfig.goals[0].positionStiffness = -1_r;
    }
    if (fault == 9) {
      badConfig.inputBot.clear();
    }
    if (fault == 10) {
      model = nullptr;
    }
    if (fault == 11) {
      badConfig.version = 8;
    }
    auto candidate = Config(-0.6_r, 0.1_r);
    auto const sentinel = SerializeStudioIKConfig(candidate);
    std::string diagnostic;
    EXPECT_FALSE(
        BuildIKInputPoseSnapshot(badConfig, model, displayed, anchor, candidate, diagnostic));
    EXPECT_FALSE(diagnostic.empty());
    EXPECT_EQ(SerializeStudioIKConfig(candidate), sentinel);
    EXPECT_EQ(SerializeStudioIKConfig(config), original);
  }
}

TEST(IKInputPoseSnapshotTest, RejectsStalePreparedMetadataBeforeNativeFK) {
  auto const config = SnapshotConfig();
  auto const anchor = Robot().worldFromRoot;
  IKInputPoseSnapshot displayed{{0.45_r, 0.25_r}, anchor};
  for (int fault = 0; fault < 7; ++fault) {
    SCOPED_TRACE(fault);
    auto inputBot = Robot();
    if (fault == 0) {
      inputBot.links[kTip].parentLink = kTip;
    }
    if (fault == 1) {
      inputBot.links[kTip]._index = 99;
    }
    if (fault == 2) {
      inputBot.links[kTip]._parentFromLink.SetTranslation(Real3{9_r, 0_r, 0_r});
    }
    if (fault == 3) {
      inputBot.links[kTip]._rootFromLink.SetTranslation(Real3{9_r, 0_r, 0_r});
    }
    if (fault == 4) {
      std::swap(inputBot._dofIndices[0], inputBot._dofIndices[1]);
    }
    if (fault == 5) {
      inputBot.links[0]._childrenIndices.clear();
    }
    if (fault == 6) {
      inputBot.defaultPose[0] = std::numeric_limits<real>::quiet_NaN();
    }
    auto candidate = Config();
    auto const sentinel = SerializeStudioIKConfig(candidate);
    std::string diagnostic;
    EXPECT_FALSE(
        BuildIKInputPoseSnapshot(config, &inputBot, displayed, anchor, candidate, diagnostic));
    EXPECT_FALSE(diagnostic.empty());
    EXPECT_EQ(SerializeStudioIKConfig(candidate), sentinel);
  }
}

TEST(IKInputPoseSnapshotTest, OverflowingNativeFKFailsBeforeCandidateMutation) {
  auto inputBot = Robot();
  inputBot.scale = 10_r;
  auto const config = SnapshotConfig();
  IKInputPoseSnapshot displayed{
      {0_r, std::numeric_limits<real>::max() * 0.75_r}, inputBot.worldFromRoot};
  auto candidate = Config();
  auto const sentinel = SerializeStudioIKConfig(candidate);
  std::string diagnostic;
  EXPECT_FALSE(BuildIKInputPoseSnapshot(
      config, &inputBot, displayed, inputBot.worldFromRoot, candidate, diagnostic));
  EXPECT_EQ(SerializeStudioIKConfig(candidate), sentinel);
}

TEST_F(IKPreviewTest, InputSnapshotUsesAuthoredAnchorAndDoesNotResetSolvedOutput) {
  auto const outputBot = Robot(ArticulatedJointType::Free);
  auto const inputBot = Robot(ArticulatedJointType::Free);
  auto config = SnapshotConfig();
  config.createParams.fixedBase = false;
  config.goals[0].outputBotLink = "root";
  config.goals[0].inputBotLink = "root";
  config.goals[0].linkFrameFromConstraintFrame = {};
  config.goals[0].baseOrInputLinkFrameFromConstraintFrame = {};
  TransformRT const relative{Quaternion::RotationZ(0.4_r), Real3{0.3_r, 0.4_r, 0.5_r}};
  IKInputPoseSnapshot displayed{{0.45_r, 0.25_r}, outputBot.worldFromRoot * relative};
  IKInputBotPose const borrowed{
      displayed.inputBotConfiguration, displayed.worldFrameFromInputBaseFrame};
  IKPreviewSession session;
  std::string diagnostic;
  ASSERT_TRUE(session.Prepare(outputBot, diagnostic));
  ASSERT_TRUE(session.SetInputBot(&inputBot, diagnostic));
  ASSERT_TRUE(session.Step(_context, _loader, config, diagnostic, &borrowed)) << diagnostic;
  auto const solvedLinks = session.GetOutputBotWorldFromLinks();
  auto const solvedQ = session.GetOutputBotJointPositions();
  ASSERT_FALSE(NearEqual(solvedLinks.front(), session.GetWorldFrameFromOutputBaseFrame(), 1e-3_r));
  BotIKConfig candidate;
  ASSERT_TRUE(BuildIKInputPoseSnapshot(
      config,
      session.GetInputBot(),
      displayed,
      session.GetWorldFrameFromOutputBaseFrame(),
      candidate,
      diagnostic))
      << diagnostic;
  EXPECT_TRUE(NearEqual(candidate.outputBaseFrameFromInputBaseFrame, relative, 1e-5_r));
  EXPECT_TRUE(session.HasSolver());
  EXPECT_EQ(session.GetOutputBotWorldFromLinks(), solvedLinks);
  EXPECT_EQ(session.GetOutputBotJointPositions(), solvedQ);
  EXPECT_EQ(session.GetWorldFrameFromOutputBaseFrame(), outputBot.worldFromRoot);
}

TEST(IKInputPoseSnapshotTest, InputFieldsRoundtripTogetherWithoutAnyRecordingDependency) {
  auto const inputBot = Robot();
  auto const config = SnapshotConfig();
  auto const outputAnchor = Robot().worldFromRoot;
  IKInputPoseSnapshot displayed{
      {0.45_r, 0.25_r},
      outputAnchor * TransformRT{Quaternion::RotationZ(0.4_r), Real3{0.3_r, 0.4_r, 0.5_r}}};
  BotIKConfig candidate;
  std::string diagnostic;
  ASSERT_TRUE(
      BuildIKInputPoseSnapshot(config, &inputBot, displayed, outputAnchor, candidate, diagnostic));
  auto const undo = SerializeStudioIKConfig(config);
  auto const redo = SerializeStudioIKConfig(candidate);
  EXPECT_EQ(redo.find("Recording"), std::string::npos);
  for (auto const& json : {undo, redo, undo, redo}) {
    BotIKConfig restored;
    ASSERT_TRUE(ParseIKConfig(json, restored, diagnostic)) << diagnostic;
    EXPECT_EQ(SerializeStudioIKConfig(restored), json);
    EXPECT_EQ(restored.outputBotConfiguration, config.outputBotConfiguration);
    EXPECT_EQ(restored.inputBotScale, config.inputBotScale);
  }
}

} // namespace
