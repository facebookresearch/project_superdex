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

#include "editors/key_pose.h"

#include <superdex_robotics/utils/bot_utils.h>

#include <mochi_core/test/mochi_test_helpers.h>

#include <gtest/gtest.h>

#include <numbers>
#include <string_view>
#include <vector>

namespace superdex::studio {
namespace {

using namespace mochi;
using mochi::test::ExpectOK;
using superdex::ArticulatedJointType;
using superdex::robotics::BotPrefab;

superdex::robotics::BotLinkPrefab MakeLink(char const* name, int parent) {
  superdex::robotics::BotLinkPrefab link;
  link.name = name;
  link.parentLink = parent;
  link.mass = 1_r;
  link.centerOfMass = Real3{};
  link.momentOfInertia = Real6{1_r, 0_r, 0_r, 1_r, 0_r, 1_r};
  return link;
}

superdex::robotics::BotJointPrefab
MakeJoint(char const* name, ArticulatedJointType type, Real3 const& translation = {}) {
  superdex::robotics::BotJointPrefab joint;
  joint.name = name;
  joint.type = type;
  joint.axis = Real3{0_r, 0_r, 1_r};
  joint.parentLinkFromJoint = TransformRT{translation};
  return joint;
}

// A chain root -> arm -> hand -> finger of 1 m revolute links about z, a slider on the arm, along
// its x, and a ball on the root. Depth first, the order RebuildBotData keeps links in.
BotPrefab MakeBot() {
  BotPrefab bot;
  bot.name = "key_pose_test";
  bot.links = {
      MakeLink("root", superdex::robotics::kIndexNone),
      MakeLink("arm", 0),
      MakeLink("hand", 1),
      MakeLink("finger", 2),
      MakeLink("slider", 1),
      MakeLink("ball", 0)};
  bot.joints = {
      MakeJoint("root_joint", ArticulatedJointType::Hard),
      MakeJoint("arm_joint", ArticulatedJointType::Revolute),
      MakeJoint("hand_joint", ArticulatedJointType::Revolute, {1_r, 0_r, 0_r}),
      MakeJoint("finger_joint", ArticulatedJointType::Revolute, {1_r, 0_r, 0_r}),
      MakeJoint("slider_joint", ArticulatedJointType::Prismatic),
      MakeJoint("ball_joint", ArticulatedJointType::Spherical, {0_r, 1_r, 0_r})};
  bot.joints[4].axis = Real3{1_r, 0_r, 0_r};
  superdex::robotics::RebuildBotData(bot, ExpectOK{});
  return bot;
}

// Arm, hand, finger, slider, then the ball's three.
std::vector<real> const kNow = {0.1_r, 0.2_r, 0.3_r, 0.05_r, 0_r, 0_r, 0_r};

TEST(KeyPoseTest, KeyedJointsTakeTheKeyAndMoveTheirLinksAndTheLinksBelow) {
  BotPrefab const bot = MakeBot();
  std::vector<std::string_view> const joints = {"hand_joint"};
  std::vector<real> const values = {1_r};

  KeyPose shown;
  ASSERT_TRUE(ComputeKeyPose(
      bot,
      mochi::MakeConstSpan(kNow),
      mochi::MakeConstSpan(joints),
      mochi::MakeConstSpan(values),
      shown));

  KeyPose const expected{
      .pose = {0.1_r, 1_r, 0.3_r, 0.05_r, 0_r, 0_r, 0_r},
      .movedLinks = {false, false, true, true, false, false}};
  EXPECT_EQ(shown, expected);
}

TEST(KeyPoseTest, RefusesJointsItCannotKey) {
  BotPrefab const bot = MakeBot();
  std::vector<real> const one = {1_r};
  auto const computes = [&](std::vector<std::string_view> const& joints,
                            std::vector<real> const& values,
                            std::vector<real> const& now) {
    KeyPose shown;
    return ComputeKeyPose(
        bot,
        mochi::MakeConstSpan(now),
        mochi::MakeConstSpan(joints),
        mochi::MakeConstSpan(values),
        shown);
  };

  EXPECT_FALSE(computes({"no_such_joint"}, one, kNow));
  EXPECT_FALSE(computes({"ball_joint"}, one, kNow)) << "three DOFs";
  EXPECT_FALSE(computes({"root_joint"}, one, kNow)) << "no DOF";
  EXPECT_FALSE(computes({"arm_joint", "hand_joint"}, one, kNow)) << "one value for two joints";
  EXPECT_FALSE(computes({"arm_joint"}, one, {0.1_r, 0.2_r})) << "a pose of the wrong size";
}

TEST(KeyPoseTest, LinkWorldTransformsPlaceTheBotAtTheRootGivenAndItsScale) {
  BotPrefab bot = MakeBot();
  bot.scale = 2_r;
  real const quarterTurn = std::numbers::pi_v<real> / 2;
  std::vector<real> const pose = {quarterTurn, 0_r, 0_r, 0.25_r, 0_r, 0_r, 0_r};

  std::vector<TransformRT> links;
  ComputeLinkWorldTransforms(
      bot, TransformRT{Real3{1_r, 2_r, 3_r}}, mochi::MakeConstSpan(pose), links, ExpectOK{});

  // The arm turns a quarter about z at the root, so the hand, 1 m along it (2 m scaled), is 2 m
  // along y from the root, and the finger 4 m. The slider is 0.25 m (0.5 m) along the arm, and the
  // ball 1 m (2 m) along y.
  Quaternion const turned = Quaternion::RotationZ(quarterTurn);
  std::vector<TransformRT> const expected = {
      TransformRT{Real3{1_r, 2_r, 3_r}},
      TransformRT{turned, Real3{1_r, 2_r, 3_r}},
      TransformRT{turned, Real3{1_r, 4_r, 3_r}},
      TransformRT{turned, Real3{1_r, 6_r, 3_r}},
      TransformRT{turned, Real3{1_r, 2.5_r, 3_r}},
      TransformRT{Real3{1_r, 4_r, 3_r}}};
  ASSERT_EQ(links.size(), expected.size());
  for (size_t i = 0; i < links.size(); ++i) {
    EXPECT_TRUE(NearEqual(links[i], expected[i], 1e-5_r)) << "link " << i;
  }
}

TEST(KeyPoseTest, PoseFromLinkWorldTransformsUndoesTheRootAndScale) {
  BotPrefab bot = MakeBot();
  bot.scale = 2_r;
  TransformRT const worldFromRoot{
      Quaternion::FromRotationVector(Real3{0.3_r, -0.2_r, 0.5_r}), Real3{1_r, 2_r, 3_r}};
  std::vector<real> const pose = {0.4_r, -0.3_r, 0.2_r, 0.15_r, 0.1_r, -0.2_r, 0.3_r};
  std::vector<TransformRT> links;
  ComputeLinkWorldTransforms(bot, worldFromRoot, mochi::MakeConstSpan(pose), links, ExpectOK{});

  std::vector<real> recovered;
  ComputePoseFromLinkWorldTransforms(
      bot, mochi::MakeConstSpan(links), mochi::MakeConstSpan(kNow), recovered, ExpectOK{});

  ASSERT_EQ(recovered.size(), pose.size());
  for (size_t i = 0; i < pose.size(); ++i) {
    EXPECT_NEAR(recovered[i], pose[i], 1e-5_r) << "DOF " << i;
  }
}

} // namespace
} // namespace superdex::studio
