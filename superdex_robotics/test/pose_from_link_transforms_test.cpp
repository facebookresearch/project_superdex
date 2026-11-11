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

#include "mochi_bots_test_helpers.h"

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_physics/mochi_physics.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <random>
#include <string>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

constexpr real kTolerance = 1e-4_r; // [rad or m]

/* @brief A random pose: revolute angles and spherical rotation-vector components in [-1, 1] rad
 * (so every joint rotation is below pi), prismatic positions in [-0.05, 0.05] m. */
DynamicArray<real> RandomPose(BotPrefab const& bot, std::mt19937& rng) {
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  DynamicArray<real> pose;
  for (int i = 1; i < isize(bot.joints); ++i) {
    switch (bot.joints[i].type) {
      case ArticulatedJointType::Revolute:
        pose.push_back(static_cast<real>(unit(rng)));
        break;
      case ArticulatedJointType::Prismatic:
        pose.push_back(static_cast<real>(0.05 * unit(rng)));
        break;
      case ArticulatedJointType::Spherical:
        for (int k = 0; k < 3; ++k) {
          pose.push_back(static_cast<real>(unit(rng)));
        }
        break;
      case ArticulatedJointType::Free:
      case ArticulatedJointType::Hard:
      case ArticulatedJointType::Cycle:
      case ArticulatedJointType::Count:
        break;
    }
  }
  return pose;
}

class PoseFromLinkTransformsTest : public ::testing::TestWithParam<std::filesystem::path> {};

/* @brief ComputePoseFromLinkTransforms inverts ComputeLinkTransformsFromPose in both spaces. */
TEST_P(PoseFromLinkTransformsTest, RoundTripsComputeLinkTransformsFromPose) {
  BotPrefab const bot = LoadBotPrefabFromFile(GetParam().string(), ExpectOK{});
  if (bot.links.empty()) {
    GTEST_SKIP() << "no links";
  }
  std::mt19937 rng(7);
  DynamicArray<real> const pose = RandomPose(bot, rng);
  ASSERT_EQ(isize(pose), bot._numDofs);

  for (LinkTransformSpace const space :
       {LinkTransformSpace::ParentFromLink, LinkTransformSpace::RootFromParent}) {
    DynamicArray<TransformRT> transforms;
    ComputeLinkTransformsFromPose(bot, MakeConstSpan(pose), transforms, space, ExpectOK{});
    DynamicArray<real> recovered;
    ComputePoseFromLinkTransforms(bot, MakeConstSpan(transforms), recovered, space, ExpectOK{});
    ASSERT_EQ(isize(recovered), isize(pose));
    for (int i = 0; i < isize(pose); ++i) {
      EXPECT_NEAR(recovered[i], pose[i], kTolerance) << "DOF " << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllBots,
    PoseFromLinkTransformsTest,
    ::testing::ValuesIn(FindAllBotFiles()),
    [](::testing::TestParamInfo<std::filesystem::path> const& info) {
      auto const& path = info.param;
      auto const botsRoot = FindBotsRoot(path);
      std::string name = botsRoot.has_value()
          ? std::filesystem::relative(path, botsRoot.value()).replace_extension().string()
          : path.stem().string();
      for (char& c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c))) {
          c = '_';
        }
      }
      return name;
    });

/* @brief The OpenXR right hand (Free root; Hard, Revolute and Spherical joints) with rest frames
 * that rotate and translate on both sides of every joint, its first revolute joint made prismatic
 * with a non-unit axis, and the next revolute axis made non-unit too. */
BotPrefab MakeKinematicsBot() {
  BotPrefab bot = LoadBotPrefabFromFile(
      (std::filesystem::path(GetAssetsDir()) / "bots" / "hands" / "openxr" / "right" /
       "openxr_hand_lowpoly_right.superdex_bot")
          .string(),
      ExpectOK{});
  int numRevolute = 0;
  for (int i = 1; i < isize(bot.links); ++i) {
    auto& joint = bot.joints[i];
    joint.parentLinkFromJoint = TransformRT{
        Quaternion::FromRotationVector(Real3{0.1_r, -0.2_r, 0.15_r}) *
            joint.parentLinkFromJoint.GetRotation(),
        joint.parentLinkFromJoint.GetTranslation()};
    bot.links[i].parentJointFromLink = TransformRT{
        Quaternion::FromRotationVector(Real3{-0.2_r, 0.05_r, 0.1_r}),
        Real3{0.001_r, -0.002_r, 0.003_r}};
    if (joint.type == ArticulatedJointType::Revolute && numRevolute++ < 2) {
      if (numRevolute == 1) {
        joint.type = ArticulatedJointType::Prismatic;
        joint.minLimit.reset();
        joint.maxLimit.reset();
      }
      joint.axis = joint.axis * 2_r;
    }
  }
  RebuildBotData(bot, ExpectOK{});
  return bot;
}

/* @brief Index of each link's first DOF in the bot-space pose, or -1 for links without DOFs. */
DynamicArray<int> FirstDofIndices(BotPrefab const& bot) {
  DynamicArray<int> first(bot.links.size(), -1);
  int iDof = 0;
  for (int i = 1; i < isize(bot.joints); ++i) {
    auto const type = bot.joints[i].type;
    int const numDofs = type == ArticulatedJointType::Spherical                             ? 3
        : type == ArticulatedJointType::Revolute || type == ArticulatedJointType::Prismatic ? 1
                                                                                            : 0;
    if (numDofs > 0) {
      first[i] = iDof;
    }
    iDof += numDofs;
  }
  return first;
}

int FirstLinkOfType(BotPrefab const& bot, ArticulatedJointType type) {
  for (int i = 1; i < isize(bot.joints); ++i) {
    if (bot.joints[i].type == type) {
      return i;
    }
  }
  return -1;
}

/* @brief Angle of the rotation between two rotations [rad], in [0, pi]. */
real RotationDistance(Quaternion const& a, Quaternion const& b) {
  Quaternion const delta = Normalize(a).GetConjugate() * Normalize(b);
  return 2_r *
      std::atan2(Norm(Real3{delta.data[0], delta.data[1], delta.data[2]}), Abs(delta.data[3]));
}

void ExpectSameTransform(TransformRT const& a, TransformRT const& b, int i) {
  EXPECT_LT(RotationDistance(a.GetRotation(), b.GetRotation()), kTolerance) << "link " << i;
  EXPECT_LT(Norm(a.GetTranslation() - b.GetTranslation()), kTolerance) << "link " << i;
}

void ExpectSamePose(Span<real const> actual, Span<real const> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], kTolerance) << "DOF " << i;
  }
}

DynamicArray<TransformRT>
Forward(BotPrefab const& bot, Span<real const> pose, LinkTransformSpace space) {
  DynamicArray<TransformRT> transforms;
  ComputeLinkTransformsFromPose(bot, pose, transforms, space, ExpectOK{});
  return transforms;
}

DynamicArray<real> Inverse(
    BotPrefab const& bot,
    Span<TransformRT const> transforms,
    LinkTransformSpace space,
    Span<real const> referencePose = {}) {
  DynamicArray<real> pose;
  ComputePoseFromLinkTransforms(bot, transforms, pose, space, ExpectOK{}, referencePose);
  return pose;
}

/* @brief Parent-from-link transforms of a random pose with every link moved by a random rotation
 * of up to about 1.2 rad and a random translation of up to about 2 cm, which in general no joint
 * can fully produce. */
DynamicArray<TransformRT> RandomOffAxisParentFromLinks(BotPrefab const& bot, std::mt19937& rng) {
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  auto Random3 = [&](real magnitude) {
    return Real3{
        static_cast<real>(magnitude * unit(rng)),
        static_cast<real>(magnitude * unit(rng)),
        static_cast<real>(magnitude * unit(rng))};
  };
  auto transforms = Forward(bot, RandomPose(bot, rng), LinkTransformSpace::ParentFromLink);
  for (int i = 1; i < isize(transforms); ++i) {
    transforms[i] = transforms[i] *
        TransformRT{Quaternion::FromRotationVector(Random3(0.7_r)), Random3(0.012_r)};
  }
  return transforms;
}

/* @brief How far link @p i posed by @p pose is from @p target relative to its parent, in what its
 * joint controls: the link rotation for rotational joints, the joint frame's origin (the link's
 * transform without parentJointFromLink) for prismatic ones. */
real JointError(BotPrefab const& bot, Span<real const> pose, int i, TransformRT const& target) {
  TransformRT const link = Forward(bot, pose, LinkTransformSpace::ParentFromLink)[i];
  if (bot.joints[i].type != ArticulatedJointType::Prismatic) {
    return RotationDistance(link.GetRotation(), target.GetRotation());
  }
  TransformRT const linkFromJoint = Invert(bot.links[i].parentJointFromLink);
  return Norm((link * linkFromJoint).GetTranslation() - (target * linkFromJoint).GetTranslation());
}

/* @brief Checks that each revolute and prismatic DOF of @p pose is the best value for its link
 * against a dense sweep, and that each spherical joint matches its link's rotation exactly. */
void ExpectJointwiseNearest(
    BotPrefab const& bot,
    Span<real const> pose,
    Span<TransformRT const> parentFromLinkTargets) {
  auto const firstDof = FirstDofIndices(bot);
  for (int i = 1; i < isize(bot.links); ++i) {
    if (firstDof[i] < 0) {
      continue;
    }
    auto const type = bot.joints[i].type;
    real const best = JointError(bot, pose, i, parentFromLinkTargets[i]);
    if (type == ArticulatedJointType::Spherical) {
      EXPECT_LT(best, kTolerance) << "spherical link " << i;
      continue;
    }
    DynamicArray<real> trial(pose.begin(), pose.end());
    int constexpr kSteps = 2000;
    for (int k = 0; k <= kSteps; ++k) {
      real const t = 2_r * static_cast<real>(k) / kSteps - 1_r;
      trial[firstDof[i]] =
          type == ArticulatedJointType::Revolute ? kPI * t : pose[firstDof[i]] + 0.1_r * t;
      EXPECT_LE(best, JointError(bot, trial, i, parentFromLinkTargets[i]) + 1e-5_r)
          << "link " << i << " at " << trial[firstDof[i]];
    }
  }
}

/* @brief Our FK is Mochi's articulated body FK for the instantiated bot, including rest frames on
 * both sides of each joint and non-unit revolute and prismatic axes (Mochi normalizes them when it
 * creates the actor), and our inverse recovers its poses. */
TEST(PoseFromLinkTransforms, MatchesMochiArticulatedBodyKinematics) {
  BotPrefab const bot = MakeKinematicsBot();
  Context* context = CreateContext(0);
  Scene* scene = context->CreateScene("pose_from_link_transforms");
  Actor* actor = AddToScene(bot, scene, ExpectOK{});
  ASSERT_NE(actor, nullptr);
  auto const info = actor->GetArticulatedShapeInfo(ExpectOK{});
  articulated::RestTransformArray rest;
  for (size_t i = 0; i < info.parents.size(); ++i) {
    rest.emplace_back(info.jointFromChildLink[i], info.parentLinkFromJoint[i]);
  }
  auto const poseInfo = articulated::SetupJointPose(info.jointTypes);
  int const numRootDofs = info.dofInfo[0].GetSize();
  int const numDofs = articulated::GetReducedDofsSize(info.dofInfo);
  ASSERT_EQ(numDofs - numRootDofs, bot._numDofs);

  std::mt19937 rng(11);
  for (int trial = 0; trial < 5; ++trial) {
    DynamicArray<real> const pose = RandomPose(bot, rng);
    ColumnVector<real> dofs(numDofs);
    std::fill(dofs.Data(), dofs.Data() + numRootDofs, 0_r);
    std::copy(pose.begin(), pose.end(), dofs.Data() + numRootDofs);
    ColumnVector<real> reducedPose(articulated::GetReducedPoseSize(poseInfo));
    articulated::ConvertDofsToPose(
        info.jointTypes, info.dofInfo, poseInfo, AsConstView(dofs), AsView(reducedPose));
    DynamicArray<TransformRT> joints(bot.links.size()), expected(bot.links.size());
    articulated::ComputeTransformsFromReducedPose(
        info.jointTypes,
        info.jointAxes,
        poseInfo,
        info.parents,
        rest,
        TransformRT::Identity(),
        AsConstView(reducedPose),
        joints,
        expected);
    auto const actual = Forward(bot, pose, LinkTransformSpace::RootFromParent);
    for (int i = 0; i < isize(expected); ++i) {
      ExpectSameTransform(actual[i], expected[i], i);
    }
    ExpectSamePose(Inverse(bot, expected, LinkTransformSpace::RootFromParent), pose);
  }
  context->DestroyScene(scene);
  DestroyContext(context);
}

/* @brief On-axis transforms round-trip in both spaces. */
TEST(PoseFromLinkTransforms, RoundTripsWithRestFramesAndNonUnitAxes) {
  BotPrefab const bot = MakeKinematicsBot();
  std::mt19937 rng(3);
  for (LinkTransformSpace const space :
       {LinkTransformSpace::ParentFromLink, LinkTransformSpace::RootFromParent}) {
    DynamicArray<real> const pose = RandomPose(bot, rng);
    ExpectSamePose(Inverse(bot, Forward(bot, pose, space), space), pose);
  }
}

/* @brief Off-axis link transforms (which lose data) give, joint by joint, the nearest pose the
 * joints can produce; projecting that pose's transforms again changes nothing; and the same
 * skeleton in RootFromParent space gives the same pose, so a link's error stays in its own joint
 * rather than moving into its children. */
TEST(PoseFromLinkTransforms, OffAxisTransformsProjectToNearestPose) {
  BotPrefab const bot = MakeKinematicsBot();
  std::mt19937 rng(5);
  for (int trial = 0; trial < 3; ++trial) {
    auto const parentFromLinks = RandomOffAxisParentFromLinks(bot, rng);
    auto const projected = Inverse(bot, parentFromLinks, LinkTransformSpace::ParentFromLink);
    ExpectJointwiseNearest(bot, projected, parentFromLinks);
    ExpectSamePose(
        Inverse(
            bot,
            Forward(bot, projected, LinkTransformSpace::ParentFromLink),
            LinkTransformSpace::ParentFromLink),
        projected);
    DynamicArray<TransformRT> rootFromLinks = parentFromLinks;
    for (int i = 1; i < isize(rootFromLinks); ++i) {
      rootFromLinks[i] = rootFromLinks[bot.links[i].parentLink] * parentFromLinks[i];
    }
    ExpectSamePose(Inverse(bot, rootFromLinks, LinkTransformSpace::RootFromParent), projected);
  }
}

/* @brief Degenerate and branch cases of the inverse: a half turn about an axis perpendicular to a
 * revolute axis (every angle equally near), negated quaternions, revolute angles past +-pi,
 * spherical rotations near a half turn, and link-to-world transforms. */
TEST(PoseFromLinkTransforms, DegenerateAndBranchCases) {
  BotPrefab const bot = MakeKinematicsBot();
  auto const firstDof = FirstDofIndices(bot);
  auto const space = LinkTransformSpace::ParentFromLink;
  DynamicArray<real> const zero(bot._numDofs, 0_r);
  int const revolute = FirstLinkOfType(bot, ArticulatedJointType::Revolute);
  int const spherical = FirstLinkOfType(bot, ArticulatedJointType::Spherical);
  ASSERT_GT(revolute, 0);
  ASSERT_GT(spherical, 0);

  // Half turn about an axis perpendicular to the revolute axis: no angle is nearer than another,
  // so the angle is the reference's, or 0 without one.
  {
    auto targets = Forward(bot, zero, space);
    auto const& joint = bot.joints[revolute];
    Real3 const axis = Normalize(joint.axis);
    Real3 const perpendicular = Normalize(
        Cross(axis, std::abs(axis[0]) < 0.9_r ? Real3{1_r, 0_r, 0_r} : Real3{0_r, 1_r, 0_r}));
    targets[revolute] = joint.parentLinkFromJoint *
        TransformRT{Quaternion::FromAxisAngle(perpendicular, kPI), Real3{}} *
        bot.links[revolute].parentJointFromLink;
    auto const pose = Inverse(bot, targets, space);
    EXPECT_EQ(pose[firstDof[revolute]], 0_r);
    ExpectJointwiseNearest(bot, pose, targets);
    DynamicArray<real> reference = zero;
    reference[firstDof[revolute]] = 0.7_r;
    EXPECT_NEAR(Inverse(bot, targets, space, reference)[firstDof[revolute]], 0.7_r, kTolerance);
  }

  // Negated quaternions are the same rotations, on and off axis.
  {
    std::mt19937 rng(9);
    auto const targets = RandomOffAxisParentFromLinks(bot, rng);
    auto negated = targets;
    for (auto& t : negated) {
      Quaternion const q = t.GetRotation();
      t.SetRotation(Quaternion(-q.data[0], -q.data[1], -q.data[2], -q.data[3]));
    }
    ExpectSamePose(Inverse(bot, negated, space), Inverse(bot, targets, space));
  }

  // Without a reference, revolute angles past +-pi wrap into [-pi, pi], to the same rotation.
  for (real const angle : {kPI - 1e-3_r, -kPI + 1e-3_r, kPI + 1e-2_r, -kPI - 1e-2_r, 7_r}) {
    DynamicArray<real> pose = zero;
    pose[firstDof[revolute]] = angle;
    auto const targets = Forward(bot, pose, space);
    real const recovered = Inverse(bot, targets, space)[firstDof[revolute]];
    EXPECT_LE(Abs(recovered), kPI);
    EXPECT_NEAR(std::remainder(recovered - angle, 2_r * kPI), 0_r, kTolerance) << angle;
  }

  // Spherical rotations near a half turn: the rotation vector has norm <= pi and is the same
  // rotation.
  for (real const angle : {kPI - 1e-3_r, kPI + 1e-3_r}) {
    DynamicArray<real> pose = zero;
    Real3 const rotVec = Normalize(Real3{0.3_r, -0.5_r, 0.8_r}) * angle;
    for (int k = 0; k < 3; ++k) {
      pose[firstDof[spherical] + k] = rotVec[k];
    }
    auto const targets = Forward(bot, pose, space);
    auto const recovered = Inverse(bot, targets, space);
    EXPECT_LE(
        Norm(
            Real3{
                recovered[firstDof[spherical]],
                recovered[firstDof[spherical] + 1],
                recovered[firstDof[spherical] + 2]}),
        kPI + kTolerance);
    ExpectSameTransform(Forward(bot, recovered, space)[spherical], targets[spherical], spherical);
  }

  // RootFromParent transforms are only used relative to each other, so link-to-world works too.
  {
    std::mt19937 rng(13);
    auto const rootSpace = LinkTransformSpace::RootFromParent;
    auto const targets = Forward(bot, RandomPose(bot, rng), rootSpace);
    TransformRT const worldFromRoot{
        Quaternion::FromRotationVector(Real3{0.3_r, -0.5_r, 0.7_r}), Real3{0.4_r, -0.2_r, 1.3_r}};
    auto world = targets;
    for (auto& t : world) {
      t = worldFromRoot * t;
    }
    ExpectSamePose(Inverse(bot, world, rootSpace), Inverse(bot, targets, rootSpace));
  }
}

/* @brief With the previous frame's pose as the reference, a revolute joint spun through several
 * turns in steps of under pi keeps counting turns in either direction. */
TEST(PoseFromLinkTransforms, ReferencePoseTracksMultipleTurns) {
  BotPrefab const bot = MakeKinematicsBot();
  int const iDof = FirstDofIndices(bot)[FirstLinkOfType(bot, ArticulatedJointType::Revolute)];
  auto const space = LinkTransformSpace::ParentFromLink;
  for (real const step : {0.9_r, -0.9_r}) {
    DynamicArray<real> pose(bot._numDofs, 0_r);
    DynamicArray<real> previous = pose;
    for (int frame = 1; frame <= 25; ++frame) { // about 3.6 turns
      pose[iDof] = step * static_cast<real>(frame);
      previous = Inverse(bot, Forward(bot, pose, space), space, previous);
      EXPECT_NEAR(previous[iDof], pose[iDof], kTolerance) << "frame " << frame;
    }
  }
}

DynamicArray<LinkPoseResidual>
Residuals(BotPrefab const& bot, Span<TransformRT const> transforms, LinkTransformSpace space) {
  DynamicArray<real> pose;
  DynamicArray<LinkPoseResidual> residuals;
  ComputePoseFromLinkTransforms(bot, transforms, pose, space, ExpectOK{}, {}, &residuals);
  return residuals;
}

Real3 PerpendicularTo(Real3 const& axis) {
  return Normalize(
      Cross(axis, std::abs(axis[0]) < 0.9_r ? Real3{1_r, 0_r, 0_r} : Real3{0_r, 1_r, 0_r}));
}

/* @brief Transforms that come from a pose leave only rounding error in every link, in both
 * spaces, and the root link's residual is zero. */
TEST(PoseFromLinkTransforms, OnAxisTransformsLeaveNoResidual) {
  BotPrefab const bot = MakeKinematicsBot();
  std::mt19937 rng(17);
  for (LinkTransformSpace const space :
       {LinkTransformSpace::ParentFromLink, LinkTransformSpace::RootFromParent}) {
    auto const residuals = Residuals(bot, Forward(bot, RandomPose(bot, rng), space), space);
    ASSERT_EQ(isize(residuals), isize(bot.links));
    EXPECT_EQ(residuals[0].rotation, 0_r);
    EXPECT_EQ(residuals[0].translation, 0_r);
    for (int i = 1; i < isize(residuals); ++i) {
      EXPECT_LT(residuals[i].rotation, kTolerance) << "link " << i;
      EXPECT_LT(residuals[i].translation, kTolerance) << "link " << i;
    }
  }
}

/* @brief Each joint type's residual is the motion it discards: a revolute joint's swing off its
 * axis and its translation, a prismatic joint's rotation and off-axis translation, a spherical
 * joint's translation, and all of a Hard joint's motion. The same skeleton in RootFromParent space
 * gives the same residuals, so a link's error stays with its own joint. */
TEST(PoseFromLinkTransforms, ResidualsMeasureWhatEachJointDiscards) {
  BotPrefab const bot = MakeKinematicsBot();
  auto const space = LinkTransformSpace::ParentFromLink;
  auto targets = Forward(bot, DynamicArray<real>(bot._numDofs, 0_r), space);
  auto const SetMotion = [&](int i, TransformRT const& motion) {
    targets[i] = bot.joints[i].parentLinkFromJoint * motion * bot.links[i].parentJointFromLink;
  };

  int const revolute = FirstLinkOfType(bot, ArticulatedJointType::Revolute);
  Real3 const revoluteAxis = Normalize(bot.joints[revolute].axis);
  // A 0.4 rad turn about the axis, then a 0.3 rad swing about a perpendicular axis.
  SetMotion(
      revolute,
      TransformRT{
          Quaternion::FromAxisAngle(revoluteAxis, 0.4_r) *
              Quaternion::FromAxisAngle(PerpendicularTo(revoluteAxis), 0.3_r),
          Real3{0.003_r, 0_r, -0.004_r}});

  int const prismatic = FirstLinkOfType(bot, ArticulatedJointType::Prismatic);
  Real3 const prismaticAxis = Normalize(bot.joints[prismatic].axis);
  SetMotion(
      prismatic,
      TransformRT{
          Quaternion::FromAxisAngle(PerpendicularTo(prismaticAxis), 0.25_r),
          prismaticAxis * 0.02_r + PerpendicularTo(prismaticAxis) * 0.005_r});

  int const spherical = FirstLinkOfType(bot, ArticulatedJointType::Spherical);
  SetMotion(
      spherical,
      TransformRT{
          Quaternion::FromRotationVector(Real3{0.2_r, -0.1_r, 0.3_r}), Real3{0_r, 0.006_r, 0_r}});

  int const hard = FirstLinkOfType(bot, ArticulatedJointType::Hard);
  ASSERT_GT(revolute, 0);
  ASSERT_GT(prismatic, 0);
  ASSERT_GT(spherical, 0);
  ASSERT_GT(hard, 0);
  SetMotion(
      hard,
      TransformRT{
          Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, 0.1_r), Real3{0_r, 0_r, 0.002_r}});

  auto const residuals = Residuals(bot, targets, space);
  EXPECT_NEAR(residuals[revolute].rotation, 0.3_r, kTolerance);
  EXPECT_NEAR(residuals[revolute].translation, 0.005_r, kTolerance);
  EXPECT_NEAR(residuals[prismatic].rotation, 0.25_r, kTolerance);
  EXPECT_NEAR(residuals[prismatic].translation, 0.005_r, kTolerance);
  EXPECT_NEAR(residuals[spherical].rotation, 0_r, kTolerance);
  EXPECT_NEAR(residuals[spherical].translation, 0.006_r, kTolerance);
  EXPECT_NEAR(residuals[hard].rotation, 0.1_r, kTolerance);
  EXPECT_NEAR(residuals[hard].translation, 0.002_r, kTolerance);

  DynamicArray<TransformRT> rootFromLinks = targets;
  for (int i = 1; i < isize(rootFromLinks); ++i) {
    rootFromLinks[i] = rootFromLinks[bot.links[i].parentLink] * targets[i];
  }
  auto const rootResiduals = Residuals(bot, rootFromLinks, LinkTransformSpace::RootFromParent);
  for (int i = 0; i < isize(residuals); ++i) {
    EXPECT_NEAR(rootResiduals[i].rotation, residuals[i].rotation, kTolerance) << "link " << i;
    EXPECT_NEAR(rootResiduals[i].translation, residuals[i].translation, kTolerance) << "link " << i;
  }
}

/* @brief Size mismatches are rejected without changing the output, as is a pending error. */
TEST(PoseFromLinkTransforms, RejectsMismatchedSizesWithoutChangingOutput) {
  BotPrefab const bot = MakeKinematicsBot();
  DynamicArray<real> const zero(bot._numDofs, 0_r);
  auto const transforms = Forward(bot, zero, LinkTransformSpace::ParentFromLink);
  DynamicArray<real> const poseSentinel(3, 123_r);
  DynamicArray<TransformRT> const transformSentinel(
      2, TransformRT{Quaternion::Identity(), Real3{1_r, 2_r, 3_r}});
  auto const shortTransforms = Span<TransformRT const>{transforms.data(), transforms.size() - 1};
  auto const shortPose = Span<real const>{zero.data(), zero.size() - 1};
  {
    DynamicArray<real> pose = poseSentinel;
    DynamicArray<LinkPoseResidual> residuals(2, LinkPoseResidual{1_r, 2_r});
    ComputePoseFromLinkTransforms(
        bot,
        shortTransforms,
        pose,
        LinkTransformSpace::ParentFromLink,
        ExpectNotOK{},
        {},
        &residuals);
    EXPECT_EQ(pose, poseSentinel);
    ASSERT_EQ(isize(residuals), 2);
    EXPECT_EQ(residuals[1].rotation, 1_r);
    EXPECT_EQ(residuals[1].translation, 2_r);
    ComputePoseFromLinkTransforms(
        bot, transforms, pose, LinkTransformSpace::ParentFromLink, ExpectNotOK{}, shortPose);
    EXPECT_EQ(pose, poseSentinel);
    DynamicArray<TransformRT> out = transformSentinel;
    ComputeLinkTransformsFromPose(
        bot, shortPose, out, LinkTransformSpace::ParentFromLink, ExpectNotOK{});
    EXPECT_EQ(out, transformSentinel);
  }
  Error existingError;
  MOCHI_ERROR_SET(existingError, "Previous failure");
  DynamicArray<real> pose = poseSentinel;
  ComputePoseFromLinkTransforms(
      bot, transforms, pose, LinkTransformSpace::ParentFromLink, existingError);
  EXPECT_EQ(pose, poseSentinel);
  DynamicArray<TransformRT> out = transformSentinel;
  ComputeLinkTransformsFromPose(bot, zero, out, LinkTransformSpace::ParentFromLink, existingError);
  EXPECT_EQ(out, transformSentinel);
}

} // namespace
