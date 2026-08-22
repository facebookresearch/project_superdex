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

#include <mochi_core/test/mochi_test_helpers.h>
#include <superdex_robotics/utils/bot_utils.h>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

BotLinkPrefab MakeLink(char const* name, int parentLink) {
  BotLinkPrefab link;
  link.name = name;
  link.parentLink = parentLink;
  return link;
}

BotJointPrefab MakeJoint(char const* name, ArticulatedJointType type, real effortLimit) {
  BotJointPrefab joint;
  joint.name = name;
  joint.type = type;
  joint.axis = {0_r, 0_r, 1_r};
  joint.effortLimit = effortLimit;
  return joint;
}

} // namespace

// Revolute + Spherical + Prismatic chain: the spherical joint's single effortLimit repeats across
// its 3 DOFs, and the array is emitted in bot-space DOF order.
TEST(EffortLimitsTest, MultiDofSpherical_RepeatsAndOrders) {
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("a_rev", 0));
  bp.links.push_back(MakeLink("b_sph", 1));
  bp.links.push_back(MakeLink("c_pri", 2));
  bp.joints.push_back(MakeJoint("root_joint", ArticulatedJointType::Hard, kEffortUnbounded));
  bp.joints.push_back(MakeJoint("a_rev_joint", ArticulatedJointType::Revolute, 10_r));
  bp.joints.push_back(MakeJoint("b_sph_joint", ArticulatedJointType::Spherical, 20_r));
  bp.joints.push_back(MakeJoint("c_pri_joint", ArticulatedJointType::Prismatic, 30_r));
  RebuildBotData(bp, ExpectOK{});

  DynamicArray<real> const limits = GetEffortLimitsPerDof(bp, ExpectOK{});
  DynamicArray<real> const expected = {10_r, 20_r, 20_r, 20_r, 30_r};
  EXPECT_EQ(limits, expected);
}

// Effort-limit values pass through verbatim across all three regimes: unbounded (< 0),
// non-actuated (0), and finite (> 0), in bot-space DOF order.
TEST(EffortLimitsTest, PassesThroughAllRegimes) {
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("a_rev", 0));
  bp.links.push_back(MakeLink("b_rev", 1));
  bp.links.push_back(MakeLink("c_rev", 2));
  bp.joints.push_back(MakeJoint("root_joint", ArticulatedJointType::Hard, kEffortUnbounded));
  bp.joints.push_back(MakeJoint("a_rev_joint", ArticulatedJointType::Revolute, kEffortUnbounded));
  bp.joints.push_back(MakeJoint("b_rev_joint", ArticulatedJointType::Revolute, 0_r));
  bp.joints.push_back(MakeJoint("c_rev_joint", ArticulatedJointType::Revolute, 25_r));
  RebuildBotData(bp, ExpectOK{});

  DynamicArray<real> const limits = GetEffortLimitsPerDof(bp, ExpectOK{});
  DynamicArray<real> const expected = {kEffortUnbounded, 0_r, 25_r};
  EXPECT_EQ(limits, expected);
}

// A stale/out-of-range _dofIndices entry (e.g. not rebuilt after edits) is reported as an error.
TEST(EffortLimitsTest, StaleDofIndices_Errors) {
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("a_rev", 0));
  bp.joints.push_back(MakeJoint("root_joint", ArticulatedJointType::Hard, kEffortUnbounded));
  bp.joints.push_back(MakeJoint("a_rev_joint", ArticulatedJointType::Revolute, 5_r));
  RebuildBotData(bp, ExpectOK{});

  bp._dofIndices.push_back(999); // out-of-range joint index
  Error error;
  DynamicArray<real> const limits = GetEffortLimitsPerDof(bp, error);
  EXPECT_NOT_OK(error);
  EXPECT_TRUE(limits.empty());
}
