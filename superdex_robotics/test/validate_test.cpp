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

#include <limits>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

// Creates a minimal valid chain-topology bot with the given number of links.
// Link 0 = root (parentLink = kIndexNone), link i has parent i-1.
// The root joint (joint[0]) is Hard; all other joints are Revolute with
// axis = {0, 0, 1} and limits along z.
BotPrefab MakeValidBotPrefab(int numLinks) {
  BotPrefab bp;
  bp.name = "test_bot";
  for (int i = 0; i < numLinks; ++i) {
    BotLinkPrefab link;
    link.name = "link_" + DynamicString(std::to_string(i));
    link.parentLink = (i == 0) ? kIndexNone : (i - 1);
    bp.links.push_back(std::move(link));

    BotJointPrefab joint;
    joint.name = "joint_" + DynamicString(std::to_string(i));
    // The root joint must be Free or Hard; all other joints are Revolute.
    joint.type = (i == 0) ? ArticulatedJointType::Hard : ArticulatedJointType::Revolute;
    joint.axis = {0_r, 0_r, 1_r};
    joint.minLimit = {0_r, 0_r, -1_r};
    joint.maxLimit = {0_r, 0_r, 1_r};
    bp.joints.push_back(std::move(joint));
  }
  return bp;
}

// Test fixture that suppresses expected MOCHI_LOG_WARNING calls from Validate.
// The Validate function logs warnings before setting errors; the test framework
// treats unexpected warnings as failures. This follows the established pattern
// used in deep_flow_map_test.cpp and mochi_context_test.cpp.
class BotUtilsValidate : public testing::Test {
 protected:
  void SetUp() override {
    _prevLogFn = GetLogCallback();
    SetLogCallback([](LogChannel, char const*, char const*, int) {});
  }
  void TearDown() override {
    SetLogCallback(_prevLogFn);
  }
  LogFn _prevLogFn;
};

} // namespace

// 1. Valid bot passes validation
TEST_F(BotUtilsValidate, ValidBot) {
  auto bp = MakeValidBotPrefab(3);
  Validate(bp, nullptr, ExpectOK{});
}

// Empty bot name
TEST_F(BotUtilsValidate, EmptyBotName) {
  auto bp = MakeValidBotPrefab(2);
  bp.name = "";
  Validate(bp, nullptr, ExpectNotOK{});
}

// 2. Mismatched joints/links count
TEST_F(BotUtilsValidate, MismatchedJointsLinksCount) {
  auto bp = MakeValidBotPrefab(3);
  bp.joints.pop_back();
  Validate(bp, nullptr, ExpectNotOK{});
}

// 3a. Empty link name
TEST_F(BotUtilsValidate, EmptyLinkName) {
  auto bp = MakeValidBotPrefab(2);
  bp.links[1].name = "";
  Validate(bp, nullptr, ExpectNotOK{});
}

// 3b. Empty joint name
TEST_F(BotUtilsValidate, EmptyJointName) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[1].name = "";
  Validate(bp, nullptr, ExpectNotOK{});
}

// 4a. Duplicate link-link names
TEST_F(BotUtilsValidate, DuplicateLinkLinkName) {
  auto bp = MakeValidBotPrefab(3);
  bp.links[2].name = bp.links[1].name;
  Validate(bp, nullptr, ExpectNotOK{});
}

// 4b. Duplicate joint-joint names
TEST_F(BotUtilsValidate, DuplicateJointJointName) {
  auto bp = MakeValidBotPrefab(3);
  bp.joints[2].name = bp.joints[1].name;
  Validate(bp, nullptr, ExpectNotOK{});
}

// 4c. Duplicate joint-link names
TEST_F(BotUtilsValidate, DuplicateJointLinkName) {
  auto bp = MakeValidBotPrefab(3);
  bp.joints[0].name = bp.links[0].name;
  Validate(bp, nullptr, ExpectNotOK{});
}

// 5. Root link with non-kIndexNone parent
TEST_F(BotUtilsValidate, RootLinkWithParent) {
  auto bp = MakeValidBotPrefab(2);
  bp.links[0].parentLink = 0;
  Validate(bp, nullptr, ExpectNotOK{});
}

// 6. Out-of-range parent index
TEST_F(BotUtilsValidate, OutOfRangeParentIndex) {
  auto bp = MakeValidBotPrefab(3);
  bp.links[2].parentLink = 99;
  Validate(bp, nullptr, ExpectNotOK{});
}

// 7. Self-referencing parent
TEST_F(BotUtilsValidate, SelfReferencingParent) {
  auto bp = MakeValidBotPrefab(3);
  bp.links[1].parentLink = 1;
  Validate(bp, nullptr, ExpectNotOK{});
}

// 8. Circular parent chain
TEST_F(BotUtilsValidate, CircularParentChain) {
  auto bp = MakeValidBotPrefab(4);
  // Create a cycle: link 1 -> 2, link 2 -> 3, link 3 -> 1
  bp.links[1].parentLink = 0;
  bp.links[2].parentLink = 3;
  bp.links[3].parentLink = 2;
  Validate(bp, nullptr, ExpectNotOK{});
}

// 9. Revolute joint with zero axis
TEST_F(BotUtilsValidate, RevoluteJointZeroAxis) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[1].axis = {0_r, 0_r, 0_r};
  Validate(bp, nullptr, ExpectNotOK{});
}

// 10. Zero limits in active axis
TEST_F(BotUtilsValidate, ZeroLimitsInActiveAxis) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[1].minLimit = {0_r, 0_r, 0_r};
  bp.joints[1].maxLimit = {0_r, 0_r, 0_r};
  Validate(bp, nullptr, ExpectNotOK{});
}

// 11. minLimit > maxLimit
TEST_F(BotUtilsValidate, MinLimitGreaterThanMaxLimit) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[1].minLimit = {0_r, 0_r, 2_r};
  bp.joints[1].maxLimit = {0_r, 0_r, -2_r};
  Validate(bp, nullptr, ExpectNotOK{});
}

// 12a. Negative viscous friction coefficient
TEST_F(BotUtilsValidate, NegativeViscousFriction) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[1].friction.viscous = -1_r;
  Validate(bp, nullptr, ExpectNotOK{});
}

// 12b. Negative coulomb friction coefficient
TEST_F(BotUtilsValidate, NegativeCoulombFriction) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[1].friction.coulomb = -1_r;
  Validate(bp, nullptr, ExpectNotOK{});
}

// 12c. Negative inertia coefficient
TEST_F(BotUtilsValidate, NegativeInertia) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[1].inertia = -1_r;
  Validate(bp, nullptr, ExpectNotOK{});
}

// 12d. Negative limit stiffness coefficient
TEST_F(BotUtilsValidate, NegativeLimitStiffness) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[1].limitStiffness = -1_r;
  Validate(bp, nullptr, ExpectNotOK{});
}

// 12e. Negative limit damping coefficient
TEST_F(BotUtilsValidate, NegativeLimitDamping) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[1].limitDamping = -1_r;
  Validate(bp, nullptr, ExpectNotOK{});
}

// 13a. Mass only is a valid combination (mochi shape provides geometry for COM/MOI)
TEST_F(BotUtilsValidate, ValidMassProperties_MassOnly) {
  auto bp = MakeValidBotPrefab(2);
  bp.links[1].shapeFile = "test.mochi.h5";
  bp.links[1].mass = 1_r;
  Validate(bp, nullptr, ExpectOK{});
}

// 13b. Invalid mass property combination: density + mass
TEST_F(BotUtilsValidate, InvalidMassPropertyCombination_DensityAndMass) {
  auto bp = MakeValidBotPrefab(2);
  bp.links[1].shapeFile = "test.mochi.h5";
  bp.links[1].density = 100_r;
  bp.links[1].mass = 1_r;
  Validate(bp, nullptr, ExpectNotOK{});
}

// 14a. Valid mass property: all null
TEST_F(BotUtilsValidate, ValidMassProperties_AllNull) {
  auto bp = MakeValidBotPrefab(2);
  Validate(bp, nullptr, ExpectOK{});
}

// 14b. Valid mass property: density only (with mochi shape)
TEST_F(BotUtilsValidate, ValidMassProperties_DensityOnly) {
  auto bp = MakeValidBotPrefab(2);
  bp.links[1].shapeFile = "test.mochi.h5";
  bp.links[1].density = 100_r;
  Validate(bp, nullptr, ExpectOK{});
}

// 14c. Valid mass property: mass + COM + MOI (with mochi shape)
TEST_F(BotUtilsValidate, ValidMassProperties_MassCOMMOI) {
  auto bp = MakeValidBotPrefab(2);
  bp.links[1].shapeFile = "test.mochi.h5";
  bp.links[1].mass = 1_r;
  bp.links[1].centerOfMass = Real3{0_r, 0_r, 0_r};
  bp.links[1].momentOfInertia = Real6{1_r, 1_r, 1_r, 0_r, 0_r, 0_r};
  Validate(bp, nullptr, ExpectOK{});
}

// 14d. Valid mass property: density + COM + MOI (with mochi shape)
TEST_F(BotUtilsValidate, ValidMassProperties_DensityCOMMOI) {
  auto bp = MakeValidBotPrefab(2);
  bp.links[1].shapeFile = "test.mochi.h5";
  bp.links[1].density = 100_r;
  bp.links[1].centerOfMass = Real3{0_r, 0_r, 0_r};
  bp.links[1].momentOfInertia = Real6{1_r, 1_r, 1_r, 0_r, 0_r, 0_r};
  Validate(bp, nullptr, ExpectOK{});
}

// 15a. Empty renderModelFile
TEST_F(BotUtilsValidate, EmptyRenderModelPath) {
  auto bp = MakeValidBotPrefab(2);
  bp.links[1].renderModelFile = "";
  Validate(bp, nullptr, ExpectOK{});
}

// 15b. Empty mochi shape path (valid — links without geometry, e.g. end effector frames, are OK)
TEST_F(BotUtilsValidate, EmptyMochiShapePath) {
  auto bp = MakeValidBotPrefab(2);
  bp.links[1].shapeFile = "";
  Validate(bp, nullptr, ExpectOK{});
}

// 15c. Non-empty renderModelFile passes
TEST_F(BotUtilsValidate, ValidRenderModelFile) {
  auto bp = MakeValidBotPrefab(2);
  bp.links[1].renderModelFile = "model.glb";
  Validate(bp, nullptr, ExpectOK{});
}

// 16a. Negative mass
TEST_F(BotUtilsValidate, NegativeMass) {
  auto bp = MakeValidBotPrefab(2);
  bp.links[1].shapeFile = "test.mochi.h5";
  bp.links[1].mass = -1_r;
  bp.links[1].centerOfMass = Real3{0_r, 0_r, 0_r};
  bp.links[1].momentOfInertia = Real6{1_r, 1_r, 1_r, 0_r, 0_r, 0_r};
  Validate(bp, nullptr, ExpectNotOK{});
}

// 16b. Negative density
TEST_F(BotUtilsValidate, NegativeDensity) {
  auto bp = MakeValidBotPrefab(2);
  bp.links[1].shapeFile = "test.mochi.h5";
  bp.links[1].density = -100_r;
  Validate(bp, nullptr, ExpectNotOK{});
}

// 17. Pre-existing error causes early return
TEST_F(BotUtilsValidate, PreExistingErrorCausesEarlyReturn) {
  auto bp = MakeValidBotPrefab(2);
  Error error;
  MOCHI_ERROR_SET(error, "pre-existing error");
  Validate(bp, nullptr, error);
  EXPECT_NOT_OK(error);
}

// 18a. Non-finite joint axis
TEST_F(BotUtilsValidate, NonFiniteJointAxis) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[1].axis = {std::numeric_limits<real>::infinity(), 0_r, 0_r};
  Validate(bp, nullptr, ExpectNotOK{});
}

// 18b. Non-finite joint minLimit
TEST_F(BotUtilsValidate, NonFiniteJointMinLimit) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[1].minLimit = {std::numeric_limits<real>::quiet_NaN(), 0_r, 0_r};
  Validate(bp, nullptr, ExpectNotOK{});
}

// 18c. Non-finite density
TEST_F(BotUtilsValidate, NonFiniteDensity) {
  auto bp = MakeValidBotPrefab(2);
  bp.links[1].shapeFile = "test.mochi.h5";
  bp.links[1].density = std::numeric_limits<real>::infinity();
  Validate(bp, nullptr, ExpectNotOK{});
}

// 18d. Non-finite mass
TEST_F(BotUtilsValidate, NonFiniteMass) {
  auto bp = MakeValidBotPrefab(2);
  bp.links[1].shapeFile = "test.mochi.h5";
  bp.links[1].mass = std::numeric_limits<real>::quiet_NaN();
  bp.links[1].centerOfMass = Real3{0_r, 0_r, 0_r};
  bp.links[1].momentOfInertia = Real6{1_r, 1_r, 1_r, 0_r, 0_r, 0_r};
  Validate(bp, nullptr, ExpectNotOK{});
}

// Empty bot (zero links)
TEST_F(BotUtilsValidate, EmptyBot) {
  BotPrefab bp;
  Validate(bp, nullptr, ExpectNotOK{});
}

// defaultPose exceeding number of joints
TEST_F(BotUtilsValidate, DefaultPoseExceedsJoints) {
  auto bp = MakeValidBotPrefab(2);
  bp.defaultPose.resize(10, 0_r);
  Validate(bp, nullptr, ExpectNotOK{});
}

// Prismatic joint with zero axis
TEST_F(BotUtilsValidate, PrismaticJointZeroAxis) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[1].type = ArticulatedJointType::Prismatic;
  bp.joints[1].axis = {0_r, 0_r, 0_r};
  Validate(bp, nullptr, ExpectNotOK{});
}

// Mass properties on a link without a mochi shape are ignored (e.g. end effector frames)
TEST_F(BotUtilsValidate, MassPropertiesIgnoredWithoutMochiShape) {
  auto bp = MakeValidBotPrefab(2);
  bp.links[1].mass = 1_r; // mass without COM or MOI would normally fail
  Validate(bp, nullptr, ExpectOK{});
}

// Hard joint passes without axis/limit checks
TEST_F(BotUtilsValidate, HardJointNoAxisValidation) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[1].type = ArticulatedJointType::Hard;
  bp.joints[1].axis = {0_r, 0_r, 0_r};
  bp.joints[1].minLimit = {0_r, 0_r, 0_r};
  bp.joints[1].maxLimit = {0_r, 0_r, 0_r};
  Validate(bp, nullptr, ExpectOK{});
}

// Unlimited joint limits (-inf, +inf) are valid
TEST_F(BotUtilsValidate, UnlimitedJointLimits) {
  auto bp = MakeValidBotPrefab(2);
  auto const inf = std::numeric_limits<real>::infinity();
  bp.joints[1].minLimit = {0_r, 0_r, -inf};
  bp.joints[1].maxLimit = {0_r, 0_r, inf};
  Validate(bp, nullptr, ExpectOK{});
}

// ValidateResults: valid bot produces empty results
TEST_F(BotUtilsValidate, ValidBotNoResults) {
  auto bp = MakeValidBotPrefab(3);
  ValidateResults results;
  Validate(bp, &results, ExpectOK{});
  EXPECT_TRUE(results.botIssues.empty());
  EXPECT_EQ(isize(results.linkIssues), 3);
  EXPECT_EQ(isize(results.jointIssues), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(results.linkIssues[i].empty());
    EXPECT_TRUE(results.jointIssues[i].empty());
  }
}

// ValidateResults: link issue collected
TEST_F(BotUtilsValidate, ResultsCollectsLinkIssue) {
  auto bp = MakeValidBotPrefab(3);
  bp.links[1].name = "";
  ValidateResults results;
  Validate(bp, &results, ExpectNotOK{});
  ASSERT_EQ(isize(results.linkIssues), 3);
  EXPECT_FALSE(results.linkIssues[1].empty());
}

// ValidateResults: joint issue collected
TEST_F(BotUtilsValidate, ResultsCollectsJointIssue) {
  auto bp = MakeValidBotPrefab(3);
  bp.joints[1].axis = {0_r, 0_r, 0_r};
  ValidateResults results;
  Validate(bp, &results, ExpectNotOK{});
  ASSERT_EQ(isize(results.jointIssues), 3);
  EXPECT_FALSE(results.jointIssues[1].empty());
}

// ValidateResults: bot issue collected
TEST_F(BotUtilsValidate, ResultsCollectsBotIssue) {
  BotPrefab bp;
  ValidateResults results;
  Validate(bp, &results, ExpectNotOK{});
  EXPECT_FALSE(results.botIssues.empty());
}

// ValidateResults: multiple link issues in same phase
TEST_F(BotUtilsValidate, ResultsCollectsMultipleIssuesInPhase) {
  auto bp = MakeValidBotPrefab(4);
  bp.links[1].name = "";
  bp.links[2].name = "";
  ValidateResults results;
  Validate(bp, &results, ExpectNotOK{});
  ASSERT_EQ(isize(results.linkIssues), 4);
  EXPECT_FALSE(results.linkIssues[1].empty());
  EXPECT_FALSE(results.linkIssues[2].empty());
}

// ValidateResults: suppressWarnings prevents logging while still collecting issues
TEST_F(BotUtilsValidate, SuppressWarningsNoWarningsEmitted) {
  // Restore the real log callback so any warning would be visible to the test framework
  SetLogCallback(_prevLogFn);
  auto bp = MakeValidBotPrefab(2);
  bp.links[1].name = "";
  ValidateResults results;
  results.suppressWarnings = true;
  Validate(bp, &results, ExpectNotOK{});
  ASSERT_EQ(isize(results.linkIssues), 2);
  EXPECT_FALSE(results.linkIssues[1].empty());
}

// Root joint may be Hard (covered by MakeValidBotPrefab; explicit for clarity)
TEST_F(BotUtilsValidate, RootJointHard_Ok) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[0].type = ArticulatedJointType::Hard;
  Validate(bp, nullptr, ExpectOK{});
}

// Root joint may be Free
TEST_F(BotUtilsValidate, RootJointFree_Ok) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[0].type = ArticulatedJointType::Free;
  Validate(bp, nullptr, ExpectOK{});
}

// Root joint may not be Revolute
TEST_F(BotUtilsValidate, RootJointRevolute_NotOk) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[0].type = ArticulatedJointType::Revolute;
  Validate(bp, nullptr, ExpectNotOK{});
}

// Root joint may not be Prismatic
TEST_F(BotUtilsValidate, RootJointPrismatic_NotOk) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[0].type = ArticulatedJointType::Prismatic;
  Validate(bp, nullptr, ExpectNotOK{});
}

// Root joint may not be Spherical
TEST_F(BotUtilsValidate, RootJointSpherical_NotOk) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[0].type = ArticulatedJointType::Spherical;
  Validate(bp, nullptr, ExpectNotOK{});
}

// Non-root joints may not be Free
TEST_F(BotUtilsValidate, NonRootJointFree_NotOk) {
  auto bp = MakeValidBotPrefab(2);
  bp.joints[1].type = ArticulatedJointType::Free;
  Validate(bp, nullptr, ExpectNotOK{});
}

// The relevant joint issue is recorded against the offending non-root joint
TEST_F(BotUtilsValidate, NonRootJointFree_ResultsCollectsJointIssue) {
  auto bp = MakeValidBotPrefab(3);
  bp.joints[2].type = ArticulatedJointType::Free;
  ValidateResults results;
  Validate(bp, &results, ExpectNotOK{});
  ASSERT_EQ(isize(results.jointIssues), 3);
  EXPECT_FALSE(results.jointIssues[2].empty());
}

// A valid cycle joint referencing two distinct in-range links passes validation.
TEST_F(BotUtilsValidate, ValidCycleJoint_Ok) {
  auto bp = MakeValidBotPrefab(3);
  ArticulatedCycleJointParams cycle;
  cycle.parentLink = 1;
  cycle.childLink = 2;
  bp.cycles.push_back(cycle);
  Validate(bp, nullptr, ExpectOK{});
}

// A cycle joint with an out-of-range childLink fails validation.
TEST_F(BotUtilsValidate, CycleJointChildOutOfRange_NotOk) {
  auto bp = MakeValidBotPrefab(3);
  ArticulatedCycleJointParams cycle;
  cycle.parentLink = 1;
  cycle.childLink = 99;
  bp.cycles.push_back(cycle);
  Validate(bp, nullptr, ExpectNotOK{});
}

// A cycle joint with an out-of-range parentLink fails validation.
TEST_F(BotUtilsValidate, CycleJointParentOutOfRange_NotOk) {
  auto bp = MakeValidBotPrefab(3);
  ArticulatedCycleJointParams cycle;
  cycle.parentLink = -1;
  cycle.childLink = 2;
  bp.cycles.push_back(cycle);
  Validate(bp, nullptr, ExpectNotOK{});
}

// A cycle joint whose parent and child are the same link fails validation.
TEST_F(BotUtilsValidate, CycleJointParentEqualsChild_NotOk) {
  auto bp = MakeValidBotPrefab(3);
  ArticulatedCycleJointParams cycle;
  cycle.parentLink = 1;
  cycle.childLink = 1;
  bp.cycles.push_back(cycle);
  Validate(bp, nullptr, ExpectNotOK{});
}

TEST_F(BotUtilsValidate, CycleJointNonFiniteTransform_NotOk) {
  auto bp = MakeValidBotPrefab(3);
  ArticulatedCycleJointParams cycle;
  cycle.parentLink = 1;
  cycle.childLink = 2;
  cycle.jointFromChildLink = TransformRT(Real3{std::numeric_limits<real>::infinity(), 0_r, 0_r});
  bp.cycles.push_back(cycle);
  Validate(bp, nullptr, ExpectNotOK{});
}

TEST_F(BotUtilsValidate, CycleJointNonUnitRotation_NotOk) {
  auto bp = MakeValidBotPrefab(3);
  ArticulatedCycleJointParams cycle;
  cycle.parentLink = 1;
  cycle.childLink = 2;
  cycle.jointFromChildLink = TransformRT(Quaternion(Vec4r{0_r, 0_r, 0_r, 0_r}));
  bp.cycles.push_back(cycle);
  Validate(bp, nullptr, ExpectNotOK{});
}

TEST_F(BotUtilsValidate, CycleJointNegativeStiffness_NotOk) {
  auto bp = MakeValidBotPrefab(3);
  ArticulatedCycleJointParams cycle;
  cycle.parentLink = 1;
  cycle.childLink = 2;
  cycle.stiffness = -1_r;
  bp.cycles.push_back(cycle);
  Validate(bp, nullptr, ExpectNotOK{});
}

TEST_F(BotUtilsValidate, CycleJointNonFiniteStiffness_NotOk) {
  auto bp = MakeValidBotPrefab(3);
  ArticulatedCycleJointParams cycle;
  cycle.parentLink = 1;
  cycle.childLink = 2;
  cycle.stiffness = std::numeric_limits<real>::infinity();
  bp.cycles.push_back(cycle);
  Validate(bp, nullptr, ExpectNotOK{});
}

// A cycle-joint issue is recorded in the bot-level issues.
TEST_F(BotUtilsValidate, CycleJoint_ResultsCollectsBotIssue) {
  auto bp = MakeValidBotPrefab(3);
  ArticulatedCycleJointParams cycle;
  cycle.parentLink = 1;
  cycle.childLink = 99;
  bp.cycles.push_back(cycle);
  ValidateResults results;
  Validate(bp, &results, ExpectNotOK{});
  EXPECT_FALSE(results.botIssues.empty());
}
