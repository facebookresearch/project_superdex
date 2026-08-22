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

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/path.h>
#include <mochi_physics/src/mochi_articulated_actor_params.h>
#include <mochi_physics/src/mochi_context.h>
#include <mochi_physics/src/mochi_shape.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <gtest/gtest.h>

#include <string>
#include <string_view>

using namespace mochi;
using namespace mochi::test;

class ArticulatedActorParamsTest : public MochiContextTestBase {};

// ---------------------------------------------------------------------------
// Helper: Create a minimal valid ArticulatedActorParams (2 links, 2 joints)
// ---------------------------------------------------------------------------
static ArticulatedActorParams MakeValidParams() {
  ArticulatedActorParams params;
  params.name = "test_actor";

  params.links.resize(2);
  params.links[0].name = "link_0";
  params.links[0].parentLink = -1;
  params.links[1].name = "link_1";
  params.links[1].parentLink = 0;

  params.joints.resize(2);
  params.joints[0].name = "joint_0";
  params.joints[0].type = ArticulatedJointType::Revolute;
  params.joints[0].axis = Real3{1_r, 0_r, 0_r};
  params.joints[1].name = "joint_1";
  params.joints[1].type = ArticulatedJointType::Revolute;
  params.joints[1].axis = Real3{1_r, 0_r, 0_r};

  return params;
}

// ===========================================================================
// AutoCorrect
// ===========================================================================

TEST_F(ArticulatedActorParamsTest, AutoCorrect_ArticulatedActorParams_AllNamesEmpty) {
  auto params = MakeValidParams();
  params.links[0].name = {};
  params.links[1].name = {};
  params.joints[0].name = {};
  params.joints[1].name = {};

  AutoCorrect(params, ExpectOK{});

  EXPECT_EQ(params.links[0].name, "link_0");
  EXPECT_EQ(params.links[1].name, "link_1");
  EXPECT_EQ(params.joints[0].name, "joint_0");
  EXPECT_EQ(params.joints[1].name, "joint_1");
}

TEST_F(ArticulatedActorParamsTest, AutoCorrect_ArticulatedActorParams_NameConflictAvoidance) {
  auto params = MakeValidParams();
  params.links.push_back({});
  params.links[2].parentLink = 0;
  params.joints.push_back({});
  params.joints[2].type = ArticulatedJointType::Revolute;
  params.joints[2].axis = Real3{1_r, 0_r, 0_r};

  // link[0] named "link_1" conflicts with what auto-gen would produce for index 1
  params.links[0].name = "link_1";
  params.links[1].name = {};
  params.links[2].name = {};
  params.joints[0].name = "joint_1";
  params.joints[1].name = {};
  params.joints[2].name = {};

  AutoCorrect(params, ExpectOK{});

  EXPECT_EQ(params.links[0].name, "link_1");
  EXPECT_EQ(params.links[1].name, "link_0");
  EXPECT_EQ(params.links[2].name, "link_2");
  EXPECT_EQ(params.joints[0].name, "joint_1");
  EXPECT_EQ(params.joints[1].name, "joint_0");
  EXPECT_EQ(params.joints[2].name, "joint_2");
}

TEST_F(ArticulatedActorParamsTest, AutoCorrect_ArticulatedJointParams_AxisNormalization) {
  ArticulatedJointParams joint;
  joint.type = ArticulatedJointType::Revolute;
  joint.axis = Real3{1_r, 2_r, 3_r};
  AutoCorrect(joint, ExpectOK{});
  EXPECT_NEAR_EQ(Normalize(Real3(1_r, 2_r, 3_r)), joint.axis);
}

TEST_F(ArticulatedActorParamsTest, AutoCorrect_ArticulatedActorParams_TransformNormalization) {
  // TransformRT logs a warning if the Quaternion magnitude is too far from 1.
  // Suppress the warning so we don't fail the test.
  auto prevLogFn = GetLogCallback();
  SetLogCallback(
      [](LogChannel /*channel*/, char const* /*message*/, char const* /*file*/, int /*line*/) {});
  MOCHI_DEFER(SetLogCallback(prevLogFn));

  Quaternion const q[] = {
      Quaternion(0_r, 0_r, 0_r, 2_r),
      Quaternion(0_r, 0_r, 0_r, 3_r),
      Quaternion(0_r, 0_r, 0_r, 4_r)};

  auto params = MakeValidParams();
  params.worldFromRoot = TransformRT(q[0], Real3{1_r, 2_r, 3_r});
  params.joints[0].parentLinkFromJoint = TransformRT(q[1]);
  params.links[1].parentJointFromLink = TransformRT(q[2]);

  AutoCorrect(params, ExpectOK{});

  EXPECT_NEAR_EQ(Normalize(q[0]), params.worldFromRoot.GetRotation());
  EXPECT_NEAR_EQ(Normalize(q[1]), params.joints[0].parentLinkFromJoint.GetRotation());
  EXPECT_NEAR_EQ(Normalize(q[2]), params.links[1].parentJointFromLink.GetRotation());
}

TEST_F(ArticulatedActorParamsTest, AutoCorrect_ArticulatedJointParams_ClearUnusedJointParams) {
  for (int iType = 0; iType < static_cast<int>(ArticulatedJointType::Count); ++iType) {
    auto jointType = static_cast<ArticulatedJointType>(iType);
    if (jointType == ArticulatedJointType::Cycle) {
      // This is not a local joint type. Skip it.
      continue;
    }

    // Set the type of the first joint
    ArticulatedJointParams joint;
    joint.type = jointType;

    // Set the joint parameters
    joint.axis = Real3{2_r, 0_r, 0_r};
    joint.minLimit = Real3{-1_r, -2_r, -3_r};
    joint.maxLimit = Real3{4_r, 5_r, 6_r};
    joint.inertia = 0.123_r;
    joint.friction.viscous = 0.234_r;
    joint.limitDamping = 0.456_r;
    joint.limitStiffness = 0.567_r;

    // Auto-correct
    AutoCorrect(joint, ExpectOK{});

    // Some joint types do not use all fields
    if (jointType == ArticulatedJointType::Free || jointType == ArticulatedJointType::Hard) {
      EXPECT_EQ(joint.axis, Real3{}); // cleared
      EXPECT_FALSE(joint.minLimit.has_value()); // cleared
      EXPECT_FALSE(joint.maxLimit.has_value()); // cleared
      EXPECT_FALSE(joint.inertia.has_value()); // cleared
      EXPECT_EQ(joint.friction, ArticulatedJointFrictionParams{}); // cleared
      EXPECT_EQ(ArticulatedJointParams{}.limitDamping, joint.limitDamping); // reset to default
      EXPECT_EQ(ArticulatedJointParams{}.limitStiffness, joint.limitStiffness); // reset to default
    } else {
      if (jointType == ArticulatedJointType::Spherical) {
        EXPECT_EQ(Real3{}, joint.axis); // cleared
      } else {
        EXPECT_EQ(Real3(1_r, 0_r, 0_r), joint.axis); // normalized
      }
      EXPECT_EQ(Real3(-1_r, -2_r, -3_r), joint.minLimit.value_or(Real3{})); // no change
      EXPECT_EQ(Real3(4_r, 5_r, 6_r), joint.maxLimit.value_or(Real3{})); // no change
      EXPECT_EQ(0.123_r, joint.inertia.value_or(real{})); // no change
      EXPECT_EQ(0.234_r, joint.friction.viscous); // no change
      EXPECT_EQ(0.456_r, joint.limitDamping); // no change
      EXPECT_EQ(0.567_r, joint.limitStiffness); // no change
    }

    static_assert(
        static_cast<int>(ArticulatedJointType::Count) == 6,
        "Please update this code if joint types are added or removed.");
  }
}

TEST_F(ArticulatedActorParamsTest, AutoCorrect_ArticulatedJointParams_Limits) {
  ArticulatedJointParams joint;
  joint.type = ArticulatedJointType::Revolute;
  joint.axis = Real3{1_r, 0_r, 0_r};

  // +/- infinity means "no limits". AutoCorrect should clear these optional fields because JSON
  // does not allow non-finite values.
  joint.minLimit = -kInf3;
  joint.maxLimit = kInf3;
  AutoCorrect(joint, ExpectOK{});
  EXPECT_FALSE(joint.minLimit.has_value());
  EXPECT_FALSE(joint.maxLimit.has_value());

  auto constexpr kMin = Real3{-1_r, -1_r, -1_r};
  auto constexpr kMax = Real3{1_r, 1_r, 1_r};

  // For revolute and prismatic joints, a non-finite dot product will result in the limit being
  // discarded.
  for (auto type : {ArticulatedJointType::Revolute, ArticulatedJointType::Prismatic}) {
    for (int i = 0; i < 3; ++i) {
      joint.type = type;
      joint.minLimit = kMin;
      joint.maxLimit = kMax;
      (*joint.minLimit)[i] = -kInf;
      AutoCorrect(joint, ExpectOK{});
      EXPECT_FALSE(joint.minLimit.has_value());
      EXPECT_TRUE(joint.maxLimit.has_value());
      joint.minLimit = kMin;
      (*joint.maxLimit)[i] = kInf;
      AutoCorrect(joint, ExpectOK{});
      EXPECT_TRUE(joint.minLimit.has_value());
      EXPECT_FALSE(joint.maxLimit.has_value());
    }
  }

  // For a spherical joint, the limits are only discarded if they are -kInf3 and +kInf3 respectively
  {
    joint.type = ArticulatedJointType::Spherical;
    joint.minLimit = -kInf3;
    joint.maxLimit = kInf3;
    AutoCorrect(joint, ExpectOK{});
    EXPECT_FALSE(joint.minLimit.has_value());
    EXPECT_FALSE(joint.maxLimit.has_value());

    // Unlinke other joint types, sphereical joints can have a mix of finite and infinite
    // values. However the infinite values will be replaced with huge finite values to
    // make JSON happy.
    for (int i = 0; i < 3; ++i) {
      joint.minLimit = kMin;
      joint.maxLimit = kMax;
      (*joint.minLimit)[i] = -kInf;
      (*joint.maxLimit)[i] = kInf;
      AutoCorrect(joint, ExpectOK{});
      EXPECT_TRUE(IsFinite((*joint.minLimit)[i]));
      EXPECT_TRUE(IsFinite((*joint.maxLimit)[i]));
      EXPECT_LT((*joint.minLimit)[i], -1e38_r);
      EXPECT_GT((*joint.maxLimit)[i], 1e38_r);
    }
  }
}

// ===========================================================================
// Validate
// ===========================================================================

TEST_F(ArticulatedActorParamsTest, Validate_ArticulatedActorParams_NoLinks) {
  ArticulatedActorParams params;
  Validate(params, ExpectNotOK{});
}

TEST_F(ArticulatedActorParamsTest, Validate_ArticulatedActorParams_LinkJointSizeMismatch) {
  auto params = MakeValidParams();
  params.joints.resize(1);
  Validate(params, ExpectNotOK{});
}

TEST_F(ArticulatedActorParamsTest, Validate_ArticulatedActorParams_OutOfRangeParentIndex) {
  auto params = MakeValidParams();
  params.links[1].parentLink = 99;
  Validate(params, ExpectNotOK{});
}

TEST_F(ArticulatedActorParamsTest, Validate_ArticulatedActorParams_ParentNotBeforeChild) {
  auto params = MakeValidParams();
  params.links[1].parentLink = 1;
  Validate(params, ExpectNotOK{});
}

TEST_F(ArticulatedActorParamsTest, Validate_ArticulatedActorParams_OutOfRangeCycleIndices) {
  auto params = MakeValidParams();
  ArticulatedCycleJointParams cycle;
  cycle.childLink = 99;
  cycle.parentLink = 0;
  params.cycles.push_back(cycle);
  Validate(params, ExpectNotOK{});
}

TEST_F(ArticulatedActorParamsTest, Validate_ArticulatedActorParams_DuplicateLinkNames) {
  auto params = MakeValidParams();
  params.links[1].name = params.links[0].name;
  Validate(params, ExpectNotOK{});
}

TEST_F(ArticulatedActorParamsTest, Validate_ArticulatedActorParams_LinkNameInvalidCharacters) {
  auto const expectRejected = [](std::string_view name) {
    auto params = MakeValidParams();
    params.links[0].name = DynamicString(name);
    Validate(params, ExpectNotOK{});
  };

  expectRejected("parent/child");
  expectRejected("parent\\child");

  std::string const nameWithNull{"parent\0child", 12};
  expectRejected(std::string_view{nameWithNull.data(), nameWithNull.size()});
}

TEST_F(ArticulatedActorParamsTest, Validate_ArticulatedActorParams_DuplicateJointNames) {
  auto params = MakeValidParams();
  params.joints[1].name = params.joints[0].name;
  Validate(params, ExpectNotOK{});
}

TEST_F(ArticulatedActorParamsTest, Validate_ArticulatedActorParams_CycleJointSelfLoop) {
  auto params = MakeValidParams();
  ArticulatedCycleJointParams cycle;
  cycle.childLink = 0;
  cycle.parentLink = 0;
  params.cycles.push_back(cycle);
  Validate(params, ExpectNotOK{});
}

TEST_F(ArticulatedActorParamsTest, Validate_ArticulatedActorParams_CycleJointType) {
  auto params = MakeValidParams();
  params.joints[0].type = ArticulatedJointType::Cycle;
  Validate(params, ExpectNotOK{});
}

TEST_F(ArticulatedActorParamsTest, Validate_ArticulatedActorParams_ValidParams) {
  auto params = MakeValidParams();
  Validate(params, ExpectOK{});
}

TEST_F(ArticulatedActorParamsTest, Validate_ArticulatedActorParams_InvalidMomentOfInertia) {
  auto params = MakeValidParams();
  params.links[1].shape = _mochiContext->CreatePlaneShape(Real3{0_r, 1_r, 0_r}, 0_r, ExpectOK{});
  params.links[1].mass = 5_r;
  params.links[1].centerOfMass = Real3{};

  // Articulated params validation: finite but physically invalid MOI warns and stays valid.
  params.links[1].momentOfInertia = Real6{1_r, 0_r, 0_r, 1_r, 0_r, 3_r};
  {
    ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
    Validate(params, ExpectOK{});
  }

  // Articulated params validation: non-finite MOI remains an error.
  params.links[1].momentOfInertia = Real6{1_r, 0_r, 0_r, 1_r, 0_r, kInf};
  Validate(params, ExpectNotOK{});
}
