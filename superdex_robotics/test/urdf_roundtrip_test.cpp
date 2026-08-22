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

#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/test/mochi_test_helpers.h>

#include <algorithm>
#include <filesystem>
#include <string>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

static std::string GetFR3UrdfPath() {
  return GetAssetsDir() + "test/urdf/fr3v2_1_urdf/robots/fr3v2_1_franka_hand.urdf";
}

// URDF has no separate joint frame — a joint's <origin> is the whole parent-link -> child-link
// transform and its <axis>/limits live in the child link frame (urdfdom: "child link frame is the
// same as the Joint frame"). A round-tripped bot therefore always comes back with
// parentJointFromLink == identity and those quantities folded into parentLinkFromJoint. Reproduce
// that folding so two bots can be compared on their frame-independent physical articulation
// regardless of how each one split the joint/link frames. Folding an already-folded (identity) bot
// is a no-op, so applying it to both sides is always safe.
static BotPrefab FoldJointFramesForUrdf(BotPrefab bot) {
  int const n = std::min(isize(bot.joints), isize(bot.links));
  for (int i = 0; i < n; ++i) {
    BotJointPrefab& joint = bot.joints[i];
    BotLinkPrefab& link = bot.links[i];
    Quaternion const jointToChild = link.parentJointFromLink.GetRotation().GetConjugate();
    joint.parentLinkFromJoint = joint.parentLinkFromJoint * link.parentJointFromLink;
    joint.axis = jointToChild * joint.axis;
    if (joint.minLimit) {
      joint.minLimit = jointToChild * *joint.minLimit;
    }
    if (joint.maxLimit) {
      joint.maxLimit = jointToChild * *joint.maxLimit;
    }
    link.parentJointFromLink = TransformRT::Identity();
  }
  return bot;
}

static void
CompareBotPrefab(BotPrefab const& aRaw, BotPrefab const& bRaw, real tolerance, int startJoint) {
  BotPrefab const a = FoldJointFramesForUrdf(aRaw);
  BotPrefab const b = FoldJointFramesForUrdf(bRaw);
  ASSERT_EQ(std::string_view(a.name), std::string_view(b.name));
  ASSERT_EQ(isize(a.links), isize(b.links));
  ASSERT_EQ(isize(a.joints), isize(b.joints));

  for (int i = 0; i < isize(a.links); ++i) {
    auto const& la = a.links[i];
    auto const& lb = b.links[i];
    EXPECT_EQ(std::string_view(la.name), std::string_view(lb.name)) << "link " << i;
    EXPECT_EQ(la.parentLink, lb.parentLink) << "link " << i;

    // Mass properties
    EXPECT_EQ(la.mass.has_value(), lb.mass.has_value()) << "link " << i;
    if (la.mass && lb.mass) {
      EXPECT_NEAR(*la.mass, *lb.mass, tolerance) << "link " << i;
    }
    EXPECT_EQ(la.centerOfMass.has_value(), lb.centerOfMass.has_value()) << "link " << i;
    if (la.centerOfMass && lb.centerOfMass) {
      EXPECT_NEAR_TOL(*la.centerOfMass, *lb.centerOfMass, tolerance);
    }
    EXPECT_EQ(la.momentOfInertia.has_value(), lb.momentOfInertia.has_value()) << "link " << i;
    if (la.momentOfInertia && lb.momentOfInertia) {
      EXPECT_NEAR_TOL(*la.momentOfInertia, *lb.momentOfInertia, tolerance);
    }
  }

  for (int i = startJoint; i < isize(a.joints); ++i) {
    auto const& ja = a.joints[i];
    auto const& jb = b.joints[i];
    EXPECT_EQ(std::string_view(ja.name), std::string_view(jb.name)) << "joint " << i;
    EXPECT_EQ(ja.type, jb.type) << "joint " << i;

    // Axis
    EXPECT_NEAR_TOL(ja.axis, jb.axis, tolerance);

    // Limits
    auto aMinLimit = ja.minLimit.value_or(-kInf3);
    auto aMaxLimit = ja.maxLimit.value_or(kInf3);
    auto bMinLimit = jb.minLimit.value_or(-kInf3);
    auto bMaxLimit = jb.maxLimit.value_or(kInf3);
    for (int c = 0; c < 3; ++c) {
      if (IsFinite(aMinLimit[c])) {
        EXPECT_TRUE(IsFinite(bMinLimit[c]));
        EXPECT_NEAR_TOL(aMinLimit[c], bMinLimit[c], tolerance);

      } else {
        EXPECT_FALSE(IsFinite(bMinLimit[c]));
      }
      if (IsFinite(aMaxLimit[c])) {
        EXPECT_TRUE(IsFinite(bMaxLimit[c]));
        EXPECT_NEAR_TOL(aMaxLimit[c], bMaxLimit[c], tolerance);

      } else {
        EXPECT_FALSE(IsFinite(bMaxLimit[c]));
      }
    }

    // Transform (use EquivalentRotation to handle q/-q sign ambiguity from RPY round-trip)
    EXPECT_TRUE(EquivalentRotation(
        ja.parentLinkFromJoint.GetRotation(), jb.parentLinkFromJoint.GetRotation(), tolerance))
        << "joint " << i;
    EXPECT_NEAR_TOL(
        ja.parentLinkFromJoint.GetTranslation(),
        jb.parentLinkFromJoint.GetTranslation(),
        tolerance);

    // Viscous Friction
    EXPECT_NEAR_TOL(ja.friction.viscous, jb.friction.viscous, tolerance);

    // Coulomb Friction
    EXPECT_NEAR_TOL(ja.friction.coulomb, jb.friction.coulomb, tolerance);

    // Effort limit (verbatim: < 0 unbounded, 0 non-actuated, > 0 finite; round-trips through URDF)
    EXPECT_NEAR(ja.effortLimit, jb.effortLimit, tolerance) << "joint " << i;
  }
}

// Load the FR3 URDF → BotPrefab → export to URDF string → re-import → compare BotPrefab
TEST(UrdfRoundtrip, FR3RoundTrip) {
  std::string const urdfPath = GetFR3UrdfPath();
  ASSERT_TRUE(std::filesystem::exists(urdfPath)) << "FR3 URDF not found at: " << urdfPath;

  // Step 1: Load original URDF from file
  BotPrefab const original = LoadBotPrefabFromUrdfFile(urdfPath, ExpectOK{});

  // Step 2: Export to URDF string (no file I/O)
  DynamicString const exportedXml = SaveToUrdfString(original, ExpectOK{});
  ASSERT_FALSE(exportedXml.empty());

  // Step 3: Re-import from the exported string
  BotPrefab const reimported = LoadBotPrefabFromUrdfString(exportedXml, ExpectOK{});

  // Step 4: Compare original and reimported BotPrefab
  real const kTolerance = 1e-4_r;
  CompareBotPrefab(original, reimported, kTolerance, 0);

  // Step 5: Export again and re-import to verify idempotency
  DynamicString const reexportedXml = SaveToUrdfString(reimported, ExpectOK{});
  BotPrefab const reimported2 = LoadBotPrefabFromUrdfString(reexportedXml, ExpectOK{});
  CompareBotPrefab(reimported, reimported2, kTolerance, 0);
}

// A bot whose child link frame differs from its joint frame (non-identity parentJointFromLink) must
// still export losslessly. URDF fuses those frames, so the exporter folds
// parentLinkFromJoint * parentJointFromLink into the joint <origin> and re-expresses the axis in
// the child link frame; the re-imported bot carries an identity parentJointFromLink with the
// physical articulation preserved. This guards the export path against Studio-authored (or
// otherwise non-URDF-derived) bots that populate parentJointFromLink.
TEST(UrdfRoundtrip, NonIdentityParentJointFromLinkFolds) {
  BotPrefab bot;
  bot.name = "pjfl_bot";

  BotLinkPrefab base;
  base.name = "base";
  base.parentLink = kIndexNone;
  bot.links.push_back(std::move(base));

  // Child link with a non-identity joint -> child-link transform (rotation + translation).
  BotLinkPrefab child;
  child.name = "link1";
  child.parentLink = 0;
  child.parentJointFromLink = TransformRT{Quaternion::RotationX(0.7_r), Real3{0.1_r, 0.2_r, 0.3_r}};
  bot.links.push_back(std::move(child));

  BotJointPrefab worldJoint;
  worldJoint.name = "world_joint";
  worldJoint.type = ArticulatedJointType::Free;
  bot.joints.push_back(std::move(worldJoint));

  BotJointPrefab joint;
  joint.name = "j1";
  joint.type = ArticulatedJointType::Revolute;
  joint.parentLinkFromJoint = TransformRT{Quaternion::RotationZ(0.3_r), Real3{0.4_r, 0_r, 0.5_r}};
  joint.axis = Real3{0_r, 0_r, 1_r};
  joint.minLimit = joint.axis * -1_r;
  joint.maxLimit = joint.axis * 1_r;
  bot.joints.push_back(std::move(joint));

  DynamicString const xml = SaveToUrdfString(bot, ExpectOK{});
  BotPrefab const re = LoadBotPrefabFromUrdfString(xml, ExpectOK{});

  ASSERT_EQ(isize(re.joints), 2);
  ASSERT_EQ(isize(re.links), 2);

  real const kTolerance = 1e-4_r;

  // The re-imported child link frame is fused into the joint frame (identity parentJointFromLink).
  EXPECT_TRUE(EquivalentRotation(
      re.links[1].parentJointFromLink.GetRotation(), Quaternion::Identity(), kTolerance));
  EXPECT_NEAR_TOL(re.links[1].parentJointFromLink.GetTranslation(), Real3{}, kTolerance);

  // Origin == parentLinkFromJoint * parentJointFromLink (computed independently from the source).
  TransformRT const expectedOrigin =
      bot.joints[1].parentLinkFromJoint * bot.links[1].parentJointFromLink;
  EXPECT_TRUE(EquivalentRotation(
      re.joints[1].parentLinkFromJoint.GetRotation(), expectedOrigin.GetRotation(), kTolerance));
  EXPECT_NEAR_TOL(
      re.joints[1].parentLinkFromJoint.GetTranslation(),
      expectedOrigin.GetTranslation(),
      kTolerance);

  // Axis re-expressed in the child link frame.
  Real3 const expectedAxis =
      bot.links[1].parentJointFromLink.GetRotation().GetConjugate() * bot.joints[1].axis;
  EXPECT_NEAR_TOL(re.joints[1].axis, expectedAxis, kTolerance);
}

// A joint origin at +90 deg pitch (rotation about Y) drives the exporter's FormatRpy into its
// gimbal-lock branch, where yaw collapses into roll at the singularity. The import -> export ->
// re-import cycle must still preserve the rotation even though the exported roll/pitch/yaw
// decomposition differs from the input.
TEST(UrdfRoundtrip, GimbalLockPitchRoundTrip) {
  std::string const urdf =
      R"(<?xml version="1.0"?>)"
      R"(<robot name="gimbal_bot">)"
      R"(  <link name="base">)"
      R"(    <inertial>)"
      R"(      <origin xyz="0 0 0" rpy="0 0 0"/>)"
      R"(      <mass value="1.0"/>)"
      R"(      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>)"
      R"(    </inertial>)"
      R"(  </link>)"
      R"(  <link name="link1">)"
      R"(    <inertial>)"
      R"(      <origin xyz="0 0 0" rpy="0 0 0"/>)"
      R"(      <mass value="2.0"/>)"
      R"(      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.2"/>)"
      R"(    </inertial>)"
      R"(  </link>)"
      R"(  <joint name="j1" type="revolute">)"
      R"(    <parent link="base"/>)"
      R"(    <child link="link1"/>)"
      R"(    <origin xyz="0 0 0" rpy="0.3 1.5707963267948966 0.7"/>)"
      R"(    <axis xyz="0 0 1"/>)"
      R"(    <limit lower="-1" upper="1" effort="10" velocity="1"/>)"
      R"(  </joint>)"
      R"(</robot>)";

  BotPrefab const imported = LoadBotPrefabFromUrdfString(urdf, ExpectOK{});
  DynamicString const exportedXml = SaveToUrdfString(imported, ExpectOK{});
  BotPrefab const reimported = LoadBotPrefabFromUrdfString(exportedXml, ExpectOK{});

  real const kTolerance = 1e-4_r;
  CompareBotPrefab(imported, reimported, kTolerance, /*startJoint=*/1);
}

// Regression for a near-singular gimbal case that the exact-pi/2 test above does not cover. This
// quaternion (from t3_right joint 'dg5f_link_1_4_seed_mount_joint') sits at pitch ~= -pi/2 with
// cos(pitch) ~= 1.5e-8, and has the structure z == x, w == -y, which makes r21 = 2(yz + xw) and
// r10 = 2(xy + zw) vanish identically. The non-gimbal RPY extraction then recovers roll = yaw = 0,
// silently dropping the coupled in-plane rotation (the exported joint collapses to a pure Y
// rotation). FormatRpy must route this through the gimbal branch so the rotation round-trips.
TEST(UrdfRoundtrip, GimbalLockCoupledAngleRoundTrip) {
  BotPrefab bot;
  bot.name = "gimbal_regression";

  BotLinkPrefab base;
  base.name = "base";
  base.parentLink = kIndexNone;
  bot.links.push_back(std::move(base));

  BotLinkPrefab child;
  child.name = "link1";
  child.parentLink = 0;
  bot.links.push_back(std::move(child));

  BotJointPrefab worldJoint;
  worldJoint.name = "world_joint";
  worldJoint.type = ArticulatedJointType::Free;
  bot.joints.push_back(std::move(worldJoint));

  BotJointPrefab joint;
  joint.name = "j1";
  joint.type = ArticulatedJointType::Revolute;
  // XYZW order; z == x and w == -y produce the identically-zero r21/r10 cancellation.
  Quaternion const rot{
      -0.50396174192428589_r,
      -0.49600660800933838_r,
      -0.50396174192428589_r,
      0.49600660800933838_r};
  joint.parentLinkFromJoint = TransformRT{rot, Real3{}};
  joint.axis = Real3{0_r, 0_r, 1_r};
  joint.minLimit = joint.axis * -1_r;
  joint.maxLimit = joint.axis * 1_r;
  bot.joints.push_back(std::move(joint));

  DynamicString const xml = SaveToUrdfString(bot, ExpectOK{});
  BotPrefab const re = LoadBotPrefabFromUrdfString(xml, ExpectOK{});

  ASSERT_EQ(isize(re.joints), 2);
  EXPECT_TRUE(EquivalentRotation(
      bot.joints[1].parentLinkFromJoint.GetRotation(),
      re.joints[1].parentLinkFromJoint.GetRotation(),
      1e-4_r));
}

// A URDF <axis> is spec'd as a unit vector, but real files sometimes carry a non-unit axis. The
// importer scales limits by the axis (minLimit = axis * lower) while the exporter recovers them via
// Dot(limitVec, Normalize(axis)); normalizing the axis on import keeps that pair consistent so a
// non-unit axis round-trips losslessly instead of rescaling the limits on every cycle.
TEST(UrdfRoundtrip, NonUnitAxisNormalizesAndRoundTrips) {
  std::string const urdf = R"(<?xml version="1.0"?>)"
                           R"(<robot name="nonunit_axis">)"
                           R"(  <link name="base"/>)"
                           R"(  <link name="link1"/>)"
                           R"(  <joint name="j1" type="revolute">)"
                           R"(    <parent link="base"/>)"
                           R"(    <child link="link1"/>)"
                           R"(    <origin xyz="0 0 0" rpy="0 0 0"/>)"
                           R"(    <axis xyz="0 0 2"/>)"
                           R"(    <limit lower="-1" upper="1" effort="10" velocity="1"/>)"
                           R"(  </joint>)"
                           R"(</robot>)";

  BotPrefab const imported = LoadBotPrefabFromUrdfString(urdf, ExpectOK{});
  ASSERT_EQ(isize(imported.joints), 2);

  real const kTolerance = 1e-4_r;

  // The non-unit "0 0 2" axis is normalized to unit length on import.
  Real3 const expectedAxis{0_r, 0_r, 1_r};
  EXPECT_NEAR_TOL(imported.joints[1].axis, expectedAxis, kTolerance);

  // Scalar limits are preserved, not rescaled by the axis magnitude.
  Real3 const expectedMin{0_r, 0_r, -1_r};
  Real3 const expectedMax{0_r, 0_r, 1_r};
  ASSERT_TRUE(imported.joints[1].minLimit.has_value());
  ASSERT_TRUE(imported.joints[1].maxLimit.has_value());
  EXPECT_NEAR_TOL(*imported.joints[1].minLimit, expectedMin, kTolerance);
  EXPECT_NEAR_TOL(*imported.joints[1].maxLimit, expectedMax, kTolerance);

  // The full export -> re-import cycle is lossless.
  DynamicString const exportedXml = SaveToUrdfString(imported, ExpectOK{});
  BotPrefab const reimported = LoadBotPrefabFromUrdfString(exportedXml, ExpectOK{});
  CompareBotPrefab(imported, reimported, kTolerance, /*startJoint=*/1);
}

// The exporter must emit enough significant digits to round-trip a value. The stream default of 6
// significant figures silently truncated larger-magnitude quantities (origins, inertia, mass). A
// value with more than six significant digits must come back within float precision, not truncated.
TEST(UrdfRoundtrip, ExportPreservesSignificantDigits) {
  BotPrefab bot;
  bot.name = "precision_bot";

  BotLinkPrefab base;
  base.name = "base";
  base.parentLink = kIndexNone;
  bot.links.push_back(std::move(base));

  // A value with 9 significant digits at a magnitude where 6-sig-fig truncation (~0.2) dwarfs
  // float precision (~1e-4).
  real const kMass = 1234.56789_r;
  BotLinkPrefab child;
  child.name = "link1";
  child.parentLink = 0;
  child.mass = kMass;
  bot.links.push_back(std::move(child));

  BotJointPrefab worldJoint;
  worldJoint.name = "world_joint";
  worldJoint.type = ArticulatedJointType::Free;
  bot.joints.push_back(std::move(worldJoint));

  BotJointPrefab joint;
  joint.name = "j1";
  joint.type = ArticulatedJointType::Hard; // exported as "fixed"
  bot.joints.push_back(std::move(joint));

  DynamicString const xml = SaveToUrdfString(bot, ExpectOK{});
  BotPrefab const re = LoadBotPrefabFromUrdfString(xml, ExpectOK{});

  ASSERT_EQ(isize(re.links), 2);
  ASSERT_TRUE(re.links[1].mass.has_value());
  // Round-trips within float precision, far tighter than the ~0.2 error six significant figures
  // would introduce at this magnitude.
  EXPECT_NEAR(*re.links[1].mass, kMass, 1e-2_r);
}

// Verify that exporting a bot whose first link is not the root (has a parent joint) fails
// gracefully. URDF's root link has no parent joint, so the exporter skips joint 0 and requires
// link 0 to be the root; the joint's name is irrelevant to this check.
TEST(UrdfRoundtrip, ExportWithNonRootFirstLinkFails) {
  BotPrefab bp;
  bp.name = "test_bot";

  BotLinkPrefab rootLink;
  rootLink.name = "link_0";
  rootLink.parentLink = 1; // Not the root: points at link_1 as its parent.
  bp.links.push_back(std::move(rootLink));

  BotLinkPrefab childLink;
  childLink.name = "link_1";
  childLink.parentLink = kIndexNone;
  bp.links.push_back(std::move(childLink));

  BotJointPrefab joint0;
  joint0.name = "joint_0";
  joint0.type = ArticulatedJointType::Revolute;
  bp.joints.push_back(std::move(joint0));

  BotJointPrefab joint1;
  joint1.name = "joint_1";
  joint1.type = ArticulatedJointType::Revolute;
  bp.joints.push_back(std::move(joint1));

  (void)SaveToUrdfString(bp, ExpectNotOK{});
}

// Verify that exporting an empty bot fails
TEST(UrdfRoundtrip, ExportEmptyBotFails) {
  BotPrefab bp;
  (void)SaveToUrdfString(bp, ExpectNotOK{});
}

// Load every .superdex_bot → export to URDF string → re-import → compare BotParams
TEST(UrdfRoundtrip, AllBotsRoundTrip) {
  DynamicArray<std::filesystem::path> const botFiles = FindAllBotFiles();
  ASSERT_FALSE(botFiles.empty()) << "No .superdex_bot files found";

  real const kTolerance = 1e-4_r;

  for (auto const& path : botFiles) {
    SCOPED_TRACE(path);

    BotPrefab const original = LoadBotPrefabFromFile(path.string(), ExpectOK{});
    if (original.links.empty()) {
      continue;
    }

    // URDF has no spherical joint type, so the exporter rejects bots that use them.
    bool const hasSphericalJoint =
        std::any_of(original.joints.begin(), original.joints.end(), [](BotJointPrefab const& j) {
          return j.type == ArticulatedJointType::Spherical;
        });
    if (hasSphericalJoint) {
      (void)SaveToUrdfString(original, ExpectNotOK{});
      continue;
    }

    DynamicString const exportedXml = SaveToUrdfString(original, ExpectOK{});
    BotPrefab const reimported = LoadBotPrefabFromUrdfString(exportedXml, ExpectOK{});

    // The world_joint (joint 0) type is a mochi-ism that URDF doesn't carry; the importer
    // always injects it as Free. Some .superdex_bot files have a Hard world_joint (e.g., from
    // being edited in SuperDex Studio) while others have Free (from original URDF import or
    // by design). Compare starting from joint 1 to skip the world_joint.
    CompareBotPrefab(original, reimported, kTolerance, /*startJoint=*/1);
  }
}

// Import must reject malformed URDF at the parse boundary rather than silently producing corrupt
// bot data. Before the tinyxml2 rewrite, urdfdom performed this validation; these cases exercise
// the hand-rolled parser's error branches.
static void ExpectUrdfImportFails(std::string_view xml) {
  (void)LoadBotPrefabFromUrdfString(xml, ExpectNotOK{});
}

// Structural / attribute errors detected while parsing the XML document.
TEST(UrdfImportNegative, RejectsStructurallyInvalidUrdf) {
  // Root element is not <robot>.
  {
    SCOPED_TRACE("non-robot root");
    ExpectUrdfImportFails(R"(<notrobot name="r"/>)");
  }
  // <robot> is missing its name attribute.
  {
    SCOPED_TRACE("missing robot name");
    ExpectUrdfImportFails(R"(<robot></robot>)");
  }
  // Syntactically invalid XML.
  {
    SCOPED_TRACE("invalid xml");
    ExpectUrdfImportFails(R"(<robot name="r"><link)");
  }
  // <link> is missing its name attribute.
  {
    SCOPED_TRACE("link missing name");
    ExpectUrdfImportFails(R"(<robot name="r"><link/></robot>)");
  }
  // <joint> is missing its name attribute.
  {
    SCOPED_TRACE("joint missing name");
    ExpectUrdfImportFails(
        R"(<robot name="r"><link name="a"/><link name="b"/>)"
        R"(<joint type="fixed"><parent link="a"/><child link="b"/></joint></robot>)");
  }
  // <joint> is missing its type attribute.
  {
    SCOPED_TRACE("joint missing type");
    ExpectUrdfImportFails(
        R"(<robot name="r"><link name="a"/><link name="b"/>)"
        R"(<joint name="j"><parent link="a"/><child link="b"/></joint></robot>)");
  }
}

// Topology errors: the joint graph is not a single fully connected tree.
TEST(UrdfImportNegative, RejectsInvalidTopology) {
  // Joint references a link that does not exist.
  {
    SCOPED_TRACE("unknown link");
    ExpectUrdfImportFails(
        R"(<robot name="r"><link name="a"/><link name="b"/>)"
        R"(<joint name="j" type="fixed"><parent link="a"/><child link="c"/></joint></robot>)");
  }
  // Joint count is not one less than the link count.
  {
    SCOPED_TRACE("joint/link count mismatch");
    ExpectUrdfImportFails(R"(<robot name="r"><link name="a"/><link name="b"/></robot>)");
  }
  // Passes the count and single-root checks, but link "c" is unreachable and forms a cycle. This
  // is the disconnected-cycle case the reachability check must catch.
  {
    SCOPED_TRACE("disconnected cyclic component");
    ExpectUrdfImportFails(
        R"(<robot name="r"><link name="a"/><link name="b"/><link name="c"/>)"
        R"(<joint name="j1" type="fixed"><parent link="a"/><child link="b"/></joint>)"
        R"(<joint name="j2" type="fixed"><parent link="c"/><child link="c"/></joint></robot>)");
  }
  // Two links share the same name, which would silently collapse to a single index and mis-wire
  // the tree if not rejected.
  {
    SCOPED_TRACE("duplicate link names");
    ExpectUrdfImportFails(
        R"(<robot name="r"><link name="a"/><link name="a"/>)"
        R"(<joint name="j" type="fixed"><parent link="a"/><child link="a"/></joint></robot>)");
  }
}

// Malformed geometric attributes on otherwise well-formed URDF.
TEST(UrdfImportNegative, RejectsMalformedAttributes) {
  // Origin xyz has only two components instead of three.
  {
    SCOPED_TRACE("truncated vector attribute");
    ExpectUrdfImportFails(
        R"(<robot name="r"><link name="a"/><link name="b"/>)"
        R"(<joint name="j" type="fixed"><origin xyz="0 0" rpy="0 0 0"/>)"
        R"(<parent link="a"/><child link="b"/></joint></robot>)");
  }
  // Actuated joint has a degenerate zero axis.
  {
    SCOPED_TRACE("zero joint axis");
    ExpectUrdfImportFails(
        R"(<robot name="r"><link name="a"/><link name="b"/>)"
        R"(<joint name="j" type="revolute"><parent link="a"/><child link="b"/>)"
        R"(<axis xyz="0 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/>)"
        R"(</joint></robot>)");
  }
}
