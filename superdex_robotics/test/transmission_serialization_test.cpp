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
#include <mochi_core/utils/file_utils.h>

#include <filesystem>
#include <string>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

// Helper to create a link with a name and parent index.
BotLinkPrefab MakeLink(char const* name, int parentLink) {
  BotLinkPrefab link;
  link.name = name;
  link.parentLink = parentLink;
  return link;
}

// Helper to create a hard (fixed) joint with the given name.
BotJointPrefab MakeHardJoint(char const* name) {
  BotJointPrefab joint;
  joint.name = name;
  joint.type = ArticulatedJointType::Hard;
  return joint;
}

// Helper to create a revolute joint with the given name.
BotJointPrefab MakeRevoluteJoint(char const* name) {
  BotJointPrefab joint;
  joint.name = name;
  joint.type = ArticulatedJointType::Revolute;
  joint.axis = {0_r, 0_r, 1_r};
  return joint;
}

// A minimal, structurally-valid bot (root + two revolute children) carrying one
// linear transmission that spans both child joints. Used to exercise the bot's
// JSON save/load path through the public API.
BotPrefab MakeBotWithTransmission() {
  BotPrefab bp;
  bp.name = "transmission_bot";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("a_link", 0));
  bp.links.push_back(MakeLink("b_link", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("a_joint"));
  bp.joints.push_back(MakeRevoluteJoint("b_joint"));
  bp.defaultPose = {0_r, 0_r};

  BotLinearTransmissionPrefab t;
  t.name = "t0";
  t.jointIndices = {1, 2};
  t.jointCoefficients = {0.01_r, -0.02_r};
  t.jointAxisDisps = {0_r, 0_r};
  t.targetDisplacement = 0.003_r;
  t.stiffness = 1234_r;
  t.damping = 5_r;
  t.allowCompressiveForce = true;
  bp.linearTransmissions.push_back(std::move(t));
  return bp;
}

// A minimal bot (root + two revolute children) carrying one cycle joint that
// closes a loop between the two children. Used to exercise JSON round-tripping of
// the cycles field.
BotPrefab MakeBotWithCycleJoint() {
  BotPrefab bp;
  bp.name = "cycle_bot";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("a_link", 0));
  bp.links.push_back(MakeLink("b_link", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("a_joint"));
  bp.joints.push_back(MakeRevoluteJoint("b_joint"));
  bp.defaultPose = {0_r, 0_r};

  ArticulatedCycleJointParams cycle;
  cycle.parentLink = 1; // a_link
  cycle.childLink = 2; // b_link
  cycle.jointFromChildLink = TransformRT(Real3{0.1_r, 0.2_r, 0.3_r});
  cycle.stiffness = 12345_r;
  bp.cycles.push_back(cycle);
  return bp;
}

// A minimal bot carrying one spatial tendon with two waypoints and one linear joint element.
BotPrefab MakeBotWithSpatialTendon() {
  BotPrefab bp;
  bp.name = "spatial_tendon_bot";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("a_link", 0));
  bp.links.push_back(MakeLink("b_link", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("a_joint"));
  bp.joints.push_back(MakeRevoluteJoint("b_joint"));
  bp.defaultPose = {0_r, 0_r};

  BotSpatialTendonPrefab s;
  s.name = "s0";
  RoutingElement wp0;
  wp0.type = RoutingElementType::Waypoint;
  wp0.index = 1;
  wp0.localPosition = {0_r, 0_r, 0_r};
  RoutingElement fj;
  fj.type = RoutingElementType::LinearJoint;
  fj.index = 1;
  fj.coefficient = 0.02_r;
  RoutingElement wp1;
  wp1.type = RoutingElementType::Waypoint;
  wp1.index = 2;
  wp1.localPosition = {0.1_r, 0_r, 0_r};
  s.routingElements = {wp0, fj, wp1};
  s.targetDisplacement = 0.005_r;
  s.stiffness = 2345_r;
  s.damping = 6_r;
  s.allowCompressiveForce = false;
  bp.spatialTendons.push_back(std::move(s));
  return bp;
}

// Create a unique temp dir, suffixed with the precision so the single- and
// double-precision test binaries do not collide on the same path.
TempDirCleanup MakeTempDir(std::string_view label) {
  constexpr std::string_view kPrecision = std::is_same_v<real, double> ? "double" : "float";
  return mochi::CreateTempDirectory(std::string(label) + "_" + std::string(kPrecision), ExpectOK{});
}

} // namespace

// A transmission survives a save + load round trip through the public bot API,
// including the new allowCompressiveForce actuator option.
TEST(TransmissionSerializationTest, RoundTripPreservesTransmissionFields) {
  auto const tempDir = MakeTempDir("transmission_roundtrip");
  auto const path = (tempDir.Path() / "bot.superdex_bot").string();

  SaveToFile(MakeBotWithTransmission(), path, ExpectOK{});
  BotPrefab const loaded = LoadBotPrefabFromFile(path, ExpectOK{});

  ASSERT_EQ(isize(loaded.linearTransmissions), 1);
  auto const& t = loaded.linearTransmissions[0];
  EXPECT_EQ(t.name, "t0");

  DynamicArray<int> const expectedIndices = {1, 2};
  EXPECT_EQ(t.jointIndices, expectedIndices);

  ASSERT_EQ(isize(t.jointCoefficients), 2);
  EXPECT_NEAR(static_cast<double>(t.jointCoefficients[0]), 0.01, 1e-6);
  EXPECT_NEAR(static_cast<double>(t.jointCoefficients[1]), -0.02, 1e-6);

  EXPECT_NEAR(static_cast<double>(t.targetDisplacement), 0.003, 1e-6);
  EXPECT_NEAR(static_cast<double>(t.stiffness), 1234.0, 1e-3);
  EXPECT_NEAR(static_cast<double>(t.damping), 5.0, 1e-6);
  EXPECT_TRUE(t.allowCompressiveForce);
}

// Saving emits the new field keys, not the legacy "tendon" aliases.
TEST(TransmissionSerializationTest, SaveEmitsNewKeysNotLegacyAliases) {
  auto const tempDir = MakeTempDir("transmission_savekeys");
  auto const path = (tempDir.Path() / "bot.superdex_bot").string();

  SaveToFile(MakeBotWithTransmission(), path, ExpectOK{});
  std::string const json = ReadFileString(path, ExpectOK{});

  EXPECT_NE(json.find("linearTransmissions"), std::string::npos);
  EXPECT_NE(json.find("jointCoefficients"), std::string::npos);
  EXPECT_NE(json.find("allowCompressiveForce"), std::string::npos);

  EXPECT_EQ(json.find("fixedTendons"), std::string::npos);
  EXPECT_EQ(json.find("jointRadii"), std::string::npos);
  EXPECT_EQ(json.find("jointTendonAlignmentFlags"), std::string::npos);
  EXPECT_EQ(json.find("jointAlignmentFlags"), std::string::npos);
}

// Spatial tendon survives save + load round trip.
TEST(TransmissionSerializationTest, SpatialTendonRoundTripPreservesFields) {
  auto const tempDir = MakeTempDir("spatial_tendon_roundtrip");
  auto const path = (tempDir.Path() / "bot.superdex_bot").string();

  SaveToFile(MakeBotWithSpatialTendon(), path, ExpectOK{});
  BotPrefab const loaded = LoadBotPrefabFromFile(path, ExpectOK{});

  ASSERT_EQ(isize(loaded.spatialTendons), 1);
  auto const& s = loaded.spatialTendons[0];
  EXPECT_EQ(s.name, "s0");

  ASSERT_EQ(isize(s.routingElements), 3);
  EXPECT_EQ(s.routingElements[0].type, RoutingElementType::Waypoint);
  EXPECT_EQ(s.routingElements[0].index, 1);
  EXPECT_EQ(s.routingElements[1].type, RoutingElementType::LinearJoint);
  EXPECT_EQ(s.routingElements[1].index, 1);
  EXPECT_NEAR(static_cast<double>(s.routingElements[1].coefficient), 0.02, 1e-6);
  EXPECT_EQ(s.routingElements[2].type, RoutingElementType::Waypoint);
  EXPECT_EQ(s.routingElements[2].index, 2);

  EXPECT_NEAR(static_cast<double>(s.targetDisplacement), 0.005, 1e-6);
  EXPECT_NEAR(static_cast<double>(s.stiffness), 2345.0, 1e-3);
  EXPECT_NEAR(static_cast<double>(s.damping), 6.0, 1e-6);
  EXPECT_FALSE(s.allowCompressiveForce);
}

// Saving emits spatialTendons key and routingElements.
TEST(TransmissionSerializationTest, SpatialTendonSaveEmitsKeys) {
  auto const tempDir = MakeTempDir("spatial_tendon_savekeys");
  auto const path = (tempDir.Path() / "bot.superdex_bot").string();

  SaveToFile(MakeBotWithSpatialTendon(), path, ExpectOK{});
  std::string const json = ReadFileString(path, ExpectOK{});

  EXPECT_NE(json.find("spatialTendons"), std::string::npos);
  EXPECT_NE(json.find("routingElements"), std::string::npos);
  // Waypoint is the default RoutingElementType value and is omitted by
  // NoSerializeDefaults; LinearJoint is non-default and should appear.
  EXPECT_NE(json.find("LinearJoint"), std::string::npos);
  // Verify waypoint data is serialized via localPosition field even though type is omitted.
  EXPECT_NE(json.find("localPosition"), std::string::npos);
}

// A cycle joint survives a save + load round trip through the public bot API.
TEST(TransmissionSerializationTest, CycleJointRoundTripPreservesFields) {
  auto const tempDir = MakeTempDir("cycle_roundtrip");
  auto const path = (tempDir.Path() / "bot.superdex_bot").string();

  SaveToFile(MakeBotWithCycleJoint(), path, ExpectOK{});
  BotPrefab const loaded = LoadBotPrefabFromFile(path, ExpectOK{});

  ASSERT_EQ(isize(loaded.cycles), 1);
  auto const& c = loaded.cycles[0];
  // Indices reference the same named links after load (root/a_link/b_link is
  // already sorted, so indices are stable).
  EXPECT_EQ(loaded.links[c.parentLink].name, "a_link");
  EXPECT_EQ(loaded.links[c.childLink].name, "b_link");
  EXPECT_NEAR_EQ(c.jointFromChildLink.GetTranslation(), (Real3{0.1_r, 0.2_r, 0.3_r}));
  EXPECT_NEAR(static_cast<double>(c.stiffness), 12345.0, 1e-2);
}

// Saving emits the cycles key when a cycle joint is present.
TEST(TransmissionSerializationTest, CycleJointSaveEmitsKey) {
  auto const tempDir = MakeTempDir("cycle_savekeys");
  auto const path = (tempDir.Path() / "bot.superdex_bot").string();

  SaveToFile(MakeBotWithCycleJoint(), path, ExpectOK{});
  std::string const json = ReadFileString(path, ExpectOK{});

  EXPECT_NE(json.find("cycles"), std::string::npos);
}
