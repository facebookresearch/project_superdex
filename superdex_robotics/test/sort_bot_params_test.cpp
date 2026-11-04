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
#include <superdex_robotics/core/loader.h>
#include <superdex_robotics/utils/bot_utils.h>

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

// Helper to create a revolute joint with the given name.
BotJointPrefab MakeRevoluteJoint(char const* name) {
  BotJointPrefab joint;
  joint.name = name;
  joint.type = ArticulatedJointType::Revolute;
  joint.axis = {0_r, 0_r, 1_r};
  joint.minLimit = {0_r, 0_r, -1_r};
  joint.maxLimit = {0_r, 0_r, 1_r};
  return joint;
}

// Helper to create a prismatic joint with the given name.
BotJointPrefab MakePrismaticJoint(char const* name) {
  BotJointPrefab joint;
  joint.name = name;
  joint.type = ArticulatedJointType::Prismatic;
  joint.axis = {0_r, 0_r, 1_r};
  return joint;
}

// Helper to create a spherical joint with the given name.
BotJointPrefab MakeSphericalJoint(char const* name) {
  BotJointPrefab joint;
  joint.name = name;
  joint.type = ArticulatedJointType::Spherical;
  return joint;
}

// Helper to create a hard (fixed) joint with the given name.
BotJointPrefab MakeHardJoint(char const* name) {
  BotJointPrefab joint;
  joint.name = name;
  joint.type = ArticulatedJointType::Hard;
  return joint;
}

// Collect link names into a vector for easy comparison.
DynamicArray<DynamicString> LinkNames(BotPrefab const& bp) {
  DynamicArray<DynamicString> names;
  for (auto const& link : bp.links) {
    names.push_back(link.name);
  }
  return names;
}

// Minimal IBotLoader that hands back a fixed, in-memory child prefab. Used to
// exercise the AttachBot path of ApplyMod without touching the filesystem.
struct InMemoryBotLoader : IBotLoader {
  BotPrefab child;
  // Optional second prefab, served when the requested path equals basePath. Lets a test compose a
  // recipe whose base and attached child differ; unset means every path resolves to `child`.
  DynamicString basePath;
  BotPrefab baseBot;

  BotFileType GetBotFileType(std::string_view /*path*/, Error& /*error*/) const override {
    return BotFileType::BotPrefab;
  }
  BotPrefab LoadBotPrefab(std::string_view path, Error& /*error*/) const override {
    return (!basePath.empty() && path == std::string_view(basePath)) ? baseBot : child;
  }
  ModBotPrefab LoadModBotPrefab(std::string_view /*path*/, Error& /*error*/) const override {
    return {};
  }
  ShapeHandle LoadShape(
      std::string_view /*path*/,
      Real3 const& /*bakeScale*/,
      TransformRT const& /*bakeTransform*/,
      Context* /*context*/,
      Error& /*error*/) const override {
    return {};
  }
};

// Test fixture that suppresses expected MOCHI_LOG_WARNING calls.
class SortBotPrefabTest : public testing::Test {
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

// Single link: nothing to sort, no error.
TEST_F(SortBotPrefabTest, SingleLink_NoOp) {
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  RebuildBotData(bp, ExpectOK{});
  EXPECT_EQ(bp.links[0].name, "root");
}

// Already sorted: data should remain unchanged.
TEST_F(SortBotPrefabTest, AlreadySorted_Unchanged) {
  // Tree: root -> A -> B
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("A", 0));
  bp.links.push_back(MakeLink("B", 1));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("A_joint"));
  bp.joints.push_back(MakeRevoluteJoint("B_joint"));
  bp.defaultPose = {0.1_r, 0.2_r};
  RebuildBotData(bp, ExpectOK{});
  DynamicArray<DynamicString> expected = {"root", "A", "B"};
  EXPECT_EQ(LinkNames(bp), expected);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[0]), 0.1, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[1]), 0.2, 1e-6);
}

// Reverse-ordered siblings: should be reordered alphabetically.
TEST_F(SortBotPrefabTest, ReverseSiblings_Sorted) {
  //   root
  //   ├── C (idx 1)
  //   ├── B (idx 2)
  //   └── A (idx 3)
  // Expected after sort: root, A, B, C
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("C", 0));
  bp.links.push_back(MakeLink("B", 0));
  bp.links.push_back(MakeLink("A", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("C_joint"));
  bp.joints.push_back(MakeRevoluteJoint("B_joint"));
  bp.joints.push_back(MakeRevoluteJoint("A_joint"));
  bp.defaultPose = {0.3_r, 0.2_r, 0.1_r};
  RebuildBotData(bp, ExpectOK{});
  DynamicArray<DynamicString> expected = {"root", "A", "B", "C"};
  EXPECT_EQ(LinkNames(bp), expected);
  // Joints follow their links
  EXPECT_EQ(bp.joints[1].name, "A_joint");
  EXPECT_EQ(bp.joints[2].name, "B_joint");
  EXPECT_EQ(bp.joints[3].name, "C_joint");
  // Default pose reordered: A=0.1, B=0.2, C=0.3
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[0]), 0.1, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[1]), 0.2, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[2]), 0.3, 1e-6);
}

// Alphanumeric (natural) sort: link2 < link10.
TEST_F(SortBotPrefabTest, AlphanumericSort_NaturalOrder) {
  //   root
  //   ├── link10 (idx 1)
  //   ├── link2  (idx 2)
  //   └── link1  (idx 3)
  // Expected: root, link1, link2, link10
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("link10", 0));
  bp.links.push_back(MakeLink("link2", 0));
  bp.links.push_back(MakeLink("link1", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("j10"));
  bp.joints.push_back(MakeRevoluteJoint("j2"));
  bp.joints.push_back(MakeRevoluteJoint("j1"));
  bp.defaultPose = {0.10_r, 0.02_r, 0.01_r};
  RebuildBotData(bp, ExpectOK{});
  DynamicArray<DynamicString> expected = {"root", "link1", "link2", "link10"};
  EXPECT_EQ(LinkNames(bp), expected);
  // Pose follows joints: link1=0.01, link2=0.02, link10=0.10
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[0]), 0.01, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[1]), 0.02, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[2]), 0.10, 1e-6);
}

// Multi-level tree: children sorted at every level.
TEST_F(SortBotPrefabTest, MultiLevelTree_SortedAtEveryLevel) {
  //   root
  //   ├── B (idx 1)
  //   │   ├── B2 (idx 2)
  //   │   └── B1 (idx 3)
  //   └── A (idx 4)
  //       └── A1 (idx 5)
  // Expected DFS order: root, A, A1, B, B1, B2
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("B", 0));
  bp.links.push_back(MakeLink("B2", 1));
  bp.links.push_back(MakeLink("B1", 1));
  bp.links.push_back(MakeLink("A", 0));
  bp.links.push_back(MakeLink("A1", 4));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("B_joint"));
  bp.joints.push_back(MakeRevoluteJoint("B2_joint"));
  bp.joints.push_back(MakeRevoluteJoint("B1_joint"));
  bp.joints.push_back(MakeRevoluteJoint("A_joint"));
  bp.joints.push_back(MakeRevoluteJoint("A1_joint"));
  bp.defaultPose = {0.1_r, 0.2_r, 0.3_r, 0.4_r, 0.5_r};
  RebuildBotData(bp, ExpectOK{});
  DynamicArray<DynamicString> expected = {"root", "A", "A1", "B", "B1", "B2"};
  EXPECT_EQ(LinkNames(bp), expected);
  // Parent indices should be valid after sort
  EXPECT_EQ(bp.links[0].parentLink, kIndexNone); // root
  EXPECT_EQ(bp.links[1].parentLink, 0); // A -> root
  EXPECT_EQ(bp.links[2].parentLink, 1); // A1 -> A
  EXPECT_EQ(bp.links[3].parentLink, 0); // B -> root
  EXPECT_EQ(bp.links[4].parentLink, 3); // B1 -> B
  EXPECT_EQ(bp.links[5].parentLink, 3); // B2 -> B
  // Pose: A=0.4, A1=0.5, B=0.1, B1=0.3, B2=0.2
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[0]), 0.4, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[1]), 0.5, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[2]), 0.1, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[3]), 0.3, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[4]), 0.2, 1e-6);
}

// Mixed joint types: hard joints produce no DOF, only revolute joints contribute to defaultPose.
TEST_F(SortBotPrefabTest, MixedJointTypes_PoseOnlyForDofs) {
  //   root (hard)
  //   ├── B (revolute) -> idx 1, old pose idx 0
  //   └── A (hard)     -> idx 2, no DOF
  // Expected order: root, A, B
  // Only B has a DOF, so defaultPose has 1 entry.
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("B", 0));
  bp.links.push_back(MakeLink("A", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("B_joint"));
  bp.joints.push_back(MakeHardJoint("A_joint"));
  bp.defaultPose = {0.5_r};
  RebuildBotData(bp, ExpectOK{});
  DynamicArray<DynamicString> expected = {"root", "A", "B"};
  EXPECT_EQ(LinkNames(bp), expected);
  ASSERT_EQ(isize(bp.defaultPose), 1);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[0]), 0.5, 1e-6);
}

// Idempotent: sorting an already-sorted bot twice yields the same result.
TEST_F(SortBotPrefabTest, Idempotent) {
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("C", 0));
  bp.links.push_back(MakeLink("A", 0));
  bp.links.push_back(MakeLink("B", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("C_joint"));
  bp.joints.push_back(MakeRevoluteJoint("A_joint"));
  bp.joints.push_back(MakeRevoluteJoint("B_joint"));
  bp.defaultPose = {0.3_r, 0.1_r, 0.2_r};
  RebuildBotData(bp, ExpectOK{});
  auto namesAfterFirst = LinkNames(bp);
  DynamicArray<real> poseAfterFirst = bp.defaultPose;
  // Sort again via another RebuildBotData call
  RebuildBotData(bp, ExpectOK{});
  EXPECT_EQ(LinkNames(bp), namesAfterFirst);
  ASSERT_EQ(bp.defaultPose.size(), poseAfterFirst.size());
  for (size_t i = 0; i < bp.defaultPose.size(); ++i) {
    EXPECT_NEAR(
        static_cast<double>(bp.defaultPose[i]), static_cast<double>(poseAfterFirst[i]), 1e-6);
  }
}

// Children indices are valid after sort + rebuild.
TEST_F(SortBotPrefabTest, ChildrenIndicesValid) {
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("C", 0));
  bp.links.push_back(MakeLink("A", 0));
  bp.links.push_back(MakeLink("B", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("C_joint"));
  bp.joints.push_back(MakeRevoluteJoint("A_joint"));
  bp.joints.push_back(MakeRevoluteJoint("B_joint"));
  RebuildBotData(bp, ExpectOK{});
  // After sort: root(0) -> A(1), B(2), C(3)
  ASSERT_EQ(isize(bp.links[0]._childrenIndices), 3);
  EXPECT_EQ(bp.links[0]._childrenIndices[0], 1); // A
  EXPECT_EQ(bp.links[0]._childrenIndices[1], 2); // B
  EXPECT_EQ(bp.links[0]._childrenIndices[2], 3); // C
}

// Spherical joint contributes 3 DOFs to _dofIndices and defaultPose.
TEST_F(SortBotPrefabTest, SphericalJoint_ContributesThreeDofs) {
  // Tree: root -> spherical -> child
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("a_sph", 0));
  bp.links.push_back(MakeLink("b_child", 1));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeSphericalJoint("a_sph_joint"));
  bp.joints.push_back(MakeHardJoint("b_child_joint"));
  RebuildBotData(bp, ExpectOK{});
  EXPECT_EQ(bp._numDofs, 3);
  ASSERT_EQ(isize(bp._dofIndices), 3);
  // Spherical joint is at link index 1
  EXPECT_EQ(bp._dofIndices[0], 1);
  EXPECT_EQ(bp._dofIndices[1], 1);
  EXPECT_EQ(bp._dofIndices[2], 1);
  ASSERT_EQ(isize(bp.defaultPose), 3);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[0]), 0.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[1]), 0.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[2]), 0.0, 1e-6);
}

// Mixed chain: revolute + spherical + prismatic interleave correctly in joint order.
TEST_F(SortBotPrefabTest, MixedJoints_InterleaveInJointOrder) {
  // Tree: root -> a_rev -> b_sph -> c_pri
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("a_rev", 0));
  bp.links.push_back(MakeLink("b_sph", 1));
  bp.links.push_back(MakeLink("c_pri", 2));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("a_rev_joint"));
  bp.joints.push_back(MakeSphericalJoint("b_sph_joint"));
  bp.joints.push_back(MakePrismaticJoint("c_pri_joint"));
  RebuildBotData(bp, ExpectOK{});
  EXPECT_EQ(bp._numDofs, 5);
  ASSERT_EQ(isize(bp._dofIndices), 5);
  // Revolute at link 1, spherical at link 2 (3x), prismatic at link 3
  EXPECT_EQ(bp._dofIndices[0], 1);
  EXPECT_EQ(bp._dofIndices[1], 2);
  EXPECT_EQ(bp._dofIndices[2], 2);
  EXPECT_EQ(bp._dofIndices[3], 2);
  EXPECT_EQ(bp._dofIndices[4], 3);
  EXPECT_EQ(isize(bp.defaultPose), 5);
}

// Spherical joint defaultPose values survive a SortBotPrefab+RebuildBotData round trip.
TEST_F(SortBotPrefabTest, SphericalDefaultPose_Preserved) {
  // Reverse-ordered siblings each carrying a spherical joint, so the sort
  // actually permutes them.
  //   root
  //   ├── C (idx 1, spherical)
  //   ├── B (idx 2, spherical)
  //   └── A (idx 3, spherical)
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("C", 0));
  bp.links.push_back(MakeLink("B", 0));
  bp.links.push_back(MakeLink("A", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeSphericalJoint("C_joint"));
  bp.joints.push_back(MakeSphericalJoint("B_joint"));
  bp.joints.push_back(MakeSphericalJoint("A_joint"));
  // C: (0.10, 0.11, 0.12), B: (0.20, 0.21, 0.22), A: (0.30, 0.31, 0.32)
  bp.defaultPose = {
      0.10_r,
      0.11_r,
      0.12_r,
      0.20_r,
      0.21_r,
      0.22_r,
      0.30_r,
      0.31_r,
      0.32_r,
  };
  RebuildBotData(bp, ExpectOK{});
  // After sort: root, A, B, C → defaultPose should be A's, B's, C's values.
  ASSERT_EQ(isize(bp.defaultPose), 9);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[0]), 0.30, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[1]), 0.31, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[2]), 0.32, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[3]), 0.20, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[4]), 0.21, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[5]), 0.22, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[6]), 0.10, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[7]), 0.11, 1e-6);
  EXPECT_NEAR(static_cast<double>(bp.defaultPose[8]), 0.12, 1e-6);
}

// RebuildBotData prunes contact overrides that reference links no longer present,
// while preserving overrides whose endpoints both still exist.
TEST_F(SortBotPrefabTest, RebuildBotData_PrunesStaleContactOverrides) {
  // Tree: root -> A, root -> B
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("A", 0));
  bp.links.push_back(MakeLink("B", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("A_joint"));
  bp.joints.push_back(MakeRevoluteJoint("B_joint"));
  bp.defaultPose = {0.0_r, 0.0_r};
  // One valid override (both links exist) and two stale ones (reference a link
  // name that is not in the prefab).
  bp.contactOverrides.push_back(BotContactOverride{"A", "B", true});
  bp.contactOverrides.push_back(BotContactOverride{"A", "ghost", true});
  bp.contactOverrides.push_back(BotContactOverride{"ghost", "B", false});
  RebuildBotData(bp, ExpectOK{});
  ASSERT_EQ(isize(bp.contactOverrides), 1);
  EXPECT_EQ(bp.contactOverrides[0].linkA, "A");
  EXPECT_EQ(bp.contactOverrides[0].linkB, "B");
  EXPECT_TRUE(bp.contactOverrides[0].enable);
}

// RebuildBotData keeps a skin<->link override while the skin is present (the skin is a valid
// override party, referenced by the bot's top-level actor name) and prunes it once the skin is
// removed.
TEST_F(SortBotPrefabTest, RebuildBotData_KeepsThenPrunesSkinOverride) {
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("A", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("A_joint"));
  bp.defaultPose = {0.0_r};
  bp.skin.emplace();
  // The skin is referenced by the bot name.
  bp.contactOverrides.push_back(BotContactOverride{bp.name, "A", true});
  RebuildBotData(bp, ExpectOK{});
  ASSERT_EQ(isize(bp.contactOverrides), 1);
  EXPECT_EQ(bp.contactOverrides[0].linkA, "test");
  EXPECT_EQ(bp.contactOverrides[0].linkB, "A");

  // Removing the skin makes the override reference an unknown party, so it is pruned.
  bp.skin.reset();
  RebuildBotData(bp, ExpectOK{});
  EXPECT_EQ(isize(bp.contactOverrides), 0);
}

// SortBotPrefab remaps cycle-joint link indices through the same permutation as
// links, so a cycle keeps referencing the same named links after a reorder.
TEST_F(SortBotPrefabTest, RebuildBotData_RemapsCycleLinkIndices) {
  //   root
  //   ├── C (idx 1)
  //   ├── B (idx 2)
  //   └── A (idx 3)
  // After sort: root(0), A(1), B(2), C(3). So old C:1->3, old A:3->1.
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("C", 0));
  bp.links.push_back(MakeLink("B", 0));
  bp.links.push_back(MakeLink("A", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("C_joint"));
  bp.joints.push_back(MakeRevoluteJoint("B_joint"));
  bp.joints.push_back(MakeRevoluteJoint("A_joint"));
  bp.defaultPose = {0.0_r, 0.0_r, 0.0_r};
  // Cycle between C (old idx 1) and A (old idx 3).
  ArticulatedCycleJointParams cycle;
  cycle.parentLink = 1; // C
  cycle.childLink = 3; // A
  bp.cycles.push_back(cycle);
  RebuildBotData(bp, ExpectOK{});
  // The cycle is preserved and its indices now point at the same named links.
  ASSERT_EQ(isize(bp.cycles), 1);
  EXPECT_EQ(bp.links[bp.cycles[0].parentLink].name, "C");
  EXPECT_EQ(bp.links[bp.cycles[0].childLink].name, "A");
}

// RebuildBotData does NOT prune cycle joints: invalid cycles (out of range or
// parent == child) are preserved so the user's edits are not silently discarded
// (e.g. transiently selecting the same link in the editor). Validity is enforced
// by Validate / actor creation, not by mutating the prefab here.
TEST_F(SortBotPrefabTest, RebuildBotData_PreservesInvalidCycles) {
  // Tree: root -> A, root -> B
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("A", 0));
  bp.links.push_back(MakeLink("B", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("A_joint"));
  bp.joints.push_back(MakeRevoluteJoint("B_joint"));
  bp.defaultPose = {0.0_r, 0.0_r};
  // A valid cycle plus a degenerate (parent == child) and an out-of-range cycle.
  ArticulatedCycleJointParams valid;
  valid.parentLink = 1; // A
  valid.childLink = 2; // B
  bp.cycles.push_back(valid);
  ArticulatedCycleJointParams degenerate;
  degenerate.parentLink = 1;
  degenerate.childLink = 1; // parent == child (must NOT disappear)
  bp.cycles.push_back(degenerate);
  ArticulatedCycleJointParams outOfRange;
  outOfRange.parentLink = 1;
  outOfRange.childLink = 99; // out of range
  bp.cycles.push_back(outOfRange);
  RebuildBotData(bp, ExpectOK{});
  // All three cycles are preserved (nothing pruned). The tree is already sorted
  // (root, A, B), so valid indices are unchanged.
  ASSERT_EQ(isize(bp.cycles), 3);
  EXPECT_EQ(bp.cycles[0].parentLink, 1);
  EXPECT_EQ(bp.cycles[0].childLink, 2);
  EXPECT_EQ(bp.cycles[1].parentLink, 1);
  EXPECT_EQ(bp.cycles[1].childLink, 1);
  EXPECT_EQ(bp.cycles[2].parentLink, 1);
  EXPECT_EQ(bp.cycles[2].childLink, 99);
}

TEST_F(SortBotPrefabTest, RemoveLinkAndDescendants_UpdatesCycles) {
  //   root
  //   |-- A
  //   |   `-- AChild
  //   |-- B
  //   `-- C
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("A", 0));
  bp.links.push_back(MakeLink("AChild", 1));
  bp.links.push_back(MakeLink("B", 0));
  bp.links.push_back(MakeLink("C", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("A_joint"));
  bp.joints.push_back(MakeRevoluteJoint("AChild_joint"));
  bp.joints.push_back(MakeRevoluteJoint("B_joint"));
  bp.joints.push_back(MakeRevoluteJoint("C_joint"));
  bp.defaultPose = {0_r, 0_r, 0_r, 0_r};

  ArticulatedCycleJointParams removedCycle;
  removedCycle.parentLink = 1; // A
  removedCycle.childLink = 4; // C
  bp.cycles.push_back(removedCycle);
  ArticulatedCycleJointParams survivingCycle;
  survivingCycle.parentLink = 3; // B
  survivingCycle.childLink = 4; // C
  bp.cycles.push_back(survivingCycle);
  RebuildBotData(bp, ExpectOK{});

  RemoveLinkAndDescendants(bp, 1, ExpectOK{});

  ASSERT_EQ(isize(bp.cycles), 1);
  EXPECT_EQ(bp.links[bp.cycles[0].parentLink].name, "B");
  EXPECT_EQ(bp.links[bp.cycles[0].childLink].name, "C");
}

// ApplyMod(AttachBot) appends the attached child's cycle joints, offsetting their
// link indices into the merged link array.
TEST_F(SortBotPrefabTest, ApplyMod_AttachBot_OffsetsCycleIndices) {
  // Base bot: just a root link to attach onto.
  BotPrefab base;
  base.name = "base";
  base.links.push_back(MakeLink("root", kIndexNone));
  base.joints.push_back(MakeHardJoint("root_joint"));
  RebuildBotData(base, ExpectOK{});

  // Child bot: croot -> tipA, croot -> tipB, with a cycle between tipA and tipB.
  InMemoryBotLoader loader;
  loader.child.name = "child";
  loader.child.links.push_back(MakeLink("croot", kIndexNone));
  loader.child.links.push_back(MakeLink("tipA", 0));
  loader.child.links.push_back(MakeLink("tipB", 0));
  loader.child.joints.push_back(MakeHardJoint("croot_joint"));
  loader.child.joints.push_back(MakeRevoluteJoint("tipA_joint"));
  loader.child.joints.push_back(MakeRevoluteJoint("tipB_joint"));
  loader.child.defaultPose = {0.0_r, 0.0_r};
  ArticulatedCycleJointParams cycle;
  cycle.parentLink = 1; // tipA
  cycle.childLink = 2; // tipB
  loader.child.cycles.push_back(cycle);
  RebuildBotData(loader.child, ExpectOK{});

  AttachBot mod;
  mod.parentLinkName = "root";
  mod.prefix = "arm/";
  mod.path = "child.superdex_bot";
  mod.joint = MakeHardJoint("attach_joint");
  ApplyMod(base, mod, loader, /*validate=*/false, ExpectOK{});
  RebuildBotData(base, ExpectOK{});

  // The child's cycle should now appear on the base, still connecting the same
  // (now-prefixed) links.
  ASSERT_EQ(isize(base.cycles), 1);
  EXPECT_EQ(base.links[base.cycles[0].parentLink].name, "arm/tipA");
  EXPECT_EQ(base.links[base.cycles[0].childLink].name, "arm/tipB");
}

TEST_F(SortBotPrefabTest, ApplyMod_AttachBot_PreservesInvalidCycleIndices) {
  BotPrefab base;
  base.name = "base";
  base.links.push_back(MakeLink("root", kIndexNone));
  base.joints.push_back(MakeHardJoint("root_joint"));
  RebuildBotData(base, ExpectOK{});

  InMemoryBotLoader loader;
  loader.child.name = "child";
  loader.child.links.push_back(MakeLink("croot", kIndexNone));
  loader.child.links.push_back(MakeLink("tip", 0));
  loader.child.joints.push_back(MakeHardJoint("croot_joint"));
  loader.child.joints.push_back(MakeRevoluteJoint("tip_joint"));
  loader.child.defaultPose = {0_r};
  ArticulatedCycleJointParams sentinelCycle;
  sentinelCycle.parentLink = kIndexNone;
  sentinelCycle.childLink = 1;
  loader.child.cycles.push_back(sentinelCycle);
  ArticulatedCycleJointParams outOfRangeCycle;
  outOfRangeCycle.parentLink = 99;
  outOfRangeCycle.childLink = kIndexNone;
  loader.child.cycles.push_back(outOfRangeCycle);

  AttachBot mod;
  mod.parentLinkName = "root";
  mod.prefix = "arm/";
  mod.path = "child.superdex_bot";
  mod.joint = MakeHardJoint("attach_joint");
  ApplyMod(base, mod, loader, /*validate=*/false, ExpectOK{});

  ASSERT_EQ(isize(base.cycles), 2);
  EXPECT_EQ(base.cycles[0].parentLink, kIndexNone);
  EXPECT_EQ(base.links[base.cycles[0].childLink].name, "arm/tip");
  EXPECT_EQ(base.cycles[1].parentLink, 99);
  EXPECT_EQ(base.cycles[1].childLink, kIndexNone);
}

// ApplyMod(AttachBot) appends the attached child's contact overrides with the
// mod's prefix applied to both link names.
TEST_F(SortBotPrefabTest, ApplyMod_AttachBot_PrefixesContactOverrides) {
  // Base bot: just a root link to attach onto.
  BotPrefab base;
  base.name = "base";
  base.links.push_back(MakeLink("root", kIndexNone));
  base.joints.push_back(MakeHardJoint("root_joint"));
  RebuildBotData(base, ExpectOK{});

  // Child bot (returned by the loader): root -> tip, with a override between them.
  InMemoryBotLoader loader;
  loader.child.name = "child";
  loader.child.links.push_back(MakeLink("croot", kIndexNone));
  loader.child.links.push_back(MakeLink("tip", 0));
  loader.child.joints.push_back(MakeHardJoint("croot_joint"));
  loader.child.joints.push_back(MakeRevoluteJoint("tip_joint"));
  loader.child.defaultPose = {0.0_r};
  loader.child.contactOverrides.push_back(BotContactOverride{"croot", "tip", true});
  RebuildBotData(loader.child, ExpectOK{});

  AttachBot mod;
  mod.parentLinkName = "root";
  mod.prefix = "arm/";
  mod.path = "child.superdex_bot";
  mod.joint = MakeHardJoint("attach_joint");
  ApplyMod(base, mod, loader, /*validate=*/false, ExpectOK{});

  // The child's single contact override should now appear on the base, with both
  // link names prefixed.
  ASSERT_EQ(isize(base.contactOverrides), 1);
  EXPECT_EQ(base.contactOverrides[0].linkA, "arm/croot");
  EXPECT_EQ(base.contactOverrides[0].linkB, "arm/tip");
  EXPECT_TRUE(base.contactOverrides[0].enable);
}

// ApplyMod(AttachBot) adopts a skinned child's skin: it records the per-bone->link mapping
// (_skinBoneLinks) in the child's link (bone) order with the mod prefix, and forwards+prefixes the
// names of the links the skin covers.
TEST_F(SortBotPrefabTest, ApplyMod_AttachBot_AdoptsChildSkin) {
  BotPrefab base;
  base.name = "base";
  base.links.push_back(MakeLink("root", kIndexNone));
  base.joints.push_back(MakeHardJoint("root_joint"));
  RebuildBotData(base, ExpectOK{});

  InMemoryBotLoader loader;
  loader.child.name = "child";
  loader.child.links.push_back(MakeLink("croot", kIndexNone));
  loader.child.links.push_back(MakeLink("tip", 0));
  loader.child.joints.push_back(MakeHardJoint("croot_joint"));
  loader.child.joints.push_back(MakeRevoluteJoint("tip_joint"));
  loader.child.defaultPose = {0.0_r};
  loader.child.skin.emplace();
  loader.child.skin->nonCollidingLinks.emplace();
  loader.child.skin->nonCollidingLinks->push_back(DynamicString("tip"));
  loader.child.skin->hiddenLinks.emplace();
  loader.child.skin->hiddenLinks->push_back(DynamicString("croot"));
  RebuildBotData(loader.child, ExpectOK{});

  AttachBot mod;
  mod.parentLinkName = "root";
  mod.prefix = "arm/";
  mod.path = "child.superdex_bot";
  mod.joint = MakeHardJoint("attach_joint");
  ApplyMod(base, mod, loader, /*validate=*/false, ExpectOK{});

  ASSERT_TRUE(base.skin.has_value());
  // The child's skin bones are identity with its (sorted) link order, so the mapping is the child
  // link names with the mod prefix, in link order.
  DynamicArray<DynamicString> const expectedBoneLinks = {"arm/croot", "arm/tip"};
  EXPECT_EQ(base._skinBoneLinks, expectedBoneLinks);
  // The names of the links the skin stands in for are forwarded with the same prefix.
  ASSERT_TRUE(base.skin->nonCollidingLinks.has_value());
  ASSERT_EQ(isize(*base.skin->nonCollidingLinks), 1);
  EXPECT_EQ((*base.skin->nonCollidingLinks)[0], "arm/tip");
  ASSERT_TRUE(base.skin->hiddenLinks.has_value());
  ASSERT_EQ(isize(*base.skin->hiddenLinks), 1);
  EXPECT_EQ((*base.skin->hiddenLinks)[0], "arm/croot");
}

// ApplyMod(AttachBot) rejects attaching a skinned child onto a bot that already has a skin: a bot
// supports at most one skin.
TEST_F(SortBotPrefabTest, ApplyMod_AttachBot_RejectsTwoSkins) {
  BotPrefab base;
  base.name = "base";
  base.links.push_back(MakeLink("root", kIndexNone));
  base.joints.push_back(MakeHardJoint("root_joint"));
  base.skin.emplace();
  RebuildBotData(base, ExpectOK{});

  InMemoryBotLoader loader;
  loader.child.name = "child";
  loader.child.links.push_back(MakeLink("croot", kIndexNone));
  loader.child.joints.push_back(MakeHardJoint("croot_joint"));
  loader.child.skin.emplace();
  RebuildBotData(loader.child, ExpectOK{});

  AttachBot mod;
  mod.parentLinkName = "root";
  mod.prefix = "arm/";
  mod.path = "child.superdex_bot";
  mod.joint = MakeHardJoint("attach_joint");
  ApplyMod(base, mod, loader, /*validate=*/false, ExpectNotOK{});
}

// A base bot that carries its own skin has no _skinBoneLinks -- its baked bone indices match its
// own link order. Mods then add links and RebuildBotData re-sorts them, so BuildBot must capture
// that identity mapping up front; otherwise each bone silently binds to whatever link ends up at
// its index. "aaa" sorts ahead of "root"'s existing child, so the added link shifts link order.
TEST_F(SortBotPrefabTest, BuildBot_CapturesIdentitySkinMappingForSkinnedBase) {
  InMemoryBotLoader loader;
  loader.basePath = "base.superdex_bot";
  loader.baseBot.name = "base";
  loader.baseBot.links.push_back(MakeLink("root", kIndexNone));
  loader.baseBot.links.push_back(MakeLink("zzz", 0));
  loader.baseBot.joints.push_back(MakeHardJoint("root_joint"));
  loader.baseBot.joints.push_back(MakeRevoluteJoint("zzz_joint"));
  loader.baseBot.defaultPose = {0.0_r};
  loader.baseBot.skin.emplace();
  RebuildBotData(loader.baseBot, ExpectOK{});
  // The base is authored without a mapping: bone b binds to link b.
  ASSERT_TRUE(loader.baseBot._skinBoneLinks.empty());
  DynamicArray<DynamicString> const baseOrder = LinkNames(loader.baseBot);

  // A mod that inserts a link sorting before the existing child, so link order changes.
  AttachLink attach;
  attach.parentLinkName = "root";
  attach.joint = MakeRevoluteJoint("aaa_joint");
  attach.link = MakeLink("aaa", 0);

  ModBotPrefab recipe;
  recipe.base = "base.superdex_bot";
  recipe.modifications.push_back(attach);

  BotPrefab const built = BuildBot(recipe, loader, /*validate=*/false, ExpectOK{});

  // The mapping was captured before the mod ran, so it still names the base's original links in
  // their original order -- not whatever occupies those indices now.
  EXPECT_EQ(built._skinBoneLinks, baseOrder);
  // Sanity: the mod really did perturb link order, so the identity assumption would have been
  // wrong had it not been captured.
  EXPECT_NE(LinkNames(built), baseOrder);
}

// A recipe's contact overrides are applied on top of the composed bot, where an attached child may
// already have contributed an override for the same link pair. The recipe's entry must REPLACE that
// one rather than sit behind it: readers that stop at the first match would report the child's
// value, while spawning applies every entry in order and would end up with the recipe's -- so the
// editor matrix and the spawned bot would disagree.
TEST_F(SortBotPrefabTest, BuildBot_RecipeContactOverrideReplacesChildOverride) {
  InMemoryBotLoader loader;
  loader.basePath = "base.superdex_bot";
  loader.baseBot.name = "base";
  loader.baseBot.links.push_back(MakeLink("root", kIndexNone));
  loader.baseBot.joints.push_back(MakeHardJoint("root_joint"));
  RebuildBotData(loader.baseBot, ExpectOK{});

  // Child contributes an override between its own two links; AttachBot prefixes both names.
  loader.child.name = "child";
  loader.child.links.push_back(MakeLink("croot", kIndexNone));
  loader.child.links.push_back(MakeLink("tip", 0));
  loader.child.joints.push_back(MakeHardJoint("croot_joint"));
  loader.child.joints.push_back(MakeRevoluteJoint("tip_joint"));
  loader.child.defaultPose = {0.0_r};
  loader.child.contactOverrides.push_back(BotContactOverride{"croot", "tip", true});
  RebuildBotData(loader.child, ExpectOK{});

  AttachBot attach;
  attach.parentLinkName = "root";
  attach.prefix = "arm/";
  attach.path = "child.superdex_bot";
  attach.joint = MakeHardJoint("attach_joint");

  ModBotPrefab recipe;
  recipe.base = "base.superdex_bot";
  recipe.modifications.push_back(attach);
  // Same pair the child contributed, opposite value, endpoints reversed to prove the match is
  // order-independent.
  recipe.contactOverrides.push_back(BotContactOverride{"arm/tip", "arm/croot", false});

  BotPrefab const built = BuildBot(recipe, loader, /*validate=*/false, ExpectOK{});

  // One entry for the pair, carrying the recipe's value -- not two with the child's first.
  ASSERT_EQ(isize(built.contactOverrides), 1);
  EXPECT_FALSE(built.contactOverrides[0].enable);
}

// Single-link bot: the root is the only link and therefore the only leaf.
TEST_F(SortBotPrefabTest, FindLeafLinkIndices_SingleLink) {
  BotPrefab bp;
  bp.links.push_back(MakeLink("root", kIndexNone));
  DynamicArray<int> const expected = {0};
  EXPECT_EQ(FindLeafLinkIndices(bp), expected);
}

// Chain root -> A -> B: only the tip (B) is a leaf.
TEST_F(SortBotPrefabTest, FindLeafLinkIndices_Chain) {
  BotPrefab bp;
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("A", 0));
  bp.links.push_back(MakeLink("B", 1));
  DynamicArray<int> const expected = {2};
  EXPECT_EQ(FindLeafLinkIndices(bp), expected);
}

// Fork with multiple leaves: leaves are returned in ascending index order.
TEST_F(SortBotPrefabTest, FindLeafLinkIndices_ForkAscending) {
  // Tree: root(0) -> A(1) -> B(2); root(0) -> C(3). Leaves: B(2), C(3).
  BotPrefab bp;
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("A", 0));
  bp.links.push_back(MakeLink("B", 1));
  bp.links.push_back(MakeLink("C", 0));
  DynamicArray<int> const expected = {2, 3};
  EXPECT_EQ(FindLeafLinkIndices(bp), expected);
}

// SortBotPrefab remaps linearTransmission joint indices through the same
// permutation as joints/links, so a transmission keeps referencing the same
// named joints after a reorder.
TEST_F(SortBotPrefabTest, SortBotPrefab_RemapsLinearTransmissionJointIndices) {
  //   root
  //   ├── C (idx 1)
  //   ├── B (idx 2)
  //   └── A (idx 3)
  // After sort: root(0), A(1), B(2), C(3). So old C:1->3, old A:3->1.
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("C", 0));
  bp.links.push_back(MakeLink("B", 0));
  bp.links.push_back(MakeLink("A", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("C_joint"));
  bp.joints.push_back(MakeRevoluteJoint("B_joint"));
  bp.joints.push_back(MakeRevoluteJoint("A_joint"));
  bp.defaultPose = {0.0_r, 0.0_r, 0.0_r};
  // Transmission traversing C (old idx 1) and A (old idx 3).
  BotLinearTransmissionPrefab t;
  t.name = "belt";
  t.jointIndices = {1, 3};
  t.jointCoefficients = {0.5_r, -0.5_r};
  bp.linearTransmissions.push_back(t);
  RebuildBotData(bp, ExpectOK{});
  // The transmission is preserved and its indices now point at the same joints.
  ASSERT_EQ(isize(bp.linearTransmissions), 1);
  auto const& indices = bp.linearTransmissions[0].jointIndices;
  ASSERT_EQ(isize(indices), 2);
  EXPECT_EQ(bp.joints[indices[0]].name, "C_joint");
  EXPECT_EQ(bp.joints[indices[1]].name, "A_joint");
}

// SortBotPrefab remaps spatial tendon routing indices through the same
// permutation. Waypoint elements reference links; LinearJoint elements reference
// joints. Because joint i travels with link i, one permutation covers both.
TEST_F(SortBotPrefabTest, SortBotPrefab_RemapsSpatialTendonIndices) {
  //   root
  //   ├── C (idx 1)
  //   ├── B (idx 2)
  //   └── A (idx 3)
  // After sort: root(0), A(1), B(2), C(3). So old A:3->1, old C:1->3.
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("C", 0));
  bp.links.push_back(MakeLink("B", 0));
  bp.links.push_back(MakeLink("A", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("C_joint"));
  bp.joints.push_back(MakeRevoluteJoint("B_joint"));
  bp.joints.push_back(MakeRevoluteJoint("A_joint"));
  bp.defaultPose = {0.0_r, 0.0_r, 0.0_r};
  BotSpatialTendonPrefab s;
  s.name = "tendon";
  RoutingElement waypoint;
  waypoint.type = RoutingElementType::Waypoint;
  waypoint.index = 3; // link A (old idx 3)
  waypoint.localPosition = {0_r, 0_r, 0_r};
  RoutingElement linearJoint;
  linearJoint.type = RoutingElementType::LinearJoint;
  linearJoint.index = 1; // joint C (old idx 1)
  linearJoint.coefficient = 0.01_r;
  s.routingElements = {waypoint, linearJoint};
  bp.spatialTendons.push_back(s);
  RebuildBotData(bp, ExpectOK{});
  // The tendon is preserved and its indices now point at the same link/joint.
  ASSERT_EQ(isize(bp.spatialTendons), 1);
  auto const& elems = bp.spatialTendons[0].routingElements;
  ASSERT_EQ(isize(elems), 2);
  EXPECT_EQ(bp.links[elems[0].index].name, "A");
  EXPECT_EQ(bp.joints[elems[1].index].name, "C_joint");
}

// RemoveLinkAndDescendants remaps surviving transmission/tendon indices through
// indexMap and drops any transmission/tendon that referenced a removed joint or
// link, mirroring the cycle handling.
TEST_F(SortBotPrefabTest, RemoveLinkAndDescendants_RemapsAndDropsTransmissionsAndTendons) {
  //   root
  //   |-- A
  //   |   `-- AChild
  //   |-- B
  //   `-- C
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("A", 0));
  bp.links.push_back(MakeLink("AChild", 1));
  bp.links.push_back(MakeLink("B", 0));
  bp.links.push_back(MakeLink("C", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("A_joint"));
  bp.joints.push_back(MakeRevoluteJoint("AChild_joint"));
  bp.joints.push_back(MakeRevoluteJoint("B_joint"));
  bp.joints.push_back(MakeRevoluteJoint("C_joint"));
  bp.defaultPose = {0_r, 0_r, 0_r, 0_r};

  // Transmission over B (3) and C (4) survives; transmission over A (1) is dropped.
  BotLinearTransmissionPrefab surviving;
  surviving.name = "survivingBelt";
  surviving.jointIndices = {3, 4};
  surviving.jointCoefficients = {0.5_r, -0.5_r};
  bp.linearTransmissions.push_back(surviving);
  BotLinearTransmissionPrefab dropped;
  dropped.name = "droppedBelt";
  dropped.jointIndices = {1};
  dropped.jointCoefficients = {1.0_r};
  bp.linearTransmissions.push_back(dropped);

  // Tendon with a waypoint on C (4) survives; tendon with a waypoint on the
  // removed AChild (2) is dropped.
  BotSpatialTendonPrefab survivingTendon;
  survivingTendon.name = "survivingTendon";
  RoutingElement cWaypoint;
  cWaypoint.type = RoutingElementType::Waypoint;
  cWaypoint.index = 4; // link C
  survivingTendon.routingElements = {cWaypoint};
  bp.spatialTendons.push_back(survivingTendon);
  BotSpatialTendonPrefab droppedTendon;
  droppedTendon.name = "droppedTendon";
  RoutingElement aChildWaypoint;
  aChildWaypoint.type = RoutingElementType::Waypoint;
  aChildWaypoint.index = 2; // link AChild (removed)
  droppedTendon.routingElements = {aChildWaypoint};
  bp.spatialTendons.push_back(droppedTendon);

  RebuildBotData(bp, ExpectOK{});
  RemoveLinkAndDescendants(bp, 1, ExpectOK{});

  // Only the surviving transmission remains, remapped to the same named joints.
  ASSERT_EQ(isize(bp.linearTransmissions), 1);
  EXPECT_EQ(bp.linearTransmissions[0].name, "survivingBelt");
  auto const& indices = bp.linearTransmissions[0].jointIndices;
  ASSERT_EQ(isize(indices), 2);
  EXPECT_EQ(bp.joints[indices[0]].name, "B_joint");
  EXPECT_EQ(bp.joints[indices[1]].name, "C_joint");
  // Only the surviving tendon remains, remapped to the same named link.
  ASSERT_EQ(isize(bp.spatialTendons), 1);
  EXPECT_EQ(bp.spatialTendons[0].name, "survivingTendon");
  ASSERT_EQ(isize(bp.spatialTendons[0].routingElements), 1);
  EXPECT_EQ(bp.links[bp.spatialTendons[0].routingElements[0].index].name, "C");
}

// AddLink inserts a new link/joint and shifts later links/joints up by one.
// Transmission/tendon references at or after the insertion point must shift too,
// while references before it stay put, so they keep targeting the same named
// joints/links.
TEST_F(SortBotPrefabTest, AddLink_ShiftsTransmissionAndTendonIndices) {
  //   root
  //   ├── A (idx 1)
  //   ├── B (idx 2)
  //   └── C (idx 3)
  // Adding a child under A inserts at idx 2, so B:2->3 and C:3->4, while A stays 1.
  BotPrefab bp;
  bp.name = "test";
  bp.links.push_back(MakeLink("root", kIndexNone));
  bp.links.push_back(MakeLink("A", 0));
  bp.links.push_back(MakeLink("B", 0));
  bp.links.push_back(MakeLink("C", 0));
  bp.joints.push_back(MakeHardJoint("root_joint"));
  bp.joints.push_back(MakeRevoluteJoint("A_joint"));
  bp.joints.push_back(MakeRevoluteJoint("B_joint"));
  bp.joints.push_back(MakeRevoluteJoint("C_joint"));
  bp.defaultPose = {0.0_r, 0.0_r, 0.0_r};
  // Transmission over A (idx 1, before insertion) and B (idx 2, after insertion).
  BotLinearTransmissionPrefab t;
  t.name = "belt";
  t.jointIndices = {1, 2};
  t.jointCoefficients = {0.5_r, -0.5_r};
  bp.linearTransmissions.push_back(t);
  // Tendon with a waypoint on C (idx 3, after insertion).
  BotSpatialTendonPrefab s;
  s.name = "tendon";
  RoutingElement waypoint;
  waypoint.type = RoutingElementType::Waypoint;
  waypoint.index = 3; // link C
  bp.spatialTendons.push_back(s);
  bp.spatialTendons[0].routingElements = {waypoint};
  RebuildBotData(bp, ExpectOK{});

  int const newIdx = AddLink(bp, 1, "child", ExpectOK{});
  EXPECT_EQ(newIdx, 2);

  // The transmission still targets A (unshifted) and B (shifted), the tendon C.
  ASSERT_EQ(isize(bp.linearTransmissions), 1);
  auto const& indices = bp.linearTransmissions[0].jointIndices;
  ASSERT_EQ(isize(indices), 2);
  EXPECT_EQ(bp.joints[indices[0]].name, "A_joint");
  EXPECT_EQ(bp.joints[indices[1]].name, "B_joint");
  ASSERT_EQ(isize(bp.spatialTendons), 1);
  ASSERT_EQ(isize(bp.spatialTendons[0].routingElements), 1);
  EXPECT_EQ(bp.links[bp.spatialTendons[0].routingElements[0].index].name, "C");
}
