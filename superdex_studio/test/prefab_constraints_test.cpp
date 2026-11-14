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

#include "assets/prefab_constraints.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

using namespace superdex::studio;
using mochi::ArticulatedJointType;
using mochi::ConstraintType;
using mochi::DynamicString;
using mochi::prefab::ScenePrefab;

namespace {

// A prefab with rigid actors "a" and "b" and articulated actor "arm" whose joints are free (root),
// revolute ("elbow") and spherical ("wrist").
ScenePrefab MakePrefab() {
  ScenePrefab prefab;
  for (char const* name : {"a", "b"}) {
    prefab.actors.rigid.emplace_back();
    prefab.actors.rigid.back().name = DynamicString{name};
  }
  prefab.actors.articulated.emplace_back();
  auto& arm = prefab.actors.articulated.back();
  arm.name = DynamicString{"arm"};
  struct Joint {
    char const* link;
    char const* joint;
    ArticulatedJointType type;
  };
  for (Joint const& j : {
           Joint{"base", "root", ArticulatedJointType::Free},
           Joint{"forearm", "elbow", ArticulatedJointType::Revolute},
           Joint{"hand", "wrist", ArticulatedJointType::Spherical},
       }) {
    arm.links.emplace_back();
    arm.links.back().name = DynamicString{j.link};
    arm.joints.emplace_back();
    arm.joints.back().name = DynamicString{j.joint};
    arm.joints.back().type = j.type;
  }
  return prefab;
}

void AddJointTarget(ScenePrefab& prefab, char const* actor, int joint) {
  ConstraintRef const ref =
      AppendConstraint(prefab.constraints, ConstraintType::ArticulatedSingleDofTarget);
  auto& c = prefab.constraints.articulatedSingleDofTarget[ref.index];
  c.actorName = DynamicString{actor};
  c.jointIndex = joint;
}

void AddSpherical(ScenePrefab& prefab, char const* actorA, char const* actorB) {
  ConstraintRef const ref =
      AppendConstraint(prefab.constraints, ConstraintType::RigidSphericalJoint);
  auto& c = prefab.constraints.rigidSphericalJoint[ref.index];
  c.actorNameA = DynamicString{actorA};
  c.actorNameB = DynamicString{actorB};
}

std::vector<int> JointTargetIndices(ScenePrefab const& prefab) {
  std::vector<int> joints;
  for (auto const& c : prefab.constraints.articulatedSingleDofTarget) {
    joints.push_back(c.jointIndex);
  }
  return joints;
}

std::vector<std::string> SphericalActors(ScenePrefab const& prefab) {
  std::vector<std::string> names;
  for (auto const& c : prefab.constraints.rigidSphericalJoint) {
    names.emplace_back(std::string(c.actorNameA) + "|" + std::string(c.actorNameB));
  }
  return names;
}

} // namespace

TEST(PrefabConstraintsTest, AppendedJointConstraintsAreSoftSprings) {
  ScenePrefab prefab;
  std::vector<ConstraintRef> expected;
  for (ConstraintType type :
       {ConstraintType::ArticulatedSingleDofTarget,
        ConstraintType::ArticulatedSingleDofRange,
        ConstraintType::Articulated3dRotationTarget,
        ConstraintType::Articulated3dRotationRange}) {
    expected.push_back(AppendConstraint(prefab.constraints, type));
  }

  EXPECT_EQ(CollectConstraints(prefab.constraints), expected);
  auto const& lists = prefab.constraints;
  for (mochi::ConstraintParams const* target :
       {static_cast<mochi::ConstraintParams const*>(&lists.articulatedSingleDofTarget[0]),
        static_cast<mochi::ConstraintParams const*>(&lists.articulated3dRotationTarget[0])}) {
    EXPECT_EQ(target->stiffness, 1.0f);
    EXPECT_EQ(target->damping, 0.01f);
  }
  EXPECT_EQ(lists.articulatedSingleDofRange[0].stiffness, 100.0f);
  EXPECT_EQ(lists.articulated3dRotationRange[0].stiffness, 100.0f);
}

TEST(PrefabConstraintsTest, DuplicateAndRemove) {
  ScenePrefab prefab = MakePrefab();
  AddSpherical(prefab, "a", "b");

  auto const copy =
      DuplicateConstraint(prefab.constraints, {ConstraintType::RigidSphericalJoint, 0});
  ASSERT_TRUE(copy.has_value());
  EXPECT_EQ(SphericalActors(prefab), (std::vector<std::string>{"a|b", "a|b"}));

  EXPECT_TRUE(RemoveConstraint(prefab.constraints, *copy));
  EXPECT_FALSE(RemoveConstraint(prefab.constraints, *copy));
  EXPECT_EQ(SphericalActors(prefab), (std::vector<std::string>{"a|b"}));
}

TEST(PrefabConstraintsTest, RenameFollowsActorAndActorsInsideIt) {
  ScenePrefab prefab;
  AddSpherical(prefab, "lamp", "lamp/arm");
  AddSpherical(prefab, "lampshade", "b");

  EXPECT_EQ(RenameConstraintActor(prefab.constraints, "lamp", "light"), 2);

  // "lampshade" only starts with "lamp"; it is another actor.
  EXPECT_EQ(SphericalActors(prefab), (std::vector<std::string>{"light|light/arm", "lampshade|b"}));
}

TEST(PrefabConstraintsTest, RenameCoversEveryActorField) {
  ScenePrefab prefab = MakePrefab();
  AddJointTarget(prefab, "arm", 1);
  auto& lists = prefab.constraints;
  AppendConstraint(lists, ConstraintType::DeformableNodeToRigid);
  lists.deformableNodeToRigid[0].rigidActorName = DynamicString{"arm/hand"};
  lists.deformableNodeToRigid[0].deformableActorName = DynamicString{"cloth"};

  EXPECT_EQ(RenameConstraintActor(lists, "arm", "limb"), 2);
  EXPECT_EQ(RenameConstraintActor(lists, "cloth", "sheet"), 1);

  EXPECT_EQ(std::string(lists.articulatedSingleDofTarget[0].actorName), "limb");
  EXPECT_EQ(std::string(lists.deformableNodeToRigid[0].rigidActorName), "limb/hand");
  EXPECT_EQ(std::string(lists.deformableNodeToRigid[0].deformableActorName), "sheet");
  EXPECT_EQ(RemoveConstraintsReferencingActor(lists, "sheet"), 1);
  EXPECT_EQ(RemoveConstraintsReferencingActor(lists, "limb"), 1);
  EXPECT_TRUE(CollectConstraints(lists).empty());
}

TEST(PrefabConstraintsTest, RemovingAnActorRemovesItsConstraints) {
  ScenePrefab prefab;
  AddSpherical(prefab, "a", "b");
  AddSpherical(prefab, "nested/a", "c");
  AddSpherical(prefab, "c", "d");

  EXPECT_EQ(RemoveConstraintsReferencingActor(prefab.constraints, "a"), 1);
  EXPECT_EQ(RemoveConstraintsReferencingActor(prefab.constraints, "nested"), 1);
  EXPECT_EQ(SphericalActors(prefab), (std::vector<std::string>{"c|d"}));
}

TEST(PrefabConstraintsTest, RemapMovesJointConstraintsAndDropsRemovedJoints) {
  ScenePrefab prefab = MakePrefab();
  AddJointTarget(prefab, "arm", 1);
  AddJointTarget(prefab, "arm", 2);
  AddJointTarget(prefab, "other", 2);

  std::vector<int> const oldToNew{0, -1, 1}; // joint 1 removed, joint 2 moved to 1
  EXPECT_EQ(RemapJointConstraints(prefab.constraints, "arm", oldToNew), 1);

  EXPECT_EQ(JointTargetIndices(prefab), (std::vector<int>{1, 2}));
  EXPECT_EQ(std::string(prefab.constraints.articulatedSingleDofTarget[1].actorName), "other");
}

TEST(PrefabConstraintsTest, CopyJointConstraintsRetargetsToTheCopy) {
  ScenePrefab prefab = MakePrefab();
  AddJointTarget(prefab, "arm", 1);

  EXPECT_EQ(CopyJointConstraints(prefab.constraints, "arm", "arm_1"), 1);

  std::vector<ConstraintRef> const original{{ConstraintType::ArticulatedSingleDofTarget, 0}};
  std::vector<ConstraintRef> const copy{{ConstraintType::ArticulatedSingleDofTarget, 1}};
  EXPECT_EQ(FindJointConstraints(prefab.constraints, "arm", 1), original);
  EXPECT_EQ(FindJointConstraints(prefab.constraints, "arm_1", 1), copy);
}

TEST(PrefabConstraintsTest, ActorConstraintsIncludeActorsInsideItButNotItsOwnJoints) {
  ScenePrefab prefab = MakePrefab();
  AddSpherical(prefab, "a", "b");
  AddSpherical(prefab, "b", "lamp/shade");
  AddJointTarget(prefab, "arm", 1);
  AddJointTarget(prefab, "lamp/arm", 1);

  std::vector<ConstraintRef> const onB{
      {ConstraintType::RigidSphericalJoint, 0}, {ConstraintType::RigidSphericalJoint, 1}};
  std::vector<ConstraintRef> const inLamp{
      {ConstraintType::ArticulatedSingleDofTarget, 1}, {ConstraintType::RigidSphericalJoint, 1}};
  EXPECT_EQ(FindActorConstraints(prefab.constraints, "b"), onB);
  EXPECT_EQ(FindActorConstraints(prefab.constraints, "lamp"), inLamp);
  EXPECT_TRUE(FindActorConstraints(prefab.constraints, "arm").empty());
}

TEST(PrefabConstraintsTest, DescribeNamesTheJoint) {
  ScenePrefab prefab = MakePrefab();
  AddJointTarget(prefab, "arm", 1);
  AddSpherical(prefab, "a", "b");

  EXPECT_EQ(
      DescribeConstraint(prefab, {ConstraintType::ArticulatedSingleDofTarget, 0}), "arm/elbow");
  EXPECT_EQ(DescribeConstraint(prefab, {ConstraintType::RigidSphericalJoint, 0}), "a & b");
}

TEST(PrefabConstraintsTest, ValidationChecksActorsJointsAndValues) {
  ScenePrefab prefab = MakePrefab();
  std::vector<std::string> const nested{"lamp/arm/forearm"}; // a link inside a nested prefab
  ConstraintActorIndex const actors = BuildConstraintActorIndex(prefab, nested);
  auto const valid = [&](ConstraintRef ref) {
    return ValidateConstraint(prefab, actors, ref).empty();
  };
  auto& lists = prefab.constraints;

  AddJointTarget(prefab, "arm", 1); // revolute: ok
  AddJointTarget(prefab, "arm", 2); // spherical: needs a 3D rotation constraint
  AddJointTarget(prefab, "arm", 7); // no such joint
  AddJointTarget(prefab, "lamp/arm", 3); // in a nested prefab: joints checked when it loads
  EXPECT_TRUE(valid({ConstraintType::ArticulatedSingleDofTarget, 0}));
  EXPECT_FALSE(valid({ConstraintType::ArticulatedSingleDofTarget, 1}));
  EXPECT_FALSE(valid({ConstraintType::ArticulatedSingleDofTarget, 2}));
  EXPECT_TRUE(valid({ConstraintType::ArticulatedSingleDofTarget, 3}));

  ConstraintRef const rotation =
      AppendConstraint(lists, ConstraintType::Articulated3dRotationTarget);
  lists.articulated3dRotationTarget[0].actorName = DynamicString{"arm"};
  lists.articulated3dRotationTarget[0].jointIndex = 2;
  EXPECT_TRUE(valid(rotation));
  lists.articulated3dRotationTarget[0].jointIndex = 1; // revolute
  EXPECT_FALSE(valid(rotation));

  AddSpherical(prefab, "a", "b");
  AddSpherical(prefab, "a", "a");
  AddSpherical(prefab, "a", "arm"); // articulated, not rigid
  AddSpherical(prefab, "a", "arm/forearm"); // a link is a rigid actor
  AddSpherical(prefab, "a", "lamp/arm/forearm"); // inside a nested prefab: kind unknown
  EXPECT_TRUE(valid({ConstraintType::RigidSphericalJoint, 0}));
  EXPECT_FALSE(valid({ConstraintType::RigidSphericalJoint, 1}));
  EXPECT_FALSE(valid({ConstraintType::RigidSphericalJoint, 2}));
  EXPECT_TRUE(valid({ConstraintType::RigidSphericalJoint, 3}));
  EXPECT_TRUE(valid({ConstraintType::RigidSphericalJoint, 4}));

  ConstraintRef const prismatic = AppendConstraint(lists, ConstraintType::RigidPrismaticJoint);
  lists.rigidPrismaticJoint[0].actorNameA = DynamicString{"a"};
  lists.rigidPrismaticJoint[0].actorNameB = DynamicString{"b"};
  lists.rigidPrismaticJoint[0].freeAxis = mochi::Real3{0.0f, 0.0f, 0.0f};
  EXPECT_FALSE(valid(prismatic));
  lists.rigidPrismaticJoint[0].freeAxis = mochi::Real3{0.0f, 0.0f, 1.0f};
  EXPECT_TRUE(valid(prismatic));
  lists.rigidPrismaticJoint[0].min = 0.1f;
  lists.rigidPrismaticJoint[0].max = -0.1f;
  EXPECT_FALSE(valid(prismatic));

  ConstraintRef const rotationRange = AppendConstraint(lists, ConstraintType::JointRotationRange);
  lists.jointRotationRange[0].actorNameA = DynamicString{"a"};
  lists.jointRotationRange[0].actorNameB = DynamicString{"b"};
  lists.jointRotationRange[0].angleRangeY = mochi::Real2{1.0f, -1.0f};
  EXPECT_FALSE(valid(rotationRange));

  ConstraintRef const pivot = AppendConstraint(lists, ConstraintType::RigidPivotPosition);
  lists.rigidPivotPosition[0].actorName = DynamicString{"a"};
  EXPECT_TRUE(valid(pivot));
  lists.rigidPivotPosition[0].damping = -1.0f;
  EXPECT_FALSE(valid(pivot));
  lists.rigidPivotPosition[0].damping = 0.0f;
  lists.rigidPivotPosition[0].saturation = 0.0f;
  EXPECT_FALSE(valid(pivot));

  ConstraintRef const range = AppendConstraint(lists, ConstraintType::ArticulatedSingleDofRange);
  lists.articulatedSingleDofRange[0].actorName = DynamicString{"arm"};
  lists.articulatedSingleDofRange[0].jointIndex = 1;
  lists.articulatedSingleDofRange[0].minValue = 1.0f;
  lists.articulatedSingleDofRange[0].maxValue = -1.0f;
  EXPECT_FALSE(valid(range));
  lists.articulatedSingleDofRange[0].maxValue = 2.0f;
  EXPECT_TRUE(valid(range));
  lists.articulatedSingleDofRange[0].stiffness = -1.0f;
  EXPECT_FALSE(valid(range));
}

TEST(PrefabConstraintsTest, RemoveInvalidConstraintsKeepsTheValidOnes) {
  ScenePrefab prefab = MakePrefab();
  AddSpherical(prefab, "a", "missing");
  AddSpherical(prefab, "a", "b");
  AddJointTarget(prefab, "arm", 9);

  std::vector<std::string> const reasons =
      RemoveInvalidConstraints(prefab, BuildConstraintActorIndex(prefab, {}));

  EXPECT_EQ(reasons.size(), 2u);
  EXPECT_EQ(SphericalActors(prefab), (std::vector<std::string>{"a|b"}));
  EXPECT_EQ(CollectConstraints(prefab.constraints).size(), 1u);
}

TEST(PrefabConstraintsTest, DeformableConstraintsNeedNodesOrFindClosest) {
  ScenePrefab prefab = MakePrefab();
  prefab.actors.soft.emplace_back();
  prefab.actors.soft.back().name = DynamicString{"cloth"};
  ConstraintActorIndex const actors = BuildConstraintActorIndex(prefab, {});
  auto& lists = prefab.constraints;

  ConstraintRef const toRigid = AppendConstraint(lists, ConstraintType::DeformableNodeToRigid);
  lists.deformableNodeToRigid[0].rigidActorName = DynamicString{"a"};
  lists.deformableNodeToRigid[0].deformableActorName = DynamicString{"cloth"};
  lists.deformableNodeToRigid[0].findClosest = false;
  lists.deformableNodeToRigid[0].deformableNodeIndex = -1;
  EXPECT_FALSE(ValidateConstraint(prefab, actors, toRigid).empty());
  lists.deformableNodeToRigid[0].findClosest = true;
  EXPECT_TRUE(ValidateConstraint(prefab, actors, toRigid).empty());
  lists.deformableNodeToRigid[0].deformableActorName = DynamicString{"a"}; // rigid, not soft
  EXPECT_FALSE(ValidateConstraint(prefab, actors, toRigid).empty());
}

TEST(PrefabConstraintsTest, MetricScaleComposesNestedAndArticulatedScales) {
  auto inner = std::make_shared<ScenePrefab>(MakePrefab());
  inner->actors.articulated[0].scale = 2.0f;
  ScenePrefab prefab = MakePrefab();
  prefab.prefabs.emplace_back();
  prefab.prefabs.back().name = DynamicString{"lamp"};
  prefab.prefabs.back().scale = 3.0f;
  prefab.prefabs.back().prefab = inner;

  EXPECT_EQ(GetConstraintActorMetricScale(prefab, "a"), 1.0f);
  EXPECT_EQ(GetConstraintActorMetricScale(prefab, "lamp/a"), 3.0f);
  EXPECT_EQ(GetConstraintActorMetricScale(prefab, "lamp/arm/forearm"), 6.0f);
}

TEST(PrefabConstraintsTest, SingleDofConstraintsAddressTranslationsAndHingeAngles) {
  EXPECT_TRUE(IsSingleDofConstraintDof(ArticulatedJointType::Revolute, 0));
  EXPECT_TRUE(IsSingleDofConstraintDof(ArticulatedJointType::Prismatic, 0));
  EXPECT_TRUE(IsSingleDofConstraintDof(ArticulatedJointType::Free, 2));
  EXPECT_FALSE(IsSingleDofConstraintDof(ArticulatedJointType::Free, 3));
  EXPECT_FALSE(IsSingleDofConstraintDof(ArticulatedJointType::Spherical, 0));
  EXPECT_TRUE(IsRotationDof(ArticulatedJointType::Revolute, 0));
  EXPECT_FALSE(IsRotationDof(ArticulatedJointType::Free, 0));
}
