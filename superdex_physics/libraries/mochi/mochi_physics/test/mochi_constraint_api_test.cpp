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

#include "mochi_constraint_test.h"

using namespace mochi;
using namespace mochi::test;
using namespace mochi::constraint_test;

/********************************************************************************
  ConstraintAPI - These use the public API to create constraints.
********************************************************************************/
class ConstraintAPI : public MochiSceneTestBase {
 protected:
  std::vector<ActorHandle> _actors;
  ConstraintParams _conParams{1.0_r};

 public:
  void SetUp() override {
    MochiSceneTestBase::SetUp();

    // Create two rigid actors
    RigidActorParams rigidParams;
    rigidParams.colliderType = ColliderType::None;
    rigidParams.shape = GetUnitCubeShape(_scene->GetContext());
    _actors.push_back(_scene->CreateRigidActor(rigidParams, ExpectOK{})->GetHandle());
    _actors.push_back(_scene->CreateRigidActor(rigidParams, ExpectOK{})->GetHandle());
  }
};

TEST_F(ConstraintAPI, ExpectAutoCompound) {
  // Create a constraint
  RigidSphericalJointConstraintParams conParams;
  conParams.actorA = _actors[0];
  conParams.actorB = _actors[1];
  auto* constraint = _scene->CreateRigidSphericalJointConstraint(conParams, ExpectOK{});
  auto constraintHandle = constraint->GetHandle();

  // Step the simulation so that the automatic compound will be created
  _scene->Step(0.01);

  // Now, peak under the hood to make sure that the constraint was automatically added to a
  // compound along with the actors.
  entt::registry const& reg = GetRegistry();
  entt::entity constraintEntity = GetEntity(constraintHandle);
  auto const* membership0 = reg.try_get<CGroupMemberInfo const>(GetEntity(_actors[0]));
  auto const* membership1 = reg.try_get<CGroupMemberInfo const>(GetEntity(_actors[1]));
  auto const* membershipC = reg.try_get<CGroupMemberInfo const>(GetEntity(constraintHandle));
  EXPECT_NE((CGroupMemberInfo const*)nullptr, membership0);
  EXPECT_NE((CGroupMemberInfo const*)nullptr, membership1);
  EXPECT_NE((CGroupMemberInfo const*)nullptr, membershipC);
  entt::entity compound = membership0->group;
  EXPECT_EQ(compound, membership1->group);
  EXPECT_EQ(compound, membershipC->group);
  EXPECT_TRUE(reg.valid(compound));
  EXPECT_TRUE((reg.all_of<TagCompoundActor const, CGroupMembers const>(compound)));
  auto const& compoundMembers = reg.get<CGroupMembers const>(compound);
  EXPECT_EQ(1, isize(compoundMembers.constraints));
  EXPECT_EQ(constraintEntity, compoundMembers.constraints[0]);
  EXPECT_EQ(2, isize(compoundMembers.actors));
  if (compoundMembers.actors[0] == GetEntity(_actors[0])) {
    EXPECT_EQ(compoundMembers.actors[1], GetEntity(_actors[1]));
  } else {
    EXPECT_EQ(compoundMembers.actors[1], GetEntity(_actors[0]));
    EXPECT_EQ(compoundMembers.actors[0], GetEntity(_actors[1]));
  }
}

TEST_F(ConstraintAPI, CreateWithInvalidObjId) {
  // Create a constraint with an invalid object ID
  RigidSphericalJointConstraintParams conParams;
  conParams.actorA = _actors[0];
  conParams.actorB = ActorHandle{kInvalidHandle}; // NOT VALID
  Constraint* constraint = _scene->CreateRigidSphericalJointConstraint(conParams, ExpectNotOK{});
  EXPECT_EQ((Constraint*)nullptr, constraint);

  // Now, peak under the hood to make sure no constraints nor compounds were created.
  entt::registry& reg = GetRegistry();
  EXPECT_EQ(0, reg.view<CConstraintInfo const>().size());
  EXPECT_EQ(0, reg.view<TagCompoundActor const>().size());
}
