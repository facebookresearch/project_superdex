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

#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/src/mochi_group.h>
#include <mochi_physics/src/mochi_island.h>

#include <vector>

using namespace mochi;

/********************************************************************************
  ConstraintTest - Test fixture that was previously used to test "compound actors".
                   Now, all that remains is one test regarding constraints.
                   TODO: Move this code to live with other constraint tests.
********************************************************************************/
namespace {
class ConstraintTest : public test::MochiSceneTestBase {
 public:
  ShapeHandle _cubeShape;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp(); // call down
    _scene->SetGravity(kDefaultGravity);

    auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
    _cubeShape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});
  }
};
} // namespace

TEST_F(ConstraintTest, DeleteConstraintLeavingActors) {
  static int constexpr kNumConstraints = 6;
  static int constexpr kNumActors = 4;

  auto& reg = GetRegistry();

  std::vector<ActorHandle> actors;
  std::vector<ConstraintHandle> constraints;

  // Create 4 rigid actors
  RigidActorParams rigidParams;
  rigidParams.colliderType = ColliderType::Box;
  rigidParams.shape = _cubeShape;
  rigidParams.layer = "object";
  actors.reserve(kNumActors);
  for (int i = 0; i < kNumActors; ++i) {
    actors.push_back(_scene->CreateRigidActor(rigidParams, test::ExpectOK{})->GetHandle());
  }

  // Create constraints between each pair of actors
  for (int i = 0; i < isize(actors); ++i) {
    for (int j = i + 1; j < isize(actors); ++j) {
      RigidSphericalJointConstraintParams jointParams;
      jointParams.actorA = actors[i];
      jointParams.actorB = actors[j];
      auto* constraint = _scene->CreateRigidSphericalJointConstraint(jointParams, test::ExpectOK{});
      constraints.push_back(constraint->GetHandle());
    }
  }

  // Confirm that the actors are not in a compound yet
  for (auto actor : actors) {
    EXPECT_FALSE(reg.try_get<CGroupMemberInfo const>(GetEntity(actor)));
  }

  // Simulate. This should create an auto-compound
  _scene->Step(0.01);
  for (auto actor : actors) {
    auto const* group = reg.try_get<CGroupMemberInfo const>(GetEntity(actor));
    EXPECT_TRUE(group);
    if (group) {
      auto compound = group->group;
      auto const& groupMembers = reg.get<CGroupMembers const>(compound);
      EXPECT_EQ(kNumActors, isize(groupMembers.actors));
      EXPECT_EQ(kNumConstraints, isize(groupMembers.constraints));
    }
  }

  // Delete all the constraints
  for (auto c : constraints) {
    _scene->DestroyConstraint(c);
  }

  // Simulate. This should remove the auto-compound
  _scene->Step(0.01);

  // Confirm that all actors are in some island and in no compound
  for (auto actor : actors) {
    auto const* group = reg.try_get<CGroupMemberInfo const>(GetEntity(actor));
    EXPECT_FALSE(group);
    auto const* island = reg.try_get<CIslandMemberInfo const>(GetEntity(actor));
    EXPECT_TRUE(island);
  }
}
