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

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_physics/src/mochi_common_components.h>

#include <gtest/gtest.h>

using namespace mochi;

namespace {
class ConservativeStepBoundsVelocityTest : public test::MochiSceneTestBase {
 protected:
  void ExpectBoundsContainPostStepGeometry(entt::entity actor) {
    auto const& reg = GetRegistry();
    Aabb const bounds = reg.get<CConservativeStepBounds const>(actor).worldAabb;
    auto const& root = reg.get<CRootTransform const>(actor);
    auto const& currentBounds = reg.get<CBoundingVolume<TimeStep::Current> const>(actor);
    Aabb const postStep = GetAabb(TransformShape(root.worldFromLocal, currentBounds.localShape));
    EXPECT_TRUE(ContainsPoint(bounds, postStep.GetMin()));
    EXPECT_TRUE(ContainsPoint(bounds, postStep.GetMax()));
  }
};
} // namespace

TEST_F(ConservativeStepBoundsVelocityTest, RigidLinearVelocity) {
  RigidActorParams params;
  params.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
  params.colliderType = ColliderType::None;
  params.hasGravity = false;
  Actor* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
  entt::entity const actorEntity = GetEntity(actor);

  actor->SetVelocity({20_r, 0_r, 0_r}, {}, test::ExpectOK{});
  GetRegistry().get<CConservativeStepBounds>(actorEntity).needsNextStepRelaxation = false;

  _scene->Step(1e-2);

  ExpectBoundsContainPostStepGeometry(actorEntity);
}

TEST_F(ConservativeStepBoundsVelocityTest, RigidAngularVelocity) {
  RigidActorParams params;
  params.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
  params.colliderType = ColliderType::None;
  params.hasGravity = false;
  params.momentOfInertia = Real6{10_r, 0_r, 0_r, 10_r, 0_r, 10_r};

  // Offset center-of-mass rotation exercises the AABB-center velocity contribution.
  params.centerOfMass = Real3{3_r, 0.5_r, 0.5_r};
  Actor* offsetActor = _scene->CreateRigidActor(params, test::ExpectOK{});
  entt::entity const offsetActorEntity = GetEntity(offsetActor);
  offsetActor->SetVelocity({}, {0_r, 0_r, 10_r}, test::ExpectOK{});
  GetRegistry().get<CConservativeStepBounds>(offsetActorEntity).needsNextStepRelaxation = false;

  // Centered rotation at this speed makes containment depend on the velocity-radius contribution.
  params.centerOfMass = Real3{0.5_r, 0.5_r, 0.5_r};
  Actor* centeredActor = _scene->CreateRigidActor(params, test::ExpectOK{});
  entt::entity const centeredActorEntity = GetEntity(centeredActor);
  centeredActor->SetVelocity({}, {0_r, 0_r, 20_r}, test::ExpectOK{});
  GetRegistry().get<CConservativeStepBounds>(centeredActorEntity).needsNextStepRelaxation = false;

  _scene->Step(1e-2);

  ExpectBoundsContainPostStepGeometry(offsetActorEntity);
  ExpectBoundsContainPostStepGeometry(centeredActorEntity);
}

TEST_F(ConservativeStepBoundsVelocityTest, ArticulatedLinkJointVelocity) {
  ShapeHandle const shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
  ArticulatedActorParams params;
  params.joints = {
      {.type = ArticulatedJointType::Hard},
      {.type = ArticulatedJointType::Revolute, .axis = {0_r, 0_r, 1_r}}};
  params.links = {
      {.parentLink = -1, .shape = shape, .colliderType = ColliderType::None, .hasGravity = false},
      {.parentLink = 0, .shape = shape, .colliderType = ColliderType::None, .hasGravity = false}};
  Actor* actor = _scene->CreateArticulatedActor(params, test::ExpectOK{});
  auto const& links = actor->GetNestedLinkActors(test::ExpectOK{});
  ASSERT_EQ(2, isize(links));

  DynamicArray<real> const velocities = {20_r};
  actor->SetArticulatedJointVelocities(velocities, test::ExpectOK{});
  for (ActorHandle const link : links) {
    GetRegistry().get<CConservativeStepBounds>(GetEntity(link)).needsNextStepRelaxation = false;
  }

  _scene->Step(1e-2);

  ExpectBoundsContainPostStepGeometry(GetEntity(links[1]));
}
