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

#include <mochi_physics/src/mochi_active_collisions.h>
#include <mochi_physics/src/mochi_contact.h>
#include <mochi_physics/src/mochi_island.h>
#include <mochi_physics/src/mochi_step.h>

using namespace mochi;

namespace {

class MochiCollidingPadding : public test::MochiSceneTestBase {
 public:
  void SetUp() override {
    MochiSceneTestBase::SetUp();
    _cubeShape = test::CreateUnitCubeTetMeshShape(_mochiContext);
  }

 protected:
  static constexpr real kGravity = 10_r;
  static constexpr real kDensity = 1000_r;
  static constexpr real kPenalty = 1e8_r;
  // Much larger than the absolute padding of conservative step bounds, so that missing padding in
  // any culling stage drops the contact.
  static constexpr real kPadding = 0.05_r;
  // For the unit cube under a linear penalty, K * area * penetration = density * volume * gravity.
  static constexpr real kRestingPenetration = kDensity * kGravity / kPenalty;

  // A static collider with its top surface at y = 0, and a penalty linear in the distance.
  [[nodiscard]] Actor* CreateGround(ColliderType colliderType) {
    RigidActorParams params;
    params.colliderType = colliderType;
    params.isStatic = true;
    params.contact.penaltyThresholdDefault = 0_r;
    params.contact.penaltySmoothingHalfDistance = 0_r;
    if (colliderType == ColliderType::Plane) {
      params.shape = _mochiContext->CreatePlaneShape(Real3{0_r, 1_r, 0_r}, 0_r, test::ExpectOK{});
    } else {
      params.shape = _cubeShape;
      params.worldFromLocal.SetTranslation(Real3{0_r, -1_r, 0_r});
    }
    return _scene->CreateRigidActor(params, test::ExpectOK{});
  }

  // A dynamic unit cube with its bottom face at y = bottom.
  [[nodiscard]] Actor* CreateCube(real bottom, ContactParams const& contact, bool hasGravity) {
    RigidActorParams params;
    params.shape = _cubeShape;
    params.colliderType = ColliderType::Box;
    params.density = kDensity;
    params.hasGravity = hasGravity;
    params.boundaryElementType = ActorBoundaryElementType::P1Q1;
    params.contact = contact;
    params.contact.penaltyCoefficient = kPenalty;
    params.worldFromLocal.SetTranslation(Real3{0_r, bottom, 0_r});
    return _scene->CreateRigidActor(params, test::ExpectOK{});
  }

  // A small cube centered over the ground, with its bottom face at y = bottom. Its contact samples
  // are partitioned into spheres smaller than kPadding, so that sample culling is effective.
  [[nodiscard]] Actor* CreateSmallCube(real bottom, ContactParams const& contact) {
    real constexpr kSize = 0.02_r;
    auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
    for (Real3& x : coordinates) {
      x *= kSize;
    }
    RigidActorParams params;
    params.shape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});
    params.colliderType = ColliderType::Box;
    params.density = kDensity;
    params.hasGravity = false;
    params.boundaryElementType = ActorBoundaryElementType::ExperimentalP1Q16;
    params.contact = contact;
    params.worldFromLocal.SetTranslation(Real3{0.5_r, bottom, 0.5_r});
    return _scene->CreateRigidActor(params, test::ExpectOK{});
  }

  [[nodiscard]] real SimulateRestingCubeBottom(
      ColliderType groundType,
      ContactParams const& cubeContact,
      real initialBottom) {
    Actor* const ground = CreateGround(groundType);
    Actor* const cube = CreateCube(initialBottom, cubeContact, true);
    _scene->Step(0.1);
    real const bottom = cube->GetAabbWorld(test::ExpectOK{}).GetMin()[1];
    _scene->DestroyActor(cube->GetHandle());
    _scene->DestroyActor(ground->GetHandle());
    return bottom;
  }

  ShapeHandle _cubeShape;
};

// The colliding padding lifts the equilibrium by the padding. Starting at equilibrium isolates the
// penalty from transients.
TEST_F(MochiCollidingPadding, RestingCubeFloatsByPadding) {
  _scene->SetGravity(Real3{0_r, -kGravity, 0_r});
  test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
  real constexpr kExpectedBottom = kPadding - kRestingPenetration;

  for (ColliderType const groundType :
       {ColliderType::Plane, ColliderType::Box, ColliderType::Sdf}) {
    SCOPED_TRACE(::testing::Message() << "groundType: " << static_cast<int>(groundType));
    ContactParams cubeContact;
    cubeContact.collidingPenaltyThresholdPadding = kPadding;
    real const bottom = SimulateRestingCubeBottom(groundType, cubeContact, kExpectedBottom);
    EXPECT_NEAR(kExpectedBottom, bottom, kTolerance);
  }
}

// Without gravity, a cube is pushed away by static ground within its colliding padding, only if
// every culling stage includes the padding. Without padding, the cube does not move.
TEST_F(MochiCollidingPadding, GroundPushesCubeWithinCollidingPadding) {
  real constexpr kGap = 0.03_r;
  for (ColliderType const groundType :
       {ColliderType::Plane, ColliderType::Box, ColliderType::Sdf}) {
    for (real const padding : {0_r, kPadding}) {
      SCOPED_TRACE(
          ::testing::Message() << "groundType: " << static_cast<int>(groundType)
                               << ", padding: " << padding);
      Actor* const ground = CreateGround(groundType);
      ContactParams cubeContact;
      cubeContact.collidingPenaltyThresholdPadding = padding;
      Actor* const cube = CreateSmallCube(kGap, cubeContact);
      _scene->Step(1e-3);
      real const velocity = cube->GetLinearVelocity(test::ExpectOK{})[1];
      if (padding > 0_r) {
        EXPECT_GT(velocity, 0_r);
      } else {
        EXPECT_EQ(0_r, velocity);
      }
      _scene->DestroyActor(cube->GetHandle());
      _scene->DestroyActor(ground->GetHandle());
    }
  }
}

// The gap between the cubes exceeds the padding of their conservative step bounds, unless the
// bounds include the colliding padding of the upper cube.
TEST_F(MochiCollidingPadding, BroadPhaseFindsDynamicPairWithinCollidingPadding) {
  real constexpr kGap = 0.03_r;
  real constexpr kDt = 1e-3_r;
  Actor* const lower = CreateCube(0_r, ContactParams{}, false);
  Actor* const upper = CreateCube(1_r + kGap, ContactParams{}, false);

  // The first step relaxes the conservative step bounds of new actors. Step twice to check that
  // there is no contact without padding.
  _scene->Step(kDt);
  _scene->Step(kDt);
  EXPECT_EQ(0_r, upper->GetLinearVelocity(test::ExpectOK{})[1]);
  EXPECT_EQ(0_r, lower->GetLinearVelocity(test::ExpectOK{})[1]);

  ContactParams contact = upper->GetContactParams(test::ExpectOK{});
  contact.collidingPenaltyThresholdPadding = kPadding;
  upper->SetContactParams(contact, test::ExpectOK{});
  _scene->Step(kDt);
  EXPECT_GT(upper->GetLinearVelocity(test::ExpectOK{})[1], 0_r);
  EXPECT_LT(lower->GetLinearVelocity(test::ExpectOK{})[1], 0_r);
}

// Eroding collider bounds and expanding rotated colliding bounds can miss corner contacts. The
// cube, rotated 45 degrees, is ~0.039 from the collider's edge at (0.5, 0.5), within the pair
// threshold -0.01 + kPadding = 0.04. Eroded by 0.01, the collider's corner would be ~0.054 from the
// cube's center along the diagonal, beyond the ~0.05 reach of the padded cube bounds.
TEST_F(MochiCollidingPadding, NegativeColliderPaddingDoesNotCullRotatedCube) {
  auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
  for (Real3& x : coordinates) {
    x = (x - Real3{0.5_r, 0.5_r, 0.5_r}) * 0.002_r;
  }
  ShapeHandle const smallCubeShape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});

  for (bool const isStaticCollider : {true, false}) {
    SCOPED_TRACE(::testing::Message() << "isStaticCollider: " << isStaticCollider);
    RigidActorParams colliderParams;
    colliderParams.shape = _cubeShape;
    colliderParams.colliderType = ColliderType::Box;
    colliderParams.isStatic = isStaticCollider;
    colliderParams.hasGravity = false;
    colliderParams.worldFromLocal.SetTranslation(Real3{-0.5_r, -0.5_r, -0.5_r});
    colliderParams.contact.penaltySmoothingHalfDistance = 0_r;
    colliderParams.contact.penaltyThresholdDefault = -0.01_r;
    Actor* const collider = _scene->CreateRigidActor(colliderParams, test::ExpectOK{});

    RigidActorParams cubeParams;
    cubeParams.shape = smallCubeShape;
    cubeParams.colliderType = ColliderType::Box;
    cubeParams.hasGravity = false;
    cubeParams.boundaryElementType = ActorBoundaryElementType::P1Q1;
    cubeParams.worldFromLocal = TransformRT(
        Quaternion::FromAxisAngle(Real3{0_r, 0_r, 1_r}, kPI / 4_r), Real3{0.528_r, 0.528_r, 0_r});
    Actor* const cube = _scene->CreateRigidActor(cubeParams, test::ExpectOK{});
    auto& reg = GetRegistry();
    entt::entity const cubeEntity = GetEntity(cube);
    entt::entity const colliderEntity = GetEntity(collider);
    // Test culling directly, without solving the dynamics of this mixed-size pair.
    auto detectContact = [&]() {
      PreStepEcs(reg);
      entt::entity const island = reg.get<CIslandMemberInfo const>(cubeEntity).island;
      CollisionDetectionPipeline<TimeStep::Current>(reg, reg.get<CIslandDescendants const>(island));
      auto hasContact = [&](auto const& collisions) {
        for (ActiveCollision const& collision : collisions) {
          if (collision.colliderEntity == colliderEntity && !collision.collisionResult.Empty()) {
            return true;
          }
        }
        return false;
      };
      return isStaticCollider
          ? hasContact(reg.get<CActiveCollisions<false, TimeStep::Current> const>(cubeEntity))
          : hasContact(reg.get<CActiveCollisions<true, TimeStep::Current> const>(cubeEntity));
    };
    EXPECT_FALSE(detectContact());

    ContactParams contact = cube->GetContactParams(test::ExpectOK{});
    contact.collidingPenaltyThresholdPadding = kPadding;
    cube->SetContactParams(contact, test::ExpectOK{});
    EXPECT_TRUE(detectContact());

    _scene->DestroyActor(cube->GetHandle());
    _scene->DestroyActor(collider->GetHandle());
  }
}

} // namespace
