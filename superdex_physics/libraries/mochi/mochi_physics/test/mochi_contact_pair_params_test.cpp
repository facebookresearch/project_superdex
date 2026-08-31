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

#include <mochi_physics/src/mochi_contact_pair_params.h>

#include <array>
#include <initializer_list>

using namespace mochi;

namespace {

class MochiContactPairParams : public test::MochiSceneTestBase {
 public:
  void SetUp() override {
    MochiSceneTestBase::SetUp();
    _shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
  }

  [[nodiscard]] Actor* CreateRigidActor() {
    RigidActorParams params;
    params.shape = _shape;
    params.colliderType = ColliderType::Box;
    params.hasGravity = false;
    return _scene->CreateRigidActor(params, test::ExpectOK{});
  }

  [[nodiscard]] Actor* CreateArticulatedActor(std::initializer_list<int> parentLinks) {
    ArticulatedActorParams params;
    params.joints.reserve(parentLinks.size());
    params.links.reserve(parentLinks.size());
    for (int parentLink : parentLinks) {
      params.joints.push_back(
          {.type = parentLink < 0 ? ArticulatedJointType::Free : ArticulatedJointType::Spherical});
      params.links.push_back(
          {.parentLink = parentLink,
           .shape = _shape,
           .colliderType = ColliderType::Box,
           .hasGravity = false});
    }
    return _scene->CreateArticulatedActor(params, test::ExpectOK{});
  }

  void ExpectOverride(
      ActorHandle actorA,
      ActorHandle actorB,
      ContactPairParamsOverride const& expected) const {
    ASSERT_TRUE(_scene->HasContactPairParamsOverride(actorA, actorB, test::ExpectOK{}));
    EXPECT_EQ(expected, _scene->GetContactPairParamsOverride(actorA, actorB, test::ExpectOK{}));
  }

  void ExpectNoOverride(ActorHandle actorA, ActorHandle actorB) const {
    EXPECT_FALSE(_scene->HasContactPairParamsOverride(actorA, actorB, test::ExpectOK{}));
    [[maybe_unused]] auto const missing =
        _scene->GetContactPairParamsOverride(actorA, actorB, test::ExpectNotOK{});
  }

 protected:
  ShapeHandle _shape;
};

TEST_F(MochiContactPairParams, SetGetReplaceClearAndSelfPairs) {
  ActorHandle const actorA = CreateRigidActor()->GetHandle();
  ActorHandle const actorB = CreateRigidActor()->GetHandle();

  ContactPairParamsOverride first;
  first.penaltyCoefficient = 12_r;
  first.coulombFrictionCoefficient = 0.3_r;
  _scene->SetContactPairParamsOverride(actorA, actorB, first, test::ExpectOK{});
  ExpectOverride(actorB, actorA, first);

  ContactPairParamsOverride replacement;
  replacement.frictionFalloffVel = 0.4_r;
  _scene->SetContactPairParamsOverride(actorB, actorA, replacement, test::ExpectOK{});
  ExpectOverride(actorA, actorB, replacement);

  _scene->ClearContactPairParamsOverride(actorB, actorA, test::ExpectOK{});
  ExpectNoOverride(actorA, actorB);
  _scene->ClearContactPairParamsOverride(actorA, actorB, test::ExpectOK{});

  _scene->SetContactPairParamsOverride(actorA, actorA, first, test::ExpectOK{});
  ExpectOverride(actorA, actorA, first);
  _scene->ClearContactPairParamsOverride(actorA, actorA, test::ExpectOK{});
  ExpectNoOverride(actorA, actorA);
}

TEST_F(MochiContactPairParams, RejectsEmptyAndInvalidPatchesWithoutMutation) {
  ActorHandle const actorA = CreateRigidActor()->GetHandle();
  ActorHandle const actorB = CreateRigidActor()->GetHandle();
  ContactPairParamsOverride baseline;
  baseline.penaltyCoefficient = 9_r;
  _scene->SetContactPairParamsOverride(actorA, actorB, baseline, test::ExpectOK{});

  std::array<ContactPairParamsOverride, 6> invalid;
  invalid[1].penaltyCoefficient = 0_r;
  invalid[2].frictionFalloffVel = -1_r;
  invalid[3].viscousFrictionCoefficient = -1_r;
  invalid[4].coulombFrictionCoefficient = -1_r;
  invalid[5].normalViscousDampingCoefficient = -1_r;
  for (auto const& params : invalid) {
    _scene->SetContactPairParamsOverride(actorA, actorB, params, test::ExpectNotOK{});
    ExpectOverride(actorA, actorB, baseline);
  }

  ContactPairParamsOverride validZeroes;
  validZeroes.frictionFalloffVel = 0_r;
  validZeroes.viscousFrictionCoefficient = 0_r;
  validZeroes.coulombFrictionCoefficient = 0_r;
  validZeroes.normalViscousDampingCoefficient = 0_r;
  _scene->SetContactPairParamsOverride(actorA, actorB, validZeroes, test::ExpectOK{});
  ExpectOverride(actorA, actorB, validZeroes);
}

TEST_F(MochiContactPairParams, RejectsInvalidHandlesWithoutMutation) {
  ActorHandle const actorA = CreateRigidActor()->GetHandle();
  ActorHandle const actorB = CreateRigidActor()->GetHandle();
  ContactPairParamsOverride params;
  params.penaltyCoefficient = 7_r;

  _scene->SetContactPairParamsOverride(actorA, actorB, params, test::ExpectOK{});
  _scene->SetContactPairParamsOverride(actorA, ActorHandle{}, params, test::ExpectNotOK{});
  _scene->ClearContactPairParamsOverride(actorA, ActorHandle{}, test::ExpectNotOK{});
  ExpectOverride(actorA, actorB, params);

  EXPECT_FALSE(_scene->HasContactPairParamsOverride(actorA, ActorHandle{}, test::ExpectNotOK{}));
  [[maybe_unused]] auto const invalidGet =
      _scene->GetContactPairParamsOverride(actorA, ActorHandle{}, test::ExpectNotOK{});
}

TEST_F(MochiContactPairParams, RejectsActorsWithoutContactParams) {
  ActorHandle const actor = CreateRigidActor()->GetHandle();
  ArticulatedActorParams articulatedParams;
  articulatedParams.joints.push_back({.type = ArticulatedJointType::Free});
  articulatedParams.links.push_back({.parentLink = -1});
  Actor* const articulation = _scene->CreateArticulatedActor(articulatedParams, test::ExpectOK{});
  ActorHandle const shapeLessLink = articulation->GetNestedLinkActors(test::ExpectOK{})[0];

  ContactPairParamsOverride params;
  params.penaltyCoefficient = 7_r;
  _scene->SetContactPairParamsOverride(shapeLessLink, actor, params, test::ExpectNotOK{});
  _scene->SetContactPairParamsOverride(
      articulation->GetHandle(), actor, params, test::ExpectNotOK{});

  auto const& table = test::GetRegistry(_scene).ctx<CContactPairParamsOverrideTable const>();
  EXPECT_TRUE(table.Empty());
}

TEST_F(MochiContactPairParams, StaleHandleFailureDoesNotMutateOtherPairs) {
  ActorHandle const actorA = CreateRigidActor()->GetHandle();
  ActorHandle const actorB = CreateRigidActor()->GetHandle();
  ActorHandle const stale = CreateRigidActor()->GetHandle();
  _scene->DestroyActor(stale);

  ContactPairParamsOverride baseline;
  baseline.penaltyCoefficient = 5_r;
  _scene->SetContactPairParamsOverride(actorA, actorB, baseline, test::ExpectOK{});

  ContactPairParamsOverride replacement;
  replacement.penaltyCoefficient = 11_r;
  _scene->SetContactPairParamsOverride(actorA, stale, replacement, test::ExpectNotOK{});
  ExpectOverride(actorA, actorB, baseline);

  _scene->ClearContactPairParamsOverride(actorA, stale, test::ExpectNotOK{});
  ExpectOverride(actorA, actorB, baseline);
}

TEST_F(MochiContactPairParams, DestroyingActorsRemovesOverridesBeforeEntityReuse) {
  ActorHandle const actorA = CreateRigidActor()->GetHandle();
  ActorHandle const actorB = CreateRigidActor()->GetHandle();
  ActorHandle const actorC = CreateRigidActor()->GetHandle();
  ActorHandle const actorD = CreateRigidActor()->GetHandle();
  ContactPairParamsOverride params;
  params.penaltyCoefficient = 6_r;
  _scene->SetContactPairParamsOverride(actorA, actorB, params, test::ExpectOK{});
  ContactPairParamsOverride survivingParams;
  survivingParams.frictionFalloffVel = 0.6_r;
  _scene->SetContactPairParamsOverride(actorC, actorD, survivingParams, test::ExpectOK{});

  entt::entity const destroyedEntity = ExtractEntity(actorA);
  _scene->DestroyActor(actorA);
  ActorHandle const replacement = CreateRigidActor()->GetHandle();
  EXPECT_EQ(
      entt::entt_traits<entt::entity>::to_entity(destroyedEntity),
      entt::entt_traits<entt::entity>::to_entity(ExtractEntity(replacement)));
  EXPECT_NE(actorA, replacement);
  auto const& table = test::GetRegistry(_scene).ctx<CContactPairParamsOverrideTable const>();
  EXPECT_EQ(nullptr, table.Find(destroyedEntity, ExtractEntity(actorB)));
  ExpectNoOverride(replacement, actorB);
  ExpectOverride(actorC, actorD, survivingParams);
}

TEST_F(MochiContactPairParams, DestroyingArticulationRemovesNestedOverrides) {
  Actor* const parent = CreateArticulatedActor({-1, 0});
  ActorHandle const nested = parent->GetNestedLinkActors(test::ExpectOK{})[0];
  entt::entity const nestedEntity = ExtractEntity(nested);
  ActorHandle const actor = CreateRigidActor()->GetHandle();
  ActorHandle const actorC = CreateRigidActor()->GetHandle();
  ActorHandle const actorD = CreateRigidActor()->GetHandle();
  ContactPairParamsOverride params;
  params.penaltyCoefficient = 8_r;
  _scene->SetContactPairParamsOverride(nested, actor, params, test::ExpectOK{});
  ContactPairParamsOverride survivingParams;
  survivingParams.normalViscousDampingCoefficient = 0.8_r;
  _scene->SetContactPairParamsOverride(actorC, actorD, survivingParams, test::ExpectOK{});

  _scene->DestroyActor(parent);

  auto const& table = test::GetRegistry(_scene).ctx<CContactPairParamsOverrideTable const>();
  EXPECT_EQ(nullptr, table.Find(nestedEntity, ExtractEntity(actor)));
  ExpectOverride(actorC, actorD, survivingParams);
}

TEST_F(MochiContactPairParams, StateRestoreDoesNotModifyOverrides) {
  ActorHandle const actorA = CreateRigidActor()->GetHandle();
  ActorHandle const actorB = CreateRigidActor()->GetHandle();
  ContactPairParamsOverride captured;
  captured.penaltyCoefficient = 4_r;
  _scene->SetContactPairParamsOverride(actorA, actorB, captured, test::ExpectOK{});
  StateHandle const state = _scene->CaptureState(test::ExpectOK{});

  ContactPairParamsOverride current;
  current.frictionFalloffVel = 0.2_r;
  _scene->SetContactPairParamsOverride(actorA, actorB, current, test::ExpectOK{});
  _scene->RestoreState(state, /*releaseImmediately=*/true, test::ExpectOK{});
  ExpectOverride(actorA, actorB, current);
}

TEST_F(MochiContactPairParams, PenaltyOverrideControlsSimulationPenetration) {
  real constexpr kGravity = 10_r;
  real constexpr kDensity = 1000_r;
  real constexpr kActorPenalty = 1e8_r;
  real constexpr kOverridePenalty = 1e6_r;
  // For the unit cube, K * area * penetration = density * volume * gravity, so the equilibrium
  // penetration is density * gravity / K. Starting there isolates the penalty from transients.
  real constexpr kExpectedPenetration = kDensity * kGravity / kOverridePenalty;

  _scene->SetGravity(Real3{0_r, -kGravity, 0_r});
  test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);

  ShapeHandle const planeShape =
      _mochiContext->CreatePlaneShape(Real3{0_r, 1_r, 0_r}, 0_r, test::ExpectOK{});
  RigidActorParams planeParams;
  planeParams.shape = planeShape;
  planeParams.colliderType = ColliderType::Plane;
  planeParams.isStatic = true;
  // Threshold and smoothing are collider-owned. Zeroing them makes the penalty exactly linear in
  // penetration, so the equilibrium above is analytic.
  planeParams.contact.penaltyThresholdDefault = 0_r;
  planeParams.contact.penaltySmoothingHalfDistance = 0_r;
  Actor* const plane = _scene->CreateRigidActor(planeParams, test::ExpectOK{});

  RigidActorParams boxParams;
  boxParams.shape = _shape;
  boxParams.colliderType = ColliderType::Box;
  boxParams.density = kDensity;
  boxParams.boundaryElementType = ActorBoundaryElementType::P1Q1;
  boxParams.contact.penaltyCoefficient = kActorPenalty;
  boxParams.worldFromLocal.SetTranslation(Real3{0_r, -kExpectedPenetration, 0_r});
  Actor* const box = _scene->CreateRigidActor(boxParams, test::ExpectOK{});

  ContactPairParamsOverride paramsOverride;
  paramsOverride.penaltyCoefficient = kOverridePenalty;
  _scene->SetContactPairParamsOverride(
      box->GetHandle(), plane->GetHandle(), paramsOverride, test::ExpectOK{});

  _scene->Step(0.1);

  real const penetration = -box->GetAabbWorld(test::ExpectOK{}).GetMin()[1];
  EXPECT_NEAR(kExpectedPenetration, penetration, kTolerance);
}

} // namespace
