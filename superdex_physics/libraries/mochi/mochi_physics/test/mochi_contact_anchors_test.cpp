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

// These tests inspect the friction anchor component and contact results, which are not exposed
// through the public API.
#include <mochi_physics/src/mochi_contact.h>
#include <mochi_physics/src/mochi_contact_anchors.h>

#include <mochi_core/integration/integration_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/reflection.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <utility>

using namespace mochi;

// Contacts of a colliding actor with one collider, with their anchor offsets, at positions twice
// those offsets, so that the anchors rebuilt from them are at points equal to the offsets.
static ActiveCollision MakeCollision(
    entt::entity collider,
    DynamicArray<int> const& samples,
    DynamicArray<Real3> const& offsets) {
  ActiveCollision collision;
  collision.colliderEntity = collider;
  collision.collisionResult.sampleIndices = samples;
  collision.collisionResult.anchorOffset = offsets;
  for (Real3 const& offset : offsets) {
    collision.collisionResult.posColliding.push_back(2_r * offset);
  }
  return collision;
}

// Point of the sample's anchor on the collider, or zero: the lookup by index that
// FillFrictionAnchorOffsets falls back to.
static Real3 FindAnchorPoint(
    CFrictionAnchors const& anchorsComp,
    CFrictionAnchorIndex const& index,
    int sample,
    entt::entity collider) {
  auto const& anchors = anchorsComp.anchors;
  auto const colliderId = entt::to_integral(collider);
  for (int i = index[sample];
       i != kSentinelIndex && i < isize(anchors) && anchors[i].sample == sample;
       ++i) {
    if (anchors[i].collider == colliderId) {
      return anchors[i].point;
    }
  }
  return Real3{};
}

// Expects the anchors, field by field and in order.
static void ExpectAnchors(
    DynamicArray<FrictionAnchor> const& expected,
    DynamicArray<FrictionAnchor> const& actual) {
  ASSERT_EQ(isize(expected), isize(actual));
  for (int i = 0; i < isize(expected); ++i) {
    EXPECT_EQ(expected[i].sample, actual[i].sample) << "anchor " << i;
    EXPECT_EQ(expected[i].collider, actual[i].collider) << "anchor " << i;
    EXPECT_EQ(expected[i].point, actual[i].point) << "anchor " << i;
  }
}

// Expects the index entries of all samples, including kSentinelIndex for samples never indexed.
static void ExpectIndex(CFrictionAnchorIndex const& index, DynamicArray<int> const& expected) {
  ASSERT_EQ(isize(expected), isize(index));
  for (int s = 0; s < isize(expected); ++s) {
    EXPECT_EQ(expected[s], index[s]) << "sample " << s;
  }
}

namespace {
class FrictionAnchors : public test::MochiSceneTestBase {
 protected:
  // The anchor components only exist with elasto-plastic friction enabled.
  void SetUp() override {
    MochiSceneTestBase::SetUp();
    SetElastoPlasticFriction(true);
  }

  void SetElastoPlasticFriction(bool enabled) {
    SolverParams params = _scene->GetSolverParams();
    params.experimentalEval.elastoPlasticFriction = enabled;
    _scene->SetSolverParams(params, test::ExpectOK{});
  }

  // Static plane at y = 0. The unit cube mesh has a corner at its origin, so actors created with
  // the identity transform rest on it. Normal contact damping applies only if both actors have it.
  void CreatePlane(real normalDamping = 0_r) {
    RigidActorParams params;
    params.name = "Plane";
    params.isStatic = true;
    params.shape = _mochiContext->CreatePlaneShape(Real3{0_r, 1_r, 0_r}, 0_r, test::ExpectOK{});
    params.contact.normalViscousDampingCoefficient = normalDamping;
    _plane = _scene->CreateRigidActor(params, test::ExpectOK{});
  }

  Actor* CreateRigidCube(real normalDamping = 0_r) {
    RigidActorParams params;
    params.name = "RigidCube";
    params.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
    params.contact.normalViscousDampingCoefficient = normalDamping;
    return _scene->CreateRigidActor(params, test::ExpectOK{});
  }

  Actor* CreateSoftCube() {
    SoftActorParams params;
    params.name = "SoftCube";
    params.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
    return _scene->CreateSoftActor(params, test::ExpectOK{});
  }

  void Step(int numSteps) {
    for (int i = 0; i < numSteps; ++i) {
      _scene->Step(1e-2_r);
    }
  }

  CFrictionAnchors& GetAnchors(Actor* actor) {
    return GetRegistry().get<CFrictionAnchors>(GetEntity(actor));
  }

  CFrictionAnchorIndex& GetIndex(Actor* actor) {
    return GetRegistry().get<CFrictionAnchorIndex>(GetEntity(actor));
  }

  // Rebuilds the anchors from the given contacts, as after a converged step.
  void RebuildAnchors(
      CFrictionAnchors& anchors,
      CFrictionAnchorIndex& index,
      DynamicArray<ActiveCollision> const& asyncCollisions,
      DynamicArray<ActiveCollision> const& syncCollisions = {}) {
    CActiveCollisions</*kIsSync*/ false, TimeStep::Current> asyncActive(/*numPartitions*/ 1);
    CActiveCollisions</*kIsSync*/ true, TimeStep::Current> syncActive(/*numPartitions*/ 1);
    ActivateCollisions(asyncCollisions, asyncActive);
    ActivateCollisions(syncCollisions, syncActive);
    UpdateFrictionAnchors(
        GetRegistry(), /*convergence*/ nullptr, asyncActive, syncActive, anchors, index);
  }

  template <bool kIsSync>
  static void ActivateCollisions(
      DynamicArray<ActiveCollision> const& collisions,
      CActiveCollisions<kIsSync, TimeStep::Current>& outActive) {
    for (auto const& collision : collisions) {
      outActive.ActivatePartitionResult(collision.colliderEntity, /*partitionId*/ 0) =
          collision.collisionResult;
    }
    outActive.FinalizeActiveCollisions();
  }

  // Rebuilds anchors of an actor with 4 samples, in which samples 0 and 2 have anchors on both
  // colliders a and b. They are interleaved across the async and sync contacts, so the rebuild has
  // to group them. Points: (2, a) = 1, (0, a) = 4, (0, b) = 2, (2, b) = 3, along x.
  void RebuildInterleavedAnchors(
      entt::entity a,
      entt::entity b,
      CFrictionAnchors& anchors,
      CFrictionAnchorIndex& index) {
    RebuildAnchors(
        anchors,
        index,
        {MakeCollision(a, {2, 0}, {Real3{1_r, 0_r, 0_r}, Real3{4_r, 0_r, 0_r}})},
        {MakeCollision(b, {0, 2}, {Real3{2_r, 0_r, 0_r}, Real3{3_r, 0_r, 0_r}})});
  }

  // Expects the actor to have both anchor components, with an index entry per contact sample.
  void ExpectAnchorComponents(Actor* actor) {
    auto& reg = GetRegistry();
    entt::entity const e = GetEntity(actor);
    ASSERT_TRUE((reg.all_of<CFrictionAnchors, CFrictionAnchorIndex>(e))) << actor->GetName();
    EXPECT_EQ(
        isize(reg.get<CContactSamples<TimeStep::Current> const>(e).positions),
        isize(reg.get<CFrictionAnchorIndex const>(e)))
        << actor->GetName();
  }

  bool HasAnyAnchorComponent(Actor* actor) {
    return GetRegistry().any_of<CFrictionAnchors, CFrictionAnchorIndex>(GetEntity(actor));
  }

  // The plane is static, so the contacts of an actor with it are async.
  CActiveCollisions</*kIsSync*/ false, TimeStep::Current> const& GetCollisions(Actor* actor) {
    return GetRegistry().get<CActiveCollisions</*kIsSync*/ false, TimeStep::Current> const>(
        GetEntity(actor));
  }

  // Colliding sample and collider-space position of the deepest contact of the actor with the
  // plane.
  std::pair<int, Real3> DeepestContactOnPlane(Actor* actor) {
    std::pair<int, Real3> deepest{-1, Real3{}};
    real minDistance = std::numeric_limits<real>::max();
    for (auto const& collision : GetCollisions(actor)) {
      if (collision.colliderEntity != GetEntity(_plane)) {
        continue;
      }
      auto const& result = collision.collisionResult;
      for (int i = 0; i < isize(result.sampleIndices); ++i) {
        if (result.sdfInfo.val[i] < minDistance) {
          minDistance = result.sdfInfo.val[i];
          deepest = {result.sampleIndices[i], result.posColliding[i]};
        }
      }
    }
    return deepest;
  }

  // Collider-space position of the actor's contact of the sample with the plane.
  Real3 ContactPositionOnPlane(Actor* actor, int sample) {
    for (auto const& collision : GetCollisions(actor)) {
      if (collision.colliderEntity != GetEntity(_plane)) {
        continue;
      }
      auto const& result = collision.collisionResult;
      for (int i = 0; i < isize(result.sampleIndices); ++i) {
        if (result.sampleIndices[i] == sample) {
          return result.posColliding[i];
        }
      }
    }
    ADD_FAILURE() << "No contact of sample " << sample << " with the plane.";
    return Real3{};
  }

  // Seeds an anchor beyond the presliding displacement along -x on the plane, for the deepest
  // contact of a resting actor, and steps once. Collision detection measures the contacts of that
  // sample from it, so they slide, and the advanced anchor trails the contact by the presliding
  // displacement along the plane, mostly along x (a sample of a soft actor also moves sideways
  // under the pull), on the line from the seeded anchor to the contact: sliding scales the whole
  // displacement from the anchor. Without the anchor, the contact would only move by the motion of
  // the step, which is far smaller.
  void ExpectSeededAnchorAdvances(Actor* actor) {
    Step(20);
    auto const [sample, position] = DeepestContactOnPlane(actor);
    ASSERT_GE(sample, 0);
    real const delta = ContactParams{}.preslidingDisplacement;
    entt::entity const plane = GetEntity(_plane);
    Real3 const seededOffset{3_r * delta, 0_r, 0_r};
    ActiveCollision seeded = MakeCollision(plane, {sample}, {seededOffset});
    seeded.collisionResult.posColliding = {position};
    RebuildAnchors(GetAnchors(actor), GetIndex(actor), {seeded});

    Step(1);
    Real3 const anchorPoint = FindAnchorPoint(GetAnchors(actor), GetIndex(actor), sample, plane);
    Real3 const contact = ContactPositionOnPlane(actor, sample);
    Real3 const offset = contact - anchorPoint;
    EXPECT_NEAR(delta, Norm(Real3{offset[0], 0_r, offset[2]}), 1e-3_r * delta);
    EXPECT_GT(offset[0], 0.9_r * delta);
    Real3 const displacement = contact - (position - seededOffset);
    real const scale = delta / Norm(Real3{displacement[0], 0_r, displacement[2]});
    EXPECT_NEAR_TOL(scale * displacement, offset, 1e-3_r * delta);
  }

  // Creates the plane and a rigid cube resting on it, with gravity tilted along x, so that the
  // tangential load is loadRatio times the Coulomb limit.
  Actor* CreateCubeUnderTangentialLoad(real loadRatio, real normalDamping = 0_r) {
    real constexpr kGravity = 9.81_r;
    real const mu = ContactParams{}.coulombFrictionCoefficient;
    _scene->SetGravity(Real3{loadRatio * mu * kGravity, -kGravity, 0_r});
    CreatePlane(normalDamping);
    return CreateRigidCube(normalDamping);
  }

  // Steps a stack of numCubes rigid cubes resting on the plane with gravity tilted along x, so that
  // the tangential load of each contact is loadRatio times the Coulomb limit, and returns the
  // distance the top cube travels along x in the second half of numSteps, relative to what it rests
  // on: the cubes below tilt slightly under the load.
  real SlideUnderTangentialLoad(
      real loadRatio,
      int numSteps,
      real normalDamping = 0_r,
      int numCubes = 1) {
    Actor* cube = CreateCubeUnderTangentialLoad(loadRatio, normalDamping);
    Actor* below = nullptr;
    for (int i = 1; i < numCubes; ++i) {
      below = cube;
      cube = CreateRigidCube(normalDamping);
      cube->SetRootTransform(TransformRT{Real3{0_r, static_cast<real>(i), 0_r}}, test::ExpectOK{});
    }
    auto const relativeX = [&] {
      TransformRT const worldFromBelow = below ? below->GetRootTransform() : TransformRT{};
      return worldFromBelow.TransformPointInverse(cube->GetRootTransform().GetTranslation())[0];
    };
    Step(numSteps / 2);
    real const x0 = relativeX();
    Step(numSteps - numSteps / 2);
    return relativeX() - x0;
  }

  Actor* _plane = nullptr;
};

// Elasto-plastic friction with each integrator that it supports.
class FrictionAnchorsIntegrator : public FrictionAnchors,
                                  public ::testing::WithParamInterface<IntegrationMethod> {
 protected:
  void SetUp() override {
    FrictionAnchors::SetUp();
    test::SetSceneIntegrationMethod(_scene, GetParam());
  }
};
} // namespace

static std::string IntegratorName(::testing::TestParamInfo<IntegrationMethod> const& info) {
  return std::string(SReflect::EnumToString(info.param));
}

// When the anchors of each sample are adjacent in contact order, the rebuild appends them in that
// order in a single pass.
TEST_F(FrictionAnchors, RebuildAppendsAdjacentAnchorsInContactOrder) {
  CreatePlane();
  entt::entity const a = GetEntity(_plane);
  entt::entity const b = GetEntity(CreateRigidCube());
  CFrictionAnchors anchors;
  CFrictionAnchorIndex index(3);
  RebuildAnchors(
      anchors,
      index,
      {MakeCollision(a, {0, 1}, {Real3{1_r, 0_r, 0_r}, Real3{2_r, 0_r, 0_r}})},
      {MakeCollision(b, {1}, {Real3{3_r, 0_r, 0_r}})});

  ExpectIndex(index, {0, 1, kSentinelIndex});
  uint32_t const ia = entt::to_integral(a);
  uint32_t const ib = entt::to_integral(b);
  DynamicArray<FrictionAnchor> const expectedAnchors{
      {.sample = 0, .collider = ia, .point = Real3{1_r, 0_r, 0_r}},
      {.sample = 1, .collider = ia, .point = Real3{2_r, 0_r, 0_r}},
      {.sample = 1, .collider = ib, .point = Real3{3_r, 0_r, 0_r}}};
  ExpectAnchors(expectedAnchors, anchors.anchors);
  EXPECT_EQ((Real3{2_r, 0_r, 0_r}), FindAnchorPoint(anchors, index, 1, a));
  EXPECT_EQ((Real3{3_r, 0_r, 0_r}), FindAnchorPoint(anchors, index, 1, b));
}

// When a sample's anchors are not adjacent in contact order, the rebuild groups them by sample,
// groups in order of first appearance. The order within a sample is unspecified, which lookups do
// not depend on: a sample has at most one anchor per rigid collider.
TEST_F(FrictionAnchors, RebuildGroupsBySample) {
  CreatePlane();
  entt::entity const a = GetEntity(_plane);
  entt::entity const b = GetEntity(CreateRigidCube());
  CFrictionAnchors anchors;
  CFrictionAnchorIndex index(4);
  RebuildInterleavedAnchors(a, b, anchors, index);

  ExpectIndex(index, {2, kSentinelIndex, 0, kSentinelIndex});
  ASSERT_EQ(4, isize(anchors.anchors));
  DynamicArray<int> const expectedSamples{2, 2, 0, 0};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(expectedSamples[i], anchors.anchors[i].sample) << "anchor " << i;
  }
  EXPECT_EQ((Real3{4_r, 0_r, 0_r}), FindAnchorPoint(anchors, index, 0, a));
  EXPECT_EQ((Real3{2_r, 0_r, 0_r}), FindAnchorPoint(anchors, index, 0, b));
  EXPECT_EQ((Real3{1_r, 0_r, 0_r}), FindAnchorPoint(anchors, index, 2, a));
  EXPECT_EQ((Real3{3_r, 0_r, 0_r}), FindAnchorPoint(anchors, index, 2, b));
  EXPECT_EQ(Real3{}, FindAnchorPoint(anchors, index, 1, a));
  EXPECT_EQ(Real3{}, FindAnchorPoint(anchors, index, 3, b));
}

// The fill finds the anchor of each contact when samples have anchors on several colliders and the
// contacts come in a different order than when the anchors were rebuilt, so that the hint misses
// and the lookup has to skip the anchor of the other collider. The offset of a contact with an
// anchor is its stage-start position relative to the anchor point. Contacts without an anchor get
// zero, and contacts with a deformable collider get no offsets.
TEST_F(FrictionAnchors, FillFindsAnchorsOfSamplesWithSeveralColliders) {
  CreatePlane();
  entt::entity const a = GetEntity(_plane);
  entt::entity const b = GetEntity(CreateRigidCube());
  entt::entity const soft = GetEntity(CreateSoftCube());
  CFrictionAnchors anchors;
  CFrictionAnchorIndex index(4);
  RebuildInterleavedAnchors(a, b, anchors, index);

  DynamicArray<ActiveCollision> collisions{
      MakeCollision(b, {2, 0, 1}, {}),
      MakeCollision(a, {0, 3, 2}, {}),
      MakeCollision(soft, {0}, {})};
  Real3 const stageStart{10_r, 0_r, 0_r};
  for (auto& collision : collisions) {
    auto& result = collision.collisionResult;
    result.posCollidingStageStart.resize(result.sampleIndices.size(), stageStart);
  }
  FillFrictionAnchorOffsets(GetRegistry(), index, anchors.anchors, MakeSpan(collisions));

  DynamicArray<Real3> const expectedB{Real3{7_r, 0_r, 0_r}, Real3{8_r, 0_r, 0_r}, Real3{}};
  DynamicArray<Real3> const expectedA{Real3{6_r, 0_r, 0_r}, Real3{}, Real3{9_r, 0_r, 0_r}};
  EXPECT_EQ(expectedB, collisions[0].collisionResult.anchorOffset);
  EXPECT_EQ(expectedA, collisions[1].collisionResult.anchorOffset);
  EXPECT_TRUE(collisions[2].collisionResult.anchorOffset.empty());
}

// Anchors store the collider's entity with its version, so a new collider that recycles the entity
// of a destroyed one does not inherit its anchors: its contacts get zero offsets.
TEST_F(FrictionAnchors, FillIgnoresAnchorsOfRecycledColliders) {
  CreatePlane();
  entt::entity const a = GetEntity(_plane);
  Actor* cube = CreateRigidCube();
  entt::entity const b = GetEntity(cube);
  CFrictionAnchors anchors;
  CFrictionAnchorIndex index(4);
  RebuildInterleavedAnchors(a, b, anchors, index);

  _scene->DestroyActor(cube);
  entt::entity const recycled = GetEntity(CreateRigidCube());
  ASSERT_EQ(
      entt::entt_traits<entt::entity>::to_entity(b),
      entt::entt_traits<entt::entity>::to_entity(recycled));
  ASSERT_NE(b, recycled);

  ActiveCollision collision = MakeCollision(recycled, {2, 0}, {});
  collision.collisionResult.posCollidingStageStart.resize(2, Real3{10_r, 0_r, 0_r});
  FillFrictionAnchorOffsets(
      GetRegistry(), index, anchors.anchors, Span<ActiveCollision>(&collision, size_t{1}));

  EXPECT_EQ((DynamicArray<Real3>{Real3{}, Real3{}}), collision.collisionResult.anchorOffset);
}

// The fill and the rebuild are inverse: anchors filled into contacts that have not moved since
// stage start are rebuilt at the same points.
TEST_F(FrictionAnchors, RebuildOfFilledOffsetsKeepsAnchorPoints) {
  CreatePlane();
  entt::entity const plane = GetEntity(_plane);
  CFrictionAnchors anchors;
  CFrictionAnchorIndex index(2);
  RebuildAnchors(
      anchors, index, {MakeCollision(plane, {0, 1}, {Real3{1_r, 0_r, 0_r}, Real3{0_r, 0_r, 2_r}})});

  ActiveCollision collision = MakeCollision(plane, {1, 0}, {});
  collision.collisionResult.posCollidingStageStart = {Real3{5_r, 6_r, 7_r}, Real3{-1_r, 0_r, 3_r}};
  collision.collisionResult.posColliding = collision.collisionResult.posCollidingStageStart;
  FillFrictionAnchorOffsets(
      GetRegistry(), index, anchors.anchors, Span<ActiveCollision>(&collision, size_t{1}));
  RebuildAnchors(anchors, index, {collision});

  EXPECT_EQ(2, isize(anchors.anchors));
  EXPECT_EQ((Real3{1_r, 0_r, 0_r}), FindAnchorPoint(anchors, index, 0, plane));
  EXPECT_EQ((Real3{0_r, 0_r, 2_r}), FindAnchorPoint(anchors, index, 1, plane));
}

// Only nonzero offsets of contacts with rigid colliders become anchors.
TEST_F(FrictionAnchors, RebuildKeepsNonzeroOffsetsWithRigidColliders) {
  CreatePlane();
  entt::entity const plane = GetEntity(_plane);
  entt::entity const soft = GetEntity(CreateSoftCube());
  CFrictionAnchors anchors;
  CFrictionAnchorIndex index(4);
  RebuildAnchors(
      anchors,
      index,
      {MakeCollision(plane, {0, 1, 2}, {Real3{1_r, 0_r, 0_r}, Real3{}, Real3{2_r, 0_r, 0_r}}),
       MakeCollision(soft, {3}, {Real3{3_r, 0_r, 0_r}})});

  ExpectIndex(index, {0, kSentinelIndex, 1, kSentinelIndex});
  EXPECT_EQ(2, isize(anchors.anchors));
  EXPECT_EQ((Real3{1_r, 0_r, 0_r}), FindAnchorPoint(anchors, index, 0, plane));
  EXPECT_EQ((Real3{2_r, 0_r, 0_r}), FindAnchorPoint(anchors, index, 2, plane));
}

TEST_F(FrictionAnchors, RebuildReusesCapacity) {
  CreatePlane();
  entt::entity const plane = GetEntity(_plane);
  CFrictionAnchors anchors;
  CFrictionAnchorIndex index(3);
  RebuildAnchors(
      anchors,
      index,
      {MakeCollision(
          plane, {0, 1, 2}, {Real3{1_r, 0_r, 0_r}, Real3{2_r, 0_r, 0_r}, Real3{3_r, 0_r, 0_r}})});
  auto const* storage = anchors.anchors.data();

  // The index keeps stale entries for samples 0 and 2, which lookups reject.
  RebuildAnchors(anchors, index, {MakeCollision(plane, {1}, {Real3{1_r, 2_r, 3_r}})});
  EXPECT_EQ(storage, anchors.anchors.data());
  EXPECT_EQ((Real3{1_r, 2_r, 3_r}), FindAnchorPoint(anchors, index, 1, plane));
  EXPECT_EQ(Real3{}, FindAnchorPoint(anchors, index, 0, plane));
  EXPECT_EQ(Real3{}, FindAnchorPoint(anchors, index, 2, plane));

  RebuildAnchors(anchors, index, {});
  EXPECT_TRUE(anchors.anchors.empty());
  EXPECT_EQ(Real3{}, FindAnchorPoint(anchors, index, 1, plane));
}

// Elasto-plastic friction requires the C1Regularized friction model: SetSolverParams rejects
// CinfRegularized with it, and keeps the previous params.
TEST_F(FrictionAnchors, RequiresC1RegularizedFriction) {
  SolverParams params = _scene->GetSolverParams();
  ASSERT_TRUE(params.experimentalEval.elastoPlasticFriction);
  params.experimentalEval.frictionModel = CoulombFrictionModel::CinfRegularized;
  _scene->SetSolverParams(params, test::ExpectNotOK{});
  EXPECT_EQ(
      CoulombFrictionModel::C1Regularized,
      _scene->GetSolverParams().experimentalEval.frictionModel);

  params.experimentalEval.elastoPlasticFriction = false;
  _scene->SetSolverParams(params, test::ExpectOK{});
}

// Elasto-plastic friction requires an integrator whose step ends at its last stage, since anchors
// advance from the contacts of the last stage: SetSolverParams rejects the others with it, and
// keeps the previous params. Without it, every integrator is accepted.
TEST_F(FrictionAnchors, RequiresIntegratorEndingAtLastStage) {
  for (int i = 0; i < static_cast<int>(IntegrationMethod::Count); ++i) {
    auto const method = static_cast<IntegrationMethod>(i);
    SolverParams params = _scene->GetSolverParams();
    params.experimentalEval.elastoPlasticFriction = true;
    params.integrationMethod = method;
    Error error;
    _scene->SetSolverParams(params, error);
    EXPECT_EQ(EndsAtLastStage(method), error.IsOK()) << SReflect::EnumToString(method);
    if (!error.IsOK()) {
      EXPECT_NE(method, _scene->GetSolverParams().integrationMethod);
    }

    params.experimentalEval.elastoPlasticFriction = false;
    _scene->SetSolverParams(params, test::ExpectOK{});
  }
}

// ExperimentalEvalParams::elastoPlasticFriction governs the anchor components of every actor with
// contact samples, with an index entry per sample: actors created with it disabled have none,
// enabling it adds empty ones, actors created with it enabled have them, and disabling it removes
// them all. The static plane is only a collider: it never has them. Stepping without the
// components runs no anchor system.
TEST_F(FrictionAnchors, ElastoPlasticFrictionGovernsAnchorComponents) {
  SetElastoPlasticFriction(false);
  CreatePlane();
  std::array<Actor*, 2> const disabledActors = {CreateRigidCube(), CreateSoftCube()};
  for (Actor* actor : disabledActors) {
    EXPECT_FALSE(HasAnyAnchorComponent(actor)) << actor->GetName();
  }
  Step(1);

  SetElastoPlasticFriction(true);
  std::array<Actor*, 4> const actors = {
      disabledActors[0], disabledActors[1], CreateRigidCube(), CreateSoftCube()};
  for (Actor* actor : actors) {
    ExpectAnchorComponents(actor);
    EXPECT_TRUE(GetAnchors(actor).anchors.empty()) << actor->GetName();
  }
  EXPECT_FALSE(GetRegistry().all_of<CContactSamples<TimeStep::Current>>(GetEntity(_plane)));
  EXPECT_FALSE(HasAnyAnchorComponent(_plane));
  Step(1);

  SetElastoPlasticFriction(false);
  for (Actor* actor : actors) {
    EXPECT_FALSE(HasAnyAnchorComponent(actor)) << actor->GetName();
  }
  Step(1);
}

TEST_F(FrictionAnchors, SoftActorAnchorsAdvance) {
  CreatePlane();
  ExpectSeededAnchorAdvances(CreateSoftCube());
}

// A seeded anchor of a rigid cube advances. Once the cube leaves the plane, it has no contacts and
// the rebuild drops its anchors.
TEST_F(FrictionAnchors, RebuildDropsAnchorsOfLostContacts) {
  CreatePlane();
  Actor* cube = CreateRigidCube();
  ExpectSeededAnchorAdvances(cube);
  cube->SetRootTransform(TransformRT{Quaternion{}, Real3{0_r, 1_r, 0_r}}, test::ExpectOK{});
  cube->SetVelocity(Real3{}, Real3{}, test::ExpectOK{});
  Step(1);
  EXPECT_TRUE(GetAnchors(cube).anchors.empty());
}

TEST_F(FrictionAnchors, RestoreStateRestoresAnchors) {
  CreatePlane();
  Actor* cube = CreateRigidCube();
  Step(1);
  entt::entity const plane = GetEntity(_plane);
  RebuildAnchors(
      GetAnchors(cube),
      GetIndex(cube),
      {MakeCollision(plane, {0, 5}, {Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}})});
  DynamicArray<FrictionAnchor> const savedAnchors = GetAnchors(cube).anchors;
  StateHandle const state = _scene->CaptureState(test::ExpectOK{});

  // Different anchors in a different order, so that the index no longer matches the captured ones.
  RebuildAnchors(
      GetAnchors(cube),
      GetIndex(cube),
      {MakeCollision(plane, {5, 0}, {Real3{7_r, 0_r, 0_r}, Real3{8_r, 0_r, 0_r}})});
  _scene->RestoreState(state, /*releaseImmediately=*/true, test::ExpectOK{});
  ExpectAnchors(savedAnchors, GetAnchors(cube).anchors);
  // The index is not captured: the post-restore reindexing points it to the restored anchors.
  EXPECT_EQ((Real3{1_r, 2_r, 3_r}), FindAnchorPoint(GetAnchors(cube), GetIndex(cube), 0, plane));
  EXPECT_EQ((Real3{4_r, 5_r, 6_r}), FindAnchorPoint(GetAnchors(cube), GetIndex(cube), 5, plane));
}

// Only contacts with rigid colliders use elasto-plastic friction and get anchor offsets: those of a
// rigid cube resting on a soft cube, and of the soft cube with the plane, use regular friction and
// elasto-plastic friction, respectively.
TEST_F(FrictionAnchors, OnlyRigidCollidersHaveAnchorOffsets) {
  CreatePlane();
  SoftActorParams softParams;
  softParams.name = "SoftCollider";
  softParams.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
  experimental::ExperimentalSoftActorParams experimentalParams;
  experimentalParams.colliderType = ColliderType::Sdf;
  experimentalParams.sdf.resolutionDelta = {0.04_r, 0.04_r, 0.04_r};
  experimentalParams.sdf.resolutionMode = GridSdfResolutionMode::Explicit;
  Actor* soft =
      experimental::CreateSoftActor(_scene, softParams, experimentalParams, test::ExpectOK{});
  Actor* rigid = CreateRigidCube();
  rigid->SetRootTransform(TransformRT{Quaternion{}, Real3{0_r, 1_r, 0_r}}, test::ExpectOK{});
  Step(10);

  auto& reg = GetRegistry();
  int numWithSoftCollider = 0;
  int numWithRigidCollider = 0;
  auto const expectOffsets = [&](auto const& collisions) {
    for (auto const& collision : collisions) {
      auto const& result = collision.collisionResult;
      if (result.sampleIndices.empty()) {
        continue;
      }
      bool const rigidCollider =
          reg.any_of<TagRigidActor, TagStaticActor>(collision.colliderEntity);
      EXPECT_EQ(rigidCollider ? isize(result.sampleIndices) : 0, isize(result.anchorOffset));
      ++(rigidCollider ? numWithRigidCollider : numWithSoftCollider);
    }
  };
  for (Actor* actor : {soft, rigid}) {
    entt::entity const e = GetEntity(actor);
    expectOffsets(reg.get<CActiveCollisions</*kIsSync*/ false, TimeStep::Current>>(e));
    expectOffsets(reg.get<CActiveCollisions</*kIsSync*/ true, TimeStep::Current>>(e));
  }
  EXPECT_GT(numWithSoftCollider, 0);
  EXPECT_GT(numWithRigidCollider, 0);
}

// Below the Coulomb limit, elasto-plastic friction holds the top cube of a stack of two on the cube
// below, a dynamic collider, once its contacts stick, with every supported integrator. The contacts
// have normal damping: without it, the stiff penalty contact chatters with the integrators less
// dissipative than backward Euler, its normal force dips toward zero, and the tangential load then
// exceeds the Coulomb limit, so the cube slides.
TEST_P(FrictionAnchorsIntegrator, HoldsTangentialLoadWithoutCreep) {
  real constexpr kNormalDamping = 100_r; // [s/m]
  EXPECT_NEAR(
      0_r,
      SlideUnderTangentialLoad(
          /*loadRatio*/ 0.5_r, /*numSteps*/ 100, kNormalDamping, /*numCubes*/ 2),
      1e-6_r);
}

// While the contacts of a cube stick, their anchors stay at the same material points of the plane,
// with every supported integrator: anchors are fixed during the stages of a step, whose stage-start
// positions are not the ends of the previous stages. The light tangential load is reversed once
// the contacts have anchors, so that they move while they stick. Only the anchors kept through
// every step are checked: a contact that briefly loses contact when the load reverses drops its
// anchor, and gets a new one elsewhere. Normal contact damping keeps the normal force from
// chattering toward zero.
TEST_P(FrictionAnchorsIntegrator, StickingAnchorsStayAtTheirMaterialPoints) {
  real constexpr kGravity = 9.81_r;
  real constexpr kNormalDamping = 100_r; // [s/m]
  real const load = 0.1_r * ContactParams{}.coulombFrictionCoefficient * kGravity;
  _scene->SetGravity(Real3{load, -kGravity, 0_r});
  CreatePlane(kNormalDamping);
  Actor* cube = CreateRigidCube(kNormalDamping);
  Step(20);
  DynamicArray<FrictionAnchor> kept = GetAnchors(cube).anchors;
  ASSERT_FALSE(kept.empty());
  DynamicArray<FrictionAnchor> const settled = kept;

  _scene->SetGravity(Real3{-load, -kGravity, 0_r});
  auto const findPoint = [&](FrictionAnchor const& anchor) {
    auto const collider = static_cast<entt::entity>(anchor.collider);
    return FindAnchorPoint(GetAnchors(cube), GetIndex(cube), anchor.sample, collider);
  };
  for (int i = 0; i < 50; ++i) {
    Step(1);
    auto const lost = std::ranges::remove_if(
        kept, [&](FrictionAnchor const& anchor) { return findPoint(anchor) == Real3{}; });
    kept.erase(lost.begin(), lost.end());
  }
  ASSERT_GE(2 * isize(kept), isize(settled)) << "Most contacts should keep their anchors.";

  real const tol = 1e-2_r * ContactParams{}.preslidingDisplacement;
  for (auto const& anchor : kept) {
    Real3 const point = findPoint(anchor);
    EXPECT_NEAR(anchor.point[0], point[0], tol) << "sample " << anchor.sample;
    EXPECT_NEAR(anchor.point[1], point[1], tol) << "sample " << anchor.sample;
    EXPECT_NEAR(anchor.point[2], point[2], tol) << "sample " << anchor.sample;
  }
}

// The same load makes regular friction creep with backward Euler, at the speed of
// velocity-regularized friction v_f (1 - sqrt(1 - loadRatio)), so the hold above is due to the
// anchors.
TEST_F(FrictionAnchors, RegularFrictionCreepsUnderTangentialLoad) {
  SetElastoPlasticFriction(false);
  real const creepSpeed = ContactParams{}.frictionFalloffVel * (1_r - Sqrt(0.5_r));
  real const distance = SlideUnderTangentialLoad(/*loadRatio*/ 0.5_r, /*numSteps*/ 100);
  EXPECT_NEAR(0.5_r * creepSpeed, distance, 0.05_r * 0.5_r * creepSpeed);
}

// Above the Coulomb limit, the cube slides with the acceleration of Coulomb friction, with every
// supported integrator, measured from its velocity change between 0.5 s and 1 s, once the contacts
// have settled. The contacts have normal damping, as in the hold tests: without it, the cube tilts
// as it slides, and tips over with backward Euler.
TEST_P(FrictionAnchorsIntegrator, SlidesAboveCoulombLimit) {
  real constexpr kNormalDamping = 100_r; // [s/m]
  real constexpr kLoadRatio = 1.5_r;
  real const acceleration =
      (kLoadRatio - 1_r) * ContactParams{}.coulombFrictionCoefficient * 9.81_r;
  Actor* cube = CreateCubeUnderTangentialLoad(kLoadRatio, kNormalDamping);
  Step(50);
  real const v0 = cube->GetLinearVelocity(test::ExpectOK{})[0];
  Step(50);
  real const v1 = cube->GetLinearVelocity(test::ExpectOK{})[0];
  EXPECT_NEAR(acceleration, (v1 - v0) / 0.5_r, 0.05_r * acceleration);
}

INSTANTIATE_TEST_SUITE_P(
    Integrators,
    FrictionAnchorsIntegrator,
    ::testing::Values(
        IntegrationMethod::BackwardEuler,
        IntegrationMethod::BDF2,
        IntegrationMethod::BDF3,
        IntegrationMethod::DIRK22,
        IntegrationMethod::DIRK33),
    IntegratorName);
