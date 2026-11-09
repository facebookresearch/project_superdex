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

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>

#include <array>
#include <limits>
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
  // the identity transform rest on it.
  void CreatePlane() {
    RigidActorParams params;
    params.name = "Plane";
    params.isStatic = true;
    params.shape = _mochiContext->CreatePlaneShape(Real3{0_r, 1_r, 0_r}, 0_r, test::ExpectOK{});
    _plane = _scene->CreateRigidActor(params, test::ExpectOK{});
  }

  Actor* CreateRigidCube() {
    RigidActorParams params;
    params.name = "RigidCube";
    params.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
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

  // Seeds an anchor on the plane at an offset from the deepest contact of a resting actor, steps
  // once, and checks that collision detection measured exactly the contacts of that sample from
  // it, and that the rebuild kept it. Without a friction law, the offset stays as filled, so the
  // rebuilt anchor follows the contact over the step.
  void ExpectSeededAnchorRoundTrips(Actor* actor) {
    Step(20);
    auto const [sample, position] = DeepestContactOnPlane(actor);
    ASSERT_GE(sample, 0);
    Real3 const offset{1e-5_r, 0_r, -2e-5_r};
    entt::entity const plane = GetEntity(_plane);
    ActiveCollision seeded = MakeCollision(plane, {sample}, {offset});
    seeded.collisionResult.posColliding = {position};
    RebuildAnchors(GetAnchors(actor), GetIndex(actor), {seeded});

    Step(1);
    // The step starts at the seeding contact, up to rounding of unit-scale positions.
    real constexpr kTol = Max(1e-7_r, 4_r * std::numeric_limits<real>::epsilon());
    int numSeeded = 0;
    for (auto const& collision : GetCollisions(actor)) {
      auto const& result = collision.collisionResult;
      ASSERT_EQ(isize(result.sampleIndices), isize(result.anchorOffset));
      for (int i = 0; i < isize(result.sampleIndices); ++i) {
        bool const isSeeded =
            result.sampleIndices[i] == sample && collision.colliderEntity == plane;
        Real3 const expected = isSeeded ? offset : Real3{};
        for (int d = 0; d < 3; ++d) {
          EXPECT_NEAR(expected[d], result.anchorOffset[i][d], kTol);
        }
        numSeeded += isSeeded ? 1 : 0;
      }
    }
    EXPECT_GT(numSeeded, 0);
    Real3 const rebuiltOffset = ContactPositionOnPlane(actor, sample) -
        FindAnchorPoint(GetAnchors(actor), GetIndex(actor), sample, plane);
    for (int d = 0; d < 3; ++d) {
      EXPECT_NEAR(offset[d], rebuiltOffset[d], kTol);
    }
    // Contacts without an anchor carry a zero offset, which is not stored.
    EXPECT_EQ(1, isize(GetAnchors(actor).anchors));
  }

  Actor* _plane = nullptr;
};
} // namespace

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

TEST_F(FrictionAnchors, SoftActorAnchorsRoundTrip) {
  CreatePlane();
  ExpectSeededAnchorRoundTrips(CreateSoftCube());
}

// A seeded anchor of a rigid cube round-trips. Once the cube leaves the plane, it has no contacts
// and the rebuild drops its anchors.
TEST_F(FrictionAnchors, RebuildDropsAnchorsOfLostContacts) {
  CreatePlane();
  Actor* cube = CreateRigidCube();
  ExpectSeededAnchorRoundTrips(cube);
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
