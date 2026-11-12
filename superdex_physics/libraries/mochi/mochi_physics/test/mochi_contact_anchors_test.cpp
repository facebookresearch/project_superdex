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

#include "mochi_articulated_test_scenes.h"
#include "mochi_friction_anchor_test_utils.h"
#include "mochi_physics_test_fixture.h"

// These tests inspect the friction anchor component and contact results, which are not exposed
// through the public API.
#include <mochi_physics/src/mochi_contact.h>
#include <mochi_physics/src/mochi_contact_anchors.h>
#include <mochi_physics/src/mochi_island.h>
#include <mochi_physics/src/mochi_rigid.h>
#include <mochi_physics/src/mochi_sleep.h>

#include <mochi_core/integration/integration_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/reflection.h>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <string>
#include <utility>

using namespace mochi;
using test::AreEqualAnchors;
using test::ExpectAnchors;

// Contacts of a colliding actor with one collider, with their anchor offsets, at positions twice
// those offsets, so that the anchors rebuilt from them are at points equal to the offsets. They all
// keep their anchors, unless told otherwise.
static ActiveCollision MakeCollision(
    entt::entity collider,
    DynamicArray<int> const& samples,
    DynamicArray<Real3> const& offsets,
    DynamicArray<bool> keepsAnchor = {}) {
  ActiveCollision collision;
  collision.colliderEntity = collider;
  collision.collisionResult.sampleIndices = samples;
  collision.collisionResult.anchorOffset = offsets;
  for (Real3 const& offset : offsets) {
    collision.collisionResult.posColliding.push_back(2_r * offset);
  }
  if (keepsAnchor.empty()) {
    keepsAnchor.resize(offsets.size(), true);
  }
  collision.collisionResult.keepsAnchor = std::move(keepsAnchor);
  return collision;
}

// Point of the sample's anchor on the collider, or zero.
static Real3 FindAnchorPoint(
    DynamicArray<FrictionAnchor> const& anchors,
    CFrictionAnchorIndex<AnchorSet::Active> const& index,
    int sample,
    entt::entity collider) {
  int const i = FindFrictionAnchor(index, anchors, sample, entt::to_integral(collider));
  return i == kSentinelIndex ? Real3{} : anchors[i].point;
}

// Expects the index entries of all samples, including kSentinelIndex for samples never indexed.
static void ExpectIndex(
    CFrictionAnchorIndex<AnchorSet::Active> const& index,
    DynamicArray<int> const& expected) {
  ASSERT_EQ(isize(expected), isize(index));
  for (int s = 0; s < isize(expected); ++s) {
    EXPECT_EQ(expected[s], index[s]) << "sample " << s;
  }
}

// FindFrictionAnchorFromHint takes the anchor at the hint when it matches and falls back to the
// index otherwise: in any lookup order it finds the anchor FindFrictionAnchor finds, and an absent
// anchor leaves the hint unchanged.
TEST(FrictionAnchorLookup, FindFromHintMatchesIndexLookup) {
  CFrictionAnchors<AnchorSet::Pending> pending;
  pending.anchors = {
      {.sample = 0, .collider = 10},
      {.sample = 0, .collider = 20},
      {.sample = 2, .collider = 10},
      {.sample = 3, .collider = 20}};
  CFrictionAnchorIndex<AnchorSet::Pending> index(5);
  IndexFrictionAnchors(pending, index);
  Span<FrictionAnchor const> const anchors = MakeConstSpan(pending.anchors);

  // In layout order, every lookup is the hint.
  int hint = 0;
  for (int i = 0; i < isize(anchors); ++i) {
    EXPECT_EQ(
        i,
        FindFrictionAnchorFromHint(index, anchors, anchors[i].sample, anchors[i].collider, hint));
    EXPECT_EQ(i + 1, hint);
  }

  // Out of order, with anchors absent for sample 1, for sample 2 on collider 20 and for sample 4.
  struct Lookup {
    int sample;
    uint32_t collider;
  };
  Lookup const lookups[] = {{3, 20}, {1, 10}, {0, 20}, {2, 20}, {0, 10}, {4, 10}, {2, 10}};
  hint = 0;
  for (auto const [sample, collider] : lookups) {
    int const expected = FindFrictionAnchor(index, anchors, sample, collider);
    int const before = hint;
    EXPECT_EQ(expected, FindFrictionAnchorFromHint(index, anchors, sample, collider, hint))
        << "sample " << sample << ", collider " << collider;
    EXPECT_EQ(expected == kSentinelIndex ? before : expected + 1, hint);
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

  // A free-rooted chain of two box-colliding unit cubes (see test::CreateUnitCubeChain).
  Actor* CreateArticulatedChain() {
    std::array const joints{ArticulatedJointType::Free, ArticulatedJointType::Spherical};
    return test::CreateUnitCubeChain(
        _scene,
        test::CreateUnitCubeTetMeshShape(_mochiContext),
        joints,
        /*dampingCoeff*/ 0_r,
        /*inertiaCoeff*/ 0_r,
        /*frictionCoeff*/ 0_r);
  }

  Actor* GetLink(Actor* articulated, int link) {
    return _scene->GetActor(articulated->GetNestedLinkActors(test::ExpectOK{})[link]);
  }

  // Tilts gravity along x, so that a cube resting on the plane carries a tangential load of
  // loadRatio times the Coulomb limit.
  void SetTangentialLoad(real loadRatio) {
    real const mu = ContactParams{}.coulombFrictionCoefficient;
    _scene->SetGravity(Real3{loadRatio * mu * kGravity, -kGravity, 0_r});
  }

  // The plane, and a rigid cube resting on it under a tangential load (see SetTangentialLoad).
  Actor* CreateCubeUnderTangentialLoad(real loadRatio, real normalDamping = 0_r) {
    SetTangentialLoad(loadRatio);
    CreatePlane(normalDamping);
    return CreateRigidCube(normalDamping);
  }

  void Step(int numSteps) {
    for (int i = 0; i < numSteps; ++i) {
      _scene->Step(1e-2_r);
    }
  }

  DynamicArray<FrictionAnchor>& GetAnchors(Actor* actor) {
    return GetRegistry().get<CFrictionAnchors<AnchorSet::Active>>(GetEntity(actor)).anchors;
  }

  DynamicArray<FrictionAnchor>& GetPendingAnchors(Actor* actor) {
    return GetRegistry().get<CFrictionAnchors<AnchorSet::Pending>>(GetEntity(actor)).anchors;
  }

  CFrictionAnchorIndex<AnchorSet::Active>& GetIndex(Actor* actor) {
    return GetRegistry().get<CFrictionAnchorIndex<AnchorSet::Active>>(GetEntity(actor));
  }

  // Copies the actor's pending anchors and indexes them, as at the start of the next step.
  void PromoteAnchors(Actor* actor) {
    ecs::InvokeOnEntity(&PromotePendingFrictionAnchors, GetRegistry(), GetEntity(actor));
    ecs::InvokeOnEntity(&IndexFrictionAnchors<AnchorSet::Active>, GetRegistry(), GetEntity(actor));
  }

  // Rebuilds the anchors from the given contacts, as after a converged step, and adopts and indexes
  // them, as at the start of the next one.
  void RebuildAnchors(
      DynamicArray<FrictionAnchor>& anchors,
      CFrictionAnchorIndex<AnchorSet::Active>& index,
      DynamicArray<ActiveCollision> const& asyncCollisions,
      DynamicArray<ActiveCollision> const& syncCollisions = {}) {
    CActiveCollisions</*kIsSync*/ false, TimeStep::Current> asyncActive(/*numPartitions*/ 1);
    CActiveCollisions</*kIsSync*/ true, TimeStep::Current> syncActive(/*numPartitions*/ 1);
    ActivateCollisions(asyncCollisions, asyncActive);
    ActivateCollisions(syncCollisions, syncActive);
    CFrictionAnchors<AnchorSet::Active> active;
    CFrictionAnchors<AnchorSet::Pending> pending;
    UpdateFrictionAnchors(
        GetRegistry(), /*convergence*/ nullptr, asyncActive, syncActive, pending, index);
    PromotePendingFrictionAnchors(pending, active);
    IndexFrictionAnchors(active, index);
    anchors = std::move(active.anchors);
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
      DynamicArray<FrictionAnchor>& anchors,
      CFrictionAnchorIndex<AnchorSet::Active>& index) {
    RebuildAnchors(
        anchors,
        index,
        {MakeCollision(a, {2, 0}, {Real3{1_r, 0_r, 0_r}, Real3{4_r, 0_r, 0_r}})},
        {MakeCollision(b, {0, 2}, {Real3{2_r, 0_r, 0_r}, Real3{3_r, 0_r, 0_r}})});
  }

  // Expects both anchor sets and an index entry per contact sample.
  void ExpectAnchorComponents(Actor* actor) {
    auto& reg = GetRegistry();
    entt::entity const e = GetEntity(actor);
    ASSERT_TRUE((reg.all_of<
                 CFrictionAnchors<AnchorSet::Active>,
                 CFrictionAnchors<AnchorSet::Pending>,
                 CFrictionAnchorIndex<AnchorSet::Active>>(e)))
        << actor->GetName();
    EXPECT_EQ(
        isize(reg.get<CContactSamples<TimeStep::Current> const>(e).positions),
        isize(reg.get<CFrictionAnchorIndex<AnchorSet::Active> const>(e)))
        << actor->GetName();
  }

  bool HasAnyAnchorComponent(Actor* actor) {
    return GetRegistry()
        .any_of<
            CFrictionAnchors<AnchorSet::Active>,
            CFrictionAnchors<AnchorSet::Pending>,
            CFrictionAnchorIndex<AnchorSet::Active>>(GetEntity(actor));
  }

  // Seeds an anchor of the actor's sample 0 on each collider, as after a step.
  void SeedAnchors(Actor* actor, DynamicArray<entt::entity> const& colliders) {
    DynamicArray<ActiveCollision> collisions;
    for (entt::entity const collider : colliders) {
      collisions.push_back(MakeCollision(collider, {0}, {Real3{1_r, 0_r, 0_r}}));
    }
    RebuildAnchors(GetAnchors(actor), GetIndex(actor), collisions);
    GetPendingAnchors(actor) = GetAnchors(actor);
  }

  // Colliders of the actor's anchors, in storage order.
  DynamicArray<entt::entity> GetAnchorColliders(Actor* actor) {
    DynamicArray<entt::entity> colliders;
    for (auto const& anchor : GetAnchors(actor)) {
      colliders.push_back(static_cast<entt::entity>(anchor.collider));
    }
    return colliders;
  }

  // Applies prescribed static motion and queued preload resets, and indexes the anchors, as at the
  // next step, without adopting pending anchors.
  void ApplyClearRequests() {
    ecs::InvokeForEachGlobal(&rigid::ApplyStaticStepStart, GetRegistry());
    PrepareFrictionAnchors(GetRegistry(), /*promotePending*/ false);
  }

  // Sets the actor's current pose. Dynamic actors request a reset; standalone statics do not.
  static void Place(Actor* actor) {
    actor->SetRootTransform(actor->GetRootTransform(), test::ExpectOK{});
  }

  // Moves a static actor and zeroes its velocity, which turns the kinematic move into a teleport.
  static void Teleport(Actor* staticActor, TransformRT const& worldFromLocal) {
    staticActor->SetRootTransform(worldFromLocal, test::ExpectOK{});
    staticActor->SetVelocity(Real3{}, Real3{}, test::ExpectOK{});
  }

  // Every setter that places an articulated actor: where it is, or at zero joint positions.
  static std::array<std::function<void()>, 3> ArticulatedPlacements(Actor* articulated) {
    DynamicArray<real> const zeroDofs(articulated->GetNumDofs(), 0_r);
    DynamicArray<TransformRT> worldFromLinks(
        articulated->GetNestedLinkActors(test::ExpectOK{}).size());
    articulated->GetArticulatedLinkTransforms(MakeSpan(worldFromLinks), test::ExpectOK{});
    return {
        [articulated] { Place(articulated); },
        [articulated, zeroDofs] {
          articulated->SetArticulatedPoseFromJoints(zeroDofs, test::ExpectOK{});
        },
        [articulated, worldFromLinks] {
          articulated->SetArticulatedPoseFromLinks(MakeConstSpan(worldFromLinks), test::ExpectOK{});
        }};
  }

  // For each placement: seeds anchors, places, applies the clear requests and checks the anchors.
  void ForEachPlacement(
      Span<std::function<void()> const> placements,
      std::function<void()> const& seed,
      std::function<void()> const& expect) {
    for (int i = 0; i < isize(placements); ++i) {
      SCOPED_TRACE("placement " + std::to_string(i));
      seed();
      placements[i]();
      ApplyClearRequests();
      expect();
    }
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
    GetPendingAnchors(actor) = GetAnchors(actor);

    Step(1);
    PromoteAnchors(actor);
    Real3 const anchorPoint = FindAnchorPoint(GetAnchors(actor), GetIndex(actor), sample, plane);
    Real3 const contact = ContactPositionOnPlane(actor, sample);
    Real3 const offset = contact - anchorPoint;
    EXPECT_NEAR(delta, Norm(Real3{offset[0], 0_r, offset[2]}), 1e-3_r * delta);
    EXPECT_GT(offset[0], 0.9_r * delta);
    Real3 const displacement = contact - (position - seededOffset);
    real const scale = delta / Norm(Real3{displacement[0], 0_r, displacement[2]});
    EXPECT_NEAR_TOL(scale * displacement, offset, 1e-3_r * delta);
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

  static constexpr real kGravity = 9.81_r;
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
class FrictionAnchorsSleepingReset : public FrictionAnchors,
                                     public ::testing::WithParamInterface<bool> {};

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
  DynamicArray<FrictionAnchor> anchors;
  CFrictionAnchorIndex<AnchorSet::Active> index(3);
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
  ExpectAnchors(expectedAnchors, anchors);
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
  DynamicArray<FrictionAnchor> anchors;
  CFrictionAnchorIndex<AnchorSet::Active> index(4);
  RebuildInterleavedAnchors(a, b, anchors, index);

  ExpectIndex(index, {2, kSentinelIndex, 0, kSentinelIndex});
  ASSERT_EQ(4, isize(anchors));
  DynamicArray<int> const expectedSamples{2, 2, 0, 0};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(expectedSamples[i], anchors[i].sample) << "anchor " << i;
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
  DynamicArray<FrictionAnchor> anchors;
  CFrictionAnchorIndex<AnchorSet::Active> index(4);
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
  FillFrictionAnchorOffsets(GetRegistry(), index, anchors, MakeSpan(collisions));

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
  DynamicArray<FrictionAnchor> anchors;
  CFrictionAnchorIndex<AnchorSet::Active> index(4);
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
      GetRegistry(), index, anchors, Span<ActiveCollision>(&collision, size_t{1}));

  EXPECT_EQ((DynamicArray<Real3>{Real3{}, Real3{}}), collision.collisionResult.anchorOffset);
}

// The fill and the rebuild are inverse: anchors filled into contacts that have not moved since
// stage start are rebuilt at the same points.
TEST_F(FrictionAnchors, RebuildOfFilledOffsetsKeepsAnchorPoints) {
  CreatePlane();
  entt::entity const plane = GetEntity(_plane);
  DynamicArray<FrictionAnchor> anchors;
  CFrictionAnchorIndex<AnchorSet::Active> index(2);
  RebuildAnchors(
      anchors, index, {MakeCollision(plane, {0, 1}, {Real3{1_r, 0_r, 0_r}, Real3{0_r, 0_r, 2_r}})});

  ActiveCollision collision = MakeCollision(plane, {1, 0}, {});
  collision.collisionResult.posCollidingStageStart = {Real3{5_r, 6_r, 7_r}, Real3{-1_r, 0_r, 3_r}};
  collision.collisionResult.posColliding = collision.collisionResult.posCollidingStageStart;
  collision.collisionResult.keepsAnchor = {true, true};
  FillFrictionAnchorOffsets(
      GetRegistry(), index, anchors, Span<ActiveCollision>(&collision, size_t{1}));
  RebuildAnchors(anchors, index, {collision});

  EXPECT_EQ(2, isize(anchors));
  EXPECT_EQ((Real3{1_r, 0_r, 0_r}), FindAnchorPoint(anchors, index, 0, plane));
  EXPECT_EQ((Real3{0_r, 0_r, 2_r}), FindAnchorPoint(anchors, index, 1, plane));
}

// Only the contacts with rigid colliders that keep their anchors have one, including zero offsets.
TEST_F(FrictionAnchors, RebuildKeepsAnchorsOfKeptContactsWithRigidColliders) {
  CreatePlane();
  entt::entity const plane = GetEntity(_plane);
  entt::entity const soft = GetEntity(CreateSoftCube());
  DynamicArray<FrictionAnchor> anchors;
  CFrictionAnchorIndex<AnchorSet::Active> index(4);
  RebuildAnchors(
      anchors,
      index,
      {MakeCollision(
           plane,
           {0, 1, 2},
           {Real3{1_r, 0_r, 0_r}, Real3{}, Real3{2_r, 0_r, 0_r}},
           {true, true, false}),
       MakeCollision(soft, {3}, {Real3{3_r, 0_r, 0_r}})});

  ExpectIndex(index, {0, 1, kSentinelIndex, kSentinelIndex});
  DynamicArray<FrictionAnchor> const expectedAnchors{
      {.sample = 0, .collider = entt::to_integral(plane), .point = Real3{1_r, 0_r, 0_r}},
      {.sample = 1, .collider = entt::to_integral(plane), .point = Real3{}}};
  ExpectAnchors(expectedAnchors, anchors);
}

// Rebuilding Pending and promoting it into Active reuse each array's capacity.
TEST_F(FrictionAnchors, RebuildReusesCapacity) {
  CreatePlane();
  entt::entity const plane = GetEntity(_plane);
  CFrictionAnchors<AnchorSet::Active> active;
  CFrictionAnchors<AnchorSet::Pending> pending;
  CFrictionAnchorIndex<AnchorSet::Active> index(3);
  active.anchors.reserve(3);
  pending.anchors.reserve(3);
  auto const* activeStorage = active.anchors.data();
  auto const* pendingStorage = pending.anchors.data();
  auto const rebuild = [&](DynamicArray<ActiveCollision> const& collisions) {
    CActiveCollisions</*kIsSync*/ false, TimeStep::Current> asyncActive(/*numPartitions*/ 1);
    CActiveCollisions</*kIsSync*/ true, TimeStep::Current> syncActive(/*numPartitions*/ 1);
    ActivateCollisions(collisions, asyncActive);
    ActivateCollisions({}, syncActive);
    UpdateFrictionAnchors(
        GetRegistry(), /*convergence*/ nullptr, asyncActive, syncActive, pending, index);
    EXPECT_EQ(pendingStorage, pending.anchors.data());
    EXPECT_EQ(3u, pending.anchors.capacity());
    PromotePendingFrictionAnchors(pending, active);
    ExpectAnchors(pending.anchors, active.anchors);
    EXPECT_EQ(activeStorage, active.anchors.data());
    EXPECT_EQ(3u, active.anchors.capacity());
    IndexFrictionAnchors(active, index);
  };
  ActiveCollision const threeContacts = MakeCollision(
      plane, {0, 1, 2}, {Real3{1_r, 0_r, 0_r}, Real3{2_r, 0_r, 0_r}, Real3{3_r, 0_r, 0_r}});
  rebuild({threeContacts});
  rebuild({threeContacts});

  // The index keeps stale entries for samples 0 and 2, which lookups reject.
  rebuild({MakeCollision(plane, {1}, {Real3{1_r, 2_r, 3_r}})});
  EXPECT_EQ((Real3{1_r, 2_r, 3_r}), FindAnchorPoint(active.anchors, index, 1, plane));
  EXPECT_EQ(Real3{}, FindAnchorPoint(active.anchors, index, 0, plane));
  EXPECT_EQ(Real3{}, FindAnchorPoint(active.anchors, index, 2, plane));

  rebuild({});
  EXPECT_TRUE(active.anchors.empty());
  EXPECT_TRUE(pending.anchors.empty());
  EXPECT_EQ(Real3{}, FindAnchorPoint(active.anchors, index, 1, plane));
}

// Placing a collider drops the anchors on it. The group of sample 0 shifts, so lookups need the
// index of the remaining anchors, built at the start of the step.
TEST_F(FrictionAnchors, PlacingColliderShiftsRemainingAnchorGroups) {
  CreatePlane();
  entt::entity const plane = GetEntity(_plane);
  Actor* collider = CreateRigidCube();
  Actor* colliding = CreateRigidCube();
  RebuildInterleavedAnchors(plane, GetEntity(collider), GetAnchors(colliding), GetIndex(colliding));
  GetPendingAnchors(colliding) = GetAnchors(colliding);

  Place(collider);
  ApplyClearRequests();
  auto const& anchors = GetAnchors(colliding);
  auto const& index = GetIndex(colliding);
  EXPECT_EQ(2, isize(anchors));
  ExpectAnchors(anchors, GetPendingAnchors(colliding));
  EXPECT_EQ((Real3{4_r, 0_r, 0_r}), FindAnchorPoint(anchors, index, 0, plane));
  EXPECT_EQ((Real3{1_r, 0_r, 0_r}), FindAnchorPoint(anchors, index, 2, plane));
  EXPECT_EQ(Real3{}, FindAnchorPoint(anchors, index, 0, GetEntity(collider)));
  EXPECT_EQ(Real3{}, FindAnchorPoint(anchors, index, 2, GetEntity(collider)));
}

// Placing an actor that no anchors are on keeps the anchors of other actors as they are. Placing
// all their colliders drops them all.
TEST_F(FrictionAnchors, PlacingOtherActorKeepsAnchors) {
  CreatePlane();
  entt::entity const plane = GetEntity(_plane);
  Actor* collider = CreateRigidCube();
  Actor* other = CreateRigidCube();
  Actor* colliding = CreateRigidCube();
  RebuildInterleavedAnchors(plane, GetEntity(collider), GetAnchors(colliding), GetIndex(colliding));
  GetPendingAnchors(colliding) = GetAnchors(colliding);
  DynamicArray<FrictionAnchor> const savedAnchors = GetAnchors(colliding);

  Place(other);
  ApplyClearRequests();
  ExpectAnchors(savedAnchors, GetAnchors(colliding));
  ExpectAnchors(savedAnchors, GetPendingAnchors(colliding));

  Place(collider);
  Teleport(_plane, TransformRT{Real3{0_r, -1e-3_r, 0_r}});
  ApplyClearRequests();
  EXPECT_TRUE(GetAnchors(colliding).empty());
  EXPECT_TRUE(GetPendingAnchors(colliding).empty());
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
// contact samples, with an index entry per sample, and the clear requests: actors created with it
// disabled have none, enabling it adds empty ones, actors created with it enabled have them, and
// disabling it removes them all. The static plane is only a collider: it never has anchor
// components. Stepping without the components runs no anchor system. A state captured with it
// disabled cannot be restored with it enabled.
TEST_F(FrictionAnchors, ElastoPlasticFrictionGovernsAnchorComponents) {
  SetElastoPlasticFriction(false);
  EXPECT_FALSE(GetRegistry().try_ctx<CFrictionAnchorClearRequests>());
  CreatePlane();
  std::array<Actor*, 2> const disabledActors = {CreateRigidCube(), CreateSoftCube()};
  for (Actor* actor : disabledActors) {
    EXPECT_FALSE(HasAnyAnchorComponent(actor)) << actor->GetName();
  }
  Step(1);
  StateHandle const disabledState = _scene->CaptureState(test::ExpectOK{});

  SetElastoPlasticFriction(true);
  EXPECT_TRUE(GetRegistry().try_ctx<CFrictionAnchorClearRequests>());
  // Same entities as the capture, so that only the anchor components reject the restore.
  StateHandle const enabledState = _scene->CaptureState(test::ExpectOK{});
  _scene->RestoreState(disabledState, /*releaseImmediately=*/true, test::ExpectNotOK{});
  _scene->RestoreState(enabledState, /*releaseImmediately=*/true, test::ExpectOK{});
  std::array<Actor*, 4> const actors = {
      disabledActors[0], disabledActors[1], CreateRigidCube(), CreateSoftCube()};
  for (Actor* actor : actors) {
    ExpectAnchorComponents(actor);
    EXPECT_TRUE(GetAnchors(actor).empty()) << actor->GetName();
    EXPECT_TRUE(GetPendingAnchors(actor).empty()) << actor->GetName();
  }
  EXPECT_FALSE(GetRegistry().all_of<CContactSamples<TimeStep::Current>>(GetEntity(_plane)));
  EXPECT_FALSE(HasAnyAnchorComponent(_plane));
  Step(1);

  SetElastoPlasticFriction(false);
  EXPECT_FALSE(GetRegistry().try_ctx<CFrictionAnchorClearRequests>());
  for (Actor* actor : actors) {
    EXPECT_FALSE(HasAnyAnchorComponent(actor)) << actor->GetName();
  }
  Step(1);
}

TEST_F(FrictionAnchors, SoftActorAnchorsAdvance) {
  CreatePlane();
  ExpectSeededAnchorAdvances(CreateSoftCube());
}

// A seeded anchor of a rigid cube advances. Upward gravity then makes the cube leave the contact
// threshold without a state override, so the rebuild drops the lost contacts' anchors.
TEST_F(FrictionAnchors, RebuildDropsAnchorsOfLostContacts) {
  CreatePlane();
  Actor* cube = CreateRigidCube();
  ExpectSeededAnchorAdvances(cube);
  _scene->SetGravity(Real3{0_r, 100_r, 0_r});
  EXPECT_TRUE(GetRegistry().ctx<CFrictionAnchorClearRequests>().entities.empty());
  Step(1);
  PromoteAnchors(cube);
  EXPECT_TRUE(GetAnchors(cube).empty());
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
  real constexpr kNormalDamping = 100_r; // [s/m]
  real constexpr kLoadRatio = 0.1_r;
  Actor* cube = CreateCubeUnderTangentialLoad(kLoadRatio, kNormalDamping);
  Step(20);
  PromoteAnchors(cube);
  DynamicArray<FrictionAnchor> kept = GetAnchors(cube);
  ASSERT_FALSE(kept.empty());
  DynamicArray<FrictionAnchor> const settled = kept;

  SetTangentialLoad(-kLoadRatio);
  auto const findPoint = [&](FrictionAnchor const& anchor) {
    auto const collider = static_cast<entt::entity>(anchor.collider);
    return FindAnchorPoint(GetAnchors(cube), GetIndex(cube), anchor.sample, collider);
  };
  for (int i = 0; i < 50; ++i) {
    Step(1);
    PromoteAnchors(cube);
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
      (kLoadRatio - 1_r) * ContactParams{}.coulombFrictionCoefficient * kGravity;
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

// Every setter that places a dynamic rigid actor drops its anchors, and those of other actors on
// it, at the start of the next step. Anchors on other colliders stay.
TEST_F(FrictionAnchors, PlacingRigidActorDropsItsAnchorsAndThoseOnIt) {
  CreatePlane();
  entt::entity const plane = GetEntity(_plane);
  Actor* placed = CreateRigidCube();
  Actor* other = CreateRigidCube();
  std::array<std::function<void()>, 2> const placements = {
      [&] {
        placed->SetRootTransform(TransformRT{Quaternion{}, Real3{2_r, 0_r, 0_r}}, test::ExpectOK{});
        EXPECT_EQ(2, isize(GetAnchors(placed))) << "Dropped only at the next step.";
      },
      [&] {
        placed->SetCenterOfMassTransform(
            placed->GetCenterOfMassTransform(test::ExpectOK{}), test::ExpectOK{});
      }};
  ForEachPlacement(
      MakeConstSpan(placements),
      [&] {
        SeedAnchors(placed, {plane, GetEntity(other)});
        SeedAnchors(other, {plane, GetEntity(placed)});
      },
      [&] {
        EXPECT_TRUE(GetAnchors(placed).empty());
        EXPECT_EQ(DynamicArray<entt::entity>{plane}, GetAnchorColliders(other));
        ExpectAnchors(GetAnchors(placed), GetPendingAnchors(placed));
        ExpectAnchors(GetAnchors(other), GetPendingAnchors(other));
      });
}

// Every explicit static velocity override clears preload; pose-only motion keeps it.
TEST_F(FrictionAnchors, StaticVelocityOverridesClearAnchors) {
  CreatePlane();
  entt::entity const plane = GetEntity(_plane);
  Actor* cube = CreateRigidCube();
  auto const move = [&] {
    TransformRT moved = _plane->GetRootTransform();
    moved.SetTranslation(moved.GetTranslation() - Real3{0_r, 1e-3_r, 0_r});
    _plane->SetRootTransform(moved, test::ExpectOK{});
  };
  auto const stop = [&] { _plane->SetVelocity(Real3{}, Real3{}, test::ExpectOK{}); };
  auto const slide = [&] {
    _plane->SetVelocity(Real3{0_r, -0.1_r, 0_r}, Real3{}, test::ExpectOK{});
  };
  struct Case {
    std::function<void()> prescribe;
    bool clearsAnchors;
  };
  std::array<Case, 10> const cases = {{
      {.prescribe = stop, .clearsAnchors = true},
      {.prescribe = move, .clearsAnchors = false},
      {.prescribe =
           [&] {
             move();
             stop();
           },
       .clearsAnchors = true},
      {.prescribe =
           [&] {
             stop();
             move();
           },
       .clearsAnchors = true},
      {.prescribe =
           [&] {
             move();
             slide();
           },
       .clearsAnchors = true},
      {.prescribe =
           [&] {
             Place(_plane);
             stop();
           },
       .clearsAnchors = true},
      {.prescribe = slide, .clearsAnchors = true},
      {.prescribe = slide, .clearsAnchors = true},
      {.prescribe =
           [&] {
             _plane->SetVelocity(
                 Real3{std::numeric_limits<real>::quiet_NaN(), 0_r, 0_r},
                 Real3{},
                 test::ExpectNotOK{});
           },
       .clearsAnchors = false},
      {.prescribe =
           [&] {
             _plane->SetVelocity(
                 Real3{},
                 Real3{std::numeric_limits<real>::quiet_NaN(), 0_r, 0_r},
                 test::ExpectNotOK{});
           },
       .clearsAnchors = false},
  }};
  for (int i = 0; i < isize(cases); ++i) {
    SCOPED_TRACE("case " + std::to_string(i));
    SeedAnchors(cube, {plane});
    cases[i].prescribe();
    EXPECT_EQ(
        cases[i].clearsAnchors,
        !GetRegistry().ctx<CFrictionAnchorClearRequests>().entities.empty());
    ApplyClearRequests();
    EXPECT_EQ(cases[i].clearsAnchors, GetAnchors(cube).empty());
  }
}

// Every setter that places a soft actor drops its anchors. Nested soft setters also reset the
// blended parent whose skin they move.
TEST_F(FrictionAnchors, PlacingSoftActorDropsItsAnchors) {
  CreatePlane();
  Actor* soft = CreateSoftCube();
  soft->RegisterQuery(QueryType::NodePositions, test::ExpectOK{});
  _scene->Step(0.0); // Updates the queries
  Span<real const> const positionsSpan = soft->GetNodePositionsLocal(test::ExpectOK{});
  DynamicArray<real> const positions(positionsSpan);
  DynamicArray<real> const displacements(soft->GetDisplacements(test::ExpectOK{}));

  std::array<std::function<void()>, 4> const placements = {
      [&] { Place(soft); },
      [&] { soft->SetNodePositionsLocal(MakeConstSpan(positions), test::ExpectOK{}); },
      [&] { soft->SetDisplacements(MakeConstSpan(displacements), test::ExpectOK{}); },
      [&] { soft->SetZeroDisplacementsAndVelocities(test::ExpectOK{}); }};
  ForEachPlacement(
      MakeConstSpan(placements),
      [&] { SeedAnchors(soft, {GetEntity(_plane)}); },
      [&] { EXPECT_TRUE(GetAnchors(soft).empty()); });

  Actor* blended = test::CreateThreeBoneSoftSkinnedActor(
      _mochiContext,
      _scene,
      test::ThreeBoneSoftSkinnedMode::BlendedFom,
      TransformRT{},
      0_r,
      0_r,
      0_r,
      ActorBoundaryElementType::Default);
  Actor* nested = _scene->GetActor(blended->GetNestedSoftActors(test::ExpectOK{})[0]);
  DynamicArray<real> const nestedDisplacements(nested->GetDisplacements(test::ExpectOK{}));
  std::array<std::function<void()>, 2> const nestedPlacements = {
      [&] { nested->SetDisplacements(MakeConstSpan(nestedDisplacements), test::ExpectOK{}); },
      [&] { nested->SetZeroDisplacementsAndVelocities(test::ExpectOK{}); }};
  ForEachPlacement(
      MakeConstSpan(nestedPlacements),
      [&] { SeedAnchors(blended, {GetEntity(_plane)}); },
      [&] { EXPECT_TRUE(GetAnchors(blended).empty()); });
}

// Every setter that places an articulated actor drops the anchors of its links, and those of other
// actors on them.
TEST_F(FrictionAnchors, PlacingArticulatedActorDropsAnchorsOfLinks) {
  CreatePlane();
  entt::entity const plane = GetEntity(_plane);
  Actor* chain = CreateArticulatedChain();
  std::array<Actor*, 2> const links = {GetLink(chain, 0), GetLink(chain, 1)};
  for (Actor* link : links) {
    ExpectAnchorComponents(link);
  }
  Actor* other = CreateRigidCube();
  ForEachPlacement(
      MakeConstSpan(ArticulatedPlacements(chain)),
      [&] {
        for (Actor* link : links) {
          SeedAnchors(link, {plane});
        }
        SeedAnchors(other, {plane, GetEntity(links[0]), GetEntity(links[1])});
      },
      [&] {
        for (Actor* link : links) {
          EXPECT_TRUE(GetAnchors(link).empty());
        }
        EXPECT_EQ(DynamicArray<entt::entity>{plane}, GetAnchorColliders(other));
      });
}

// Every setter that places an articulated actor drops the anchors of its skin, which belong to the
// actor itself, and those of other actors on its links, here on its static root link.
TEST_F(FrictionAnchors, PlacingArticulatedActorDropsAnchorsOfSkin) {
  CreatePlane();
  entt::entity const plane = GetEntity(_plane);
  Actor* skinned = test::CreateThreeBoneSkinnedActor(
      _mochiContext, _scene, 0_r, 0_r, 0_r, ActorBoundaryElementType::Default);
  ExpectAnchorComponents(skinned);
  Actor* staticLink = GetLink(skinned, 0);
  ASSERT_TRUE(GetRegistry().all_of<TagStaticActor>(GetEntity(staticLink)));
  Actor* other = CreateRigidCube();
  ForEachPlacement(
      MakeConstSpan(ArticulatedPlacements(skinned)),
      [&] {
        SeedAnchors(skinned, {plane});
        SeedAnchors(other, {plane, GetEntity(staticLink)});
      },
      [&] {
        EXPECT_TRUE(GetAnchors(skinned).empty());
        EXPECT_EQ(DynamicArray<entt::entity>{plane}, GetAnchorColliders(other));
      });
}

// Every setter that places an articulated actor drops the anchors of its nested soft actors.
TEST_F(FrictionAnchors, PlacingArticulatedActorDropsAnchorsOfNestedSoftActors) {
  CreatePlane();
  Actor* skinned = test::CreateThreeBoneSoftSkinnedActor(
      _mochiContext,
      _scene,
      test::ThreeBoneSoftSkinnedMode::Fom,
      TransformRT{},
      0_r,
      0_r,
      0_r,
      ActorBoundaryElementType::Default);
  Actor* nested = _scene->GetActor(skinned->GetNestedSoftActors(test::ExpectOK{})[0]);
  ExpectAnchorComponents(nested);
  ForEachPlacement(
      MakeConstSpan(ArticulatedPlacements(skinned)),
      [&] { SeedAnchors(nested, {GetEntity(_plane)}); },
      [&] { EXPECT_TRUE(GetAnchors(nested).empty()); });
}

// Velocity and center-of-mass overrides clear friction preload when they invalidate step history.
TEST_F(FrictionAnchors, VelocityAndCenterOfMassSettersClearAnchors) {
  CreatePlane();
  Actor* rigid = CreateRigidCube();
  Actor* soft = CreateSoftCube();
  Actor* chain = CreateArticulatedChain();
  Real3 const velocity{1_r, 0_r, 0_r};
  struct Case {
    Actor* actor;
    std::function<void()> reset;
  };
  std::array<Case, 5> const cases = {{
      {.actor = rigid,
       .reset = [&] { rigid->SetVelocity(velocity, Real3{0_r, 1_r, 0_r}, test::ExpectOK{}); }},
      {.actor = rigid,
       .reset =
           [&] {
             rigid->SetInertiaProperties(
                 rigid->GetMass(test::ExpectOK{}),
                 rigid->GetRigidCenterOfMassLocal(test::ExpectOK{}) + Real3{0.1_r, 0_r, 0_r},
                 rigid->GetRigidMomentOfInertiaLocal(test::ExpectOK{}),
                 test::ExpectOK{});
           }},
      {.actor = soft, .reset = [&] { soft->SetVelocity(velocity, Real3{}, test::ExpectOK{}); }},
      {.actor = soft,
       .reset = [&] { test::SetRigidNodeVelocitiesLocal(soft, velocity, Real3{}, Real3{}); }},
      {.actor = GetLink(chain, 0),
       .reset =
           [&] {
             chain->SetArticulatedJointVelocities(
                 DynamicArray<real>(chain->GetNumDofs(), 1_r), test::ExpectOK{});
           }},
  }};
  for (int i = 0; i < isize(cases); ++i) {
    SCOPED_TRACE("case " + std::to_string(i));
    SeedAnchors(cases[i].actor, {GetEntity(_plane)});
    cases[i].reset();
    ApplyClearRequests();
    EXPECT_TRUE(GetAnchors(cases[i].actor).empty());
  }
}

// A placed actor destroyed before the next step is skipped, and the anchors of other actors stay.
TEST_F(FrictionAnchors, DestroyedPlacedActorIsSkipped) {
  CreatePlane();
  Actor* placed = CreateRigidCube();
  Actor* other = CreateRigidCube();
  SeedAnchors(other, {GetEntity(_plane)});
  DynamicArray<FrictionAnchor> const savedAnchors = GetAnchors(other);

  Place(placed);
  _scene->DestroyActor(placed);
  ApplyClearRequests();
  ExpectAnchors(savedAnchors, GetAnchors(other));
  EXPECT_TRUE(GetRegistry().ctx<CFrictionAnchorClearRequests>().entities.empty());
}

// Clear requests are captured with the state. Restoring a state captured after a placement, before
// the next step, brings back its pending request along with the anchors it applies to. Restoring a
// state captured before the placement drops the pending request, so the anchors of that state stay.
TEST_F(FrictionAnchors, RestoreStateRestoresClearRequests) {
  CreatePlane();
  Actor* cube = CreateRigidCube();
  SeedAnchors(cube, {GetEntity(_plane)});
  DynamicArray<FrictionAnchor> const savedAnchors = GetAnchors(cube);
  StateHandle const beforePlacement = _scene->CaptureState(test::ExpectOK{});
  Place(cube);
  StateHandle const afterPlacement = _scene->CaptureState(test::ExpectOK{});
  ApplyClearRequests();
  ASSERT_TRUE(GetAnchors(cube).empty());
  EXPECT_TRUE(GetPendingAnchors(cube).empty());

  _scene->RestoreState(afterPlacement, /*releaseImmediately=*/false, test::ExpectOK{});
  ExpectAnchors(savedAnchors, GetAnchors(cube));
  ExpectAnchors(savedAnchors, GetPendingAnchors(cube));
  ApplyClearRequests();
  EXPECT_TRUE(GetAnchors(cube).empty());
  EXPECT_TRUE(GetPendingAnchors(cube).empty());

  _scene->RestoreState(afterPlacement, /*releaseImmediately=*/true, test::ExpectOK{});
  ASSERT_FALSE(GetRegistry().ctx<CFrictionAnchorClearRequests>().entities.empty());
  _scene->RestoreState(beforePlacement, /*releaseImmediately=*/true, test::ExpectOK{});
  ApplyClearRequests();
  ExpectAnchors(savedAnchors, GetAnchors(cube));
  ExpectAnchors(savedAnchors, GetPendingAnchors(cube));
}

// Once the load is removed, the anchors of a held cube pull it back by their pre-load. A placed
// cube instead starts without pre-load: moved elsewhere on the plane, it stays put. The placement
// is applied at the next step, which is the first one to use the dropped anchors.
TEST_F(FrictionAnchors, PlacementDropsPreload) {
  Actor* held = CreateCubeUnderTangentialLoad(0.5_r);
  Actor* placed = CreateRigidCube();
  placed->SetRootTransform(TransformRT{Real3{3_r, 0_r, 0_r}}, test::ExpectOK{});
  Step(50);
  SetTangentialLoad(0_r);
  ASSERT_FALSE(GetAnchors(held).empty());
  ASSERT_FALSE(GetAnchors(placed).empty());

  real const heldX0 = held->GetRootTransform().GetTranslation()[0];
  TransformRT moved = placed->GetRootTransform();
  moved.SetTranslation(moved.GetTranslation() + Real3{1_r, 0_r, 0_r});
  placed->SetRootTransform(moved, test::ExpectOK{});
  placed->SetVelocity(Real3{}, Real3{}, test::ExpectOK{});
  Step(50);
  EXPECT_GT(Abs(held->GetRootTransform().GetTranslation()[0] - heldX0), 1e-5_r);
  EXPECT_NEAR(moved.GetTranslation()[0], placed->GetRootTransform().GetTranslation()[0], 1e-6_r);
}

// Active holds the anchors used by a step; Pending holds those advanced for the next one.
// A diverged step retains the anchors it used in both sets.
TEST_F(FrictionAnchors, StepAdoptsAnchorsPendingFromLastStep) {
  test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
  Actor* cube = CreateCubeUnderTangentialLoad(1.5_r);
  Step(20);
  ASSERT_FALSE(GetPendingAnchors(cube).empty());
  DynamicArray<FrictionAnchor> const pending = GetPendingAnchors(cube);
  ASSERT_FALSE(AreEqualAnchors(pending, GetAnchors(cube)));

  Step(1);
  ExpectAnchors(pending, GetAnchors(cube));
  ASSERT_FALSE(GetPendingAnchors(cube).empty());

  // Zero tolerances and a tiny divergence threshold make the next step diverge.
  DynamicArray<FrictionAnchor> const adopted = GetPendingAnchors(cube);
  SolverParams params = _scene->GetSolverParams();
  params.nonLinearSolver.absTol = 0_r;
  params.nonLinearSolver.relTol = 0_r;
  params.nonLinearSolver.relStepTol = 0_r;
  params.nonLinearSolver.absDivTol = 1e-30_r;
  _scene->SetSolverParams(params, test::ExpectOK{});
  Step(1);
  ASSERT_EQ(ConvergenceStatus::Diverged, cube->GetConvergenceStatus());
  ExpectAnchors(adopted, GetAnchors(cube));
  ExpectAnchors(adopted, GetPendingAnchors(cube));
}

// A placement after a step drops the anchors that the next step adopts, which would otherwise bring
// back those dropped from the anchors the last step used. A solve-free assembly in between, which a
// force query requires, applies the clear request, so it reports the friction without preload, and
// keeps it for that step.
TEST_F(FrictionAnchors, PlacementAfterStepDropsAdoptedAnchors) {
  Actor* cube = CreateCubeUnderTangentialLoad(0.5_r);
  cube->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  Step(20);
  ASSERT_FALSE(GetPendingAnchors(cube).empty());
  DynamicArray<FrictionAnchor> const current = GetAnchors(cube);
  DynamicArray<FrictionAnchor> const pending = GetPendingAnchors(cube);
  Real3 const held = cube->GetContactForceWorld(test::ExpectOK{});
  Place(cube);
  _scene->Step(0.0);
  EXPECT_LT(Abs(cube->GetContactForceWorld(test::ExpectOK{})[0]), 0.1_r * Abs(held[0]));
  _scene->UpdateQueries(test::ExpectOK{});
  EXPECT_FALSE(GetRegistry().ctx<CFrictionAnchorClearRequests>().entities.empty());
  ExpectAnchors(current, GetAnchors(cube));
  ExpectAnchors(pending, GetPendingAnchors(cube));

  Step(1);
  EXPECT_TRUE(GetAnchors(cube).empty());
}

// A sleeping island retains its last solved anchors through steps, query refreshes, and restore.
TEST_F(FrictionAnchors, SleepingIslandKeepsAnchorsOfLastSolvedStep) {
  experimental::SetSleepParams(
      _scene, experimental::SleepParams{.canSleep = true}, test::ExpectOK{});
  real constexpr kNormalDamping = 100_r; // [s/m]
  Actor* cube = CreateCubeUnderTangentialLoad(0.5_r, kNormalDamping);
  cube->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  auto const isAwake = [&] {
    auto const& reg = GetRegistry();
    return sleep::IsIslandAwake(reg, reg.get<CIslandMemberInfo const>(GetEntity(cube)).island);
  };
  int constexpr kMaxSteps = 500;
  DynamicArray<FrictionAnchor> lastPending;
  int numSteps = 0;
  for (; numSteps < kMaxSteps; ++numSteps) {
    Step(1);
    if (!isAwake()) {
      break;
    }
    lastPending = GetPendingAnchors(cube);
  }
  ASSERT_LT(numSteps, kMaxSteps) << "The cube should fall asleep.";
  ASSERT_FALSE(lastPending.empty());

  for (int i = 0; i < 3; ++i) {
    SCOPED_TRACE("asleep step " + std::to_string(i));
    Step(1);
    _scene->UpdateQueries(test::ExpectOK{});
    EXPECT_FALSE(isAwake());
    ExpectAnchors(lastPending, GetAnchors(cube));
    ExpectAnchors(lastPending, GetPendingAnchors(cube));
  }
  StateHandle const asleepState = _scene->CaptureState(test::ExpectOK{});

  SetTangentialLoad(-0.5_r);
  Step(1);
  EXPECT_TRUE(isAwake());
  ExpectAnchors(lastPending, GetAnchors(cube));
  ASSERT_FALSE(GetPendingAnchors(cube).empty());

  _scene->RestoreState(asleepState, /*releaseImmediately=*/true, test::ExpectOK{});
  _scene->UpdateQueries(test::ExpectOK{});
  ExpectAnchors(lastPending, GetAnchors(cube));
  ExpectAnchors(lastPending, GetPendingAnchors(cube));
}

// A sleeping actor reset, or a reset of its static collider, drops the anchors from both sets
// and wakes every affected island, whose solved step rebuilds Pending.
TEST_P(FrictionAnchorsSleepingReset, ResetRebuildsPendingBeforeNextPromotion) {
  experimental::SetSleepParams(
      _scene, experimental::SleepParams{.canSleep = true}, test::ExpectOK{});
  real constexpr kNormalDamping = 100_r; // [s/m]
  std::array<Actor*, 2> const actors{
      CreateCubeUnderTangentialLoad(0.5_r, kNormalDamping), CreateRigidCube(kNormalDamping)};
  actors[1]->SetRootTransform(TransformRT{Real3{3_r, 0_r, 0_r}}, test::ExpectOK{});
  auto const isAwake = [&](Actor* actor) {
    auto const& reg = GetRegistry();
    return sleep::IsIslandAwake(reg, reg.get<CIslandMemberInfo const>(GetEntity(actor)).island);
  };
  int constexpr kMaxSteps = 500;
  int numSteps = 0;
  for (; numSteps < kMaxSteps; ++numSteps) {
    Step(1);
    if (std::ranges::none_of(actors, isAwake)) {
      break;
    }
  }
  ASSERT_LT(numSteps, kMaxSteps) << "Both cubes should fall asleep.";
  std::array<DynamicArray<FrictionAnchor>, 2> const beforeReset{
      GetPendingAnchors(actors[0]), GetPendingAnchors(actors[1])};
  for (int i = 0; i < 2; ++i) {
    ASSERT_FALSE(isAwake(actors[i]));
    ASSERT_FALSE(beforeReset[i].empty());
  }

  bool const resetCollider = GetParam();
  Actor* resetActor = resetCollider ? _plane : actors[0];
  if (resetCollider) {
    resetActor->SetVelocity(Real3{}, Real3{}, test::ExpectOK{});
  } else {
    TransformRT moved = resetActor->GetRootTransform();
    moved.SetTranslation(moved.GetTranslation() + Real3{1_r, 0_r, 0_r});
    resetActor->SetRootTransform(moved, test::ExpectOK{});
  }
  auto const& reg = GetRegistry();
  auto const& requests = reg.ctx<CFrictionAnchorClearRequests>().entities;
  EXPECT_NE(requests.end(), std::ranges::find(requests, entt::to_integral(GetEntity(resetActor))));
  auto const& wake = reg.ctx<CWakeUpRequests>();
  int const numAffected = resetCollider ? 2 : 1;
  for (int i = 0; i < numAffected; ++i) {
    auto const island = reg.get<CIslandMemberInfo const>(GetEntity(actors[i])).island;
    EXPECT_TRUE(reg.all_of<TagWakeUp>(island) || wake.wakeAll || !wake.regions.empty());
    ExpectAnchors(beforeReset[i], GetPendingAnchors(actors[i]));
  }

  Step(1);
  ASSERT_TRUE(GetRegistry().ctx<CFrictionAnchorClearRequests>().entities.empty());
  for (int i = 0; i < numAffected; ++i) {
    SCOPED_TRACE("affected actor " + std::to_string(i));
    EXPECT_TRUE(isAwake(actors[i]));
    ASSERT_NE(ConvergenceStatus::Diverged, actors[i]->GetConvergenceStatus());
    EXPECT_TRUE(GetAnchors(actors[i]).empty());
    ASSERT_FALSE(GetPendingAnchors(actors[i]).empty());
    EXPECT_FALSE(AreEqualAnchors(beforeReset[i], GetPendingAnchors(actors[i])));
  }
  std::array<DynamicArray<FrictionAnchor>, 2> const next{
      GetPendingAnchors(actors[0]), GetPendingAnchors(actors[1])};
  StateHandle const resetState = _scene->CaptureState(test::ExpectOK{});
  Step(1);
  for (int i = 0; i < numAffected; ++i) {
    ExpectAnchors(next[i], GetAnchors(actors[i]));
    EXPECT_FALSE(AreEqualAnchors(beforeReset[i], GetAnchors(actors[i])));
  }

  _scene->RestoreState(resetState, /*releaseImmediately=*/true, test::ExpectOK{});
  for (int i = 0; i < numAffected; ++i) {
    EXPECT_TRUE(GetAnchors(actors[i]).empty());
    ExpectAnchors(next[i], GetPendingAnchors(actors[i]));
  }
  Step(1);
  for (int i = 0; i < numAffected; ++i) {
    ExpectAnchors(next[i], GetAnchors(actors[i]));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ResetRoutes,
    FrictionAnchorsSleepingReset,
    ::testing::Bool(),
    [](::testing::TestParamInfo<bool> const& info) {
      return info.param ? "StaticCollider" : "Actor";
    });

// Both anchor sets restore bit for bit. Query refresh reproduces the completed step's forces
// from Active without consuming Pending, which the next advancing step then uses. The index is not
// captured: query refresh indexes the restored anchors, whatever the index was.
TEST_F(FrictionAnchors, RestoreStateRestoresActiveAndPendingAnchors) {
  Actor* cube = CreateCubeUnderTangentialLoad(1.5_r);
  cube->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  Step(20);
  DynamicArray<FrictionAnchor> const anchors = GetAnchors(cube);
  DynamicArray<FrictionAnchor> const pending = GetPendingAnchors(cube);
  ASSERT_FALSE(anchors.empty());
  ASSERT_FALSE(pending.empty());
  ASSERT_FALSE(AreEqualAnchors(anchors, pending));
  Real3 const force = cube->GetContactForceWorld(test::ExpectOK{});
  ASSERT_GT(Abs(force[0]), 0_r);
  double const time = _scene->GetTotalSimulationTime();
  StateHandle const state = _scene->CaptureState(test::ExpectOK{});
  Step(1);
  DynamicArray<FrictionAnchor> const advanced = GetPendingAnchors(cube);
  TransformRT const nextPose = cube->GetRootTransform();

  _scene->RestoreState(state, /*releaseImmediately=*/true, test::ExpectOK{});
  ExpectAnchors(anchors, GetAnchors(cube));
  ExpectAnchors(pending, GetPendingAnchors(cube));

  CFrictionAnchorIndex<AnchorSet::Active>& index = GetIndex(cube);
  for (int s = 0; s < isize(index); ++s) {
    index[s] = kSentinelIndex;
  }
  real const forceTolerance = 1e-4_r * Max(1_r, Norm(force));
  for (int i = 0; i < 3; ++i) {
    SCOPED_TRACE("query refresh " + std::to_string(i));
    _scene->UpdateQueries(test::ExpectOK{});
    Real3 const replayedForce = cube->GetContactForceWorld(test::ExpectOK{});
    for (int axis = 0; axis < 3; ++axis) {
      EXPECT_NEAR(force[axis], replayedForce[axis], forceTolerance);
    }
    EXPECT_EQ(time, _scene->GetTotalSimulationTime());
    ExpectAnchors(anchors, GetAnchors(cube));
    ExpectAnchors(pending, GetPendingAnchors(cube));
  }

  for (FrictionAnchor const& anchor : anchors) {
    EXPECT_EQ(
        anchor.point,
        FindAnchorPoint(
            GetAnchors(cube), index, anchor.sample, static_cast<entt::entity>(anchor.collider)))
        << "sample " << anchor.sample;
  }
  Step(1);
  ExpectAnchors(pending, GetAnchors(cube));
  ExpectAnchors(advanced, GetPendingAnchors(cube));
  EXPECT_TRUE(NearEqual(nextPose, cube->GetRootTransform(), 1e-6_r));
}
