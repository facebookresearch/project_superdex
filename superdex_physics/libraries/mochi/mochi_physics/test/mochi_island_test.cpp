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

#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/task_scheduler.h>

// This test peaks at the src implementation to verify island behavior which
// is not directly accessible through the public API
#include <mochi_physics/src/mochi_articulated_body.h>
#include <mochi_physics/src/mochi_compound.h>
#include <mochi_physics/src/mochi_contact.h>
#include <mochi_physics/src/mochi_island.h>

#include <algorithm>
#include <memory>
#include <vector>

using namespace mochi;

static void RemoveAllItemsFromCompound(entt::registry& reg, entt::entity compound) {
  MOCHI_ASSERT(
      (reg.valid(compound) && reg.all_of<TagCompoundActor, CGroupMembers>(compound) &&
       !reg.any_of<TagArticulatedActor>(compound)),
      "Unexpected actor type.");

  auto& members = reg.get<CGroupMembers>(compound);
  std::vector<entt::entity> actorsCopy = members.actors;
  std::vector<entt::entity> constraintsCopy = members.constraints;

  // Remove constraints
  for (auto const& constraint : constraintsCopy) {
    RemoveConstraintFromCompound(reg, compound, constraint, ErrorAssert{});
  }

  // Remove actors
  for (auto const& actor : actorsCopy) {
    RemoveActorFromCompound(reg, compound, actor, ErrorAssert{});
  }
}

/***************************************************************************************************
  Test Fixture Class
*/
class MochiIsland : public test::MochiSceneTestBase {
 public:
  using BaseClass = test::MochiSceneTestBase;
  static constexpr double kTimeStep = 0.01;

  // These tests modify CConservativePotentialColliders right before islands are updated.
  // This lets us test the island code independent of collision detection.
  struct FakePotentialCollider {
    FakePotentialCollider(ActorHandle a, ActorHandle b, ContactType type)
        : actorA(a), actorB(b), contactType(type) {}
    ActorHandle actorA;
    ActorHandle actorB;
    ContactType contactType = {};
  };
  std::vector<FakePotentialCollider> _fakePotentialColliders;

  void SetUp() override {
    BaseClass::SetUp();
    EXPECT_EQ(0, GetNumIslands());

    // Request callbacks during simulation
    auto& reg = GetRegistry();
    island::SetTestCallback_PreIslandUpdate(reg, [this]() { PreIslandUpdate(); });
    island::SetTestCallback_PostIslandUpdate(reg, [this]() { PostIslandUpdate(); });

    // We will be faking CConservativePotentialColliders, but we don't want any actual interactions.
    _scene->EnableLayerContactSymmetric("object", "object", false, test::ExpectOK{});

    // No need for gravity
    _scene->SetGravity({});
  }

  void TearDown() override {
    _fakePotentialColliders.clear();

    // The tests in this file are expected to destroy all their actors.
    // All islands should also be gone after one set at the latest.
    Step();
    EXPECT_EQ(0, GetNumIslands());

    BaseClass::TearDown();
  }

  void Step() {
    // Step the scene. PreIslandUdpate and PostIslandUpdate will be called once each.
    _scene->Step(kTimeStep);
  }

  // Called during simulation just BEFORE islands are updated
  void PreIslandUpdate() {
    // Clear all potential colliders reported by actual collision detection.
    auto& reg = GetRegistry();
    reg.view<CConservativePotentialColliders<mochi::ContactType::Async>>().each(
        [&](auto& pc) { pc.clear(); });
    reg.view<CConservativePotentialColliders<mochi::ContactType::Sync>>().each(
        [&](auto& pc) { pc.clear(); });

    // Report just the "fake" potential colliders (relationships may not be symmetrical)
    for (auto [a, b, type] : _fakePotentialColliders) {
      if (type == ContactType::Async) {
        auto* potentialColliders =
            reg.try_get<CConservativePotentialColliders<mochi::ContactType::Async>>(GetEntity(a));
        EXPECT_NE((decltype(potentialColliders))nullptr, potentialColliders);
        auto colliderEntity = GetEntity(b);
        EXPECT_TRUE(reg.all_of<TagStaticActor>(colliderEntity))
            << "ContactType::Async should only be used for static colliders";
        potentialColliders->emplace_back(colliderEntity);
      } else if (type == ContactType::Sync) {
        auto* potentialColliders =
            reg.try_get<CConservativePotentialColliders<mochi::ContactType::Sync>>(GetEntity(a));
        EXPECT_NE((decltype(potentialColliders))nullptr, potentialColliders);
        potentialColliders->emplace_back(GetEntity(b));
      }
    }
  }

  // Called during simulation just AFTER islands are updated
  void PostIslandUpdate() {
    CheckAllIslands();
    CheckAllPotentialColliders();
  }

  ActorHandle CreateRigidActor(bool isStatic = false) {
    real constexpr kScale = 0.1_r;
    auto&& [coordinates, connectivity] =
        test::CreateMinimalTetMeshUnitCube(Real3{kScale, kScale, kScale});
    auto shape = _scene->GetContext()->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});
    RigidActorParams params;
    params.shape = shape;
    params.layer = "object";
    params.colliderType = ColliderType::Box;
    params.isStatic = isStatic;
    return _scene->CreateRigidActor(params, test::ExpectOK{})->GetHandle();
  }

  int GetNumIslands() const {
    int count = CountEntitiesWith<TagIsland>();
    EXPECT_EQ(count, CountEntitiesWith<CIslandMembers>()); // Every islands should have this too
    return count;
  }

  // Return the island entity containing an actor. Fail the test if not in an island.
  entt::entity GetIsland(entt::entity e) const {
    auto const& reg = GetRegistry();
    auto const* memberInfo = reg.try_get<CIslandMemberInfo const>(e);
    EXPECT_NE(static_cast<CIslandMemberInfo const*>(nullptr), memberInfo);
    EXPECT_TRUE(reg.valid(memberInfo->island));
    return memberInfo->island;
  }
  entt::entity GetIsland(ActorHandle actor) const {
    return GetIsland(GetEntity(actor));
  }

  void CheckAllIslands() const {
    auto const& reg = GetRegistry();

    // Check CIslandMembers for all islands
    reg.view<TagIsland>().each([&](auto island) {
      auto const* members = reg.try_get<CIslandMembers const>(island);
      EXPECT_NE(static_cast<CIslandMembers const*>(nullptr), members);
      for (auto actor : members->actors) {
        EXPECT_TRUE(reg.valid(actor));
        auto const* memberInfo = reg.try_get<CIslandMemberInfo const>(actor);
        EXPECT_NE(static_cast<CIslandMemberInfo const*>(nullptr), memberInfo);
        EXPECT_EQ(island, memberInfo->island);
        EXPECT_FALSE(memberInfo->isNested);
      }
    });

    // Check CIslandMemberInfo for all actors, including nested ones.
    struct Local {
      static void CheckIslandMembershipRecursive(entt::registry const& reg, entt::entity actor) {
        EXPECT_TRUE(reg.valid(actor));
        auto const* islandMembership = reg.try_get<CIslandMemberInfo const>(actor);
        EXPECT_NE(static_cast<CIslandMemberInfo const*>(nullptr), islandMembership);
        entt::entity island = islandMembership->island;
        EXPECT_TRUE(reg.valid(island));
        EXPECT_TRUE((reg.all_of<TagIsland, CIslandMembers>(island)));
        bool isDirectMember = Contains(reg.get<CIslandMembers const>(island).actors, actor);
        EXPECT_EQ(islandMembership->isNested, !isDirectMember);
        if (!isDirectMember) {
          // If an actor is nested in an island, then we should be able to walk up the hierarchy
          auto const* groupMembership = reg.try_get<CGroupMemberInfo const>(actor);
          EXPECT_NE(static_cast<CGroupMemberInfo const*>(nullptr), groupMembership);
          CheckIslandMembershipRecursive(reg, groupMembership->group); // check parents recursively
        }
      }
    };
    reg.view<CIslandMemberInfo const>().each([&](auto actor, auto const& /*islandMembership*/) {
      Local::CheckIslandMembershipRecursive(reg, actor);
    });
  }

  void CheckAllPotentialColliders() const {
    auto const& reg = GetRegistry();

    // Make sure that all potential colliders using ContactType::Async are valid
    reg.view<CConservativePotentialColliders<mochi::ContactType::Async>>().each(
        [&](auto const& potentialColliders) {
          for (auto const& collider : potentialColliders) {
            EXPECT_TRUE(reg.valid(collider.entity));
          }
        });

    // Make sure that all potential colliders using ContactType::Sync are in the same islands.
    reg.view<CConservativePotentialColliders<mochi::ContactType::Sync>>().each(
        [&](auto e, auto const& potentialColliders) {
          entt::entity island = GetIsland(e);
          for (auto const& collider : potentialColliders) {
            EXPECT_TRUE(reg.valid(collider.entity));
            EXPECT_EQ(island, GetIsland(collider.entity)); // Should be in same island by now
          }
        });
  }
};

class MochiIslandParallel : public MochiIsland {
 public:
  void SetUp() override {
    _numWorkerThreads = Min(2, TaskScheduler::GetNumSupportedLogicalProcessors());
    MochiIsland::SetUp();
  }
};

/***************************************************************************************************
  Test Cases
*/

// Verifies that a static articulated link remains stable while serving as an async collider.
TEST_F(MochiIslandParallel, StaticArticulatedLinkIsStableAsyncCollider) {
  TransformRT const prescribedRoot{Real3{0.25_r, -0.5_r, 0.75_r}};
  auto linkShape = test::CreateUnitCubeTetMeshShape(_mochiContext);

  ArticulatedActorParams params;
  params.worldFromRoot = prescribedRoot;
  params.joints = {
      {.type = ArticulatedJointType::Hard},
      {.type = ArticulatedJointType::Revolute, .axis = Real3{0_r, 0_r, 1_r}}};
  params.links = {
      {.parentLink = -1, .shape = linkShape, .layer = "object", .colliderType = ColliderType::Box},
      {.parentLink = 0, .shape = linkShape, .layer = "object", .colliderType = ColliderType::None}};
  auto* articulation = _scene->CreateArticulatedActor(params, test::ExpectOK{});
  ASSERT_NE(nullptr, articulation);
  ActorHandle const articulationHandle = articulation->GetHandle();
  ASSERT_EQ(1, articulation->GetNumDofs());
  articulation->SetArticulatedJointVelocities(DynamicArray<real>{1_r}, test::ExpectOK{});

  auto const& links = articulation->GetNestedLinkActors(test::ExpectOK{});
  ASSERT_EQ(2, isize(links));
  ActorHandle const rootHandle = links[0];
  ActorHandle const childHandle = links[1];
  auto* root = _scene->GetActor(rootHandle);
  auto* child = _scene->GetActor(childHandle);
  ASSERT_NE(nullptr, root);
  ASSERT_NE(nullptr, child);
  TransformRT const rootBefore = root->GetRootTransform();
  TransformRT const childBefore = child->GetRootTransform();

  ActorHandle const cubeHandle = CreateRigidActor();
  TransformRT cubeTransform = rootBefore;
  // Center the 0.1 cube inside the unit root collider so bounds overlap without contact response.
  cubeTransform.SetTranslation(rootBefore.GetTranslation() + Real3{0.45_r, 0.45_r, 0.45_r});
  _scene->GetActor(cubeHandle)->SetRootTransform(cubeTransform, test::ExpectOK{});

  auto& reg = GetRegistry();
  entt::entity const rootEntity = GetEntity(rootHandle);
  entt::entity const childEntity = GetEntity(childHandle);
  entt::entity const cubeEntity = GetEntity(cubeHandle);
  EXPECT_TRUE(reg.all_of<TagStaticActor>(rootEntity));
  EXPECT_FALSE(reg.all_of<TagStaticActor>(childEntity));

  _fakePotentialColliders.emplace_back(cubeHandle, rootHandle, ContactType::Async);

  test::SetSceneIntegrationMethod(_scene, IntegrationMethod::DIRK33);
  Step();
  EXPECT_NE(GetIsland(cubeHandle), GetIsland(articulationHandle));

  auto const* potentialColliders =
      reg.try_get<CPotentialColliders<ContactType::Async> const>(cubeEntity);
  ASSERT_NE(nullptr, potentialColliders);
  EXPECT_TRUE(
      std::any_of(
          potentialColliders->begin(),
          potentialColliders->end(),
          [rootEntity](auto const& collider) { return collider.entity == rootEntity; }));

  int constexpr kNumSteps = 16;
  for (int step = 1; step < kNumSteps; ++step) {
    Step();
  }

  EXPECT_FALSE(
      NearEqual(childBefore.GetRotation(), child->GetRootTransform().GetRotation(), kTolerance));
  EXPECT_EQ(rootBefore.GetRotation(), root->GetRootTransform().GetRotation());
  EXPECT_EQ(rootBefore.GetTranslation(), root->GetRootTransform().GetTranslation());
  auto const& linkTransforms =
      reg.get<CArticulatedLinkTransforms<TimeStep::Current> const>(GetEntity(articulationHandle));
  auto const& rootState = reg.get<CRigidState<TimeStep::Current> const>(rootEntity).value;
  EXPECT_EQ(linkTransforms[0].GetRotation(), rootState.GetRotation());
  EXPECT_EQ(linkTransforms[0].GetTranslation(), rootState.GetTranslation());

  _scene->DestroyActor(cubeHandle);
  _scene->DestroyActor(articulationHandle);
}

TEST_F(MochiIsland, StaticActorNoIsland) {
  auto& reg = GetRegistry();

  // Static actors should not be in any island
  ActorHandle actor = CreateRigidActor(true);
  Step();
  EXPECT_FALSE(reg.any_of<CIslandMemberInfo>(GetEntity(actor)));
  EXPECT_EQ(0, GetNumIslands());

  // Cleanup
  _scene->DestroyActor(actor);
}

TEST_F(MochiIsland, SingleDynamicActor) {
  auto& reg = GetRegistry();

  // A single dynamic actor should have its own island (after one step at the latest)
  ActorHandle actor = CreateRigidActor(false);
  Step();
  EXPECT_EQ(1, GetNumIslands());
  entt::entity island = GetIsland(actor);
  auto const& members = reg.get<CIslandMembers const>(island);
  EXPECT_EQ(1, isize(members.actors));
  EXPECT_EQ(GetEntity(actor), members.actors[0]);

  // Destroying the actor should destroy the island after one step at the latest.
  _scene->DestroyActor(actor);
  Step();
  EXPECT_EQ(0, GetNumIslands());
}

TEST_F(MochiIsland, WithCompound) {
  // Create 2 actors
  ActorHandle a = CreateRigidActor();
  ActorHandle b = CreateRigidActor();
  Step();

  // They should be in separate islands.
  EXPECT_EQ(2, GetNumIslands());
  EXPECT_NE(GetIsland(a), GetIsland(b));

  // Putting them in a compound, should merge the islands
  auto& reg = GetRegistry();
  auto compound = reg.create();
  InitCompoundActor(reg, compound, test::ExpectOK{});
  AddActorToCompound(reg, compound, GetEntity(a), test::ExpectOK{});
  AddActorToCompound(reg, compound, GetEntity(b), test::ExpectOK{});

  Step();
  EXPECT_EQ(1, GetNumIslands());
  EXPECT_EQ(GetIsland(a), GetIsland(b));
  EXPECT_EQ(GetIsland(a), GetIsland(compound));

  // Destroying an actor in the compound should be OK
  _scene->DestroyActor(a);
  Step();
  EXPECT_EQ(1, GetNumIslands());
  EXPECT_EQ(GetIsland(b), GetIsland(compound));

  // Replace it with a new actor
  auto c = CreateRigidActor();
  AddActorToCompound(reg, compound, GetEntity(c), test::ExpectOK{});
  Step();
  EXPECT_EQ(1, GetNumIslands());
  EXPECT_EQ(GetIsland(b), GetIsland(c));
  EXPECT_EQ(GetIsland(b), GetIsland(compound));

  // Destroy the compound and restore the two islands
  RemoveAllItemsFromCompound(reg, compound);
  reg.destroy(compound);
  island::CreateForActor(reg, GetEntity(b));
  island::CreateForActor(reg, GetEntity(c));

  // When we step the scene, those islands should not merge because there is no contact
  Step();
  EXPECT_EQ(2, GetNumIslands());
  EXPECT_NE(GetIsland(b), GetIsland(c));

  // Cleanup
  _scene->DestroyActor(b);
  _scene->DestroyActor(c);
}

TEST_F(MochiIsland, VariousContactType) {
  auto& reg = GetRegistry();

  // Create 3 actors
  ActorHandle actorA = CreateRigidActor();
  ActorHandle actorB = CreateRigidActor();

  // Sometimes actorB will be nested in a compound
  for (int nested = 0; nested < 2; ++nested) {
    ActorHandle actorC;
    entt::entity compound = entt::null;
    if (nested) {
      // Put actorB in a nested compound. Also put a second actor in that compound
      // just to prove that it remains in the same island as actorB.
      actorC = CreateRigidActor();

      compound = reg.create();
      InitCompoundActor(reg, compound, test::ExpectOK{});
      AddActorToCompound(reg, compound, GetEntity(actorB), test::ExpectOK{});
      AddActorToCompound(reg, compound, GetEntity(actorC), test::ExpectOK{});
    }

    // Try various types of contact interactions (not necessarily symetrical)
    for (int enableAB = 0; enableAB < 2; ++enableAB) {
      for (int enableBA = 0; enableBA < 2; ++enableBA) {
        auto abContactType = enableAB ? ContactType::Sync : ContactType::None;
        auto baContactType = enableBA ? ContactType::Sync : ContactType::None;

        // Override CConservativePotentialColliders so contact code sees no potential colliders
        _fakePotentialColliders.clear();

        // The actors should end up in two islands.
        Step();
        EXPECT_EQ(2, GetNumIslands());
        EXPECT_NE(GetIsland(actorA), GetIsland(actorB));
        if (nested) {
          EXPECT_EQ(GetIsland(actorB), GetIsland(actorC)); // C is in the same compound as B
        }

        // Override CConservativePotentialColliders so contact code sees potential colliders
        // according to abContactType and baContactType.
        _fakePotentialColliders.clear();
        _fakePotentialColliders.emplace_back(actorA, actorB, abContactType);
        _fakePotentialColliders.emplace_back(actorB, actorA, baContactType);

        // The actors islands should split or merge according to ContactType
        Step();
        bool shouldMerge =
            (abContactType == ContactType::Sync) || (baContactType == ContactType::Sync);
        if (shouldMerge) {
          EXPECT_EQ(1, GetNumIslands());
          EXPECT_EQ(GetIsland(actorA), GetIsland(actorB));
        } else {
          EXPECT_EQ(2, GetNumIslands());
          EXPECT_NE(GetIsland(actorA), GetIsland(actorB));
        }
        if (nested) {
          EXPECT_EQ(GetIsland(actorB), GetIsland(actorC)); // C is in the same compound as B
        }
      }
    }

    // Cleanup
    if (compound != entt::null) {
      RemoveAllItemsFromCompound(reg, compound);
      reg.destroy(compound);
    }
    _scene->DestroyActor(actorC);
  }

  // Cleanup
  _scene->DestroyActor(actorA);
  _scene->DestroyActor(actorB);
}

TEST_F(MochiIsland, MergeSplitSameStep) {
  // Create 4 actors.
  ActorHandle a = CreateRigidActor();
  ActorHandle b = CreateRigidActor();
  ActorHandle c = CreateRigidActor();
  ActorHandle d = CreateRigidActor();

  // Disable actual contact so that we are the only ones setting CConservativePotentialColliders
  _scene->EnableLayerContactSymmetric("object", "object", false, test::ExpectOK{});

  // Override CConservativePotentialColliders so that two islands will be formed: {a, b} and {c, d}
  _fakePotentialColliders.clear();
  _fakePotentialColliders.emplace_back(a, b, ContactType::Sync);
  _fakePotentialColliders.emplace_back(c, d, ContactType::Sync);
  Step();
  EXPECT_EQ(2, GetNumIslands());
  EXPECT_EQ(GetIsland(a), GetIsland(b));
  EXPECT_EQ(GetIsland(c), GetIsland(d));
  EXPECT_NE(GetIsland(b), GetIsland(c));

  // Cause islands {b, c} and {a, d} to be formed (requires a merge and a split in the same step)
  _fakePotentialColliders.clear();
  _fakePotentialColliders.emplace_back(b, c, ContactType::Sync);
  _fakePotentialColliders.emplace_back(a, d, ContactType::Sync);
  Step();
  EXPECT_EQ(2, GetNumIslands());
  EXPECT_EQ(GetIsland(a), GetIsland(d));
  EXPECT_EQ(GetIsland(b), GetIsland(c));
  EXPECT_NE(GetIsland(a), GetIsland(b));

  // Cause isands {a, c} and {b, d} to be formed.
  _fakePotentialColliders.clear();
  _fakePotentialColliders.emplace_back(a, c, ContactType::Sync);
  _fakePotentialColliders.emplace_back(b, d, ContactType::Sync);
  Step();
  EXPECT_EQ(2, GetNumIslands());
  EXPECT_EQ(GetIsland(a), GetIsland(c));
  EXPECT_EQ(GetIsland(b), GetIsland(d));
  EXPECT_NE(GetIsland(a), GetIsland(b));

  // Finally, cause islands {B, C} and {A} and {D} to be formed.
  _fakePotentialColliders.clear();
  _fakePotentialColliders.emplace_back(b, c, ContactType::Sync);
  _fakePotentialColliders.emplace_back(d, c, ContactType::None); // Does not cause merger
  Step();
  EXPECT_EQ(3, GetNumIslands());
  EXPECT_EQ(GetIsland(b), GetIsland(c));
  EXPECT_NE(GetIsland(a), GetIsland(b));
  EXPECT_NE(GetIsland(d), GetIsland(b));

  // Cleanup
  _scene->DestroyActor(a);
  _scene->DestroyActor(b);
  _scene->DestroyActor(c);
  _scene->DestroyActor(d);
}

TEST_F(MochiIsland, SplitMultiple) {
  // Create 10 islands with 2 actors each
  std::vector<ActorHandle> actors(20);
  for (auto& handle : actors) {
    handle = CreateRigidActor();
  }
  for (int i = 0; i < isize(actors); i += 2) {
    _fakePotentialColliders.emplace_back(actors[i], actors[i + 1], ContactType::Sync);
  }
  Step();
  EXPECT_EQ(10, GetNumIslands());

  // Cause half of the islands to split in a single step.
  for (int i = isize(_fakePotentialColliders) - 1; i >= 0; i -= 2) {
    _fakePotentialColliders.erase(_fakePotentialColliders.begin() + i);
  }
  EXPECT_EQ(5, isize(_fakePotentialColliders));
  Step();
  EXPECT_EQ(15, GetNumIslands());

  // Cleanup
  for (auto& handle : actors) {
    _scene->DestroyActor(handle);
  }
}

TEST_F(MochiIsland, SplitPermutation) {
  auto& reg = GetRegistry();

  // Create multiple dynamic actors in one island
  int constexpr kNumActors = 4;
  std::vector<ActorHandle> actors(kNumActors);
  for (auto& a : actors) {
    a = CreateRigidActor();
  }

  // Helper: Fakes potential colliders to cause a single island to form
  auto formSingleIsland = [&]() {
    _fakePotentialColliders.clear();
    for (int i = 0; i < kNumActors; ++i) {
      for (int j = i + 1; j < kNumActors; ++j) {
        _fakePotentialColliders.emplace_back(actors[i], actors[j], ContactType::Sync);
      }
    }
    // Uses island::PreStep not Scene::Step for speed.
    island::PreStep(GetRegistry());
    CheckAllIslands();
    CheckAllPotentialColliders();
    EXPECT_EQ(1, GetNumIslands());
  };

  // Test all permutations of potential colliders using bit masks.
  int mask[kNumActors] = {};
  int constexpr kMaskEnd = (1 << kNumActors);
  for (mask[0] = 0; mask[0] < kMaskEnd; mask[0]++) {
    for (mask[1] = 0; mask[1] < kMaskEnd; mask[1]++) {
      for (mask[2] = 0; mask[2] < kMaskEnd; mask[2]++) {
        for (mask[3] = 0; mask[3] < kMaskEnd; mask[3]++) {
          // Run extra checks in optimized builds. They would make debug builds too slow.
          int const kNumOrderVariations = MOCHI_DEBUG ? 1 : 4;
          for (int order = 0; order < kNumOrderVariations; ++order) {
            formSingleIsland();
            _fakePotentialColliders.clear();
            for (int i = 0; i < kNumActors; ++i) {
              for (int j = 0; j < kNumActors; ++j) {
                if (mask[i] & (1 << j)) {
                  _fakePotentialColliders.emplace_back(actors[i], actors[j], ContactType::Sync);
                }
              }
            }

            // Sometimes reverse the order of island members, or potential colliders, or both.
            // If any part of the island splitting behavior is order dependent, then this will
            // hopefully show it.
            switch (order) {
              case 1:
                std::reverse(_fakePotentialColliders.begin(), _fakePotentialColliders.end());
                break;
              case 2:
                reg.view<CIslandMembers>().each([](auto& members) {
                  std::reverse(members.actors.begin(), members.actors.end());
                });
                break;
              case 3:
                std::reverse(_fakePotentialColliders.begin(), _fakePotentialColliders.end());
                reg.view<CIslandMembers>().each([](auto& members) {
                  std::reverse(members.actors.begin(), members.actors.end());
                });
                break;
            }

            // Uses island::PreStep not Scene::Step for speed.
            island::PreStep(GetRegistry());
            EXPECT_LE(1, GetNumIslands());
            EXPECT_GE(kNumActors, GetNumIslands());
            CheckAllIslands();
            CheckAllPotentialColliders();
          }
        }
      }
    }
  }

  // Cleanup
  for (auto a : actors) {
    _scene->DestroyActor(a);
  }
}

TEST_F(MochiIsland, ForceSingleIsland) {
  // Create 2 actors
  ActorHandle a = CreateRigidActor();
  ActorHandle b = CreateRigidActor();
  MOCHI_DEFER(_scene->DestroyActor(a));
  MOCHI_DEFER(_scene->DestroyActor(b));
  Step();

  // Helper
  auto& reg = GetRegistry();
  auto expectSingleIsland = [&](bool expectSingleIsland) {
    // Get islands and make sure they are valid.
    auto islandA = reg.get<CIslandMemberInfo const>(GetEntity(a)).island;
    auto islandB = reg.get<CIslandMemberInfo const>(GetEntity(b)).island;
    EXPECT_TRUE(reg.valid(islandA));
    EXPECT_TRUE(reg.valid(islandB));

    // But are they the same?
    EXPECT_EQ(expectSingleIsland, (islandA == islandB));
  };

  // These actors should be in separate islands because we disabled collision.
  expectSingleIsland(false);

  // Add TagForceSingleIsland to actor A
  reg.emplace<TagForceSingleIsland>(GetEntity(a));
  Step();

  // These actors should be in the same island now.
  expectSingleIsland(true);

  // Add TagForceSingleIsland to actor B. Nothing should change.
  reg.emplace<TagForceSingleIsland>(GetEntity(b));
  Step();
  expectSingleIsland(true);

  // Remove the tag from actor A. Nothing should change because B still has it.
  reg.erase<TagForceSingleIsland>(GetEntity(a));
  Step();
  expectSingleIsland(true);

  // Remove the tag from actor B. Now, the island should split up.
  reg.erase<TagForceSingleIsland>(GetEntity(b));
  Step();
  expectSingleIsland(false);

  // This time, use the public API to force a single island.
  _scene->SetForceSingleIsland(true);
  Step();
  expectSingleIsland(true);
  EXPECT_TRUE(_scene->GetForceSingleIsland());

  _scene->SetForceSingleIsland(false);
  Step();
  expectSingleIsland(false);
  EXPECT_FALSE(_scene->GetForceSingleIsland());

  // Combine the public API with the tag.
  _scene->SetForceSingleIsland(true);
  reg.emplace<TagForceSingleIsland>(GetEntity(a));
  Step();
  expectSingleIsland(true);
  _scene->SetForceSingleIsland(false);
  Step();
  expectSingleIsland(true); // Still one island because of tag
  reg.erase<TagForceSingleIsland>(GetEntity(a));
  Step();
  expectSingleIsland(false);
  _scene->SetForceSingleIsland(true);
  reg.emplace<TagForceSingleIsland>(GetEntity(a));
  Step();
  expectSingleIsland(true);
  reg.erase<TagForceSingleIsland>(GetEntity(a));
  Step();
  expectSingleIsland(true); // Still one island because of public API call
  _scene->SetForceSingleIsland(false);
  Step();
  expectSingleIsland(false);

  // Cleanup
  _scene->DestroyActor(a);
  _scene->DestroyActor(b);
}

TEST_F(MochiIsland, ShouldRunSingleThreaded) {
  entt::registry reg;
  entt::entity const islandEntity = reg.create();
  reg.emplace<TagIsland>(islandEntity);
  auto& dofInfo = reg.emplace<CIslandDofInfo>(islandEntity);
  CIslandDescendants descendants;

  auto addRigidActor = [&](int numContactSamples) {
    entt::entity const actor = reg.create();
    reg.emplace<TagRigidActor>(actor);
    descendants.actors.push_back(actor);
    if (numContactSamples > 0) {
      reg.emplace<TagUseContact>(actor);
      reg.emplace<CContactSamples<TimeStep::Current>>(actor, numContactSamples);
    }
    return actor;
  };

  // Check the DoF threshold.
  addRigidActor(/*numContactSamples*/ 0);
  dofInfo.dofsSize = 127;
  EXPECT_TRUE(island::ShouldRunSingleThreaded(reg, islandEntity, descendants));
  dofInfo.dofsSize = 128;
  EXPECT_FALSE(island::ShouldRunSingleThreaded(reg, islandEntity, descendants));
  dofInfo.dofsSize = 0;

  // Check the contact sample threshold.
  addRigidActor(/*numContactSamples*/ 3000);
  entt::entity const secondActor = addRigidActor(/*numContactSamples*/ 2999);
  auto& secondSamples = reg.get<CContactSamples<TimeStep::Current>>(secondActor);
  EXPECT_TRUE(island::ShouldRunSingleThreaded(reg, islandEntity, descendants));
  secondSamples.positions.emplace_back();
  EXPECT_FALSE(island::ShouldRunSingleThreaded(reg, islandEntity, descendants));

  // Actors with active boundary faces are counted by their active faces, not by their contact
  // samples, so that the count is valid before the first step populates the active samples.
  using Element = CFemSurfaceDiscretizationP1Q1::ElementT;
  auto mesh = std::make_shared<TriangularMesh const>(test::CreateMinimalTriMeshUnitCube());
  DynamicArray<Element> faces;
  for (int i = 0; i < mesh->GetNumElements(); ++i) {
    faces.emplace_back(
        i,
        mesh->GetActiveNodeCoordinates(),
        Unflatten<Int3 const>(mesh->GetActiveNodesFlatConnectivity()));
  }
  DynamicArray<int> const activeFaces = {0, 1};
  reg.emplace<CActiveBoundaryFaces>(
      secondActor, mesh, MakeConstSpan(activeFaces), MakeConstSpan(faces));
  EXPECT_TRUE(island::ShouldRunSingleThreaded(reg, islandEntity, descendants));

  // Check that non-rigid descendants are rejected.
  entt::entity const softActor = reg.create();
  reg.emplace<TagSoftActor>(softActor);
  descendants.actors.push_back(softActor);
  EXPECT_FALSE(island::ShouldRunSingleThreaded(reg, islandEntity, descendants));
}
