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

#include <mochi_core/utils/defer.h>
#include <mochi_physics/src/mochi_contact_filter.h>
#include <mochi_physics/src/mochi_island.h>

#include <algorithm>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

using namespace mochi;

class MochiContactFilter : public test::MochiSceneTestBase {
 public:
  Actor* CreateActor(std::string_view layer) {
    auto* mochiContext = _scene->GetContext();
    auto meshData = test::CreateMinimalTetMeshUnitCube();
    ShapeHandle shape = mochiContext->CreateTetMeshShape(
        Flatten(MakeConstSpan(meshData.first)),
        Flatten(MakeConstSpan(meshData.second)),
        test::ExpectOK{});

    RigidActorParams params;
    params.colliderType = ColliderType::Sphere;
    params.layer = layer;
    params.isStatic = false;
    params.shape = shape;

    return _scene->CreateRigidActor(params, test::ExpectOK{});
  }

  [[nodiscard]] Actor* CreateContactFilterArticulatedActor(std::initializer_list<int> parentLinks) {
    auto* mochiContext = _scene->GetContext();
    auto meshData = test::CreateMinimalTetMeshUnitCube();
    ShapeHandle shape = mochiContext->CreateTetMeshShape(
        Flatten(MakeConstSpan(meshData.first)),
        Flatten(MakeConstSpan(meshData.second)),
        test::ExpectOK{});

    ArticulatedActorParams params;
    params.joints.reserve(parentLinks.size());
    params.links.reserve(parentLinks.size());
    for (int parentLink : parentLinks) {
      ArticulatedJointParams joint;
      joint.type = parentLink < 0 ? ArticulatedJointType::Free : ArticulatedJointType::Spherical;
      params.joints.push_back(joint);

      ArticulatedLinkParams link;
      link.parentLink = parentLink;
      link.shape = shape;
      link.colliderType = ColliderType::Box;
      params.links.push_back(link);
    }

    return _scene->CreateArticulatedActor(params, test::ExpectOK{});
  }

  [[nodiscard]] Actor* CreateContactFilterSoftSkinnedActor() {
    SoftSkinnedActorParams params;
    params.skeletonParams.joints = {{.type = ArticulatedJointType::Free}};
    params.skeletonParams.links = {
        {.parentLink = -1,
         .shape = test::CreateUnitCubeTetMeshShape(_mochiContext),
         .colliderType = ColliderType::Box}};

    SoftActorParams softParams;
    softParams.shape = test::CreateUnitCubeTetSoftShape(_mochiContext);
    softParams.hasGravity = false;
    params.softParams.push_back(softParams);

    return _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});
  }
};

TEST_F(MochiContactFilter, GetContactLayerId) {
  // Test internals using the ECS registry
  auto& reg = test::GetRegistry(_scene);
  auto& table = reg.ctx<CContactFilterTable>();
  EXPECT_EQ(0, table.layerNameToId.size());

  // Empty string
  EXPECT_EQ(ContactLayerId::None, GetContactLayerId(table, ""));
  EXPECT_EQ(ContactLayerId::None, GetOrAddContactLayerId(table, ""));
  EXPECT_EQ(0, table.layerNameToId.size());

  // Non-empty string
  EXPECT_EQ(ContactLayerId::None, GetContactLayerId(table, "fred"));
  auto fred = GetOrAddContactLayerId(table, "fred");
  EXPECT_NE(ContactLayerId::None, fred);
  EXPECT_EQ(fred, GetContactLayerId(table, "fred"));
  EXPECT_EQ(1, table.layerNameToId.size());

  // Again
  EXPECT_EQ(ContactLayerId::None, GetContactLayerId(table, "george"));
  auto george = GetOrAddContactLayerId(table, "george");
  EXPECT_NE(ContactLayerId::None, george);
  EXPECT_NE(fred, george);
  EXPECT_EQ(fred, GetContactLayerId(table, "fred"));
  EXPECT_EQ(george, GetContactLayerId(table, "george"));
  EXPECT_EQ(2, table.layerNameToId.size());
}

TEST_F(MochiContactFilter, LayerVsLayer) {
  // Contact must be enabled between all layers by default.
  EXPECT_TRUE(_scene->IsLayerContactEnabled("LayerA", "LayerA"));
  EXPECT_TRUE(_scene->IsLayerContactEnabled("LayerA", "LayerB"));
  EXPECT_TRUE(_scene->IsLayerContactEnabled("LayerB", "LayerA"));
  EXPECT_TRUE(_scene->IsLayerContactEnabled("LayerA", ""));
  EXPECT_TRUE(_scene->IsLayerContactEnabled("", "LayerB"));
  EXPECT_TRUE(_scene->IsLayerContactEnabled("", ""));

  // Set and check various combinations
  for (int enableAA = 0; enableAA < 2; ++enableAA) {
    for (int enableAB = 0; enableAB < 2; ++enableAB) {
      for (int enableBA = 0; enableBA < 2; ++enableBA) {
        _scene->EnableLayerContactAsymmetric("LayerA", "LayerA", enableAA, test::ExpectOK{});
        EXPECT_EQ(enableAA, _scene->IsLayerContactEnabled("LayerA", "LayerA"));
        _scene->EnableLayerContactAsymmetric("LayerA", "LayerB", enableAB, test::ExpectOK{});
        EXPECT_EQ(enableAB, _scene->IsLayerContactEnabled("LayerA", "LayerB"));
        _scene->EnableLayerContactAsymmetric("LayerB", "LayerA", enableBA, test::ExpectOK{});
        EXPECT_EQ(enableBA, _scene->IsLayerContactEnabled("LayerB", "LayerA"));
      }
    }
  }

  // Try to set an empty string
  _scene->EnableLayerContactAsymmetric("LayerA", "LayerB", true, test::ExpectOK{});
  EXPECT_TRUE(_scene->IsLayerContactEnabled("LayerA", "LayerB"));
  _scene->EnableLayerContactAsymmetric("LayerA", "", false, test::ExpectNotOK{});
  _scene->EnableLayerContactAsymmetric("", "LayerB", false, test::ExpectNotOK{});
  EXPECT_TRUE(_scene->IsLayerContactEnabled("LayerA", "LayerB")); // no change
}

TEST_F(MochiContactFilter, EnumerateContactLayerNames) {
  // No names in the table
  std::vector<std::string> names;
  _scene->EnumerateContactLayerNames([&](auto name) { names.emplace_back(name); });
  EXPECT_EQ(0, names.size());

  // EnableLayerContactAsymmetric adds names (only once each)
  _scene->EnableLayerContactAsymmetric("A", "B", false, test::ExpectOK{});
  _scene->EnableLayerContactAsymmetric("A", "B", true, test::ExpectOK{});
  _scene->EnumerateContactLayerNames([&](auto name) { names.emplace_back(name); });
  std::sort(names.begin(), names.end());
  EXPECT_EQ(2, names.size());
  EXPECT_EQ("A", names[0]);
  EXPECT_EQ("B", names[1]);

  // Actors also put names into the list (only once each)
  auto* actorA = CreateActor("A");
  auto* actorA2 = CreateActor("A");
  auto* actorB = CreateActor("B");
  auto* actorC = CreateActor("C");
  MOCHI_DEFER(_scene->DestroyActor(actorA));
  MOCHI_DEFER(_scene->DestroyActor(actorA2));
  MOCHI_DEFER(_scene->DestroyActor(actorB));
  MOCHI_DEFER(_scene->DestroyActor(actorC));
  names.clear();
  _scene->EnumerateContactLayerNames([&](auto name) { names.emplace_back(name); });
  std::sort(names.begin(), names.end());
  EXPECT_EQ(3, names.size());
  EXPECT_EQ("A", names[0]);
  EXPECT_EQ("B", names[1]);
  EXPECT_EQ("C", names[2]);
}

TEST_F(MochiContactFilter, ActorVsActor) {
  auto* sceneImpl = assert_cast<SceneImpl*>(_scene);

  // Create some actors
  auto actorA = CreateActor("SomeLayer")->GetHandle();
  auto actorB = CreateActor("SomeLayer")->GetHandle();
  auto actorC = CreateActor("SomeLayer")->GetHandle();

  // Force them into one island
  _scene->SetForceSingleIsland(true);
  island::PreStep(GetRegistry());

  // Contact enabled by default
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorA, actorA, test::ExpectOK{}));
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorA, actorB, test::ExpectOK{}));
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorB, actorA, test::ExpectOK{}));
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorA, actorC, test::ExpectOK{}));

  // Disable A-vs-A contact
  _scene->EnableActorContactAsymmetric(
      actorA, actorA, /*enable*/ false, IncludeNestedActors::No, test::ExpectOK{});
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(actorA, actorA, test::ExpectOK{}));

  // Disable A-vs-B contact
  _scene->EnableActorContactAsymmetric(
      actorA, actorB, /*enable*/ false, IncludeNestedActors::No, test::ExpectOK{});
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(actorA, actorB, test::ExpectOK{}));
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorB, actorA, test::ExpectOK{})); // unchanged

  // Disable B-vs-A contact
  _scene->EnableActorContactAsymmetric(
      actorB, actorA, /*enable*/ false, IncludeNestedActors::No, test::ExpectOK{});
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(actorA, actorB, test::ExpectOK{}));
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(actorB, actorA, test::ExpectOK{}));

  // Test symmetric overload
  _scene->EnableActorContactSymmetric(
      actorA, actorB, /*enable*/ true, IncludeNestedActors::No, test::ExpectOK{});
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorA, actorB, test::ExpectOK{}));
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorB, actorA, test::ExpectOK{}));
  _scene->EnableActorContactSymmetric(
      actorA, actorB, /*enable*/ false, IncludeNestedActors::No, test::ExpectOK{});

  // Disable A-vs-C contact
  _scene->EnableActorContactAsymmetric(
      actorA, actorC, /*enable*/ false, IncludeNestedActors::No, test::ExpectOK{});
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(actorA, actorC, test::ExpectOK{}));
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorC, actorA, test::ExpectOK{})); // unchanged

  // Revert override for A-vs-B contact
  _scene->EnableActorContactAsymmetric(
      actorA, actorB, /*enable*/ true, IncludeNestedActors::No, test::ExpectOK{});
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorA, actorB, test::ExpectOK{}));
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(actorB, actorA, test::ExpectOK{})); // unchanged

  // Verify IsActorContactEnabled() is not affected by layers
  _scene->EnableLayerContactAsymmetric("SomeLayer", "SomeLayer", false, test::ExpectOK{});
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorA, actorB, test::ExpectOK{})); // unchanged
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(actorB, actorA, test::ExpectOK{})); // unchanged
  _scene->EnableLayerContactAsymmetric("SomeLayer", "SomeLayer", true, test::ExpectOK{});
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorA, actorB, test::ExpectOK{})); // unchanged
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(actorB, actorA, test::ExpectOK{})); // unchanged

  // Peak at the registry to ensure entries are removed from the table as actors are destroyed.
  auto const& reg = test::GetRegistry(_scene);
  auto const& table = reg.ctx<CContactFilterTable>();
  EXPECT_EQ(3, table.entitiesWithNoContact.size()); // AA, BA, AC
  _scene->DestroyActor(actorB);
  EXPECT_EQ(2, table.entitiesWithNoContact.size()); // AA, AC
  _scene->DestroyActor(actorC);
  EXPECT_EQ(1, table.entitiesWithNoContact.size()); // AA
  _scene->DestroyActor(actorA);
  EXPECT_EQ(0, table.entitiesWithNoContact.size()); // All gone
}

TEST_F(MochiContactFilter, ActorVsActor_InvalidIncludeNestedActorsErrorsWithoutMutation) {
  auto* sceneImpl = assert_cast<SceneImpl*>(_scene);

  auto actorA = CreateActor("SomeLayer")->GetHandle();
  auto actorB = CreateActor("SomeLayer")->GetHandle();

  _scene->EnableActorContactSymmetric(
      actorA, actorB, /*enable*/ false, IncludeNestedActors::Count, test::ExpectNotOK{});
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorA, actorB, test::ExpectOK{}));
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorB, actorA, test::ExpectOK{}));

  _scene->EnableActorContactAsymmetric(
      actorA,
      actorB,
      /*enable*/ false,
      static_cast<IncludeNestedActors>(999),
      test::ExpectNotOK{});
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorA, actorB, test::ExpectOK{}));
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorB, actorA, test::ExpectOK{}));
}

TEST_F(MochiContactFilter, ActorVsActor_IncludeNestedActorsAppliesResolvedCrossProduct) {
  auto* sceneImpl = assert_cast<SceneImpl*>(_scene);

  auto* parentA = CreateContactFilterArticulatedActor({-1, 0});
  auto* parentB = CreateContactFilterArticulatedActor({-1, 0, 1});

  auto const& linksA = parentA->GetNestedLinkActors(test::ExpectOK{});
  auto const& linksB = parentB->GetNestedLinkActors(test::ExpectOK{});
  ASSERT_EQ(2, isize(linksA));
  ASSERT_EQ(3, isize(linksB));

  std::vector<ActorHandle> handlesA = {parentA->GetHandle()};
  handlesA.insert(handlesA.end(), linksA.begin(), linksA.end());
  std::vector<ActorHandle> handlesB = {parentB->GetHandle()};
  handlesB.insert(handlesB.end(), linksB.begin(), linksB.end());

  _scene->SetForceSingleIsland(true);
  island::PreStep(GetRegistry());

  _scene->EnableActorContactSymmetric(
      parentA->GetHandle(),
      parentB->GetHandle(),
      /*enable*/ false,
      IncludeNestedActors::No,
      test::ExpectOK{});

  for (auto actorA : handlesA) {
    for (auto actorB : handlesB) {
      bool const isExactParentPair =
          actorA == parentA->GetHandle() && actorB == parentB->GetHandle();
      if (isExactParentPair) {
        EXPECT_FALSE(sceneImpl->IsActorContactEnabled(actorA, actorB, test::ExpectOK{}));
        EXPECT_FALSE(sceneImpl->IsActorContactEnabled(actorB, actorA, test::ExpectOK{}));
      } else {
        EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorA, actorB, test::ExpectOK{}));
        EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorB, actorA, test::ExpectOK{}));
      }
    }
  }

  _scene->EnableActorContactSymmetric(
      parentA->GetHandle(),
      parentB->GetHandle(),
      /*enable*/ true,
      IncludeNestedActors::No,
      test::ExpectOK{});
  _scene->EnableActorContactSymmetric(
      linksA[0], linksA[1], /*enable*/ true, IncludeNestedActors::No, test::ExpectOK{});
  _scene->EnableActorContactSymmetric(
      linksB[0], linksB[1], /*enable*/ true, IncludeNestedActors::No, test::ExpectOK{});
  _scene->EnableActorContactSymmetric(
      linksB[1], linksB[2], /*enable*/ true, IncludeNestedActors::No, test::ExpectOK{});

  _scene->EnableActorContactAsymmetric(
      parentA->GetHandle(),
      parentB->GetHandle(),
      /*enable*/ false,
      IncludeNestedActors::Yes,
      test::ExpectOK{});

  for (auto actorA : handlesA) {
    for (auto actorB : handlesB) {
      EXPECT_FALSE(sceneImpl->IsActorContactEnabled(actorA, actorB, test::ExpectOK{}));
      EXPECT_TRUE(sceneImpl->IsActorContactEnabled(actorB, actorA, test::ExpectOK{}));
    }
  }
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(linksA[0], linksA[1], test::ExpectOK{}));
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(linksB[0], linksB[1], test::ExpectOK{}));
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(linksB[1], linksB[2], test::ExpectOK{}));

  _scene->EnableActorContactSymmetric(
      parentA->GetHandle(),
      parentA->GetHandle(),
      /*enable*/ false,
      IncludeNestedActors::Yes,
      test::ExpectOK{});

  for (auto actorA : handlesA) {
    for (auto actorB : handlesA) {
      EXPECT_FALSE(sceneImpl->IsActorContactEnabled(actorA, actorB, test::ExpectOK{}));
    }
  }
}

TEST_F(MochiContactFilter, ActorVsActor_IncludeNestedActorsIncludesNestedSoftActors) {
  auto* sceneImpl = assert_cast<SceneImpl*>(_scene);

  auto* parent = CreateContactFilterSoftSkinnedActor();
  auto const& links = parent->GetNestedLinkActors(test::ExpectOK{});
  auto const& softActors = parent->GetNestedSoftActors(test::ExpectOK{});
  ASSERT_EQ(1, isize(links));
  ASSERT_EQ(1, isize(softActors));

  _scene->SetForceSingleIsland(true);
  island::PreStep(GetRegistry());

  _scene->EnableActorContactSymmetric(
      parent->GetHandle(),
      parent->GetHandle(),
      /*enable*/ false,
      IncludeNestedActors::Yes,
      test::ExpectOK{});

  ActorHandle const handles[] = {parent->GetHandle(), links[0], softActors[0]};
  for (auto actorA : handles) {
    for (auto actorB : handles) {
      EXPECT_FALSE(sceneImpl->IsActorContactEnabled(actorA, actorB, test::ExpectOK{}));
    }
  }
}
