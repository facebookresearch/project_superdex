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

#include <mochi_physics/src/mochi_ecs_utils.h>

#include <mochi_physics/src/mochi_common_components.h>
#include <mochi_physics/src/mochi_ecs.h>

#include <mochi_core/test/mochi_test_helpers.h>

#include <gtest/gtest.h>

#include <cstdint>

using namespace mochi;
using namespace mochi::ecs;

namespace {

TEST(HandleUtils, GetActorHandle) {
  // Test basic encoding with valid scene handle and entity.
  SceneHandle scene{42};
  auto const entity = static_cast<entt::entity>(123);

  ActorHandle handle = GetActorHandle(entity, scene);

  EXPECT_TRUE(handle.IsValid());

  auto const lower = static_cast<uint32_t>(handle.value);
  uint64_t const upper = handle.value >> 32;
  EXPECT_EQ(lower, static_cast<uint32_t>(entity)); // Lower 32 bits contain entity.
  EXPECT_EQ(upper, scene.value); // Upper 32 bits contain scene handle.
}

TEST(HandleUtils, GetConstraintHandle) {
  // Test basic encoding with valid scene handle and entity.
  SceneHandle scene{99};
  auto const entity = static_cast<entt::entity>(456);

  ConstraintHandle handle = GetConstraintHandle(entity, scene);

  EXPECT_TRUE(handle.IsValid());

  auto const lower = static_cast<uint32_t>(handle.value);
  uint64_t const upper = handle.value >> 32;
  EXPECT_EQ(lower, static_cast<uint32_t>(entity)); // Lower 32 bits contain entity.
  EXPECT_EQ(upper, scene.value); // Upper 32 bits contain scene handle.
}

TEST(HandleUtils, ExtractEntityFromActorHandle) {
  SceneHandle scene{7};
  auto const originalEntity = static_cast<entt::entity>(789);
  ActorHandle handle = GetActorHandle(originalEntity, scene);
  EXPECT_EQ(originalEntity, ExtractEntity(handle));
}

TEST(HandleUtils, ExtractEntityFromConstraintHandle) {
  SceneHandle scene{13};
  auto const originalEntity = static_cast<entt::entity>(321);
  ConstraintHandle handle = GetConstraintHandle(originalEntity, scene);
  EXPECT_EQ(originalEntity, ExtractEntity(handle));
}

TEST(HandleUtils, InvalidInputs) {
  // Test that invalid inputs produce invalid handles.
  auto const validEntity = static_cast<entt::entity>(100);
  SceneHandle validScene{5};
  entt::entity invalidEntity = entt::null;
  SceneHandle invalidScene{0};

  ActorHandle actorHandle = GetActorHandle(validEntity, validScene);
  EXPECT_TRUE(actorHandle.IsValid());

  ConstraintHandle constraintHandle = GetConstraintHandle(validEntity, validScene);
  EXPECT_TRUE(constraintHandle.IsValid());

  actorHandle = GetActorHandle(validEntity, invalidScene);
  EXPECT_FALSE(actorHandle.IsValid());

  constraintHandle = GetConstraintHandle(validEntity, invalidScene);
  EXPECT_FALSE(constraintHandle.IsValid());

  actorHandle = GetActorHandle(invalidEntity, validScene);
  EXPECT_FALSE(actorHandle.IsValid());

  constraintHandle = GetConstraintHandle(invalidEntity, validScene);
  EXPECT_FALSE(constraintHandle.IsValid());
}

TEST(HandleUtils, DifferentScenesProduceDifferentActorHandles) {
  // Verify that same entity in different scenes has different actor handles.
  SceneHandle scene1{10};
  SceneHandle scene2{20};
  auto const entity = static_cast<entt::entity>(500);

  ActorHandle handle1 = GetActorHandle(entity, scene1);
  ActorHandle handle2 = GetActorHandle(entity, scene2);

  EXPECT_NE(handle1, handle2);
  EXPECT_TRUE(handle1.IsValid());
  EXPECT_TRUE(handle2.IsValid());

  // Both should extract to the same entity.
  EXPECT_EQ(ExtractEntity(handle1), entity);
  EXPECT_EQ(ExtractEntity(handle2), entity);
}

TEST(HandleUtils, DifferentScenesProduceDifferentConstraintHandles) {
  // Verify that same entity in different scenes has different constraint handles.
  SceneHandle scene1{10};
  SceneHandle scene2{20};
  auto const entity = static_cast<entt::entity>(500);

  ConstraintHandle handle1 = GetConstraintHandle(entity, scene1);
  ConstraintHandle handle2 = GetConstraintHandle(entity, scene2);

  EXPECT_NE(handle1, handle2);
  EXPECT_TRUE(handle1.IsValid());
  EXPECT_TRUE(handle2.IsValid());

  // Both should extract to the same entity.
  EXPECT_EQ(ExtractEntity(handle1), entity);
  EXPECT_EQ(ExtractEntity(handle2), entity);
}

TEST(HandleUtils, ActorHandleBelongsToScene) {
  SceneHandle scene1{10};
  SceneHandle scene2{20};
  auto const entity = static_cast<entt::entity>(500);
  ActorHandle handle = GetActorHandle(entity, scene1);

  EXPECT_TRUE(ActorHandleBelongsToScene(handle, scene1));
  EXPECT_FALSE(ActorHandleBelongsToScene(handle, scene2));
  EXPECT_FALSE(ActorHandleBelongsToScene(ActorHandle{}, scene1));
}

TEST(HandleUtils, ConstraintHandleBelongsToScene) {
  SceneHandle scene1{10};
  SceneHandle scene2{20};
  auto const entity = static_cast<entt::entity>(500);
  ConstraintHandle handle = GetConstraintHandle(entity, scene1);

  EXPECT_TRUE(ConstraintHandleBelongsToScene(handle, scene1));
  EXPECT_FALSE(ConstraintHandleBelongsToScene(handle, scene2));
  EXPECT_FALSE(ConstraintHandleBelongsToScene(ConstraintHandle{}, scene1));
}

TEST(HandleUtils, GetActorEntityRejectsDifferentScene) {
  entt::registry reg;
  reg.set<CSceneHandle>(SceneHandle{20});
  ActorHandle handle = GetActorHandle(reg.create(), SceneHandle{10});
  EXPECT_EQ(entt::entity{entt::null}, GetEntity(reg, handle, test::ExpectNotOK{}));
}

TEST(HandleUtils, GetActorEntityRejectsDestroyedEntity) {
  entt::registry reg;
  SceneHandle scene{10};
  reg.set<CSceneHandle>(scene);
  entt::entity entity = reg.create();
  ActorHandle handle = GetActorHandle(entity, scene);
  reg.destroy(entity);
  EXPECT_EQ(entt::entity{entt::null}, GetEntity(reg, handle, test::ExpectNotOK{}));
}

TEST(HandleUtils, GetActorEntityReturnsEarlyForPresetError) {
  entt::registry reg;
  Error error;
  MOCHI_ERROR_SET(error, "Pre-set error.");

  EXPECT_EQ(entt::entity{entt::null}, GetEntity(reg, ActorHandle{}, error));
}

TEST(HandleUtils, GetConstraintEntityRejectsDifferentScene) {
  entt::registry reg;
  reg.set<CSceneHandle>(SceneHandle{20});
  ConstraintHandle handle = GetConstraintHandle(reg.create(), SceneHandle{10});
  EXPECT_EQ(entt::entity{entt::null}, GetEntity(reg, handle, test::ExpectNotOK{}));
}

TEST(HandleUtils, GetConstraintEntityRejectsDestroyedEntity) {
  entt::registry reg;
  SceneHandle scene{10};
  reg.set<CSceneHandle>(scene);
  entt::entity entity = reg.create();
  ConstraintHandle handle = GetConstraintHandle(entity, scene);
  reg.destroy(entity);
  EXPECT_EQ(entt::entity{entt::null}, GetEntity(reg, handle, test::ExpectNotOK{}));
}

TEST(HandleUtils, GetConstraintEntityReturnsEarlyForPresetError) {
  entt::registry reg;
  Error error;
  MOCHI_ERROR_SET(error, "Pre-set error.");

  EXPECT_EQ(entt::entity{entt::null}, GetEntity(reg, ConstraintHandle{}, error));
}

TEST(HandleUtils, HandleBelongsToSceneRejectsOverflowSceneHandle) {
  SceneHandle scene{10};
  SceneHandle overflowScene{uint64_t{1} << 32};
  auto const entity = static_cast<entt::entity>(500);

  EXPECT_TRUE(SceneHandleFitsInEntityHandle(scene));
  EXPECT_FALSE(SceneHandleFitsInEntityHandle(overflowScene));
  EXPECT_FALSE(ActorHandleBelongsToScene(GetActorHandle(entity, scene), overflowScene));
  EXPECT_FALSE(ConstraintHandleBelongsToScene(GetConstraintHandle(entity, scene), overflowScene));
}

TEST(HandleUtils, RoundTripConsistency) {
  // Test that encode -> extract produces original entity.
  SceneHandle scene{888};

  for (uint32_t i = 0; i < 1000; i += 100) {
    auto const entity = static_cast<entt::entity>(i);

    ActorHandle actorHandle = GetActorHandle(entity, scene);
    EXPECT_EQ(ExtractEntity(actorHandle), entity);

    ConstraintHandle constraintHandle = GetConstraintHandle(entity, scene);
    EXPECT_EQ(ExtractEntity(constraintHandle), entity);
  }
}

} // namespace
