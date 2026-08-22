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

#include "mochi_differentiable_test_utils.h"

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <gtest/gtest.h>

#include <string>

namespace mochi::test {

Actor* FindActorByName(Scene* scene, std::string_view actorName) {
  Actor* foundActor = nullptr;
  scene->ForEachActor([&](Actor* actor) {
    if (actor->GetName() == actorName) {
      foundActor = actor;
    }
  });
  EXPECT_NE(foundActor, nullptr);
  return foundActor;
}

void LoadScenePrefab(Scene* scene, std::string_view prefabName) {
  // Do not apply scene settings; differentiability tests configure solver settings explicitly.
  prefab::PrefabParams params{.applySceneSettings = false};
  prefab::AddToScene(
      GetAssetPath(std::string("differentiability_test/") + std::string(prefabName)),
      GetAssetPath(""),
      scene,
      params,
      ExpectOK{});
}

Actor* LoadScenePrefab(Scene* scene, std::string_view prefabName, std::string_view actorName) {
  LoadScenePrefab(scene, prefabName);
  return FindActorByName(scene, actorName);
}

} // namespace mochi::test
