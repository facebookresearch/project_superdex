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

#include "mochi_scene_profiler.h"

#include "mochi_ecs_registry.h"
#include "mochi_island.h"

namespace mochi::profiler {

void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<TagProfilerEnabled>(reg);
  ecs::RegisterComponent<CIslandStats>(reg);
}

void Enable(entt::registry& reg) {
  if (IsEnabled(reg)) {
    return;
  }
  reg.set<TagProfilerEnabled>();
  for (auto island : reg.view<TagIsland const>()) {
    reg.emplace<CIslandStats>(island);
  }
}

void Disable(entt::registry& reg) {
  reg.unset<TagProfilerEnabled>();
  reg.clear<CIslandStats>();
}

bool IsEnabled(entt::registry const& reg) {
  return reg.try_ctx<TagProfilerEnabled const>() != nullptr;
}

} // namespace mochi::profiler
