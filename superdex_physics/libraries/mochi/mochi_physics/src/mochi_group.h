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

#pragma once

#include "mochi_ecs.h"

#include <vector>

namespace mochi {

/**************************************************************************
  ECS Components
*/

// Stores the list of actors and constraints that belong to a group (e.g compound).
struct CGroupMembers : NoCopy {
  std::vector<entt::entity> actors;
  std::vector<entt::entity> constraints;
};

// Each entity member of a group (listed by CGroupMembers) should have CGroupMemberInfo.
struct CGroupMemberInfo : NoCopy {
  explicit CGroupMemberInfo(entt::entity g) : group(g) {}
  entt::entity group = {};
};

/**************************************************************************
  Utilities
*/

// Enumerate the actors within a group, including those nested within sub-groups.
// OnEachFunc is any callable object with signature: void(entt::entity)
template <typename RegistryT, typename OnEachFunc>
void ForEachDescendant(
    RegistryT const& reg,
    CGroupMembers const& children,
    OnEachFunc const& onEach) {
  for (entt::entity child : children.actors) {
    onEach(child);
    if (auto const* grandchildren = reg.template try_get<CGroupMembers const>(child)) {
      ForEachDescendant(reg, *grandchildren, onEach);
    }
  }
}

// Call OnEachFunc for the given "parent" entity. If it is a group, then also
// call ForEachDescendant (see above).
template <typename RegistryT, typename OnEachFunc>
void ForEntityAndEachDescendant(
    RegistryT const& reg,
    entt::entity& parent,
    OnEachFunc const& onEach) {
  onEach(parent);
  if (auto const* children = reg.template try_get<CGroupMembers>(parent)) {
    ForEachDescendant(reg, *children, onEach);
  }
}

namespace group {
void InitializeOnce(entt::registry& reg);
}

} // namespace mochi
