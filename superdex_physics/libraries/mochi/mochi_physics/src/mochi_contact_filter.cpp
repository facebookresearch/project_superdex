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

#include "mochi_contact_filter.h"

#include <string>
#include <string_view>
#include <utility>

using namespace mochi;

/**************************************************************************************************
  CContactFilterTable
*/

void CContactFilterTable::EnableLayerContact(
    ContactLayerId collidingLayer,
    ContactLayerId colliderLayer,
    bool enable) {
  MOCHI_ASSERT(collidingLayer != ContactLayerId::None);
  MOCHI_ASSERT(colliderLayer != ContactLayerId::None);
  auto key = std::make_pair(collidingLayer, colliderLayer);
  if (enable) {
    layersWithNoContact.erase(key); // Noop if not present
  } else {
    layersWithNoContact.insert(key);
  }
}

void CContactFilterTable::EnableEntityContact(
    entt::entity collidingEntity,
    entt::entity colliderEntity,
    bool enable) {
  MOCHI_ASSERT(collidingEntity != entt::null);
  MOCHI_ASSERT(colliderEntity != entt::null);
  auto key = std::make_pair(collidingEntity, colliderEntity);
  if (enable) {
    entitiesWithNoContact.erase(key); // Noop if not present
  } else {
    entitiesWithNoContact.insert(key);
  }
}

void CContactFilterTable::RemoveEntity(entt::entity e) {
  for (auto it = entitiesWithNoContact.begin(); it != entitiesWithNoContact.end();) {
    if (it->first == e || it->second == e) {
      it = entitiesWithNoContact.erase(it);
    } else {
      ++it;
    }
  }
}

/**************************************************************************************************
  Utilities
*/

ContactLayerId mochi::GetContactLayerId(
    CContactFilterTable const& table,
    std::string_view layerName) {
  auto it = table.layerNameToId.find(std::string(layerName));
  return (it == table.layerNameToId.end()) ? ContactLayerId::None : it->second;
}

ContactLayerId mochi::GetOrAddContactLayerId(
    CContactFilterTable& table,
    std::string_view layerName) {
  if (layerName.empty()) {
    return ContactLayerId::None;
  } else {
    auto [it, wasInserted] =
        table.layerNameToId.insert(std::make_pair(layerName, ContactLayerId{}));
    if (wasInserted) {
      // Assign ids sequentially starting with 1
      it->second = static_cast<ContactLayerId>(table.layerNameToId.size());
    }
    return it->second;
  }
}

CContactLayer&
mochi::EmplaceContactLayer(entt::registry& reg, entt::entity e, std::string_view layerName) {
  auto& table = reg.ctx<CContactFilterTable>();
  auto& layer = reg.emplace_or_replace<CContactLayer>(e);
  layer.id = GetOrAddContactLayerId(table, layerName);
  layer.name = layerName;
  return layer;
}

namespace mochi::contact_filter {
void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CContactFilterTable>(reg);
  ecs::RegisterComponent<CContactLayer>(reg);
}
} // namespace mochi::contact_filter
