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

#include "mochi_contact.h"
#include "mochi_ecs.h"

#include <mochi_core/utils/container_utils.h>

#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mochi {

// Every unique layer string gets a unique ContactLayerId.
// The empty string get ContactLayerId::None.
enum class ContactLayerId : int { None = 0 };

/**************************************************************************************************
  ECS Components
*/

/**
 * @brief Stores the ContactLayerId for an actor
 * @see CContactFilterTable
 *
 */
struct CContactLayer : public NoCopy {
  ContactLayerId id = {};
  std::string name; // kept for debugging
};

/**
 * @brief Stores collision filter settings for layers and actors. Global to the scene.
 *
 * @remarks By default collision is enabled for all pairs of actors. However, this table may disable
 * collision for certain pairs of contact layers, and/or certain pairs of entities.
 */
struct CContactFilterTable : public NoCopy {
  // Every unique layer name and its ContactLayerId
  std::unordered_map<std::string, ContactLayerId> layerNameToId;

  // Lookup table for layer-vs-layer interactions (order matters).
  // If the pair is in the table, then they entities with those layers should NOT contact.
  using LayerPair = std::pair<ContactLayerId, ContactLayerId>;
  using LayerPairHash = PairHash<ContactLayerId, ContactLayerId>;
  std::unordered_set<LayerPair, LayerPairHash> layersWithNoContact;

  // Lookup table for entity-vs-entity interactions (order matters).
  // If the pair is in the table, then those specific entities should not contact each other.
  using EntityPair = std::pair<entt::entity, entt::entity>;
  using EntityPairHash = PairHash<entt::entity, entt::entity>;
  std::unordered_set<EntityPair, EntityPairHash> entitiesWithNoContact;

  /**
   * @brief Return true if contact is enabled (not disabled) for collidingLayer-vs-colliderLayer.
   * Order matters.
   * @remark IsLayerContactEnabled and IsEntityContactEnabled must both be true for contact to
   * occur.
   */
  bool IsLayerContactEnabled(ContactLayerId collidingLayer, ContactLayerId colliderLayer) const {
    auto it = layersWithNoContact.find(std::make_pair(collidingLayer, colliderLayer));
    return it == layersWithNoContact.end();
  }

  /**
   * @brief Enable or disable contact for collidingLayer-vs-colliderLayer. Order matters.
   */
  void EnableLayerContact(ContactLayerId collidingLayer, ContactLayerId colliderLayer, bool enable);

  /**
   * @brief Return true if contact is enabled (not disabled) for collidingEntity-vs-colliderEntity.
   * Order matters.
   * @remark IsLayerContactEnabled and IsEntityContactEnabled must both be true for contact to
   * occur.
   */
  bool IsEntityContactEnabled(entt::entity collidingEntity, entt::entity colliderEntity) const {
    auto it = entitiesWithNoContact.find(std::make_pair(collidingEntity, colliderEntity));
    return it == entitiesWithNoContact.end();
  }

  /**
   * @brief Enable or disable contact for collidingEntity-vs-colliderEntity. Order matters.
   */
  void EnableEntityContact(entt::entity collidingEntity, entt::entity colliderEntity, bool enable);

  /**
   * @brief Return true if contact is enabled, according to all of the above heuristics.
   */
  bool IsContactEnabled(
      entt::entity collidingEntity,
      entt::entity colliderEntity,
      ContactLayerId collidingLayer,
      ContactLayerId colliderLayer) const {
    return IsLayerContactEnabled(collidingLayer, colliderLayer) &&
        IsEntityContactEnabled(collidingEntity, colliderEntity);
  }

  /**
   * @brief Remove all references to the specified entity from the table. Called when the entity is
   * destroyed.
   */
  void RemoveEntity(entt::entity e);
};

/**************************************************************************************************
  Utilities
*/

/**
 * @brief Lookup a ContactLayerId by name. Return ContactLayerId::None if layerName is empty string
 * or not found.
 */
MOCHI_API ContactLayerId
GetContactLayerId(CContactFilterTable const& table, std::string_view layerName);

/**
 * @brief Lookup a ContactLayerId by name. Add it the first time (if not empty string).
 */
MOCHI_API ContactLayerId
GetOrAddContactLayerId(CContactFilterTable& table, std::string_view layerName);

/**
 * @brief Emplace CContactLayer on the specified entity, and initialize it with
 * GetOrAddContactLayerId.
 */
CContactLayer& EmplaceContactLayer(entt::registry& reg, entt::entity e, std::string_view layerName);

namespace contact_filter {
void InitializeOnce(entt::registry& reg);
}

} // namespace mochi
