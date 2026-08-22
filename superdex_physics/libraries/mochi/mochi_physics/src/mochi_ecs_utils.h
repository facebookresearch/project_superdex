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

#include <mochi_physics/mochi_physics.h>

#include <cstdint>
#include <limits>

namespace mochi {

/**************************************************************************
  Identifier Conversion
*/

// Actor and constraint handles reserve the high 32 bits for the scene handle.
// Scenes with handles outside that range cannot be represented by this encoding.
inline constexpr int kEntityHandleSceneBits = 32;
inline constexpr uint64_t kEntityHandleLocalEntityMask =
    (uint64_t{1} << kEntityHandleSceneBits) - 1;
static_assert(
    sizeof(entt::entity) == sizeof(uint32_t),
    "A change to entt::entity size requires updating actor/constraint handle encoding.");

[[nodiscard]] inline bool SceneHandleFitsInEntityHandle(SceneHandle sceneHandle) {
  return sceneHandle.value <= std::numeric_limits<uint32_t>::max();
}

[[nodiscard]] inline uint64_t PackEntityHandleValue(entt::entity e, SceneHandle sceneHandle) {
  auto const eValue = static_cast<uint32_t>(e);
  MOCHI_ASSERT_VERBOSE(
      SceneHandleFitsInEntityHandle(sceneHandle),
      "SceneHandle values overflow actor/constraint handle encoding.");
  return (static_cast<uint64_t>(sceneHandle.value) << kEntityHandleSceneBits) |
      static_cast<uint64_t>(eValue);
}

[[nodiscard]] SceneHandle GetSceneHandle(entt::registry const& reg);

[[nodiscard]] inline ActorHandle GetActorHandle(entt::entity e, SceneHandle sceneHandle) {
  if (e == entt::null || !sceneHandle.IsValid()) {
    return ActorHandle{};
  }

  return ActorHandle{PackEntityHandleValue(e, sceneHandle)};
}

[[nodiscard]] inline ConstraintHandle GetConstraintHandle(entt::entity e, SceneHandle sceneHandle) {
  if (e == entt::null || !sceneHandle.IsValid()) {
    return ConstraintHandle{};
  }

  return ConstraintHandle{PackEntityHandleValue(e, sceneHandle)};
}

// Extract the entt::entity from an ActorHandle.
[[nodiscard]] inline entt::entity ExtractEntity(ActorHandle handle) {
  return static_cast<entt::entity>(handle.value & kEntityHandleLocalEntityMask);
}

// Extract the entt::entity from a ConstraintHandle.
[[nodiscard]] inline entt::entity ExtractEntity(ConstraintHandle handle) {
  return static_cast<entt::entity>(handle.value & kEntityHandleLocalEntityMask);
}

// Get the entt::entity corresponding to an ActorHandle.
// Does NOT guarantee the entity is still valid. See also GetEntity().
[[nodiscard]] inline entt::entity GetEntityUnchecked(ActorHandle handle) {
  return handle.IsValid() ? ExtractEntity(handle) : entt::null;
}

template <typename HandleT>
[[nodiscard]] bool EntityHandleBelongsToScene(HandleT const& handle, SceneHandle sceneHandle) {
  return handle.IsValid() && sceneHandle.IsValid() && SceneHandleFitsInEntityHandle(sceneHandle) &&
      (handle.value >> kEntityHandleSceneBits) == sceneHandle.value;
}

[[nodiscard]] inline bool ActorHandleBelongsToScene(ActorHandle handle, SceneHandle sceneHandle) {
  return EntityHandleBelongsToScene(handle, sceneHandle);
}

// Get the entt::entity corresponding to an ActorHandle.
// Sets an error if the handle belongs to a different scene or the entity is no longer valid.
[[nodiscard]] entt::entity GetEntity(entt::registry const& reg, ActorHandle handle, Error& error);

// Get the entt::entity corresponding to an ConstraintHandle.
// Does NOT guarantee the entity is still valid. See also GetEntity().
[[nodiscard]] inline entt::entity GetEntityUnchecked(ConstraintHandle handle) {
  return handle.IsValid() ? ExtractEntity(handle) : entt::null;
}

[[nodiscard]] inline bool ConstraintHandleBelongsToScene(
    ConstraintHandle handle,
    SceneHandle sceneHandle) {
  return EntityHandleBelongsToScene(handle, sceneHandle);
}

// Get the entt::entity corresponding to a ConstraintHandle.
// Sets an error if the handle belongs to a different scene or the entity is no longer valid.
[[nodiscard]] entt::entity
GetEntity(entt::registry const& reg, ConstraintHandle handle, Error& error);

/**************************************************************************
  Component Utils
*/

// MOCHI_TRY_GET returns a component pointer or return nullptr and sets an error.
#define MOCHI_TRY_GET(ComponentT, reg, e, error)                                  \
  [&]() {                                                                         \
    using ReturnT = decltype((reg).try_get<ComponentT>(e));                       \
    if ((error).IsOK()) {                                                         \
      ReturnT comp = (reg).try_get<ComponentT>(e);                                \
      MOCHI_ERROR_IF(comp == nullptr, error, #ComponentT " component not found"); \
      return comp;                                                                \
    }                                                                             \
    return ReturnT(nullptr);                                                      \
  }()

// If the entity has a component of type ComponentT, then INCREMENT ComponentT::referenceCount.
// Else, add a component of type ComponentT and set ComponentT::referenceCount to 1. Works with
// any default-constructible component type that has an integer member named "referenceCount".
template <class ComponentT>
inline ComponentT* AddOrIncRefComponent(entt::registry& reg, entt::entity e) {
  auto* comp = reg.try_get<ComponentT>(e);
  if (comp) {
    ++comp->referenceCount;
    return comp;
  } else {
    comp = &reg.emplace<ComponentT>(e);
    comp->referenceCount = 1;
    return comp;
  }
}

// If you called AddOrIncRefComponent then call DecRefOrRemoveComponent exactly once when you are
// done with it. This will decrement Component::referenceCount and remove the component when it
// reaches zero.
template <class ComponentT>
inline void RemoveOrDecRefComponent(entt::registry& reg, entt::entity e) {
  auto* comp = reg.try_get<ComponentT>(e);
  MOCHI_ASSERT(
      comp != nullptr,
      "Component not found. Please make sure that DecRefComponent is called exactly once for each call to AddOrIncRefComponent.");
  if (comp) {
    MOCHI_ASSERT(comp->referenceCount > 0, "It should have already been removed.");
    --comp->referenceCount;
    if (comp->referenceCount == 0) {
      reg.erase<ComponentT>(e);
    }
  }
}

// Call AddOrIncRefComponent if (increment == true), else call DecRefComponent.
// Return the component address if a component was added or its reference counter incremented.
template <class ComponentT>
inline ComponentT* AddRemoveOrRefComponent(entt::registry& reg, entt::entity e, bool increment) {
  if (increment) {
    return AddOrIncRefComponent<ComponentT>(reg, e);
  } else {
    RemoveOrDecRefComponent<ComponentT>(reg, e);
    return nullptr;
  }
}

} // namespace mochi
