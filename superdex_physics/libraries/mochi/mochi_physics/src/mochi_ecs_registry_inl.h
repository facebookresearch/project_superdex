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

#include "mochi_ecs_registry.h" // Reverse include for intellisense

namespace mochi::ecs {

namespace detail {
MOCHI_API void RegisterComponentImpl(
    entt::registry& reg,
    SReflect::TypeInfo const* reflectionInfo,
    entt::type_info const& enttInfo,
    std::type_info const& stdInfo,
    std::function<entt::registry::base_type&(entt::registry& reg)>&& getStorage,
    std::function<void*(entt::registry::base_type& storage, entt::entity)>&& getComponent,
    std::function<void*(entt::registry::base_type& storage)>&& compactAndGetRawStorage,
    std::function<void*(entt::registry& reg)>&& tryCtx);

} // namespace detail

template <class ComponentT>
void RegisterComponent(entt::registry& reg) {
  if constexpr (entt::ignore_as_empty_v<ComponentT>) {
    // This is a "Tag" component. It has no per-entity storage.
    ecs::detail::RegisterComponentImpl(
        reg,
        SReflect::TryGetTypeInfo<ComponentT>(),
        entt::type_id<ComponentT>(),
        typeid(ComponentT),
        [](entt::registry& reg) -> entt::registry::base_type& { return reg.storage<ComponentT>(); },
        {}, // getComponent
        {}, // compactAndGetRawStorage
        [](entt::registry& reg) -> void* { return reg.try_ctx<ComponentT>(); });
  } else {
    // Capture lambdas that can operate on the strongly typed storage pool
    ecs::detail::RegisterComponentImpl(
        reg,
        SReflect::TryGetTypeInfo<ComponentT>(),
        entt::type_id<ComponentT>(),
        typeid(ComponentT),
        [](entt::registry& reg) -> entt::registry::base_type& { return reg.storage<ComponentT>(); },
        [](entt::registry::base_type& storage, entt::entity e) -> void* {
          using StorageT = typename entt::storage_traits<entt::entity, ComponentT>::storage_type;
          return &assert_cast<StorageT&>(storage).get(e);
        },
        [](entt::registry::base_type& storage) -> void* {
          using StorageT = typename entt::storage_traits<entt::entity, ComponentT>::storage_type;
          auto& storageT = assert_cast<StorageT&>(storage);
          storageT.compact();
          return storageT.raw();
        },
        [](entt::registry& reg) -> void* { return reg.try_ctx<ComponentT>(); });
  }
}

inline bool ComponentTypeInfo::IsTag() const {
  return !_getComponent; // Not a valid function for tag components
}

inline Span<entt::entity const> ComponentTypeInfo::GetEntities(entt::registry const& reg) const {
  // This requires a const_cast (to avoid needing two version of getStorage), but it does not
  // leak any mutable data to the caller.
  auto const& storage = _getStorage(const_cast<entt::registry&>(reg));
  return Span<entt::entity const>{storage.data(), storage.size()};
}

inline bool ComponentTypeInfo::ContainsEntity(entt::registry const& reg, entt::entity e) const {
  return _getStorage(const_cast<entt::registry&>(reg)).contains(e);
}

inline void* ComponentTypeInfo::TryGet(entt::registry& reg, entt::entity e) const {
  if (_getComponent) {
    auto& storage = _getStorage(reg);
    return storage.contains(e) ? _getComponent(storage, e) : nullptr;
  } else {
    return nullptr; // Tags have no per-entity storage
  }
}

inline void const* ComponentTypeInfo::TryGet(entt::registry const& reg, entt::entity e) const {
  return TryGet(const_cast<entt::registry&>(reg), e); // const-in-const-out is OK
}

inline void* ComponentTypeInfo::TryCtx(entt::registry& reg) const {
  return _tryCtx(reg);
}

inline void const* ComponentTypeInfo::TryCtx(entt::registry const& reg) const {
  return _tryCtx(const_cast<entt::registry&>(reg)); // const-in-const-out is OK
}

inline void* const* ComponentTypeInfo::TryGetPageTable(entt::registry& reg) const {
  return _compactAndGetRawStorage
      ? static_cast<void* const*>(_compactAndGetRawStorage(_getStorage(reg)))
      : nullptr;
}

inline SReflect::TypeInfo const* ComponentTypeInfo::TryGetReflectionInfo() const {
  return _reflectionInfo;
}

inline std::string const& ComponentTypeInfo::GetTypeName() const {
  return _typeName;
}

inline entt::id_type ComponentTypeInfo::GetEnttHash() const {
  return _enttHash;
}

template <class AttributeT>
inline void EnumerateComponentsWithAttribute(
    entt::registry const& reg,
    std::function<void(ComponentTypeInfo const& info)> const& onEach) {
  EnumerateComponentsWithAttribute(reg, SReflect::GetTypeId<AttributeT>(), onEach);
}

} // namespace mochi::ecs
