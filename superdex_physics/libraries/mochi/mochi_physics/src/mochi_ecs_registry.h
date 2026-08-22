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

#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/span.h>
#include <mochi_physics/utils/mochi_physics_macros.h>

MOCHI_WARNING_PUSH_IGNORE_ALL();
#include <entt/entity/entity.hpp>
#include <entt/entity/registry.hpp>
MOCHI_WARNING_POP();

#include <functional>
#include <string>
#include <typeinfo>

namespace mochi::ecs {

/**
 * @brief Provides type information and data access to an ECS component type in generic code,
 * without the compile-time type.
 */
class ComponentTypeInfo {
 public:
  MOCHI_DECLARE_MOVE_ONLY(ComponentTypeInfo);
  ~ComponentTypeInfo() = default;
  ComponentTypeInfo(
      std::string&& typeName,
      SReflect::TypeInfo const* reflectionInfo,
      entt::id_type const& enttHash,
      std::function<entt::registry::base_type&(entt::registry& reg)>&& getStorage,
      std::function<void*(entt::registry::base_type& storage, entt::entity)>&& getComponent,
      std::function<void*(entt::registry::base_type& storage)>&& compactAndGetRawStorage,
      std::function<void*(entt::registry& reg)>&& tryCtx);

  /** @brief Return true if this is a tag component (no storage) */
  bool IsTag() const;

  /** @brief Return the span of entities that have this component. */
  Span<entt::entity const> GetEntities(entt::registry const& reg) const;

  /** @brief Return true if the given entity has this component. */
  bool ContainsEntity(entt::registry const& reg, entt::entity e) const;

  /**
   * @brief If the entity has this component, then return the address. Else, return nullptr.
   * @note Similar to: reg.try_get<ComponentT>(e)
   * @warning Void pointers are dangerous! Consider using TryGetReflectionInfo to access it.
   * @warning Do not store the pointer for later. It may be invalidated by other operations.
   * @warning Always returns nullptr for tag components, which have no per-entity storage.
   */
  void* TryGet(entt::registry& reg, entt::entity e) const;
  void const* TryGet(entt::registry const& reg, entt::entity e) const; // Const overload

  /**
   * @brief If the component has been set as global context, then return the component address.
   * Else, return nullptr.
   * @note Similar to reg.try_ctx<ComponentT>()
   * @note Works for tag components, unlike TryGet.
   * @warning Void pointers are dangerous! Consider using TryGetReflectionInfo to access it.
   * @warning Do not store the pointer for later. It may be invalidated by other operations.
   */
  void* TryCtx(entt::registry& reg) const;
  void const* TryCtx(entt::registry const& reg) const; // Const overload

  /**
   * @brief [ADVANCED] Ensure that components are stored contiguously within each page, then return
   * the address of the start of the page table. Return nullptr if this is a tag component.
   *
   * @note EnTT components are stored in pages of up to ENTT_PACKED_PAGE element each.
   * @note Component storage is 1:1 with GetEntities().
   * @note For components of type T, the component with index 'i' is stored at:
   * auto const* pageTable = static_cast<T*const*>(TryGetDensePageTable(reg));
   * auto const& component = pageTable[i % ENTT_PACKED_PAGE][i];
   */
  void* const* TryGetPageTable(entt::registry& reg) const;

  /**
   * @brief Get the component's reflection type information, or nullptr if the component doesn't
   * support reflection (requires macros like MOCHI_STRUCT_BEGIN/END).
   */
  SReflect::TypeInfo const* TryGetReflectionInfo() const;

  /**
   * @brief Get the name of the component class or struct, including its namespace.
   * @brief Information comes from SReflect::TypeTraits if available. Falls back on RTTI name.
   */
  std::string const& GetTypeName() const;

  /**
   * @brief Get EnTT's type identifier for this component
   */
  entt::id_type GetEnttHash() const;

 private:
  std::string _typeName;
  SReflect::TypeInfo const* _reflectionInfo = nullptr;
  entt::id_type _enttHash;
  std::function<entt::registry::base_type&(entt::registry& reg)> _getStorage;
  std::function<void*(entt::registry::base_type& storage, entt::entity)> _getComponent;
  std::function<void*(entt::registry::base_type& storage)> _compactAndGetRawStorage;
  std::function<void*(entt::registry& reg)> _tryCtx;
};

/** @brief Call this once before any components are registered. */
MOCHI_API void InitializeComponentRegistryOnce(entt::registry& reg);

/**
 * @brief Call RegisterComponent<YourComponentClass> once on startup for each component type.
 * @note This is typically done in a function called "InitializeOnce".
 * @warning All component types must be registered before stepping a scene. Failure to do so will
 * result in UNDEFINED BEHAVIOR and may trigger ASAN failures in CI.
 * @tparam ComponentT - Your ECS component type.
 */
template <class ComponentT>
void RegisterComponent(entt::registry& reg);

/**
 * @brief Call this once after all component types have been registered.
 * @warning If you attempt to use any new component after this point, it will be considered a
 * runtime error.
 */
MOCHI_API void FinalizeComponentRegistration(entt::registry& reg);

/**
 * @brief Call this periodically after FinalizeComponentRegistration to ensure that every
 * component type was correctly registered before it was used.
 * @see RegisterComponent, FinalizeComponentRegistration
 */
MOCHI_API void DetectUnregisteredComponents(entt::registry const& reg);

/**
 * @brief Return information about every component type that has been registered.
 */
MOCHI_API Span<ComponentTypeInfo const> GetAllComponentTypes(entt::registry const& reg);

/**
 * @brief Enumerate all component types emplaced on a specific entity.
 * @note Types must first be registered via RegisterComponent.
 * @note Not fast.
 */
MOCHI_API void EnumerateComponentTypesForEntity(
    entt::registry const& reg,
    entt::entity e,
    std::function<void(ComponentTypeInfo const& info)> const& onEach);

/**
 * @brief Enumerate all component types have have been set as global context via
 * entt::registry::set<T>().
 * @note Types must first be registered via RegisterComponent
 * @note Not fast.
 */
MOCHI_API void EnumerateGlobalCtxComponentTypes(
    entt::registry const& reg,
    std::function<void(ComponentTypeInfo const& info)> const& onEach);

/**
 * @brief Enumerate all component types with a specific reflection attribute.
 * @note Types must first be registered via RegisterComponent
 */
template <class AttributeT>
void EnumerateComponentsWithAttribute(
    entt::registry const& reg,
    std::function<void(ComponentTypeInfo const& info)> const& onEach);

/**
 * @brief Enumerate all component types with a specific reflection attribute (non-template version).
 * @note Types must first be registered via RegisterComponent
 */
MOCHI_API void EnumerateComponentsWithAttribute(
    entt::registry const& reg,
    SReflect::TypeId attribute,
    std::function<void(ComponentTypeInfo const& info)> const& onEach);

/** @brief Look up type information by reflection type ID. Return nullptr if not found. */
MOCHI_API ComponentTypeInfo const* TryGetComponentTypeInfo(
    entt::registry const& reg,
    SReflect::TypeId typeId);

/** @brief Look up type information by entt type ID. Return nullptr if not found. */
MOCHI_API ComponentTypeInfo const* TryGetComponentTypeInfo(
    entt::registry const& reg,
    entt::id_type typeId);

} // namespace mochi::ecs

#include "mochi_ecs_registry_inl.h"
