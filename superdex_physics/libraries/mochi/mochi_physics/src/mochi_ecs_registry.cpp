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

#include "mochi_ecs_registry.h"

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>

#include <unordered_map>

#define MOCHI_USE_GNU_NAME_DEMANGLING                                          \
  ((MOCHI_PLATFORM_ANDROID || MOCHI_PLATFORM_MACOS || MOCHI_PLATFORM_LINUX) && \
   (MOCHI_COMPILER_CLANG || MOCHI_COMPILER_GCC))

#if MOCHI_USE_GNU_NAME_DEMANGLING
#include <cxxabi.h>
#include <cstdlib>
#include <memory>
#endif

using namespace mochi;
using namespace mochi::ecs;

#if MOCHI_USE_GNU_NAME_DEMANGLING
// Helper function to demangle C++ type names on GCC/Clang
static std::string DemangleTypeName(char const* mangledName) {
  int status = 0;
  char* demangledName = abi::__cxa_demangle(mangledName, nullptr, nullptr, &status);
  MOCHI_DEFER(std::free(demangledName));
  if (status == 0 && demangledName) {
    return std::string{demangledName};
  } else {
    // If demangling fails, return the original mangled name
    MOCHI_ASSERT(false, "Failed to demangle type name");
    return std::string{mangledName};
  }
}
#endif

// NOTE: If we had a wrapper for entt::registry, then we could extend its features.
// Instead, this file adds features and data is stored in this global context component.
namespace {
struct CComponentMetadata {
  CComponentMetadata() = default;
  ~CComponentMetadata() = default;
  MOCHI_DECLARE_MOVE_ONLY(CComponentMetadata);

  // Set to true by FinalizeComponentRegistration
  bool isFinal = false;

  // Used to prevent logging spam
  mutable bool hasWarned = false;

  // Number of unregistered types when FinalizeComponentRegistration was called (ideally zero).
  int numComponentTypesWhenFinalized = 0;

  // Populated by calls to RegisterComponent. Locked after FinalizeComponentRegistration.
  DynamicArray<ComponentTypeInfo> allComponents;

  // Easy lookup by entt::id_type to get allComponents array index.
  std::unordered_map<entt::id_type, int> enttTypeToIndex;

  // Easy lookup by SReflect::TypeId to get allComponents array index
  std::unordered_map<SReflect::TypeId, int> reflectionTypeToIndex;

  // Lookup a reflection attribute type to get quick access to all components of that type.
  std::unordered_map<SReflect::TypeId, DynamicArray<int>> componentsByAttribute;
};
} // namespace

ecs::ComponentTypeInfo::ComponentTypeInfo(
    std::string&& typeName,
    SReflect::TypeInfo const* reflectionInfo,
    entt::id_type const& enttHash,
    std::function<entt::registry::base_type&(entt::registry& reg)>&& getStorage,
    std::function<void*(entt::registry::base_type& storage, entt::entity)>&& getComponent,
    std::function<void*(entt::registry::base_type& storage)>&& compactAndGetRawStorage,
    std::function<void*(entt::registry& reg)>&& tryCtx)
    : _typeName(typeName),
      _reflectionInfo(reflectionInfo),
      _enttHash(enttHash),
      _getStorage(std::move(getStorage)),
      _getComponent(std::move(getComponent)),
      _compactAndGetRawStorage(std::move(compactAndGetRawStorage)),
      _tryCtx(std::move(tryCtx)) {};

void ecs::InitializeComponentRegistryOnce(entt::registry& reg) {
  MOCHI_ASSERT(
      !reg.try_ctx<CComponentMetadata>(), "Redundant call to InitializeComponentRegistryOnce");
  reg.set<CComponentMetadata>();
}

static CComponentMetadata& GetComponentMetadata(entt::registry& reg, bool finalOnly = true) {
  auto* metadata = reg.try_ctx<CComponentMetadata>();
  MOCHI_ASSERT(metadata != nullptr, "Please call InitializeComponentRegistryOnce first.");
  MOCHI_ASSERT(
      metadata->isFinal || !finalOnly, "You must call FinalizeComponentRegistration first.");
  return *metadata;
}

static CComponentMetadata const& GetComponentMetadata(
    entt::registry const& reg,
    bool finalOnly = true) {
  return GetComponentMetadata(
      const_cast<entt::registry&>(reg), finalOnly); // Share non-const implementation
}

void ecs::detail::RegisterComponentImpl(
    entt::registry& reg,
    SReflect::TypeInfo const* reflectionInfo,
    entt::type_info const& enttInfo,
    std::type_info const& stdInfo,
    std::function<entt::registry::base_type&(entt::registry& reg)>&& getStorage,
    std::function<void*(entt::registry::base_type& storage, entt::entity e)>&& getComponent,
    std::function<void*(entt::registry::base_type& storage)>&& compactAndGetRawStorage,
    std::function<void*(entt::registry& reg)>&& tryCtx) {
  MOCHI_ASSERT(!!getStorage, "All components must implement this function.");
  MOCHI_ASSERT(!!tryCtx, "All components must implement this function.");
  MOCHI_ASSERT(
      !reflectionInfo || (reflectionInfo->_constructInPlace && reflectionInfo->_destructInPlace),
      "Please add a default constructor to %s for maximum compatibility with generic reflection-based features.",
      reflectionInfo->_nameWithNamespace);

  // We must call getStorage() now because the entt::registry will create the storage pool the first
  // time the type is accessed. That initialization is not thread-safe, so we force it to happen
  // early.
  getStorage(reg);

  // We store information on a global ctx component
  auto& metadata = GetComponentMetadata(reg, false /*finalOnly*/);
  MOCHI_ASSERT(
      !metadata.isFinal,
      "It is illegal to register components after calling FinalizeComponentRegistration.");

  std::string typeName;
  if (reflectionInfo) {
    // Use the type name from SReflect::TypeInfo if available, because it is higher quality and
    // guaranteed to be the same across all compilers.
    typeName = reflectionInfo->_nameWithNamespace;
  } else {
    // Fall back on the name that RTTI reports (compiler-specific string).
#if MOCHI_USE_GNU_NAME_DEMANGLING
    // std::type_info::name() returns a mangled string on this platform.
    typeName = DemangleTypeName(stdInfo.name());
#elif MOCHI_PLATFORM_WINDOWS
    // On Windows, std::type_info::name() returns a string that has already been demangled,
    // but it may be prefixed with "struct" or "class", which we choose to omit.
    typeName = stdInfo.name();
    if (typeName.starts_with("struct ")) {
      typeName = typeName.substr(7);
    } else if (typeName.starts_with("class ")) {
      typeName = typeName.substr(6);
    }
#else
#error TODO: Initialize typeName with a demangled string for this compiler/platform combination.
#endif
  }

  // Append to allComponents
  metadata.allComponents.emplace_back(
      ComponentTypeInfo{
          std::move(typeName),
          reflectionInfo,
          enttInfo.hash(),
          std::move(getStorage),
          std::move(getComponent),
          std::move(compactAndGetRawStorage),
          std::move(tryCtx)});
}

void ecs::FinalizeComponentRegistration(entt::registry& reg) {
  auto& metadata = GetComponentMetadata(reg, false /*finalOnly*/);
  MOCHI_ASSERT(!metadata.isFinal, "FinalizeComponentRegistration has already been called.");
  metadata.isFinal = true;

  // It is nice to have all component types in a stable order, so sort them by name.
  std::sort(
      metadata.allComponents.begin(),
      metadata.allComponents.end(),
      [](ComponentTypeInfo const& a, ComponentTypeInfo const& b) {
        return a.GetTypeName() < b.GetTypeName();
      });

  // Populate lookup tables.
  for (int i = 0; i < isize(metadata.allComponents); ++i) {
    auto const& info = metadata.allComponents[i];

    {
      auto [it, wasAdded] = metadata.enttTypeToIndex.insert(std::make_pair(info.GetEnttHash(), i));
      MOCHI_ASSERT(
          wasAdded,
          "The ECS component %s has already been registered.",
          std::string(info.GetTypeName()).c_str());
    }

    auto const* reflectionInfo = info.TryGetReflectionInfo();
    if (reflectionInfo) {
      auto [it, wasAdded] =
          metadata.reflectionTypeToIndex.insert(std::make_pair(reflectionInfo->_typeId, i));
      MOCHI_ASSERT(
          wasAdded,
          "The SReflect::TypeId for ECS component %s has already been registered. This should not be "
          "possible unless the entt::id_type has also been registered. See assert above.",
          reflectionInfo->_nameWithNamespace);

      for (auto&& [id, ptr] : reflectionInfo->_attributes) {
        metadata.componentsByAttribute[id].push_back(i);
      }
    }
  }

  // Count all types that the entt::registry knows about, and how many remain unregistered.
  int numTypes = 0, numUnregistered = 0;
  for (auto const&& [id, storage] : reg.storage()) {
    ++numTypes;
    if (metadata.enttTypeToIndex.find(id) == metadata.enttTypeToIndex.end()) {
      if (id == entt::type_id<CComponentMetadata>().hash()) {
        // This one is ours. It is hidden to simplify unit tests. Don't count it.
      } else {
        ++numUnregistered;
      }
    }
  }

  // Remember this number for DetectUnregisteredComponents
  metadata.numComponentTypesWhenFinalized = numTypes;

  // Warn about component types that should have been registered already.
  if (numUnregistered != 0) {
    MOCHI_LOG_WARNING(
        "There are %d ECS components that have been used with the entt::registry but not "
        "explicitly registered via mochi::ecs::RegisterComponent. Unfortunately, we don't "
        "know what they are called. Please find them and register them.",
        numUnregistered);
  }
}

void ecs::DetectUnregisteredComponents([[maybe_unused]] entt::registry const& reg) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
  auto const& metadata = GetComponentMetadata(reg);
  if (!metadata.hasWarned) {
    int numTypes = 0;
    for ([[maybe_unused]] auto unused : reg.storage()) {
      ++numTypes;
    }

    int numTypesAddedAfterFinalize = numTypes - metadata.numComponentTypesWhenFinalized;
    if (numTypesAddedAfterFinalize != 0) {
      MOCHI_LOG_WARNING(
          "%d unregistered ECS component types were used with the entt::registry for the first time after "
          "mochi::ecs::FinalizeComponentRegistration was called! Unfortunately this code cannot determine the "
          "class/struct name(s). If you just introduced a new component type, then please register it by adding "
          "a call to mochi::ecs::RegisterComponent in the appropriate file's InitializeOnce method. If you "
          "can't figure out what component type it was, then the instructions in D84765137 should help.",
          numTypesAddedAfterFinalize);
      metadata.hasWarned = true;
    }
  }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
}

Span<ComponentTypeInfo const> ecs::GetAllComponentTypes(entt::registry const& reg) {
  // CComponentMetadata is the first component in the list.
  auto const& metadata = GetComponentMetadata(reg);
  return MakeSpan(metadata.allComponents);
}

void ecs::EnumerateComponentsWithAttribute(
    entt::registry const& reg,
    SReflect::TypeId attributeId,
    std::function<void(ComponentTypeInfo const& info)> const& onEach) {
  MOCHI_ASSERT_VERBOSE(!!onEach, "Invalid callback");
  auto const& metadata = GetComponentMetadata(reg);
  auto it = metadata.componentsByAttribute.find(attributeId);
  if (it != metadata.componentsByAttribute.end()) {
    for (int i : it->second) {
      onEach(metadata.allComponents[i]);
    }
  }
}

void ecs::EnumerateComponentTypesForEntity(
    entt::registry const& reg,
    entt::entity e,
    std::function<void(ComponentTypeInfo const& info)> const& onEach) {
  MOCHI_ASSERT_VERBOSE(!!onEach, "Invalid callback");
  auto const& metadata = GetComponentMetadata(reg);
  for (auto const& comp : metadata.allComponents) {
    if (comp.ContainsEntity(reg, e)) {
      onEach(comp);
    }
  }
}

void ecs::EnumerateGlobalCtxComponentTypes(
    entt::registry const& reg,
    std::function<void(ComponentTypeInfo const& info)> const& onEach) {
  MOCHI_ASSERT_VERBOSE(!!onEach, "Invalid callback");
  auto const& metadata = GetComponentMetadata(reg);
  for (auto const& comp : metadata.allComponents) {
    if (comp.TryCtx(reg) != nullptr) {
      onEach(comp);
    }
  }
}

ComponentTypeInfo const* ecs::TryGetComponentTypeInfo(
    entt::registry const& reg,
    SReflect::TypeId typeId) {
  auto const& metadata = GetComponentMetadata(reg);
  auto it = metadata.reflectionTypeToIndex.find(typeId);
  return (it == metadata.reflectionTypeToIndex.end()) ? nullptr
                                                      : &metadata.allComponents[it->second];
}

ComponentTypeInfo const* ecs::TryGetComponentTypeInfo(
    entt::registry const& reg,
    entt::id_type typeId) {
  auto const& metadata = GetComponentMetadata(reg);
  auto it = metadata.enttTypeToIndex.find(typeId);
  return (it == metadata.enttTypeToIndex.end()) ? nullptr : &metadata.allComponents[it->second];
}
