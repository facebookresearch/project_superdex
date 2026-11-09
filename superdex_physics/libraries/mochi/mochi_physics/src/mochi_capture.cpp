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

#include "mochi_capture.h"
#include "mochi_attributes.h"
#include "mochi_common_components.h"
#include "mochi_constraint.h"

#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/stream.h>

#include <picojson/picojson.h>

#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <string_view>
#include <utility>

using namespace mochi;
using namespace mochi::capture;

namespace {

// Serialized at the beginning of each block of binary component data.
struct ComponentHeader {
  static constexpr size_t kMaxDataSize = UINT32_MAX;
  [[maybe_unused]] static constexpr size_t kMaxCount = UINT32_MAX;
  bool operator==(ComponentHeader const& rhs) const = default;

  SReflect::TypeId id{};
  uint64_t entityHash = 0;
  uint32_t dataSize = 0;
  uint32_t count = 0;
};
static_assert(sizeof(ComponentHeader) == 24, "Unexpected size");

// Serialized at the end of the binary capture data.
struct CaptureFooter {
  std::array<char, 4> label{'m', 'e', 'n', 'd'}; // "Mochi capture END"
};
static_assert(sizeof(CaptureFooter) == 4, "Unexpected size");

// Serialized at the beginning of the binary capture data.
struct CaptureHeader {
  [[maybe_unused]] static constexpr size_t kMaxNumCtxComponentTypes = UINT16_MAX;
  [[maybe_unused]] static constexpr size_t kMaxNumEntityComponentTypes = UINT16_MAX;
  [[maybe_unused]] static constexpr size_t kMaxDataSize = UINT64_MAX;
  bool operator==(CaptureHeader const& rhs) const = default;

  std::array<char, 4> label{'m', 'c', 'a', 'p'}; // "Mochi CAPture"
  uint16_t numCtxComponentTypes = 0;
  uint16_t numEntityComponentTypes = 0;
  uint32_t unused = 0; // Just letting you see where the padding is. Feel free to replace.
  uint64_t dataSize = 0; // Number of bytes after the header
};
static_assert(sizeof(CaptureHeader) == 24, "Unexpected size");

// Global context component, which stores registered callbacks.
struct CCaptureCallbacks {
  DynamicArray<std::function<void(entt::registry&)>> postRestore;
};

} // namespace

// Serialize a global context component if it is currently in use.
static void SerializeCtxComponentData(
    entt::registry& reg,
    ecs::ComponentTypeInfo const& componentType,
    DynamicArrayStreamWriter& outStream,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  void const* ctxData = componentType.TryCtx(reg);
  if (ctxData == nullptr) {
    // This component type is not currently set as global context
    return;
  }

  auto const* typeInfo = componentType.TryGetReflectionInfo();
  MOCHI_ASSERT_VERBOSE(typeInfo != nullptr);
  MOCHI_ASSERT_VERBOSE(!componentType.IsTag(), "Tag serialization is not currently supported");

  // Write a placeholder for the component type header
  size_t const headerPos = outStream.GetPosition();
  ComponentHeader header;
  StreamWrite(header, outStream, error);

  // Write component data
  size_t dataSizeBytes = 0;
  if (typeInfo->IsMemCopySafe()) {
    dataSizeBytes = typeInfo->_sizeInBytes;
    outStream.Write(ctxData, dataSizeBytes, error);
  } else {
    size_t const dataStartPos = outStream.GetPosition();
    bool success = typeInfo->SerializeToBytesInner(ctxData, outStream);
    MOCHI_ERROR_IF(!success, error, "Failed to serialize component data");
    dataSizeBytes = outStream.GetPosition() - dataStartPos;
  }

  MOCHI_ERROR_IF(
      dataSizeBytes > ComponentHeader::kMaxDataSize,
      error,
      "Data size too large for the current binary format");

  // Go back and fill in the component block header
  header.id = typeInfo->_typeId;
  header.dataSize = static_cast<decltype(header.dataSize)>(dataSizeBytes);
  StreamWriteAt(headerPos, header, outStream, error);
}

static ecs::ComponentTypeInfo const*
LookUpComponentTypeInfo(entt::registry const& reg, ComponentHeader const& header, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  auto const* componentType = ecs::TryGetComponentTypeInfo(reg, header.id);
  MOCHI_ERROR_IF(componentType == nullptr, error, "Unknown component type");
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ASSERT_VERBOSE(!componentType->IsTag(), "Tag serialization is not currently supported");
  return componentType;
}

// Return true if the component has any of the specified attributes
static bool ComponentHasAttribute(
    ecs::ComponentTypeInfo const* componentType,
    Span<SReflect::TypeId const> attributes) {
  MOCHI_ASSERT_VERBOSE(componentType != nullptr);
  auto const* typeInfo = componentType->TryGetReflectionInfo();
  MOCHI_ASSERT_VERBOSE(typeInfo != nullptr);
  for (auto id : attributes) {
    if (typeInfo->HasAttribute(id)) {
      return true;
    }
  }
  return false;
}

namespace {

constexpr size_t kFilteredEntityStackCount = 1024;

struct CaptureEntityList {
  Span<entt::entity const> entities;
  bool requiresPerEntityAccess = false;
};

} // namespace

// Return entities in component storage order, optionally filtered by
// attribute::CaptureState::onlyCaptureWith. The returned span may alias filteredEntities.
static CaptureEntityList GetCaptureEntities(
    entt::registry const& reg,
    Span<entt::entity const> allEntities,
    attribute::CaptureState const& attrib,
    DynamicArray<entt::entity>& filteredEntities) {
  if (!attrib.onlyCaptureWith) {
    return {allEntities, false};
  }

  auto const* filterComponentType = ecs::TryGetComponentTypeInfo(reg, attrib.onlyCaptureWith);
  MOCHI_ASSERT_VERBOSE(filterComponentType != nullptr, "Must have been registered");
  auto const filterEntities = filterComponentType->GetEntities(reg);
  if (filterEntities.empty()) {
    return {Span<entt::entity const>{}, true};
  }

  bool requiresPerEntityAccess = false;
  for (size_t i = 0; i < allEntities.size(); ++i) {
    entt::entity const entity = allEntities[i];
    if (filterComponentType->ContainsEntity(reg, entity)) {
      if (requiresPerEntityAccess) {
        filteredEntities.push_back(entity);
      }
    } else if (!requiresPerEntityAccess) {
      requiresPerEntityAccess = true;
      filteredEntities.reserve(Min(allEntities.size(), filterEntities.size()));
      filteredEntities.append(allEntities.begin(), allEntities.begin() + i);
    }
  }
  return requiresPerEntityAccess ? CaptureEntityList{MakeConstSpan(filteredEntities), true}
                                 : CaptureEntityList{allEntities, false};
}

namespace {
struct CtxComponentToRestore {
  SReflect::TypeId id;
  bool pending = true;
};
} // namespace

// Deserialize a global context component.
// Must match logic in SerializeCtxComponentData.
static void DeserializeCtxComponentData(
    entt::registry& reg,
    SpanStreamReader& stream,
    Span<SReflect::TypeId const> excludedAttributes,
    Span<CtxComponentToRestore> components,
    Error& error) {
  // Read the component header
  ComponentHeader header;
  StreamRead(header, stream, error);

  // Look up the component type information
  auto const* componentType = LookUpComponentTypeInfo(reg, header, error);
  MOCHI_ERROR_RETURN(error);
  auto const* typeInfo = componentType->TryGetReflectionInfo();
  MOCHI_ASSERT_VERBOSE(typeInfo != nullptr);

  if (ComponentHasAttribute(componentType, excludedAttributes)) {
    // Skip this component
    stream.Advance(header.dataSize, error);
    return;
  }

  // Check if the component is currently in the registry
  auto* const component = std::ranges::lower_bound(
      components,
      header.id,
      [](SReflect::TypeId a, SReflect::TypeId b) { return a < b; },
      &CtxComponentToRestore::id);
  MOCHI_ERROR_IF(
      component == components.end() || component->id != header.id || !component->pending,
      error,
      "Unable to restore state because the composition of the scene context has changed");
  MOCHI_ERROR_RETURN(error);
  component->pending = false;

  // Deserialize component data
  void* const ctxData = componentType->TryCtx(reg);
  MOCHI_ASSERT_VERBOSE(ctxData != nullptr, "Restored ctx components are present in the registry.");
  if (typeInfo->IsMemCopySafe()) {
    MOCHI_ERROR_IF((size_t)header.dataSize != typeInfo->_sizeInBytes, error, "Data size mismatch");
    stream.Read(ctxData, typeInfo->_sizeInBytes, error);
  } else {
    size_t dataStartPos = stream.GetPosition();
    bool success = typeInfo->DeserializeFromBytes(stream, ctxData);
    MOCHI_ERROR_IF(!success, error, "Failed to deserialize component data");
    auto dataBytesRead = stream.GetPosition() - dataStartPos;
    MOCHI_ERROR_IF(dataBytesRead != header.dataSize, error, "Data size mismatch");
  }
}

// Serialize all data for one type of component, if it is currently in use.
static void SerializeEntityComponentData(
    entt::registry& reg,
    ecs::ComponentTypeInfo const& componentType,
    DynamicArrayStreamWriter& outStream,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto const allEntities = componentType.GetEntities(reg);
  if (allEntities.empty()) {
    // This component type is not currently used by any entity.
    return;
  }

  // Look up type information
  auto const* typeInfo = componentType.TryGetReflectionInfo();
  MOCHI_ASSERT_VERBOSE(typeInfo != nullptr);
  MOCHI_ASSERT_VERBOSE(!componentType.IsTag(), "Tag serialization is not currently supported");

  // Some component types should only be serialized if the entity also has some other component
  // type. For example, TagSoftActor might be specified to indicate that this component type should
  // only be serialized for soft actors.
  auto const* attrib = typeInfo->GetAttribute<attribute::CaptureState>();
  MOCHI_ASSERT_VERBOSE(attrib != nullptr);
  MOCHI_FILO_STACK_ALLOCATOR(entityAllocator, kFilteredEntityStackCount * sizeof(entt::entity));
  DynamicArray<entt::entity> filteredEntities(&entityAllocator);
  auto const captureEntities = GetCaptureEntities(reg, allEntities, *attrib, filteredEntities);
  auto const entities = captureEntities.entities;
  if (entities.empty()) {
    // This component is filtered out by all existing entities.
    return;
  }

  size_t const count = entities.size();
  MOCHI_ASSERT_VERBOSE(
      count <= ComponentHeader::kMaxCount,
      "Too many entities to serialize. This shouldn't be possible unless entt::entity was redefined as a 64-bit type.");

  // Write a placeholder for the component type header
  size_t const headerPos = outStream.GetPosition();
  ComponentHeader header;
  StreamWrite(header, outStream, error);

  // Instead of serializing the array of entities, we only serialize a hash.
  // This will be used to verify that the entities match when deserializing.
  uint64_t const entityHash = SReflect::CalcHash64(entities.data(), count * sizeof(entt::entity));

  // Write per-entity component data
  size_t const dataStartPos = outStream.GetPosition();
  if (captureEntities.requiresPerEntityAccess) {
    // Filtered components are not necessarily contiguous in the component storage.
    bool const isMemCopySafe = typeInfo->IsMemCopySafe();
    for (entt::entity const entity : entities) {
      void const* src = componentType.TryGet(reg, entity);
      MOCHI_ASSERT_VERBOSE(src != nullptr);
      if (isMemCopySafe) {
        outStream.Write(src, typeInfo->_sizeInBytes, error);
      } else {
        bool const success = typeInfo->SerializeToBytesInner(src, outStream);
        MOCHI_ERROR_IF(!success, error, "Failed to serialize component data");
      }
      MOCHI_ERROR_RETURN(error);
    }
  } else {
    void const* const* pages = componentType.TryGetPageTable(reg);
    MOCHI_ASSERT_VERBOSE(
        pages != nullptr,
        "This data should exist for all component types that are not \"tag\" components.");
    if (typeInfo->IsMemCopySafe()) {
      // Fast Path: Copy raw bytes one page at a time.
      size_t iPage = 0;
      for (size_t i = 0; i < count; i += size_t(ENTT_PACKED_PAGE), ++iPage) {
        size_t copySize = Min(size_t(ENTT_PACKED_PAGE), count - i) * typeInfo->_sizeInBytes;
        outStream.Write(pages[iPage], copySize, error);
      }
    } else {
      // Slower Path: Let each component serialize itself.
      size_t iPage = 0;
      for (size_t i = 0; i < count; i += ENTT_PACKED_PAGE, ++iPage) {
        auto const* pageBegin = static_cast<uint8_t const*>(pages[iPage]);
        MOCHI_ASSERT_VERBOSE(pageBegin != nullptr);
        auto const pageSize = Min(size_t(ENTT_PACKED_PAGE), count - i) * typeInfo->_sizeInBytes;
        auto const* pageEnd = pageBegin + pageSize;
        for (auto const* src = pageBegin; src < pageEnd; src += typeInfo->_sizeInBytes) {
          bool success = typeInfo->SerializeToBytesInner(src, outStream);
          MOCHI_ERROR_IF(!success, error, "Failed to serialize component data");
          MOCHI_ERROR_RETURN(error);
        }
      }
    }
  }

  size_t dataSizeBytes = outStream.GetPosition() - dataStartPos;
  MOCHI_ERROR_IF(
      dataSizeBytes > ComponentHeader::kMaxDataSize,
      error,
      "Data size too large for the current binary format");
  MOCHI_ERROR_RETURN(error);

  // Go back and fill in the component block header
  header.id = typeInfo->_typeId;
  header.entityHash = entityHash;
  header.dataSize = static_cast<decltype(header.dataSize)>(dataSizeBytes);
  header.count = static_cast<decltype(header.count)>(count);
  StreamWriteAt(headerPos, header, outStream, error);
}

// Deserialize all data for one type of component.
// Must match SerializeEntityComponentData.
static void DeserializeEntityComponentData(
    entt::registry& reg,
    SpanStreamReader& stream,
    Span<SReflect::TypeId const> excludedAttributes,
    Error& error) {
  // Read the component header
  ComponentHeader header;
  StreamRead(header, stream, error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ASSERT_VERBOSE(header.count > 0, "Empty headers should never be written");

  // Look up the component type information
  auto const* componentType = LookUpComponentTypeInfo(reg, header, error);
  MOCHI_ERROR_RETURN(error);
  auto const* typeInfo = componentType->TryGetReflectionInfo();
  MOCHI_ASSERT_VERBOSE(typeInfo != nullptr);

  if (ComponentHasAttribute(componentType, excludedAttributes)) {
    // Skip this component
    stream.Advance(header.dataSize, error);
    return;
  }

  // Verify the entity count
  auto const* attrib = typeInfo->GetAttribute<attribute::CaptureState>();
  MOCHI_ERROR_IF(
      attrib == nullptr, error, "Component type is not valid in an entity capture block");
  MOCHI_ERROR_RETURN(error);
  MOCHI_FILO_STACK_ALLOCATOR(entityAllocator, kFilteredEntityStackCount * sizeof(entt::entity));
  DynamicArray<entt::entity> filteredEntities(&entityAllocator);
  auto const allEntities = componentType->GetEntities(reg);
  auto const captureEntities = GetCaptureEntities(reg, allEntities, *attrib, filteredEntities);
  auto const entities = captureEntities.entities;
  MOCHI_ERROR_IF(
      header.count != entities.size(),
      error,
      "Unable to restore state because the number of entities has changed.");
  MOCHI_ERROR_RETURN(error);

  uint64_t const entityHash =
      SReflect::CalcHash64(entities.data(), entities.size() * sizeof(entt::entity));
  MOCHI_ERROR_IF(
      header.entityHash != entityHash,
      error,
      "Unable to restore state because the composition of the scene has changed.");
  MOCHI_ERROR_RETURN(error);

  // Deserialize per-entity data
  if (captureEntities.requiresPerEntityAccess) {
    bool const isMemCopySafe = typeInfo->IsMemCopySafe();
    MOCHI_ERROR_IF(
        isMemCopySafe && header.dataSize != header.count * typeInfo->_sizeInBytes,
        error,
        "Data size mismatch");
    MOCHI_ERROR_RETURN(error);
    size_t const dataStartPos = stream.GetPosition();
    for (entt::entity const entity : entities) {
      void* dst = componentType->TryGet(reg, entity);
      MOCHI_ASSERT_VERBOSE(dst != nullptr);
      if (isMemCopySafe) {
        stream.Read(dst, typeInfo->_sizeInBytes, error);
      } else {
        bool const success = typeInfo->DeserializeFromBytes(stream, dst);
        MOCHI_ERROR_IF(!success, error, "Failed to deserialize component data");
      }
      MOCHI_ERROR_RETURN(error);
    }
    size_t const dataBytesRead = stream.GetPosition() - dataStartPos;
    MOCHI_ERROR_IF(dataBytesRead != header.dataSize, error, "Data size mismatch");
  } else {
    void* const* pages = componentType->TryGetPageTable(reg);
    MOCHI_ASSERT_VERBOSE(
        pages != nullptr,
        "This data should exist for all component types that are not \"tag\" components.");
    if (typeInfo->IsMemCopySafe()) {
      // Fast Path: Copy raw bytes one page at a time.
      MOCHI_ERROR_IF(
          header.dataSize != header.count * typeInfo->_sizeInBytes, error, "Data size mismatch");
      size_t iPage = 0;
      for (size_t i = 0; i < header.count; i += ENTT_PACKED_PAGE, ++iPage) {
        size_t copySize =
            Min(size_t(ENTT_PACKED_PAGE), size_t(header.count) - i) * typeInfo->_sizeInBytes;
        stream.Read(pages[iPage], copySize, error);
      }
    } else {
      // Slower Path: Let each component serialize itself.
      size_t dataStartPos = stream.GetPosition();
      size_t iPage = 0;
      for (size_t i = 0; i < header.count; i += ENTT_PACKED_PAGE, ++iPage) {
        auto* pageBegin = static_cast<uint8_t*>(pages[iPage]);
        MOCHI_ASSERT_VERBOSE(pageBegin != nullptr);
        auto const pageSize =
            Min(size_t(ENTT_PACKED_PAGE), size_t(header.count) - i) * typeInfo->_sizeInBytes;
        auto const* pageEnd = pageBegin + pageSize;
        for (auto* dst = pageBegin; dst < pageEnd; dst += typeInfo->_sizeInBytes) {
          bool success = typeInfo->DeserializeFromBytes(stream, dst);
          MOCHI_ERROR_IF(!success, error, "Failed to deserialize component data");
          MOCHI_ERROR_RETURN(error);
        }
      }
      auto dataBytesRead = stream.GetPosition() - dataStartPos;
      MOCHI_ERROR_IF(dataBytesRead != header.dataSize, error, "Data size mismatch");
    }
  }
}

void mochi::capture::CaptureState(
    entt::registry& reg,
    DynamicArray<uint8_t>& outData,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  DynamicArrayStreamWriter outStream(outData);

  // WARNING: If you modify the logic in this function, then please make sure that
  // CaptureStateToJson still matches it.

  // Write a placeholder for the header, which we will fill in later.
  auto headerPos = outData.size();
  CaptureHeader header;
  StreamWrite(header, outStream, error);
  size_t dataStartPos = outStream.GetPosition();

  // Save global context component data by type
  size_t numCtxComponentTypes = 0;
  size_t prevPos = dataStartPos;
  ecs::EnumerateComponentsWithAttribute(
      reg,
      SReflect::GetTypeId<attribute::CaptureStateCtx>(),
      [&](ecs::ComponentTypeInfo const& componentType) {
        SerializeCtxComponentData(reg, componentType, outStream, error);
        auto pos = outStream.GetPosition();
        numCtxComponentTypes += static_cast<int>(pos != prevPos); // Inc if bytes were written
        prevPos = pos;
      });

  // Save entity component data by type
  size_t numEntityComponentTypes = 0;
  ecs::EnumerateComponentsWithAttribute(
      reg,
      SReflect::GetTypeId<attribute::CaptureState>(),
      [&](ecs::ComponentTypeInfo const& componentType) {
        SerializeEntityComponentData(reg, componentType, outStream, error);
        auto pos = outStream.GetPosition();
        numEntityComponentTypes += static_cast<int>(pos != prevPos); // Inc if bytes were written
        prevPos = pos;
      });

  // Serialize the footer (for error detection)
  CaptureFooter footer;
  StreamWrite(footer, outStream, error);
  size_t dataSizeBytes = outStream.GetPosition() - dataStartPos;

  // These should never fail unless there's a problem with the code.
  MOCHI_ASSERT_VERBOSE(
      numCtxComponentTypes <= CaptureHeader::kMaxNumCtxComponentTypes, "Too many component types");
  MOCHI_ASSERT_VERBOSE(
      numEntityComponentTypes <= CaptureHeader::kMaxNumEntityComponentTypes,
      "Too many component types");
  MOCHI_ASSERT_VERBOSE(dataSizeBytes <= CaptureHeader::kMaxDataSize, "Too many bytes");

  // Go back and fill in the header
  header.numCtxComponentTypes =
      static_cast<decltype(header.numCtxComponentTypes)>(numCtxComponentTypes);
  header.numEntityComponentTypes =
      static_cast<decltype(header.numEntityComponentTypes)>(numEntityComponentTypes);
  header.dataSize = static_cast<decltype(header.dataSize)>(dataSizeBytes);
  StreamWriteAt(headerPos, header, outStream, error);
}

static void ValidateCaptureHeader(CaptureHeader const& header, Error& error) {
  MOCHI_ERROR_IF(
      std::string_view(header.label.data(), header.label.size()) != "mcap",
      error,
      "Invalid capture data header");
}

static void ValidateCaptureFooter(CaptureFooter const& footer, Error& error) {
  MOCHI_ERROR_IF(
      std::string_view(footer.label.data(), footer.label.size()) != "mend",
      error,
      "Capture data is malformed. Expected footer tag at the end.");
}

void mochi::capture::RestorePartialState(
    entt::registry& reg,
    Span<uint8_t const> data,
    Span<SReflect::TypeId const> excludedAttributes,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  SpanStreamReader stream(MakeConstSpan(data));

  // Deserialize and check the header
  CaptureHeader header;
  StreamRead(header, stream, error);
  ValidateCaptureHeader(header, error);
  MOCHI_ERROR_RETURN(error);
  size_t dataStartPos = stream.GetPosition();

  // Identify the active non-excluded ctx components.
  MOCHI_FILO_STACK_ALLOCATOR(typeAllocator, 128 * sizeof(CtxComponentToRestore));
  DynamicArray<CtxComponentToRestore> components(&typeAllocator);
  size_t maxComponents = 0;
  ecs::EnumerateComponentsWithAttribute(
      reg, SReflect::GetTypeId<attribute::CaptureStateCtx>(), [&](ecs::ComponentTypeInfo const&) {
        ++maxComponents;
      });
  components.reserve(maxComponents);
  ecs::EnumerateComponentsWithAttribute(
      reg,
      SReflect::GetTypeId<attribute::CaptureStateCtx>(),
      [&](ecs::ComponentTypeInfo const& componentType) {
        if (componentType.TryCtx(reg) != nullptr &&
            !ComponentHasAttribute(&componentType, excludedAttributes)) {
          auto const* typeInfo = componentType.TryGetReflectionInfo();
          MOCHI_ASSERT_VERBOSE(typeInfo != nullptr);
          components.push_back(CtxComponentToRestore{.id = typeInfo->_typeId});
        }
      });
  std::ranges::sort(components, [](auto const& a, auto const& b) { return a.id < b.id; });

  // Deserialize ctx component data by type
  for (size_t i = 0; i < header.numCtxComponentTypes; ++i) {
    DeserializeCtxComponentData(reg, stream, excludedAttributes, components, error);
    MOCHI_ERROR_RETURN(error);
  }
  MOCHI_ERROR_IF(
      std::ranges::any_of(components, &CtxComponentToRestore::pending),
      error,
      "Unable to restore state because the composition of the scene context has changed");
  MOCHI_ERROR_RETURN(error);

  // Deserialize entity component data by type
  for (size_t i = 0; i < (size_t)header.numEntityComponentTypes; ++i) {
    DeserializeEntityComponentData(reg, stream, excludedAttributes, error);
  }

  // Deserialize and check the footer
  CaptureFooter footer;
  StreamRead(footer, stream, error);
  ValidateCaptureFooter(footer, error);

  // Make sure we read the exact number of bytes expected.
  size_t dataSize = stream.GetPosition() - dataStartPos;
  MOCHI_ERROR_IF(
      dataSize != header.dataSize,
      error,
      "Capture data is malformed. Incorrect number of bytes processed.");
  MOCHI_ERROR_RETURN(error);

  // Run any post-restore callbacks
  if (auto const* callbacks = reg.try_ctx<CCaptureCallbacks>()) {
    for (auto const& fn : callbacks->postRestore) {
      fn(reg);
    }
  }
}

void mochi::capture::RestoreState(entt::registry& reg, Span<uint8_t const> data, Error& error) {
  RestorePartialState(reg, data, {}, error);
}

std::string
mochi::capture::CaptureStateToJson(entt::registry& reg, bool prettyMultiLine, Error& error) {
  MOCHI_ERROR_RETURN(error, "");

  // WARNING: This function must be kept in sync with the logic in CaptureState.

  // NOTE: The purpose of this function is to help developers understand what data is being
  // captured. It must be accurate and human readable, but the code does not need to be fast.

  // Serialize ctx components by type
  picojson::object ctxComponents;
  ecs::EnumerateComponentsWithAttribute(
      reg,
      SReflect::GetTypeId<attribute::CaptureStateCtx>(),
      [&](ecs::ComponentTypeInfo const& componentType) {
        auto const* typeInfo = componentType.TryGetReflectionInfo();
        MOCHI_ASSERT_VERBOSE(typeInfo != nullptr);
        if (void const* ctxData = componentType.TryCtx(reg)) {
          typeInfo->SerializeInner(ctxData, ctxComponents[typeInfo->_name]);
        }
      });

  // Serialize and categorize entity components by type.
  // Use std::map not std::unordered_map so that entities will be sorted.
  // This ensures that any name collisions will be resolved in a predictable order.
  std::map<entt::entity, picojson::object> actorComponents;
  std::map<entt::entity, picojson::object> constraintComponents;
  std::map<entt::entity, picojson::object> miscComponents;
  ecs::EnumerateComponentsWithAttribute(
      reg,
      SReflect::GetTypeId<attribute::CaptureState>(),
      [&](ecs::ComponentTypeInfo const& componentType) {
        auto const* typeInfo = componentType.TryGetReflectionInfo();
        MOCHI_ASSERT_VERBOSE(typeInfo != nullptr);
        auto const* attrib = typeInfo->GetAttribute<attribute::CaptureState>();
        MOCHI_ASSERT_VERBOSE(attrib != nullptr);
        MOCHI_FILO_STACK_ALLOCATOR(
            entityAllocator, kFilteredEntityStackCount * sizeof(entt::entity));
        DynamicArray<entt::entity> filteredEntities(&entityAllocator);
        auto const allEntities = componentType.GetEntities(reg);
        auto const captureEntities =
            GetCaptureEntities(reg, allEntities, *attrib, filteredEntities);
        for (entt::entity e : captureEntities.entities) {
          picojson::value* value = nullptr;
          if (reg.all_of<CActorInfo const>(e)) {
            value = &actorComponents[e][typeInfo->_name];
          } else if (reg.all_of<CConstraintInfo const>(e)) {
            value = &constraintComponents[e][typeInfo->_name];
          } else {
            value = &miscComponents[e][typeInfo->_name];
          }
          void* obj = componentType.TryGet(reg, e);
          MOCHI_ASSERT_VERBOSE(obj != nullptr);
          typeInfo->SerializeInner(obj, *value);
        }
      });

  // Build a JSON dictionary of actors data
  auto actors = picojson::object();
  for (auto&& [e, components] : actorComponents) {
    // Give this actor a unique name
    auto const& actorInfo = reg.get<CActorInfo>(e);
    std::string type = SReflect::EnumToString(actorInfo.type);
    std::string name = actorInfo.name.empty() ? type : actorInfo.name;
    if (actors.find(name) != actors.end()) {
      int counter = 2;
      std::string newName;
      do {
        newName = Format("%s%d", name.c_str(), counter);
        ++counter;
      } while (actors.find(newName) != actors.end());
      name = std::move(newName);
    }

    auto [it, wasInserted] = actors.insert(std::make_pair(name, picojson::object()));
    MOCHI_ASSERT_VERBOSE(wasInserted, "Name should have been unique");
    auto& actor = it->second;
    actor["_handle"] = static_cast<entt::id_type>(e);
    actor["_name"] = actorInfo.name;
    actor["_type"] = type;
    actor["components"] = components;
  }

  // Build a JSON dictionary of constraint data
  auto constraints = picojson::object();
  for (auto&& [e, components] : constraintComponents) {
    auto const& constraintInfo = reg.get<CConstraintInfo const>(e);
    std::string typeName = SReflect::EnumToString(constraintInfo.type);

    // Give this constraint a unique name
    std::string name = typeName;
    int counter = 2;
    while (constraints.find(name) != constraints.end()) {
      name = Format("%s%d", typeName.c_str(), counter);
      ++counter;
    }

    auto& constraint = constraints[name];
    constraint["_handle"] = static_cast<double>(e);
    constraint["_type"] = typeName;
    constraint["components"] = components;
  }

  // Built a JSON dictionary of any misc components (usually none)
  auto misc = picojson::array();
  for (auto&& [e, components] : miscComponents) {
    misc.emplace_back(picojson::object());
    auto& obj = misc.back();
    obj["_handle"] = static_cast<double>(e);
    obj["components"] = components;
  }

  // Put it all together into a human-friendly document
  auto doc = picojson::object();
  if (!actors.empty()) {
    doc["actors"] = actors;
  }
  if (!constraints.empty()) {
    doc["constraints"] = constraints;
  }
  if (!misc.empty()) {
    doc["misc"] = misc;
  }
  if (!ctxComponents.empty()) {
    doc["scene"] = ctxComponents;
  }

  auto json = picojson::value(doc);
  return json.serialize(prettyMultiLine);
}

// Reads an object from each stream. Returns true if they are equal (at least as far as
// serialization is concerned). We can't compre raw bytes because of potential padding.
static bool IsNextObjectEqual(
    SReflect::TypeInfo const& typeInfo,
    SpanStreamReader& streamA,
    SpanStreamReader& streamB) {
  // Re-serialize both to picojson::value
  picojson::value jsonA, jsonB;

  {
    void* obj = typeInfo.New();
    MOCHI_DEFER(typeInfo.Delete(obj));
    [[maybe_unused]] bool success = typeInfo.DeserializeFromBytesInner(streamA, obj);
    MOCHI_ASSERT(success);
    typeInfo.SerializeInner(obj, jsonA);
  }

  {
    void* obj = typeInfo.New();
    MOCHI_DEFER(typeInfo.Delete(obj));
    [[maybe_unused]] bool success = typeInfo.DeserializeFromBytesInner(streamB, obj);
    MOCHI_ASSERT(success);
    typeInfo.SerializeInner(obj, jsonB);
  }

  // Compare the two document models (not JSON text).
  return jsonA == jsonB;
}

bool mochi::capture::IsEqualState(
    entt::registry const& reg,
    Span<uint8_t const> stateA,
    Span<uint8_t const> stateB) {
  // This function does not return an Error parameter because it should never fail, assuming the
  // buffers were captured successfully by this same running process. The buffers might represent
  // equal state even if they are not byte-for-byte identical, because of padding.

  if (stateA.size() != stateB.size()) {
    return false; // Different sizes
  }
  if (0 == std::memcmp(stateA.data(), stateB.data(), stateA.size())) {
    return true; // Same bytes
  }

  SpanStreamReader streamA(stateA);
  SpanStreamReader streamB(stateB);

  // CaptureHeader
  CaptureHeader headerA, headerB;
  StreamRead(headerA, streamA, ErrorAssert{});
  StreamRead(headerB, streamB, ErrorAssert{});
  ValidateCaptureHeader(headerA, ErrorAssert{});
  ValidateCaptureHeader(headerB, ErrorAssert{});
  if (headerA != headerB) {
    return false;
  }

  size_t startPosA = streamA.GetPosition();
  size_t startPosB = streamB.GetPosition();

  // Ctx Components
  for (size_t iType = 0; iType < headerA.numCtxComponentTypes; ++iType) {
    ComponentHeader compHeaderA, compHeaderB;
    StreamRead(compHeaderA, streamA, ErrorAssert{});
    StreamRead(compHeaderB, streamB, ErrorAssert{});
    if (compHeaderA != compHeaderB) {
      return false;
    }
    auto const* componentType = ecs::TryGetComponentTypeInfo(reg, compHeaderA.id);
    MOCHI_ASSERT(componentType != nullptr);
    auto const* typeInfo = componentType->TryGetReflectionInfo();
    MOCHI_ASSERT(typeInfo != nullptr);
    size_t compStartPosA = streamA.GetPosition();
    size_t compStartPosB = streamB.GetPosition();
    if (!IsNextObjectEqual(*typeInfo, streamA, streamB)) {
      return false;
    }
    MOCHI_ASSERT((streamA.GetPosition() - compStartPosA) == compHeaderA.dataSize);
    MOCHI_ASSERT((streamB.GetPosition() - compStartPosB) == compHeaderB.dataSize);
  }

  // Entity Components
  for (size_t iType = 0; iType < headerA.numEntityComponentTypes; ++iType) {
    ComponentHeader compHeaderA, compHeaderB;
    StreamRead(compHeaderA, streamA, ErrorAssert{});
    StreamRead(compHeaderB, streamB, ErrorAssert{});
    if (compHeaderA != compHeaderB) {
      return false;
    }
    auto const* componentType = ecs::TryGetComponentTypeInfo(reg, compHeaderA.id);
    MOCHI_ASSERT(componentType != nullptr);
    auto const* typeInfo = componentType->TryGetReflectionInfo();
    MOCHI_ASSERT(typeInfo != nullptr);
    size_t compStartPosA = streamA.GetPosition();
    size_t compStartPosB = streamB.GetPosition();
    for (size_t iEntity = 0; iEntity < compHeaderA.count; ++iEntity) {
      if (!IsNextObjectEqual(*typeInfo, streamA, streamB)) {
        return false;
      }
    }
    MOCHI_ASSERT((streamA.GetPosition() - compStartPosA) == compHeaderA.dataSize);
    MOCHI_ASSERT((streamB.GetPosition() - compStartPosB) == compHeaderB.dataSize);
  }

  CaptureFooter footerA, footerB;
  StreamRead(footerA, streamA, ErrorAssert{});
  StreamRead(footerB, streamB, ErrorAssert{});
  ValidateCaptureFooter(footerA, ErrorAssert{});
  ValidateCaptureFooter(footerB, ErrorAssert{});
  MOCHI_ASSERT((streamA.GetPosition() - startPosA) == headerA.dataSize);
  MOCHI_ASSERT((streamB.GetPosition() - startPosB) == headerB.dataSize);

  return true;
}

void mochi::capture::details::RegisterPostRestoreCallback(
    entt::registry& reg,
    std::function<void(entt::registry&)> fn) {
  MOCHI_ASSERT(!!fn, "Invalid function");

  // Find or create global context component
  auto* callbacks = reg.try_ctx<CCaptureCallbacks>();
  if (!callbacks) {
    callbacks = &reg.set<CCaptureCallbacks>();
  }

  callbacks->postRestore.push_back(std::move(fn));
}

void mochi::capture::InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CCaptureCallbacks>(reg);
}
