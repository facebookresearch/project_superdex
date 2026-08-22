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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>
#include <mochi_physics/src/mochi_ecs_registry.h>

#include <mochi_physics/mochi_physics.h>

#include <algorithm>
#include <string>
#include <string_view>

using namespace mochi;

namespace ecs_registry_test {

// Reflection attribute
struct AttrA : public SReflect::Attribute {
  MOCHI_STRUCT_BEGIN(ecs_registry_test::AttrA);
  SR_BaseClass(SReflect::Attribute);
  MOCHI_STRUCT_END();
};

// Reflection attribute with value
struct AttrB : public SReflect::Attribute {
  std::string _string;
  AttrB(std::string_view str) : _string(str) {}
  MOCHI_STRUCT_BEGIN(ecs_registry_test::AttrB);
  MOCHI_BASE_CLASS(SReflect::Attribute);
  MOCHI_FIELD(_string);
  SR_EndStruct();
};

struct AttrC : public SReflect::Attribute {
  MOCHI_STRUCT_BEGIN(ecs_registry_test::AttrC);
  SR_BaseClass(SReflect::Attribute);
  MOCHI_STRUCT_END();
};

// Macros to use the above attributes in reflection markup (AttrC is unused)
#define MOCHI_TEST_ATTR_A()                                                \
  {                                                                        \
    SReflect::detail::AddAttribute(*myInfo, new ecs_registry_test::AttrA); \
  }
#define MOCHI_TEST_ATTR_B(str)                                                  \
  {                                                                             \
    SReflect::detail::AddAttribute(*myInfo, new ecs_registry_test::AttrB{str}); \
  }

// Tag component
struct TagA {};

// Tag component with reflection support
struct TagB {
  MOCHI_STRUCT_BEGIN(ecs_registry_test::TagB);
  MOCHI_STRUCT_END();
};

// Tag component with reflection attribute
struct TagC {
  MOCHI_STRUCT_BEGIN(ecs_registry_test::TagC);
  MOCHI_TEST_ATTR_A();
  MOCHI_STRUCT_END();
};

// Component with value
struct CCompA {
  int32_t value = 0;
};

// Component with value and reflection support
struct CCompB {
  int32_t value = 0;
  MOCHI_STRUCT_BEGIN(ecs_registry_test::CCompB);
  MOCHI_FIELD(value);
  MOCHI_STRUCT_END();
};

// Component with value and reflection attribute A
struct CCompC {
  std::string value;
  MOCHI_STRUCT_BEGIN(ecs_registry_test::CCompC);
  MOCHI_TEST_ATTR_A();
  MOCHI_FIELD(value);
  MOCHI_STRUCT_END();
};

// Component with value and two reflection attributes
struct CCompD {
  double value = 0.0;
  MOCHI_STRUCT_BEGIN(ecs_registry_test::CCompD);
  MOCHI_TEST_ATTR_B("Woot");
  MOCHI_TEST_ATTR_A();
  MOCHI_FIELD(value);
  MOCHI_STRUCT_END();
};

} // namespace ecs_registry_test

using namespace ecs_registry_test;

TEST(EcsRegistry, NoComponents) {
  entt::registry reg;
  ecs::InitializeComponentRegistryOnce(reg);
  ecs::FinalizeComponentRegistration(reg);
  EXPECT_EQ(0, ecs::GetAllComponentTypes(reg).size());
}

template <class T>
void ExpectComponentInfo(
    entt::registry& reg,
    ecs::ComponentTypeInfo const& info,
    char const* expectedName,
    bool isTag) {
  // IsTag
  EXPECT_EQ(isTag, info.IsTag());

  // GetEntities
  EXPECT_EQ(reg.storage<T>().data(), info.GetEntities(reg).data()); // Same address
  EXPECT_EQ(reg.storage<T>().size(), info.GetEntities(reg).size()); // Same size

  // ContainsEntity
  auto newEntity = reg.create();
  MOCHI_DEFER(reg.destroy(newEntity));
  EXPECT_EQ(false, info.ContainsEntity(reg, newEntity));
  for (auto e : info.GetEntities(reg)) {
    EXPECT_EQ(true, info.ContainsEntity(reg, e));
  }

  // TryGet
  EXPECT_EQ((void*)nullptr, info.TryGet(reg, newEntity));
  for (auto e : info.GetEntities(reg)) {
    EXPECT_EQ(isTag, info.TryGet(reg, e) == nullptr); // Only returns nullptr for tags
  }

  // TryGetReflectionInfo
  auto const* ti = SReflect::TryGetTypeInfo<T>();
  if (ti) {
    // Compare by TypeId, not address
    ASSERT_NE((SReflect::TypeInfo const*)nullptr, info.TryGetReflectionInfo());
    EXPECT_EQ(ti->_typeId, info.TryGetReflectionInfo()->_typeId);
    EXPECT_STREQ(expectedName, info.TryGetReflectionInfo()->_nameWithNamespace);
  } else {
    EXPECT_EQ((SReflect::TypeInfo const*)nullptr, info.TryGetReflectionInfo());
  }
}

TEST(EcsRegistry, RegisterComponent) {
  entt::registry reg;
  ecs::InitializeComponentRegistryOnce(reg);
  ecs::RegisterComponent<TagA>(reg);
  ecs::RegisterComponent<CCompA>(reg);
  ecs::RegisterComponent<TagB>(reg);
  ecs::RegisterComponent<CCompB>(reg);
  ecs::RegisterComponent<TagC>(reg);
  ecs::RegisterComponent<CCompC>(reg);
  ecs::RegisterComponent<CCompD>(reg);
  ecs::FinalizeComponentRegistration(reg);

  // Collect component types (already in sorted order)
  auto const& componentTypes = ecs::GetAllComponentTypes(reg);
  ASSERT_EQ(7, componentTypes.size());
  ExpectComponentInfo<CCompA>(reg, componentTypes[0], "ecs_registry_test::CCompA", false);
  ExpectComponentInfo<CCompB>(reg, componentTypes[1], "ecs_registry_test::CCompB", false);
  ExpectComponentInfo<CCompC>(reg, componentTypes[2], "ecs_registry_test::CCompC", false);
  ExpectComponentInfo<CCompD>(reg, componentTypes[3], "ecs_registry_test::CCompD", false);
  ExpectComponentInfo<TagA>(reg, componentTypes[4], "ecs_registry_test::TagA", true);
  ExpectComponentInfo<TagB>(reg, componentTypes[5], "ecs_registry_test::TagB", true);
  ExpectComponentInfo<TagC>(reg, componentTypes[6], "ecs_registry_test::TagC", true);
}

TEST(EcsRegistry, EnumerateComponentTypesForEntity) {
  entt::registry reg;

  // Collect components via callback
  DynamicArray<ecs::ComponentTypeInfo const*> types;
  types.reserve(4);
  auto onEach = [&](ecs::ComponentTypeInfo const& typeInfo) { types.push_back(&typeInfo); };

  // Helper to collect component types on an entity (already in sorted order)
  auto collectTypesForEntity = [&](entt::entity e) {
    types.clear();
    ecs::EnumerateComponentTypesForEntity(reg, e, onEach);
  };

  // Components
  ecs::InitializeComponentRegistryOnce(reg);
  ecs::RegisterComponent<TagA>(reg);
  ecs::RegisterComponent<TagB>(reg);
  ecs::RegisterComponent<CCompA>(reg);
  ecs::RegisterComponent<CCompB>(reg);
  ecs::FinalizeComponentRegistration(reg);

  // Create two entities
  entt::entity a = reg.create();
  entt::entity b = reg.create();

  // No components yet
  collectTypesForEntity(a);
  EXPECT_EQ(0, types.size());
  collectTypesForEntity(b);
  EXPECT_EQ(0, types.size());

  // Add one tag to each
  reg.emplace<TagA>(a);
  reg.emplace<TagB>(b);
  collectTypesForEntity(a);
  ASSERT_EQ(1, types.size());
  EXPECT_STREQ("ecs_registry_test::TagA", std::string(types[0]->GetTypeName()).c_str());
  collectTypesForEntity(b);
  ASSERT_EQ(1, types.size());
  EXPECT_STREQ("ecs_registry_test::TagB", std::string(types[0]->GetTypeName()).c_str());
}

TEST(EcsRegistry, EnumerateComponentsWithAttribute) {
  entt::registry reg;

  // Collect components via callback
  DynamicArray<ecs::ComponentTypeInfo const*> types;
  types.reserve(4);
  auto onEach = [&](ecs::ComponentTypeInfo const& typeInfo) { types.push_back(&typeInfo); };

  // Components
  ecs::InitializeComponentRegistryOnce(reg);
  ecs::RegisterComponent<TagA>(reg);
  ecs::RegisterComponent<TagB>(reg);
  ecs::RegisterComponent<TagC>(reg);
  ecs::RegisterComponent<CCompA>(reg);
  ecs::RegisterComponent<CCompB>(reg);
  ecs::RegisterComponent<CCompC>(reg);
  ecs::RegisterComponent<CCompD>(reg);
  ecs::FinalizeComponentRegistration(reg);

  // Components with AttrA
  ecs::EnumerateComponentsWithAttribute<AttrA>(reg, onEach);
  ASSERT_EQ(3, types.size());
  EXPECT_STREQ("ecs_registry_test::CCompC", std::string(types[0]->GetTypeName()).c_str());
  EXPECT_STREQ("ecs_registry_test::CCompD", std::string(types[1]->GetTypeName()).c_str());
  EXPECT_STREQ("ecs_registry_test::TagC", std::string(types[2]->GetTypeName()).c_str());

  // Components with AttrB
  types.clear();
  ecs::EnumerateComponentsWithAttribute<AttrB>(reg, onEach);
  ASSERT_EQ(1, types.size());
  EXPECT_STREQ("ecs_registry_test::CCompD", std::string(types[0]->GetTypeName()).c_str());

  // Components with AttrC (none)
  types.clear();
  ecs::EnumerateComponentsWithAttribute<AttrC>(reg, onEach);
  EXPECT_EQ(0, types.size());
}

TEST(EcsRegistry, TryGetComponentTypeInfo) {
  entt::registry reg;

  // Components
  ecs::InitializeComponentRegistryOnce(reg);
  ecs::RegisterComponent<TagA>(reg);
  ecs::RegisterComponent<TagB>(reg);
  ecs::RegisterComponent<CCompA>(reg);
  ecs::RegisterComponent<CCompB>(reg);
  ecs::FinalizeComponentRegistration(reg);

  // Lookup an invalid TypeId
  EXPECT_EQ(
      (ecs::ComponentTypeInfo const*)nullptr,
      ecs::TryGetComponentTypeInfo(reg, SReflect::TypeId{}));

  // Lookup a TypeId that was not registered.
  EXPECT_EQ(
      (ecs::ComponentTypeInfo const*)nullptr,
      ecs::TryGetComponentTypeInfo(reg, SReflect::GetTypeId<CCompC>()));

  // Lookup TagB
  auto const* info = ecs::TryGetComponentTypeInfo(reg, SReflect::GetTypeId<TagB>());
  ASSERT_NE((ecs::ComponentTypeInfo const*)nullptr, info);
  ExpectComponentInfo<TagB>(reg, *info, "ecs_registry_test::TagB", true);

  // Lookup CCompB
  info = ecs::TryGetComponentTypeInfo(reg, SReflect::GetTypeId<CCompB>());
  ASSERT_NE((ecs::ComponentTypeInfo const*)nullptr, info);
  ExpectComponentInfo<CCompB>(reg, *info, "ecs_registry_test::CCompB", false);

  // Repeat the above, but look up the components by entt::id_type instead.
  EXPECT_EQ(
      (ecs::ComponentTypeInfo const*)nullptr, ecs::TryGetComponentTypeInfo(reg, entt::id_type{}));
  EXPECT_EQ(
      (ecs::ComponentTypeInfo const*)nullptr,
      ecs::TryGetComponentTypeInfo(reg, entt::type_id<CCompC>().hash()));
  info = ecs::TryGetComponentTypeInfo(reg, entt::type_id<TagB>().hash());
  ASSERT_NE((ecs::ComponentTypeInfo const*)nullptr, info);
  ExpectComponentInfo<TagB>(reg, *info, "ecs_registry_test::TagB", true);
  info = ecs::TryGetComponentTypeInfo(reg, entt::type_id<CCompB>().hash());
  ASSERT_NE((ecs::ComponentTypeInfo const*)nullptr, info);
  ExpectComponentInfo<CCompB>(reg, *info, "ecs_registry_test::CCompB", false);
}

TEST(EcsRegistry, TryGet) {
  entt::registry reg;

  // Components
  ecs::InitializeComponentRegistryOnce(reg);
  ecs::RegisterComponent<TagA>(reg);
  ecs::RegisterComponent<CCompA>(reg);
  ecs::RegisterComponent<CCompB>(reg);
  ecs::RegisterComponent<CCompC>(reg);
  ecs::RegisterComponent<CCompD>(reg);
  ecs::FinalizeComponentRegistration(reg);

  // Get component types (already in sorted order)
  auto const& componentTypes = ecs::GetAllComponentTypes(reg);
  ASSERT_EQ(5, componentTypes.size());

  // Create entiteis
  entt::entity entitiesBuffer[] = {reg.create(), reg.create(), reg.create(), reg.create()};
  auto entities = MakeSpan(entitiesBuffer);

  // No entities have components yet
  EXPECT_EQ(0, componentTypes[0].GetEntities(reg).size());
  EXPECT_EQ(0, componentTypes[1].GetEntities(reg).size());
  EXPECT_EQ(0, componentTypes[2].GetEntities(reg).size());

  // Add various components to various entities
  reg.emplace<CCompA>(entities[0], 11);

  reg.emplace<CCompB>(entities[0], 22);
  reg.emplace<CCompB>(entities[1], 33);

  reg.emplace<CCompC>(entities[0], std::string{"44"});
  reg.emplace<CCompC>(entities[1], std::string{"55"});
  reg.emplace<CCompC>(entities[2], std::string{"66"});

  reg.emplace<CCompD>(entities[0], 7.0);
  reg.emplace<CCompD>(entities[1], 7.1);
  reg.emplace<CCompD>(entities[2], 7.2);
  reg.emplace<CCompD>(entities[3], 7.3);

  reg.emplace<TagA>(entities[0]);
  reg.emplace<TagA>(entities[2]);

  // Global context is not reported as part of the per-entity storage.
  reg.set<CCompA>();
  reg.set<TagA>();

  // Access storage for CCompA
  {
    auto const& compType = componentTypes[0];
    ExpectComponentInfo<CCompA>(reg, compType, "ecs_registry_test::CCompA", false);
    entt::entity const expectedEntities[] = {entities[0]};
    EXPECT_TRUE(test::EqualSpanUnordered(MakeSpan(expectedEntities), compType.GetEntities(reg)));
    EXPECT_EQ(11, static_cast<CCompA const*>(compType.TryGet(reg, entities[0]))->value);
  }

  // Access storage for CCompB
  {
    auto const& compType = componentTypes[1];
    ExpectComponentInfo<CCompB>(reg, compType, "ecs_registry_test::CCompB", false);
    entt::entity const expectedEntities[] = {entities[0], entities[1]};
    EXPECT_TRUE(test::EqualSpanUnordered(MakeSpan(expectedEntities), compType.GetEntities(reg)));
    EXPECT_EQ(22, static_cast<CCompB const*>(compType.TryGet(reg, entities[0]))->value);
    EXPECT_EQ(33, static_cast<CCompB const*>(compType.TryGet(reg, entities[1]))->value);
  }

  // Access storage for CCompC
  {
    auto const& compType = componentTypes[2];
    ExpectComponentInfo<CCompC>(reg, compType, "ecs_registry_test::CCompC", false);
    entt::entity const expectedEntities[] = {entities[0], entities[1], entities[2]};
    EXPECT_TRUE(test::EqualSpanUnordered(MakeSpan(expectedEntities), compType.GetEntities(reg)));
    EXPECT_STREQ(
        "44", static_cast<CCompC const*>(compType.TryGet(reg, entities[0]))->value.c_str());
    EXPECT_STREQ(
        "55", static_cast<CCompC const*>(compType.TryGet(reg, entities[1]))->value.c_str());
    EXPECT_STREQ(
        "66", static_cast<CCompC const*>(compType.TryGet(reg, entities[2]))->value.c_str());
  }

  // Access storage for CCompC
  {
    auto const& compType = componentTypes[3];
    ExpectComponentInfo<CCompD>(reg, compType, "ecs_registry_test::CCompD", false);
    entt::entity const expectedEntities[] = {entities[0], entities[1], entities[2], entities[3]};
    EXPECT_TRUE(test::EqualSpanUnordered(MakeSpan(expectedEntities), compType.GetEntities(reg)));
    EXPECT_EQ(7.0, static_cast<CCompD const*>(compType.TryGet(reg, entities[0]))->value);
    EXPECT_EQ(7.1, static_cast<CCompD const*>(compType.TryGet(reg, entities[1]))->value);
    EXPECT_EQ(7.2, static_cast<CCompD const*>(compType.TryGet(reg, entities[2]))->value);
    EXPECT_EQ(7.3, static_cast<CCompD const*>(compType.TryGet(reg, entities[3]))->value);
  }

  // Access storage for TagA
  {
    auto const& compType = componentTypes[4];
    ExpectComponentInfo<TagA>(reg, compType, "ecs_registry_test::TagA", true);
    entt::entity const expectedEntities[] = {entities[0], entities[2]};
    EXPECT_TRUE(test::EqualSpanUnordered(MakeSpan(expectedEntities), compType.GetEntities(reg)));
    EXPECT_EQ((void const*)nullptr, compType.TryGet(reg, entities[0]));
    EXPECT_EQ((void const*)nullptr, compType.TryGet(reg, entities[2]));
  }
}

TEST(EcsRegistry, TryGetPageTable) {
  // EnTT stores component memory in fixed-size pages. This is useful to know for bulk operations
  // (e.g. memcpy) on POD types. However, it is a detail of EnTT that could change, so we test it.
  entt::registry reg;

  // Components
  ecs::InitializeComponentRegistryOnce(reg);
  ecs::RegisterComponent<CCompA>(reg);
  ecs::FinalizeComponentRegistration(reg);
  auto const& compType = ecs::GetAllComponentTypes(reg)[0];
  ExpectComponentInfo<CCompA>(reg, compType, "ecs_registry_test::CCompA", false);

  // Create many entities with the same component type. Each with a unique value.
  int constexpr kPageSize = ENTT_PACKED_PAGE;
  int constexpr kNumEntities = kPageSize * 5 / 2; // More than 2 pages
  DynamicArray<entt::entity> entities(kNumEntities);
  for (int i = 0; i < kNumEntities; ++i) {
    entities[i] = reg.create();
    auto& comp = reg.emplace<CCompA>(entities[i]);
    comp.value = i;
  }

  // Global context is not reported as part of the per-entity storage.
  reg.set<CCompA>();

  // Read each component in each page
  auto const* pageTable = reinterpret_cast<CCompA const* const*>(compType.TryGetPageTable(reg));
  for (int i = 0; i < kNumEntities; ++i) {
    int iPage = i / ENTT_PACKED_PAGE;
    int iOffset = i % ENTT_PACKED_PAGE;
    EXPECT_EQ(i, pageTable[iPage][iOffset].value);
  }
}

TEST(EcsRegistry, TryCtx) {
  entt::registry reg;
  entt::registry const& creg = reg;

  // Components
  ecs::InitializeComponentRegistryOnce(reg);
  ecs::RegisterComponent<CCompA>(reg);
  ecs::RegisterComponent<TagA>(reg);
  ecs::FinalizeComponentRegistration(reg);

  auto const& componentTypes = ecs::GetAllComponentTypes(reg);
  ASSERT_EQ(2, isize(componentTypes));
  auto const& compAInfo = componentTypes[0];
  auto const& tagAInfo = componentTypes[1];
  ExpectComponentInfo<CCompA>(reg, compAInfo, "ecs_registry_test::CCompA", false);
  ExpectComponentInfo<TagA>(reg, tagAInfo, "ecs_registry_test::TagA", true);

  // No components have been set as global context yet
  EXPECT_EQ((void*)nullptr, compAInfo.TryCtx(reg));
  EXPECT_EQ((void*)nullptr, compAInfo.TryCtx(creg));
  EXPECT_EQ((void*)nullptr, tagAInfo.TryCtx(reg));
  EXPECT_EQ((void*)nullptr, tagAInfo.TryCtx(creg));

  // Set CCompA
  reg.set<CCompA>(123);
  auto* compA = static_cast<CCompA*>(compAInfo.TryCtx(reg));
  EXPECT_EQ(compA, compAInfo.TryCtx(creg)); // same address
  ASSERT_NE((CCompA*)nullptr, compA);
  EXPECT_EQ(123, compA->value);
  EXPECT_EQ((void*)nullptr, tagAInfo.TryCtx(reg));
  EXPECT_EQ((void*)nullptr, tagAInfo.TryCtx(creg));

  // Set TagA (unlike tags emplaced on entities, these actually have a storage address)
  reg.set<TagA>();
  EXPECT_EQ(compA, compAInfo.TryCtx(reg)); // no change
  EXPECT_EQ(compA, compAInfo.TryCtx(creg)); // no change
  ASSERT_NE((TagA*)nullptr, static_cast<TagA*>(tagAInfo.TryCtx(reg)));
  ASSERT_NE((TagA*)nullptr, static_cast<TagA const*>(tagAInfo.TryCtx(creg)));
}

TEST(EcsRegistry, EnumerateGlobalCtxComponentTypes) {
  entt::registry reg;
  entt::registry const& creg = reg;

  // Components
  ecs::InitializeComponentRegistryOnce(reg);
  ecs::RegisterComponent<CCompA>(reg);
  ecs::RegisterComponent<TagA>(reg);
  ecs::FinalizeComponentRegistration(reg);

  auto const& componentTypes = ecs::GetAllComponentTypes(reg);
  ASSERT_EQ(2, isize(componentTypes));
  auto const& compAInfo = componentTypes[0];
  auto const& tagAInfo = componentTypes[1];
  ExpectComponentInfo<CCompA>(reg, compAInfo, "ecs_registry_test::CCompA", false);
  ExpectComponentInfo<TagA>(reg, tagAInfo, "ecs_registry_test::TagA", true);

  // Helper to collect comonent types via callback
  DynamicArray<ecs::ComponentTypeInfo const*> types;
  types.reserve(2);
  auto onEach = [&](ecs::ComponentTypeInfo const& typeInfo) { types.push_back(&typeInfo); };
  auto collectCtxTypes = [&]() {
    types.clear();
    ecs::EnumerateGlobalCtxComponentTypes(creg, onEach);
  };

  // No components have been set as global context yet
  collectCtxTypes();
  EXPECT_EQ(0, isize(types));

  // Components emplaced on entities are not reported as ctx components
  auto e = reg.create();
  reg.emplace<CCompA>(e);
  collectCtxTypes();
  EXPECT_EQ(0, isize(types)); // no change

  // Set CCompA
  reg.set<CCompA>();
  collectCtxTypes();
  ASSERT_EQ(1, isize(types));
  EXPECT_EQ(&compAInfo, types[0]);

  // Set TagA
  reg.set<TagA>();
  collectCtxTypes(); // Note that types are sorted by name
  ASSERT_EQ(2, isize(types));
  EXPECT_EQ(&compAInfo, types[0]); // no change
  EXPECT_EQ(&tagAInfo, types[1]);
}
