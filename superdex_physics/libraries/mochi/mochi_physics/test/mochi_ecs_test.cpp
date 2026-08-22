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

#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/src/mochi_ecs.h>

#include <gtest/gtest.h>

#include <atomic>
#include <functional>
#include <memory>
#include <type_traits>
#include <unordered_set>
#include <vector>

using namespace mochi;
using namespace mochi::ecs;

namespace {

// NOLINTBEGIN(clang-diagnostic-unused-parameter) - This file contains many lambdas to test ESC
// system functionality, but they don't actually use the lambda parameters.

/******************************************************************************************
 ECS Components
*/
struct ComponentA {
  int value;
};

struct ComponentB {
  int value;
};

struct ComponentC {
  int value;
};

struct TagA {};

struct ExternalData {
  int valueToAdd;
};

struct ExternalDataOutput {
  int value;
};

struct GlobalData {
  int valueToAdd;
};

struct GlobalData2 {
  int valueToAdd;
};

/******************************************************************************************
 ECS Systems
*/

// A system that is invoked on the whole registry, rather than an individual system
void GlobalSystem(View<ComponentA, ComponentB> view, GlobalData const& global) {
  // Add global.valueToAdd to all ComponentA's
  for (auto e : view) {
    view.get<ComponentA>(e).value += global.valueToAdd;
  }

  int sum = 0;
  for (auto e : view) {
    sum += view.get<ComponentA>(e).value;
  }

  // Set the value of all ComponentB's to the sum of all ComponentA's
  for (auto e : view) {
    view.get<ComponentB>(e).value = sum;
  }
}

void GlobalSystem2(
    View<ComponentA const, ComponentB> view,
    View<ComponentA const, ComponentC, TagA const> view2) {
  for (auto e : view) {
    if (view.get<ComponentA const>(e).value > 0) {
      view.get<ComponentB>(e).value = 1000;
    }
  }

  for (auto e : view2) {
    view2.get<ComponentC>(e).value = 1000;
  }
}

void GlobalSystemWithPartialRegistry(
    PartialRegistry<ComponentA, ComponentB, ComponentC, TagA const> reg) {
  auto vA = reg.view<ComponentA>();
  for (auto e : vA) {
    vA.get<ComponentA>(e).value += 1;
  }

  auto vB = reg.view<ComponentB, TagA const>();
  for (auto e : vB) {
    vB.get<ComponentB>(e).value += 1;
  }

  auto vANB = reg.view<ComponentA>(entt::exclude_t<ComponentB>());
  for (auto e : vANB) {
    reg.emplace<ComponentB>(e, 1000);
  }

  auto vAC = reg.view<ComponentA, ComponentC>();
  for (auto e : vAC) {
    reg.remove<ComponentA>(e);
  }
}

/******************************************************************************************
 Test Cases
*/

TEST(EcsSystem, CatTypeLists) {
  // If this test compiles, then it passes.
  // clang-format off
  static_assert(std::is_same_v<entt::type_list<>, decltype(CatTypeLists(entt::type_list<>{}, entt::type_list<>{}))>);
  static_assert(std::is_same_v<entt::type_list<int>, decltype(CatTypeLists(entt::type_list<int>{}, entt::type_list<>{}))>);
  static_assert(std::is_same_v<entt::type_list<int>, decltype(CatTypeLists(entt::type_list<>{}, entt::type_list<int>{}))>);
  static_assert(std::is_same_v<entt::type_list<int, float, int, double>, decltype(CatTypeLists(entt::type_list<>{}, entt::type_list<int>{}, entt::type_list<float, int, double>{}))>);
  // clang-format on
}

TEST(EcsSystem, GetUniqueTypes) {
  // If this test compiles, then it passes.
  // clang-format off
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetUniqueTypes(entt::type_list<>{}))>);
  static_assert(std::is_same_v<entt::type_list<int>, decltype(GetUniqueTypes(entt::type_list<int>{}))>);
  static_assert(std::is_same_v<entt::type_list<int, float>, decltype(GetUniqueTypes(entt::type_list<int, float>{}))>);
  static_assert(std::is_same_v<entt::type_list<float, int, double>, decltype(GetUniqueTypes(entt::type_list<int, float, int, double>{}))>);
  static_assert(std::is_same_v<entt::type_list<float, int, double>, decltype(GetUniqueTypes(entt::type_list<>{}, entt::type_list<int>{}, entt::type_list<float, int, double>{}))>);
  // clang-format on
}

TEST(EcsSystem, SelectConstTypes) {
  // If this test compiles, then it passes.
  // clang-format off
  static_assert(std::is_same_v<entt::type_list<>, decltype(SelectConstTypes(entt::type_list<>{}))>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(SelectConstTypes(entt::type_list<int>{}))>);
  static_assert(std::is_same_v<entt::type_list<int const>, decltype(SelectConstTypes(entt::type_list<int const>{}))>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(SelectConstTypes(entt::type_list<int*>{}))>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(SelectConstTypes(entt::type_list<int const*>{}))>); // std::is_const_v<int const*> is false
  static_assert(std::is_same_v<entt::type_list<int const* const>, decltype(SelectConstTypes(entt::type_list<int const* const>{}))>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(SelectConstTypes(entt::type_list<int&>{}))>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(SelectConstTypes(entt::type_list<int const&>{}))>); // std::is_const_v<int const&> is false
  static_assert(std::is_same_v<entt::type_list<>, decltype(SelectConstTypes(entt::type_list<float, int>{}))>);
  static_assert(std::is_same_v<entt::type_list<float const>, decltype(SelectConstTypes(entt::type_list<float const, int>{}))>);
  static_assert(std::is_same_v<entt::type_list<int const>, decltype(SelectConstTypes(entt::type_list<float, int const>{}))>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(SelectConstTypes(entt::type_list<float, int, double>{}))>);
  static_assert(std::is_same_v<entt::type_list<float const>, decltype(SelectConstTypes(entt::type_list<float const, int, double>{}))>);
  static_assert(std::is_same_v<entt::type_list<int const>, decltype(SelectConstTypes(entt::type_list<float, int const, double>{}))>);
  static_assert(std::is_same_v<entt::type_list<double const>, decltype(SelectConstTypes(entt::type_list<float, int, double const>{}))>);
  static_assert(std::is_same_v<entt::type_list<float const, double const>, decltype(SelectConstTypes(entt::type_list<float const, int, double const>{}))>);
  // clang-format on
}

TEST(EcsSystem, SelectNonConstTypes) {
  // If this test compiles, then it passes.
  // clang-format off
  static_assert(std::is_same_v<entt::type_list<>, decltype(SelectNonConstTypes(entt::type_list<>{}))>);
  static_assert(std::is_same_v<entt::type_list<int>, decltype(SelectNonConstTypes(entt::type_list<int>{}))>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(SelectNonConstTypes(entt::type_list<int const>{}))>);
  static_assert(std::is_same_v<entt::type_list<int*>, decltype(SelectNonConstTypes(entt::type_list<int*>{}))>);
  static_assert(std::is_same_v<entt::type_list<int const*>, decltype(SelectNonConstTypes(entt::type_list<int const*>{}))>); // std::is_const_v<int const*> is false
  static_assert(std::is_same_v<entt::type_list<>, decltype(SelectNonConstTypes(entt::type_list<int const* const>{}))>);
  static_assert(std::is_same_v<entt::type_list<int&>, decltype(SelectNonConstTypes(entt::type_list<int&>{}))>);
  static_assert(std::is_same_v<entt::type_list<int const&>, decltype(SelectNonConstTypes(entt::type_list<int const&>{}))>); // std::is_const_v<int const&> is false
  static_assert(std::is_same_v<entt::type_list<float, int>, decltype(SelectNonConstTypes(entt::type_list<float, int>{}))>);
  static_assert(std::is_same_v<entt::type_list<int>, decltype(SelectNonConstTypes(entt::type_list<float const, int>{}))>);
  static_assert(std::is_same_v<entt::type_list<float>, decltype(SelectNonConstTypes(entt::type_list<float, int const>{}))>);
  static_assert(std::is_same_v<entt::type_list<float, int, double>, decltype(SelectNonConstTypes(entt::type_list<float, int, double>{}))>);
  static_assert(std::is_same_v<entt::type_list<int, double>, decltype(SelectNonConstTypes(entt::type_list<float const, int, double>{}))>);
  static_assert(std::is_same_v<entt::type_list<float, double>, decltype(SelectNonConstTypes(entt::type_list<float, int const, double>{}))>);
  static_assert(std::is_same_v<entt::type_list<float, int>, decltype(SelectNonConstTypes(entt::type_list<float, int, double const>{}))>);
  static_assert(std::is_same_v<entt::type_list<int>, decltype(SelectNonConstTypes(entt::type_list<float const, int, double const>{}))>);
  // clang-format on
}

TEST(EcsSystem, AddConst) {
  // If this test compiles, then it passes.
  // clang-format off
  static_assert(std::is_same_v<entt::type_list<>, decltype(AddConst(entt::type_list<>{}))>);
  static_assert(std::is_same_v<entt::type_list<int const>, decltype(AddConst(entt::type_list<int>{}))>);
  static_assert(std::is_same_v<entt::type_list<int const>, decltype(AddConst(entt::type_list<int const>{}))>);
  static_assert(std::is_same_v<entt::type_list<int* const>, decltype(AddConst(entt::type_list<int*>{}))>);
  static_assert(std::is_same_v<entt::type_list<int* const>, decltype(AddConst(entt::type_list<int* const>{}))>);
  static_assert(std::is_same_v<entt::type_list<int&>, decltype(AddConst(entt::type_list<int&>{}))>); // std::add_const_v<int&> is still int&
  static_assert(std::is_same_v<entt::type_list<int const&>, decltype(AddConst(entt::type_list<int const&>{}))>);
  static_assert(std::is_same_v<entt::type_list<int const, float const>, decltype(AddConst(entt::type_list<int, float>{}))>);
  static_assert(std::is_same_v<entt::type_list<int const, float const>, decltype(AddConst(entt::type_list<int const, float>{}))>);
  static_assert(std::is_same_v<entt::type_list<int const, float const>, decltype(AddConst(entt::type_list<int, float const>{}))>);
  static_assert(std::is_same_v<entt::type_list<int const, float const>, decltype(AddConst(entt::type_list<int const, float const>{}))>);
  // clang-format on
}

TEST(EcsSystem, GetTypeListIntersection) {
  // If this test compiles, then it passes.
  // clang-format off
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetTypeListIntersection(entt::type_list<>{}, entt::type_list<>{}))>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetTypeListIntersection(entt::type_list<int>{}, entt::type_list<>{}))>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetTypeListIntersection(entt::type_list<>{}, entt::type_list<int>{}))>);
  static_assert(std::is_same_v<entt::type_list<int>, decltype(GetTypeListIntersection(entt::type_list<int>{}, entt::type_list<int>{}))>);
  static_assert(std::is_same_v<entt::type_list<int>, decltype(GetTypeListIntersection(entt::type_list<int, float>{}, entt::type_list<int>{}))>);
  static_assert(std::is_same_v<entt::type_list<int>, decltype(GetTypeListIntersection(entt::type_list<float, int>{}, entt::type_list<int>{}))>);
  static_assert(std::is_same_v<entt::type_list<int>, decltype(GetTypeListIntersection(entt::type_list<int, int>{}, entt::type_list<int>{}))>);
  static_assert(std::is_same_v<entt::type_list<int>, decltype(GetTypeListIntersection(entt::type_list<int>{}, entt::type_list<int, int>{}))>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetTypeListIntersection(entt::type_list<int, float, double>{}, entt::type_list<int*>{}))>);
  static_assert(std::is_same_v<entt::type_list<int>, decltype(GetTypeListIntersection(entt::type_list<int, float, double>{}, entt::type_list<int>{}))>);
  static_assert(std::is_same_v<entt::type_list<float>, decltype(GetTypeListIntersection(entt::type_list<int, float, double>{}, entt::type_list<float>{}))>);
  static_assert(std::is_same_v<entt::type_list<double>, decltype(GetTypeListIntersection(entt::type_list<int, float, double>{}, entt::type_list<double>{}))>);
  static_assert(std::is_same_v<entt::type_list<int, float>, decltype(GetTypeListIntersection(entt::type_list<int, float, double>{}, entt::type_list<int, float>{}))>);
  static_assert(std::is_same_v<entt::type_list<double, int>, decltype(GetTypeListIntersection(entt::type_list<int, float, double>{}, entt::type_list<double, int>{}))>);
  // clang-format on
}

TEST(EcsSystem, GetReadParamDependencies) {
  // If this test compiles, then it passes.
  // clang-format off
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetReadParamDependencies<>())>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetReadParamDependencies<ComponentA&>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA const>, decltype(GetReadParamDependencies<ComponentA const&>())>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetReadParamDependencies<ComponentA*>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA const>, decltype(GetReadParamDependencies<ComponentA const*>())>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetReadParamDependencies<ComponentA&, ComponentB&>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA const>, decltype(GetReadParamDependencies<ComponentA const&, ComponentB&>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentB const>, decltype(GetReadParamDependencies<ComponentA&, ComponentB const&>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA const, ComponentB const>, decltype(GetReadParamDependencies<ComponentA const&, ComponentB const&>())>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetReadParamDependencies<CtxGlobal<ComponentA>>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA const>, decltype(GetReadParamDependencies<CtxGlobal<ComponentA const>>())>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetReadParamDependencies<View<ComponentA>>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA const>, decltype(GetReadParamDependencies<View<ComponentA const>>())>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetReadParamDependencies<View<ComponentA, ComponentB>>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA const>, decltype(GetReadParamDependencies<View<ComponentA const, ComponentB>>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentB const>, decltype(GetReadParamDependencies<View<ComponentA, ComponentB const>>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA const, ComponentB const>, decltype(GetReadParamDependencies<View<ComponentA const, ComponentB const>>())>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetReadParamDependencies<PartialRegistry<ComponentA, ComponentB>>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA const>, decltype(GetReadParamDependencies<PartialRegistry<ComponentA const, ComponentB>>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentB const>, decltype(GetReadParamDependencies<PartialRegistry<ComponentA, ComponentB const>>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA const, ComponentB const>, decltype(GetReadParamDependencies<PartialRegistry<ComponentA const, ComponentB const>>())>);
  // clang-format on
}

TEST(EcsSystem, GetWriteParamDependencies) {
  // If this test compiles, then it passes.
  // clang-format off
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetWriteParamDependencies<>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA>, decltype(GetWriteParamDependencies<ComponentA&>())>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetWriteParamDependencies<ComponentA const&>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA>, decltype(GetWriteParamDependencies<ComponentA*>())>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetWriteParamDependencies<ComponentA const*>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA, ComponentB>, decltype(GetWriteParamDependencies<ComponentA&, ComponentB&>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentB>, decltype(GetWriteParamDependencies<ComponentA const&, ComponentB&>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA>, decltype(GetWriteParamDependencies<ComponentA&, ComponentB const&>())>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetWriteParamDependencies<ComponentA const&, ComponentB const&>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA>, decltype(GetWriteParamDependencies<CtxGlobal<ComponentA>>())>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetWriteParamDependencies<CtxGlobal<ComponentA const>>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA>, decltype(GetWriteParamDependencies<View<ComponentA>>())>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetWriteParamDependencies<View<ComponentA const>>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA, ComponentB>, decltype(GetWriteParamDependencies<View<ComponentA, ComponentB>>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentB>, decltype(GetWriteParamDependencies<View<ComponentA const, ComponentB>>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA>, decltype(GetWriteParamDependencies<View<ComponentA, ComponentB const>>())>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetWriteParamDependencies<View<ComponentA const, ComponentB const>>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA, ComponentB>, decltype(GetWriteParamDependencies<PartialRegistry<ComponentA, ComponentB>>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentB>, decltype(GetWriteParamDependencies<PartialRegistry<ComponentA const, ComponentB>>())>);
  static_assert(std::is_same_v<entt::type_list<ComponentA>, decltype(GetWriteParamDependencies<PartialRegistry<ComponentA, ComponentB const>>())>);
  static_assert(std::is_same_v<entt::type_list<>, decltype(GetWriteParamDependencies<PartialRegistry<ComponentA const, ComponentB const>>())>);
  // clang-format on
}

TEST(EcsSystem, AreReadsAndWritesIndependent) {
  // These reads are NOT affected by the writes
  static_assert(AreReadsAndWritesIndependent<>());
  static_assert(AreReadsAndWritesIndependent<ComponentA&>());
  static_assert(AreReadsAndWritesIndependent<ComponentA const&>());
  static_assert(AreReadsAndWritesIndependent<ComponentA&, ComponentB&>());
  static_assert(AreReadsAndWritesIndependent<ComponentA const&, ComponentB&>());
  static_assert(AreReadsAndWritesIndependent<ComponentA&, ComponentB const&>());
  static_assert(AreReadsAndWritesIndependent<ComponentA const*, ComponentB*>());
  static_assert(AreReadsAndWritesIndependent<View<ComponentA const>, ComponentB&>());
  static_assert(AreReadsAndWritesIndependent<ComponentA, View<ComponentB const&>>());
  static_assert(AreReadsAndWritesIndependent<PartialRegistry<ComponentA const>, ComponentB&>());
  static_assert(AreReadsAndWritesIndependent<PartialRegistry<ComponentA>, ComponentB const&>());
  static_assert(AreReadsAndWritesIndependent<CtxGlobal<ComponentA const>, ComponentB&>());
  static_assert(AreReadsAndWritesIndependent<CtxGlobal<ComponentA>, ComponentB const&>());

  // These reads ARE affected by the writes
  static_assert(!AreReadsAndWritesIndependent<ComponentA&, ComponentA const&>());
  static_assert(!AreReadsAndWritesIndependent<ComponentA const&, ComponentA&>());
  static_assert(!AreReadsAndWritesIndependent<ComponentA*, ComponentA const*>());
  static_assert(!AreReadsAndWritesIndependent<View<ComponentA const>, ComponentA&>());
  static_assert(!AreReadsAndWritesIndependent<PartialRegistry<ComponentA const>, ComponentA&>());
}

TEST(EcsSystem, LooksLikeMutableDataAccess) {
  // If this test compiles, then it passes.
  struct Thing {};

  // Looks mutable (bad)
  static_assert(LooksLikeMutableDataAccess<int*>());
  static_assert(LooksLikeMutableDataAccess<int* const>());
  static_assert(LooksLikeMutableDataAccess<int&>());
  static_assert(LooksLikeMutableDataAccess<int&&>());
  static_assert(LooksLikeMutableDataAccess<int*, float>());
  static_assert(LooksLikeMutableDataAccess<int, float*>());
  static_assert(LooksLikeMutableDataAccess<int* const, float>());
  static_assert(LooksLikeMutableDataAccess<int, float* const>());
  static_assert(LooksLikeMutableDataAccess<Thing&>());
  static_assert(LooksLikeMutableDataAccess<Thing&&>());
  static_assert(LooksLikeMutableDataAccess<Thing*>());
  static_assert(LooksLikeMutableDataAccess<std::shared_ptr<int>>());
  static_assert(LooksLikeMutableDataAccess<Span<int>>());
  static_assert(LooksLikeMutableDataAccess<Span<int> const&>());
  static_assert(LooksLikeMutableDataAccess<std::vector<int>&>());
  static_assert(LooksLikeMutableDataAccess<std::reference_wrapper<int>>());

  // Looks immutable (good)
  static_assert(!LooksLikeMutableDataAccess<int>());
  static_assert(!LooksLikeMutableDataAccess<int, float>());
  static_assert(!LooksLikeMutableDataAccess<int const*>());
  static_assert(!LooksLikeMutableDataAccess<int const&>());
  static_assert(!LooksLikeMutableDataAccess<int const*, float const&>());
  static_assert(!LooksLikeMutableDataAccess<int const&, float const*>());
  static_assert(!LooksLikeMutableDataAccess<Thing>());
  static_assert(!LooksLikeMutableDataAccess<Thing const&>());
  static_assert(!LooksLikeMutableDataAccess<Thing const*>());
  static_assert(!LooksLikeMutableDataAccess<std::shared_ptr<int const>>());
  static_assert(!LooksLikeMutableDataAccess<Span<int const>>());
  static_assert(!LooksLikeMutableDataAccess<Span<int const> const&>());
  static_assert(!LooksLikeMutableDataAccess<std::vector<int> const&>());
  static_assert(!LooksLikeMutableDataAccess<std::reference_wrapper<int const>>());
}

TEST(EcsSystem, PerEntitySystemMetadata) {
  {
    auto info = GetSystemInfo(+[](ComponentA& /*comp*/) {});
    EXPECT_EQ(info.writes.size(), 1);
    EXPECT_TRUE(info.writes.find(entt::type_id<ComponentA>()) != info.writes.end());
    EXPECT_TRUE(info.parameters[0].paramType == SystemParamType::ComponentWrite);
    EXPECT_TRUE(info.parameters[0].IsComponent());
    EXPECT_TRUE(!info.parameters[0].IsOptional());
    EXPECT_TRUE(info.parameters[0].dependencies[0].type == entt::type_id<ComponentA>());
    EXPECT_TRUE(info.parameters[0].dependencies[0].mode == AccessMode::Write);
  }
  {
    auto info = GetSystemInfo(+[](ComponentA const&, ComponentB&) {});
    EXPECT_EQ(info.writes.size(), 1);
    EXPECT_EQ(info.reads.size(), 1);
    EXPECT_TRUE(info.reads.find(entt::type_id<ComponentA>()) != info.reads.end());
    EXPECT_TRUE(info.writes.find(entt::type_id<ComponentB>()) != info.writes.end());
    EXPECT_TRUE(info.parameters[0].dependencies[0].type == entt::type_id<ComponentA>());
    EXPECT_TRUE(info.parameters[0].dependencies[0].mode == AccessMode::Read);
    EXPECT_TRUE(info.parameters[0].IsComponent());
    EXPECT_TRUE(info.parameters[1].dependencies[0].type == entt::type_id<ComponentB>());
    EXPECT_TRUE(info.parameters[1].dependencies[0].mode == AccessMode::Write);
    EXPECT_TRUE(info.parameters[1].IsComponent());
  }
  {
    auto info = GetSystemInfo(+[](ComponentA&, OptionalTag<TagA>) {});
    EXPECT_EQ(info.writes.size(), 1);
    EXPECT_EQ(info.reads.size(), 1);
    EXPECT_TRUE(info.reads.find(entt::type_id<TagA>()) != info.reads.end());
    EXPECT_TRUE(info.writes.find(entt::type_id<ComponentA>()) != info.writes.end());
    EXPECT_TRUE(info.parameters[0].dependencies[0].type == entt::type_id<ComponentA>());
    EXPECT_TRUE(info.parameters[0].dependencies[0].mode == AccessMode::Write);
    EXPECT_TRUE(info.parameters[0].IsComponent());
    EXPECT_TRUE(info.parameters[1].dependencies[0].type == entt::type_id<TagA>());
    EXPECT_TRUE(info.parameters[1].dependencies[0].mode == AccessMode::Read);
  }
  {
    auto info = GetSystemInfo(+[](ComponentA const*, ComponentB&) {});
    EXPECT_EQ(info.writes.size(), 1);
    EXPECT_EQ(info.reads.size(), 1);
    EXPECT_TRUE(info.reads.find(entt::type_id<ComponentA>()) != info.reads.end());
    EXPECT_TRUE(info.writes.find(entt::type_id<ComponentB>()) != info.writes.end());
    EXPECT_TRUE(info.parameters[0].dependencies[0].type == entt::type_id<ComponentA>());
    EXPECT_TRUE(info.parameters[0].dependencies[0].mode == AccessMode::Read);
    EXPECT_TRUE(info.parameters[0].IsOptional());
    EXPECT_TRUE(info.parameters[1].dependencies[0].type == entt::type_id<ComponentB>());
    EXPECT_TRUE(info.parameters[1].dependencies[0].mode == AccessMode::Write);
  }
  {
    auto info = GetSystemInfo(+[](entt::registry const&, entt::entity) {});
    EXPECT_TRUE(info.wildcard == AccessWildcard::Read);
  }
}

TEST(EcsSystem, InvokeGlobal) {
  entt::registry reg;

  reg.set<GlobalData>(GlobalData{1});

  auto e1 = reg.create();
  auto e2 = reg.create();
  auto e3 = reg.create();

  reg.emplace<ComponentA>(e1, 0);
  reg.emplace<ComponentB>(e1, 0);
  reg.emplace<TagA>(e1);

  reg.emplace<ComponentA>(e2, 0);
  reg.emplace<ComponentB>(e2, 0);

  reg.emplace<ComponentA>(e3, 1);
  reg.emplace<ComponentC>(e3, 1);
  reg.emplace<TagA>(e3);

  InvokeGlobal(&GlobalSystem, reg);

  EXPECT_EQ(reg.get<ComponentA>(e1).value, 1);
  EXPECT_EQ(reg.get<ComponentA>(e2).value, 1);
  EXPECT_EQ(reg.get<ComponentA>(e3).value, 1);
  EXPECT_EQ(reg.get<ComponentB>(e1).value, 2);
  EXPECT_EQ(reg.get<ComponentB>(e2).value, 2);

  InvokeGlobal(&GlobalSystem2, reg);

  EXPECT_EQ(reg.get<ComponentB>(e1).value, 1000);
  EXPECT_EQ(reg.get<ComponentB>(e2).value, 1000);
  EXPECT_EQ(reg.get<ComponentC>(e3).value, 1000);
}

TEST(EcsSystem, GlobalSystemsMetadata) {
  {
    auto info = GetSystemInfo(+[](View<ComponentA, ComponentB>, GlobalData const&) {});

    EXPECT_EQ(info.parameters[0].paramType, SystemParamType::View);
    EXPECT_EQ(info.parameters[0].dependencies[0].type, entt::type_id<ComponentA>());
    EXPECT_EQ(info.parameters[0].dependencies[1].type, entt::type_id<ComponentB>());
    EXPECT_TRUE(info.parameters[0].wildcard == AccessWildcard::None);

    EXPECT_TRUE(info.writes.find(entt::type_id<ComponentA>()) != info.writes.end());
    EXPECT_TRUE(info.writes.find(entt::type_id<ComponentB>()) != info.writes.end());
    EXPECT_TRUE(info.reads.find(entt::type_id<GlobalData>()) != info.reads.end());
  }
  {
    auto info = GetSystemInfo(
        +[](View<ComponentA const, ComponentB>, View<ComponentA const, ComponentC, TagA const>) {});

    EXPECT_EQ(info.parameters[0].paramType, SystemParamType::View);
    EXPECT_EQ(info.parameters[0].dependencies[0].type, entt::type_id<ComponentA>());
    EXPECT_EQ(info.parameters[0].dependencies[0].mode, AccessMode::Read);
    EXPECT_EQ(info.parameters[0].dependencies[1].type, entt::type_id<ComponentB>());
    EXPECT_EQ(info.parameters[0].dependencies[1].mode, AccessMode::Write);
    EXPECT_TRUE(info.parameters[0].wildcard == AccessWildcard::None);

    EXPECT_EQ(info.parameters[1].paramType, SystemParamType::View);
    EXPECT_EQ(info.parameters[1].dependencies[0].type, entt::type_id<ComponentA>());
    EXPECT_EQ(info.parameters[1].dependencies[0].mode, AccessMode::Read);
    EXPECT_EQ(info.parameters[1].dependencies[1].type, entt::type_id<ComponentC>());
    EXPECT_EQ(info.parameters[1].dependencies[1].mode, AccessMode::Write);
    EXPECT_EQ(info.parameters[1].dependencies[2].type, entt::type_id<TagA>());
    EXPECT_EQ(info.parameters[1].dependencies[2].mode, AccessMode::Read);
    EXPECT_TRUE(info.parameters[0].wildcard == AccessWildcard::None);

    EXPECT_TRUE(info.writes.find(entt::type_id<ComponentB>()) != info.writes.end());
    EXPECT_TRUE(info.writes.find(entt::type_id<ComponentC>()) != info.writes.end());
    EXPECT_TRUE(info.reads.find(entt::type_id<TagA>()) != info.reads.end());
    EXPECT_TRUE(info.reads.find(entt::type_id<ComponentA>()) != info.reads.end());
  }
}

// Helper to collect all entities into a std::vector
auto GetAllEntities(entt::registry const& reg) {
  std::vector<entt::entity> entities;
  reg.each([&](auto e) { entities.push_back(e); });
  return entities;
}

// Used to repeat the same test code with different invocation functions
enum struct MethodToTest {
  InvokeOnEntity,
  TryInvokeOnEntity,
  TryScheduleInvokeOnEntity,
  InvokeForEach,
  InvokeForEachGlobal,
  ParallelInvokeForEach,
  ParallelInvokeForEachGlobal,
  ScheduleInvokeForEach,
  ScheduleInvokeForEachGlobal,
};

// Helper to call to call a per-entity system for a list of entities
template <MethodToTest kMethod, typename SystemT, typename SubsetT, typename... ExternalT>
static void TestInvokeForEachLocal(
    SystemT system,
    entt::registry& reg,
    SubsetT const& entities,
    ExternalT... extParams) {
  if constexpr (kMethod == MethodToTest::InvokeForEach) {
    InvokeForEach(system, reg, entities, extParams...);
  } else if constexpr (kMethod == MethodToTest::ParallelInvokeForEach) {
    ParallelInvokeForEach("label", system, reg, entities, extParams...);
  } else if constexpr (kMethod == MethodToTest::ScheduleInvokeForEach) {
    TaskSemaphore sem;
    ScheduleInvokeForEach(sem, "label", system, reg, entities, extParams...);
    sem.Wait();
  } else {
    static_assert(std::is_void_v<SystemT>, "Unsupported MethodToTest");
  }
}

// Helper to call to call a per-entity system for each entity in the registry,
// using the specified invocation method
template <MethodToTest kMethod, typename SystemT, typename... ExternalT>
static void TestInvokeForEach(SystemT system, entt::registry& reg, ExternalT... extParams) {
  if constexpr (kMethod == MethodToTest::InvokeOnEntity) {
    auto invoker = InvokerImpl<entt::type_list<>, ExternalT...>{extParams...};
    for (auto e : GetAllEntities(reg)) {
      if (invoker.CanInvokeOnEntity(system, reg, e)) {
        InvokeOnEntity(system, reg, e, extParams...);
      }
    }
  } else if constexpr (kMethod == MethodToTest::TryInvokeOnEntity) {
    auto invoker = InvokerImpl<entt::type_list<>, ExternalT...>(extParams...);
    for (auto e : GetAllEntities(reg)) {
      bool wasCalled = TryInvokeOnEntity(system, reg, e, extParams...);
      EXPECT_EQ(invoker.CanInvokeOnEntity(system, reg, e), wasCalled);
    }
  } else if constexpr (kMethod == MethodToTest::TryScheduleInvokeOnEntity) {
    TaskSemaphore sem;
    TryScheduleInvokeOnEntity(sem, "label", system, reg, extParams...);
    sem.Wait();
  } else if constexpr (kMethod == MethodToTest::InvokeForEachGlobal) {
    InvokeForEachGlobal(system, reg, extParams...);
  } else if constexpr (kMethod == MethodToTest::ParallelInvokeForEachGlobal) {
    ParallelInvokeForEachGlobal("label", system, reg, extParams...);
  } else if constexpr (kMethod == MethodToTest::ScheduleInvokeForEachGlobal) {
    TaskSemaphore sem;
    ScheduleInvokeForEachGlobal(sem, "label", system, reg, extParams...);
    sem.Wait();
  } else {
    // It must be one of the methods that takes an entity list. Use all entities, to match the
    // behavior of the cases above.
    TestInvokeForEachLocal<kMethod>(system, reg, GetAllEntities(reg), extParams...);
  }
}

template <MethodToTest kMethod>
static void TestInvokeForEachLocal() {
  // This function tests per-entity systems that use an entity list.
  // It focuses on the affect of the list. It does not re-test all the argument combinations.
  entt::registry reg;

  // CtxGlobal with the sneaky ability to collect results. Don't actually do this.
  struct MutableResult {
    mutable std::atomic<int> count{0};
    mutable std::atomic<int> total{0};
    void Reset() {
      count.store(0);
      total.store(0);
    }
  };

  auto& result = reg.set<MutableResult>();
  constexpr int kNumEntities = 10;
  int totalValue = 0;
  std::vector<entt::entity> allEntities;
  allEntities.reserve(kNumEntities);
  for (int i = 1; i <= kNumEntities; ++i) {
    auto e = reg.create();
    auto& comp = reg.emplace<ComponentA>(e);
    comp.value = i;
    totalValue += i;
    allEntities.push_back(e);
  }

  // A system that adds all the component values
  auto sumFn = +[](ComponentA const& comp, CtxGlobal<MutableResult const> out) {
    out->count++;
    out->total += comp.value;
  };

  using List = std::vector<entt::entity>;

  // Empty
  {
    result.Reset();
    TestInvokeForEachLocal<kMethod>(sumFn, reg, List{});
    EXPECT_EQ(0, result.count.load());
    EXPECT_EQ(0, result.total.load());
  }

  // Just 1
  for (auto e : allEntities) {
    result.Reset();
    TestInvokeForEachLocal<kMethod>(sumFn, reg, List{e});
    EXPECT_EQ(1, result.count.load());
    EXPECT_EQ(reg.get<ComponentA const>(e).value, result.total.load());
  }

  // 2 at a time
  for (int i = 2; i < kNumEntities; i += 2) {
    int i0 = i - 2;
    int i1 = i; // i0 and i1 are not consecutive
    result.Reset();
    TestInvokeForEachLocal<kMethod>(sumFn, reg, List{allEntities[i0], allEntities[i1]});
    EXPECT_EQ(2, result.count.load());
    int expectedTotal = reg.get<ComponentA const>(allEntities[i0]).value +
        reg.get<ComponentA const>(allEntities[i1]).value;
    EXPECT_EQ(expectedTotal, result.total.load());
  }

  // All
  {
    result.Reset();
    TestInvokeForEachLocal<kMethod>(sumFn, reg, allEntities);
    EXPECT_EQ(kNumEntities, result.count.load());
    EXPECT_EQ(totalValue, result.total.load());
  }

  // Use a std::unordered_set, just to show we can
  {
    result.Reset();
    std::unordered_set<entt::entity> set;
    for (auto e : allEntities) {
      set.insert(e);
    }
    TestInvokeForEachLocal<kMethod>(sumFn, reg, set);
    EXPECT_EQ(kNumEntities, result.count.load());
    EXPECT_EQ(totalValue, result.total.load());
  }
}

template <MethodToTest kMethod>
static void TestInvokeForEach() {
  // This function tests multiple methods of invoking systems (TryInvokeOnEntity, InvokeForEach,
  // ParallelInvokeForEach, etc...). It also tests both the global and local overloads of these
  // methods. In all cases, the system will be invoked for each matching entity in the registry.
  // That way, we know what outcomes to expect.
  entt::registry reg;

  auto e1 = reg.create();
  auto e2 = reg.create();
  auto e3 = reg.create();
  auto e4 = reg.create();
  auto e5 = reg.create();

  reg.emplace<ComponentA>(e1, 0);
  reg.emplace<ComponentB>(e1, 0);

  reg.emplace<ComponentA>(e2, 0);
  reg.emplace<ComponentB>(e2, 0);

  reg.emplace<ComponentA>(e3, 0);
  reg.emplace<TagA>(e3);

  reg.emplace<ComponentB>(e4, 0);

  reg.emplace<TagA>(e5);

  reg.set<GlobalData>(10);

  // Reset all component values
  auto reset = [&](int newValue) {
    reg.view<ComponentA>().each([=](auto& comp) { comp.value = newValue; });
    reg.view<ComponentB>().each([=](auto& comp) { comp.value = newValue; });
  };

  // External param struct with a sneaky way to detect call count
  struct MyExternalData {
    int valueToAdd = 123;
    mutable std::atomic<int> callCount{0};
  };

  // Write ComponentA
  {
    reset(1);
    auto fn = +[](ComponentA& comp) { comp.value += 1; };
    TestInvokeForEach<kMethod>(fn, reg);
    EXPECT_EQ(reg.get<ComponentA>(e1).value, 2); // A += 1
    EXPECT_EQ(reg.get<ComponentA>(e2).value, 2); // A += 1
    EXPECT_EQ(reg.get<ComponentA>(e3).value, 2); // A += 1
  }

  // Read ComponentA, write ComponentB
  {
    reset(1);
    auto fn = +[](ComponentA const& compA, ComponentB& compB) { compB.value += compA.value; };
    TestInvokeForEach<kMethod>(fn, reg);
    EXPECT_EQ(reg.get<ComponentA>(e1).value, 1); // A is const
    EXPECT_EQ(reg.get<ComponentB>(e1).value, 2); // B += A
    EXPECT_EQ(reg.get<ComponentA>(e2).value, 1); // A is const
    EXPECT_EQ(reg.get<ComponentB>(e2).value, 2); // B += A
    EXPECT_EQ(reg.get<ComponentA>(e3).value, 1); // A is const
    EXPECT_EQ(reg.get<ComponentB>(e4).value, 1); // Skipped. No A.
  }

  // Optional read ComponentA, write ComponentB
  {
    reset(1);
    auto fn = +[](ComponentA const* compA, ComponentB& compB) {
      compB.value += compA ? compA->value : -1;
    };
    TestInvokeForEach<kMethod>(fn, reg);
    EXPECT_EQ(reg.get<ComponentB>(e1).value, 2); // B += A
    EXPECT_EQ(reg.get<ComponentB>(e2).value, 2); // B += A
    EXPECT_EQ(reg.get<ComponentB>(e4).value, 0); // No A, B -= 1
  }

  // Write ComponentA, optional read ComponentB
  {
    reset(1);
    auto fn = +[](ComponentA& compA, ComponentB* compB) {
      compA.value += compB ? compB->value : -1;
      if (compB) {
        compB->value += 2;
      }
    };
    TestInvokeForEach<kMethod>(fn, reg);
    EXPECT_EQ(reg.get<ComponentA>(e1).value, 2); // A += B
    EXPECT_EQ(reg.get<ComponentB>(e1).value, 3); // B += 2
    EXPECT_EQ(reg.get<ComponentA>(e2).value, 2); // A += B
    EXPECT_EQ(reg.get<ComponentB>(e2).value, 3); // B += 2
    EXPECT_EQ(reg.get<ComponentA>(e3).value, 0); // A -= 1
    EXPECT_EQ(reg.get<ComponentB>(e4).value, 1); // No A. Unchanged.
  }

  // Write ComponentA, optional TagA
  {
    reset(1);
    auto fn = +[](ComponentA& comp, OptionalTag<TagA> hasTagA) { comp.value += hasTagA ? 1 : -1; };
    TestInvokeForEach<kMethod>(fn, reg);
    EXPECT_EQ(reg.get<ComponentA>(e1).value, 0); // No TagA
    EXPECT_EQ(reg.get<ComponentA>(e2).value, 0); // No TagA
    EXPECT_EQ(reg.get<ComponentA>(e3).value, 2); // A += 1
  }

  // External dat, Required TagA, optional ComponentA
  {
    reset(1);
    auto fn = +[](MyExternalData const& data, RequiredTag<TagA>, ComponentA* compA) {
      data.callCount++;
      if (compA) {
        compA->value++;
      }
    };
    MyExternalData data;
    TestInvokeForEach<kMethod>(fn, reg, std::cref(data));
    EXPECT_EQ(reg.get<ComponentA>(e1).value, 1); // No TagA. Unchanged.
    EXPECT_EQ(reg.get<ComponentA>(e2).value, 1); // No TagA. Unchanged.
    EXPECT_EQ(reg.get<ComponentA>(e3).value, 2); // A += 1
    EXPECT_EQ(2, data.callCount.load()); // Was also called for e5
  }

  // Write ComponentA, exclude TagA
  {
    reset(1);
    auto fn = +[](ComponentA& compA, Excluded<TagA>) { compA.value++; };
    TestInvokeForEach<kMethod>(fn, reg);
    EXPECT_EQ(reg.get<ComponentA>(e1).value, 2); // A += 1
    EXPECT_EQ(reg.get<ComponentA>(e2).value, 2); // A += 1
    EXPECT_EQ(reg.get<ComponentA>(e3).value, 1); // TagA, no change
  }

  // External data passed by value
  {
    reset(0);
    auto fn = +[](int val1, int val2, ComponentA& comp) { comp.value += val1 + val2; };
    TestInvokeForEach<kMethod>(fn, reg, 123, 456);
    EXPECT_EQ(reg.get<ComponentA>(e1).value, 579); // A += val1 + val2
    EXPECT_EQ(reg.get<ComponentA>(e2).value, 579); // A += val1 + val2
    EXPECT_EQ(reg.get<ComponentA>(e3).value, 579); // A += val1 + val2
  }

  // External data passed by std::cref, writes ComponentA
  {
    reset(0);
    auto fn = +[](MyExternalData const& data, ComponentA& comp) {
      comp.value += data.valueToAdd;
      data.callCount++;
    };
    MyExternalData data;
    TestInvokeForEach<kMethod>(fn, reg, std::cref(data));
    EXPECT_EQ(reg.get<ComponentA>(e1).value, 123); // A += data.valueToAdd
    EXPECT_EQ(reg.get<ComponentA>(e2).value, 123); // A += data.valueToAdd
    EXPECT_EQ(reg.get<ComponentA>(e3).value, 123); // A += data.valueToAdd
    EXPECT_EQ(3, data.callCount.load()); // Proof that it was actually passed by cref, not copied
  }

  // View reads ComponentA, write ComponentB
  {
    reset(1);
    auto fn = +[](View<ComponentA const> view, ComponentB& compB) {
      view.each([&compB](ComponentA const& compA) { compB.value += compA.value; });
    };
    TestInvokeForEach<kMethod>(fn, reg);
    EXPECT_EQ(reg.get<ComponentB>(e4).value, 4); // B += count of A
  }

  // PartialRegistry reads ComponentA, write ComponentB
  {
    reset(1);
    auto fn = +[](PartialRegistry<ComponentA const> reg, ComponentB& compB) {
      reg.view<ComponentA const>().each(
          [&compB](ComponentA const& compA) { compB.value += compA.value; });
    };
    TestInvokeForEach<kMethod>(fn, reg);
    EXPECT_EQ(reg.get<ComponentB>(e4).value, 4); // B += count of A
  }

  // Writes componentA, read GlobalData, optional read GlobalData2
  {
    reset(0);
    auto fn = +[](ComponentA& compA,
                  CtxGlobal<GlobalData const> global,
                  OptionalCtxGlobal<GlobalData2 const> global2) {
      compA.value += global->valueToAdd;
      compA.value += global2 ? global2->valueToAdd : 0;
    };
    TestInvokeForEach<kMethod>(fn, reg);
    EXPECT_EQ(reg.get<ComponentA>(e1).value, 10); // A += global->valueToAdd
    EXPECT_EQ(reg.get<ComponentA>(e2).value, 10); // A += global->valueToAdd
    EXPECT_EQ(reg.get<ComponentA>(e3).value, 10); // A += global->valueToAdd
    reg.set<GlobalData2>(1); // Add GlobalData2 and rerun
    TestInvokeForEach<kMethod>(fn, reg);
    EXPECT_EQ(reg.get<ComponentA>(e1).value, 21); // A += global->valueToAdd + global2->valueToAdd
    EXPECT_EQ(reg.get<ComponentA>(e2).value, 21); // A += global->valueToAdd + global2->valueToAdd
    EXPECT_EQ(reg.get<ComponentA>(e3).value, 21); // A += global->valueToAdd + global2->valueToAdd
  }

  if constexpr (
      (kMethod == MethodToTest::InvokeForEach) ||
      (kMethod == MethodToTest::ParallelInvokeForEach) ||
      (kMethod == MethodToTest::ScheduleInvokeForEach)) {
    // These methods use an entity list, but all of the tests above use the list of all entities in
    // the registry. Now it is time to find out what happens if we use other lists.
    TestInvokeForEachLocal<kMethod>();
  }

  // The following systems should not compile because they break the rules for per-entity systems.
  // Uncomment them to try for yourself. Replace InvokeForEach with the other flavors and try them
  // too.
  //
  // clang-format off
  // InvokeForEachGlobal(+[](entt::registry&, ComponentA const&){}, reg); // full registry
  // InvokeForEachGlobal(+[](entt::registry const&, ComponentA const&){}, reg); // full registry
  // InvokeForEachGlobal(+[](View<ComponentA>, ComponentB const&){}, reg); // View write
  // InvokeForEachGlobal(+[](View<ComponentA const>, ComponentA&){}, reg); // View read + comp write
  // InvokeForEachGlobal(+[](PartialRegistry<ComponentA>, ComponentB const&){}, reg); // PartialRegistry write
  // InvokeForEachGlobal(+[](PartialRegistry<ComponentA const>, ComponentA&){}, reg); // PartialRegistry read + component write
  // InvokeForEachGlobal(+[](PartialRegistry<WildcardRead>, ComponentB const&){}, reg); // WildcardRead
  // InvokeForEachGlobal(+[](PartialRegistry<WildcardWrite>, ComponentB const&){}, reg); // WildcardWrite
  // InvokeForEachGlobal(+[](CtxGlobal<ComponentA>, ComponentB const&){}, reg); // CtxGlobal write
  // clang-format on
}

TEST(EcsSystem, InvokeOnEntity) {
  TestInvokeForEach<MethodToTest::InvokeOnEntity>();
}

TEST(EcsSystem, TryInvokeOnEntity) {
  TestInvokeForEach<MethodToTest::TryInvokeOnEntity>();
}

TEST(EcsSystem, InvokeForEach) {
  TestInvokeForEach<MethodToTest::InvokeForEach>();
  TestInvokeForEach<MethodToTest::InvokeForEachGlobal>();
}

TEST(EcsSystem, ParallelInvokeForEach) {
  TestInvokeForEach<MethodToTest::ParallelInvokeForEach>();
  TestInvokeForEach<MethodToTest::ParallelInvokeForEachGlobal>();
}

TEST(EcsSystem, ScheduleInvokeForEach) {
  TestInvokeForEach<MethodToTest::ScheduleInvokeForEach>();
  TestInvokeForEach<MethodToTest::ScheduleInvokeForEachGlobal>();
}

TEST(EcsSystem, SystemObjects) {
  auto IncrementASystem = +[](ComponentA& comp) { comp.value += 1; };
  System system1 = CreatePerEntitySystem(IncrementASystem, "IncrementASystem");
  System system2 = CreateSystem(&GlobalSystem2, "GlobalSystem2");

  entt::registry reg;
  auto e1 = reg.create();
  reg.emplace<ComponentA>(e1, 0);

  system1(reg);

  EXPECT_EQ(reg.get<ComponentA>(e1).value, 1);

  system2(reg);

  EXPECT_EQ(reg.get<ComponentA>(e1).value, 1);
}

TEST(EcsSystem, TestAccessTokens) {
  // If this test compiles, then it passes.
  // clang-format off
  static_assert(IsAccessTokenValid_v<entt::type_list<>, entt::type_list<>>);
  static_assert(IsAccessTokenValid_v<entt::type_list<>, entt::type_list<ComponentA>>);
  static_assert(!IsAccessTokenValid_v<entt::type_list<ComponentA>, entt::type_list<>>);
  static_assert(IsAccessTokenValid_v<entt::type_list<ComponentA>, entt::type_list<ComponentA>>);
  static_assert(IsAccessTokenValid_v<entt::type_list<ComponentA const>, entt::type_list<ComponentA>>);
  static_assert(IsAccessTokenValid_v<entt::type_list<ComponentA const, ComponentB const>, entt::type_list<ComponentA, ComponentB>>);
  static_assert(IsAccessTokenValid_v<entt::type_list<ComponentA, ComponentB const>, entt::type_list<ComponentA, ComponentB>>);
  static_assert(!IsAccessTokenValid_v<entt::type_list<ComponentA, ComponentB const>, entt::type_list<ComponentA const, ComponentB>>);
  static_assert(IsAccessTokenValid_v<entt::type_list<ComponentA const>, entt::type_list<WildcardRead>>);
  static_assert(!IsAccessTokenValid_v<entt::type_list<ComponentA>, entt::type_list<WildcardRead>>);
  static_assert(IsAccessTokenValid_v<entt::type_list<ComponentA const>, entt::type_list<WildcardWrite>>);
  static_assert(IsAccessTokenValid_v<entt::type_list<ComponentA>, entt::type_list<WildcardWrite>>);
  static_assert(IsAccessTokenValid_v<entt::type_list<WildcardRead>, entt::type_list<WildcardRead>>);
  static_assert(IsAccessTokenValid_v<entt::type_list<WildcardRead>, entt::type_list<WildcardWrite>>);
  static_assert(IsAccessTokenValid_v<entt::type_list<WildcardWrite>, entt::type_list<WildcardWrite>>);
  // clang-format on
}

TEST(EcsSystem, PartialRegistry) {
  System system = CreateSystem(&GlobalSystemWithPartialRegistry, "GlobalSystemWithPartialRegistry");

  entt::registry reg;

  auto e1 = reg.create();
  auto e2 = reg.create();
  auto e3 = reg.create();

  reg.emplace<ComponentA>(e1, 0);
  reg.emplace<ComponentB>(e1, 0);
  reg.emplace<ComponentC>(e1, 0);

  reg.emplace<ComponentA>(e2, 0);
  reg.emplace<ComponentB>(e2, 0);
  reg.emplace<TagA>(e2);

  reg.emplace<ComponentA>(e3, 0);

  system(reg);

  EXPECT_TRUE(!reg.all_of<ComponentA>(e1));
  EXPECT_EQ(reg.get<ComponentB>(e1).value, 0);

  EXPECT_EQ(reg.get<ComponentB>(e2).value, 1);
  EXPECT_TRUE(reg.all_of<ComponentB>(e3));
  EXPECT_EQ(reg.get<ComponentB>(e3).value, 1000);

  EXPECT_EQ(reg.get<ComponentA>(e2).value, 1);
  EXPECT_EQ(reg.get<ComponentA>(e3).value, 1);
}

TEST(EcsSystem, PartialRegistryCasting) {
  entt::registry reg;

  // Compile-time test
  PartialRegistry<WildcardWrite> preg(reg);
  PartialRegistry<ComponentA, ComponentB, ComponentC> preg2(preg);
  PartialRegistry<ComponentA const, ComponentB const, ComponentC const> preg3(preg2);
  PartialRegistry<ComponentA const> preg4(preg3);
  PartialRegistry<ComponentA> preg5(preg2);
  PartialRegistry<WildcardRead> preg6(preg);
  PartialRegistry<ComponentA const> preg7(preg);

  entt::registry const& creg = reg;
  PartialRegistry<WildcardRead> preg8(creg);
  PartialRegistry<ComponentA const, ComponentB const> preg9(creg);
}

TEST(EcsSystem, PartialRegistryCallFunctions) {
  entt::registry reg;

  auto e = reg.create();

  PartialRegistry<WildcardWrite> preg_w(reg);

  preg_w.emplace<ComponentB>(e);
  preg_w.emplace_or_replace<ComponentB>(e);
  preg_w.erase<ComponentB>(e);
  preg_w.clear<ComponentB>();
  preg_w.destroy(e);

  e = reg.create();
  preg_w.get_or_emplace<ComponentB>(e).value = 1;
  EXPECT_EQ(preg_w.get<ComponentB>(e).value, 1);
  EXPECT_EQ(preg_w.try_get<ComponentB>(e)->value, 1);
  EXPECT_TRUE(preg_w.any_of<ComponentB>(e));
  EXPECT_TRUE(preg_w.all_of<ComponentB>(e));
  preg_w.patch<ComponentB>(e, [](ComponentB& b) { b.value = 2; });
  EXPECT_EQ(preg_w.get<ComponentB>(e).value, 2);

  preg_w.set<GlobalData>().valueToAdd = 1;
  preg_w.ctx_or_set<GlobalData>().valueToAdd = 1;
  EXPECT_EQ(preg_w.ctx<GlobalData>().valueToAdd, 1);
  EXPECT_EQ(preg_w.try_ctx<GlobalData>()->valueToAdd, 1);
  preg_w.unset<GlobalData>();

  preg_w.set<GlobalData>().valueToAdd = 1;
  auto view = preg_w.view<ComponentB>();
  EXPECT_EQ(view.size(), 1);

  PartialRegistry<WildcardRead> preg_r(preg_w);

  EXPECT_FALSE(preg_r.all_of<ComponentA>(e));
  EXPECT_FALSE(preg_r.any_of<ComponentA>(e));
  EXPECT_EQ(preg_r.ctx<GlobalData const>().valueToAdd, 1);
  EXPECT_EQ(preg_r.try_ctx<GlobalData const>()->valueToAdd, 1);
  EXPECT_EQ(preg_r.get<ComponentB const>(e).value, 2);
  EXPECT_EQ(preg_r.try_get<ComponentB const>(e)->value, 2);

  auto view2 = preg_r.view<ComponentB const>();
  EXPECT_EQ(view2.size(), 1);
}

// NOLINTEND(clang-diagnostic-unused-parameter)

} // namespace
