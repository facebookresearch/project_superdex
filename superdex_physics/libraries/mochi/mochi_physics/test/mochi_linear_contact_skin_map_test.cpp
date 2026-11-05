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

#include <mochi_physics/src/mochi_contact.h>
#include <mochi_physics/src/mochi_linear_contact_skin.h>

#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/test/mochi_test_helpers.h>

#include <gtest/gtest.h>

#include <array>
#include <initializer_list>
#include <utility>

using namespace mochi;

namespace {

using Influence = std::pair<int, real>;

CContactSkinningData MakeSkinningData(
    std::array<std::initializer_list<Influence>, 4> const& influences,
    int numPhysicsNodes) {
  DynamicArray<int> rowPointers;
  DynamicArray<int> columnIndices;
  DynamicArray<Real3> values;
  rowPointers.push_back(0);
  for (auto const& row : influences) {
    for (auto const& [physicsNode, weight] : row) {
      for (int component = 0; component < kSpaceDim3; ++component) {
        columnIndices.push_back(kSpaceDim3 * physicsNode + component);
        Real3 value{};
        value[component] = weight;
        values.push_back(value);
      }
    }
    rowPointers.push_back(isize(columnIndices));
  }

  CContactSkinningData skinning;
  skinning.jacobian = SparseMatrix<Real3>{
      kSpaceDim3 * numPhysicsNodes,
      std::move(rowPointers),
      std::move(columnIndices),
      std::move(values)};
  return skinning;
}

struct LinearMapFixture {
  std::array<Real3, 4> coordinates = {
      Real3{0_r, 0_r, 0_r},
      Real3{1_r, 0_r, 0_r},
      Real3{0_r, 1_r, 0_r},
      Real3{1_r, 1_r, 0_r}};
  std::array<Int3, 2> connectivity = {Int3{0, 1, 2}, Int3{2, 1, 3}};
  TriangularMesh mesh{MakeConstSpan(coordinates), MakeConstSpan(connectivity)};

  static CContactSkinningData MakeValidSkinning() {
    return MakeSkinningData(
        {{{Influence{0, 0.5_r}, Influence{1, 0.5_r}},
          {Influence{1, 1_r}},
          {Influence{2, 1_r}},
          {Influence{3, 0.25_r}, Influence{4, 0.75_r}}}},
        /*numPhysicsNodes=*/5);
  }

  CLinearContactSkinElementMap Build(ActorBoundaryElementType elementType) const {
    auto surfaceDisc = CFemSurfaceDiscretization::Create(elementType, mesh);
    auto const skinning = MakeValidSkinning();
    CLinearContactSkinElementMap map;
    linear_contact_skin::InitializeElementMap(surfaceDisc, skinning, map);
    return map;
  }
};

void ExpectWeights(Span<Real3 const> actual, std::initializer_list<Real3> expected) {
  ASSERT_EQ(expected.size(), actual.size());
  int i = 0;
  for (Real3 const& value : expected) {
    EXPECT_NEAR_EQ(value, actual[i]);
    ++i;
  }
}

void ExpectGraphRow(Graph<int, int> const& graph, int row, std::initializer_list<int> expected) {
  Span<int const> const expectedSpan{expected.begin(), expected.size()};
  EXPECT_SPAN_EQ(expectedSpan, graph[row]);
}

} // namespace

TEST(CLinearContactSkinElementMap, BuildsOverlappingVariableWidthMapsForSupportedQuadrature) {
  LinearMapFixture fixture;
  for (auto const& [elementType, samplesPerElement] :
       {std::pair{ActorBoundaryElementType::P1Q1, 1},
        std::pair{ActorBoundaryElementType::P1Q3, 3},
        std::pair{ActorBoundaryElementType::P1Q6, 6}}) {
    CLinearContactSkinElementMap const map = fixture.Build(elementType);

    EXPECT_EQ(samplesPerElement, map.samplesPerElement);
    EXPECT_SPAN_EQ(MakeConstSpan(std::array{0, 3, 7}), map.weights.Pointers());
    EXPECT_EQ(5, map.weights.Cols());
    EXPECT_SPAN_EQ(MakeConstSpan(std::array{0, 1, 2}), map.PhysicsNodeIndices(0));
    EXPECT_SPAN_EQ(MakeConstSpan(std::array{1, 2, 3, 4}), map.PhysicsNodeIndices(1));
    ExpectWeights(
        map.SkinNodeWeights(0),
        {Real3{0.5_r, 0_r, 0_r}, Real3{0.5_r, 1_r, 0_r}, Real3{0_r, 0_r, 1_r}});
    ExpectWeights(
        map.SkinNodeWeights(1),
        {Real3{0_r, 1_r, 0_r},
         Real3{1_r, 0_r, 0_r},
         Real3{0_r, 0_r, 0.25_r},
         Real3{0_r, 0_r, 0.75_r}});
    EXPECT_DOUBLE_EQ(10.5, map.AverageDofsPerElement());
  }
}

TEST(CLinearContactSkinElementMap, AppliesDofOffsetOnlyWhenBuildingScalarAndBsr3Graphs) {
  LinearMapFixture fixture;
  CLinearContactSkinElementMap map = fixture.Build(ActorBoundaryElementType::P1Q3);
  EXPECT_SPAN_EQ(MakeConstSpan(std::array{0, 1, 2}), map.PhysicsNodeIndices(0));

  entt::registry reg;
  entt::entity const actor = reg.create();
  reg.emplace<CLinearContactSkinElementMap>(actor, std::move(map));
  auto& dofOffset = reg.emplace<CDofOffset>(actor);
  dofOffset.dofsOffset = 9;
  auto& collisions = reg.emplace<CActiveCollisions</*kIsSync*/ false, TimeStep::Current>>(actor, 1);
  entt::entity const collider = entt::null;
  collisions.SetUp(MakeSingletonConstSpan(collider));

  auto const expectGraphs = [&] {
    Graph<int, int> const scalar =
        MakeContactGraph<1, /*kIsSync*/ false>(reg, MakeSingletonConstSpan(actor));
    EXPECT_EQ(24, scalar.size());
    for (int row = 0; row < 9; ++row) {
      EXPECT_TRUE(scalar[row].empty());
    }
    for (int row = 9; row < 12; ++row) {
      ExpectGraphRow(scalar, row, {9, 10, 11, 12, 13, 14, 15, 16, 17});
    }
    for (int row = 12; row < 18; ++row) {
      ExpectGraphRow(scalar, row, {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    }
    for (int row = 18; row < 24; ++row) {
      ExpectGraphRow(scalar, row, {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    }

    Graph<int, int> const bsr3 =
        MakeContactGraph<3, /*kIsSync*/ false>(reg, MakeSingletonConstSpan(actor));
    EXPECT_EQ(8, bsr3.size());
    for (int row = 0; row < 3; ++row) {
      EXPECT_TRUE(bsr3[row].empty());
    }
    ExpectGraphRow(bsr3, 3, {3, 4, 5});
    ExpectGraphRow(bsr3, 4, {3, 4, 5, 6, 7});
    ExpectGraphRow(bsr3, 5, {3, 4, 5, 6, 7});
    ExpectGraphRow(bsr3, 6, {4, 5, 6, 7});
    ExpectGraphRow(bsr3, 7, {4, 5, 6, 7});
  };

  // One contact per element.
  collisions.front().collisionResult.sampleIndices = {0, 3};
  expectGraphs();

  // With three samples per element, this repeats element 0 in adjacent contacts, which collapse
  // into one row, and again after element 1. The coupled DoFs must not change.
  collisions.front().collisionResult.sampleIndices = {0, 1, 2, 3, 0};
  expectGraphs();
}

TEST(LinearContactSkin, SortContactsBySample) {
  ContactDetectionResult query;
  query.sampleIndices = {7, 4, 2, 4, 9, 1};
  // Contacts 1 to 4 are out of sample order and contacts 1 and 3 share a sample, so contact order
  // breaks the tie. Contacts 2 to 4 are already in order.
  DynamicArray<int> const unsorted = linear_contact_skin::details::SortContactsBySample(
      Interval<int>{1, 5}, query, GetDefaultAllocator());
  DynamicArray<int> const sorted = linear_contact_skin::details::SortContactsBySample(
      Interval<int>{2, 5}, query, GetDefaultAllocator());
  EXPECT_SPAN_EQ(MakeConstSpan(std::array{2, 1, 3, 4}), MakeConstSpan(unsorted));
  EXPECT_SPAN_EQ(MakeConstSpan(std::array{2, 3, 4}), MakeConstSpan(sorted));
}
