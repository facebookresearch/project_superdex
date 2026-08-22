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

#include <mochi_core/elements/triangular/basis_functions_evaluated.h>
#include <mochi_core/elements/triangular/finite_element.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/local_to_global_map.h>
#include <mochi_core/utils/math_utils.h>

#include <gtest/gtest.h>

#include <iterator>
#include <vector>

using namespace mochi;

/**
Test the correct evaluation of basis functions over a triangle
*/
TEST(Triangular, BasisFunctions) {
  constexpr int kPolyOrder = 1;
  using Basis = triangular::BarycentricBasisTriangular<kPolyOrder>;

  // Create the basis functions over the parametric domain
  constexpr Basis b = {};

  // constants
  static_assert(3 == b.kVertexDof);
  static_assert(1 == b.kPolyOrder);
  static_assert(3 == b.kNumDofs);
  static_assert(0 == b.kEdgeDof);
  static_assert(0 == b.kFaceDof);

  // indices
  static_assert(3 == b.kIndices.size());
  static_assert(Int3{0, 0, 1} == b.kIndices[0]);
  static_assert(Int3{1, 0, 0} == b.kIndices[1]);
  static_assert(Int3{0, 1, 0} == b.kIndices[2]);

  // dof_nodes
  static_assert(3 == b.kDofNodes.size());
  static_assert(Real3{0_r, 0_r, 1_r} == b.kDofNodes[0]);
  static_assert(Real3{1_r, 0_r, 0_r} == b.kDofNodes[1]);
  static_assert(Real3{0_r, 1_r, 0_r} == b.kDofNodes[2]);

  // one_dim_knot
  static_assert(2 == b.kOneDimKnot.size());
  static_assert(0_r == b.kOneDimKnot[0]);
  static_assert(1_r == b.kOneDimKnot[1]);

  // GetTValue(index, y)
  static_assert(1.0_r == b.GetTValue(0, 0.0_r));
  static_assert(1.0_r == b.GetTValue(0, 0.1_r));
  static_assert(1.0_r == b.GetTValue(0, 0.5_r));
  static_assert(1.0_r == b.GetTValue(0, 0.9_r));
  static_assert(1.0_r == b.GetTValue(0, 1.0_r));
  static_assert(0.0_r == b.GetTValue(1, 0.0_r));
  static_assert(0.1_r == b.GetTValue(1, 0.1_r));
  static_assert(0.5_r == b.GetTValue(1, 0.5_r));
  static_assert(0.9_r == b.GetTValue(1, 0.9_r));
  static_assert(1.0_r == b.GetTValue(1, 1.0_r));

  // GetDtValue(index, y)
  static_assert(0.0_r == b.GetDtValue(0, 0.0_r));
  static_assert(0.0_r == b.GetDtValue(0, 0.1_r));
  static_assert(0.0_r == b.GetDtValue(0, 0.5_r));
  static_assert(0.0_r == b.GetDtValue(0, 0.9_r));
  static_assert(0.0_r == b.GetDtValue(0, 1.0_r));
  static_assert(1.0_r == b.GetDtValue(1, 0.0_r));
  static_assert(1.0_r == b.GetDtValue(1, 0.1_r));
  static_assert(1.0_r == b.GetDtValue(1, 0.5_r));
  static_assert(1.0_r == b.GetDtValue(1, 0.9_r));
  static_assert(1.0_r == b.GetDtValue(1, 1.0_r));

  // GetValue(baseIndex, x)
  {
    // Just to prove that GetValue is a constexpr
    static_assert(NearEqual(0.7_r, b.GetValue(0, Real2{0.1_r, 0.2_r})));
    static_assert(NearEqual(0.3_r, b.GetValue(1, Real2{0.3_r, 0.4_r})));

    // Expected values generated using pyfea as follows:
    constexpr double kExpectedValues[3][9][9] = {
        {
            // baseIndex = 0
            {1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0},
            {0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0, -0.125},
            {0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0, -0.125, -0.25},
            {0.625, 0.5, 0.375, 0.25, 0.125, 0.0, -0.125, -0.25, -0.375},
            {0.5, 0.375, 0.25, 0.125, 0.0, -0.125, -0.25, -0.375, -0.5},
            {0.375, 0.25, 0.125, 0.0, -0.125, -0.25, -0.375, -0.5, -0.625},
            {0.25, 0.125, 0.0, -0.125, -0.25, -0.375, -0.5, -0.625, -0.75},
            {0.125, 0.0, -0.125, -0.25, -0.375, -0.5, -0.625, -0.75, -0.875},
            {0.0, -0.125, -0.25, -0.375, -0.5, -0.625, -0.75, -0.875, -1.0},
        },
        {
            // baseIndex = 1
            {0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0},
            {0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0},
            {0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0},
            {0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0},
            {0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0},
            {0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0},
            {0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0},
            {0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0},
            {0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0},
        },
        {
            // baseIndex = 2
            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125},
            {0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25},
            {0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375},
            {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
            {0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625},
            {0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75},
            {0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875},
            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        },
    };
    for (int baseIndex = 0; baseIndex < 2; ++baseIndex) {
      for (int iy = 0; iy < 9; ++iy) {
        real const y = static_cast<real>(iy) / 8_r; // range [0.0, 1.0]
        for (int ix = 0; ix < 9; ++ix) {
          real const x = static_cast<real>(ix) / 8_r; // range [0.0, 1.0]
          real const expected = static_cast<real>(kExpectedValues[baseIndex][iy][ix]);
          real const actual = b.GetValue(baseIndex, Real2{x, y});
          EXPECT_NEAR(expected, actual, 1e-6_r);
        }
      }
    }
  }

  // GetDValue(baseIndex, x)
  {
    // Just to prove that GetValue is a constexpr
    static_assert(NearEqual(Real2{-1._r, -1._r}, b.GetDValue(0, Real2{0.1_r, 0.2_r})));
    static_assert(NearEqual(Real2{1.0_r, 0.0_r}, b.GetDValue(1, Real2{0.3_r, 0.4_r})));

    // Test several more inputs. The result should only depend on the baseIndex.
    for (int baseIndex = 0; baseIndex < 2; ++baseIndex) {
      for (int iy = 0; iy < 9; ++iy) {
        real const y = static_cast<real>(iy) / 8_r; // range [0.0, 1.0]
        for (int ix = 0; ix < 9; ++ix) {
          real const x = static_cast<real>(ix) / 8_r; // range [0.0, 1.0]
          Real2 const actual = b.GetDValue(baseIndex, Real2{x, y});
          if (baseIndex == 0) {
            EXPECT_TRUE(NearEqual(Real2{-1_r, -1_r}, actual));
          } else {
            EXPECT_TRUE(NearEqual(Real2{1_r, 0_r}, actual));
          }
        }
      }
    }
  }
}

TEST(Triangular, SimplexQuadrature) {
  constexpr int kPolyOrder = 1;
  using QuadType = mochi::triangular::TriangleQuadrature<kPolyOrder>;
  static_assert(1 == std::size(QuadType::points));
  static_assert(NearEqual(Real2{0.333333_r, 0.333333_r}, QuadType::points[0]));
  static_assert(1 == std::size(QuadType::weights));
  static_assert(NearEqual(0.5_r, QuadType::weights[0]));
}

TEST(Triangular, BasisFunctionsEvaluated) {
  constexpr int kPolyOrder = 1;
  using BasisFunctionsEvaluatedType = triangular::BasisFunctionsEvaluatedTriangular<kPolyOrder>;

  // constats
  static_assert(1 == BasisFunctionsEvaluatedType::kPolyOrder);
  static_assert(2 == BasisFunctionsEvaluatedType::kSpaceDimParam);
  static_assert(3 == BasisFunctionsEvaluatedType::kNumDofs);
  static_assert(1 == BasisFunctionsEvaluatedType::kNumQuadPoints);

  // basisEvaluated
  static_assert(2 == BasisFunctionsEvaluatedType::BasisEvaluatedType::num_dims);
  static_assert(1 == BasisFunctionsEvaluatedType::BasisEvaluatedType::dims[0]);
  static_assert(3 == BasisFunctionsEvaluatedType::BasisEvaluatedType::dims[1]);
  constexpr NdArray<real, 1, 3> kExpectedBasisEvaluated = {
      Real3{0.33333333_r, 0.33333333_r, 0.33333333_r}};
  static_assert(NearEqual(kExpectedBasisEvaluated, BasisFunctionsEvaluatedType::kBasisEvaluated));

  // dBasisEvaluated
  static_assert(3 == BasisFunctionsEvaluatedType::DBasisEvaluatedType::num_dims);
  static_assert(1 == BasisFunctionsEvaluatedType::DBasisEvaluatedType::dims[0]);
  static_assert(3 == BasisFunctionsEvaluatedType::DBasisEvaluatedType::dims[1]);
  static_assert(2 == BasisFunctionsEvaluatedType::DBasisEvaluatedType::dims[2]);
  static_assert(Real2{-1_r, -1_r} == BasisFunctionsEvaluatedType::kDBasisEvaluated[0][0]);
  static_assert(Real2{+1_r, +0_r} == BasisFunctionsEvaluatedType::kDBasisEvaluated[0][1]);
  static_assert(Real2{+0_r, +1_r} == BasisFunctionsEvaluatedType::kDBasisEvaluated[0][2]);
}

TEST(Triangular, LocalToGlobalMap) {
  // A solid unit cube with one corner at (0,0,0)
  //
  //         6 ------- 7
  //       / |       / |
  //      /  |      /  |
  //     2 ------- 3   |
  //     |   4 ----|-- 5
  //     |  /      |  /
  //     | /       | /
  //     0 ------- 1
  //
  std::vector<Real3> const coordinates = {
      Real3{0.0_r, 0.0_r, 0.0_r}, // 0
      Real3{1.0_r, 0.0_r, 0.0_r}, // 1
      Real3{0.0_r, 1.0_r, 0.0_r}, // 2
      Real3{1.0_r, 1.0_r, 0.0_r}, // 3
      Real3{0.0_r, 0.0_r, 1.0_r}, // 4
      Real3{1.0_r, 0.0_r, 1.0_r}, // 5
      Real3{0.0_r, 1.0_r, 1.0_r}, // 6
      Real3{1.0_r, 1.0_r, 1.0_r}, // 7
  };
  std::vector<Int3> const connectivity = {
      Int3{1, 3, 5}, // +x face
      Int3{3, 7, 5},
      Int3{5, 7, 4}, // +z face
      Int3{7, 6, 4},
      Int3{4, 6, 0}, // -x face
      Int3{6, 2, 0},
      Int3{0, 2, 1}, // -z face
      Int3{2, 3, 1},
      Int3{2, 6, 3}, // +y face
      Int3{6, 7, 3},
      Int3{4, 0, 5}, // -y face
      Int3{0, 1, 5},
  };

  int const numNodes = isize(coordinates);
  int const numElements = isize(connectivity);

  TriangularMesh mesh(coordinates, connectivity);

  triangular::BarycentricBasisTriangular basis = {};
  Local2GlobalMap map;
  map.InitializeFromMeshAndBasis(&mesh, basis, 3);

  EXPECT_EQ(numElements * 9, map.GetNumIndices());
  EXPECT_EQ(numElements, map.GetNumElements());
  EXPECT_EQ(0, map.GetGlobalRange().Min());
  EXPECT_EQ(numNodes * 3, map.GetGlobalRange().Max() + 1);

  for (int e = 0; e < numElements; ++e) {
    EXPECT_EQ(9, map.GetElementSizes()[e]);
    EXPECT_EQ(9, map.GetElementSize(e));
    EXPECT_EQ(e * 9, map.GetElementOffsets()[e]);
    EXPECT_EQ(e * 9, map.GetElementOffset(e));
  }

  // Evaluated
  std::vector<int> expectedIndices = {
      3,  4,  5,  9,  10, 11, 15, 16, 17, // element 0
      9,  10, 11, 21, 22, 23, 15, 16, 17, // element 1
      15, 16, 17, 21, 22, 23, 12, 13, 14, // element 2
      21, 22, 23, 18, 19, 20, 12, 13, 14, // element 3
      12, 13, 14, 18, 19, 20, 0,  1,  2, // element 4
      18, 19, 20, 6,  7,  8,  0,  1,  2, // element 5
      0,  1,  2,  6,  7,  8,  3,  4,  5, // element 6
      6,  7,  8,  9,  10, 11, 3,  4,  5, // element 7
      6,  7,  8,  18, 19, 20, 9,  10, 11, // element 8
      18, 19, 20, 21, 22, 23, 9,  10, 11, // element 9
      12, 13, 14, 0,  1,  2,  15, 16, 17, // element 10
      0,  1,  2,  3,  4,  5,  15, 16, 17, // element 11
  };
  EXPECT_EQ(numElements * 9, map.GetGlobalIndices().size());
  EXPECT_SPAN_EQ(Span<int const>{expectedIndices}, map.GetGlobalIndices());
  for (int e = 0; e < numElements; ++e) {
    EXPECT_SPAN_EQ(Span<int const>(&expectedIndices[e * 9], 9), map.GetGlobalIndices(e));
    EXPECT_SPAN_EQ(Span<int const>(&expectedIndices[e * 9], 9), map.GetGlobalIndices(e));
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(expectedIndices[e * 9 + i], map.GetGlobalIndex(e, i));
    }
  }

  // Index range per element
  Interval<int> constexpr kExpectedRanges[] = {
      Interval<int>{3, 18}, // element 0
      Interval<int>{9, 24}, // element 1
      Interval<int>{12, 24}, // element 2
      Interval<int>{12, 24}, // element 3
      Interval<int>{0, 21}, // element 4
      Interval<int>{0, 21}, // element 5
      Interval<int>{0, 9}, // element 6
      Interval<int>{3, 12}, // element 7
      Interval<int>{6, 21}, // element 8
      Interval<int>{9, 24}, // element 9
      Interval<int>{0, 18}, // element 10
      Interval<int>{0, 18}, // element 11
  };
  for (int e = 0; e < numElements; ++e) {
    EXPECT_EQ(kExpectedRanges[e].Min(), map.GetGlobalRange(e).Min());
    EXPECT_EQ(kExpectedRanges[e].Max(), map.GetGlobalRange(e).Max());
  }
}

template <int kPolyOrder, int kNumQuadPoints>
void TestPk2D() {
  auto&& [coordinates, connectivity] = test::CreateMinimalTriMeshUnitCube();

  int numElements = isize(connectivity);
  TriangularMesh mesh(coordinates, connectivity);

  constexpr real kExpectedVolume = 1.0_r;
  constexpr real kExpectedArea = 6.0_r;
  using ElementT = triangular::Pk2DElement<kPolyOrder, kNumQuadPoints>;
  std::vector<ElementT> elements;
  elements.reserve(numElements);
  for (int e = 0; e < numElements; e++) {
    elements.emplace_back(e, mesh.GetNodeCoordinates(), mesh.GetElementConnectivity());
  }

  // Evaluate Surface Area
  real area = 0_r;
  for (int e = 0; e < numElements; e++) {
    for (int q = 0; q < elements[e].kNumQuadPoints; q++) {
      area += elements[e].quadWeights[q];
    }
  }
  EXPECT_NEAR(area, kExpectedArea, 1.e-4 * kExpectedArea);

  // Evaluate Volume
  real volume = 0_r;
  constexpr int kSpaceDim = 3;
  for (int e = 0; e < numElements; e++) {
    for (int q = 0; q < elements[e].kNumQuadPoints; q++) {
      volume +=
          Dot(elements[e].mapEvaluated[q], elements[e].normals[q]) * elements[e].quadWeights[q];
    }
  }
  volume /= kSpaceDim;
  EXPECT_NEAR(volume, kExpectedVolume, 1.e-4 * kExpectedVolume);

  // Evaluate Normals Flux
  Real3 flux = {};
  for (int e = 0; e < numElements; e++) {
    for (int q = 0; q < elements[e].kNumQuadPoints; q++) {
      flux += elements[e].normals[q] * elements[e].quadWeights[q];
    }
  }
  EXPECT_NEAR(Norm(flux), 0_r, 1.e-4);
}

TEST(Triangular, Pk2DElement) {
  TestPk2D<1, 1>();
  TestPk2D<1, 3>();
  TestPk2D<1, 6>();
}
