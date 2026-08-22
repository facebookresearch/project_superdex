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

#include <mochi_core/elements/tetrahedral/basis_functions.h>
#include <mochi_core/elements/tetrahedral/basis_functions_evaluated.h>
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/elements/tetrahedral/finite_element_trace.h>
#include <mochi_core/elements/tetrahedral/simplex_quadrature.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/local_to_global_map.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/rand_utils.h>

#include <gtest/gtest.h>

#include <vector>

using namespace mochi;

/**
  Test the correct evaluation of basis functions over a tetrahedron.
*/
TEST(Tetrahedral, BasisFunctions) {
  constexpr int kPolyOrder = 1;
  using Basis = tetrahedral::BarycentricBasisTetrahedra<kPolyOrder>;

  // Create the basis functions over the parametric domain
  constexpr Basis b = {};

  // constants
  static_assert(4 == b.kVertexDof);
  static_assert(1 == b.kPolyOrder);
  static_assert(4 == b.kNumDofs);
  static_assert(0 == b.kEdgeDof);
  static_assert(0 == b.kFaceDof);

  // indices
  static_assert(4 == b.kIndices.size());
  static_assert(Int4{0, 0, 0, 1} == b.kIndices[0]);
  static_assert(Int4{1, 0, 0, 0} == b.kIndices[1]);
  static_assert(Int4{0, 1, 0, 0} == b.kIndices[2]);
  static_assert(Int4{0, 0, 1, 0} == b.kIndices[3]);

  // dof_nodes
  static_assert(4 == b.kDofNodes.size());
  static_assert(Real4{0_r, 0_r, 0_r, 1_r} == b.kDofNodes[0]);
  static_assert(Real4{1_r, 0_r, 0_r, 0_r} == b.kDofNodes[1]);
  static_assert(Real4{0_r, 1_r, 0_r, 0_r} == b.kDofNodes[2]);
  static_assert(Real4{0_r, 0_r, 1_r, 0_r} == b.kDofNodes[3]);

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
    static_assert(NearEqual(0.4_r, b.GetValue(0, Real3{0.1_r, 0.2_r, 0.3_r})));
    static_assert(NearEqual(0.4_r, b.GetValue(1, Real3{0.4_r, 0.5_r, 0.6_r})));

    // Expected values generated using pyfea as follows:
    constexpr double kExpectedValues[4][5][5][5] = {
        {
            {
                {1.0, 0.75, 0.5, 0.25, 0.0},
                {0.75, 0.5, 0.25, 0.0, -0.25},
                {0.5, 0.25, 0.0, -0.25, -0.5},
                {0.25, 0.0, -0.25, -0.5, -0.75},
                {0.0, -0.25, -0.5, -0.75, -1.0},
            },
            {
                {0.75, 0.5, 0.25, 0.0, -0.25},
                {0.5, 0.25, 0.0, -0.25, -0.5},
                {0.25, 0.0, -0.25, -0.5, -0.75},
                {0.0, -0.25, -0.5, -0.75, -1.0},
                {-0.25, -0.5, -0.75, -1.0, -1.25},
            },
            {
                {0.5, 0.25, 0.0, -0.25, -0.5},
                {0.25, 0.0, -0.25, -0.5, -0.75},
                {0.0, -0.25, -0.5, -0.75, -1.0},
                {-0.25, -0.5, -0.75, -1.0, -1.25},
                {-0.5, -0.75, -1.0, -1.25, -1.5},
            },
            {
                {0.25, 0.0, -0.25, -0.5, -0.75},
                {0.0, -0.25, -0.5, -0.75, -1.0},
                {-0.25, -0.5, -0.75, -1.0, -1.25},
                {-0.5, -0.75, -1.0, -1.25, -1.5},
                {-0.75, -1.0, -1.25, -1.5, -1.75},
            },
            {
                {0.0, -0.25, -0.5, -0.75, -1.0},
                {-0.25, -0.5, -0.75, -1.0, -1.25},
                {-0.5, -0.75, -1.0, -1.25, -1.5},
                {-0.75, -1.0, -1.25, -1.5, -1.75},
                {-1.0, -1.25, -1.5, -1.75, -2.0},
            },
        },
        {
            {
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
            },
            {
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
            },
            {
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
            },
            {
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
            },
            {
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
                {0.0, 0.25, 0.5, 0.75, 1.0},
            },
        },
        {
            {
                {0.0, 0.0, 0.0, 0.0, 0.0},
                {0.25, 0.25, 0.25, 0.25, 0.25},
                {0.5, 0.5, 0.5, 0.5, 0.5},
                {0.75, 0.75, 0.75, 0.75, 0.75},
                {1.0, 1.0, 1.0, 1.0, 1.0},
            },
            {
                {0.0, 0.0, 0.0, 0.0, 0.0},
                {0.25, 0.25, 0.25, 0.25, 0.25},
                {0.5, 0.5, 0.5, 0.5, 0.5},
                {0.75, 0.75, 0.75, 0.75, 0.75},
                {1.0, 1.0, 1.0, 1.0, 1.0},
            },
            {
                {0.0, 0.0, 0.0, 0.0, 0.0},
                {0.25, 0.25, 0.25, 0.25, 0.25},
                {0.5, 0.5, 0.5, 0.5, 0.5},
                {0.75, 0.75, 0.75, 0.75, 0.75},
                {1.0, 1.0, 1.0, 1.0, 1.0},
            },
            {
                {0.0, 0.0, 0.0, 0.0, 0.0},
                {0.25, 0.25, 0.25, 0.25, 0.25},
                {0.5, 0.5, 0.5, 0.5, 0.5},
                {0.75, 0.75, 0.75, 0.75, 0.75},
                {1.0, 1.0, 1.0, 1.0, 1.0},
            },
            {
                {0.0, 0.0, 0.0, 0.0, 0.0},
                {0.25, 0.25, 0.25, 0.25, 0.25},
                {0.5, 0.5, 0.5, 0.5, 0.5},
                {0.75, 0.75, 0.75, 0.75, 0.75},
                {1.0, 1.0, 1.0, 1.0, 1.0},
            },
        },
        {
            {
                {0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0},
            },
            {
                {0.25, 0.25, 0.25, 0.25, 0.25},
                {0.25, 0.25, 0.25, 0.25, 0.25},
                {0.25, 0.25, 0.25, 0.25, 0.25},
                {0.25, 0.25, 0.25, 0.25, 0.25},
                {0.25, 0.25, 0.25, 0.25, 0.25},
            },
            {
                {0.5, 0.5, 0.5, 0.5, 0.5},
                {0.5, 0.5, 0.5, 0.5, 0.5},
                {0.5, 0.5, 0.5, 0.5, 0.5},
                {0.5, 0.5, 0.5, 0.5, 0.5},
                {0.5, 0.5, 0.5, 0.5, 0.5},
            },
            {
                {0.75, 0.75, 0.75, 0.75, 0.75},
                {0.75, 0.75, 0.75, 0.75, 0.75},
                {0.75, 0.75, 0.75, 0.75, 0.75},
                {0.75, 0.75, 0.75, 0.75, 0.75},
                {0.75, 0.75, 0.75, 0.75, 0.75},
            },
            {
                {1.0, 1.0, 1.0, 1.0, 1.0},
                {1.0, 1.0, 1.0, 1.0, 1.0},
                {1.0, 1.0, 1.0, 1.0, 1.0},
                {1.0, 1.0, 1.0, 1.0, 1.0},
                {1.0, 1.0, 1.0, 1.0, 1.0},
            },
        },
    };
    for (int baseIndex = 0; baseIndex < 4; ++baseIndex) {
      for (int iz = 0; iz < 5; ++iz) {
        real const z = static_cast<real>(iz) / 4_r; // range [0.0, 1.0]
        for (int iy = 0; iy < 5; ++iy) {
          real const y = static_cast<real>(iy) / 4_r; // range [0.0, 1.0]
          for (int ix = 0; ix < 5; ++ix) {
            real const x = static_cast<real>(ix) / 4_r; // range [0.0, 1.0]
            real const expected = static_cast<real>(kExpectedValues[baseIndex][iz][iy][ix]);
            real const actual = b.GetValue(baseIndex, Real3{x, y, z});
            EXPECT_NEAR(expected, actual, 1e-6_r);
          }
        }
      }
    }
  }

  // GetDValue(baseIndex, x)
  {
    // Just to prove that GetValue is a constexpr
    static_assert(
        NearEqual(Real3{-1._r, -1._r, -1._r}, b.GetDValue(0, Real3{0.1_r, 0.2_r, 0.3_r})));
    static_assert(
        NearEqual(Real3{1.0_r, 0.0_r, 0.0_r}, b.GetDValue(1, Real3{0.4_r, 0.5_r, 0.6_r})));

    // Test several more inputs. The result should only depend on the baseIndex.
    for (int baseIndex = 0; baseIndex < 2; ++baseIndex) {
      for (int iz = 0; iz < 5; ++iz) {
        real const z = static_cast<real>(iz) / 4_r; // range [0.0, 1.0]
        for (int iy = 0; iy < 5; ++iy) {
          real const y = static_cast<real>(iy) / 4_r; // range [0.0, 1.0]
          for (int ix = 0; ix < 5; ++ix) {
            real const x = static_cast<real>(ix) / 4_r; // range [0.0, 1.0]
            Real3 const actual = b.GetDValue(baseIndex, Real3{x, y, z});
            if (baseIndex == 0) {
              EXPECT_TRUE(NearEqual(Real3{-1_r, -1_r, -1_r}, actual));
            } else {
              EXPECT_TRUE(NearEqual(Real3{1_r, 0_r, 0_r}, actual));
            }
          }
        }
      }
    }
  }
}

TEST(Tetrahedral, SimplexQuadrature) {
  constexpr auto kQuad = tetrahedral::kTetrahedralQuadrature1;
  static_assert(1 == kQuad.points.size());
  static_assert(NearEqual(Real3{0.25_r, 0.25_r, 0.25_r}, kQuad.points[0]));
  static_assert(1 == kQuad.weights.size());
  static_assert(NearEqual(0.16666667_r, kQuad.weights[0]));
}

TEST(Tetrahedral, BasisFunctionsEvaluated) {
  constexpr int kPolyOrder = 1;
  constexpr int kNumQuadPoints = 1;
  auto b = tetrahedral::BasisFunctionsEvaluated<kPolyOrder, kNumQuadPoints>(
      tetrahedral::kTetrahedralQuadrature1);

  // constats
  static_assert(1 == b.kPolyOrder);
  static_assert(3 == b.kSpaceDim);
  static_assert(4 == b.kNumDofs);
  static_assert(1 == b.kNumQuadPoints);

  // basisEvaluated
  static_assert(2 == b.basisEvaluated.num_dims);
  static_assert(1 == b.basisEvaluated.dims[0]);
  static_assert(4 == b.basisEvaluated.dims[1]);
  assert(NearEqual(NdArray<real, 1, 4>{Real4{0.25_r, 0.25_r, 0.25_r, 0.25_r}}, b.basisEvaluated));

  // dBasisEvaluated
  static_assert(3 == b.dBasisEvaluated.num_dims);
  static_assert(1 == b.dBasisEvaluated.dims[0]);
  static_assert(4 == b.dBasisEvaluated.dims[1]);
  static_assert(3 == b.dBasisEvaluated.dims[2]);
  assert((Real3{-1_r, -1_r, -1_r} == b.dBasisEvaluated[0][0]));
  assert((Real3{+1_r, +0_r, +0_r} == b.dBasisEvaluated[0][1]));
  assert((Real3{+0_r, +1_r, +0_r} == b.dBasisEvaluated[0][2]));
  assert((Real3{+0_r, +0_r, +1_r} == b.dBasisEvaluated[0][3]));
}

/**
  Test the correct evaluation of basis functions over a tetrahedron.
*/
TEST(Tetrahedral, Pk3DElement) {
  constexpr int kPolyOrder = 1;
  using FiniteElement = tetrahedral::Pk3DElement<kPolyOrder>;

  std::vector<Real3> const coords = {
      Real3{0.0_r, 0.0_r, 0.0_r}, // 0
      Real3{1.0_r, 0.0_r, 0.0_r}, // 1
      Real3{0.0_r, 1.0_r, 0.0_r}, // 2
      Real3{1.0_r, 1.0_r, 0.0_r}, // 3
      Real3{0.0_r, 0.0_r, 1.0_r}, // 4
      Real3{1.0_r, 0.0_r, 1.0_r}, // 5
      Real3{0.0_r, 1.0_r, 1.0_r}, // 6
      Real3{1.0_r, 1.0_r, 1.0_r}, // 7
  };
  std::vector<Int4> const indices = {
      Int4{2, 6, 3, 0}, // corner vert 2
      Int4{7, 3, 6, 5}, // corner vert 7
      Int4{1, 3, 5, 0}, // corner vert 1
      Int4{4, 0, 5, 6}, // corner vert 4
      Int4{6, 0, 3, 5}, // the one fully interior tetrahedron
  };
  TetrahedralMesh mesh(coords, indices);
  Span<Real3 const> coordinates = mesh.GetNodeCoordinates();
  Span<Int4 const> connectivity = mesh.GetElementConnectivity();

  // Create the elements
  std::vector<tetrahedral::Pk3DElement<kPolyOrder>> elements;
  elements.reserve(mesh.GetNumElements());
  for (int i = 0; i < mesh.GetNumElements(); ++i) {
    elements.emplace_back(i, coordinates, connectivity);
  }

  // Verify initial values
  for (int e = 0; e < isize(elements); ++e) {
    // elementIndex
    FiniteElement const& elem = elements[e];
    EXPECT_EQ(e, elem.elementIndex);

    // kNodesCrdsParam
    constexpr NdArray<real, 4, 3> kExpectedNodesCrdsParam = {
        Real3{0_r, 0_r, 0_r}, Real3{1_r, 0_r, 0_r}, Real3{0_r, 1_r, 0_r}, Real3{0_r, 0_r, 1_r}};
    static_assert(kExpectedNodesCrdsParam == FiniteElement::kNodesCrdsParam);

    // coordinates (points to mesh data)
    EXPECT_EQ(coordinates.size(), elem.coordinates.size());
    EXPECT_EQ(coordinates.data(), elem.coordinates.data());

    // connectivity (points to mesh data)
    EXPECT_EQ(connectivity.size(), elem.connectivity.size());
    EXPECT_EQ(connectivity.data(), elem.connectivity.data());

    // nodesCrdsPhys
    Int4 const tet = connectivity[elem.elementIndex];
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(coordinates[tet[i]], elem.nodesCrdsPhys[i]);
    }

    // mapEvaluated
    NdArray<real, 1, 3> const expectedMapEvaluated[] = {
        {Real3{0.25_r, 0.75_r, 0.25_r}},
        {Real3{0.75_r, 0.75_r, 0.75_r}},
        {Real3{0.75_r, 0.25_r, 0.25_r}},
        {Real3{0.25_r, 0.25_r, 0.75_r}},
        {Real3{0.5_r, 0.5_r, 0.5_r}}};
    EXPECT_TRUE(NearEqual(expectedMapEvaluated[e], elem.mapEvaluated));

    // dMapEvaluated
    NdArray<real, 1, 3, 3> const expectedDmapEvaluated[] = {
        {Matrix3x3r{Real3{0_r, 1_r, 0_r}, Real3{0_r, 0_r, -1_r}, Real3{-1_r, -1_r, -1_r}}},
        {Matrix3x3r{Real3{0_r, -1_r, 0_r}, Real3{0_r, 0_r, -1_r}, Real3{1_r, 1_r, 1_r}}},
        {Matrix3x3r{Real3{0_r, 0_r, -1_r}, Real3{-1_r, -1_r, -1_r}, Real3{0_r, 1_r, 0_r}}},
        {Matrix3x3r{Real3{0_r, 1_r, 0_r}, Real3{0_r, 0_r, 1_r}, Real3{1_r, 1_r, 1_r}}},
        {Matrix3x3r{Real3{0_r, 1_r, 1_r}, Real3{-1_r, 0_r, -1_r}, Real3{-1_r, -1_r, 0_r}}}};
    EXPECT_TRUE(NearEqual(expectedDmapEvaluated[e], elem.dMapEvaluated));

    // dMapEvaluatedInv
    NdArray<real, 1, 3, 3> const expectedDmapEvaludatedInv[] = {
        {Matrix3x3r{Real3{-1_r, 1_r, -1_r}, Real3{1_r, 0_r, 0_r}, Real3{0_r, -1_r, 0_r}}},
        {Matrix3x3r{Real3{1_r, 1_r, 1_r}, Real3{-1_r, 0_r, 0_r}, Real3{0_r, -1_r, 0_r}}},
        {Matrix3x3r{Real3{1_r, -1_r, -1_r}, Real3{0_r, 0_r, 1_r}, Real3{-1_r, 0_r, 0_r}}},
        {Matrix3x3r{Real3{-1_r, -1_r, 1_r}, Real3{1_r, 0_r, 0_r}, Real3{0_r, 1_r, 0_r}}},
        {Matrix3x3r{
            Real3{-0.5_r, -0.5_r, -0.5_r},
            Real3{0.5_r, 0.5_r, -0.5_r},
            Real3{0.5_r, -0.5_r, 0.5_r}}}};
    EXPECT_TRUE(NearEqual(expectedDmapEvaludatedInv[e], elem.dMapEvaluatedInv));

    // dMapEvaluatedDet
    NdArray<real, 1> const expectedDmapEvaluatedDet[] = {{1_r}, {1_r}, {1_r}, {1_r}, {2_r}};
    EXPECT_TRUE(NearEqual(expectedDmapEvaluatedDet[e], elem.dMapEvaluatedDet));

    // kDBasisEvaluated
    NdArray<real, 1, 4, 3> expected_dbasis_evaluated[] = {
        {NdArray<real, 4, 3>{
            Real3{0_r, 0_r, 1_r},
            Real3{-1_r, 1_r, -1_r},
            Real3{1_r, 0_r, 0_r},
            Real3{0_r, -1_r, 0_r}}},
        {NdArray<real, 4, 3>{
            Real3{0_r, 0_r, -1_r},
            Real3{1_r, 1_r, 1_r},
            Real3{-1_r, 0_r, 0_r},
            Real3{0_r, -1_r, 0_r}}},
        {NdArray<real, 4, 3>{
            Real3{0_r, 1_r, 0_r},
            Real3{1_r, -1_r, -1_r},
            Real3{0_r, 0_r, 1_r},
            Real3{-1_r, 0_r, 0_r}}},
        {NdArray<real, 4, 3>{
            Real3{0_r, 0_r, -1_r},
            Real3{-1_r, -1_r, 1_r},
            Real3{1_r, 0_r, 0_r},
            Real3{0_r, 1_r, 0_r}}},
        {NdArray<real, 4, 3>{
            Real3{-0.5_r, 0.5_r, 0.5_r},
            Real3{-0.5_r, -0.5_r, -0.5_r},
            Real3{0.5_r, 0.5_r, -0.5_r},
            Real3{0.5_r, -0.5_r, 0.5_r}}},
    };
    EXPECT_TRUE(NearEqual(expected_dbasis_evaluated[e], elem.dBasisEvaluated));
  }
}

TEST(Tetrahedral, LocalToGlobalMap) {
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
  Real3 constexpr kCoordinates[] = {
      Real3{0.0_r, 0.0_r, 0.0_r}, // 0
      Real3{1.0_r, 0.0_r, 0.0_r}, // 1
      Real3{0.0_r, 1.0_r, 0.0_r}, // 2
      Real3{1.0_r, 1.0_r, 0.0_r}, // 3
      Real3{0.0_r, 0.0_r, 1.0_r}, // 4
      Real3{1.0_r, 0.0_r, 1.0_r}, // 5
      Real3{0.0_r, 1.0_r, 1.0_r}, // 6
      Real3{1.0_r, 1.0_r, 1.0_r}, // 7
  };
  Int4 constexpr kConnectivity[] = {
      Int4{2, 6, 3, 0}, // corner vert 2
      Int4{7, 3, 6, 5}, // corner vert 7
      Int4{1, 3, 5, 0}, // corner vert 1
      Int4{4, 0, 5, 6}, // corner vert 4
      Int4{6, 0, 3, 5}, // the one fully interior tetrahedron
  };

  int const numNodes = isize(kCoordinates);
  int const numElements = isize(kConnectivity);

  using Basis = tetrahedral::BarycentricBasisTetrahedra<1>;
  TetrahedralMesh mesh(kCoordinates, kConnectivity);
  Local2GlobalMap map;
  map.InitializeFromMeshAndBasis(&mesh, Basis{}, 3);

  EXPECT_EQ(numElements * 12, map.GetNumIndices());
  EXPECT_EQ(numElements, map.GetNumElements());
  EXPECT_EQ(0, map.GetGlobalRange().Min());
  EXPECT_EQ(numNodes * 3, map.GetGlobalRange().Max() + 1);

  for (int e = 0; e < numElements; ++e) {
    EXPECT_EQ(12, map.GetElementSizes()[e]);
    EXPECT_EQ(12, map.GetElementSize(e));
    EXPECT_EQ(e * 12, map.GetElementOffsets()[e]);
    EXPECT_EQ(e * 12, map.GetElementOffset(e));
  }

  // Evaluated Indices
  int constexpr kExpectedIndices[] = {
      18, 19, 20, 6,  7,  8,  9,  10, 11, 0,  1,  2, // element 0
      9,  10, 11, 21, 22, 23, 18, 19, 20, 15, 16, 17, // element 1
      9,  10, 11, 3,  4,  5,  15, 16, 17, 0,  1,  2, // element 2
      0,  1,  2,  12, 13, 14, 15, 16, 17, 18, 19, 20, // element 3
      18, 19, 20, 0,  1,  2,  9,  10, 11, 15, 16, 17, // element 4
  };
  EXPECT_EQ(numElements * 12, map.GetGlobalIndices().size());
  EXPECT_SPAN_EQ(Span<int const>{kExpectedIndices}, map.GetGlobalIndices());
  for (int e = 0; e < numElements; ++e) {
    EXPECT_SPAN_EQ(Span<int const>(&kExpectedIndices[e * 12], 12), map.GetGlobalIndices(e));
    EXPECT_SPAN_EQ(Span<int const>(&kExpectedIndices[e * 12], 12), map.GetGlobalIndices(e));
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(kExpectedIndices[e * 12 + i], map.GetGlobalIndex(e, i));
    }
  }

  // Index range per element
  Interval<int> constexpr kExpectedRanges[] = {
      Interval<int>{0, 21}, // element 0
      Interval<int>{9, 24}, // element 1
      Interval<int>{0, 18}, // element 2
      Interval<int>{0, 21}, // element 3
      Interval<int>{0, 21}, // element 4
  };
  for (int e = 0; e < numElements; ++e) {
    EXPECT_EQ(kExpectedRanges[e].Min(), map.GetGlobalRange(e).Min());
    EXPECT_EQ(kExpectedRanges[e].Max(), map.GetGlobalRange(e).Max());
  }
}

/**
  Test the correct evaluation of basis functions over a tetrahedron.
*/
template <int kQuadDegreeTrace>
void TestPk3DElementTrace(
    NdArray<tetrahedral::TetrahedralQuadrature<kQuadDegreeTrace>, 4> const& quadrature) {
  constexpr Real4 kExactArea = Real4{1_r / 2_r, 1_r / 2_r, 1_r / 2_r, 1.732050807568877_r / 2_r};
  constexpr int kNumFaces = 4;
  constexpr int kPolyOrder = 1;
  constexpr int kQuadDegree = 1;

  // The element types for the base element and the trace
  using ElementT = tetrahedral::Pk3DElement<kPolyOrder, kQuadDegree>;
  using ElementTraceT = tetrahedral::Pk3DElementTrace<ElementT, kQuadDegreeTrace>;

  // Create the one tet mesh
  TetrahedralMesh mesh = test::CreateMinimalTetMeshSingleTet();

  // Create the volume element
  auto element = ElementT{
      0,
      mesh.GetNodeCoordinates(),
      mesh.GetElementConnectivity(),
      tetrahedral::kTetrahedralQuadrature1};

  // Create the four element traces over the base element
  std::vector<ElementTraceT> elementTraces;
  elementTraces.reserve(kNumFaces);
  for (int f = 0; f < kNumFaces; ++f) {
    elementTraces.emplace_back(element, f, quadrature[f]);
  }

  // Compute the area for the face
  real areas[4] = {};
  for (int f = 0; f < kNumFaces; ++f) {
    for (int q = 0; q < elementTraces[f].kNumQuadPoints; ++q) {
      areas[f] += elementTraces[f].quadWeights[q];
    }
  }

  // Check that all areas for faces are correct
  for (int f = 0; f < kNumFaces; ++f) {
    EXPECT_NEAR(areas[f], kExactArea[f], 1.e-6_r);
  }

  // Check that the basis functions are evaluated coorectly

  for (int f = 0; f < kNumFaces; ++f) {
    for (int q = 0; q < elementTraces[f].kNumQuadPoints; ++q) {
      for (int i = 1; i < elementTraces[f].kNumDofs; ++i) {
        EXPECT_NEAR(
            elementTraces[f].basisEvaluated[q][i],
            elementTraces[f].quadrature.points[q][(i - 1)],
            1.e-6_r);
      }
    }
  }

  // Check that the volume computed using bulk quadrature or trace quadrature with the
  // divergence theorem are consistent
  real volume = 0;
  for (int f = 0; f < kNumFaces; ++f) {
    for (int q = 0; q < elementTraces[f].kNumQuadPoints; q++) {
      for (int i = 0; i < elementTraces[f].kSpaceDim; ++i) {
        volume += elementTraces[f].mapEvaluated[q][i] * elementTraces[f].normals[q][i] *
            elementTraces[f].quadWeights[q];
      }
    }
  }
  volume /= elementTraces[0].kSpaceDim;

  real volume_bulk = 0;
  for (int q = 0; q < element.kNumQuadPoints; q++) {
    volume_bulk += element.quadWeights[q];
  }

  EXPECT_NEAR(volume, volume_bulk, 1.e-6_r);
};
TEST(Tetrahedral, Pk3DElementTrace) {
  TestPk3DElementTrace<1>(tetrahedral::kTetrahedralTraceQuadrature1);
  TestPk3DElementTrace<3>(tetrahedral::kTetrahedralTraceQuadrature3);
  TestPk3DElementTrace<6>(tetrahedral::kTetrahedralTraceQuadrature6);
  TestPk3DElementTrace<7>(tetrahedral::kTetrahedralTraceQuadrature7);
  TestPk3DElementTrace<12>(tetrahedral::kTetrahedralTraceQuadrature12);
  TestPk3DElementTrace<16>(tetrahedral::kTetrahedralTraceQuadrature16);
}

/**
  Test the correct evaluation of basis functions over an irregular tetrahedron.
*/

template <int kQuadDegreeTrace>
void TestPk3DElementTraceIrregular(
    NdArray<tetrahedral::TetrahedralQuadrature<kQuadDegreeTrace>, 4> const& quadrature) {
  constexpr int kNumFaces = 4;
  constexpr int kPolyOrder = 1;
  constexpr int kQuadDegree = 1;

  // The element types for the base element and the trace
  using ElementT = tetrahedral::Pk3DElement<kPolyOrder, kQuadDegree>;
  using ElementTraceT = tetrahedral::Pk3DElementTrace<ElementT, kQuadDegreeTrace>;

  // Create the one tet mesh
  auto generator = RandomGenerator(42);
  std::vector<Real3> coordinates(4);
  SetRandom(generator, -0.2_r, 0.2_r, MakeSpan(coordinates));
  coordinates[1][0] += 1_r;
  coordinates[2][1] += 1_r;
  coordinates[3][2] += 1_r;

  std::vector<Int4> const connectivity = {
      Int4{0, 1, 2, 3},
  };
  TetrahedralMesh mesh(coordinates, connectivity);

  // Create the volume element
  auto element = ElementT{
      0,
      mesh.GetNodeCoordinates(),
      mesh.GetElementConnectivity(),
      tetrahedral::kTetrahedralQuadrature1};

  // Create the four element traces over the base element
  std::vector<ElementTraceT> elementTraces;
  elementTraces.reserve(kNumFaces);
  for (int f = 0; f < kNumFaces; ++f) {
    elementTraces.emplace_back(element, f, quadrature[f]);
  }

  // Compute the area for the face
  real areas[4] = {};
  for (int f = 0; f < kNumFaces; ++f) {
    for (int q = 0; q < elementTraces[f].kNumQuadPoints; ++q) {
      areas[f] += elementTraces[f].quadWeights[q];
    }
  }

  // Check that the basis functions are evaluated coorectly

  for (int f = 0; f < kNumFaces; ++f) {
    for (int q = 0; q < elementTraces[f].kNumQuadPoints; ++q) {
      for (int i = 1; i < elementTraces[f].kNumDofs; ++i) {
        EXPECT_NEAR(
            elementTraces[f].basisEvaluated[q][i],
            elementTraces[f].quadrature.points[q][(i - 1)],
            1.e-6);
      }
    }
  }

  // Check that the volume computed using bulk quadrature or trace quadrature with the
  // divergence theorem are consistent
  real volume = 0;
  for (int f = 0; f < kNumFaces; ++f) {
    for (int q = 0; q < elementTraces[f].kNumQuadPoints; q++) {
      for (int i = 0; i < elementTraces[f].kSpaceDim; ++i) {
        volume += elementTraces[f].mapEvaluated[q][i] * elementTraces[f].normals[q][i] *
            elementTraces[f].quadWeights[q];
      }
    }
  }
  volume /= elementTraces[0].kSpaceDim;

  real volume_bulk = 0;
  for (int q = 0; q < element.kNumQuadPoints; q++) {
    volume_bulk += element.quadWeights[q];
  }

  EXPECT_NEAR(volume, volume_bulk, 1.e-6_r);
}

TEST(Tetrahedral, Pk3DElementTraceIrregular) {
  TestPk3DElementTraceIrregular<1>(tetrahedral::kTetrahedralTraceQuadrature1);
  TestPk3DElementTraceIrregular<3>(tetrahedral::kTetrahedralTraceQuadrature3);
  TestPk3DElementTraceIrregular<6>(tetrahedral::kTetrahedralTraceQuadrature6);
  TestPk3DElementTraceIrregular<7>(tetrahedral::kTetrahedralTraceQuadrature7);
  TestPk3DElementTraceIrregular<12>(tetrahedral::kTetrahedralTraceQuadrature12);
  TestPk3DElementTraceIrregular<16>(tetrahedral::kTetrahedralTraceQuadrature16);
}
