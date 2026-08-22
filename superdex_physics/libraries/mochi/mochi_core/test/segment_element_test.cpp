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

#include <mochi_core/elements/segment/finite_element.h>
#include <mochi_core/elements/segment/simplex_quadrature.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

#include <gtest/gtest.h>

using namespace mochi;

namespace {

// Compute the exact integral of x^p over the unit interval [0,1].
// The result is 1/(p+1).
constexpr real ExactMonomialIntegral(int p) {
  return 1_r / static_cast<real>(p + 1);
}

// Compute the numerical integral of x^p using a given quadrature rule.
template <int kNumQuadPoints>
real NumericalMonomialIntegral(int p) {
  using Quad = segment::SegmentQuadrature<kNumQuadPoints>;
  real integral = 0_r;
  for (size_t i = 0; i < Quad::kNumQuadPoints; ++i) {
    real const x = Quad::points[i][0];
    real const w = Quad::weights[i];
    integral += w * Pow(x, static_cast<real>(p));
  }
  return integral;
}

// Test that n-point Gaussian quadrature integrates monomials through degree 2n-1 exactly.
template <int kNumQuadPoints>
void TestQuadratureIntegratesMonomials() {
  int constexpr kMaxExactDegree = 2 * kNumQuadPoints - 1;
  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-10_r : 1e-6_r;

  for (int p = 0; p <= kMaxExactDegree; ++p) {
    real const exact = ExactMonomialIntegral(p);
    real const numerical = NumericalMonomialIntegral<kNumQuadPoints>(p);
    EXPECT_NEAR(exact, numerical, kTol) << "Failed for monomial degree p = " << p;
  }
}

} // namespace

TEST(SegmentQuadrature, OnePointIntegratesMonomialsThroughDegree1) {
  TestQuadratureIntegratesMonomials<1>();
}

TEST(SegmentQuadrature, TwoPointIntegratesMonomialsThroughDegree3) {
  TestQuadratureIntegratesMonomials<2>();
}

TEST(SegmentQuadrature, ThreePointIntegratesMonomialsThroughDegree5) {
  TestQuadratureIntegratesMonomials<3>();
}

// Test that the integral of 1 over a polyline equals the sum of element lengths.
template <int kNumQuadPoints>
void TestPolylineIntegralEqualsLength(bool isClosedLoop = false) {
  // Create a polyline with several segments in 3D space
  DynamicArray<Real3> const coordinates = {
      Real3{0_r, 0_r, 0_r},
      Real3{1_r, 0_r, 0_r},
      Real3{1_r, 1_r, 0_r},
      Real3{1_r, 1_r, 1_r},
      Real3{2_r, 2_r, 2_r},
  };
  int const numNodes = isize(coordinates);
  int const numElements = isClosedLoop ? numNodes : numNodes - 1;

  // Compute the expected total length directly using Pk1DElement::Nodes() for node indexing
  real expectedLength = 0_r;
  for (int e = 0; e < numElements; ++e) {
    segment::Pk1DElement<1, kNumQuadPoints> const element(e, MakeSpan(coordinates));
    Int2 const nodes = element.Nodes();
    expectedLength += Norm(coordinates[nodes[1]] - coordinates[nodes[0]]);
  }

  // Compute the integral of 1 over the polyline using quadrature
  real integratedLength = 0_r;
  for (int e = 0; e < numElements; ++e) {
    segment::Pk1DElement<1, kNumQuadPoints> const element(e, MakeSpan(coordinates));
    for (int q = 0; q < element.kNumQuadPoints; ++q) {
      integratedLength += element.quadWeights[q];
    }
  }

  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-10_r : 1e-6_r;
  EXPECT_NEAR(expectedLength, integratedLength, kTol);
}

TEST(SegmentElement, OnePointQuadratureIntegralEqualsLength) {
  TestPolylineIntegralEqualsLength<1>();
}

TEST(SegmentElement, TwoPointQuadratureIntegralEqualsLength) {
  TestPolylineIntegralEqualsLength<2>();
}

TEST(SegmentElement, ThreePointQuadratureIntegralEqualsLength) {
  TestPolylineIntegralEqualsLength<3>();
}

// Test that the integral of each node's basis function equals half the sum of the lengths of
// elements connected to that node (which is the analytical expected value for linear basis
// functions).
template <int kNumQuadPoints>
void TestBasisFunctionIntegrals(bool isClosedLoop = false) {
  // Create a polyline with several segments in 3D space
  DynamicArray<Real3> const coordinates = {
      Real3{0_r, 0.1_r, 0.2_r},
      Real3{1.3_r, 0.4_r, 0.5_r},
      Real3{1.6_r, 1.7_r, 0.8_r},
      Real3{1.9_r, 1_r, 1_r},
      Real3{2_r, 2_r, 2_r},
  };
  int const numNodes = isize(coordinates);
  int const numElements = isClosedLoop ? numNodes : numNodes - 1;

  // Compute element lengths directly using Pk1DElement::Nodes()
  DynamicArray<real> elementLengths(numElements);
  for (int e = 0; e < numElements; ++e) {
    segment::Pk1DElement<1, kNumQuadPoints> const element(e, MakeSpan(coordinates));
    Int2 const nodes = element.Nodes();
    elementLengths[e] = Norm(coordinates[nodes[1]] - coordinates[nodes[0]]);
  }

  // Compute the expected integral for each node's basis function:
  DynamicArray<real> expectedIntegrals(numNodes);
  for (int n = 0; n < numNodes; ++n) {
    real connectedLength = 0_r;
    if (isClosedLoop) {
      // Every node has exactly 2 connected elements in a closed-loop polyline
      connectedLength += elementLengths[(n - 1 + numNodes) % numNodes]; // left element
      connectedLength += elementLengths[n]; // right element
    } else {
      // Element to the left (if exists)
      if (n > 0) {
        connectedLength += elementLengths[n - 1];
      }
      // Element to the right (if exists)
      if (n < numElements) {
        connectedLength += elementLengths[n];
      }
    }
    expectedIntegrals[n] = connectedLength / 2_r;
  }

  // Compute the integral of each node's basis function using quadrature
  DynamicArray<real> computedIntegrals(numNodes, 0_r);
  for (int e = 0; e < numElements; ++e) {
    segment::Pk1DElement<1, kNumQuadPoints> const element(e, MakeSpan(coordinates));
    Int2 const nodes = element.Nodes();
    for (int q = 0; q < element.kNumQuadPoints; ++q) {
      for (int f = 0; f < element.kNumDofs; ++f) {
        int const globalNode = nodes[f];
        real const basisValue = element.kBasisEvaluated[q][f];
        computedIntegrals[globalNode] += basisValue * element.quadWeights[q];
      }
    }
  }

  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-10_r : 1e-6_r;
  for (int n = 0; n < numNodes; ++n) {
    EXPECT_NEAR(expectedIntegrals[n], computedIntegrals[n], kTol) << "Failed for node " << n;
  }
}

TEST(SegmentElement, OnePointBasisIntegrals) {
  TestBasisFunctionIntegrals<1>();
}

TEST(SegmentElement, TwoPointBasisIntegrals) {
  TestBasisFunctionIntegrals<2>();
}

TEST(SegmentElement, ThreePointBasisIntegrals) {
  TestBasisFunctionIntegrals<3>();
}

TEST(SegmentElement, OnePointQuadratureIntegralEqualsLength_ClosedLoop) {
  TestPolylineIntegralEqualsLength<1>(/*isClosedLoop=*/true);
}

TEST(SegmentElement, TwoPointQuadratureIntegralEqualsLength_ClosedLoop) {
  TestPolylineIntegralEqualsLength<2>(/*isClosedLoop=*/true);
}

TEST(SegmentElement, ThreePointQuadratureIntegralEqualsLength_ClosedLoop) {
  TestPolylineIntegralEqualsLength<3>(/*isClosedLoop=*/true);
}

TEST(SegmentElement, OnePointBasisIntegrals_ClosedLoop) {
  TestBasisFunctionIntegrals<1>(/*isClosedLoop=*/true);
}

TEST(SegmentElement, TwoPointBasisIntegrals_ClosedLoop) {
  TestBasisFunctionIntegrals<2>(/*isClosedLoop=*/true);
}

TEST(SegmentElement, ThreePointBasisIntegrals_ClosedLoop) {
  TestBasisFunctionIntegrals<3>(/*isClosedLoop=*/true);
}

// Test that the isoparametric map and its tangent/Jacobian-determinant fields agree with the
// analytical values for a linear element: position is a linear interpolation of the endpoints,
// the tangent equals the endpoint difference, and its norm equals the segment length.
TEST(SegmentElement, MappedQuadraturePointsAndTangentMatchLinearInterpolation) {
  // A polyline whose middle element (index 1) is the one we will inspect.
  DynamicArray<Real3> const coordinates = {
      Real3{0.1_r, 0.2_r, 0.3_r},
      Real3{1.4_r, 0.5_r, 0.6_r},
      Real3{2.9_r, 1.7_r, 1.1_r},
      Real3{3.0_r, 2.5_r, 2.4_r},
  };
  int constexpr kElementIndex = 1;
  Real3 const x0 = coordinates[kElementIndex];
  Real3 const x1 = coordinates[kElementIndex + 1];
  Real3 const tangent = x1 - x0;
  real const expectedLength = Norm(tangent);

  int constexpr kNumQuadPoints = 3;
  using Element = segment::Pk1DElement<1, kNumQuadPoints>;
  using Quad = segment::SegmentQuadrature<kNumQuadPoints>;
  Element const element(kElementIndex, MakeSpan(coordinates));

  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-10_r : 1e-6_r;
  for (int q = 0; q < kNumQuadPoints; ++q) {
    real const xi = Quad::points[q][0];
    Real3 const expectedPosition = (1_r - xi) * x0 + xi * x1;
    for (int i = 0; i < Element::kSpaceDim; ++i) {
      EXPECT_NEAR(expectedPosition[i], element.mapEvaluated[q][i], kTol);
      EXPECT_NEAR(tangent[i], element.dMapEvaluated[q][i][0], kTol);
    }
    EXPECT_NEAR(expectedLength, element.dMapEvaluatedDet[q], kTol);
  }
}
