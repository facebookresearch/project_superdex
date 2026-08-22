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

#include <mochi_core/elements/triangular/basis_functions.h>
#include <mochi_core/elements/triangular/simplex_quadrature.h>
#include <mochi_core/utils/nd_array.h>

namespace mochi::triangular {

template <int kPolyOrder_ = 1, int kNumQuadPoints_ = 1>
class BasisFunctionsEvaluatedTriangular final {
  // WARNING: In order to implement higher order polynomials we need to figure
  // out a better node numbering scheme. Most of the infrastructure is in place
  static_assert(kPolyOrder_ == 1, "higher order polynomials not yet implemented");

 public:
  static constexpr int kPolyOrder = kPolyOrder_;
  static constexpr int kNumQuadPoints = kNumQuadPoints_;

  BasisFunctionsEvaluatedTriangular() = default;
  /**
    Get the basis value evaluated at the quad points.

    Args:
      baseIndex (int) : the index of the basis function to evaluate
      quadNumber (int): the index of the quadrature points where to eval

    Returns:
      float: the value of the base function
  */
  constexpr real GetBasisValue(int quadNumber, int baseIndex) const {
    return kBasisEvaluated[quadNumber][baseIndex];
  }

  /**
    Get the basis value evaluated at the quad points.

    Args:
      baseIndex (int) : the index of the basis function to evaluate
      quadNumber (int): the index of the quadrature points where to eval

    Returns:
      ndarray: the value of the base function derivatives wrt param crds
  */
  constexpr Real2 GetBasisDValue(int quadNumber, int baseIndex) const {
    return kDBasisEvaluated[quadNumber][baseIndex];
  }

  static constexpr int kSpaceDimParam = 2;
  static constexpr triangular::TriangleQuadrature<kNumQuadPoints> kQuadrature = {};
  static constexpr triangular::BarycentricBasisTriangular<kPolyOrder> kBasis = {};
  static constexpr int kNumDofs = kBasis.kNumDofs;

  using BasisEvaluatedType = NdArray<real, kNumQuadPoints, kNumDofs>;
  using DBasisEvaluatedType = NdArray<real, kNumQuadPoints, kNumDofs, kSpaceDimParam>;

  // Compile-time algorithm to compute kBasisEvaluated
  static constexpr auto PrecomputeBasisEvaluated() {
    BasisEvaluatedType basisEvaluated;
    for (int q = 0; q < kNumQuadPoints; ++q) {
      for (int f = 0; f < kBasis.kNumDofs; ++f) {
        basisEvaluated[q][f] = kBasis.GetValue(f, kQuadrature.points[q]);
      }
    }
    return basisEvaluated;
  }

  // Compile-time algorithm to compute kDBasisEvaluated
  static constexpr auto PrecomputeDBasisEvaluated() {
    DBasisEvaluatedType dBasisEvaluated;
    for (int q = 0; q < kNumQuadPoints; ++q) {
      for (int f = 0; f < kBasis.kNumDofs; ++f) {
        dBasisEvaluated[q][f] = kBasis.GetDValue(f, kQuadrature.points[q]);
      }
    }
    return dBasisEvaluated;
  }

  static constexpr BasisEvaluatedType kBasisEvaluated = PrecomputeBasisEvaluated();
  static constexpr DBasisEvaluatedType kDBasisEvaluated = PrecomputeDBasisEvaluated();
};

} // namespace mochi::triangular
