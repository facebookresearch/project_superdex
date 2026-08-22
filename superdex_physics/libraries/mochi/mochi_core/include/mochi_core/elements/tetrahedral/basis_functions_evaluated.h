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

#include <mochi_core/elements/tetrahedral/basis_functions.h>
#include <mochi_core/elements/tetrahedral/simplex_quadrature.h>
#include <mochi_core/utils/nd_array.h>

namespace mochi::tetrahedral {

template <int kPolyOrder_ = 1, int kNumQuadPoints_ = 1>
class BasisFunctionsEvaluated final {
  // WARNING: In order to implement higher order polynomials we need to figure
  // out a better node numbering scheme. Most of the infrastructure is in place
  static_assert(kPolyOrder_ == 1, "Higher order polynomials not yet implemented");

 public:
  static constexpr int kPolyOrder = kPolyOrder_;
  static constexpr int kNumQuadPoints = kNumQuadPoints_;

  BasisFunctionsEvaluated(TetrahedralQuadrature<kNumQuadPoints> const& quad) : quadrature(quad) {};

  /**
    Get the basis value evaluated at the quad points.

    Args:
      baseIndex (int) : the index of the basis function to evaluate
      quadNumber (int): the index of the quadrature points where to eval

    Returns:
      float: the value of the base function
  */
  constexpr real GetBasisValue(int quadNumber, int baseIndex) const {
    return basisEvaluated[quadNumber][baseIndex];
  }

  /**
    Get the basis value evaluated at the quad points.

    Args:
      baseIndex(int) : the index of the basis function to evaluate
      quadNumber(int) : the index of the quadrature points where to eval

    Returns :
      ndarray: the value of the base function derivatives wrt param crds
   */
  constexpr Real3 GetBasisDValue(int quadNumber, int baseIndex) const {
    return dBasisEvaluated[quadNumber][baseIndex];
  }

  // @TODO[MAURIZIO] Make all of these private and have accessors/setters instead
  TetrahedralQuadrature<kNumQuadPoints> const& quadrature;
  static constexpr int kSpaceDim = 3;
  static constexpr BarycentricBasisTetrahedra<kPolyOrder> kBasis = {};
  static constexpr int kNumDofs = BarycentricBasisTetrahedra<kPolyOrder>::kNumDofs;

 private:
  using BasisEvaluatedType = NdArray<real, kNumQuadPoints, kNumDofs>;
  using DBasisEvaluatedType = NdArray<real, kNumQuadPoints, kNumDofs, kSpaceDim>;

  // Compile-time algorithm to compute basisEvaluated
  BasisEvaluatedType ComputeBasisEvaluated() {
    BasisEvaluatedType result;
    for (int q = 0; q < kNumQuadPoints; ++q) {
      for (int f = 0; f < kNumDofs; ++f) {
        result[q][f] = kBasis.GetValue(f, quadrature.points[q]);
      }
    }
    return result;
  }

  // Compile-time algorithm to compute dBasisEvaluated
  DBasisEvaluatedType ComputeDBasisEvaluated() {
    DBasisEvaluatedType result;
    for (int q = 0; q < kNumQuadPoints; ++q) {
      for (int f = 0; f < kNumDofs; ++f) {
        result[q][f] = kBasis.GetDValue(f, quadrature.points[q]);
      }
    }
    return result;
  }

 public:
  BasisEvaluatedType const basisEvaluated = ComputeBasisEvaluated();
  DBasisEvaluatedType const dBasisEvaluated = ComputeDBasisEvaluated();
};

} // namespace mochi::tetrahedral
