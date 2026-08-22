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

#include <mochi_core/linear_algebra/strided_matrix.h>

#include <cmath>
#include <cstddef>

namespace mochi {

/**
 * Stable neo-Hookean material model by Smith et al. (2018).
 *
 * Note:
 *   - This is an alternative implementation of smith_neo_hookean.{h,cpp} designed for batched
 *     evaluation using strided matrices in CUDA.
 *   - PSD projection is not supported yet. The Hessian may be indefinite.
 */

/**
 * T_i + H_i term.
 * diagCoef = \mu (1 - 1/(I_C + 1))
 * offcoef = \lambda (J - \alpha)
 */
template <typename Scalar = double>
MOCHI_ANY auto BaseConstitutive(auto& F, Scalar diagCoef, Scalar offCoef) {
  StridedMatrix<Scalar, 9, 9> result{};
  for (int i = 0; i < 9; ++i) {
    for (int j = 0; j < 9; ++j) {
      result(i, j) = (i == j) ? diagCoef : Scalar{0};
    }
  }
  result(4, 0) = result(0, 4) = offCoef * F(2, 2);
  result(5, 0) = result(0, 5) = -offCoef * F(1, 2);
  result(7, 0) = result(0, 7) = -offCoef * F(2, 1);
  result(8, 0) = result(0, 8) = offCoef * F(1, 1);
  result(3, 1) = result(1, 3) = -offCoef * F(2, 2);
  result(3, 2) = result(2, 3) = offCoef * F(1, 2);
  result(4, 2) = result(2, 4) = -offCoef * F(0, 2);
  result(5, 1) = result(1, 5) = offCoef * F(0, 2);
  result(6, 1) = result(1, 6) = offCoef * F(2, 1);
  result(6, 2) = result(2, 6) = -offCoef * F(1, 1);
  result(7, 2) = result(2, 7) = offCoef * F(0, 1);
  result(8, 1) = result(1, 8) = -offCoef * F(0, 1);
  result(6, 4) = result(4, 6) = -offCoef * F(2, 0);
  result(6, 5) = result(5, 6) = offCoef * F(1, 0);
  result(7, 3) = result(3, 7) = offCoef * F(2, 0);
  result(7, 5) = result(5, 7) = -offCoef * F(0, 0);
  result(8, 3) = result(3, 8) = -offCoef * F(1, 0);
  result(8, 4) = result(4, 8) = offCoef * F(0, 0);
  return result;
}

namespace materials {

template <bool kHessian>
MOCHI_ANY auto AltSmithNeoHookeanHessian(
    real mu,
    real lambda,
    real alpha,
    real I_C,
    real J,
    StridedMatrix<real, 3, 3> const& F,
    StridedMatrix<real, 3, 3> const& K) {
  if constexpr (kHessian) {
    auto C = BaseConstitutive(F, mu * (1_r - 1_r / (I_C + 1_r)), lambda * (J - alpha));
    // Add the two rank one terms
    StridedVectorView<real const, 9> f(F.Data());
    C += (2_r * mu) / Sqr(I_C + 1_r) * f * f.Transpose();
    StridedVectorView<real const, 9> g(K.Data());
    C += lambda * g * g.Transpose();
    return C;
  } else {
    return Void{};
  }
}

template <bool kStress>
MOCHI_ANY auto AltSmithNeoHookeanStress(
    real mu,
    real lambda,
    real alpha,
    real I_C,
    real J,
    StridedMatrix<real, 3, 3> const& F,
    StridedMatrix<real, 3, 3> const& K) {
  if constexpr (kStress) {
    return StridedMatrix<real, 3, 3>(mu * (1_r - 1_r / (I_C + 1_r)) * F + lambda * (J - alpha) * K);
  } else {
    return Void{};
  }
}

template <bool kEnergy>
MOCHI_ANY auto AltSmithNeoHookeanEnergy(real mu, real lambda, real alpha, real I_C, real J) {
  if constexpr (kEnergy) {
    return 0.5 * (mu * (I_C - 3.0 - std::log(I_C + 1.0)) + lambda * Sqr(J - alpha));
  } else {
    return Void{};
  }
}

template <typename ET, typename ST, typename HT>
struct MaterialState {
  ET energy;
  ST stress;
  HT hessian;
};
// This deduction guide is not needed in C++20.
// However there seems to be a bug in the compiler for platform010/*, so we need it until it is
// fixed.
template <typename ET, typename ST, typename HT>
MaterialState(ET&&, ST&&, HT&&) -> MaterialState<ET, ST, HT>;

/**
 * @brief Compute a subset of strain energy, stress and hessian for a given deformation state.
 * @tparam kEnergy computes the strain energy if true.
 * @tparam kStress computes the first Piola Kirchhoff stress if true.
 * @tparam kHessian computes the energy hessian if true.
 *
 * The result is a triplet that has either the actual values of a Void object.
 */
template <bool kEnergy, bool kStress, bool kHessian>
MOCHI_ANY auto
AltSmithNeoHookeanState(real mu, real lambda, real alpha, StridedMatrix<real, 3, 3> const& F) {
  auto K = StridedCofactors3x3(F);
  // Determinant written in terms of the cofactors.
  auto J = F(0, 0) * K(0, 0) + F(1, 0) * K(1, 0) + F(2, 0) * K(2, 0);
  auto sum = []<std::size_t... I>(auto const* v, std::index_sequence<I...>) {
    return (... + (v[I] * v[I]));
  };
  auto I_C = sum(F.Data(), std::make_index_sequence<3 * 3>{});
  return MaterialState{
      AltSmithNeoHookeanEnergy<kEnergy>(mu, lambda, alpha, I_C, J),
      AltSmithNeoHookeanStress<kStress>(mu, lambda, alpha, I_C, J, F, K),
      AltSmithNeoHookeanHessian<kHessian>(mu, lambda, alpha, I_C, J, F, K)};
}

inline MOCHI_ANY auto
AltSmithNeoHookean(real mu, real lambda, real alpha, StridedMatrix<real, 3, 3> const& F) {
  return AltSmithNeoHookeanState<false, false, true>(mu, lambda, alpha, F).hessian;
}

} // namespace materials

} // namespace mochi
