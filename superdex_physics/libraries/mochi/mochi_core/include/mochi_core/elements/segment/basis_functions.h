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

#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/lagrange_polynomials.h>
#include <mochi_core/utils/nd_array.h>

namespace mochi::segment {

/**
 * @brief Basis functions defined over the unit interval [0,1].
 *
 * For now, there is only a minimal implementation of the trivial case of linear polynomials, but
 * it is formally templated on the polynomial order, to facilitate future extension to higher order
 * polynomials.
 */
template <int kPolyOrder_ = 1>
class BasisSegment final {
 public:
  /**
   * @brief Evaluate basis function at a parametric coordinate.
   * @param baseIndex Local index of the basis function.
   * @param x Parametric coordinate in the unit interval [0,1].
   * @return The basis function value.
   */
  static constexpr real GetValue(int baseIndex, Real1 const& x) {
    static_assert(kPolyOrder_ == 1, "Higher order polynomials not yet implemented");
    MOCHI_ASSERT_VERBOSE(baseIndex == 0 || baseIndex == 1, "Base index is out of range.");
    if (baseIndex == 0) {
      return 1_r - x[0];
    } else {
      return x[0];
    }
  }
  /**
   * @brief Evaluate the derivative of a basis function w.r.t. the parametric coordinate.
   * @param baseIndex Local index of the basis function.
   * @param x Parametric coordinate in the unit interval [0,1].
   * @return The basis function derivative w.r.t. x[0].
   */
  static constexpr Real1 GetDValue(int baseIndex, Real1 const& /*x*/) {
    static_assert(kPolyOrder_ == 1, "Higher order polynomials not yet implemented");
    MOCHI_ASSERT_VERBOSE(baseIndex == 0 || baseIndex == 1, "Base index is out of range.");
    if (baseIndex == 0) {
      return Real1{-1_r};
    } else {
      return Real1{1_r};
    }
  }
  static constexpr int kPolyOrder = kPolyOrder_; // The polynomial order
  static constexpr int kNumDofs = kPolyOrder + 1; // The number of degrees of freedom
};

} // namespace mochi::segment
