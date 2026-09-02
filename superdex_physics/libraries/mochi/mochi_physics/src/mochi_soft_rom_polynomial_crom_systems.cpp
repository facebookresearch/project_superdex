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

#include "mochi_soft_rom_polynomial_crom_systems.h"

#include "mochi_common_components.h"

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/utils/basic_utils.h>

#include <algorithm>
#include <vector>

using namespace mochi;

// Evaluates monomials 1, val^1, val^2, ..., val^order at the given value.
// Returns a vector of size (order + 1) containing the results.
// NOTE: Evaluation is performed in double-precision arithmetic.
[[nodiscard]] static auto EvaluateMonomialsAt(int order, real val) {
  MOCHI_ASSERT(order >= 0, "Order must be non-negative.");

  std::vector<double> result(order + 1);
  result[0] = 1.0;
  for (int o = 1; o <= order; ++o) {
    result[o] = result[o - 1] * static_cast<double>(val);
  }

  return result;
}

[[nodiscard]] std::vector<Int3> rom::polynomial_crom::ComputeMultiIndexSortedByTotalOrder(
    int order) {
  MOCHI_ASSERT(order >= 0, "Order must be non-negative.");

  int const estimatedCount = Pow(order + 1, 3);
  std::vector<Int3> mi;
  mi.reserve(estimatedCount);

  for (int i = 0; i <= order; ++i) {
    for (int j = 0; j <= order - i; ++j) {
      for (int k = 0; k <= order - i - j; ++k) {
        mi.emplace_back(k, j, i);
      }
    }
  }

  // Sort by total order without regard to the order of terms with the same total order.
  std::sort(mi.begin(), mi.end(), [](auto const& a, auto const& b) {
    return (a[0] + a[1] + a[2]) < (b[0] + b[1] + b[2]);
  });

  return mi;
}

[[nodiscard]] RowMatrix<real> rom::polynomial_crom::CreateBasisMatrix(
    int order,
    Span<Real3 const> nodePositions) {
  MOCHI_ASSERT(order >= 0, "Order must be non-negative.");

  auto const multiIndex = ComputeMultiIndexSortedByTotalOrder(order);
  int const totalTerms = isize(multiIndex) * kSpaceDim3;

  int const nodeCount = isize(nodePositions);
  auto M = RowMatrix<real>::Zero(nodeCount * kSpaceDim3, totalTerms);

  // Expansion terms are evaluated in double-precision arithmetic.
  std::vector<double> expansionTerms(multiIndex.size(), 0.0);
  for (int i = 0; i < nodeCount; ++i) {
    auto const mx = EvaluateMonomialsAt(order, nodePositions[i][0]);
    auto const my = EvaluateMonomialsAt(order, nodePositions[i][1]);
    auto const mz = EvaluateMonomialsAt(order, nodePositions[i][2]);

    // At current point, evaluate all expansion terms as product of monomials according to the
    // corresponding multiIndex entry.
    for (int k = 0; k < isize(multiIndex); ++k) {
      auto const& ind = multiIndex[k];
      expansionTerms[k] = mx[ind[0]] * my[ind[1]] * mz[ind[2]];
    }

    /*  For each node, we have the following layout:
     *
     *              0   1   2   3   4   5   6   7   8   9  10  11  ...
     *            +--------------------------------------------+
     * dofInd     | 1   0   0   x   0   0   y   0   0   z   0   0  ... |
     * dofInd + 1 | 0   1   0   0   x   0   0   y   0   0   z   0  ... |
     * dofInd + 2 | 0   0   1   0   0   x   0   0   y   0   0   z  ... |
     */
    for (int k = 0; k < kSpaceDim3; ++k) {
      int const dofInd = i * kSpaceDim3 + k;
      int colShift = k;
      std::for_each(expansionTerms.cbegin(), expansionTerms.cend(), [&](double value) {
        M(dofInd, colShift) = static_cast<real>(value);
        colShift += kSpaceDim3;
      });
    }
  }
  return M;
}
