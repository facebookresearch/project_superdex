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

#include <mochi_core/linear_algebra/matrix_fwd.h>
#include <mochi_physics/utils/mochi_physics_macros.h>

#include <vector>

namespace mochi::rom::polynomial_crom {

/*
 * Generates and returns a vector of 3D multi-indices (k, j, i) such that the sum k + j + i (i.e.,
 * the total polynomial order) is less than or equal to the specified 'order'. This corresponds to
 * all monomials of the form x^k * y^j * z^i with total degree <= order.
 *
 * The returned vector is sorted by total degree (i.e., k + j + i), so that lower-order terms appear
 * first but no guarantee is made about the order of terms with the same total degree. In other
 * words, it is guaranteed that all terms with total degree 0 appear before all terms with total
 * degree 1, and so on. But for example for order = 2, the order of (1,0,1) and (0,1,1) is not
 * guaranteed.
 *
 * Example:
 *   For order = 2, the function may return:
 *   (0,0,0), (1,0,0), (0,1,0), (0,0,1), (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)
 */
[[nodiscard]] MOCHI_API std::vector<Int3> ComputeMultiIndexSortedByTotalOrder(int order);

/*
 * Constructs and returns a basis matrix of polynomials up to a specified total order computed from
 * a tensor product of monomials.
 *
 * For each node position (x, y, z), the function evaluates all monomials x^k * y^j * z^i for (k, j,
 * i) such that k + j + i <= order. These are evaluated at the nodePositions, and the resulting
 * values are used to populate a matrix. The output matrix M has dimensions (nodeCount * 3) x (3 *
 * numberOfMonomials), where the factor of 3 accounts for each spatial dimension.
 *
 * Example:
 * --------
 * 2 nodes, order = 1 -> monomials = {1, x, y, z}
 *
 * totalTerms = 4 monomials x 3 components = 12 columns
 * rows       = 2 nodes x 3 components     = 6 rows
 *
 * Let monomials evaluated at node0 = [1, b, c, d]
 * and monomials evaluated at node1 = [1, f, g, h]
 * Then the returned matrix looks like:
 *
 *         0   1   2   3   4   5   6   7   8   9  10  11
 *       +---------------------------------------------+
 * row 0 | 1   0   0   b   0   0   c   0   0   d   0   0 | node0, dof0
 * row 1 | 0   1   0   0   b   0   0   c   0   0   d   0 | node0, dof1
 * row 2 | 0   0   1   0   0   b   0   0   c   0   0   d | node0, dof2
 * row 3 | 1   0   0   f   0   0   g   0   0   h   0   0 | node1, dof0
 * row 4 | 0   1   0   0   f   0   0   g   0   0   h   0 | node1, dof1
 * row 5 | 0   0   1   0   0   f   0   0   g   0   0   h | node1, dof2
 *       +---------------------------------------------+
 */
[[nodiscard]] MOCHI_API RowMatrix<real> CreateBasisMatrix(
    int order,
    Span<Real3 const> nodePositions);

} // namespace mochi::rom::polynomial_crom
