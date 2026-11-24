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

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/span.h>

namespace mochi {

/// @brief Approximate minimum degree (AMD) fill-reducing ordering of a symmetric sparsity pattern.
///
/// @details Implements the algorithm of Amestoy, Davis and Duff, "An Approximate Minimum Degree
/// Ordering Algorithm", SIAM J. Matrix Anal. Appl. 17(4), 1996, without dense-row detection.
///
/// @param graph Sparsity pattern of the matrix. Diagonal and duplicate entries are ignored.
///
/// @pre @p graph is structurally symmetric: row `i` contains `j` if and only if row `j` contains
/// `i`.
///
/// @return Elimination order `perm` of length `graph.size()`: `perm[k]` is the original vertex
/// eliminated `k`-th. Unlike SuiteSparse's `amd_order`, it isn't postordered.
[[nodiscard]] DynamicArray<int> RenumberByAMD(Graph<int const, int const, Span> const& graph);

} // namespace mochi
