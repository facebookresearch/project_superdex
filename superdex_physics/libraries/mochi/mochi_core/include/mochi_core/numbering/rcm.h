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

/// @brief Reverse Cuthill-McKee (RCM) ordering of a symmetric sparsity pattern, a bandwidth- and
/// profile-reducing heuristic.
///
/// @details Implements the algorithm described by George and Liu, "Computer Solution of Large
/// Sparse Positive Definite Systems", Prentice-Hall, 1981: each connected component, in increasing
/// order of its smallest vertex, is numbered breadth first from a pseudo-peripheral vertex,
/// visiting the unnumbered neighbors of each vertex by increasing degree, and the whole order is
/// reversed. The pseudo-peripheral search starts from the component's vertex of minimum degree.
/// Degrees count distinct neighbors, and ties go to the lower index, so the order depends only on
/// the edge set, not on the order of the entries within rows.
///
/// @param graph Sparsity pattern of the matrix. Diagonal and duplicate entries are ignored.
///
/// @pre Every entry of @p graph is in `[0, graph.size())`, and @p graph is structurally symmetric:
/// row `i` contains `j` if and only if row `j` contains `i`.
///
/// @return Order `perm` of length `graph.size()`: `perm[k]` is the original vertex placed `k`-th.
[[nodiscard]] DynamicArray<int> RenumberByRCM(Graph<int const, int const, Span> const& graph);

} // namespace mochi
