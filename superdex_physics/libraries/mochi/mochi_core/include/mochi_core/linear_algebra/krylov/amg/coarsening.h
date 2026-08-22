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

#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>

#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace mochi::krylov::details {

template <typename Range, typename Idx>
bool Unassigned(Range const& r, DynamicArray<Idx> const& f, Idx const flag) {
  for (auto v : r) {
    if (f[v] != flag) {
      return false;
    }
  }
  return true;
}

/** @details This function makes use of the algorithm described in
 * Algebraic multigrid by smoothed aggregation for second and fourth order elliptic problems
 * by Vanek, Mandel and Brezina. It differs from that particular paper in that it considers
 * all graph neighbors to be strongly coupled.
 * The shape functions are computed based on the nodal graph only, without regard to the matrix
 * values. The shape functions are applied to each displacement degree of freedom equally.
 * The result should work properly as long as there isn't a strong anisotropy.
 *
 * The Graph type is required to have:
 *   size() : number of vertices
 *   operator[](int v) : obtain a range for vertex v
 *
 * @tparam Graph
 * @param g An undirected symmetric graph including (v,v) diagonal edges.
 * @return A pair with a vector of assignments to an aggregate and the number of aggregates.
 */
template <typename Graph>
auto Aggregate(Graph const& g) {
  auto numVertices = g.size();
  using Idx = decltype(numVertices);
  // Mark all vertices as unassigned.
  auto const unassignedFlag = std::numeric_limits<Idx>::max();
  MOCHI_ASSERT_VERBOSE(numVertices < unassignedFlag, "Too many vertices");
  DynamicArray<Idx> assigned(numVertices, unassignedFlag);
  Idx agg = 0;

  // Step 1: Select disjoint neighborhoods.
  for (Idx v = 0; v < numVertices; ++v) {
    if ((assigned[v] == unassignedFlag) && Unassigned(g[v], assigned, unassignedFlag)) {
      for (Idx w : g[v]) {
        assigned[w] = agg;
      }
      // If we could guarantee that v is in g[v], the next lines could be simplified
      assigned[v] = agg;
      ++agg;
    }
  }

  // Step 2: Add each remaining vertex to one of the sets already selected to which it
  // is connected
  auto const firstPassAggCount = agg;
  for (Idx v = 0; v < numVertices; ++v) {
    if (assigned[v] == unassignedFlag) {
      for (Idx neighbor : g[v]) {
        if (assigned[neighbor] < firstPassAggCount) {
          assigned[v] = assigned[neighbor];
          break;
        }
      }
    }
  }

  // Step 3: Group the remaining vertices into aggregates that consist of subsets of
  // coupled neighborhoods
  /// TODO Add a unit test where this step is required.
  for (Idx v = 0; v < numVertices; ++v) {
    if (assigned[v] == unassignedFlag) {
      assigned[v] = agg;
      for (Idx neighbor : g[v]) {
        if (assigned[neighbor] == unassignedFlag) {
          assigned[neighbor] = agg;
        }
      }
      ++agg;
    }
  }
  return std::pair{std::move(assigned), agg};
}

/**
 * @brief Form the P matrix of smooth coarse functions.
 * @details The formula used derives from Vanek, Mandel and Brezina and guarantees that the coarse
 * shape functions form a partition of unity.
 *
 * @return The sparse interpolation matrix of fine vertex interpolation from coarse ones.
 */
template <
    typename Scalar,
    int kDofsPerNode,
    typename InputIdx,
    template <typename, typename...> typename Storage,
    typename Index>
auto Smoothing(
    BlockSparseMatrix<Scalar, kDofsPerNode, InputIdx, InputIdx, Storage> const& A,
    int nCoarse,
    DynamicArray<Index> const& partition,
    std::remove_const_t<Scalar> omega) {
  static_assert(std::is_same_v<Index const, InputIdx const>, "Incompatible Integers");
  using Idx = std::remove_const_t<InputIdx>;
  using NonConstScalar = std::remove_const_t<Scalar>;
  //
  auto nToN = AsGraphView(A);
  auto const n = nToN.size();
  SparseMatrix<NonConstScalar, Idx, Idx> S(n, nToN);
  for (Idx i = 0; i < n; ++i) {
    auto const colIdx = S.Indices(i);
    auto values = S.Values(i);
    if (isize(colIdx) == 1) {
      values[0] = Scalar(1.0);
      continue;
    }
    auto ratio = omega / Scalar(isize(colIdx) - 1.0);
    for (int k = 0; k < isize(colIdx); ++k) {
      values[k] = (colIdx[k] == i) ? Scalar(1.0 - omega) : ratio;
    }
  }
  // Build the tentative prolongator based on piecewise-constant interpolation
  DynamicArray<Idx> pptr;
  pptr.resize_noinit(n + 1);
  pptr[0] = 0;
  DynamicArray<Idx> pcIdx;
  pcIdx.reserve(n);
  for (Idx i = 0; i < n; ++i) {
    // Note that the next lines assume that every node will belong
    // to exactly one coarse patch
    pptr[i + 1] = pptr[i] + 1;
    pcIdx.push_back(partition[i]);
  }
  DynamicArray<NonConstScalar> pvalues(pcIdx.size(), NonConstScalar(1.0));
  SparseMatrix<NonConstScalar, Idx, Idx> P(
      nCoarse, std::move(pptr), std::move(pcIdx), std::move(pvalues));
  // Smooth the tentative prolongator
  auto newP = S * P;
  return newP;
}

} // namespace mochi::krylov::details
