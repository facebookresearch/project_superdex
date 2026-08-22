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
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/utils/graph.h>

namespace mochi {

/** @brief Helper for the assembly of BlockSparseMatrix into super-nodes' L. */
class AssemblyHelper {
 public:
  struct EntryAndDestination {
    int entry; //!< @brief Index in an input matrix row.
    int destination; //!< @brief Destination index in the L column.
  };
  AssemblyHelper(
      Graph<int const, int const, Span> matrixGraph,
      Graph<int const, size_t const, Span> snNodeIndices,
      Span<int const> superBounds,
      Span<int const> order,
      Span<int const> position);

  [[nodiscard]] auto FirstNode(int superNode) const {
    return _superBounds[superNode];
  }

  [[nodiscard]] auto SuperColNodeCount(int superNode) const {
    return _snNodeIndices.EdgeCount(superNode);
  }

  [[nodiscard]] auto InputRow(IsBlockSparseMatrix auto& A, int node) const;
  /// @brief Get the original matrix block placements into the L of a node
  [[nodiscard]] auto LPlacements(int node) const {
    return _entryDestination[node];
  }

  [[nodiscard]] auto SuperNodeIndices(int superNode) const {
    return _snNodeIndices[superNode];
  }

 private:
  Graph<EntryAndDestination, int> _entryDestination;
  Graph<int const, size_t const, Span> _snNodeIndices;
  Span<int const> _superBounds;
  Span<int const> _order;
  Span<int const> _position;
};

auto AssemblyHelper::InputRow(IsBlockSparseMatrix auto& A, int node) const {
  return A.Values(_order[node]);
}

} // namespace mochi
