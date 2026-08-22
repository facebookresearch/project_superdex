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

#include <mochi_core/linear_algebra/multi_frontal/assembly_helper.h>
#include <mochi_core/utils/graph_utils.h>

namespace mochi {

AssemblyHelper::AssemblyHelper(
    Graph<int const, int const, Span> matrixGraph,
    Graph<int const, size_t const, Span> snNodeIndices,
    Span<int const> superBounds,
    Span<int const> order,
    Span<int const> position)
    : _snNodeIndices(snNodeIndices), _superBounds(superBounds), _order(order), _position(position) {
  int numNodes = superBounds.back();
  // The matrix is symmetric and half of the non-diagonal terms are not assembled.
  auto numTargets = (2 * matrixGraph.NumTargets() + numNodes) / 2;
  GraphBuilder<EntryAndDestination, int> graphBuilder(superBounds.back() + 1, numTargets);
  for (auto [sn, snIndices] : snNodeIndices) {
    auto lIndices = snNodeIndices[sn];
    auto lBegin = lIndices.begin();
    auto lEnd = lIndices.end();
    for (int nd = superBounds[sn]; nd < superBounds[sn + 1]; ++nd) {
      graphBuilder.StartSet();
      auto aRow = order[nd];
      for (int aIdx{0}; auto aCol : matrixGraph[aRow]) {
        auto lNd = position[aCol];
        if (lNd >= nd) {
          auto it = std::find(lBegin, lEnd, lNd);
          MOCHI_ASSERT_VERBOSE(it != lIndices.end());
          auto lIdx = static_cast<int>(it - lBegin);
          graphBuilder.InsertTarget({aIdx, lIdx});
        }
        ++aIdx;
      }
      ++lBegin;
    }
  }
  _entryDestination = graphBuilder.Build();
}

} // namespace mochi
