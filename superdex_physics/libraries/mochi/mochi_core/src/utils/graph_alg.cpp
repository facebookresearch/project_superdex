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

#include <mochi_core/utils/graph_alg.h>
#include <mochi_core/utils/graph_utils.h>
#include <mochi_core/utils/interval.h>

#include <algorithm>
#include <vector>

namespace mochi {

template <typename Graph>
static auto GreedyGrow(
    Graph const& graph,
    DynamicArray<bool>& unused,
    DynamicArray<int>& group,
    int seed,
    int targetSize) {
  auto nextIdx = group.size();
  group.push_back(seed);
  unused[seed] = false;
  while (group.size() < targetSize && nextIdx < group.size()) {
    for (auto v : graph[group[nextIdx]]) {
      if (unused[v]) {
        unused[v] = false;
        group.push_back(v);
      }
    }
    ++nextIdx;
  }
  return nextIdx;
}

Graph<int, int> GreedyDecompose(Graph<int, int> const& nToN, int size) {
  auto numNodes = nToN.size();
  DynamicArray<bool> unusedNode(numNodes, true);
  DynamicArray<int> subNodes;
  subNodes.reserve(2 * size);
  DynamicArray<int> nodes;
  nodes.reserve(numNodes);
  int useCount = 0;
  int nextCheck = 0;
  GraphBuilder<int, int> builder((numNodes + size - 1) / size, numNodes);
  // TODO(@micles): Find a pseudo diametral node to start.
  while (useCount != numNodes) {
    auto candidates = Interval<int>{nextCheck, numNodes};
    auto seed = *find(candidates, [&](auto n) { return unusedNode[n]; });
    nextCheck = seed + 1;
    while (seed != -1) {
      auto next = GreedyGrow(nToN, unusedNode, subNodes, seed, size);
      // Find a potential next seed neighboring the current elements
      for (seed = -1; seed == -1 && next < subNodes.size(); ++next) {
        for (auto ngb : nToN[subNodes[next]]) {
          if (unusedNode[ngb]) {
            seed = ngb;
            break; // Double loop break.
          }
        }
      }
      builder.append(subNodes);
      useCount += isize(subNodes);
      subNodes.clear();
    }
  }
  return builder.Build();
}

Graph<int, int>
AggregateElements(Graph<int, int> const& nodeDec, Graph<int, int> const& nToE, int numEle) {
  // Put node groups at or below a minimum size threshold at the end. The threshold was chosen
  // empirically from sweeps over soft/shell/rod tet, tri, and segment assembly. 4 was in the middle
  // of a robust, near-optimal performance plateau across mesh sizes and thread counts.
  constexpr int kTinySize = 4;
  auto order = Interval<int>{nodeDec.size()}.to<DynamicArray<int>>();
  auto* itTiny = partition(order, [&nodeDec](int s) { return nodeDec[s].size() > kTinySize; });
  std::sort(order.begin(), itTiny, [&nodeDec](auto a, auto b) {
    return nodeDec[a].size() < nodeDec[b].size();
  });
  GraphBuilder<int, int> decBuilder(nodeDec.size(), numEle);
  DynamicArray<bool> assigned(numEle, false);
  DynamicArray<int> elements;
  for (auto& it : order) {
    for (auto n : nodeDec[it]) {
      for (auto e : nToE[n]) {
        if (!assigned[e]) {
          assigned[e] = true;
          elements.push_back(e);
        }
      }
    }
    if (!elements.empty()) {
      decBuilder.append(elements);
      elements.clear();
    }
  }
  return decBuilder.Build();
}

} // namespace mochi
