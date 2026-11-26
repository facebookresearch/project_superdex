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

#include <mochi_core/numbering/rcm.h>

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>

#include <algorithm>
#include <utility>

namespace mochi {

DynamicArray<int> RenumberByRCM(Graph<int const, int const, Span> const& graph) {
  int const n = graph.size();

  // Count distinct neighbors other than the vertex itself, marking the vertex and its neighbors
  // with the vertex.
  DynamicArray<int> degree(n, 0);
  DynamicArray<int> marks(n, -1);
  for (int v = 0; v < n; ++v) {
    marks[v] = v;
    for (int const w : graph[v]) {
      if (marks[w] != v) {
        marks[w] = v;
        ++degree[v];
      }
    }
  }

  auto const byDegree = [&](int a, int b) {
    return std::pair{degree[a], a} < std::pair{degree[b], b};
  };

  DynamicArray<int> perm;
  perm.reserve(n);
  DynamicArray<bool> isVisited(n, false);

  // Replace perm[componentBegin:], which holds the previous search of root's component, if any,
  // with a breadth-first search from root that visits the neighbors of each vertex by increasing
  // degree. The finalized prefix stays marked as visited: each search clears and rebuilds only the
  // current component's suffix. Return the number of levels of the search and the vertex of minimum
  // degree in its last level.
  auto const search = [&](int root, int componentBegin) {
    for (int i = componentBegin; i < isize(perm); ++i) {
      isVisited[perm[i]] = false;
    }
    perm.resize(componentBegin);

    perm.push_back(root);
    isVisited[root] = true;
    int numLevels = 0;
    int levelBegin = componentBegin;
    for (int head = componentBegin, levelEnd = componentBegin; head < isize(perm); ++head) {
      if (head == levelEnd) {
        ++numLevels;
        levelBegin = head;
        levelEnd = isize(perm);
      }
      int const neighborsBegin = isize(perm);
      for (int const w : graph[perm[head]]) {
        if (!isVisited[w]) {
          isVisited[w] = true;
          perm.push_back(w);
        }
      }
      std::sort(perm.begin() + neighborsBegin, perm.end(), byDegree);
    }

    return std::pair{numLevels, *std::min_element(perm.begin() + levelBegin, perm.end(), byDegree)};
  };

  for (int v = 0; v < n; ++v) {
    if (isVisited[v]) {
      continue;
    }

    // Collect the component with a search from v, then search from its vertex of minimum degree,
    // then from the vertex of minimum degree in the last level while the number of levels grows.
    // The last search, rooted at a pseudo-peripheral vertex, numbers the component.
    int const componentBegin = isize(perm);
    search(v, componentBegin);
    int root = *std::min_element(perm.begin() + componentBegin, perm.end(), byDegree);
    for (int previousLevels = 0;;) {
      auto const [numLevels, candidate] = search(root, componentBegin);
      if (numLevels <= previousLevels) {
        break;
      }
      previousLevels = numLevels;
      root = candidate;
    }
  }

  MOCHI_ASSERT_VERBOSE(
      isize(perm) == n, "The order is incomplete: the graph must be structurally symmetric.");
  std::ranges::reverse(perm);
  return perm;
}

} // namespace mochi
