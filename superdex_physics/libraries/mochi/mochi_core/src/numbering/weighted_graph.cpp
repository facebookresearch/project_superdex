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

#include <mochi_core/numbering/weighted_graph.h>

namespace mochi {

WeightedGraph MakeWeightedGraph(Graph<int, int> const& graph, int weight) {
  DynamicArray<int> pointers(graph.GetPointers().begin(), graph.GetPointers().end());
  DynamicArray<WeightedEdge> targets;
  targets.reserve(graph.NumTargets());
  for (int v : graph.GetTargets()) {
    targets.push_back({.vertex = v, .weight = weight});
  }
  return WeightedGraph{std::move(pointers), std::move(targets)};
}

} // namespace mochi
