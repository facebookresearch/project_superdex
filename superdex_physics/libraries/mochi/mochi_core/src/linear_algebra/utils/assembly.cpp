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

#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/linear_algebra/utils/assembly.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph_alg.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/range_algorithms.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <utility>

namespace mochi {

/// @brief Target average element count per @ref NodalBasedStructure group.
///
/// @details @ref GreedyDecompose is parameterized by nodes per group, but assembly work is driven
/// mostly by the elements processed by each group. @ref ComputeTargetNodesPerGroup converts this
/// element target to a node target using the unique-referenced-node per element ratio of the
/// assembly connectivity.
///
/// @note A more adaptive policy could choose this target from an estimated groups-per-color budget:
/// estimate useful assembler threads from element count and per-element cost, reserve enough groups
/// per color for load balancing, estimate color count from the effective stencil width, then
/// convert the resulting elements-per-group target back to nodes-per-group. This may improve
/// performance for large meshes at high thread counts, but would make @ref NodalBasedStructure
/// construction depend on assembly timing assumptions.
static constexpr int kTargetElementsPerGroup = 30;

/// @brief Minimum node group size passed to @ref GreedyDecompose.
static constexpr int kMinNodesPerGroup = 1;

/// @brief Maximum node group size passed to @ref GreedyDecompose.
static constexpr int kMaxNodesPerGroup = 128;

// Note that this function assumes indices to be sorted. This is guaranteed in the context of
// NodalBasedStructure, where its inputs (rows of NodalBasedStructure::_nToN) are explicitly sorted.
[[nodiscard]] static int FindIndex(int n, Span<int const> indices) {
  MOCHI_ASSERT_VERBOSE(std::ranges::is_sorted(indices), "Indices are not sorted");
  auto const* it = std::ranges::lower_bound(indices, n);
  MOCHI_ASSERT(it != indices.end() && (*it == n), "Index not found.");
  return static_cast<int>(it - indices.begin());
}

// `nToE` is the reverse of the assembly connectivity. Counting non-empty rows gives the number of
// unique referenced nodes (repeated pad nodes in extended stencils do not inflate the count).
[[nodiscard]] static int CountReferencedNodes(Graph<int, int> const& nToE) {
  int numReferencedNodes = 0;
  for (auto&& nodeElements : nToE) {
    numReferencedNodes += nodeElements.targets.empty() ? 0 : 1;
  }
  return numReferencedNodes;
}

static int ComputeTargetNodesPerGroup(int numElements, int numReferencedNodes) {
  if (numElements <= 0 || numReferencedNodes <= 0) {
    return kMinNodesPerGroup;
  }

  int64_t const targetNodes =
      (int64_t{kTargetElementsPerGroup} * numReferencedNodes + numElements / 2) / numElements;
  return Clamp(static_cast<int>(targetNodes), kMinNodesPerGroup, kMaxNodesPerGroup);
}

NodalBasedStructure::NodalBasedStructure(Graph<int, int> eToN, Graph<int, int> const& nToN)
    : _eToN(std::move(eToN)) {
  _nToE = Reverse<int, int>(_eToN);

#if MOCHI_ASSERT_VERBOSE_ENABLED
  if (nToN.size() != 0) {
    MOCHI_ASSERT_VERBOSE(
        nToN.size() >= _nToE.size(), "nToN must cover all nodes referenced by eToN.");
    for (auto const elemNodes : _eToN) {
      for (int ei : elemNodes.targets) {
        auto const row = nToN[ei];
        MOCHI_ASSERT_VERBOSE(std::ranges::is_sorted(row), "nToN row is not sorted.");
        for (int ej : elemNodes.targets) {
          MOCHI_ASSERT_VERBOSE(
              std::ranges::binary_search(row, ej), "nToN is missing a node pair required by eToN.");
        }
      }
    }
  }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

  _nToN = nToN.size() == 0 ? Traverse(_nToE, _eToN).SortTargets() : nToN;

  int const numElements = isize(_eToN);
  // Find the maximum number of nodes per element first, for allocation and lookup purposes.
  int maxNodesPerElement = 0;
  for (auto const elemNodes : _eToN) {
    maxNodesPerElement = Max(maxNodesPerElement, isize(elemNodes.targets));
  }
  _maxNodesPerElementSquared = maxNodesPerElement * maxNodesPerElement;
  _nodeSparseIndices.reserve(numElements * _maxNodesPerElementSquared);
  for (auto const elemNodes : _eToN) {
    for (int ei : elemNodes.targets) {
      auto columnIndices = _nToN[ei];
      for (int ej : elemNodes.targets) {
        _nodeSparseIndices.push_back(FindIndex(ej, columnIndices));
      }
    }
    // Padding to max size for O(1) access of each element's sparse indices without additional layer
    // of indirection to look up pointers into the array.
    while (isize(_nodeSparseIndices) % _maxNodesPerElementSquared != 0) {
      // The these padding indices should never be accessed, so the value can be arbitrary, but
      // using a known sentinel value can be helpful for debugging.
      _nodeSparseIndices.push_back(kSentinelIndex);
    }
  }

  int const nodeGroupSize = ComputeTargetNodesPerGroup(numElements, CountReferencedNodes(_nToE));
  auto nDec = GreedyDecompose(
      /*nToN without sorting targets*/ Traverse(_nToE, _eToN),
      /*subdomainSize*/ nodeGroupSize);
  _elemGroups = AggregateElements(nDec, _nToE, isize(_eToN));

  CreateSubTasks();
  CreateDepMasks();

  // Consistency checks.
#if MOCHI_ASSERT_VERBOSE_ENABLED
  MOCHI_ASSERT_VERBOSE(isize(_groupDependsOn) == isize(_dependentGroups));
  for (auto iG : _elemGroupColoring[0]) {
    MOCHI_ASSERT_VERBOSE(
        isize(_groupDependsOn[iG]) == 0,
        "Groups in the first color must not depend on other groups.");
  }

  for (int iG = 0; iG < isize(_groupDependsOn); ++iG) {
    if (isize(_groupDependsOn[iG]) == 0) {
      bool found = false;
      for (auto jG : _elemGroupColoring[0]) {
        if (iG == jG) {
          found = true;
          break;
        }
      }
      MOCHI_ASSERT_VERBOSE(found, "Groups without dependencies must be in the first color.");
    }

    for (auto jG : _groupDependsOn[iG]) {
      bool found = false;
      for (auto kG : _dependentGroups[jG]) {
        if (iG == kG) {
          found = true;
          break;
        }
      }
      MOCHI_ASSERT_VERBOSE(found, "Inconsistent group dependencies.");
    }

    for (auto jG : _dependentGroups[iG]) {
      bool found = false;
      for (auto kG : _groupDependsOn[jG]) {
        if (iG == kG) {
          found = true;
          break;
        }
      }
      MOCHI_ASSERT_VERBOSE(found, "Inconsistent group dependencies.");
    }
  }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
}

// For triangle meshes without bending:
template NodalBasedStructure::NodalBasedStructure(Span<NdArray<int, 3> const> const& connectivity);
// For tetrahedral meshes:
template NodalBasedStructure::NodalBasedStructure(Span<NdArray<int, 4> const> const& connectivity);
// For triangle meshes with bending:
template NodalBasedStructure::NodalBasedStructure(Span<NdArray<int, 6> const> const& connectivity);

void NodalBasedStructure::CreateDepMasks() {
  GraphBuilder<details::AtomicDetails, int> builder(
      _groupDependsOn.size() + 1, _groupDependsOn.NumTargets());
  for ([[maybe_unused]] auto [i, ngSet] : _groupDependsOn) {
    details::AtomicDetails last{0, UINT32_MAX};
    builder.StartSet();
    for (auto ng : ngSet) {
      auto ad = AtomicIdx(ng);
      if (ad.index != last.index) {
        if (last.index != UINT32_MAX) {
          builder.InsertTarget(last);
        }
        last = ad;
      } else {
        last.mask |= ad.mask;
      }
    }
    if (last.index != UINT32_MAX) {
      builder.InsertTarget(last);
    }
  }
  _depMasks = builder.Build();
  _initialReady.resize((_elemGroups.size() + kMaskBits - 1) / kMaskBits, 0);
  for (auto g : _elemGroupColoring[0]) {
    auto [mask, idx] = AtomicIdx(g);
    _initialReady[idx] |= mask;
  }
}

/**
 * Creating a reduced graph based on coloring:
 *  Easy:
 *    - Connections to subs that are in a later color should be removed.
 *    - Connections to subs in the immediate previous color should be kept.
 *  Difficult:
 *    - Connections to subs in colors before the previous that are reachable through the reduced
 *      graph above should be removed. What is the cost of finding if a sub is reachable? It is at
 *      most the number of edges for each vertex. In practice, it is much less.
 */
static auto CreateDependencyGraph(Graph<int, int> const& coloring, Graph<int, int> const& vToV) {
  auto vtxNumbering = coloring.GetTargets();
  auto subMap = ReverseMap(vtxNumbering);
  auto numSubs = vtxNumbering.size();
  DynamicArray<int> vtxColor(coloring.NumTargets(), -1);
  for (auto [color, subs] : coloring) {
    for (auto s : subs) {
      vtxColor[s] = color;
    }
  }
  GraphBuilder<int, int> colorDepBuilder(isize(subMap), vToV.NumTargets());
  DynamicArray<int> deps;
  DynamicArray<int> activeDeps;
  DynamicArray<int> visitList; // Array of starting vertices for a given color.
  DynamicArray<int> visitedFor(numSubs, -1);
  // Subs of first color have no dependencies
  for ([[maybe_unused]] auto _ : coloring[0]) {
    colorDepBuilder.append(deps);
  }
  for (int color = 1; color < coloring.size(); ++color) {
    for (auto v : coloring[color]) {
      deps.clear();
      // Only neighbors of lower color than the current color are possible.
      for (auto n : vToV[v]) {
        if (vtxColor[n] < color) {
          deps.push_back(n);
        }
      }
      {
        // Remove dependencies that are transitively satisfied by higher-indexed color
        // dependencies.
        // Allow to consider a vertex only once for each v
        auto accept = [&visitedFor, v, &vtxColor, color](int n) {
          if (vtxColor[n] < color && visitedFor[n] != v) {
            visitedFor[n] = v;
            return true;
          }
          return false;
        };
        // The dependency graph is a subgraph of vToV, having only edges to lower color targets.
        // There is also no need to search through a vertex more than once.
        auto acceptEdge = [&](int from, int to) {
          if (vtxColor[to] < vtxColor[from] && visitedFor[to] != v) {
            visitedFor[to] = v;
            return true;
          }
          return false;
        };
        // The graph is examined by neighbors of decreasing color index.
        std::ranges::sort(deps, [&vtxColor](int a, int b) { return vtxColor[a] > vtxColor[b]; });
        activeDeps.clear();
        auto* b = deps.begin();
        auto* e = deps.end();
        while (b != e) { // Traverse the graph color by color of the possible dependencies
          visitList.clear(); // List of vertices from which breadth-first search will be done
          // Find all the the vertices of the same color as the first.
          auto* sameEnd = std::find_if(
              b, e, [&vtxColor, c = vtxColor[*b]](auto v) { return vtxColor[v] != c; });
          for (; b != sameEnd; ++b) {
            // Add them to the actual dependencies and visit list unless they were already
            // satisfied.
            if (accept(*b)) {
              activeDeps.push_back(*b);
              visitList.push_back(*b);
            }
          }
          BreadthFirstSearch(vToV, visitList, acceptEdge);
        }
      }
      colorDepBuilder.append(activeDeps);
    }
  }
  // The graph sources have an implicit map given by the color ordering
  auto byColorDependencies = colorDepBuilder.Build();
  // We need to reorder the sources of this graph by vertex number. The current graph
  // is for vertices in the coloring order.
  auto const& graphSrcOrder = coloring.GetTargets();
  DynamicArray<int> invOrder(graphSrcOrder.size());
  for (int i = 0; i < graphSrcOrder.size(); ++i) {
    invOrder[graphSrcOrder[i]] = i;
  }
  GraphBuilder<int, int> finalGraph(
      byColorDependencies.size() + 1, byColorDependencies.NumTargets());
  for (auto io : invOrder) {
    finalGraph.append(byColorDependencies[io]);
  }
  return finalGraph.Build();
}

/** @brief Create groupings via greedy decompositions.
 * @details The decomposition is first done on a nodal basis. In the second step, elements are
 * aggregated based to one of the nodal subdomain that they touch. The order of aggregation is done
 * to avoid tiny subdomain. Some nodal subdomains have only one node. Under the assumption of a
 * connected graph, as long as they are considered last, no elements will be assigned to them and
 * thus they are entirely absorbed into bigger subdomains. This fact can easily be shown to be
 * always the case because none of these tiny nodal subdomain can be interconnected since their size
 * is lower than the stop size of greedy. If they were interconnected, the greedy of the first of
 * two hypothetically interconnected subdomain would have continued with the nodes of the second.
 * QED.
 */
void NodalBasedStructure::CreateSubTasks() {
  auto sToN = Traverse(_elemGroups, _eToN);
  _elemGtoG = Traverse(sToN, Reverse(sToN)).SortTargets();
  _groupColor = SmallestLastColoring(_elemGtoG);
  _elemGroupColoring = GraphFromAssignments<int, int>(_groupColor);
  _groupDependsOn = CreateDependencyGraph(_elemGroupColoring, _elemGtoG);
  _dependentGroups = Reverse<int, int>(_groupDependsOn, isize(_elemGroups));

  // TODO: Investigate group ordering to improve cache locality / load balance in parallel assembly.
}

void WorkState::Complete(int group, Span<uint64_t> taskReadyMask) {
  auto [mask, uIdx] = AtomicIdx(group);
  // Sequentially consistent ordering to ensure (1) at least one thread seeing 'isReady' as true,
  // and (2) visibility of side effects in threads executing dependent groups.
  finished[uIdx].fetch_or(mask, std::memory_order_seq_cst);
  for (auto depGr : dependents[group]) {
    bool isReady = true;
    for (auto [depMask, depIdx] : dependsOn[depGr]) {
      isReady &= ((finished[depIdx].load(std::memory_order_acquire) & depMask) ^ depMask) == 0;
    }
    if (isReady) {
      auto [rMask, rIdx] = AtomicIdx(depGr);
      ready[rIdx].fetch_or(rMask, std::memory_order_release);
      if (taskReadyMask) {
        taskReadyMask[rIdx] |= rMask;
      }
    }
  }
}

MOCHI_FORCE_INLINE static int BitIndex(uint64_t v) {
  MOCHI_ASSERT_VERBOSE(std::has_single_bit(v), "Expected mask with a single bit.");
  return std::countr_zero(v);
}

int WorkState::Acquire(Span<uint64_t> taskReadyMask) {
  for (int idx = 0; idx < taskReadyMask.size(); ++idx) {
    uint64_t mask = taskReadyMask[idx];
    while (mask != 0) {
      auto best = mask & ((~mask) + 1);
      mask ^= best;
      auto previous = acquired[idx].fetch_or(best, std::memory_order_relaxed);
      if (previous & best) { // Make sure nobody acquired it yet.
        continue;
      }
      taskReadyMask[idx] = mask;
      return idx * kMaskBits + BitIndex(best);
    }
    taskReadyMask[idx] = mask;
  }
  return AcquireAny();
}

int WorkState::AcquireAny() {
  int idx = 0;
  int doneCount = 0;
  do {
    auto msk = ready[idx].load(std::memory_order_relaxed);
    if (msk == 0) {
      auto doneMask = acquired[idx].load(std::memory_order_relaxed);
      // The last `acquired` is initialized with unused bit set to 1 so that this works.
      doneCount += (~doneMask == 0) ? 1 : 0;
      idx = idx + 1;
      if (idx == isize(ready)) {
        if (doneCount == isize(ready)) {
          return -1;
        }
        doneCount = 0;
        idx = 0;
        MOCHI_NOP_50();
        continue;
      }
      continue;
    }
    msk = msk & ((~msk) + 1);
    auto previous = ready[idx].fetch_and(~msk, std::memory_order_acquire);
    if (msk & previous) { // If non zero, we may try to acquire
      previous = acquired[idx].fetch_or(msk, std::memory_order_relaxed);
      if (previous & msk) { // Make sure nobody acquired it yet.
        continue;
      }
      return idx * kMaskBits + BitIndex(msk);
    }
  } while (true);
}

} // namespace mochi
