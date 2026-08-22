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

#include <mochi_core/linear_algebra/multi_frontal/elim_tree.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/graph_utils.h>
#include <mochi_core/utils/interval.h>

#include <algorithm>
#include <tuple>
#include <utility>

namespace mochi {

namespace {

/// @brief Permute values in-place so that result[position[i]] = values[i].
void Permute(Span<int> values, Span<int const> position) {
  DynamicArray<int> temp;
  temp.resize_noinit(values.size());
  for (int i = 0; i < values.size(); ++i) {
    temp[position[i]] = values[i];
  }
  for (int i = 0; i < values.size(); ++i) {
    values[i] = temp[i];
  }
}

/// @brief Compute nodal positions from a super-node permutation.
///
/// Nodes within each super-node retain their relative order, but are renumbered
/// contiguously according to the new super-node ordering.
///
/// @param superBounds Super-node bounds (size numSuper+1).
/// @param snPosition Mapping from old super-node ID to new super-node position.
/// @return Nodal position vector (old node index → new node index).
auto NodalPositionsFromSuperNodePermutation(
    DynamicArray<int> const& superBounds,
    DynamicArray<int> const& snPosition) {
  auto numSuper = static_cast<int>(snPosition.size());
  auto N = superBounds.back();

  // Build super-node order: newSN → oldSN
  auto snOrder = ReverseMap(snPosition);

  // Compute new super-node starting positions
  DynamicArray<int> newStart;
  newStart.resize_noinit(numSuper);
  int offset = 0;
  for (int newS = 0; newS < numSuper; ++newS) {
    newStart[newS] = offset;
    offset += superBounds[snOrder[newS] + 1] - superBounds[snOrder[newS]];
  }

  // Map each node to its new position
  DynamicArray<int> nodalPosition;
  nodalPosition.resize_noinit(N);
  for (int s = 0; s < numSuper; ++s) {
    auto dst = newStart[snPosition[s]];
    for (int nd = superBounds[s]; nd < superBounds[s + 1]; ++nd) {
      nodalPosition[nd] = dst + (nd - superBounds[s]);
    }
  }
  return nodalPosition;
}

/// @brief Remap super-node bounds through a super-node and nodal permutation.
///
/// Each old super-node @c s starts at nodal index @c superBounds[s]; after
/// permutation it becomes super-node @c snPosition[s] and starts at
/// @c nodalPositions[superBounds[s]]. The trailing sentinel (total node count)
/// is invariant under the permutation.
///
/// @param superBounds Old super-node bounds (size numSuper+1).
/// @param snPosition Mapping from old super-node ID to new super-node position.
/// @param nodalPositions Mapping from old node index to new node index.
/// @return New super-node bounds (size numSuper+1).
auto RemapSuperBounds(
    DynamicArray<int> const& superBounds,
    DynamicArray<int> const& snPosition,
    DynamicArray<int> const& nodalPositions) {
  auto const numSuper = static_cast<int>(snPosition.size());
  DynamicArray<int> newSuperBounds;
  newSuperBounds.resize_noinit(superBounds.size());
  for (int s = 0; s < numSuper; ++s) {
    newSuperBounds[snPosition[s]] = nodalPositions[superBounds[s]];
  }
  newSuperBounds[numSuper] = superBounds[numSuper];
  return newSuperBounds;
}

/// @brief Compute post-order positions for all nodes in a forest given as a Graph.
///
/// @param treeGraph Adjacency graph where treeGraph[node] lists children.
///                  The last entry (treeGraph[numNodes]) lists roots.
/// @param numNodes Number of real nodes (excludes the virtual root entry).
/// @return A vector mapping each node to its post-order position.
auto PostOrderPositions(Graph<int, int, DynamicArray> const& treeGraph, int numNodes) {
  DynamicArray<int> position;
  position.resize_noinit(numNodes);
  int currentPos = 0;

  struct StackFrame {
    int node;
    int childIdx;
  };

  auto roots = treeGraph[numNodes];
  DynamicArray<StackFrame> stack;
  stack.reserve(numNodes);
  for (int root : roots) {
    stack.clear();
    stack.push_back({root, 0});

    while (!stack.empty()) {
      auto& frame = stack.back();
      auto children = treeGraph[frame.node];

      if (frame.childIdx < children.size()) {
        int child = children[frame.childIdx];
        ++frame.childIdx;
        stack.push_back({child, 0});
      } else {
        position[frame.node] = currentPos++;
        stack.pop_back();
      }
    }
  }
  return position;
}

/// @brief Reorder children of a super-node in decreasing column sizes.
///
/// @param[inout] snParents Super-node parent array (will be updated)
/// @param[in] superBounds Super-node bounds (nodal indices)
/// @param[in] columnSizes Column sizes for nodal indices
/// @param[inout] assignedSuper Mapping from nodal index to super-node ID (will be updated)
///
/// @return Order of old super-node ID in new post-order numbering.
auto ReorderSuperNodeChildren(
    DynamicArray<int>& snParents,
    DynamicArray<int> const& superBounds,
    Span<int const> columnSizes,
    DynamicArray<int>& assignedSuper) {
  auto numSuper = isize(snParents);

  // Compute super-node column sizes
  DynamicArray<int> snColumnSizes;
  snColumnSizes.resize_noinit(numSuper);
  for (int s = 0; s < numSuper; ++s) {
    snColumnSizes[s] = columnSizes[superBounds[s]];
  }

  // Build temporary tree graph from snParents
  auto treeGraph = GraphFromAssignments<int, int>(
      snParents, [n = numSuper](int p) { return p == kMinusOne<int> ? n : p; });

  // Sort children of each super-node by decreasing column size
  // treeGraph[s] is mutable - we can sort it directly!
  for (int s = 0; s < numSuper; ++s) {
    std::ranges::sort(
        treeGraph[s], [&](int a, int b) { return snColumnSizes[a] > snColumnSizes[b]; });
  }

  // Assign post-order positions
  auto position = PostOrderPositions(treeGraph, numSuper);

  // Update snParents
  DynamicArray<int> newSnParents;
  newSnParents.resize_noinit(numSuper);
  for (int s = 0; s < numSuper; ++s) {
    auto newPos = position[s];
    auto p = snParents[s];
    newSnParents[newPos] = (p == kMinusOne<int>) ? kMinusOne<int> : position[p];
  }
  snParents = std::move(newSnParents);

  // Update assignedSuper
  for (int& s : assignedSuper) {
    s = position[s];
  }

  return position;
}
} // namespace

/**
 * @brief Form the parent vector.
 *
 * The parent of a node `n` is the lowest node number connected to `n` that is
 * greater than `n` in the filled graph. (see e.g.
 * https://www.cs.purdue.edu/homes/apothen/Papers/elimination-DS2004.pdf)
 * However, the input graph is not the filled graph.
 * The algorithm works by constructing a compressed representation
 * of the subtrees up to, but excluding, node `i`.
 * Given a neighbor of `i`, for which we have formed its subtree trimmed at
 * `i-1`, the known root of that subtree must have `i` as a parent.
 * To not have to traverse the complete `parent of parent` from a neighbor,
 * a vector `ancestor` is maintained, pointing directly to the largest known
 * ancestor so far of a given node.
 *
 * @param graph Matrix graph in original numbering.
 * @param order Elimination order of the nodes.
 * @param position Position of the nodes in the elimination order.
 * @return The parents of each node in the elimination tree.
 *
 * @note This code is based on work by Joseph W.H. Liu (1985)
 */
static auto FormParents(
    Graph<int const, int const, Span> const& graph,
    Span<int> const& order,
    Span<int> const& position) {
  auto N = graph.size();
  using IVector = DynamicArray<int>;
  IVector parent(N, kMinusOne<int>);
  IVector ancestor(N, kMinusOne<int>);
  for (int i = 0; i < N; ++i) {
    auto nd = order[i];
    for (auto n : graph[nd]) {
      auto neighbor = position[n];
      if (neighbor < i) {
        // For each lower neighbor, find the root of its current elimination tree.
        // Perform path compression as the subtree is traversed.
        while (ancestor[neighbor] != i && ancestor[neighbor] != kMinusOne<int>) {
          auto next = ancestor[neighbor];
          ancestor[neighbor] = i;
          neighbor = next;
        }
        if (ancestor[neighbor] == i) {
          continue;
        }
        // Now neighbor is the root of the subtree.
        // Make i the parent node of this root.
        ancestor[neighbor] = i;
        parent[neighbor] = i;
      }
    }
  }
  return parent;
}

/**
 * @brief Compute a node reordering so that it is an elimination-tree post-order.
 *
 * This function is based on an algorithm by Michel Lesoinne (2020).
 * It is quicker and takes less temporary storage than other published algorithms.
 *
 * @param parents [inout] The array of parent node in the elimination tree. Updated
 * on output to the new numbering.
 * @return A position vector for the post-ordering of the tree nodes.
 */
auto PostOrder(auto& parents) {
  auto N = parents.size();
  DynamicArray<int> subtreeSize(N, 1);
  int rootCount = 0;
  // First pass: compute the size of the subtrees rooted at each node.
  for (int i = 0; i < N; ++i) {
    auto p = parents[i];
    if (p != kMinusOne<int>) {
      subtreeSize[p] += subtreeSize[i];
    } else {
      rootCount += subtreeSize[i];
    }
  }
  DynamicArray<int> position(N);
  // Second pass: starting from the root of the tree, assign the numbering
  // as the last known index given to a descendant of the parent minus one.
  // Keep the last index in subtreeSize for indices higher than the current i.
  for (auto i = N; i-- != 0;) {
    auto p = parents[i];
    auto& lastParentDescendant = p == kMinusOne<int> ? rootCount : subtreeSize[p];
    position[i] = lastParentDescendant - 1;
    lastParentDescendant -= subtreeSize[i];
    subtreeSize[i] = position[i];
  }
  // Third pass: form the updated parent vector in subtreeSize.
  for (auto i = N; i-- != 0;) {
    auto p = parents[i];
    subtreeSize[position[i]] = p == kMinusOne<int> ? kMinusOne<int> : position[p];
  }
  parents = subtreeSize;
  return position;
}

/**
 * @brief Find the Least Common Ancestor between the previous leaf and the current
 * node.
 *
 * The current node is implicitly represented by setRep.
 *
 * @param pleaf Previous Leaf
 * @param setRep [inout] Set Representative at the previous/current level.
 * @return The Least Common Ancestor.
 */
auto FindLCA(auto pleaf, auto& setRep) {
  auto last1 = pleaf;
  auto last2 = setRep[last1];
  auto lca = setRep[last2];
  while (lca != last2) {
    setRep[last1] = lca;
    last1 = lca;
    last2 = setRep[last1];
    lca = setRep[last2];
  }
  return lca;
}

/**
 * @brief Compute the size for the columns of L.
 *
 * This function implements the algorithm described in figure 3 of
 * "An Efficient Algorithm to Compute Row and Column Counts for Sparse Cholesky Factorization".
 * Note the shift by one in indices.
 *
 * Test case from Fig 1 of https://www.osti.gov/servlets/purl/6756314
 *
 * @param graph Original matrix graph
 * @param order order of nodes after renumbering
 * @param position position of each node in order.
 * @param parent Parents of each node in the elimination tree.
 * @return The total size of L and a vector with the size of each column of L.
 */
static auto FilledGraphColumnSizes(
    Graph<int const, int const, Span> const& graph,
    Span<int> const& order,
    Span<int> const& position,
    Span<int const> parent) {
  auto N = graph.size();
  DynamicArray<int> level(N, 0);
  DynamicArray<int> childrenCount(N, 0);
  DynamicArray<int> weight(N, 1);
  // Compute level[node] and childrenCount
  for (int nd = N; --nd >= 0;) {
    auto p = parent[nd];
    if (p != kMinusOne<int>) {
      level[nd] = level[p] + 1;
      childrenCount[p] += 1;
      weight[p] = 0; // Non-leaves start with zero weight.
    }
  }
  DynamicArray<int> firstDesc(N);
  // Representative vertex of sets.
  DynamicArray<int> setRep(N, 0);
  for (int nd = 0; nd < N; ++nd) {
    firstDesc[nd] = nd;
    setRep[nd] = nd;
  }
  // Compute the first descendant.
  for (int nd = 0; nd < N; ++nd) {
    auto p = parent[nd];
    if (p != kMinusOne<int>) {
      firstDesc[p] = std::min(firstDesc[p], firstDesc[nd]);
    }
  }
  DynamicArray<int> columnSizes(N);
  DynamicArray<int> prevP(N, kMinusOne<int>); // AKA previous leaf
  DynamicArray<int> prevNbr(N, kMinusOne<int>);

  int superStart = 0; // Starting node of the current super-node.
  for (int nd = 0; nd < N; ++nd) {
    auto pr = parent[nd];
    if (pr != kMinusOne<int>) {
      --weight[pr];
    }
    bool lflag = false;
    for (auto o : graph[order[nd]]) {
      auto u = position[o];
      if (u > nd) { // Upper part of the adjacency.
        if (firstDesc[nd] > prevNbr[u]) {
          ++weight[nd];
          auto pPrime = prevP[u];
          if (pPrime != kMinusOne<int>) {
            auto q = FindLCA(pPrime, setRep);
            --weight[q];
          }
          prevP[u] = nd;
          lflag = true;
        }
        prevNbr[u] = nd;
      }
    }
    // Union
    if (lflag || childrenCount[nd] > 1) {
      superStart = nd;
    }
    setRep[superStart] = pr;
  }
  // Finalize the count
  int64_t lNZCount = 0;
  for (int nd = 0; nd < N; ++nd) {
    lNZCount += weight[nd];
    if (parent[nd] != kMinusOne<int>) {
      weight[parent[nd]] += weight[nd];
    }
  }
  return std::pair{lNZCount, std::move(weight)};
}

/**
 * @brief Reorder children nodes based on their column sizes.
 *
 * @param parents Elimination tree parent of each node.
 * @param columnSizes Size of the column for each node.
 * @return A position vector for the reordered nodes.
 */
static auto ReorderChildren(Span<int> const& parents, Span<int> const& columnSizes) {
  auto N = parents.size();
  DynamicArray<int> largestChildSize(N, 0);
  DynamicArray<int> largestChildNode(N, kMinusOne<int>);
  DynamicArray<int> descendantCount(N, 0);
  int rootCount = 0;
  DynamicArray<int> position(N, 0);
  // Find largest child of each parent and count descendants.
  for (int nd = 0; nd < N; ++nd) {
    auto prnt = parents[nd];
    if (prnt != kMinusOne<int>) {
      descendantCount[prnt] += descendantCount[nd] + 1;
      if (columnSizes[nd] > largestChildSize[prnt]) {
        largestChildSize[prnt] = columnSizes[nd];
        largestChildNode[prnt] = nd;
      }
    } else {
      rootCount += descendantCount[nd] + 1;
    }
  }
  // Assign positions to nodes.
  // For i > nd, descendantCount of a parent contains the last assigned position
  // to one of its descendants or to itself at first.
  for (auto nd = N; nd-- != 0;) {
    auto prnt = parents[nd];
    if (prnt != kMinusOne<int>) {
      if (largestChildNode[prnt] == nd) {
        position[nd] = position[prnt] - 1;
      } else {
        position[nd] = descendantCount[prnt] - 1;
        descendantCount[prnt] -= descendantCount[nd] + 1;
      }
    } else {
      position[nd] = rootCount - 1;
      rootCount -= descendantCount[nd] + 1;
    }
    descendantCount[nd] = position[nd];
    if (largestChildNode[nd] != kMinusOne<int>) {
      descendantCount[nd] -= descendantCount[largestChildNode[nd]] + 1;
    }
  }
  // Aliasing descendantCount which is not used anymore.
  auto& newParent = descendantCount;
  auto& newColCount = largestChildSize;
  for (int i = 0; i < N; ++i) {
    auto dest = position[i];
    newParent[dest] = parents[i] == kMinusOne<int> ? kMinusOne<int> : position[parents[i]];
    newColCount[dest] = columnSizes[i];
  }
  for (int i = 0; i < N; ++i) {
    parents[i] = newParent[i];
    columnSizes[i] = newColCount[i];
  }
  return position;
}

static auto FormSupernodes(Span<int> const& parents, Span<int> const& columnSizes) {
  auto N = parents.size();
  DynamicArray<int> assignedSuper;
  assignedSuper.reserve(N);
  size_t indexCount = columnSizes[0];
  int currentSuper = 0;
  assignedSuper.push_back(currentSuper);
  for (int i = 1; i < N; ++i) {
    // i belongs to the current supernode only if it is the parent of i-1 and has
    // one fewer rows than i-1.
    if (parents[i - 1] != i || columnSizes[i] != columnSizes[i - 1] - 1) {
      ++currentSuper;
      indexCount += columnSizes[i];
    }
    assignedSuper.push_back(currentSuper);
  }
  DynamicArray<int> superBounds;
  superBounds.reserve(currentSuper + 1);
  superBounds.push_back(0);
  currentSuper = 0;
  for (int i = 1; i < N; ++i) {
    if (assignedSuper[i] != currentSuper) {
      ++currentSuper;
      superBounds.push_back(i);
    }
  }
  superBounds.push_back(static_cast<int>(N));
  return std::tuple{indexCount, std::move(assignedSuper), std::move(superBounds)};
}

static auto FormSNParents(
    DynamicArray<int> const& ndParents,
    DynamicArray<int> const& assignedSN,
    DynamicArray<int> const& snBounds) {
  auto numSuper = snBounds.size() - 1;
  DynamicArray<int> snParents(numSuper);
  for (int i = 0; i < numSuper; ++i) {
    auto superLast = snBounds[i + 1] - 1;
    auto lastNodalParent = ndParents[superLast];
    snParents[i] = lastNodalParent == kMinusOne<int> ? kMinusOne<int> : assignedSN[lastNodalParent];
  }
  return snParents;
}

static auto SymbolicFactorization(
    Graph<int const, int const, Span> const& graph,
    Span<int> order,
    Span<int> position,
    DynamicArray<int> const& snParents,
    DynamicArray<int> const& assignedSuper,
    DynamicArray<int> const& superBounds,
    DynamicArray<int> const& snColSizes) {
  auto N = graph.size();
  auto numSuper = superBounds.size() - 1;
  DynamicArray<size_t> superIndicesBounds;
  superIndicesBounds.reserve(numSuper + 1);
  size_t idxCount = 0;
  for (int i = 0; i < numSuper; ++i) {
    superIndicesBounds.push_back(idxCount);
    idxCount += snColSizes[i];
  }
  superIndicesBounds.push_back(idxCount);
  // Keeping track of where we're filling the indices for each super-node.
  auto idxInsertPointers = superIndicesBounds;
  DynamicArray<int> superIndices(idxCount, kMinusOne<int>);
  DynamicArray<int> lastNode(numSuper, kMinusOne<int>);
  for (int nd = 0; nd < N; ++nd) {
    auto superNode = assignedSuper[nd];
    for (auto o : graph[order[nd]]) {
      auto u = position[o];
      if (u <= nd) {
        auto superU = assignedSuper[u];
        while (lastNode[superU] != nd) {
          // insert nd in u's indices and move to the parent
          superIndices[idxInsertPointers[superU]++] = nd;
          lastNode[superU] = nd;
          superU = snParents[superU];
          if (superU == kMinusOne<int> || superU > superNode) {
            break; // Done going towards the root of the elimination tree.
          }
        }
      }
    }
  }
  return Graph<int, size_t, DynamicArray>{std::move(superIndicesBounds), std::move(superIndices)};
}

EliminationTree::EliminationTree(
    Graph<int const, int const, Span> const& graph,
    Span<int> const& order,
    Span<int> const& position) {
  auto parents = FormParents(graph, order, position);
  auto updateOrder = [&](auto const& newPositions) {
    for (int i = 0; auto& p : position) {
      p = newPositions[p];
      order[p] = i++;
    }
  };
  // Postorder based on parents. parents is updated to match the new ordering.
  updateOrder(PostOrder(parents));

  auto [lNZCount, columnSizes] = FilledGraphColumnSizes(graph, order, position, parents);

  // Reorder the children so that the child with the largest column size is last.
  // This increases the chance that this child will form a larger super-node with its parent.
  updateOrder(ReorderChildren(parents, columnSizes));

  auto [numIndices, superAssignment, superBounds] = FormSupernodes(parents, columnSizes);

  _snParents = FormSNParents(parents, superAssignment, superBounds);

  // Reorder super-node children by decreasing column size.
  // This maintains post-order while ordering siblings by column size.
  auto snPosition = ReorderSuperNodeChildren(_snParents, superBounds, columnSizes, superAssignment);
  auto nodalPositions = NodalPositionsFromSuperNodePermutation(superBounds, snPosition);
  updateOrder(nodalPositions);
  Permute(columnSizes, nodalPositions);
  Permute(superAssignment, nodalPositions);
  superBounds = RemapSuperBounds(superBounds, snPosition, nodalPositions);

  auto nSuper = _snParents.size();
  _leafDistance.resize(nSuper + 1, 0);
  for (size_t sn = 0; sn < _snParents.size(); ++sn) {
    auto snParent = _snParents[sn] == kMinusOne<int> ? nSuper : _snParents[sn];
    _leafDistance[snParent] = std::max(_leafDistance[snParent], _leafDistance[sn] + 1);
  }

  // Populate supernode column sizes
  _snColSizes.resize(nSuper);
  for (size_t sn = 0; sn < nSuper; ++sn) {
    _snColSizes[sn] = columnSizes[superBounds[sn]];
  }

  _snBounds = std::move(superBounds);

  _treeGraph = GraphFromAssignments<int, int>(
      _snParents, [n = isize(_snParents)](int p) { return p == kMinusOne<int> ? n : p; });
}

Interval<int> EliminationTree::SubtreeRange(int superNode) const {
  auto graph = TreeGraph();
  auto start = superNode;
  while (graph.EdgeCount(start) != 0) {
    start = graph[start][0];
  }
  return {start, superNode + 1};
}

FactorMetrics EliminationTree::ComputeFactorMetrics(int dofsPerNode) const {
  double flops = 0.0;
  size_t storage = 0;
  for (int sn = 0; sn < NumSuperNodes(); ++sn) {
    auto const nInSuper = static_cast<double>(SuperSize(sn) * dofsPerNode);
    auto const superLRows = static_cast<double>(_snColSizes[sn] * dofsPerNode);
    auto const nCoupling = superLRows - nInSuper;

    // Sum of diagonal factorization, application of the diagonal to the lower rectangle and
    // Schur complement.
    flops += (2.0 / 3.0) * nInSuper * nInSuper * nInSuper + nInSuper * nInSuper * nCoupling +
        nCoupling * nCoupling * nInSuper;

    storage += static_cast<size_t>(nInSuper * superLRows - nInSuper * (nInSuper - 1) / 2);
  }
  return {flops, storage};
}

SymbolicEliminationTree::SymbolicEliminationTree(
    Graph<int const, int const, Span> const& graph,
    Span<int> const& order,
    Span<int> const& position)
    : EliminationTree(graph, order, position) {
  auto N = graph.size();
  auto numSuper = NumSuperNodes();
  DynamicArray<int> superAssignment(N);
  for (int sn = 0; sn < numSuper; ++sn) {
    for (int i = _snBounds[sn]; i < _snBounds[sn + 1]; ++i) {
      superAssignment[i] = sn;
    }
  }

  _snIndices = SymbolicFactorization(
      graph, order, position, _snParents, superAssignment, _snBounds, _snColSizes);
}

} // namespace mochi
