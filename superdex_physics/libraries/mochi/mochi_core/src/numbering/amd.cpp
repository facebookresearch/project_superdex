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

#include <mochi_core/numbering/amd.h>

#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/range_algorithms.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

namespace mochi {

namespace {

using Idx = int;

constexpr auto kInvalidIdx = kMinusOne<Idx>;

/// @brief Variables bucketed by approximate degree.
///
/// @details Each bucket is a doubly linked list with the most recently inserted variable first, so
/// insertion and removal take O(1).
class DegreeLists {
 public:
  /// @param n Number of variables. Degrees must be in `[0, n)`.
  explicit DegreeLists(Idx n) : _head(n, kInvalidIdx) {
    _next.resize_noinit(n);
    _prev.resize_noinit(n);
    _degree.resize_noinit(n);
  }

  /// @pre @p var is not in the lists.
  void Insert(Idx var, Idx degree) {
    MOCHI_ASSERT_VERBOSE(degree >= 0 && degree < isize(_head), "Degree out of range.");
    _degree[var] = degree;
    _prev[var] = kInvalidIdx;
    _next[var] = _head[degree];
    if (_next[var] != kInvalidIdx) {
      _prev[_next[var]] = var;
    }
    _head[degree] = var;
    _minDegree = Min(_minDegree, degree);
  }

  /// @pre @p var is in the lists.
  void Remove(Idx var) {
    if (_prev[var] == kInvalidIdx) {
      _head[_degree[var]] = _next[var];
    } else {
      _next[_prev[var]] = _next[var];
    }
    if (_next[var] != kInvalidIdx) {
      _prev[_next[var]] = _prev[var];
    }
  }

  /// @brief Remove and return the most recently inserted variable of minimum degree.
  ///
  /// @pre The lists are not empty.
  Idx ExtractMin() {
    while (_head[_minDegree] == kInvalidIdx) {
      ++_minDegree;
      MOCHI_ASSERT_VERBOSE(_minDegree < isize(_head), "The degree lists are empty.");
    }
    Idx const var = _head[_minDegree];
    Remove(var);
    return var;
  }

 private:
  /// @brief First variable of each degree, or `kInvalidIdx`.
  DynamicArray<Idx> _head;
  DynamicArray<Idx> _next;
  DynamicArray<Idx> _prev;
  DynamicArray<Idx> _degree;

  /// @brief Lower bound on the smallest degree present.
  Idx _minDegree = 0;
};

/// @brief Quotient graph of a symmetric sparsity pattern during its symbolic elimination.
///
/// @details Nodes start as variables. Eliminating a variable, the pivot `p`, turns it into an
/// element whose pattern `Lp` is the set of variables it couples, and absorbs the elements adjacent
/// to it. `Le` denotes the variables of element `e`, and `Ai` the variables adjacent to variable
/// `i`. Indistinguishable variables are merged into supervariables, represented by their principal
/// variable; sizes and degrees count variables.
///
/// Each live node's adjacency is stored in `_dataSpace` as its own index, then its elements, in
/// elimination order, then its variables. A variable keeps its variables sorted, and an element
/// keeps them in the order Lp was built. Cleanup is lazy: merged variables stay in other variables'
/// lists until a walk prunes them, absorbed elements are pruned in the step that absorbs them, and
/// a pivot stays among its direct neighbors' variables until their walks in its step. An abandoned
/// slot starts with minus its size, so @ref CompactStorage can skip it.
///
/// Dense rows are not set aside.
class QuotientGraph {
 public:
  explicit QuotientGraph(Graph<int const, int const, Span> const& graph);

  /// @brief Eliminate all variables and return the elimination order of the original vertices.
  [[nodiscard]] DynamicArray<Idx> ComputeOrdering();

 private:
  using Ptr = size_t;
  static_assert(std::is_signed_v<Idx>, "Abandoned slots store minus their size.");

  static constexpr auto kInvalidPtr = kMinusOne<Ptr>;

  struct NodeHeader {
    Idx numElements;
    Idx numVariables;
    /// @brief Number of entries in the node's slot, its own index included.
    Idx slotSize;
  };

  [[nodiscard]] bool IsPrincipal(Idx i) const {
    return _superSize[i] != 0;
  }

  /// @brief Return true once element @p e has been absorbed into another element.
  [[nodiscard]] bool IsAbsorbed(Idx e) const {
    return _nodeStart[e] == kInvalidPtr;
  }

  [[nodiscard]] Span<Idx, Idx> NeighborNodes(Idx i) {
    MOCHI_ASSERT_VERBOSE(
        _nodeStart[i] != kInvalidPtr && _dataSpace[_nodeStart[i]] == i, "Node is not live.");
    NodeHeader const& header = _nodeInfo[i];
    return {_dataSpace.data() + _nodeStart[i] + 1, header.numElements + header.numVariables};
  }

  [[nodiscard]] Span<Idx, Idx> Elements(Idx i) {
    return NeighborNodes(i).subspan(0, _nodeInfo[i].numElements);
  }

  [[nodiscard]] Span<Idx, Idx> Variables(Idx i) {
    return NeighborNodes(i).subspan(_nodeInfo[i].numElements);
  }

  /// @brief Return true if principal @p var is @p pivot or a variable of Lp.
  [[nodiscard]] bool InPivotElement(Idx var, Idx pivot) const {
    return _mark[var] == pivot;
  }

  /// @brief Return true the first time @p pivot visits element @p e.
  bool FirstVisit(Idx e, Idx pivot) {
    bool const first = _mark[e] != pivot;
    _mark[e] = pivot;
    return first;
  }

  /// @brief Make @p pivot an element.
  void Eliminate(Idx pivot);

  /// @brief Turn @p pivot, which has no adjacent element, into an element in place.
  ///
  /// @return |Lp|.
  [[nodiscard]] Idx CreateElementInPlace(Idx pivot);

  /// @brief Turn @p pivot into a new element at the end of `_dataSpace`, absorbing its elements.
  ///
  /// @return |Lp|.
  [[nodiscard]] Idx CreateElement(Idx pivot);

  /// @brief Compute |Le \ Lp| for every live element `e` in the element lists of Lp's variables.
  void ComputeExternalDegrees(Idx pivot);

  /// @brief Walk the variables of Lp, mass-eliminating those whose only remaining adjacency is
  /// pivot's element and bounding the degree of the others.
  void AbsorbAndUpdateNeighbors(Idx pivot);

  /// @brief Compact @p var's adjacency, absorbing elements whose pattern is a subset of Lp.
  ///
  /// @details Appends @p pivot to the elements, so all variables of Lp end up with comparable
  /// adjacency lists, and hashes the rest of the adjacency for @ref DetectSupervariables.
  ///
  /// @return |Ai \ i| + Σ |Le \ Lp| over the remaining elements `e` other than @p pivot: an upper
  /// bound on @p var's external degree outside Lp, since element patterns can overlap.
  [[nodiscard]] Idx WalkAndCompactNeighborhood(Idx var, Idx pivot);

  /// @brief Merge the indistinguishable variables of Lp into supervariables.
  void DetectSupervariables(Idx pivot);

  /// @brief Finish the approximate degree of the variables of Lp and update the degree lists.
  void UpdateDegrees(Idx pivot);

  /// @brief Merge @p var into the supervariable of @p principal.
  void JoinVarTo(Idx var, Idx principal);

  /// @brief Release @p node's slot in `_dataSpace`.
  void AbandonNode(Idx node);

  /// @brief Move the live adjacency lists to the front of `_dataSpace`.
  void CompactStorage();

  /// @brief Expand the order of elimination steps into the order of the original vertices.
  [[nodiscard]] DynamicArray<Idx> ExpandedOrder();

  /// @brief Element and variable counts and slot size of each node.
  DynamicArray<NodeHeader> _nodeInfo;

  /// @brief Adjacency of all live nodes (see the class documentation).
  DynamicArray<Idx> _dataSpace;

  /// @brief Start of each node's slot in `_dataSpace`, or `kInvalidPtr` once abandoned.
  DynamicArray<Ptr> _nodeStart;

  /// @brief End of the used part of `_dataSpace`.
  Ptr _tail = 0;

  /// @brief Last pivot that marked each node: the pivot itself, its variables, the variables of Lp,
  /// and the elements visited by @ref ComputeExternalDegrees.
  DynamicArray<Idx> _mark;

  /// @brief Approximate external degree of each variable.
  DynamicArray<Idx> _approximateDegree;

  /// @brief |Le| of each element `e`.
  DynamicArray<Idx> _elementDegree;

  /// @brief |Le \ Lp| of each element `e` adjacent to Lp, for the current pivot.
  DynamicArray<Idx> _externalDegree;

  /// @brief Principal variables not eliminated yet, bucketed by approximate degree.
  DegreeLists _degreeLists;

  /// @brief Sum of the neighbor indices of each variable of Lp, other than pivot, after its walk.
  DynamicArray<uint32_t> _adjacencyHash;

  /// @brief First variable of each hash bucket, or `kInvalidIdx`. Empty between pivots. The number
  /// of buckets is a power of two, so the low bits of a hash pick its bucket.
  DynamicArray<Idx> _bucketHead;

  /// @brief Next variable in the hash bucket of each variable of Lp.
  DynamicArray<Idx> _bucketNext;

  /// @brief Principal of each elimination step, in elimination order.
  DynamicArray<Idx> _order;

  /// @brief Principal that each merged variable joined.
  DynamicArray<Idx> _superParent;

  /// @brief Number of variables in each principal's supervariable, or 0 once merged.
  DynamicArray<Idx> _superSize;

  /// @brief Number of variables not eliminated yet.
  Idx _remainingVars = 0;
};

QuotientGraph::QuotientGraph(Graph<int const, int const, Span> const& graph)
    : _mark(graph.size(), kInvalidIdx),
      _degreeLists(graph.size()),
      _bucketHead(NextPowerOfTwo(static_cast<uint32_t>(graph.size())), kInvalidIdx),
      _superSize(graph.size(), 1) {
  Idx const n = graph.size();
  _nodeInfo.resize_noinit(n);
  _nodeStart.resize_noinit(n);
  _approximateDegree.resize_noinit(n);
  _elementDegree.resize_noinit(n);
  _externalDegree.resize_noinit(n);
  _adjacencyHash.resize_noinit(n);
  _bucketNext.resize_noinit(n);
  _superParent.resize_noinit(n);
  _order.reserve(n);
  _remainingVars = n;

  // Live data never grows: a new element fits in the slots of the pivot and the elements it
  // absorbs. So compacted data takes at most n + NumTargets entries, and a new element at most
  // n + 1.
  _dataSpace.resize_noinit(
      static_cast<size_t>(graph.NumTargets()) + 2 * static_cast<size_t>(n) + 1);

  for (Idx i = 0; i < n; ++i) {
    Ptr const start = _tail;
    _dataSpace[start] = i;
    Ptr end = start + 1;
    bool increasing = true;
    Idx previous = kInvalidIdx;
    for (Idx const j : graph[i]) {
      if (j != i) {
        increasing &= j > previous;
        previous = j;
        _dataSpace[end++] = j;
      }
    }

    // Sorted variables make indistinguishable variables cheap to compare, and duplicates adjacent.
    // Rows are usually sorted and without duplicates already, so only the others are sorted.
    if (!increasing) {
      auto const variables = MakeSpan(_dataSpace).subspan(start + 1, end - start - 1);
      std::ranges::sort(variables);
      end -= std::ranges::unique(variables).size();
    }

    auto const numVariables = static_cast<Idx>(end - start - 1);
    _nodeInfo[i] = {.numElements = 0, .numVariables = numVariables, .slotSize = numVariables + 1};
    _approximateDegree[i] = numVariables;
    if (numVariables == 0) { // Isolated vertices are eliminated right away.
      _nodeStart[i] = kInvalidPtr;
      _order.push_back(i);
      --_remainingVars;
    } else {
      _nodeStart[i] = start;
      _tail = end;
      _degreeLists.Insert(i, numVariables);
    }
  }
}

DynamicArray<Idx> QuotientGraph::ComputeOrdering() {
  // The degree lists hold the principals not eliminated yet, whose sizes sum to _remainingVars.
  while (_remainingVars > 0) {
    Idx const pivot = _degreeLists.ExtractMin();
    Eliminate(pivot);
    ComputeExternalDegrees(pivot);
    AbsorbAndUpdateNeighbors(pivot);
    DetectSupervariables(pivot);
    UpdateDegrees(pivot);
  }

  return ExpandedOrder();
}

void QuotientGraph::Eliminate(Idx const pivot) {
  _order.push_back(pivot);
  // Mark pivot and its variables, so CreateElement skips them among the variables of its elements,
  // and the walks drop them.
  _mark[pivot] = pivot;
  for (Idx const var : Variables(pivot)) {
    _mark[var] = pivot;
  }
  _elementDegree[pivot] =
      Elements(pivot).empty() ? CreateElementInPlace(pivot) : CreateElement(pivot);
  _remainingVars -= _superSize[pivot];
}

Idx QuotientGraph::CreateElementInPlace(Idx const pivot) {
  // Lp is the principal variables of pivot: compact them in place. The slot keeps its size, which
  // AbandonNode and CompactStorage need.
  auto vars = Variables(pivot);
  auto* kept = vars.begin();
  Idx lpSize = 0;
  for (Idx const var : vars) {
    if (IsPrincipal(var)) {
      lpSize += _superSize[var];
      *kept++ = var;
    }
  }
  _nodeInfo[pivot].numVariables = static_cast<Idx>(kept - vars.begin());

  return lpSize;
}

Idx QuotientGraph::CreateElement(Idx const pivot) {
  // Lp is the union of the principal variables of pivot and of its elements, minus pivot. Each
  // principal adds at least one to |Lp|, which pivot's approximate degree bounds.
  Ptr const maxSlotSize = 1 + static_cast<Ptr>(_approximateDegree[pivot]);
  if (_tail + maxSlotSize > _dataSpace.size()) {
    CompactStorage();
  }
  MOCHI_ASSERT_VERBOSE(
      _tail + maxSlotSize <= _dataSpace.size(), "Quotient graph storage overflow.");

  Ptr const start = _tail;
  _dataSpace[_tail++] = pivot;
  Idx numVariables = 0;
  Idx lpSize = 0;

  // Lp lists the variables of the elements, newest element first, then pivot's own variables.
  // UpdateDegrees reinserts them in this order at the front of the degree lists, which decides
  // degree ties.
  for (Idx const e : ReverseRange(Elements(pivot))) {
    for (Idx const var : Variables(e)) {
      if (IsPrincipal(var) && !InPivotElement(var, pivot)) {
        lpSize += _superSize[var];
        _mark[var] = pivot;
        _dataSpace[_tail++] = var;
        ++numVariables;
      }
    }
    AbandonNode(e);
  }

  for (Idx const var : Variables(pivot)) {
    if (IsPrincipal(var)) {
      lpSize += _superSize[var];
      _dataSpace[_tail++] = var;
      ++numVariables;
    }
  }

  MOCHI_ASSERT_VERBOSE(_tail <= start + maxSlotSize, "Lp outgrew the pivot's approximate degree.");
  AbandonNode(pivot);
  _nodeInfo[pivot] = {.numElements = 0, .numVariables = numVariables, .slotSize = numVariables + 1};
  _nodeStart[pivot] = start;

  return lpSize;
}

void QuotientGraph::ComputeExternalDegrees(Idx const pivot) {
  // |Le \ Lp| = |Le| - |Le ∩ Lp|. The elements pivot absorbed are dropped by the walk.
  for (Idx const var : Variables(pivot)) {
    for (Idx const e : Elements(var)) {
      if (IsAbsorbed(e)) {
        continue;
      }
      if (FirstVisit(e, pivot)) {
        _externalDegree[e] = _elementDegree[e];
      }
      _externalDegree[e] -= _superSize[var];
    }
  }
}

void QuotientGraph::AbsorbAndUpdateNeighbors(Idx const pivot) {
  auto vars = Variables(pivot);
  auto* kept = vars.begin();
  for (Idx const var : vars) {
    Idx const partial = WalkAndCompactNeighborhood(var, pivot);
    if (partial == 0) {
      // Mass elimination: pivot's element is var's only remaining adjacency, so var is eliminated
      // with pivot.
      _remainingVars -= _superSize[var];
      _elementDegree[pivot] -= _superSize[var];
      JoinVarTo(var, pivot);
    } else {
      _approximateDegree[var] = Min(_approximateDegree[var], partial);
      *kept++ = var;
    }
  }
  _nodeInfo[pivot].numVariables = static_cast<Idx>(kept - vars.begin());
}

Idx QuotientGraph::WalkAndCompactNeighborhood(Idx const var, Idx const pivot) {
  // A single forward sweep writes the kept elements, pivot, then the kept variables, each variable
  // one entry late, so each write lands at or before the entry just read. The adjacency loses at
  // least one entry (pivot itself, for its direct neighbors, or the absorbed element that made var
  // a member of Lp), which leaves room for pivot among the elements.
  NodeHeader& header = _nodeInfo[var];
  Idx* const elemBegin = NeighborNodes(var).data();
  Idx* const elemEnd = elemBegin + header.numElements;
  Idx* const varEnd = elemEnd + header.numVariables;
  Idx partial = 0;
  uint32_t hash = 0;
  Idx* out = elemBegin;
  for (Idx* in = elemBegin; in != elemEnd; ++in) {
    Idx const e = *in;
    if (IsAbsorbed(e)) {
      continue;
    }
    if (_externalDegree[e] == 0) {
      // Aggressive absorption: Le is a subset of Lp, so pivot's element covers e.
      AbandonNode(e);
      continue;
    }
    partial += _externalDegree[e];
    hash += static_cast<uint32_t>(e);
    *out++ = e;
  }
  auto const numElements = static_cast<Idx>(out - elemBegin) + 1;

  Idx pending = pivot;
  for (Idx* in = elemEnd; in != varEnd; ++in) {
    Idx const v = *in;
    // Merged variables are represented by their principal, and members of Lp by pivot.
    if (!IsPrincipal(v) || InPivotElement(v, pivot)) {
      continue;
    }
    partial += _superSize[v];
    hash += static_cast<uint32_t>(v);
    *out++ = std::exchange(pending, v);
  }
  MOCHI_ASSERT_VERBOSE(out < varEnd, "No entry left for the pivot.");
  *out++ = pending;
  header.numElements = numElements;
  header.numVariables = static_cast<Idx>(out - elemBegin) - numElements;
  _adjacencyHash[var] = hash;

  return partial;
}

void QuotientGraph::DetectSupervariables(Idx const pivot) {
  // Variables with the same adjacency are indistinguishable, and have the same hash. Bucket the
  // variables of Lp by hash, then merge each bucket's variables of equal adjacency into the one
  // that comes last in Lp, since buckets are LIFO. Lp keeps its order, which decides degree ties.
  auto vars = Variables(pivot);
  auto const bucketMask = static_cast<uint32_t>(_bucketHead.size()) - 1;
  for (Idx const var : vars) {
    Idx& head = _bucketHead[_adjacencyHash[var] & bucketMask];
    _bucketNext[var] = head;
    head = var;
  }

  for (Idx const var : vars) {
    // The first variable of a bucket to come up empties it.
    Idx principal = std::exchange(_bucketHead[_adjacencyHash[var] & bucketMask], kInvalidIdx);
    for (; principal != kInvalidIdx; principal = _bucketNext[principal]) {
      auto const adjacency = NeighborNodes(principal);
      Idx previous = principal;
      while (_bucketNext[previous] != kInvalidIdx) {
        Idx const other = _bucketNext[previous];
        if (NeighborNodes(other) == adjacency) {
          _bucketNext[previous] = _bucketNext[other];
          JoinVarTo(other, principal);
        } else {
          previous = other;
        }
      }
    }
  }

  auto* kept = vars.begin();
  for (Idx const var : vars) {
    if (IsPrincipal(var)) {
      *kept++ = var;
    }
  }
  _nodeInfo[pivot].numVariables = static_cast<Idx>(kept - vars.begin());
}

void QuotientGraph::UpdateDegrees(Idx const pivot) {
  // The walk left min(the previous degree, |Ai \ i| + Σ |Le \ Lp|) in _approximateDegree.
  // Adding |Lp \ i| completes the approximate degree, bounded by the number of other variables not
  // eliminated yet, which also keeps it below n, the size of the degree lists.
  Idx const lpSize = _elementDegree[pivot];
  for (Idx const var : Variables(pivot)) {
    Idx const varSize = _superSize[var];
    Idx const degree = Min(_approximateDegree[var] + lpSize - varSize, _remainingVars - varSize);
    _approximateDegree[var] = degree;
    _degreeLists.Remove(var);
    _degreeLists.Insert(var, degree);
  }
}

void QuotientGraph::JoinVarTo(Idx const var, Idx const principal) {
  _superParent[var] = principal;
  _superSize[principal] += _superSize[var];
  _superSize[var] = 0;
  _degreeLists.Remove(var);
  AbandonNode(var);
}

void QuotientGraph::AbandonNode(Idx const node) {
  _dataSpace[_nodeStart[node]] = -_nodeInfo[node].slotSize;
  _nodeStart[node] = kInvalidPtr;
}

void QuotientGraph::CompactStorage() {
  Ptr dest = 0;
  for (Ptr src = 0; src < _tail;) {
    Idx const node = _dataSpace[src];
    if (node < 0) { // Abandoned slot of size -node.
      src += static_cast<Ptr>(-node);
      continue;
    }

    NodeHeader& header = _nodeInfo[node];
    Idx const length = 1 + header.numElements + header.numVariables;
    std::memmove(
        _dataSpace.data() + dest,
        _dataSpace.data() + src,
        static_cast<size_t>(length) * sizeof(Idx));
    _nodeStart[node] = dest;
    src += static_cast<Ptr>(header.slotSize);
    header.slotSize = length;
    dest += static_cast<Ptr>(length);
  }
  _tail = dest;
}

DynamicArray<Idx> QuotientGraph::ExpandedOrder() {
  // Each elimination step takes the next positions for its principal's supervariable: merged
  // variables first, in index order, then the principal.
  Idx const n = isize(_nodeStart);
  DynamicArray<Idx> position;
  position.resize_noinit(n);
  Idx next = 0;
  for (Idx const principal : _order) {
    position[principal] = next;
    next += _superSize[principal];
  }

  for (Idx i = 0; i < n; ++i) {
    if (IsPrincipal(i)) {
      continue;
    }

    Idx principal = _superParent[i];
    while (!IsPrincipal(principal)) {
      principal = _superParent[principal];
    }

    // Compress the path, so other variables of the supervariable find their principal at once.
    for (Idx j = i; j != principal;) {
      Idx const parent = _superParent[j];
      _superParent[j] = principal;
      j = parent;
    }
    position[i] = position[principal]++;
  }

  return InversePermutation(position);
}

} // namespace

DynamicArray<int> RenumberByAMD(Graph<int const, int const, Span> const& graph) {
  return QuotientGraph(graph).ComputeOrdering();
}

} // namespace mochi
