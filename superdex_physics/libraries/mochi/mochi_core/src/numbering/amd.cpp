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
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/indexed_heap.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <type_traits>

namespace mochi {

namespace {

using Idx = int;

constexpr auto kInvalidIdx = kMinusOne<Idx>;

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
/// keeps them in the order @ref DetectSupervariables sorts them into, by adjacency. Cleanup is
/// lazy: merged variables stay in other variables' lists until a walk prunes them, and absorbed
/// elements are pruned in the step that absorbs them. An abandoned slot starts with minus its size,
/// so @ref CompactStorage can skip it.
///
/// Unlike the paper, a binary heap replaces the degree lists, supervariables are detected by
/// sorting candidates by adjacency, and dense rows are not set aside.
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

  /// @brief In @p node's adjacency, move @p pivot from the variables to the end of the elements.
  void MoveToElements(Idx node, Idx pivot);

  /// @brief Compute |Le \ Lp| for every live element `e` adjacent to Lp (Algorithm 2 of the paper).
  void ComputeExternalDegrees(Idx pivot);

  /// @brief Walk the variables of Lp, mass-eliminating those whose only remaining adjacency is
  /// pivot's element and bounding the degree of the others.
  void AbsorbAndUpdateNeighbors(Idx pivot);

  /// @brief Compact @p var's adjacency, absorbing elements whose pattern is a subset of Lp.
  ///
  /// @details Appends @p pivot to the elements, so all variables of Lp end up with comparable
  /// adjacency lists.
  ///
  /// @return |Ai \ i| + Σ |Le \ Lp| over the remaining elements `e` other than @p pivot: an upper
  /// bound on @p var's external degree outside Lp, since element patterns can overlap.
  [[nodiscard]] Idx WalkAndCompactNeighborhood(Idx var, Idx pivot);

  /// @brief Merge the indistinguishable variables of Lp into supervariables.
  void DetectSupervariables(Idx pivot);

  /// @brief Finish the approximate degree of the variables of Lp and update the degree heap.
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

  /// @brief Last pivot that marked each node: the variables of Lp, and the elements visited by
  /// @ref ComputeExternalDegrees.
  DynamicArray<Idx> _mark;

  /// @brief Approximate external degree of each variable.
  DynamicArray<Idx> _approximateDegree;

  /// @brief |Le| of each element `e`.
  DynamicArray<Idx> _elementDegree;

  /// @brief |Le \ Lp| of each element `e` adjacent to Lp, for the current pivot.
  DynamicArray<Idx> _externalDegree;

  /// @brief Principal variables not eliminated yet, ordered by approximate degree.
  IndexedHeap<Idx, Idx> _degreeHeap;

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
    : _mark(graph.size(), kInvalidIdx), _degreeHeap(graph.size()), _superSize(graph.size(), 1) {
  Idx const n = graph.size();
  _nodeInfo.resize_noinit(n);
  _nodeStart.resize_noinit(n);
  _approximateDegree.resize_noinit(n);
  _elementDegree.resize_noinit(n);
  _externalDegree.resize_noinit(n);
  _superParent.resize_noinit(n);
  _order.reserve(n);
  _remainingVars = n;

  // Live data never grows: a new element fits in the slots of the pivot and the elements it
  // absorbs. So compacted data takes at most n + NumTargets entries, and a new element at most
  // n + 1.
  _dataSpace.resize_noinit(
      static_cast<size_t>(graph.NumTargets()) + 2 * static_cast<size_t>(n) + 1);

  for (Idx i = 0; i < n; ++i) {
    _nodeStart[i] = _tail;
    _dataSpace[_tail++] = i;
    for (Idx const j : graph[i]) {
      if (j != i) {
        _dataSpace[_tail++] = j;
      }
    }

    // Sorted variables make indistinguishable variables cheap to find, and duplicates adjacent.
    Idx* const begin = _dataSpace.data() + _nodeStart[i] + 1;
    std::sort(begin, _dataSpace.data() + _tail);
    Idx* const end = std::unique(begin, _dataSpace.data() + _tail);
    _tail = static_cast<Ptr>(end - _dataSpace.data());

    Idx const numVariables = static_cast<Idx>(end - begin);
    _nodeInfo[i] = {.numElements = 0, .numVariables = numVariables, .slotSize = numVariables + 1};
    _approximateDegree[i] = numVariables;
    if (numVariables == 0) { // Isolated vertices are eliminated right away.
      --_tail;
      _nodeStart[i] = kInvalidPtr;
      _order.push_back(i);
      --_remainingVars;
    } else {
      _degreeHeap.Insert(i, numVariables);
    }
  }
}

DynamicArray<Idx> QuotientGraph::ComputeOrdering() {
  while (!_degreeHeap.IsEmpty()) {
    Idx const pivot = _degreeHeap.ExtractMin().key;
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
  // Mark pivot's variables as members of Lp. CreateElement marks the variables of its elements.
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
      MoveToElements(var, pivot);
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

  // Pivot's own variables, already marked by Eliminate.
  for (Idx const var : Variables(pivot)) {
    if (IsPrincipal(var)) {
      lpSize += _superSize[var];
      MoveToElements(var, pivot);
      _dataSpace[_tail++] = var;
      ++numVariables;
    }
  }

  for (Idx const e : Elements(pivot)) {
    for (Idx const var : Variables(e)) {
      // Pivot is a variable of its elements, but not of Lp.
      if (var != pivot && IsPrincipal(var) && !InPivotElement(var, pivot)) {
        lpSize += _superSize[var];
        _mark[var] = pivot;
        _dataSpace[_tail++] = var;
        ++numVariables;
      }
    }
    AbandonNode(e);
  }

  MOCHI_ASSERT_VERBOSE(_tail <= start + maxSlotSize, "Lp outgrew the pivot's approximate degree.");
  AbandonNode(pivot);
  _nodeInfo[pivot] = {.numElements = 0, .numVariables = numVariables, .slotSize = numVariables + 1};
  _nodeStart[pivot] = start;

  return lpSize;
}

void QuotientGraph::MoveToElements(Idx const node, Idx const pivot) {
  // Rotating pivot to the front of the sorted variables keeps them sorted and puts pivot last among
  // the elements.
  auto vars = Variables(node);
  auto* const it = std::ranges::find(vars, pivot);
  MOCHI_ASSERT_VERBOSE(it != vars.end(), "Pivot must be a variable of the node.");
  std::rotate(vars.begin(), it, it + 1);
  ++_nodeInfo[node].numElements;
  --_nodeInfo[node].numVariables;
}

void QuotientGraph::ComputeExternalDegrees(Idx const pivot) {
  // |Le \ Lp| = |Le| - |Le ∩ Lp|. Pivot and the elements it absorbed are dropped by the walk.
  for (Idx const var : Variables(pivot)) {
    for (Idx const e : Elements(var)) {
      if (e == pivot || IsAbsorbed(e)) {
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
      _approximateDegree[var] = std::min(_approximateDegree[var], partial);
      *kept++ = var;
    }
  }
  _nodeInfo[pivot].numVariables = static_cast<Idx>(kept - vars.begin());
}

Idx QuotientGraph::WalkAndCompactNeighborhood(Idx const var, Idx const pivot) {
  // A single forward sweep writes the kept elements, pivot, then the kept variables. Writes never
  // overtake reads: pivot takes the entry of a dropped element, either pivot itself, for its direct
  // neighbors, or the absorbed element that made var a member of Lp.
  NodeHeader& header = _nodeInfo[var];
  Idx* const elemBegin = NeighborNodes(var).data();
  Idx* const elemEnd = elemBegin + header.numElements;
  Idx* const varEnd = elemEnd + header.numVariables;
  Idx partial = 0;
  Idx* out = elemBegin;
  for (Idx* in = elemBegin; in != elemEnd; ++in) {
    Idx const e = *in;
    if (e == pivot || IsAbsorbed(e)) {
      continue;
    }
    if (_externalDegree[e] == 0) {
      // Aggressive absorption: Le is a subset of Lp, so pivot's element covers e.
      AbandonNode(e);
      continue;
    }
    partial += _externalDegree[e];
    *out++ = e;
  }
  MOCHI_ASSERT_VERBOSE(out < elemEnd, "No element entry left for the pivot.");
  *out++ = pivot;

  Idx* const varBegin = out;
  for (Idx* in = elemEnd; in != varEnd; ++in) {
    Idx const v = *in;
    // Merged variables are represented by their principal, and members of Lp by pivot.
    if (!IsPrincipal(v) || InPivotElement(v, pivot)) {
      continue;
    }
    partial += _superSize[v];
    *out++ = v;
  }
  header.numElements = static_cast<Idx>(varBegin - elemBegin);
  header.numVariables = static_cast<Idx>(out - varBegin);

  return partial;
}

void QuotientGraph::DetectSupervariables(Idx const pivot) {
  // Variables with the same adjacency are indistinguishable. Sort the variables of Lp by adjacency,
  // then merge each run of equal adjacencies into its first variable.
  auto vars = Variables(pivot);
  std::ranges::sort(vars, [this](Idx a, Idx b) {
    auto const na = NeighborNodes(a);
    auto const nb = NeighborNodes(b);
    return na.size() != nb.size() ? na.size() < nb.size()
                                  : std::ranges::lexicographical_compare(na, nb);
  });

  auto* kept = vars.begin();
  for (Idx const var : vars) {
    if (kept != vars.begin() && NeighborNodes(kept[-1]) == NeighborNodes(var)) {
      JoinVarTo(var, kept[-1]);
    } else {
      *kept++ = var;
    }
  }
  _nodeInfo[pivot].numVariables = static_cast<Idx>(kept - vars.begin());
}

void QuotientGraph::UpdateDegrees(Idx const pivot) {
  // The walk left min(the previous degree, |Ai \ i| + Σ |Le \ Lp|) in _approximateDegree.
  // Equation (4) of the paper adds |Lp \ i| and bounds the result by the number of other variables
  // not eliminated yet.
  Idx const lpSize = _elementDegree[pivot];
  for (Idx const var : Variables(pivot)) {
    Idx const varSize = _superSize[var];
    Idx const degree =
        std::min(_approximateDegree[var] + lpSize - varSize, _remainingVars - varSize);
    _approximateDegree[var] = degree;
    _degreeHeap.UpdateCost(var, degree);
  }
}

void QuotientGraph::JoinVarTo(Idx const var, Idx const principal) {
  _superParent[var] = principal;
  _superSize[principal] += _superSize[var];
  _superSize[var] = 0;
  _degreeHeap.Delete(var);
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
