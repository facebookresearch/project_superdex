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

#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/range_algorithms.h>
#include <mochi_core/utils/span.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace mochi {

/// @brief Compute a greedy coloring of the input graph
/// The algorithm visits the vertices of the input graph in sequence
/// (here in non-increasing order of degree) and assigns each vertex
/// its first available color.
/// The algorithm finds the coloring in linear time but it may not
/// use the minimum number of colors possible.
///
/// @param[in] Input graph 'g'
///
/// @return Graph from color to vertices of 'g'
///
template <typename Idx, typename Ptr, template <typename, typename...> class Storage>
[[nodiscard]] auto GreedyColoring(Graph<Idx, Ptr, Storage> const& g) {
  using NonConstIdx = std::remove_const_t<Idx>;
  using NonConstPtr = std::remove_const_t<Ptr>;

  auto const n = g.size();
  NonConstIdx maxCount = 0;
  for (NonConstIdx i = 0; i < n; ++i) {
    maxCount = Max(maxCount, g.EdgeCount(i));
  }

  DynamicArray<int> degreeLen(maxCount + 1, 0);
  for (NonConstIdx i = 0; i < n; ++i) {
    degreeLen[g.EdgeCount(i)] += 1;
  }

  DynamicArray<int> color(n, -1);
  int maxColorID = -1;

  {
    DynamicArray<DynamicArray<NonConstIdx>> degreeList(maxCount + 1);
    for (NonConstIdx count = 0; count <= maxCount; ++count) {
      degreeList[count].reserve(degreeLen[count]);
    }
    for (NonConstIdx i = 0; i < n; ++i) {
      degreeList[g.EdgeCount(i)].push_back(i);
    }

    // When the graph has maximum degree, 'maxCount', at most 'maxCount + 1' colors are needed
    DynamicArray<bool> available(maxCount + 1, true);

    for (int i = maxCount; i >= 0; --i) {
      // Treat the "nodes" ordered according to the number of connections
      for (auto node : degreeList[i]) {
        auto list = g[node];
        // Update array of available colors
        for (auto neighbor : list) {
          if (color[neighbor] == -1) {
            continue;
          }
          available[color[neighbor]] = false;
        }
        //
        auto ptr = std::ranges::find(available, true);
        auto myColor = int(ptr - available.begin());
        color[node] = myColor;
        maxColorID = Max(maxColorID, myColor);
        // Reset array of available colors
        std::ranges::fill(available, true);
      }
    }
  }

  // Re-use memory from degreeLen
  Span<int> colorStored{degreeLen.data(), size_t(maxColorID) + 1};
  int numColors = maxColorID + 1;
  DynamicArray<NonConstPtr> pointer(numColors + 1, 0);
  for (auto icolor : color) {
    pointer[icolor + 1] += 1;
  }
  for (int i = 0; i < numColors; ++i) {
    colorStored[i] = 0;
    pointer[i + 1] += pointer[i];
  }
  MOCHI_ASSERT_VERBOSE(
      pointer[numColors] == n, "Incorrect length (%d != %d)", pointer[numColors], n);

  DynamicArray<NonConstIdx> targets(n);
  for (NonConstIdx i = 0; i < n; ++i) {
    auto const myColor = color[i];
    targets[pointer[myColor] + colorStored[myColor]] = i;
    colorStored[myColor] += 1;
  }

  return Graph<NonConstIdx, NonConstPtr>{std::move(pointer), std::move(targets)};
}

[[nodiscard]] inline auto ReverseMap(Span<int const> map, int size = -1, int defVal = -1) {
  if (size == -1) {
    size = isize(map);
  }
  DynamicArray<int> rMap(size, defVal);
  for (int i = 0; i < map.size(); ++i) {
    rMap[map[i]] = i;
  }
  return rMap;
}

class ImplicitGraph {
 public:
  using VertexType = int;
  using PointerType = int;
  ImplicitGraph(Span<int const> targets, int nPerSource)
      : _nPerSource(nPerSource), _size(isize(targets) / nPerSource), _targets(targets) {
    MOCHI_ASSERT_VERBOSE(targets.size() % _nPerSource == 0, "Invalid target size");
  }

  [[nodiscard]] auto size() const {
    return _size;
  }

  [[nodiscard]] auto NumTargets() const {
    return _targets.size();
  }

  [[nodiscard]] Span<int const, int> operator[](int s) const {
    return {_targets.data() + s * _nPerSource, _nPerSource};
  }

 private:
  int _nPerSource;
  int _size;
  Span<VertexType const> _targets;
};

/// @brief Create a directed graph from a range of ranges object.
/// @details The index of the outer range form the source vertices of the graph.
/// The values of each of the nested ranges are the target vertices of the graph.
template <typename Idx = int, typename Ptr = std::make_signed_t<size_t>, typename RangeOfRanges>
[[nodiscard]] Graph<Idx, Ptr> GraphFromRangeOfRanges(RangeOfRanges&& rOr) {
  MOCHI_PROFILE_SCOPE();
  Idx numSource = static_cast<Idx>(std::distance(rOr.begin(), rOr.end()));
  DynamicArray<Ptr> pointers;
  pointers.reserve(numSource + 1);
  DynamicArray<Idx> targets;
  targets.reserve(4 * numSource); // Arbitrary guess.
  pointers.push_back(Ptr{0});
  for (auto tgs : rOr) {
    for (auto tg : tgs) {
      targets.push_back(static_cast<Idx>(tg));
    }
    pointers.push_back(static_cast<Ptr>(targets.size()));
  }
  return {std::move(pointers), std::move(targets)};
}

/// @brief Form the graph bucket to element from the bucket assignments of elements.
/// @details The motivating use of this Graph builder is decompositions. Decomposers such
/// as METIS output a vector of assignment of entities to a subdomain. Calling this
/// function with the vector of assignments will output a directed graph of subdomain to entity.
///
/// @return The graph from the assigned group to the elements.
template <
    typename Idx = int,
    typename Ptr = std::make_signed_t<size_t>,
    typename Assignments,
    typename Projection = IdentityProjection>
[[nodiscard]] Graph<Idx, Ptr> GraphFromAssignments(
    Assignments&& assignments,
    Projection&& projection = Projection{}) {
  MOCHI_PROFILE_SCOPE();
  auto max_it = mochi::max_element(assignments, {}, projection);
  Idx mx = (max_it == std::end(assignments)) ? 0 : projection(*max_it) + 1;
  DynamicArray<Ptr> ptr(mx + 1, 0);
  for (auto a : assignments) {
    ++ptr[projection(a) + 1]; // To get sorted targets
  }
  auto lastCount = ptr.back();
  std::exclusive_scan(std::begin(ptr) + 1, std::end(ptr), std::begin(ptr) + 1, 0);
  DynamicArray<Idx> targets(ptr.back() + lastCount);
  for (Idx i = 0; i < assignments.size(); ++i) {
    auto a = assignments[i];
    targets[ptr[projection(a) + 1]++] = i;
  }
  return {std::move(ptr), std::move(targets)};
}

template <typename Tg = int, typename Ptr = std::make_signed_t<size_t>>
struct GraphBuilder {
  using Idx = std::conditional_t<std::is_integral_v<Tg>, Tg, Ptr>;
  GraphBuilder(Idx initialPtrSize, Ptr initialTargetSize) {
    _targets.reserve(initialTargetSize);
    _pointers.reserve(initialPtrSize + 1);
    _pointers.push_back(0);
  }

  [[nodiscard]] Span<Idx const, Ptr> operator[](Idx i) {
    return {&_targets[_pointers[i]], _pointers[i + 1] - _pointers[i]};
  }

  template <typename It, typename Sentinel>
  Idx append(It begin, Sentinel const& end) {
    Append(_targets, begin, end);
    _pointers.push_back(static_cast<Ptr>(_targets.size()));
    return static_cast<Idx>(_pointers.size()) - 1;
  }

  template <typename R>
  Idx append(R&& range) {
    return append(range.begin(), range.end());
  }

  /** @brief Starts a set for the next source and return its index. */
  Idx StartSet() {
    _pointers.push_back(static_cast<Ptr>(_targets.size()));
    return static_cast<Idx>(_pointers.size()) - 2;
  }

  Idx InsertTarget(Tg const& tg) {
    _targets.push_back(tg);
    ++_pointers.back();
    return static_cast<Idx>(_pointers.size()) - 1;
  }

  [[nodiscard]] Graph<Tg, Ptr> Build() {
    return {std::move(_pointers), std::move(_targets)};
  }

  [[nodiscard]] auto& Pointers() const {
    return _pointers;
  }

  [[nodiscard]] auto& Indices() const {
    return _targets;
  }

  /// @brief Get a view of the graph that Build() would construct.
  /// @details The view is invalidated by any call modifying the builder (append).
  [[nodiscard]] auto CurrentView() const {
    return Graph<Tg const, Ptr const, Span>(_pointers, _targets);
  }

  [[nodiscard]] Idx CurrentIndex() const {
    return static_cast<Idx>(_pointers.size()) - 2;
  }

 private:
  DynamicArray<Tg> _targets;
  DynamicArray<Ptr> _pointers;
};

/** @brief A priority queue with dynamic values.
 * @details Elements are inserted in the queue with a value and that value
 * can be decremented. At any time, one of the elements with the lowest value
 * can be retrieved and removed from the queue.
 */
class PriorityQueue {
 public:
  template <typename PriorityFtor>
  PriorityQueue(int nValues, PriorityFtor&& getPriority)
      : _min(std::numeric_limits<int>::max()), _max(std::numeric_limits<int>::min()) {
    if (nValues == 0) {
      return;
    }

    _prio.reserve(nValues);
    for (int i = 0; i < nValues; ++i) {
      auto p = getPriority(i);
      _prio.push_back(p);
      _min = std::min(p, _min);
      _max = std::max(p, _max);
    }
    auto g = GraphFromAssignments<int, int>(_prio);
    _ptr = std::move(g.GetMovablePointers());
    _index = std::move(g.GetMovableTargets());
    _count.reserve(_ptr.size() - 1);
    for (int i = 0; i < _ptr.size() - 1; ++i) {
      _count.push_back(_ptr[i + 1] - _ptr[i]);
    }
    _position.resize(nValues);
    for (int i = 0; i < nValues; ++i) {
      _position[_index[i]] = i;
    }
  }

  int RemoveMinElement() {
    MOCHI_ASSERT_VERBOSE(_count[_min] != 0);
    auto idx = _index[_ptr[_min]];
    ++_ptr[_min];
    --_count[_min];
    while (_min < _count.size() && _count[_min] == 0) {
      ++_min;
    }
    return idx;
  }

  void Decrement(int idx) {
    PopFront(idx);
    auto newPrio = --_prio[idx];
    _min = std::min(_min, newPrio);
    InsertBack(idx, newPrio);
  }

 private:
  DynamicArray<int> _ptr;
  DynamicArray<int> _count;
  DynamicArray<int> _index;
  DynamicArray<int> _position;
  DynamicArray<int> _prio;
  int _min;
  int _max;

  void InsertBack(int idx, int p) {
    int backPos = _ptr[p] + _count[p];
    MOCHI_ASSERT_VERBOSE(backPos < _ptr[p + 1]);
    _index[backPos] = idx;
    _position[idx] = backPos;
    ++_count[p];
  }

  /** @brief Remove element at a given position with known priority shifting its the front of its
   * bucket by one to the right.
   */
  void PopFront(int idx) {
    auto iPriority = _prio[idx];
    auto pos = _position[idx];
    MOCHI_ASSERT_VERBOSE(
        iPriority >= 1 && pos >= _ptr[iPriority] && pos < _ptr[iPriority] + _count[iPriority]);
    auto front = _index[_ptr[iPriority]];
    // Move the front element of the bucket to where idx was.
    // If idx == front, the operation has no effect.
    _index[pos] = front;
    _position[front] = pos;
    // Record the removal of idx.
    _position[idx] = -1;
    // Move the bucket front to the right.
    _index[_ptr[iPriority]] = -1;
    ++_ptr[iPriority];
    --_count[iPriority];
  }
};

template <typename Idx, typename Ptr, template <typename, typename...> class Storage>
[[nodiscard]] auto OrderedColoring(Graph<Idx, Ptr, Storage> const& g, Span<Idx const> order) {
  using NonConstIdx = std::remove_const_t<Idx>;
  NonConstIdx maxCount = 0;
  auto n = order.size();
  for (NonConstIdx i = 0; i < n; ++i) {
    maxCount = Max(maxCount, g.EdgeCount(i));
  }
  DynamicArray<int> color(n, -1);
  DynamicArray<bool> available;
  available.resize_noinit(maxCount + 1);
  for (auto nd : order) {
    std::fill(available.begin(), available.end(), true);
    for (auto neighbor : g[nd]) {
      if (color[neighbor] != -1) {
        available[color[neighbor]] = false;
      }
    }
    auto ptr = std::ranges::find(available, true);
    auto myColor = static_cast<int>(ptr - available.begin());
    color[nd] = myColor;
  }
  return color;
}

template <typename Idx, typename Ptr, template <typename, typename...> class Storage>
[[nodiscard]] auto SmallestLastColoring(Graph<Idx, Ptr, Storage> const& g) {
  int n = g.size();
  // Create a smallest last ordering.
  PriorityQueue pq(n, [&g](int v) { return g[v].size(); });
  DynamicArray<bool> unordered(n, true);
  DynamicArray<int> smallLastOrder(n);
  for (int i = n; --i >= 0;) {
    auto next = pq.RemoveMinElement();
    smallLastOrder[i] = next;
    unordered[next] = false;
    for (auto ngb : g[next]) {
      if (unordered[ngb]) {
        pq.Decrement(ngb);
      }
    }
  }
  return OrderedColoring(g, Span<Idx const>{smallLastOrder});
}

template <typename G, typename Idx, typename Accept>
void BreadthFirstSearch(G&& g, DynamicArray<Idx>& list, Accept&& acceptEdge) {
  size_t idx = 0;
  for (; idx < list.size(); ++idx) {
    for (auto n : g[list[idx]]) {
      if (acceptEdge(list[idx], n)) {
        list.push_back(n);
      }
    }
  }
}

} // namespace mochi
