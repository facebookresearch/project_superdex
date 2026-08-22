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
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/task_scheduler.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>

namespace mochi {

template <typename T, typename GView>
concept GraphConvertible = requires(T& g) {
  { g.GetTargets() } -> std::convertible_to<typename GView::TgStorage>;
  { g.GetPointers() } -> std::convertible_to<typename GView::PtrStorage>;
};

/**
 * @brief Directed graph structure with arbitrary storage type.
 * @details The graph can map indices to any type. Edges of the graph link source indices
 * to target objects of any type.
 * @tparam Tg Type of the graph target.
 * @tparam Ptr Integer type for start of edges from a given vertex.
 * @tparam Storage template class for storing pointers and edge targets.
 */
template <
    typename Tg = int,
    typename Ptr = std::make_signed_t<size_t>,
    template <typename, typename...> class Storage = DynamicArray>
class Graph {
 public:
  using Idx = std::conditional_t<std::is_integral_v<Tg>, Tg, Ptr>;
  using VertexType = Idx;
  using PointerType = Ptr;
  using TgStorage = Storage<Tg>;
  using PtrStorage = Storage<Ptr>;

  Graph() = default;
  Graph(Storage<Ptr> pointers, Storage<Tg> targets)
      : _pointers(std::move(pointers)), _targets(std::move(targets)) {}
  Graph(Graph const&) = default;
  Graph(Graph&&) noexcept = default;
  /// \brief Constructor for a view
  Graph(GraphConvertible<Graph> auto&& g) : Graph(g.GetPointers(), g.GetTargets()) {}

  Graph& operator=(Graph&& rhs) noexcept {
    // Polymorphic_allocator does not propagate on container copy assignment, move assignment, or
    // swap. As a result, move assignment of a polymorphic_allocator-using container can throw, and
    // swapping two polymorphic_allocator-using containers whose allocators do not compare equal
    // results in undefined behavior.
    if (std::addressof(rhs) != this) {
      this->Reset(std::move(rhs));
    }
    return *this;
  }

  Graph& operator=(Graph const& rhs) = default;

  // Reset this Graph object using the arguments for any of its constructors.
  template <typename... Args>
  Graph& Reset(Args&&... args) {
    this->~Graph();
    new (this) Graph(std::forward<Args>(args)...);
    return *this;
  }

  Idx size() const {
    return _pointers.empty() ? 0 : static_cast<Idx>(_pointers.size() - 1);
  }
  Idx EdgeCount(Idx v) const {
    return _pointers[v + 1] - _pointers[v];
  }
  Span<Tg, Ptr> operator[](Idx v) {
    return {_targets.data() + _pointers[v], static_cast<Ptr>(EdgeCount(v))};
  }
  Span<Tg const, Ptr> operator[](Idx v) const {
    return {_targets.data() + _pointers[v], static_cast<Ptr>(EdgeCount(v))};
  }
  Span<Ptr const> GetPointers() const {
    return _pointers;
  }
  Span<Tg> GetTargets() {
    return _targets;
  }
  Span<Tg const> GetTargets() const {
    return _targets;
  }

  // Get the underlying storage so it can be moved via std::move.
  auto&& GetMovablePointers() {
    return _pointers;
  }
  auto&& GetMovableTargets() {
    return _targets;
  }

  template <typename Cmp = std::less<Tg>>
  Graph SortTargets(Cmp&& cmp = {}) && {
    MOCHI_PROFILE_SCOPE();
    if (NumTargets() == 0) {
      return *this;
    }

    // NOTE: The parallelization
    // - Is tuned for a number of targets per vertex such that the log(N) term in sorting is small.
    //   If the number of targets per vertex is very large, a task will be created per vertex
    //   anyway.
    // - Assumes the number of targets per vertex is evenly distributed. If this is not the case, it
    //   could be resolved by distributing work based on _pointers.
    constexpr long long kMinTargetsPerTask = 5000; // Empirical value (~50 μs @ 1e8 targets/s)
    int const minVerticesPerTask =
        Max(1, Min(isize(*this), static_cast<int>(kMinTargetsPerTask * size() / NumTargets())));
    ParallelForN("SortTargets", size(), minVerticesPerTask, [&](int v) {
      auto r = (*this)[v];
      std::sort(r.begin(), r.end(), cmp);
    });
    return *this;
  }

  Ptr NumTargets() const {
    return static_cast<Ptr>(_targets.size());
  }

  /// @brief Returns the maximal target.
  /// @return Maximal target.
  /// @note The function is O(N) in the number of targets.
  /// @note The function returns min when the graph is empty.
  Tg MaxTarget(Tg min = -1) const {
    auto maxElement = std::max_element(_targets.begin(), _targets.end());
    return maxElement == _targets.end() ? min : *maxElement;
  }

  /// @brief Value associated with iterating over a graph.
  template <typename W>
  struct VertexRange {
    VertexRange(Idx v, Span<W> tg) : vertex(v), targets(tg) {}
    auto operator[](Idx i) {
      return targets[i];
    }
    Idx vertex; /// @brief Vertex index
    Span<W> targets; /// @brief Span over the targets for vertex.
  };

  template <typename G>
  struct iterator_type {
    using T = std::conditional_t<std::is_const_v<G>, Tg const, Tg>;
    auto operator*() const {
      return VertexRange<T>{n, g[n]};
    }
    iterator_type& operator++() {
      ++n;
      return *this;
    }
    bool operator==(iterator_type const& b) const {
      return n == b.n;
    }
    // Not necessary in C++20
    bool operator!=(iterator_type const& b) const {
      return n != b.n;
    }
    G& g;
    std::decay_t<Idx> n;
  };

  using iterator = iterator_type<Graph>;
  using const_iterator = iterator_type<Graph const>;

  iterator begin() {
    return {*this, 0};
  }
  iterator end() {
    return {*this, size()};
  }
  const_iterator begin() const {
    return {*this, 0};
  }
  const_iterator end() const {
    return {*this, size()};
  }

 private:
  /// @brief Pointers into _targets for the start of edges for a vertex with a trailing end pointer.
  Storage<Ptr> _pointers;
  /// @brief Target vertices of edges from a given vertex.
  Storage<Tg> _targets;
};

template <typename T>
using GraphTargetType = std::decay_t<decltype(std::declval<T const>()[0][0])>;

template <typename Idx, typename Ptr, template <typename, typename...> class Storage>
auto NumTargets(Graph<Idx, Ptr, Storage> const& g) {
  return g.NumTargets();
}

template <typename GType>
auto NumTargets(GType const& g) {
  size_t count{0};
  for (size_t s = 0; s < g.size(); ++s) {
    count += g[s].size();
  }
  return count;
}

/// @brief Returns the pair of the maximal target index stored and the number of targets.
/// @note The function is O(N) in the number of targets.
/// @note The function returns min, as maximal target, when the graph is empty.
template <typename Idx, typename Ptr, template <typename, typename...> class Storage>
auto MaxAndCountTargets(Graph<Idx, Ptr, Storage> const& g, Idx min = -1) {
  return std::pair{+g.MaxTarget(min), +g.NumTargets()}; // + is for future genericity
}

template <typename GType>
auto MaxAndCountTargets(GType const& g, GraphTargetType<GType> min = -1) {
  auto mx = min;
  size_t count{0};
  using Idx = std::decay_t<decltype(g.size())>;
  for (Idx s = 0; s < g.size(); ++s) {
    count += g[s].size();
    for (auto const& t : g[s]) {
      mx = std::max(mx, t);
    }
  }
  return std::pair{mx, count};
}

template <typename Idx, typename Ptr, class GraphType>
auto Reverse(GraphType&& g, Idx size = -1) {
  MOCHI_PROFILE_SCOPE();
  // Find the maximum target, it determines the number of sources in the result.
  auto [maxTarget, numTargets] = MaxAndCountTargets(g);
  MOCHI_ASSERT_VERBOSE(size == -1 || size >= maxTarget + 1);
  Idx resSize = size == -1 ? static_cast<Idx>(maxTarget) + 1 : size;
  DynamicArray<Ptr> pointers(resSize + 1, 0);
  DynamicArray<Idx> targets;
  targets.resize_noinit(numTargets);
  for (Idx src = 0; src < g.size(); ++src) {
    for (auto v : g[src]) {
      ++pointers[v + 1];
    }
  }
  std::exclusive_scan(pointers.begin(), pointers.end(), pointers.begin(), 0);
  for (Idx src = 0; src < g.size(); ++src) {
    for (auto v : g[src]) {
      targets[pointers[v + 1]++] = src;
    }
  }
  return Graph<Idx, Ptr>{std::move(pointers), std::move(targets)};
}

/// @brief Build the reverse graph.
///
/// @tparam GraphType
/// @param g The graph to reverse.
/// @return A graph where for every edge (a,b) of g, there is an edge (b,a).
template <class GraphType>
auto Reverse(GraphType&& g) {
  using Idx = std::decay_t<typename std::decay_t<GraphType>::VertexType>;
  using Ptr = std::decay_t<typename std::decay_t<GraphType>::PointerType>;
  return Reverse<Idx, Ptr>(g);
}

/// @brief Build the transitive graph.
///
/// @tparam GraphTypeA
/// @tparam GraphTypeB
/// @param g_a
/// @param g_b
/// @return
template <class GraphTypeA, class GraphTypeB>
auto Traverse(GraphTypeA&& g_a, GraphTypeB&& g_b) {
  MOCHI_PROFILE_SCOPE();
  using IdxA = std::decay_t<typename std::decay_t<GraphTypeA>::VertexType>;
  using PtrA = std::decay_t<typename std::decay_t<GraphTypeA>::PointerType>;
  // Find the maximum target of the second graph to size the flagging array.
  auto [maxTarget, countB] = MaxAndCountTargets(g_b);
  using IdxB = std::decay_t<decltype(maxTarget)>;
  DynamicArray<IdxB> flag(maxTarget + 1, static_cast<IdxB>(-1));
  DynamicArray<PtrA> pointers;
  DynamicArray<IdxB> targets;
  // Preallocate targets assuming 50% overlap factor.
  // NOTE: This is a general heuristic. It could be optimized if the input graphs have known
  // topology, e.g. eToN from a tet mesh.
  auto const avgDegreeB = static_cast<double>(countB) / Max(1, isize(g_b));
  targets.reserve(Min(/* Estimate */ static_cast<size_t>(0.5 * avgDegreeB * g_a.NumTargets()),
                      /* Upper bound */ static_cast<size_t>(countB) * g_a.size()));
  pointers.reserve(g_a.size() + 1);
  pointers.push_back(0);
  for (IdxA src = 0; src < g_a.size(); ++src) {
    for (auto v : g_a[src]) {
      for (auto v_B : g_b[v]) {
        if (flag[v_B] != src) {
          flag[v_B] = src;
          targets.push_back(v_B);
        }
      }
    }
    pointers.push_back(static_cast<PtrA>(targets.size()));
  }
  return Graph<IdxB, PtrA>{std::move(pointers), std::move(targets)};
}

} // namespace mochi
