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
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>

#include <concepts>
#include <optional>
#include <type_traits>

namespace mochi {

/// @brief A binary min-heap with O(1) key lookup and O(log N) insert, update, and extract-min.
///
/// @details Each entry maps a key (integer id) to a cost. The heap maintains the min-cost key
/// at the top. Costs associated with Key can be updated efficiently via the index array that tracks
/// each key's position in the heap.
///
/// Invariants:
/// - `_heap` stores (key, cost) pairs in standard binary min-heap order by cost.
/// - `_index[k]` gives the position of key `k` in `_heap`, or @ref kAbsent if `k` is not in
/// the heap.
///
/// @note Key is also used as the index type for the heap.
///
/// @tparam Cost Arithmetic type for the cost (e.g. int, float, double).
/// @tparam Key Integer type for key indices (default: int).
template <typename Cost, typename Key = int>
  requires(std::is_arithmetic_v<Cost> && std::integral<Key> && sizeof(Key) >= 2)
class IndexedHeap {
 public:
  static constexpr Key kAbsent = kMinusOne<Key>;
  struct Entry {
    Key key;
    Cost cost;
  };

  /// @brief Construct an empty indexed heap.
  ///
  /// @param endKey One past the maximum key index that can be stored. The index array is
  /// allocated with this size and initialized to @ref kAbsent.
  /// @param maxGuess Optional expected maximum number of elements. Used to pre-reserve heap
  /// storage. If not provided, endKey is used.
  explicit IndexedHeap(Key endKey, std::optional<Key> maxGuess = std::nullopt)
      : _index(ValidatedSize(endKey), kAbsent) {
    _heap.reserve(maxGuess.value_or(endKey));
  }

  /// @brief Check if the heap is empty.
  [[nodiscard]] bool IsEmpty() const {
    return _heap.empty();
  }

  /// @brief Get the current number of elements.
  [[nodiscard]] size_t Size() const {
    return _heap.size();
  }

  /// @brief Check if a key is in the heap.
  /// @note In debug builds, this function asserts that the key index is valid.
  [[nodiscard]] bool Contains(Key key) const {
    if constexpr (std::signed_integral<Key>) {
      MOCHI_ASSERT_VERBOSE(key >= 0);
    }
    MOCHI_ASSERT_VERBOSE(key < static_cast<Key>(_index.size()));
    return _index[key] != kAbsent;
  }

  /// @brief Get the cost of a key. The key must be in the heap.
  [[nodiscard]] Cost GetCost(Key key) const {
    MOCHI_ASSERT_VERBOSE(Contains(key));
    return _heap[_index[key]].cost;
  }

  /// @brief Insert a key with given cost.
  ///
  /// @param key Key to insert (must not already be in the heap).
  /// @param cost Cost associated with the key.
  void Insert(Key key, Cost cost) {
    MOCHI_ASSERT_VERBOSE(!Contains(key), "Key already in heap.");
    Key const pos = static_cast<Key>(Size());
    _heap.push_back({key, cost});
    _index[key] = pos;
    BubbleUp(pos);
  }

  /// @brief Update the cost of an existing key.
  ///
  /// @param key Key (must be in the heap).
  /// @param newCost New cost value.
  void UpdateCost(Key key, Cost newCost) {
    MOCHI_ASSERT_VERBOSE(Contains(key));
    Key pos = _index[key];
    Cost oldCost = _heap[pos].cost;
    _heap[pos].cost = newCost;
    if (newCost < oldCost) {
      BubbleUp(pos);
    } else if (newCost > oldCost) {
      BubbleDown(pos);
    }
  }

  /// @brief Get the key with minimum cost without removing it.
  ///
  /// @return Reference to the entry with minimum cost.
  [[nodiscard]] Entry const& FindMin() const {
    MOCHI_ASSERT_VERBOSE(!IsEmpty());
    return _heap[0];
  }

  /// @brief Remove and return the key with minimum cost.
  ///
  /// @return The entry that was at the top of the heap.
  Entry ExtractMin() {
    MOCHI_ASSERT_VERBOSE(!IsEmpty());
    Entry result = _heap[0];
    Delete(result.key);
    return result;
  }

  /// @brief Delete a key from the heap.
  ///
  /// @param key Key (must be in the heap).
  void Delete(Key key) {
    MOCHI_ASSERT_VERBOSE(Contains(key));
    Key pos = _index[key];
    _index[key] = kAbsent;

    Key const last = static_cast<Key>(Size()) - 1;
    if (pos == last) {
      _heap.pop_back();
      return;
    }

    Cost oldCost = _heap[pos].cost;
    _heap[pos] = _heap[last];
    _index[_heap[pos].key] = pos;
    _heap.pop_back();

    if (_heap[pos].cost < oldCost) {
      BubbleUp(pos);
    } else {
      BubbleDown(pos);
    }
  }

 private:
  static Key ValidatedSize(Key endKey) {
    if constexpr (std::signed_integral<Key>) {
      MOCHI_ASSERT_VERBOSE(endKey >= 0);
    }
    return endKey;
  }

  /// @brief Move an element up in the heap until the heap property is restored.
  void BubbleUp(Key pos) {
    while (pos > 0) {
      Key parent = (pos - 1) / 2;
      if (_heap[parent].cost <= _heap[pos].cost) {
        break;
      }
      Swap(pos, parent);
      pos = parent;
    }
  }

  /// @brief Move an element down in the heap until the heap property is restored.
  void BubbleDown(Key pos) {
    // Compute child indices in size_t to avoid overflow when Key is a 16-bit type.
    // Once compared against n, the result is guaranteed to fit in Key.
    auto const n = Size();
    for (;;) {
      auto const left = static_cast<size_t>(pos) * 2 + 1;
      size_t const right = left + 1;
      Key smallest = pos;
      if (left < n && _heap[left].cost < _heap[smallest].cost) {
        smallest = static_cast<Key>(left);
      }
      if (right < n && _heap[right].cost < _heap[smallest].cost) {
        smallest = static_cast<Key>(right);
      }
      if (smallest == pos) {
        break;
      }
      Swap(pos, smallest);
      pos = smallest;
    }
  }

  /// @brief Swap two entries in the heap array and update the index.
  void Swap(Key a, Key b) {
    _index[_heap[a].key] = b;
    _index[_heap[b].key] = a;
    auto tmp = _heap[a];
    _heap[a] = _heap[b];
    _heap[b] = tmp;
  }

  DynamicArray<Entry> _heap; ///< Binary min-heap ordered by cost.
  DynamicArray<Key> _index; ///< key -> position in _heap, or @ref kAbsent if absent.
};

} // namespace mochi
