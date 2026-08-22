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

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/indexed_heap.h>

#include <gtest/gtest.h>

#include <random>

namespace mochi {
namespace {

TEST(IndexedHeapTest, EmptyHeap) {
  IndexedHeap<int> heap(100);
  EXPECT_TRUE(heap.IsEmpty());
  EXPECT_EQ(heap.Size(), 0);
}

TEST(IndexedHeapTest, SingleInsertAndExtract) {
  IndexedHeap<int> heap(100);

  heap.Insert(42, 10);
  EXPECT_FALSE(heap.IsEmpty());
  EXPECT_EQ(heap.Size(), 1);
  EXPECT_TRUE(heap.Contains(42));
  EXPECT_EQ(heap.GetCost(42), 10);

  auto const& min = heap.FindMin();
  EXPECT_EQ(min.key, 42);
  EXPECT_EQ(min.cost, 10);

  auto extracted = heap.ExtractMin();
  EXPECT_EQ(extracted.key, 42);
  EXPECT_EQ(extracted.cost, 10);
  EXPECT_TRUE(heap.IsEmpty());
  EXPECT_FALSE(heap.Contains(42));
}

TEST(IndexedHeapTest, MultipleInserts) {
  IndexedHeap<int> heap(100);

  heap.Insert(10, 50);
  heap.Insert(20, 30);
  heap.Insert(30, 40);

  EXPECT_EQ(heap.Size(), 3);

  auto const& min = heap.FindMin();
  EXPECT_EQ(min.key, 20);
  EXPECT_EQ(min.cost, 30);
}

TEST(IndexedHeapTest, ExtractMinOrdering) {
  IndexedHeap<int> heap(100);

  heap.Insert(5, 50);
  heap.Insert(3, 30);
  heap.Insert(7, 70);
  heap.Insert(1, 10);
  heap.Insert(4, 40);

  // Extract should return in cost order.
  DynamicArray<int> extractedCosts;
  DynamicArray<int> extractedVertices;
  while (!heap.IsEmpty()) {
    auto entry = heap.ExtractMin();
    extractedVertices.push_back(entry.key);
    extractedCosts.push_back(entry.cost);
  }

  DynamicArray<int> const expectedVertices = {1, 3, 4, 5, 7};
  DynamicArray<int> const expectedCosts = {10, 30, 40, 50, 70};
  EXPECT_EQ(extractedVertices, expectedVertices);
  EXPECT_EQ(extractedCosts, expectedCosts);
}

TEST(IndexedHeapTest, UpdateCostDecrease) {
  IndexedHeap<int> heap(100);

  heap.Insert(1, 100);
  heap.Insert(2, 50);
  heap.Insert(3, 75);

  // Initial min is key 2 with cost 50.
  EXPECT_EQ(heap.FindMin().key, 2);
  EXPECT_EQ(heap.FindMin().cost, 50);

  // Decrease cost of key 1 to become the new minimum.
  heap.UpdateCost(1, 25);
  EXPECT_EQ(heap.GetCost(1), 25);
  EXPECT_EQ(heap.FindMin().key, 1);
  EXPECT_EQ(heap.FindMin().cost, 25);
}

TEST(IndexedHeapTest, UpdateCostIncrease) {
  IndexedHeap<int> heap(100);

  heap.Insert(1, 10);
  heap.Insert(2, 50);
  heap.Insert(3, 30);

  // Initial min is key 1 with cost 10.
  EXPECT_EQ(heap.FindMin().key, 1);

  // Increase cost of key 1, so key 3 becomes minimum.
  heap.UpdateCost(1, 100);
  EXPECT_EQ(heap.GetCost(1), 100);
  EXPECT_EQ(heap.FindMin().key, 3);
  EXPECT_EQ(heap.FindMin().cost, 30);
}

TEST(IndexedHeapTest, UpdateCostNoChange) {
  // Update a non-root vertex to its current cost so that the
  // newCost == oldCost branch (which avoids both Bubble calls) is exercised.
  // The heap is structured so vertex 2 has both a parent and two children,
  // so any spurious BubbleUp would swap with the parent and any spurious
  // BubbleDown would swap with the smallest child — either of which would
  // perturb the ExtractMin order verified below.
  //
  // Heap layout after the inserts:
  //   pos 0: (1, 10)
  //   pos 1: (2, 20)   <- updated; parent (1,10), children (4,40) and (5,50)
  //   pos 2: (3, 30)
  //   pos 3: (4, 40)
  //   pos 4: (5, 50)
  //   pos 5: (6, 60)
  IndexedHeap<int> heap(10);
  heap.Insert(1, 10);
  heap.Insert(2, 20);
  heap.Insert(3, 30);
  heap.Insert(4, 40);
  heap.Insert(5, 50);
  heap.Insert(6, 60);

  heap.UpdateCost(2, 20);
  EXPECT_EQ(heap.GetCost(2), 20);
  EXPECT_EQ(heap.Size(), 6);

  // Drain the heap and verify the keys and costs come out in the expected
  // order, confirming the no-op left every position untouched.
  DynamicArray<int> const expectedKeys = {1, 2, 3, 4, 5, 6};
  DynamicArray<int> const expectedCosts = {10, 20, 30, 40, 50, 60};
  DynamicArray<int> actualKeys;
  DynamicArray<int> actualCosts;
  while (!heap.IsEmpty()) {
    auto entry = heap.ExtractMin();
    actualKeys.push_back(entry.key);
    actualCosts.push_back(entry.cost);
  }
  EXPECT_EQ(actualKeys, expectedKeys);
  EXPECT_EQ(actualCosts, expectedCosts);
}

TEST(IndexedHeapTest, DeleteMiddleElement) {
  IndexedHeap<int> heap(100);

  heap.Insert(1, 10);
  heap.Insert(2, 20);
  heap.Insert(3, 30);

  heap.Delete(2);
  EXPECT_EQ(heap.Size(), 2);
  EXPECT_FALSE(heap.Contains(2));
  EXPECT_TRUE(heap.Contains(1));
  EXPECT_TRUE(heap.Contains(3));

  EXPECT_EQ(heap.FindMin().key, 1);
  EXPECT_EQ(heap.FindMin().cost, 10);
}

TEST(IndexedHeapTest, DeleteMinElement) {
  IndexedHeap<int> heap(100);

  heap.Insert(1, 10);
  heap.Insert(2, 20);
  heap.Insert(3, 30);

  heap.Delete(1);
  EXPECT_EQ(heap.Size(), 2);
  EXPECT_FALSE(heap.Contains(1));

  EXPECT_EQ(heap.FindMin().key, 2);
  EXPECT_EQ(heap.FindMin().cost, 20);
}

TEST(IndexedHeapTest, DeleteLastElement) {
  IndexedHeap<int> heap(100);

  heap.Insert(5, 50);
  heap.Delete(5);
  EXPECT_TRUE(heap.IsEmpty());
  EXPECT_FALSE(heap.Contains(5));
}

TEST(IndexedHeapTest, DeleteTriggersBubbleUp) {
  // Exercise the Delete() path where the replacement element (moved in from the
  // 'last' slot) has a cost strictly less than the deleted key's old cost
  // AND must actually climb past at least one parent.
  //
  // Heap after the inserts below:
  //   pos 0: (1,   1)
  //   pos 1: (2, 500)   <- parent of the deleted key's slot
  //   pos 2: (3, 100)
  //   pos 3: (4, 600)   <- key to delete
  //   pos 4: (5, 700)
  //   pos 5: (6, 200)   <- last; moved into pos 3 on delete, then bubbles up
  //
  // Delete(4) moves (6, 200) into pos 3; 200 < 600 selects the BubbleUp branch,
  // and BubbleUp swaps (6, 200) with its parent (2, 500) at pos 1.
  IndexedHeap<int> heap(10);
  heap.Insert(1, 1);
  heap.Insert(2, 500);
  heap.Insert(3, 100);
  heap.Insert(4, 600);
  heap.Insert(5, 700);
  heap.Insert(6, 200);

  heap.Delete(4);
  EXPECT_EQ(heap.Size(), 5);
  EXPECT_FALSE(heap.Contains(4));

  // Drain the heap and verify the resulting min-heap order is correct, which
  // confirms BubbleUp placed the moved entry at the right position.
  DynamicArray<int> const expectedVertices = {1, 3, 6, 2, 5};
  DynamicArray<int> const expectedCosts = {1, 100, 200, 500, 700};
  DynamicArray<int> actualVertices;
  DynamicArray<int> actualCosts;
  while (!heap.IsEmpty()) {
    auto entry = heap.ExtractMin();
    actualVertices.push_back(entry.key);
    actualCosts.push_back(entry.cost);
  }
  EXPECT_EQ(actualVertices, expectedVertices);
  EXPECT_EQ(actualCosts, expectedCosts);
}

TEST(IndexedHeapTest, ReinsertAfterDelete) {
  IndexedHeap<int> heap(100);

  heap.Insert(1, 10);
  heap.Delete(1);
  EXPECT_FALSE(heap.Contains(1));

  // Reinsert with different cost.
  heap.Insert(1, 20);
  EXPECT_TRUE(heap.Contains(1));
  EXPECT_EQ(heap.GetCost(1), 20);
}

TEST(IndexedHeapTest, ReinsertAfterExtract) {
  IndexedHeap<int> heap(100);

  heap.Insert(1, 10);
  heap.ExtractMin();

  heap.Insert(1, 30);
  EXPECT_TRUE(heap.Contains(1));
  EXPECT_EQ(heap.GetCost(1), 30);
}

TEST(IndexedHeapTest, EqualCosts) {
  IndexedHeap<int> heap(100);

  heap.Insert(1, 50);
  heap.Insert(2, 50);
  heap.Insert(3, 50);

  EXPECT_EQ(heap.Size(), 3);
  EXPECT_EQ(heap.FindMin().cost, 50);

  // Extract all — costs should remain equal.
  for (int i = 0; i < 3; ++i) {
    auto entry = heap.ExtractMin();
    EXPECT_EQ(entry.cost, 50);
  }
  EXPECT_TRUE(heap.IsEmpty());
}

TEST(IndexedHeapTest, FloatCosts) {
  IndexedHeap<float> heap(100);

  heap.Insert(0, 3.14f);
  heap.Insert(1, 1.41f);
  heap.Insert(2, 2.72f);

  EXPECT_EQ(heap.FindMin().key, 1);
  EXPECT_FLOAT_EQ(heap.FindMin().cost, 1.41f);

  heap.UpdateCost(2, 0.5f);
  EXPECT_EQ(heap.FindMin().key, 2);
  EXPECT_FLOAT_EQ(heap.FindMin().cost, 0.5f);
}

TEST(IndexedHeapTest, DoubleCosts) {
  IndexedHeap<double> heap(10);

  heap.Insert(0, 1.0e-10);
  heap.Insert(1, 2.0e-10);

  EXPECT_EQ(heap.FindMin().key, 0);
  EXPECT_DOUBLE_EQ(heap.FindMin().cost, 1.0e-10);
}

TEST(IndexedHeapTest, CustomVertexType) {
  IndexedHeap<int, int64_t> heap(int64_t{100});

  heap.Insert(int64_t{10}, 50);
  heap.Insert(int64_t{20}, 30);

  EXPECT_EQ(heap.FindMin().key, int64_t{20});
  EXPECT_EQ(heap.FindMin().cost, 30);
}

TEST(IndexedHeapTest, MaxGuessZeroNoReserve) {
  // maxGuess = 0 skips reservation. Insertions should still work via dynamic growth.
  IndexedHeap<int> heap(100, 0);

  heap.Insert(1, 10);
  heap.Insert(2, 20);
  EXPECT_EQ(heap.Size(), 2);
  EXPECT_EQ(heap.FindMin().key, 1);
}

TEST(IndexedHeapTest, MaxGuessCustomCapacity) {
  IndexedHeap<int> heap(1000, 10);

  for (int i = 0; i < 10; ++i) {
    heap.Insert(i, 100 - i);
  }
  EXPECT_EQ(heap.Size(), 10);
  EXPECT_EQ(heap.FindMin().key, 9);
}

TEST(IndexedHeapTest, LargeSequentialInsertExtract) {
  int const n = 500;
  IndexedHeap<int> heap(n);

  for (int i = 0; i < n; ++i) {
    heap.Insert(i, n - i);
  }
  EXPECT_EQ(heap.Size(), n);

  // Extract all — should come out in ascending cost order.
  for (int expectedCost = 1; expectedCost <= n; ++expectedCost) {
    auto entry = heap.ExtractMin();
    EXPECT_EQ(entry.cost, expectedCost);
    EXPECT_EQ(entry.key, n - expectedCost);
  }
  EXPECT_TRUE(heap.IsEmpty());
}

TEST(IndexedHeapTest, LargeRandomWorkload) {
  int const maxVertex = 10000;
  IndexedHeap<int> heap(maxVertex);
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> keyDist(0, maxVertex - 1);
  std::uniform_int_distribution<int> costDist(1, 100000);

  DynamicArray<int> insertedVertices;

  // Insert 500 random elements.
  for (int i = 0; i < 500; ++i) {
    int key = keyDist(rng);
    if (!heap.Contains(key)) {
      int cost = costDist(rng);
      heap.Insert(key, cost);
      insertedVertices.push_back(key);
    }
  }

  auto const insertCount = heap.Size();
  EXPECT_GT(insertCount, 0);

  // Do some update-cost operations.
  for (int i = 0; i < 100 && i < isize(insertedVertices); ++i) {
    int key = insertedVertices[i];
    if (heap.Contains(key)) {
      int newCost = costDist(rng);
      heap.UpdateCost(key, newCost);
      EXPECT_EQ(heap.GetCost(key), newCost);
    }
  }

  // Delete some elements.
  for (int i = 0; i < 50 && i < isize(insertedVertices); ++i) {
    int key = insertedVertices[i];
    if (heap.Contains(key)) {
      heap.Delete(key);
      EXPECT_FALSE(heap.Contains(key));
    }
  }

  // Extract all and verify ordering.
  DynamicArray<int> extractedCosts;
  while (!heap.IsEmpty()) {
    auto entry = heap.ExtractMin();
    extractedCosts.push_back(entry.cost);
    EXPECT_FALSE(heap.Contains(entry.key));
  }

  for (int i = 1; i < isize(extractedCosts); ++i) {
    EXPECT_LE(extractedCosts[i - 1], extractedCosts[i]);
  }
}

TEST(IndexedHeapTest, Contains) {
  IndexedHeap<int> heap(100);

  EXPECT_FALSE(heap.Contains(42));

  heap.Insert(42, 10);
  EXPECT_TRUE(heap.Contains(42));

  heap.Delete(42);
  EXPECT_FALSE(heap.Contains(42));
}

TEST(IndexedHeapTest, StressInsertDeleteReinsert) {
  int const n = 200;
  IndexedHeap<int> heap(n);

  // Insert all.
  for (int i = 0; i < n; ++i) {
    heap.Insert(i, i * 3);
  }
  EXPECT_EQ(heap.Size(), n);

  // Delete even vertices.
  for (int i = 0; i < n; i += 2) {
    heap.Delete(i);
  }
  EXPECT_EQ(heap.Size(), n / 2);

  // Verify min is key 1.
  EXPECT_EQ(heap.FindMin().key, 1);
  EXPECT_EQ(heap.FindMin().cost, 3);

  // Reinsert even vertices with lower costs.
  for (int i = 0; i < n; i += 2) {
    heap.Insert(i, i);
  }
  EXPECT_EQ(heap.Size(), n);

  // Vertex 0 (cost 0) should now be the minimum.
  EXPECT_EQ(heap.FindMin().key, 0);
  EXPECT_EQ(heap.FindMin().cost, 0);

  // Extract all and verify non-decreasing costs.
  DynamicArray<int> costs;
  while (!heap.IsEmpty()) {
    costs.push_back(heap.ExtractMin().cost);
  }
  for (int i = 1; i < isize(costs); ++i) {
    EXPECT_LE(costs[i - 1], costs[i]);
  }
}

} // namespace
} // namespace mochi
