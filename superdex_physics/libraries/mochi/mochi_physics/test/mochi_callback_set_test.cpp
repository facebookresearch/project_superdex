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

#include <mochi_physics/src/mochi_callback_set.h>

#include <mochi_core/mochi_platform.h>

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace mochi;

// =============================================================================
// Helpers
// =============================================================================

static constexpr int kNumThreads = 4;
static constexpr int kNumIterations = 100;

static void StartBarrier(std::atomic<int>& ready, int expected) {
  ++ready;
  while (ready.load() < expected) {
    std::this_thread::yield();
  }
}

static void RunConcurrently(int numThreads, int iters, std::function<void(int)> fn) {
  std::atomic<int> ready{0};
  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  for (int t = 0; t < numThreads; ++t) {
    threads.emplace_back([&, t]() {
      StartBarrier(ready, numThreads);
      for (int i = 0; i < iters; ++i) {
        fn(t);
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
}

// =============================================================================
// Test Cases
// =============================================================================

TEST(CallbackSet, RegisterAndCall) {
  int sum = 0;

  CallbackSet<void(int const&)> cs;
  EXPECT_TRUE(cs.IsEmpty());

  // Single callback
  cs.Register(1, [&](int v) { sum += v; });
  EXPECT_FALSE(cs.IsEmpty());
  cs.Call(42);
  EXPECT_EQ(sum, 42);
  sum = 0;

  // Multiple callbacks + operator()
  cs.Register(2, [&](int v) { sum += v * 10; });
  cs.Register(3, [&](int v) { sum += v * 100; });
  cs.Call(1);
  EXPECT_EQ(sum, 111);
  sum = 0;
  cs(1);
  EXPECT_EQ(sum, 111);
}

TEST(CallbackSet, ConstReferencePayloadDoesNotRequireCopyableType) {
  struct NonCopyablePayload {
    explicit NonCopyablePayload(int value_) : value(value_) {}
    MOCHI_DECLARE_NO_COPY_NO_MOVE(NonCopyablePayload);
    int value = 0;
  };

  CallbackSet<void(NonCopyablePayload const&)> cs;

  int sum = 0;
  cs.Register(1, [&](NonCopyablePayload const& payload) { sum += payload.value; });
  cs.Register(2, [&](NonCopyablePayload const& payload) { sum += payload.value * 10; });

  NonCopyablePayload payload{3};
  cs.Call(payload);

  EXPECT_EQ(sum, 33);
}

TEST(CallbackSet, PriorityOrdering) {
  // Basic ordering: High < Normal < Low
  {
    CallbackSet<void()> cs;
    std::vector<int> order;
    cs.Register(1, [&]() { order.push_back(1); }, kCallbackPriorityLow);
    cs.Register(2, [&]() { order.push_back(2); }, kCallbackPriorityHigh);
    cs.Register(3, [&]() { order.push_back(3); }, kCallbackPriorityNormal);
    cs.Call();
    EXPECT_EQ(order, (std::vector<int>{2, 3, 1}));
  }

  // Extremes: Max < High < Normal < Low < Lowest
  {
    CallbackSet<void()> cs;
    std::vector<int> order;
    cs.Register(1, [&]() { order.push_back(1); }, kCallbackPriorityLowest);
    cs.Register(2, [&]() { order.push_back(2); }, kCallbackPriorityMax);
    cs.Register(3, [&]() { order.push_back(3); }, kCallbackPriorityNormal);
    cs.Register(4, [&]() { order.push_back(4); }, kCallbackPriorityHigh);
    cs.Register(5, [&]() { order.push_back(5); }, kCallbackPriorityLow);
    cs.Call();
    EXPECT_EQ(order, (std::vector<int>{2, 4, 3, 5, 1}));
  }

  // Same priority: all fire (order is not specified)
  {
    CallbackSet<void()> cs;
    int count = 0;
    cs.Register(1, [&]() { ++count; }, kCallbackPriorityNormal);
    cs.Register(2, [&]() { ++count; }, kCallbackPriorityNormal);
    cs.Register(3, [&]() { ++count; }, kCallbackPriorityNormal);
    cs.Call();
    EXPECT_EQ(count, 3);
  }
}

TEST(CallbackSet, PredicateCall) {
  // Empty set: predicate never called (void and non-void)
  {
    int predCount = 0;
    CallbackSet<void(int const&)> cs;
    cs.Call(0, [&]() { ++predCount; });
    EXPECT_EQ(predCount, 0);
    CallbackSet<int(int const&)> cs2;
    cs2.Call(0, [&](int) { ++predCount; });
    EXPECT_EQ(predCount, 0);
  }

  // Void return: predicate called once per callback
  {
    CallbackSet<void(int const&)> cs;
    cs.Register(1, [](int) {});
    cs.Register(2, [](int) {});
    cs.Register(3, [](int) {});
    int predCount = 0;
    cs.Call(0, [&]() { ++predCount; });
    EXPECT_EQ(predCount, 3);
  }

  // Non-void return: predicate receives values in priority order
  {
    CallbackSet<int(int const&)> cs;
    cs.Register(1, [](int v) { return v * 1; }, kCallbackPriorityHigh);
    cs.Register(2, [](int v) { return v * 2; }, kCallbackPriorityNormal);
    cs.Register(3, [](int v) { return v * 3; }, kCallbackPriorityLow);
    std::vector<int> results;
    cs.Call(5, [&](int r) { results.push_back(r); });
    EXPECT_EQ(results, (std::vector<int>{5, 10, 15}));
  }

  // operator() with predicate
  {
    CallbackSet<int(int const&)> cs;
    cs.Register(1, [](int v) { return v * 2; }, kCallbackPriorityHigh);
    cs.Register(2, [](int v) { return v * 3; }, kCallbackPriorityNormal);
    std::vector<int> results;
    cs(5, [&](int r) { results.push_back(r); });
    EXPECT_EQ(results, (std::vector<int>{10, 15}));
  }

  // Predicate skips deregistered callback
  {
    CallbackSet<void()> cs;
    int predCount = 0;
    CallbackId id2 = 2;
    cs.Register(1, [&]() { cs.Deregister(id2); }, kCallbackPriorityHigh);
    cs.Register(id2, []() {}, kCallbackPriorityNormal);
    cs.Call([&]() { ++predCount; });
    EXPECT_EQ(predCount, 1); // cb1 fires; cb2 deregistered and skipped
  }

  // Predicate with reentrant call: nested plain Call does not invoke outer predicate
  {
    CallbackSet<void()> cs;
    std::vector<int> order;
    bool nestedDone = false;
    cs.Register(
        1,
        [&]() {
          order.push_back(10);
          if (!nestedDone) {
            nestedDone = true;
            cs.Call();
          }
        },
        kCallbackPriorityHigh);
    cs.Register(2, [&]() { order.push_back(20); }, kCallbackPriorityNormal);
    int predCount = 0;
    cs.Call([&]() { ++predCount; });
    EXPECT_EQ(order, (std::vector<int>{10, 10, 20, 20}));
    EXPECT_EQ(predCount, 2);
  }
}

TEST(CallbackSet, Deregister) {
  // Before call: id zeroed, callback does not fire
  {
    CallbackSet<void()> cs;
    int count = 0;
    CallbackId id = 1;
    cs.Register(id, [&]() { ++count; });
    cs.Deregister(id);
    EXPECT_EQ(id, kInvalidCallbackId);
    EXPECT_TRUE(cs.IsEmpty());
    cs.Call();
    EXPECT_EQ(count, 0);
  }

  // Unknown id: no-op, id unchanged, registered callback unaffected
  {
    CallbackSet<void()> cs;
    int count = 0;
    cs.Register(1, [&]() { ++count; });
    CallbackId id = 999;
    cs.Deregister(id);
    EXPECT_EQ(id, static_cast<CallbackId>(999));
    cs.Call();
    EXPECT_EQ(count, 1);
  }

  // Already-removed id: numeric value not in map; no-op, variable unchanged
  {
    CallbackSet<void()> cs;
    cs.Register(1, []() {});
    CallbackId id = 1;
    cs.Deregister(id);
    EXPECT_EQ(id, kInvalidCallbackId);
    CallbackId staleId = 1;
    cs.Deregister(staleId);
    EXPECT_EQ(staleId, static_cast<CallbackId>(1)); // unchanged — not found in map
  }

  // Self-deregister inside callback: runs to completion, skipped on next call
  {
    CallbackSet<void()> cs;
    std::vector<int> order;
    CallbackId id1 = 1, id2 = 2;
    cs.Register(
        id1,
        [&]() {
          order.push_back(1);
          cs.Deregister(id1);
        },
        kCallbackPriorityHigh);
    cs.Register(id2, [&]() { order.push_back(2); }, kCallbackPriorityNormal);
    cs.Call();
    EXPECT_EQ(order, (std::vector<int>{1, 2}));
    EXPECT_EQ(id1, kInvalidCallbackId);
    order.clear();
    cs.Call();
    EXPECT_EQ(order, (std::vector<int>{2}));
  }

  // Deregister other inside callback: other is skipped in same call
  {
    CallbackSet<void()> cs;
    std::vector<int> order;
    CallbackId id2 = 2;
    cs.Register(
        1,
        [&]() {
          order.push_back(1);
          cs.Deregister(id2);
        },
        kCallbackPriorityHigh);
    cs.Register(id2, [&]() { order.push_back(2); }, kCallbackPriorityNormal);
    cs.Call();
    EXPECT_EQ(order, (std::vector<int>{1}));
    EXPECT_EQ(id2, kInvalidCallbackId);
  }
}

TEST(CallbackSet, ClearAndIsEmpty) {
  // Clear removes all; set is empty and callbacks don't fire
  {
    CallbackSet<void()> cs;
    int count = 0;
    cs.Register(1, [&]() { ++count; });
    cs.Register(2, [&]() { ++count; });
    cs.Clear();
    EXPECT_TRUE(cs.IsEmpty());
    cs.Call();
    EXPECT_EQ(count, 0);
  }

  // Double clear is safe; set remains usable after
  {
    CallbackSet<void()> cs;
    cs.Register(1, []() {});
    cs.Clear();
    cs.Clear();
    EXPECT_TRUE(cs.IsEmpty());
    int count = 0;
    cs.Register(2, [&]() { ++count; });
    cs.Call();
    EXPECT_EQ(count, 1);
  }

  // Clear inside callback: cb2 skipped; set empty after call
  {
    CallbackSet<void()> cs;
    std::vector<int> order;
    cs.Register(
        1,
        [&]() {
          order.push_back(1);
          cs.Clear();
        },
        kCallbackPriorityHigh);
    cs.Register(2, [&]() { order.push_back(2); }, kCallbackPriorityNormal);
    cs.Call();
    EXPECT_EQ(order, (std::vector<int>{1}));
    EXPECT_TRUE(cs.IsEmpty());
  }

  // IsEmpty during call: deferred erasure means map still has entries (Clear and self-deregister)
  {
    CallbackSet<void()> cs;
    bool seenAfterClear = false;
    cs.Register(1, [&]() {
      cs.Clear();
      seenAfterClear = cs.IsEmpty();
    });
    cs.Call();
    EXPECT_FALSE(seenAfterClear); // deferred erasure: _callbacks not yet empty
    EXPECT_TRUE(cs.IsEmpty()); // cleanup ran after Call()
  }
  {
    CallbackSet<void()> cs;
    bool seenAfterDeregister = false;
    CallbackId id = 1;
    cs.Register(id, [&]() {
      cs.Deregister(id);
      seenAfterDeregister = cs.IsEmpty();
    });
    cs.Call();
    EXPECT_FALSE(seenAfterDeregister);
    EXPECT_TRUE(cs.IsEmpty());
  }

  // Regression: Clear() at callDepth==0 resets _deregisteredCallbacksDirty
  {
    CallbackSet<void()> cs;
    CallbackId id1 = 1, id2 = 2;
    int id2FireCount = 0, newCount = 0;
    cs.Register(id1, [&]() { cs.Deregister(id2); }, kCallbackPriorityHigh);
    cs.Register(id2, [&]() { ++id2FireCount; }, kCallbackPriorityNormal);
    cs.Call();
    EXPECT_EQ(id2FireCount, 0);
    cs.Clear();
    cs.Register(3, [&]() { ++newCount; });
    cs.Call();
    EXPECT_EQ(newCount, 1);
  }
}

TEST(CallbackSet, MoveSemantics) {
  // Move construct: source emptied, callbacks transferred
  {
    int received = 0;
    CallbackSet<void(int const&)> cs1;
    cs1.Register(1, [&](int v) { received = v; });
    CallbackSet<void(int const&)> cs2 = std::move(cs1);
    EXPECT_TRUE(cs1.IsEmpty()); // NOLINT(bugprone-use-after-move)
    EXPECT_FALSE(cs2.IsEmpty());
    cs2.Call(42);
    EXPECT_EQ(received, 42);
    cs1.Call(99);
    EXPECT_EQ(received, 42); // cs1 has no callbacks
  }

  // Move assign: source emptied, target's old callbacks replaced
  {
    int count1 = 0, count2 = 0;
    CallbackSet<void()> cs1, cs2;
    cs1.Register(1, [&]() { ++count1; });
    cs2.Register(2, [&]() { ++count2; });
    cs2 = std::move(cs1);
    EXPECT_TRUE(cs1.IsEmpty()); // NOLINT(bugprone-use-after-move)
    cs2.Call();
    EXPECT_EQ(count1, 1);
    EXPECT_EQ(count2, 0); // replaced
  }

  // Moved-from source is reusable
  {
    int countOrig = 0, countNew = 0;
    CallbackSet<void()> cs1;
    cs1.Register(1, [&]() { ++countOrig; });
    CallbackSet<void()> cs2 = std::move(cs1);
    cs1.Register(2, [&]() { ++countNew; }); // NOLINT(bugprone-use-after-move)
    cs1.Call();
    cs2.Call();
    EXPECT_EQ(countNew, 1);
    EXPECT_EQ(countOrig, 1);
  }

  // Self-assign: guarded no-op
  {
    CallbackSet<void()> cs;
    int count = 0;
    cs.Register(1, [&]() { ++count; });
    cs.Register(2, [&]() { ++count; });
    CallbackSet<void()>* pCs = &cs;
    cs = std::move(*pCs);
    EXPECT_FALSE(cs.IsEmpty());
    cs.Call();
    EXPECT_EQ(count, 2);
  }
}

TEST(CallbackSet, ReentrantCalls) {
  // Simple reentry: depth-first traversal
  {
    CallbackSet<void(int const&)> cs;
    std::vector<int> order;
    cs.Register(1, [&](int depth) {
      order.push_back(depth);
      if (depth > 0) {
        cs.Call(depth - 1);
      }
    });
    cs.Call(2);
    EXPECT_EQ(order, (std::vector<int>{2, 1, 0}));
  }

  // Double nesting: A0 → A1 → A2/B2 → B1 → B0
  {
    CallbackSet<void()> cs;
    std::vector<std::string> order;
    int level = 0;
    cs.Register(
        1,
        [&]() {
          order.push_back("A" + std::to_string(level));
          if (level < 2) {
            ++level;
            cs.Call();
            --level;
          }
        },
        kCallbackPriorityHigh);
    cs.Register(
        2, [&]() { order.push_back("B" + std::to_string(level)); }, kCallbackPriorityNormal);
    cs.Call();
    EXPECT_EQ(order, (std::vector<std::string>{"A0", "A1", "A2", "B2", "B1", "B0"}));
  }

  // Register inside callback: not in current snapshot; visible on next call
  {
    CallbackSet<void()> cs;
    std::vector<int> order;
    bool registered = false;
    cs.Register(1, [&]() {
      order.push_back(1);
      if (!registered) {
        registered = true;
        cs.Register(2, [&]() { order.push_back(2); });
      }
    });
    cs.Call();
    EXPECT_EQ(order, (std::vector<int>{1}));
    order.clear();
    cs.Call();
    EXPECT_EQ(order.size(), 2u);
  }

  // Nested call: callback self-deregisters; outer snapshot still has it but skips it
  {
    CallbackSet<void()> cs;
    int cbACount = 0, cbBCount = 0;
    CallbackId idB = 2;
    bool nestedDone = false;
    cs.Register(
        1,
        [&]() {
          ++cbACount;
          if (!nestedDone) {
            nestedDone = true;
            cs.Call();
          }
        },
        kCallbackPriorityHigh);
    cs.Register(
        idB,
        [&]() {
          ++cbBCount;
          cs.Deregister(idB);
        },
        kCallbackPriorityNormal);
    cs.Call();
    EXPECT_EQ(cbACount, 2); // fires in outer and nested
    EXPECT_EQ(cbBCount, 1); // fires in nested, self-deregisters; outer skips
    EXPECT_EQ(idB, kInvalidCallbackId);
  }

  // Nested call: callback registered before nested Call is in nested snapshot; not in outer
  {
    CallbackSet<void()> cs;
    int cb1Count = 0, cb2Count = 0;
    bool nestedDone = false;
    cs.Register(1, [&]() {
      ++cb1Count;
      if (!nestedDone) {
        nestedDone = true;
        cs.Register(2, [&]() { ++cb2Count; });
        cs.Call();
      }
    });
    cs.Call();
    EXPECT_EQ(cb1Count, 2); // outer + nested
    EXPECT_EQ(cb2Count, 1); // nested snapshot only
  }

  // Nested call: Clear marks outer callbacks inactive; set empty and reusable after
  {
    CallbackSet<void()> cs;
    std::vector<int> order;
    bool nestedDone = false;
    cs.Register(
        1,
        [&]() {
          order.push_back(10);
          if (!nestedDone) {
            nestedDone = true;
            cs.Call();
          }
        },
        kCallbackPriorityHigh);
    cs.Register(
        2,
        [&]() {
          order.push_back(20);
          cs.Clear();
        },
        kCallbackPriorityNormal);
    cs.Call();
    EXPECT_EQ(order, (std::vector<int>{10, 10, 20})); // outer cb2 skipped after nested Clear
    EXPECT_TRUE(cs.IsEmpty());
    int postCount = 0;
    cs.Register(3, [&]() { ++postCount; });
    cs.Call();
    EXPECT_EQ(postCount, 1);
  }
}

TEST(CallbackSet, ExceptionSafety) {
  // Basic throw: _callDepth decremented, set fully usable after
  {
    CallbackSet<void()> cs;
    CallbackId id = 1;
    cs.Register(id, []() { throw std::runtime_error("test"); });
    EXPECT_THROW(cs.Call(), std::runtime_error);
    cs.Deregister(id);
    EXPECT_EQ(id, kInvalidCallbackId);
    int count = 0;
    cs.Register(2, [&]() { ++count; });
    cs.Call();
    EXPECT_EQ(count, 1);
  }

  // Regression: cleanup acquires only dataLock (callingLock already held).
  // cb1 deregisters cb2 → dirty=true; cb3 throws; cleanup must erase cb2 without deadlock.
  {
    CallbackSet<void()> cs;
    CallbackId idB = 2, idThrow = 3;
    cs.Register(1, [&]() { cs.Deregister(idB); }, kCallbackPriorityHigh);
    cs.Register(idB, []() {}, kCallbackPriorityNormal);
    cs.Register(idThrow, []() { throw std::runtime_error("test"); }, kCallbackPriorityLow);
    EXPECT_THROW(cs.Call(), std::runtime_error);
    EXPECT_EQ(idB, kInvalidCallbackId);
    cs.Deregister(idThrow);
    int count = 0;
    cs.Register(100, [&]() { ++count; });
    cs.Call();
    EXPECT_EQ(count, 1);
  }

  // Exception in second callback: first's side-effect persists; third never runs
  {
    CallbackSet<void()> cs;
    int aFired = 0;
    bool cFired = false;
    CallbackId id2 = 2;
    cs.Register(1, [&]() { ++aFired; }, kCallbackPriorityHigh);
    cs.Register(id2, []() { throw std::runtime_error("B throws"); }, kCallbackPriorityNormal);
    cs.Register(3, [&]() { cFired = true; }, kCallbackPriorityLow);
    EXPECT_THROW(cs.Call(), std::runtime_error);
    EXPECT_EQ(aFired, 1);
    EXPECT_FALSE(cFired);
    cs.Deregister(id2);
    cs.Call();
    EXPECT_EQ(aFired, 2);
    EXPECT_TRUE(cFired);
  }

  // Exception in nested call propagates through both levels; _callDepth returns to 0
  {
    CallbackSet<void()> cs;
    std::vector<int> order;
    bool nestedDone = false;
    CallbackId id2 = 2;
    cs.Register(
        1,
        [&]() {
          order.push_back(1);
          if (!nestedDone) {
            nestedDone = true;
            cs.Call();
          }
        },
        kCallbackPriorityHigh);
    cs.Register(
        id2,
        [&]() {
          order.push_back(2);
          throw std::runtime_error("nested throws");
        },
        kCallbackPriorityNormal);
    EXPECT_THROW(cs.Call(), std::runtime_error);
    EXPECT_EQ(order, (std::vector<int>{1, 1, 2}));
    cs.Deregister(id2);
    int count = 0;
    cs.Register(3, [&]() { ++count; });
    cs.Call();
    EXPECT_EQ(count, 1);
  }

  // Exception with predicate: predicate count is correct; set usable after
  {
    CallbackSet<void()> cs;
    int predCount = 0;
    CallbackId id2 = 2;
    cs.Register(1, []() {}, kCallbackPriorityHigh);
    cs.Register(
        id2, []() { throw std::runtime_error("callback throws"); }, kCallbackPriorityNormal);
    EXPECT_THROW(cs.Call([&]() { ++predCount; }), std::runtime_error);
    EXPECT_EQ(predCount, 1); // called after cb1, not after cb2
    cs.Deregister(id2);
    cs.Clear();
    EXPECT_TRUE(cs.IsEmpty());
  }
}

TEST(CallbackSet, NonVoidReturn) {
  // Plain Call() and operator() silently discard return values
  {
    CallbackSet<int(int const&)> cs;
    int fireCount = 0;
    cs.Register(1, [&](int v) {
      ++fireCount;
      return v * 2;
    });
    cs.Call(5);
    EXPECT_EQ(fireCount, 1);

    CallbackSet<int()> cs2;
    cs2.Register(1, [&]() {
      ++fireCount;
      return 42;
    });
    cs2();
    EXPECT_EQ(fireCount, 2);
  }

  // Deregister during non-void call works like void
  {
    CallbackSet<int()> cs;
    int cb1Count = 0, cb2Count = 0;
    CallbackId id2 = 2;
    cs.Register(
        1,
        [&]() {
          ++cb1Count;
          cs.Deregister(id2);
          return 1;
        },
        kCallbackPriorityHigh);
    cs.Register(
        id2,
        [&]() {
          ++cb2Count;
          return 2;
        },
        kCallbackPriorityNormal);
    std::vector<int> results;
    cs.Call([&](int v) { results.push_back(v); });
    EXPECT_EQ(cb1Count, 1);
    EXPECT_EQ(cb2Count, 0);
    EXPECT_EQ(results, (std::vector<int>{1}));
    EXPECT_EQ(id2, kInvalidCallbackId);
  }
}

TEST(CallbackSet, ConcurrentCallFromMultipleThreads) {
  CallbackSet<void()> cs;
  std::atomic<int> totalCalls{0};
  cs.Register(1, [&]() { ++totalCalls; });
  RunConcurrently(kNumThreads, kNumIterations, [&](int) { cs.Call(); });
  EXPECT_EQ(totalCalls.load(), kNumThreads * kNumIterations);
}

TEST(CallbackSet, ConcurrentNoCrash) {
  // Call + Register concurrently
  {
    CallbackSet<void()> cs;
    std::atomic<int> totalCalls{0};
    std::atomic<uint64_t> nextId{100};
    cs.Register(1, [&]() { ++totalCalls; });
    RunConcurrently(kNumThreads, kNumIterations, [&](int t) {
      if (t < kNumThreads / 2) {
        cs.Call();
      } else {
        cs.Register(nextId.fetch_add(1), [&]() { ++totalCalls; });
      }
    });
    EXPECT_GT(totalCalls.load(), 0);
  }

  // Call + Clear (with re-register) concurrently
  {
    CallbackSet<void()> cs;
    std::atomic<uint64_t> nextId{100};
    cs.Register(1, []() {});
    RunConcurrently(kNumThreads, kNumIterations, [&](int t) {
      if (t < kNumThreads / 2) {
        cs.Call();
      } else {
        cs.Clear();
        cs.Register(nextId.fetch_add(1), []() {});
      }
    });
  }

  // Register + Deregister concurrently: every registered callback is deregistered
  {
    CallbackSet<void()> cs;
    std::atomic<uint64_t> nextId{1};
    RunConcurrently(kNumThreads, kNumIterations, [&](int) {
      auto id = static_cast<CallbackId>(nextId.fetch_add(1));
      cs.Register(id, []() {});
      cs.Deregister(id);
      EXPECT_EQ(id, kInvalidCallbackId);
    });
  }

  // Concurrent self-deregister
  {
    constexpr int kCallbackCount = kNumThreads * kNumIterations;
    CallbackSet<void()> cs;
    std::array<CallbackId, kCallbackCount> ids;
    std::array<std::atomic<bool>, kCallbackCount> deregistered;
    for (int i = 0; i < kCallbackCount; ++i) {
      deregistered[i].store(false);
      ids[i] = i + 1;
      cs.Register(ids[i], [&cs, &ids, &deregistered, i]() {
        bool expected = false;
        if (deregistered[i].compare_exchange_strong(expected, true)) {
          cs.Deregister(ids[i]);
        }
      });
    }
    RunConcurrently(kNumThreads, kNumIterations, [&](int) { cs.Call(); });
  }
}

TEST(CallbackSet, ConcurrentCallAndDeregister) {
  // Permanent callback ensures minimum call count regardless of deregister timing.
  constexpr int kCallbackCount = 200;
  CallbackSet<void()> cs;
  std::atomic<int> totalCalls{0};
  cs.Register(kCallbackCount + 1, [&]() { ++totalCalls; }); // permanent
  std::vector<CallbackId> ids(kCallbackCount);
  for (uint64_t i = 0; i < kCallbackCount; ++i) {
    ids[i] = i + 1;
    cs.Register(ids[i], [&]() { ++totalCalls; });
  }
  std::atomic<int> deregIdx{0};
  RunConcurrently(kNumThreads, kNumIterations, [&](int t) {
    if (t < kNumThreads / 2) {
      cs.Call();
    } else {
      int const idx = deregIdx.fetch_add(1);
      if (idx < kCallbackCount) {
        cs.Deregister(ids[idx]);
      }
    }
  });
  EXPECT_GE(totalCalls.load(), (kNumThreads / 2) * kNumIterations);
}

TEST(CallbackSet, ConcurrentMixedStress) {
  CallbackSet<void()> cs;
  constexpr int kStressIterations = 1000;
  std::atomic<int> ready{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&, t]() {
      StartBarrier(ready, kNumThreads);
      for (int i = 0; i < kStressIterations; ++i) {
        auto id = static_cast<CallbackId>(static_cast<uint64_t>(t) * kStressIterations + i + 1);
        if (i % 4 == 0) {
          cs.Register(id, []() {});
        } else if (i % 4 == 1) {
          cs.Call();
        } else if (i % 4 == 2) {
          auto prev =
              static_cast<CallbackId>(static_cast<uint64_t>(t) * kStressIterations + (i - 2) + 1);
          cs.Deregister(prev);
        } else {
          cs.Clear();
        }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  // No crash or TSAN error is the primary assertion.
}

TEST(CallbackSet, ConcurrentIsEmptyAndRegister) {
  // Regression: IsEmpty() must hold _callbackDataMutex when reading _callbacks.empty().
  CallbackSet<void()> cs;
  std::atomic<uint64_t> nextId{1};
  std::atomic<bool> done{false};
  std::atomic<int> ready{0};

  std::thread mutator([&]() {
    StartBarrier(ready, 2);
    for (int i = 0; i < kNumIterations * 10; ++i) {
      cs.Register(nextId.fetch_add(1), []() {});
    }
    done.store(true, std::memory_order_release);
  });

  std::thread reader([&]() {
    StartBarrier(ready, 2);
    while (!done.load(std::memory_order_acquire)) {
      (void)cs.IsEmpty();
    }
    (void)cs.IsEmpty();
  });

  mutator.join();
  reader.join();
  // No crash or TSAN error is the primary assertion.
}

TEST(CallbackSet, ConcurrentCallWithReentrantCallback) {
  // Multiple threads call cs.Call() concurrently; one callback performs a single nested
  // cs.Call(). Verifies _callDepth accounting is correct under concurrent + reentrant use.
  CallbackSet<void()> cs;
  std::atomic<int> totalFires{0};
  std::atomic<bool> nestedDone{false};

  cs.Register(1, [&]() {
    ++totalFires;
    bool expected = false;
    if (nestedDone.compare_exchange_strong(expected, true)) {
      cs.Call(); // one nested call, performed only once across all threads
    }
  });

  RunConcurrently(kNumThreads, kNumIterations, [&](int) { cs.Call(); });
  // kNumThreads*kNumIterations outer calls + exactly 1 nested call.
  EXPECT_EQ(totalFires.load(), kNumThreads * kNumIterations + 1);
}
