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
#include <mochi_core/utils/guarded.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <numeric>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

using namespace mochi;

// Starts `numThreads` workers and releases them simultaneously via a spin gate so they contend on
// the guard at the same time. No sleeps are used for synchronization.
template <typename Fn>
static void RunConcurrently(int numThreads, Fn fn) {
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};

  DynamicArray<std::thread> threads;
  threads.reserve(numThreads);
  for (int t = 0; t < numThreads; ++t) {
    threads.emplace_back([&, t] {
      ready.fetch_add(1, std::memory_order_acq_rel);
      while (!go.load(std::memory_order_acquire)) {
      }
      fn(t);
    });
  }

  while (ready.load(std::memory_order_acquire) < numThreads) {
  }
  go.store(true, std::memory_order_release);

  for (auto& thread : threads) {
    thread.join();
  }
}

namespace {

struct Widget {
  int id = 0;
  std::string name;

  int Doubled() const {
    return id * 2;
  }
};

struct Pair {
  int a = 0;
  int b = 0;
};

// Move-only and NOT default constructible: proves guarded<> never requires a default constructor.
struct NoDefault {
  int v;
  explicit NoDefault(int x) : v(x) {}
  NoDefault(NoDefault&&) = default;
  NoDefault& operator=(NoDefault&&) = default;
  NoDefault(NoDefault const&) = delete;
  NoDefault& operator=(NoDefault const&) = delete;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// Construction, value access, and type traits (single-threaded)
// -------------------------------------------------------------------------------------------------

TEST(Guarded, ConstructLoadStoreExchange) {
  Guarded<int> def;
  EXPECT_EQ(def.Load(), 0);

  Guarded<int> g(42);
  EXPECT_EQ(g.Load(), 42);

  // Implicit conversion copies the value under a read lock.
  int const asValue = g;
  EXPECT_EQ(asValue, 42);

  g.Store(7);
  EXPECT_EQ(g.Load(), 7);

  g = 100;
  EXPECT_EQ(g.Load(), 100);

  int const old = g.Exchange(5);
  EXPECT_EQ(old, 100);
  EXPECT_EQ(g.Load(), 5);

  // Exchange() with no argument swaps in a default-constructed value.
  int const swappedOut = g.Exchange();
  EXPECT_EQ(swappedOut, 5);
  EXPECT_EQ(g.Load(), 0);

  // Assign from an int lvalue, and from a convertible type (const char* -> std::string).
  int const lvalue = 3;
  g = lvalue;
  EXPECT_EQ(g.Load(), 3);

  Guarded<std::string> s;
  s = "literal";
  EXPECT_EQ(s.Read([](auto const& str) { return str; }), "literal");
}

TEST(Guarded, ForwardingConstructorBuildsValueInPlace) {
  Guarded<DynamicArray<int>> v(3, 7);
  EXPECT_EQ(v.Read([](auto const& vec) { return vec; }), (DynamicArray<int>{7, 7, 7}));

  Guarded<std::string> s("hi");
  EXPECT_EQ(s.Read([](auto const& str) { return str; }), "hi");
}

TEST(Guarded, CopyDeletedMoveAllowed) {
  static_assert(!std::is_copy_constructible_v<Guarded<int>>);
  static_assert(!std::is_copy_assignable_v<Guarded<int>>);
  static_assert(std::is_move_constructible_v<Guarded<int>>);
  static_assert(std::is_move_assignable_v<Guarded<int>>);

  Guarded<int> a(5);
  Guarded<int> b(std::move(a));
  EXPECT_EQ(b.Load(), 5);

  Guarded<int> c(1);
  c = std::move(b);
  EXPECT_EQ(c.Load(), 5);

  // Self-move-assignment is a no-op (covers the this == &rhs guard). The reference indirection
  // avoids a -Wself-move diagnostic.
  Guarded<int>& cRef = c;
  c = std::move(cRef);
  EXPECT_EQ(c.Load(), 5);
}

TEST(Guarded, SupportsNonDefaultConstructibleValue) {
  static_assert(!std::is_default_constructible_v<NoDefault>);

  // Move-construction must not require value_type to be default constructible.
  Guarded<NoDefault> g(NoDefault{5});
  Guarded<NoDefault> moved(std::move(g));
  EXPECT_EQ(moved.Read([](auto const& n) { return n.v; }), 5);

  // Move-assignment into an already-constructed guard.
  Guarded<NoDefault> target(NoDefault{1});
  target = std::move(moved);
  EXPECT_EQ(target.Read([](auto const& n) { return n.v; }), 5);
}

// -------------------------------------------------------------------------------------------------
// Return-value flow out of guarded sections
// -------------------------------------------------------------------------------------------------

TEST(Guarded, ReadAndMutateReturnValuesByValue) {
  Guarded<int> g(10);

  int const doubled = g.Read([](auto const& v) { return v * 2; });
  EXPECT_EQ(doubled, 20);

  // Mutate returns whatever the callback returns; the lock is released afterwards.
  int const previous = g.Mutate([](auto& v) {
    int const old = v;
    v = 99;
    return old;
  });
  EXPECT_EQ(previous, 10);
  EXPECT_EQ(g.Load(), 99);

  int const unsafe = g.UnsafeRead([](auto const& v) { return v; });
  EXPECT_EQ(unsafe, 99);

  // Extra arguments are forwarded through to the callback.
  int const sum = g.Mutate([](auto& v, int add) { return v + add; }, 1);
  EXPECT_EQ(sum, 100);
}

TEST(Guarded, WithLockVariantsHoldLockDuringCallback) {
  Guarded<int> g(3);

  bool const readOwnsLock =
      g.ReadWithLock([](auto& lock, auto const&) { return lock.owns_lock(); });
  EXPECT_TRUE(readOwnsLock);

  g.MutateWithLock([](auto& lock, auto& v) {
    EXPECT_TRUE(lock.owns_lock());
    v = 21;
  });
  EXPECT_EQ(g.Load(), 21);
}

TEST(Guarded, MemberPointerAccess) {
  Guarded<Widget> g(Widget{2, "alpha"});

  EXPECT_EQ(g.LoadMemberObject(&Widget::id), 2);
  EXPECT_EQ(g.Read(&Widget::Doubled), 4);

  g.StoreMemberObject(&Widget::id, 8);
  EXPECT_EQ(g.LoadMemberObject(&Widget::id), 8);

  int const old = g.ExchangeMemberObject(&Widget::id, 9);
  EXPECT_EQ(old, 8);
  EXPECT_EQ(g.LoadMemberObject(&Widget::id), 9);
}

TEST(Guarded, ConstObjectReadPaths) {
  // Every read-side method is const-qualified; exercise them through const objects so a break in
  // const-correctness fails to compile.
  Guarded<int> const g(5);
  EXPECT_EQ(g.Load(), 5);
  EXPECT_EQ(g.Read([](auto const& v) { return v; }), 5);
  EXPECT_EQ(g.UnsafeRead([](auto const& v) { return v; }), 5);
  EXPECT_TRUE(g.ReadWithLock([](auto& lock, auto const&) { return lock.owns_lock(); }));
  int const asValue = g;
  EXPECT_EQ(asValue, 5);

  Guarded<Widget> const w(Widget{2, "beta"});
  EXPECT_EQ(w.LoadMemberObject(&Widget::id), 2);
  EXPECT_EQ(w.Read(&Widget::name), "beta");

  // The shared-mutex read path takes a shared_lock, which is a distinct const code path.
  SharedGuarded<int> const sg(9);
  EXPECT_EQ(sg.Read([](auto const& v) { return v; }), 9);
  EXPECT_EQ(sg.Load(), 9);
}

TEST(Guarded, MemberPointerReturnsAreDecayedCopies) {
  // The headline safety property: member-pointer access returns a value (copy), never a reference
  // into the guarded object, so nothing can escape the lock. Proven on a non-trivial member.
  static_assert(std::is_same_v<
                decltype(std::declval<Guarded<Widget>&>().LoadMemberObject(&Widget::name)),
                std::string>);
  static_assert(std::is_same_v<
                decltype(std::declval<Guarded<Widget> const&>().Read(&Widget::name)),
                std::string>);

  Guarded<Widget> g(Widget{7, "hello"});
  EXPECT_EQ(g.LoadMemberObject(&Widget::name), "hello");
  EXPECT_EQ(g.Read(&Widget::name), "hello");
}

// -------------------------------------------------------------------------------------------------
// TryMutate: runs only when the lock is free
// -------------------------------------------------------------------------------------------------

TEST(Guarded, TryMutateRunsWhenUncontended) {
  Guarded<int> g(0);

  bool const ran = g.TryMutate([](auto& v) { v = 5; });
  EXPECT_TRUE(ran);
  EXPECT_EQ(g.Load(), 5);
}

TEST(Guarded, TryMutateFailsWhenLockHeld) {
  Guarded<int> g(1);

  std::atomic<bool> lockHeld{false};
  std::atomic<bool> release{false};

  std::thread holder([&] {
    g.MutateWithLock([&](auto&, auto&) {
      lockHeld.store(true, std::memory_order_release);
      while (!release.load(std::memory_order_acquire)) {
      }
    });
  });

  while (!lockHeld.load(std::memory_order_acquire)) {
  }

  bool const ran = g.TryMutate([](auto& v) { v = 999; });
  EXPECT_FALSE(ran);

  release.store(true, std::memory_order_release);
  holder.join();

  // The value was never modified by the failed TryMutate.
  EXPECT_EQ(g.Load(), 1);
}

// -------------------------------------------------------------------------------------------------
// Multi-threaded serialization: these fail (lost updates / TSAN races) without a real lock
// -------------------------------------------------------------------------------------------------

TEST(Guarded, ConcurrentMutationsHaveNoLostUpdates) {
  constexpr int kThreads = 8;
  constexpr int kIncrements = 5000;

  Guarded<long long> counter(0);

  RunConcurrently(kThreads, [&](int) {
    for (int i = 0; i < kIncrements; ++i) {
      counter.Mutate([](auto& v) { ++v; });
    }
  });

  EXPECT_EQ(counter.Load(), static_cast<long long>(kThreads) * kIncrements);
}

TEST(Guarded, ConcurrentContainerInsertionsPreserveAllElements) {
  constexpr int kThreads = 8;
  constexpr int kPerThread = 1000;

  Guarded<DynamicArray<int>> values;

  RunConcurrently(kThreads, [&](int t) {
    for (int i = 0; i < kPerThread; ++i) {
      int const value = t * kPerThread + i;
      values.Mutate([value](auto& v) { v.push_back(value); });
    }
  });

  // Returning the container by value also exercises the return-value path.
  DynamicArray<int> result = values.Mutate([](auto& v) { return v; });
  ASSERT_EQ(result.size(), static_cast<size_t>(kThreads * kPerThread));

  std::sort(result.begin(), result.end());
  DynamicArray<int> expected(kThreads * kPerThread);
  std::iota(expected.begin(), expected.end(), 0);
  EXPECT_EQ(result, expected);
}

// -------------------------------------------------------------------------------------------------
// SharedGuarded: concurrent readers never observe a half-applied mutation
// -------------------------------------------------------------------------------------------------

TEST(SharedGuarded, ConcurrentReadersNeverSeeTornWrites) {
  constexpr int kWriters = 4;
  constexpr int kWrites = 5000;
  constexpr int kReaders = 4;
  constexpr int kReads = 20000;

  // Invariant maintained by writers: a == b at all times outside a write section.
  SharedGuarded<Pair> shared(Pair{0, 0});

  std::atomic<long long> inconsistentReads{0};

  RunConcurrently(kWriters + kReaders, [&](int t) {
    if (t < kWriters) {
      for (int i = 0; i < kWrites; ++i) {
        shared.Mutate([](auto& p) {
          ++p.a;
          ++p.b;
        });
      }
    } else {
      for (int i = 0; i < kReads; ++i) {
        bool const consistent = shared.Read([](auto const& p) { return p.a == p.b; });
        if (!consistent) {
          inconsistentReads.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
  });

  EXPECT_EQ(inconsistentReads.load(), 0);

  Pair const finalPair = shared.Read([](auto const& p) { return p; });
  EXPECT_EQ(finalPair.a, kWriters * kWrites);
  EXPECT_EQ(finalPair.b, kWriters * kWrites);
}

TEST(SharedGuarded, AllowsConcurrentReaders) {
  if (std::thread::hardware_concurrency() < 2) {
    GTEST_SKIP() << "Concurrent reader overlap requires at least 2 hardware threads.";
  }

  constexpr int kReaders = 4;
  constexpr int kRequiredOverlap = 2;
  constexpr auto kTimeout = std::chrono::seconds(60); // Way more than enough time.

  SharedGuarded<int> shared(0);
  std::atomic<int> activeReaders{0};
  std::atomic<int> maxObserved{0};
  std::atomic<int> arrived{0};

  RunConcurrently(kReaders, [&](int) {
    shared.Read([&](auto const&) {
      int const now = activeReaders.fetch_add(1, std::memory_order_acq_rel) + 1;
      int prev = maxObserved.load(std::memory_order_relaxed);
      while (prev < now &&
             !maxObserved.compare_exchange_weak(prev, now, std::memory_order_relaxed)) {
      }
      // Hold the shared lock until enough peer readers have entered, so overlap is observed
      // deterministically rather than relying on a timing-sensitive spin. The bounded timeout
      // ensures the test cannot hang if reads were wrongly exclusive (it just fails the final
      // assertion instead).
      arrived.fetch_add(1, std::memory_order_acq_rel);
      auto const deadline = std::chrono::steady_clock::now() + kTimeout;
      while (arrived.load(std::memory_order_acquire) < kRequiredOverlap &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
      }
      activeReaders.fetch_sub(1, std::memory_order_acq_rel);
    });
  });

  // With an exclusive lock this would never exceed 1.
  EXPECT_GE(maxObserved.load(), kRequiredOverlap);
}

// -------------------------------------------------------------------------------------------------
// Cross-traits move (Guarded <-> SharedGuarded)
// -------------------------------------------------------------------------------------------------

TEST(Guarded, CrossTraitsMoveTransfersValue) {
  Guarded<int> g(0);
  SharedGuarded<int> sg(7);
  g = std::move(sg);
  EXPECT_EQ(g.Load(), 7);

  SharedGuarded<int> sg2(0);
  Guarded<int> g2(42);
  sg2 = std::move(g2);
  EXPECT_EQ(sg2.Load(), 42);
}

TEST(Guarded, CrossTraitsMoveIsDeadlockFreeUnderContention) {
  constexpr int kIters = 5000;

  // Both start equal, so every move copies the same value regardless of interleaving; the result is
  // deterministic. The test would hang (deadlock) if the two assignment directions used an
  // inconsistent lock order.
  Guarded<int> g1(5);
  SharedGuarded<int> g2(5);

  std::thread forward([&] {
    for (int i = 0; i < kIters; ++i) {
      g1 = std::move(g2);
    }
  });
  std::thread reverse([&] {
    for (int i = 0; i < kIters; ++i) {
      g2 = std::move(g1);
    }
  });

  forward.join();
  reverse.join();

  EXPECT_EQ(g1.Load(), 5);
  EXPECT_EQ(g2.Load(), 5);
}

// -------------------------------------------------------------------------------------------------
// RecursiveGuarded: same thread may re-enter the lock
// -------------------------------------------------------------------------------------------------

TEST(RecursiveGuarded, ReentrantMutateOnSameThread) {
  RecursiveGuarded<int> g(0);

  g.Mutate([&](auto& v) {
    v += 1;
    // Re-entering the same guard from the holding thread would deadlock a non-recursive mutex.
    g.Mutate([](auto& inner) { inner += 10; });
  });

  EXPECT_EQ(g.Load(), 11);
}
