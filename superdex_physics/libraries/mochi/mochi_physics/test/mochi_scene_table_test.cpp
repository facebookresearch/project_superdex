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

#include <mochi_physics/src/mochi_scene_table.h>

#include <mochi_core/utils/dynamic_array.h>

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <latch>
#include <limits>
#include <thread>
#include <unordered_set>

namespace mochi {
namespace {

// A distinct fake scene per handle. The table never dereferences scenes.
Scene* FakeScene(SceneHandle handle) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<Scene*>(static_cast<uintptr_t>(handle.value) << 4);
}

TEST(SceneTable, FindsOnlyPublishedScenes) {
  SceneTable table(4);
  SceneHandle const handle = table.Reserve();
  ASSERT_TRUE(handle.IsValid());
  EXPECT_EQ(table.Find(handle), nullptr);

  table.Publish(handle, FakeScene(handle));
  EXPECT_EQ(table.Find(handle), FakeScene(handle));
  EXPECT_EQ(table.Find(SceneHandle{handle.value | (uint64_t{1} << 32)}), nullptr);
  EXPECT_EQ(table.Find(SceneHandle{handle.value + 4}), nullptr); // same slot, another handle
  EXPECT_EQ(table.Find(SceneHandle{}), nullptr);

  table.Release(handle);
  EXPECT_EQ(table.Find(handle), nullptr);
}

TEST(SceneTable, StaleHandleMissesTheSceneThatReusesItsSlot) {
  SceneTable table(2);
  SceneHandle const stale = table.Reserve();
  table.Publish(stale, FakeScene(stale));
  table.Release(stale);
  ASSERT_TRUE(table.Reserve().IsValid()); // the other slot
  SceneHandle const reused = table.Reserve();
  table.Publish(reused, FakeScene(reused));

  EXPECT_EQ(reused.value % 2, stale.value % 2);
  EXPECT_EQ(table.Find(stale), nullptr);
  EXPECT_EQ(table.Find(reused), FakeScene(reused));
}

TEST(SceneTable, ReserveFailsOnlyWhileEverySlotIsTaken) {
  SceneTable table;
  DynamicArray<SceneHandle> handles(SceneTable::kMaxCapacity);
  for (SceneHandle& handle : handles) {
    handle = table.Reserve();
    ASSERT_TRUE(handle.IsValid());
  }
  EXPECT_FALSE(table.Reserve().IsValid());

  table.Release(handles[100]);
  EXPECT_TRUE(table.Reserve().IsValid());
}

TEST(SceneTable, HandlesIncreaseWhileASlotStaysTaken) {
  SceneTable table(2);
  SceneHandle const held = table.Reserve();
  // Reuse the other slot several times while the held slot is skipped.
  SceneHandle previous = held;
  for (int i = 0; i < 3; ++i) {
    SceneHandle const handle = table.Reserve();
    EXPECT_GT(handle.value, previous.value);
    table.Release(handle);
    previous = handle;
  }
  table.Release(held);
  EXPECT_GT(table.Reserve().value, previous.value);
}

TEST(SceneTable, ReserveFailsAfterTheLastHandle) {
  constexpr uint32_t kLast = std::numeric_limits<uint32_t>::max();
  SceneTable table(2, kLast - 1);
  SceneHandle const first = table.Reserve();
  table.Release(first);
  SceneHandle const last = table.Reserve();
  table.Release(last);

  EXPECT_EQ(first.value, kLast - 1);
  EXPECT_EQ(last.value, kLast);
  EXPECT_FALSE(table.Reserve().IsValid());
}

TEST(SceneTable, WritersOnManyThreadsNeverShareAHandle) {
  constexpr int kWriters = 4;
  constexpr int kChurnsPerWriter = 20'000;
  // More slots than writers, so Reserve never runs out, but few enough that writers reuse each
  // other's slots. Two writers given one handle, or a Release racing the slot's next owner, fails
  // an assertion in Publish or Release.
  SceneTable table(8);
  std::latch start{kWriters};
  std::atomic<int> wrong{0};
  std::array<DynamicArray<uint64_t>, kWriters> reserved;
  DynamicArray<std::thread> writers;
  writers.reserve(kWriters);
  for (int w = 0; w < kWriters; ++w) {
    writers.emplace_back([&, w] {
      start.arrive_and_wait();
      for (int i = 0; i < kChurnsPerWriter; ++i) {
        SceneHandle const handle = table.Reserve();
        if (!handle.IsValid()) {
          wrong.fetch_add(1);
          continue;
        }
        reserved[w].push_back(handle.value);
        table.Publish(handle, FakeScene(handle));
        wrong.fetch_add(table.Find(handle) != FakeScene(handle) ? 1 : 0);
        table.Release(handle);
      }
    });
  }
  for (std::thread& writer : writers) {
    writer.join();
  }
  EXPECT_EQ(wrong.load(), 0);

  // Handle values are never reused, so no two reservations share one.
  std::unordered_set<uint64_t> unique;
  size_t total = 0;
  for (DynamicArray<uint64_t> const& values : reserved) {
    unique.insert(values.begin(), values.end());
    total += values.size();
  }
  EXPECT_EQ(unique.size(), total);
}

TEST(SceneTable, FindNeverReturnsAnotherHandlesSceneWhileSlotsAreReused) {
  constexpr int kReaders = 4;
  constexpr int kStable = 4;
  constexpr int kChurned = 4;
  constexpr int kMinChurns = 20'000;
  constexpr int kMaxChurns = 5'000'000; // each churn uses at most 8 handle values
  constexpr int kMinReaderPasses = 1'000;
  // Every slot is taken, so each churn reuses the slot it just freed, under a new handle.
  SceneTable table(kStable + kChurned);
  std::array<SceneHandle, kStable> stable;
  for (SceneHandle& handle : stable) {
    handle = table.Reserve();
    table.Publish(handle, FakeScene(handle));
  }
  std::array<std::atomic<uint64_t>, kChurned> churned{};
  for (auto& value : churned) {
    SceneHandle const handle = table.Reserve();
    table.Publish(handle, FakeScene(handle));
    value.store(handle.value);
  }
  // Handles increase, so a handle above this one was reserved while churning.
  uint64_t const lastBeforeChurn = churned.back().load();

  std::latch start{kReaders + 1};
  std::atomic<bool> churning{false};
  std::atomic<bool> done{false};
  std::atomic<int> passesWhileChurning{0};
  std::atomic<int> wrongLookups{0};
  std::atomic<int> churnedHits{0};
  DynamicArray<std::thread> readers;
  readers.reserve(kReaders);
  for (int r = 0; r < kReaders; ++r) {
    readers.emplace_back([&] {
      int wrong = 0;
      int hits = 0;
      start.arrive_and_wait();
      while (!done.load()) {
        bool const startedWhileChurning = churning.load();
        for (SceneHandle const handle : stable) {
          wrong += table.Find(handle) != FakeScene(handle) ? 1 : 0;
        }
        for (auto const& value : churned) {
          SceneHandle const handle{value.load()};
          Scene* const scene = table.Find(handle);
          wrong += scene != nullptr && scene != FakeScene(handle) ? 1 : 0;
          hits += scene != nullptr && handle.value > lastBeforeChurn ? 1 : 0;
        }
        // Slot 0 is churned, so this also looks up a slot while it is being freed.
        wrong += table.Find(SceneHandle{}) != nullptr ? 1 : 0;
        if (startedWhileChurning && !done.load()) {
          passesWhileChurning.fetch_add(1);
        }
      }
      wrongLookups.fetch_add(wrong);
      churnedHits.fetch_add(hits);
    });
  }
  start.arrive_and_wait();
  churning.store(true);
  // Churn until the readers have made their passes during it; past the minimum, yield so they can.
  for (int i = 0;
       i < kMaxChurns && (i < kMinChurns || passesWhileChurning.load() < kMinReaderPasses);
       ++i) {
    std::atomic<uint64_t>& value = churned[i % kChurned];
    table.Release(SceneHandle{value.load()});
    SceneHandle const handle = table.Reserve();
    table.Publish(handle, FakeScene(handle));
    value.store(handle.value);
    if (i >= kMinChurns) {
      std::this_thread::yield();
    }
  }
  done.store(true);
  for (std::thread& reader : readers) {
    reader.join();
  }

  EXPECT_EQ(wrongLookups.load(), 0);
  EXPECT_GE(passesWhileChurning.load(), kMinReaderPasses);
  EXPECT_GT(churnedHits.load(), 0);
}

} // namespace
} // namespace mochi
