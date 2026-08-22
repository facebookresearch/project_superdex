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

#include "config.h"

#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/rand_utils.h>

#include <algorithm>
#include <cstddef>
#include <vector>

static constexpr int kAllocCount = 1000;

static auto GetRandSizes(size_t maxValue) {
  std::vector<size_t> sizes(kAllocCount);
  auto rand = mochi::RandomGenerator(123);
  mochi::SetRandom(rand, size_t(1), maxValue, mochi::MakeSpan(sizes));
  return sizes;
}

static void MallocAllocatorRandSizes(benchmark::State& state, size_t maxAllocSize) {
  auto sizes = GetRandSizes(maxAllocSize);
  std::vector<void*> pointers(sizes.size());
  for (auto _ : state) {
    for (int i = 0; i < kAllocCount; ++i) {
      pointers[i] = malloc(sizes[i]);
    }
    for (int i = kAllocCount - 1; i >= 0; --i) {
      free(pointers[i]);
    }
  }
}

static void FiloAllocatorRandSizes(
    benchmark::State& state,
    size_t maxAllocSize,
    size_t reserveSize,
    size_t pageSize) {
  auto sizes = GetRandSizes(maxAllocSize);
  std::vector<void*> pointers(sizes.size());

  // Reserve extra memory for the header so that 'reserveSize' bytes are available for allocation.
  // Then round up to a multiple of kMinAlignment, as required by FiloAllocator. Note that these
  // steps are usually performed at compile time by MOCHI_FILO_STACK_ALLOCATOR, but we do it
  // manually here so the reserve size can e dynamic.
  reserveSize += mochi::FiloAllocator::kHeaderSize;
  reserveSize = mochi::RoundUp(reserveSize, mochi::FiloAllocator::kMinAlignment);

  // If (reserveSize == 0), then use the minimum buffer size so that we can use one code path. The
  // minimum buffer size will not be sufficient, so the first allocation will come from the heap
  // (same as if we had not reserved anything).
  reserveSize = std::max(reserveSize, mochi::FiloAllocator::kMinBufferSize);

  // Similar to MOCHI_FILO_STACK_ALLOCATOR(alloc, reserveSize) but the size is dynamic
  std::vector<std::byte> buffer(reserveSize);
  mochi::FiloAllocator alloc(buffer.data(), buffer.size());
  alloc.SetNextPageSize(pageSize);

  for (auto _ : state) {
    for (int i = 0; i < kAllocCount; ++i) {
      pointers[i] = alloc.allocate(sizes[i]);
    }
    for (int i = kAllocCount - 1; i >= 0; --i) {
      alloc.deallocate(pointers[i], sizes[i]);
    }
  }
}

BENCHMARK_CAPTURE(MallocAllocatorRandSizes, MAX_4, 4);
BENCHMARK_CAPTURE(MallocAllocatorRandSizes, MAX_40, 40);
BENCHMARK_CAPTURE(MallocAllocatorRandSizes, MAX_400, 400);
BENCHMARK_CAPTURE(MallocAllocatorRandSizes, MAX_4000, 4000);
BENCHMARK_CAPTURE(MallocAllocatorRandSizes, MAX_40000, 40000);

// Benchmark for given reserve size (res) and page size (pg)
#define MOCHI_BENCHMARK_FILO_ALLOCATOR(res, pg)                                             \
  BENCHMARK_CAPTURE(FiloAllocatorRandSizes, RES_##res##_PG_##pg##_MAX_4, 4, res, pg);       \
  BENCHMARK_CAPTURE(FiloAllocatorRandSizes, RES_##res##_PG_##pg##_MAX_40, 40, res, pg);     \
  BENCHMARK_CAPTURE(FiloAllocatorRandSizes, RES_##res##_PG_##pg##_MAX_400, 400, res, pg);   \
  BENCHMARK_CAPTURE(FiloAllocatorRandSizes, RES_##res##_PG_##pg##_MAX_4000, 4000, res, pg); \
  BENCHMARK_CAPTURE(FiloAllocatorRandSizes, RES_##res##_PG_##pg##_MAX_40000, 40000, res, pg);

MOCHI_BENCHMARK_FILO_ALLOCATOR(0, 4096);
MOCHI_BENCHMARK_FILO_ALLOCATOR(512, 4096);
MOCHI_BENCHMARK_FILO_ALLOCATOR(4096, 4096);
MOCHI_BENCHMARK_FILO_ALLOCATOR(65536, 65536);
MOCHI_BENCHMARK_FILO_ALLOCATOR(64000000, 4096);
