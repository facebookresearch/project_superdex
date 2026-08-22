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

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/memory/allocator.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/simd.h>

#include <cmath>
#include <functional>
#include <utility>

#define MOCHI_SIMD_BENCHMARK_VEC2D(fn, ...) BENCHMARK_TEMPLATE(fn, __VA_ARGS__)
#define MOCHI_SIMD_BENCHMARK_VEC4D(fn, ...) BENCHMARK_TEMPLATE(fn, __VA_ARGS__)
#define MOCHI_SIMD_BENCHMARK_VEC4F(fn, ...) BENCHMARK_TEMPLATE(fn, __VA_ARGS__)
#define MOCHI_SIMD_BENCHMARK_VEC4I(fn, ...) BENCHMARK_TEMPLATE(fn, __VA_ARGS__)
#define MOCHI_SIMD_BENCHMARK_VEC8F(fn, ...) BENCHMARK_TEMPLATE(fn, __VA_ARGS__)
#define MOCHI_SIMD_BENCHMARK_VEC8I(fn, ...) BENCHMARK_TEMPLATE(fn, __VA_ARGS__)

// Place your code between these macros to measure instruction latency.
// Subtract the latency of the pair of macros with nothing in between.
#if MOCHI_ARCH_ARM_NEON
#define MOCHI_CPU_LATENCY_TEST_PRE() asm volatile("dsb sy\nisb" ::: "memory");
#define MOCHI_CPU_LATENCY_TEST_POST() asm volatile("dsb sy\nisb" ::: "memory");
#elif MOCHI_ARCH_X64
#if MOCHI_COMPILER_MSVC
#include <intrin.h>
#define MOCHI_CPU_LATENCY_TEST_PRE() _mm_lfence()
#define MOCHI_CPU_LATENCY_TEST_POST() \
  do {                                \
    _mm_lfence();                     \
    __rdtsc();                        \
  } while (0)
#else
#define MOCHI_CPU_LATENCY_TEST_PRE() asm volatile("lfence" ::: "memory");
#define MOCHI_CPU_LATENCY_TEST_POST() asm volatile("lfence\nrdtsc" : : : "eax", "edx", "memory");
#endif
#endif

namespace mochi_benchmark {

using Vec2d = mochi::Vec2d;
using Vec4d = mochi::Vec4d;
using Vec4f = mochi::Vec4f;
using Vec4i = mochi::Vec4i;
using Vec8f = mochi::Vec8f;
using Vec8i = mochi::Vec8i;

/****************************************************************************************
  LoadIndexed
*/
template <class V>
static void LoadIndexed(benchmark::State& state) {
  auto const indices = mochi::Simd<int, V::kSize>(0);
  alignas(alignof(V)) typename V::Scalar const buf[2] = {};
  auto const* src = buf + 1; // not aligned
  V v = {};
  for (auto x : state) {
    v = mochi::LoadIndexed<V>(src, indices);
    MOCHI_NO_DISCARD_IN_LOOP(v);
  }
  state.counters["Values/second"] =
      benchmark::Counter(state.iterations() * V::kSize, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(v);
}
MOCHI_SIMD_BENCHMARK_VEC4D(LoadIndexed, Vec4d);
MOCHI_SIMD_BENCHMARK_VEC4F(LoadIndexed, Vec4f);
MOCHI_SIMD_BENCHMARK_VEC8F(LoadIndexed, Vec8f);

/****************************************************************************************
  Load
*/
namespace {
template <class V>
static void Load(benchmark::State& state) {
  alignas(alignof(V)) typename V::Scalar const buf[V::kSize + 1] = {};
  auto const* src = buf + 1; // not aligned
  V v = {};
  for (auto x : state) {
    v = mochi::Load<V>(src);
    MOCHI_NO_DISCARD_IN_LOOP(v);
  }
  state.counters["Values/second"] =
      benchmark::Counter(state.iterations() * V::kSize, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(v);
}
MOCHI_SIMD_BENCHMARK_VEC2D(Load, Vec2d);
MOCHI_SIMD_BENCHMARK_VEC4D(Load, Vec4d);
MOCHI_SIMD_BENCHMARK_VEC4F(Load, Vec4f);
MOCHI_SIMD_BENCHMARK_VEC4I(Load, Vec4i);
MOCHI_SIMD_BENCHMARK_VEC8F(Load, Vec8f);
MOCHI_SIMD_BENCHMARK_VEC8I(Load, Vec8i);
} // namespace

/****************************************************************************************
  Load<N>
*/
namespace {
template <int N, class V>
static void Load(benchmark::State& state) {
  alignas(alignof(V)) typename V::Scalar const buf[N + 1] = {};
  auto const* src = buf + 1; // not aligned
  V v = {};
  for (auto x : state) {
    v = mochi::Load<N, V>(src);
    MOCHI_NO_DISCARD_IN_LOOP(v);
  }
  state.counters["Values/second"] =
      benchmark::Counter(state.iterations() * N, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(v);
}
MOCHI_SIMD_BENCHMARK_VEC2D(Load, 1, Vec2d);
MOCHI_SIMD_BENCHMARK_VEC4D(Load, 3, Vec4d);
MOCHI_SIMD_BENCHMARK_VEC4F(Load, 3, Vec4f);
MOCHI_SIMD_BENCHMARK_VEC4I(Load, 3, Vec4i);
MOCHI_SIMD_BENCHMARK_VEC8F(Load, 7, Vec8f);
MOCHI_SIMD_BENCHMARK_VEC8I(Load, 7, Vec8i);
} // namespace

/****************************************************************************************
  Load(ptr, n)
*/
namespace {
template <class V>
static void VLoad_n(benchmark::State& state) {
  alignas(alignof(V)) typename V::Scalar const buf[V::kSize + 1] = {};
  auto const* src = buf + 1; // not aligned
  V v = {};
  int n = DoNotOptimizeRuntimeVar(V::kSize - 1);
  for (auto x : state) {
    v = mochi::Load<V>(src, n);
    MOCHI_NO_DISCARD_IN_LOOP(v);
  }
  state.counters["Values/second"] =
      benchmark::Counter(state.iterations() * n, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(v);
}
MOCHI_SIMD_BENCHMARK_VEC2D(VLoad_n, Vec2d);
MOCHI_SIMD_BENCHMARK_VEC4D(VLoad_n, Vec4d);
MOCHI_SIMD_BENCHMARK_VEC4F(VLoad_n, Vec4f);
MOCHI_SIMD_BENCHMARK_VEC4I(VLoad_n, Vec4i);
MOCHI_SIMD_BENCHMARK_VEC8F(VLoad_n, Vec8f);
MOCHI_SIMD_BENCHMARK_VEC8I(VLoad_n, Vec8i);
} // namespace

/****************************************************************************************
  LoadTransposed
*/
namespace {
template <class T, int N, int M, int kVersion, bool kTestLatency>
static void LoadTransposed(benchmark::State& state) {
  static_assert(M == 3, "Only supported interleaving stride for now");

  // Allocate a 64-byte aligned buffer. This is the size of a cache line on all of our current
  // platforms. Alignment ensures that the same number of cache lines are involved, in case that
  // affects micro-op operformance.
  constexpr int kAlignment = 64;
  auto memSize = M * N * sizeof(T) * 2 + kAlignment;
  auto* mem = mochi::GetDefaultAllocator();
  auto* buf = static_cast<T*>(mem->allocate(memSize, kAlignment));
  memset(buf, 0, memSize);
  MOCHI_DEFER(mem->deallocate(buf, memSize, kAlignment));

  using V = mochi::Simd<T, N>;
  V a, b, c;
  for (auto _ : state) {
    if constexpr (kTestLatency) {
      MOCHI_CPU_LATENCY_TEST_PRE();
    }

    if constexpr (kVersion == 0) {
      // Empty
    } else if constexpr (kVersion == 1) {
      // Scalar code to be optimized by compiler
      if constexpr (N == 4) {
        a = V{buf[0], buf[3], buf[6], buf[9]};
        b = V{buf[1], buf[4], buf[7], buf[10]};
        c = V{buf[2], buf[5], buf[8], buf[11]};
      } else {
        a = V{buf[0], buf[3], buf[6], buf[9], buf[12], buf[15], buf[18], buf[21]};
        b = V{buf[1], buf[4], buf[7], buf[10], buf[13], buf[16], buf[19], buf[22]};
        c = V{buf[2], buf[5], buf[8], buf[11], buf[14], buf[17], buf[20], buf[23]};
      }
    } else if constexpr (kVersion == 2) {
      // Mochi's version
      mochi::LoadTransposed(buf, a, b, c);
    }

    if constexpr (kTestLatency) {
      MOCHI_CPU_LATENCY_TEST_POST();
    }

    MOCHI_NO_DISCARD_IN_LOOP(a);
    MOCHI_NO_DISCARD_IN_LOOP(b);
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }

  benchmark::DoNotOptimize(a);
  benchmark::DoNotOptimize(b);
  benchmark::DoNotOptimize(c);
}

// Load Vec4r[3] transposed
BENCHMARK_TEMPLATE(LoadTransposed, mochi::real, 4, 3, 0, false)
    ->Name("SIMD/LoadTransposed/Vec4r/Baseline/Throughput");
BENCHMARK_TEMPLATE(LoadTransposed, mochi::real, 4, 3, 0, true)
    ->Name("SIMD/LoadTransposed/Vec4r/Baseline/Latency");
BENCHMARK_TEMPLATE(LoadTransposed, mochi::real, 4, 3, 1, false)
    ->Name("SIMD/LoadTransposed/Vec4r/Scalar/Throughput");
BENCHMARK_TEMPLATE(LoadTransposed, mochi::real, 4, 3, 1, true)
    ->Name("SIMD/LoadTransposed/Vec4r/Scalar/Latency");
BENCHMARK_TEMPLATE(LoadTransposed, mochi::real, 4, 3, 2, false)
    ->Name("SIMD/LoadTransposed/Vec4r/Mochi/Throughput");
BENCHMARK_TEMPLATE(LoadTransposed, mochi::real, 4, 3, 2, true)
    ->Name("SIMD/LoadTransposed/Vec4r/Mochi/Latency");

// Load Vec8r[3] transposed
BENCHMARK_TEMPLATE(LoadTransposed, mochi::real, 8, 3, 0, false)
    ->Name("SIMD/LoadTransposed/Vec8r/Baseline/Throughput");
BENCHMARK_TEMPLATE(LoadTransposed, mochi::real, 8, 3, 0, true)
    ->Name("SIMD/LoadTransposed/Vec8r/Baseline/Latency");
BENCHMARK_TEMPLATE(LoadTransposed, mochi::real, 8, 3, 1, false)
    ->Name("SIMD/LoadTransposed/Vec8r/Scalar/Throughput");
BENCHMARK_TEMPLATE(LoadTransposed, mochi::real, 8, 3, 1, true)
    ->Name("SIMD/LoadTransposed/Vec8r/Scalar/Latency");
BENCHMARK_TEMPLATE(LoadTransposed, mochi::real, 8, 3, 2, false)
    ->Name("SIMD/LoadTransposed/Vec8r/Mochi/Throughput");
BENCHMARK_TEMPLATE(LoadTransposed, mochi::real, 8, 3, 2, true)
    ->Name("SIMD/LoadTransposed/Vec8r/Mochi/Latency");
} // namespace

/****************************************************************************************
  StoreTransposed
*/

namespace {
template <class T, int N, int M, int kVersion, bool kTestLatency>
static void StoreTransposed(benchmark::State& state) {
  static_assert(M == 3, "Only supported interleaving stride for now");

  // Allocate a 64-byte aligned buffer. This is the size of a cache line on all of our current
  // platforms. Alignment ensures that the same number of cache lines are involved, in case that
  // affects micro-op operformance.
  constexpr int kAlignment = 64;
  auto memSize = M * N * sizeof(T) * 2 + kAlignment;
  auto* mem = mochi::GetDefaultAllocator();
  auto* buf = static_cast<T*>(mem->allocate(memSize, kAlignment));
  memset(buf, 0, memSize);
  MOCHI_DEFER(mem->deallocate(buf, memSize, kAlignment));

  using V = mochi::Simd<T, N>;
  V a{}, b{}, c{};

  for (auto _ : state) {
    if constexpr (kTestLatency) {
      MOCHI_CPU_LATENCY_TEST_PRE();
    }

    if constexpr (kVersion == 0) {
      // Empty
    } else if constexpr (kVersion == 1) {
      // Scalar code to be optimized by compiler
      if constexpr (N == 4) {
        // clang-format off
        buf[0]  = a[0]; buf[1]  = b[0]; buf[2]  = c[0];
        buf[3]  = a[1]; buf[4]  = b[1]; buf[5]  = c[1];
        buf[6]  = a[2]; buf[7]  = b[2]; buf[8]  = c[2];
        buf[9]  = a[3]; buf[10] = b[3]; buf[11] = c[3];
        // clang-format on
      } else {
        // clang-format off
        buf[0]  = a[0]; buf[1]  = b[0]; buf[2]  = c[0];
        buf[3]  = a[1]; buf[4]  = b[1]; buf[5]  = c[1];
        buf[6]  = a[2]; buf[7]  = b[2]; buf[8]  = c[2];
        buf[9]  = a[3]; buf[10] = b[3]; buf[11] = c[3];
        buf[12] = a[4]; buf[13] = b[4]; buf[14] = c[4];
        buf[15] = a[5]; buf[16] = b[5]; buf[17] = c[5];
        buf[18] = a[6]; buf[19] = b[6]; buf[20] = c[6];
        buf[21] = a[7]; buf[22] = b[7]; buf[23] = c[7];
        // clang-format on
      }
    } else if constexpr (kVersion == 2) {
      // Mochi's version
      mochi::StoreTransposed(buf, a, b, c);
    }

    if constexpr (kTestLatency) {
      MOCHI_CPU_LATENCY_TEST_POST();
    }

    MOCHI_NO_DISCARD_IN_LOOP(*buf);
  }

  benchmark::DoNotOptimize(*buf);
}

// Store Vec4r[3] transposed
BENCHMARK_TEMPLATE(StoreTransposed, mochi::real, 4, 3, 0, false)
    ->Name("SIMD/StoreTransposed/Vec4r/Baseline/Throughput");
BENCHMARK_TEMPLATE(StoreTransposed, mochi::real, 4, 3, 0, true)
    ->Name("SIMD/StoreTransposed/Vec4r/Baseline/Latency");
BENCHMARK_TEMPLATE(StoreTransposed, mochi::real, 4, 3, 1, false)
    ->Name("SIMD/StoreTransposed/Vec4r/Scalar/Throughput");
BENCHMARK_TEMPLATE(StoreTransposed, mochi::real, 4, 3, 1, true)
    ->Name("SIMD/StoreTransposed/Vec4r/Scalar/Latency");
BENCHMARK_TEMPLATE(StoreTransposed, mochi::real, 4, 3, 2, false)
    ->Name("SIMD/StoreTransposed/Vec4r/Mochi/Throughput");
BENCHMARK_TEMPLATE(StoreTransposed, mochi::real, 4, 3, 2, true)
    ->Name("SIMD/StoreTransposed/Vec4r/Mochi/Latency");

// Store Vec8r[3] transposed
BENCHMARK_TEMPLATE(StoreTransposed, mochi::real, 8, 3, 0, false)
    ->Name("SIMD/StoreTransposed/Vec8r/Baseline/Throughput");
BENCHMARK_TEMPLATE(StoreTransposed, mochi::real, 8, 3, 0, true)
    ->Name("SIMD/StoreTransposed/Vec8r/Baseline/Latency");
BENCHMARK_TEMPLATE(StoreTransposed, mochi::real, 8, 3, 1, false)
    ->Name("SIMD/StoreTransposed/Vec8r/Scalar/Throughput");
BENCHMARK_TEMPLATE(StoreTransposed, mochi::real, 8, 3, 1, true)
    ->Name("SIMD/StoreTransposed/Vec8r/Scalar/Latency");
BENCHMARK_TEMPLATE(StoreTransposed, mochi::real, 8, 3, 2, false)
    ->Name("SIMD/StoreTransposed/Vec8r/Mochi/Throughput");
BENCHMARK_TEMPLATE(StoreTransposed, mochi::real, 8, 3, 2, true)
    ->Name("SIMD/StoreTransposed/Vec8r/Mochi/Latency");
} // namespace

/****************************************************************************************
  Store
*/
namespace {
template <class V>
static void Store(benchmark::State& state) {
  alignas(alignof(V)) typename V::Scalar buf[V::kSize + 1] = {};
  auto* dst = buf + 1; // not aligned
  V v = {};
  for (auto x : state) {
    mochi::Store(dst, v);
    MOCHI_NO_DISCARD_IN_LOOP(*dst);
  }
  state.counters["Values/second"] =
      benchmark::Counter(state.iterations() * V::kSize, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(*dst);
}
MOCHI_SIMD_BENCHMARK_VEC2D(Store, Vec2d);
MOCHI_SIMD_BENCHMARK_VEC4D(Store, Vec4d);
MOCHI_SIMD_BENCHMARK_VEC4F(Store, Vec4f);
MOCHI_SIMD_BENCHMARK_VEC4I(Store, Vec4i);
MOCHI_SIMD_BENCHMARK_VEC8F(Store, Vec8f);
MOCHI_SIMD_BENCHMARK_VEC8I(Store, Vec8i);
} // namespace

/****************************************************************************************
  Store<N>
*/
namespace {
template <int N, class V>
static void Store(benchmark::State& state) {
  alignas(alignof(V)) typename V::Scalar buf[N + 1] = {};
  auto* dst = buf + 1; // not aligned
  V v = {};
  for (auto x : state) {
    mochi::Store<N>(dst, v);
    MOCHI_NO_DISCARD_IN_LOOP(*dst);
  }
  state.counters["Values/second"] =
      benchmark::Counter(state.iterations() * N, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(*dst);
}
MOCHI_SIMD_BENCHMARK_VEC2D(Store, 1, Vec2d);
MOCHI_SIMD_BENCHMARK_VEC4D(Store, 3, Vec4d);
MOCHI_SIMD_BENCHMARK_VEC4F(Store, 3, Vec4f);
MOCHI_SIMD_BENCHMARK_VEC4I(Store, 3, Vec4i);
MOCHI_SIMD_BENCHMARK_VEC8F(Store, 7, Vec8f);
MOCHI_SIMD_BENCHMARK_VEC8I(Store, 7, Vec8i);
} // namespace

/****************************************************************************************
  Store(ptr, v, n)
*/
namespace {
template <class V>
static void VStore_n(benchmark::State& state) {
  alignas(alignof(V)) typename V::Scalar buf[V::kSize + 1] = {};
  auto* dst = buf + 1; // not aligned
  V v = {};
  int n = DoNotOptimizeRuntimeVar(V::kSize - 1);
  for (auto x : state) {
    mochi::Store(dst, v, n);
    MOCHI_NO_DISCARD_IN_LOOP(*dst);
  }
  state.counters["Values/second"] =
      benchmark::Counter(state.iterations() * n, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(*dst);
}
MOCHI_SIMD_BENCHMARK_VEC2D(VStore_n, Vec2d);
MOCHI_SIMD_BENCHMARK_VEC4D(VStore_n, Vec4d);
MOCHI_SIMD_BENCHMARK_VEC4F(VStore_n, Vec4f);
MOCHI_SIMD_BENCHMARK_VEC4I(VStore_n, Vec4i);
MOCHI_SIMD_BENCHMARK_VEC8F(VStore_n, Vec8f);
MOCHI_SIMD_BENCHMARK_VEC8I(VStore_n, Vec8i);
} // namespace

/****************************************************************************************
  VAdd
*/
template <class V>
static void VAdd(benchmark::State& state) {
  V a = {}, b = {}, c = {};
  for (auto x : state) {
    c = a + b;
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] =
      benchmark::Counter(state.iterations() * V::kSize, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}
MOCHI_SIMD_BENCHMARK_VEC2D(VAdd, Vec2d);
MOCHI_SIMD_BENCHMARK_VEC4D(VAdd, Vec4d);
MOCHI_SIMD_BENCHMARK_VEC4F(VAdd, Vec4f);
MOCHI_SIMD_BENCHMARK_VEC4I(VAdd, Vec4i);
MOCHI_SIMD_BENCHMARK_VEC8F(VAdd, Vec8f);
MOCHI_SIMD_BENCHMARK_VEC8I(VAdd, Vec8i);

/****************************************************************************************
  VMul
*/
template <class V>
static void VMul(benchmark::State& state) {
  V a = {}, b = {}, c = {};
  for (auto x : state) {
    c = a * b;
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] =
      benchmark::Counter(state.iterations() * V::kSize, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}
MOCHI_SIMD_BENCHMARK_VEC2D(VMul, Vec2d);
MOCHI_SIMD_BENCHMARK_VEC4D(VMul, Vec4d);
MOCHI_SIMD_BENCHMARK_VEC4F(VMul, Vec4f);
MOCHI_SIMD_BENCHMARK_VEC4I(VMul, Vec4i);
MOCHI_SIMD_BENCHMARK_VEC8F(VMul, Vec8f);
MOCHI_SIMD_BENCHMARK_VEC8I(VMul, Vec8i);

/****************************************************************************************
  MulAdd
*/
template <class V>
static void MulAdd(benchmark::State& state) {
  V a = {}, b = {}, c = {};
  for (auto x : state) {
    c = mochi::MulAdd(a, b, c);
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] =
      benchmark::Counter(state.iterations() * 2 * V::kSize, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}
MOCHI_SIMD_BENCHMARK_VEC2D(MulAdd, Vec2d);
MOCHI_SIMD_BENCHMARK_VEC4D(MulAdd, Vec4d);
MOCHI_SIMD_BENCHMARK_VEC4F(MulAdd, Vec4f);
MOCHI_SIMD_BENCHMARK_VEC8F(MulAdd, Vec8f);

// Benchmarks for chained fused multiply-add (FMA) instructions. Useful to assess the latency of FMA
// instructions on a given architecture.
template <class V, int kStride>
static void ChainedVMulAdd(benchmark::State& state) {
  V a = {}, b = {}, c1 = {};
  [[maybe_unused]] V c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {}, c8 = {}, c9 = {},
                     c10 = {}, c11 = {}, c12 = {};
  for (auto x : state) {
    if constexpr (kStride == 1) {
      c1 = mochi::MulAdd(a, b, c1);
      c1 = mochi::MulAdd(a, b, c1);
      c1 = mochi::MulAdd(a, b, c1);
      c1 = mochi::MulAdd(a, b, c1);
      c1 = mochi::MulAdd(a, b, c1);
      c1 = mochi::MulAdd(a, b, c1);
      c1 = mochi::MulAdd(a, b, c1);
      c1 = mochi::MulAdd(a, b, c1);
      c1 = mochi::MulAdd(a, b, c1);
      c1 = mochi::MulAdd(a, b, c1);
      c1 = mochi::MulAdd(a, b, c1);
      c1 = mochi::MulAdd(a, b, c1);
      MOCHI_NO_DISCARD_IN_LOOP(c1);
    } else if constexpr (kStride == 2) {
      c1 = mochi::MulAdd(a, b, c1);
      c2 = mochi::MulAdd(a, b, c2);
      c1 = mochi::MulAdd(a, b, c1);
      c2 = mochi::MulAdd(a, b, c2);
      c1 = mochi::MulAdd(a, b, c1);
      c2 = mochi::MulAdd(a, b, c2);
      c1 = mochi::MulAdd(a, b, c1);
      c2 = mochi::MulAdd(a, b, c2);
      c1 = mochi::MulAdd(a, b, c1);
      c2 = mochi::MulAdd(a, b, c2);
      c1 = mochi::MulAdd(a, b, c1);
      c2 = mochi::MulAdd(a, b, c2);
      MOCHI_NO_DISCARD_IN_LOOP(c1);
      MOCHI_NO_DISCARD_IN_LOOP(c2);
    } else if constexpr (kStride == 3) {
      c1 = mochi::MulAdd(a, b, c1);
      c2 = mochi::MulAdd(a, b, c2);
      c3 = mochi::MulAdd(a, b, c3);
      c1 = mochi::MulAdd(a, b, c1);
      c2 = mochi::MulAdd(a, b, c2);
      c3 = mochi::MulAdd(a, b, c3);
      c1 = mochi::MulAdd(a, b, c1);
      c2 = mochi::MulAdd(a, b, c2);
      c3 = mochi::MulAdd(a, b, c3);
      c1 = mochi::MulAdd(a, b, c1);
      c2 = mochi::MulAdd(a, b, c2);
      c3 = mochi::MulAdd(a, b, c3);
      MOCHI_NO_DISCARD_IN_LOOP(c1);
      MOCHI_NO_DISCARD_IN_LOOP(c2);
      MOCHI_NO_DISCARD_IN_LOOP(c3);
    } else if constexpr (kStride == 4) {
      c1 = mochi::MulAdd(a, b, c1);
      c2 = mochi::MulAdd(a, b, c2);
      c3 = mochi::MulAdd(a, b, c3);
      c4 = mochi::MulAdd(a, b, c4);
      c1 = mochi::MulAdd(a, b, c1);
      c2 = mochi::MulAdd(a, b, c2);
      c3 = mochi::MulAdd(a, b, c3);
      c4 = mochi::MulAdd(a, b, c4);
      c1 = mochi::MulAdd(a, b, c1);
      c2 = mochi::MulAdd(a, b, c2);
      c3 = mochi::MulAdd(a, b, c3);
      c4 = mochi::MulAdd(a, b, c4);
      MOCHI_NO_DISCARD_IN_LOOP(c1);
      MOCHI_NO_DISCARD_IN_LOOP(c2);
      MOCHI_NO_DISCARD_IN_LOOP(c3);
      MOCHI_NO_DISCARD_IN_LOOP(c4);
    } else if constexpr (kStride == 6) {
      c1 = mochi::MulAdd(a, b, c1);
      c2 = mochi::MulAdd(a, b, c2);
      c3 = mochi::MulAdd(a, b, c3);
      c4 = mochi::MulAdd(a, b, c4);
      c5 = mochi::MulAdd(a, b, c5);
      c6 = mochi::MulAdd(a, b, c6);
      c1 = mochi::MulAdd(a, b, c1);
      c2 = mochi::MulAdd(a, b, c2);
      c3 = mochi::MulAdd(a, b, c3);
      c4 = mochi::MulAdd(a, b, c4);
      c5 = mochi::MulAdd(a, b, c5);
      c6 = mochi::MulAdd(a, b, c6);
      MOCHI_NO_DISCARD_IN_LOOP(c1);
      MOCHI_NO_DISCARD_IN_LOOP(c2);
      MOCHI_NO_DISCARD_IN_LOOP(c3);
      MOCHI_NO_DISCARD_IN_LOOP(c4);
      MOCHI_NO_DISCARD_IN_LOOP(c5);
      MOCHI_NO_DISCARD_IN_LOOP(c6);
    } else if constexpr (kStride == 12) {
      c1 = mochi::MulAdd(a, b, c1);
      c2 = mochi::MulAdd(a, b, c2);
      c3 = mochi::MulAdd(a, b, c3);
      c4 = mochi::MulAdd(a, b, c4);
      c5 = mochi::MulAdd(a, b, c5);
      c6 = mochi::MulAdd(a, b, c6);
      c7 = mochi::MulAdd(a, b, c7);
      c8 = mochi::MulAdd(a, b, c8);
      c9 = mochi::MulAdd(a, b, c9);
      c10 = mochi::MulAdd(a, b, c10);
      c11 = mochi::MulAdd(a, b, c11);
      c12 = mochi::MulAdd(a, b, c12);
      MOCHI_NO_DISCARD_IN_LOOP(c1);
      MOCHI_NO_DISCARD_IN_LOOP(c2);
      MOCHI_NO_DISCARD_IN_LOOP(c3);
      MOCHI_NO_DISCARD_IN_LOOP(c4);
      MOCHI_NO_DISCARD_IN_LOOP(c5);
      MOCHI_NO_DISCARD_IN_LOOP(c6);
      MOCHI_NO_DISCARD_IN_LOOP(c7);
      MOCHI_NO_DISCARD_IN_LOOP(c8);
      MOCHI_NO_DISCARD_IN_LOOP(c9);
      MOCHI_NO_DISCARD_IN_LOOP(c10);
      MOCHI_NO_DISCARD_IN_LOOP(c11);
      MOCHI_NO_DISCARD_IN_LOOP(c12);
    } else {
      static_assert(kStride == 12, "Case not supported");
    }
  }
  state.counters["FLOPs"] =
      benchmark::Counter(state.iterations() * 12 * 2 * V::kSize, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c1);
  benchmark::DoNotOptimize(c2);
  benchmark::DoNotOptimize(c3);
  benchmark::DoNotOptimize(c4);
  benchmark::DoNotOptimize(c5);
  benchmark::DoNotOptimize(c6);
  benchmark::DoNotOptimize(c7);
  benchmark::DoNotOptimize(c8);
  benchmark::DoNotOptimize(c9);
  benchmark::DoNotOptimize(c10);
  benchmark::DoNotOptimize(c11);
  benchmark::DoNotOptimize(c12);
}
MOCHI_SIMD_BENCHMARK_VEC2D(ChainedVMulAdd, Vec2d, 1);
MOCHI_SIMD_BENCHMARK_VEC2D(ChainedVMulAdd, Vec2d, 2);
MOCHI_SIMD_BENCHMARK_VEC2D(ChainedVMulAdd, Vec2d, 3);
MOCHI_SIMD_BENCHMARK_VEC2D(ChainedVMulAdd, Vec2d, 4);
MOCHI_SIMD_BENCHMARK_VEC2D(ChainedVMulAdd, Vec2d, 6);
MOCHI_SIMD_BENCHMARK_VEC2D(ChainedVMulAdd, Vec2d, 12);
MOCHI_SIMD_BENCHMARK_VEC4D(ChainedVMulAdd, Vec4d, 1);
MOCHI_SIMD_BENCHMARK_VEC4D(ChainedVMulAdd, Vec4d, 2);
MOCHI_SIMD_BENCHMARK_VEC4D(ChainedVMulAdd, Vec4d, 3);
MOCHI_SIMD_BENCHMARK_VEC4D(ChainedVMulAdd, Vec4d, 4);
MOCHI_SIMD_BENCHMARK_VEC4D(ChainedVMulAdd, Vec4d, 6);
MOCHI_SIMD_BENCHMARK_VEC4D(ChainedVMulAdd, Vec4d, 12);
MOCHI_SIMD_BENCHMARK_VEC4F(ChainedVMulAdd, Vec4f, 1);
MOCHI_SIMD_BENCHMARK_VEC4F(ChainedVMulAdd, Vec4f, 2);
MOCHI_SIMD_BENCHMARK_VEC4F(ChainedVMulAdd, Vec4f, 3);
MOCHI_SIMD_BENCHMARK_VEC4F(ChainedVMulAdd, Vec4f, 4);
MOCHI_SIMD_BENCHMARK_VEC4F(ChainedVMulAdd, Vec4f, 6);
MOCHI_SIMD_BENCHMARK_VEC4F(ChainedVMulAdd, Vec4f, 12);
MOCHI_SIMD_BENCHMARK_VEC8F(ChainedVMulAdd, Vec8f, 1);
MOCHI_SIMD_BENCHMARK_VEC8F(ChainedVMulAdd, Vec8f, 2);
MOCHI_SIMD_BENCHMARK_VEC8F(ChainedVMulAdd, Vec8f, 3);
MOCHI_SIMD_BENCHMARK_VEC8F(ChainedVMulAdd, Vec8f, 4);
MOCHI_SIMD_BENCHMARK_VEC8F(ChainedVMulAdd, Vec8f, 6);
MOCHI_SIMD_BENCHMARK_VEC8F(ChainedVMulAdd, Vec8f, 12);

/****************************************************************************************
  VDot
*/
template <class V>
static void VDot(benchmark::State& state) {
  V a = {}, b = {}, d = {};
  for (auto x : state) {
    d = mochi::VDot(a, b);
    MOCHI_NO_DISCARD_IN_LOOP(d);
  }
  state.counters["FLOPs"] =
      benchmark::Counter(state.iterations() * (2 * V::kSize - 1), benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(d);
}
MOCHI_SIMD_BENCHMARK_VEC2D(VDot, Vec2d);
MOCHI_SIMD_BENCHMARK_VEC4D(VDot, Vec4d);
MOCHI_SIMD_BENCHMARK_VEC4F(VDot, Vec4f);
MOCHI_SIMD_BENCHMARK_VEC8F(VDot, Vec8f);

/****************************************************************************************
  HSum
*/
template <class V>
static void HSum(benchmark::State& state) {
  V a = {};
  typename V::Scalar s = {};
  for (auto x : state) {
    s = mochi::HSum(a);
    MOCHI_NO_DISCARD_IN_LOOP(s);
  }
  state.counters["FLOPs"] =
      benchmark::Counter(state.iterations() * (V::kSize - 1), benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(s);
}
MOCHI_SIMD_BENCHMARK_VEC2D(HSum, Vec2d);
MOCHI_SIMD_BENCHMARK_VEC4D(HSum, Vec4d);
MOCHI_SIMD_BENCHMARK_VEC4F(HSum, Vec4f);
MOCHI_SIMD_BENCHMARK_VEC8F(HSum, Vec8f);

/****************************************************************************************
  Exp
*/

template <typename Scalar>
static void ExpForLoop(benchmark::State& state) {
  int n = int(state.range(0));
  mochi::ColumnVector<Scalar> v(n), expV(n);
  v.SetRandom(123, -10.0, 10.0);
  for (auto x : state) {
    for (int i = 0; i < n; ++i) {
      expV(i, 0) = std::exp(v(i, 0));
    }
    MOCHI_NO_DISCARD_IN_LOOP(v);
    MOCHI_NO_DISCARD_IN_LOOP(expV);
  }
  benchmark::DoNotOptimize(v);
  benchmark::DoNotOptimize(expV);
}

template <typename Scalar, class VType>
static void ExpSIMD(benchmark::State& state) {
  int n = int(state.range(0));
  mochi::ColumnVector<Scalar> v(n), expV(n);
  v.SetRandom(123, -10.0, 10.0);
  for (auto x : state) {
    int k = 0;
    for (; k + VType::kSize <= n; k += VType::kSize) {
      auto w = VType::Load(v.Data() + k);
      VType::Store(expV.Data() + k, Exp(w));
    }
    for (; k < n; ++k) {
      expV(k, 0) = std::exp(v(k, 0));
    }
    MOCHI_NO_DISCARD_IN_LOOP(v);
    MOCHI_NO_DISCARD_IN_LOOP(expV);
  }
  benchmark::DoNotOptimize(v);
  benchmark::DoNotOptimize(expV);
}

BENCHMARK_TEMPLATE(ExpForLoop, float)->Args({512});
BENCHMARK_TEMPLATE(ExpSIMD, float, mochi::Simd<float, 4>)->Args({512});
BENCHMARK_TEMPLATE(ExpSIMD, float, mochi::Simd<float, 8>)->Args({512});
BENCHMARK_TEMPLATE(ExpForLoop, double)->Args({512});
BENCHMARK_TEMPLATE(ExpSIMD, double, mochi::Simd<double, 2>)->Args({512});
BENCHMARK_TEMPLATE(ExpSIMD, double, mochi::Simd<double, 4>)->Args({512});

/****************************************************************************************
  Sin
*/

template <class V>
static void Sin(benchmark::State& state) {
  V input = static_cast<typename V::Scalar>(2); // Arbitrary non-special-case value
  V output = {};
  std::function<void()> fn = [&]() { output = mochi::Sin(input); };
  for (auto _ : state) {
    CallNoInline(fn);
  }
  benchmark::DoNotOptimize(output);
}

BENCHMARK_TEMPLATE(Sin, mochi::Vec2d);
BENCHMARK_TEMPLATE(Sin, mochi::Vec4f);
BENCHMARK_TEMPLATE(Sin, mochi::Vec4d);
BENCHMARK_TEMPLATE(Sin, mochi::Vec8f);

/****************************************************************************************
  Cos
*/

template <class V>
static void Cos(benchmark::State& state) {
  V input = static_cast<typename V::Scalar>(2); // Arbitrary non-special-case value
  V output = {};
  std::function<void()> fn = [&]() { output = mochi::Cos(input); };
  for (auto _ : state) {
    CallNoInline(fn);
  }
  benchmark::DoNotOptimize(output);
}

BENCHMARK_TEMPLATE(Cos, mochi::Vec2d);
BENCHMARK_TEMPLATE(Cos, mochi::Vec4f);
BENCHMARK_TEMPLATE(Cos, mochi::Vec4d);
BENCHMARK_TEMPLATE(Cos, mochi::Vec8f);

/****************************************************************************************
  SinCos
*/

template <class V>
static void SinCos(benchmark::State& state) {
  V input = static_cast<typename V::Scalar>(2); // Arbitrary non-special-case value
  std::pair<V, V> output = {};
  std::function<void()> fn = [&]() { output = mochi::SinCos(input); };
  for (auto _ : state) {
    CallNoInline(fn);
  }
  benchmark::DoNotOptimize(output);
}

BENCHMARK_TEMPLATE(SinCos, mochi::Vec2d);
BENCHMARK_TEMPLATE(SinCos, mochi::Vec4f);
BENCHMARK_TEMPLATE(SinCos, mochi::Vec4d);
BENCHMARK_TEMPLATE(SinCos, mochi::Vec8f);

/****************************************************************************************
  StoreSelected
*/

template <class T, int N>
static int RunStoreSelectedBenchmark(mochi::Span<T const> src, mochi::Span<T> dst) {
  MOCHI_ASSERT_VERBOSE(src.size() == dst.size());
  MOCHI_ASSERT_VERBOSE((src.size() % N) == 0);

  // Write just the positive values to dst. Return the count.
  T const* srcData = src.data();
  T* dstData = dst.data();
  int const numValues = isize(src);
  int count = 0; // output count
  for (int i = 0; i < numValues; i += N) {
    if constexpr (N == 1) {
      // Trivial scalar implementation (with branch)
      if (srcData[i] >= T(0)) {
        dstData[count++] = srcData[i];
      }
    } else {
      // SIMD implementation (branchless)
      auto values = mochi::Load<mochi::Simd<T, N>>(srcData + i);
      count += mochi::StoreSelected(dstData + count, values >= T{0}, values);
    }
  }
  return count;
}

template <class T, int N>
static void StoreSelectedBenchmark(benchmark::State& state, int numValues) {
  using namespace mochi;
  MOCHI_ASSERT_VERBOSE((numValues % N) == 0, "This benchmark expects an even multiple");

  // Initialize an array of random source values between -1 and 1.
  // The benchmark will use StoreSelected to copy the positive values (~50%) to an output buffer.
  DynamicArray<T> srcValues(numValues);
  auto rng = RandomGenerator(123);
  SetRandom(rng, T(-1), T(1), MakeSpan(srcValues));
  DynamicArray<T> dstValues(numValues);
  auto srcSpan = MakeConstSpan(srcValues);
  auto dstSpan = MakeSpan(dstValues);
  int lastCount = 0;

  for (auto _ : state) {
    CallNoInline([&]() { lastCount = RunStoreSelectedBenchmark<T, N>(srcSpan, dstSpan); });
  }

  // Report the rate at which source values were processed
  state.counters["values/second"] =
      benchmark::Counter(state.iterations() * numValues, benchmark::Counter::kIsRate);

  // Sanity check:
  size_t expectedCount = 0;
  for (size_t i = 0; i < numValues; ++i) {
    if (srcValues[i] >= 0) {
      MOCHI_ASSERT(dstValues[expectedCount++] == srcValues[i]);
    }
  }
  MOCHI_ASSERT(lastCount == expectedCount);
}

// Wrapper so we can use BENCHMARK_TEMPLATE without quite so much bloat
template <class T, int N, size_t kNumPoints>
static void StoreSelected(benchmark::State& state) {
  StoreSelectedBenchmark<T, N>(state, kNumPoints);
}

BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 1, 80);
BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 1, 800);
BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 1, 8000);
BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 1, 80000);
BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 1, 800000);

#if MOCHI_USE_DOUBLE_PRECISION
BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 2, 80);
BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 2, 800);
BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 2, 8000);
BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 2, 80000);
BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 2, 800000);
#endif

BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 4, 80);
BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 4, 800);
BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 4, 8000);
BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 4, 80000);
BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 4, 800000);

BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 8, 80);
BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 8, 800);
BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 8, 8000);
BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 8, 80000);
BENCHMARK_TEMPLATE(StoreSelected, mochi::real, 8, 800000);

} // namespace mochi_benchmark
