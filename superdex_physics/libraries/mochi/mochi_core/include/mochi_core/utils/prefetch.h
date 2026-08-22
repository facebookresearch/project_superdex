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

#include <mochi_core/mochi_platform.h>

#if MOCHI_COMPILER_MSVC
#include <intrin.h>
#endif

namespace mochi {

/** @brief Temporal locality hint for software prefetch operations. */
enum class PrefetchLocality {
  NonTemporal = 0, ///< Data accessed once — don't pollute caches.
  Low = 1, ///< Low temporal locality.
  Moderate = 2, ///< Moderate temporal locality.
  High = 3, ///< High temporal locality — keep in all cache levels.
  Count ///< Number of prefetch locality enum values.
};

/**
 * @brief Software prefetch hints for hiding latency when the access pattern is known.
 *
 * @warning Do not use without empirical evidence. Modern CPUs prefetch automatically for sequential
 * and strided access patterns.
 *
 * @note Example: Double-buffered prefetch: precompute next iteration's pointers and prefetch their
 * cache lines while processing the current iteration.
 * @code
 *   real* current = GetRow(0);
 *   for (int i = 0; i < n - 1; ++i) {
 *     real* next = GetRow(i + 1);
 *     PrefetchWrite(next);
 *     ScatterAddToRow(current);
 *     current = next;
 *   }
 *   if (n > 0) {
 *     ScatterAddToRow(current);
 *   }
 * @endcode
 *
 * @tparam Locality Temporal locality hint. Use @ref PrefetchLocality::High (default) when the data
 * will be accessed multiple times soon. Use lower values for streaming access patterns.
 */
template <PrefetchLocality Locality = PrefetchLocality::High>
MOCHI_FORCE_INLINE void PrefetchRead(void const* ptr) {
  static_assert(Locality >= PrefetchLocality::NonTemporal && Locality < PrefetchLocality::Count);
#if MOCHI_COMPILER_CUDA_GPU
  // Manual PTX prefetching (e.g., prefetch.global.L1) is intentionally disabled:
  // 1. In a generic function, pointer provenance (global vs. shared vs. local) is unknown. Applying
  // a global prefetch instruction to a shared memory address causes hardware faults or silent
  // failures.
  // 2. GPUs hide memory latency via massive thread concurrency (warp scheduling), unlike CPUs which
  // rely heavily on cache prefetching.
  // 3. Forcing manual prefetch across thousands of active threads frequently causes L1 cache
  // thrashing, degrading overall kernel performance.
  (void)ptr;
#elif MOCHI_COMPILER_MSVC
  // Map PrefetchLocality to MSVC _MM_HINT: High→T0, Moderate→T1, Low→T2, NonTemporal→NTA.
  constexpr auto kHint = (Locality == PrefetchLocality::High) ? _MM_HINT_T0
      : (Locality == PrefetchLocality::Moderate)              ? _MM_HINT_T1
      : (Locality == PrefetchLocality::Low)                   ? _MM_HINT_T2
                                                              : _MM_HINT_NTA;
  _mm_prefetch(reinterpret_cast<char const*>(ptr), kHint);
#elif MOCHI_COMPILER_GCC || MOCHI_COMPILER_CLANG
  __builtin_prefetch(ptr, 0, static_cast<int>(Locality));
#else
  (void)ptr;
#endif
}

template <PrefetchLocality Locality = PrefetchLocality::High>
MOCHI_FORCE_INLINE void PrefetchWrite(void const* ptr) {
  static_assert(Locality >= PrefetchLocality::NonTemporal && Locality < PrefetchLocality::Count);
#if MOCHI_COMPILER_CUDA_GPU
  (void)ptr;
#elif MOCHI_COMPILER_MSVC
  // MSVC: _m_prefetchw requires PREFETCHW support (not available on pre-Broadwell Intel CPUs). We
  // fall back to read-prefetch (_mm_prefetch), which still brings the cache line in and provides
  // most of the latency-hiding benefit.
  constexpr auto kHint = (Locality == PrefetchLocality::High) ? _MM_HINT_T0
      : (Locality == PrefetchLocality::Moderate)              ? _MM_HINT_T1
      : (Locality == PrefetchLocality::Low)                   ? _MM_HINT_T2
                                                              : _MM_HINT_NTA;
  _mm_prefetch(reinterpret_cast<char const*>(ptr), kHint);
#elif MOCHI_COMPILER_GCC || MOCHI_COMPILER_CLANG
  __builtin_prefetch(ptr, 1, static_cast<int>(Locality));
#else
  (void)ptr;
#endif
}

} // namespace mochi
