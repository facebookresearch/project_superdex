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

#include <benchmark/benchmark.h>

#include <functional>
#include <string>
#include <thread>

namespace mochi_benchmark {

// MOCHI_NO_DISCARD_IN_LOOP goes inside the loop being profiled. It convinces the compiler that
// the given variable or expression should not be optimized away even if it otherwise has no side
// effects. Without this macro, the MSVC compiler would still do the work but clang/gcc would
// optimize it away.
#if MOCHI_COMPILER_MSVC
#define MOCHI_NO_DISCARD_IN_LOOP(x) ((void)(x))
#else
#define MOCHI_NO_DISCARD_IN_LOOP(x)             \
  {                                             \
    auto const& no_discard_x_ref = x;           \
    benchmark::DoNotOptimize(no_discard_x_ref); \
  }
#endif

// Macro to skip a benchmark.
#define MOCHI_SKIP_BENCHMARK(state) \
  for (auto _ : (state)) {          \
  }

// Number of worker threads for mochi::TaskScheduler to use.
// Set to -1 to use all available threads.
// Set to 0 for single threaded mode (slow but more repeatable performance)
static int const kNumWorkerThreads = std::thread::hardware_concurrency();

// Get a file path relative to the "assets" directory
std::string GetAssetPath(std::string const& relative);

// Returns the same int, but the compiler won't know that as long as the function does not inline
int DoNotOptimizeRuntimeVar(int var);

// If you want to ensure that a function gets called and that no part of it gets optimized out, then
// you can wrap it in a std::function and use CallNoInline. The downside is that the call overhead
// will add a small amount to the measured time (~1-2 ns). Example:
//
//  std::function<void()> fn = [&]() { /* your code here */ }; // outside the loop
//  for (auto _ : state) {
//    CallNoInline(fn);
//  }
//
void CallNoInline(std::function<void()> const& fn);

} // namespace mochi_benchmark
