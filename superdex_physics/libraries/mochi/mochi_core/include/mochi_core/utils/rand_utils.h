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

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

#include <chrono>
#include <random>
#include <type_traits>

namespace mochi {

// Use an explicit random engine so the selection is not implementation-defined.
using mochi_default_random_engine = std::mt19937; // mersenne twister

// Generate a random seed based on the system clock. DO NOT USE THIS IN UNIT TESTS.
[[nodiscard]] inline unsigned int GetRandomSeed() {
  return static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
}

// Initialize our default random engine with a specified seed.
[[nodiscard]] inline mochi_default_random_engine RandomGenerator(unsigned int seed) {
  return mochi_default_random_engine((unsigned int)seed);
}

// Returns a 32-bit XorShift random number generator function, which returns a new pseudo-random
// uint32_t value on each call. The returned function maintains internal state between calls. This
// is a lightweight alternative to the default random engine (std::mt19937) with lower quality
// randomness but better performance.
// Reference: Marsaglia, G. (2003). "Xorshift RNGs". Journal of Statistical Software, 8(14), 1-6.
[[nodiscard]] inline auto XorShift32Generator(uint32_t seed) {
  uint32_t state = Max(seed, uint32_t(1)); // seed = 0 would generate a sequence of all 0's
  return [state]() mutable -> uint32_t {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  };
}

// If T is an integral type, then return a random value in the inclusive range [min, max].
// If T is a floating-point type, then return a random value in the half-open range [min, max).
template <typename T, typename Engine>
[[nodiscard]] T RandomUniformValue(Engine& generator, T min, T max) {
  if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<T> uniformDist(min, max);
    return uniformDist(generator);
  } else {
    // std::uniform_int_distribution is only specified for short/int/long/long long and their
    // unsigned counterparts (N4950 [rand.req.genl]/1.5). MSVC's STL enforces this, so promote
    // narrow integer types (e.g. (un)signed char, bool) to a supported type and cast back.
    using DistT = std::conditional_t<
        (sizeof(T) < sizeof(short)),
        std::conditional_t<std::is_signed_v<T>, short, unsigned short>,
        T>;
    std::uniform_int_distribution<DistT> uniformDist(min, max);
    return static_cast<T>(uniformDist(generator));
  }
}

// Set a random value. The range is inclusive for integral types.
// See RandomUniformValue.
template <typename T, typename Engine>
void SetRandom(Engine& generator, T min, T max, T& out) {
  out = RandomUniformValue(generator, min, max);
}

// Fill an NdArray with random values. The range is inclusive for integral types.
// See RandomUniformValue.
template <typename T, size_t D0, size_t... DIMS, typename Engine>
void SetRandom(Engine& generator, T min, T max, NdArray<T, D0, DIMS...>& out) {
  for (size_t i = 0; i < D0; ++i) {
    SetRandom(generator, min, max, out[i]);
  }
}

// Fill a Span of NdArray with random values. The range is inclusive for integral types.
// See RandomUniformValue.
template <typename T, typename U, typename SZ, typename Engine>
void SetRandom(Engine& generator, T min, T max, Span<U, SZ> out) {
  for (auto& val : out) {
    SetRandom(generator, min, max, val);
  }
}

} // namespace mochi
