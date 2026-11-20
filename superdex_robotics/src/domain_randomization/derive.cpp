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

#include <superdex_robotics/domain_randomization/derive.h>

#include <cstdint>

namespace superdex::robotics {
namespace {

/* FNV-1a (64-bit) parameters -- a stable, portable string hash. */
constexpr uint64_t kFnv64OffsetBasis = 0xCBF29CE484222325ULL;
constexpr uint64_t kFnv64Prime = 0x100000001B3ULL;

/* SplitMix64 finalizer multipliers (Steele, Lea & Flood, 2014). */
constexpr uint64_t kSplitMix64MulA = 0xBF58476D1CE4E5B9ULL;
constexpr uint64_t kSplitMix64MulB = 0x94D049BB133111EBULL;

/* FNV-1a 64-bit hash of the key's bytes. uint64_t arithmetic wraps mod 2^64,
 * matching the Python reference's explicit `& MASK64`. */
uint64_t StableKeyHash(std::string_view key) {
  uint64_t h = kFnv64OffsetBasis;
  for (char const c : key) {
    h = (h ^ static_cast<uint8_t>(c)) * kFnv64Prime;
  }
  return h;
}

/* SplitMix64 finalizer: a bijective 64-bit avalanche mix. */
uint64_t SplitMix64Mix(uint64_t z) {
  z = (z ^ (z >> 30)) * kSplitMix64MulA;
  z = (z ^ (z >> 27)) * kSplitMix64MulB;
  return z ^ (z >> 31);
}

} // namespace

/* Parity-locked to the Python reference (see derive.h). Do NOT
 * change the constants or the mixing: it would invalidate every committed
 * DREpisode list. `index` is unsigned (indices are non-negative counts) and
 * matches the Python reference for any index >= 0. */
uint64_t Derive(uint64_t seed, std::string_view key, std::optional<uint64_t> index) {
  uint64_t acc = SplitMix64Mix(seed ^ StableKeyHash(key));
  if (index.has_value()) {
    acc = SplitMix64Mix(acc ^ *index);
  }
  return acc;
}

} // namespace superdex::robotics
