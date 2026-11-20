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

#include <superdex_physics.h>

#include <cstdint>
#include <optional>
#include <string_view>

namespace superdex::robotics {

/* @brief Deterministic 64-bit sub-seed from a seed and a stable string key.
 *
 * Maps (seed, key[, index]) to a deterministic 64-bit sub-seed over arbitrary
 * string keys (FNV-1a key hash XOR seed, SplitMix64-finalized). Stateless and
 * order-independent: a key's sub-seed depends only on (seed, key, index), never
 * on what else was derived or in what order. It yields seeds, not values -- the
 * caller turns each sub-seed into random numbers with its own RNG.
 *
 * Optional index distinguishes acquisitions; absence and zero are distinct.
 * Parity-locked to the Python reference; golden vectors define the
 * cross-language contract.
 *
 * @param[in] seed Master seed.
 * @param[in] key Stable string key (its UTF-8 bytes are hashed).
 * @param[in] index Optional index; std::nullopt and 0 are intentionally distinct.
 * @return Deterministic 64-bit sub-seed in [0, 2^64).
 */
[[nodiscard]] MOCHI_API uint64_t
Derive(uint64_t seed, std::string_view key, std::optional<uint64_t> index = std::nullopt);

} // namespace superdex::robotics
