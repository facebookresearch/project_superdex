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

#include <mochi_core/utils/hash_utils.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

using namespace mochi;

// The increment by which the splitmix64 generator advances its state.
static constexpr uint64_t kIncrement = 0x9e3779b97f4a7c15ull;

TEST(HashUtils, MixBits) {
  // The first outputs of the splitmix64 generator seeded with 1234567, which advances its state by
  // kIncrement and mixes its bits.
  constexpr uint64_t kSeed = 1234567;
  static_assert(MixBits(kSeed + kIncrement) == 6457827717110365317ull);
  static_assert(MixBits(kSeed + 2 * kIncrement) == 3203168211198807973ull);
  static_assert(MixBits(kSeed + 3 * kIncrement) == 9817491932198370423ull);
}

TEST(HashUtils, HashUnordered) {
  // The sum of the values' mixed bits, after adding kIncrement so that 0 counts too, which ignores
  // their order and counts repeated values.
  constexpr auto mixed = [](uint64_t value) { return MixBits(value + kIncrement); };
  static_assert(HashUnordered(std::array<int, 0>{}) == 0);
  static_assert(HashUnordered(std::array{2, 0, 1, 2}) == mixed(0) + mixed(1) + 2 * mixed(2));

  // Enum values hash as their underlying values.
  enum class Fruit : uint8_t { Apple = 1, Kumquat = 2 };
  static_assert(
      HashUnordered(std::array{Fruit::Kumquat, Fruit::Apple}) == HashUnordered(std::array{1, 2}));
}
