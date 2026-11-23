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

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

using namespace superdex::robotics;

namespace {

struct GoldenVector {
  uint64_t seed;
  std::string_view key;
  std::optional<uint64_t> index;
  uint64_t expected;
};

/* The cross-language contract: reference implementations must match these values
 * byte-for-byte. Regenerate ONLY with an intentional, reviewed, versioned change
 * to the derive spec, since saved episode lists depend on these values. */
constexpr std::array<GoldenVector, 10> kGoldenVectors = {{
    {.seed = 0, .key = "episode", .index = std::nullopt, .expected = 0xB8585E8A9C0DBDCFULL},
    {.seed = 0, .key = "episode", .index = 0, .expected = 0x7EFA5F8AD6B623CBULL},
    {.seed = 0, .key = "dr/scene/light", .index = std::nullopt, .expected = 0x5030FA1CE5A18882ULL},
    {.seed = 1, .key = "dr/scene/light", .index = std::nullopt, .expected = 0x514779E16CBE3052ULL},
    {.seed = 1, .key = "dr/scene/light", .index = 0, .expected = 0x213A364625D030BDULL},
    {.seed = 1, .key = "dr/scene/light", .index = 1, .expected = 0x349CF06D1B43E6DFULL},
    {.seed = 12345, .key = "dr/robot/obs_noise", .index = 7, .expected = 0x77412850768BACA3ULL},
    {.seed = 0xFFFFFFFFFFFFFFFFULL,
     .key = "dr/task/shape_pose",
     .index = 100,
     .expected = 0x55834CAD71CF82E0ULL},
    {.seed = 2026,
     .key = "dr/global/difficulty",
     .index = std::nullopt,
     .expected = 0x9CED29E36394A848ULL},
    {.seed = 2026, .key = "", .index = 0, .expected = 0xD757873921833D22ULL},
}};

} // namespace

TEST(DrDeriveTest, MatchesGoldenVectors) {
  for (GoldenVector const& v : kGoldenVectors) {
    EXPECT_EQ(Derive(v.seed, v.key, v.index), v.expected) << "seed=" << v.seed << " key=" << v.key;
  }
}

TEST(DrDeriveTest, IndexNoneDistinctFromZero) {
  EXPECT_NE(Derive(7, "dr/scene/light"), Derive(7, "dr/scene/light", 0));
}

TEST(DrDeriveTest, DistinctKeysGiveDistinctValues) {
  // Each value is a pure function of (seed, key) with no shared state, regardless
  // of derivation order. These example keys should not alias; arbitrary keys can
  // still collide. Determinism itself is covered by MatchesGoldenVectors.
  uint64_t const seed = Derive(2026, "episode", 3);
  EXPECT_NE(Derive(seed, "dr/scene/light"), Derive(seed, "dr/robot/obs_noise"));
}
