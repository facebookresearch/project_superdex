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

#include <mochi_core/utils/half.h>

#include <mochi_core/test/mochi_test_helpers.h>

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <type_traits>

using namespace mochi;

static_assert(sizeof(Half) == 2);
static_assert(alignof(Half) == 2);
static_assert(IsHalf<Half>);
static_assert(IsHalf<Half const>);
static_assert(IsHalf<Half&>);
static_assert(IsHalf<Half const&>);
static_assert(!IsHalf<float>);
static_assert(!IsHalf<int>);
static_assert(!IsHalf<uint16_t>);

static_assert(std::is_convertible_v<Half, float>);
static_assert(std::is_convertible_v<Half, double>);
static_assert(!std::is_convertible_v<float, Half>);
static_assert(!std::is_convertible_v<double, Half>);
static_assert(!std::is_convertible_v<int, Half>);
static_assert(std::is_trivially_copyable_v<Half>);

static uint16_t GetBits(Half h) {
  return ReinterpretCast<uint16_t>(h);
}

static uint32_t GetBits(float f) {
  return std::bit_cast<uint32_t>(f);
}

static float FloatFromBits(uint32_t bits) {
  return std::bit_cast<float>(bits);
}

static Half HalfFromBits(uint16_t bits) {
  return ReinterpretCast<Half>(bits);
}

TEST(Half, DefaultConstruct) {
  // There is no guarantee that a Half variable will be initialized on the stack (similar to float).
  // However, Half{} is guaranteed to be zero-initialized.
  EXPECT_EQ(HalfFromBits(0x0000), Half{});
}

TEST(Half, CopyConstruct) {
  Half const a = HalfFromBits(0x3C00); // 1.0
  Half const b = a;
  EXPECT_EQ(GetBits(a), GetBits(b));

  Half const zero = HalfFromBits(0x0000); // 0.0
  Half const zeroCopy = zero;
  EXPECT_EQ(GetBits(zero), GetBits(zeroCopy));
}

TEST(Half, Equality) {
  EXPECT_EQ(Half(1.0f), Half(1.0f));
  EXPECT_EQ(Half(-2.0f), Half(-2.0f));
  EXPECT_NE(Half(1.0f), Half(1.1f));
  EXPECT_NE(Half(-2.0f), Half(-2.1f));

  // +0 == -0 (IEEE 754 semantics)
  EXPECT_EQ(Half(0.0f), Half(-0.0f));

  // NaN != NaN (IEEE 754 semantics)
  auto const nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_NE(Half(nan), Half(nan));
}

TEST(Half, ImplicitConversions) {
  // float <-- Half
  {
    float f = HalfFromBits(0x3C00);
    EXPECT_EQ(1.0f, f);
    f = HalfFromBits(0xBC00);
    EXPECT_EQ(-1.0f, f);
    f = HalfFromBits(0x0000);
    EXPECT_EQ(0.0f, f);
  }

  // double <-- Half
  {
    double d = HalfFromBits(0x4000);
    EXPECT_EQ(2.0, d);
    d = HalfFromBits(0xBC00);
    EXPECT_EQ(-1.0, d);
    d = HalfFromBits(0x0000);
    EXPECT_EQ(0.0, d);
  }
}

TEST(Half, ExplicitConversions) {
  // Half <-- float
  {
    EXPECT_EQ(0x3C00, GetBits(Half(1.0f)));
    EXPECT_EQ(0xC000, GetBits(Half(-2.0f)));
    EXPECT_EQ(0x0000, GetBits(Half(0.0f)));
    EXPECT_EQ(0x7C00, GetBits(Half(std::numeric_limits<float>::infinity())));
    EXPECT_EQ(0xFC00, GetBits(Half(-std::numeric_limits<float>::infinity())));
  }

  // Half <-- double
  {
    EXPECT_EQ(0x3C00, GetBits(Half(1.0)));
    EXPECT_EQ(0xC000, GetBits(Half(-2.0)));
    EXPECT_EQ(0x0000, GetBits(Half(0.0)));
    EXPECT_EQ(0x7C00, GetBits(Half(std::numeric_limits<double>::infinity())));
    EXPECT_EQ(0xFC00, GetBits(Half(-std::numeric_limits<double>::infinity())));
  }

  // Half <-- int
  {
    EXPECT_EQ(HalfFromBits(0x4000), Half((2)));
    EXPECT_EQ(HalfFromBits(0xBC00), Half((-1)));
  }
}

TEST(Half, StaticCast) {
  // float <-- Half
  EXPECT_EQ(1.0f, StaticCast<float>(HalfFromBits(0x3C00)));
  EXPECT_EQ(-2.0f, StaticCast<float>(HalfFromBits(0xC000)));
  EXPECT_EQ(0.0f, StaticCast<float>(HalfFromBits(0x0000)));

  // Half <-- float
  EXPECT_EQ(HalfFromBits(0x3C00), StaticCast<Half>(1.0f));
  EXPECT_EQ(HalfFromBits(0xC000), StaticCast<Half>(-2.0f));
  EXPECT_EQ(HalfFromBits(0x0000), StaticCast<Half>(0.0f));

  // double <-- Half
  EXPECT_EQ(2.0, StaticCast<double>(HalfFromBits(0x4000)));
  EXPECT_EQ(-1.0, StaticCast<double>(HalfFromBits(0xBC00)));
  EXPECT_EQ(0.0, StaticCast<double>(HalfFromBits(0x0000)));

  // Half <-- double
  EXPECT_EQ(HalfFromBits(0x4000), StaticCast<Half>(2.0));
  EXPECT_EQ(HalfFromBits(0xBC00), StaticCast<Half>(-1.0));
  EXPECT_EQ(HalfFromBits(0x0000), StaticCast<Half>(0.0));

  // int <-- Half
  EXPECT_EQ(2, StaticCast<int>(HalfFromBits(0x4000)));
  EXPECT_EQ(-1, StaticCast<int>(HalfFromBits(0xBC00)));

  // Half <-- int
  EXPECT_EQ(HalfFromBits(0x4000), StaticCast<Half>(2));
  EXPECT_EQ(HalfFromBits(0xBC00), StaticCast<Half>(-1));
}

TEST(Half, ReinterpretCast) {
  uint16_t const testValues[] = {0x0000, 0x3C00, 0x7C00, 0x7E00, 0xFFFF, 0x0001};
  for (uint16_t val : testValues) {
    // Round trip via uint16_t
    auto h = ReinterpretCast<Half>(val);
    EXPECT_EQ(val, ReinterpretCast<uint16_t>(h));

    // Round trip via another 2-byte type
    using TwoBytes = std::array<uint8_t, 2>;
    TwoBytes val2;
    memcpy(&val2, &val, sizeof(val));
    h = ReinterpretCast<Half>(val2);
    auto val3 = ReinterpretCast<TwoBytes>(h);
    EXPECT_EQ(0, memcmp(&val2, &val3, sizeof(val2)));
  }
}

TEST(Half, HalfBitsToFloat) {
  // Normal values
  EXPECT_EQ(1.0f, HalfBitsToFloat(0x3C00));
  EXPECT_EQ(2.0f, HalfBitsToFloat(0x4000));
  EXPECT_EQ(0.5f, HalfBitsToFloat(0x3800));
  EXPECT_EQ(-1.0f, HalfBitsToFloat(0xBC00));
  EXPECT_EQ(65504.0f, HalfBitsToFloat(0x7BFF));

  // Denormal values
  float minDenormal = HalfBitsToFloat(0x0001);
  EXPECT_GT(minDenormal, 0.0f);
  EXPECT_NEAR(minDenormal, 5.960464477539063e-8f, 1e-15f);
  float minNormal = HalfBitsToFloat(0x0400);
  EXPECT_NEAR(minNormal, 6.103515625e-5f, 1e-10f);

  // Negative denormal
  float negMinDenormal = HalfBitsToFloat(0x8001);
  EXPECT_LT(negMinDenormal, 0.0f);
  EXPECT_NEAR(negMinDenormal, -5.960464477539063e-8f, 1e-15f);

  // Special values
  EXPECT_EQ(0.0f, HalfBitsToFloat(0x0000));
  EXPECT_EQ(GetBits(HalfBitsToFloat(0x8000)), GetBits(-0.0f));
  EXPECT_EQ(std::numeric_limits<float>::infinity(), HalfBitsToFloat(0x7C00));
  EXPECT_EQ(-std::numeric_limits<float>::infinity(), HalfBitsToFloat(0xFC00));
  EXPECT_TRUE(std::isnan(HalfBitsToFloat(0x7E00)));
  EXPECT_TRUE(std::isnan(HalfBitsToFloat(0x7C01)));
}

TEST(Half, FloatToHalfBits) {
  // Normal values
  {
    EXPECT_EQ(0x3C00, FloatToHalfBits(1.0f));
    EXPECT_EQ(0x4000, FloatToHalfBits(2.0f));
    EXPECT_EQ(0x3800, FloatToHalfBits(0.5f));
    EXPECT_EQ(0xBC00, FloatToHalfBits(-1.0f));
    EXPECT_EQ(0x7BFF, FloatToHalfBits(65504.0f));
  }

  // Special values
  {
    // Zeros
    EXPECT_EQ(0x0000, FloatToHalfBits(0.0f));
    EXPECT_EQ(0x8000, FloatToHalfBits(-0.0f));
    // Infinities
    EXPECT_EQ(0x7C00, FloatToHalfBits(std::numeric_limits<float>::infinity()));
    EXPECT_EQ(0xFC00, FloatToHalfBits(-std::numeric_limits<float>::infinity()));
    // NaN
    uint16_t nanBits = FloatToHalfBits(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(nanBits & 0x7C00, 0x7C00);
    EXPECT_NE(nanBits & 0x03FF, 0);
    // NaN with low payload bits (exercises payload fixup path)
    uint16_t lowPayloadNan = FloatToHalfBits(FloatFromBits(0x7F800001));
    EXPECT_EQ(lowPayloadNan & 0x7C00, 0x7C00); // Must be NaN (exponent all 1s)
    EXPECT_NE(lowPayloadNan & 0x03FF, 0); // Must have nonzero payload
    // Overflow
    EXPECT_EQ(0x7C00, FloatToHalfBits(100000.0f));
    EXPECT_EQ(0xFC00, FloatToHalfBits(-100000.0f));
    // Underflow
    EXPECT_EQ(0x0000, FloatToHalfBits(1e-20f));
    EXPECT_EQ(0x8000, FloatToHalfBits(-1e-20f));
  }

  // Rounding tie to even
  {
    // Tie at even mantissa (0) -> stay
    float const evenTie = 1.0f + 0.5f * (1.0f / 1024.0f);
    EXPECT_EQ(0x3C00, FloatToHalfBits(evenTie));

    // Just above tie -> round up
    float const aboveTie = 1.0f + 0.5f * (1.0f / 1024.0f) + 1e-7f;
    EXPECT_EQ(0x3C01, FloatToHalfBits(aboveTie));

    // Tie at odd mantissa (1) -> round up to even
    float const oddTie = 1.0f + 1.5f * (1.0f / 1024.0f);
    EXPECT_EQ(0x3C02, FloatToHalfBits(oddTie));
  }

  // Rounding edge cases
  {
    // Mantissa overflow causes exponent increment: just under 2.0 rounds to 2.0
    EXPECT_EQ(0x4000, FloatToHalfBits(FloatFromBits(0x3FFFFFFF)));

    // Rounding at max normal causes overflow to infinity
    EXPECT_EQ(0x7C00, FloatToHalfBits(65520.0f));
    EXPECT_EQ(0xFC00, FloatToHalfBits(-65520.0f));

    // Denormal rounding: exact midpoint between half denormals 0x0001 and 0x0002
    // 1.5 * 2^-24 is exactly halfway; m=1 is odd, so round-to-even rounds UP to 0x0002.
    float const denormalTie = FloatFromBits(0x33C00000); // Exactly 1.5 * 2^-24
    EXPECT_EQ(0x0002, FloatToHalfBits(denormalTie));
  }
}

TEST(Half, Constants) {
  // Verify bit patterns
  EXPECT_EQ(0x0400, GetBits(kHalfMin));
  EXPECT_EQ(0x7BFF, GetBits(kHalfMax));

  // Verify float values
  EXPECT_EQ(6.103515625e-05f, static_cast<float>(kHalfMin));
  EXPECT_EQ(65504.0f, static_cast<float>(kHalfMax));

  // Verify round-trip through Half(float)
  EXPECT_EQ(kHalfMin, Half(6.103515625e-05f));
  EXPECT_EQ(kHalfMax, Half(65504.0f));
}

TEST_IF(MOCHI_OPTIMIZED, Half, Exhaustive_HalfToFloat) {
  // Verify software HalfBitsToFloat matches hardware (static_cast) for all 65536 half bit patterns.
  uint32_t mismatches = 0;
  for (uint32_t i = 0; i <= 0xFFFF; ++i) {
    auto const bits = uint16_t(i);
    float const fromSoftware = HalfBitsToFloat(bits);
    auto const fromHardware = static_cast<float>(HalfFromBits(bits));

    bool const isNaN = ((bits >> 10) & 0x1F) == 0x1F && (bits & 0x3FF) != 0;

    if (isNaN) {
      EXPECT_TRUE(std::isnan(fromSoftware)) << "Software should be NaN for 0x" << std::hex << bits;
      EXPECT_TRUE(std::isnan(fromHardware)) << "Hardware should be NaN for 0x" << std::hex << bits;
    } else {
      if (GetBits(fromSoftware) != GetBits(fromHardware)) {
        if (++mismatches <= 10) {
          ADD_FAILURE() << "Half->Float mismatch for 0x" << std::hex << bits << ": software=0x"
                        << GetBits(fromSoftware) << " hardware=0x" << GetBits(fromHardware);
        }
      }
    }
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " total Half->Float mismatches";
}

TEST_IF(MOCHI_OPTIMIZED, Half, Exhaustive_FloatToHalf) {
  // Verify software FloatToHalfBits matches hardware (static_cast) for all 2^32 float bit patterns.
  uint32_t mismatches = 0;
  for (uint64_t i = 0; i <= 0xFFFFFFFF; ++i) {
    auto const bits = uint32_t(i);
    float const f = FloatFromBits(bits);
    uint16_t const fromSoftware = FloatToHalfBits(f);
    uint16_t const fromHardware = GetBits(static_cast<Half>(f));

    if (std::isnan(f)) {
      bool const swIsNaN = ((fromSoftware & 0x7C00) == 0x7C00) && ((fromSoftware & 0x03FF) != 0);
      bool const hwIsNaN = ((fromHardware & 0x7C00) == 0x7C00) && ((fromHardware & 0x03FF) != 0);
      if (!swIsNaN || !hwIsNaN) {
        if (++mismatches <= 10) {
          ADD_FAILURE() << "Float->Half NaN mismatch for float bits 0x" << std::hex << bits
                        << ": software=0x" << fromSoftware << " hardware=0x" << fromHardware;
        }
      }
    } else {
      if (fromSoftware != fromHardware) {
        if (++mismatches <= 10) {
          ADD_FAILURE() << "Float->Half mismatch for float bits 0x" << std::hex << bits << " (" << f
                        << "): software=0x" << fromSoftware << " hardware=0x" << fromHardware;
        }
      }
    }
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " total Float->Half mismatches";
}
