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

#include "simd_test_half.h"

#include <type_traits>

#if MOCHI_HAS_SIMD_HALF

using namespace mochi;
using namespace mochi::simd_half_test;

MOCHI_SIMD_TEST_SCALAR_CONVERSIONS_TO_HALF(Vec16h);

TEST(Vec16h, TypeProperties) {
  static_assert(Vec16h::kIsSupported, "Vec16h should be supported");
  static_assert(Vec16h::kSize == 16);
  static_assert(sizeof(Vec16h) == 32);
  static_assert(std::is_same_v<Vec16h::Scalar, Half>);
  static_assert(!Vec16h::kIsEmulated);

#if MOCHI_ARCH_X64_AVX2
  static_assert(!Vec16h::kIsComposite);
#elif MOCHI_ARCH_ARM_NEON
  static_assert(Vec16h::kIsComposite);
#endif
}

MOCHI_DEFINE_SIMD_HALF_COMMON_TESTS(Vec16h)

TEST(Vec16h, GetHalf) {
  auto testValues = GetTestValues(Vec16h::kSize);
  auto v = Load<Vec16h>(testValues.data());

  auto lo = Vec16h::GetHalf<0>(v);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(testValues[i], lo[i]);
  }

  auto hi = Vec16h::GetHalf<1>(v);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(testValues[i + 8], hi[i]);
  }
}

TEST(Vec16h, TwoHalfConstructor) {
  auto testValues = GetTestValues(Vec16h::kSize);
  auto v = Load<Vec16h>(testValues.data());
  auto lo = Vec16h::GetHalf<0>(v);
  auto hi = Vec16h::GetHalf<1>(v);
  Vec16h combined{lo, hi};
  EXPECT_EQ(v, combined);
}

TEST(Vec16h, ScalarConstructor) {
  auto values = GetTestValues(Vec16h::kSize);

  // Construct from 2 scalars
  {
    Vec16h v{values[0], values[1]};
    for (int i = 0; i < Vec16h::kSize; ++i) {
      EXPECT_EQ(i < 2 ? values[i] : Half{}, v[i]);
    }
  }

  // Construct from 3 scalars
  {
    Vec16h v{values[0], values[1], values[2]}; // 3 args
    for (int i = 0; i < Vec16h::kSize; ++i) {
      EXPECT_EQ(i < 3 ? values[i] : Half{}, v[i]);
    }
  }

  // Construct from 16 scalars
  {
    Vec16h v{
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
        values[9],
        values[10],
        values[11],
        values[12],
        values[13],
        values[14],
        values[15]};
    for (int i = 0; i < Vec16h::kSize; ++i) {
      EXPECT_EQ(values[i], v[i]);
    }
  }
}

#endif // MOCHI_HAS_SIMD_HALF
