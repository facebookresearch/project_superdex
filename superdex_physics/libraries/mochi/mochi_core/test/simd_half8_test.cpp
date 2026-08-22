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

MOCHI_SIMD_TEST_SCALAR_CONVERSIONS_TO_HALF(Vec8h);

TEST(Vec8h, TypeProperties) {
  static_assert(Vec8h::kIsSupported, "Vec8h should be supported");
  static_assert(Vec8h::kSize == 8);
  static_assert(sizeof(Vec8h) == 16);
  static_assert(std::is_same_v<Vec8h::Scalar, Half>);
  static_assert(!Vec8h::kIsComposite);
  static_assert(!Vec8h::kIsEmulated);
}

MOCHI_DEFINE_SIMD_HALF_COMMON_TESTS(Vec8h)

TEST(Vec8h, ScalarConstruction) {
  auto values = GetTestValues(Vec8h::kSize);

  // Construct from 8 scalars
  {
    Vec8h v{values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]};
    for (int i = 0; i < Vec8h::kSize; ++i) {
      EXPECT_EQ(values[i], v[i]);
    }
  }

  // Construct from 2 scalars
  {
    Vec8h v{values[0], values[1]};
    for (int i = 0; i < Vec8h::kSize; ++i) {
      EXPECT_EQ(i < 2 ? values[i] : Half{}, v[i]);
    }
  }

  // Construct from 3 scalars
  {
    Vec8h v{values[0], values[1], values[2]};
    for (int i = 0; i < Vec8h::kSize; ++i) {
      EXPECT_EQ(i < 3 ? values[i] : Half{}, v[i]);
    }
  }
}

#endif // MOCHI_HAS_SIMD_HALF
