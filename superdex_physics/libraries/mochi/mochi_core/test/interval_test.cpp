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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/interval.h>

#include <vector>

using namespace mochi;

TEST(BasicTypes, Interval) {
  // Test a valid range.
  constexpr Interval<int> kValidRange{-5, 5};
  static_assert(kValidRange.Valid());
  static_assert(kValidRange.Size() == 10);
  static_assert(!kValidRange.Within(-10));
  static_assert(kValidRange.Within(-5));
  static_assert(kValidRange.Within(0));
  static_assert(!kValidRange.Within(5));
  static_assert(!kValidRange.Within(10));

  // Test an invalid range.
  constexpr Interval<int> kInvalidRange{5, -5};
  static_assert(!kInvalidRange.Valid());

  // Test union and intersection operations.
  constexpr Interval<int> kRange1{0, 4};
  constexpr Interval<int> kRange2{2, 6};
  constexpr Interval<int> kUnionRange = kRange1.Union(kRange2);
  constexpr Interval<int> kIntersectionRange = kRange1.Intersect(kRange2);
  static_assert(kUnionRange.Min() == 0);
  static_assert(kUnionRange.Max() + 1 == 6);
  static_assert(kIntersectionRange.Min() == 2);
  static_assert(kIntersectionRange.Max() + 1 == 4);
  auto vec = kRange2.to<std::vector<int>>();
  EXPECT_TRUE((vec == std::vector<int>{2, 3, 4, 5}));
}
