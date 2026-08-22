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

#include <mochi_core/utils/array.h>

#include <mochi_core/test/mochi_test_helpers.h>

#include <gtest/gtest.h>

using namespace mochi;

template <typename T, size_t N>
static void TestArray() {
  Array<T, N> array{};
  static_assert(sizeof(array) == sizeof(T) * N);

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(T{}, array[i]);
  }

  EXPECT_EQ(array.data(), array.begin());
  EXPECT_EQ(array.begin() + N, array.end());
  static_assert(Array<T, N>::size() == N);
  EXPECT_EQ(N, array.size());

  for (size_t i = 0; i < N; ++i) {
    array[i] = static_cast<T>(i);
    EXPECT_EQ(static_cast<T>(i), array[i]);
    EXPECT_EQ(&array[i], array.data() + i);
  }

  EXPECT_EQ(array[0], array.front());
  EXPECT_EQ(array[N - 1], array.back());
}

template <typename T>
static void TestArray() {
  TestArray<T, 1>();
  TestArray<T, 2>();
  TestArray<T, 3>();
  TestArray<T, 4>();
  TestArray<T, 5>();
  TestArray<T, 100>();
}

TEST(Array, Array) {
  TestArray<bool>();
  TestArray<char>();
  TestArray<real>();
  TestArray<int>();
  TestArray<size_t>();
  TestArray<uint32_t>();
  TestArray<int64_t>();
}
