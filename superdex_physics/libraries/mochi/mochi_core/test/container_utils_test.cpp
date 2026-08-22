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
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/rand_utils.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace mochi;

TEST(ContainerUtils, Append) {
  // std::vector <-- std::vector
  {
    std::vector<int> vec;
    Append(vec, std::vector<int>{});
    EXPECT_EQ(0, isize(vec));
    Append(vec, std::vector<int>{1, 2, 3});
    EXPECT_EQ((std::vector<int>{1, 2, 3}), vec);
    Append(vec, std::vector<int>{4, 5, 6});
    EXPECT_EQ((std::vector<int>{1, 2, 3, 4, 5, 6}), vec);
  }

  // std::vector <-- int[]
  {
    int constexpr kValues1[] = {1, 2, 3};
    int constexpr kValues2[] = {4, 5, 6};
    std::vector<int> vec;
    Append(vec, kValues1);
    EXPECT_EQ((std::vector<int>{1, 2, 3}), vec);
    Append(vec, kValues2);
    EXPECT_EQ((std::vector<int>{1, 2, 3, 4, 5, 6}), vec);
  }

  // std::vector <-- Span
  {
    int constexpr kValues1[] = {1, 2, 3};
    int constexpr kValues2[] = {4, 5, 6};
    std::vector<int> vec;
    Append(vec, Span<int const>{});
    EXPECT_EQ(0, isize(vec));
    Append(vec, MakeSpan(kValues1));
    EXPECT_EQ((std::vector<int>{1, 2, 3}), vec);
    Append(vec, MakeSpan(kValues2));
    EXPECT_EQ((std::vector<int>{1, 2, 3, 4, 5, 6}), vec);
  }
}

TEST(ContainerUtils, AppendSum) {
  std::array<int, 3> values1 = {1, 2, 3};
  std::array<int, 3> values2 = {4, 5, 6};

  // std::vector<T> <-- std::vector<T>, T
  {
    std::vector<int> vec;
    AppendSum(vec, std::vector<int>{}, 0);
    EXPECT_EQ(0, isize(vec));
    AppendSum(vec, std::vector<int>{1, 2, 3}, -1);
    EXPECT_EQ((std::vector<int>{0, 1, 2}), vec);
    AppendSum(vec, std::vector<int>{4, 5, 6}, 0);
    EXPECT_EQ((std::vector<int>{0, 1, 2, 4, 5, 6}), vec);
  }

  // std::vector<T> <-- std::array<T>, T
  {
    std::vector<int> vec = {-1, 2}; // Non-empty
    AppendSum(vec, values1, -1);
    EXPECT_EQ((std::vector<int>{-1, 2, 0, 1, 2}), vec);
    AppendSum(vec, values2, 0);
    EXPECT_EQ((std::vector<int>{-1, 2, 0, 1, 2, 4, 5, 6}), vec);
  }

  // std::vector<T> <-- Span<T>, T
  {
    std::vector<int> vec;
    AppendSum(vec, Span<int const>{}, 0);
    EXPECT_EQ(0, isize(vec));
    AppendSum(vec, MakeSpan(values1), -1);
    EXPECT_EQ((std::vector<int>{0, 1, 2}), vec);
    AppendSum(vec, MakeSpan(values2), 0);
    EXPECT_EQ((std::vector<int>{0, 1, 2, 4, 5, 6}), vec);
  }

  // std::vector<T> <-- std::vector<T>, std::vector<T>
  {
    std::vector<int> vec = {-5, -1}; // Non-empty
    AppendSum(vec, std::vector<int>{1, 2, 3}, std::vector<int>{1, 2, 3});
    EXPECT_EQ((std::vector<int>{-5, -1, 2, 4, 6}), vec);
    AppendSum(vec, std::vector<int>{1, 2, 3}, std::vector<int>{4, 5, 6});
    EXPECT_EQ((std::vector<int>{-5, -1, 2, 4, 6, 5, 7, 9}), vec);
  }

  // std::vector<T> <-- std::array<T>, std::array<T>
  {
    std::vector<int> vec = {0}; // Non-empty
    AppendSum(vec, values1, values1);
    EXPECT_EQ((std::vector<int>{0, 2, 4, 6}), vec);
    AppendSum(vec, values1, values2);
    EXPECT_EQ((std::vector<int>{0, 2, 4, 6, 5, 7, 9}), vec);
  }

  // std::vector<T> <-- Span<T>, Span<T>
  {
    std::vector<int> vec;
    AppendSum(vec, MakeSpan(values1), MakeSpan(values1));
    EXPECT_EQ((std::vector<int>{2, 4, 6}), vec);
    AppendSum(vec, MakeSpan(values2), MakeSpan(values1));
    EXPECT_EQ((std::vector<int>{2, 4, 6, 5, 7, 9}), vec);
  }
}

TEST(ContainerUtils, EraseIndexUnordered) {
  std::string str = "ABCDEFG";

  // Erase front
  EraseIndexUnordered(str, 0);
  EXPECT_STREQ("GBCDEF", str.c_str());

  // Erase middle
  EraseIndexUnordered(str, 2);
  EXPECT_STREQ("GBFDE", str.c_str());

  // Erase back
  EraseIndexUnordered(str, 4);
  EXPECT_STREQ("GBFD", str.c_str());

  // Keep erasing until empty
  EraseIndexUnordered(str, 0);
  EXPECT_STREQ("DBF", str.c_str());
  EraseIndexUnordered(str, 0);
  EXPECT_STREQ("FB", str.c_str());
  EraseIndexUnordered(str, 0);
  EXPECT_STREQ("B", str.c_str());
  EraseIndexUnordered(str, 0);
  EXPECT_STREQ("", str.c_str());

  // Yes it works for std::vector too
  std::vector<int> vec = {0, 1, 2};
  EraseIndexUnordered(vec, 1);
  EXPECT_EQ((std::vector<int>{0, 2}), vec);
}

TEST(ContainerUtils, Contains) {
  // c-style array
  {
    constexpr int kValues[] = {1, 2};
    static_assert(!details::IsAssociative<decltype(kValues)>::value);
    EXPECT_TRUE(Contains(kValues, 1));
    EXPECT_TRUE(Contains(kValues, 2));
    EXPECT_FALSE(Contains(kValues, 3));
  }

  // std::vector
  {
    std::vector<int> values = {1, 2};
    static_assert(!details::IsAssociative<decltype(values)>::value);
    EXPECT_TRUE(Contains(values, 1));
    EXPECT_TRUE(Contains(values, 2));
    EXPECT_FALSE(Contains(values, 3));
  }

  // std::unordered_set
  {
    std::unordered_set<int> values = {1, 2};
    static_assert(details::IsAssociative<decltype(values)>::value);
    EXPECT_TRUE(Contains(values, 1));
    EXPECT_TRUE(Contains(values, 2));
    EXPECT_FALSE(Contains(values, 3));
  }
}

TEST(ContainerUtils, PairHash) {
  // int, int
  {
    PairHash<int, int> hash;
    EXPECT_NE(hash({1, 2}), hash({1, 1}));
    EXPECT_NE(hash({1, 2}), hash({2, 1}));
    EXPECT_EQ(hash({1, 2}), hash({1, 2}));
  }

  // int8_t, enum class
  {
    enum class Fruit { Apple, Kumquat };
    PairHash<int8_t, Fruit> hash;
    EXPECT_NE(hash({1, Fruit::Apple}), hash({1, Fruit::Kumquat}));
    EXPECT_NE(hash({1, Fruit::Kumquat}), hash({2, Fruit::Kumquat}));
    EXPECT_EQ(hash({1, Fruit::Kumquat}), hash({1, Fruit::Kumquat}));
  }

  // Prove we can use it in a std container
  {
    std::unordered_set<std::pair<int, bool>, PairHash<int, bool>> set;
    EXPECT_EQ(set.end(), set.find({123, true}));
    set.insert({123, true});
    EXPECT_NE(set.end(), set.find({123, true}));
  }
}

TEST(ContainerUtils, RandomSubset) {
  auto runTests = [](auto&& rng) {
    for (auto N0 : {0, 1, 5, 10, 20}) {
      std::vector<int> vec0(N0);
      std::iota(vec0.begin(), vec0.end(), 1);
      std::vector<int> vec = vec0;

      // Shrink to same size (no-op).
      RandomSubset(vec, size_t(N0), rng); // SizeT = size_t specialization
      EXPECT_TRUE(vec == vec0);

      // Shrink to larger size (no-op).
      RandomSubset(vec, int(N0 + 1), rng); // SizeT = int specialization
      EXPECT_TRUE(vec == vec0);

      // Shrink to smaller size.
      if (N0 > 0) {
        RandomSubset(vec, uint32_t(N0 - 1), rng); // SizeT = uint32_t specialization
        auto sorted = vec;
        std::sort(sorted.begin(), sorted.end());
        EXPECT_EQ(vec.size(), N0 - 1);
        for (int i = 0; i < isize(sorted) - 1; ++i) {
          EXPECT_TRUE(sorted[i] != sorted[i + 1]);
        }
        for (int elem : vec) {
          EXPECT_TRUE(elem >= 1 && elem <= N0);
        }
      }

      // Shrink to zero.
      RandomSubset(vec, 0, rng);
      EXPECT_TRUE(vec.empty());
    }
  };

  constexpr uint32_t kSeeds[6] = {0, 1, 123, 12345, 1234567, 123456789};
  for (auto seed : kSeeds) {
    runTests(RandomGenerator(seed)); // Default random generator
    runTests(XorShift32Generator(seed)); // 32-bit xorshift random generator
  }
}
