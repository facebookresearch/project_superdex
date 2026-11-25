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

#include <gtest/gtest.h>

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/utils/generator.h>

#include <vector>

using namespace mochi;

namespace {

Generator<int> EvenOdd(auto&& range) {
  for (auto x : range) {
    if (x % 2 == 0) {
      co_yield x;
    }
  }
  for (auto x : range) {
    if (x % 2 != 0) {
      co_yield x;
    }
  }
}

Generator<MatrixView<float>> MatrixGenerator(float* space) {
  for (int i = 0; i < 4; ++i) {
    co_yield MatrixView<float>(space + 3 * i, 2 + i, 1);
  }
}

Generator<int> IntGenerator(int n) {
  for (int i = 0; i < n; ++i) {
    co_yield i;
  }
}

void TestGenerator() {
  std::vector<int> v;
  int N = 100;
  v.reserve(N);
  for (int i = 0; i < N; ++i) {
    v.push_back(i);
  }
  std::vector<int> result;
  result.reserve(N);
  for (auto i : EvenOdd(v)) {
    result.push_back(i);
  }
  std::vector<int> expectedResult;
  expectedResult.reserve(N);
  for (int i = 0; i < 100; ++i) {
    expectedResult.push_back(i < 50 ? 2 * i : 2 * (i - 50) + 1);
  }
  EXPECT_EQ(result, expectedResult);

  result.clear();
  {
    auto g = EvenOdd(v);
    while (g.Next()) {
      result.push_back(g.Value());
    }
  }
  EXPECT_EQ(result, expectedResult);

  // Test that the generator does not try to use operator= of its return type.
  std::vector<float> space(100);
  for (int i = 0; auto block : MatrixGenerator(space.data())) {
    EXPECT_EQ(&block(0, 0), space.data() + 3 * i);
    EXPECT_EQ(block.Rows(), 2 + i);
    EXPECT_EQ(block.Cols(), 1);
    ++i;
  }

  int count = 0;
  // Test that a non yielding generator does not do any iteration.
  for ([[maybe_unused]] auto _ : IntGenerator(0)) {
    ++count;
  }
  EXPECT_EQ(count, 0);
}

} // namespace

TEST(Generator, Iteration) {
  TestGenerator();
}
