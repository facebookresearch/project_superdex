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

#include <mochi_core/utils/rand_utils.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

using namespace mochi;

template <typename UIntT, typename RngT>
static void TestRng(RngT&& rng) {
  // Generate a large sample of random numbers.
  constexpr size_t kSampleSize = 10000;
  std::vector<UIntT> samples;
  samples.reserve(kSampleSize);
  for (size_t i = 0; i < kSampleSize; ++i) {
    samples.push_back(static_cast<UIntT>(rng()));
  }

  // Test 1: Values span the full UIntT range.
  UIntT minValue = *std::min_element(samples.begin(), samples.end());
  UIntT maxValue = *std::max_element(samples.begin(), samples.end());
  EXPECT_LT(minValue, maxValue / 100);
  EXPECT_GT(maxValue, std::numeric_limits<UIntT>::max() / 2);

  // Test 2: Uniformity using chi-square test.
  constexpr int kNumBins = 10;
  std::vector<size_t> bins(kNumBins, 0);
  for (auto value : samples) {
    auto binIndex = static_cast<size_t>(
        (static_cast<double>(value) / std::numeric_limits<UIntT>::max()) * kNumBins);
    if (binIndex == kNumBins) {
      binIndex--; // Handle edge case for max value.
    }
    bins[binIndex]++;
  }

  // Expected count per bin for uniform distribution.
  double expectedCount = static_cast<double>(kSampleSize) / kNumBins;
  double chiSquare = 0.0;
  for (auto count : bins) {
    double diff = count - expectedCount;
    chiSquare += (diff * diff) / expectedCount;
  }

  // For 9 degrees of freedom (10 bins - 1) at 99.9% confidence, chi-square should be < 27.88.
  // This is a conservative threshold to avoid flaky tests.
  EXPECT_LT(chiSquare, 27.88);

  // Test 3: Consecutive values are different.
  size_t differentConsecutiveValues = 0;
  for (size_t i = 1; i < samples.size(); ++i) {
    if (samples[i] != samples[i - 1]) {
      differentConsecutiveValues++;
    }
  }

  // Expect at least 99.9% of consecutive values to be different.
  EXPECT_GT(differentConsecutiveValues, 0.999 * (kSampleSize - 1));
}

TEST(RandUtils, DefaultGenerator) {
  constexpr unsigned int kSeeds[6] = {0, 1, 123, 12345, 1234567, 123456789};
  for (auto seed : kSeeds) {
    // The default generator produces 32-bit unsigned integers of type uint_fast32_t, which may be
    // defined as 64-bit on some platforms.
    TestRng<uint32_t>(RandomGenerator(seed));
  }
}

TEST(RandUtils, XorShift32Generator) {
  constexpr uint32_t kSeeds[6] = {0, 1, 123, 12345, 1234567, 123456789};
  for (auto seed : kSeeds) {
    TestRng<uint32_t>(XorShift32Generator(seed));
  }
}

TEST(RandUtils, RandomUniformValue_int) {
  auto rng = RandomGenerator(123);

  // RandomUniformValue should be able to produce every value in the range, including the max value.
  std::unordered_set<int> values;
  for (int i = 0; i < 1000; ++i) {
    int val = RandomUniformValue(rng, -3, 7);
    EXPECT_LE(-3, val);
    EXPECT_GE(7, val); // Range includes the max value
    values.insert(val);
    if (isize(values) == 11) {
      break;
    }
  }
  EXPECT_EQ(11, isize(values));
}

TEST(RandUtils, RandomUniformValue_real) {
  auto rng = RandomGenerator(123);

  // RandomUniformValue should be able to produce floating point values that round to each of the
  // integer buckets.
  std::unordered_set<int> values;
  for (int i = 0; i < 1000; ++i) {
    real val = RandomUniformValue(rng, -3_r, 7_r);
    EXPECT_LE(-3_r, val);
    EXPECT_GT(7_r, val); // Range does not include the max value
    values.insert(static_cast<int>(Round(val)));
    if (isize(values) == 11) {
      break;
    }
  }
  EXPECT_EQ(11, isize(values));
}

TEST(RandUtils, SetRandom) {
  auto rng = RandomGenerator(123);

  auto expectRandomValues = [](Span<real const> values) {
    for (int i = 0; i < isize(values); ++i) {
      EXPECT_NE(0_r, values[i]); // Very unlikely
      EXPECT_LE(-1_r, values[i]);
      EXPECT_GT(1_r, values[i]);
    }
    std::vector<real> sortedValues(values.begin(), values.end());
    std::sort(sortedValues.begin(), sortedValues.end());
    EXPECT_TRUE(
        std::unique(sortedValues.begin(), sortedValues.end()) ==
        sortedValues.end()); // Expect no duplicates
  };

  // 1D NdArray
  Real3 v;
  SetRandom(rng, -1_r, 1_r, v);
  expectRandomValues(v);

  // 2D NdArray
  Matrix3x3r m;
  SetRandom(rng, -1_r, 1_r, m);
  expectRandomValues(Flatten(m));

  // Span
  std::vector<real> values(100);
  SetRandom(rng, -1_r, 1_r, MakeSpan(values));
  expectRandomValues(values);

  // Span of NdArray
  std::vector<Real3> points(100);
  SetRandom(rng, -1_r, 1_r, MakeSpan(points));
  expectRandomValues(Span{&points[0][0], 300});
}
