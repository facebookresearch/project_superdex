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

#include <mochi_core/utils/subset_map.h>

#include <gtest/gtest.h>

#include <iterator>
#include <vector>

using namespace mochi;

TEST(SubsetMap, TestExtract) {
  int N = 6;
  SubsetMap map({1, 2, 5}, N);

  std::vector<real> values = {0.0_r, 2.0_r, 4.0_r, 5.0_r, 6.0_r, 1.0_r};

  auto result = map.Extract<real>(values);

  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], 2.0_r);
  EXPECT_EQ(result[1], 4.0_r);
  EXPECT_EQ(result[2], 1.0_r);
}

TEST(SubsetMap, GetIndices) {
  int N = 6;
  SubsetMap map({1, 2, 5}, N);

  EXPECT_EQ(map.GetFullIndexFromSubsetIndex(0), 1);
  EXPECT_EQ(map.GetFullIndexFromSubsetIndex(1), 2);
  EXPECT_EQ(map.GetFullIndexFromSubsetIndex(2), 5);

  std::vector<int> subsetIndices = {0, 1, 2};
  std::vector<int> fullIndices;

  map.GetFullIndicesFromSubsetIndices(
      subsetIndices.begin(), subsetIndices.end(), std::back_inserter(fullIndices));

  EXPECT_EQ(fullIndices.size(), 3);
  EXPECT_EQ(fullIndices[0], 1);
  EXPECT_EQ(fullIndices[1], 2);
  EXPECT_EQ(fullIndices[2], 5);

  EXPECT_FALSE(map.GetSubsetIndexFromFullIndex(0));
  EXPECT_EQ(*map.GetSubsetIndexFromFullIndex(1), 0);
  EXPECT_EQ(*map.GetSubsetIndexFromFullIndex(2), 1);
  EXPECT_FALSE(map.GetSubsetIndexFromFullIndex(3));
  EXPECT_FALSE(map.GetSubsetIndexFromFullIndex(4));
  EXPECT_EQ(*map.GetSubsetIndexFromFullIndex(5), 2);

  subsetIndices.clear();
  map.GetSubsetIndicesFromFullIndices(
      fullIndices.begin(), fullIndices.end(), std::back_inserter(subsetIndices), ErrorAssert{});

  EXPECT_EQ(subsetIndices.size(), 3);
  EXPECT_EQ(subsetIndices[0], 0);
  EXPECT_EQ(subsetIndices[1], 1);
  EXPECT_EQ(subsetIndices[2], 2);

  fullIndices = {0, 1, 2};

  Error error;
  subsetIndices.clear();
  map.GetSubsetIndicesFromFullIndices(
      fullIndices.begin(), fullIndices.end(), std::back_inserter(subsetIndices), error);

  EXPECT_FALSE(error.IsOK());
}

TEST(SubsetMap, Intersection) {
  int N = 6;
  SubsetMap subset1({0, 2, 3, 4}, N);
  SubsetMap subset2({1, 3, 4}, N);

  auto result = Intersection(subset1, subset2);

  EXPECT_EQ(result.GetSubsetSize(), 2);
  EXPECT_EQ(result[0], 3);
  EXPECT_EQ(result[1], 4);
}

TEST(SubsetMap, Union) {
  int N = 6;
  SubsetMap subset1({0, 2, 3, 4}, N);
  SubsetMap subset2({1, 3, 4}, N);

  auto result = Union(subset1, subset2);

  EXPECT_EQ(result.GetSubsetSize(), 5);
  EXPECT_EQ(result[0], 0);
  EXPECT_EQ(result[1], 1);
  EXPECT_EQ(result[2], 2);
  EXPECT_EQ(result[3], 3);
  EXPECT_EQ(result[4], 4);
}

TEST(SubsetMap, Complement) {
  int N = 6;
  SubsetMap subset1({0, 4, 5}, N);
  SubsetMap subset2({1, 2, 3}, N);
  SubsetMap subset3({1, 4}, N);

  SubsetMap complement1 = subset1.Complement();
  SubsetMap complement2 = subset2.Complement();
  SubsetMap complement3 = subset3.Complement();

  EXPECT_EQ(complement1.GetSubsetSize(), 3);
  EXPECT_EQ(complement1[0], 1);
  EXPECT_EQ(complement1[1], 2);
  EXPECT_EQ(complement1[2], 3);

  EXPECT_EQ(complement2.GetSubsetSize(), 3);
  EXPECT_EQ(complement2[0], 0);
  EXPECT_EQ(complement2[1], 4);
  EXPECT_EQ(complement2[2], 5);

  EXPECT_EQ(complement3.GetSubsetSize(), 4);
  EXPECT_EQ(complement3[0], 0);
  EXPECT_EQ(complement3[1], 2);
  EXPECT_EQ(complement3[2], 3);
  EXPECT_EQ(complement3[3], 5);
}
