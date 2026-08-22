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

#include <mochi_core/contact/contact_partition.h>
#include <mochi_core/test/mochi_test_helpers.h>

#include <gtest/gtest.h>

#include <array>

using namespace mochi;

namespace {
int constexpr kNumSamples = 10;
int constexpr kNumPartitions = 4;
}; // namespace

template <typename T>
static std::array<T, kNumSamples> InitializeSamples(
    std::array<T, kNumPartitions> const& partitions,
    std::array<int, kNumSamples> const& samples) {
  std::array<T, kNumSamples> result{};
  for (int i = 0; i < kNumSamples; i++) {
    result[i] = partitions[samples[i]];
  }
  return result;
}

static std::array<IndexGroups, kNumSamples> InitializeIndexGroups() {
  // Define some index groups. IndexGroup::src is irrelevant, partitions are defined based on
  // IndexGroup::dst and IndexGroup::count. To create different index groups, all we need to do is
  // vary e.g. IndexGroup::dst.
  IndexGroup indA{0, 0, 4};
  IndexGroup indB{0, 1, 4};
  IndexGroup indC{0, 2, 4};
  IndexGroup indD{0, 3, 4};

  // Define some combinations of index groups. These will define the number of partitions.
  std::array<IndexGroups, kNumPartitions> groups{
      IndexGroups{{indA}},
      IndexGroups{{indA, indB}},
      IndexGroups{{indB}},
      IndexGroups{{indC, indD, indA}}};

  // Define index groups for a set of samples
  std::array<int, kNumSamples> samples{{0, 1, 0, 2, 3, 1, 3, 0, 1, 0}};
  return InitializeSamples(groups, samples);
}

static std::array<int, kNumSamples> InitializeIds() {
  // Define some ids.
  std::array<int, kNumPartitions> ids{{-1, 0, 1, 2}};

  // Define ids for a set of samples
  std::array<int, kNumSamples> samples{{2, 1, 1, 3, 1, 0, 3, 2, 0, 1}};
  return InitializeSamples(ids, samples);
}

static void TestPartitions(Span<ContactPartition const> partitions) {
  // Test if the intersection of all partitions is empty
  for (int i = 0; i < partitions.size(); i++) {
    auto const& partitionA = partitions[i].GetSubset();
    for (int j = i + 1; j < partitions.size(); j++) {
      auto const& partitionB = partitions[j].GetSubset();
      auto intersection = partitionA.Intersection(partitionB);
      EXPECT_TRUE(intersection.IsEmpty());
    }
  }

  // Test if the union of all partitions is the full set
  SubsetMap all{partitions[0].GetSubset()};
  for (int i = 1; i < partitions.size(); i++) {
    all = all.Union(partitions[i].GetSubset());
  }
  EXPECT_TRUE(all.GetSubsetSize() == all.GetFullSetSize());
}

TEST(ContactPartition, IndexGroups) {
  auto samples = InitializeIndexGroups();
  auto partitions = CreateContactPartitions<IndexGroupsDescriptor>(samples);

  EXPECT_EQ(partitions.size(), 4);
  TestPartitions(partitions);
}

TEST(ContactPartition, Ids) {
  auto samples = InitializeIds();
  auto partitions = CreateContactPartitions<IdDescriptor>(samples);

  EXPECT_EQ(partitions.size(), 4);
  TestPartitions(partitions);
}

TEST(ContactPartition, Combined) {
  auto partitionsA = CreateContactPartitions<IndexGroupsDescriptor>(InitializeIndexGroups());
  auto partitionsB = CreateContactPartitions<IdDescriptor>(InitializeIds());
  auto partitions = CombinePartitions(partitionsA, partitionsB);

  EXPECT_GE(partitions.size(), 4);
  EXPECT_LE(partitions.size(), kNumSamples);
  TestPartitions(partitions);
}
