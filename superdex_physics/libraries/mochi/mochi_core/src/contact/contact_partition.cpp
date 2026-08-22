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

#include <utility>
#include <vector>

using namespace mochi;

template <typename DescriptorT>
std::vector<ContactPartition> mochi::CreateContactPartitions(
    Span<typename DescriptorT::Descriptor const> samples) {
  static_assert(
      std::is_base_of_v<PartitionDescriptorImpl<typename DescriptorT::Descriptor>, DescriptorT>);

  MOCHI_PROFILE_SCOPE();

  //  Insert the sample descriptors in a set (to find all unique descriptors) and an unordered_map
  //  from descriptor to sample id.
  typename DescriptorT::Set descriptors;
  typename DescriptorT::Map descriptorToSamples;
  for (int s = 0; s < samples.size(); s++) {
    descriptors.insert(samples[s]);
    descriptorToSamples.emplace(samples[s], s);
  }

  // Create partitions
  std::vector<ContactPartition> outPartitions;
  outPartitions.reserve(descriptors.size());
  size_t numSamplesAdded = 0;
  for (auto const& descriptor : descriptors) {
    DescriptorT descriptorWrapper(descriptor);
    size_t numSamples = descriptorToSamples.count(descriptor);
    auto sampleRange = descriptorToSamples.equal_range(descriptor);
    std::vector<int> indices;
    indices.reserve(numSamples);
    for (auto it = sampleRange.first; it != sampleRange.second; it++) {
      indices.emplace_back(it->second);
    }
    auto subset = SubsetMap::FromUnsortedList(indices, isize(samples));
    ContactPartition::DoFDescriptors dofs{descriptorWrapper.GetDoFs()};
    ContactPartition partition{std::move(subset), std::move(dofs)};
    numSamplesAdded += numSamples;
    outPartitions.emplace_back(std::move(partition));
  }

  MOCHI_ASSERT(numSamplesAdded == samples.size(), "Total samples in the partitions don't match.");

  return outPartitions;
}

template std::vector<ContactPartition> mochi::CreateContactPartitions<IndexGroupsDescriptor>(
    Span<typename IndexGroupsDescriptor::Descriptor const> samples);

template std::vector<ContactPartition> mochi::CreateContactPartitions<IdDescriptor>(
    Span<typename IdDescriptor::Descriptor const> samples);

std::vector<ContactPartition> mochi::CombinePartitions(
    Span<ContactPartition const> partitionsA,
    Span<ContactPartition const> partitionsB) {
  // Intersect the two sets of partitions, and create a new partition whenever the intersection is
  // non-empty.
  std::vector<ContactPartition> outPartitions;
  outPartitions.reserve(partitionsA.size() * partitionsB.size());
  for (auto const& pA : partitionsA) {
    for (auto const& pB : partitionsB) {
      auto intersection = pA.GetSubset().Intersection(pB.GetSubset());
      if (!intersection.IsEmpty()) {
        ContactPartition::DoFDescriptors dofs;
        dofs.reserve(pA.GetDofDescriptors().size() + pB.GetDofDescriptors().size());
        dofs.insert(dofs.end(), pA.GetDofDescriptors().begin(), pA.GetDofDescriptors().end());
        dofs.insert(dofs.end(), pB.GetDofDescriptors().begin(), pB.GetDofDescriptors().end());
        ContactPartition partition{std::move(intersection), std::move(dofs)};
        outPartitions.emplace_back(std::move(partition));
      }
    }
  }
  return outPartitions;
}
