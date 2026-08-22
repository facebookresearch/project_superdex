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

#pragma once

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/spmat_utils.h>
#include <mochi_core/utils/subset_map.h>

#include <set>
#include <unordered_map>
#include <variant>
#include <vector>

/**************************************************************************************************
 * The classes in this file are used for the creation of contact partitions. Given the full set of
 * contact samples of a surface {xi} and a descriptor associated with each sample {fi}, the contact
 * samples are partitioned into a set of disjoint partitions {Sk = {... xi, xj... }}, such that all
 * the samples in the same partition share the same descriptor, i.e., fi = fj.
 *
 * Contact partitions may be created according to different strategies, i.e., different types of
 * sample descriptors. This is done by templatizing the functions according to the descriptor type.
 * Partition strategies can also be combined. Given two descriptor types f and g, the combined
 * descriptor is simply the tuple (f, g). Then, two samples xi and xj belong to the same partition
 * if fi = fj and gi = gj.
 *
 * There are two notable functions for the creation of contact partitions:
 * 1) CreateContactPartitions() takes as input the descriptor values for all contact samples, and
 * outputs the resulting partitions.
 * 2) CombinePartitions() takes as input partitions created according to two descriptors f and g,
 * and outputs the partitions resulting from the combined descriptor (f, g).
 *
 * Currently, contact partitions may be created according to two possible descriptors/strategies:
 * 1) IndexGroups: This descriptor/strategy is used for partitioning the surface of skinned meshes,
 * based on the underlying articulated-body DoFs that govern the contact samples.
 * 2) int: This descriptor/strategy is used for partitioning the surface of soft skinned actors,
 * based on the id of the governing soft actor.
 */

namespace mochi {

/**************************************************************************************************
 * Each descriptor type is connected to some information that is representative of simulation DoFs.
 * At simulation runtime, the DoF representation will be fetched by contact assembly and used
 * appropriately assuming knowledge of the underlying partition descriptor/strategy. This variant
 * wraps the DoF representations for the currently supported descriptors:
 * - std::vector<int>: For IndexGroups descriptor. It stores the DoF indices of an articulated body.
 * - int: For int descriptor. It stores the soft-actor id.
 */
using VariantDofDescriptor = std::variant<DynamicArray<int>, int>;

/**************************************************************************************************
 * This is a wrapper class for descriptor types, that defines the interface needed for the execution
 * of CreateContactPartitions(). This function CreateContactPartitions() must be templatized by a
 * child class of PartitionDescriptorImpl.
 */
template <typename T>
class PartitionDescriptorImpl {
 public:
  // Types used by CreateContactPartitions()
  using Descriptor = T;
  using Map = std::unordered_multimap<T, int>;
  using Set = std::set<T>;

  PartitionDescriptorImpl(T const& descriptor) : _descriptor(descriptor) {}

  virtual ~PartitionDescriptorImpl() = default;

  // Mandatory interface of all descriptors, to fetch the runtime DoF representation associated with
  // the descriptor type.
  virtual VariantDofDescriptor GetDoFs() const = 0;

 protected:
  T const& _descriptor;
};

/**************************************************************************************************
 * Partition descriptor for index groups. Used with articulated bodies.
 */
class IndexGroupsDescriptor : public PartitionDescriptorImpl<IndexGroups> {
 public:
  // Overwrite Map and Set with type-specific compare classes
  using Map = std::unordered_multimap<IndexGroups, int, IndexGroupsHash, IndexGroupsEqual>;
  using Set = std::set<IndexGroups, IndexGroupsLess>;

  IndexGroupsDescriptor(IndexGroups const& descriptor) : PartitionDescriptorImpl(descriptor) {}

  VariantDofDescriptor GetDoFs() const override {
    return _descriptor.GetAllDofs();
  }
};

/**************************************************************************************************
 * Partition descriptor for ids. Used with soft actors.
 */
class IdDescriptor : public PartitionDescriptorImpl<int> {
 public:
  IdDescriptor(int const& descriptor) : PartitionDescriptorImpl(descriptor) {}

  VariantDofDescriptor GetDoFs() const override {
    return _descriptor;
  }
};

/***************************************************************************************************
 * Class to hold all contact samples belonging to the same partition. It holds, most importantly,
 * the indices of the contact samples and the DoF representation of the descriptor value for the
 * partition. Note that, if the partition is created from several descriptor types, then there must
 * be a DoF representation for each descriptor type.
 */
class ContactPartition {
 public:
  using DoFDescriptors = std::vector<VariantDofDescriptor>;

  ContactPartition(SubsetMap&& subset, DoFDescriptors&& dofs) : _subset(subset), _dofs(dofs) {}

  Span<int const> GetIndices() const {
    return _subset.GetStorage();
  }

  Span<VariantDofDescriptor const> GetDofDescriptors() const {
    return _dofs;
  }

  SubsetMap const& GetSubset() const {
    return _subset;
  }

 private:
  SubsetMap _subset; // Indices of the samples that belong to the partition
  DoFDescriptors _dofs; // DoF representations for the partition
};

/***************************************************************************************************
 * Function to create partitions from a span of collision samples with their descriptors. All
 * samples with the same partition descriptor are placed in the same partition. The function is
 * templatized according to the type of partition descriptor, which must inherit from
 * PartitionDescriptorImpl.
 */
template <typename DescriptorT>
std::vector<ContactPartition> CreateContactPartitions(
    Span<typename DescriptorT::Descriptor const> samples);

/***************************************************************************************************
 * Function to create new partitions based on the combination of two descriptors. If partitionsA and
 * partitionsB have been created according to one descriptor type each, then the output partitions
 * use the union of both descriptor types.
 */
std::vector<ContactPartition> CombinePartitions(
    Span<ContactPartition const> partitionsA,
    Span<ContactPartition const> partitionsB);

} // namespace mochi
