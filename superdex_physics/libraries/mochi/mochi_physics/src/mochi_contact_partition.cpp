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

#include "mochi_contact_partition.h"
#include "mochi_articulated_body.h"
#include "mochi_blended.h"
#include "mochi_discretization_components.h"
#include "mochi_group.h"

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

using namespace mochi;

static std::vector<Int3> GetNodeIndices(entt::registry const& reg, entt::entity entity) {
  auto extract = [](auto const& discretization) {
    return discretization.Visit([&](auto const& discretizationImpl) {
      using ElementT = typename std::decay_t<decltype(discretizationImpl)>::ElementT;
      int const numSamples = ElementT::kNumQuadPoints * isize(discretizationImpl.femElements);
      std::vector<Int3> outNodeIndices(numSamples);
      for (int i = 0; i < numSamples; i++) {
        outNodeIndices[i] = GetNodeInfo<ElementT>(discretizationImpl.femElements, i);
      }
      return outNodeIndices;
    });
  };
  if (auto const* boundary = reg.try_get<CFemBoundaryDiscretization const>(entity)) {
    return extract(*boundary);
  }
  return extract(reg.get<CFemSurfaceDiscretization const>(entity));
}

static std::vector<IndexGroups>
CreateSkinningDofGroups(entt::registry const& reg, entt::entity entity, entt::entity articulated) {
  // Fetch skinning of the entity that will be partitioned.
  auto const& skinningData = reg.get<CArticulatedSkinningData const>(entity).skinningData;

  // Fetch components of the articulated actor: bone actors, articulated DoFs
  auto const& boneActors = reg.get<CGroupMembers const>(articulated).actors;
  int numArticulatedDofs = reg.get<CArticulatedProps const>(articulated).reducedDofsDim;

  // Prepare node indices per collision sample
  auto nodeIndices = GetNodeIndices(reg, entity);

  // Prepare a vector of joint dofs per bone
  std::vector<Span<int const>> jointDofs;
  jointDofs.reserve(boneActors.size());
  for (auto bone : boneActors) {
    jointDofs.emplace_back(reg.get<CArticulatedRigidJacobian const>(bone).dofs);
  }

  // Traverse all the collision samples and compute their DoF index groups.
  static int constexpr kMaxIndexGroups = 4;
  std::vector<IndexGroups> indexGroups;
  indexGroups.reserve(nodeIndices.size());
  for (auto const& indices : nodeIndices) {
    // Collect bone indices for the nodes
    std::vector<int> boneIndices;
    for (int i = 0; i < 3; i++) {
      // Fetch bone indices per node
      int const spanSize = skinningData.weightsPerNode;
      auto const boneWeights = Span(&skinningData.weights[indices[i] * spanSize], spanSize);
      auto const boneIndicesThis = Span(&skinningData.indices[indices[i] * spanSize], spanSize);
      for (int j = 0; j < spanSize; j++) {
        // Check if bone indices are valid and insert
        if (boneWeights[j] != 0_r) {
          boneIndices.push_back(boneIndicesThis[j]);
        }
      }
    }
    std::sort(boneIndices.begin(), boneIndices.end());
    boneIndices.erase(std::unique(boneIndices.begin(), boneIndices.end()), boneIndices.end());

    // Collect joint DoFs for the bones
    std::vector<int> dofs;
    for (auto const& boneId : boneIndices) {
      auto const& jointDofsBone = jointDofs[boneId];
      dofs.insert(dofs.end(), jointDofsBone.begin(), jointDofsBone.end());
    }
    std::sort(dofs.begin(), dofs.end());
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());

    // Convert to index groups
    IndexGroups sampleDoFs = CreateIndexGroups(std::vector<int>(dofs.begin(), dofs.end()), false);
    if (sampleDoFs.size() > kMaxIndexGroups) {
      // Use all DoFs if the number of index groups is too large
      sampleDoFs.clear();
      sampleDoFs.push_back({0, 0, numArticulatedDofs});
    }
    indexGroups.emplace_back(std::move(sampleDoFs));
  }

  return indexGroups;
}

static std::vector<int> CreateSoftActorIds(entt::registry const& reg, entt::entity entity) {
  // Fetch blending of the entity that will be partitioned.
  auto const& blending = reg.get<CBlendingData const>(entity);

  // Prepare node indices per collision sample
  auto nodeIndices = GetNodeIndices(reg, entity);

  // Traverse all the collision samples and compute their soft-actor id
  std::vector<int> ids;
  ids.reserve(nodeIndices.size());
  for (auto const& indices : nodeIndices) {
    int idSoft = -1;
    // Check if some element node blends some soft actor
    for (int i = 0; i < 3; i++) {
      for (int id = 0; id < blending.size(); id++) {
        if (blending[id].mappingTargetToSource[indices[i]] != -1) {
          MOCHI_ASSERT(
              idSoft == -1 || idSoft == id,
              "A collision sample can't be governed by more than one soft actor");
          idSoft = id;
        }
      }
    }
    ids.emplace_back(idSoft);
  }

  return ids;
}

static std::vector<ContactPartition> CreatePartitionsFromStrategy(
    entt::registry const& reg,
    entt::entity entity,
    entt::entity articulated,
    ContactPartitionStrategy strategy) {
  switch (strategy) {
    case ContactPartitionStrategy::SoftActorId: {
      auto ids = CreateSoftActorIds(reg, entity);
      return CreateContactPartitions<IdDescriptor>(ids);
    }
    case ContactPartitionStrategy::SkinningDofGroups:
    default: {
      auto dofGroups = CreateSkinningDofGroups(reg, entity, articulated);
      return CreateContactPartitions<IndexGroupsDescriptor>(dofGroups);
    }
  }
}

std::vector<ContactPartition> mochi::InitializeContactPartitions(
    entt::registry const& reg,
    entt::entity entity,
    entt::entity articulated,
    Span<ContactPartitionStrategy const> strategies) {
  MOCHI_ASSERT(!strategies.empty(), "No strategies for contact partitions");

  // Initialize result with the first strategy
  std::vector<ContactPartition> result =
      CreatePartitionsFromStrategy(reg, entity, articulated, strategies[0]);

  // Combine the other strategies
  for (int i = 1; i < strategies.size(); i++) {
    std::vector<ContactPartition> partitions =
        CreatePartitionsFromStrategy(reg, entity, articulated, strategies[i]);
    result = CombinePartitions(result, partitions);
  }

  return result;
}
