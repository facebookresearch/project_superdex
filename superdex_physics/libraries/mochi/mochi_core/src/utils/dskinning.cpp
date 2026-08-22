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

#include <mochi_core/utils/dskinning.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <unordered_map>
#include <utility>
#include <vector>

struct VertexBoneKey {
  int boneId;
  int vertexId;

  inline bool operator==(VertexBoneKey const& other) const {
    return other.boneId == boneId && other.vertexId == vertexId;
  }
};

struct VertexBoneKeyHash {
  size_t operator()(VertexBoneKey const& key) const {
    return std::hash<int>()(key.boneId) ^ std::hash<int>()(key.vertexId);
  }
};

namespace mochi {

DTransformParameterizationCollection DTransformParameterizationCollection::FromRootFromBone(
    std::vector<TransformRT> const& referenceRootFromBone,
    real scale) {
  std::vector<TransformSRT> temp;
  temp.reserve(referenceRootFromBone.size());
  std::transform(
      referenceRootFromBone.begin(),
      referenceRootFromBone.end(),
      std::back_inserter(temp),
      [](TransformRT const& transform) { return TransformSRT(transform); });
  return FromRootFromBone(temp, scale);
}

DTransformParameterizationCollection DTransformParameterizationCollection::FromRootFromBone(
    std::vector<TransformSRT> const& referenceRootFromBone,
    real scale) {
  DTransformParameterizationCollection result;
  result.reserve(referenceRootFromBone.size());
  std::transform(
      referenceRootFromBone.begin(),
      referenceRootFromBone.end(),
      std::back_inserter(result),
      [scale](TransformSRT const& rootFromBone) {
        auto postTransform = TransformSRT::Identity();
        auto preTransform =
            TransformSRT(scale) * TransformSRT(Invert(rootFromBone)) * Invert(TransformSRT(scale));

        return DTransformParameterization{preTransform, postTransform};
      });

  return result;
}

/*
Implementation for SkinningWeightsByBone.
*/
SkinningWeightsByBone::SkinningWeightsByBone(
    Span<int const> skinIdx,
    Span<real const> skinWeight,
    int weightsPerNode,
    int numBones) {
  MOCHI_ASSERT(
      skinIdx.size() == skinWeight.size(), "Weights and indices arrays do not have the same size!");

  std::unordered_map<VertexBoneKey, real, VertexBoneKeyHash> vbPairsMap;

  int vertCount = isize(skinWeight) / weightsPerNode;
  MOCHI_ASSERT(vertCount * weightsPerNode == skinWeight.size());

  // Construct all bone vertex pairs
  int idx = 0;
  for (int vertexId = 0; vertexId < vertCount; ++vertexId) {
    for (int vertexBone = 0; vertexBone < weightsPerNode; ++vertexBone, ++idx) {
      auto weight = skinWeight[idx];
      auto boneId = skinIdx[idx];

      auto key = VertexBoneKey{boneId, vertexId};
      auto it = vbPairsMap.find(key);

      if (weight != 0_r) {
        if (it != vbPairsMap.end()) {
          // Duplicate weight! Simply add to existing
          it->second += weight;
        } else {
          vbPairsMap.emplace_hint(it, key, weight);
        }
      }
    }
  }

  // Drain the map into a vector
  std::vector<VertexBonePair<real>> vbPairsByBone;
  vbPairsByBone.reserve(vbPairsMap.size());
  for (auto const& [key, weight] : vbPairsMap) {
    vbPairsByBone.emplace_back(VertexBonePair<real>{key.boneId, key.vertexId, weight});
  }

  // Sort bone vertex pairs by bone id.
  std::stable_sort(vbPairsByBone.begin(), vbPairsByBone.end(), [](auto const& p1, auto const& p2) {
    return p1.boneId < p2.boneId;
  });

  // Find the boundaries between subsets of the bone vertex pairs of the same bone id.
  boneRanges.emplace_back(0);
  int currentBone = 0;
  for (int i = 0; i < vbPairsByBone.size(); ++i) {
    for (auto const& pair = vbPairsByBone[i]; pair.boneId > currentBone; ++currentBone) {
      boneRanges.emplace_back(i);
    }
  }

  // Insert remaining bones (if any and all bones requested)
  MOCHI_ASSERT(boneRanges.size() <= numBones);
  boneRanges.insert(boneRanges.end(), numBones - boneRanges.size(), vbPairsByBone.size());

  // Emplace last
  boneRanges.emplace_back(vbPairsByBone.size());

  // Initialize the data structure.
  this->boneVertexPairsByBone = std::move(vbPairsByBone);
  this->vertexCount = vertCount;
}

std::vector<DSkinningTransform::VertexBones> DSkinningTransform::BuildPerVertexBones(
    SkinningWeightsByBone const& weights) {
  std::vector<VertexBones> perVertexBones(weights.GetVertexCount());
  for (auto const& pair : weights.boneVertexPairsByBone) {
    perVertexBones[pair.vertexId].emplace_back(pair.boneId, pair.weight);
  }

  // Sort the bones of each vertex by bone ID to improve memory access in dskinning transformations.
  for (auto& vertexBones : perVertexBones) {
    std::sort(vertexBones.begin(), vertexBones.end(), [](auto const& a, auto const& b) {
      return a.first < b.first;
    });
  }

  return perVertexBones;
}

bool SkinningWeightsByBone::HasUnusedBones() const {
  for (int bone = 0; bone < GetBoneCount(); ++bone) {
    auto weightStart = boneRanges[bone];
    auto weightEnd = boneRanges[bone + 1];

    // There are vertices attached to this bone
    if (weightStart == weightEnd) {
      return true;
    }
  }

  return false;
}

} // namespace mochi
