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
#include <utility>

mochi::DSkinningTransform::DSkinningTransform(
    Span<int const> skinIndices,
    Span<real const> skinWeights,
    int weightsPerNode,
    DynamicArray<TransformRT> bonePreTransforms)
    : preTransforms(std::move(bonePreTransforms)) {
  MOCHI_ASSERT(
      skinIndices.size() == skinWeights.size(),
      "Weights and indices arrays do not have the same size!");
  MOCHI_ASSERT(weightsPerNode > 0, "Weights per node must be positive!");
  MOCHI_ASSERT(
      isize(skinWeights) % weightsPerNode == 0, "Weights must be a multiple of weights per node!");

  int const vertexCount = isize(skinWeights) / weightsPerNode;
  perVertexBones.resize(vertexCount);

  for (int vertexId = 0; vertexId < vertexCount; ++vertexId) {
    auto& vertexBones = perVertexBones[vertexId];
    vertexBones.reserve(weightsPerNode);
    for (int vertexBone = 0; vertexBone < weightsPerNode; ++vertexBone) {
      int const index = vertexId * weightsPerNode + vertexBone;
      real const weight = skinWeights[index];
      if (weight == 0_r) {
        continue;
      }

      int const boneId = skinIndices[index];
      MOCHI_ASSERT_VERBOSE(boneId >= 0 && boneId < GetBoneCount(), "Bone index out of range!");
      auto const matchesBoneId = [boneId](auto const& bone) { return bone.first == boneId; };
      auto const position = std::ranges::find_if(vertexBones, matchesBoneId);
      if (position != vertexBones.end()) {
        position->second += weight;
      } else {
        vertexBones.emplace_back(boneId, weight);
      }
    }
    std::ranges::sort(vertexBones);
    totalPairs += isize(vertexBones);
  }
}
