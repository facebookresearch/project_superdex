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

#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>

#include <utility>

namespace mochi {

/*
   Data structure for mesh blending stored from the point-of-view of a source mesh.
*/
struct BlendingDataSourceMesh {
  // Indices of source nodes that are used in the blending (i.e., weight source != 0).
  DynamicArray<int> nodesSource;
  // Weights of source mesh nodes, of size equal to the source mesh.
  DynamicArray<real> weightsSource;
  // Mapping from indices of the source mesh to indices of the target mesh. A value of -1 indicates
  // that the source blending weight is 0, and hence there's no corresponding target node.
  DynamicArray<int> mappingSourceToTarget;
  // Mapping from indices of the target mesh to indices of the source mesh. A value of -1 indicates
  // that the source blending weight is 0, and hence there's no corresponding source node.
  // Note that each valid entry in mappingTargetToSource is the node index wrt the source ordering.
  DynamicArray<int> mappingTargetToSource;
};

/*
   Data structure for mesh blending stored from the point-of-view of the target mesh.
*/
struct BlendingDataTargetMesh {
  // For every node of the target mesh, two node indices in the source meshes. If a weight is zero,
  // the corresponding index is irrelevant.
  DynamicArray<int> indices;
  // For every node of the target mesh, the weights of the two source nodes.
  DynamicArray<real> weights;

  // This function extracts the blending data for one of the source meshes. It requires as input the
  // number of nodes in the source mesh.
  template <int sourceId>
  BlendingDataSourceMesh GetSourceBlendingData(int sourceNodes) const {
    static_assert(sourceId >= 0 && sourceId <= 1);
    DynamicArray<int> outNodesSource;
    outNodesSource.reserve(sourceNodes);
    DynamicArray<real> outWeights(sourceNodes, 0_r);
    DynamicArray<int> outMappingSourceToTarget(sourceNodes, -1);
    DynamicArray<int> outMappingTargetToSource(indices.size() / 2, -1);
    for (int i = sourceId; i < indices.size(); i += 2) {
      if (weights[i] > 0_r) {
        MOCHI_ASSERT(indices[i] >= 0 && indices[i] < sourceNodes, "Wrong node index");
        outNodesSource.emplace_back(indices[i]);
        outWeights[indices[i]] = weights[i];
        outMappingSourceToTarget[indices[i]] = i / 2;
        outMappingTargetToSource[i / 2] = indices[i];
      }
    }
    return {
        std::move(outNodesSource),
        std::move(outWeights),
        std::move(outMappingSourceToTarget),
        std::move(outMappingTargetToSource)};
  }
};

} // namespace mochi
