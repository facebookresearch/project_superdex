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

// PLEASE DO NOT ADD OTHER INCLUDES HERE. This header is included in the mochi_physics public API.
#include "mesh_data.h" // Reverse include for intellisense

namespace mochi {

inline BlendingDataView::BlendingDataView(BlendingData const& src)
    : sourceShape(src.sourceShape), indices(src.indices), weights(src.weights) {}

inline BlendingData::BlendingData(BlendingDataView const& src)
    : sourceShape(src.sourceShape), indices(src.indices), weights(src.weights) {}

inline SkinningDataView::SkinningDataView(SkinningData const& src)
    : weightsPerNode(src.weightsPerNode), indices(src.indices), weights(src.weights) {}

inline SkinningData::SkinningData(SkinningDataView const& src)
    : weightsPerNode(src.weightsPerNode), indices(src.indices), weights(src.weights) {}

inline MeshDataView::MeshDataView(MeshData const& src)
    : nodesPerElement(src.nodesPerElement),
      coordinates(src.coordinates),
      connectivity(src.connectivity),
      skinning(src.skinning ? std::make_optional(SkinningDataView{*src.skinning}) : std::nullopt) {}

inline MeshData::MeshData(MeshDataView const& src)
    : nodesPerElement(src.nodesPerElement),
      coordinates(src.coordinates),
      connectivity(src.connectivity),
      skinning(src.skinning ? std::make_optional(SkinningData{*src.skinning}) : std::nullopt) {}

inline int MeshData::GetNumNodes() const {
  MOCHI_ASSERT_VERBOSE(isize(coordinates) % kMeshDataSpaceDim == 0);
  return isize(coordinates) / kMeshDataSpaceDim;
}

inline int MeshData::GetNumElements() const {
  return (!connectivity.empty() && (nodesPerElement > 0)) ? isize(connectivity) / nodesPerElement
                                                          : 0;
}

inline bool MeshData::IsEmpty() const {
  return GetNumNodes() == 0;
}

inline int MeshDataView::GetNumNodes() const {
  MOCHI_ASSERT_VERBOSE(isize(coordinates) % kMeshDataSpaceDim == 0);
  return isize(coordinates) / kMeshDataSpaceDim;
}

inline int MeshDataView::GetNumElements() const {
  return (!connectivity.empty() && (nodesPerElement > 0)) ? isize(connectivity) / nodesPerElement
                                                          : 0;
}

inline bool MeshDataView::IsEmpty() const {
  return GetNumNodes() == 0;
}

} // namespace mochi
