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
#include "model_data.h" // Reverse include for intellisense

namespace mochi {

inline GridSdfDataView::GridSdfDataView(GridSdfData const& src)
    : dims(src.dims),
      values(MakeConstSpan(src.values)),
      bounds(src.bounds),
      negativeValueBounds(src.negativeValueBounds),
      scale(src.scale),
      rotation(src.rotation),
      translation(src.translation) {}

inline GridSdfData::GridSdfData(GridSdfDataView const& src)
    : dims(src.dims),
      values(src.values),
      bounds(src.bounds),
      negativeValueBounds(src.negativeValueBounds),
      scale(src.scale),
      rotation(src.rotation),
      translation(src.translation) {}

inline ModelDataView::ModelDataView(ModelData const& src)
    : box(src.box),
      plane(src.plane),
      sphere(src.sphere),
      experimentalDataDetected(src.experimentalDataDetected) {
  if (src.mesh) {
    this->mesh.emplace(MeshDataView{*src.mesh});
  }
  if (src.visualMesh) {
    this->visualMesh.emplace(MeshDataView{*src.visualMesh});
  }
  if (src.blending) {
    DynamicArray<BlendingDataView> blendingDataViewArray;
    blendingDataViewArray.reserve(src.blending->size());
    for (auto const& blendingData : *src.blending) {
      blendingDataViewArray.emplace_back(BlendingDataView{blendingData});
    }
    this->blending.emplace(std::move(blendingDataViewArray));
  }
  if (src.constrainedNodes) {
    this->constrainedNodes.emplace(Span<int const>{*src.constrainedNodes});
  }
  if (src.elementFrameAxes) {
    this->elementFrameAxes.emplace(Span<real const>{*src.elementFrameAxes});
  }
  if (src.sdf) {
    this->sdf.emplace(GridSdfDataView{*src.sdf});
  }
  if (src.material) {
    this->material.emplace(PerElementSoftMaterialDataView{*src.material});
  }
}

inline ModelData::ModelData(ModelDataView const& src)
    : box(src.box),
      plane(src.plane),
      sphere(src.sphere),
      experimentalDataDetected(src.experimentalDataDetected) {
  if (src.mesh) {
    this->mesh.emplace(MeshData{*src.mesh});
  }
  if (src.visualMesh) {
    this->visualMesh.emplace(MeshData{*src.visualMesh});
  }
  if (src.blending) {
    DynamicArray<BlendingData> blendingDataArray;
    blendingDataArray.reserve(src.blending->size());
    for (auto const& blendingData : *src.blending) {
      blendingDataArray.emplace_back(BlendingData{blendingData});
    }
    this->blending.emplace(std::move(blendingDataArray));
  }
  if (src.constrainedNodes) {
    this->constrainedNodes.emplace(DynamicArray<int>{*src.constrainedNodes});
  }
  if (src.elementFrameAxes) {
    this->elementFrameAxes.emplace(DynamicArray<real>{*src.elementFrameAxes});
  }
  if (src.sdf) {
    this->sdf.emplace(GridSdfData{*src.sdf});
  }
  if (src.material) {
    this->material.emplace(PerElementSoftMaterialData{*src.material});
  }
}

} // namespace mochi
