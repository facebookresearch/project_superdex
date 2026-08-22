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

// PLEASE DO NOT ADD OTHER INCLUDES HERE. This header is included in the mochi_physics public API.
#include "material_params.h" // Reverse include for intellisense

namespace mochi {

inline PerElementSoftMaterialDataView::PerElementSoftMaterialDataView(
    PerElementSoftMaterialData const& other)
    : type(other.type),
      psdStrategy(other.psdStrategy),
      youngsModulus(other.youngsModulus),
      poissonRatio(other.poissonRatio),
      anisoAlpha(other.anisoAlpha),
      anisoLength(other.anisoLength),
      anisoTheta(other.anisoTheta),
      anisoPhi(other.anisoPhi),
      arapStiffness(other.arapStiffness),
      shapeTargetTensor(other.shapeTargetTensor) {}

inline PerElementSoftMaterialData::PerElementSoftMaterialData(
    PerElementSoftMaterialDataView const& other)
    : type(other.type),
      psdStrategy(other.psdStrategy),
      youngsModulus(other.youngsModulus),
      poissonRatio(other.poissonRatio),
      anisoAlpha(other.anisoAlpha),
      anisoLength(other.anisoLength),
      anisoTheta(other.anisoTheta),
      anisoPhi(other.anisoPhi),
      arapStiffness(other.arapStiffness),
      shapeTargetTensor(other.shapeTargetTensor) {}

} // namespace mochi
