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

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Include this header instead of <mochi_core/geometry/model_utils.h> if you want to call these
// methods via dynamic linking mochi_physics, rather than static linking mochi_core. Usage syntax
// will be identical except for the namespace (mochi::model_utils vs mochi::model).
//
// See <mochi_core/geometry/model_utils.h> for API documentation.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <mochi_core/geometry/grid_sdf_params.h>
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_physics/utils/mochi_physics_macros.h>

#include <string_view>

namespace mochi {
struct CoordinateSpace;
} // namespace mochi

namespace mochi::model_utils {
[[nodiscard]] MOCHI_API ModelData LoadFromFile(std::string_view path, Error& error);
[[nodiscard]] MOCHI_API ModelData LoadFromFileUnchecked(std::string_view path, Error& error);
[[nodiscard]] MOCHI_API ModelData
LoadFromBytes(Span<char const> data, MeshFileType format, Error& error);
[[nodiscard]] MOCHI_API ModelData
LoadFromBytesUnchecked(Span<char const> data, MeshFileType format, Error& error);
[[nodiscard]] inline ModelData LoadFromBytes(Span<char const> data, Error& error) {
  return LoadFromBytes(data, MeshFileType::Legacy, error);
}
[[nodiscard]] inline ModelData LoadFromBytesUnchecked(Span<char const> data, Error& error) {
  return LoadFromBytesUnchecked(data, MeshFileType::Legacy, error);
}
MOCHI_API void
SaveToFile(ModelData const& data, std::string_view path, FileFormat format, Error& error);
MOCHI_API void
SaveToFile(ModelDataView const& data, std::string_view path, FileFormat format, Error& error);
[[nodiscard]] MOCHI_API DynamicString SaveToJsonString(ModelData const& data, Error& error);
MOCHI_API void Validate(ModelDataView const& data, Error& error);
MOCHI_API void AutoCorrect(ModelData& data, Error& error);
MOCHI_API void BakeTransform(
    ModelData& data,
    Real3 const& scale,
    Quaternion const& rotation,
    Real3 const& translation,
    Error& error);
MOCHI_API void
BakeTransform(ModelData& data, Real3 const& scale, TransformRT const& transform, Error& error);
MOCHI_API void BakeCoordinateSpaceTransform(
    MeshData& data,
    CoordinateSpace const& fromSpace,
    CoordinateSpace const& toSpace,
    Error& error);
MOCHI_API void BakeCoordinateSpaceTransform(
    ModelData& data,
    CoordinateSpace const& fromSpace,
    CoordinateSpace const& toSpace,
    Error& error);
MOCHI_API void BakeSdf(ModelData& data, GridSdfParams const& params, Error& error);
MOCHI_API void GenerateVisualMeshEmbedding(ModelData& data, Error& error);
MOCHI_API void FlipWindingOrder(MeshData& data, Error& error);
MOCHI_API void FlipWindingOrder(ModelData& data, Error& error);
} // namespace mochi::model_utils
