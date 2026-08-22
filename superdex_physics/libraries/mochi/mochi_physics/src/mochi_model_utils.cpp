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

#include <mochi_physics/utils/mochi_model_utils.h>

#include <mochi_core/geometry/model_utils.h>

using namespace mochi;

ModelData model_utils::LoadFromFile(std::string_view path, Error& error) {
  return model::LoadFromFile(path, error);
}

ModelData model_utils::LoadFromFileUnchecked(std::string_view path, Error& error) {
  return model::LoadFromFileUnchecked(path, error);
}

ModelData model_utils::LoadFromBytes(Span<char const> data, MeshFileType format, Error& error) {
  return model::LoadFromBytes(data, format, error);
}

ModelData
model_utils::LoadFromBytesUnchecked(Span<char const> data, MeshFileType format, Error& error) {
  return model::LoadFromBytesUnchecked(data, format, error);
}

void model_utils::SaveToFile(
    ModelData const& data,
    std::string_view path,
    FileFormat format,
    Error& error) {
  return model::SaveToFile(data, path, format, error);
}

void model_utils::SaveToFile(
    ModelDataView const& data,
    std::string_view path,
    FileFormat format,
    Error& error) {
  return model::SaveToFile(data, path, format, error);
}

DynamicString model_utils::SaveToJsonString(ModelData const& data, Error& error) {
  return model::SaveToJsonString(data, error);
}

void model_utils::Validate(ModelDataView const& data, Error& error) {
  return model::Validate(data, error);
}

void model_utils::AutoCorrect(ModelData& data, Error& error) {
  return model::AutoCorrect(data, error);
}

void model_utils::BakeTransform(
    ModelData& data,
    Real3 const& scale,
    Quaternion const& rotation,
    Real3 const& translation,
    Error& error) {
  return model::BakeTransform(data, scale, rotation, translation, error);
}

void model_utils::BakeTransform(
    ModelData& data,
    Real3 const& scale,
    TransformRT const& transform,
    Error& error) {
  return model::BakeTransform(data, scale, transform, error);
}

void model_utils::BakeCoordinateSpaceTransform(
    MeshData& data,
    CoordinateSpace const& fromSpace,
    CoordinateSpace const& toSpace,
    Error& error) {
  model::BakeCoordinateSpaceTransform(data, fromSpace, toSpace, error);
}

void model_utils::BakeCoordinateSpaceTransform(
    ModelData& data,
    CoordinateSpace const& fromSpace,
    CoordinateSpace const& toSpace,
    Error& error) {
  model::BakeCoordinateSpaceTransform(data, fromSpace, toSpace, error);
}

void model_utils::BakeSdf(ModelData& data, GridSdfParams const& params, Error& error) {
  return model::BakeSdf(data, params, error);
}

void model_utils::GenerateVisualMeshEmbedding(ModelData& data, Error& error) {
  return model::GenerateVisualMeshEmbedding(data, error);
}

void model_utils::FlipWindingOrder(MeshData& data, Error& error) {
  return model::FlipWindingOrder(data, error);
}

void model_utils::FlipWindingOrder(ModelData& data, Error& error) {
  return model::FlipWindingOrder(data, error);
}
