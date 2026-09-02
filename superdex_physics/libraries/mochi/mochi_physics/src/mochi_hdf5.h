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

#include "mochi_shape.h"

#include <mochi_core/ai/mlp.h>
#include <mochi_core/articulated_body/articulated_body_params.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/utils/transform_srt.h>
#include <mochi_physics/mochi_physics_experimental.h>

#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <tuple>
#include <unordered_map>

namespace mochi {

// This data can be loaded from HDF5 format, but is not currently supported by the ModelData struct
// from mochi_core.
struct ExperimentalModelData {
  std::unordered_map<std::string, RomData> romData;
  std::unordered_map<std::string, SampleMeshInfo> sampleMeshes;
  std::unordered_map<std::string, ContactSamplesBsh> bshs;
};

} // namespace mochi

namespace mochi::hdf5 {

#if MOCHI_USE_HDF5
// You can load experimental data from an HDF5 file after the main ModelData has been loaded.
ExperimentalModelData LoadExperimentalModelDataFromFile(
    std::string_view filePath,
    ModelData const& model,
    Real3 const& bakeScale,
    TransformRT const& bakeTransform,
    Error& error);

// You can load experimental data from an HDF5 file after the main ModelData has been loaded.
ExperimentalModelData LoadExperimentalModelDataFromBytes(
    Span<char const> fileData,
    ModelData const& model,
    Real3 const& bakeScale,
    TransformRT const& bakeTransform,
    Error& error);
#endif // MOCHI_USE_HDF5

constexpr std::size_t kMaxMeshTransformsNameLength = 64;

void ReadMeshTransformsBytes(
    Span<char const> fileData,
    Span<Quaternion> outRotations,
    Span<Real3> outTranslations,
    Span<int> outLinkParents,
    Span<ArticulatedJointType> outJointTypes,
    Span<ArticulatedCycleJoint> outCycleJoints,
    Span<Real3> outJointAxes,
    Span<Real3> outJointXYZs,
    bool& outHasJointLimits,
    Span<Real3> outJointMinLimits,
    Span<Real3> outJointMaxLimits,
    bool& outHasJointNames,
    Span<std::array<char, kMaxMeshTransformsNameLength>> outJointNames,
    bool& outHasLinkNames,
    Span<std::array<char, kMaxMeshTransformsNameLength>> outLinkNames,
    Error& error);

int ReadMeshTransformsBytesBodyCount(Span<char const> fileData, Error& error);
int ReadMeshTransformsBytesJointCount(Span<char const> fileData, Error& error);

std::optional<std::tuple<ai::Mlp<real>, ColumnVector<real>>> LoadCromDecoderModel(
    std::string_view filePath,
    std::string_view groupName,
    real scale,
    Error& error);

} // namespace mochi::hdf5
