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

#include "model_utils_hdf5.h"

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/group_rw.h>
#include <mochi_core/utils/hdf5_utils.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/string_utils.h>
#include <simple_reflection/simple_reflection.h>

#include <filesystem>

using namespace mochi;

#if MOCHI_USE_HDF5

static void LoadSkinningData(GroupReader& reader, SkinningData& outData, Error& error) {
  MOCHI_ERROR_RETURN(error);
  outData = {};

  NdArray<size_t, 2> weightsDims = {};
  NdArray<size_t, 2> indicesDims = {};
  reader.ReadDataSet("weights", outData.weights, weightsDims, error);
  reader.ReadDataSet("indices", outData.indices, indicesDims, error);
  MOCHI_ERROR_RETURN(error);

  outData.weightsPerNode = static_cast<int>(weightsDims[1]);
  MOCHI_ERROR_IF(
      weightsDims != indicesDims, error, "Skinning weights and indices should be the same size");
}

static void SaveSkinningData(GroupWriter& writer, SkinningDataView const& data, Error& error) {
  MOCHI_ERROR_IF(
      data.weightsPerNode <= 0, error, "Skinning data must have at least one weight per node.");
  MOCHI_ERROR_RETURN(error);
  auto weightsPerNode = static_cast<size_t>(data.weightsPerNode);
  auto dims = NdArray<size_t, 2>{data.weights.size() / weightsPerNode, weightsPerNode};
  writer.AddDataSet("weights", MakeConstSpan(data.weights), MakeConstSpan(dims), error);
  dims = NdArray<size_t, 2>{data.indices.size() / weightsPerNode, weightsPerNode};
  writer.AddDataSet("indices", MakeConstSpan(data.indices), MakeConstSpan(dims), error);
}

static void LoadMeshData(GroupReader& reader, MeshData& outData, Error& error) {
  MOCHI_ERROR_RETURN(error);
  outData = {};

  NdArray<size_t, 2> dims = {};
  reader.ReadDataSet("coordinates", outData.coordinates, dims, error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(dims[1] != 3, error, "Mesh coordinates should be an Nx3 dataset for N nodes.");

  if (reader.HasDataSet("connectivity")) {
    reader.ReadDataSet("connectivity", outData.connectivity, dims, error);
    outData.nodesPerElement = static_cast<int>(dims[1]);
  } else {
    // It is only legal to omit the connectivity array for a polyline mesh.
    outData.nodesPerElement = 2;
  }

  // Historically, this group has been called "skinning" in the main mesh and "embedding" in the
  // visual mesh. It would be nice to standardize terminology. For now, we accept either.
  char const* skinningGroupName = reader.HasGroup("embedding") ? "embedding" : "skinning";
  if (reader.HasGroup(skinningGroupName)) {
    auto skinningGroup = reader.EnterGroup(skinningGroupName, error);
    outData.skinning.emplace(SkinningData{});
    LoadSkinningData(reader, *outData.skinning, error);
  }
}

static void
SaveMeshData(GroupWriter& writer, MeshDataView const& data, bool isVisualMesh, Error& error) {
  MOCHI_ERROR_IF(
      data.nodesPerElement <= 0, error, "Mesh data must have at least one node per element.");
  MOCHI_ERROR_RETURN(error);
  {
    size_t dims[2] = {data.coordinates.size() / 3, size_t(3)};
    writer.AddDataSet("coordinates", MakeConstSpan(data.coordinates), MakeConstSpan(dims), error);
  }
  if (!data.connectivity.empty()) {
    auto nodesPerElement = static_cast<size_t>(data.nodesPerElement);
    size_t dims[2] = {data.connectivity.size() / nodesPerElement, nodesPerElement};
    writer.AddDataSet("connectivity", MakeConstSpan(data.connectivity), MakeConstSpan(dims), error);
  }
  if (data.skinning) {
    auto group = writer.EnterGroup(isVisualMesh ? "embedding" : "skinning", error);
    SaveSkinningData(writer, *data.skinning, error);
  }
}

static void
LoadBlendingData(GroupReader& reader, DynamicArray<BlendingData>& outDataArray, Error& error) {
  MOCHI_ERROR_RETURN(error);
  outDataArray.clear();

  auto groupNames = reader.GetGroupNames(error);
  outDataArray.resize(groupNames.size());

  for (size_t i = 0; i < groupNames.size(); ++i) {
    auto group = reader.EnterGroup(groupNames[i], error);
    auto& outData = outDataArray[i];
    outData.sourceShape = groupNames[i];

    NdArray<size_t, 2> weightsDims = {};
    reader.ReadDataSet("weights", outData.weights, weightsDims, error);
    NdArray<size_t, 2> indicesDims = {};
    reader.ReadDataSet("indices", outData.indices, indicesDims, error);
    MOCHI_ERROR_IF(
        indicesDims != weightsDims,
        error,
        "Blending dataset dimensions do not match for weights and indices.");
    MOCHI_ERROR_IF(
        weightsDims[1] != 2,
        error,
        "Blending weights and indices datasets should have dimension Nx2 for N nodes.");
  }
}

static void SaveBlendingData(GroupWriter& writer, Span<BlendingDataView const> data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  for (auto const& item : data) {
    auto group = writer.EnterGroup(item.sourceShape, error);
    size_t dims[2] = {item.weights.size() / 2, 2};
    writer.AddDataSet("weights", MakeConstSpan(item.weights), MakeConstSpan(dims), error);
    writer.AddDataSet("indices", MakeConstSpan(item.indices), MakeConstSpan(dims), error);
  }
}

static void LoadGridSdf(GroupReader& reader, GridSdfData& outData, Error& error) {
  MOCHI_ERROR_RETURN(error);
  outData = {};

  NdArray<size_t, 3> dims = {};
  reader.ReadDataSet("grid", outData.values, dims, error);
  MOCHI_ERROR_IF(Max(dims) > INT_MAX, error, "Grid SDF data is too large");
  outData.dims = StaticCast<Int3>(dims);

  // Get attributes of the "grid" dataset
  Real3 min{}, max{};
  reader.ReadAttribute("grid_min", min, error);
  reader.ReadAttribute("grid_max", max, error);
  outData.bounds = Aabb{min, max};
  reader.ReadAttribute("collider_min", min, error);
  reader.ReadAttribute("collider_max", max, error);
  outData.negativeValueBounds = Aabb{min, max};
  if (reader.HasAttribute("scale")) {
    if (reader.GetAttributeSize("scale", error) == 1) {
      // Older files may have stored uniform scale as a single value.
      real scale{};
      reader.ReadAttribute("scale", scale, error);
      outData.scale = Real3{scale, scale, scale};
    } else {
      // Load (possibly non-uniform) scale
      Real3 scale{};
      reader.ReadAttribute("scale", scale, error);
      outData.scale = scale;
    }
  }
  if (reader.HasAttribute("rotation")) {
    Real4 values{};
    reader.ReadAttribute("rotation", MakeSpan(values), error);
    outData.rotation = Quaternion{values};
  }
  if (reader.HasAttribute("translation")) {
    Real3 translation{};
    reader.ReadAttribute("translation", MakeSpan(translation), error);
    outData.translation = translation;
  }
}

static void SaveGridSdf(GroupWriter& writer, GridSdfDataView const& data, Error& error) {
  MOCHI_ERROR_IF(
      data.dims[0] <= 0 || data.dims[1] <= 0 || data.dims[2] <= 0,
      error,
      "Invalid SDF grid dimensions");
  MOCHI_ERROR_RETURN(error);
  auto gridDims = StaticCast<NdArray<size_t, 3>>(data.dims);
  writer.AddDataSet("grid", MakeConstSpan(data.values), MakeConstSpan(gridDims), error);
  writer.AddAttribute("grid_min", MakeConstSpan(data.bounds.GetMin()), error);
  writer.AddAttribute("grid_max", MakeConstSpan(data.bounds.GetMax()), error);
  writer.AddAttribute("collider_min", MakeConstSpan(data.negativeValueBounds.GetMin()), error);
  writer.AddAttribute("collider_max", MakeConstSpan(data.negativeValueBounds.GetMax()), error);
  if (data.scale) {
    writer.AddAttribute("scale", *data.scale, error);
  }
  if (data.rotation) {
    writer.AddAttribute("rotation", MakeConstSpan(data.rotation->ToReal4()), error);
  }
  if (data.translation) {
    writer.AddAttribute("translation", MakeConstSpan(*data.translation), error);
  }
}

static void
LoadSoftMaterialData(GroupReader& reader, PerElementSoftMaterialData& outData, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      reader.HasAttribute("type") || reader.HasAttribute("material_type"),
      error,
      "Missing material type.");
  MOCHI_ERROR_IF(
      reader.HasAttribute("type") && reader.HasAttribute("material_type"),
      error,
      "Both 'type' and 'material_type' found. Exactly one must be provided.");
  MOCHI_ERROR_IF(
      reader.HasDataSet("arap_stiffness") && reader.HasDataSet("mu"),
      error,
      "Both 'arap_stiffness' and 'mu' found. At most one may be provided.");
  MOCHI_ERROR_IF(
      reader.HasDataSet("shape_target_tensor") && reader.HasDataSet("shape_targets"),
      error,
      "Both 'shape_target_tensor' and 'shape_targets' found. At most one may be provided.");
  MOCHI_ERROR_RETURN(error);

  outData = {};

  // Read material type.
  {
    std::string_view const attrName = reader.HasAttribute("type") ? "type" : "material_type";
    if (reader.HasAttribute("material_type")) {
      // Backwards compatibility.
      MOCHI_LOG_WARNING(
          "HDF5 attribute ['material']['material_type'] is deprecated. Support will be removed in a future release. "
          "Please rename to ['material']['type'] or use model::SaveToFile to re-export the HDF5 file with the new schema.");
    }
    std::string typeStr;
    Error strReadError;
    reader.ReadAttribute(attrName, typeStr, strReadError);
    if (strReadError.IsOK()) {
      auto const* item = SReflect::GetTypeInfo<SoftMaterialType>().FindItemByName(typeStr);
      MOCHI_ERROR_IF_NOT(item, error, "Unknown SoftMaterialType in HDF5 file.");
      MOCHI_ERROR_RETURN(error);
      outData.type = static_cast<SoftMaterialType>(item->_value);
    } else {
      // Backwards compatibility: integer-based encoding.
      int iType{};
      reader.ReadAttribute(attrName, iType, error);
      MOCHI_ERROR_RETURN(error);
      MOCHI_LOG_WARNING(
          "Integer-based encoding for HDF5 attribute ['material']['%s'] is deprecated. Support will be removed in a future release. "
          "Please use model::SaveToFile to re-export the HDF5 file with string-based encoding.",
          std::string(attrName).c_str());
      outData.type = static_cast<SoftMaterialType>(iType);
    }
  }

  // Read PSD strategy.
  if (reader.HasAttribute("psd_strategy")) { // Not present in most files
    std::string psdStr;
    Error strReadError;
    reader.ReadAttribute("psd_strategy", psdStr, strReadError);
    if (strReadError.IsOK()) {
      auto const* item = SReflect::GetTypeInfo<MaterialPsdStrategy>().FindItemByName(psdStr);
      MOCHI_ERROR_IF_NOT(item, error, "Unknown MaterialPsdStrategy in HDF5 file.");
      MOCHI_ERROR_RETURN(error);
      outData.psdStrategy = static_cast<MaterialPsdStrategy>(item->_value);
    } else {
      MOCHI_ERROR_SET(
          error,
          "Integer-based encoding for HDF5 attribute 'psd_strategy' is no longer supported. Please update the HDF5 file to use string-based encoding.");
      MOCHI_ERROR_RETURN(error);
    }
  }

  if (reader.HasDataSet("youngs_modulus")) {
    reader.ReadDataSet("youngs_modulus", outData.youngsModulus, error);
  }
  if (reader.HasDataSet("poisson_ratio")) {
    reader.ReadDataSet("poisson_ratio", outData.poissonRatio, error);
  }
  if (reader.HasDataSet("aniso_alpha")) {
    reader.ReadDataSet("aniso_alpha", outData.anisoAlpha, error);
  }
  if (reader.HasDataSet("aniso_length")) {
    reader.ReadDataSet("aniso_length", outData.anisoLength, error);
  }
  if (reader.HasDataSet("aniso_theta")) {
    reader.ReadDataSet("aniso_theta", outData.anisoTheta, error);
  }
  if (reader.HasDataSet("aniso_phi")) {
    reader.ReadDataSet("aniso_phi", outData.anisoPhi, error);
  }
  if (reader.HasDataSet("arap_stiffness")) {
    reader.ReadDataSet("arap_stiffness", outData.arapStiffness, error);
  } else if (reader.HasDataSet("mu")) {
    // For backwards compatibility.
    MOCHI_LOG_WARNING(
        "HDF5 dataset 'mu' for ARAP stiffness is deprecated. Support will be removed in a future release. "
        "Please rename to 'arap_stiffness' or use model::SaveToFile to re-export the HDF5 file with the new schema.");
    reader.ReadDataSet("mu", outData.arapStiffness, error);
  }
  if (reader.HasDataSet("shape_target_tensor")) {
    NdArray<size_t, 2> dims{};
    reader.ReadDataSet("shape_target_tensor", outData.shapeTargetTensor, dims, error);
    MOCHI_ERROR_IF(dims[1] != 6, error, "shape_target_tensor has incorrect number of columns.");
  } else if (reader.HasDataSet("shape_targets")) {
    // For backwards compatibility.
    MOCHI_LOG_WARNING(
        "HDF5 dataset 'shape_targets' for shape target tensor is deprecated. Support will be removed in a future release. "
        "Please rename to 'shape_target_tensor' or use model::SaveToFile to re-export the HDF5 file with the new schema.");
    NdArray<size_t, 2> dims{};
    reader.ReadDataSet("shape_targets", outData.shapeTargetTensor, dims, error);
    MOCHI_ERROR_IF(dims[1] != 6, error, "shape_targets has incorrect number of columns.");
  }
}

static void SaveSoftMaterialData(
    GroupWriter& writer,
    PerElementSoftMaterialDataView const& data,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  char const* typeStr = SReflect::EnumToString(data.type);
  MOCHI_ASSERT(typeStr && typeStr[0] != '\0', "Unknown SoftMaterialType enum value.");
  writer.AddAttribute("type", typeStr, error);

  char const* psdStr = SReflect::EnumToString(data.psdStrategy);
  MOCHI_ASSERT(psdStr && psdStr[0] != '\0', "Unknown MaterialPsdStrategy enum value.");
  writer.AddAttribute("psd_strategy", psdStr, error);

  if (!data.youngsModulus.empty()) {
    size_t dims[1] = {data.youngsModulus.size()};
    writer.AddDataSet(
        "youngs_modulus", MakeConstSpan(data.youngsModulus), MakeConstSpan(dims), error);
  }
  if (!data.poissonRatio.empty()) {
    size_t dims[1] = {data.poissonRatio.size()};
    writer.AddDataSet(
        "poisson_ratio", MakeConstSpan(data.poissonRatio), MakeConstSpan(dims), error);
  }
  if (!data.anisoAlpha.empty()) {
    size_t dims[1] = {data.anisoAlpha.size()};
    writer.AddDataSet("aniso_alpha", MakeConstSpan(data.anisoAlpha), MakeConstSpan(dims), error);
  }
  if (!data.anisoLength.empty()) {
    size_t dims[1] = {data.anisoLength.size()};
    writer.AddDataSet("aniso_length", MakeConstSpan(data.anisoLength), MakeConstSpan(dims), error);
  }
  if (!data.anisoTheta.empty()) {
    size_t dims[1] = {data.anisoTheta.size()};
    writer.AddDataSet("aniso_theta", MakeConstSpan(data.anisoTheta), MakeConstSpan(dims), error);
  }
  if (!data.anisoPhi.empty()) {
    size_t dims[1] = {data.anisoPhi.size()};
    writer.AddDataSet("aniso_phi", MakeConstSpan(data.anisoPhi), MakeConstSpan(dims), error);
  }
  if (!data.arapStiffness.empty()) {
    size_t dims[1] = {data.arapStiffness.size()};
    writer.AddDataSet(
        "arap_stiffness", MakeConstSpan(data.arapStiffness), MakeConstSpan(dims), error);
  }
  if (!data.shapeTargetTensor.empty()) {
    size_t dims[2] = {data.shapeTargetTensor.size() / 6, size_t(6)};
    writer.AddDataSet(
        "shape_target_tensor", MakeConstSpan(data.shapeTargetTensor), MakeConstSpan(dims), error);
  }
}

static void LoadBoxData(GroupReader& reader, Box& outData, Error& error) {
  MOCHI_ERROR_RETURN(error);
  reader.ReadAttribute("center", MakeSpan(outData.center), error);
  reader.ReadAttribute("half_extents", MakeSpan(outData.halfExtents), error);
  Real4 rotation{};
  reader.ReadAttribute("rotation", MakeSpan(rotation), error);
  outData.rotation = Quaternion{rotation};
}

static void SaveBoxData(GroupWriter& writer, Box const& data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  writer.AddAttribute("center", MakeConstSpan(data.center), error);
  writer.AddAttribute("half_extents", MakeConstSpan(data.halfExtents), error);
  writer.AddAttribute("rotation", MakeConstSpan(data.rotation.ToReal4()), error);
}

static void LoadSphereData(GroupReader& reader, Sphere& outData, Error& error) {
  MOCHI_ERROR_RETURN(error);
  Real3 center{};
  reader.ReadAttribute("center", MakeSpan(center), error);
  real radius{};
  reader.ReadAttribute("radius", radius, error);
  outData = Sphere{center, radius};
}

static void SaveSphereData(GroupWriter& writer, Sphere const& data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  writer.AddAttribute("center", MakeConstSpan(data.GetCenter()), error);
  writer.AddAttribute("radius", data.GetRadius(), error);
}

static void LoadPlaneData(GroupReader& reader, Plane& outData, Error& error) {
  MOCHI_ERROR_RETURN(error);
  Real3 normal{};
  reader.ReadAttribute("normal", MakeSpan(normal), error);
  real distance{};
  reader.ReadAttribute("distance", distance, error);
  outData = Plane{normal, distance};
}

static void SavePlaneData(GroupWriter& writer, Plane const& data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  writer.AddAttribute("normal", MakeConstSpan(data.GetNormal()), error);
  writer.AddAttribute("distance", data.GetDistanceFromOrigin(), error);
}

// Detect group names that are known to have experimental data not representable via the ModelData
// structure. Optionally return their names.
static bool HasExperimentalDataGroups(
    GroupReader& reader,
    DynamicArray<std::string>* outGroupNames = nullptr) {
  constexpr std::string_view kExperimentalGroupNames[] = {"rom", "sample_meshes", "bsh"};
  bool hasExperimentalData = false;
  for (auto name : kExperimentalGroupNames) {
    if (reader.HasGroup(name)) {
      hasExperimentalData = true;
      if (outGroupNames) {
        outGroupNames->push_back(std::string(name));
      }
    }
  }
  return hasExperimentalData;
}

static void LoadModelData(GroupReader& reader, ModelData& outData, Error& error) {
  MOCHI_ERROR_RETURN(error);
  outData = {};

  if (reader.HasGroup("mesh")) {
    auto meshGroup = reader.EnterGroup("mesh", error);
    outData.mesh.emplace(MeshData{});
    LoadMeshData(reader, *outData.mesh, error);
    MOCHI_ERROR_RETURN(error);

    if (reader.HasGroup("blending")) {
      auto blendingGroup = reader.EnterGroup("blending", error);
      outData.blending.emplace(DynamicArray<BlendingData>{});
      LoadBlendingData(reader, *outData.blending, error);
      MOCHI_ERROR_RETURN(error);
    }

    if (reader.HasDataSet("constrained_nodes")) {
      outData.constrainedNodes.emplace(DynamicArray<int>{});
      reader.ReadDataSet("constrained_nodes", *outData.constrainedNodes, error);
      MOCHI_ERROR_RETURN(error);
    }

    if (reader.HasDataSet("element_frame_axes")) {
      outData.elementFrameAxes.emplace(DynamicArray<real>{});
      NdArray<size_t, 2> dims{};
      reader.ReadDataSet("element_frame_axes", *outData.elementFrameAxes, dims, error);
      MOCHI_ERROR_RETURN(error);
      MOCHI_ERROR_IF(
          dims[1] != 3,
          error,
          "The element_frame_axes dataset should have dimensions Nx3 for N elements.");
    }
  }

  if (reader.HasGroup("visual_mesh")) {
    auto group = reader.EnterGroup("visual_mesh", error);

    // A number of our assets have an empty group called "visual_mesh". These should be ignored.
    // If a non-empty visual_mesh fails to load, then we should log about it.
    bool isEmptyGroup = reader.GetDataSetNames(error).empty();
    if (!isEmptyGroup) {
      outData.visualMesh.emplace(MeshData{});
      LoadMeshData(reader, *outData.visualMesh, error);
      MOCHI_ERROR_RETURN(error);
    }
  }

  if (reader.HasGroup("sdf")) {
    auto group = reader.EnterGroup("sdf", error);
    outData.sdf.emplace(GridSdfData{});
    LoadGridSdf(reader, *outData.sdf, error);
    MOCHI_ERROR_RETURN(error);
  }

  if (reader.HasGroup("material")) {
    auto group = reader.EnterGroup("material", error);
    outData.material.emplace(PerElementSoftMaterialData{});
    LoadSoftMaterialData(reader, *outData.material, error);
    MOCHI_ERROR_RETURN(error);
  }

  if (reader.HasGroup("box")) {
    auto group = reader.EnterGroup("box", error);
    outData.box.emplace(Box{});
    LoadBoxData(reader, *outData.box, error);
    MOCHI_ERROR_RETURN(error);
  }

  if (reader.HasGroup("sphere")) {
    auto group = reader.EnterGroup("sphere", error);
    outData.sphere.emplace(Sphere{});
    LoadSphereData(reader, *outData.sphere, error);
    MOCHI_ERROR_RETURN(error);
  }

  if (reader.HasGroup("plane")) {
    auto group = reader.EnterGroup("plane", error);
    outData.plane.emplace(Plane{});
    LoadPlaneData(reader, *outData.plane, error);
    MOCHI_ERROR_RETURN(error);
  }

  outData.experimentalDataDetected = HasExperimentalDataGroups(reader);
}

static void SaveModelData(GroupWriter& writer, ModelDataView const& data, Error& error) {
  MOCHI_ERROR_RETURN(error);

  if (data.mesh) {
    auto group = writer.EnterGroup("mesh", error);
    SaveMeshData(writer, *data.mesh, /*isVisualMesh*/ false, error);
    MOCHI_ERROR_RETURN(error);
    if (data.blending && !data.blending->empty()) {
      auto blendingGroup = writer.EnterGroup("blending", error);
      SaveBlendingData(writer, *data.blending, error);
      MOCHI_ERROR_RETURN(error);
    }
    if (data.constrainedNodes && !data.constrainedNodes->empty()) {
      size_t dims[1] = {data.constrainedNodes->size()};
      writer.AddDataSet("constrained_nodes", *data.constrainedNodes, dims, error);
      MOCHI_ERROR_RETURN(error);
    }
    if (data.elementFrameAxes && !data.elementFrameAxes->empty()) {
      size_t dims[2] = {data.elementFrameAxes->size() / size_t(3), size_t(3)};
      writer.AddDataSet("element_frame_axes", *data.elementFrameAxes, dims, error);
      MOCHI_ERROR_RETURN(error);
    }
  }

  if (data.visualMesh) {
    auto group = writer.EnterGroup("visual_mesh", error);
    SaveMeshData(writer, *data.visualMesh, /*isVisualMesh*/ true, error);
    MOCHI_ERROR_RETURN(error);
  }

  if (data.sdf) {
    auto group = writer.EnterGroup("sdf", error);
    SaveGridSdf(writer, *data.sdf, error);
    MOCHI_ERROR_RETURN(error);
  }

  if (data.material) {
    auto group = writer.EnterGroup("material", error);
    SaveSoftMaterialData(writer, *data.material, error);
    MOCHI_ERROR_RETURN(error);
  }

  if (data.box) {
    auto group = writer.EnterGroup("box", error);
    SaveBoxData(writer, *data.box, error);
    MOCHI_ERROR_RETURN(error);
  }

  if (data.sphere) {
    auto group = writer.EnterGroup("sphere", error);
    SaveSphereData(writer, *data.sphere, error);
    MOCHI_ERROR_RETURN(error);
  }

  if (data.plane) {
    auto group = writer.EnterGroup("plane", error);
    SavePlaneData(writer, *data.plane, error);
    MOCHI_ERROR_RETURN(error);
  }
}

#endif // #if MOCHI_USE_HDF5

void mochi::model::LoadFromFileUncheckedH5(
    [[maybe_unused]] ModelData& outData,
    [[maybe_unused]] std::string_view filePath,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

#if MOCHI_USE_HDF5
  std::lock_guard lock{hdf5::GetGlobalMutex()};
  auto reader = CreateGroupReaderHDF5(filePath, error);
  MOCHI_ERROR_RETURN(error);
  LoadModelData(*reader, outData, error);
#else
  MOCHI_ERROR_SET(
      error,
      "Mochi was not built with HDF5 support. To use this file format, you will need to update "
      "your build system to link the HDF5 library and define MOCHI_USE_HDF5 to 1.");
#endif
}

void mochi::model::LoadFromBytesUncheckedH5(
    [[maybe_unused]] ModelData& outData,
    [[maybe_unused]] Span<char const> fileData,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
#if MOCHI_USE_HDF5
  std::lock_guard lock{hdf5::GetGlobalMutex()};
  auto reader = CreateGroupReaderFromBytesHDF5(fileData, error);
  MOCHI_ERROR_RETURN(error);
  LoadModelData(*reader, outData, error);
#else
  MOCHI_ERROR_SET(
      error,
      "Mochi was not built with HDF5 support. To use this file format, you will need to update "
      "your build system to link the HDF5 library and define MOCHI_USE_HDF5 to 1.");
#endif
}

void mochi::model::SaveToFileH5(
    [[maybe_unused]] ModelDataView const& data,
    [[maybe_unused]] std::string_view filePath,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
#if MOCHI_USE_HDF5
  std::lock_guard lock{hdf5::GetGlobalMutex()};
  std::unique_ptr<GroupWriter> writer = CreateGroupWriterHDF5(filePath, error);
  MOCHI_ERROR_RETURN(error);
  SaveModelData(*writer, data, error);
#else
  MOCHI_ERROR_SET(
      error,
      "Mochi was not built with HDF5 support. To use this file format, you will need to update "
      "your build system to link the HDF5 library and define MOCHI_USE_HDF5 to 1.");
#endif
}

void mochi::model::WarnIfOverwritingExperimentalDataH5([[maybe_unused]] std::string_view filePath) {
#if MOCHI_USE_HDF5
  if (!std::filesystem::exists(filePath)) {
    // We are not overwriting anything. That's OK.
    return;
  }

  std::lock_guard lock{hdf5::GetGlobalMutex()};

  // Temporarily suppress HDF5's automatic logging of exception callstacks.
  H5E_auto2_t oldPrintFunc{};
  void* oldClientData{};
  H5::Exception::getAutoPrint(oldPrintFunc, &oldClientData);
  H5::Exception::dontPrint();
  MOCHI_DEFER(H5::Exception::setAutoPrint(oldPrintFunc, oldClientData)); // Restore

  // Temporarily suppress Mochi's error logging because CreateGroupReaderHDF5 always logs the
  // description string from an H5 exception.
  bool wasLogErrorEnabled = IsLogChannelEnabled(LogChannel::Error);
  EnableLogChannel(LogChannel::Error, false);
  MOCHI_DEFER(EnableLogChannel(LogChannel::Error, wasLogErrorEnabled));

  // Try to read the file that we are about to overwrite.
  Error error;
  auto reader = CreateGroupReaderHDF5(filePath, error);
  if (!error.IsOK()) {
    // The file is unreadable. We have to assume it does not contain useful experimental data.
    return; // Return silently.
  }

  // Check for known group names.
  DynamicArray<std::string> groupNames;
  if (HasExperimentalDataGroups(*reader, &groupNames)) {
    MOCHI_LOG_WARNING(
        "Model file \"%s\" is being overwritten in a way that will lose data. The file previously contained the "
        "following data types, which will not be preserved because they are not supported by the ModelData struct: %s. "
        "If this was a mistake, then revert the change.",
        std::string(filePath).c_str(),
        Join(groupNames, ", ").c_str());
  }
#endif // MOCHI_USE_HDF5
}
