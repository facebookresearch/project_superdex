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
#include "model_utils_meshio.h"

#include <mochi_core/geometry/bvh_tree.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/grid_sdf.h>
#include <mochi_core/geometry/mesh_data_utils.h>
#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/materials/material_params_utils.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/hdf5_utils.h>
#include <mochi_core/utils/json_utils.h> // For loading deprecated files
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/quaternion_utils.h>
#include <mochi_core/utils/string_utils.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_core/utils/transform_srt.h>
#include <mochi_core/utils/vmatrix.h>

#include <algorithm>
#include <filesystem>
#include <memory>

using namespace mochi;
using namespace mochi::model;

namespace {

struct BlendingDataDeprecated {
  std::string bone_subset;
  DynamicArray<real> weights;
  DynamicArray<int> indices;

  MOCHI_STRUCT_BEGIN(BlendingDataDeprecated);
  MOCHI_FIELD(bone_subset);
  MOCHI_FIELD(weights);
  MOCHI_FIELD(indices);
  MOCHI_STRUCT_END();
};

// This struct matches the deprecated ".mochi.json" model format.
struct JsonModelDataDeprecated {
  int element_size = 4;
  DynamicArray<real> coordinates;
  DynamicArray<int> connectivity;

  int visual_element_size = 3;
  DynamicArray<real> visual_coordinates;
  DynamicArray<int> visual_connectivity;
  DynamicArray<real> embedding_weights;
  DynamicArray<int> embedding_indices;

  int num_skin_nodes = 0;
  DynamicArray<real> weights;
  DynamicArray<int> indices;

  DynamicArray<BlendingDataDeprecated> blending_data;

  int num_constrained_nodes = 0;
  DynamicArray<int> constrained_nodes;

  MOCHI_STRUCT_BEGIN(JsonModelDataDeprecated);
  MOCHI_FIELD(element_size);
  MOCHI_FIELD(coordinates);
  MOCHI_FIELD(connectivity);
  MOCHI_FIELD(visual_element_size);
  MOCHI_FIELD(visual_coordinates);
  MOCHI_FIELD(visual_connectivity);
  MOCHI_FIELD(embedding_weights);
  MOCHI_FIELD(embedding_indices);
  MOCHI_FIELD(num_skin_nodes);
  MOCHI_FIELD(weights);
  MOCHI_FIELD(indices);
  MOCHI_FIELD(blending_data);
  MOCHI_FIELD(num_constrained_nodes);
  MOCHI_FIELD(constrained_nodes);
  MOCHI_STRUCT_END();
};

} // namespace

static void LoadJsonDeprecated(ModelData& outData, picojson::value const& json, Error& error) {
  MOCHI_ERROR_RETURN(error);

  int numIssues = 0;
  JsonModelDataDeprecated deprecated;
  SReflect::FromJsonValue(deprecated, json, {}, numIssues);
  MOCHI_ERROR_IF(numIssues, error, "Failed to parse JSON model file");

  // Check a few things up front
  MOCHI_ERROR_IF(
      deprecated.element_size != 3 && deprecated.element_size != 4, error, "Invalid element size.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      deprecated.connectivity.empty() || deprecated.coordinates.empty(),
      error,
      "Connectivity and coordinates arrays must not be empty.");
  MOCHI_ERROR_IF(
      isize(deprecated.coordinates) % 3 != 0,
      error,
      "Coordinates array must have 3 values per node.");
  MOCHI_ERROR_IF(
      isize(deprecated.connectivity) % deprecated.element_size != 0,
      error,
      "Connectivity array length must be a multiple of the element size.");
  MOCHI_ERROR_RETURN(error);

  // Transform data to the new format
  outData = {};
  outData.mesh.emplace(MeshData{});
  outData.mesh->nodesPerElement = deprecated.element_size;
  outData.mesh->coordinates = std::move(deprecated.coordinates);
  outData.mesh->connectivity = std::move(deprecated.connectivity);
  if (!deprecated.weights.empty()) {
    outData.mesh->skinning.emplace(SkinningData{});
    outData.mesh->skinning->weightsPerNode =
        isize(deprecated.indices) / outData.mesh->GetNumNodes();
    outData.mesh->skinning->indices = std::move(deprecated.indices);
    outData.mesh->skinning->weights = std::move(deprecated.weights);
  }
  if (!deprecated.blending_data.empty()) {
    outData.blending.emplace(DynamicArray<BlendingData>(deprecated.blending_data.size()));
    for (int i = 0; i < isize(deprecated.blending_data); ++i) {
      auto& inBlend = deprecated.blending_data[i];
      auto& outBlend = (*outData.blending)[i];
      outBlend.sourceShape = DynamicString{inBlend.bone_subset};
      outBlend.indices = std::move(inBlend.indices);
      outBlend.weights = std::move(inBlend.weights);
    }
  }
  if (!deprecated.constrained_nodes.empty()) {
    outData.constrainedNodes = std::move(deprecated.constrained_nodes);
  }
  if (!deprecated.visual_coordinates.empty()) {
    outData.visualMesh.emplace(MeshData{});
    outData.visualMesh->nodesPerElement = 3;
    outData.visualMesh->coordinates = std::move(deprecated.visual_coordinates);
    outData.visualMesh->connectivity = std::move(deprecated.visual_connectivity);
    outData.visualMesh->skinning.emplace(SkinningData{});
    outData.visualMesh->skinning->weightsPerNode = 4;
    outData.visualMesh->skinning->weights = std::move(deprecated.embedding_weights);
    outData.visualMesh->skinning->indices = std::move(deprecated.embedding_indices);
  }
}

static void LoadFromJsonImpl(ModelData& outData, picojson::value const& jsonValue, Error& error) {
  MOCHI_ERROR_RETURN(error);
  outData = {};
  if (jsonValue.contains("coordinates")) {
    // Backwards compatibility for old JSON model files.
    LoadJsonDeprecated(outData, jsonValue, error);
  } else {
    int numIssues = 0;
    SReflect::FromJsonValue(outData, jsonValue, {}, numIssues);
    MOCHI_ERROR_IF(numIssues != 0, error, "Failed to load model file.");
  }
}

ModelData mochi::model::LoadFromFileUnchecked(std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  ModelData outData;
  if (path.ends_with(".h5")) {
    LoadFromFileUncheckedH5(outData, path, error);
  } else if (EndsWithCaseInsensitive(path, ".obj")) {
    LoadObjFromFile(outData, path, error);
  } else if (EndsWithCaseInsensitive(path, ".stl")) {
    LoadStlFromFile(outData, path, error);
  } else if (EndsWithCaseInsensitive(path, ".ply")) {
    LoadPlyFromFile(outData, path, error);
  } else if (EndsWithCaseInsensitive(path, ".off")) {
    LoadOffFromFile(outData, path, error);
  } else {
    LoadFromJsonImpl(outData, ParseJsonFromFile(path, error), error);
  }
  return outData;
}

ModelData mochi::model::LoadFromFile(std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  ModelData outData = LoadFromFileUnchecked(path, error);
  AutoCorrect(outData, error);
  Validate(outData, error);
  return outData;
}

ModelData
mochi::model::LoadFromBytesUnchecked(Span<char const> data, MeshFileType format, Error& error) {
  MOCHI_ERROR_IF(data.empty(), error, "Zero byte file data");
  MOCHI_ERROR_RETURN(error, {});
  ModelData outData;
  switch (format) {
    case MeshFileType::Legacy: {
      if (hdf5::LooksLikeHDF5(data)) {
        LoadFromBytesUncheckedH5(outData, data, error);
      } else {
        LoadFromJsonImpl(outData, ParseJsonFromString(data, error), error);
      }
    } break;
    case MeshFileType::OBJ:
      LoadObjFromBytes(outData, data, error);
      break;
    case MeshFileType::STL:
      LoadStlFromBytes(outData, data, error);
      break;
    case MeshFileType::PLY:
      LoadPlyFromBytes(outData, data, error);
      break;
    case MeshFileType::OFF:
      LoadOffFromBytes(outData, data, error);
      break;
    default:
      MOCHI_ERROR_SET(error, "Invalid MeshFileType");
      break;
  }
  return outData;
}

ModelData mochi::model::LoadFromBytes(Span<char const> data, MeshFileType format, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  ModelData outData = LoadFromBytesUnchecked(data, format, error);
  AutoCorrect(outData, error);
  Validate(outData, error);
  return outData;
}

// Save a ModelData or ModelDataView to a file
template <class ModelDataT>
static void
SaveToFileImpl(ModelDataT const& data, std::string_view path, FileFormat format, Error& error) {
  MOCHI_ERROR_RETURN(error);

  // Create the directory path if necessary
  auto parentDir = std::filesystem::path(path).parent_path();
  if (!parentDir.empty() && !std::filesystem::exists(parentDir)) {
    std::error_code ec;
    std::filesystem::create_directories(parentDir, ec);
    MOCHI_ERROR_IF(ec, error, "Failed to create directory path for model file.");
    MOCHI_ERROR_RETURN(error);
  }

  switch (format) {
    case FileFormat::JSON: {
      bool success = false;
      if constexpr (std::is_same_v<ModelDataT, ModelDataView>) {
        // Copy to a temporary ModelData object because ModelDataView does not support reflection.
        success = SReflect::SaveToJsonFile(ModelData{data}, std::string(path).c_str());
      } else {
        success = SReflect::SaveToJsonFile(data, std::string(path).c_str());
      }
      MOCHI_ERROR_IF(!success, error, "Failed to save model file.");
    } break;
    case FileFormat::H5: {
      model::WarnIfOverwritingExperimentalDataH5(path);
      model::SaveToFileH5(data, path, error);
    } break;
    default:
      MOCHI_ERROR_SET(error, "Unknown file format");
      break;
  }
}

void mochi::model::SaveToFile(
    ModelData const& data,
    std::string_view path,
    FileFormat format,
    Error& error) {
  return SaveToFileImpl(data, path, format, error);
}

void mochi::model::SaveToFile(
    ModelDataView const& data,
    std::string_view path,
    FileFormat format,
    Error& error) {
  return SaveToFileImpl(data, path, format, error);
}

DynamicString mochi::model::SaveToJsonString(ModelData const& data, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  auto json = SReflect::ToJsonString(data, /*pretty*/ true);
  return DynamicString{json.c_str(), json.size()};
}

static void
ValidateSkinning(SkinningDataView const& data, int numNodes, int maxSkinningIndex, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      data.weightsPerNode < 1,
      error,
      "Mesh skinning must provide at least one weight and one index per node");
  MOCHI_ERROR_IF(
      data.weights.size() >= INT_MAX,
      error,
      "Mesh skinning weights array is larger than what Mochi currently supports.");
  MOCHI_ERROR_IF(
      isize(data.weights) != data.weightsPerNode * numNodes,
      error,
      "Mesh skinning weights array length is incorrect. Must be the number of nodes times the number of values per node.");
  MOCHI_ERROR_IF(
      data.indices.size() != data.weights.size(),
      error,
      "Mesh skinning indices array must be the same length as the weights array.");
  MOCHI_ERROR_IF(
      !IsFinite(MakeConstSpan(data.weights)),
      error,
      "Mesh skinning weights contains one or more non-finite values.");
  MOCHI_ERROR_RETURN(error);
  auto const [minIdx, maxIdx] = MinMax(MakeConstSpan(data.indices));
  MOCHI_ERROR_IF(
      (minIdx < 0) || (maxIdx > maxSkinningIndex),
      error,
      "Mesh skinning contains one or more out-of-bounds indices.");
  MOCHI_ERROR_RETURN(error);

  // Weights should add to 1.0 for each node (approximately).
  real constexpr kTolerance = 1e-5_r;
  for (int i = 0; i < numNodes; ++i) {
    real sum = 0_r;
    for (int j = 0; j < data.weightsPerNode; ++j) {
      sum += data.weights[i * data.weightsPerNode + j];
    }
    MOCHI_ERROR_IF(
        !NearEqual(1_r, sum, kTolerance), error, "Mesh skinning weights should add to one.");
  }
}

static void ValidateBlending(BlendingDataView const& data, int numNodes, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      data.sourceShape.empty(), error, "Mesh blending must provide a source shape name.");
  MOCHI_ERROR_IF(
      data.weights.size() != 2 * size_t(numNodes),
      error,
      "Mesh blending weights array length is incorrect. Expected 2 values per node.");
  MOCHI_ERROR_IF(
      !IsFinite(MakeConstSpan(data.weights)),
      error,
      "Mesh blending weights array contains non-finite values.");
  MOCHI_ERROR_IF(
      data.indices.size() != 2 * size_t(numNodes),
      error,
      "Mesh blending indices array length is incorrect. Expected 2 values per node.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      Min(MakeConstSpan(data.indices)) < 0, error, "Mesh blending indices cannot be negative.");
  // Index values can't be validated yet. We don't know the other mesh.
}

void mochi::model::ValidatePolylineGeometry(
    Span<Real3 const> nodes,
    bool isClosedLoop,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  int const numNodes = isize(nodes);
  MOCHI_ERROR_IF(numNodes < 2, error, "Polyline must have at least 2 nodes");
  // It is topologically-possible to form a loop with just two nodes, but it would necessarily
  // violate geometric constraints on the edge orientations.
  MOCHI_ERROR_IF(
      isClosedLoop && numNodes < 3, error, "Closed-loop polyline must have at least 3 nodes");
  MOCHI_ERROR_RETURN(error);

  // Edge length must be nonzero for the unit tangent to be well-defined.
  real constexpr kLengthTolerance = 1e-6_r; // Has units of length
  // Consecutive element tangents must not be 180-degree rotations for parallel transport
  // (Bishop frame generation) to have a well-defined rotation axis. Checked with a fairly loose
  // tolerance to permit flexibility in geometry creation workflows.
  real constexpr kOrthogonalityTolerance = 1e-3_r; // Unitless

  int const numEdges = isClosedLoop ? numNodes : numNodes - 1;
  Real3 prevTangent = {};
  Real3 firstTangent = {};
  for (int i = 0; i < numEdges; ++i) {
    Real3 const edge = nodes[(i + 1) % numNodes] - nodes[i];
    real const edgeLength = Norm(edge);
    MOCHI_ERROR_IF(
        edgeLength < kLengthTolerance,
        error,
        "Polyline elements must have nonzero length (nodes cannot be coincident)");
    MOCHI_ERROR_RETURN(error);
    Real3 const tangent = edge / edgeLength;
    if (i == 0) {
      firstTangent = tangent;
    } else {
      MOCHI_ERROR_IF(
          Dot(prevTangent, tangent) < kOrthogonalityTolerance - 1_r,
          error,
          "Consecutive polyline elements cannot have 180-degree tangent rotations");
    }
    prevTangent = tangent;
  }
  // For closed-loop polylines, also check the wrap-around tangent from the closing edge
  // back to the first edge.
  if (isClosedLoop && numEdges > 1) {
    MOCHI_ERROR_IF(
        Dot(prevTangent, firstTangent) < kOrthogonalityTolerance - 1_r,
        error,
        "Consecutive polyline elements cannot have 180-degree tangent rotations");
  }
}

static void ValidateMesh(MeshDataView const& data, int maxSkinningIndex, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      data.nodesPerElement != 2 && data.nodesPerElement != 3 && data.nodesPerElement != 4,
      error,
      "Mesh element size must be 2 for a polyline mesh, 3 for triangle mesh, or 4 for tetrahedral mesh.");

  // Coordinates
  {
    MOCHI_ERROR_IF(
        data.coordinates.size() > INT_MAX,
        error,
        "Mesh coordinates array is larger than what Mochi currently supports.");
    MOCHI_ERROR_IF(
        (data.coordinates.size() % kMeshDataSpaceDim) != 0,
        error,
        "Mesh coordinates array must be a multiple of 3 (the spatial dimension).");
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF(
        data.GetNumNodes() < data.nodesPerElement,
        error,
        "Mesh must have coordinates for at least one element.");
    MOCHI_ERROR_IF(
        !IsFinite(MakeConstSpan(data.coordinates)),
        error,
        "Mesh coordinates array contains non-finite values.");
    MOCHI_ERROR_RETURN(error);
  }

  // Connectivity
  {
    MOCHI_ERROR_IF(
        static_cast<int64_t>(data.connectivity.size()) > static_cast<int64_t>(INT_MAX),
        error,
        "Mesh connectivity array is larger than what Mochi currently supports.");
    MOCHI_ERROR_IF(
        (isize(data.connectivity) % data.nodesPerElement) != 0,
        error,
        "Mesh connectivity array length must be a multiple of the element size (example: 3 indices per triangle).");
    if (!data.connectivity.empty()) {
      auto const [minIdx, maxIdx] = MinMax(MakeConstSpan(data.connectivity));
      MOCHI_ERROR_IF(
          minIdx < 0 || maxIdx >= data.GetNumNodes(),
          error,
          "Mesh connectivity array contains one or more indices that are out-of-bounds. Must be greater than zero and less than the number of nodes.");
    }
    MOCHI_ERROR_RETURN(error);
  }

  // Skinning
  if (data.skinning) {
    ValidateSkinning(*data.skinning, data.GetNumNodes(), maxSkinningIndex, error);
  }

  // Additional checks for a polyline mesh
  if (data.nodesPerElement == 2) {
    int const numNodes = data.GetNumNodes();
    MOCHI_ERROR_IF(numNodes < 2, error, "Polyline must have at least 2 nodes");
    MOCHI_ERROR_RETURN(error);

    // Closed-loop topology is encoded in the connectivity array. When connectivity is empty the
    // polyline is open (numNodes-1 implicit segments). When provided, 2*numNodes elements indicate
    // a closed loop and 2*(numNodes-1) elements indicate an open polyline.
    bool const isClosedLoop = IsPolylineClosedLoop(data);

    // Validate element geometry: nonzero edge lengths and no 180-degree consecutive tangent
    // rotations. These checks are required before attempting parallel transport (Bishop frame
    // generation) and are also fundamental constraints for valid polyline geometry.
    auto const nodes = Unflatten<Real3 const>(MakeConstSpan(data.coordinates));
    mochi::model::ValidatePolylineGeometry(nodes, isClosedLoop, error);
    MOCHI_ERROR_RETURN(error);

    // If connectivity is provided, validate size and sequential ordering.
    if (!data.connectivity.empty()) {
      int const numElements = isClosedLoop ? numNodes : numNodes - 1;
      MOCHI_ERROR_IF(
          isize(data.connectivity) != 2 * numElements,
          error,
          "Polyline connectivity size must be 2 * numNodes (closed loop) or 2 * (numNodes - 1) (open).");
      MOCHI_ERROR_RETURN(error);
      for (int i = 0; i < numElements; ++i) {
        MOCHI_ERROR_IF(
            (data.connectivity[2 * i] != i) || (data.connectivity[2 * i + 1] != (i + 1) % numNodes),
            error,
            "Polyline connectivity must be sequential: [0,1], [1,2], ..., [n-1,0] (closed loop) or [0,1], ..., [n-2,n-1] (open).");
      }
      MOCHI_ERROR_RETURN(error);
    }
  }
}

static void ValidateConstrainedNodes(Span<int const> data, int numNodes, Error& error) {
  MOCHI_ERROR_RETURN(error);

  // No constrained nodes is OK.
  if (data.empty()) {
    return;
  }

  MOCHI_ERROR_IF(
      data.size() >= INT_MAX,
      error,
      "Mesh constrained nodes array size is larger than Mochi currently supports.");
  auto const [minIdx, maxIdx] = MinMax(MakeConstSpan(data));
  MOCHI_ERROR_IF(
      minIdx < 0 || maxIdx >= numNodes,
      error,
      "Mesh constrained nodes array contains one or more out-of-bounds indices");
}

static void ValidateBox(Box const& data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      !IsFinite(data.center) || !IsFinite(data.halfExtents) || !IsFinite(data.rotation),
      error,
      "Implicit box data contains non-finite values");
  MOCHI_ERROR_IF(
      data.halfExtents[0] < std::numeric_limits<real>::epsilon() ||
          data.halfExtents[1] < std::numeric_limits<real>::epsilon() ||
          data.halfExtents[2] < std::numeric_limits<real>::epsilon(),
      error,
      "Implicit box must have positive volume");
  real constexpr kEpsilon = 1e-6_r;
  MOCHI_ERROR_IF(
      !NearEqual(Norm(data.rotation.ToReal4()), 1_r, kEpsilon),
      error,
      "Implicit box rotation should be a unit quaternion.");
}

static void ValidatePlane(Plane const& data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      !IsFinite(data.GetDistanceFromOrigin()) || !IsFinite(data.GetNormal()),
      error,
      "Implicit plane data contains non-finite values");
  real constexpr kEpsilon = 1e-6_r;
  MOCHI_ERROR_IF(
      !NearEqual(Norm(data.GetNormal()), 1_r, kEpsilon),
      error,
      "Implicit plane normal must be unit length");
}

static void ValidateSphere(Sphere const& data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      !IsFinite(data.GetCenter()) || !IsFinite(data.GetRadius()),
      error,
      "Implicit sphere data contains non-finite values.");
  MOCHI_ERROR_IF(
      data.GetRadius() <= 0_r, error, "Implicit sphere radius must be greater than zero.");
}

// Aabb must have positive finite volume
static bool IsValidAabb(Aabb const& data) {
  Real3 const& min = data.GetMin();
  Real3 const& max = data.GetMax();
  Real3 const diag = max - min;
  return IsFinite(min) && IsFinite(max) && (diag[0] > 0_r) && (diag[1] > 0_r) && (diag[2] > 0_r);
}

static bool IsUniformPositiveScale(Real3 const& scale) {
  return (scale[0] > 0_r) && NearEqualRel(scale[0], scale[1]) && NearEqualRel(scale[0], scale[2]);
}

static bool IsUniformAbsScale(Real3 const& scale) {
  return IsUniformPositiveScale(Abs(scale));
}

template <class T>
static auto NormalizeIfNecessary(T const& vec) {
  auto norm = Norm(vec);
  if (NearEqual(1_r, norm, 1e-6_r)) {
    return vec; // Close enough. Do not modify the data unecessarily.
  } else {
    return vec * (1_r / (norm + std::numeric_limits<real>::min()));
  }
}

static auto NormalizeIfNecessary(Quaternion const& q) {
  auto const result = Quaternion(NormalizeIfNecessary(q.ToReal4()));
  return (result == -Quaternion::Identity()) ? Quaternion::Identity()
                                             : result; // Standardize on positive identity
}

static void ValidateSdf(GridSdfDataView const& data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      data.dims[0] < 2 || data.dims[1] < 2 || data.dims[2] < 2,
      error,
      "Model SDF has invalid dimensions. The minimum is 2x2x2.");
  MOCHI_ERROR_IF(data.values.size() >= INT_MAX, error, "Model SDF grid is too large.");
  MOCHI_ERROR_IF(
      isize(data.values) != static_cast<int64_t>(data.dims[0]) * data.dims[1] * data.dims[2],
      error,
      "Model SDF value array size must be the product of the grid dimensions.");
  MOCHI_ERROR_IF(
      !IsValidAabb(data.bounds),
      error,
      "Model SDF bounds must have finite volume with (max[i] > min[i]) for each axis.");
  MOCHI_ERROR_IF(
      !IsValidAabb(data.negativeValueBounds),
      error,
      "Model SDF negative value bounds must have finite volume with (max[i] > min[i]) for each axis.");
  MOCHI_ERROR_IF(
      GetAabb(data.bounds, data.negativeValueBounds) != data.bounds,
      error,
      "Model SDF negative value bounds must fit within the overall bounds.");
  MOCHI_ERROR_IF(
      !IsFinite(MakeConstSpan(data.values)), error, "Model SDF contains non-finite values.");
  if (data.scale) {
    auto scale = *data.scale;
    MOCHI_ERROR_IF(!IsFinite(scale), error, "Model SDF scale must be finite.");
    MOCHI_ERROR_IF(
        NearEqual(0_r, Min(Abs(scale))), error, "Model SDF scale cannot be zero on any axis.");
    MOCHI_ERROR_IF(
        !IsUniformAbsScale(scale),
        error,
        "Model SDF scale must be uniform (by absolute value) on all axes.");
  }
  if (data.rotation) {
    auto rot = *data.rotation;
    MOCHI_ERROR_IF(!IsFinite(rot), error, "Model SDF rotation must be finite.");
    MOCHI_ERROR_IF(
        !NearEqual(1_r, Norm(rot.ToReal4()), 1e-6_r),
        error,
        "Model SDF rotation must be a unit quaternion.");
  }
  if (data.translation) {
    auto trans = *data.translation;
    MOCHI_ERROR_IF(!IsFinite(trans), error, "Model SDF translation must be finite.");
  }
}

void mochi::model::ValidatePolylineElementFrameAxes(
    Span<Real3 const> elementFrameAxes,
    Span<Real3 const> nodes,
    bool isClosedLoop,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  int const numNodes = isize(nodes);
  int const numElements = isClosedLoop ? numNodes : numNodes - 1;
  MOCHI_ERROR_IF(
      isize(elementFrameAxes) != numElements,
      error,
      "Element frame axes array size is incorrect. Expected one axis per element.");
  MOCHI_ERROR_RETURN(error);

  real constexpr kOrthogonalityTolerance = 1e-3_r; // Unitless
  for (int i = 0; i < numElements; ++i) {
    Real3 const& axis = elementFrameAxes[i];
    MOCHI_ERROR_IF(!IsFinite(axis), error, "Element frame axes must be finite.");
    MOCHI_ERROR_IF(
        Abs(Norm(axis) - 1_r) > kOrthogonalityTolerance,
        error,
        "Element frame axes must be unit vectors");
    Real3 const edge = Normalize(nodes[(i + 1) % numNodes] - nodes[i]);
    MOCHI_ERROR_IF(
        Abs(Dot(axis, edge)) > kOrthogonalityTolerance,
        error,
        "Element frame axes must be orthogonal to elements");
  }
}

static void ValidateElementFrameAxes(
    Span<real const> elementFrameAxes,
    Span<real const> coordinates,
    bool isClosedLoop,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      isize(coordinates) % 3 != 0, error, "Coordinates array must be a multiple of 3 in size.");
  MOCHI_ERROR_IF(
      elementFrameAxes.size() >= INT_MAX, error, "Element frame axes array is too large.");
  MOCHI_ERROR_IF(
      isize(elementFrameAxes) % 3 != 0,
      error,
      "Element frame axes array size must be a multiple of 3.");
  MOCHI_ERROR_RETURN(error);

  mochi::model::ValidatePolylineElementFrameAxes(
      Unflatten<Real3 const>(elementFrameAxes),
      Unflatten<Real3 const>(coordinates),
      isClosedLoop,
      error);
}

void mochi::model::Validate(ModelDataView const& data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  int shapeTypesFound = (int)data.box.has_value() + (int)data.plane.has_value() +
      (int)data.sphere.has_value() + (int)data.mesh.has_value();
  MOCHI_ERROR_IF(shapeTypesFound != 1, error, "Model file must contain exactly one type of shape.");
  MOCHI_ERROR_RETURN(error);

  if (data.mesh) {
    ValidateMesh(*data.mesh, /* maxSkinningIndex */ INT_MAX, error);
    MOCHI_ERROR_RETURN(error);
  }

  if (data.visualMesh) {
    MOCHI_ERROR_RETURN(error);
    // Polyline visual-mesh skinning indices reference *elements*, not *nodes*. An open polyline
    // has (numNodes - 1) elements; a closed-loop polyline has numNodes elements. For triangular
    // and tetrahedral simulation meshes, skinning indices reference *nodes*.
    int maxSkinningIndex = INT_MAX;
    if (data.mesh) {
      int const numNodes = data.mesh->GetNumNodes();
      bool const isPolyline = (data.mesh->nodesPerElement == 2);
      if (isPolyline) {
        bool const isClosedLoop = IsPolylineClosedLoop(*data.mesh);
        maxSkinningIndex = isClosedLoop ? numNodes - 1 : numNodes - 2;
      } else {
        maxSkinningIndex = numNodes - 1;
      }
    }
    ValidateMesh(*data.visualMesh, maxSkinningIndex, error);
    MOCHI_ERROR_IF(
        data.visualMesh->nodesPerElement != 3,
        error,
        "Visual mesh must have 3 nodes per element (triangles).");
    MOCHI_ERROR_RETURN(error);
    // Polyline meshes require skinning data to embed the visual mesh into the rod's element frames.
    if (data.mesh && data.mesh->nodesPerElement == 2) {
      MOCHI_ERROR_IF_NOT(
          data.visualMesh->skinning.has_value(),
          error,
          "Polyline mesh with visual mesh requires skinning data for embedding.");
    }
    MOCHI_ERROR_RETURN(error);
  }

  if (data.blending) {
    MOCHI_ERROR_IF(!data.mesh, error, "Model has blending data but no simulation mesh.");
    MOCHI_ERROR_RETURN(error);
    for (auto const& blendShape : *data.blending) {
      ValidateBlending(blendShape, data.mesh->GetNumNodes(), error);
      MOCHI_ERROR_RETURN(error);
    }
  }

  if (data.constrainedNodes) {
    MOCHI_ERROR_IF(!data.mesh, error, "Model has constrained nodes but no simulation mesh.");
    MOCHI_ERROR_RETURN(error);
    ValidateConstrainedNodes(*data.constrainedNodes, data.mesh->GetNumNodes(), error);
    MOCHI_ERROR_RETURN(error);
  }

  if (data.elementFrameAxes) {
    MOCHI_ERROR_IF(
        !data.mesh || (data.mesh->nodesPerElement != 2),
        error,
        "Model should only have element frame axes if the mesh is a polyline mesh (2 nodes per element).");
    MOCHI_ERROR_RETURN(error);
    bool const isClosedLoop = IsPolylineClosedLoop(*data.mesh);
    ValidateElementFrameAxes(*data.elementFrameAxes, data.mesh->coordinates, isClosedLoop, error);
    MOCHI_ERROR_RETURN(error);
  }

  if (data.box) {
    ValidateBox(*data.box, error);
    MOCHI_ERROR_RETURN(error);
  }

  if (data.plane) {
    ValidatePlane(*data.plane, error);
    MOCHI_ERROR_RETURN(error);
  }

  if (data.sphere) {
    ValidateSphere(*data.sphere, error);
    MOCHI_ERROR_RETURN(error);
  }

  if (data.sdf) {
    ValidateSdf(*data.sdf, error);
    MOCHI_ERROR_RETURN(error);
  }

  if (data.material) {
    MOCHI_ERROR_IF(
        !data.mesh, error, "Model has per-element material data, but no simulation mesh.");
    MOCHI_ERROR_RETURN(error);
    ValidateSoftMaterialParams(*data.material, data.mesh->GetNumElements(), error);
    MOCHI_ERROR_RETURN(error);
  }
}

static void AutoCorrectSdf(GridSdfData& data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  // Clear std::optional fields with default values so they don't get resaved.
  if (data.scale && (*data.scale == Real3{1_r, 1_r, 1_r})) {
    data.scale = std::nullopt;
  }
  if (data.rotation) {
    if ((data.rotation == Quaternion::Identity()) || (data.rotation == -Quaternion::Identity())) {
      data.rotation = std::nullopt;
    } else {
      data.rotation = NormalizeIfNecessary(*data.rotation);
    }
  }
  if (data.translation && (*data.translation == Real3{})) {
    data.translation = std::nullopt;
  }
}

static void AutoCorrectElementFrameAxes(Span<real> data, Error& error) {
  MOCHI_ERROR_IF(
      data.size() % 3 != 0, error, "Element frame axes array size must be a multiple of 3");
  MOCHI_ERROR_RETURN(error);
  for (auto& axis : Unflatten<Real3>(MakeSpan(data))) {
    axis = NormalizeIfNecessary(axis);
  }
}

static void NormalizeWeights(SkinningData& data, Error& error) {
  MOCHI_ERROR_IF(
      data.weightsPerNode <= 0,
      error,
      "Skinning data must provide at least one weight and one index per node");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      data.indices.size() != data.weights.size(),
      error,
      "Skinning requires the same number of weights and indices");
  MOCHI_ERROR_IF(
      data.indices.size() % data.weightsPerNode != 0,
      error,
      "Skinning weights and indices must be an even multiple of weightsPerNode.");
  MOCHI_ERROR_RETURN(error);
  for (size_t i = 0; i < data.weights.size(); i += data.weightsPerNode) {
    auto* rangeStart = &data.weights[i];
    auto* rangeEnd = rangeStart + data.weightsPerNode;
    real sum = std::accumulate(rangeStart, rangeEnd, 0.0_r);
    real constexpr kEpsilon = 1e-6_r;
    if (NearEqual(1_r, sum, kEpsilon)) {
      // Already normalized (close enough)
    } else if (NearEqual(0_r, sum, kEpsilon)) {
      MOCHI_ERROR_SET(error, "Skinning weights for each node must be non-zero.");
    } else {
      real invSum = 1_r / sum;
      std::transform(rangeStart, rangeEnd, rangeStart, [invSum](real in) { return in *= invSum; });
    }
  }
}

void mochi::model::AutoCorrect(ModelData& data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (data.mesh) {
    if (data.mesh->skinning) {
      NormalizeWeights(*data.mesh->skinning, error);
    }
  }

  if (data.visualMesh) {
    if (data.visualMesh->connectivity.empty()) {
      // Some old H5 files have empty datasets for the visual mesh. We delete meshes like that.
      data.visualMesh = std::nullopt;
    } else {
      if (data.visualMesh->skinning) {
        NormalizeWeights(*data.visualMesh->skinning, error);
      }
    }
  }

  if (data.blending) {
    if (data.blending->empty()) {
      data.blending = std::nullopt; // Prune empty array
    }
  }

  if (data.constrainedNodes) {
    if (data.constrainedNodes->empty()) {
      // Prune empty array
      data.constrainedNodes = std::nullopt;
    } else {
      // Sort and prune duplicates
      SortAndRemoveDuplicates(*data.constrainedNodes);
    }
  }

  if (data.elementFrameAxes) {
    if (data.elementFrameAxes->empty()) {
      // Prune empty array
      data.elementFrameAxes = std::nullopt;
    } else {
      AutoCorrectElementFrameAxes(*data.elementFrameAxes, error);
    }
  }

  if (data.box) {
    data.box->rotation = NormalizeIfNecessary(data.box->rotation);
  }

  if (data.plane) {
    auto normal = NormalizeIfNecessary(data.plane->GetNormal());
    auto distance = data.plane->GetDistanceFromOrigin();
    data.plane = Plane{normal, distance};
  }

  if (data.sdf) {
    AutoCorrectSdf(*data.sdf, error);
  }
}

namespace {
// A transform expressed in several ways
struct BakeTransformInput {
  Real3 scale{1_r, 1_r, 1_r}; // Per-axis scale. Negative scale mirrors.
  TransformRT rt; // Rotation and translation (no scale)
  VMatrix3x3r directionMatrix; // Change of basis (rotation/mirroring, no scale).
  VMatrix4x4r matrix; // Transforms points (scale/mirroring, rotation, translation)
};
} // namespace

static bool IsExactlyUniformMagnitude(Real3 const& scale) {
  return (Abs(scale[0]) == Abs(scale[1])) && (Abs(scale[0]) == Abs(scale[2]));
}

static BakeTransformInput MakeBakeTransformInput(Real3 const& scale, TransformRT const& rt) {
  Quaternion const q = NormalizeIfNecessary(rt.GetRotation());
  TransformRT const rtNormalized = TransformRT{q, rt.GetTranslation()};
  return BakeTransformInput{
      scale,
      rtNormalized,
      // Only the sign of the scale affects the change-of-basis matrix.
      Dot3x3(ToVMatrix3x3(q), VDiagonalMatrix<3>(ToSimd(Sign(scale), 0_r))),
      Dot4x4(ToVMatrix4x4(rtNormalized), VDiagonalMatrix<4>(ToSimd(scale, 1_r)))};
}

static BakeTransformInput MakeBakeTransformInput(CoordinateSpaceConverter const& converter) {
  real const scale = converter.GetScale();

  // A quaternion cannot hold a reflection, so when the conversion flips handedness the reflection
  // has to move into the scale, leaving a proper rotation behind. Which axis carries it is not
  // determined by the conversion: a signed permutation with negative determinant need not negate
  // any axis at all (FLU to LFU is a plain X/Y swap). Any odd number of negated axes reproduces the
  // same transform, so pick axis 0 to match @ref DecomposeMatrixTransform. The choice cancels out
  // for everything except the quaternion itself, which stays a valid orientation either way.
  Matrix3x3r rotation = converter.GetDirectionMatrix();
  Real3 scale3{scale, scale, scale};
  if (converter.FlipsHandedness()) {
    scale3[0] = -scale3[0];
    for (int row = 0; row < 3; ++row) {
      rotation[row][0] = -rotation[row][0];
    }
  }

  return BakeTransformInput{
      scale3,
      TransformRT{QuaternionFromMatrix(rotation), Real3{}},
      ToSimdMatrix(converter.GetDirectionMatrix()),
      ToSimdMatrix(converter.GetTransformMatrix())};
}

static void BakeTransformBox(Box& data, BakeTransformInput const& transform, Error& error) {
  Real3 const& scale = transform.scale;
  bool const isRotated = !EquivalentRotation(data.rotation, Quaternion::Identity());
  MOCHI_ERROR_IF(
      isRotated && !IsUniformAbsScale(scale),
      error,
      "Cannot apply non-uniform scale to a rotated implicit box.");
  MOCHI_ERROR_RETURN(error);

  // If the box is axis-aligned or the scale is exactly equal and positive, then we can compose
  // the quaternion rotations. This preserves precision by avoiding a DecomposeMatrixTransform.
  bool const keepsBoxAxes = (data.rotation == Quaternion::Identity()) ||
      (data.rotation == -Quaternion::Identity()) ||
      ((scale[0] > 0_r) && (scale[0] == scale[1]) && (scale[0] == scale[2]));

  if (keepsBoxAxes) {
    data.rotation = Normalize(transform.rt.GetRotation() * data.rotation);
  } else {
    VMatrix4x4r const boxMat = ToVMatrix4x4(TransformRT{data.rotation, data.center});
    data.rotation =
        Normalize(DecomposeMatrixTransform(Dot4x4(transform.matrix, boxMat)).second.GetRotation());
  }
  data.halfExtents *= Abs(scale);
  data.center = ToReal3(DotMatVec4x4(transform.matrix, ToSimd(data.center, 1_r)));
}

static void BakeTransformPlane(Plane& data, BakeTransformInput const& transform, Error& error) {
  MOCHI_ERROR_RETURN(error);
  Real3 const& scale = transform.scale;

  Real3 normal = {};
  real distance = 0_r;
  if (IsExactlyUniformMagnitude(scale)) {
    // The direction matrix alone transforms the normal, and it preserves length, so the normal is
    // left exactly as the change of basis produced it. Every distance from the origin then scales
    // by the shared magnitude.
    normal = ToReal3(DotMatVec3x3(transform.directionMatrix, ToSimd(data.GetNormal(), 0_r)));
    distance = data.GetDistanceFromOrigin() * Abs(scale[0]);
  } else {
    // Non-uniform scale stretches a normal by an amount that depends on its direction, so divide by
    // the magnitude first to form the inverse transpose. The factor that renormalizes the result is
    // also the one that rescales the distance, because
    // Dot(InverseTranspose(A) * n, A * p) == Dot(n, p).
    Real3 const stretched = ToReal3(
        DotMatVec3x3(transform.directionMatrix, ToSimd(data.GetNormal() / Abs(scale), 0_r)));
    real const invLength = 1_r / (Norm(stretched) + std::numeric_limits<real>::min());
    normal = Normalize(stretched * invLength);
    distance = data.GetDistanceFromOrigin() * invLength;
  }

  data = Plane{normal, distance + Dot(normal, transform.rt.GetTranslation())};
}

static void BakeTransformSphere(Sphere& data, BakeTransformInput const& transform, Error& error) {
  MOCHI_ERROR_IF(
      !IsUniformAbsScale(transform.scale),
      error,
      "Cannot apply non-uniform scale to an implicit sphere.");
  MOCHI_ERROR_RETURN(error);
  auto const center = DotMatVec4x4(transform.matrix, data.VGetCenter());
  auto const radius = data.GetRadius() * Abs(transform.scale[0]);
  data = Sphere{center, radius};
}

static void BakeTransformMesh(MeshData& data, VMatrix4x4r const& matrix, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      data.coordinates.size() % 3 != 0,
      error,
      "Coordinates array size must be a multiple of three.");
  MOCHI_ERROR_RETURN(error);
  ArrayTransformPoints_MatT(
      Unflatten<Real3>(data.coordinates),
      Unflatten<Real3 const>(data.coordinates),
      Transpose4x4(matrix));
}

static void BakeTransformElementFrameAxes(
    Span<real> elementFrameAxes,
    MeshData const& mesh,
    BakeTransformInput const& transform,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  MOCHI_ERROR_IF(
      mesh.nodesPerElement != 2,
      error,
      "Element frame axes can only be transformed for polyline meshes.");

  bool const isClosedLoop = IsPolylineClosedLoop(mesh);
  int const numElements = isClosedLoop ? mesh.GetNumNodes() : mesh.GetNumNodes() - 1;
  MOCHI_ERROR_IF(
      isize(elementFrameAxes) != numElements * 3,
      error,
      "Element frame axes array size is incorrect. Expected 3 values per element.");
  MOCHI_ERROR_IF(!IsFinite(elementFrameAxes), error, "Element frame axes must be finite.");
  MOCHI_ERROR_RETURN(error);

  // Frame axes are normal directions: transform by inverse-transpose of the linear transform so
  // that they stay orthogonal to the transformed element tangents under non-uniform scale.
  if (IsExactlyUniformMagnitude(transform.scale)) {
    // The direction matrix preserves length, so a unit axis stays exactly unit.
    ArrayRotateVectors_MatT(
        Unflatten<Real3>(elementFrameAxes),
        Unflatten<Real3 const>(elementFrameAxes),
        Transpose3x3(transform.directionMatrix));
  } else {
    Real3 const invScale = 1_r / Abs(transform.scale);
    for (auto& axis : Unflatten<Real3>(elementFrameAxes)) {
      axis =
          Normalize(ToReal3(DotMatVec3x3(transform.directionMatrix, ToSimd(axis * invScale, 0_r))));
    }
  }
}

static void BakeTransformSdf(GridSdfData& data, Real3 scale, TransformRT const& rt, Error& error) {
  MOCHI_ERROR_RETURN(error);

  MOCHI_ERROR_IF(
      !IsUniformAbsScale(scale),
      error,
      "Cannot bake scale into an SDF grid unless it is uniform (by absolute value) on all axes.");
  MOCHI_ERROR_IF(
      data.scale && NearEqual(Min(Abs(*data.scale)), 0_r),
      error,
      "Cannot bake scale into an SDF grid because it already has zero scale (or nearly zero).");
  MOCHI_ERROR_IF(
      data.rotation && NearEqual(data.rotation->ToReal4(), Real4{}),
      error,
      "Cannot bake scale into an SDF grid because it already has a rotation quaternion of zero magnitude (or nearly zero).");
  MOCHI_ERROR_RETURN(error);

  // If the input scale was not exactly uniform (by absolue value), then clamp to the average value.
  if ((scale[0] != scale[1]) || (scale[0] != scale[2])) {
    scale = Mean(Abs(scale)) * Sign(scale);
  }

  if (!data.scale && !data.rotation && !data.translation) {
    // This is the first transformation to be baked into the SDF (majority case).
    // Store the values losslessly.
    data.scale = scale;
    data.rotation = rt.GetRotation();
    data.translation = rt.GetTranslation();
  } else {
    auto prevScale = data.scale.value_or(Real3{1_r, 1_r, 1_r});
    auto prevRot = data.rotation ? NormalizeIfNecessary(*data.rotation) : Quaternion::Identity();
    auto prevTrans = data.translation.value_or(Real3{});
    if (IsUniformPositiveScale(prevScale) && IsUniformPositiveScale(scale)) {
      // Use TransformSRT to concatenate transforms
      auto result = TransformSRT{scale[0], rt} * TransformSRT{prevScale[0], prevRot, prevTrans};
      data.scale = Real3{result.GetScale(), result.GetScale(), result.GetScale()};
      data.rotation = result.GetRotation();
      data.translation = result.GetTranslation();
    } else {
      // Use matrices to concatenate transforms that include negative scale (mirroring).
      VMatrix4x4r prevMat = Dot4x4(
          ToVMatrix4x4(TransformRT{prevRot, prevTrans}),
          VDiagonalMatrix<4>(ToSimd(prevScale, 1_r)));
      VMatrix4x4r newMat = Dot4x4(ToVMatrix4x4(rt), VDiagonalMatrix<4>(ToSimd(scale, 1_r)));
      VMatrix4x4r resultMat = Dot4x4(newMat, prevMat);

      // Extract scale, rotation, and translation
      auto [resultScale, resultRT] = DecomposeMatrixTransform(resultMat);

      // Scale should still be uniform by absolute value.
      MOCHI_ASSERT_VERBOSE(
          NearEqualRel(Abs(resultScale[0]), Abs(resultScale[1])) &&
              NearEqualRel(Abs(resultScale[0]), Abs(resultScale[2])),
          "The concatenated transform should have uniform scale (by absolute value) since both input "
          "transformations had uniform scale (by absolute value).");

      if ((resultScale[0] != resultScale[1]) || (resultScale[0] != resultScale[2])) {
        // Clamp scale to the average so that it is exactly uniform (by absolute value).
        resultScale = Mean(Abs(resultScale)) * Sign(resultScale);
      }

      // Store the results
      data.scale = resultScale;
      data.rotation = resultRT.GetRotation();
      data.translation = resultRT.GetTranslation();
    }
  }

  // AutoCorrect to prune default values
  AutoCorrectSdf(data, error);
}

static void BakeTransformImpl(ModelData& data, BakeTransformInput const& transform, Error& error) {
  MOCHI_ERROR_RETURN(error);

  MOCHI_ERROR_IF(
      !IsFinite(transform.rt) || !IsFinite(transform.scale), error, "Parameters must be finite.");
  MOCHI_ERROR_IF(
      NearEqual(0_r, Norm(transform.rt.GetRotation().ToReal4())),
      error,
      "Rotation quaternion cannot have zero magnitude.");
  MOCHI_ERROR_IF(
      Min(Abs(transform.scale)) < std::numeric_limits<real>::epsilon(),
      error,
      "Scale of zero is not allowed since it would result in zero volume.");
  MOCHI_ERROR_RETURN(error);

  // Early out if identity
  if ((transform.rt == TransformRT::Identity()) && (transform.scale == Real3{1_r, 1_r, 1_r})) {
    return;
  }

  MOCHI_ERROR_IF(
      data.material.has_value() && !IsUniformPositiveScale(transform.scale) &&
          (!data.material->anisoTheta.empty() || !data.material->anisoPhi.empty()),
      error,
      "Transformation of per-element ActiveNeoHookean material data (anisoTheta and anisoPhi) "
      "is not currently supported.");
  MOCHI_ERROR_RETURN(error);

  if (data.box) {
    BakeTransformBox(*data.box, transform, error);
  }

  if (data.sphere) {
    BakeTransformSphere(*data.sphere, transform, error);
  }

  if (data.plane) {
    BakeTransformPlane(*data.plane, transform, error);
  }

  if (data.mesh) {
    BakeTransformMesh(*data.mesh, transform.matrix, error);
  }

  if (data.elementFrameAxes) {
    MOCHI_ERROR_IF(!data.mesh, error, "Model has element frame axes but no simulation mesh.");
    MOCHI_ERROR_RETURN(error);
    BakeTransformElementFrameAxes(*data.elementFrameAxes, *data.mesh, transform, error);
  }

  if (data.visualMesh) {
    BakeTransformMesh(*data.visualMesh, transform.matrix, error);
  }

  if (data.sdf) {
    Error sdfError;
    BakeTransformSdf(*data.sdf, transform.scale, transform.rt, sdfError);
    if (!sdfError.IsOK()) {
      // Since we couldn't bake this transform into the SDF grid, we discard the data.
      // If necessary, a new one will be computed on-demand based on the transformed mesh data.
      data.sdf = std::nullopt;
    }
  }

  // Mirroring on an odd number of axes turns the mesh inside out. We detect and fix this
  // automatically.
  if (Prod(transform.scale) < 0_r) {
    FlipWindingOrder(data, error);
  }
}

void mochi::model::BakeTransform(
    ModelData& data,
    Real3 const& scale,
    TransformRT const& transform,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  BakeTransformImpl(data, MakeBakeTransformInput(scale, transform), error);
}

void mochi::model::BakeTransform(
    ModelData& data,
    Real3 const& scale,
    Quaternion const& rotation,
    Real3 const& translation,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  BakeTransformImpl(data, MakeBakeTransformInput(scale, TransformRT{rotation, translation}), error);
}

void mochi::model::BakeSdf(ModelData& data, GridSdfParams const& params, Error& error) {
  MOCHI_ERROR_RETURN(error);

  // This operation requires a valid mesh
  MOCHI_ERROR_IF(!data.mesh.has_value(), error, "Baking an SDF requires a mesh");
  MOCHI_ERROR_RETURN(error);
  ValidateMesh(*data.mesh, /* maxSkinningIndex */ INT_MAX, error);
  MOCHI_ERROR_RETURN(error);

  // Get a triangle mesh from the model data.
  std::shared_ptr<TriangularMesh const> triMesh;
  if (data.mesh->nodesPerElement == 4) {
    auto tetMesh = TetrahedralMesh{
        Unflatten<Real3 const>(MakeConstSpan(data.mesh->coordinates)),
        Unflatten<Int4 const>(MakeConstSpan(data.mesh->connectivity))};
    triMesh = std::make_shared<TriangularMesh>(CreateBoundaryMesh(tetMesh));
  } else if (data.mesh->nodesPerElement == 3) {
    triMesh = std::make_shared<TriangularMesh>(
        Unflatten<Real3 const>(MakeConstSpan(data.mesh->coordinates)),
        Unflatten<Int3 const>(MakeConstSpan(data.mesh->connectivity)));
  } else {
    MOCHI_ERROR_SET(error, "Unsupported mesh type");
    return;
  }

  // Compute an GridSdf. This may take a while.
  GridSdf gridSdf{triMesh, params, error};
  MOCHI_ERROR_RETURN(error);
  auto const& grid = gridSdf.GetDistanceGrid();

  // Extract the serializable data
  GridSdfData gridSdfData;
  gridSdfData.dims = grid.GetDimensions();
  gridSdfData.values = grid.GetData();
  gridSdfData.bounds = grid.GetBounds();
  gridSdfData.negativeValueBounds = grid.GetNegativeValueBounds();

  // Store the results (may replace existing)
  data.sdf = std::move(gridSdfData);
}

void mochi::model::GenerateVisualMeshEmbedding(ModelData& data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      data.mesh && data.mesh->nodesPerElement == 4,
      error,
      "GenerateVisualMeshEmbedding requires a tetrahedral simulation mesh.");
  MOCHI_ERROR_IF_NOT(
      data.visualMesh && !data.visualMesh->coordinates.empty(),
      error,
      "GenerateVisualMeshEmbedding requires a visual mesh with coordinates.");
  MOCHI_ERROR_RETURN(error);
  ValidateMesh(*data.mesh, /* maxSkinningIndex */ INT_MAX, error);
  ValidateMesh(*data.visualMesh, /* maxSkinningIndex */ INT_MAX, error);
  MOCHI_ERROR_RETURN(error);

  auto const tetCoords = Unflatten<Real3 const>(MakeConstSpan(data.mesh->coordinates));
  auto const tetConn = Unflatten<Int4 const>(MakeConstSpan(data.mesh->connectivity));
  MOCHI_ERROR_IF(tetConn.empty(), error, "Tetrahedral mesh has no elements.");
  MOCHI_ERROR_RETURN(error);

  auto const tetMesh = std::make_shared<TetrahedralMesh>(tetCoords, tetConn);
  TetrahedralMeshAabbObject bvhObject(tetMesh);
  AabbTree tree(&bvhObject, BvhTreeParams{});

  auto const visCoords = Unflatten<Real3 const>(MakeConstSpan(data.visualMesh->coordinates));
  int const numVis = isize(visCoords);
  int constexpr kWeightsPerNode = 4;

  SkinningData skinning;
  skinning.weightsPerNode = kWeightsPerNode;
  skinning.indices.resize_noinit(static_cast<size_t>(numVis) * kWeightsPerNode);
  skinning.weights.resize_noinit(static_cast<size_t>(numVis) * kWeightsPerNode);

  ParallelForN("GenerateVisualMeshEmbedding", numVis, 256, [&](int i) {
    int const tetIdx = tree.FindClosest(visCoords[i]); // const, thread-safe
    Int4 const tet = tetConn[tetIdx];
    NdArray<real, 4> const w = BarycentricCoords4(
        tetCoords[tet[0]], tetCoords[tet[1]], tetCoords[tet[2]], tetCoords[tet[3]], visCoords[i]);
    size_t const base = static_cast<size_t>(i) * kWeightsPerNode;
    for (int j = 0; j < kWeightsPerNode; ++j) {
      skinning.indices[base + j] = tet[j];
      skinning.weights[base + j] = w[j];
    }
  });

  // Warn if any visual node extrapolates by more than a tetrahedron's vertex-to-face height.
  real const largestExtrapolation = MaxVisualMeshExtrapolation(skinning);
  if (largestExtrapolation > 1_r) {
    MOCHI_LOG_WARNING(
        "Some visual nodes lie up to %.2f element-heights outside the tetrahedral mesh.",
        largestExtrapolation);
  }

  data.visualMesh->skinning = std::move(skinning);
}

real mochi::model::MaxVisualMeshExtrapolation(SkinningDataView const& skinning) {
  real largest = 0_r;
  for (real const w : skinning.weights) {
    largest = Max(largest, -w);
  }
  return largest;
}

void model::FlipWindingOrder(MeshData& data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      data.nodesPerElement != 2 && data.nodesPerElement != 3 && data.nodesPerElement != 4,
      error,
      "Mesh element size must be 2 for a polyline mesh, 3 for triangle mesh, or 4 for tetrahedral mesh.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      isize(data.connectivity) % data.nodesPerElement != 0,
      error,
      "Invalid mesh connectivity. Must be a multiple of the element size.");
  MOCHI_ERROR_RETURN(error);

  if (data.nodesPerElement >= 3) {
    for (int i = 0; i < isize(data.connectivity); i += data.nodesPerElement) {
      auto* elem = &data.connectivity[i];
      std::swap(elem[0], elem[1]);
    }
  }
}

void mochi::model::FlipWindingOrder(ModelData& data, Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (data.mesh) {
    FlipWindingOrder(*data.mesh, error);
  }
  if (data.visualMesh) {
    FlipWindingOrder(*data.visualMesh, error);
  }
}

void mochi::model::BakeCoordinateSpaceTransform(
    MeshData& data,
    CoordinateSpace const& fromSpace,
    CoordinateSpace const& toSpace,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  fromSpace.Validate(error);
  toSpace.Validate(error);
  MOCHI_ERROR_RETURN(error);
  if (fromSpace == toSpace) {
    return;
  }

  CoordinateSpaceConverter const converter{fromSpace, toSpace};
  BakeTransformMesh(data, ToSimdMatrix(converter.GetTransformMatrix()), error);
  if (converter.FlipsHandedness()) {
    FlipWindingOrder(data, error);
  }
}

void mochi::model::BakeCoordinateSpaceTransform(
    ModelData& data,
    CoordinateSpace const& fromSpace,
    CoordinateSpace const& toSpace,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  fromSpace.Validate(error);
  toSpace.Validate(error);
  MOCHI_ERROR_RETURN(error);
  if (fromSpace == toSpace) {
    return;
  }

  CoordinateSpaceConverter const converter{fromSpace, toSpace};
  BakeTransformImpl(data, MakeBakeTransformInput(converter), error);
}
