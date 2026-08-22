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
#include <mochi_core/geometry/aabb.h>
#include <mochi_core/geometry/box.h>
#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/geometry/plane.h>
#include <mochi_core/geometry/sphere.h>
#include <mochi_core/materials/material_params.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/reflection.h>

#include <optional>

namespace mochi {

// Forwards:
struct GridSdfDataView;
struct ModelDataView;

/**
 * @brief Data for a pre-computed SDF grid.
 *
 * @see GridSdfDataView
 */
struct GridSdfData {
  GridSdfData() = default;

  /**
   * @brief Copy from @ref GridSdfDataView.
   *
   * @param[in] src Source data.
   */
  explicit GridSdfData(GridSdfDataView const& src);

  /** @brief Dimensions of the SDF grid in X, Y, and Z. */
  Int3 dims{};

  /**
   * @brief Signed distance values [m].
   *
   * @note Size must be (dims[0] * dims[1] * dims[2])
   */
  DynamicArray<real> values;

  /** @brief Spatial bounds of the SDF grid. Values are distributed uniformly within this volume. */
  Aabb bounds;

  /**
   * @brief Spatial bounds of the portion of the SDF grid with negative values.
   *
   * @note This is generally the bounds of the mesh for which the SDF grid was computed, while the
   * overall grid bounds may be larger due to padding for penalty fall-off distance.
   */
  Aabb negativeValueBounds;

  /**
   * @brief Optional parent-from-grid per-axis scale to apply at runtime.
   *
   * @note Applied order is scale, then rotation, then translation.
   * @note Typically set when a transform is baked into the containing model.
   */
  std::optional<Real3> scale;

  /**
   * @brief Optional parent-from-grid rotation to apply at runtime.
   *
   * @note Applied order is scale, then rotation, then translation.
   * @note Typically set when a transform is baked into the containing model.
   */
  std::optional<Quaternion> rotation;

  /**
   * @brief Optional parent-from-grid translation [m] to apply at runtime.
   *
   * @note Applied order is scale, then rotation, then translation.
   * @note Typically set when a transform is baked into the containing model.
   */
  std::optional<Real3> translation;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(GridSdfData const& other) const = default;
  bool operator!=(GridSdfData const& other) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::GridSdfData)
  MOCHI_FIELD(dims)
  MOCHI_FIELD(values) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(bounds)
  MOCHI_FIELD(negativeValueBounds)
  MOCHI_FIELD(scale)
  MOCHI_FIELD(rotation)
  MOCHI_FIELD(translation)
  MOCHI_STRUCT_END()
};

/**
 * @brief A non-owning view of precomputed grid-based signed-distance-field data.
 *
 * @see GridSdfData
 */
struct GridSdfDataView {
  GridSdfDataView() = default;

  /**
   * @brief Implicit conversion from @ref GridSdfData.
   *
   * @param[in] src Source data.
   */
  GridSdfDataView(GridSdfData const& src);

  /** @brief Dimensions of the SDF grid in X, Y, and Z. */
  Int3 dims{};

  /**
   * @brief Signed distance values [m].
   *
   * @note Size must be (dims[0] * dims[1] * dims[2])
   */
  Span<real const> values;

  /** @brief Spatial bounds of the SDF grid. Values are distributed uniformly within this volume. */
  Aabb bounds;

  /**
   * @brief Spatial bounds of the portion of the SDF grid with negative values.
   *
   * @note This is generally the bounds of the mesh for which the SDF grid was computed, while the
   * overall grid bounds may be larger due to padding for penalty fall-off distance.
   */
  Aabb negativeValueBounds;

  /**
   * @brief Optional parent-from-grid per-axis scale to apply at runtime.
   *
   * @note Applied order is scale, then rotation, then translation.
   * @note Typically set when a transform is baked into the containing model.
   */
  std::optional<Real3> scale;

  /**
   * @brief Optional parent-from-grid rotation to apply at runtime.
   *
   * @note Applied order is scale, then rotation, then translation.
   * @note Typically set when a transform is baked into the containing model.
   */
  std::optional<Quaternion> rotation;

  /**
   * @brief Optional parent-from-grid translation [m] to apply at runtime.
   *
   * @note Applied order is scale, then rotation, then translation.
   * @note Typically set when a transform is baked into the containing model.
   */
  std::optional<Real3> translation;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(GridSdfDataView const& other) const = default;
  bool operator!=(GridSdfDataView const& other) const = default;
#endif
};

/**
 * @brief The contents of a Mochi model file.
 *
 * @see ModelDataView
 */
struct ModelData {
  ModelData() = default;

  /**
   * @brief Copy from @ref ModelDataView.
   *
   * @param[in] src Source data.
   */
  explicit ModelData(ModelDataView const& src);

  // Mesh
  std::optional<MeshData> mesh;
  std::optional<MeshData> visualMesh;
  std::optional<DynamicArray<BlendingData>> blending;

  /** @brief Indices of mesh nodes that are constrained. */
  std::optional<DynamicArray<int>> constrainedNodes;

  /**
   * @brief Per-element reference frame axes for polyline meshes.
   *
   * @details Flat array of unit vectors (3 reals per element), each orthogonal to its
   * element's tangent. Only valid when the mesh is a polyline (@ref MeshData::nodesPerElement ==
   * 2).
   */
  std::optional<DynamicArray<real>> elementFrameAxes;

  // Implicit Geometry
  std::optional<Box> box;
  std::optional<Plane> plane;
  std::optional<Sphere> sphere;

  // SDF Grid
  std::optional<GridSdfData> sdf;

  // Soft Material Data (per element)
  std::optional<PerElementSoftMaterialData> material;

  /**
   * @brief True if the loaded model contains unsupported experimental data.
   *
   * @details Some model files contain additional data for experimental features (e.g. ROMs),
   * which cannot be represented by this struct. This flag is set during loading when such data is
   * detected.
   */
  bool experimentalDataDetected = false;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ModelData const& other) const = default;
  bool operator!=(ModelData const& other) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::ModelData)
  MOCHI_FIELD(mesh)
  MOCHI_FIELD(visualMesh)
  MOCHI_FIELD(blending)
  MOCHI_FIELD(constrainedNodes)
  MOCHI_FIELD(elementFrameAxes)
  MOCHI_FIELD(box)
  MOCHI_FIELD(plane)
  MOCHI_FIELD(sphere)
  MOCHI_FIELD(sdf)
  MOCHI_FIELD(material)
  MOCHI_FIELD(experimentalDataDetected) MOCHI_ATTRIBUTE(NoSerialize());
  MOCHI_STRUCT_END()
};

/**
 * @brief A non-owning view of the contents of a Mochi model file.
 *
 * @see ModelData
 */
struct ModelDataView {
  ModelDataView() = default;

  /**
   * @brief Implicit conversion from @ref ModelData.
   *
   * @param[in] src Source data.
   */
  ModelDataView(ModelData const& src);

  // Mesh
  std::optional<MeshDataView> mesh;
  std::optional<MeshDataView> visualMesh;
  // DynamicArray needed here because blending is an array-of-structures.
  std::optional<DynamicArray<BlendingDataView>> blending;

  /** @brief Indices of mesh nodes that are constrained. */
  std::optional<Span<int const>> constrainedNodes;

  /**
   * @brief Per-element reference frame axes for polyline meshes.
   *
   * @details Flat array of unit vectors (3 reals per element), each orthogonal to its
   * element's tangent. Only valid when the mesh is a polyline (@ref MeshDataView::nodesPerElement
   * == 2).
   */
  std::optional<Span<real const>> elementFrameAxes;

  // Implicit Geometry
  std::optional<Box> box;
  std::optional<Plane> plane;
  std::optional<Sphere> sphere;

  // SDF Grid
  std::optional<GridSdfDataView> sdf;

  // Soft Material Data (per element)
  std::optional<PerElementSoftMaterialDataView> material;

  /**
   * @brief True if the loaded model contains unsupported experimental data.
   *
   * @details Some model files contain additional data for experimental features (e.g. ROMs),
   * which cannot be represented by this struct. This flag is set during loading when such data is
   * detected.
   */
  bool experimentalDataDetected = false;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ModelDataView const& other) const = default;
  bool operator!=(ModelDataView const& other) const = default;
#endif
};

// File format options used when saving model data.
enum class FileFormat {
  JSON, ///< JSON format (text)
  H5, ///< HDF5 format (binary). Requires MOCHI_USE_HDF5.
  Count,
};

/// Mesh file format hint for @ref Context::LoadShapeFromBytes and model-loading APIs that read
/// from byte buffers.
enum class MeshFileType {
  Legacy, ///< Auto-detect between HDF5 and JSON (default behavior).
  PLY, ///< PLY format (Stanford Polygon).
  OFF, ///< OFF format (Object File Format).
  STL, ///< STL format (Stereolithography).
  OBJ, ///< OBJ format (Wavefront).
  Count,
};

} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::FileFormat)
MOCHI_ENUM_ITEM(JSON)
MOCHI_ENUM_ITEM(H5)
MOCHI_ENUM_END()

MOCHI_ENUM_BEGIN(mochi::MeshFileType)
MOCHI_ENUM_ITEM(Legacy)
MOCHI_ENUM_ITEM(PLY)
MOCHI_ENUM_ITEM(OFF)
MOCHI_ENUM_ITEM(STL)
MOCHI_ENUM_ITEM(OBJ)
MOCHI_ENUM_END()

#include "model_data_inl.h"
