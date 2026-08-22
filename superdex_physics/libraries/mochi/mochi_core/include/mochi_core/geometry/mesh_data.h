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
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/span.h>

#include <optional>
#include <string_view>

namespace mochi {

// Forwards:
struct BlendingDataView;
struct SkinningDataView;
struct MeshDataView;

/** @brief Spatial dimensionality of mesh data. */
inline constexpr int kMeshDataSpaceDim = 3;

/**
 * @brief Blending data for one source shape within a soft skinned mesh.
 *
 * @see BlendingDataView
 */
struct BlendingData {
  BlendingData() = default; ///< Default constructor

  /**
   * @brief Copy from @ref BlendingDataView.
   *
   * @param[in] src Source data.
   */
  explicit BlendingData(BlendingDataView const& src);

  DynamicString sourceShape; ///< Name of the soft source shape.
  DynamicArray<int> indices; ///< Indices for blending. Size = numNodes * 2.
  DynamicArray<real> weights; ///< Weights for blending Size = numNodes * 2.

#if MOCHI_LANGUAGE_CPP20
  bool operator==(BlendingData const& other) const = default;
  bool operator!=(BlendingData const& other) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::BlendingData)
  MOCHI_FIELD(sourceShape)
  MOCHI_FIELD(indices)
  MOCHI_FIELD(weights)
  MOCHI_STRUCT_END()
};

/**
 * @brief A non-owning view of the data blending data for one source shape within a soft skinned
 * mesh.
 *
 * @see BlendingData
 */
struct BlendingDataView {
  BlendingDataView() = default;

  /**
   * @brief Implicit conversion from @ref BlendingData.
   *
   * @param[in] src Source data.
   */
  BlendingDataView(BlendingData const& src);

  std::string_view sourceShape; ///< Name of the soft source shape
  Span<int const> indices; ///< Indices for blending. Size = numNodes * 2.
  Span<real const> weights; ///< Weights for blending Size = numNodes * 2.

#if MOCHI_LANGUAGE_CPP20
  bool operator==(BlendingDataView const& other) const = default;
  bool operator!=(BlendingDataView const& other) const = default;
#endif
};

/**
 * @brief Skinning data for a Mochi mesh.
 *
 * @see SkinningDataView
 */
struct SkinningData {
  SkinningData() = default; ///< Default constructor

  /**
   * @brief Copy from @ref SkinningDataView.
   *
   * @param[in] src Source data.
   */
  explicit SkinningData(SkinningDataView const& src);

  int weightsPerNode = 0; ///< Number of weights and indices per node being skinned.
  DynamicArray<int> indices; ///< Indices for each node. Size = numNodes * weightsPerNode.
  DynamicArray<real> weights; ///< Weights for each node. Size = numNodes * weightsPerNode.

#if MOCHI_LANGUAGE_CPP20
  bool operator==(SkinningData const& other) const = default;
  bool operator!=(SkinningData const& other) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::SkinningData)
  MOCHI_FIELD(weightsPerNode)
  MOCHI_FIELD(indices)
  MOCHI_FIELD(weights)
  MOCHI_STRUCT_END()
};

/**
 * @brief A non-owning view of the skinning data for a Mochi mesh.
 *
 * @see SkinningData
 */
struct SkinningDataView {
  SkinningDataView() = default;

  /**
   * @brief Implicit conversion from @ref SkinningData.
   *
   * @param[in] src Source data.
   */
  SkinningDataView(SkinningData const& src);

  int weightsPerNode = 0; ///< Number of weights and indices per node being skinned.
  Span<int const> indices; ///< Indices for each node. Size = numNodes * weightsPerNode.
  Span<real const> weights; ///< Weights for each node. Size = numNodes * weightsPerNode.

#if MOCHI_LANGUAGE_CPP20
  bool operator==(SkinningDataView const& other) const = default;
  bool operator!=(SkinningDataView const& other) const = default;
#endif
};

/**
 * @brief Mesh data for a Mochi actor or model file.
 *
 * @see MeshDataView
 */
struct MeshData {
  MeshData() = default; ///< Default constructor

  /**
   * @brief Copy from @ref MeshDataView.
   *
   * @param[in] src Source data.
   */
  explicit MeshData(MeshDataView const& src);

  /**
   * @brief Number of nodes (vertices) per element.
   *
   * @note 2 for polyline mesh, 3 for triangle mesh, 4 for tetrahedral mesh
   */
  int nodesPerElement = 0;

  /**
   * @brief Flat array of node coordinate values [m].
   *
   * @note Size must be a multiple of @ref kMeshDataSpaceDim.
   */
  DynamicArray<real> coordinates;

  /**
   * @brief Flat array of node indices forming the elements.
   *
   * @note Size must be a multiple of @ref nodesPerElement.
   */
  DynamicArray<int> connectivity;

  /** @brief Optional skinning */
  std::optional<SkinningData> skinning;

  /** @brief Return the number of nodes */
  int GetNumNodes() const;

  /** @brief Return the number of elements */
  int GetNumElements() const;

  /** @brief Return true if the mesh is empty. */
  [[nodiscard]] bool IsEmpty() const;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(MeshData const& other) const = default;
  bool operator!=(MeshData const& other) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::MeshData)
  MOCHI_FIELD(nodesPerElement)
  MOCHI_FIELD(coordinates) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(connectivity)
  MOCHI_FIELD(skinning)
  MOCHI_STRUCT_END()
};

/**
 * @brief A non-owning view of the mesh data for a Mochi actor or model file.
 *
 * @see MeshData
 */
struct MeshDataView {
  MeshDataView() = default;

  /**
   * @brief Implicit conversion from @ref MeshData.
   *
   * @param[in] src Source data.
   */
  MeshDataView(MeshData const& src);

  /**
   * @brief Number of nodes (vertices) per element.
   *
   * @note 2 for polyline mesh, 3 for triangle mesh, 4 for tetrahedral mesh
   */
  int nodesPerElement = 0;

  /**
   * @brief Flat array of node coordinate values [m].
   *
   * @note Size must be a multiple of @ref kMeshDataSpaceDim.
   */
  Span<real const> coordinates;

  /**
   * @brief Flat array of node indices forming the elements.
   *
   * @note Size must be a multiple of @ref nodesPerElement.
   */
  Span<int const> connectivity;

  /** @brief Optional skinning */
  std::optional<SkinningDataView> skinning;

  /** @brief Return the number of nodes */
  int GetNumNodes() const;

  /** @brief Return the number of elements */
  int GetNumElements() const;

  /** @brief Return true if the mesh is empty. */
  [[nodiscard]] bool IsEmpty() const;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(MeshDataView const& other) const = default;
  bool operator!=(MeshDataView const& other) const = default;
#endif
};

} // namespace mochi

#include "mesh_data_inl.h"
