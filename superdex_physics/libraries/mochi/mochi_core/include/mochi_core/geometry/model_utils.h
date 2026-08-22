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

#include <mochi_core/geometry/grid_sdf_params.h>
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/transform_rt.h>

#include <string_view>

namespace mochi {
struct CoordinateSpace;
} // namespace mochi

namespace mochi::model {

/**
 * @brief Load model data from a file. Then, @ref model::AutoCorrect and @ref model::Validate
 * will be called automatically.
 *
 * @details Supported formats: JSON (.mochi.json), HDF5 (.mochi.h5), OBJ (.obj), OFF (.off),
 * PLY (.ply), and STL (.stl).
 *
 * @param[in] path File path to load.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return @ref ModelData that was loaded.
 */
[[nodiscard]] ModelData LoadFromFile(std::string_view path, Error& error);

/**
 * @brief Load model data from a file without calling @ref model::AutoCorrect nor @ref
 * model::Validate. Can be used to load a model that is not currently in a valid state.
 *
 * @details Supported formats: JSON (.mochi.json), HDF5 (.mochi.h5), OBJ (.obj), OFF (.off),
 * PLY (.ply), and STL (.stl).
 *
 * @param[in] path File path to load.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return @ref ModelData that was loaded.
 */
[[nodiscard]] ModelData LoadFromFileUnchecked(std::string_view path, Error& error);

/**
 * @brief Load model data from a file in memory. Then, @ref model::AutoCorrect and @ref
 * model::Validate will be called automatically.
 *
 * @param[in] data File contents in memory.
 * @param[in] format Mesh file format hint. When @ref MeshFileType::Legacy (the default),
 * auto-detects between HDF5 and JSON via header bytes. When a surface mesh format
 * (@ref MeshFileType::PLY, @ref MeshFileType::OFF, @ref MeshFileType::STL, @ref MeshFileType::OBJ),
 * dispatches directly to that reader.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return @ref ModelData that was loaded.
 */
[[nodiscard]] ModelData LoadFromBytes(Span<char const> data, MeshFileType format, Error& error);

/// @overload Convenience overload using @ref MeshFileType::Legacy.
[[nodiscard]] inline ModelData LoadFromBytes(Span<char const> data, Error& error) {
  return LoadFromBytes(data, MeshFileType::Legacy, error);
}

/**
 * @brief Load model data from a file in memory without calling @ref model::AutoCorrect nor @ref
 * model::Validate. Can be used to load a model that is not currently in a valid state.
 *
 * @param[in] data File contents in memory.
 * @param[in] format Mesh file format hint. When @ref MeshFileType::Legacy (the default),
 * auto-detects between HDF5 and JSON via header bytes. When a surface mesh format
 * (@ref MeshFileType::PLY, @ref MeshFileType::OFF, @ref MeshFileType::STL, @ref MeshFileType::OBJ),
 * dispatches directly to that reader.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return @ref ModelData that was loaded.
 */
[[nodiscard]] ModelData
LoadFromBytesUnchecked(Span<char const> data, MeshFileType format, Error& error);

/// @overload Convenience overload using @ref MeshFileType::Legacy.
[[nodiscard]] inline ModelData LoadFromBytesUnchecked(Span<char const> data, Error& error) {
  return LoadFromBytesUnchecked(data, MeshFileType::Legacy, error);
}

/**
 * @brief Save a model to a file of the specified format.
 *
 * @param[in] data Model data to write.
 * @param[in] path File path to write.
 * @param[in] format File format to write.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @warning Some model files contain additional data for experimental features (e.g. ROMs), which
 * cannot be represented by the @ref ModelData struct. That data will be lost if you use the @ref
 * ModelData struct to save over the original file.
 *
 * @note Creates the destination directories if necessary.
 */
void SaveToFile(ModelData const& data, std::string_view path, FileFormat format, Error& error);

/**
 * @brief Save a model to a file of the specified format.
 *
 * @param[in] data A non-owning view of the model data to write.
 * @param[in] path File path to write.
 * @param[in] format File format to write.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @warning Some model files contain additional data for experimental features (e.g. ROMs), which
 * cannot be represented by the @ref ModelData struct. That data will be lost if you use the @ref
 * ModelData struct to save over the original file.
 *
 * @note Creates the destination directories if necessary.
 */
void SaveToFile(ModelDataView const& data, std::string_view path, FileFormat format, Error& error);

/**
 * @brief Save a model to a JSON string in memory.
 *
 * @param[in] data Model data to serialize.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return JSON string containing the model data.
 */
[[nodiscard]] DynamicString SaveToJsonString(ModelData const& data, Error& error);

/**
 * @brief Check the model for errors.
 *
 * @param[in] data Non-owning view of the model data to check.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
void Validate(ModelDataView const& data, Error& error);

/**
 * @brief Validate that a polyline has well-defined element tangents.
 *
 * @details Checks two geometric preconditions required to compute element tangents and a
 * discrete Bishop frame via parallel transport:
 *   1. Every edge has nonzero length (no coincident consecutive nodes), so the unit tangent
 *      is defined.
 *   2. No two consecutive element tangents are 180-degree rotations of each other, so the
 *      rotation axis of the parallel-transport step is defined.
 * For a closed loop the wrap-around edge participates in both checks. Iterates over
 * `numNodes - 1` edges for an open polyline, or `numNodes` edges for a closed loop.
 *
 * @param[in] nodes Polyline centerline node positions [m]. Must have at least 2 nodes
 *            (3 if @p isClosedLoop is true).
 * @param[in] isClosedLoop If true, also check the wrap-around edge from the last node to the
 *            first.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
void ValidatePolylineGeometry(Span<Real3 const> nodes, bool isClosedLoop, Error& error);

/**
 * @brief Validate per-element frame axes of a polyline.
 *
 * @details Each axis must be a unit vector that is orthogonal to its element's edge tangent
 * (the unit vector from the element's first node to its second node). Both checks use a
 * fairly loose tolerance, to permit flexibility in geometry creation workflows.
 *
 * @param[in] elementFrameAxes One axis per element. Must contain `numNodes - 1` entries for an
 *            open polyline, or `numNodes` entries for a closed loop.
 * @param[in] nodes Polyline centerline node positions [m]. Caller must ensure all edges have
 *            nonzero length (e.g. via @ref ValidatePolylineGeometry first); otherwise the
 *            orthogonality check is ill-defined.
 * @param[in] isClosedLoop If true, the last element wraps around to the first node.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
void ValidatePolylineElementFrameAxes(
    Span<Real3 const> elementFrameAxes,
    Span<Real3 const> nodes,
    bool isClosedLoop,
    Error& error);

/**
 * @brief Apply automatic in-place fixes to the model data, such as normalizing vectors and weights.
 *
 * @param[in,out] data @ref ModelData to check and possibly modify.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
void AutoCorrect(ModelData& data, Error& error);

/**
 * @brief Modify the @ref ModelData by applying a scale, rotation, and translation (in that order).
 *
 * @param[in,out] data @ref ModelData to modify.
 * @param[in] scale Scale to apply (possibly non-uniform, i.e., 3 unequal absolute values).
 * @param[in] rotation Rotation to apply (quaternion in [x, y, z, w] order).
 * @param[in] translation Translation to apply.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @note Negative scale can be used to mirror the model. In that case, @ref FlipWindingOrder will be
 * called automatically to avoid turning the model inside out.
 * @note If @ref ModelData::elementFrameAxes is present, axes are transformed as normal directions
 * using the inverse-transpose of the scale-rotation transform, then normalized. This preserves
 * orthogonality with transformed polyline element tangents under non-uniform scale.
 *
 * @warning Some model data cannot bake arbitrary non-uniform scale, resulting in an error.
 * @warning Precomputed grid SDF data is preserved only when @p scale is uniform by absolute value.
 * Non-uniform scale by absolute value discards the precomputed SDF. If an SDF collider later
 * requires SDF data, Mochi regenerates the SDF from the transformed mesh at runtime, which may be
 * expensive.
 */
void BakeTransform(
    ModelData& data,
    Real3 const& scale,
    Quaternion const& rotation,
    Real3 const& translation,
    Error& error);

/**
 * @brief Modify the @ref ModelData by applying a scale, rotation, and translation (in that order).
 *
 * @overload
 * @details Takes a combined @ref TransformRT instead of separate rotation and translation.
 *
 * @param[in,out] data @ref ModelData to modify.
 * @param[in] scale Scale to apply (possibly non-uniform, i.e., 3 unequal absolute values).
 * @param[in] transform Combined rotation and translation to apply.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @note Negative scale can be used to mirror the model. In that case, @ref FlipWindingOrder will be
 * called automatically to avoid turning the model inside out.
 * @note If @ref ModelData::elementFrameAxes is present, axes are transformed as normal directions
 * using the inverse-transpose of the scale-rotation transform, then normalized. This preserves
 * orthogonality with transformed polyline element tangents under non-uniform scale.
 *
 * @warning Some model data cannot bake arbitrary non-uniform scale, resulting in an error.
 * @warning Precomputed grid SDF data is preserved only when @p scale is uniform by absolute value.
 * Non-uniform scale by absolute value discards the precomputed SDF. If an SDF collider later
 * requires SDF data, Mochi regenerates the SDF from the transformed mesh at runtime, which may be
 * expensive.
 */
void BakeTransform(ModelData& data, Real3 const& scale, TransformRT const& transform, Error& error);

/**
 * @brief Modify the @ref MeshData to convert it from one @ref CoordinateSpace to another.
 *
 * @details If the transformation flips handedness, the mesh winding order will also be reversed.
 *
 * @note The conversion is exact when @ref CoordinateSpace::unitsPerMeter is unchanged, because it
 * only permutes and negates axes.
 *
 * @param[in,out] data @ref MeshData to modify.
 * @param[in] fromSpace Coordinate space the mesh is currently expressed in.
 * @param[in] toSpace Coordinate space to convert the mesh to.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
void BakeCoordinateSpaceTransform(
    MeshData& data,
    CoordinateSpace const& fromSpace,
    CoordinateSpace const& toSpace,
    Error& error);

/**
 * @brief Modify the @ref ModelData to convert it from one @ref CoordinateSpace to another.
 *
 * @overload
 * @details All spatial data within the model will be transformed. If the transformation flips
 * handedness, the winding order of any meshes will be reversed.
 *
 * @note The conversion is exact when @ref CoordinateSpace::unitsPerMeter is unchanged, except for
 * orientations stored as a quaternion (implicit box and grid SDF), which cannot represent a
 * 90 degree rotation exactly.
 *
 * @param[in,out] data @ref ModelData to modify.
 * @param[in] fromSpace Coordinate space the model is currently expressed in.
 * @param[in] toSpace Coordinate space to convert the model to.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
void BakeCoordinateSpaceTransform(
    ModelData& data,
    CoordinateSpace const& fromSpace,
    CoordinateSpace const& toSpace,
    Error& error);

/**
 * @brief Compute and bake an SDF grid into the model data.
 *
 * @details Computes a signed-distance field (SDF) grid from the model's mesh (triangular or
 * tetrahedral) and stores it in the model's @ref ModelData::sdf field, replacing any existing SDF.
 *
 * @param[in,out] data @ref ModelData to modify. Must contain a triangle or tetrahedral mesh.
 * @param[in] params Parameters to control grid resolution and padding.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @warning SDF computation may be slow.
 */
void BakeSdf(ModelData& data, GridSdfParams const& params, Error& error);

/**
 * @brief Generate a linear skinning embedding of the visual mesh into the tetrahedral simulation
 * mesh, filling @ref ModelData::visualMesh 's @ref MeshData::skinning (weightsPerNode = 4).
 *
 * @details For each visual node, finds the containing tetrahedron (or nearest tet if outside all
 * tets) via a BVH over @ref ModelData::mesh, and computes 4 barycentric weights. Weights are not
 * clamped (exterior nodes extrapolate affinely) and sum to 1. Overwrites any existing skinning.
 * The simulation and visual meshes must already share the same coordinate frame.
 *
 * @param[in,out] data Model with a tetrahedral @ref ModelData::mesh (nodesPerElement == 4) and a
 * @ref ModelData::visualMesh that has coordinates. On success, visualMesh->skinning is populated.
 * @param[in,out] error Set if mesh/visualMesh preconditions are not met or the tet mesh is empty.
 */
void GenerateVisualMeshEmbedding(ModelData& data, Error& error);

/**
 * @brief Return how far a visual-mesh embedding extrapolates outside the tetrahedral mesh,
 * normalized by element height.
 *
 * @details A node's negative barycentric weight means it lies beyond the face opposite that vertex.
 * The weight's magnitude is the distance past that face expressed as a fraction of the tet's height
 * from the vertex to it. This returns the largest fraction over all nodes.
 *
 * @param[in] skinning Stored weights, e.g. from @ref GenerateVisualMeshEmbedding.
 * @return Largest extrapolation, or @c 0 if all nodes are interior or the weights are empty.
 */
[[nodiscard]] real MaxVisualMeshExtrapolation(SkinningDataView const& skinning);

/**
 * @brief Flip mesh winding order by swapping the connectivity indices within each element.
 *
 * @param[in,out] data @ref MeshData to modify.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
void FlipWindingOrder(MeshData& data, Error& error);

/**
 * @brief Flip mesh winding order by swapping the connectivity indices within each element.
 *
 * @overload
 * @details Operates on both the simulation mesh (@ref ModelData::mesh) and the visual mesh
 * (@ref ModelData::visualMesh) when present. Implicit shapes and SDF data are not modified.
 *
 * @param[in,out] data @ref ModelData to modify.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
void FlipWindingOrder(ModelData& data, Error& error);

} // namespace mochi::model
