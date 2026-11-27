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
#include <mochi_core/utils/dynamic_array.h>
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
 * @brief Check skinning data for errors.
 *
 * @param[in] data Non-owning view of the skinning data to check.
 * @param[in] numNodes Number of nodes being skinned.
 * @param[in] numSkinningSources Number of available skinning sources.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
void ValidateSkinning(
    SkinningDataView const& data,
    int numNodes,
    int numSkinningSources,
    Error& error);

/**
 * @brief Validate that a polyline has well-defined element tangents.
 *
 * @details Checks geometric preconditions required to compute element tangents and a
 * discrete Bishop frame via parallel transport:
 *   1. Every edge has finite, nonzero length (finite nodes and no coincident consecutive nodes),
 *      so the unit tangent is defined.
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
 * @brief Generate the frame axes of a polyline's discrete Bishop frame.
 *
 * @details The first element's axis is the unit cross product of its tangent with whichever
 * coordinate axis gives the largest cross product. Each subsequent axis is the previous one
 * parallel-transported by the minimal rotation between consecutive element tangents. For a closed
 * loop, any holonomy (the angle by which transport around a non-planar loop fails to return the
 * first axis to itself) is distributed evenly with respect to arc length, minimizing the peak rate
 * of twist.
 *
 * @param[in] nodes Polyline centerline node positions [m]. Caller must ensure they pass
 *            @ref ValidatePolylineGeometry.
 * @param[in] isClosedLoop If true, the last element wraps around to the first node.
 * @return One unit axis per element, orthogonal to the element's tangent: `numNodes - 1` axes for
 *         an open polyline, or `numNodes` for a closed loop.
 */
[[nodiscard]] DynamicArray<Real3> GenerateDiscreteBishopFrame(
    Span<Real3 const> nodes,
    bool isClosedLoop);

/**
 * @brief Generate missing frame axes of a polyline model.
 *
 * @details Validates the model before generation. Call before @ref BakeTransform so default axes
 * transform like authored axes. Models with frame axes and non-polyline models are unchanged.
 *
 * @param[in,out] model Model data. Unchanged on error.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
void GenerateMissingPolylineFrameAxes(ModelData& model, Error& error);

/**
 * @brief Generate missing frame axes from validated model data.
 *
 * @details Caller must ensure @ref Validate succeeds before calling this overload.
 * Call before @ref BakeTransform so default axes transform like authored axes.
 * Models with frame axes and non-polyline models are unchanged.
 *
 * @param[in,out] model Valid model data.
 */
void GenerateMissingPolylineFrameAxes(ModelData& model);

/**
 * @brief Validate polyline geometry and tube dimensions for @ref GenerateTubularSkinMesh.
 *
 * @details Checks @ref ValidatePolylineGeometry, a positive finite radius, at least three
 * cross-section segments, and that the generated mesh fits Mochi's array-size limits.
 *
 * @param[in] nodes Polyline centerline node positions [m].
 * @param[in] isClosedLoop If true, the last element wraps around to the first node.
 * @param[in] radius Tube radius [m].
 * @param[in] numCrossSectionSegments Number of nodes in each ring.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
void ValidateTubularSkinMeshParameters(
    Span<Real3 const> nodes,
    bool isClosedLoop,
    real radius,
    int numCrossSectionSegments,
    Error& error);

/**
 * @brief Generate a tubular triangle mesh around a polyline, skinned to the polyline's elements.
 *
 * @details Places a ring of @p numCrossSectionSegments nodes at distance @p radius around each
 * element's midpoint, in the plane spanned by its @p elementFrameAxes axis and binormal, and joins
 * consecutive rings by triangle strips.
 * An open polyline also gets a ring around each end node, closed by a triangle fan around a cap
 * node at that end node. Each node is skinned with weight 1 to the element its ring belongs to (the
 * first or last element for the end rings and caps), which makes the mesh valid as either the
 * @ref ModelData::contactSkinMesh or the @ref ModelData::visualMesh of a polyline model.
 *
 * @param[in] nodes Polyline centerline node positions [m]. Must pass
 *            @ref ValidatePolylineGeometry.
 * @param[in] elementFrameAxes One unit axis per element, orthogonal to its tangent. Must pass
 *            @ref ValidatePolylineElementFrameAxes. Generate defaults with
 *            @ref GenerateDiscreteBishopFrame before calling if needed.
 * @param[in] isClosedLoop If true, the last element wraps around to the first node.
 * @param[in] radius Tube radius [m]. Must be positive and finite.
 * @param[in] numCrossSectionSegments Number of nodes in each ring. Must be at least 3.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return Triangle mesh with @ref MeshData::skinning whose indices refer to polyline elements, or
 *         an empty mesh on error.
 */
[[nodiscard]] MeshData GenerateTubularSkinMesh(
    Span<Real3 const> nodes,
    Span<Real3 const> elementFrameAxes,
    bool isClosedLoop,
    real radius,
    int numCrossSectionSegments,
    Error& error);

/**
 * @brief Generate a tubular skin mesh from validated inputs.
 *
 * @details Uses the same construction as the overload taking @ref Error. Caller must ensure
 * @ref ValidateTubularSkinMeshParameters and @ref ValidatePolylineElementFrameAxes succeed before
 * calling this overload. Generated coordinates can still overflow; validate the resulting mesh
 * before exposing it through a public API.
 *
 * @param[in] nodes Validated polyline centerline node positions [m].
 * @param[in] elementFrameAxes Validated per-element frame axes.
 * @param[in] isClosedLoop If true, the last element wraps around to the first node.
 * @param[in] radius Validated tube radius [m].
 * @param[in] numCrossSectionSegments Validated number of nodes in each ring.
 * @return Triangle mesh with element-indexed @ref MeshData::skinning.
 */
[[nodiscard]] MeshData GenerateTubularSkinMesh(
    Span<Real3 const> nodes,
    Span<Real3 const> elementFrameAxes,
    bool isClosedLoop,
    real radius,
    int numCrossSectionSegments);

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
 * @brief Embed a mesh into a tetrahedral mesh by linear skinning, filling @p outMesh 's
 * @ref MeshData::skinning (weightsPerNode = 4).
 *
 * @details For each node of @p outMesh, finds the containing tetrahedron (or nearest tet if outside
 * all tets) via a BVH over @p tetMesh, and computes 4 barycentric weights. Weights are not clamped
 * (exterior nodes extrapolate affinely) and sum to 1. Overwrites any existing skinning. Use it to
 * embed a model's auxiliary meshes (@ref ModelData::visualMesh, @ref ModelData::contactSkinMesh)
 * into its tetrahedral @ref ModelData::mesh. Both meshes must already share the same coordinate
 * frame.
 *
 * @param[in] tetMesh Tetrahedral mesh (nodesPerElement == 4) to embed into.
 * @param[in,out] outMesh Mesh to embed; must have coordinates. On success, its skinning is
 * populated.
 * @param[in,out] error Set if either mesh is invalid or @p tetMesh has no elements.
 */
void EmbedInTetrahedralMesh(MeshDataView const& tetMesh, MeshData& outMesh, Error& error);

/**
 * @brief Return how far an auxiliary-mesh embedding extrapolates outside the tetrahedral mesh,
 * normalized by element height.
 *
 * @details A node's negative barycentric weight means it lies beyond the face opposite that vertex.
 * The weight's magnitude is the distance past that face expressed as a fraction of the tet's height
 * from the vertex to it. This returns the largest fraction over all nodes.
 *
 * @param[in] skinning Stored weights, e.g. from @ref EmbedInTetrahedralMesh.
 * @return Largest extrapolation, or @c 0 if all nodes are interior or the weights are empty.
 */
[[nodiscard]] real MaxAuxiliaryMeshExtrapolation(SkinningDataView const& skinning);

/**
 * @brief Compute linear skinning of a mesh to a polyline's elements, filling @p outMesh 's
 * @ref MeshData::skinning.
 *
 * @details Blends the elements nearest each node of @p outMesh, measured by squared distance to
 * each element's segment, with a smooth kernel of compact support. The weights are nonnegative, sum
 * to 1, and vary continuously with node position, except where a node is equally far from more
 * than @p maxWeightsPerNode elements, e.g. on the axis of a circular arc. The skinning's
 * weightsPerNode is the largest number of elements any node binds to. Overwrites any existing
 * skinning. Logs a warning if nodes bind to @p maxWeightsPerNode elements, as near a bend's center
 * of curvature, where weights vary steeply, or to elements that are not consecutive, where the mesh
 * can tear. The polyline and @p outMesh must already share the same reference coordinate frame
 * and units.
 *
 * @param[in] nodes Polyline centerline node positions [m] in the reference configuration. Must pass
 *            @ref ValidatePolylineGeometry.
 * @param[in] isClosedLoop If true, the last element wraps around to the first node.
 * @param[in] nominalWeightsPerNode Most elements a node binds to where the polyline is straight or
 *            gently curved. With 1, two neighboring elements still blend around a joint. Larger
 *            values blend more widely, with more volume loss at bends. Must be at least 1.
 * @param[in] maxWeightsPerNode Most elements any node binds to, e.g. near a bend's center of
 *            curvature. Must exceed @p nominalWeightsPerNode.
 * @param[in,out] outMesh Mesh to skin; must have coordinates. On success, its skinning is
 *                populated, with indices referring to polyline elements.
 * @param[in,out] error Set if any input is invalid or a point-to-element distance is not finite.
 */
void ComputePolylineSkinning(
    Span<Real3 const> nodes,
    bool isClosedLoop,
    int nominalWeightsPerNode,
    int maxWeightsPerNode,
    MeshData& outMesh,
    Error& error);

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
 * @details Operates on the simulation mesh (@ref ModelData::mesh), visual mesh
 * (@ref ModelData::visualMesh), and contact skin (@ref ModelData::contactSkinMesh) when present.
 * Implicit shapes and SDF data are not modified.
 *
 * @param[in,out] data @ref ModelData to modify.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
void FlipWindingOrder(ModelData& data, Error& error);

/**
 * @brief Reorder the nodes and elements of a mesh for memory locality.
 *
 * @details For a triangle or tetrahedral mesh, the nodes are numbered by reverse Cuthill-McKee and
 * the elements are sorted by their node indices. Each node keeps its coordinates and its skinning
 * weights and indices, and each element keeps its corner order, so orientation and winding are
 * preserved. Reordering a mesh twice gives the same result as once. Polylines are left unchanged.
 *
 * @param[in,out] mesh @ref MeshData to reorder. Unchanged on error.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @note Reordering can speed up the simulation of poorly numbered meshes. Well-numbered meshes see
 * little or no change.
 *
 * @warning Reorder the mesh before creating data that refers to its nodes, elements or boundary
 * faces by index, e.g. @ref ModelData::constrainedNodes, per-element values in
 * @ref ModelData::material, @ref ModelData::blending or the skinning of meshes embedded in this
 * one. Reordering invalidates such data.
 */
void ReorderForLocality(MeshData& mesh, Error& error);

} // namespace mochi::model
