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

#include <mochi_core/ai/mlp.h>
#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/articulated_body/articulated_body_params.h>
#include <mochi_core/contact/contact_samples_bsh.h>
#include <mochi_core/geometry/any_shape.h>
#include <mochi_core/geometry/deep_flow_map.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/grid_sdf.h>
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/materials/material_params.h>
#include <mochi_core/utils/blending.h>
#include <mochi_core/utils/dskinning.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/mesh_embedding.h>
#include <mochi_core/utils/no_copy.h>
#include <mochi_physics/mochi_physics.h>

#include <marl/mutex.h> // for mochi::TaskMutex

#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace mochi {

struct RodSurfaceEmbeddingData;
struct BlendingDataMap;

// Base class for shapes
class Shape {
 public:
  Shape() = default;
  virtual ~Shape() = default;

  // Return a bounding volume that contains the shape's geometry.
  virtual AnyShape GetBoundingVolume(Error& error) const = 0;

  // Return the volume of the shape, if possible.
  virtual std::optional<real> GetVolume() const = 0;

  // Return the centroid (center-of-mass assuming uniform density), if possible.
  virtual std::optional<Real3> GetCentroid() const = 0;

  // Return this shape's raw triangular surface mesh, or nullptr if it doesn't have one.
  // For tetrahedral mesh shapes, the raw surface mesh is the boundary mesh in volume-mesh
  // node-index space and may contain non-boundary nodes. Use GetSurfaceMeshData() for API data
  // whose coordinate array contains only nodes referenced by surface connectivity.
  virtual std::shared_ptr<TriangularMesh const> const& GetSurfaceMesh() const {
    static std::shared_ptr<TriangularMesh const> const kEmpty;
    return kEmpty;
  }

  // Return a non-owning view of this shape's surface mesh data, or an empty view if the shape has
  // no surface mesh. The returned coordinates contain exactly the nodes referenced by the surface
  // connectivity, and the connectivity is remapped into that returned coordinate array.
  MeshDataView GetSurfaceMeshData() const {
    MeshDataView view{};

    auto const& mesh = GetSurfaceMesh();
    if (mesh) {
      view.nodesPerElement = mesh->GetNumNodesPerElement();
      view.coordinates = Flatten(mesh->GetActiveNodeCoordinates());
      view.connectivity = mesh->GetActiveNodesFlatConnectivity();
    }

    return view;
  }

  // Return the per-skin-node skinning weights carried by this shape, or a null shared_ptr if
  // this shape kind does not carry skinning data.
  virtual std::shared_ptr<SkinningData const> const& GetMeshSkinning() const {
    static std::shared_ptr<SkinningData const> const kEmpty;
    return kEmpty;
  }

  // Return the per-source-shape blending map carried by this shape, or a null shared_ptr if
  // this shape kind does not carry blending data.
  virtual std::shared_ptr<BlendingDataMap const> const& GetMeshBlending() const {
    static std::shared_ptr<BlendingDataMap const> const kEmpty;
    return kEmpty;
  }

  // Populate a ModelData struct based on the derived Shape implementation, or fail with an Error.
  virtual ModelData GetModelData(Error& error) const = 0;
};

using ShapePtr = std::shared_ptr<Shape>;
using ConstShapePtr = std::shared_ptr<Shape const>;

// ECS component used to store a Shape (base class) pointer
struct CShape : NoCopy {
  CShape() = default;
  explicit CShape(ConstShapePtr const& s) : shape(s) {}

  ConstShapePtr shape;
};

class ImplicitRigidShape final : public Shape {
 public:
  ImplicitRigidShape() = default;
  explicit ImplicitRigidShape(AnyShape const& src) : shape(src) {}

  AnyShape GetBoundingVolume(Error& /*error*/) const override {
    return shape;
  }

  std::optional<real> GetVolume() const override {
    return mochi::GetVolume(shape);
  }

  std::optional<Real3> GetCentroid() const override {
    return mochi::GetBoundingSphere(shape).GetCenter();
  }

  ModelData GetModelData(Error& error) const override;

  AnyShape shape;
};

class DeepFlowShape final : public Shape {
 public:
  AnyShape GetBoundingVolume(Error& error) const override {
    MOCHI_ERROR_SET(error, "Bounding volume is not available for a deep flow shape.");
    return {};
  }

  std::optional<real> GetVolume() const override {
    return std::nullopt; // Not supported
  }

  std::optional<Real3> GetCentroid() const override {
    return std::nullopt; // Not supported
  }

  ModelData GetModelData(Error& error) const override {
    MOCHI_ERROR_SET(error, "Model data export not support for Deep Flow shapes.");
    return {};
  }

  std::shared_ptr<DeepFlow> flow;
};

struct ConstrainedNodesData {
  DynamicArray<int> constrainedNodes;

  ConstrainedNodesData() = default;
  explicit ConstrainedNodesData(DynamicArray<int> indices) : constrainedNodes(std::move(indices)) {}
};

// Stores a map of data which originally came from a model's BlendingData.
struct BlendingDataMap {
  // Data stored in a map, for internal use in Mochi
  std::map<DynamicString, BlendingDataTargetMesh> perSourceShapeData;
  // Data re-copied to vectors, to enable sharing spans externally
  DynamicArray<DynamicString> sourceShapes; // size = arbitrary
  DynamicArray<Span<int const>> perSourceShapeIndices; // size = numSourceShapes * numNodes * 2
  DynamicArray<Span<real const>> perSourceShapeWeights; // size = numSourceShapes * numNodes * 2

  void CopyMapToVectors() {
    int numSourceShapes = perSourceShapeData.size();
    sourceShapes.reserve(numSourceShapes);
    perSourceShapeWeights.reserve(numSourceShapes);
    perSourceShapeIndices.reserve(numSourceShapes);
    for (auto const& it : perSourceShapeData) {
      sourceShapes.emplace_back(it.first);
      perSourceShapeIndices.emplace_back((it.second).indices);
      perSourceShapeWeights.emplace_back((it.second).weights);
    }
  }
};

struct LinearRomData {
  bool needsRigidTransformLayer = false;
  RowMatrix<real> basis;
};

struct PolynomialCromData {
  int order = std::numeric_limits<int>::max();
  bool needsRigidTransformLayer = false;
};

struct NeuralNetCromData {
  bool needsRigidTransformLayer = false;
  int latentStateSize = 0;
  std::optional<ai::Mlp<real>> encoder; // Optional encoder network
  std::optional<ColumnVector<real>> meanAndStdevForInputStandardize = {};
  ai::Mlp<real> decoder; // Required decoder network
  ColumnVector<real> meanAndStdevForOutputInverseStandardize = {};
  ColumnVector<real> latentStateForZeroDisplacement = {};
};

// Once we have a larger variety of ROMs, we will replace this with an appropriate variant type.
using RomDataVariant = std::variant<LinearRomData, PolynomialCromData, NeuralNetCromData>;
struct RomData : public RomDataVariant {
  using variant::variant;

  [[nodiscard]] RomDataVariant& GetVariant() {
    return *this;
  }

  [[nodiscard]] RomDataVariant const& GetVariant() const {
    return *this;
  }

  bool NeedsRigidTransformLayer() const {
    return std::visit([](auto const& item) { return item.needsRigidTransformLayer; }, GetVariant());
  }
};

struct SampleMeshInfo {
  DynamicArray<int> volumeElements;
  DynamicArray<real> volumeElementWeights;
  DynamicArray<int> boundaryElements;
  DynamicArray<real> boundaryElementWeights;
};

// Base class for shapes that have a GridSdf.
class GridSdfShape : public Shape {
 public:
  GridSdfShape() = default;
  explicit GridSdfShape(std::shared_ptr<GridSdf const> gridSdf = {});
  ~GridSdfShape() override;
  GridSdfShape(GridSdfShape const&) = delete;
  GridSdfShape& operator=(GridSdfShape const&) = delete;
  GridSdfShape(GridSdfShape&&) = delete;
  GridSdfShape& operator=(GridSdfShape&&) = delete;

  // Return this shape's GridSdf if it has one.
  MOCHI_API std::shared_ptr<GridSdf const> GetGridSdf() const;

  // If this shape already has a GridSdf, then return it.
  // Else if this shape does not support GridSdf, then set an error and return nullptr.
  // Else if a GridSdf can be computed, then start an async task (if not already started) and
  // return nullptr with (*outIsPending = true). To get the result of the async task, call
  // GetGridSdfSemaphore().Wait(), then GetGridSdf().
  MOCHI_API std::shared_ptr<GridSdf const> RequestGridSdf(
      GridSdfParams const& params,
      bool* outIsPending) const;

  // If you RequestGridSdf starts an async job, then you can use this semaphore to wait for
  // completion (see RequestGridSdf above).
  TaskSemaphore GetGridSdfSemaphore() const {
    return _gridSdfSem;
  }

 private:
  mutable TaskMutex _gridSdfMutex;
  mutable TaskSemaphore _gridSdfSem;
  mutable std::shared_ptr<GridSdf const> _gridSdf;
};

// Describes a soft tetrahedral mesh. Can be shared by actors.
class TetrahedralMeshShape final : public GridSdfShape {
 public:
  explicit TetrahedralMeshShape(
      std::shared_ptr<TetrahedralMesh const> mesh,
      std::shared_ptr<SkinningData const> meshSkinningData = {},
      std::shared_ptr<ConstrainedNodesData const> constrainedNodesData = {},
      std::shared_ptr<BlendingDataMap const> meshBlendingData = {},
      std::shared_ptr<TriangularMesh const> visualMesh = {},
      std::shared_ptr<MeshEmbedding const> visualEmbedding = {},
      std::shared_ptr<GridSdf const> gridSdf = {},
      std::unordered_map<std::string, RomData> roms = {},
      std::unordered_map<std::string, SampleMeshInfo> sampleMeshes = {},
      std::unordered_map<std::string, ContactSamplesBsh> bshs = {},
      std::shared_ptr<PerElementSoftMaterialData const> materialParamsField = {})
      : GridSdfShape(std::move(gridSdf)),
        _mesh(std::move(mesh)),
        _meshSkinningData(std::move(meshSkinningData)),
        _constrainedNodesData(std::move(constrainedNodesData)),
        _meshBlendingData(std::move(meshBlendingData)),
        _visualMesh(std::move(visualMesh)),
        _visualEmbedding(std::move(visualEmbedding)),
        _romData(std::move(roms)),
        _sampleMeshes(std::move(sampleMeshes)),
        _bshs(std::move(bshs)),
        _materialParamsField(std::move(materialParamsField)) {}

  AnyShape GetBoundingVolume(Error& error) const override {
    MOCHI_ERROR_RETURN(error, {});
    return AnyShape{mochi::GetObb(_mesh->GetAabb())}; // Obb because it can be transformed cleanly
  }

  std::optional<real> GetVolume() const override {
    return _mesh->GetTotalMeasure();
  }

  std::optional<Real3> GetCentroid() const override {
    return _mesh->GetBarycenter();
  }

  std::shared_ptr<TetrahedralMesh const> const& GetMesh() const {
    return _mesh;
  }

  std::shared_ptr<SkinningData const> const& GetMeshSkinning() const override {
    return _meshSkinningData;
  }

  std::shared_ptr<ConstrainedNodesData const> const& GetMeshConstrainedNodes() const {
    return _constrainedNodesData;
  }

  std::shared_ptr<BlendingDataMap const> const& GetMeshBlending() const override {
    return _meshBlendingData;
  }

  std::shared_ptr<TriangularMesh const> const& GetSurfaceMesh() const override {
    return _mesh->GetBoundaryMesh();
  }

  std::shared_ptr<TriangularMesh const> const& GetVisualMesh() const {
    return _visualMesh;
  }

  std::shared_ptr<MeshEmbedding const> const& GetVisualEmbedding() const {
    return _visualEmbedding;
  }

  std::shared_ptr<PerElementSoftMaterialData const> const& GetSoftMaterialParamsField() const {
    return _materialParamsField;
  }

  std::unordered_map<std::string, RomData> const& GetRomData() const {
    return _romData;
  }

  std::unordered_map<std::string, SampleMeshInfo> const& GetSampleMeshes() const {
    return _sampleMeshes;
  }

  std::unordered_map<std::string, ContactSamplesBsh> const& GetBoundingSphereHierarchies() const {
    return _bshs;
  }

  ModelData GetModelData(Error& error) const override;

 private:
  std::shared_ptr<TetrahedralMesh const> _mesh;
  std::shared_ptr<SkinningData const> _meshSkinningData;
  std::shared_ptr<ConstrainedNodesData const> _constrainedNodesData;
  std::shared_ptr<BlendingDataMap const> _meshBlendingData;
  std::shared_ptr<TriangularMesh const> _visualMesh;
  std::shared_ptr<MeshEmbedding const> _visualEmbedding;
  std::unordered_map<std::string, RomData> _romData;
  std::unordered_map<std::string, SampleMeshInfo> _sampleMeshes;
  std::unordered_map<std::string, ContactSamplesBsh> _bshs;
  std::shared_ptr<PerElementSoftMaterialData const> _materialParamsField;
};

// Describes a triangular mesh. Can be shared by actors.
class TriangularMeshShape final : public GridSdfShape {
 public:
  explicit TriangularMeshShape(
      std::unique_ptr<TriangularMesh> mesh,
      std::shared_ptr<SkinningData const> meshSkinningData = {},
      std::unique_ptr<ConstrainedNodesData> constrainedNodesData = {},
      std::shared_ptr<BlendingDataMap const> meshBlendingData = {},
      std::unique_ptr<TriangularMesh> visualMesh = {},
      std::unique_ptr<MeshEmbedding> visualEmbedding = {},
      std::shared_ptr<GridSdf> gridSdf = {})
      : GridSdfShape(std::move(gridSdf)),
        _mesh(std::move(mesh)),
        _meshSkinningData(std::move(meshSkinningData)),
        _constrainedNodesData(std::move(constrainedNodesData)),
        _meshBlendingData(std::move(meshBlendingData)),
        _visualMesh(std::move(visualMesh)),
        _visualEmbedding(std::move(visualEmbedding)) {}

  AnyShape GetBoundingVolume(Error& error) const override {
    MOCHI_ERROR_RETURN(error, {});
    return AnyShape{mochi::GetObb(_mesh->GetAabb())}; // Obb because it can be transformed cleanly
  }

  std::optional<real> GetVolume() const override {
    // TODO: This is information that could be precomputed and stored. If it were, then each rigid
    // actor using this shape would be able to reuse the information instead of recalculating it.
    return std::nullopt;
  }

  std::optional<Real3> GetCentroid() const override {
    // TODO: This is information that could be precomputed and stored. If it were, then each rigid
    // actor using this shape would be able to reuse the information instead of recalculating it.
    return std::nullopt;
  }

  std::shared_ptr<TriangularMesh const> const& GetMesh() const {
    return _mesh;
  }

  std::shared_ptr<SkinningData const> const& GetMeshSkinning() const override {
    return _meshSkinningData;
  }

  std::shared_ptr<ConstrainedNodesData const> const& GetMeshConstrainedNodes() const {
    return _constrainedNodesData;
  }

  std::shared_ptr<BlendingDataMap const> const& GetMeshBlending() const override {
    return _meshBlendingData;
  }

  std::shared_ptr<TriangularMesh const> const& GetSurfaceMesh() const override {
    return _mesh;
  }

  std::shared_ptr<TriangularMesh const> const& GetVisualMesh() const {
    return _visualMesh;
  }

  std::shared_ptr<MeshEmbedding const> const& GetVisualEmbedding() const {
    return _visualEmbedding;
  }

  ModelData GetModelData(Error& error) const override;

 private:
  std::shared_ptr<TriangularMesh const> _mesh;
  std::shared_ptr<SkinningData const> _meshSkinningData;
  std::shared_ptr<ConstrainedNodesData const> _constrainedNodesData;
  std::shared_ptr<BlendingDataMap const> _meshBlendingData;
  std::shared_ptr<TriangularMesh const> _visualMesh;
  std::shared_ptr<MeshEmbedding const> _visualEmbedding;
};

// Describes a polyline mesh (used for rod actor geometry). Can be shared by actors.
class PolylineShape final : public Shape {
 public:
  explicit PolylineShape(
      DynamicArray<Real3> nodes,
      DynamicArray<Real3> elementFrameAxes,
      bool isClosedLoop);

  PolylineShape(
      DynamicArray<Real3> nodes,
      DynamicArray<Real3> elementFrameAxes,
      std::shared_ptr<TriangularMesh const> visualMesh,
      std::shared_ptr<RodSurfaceEmbeddingData const> rodVisualEmbedding,
      bool isClosedLoop);

  AnyShape GetBoundingVolume(Error& error) const override {
    MOCHI_ERROR_RETURN(error, {});
    return AnyShape{mochi::GetObb(CalcAabb(MakeSpan(_nodes)))}; // Obb for clean transforms
  }

  std::optional<real> GetVolume() const override {
    return std::nullopt;
  }

  std::optional<Real3> GetCentroid() const override {
    return std::nullopt;
  }

  DynamicArray<Real3> const& GetNodes() const {
    return _nodes;
  }

  DynamicArray<Real3> const& GetElementFrameAxes() const {
    return _elementFrameAxes;
  }

  bool IsClosedLoop() const {
    return _isClosedLoop;
  }

  std::shared_ptr<TriangularMesh const> const& GetVisualMesh() const {
    return _visualMesh;
  }

  std::shared_ptr<RodSurfaceEmbeddingData const> const& GetRodVisualEmbedding() const {
    return _rodVisualEmbedding;
  }

  /// Returns the polyline's flat connectivity array. For an open polyline this is
  /// 2*(numNodes-1) entries [0,1, 1,2, ..., numNodes-2, numNodes-1]; for a closed-loop
  /// polyline a final wrap-around segment [numNodes-1, 0] is appended.
  Span<int const> GetFlatConnectivity() const {
    return MakeConstSpan(_connectivity);
  }

  ModelData GetModelData(Error& error) const override;

 private:
  DynamicArray<Real3> _nodes;
  DynamicArray<Real3> _elementFrameAxes;
  bool _isClosedLoop = false;
  DynamicArray<int> _connectivity;
  std::shared_ptr<TriangularMesh const> _visualMesh;
  std::shared_ptr<RodSurfaceEmbeddingData const> _rodVisualEmbedding;
};

// Define data structure for bones
struct BoneData {
  // Rest pose of each link's local frame in the shape's root frame. Cached forward-kinematics
  // product of parentLinkFromJoint * jointFromChildLink up the parent chain (see JointsData).
  std::vector<TransformRT> restRootFromBone;
  DynamicArray<DynamicString> boneNames;
};

// Define data structure for joints using only base data types
struct JointsData {
  std::vector<ArticulatedJointType> jointTypes;
  std::vector<ArticulatedCycleJoint> cycles; // Possibly empty
  std::vector<Real3> jointAxes;
  DynamicArray<ArticulatedDofInfo> dofInfo;
  std::vector<TransformRT> jointFromChildLink;
  std::vector<TransformRT> parentLinkFromJoint;
  std::vector<Real3> jointMinLimits; // Possibly empty
  std::vector<Real3> jointMaxLimits; // Possibly empty
  DynamicArray<DynamicString> jointNames;
  std::vector<int> jointsParentLinks;
  std::vector<int> jointsChildLinks;
};

class ArticulatedBodyShape final : public Shape {
 public:
  explicit ArticulatedBodyShape(ArticulatedActorParams const& params, Error& error);

  AnyShape GetBoundingVolume(Error& error) const override {
    MOCHI_ERROR_SET(
        error,
        "Bounding volume is not available for an articulated shape (independent of the link meshes).");
    return {};
  }

  std::optional<real> GetVolume() const override {
    return std::nullopt; // Not supported
  }

  std::optional<Real3> GetCentroid() const override {
    return std::nullopt; // Not supported
  }

  int GetNumBones() const {
    return isize(_boneData->restRootFromBone);
  }

  int GetNumJoints() const {
    return isize(_jointsData->jointTypes);
  }

  BoneData const* GetBoneData() const {
    return _boneData.get();
  }

  articulated::ParentIndexArray const* GetBoneParents() const {
    return _boneParents.get();
  }

  JointsData const* GetJointsData() const {
    return _jointsData.get();
  }

  ArticulatedShapeInfo GetArticulatedShapeInfo() const;

  ModelData GetModelData(Error& error) const override {
    MOCHI_ERROR_SET(error, "Model data export not supported for articulated shapes.");
    return {};
  }

 private:
  std::unique_ptr<BoneData> _boneData;
  std::unique_ptr<articulated::ParentIndexArray> _boneParents;
  std::unique_ptr<JointsData> _jointsData;
};

// Component to store ArticulatedBodyShape of an articulated body
struct CArticulatedBodyShape {
  std::shared_ptr<ArticulatedBodyShape const> shape;
};

} // namespace mochi
