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

#include "mochi_shape.h"
#include "mochi_ecs_registry.h"
#include "mochi_rod.h"

#include <mochi_core/geometry/mesh_data_utils.h>
#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/overload_visitor.h>
#include <mochi_core/utils/transform_rt.h>

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

using namespace mochi;

ModelData ImplicitRigidShape::GetModelData(Error& error) const {
  MOCHI_ERROR_RETURN(error, {});
  ModelData outData;
  std::visit(
      OverloadVisitor{
          [&](Aabb const& aabb) { outData.box = Box{aabb.GetCenter(), aabb.GetHalfExtents()}; },
          [&](Sphere const& sphere) { outData.sphere = sphere; },
          [&](Obb const& obb) {
            outData.box =
                Box{obb.GetCenter(), obb.GetHalfExtents(), QuaternionFromMatrix(obb.GetRotation())};
          },
          [&](Plane const& plane) { outData.plane = plane; }},
      shape);
  return outData;
}

GridSdfShape::GridSdfShape(std::shared_ptr<GridSdf const> gridSdf) : _gridSdf(std::move(gridSdf)) {}

GridSdfShape::~GridSdfShape() {
  // Wait in case an async task is still running.
  _gridSdfSem.Wait();
}

std::shared_ptr<GridSdf const> GridSdfShape::GetGridSdf() const {
  std::lock_guard lock(_gridSdfMutex);
  return _gridSdf;
}

std::shared_ptr<GridSdf const> GridSdfShape::RequestGridSdf(
    GridSdfParams const& params,
    bool* outIsPending) const {
  *outIsPending = false;
  bool shouldSchedule = false;

  // Synchronize
  {
    std::lock_guard lock(_gridSdfMutex);
    if (_gridSdf) {
      // GridSdf was already computed or loaded from file
      return _gridSdf;
    } else if (!_gridSdfSem.IsDone()) {
      // Someone else already started the async task
      *outIsPending = true;
      return {};
    } else {
      // We will start the async task for ourselves.
      *outIsPending = true;
      shouldSchedule = true;
      _gridSdfSem.Add(1);
    }
  }

  // Start an async task if necessary
  if (shouldSchedule) {
    // Snapshot the surface mesh before scheduling: ~GridSdfShape waits for this task only in its
    // base destructor, by which point the derived shape's mesh members are already destroyed.
    std::shared_ptr<TriangularMesh const> surfaceMesh = GetSurfaceMesh();
    MOCHI_ASSERT(surfaceMesh != nullptr, "Every derived shape class should have one");
    Schedule(
        "GridSdf",
        [this,
         surfaceMesh = std::move(surfaceMesh),
         params,
         /* Must capture sem by value */ sem = _gridSdfSem]() {
          // Safe to reach base members via `this`: ~GridSdfShape() blocks in _gridSdfSem.Wait()
          // until sem.Done() below, so the base subobject outlives every use of `this` here.
          // Keep sem.Done() last — the base may be freed the moment it runs. Derived members
          // do not survive, hence surfaceMesh is captured by value above.
          Error error;
          auto newGridSdf = std::make_shared<GridSdf>(surfaceMesh, params, error);
          if (error.IsOK()) {
            std::lock_guard lock(_gridSdfMutex);
            _gridSdf = std::move(newGridSdf);
          } else {
            MOCHI_LOG_ERROR("Failed to create GridSdf: %s", error.ToString().c_str());
          }
          sem.Done();
        });
  }

  return {};
}

template <typename ShapeT>
ModelData GetModelDataImpl(ShapeT const& shape, Error& error) {
  bool constexpr kIsTetShape = std::is_same_v<ShapeT, TetrahedralMeshShape>;
  bool constexpr kIsTriShape = std::is_same_v<ShapeT, TriangularMeshShape>;
  static_assert(kIsTetShape || kIsTriShape);

  MOCHI_ERROR_RETURN(error, {});

  ModelData outData;
  outData.mesh.emplace();
  auto const mesh = shape.GetMesh();
  outData.mesh->nodesPerElement = kIsTetShape ? 4 : 3;
  outData.mesh->coordinates = Flatten(mesh->GetNodeCoordinates());
  outData.mesh->connectivity = mesh->GetFlatConnectivity();

  if (auto const meshSkinningData = shape.GetMeshSkinning()) {
    outData.mesh->skinning = *meshSkinningData;
  }

  if (auto const visualMesh = shape.GetVisualMesh()) {
    outData.visualMesh.emplace();
    outData.visualMesh->nodesPerElement = 3;
    outData.visualMesh->coordinates = Flatten(visualMesh->GetNodeCoordinates());
    outData.visualMesh->connectivity = visualMesh->GetFlatConnectivity();
    if (auto const* linearEmbedding =
            dynamic_cast<LinearMeshEmbedding const*>(shape.GetVisualEmbedding().get())) {
      outData.visualMesh->skinning.emplace();
      outData.visualMesh->skinning->weightsPerNode =
          static_cast<int>(linearEmbedding->GetNumSkinningWeightsPerEntry());
      outData.visualMesh->skinning->indices = linearEmbedding->GetIndices();
      outData.visualMesh->skinning->weights = linearEmbedding->GetWeights();
    }
  }

  if (auto const blending = shape.GetMeshBlending()) {
    MOCHI_ASSERT(blending->sourceShapes.size() == blending->perSourceShapeIndices.size());
    MOCHI_ASSERT(blending->sourceShapes.size() == blending->perSourceShapeWeights.size());
    outData.blending.emplace();
    outData.blending->resize(blending->sourceShapes.size());
    for (int i = 0; i < isize(blending->sourceShapes); ++i) {
      (*outData.blending)[i].sourceShape = DynamicString{blending->sourceShapes[i]};
      (*outData.blending)[i].indices = blending->perSourceShapeIndices[i];
      (*outData.blending)[i].weights = blending->perSourceShapeWeights[i];
    }
  }

  if (auto const constrainedNodesData = shape.GetMeshConstrainedNodes()) {
    outData.constrainedNodes = constrainedNodesData->constrainedNodes;
  }

  if constexpr (kIsTetShape) {
    if (auto const materialParamsField = shape.GetSoftMaterialParamsField()) {
      outData.material = *materialParamsField;
    }
  }

  shape.GetGridSdfSemaphore().Wait(); // In case it was pending
  if (auto const sdf = shape.GetGridSdf()) {
    outData.sdf.emplace();
    sdf->GetGridSdfData(*outData.sdf);
  }

  return outData;
}

ModelData TetrahedralMeshShape::GetModelData(Error& error) const {
  return GetModelDataImpl(*this, error);
}

ModelData TriangularMeshShape::GetModelData(Error& error) const {
  return GetModelDataImpl(*this, error);
}

template <typename SpanT>
static bool IsSizeValid(SpanT const& span, int size) {
  return span.empty() || isize(span) == size;
}

static std::optional<DynamicArray<DynamicString>>
FillWithNoDuplicates(int size, std::string_view prefix, Span<DynamicString const> names) {
  // If the names are empty, just fill them as prefix_0, prefix_1, ...
  if (names.empty()) {
    DynamicArray<DynamicString> result;
    result.reserve(size);
    for (int i = 0; i < size; ++i) {
      result.emplace_back(std::string(prefix) + "_" + std::to_string(i));
    }
    return result;
  }

  MOCHI_ASSERT(names.size() == size, "Wrong size");

  // Ensure there are no empty strings
  bool modified = false;
  DynamicArray<DynamicString> result = {names.begin(), names.end()};
  for (int i = 0; i < size; ++i) {
    if (result[i].empty()) {
      modified = true;
      result[i] = DynamicString(prefix);
    }
  }

  // Ensure there are no duplicates
  // NOTE: We use std::string as the map key (not std::string_view) because modifying result[i]
  // below would invalidate any string_view pointing into the old DynamicString's memory.
  std::unordered_map<std::string, int> mapToPos;
  std::unordered_map<std::string, int> mapToCount;
  for (int i = 0; i < size; ++i) {
    std::string key{result[i]};
    if (auto it = mapToPos.find(key); it != mapToPos.end()) {
      modified = true;
      int const firstPos = it->second;
      int& count = mapToCount[key];
      if (count == 1) {
        std::string firstName = key + "_0";
        while (mapToPos.count(firstName)) {
          firstName = key + "_" + std::to_string(count++);
        }
        result[firstPos] = DynamicString(firstName);
        mapToPos[firstName] = firstPos;
      }
      std::string newName = key + "_" + std::to_string(count);
      while (mapToPos.count(newName)) {
        newName = key + "_" + std::to_string(++count);
      }
      result[i] = DynamicString(newName);
      mapToPos[newName] = i;
      ++count;
    } else {
      mapToPos[key] = i;
      mapToCount[key] = 1;
    }
  }

  if (modified) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
    std::unordered_set<std::string> uniqueNames;
    for (auto const& name : result) {
      MOCHI_ASSERT_VERBOSE(
          uniqueNames.insert(std::string(name)).second,
          "FillWithNoDuplicates produced duplicate names.");
    }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
    return result;
  } else {
    return std::nullopt;
  }
}

ArticulatedBodyShape::ArticulatedBodyShape(ArticulatedActorParams const& params, Error& error) {
  MOCHI_ERROR_RETURN(error);

  int const numLinks = isize(params.links);
  int const numCycles = isize(params.cycles);
  MOCHI_ERROR_IF(numLinks == 0, error, "Articulated shape requires at least one link");
  MOCHI_ERROR_IF(
      isize(params.joints) != numLinks, error, "Expected an equal number of joints and links.");
  MOCHI_ERROR_RETURN(error);

  // Flatten the articulated actor params into the internal joint/link arrays. Cycle joints are
  // appended after the per-link joints, so the joint arrays hold numLinks + numCycles entries.
  int const numJoints = numLinks + numCycles;
  std::vector<ArticulatedCycleJoint> cycles(numCycles);
  std::vector<ArticulatedJointType> jointTypes(numJoints);
  std::vector<Real3> jointAxes(numJoints);
  std::vector<TransformRT> jointFromChildLink(numJoints);
  std::vector<TransformRT> parentLinkFromJoint(numJoints);
  std::vector<Real3> jointMinLimits(numJoints);
  std::vector<Real3> jointMaxLimits(numJoints);
  std::vector<DynamicString> jointNamesStorage(numJoints);
  std::vector<TransformRT> rootFromLinksAtRest(numLinks);
  std::vector<int> parents(numLinks);
  std::vector<DynamicString> linkNamesStorage(numLinks);

  for (int i = 0; i < numLinks; ++i) {
    auto const& joint = params.joints[i];
    jointTypes[i] = joint.type;
    jointAxes[i] = joint.axis;
    jointFromChildLink[i] = params.links[i].parentJointFromLink;
    parentLinkFromJoint[i] = joint.parentLinkFromJoint;
    jointMinLimits[i] = joint.minLimit.value_or(-kInf3);
    jointMaxLimits[i] = joint.maxLimit.value_or(kInf3);
    jointNamesStorage[i] = joint.name;
  }
  for (int i = 0; i < numCycles; ++i) {
    int const idx = numLinks + i;
    auto const& cycle = params.cycles[i];
    cycles[i].parent = cycle.parentLink;
    cycles[i].child = cycle.childLink;
    jointTypes[idx] = ArticulatedJointType::Cycle;
    jointAxes[idx] = {};
    jointFromChildLink[idx] = cycle.jointFromChildLink;
    // Cycle joints do not define a parent-link-from-joint transform; leave it identity (unused).
    parentLinkFromJoint[idx] = TransformRT::Identity();
    jointMinLimits[idx] = -kInf3;
    jointMaxLimits[idx] = kInf3;
  }

  // The first numLinks joints must be free of cycles, and the parent id must be lower than the link
  // id. Free joints and unrooted links are allowed everywhere in the hierarchy. This is because
  // articulated actors are the underlying representation for skeletal skinning.
  for (int i = 0; i < numLinks; ++i) {
    int const parentLink = params.links[i].parentLink;
    MOCHI_ERROR_IF(jointTypes[i] == ArticulatedJointType::Cycle, error, "Wrong joint type");
    MOCHI_ERROR_IF(parentLink < -1, error, "Unsupported parent index");
    MOCHI_ERROR_IF(parentLink >= i, error, "Parent must come before child");
  }

  // Joints forming cycles must reference valid, distinct links.
  for (int i = 0; i < numCycles; ++i) {
    MOCHI_ERROR_IF(cycles[i].child < 0 || cycles[i].child >= numLinks, error, "Wrong child index");
    MOCHI_ERROR_IF(
        cycles[i].parent < 0 || cycles[i].parent >= numLinks, error, "Wrong parent index");
    MOCHI_ERROR_IF(
        cycles[i].child == cycles[i].parent, error, "Child and parent cannot be the same");
  }
  MOCHI_ERROR_IF_NOT(IsFinite(MakeConstSpan(jointAxes)), error, "Joint axes must be finite.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(MakeConstSpan(jointFromChildLink)), error, "Joint transforms must be finite.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(MakeConstSpan(parentLinkFromJoint)), error, "Joint transforms must be finite.");
  MOCHI_ERROR_RETURN(error);

  // Compute each link's rest transform relative to the root via forward kinematics. Parents are
  // guaranteed to precede their children (validated above), so the parent transform is already set.
  for (int i = 0; i < numLinks; ++i) {
    auto const& link = params.links[i];
    auto const& joint = params.joints[i];
    auto const& rootFromParentLink =
        (link.parentLink >= 0) ? rootFromLinksAtRest[link.parentLink] : TransformRT::Identity();
    rootFromLinksAtRest[i] = NormalizeRotation(
        rootFromParentLink * joint.parentLinkFromJoint * link.parentJointFromLink);
    parents[i] = link.parentLink;
    linkNamesStorage[i] = link.name;
  }
  MOCHI_ERROR_IF_NOT(
      IsFinite(MakeConstSpan(rootFromLinksAtRest)), error, "Link transforms must be finite.");
  MOCHI_ERROR_RETURN(error);

  Span<DynamicString const> jointNames = jointNamesStorage;
  Span<DynamicString const> linkNames = linkNamesStorage;

  // Ensure link and joint names are full and with no duplicates
  auto jointsNamesNew = FillWithNoDuplicates(numJoints, "joint", jointNames);
  if (jointsNamesNew.has_value()) {
    jointNames = MakeConstSpan(*jointsNamesNew);
  }
  auto linksNamesNew = FillWithNoDuplicates(numLinks, "link", linkNames);
  if (linksNamesNew.has_value()) {
    linkNames = MakeConstSpan(*linksNamesNew);
  }

  // Create joints data.
  std::vector<int> jointsChildren;
  std::vector<int> jointsParents;
  jointsChildren.reserve(numJoints);
  jointsParents.reserve(numJoints);
  for (int i = 0; i < numLinks; ++i) {
    jointsChildren.emplace_back(i);
    jointsParents.emplace_back(parents[i]);
  }
  for (auto const& cycle : cycles) {
    jointsChildren.emplace_back(cycle.child);
    jointsParents.emplace_back(cycle.parent);
  }
  DynamicArray<ArticulatedDofInfo> dofInfo = articulated::SetupJointDofs(jointTypes);
  _jointsData = std::make_unique<JointsData>(JointsData{
      std::move(jointTypes),
      std::move(cycles),
      std::move(jointAxes),
      std::move(dofInfo),
      std::move(jointFromChildLink),
      std::move(parentLinkFromJoint),
      std::move(jointMinLimits),
      std::move(jointMaxLimits),
      {jointNames.begin(), jointNames.end()},
      std::move(jointsParents),
      std::move(jointsChildren)});

  // Create bone data
  _boneData = std::make_unique<BoneData>(
      BoneData{std::move(rootFromLinksAtRest), {linkNames.begin(), linkNames.end()}});

  // Create link hierarchy data.
  _boneParents = std::make_unique<articulated::ParentIndexArray>(std::move(parents));
}

ArticulatedShapeInfo ArticulatedBodyShape::GetArticulatedShapeInfo() const {
  MOCHI_PROFILE_SCOPE();

  ArticulatedShapeInfo description;
  description.rootFromLinksAtRest = _boneData->restRootFromBone;
  description.linkNames = _boneData->boneNames;
  description.parents = *_boneParents;
  description.jointTypes = _jointsData->jointTypes;
  description.cycles = _jointsData->cycles;
  description.jointAxes = _jointsData->jointAxes;
  description.dofInfo = _jointsData->dofInfo;
  description.jointFromChildLink = _jointsData->jointFromChildLink;
  description.parentLinkFromJoint = _jointsData->parentLinkFromJoint;
  description.jointMinLimits = _jointsData->jointMinLimits;
  description.jointMaxLimits = _jointsData->jointMaxLimits;
  description.jointNames = _jointsData->jointNames;

  return description;
}

PolylineShape::PolylineShape(
    DynamicArray<Real3> nodes,
    DynamicArray<Real3> elementFrameAxes,
    bool isClosedLoop)
    : PolylineShape(std::move(nodes), std::move(elementFrameAxes), nullptr, nullptr, isClosedLoop) {
}

PolylineShape::PolylineShape(
    DynamicArray<Real3> nodes,
    DynamicArray<Real3> elementFrameAxes,
    std::shared_ptr<TriangularMesh const> visualMesh,
    std::shared_ptr<RodSurfaceEmbeddingData const> rodVisualEmbedding,
    bool isClosedLoop)
    : _nodes(std::move(nodes)),
      _elementFrameAxes(std::move(elementFrameAxes)),
      _isClosedLoop(isClosedLoop),
      _connectivity(MakeSequentialPolylineConnectivity(isize(_nodes), isClosedLoop)),
      _visualMesh(std::move(visualMesh)),
      _rodVisualEmbedding(std::move(rodVisualEmbedding)) {
  MOCHI_ASSERT(
      static_cast<bool>(_visualMesh) == static_cast<bool>(_rodVisualEmbedding),
      "Visual mesh and rod visual embedding must be both provided or both null.");
  // Validate polyline geometry defensively so direct callers of this constructor cannot trigger
  // a division by zero or undefined parallel-transport rotation in GenerateDiscreteBishopFrame
  // below. Factory paths (e.g. CreatePolylineShape, CreateShapeFromModelData) validate upstream
  // via model::Validate.
  mochi::model::ValidatePolylineGeometry(MakeConstSpan(_nodes), _isClosedLoop, ErrorAssert{});
  if (_elementFrameAxes.empty()) {
    _elementFrameAxes = mochi::GenerateDiscreteBishopFrame(_nodes, _isClosedLoop);
  } else {
    mochi::model::ValidatePolylineElementFrameAxes(
        MakeConstSpan(_elementFrameAxes), MakeConstSpan(_nodes), _isClosedLoop, ErrorAssert{});
  }
}

ModelData PolylineShape::GetModelData(Error& error) const {
  MOCHI_ERROR_RETURN(error, {});

  ModelData outData;
  outData.mesh.emplace();
  outData.mesh->nodesPerElement = 2;
  outData.mesh->coordinates = Flatten(MakeConstSpan(_nodes));
  // Closed-loop topology is encoded in the connectivity array (size 2*numNodes for closed loop,
  // 2*(numNodes-1) for open). Open polylines could omit it, but emitting it explicitly
  // makes the round-trip self-describing.
  outData.mesh->connectivity = _connectivity;

  if (!_elementFrameAxes.empty()) {
    outData.elementFrameAxes = DynamicArray<real>{Flatten(MakeConstSpan(_elementFrameAxes))};
  }

  if (_visualMesh) {
    outData.visualMesh.emplace();
    outData.visualMesh->nodesPerElement = 3;
    outData.visualMesh->coordinates = Flatten(_visualMesh->GetNodeCoordinates());
    outData.visualMesh->connectivity = _visualMesh->GetFlatConnectivity();
    outData.visualMesh->skinning.emplace();
    outData.visualMesh->skinning->weightsPerNode = _rodVisualEmbedding->weightsPerNode;
    outData.visualMesh->skinning->indices = _rodVisualEmbedding->elementIndices;
    outData.visualMesh->skinning->weights = _rodVisualEmbedding->weights;
  }

  return outData;
}

namespace mochi::shape {
// This suppresses a warning about no previous declaration for InitializeOnce.
void InitializeOnce(entt::registry& reg);

void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CShape>(reg);
  ecs::RegisterComponent<CArticulatedBodyShape>(reg);
}
} // namespace mochi::shape
