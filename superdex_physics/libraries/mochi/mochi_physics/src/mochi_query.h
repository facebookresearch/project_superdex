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

#include "mochi_ecs.h"
#include "mochi_scene_recorder.h"

#include <mochi_core/contact/contact_utils.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_physics/mochi_physics.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mochi {

/**********************************************************************
  ECS Components
*/

// Global to the mochi::Scene. Used to allocate QueryHandles.
struct CQueryHandleAllocator : NoCopy {
  // Combine the QueryType with a new ID to form a new QueryHandle
  QueryHandle NewHandleThreadSafe(QueryType type);

  // Next unique ID to return to the user as a QueryHandle.
  // std::atomic is used so that AsyncScene can pre-allocate handles from another thread.
  // std::unique_ptr is used so that the ECS component can be move constructed/assigned.
  std::unique_ptr<std::atomic<uint32_t>> nextId = std::make_unique<std::atomic<uint32_t>>(0);
};

// Stores the QueryHandles currently registered with an actor
struct CActiveQuerySet : NoCopy {
  std::unordered_set<QueryHandle::ValueType> handles;
};

// Stores the quadrature point positions in the reference configuration
// and in world space. The order of the quadrature points follows the order
// of the elements in the mesh. Namely, the zeroth quadrature point is the first
// quadrature point of the 0th element and the last quadrature point is the last
// quadrature point of the last element. Thus the jth quadrature point of the
// nth element (all assumed having nQuads quadrature point) will be store at
// n*nQuads*spaceDim + j*spaceDim + i with i=0,1,2 for the spatial dims.
struct CQueryQuadraturePointsPosition : RefCounted {
  std::vector<real> quadraturePointsReferencePosition;
  std::vector<real> quadraturePointsWorldPosition;
};

// Indicates that node positions should be calculated and stored for Actor::GetNodePositionsLocal().
struct CQueryNodePositions : RefCounted {
  std::vector<real> nodePositions; // 3 per node
};

// Indicates that element deformation gradients should be calculated and stored for
//  Actor::GetElementDeformationGradient().
struct CQueryElementsDeformationGradient : RefCounted {
  std::vector<real> elementsDeformationGradient; // 9 per element ordered as F11, F12, F13, F21, ...
};

// Indicates that node positions should be calculated and stored for
// Actor::GetSurfaceMeshNodePositionsLocal().
struct CQuerySurfaceNodePositions : RefCounted {
  std::vector<real> nodePositions; // 3 per node
};

// Indicates that node positions should be calculated and stored for Actor::GetSurfaceNodeNormals().
struct CQuerySurfaceNodeNormals : RefCounted {
  std::vector<real> nodeNormals; // 3 per node
  std::vector<Vec4r> faceCrossProducts; // Used to calculate nodeNormals.
};

// Indicates that the visual node positions should be calculated and stored for
// Actor::GetVisualMeshNodePositionsLocal()
struct CQueryVisualNodePositions : RefCounted {
  std::vector<real> nodePositions;
};

// Indicates that the visual node normals should be calculated and stored for
// Actor::GetVisualMeshNodeNormalsLocal()
struct CQueryVisualNodeNormals : RefCounted {
  std::vector<real> nodeNormals;
};

// Indicates that we should calculate the total elastic energy of a soft actor. Enabling this
// feature requires significant additional computation.
struct CQueryElasticEnergy : RefCounted {
  real energy = 0_r; // Energy in the current deformed state
  real energyAtRest = 0_r; // Energy at rest (must be set once by the actor)
  bool isEnergyAtRestInitialized = false;
};

// Indicates that the actor's SDF should be sampled to produce an arbitrary distribution of
// of positions and normals along the SDF surface (~zero distance).
struct CQuerySdfSurface : RefCounted {
  std::vector<Real3> positions; // local-space
  std::vector<Real3> normals;
};

// Indicates that actor's potential contact samples should be computed and stored.
struct CQueryContactSamples : RefCounted {
  std::vector<real> contactSamples; // 3 * num contact samples
};

// Helper component to handle alternatively CQueryContactPoints, CQueryNodeContactForces and/or
// CQueryActorContactForces
struct TagQueryActiveContacts : RefCounted {};

// Indicates that actor's current active collision points should be computed and stored.
struct CQueryContactPoints : RefCounted {
  std::vector<ContactPoint> contactPoints;
  bool isInitialized = false; // Set to true when query data is populated
};

// A lightweight version of CQueryContactPoints that is specifically used to evaluate the SDF
// at long distances. Since CRomRequiresFarSdfEvaluation is incompatible with the
// CQueryContactPoints query, one must use this query instead.
// The frontend API equivalent for this component is SdfDistances
struct CQuerySdfDistances : RefCounted {
  bool isInitialized = false; // Set to true when query data is populated
  std::vector<int> sampleIndices;
  std::vector<real> distances;
  std::vector<Real3> worldPositions;
  // Gradients in world space. Same size as worldPositions.
  std::vector<Real3> distanceGrads;
  // Only used if the actor has far SDF evaluation enabled.
  // Holds the extra padding added to the SDF evaluation distance.
  // Same units as distances.
  real maxSdfFarDistanceEvaluation = 0.0_r;

  void Clear() {
    sampleIndices.clear();
    distances.clear();
    worldPositions.clear();
    distanceGrads.clear();
  }

  void Reserve(std::size_t capacity) {
    sampleIndices.reserve(capacity);
    distances.reserve(capacity);
    worldPositions.reserve(capacity);
    distanceGrads.reserve(capacity);
  }
};

// Indicates that actor's node contact forces should be computed and stored.
struct CQueryNodeContactForces : RefCounted {
  std::vector<NodeContactForce> nodeContactForces;
  // This could be local to UpdateQueryActiveContactsWorldSpace(), but it's made
  // a member to minimize memory reallocation.
  std::unordered_map<int, Real3> nodeContactForcesMap;
  bool isInitialized = false; // Set to true when query data is populated

  void FinalizeContactForces() {
    nodeContactForces.resize(nodeContactForcesMap.size());
    std::transform(
        nodeContactForcesMap.cbegin(),
        nodeContactForcesMap.cend(),
        nodeContactForces.begin(),
        [](auto const& pair) { return NodeContactForce{pair.first, pair.second}; });
    isInitialized = true;
  }

  void AddContactForce(Real3 force, Int3 nodeIndices, Real3 nodeWeights) {
    for (int i = 0; i < 3; i++) {
      auto nodeForce = nodeWeights[i] * force;
      auto nodeTotalForce = nodeContactForcesMap.find(nodeIndices[i]);
      if (nodeTotalForce == nodeContactForcesMap.end()) {
        nodeContactForcesMap[nodeIndices[i]] = nodeForce;
      } else {
        nodeTotalForce->second += nodeForce;
      }
    }
  }
};

// Indicates that pair-wise actor contact forces should be computed and stored.
// Force and torque are expressed in world space.
class CQueryActorContactForces : public RefCounted {
 private:
  struct ForceAndTorque {
    entt::entity e = {};
    Vec4r force = {};
    Vec4r torque = {};
  };

  DynamicArray<ForceAndTorque> _entries;

  MOCHI_FORCE_INLINE auto Find(entt::entity actor) const {
    return std::find_if(
        _entries.begin(), _entries.end(), [actor](auto const& entry) { return entry.e == actor; });
  }

  MOCHI_FORCE_INLINE auto Find(entt::entity actor) {
    return std::find_if(
        _entries.begin(), _entries.end(), [actor](auto const& entry) { return entry.e == actor; });
  }

 public:
  bool isInitialized = false; // Set to true when query data is populated

  MOCHI_FORCE_INLINE void Reserve(int numActors) {
    _entries.reserve(numActors);
  }

  MOCHI_FORCE_INLINE void Clear() {
    _entries.clear();
  }

  MOCHI_FORCE_INLINE void Add(entt::entity actor, Vec4r const& force, Vec4r const& torque) {
    auto* it = Find(actor);
    if (it != _entries.end()) {
      it->force += force;
      it->torque += torque;
    } else {
      _entries.push_back(ForceAndTorque{.e = actor, .force = force, .torque = torque});
    }
  }

  MOCHI_FORCE_INLINE Vec4r GetForce(entt::entity actor) const {
    auto const* it = Find(actor);
    return it != _entries.end() ? it->force : Vec4r{};
  }

  MOCHI_FORCE_INLINE Vec4r GetTorque(entt::entity actor) const {
    auto const* it = Find(actor);
    return it != _entries.end() ? it->torque : Vec4r{};
  }

  MOCHI_FORCE_INLINE Vec4r GetTotalForce() const {
    Vec4r force = {};
    for (auto const& entry : _entries) {
      force += entry.force;
    }
    return force;
  }

  MOCHI_FORCE_INLINE Vec4r GetTotalTorque() const {
    Vec4r torque = {};
    for (auto const& entry : _entries) {
      torque += entry.torque;
    }
    return torque;
  }
};

// Indicates that we should calculate the force exerted by a constraint.
struct CQueryConstraintForce : RefCounted {
  ColumnVector<real> force;
};

// Indicates that we should calculate the force exerted by each DOF of an articulated pose
// controller.
struct CQueryArticulatedControllerForce : RefCounted {
  ColumnVector<real> force;
};

/**********************************************************************
  QueryHandle Utilities
*/
inline QueryHandle CreateQueryHandle(QueryType type, uint32_t id) {
  uint64_t value = (static_cast<uint64_t>(type) << 32) | static_cast<uint64_t>(id);
  return QueryHandle{value};
}

inline QueryType GetQueryType(QueryHandle handle) {
  return static_cast<QueryType>(handle.value >> 32);
}

inline uint32_t GetQueryId(QueryHandle handle) {
  return static_cast<uint32_t>(handle.value);
}

/**********************************************************************
  Query Registration
*/

// Sets an error if the query type is not supported for this entity.
void SetErrorIfQueryNotSupported(
    entt::registry const& reg,
    entt::entity e,
    QueryType type,
    Error& error);

// Register a new query by QueryType
QueryHandle RegisterQuery(
    entt::registry& reg,
    entt::entity e,
    QueryType type,
    bool computeImmediately,
    Error& error);

// Register a new query using a QueryHandle that was already allocated.
// Used to support registration via AsyncScene.
void RegisterQuery(
    entt::registry& reg,
    entt::entity e,
    QueryHandle preallocatedHandle,
    bool computeImmediately,
    Error& error);

// Cancel a previous query by QueryHandle
void CancelQuery(entt::registry& reg, entt::entity e, QueryHandle handle);

// Set an error if the QueryHandle is NOT currently registered with the entity's CActiveQuerySet
void ValidateQueryHandle(
    entt::registry const& reg,
    entt::entity e,
    QueryHandle handle,
    Error& error);

// Call AddRemoveOrRefComponents for the component type(s) required to implement the specified
// QueryType.
void AddRemoveOrRefComponentsForQuery(
    entt::registry& reg,
    entt::entity e,
    QueryType type,
    bool add,
    bool computeImmediately = false); // namespace mochi

/**********************************************************************
  Query Recording
*/

// A system to record per-actor contact-point query data
void RecordQueryContactPoints(CQueryContactPoints const& contacts, CRecordingData& outData);

// A system to record per-actor node-contact-force query data
void RecordQueryNodeContactForces(CQueryNodeContactForces const& forces, CRecordingData& outData);

// A system to record per-actor SDF query data on the actor's surface
void RecordQuerySdfDistances(CQuerySdfDistances const& distances, CRecordingData& outData);

namespace query {
void InitializeOnce(entt::registry& reg);
} // namespace query

} // namespace mochi
