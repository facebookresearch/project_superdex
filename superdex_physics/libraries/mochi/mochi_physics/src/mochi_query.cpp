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

#include "mochi_query.h"

#include "mochi_constraint.h"
#include "mochi_contact.h"
#include "mochi_discretization_components.h"
#include "mochi_discretization_functions.h"
#include "mochi_ecs_utils.h"
#include "mochi_island.h"
#include "mochi_pose_controller.h"
#include "mochi_rod.h"
#include "mochi_soft.h"
#include "mochi_soft_skinned.h"

#include <array>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace mochi;

QueryHandle CQueryHandleAllocator::NewHandleThreadSafe(QueryType type) {
  return CreateQueryHandle(type, ++(*nextId));
}

void mochi::AddRemoveOrRefComponentsForQuery(
    entt::registry& reg,
    entt::entity e,
    QueryType type,
    bool add,
    bool computeImmediately) {
  switch (type) {
    case QueryType::NodePositions: {
      auto* component = AddRemoveOrRefComponent<CQueryNodePositions>(reg, e, add);
      if (computeImmediately && component && component->nodePositions.empty()) {
        ecs::TryInvokeOnEntity(&UpdateQueryNodePositions, reg, e); // compute results for new query
      }
    } break;

    case QueryType::ElementsDeformationGradient: {
      auto* component = AddRemoveOrRefComponent<CQueryElementsDeformationGradient>(reg, e, add);
      if (computeImmediately && component) {
        ecs::TryInvokeOnEntity(
            &soft::UpdateQueryElementsDeformationGradient, reg, e); // compute results for new query
      }
    } break;

    case QueryType::SurfaceNodePositions: {
      auto* component = AddRemoveOrRefComponent<CQuerySurfaceNodePositions>(reg, e, add);
      if (computeImmediately && component && component->nodePositions.empty()) {
        ecs::TryInvokeOnEntity(
            &UpdateQuerySurfaceNodePositions, reg, e); // compute results for new query
      }
    } break;

    case QueryType::SdfDistances: {
      AddRemoveOrRefComponent<CQuerySdfDistances>(reg, e, add);
      if (computeImmediately) {
        MOCHI_LOG_WARNING(
            "QueryType::SdfDistances does not support the 'computeImmediately' option. Ignoring.");
      }
    } break;

    case QueryType::SurfaceNodeNormals: {
      AddRemoveOrRefComponentsForQuery(
          reg, e, QueryType::SurfaceNodePositions, add, computeImmediately); // pre-requisite
      auto* component = AddRemoveOrRefComponent<CQuerySurfaceNodeNormals>(reg, e, add);
      if (computeImmediately && component && component->nodeNormals.empty()) {
        ecs::TryInvokeOnEntity(
            &UpdateQuerySurfaceNodeNormals, reg, e); // compute results for new query
      }
    } break;

    // These two are computed together
    case QueryType::VisualNodePositions:
    case QueryType::VisualNodeNormals: {
      if (!reg.any_of<TagRigidActor>(e) && !reg.any_of<TagRodActor>(e)) {
        // Prerequisite for deformable actors (rods don't have CSimplicialMesh).
        AddRemoveOrRefComponentsForQuery(reg, e, QueryType::NodePositions, add, computeImmediately);
      }
      auto* component = AddRemoveOrRefComponent<CQueryVisualNodePositions>(reg, e, add);
      AddRemoveOrRefComponent<CQueryVisualNodeNormals>(reg, e, add);
      if (computeImmediately && component && component->nodePositions.empty()) {
        ecs::TryInvokeOnEntity(&rod::UpdateQueryVisualNodePositionsAndNormals, reg, e);
        ecs::TryInvokeOnEntity(&UpdateQueryVisualNodePositionsAndNormals, reg, e);
      }
    } break;

    case QueryType::ContactPoints: {
      AddRemoveOrRefComponent<TagQueryActiveContacts>(reg, e, add);
      AddRemoveOrRefComponent<CQueryContactPoints>(reg, e, add);
      if (computeImmediately) {
        MOCHI_LOG_WARNING(
            "QueryType::ContactPoints does not support the 'computeImmediately' option. Ignoring.");
      }
    } break;

    case QueryType::NodeContactForces: {
      AddRemoveOrRefComponent<TagQueryActiveContacts>(reg, e, add);
      AddRemoveOrRefComponent<CQueryNodeContactForces>(reg, e, add);
      if (computeImmediately) {
        MOCHI_LOG_WARNING(
            "QueryType::NodeContactForces does not support the 'computeImmediately' option. Ignoring.");
      }
    } break;

    case QueryType::TotalContactForce: {
      AddRemoveOrRefComponent<TagQueryActiveContacts>(reg, e, add);
      AddRemoveOrRefComponent<CQueryActorContactForces>(reg, e, add);
      if (computeImmediately) {
        MOCHI_LOG_WARNING(
            "QueryType::TotalContactForce does not support the 'computeImmediately' option. Ignoring.");
      }
    } break;

    case QueryType::ElasticEnergy: {
      AddRemoveOrRefComponent<CQueryElasticEnergy>(reg, e, add);
      if (computeImmediately) {
        ecs::TryInvokeOnEntity(&soft::UpdateQueryElasticEnergy, reg, e);
      }
    } break;

    case QueryType::ConstraintForce: {
      AddRemoveOrRefComponent<CQueryConstraintForce>(reg, e, add);
      if (computeImmediately) {
        MOCHI_LOG_WARNING(
            "QueryType::ConstraintForce does not support the 'computeImmediately' option. Ignoring.");
      }
    } break;

    case QueryType::ArticulatedControllerForce: {
      // Enable the equivalent of CQueryConstraintForce on each constraint
      if (auto* constraints = reg.try_get<CControllerConstraints>(e)) {
        if (computeImmediately) {
          MOCHI_LOG_WARNING(
              "QueryType::ArticulatedControllerForce does not support the 'computeImmediately' option. Ignoring.");
        }
        for (auto const& c : constraints->impl) {
          // Get the constraint entity. It must be valid if the Constraint* address is valid.
          entt::entity cEntity = GetEntity(reg, c.constraint->GetHandle(), ErrorAssert{});
          AddRemoveOrRefComponentsForQuery(
              reg, cEntity, QueryType::ConstraintForce, add, /*computeImmediately*/ false);
        }
        AddRemoveOrRefComponent<CQueryArticulatedControllerForce>(reg, e, add);
      }
    } break;

    default: {
      MOCHI_LOG_ERROR(
          "Unsupported QueryType called in AddRemoveOrRefComponentsForQuery. Ignoring.");
    }
  }
}

void mochi::SetErrorIfQueryNotSupported(
    entt::registry const& reg,
    entt::entity e,
    QueryType type,
    Error& error) {
  static_assert(
      static_cast<int>(QueryType::Count) == 13,
      "Please update the switch statement below if QueryType enum changes");
  switch (type) {
    case QueryType::NodePositions:
      MOCHI_ERROR_IF_NOT(
          (reg.any_of<TagSoftActor const, TagShellActor const>(e)),
          error,
          "QueryType::NodePositions is only supported for soft and shell actors.");
      break;
    case QueryType::SurfaceNodePositions:
    case QueryType::SurfaceNodeNormals:
      MOCHI_ERROR_IF_NOT(
          reg.any_of<CSurfaceMesh const>(e),
          error,
          "Surface node queries are only supported for actors with a surface mesh.");
      break;
    case QueryType::VisualNodePositions:
    case QueryType::VisualNodeNormals:
      MOCHI_ERROR_IF_NOT(
          reg.all_of<CVisualMesh const>(e) &&
              (reg.any_of<TagRigidActor const, CRodVisualMeshEmbedding const>(e) ||
               reg.get<CVisualMesh>(e).embedding),
          error,
          "Visual node queries are only supported for actors with a visual mesh. "
          "Deformable actors also require a visual mesh embedding.");
      break;
    case QueryType::ArticulatedControllerForce:
      MOCHI_ERROR_IF_NOT(
          reg.any_of<CControllerConstraints>(e),
          error,
          "QueryType::ArticulatedControllerForce is only supported for articulated actors with a pose controller.");
      break;
    case QueryType::ConstraintForce:
      MOCHI_ERROR_IF_NOT(
          reg.any_of<CConstraintInfo>(e),
          error,
          "QueryType::ConstraintForce is only supported for constraints.");
      break;
    case QueryType::NodeContactForces:
      MOCHI_ERROR_IF(
          reg.any_of<TagRodActor const>(e),
          error,
          "QueryType::NodeContactForces is not supported for rod actors.");
      [[fallthrough]];
    case QueryType::ContactPoints: // fallthrough
    case QueryType::TotalContactForce:
      MOCHI_ERROR_IF(
          reg.all_of<CRequiresFarSdfEvaluation const>(e),
          error,
          "Contact queries are not supported for actors with far SDF evaluation enabled.");
      MOCHI_ERROR_IF_NOT(
          reg.all_of<CContactSamples<TimeStep::Current> const>(e),
          error,
          "Contact queries are only supported for actors with contact sample points.");
      break;
    case QueryType::ElementsDeformationGradient:
      MOCHI_ERROR_IF(
          reg.any_of<TagRomActor const>(e),
          error,
          "QueryType::ElementsDeformationGradient is not supported for ROM actors.");
      [[fallthrough]];
    case QueryType::ElasticEnergy:
      MOCHI_ERROR_IF_NOT(
          reg.all_of<TagSoftActor const>(e), error, "Query only supported for soft actors.");
      break;
    case QueryType::SdfDistances:
      break;
    default:
      MOCHI_ERROR_SET(error, "Unsupported query type.");
      break;
  }
}

QueryHandle mochi::RegisterQuery(
    entt::registry& reg,
    entt::entity e,
    QueryType type,
    bool computeImmediately,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  QueryHandle newHandle = reg.ctx<CQueryHandleAllocator>().NewHandleThreadSafe(type);
  RegisterQuery(reg, e, newHandle, computeImmediately, error);
  return error.IsOK() ? newHandle : QueryHandle{};
}

void mochi::RegisterQuery(
    entt::registry& reg,
    entt::entity e,
    QueryHandle preallocatedHandle,
    bool computeImmediately,
    Error& error) {
  MOCHI_ERROR_IF(!preallocatedHandle.IsValid(), error, "Invalid QueryHandle");
  MOCHI_ERROR_RETURN(error);
  QueryType queryType = GetQueryType(preallocatedHandle);

  // Check QueryType requirements
  SetErrorIfQueryNotSupported(reg, e, queryType, error);
  MOCHI_ERROR_RETURN(error);

  // Store the handle on CActiveQuerySet component (create it the first time)
  auto& querySet = reg.get_or_emplace<CActiveQuerySet>(e);
  [[maybe_unused]] auto [it, wasInserted] = querySet.handles.insert(preallocatedHandle.value);
  MOCHI_ASSERT(wasInserted, "QueryHandle not unique!");

  // Static actors don't process queries during the simulation, so compute the data now.
  if (wasInserted && reg.all_of<TagStaticActor>(e)) {
    computeImmediately = true;
  }

  // Set up the component(s) required to implement these queries
  AddRemoveOrRefComponentsForQuery(reg, e, queryType, true, computeImmediately);

  // No partial success
  if (!error.IsOK()) {
    querySet.handles.erase(preallocatedHandle.value);
  }
}

void mochi::CancelQuery(entt::registry& reg, entt::entity e, QueryHandle handle) {
  auto* querySet = reg.try_get<CActiveQuerySet>(e);
  if (!querySet) {
    return; // No queries registered. Must be invalid.
  }

  // Remove it from the set
  size_t wasRemoved = querySet->handles.erase(handle.value);
  if (!wasRemoved) {
    return; // Not in the set. Must be invalid or already canceled.
  }

  // Decrement the reference count on the associated component(s)
  QueryType type = GetQueryType(handle);
  AddRemoveOrRefComponentsForQuery(reg, e, type, false);
}

void mochi::ValidateQueryHandle(
    entt::registry const& reg,
    entt::entity e,
    QueryHandle handle,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto const* querySet = reg.try_get<CActiveQuerySet>(e);
  bool isRegistered = handle.IsValid() && querySet &&
      (querySet->handles.find(handle.value) != querySet->handles.end());
  MOCHI_ERROR_IF(!isRegistered, error, "Invalid QueryHandle");
}

namespace mochi::query {
void InitializeOnce(entt::registry& reg) {
  // ECS Component Types
  ecs::RegisterComponent<CActiveQuerySet>(reg);
  ecs::RegisterComponent<CQueryActorContactForces>(reg);
  ecs::RegisterComponent<CQueryArticulatedControllerForce>(reg);
  ecs::RegisterComponent<CQueryContactPoints>(reg);
  ecs::RegisterComponent<CQueryContactSamples>(reg);
  ecs::RegisterComponent<CQueryHandleAllocator>(reg);
  ecs::RegisterComponent<CQuerySdfDistances>(reg);
  ecs::RegisterComponent<CQueryElasticEnergy>(reg);
  ecs::RegisterComponent<CQueryNodePositions>(reg);
  ecs::RegisterComponent<CQueryNodeContactForces>(reg);
  ecs::RegisterComponent<CQuerySdfSurface>(reg);
  ecs::RegisterComponent<CQuerySurfaceNodeNormals>(reg);
  ecs::RegisterComponent<CQuerySurfaceNodePositions>(reg);
  ecs::RegisterComponent<CQueryVisualNodeNormals>(reg);
  ecs::RegisterComponent<CQueryVisualNodePositions>(reg);
  ecs::RegisterComponent<CQueryQuadraturePointsPosition>(reg);
  ecs::RegisterComponent<CQueryElementsDeformationGradient>(reg);
  ecs::RegisterComponent<TagQueryActiveContacts>(reg);
  ecs::RegisterComponent<CQueryConstraintForce>(reg);

  // Global Context
  reg.set<CQueryHandleAllocator>();
}
} // namespace mochi::query

namespace {
// Recursive reserve of containers in a tuple "outContainers"
template <size_t kNumContainers, typename TupleContainers>
void Reserve(size_t size, TupleContainers& outContainers) {
  std::get<kNumContainers - 1>(outContainers).reserve(size);
  if constexpr (kNumContainers > 1) {
    Reserve<kNumContainers - 1>(size, outContainers);
  }
}

// Selective cast for ActorHandle
template <typename T>
auto Cast(T data) {
  if constexpr (std::is_same<T, ActorHandle>::value) {
    return static_cast<int>(data.value);
  } else {
    return data;
  }
}

// Recursive emplace_back of elements from a tuple "data" into a tuple of containers "outContainers"
template <size_t kNumContainers, typename TupleContainers, typename TupleData>
void Emplace(TupleData const& data, TupleContainers& outContainers) {
  std::get<kNumContainers - 1>(outContainers)
      .emplace_back(Cast(std::get<kNumContainers - 1>(data)));
  if constexpr (kNumContainers > 1) {
    Emplace<kNumContainers - 1>(data, outContainers);
  }
}

// Selective Flatten for Real3
template <typename T>
auto Flatten(Span<T const> data) {
  if constexpr (std::is_same<T, Real3>::value) {
    return Flatten<real>(data);
  } else {
    return data;
  }
}

// Recursive recording of data from a tuple of containers "containers"
template <size_t kNumContainers, size_t kTotalContainers, typename TupleContainers>
void Record(
    std::string const& prefix,
    std::array<std::string, kTotalContainers> const& names,
    TupleContainers const& containers,
    CRecordingData& outData) {
  RecordAttribute(
      prefix + names[kNumContainers - 1],
      Flatten(MakeConstSpan(std::get<kNumContainers - 1>(containers))),
      outData);
  if constexpr (kNumContainers > 1) {
    Record<kNumContainers - 1>(prefix, names, containers, outData);
  }
}
} // namespace

void mochi::RecordQueryContactPoints(CQueryContactPoints const& contacts, CRecordingData& outData) {
  if (!outData.params.recordContactPoints || !contacts.isInitialized ||
      contacts.contactPoints.empty()) {
    return;
  }

  // Create temp containers of contact data for all contacts
  std::vector<int> actorA, actorB, sampleIndex, elementIndex;
  std::vector<real> distance, intWeight;
  std::vector<Real3> posA, posB, normal, force, pointVelocityA, pointVelocityB, parametricCoords;
  // clang-format off
  auto containers = std::make_tuple(
      std::ref(actorA), std::ref(actorB), std::ref(distance), std::ref(posA), std::ref(posB), std::ref(normal), std::ref(force), std::ref(pointVelocityA), std::ref(pointVelocityB), std::ref(sampleIndex), std::ref(intWeight), std::ref(elementIndex), std::ref(parametricCoords));
  // clang-format on
  constexpr size_t kNumContainers = std::tuple_size<decltype(containers)>::value;
  Reserve<kNumContainers>(contacts.contactPoints.size(), containers);

  // Copy the data to the containers
  for (auto const& c : contacts.contactPoints) {
    // clang-format off
    auto contact = std::make_tuple(
        c.actorA, c.actorB, c.distance, c.posA, c.posB, c.normal, c.force, c.pointVelocityA, c.pointVelocityB, c.sampleIndex, c.intWeight, c.elementIndex, c.parametricCoords);
    // clang-format on
    Emplace<kNumContainers>(contact, containers);
  }

  // Write to CRecordingData
  // clang-format off
  std::array<std::string, kNumContainers> names = {
      "actorA", "actorB", "distance", "posA", "posB", "normal", "force", "pointVelocityA", "pointVelocityB", "sampleIndex", "intWeight", "elementIndex", "parametricCoords"};
  // clang-format on
  Record<kNumContainers>("contacts_", names, containers, outData);
}

void mochi::RecordQueryNodeContactForces(
    CQueryNodeContactForces const& forces,
    CRecordingData& outData) {
  if (!outData.params.recordNodeContactForces || !forces.isInitialized ||
      forces.nodeContactForces.empty()) {
    return;
  }

  // Create temp containers of contact data for all contacts
  std::vector<int> nodeIndex;
  std::vector<Real3> force;
  auto containers = std::make_tuple(std::ref(nodeIndex), std::ref(force));
  constexpr size_t kNumContainers = std::tuple_size<decltype(containers)>::value;
  Reserve<kNumContainers>(forces.nodeContactForces.size(), containers);

  // Copy the data to the containers
  for (auto const& f : forces.nodeContactForces) {
    Emplace<kNumContainers>(std::make_tuple(f.index, f.force), containers);
  }

  // Write to CRecordingData
  std::array<std::string, kNumContainers> names = {"node", "force"};
  Record<kNumContainers>("contact_forces_", names, containers, outData);
}

void mochi::RecordQuerySdfDistances(CQuerySdfDistances const& distances, CRecordingData& outData) {
  if (!outData.params.recordSdfDistances || !distances.isInitialized ||
      distances.distances.empty()) {
    return;
  }

  // Create temp containers of contact data for all contacts
  auto containers = std::make_tuple(
      std::cref(distances.sampleIndices),
      std::cref(distances.distances),
      std::cref(distances.distanceGrads),
      std::cref(distances.worldPositions));
  constexpr size_t kNumContainers = std::tuple_size<decltype(containers)>::value;

  // Write to CRecordingData
  std::array<std::string, kNumContainers> names = {
      "sample_indices", "distances", "distance_grads", "world_positions"};
  Record<kNumContainers>("sdf_distances_", names, containers, outData);
  RecordAttribute<real>(
      "max_sdf_far_distance_evaluation",
      MakeSingletonConstSpan(distances.maxSdfFarDistanceEvaluation),
      outData);
}
