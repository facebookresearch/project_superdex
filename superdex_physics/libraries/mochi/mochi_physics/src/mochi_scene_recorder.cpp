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

#include "mochi_scene_recorder.h"

#include "mochi_actor.h"
#include "mochi_articulated_body.h"
#include "mochi_common_components.h"
#include "mochi_deformable.h"
#include "mochi_ecs_utils.h"
#include "mochi_island.h"
#include "mochi_rod.h"
#include "mochi_soft_rom_systems.h"

#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/reflection.h>

#include <algorithm>
#include <cinttypes>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace mochi {

static void WriteActorRecordingData(GroupWriter& writer, CRecordingData const& data, Error& error);

SceneRecorder::SceneRecorder(
    std::unique_ptr<GroupWriter> writer,
    entt::registry& registry,
    RecordingParams const& params)
    : _registry(registry), _writer(std::move(writer)), _params(params) {
  // Create a scope called "events" at the root of the file. Every new event will be added as a
  // sub-group. Note that groups can remember the order in which their members were created, so they
  // can be iterated in the same order later. The root of an HDF5 file does not appear to have this
  // same ordering property. Store the ScopeGuard so that it stays open.
  _eventsGroup =
      std::make_unique<GroupWriter::ScopeGuard>(_writer->EnterGroup("events", ErrorLog{}));

  // Call OnCreateActor for all actors that already exist
  _registry.view<CActorInfo>().each(
      [this](entt::entity e, auto& /*actorInfo*/) { OnCreateActor(_registry, e); });

  // Register for callbacks when actors are created/destroyed after this point
  _registry.on_construct<CActorInfo>().connect<&SceneRecorder::OnCreateActor>(this);
  _registry.on_destroy<CActorInfo>().connect<&SceneRecorder::OnDestroyActor>(this);

  // Show that recording is enabled by adding a tag to the global ECS context
  _registry.set<TagSceneRecordingEnabled>();
  _registry.set<CRecordingParams>(_params);
}

SceneRecorder::~SceneRecorder() {
  // No longer recording
  _registry.unset<TagSceneRecordingEnabled>();

  // Deregister callbacks
  _registry.on_construct<CActorInfo>().disconnect<&SceneRecorder::OnCreateActor>(this);
  _registry.on_destroy<CActorInfo>().disconnect<&SceneRecorder::OnDestroyActor>(this);

  // Close the "events" group
  _eventsGroup.reset();

  // Cancel queries
  for (auto& query : _queries) {
    CancelQuery(_registry, query.first, query.second);
  }

  // Destroy all instances of CRecordingData that we may have added to actors
  _registry.clear<CRecordingData>();
}

void SceneRecorder::OnStepBegin(double timeStepSec, Real3 stepGravity) {
  MOCHI_PROFILE_SCOPE();
  _timeStep = timeStepSec;
  _timeTotal += timeStepSec;
  _gravity = stepGravity;
}

void SceneRecorder::OnStepEnd() {
  MOCHI_ERROR_RETURN(_error);
  MOCHI_PROFILE_SCOPE();

  AddDestroyActorEvents();
  AddCreateActorEvents();
  AddStepEvent();

  // Clear CRecordingData for each actor
  _registry.view<CRecordingData>().each([](auto& data) { data.entries.clear(); });

  CheckForErrors();
}

void SceneRecorder::CheckForErrors() {
  // Log a warning on the first error, but don't spam
  if (!_error.IsOK() && !_hasReportedError) {
    _hasReportedError = true;
    MOCHI_LOG_WARNING("Recording Failed! Reason: %s", _error.ToString().c_str());
  }
}

void SceneRecorder::OnCreateActor(entt::registry& reg, entt::entity e) {
  // Keep track of this new actor until the next simulation step
  _createdActors.push_back(e);

  // Add CRecordingData so that systems can record additional data during the simulation.
  reg.emplace<CRecordingData>(e, _params);

  // Enable queries that the user wants to record
  if (!reg.any_of<TagStaticActor>(e)) {
    constexpr bool kComputeImmediately = false;
    if (_params.recordContactPoints) {
      Error ignore; // Some actors may not support this query. Just skip it.
      auto query = RegisterQuery(reg, e, QueryType::ContactPoints, kComputeImmediately, ignore);
      _queries.emplace_back(e, query);
    }
    if (_params.recordNodeContactForces) {
      Error ignore; // Some actors may not support this query. Just skip it.
      auto query = RegisterQuery(reg, e, QueryType::NodeContactForces, kComputeImmediately, ignore);
      _queries.emplace_back(e, query);
    }
    if (_params.recordSdfDistances) {
      Error ignore; // Some actors may not support this query. Just skip it.
      auto query = RegisterQuery(reg, e, QueryType::SdfDistances, kComputeImmediately, ignore);
      _queries.emplace_back(e, query);
    }
  }
}

void SceneRecorder::OnDestroyActor(entt::registry& /*reg*/, entt::entity e) {
  // Remove it form _createdActors
  _createdActors.erase(
      std::remove(_createdActors.begin(), _createdActors.end(), e), _createdActors.end());

  // Keep track of the fact that this actor was destroyed until the next simulation step
  _destroyedActors.push_back(e);
}

GroupWriter::ScopeGuard SceneRecorder::EnterNewEvent(std::string_view eventType) {
  MOCHI_ERROR_RETURN(_error, {});

  // Each event is stored in a group. This function enters that group. The group name is just the
  // event number as a string. This is helpful when inspecting files with HDFView.
  char groupName[32];
  size_t groupNameLen = snprintf(groupName, sizeof(groupName), "%" PRIu64, _eventCounter++);
  auto guard = _writer->EnterGroup(std::string_view{groupName, groupNameLen}, _error);

  // All events have this attribute
  _writer->AddAttribute("eventType", eventType, _error);
  return guard;
}

static void WriteActorInfoAttributes(
    GroupWriter& writer,
    entt::registry const& reg,
    entt::entity e,
    CActorInfo const& info,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  SceneHandle sceneHandle = reg.ctx<CSceneHandle const>().value;
  writer.AddAttribute("handle", GetActorHandle(e, sceneHandle).value, error);
  writer.AddAttribute("type", SReflect::EnumToString(info.type), error);
  writer.AddAttribute("name", info.name.empty() ? "Unnamed" : info.name, error);
}

static void WriteTransformAttributes(
    std::string const& translationName,
    std::string const& rotationName,
    TransformRT const& transform,
    GroupWriter& writer,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto rotation = ToReal4(transform.GetRotation().data);
  writer.AddAttribute(translationName, transform.GetTranslation(), error);
  writer.AddAttribute(rotationName, rotation, error);
}

void WriteTransformAttributes(
    std::string const& translationName,
    std::string const& rotationName,
    TransformRT const& transform,
    CRecordingData& data) {
  auto rotation = ToReal4(transform.GetRotation().data);
  RecordAttribute<real>(translationName, transform.GetTranslation(), data);
  RecordAttribute<real>(rotationName, rotation, data);
}

void SceneRecorder::AddCreateActorEvents() {
  MOCHI_PROFILE_SCOPE();

  // Write an event for each actor that has been created since the last call
  for (entt::entity e : _createdActors) {
    auto const& info = _registry.get<CActorInfo>(e);

    auto group = EnterNewEvent("CreateActor");
    WriteActorInfoAttributes(*_writer, _registry, e, info, _error);

    int isStatic = _registry.all_of<TagStaticActor>(e) ? 1 : 0;
    _writer->AddAttribute("static", isStatic, _error);

    if (auto const* root = _registry.try_get<CRootTransform const>(e)) {
      WriteTransformAttributes("translation", "rotation", root->worldFromLocal, *_writer, _error);
    }

    if (auto const* evalPoint = _registry.try_get<CRigidTransformEvalPoint const>(e)) {
      _writer->AddAttribute("rigidTransformEvalPoint", evalPoint->GetPositionReference(), _error);
    }

    // TODO: Save the rest of the information necessary to replay the creation of this actor

    if (_params.recordActorMeshes) {
      if (auto const* volMesh = _registry.try_get<CTetrahedralMesh>(e)) {
        auto const* mesh = volMesh->mesh.get();
        _writer->AddAttribute("numNodes", mesh->GetNumNodes(), _error);
        _writer->AddAttribute("numEdges", mesh->GetNumEdges(), _error);
        _writer->AddAttribute("numPolys", mesh->GetNumElements(), _error);

        Span<int const> edgeIndices = Flatten(mesh->GetEdges());
        size_t const edgeDims[2] = {edgeIndices.size(), 1};
        _writer->AddDataSet("edgeIndices", edgeIndices, MakeSpan(edgeDims), _error);

        Span<int const> polyIndices = mesh->GetFlatConnectivity();
        size_t const polyDims[2] = {polyIndices.size(), 1};
        _writer->AddDataSet("polyIndices", polyIndices, MakeSpan(polyDims), _error);

        Span<real const> refPos = Flatten(mesh->GetNodeCoordinates());
        MOCHI_ASSERT(refPos.size() % 3 == 0);
        size_t const posDims[2] = {refPos.size() / 3, 3};
        _writer->AddDataSet("refPositions", refPos, MakeSpan(posDims), _error);
      }

      if (auto const* surfMesh = _registry.try_get<CSurfaceMesh>(e)) {
        auto const* mesh = surfMesh->mesh.get();
        _writer->AddAttribute("numSurfNodes", mesh->GetNumNodes(), _error);
        _writer->AddAttribute("numSurfEdges", mesh->GetNumEdges(), _error);
        _writer->AddAttribute("numSurfFaces", mesh->GetNumElements(), _error);

        Span<int const> surfNodeIndices = mesh->GetActiveNodes();
        size_t const surfNodeDims[2] = {surfNodeIndices.size(), 1};
        _writer->AddDataSet("surfNodeIndices", surfNodeIndices, MakeSpan(surfNodeDims), _error);

        Span<int const> surfEdgeIndices = Flatten(mesh->GetActiveNodesEdges());
        size_t const surfEdgeDims[2] = {surfEdgeIndices.size(), 1};
        _writer->AddDataSet("surfEdgeIndices", surfEdgeIndices, MakeSpan(surfEdgeDims), _error);

        Span<int const> surfFaceIndices = mesh->GetFlatConnectivity();
        size_t const surfFaceDims[2] = {surfFaceIndices.size(), 1};
        _writer->AddDataSet("surfFaceIndices", surfFaceIndices, MakeSpan(surfFaceDims), _error);

        Span<real const> refPos = Flatten(mesh->GetNodeCoordinates());
        MOCHI_ASSERT(refPos.size() % 3 == 0);
        size_t const posDims[2] = {refPos.size() / 3, 3};
        _writer->AddDataSet("surfRefPositions", refPos, MakeSpan(posDims), _error);
      }

      if (auto const* polylineMesh = _registry.try_get<CPolylineMesh>(e)) {
        int const numNodes = isize(polylineMesh->nodes);

        // For rod actors, all nodes are "surface" nodes
        DynamicArray<int> surfNodeIndices(numNodes);
        std::iota(surfNodeIndices.begin(), surfNodeIndices.end(), 0);
        size_t const surfNodeDims[2] = {static_cast<size_t>(numNodes), 1};
        _writer->AddDataSet(
            "surfNodeIndices", MakeConstSpan(surfNodeIndices), MakeSpan(surfNodeDims), _error);

        Span<real const> refPos = Flatten(MakeConstSpan(polylineMesh->nodes));
        MOCHI_ASSERT(refPos.size() % 3 == 0);
        size_t const posDims[2] = {refPos.size() / 3, 3};
        _writer->AddDataSet("surfRefPositions", refPos, MakeSpan(posDims), _error);
      }

      if (auto const* vizMesh = _registry.try_get<CVisualMesh>(e)) {
        auto const* mesh = vizMesh->mesh.get();

        Span<int const> vizFaceIndices = mesh->GetFlatConnectivity();
        size_t const faceDims[2] = {vizFaceIndices.size(), 1};
        _writer->AddDataSet("visualMeshIndices", vizFaceIndices, MakeSpan(faceDims), _error);

        Span<real const> vizPositions = Flatten(mesh->GetNodeCoordinates());
        size_t const positionsDims[2] = {vizPositions.size(), 1};
        _writer->AddDataSet("visualMeshPositions", vizPositions, MakeSpan(positionsDims), _error);

        if (vizMesh->embedding) {
          auto const* embedding =
              dynamic_cast<LinearMeshEmbedding const*>(vizMesh->embedding.get());
          if (embedding) {
            Span<real const> skinWeights = embedding->GetWeights();
            size_t const skinWeightsDim[2] = {skinWeights.size(), 1};
            _writer->AddDataSet("skinWeights", skinWeights, MakeSpan(skinWeightsDim), _error);

            Span<int const> skinIndices = embedding->GetIndices();
            size_t const skinIndicesDim[2] = {skinIndices.size(), 1};
            _writer->AddDataSet("skinIndices", skinIndices, MakeSpan(skinIndicesDim), _error);
          }
        }
      }
    }

    if (_params.recordActorLocalToGlobalMap) {
      if (auto const* local2Global = _registry.try_get<CLocal2GlobalMap>(e)) {
        // Record local 2 global map
        auto l2gGroup = _writer->EnterGroup("localToGlobal", _error);

        size_t const globalIndicesDim[2] = {static_cast<size_t>(local2Global->GetNumIndices()), 1};
        _writer->AddDataSet(
            "globalIndices", local2Global->GetGlobalIndices(), globalIndicesDim, _error);

        size_t const elementOffsetsDim[2] = {
            static_cast<size_t>(local2Global->GetNumElements()), 1};
        _writer->AddDataSet(
            "elementOffsets", local2Global->GetElementOffsets(), elementOffsetsDim, _error);
      }
    }

    if (_params.recordActorMassMatrix) {
      auto const* massMatrix = _registry.try_get<CMassMatrix const>(e);
      auto const* fullSparsity = _registry.try_get<CFullSparsityPattern const>(e);
      if (massMatrix && fullSparsity) {
        int numRows = fullSparsity->graph.size();
        int numCols = numRows;
        CRecordingData massMatrixRecording;
        RecordDatasetSparseMatrixCSR<real>(
            "massMatrix",
            numRows,
            numCols,
            fullSparsity->graph.GetPointers(),
            fullSparsity->graph.GetTargets(),
            MakeConstSpan(massMatrix->values),
            massMatrixRecording);
        WriteActorRecordingData(*_writer, massMatrixRecording, _error);
      }
    }
  }

  _createdActors.clear();
}

void SceneRecorder::AddDestroyActorEvents() {
  MOCHI_PROFILE_SCOPE();

  // Write an event for each actor that has been destroyed since the last call
  SceneHandle sceneHandle = _registry.ctx<CSceneHandle const>().value;
  for (entt::entity e : _destroyedActors) {
    auto group = EnterNewEvent("DestroyActor");
    _writer->AddAttribute("handle", GetActorHandle(e, sceneHandle).value, _error);
  }

  _destroyedActors.clear();
}

// Copy the contents of an actor's CRecordingData into the current scope
static void WriteActorRecordingData(GroupWriter& writer, CRecordingData const& data, Error& error) {
  MOCHI_ERROR_RETURN(error);

  /*
    Due to the way that the group writer works, we must write global attributes first before
    anything else. Global attributes are attributes of the current group of the input writer. Hence,
    the below computation is done in two passes: first we write the global attributes. Then, we
    write everything else.
  */
  enum class DataPass { GlobalAttributes, Rest };

  // Returns true if we should write the given entry during the pass
  auto passFilter = [](DataPass pass, std::string const& name, CRecordingData::Entry const& entry) {
    bool isGlobalAttrib = entry.isAttribute && (name.find_first_of('/') == std::string::npos);
    switch (pass) {
      case DataPass::GlobalAttributes: {
        return isGlobalAttrib;
      }
      default: {
        return !isGlobalAttrib;
      }
    }
  };

  // Writes a single pass
  auto writeDataPass = [&](DataPass pass) {
    std::string prevDatasetName;
    // Iterate over the entries in alphabetically sorted order.
    for (auto const& [name, entry] : data.entries) {
      // Only write things that we need to write during this pass
      if (passFilter(pass, name, entry)) {
        // Get the value size from the type enum
        size_t valueSize = 0;
        switch (entry.type) {
          case CRecordingData::Type::Double:
            valueSize = sizeof(double);
            break;
          case CRecordingData::Type::Float:
            valueSize = sizeof(float);
            break;
          case CRecordingData::Type::Int:
            valueSize = sizeof(int);
            break;
          default:
            MOCHI_ASSERT(false, "CRecordingData entry has unknown value type");
            break;
        }

        // Compute the product of all dimensions
        size_t totalDims = 1;
        for (size_t d : entry.dims) {
          totalDims *= d;
        }

        // Use asserts to check for malformed entries (would be a programmer error)
        MOCHI_ASSERT(!name.empty(), "CRecordingData entries must have a name");
        MOCHI_ASSERT(!entry.data.empty(), "CRecordingData entry has no data");
        MOCHI_ASSERT(
            entry.data.size() == valueSize * totalDims, "CRecordingData data size mismatch");
        MOCHI_ASSERT(
            (entry.dims.size() <= 1) || !entry.isAttribute,
            "Multi-dimensional attribute data not supported. Must be a data set instead.");

        std::string datasetName = entry.isAttribute ? "" : name;
        std::string attributeName = entry.isAttribute ? name : "";

        // If the name contains a slash, then it must be of the form
        // "<dataset_name>/<attribute_name>". Since entries are enumerated in sorted order, these
        // attributes should come immediately after the dataset that they describe.
        auto slash = name.find_first_of('/');
        if (slash != std::string::npos) {
          MOCHI_ASSERT(
              entry.isAttribute,
              "Only attributes are allowed to contains a forward slash character");
          datasetName = name.substr(0, slash);
          attributeName = name.substr(slash + 1);
          MOCHI_ASSERT(prevDatasetName == datasetName, "Dataset not found for attribute");
        }

        // The number of values stored in the byte array
        size_t const valueCount = isize(entry.data) / valueSize;

        // Add an attribute or dataset
        switch (entry.type) {
          case CRecordingData::Type::Double:
            if (entry.isAttribute) {
              writer.AddAttribute(
                  attributeName,
                  reinterpret_cast<double const*>(entry.data.data()),
                  valueCount,
                  error);
            } else {
              writer.AddDataSet(
                  datasetName,
                  Span{reinterpret_cast<double const*>(entry.data.data()), valueCount},
                  Span{entry.dims.data(), entry.dims.size()},
                  error);
            }
            break;
          case CRecordingData::Type::Float:
            if (entry.isAttribute) {
              writer.AddAttribute(
                  attributeName,
                  reinterpret_cast<float const*>(entry.data.data()),
                  valueCount,
                  error);
            } else {
              writer.AddDataSet(
                  datasetName,
                  Span{reinterpret_cast<float const*>(entry.data.data()), valueCount},
                  Span{entry.dims.data(), entry.dims.size()},
                  error);
            }
            break;
          case CRecordingData::Type::Int:
            if (entry.isAttribute) {
              writer.AddAttribute(
                  attributeName,
                  reinterpret_cast<int const*>(entry.data.data()),
                  valueCount,
                  error);
            } else {
              writer.AddDataSet(
                  datasetName,
                  Span{reinterpret_cast<int const*>(entry.data.data()), valueCount},
                  Span{entry.dims.data(), entry.dims.size()},
                  error);
            }
            break;
          default:
            break;
        }

        prevDatasetName = std::move(datasetName);
      }
    }
  };

  // Write everything
  writeDataPass(DataPass::GlobalAttributes);
  writeDataPass(DataPass::Rest);
}

// Write all per-step data about an actor to the current scope
static void WriteActorStepEvent(
    GroupWriter& writer,
    std::optional<GroupWriter::ScopeGuard>& actorListScope,
    RecordingParams const& params,
    entt::registry const& reg,
    entt::entity e,
    CActorInfo const& info,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  // Enter the "actors" group if we haven't already
  if (!actorListScope) {
    actorListScope.emplace(writer.EnterGroup("actors", error));
  }

  SceneHandle sceneHandle = reg.ctx<CSceneHandle const>().value;

  // Enter a group named according to the actor's handle value
  char buf[64];
  size_t len = snprintf(buf, sizeof(buf), "%" PRIu64, GetActorHandle(e, sceneHandle).value);
  GroupWriter::ScopeGuard actorScope = writer.EnterGroup(std::string_view(buf, len), error);
  MOCHI_ERROR_RETURN(error);

  // CActorInfo is nice to have inline for human inspection in HDFView
  WriteActorInfoAttributes(writer, reg, e, info, error);
  MOCHI_ERROR_RETURN(error);

  // Write information about the state of the actor, if requested.
  // These fields are hard coded rather than being part of CRecordingData.
  bool isStatic = reg.all_of<TagStaticActor>(e);
  if ((isStatic && params.recordStaticActorState) ||
      (!isStatic && params.recordDynamicActorState)) {
    auto const* root = reg.try_get<CRootTransform const>(e);
    if (root) {
      WriteTransformAttributes("translation", "rotation", root->worldFromLocal, writer, error);
      MOCHI_ERROR_RETURN(error);
    }

    // Info of an actor (not state) that is not guaranteed to be invariant across simulation
    // runs
    if (auto const* memberInfo = reg.try_get<CIslandMemberInfo const>(e)) {
      writer.AddAttribute(
          "solverGroup", GetActorHandle(memberInfo->island, sceneHandle).value, error);
      MOCHI_ERROR_RETURN(error);
    }

    if (auto const* dofOffset = reg.try_get<CDofOffset const>(e)) {
      writer.AddAttribute("solverDofOffset", dofOffset->dofsOffset, error);
      writer.AddAttribute("solverPoseOffset", dofOffset->poseOffset, error);
      MOCHI_ERROR_RETURN(error);
    }
  }

  // Write any additional information that ECS systems recorded for this actor.
  if (auto const* recordingData = reg.try_get<CRecordingData const>(e)) {
    if (!recordingData->entries.empty()) {
      WriteActorRecordingData(writer, *recordingData, error);
    }
  }
}

void SceneRecorder::AddStepEvent() {
  MOCHI_ERROR_RETURN(_error);
  MOCHI_PROFILE_SCOPE();

  // Enter a new "Step" event
  auto eventGroup = EnterNewEvent("Step");
  _writer->AddAttribute("timeTotal", _timeTotal, _error);
  _writer->AddAttribute("timeStep", _timeStep, _error);
  _writer->AddAttribute("gravity", _gravity, _error);

  // If we have any actors that need to record per-step information, then we will add a group
  // called "actors" and a sub-group for each actor.
  std::optional<GroupWriter::ScopeGuard> actorListScope;
  _registry.view<CActorInfo const>().each([&](auto e, auto const& actorInfo) {
    WriteActorStepEvent(*_writer, actorListScope, _params, _registry, e, actorInfo, _error);
  });
}

namespace scene_recorder {
void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CRecordingData>(reg);
  ecs::RegisterComponent<CRecordingParams>(reg);
}
} // namespace scene_recorder

} // namespace mochi
