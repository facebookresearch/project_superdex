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

#include <superdex_robotics/sensors/interaction_wrench_metadata.h>
#include <superdex_robotics/sensors/sensor_link_vertex_force.h>

#include <superdex_robotics/sensors/sensor_recording.h>
#include <superdex_robotics/utils/file_utils.h>

#include <sensors/sensor_timing.h>

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <algorithm>
#include <array>
#include <cmath>

using namespace mochi;
using namespace superdex::robotics;

SensorLinkVertexForce::SensorLinkVertexForce(
    Actor* linkActor,
    std::string_view paramArgs,
    Error& error)
    : SensorBase(linkActor, error) {
  Params const params = paramArgs.empty()
      ? Params{}
      : LoadParamsFromPathOrJson<Params>(paramArgs, error, /*allowPartial=*/false);
  MOCHI_ERROR_RETURN(error);
  Initialize(params, error);
}

SensorLinkVertexForce::SensorLinkVertexForce(Actor* linkActor, Params const& params, Error& error)
    : SensorBase(linkActor, error) {
  Initialize(params, error);
}

void SensorLinkVertexForce::Initialize(Params const& params, Error& error) {
  MOCHI_ERROR_IF(
      !std::isfinite(params.sampleRateHz) || params.sampleRateHz < 0_r,
      error,
      "SensorLinkVertexForce sampleRateHz must be finite and nonnegative");
  MOCHI_ERROR_RETURN(error);
  if (GetActor() == nullptr) {
    MOCHI_ERROR_SET(error, "SensorLinkVertexForce requires a link actor");
    return;
  }
  _params = params;
}

SensorLinkVertexForce::~SensorLinkVertexForce() {
  Actor* const actor = GetActor();
  if (actor != nullptr && _queryRegistered) {
    actor->CancelQuery(_nodeForceQuery);
  }
}

void SensorLinkVertexForce::ResolveLayout(Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (!_layout.positions.empty()) {
    return;
  }
  Actor* const actor = GetActor();
  if (actor == nullptr) {
    MOCHI_ERROR_SET(error, "SensorLinkVertexForce has no live link actor");
    return;
  }
  MeshDataView const contactSkin = actor->GetContactSkinMesh();
  MeshDataView const mesh = actor->GetMesh();
  MeshDataView const& contactMesh = contactSkin.IsEmpty() ? mesh : contactSkin;
  // Actor contact topology is immutable; the reader repeats this same
  // contact-skin-first selection to validate the cached layout.
  MOCHI_ERROR_IF(
      contactMesh.IsEmpty(), error, "SensorLinkVertexForce requires a mesh or contact skin");
  MOCHI_ERROR_RETURN(error);

  Span<Real3 const> const parentPositions = Unflatten<Real3 const>(contactMesh.coordinates);
  MOCHI_ERROR_IF(
      parentPositions.empty(), error, "SensorLinkVertexForce requires at least one contact vertex");
  MOCHI_ERROR_RETURN(error);
  DynamicArray<Real3> sensorPositions(parentPositions.size());
  std::ranges::transform(parentPositions, sensorPositions.begin(), [this](Real3 const& position) {
    return GetParentFromSensor().TransformPointInverse(position);
  });
  _layout = MakeTactileTaxelLayout(sensorPositions, error);
  MOCHI_ERROR_RETURN(error);
  _metadata.coordinateFrame = "sensor";
  _metadata.vertexAxis = 0;
  _metadata.positionsMeters = MakeConstSpan(_layout.positions);
  _metadata.normals = {};
}

SensorFormat SensorLinkVertexForce::InitSensor(SensorScope const& /*scope*/, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF_NOT(Inputs().empty(), error, "SensorLinkVertexForce does not accept inputs");
  ResolveLayout(error);
  MOCHI_ERROR_RETURN(error, {});
  Actor* const actor = GetActor();
  if (actor == nullptr) {
    MOCHI_ERROR_SET(error, "SensorLinkVertexForce requires an actor");
    return {};
  }
  if (!actor->IsQuerySupported(QueryType::NodeContactForces)) {
    MOCHI_ERROR_SET(error, "SensorLinkVertexForce requires node contact-force query support");
    return {};
  }
  if (!_queryRegistered) {
    _nodeForceQuery = actor->RegisterQuery(QueryType::NodeContactForces, error);
    MOCHI_ERROR_RETURN(error, {});
    _queryRegistered = true;
  }
  _graphForces.forces.resize_noinit(_layout.positions.size());
  _graphOutput.resize_noinit(3 * _layout.positions.size());
  return {
      .kind = DynamicString{kSensorFormatKindForce3D},
      .shape = {isize(_layout.positions), 3},
      .axes =
          {
              {.name = "vertex", .labels = {}, .units = {}},
              MakeForceComponentAxis(),
          },
  };
}

SensorStepResult SensorLinkVertexForce::Step(StepTime const& now, Error& error) {
  Span<SensorBase const* const> const inputs = Inputs();
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  MOCHI_ERROR_IF_NOT(inputs.empty(), error, "SensorLinkVertexForce does not accept inputs");
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  auto const samplePeriod =
      detail::SensorCapturePeriod(now, _params.sampleRateHz, _lastSamplePeriod);
  if (!samplePeriod) {
    return SensorStepResult::NoUpdate;
  }
  ReadTactileTaxelForcesInto(GetActor(), GetWorldTransform(), _layout, _graphForces, error);
  FlattenTactileTaxelForcesInto(
      _graphForces, Span<float>{_graphOutput.data(), _graphOutput.size()}, error);
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  PublishSample(now, Span<float const>{_graphOutput}, error);
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  _lastSamplePeriod = samplePeriod;
  return SensorStepResult::Published;
}

void SensorLinkVertexForce::ResetSensor() {
  _lastSamplePeriod.reset();
}

TactileTaxelForces SensorLinkVertexForce::ComputeSignal(Error& error) {
  ResolveLayout(error);
  return ReadTactileTaxelForces(GetActor(), GetWorldTransform(), _layout, error);
}

Span<std::byte const> SensorLinkVertexForce::TryGetMetadata(std::string_view key) const {
  if (!_layout.positions.empty() && key == SensorMetadataKey<VertexListMeta>::Key) {
    return {reinterpret_cast<std::byte const*>(&_metadata), sizeof(_metadata)};
  }
  if (!_layout.positions.empty() && key == SensorMetadataKey<InteractionWrenchMeta>::Key) {
    return {
        reinterpret_cast<std::byte const*>(&_interactionMetadata), sizeof(_interactionMetadata)};
  }
  return SensorBase::TryGetMetadata(key);
}

void SensorLinkVertexForce::ConfigureRecording(SensorRecordingSink& sink, Error& error) const {
  SensorBase::ConfigureRecording(sink, error);
  constexpr auto typeId = SensorMetadataKey<VertexListMeta>::Key;
  constexpr std::string_view path = "metadata/vertex_list";
  sink.DeclareMetadataType(typeId, error);
  sink.WriteConstantText(path, "metadata_type", typeId, error);
  sink.WriteConstantText(path, "coordinate_frame", _metadata.coordinateFrame, error);
  std::array<double, 1> const vertexAxis{static_cast<double>(_metadata.vertexAxis)};
  sink.WriteConstantScalars(path, "observation_vertex_axis", vertexAxis, error);
  sink.WriteConstantReal3s(path, "positions_meters", _metadata.positionsMeters, error);
  sink.WriteConstantReal3s(path, "normals", _metadata.normals, error);
}
