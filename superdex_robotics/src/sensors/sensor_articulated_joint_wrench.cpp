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

#include <superdex_robotics/sensors/sensor_articulated_joint_wrench.h>

#include <superdex_robotics/sensors/sensor_recording.h>

#include <mochi_core/utils/basic_utils.h>

#include <limits>

using namespace mochi;
using namespace superdex::robotics;

DynamicArray<bool> superdex::robotics::ClassifyStaticArticulatedLinks(
    Span<ArticulatedJointType const> jointTypes,
    Span<int32_t const> parents,
    Error& error) {
  DynamicArray<bool> result(jointTypes.size(), false);
  MOCHI_ERROR_IF(
      jointTypes.empty() || jointTypes.size() != parents.size(),
      error,
      "Articulated-joint wrench layout requires matching nonempty joint and parent arrays");
  MOCHI_ERROR_IF(
      jointTypes.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()),
      error,
      "Articulated-joint wrench layout exceeds the int32 index range");
  MOCHI_ERROR_RETURN(error, {});

  for (std::size_t link = 0; link < jointTypes.size(); ++link) {
    auto const linkIndex = static_cast<int32_t>(link);
    int32_t const parent = parents[link];
    MOCHI_ERROR_IF(
        parent < -1 || parent >= linkIndex,
        error,
        "Articulated-joint wrench layout requires parents before children");
    MOCHI_ERROR_RETURN(error, {});
    result[link] = jointTypes[link] == ArticulatedJointType::Hard && (parent < 0 || result[parent]);
  }
  return result;
}

SensorArticulatedJointWrench::SensorArticulatedJointWrench(
    Actor* linkActor,
    std::string_view paramArgs,
    Error& error)
    : SensorBase(linkActor, error) {
  Initialize(linkActor, paramArgs, error);
}

SensorArticulatedJointWrench::~SensorArticulatedJointWrench() {
  Shutdown();
}

void SensorArticulatedJointWrench::Initialize(
    Actor* linkActor,
    std::string_view paramArgs,
    Error& error) {
  MOCHI_ERROR_IF(
      !paramArgs.empty() && paramArgs != "{}",
      error,
      "SensorArticulatedJointWrench does not accept parameters");
  MOCHI_ERROR_RETURN(error);
  if (linkActor == nullptr) {
    MOCHI_ERROR_SET(
        error, "SensorArticulatedJointWrench must be attached to a nested articulated link");
    return;
  }
  MOCHI_ERROR_IF_NOT(
      linkActor->IsNestedLinkActor(),
      error,
      "SensorArticulatedJointWrench must be attached to a nested articulated link");
  MOCHI_ERROR_RETURN(error);

  Scene* const scene = linkActor->GetScene();
  if (scene == nullptr) {
    MOCHI_ERROR_SET(error, "SensorArticulatedJointWrench link has no scene");
    return;
  }
  ActorHandle const articulatedHandle = linkActor->GetArticulatedActor(error);
  MOCHI_ERROR_RETURN(error);
  Actor* const articulated = scene->GetActor(articulatedHandle);
  if (articulated == nullptr) {
    MOCHI_ERROR_SET(error, "SensorArticulatedJointWrench owning articulated actor is unavailable");
    return;
  }
  Actor& articulatedActor = *articulated;

  auto const shapeInfo = articulatedActor.GetArticulatedShapeInfo(error);
  Span<ActorHandle const> const linkHandles = articulatedActor.GetNestedLinkActors(error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      shapeInfo.jointTypes.size() == linkHandles.size() &&
          shapeInfo.parents.size() == linkHandles.size(),
      error,
      "SensorArticulatedJointWrench shape info does not match its links");
  MOCHI_ERROR_RETURN(error);

  DynamicArray<int32_t> parents;
  parents.reserve(shapeInfo.parents.size());
  for (auto const parent : shapeInfo.parents) {
    MOCHI_ERROR_IF(
        static_cast<int64_t>(parent) < std::numeric_limits<int32_t>::min() ||
            static_cast<int64_t>(parent) > std::numeric_limits<int32_t>::max(),
        error,
        "SensorArticulatedJointWrench parent index exceeds the int32 metadata range");
    MOCHI_ERROR_RETURN(error);
    parents.push_back(static_cast<int32_t>(parent));
  }
  DynamicArray<bool> const isStatic =
      ClassifyStaticArticulatedLinks(shapeInfo.jointTypes, parents, error);
  MOCHI_ERROR_RETURN(error);

  _layout.scene = scene;
  _layout.articulatedActor = articulatedHandle;
  _layout.coordinateFrame = DynamicString("world");
  _layout.momentReference = DynamicString("child_joint_origin");
  for (std::size_t link = 0; link < isStatic.size(); ++link) {
    auto const linkIndex = static_cast<int32_t>(link);
    if (isStatic[link]) {
      _omittedStaticLinkIndices.push_back(linkIndex);
      continue;
    }
    _rowJointIndices.push_back(linkIndex);
    _parentLinkIndices.push_back(parents[link]);
  }
  MOCHI_ERROR_IF(
      _rowJointIndices.empty(),
      error,
      "SensorArticulatedJointWrench actor has no measurable articulated joints");
  MOCHI_ERROR_RETURN(error);
  _layout.rowJointIndices = MakeConstSpan(_rowJointIndices);
  _layout.parentLinkIndices = MakeConstSpan(_parentLinkIndices);
  _layout.omittedStaticLinkIndices = MakeConstSpan(_omittedStaticLinkIndices);

  QueryHandle const query =
      articulatedActor.RegisterQuery(QueryType::ArticulatedJointWrenches, error);
  MOCHI_ERROR_RETURN(error);
  _query = query;
  _queryRegistered = true;
  _articulatedHandle = articulatedHandle;
  _queryValues.resize(_layout.rowJointIndices.size() * 6);
  _initialized = true;
}

void SensorArticulatedJointWrench::Shutdown() {
  Scene* const scene = GetScene();
  if (_queryRegistered && scene != nullptr) {
    if (Actor* const articulated = scene->GetActor(_articulatedHandle)) {
      articulated->CancelQuery(_query);
    }
  }
  _queryRegistered = false;
  _initialized = false;
  _queryValues.clear();
}

SensorFormat SensorArticulatedJointWrench::InitSensor(SensorScope const& /*scope*/, Error& error) {
  MOCHI_ERROR_IF_NOT(_initialized, error, "SensorArticulatedJointWrench is not initialized");
  MOCHI_ERROR_IF_NOT(
      Inputs().empty(), error, "SensorArticulatedJointWrench does not accept graph inputs");
  MOCHI_ERROR_RETURN(error, {});
  return {
      .kind = DynamicString{kSensorFormatKindWrench6DList},
      .shape = {static_cast<int32_t>(_layout.rowJointIndices.size()), 6},
      .axes =
          {
              {.name = "joint", .labels = {}, .units = {}},
              MakeWrenchComponentAxis(),
          },
  };
}

void SensorArticulatedJointWrench::ReadQuery(Error& error) {
  MOCHI_ERROR_IF_NOT(_initialized, error, "SensorArticulatedJointWrench is not initialized");
  MOCHI_ERROR_RETURN(error);
  Scene* const scene = GetScene();
  if (scene == nullptr) {
    MOCHI_ERROR_SET(error, "SensorArticulatedJointWrench scene is gone");
    return;
  }
  Actor const* const articulated = scene->GetActor(_articulatedHandle);
  if (articulated == nullptr) {
    MOCHI_ERROR_SET(error, "SensorArticulatedJointWrench actor is gone");
    return;
  }
  Actor const& articulatedActor = *articulated;

  for (std::size_t row = 0; row < _layout.rowJointIndices.size(); ++row) {
    int32_t const joint = _layout.rowJointIndices[row];
    Error queryError;
    Real3 const force = articulatedActor.GetArticulatedJointForce(joint, queryError);
    Real3 torque;
    if (static_cast<mochi::Error const&>(queryError).IsOK()) {
      torque = articulatedActor.GetArticulatedJointTorque(joint, queryError);
    }
    if (!static_cast<mochi::Error const&>(queryError).IsOK()) {
      error = queryError.Copy();
      return;
    }
    std::size_t const offset = row * 6;
    _queryValues[offset] = static_cast<float>(force[0]);
    _queryValues[offset + 1] = static_cast<float>(force[1]);
    _queryValues[offset + 2] = static_cast<float>(force[2]);
    _queryValues[offset + 3] = static_cast<float>(torque[0]);
    _queryValues[offset + 4] = static_cast<float>(torque[1]);
    _queryValues[offset + 5] = static_cast<float>(torque[2]);
  }
}

SensorStepResult SensorArticulatedJointWrench::Step(StepTime const& now, Error& error) {
  if (now.dt == 0.0) {
    return SensorStepResult::NoUpdate;
  }
  ReadQuery(error);
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  PublishSample(now, MakeConstSpan(_queryValues), error);
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  return SensorStepResult::Published;
}

Span<std::byte const> SensorArticulatedJointWrench::TryGetMetadata(std::string_view key) const {
  if (_initialized && key == SensorMetadataKey<ArticulatedJointWrenchMeta>::Key) {
    return {reinterpret_cast<std::byte const*>(&_layout), sizeof(_layout)};
  }
  return {};
}

void SensorArticulatedJointWrench::ConfigureRecording(SensorRecordingSink& sink, Error& error)
    const {
  SensorBase::ConfigureRecording(sink, error);
  constexpr auto typeId = SensorMetadataKey<ArticulatedJointWrenchMeta>::Key;
  constexpr std::string_view path = "metadata/articulated_joint_wrench";
  sink.DeclareMetadataType(typeId, error);
  sink.WriteConstantText(path, "metadata_type", typeId, error);
  sink.WriteConstantText(path, "coordinate_frame", _layout.coordinateFrame, error);
  sink.WriteConstantText(path, "moment_reference", _layout.momentReference, error);
  sink.WriteConstantInt32s(path, "row_joint_indices", _layout.rowJointIndices, error);
  sink.WriteConstantInt32s(path, "parent_link_indices", _layout.parentLinkIndices, error);
  sink.WriteConstantInt32s(
      path, "omitted_static_link_indices", _layout.omittedStaticLinkIndices, error);
}
