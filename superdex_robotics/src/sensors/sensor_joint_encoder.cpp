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

#include <superdex_robotics/sensors/sensor_joint_encoder.h>

#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/file_utils.h>

#include <algorithm>
#include <cstddef>
#include <utility>

using namespace mochi;
using namespace superdex::robotics;

SensorJointEncoder::SensorJointEncoder(Actor* linkActor, std::string_view paramArgs, Error& error)
    : SensorBase(linkActor, error) {
  MOCHI_ERROR_RETURN(error);
  Params const params = paramArgs.empty()
      ? Params{}
      : LoadParamsFromPathOrJson<Params>(paramArgs, error, /*allowPartial=*/false);
  SetParams(params, error);
}

SensorJointEncoder::SensorJointEncoder(Actor* linkActor, Params const& params, Error& error)
    : SensorBase(linkActor, error) {
  SetParams(params, error);
}

SensorJointEncoder::~SensorJointEncoder() = default;

void SensorJointEncoder::SetParams(Params const& params, Error& error) {
  MOCHI_ERROR_RETURN(error);
  bool const initialized = !_dofs.empty();
  std::size_t const joints = initialized ? _dofs.size() : params.joints.size();
  MOCHI_ERROR_IF(
      initialized && params.joints != _params.joints,
      error,
      "SensorJointEncoder joints cannot change after initialization");
  MOCHI_ERROR_IF(
      !params.offset.empty() && joints != 0 && params.offset.size() != joints,
      error,
      "SensorJointEncoder needs one offset per joint, or none");
  MOCHI_ERROR_IF(
      !std::all_of(params.offset.begin(), params.offset.end(), [](real o) { return IsFinite(o); }),
      error,
      "SensorJointEncoder offsets must be finite");
  MOCHI_ERROR_RETURN(error);
  _params = params;
}

SensorFormat SensorJointEncoder::InitSensor(SensorScope const& /*scope*/, Error& error) {
  _dofs.clear();
  MOCHI_ERROR_IF_NOT(Inputs().empty(), error, "SensorJointEncoder accepts no inputs");
  Bot const* const bot = GetOwningBot();
  Actor const* const actor = bot != nullptr ? bot->GetArticulatedActor() : nullptr;
  MOCHI_ERROR_IF(
      actor == nullptr, error, "SensorJointEncoder needs an owning bot with an articulated actor");
  MOCHI_ERROR_RETURN(error, {});
  ArticulatedShapeInfo const shape = actor->GetArticulatedShapeInfo(error);
  MOCHI_ERROR_RETURN(error, {});
  // The articulated joint of each name, read when it has one DOF and is not the root, whose DOFs
  // the bot's pose excludes.
  auto const find = [&](DynamicString const& name) {
    return static_cast<std::size_t>(
        std::find(shape.jointNames.begin(), shape.jointNames.end(), name) -
        shape.jointNames.begin());
  };
  auto const isOneDof = [&](std::size_t joint) {
    return joint < shape.jointNames.size() && shape.parents[joint] >= 0 &&
        shape.dofInfo[joint].GetSize() == 1;
  };
  DynamicArray<std::size_t> joints;
  if (_params.joints.empty()) {
    for (BotJointPrefab const& botJoint : bot->GetBotPrefab().joints) {
      std::size_t const joint = find(botJoint.name);
      if (joint == shape.jointNames.size()) {
        MOCHI_LOG_ERROR(
            "SensorJointEncoder: bot joint '%s' is missing from its articulated actor",
            botJoint.name.c_str());
        MOCHI_ERROR_SET(
            error,
            "SensorJointEncoder bot joint missing from its articulated actor; see the log for it");
        return {};
      }
      if (isOneDof(joint)) {
        joints.push_back(joint);
      }
    }
  }
  for (DynamicString const& name : _params.joints) {
    std::size_t const joint = find(name);
    if (!isOneDof(joint)) {
      MOCHI_LOG_ERROR("SensorJointEncoder: '%s' is not a one-DOF joint of its bot", name.c_str());
      MOCHI_ERROR_SET(
          error,
          "SensorJointEncoder joints must name one-DOF joints of its bot; see the log for it");
      return {};
    }
    joints.push_back(joint);
  }
  MOCHI_ERROR_IF(joints.empty(), error, "SensorJointEncoder reads no joints");
  MOCHI_ERROR_IF(
      !_params.offset.empty() && _params.offset.size() != joints.size(),
      error,
      "SensorJointEncoder needs one offset per joint, or none");
  MOCHI_ERROR_RETURN(error, {});

  SensorAxis jointAxis = {.name = "joint", .labels = {}, .units = {}};
  for (std::size_t const joint : joints) {
    _dofs.push_back(shape.dofInfo[joint].offset);
    jointAxis.labels.push_back(shape.jointNames[joint]);
  }
  // A format has units on one axis only: per row when every joint is of one kind, else none.
  auto const isPrismatic = [&](std::size_t joint) {
    return shape.jointTypes[joint] == ArticulatedJointType::Prismatic;
  };
  SensorAxis quantityAxis = {.name = "quantity", .labels = {"position", "velocity"}, .units = {}};
  if (std::all_of(joints.begin(), joints.end(), isPrismatic)) {
    quantityAxis.units = {"m", "m/s"};
  } else if (std::none_of(joints.begin(), joints.end(), isPrismatic)) {
    quantityAxis.units = {"rad", "rad/s"};
  }
  auto const dofs = static_cast<std::size_t>(actor->GetNumDofs());
  _pose.resize(dofs);
  _velocities.resize(dofs);
  _values.resize(2 * _dofs.size());
  return {
      .kind = "JointState",
      .shape = {2, static_cast<int32_t>(_dofs.size())},
      .axes = {std::move(quantityAxis), std::move(jointAxis)},
  };
}

SensorStepResult SensorJointEncoder::Step(StepTime const& now, Error& error) {
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  Bot const* const bot = GetOwningBot();
  Actor const* const actor = bot != nullptr ? bot->GetArticulatedActor() : nullptr;
  MOCHI_ERROR_IF(actor == nullptr, error, "SensorJointEncoder's articulated actor is gone");
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  actor->GetArticulatedPose(MakeSpan(_pose), error);
  actor->GetArticulatedJointVelocities(MakeSpan(_velocities), error);
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  std::size_t const joints = _dofs.size();
  for (std::size_t i = 0; i < joints; ++i) {
    auto const dof = static_cast<std::size_t>(_dofs[i]);
    real const offset = _params.offset.empty() ? 0_r : _params.offset[i];
    _values[i] = static_cast<float>(_pose[dof] + offset);
    _values[joints + i] = static_cast<float>(_velocities[dof]);
  }
  return PublishSample(now, MakeConstSpan(_values), error) ? SensorStepResult::Published
                                                           : SensorStepResult::NoUpdate;
}

void SensorJointEncoder::ResetSensor() {}
