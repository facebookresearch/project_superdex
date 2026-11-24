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

#include "superdex_asset_tuner/tuner_session.h"

#include "scene_utils.h"
#include "superdex_asset_tuner/load_math.h"

#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/math_utils.h>
#include <superdex_robotics/scene/bot_task.h>
#include <superdex_robotics/superdex_robotics.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace superdex::tuner {

namespace {

// A Pose input: x y z, then qx qy qz qw, in the bot's root frame.
constexpr size_t kPoseInputSize = 7;

using mochi::Actor;
using mochi::ActorType;

std::string JoinNames(Span<DynamicString const> names) {
  std::string joined;
  for (auto const& name : names) {
    joined += joined.empty() ? "" : ", ";
    joined += std::string_view(name);
  }
  return joined;
}

Actor*
FindActor(mochi::Scene& scene, std::string_view name, ActorType type, std::string& outError) {
  DynamicArray<DynamicString> available;
  for (Actor* actor : details::GetTopLevelActors(scene)) {
    if (actor->GetType() != type) {
      continue;
    }
    if (name == actor->GetName()) {
      return actor;
    }
    available.emplace_back(actor->GetName());
  }
  outError = "No " + std::string(type == ActorType::Rigid ? "rigid" : "articulated") +
      " actor named '" + std::string(name) + "'. Available: [" + JoinNames(available) + "]";
  return nullptr;
}

// Splits "actor/part" at the last '/' (actor names may contain '/' from nested prefabs).
bool SplitTarget(
    std::string_view target,
    std::string_view& outActor,
    std::string_view& outPart,
    std::string& outError) {
  size_t const slash = target.rfind('/');
  if (slash == std::string_view::npos || slash == 0 || slash + 1 == target.size()) {
    outError = "Target '" + std::string(target) + "' must be of the form 'actor/name'.";
    return false;
  }
  outActor = target.substr(0, slash);
  outPart = target.substr(slash + 1);
  return true;
}

int FindName(Span<DynamicString const> names, std::string_view name) {
  for (size_t i = 0; i < names.size(); ++i) {
    if (std::string_view(names[i]) == name) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

struct ArticulatedPart {
  Actor* articulated = nullptr;
  mochi::ActorHandle link;
  int index = -1;
};

// Resolves "actor/link" (linkNames) or "actor/joint" (jointNames). Joint i drives link i.
bool ResolveArticulatedPart(
    mochi::Scene& scene,
    std::string_view target,
    bool isJoint,
    ArticulatedPart& out,
    std::string& outError) {
  std::string_view actorName;
  std::string_view partName;
  if (!SplitTarget(target, actorName, partName, outError)) {
    return false;
  }
  out.articulated = FindActor(scene, actorName, ActorType::Articulated, outError);
  if (out.articulated == nullptr) {
    return false;
  }
  mochi::ErrorLog error;
  mochi::ArticulatedShapeInfo const info = out.articulated->GetArticulatedShapeInfo(error);
  Span<DynamicString const> const names = isJoint ? info.jointNames : info.linkNames;
  out.index = FindName(names, partName);
  if (out.index < 0) {
    outError = "Articulated actor '" + std::string(actorName) + "' has no " +
        (isJoint ? "joint" : "link") + " named '" + std::string(partName) + "'. Available: [" +
        JoinNames(names) + "]";
    return false;
  }
  out.link = out.articulated->GetNestedLinkActors(error)[static_cast<size_t>(out.index)];
  return true;
}

Real3 NormalizedOrZero(Real3 const& v) {
  real const norm = mochi::Norm(v);
  return norm > 0 ? v * (1 / norm) : Real3{0.0f, 0.0f, 0.0f};
}

// Of the rotation vectors equivalent to `rotvec` (axis * (angle + 2 pi k)), the one nearest
// `previous`, so the trace stays continuous through +-pi instead of flipping sign.
Real3 UnwrapRotationVector(Real3 const& rotvec, Real3 const& previous) {
  constexpr real kTwoPi = 2 * mochi::kPI;
  // Below this angle the axis is numerical noise, so whole turns are added along the previous axis.
  constexpr real kMinAxisAngle = 1e-3f;
  real const angle = mochi::Norm(rotvec);
  if (angle < kMinAxisAngle) {
    real const turns = std::round(mochi::Norm(previous) / kTwoPi);
    return rotvec + NormalizedOrZero(previous) * (turns * kTwoPi);
  }
  Real3 const axis = rotvec * (1 / angle);
  real const turns = std::round((mochi::Dot(axis, previous) - angle) / kTwoPi);
  return axis * (angle + turns * kTwoPi);
}

// Approximate volume [m^3] by which `actor` overlaps other actors: depth times contact area, summed
// over its overlapping contact points.
real PenetrationVolume(mochi::Scene& scene, Actor const& actor, bool ignoreGround) {
  // The overlap with each other actor can be sampled from either side (or both), so each side is
  // summed separately and the larger kept, rather than counting it twice.
  struct Overlap {
    mochi::ActorHandle other;
    real fromSelf = 0;
    real fromOther = 0;
  };
  DynamicArray<Overlap> overlaps;
  mochi::ActorHandle const self = actor.GetHandle();
  mochi::ErrorLog error;
  for (mochi::ContactPoint const& contact : actor.GetContactPointsWorld(error)) {
    bool const selfIsA = contact.actorA == self;
    mochi::ActorHandle const other = selfIsA ? contact.actorB : contact.actorA;
    if (contact.distance >= 0 || other == self) {
      continue;
    }
    auto it = std::find_if(overlaps.begin(), overlaps.end(), [&](Overlap const& overlap) {
      return overlap.other == other;
    });
    if (it == overlaps.end()) {
      overlaps.push_back({other});
      it = overlaps.end() - 1;
    }
    (selfIsA ? it->fromSelf : it->fromOther) += -contact.distance * contact.intWeight;
  }
  real volume = 0;
  for (Overlap const& overlap : overlaps) {
    Actor const* other = scene.GetActor(overlap.other);
    if (ignoreGround && other != nullptr && other->IsStatic() && !details::IsBounded(*other)) {
      continue;
    }
    volume += std::max(overlap.fromSelf, overlap.fromOther);
  }
  return volume;
}

// A rigid actor, or an articulated link as "actor/link".
Actor* FindBody(mochi::Scene& scene, std::string_view target, std::string& outError) {
  std::string rigidError;
  if (Actor* actor = FindActor(scene, target, ActorType::Rigid, rigidError)) {
    return actor;
  }
  ArticulatedPart part;
  std::string linkError;
  if (target.find('/') == std::string_view::npos ||
      !ResolveArticulatedPart(scene, target, false, part, linkError)) {
    outError = rigidError + (linkError.empty() ? "" : " " + linkError);
    return nullptr;
  }
  return scene.GetActor(part.link);
}

// Total mass of the articulation `body` is a link of, or 0 if it is not a link.
real ArticulationMass(mochi::Scene& scene, Actor const& body) {
  mochi::ErrorLog error;
  for (Actor* actor : details::GetTopLevelActors(scene)) {
    if (actor->GetType() != ActorType::Articulated) {
      continue;
    }
    Span<mochi::ActorHandle const> const links = actor->GetNestedLinkActors(error);
    if (std::find(links.begin(), links.end(), body.GetHandle()) == links.end()) {
      continue;
    }
    real total = 0;
    for (mochi::ActorHandle const link : links) {
      if (Actor const* linkActor = scene.GetActor(link)) {
        total += linkActor->GetMass(error);
      }
    }
    return total;
  }
  return 0;
}

// The load.<name>.drive channel: 0 released, 1 position driven, 2 position and orientation driven.
real DriveChannelValue(LoadSpec const& load, PathDriveMode mode) {
  switch (mode) {
    case PathDriveMode::On:
      return load.drive.rotation ? 2 : 1;
    case PathDriveMode::PositionOnly:
      return 1;
    case PathDriveMode::Off:
      break;
  }
  return 0;
}

// Fails, naming them, if two enabled Controller Inputs drive one controller.
bool CheckOneDriverPerController(Span<LoadSpec const> loads, std::string& outError) {
  for (size_t i = 0; i < loads.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      bool const both = loads[i].enabled && loads[j].enabled &&
          loads[i].type == LoadType::ControllerInput &&
          loads[j].type == LoadType::ControllerInput && loads[i].target == loads[j].target;
      if (both) {
        outError = "Loads '" + std::string(loads[j].name) + "' and '" + std::string(loads[i].name) +
            "' both drive '" + std::string(loads[i].target) + "'.";
        return false;
      }
    }
  }
  return true;
}
} // namespace

int TunerSession::GetOrAddAccumulator(Actor const& actor) {
  mochi::ActorHandle const handle = actor.GetHandle();
  for (size_t i = 0; i < _accumulators.size(); ++i) {
    if (_accumulators[i].actor == handle) {
      return static_cast<int>(i);
    }
  }
  _accumulators.emplace_back();
  Accumulator& accumulator = _accumulators.back();
  accumulator.actor = handle;
  accumulator.forces.resize(static_cast<size_t>(actor.GetNumDofs()), 0);
  return static_cast<int>(_accumulators.size() - 1);
}

void TunerSession::AddChannels(
    std::string_view prefix,
    std::initializer_list<std::string_view> suffixes) {
  for (std::string_view const suffix : suffixes) {
    DynamicString name(prefix);
    name += '.';
    name += suffix;
    _channelNames.push_back(std::move(name));
  }
}

bool TunerSession::ResolveLoad(mochi::Scene& scene, LoadSpec const& load, std::string& outError) {
  _resolvedLoads.emplace_back();
  ResolvedLoad& resolved = _resolvedLoads.back();
  switch (load.type) {
    case LoadType::Force:
    case LoadType::Torque: {
      Actor* body = FindBody(scene, load.target, outError);
      if (body == nullptr) {
        return false;
      }
      if (body->IsStatic()) {
        outError = "'" + std::string(load.target) + "' is static.";
        return false;
      }
      resolved.body = body->GetHandle();
      // A link's load becomes joint forces of its articulated actor, through the link Jacobian.
      resolved.isLink = body->IsNestedLinkActor();
      mochi::ErrorLog error;
      Actor const* owner =
          resolved.isLink ? scene.GetActor(body->GetArticulatedActor(error)) : body;
      resolved.accumulator = GetOrAddAccumulator(*owner);
      return true;
    }
    case LoadType::JointEffort: {
      ArticulatedPart part;
      if (!ResolveArticulatedPart(scene, load.target, true, part, outError)) {
        return false;
      }
      resolved.body = part.link;
      resolved.accumulator = GetOrAddAccumulator(*part.articulated);
      mochi::ErrorLog error;
      auto const& dofInfo =
          part.articulated->GetArticulatedShapeInfo(error).dofInfo[static_cast<size_t>(part.index)];
      resolved.dofOffset = dofInfo.offset;
      resolved.dofCount = dofInfo.transSize + dofInfo.rotSize;
      if (resolved.dofCount < 1 || resolved.dofCount > 3) {
        outError = "Joint '" + std::string(load.target) + "' has " +
            std::to_string(resolved.dofCount) +
            " DoFs; JointEffort supports 1-3. Use Force or Torque on its link instead.";
        return false;
      }
      return true;
    }
    case LoadType::PosePath:
      return ResolvePosePath(scene, load, resolved, outError);
    case LoadType::ControllerInput:
      return ResolveControllerInput(load, resolved, outError);
  }
  outError = "Unknown load type.";
  return false;
}

bool TunerSession::BindControllers(robotics::BotScene& bots, std::string& outError) {
  for (details::FloatInputController const& found : details::FindFloatInputControllers(bots)) {
    Actor* const robot = found.controller->GetActor();
    _controllers.emplace_back();
    BoundController& bound = _controllers.back();
    bound.name = found.name;
    bound.controller = found.controller;
    bound.accumulator = GetOrAddAccumulator(*robot);
    bound.input.resize(static_cast<size_t>(found.controller->GetInputSize()));
    bound.efforts.resize(static_cast<size_t>(robot->GetNumDofs()));
    bound.scratch.resize(bound.input.size());
    mochi::Error error;
    found.controller->GetCurrentInput(mochi::MakeSpan(bound.input), error);
    if (!error.IsOK()) {
      outError = "Controller '" + std::string(bound.name) + "': " + error.GetDescription();
      return false;
    }
    bound.start = bound.input;
  }
  return true;
}

bool TunerSession::ResolveControllerInput(
    LoadSpec const& load,
    ResolvedLoad& resolved,
    std::string& outError) {
  DynamicArray<DynamicString> names;
  for (BoundController const& bound : _controllers) {
    names.push_back(bound.name);
  }
  resolved.controller = FindName(names, load.target);
  if (resolved.controller < 0) {
    outError = "No bot controller with a float input named '" + std::string(load.target) +
        "'. Available: [" + JoinNames(names) + "]";
    return false;
  }
  BoundController const& bound = _controllers[static_cast<size_t>(resolved.controller)];
  robotics::ControllerBase const& controller = *bound.controller;
  // Also called mid-run (UpdateLoads), so everything is rebuilt from the run's starting input.
  DynamicArray<real> const& start = bound.start;
  resolved.waypoints.clear();
  resolved.inputIndices.clear();
  resolved.inputKeys.clear();
  if (controller.GetInputKind() == robotics::ControllerInputKind::Pose) {
    if (load.waypoints.empty() || !load.inputKeys.empty() || !load.joints.empty()) {
      outError = "'" + std::string(load.target) + "' takes a tool pose: give waypoints only.";
      return false;
    }
    if (start.size() != kPoseInputSize) {
      outError = "'" + std::string(load.target) + "' reports a pose input of " +
          std::to_string(start.size()) + " values, not x y z qx qy qz qw.";
      return false;
    }
    // The path starts at the tool's pose, so a first waypoint after t = 0 is approached from it.
    if (load.waypoints.front().time > 0) {
      PoseWaypoint first = load.waypoints.front();
      first.time = 0;
      first.position = Real3{start[0], start[1], start[2]};
      first.rotation = mochi::Quaternion(start[3], start[4], start[5], start[6]).ToRotationVector();
      resolved.waypoints.push_back(first);
    }
    resolved.waypoints.append(load.waypoints);
    return true;
  }
  if (load.inputKeys.empty() || !load.waypoints.empty()) {
    outError = "'" + std::string(load.target) + "' takes input values: give inputKeys only.";
    return false;
  }
  if (load.joints.empty()) {
    for (int i = 0; i < isize(start); ++i) {
      resolved.inputIndices.push_back(i);
    }
  } else {
    if (controller.GetInputKind() != robotics::ControllerInputKind::JointPositions) {
      outError = "Only a joint-position input can name joints.";
      return false;
    }
    mochi::ErrorLog error;
    mochi::ArticulatedShapeInfo const info = controller.GetActor()->GetArticulatedShapeInfo(error);
    Span<int const> const dofs = controller.GetControlledDofs();
    for (DynamicString const& joint : load.joints) {
      int const j = FindName(info.jointNames, joint);
      int input = -1;
      if (j >= 0 && info.dofInfo[static_cast<size_t>(j)].GetSize() == 1) {
        auto const* const found =
            std::find(dofs.begin(), dofs.end(), info.dofInfo[static_cast<size_t>(j)].offset);
        input = found != dofs.end() ? static_cast<int>(found - dofs.begin()) : -1;
      }
      if (input < 0) {
        outError = "'" + std::string(load.target) + "' does not drive a one-DoF joint named '" +
            std::string(joint) + "'. Joints: [" + JoinNames(info.jointNames) + "]";
        return false;
      }
      if (std::find(resolved.inputIndices.begin(), resolved.inputIndices.end(), input) !=
          resolved.inputIndices.end()) {
        outError = "'" + std::string(load.target) + "' lists joint '" + std::string(joint) +
            "' more than once.";
        return false;
      }
      resolved.inputIndices.push_back(input);
    }
  }
  for (InputKey const& key : load.inputKeys) {
    if (key.values.size() != resolved.inputIndices.size()) {
      outError = "Each input key needs " + std::to_string(resolved.inputIndices.size()) +
          " values (one per joint, or the whole input), not " + std::to_string(key.values.size()) +
          ".";
      return false;
    }
  }
  if (load.inputKeys.front().time > 0) {
    InputKey first = load.inputKeys.front();
    first.time = 0;
    for (size_t k = 0; k < resolved.inputIndices.size(); ++k) {
      first.values[k] = start[static_cast<size_t>(resolved.inputIndices[k])];
    }
    resolved.inputKeys.push_back(first);
  }
  resolved.inputKeys.append(load.inputKeys);
  return true;
}

void TunerSession::DriveControllerInput(
    ResolvedLoad const& resolved,
    AppliedLoad& applied,
    double t) {
  BoundController& bound = _controllers[static_cast<size_t>(resolved.controller)];
  if (!resolved.waypoints.empty()) {
    mochi::TransformRT const pose = EvaluatePosePath(resolved.waypoints, t);
    Real3 const translation = pose.GetTranslation();
    for (int i = 0; i < 3; ++i) {
      bound.input[static_cast<size_t>(i)] = translation[i];
    }
    for (int i = 0; i < 4; ++i) {
      bound.input[static_cast<size_t>(i) + 3] = pose.GetRotation().data[i];
    }
    if (Actor const* robot = bound.controller->GetActor()) {
      applied.targetPose = robot->GetRootTransform() * pose;
    }
    return;
  }
  Span<real> const values{bound.scratch.data(), resolved.inputIndices.size()};
  EvaluateInputKeys(resolved.inputKeys, t, values);
  for (size_t k = 0; k < values.size(); ++k) {
    bound.input[static_cast<size_t>(resolved.inputIndices[k])] = values[k];
  }
}

void TunerSession::SampleControllerInput(
    LoadSpec const& load,
    ResolvedLoad const& resolved,
    AppliedLoad& applied) {
  BoundController& bound = _controllers[static_cast<size_t>(resolved.controller)];
  Actor const* robot = bound.controller->GetActor();
  if (robot == nullptr) {
    return;
  }
  mochi::ErrorLog error;
  bound.controller->GetCurrentInput(mochi::MakeSpan(bound.scratch), error);
  applied.input = bound.scratch;
  if (!load.enabled) {
    return;
  }
  if (!resolved.waypoints.empty()) {
    mochi::TransformRT const rootFromTool{
        mochi::Quaternion(bound.scratch[3], bound.scratch[4], bound.scratch[5], bound.scratch[6]),
        Real3{bound.scratch[0], bound.scratch[1], bound.scratch[2]}};
    mochi::TransformRT const worldFromTool = robot->GetRootTransform() * rootFromTool;
    applied.pointWorld = worldFromTool.GetTranslation();
    applied.trackingError = mochi::Norm(applied.targetPose.GetTranslation() - applied.pointWorld);
    applied.angleError =
        mochi::Norm((applied.targetPose.GetRotation() * worldFromTool.GetRotation().GetConjugate())
                        .ToRotationVector());
    return;
  }
  applied.pointWorld = robot->GetRootTransform().GetTranslation();
  for (int const index : resolved.inputIndices) {
    auto const i = static_cast<size_t>(index);
    applied.trackingError =
        std::max(applied.trackingError, std::abs(bound.input[i] - bound.scratch[i]));
  }
}

bool TunerSession::ResolvePosePath(
    mochi::Scene& scene,
    LoadSpec const& load,
    ResolvedLoad& resolved,
    std::string& outError) {
  Actor* body = FindBody(scene, load.target, outError);
  if (body == nullptr) {
    return false;
  }
  if (body->IsStatic()) {
    outError = "'" + std::string(load.target) + "' is static.";
    return false;
  }
  if (load.waypoints.empty()) {
    outError = "A pose path needs at least one waypoint.";
    return false;
  }
  resolved.body = body->GetHandle();
  mochi::ErrorLog error;
  DriveScale const scale = MakeDriveScale(
      body->GetMass(error),
      MeanDiagonalInertia(body->GetRigidMomentOfInertiaLocal(error)),
      ArticulationMass(scene, *body));
  resolved.driveMass = scale.mass;
  resolved.driveInertia = scale.inertia;

  // Created at the path's start, inert until the first step sets their gains and targets.
  mochi::TransformRT const start = EvaluatePosePath(load.waypoints, 0.0);
  mochi::RigidPivotPositionConstraintParams position;
  position.actor = resolved.body;
  position.localPosition = load.pointLocal;
  position.targetPosition = start.GetTranslation();
  position.stiffness = 0;
  if (mochi::Constraint* constraint = scene.CreateRigidPivotPositionConstraint(position, error)) {
    // The force query lives on the constraint, so destroying it cancels the query.
    constraint->RegisterQuery(mochi::QueryType::ConstraintForce, error);
    resolved.positionDrive = constraint->GetHandle();
  }
  mochi::RigidPivotRotationConstraintParams rotation;
  rotation.actor = resolved.body;
  rotation.targetRotation = start.GetRotation().ToRotationVector();
  rotation.stiffness = 0;
  if (mochi::Constraint* constraint = scene.CreateRigidPivotRotationConstraint(rotation, error)) {
    resolved.rotationDrive = constraint->GetHandle();
  }
  if (!resolved.positionDrive.IsValid() || !resolved.rotationDrive.IsValid()) {
    scene.DestroyConstraint(resolved.positionDrive);
    scene.DestroyConstraint(resolved.rotationDrive);
    resolved.positionDrive = {};
    resolved.rotationDrive = {};
    outError = "Could not attach the drive to '" + std::string(load.target) + "'.";
    return false;
  }
  resolved.driveGains = {0, 0, -1, 0, 0, -1};
  return true;
}

void TunerSession::DrivePosePath(
    mochi::Scene& scene,
    LoadSpec const& load,
    ResolvedLoad& resolved,
    AppliedLoad& applied,
    double timeStep) {
  mochi::Constraint* position = scene.GetConstraint(resolved.positionDrive);
  mochi::Constraint* rotation = scene.GetConstraint(resolved.rotationDrive);
  if (position == nullptr || rotation == nullptr) {
    return;
  }
  bool const active = load.enabled && !load.waypoints.empty();
  // Mid-step, so a waypoint's mode applies from the step starting at its time despite time drift.
  PathDriveMode const mode =
      active ? EvaluatePathDrive(load.waypoints, _time + 0.5 * timeStep) : PathDriveMode::Off;
  applied.driveMode = mode;
  PathDrive const& drive = load.drive;
  std::array<real, 3> const inert{0, 0, -1};
  std::array<real, 3> const positionGains = mode != PathDriveMode::Off
      ? SpringGains(resolved.driveMass, drive.responseHz, drive.dampingRatio, drive.maxAcceleration)
      : inert;
  std::array<real, 3> const rotationGains = mode == PathDriveMode::On && drive.rotation
      ? SpringGains(
            resolved.driveInertia,
            drive.rotResponseHz,
            drive.rotDampingRatio,
            drive.maxAngularAcceleration)
      : inert;
  std::array<real, 6> const gains{
      positionGains[0],
      positionGains[1],
      positionGains[2],
      rotationGains[0],
      rotationGains[1],
      rotationGains[2]};
  mochi::ErrorLog error;
  if (gains != resolved.driveGains) {
    position->SetStiffness(gains[0], error);
    position->SetDamping(gains[1], error);
    position->SetSaturation(gains[2], error);
    rotation->SetStiffness(gains[3], error);
    rotation->SetDamping(gains[4], error);
    rotation->SetSaturation(gains[5], error);
    resolved.driveGains = gains;
  }
  if (!active) {
    return;
  }
  // The implicit solve pulls toward where the path is at the end of this step; the change in
  // target since the last step also gives the drive the path's velocity. Targets keep moving while
  // the drive is off, so it re-engages at the path's velocity rather than with a jump.
  applied.targetPose = EvaluatePosePath(load.waypoints, _time + timeStep);
  position->SetTargetPosition(applied.targetPose.GetTranslation(), error);
  rotation->SetTargetRotation(applied.targetPose.GetRotation(), error);
}

void TunerSession::SamplePosePath(
    mochi::Scene& scene,
    LoadSpec const& load,
    ResolvedLoad const& resolved,
    AppliedLoad& applied) const {
  Actor const* body = scene.GetActor(resolved.body);
  mochi::Constraint const* position = scene.GetConstraint(resolved.positionDrive);
  if (body == nullptr || position == nullptr) {
    return;
  }
  mochi::TransformRT const worldFromBody = body->GetRootTransform();
  applied.pointWorld = worldFromBody.TransformPoint(load.pointLocal);
  if (!load.enabled || load.waypoints.empty()) {
    applied.targetPose = mochi::TransformRT(worldFromBody.GetRotation(), applied.pointWorld);
    return;
  }
  if (applied.driveMode != PathDriveMode::Off) {
    applied.trackingError = mochi::Norm(applied.targetPose.GetTranslation() - applied.pointWorld);
  }
  if (applied.driveMode == PathDriveMode::On) {
    applied.angleError =
        mochi::Norm((applied.targetPose.GetRotation() * worldFromBody.GetRotation().GetConjugate())
                        .ToRotationVector());
  }
  mochi::ErrorLog error;
  Span<real const> const force = position->GetForce(error);
  if (force.size() >= 3) {
    applied.vectorWorld = Real3{force[0], force[1], force[2]};
    applied.magnitude = mochi::Norm(applied.vectorWorld);
  }
}

bool TunerSession::ResolveProbe(
    mochi::Scene& scene,
    ProbeSpec const& probe,
    std::string& outError) {
  _probes.emplace_back();
  ResolvedProbe& resolved = _probes.back();
  resolved.kind = probe.kind;
  std::string_view const target = probe.target;
  mochi::ErrorLog error;
  switch (probe.kind) {
    case ProbeKind::BodyPose:
    case ProbeKind::BodyVelocity:
    case ProbeKind::ContactForce:
    case ProbeKind::Penetration: {
      Actor* actor = FindBody(scene, target, outError);
      if (actor == nullptr) {
        return false;
      }
      resolved.actor = actor->GetHandle();
      if (probe.kind == ProbeKind::BodyPose) {
        AddChannels(
            target,
            {"pos.x",
             "pos.y",
             "pos.z",
             "rotvec.x",
             "rotvec.y",
             "rotvec.z",
             "rotvecUnwrapped.x",
             "rotvecUnwrapped.y",
             "rotvecUnwrapped.z"});
      } else if (probe.kind == ProbeKind::BodyVelocity) {
        AddChannels(target, {"vel.x", "vel.y", "vel.z", "angvel.x", "angvel.y", "angvel.z"});
      } else if (probe.kind == ProbeKind::Penetration) {
        if (!actor->IsQuerySupported(mochi::QueryType::ContactPoints)) {
          outError = "'" + std::string(target) +
              "' does not report contact points, so its penetration cannot be measured.";
          return false;
        }
        resolved.query = actor->RegisterQuery(mochi::QueryType::ContactPoints, error);
        resolved.ignoreGround = probe.ignoreGround;
        AddChannels(target, {"penetration.volume"});
      } else {
        if (!actor->IsQuerySupported(mochi::QueryType::TotalContactForce)) {
          outError = "'" + std::string(target) + "' does not report contact forces.";
          return false;
        }
        resolved.query = actor->RegisterQuery(mochi::QueryType::TotalContactForce, error);
        AddChannels(target, {"contactForce.x", "contactForce.y", "contactForce.z"});
      }
      return true;
    }
    case ProbeKind::JointState: {
      ArticulatedPart part;
      if (!ResolveArticulatedPart(scene, target, true, part, outError)) {
        return false;
      }
      resolved.actor = part.articulated->GetHandle();
      auto const& dofInfo =
          part.articulated->GetArticulatedShapeInfo(error).dofInfo[static_cast<size_t>(part.index)];
      int const count = dofInfo.transSize + dofInfo.rotSize;
      for (int i = 0; i < count; ++i) {
        resolved.dofIndices.push_back(dofInfo.offset + i);
        std::string const index = std::to_string(i);
        AddChannels(target, {"q" + index, "qd" + index});
      }
      resolved.dofValues.resize(static_cast<size_t>(count), 0);
      resolved.scratch.resize(static_cast<size_t>(part.articulated->GetNumDofs()), 0);
      return true;
    }
    case ProbeKind::RelativePose: {
      Actor* actor = FindBody(scene, target, outError);
      if (actor == nullptr) {
        return false;
      }
      Actor* frame = FindBody(scene, probe.frame, outError);
      if (frame == nullptr) {
        outError = "Frame: " + outError;
        return false;
      }
      resolved.actor = actor->GetHandle();
      resolved.frame = frame->GetHandle();
      DynamicString const prefix = DynamicString(target) + ".in." + probe.frame;
      AddChannels(prefix, {"pos.x", "pos.y", "pos.z", "rotvec.x", "rotvec.y", "rotvec.z"});
      return true;
    }
    case ProbeKind::SolverStats:
      AddChannels("solver", {"iters", "residual", "lineSearchIters", "convergence"});
      return true;
    case ProbeKind::Performance:
      AddChannels("perf", {"solveMs"});
      return true;
  }
  outError = "Unknown probe kind.";
  return false;
}

bool TunerSession::Bind(
    mochi::Scene& scene,
    TunerSpec const& spec,
    std::string& outError,
    robotics::BotScene* bots) {
  Unbind(scene);
  if (bots != nullptr && !BindControllers(*bots, outError)) {
    Unbind(scene);
    return false;
  }
  _bots = bots;
  _loads = spec.loads;
  for (LoadSpec& load : _loads) {
    SortByTime(load);
  }
  if (!CheckOneDriverPerController(_loads, outError)) {
    Unbind(scene);
    return false;
  }
  for (size_t i = 0; i < _loads.size(); ++i) {
    LoadSpec const& load = _loads[i];
    if (!ResolveLoad(scene, load, outError)) {
      if (!load.enabled) {
        // A disabled load may point at something that cannot take loads (e.g. a body made static by
        // an override). Keep its slot so indices line up; it is never applied.
        _resolvedLoads.back() = {};
        outError.clear();
      } else {
        outError = "Load '" + std::string(load.name) + "': " + outError;
        Unbind(scene);
        return false;
      }
    }
    DynamicString const prefix = DynamicString("load.") + load.name;
    AddChannels(prefix, {"mag"});
    if (load.type == LoadType::PosePath) {
      AddChannels(prefix, {"error", "angleError", "drive"});
    } else if (load.type == LoadType::ControllerInput) {
      AddChannels(prefix, {"error", "angleError"});
    }
  }
  _appliedLoads.resize(_loads.size());
  for (ProbeSpec const& probe : spec.probes) {
    size_t const firstChannel = _channelNames.size();
    if (!ResolveProbe(scene, probe, outError)) {
      outError = "Probe '" + std::string(probe.target) + "': " + outError;
      Unbind(scene);
      return false;
    }
    _probes.back().numChannels = _channelNames.size() - firstChannel;
  }
  // After the probes: threshold conditions refer to their channels.
  for (size_t i = 0; i < spec.stopConditions.size(); ++i) {
    StopCondition const& stop = spec.stopConditions[i];
    if (!stop.enabled) {
      continue;
    }
    if (!ResolveStop(scene, stop, outError)) {
      outError = "Stop condition '" + std::string(stop.name) + "': " + outError;
      Unbind(scene);
      return false;
    }
    _stops.back().specIndex = static_cast<int>(i);
  }
  return true;
}

bool TunerSession::ResolveStop(
    mochi::Scene& scene,
    StopCondition const& stop,
    std::string& outError) {
  ResolvedStop resolved;
  resolved.condition = stop;
  switch (stop.kind) {
    case StopKind::LeavesBox:
    case StopKind::EntersBox: {
      Actor* body = FindBody(scene, stop.target, outError);
      if (body == nullptr) {
        return false;
      }
      resolved.body = body->GetHandle();
      break;
    }
    case StopKind::ChannelThreshold: {
      resolved.channel = FindName(_channelNames, stop.channel);
      if (resolved.channel < 0) {
        outError = "No channel named '" + std::string(stop.channel) +
            "'. Add a probe that produces it. Available: [" + JoinNames(_channelNames) + "]";
        return false;
      }
      break;
    }
  }
  _stops.push_back(std::move(resolved));
  return true;
}

void TunerSession::CheckStopConditions(mochi::Scene& scene, Sample& inOutSample) {
  constexpr double kTimeEpsilon = 1e-9;
  char message[256];
  for (ResolvedStop& stop : _stops) {
    StopCondition const& condition = stop.condition;
    if (_time < condition.start) {
      stop.holdingSince = -1.0;
      continue;
    }
    bool holds = false;
    switch (condition.kind) {
      case StopKind::LeavesBox:
      case StopKind::EntersBox: {
        Actor const* body = scene.GetActor(stop.body);
        if (body == nullptr) {
          break; // the body is gone: the condition cannot hold
        }
        Real3 const p = body->GetRootTransform().GetTranslation();
        bool inside = true;
        for (int d = 0; d < 3; ++d) {
          inside = inside && p[d] >= condition.boxMin[d] && p[d] <= condition.boxMax[d];
        }
        bool const enters = condition.kind == StopKind::EntersBox;
        holds = enters ? inside : !inside;
        if (holds) {
          std::snprintf(
              message,
              sizeof(message),
              "'%s': %s %s the box at (%.3g, %.3g, %.3g)",
              condition.name.c_str(),
              condition.target.c_str(),
              enters ? "entered" : "left",
              static_cast<double>(p[0]),
              static_cast<double>(p[1]),
              static_cast<double>(p[2]));
        }
        break;
      }
      case StopKind::ChannelThreshold: {
        real const value = inOutSample.values[static_cast<size_t>(stop.channel)];
        bool const above = condition.comparison == StopComparison::Above;
        holds = above ? value > condition.threshold : value < condition.threshold;
        if (holds) {
          std::snprintf(
              message,
              sizeof(message),
              "'%s': %s = %.4g %s %.4g",
              condition.name.c_str(),
              condition.channel.c_str(),
              static_cast<double>(value),
              above ? ">" : "<",
              static_cast<double>(condition.threshold));
        }
        break;
      }
    }
    if (!holds) {
      stop.holdingSince = -1.0;
      continue;
    }
    if (stop.holdingSince < 0) {
      stop.holdingSince = _time;
    }
    // Every condition keeps its hold timer up to date; the first triggered in spec order wins.
    if (_time - stop.holdingSince + kTimeEpsilon < condition.holdTime ||
        inOutSample.stopCondition >= 0) {
      continue;
    }
    std::string text = message;
    if (condition.holdTime > 0) {
      char held[48];
      std::snprintf(held, sizeof(held), " for %.3g s", condition.holdTime);
      text += held;
    }
    inOutSample.stopCondition = stop.specIndex;
    inOutSample.stopMessage = DynamicString(text);
  }
}

void TunerSession::Unbind(mochi::Scene& scene) {
  for (Accumulator const& accumulator : _accumulators) {
    if (Actor* actor = scene.GetActor(accumulator.actor)) {
      actor->ClearExternalForces();
    }
  }
  for (ResolvedLoad const& resolved : _resolvedLoads) {
    scene.DestroyConstraint(resolved.positionDrive);
    scene.DestroyConstraint(resolved.rotationDrive);
  }
  for (ResolvedProbe const& probe : _probes) {
    if (probe.query.IsValid()) {
      if (Actor* actor = scene.GetActor(probe.actor)) {
        actor->CancelQuery(probe.query);
      }
    }
  }
  _loads.clear();
  _resolvedLoads.clear();
  _appliedLoads.clear();
  _accumulators.clear();
  _probes.clear();
  _stops.clear();
  _channelNames.clear();
  _controllers.clear();
  _bots = nullptr;
  _failure.clear();
  _time = 0.0;
}

bool TunerSession::UpdateLoads(Span<LoadSpec const> loads, std::string& outError) {
  bool compatible = loads.size() == _loads.size();
  for (size_t i = 0; compatible && i < loads.size(); ++i) {
    // A pose path's drive is attached at its point, so moving the point needs a new run.
    compatible = loads[i].name == _loads[i].name && loads[i].type == _loads[i].type &&
        loads[i].target == _loads[i].target &&
        (loads[i].type != LoadType::PosePath || loads[i].pointLocal == _loads[i].pointLocal);
  }
  if (!compatible) {
    outError =
        "Adding, removing, renaming or retargeting loads, or moving a pose path's point, takes "
        "effect on the next run.";
    return false;
  }
  if (!CheckOneDriverPerController(loads, outError)) {
    return false;
  }
  // A controller input drives from its resolved path or keys, so those are rebuilt; all of the
  // update applies or none of it.
  DynamicArray<LoadSpec> updated(loads.begin(), loads.end());
  DynamicArray<ResolvedLoad> resolved = _resolvedLoads;
  for (size_t i = 0; i < updated.size(); ++i) {
    SortByTime(updated[i]);
    if (updated[i].type == LoadType::ControllerInput &&
        !ResolveControllerInput(updated[i], resolved[i], outError)) {
      if (updated[i].enabled) {
        outError = "Load '" + std::string(updated[i].name) + "': " + outError;
        return false;
      }
      // As in Bind, a disabled load that does not resolve is never applied.
      resolved[i] = {};
      outError.clear();
    }
  }
  _loads = std::move(updated);
  _resolvedLoads = std::move(resolved);
  return true;
}

void TunerSession::OnPreStep(mochi::StepInfo const& info) {
  mochi::Scene& scene = *info.scene;
  for (Accumulator& accumulator : _accumulators) {
    std::fill(accumulator.forces.begin(), accumulator.forces.end(), real{0});
  }
  mochi::ErrorLog error;
  for (size_t i = 0; i < _loads.size(); ++i) {
    LoadSpec const& load = _loads[i];
    ResolvedLoad& resolved = _resolvedLoads[i];
    AppliedLoad& applied = _appliedLoads[i];
    applied = {};
    if (load.type == LoadType::ControllerInput) {
      if (load.enabled && resolved.controller >= 0) {
        // Toward where the input is at the end of this step, as a pose path's drive pulls.
        DriveControllerInput(resolved, applied, _time + info.timeStepSec);
      }
      continue;
    }
    if (!resolved.body.IsValid()) {
      continue;
    }
    if (load.type == LoadType::PosePath) {
      DrivePosePath(scene, load, resolved, applied, info.timeStepSec);
      continue;
    }
    Actor* body = scene.GetActor(resolved.body);
    if (body == nullptr) {
      continue; // the body is gone
    }
    mochi::TransformRT const worldFromBody = body->GetRootTransform();
    applied.pointWorld = worldFromBody.GetTranslation();
    if (!load.enabled) {
      continue;
    }
    applied.magnitude = EvaluateProfile(load.profile, _time);
    Span<real> const forces =
        mochi::MakeSpan(_accumulators[static_cast<size_t>(resolved.accumulator)].forces);
    Real3 const loadDirection = EvaluateLoadDirection(load, _time);
    Real3 const direction = load.frame == LoadFrame::Local
        ? worldFromBody.TransformDirection(loadDirection)
        : loadDirection;
    switch (load.type) {
      case LoadType::Force: {
        Real3 const force = NormalizedOrZero(direction) * applied.magnitude;
        Real3 const com = body->GetCenterOfMassTransform(error).GetTranslation();
        Real3 const point = EvaluateLoadPoint(load, _time);
        applied.pointWorld =
            load.frame == LoadFrame::WorldPoint ? point : worldFromBody.TransformPoint(point);
        applied.vectorWorld = force;
        if (resolved.isLink) {
          AddLinkForceToGeneralized(
              body->GetArticulatedJacobian(error), force, applied.pointWorld, com, forces);
        } else {
          auto const dofs = RigidPointForceToDofs(force, applied.pointWorld, com);
          for (size_t d = 0; d < dofs.size(); ++d) {
            forces[d] += dofs[d];
          }
        }
        break;
      }
      case LoadType::Torque: {
        applied.vectorWorld = NormalizedOrZero(direction) * applied.magnitude;
        if (resolved.isLink) {
          AddLinkTorqueToGeneralized(
              body->GetArticulatedJacobian(error), applied.vectorWorld, forces);
        } else {
          for (size_t d = 0; d < 3; ++d) {
            forces[3 + d] += applied.vectorWorld[d];
          }
        }
        break;
      }
      case LoadType::JointEffort: {
        for (int d = 0; d < resolved.dofCount; ++d) {
          real const effort = loadDirection[d] * applied.magnitude;
          applied.vectorWorld[d] = effort;
          forces[static_cast<size_t>(resolved.dofOffset + d)] += effort;
        }
        break;
      }
      case LoadType::PosePath:
      case LoadType::ControllerInput:
        break; // driven above
    }
  }

  for (BoundController& bound : _controllers) {
    mochi::Error controllerError;
    bound.controller->ComputeOutputFromInput(
        mochi::MakeConstSpan(bound.input),
        static_cast<real>(info.timeStepSec),
        mochi::MakeSpan(bound.efforts),
        controllerError);
    if (!controllerError.IsOK()) {
      if (_failure.empty()) {
        _failure =
            "Controller '" + std::string(bound.name) + "': " + controllerError.GetDescription();
      }
      continue;
    }
    Span<real> const forces =
        mochi::MakeSpan(_accumulators[static_cast<size_t>(bound.accumulator)].forces);
    for (size_t d = 0; d < forces.size(); ++d) {
      forces[d] += bound.efforts[d];
    }
  }
  for (size_t i = 0; i < _loads.size(); ++i) {
    ResolvedLoad const& resolved = _resolvedLoads[i];
    if (_loads[i].type == LoadType::ControllerInput && resolved.controller >= 0) {
      Span<real const> const efforts =
          mochi::MakeConstSpan(_controllers[static_cast<size_t>(resolved.controller)].efforts);
      real sumSq = 0;
      for (real const effort : efforts) {
        sumSq += effort * effort;
      }
      // A disabled load reads 0, as every other load does.
      _appliedLoads[i].magnitude = _loads[i].enabled ? std::sqrt(sumSq) : real{0};
    }
  }

  for (Accumulator& accumulator : _accumulators) {
    accumulator.dofs.clear();
    accumulator.values.clear();
    for (size_t d = 0; d < accumulator.forces.size(); ++d) {
      if (accumulator.forces[d] != 0) {
        accumulator.dofs.push_back(static_cast<int>(d));
        accumulator.values.push_back(accumulator.forces[d]);
      }
    }
    Actor* actor = scene.GetActor(accumulator.actor);
    if (actor == nullptr) {
      continue;
    }
    if (accumulator.dofs.empty()) {
      actor->ClearExternalForces();
    } else {
      actor->SetExternalForcesOnDofs(
          mochi::MakeConstSpan(accumulator.dofs), mochi::MakeConstSpan(accumulator.values), error);
    }
  }
}

void TunerSession::OnPostStep(mochi::StepInfo const& info, Sample& outSample) {
  _time += info.timeStepSec;
  mochi::Scene& scene = *info.scene;
  mochi::ErrorLog error;
  outSample.time = _time;
  outSample.failure = DynamicString(_failure);
  // The bot scene decided its task in a post-step callback that ran before this one.
  robotics::TerminationOutcome const task =
      _bots != nullptr ? _bots->GetTerminationOutcome() : robotics::TerminationOutcome{};
  outSample.task = task.failureAuthored && task.failure ? TaskOutcome::Failure
      : task.successAuthored && task.success            ? TaskOutcome::Success
      : task.successAuthored                            ? TaskOutcome::Pending
                                                        : TaskOutcome::None;
  outSample.values.clear();
  outSample.values.reserve(_channelNames.size());
  auto push3 = [&](Real3 const& v) {
    for (int d = 0; d < 3; ++d) {
      outSample.values.push_back(v[d]);
    }
  };
  for (size_t i = 0; i < _appliedLoads.size(); ++i) {
    AppliedLoad& applied = _appliedLoads[i];
    bool const isPath = _loads[i].type == LoadType::PosePath;
    bool const isInput = _loads[i].type == LoadType::ControllerInput;
    if (isPath && _resolvedLoads[i].body.IsValid()) {
      SamplePosePath(scene, _loads[i], _resolvedLoads[i], applied);
    }
    if (isInput && _resolvedLoads[i].controller >= 0) {
      SampleControllerInput(_loads[i], _resolvedLoads[i], applied);
    }
    outSample.values.push_back(applied.magnitude);
    if (isPath) {
      outSample.values.push_back(applied.trackingError);
      outSample.values.push_back(applied.angleError);
      outSample.values.push_back(DriveChannelValue(_loads[i], applied.driveMode));
    } else if (isInput) {
      outSample.values.push_back(applied.trackingError);
      outSample.values.push_back(applied.angleError);
    }
  }
  for (ResolvedProbe& probe : _probes) {
    Actor* actor = probe.actor.IsValid() ? scene.GetActor(probe.actor) : nullptr;
    if (probe.actor.IsValid() && actor == nullptr) {
      // The target is gone. Fill its channels with NaN to keep every sample one value per channel;
      // a NaN never trips a threshold.
      for (size_t c = 0; c < probe.numChannels; ++c) {
        outSample.values.push_back(std::numeric_limits<real>::quiet_NaN());
      }
      continue;
    }
    switch (probe.kind) {
      case ProbeKind::BodyPose: {
        mochi::TransformRT const pose = actor->GetRootTransform();
        Real3 const rotvec = pose.GetRotation().ToRotationVector();
        probe.unwrappedRotvec = UnwrapRotationVector(rotvec, probe.unwrappedRotvec);
        push3(pose.GetTranslation());
        push3(rotvec);
        push3(probe.unwrappedRotvec);
        break;
      }
      case ProbeKind::BodyVelocity:
        push3(actor->GetLinearVelocity(error));
        push3(actor->GetAngularVelocity(error));
        break;
      case ProbeKind::ContactForce:
        push3(actor->GetContactForceWorld(error));
        break;
      case ProbeKind::Penetration:
        outSample.values.push_back(PenetrationVolume(scene, *actor, probe.ignoreGround));
        break;
      case ProbeKind::JointState: {
        if (probe.dofIndices.empty()) {
          break; // a Hard joint; GetDofValues reads empty indices as all DoFs
        }
        actor->GetDofValues(
            mochi::MakeConstSpan(probe.dofIndices), mochi::MakeSpan(probe.dofValues), error);
        actor->GetArticulatedJointVelocities(mochi::MakeSpan(probe.scratch), error);
        for (size_t i = 0; i < probe.dofIndices.size(); ++i) {
          outSample.values.push_back(probe.dofValues[i]);
          outSample.values.push_back(probe.scratch[static_cast<size_t>(probe.dofIndices[i])]);
        }
        break;
      }
      case ProbeKind::RelativePose: {
        Actor const* frame = scene.GetActor(probe.frame);
        if (frame == nullptr) {
          for (size_t c = 0; c < probe.numChannels; ++c) {
            outSample.values.push_back(std::numeric_limits<real>::quiet_NaN());
          }
          break;
        }
        mochi::TransformRT const relative =
            Invert(frame->GetRootTransform()) * actor->GetRootTransform();
        push3(relative.GetTranslation());
        push3(relative.GetRotation().ToRotationVector());
        break;
      }
      case ProbeKind::SolverStats: {
        mochi::SolverStats const stats = scene.GetSolverStats();
        outSample.values.push_back(static_cast<real>(stats.maxNonLinearIters));
        outSample.values.push_back(static_cast<real>(stats.residualNorm));
        outSample.values.push_back(static_cast<real>(stats.maxLineSearchIters));
        outSample.values.push_back(static_cast<real>(stats.convergenceStatus));
        break;
      }
      case ProbeKind::Performance:
        // totalStepDurationSec is only set after the post-step callbacks, so it would lag a step.
        outSample.values.push_back(
            static_cast<real>(scene.GetPerformanceStats().solveStepDurationSec * 1000.0));
        break;
    }
  }
  outSample.loads = _appliedLoads;
  outSample.stopCondition = -1;
  outSample.stopMessage.clear();
  CheckStopConditions(scene, outSample);
}

} // namespace superdex::tuner
