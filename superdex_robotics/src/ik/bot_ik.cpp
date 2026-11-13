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

#include <superdex_robotics/core/loader.h>
#include <superdex_robotics/ik/bot_ik.h>
#include <superdex_robotics/ik/bot_ik_config.h>
#include <superdex_robotics/utils/bot_utils.h>

#include <mochi_core/utils/math_utils.h>

#include <utility>

using namespace mochi;
using namespace superdex::robotics;

namespace {

void ValidatePreparedPrefab(BotPrefab const& prefab, Error& error) {
  Validate(prefab, nullptr, error);
  MOCHI_ERROR_RETURN(error);
  int const numLinks = isize(prefab.links);
  DynamicArray<DynamicArray<int>> children(numLinks);
  DynamicArray<TransformRT> rootFromLinks(numLinks);
  DynamicArray<int> dofIndices;
  for (int i = 0; i < numLinks; ++i) {
    auto const& link = prefab.links[i];
    MOCHI_ERROR_IF(link._index != i, error, "Prefab link indices have not been prepared.");
    TransformRT const parentFromLink =
        i == 0 ? TransformRT{} : prefab.joints[i].parentLinkFromJoint;
    if (i > 0) {
      MOCHI_ERROR_IF_NOT(
          link.parentLink >= 0 && link.parentLink < i,
          error,
          "Prepared prefab must keep parents before children.");
      MOCHI_ERROR_RETURN(error);
      children[link.parentLink].push_back(i);
      rootFromLinks[i] = rootFromLinks[link.parentLink] * parentFromLink;
    }
    MOCHI_ERROR_IF_NOT(
        link._parentFromLink == parentFromLink && link._rootFromLink == rootFromLinks[i],
        error,
        "Prefab link transforms have not been prepared.");
    auto const type = prefab.joints[i].type;
    int const numDofs = type == ArticulatedJointType::Spherical
        ? 3
        : (type == ArticulatedJointType::Revolute || type == ArticulatedJointType::Prismatic ? 1
                                                                                             : 0);
    for (int j = 0; j < numDofs; ++j) {
      dofIndices.push_back(i);
    }
  }
  for (int i = 0; i < numLinks; ++i) {
    MOCHI_ERROR_IF_NOT(
        prefab.links[i]._childrenIndices == children[i],
        error,
        "Prefab children have not been prepared.");
  }
  MOCHI_ERROR_IF_NOT(
      prefab._dofIndices == dofIndices && prefab._numDofs == isize(dofIndices) &&
          isize(prefab.defaultPose) == prefab._numDofs,
      error,
      "Prefab DOF data and default pose are inconsistent.");
  MOCHI_ERROR_IF_NOT(IsFinite(prefab.worldFromRoot), error, "Non-finite world placement.");
  for (real const position : prefab.defaultPose) {
    MOCHI_ERROR_IF_NOT(IsFinite(position), error, "Non-finite default joint position.");
  }
}

int ResolveCouplingJoint(BotPrefab const& prefab, DynamicString const& name, Error& error) {
  int index = kIndexNone;
  for (int i = 0; i < isize(prefab.joints); ++i) {
    if (prefab.joints[i].name == std::string_view(name)) {
      MOCHI_ERROR_IF(index != kIndexNone, error, "Coupling joint name is ambiguous.");
      MOCHI_ERROR_RETURN(error, kIndexNone);
      index = i;
    }
  }
  MOCHI_ERROR_IF_NOT(index > 0, error, "Coupling joint must exist and must not be the root.");
  MOCHI_ERROR_RETURN(error, kIndexNone);
  auto const type = prefab.joints[index].type;
  MOCHI_ERROR_IF_NOT(
      type == ArticulatedJointType::Revolute || type == ArticulatedJointType::Prismatic,
      error,
      "Coupling joint must be single-DOF revolute or prismatic.");
  MOCHI_ERROR_RETURN(error, kIndexNone);
  return index;
}

DynamicArray<experimental::LinearTransmissionParams> ResolveJointCouplings(
    BotPrefab const& prefab,
    Span<BotIKJointCouplingParams const> couplings,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  DynamicArray<experimental::LinearTransmissionParams> resolved(couplings.size());
  for (int i = 0; i < isize(couplings); ++i) {
    auto const& coupling = couplings[i];
    MOCHI_ERROR_IF_NOT(
        !coupling.jointA.empty() && !coupling.jointB.empty() &&
            coupling.jointA != std::string_view(coupling.jointB),
        error,
        "Coupling joint names must be nonempty and distinct.");
    MOCHI_ERROR_IF_NOT(
        IsFinite(coupling.ratio) && coupling.ratio != 0_r && IsFinite(coupling.offset) &&
            IsFinite(coupling.stiffness) && coupling.stiffness >= 0_r,
        error,
        "Coupling requires finite nonzero ratio, finite offset and nonnegative stiffness.");
    MOCHI_ERROR_RETURN(error, {});
    if (!coupling.enabled || coupling.stiffness == 0_r) {
      continue;
    }
    int const jointA = ResolveCouplingJoint(prefab, coupling.jointA, error);
    int const jointB = ResolveCouplingJoint(prefab, coupling.jointB, error);
    MOCHI_ERROR_RETURN(error, {});
    resolved[i].jointIndices = {jointA, jointB};
    resolved[i].jointCoefficients = {coupling.ratio, -1_r};
  }
  return resolved;
}

Quaternion NormalizeRotation(Quaternion const& rotation, Error& error) {
  Real4 const components = rotation.ToReal4();
  MOCHI_ERROR_IF_NOT(IsFinite(components), error, "Rotation must be finite.");
  MOCHI_ERROR_RETURN(error, {});
  real scale = 0_r;
  for (real const component : components) {
    scale = Max(scale, Abs(component));
  }
  MOCHI_ERROR_IF_NOT(scale > 0_r, error, "Rotation must be nonzero.");
  MOCHI_ERROR_RETURN(error, {});
  Real4 const scaled = components / scale;
  return Quaternion{scaled / Norm(scaled)};
}

} // namespace

struct BotIK::Impl {
  Impl() = default;
  Impl(Impl const&) = delete;
  Impl& operator=(Impl const&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  Context* context = nullptr;
  RoboticsContext* roboticsContext = nullptr;
  Scene* scene = nullptr;
  Bot* bot = nullptr;
  Actor* actor = nullptr;
  experimental::IKSolver* solver = nullptr;
  int baseDofs = 0;
  DynamicArray<ActorHandle> links;
  DynamicArray<DynamicString> linkNames;
  TransformRT worldFrameFromOutputBaseFrame;
  DynamicArray<bool> rigidRootLinks;
  DynamicArray<real> actorPose;
  DynamicArray<real> jointPositions;
  DynamicArray<TransformRT> worldFromLinks;

  struct PositionGoal {
    Constraint* constraint = nullptr;
    Real3 localPosition;
  };
  struct RotationGoal {
    Constraint* constraint = nullptr;
    Quaternion localRotation;
  };
  DynamicArray<PositionGoal> positionGoals;
  DynamicArray<RotationGoal> rotationGoals;

  ~Impl() {
    if (solver != nullptr) {
      ClearAllGoals(ErrorAssert{});
    }
    if (bot != nullptr) {
      roboticsContext->DestroyBot(scene, bot->GetHandle());
    }
    if (solver != nullptr) {
      experimental::DestroyIKSolver(solver, context, ErrorAssert{});
    } else if (scene != nullptr) {
      context->DestroyScene(scene);
    }
    DestroyRoboticsContext(roboticsContext);
  }

  void CheckLinkIndex(int linkIndex, Error& error) const {
    MOCHI_ERROR_IF_NOT(
        linkIndex >= 0 && linkIndex < isize(links), error, "Link index is out of range.");
  }

  void ClearAllGoals(Error& error) {
    MOCHI_ERROR_RETURN(error);
    for (int i = 0; i < isize(links); ++i) {
      if (positionGoals[i].constraint != nullptr) {
        solver->ClearPositionTarget(links[i], error);
        positionGoals[i].constraint = nullptr;
      }
      if (rotationGoals[i].constraint != nullptr) {
        solver->ClearRotationTarget(links[i], error);
        rotationGoals[i].constraint = nullptr;
      }
    }
  }

  void RefreshSnapshots(Error& error) {
    MOCHI_ERROR_RETURN(error);
    actor->GetArticulatedPose(actorPose, error);
    actor->GetArticulatedLinkTransforms(worldFromLinks, error);
    MOCHI_ERROR_RETURN(error);
    for (int i = 0; i < isize(jointPositions); ++i) {
      jointPositions[i] = actorPose[i + baseDofs];
    }
  }
};

BotIK::BotIK(std::unique_ptr<Impl> impl) : _impl(std::move(impl)) {}
BotIK::~BotIK() = default;

void BotIK::Destroy(BotIK* ik) {
  delete ik;
}

void BotIK::ValidateJointCouplings(
    BotPrefab const& prefab,
    Span<BotIKJointCouplingParams const> couplings,
    Error& error) {
  (void)ResolveJointCouplings(prefab, couplings, error);
}

BotIK* BotIK::Create(
    Context* context,
    BotPrefab const& prefab,
    BotIKCreateParams const& params,
    IBotLoader const& loader,
    Error& error) {
  return Create(context, prefab, params, {}, loader, error);
}

BotIK* BotIK::Create(
    Context* context,
    BotPrefab const& prefab,
    BotIKCreateParams const& params,
    Span<BotIKJointCouplingParams const> couplings,
    IBotLoader const& loader,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(context == nullptr, error, "Mochi context is null.");
  ValidatePreparedPrefab(prefab, error);
  MOCHI_ERROR_RETURN(error, {});
  auto const resolved = ResolveJointCouplings(prefab, couplings, error);
  MOCHI_ERROR_RETURN(error, {});

  BotPrefab privatePrefab = prefab;
  if (params.fixedBase) {
    privatePrefab.joints[0].type = ArticulatedJointType::Hard;
  }
  for (auto& link : privatePrefab.links) {
    link.sensors.clear();
    link.actuators.clear();
    if (params.excludeCollisionShapes) {
      link.shapeFile.clear();
      link.colliderType = ColliderType::None;
    }
  }
  if (params.excludeCollisionShapes) {
    privatePrefab.skin.reset();
    privatePrefab._skinBoneLinks.clear();
  }
  // Contact is never part of this kinematic solve, including authored pair overrides.
  privatePrefab.contactOverrides.clear();

  auto impl = std::make_unique<Impl>();
  impl->context = context;
  impl->roboticsContext = CreateRoboticsContext();
  impl->scene = context->CreateScene("BotIK");
  MOCHI_ERROR_IF(impl->scene == nullptr, error, "Failed to create IK scene.");
  MOCHI_ERROR_RETURN(error, {});
  impl->bot = impl->roboticsContext->CreateBot(impl->scene, privatePrefab, loader, error);
  MOCHI_ERROR_RETURN(error, {});
  impl->actor = impl->bot->GetArticulatedActor();
  for (int i = 0; i < isize(resolved); ++i) {
    if (resolved[i].jointIndices.empty()) {
      continue;
    }
    int const index = experimental::AddLinearTransmission(impl->actor, resolved[i], error);
    MOCHI_ERROR_RETURN(error, {});
    experimental::DisplacementControlActuatorParams actuator;
    actuator.targetDisplacement = -couplings[i].offset;
    actuator.stiffness = couplings[i].stiffness;
    actuator.damping = 0_r;
    actuator.allowCompressiveForce = true;
    experimental::AttachDisplacementControlActuator(impl->actor, index, actuator, error);
    MOCHI_ERROR_RETURN(error, {});
  }
  impl->links = impl->actor->GetNestedLinkActors(error);
  impl->worldFrameFromOutputBaseFrame = prefab.worldFromRoot;
  for (auto const& link : prefab.links) {
    impl->linkNames.push_back(link.name);
  }
  auto const& shapeInfo = impl->actor->GetArticulatedShapeInfo(error);
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF_NOT(
      impl->links.size() == privatePrefab.links.size() && !shapeInfo.dofInfo.empty(),
      error,
      "Built actor layout does not match the prefab.");
  MOCHI_ERROR_RETURN(error, {});
  impl->baseDofs = shapeInfo.dofInfo[0].GetSize();
  int const actorDofs = impl->actor->GetNumDofs();
  MOCHI_ERROR_IF_NOT(
      actorDofs - impl->baseDofs == prefab._numDofs,
      error,
      "Built actor DOF layout is inconsistent.");
  MOCHI_ERROR_RETURN(error, {});
  impl->actorPose.resize(actorDofs);
  impl->jointPositions.resize(prefab._numDofs);
  impl->worldFromLinks.resize(prefab.links.size());
  impl->rigidRootLinks.resize(prefab.links.size());
  impl->positionGoals.resize(prefab.links.size());
  impl->rotationGoals.resize(prefab.links.size());
  impl->rigidRootLinks[0] = true;
  for (int i = 1; i < isize(prefab.links); ++i) {
    impl->rigidRootLinks[i] = impl->rigidRootLinks[prefab.links[i].parentLink] &&
        prefab.joints[i].type == ArticulatedJointType::Hard;
  }
  impl->scene->EnableActorContactSymmetric(
      impl->actor->GetHandle(), impl->actor->GetHandle(), false, IncludeNestedActors::Yes, error);
  MOCHI_ERROR_RETURN(error, {});
  impl->solver = experimental::CreateIKSolver(impl->scene, context, error);
  MOCHI_ERROR_RETURN(error, {});
  impl->solver->SetSolverParams(params.solverParams);
  impl->RefreshSnapshots(error);
  MOCHI_ERROR_RETURN(error, {});
  return new BotIK(std::move(impl));
}

BotIK* BotIK::Create(
    Context* context,
    BotPrefab const& prefab,
    BotIKCreateParams const& params,
    Error& error) {
  return Create(context, prefab, params, {}, FileBotLoader{}, error);
}

BotIK* BotIK::Create(
    Context* context,
    BotPrefab const& prefab,
    BotIKCreateParams const& params,
    Span<BotIKJointCouplingParams const> couplings,
    Error& error) {
  return Create(context, prefab, params, couplings, FileBotLoader{}, error);
}

void BotIK::SetSolverParams(experimental::IKSolverParams const& params) {
  _impl->solver->SetSolverParams(params);
}

experimental::IKSolverParams BotIK::GetSolverParams() const {
  return _impl->solver->GetSolverParams();
}

void BotIK::SetPositionGoal(
    int linkIndex,
    Real3 const& localPosition,
    Real3 const& worldPosition,
    real stiffness,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  _impl->CheckLinkIndex(linkIndex, error);
  MOCHI_ERROR_IF_NOT(
      IsFinite(localPosition) && IsFinite(worldPosition), error, "Position goal must be finite.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(stiffness) && stiffness >= 0_r, error, "Stiffness must be finite and nonnegative.");
  MOCHI_ERROR_RETURN(error);
  auto& goal = _impl->positionGoals[linkIndex];
  if (goal.constraint != nullptr && goal.localPosition == localPosition) {
    goal.constraint->SetTargetPosition(worldPosition, error);
    goal.constraint->SetStiffness(stiffness, error);
  } else {
    auto* constraint = _impl->solver->CreatePositionTarget(
        _impl->links[linkIndex], localPosition, worldPosition, stiffness, error);
    MOCHI_ERROR_RETURN(error);
    goal.constraint = constraint;
    goal.localPosition = localPosition;
  }
}

void BotIK::SetRotationGoal(
    int linkIndex,
    Quaternion const& localRotation,
    Quaternion const& worldRotation,
    real stiffness,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  _impl->CheckLinkIndex(linkIndex, error);
  MOCHI_ERROR_IF_NOT(
      IsFinite(stiffness) && stiffness >= 0_r, error, "Stiffness must be finite and nonnegative.");
  Quaternion const local = NormalizeRotation(localRotation, error);
  Quaternion const world = NormalizeRotation(worldRotation, error);
  MOCHI_ERROR_RETURN(error);
  auto& goal = _impl->rotationGoals[linkIndex];
  if (goal.constraint != nullptr && goal.localRotation == local) {
    goal.constraint->SetTargetRotation(world, error);
    goal.constraint->SetStiffness(stiffness, error);
  } else {
    auto* constraint = _impl->solver->CreateRotationTarget(
        _impl->links[linkIndex],
        local.ToRotationVector(),
        world.ToRotationVector(),
        stiffness,
        error);
    MOCHI_ERROR_RETURN(error);
    goal.constraint = constraint;
    goal.localRotation = local;
  }
}

void BotIK::ClearPositionGoal(int linkIndex, Error& error) {
  MOCHI_ERROR_RETURN(error);
  _impl->CheckLinkIndex(linkIndex, error);
  MOCHI_ERROR_RETURN(error);
  _impl->solver->ClearPositionTarget(_impl->links[linkIndex], error);
  MOCHI_ERROR_RETURN(error);
  _impl->positionGoals[linkIndex].constraint = nullptr;
}

void BotIK::ClearRotationGoal(int linkIndex, Error& error) {
  MOCHI_ERROR_RETURN(error);
  _impl->CheckLinkIndex(linkIndex, error);
  MOCHI_ERROR_RETURN(error);
  _impl->solver->ClearRotationTarget(_impl->links[linkIndex], error);
  MOCHI_ERROR_RETURN(error);
  _impl->rotationGoals[linkIndex].constraint = nullptr;
}

void BotIK::ClearAllGoals(Error& error) {
  _impl->ClearAllGoals(error);
}

void BotIK::SetInputBotConfiguration(
    BotPrefab const& inputBot,
    Span<real const> configuration,
    TransformRT const& worldFrameFromInputBaseFrame,
    Span<BotIKGoalConfig const> goals,
    Error& error) {
  SetInputBotConfiguration(
      inputBot, configuration, worldFrameFromInputBaseFrame, 1_r, goals, error);
}

void BotIK::SetInputBotConfiguration(
    BotPrefab const& inputBot,
    Span<real const> configuration,
    TransformRT const& worldFrameFromInputBaseFrame,
    real inputBotScale,
    Span<BotIKGoalConfig const> goals,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  DynamicArray<TransformRT> inputLinks;
  ComputeIKInputBotTransforms(
      inputBot, configuration, worldFrameFromInputBaseFrame, inputBotScale, inputLinks, error);
  MOCHI_ERROR_RETURN(error);
  DynamicArray<BotIKGoalConfig const*> bindings(_impl->links.size());
  DynamicArray<TransformRT> goalTransforms(_impl->links.size());
  for (auto const& goal : goals) {
    int index = kIndexNone;
    for (int i = 0; i < isize(_impl->linkNames); ++i) {
      if (_impl->linkNames[i] == std::string_view(goal.outputBotLink)) {
        index = i;
        break;
      }
    }
    MOCHI_ERROR_IF_NOT(index != kIndexNone, error, "Unknown IK bot link.");
    MOCHI_ERROR_RETURN(error);
    bindings[index] = &goal;
    ComputeIKGoalWorldConstraintTransform(
        goal,
        _impl->worldFrameFromOutputBaseFrame,
        &inputBot,
        MakeConstSpan(inputLinks),
        inputBotScale,
        goalTransforms[index],
        error);
    MOCHI_ERROR_RETURN(error);
  }
  for (int i = 0; i < isize(bindings); ++i) {
    auto const* goal = bindings[i];
    if (goal && goal->enabled && goal->positionStiffness > 0_r) {
      SetPositionGoal(
          i,
          goal->linkFrameFromConstraintFrame.GetTranslation(),
          goalTransforms[i].GetTranslation(),
          goal->positionStiffness,
          error);
    } else if (_impl->positionGoals[i].constraint) {
      ClearPositionGoal(i, error);
    }
    if (goal && goal->enabled && goal->rotationStiffness > 0_r) {
      SetRotationGoal(
          i,
          goal->linkFrameFromConstraintFrame.GetRotation(),
          goalTransforms[i].GetRotation(),
          goal->rotationStiffness,
          error);
    } else if (_impl->rotationGoals[i].constraint) {
      ClearRotationGoal(i, error);
    }
    MOCHI_ERROR_RETURN(error);
  }
}

bool BotIK::SolveIK(Error& error) {
  MOCHI_ERROR_RETURN(error, false);
  bool const reached = _impl->solver->SolveIK(error);
  _impl->RefreshSnapshots(error);
  return error.IsOK() && reached;
}

SolverStats BotIK::GetLastSolverStats() const {
  return _impl->scene->GetSolverStats();
}

Span<real const> BotIK::GetJointPositions() const {
  return _impl->jointPositions;
}

Span<TransformRT const> BotIK::GetWorldFromLinks() const {
  return _impl->worldFromLinks;
}

void BotIK::SetJointPositions(Span<real const> positions, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      positions.size() == _impl->jointPositions.size(),
      error,
      "Joint position count is incorrect.");
  for (real const position : positions) {
    MOCHI_ERROR_IF_NOT(IsFinite(position), error, "Joint positions must be finite.");
  }
  MOCHI_ERROR_RETURN(error);
  // Read the actual root DOFs; zero-filling them here would restore the creation placement.
  _impl->actor->GetArticulatedPose(_impl->actorPose, error);
  MOCHI_ERROR_RETURN(error);
  for (int i = 0; i < isize(positions); ++i) {
    _impl->actorPose[i + _impl->baseDofs] = positions[i];
  }
  _impl->actor->SetArticulatedPoseFromJoints(_impl->actorPose, error);
  _impl->RefreshSnapshots(error);
}

int BotIK::GetNumLinks() const {
  return isize(_impl->links);
}

int BotIK::GetNumDofs() const {
  return isize(_impl->jointPositions);
}

bool BotIK::IsLinkRigidlyAttachedToRootLink(int linkIndex) const {
  return linkIndex >= 0 && linkIndex < GetNumLinks() && _impl->rigidRootLinks[linkIndex];
}
