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

#include "MochiArticulationUtil.h"

#include "MochiArticulationTypes.h"
#include "MochiConversions.h"

#include <numeric>

namespace {
// The DOFs of a Free root joint belong to the base.
int GetNumBaseDofs(const mochi::ArticulatedShapeInfo& ShapeInfo) {
  return ShapeInfo.jointTypes[0] == mochi::ArticulatedJointType::Free
      ? GetArticulatedJointNumDofs(EMochiArticulatedJointType::Free)
      : 0;
}

// Converts the joint DOFs of an articulated actor between the Mochi and Unreal conventions.
void ConvertArticulatedActorDofs(
    const mochi::Actor* ArticulatedActor,
    TArray<mochi::real>& JointDofs,
    bool bToUnreal) {
  mochi::ErrorLog E{};
  const mochi::ArticulatedShapeInfo ShapeInfo = ArticulatedActor->GetArticulatedShapeInfo(E);
  TArray<EMochiArticulatedJointType, TInlineAllocator<64>> Types;
  bool bNeedsConversion = false;
  for (const mochi::ArticulatedJointType Type : ShapeInfo.jointTypes) {
    Types.Add(::ToUnreal(Type));
    bNeedsConversion |= Type == mochi::ArticulatedJointType::Spherical ||
        (Type == mochi::ArticulatedJointType::Free && Types.Num() > 1);
  }
  if (!bNeedsConversion) {
    return;
  }
  thread_local TArray<double> Dofs;
  Dofs.Reset(JointDofs.Num());
  for (const mochi::real Dof : JointDofs) {
    Dofs.Add(Dof);
  }
  if (bToUnreal) {
    ConvertArticulatedDofsToUnreal(Types, false, Dofs, FMochiToUnrealConverter{});
  } else {
    ConvertArticulatedDofsToMochi(Types, false, Dofs, FUnrealToMochiConverter{});
  }
  for (int i = 0; i < Dofs.Num(); ++i) {
    JointDofs[i] = static_cast<mochi::real>(Dofs[i]);
  }
}
} // namespace

FMochiArticulationState ExtractArticulatedActorState(const mochi::Actor* ArticulatedActor) {
  mochi::ErrorLog E{};
  FMochiToUnrealConverter M2u;
  // static storage
  FMochiArticulationState State;
  thread_local std::vector<mochi::TransformRT> WorldFromLinks;
  thread_local std::vector<mochi::real> JointPositions;
  thread_local std::vector<mochi::real> JointVelocities;
  // count DOFs
  const mochi::ArticulatedShapeInfo ShapeInfo = ArticulatedActor->GetArticulatedShapeInfo(E);
  const int NumMochiDofs = ArticulatedActor->GetNumDofs();
  const int NumMochiBaseDofs = GetNumBaseDofs(ShapeInfo);
  const int NumMochiJointDofs = NumMochiDofs - NumMochiBaseDofs;
  // get base transform
  const mochi::Span<mochi::ActorHandle const> LinkActors = ArticulatedActor->GetNestedLinkActors(E);
  const int NumLinks = LinkActors.size();
  WorldFromLinks.resize(NumLinks);
  ArticulatedActor->GetArticulatedLinkTransforms(WorldFromLinks, E);
  State.BaseTransform = M2u.TransformToUnreal(WorldFromLinks[0]);
  // get joint positions
  JointPositions.resize(NumMochiDofs);
  ArticulatedActor->GetArticulatedPose(JointPositions, E);
  State.JointPositions.SetNum(NumMochiJointDofs);
  for (int i = NumMochiBaseDofs; i < NumMochiDofs; ++i) {
    State.JointPositions[i - NumMochiBaseDofs] = JointPositions[i];
  }
  // get joint velocities
  JointVelocities.resize(NumMochiDofs);
  ArticulatedActor->GetArticulatedJointVelocities(JointVelocities, E);
  State.JointVelocities.SetNum(NumMochiJointDofs);
  for (int i = NumMochiBaseDofs; i < NumMochiDofs; ++i) {
    State.JointVelocities[i - NumMochiBaseDofs] = JointVelocities[i];
  }
  return State;
}

void SetArticulatedActorJointEfforts(
    mochi::Actor* ArticulatedActor,
    const TArray<mochi::real>& JointEfforts) {
  mochi::ErrorLog E{};
  // static storage
  thread_local std::vector<int> JointDofIndices;
  thread_local std::vector<mochi::real> NativeJointEfforts;
  // count DOFs
  const mochi::ArticulatedShapeInfo ShapeInfo = ArticulatedActor->GetArticulatedShapeInfo(E);
  const int NumMochiDofs = ArticulatedActor->GetNumDofs();
  const int NumMochiBaseDofs = GetNumBaseDofs(ShapeInfo);
  const int NumMochiJointDofs = NumMochiDofs - NumMochiBaseDofs;
  check(JointEfforts.Num() >= NumMochiJointDofs);
  // set efforts
  JointDofIndices.resize(NumMochiJointDofs);
  std::iota(JointDofIndices.begin(), JointDofIndices.end(), NumMochiBaseDofs);
  NativeJointEfforts.resize(NumMochiJointDofs);
  for (int i = 0; i < NumMochiJointDofs; ++i) {
    NativeJointEfforts[i] = JointEfforts[i];
  }
  ArticulatedActor->SetExternalForcesOnDofs(JointDofIndices, NativeJointEfforts, E);
}

void TeleportArticulatedActor(
    mochi::Actor* ArticulatedActor,
    mochi::Span<const mochi::TransformRT> WorldFromLinks,
    mochi::Error& Error) {
  ArticulatedActor->SetArticulatedPoseFromLinks(WorldFromLinks, Error);
  MOCHI_ERROR_RETURN(Error);
  thread_local std::vector<mochi::real> zeroVelocities;
  zeroVelocities.assign(ArticulatedActor->GetNumDofs(), 0);
  ArticulatedActor->SetArticulatedJointVelocities(zeroVelocities, Error);
}

TArray<mochi::real> BuildPoseControllerTargets(
    const mochi::Actor* ArticulatedActor,
    const FTransform& BaseTransformTarget,
    const TArray<mochi::real>& JointPositionTargets) {
  mochi::ErrorLog E{};
  FMochiToUnrealConverter M2u;
  // static storage
  thread_local TArray<mochi::real> Targets;
  // count DOFs
  const mochi::ArticulatedShapeInfo ShapeInfo = ArticulatedActor->GetArticulatedShapeInfo(E);
  const int NumMochiDofs = ArticulatedActor->GetNumDofs();
  const int NumMochiBaseDofs = GetNumBaseDofs(ShapeInfo);
  const int NumMochiJointDofs = NumMochiDofs - NumMochiBaseDofs;
  check(JointPositionTargets.Num() >= NumMochiJointDofs);
  // build pose target
  Targets.SetNum(NumMochiJointDofs);
  for (int i = 0; i < NumMochiJointDofs; ++i) {
    Targets[i] = JointPositionTargets[i];
  }
  if (NumMochiBaseDofs > 0) {
    FUnrealToMochiConverter U2m;
    const mochi::TransformRT BaseToWorld = U2m.TransformToMochi_RT(BaseTransformTarget);
    const mochi::TransformRT BaseTarget = mochi::Invert(ShapeInfo.parentLinkFromJoint[0]) *
        BaseToWorld * mochi::Invert(ShapeInfo.jointFromChildLink[0]);
    const mochi::Real3 Pos = BaseTarget.GetTranslation();
    const mochi::Real3 Rot = BaseTarget.GetRotation().ToRotationVector();
    Targets.Insert({Pos[0], Pos[1], Pos[2], Rot[0], Rot[1], Rot[2]}, 0);
  }
  return Targets;
}

void ConvertArticulatedActorDofsToUnreal(
    const mochi::Actor* ArticulatedActor,
    TArray<mochi::real>& JointDofs) {
  ConvertArticulatedActorDofs(ArticulatedActor, JointDofs, true);
}

void ConvertArticulatedActorDofsToMochi(
    const mochi::Actor* ArticulatedActor,
    TArray<mochi::real>& JointDofs) {
  ConvertArticulatedActorDofs(ArticulatedActor, JointDofs, false);
}
