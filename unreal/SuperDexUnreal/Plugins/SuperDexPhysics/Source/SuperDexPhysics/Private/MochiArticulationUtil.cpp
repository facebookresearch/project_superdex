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
  mochi::ErrorLog e{};
  const auto shapeInfo = ArticulatedActor->GetArticulatedShapeInfo(e);
  TArray<EMochiArticulatedJointType, TInlineAllocator<64>> types;
  bool bNeedsConversion = false;
  for (const mochi::ArticulatedJointType type : shapeInfo.jointTypes) {
    types.Add(static_cast<EMochiArticulatedJointType>(type));
    bNeedsConversion |= type == mochi::ArticulatedJointType::Spherical ||
        (type == mochi::ArticulatedJointType::Free && types.Num() > 1);
  }
  if (!bNeedsConversion) {
    return;
  }
  thread_local TArray<double> dofs;
  dofs.Reset(JointDofs.Num());
  for (const mochi::real dof : JointDofs) {
    dofs.Add(dof);
  }
  if (bToUnreal) {
    ConvertArticulatedDofsToUnreal(types, false, dofs, FMochiToUnrealConverter{});
  } else {
    ConvertArticulatedDofsToMochi(types, false, dofs, FUnrealToMochiConverter{});
  }
  for (int i = 0; i < dofs.Num(); ++i) {
    JointDofs[i] = static_cast<mochi::real>(dofs[i]);
  }
}
} // namespace

FMochiArticulationState ExtractArticulatedActorState(const mochi::Actor* ArticulatedActor) {
  mochi::ErrorLog e{};
  FMochiToUnrealConverter m2u;
  // static storage
  FMochiArticulationState state;
  thread_local std::vector<mochi::TransformRT> worldFromLinks;
  thread_local std::vector<mochi::real> jointPositions;
  thread_local std::vector<mochi::real> jointVelocities;
  // count DOFs
  const auto shapeInfo = ArticulatedActor->GetArticulatedShapeInfo(e);
  const int numMochiDofs = ArticulatedActor->GetNumDofs();
  const int numMochiBaseDofs = GetNumBaseDofs(shapeInfo);
  const int numMochiJointDofs = numMochiDofs - numMochiBaseDofs;
  // get base transform
  const auto linkActors = ArticulatedActor->GetNestedLinkActors(e);
  const int numLinks = linkActors.size();
  worldFromLinks.resize(numLinks);
  ArticulatedActor->GetArticulatedLinkTransforms(worldFromLinks, e);
  state.BaseTransform = m2u.TransformToUnreal(worldFromLinks[0]);
  // get joint positions
  jointPositions.resize(numMochiDofs);
  ArticulatedActor->GetArticulatedPose(jointPositions, e);
  state.JointPositions.SetNum(numMochiJointDofs);
  for (int i = numMochiBaseDofs; i < numMochiDofs; ++i) {
    state.JointPositions[i - numMochiBaseDofs] = jointPositions[i];
  }
  // get joint velocities
  jointVelocities.resize(numMochiDofs);
  ArticulatedActor->GetArticulatedJointVelocities(jointVelocities, e);
  state.JointVelocities.SetNum(numMochiJointDofs);
  for (int i = numMochiBaseDofs; i < numMochiDofs; ++i) {
    state.JointVelocities[i - numMochiBaseDofs] = jointVelocities[i];
  }
  return state;
}

void SetArticulatedActorJointEfforts(
    mochi::Actor* ArticulatedActor,
    const TArray<mochi::real>& JointEfforts) {
  mochi::ErrorLog e{};
  // static storage
  thread_local std::vector<int> jointDofIndices;
  thread_local std::vector<mochi::real> jointEfforts;
  // count DOFs
  const auto shapeInfo = ArticulatedActor->GetArticulatedShapeInfo(e);
  const int numMochiDofs = ArticulatedActor->GetNumDofs();
  const int numMochiBaseDofs = GetNumBaseDofs(shapeInfo);
  const int numMochiJointDofs = numMochiDofs - numMochiBaseDofs;
  check(JointEfforts.Num() >= numMochiJointDofs);
  // set efforts
  jointDofIndices.resize(numMochiJointDofs);
  std::iota(jointDofIndices.begin(), jointDofIndices.end(), numMochiBaseDofs);
  jointEfforts.resize(numMochiJointDofs);
  for (int i = 0; i < numMochiJointDofs; ++i) {
    jointEfforts[i] = JointEfforts[i];
  }
  ArticulatedActor->SetExternalForcesOnDofs(jointDofIndices, jointEfforts, e);
}

TArray<mochi::real> BuildPoseControllerTargets(
    const mochi::Actor* ArticulatedActor,
    const FTransform& BaseTransformTarget,
    const TArray<mochi::real>& JointPositionTargets) {
  mochi::ErrorLog e{};
  FMochiToUnrealConverter m2u;
  // static storage
  thread_local TArray<mochi::real> targets;
  // count DOFs
  const auto shapeInfo = ArticulatedActor->GetArticulatedShapeInfo(e);
  const int numMochiDofs = ArticulatedActor->GetNumDofs();
  const int numMochiBaseDofs = GetNumBaseDofs(shapeInfo);
  const int numMochiJointDofs = numMochiDofs - numMochiBaseDofs;
  check(JointPositionTargets.Num() >= numMochiJointDofs);
  // build pose target
  targets.SetNum(numMochiJointDofs);
  for (int i = 0; i < numMochiJointDofs; ++i) {
    targets[i] = JointPositionTargets[i];
  }
  if (numMochiBaseDofs > 0) {
    FUnrealToMochiConverter u2m;
    const mochi::TransformRT baseToWorld = u2m.TransformToMochi_RT(BaseTransformTarget);
    const mochi::TransformRT baseTarget = mochi::Invert(shapeInfo.parentLinkFromJoint[0]) *
        baseToWorld * mochi::Invert(shapeInfo.jointFromChildLink[0]);
    const mochi::Real3 pos = baseTarget.GetTranslation();
    const mochi::Real3 rot = baseTarget.GetRotation().ToRotationVector();
    targets.Insert({pos[0], pos[1], pos[2], rot[0], rot[1], rot[2]}, 0);
  }
  return targets;
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
