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

#include "MochiArticulationJointConstraintComponent.h"
#include "MochiArticulation.h"
#include "MochiArticulationComponent.h"
#include "MochiConversions.h"

void UMochiArticulationJointConstraintComponent::SetTargetValue(double NewTargetValue) {
  TargetValue = NewTargetValue;
  const mochi::real target = MochiDofSign * NewTargetValue;
  ForEachConstraint([target](mochi::Constraint& C) { C.SetTargetDof(target, mochi::ErrorLog{}); });
}

void UMochiArticulationJointConstraintComponent::SetTargetRotation(FRotator NewTargetRotation) {
  TargetRotation = NewTargetRotation;
  const mochi::Quaternion target =
      FUnrealToMochiConverter().RotationToMochi(NewTargetRotation.Quaternion());
  ForEachConstraint(
      [target](mochi::Constraint& C) { C.SetTargetRotation(target, mochi::ErrorLog{}); });
}

double UMochiArticulationJointConstraintComponent::GetSaturationScale() const {
  return 1.0;
}

EMochiBodyState UMochiArticulationJointConstraintComponent::BuildConstraints(
    FConstraintRequest& OutRequest) {
  FMochiConstraintBody body;
  mochi::ActorHandle actor;
  const EMochiBodyState state = ResolveBody(Articulation, body, OutRequest.Scene, actor);
  if (state != EMochiBodyState::Ready) {
    return state;
  }
  const UMochiArticulation* asset =
      body.Articulation ? body.Articulation->GetArticulationAsset() : nullptr;
  if (asset == nullptr || !body.Link.IsNone()) {
    LogError(FString::Printf(TEXT("\"%s\" is not an articulation."), *Articulation));
    return EMochiBodyState::Failed;
  }
  const int jointIndex = asset->GetJointIndex(Joint);
  if (jointIndex < 0) {
    LogError(
        FString::Printf(
            TEXT("articulation %s has no joint %s."), *Articulation, *Joint.ToString()));
    return EMochiBodyState::Failed;
  }
  const EMochiArticulatedJointType jointType = asset->GetJoints()[jointIndex].Type;
  const FUnrealToMochiConverter u2m;

  switch (Type) {
    case EMochiArticulationJointConstraintType::Target:
    case EMochiArticulationJointConstraintType::Range: {
      // Mochi constrains single DOFs of translations and of 1-DOF rotations only.
      const bool bRotationVector = jointType == EMochiArticulatedJointType::Spherical ||
          (jointType == EMochiArticulatedJointType::Free && Dof >= 3);
      int32 dof = 0;
      if (bRotationVector ||
          !ConvertArticulatedDofIndexToMochi(jointType, Dof, u2m, dof, MochiDofSign)) {
        LogError(
            FString::Printf(
                TEXT(
                    "joint %s has no translation or 1-DOF rotation DOF %d (use Joint Rotation Target "
                    "or Range for rotations of spherical and free joints)."),
                *Joint.ToString(),
                Dof));
        return EMochiBodyState::Failed;
      }
      if (Type == EMochiArticulationJointConstraintType::Target) {
        mochi::ArticulatedSingleDofTargetConstraintParams params;
        FillCommonParams(params);
        params.actor = actor;
        params.jointIndex = jointIndex;
        params.dofIndex = dof;
        params.targetValue = MochiDofSign * TargetValue;
        OutRequest.CreateOne(params, &mochi::Scene::CreateArticulatedSingleDofTargetConstraint);
      } else {
        mochi::ArticulatedSingleDofRangeConstraintParams params;
        FillCommonParams(params);
        params.actor = actor;
        params.jointIndex = jointIndex;
        params.dofIndex = dof;
        params.minValue = FMath::Min(MochiDofSign * MinValue, MochiDofSign * MaxValue);
        params.maxValue = FMath::Max(MochiDofSign * MinValue, MochiDofSign * MaxValue);
        OutRequest.CreateOne(params, &mochi::Scene::CreateArticulatedSingleDofRangeConstraint);
      }
      break;
    }
    case EMochiArticulationJointConstraintType::RotationTarget: {
      mochi::Articulated3dRotationTargetConstraintParams params;
      FillCommonParams(params);
      params.actor = actor;
      params.jointIndex = jointIndex;
      params.target = u2m.RotationToMochi(TargetRotation.Quaternion());
      OutRequest.CreateOne(params, &mochi::Scene::CreateArticulated3dRotationTargetConstraint);
      break;
    }
    case EMochiArticulationJointConstraintType::RotationRange: {
      mochi::Articulated3dRotationRangeConstraintParams params;
      FillCommonParams(params);
      params.actor = actor;
      params.jointIndex = jointIndex;
      ConvertRotationBoundsToMochi(
          MinRotation, MaxRotation, u2m, params.minValues, params.maxValues);
      OutRequest.CreateOne(params, &mochi::Scene::CreateArticulated3dRotationRangeConstraint);
      break;
    }
  }
  return EMochiBodyState::Ready;
}
