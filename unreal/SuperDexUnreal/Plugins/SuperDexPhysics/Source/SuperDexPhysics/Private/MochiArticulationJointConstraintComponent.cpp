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
  const mochi::real Target = MochiDofSign * NewTargetValue;
  ForEachConstraint([Target](mochi::Constraint& C) { C.SetTargetDof(Target, mochi::ErrorLog{}); });
}

void UMochiArticulationJointConstraintComponent::SetTargetRotation(FRotator NewTargetRotation) {
  TargetRotation = NewTargetRotation;
  const mochi::Quaternion Target =
      FUnrealToMochiConverter().RotationToMochi(NewTargetRotation.Quaternion());
  ForEachConstraint(
      [Target](mochi::Constraint& C) { C.SetTargetRotation(Target, mochi::ErrorLog{}); });
}

double UMochiArticulationJointConstraintComponent::GetSaturationScale() const {
  return 1.0;
}

EMochiBodyState UMochiArticulationJointConstraintComponent::BuildConstraints(
    FConstraintRequest& OutRequest) {
  FMochiConstraintBody Body;
  mochi::ActorHandle Actor;
  const EMochiBodyState State = ResolveBody(Articulation, Body, OutRequest.Scene, Actor);
  if (State != EMochiBodyState::Ready) {
    return State;
  }
  const UMochiArticulation* Asset =
      Body.Articulation ? Body.Articulation->GetArticulationAsset() : nullptr;
  if (Asset == nullptr || !Body.Link.IsNone()) {
    LogError(FString::Printf(TEXT("\"%s\" is not an articulation."), *Articulation));
    return EMochiBodyState::Failed;
  }
  const int JointIndex = Asset->GetJointIndex(Joint);
  if (JointIndex < 0) {
    LogError(
        FString::Printf(
            TEXT("articulation %s has no joint %s."), *Articulation, *Joint.ToString()));
    return EMochiBodyState::Failed;
  }
  const EMochiArticulatedJointType JointType = Asset->GetJoints()[JointIndex].Type;
  const FUnrealToMochiConverter U2m;

  switch (Type) {
    case EMochiArticulationJointConstraintType::Target:
    case EMochiArticulationJointConstraintType::Range: {
      // Mochi constrains single DOFs of translations and of 1-DOF rotations only.
      const bool bRotationVector = JointType == EMochiArticulatedJointType::Spherical ||
          (JointType == EMochiArticulatedJointType::Free && Dof >= 3);
      int32 MochiDof = 0;
      if (bRotationVector ||
          !ConvertArticulatedDofIndexToMochi(JointType, Dof, U2m, MochiDof, MochiDofSign)) {
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
        mochi::ArticulatedSingleDofTargetConstraintParams Params;
        FillCommonParams(Params);
        Params.actor = Actor;
        Params.jointIndex = JointIndex;
        Params.dofIndex = MochiDof;
        Params.targetValue = MochiDofSign * TargetValue;
        OutRequest.CreateOne(Params, &mochi::Scene::CreateArticulatedSingleDofTargetConstraint);
      } else {
        mochi::ArticulatedSingleDofRangeConstraintParams Params;
        FillCommonParams(Params);
        Params.actor = Actor;
        Params.jointIndex = JointIndex;
        Params.dofIndex = MochiDof;
        Params.minValue = FMath::Min(MochiDofSign * MinValue, MochiDofSign * MaxValue);
        Params.maxValue = FMath::Max(MochiDofSign * MinValue, MochiDofSign * MaxValue);
        OutRequest.CreateOne(Params, &mochi::Scene::CreateArticulatedSingleDofRangeConstraint);
      }
      break;
    }
    case EMochiArticulationJointConstraintType::RotationTarget: {
      mochi::Articulated3dRotationTargetConstraintParams Params;
      FillCommonParams(Params);
      Params.actor = Actor;
      Params.jointIndex = JointIndex;
      Params.target = U2m.RotationToMochi(TargetRotation.Quaternion());
      OutRequest.CreateOne(Params, &mochi::Scene::CreateArticulated3dRotationTargetConstraint);
      break;
    }
    case EMochiArticulationJointConstraintType::RotationRange: {
      mochi::Articulated3dRotationRangeConstraintParams Params;
      FillCommonParams(Params);
      Params.actor = Actor;
      Params.jointIndex = JointIndex;
      ConvertRotationBoundsToMochi(
          MinRotation, MaxRotation, U2m, Params.minValues, Params.maxValues);
      OutRequest.CreateOne(Params, &mochi::Scene::CreateArticulated3dRotationRangeConstraint);
      break;
    }
  }
  return EMochiBodyState::Ready;
}
