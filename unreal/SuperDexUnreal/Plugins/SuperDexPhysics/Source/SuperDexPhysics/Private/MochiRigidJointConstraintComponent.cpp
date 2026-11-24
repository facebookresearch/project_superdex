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

#include "MochiRigidJointConstraintComponent.h"
#include "MochiArticulationTypes.h"
#include "MochiConversions.h"

void UMochiRigidJointConstraintComponent::SetTargetRotation(FRotator NewTargetRotation) {
  const mochi::Quaternion Target =
      FUnrealToMochiConverter().RotationToMochi(NewTargetRotation.Quaternion());
  TargetRotation = Target;
  ForEachConstraint(
      [Target](mochi::Constraint& C) { C.SetTargetRotation(Target, mochi::ErrorLog{}); });
}

double UMochiRigidJointConstraintComponent::GetSaturationScale() const {
  switch (Type) {
    case EMochiRigidJointConstraintType::SphericalJoint:
    case EMochiRigidJointConstraintType::PrismaticJoint:
      return FUnrealToMochiConverter().MeasureToMochi(1.0);
    case EMochiRigidJointConstraintType::RotationTracking:
    case EMochiRigidJointConstraintType::RotationRange:
      break;
  }
  return FMath::DegreesToRadians(1.0);
}

EMochiBodyState UMochiRigidJointConstraintComponent::BuildConstraints(
    FConstraintRequest& OutRequest) {
  FMochiConstraintBody ResolvedBodyA;
  FMochiConstraintBody ResolvedBodyB;
  mochi::ActorHandle ActorA;
  mochi::ActorHandle ActorB;
  UMochiSceneObject* SceneB = nullptr;
  const EMochiBodyState StateA = ResolveBody(BodyA, ResolvedBodyA, OutRequest.Scene, ActorA);
  const EMochiBodyState StateB = ResolveBody(BodyB, ResolvedBodyB, SceneB, ActorB);
  if (StateA != EMochiBodyState::Ready || StateB != EMochiBodyState::Ready) {
    return StateA == EMochiBodyState::Failed || StateB == EMochiBodyState::Failed
        ? EMochiBodyState::Failed
        : EMochiBodyState::Pending;
  }
  if (OutRequest.Scene != SceneB) {
    LogError(FString::Printf(TEXT("%s and %s are not in the same Mochi scene."), *BodyA, *BodyB));
    return EMochiBodyState::Failed;
  }
  if (!bAllowContact) {
    OutRequest.NoContactPairs.emplace_back(ActorA, ActorB);
  }

  const FUnrealToMochiConverter U2m;
  const mochi::Real3 RefFrameRotVec = U2m.RotationToMochi(GetComponentQuat()).ToRotationVector();
  switch (Type) {
    case EMochiRigidJointConstraintType::SphericalJoint: {
      mochi::RigidSphericalJointConstraintParams Params;
      FillCommonParams(Params);
      Params.actorA = ActorA;
      Params.actorB = ActorB;
      Params.localPosA = ResolvedBodyA.LocalPositionToMochi(LocalPositionA);
      Params.localPosB = ResolvedBodyB.LocalPositionToMochi(LocalPositionB);
      OutRequest.CreateOne(Params, &mochi::Scene::CreateRigidSphericalJointConstraint);
      break;
    }
    case EMochiRigidJointConstraintType::PrismaticJoint: {
      mochi::RigidPrismaticJointConstraintParams Params;
      FillCommonParams(Params);
      Params.actorA = ActorA;
      Params.actorB = ActorB;
      Params.freeAxis = U2m.DirectionToMochi(GetForwardVector());
      // The limits scale with this component, as with the scale of an imported nested prefab.
      const double Scale = GetComponentScale().X;
      if (bHasMinLimit) {
        Params.min = U2m.MeasureToMochi(MinLimit * Scale);
      }
      if (bHasMaxLimit) {
        Params.max = U2m.MeasureToMochi(MaxLimit * Scale);
      }
      OutRequest.CreateOne(Params, &mochi::Scene::CreateRigidPrismaticJointConstraint);
      break;
    }
    case EMochiRigidJointConstraintType::RotationTracking: {
      mochi::JointRotationTrackingConstraintParams Params;
      FillCommonParams(Params);
      Params.actorA = ActorA;
      Params.actorB = ActorB;
      Params.refFrameRotVec = RefFrameRotVec;
      // Without a target set, the constraint holds the bodies' relative rotation at creation.
      OutRequest.Create = [Params, Target = TargetRotation](mochi::Scene* Scene) {
        mochi::ErrorLog E;
        mochi::Constraint* Constraint = Scene->CreateJointRotationTrackingConstraint(Params, E);
        if (Constraint && Target) {
          Constraint->SetTargetRotation(*Target, E);
        }
        return std::vector{Constraint ? Constraint->GetHandle() : mochi::ConstraintHandle{}};
      };
      break;
    }
    case EMochiRigidJointConstraintType::RotationRange: {
      mochi::JointRotationRangeConstraintParams Params;
      FillCommonParams(Params);
      Params.actorA = ActorA;
      Params.actorB = ActorB;
      Params.refFrameRotVec = RefFrameRotVec;
      Params.rangeAroundRest = bRangeAroundRest;
      mochi::Real3 MochiMinAngles;
      mochi::Real3 MochiMaxAngles;
      ConvertRotationBoundsToMochi(
          FMath::DegreesToRadians(MinAngles),
          FMath::DegreesToRadians(MaxAngles),
          U2m,
          MochiMinAngles,
          MochiMaxAngles);
      Params.angleRangeX = {MochiMinAngles[0], MochiMaxAngles[0]};
      Params.angleRangeY = {MochiMinAngles[1], MochiMaxAngles[1]};
      Params.angleRangeZ = {MochiMinAngles[2], MochiMaxAngles[2]};
      OutRequest.CreateOne(Params, &mochi::Scene::CreateJointRotationRangeConstraint);
      break;
    }
  }
  return EMochiBodyState::Ready;
}
