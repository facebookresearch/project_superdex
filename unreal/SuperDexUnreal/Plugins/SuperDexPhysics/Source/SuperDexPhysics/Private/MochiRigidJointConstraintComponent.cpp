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
  const mochi::Quaternion target =
      FUnrealToMochiConverter().RotationToMochi(NewTargetRotation.Quaternion());
  TargetRotation = target;
  ForEachConstraint(
      [target](mochi::Constraint& C) { C.SetTargetRotation(target, mochi::ErrorLog{}); });
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
  FMochiConstraintBody bodyA;
  FMochiConstraintBody bodyB;
  mochi::ActorHandle actorA;
  mochi::ActorHandle actorB;
  UMochiSceneObject* sceneB = nullptr;
  const EMochiBodyState stateA = ResolveBody(BodyA, bodyA, OutRequest.Scene, actorA);
  const EMochiBodyState stateB = ResolveBody(BodyB, bodyB, sceneB, actorB);
  if (stateA != EMochiBodyState::Ready || stateB != EMochiBodyState::Ready) {
    return stateA == EMochiBodyState::Failed || stateB == EMochiBodyState::Failed
        ? EMochiBodyState::Failed
        : EMochiBodyState::Pending;
  }
  if (OutRequest.Scene != sceneB) {
    LogError(FString::Printf(TEXT("%s and %s are not in the same Mochi scene."), *BodyA, *BodyB));
    return EMochiBodyState::Failed;
  }
  if (!bAllowContact) {
    OutRequest.NoContactPairs.emplace_back(actorA, actorB);
  }

  const FUnrealToMochiConverter u2m;
  const mochi::Real3 refFrameRotVec = u2m.RotationToMochi(GetComponentQuat()).ToRotationVector();
  switch (Type) {
    case EMochiRigidJointConstraintType::SphericalJoint: {
      mochi::RigidSphericalJointConstraintParams params;
      FillCommonParams(params);
      params.actorA = actorA;
      params.actorB = actorB;
      params.localPosA = bodyA.LocalPositionToMochi(LocalPositionA);
      params.localPosB = bodyB.LocalPositionToMochi(LocalPositionB);
      OutRequest.CreateOne(params, &mochi::Scene::CreateRigidSphericalJointConstraint);
      break;
    }
    case EMochiRigidJointConstraintType::PrismaticJoint: {
      mochi::RigidPrismaticJointConstraintParams params;
      FillCommonParams(params);
      params.actorA = actorA;
      params.actorB = actorB;
      params.freeAxis = u2m.DirectionToMochi(GetForwardVector());
      // The limits scale with this component, as with the scale of an imported nested prefab.
      const double scale = GetComponentScale().X;
      if (bHasMinLimit) {
        params.min = u2m.MeasureToMochi(MinLimit * scale);
      }
      if (bHasMaxLimit) {
        params.max = u2m.MeasureToMochi(MaxLimit * scale);
      }
      OutRequest.CreateOne(params, &mochi::Scene::CreateRigidPrismaticJointConstraint);
      break;
    }
    case EMochiRigidJointConstraintType::RotationTracking: {
      mochi::JointRotationTrackingConstraintParams params;
      FillCommonParams(params);
      params.actorA = actorA;
      params.actorB = actorB;
      params.refFrameRotVec = refFrameRotVec;
      // Without a target set, the constraint holds the bodies' relative rotation at creation.
      OutRequest.Create = [params, target = TargetRotation](mochi::Scene* Scene) {
        mochi::ErrorLog e;
        mochi::Constraint* constraint = Scene->CreateJointRotationTrackingConstraint(params, e);
        if (constraint && target) {
          constraint->SetTargetRotation(*target, e);
        }
        return std::vector{constraint ? constraint->GetHandle() : mochi::ConstraintHandle{}};
      };
      break;
    }
    case EMochiRigidJointConstraintType::RotationRange: {
      mochi::JointRotationRangeConstraintParams params;
      FillCommonParams(params);
      params.actorA = actorA;
      params.actorB = actorB;
      params.refFrameRotVec = refFrameRotVec;
      params.rangeAroundRest = bRangeAroundRest;
      mochi::Real3 minAngles;
      mochi::Real3 maxAngles;
      ConvertRotationBoundsToMochi(
          FMath::DegreesToRadians(MinAngles),
          FMath::DegreesToRadians(MaxAngles),
          u2m,
          minAngles,
          maxAngles);
      params.angleRangeX = {minAngles[0], maxAngles[0]};
      params.angleRangeY = {minAngles[1], maxAngles[1]};
      params.angleRangeZ = {minAngles[2], maxAngles[2]};
      OutRequest.CreateOne(params, &mochi::Scene::CreateJointRotationRangeConstraint);
      break;
    }
  }
  return EMochiBodyState::Ready;
}
