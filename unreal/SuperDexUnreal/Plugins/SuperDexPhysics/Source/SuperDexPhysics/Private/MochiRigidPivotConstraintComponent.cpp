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

#include "MochiRigidPivotConstraintComponent.h"
#include "MochiConversions.h"

UMochiRigidPivotConstraintComponent::UMochiRigidPivotConstraintComponent() {
  bWantsOnUpdateTransform = true;
}

double UMochiRigidPivotConstraintComponent::GetSaturationScale() const {
  return Type == EMochiRigidPivotConstraintType::Rotation
      ? FMath::DegreesToRadians(1.0)
      : FUnrealToMochiConverter().MeasureToMochi(1.0);
}

void UMochiRigidPivotConstraintComponent::OnUpdateTransform(
    EUpdateTransformFlags UpdateTransformFlags,
    ETeleportType Teleport) {
  Super::OnUpdateTransform(UpdateTransformFlags, Teleport);
  const FUnrealToMochiConverter U2m;
  const mochi::Real3 Position = U2m.TranslationToMochi(GetComponentLocation());
  const mochi::Quaternion Rotation = U2m.RotationToMochi(GetComponentQuat());
  const EMochiRigidPivotConstraintType ConstraintType = Type;
  ForEachConstraint([ConstraintType, Position, Rotation](mochi::Constraint& C) {
    if (ConstraintType != EMochiRigidPivotConstraintType::Rotation) {
      C.SetTargetPosition(Position, mochi::ErrorLog{});
    }
    if (ConstraintType != EMochiRigidPivotConstraintType::Position) {
      C.SetTargetRotation(Rotation, mochi::ErrorLog{});
    }
  });
}

EMochiBodyState UMochiRigidPivotConstraintComponent::BuildConstraints(
    FConstraintRequest& OutRequest) {
  FMochiConstraintBody ResolvedBody;
  mochi::ActorHandle Actor;
  const EMochiBodyState State = ResolveBody(Body, ResolvedBody, OutRequest.Scene, Actor);
  if (State != EMochiBodyState::Ready) {
    return State;
  }
  const FUnrealToMochiConverter U2m;
  switch (Type) {
    case EMochiRigidPivotConstraintType::Position: {
      mochi::RigidPivotPositionConstraintParams Params;
      FillCommonParams(Params);
      Params.actor = Actor;
      Params.localPosition = ResolvedBody.LocalPositionToMochi(LocalPosition);
      Params.targetPosition = U2m.TranslationToMochi(GetComponentLocation());
      OutRequest.CreateOne(Params, &mochi::Scene::CreateRigidPivotPositionConstraint);
      break;
    }
    case EMochiRigidPivotConstraintType::ToRigidTarget: {
      mochi::RigidPivotToRigidTargetConstraintParams Params;
      FillCommonParams(Params);
      Params.actor = Actor;
      Params.localPosition = ResolvedBody.LocalPositionToMochi(LocalPosition);
      Params.targetTransform = U2m.TransformToMochi_RT(GetComponentTransform());
      OutRequest.CreateOne(Params, &mochi::Scene::CreateRigidPivotToRigidTargetConstraint);
      break;
    }
    case EMochiRigidPivotConstraintType::Rotation: {
      mochi::RigidPivotRotationConstraintParams Params;
      FillCommonParams(Params);
      Params.actor = Actor;
      Params.localRotation = U2m.RotationToMochi(LocalRotation.Quaternion()).ToRotationVector();
      Params.targetRotation = U2m.RotationToMochi(GetComponentQuat()).ToRotationVector();
      OutRequest.CreateOne(Params, &mochi::Scene::CreateRigidPivotRotationConstraint);
      break;
    }
  }
  return EMochiBodyState::Ready;
}
