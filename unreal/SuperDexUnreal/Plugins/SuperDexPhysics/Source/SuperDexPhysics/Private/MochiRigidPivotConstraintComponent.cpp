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
  const FUnrealToMochiConverter u2m;
  const mochi::Real3 position = u2m.TranslationToMochi(GetComponentLocation());
  const mochi::Quaternion rotation = u2m.RotationToMochi(GetComponentQuat());
  const EMochiRigidPivotConstraintType type = Type;
  ForEachConstraint([type, position, rotation](mochi::Constraint& C) {
    if (type != EMochiRigidPivotConstraintType::Rotation) {
      C.SetTargetPosition(position, mochi::ErrorLog{});
    }
    if (type != EMochiRigidPivotConstraintType::Position) {
      C.SetTargetRotation(rotation, mochi::ErrorLog{});
    }
  });
}

EMochiBodyState UMochiRigidPivotConstraintComponent::BuildConstraints(
    FConstraintRequest& OutRequest) {
  FMochiConstraintBody body;
  mochi::ActorHandle actor;
  const EMochiBodyState state = ResolveBody(Body, body, OutRequest.Scene, actor);
  if (state != EMochiBodyState::Ready) {
    return state;
  }
  const FUnrealToMochiConverter u2m;
  switch (Type) {
    case EMochiRigidPivotConstraintType::Position: {
      mochi::RigidPivotPositionConstraintParams params;
      FillCommonParams(params);
      params.actor = actor;
      params.localPosition = body.LocalPositionToMochi(LocalPosition);
      params.targetPosition = u2m.TranslationToMochi(GetComponentLocation());
      OutRequest.CreateOne(params, &mochi::Scene::CreateRigidPivotPositionConstraint);
      break;
    }
    case EMochiRigidPivotConstraintType::ToRigidTarget: {
      mochi::RigidPivotToRigidTargetConstraintParams params;
      FillCommonParams(params);
      params.actor = actor;
      params.localPosition = body.LocalPositionToMochi(LocalPosition);
      params.targetTransform = u2m.TransformToMochi_RT(GetComponentTransform());
      OutRequest.CreateOne(params, &mochi::Scene::CreateRigidPivotToRigidTargetConstraint);
      break;
    }
    case EMochiRigidPivotConstraintType::Rotation: {
      mochi::RigidPivotRotationConstraintParams params;
      FillCommonParams(params);
      params.actor = actor;
      params.localRotation = u2m.RotationToMochi(LocalRotation.Quaternion()).ToRotationVector();
      params.targetRotation = u2m.RotationToMochi(GetComponentQuat()).ToRotationVector();
      OutRequest.CreateOne(params, &mochi::Scene::CreateRigidPivotRotationConstraint);
      break;
    }
  }
  return EMochiBodyState::Ready;
}
