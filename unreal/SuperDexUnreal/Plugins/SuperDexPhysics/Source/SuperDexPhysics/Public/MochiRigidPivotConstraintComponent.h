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

#pragma once

#include "CoreMinimal.h"
#include "MochiConstraintComponentBase.h"

#include "MochiRigidPivotConstraintComponent.generated.h"

UENUM(BlueprintType)
enum class EMochiRigidPivotConstraintType : uint8 {
  // Pulls a point on the body to this component's location (Mochi RigidPivotPosition).
  Position UMETA(DisplayName = "Pivot Position"),
  // Pulls a point on the body to where it would be if the body's center of mass were at this
  // component's transform (Mochi RigidPivotToRigidTarget).
  ToRigidTarget UMETA(DisplayName = "Pivot to Rigid Target"),
  // Turns a frame on the body to this component's rotation (Mochi RigidPivotRotation).
  Rotation UMETA(DisplayName = "Pivot Rotation"),
};

// Ties a point or frame of a rigid body (a rigid component or an articulation link) to this
// component. Move, animate or attach the component to move the target. Saturation is in [cm] for
// Pivot Position and Pivot to Rigid Target and [deg] for Pivot Rotation.
UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiRigidPivotConstraintComponent
    : public UMochiConstraintComponentBase {
  GENERATED_BODY()

 public:
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Constraint")
  EMochiRigidPivotConstraintType Type = EMochiRigidPivotConstraintType::Position;
  // The body, by name: "lid", "arm/elbow" for a link, or "inner/lid" inside a child actor.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Constraint")
  FString Body;
  // The pivot point [cm], in the body's local frame.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type != EMochiRigidPivotConstraintType::Rotation", EditConditionHides))
  FVector LocalPosition = FVector::ZeroVector;
  // The pivot frame, in the body's local frame.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type == EMochiRigidPivotConstraintType::Rotation", EditConditionHides))
  FRotator LocalRotation = FRotator::ZeroRotator;

  UMochiRigidPivotConstraintComponent();

 protected:
  virtual EMochiBodyState BuildConstraints(FConstraintRequest& OutRequest) override;
  virtual double GetSaturationScale() const override;
  virtual void OnUpdateTransform(EUpdateTransformFlags UpdateTransformFlags, ETeleportType Teleport)
      override;
};
