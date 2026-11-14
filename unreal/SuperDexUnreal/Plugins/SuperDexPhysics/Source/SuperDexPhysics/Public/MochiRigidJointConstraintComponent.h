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

#include "MochiRigidJointConstraintComponent.generated.h"

UENUM(BlueprintType)
enum class EMochiRigidJointConstraintType : uint8 {
  // Ties a point on each body together; the bodies rotate freely (Mochi RigidSphericalJoint).
  SphericalJoint UMETA(DisplayName = "Spherical Joint"),
  // Lets the bodies slide along one axis only, optionally within limits (Mochi
  // RigidPrismaticJoint).
  PrismaticJoint UMETA(DisplayName = "Prismatic Joint"),
  // Holds the bodies' relative rotation (Mochi JointRotationTracking).
  RotationTracking UMETA(DisplayName = "Rotation Tracking"),
  // Keeps the bodies' relative rotation within per-axis ranges (Mochi JointRotationRange).
  RotationRange UMETA(DisplayName = "Rotation Range"),
};

// A joint between two rigid bodies (rigid components or articulation links). Prismatic Joint slides
// along this component's X axis, and Rotation Tracking and Rotation Range measure rotations in this
// component's frame, as placed when the constraint is created. Saturation is in [cm] for the
// spherical and prismatic joints and [deg] for the rotation constraints.
UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiRigidJointConstraintComponent
    : public UMochiConstraintComponentBase {
  GENERATED_BODY()

 public:
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Constraint")
  EMochiRigidJointConstraintType Type = EMochiRigidJointConstraintType::SphericalJoint;
  // The first body, by name: "lid", "arm/elbow" for a link, or "inner/lid" inside a child actor.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Constraint")
  FString BodyA;
  // The second body.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Constraint")
  FString BodyB;
  // Whether the two bodies can touch once constrained. If true, make sure they don't overlap in
  // their constrained configuration. Contact stays disabled after the constraint is destroyed.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Constraint")
  bool bAllowContact = false;
  // The joint point on body A [cm], in its local frame.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type == EMochiRigidJointConstraintType::SphericalJoint",
           EditConditionHides))
  FVector LocalPositionA = FVector::ZeroVector;
  // The joint point on body B [cm], in its local frame.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type == EMochiRigidJointConstraintType::SphericalJoint",
           EditConditionHides))
  FVector LocalPositionB = FVector::ZeroVector;
  // Whether the slide has a lower limit.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type == EMochiRigidJointConstraintType::PrismaticJoint",
           EditConditionHides))
  bool bHasMinLimit = false;
  // Lower limit of the slide from the rest position [cm].
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type == EMochiRigidJointConstraintType::PrismaticJoint && bHasMinLimit",
           EditConditionHides))
  double MinLimit = 0.0;
  // Whether the slide has an upper limit.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type == EMochiRigidJointConstraintType::PrismaticJoint",
           EditConditionHides))
  bool bHasMaxLimit = false;
  // Upper limit of the slide from the rest position [cm].
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type == EMochiRigidJointConstraintType::PrismaticJoint && bHasMaxLimit",
           EditConditionHides))
  double MaxLimit = 0.0;
  // Range of each component of the relative rotation vector about this component's axes [deg].
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type == EMochiRigidJointConstraintType::RotationRange",
           EditConditionHides))
  FVector MinAngles = FVector(-360.0);
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type == EMochiRigidJointConstraintType::RotationRange",
           EditConditionHides))
  FVector MaxAngles = FVector(360.0);
  // Whether the ranges are relative to the bodies' relative rotation when the constraint is
  // created, rather than to this component's frame.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type == EMochiRigidJointConstraintType::RotationRange",
           EditConditionHides))
  bool bRangeAroundRest = true;

  // Set the target rotation of body B relative to body A of a Rotation Tracking constraint.
  UFUNCTION(BlueprintCallable, Category = "Mochi Constraint")
  void SetTargetRotation(FRotator NewTargetRotation);

 protected:
  virtual EMochiBodyState BuildConstraints(FConstraintRequest& OutRequest) override;
  virtual double GetSaturationScale() const override;

 private:
  // The target set by SetTargetRotation, kept for constraints created later.
  TOptional<mochi::Quaternion> TargetRotation;
};
