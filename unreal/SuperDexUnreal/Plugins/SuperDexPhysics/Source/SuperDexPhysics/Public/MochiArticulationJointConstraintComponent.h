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

#include "MochiArticulationJointConstraintComponent.generated.h"

UENUM(BlueprintType)
enum class EMochiArticulationJointConstraintType : uint8 {
  // Drives one DOF of the joint toward a target value (Mochi ArticulatedSingleDofTarget).
  Target UMETA(DisplayName = "Joint Target"),
  // Keeps one DOF of the joint within a range (Mochi ArticulatedSingleDofRange).
  Range UMETA(DisplayName = "Joint Range"),
  // Drives the rotation of a spherical joint toward a target (Mochi Articulated3dRotationTarget).
  RotationTarget UMETA(DisplayName = "Joint Rotation Target"),
  // Keeps the rotation of a spherical joint within per-axis ranges (Mochi
  // Articulated3dRotationRange).
  RotationRange UMETA(DisplayName = "Joint Rotation Range"),
};

// Drives or limits a joint of an articulation. Values are in the articulation's joint units [m or
// rad], like its joint limits and positions.
UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiArticulationJointConstraintComponent
    : public UMochiConstraintComponentBase {
  GENERATED_BODY()

 public:
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Constraint")
  EMochiArticulationJointConstraintType Type = EMochiArticulationJointConstraintType::Target;
  // The articulation component, by name: "arm", or "inner/arm" inside a child actor.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Constraint")
  FString Articulation;
  // The joint, by name.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Constraint")
  FName Joint = NAME_None;
  // The DOF of the joint: 0 for revolute and prismatic joints, or the translation axis of a free
  // joint (Mochi has no single-DOF constraints on the rotations of spherical and free joints).
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (ClampMin = "0",
           ClampMax = "2",
           EditCondition =
               "Type == EMochiArticulationJointConstraintType::Target || Type == EMochiArticulationJointConstraintType::Range",
           EditConditionHides))
  int32 Dof = 0;
  // Target value of the DOF [m or rad].
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type == EMochiArticulationJointConstraintType::Target",
           EditConditionHides))
  double TargetValue = 0.0;
  // Range of the DOF [m or rad].
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type == EMochiArticulationJointConstraintType::Range",
           EditConditionHides))
  double MinValue = 0.0;
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type == EMochiArticulationJointConstraintType::Range",
           EditConditionHides))
  double MaxValue = 0.0;
  // Target rotation of the joint relative to its rest orientation, in the joint frame.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type == EMochiArticulationJointConstraintType::RotationTarget",
           EditConditionHides))
  FRotator TargetRotation = FRotator::ZeroRotator;
  // Range of each component of the joint's rotation vector [rad], in the joint frame.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type == EMochiArticulationJointConstraintType::RotationRange",
           EditConditionHides))
  FVector MinRotation = FVector::ZeroVector;
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type == EMochiArticulationJointConstraintType::RotationRange",
           EditConditionHides))
  FVector MaxRotation = FVector::ZeroVector;

  // Set the target value of a Joint Target [m or rad].
  UFUNCTION(BlueprintCallable, Category = "Mochi Constraint")
  void SetTargetValue(double NewTargetValue);
  // Set the target rotation of a Joint Rotation Target.
  UFUNCTION(BlueprintCallable, Category = "Mochi Constraint")
  void SetTargetRotation(FRotator NewTargetRotation);

 protected:
  virtual EMochiBodyState BuildConstraints(FConstraintRequest& OutRequest) override;
  virtual double GetSaturationScale() const override;

 private:
  // The sign of the Mochi DOF relative to Dof, set when the constraint is built.
  double MochiDofSign = 1.0;
};
