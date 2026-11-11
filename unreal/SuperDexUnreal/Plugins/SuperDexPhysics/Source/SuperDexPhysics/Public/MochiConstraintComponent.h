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

#include "Components/ActorComponent.h"
#include "CoreMinimal.h"
#include "MochiMeshComponentInterface.h"

#include <memory>

#include "MochiConstraintComponent.generated.h"

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiConstraintComponent : public USceneComponent {
  GENERATED_BODY()

 public:
  // Mode of constraint.
  UPROPERTY(EditAnywhere, Category = "Mochi Constraint")
  EMochiConstraintComponentMode Mode = EMochiConstraintComponentMode::OneBody;

  // Name of first Mochi component to constrain. Must belong to Owner common to this component.
  UPROPERTY(EditAnywhere, Category = "Mochi Constraint")
  FName ComponentName1 = NAME_None;
  // Name of second Mochi component to constrain. Must belong to Owner common to this component.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint",
      meta = (EditCondition = "Mode == EMochiConstraintComponentMode::TwoBody"))
  FName ComponentName2 = NAME_None;

  // Should the constrained actors be able to collide with each other? If true, then be careful to
  // ensure that the actors are not touching when the constraint is in its resting configuration.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint",
      meta = (EditCondition = "Mode == EMochiConstraintComponentMode::TwoBody"))
  bool AllowContact = false;

  //----------------------------------------------------------------------------
  // LOCATION CONSTRAINT
  //----------------------------------------------------------------------------

  // Enable to constraint translation between component 1 and this component, or between component 1
  // and component 2.
  UPROPERTY(EditAnywhere, Category = "Mochi Constraint|Location Constraint")
  bool bConstrainLocation = true;
  // Stiffness for location constraint.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint|Location Constraint",
      meta = (EditCondition = "bConstrainLocation"))
  float LinearStiffness = 1.0f;
  // Damping for location constraint.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint|Location Constraint",
      meta = (EditCondition = "bConstrainLocation"))
  float LinearDamping = 0.1f;
  // Optionally define a free translation axis.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint|Location Constraint",
      meta =
          (EditCondition = "Mode == EMochiConstraintComponentMode::TwoBody && bConstrainLocation"))
  TEnumAsByte<EAxis::Type> FreeAxis = EAxis::None;
  // Limit positive and negative translation along the free axis.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint|Location Constraint",
      meta =
          (EditCondition =
               "Mode == EMochiConstraintComponentMode::TwoBody && bConstrainLocation && FreeAxis != EAxis::None"))
  bool bLimitFreeAxis = false;
  // Free axis translation range if bLimitFreeAxisRange.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint|Location Constraint",
      meta =
          (EditCondition =
               "Mode == EMochiConstraintComponentMode::TwoBody && bConstrainLocation && FreeAxis != EAxis::None && bLimitFreeAxis"))
  FFloatRange FreeAxisRange = FFloatRange(-10.0f, 10.0f);
  // Stiffness of free axis. Should be less than LinearStiffness.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint|Location Constraint",
      meta =
          (EditCondition =
               "Mode == EMochiConstraintComponentMode::TwoBody && bConstrainLocation && FreeAxis != EAxis::None"))
  float FreeAxisLinearStiffness = 0.0f;
  // Damping of free axis. Should be less than LinearDamping.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint|Location Constraint",
      meta =
          (EditCondition =
               "Mode == EMochiConstraintComponentMode::TwoBody && bConstrainLocation && FreeAxis != EAxis::None"))
  float FreeAxisLinearDamping = 0.0f;

  //----------------------------------------------------------------------------
  // LOCATION LIMITS
  //----------------------------------------------------------------------------

  // TODO

  //----------------------------------------------------------------------------
  // ROTATION CONSTRAINT
  //----------------------------------------------------------------------------

  // Enable to constraint rotation between component 1 and this component.
  UPROPERTY(EditAnywhere, Category = "Mochi Constraint|Rotation Constraint")
  bool bConstrainRotation = false;
  // Stiffness for rotation constraint.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint|Rotation Constraint",
      meta = (EditCondition = "bConstrainRotation"))
  float TorsionalStiffness = 0.01f;
  // Stiffness for rotation constraint.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint|Rotation Constraint",
      meta = (EditCondition = "bConstrainRotation"))
  float TorsionalDamping = 0.01f;
  // Multiplier applied to TorsionalStiffness and TorsionalDamping, for values smaller than the
  // Details panel allows you to enter.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint|Rotation Constraint",
      meta = (EditCondition = "bConstrainRotation"))
  float TorsionalGain = 1.0f;

  //----------------------------------------------------------------------------
  // ROTATION LIMITS
  //----------------------------------------------------------------------------

  // Enable to limit relative rotation between component 1 and component 2.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint|Rotation Limits",
      meta = (EditCondition = "Mode == EMochiConstraintComponentMode::TwoBody"))
  bool bLimitRotation = false;
  // Allowed relative rotation between component 1 and component 2 along RefFrameRotationVector
  // X-axis.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint|Rotation Limits",
      meta = (EditCondition = "Mode == EMochiConstraintComponentMode::TwoBody && bLimitRotation"))
  FFloatRange AngleRangeX = FFloatRange(-360.0f, 360.0f);
  // Allowed relative rotation between component 1 and component 2 along RefFrameRotationVector
  // Y-axis.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint|Rotation Limits",
      meta = (EditCondition = "Mode == EMochiConstraintComponentMode::TwoBody && bLimitRotation"))
  FFloatRange AngleRangeY = FFloatRange(-360.0f, 360.0f);
  // Allowed relative rotation between component 1 and component 2 along RefFrameRotationVector
  // Z-axis.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint|Rotation Limits",
      meta = (EditCondition = "Mode == EMochiConstraintComponentMode::TwoBody && bLimitRotation"))
  FFloatRange AngleRangeZ = FFloatRange(-360.0f, 360.0f);
  // Stiffness of rotation limit.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint|Rotation Limits",
      meta = (EditCondition = "Mode == EMochiConstraintComponentMode::TwoBody && bLimitRotation"))
  float RotationLimitStiffness = 0.1f;
  // Damping of rotation limit.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint|Rotation Limits",
      meta = (EditCondition = "Mode == EMochiConstraintComponentMode::TwoBody && bLimitRotation"))
  float RotationLimitDamping = 0.0f;

  // Set the location constraint stiffness to a new value.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void SetLinearStiffness(float Stiffness);
  // Set the location constraint damping to a new value.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void SetLinearDamping(float Damping);
  // Set the rotation constraint stiffness to a new value.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void SetTorsionalStiffness(float Stiffness);
  // Set the rotation constraint damping to a new value.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void SetTorsionalDamping(float Damping);

  // Get the first constrained Mochi component if it exists.
  UFUNCTION(BlueprintPure, Category = "Mochi")
  USceneComponent* GetComponent1() const {
    return Component1;
  }
  // Get the second constrained Mochi component if it exists.
  UFUNCTION(BlueprintPure, Category = "Mochi")
  USceneComponent* GetComponent2() const {
    return Component2;
  }

  // (Re)create constraints created by this component.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void CreateConstraints();
  // Destroy constraints created by this component.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void DestroyConstraints();

  UFUNCTION(BlueprintCallable, Category = "Mochi")
  bool OverrideComponent1(USceneComponent* Component, FVector ConnectionPoint);

  UFUNCTION(BlueprintCallable, Category = "Mochi")
  bool OverrideComponent2(USceneComponent* Component, FVector ConnectionPoint);

 public:
  UMochiConstraintComponent();

 protected:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
  virtual void OnUpdateTransform(EUpdateTransformFlags UpdateTransformFlags, ETeleportType Teleport)
      override;

 public:
  virtual void TickComponent(
      float DeltaTime,
      ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction) override;

 private:
  void TryCreateConstraints_OneBody();
  void TryCreateConstraints_TwoBody();

 private:
  TWeakObjectPtr<UMochiSceneObject> MochiSceneObject;
  bool bConstraintsCreated = false;

  USceneComponent* Component1 = nullptr;
  USceneComponent* Component2 = nullptr;

  bool bOverrideComponent1ConnectionPoint = false;
  FVector Component1ConnectionPointOverride = FVector::Zero();

  bool bOverrideComponent2ConnectionPoint = false;
  FVector Component2ConnectionPointOverride = FVector::Zero();

  struct MochiState {
    // constraints for EMochiConstraintComponentMode::TwoBody
    mochi::ConstraintHandle RigidSphericalJointConstraintHandle;
    mochi::ConstraintHandle RigidPrismaticJointConstraintHandle;
    mochi::ConstraintHandle JointRotationRangeConstraintHandle;
    mochi::ConstraintHandle JointRotationTrackingConstraintHandle;

    // constraints for EMochiConstraintComponentMode::OneBody
    mochi::ConstraintHandle RigidPivotPositionConstraintHandle;
    mochi::ConstraintHandle RigidPivotRotationConstraintHandle;
  };

  // Accessed by the Mochi thread only!
  std::shared_ptr<MochiState> MochiThreadState = std::make_shared<MochiState>();
};
