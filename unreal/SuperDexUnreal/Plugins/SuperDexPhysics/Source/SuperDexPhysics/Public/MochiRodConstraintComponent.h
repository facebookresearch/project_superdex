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

#include "Components/SceneComponent.h"
#include "CoreMinimal.h"
#include "MochiMeshComponentInterface.h"

#include <mochi_physics/mochi_physics.h>

#include <memory>

#include "MochiRodConstraintComponent.generated.h"

/** Component attaching a native rod endpoint to a rigid actor. */
UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiRodConstraintComponent : public USceneComponent {
  GENERATED_BODY()

  /** Name of the UMochiRodMeshComponent to constrain. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics|Rod Constraint")
  FName RodComponentName = NAME_None;

  /** Name of the rigid Mochi mesh component to constrain to. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics|Rod Constraint")
  FName RigidComponentName = NAME_None;

  /** Which rod endpoint to attach. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics|Rod Constraint")
  EMochiRodAttachmentEnd AttachmentEnd = EMochiRodAttachmentEnd::Last;

  /** Whether to allow contact between the constrained pair. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics|Rod Constraint")
  bool bAllowContact = false;

  //----------------------------------------------------------------------------
  // POSITION CONSTRAINT
  //----------------------------------------------------------------------------

  /** Create a DeformableNodeToRigid constraint for position. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics|Rod Constraint|Position Constraint")
  bool bConstrainPosition = true;

  /** Stiffness for position constraint. */
  UPROPERTY(
      EditAnywhere,
      Category = "SuperDex Physics|Rod Constraint|Position Constraint",
      meta = (EditCondition = "bConstrainPosition"))
  float PositionStiffness = mochi::DeformableNodeToRigidConstraintParams{}.stiffness;

  /** Damping for position constraint. */
  UPROPERTY(
      EditAnywhere,
      Category = "SuperDex Physics|Rod Constraint|Position Constraint",
      meta = (EditCondition = "bConstrainPosition"))
  float PositionDamping = mochi::DeformableNodeToRigidConstraintParams{}.damping;

  //----------------------------------------------------------------------------
  // ROTATION CONSTRAINT
  //----------------------------------------------------------------------------

  /** Create a RodElementRotationToRigid constraint for rotation. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics|Rod Constraint|Rotation Constraint")
  bool bConstrainRotation = true;

  /** Stiffness for rotation constraint. */
  UPROPERTY(
      EditAnywhere,
      Category = "SuperDex Physics|Rod Constraint|Rotation Constraint",
      meta = (EditCondition = "bConstrainRotation"))
  float RotationStiffness = mochi::RodElementRotationToRigidConstraintParams{}.stiffness;

  /** Damping for rotation constraint. */
  UPROPERTY(
      EditAnywhere,
      Category = "SuperDex Physics|Rod Constraint|Rotation Constraint",
      meta = (EditCondition = "bConstrainRotation"))
  float RotationDamping = mochi::RodElementRotationToRigidConstraintParams{}.damping;

 public:
  /**
   * Set the position constraint stiffness.
   *
   * @param NewStiffness Constraint stiffness to configure.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Physics")
  void SetPositionStiffness(float NewStiffness);

  /**
   * Set the position constraint damping.
   *
   * @param NewDamping Constraint damping to configure.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Physics")
  void SetPositionDamping(float NewDamping);

  /**
   * Set the rotation constraint stiffness.
   *
   * @param NewStiffness Constraint stiffness to configure.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Physics")
  void SetRotationStiffness(float NewStiffness);

  /**
   * Set the rotation constraint damping.
   *
   * @param NewDamping Constraint damping to configure.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Physics")
  void SetRotationDamping(float NewDamping);

  /** (Re)create constraints created by this component. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Physics")
  void CreateConstraints();

  /** Destroy constraints created by this component. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Physics")
  void DestroyConstraints();

 public:
  /** Initialize the rod-constraint component. */
  UMochiRodConstraintComponent();

 protected:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

 private:
  TWeakObjectPtr<UMochiSceneObject> MochiSceneObject;
  bool bConstraintsCreated = false;

  struct FMochiState {
    mochi::ConstraintHandle PositionConstraintHandle;
    mochi::ConstraintHandle RotationConstraintHandle;
  };

  /** Accessed by the Mochi thread only! */
  std::shared_ptr<FMochiState> MochiThreadState = std::make_shared<FMochiState>();
};
