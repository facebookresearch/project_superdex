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

#include <memory>

#include "MochiRodConstraintComponent.generated.h"

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiRodConstraintComponent : public USceneComponent {
  GENERATED_BODY()

  // Name of the UMochiRodMeshComponent to constrain.
  UPROPERTY(EditAnywhere, Category = "Mochi Rod Constraint")
  FName RodComponentName = NAME_None;

  // Name of the rigid Mochi mesh component to constrain to.
  UPROPERTY(EditAnywhere, Category = "Mochi Rod Constraint")
  FName RigidComponentName = NAME_None;

  // Which rod endpoint to attach.
  UPROPERTY(EditAnywhere, Category = "Mochi Rod Constraint")
  EMochiRodAttachmentEnd AttachmentEnd = EMochiRodAttachmentEnd::Last;

  // Whether to allow contact between the constrained pair.
  UPROPERTY(EditAnywhere, Category = "Mochi Rod Constraint")
  bool bAllowContact = false;

  //----------------------------------------------------------------------------
  // POSITION CONSTRAINT
  //----------------------------------------------------------------------------

  // Create a DeformableNodeToRigid constraint for position.
  UPROPERTY(EditAnywhere, Category = "Mochi Rod Constraint|Position Constraint")
  bool bConstrainPosition = true;

  // Stiffness for position constraint.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Rod Constraint|Position Constraint",
      meta = (EditCondition = "bConstrainPosition"))
  float PositionStiffness = 1e4f;

  // Damping for position constraint.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Rod Constraint|Position Constraint",
      meta = (EditCondition = "bConstrainPosition"))
  float PositionDamping = 0.0f;

  //----------------------------------------------------------------------------
  // ROTATION CONSTRAINT
  //----------------------------------------------------------------------------

  // Create a RodElementRotationToRigid constraint for rotation.
  UPROPERTY(EditAnywhere, Category = "Mochi Rod Constraint|Rotation Constraint")
  bool bConstrainRotation = true;

  // Stiffness for rotation constraint.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Rod Constraint|Rotation Constraint",
      meta = (EditCondition = "bConstrainRotation"))
  float RotationStiffness = 1e2f;

  // Damping for rotation constraint.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Rod Constraint|Rotation Constraint",
      meta = (EditCondition = "bConstrainRotation"))
  float RotationDamping = 0.0f;

 public:
  // Set the position constraint stiffness.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void SetPositionStiffness(float NewStiffness);

  // Set the position constraint damping.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void SetPositionDamping(float NewDamping);

  // Set the rotation constraint stiffness.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void SetRotationStiffness(float NewStiffness);

  // Set the rotation constraint damping.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void SetRotationDamping(float NewDamping);

  // (Re)create constraints created by this component.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void CreateConstraints();

  // Destroy constraints created by this component.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void DestroyConstraints();

 public:
  UMochiRodConstraintComponent();

 protected:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

 private:
  TWeakObjectPtr<UMochiSceneObject> MochiSceneObject;
  bool bConstraintsCreated = false;

  struct MochiState {
    mochi::ConstraintHandle PositionConstraintHandle;
    mochi::ConstraintHandle RotationConstraintHandle;
  };

  // Accessed by the Mochi thread only!
  std::shared_ptr<MochiState> MochiThreadState = std::make_shared<MochiState>();
};
