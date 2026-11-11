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

#include "MochiSoftConstraintComponent.generated.h"

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiSoftConstraintComponent : public USceneComponent {
  GENERATED_BODY()

  // Mode of constraint.
  UPROPERTY(EditAnywhere, Category = "Mochi Constraint")
  EMochiConstraintComponentMode Mode = EMochiConstraintComponentMode::TwoBody;

  // Name of first Mochi component to constrain. Must belong to Owner common to this component.
  UPROPERTY(EditAnywhere, Category = "Mochi Constraint")
  FName ComponentName1 = NAME_None;
  // Name of second Mochi component to constrain. Must belong to Owner common to this component.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint",
      meta = (EditCondition = "Mode == EMochiConstraintComponentMode::TwoBody"))
  FName ComponentName2 = NAME_None;
  // The name of either a *BoxCollision* or *SphereCollision* component that defines the volume of
  // soft nodes to constrain.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint",
      meta = (EditCondition = "Mode == EMochiConstraintComponentMode::OneBody"))
  FName OneBodyVolumeName = NAME_None;

  // Stiffness for node constraints.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint",
      meta = (EditCondition = "Mode == EMochiConstraintComponentMode::TwoBody"))
  float Stiffness = 10.0f;
  // Damping for node constraints.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint",
      meta = (EditCondition = "Mode == EMochiConstraintComponentMode::TwoBody"))
  float Damping = 0.0f;

  // Should the constrained actors be able to collide with each other? If true, then be careful to
  // ensure that the actors are not touching when the constraint is in its resting configuration.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Constraint",
      meta = (EditCondition = "Mode == EMochiConstraintComponentMode::TwoBody"))
  bool AllowContact = false;

 public:
  // Set the node constraint stiffness to a new value. Only supported in TwoBody case.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void SetStiffness(float NewStiffness);
  // Set the node constraint damping to a new value. Only supported in TwoBody case.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void SetDamping(float NewDamping);
  // Get the indices of the nodes constrained by this component.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  bool GetConstrainedNodeIndices(TArray<int>& ConstrainedNodeIndices);

  // (Re)create constraints created by this component.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void CreateConstraints();
  // Destroy constraints created by this component.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void DestroyConstraints();

 public:
  UMochiSoftConstraintComponent();

 protected:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

 public:
  virtual void TickComponent(
      float DeltaTime,
      ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction) override;

 private:
  void TryCreateConstraints_OneBody();
  void TryCreateConstraints_TwoBody();

 private:
  USceneComponent* Component1 = nullptr;
  USceneComponent* Component2 = nullptr;
  USceneComponent* OneBodyVolume = nullptr;
  TWeakObjectPtr<UMochiSceneObject> MochiSceneObject;
  std::atomic_bool bConstraintsCreated = false;

  struct MochiState {
    TArray<mochi::ConstraintHandle> ConstraintHandles;
    TArray<int> ConstrainedNodeIndices;
    mochi::ActorHandle DeformableActorHandle;
  };

  // Accessed by the Mochi thread only!
  std::shared_ptr<MochiState> MochiThreadState = std::make_shared<MochiState>();
};
