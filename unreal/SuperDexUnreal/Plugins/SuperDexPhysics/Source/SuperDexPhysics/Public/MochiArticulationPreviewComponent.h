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

#include "Components/PoseableMeshComponent.h"
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MochiArticulationBase.h"

#include "MochiArticulationPreviewComponent.generated.h"

UCLASS(meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiArticulationPreviewComponent : public UPoseableMeshComponent {
  GENERATED_BODY()
 protected:
  // The articulation to preview.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Articulation Poseable Mesh Component")
  TObjectPtr<UMochiArticulationBase> Articulation = nullptr;
  // The joint positions to pose the articulation preview.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      EditFixedSize,
      meta = (EditFixedOrder),
      Category = "Articulation Poseable Mesh Component")
  TArray<double> JointPositions;

 public:
  UMochiArticulationPreviewComponent();
  // Set the articulation to preview.
  UFUNCTION(BlueprintCallable, Category = "Articulation Poseable Mesh Component")
  void SetArticulation(UMochiArticulationBase* NewArticulation);
  // Get the articulation being previewed.
  UFUNCTION(BlueprintPure, Category = "Articulation Poseable Mesh Component")
  UMochiArticulationBase* GetArticulation() const;
  // Set the joint positions to pose the articulation preview.
  UFUNCTION(BlueprintCallable, Category = "Articulation Poseable Mesh Component")
  void SetJointPositions(const TArray<double>& NewJointPositions);

  UFUNCTION(BlueprintCallable, Category = "Articulation Poseable Mesh Component")
  void SetBoneLocalTransformByName(FName BoneName, const FTransform& InTransform);
  UFUNCTION(BlueprintCallable, Category = "Articulation Poseable Mesh Component")
  void SetBoneLocalRotations(const TArray<FQuat>& InRotations);

 protected:
#if WITH_EDITOR
  virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif
  virtual void OnRegister() override;
};
