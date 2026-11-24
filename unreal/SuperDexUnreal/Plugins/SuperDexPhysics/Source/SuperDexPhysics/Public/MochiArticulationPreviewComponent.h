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

/** Poseable mesh component for displaying articulation poses. */
UCLASS(meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiArticulationPreviewComponent : public UPoseableMeshComponent {
  GENERATED_BODY()
 protected:
  /** The articulation to preview. */
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "SuperDex Physics|Articulation Poseable Mesh Component")
  TObjectPtr<UMochiArticulationBase> Articulation = nullptr;

  /** The joint positions to pose the articulation preview. */
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      EditFixedSize,
      meta = (EditFixedOrder),
      Category = "SuperDex Physics|Articulation Poseable Mesh Component")
  TArray<double> JointPositions;

 public:
  /** Initialize the poseable articulation preview component. */
  UMochiArticulationPreviewComponent();

  /**
   * Set the articulation to preview.
   *
   * @param NewArticulation Articulation asset to assign.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Physics|Articulation Poseable Mesh Component")
  void SetArticulation(UMochiArticulationBase* NewArticulation);

  /**
   * Get the articulation being previewed.
   *
   * @return The articulation.
   */
  UFUNCTION(BlueprintPure, Category = "SuperDex Physics|Articulation Poseable Mesh Component")
  UMochiArticulationBase* GetArticulation() const;

  /**
   * Set the joint positions to pose the articulation preview.
   *
   * @param NewJointPositions Joint positions to apply.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Physics|Articulation Poseable Mesh Component")
  void SetJointPositions(const TArray<double>& NewJointPositions);

  /**
   * Set a named bone transform in the articulation preview.
   *
   * @param BoneName Name of the preview bone.
   * @param InTransform Transform to apply to the preview bone.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Physics|Articulation Poseable Mesh Component")
  void SetBoneLocalTransformByName(FName BoneName, const FTransform& InTransform);

  /**
   * Set the local bone rotations in the articulation preview.
   *
   * @param InRotations Local bone rotations to apply.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Physics|Articulation Poseable Mesh Component")
  void SetBoneLocalRotations(const TArray<FQuat>& InRotations);

 protected:
#if WITH_EDITOR
  virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif
  virtual void OnRegister() override;
};
