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
#include "GameFramework/Actor.h"
#include "MochiArticulation.h"
#include "MochiArticulationComponent.h"
#include "MochiArticulationPreviewComponent.h"
#include "MochiSoftMeshComponent.h"

#include "MochiActors.generated.h"

//------------------------------------------------------------------------------
// MOCHI SOFT MESH ACTOR
//------------------------------------------------------------------------------

/** Actor owning a simulated Mochi soft mesh component. */
UCLASS(
    hidecategories = (Input),
    showcategories = ("Input|MouseInput", "Input|TouchInput"),
    ConversionRoot,
    ComponentWrapperClass,
    meta = (ChildCanTick))
class SUPERDEXPHYSICS_API AMochiSoftMeshActor : public AActor {
  GENERATED_UCLASS_BODY()

 private:
  /** Soft mesh component simulated by this actor. */
  UPROPERTY(
      Category = MochiSoftMeshActor,
      VisibleAnywhere,
      BlueprintReadOnly,
      meta =
          (ExposeFunctionCategories = "Mesh,Rendering,Physics,Components|MochiSoftMesh",
           AllowPrivateAccess = "true"))
  TObjectPtr<class UMochiSoftMeshComponent> MochiSoftMeshComponent;

 protected:
  virtual void BeginPlay() override;

 public:
#if WITH_EDITOR
  virtual void CheckForErrors() override;
  virtual bool GetReferencedContentObjects(TArray<UObject*>& Objects) const override;
#endif // WITH_EDITOR

  /**
   * Get the soft mesh component owned by this actor.
   *
   * @return The mochi soft mesh component.
   */
  class UMochiSoftMeshComponent* GetMochiSoftMeshComponent() const {
    return MochiSoftMeshComponent;
  }

  static FName MochiSoftMeshComponentName;
};

//------------------------------------------------------------------------------
// MOCHI ARTICULATION ACTOR
//------------------------------------------------------------------------------

/** Actor that owns a simulated Mochi articulation component. */
UCLASS(
    Blueprintable,
    HideCategories = (Input),
    ShowCategories = ("Input|MouseInput", "Input|TouchInput"),
    ConversionRoot,
    ComponentWrapperClass,
    meta = (ChildCanTick))
class SUPERDEXPHYSICS_API AMochiArticulationActor : public AActor {
  GENERATED_BODY()

 private:
  /** Component that creates and controls the articulation. */
  UPROPERTY(
      Category = MochiArticulationPreviewActor,
      VisibleAnywhere,
      BlueprintReadOnly,
      meta =
          (ExposeFunctionCategories = "Mesh,Rendering,Components|MochiArticulationMeshComponent",
           AllowPrivateAccess = "true"))
  TObjectPtr<class UMochiArticulationComponent> ArticulationComponent = nullptr;

 public:
  /**
   * Set the articulation asset used by the component.
   *
   * @param NewArticulation Articulation asset to assign.
   */
  UFUNCTION(BlueprintCallable, Category = "MochiArticulationActor")
  void SetArticulationAsset(UMochiArticulation* NewArticulation);

  /**
   * Get the articulation asset assigned to the component.
   *
   * @return The articulation asset.
   */
  UFUNCTION(BlueprintPure, Category = "MochiArticulationActor")
  UMochiArticulation* GetArticulationAsset() const;

  /**
   * Get the component that simulates the articulation.
   *
   * @return The articulation component.
   */
  UFUNCTION(BlueprintPure, Category = "MochiArticulationActor")
  UMochiArticulationComponent* GetArticulationComponent() const;

 public:
  /** Create the articulation actor's default component. */
  AMochiArticulationActor();
  virtual void Tick(float DeltaTime) override;

 protected:
  virtual void BeginPlay() override;

#if WITH_EDITOR
  virtual void CheckForErrors() override;
  virtual bool GetReferencedContentObjects(TArray<UObject*>& Objects) const override;
#endif

 private:
  static FName MochiArticulationComponentName;
};

//------------------------------------------------------------------------------
// MOCHI ARTICULATION PREVIEW ACTOR
//------------------------------------------------------------------------------

/** Actor that previews articulation poses without creating a simulated articulation. */
UCLASS(
    Blueprintable,
    HideCategories = (Input),
    ShowCategories = ("Input|MouseInput", "Input|TouchInput"),
    ConversionRoot,
    ComponentWrapperClass,
    meta = (ChildCanTick))
class SUPERDEXPHYSICS_API AMochiArticulationPreviewActor : public AActor {
  GENERATED_BODY()

 private:
  /** Poseable component used to display the articulation preview. */
  UPROPERTY(
      Category = MochiArticulationPreviewActor,
      VisibleAnywhere,
      BlueprintReadOnly,
      meta =
          (ExposeFunctionCategories =
               "Mesh,Rendering,Components|MochiArticulationPoseableMeshComponent",
           AllowPrivateAccess = "true"))
  TObjectPtr<class UMochiArticulationPreviewComponent> PoseableMeshComponent = nullptr;

 public:
  /**
   * Set the articulation to preview.
   *
   * @param NewArticulation Articulation asset to assign.
   */
  UFUNCTION(BlueprintCallable, Category = "MochiArticulationPreviewActor")
  void SetArticulation(UMochiArticulationBase* NewArticulation);
  /**
   * Get the articulation being previewed.
   *
   * @return The articulation.
   */
  UFUNCTION(BlueprintPure, Category = "MochiArticulationPreviewActor")
  UMochiArticulationBase* GetArticulation() const;
  /**
   * Set the joint positions to pose the articulation preview.
   *
   * @param NewJointPositions Joint positions to apply.
   */
  UFUNCTION(BlueprintCallable, Category = "MochiArticulationPreviewActor")
  void SetJointPositions(const TArray<double>& NewJointPositions);
  /**
   * Get the poseable mesh component used for previewing the articulation.
   *
   * @return The poseable mesh component.
   */
  UFUNCTION(BlueprintPure, Category = "MochiArticulationPreviewActor")
  UMochiArticulationPreviewComponent* GetPoseableMeshComponent() const;

 public:
  /** Create the articulation preview actor's default component. */
  AMochiArticulationPreviewActor();
  virtual void Tick(float DeltaTime) override;

 protected:
  virtual void BeginPlay() override;

#if WITH_EDITOR
  virtual void CheckForErrors() override;
  virtual bool GetReferencedContentObjects(TArray<UObject*>& Objects) const override;
#endif

 private:
  static FName PoseableMeshComponentName;
};
