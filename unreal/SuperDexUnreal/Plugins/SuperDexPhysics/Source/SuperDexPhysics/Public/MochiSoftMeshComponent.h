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
#include "MochiActorObject.h"
#include "MochiDeformableMeshComponent.h"
#include "MochiMeshComponentInterface.h"

#include "MochiSoftMeshComponent.generated.h"

UCLASS(ClassGroup = (Custom), ShowCategories = (Mobility), meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiSoftMeshComponent : public UMochiDeformableMeshComponent,
                                                    public IMochiSoftMeshComponentInterface {
  GENERATED_BODY()

 public:
  // Parameters controlling physical material properties.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  UMochiMaterialParams* MaterialParams = nullptr;
  UPROPERTY(EditAnywhere, Category = "Mochi", meta = (ValidEnumValues = "None, Auto, Sdf"))
  EMochiColliderType ColliderType = EMochiColliderType::None;
  // The boundary element type used for the soft actor's surface discretization.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi",
      meta = (ValidEnumValues = "P1Q1, P1Q3, P1Q6, Default"))
  EMochiActorBoundaryElementType BoundaryElementType = EMochiActorBoundaryElementType::Default;
  // Use the model's embedded contact skin for contact. Rendering does not depend on this setting:
  // without a visual mesh, the model's contact skin is rendered whenever it has skinning data.
  UPROPERTY(EditAnywhere, Category = "Mochi", meta = (EditCondition = "!bEnableRom"))
  bool bUseContactSkin = false;
  // Modifiers for individual elements that get applied during initialization
  UPROPERTY(EditAnywhere, Category = "Mochi")
  UMochiModelMaterialModifiers* MaterialModifiers = nullptr;
  // Recomputes local soft actor frame for more optimal performance.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  bool bUseRecentering = true;
  // Enable volume node positions query.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  bool bEnableVolumeNodePositionsQuery = false;
  // Enable elastic energy query.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  bool bEnableElasticEnergyQuery = false;
  // TEXCOORD_0 and COLOR_0 (linear RGBA) of the prefab render model, one per visual-mesh node, set
  // by the prefab importer when the render model is the visual mesh. Either may be empty. They are
  // drawn only when bUseMochiNodeNormals is set, since flat shading is not supported. Not shown in
  // the Details panel, which would build a property node per element.
  UPROPERTY()
  TArray<FVector2f> RenderModelUVs;
  UPROPERTY()
  TArray<FLinearColor> RenderModelColors;

  // EXPERIMENTAL

  // Enable reduced order model (ROM). The imported .mochi.h5 MochiModel must provide ROM sources
  // inside of a rom group.
  UPROPERTY(EditAnywhere, Category = "MochiSoftMeshComponent|Experimental")
  bool bEnableRom = false;
  UPROPERTY(
      EditAnywhere,
      Category = "MochiSoftMeshComponent|Experimental",
      meta = (ShowOnlyInnerProperties, EditCondition = "bEnableRom"))
  FMochiRomParams RomParams;

 public:
  UFUNCTION(BlueprintPure, Category = "Mochi")
  float GetElasticEnergy() const {
    return ElasticEnergy;
  }

 public:
  // Begin IMochiMeshComponentInterface
  virtual void CreateMochiActor() override;
  virtual UMochiMaterialParams* GetMaterialParams() const override {
    return MaterialParams;
  }
  // IMochiSoftMeshComponentInterface
  virtual const TArray<FVector>& GetVolumeNodeReferencePositions() const override;
  virtual const TArray<FVector>& GetVolumeNodePositions() const override;
  virtual const void ApplyMaterialModifiers(UMochiModelMaterialModifiers* Modifiers) const override;

 public:
  UMochiSoftMeshComponent(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

  void SetMochiModel(UMochiModel* NewMochiModel);

  virtual void BeginPlay() override;

  // TODO: No the most graceful, but permits a path to create a dynamic mesh component from an
  // existing Mochi actor handle.
  void SetExistingMochiActorHandle(mochi::ActorHandle Handle);

  // Begin UObject Interface.
#if WITH_EDITOR
  virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

 protected:
  virtual void OnRegister() override;
  virtual void OnQueryDataApplied(const FMochiActorQueryData& QueryData) override;

 private:
  void WarnIfRenderModelAttributesNotDrawn() const;

  // TODO: Again, not very graceful, but these fields permit construction of this component from an
  // existing actor, specifically for skinned hand meshes.
  mochi::ActorHandle MochiExistingActor;

  // Cached node positions in local space.
  TArray<FVector> VolumeNodeReferencePositions;
  TArray<FVector> VolumeNodePositions;
  // Total elastic energy in this soft body, relative to rest, received from TryGetQueryData
  float ElasticEnergy = 0.0f;
};
