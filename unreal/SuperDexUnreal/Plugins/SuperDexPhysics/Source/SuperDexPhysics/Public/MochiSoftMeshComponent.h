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

/** Dynamic mesh component backed by a native volumetric soft actor. */
UCLASS(ClassGroup = (Custom), ShowCategories = (Mobility), meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiSoftMeshComponent : public UMochiDeformableMeshComponent,
                                                    public IMochiSoftMeshComponentInterface {
  GENERATED_BODY()

 public:
  /** Parameters controlling physical material properties. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics")
  UMochiMaterialParams* MaterialParams = nullptr;

  /** Collider representation used by the soft actor. */
  UPROPERTY(
      EditAnywhere,
      Category = "SuperDex Physics",
      meta = (ValidEnumValues = "None, Auto, Sdf"))
  EMochiColliderType ColliderType =
      ::ToUnreal(mochi::experimental::ExperimentalSoftActorParams{}.colliderType);

  /** The boundary element type used for the soft actor's surface discretization. */
  UPROPERTY(
      EditAnywhere,
      Category = "SuperDex Physics",
      meta = (ValidEnumValues = "P1Q1, P1Q3, P1Q6, Default"))
  EMochiActorBoundaryElementType BoundaryElementType =
      ::ToUnreal(mochi::SoftActorParams{}.boundaryElementType);

  /**
   * Use the model's embedded contact skin for contact. Rendering does not depend on this setting:
   * without a visual mesh, the model's contact skin is rendered whenever it has skinning data.
   */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics", meta = (EditCondition = "!bEnableRom"))
  bool bUseContactSkin = mochi::SoftActorParams{}.useContactSkin;

  /** Modifiers for individual elements that get applied during initialization */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics")
  UMochiModelMaterialModifiers* MaterialModifiers = nullptr;

  /** Recomputes local soft actor frame for more optimal performance. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics")
  bool bUseRecentering = mochi::experimental::ExperimentalSoftActorParams{}.useRecentering;

  /** Enable volume node positions query. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics")
  bool bEnableVolumeNodePositionsQuery = false;

  /** Enable elastic energy query. */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics")
  bool bEnableElasticEnergyQuery = false;

  // EXPERIMENTAL

  /**
   * Enable reduced order model (ROM). The imported .mochi.h5 MochiModel must provide ROM sources
   * inside of a rom group.
   */
  UPROPERTY(EditAnywhere, Category = "SuperDex Physics|Soft Mesh|Experimental")
  bool bEnableRom = false;

  /** Reduced-order-model source and sampling settings. */
  UPROPERTY(
      EditAnywhere,
      Category = "SuperDex Physics|Soft Mesh|Experimental",
      meta = (ShowOnlyInnerProperties, EditCondition = "bEnableRom"))
  FMochiRomParams RomParams;

 public:
  /**
   * Get the elastic energy collected by the native query.
   *
   * @return Elastic energy cached from the native query.
   */
  UFUNCTION(BlueprintPure, Category = "SuperDex Physics")
  float GetElasticEnergy() const {
    return ElasticEnergy;
  }

 public:
  // Begin IMochiMeshComponentInterface
  virtual void CreateMochiActor() override;
  virtual UMochiMaterialParams* GetMaterialParams() const override {
    return MaterialParams;
  }

  /**
   * IMochiSoftMeshComponentInterface
   *
   * @return The volume node reference positions.
   */
  virtual const TArray<FVector>& GetVolumeNodeReferencePositions() const override;
  virtual const TArray<FVector>& GetVolumeNodePositions() const override;
  virtual const void ApplyMaterialModifiers(UMochiModelMaterialModifiers* Modifiers) const override;

 public:
  /**
   * Initialize the soft mesh component and its native-actor lifecycle settings.
   *
   * @param ObjectInitializer Unreal object-initialization settings.
   */
  UMochiSoftMeshComponent(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

  /**
   * Assign the model asset used by this soft component.
   *
   * @param NewMochiModel Model asset to assign.
   */
  void SetMochiModel(UMochiModel* NewMochiModel);

  // TODO: No the most graceful, but permits a path to create a dynamic mesh component from an
  // existing Mochi actor handle.
  /**
   * Configure this component to use an existing native actor handle.
   *
   * @param Handle Native actor handle to use for this component.
   */
  void SetExistingMochiActorHandle(mochi::ActorHandle Handle);

  // Begin UObject Interface.
#if WITH_EDITOR
  virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

 protected:
  virtual void OnRegister() override;
  virtual void OnQueryDataApplied(const FMochiActorQueryData& QueryData) override;

 private:
  /** Cached node positions in local space. */
  TArray<FVector> VolumeNodeReferencePositions;
  TArray<FVector> VolumeNodePositions;

  /** Total elastic energy in this soft body, relative to rest, received from TryGetQueryData */
  float ElasticEnergy = 0.0f;
};
