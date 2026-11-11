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

#include "Components/DynamicMeshComponent.h"
#include "CoreMinimal.h"
#include "MochiActorObject.h"
#include "MochiMeshComponentInterface.h"

#include "MochiDeformableMeshComponent.generated.h"

/// Abstract base for Mochi deformable mesh components (shell, soft).
/// Owns shared UPROPERTYs, lifecycle, and the per-step query pump. Subclasses
/// implement @ref CreateMochiActor and may override @ref OnQueryDataApplied.
UCLASS(Abstract, ClassGroup = (Custom), ShowCategories = (Mobility))
class SUPERDEXPHYSICS_API UMochiDeformableMeshComponent : public UDynamicMeshComponent,
                                                          public IMochiMeshComponentInterface {
  GENERATED_BODY()

 public:
  // The Mochi Model to use.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  UMochiModel* Model = nullptr;
  // Should this mesh be influenced by scene gravity?
  UPROPERTY(EditAnywhere, Category = "Mochi")
  bool bHasGravity = true;
  // The layer this object is simulated on.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  FName Layer = FMochiLayer::GenericObject;
  // Parameters controlling contacts.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  UMochiContactParams* ContactParams = nullptr;
  // Use Mochi computed node normals to set the dynamic mesh normals.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  bool bUseMochiNodeNormals = true;
  // Enable contact queries and events.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  bool bEnableContacts = false;
  // Hide self-contacts (where the other component is this same component, e.g.
  // shell point-cloud self-contacts) from the contact event delegates and
  // GetContactPoints() - including downstream consumers like
  // MochiForceVisualizerComponent. This does NOT affect whether self-contacts
  // are simulated; that is controlled by the actor's own simulation parameters
  // (e.g. UMochiShellMeshComponent::bPointCloudSelfContact). Defaults to true
  // to preserve historical behavior.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi",
      meta = (EditCondition = "bEnableContacts", DisplayName = "Filter Self-Contacts From Events"))
  bool bFilterSelfContacts = true;

  // Emitted when this component begins contact with another Mochi component.
  // bEnableContacts must be enabled.
  UPROPERTY(BlueprintAssignable, Category = "Mochi")
  FOnMochiComponentBeginContact OnMochiComponentBeginContact;
  // Emitted when this component ends contact with another Mochi component.
  // bEnableContacts must be enabled.
  UPROPERTY(BlueprintAssignable, Category = "Mochi")
  FOnMochiComponentEndContact OnMochiComponentEndContact;
  // Emitted when this component is contacting any other Mochi components.
  // bEnableContacts must be enabled.
  UPROPERTY(BlueprintAssignable, Category = "Mochi")
  FOnMochiComponentContact OnMochiComponentContact;

 public:
  UFUNCTION(BlueprintPure, Category = "Mochi")
  const TArray<FMochiContactPoint>& GetContactPoints() const;

 public:
  // Begin IMochiMeshComponentInterface
  /// `PURE_VIRTUAL` because `UCLASS(Abstract)` still instantiates the CDO.
  virtual void CreateMochiActor() override
      PURE_VIRTUAL(UMochiDeformableMeshComponent::CreateMochiActor, );
  virtual void DestroyMochiActor() override;
  virtual UMochiActorObject* GetMochiActor() const override;
  virtual UMochiContactParams* GetContactParams() const override {
    return ContactParams;
  }
  /// Default null for subclasses whose material isn't a `UMochiMaterialParams` (e.g., shell).
  /// Override otherwise.
  virtual UMochiMaterialParams* GetMaterialParams() const override {
    return nullptr;
  }
  virtual UMochiModel* GetMochiModel() const override;
  virtual bool GetContactsEnabled() const override;

 public:
  UMochiDeformableMeshComponent(
      const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
  virtual void TickComponent(
      float DeltaTime,
      enum ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction) override;

  // Begin UPrimitiveComponent Interface
  virtual UMaterialInterface* GetMaterial(int32 ElementIndex) const override;
  virtual void SetMaterial(int32 ElementIndex, UMaterialInterface* Material) override;

 protected:
  /// Subclass hook called from @ref TickComponent after the standard per-step update.
  /// Default no-op.
  virtual void OnQueryDataApplied(const FMochiActorQueryData& QueryData) {}

 protected:
  UPROPERTY()
  UMochiActorObject* MochiActorObject = nullptr;
  FMochiContactEventHandler ContactEventHandler;
};
