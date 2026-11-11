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

#include "Components/StaticMeshComponent.h"
#include "CoreMinimal.h"
#include "MochiActorObject.h"
#include "MochiMeshComponentInterface.h"
#include "MochiTypes.h"

#include "MochiRigidMeshComponent.generated.h"

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent), HideCategories = (Physics))
class SUPERDEXPHYSICS_API UMochiRigidMeshComponent : public UStaticMeshComponent,
                                                     public IMochiMeshComponentInterface {
  GENERATED_BODY()

  // If enabled, the static mesh will be simulated directly in mochi as a triangular mesh
  // If disabled, the mochi model asset will be simulated in mochi as a tetrahedral mesh
  // Note that it is HIGHLY recommended you toggle bAllowCPUAccess in the static mesh's
  // properties if this is true. Failure to do this may result in undefined behavior
  // (particularly packaged builds) due to GPU synchronization issues.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  bool bUseStaticMeshAsMochiModel = false;

  // How to pull the triangular mesh from the static mesh.
  UPROPERTY(EditAnywhere, meta = (EditCondition = "bUseStaticMeshAsMochiModel"), Category = "Mochi")
  ETriangularMeshCollisionSource CollisionSource = ETriangularMeshCollisionSource::Complex;

  // The Mochi Model to use.
  UPROPERTY(
      EditAnywhere,
      meta = (EditCondition = "!bUseStaticMeshAsMochiModel"),
      Category = "Mochi")
  UMochiModel* Model = nullptr;
  // Is this a static Mochi actor (i.e. non-simulated)?
  UPROPERTY(EditAnywhere, Category = "Mochi")
  bool bIsStatic = false;
  // Should this mesh be influenced by scene gravity?
  UPROPERTY(EditAnywhere, meta = (EditCondition = "!bIsStatic"), Category = "Mochi")
  bool bHasGravity = true;
  // The layer this object is simulated on.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  FName Layer = FMochiLayer::GenericObject;
  // The Mochi collider type. Prefer to use Sdf in most cases.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi",
      meta = (ValidEnumValues = "None, Auto, Sphere, Box, Mesh, Plane, Sdf"))
  EMochiColliderType ColliderType = EMochiColliderType::Sdf;
  // Optionally override default grid SDF params
  UPROPERTY(
      EditAnywhere,
      meta =
          (EditCondition = "ColliderType==EMochiColliderType::Sdf && bUseStaticMeshAsMochiModel",
           EditConditionHides),
      Category = "Mochi")
  FMochiGridSdfParams GridSdfParams = {};
  // The boundary element type used for the rigid actor's surface discretization.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi",
      meta = (ValidEnumValues = "P1Q1, P1Q3, P1Q6, Default"))
  EMochiActorBoundaryElementType BoundaryElementType = EMochiActorBoundaryElementType::Default;
  // Enable boundary subsampling to reduce contact sample count.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  bool bEnableBoundarySubsampling = false;
  // Boundary subsampling parameters (density + strategy).
  UPROPERTY(
      EditAnywhere,
      meta = (EditCondition = "bEnableBoundarySubsampling", EditConditionHides),
      Category = "Mochi")
  FMochiBoundarySubsamplingParams BoundarySubsamplingParams;
  // Parameters controlling physical material properties.
  UPROPERTY(EditAnywhere, meta = (EditCondition = "!bIsStatic"), Category = "Mochi")
  UMochiMaterialParams* MaterialParams = nullptr;
  // Parameters controlling contacts.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  UMochiContactParams* ContactParams = nullptr;
  // Override mass value.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  bool bOverrideMass = false;
  // Overriding mass [kg]
  UPROPERTY(
      EditAnywhere,
      meta = (EditCondition = "bOverrideMass", EditConditionHides),
      Category = "Mochi")
  double Mass = 0;
  // Override inertial properties.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  bool bOverrideInertialParams = false;
  // Override center of mass.
  UPROPERTY(
      EditAnywhere,
      meta = (EditCondition = "bOverrideInertialParams", EditConditionHides),
      Category = "Mochi")
  FVector CenterOfMass = {};
  UPROPERTY(
      EditAnywhere,
      meta = (EditCondition = "bOverrideInertialParams", EditConditionHides),
      Category = "Mochi")
  FMochiMomentOfInertia MomentOfInertia = {};
  // Enable contact queries and events.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  bool bEnableContacts = false;
  // Freeze actor translation and/or rotation to world-aligned axes.
  UPROPERTY(
      EditAnywhere,
      meta = (Bitmask, BitmaskEnum = "/Script/SuperDexPhysics.EMochiFreezeConstraint"),
      Category = "Mochi")
  int FreezeConstraints = 0;

 public:
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
  // Set the model to be used. Only valid if the Mochi actor has not yet been created
  // (i.e. before BeginPlay). Returns true if successful.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  bool SetModel(UMochiModel* NewModel);

  // Set the world location and rotation as boundarty conditions of the Mochi mesh component.
  // This allows the Mochi actor to update dynamic states, namely previous state, previous velocity
  // which allow, on the clearing of boundary conditions, for the actor to continue along the the
  // trajectory imposed by previous boundary conditions (think of pinch and release to throw and
  // object).
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void SetWorldLocationAndRotationMochiWithDynamicsTransform(const FTransform& NewTransform);

  // Same as above (the above call this function when invoked filtering scale)
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void SetWorldLocationAndRotationMochiWithDynamics(
      const FVector& NewLocation,
      const FRotator& NewRotation);

  UFUNCTION(BlueprintPure, Category = "Mochi")
  FVector GetWorldLinearVelocity() const;
  UFUNCTION(BlueprintPure, Category = "Mochi")
  FVector GetWorldAngularVelocity() const;
  UFUNCTION(BlueprintPure, Category = "Mochi")
  const TArray<FMochiContactPoint>& GetContactPoints() const;

 public:
  // Begin IMochiMeshComponentInterface
  virtual void CreateMochiActor() override;
  virtual void DestroyMochiActor() override;
  virtual UMochiActorObject* GetMochiActor() const override;
  virtual UMochiContactParams* GetContactParams() const override {
    return ContactParams;
  }
  virtual UMochiMaterialParams* GetMaterialParams() const override {
    return MaterialParams;
  }
  virtual UMochiModel* GetMochiModel() const override;
  virtual bool GetContactsEnabled() const override;

 public:
  UMochiRigidMeshComponent(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

  void SetMochiModel(UMochiModel* NewMochiModel);

 public:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
  virtual void TickComponent(
      float DeltaTime,
      enum ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction) override;

  // Begin UObject Interface.
#if WITH_EDITOR
  virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

 private:
  UPROPERTY()
  UMochiActorObject* MochiActorObject = nullptr;
  FMochiContactEventHandler ContactEventHandler;

  FVector LinearVelocity;
  FVector AngularVelocity;

  friend class FSuperDexPrefabImporter;
};
