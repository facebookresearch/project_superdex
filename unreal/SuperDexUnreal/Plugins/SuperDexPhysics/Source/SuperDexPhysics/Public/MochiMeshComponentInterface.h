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
#include "UObject/Object.h"

#include "MochiMeshComponentInterface.generated.h"

// Interface common to all Mochi Mesh types.
UINTERFACE(NotBlueprintable)
class SUPERDEXPHYSICS_API UMochiMeshComponentInterface : public UInterface {
 public:
  GENERATED_BODY()
};

// Interface common to all Mochi Mesh types.
class SUPERDEXPHYSICS_API IMochiMeshComponentInterface {
 public:
  GENERATED_BODY()

  // (Re)create the underlying MochiActorObject controlling this mesh component.
  // For advanced use only. Note this will break constraints on this actor.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  virtual void CreateMochiActor() = 0;
  // Destroy the underlying MochiActorObject controlling this mesh component. This stops
  // the simulation but keeps the component alive. For advanced use only.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  virtual void DestroyMochiActor() = 0;
  // Get the underlying MochiActorObject controller this mesh component.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  virtual UMochiActorObject* GetMochiActor() const = 0;

  // Set the world transform of the Mochi mesh component.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  virtual void SetWorldTransformMochi(const FTransform& NewTransform);
  // Set the world location and rotation of the Mochi mesh component.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  virtual void SetWorldLocationAndRotationMochi(FVector NewLocation, FRotator NewRotation);

  // Get the contact params configured on this component (editor/initial value).
  // Note: does not reflect runtime changes made via SetContactParams.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  virtual UMochiContactParams* GetContactParams() const = 0;
  // Get the material params configured on this component (editor/initial value).
  // Note: does not reflect runtime changes made via SetMaterialParams.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  virtual UMochiMaterialParams* GetMaterialParams() const = 0;

  // Set the contact params of the Mochi mesh component.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  virtual void SetContactParams(const UMochiContactParams* ContactParams);
  // Set the material params of the Mochi mesh component.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  virtual void SetMaterialParams(const UMochiMaterialParams* MaterialParams);

  // Returns the MochiModel used to create this Mochi mesh component. Might be null.
  UFUNCTION(Blueprintable, Category = "Mochi")
  virtual UMochiModel* GetMochiModel() const = 0;
  // Returns true if the mesh has enabled contact queries.
  UFUNCTION(Blueprintable, Category = "Mochi")
  virtual bool GetContactsEnabled() const = 0;

  // Clear boundary conditions
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  virtual void ClearBoundaryConditions();
  // Resets the local deformation to zero
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  virtual void SetZeroDeformation() const;
  // Set the linear and angular velocities
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  virtual void SetVelocities(FVector LinearVelocity, FVector AngularVelocity) const;
  // Resets the linear and angular velocities to zero
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  virtual void SetZeroVelocities();
  // Get inertial properties of the actor (mass [kg], COM [cm], MOI [kg*cm^2])
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  virtual void
  GetInertialParams(double& Mass, FVector& CenterOfMass, FMochiMomentOfInertia& MomentOfInertia);
};

// Interface common to soft/dynamic Mochi Mesh types.
UINTERFACE(NotBlueprintable, Category = "Mochi")
class SUPERDEXPHYSICS_API UMochiSoftMeshComponentInterface : public UInterface {
 public:
  GENERATED_BODY()
};

// Interface common to soft/dynamic Mochi Mesh types.
class SUPERDEXPHYSICS_API IMochiSoftMeshComponentInterface {
 public:
  GENERATED_BODY()
  // Get the mesh volume node reference (i.e. rest) positions in local space.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  virtual const TArray<FVector>& GetVolumeNodeReferencePositions() const = 0;
  // Get the mesh volume node current positions in local space.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  virtual const TArray<FVector>& GetVolumeNodePositions() const = 0;
  // Sets the MochiModel's elements using the given MaterialModifiers
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  virtual const void ApplyMaterialModifiers(UMochiModelMaterialModifiers* Modifiers) const = 0;
};
