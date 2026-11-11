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
#include "MochiMeshComponentInterface.h"

#include "MochiImplicitShapeComponent.generated.h"

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiImplicitShapeComponent : public USceneComponent,
                                                         public IMochiMeshComponentInterface {
  GENERATED_BODY()

  // Implicit shape type.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  EMochiImplicitShape Shape = EMochiImplicitShape::Plane;
  // Sphere radius in world units (centimeters).
  UPROPERTY(
      EditAnywhere,
      meta = (EditCondition = "Shape == EMochiImplicitShape::Sphere", EditConditionHides),
      Category = "Mochi")
  float SphereRadius = 50;
  // The layer this object is simulated on.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  FName Layer = FMochiLayer::GenericObject;
  // Parameters controlling physical material properties.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  UMochiMaterialParams* MaterialParams = nullptr;
  // Parameters controlling contacts.
  UPROPERTY(EditAnywhere, Category = "Mochi")
  UMochiContactParams* ContactParams;

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
  virtual bool GetContactsEnabled() const override;
  virtual UMochiModel* GetMochiModel() const override;

 protected:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

 private:
  UPROPERTY()
  UMochiActorObject* MochiActorObject = nullptr;
};
