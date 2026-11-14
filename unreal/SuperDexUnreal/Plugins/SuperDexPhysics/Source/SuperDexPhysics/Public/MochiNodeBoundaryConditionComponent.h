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

#include <atomic>
#include <memory>

#include "MochiNodeBoundaryConditionComponent.generated.h"

// Pins every node of a deformable body (a soft or shell component) inside a Box or Sphere Collision
// component where it is at BeginPlay, with Mochi boundary conditions: the nodes cannot move at all.
// For a pin that pulls with a stiffness, use a Node Position constraint.
UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiNodeBoundaryConditionComponent : public UActorComponent {
  GENERATED_BODY()

 public:
  // The deformable body, by name: "cloth", or "inner/cloth" inside a child actor.
  UPROPERTY(EditAnywhere, Category = "Mochi Boundary Condition")
  FString Body;
  // The name of the Box Collision or Sphere Collision component whose volume holds the nodes to
  // pin.
  UPROPERTY(EditAnywhere, Category = "Mochi Boundary Condition")
  FName Volume = NAME_None;

  // Get the indices of the pinned nodes.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  bool GetPinnedNodeIndices(TArray<int>& OutNodeIndices) const;

  // (Re)pin the nodes inside the volume where they are now.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void CreateBoundaryConditions();
  // Release the pinned nodes. Clears every boundary condition of the body.
  UFUNCTION(BlueprintCallable, Category = "Mochi")
  void DestroyBoundaryConditions();

 protected:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

 private:
  TWeakObjectPtr<UMochiSceneObject> MochiSceneObject;
  bool bCreated = false;

  struct FMochiState {
    TArray<int> PinnedNodeIndices;
    mochi::ActorHandle DeformableActorHandle;
    std::atomic<bool> bReady = false;
  };

  // Accessed by the Mochi thread only, except PinnedNodeIndices once bReady.
  std::shared_ptr<FMochiState> MochiThreadState = std::make_shared<FMochiState>();
};
