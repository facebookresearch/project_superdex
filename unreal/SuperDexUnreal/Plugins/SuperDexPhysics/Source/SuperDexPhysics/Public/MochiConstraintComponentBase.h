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

#include "Components/SceneComponent.h"
#include "CoreMinimal.h"
#include "MochiConstraintUtil.h"

#include <functional>
#include <memory>
#include <vector>

#include "MochiConstraintComponentBase.generated.h"

class UMochiSceneObject;

// Base of the Mochi constraint components. Each creates Mochi constraints of the type it is set to
// between the bodies it names, at BeginPlay or as soon as their Mochi actors exist, and destroys
// them at EndPlay.
UCLASS(Abstract)
class SUPERDEXPHYSICS_API UMochiConstraintComponentBase : public USceneComponent {
  GENERATED_BODY()

 public:
  // Stiffness [N/m for a distance, N*m/rad for an angle].
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Constraint", meta = (ClampMin = "0"))
  float Stiffness = 1e6f;
  // Damping [N*s/m for a distance, N*m*s/rad for an angle].
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Constraint", meta = (ClampMin = "0"))
  float Damping = 0.0f;
  // Deviation beyond which the constraint force stops growing, in the unit of what the constraint
  // drives (see the subclass). Negative disables saturation.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Constraint")
  float Saturation = -1.0f;

  UFUNCTION(BlueprintCallable, Category = "Mochi Constraint")
  void SetStiffness(float NewStiffness);
  UFUNCTION(BlueprintCallable, Category = "Mochi Constraint")
  void SetDamping(float NewDamping);
  UFUNCTION(BlueprintCallable, Category = "Mochi Constraint")
  void SetSaturation(float NewSaturation);

  // (Re)create the constraints from the current properties, as soon as the bodies' Mochi actors
  // exist.
  UFUNCTION(BlueprintCallable, Category = "Mochi Constraint")
  void CreateConstraints();
  // Destroy the constraints.
  UFUNCTION(BlueprintCallable, Category = "Mochi Constraint")
  void DestroyConstraints();
  // Whether the constraints have been queued for creation.
  UFUNCTION(BlueprintPure, Category = "Mochi Constraint")
  bool AreConstraintsCreated() const {
    return bCreated;
  }

  UMochiConstraintComponentBase();

 protected:
  // What a subclass resolved and wants created.
  struct FConstraintRequest {
    UMochiSceneObject* Scene = nullptr;
    // Creates the constraints on the Mochi thread and returns their handles.
    std::function<std::vector<mochi::ConstraintHandle>(mochi::Scene*)> Create;
    // Pairs of actors that must not touch once the constraints are created.
    std::vector<std::pair<mochi::ActorHandle, mochi::ActorHandle>> NoContactPairs;

    // Sets Create to create one constraint from Params with a Scene::Create*Constraint function.
    template <typename TParams>
    void CreateOne(
        const TParams& Params,
        mochi::Constraint* (mochi::Scene::*CreateFn)(TParams const&, mochi::Error&)) {
      Create = [Params, CreateFn](mochi::Scene* Scene) {
        mochi::ErrorLog e;
        const mochi::Constraint* constraint = (Scene->*CreateFn)(Params, e);
        return std::vector{constraint ? constraint->GetHandle() : mochi::ConstraintHandle{}};
      };
    }
  };

  // Resolves the bodies and fills OutRequest when they are Ready. Pending is retried next tick;
  // Failed is logged.
  virtual EMochiBodyState BuildConstraints(FConstraintRequest& OutRequest) PURE_VIRTUAL(
      UMochiConstraintComponentBase::BuildConstraints, return EMochiBodyState::Failed;);
  // Mochi units [m or rad] per unit of Saturation.
  virtual double GetSaturationScale() const
      PURE_VIRTUAL(UMochiConstraintComponentBase::GetSaturationScale, return 1.0;);

  // Saturation in Mochi units.
  float GetMochiSaturation() const;
  // Sets the stiffness, damping and saturation of Params from the properties.
  void FillCommonParams(mochi::ConstraintParams& Params) const;
  // Logs an error about this component's constraints.
  void LogError(const FString& Message) const;
  // Runs Fn on each created constraint, on the Mochi thread.
  void ForEachConstraint(std::function<void(mochi::Constraint&)> Fn) const;
  // Resolves Path and gets its Mochi actor.
  EMochiBodyState ResolveBody(
      const FString& Path,
      FMochiConstraintBody& OutBody,
      UMochiSceneObject*& OutScene,
      mochi::ActorHandle& OutActor);

  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
  virtual void TickComponent(
      float DeltaTime,
      ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction) override;

 private:
  void TryCreateConstraints();

  struct FMochiState {
    std::vector<mochi::ConstraintHandle> Constraints;
  };

  TWeakObjectPtr<UMochiSceneObject> MochiScene;
  // Accessed by the Mochi thread only!
  std::shared_ptr<FMochiState> MochiThreadState = std::make_shared<FMochiState>();
  bool bCreatePending = false;
  bool bCreated = false;
};
