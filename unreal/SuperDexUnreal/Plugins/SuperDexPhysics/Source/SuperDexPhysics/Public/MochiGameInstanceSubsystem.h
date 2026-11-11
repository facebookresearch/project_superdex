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

#include "MochiModel.h"
#include "MochiPhysicsSettings.h"
#include "MochiSceneObject.h"

#include "Components/ActorComponent.h"
#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "Tickable.h"

#include "MochiGameInstanceSubsystem.generated.h"

struct FMochiModelMapping {
  const UMochiModel* MochiModel = nullptr;
  EMochiColliderType ColliderUniformScale = EMochiColliderType::None;
  EMochiColliderType ColliderNonUniformScale = EMochiColliderType::None;
  bool IsValid() const {
    return MochiModel != nullptr;
  }
  EMochiColliderType GetColliderTypeForScale3D(const FVector& Scale3D) const {
    return Scale3D.IsUniform() ? ColliderUniformScale : ColliderNonUniformScale;
  }
};

struct FMochiEditorPreferencesBeforeOverride {
  bool bThrottleCPUWhenNotForeground = true;
};

UCLASS()
class SUPERDEXPHYSICS_API UMochiGameInstanceSubsystem final : public UGameInstanceSubsystem,
                                                              public FTickableGameObject {
  GENERATED_BODY()
 public:
  UMochiGameInstanceSubsystem();

  // Get the game instance scene.
  UFUNCTION(BlueprintPure, Category = "Mochi")
  UMochiSceneObject* GetScene() const;

  // Enable Mochi debug draw for the managed scene.
  void EnableDebugDraw() const;
  // Disable Mochi debug draw for the managed scene.
  void DisableDebugDraw() const;
  // Returns true if Mochi debug draw is enabled for the manged scene.
  bool IsDebugDrawEnabled() const;

  // UGameInstanceSubsystem Implementation
  virtual bool ShouldCreateSubsystem(UObject* Outer) const override;
  virtual void Initialize(FSubsystemCollectionBase& Collection) override;
  virtual void Deinitialize() override;

  // FTickableGameObject Implementation
  virtual UWorld* GetTickableGameObjectWorld() const override;
  virtual ETickableTickType GetTickableTickType() const override;
  virtual bool IsAllowedToTick() const override final;
  virtual void Tick(float DeltaTime) override;
  virtual TStatId GetStatId() const override;

 private:
  void CreateScene();
  void DestroyScene();

  void StoreAndSetEditorPreferencesOverrides();
  void RestoreEditorPreferencesOverrides() const;

 public:
  // Start the subsystem scene paused
  bool bStartPaused = false;

  /**
   * True when the scene's contents are owned by a runtime loader (a .superdex_scene populated by
   * USuperDexBotSceneSubsystem) rather than by the physics components placed in the level. The
   * flag lives on the scene object (UMochiSceneObject::IsContentsExternallyOwned), which also
   * refuses new actors from Unreal components and keeps Unreal settings from overriding the
   * loader's; these accessors exist because SuperDexPhysics cannot see SuperDexRobotics where the
   * loader lives.
   */
  bool IsSceneExternallyOwned() const {
    return Scene && Scene->IsContentsExternallyOwned();
  }
  void SetSceneExternallyOwned(bool bExternallyOwned) {
    if (Scene) {
      Scene->SetContentsExternallyOwned(bExternallyOwned);
    }
  }

 private:
  // Mochi game instance scene.
  UPROPERTY()
  UMochiSceneObject* Scene = nullptr;
  bool bSceneRunning = false;
  FMochiSceneSettings RetrieveMochiSceneSettings() const;

  FMochiEditorPreferencesBeforeOverride EditorPreferencesBeforeOverride = {};
};
