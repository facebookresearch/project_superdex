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

/** Native shape and collider mapping associated with a model asset. */
struct FMochiModelMapping {
  /** Model asset associated with the native shape mapping. */
  const UMochiModel* MochiModel = nullptr;
  /** Collider representation used for uniform model scale. */
  EMochiColliderType ColliderUniformScale = EMochiColliderType::None;
  /** Collider representation used for non-uniform model scale. */
  EMochiColliderType ColliderNonUniformScale = EMochiColliderType::None;
  /**
   * Get whether the model mapping contains a model-asset pointer.
   *
   * @return Whether the model-asset pointer is non-null.
   */
  bool IsValid() const {
    return MochiModel != nullptr;
  }
  /**
   * Choose the collider representation compatible with the requested model scale.
   *
   * @param Scale3D Scale applied to the geometry.
   * @return Collider representation selected for uniform or non-uniform scale.
   */
  EMochiColliderType GetColliderTypeForScale3D(const FVector& Scale3D) const {
    return Scale3D.IsUniform() ? ColliderUniformScale : ColliderNonUniformScale;
  }
};

/** Editor preferences retained while play-session overrides are active. */
struct FMochiEditorPreferencesBeforeOverride {
  /** Editor background-throttling preference retained before the play-session override. */
  bool bThrottleCPUWhenNotForeground = true;
};

/** Game-instance subsystem managing the physics scene and model mappings. */
UCLASS()
class SUPERDEXPHYSICS_API UMochiGameInstanceSubsystem final : public UGameInstanceSubsystem,
                                                              public FTickableGameObject {
  GENERATED_BODY()
 public:
  /** Initialize the game-instance physics subsystem. */
  UMochiGameInstanceSubsystem();

  /**
   * Get the game instance scene.
   *
   * @return The scene.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi")
  UMochiSceneObject* GetScene() const;

  /** Enable Mochi debug draw for the managed scene. */
  void EnableDebugDraw() const;
  /** Disable Mochi debug draw for the managed scene. */
  void DisableDebugDraw() const;
  /**
   * Returns true if Mochi debug draw is enabled for the manged scene.
   *
   * @return The is debug draw enabled state.
   */
  bool IsDebugDrawEnabled() const;

  /**
   * UGameInstanceSubsystem Implementation
   *
   * @param Outer Unreal outer for the created wrapper object.
   * @return Whether this game instance should create the physics subsystem.
   */
  virtual bool ShouldCreateSubsystem(UObject* Outer) const override;
  virtual void Initialize(FSubsystemCollectionBase& Collection) override;
  virtual void Deinitialize() override;

  /**
   * FTickableGameObject Implementation
   *
   * @return The tickable game object world.
   */
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
  /** Start the subsystem scene paused */
  bool bStartPaused = false;

  /**
   * True when the scene's contents are owned by a runtime loader (a .superdex_scene populated by
   * USuperDexBotSceneSubsystem) rather than by the physics components placed in the level. The
   * flag lives on the scene object (UMochiSceneObject::IsContentsExternallyOwned), which also
   * refuses new actors from Unreal components and keeps Unreal settings from overriding the
   * loader's; these accessors exist because SuperDexPhysics cannot see SuperDexRobotics where the
   * loader lives.
   *
   * @return Whether scene contents/settings are managed externally.
   */
  bool IsSceneExternallyOwned() const {
    return Scene && Scene->IsContentsExternallyOwned();
  }
  /**
   * Set whether scene contents/settings are managed by an external owner.
   *
   * @param bExternallyOwned Whether scene contents/settings are managed by an external owner.
   */
  void SetSceneExternallyOwned(bool bExternallyOwned) {
    if (Scene) {
      Scene->SetContentsExternallyOwned(bExternallyOwned);
    }
  }

 private:
  /** Mochi game instance scene. */
  UPROPERTY()
  UMochiSceneObject* Scene = nullptr;
  bool bSceneRunning = false;
  FMochiSceneSettings RetrieveMochiSceneSettings() const;

  FMochiEditorPreferencesBeforeOverride EditorPreferencesBeforeOverride = {};
};
