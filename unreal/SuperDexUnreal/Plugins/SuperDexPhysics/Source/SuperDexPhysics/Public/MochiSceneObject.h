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
#include "MochiAssets.h"
#include "MochiDebugDraw.h"
#include "MochiPhysicsSettings.h"
#include "UObject/Object.h"

#include "MochiSceneObject.generated.h"

struct FMochiSceneStats {
  mochi::PerformanceStats Performance;
  double LastTimeStepSec = 0;
  int NumActors;
  int NumLayers;
  int NumConstraints;
  int numMochiStepsSinceStart = 0;
  double totalSimulatedTimeSinceStartSec = 0;
};

UENUM(BlueprintType)
enum class EMochiSceneType : uint8 { Async, Sync };

// Owns a mochi::Scene or mochi::AsyncScene and provides wrapper functions to appropriately handle
// calling standard mochi scene functions on them so that our mochi related objects can support both
// types of scene without branching code.
UCLASS()
class SUPERDEXPHYSICS_API UMochiSceneObject : public UObject {
  GENERATED_BODY()
 public:
  // Static creation function for the UMochiSceneObject
  static UMochiSceneObject* CreateScene(
      UObject* Outer,
      EMochiSceneType SceneType,
      const FString& SceneName,
      const FMochiSceneSettings& InSettings,
      bool bStartPaused);

  // Force destroy the underlying Mochi scene and all things in it.
  // This will also happen in BeginDestroy when GC'd.
  void DestroyScene();

  // Returns true if the Mochi scene is valid.
  bool IsValid() const;

  // Queue a command to be executed on the Mochi async scene. Or execute the command immediately if
  // using a synchronous scene. This function is just a wrapper around the Mochi API.
  void QueueOrExecuteCommand(std::function<void(mochi::Scene*)> callback) const;

  // Queue a command to be executed on the Mochi async scene for a specific actor. Or execute the
  // command immediately if using a synchronous scene. This function is just a wrapper around the
  // Mochi API.
  void QueueOrExecuteActorCommand(
      mochi::ActorHandle actor,
      std::function<void(mochi::Actor*)> callback) const;

  // Waits for all queued commands to be executed. This is a blocking call. On an AsyncScene, this
  // will wait for the mochi thread to process all queued commands. On a synchronous scene, this
  // will immediately return. This function is just a wrapper around the Mochi API.
  void WaitForQueuedCommands() const;

  // Register a callback to execute before each simulation step to whatever scene is currently owned
  // by this object. This function is just a wrapper around the Mochi API.
  mochi::CallbackHandle RegisterPreStepCallback(
      std::string_view debugName,
      std::function<void(mochi::StepInfo const&)> fn,
      int priority = mochi::Scene::kDefaultCallbackPriority) const;

  // Register a callback to execute after each simulation step to whatever scene is currently owned
  // by this object. This function is just a wrapper around the Mochi API.
  mochi::CallbackHandle RegisterPostStepCallback(
      std::string_view debugName,
      std::function<void(mochi::StepInfo const&)> fn,
      int priority = mochi::Scene::kDefaultCallbackPriority) const;

  // Cancel a previously registered callback on whatever scene is currently owned by this object.
  // This function is just a wrapper around the Mochi API.
  void CancelCallback(mochi::CallbackHandle handle) const;

  // Set scene gravity vector (default = FVector(0, 0, -9.81)).
  void SetGravity(const FVector& GravityVector) const;
  // Set or change an entry in the scene's contact matrix.
  void SetContactMatrixEntry(const FMochiContactMatrixEntry& Entry) const;
  // (Re)configure settings after the AsyncScene has been created. While the contents are externally
  // owned, only step params and debug draw are applied; gravity, solver params and contact layers
  // belong to the loader.
  void ConfigureSettings(const FMochiSceneSettings& InSettings);

  // True when a runtime loader (a .superdex_scene) owns this scene's contents, settings included.
  // Unreal components then create no actors in it and Unreal settings do not override it.
  bool IsContentsExternallyOwned() const {
    return bContentsExternallyOwned;
  }
  void SetContentsExternallyOwned(bool bExternallyOwned) {
    bContentsExternallyOwned = bExternallyOwned;
  }
  // Get the last configured settings.
  FMochiSceneSettings GetSettings() const;
  // Configure debug drawing for the current Mochi async scene.
  void ConfigureDebugDraw(bool bEnableDebugDraw);
  void ConfigureDebugDraw(bool bEnableDebugDraw, const FMochiDebugDrawFlags& DebugDrawFlags);
  // Returns true if debug draw is enabled
  bool IsDebugDrawEnabled() const;
  // Get the last configured debug draw flags.
  FMochiDebugDrawFlags GetDebugDrawFlags() const;
  // Debug draw the current Mochi scene. Optionally pass a world other than the one this UMochiScene
  // belongs to.
  void DebugDrawScene(UWorld* World = nullptr);
  // Pause or unpause the simulation. When paused, the simulation will not be stepped, but queued
  // commands will still be executed. This will report an error if the scene is not an AsyncScene.
  void SetPaused(bool bPaused) const;
  // Returns true if the scene is paused or invalid. This will report an error if the scene is not
  // an AsyncScene.
  bool IsPaused() const;
  // Step the simulation by exactly one step. Then if an AsyncScene pause (or re-pause). This steps
  // by the scene's FMochiSceneSettings.StepParams.FixedTimeStepSeconds regardless of dynamic time
  // step setting. For finer control of scene stepping, get the scene pointer and use the Mochi API
  // directly.
  void StepPause() const;
  // Start recording data in tthe scene
  void StartRecording(const FString& filePath) const;
  // Stop recording
  void StopRecording() const;

  // Update stats group. Call this one per frame.
  void UpdateStatsGroups();

  // Get the stats value for NumMochiStepsSinceStart. This represents a incrementing value each time
  // the PostStepCallback is invoked from the mochi thread.
  UFUNCTION(BlueprintPure, Category = "Mochi")
  int GetNumMochiStepsSinceStart() const;

  // Get the stats value for TotalSimulatedTimeSinceStartSec. This represents the cumulative sum of
  // all simulatedTimeStepSec on the mochi thread. This value is updated once per PostStepCallback
  // and will remain constant until mochi finishes its next step.
  UFUNCTION(BlueprintPure, Category = "Mochi")
  double GetTotalSimulatedTimeSinceStartSec() const;

  // Helper to retrieve step duration stat for in-game display
  UFUNCTION(BlueprintPure, Category = "Mochi")
  double GetTotalStepDuration() const;

 protected:
  // Ensures Mochi async scene is destroyed when this UMochiSceneObject is GC.
  virtual void BeginDestroy() override;

  // Apply the contact matrix, solver params and gravity from settings. Skipped while the contents
  // are externally owned.
  void ConfigureSceneContentSettings(const FMochiSceneSettings& InSettings);

  // Initializes contact layers for our common use cases.
  void InitDefaultContactLayers();
  void RegisterDebugDrawAndStatsCallbacks();

 protected:
  // When an IKSolver is created it gains ownership of the scene and will itself call
  // mochiContext->DestroyScene() at the appropriate time. The MochiSceneObject should not call
  // DestroyScene() during cleanup in this case to avoid deleting the scene out from under the
  // owning IKSolver.
  friend class USuperDexIKSolverObject;
  bool bDestructionOwnedByFriend = false;

 private:
  FMochiSceneSettings Settings;
  bool bContentsExternallyOwned = false;
  mochi::AsyncScene* MochiAsyncScene = nullptr;
  mochi::Scene* MochiScene = nullptr;
  FMochiSceneStats Stats;
  mutable FRWLock StatsLock;

  struct DebugDrawLineVerticesBuffer {
    std::vector<mochi::Real3> positions;
    std::vector<mochi::Color> colors;
  };

  struct DebugDrawSpheresBuffer {
    std::vector<mochi::Real3> positions;
    std::vector<mochi::real> radii;
    std::vector<mochi::Color> colors;
  };

  struct DebugDrawData {
    DebugDrawLineVerticesBuffer lineVerts;
    DebugDrawSpheresBuffer spheres;
    bool hasChanged = false;
  };

  FRWLock debugDrawDataMutex;
  DebugDrawData debugDrawDataMochi;
  DebugDrawData debugDrawDataPending;
  DebugDrawData debugDrawDataRender;
};
