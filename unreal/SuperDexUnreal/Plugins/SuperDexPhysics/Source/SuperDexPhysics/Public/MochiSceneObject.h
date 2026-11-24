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

/** Native scene performance data and accumulated simulation counters. */
struct FMochiSceneStats {
  /** Performance statistics collected from the native scene. */
  mochi::PerformanceStats Performance;

  /** Most recently reported native simulation time step. */
  double LastTimeStepSec = 0;

  /** Actor count reported by the native scene. */
  int NumActors;

  /** Contact-layer count reported by the native scene. */
  int NumLayers;

  /** Constraint count reported by the native scene. */
  int NumConstraints;

  /** Native steps accumulated since scene startup. */
  int NumMochiStepsSinceStart = 0;

  /** Simulated time accumulated since scene startup. */
  double TotalSimulatedTimeSinceStartSec = 0;
};

/** Native scene execution mode. */
UENUM(BlueprintType)
enum class EMochiSceneType : uint8 { Async, Sync };

/**
 * Owns a mochi::Scene or mochi::AsyncScene and provides wrapper functions to appropriately handle
 * calling standard mochi scene functions on them so that our mochi related objects can support both
 * types of scene without branching code.
 */
UCLASS()
class SUPERDEXPHYSICS_API UMochiSceneObject : public UObject {
  GENERATED_BODY()
 public:
  /**
   * Static creation function for the UMochiSceneObject
   *
   * @param Outer Unreal outer for the created wrapper object.
   * @param SceneType Native scene execution mode to create.
   * @param SceneName Name to assign to the created scene.
   * @param InSettings Scene settings to apply.
   * @param bStartPaused Whether to create the scene with stepping paused.
   * @return The object returned by creation.
   */
  static UMochiSceneObject* CreateScene(
      UObject* Outer,
      EMochiSceneType SceneType,
      const FString& SceneName,
      const FMochiSceneSettings& InSettings,
      bool bStartPaused);

  /**
   * Force destroy the underlying Mochi scene and all things in it.
   * This will also happen in BeginDestroy when GC'd.
   */
  void DestroyScene();

  /**
   * Returns true if the Mochi scene is valid.
   *
   * @return Whether a synchronous or asynchronous native scene pointer is assigned.
   */
  bool IsValid() const;

  /**
   * Queue a command to be executed on the Mochi async scene. Or execute the command immediately if
   * using a synchronous scene. This function is just a wrapper around the Mochi API.
   *
   * @param Callback Callback supplied to the query or command.
   */
  void QueueOrExecuteCommand(std::function<void(mochi::Scene*)> Callback) const;

  /**
   * Queue a command to be executed on the Mochi async scene for a specific actor. Or execute the
   * command immediately if using a synchronous scene. This function is just a wrapper around the
   * Mochi API.
   *
   * @param Actor Handle of the actor supplied to the command.
   * @param Callback Callback supplied to the query or command.
   */
  void QueueOrExecuteActorCommand(
      mochi::ActorHandle Actor,
      std::function<void(mochi::Actor*)> Callback) const;

  /**
   * Waits for all queued commands to be executed. This is a blocking call. On an AsyncScene, this
   * will wait for the mochi thread to process all queued commands. On a synchronous scene, this
   * will immediately return. This function is just a wrapper around the Mochi API.
   */
  void WaitForQueuedCommands() const;

  /**
   * Register a callback to execute before each simulation step to whatever scene is currently owned
   * by this object. This function is just a wrapper around the Mochi API.
   *
   * @param DebugName Debug label identifying the callback.
   * @param Fn Callback invoked at the native step boundary.
   * @param Priority Native callback ordering priority.
   * @return Native callback handle used for cancellation.
   */
  mochi::CallbackHandle RegisterPreStepCallback(
      std::string_view DebugName,
      std::function<void(mochi::StepInfo const&)> Fn,
      int Priority = mochi::Scene::kDefaultCallbackPriority) const;

  /**
   * Register a callback to execute after each simulation step to whatever scene is currently owned
   * by this object. This function is just a wrapper around the Mochi API.
   *
   * @param DebugName Debug label identifying the callback.
   * @param Fn Callback invoked at the native step boundary.
   * @param Priority Native callback ordering priority.
   * @return Native callback handle used for cancellation.
   */
  mochi::CallbackHandle RegisterPostStepCallback(
      std::string_view DebugName,
      std::function<void(mochi::StepInfo const&)> Fn,
      int Priority = mochi::Scene::kDefaultCallbackPriority) const;

  /**
   * Cancel a previously registered callback on whatever scene is currently owned by this object.
   * This function is just a wrapper around the Mochi API.
   *
   * @param Handle Native handle supplied to the operation.
   */
  void CancelCallback(mochi::CallbackHandle Handle) const;

  /**
   * Set scene gravity vector [m/s^2].
   *
   * @param GravityVector Gravity vector to apply to the scene.
   */
  void SetGravity(const FVector& GravityVector) const;

  /**
   * Set or change an entry in the scene's contact matrix.
   *
   * @param Entry Layer-to-layer contact rule.
   */
  void SetContactMatrixEntry(const FMochiContactMatrixEntry& Entry) const;

  /**
   * (Re)configure settings after the AsyncScene has been created. While the contents are externally
   * owned, only step params and debug draw are applied; gravity, solver params and contact layers
   * belong to the loader.
   *
   * @param InSettings Scene settings to apply.
   */
  void ConfigureSettings(const FMochiSceneSettings& InSettings);

  /**
   * True when a runtime loader (a .superdex_scene) owns this scene's contents, settings included.
   * Unreal components then create no actors in it and Unreal settings do not override it.
   *
   * @return Whether native scene contents are managed externally.
   */
  bool IsContentsExternallyOwned() const {
    return bContentsExternallyOwned;
  }

  /**
   * Set whether native scene contents are managed by an external owner.
   *
   * @param bExternallyOwned Whether scene contents/settings are managed by an external owner.
   */
  void SetContentsExternallyOwned(bool bExternallyOwned) {
    bContentsExternallyOwned = bExternallyOwned;
  }

  /**
   * Get the last configured settings.
   *
   * @return The settings.
   */
  FMochiSceneSettings GetSettings() const;

  /**
   * Configure debug drawing for the current Mochi async scene.
   *
   * @param bEnableDebugDraw Requested debug-draw enable state.
   */
  void ConfigureDebugDraw(bool bEnableDebugDraw);

  /**
   * Configure debug drawing and its entity flags.
   *
   * @param bEnableDebugDraw Requested debug-draw enable state.
   * @param DebugDrawFlags Native debug-draw entities to select.
   */
  void ConfigureDebugDraw(bool bEnableDebugDraw, const FMochiDebugDrawFlags& DebugDrawFlags);

  /**
   * Returns true if debug draw is enabled
   *
   * @return The is debug draw enabled state.
   */
  bool IsDebugDrawEnabled() const;

  /**
   * Get the last configured debug draw flags.
   *
   * @return The debug draw flags.
   */
  FMochiDebugDrawFlags GetDebugDrawFlags() const;

  /**
   * Debug draw the current Mochi scene. Optionally pass a world other than the one this UMochiScene
   * belongs to.
   *
   * @param World World in which debug geometry is drawn.
   */
  void DebugDrawScene(UWorld* World = nullptr);

  /**
   * Pause or unpause the simulation. When paused, the simulation will not be stepped, but queued
   * commands will still be executed. This will report an error if the scene is not an AsyncScene.
   *
   * @param bPaused Requested pause state.
   */
  void SetPaused(bool bPaused) const;

  /**
   * Returns true if the scene is paused or invalid. This will report an error if the scene is not
   * an AsyncScene.
   *
   * @return Pause state of the native scene.
   */
  bool IsPaused() const;

  /**
   * Step the simulation by exactly one step. Then if an AsyncScene pause (or re-pause). This steps
   * by the scene's FMochiSceneSettings.StepParams.FixedTimeStepSeconds regardless of dynamic time
   * step setting. For finer control of scene stepping, get the scene pointer and use the Mochi API
   * directly.
   */
  void StepPause() const;

  /**
   * Start recording data in tthe scene
   *
   * @param FilePath Directory in which to store the recording or exported scene.
   */
  void StartRecording(const FString& FilePath) const;

  /** Stop recording */
  void StopRecording() const;

  /** Update stats group. Call this one per frame. */
  void UpdateStatsGroups();

  /**
   * Get the stats value for NumMochiStepsSinceStart. This represents a incrementing value each time
   * the PostStepCallback is invoked from the mochi thread.
   *
   * @return Number of native scene steps accumulated since startup.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi")
  int GetNumMochiStepsSinceStart() const;

  /**
   * Get the stats value for TotalSimulatedTimeSinceStartSec. This represents the cumulative sum of
   * all simulatedTimeStepSec on the mochi thread. This value is updated once per PostStepCallback
   * and will remain constant until mochi finishes its next step.
   *
   * @return Accumulated simulated time in seconds.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi")
  double GetTotalSimulatedTimeSinceStartSec() const;

  /**
   * Helper to retrieve step duration stat for in-game display
   *
   * @return Step duration reported by the scene statistics.
   */
  UFUNCTION(BlueprintPure, Category = "Mochi")
  double GetTotalStepDuration() const;

 protected:
  /** Ensures Mochi async scene is destroyed when this UMochiSceneObject is GC. */
  virtual void BeginDestroy() override;

  /**
   * Apply the contact matrix, solver params and gravity from settings. Skipped while the contents
   * are externally owned.
   */
  void ConfigureSceneContentSettings(const FMochiSceneSettings& InSettings);

  /** Initializes contact layers for our common use cases. */
  void InitDefaultContactLayers();
  void RegisterDebugDrawAndStatsCallbacks();

 protected:
  /**
   * When an IKSolver is created it gains ownership of the scene and will itself call
   * mochiContext->DestroyScene() at the appropriate time. The MochiSceneObject should not call
   * DestroyScene() during cleanup in this case to avoid deleting the scene out from under the
   * owning IKSolver.
   */
  friend class USuperDexIKSolverObject;
  bool bDestructionOwnedByFriend = false;

 private:
  FMochiSceneSettings Settings;
  bool bContentsExternallyOwned = false;
  mochi::AsyncScene* MochiAsyncScene = nullptr;
  mochi::Scene* MochiScene = nullptr;
  FMochiSceneStats Stats;
  mutable FRWLock StatsLock;

  struct FDebugDrawLineVerticesBuffer {
    std::vector<mochi::Real3> Positions;
    std::vector<mochi::Color> Colors;
  };

  struct FDebugDrawSpheresBuffer {
    std::vector<mochi::Real3> Positions;
    std::vector<mochi::real> Radii;
    std::vector<mochi::Color> Colors;
  };

  struct FDebugDrawData {
    FDebugDrawLineVerticesBuffer LineVerts;
    FDebugDrawSpheresBuffer Spheres;
    bool bHasChanged = false;
  };

  FRWLock DebugDrawDataMutex;
  FDebugDrawData DebugDrawDataMochi;
  FDebugDrawData DebugDrawDataPending;
  FDebugDrawData DebugDrawDataRender;
};
