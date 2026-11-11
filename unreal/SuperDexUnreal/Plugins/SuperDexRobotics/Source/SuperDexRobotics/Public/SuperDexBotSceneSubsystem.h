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

#include <mochi_core/utils/transform_rt.h>
#include <mochi_physics/cpp_api/mochi_handle.h>
#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"

#include "SuperDexBotSceneSubsystem.generated.h"

namespace superdex::robotics {
class BotScene;
struct BotTaskPrefab;
} // namespace superdex::robotics

UENUM(BlueprintType)
enum class ESuperDexBotSceneState : uint8 {
  Disabled UMETA(DisplayName = "Disabled"),
  Loaded UMETA(DisplayName = "Loaded"),
  Failed UMETA(DisplayName = "Failed"),
};

/** Broadcast on the game thread when the scene transitions Disabled -> Loaded. */
DECLARE_MULTICAST_DELEGATE(FOnBotSceneLoaded);

/** Broadcast on the game thread after task instances or bots are added or removed. */
DECLARE_MULTICAST_DELEGATE(FOnBotSceneContentsChanged);

/**
 * Runtime owner of a .superdex_scene / .superdex_scene_archive and its optional
 * .superdex_task, loaded into the scene owned by UMochiGameInstanceSubsystem.
 *
 * This subsystem is the single load point, and the loaded file is the sole authority over the
 * Mochi scene: its actors, gravity, solver params and contact layers. Nothing may populate the
 * scene before it loads, so both paths load before any actor BeginPlay. Initialize() resolves the
 * command-line path; for editor iteration, a level actor supplies a fallback path via
 * EnsureLoaded() from PostInitializeComponents. EnsureLoaded() is idempotent, and consumers that
 * miss the transition are notified through OnSceneLoaded(). Nothing polls.
 *
 * The scene's bots are declared but not initialized at load. A scene may offer several bots for
 * one slot and only the consumer knows which it drives, so the consumer initializes the bots it
 * uses with BotScene::InitializeBot (and removes them with RemoveBot), then calls
 * NotifySceneContentsChanged().
 *
 * State machine:
 *   Disabled - no scene requested (yet).
 *   Loaded   - the archive populated the authoritative Mochi scene.
 *   Failed   - a scene was requested but could not be loaded. Terminal: callers fail closed
 *              rather than silently falling back to level-authored physics.
 */
UCLASS()
class SUPERDEXROBOTICS_API USuperDexBotSceneSubsystem : public UGameInstanceSubsystem {
  GENERATED_BODY()

 public:
  USuperDexBotSceneSubsystem();
  virtual ~USuperDexBotSceneSubsystem();

  virtual void Initialize(FSubsystemCollectionBase& Collection) override;
  virtual void Deinitialize() override;

  ESuperDexBotSceneState GetState() const;
  FString GetScenePath() const;
  FString GetTaskPath() const;
  FString GetErrorMessage() const;

  /** Error message, or a generic one when the loader supplied none. */
  FString GetErrorMessageOrDefault() const;

  /** True when callers should suppress level-authored physics: the scene owns it. */
  bool IsSceneMode() const;

  /**
   * Load the scene if it is not loaded already, using the supplied paths when the command line
   * did not provide them. Idempotent: a no-op once the state leaves Disabled. Call it before any
   * actor BeginPlay (e.g. from PostInitializeComponents); a load into a scene that already holds
   * actors fails. Empty fallbacks with no command line leave the state Disabled.
   *
   * @return true when the state is Loaded on return.
   */
  bool EnsureLoaded(const FString& FallbackScenePath, const FString& FallbackTaskPath);

  FOnBotSceneLoaded& OnSceneLoaded() {
    return SceneLoadedDelegate;
  }

  FOnBotSceneContentsChanged& OnSceneContentsChanged() {
    return SceneContentsChangedDelegate;
  }

  /**
   * Borrowed pointer to the loaded BotScene, valid only while GetState() == Loaded. Owned by this
   * subsystem; do not retain across a Deinitialize or a Failed transition. May be nullptr.
   */
  const superdex::robotics::BotScene* GetBotScene() const;
  superdex::robotics::BotScene* GetBotScene();

  /**
   * Borrowed pointer to the parsed .superdex_task, or nullptr when none was supplied. The task
   * declares which spawnable prefabs an episode places and where; this subsystem parses and holds
   * it but does not apply it, because placement needs domain randomization that lives in the
   * project module.
   */
  const superdex::robotics::BotTaskPrefab* GetBotTask() const;

  /**
   * Instantiate one task spawn into the loaded scene, marshalled onto the Mochi thread.
   * InstanceName is the identity the recorder and dataset key off (see SpawnEntry::name).
   *
   * @return true on success, with OutActors holding the created actor handles.
   */
  bool SpawnTaskInstance(
      const FString& InstanceName,
      const FString& PrefabName,
      const mochi::TransformRT& WorldFromInstance,
      TArray<mochi::ActorHandle>& OutActors,
      FString& OutError);

  /** Destroy every live task instance, marshalled onto the Mochi thread. */
  void DestroyAllTaskInstances();

  /**
   * Load the next task of the `-SuperDexTaskFolder=` playlist, wrapping past the end. The scene,
   * its bots and their controllers are left untouched: only the parsed task changes, so a caller
   * can step through a folder of tasks against one loaded scene.
   *
   * @return Whether a new task was parsed. False when no folder was supplied, when it holds a
   * single task, or when the next file failed to parse. A parse failure leaves the current task
   * live and still moves the cursor, so calling again steps past the bad file instead of retrying
   * it -- and, unlike the initial load, does not fail the scene.
   */
  bool AdvanceToNextTask();

  /**
   * Announce that the set of scene contents has changed. Spawning does not broadcast on its own:
   * a task places several instances in a row, and consumers such as the visualization rebuild
   * their whole mirror per notification. The caller signals once when the batch is complete.
   */
  void NotifySceneContentsChanged();

 private:
  struct Impl;
  Impl* Pimpl = nullptr;

  /**
   * Shared body of every load path. Marks the Mochi scene externally owned, fails if it already
   * holds actors, then populates it on the Mochi thread and waits, sets State and (on success)
   * broadcasts OnSceneLoaded. The scene stays externally owned after a failure.
   */
  bool LoadScene(const FString& InScenePath, const FString& InTaskPath);

  /** Parse the .superdex_task at TaskPath into Pimpl. Failure fails the whole load. */
  bool LoadTask(const FString& InTaskPath);

  /**
   * Parse a .superdex_task into Pimpl->BotTask, reporting the failure rather than acting on it.
   * Split out of LoadTask because the two callers need opposite failure handling: an unparseable
   * task at load time fails the scene, while one reached by stepping through a playlist must not.
   */
  bool ParseTaskFile(const FString& InTaskPath, FString& OutError);

  /**
   * Populate TaskPlaylist from the bot task files (.superdex_task) directly in FolderPath, sorted
   * by name so the order is the same on every machine. Returns false, with OutError set, when the
   * folder holds none.
   */
  bool BuildTaskPlaylist(const FString& FolderPath, FString& OutError);

  /**
   * Resolve the task to load alongside the scene, building the playlist when
   * `-SuperDexTaskFolder=` is set. The folder wins over a `-SuperDexTaskFile=` naming one task.
   */
  bool ResolveInitialTask(const FString& FallbackTaskPath, FString& OutTaskPath, FString& OutError);

  void SetFailed(const FString& InError);

  FString ScenePath;
  FString TaskPath;

  /** Task files from `-SuperDexTaskFolder=`, sorted; empty unless that switch was supplied. */
  TArray<FString> TaskPlaylist;
  int32 TaskCursor = INDEX_NONE;
  ESuperDexBotSceneState State = ESuperDexBotSceneState::Disabled;
  FString ErrorMessage;

  FOnBotSceneLoaded SceneLoadedDelegate;
  FOnBotSceneContentsChanged SceneContentsChangedDelegate;
};
