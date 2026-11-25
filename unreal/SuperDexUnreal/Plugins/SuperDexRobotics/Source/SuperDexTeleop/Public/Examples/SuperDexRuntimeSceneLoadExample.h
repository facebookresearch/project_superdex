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
#include "Engine/EngineTypes.h"
#include "Framework/Commands/InputChord.h"
#include "GameFramework/Actor.h"
#include "SuperDexHandTypes.h"

#include "SuperDexRuntimeSceneLoadExample.generated.h"

class ATeleopRoboperatorActor;
class USuperDexBotSceneVisualizationComponent;
struct FSuperDexRuntimeSceneLoadExampleState;

/** How the example treats one of the scene's slots. */
USTRUCT(BlueprintType)
struct SUPERDEXTELEOP_API FSuperDexRuntimeSceneLoadExampleSlot {
  GENERATED_BODY()

  /** The slot's name in the scene's `slots`. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex Robotics|Runtime Scene Load")
  FString Name;

  /** The hand that drives the slot's roboperator. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex Robotics|Runtime Scene Load")
  ESuperDexHandedness Handedness = ESuperDexHandedness::Left;

  /** Switches the slot to its next bot. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex Robotics|Runtime Scene Load")
  FInputChord CycleBotKey;
};

/**
 * EXAMPLE: teleoperate the bots of a `.superdex_scene` loaded at runtime.
 *
 * Place one in an otherwise empty level. It loads the scene (USuperDexBotSceneSubsystem), draws it,
 * and builds an episode:
 *  - Each of the scene's slots brings up its selected bot, at first the slot's first one; the other
 *    bots of the slot are removed. Bots in no slot are always up.
 *  - A selected bot gets the roboperator in Roboperators whose class name is the bot's
 *    `roboperatorClass`, spawned at the bot's base and posing the scene bot.
 *  - Before every physics step, the bot's scene controllers are driven from what its roboperator's
 *    TeleopNode last published, through their generic float inputs:
 *      - a `Pose` controller takes the arm pose target;
 *      - a `JointPositions` controller takes the hand joint targets on the bot's DOFs after its
 *        first NumArmJoints, and the arm joint targets on those first DOFs when no pose controller
 *        drives them; it holds its other DOFs;
 *      - a `MOCHI_ARTICULATED_POSE` controller follows the roboperator's tracked-hand poseable.
 *    Until a target arrives, a controller holds the bot where the episode started. Others are not
 *    driven. The controllers' efforts are summed.
 *  - The current task's objects are spawned.
 *
 * Win64 builds record episodes with SuperDexRecorder when bRecord or -SuperDexRecord is set. Each
 * recording references the loaded scene and task files by name and content hash.
 *
 * Command line: -SuperDexSceneFile=, -SuperDexTaskFile=, -SuperDexTaskFolder=, -SuperDexRecord,
 * -SuperDexRecordDir=, -SuperDexRecordContacts.
 */
UCLASS(meta = (PrioritizeCategories = "SuperDex Robotics"))
class SUPERDEXTELEOP_API ASuperDexRuntimeSceneLoadExample : public AActor {
  GENERATED_BODY()

 public:
  ASuperDexRuntimeSceneLoadExample();

  /** The scene; relative to the project directory. Override: -SuperDexSceneFile=. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex Robotics|Runtime Scene Load")
  FFilePath SceneFile;

  /**
   * A `.superdex_task`, or a folder whose tasks play in name order; relative to the project
   * directory. Override: -SuperDexTaskFolder= or -SuperDexTaskFile=.
   */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex Robotics|Runtime Scene Load")
  FFilePath TaskFile;

  /** The roboperators the scene's bots name in their `roboperatorClass`. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex Robotics|Runtime Scene Load")
  TArray<TSubclassOf<ATeleopRoboperatorActor>> Roboperators;

  /** How the example treats each of the scene's slots, by name. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex Robotics|Runtime Scene Load")
  TArray<FSuperDexRuntimeSceneLoadExampleSlot> Slots;

  /** Record episodes (Win64). Override: -SuperDexRecord. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex Robotics|Runtime Scene Load")
  bool bRecord = false;

  /** Empty means <Project>/Saved/SuperDexRecorder. Override: -SuperDexRecordDir=. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex Robotics|Runtime Scene Load")
  FDirectoryPath RecordingDirectory;

  /** Override: -SuperDexRecordContacts. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex Robotics|Runtime Scene Load")
  bool bRecordContactForces = false;

  /** Steps to the previous task. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex Robotics|Runtime Scene Load")
  FInputChord PreviousTaskKey;

  /** Steps to the next task. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex Robotics|Runtime Scene Load")
  FInputChord NextTaskKey;

  /** Puts the task's objects back where the task places them. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex Robotics|Runtime Scene Load")
  FInputChord ResetTaskKey;

  /** Starts recording an episode, or keeps the one recording and starts the next. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex Robotics|Runtime Scene Load")
  FInputChord RecordOrKeepKey;

  /** Discards the episode recording and starts the next. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex Robotics|Runtime Scene Load")
  FInputChord DiscardKey;

  /** Switch a slot to its next bot and rebuild the episode. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Robotics|Runtime Scene Load")
  void CycleSlotBot(FString SlotName);

  /** Step to the previous task and spawn its objects. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Robotics|Runtime Scene Load")
  void PreviousTask();

  /** Step to the next task and spawn its objects. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Robotics|Runtime Scene Load")
  void NextTask();

  /** Put the task's objects back where the task places them. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Robotics|Runtime Scene Load")
  void ResetTask();

  /** Start recording an episode, or keep the one recording and start the next. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Robotics|Runtime Scene Load")
  void RecordOrKeep();

  /** Discard the episode recording and start the next. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex Robotics|Runtime Scene Load")
  void Discard();

  /** Whether an episode is being recorded. */
  UFUNCTION(BlueprintPure, Category = "SuperDex Robotics|Runtime Scene Load")
  bool IsRecording() const;

  /** The current task's file name, without its extension. */
  UFUNCTION(BlueprintPure, Category = "SuperDex Robotics|Runtime Scene Load")
  FString GetTaskName() const;

  virtual void PostInitializeComponents() override;
  virtual void Tick(float DeltaSeconds) override;

 protected:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

 private:
  void BindKeys();
  void BeginEpisode();
  void EndEpisode();
  void RestartEpisode();
  void SpawnTaskObjects();
  void StopRecording(bool bKeep);

  /** Draws the scene. */
  UPROPERTY(VisibleAnywhere, Category = "SuperDex Robotics|Runtime Scene Load")
  TObjectPtr<USuperDexBotSceneVisualizationComponent> Visualization;

  UPROPERTY(Transient)
  TArray<TObjectPtr<ATeleopRoboperatorActor>> SpawnedRoboperators;

  TSharedPtr<FSuperDexRuntimeSceneLoadExampleState> State;
  TArray<FString> TaskFiles;
  int32 TaskIndex = 0;
  /** Each slot's selected bot, an index into the slot's `controlledBots`. */
  TMap<FString, int32> SlotSelection;
};
