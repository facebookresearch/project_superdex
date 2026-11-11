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
#include "GameFramework/Actor.h"
// The complete type, because the generated code destroys Recorder.
#include "SuperDexSceneRecorder.h"
#include "Templates/UniquePtr.h"

#include "SuperDexRecorder.generated.h"

class USuperDexRecordingSourceComponent;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FSuperDexRecordingStarted, const FString&, EpisodeId);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(
    FSuperDexRecordingCommitted,
    const FString&,
    OutputPath);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FSuperDexRecordingAborted, const FString&, Reason);

/*
 * Records SuperDex episodes from Blueprint: the Mochi scene and its robots, plus the streams of
 * USuperDexRecordingSourceComponents on this actor and on SourceActors, plus the events and scalars
 * raised through RecordEvent, RecordScalar and RecordScalars. Each episode is one RRD file; record
 * study blocks as consecutive episodes.
 *
 * Place one per level. A second recorder in a world would record everything again, so only the
 * first to initialize records: the others log an error and never record. GetActiveRecorder finds
 * the recording one.
 *
 * Starts automatically by default, once the Mochi scene is ready, every actor in
 * WaitForActorsBeforeStart has begun play, and bPreventAutomaticStart is false. Commits when play
 * ends, unless bCommitOnEndPlay is false.
 */
UCLASS(Blueprintable, ClassGroup = SuperDex, meta = (DisplayName = "SuperDex Recorder"))
class SUPERDEXRECORDER_API ASuperDexRecorder : public AActor {
  GENERATED_BODY()

 public:
  ASuperDexRecorder();

  /* Where episodes are written. Empty uses <Project>/Saved/SuperDexRecorder. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex|Recording")
  FDirectoryPath OutputDirectory;

  /* The scene archive the episode references, so the scene can be rebuilt offline. Ignored with
   * bExportLiveScene. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex|Recording")
  FFilePath SceneArchive;

  /*
   * Export the live Mochi scene, as it is when the first episode starts, to an archive in
   * OutputDirectory's "scenes" folder, and reference it instead of SceneArchive. Later episodes
   * reuse it; call ExportLiveSceneAtNextEpisode after adding or removing physics actors or bots.
   * The simulation pauses while it exports. Only editor builds (PIE included) with bot-scene
   * support can export; packaged builds record without a scene archive.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex|Recording")
  bool bExportLiveScene = false;

  /* Refuse to record without a scene archive (SceneArchive, or the live scene's). */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex|Recording")
  bool bRequireSceneArchive = false;

  /* Stored with each episode, e.g. the participant, condition or task. Set them before each
   * StartRecording. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex|Recording")
  TMap<FString, FString> Properties;

  /* Records Mochi's contact forces, at a cost per contact point in every step. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex|Recording")
  bool bRecordContactForces = false;

  /* Other actors whose USuperDexRecordingSourceComponents this recorder uses. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex|Recording")
  TArray<TObjectPtr<AActor>> SourceActors;

  /* Start the first episode on its own once the scene and WaitForActorsBeforeStart are ready. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex|Recording")
  bool bStartAutomatically = true;

  /* While true, holds back the automatic start, e.g. until the application is ready. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex|Recording")
  bool bPreventAutomaticStart = false;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex|Recording")
  TArray<TObjectPtr<AActor>> WaitForActorsBeforeStart;

  /* Commit the episode in progress when play ends; otherwise it's discarded. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex|Recording")
  bool bCommitOnEndPlay = true;

  UPROPERTY(BlueprintAssignable, Category = "SuperDex|Recording")
  FSuperDexRecordingStarted OnRecordingStarted;

  UPROPERTY(BlueprintAssignable, Category = "SuperDex|Recording")
  FSuperDexRecordingCommitted OnRecordingCommitted;

  UPROPERTY(BlueprintAssignable, Category = "SuperDex|Recording")
  FSuperDexRecordingAborted OnRecordingAborted;

  /* The world's recording SuperDex Recorder, or none if it has none (or it hasn't initialized
   * yet). */
  UFUNCTION(
      BlueprintPure,
      Category = "SuperDex|Recording",
      meta = (WorldContext = "WorldContextObject"))
  static ASuperDexRecorder* GetActiveRecorder(const UObject* WorldContextObject);

  /* Begins an episode. Returns false if one is already recording or it can't start. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex|Recording")
  bool StartRecording();

  /* Ends the episode and writes its file, blocking until it's written. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex|Recording")
  bool CommitRecording();

  /* Ends the episode and discards it. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex|Recording")
  void AbortRecording(const FString& Reason);

  /* With bExportLiveScene, export the live scene again when the next episode starts, e.g. after
   * physics actors or bots were added or removed. Recorded episodes keep their archive. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex|Recording")
  void ExportLiveSceneAtNextEpisode();

  UFUNCTION(BlueprintPure, Category = "SuperDex|Recording")
  bool IsRecording() const;

  UFUNCTION(BlueprintPure, Category = "SuperDex|Recording")
  FString GetEpisodeId() const;

  UFUNCTION(BlueprintPure, Category = "SuperDex|Recording")
  FString GetOutputPath() const;

  /* Records Text on Stream at the next Mochi step, under "data/events/<Stream>". Ignored while not
   * recording. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex|Recording")
  void RecordEvent(const FString& Stream, const FString& Text);

  /* Records Value at the next Mochi step, under "data/scalars/<Name>". Ignored while not
   * recording. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex|Recording")
  void RecordScalar(const FString& Name, double Value);

  /* Records Values as one sample at the next Mochi step, under "data/scalars/<Name>". The
   * recording doesn't fix the number of values: keep it consistent per name, and interpret the
   * values when reading. Ignored while not recording. */
  UFUNCTION(BlueprintCallable, Category = "SuperDex|Recording")
  void RecordScalars(const FString& Name, const TArray<double>& Values);

  void PostInitializeComponents() override;
  void Tick(float DeltaSeconds) override;
  void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

 private:
  bool IsReadyToStartAutomatically();
  /* The live scene's archive, exported now unless an earlier episode's can be reused, or empty if
   * the export failed. */
  FString GetOrExportLiveSceneArchive();
  /* Tells the sources of the episode that ended, if they haven't been told yet. */
  void EndSources();

  TUniquePtr<FSuperDexSceneRecorder> Recorder;
  /* The sources configured for the current (or just ended) episode. */
  TArray<TWeakObjectPtr<USuperDexRecordingSourceComponent>> RecordingSources;
  /* The archive the live scene was last exported to, or empty to export at the next episode. */
  FString LiveSceneArchivePath;
  bool bIsActiveRecorder = false;
  bool bAutomaticStartDone = false;
};
