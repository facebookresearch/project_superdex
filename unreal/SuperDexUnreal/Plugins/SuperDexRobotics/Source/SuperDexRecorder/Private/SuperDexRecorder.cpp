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

#include "SuperDexRecorder.h"

#include "MochiGameInstanceSubsystem.h"
#include "MochiPhysicsFunctionLibrary.h"
#include "SuperDexRecorderModule.h"
#include "SuperDexRecorderWorldSubsystem.h"
#include "SuperDexRecordingSourceComponent.h"
#include "SuperDexRoboticsFunctionLibrary.h"
#include "SuperDexSceneRecorder.h"

#include "Components/SceneComponent.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "HAL/FileManager.h"
#include "Misc/App.h"
#include "Misc/DateTime.h"
#include "Misc/Paths.h"

ASuperDexRecorder::ASuperDexRecorder() {
  RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
  PrimaryActorTick.bCanEverTick = true;
}

ASuperDexRecorder* ASuperDexRecorder::GetActiveRecorder(const UObject* WorldContextObject) {
  const UWorld* const World = GEngine != nullptr
      ? GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull)
      : nullptr;
  const USuperDexRecorderWorldSubsystem* const Subsystem =
      World != nullptr ? World->GetSubsystem<USuperDexRecorderWorldSubsystem>() : nullptr;
  return Subsystem != nullptr ? Subsystem->GetActiveRecorder() : nullptr;
}

bool ASuperDexRecorder::StartRecording() {
  if (!bIsActiveRecorder) {
    UE_LOG(
        LogSuperDexRecorder,
        Error,
        TEXT("%s doesn't record: another SuperDex Recorder records this world."),
        *GetActorNameOrLabel());
    return false;
  }
  if (IsRecording()) {
    UE_LOG(LogSuperDexRecorder, Warning, TEXT("%s is already recording."), *GetActorNameOrLabel());
    return false;
  }
  if (!Recorder.IsValid()) {
    Recorder = MakeUnique<FSuperDexSceneRecorder>();
  }
  EndSources();

  FSuperDexSceneRecorderConfig Config;
  Config.OutputDirectory = OutputDirectory.Path;
  // Both record the game-instance scene: the recorder doesn't set Config.Scene.
  Config.SceneArchivePath =
      bExportLiveScene ? GetOrExportLiveSceneArchive() : SceneArchive.FilePath;
  Config.bRequireSceneArchive = bRequireSceneArchive;
  Config.Properties = Properties;
  Config.Properties.FindOrAdd(TEXT("source"), FString{TEXT("unreal.superdex_recorder")});
  Config.Properties.FindOrAdd(TEXT("unreal.project"), FString{FApp::GetProjectName()});
  if (const UWorld* const World = GetWorld()) {
    Config.Properties.FindOrAdd(TEXT("unreal.map"), World->GetMapName());
  }
  Config.bRecordContactForces = bRecordContactForces;

  TArray<AActor*> Actors{this};
  for (AActor* const Actor : SourceActors) {
    Actors.AddUnique(Actor);
  }
  for (const AActor* const Actor : Actors) {
    if (!IsValid(Actor)) {
      continue;
    }
    TArray<USuperDexRecordingSourceComponent*> Sources;
    Actor->GetComponents<USuperDexRecordingSourceComponent>(Sources);
    for (USuperDexRecordingSourceComponent* const Source : Sources) {
      Source->ConfigureRecording(Config);
      Source->SetComponentTickEnabled(true);
      RecordingSources.Add(Source);
    }
  }

  if (!Recorder->BeginEpisode(this, Config)) {
    EndSources();
    return false;
  }
  OnRecordingStarted.Broadcast(Recorder->GetEpisodeId());
  return true;
}

bool ASuperDexRecorder::CommitRecording() {
  if (!IsRecording()) {
    return false;
  }
  const FString OutputPath = Recorder->GetOutputPath();
  const bool bCommitted = Recorder->CommitEpisode();
  EndSources();
  if (!bCommitted) {
    OnRecordingAborted.Broadcast(TEXT("Commit failed."));
    return false;
  }
  OnRecordingCommitted.Broadcast(OutputPath);
  return true;
}

void ASuperDexRecorder::AbortRecording(const FString& Reason) {
  if (!IsRecording()) {
    return;
  }
  Recorder->AbortEpisode(Reason);
  EndSources();
  OnRecordingAborted.Broadcast(Reason);
}

void ASuperDexRecorder::ExportLiveSceneAtNextEpisode() {
  LiveSceneArchivePath.Reset();
}

bool ASuperDexRecorder::IsRecording() const {
  return Recorder.IsValid() && Recorder->IsRecording();
}

FString ASuperDexRecorder::GetEpisodeId() const {
  return IsRecording() ? Recorder->GetEpisodeId() : FString{};
}

FString ASuperDexRecorder::GetOutputPath() const {
  return IsRecording() ? Recorder->GetOutputPath() : FString{};
}

void ASuperDexRecorder::RecordEvent(const FString& Stream, const FString& Text) {
  if (Recorder.IsValid()) {
    Recorder->RecordEvent(Stream, Text);
  }
}

void ASuperDexRecorder::RecordScalar(const FString& Name, double Value) {
  if (Recorder.IsValid()) {
    Recorder->RecordScalar(Name, Value);
  }
}

void ASuperDexRecorder::RecordScalars(const FString& Name, const TArray<double>& Values) {
  if (Recorder.IsValid()) {
    Recorder->RecordScalars(Name, Values);
  }
}

void ASuperDexRecorder::PostInitializeComponents() {
  Super::PostInitializeComponents();
  // Before any actor begins play, so GetActiveRecorder works from every BeginPlay.
  UWorld* const World = GetWorld();
  USuperDexRecorderWorldSubsystem* const Subsystem = World != nullptr && World->IsGameWorld()
      ? World->GetSubsystem<USuperDexRecorderWorldSubsystem>()
      : nullptr;
  if (Subsystem != nullptr) {
    bIsActiveRecorder = Subsystem->Register(this);
  }
}

void ASuperDexRecorder::Tick(float DeltaSeconds) {
  Super::Tick(DeltaSeconds);
  if (!IsRecording()) {
    // The episode may have failed on the Mochi thread.
    EndSources();
  }
  if (bIsActiveRecorder && bStartAutomatically && !bAutomaticStartDone && !IsRecording() &&
      IsReadyToStartAutomatically()) {
    // One attempt: a configuration error would fail the same way every tick.
    bAutomaticStartDone = true;
    StartRecording();
  }
}

void ASuperDexRecorder::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  if (IsRecording()) {
    if (bCommitOnEndPlay) {
      CommitRecording();
    } else {
      AbortRecording(TEXT("Play ended."));
    }
  }
  EndSources();
  if (UWorld* const World = GetWorld()) {
    if (USuperDexRecorderWorldSubsystem* const Subsystem =
            World->GetSubsystem<USuperDexRecorderWorldSubsystem>()) {
      Subsystem->Unregister(this);
    }
  }
  bIsActiveRecorder = false;
  Super::EndPlay(EndPlayReason);
}

bool ASuperDexRecorder::IsReadyToStartAutomatically() {
  if (bPreventAutomaticStart) {
    return false;
  }
  const UMochiGameInstanceSubsystem* const Subsystem =
      UMochiPhysicsFunctionLibrary::GetMochiGameInstanceSubsystem(this);
  const UMochiSceneObject* const Scene = Subsystem != nullptr ? Subsystem->GetScene() : nullptr;
  if (Scene == nullptr || !Scene->IsValid()) {
    return false;
  }
  for (const AActor* const Actor : WaitForActorsBeforeStart) {
    if (Actor != nullptr && !Actor->HasActorBegunPlay()) {
      return false;
    }
  }
  return true;
}

FString ASuperDexRecorder::GetOrExportLiveSceneArchive() {
  if (!LiveSceneArchivePath.IsEmpty() && FPaths::FileExists(LiveSceneArchivePath)) {
    return LiveSceneArchivePath;
  }
  LiveSceneArchivePath.Reset();
  if (!SceneArchive.FilePath.IsEmpty()) {
    UE_LOG(
        LogSuperDexRecorder,
        Warning,
        TEXT("%s ignores SceneArchive %s: it exports the live scene."),
        *GetActorNameOrLabel(),
        *SceneArchive.FilePath);
  }

  // A new name per export, so a later export never replaces the archive that recorded episodes
  // reference by hash.
  const UWorld* const World = GetWorld();
  const FString MapName =
      World != nullptr ? UWorld::RemovePIEPrefix(World->GetMapName()) : FString{TEXT("Scene")};
  const FString ExportName = FString::Printf(
      TEXT("%s_%s"), *MapName, *FDateTime::UtcNow().ToString(TEXT("%Y%m%d_%H%M%S_%s")));
  const FString StagingDirectory =
      FSuperDexSceneRecorder::ResolveOutputDirectory(OutputDirectory.Path) / TEXT("scenes") /
      ExportName;
  if (!USuperDexRoboticsFunctionLibrary::ExportSuperDexBotScene(
          this, ExportName, StagingDirectory, /*bCaptureLiveScene=*/true)) {
    UE_LOG(
        LogSuperDexRecorder,
        Warning,
        TEXT("%s couldn't export the live scene; see the log above."),
        *GetActorNameOrLabel());
    return {};
  }
  // The archive holds everything the export staged.
  IFileManager::Get().DeleteDirectory(*StagingDirectory, /*RequireExists=*/false, /*Tree=*/true);
  LiveSceneArchivePath = StagingDirectory + TEXT(".superdex_scene_archive");
  UE_LOG(
      LogSuperDexRecorder,
      Log,
      TEXT("%s exported the live scene to %s"),
      *GetActorNameOrLabel(),
      *LiveSceneArchivePath);
  return LiveSceneArchivePath;
}

void ASuperDexRecorder::EndSources() {
  for (const TWeakObjectPtr<USuperDexRecordingSourceComponent>& Source : RecordingSources) {
    if (USuperDexRecordingSourceComponent* const Valid = Source.Get()) {
      Valid->SetComponentTickEnabled(false);
      Valid->OnRecordingEnded();
    }
  }
  RecordingSources.Reset();
}
