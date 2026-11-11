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

#include "SuperDexSceneRecorder.h"

#include "ISuperDexRoboticsModule.h"
#include "MochiPhysicsFunctionLibrary.h"
#include "SuperDexBotComponent.h"
#include "SuperDexRecorderModule.h"

#include <superdex_robotics/utils/file_utils.h>

#include "EngineUtils.h"
#include "Misc/Paths.h"
#include "Misc/ScopeExit.h"
#include "Misc/ScopeLock.h"

#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <utility>

namespace {

std::string ToUtf8(const FString& Value) {
  FTCHARToUTF8 Converted(*Value);
  return {Converted.Get(), static_cast<size_t>(Converted.Length())};
}

bool AddFileReference(
    const FString& PropertyPrefix,
    const FString& Path,
    bool bRequired,
    std::map<std::string, std::string>& Properties,
    FString& OutError) {
  if (Path.IsEmpty()) {
    if (bRequired) {
      OutError = FString::Printf(TEXT("%s path is empty."), *PropertyPrefix);
      return false;
    }
    return true;
  }

  const FString FullPath = FPaths::ConvertRelativePathToFull(Path);
  if (!FPaths::FileExists(FullPath)) {
    OutError = FString::Printf(TEXT("%s file does not exist: %s"), *PropertyPrefix, *FullPath);
    return false;
  }

  mochi::Error Error;
  const auto Hash = superdex::robotics::HashGenericFile(ToUtf8(FullPath), Error);
  if (!Error.IsOK()) {
    OutError = FString::Printf(
        TEXT("Failed to hash %s: %s"), *PropertyPrefix, UTF8_TO_TCHAR(Error.ToString().c_str()));
    return false;
  }

  const std::string Prefix = ToUtf8(PropertyPrefix);
  Properties[Prefix + ".filename"] = ToUtf8(FPaths::GetCleanFilename(FullPath));
  Properties[Prefix + ".hash"] = std::string{Hash.c_str(), Hash.size()};
  Properties[Prefix + ".hash_algorithm"] = "xxh3_64";
  return true;
}

// Fills the fields that a robot's skeleton shadow and a standalone skeleton share.
template <typename SkeletonConfig>
void FillSkeletonConfig(const FSuperDexSkeletonSource& Source, SkeletonConfig& Config) {
  Config.label = ToUtf8(Source.Label);
  for (const FString& Name : Source.KeypointNames) {
    Config.keypointNames.push_back(ToUtf8(Name));
  }
  Config.parentIndices.assign(
      Source.ParentIndices.GetData(), Source.ParentIndices.GetData() + Source.ParentIndices.Num());
  Config.colorRgba = {Source.Color.R, Source.Color.G, Source.Color.B, Source.Color.A};
}

// The scene each capturing recorder records. Used on the game thread only.
TArray<TPair<TWeakObjectPtr<UMochiSceneObject>, const FSuperDexSceneRecorder*>>&
CapturingRecorders() {
  static TArray<TPair<TWeakObjectPtr<UMochiSceneObject>, const FSuperDexSceneRecorder*>> Recorders;
  return Recorders;
}

} // namespace

FSuperDexSceneRecorder::FSuperDexSceneRecorder() = default;

FSuperDexSceneRecorder::~FSuperDexSceneRecorder() {
  AbortEpisode(TEXT("Recorder destroyed before commit."));
}

bool FSuperDexSceneRecorder::BeginEpisode(
    UObject* WorldContext,
    const FSuperDexSceneRecorderConfig& Config) {
  if (Episode != nullptr) {
    UE_LOG(LogSuperDexRecorder, Error, TEXT("An episode is already active."));
    return false;
  }
  if (!IsValid(WorldContext) || WorldContext->GetWorld() == nullptr) {
    UE_LOG(LogSuperDexRecorder, Error, TEXT("Cannot record without a valid Unreal world."));
    return false;
  }

  UMochiSceneObject* NewSceneObject = Config.Scene.Get();
  if (Config.Scene.IsExplicitlyNull()) {
    auto* Subsystem = UMochiPhysicsFunctionLibrary::GetMochiGameInstanceSubsystem(WorldContext);
    NewSceneObject = Subsystem != nullptr ? Subsystem->GetScene() : nullptr;
  }
  if (NewSceneObject == nullptr || !NewSceneObject->IsValid()) {
    UE_LOG(LogSuperDexRecorder, Error, TEXT("Cannot record without a valid Mochi scene."));
    return false;
  }

  std::map<std::string, std::string> Properties;
  for (const auto& Property : Config.Properties) {
    Properties[ToUtf8(Property.Key)] = ToUtf8(Property.Value);
  }
  FString MetadataError;
  if (!AddFileReference(
          TEXT("scene.archive"),
          Config.SceneArchivePath,
          Config.bRequireSceneArchive,
          Properties,
          MetadataError) ||
      !AddFileReference(TEXT("scene.task"), Config.TaskPath, false, Properties, MetadataError)) {
    UE_LOG(LogSuperDexRecorder, Error, TEXT("Cannot start recording: %s"), *MetadataError);
    return false;
  }

  const FString OutputDirectory = ResolveOutputDirectory(Config.OutputDirectory);
  Recorder = std::make_unique<superdex::recorder::Recorder>(
      superdex::recorder::RecorderConfig{std::filesystem::path{ToUtf8(OutputDirectory)}});
  mochi::Error Error;
  Episode = Recorder->BeginEpisode({std::move(Properties)}, Error);
  if (!Error.IsOK() || Episode == nullptr) {
    UE_LOG(
        LogSuperDexRecorder,
        Error,
        TEXT("Failed to begin episode: %s"),
        UTF8_TO_TCHAR(Error.ToString().c_str()));
    Recorder.reset();
    return false;
  }

  EpisodeId = UTF8_TO_TCHAR(Episode->GetEpisodeId().c_str());
  OutputPath = UTF8_TO_TCHAR(Episode->GetOutputPath().string().c_str());
  SceneObject = NewSceneObject;
  PhysicsConfig.recordContactForces = Config.bRecordContactForces;
  PhysicsConfig.spawnNames.clear();
  for (const TPair<mochi::ActorHandle, FString>& SpawnName : Config.SpawnNames) {
    PhysicsConfig.spawnNames.emplace_back(SpawnName.Key, ToUtf8(SpawnName.Value));
  }
  ISuperDexRoboticsModule* const RoboticsModule = ISuperDexRoboticsModule::Get();
  RoboticsContext = RoboticsModule != nullptr ? RoboticsModule->GetRoboticsContext() : nullptr;
  if (RoboticsContext == nullptr) {
    UE_LOG(LogSuperDexRecorder, Error, TEXT("Cannot record without a robotics context."));
    AbortEpisode(TEXT("Robotics context is unavailable."));
    return false;
  }
  RobotActorHandles.Reset();
  RobotControllers.Reset();
  RobotTargetSources.Reset();
  TArray<const USuperDexBotComponent*> RobotComponents;
  TSet<uint64> SeenActors;
  for (TActorIterator<AActor> It(WorldContext->GetWorld()); It; ++It) {
    TArray<USuperDexBotComponent*> Components;
    It->GetComponents<USuperDexBotComponent>(Components);
    for (USuperDexBotComponent* const Component : Components) {
      if (Component == nullptr) {
        continue;
      }
      auto* const Articulation = Component->GetMochiArticulatedActor();
      if (Articulation == nullptr || !Articulation->IsValid()) {
        if (Component->GetBotAsset() != nullptr) {
          UE_LOG(
              LogSuperDexRecorder,
              Error,
              TEXT("Not recording SuperDex bot %s on %s: it has no Mochi articulation."),
              *Component->GetName(),
              *It->GetName());
        }
        continue;
      }
      if (Articulation->GetMochiScene() != NewSceneObject) {
        // Lives in another Mochi scene, such as an IK solver's.
        continue;
      }
      const mochi::ActorHandle Handle = Articulation->GetArticulatedActorHandle();
      if (!Handle.IsValid() || SeenActors.Contains(Handle.value)) {
        continue;
      }
      SeenActors.Add(Handle.value);
      RobotActorHandles.Add(Handle);
      RobotControllers.Add(Component->GetController());
      RobotComponents.Add(Component);
      const auto* const Source = Config.RecordingTargetSources.FindByPredicate(
          [Handle](const auto& Entry) { return Entry.Key.value == Handle.value; });
      RobotTargetSources.Add(Source != nullptr ? Source->Value : nullptr);
    }
  }
  RecordedSkeletons.clear();
  RecordedSkeletons.reserve(Config.SkeletonSources.Num());
  RecordedSourceSkeletons.clear();
  RecordedTransforms.clear();
  for (const FSuperDexSkeletonSource& Source : Config.SkeletonSources) {
    if (Source.Robot.IsExplicitlyNull()) {
      if (Source.Id.IsEmpty() || !Source.SampleWorldKeypoints) {
        UE_LOG(
            LogSuperDexRecorder,
            Warning,
            TEXT("Not recording skeleton %s: it has no ID or no sampler."),
            *Source.Label);
        continue;
      }
      FRecordedSourceSkeleton& Skeleton = RecordedSourceSkeletons.emplace_back();
      Skeleton.Id = ToUtf8(Source.Id);
      FillSkeletonConfig(Source, Skeleton.Config);
      Skeleton.SampleWorldKeypoints = Source.SampleWorldKeypoints;
      Skeleton.Keypoints.SetNum(Source.KeypointNames.Num());
      continue;
    }
    const int32 RobotIndex = RobotComponents.Find(Source.Robot.Get());
    if (RobotIndex == INDEX_NONE || !Source.SampleWorldKeypoints) {
      UE_LOG(
          LogSuperDexRecorder,
          Warning,
          TEXT("Not recording skeleton %s: it has no sampler or its robot is not recorded."),
          Source.Label.IsEmpty() ? *Source.Id : *Source.Label);
      continue;
    }
    FRecordedSkeleton& Skeleton = RecordedSkeletons.emplace_back();
    Skeleton.RobotIndex = static_cast<size_t>(RobotIndex);
    Skeleton.Config.id = ToUtf8(Source.Id);
    FillSkeletonConfig(Source, Skeleton.Config);
    Skeleton.SampleWorldKeypoints = Source.SampleWorldKeypoints;
    Skeleton.Keypoints.SetNum(Source.KeypointNames.Num());
  }
  for (const FSuperDexTransformSource& Source : Config.TransformSources) {
    if (Source.Id.IsEmpty() || !Source.SampleWorldTransform) {
      UE_LOG(
          LogSuperDexRecorder,
          Warning,
          TEXT("Not recording transform source '%s': it has no ID or no sampler."),
          *Source.Id);
      continue;
    }
    RecordedTransforms.push_back({ToUtf8(Source.Id), Source.SampleWorldTransform});
  }

  CaptureError.Reset();
  NextStateFrame = 1;
  bRoboticsInitialized = false;
  NewSceneObject->QueueOrExecuteCommand(
      [this](mochi::Scene* Scene) { InitializeOnMochiThread(*Scene); });
  NewSceneObject->WaitForQueuedCommands();
  if (!CaptureError.IsEmpty()) {
    AbortEpisode(TEXT("Initialization failed."));
    return false;
  }

  bCaptureActive = true;
  for (const auto& Other : CapturingRecorders()) {
    if (Other.Key.Get() == NewSceneObject) {
      UE_LOG(
          LogSuperDexRecorder,
          Warning,
          TEXT(
              "Episode %s records Mochi scene %s, which episode %s is already recording: both record "
              "every body and robot in it."),
          *EpisodeId,
          *NewSceneObject->GetName(),
          *Other.Value->EpisodeId);
    }
  }
  CapturingRecorders().Emplace(NewSceneObject, this);
  {
    FScopeLock Lock(&QueueLock);
    QueuedEvents.Reset();
    QueuedScalars.Reset();
    bAcceptingQueuedValues = true;
  }
  UE_LOG(
      LogSuperDexRecorder,
      Log,
      TEXT("Recording SuperDex episode %s to %s"),
      *EpisodeId,
      *OutputPath);
  return true;
}

void FSuperDexSceneRecorder::InitializeOnMochiThread(mochi::Scene& Scene) {
  mochi::Error Error;
  std::set<std::string> RobotNames;
  for (const mochi::ActorHandle Handle : RobotActorHandles) {
    mochi::Actor* const Actor = Scene.GetActor(Handle);
    const char* const Name = Actor != nullptr ? Actor->GetName() : nullptr;
    if (Name != nullptr && !RobotNames.insert(Name).second) {
      UE_LOG(
          LogSuperDexRecorder,
          Error,
          TEXT("Cannot record more than one SuperDex robot named %s."),
          UTF8_TO_TCHAR(Name));
      MOCHI_ERROR_SET(Error, "Recorded SuperDex robots need unique names.");
      break;
    }
  }
  if (Error.IsOK()) {
    PhysicsRecorder = std::make_unique<superdex::recorder::PhysicsRecorder>(
        *Episode, Scene, PhysicsConfig, Error);
  }
  if (Error.IsOK()) {
    InitializeSourceRecorder(Error);
  }
  if (Error.IsOK()) {
    LastStateTime = {
        .frame = 0,
        .simulationTimeSec = Scene.GetTotalSimulationTime(),
        .wallTimeSec = Episode->GetWallTimeSec()};
    PhysicsRecorder->Record(LastStateTime, Error);
  }
  if (Error.IsOK()) {
    constexpr int kPriority = mochi::Scene::kDefaultCallbackPriority + 100;
    PostStepCallback = Scene.RegisterPostStepCallback(
        "FSuperDexSceneRecorder::PostStep",
        [this](const mochi::StepInfo& StepInfo) { RecordPostStep(StepInfo); },
        kPriority);
  }
  if (!Error.IsOK()) {
    SetCaptureError(Error);
  }
}

void FSuperDexSceneRecorder::InitializeSourceRecorder(mochi::Error& Error) {
  SourceRecorder = std::make_unique<superdex::recorder::SourceRecorder>(*Episode);
  for (const FRecordedSourceSkeleton& Skeleton : RecordedSourceSkeletons) {
    SourceRecorder->AddSkeleton(Skeleton.Id, Skeleton.Config, Error);
    MOCHI_ERROR_RETURN(Error);
  }
}

void FSuperDexSceneRecorder::InitializeRoboticsRecorders(mochi::Scene& Scene, mochi::Error& Error) {
  RoboticsRecorders.reserve(RobotActorHandles.Num());
  BotHandles.reserve(RobotActorHandles.Num());
  RobotShadows.reserve(RobotActorHandles.Num());
  for (const mochi::ActorHandle Handle : RobotActorHandles) {
    mochi::Actor* const Actor = Scene.GetActor(Handle);
    superdex::robotics::Bot* const Robot =
        RoboticsContext != nullptr ? RoboticsContext->GetBotContainingActor(Actor) : nullptr;
    if (Robot == nullptr) {
      MOCHI_ERROR_SET(Error, "Failed to resolve a recorded SuperDex robot.");
      return;
    }
    auto BotRecorder =
        std::make_unique<superdex::recorder::RoboticsRecorder>(*Episode, *Robot, Error);
    MOCHI_ERROR_RETURN(Error);
    RoboticsRecorders.push_back(std::move(BotRecorder));
    BotHandles.push_back(Robot->GetHandle());
    RobotShadows.push_back(nullptr);
  }
  for (FRecordedSkeleton& Skeleton : RecordedSkeletons) {
    superdex::robotics::Bot* const Robot = RoboticsContext->GetBot(BotHandles[Skeleton.RobotIndex]);
    if (Robot == nullptr) {
      MOCHI_ERROR_SET(Error, "A recorded SuperDex robot was destroyed.");
      return;
    }
    Skeleton.Shadow = std::make_unique<superdex::recorder::SkeletonShadow>(
        *Episode, *Robot, Skeleton.Config, Error);
    MOCHI_ERROR_RETURN(Error);
  }
  bRoboticsInitialized = true;
}

void FSuperDexSceneRecorder::RecordPostStep(const mochi::StepInfo& StepInfo) {
  if (!CaptureError.IsEmpty() || Episode == nullptr || PhysicsRecorder == nullptr ||
      StepInfo.scene == nullptr) {
    return;
  }

  mochi::Error Error;
  if (!bRoboticsInitialized) {
    InitializeRoboticsRecorders(*StepInfo.scene, Error);
  }
  const superdex::recorder::SampleTime CommandTime = LastStateTime;
  for (size_t Index = 0; Error.IsOK() && Index < RoboticsRecorders.size(); ++Index) {
    RoboticsRecorders[Index]->Record(CommandTime, Error);
    if (!Error.IsOK()) {
      break;
    }
    int32 AnchorLinkIndex = INDEX_NONE;
    mochi::TransformRT WorldFromAnchor;
    TConstArrayView<mochi::real> JointPositions;
    bool bHasTarget = false;
    if (USuperDexBotController* const Controller =
            RobotControllers[static_cast<int32>(Index)].Get()) {
      bHasTarget = Controller->GetRecordingTarget(AnchorLinkIndex, WorldFromAnchor, JointPositions);
    } else if (
        const FSuperDexRecordingTargetSourcePtr& Source =
            RobotTargetSources[static_cast<int32>(Index)]) {
      bHasTarget =
          Source->GetRecordingTarget(AnchorLinkIndex, WorldFromAnchor, SourceJointPositions);
      JointPositions = SourceJointPositions;
    }
    if (bHasTarget) {
      if (RobotShadows[Index] == nullptr) {
        superdex::robotics::Bot* const Robot = RoboticsContext->GetBot(BotHandles[Index]);
        if (Robot == nullptr) {
          MOCHI_ERROR_SET(Error, "A recorded SuperDex robot was destroyed.");
          break;
        }
        RobotShadows[Index] = std::make_unique<superdex::recorder::RobotShadow>(
            *Episode,
            *Robot,
            superdex::recorder::RobotShadowConfig{
                .id = "command",
                .subtreeRootLinkIndex = static_cast<size_t>(AnchorLinkIndex),
            },
            Error);
      }
      if (Error.IsOK()) {
        RobotShadows[Index]->Record(
            CommandTime,
            WorldFromAnchor,
            {JointPositions.GetData(), static_cast<size_t>(JointPositions.Num())},
            Error);
      }
    }
  }
  for (size_t Index = 0; Error.IsOK() && Index < RecordedSkeletons.size(); ++Index) {
    FRecordedSkeleton& Skeleton = RecordedSkeletons[Index];
    if (Skeleton.SampleWorldKeypoints(Skeleton.Keypoints)) {
      Skeleton.Shadow->Record(
          CommandTime,
          {Skeleton.Keypoints.GetData(), static_cast<size_t>(Skeleton.Keypoints.Num())},
          Error);
    } else {
      Skeleton.Shadow->RecordUntracked(CommandTime, Error);
    }
  }
  RecordSources(CommandTime, Error);
  RecordQueuedValues(CommandTime, Error);
  if (Error.IsOK()) {
    LastStateTime = {
        .frame = NextStateFrame,
        .simulationTimeSec = StepInfo.scene->GetTotalSimulationTime(),
        .wallTimeSec = Episode->GetWallTimeSec()};
    PhysicsRecorder->Record(LastStateTime, Error);
    ++NextStateFrame;
  }
  if (!Error.IsOK()) {
    SetCaptureError(Error);
  }
}

void FSuperDexSceneRecorder::RecordSources(
    const superdex::recorder::SampleTime& Time,
    mochi::Error& Error) {
  MOCHI_ERROR_RETURN(Error);
  for (FRecordedTransform& Transform : RecordedTransforms) {
    mochi::TransformRT WorldTransform;
    if (Transform.SampleWorldTransform(WorldTransform)) {
      SourceRecorder->RecordTransform(Time, Transform.Id, WorldTransform, Error);
      MOCHI_ERROR_RETURN(Error);
    }
  }
  for (FRecordedSourceSkeleton& Skeleton : RecordedSourceSkeletons) {
    if (Skeleton.SampleWorldKeypoints(Skeleton.Keypoints)) {
      SourceRecorder->RecordSkeleton(
          Time,
          Skeleton.Id,
          {Skeleton.Keypoints.GetData(), static_cast<size_t>(Skeleton.Keypoints.Num())},
          Error);
    } else {
      SourceRecorder->RecordSkeletonUntracked(Time, Skeleton.Id, Error);
    }
    MOCHI_ERROR_RETURN(Error);
  }
}

void FSuperDexSceneRecorder::RecordQueuedValues(
    const superdex::recorder::SampleTime& Time,
    mochi::Error& Error) {
  MOCHI_ERROR_RETURN(Error);
  TArray<TPair<FString, FString>> Events;
  TMap<FString, TArray<double>> Scalars;
  {
    FScopeLock Lock(&QueueLock);
    Events = MoveTemp(QueuedEvents);
    Scalars = MoveTemp(QueuedScalars);
    QueuedEvents.Reset();
    QueuedScalars.Reset();
  }

  // One sample per stream per step, holding that stream's texts in the order they were raised.
  TMap<FString, std::vector<std::string>> TextsByStream;
  for (const TPair<FString, FString>& Event : Events) {
    TextsByStream.FindOrAdd(Event.Key).push_back(ToUtf8(Event.Value));
  }
  for (const TPair<FString, std::vector<std::string>>& Stream : TextsByStream) {
    SourceRecorder->RecordTexts(Time, ToUtf8(Stream.Key), Stream.Value, Error);
    MOCHI_ERROR_RETURN(Error);
  }
  std::vector<mochi::real> Values;
  for (const TPair<FString, TArray<double>>& Scalar : Scalars) {
    Values.clear();
    for (const double Value : Scalar.Value) {
      Values.push_back(static_cast<mochi::real>(Value));
    }
    SourceRecorder->RecordScalars(Time, ToUtf8(Scalar.Key), Values, Error);
    MOCHI_ERROR_RETURN(Error);
  }
}

void FSuperDexSceneRecorder::RecordEvent(FStringView Stream, FStringView Text) {
  FScopeLock Lock(&QueueLock);
  if (bAcceptingQueuedValues) {
    QueuedEvents.Emplace(FString{Stream}, FString{Text});
  }
}

void FSuperDexSceneRecorder::RecordScalar(FStringView Name, double Value) {
  RecordScalars(Name, MakeArrayView(&Value, 1));
}

void FSuperDexSceneRecorder::RecordScalars(FStringView Name, TConstArrayView<double> Values) {
  FScopeLock Lock(&QueueLock);
  if (bAcceptingQueuedValues) {
    QueuedScalars.Add(FString{Name}, TArray<double>{Values});
  }
}

void FSuperDexSceneRecorder::SetCaptureError(const mochi::Error& Error) {
  CaptureError = UTF8_TO_TCHAR(Error.ToString().c_str());
  UE_LOG(LogSuperDexRecorder, Error, TEXT("Recording failed: %s"), *CaptureError);
}

void FSuperDexSceneRecorder::EndCapture() {
  {
    FScopeLock Lock(&QueueLock);
    bAcceptingQueuedValues = false;
  }
  ON_SCOPE_EXIT {
    FScopeLock Lock(&QueueLock);
    QueuedEvents.Reset();
    QueuedScalars.Reset();
  };
  if (!bCaptureActive && PhysicsRecorder == nullptr && RoboticsRecorders.empty() &&
      RecordedSkeletons.empty() && SourceRecorder == nullptr) {
    return;
  }
  UMochiSceneObject* const CurrentSceneObject = SceneObject.Get();
  if (CurrentSceneObject != nullptr && CurrentSceneObject->IsValid()) {
    if (PostStepCallback.IsValid()) {
      CurrentSceneObject->CancelCallback(PostStepCallback);
    }
    CurrentSceneObject->QueueOrExecuteCommand([this](mochi::Scene*) {
      // Events and scalars raised since the last step (such as an outcome decided right before
      // the episode ends) are written at the last recorded step rather than dropped.
      if (CaptureError.IsEmpty() && SourceRecorder != nullptr) {
        mochi::Error Error;
        RecordQueuedValues(LastStateTime, Error);
        if (!Error.IsOK()) {
          SetCaptureError(Error);
        }
      }
      RecordedSkeletons.clear();
      RecordedSourceSkeletons.clear();
      RecordedTransforms.clear();
      SourceRecorder.reset();
      RobotShadows.clear();
      RoboticsRecorders.clear();
      BotHandles.clear();
      PhysicsRecorder.reset();
    });
    CurrentSceneObject->WaitForQueuedCommands();
  } else {
    UE_LOG(
        LogSuperDexRecorder,
        Warning,
        TEXT("The Mochi scene was destroyed during episode %s; recording ends at its last step."),
        *EpisodeId);
    RecordedSkeletons.clear();
    RecordedSourceSkeletons.clear();
    RecordedTransforms.clear();
    SourceRecorder.reset();
    RobotShadows.clear();
    RoboticsRecorders.clear();
    BotHandles.clear();
    // Leaked on purpose: its destructor may cancel contact queries through the destroyed scene.
    static_cast<void>(PhysicsRecorder.release());
  }
  PostStepCallback = {};
  bCaptureActive = false;
  CapturingRecorders().RemoveAll([this](const auto& Entry) { return Entry.Value == this; });
}

bool FSuperDexSceneRecorder::CommitEpisode() {
  if (Episode == nullptr) {
    return false;
  }
  EndCapture();
  if (!CaptureError.IsEmpty()) {
    AbortEpisode(TEXT("Capture failed."));
    return false;
  }

  mochi::Error Error;
  Episode->Commit(Error);
  if (!Error.IsOK()) {
    UE_LOG(
        LogSuperDexRecorder,
        Error,
        TEXT("Failed to commit episode %s: %s"),
        *EpisodeId,
        UTF8_TO_TCHAR(Error.ToString().c_str()));
    mochi::Error AbortError;
    Episode->Abort("Commit failed.", AbortError);
    Episode.reset();
    Recorder.reset();
    return false;
  }
  UE_LOG(LogSuperDexRecorder, Log, TEXT("Committed SuperDex recording: %s"), *OutputPath);
  Episode.reset();
  Recorder.reset();
  return true;
}

void FSuperDexSceneRecorder::AbortEpisode(FStringView Reason) {
  if (Episode == nullptr) {
    return;
  }
  EndCapture();
  mochi::Error Error;
  Episode->Abort(ToUtf8(FString{Reason.Len(), Reason.GetData()}), Error);
  if (!Error.IsOK()) {
    UE_LOG(
        LogSuperDexRecorder,
        Error,
        TEXT("Failed to abort episode %s: %s"),
        *EpisodeId,
        UTF8_TO_TCHAR(Error.ToString().c_str()));
  }
  Episode.reset();
  Recorder.reset();
}

bool FSuperDexSceneRecorder::IsRecording() const {
  return Episode != nullptr;
}

FString FSuperDexSceneRecorder::GetEpisodeId() const {
  return EpisodeId;
}

FString FSuperDexSceneRecorder::GetOutputPath() const {
  return OutputPath;
}

FString FSuperDexSceneRecorder::ResolveOutputDirectory(const FString& OutputDirectory) {
  return FPaths::ConvertRelativePathToFull(
      OutputDirectory.IsEmpty() ? FPaths::ProjectSavedDir() / TEXT("SuperDexRecorder")
                                : OutputDirectory);
}
