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

#include "SuperDexBotSceneSubsystem.h"

#include "ISuperDexRoboticsModule.h"
#include "MochiGameInstanceSubsystem.h"
#include "MochiPhysicsModule.h"
#include "MochiSceneObject.h"

#include "HAL/FileManager.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"

#include <superdex_robotics/scene/bot_scene.h>
#include <superdex_robotics/scene/bot_task.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/file_utils.h>

#include <optional>
#include <string>

namespace {

/**
 * Read a path from the command line, tolerating the leading '=' and the quotes that shell quoting
 * can leave behind, and make it absolute. Returns empty when the switch is absent. LegacySwitch is
 * the name the switch had before the SuperDex rename; it is still accepted, with a deprecation
 * warning, so launchers made for older builds keep working. bOutPresent reports whether either
 * switch was on the command line, even with an empty value.
 */
FString
ParsePathSwitch(const TCHAR* Switch, const TCHAR* LegacySwitch, bool* bOutPresent = nullptr) {
  FString RawPath;
  bool bPresent = FParse::Value(FCommandLine::Get(), Switch, RawPath);
  if (!bPresent && FParse::Value(FCommandLine::Get(), LegacySwitch, RawPath)) {
    bPresent = true;
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("SuperDexBotSceneSubsystem: -%s is deprecated; use -%s instead."),
        LegacySwitch,
        Switch);
  }
  if (bOutPresent) {
    *bOutPresent = bPresent;
  }
  if (!bPresent) {
    return {};
  }
  RawPath.TrimStartAndEndInline();
  if (RawPath.StartsWith(TEXT("="))) {
    RawPath.RightChopInline(1);
  }
  RawPath.TrimQuotesInline();
  return RawPath;
}

FString NormalizePath(const FString& InPath) {
  FString Path = InPath;
  Path.TrimStartAndEndInline();
  if (Path.StartsWith(TEXT("="))) {
    Path.RightChopInline(1);
  }
  Path.TrimQuotesInline();
  if (!Path.IsEmpty() && FPaths::IsRelative(Path)) {
    Path = FPaths::ConvertRelativePathToFull(Path);
  }
  return Path;
}

} // namespace

struct USuperDexBotSceneSubsystem::Impl {
  std::optional<superdex::robotics::BotScene> BotScene;
  std::optional<superdex::robotics::BotTaskPrefab> BotTask;

  /**
   * Resolve the Mochi scene object. Every load and mutation is marshalled through it, because
   * BotScene population and teardown must run on the Mochi thread.
   */
  static UMochiSceneObject* ResolveSceneObject(const UGameInstance* GameInstance) {
    const UMochiGameInstanceSubsystem* MochiSubsystem =
        GameInstance ? GameInstance->GetSubsystem<UMochiGameInstanceSubsystem>() : nullptr;
    UMochiSceneObject* SceneObject = MochiSubsystem ? MochiSubsystem->GetScene() : nullptr;
    return (SceneObject && SceneObject->IsValid()) ? SceneObject : nullptr;
  }

  /** BotScene teardown destroys Mochi actors, so it has to run on the Mochi thread. */
  void ResetBotScene(const UGameInstance* GameInstance) {
    BotTask.reset();
    if (!BotScene.has_value()) {
      return;
    }
    if (UMochiSceneObject* SceneObject = ResolveSceneObject(GameInstance)) {
      SceneObject->QueueOrExecuteCommand([this](mochi::Scene*) { BotScene.reset(); });
      SceneObject->WaitForQueuedCommands();
    } else {
      BotScene.reset();
    }
  }
};

USuperDexBotSceneSubsystem::USuperDexBotSceneSubsystem() : Pimpl(new Impl()) {}

USuperDexBotSceneSubsystem::~USuperDexBotSceneSubsystem() {
  delete Pimpl;
}

void USuperDexBotSceneSubsystem::Initialize(FSubsystemCollectionBase& Collection) {
  Super::Initialize(Collection);
  Collection.InitializeDependency<UMochiGameInstanceSubsystem>();

  // Command-line paths are resolved here, before any actor BeginPlay, so packaged runs never race:
  // by the time anything can observe this subsystem the scene is already Loaded or Failed.
  bool bSceneSwitchPresent = false;
  const FString CommandLineScene =
      ParsePathSwitch(TEXT("SuperDexSceneFile="), TEXT("MochiBotSceneFile="), &bSceneSwitchPresent);
  if (CommandLineScene.IsEmpty() && bSceneSwitchPresent) {
    // Present but empty is a broken launch, not a request for legacy physics.
    SetFailed(TEXT("-SuperDexSceneFile= was supplied with an empty path"));
    return;
  }
  if (CommandLineScene.IsEmpty()) {
    UE_LOG(
        LogMochiPhysics,
        Display,
        TEXT(
            "SuperDexBotSceneSubsystem: no -SuperDexSceneFile= supplied; awaiting an editor fallback or staying disabled."));
    return;
  }
  FString InitialTask;
  FString TaskError;
  if (!ResolveInitialTask(/*FallbackTaskPath=*/{}, InitialTask, TaskError)) {
    SetFailed(TaskError);
    return;
  }
  LoadScene(CommandLineScene, InitialTask);
}

bool USuperDexBotSceneSubsystem::EnsureLoaded(
    const FString& FallbackScenePath,
    const FString& FallbackTaskPath) {
  if (State != ESuperDexBotSceneState::Disabled) {
    return State == ESuperDexBotSceneState::Loaded;
  }
  // The command line always wins; the fallback only fills a gap Initialize could not.
  const FString ResolvedScene = FallbackScenePath;
  if (ResolvedScene.IsEmpty()) {
    return false;
  }
  FString ResolvedTask;
  FString TaskError;
  if (!ResolveInitialTask(FallbackTaskPath, ResolvedTask, TaskError)) {
    SetFailed(TaskError);
    return false;
  }
  return LoadScene(ResolvedScene, ResolvedTask);
}

void USuperDexBotSceneSubsystem::SetFailed(const FString& InError) {
  State = ESuperDexBotSceneState::Failed;
  ErrorMessage = InError;
  if (Pimpl) {
    Pimpl->ResetBotScene(GetGameInstance());
  }
  UE_LOG(
      LogMochiPhysics,
      Error,
      TEXT("SuperDexBotSceneSubsystem: failed to load '%s': %s"),
      *ScenePath,
      *ErrorMessage);
}

bool USuperDexBotSceneSubsystem::ParseTaskFile(const FString& InTaskPath, FString& OutError) {
  superdex::Error TaskError;
  std::string StdTaskPath(TCHAR_TO_UTF8(*InTaskPath));
  auto Task = superdex::robotics::LoadBotTaskPrefabFromFile(StdTaskPath, TaskError);
  if (!TaskError.IsOK()) {
    OutError = FString::Printf(
        TEXT("Bot task '%s': %s"), *InTaskPath, UTF8_TO_TCHAR(TaskError.ToString().c_str()));
    return false;
  }
  Pimpl->BotTask = std::move(Task);
  return true;
}

bool USuperDexBotSceneSubsystem::LoadTask(const FString& InTaskPath) {
  if (InTaskPath.IsEmpty()) {
    return true;
  }
  FString ParseError;
  if (!ParseTaskFile(InTaskPath, ParseError)) {
    SetFailed(ParseError);
    return false;
  }
  return true;
}

bool USuperDexBotSceneSubsystem::BuildTaskPlaylist(const FString& FolderPath, FString& OutError) {
  const FString Folder = NormalizePath(FolderPath);
  TArray<FString> Names;
  IFileManager::Get().FindFiles(Names, *(Folder / TEXT("*")), true, false);
  // The native library decides which extensions are bot tasks.
  Names.RemoveAll(
      [](const FString& Name) { return !superdex::robotics::IsBotTaskPath(TCHAR_TO_UTF8(*Name)); });
  if (Names.IsEmpty()) {
    OutError =
        FString::Printf(TEXT("-SuperDexTaskFolder='%s' contains no bot task files"), *Folder);
    return false;
  }
  // Sorted so the cycle order is the same on every machine; FindFiles does not promise an order.
  Names.Sort();
  TaskPlaylist.Reset(Names.Num());
  for (const FString& Name : Names) {
    TaskPlaylist.Add(Folder / Name);
  }
  TaskCursor = 0;
  return true;
}

bool USuperDexBotSceneSubsystem::ResolveInitialTask(
    const FString& FallbackTaskPath,
    FString& OutTaskPath,
    FString& OutError) {
  const FString TaskFolder =
      ParsePathSwitch(TEXT("SuperDexTaskFolder="), TEXT("MochiBotTaskFolder="));
  const FString TaskFile = ParsePathSwitch(TEXT("SuperDexTaskFile="), TEXT("MochiBotTaskFile="));
  if (TaskFolder.IsEmpty()) {
    OutTaskPath = TaskFile.IsEmpty() ? FallbackTaskPath : TaskFile;
    return true;
  }
  if (!TaskFile.IsEmpty()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("SuperDexBotSceneSubsystem: -SuperDexTaskFolder= overrides -SuperDexTaskFile='%s'."),
        *TaskFile);
  }
  if (!BuildTaskPlaylist(TaskFolder, OutError)) {
    return false;
  }
  OutTaskPath = TaskPlaylist[0];
  UE_LOG(
      LogMochiPhysics,
      Display,
      TEXT("SuperDexBotSceneSubsystem: bot task playlist has %d file(s); starting at '%s'."),
      TaskPlaylist.Num(),
      *OutTaskPath);
  return true;
}

bool USuperDexBotSceneSubsystem::AdvanceToNextTask() {
  if (TaskPlaylist.Num() < 2) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("SuperDexBotSceneSubsystem: nothing to advance to (%d task(s) in the playlist)."),
        TaskPlaylist.Num());
    return false;
  }
  const int32 PreviousCursor = TaskCursor;
  TaskCursor = (TaskCursor + 1) % TaskPlaylist.Num();
  const FString& NextPath = TaskPlaylist[TaskCursor];

  FString ParseError;
  if (!ParseTaskFile(NextPath, ParseError)) {
    // Deliberately not SetFailed: that tears the scene down, and one bad file in a folder should
    // not end the run. The previous task stays live and the cursor has already moved, so the next
    // call steps past this file rather than retrying it.
    UE_LOG(LogMochiPhysics, Error, TEXT("SuperDexBotSceneSubsystem: %s"), *ParseError);
    return false;
  }
  TaskPath = NextPath;
  UE_LOG(
      LogMochiPhysics,
      Display,
      TEXT("SuperDexBotSceneSubsystem: bot task %d/%d%s: '%s'"),
      TaskCursor + 1,
      TaskPlaylist.Num(),
      TaskCursor <= PreviousCursor ? TEXT(" (wrapped)") : TEXT(""),
      *TaskPath);
  return true;
}

bool USuperDexBotSceneSubsystem::LoadScene(const FString& InScenePath, const FString& InTaskPath) {
  ScenePath = NormalizePath(InScenePath);
  TaskPath = NormalizePath(InTaskPath);

  if (ScenePath.IsEmpty()) {
    SetFailed(TEXT("Empty bot scene path"));
    return false;
  }

  UMochiSceneObject* SceneObject = Impl::ResolveSceneObject(GetGameInstance());
  if (!SceneObject) {
    SetFailed(TEXT("Mochi scene unavailable"));
    return false;
  }

  // The scene file is the sole authority over this scene: claim it before loading so no Unreal
  // component or setting adds to it, and stay claimed if the load fails so the session is rejected
  // rather than silently run on Unreal physics.
  SceneObject->SetContentsExternallyOwned(true);
  int32 NumPreexistingActors = 0;
  SceneObject->QueueOrExecuteCommand([&NumPreexistingActors](mochi::Scene* Scene) {
    NumPreexistingActors = Scene ? Scene->GetNumActors() : 0;
  });
  SceneObject->WaitForQueuedCommands();
  if (NumPreexistingActors > 0) {
    SetFailed(
        FString::Printf(
            TEXT(
                "The Mochi scene already holds %d actor(s) created before the bot scene loaded; the "
                "scene file must be the only source of physics actors."),
            NumPreexistingActors));
    return false;
  }

  mochi::Context* MochiContext =
      IMochiPhysicsModule::Get() ? IMochiPhysicsModule::Get()->GetMochiContext() : nullptr;
  superdex::robotics::RoboticsContext* RoboticsContext = ISuperDexRoboticsModule::Get()
      ? ISuperDexRoboticsModule::Get()->GetRoboticsContext()
      : nullptr;
  if (!MochiContext || !RoboticsContext) {
    SetFailed(TEXT("Mochi or Robotics context unavailable"));
    return false;
  }

  // The canonical loader handles archive extraction and path resolution transparently.
  const std::string StdScenePath(TCHAR_TO_UTF8(*ScenePath));
  std::string LoadErrorString;
  bool bLoadSucceeded = false;
  auto* PimplPtr = Pimpl;
  SceneObject->QueueOrExecuteCommand(
      [PimplPtr, StdScenePath, MochiContext, RoboticsContext, &LoadErrorString, &bLoadSucceeded](
          mochi::Scene* Scene) {
        superdex::Error LocalError;
        auto Loaded = superdex::robotics::LoadBotScene(
            Scene,
            StdScenePath,
            /*botsToInitialize=*/{},
            MochiContext,
            RoboticsContext,
            LocalError);
        if (!LocalError.IsOK()) {
          // mochi::Error is move-only, so capture its text rather than the error itself.
          LoadErrorString = LocalError.ToString();
          return;
        }
        PimplPtr->BotScene = std::move(Loaded);
        bLoadSucceeded = true;
      });
  SceneObject->WaitForQueuedCommands();

  if (!bLoadSucceeded || !Pimpl->BotScene.has_value()) {
    SetFailed(
        LoadErrorString.empty() ? TEXT("Failed to load bot scene")
                                : FString(UTF8_TO_TCHAR(LoadErrorString.c_str())));
    return false;
  }

  if (!LoadTask(TaskPath)) {
    return false;
  }

  State = ESuperDexBotSceneState::Loaded;

  int32 NumActors = 0;
  SceneObject->QueueOrExecuteCommand(
      [&NumActors](mochi::Scene* Scene) { NumActors = Scene ? Scene->GetNumActors() : 0; });
  SceneObject->WaitForQueuedCommands();

  auto const& Prefab = Pimpl->BotScene->GetBotScenePrefab();
  int32 NumControllers = 0;
  for (auto const& Bot : Prefab.bots) {
    NumControllers += static_cast<int32>(Bot.controllers.size());
  }
  UE_LOG(
      LogMochiPhysics,
      Display,
      TEXT(
          "SuperDexBotSceneSubsystem: loaded '%s' - %d actors, %d bots declared with %d controllers (none initialized yet), %d task spawn(s)"),
      *ScenePath,
      NumActors,
      static_cast<int32>(Prefab.bots.size()),
      NumControllers,
      Pimpl->BotTask.has_value() ? static_cast<int32>(Pimpl->BotTask->resetState.size()) : 0);

  SceneLoadedDelegate.Broadcast();
  return true;
}

bool USuperDexBotSceneSubsystem::SpawnTaskInstance(
    const FString& InstanceName,
    const FString& PrefabName,
    const mochi::TransformRT& WorldFromInstance,
    TArray<mochi::ActorHandle>& OutActors,
    FString& OutError) {
  OutActors.Reset();
  if (State != ESuperDexBotSceneState::Loaded || !Pimpl || !Pimpl->BotScene.has_value()) {
    OutError = TEXT("Bot scene is not loaded");
    return false;
  }
  UMochiSceneObject* SceneObject = Impl::ResolveSceneObject(GetGameInstance());
  if (!SceneObject) {
    OutError = TEXT("Mochi scene unavailable");
    return false;
  }

  const std::string StdInstanceName(TCHAR_TO_UTF8(*InstanceName));
  const std::string StdPrefabName(TCHAR_TO_UTF8(*PrefabName));
  std::string SpawnErrorString;
  TArray<mochi::ActorHandle> Actors;
  auto* PimplPtr = Pimpl;
  SceneObject->QueueOrExecuteCommand(
      [PimplPtr, StdInstanceName, StdPrefabName, WorldFromInstance, &Actors, &SpawnErrorString](
          mochi::Scene*) {
        superdex::Error LocalError;
        auto Handles = PimplPtr->BotScene->SpawnPrefabInstance(
            StdInstanceName, StdPrefabName, WorldFromInstance, LocalError);
        if (!LocalError.IsOK()) {
          SpawnErrorString = LocalError.ToString();
          return;
        }
        Actors.Reserve(static_cast<int32>(Handles.size()));
        for (auto Handle : Handles) {
          Actors.Add(Handle);
        }
      });
  SceneObject->WaitForQueuedCommands();

  if (!SpawnErrorString.empty()) {
    OutError = UTF8_TO_TCHAR(SpawnErrorString.c_str());
    return false;
  }
  OutActors = MoveTemp(Actors);
  return true;
}

void USuperDexBotSceneSubsystem::NotifySceneContentsChanged() {
  SceneContentsChangedDelegate.Broadcast();
}

void USuperDexBotSceneSubsystem::DestroyAllTaskInstances() {
  if (!Pimpl || !Pimpl->BotScene.has_value()) {
    return;
  }
  UMochiSceneObject* SceneObject = Impl::ResolveSceneObject(GetGameInstance());
  if (!SceneObject) {
    return;
  }
  auto* PimplPtr = Pimpl;
  SceneObject->QueueOrExecuteCommand(
      [PimplPtr](mochi::Scene*) { PimplPtr->BotScene->DestroyAllPrefabInstances(); });
  SceneObject->WaitForQueuedCommands();
  SceneContentsChangedDelegate.Broadcast();
}

void USuperDexBotSceneSubsystem::Deinitialize() {
  SceneLoadedDelegate.Clear();
  SceneContentsChangedDelegate.Clear();

  if (auto* MochiSubsystem = GetGameInstance()
          ? GetGameInstance()->GetSubsystem<UMochiGameInstanceSubsystem>()
          : nullptr) {
    MochiSubsystem->SetSceneExternallyOwned(false);
  }

  // Must finish before UMochiGameInstanceSubsystem::Deinitialize destroys the Scene.
  if (Pimpl) {
    Pimpl->ResetBotScene(GetGameInstance());
  }
  Super::Deinitialize();
}

ESuperDexBotSceneState USuperDexBotSceneSubsystem::GetState() const {
  return State;
}

FString USuperDexBotSceneSubsystem::GetScenePath() const {
  return ScenePath;
}

FString USuperDexBotSceneSubsystem::GetTaskPath() const {
  return TaskPath;
}

FString USuperDexBotSceneSubsystem::GetErrorMessage() const {
  return ErrorMessage;
}

FString USuperDexBotSceneSubsystem::GetErrorMessageOrDefault() const {
  return ErrorMessage.IsEmpty() ? FString(TEXT("Failed to load bot scene")) : ErrorMessage;
}

bool USuperDexBotSceneSubsystem::IsSceneMode() const {
  return State != ESuperDexBotSceneState::Disabled;
}

const superdex::robotics::BotScene* USuperDexBotSceneSubsystem::GetBotScene() const {
  return Pimpl && Pimpl->BotScene.has_value() ? &*Pimpl->BotScene : nullptr;
}

superdex::robotics::BotScene* USuperDexBotSceneSubsystem::GetBotScene() {
  return Pimpl && Pimpl->BotScene.has_value() ? &*Pimpl->BotScene : nullptr;
}

const superdex::robotics::BotTaskPrefab* USuperDexBotSceneSubsystem::GetBotTask() const {
  return Pimpl && Pimpl->BotTask.has_value() ? &*Pimpl->BotTask : nullptr;
}
