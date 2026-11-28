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

#include "Examples/SuperDexRuntimeSceneLoadExample.h"

#include "Components/InputComponent.h"
#include "Controllers/Controller_MochiArticulatedPose.h"
#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "GameFramework/PlayerController.h"
#include "HAL/FileManager.h"
#include "ISuperDexRoboticsModule.h"
#include "Misc/App.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "MochiArticulationUtil.h"
#include "MochiConversions.h"
#include "MochiGameInstanceSubsystem.h"
#include "MochiSceneObject.h"
#include "SuperDexBot.h"
#include "SuperDexBotComponent.h"
#include "SuperDexBotSceneSubsystem.h"
#include "SuperDexBotSceneVisualizationComponent.h"
#include "SuperDexPoseUtils.h"
#include "SuperDexPoseableComponent.h"
#include "TeleopNode.h"
#include "TeleopRoboperatorActor.h"
#include "TimerManager.h"

#if WITH_SUPERDEX_RECORDER
#include "SuperDexSceneRecorder.h"
#endif

#include <superdex_robotics/controllers/controller_base.h>
#include <superdex_robotics/controllers/controller_mochi_articulated_pose.h>
#include <superdex_robotics/scene/bot_scene.h>
#include <superdex_robotics/scene/bot_task.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/bot_utils.h>

#include <string>

DEFINE_LOG_CATEGORY_STATIC(LogSuperDexRuntimeSceneLoadExample, Log, All);

namespace sr = superdex::robotics;

namespace {

/** The floats of a ControllerInputKind::Pose input: x y z [m], qx qy qz qw. */
constexpr int32 PoseInputSize = 7;

/** Log a message and print it on screen, where the operator sees it. Game thread only. */
void Announce(const FString& Message, const FColor& Color = FColor::White) {
  UE_LOG(LogSuperDexRuntimeSceneLoadExample, Display, TEXT("%s"), *Message);
  SUPERDEX_SCREEN_PRINTF_C(Color, "%s", *Message);
}

void Warn(const FString& Message) {
  UE_LOG(LogSuperDexRuntimeSceneLoadExample, Warning, TEXT("%s"), *Message);
  SUPERDEX_SCREEN_PRINTF_C(FColor::Yellow, "%s", *Message);
}

void Fail(const FString& Message) {
  UE_LOG(LogSuperDexRuntimeSceneLoadExample, Error, TEXT("%s"), *Message);
  SUPERDEX_SCREEN_PRINTF_C(FColor::Red, "%s", *Message);
}

template <typename String>
FString ToFString(const String& Value) {
  return FString(UTF8_TO_TCHAR(Value.c_str()));
}

std::string ToStdString(const FString& Value) {
  return std::string(TCHAR_TO_UTF8(*Value));
}

/** The `<bot>/<controller>` key the scene names a bot's controller by. */
std::string ControllerKey(const sr::BotEntry& Bot, const sr::ControllerEntry& Controller) {
  return std::string(std::string_view(Bot.name)) + "/" +
      std::string(std::string_view(Controller.name));
}

/** A relative path names a file or folder under the project directory, if one is there. */
FString ResolveProjectPath(const FString& Path) {
  if (Path.IsEmpty() || !FPaths::IsRelative(Path)) {
    return Path;
  }
  const FString ProjectRelative = FPaths::ConvertRelativePathToFull(FPaths::ProjectDir(), Path);
  return FPaths::FileExists(ProjectRelative) || FPaths::DirectoryExists(ProjectRelative)
      ? ProjectRelative
      : Path;
}

FString CommandLineValue(const TCHAR* Key, const FString& Default) {
  FString Value;
  return FParse::Value(FCommandLine::Get(), Key, Value) ? Value : Default;
}

/** The task files of a `.superdex_task` or of a folder of them, in name order. */
TArray<FString> ListTaskFiles(const FString& Path) {
  TArray<FString> Files;
  if (FPaths::DirectoryExists(Path)) {
    IFileManager::Get().FindFiles(Files, *(Path / TEXT("*.superdex_task")), true, false);
    Files.Sort();
    for (FString& File : Files) {
      File = Path / File;
    }
  } else if (!Path.IsEmpty()) {
    Files.Add(Path);
  }
  return Files;
}

USuperDexBotSceneSubsystem* GetSceneSubsystem(const AActor& Actor) {
  const UGameInstance* GameInstance = Actor.GetGameInstance();
  return GameInstance ? GameInstance->GetSubsystem<USuperDexBotSceneSubsystem>() : nullptr;
}

UMochiSceneObject* GetMochiScene(const AActor& Actor) {
  const UGameInstance* GameInstance = Actor.GetGameInstance();
  const UMochiGameInstanceSubsystem* Subsystem =
      GameInstance ? GameInstance->GetSubsystem<UMochiGameInstanceSubsystem>() : nullptr;
  return Subsystem ? Subsystem->GetScene() : nullptr;
}

/** A scene controller driven through its generic float input. */
struct FDrivenController {
  sr::ControllerHandle Handle;
  sr::ControllerInputKind Kind = sr::ControllerInputKind::None;
  TArray<int32> Dofs;
  /** The input the controller holds: where the episode started, updated by each target. */
  TArray<mochi::real> Held;
  bool bHasHeld = false;
};

/** A bot and the roboperator that drives it. */
struct FDrivenBot {
  FString Name;
  mochi::ActorHandle Actor;
  /** The recorder names robots by their Mochi actor, which every instance of a bot file shares. */
  FString ActorName;
  /** The DOFs of a free base, which come first among the actor's DOFs. */
  int32 NumBaseDofs = 0;
  // Operator and Robot outlive the step callback: EndEpisode cancels it before destroying them.
  UTeleopNodeOperatorComponent* Operator = nullptr;
  /** Takes the bot's joint states, which the roboperator's retargeters and clutch pose from. */
  UTeleopNodeRobotComponent* Robot = nullptr;
  int32 NumArmJoints = 0;
  int32 NumHandJoints = 0;
  bool bArmPosed = false;
  TArray<FDrivenController> Controllers;

  sr::ControllerHandle PoseFollower;
  TWeakObjectPtr<USuperDexPoseableComponent> PoseSource;
  FMochiArticulatedPoseGeometry PoseGeometry;
  FCriticalSection PoseTargetMutex;
  TOptional<sr::ControllerMochiArticulatedPose::Target> PoseTarget;

  // Scratch for each step.
  TArray<double> ArmJoints;
  TArray<double> HandJoints;
  TArray<mochi::real> Input;
  TArray<mochi::real> Efforts;
  TArray<mochi::real> Tau;
  TArray<int> Indices;
  TArray<double> Reported;
};

/**
 * Add the scene controllers of a bot that this example can drive, and warn about the rest. Runs on
 * the scene thread.
 */
void AddControllers(
    FDrivenBot& Driven,
    const sr::BotEntry& Entry,
    const sr::BotScene& Scene,
    const sr::RoboticsContext& Ctx,
    TArray<FString>& Warnings) {
  for (const sr::ControllerEntry& ControllerEntry : Entry.controllers) {
    const sr::ControllerHandle Handle =
        Scene.GetControllerHandle(ControllerKey(Entry, ControllerEntry));
    const sr::ControllerBase* Controller = Ctx.GetController(Handle);
    if (!Controller) {
      continue;
    }
    if (Controller->GetTypeName() == sr::ControllerMochiArticulatedPose::TypeName()) {
      Driven.PoseFollower = Handle;
      continue;
    }
    const FString ControllerName = ToFString(ControllerEntry.name);
    const sr::ControllerInputKind Kind = Controller->GetInputKind();
    if (Kind != sr::ControllerInputKind::Pose && Kind != sr::ControllerInputKind::JointPositions) {
      Warnings.Add(
          FString::Printf(
              TEXT(
                  "Controller '%s' of bot '%s' takes no pose or joint-position input; it is not driven."),
              *ControllerName,
              *Driven.Name));
      continue;
    }
    if (Kind == sr::ControllerInputKind::Pose) {
      if (Driven.bArmPosed || Controller->GetInputSize() != PoseInputSize) {
        Warnings.Add(
            FString::Printf(
                TEXT("Pose controller '%s' of bot '%s' is not driven: a bot takes one arm pose."),
                *ControllerName,
                *Driven.Name));
        continue;
      }
      Driven.bArmPosed = true;
    }
    FDrivenController& Added = Driven.Controllers.AddDefaulted_GetRef();
    Added.Handle = Handle;
    Added.Kind = Kind;
    const mochi::Span<int const> Dofs = Controller->GetControlledDofs();
    Added.Dofs.Append(Dofs.data(), static_cast<int32>(Dofs.size()));
  }
}

/**
 * Put a bot at rest in the pose the scene first brought it up in, which is remembered when it comes
 * up, and hold it there. Runs on the scene thread.
 */
void RestoreInitialPose(
    mochi::Actor& Actor,
    TArray<mochi::real>& InitialPose,
    bool bJustBroughtUp) {
  const int32 NumDofs = Actor.GetNumDofs();
  mochi::Error Error;
  if (bJustBroughtUp || InitialPose.Num() != NumDofs) {
    InitialPose.SetNumUninitialized(NumDofs);
    Actor.GetArticulatedPose(mochi::Span<mochi::real>(InitialPose.GetData(), NumDofs), Error);
  }
  TArray<mochi::real> Zeros;
  Zeros.SetNumZeroed(NumDofs);
  const mochi::Span<mochi::real const> Pose(InitialPose.GetData(), NumDofs);
  Actor.SetArticulatedPoseFromJoints(Pose, Error);
  Actor.SetArticulatedJointVelocities(
      mochi::Span<mochi::real const>(Zeros.GetData(), NumDofs), Error);
  Actor.SetArticulatedTargetPose(Pose, Error);
}

/** Report the bot's arm and hand joint states and applied efforts to its roboperator. */
void ReportState(const mochi::Actor& Actor, FDrivenBot& Bot) {
  if (!Bot.Robot) {
    return;
  }
  const FMochiArticulationState State = ExtractArticulatedActorState(&Actor);
  // Joint positions exclude the base DOFs, which come first among the actor's DOFs.
  const int32 NumBaseDofs = Actor.GetNumDofs() - State.JointPositions.Num();
  auto Gather =
      [&Bot](const TArray<mochi::real>& Values, int32 First, int32 Count) -> const TArray<double>& {
    Bot.Reported.SetNumZeroed(Count);
    for (int32 i = 0; i < Count && First + i < Values.Num(); ++i) {
      Bot.Reported[i] = Values[First + i];
    }
    return Bot.Reported;
  };
  UTeleopNodeRobotComponent& Robot = *Bot.Robot;
  const int32 Arm = Bot.NumArmJoints;
  const int32 Hand = Bot.NumHandJoints;
  Robot.SetArmJointStates(Gather(State.JointPositions, 0, Arm));
  Robot.SetArmJointVelocities(Gather(State.JointVelocities, 0, Arm));
  Robot.SetArmJointEfforts(Gather(Bot.Tau, NumBaseDofs, Arm));
  Robot.SetArmJointControlPolicyEfforts(Bot.Reported);
  Robot.SetHandJointStates(Gather(State.JointPositions, Arm, Hand));
  Robot.SetHandJointVelocities(Gather(State.JointVelocities, Arm, Hand));
  Robot.SetHandJointEfforts(Gather(Bot.Tau, NumBaseDofs + Arm, Hand));
  Robot.SetHandJointControlPolicyEfforts(Bot.Reported);
}

void DriveBot(sr::RoboticsContext& Ctx, mochi::Scene& Scene, FDrivenBot& Bot, mochi::real Dt) {
  mochi::Actor* Actor = Scene.GetActor(Bot.Actor);
  if (!Actor) {
    return;
  }

  if (Bot.PoseFollower.IsValid()) {
    TOptional<sr::ControllerMochiArticulatedPose::Target> Target;
    {
      FScopeLock Lock(&Bot.PoseTargetMutex);
      Target = Bot.PoseTarget;
    }
    if (sr::ControllerMochiArticulatedPose* Follower =
            static_cast<sr::ControllerMochiArticulatedPose*>(Ctx.GetController(Bot.PoseFollower));
        Follower && Target.IsSet()) {
      superdex::Error Error;
      Follower->ComputeOutput({}, Target.GetValue(), Error);
    }
  }
  if (Bot.Controllers.IsEmpty()) {
    return;
  }

  FTransform ArmPose;
  const bool bArmPose = Bot.Operator && Bot.Operator->GetLastPublishedArmPoseTarget(ArmPose);
  const bool bHandJoints =
      Bot.Operator && Bot.Operator->GetLastPublishedHandJointTargets(Bot.HandJoints);
  const bool bArmJoints = Bot.Operator && !Bot.bArmPosed &&
      Bot.Operator->GetLastPublishedArmJointTargets(Bot.ArmJoints);

  const int32 NumDofs = Actor->GetNumDofs();
  Bot.Tau.Init(0, NumDofs);
  Bot.Efforts.SetNumUninitialized(NumDofs);
  bool bDriven = false;
  for (FDrivenController& Driven : Bot.Controllers) {
    sr::ControllerBase* Controller = Ctx.GetController(Driven.Handle);
    if (!Controller) {
      continue;
    }
    const int32 InputSize = Controller->GetInputSize();
    superdex::Error Error;
    if (!Driven.bHasHeld) {
      Driven.Held.SetNumZeroed(InputSize);
      Controller->GetCurrentInput(
          mochi::Span<mochi::real>(Driven.Held.GetData(), InputSize), Error);
      if (!Error.IsOK()) {
        continue;
      }
      Driven.bHasHeld = true;
    }

    if (Driven.Kind == sr::ControllerInputKind::Pose) {
      if (bArmPose) {
        const mochi::TransformRT RootFromTarget =
            FUnrealToMochiConverter().TransformToMochi_RT(ArmPose);
        const mochi::Real3 Translation = RootFromTarget.GetTranslation();
        for (int32 i = 0; i < 3; ++i) {
          Driven.Held[i] = Translation[i];
        }
        for (int32 i = 0; i < 4; ++i) {
          Driven.Held[3 + i] = RootFromTarget.GetRotation().data[i];
        }
      }
      Bot.Input = Driven.Held;
    } else {
      // Arm DOFs a pose controller drives track the arm's current positions, so this controller
      // does not hold the arm against it. Teleop joint targets exclude a free base, whose DOFs
      // stay where they are held.
      Bot.Input.SetNumUninitialized(InputSize);
      Controller->GetCurrentInput(mochi::Span<mochi::real>(Bot.Input.GetData(), InputSize), Error);
      for (int32 k = 0; k < InputSize && k < Driven.Dofs.Num(); ++k) {
        const int32 Joint = Driven.Dofs[k] - Bot.NumBaseDofs;
        if (Joint >= 0 && Joint < Bot.NumArmJoints) {
          if (Bot.bArmPosed) {
            continue;
          }
          if (bArmJoints && Joint < Bot.ArmJoints.Num()) {
            Driven.Held[k] = static_cast<mochi::real>(Bot.ArmJoints[Joint]);
          }
        } else if (const int32 Hand = Joint - Bot.NumArmJoints;
                   Joint >= 0 && bHandJoints && Hand < Bot.HandJoints.Num()) {
          Driven.Held[k] = static_cast<mochi::real>(Bot.HandJoints[Hand]);
        }
        Bot.Input[k] = Driven.Held[k];
      }
    }

    // Writes every DOF's effort, zero on DOFs the controller does not control.
    Controller->ComputeOutputFromInput(
        mochi::Span<mochi::real const>(Bot.Input.GetData(), InputSize),
        Dt,
        mochi::Span<mochi::real>(Bot.Efforts.GetData(), NumDofs),
        Error);
    if (!Error.IsOK()) {
      continue;
    }
    for (int32 Dof = 0; Dof < NumDofs; ++Dof) {
      Bot.Tau[Dof] += Bot.Efforts[Dof];
    }
    bDriven = true;
  }
  if (bDriven) {
    if (Bot.Indices.Num() != NumDofs) {
      Bot.Indices.SetNumUninitialized(NumDofs);
      for (int32 Dof = 0; Dof < NumDofs; ++Dof) {
        Bot.Indices[Dof] = Dof;
      }
    }
    mochi::Error Error;
    Actor->SetExternalForcesOnDofs(
        mochi::Span<int const>(Bot.Indices.GetData(), NumDofs),
        mochi::Span<mochi::real const>(Bot.Tau.GetData(), NumDofs),
        Error);
  } else {
    // External forces persist until replaced, so a step without output must not keep the last.
    Actor->ClearExternalForces();
  }
  ReportState(*Actor, Bot);
}

} // namespace

struct FSuperDexRuntimeSceneLoadExampleState {
  TArray<TSharedPtr<FDrivenBot>> Bots;
  mochi::CallbackHandle StepCallback;
  /** A restart's BeginEpisode is queued for the next tick. */
  bool bRestartPending = false;
  /** Each bot's pose when the scene first brought it up, which later episodes restore. */
  TMap<FString, TArray<mochi::real>> InitialPoses;
  /** Each selected bot of a slot, by slot. */
  TMap<FString, FString> SelectedBots;
  /** Preparation can succeed even when a selected bot later fails to load. */
  TSet<FString> PreparedBots;
  TWeakObjectPtr<UMochiSceneObject> PreparedSceneObject;
  TArray<TPair<mochi::ActorHandle, FString>> SpawnNames;
  TMap<FString, FString> SpawnPrefabs;
#if WITH_SUPERDEX_RECORDER
  TUniquePtr<FSuperDexSceneRecorder> Recorder;
#endif
};

ASuperDexRuntimeSceneLoadExample::ASuperDexRuntimeSceneLoadExample() {
  PrimaryActorTick.bCanEverTick = true;
  RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
  Visualization =
      CreateDefaultSubobject<USuperDexBotSceneVisualizationComponent>(TEXT("SceneVisualization"));
  Visualization->SetupAttachment(RootComponent);
  Visualization->SetMobility(EComponentMobility::Movable);

  auto AddSlot = [this](const TCHAR* Name, ESuperDexHandedness Handedness, const FInputChord& Key) {
    FSuperDexRuntimeSceneLoadExampleSlot& Slot = Slots.AddDefaulted_GetRef();
    Slot.Name = Name;
    Slot.Handedness = Handedness;
    Slot.CycleBotKey = Key;
  };
  AddSlot(
      TEXT("left"),
      ESuperDexHandedness::Left,
      FInputChord(EKeys::A, /*bShift=*/false, /*bCtrl=*/true, /*bAlt=*/false, /*bCmd=*/false));
  AddSlot(
      TEXT("right"),
      ESuperDexHandedness::Right,
      FInputChord(EKeys::A, /*bShift=*/false, /*bCtrl=*/false, /*bAlt=*/true, /*bCmd=*/false));
  PreviousTaskKey = FInputChord(EKeys::B);
  NextTaskKey = FInputChord(EKeys::N);
  ResetTaskKey = FInputChord(EKeys::M);
  RecordOrKeepKey = FInputChord(EKeys::Enter);
  DiscardKey = FInputChord(EKeys::BackSpace);
}

void ASuperDexRuntimeSceneLoadExample::PostInitializeComponents() {
  Super::PostInitializeComponents();
  UWorld* World = GetWorld();
  if (!World || !World->IsGameWorld()) {
    return;
  }
  const FString TaskPath = ResolveProjectPath(CommandLineValue(
      TEXT("SuperDexTaskFolder="), CommandLineValue(TEXT("SuperDexTaskFile="), TaskFile.FilePath)));
  TaskFiles = ListTaskFiles(TaskPath);
  // Before any actor's BeginPlay: the subsystem loads into a scene that holds nothing yet.
  if (USuperDexBotSceneSubsystem* Subsystem = GetSceneSubsystem(*this)) {
    Subsystem->EnsureLoaded(
        ResolveProjectPath(SceneFile.FilePath), TaskFiles.IsEmpty() ? FString() : TaskFiles[0]);
  }
}

void ASuperDexRuntimeSceneLoadExample::BeginPlay() {
  Super::BeginPlay();
  State = MakeShared<FSuperDexRuntimeSceneLoadExampleState>();
  USuperDexBotSceneSubsystem* Subsystem = GetSceneSubsystem(*this);
  if (!Subsystem || Subsystem->GetState() != ESuperDexBotSceneState::Loaded) {
    Fail(
        FString::Printf(
            TEXT("No SuperDex scene loaded: %s"),
            Subsystem ? *Subsystem->GetErrorMessageOrDefault() : TEXT("no scene subsystem")));
    return;
  }
#if WITH_SUPERDEX_RECORDER
  if (bRecord || FParse::Param(FCommandLine::Get(), TEXT("SuperDexRecord"))) {
    State->Recorder = MakeUnique<FSuperDexSceneRecorder>();
  }
#endif
  BindKeys();
  BeginEpisode();
}

void ASuperDexRuntimeSceneLoadExample::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  if (State.IsValid()) {
#if WITH_SUPERDEX_RECORDER
    if (State->Recorder && State->Recorder->IsRecording()) {
      State->Recorder->CommitEpisodeSync();
    }
#endif
    State->bRestartPending = false;
    EndEpisode();
  }
  Super::EndPlay(EndPlayReason);
}

void ASuperDexRuntimeSceneLoadExample::BindKeys() {
  APlayerController* PlayerController = GetWorld()->GetFirstPlayerController();
  if (!PlayerController) {
    return;
  }
  EnableInput(PlayerController);
  if (!InputComponent) {
    return;
  }
  for (const FSuperDexRuntimeSceneLoadExampleSlot& Slot : Slots) {
    if (Slot.CycleBotKey.Key.IsValid()) {
      InputComponent->KeyBindings.Emplace_GetRef(Slot.CycleBotKey, IE_Pressed)
          .KeyDelegate.GetDelegateForManualSet()
          .BindUObject(this, &ASuperDexRuntimeSceneLoadExample::CycleSlotBot, Slot.Name);
    }
  }
  using FAction = void (ASuperDexRuntimeSceneLoadExample::*)();
  const TPair<const FInputChord*, FAction> Actions[] = {
      {&PreviousTaskKey, &ASuperDexRuntimeSceneLoadExample::PreviousTask},
      {&NextTaskKey, &ASuperDexRuntimeSceneLoadExample::NextTask},
      {&ResetTaskKey, &ASuperDexRuntimeSceneLoadExample::ResetTask},
      {&RecordOrKeepKey, &ASuperDexRuntimeSceneLoadExample::RecordOrKeep},
      {&DiscardKey, &ASuperDexRuntimeSceneLoadExample::Discard},
  };
  for (const TPair<const FInputChord*, FAction>& Action : Actions) {
    if (Action.Key->Key.IsValid()) {
      InputComponent->BindKey(*Action.Key, IE_Pressed, this, Action.Value);
    }
  }
}

void ASuperDexRuntimeSceneLoadExample::BeginEpisode() {
  USuperDexBotSceneSubsystem* Subsystem = GetSceneSubsystem(*this);
  sr::BotScene* Scene = Subsystem ? Subsystem->GetBotScene() : nullptr;
  UMochiSceneObject* SceneObject = GetMochiScene(*this);
  sr::RoboticsContext* Ctx = ISuperDexRoboticsModule::Get()
      ? ISuperDexRoboticsModule::Get()->GetRoboticsContext()
      : nullptr;
  if (!Scene || !SceneObject || !Ctx) {
    return;
  }
  if (State->StepCallback.IsValid()) {
    EndEpisode();
  }
  const sr::BotScenePrefab& ScenePrefab = Scene->GetBotScenePrefab();

  // The bots this episode brings up: each slot's selected bot, and every bot in no slot.
  State->SelectedBots.Reset();
  TSet<FString> SlotBots;
  for (const sr::SlotEntry& Slot : ScenePrefab.slots) {
    const FString SlotName = ToFString(Slot.name);
    for (const superdex::DynamicString& Bot : Slot.controlledBots) {
      SlotBots.Add(ToFString(Bot));
    }
    if (!Slot.controlledBots.empty()) {
      const int32 Index =
          SlotSelection.FindOrAdd(SlotName) % static_cast<int32>(Slot.controlledBots.size());
      State->SelectedBots.Add(SlotName, ToFString(Slot.controlledBots[Index]));
    }
  }
  TSet<FString> Wanted;
  for (const sr::BotEntry& Entry : ScenePrefab.bots) {
    const FString Name = ToFString(Entry.name);
    if (!SlotBots.Contains(Name) || State->SelectedBots.FindKey(Name)) {
      Wanted.Add(Name);
    }
  }

  // Bring them up in their initial poses, and read what driving each slot's bot takes. The scene
  // steps asynchronously, so everything read off it is read on the scene thread.
  struct FSlotBot {
    const sr::BotEntry* Entry = nullptr;
    sr::Bot* Bot = nullptr;
    mochi::TransformRT WorldFromBase;
    TSharedPtr<FDrivenBot> Driven;
  };
  TMap<FString, FSlotBot> SlotBotsUp;
  TArray<FString> Messages;
  SceneObject->QueueOrExecuteCommand([&](mochi::Scene*) {
    // Keep loaded bots when this episode chooses the same prepared set.
    superdex::Error SelectionError;
    superdex::DynamicArray<superdex::DynamicString> SelectedNames;
    for (const sr::BotEntry& Entry : ScenePrefab.bots) {
      const bool bWanted = Wanted.Contains(ToFString(Entry.name));
      if (bWanted) {
        SelectedNames.push_back(Entry.name);
      }
    }
    if (State->PreparedSceneObject.Get() != SceneObject ||
        State->PreparedBots.Num() != Wanted.Num() || !State->PreparedBots.Includes(Wanted)) {
      for (const sr::BotEntry& Entry : ScenePrefab.bots) {
        const std::string_view Name(Entry.name);
        if (Scene->GetBot(Name)) {
          Scene->RemoveBot(Name, SelectionError);
        }
      }
    }
    Scene->PrepareBots(mochi::MakeConstSpan(SelectedNames), SelectionError);
    if (!SelectionError.IsOK()) {
      Messages.Add(
          FString::Printf(
              TEXT("Bot selection could not be prepared: %s"),
              UTF8_TO_TCHAR(SelectionError.ToString().c_str())));
      return;
    }
    State->PreparedBots = Wanted;
    State->PreparedSceneObject = SceneObject;
    for (const sr::BotEntry& Entry : ScenePrefab.bots) {
      const std::string_view Name(Entry.name);
      const FString BotName = ToFString(Entry.name);
      superdex::Error Error;
      sr::Bot* Bot = Scene->GetBot(Name);
      if (!Wanted.Contains(BotName)) {
        if (Bot) {
          Scene->RemoveBot(Name, Error);
        }
        continue;
      }
      const bool bJustBroughtUp = !Bot;
      if (bJustBroughtUp) {
        Bot = Scene->InitializeBot(Name, Error);
      }
      mochi::Actor* Actor = Bot ? Bot->GetArticulatedActor() : nullptr;
      if (!Actor) {
        Messages.Add(
            FString::Printf(
                TEXT("Bot '%s' did not come up: %s"),
                *BotName,
                UTF8_TO_TCHAR(Error.ToString().c_str())));
        continue;
      }
      RestoreInitialPose(*Actor, State->InitialPoses.FindOrAdd(BotName), bJustBroughtUp);
      for (const sr::ControllerEntry& Controller : Entry.controllers) {
        if (sr::ControllerBase* Base =
                Ctx->GetController(Scene->GetControllerHandle(ControllerKey(Entry, Controller)))) {
          Base->Reset();
        }
      }
      if (!State->SelectedBots.FindKey(BotName)) {
        continue;
      }
      const sr::BotPrefab& Prefab = Bot->GetBotPrefab();
      TSharedPtr<FDrivenBot> Driven = MakeShared<FDrivenBot>();
      Driven->Name = BotName;
      Driven->Actor = Actor->GetHandle();
      Driven->ActorName = UTF8_TO_TCHAR(Actor->GetName());
      Driven->NumBaseDofs =
          Actor->GetNumDofs() - ExtractArticulatedActorState(Actor).JointPositions.Num();
      Driven->PoseGeometry = FMochiArticulatedPoseGeometry::FromBotPrefab(Prefab, Prefab.scale);
      AddControllers(*Driven, Entry, *Scene, *Ctx, Messages);
      SlotBotsUp.Add(BotName, FSlotBot{&Entry, Bot, Actor->GetRootTransform(), Driven});
    }
  });
  SceneObject->WaitForQueuedCommands();
  for (const FString& Message : Messages) {
    Warn(Message);
  }
  Subsystem->NotifySceneContentsChanged();

  // A roboperator for each slot's bot, at the bot's base: its arm pose target is relative to it.
  for (const TPair<FString, FString>& Selected : State->SelectedBots) {
    const FString& BotName = Selected.Value;
    const FSlotBot* Up = SlotBotsUp.Find(BotName);
    if (!Up) {
      continue;
    }
    const FString ClassKey = ToFString(Up->Entry->roboperatorClass);
    const TSubclassOf<ATeleopRoboperatorActor>* Class =
        Roboperators.FindByPredicate([&ClassKey](const TSubclassOf<ATeleopRoboperatorActor>& C) {
          FString Name = C ? C->GetName() : FString();
          Name.RemoveFromEnd(TEXT("_C"));
          return C && (Name == ClassKey || C->GetPathName() == ClassKey);
        });
    if (!Class) {
      Warn(
          FString::Printf(
              TEXT("No roboperator named '%s' for bot '%s'; it is not driven."),
              *ClassKey,
              *BotName));
      continue;
    }
    const FTransform SpawnTransform =
        FMochiToUnrealConverter().TransformToUnreal(Up->WorldFromBase);
    ATeleopRoboperatorActor* Roboperator = GetWorld()->SpawnActorDeferred<ATeleopRoboperatorActor>(
        Class->Get(),
        SpawnTransform,
        this,
        nullptr,
        ESpawnActorCollisionHandlingMethod::AlwaysSpawn);
    if (!Roboperator) {
      continue;
    }
    if (const FSuperDexRuntimeSceneLoadExampleSlot* Slot =
            Slots.FindByPredicate([&Selected](const FSuperDexRuntimeSceneLoadExampleSlot& S) {
              return S.Name == Selected.Key;
            })) {
      Roboperator->Handedness = Slot->Handedness;
    }
    Roboperator->FinishSpawning(SpawnTransform);
    SpawnedRoboperators.Add(Roboperator);

    // The roboperator's bot component poses the scene bot, which its retargeters read.
    USuperDexBotComponent* BotComponent =
        Roboperator->FindComponentByClass<USuperDexBotComponent>();
    if (!BotComponent || !BotComponent->BindToExistingBot(Up->Bot)) {
      Warn(
          FString::Printf(
              TEXT("%s could not bind to bot '%s'; its tracking will not engage."),
              *Roboperator->GetName(),
              *BotName));
    }

    FDrivenBot& Driven = *Up->Driven;
    Driven.Operator = Roboperator->TeleopNodeOperator.Get();
    Driven.Robot = Roboperator->TeleopNodeRobot.Get();
    Driven.NumArmJoints = Driven.Operator ? Driven.Operator->NumArmJoints : 0;
    Driven.NumHandJoints = Driven.Robot ? Driven.Robot->NumHandJoints : 0;
    if (Driven.PoseFollower.IsValid() && BotComponent) {
      Driven.PoseSource =
          UController_MochiArticulatedPose::FindChildSourcePoseable(BotComponent, FString());
      Driven.PoseGeometry.Bot = BotComponent->GetBotAsset();
    }
    UE_LOG(
        LogSuperDexRuntimeSceneLoadExample,
        Display,
        TEXT("Driving bot '%s' with %s: %d controller(s)%s."),
        *BotName,
        *Roboperator->GetClass()->GetName(),
        Driven.Controllers.Num(),
        Driven.PoseFollower.IsValid() ? TEXT(" and a pose follower") : TEXT(""));
    SUPERDEX_SCREEN_PRINTF("%s: %s", *Selected.Key, *BotName);
    State->Bots.Add(Up->Driven);
  }

  SpawnTaskObjects();

  TArray<TSharedPtr<FDrivenBot>> Bots = State->Bots;
  State->StepCallback = SceneObject->RegisterPreStepCallback(
      "SuperDexRuntimeSceneLoadExample", [Bots, Ctx](mochi::StepInfo const& Info) {
        if (!Info.scene) {
          return;
        }
        for (const TSharedPtr<FDrivenBot>& Bot : Bots) {
          DriveBot(*Ctx, *Info.scene, *Bot, static_cast<mochi::real>(Info.timeStepSec));
        }
      });
}

void ASuperDexRuntimeSceneLoadExample::EndEpisode() {
  if (UMochiSceneObject* SceneObject = GetMochiScene(*this);
      SceneObject && State->StepCallback.IsValid()) {
    SceneObject->CancelCallback(State->StepCallback);
    // The bots must not keep the last step's efforts until the next episode drives them.
    TArray<mochi::ActorHandle> Actors;
    for (const TSharedPtr<FDrivenBot>& Bot : State->Bots) {
      Actors.Add(Bot->Actor);
    }
    SceneObject->QueueOrExecuteCommand([Actors](mochi::Scene* Scene) {
      for (const mochi::ActorHandle Handle : Actors) {
        if (mochi::Actor* Actor = Scene ? Scene->GetActor(Handle) : nullptr) {
          Actor->ClearExternalForces();
        }
      }
    });
    SceneObject->WaitForQueuedCommands();
  }
  State->StepCallback = {};
  State->Bots.Reset();
  for (ATeleopRoboperatorActor* Roboperator : SpawnedRoboperators) {
    if (IsValid(Roboperator)) {
      Roboperator->Destroy();
    }
  }
  SpawnedRoboperators.Reset();
  if (USuperDexBotSceneSubsystem* Subsystem = GetSceneSubsystem(*this)) {
    Subsystem->DestroyAllTaskInstances();
  }
}

void ASuperDexRuntimeSceneLoadExample::RestartEpisode() {
  EndEpisode();
  if (State->bRestartPending) {
    return;
  }
  // A tick later, so the destroyed actors are gone before the next episode's spawn. Requests until
  // then share that one restart.
  State->bRestartPending = true;
  TWeakObjectPtr<ASuperDexRuntimeSceneLoadExample> WeakThis(this);
  GetWorld()->GetTimerManager().SetTimerForNextTick([WeakThis]() {
    ASuperDexRuntimeSceneLoadExample* Self = WeakThis.Get();
    if (Self && Self->State.IsValid() && Self->State->bRestartPending) {
      Self->State->bRestartPending = false;
      Self->BeginEpisode();
    }
  });
}

void ASuperDexRuntimeSceneLoadExample::SpawnTaskObjects() {
  USuperDexBotSceneSubsystem* Subsystem = GetSceneSubsystem(*this);
  const sr::BotScene* Scene = Subsystem ? Subsystem->GetBotScene() : nullptr;
  State->SpawnNames.Reset();
  State->SpawnPrefabs.Reset();
  if (!Scene || !TaskFiles.IsValidIndex(TaskIndex)) {
    return;
  }
  auto Failed = [this](const superdex::Error& Error) {
    Fail(
        FString::Printf(
            TEXT("Task '%s': %s"), *TaskFiles[TaskIndex], UTF8_TO_TCHAR(Error.ToString().c_str())));
  };
  superdex::Error Error;
  const sr::BotTaskPrefab Task =
      sr::LoadBotTaskPrefabFromFile(ToStdString(TaskFiles[TaskIndex]), Error);
  if (!Error.IsOK()) {
    return Failed(Error);
  }
  const superdex::DynamicArray<sr::ResolvedSpawn> Spawns = sr::ResolveResetState(
      Task, {}, sr::ComputeWorldFromTask(Scene->GetBotScenePrefab().scene, Task), Error);
  if (!Error.IsOK()) {
    return Failed(Error);
  }
  for (const sr::ResolvedSpawn& Spawn : Spawns) {
    const FString Name = ToFString(Spawn.name);
    const FString Prefab = ToFString(Spawn.prefabName);
    TArray<mochi::ActorHandle> Actors;
    FString SpawnError;
    if (!Subsystem->SpawnTaskInstance(Name, Prefab, Spawn.worldFromSpawn, Actors, SpawnError)) {
      Fail(FString::Printf(TEXT("Task spawn '%s': %s"), *Name, *SpawnError));
      continue;
    }
    State->SpawnPrefabs.Add(Name, Prefab);
    for (const mochi::ActorHandle Actor : Actors) {
      State->SpawnNames.Emplace(Actor, Name);
    }
  }
  Subsystem->NotifySceneContentsChanged();
  Announce(TEXT("Task: ") + GetTaskName());
}

void ASuperDexRuntimeSceneLoadExample::Tick(float DeltaSeconds) {
  Super::Tick(DeltaSeconds);
  if (!State.IsValid()) {
    return;
  }
  // The step callback drives the roboperators' components, so it stops before they are collected.
  for (const TObjectPtr<ATeleopRoboperatorActor>& Roboperator : SpawnedRoboperators) {
    if (!IsValid(Roboperator)) {
      Warn(TEXT("A roboperator went away; restarting the episode."));
      StopRecording(/*bKeep=*/false);
      RestartEpisode();
      return;
    }
  }
  if (IsRecording()) {
    SUPERDEX_SCREEN_PRINTF_KC(
        GetUniqueID(), FColor::Red, "Recording (Enter keeps, Backspace discards)");
  }
  // A pose follower's target is built here, on the game thread, from its tracked-hand poseable.
  for (const TSharedPtr<FDrivenBot>& Bot : State->Bots) {
    const USuperDexPoseableComponent* Source = Bot->PoseSource.Get();
    FSuperDexPose Pose;
    if (!Bot->PoseFollower.IsValid() || !Source || !Source->GetOwner() ||
        !Source->GetCachedPose(Pose)) {
      continue;
    }
    sr::ControllerMochiArticulatedPose::Target Target;
    if (UController_MochiArticulatedPose::BuildPoseTarget(
            Pose, Source->GetOwner()->GetActorTransform(), Bot->PoseGeometry, Target) !=
        EMochiPoseTargetResult::Built) {
      continue;
    }
    FScopeLock Lock(&Bot->PoseTargetMutex);
    Bot->PoseTarget = MoveTemp(Target);
  }
}

void ASuperDexRuntimeSceneLoadExample::CycleSlotBot(FString SlotName) {
  if (!State.IsValid()) {
    return;
  }
  StopRecording(/*bKeep=*/false);
  ++SlotSelection.FindOrAdd(SlotName);
  RestartEpisode();
}

void ASuperDexRuntimeSceneLoadExample::PreviousTask() {
  if (TaskFiles.Num() > 1) {
    TaskIndex = (TaskIndex + TaskFiles.Num() - 1) % TaskFiles.Num();
    ResetTask();
  }
}

void ASuperDexRuntimeSceneLoadExample::NextTask() {
  if (TaskFiles.Num() > 1) {
    TaskIndex = (TaskIndex + 1) % TaskFiles.Num();
    ResetTask();
  }
}

void ASuperDexRuntimeSceneLoadExample::ResetTask() {
  // A pending restart spawns the current task's objects itself.
  if (!State.IsValid() || State->bRestartPending) {
    return;
  }
  // A recording cannot outlive the objects it records.
  StopRecording(/*bKeep=*/false);
  if (USuperDexBotSceneSubsystem* Subsystem = GetSceneSubsystem(*this)) {
    Subsystem->DestroyAllTaskInstances();
  }
  SpawnTaskObjects();
}

void ASuperDexRuntimeSceneLoadExample::RecordOrKeep() {
  if (IsRecording()) {
    StopRecording(/*bKeep=*/true);
    RestartEpisode();
    return;
  }
#if WITH_SUPERDEX_RECORDER
  USuperDexBotSceneSubsystem* Subsystem = GetSceneSubsystem(*this);
  if (!State.IsValid() || !State->Recorder || !Subsystem) {
    Warn(TEXT("Recording is off: tick Record on the example actor or pass -SuperDexRecord."));
    return;
  }
  if (State->bRestartPending) {
    return;
  }
  TSet<FString> RobotNames;
  for (const TSharedPtr<FDrivenBot>& Bot : State->Bots) {
    bool bAlreadyRecorded = false;
    RobotNames.Add(Bot->ActorName, &bAlreadyRecorded);
    if (bAlreadyRecorded) {
      Fail(
          FString::Printf(
              TEXT(
                  "Cannot record two '%s' bots at once, which the recorder names alike; select another bot in one slot."),
              *Bot->ActorName));
      return;
    }
  }
  FSuperDexSceneRecorderConfig Config;
  Config.OutputDirectory = CommandLineValue(TEXT("SuperDexRecordDir="), RecordingDirectory.Path);
  // The recorder stores the scene it was given by name and hash, under its archive field.
  Config.SceneArchivePath = Subsystem->GetScenePath();
  Config.TaskPath = TaskFiles.IsValidIndex(TaskIndex) ? TaskFiles[TaskIndex] : FString();
  Config.bRecordContactForces =
      bRecordContactForces || FParse::Param(FCommandLine::Get(), TEXT("SuperDexRecordContacts"));
  Config.SpawnNames = State->SpawnNames;
  Config.Properties.Add(TEXT("source"), TEXT("unreal.superdex"));
  Config.Properties.Add(TEXT("unreal.project"), FApp::GetProjectName());
  Config.Properties.Add(TEXT("unreal.map"), GetWorld()->GetMapName());
  for (const TPair<FString, FString>& Spawn : State->SpawnPrefabs) {
    Config.Properties.Add(TEXT("scene.prefab.") + Spawn.Key, Spawn.Value);
  }
  for (const TPair<FString, FString>& Selected : State->SelectedBots) {
    Config.Properties.Add(TEXT("scene.slot.") + Selected.Key, Selected.Value);
  }
  if (State->Recorder->BeginEpisode(this, Config)) {
    Announce(TEXT("Recording to ") + State->Recorder->GetOutputPath(), FColor::Red);
  } else {
    Fail(TEXT("Recording did not start; see the log."));
  }
#endif
}

void ASuperDexRuntimeSceneLoadExample::Discard() {
  if (IsRecording()) {
    StopRecording(/*bKeep=*/false);
    RestartEpisode();
  }
}

void ASuperDexRuntimeSceneLoadExample::StopRecording(bool bKeep) {
#if WITH_SUPERDEX_RECORDER
  if (!IsRecording()) {
    return;
  }
  if (bKeep) {
    const FString OutputPath = State->Recorder->GetOutputPath();
    if (State->Recorder->CommitEpisodeAsync()) {
      Announce(TEXT("Episode kept; saving it to ") + OutputPath, FColor::Green);
    } else {
      Fail(TEXT("The episode could not be kept; see the log."));
    }
  } else {
    State->Recorder->AbortEpisode(TEXT("Episode discarded."));
    Announce(TEXT("Episode discarded."), FColor::Yellow);
  }
  if (GEngine) {
    GEngine->RemoveOnScreenDebugMessage(GetUniqueID());
  }
#endif
}

bool ASuperDexRuntimeSceneLoadExample::IsRecording() const {
#if WITH_SUPERDEX_RECORDER
  return State.IsValid() && State->Recorder && State->Recorder->IsRecording();
#else
  return false;
#endif
}

FString ASuperDexRuntimeSceneLoadExample::GetTaskName() const {
  return TaskFiles.IsValidIndex(TaskIndex) ? FPaths::GetBaseFilename(TaskFiles[TaskIndex])
                                           : FString();
}
