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

#include "MochiSceneObject.h"

#include "MochiConversions.h"
#include "MochiPhysicsModule.h"

#include <mochi_core/utils/profile.h>
#include <mochi_physics/mochi_physics_experimental.h>

#include <filesystem>

// clang-format off
DECLARE_STATS_GROUP(TEXT("MochiThreads"), STATGROUP_MochiThreads, STATCAT_Advanced);
DECLARE_DWORD_COUNTER_STAT(TEXT("Num Threads"), STAT_NumMochiThreads, STATGROUP_MochiThreads);

DECLARE_STATS_GROUP(TEXT("MochiScene"), STATGROUP_MochiScene, STATCAT_Advanced);
DECLARE_DWORD_COUNTER_STAT(TEXT("Num Actors"), STAT_NumMochiActors, STATGROUP_MochiScene);
DECLARE_DWORD_COUNTER_STAT(TEXT("Num Layers"), STAT_NumMochiLayers, STATGROUP_MochiScene);
DECLARE_DWORD_COUNTER_STAT(TEXT("Num Constaints"), STAT_NumMochiConstraints, STATGROUP_MochiScene);

DECLARE_STATS_GROUP(TEXT("MochiStep"), STATGROUP_MochiStep, STATCAT_Advanced);
DECLARE_FLOAT_COUNTER_STAT(TEXT("Total Step"), STAT_TotalStepDuration, STATGROUP_MochiStep);
DECLARE_FLOAT_COUNTER_STAT(TEXT("Simulated Time Step"), STAT_SimulatedTimeStep, STATGROUP_MochiStep);
DECLARE_FLOAT_COUNTER_STAT(TEXT("Pre Step"), STAT_PreStepCallbacksDuration, STATGROUP_MochiStep);
DECLARE_FLOAT_COUNTER_STAT(TEXT("Post Step"), STAT_PostStepCallbacksDuration, STATGROUP_MochiStep);

// clang-format on

bool UMochiSceneObject::IsValid() const {
  return ((MochiAsyncScene != nullptr) || (MochiScene != nullptr));
}

UMochiSceneObject* UMochiSceneObject::CreateScene(
    UObject* Outer,
    EMochiSceneType SceneType,
    const FString& SceneName,
    const FMochiSceneSettings& InSettings,
    bool bStartPaused) {
  MOCHI_PROFILE_SCOPE();
  // check that outer is valid
  if (Outer == nullptr) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("UMochiSceneObject::Failed to create UMochiSceneObject because Outer was null!"));
    return nullptr;
  }

  // Create new object
  FString NameStr =
      SceneName + ((SceneType == EMochiSceneType::Async) ? "___mochiAsyncScene" : "___mochiScene");
  UMochiSceneObject* MochiSceneObj = NewObject<UMochiSceneObject>(Outer, FName(*NameStr));
  check(MochiSceneObj);

  // Get MochiPhysics world
  mochi::Context* MochiContext = IMochiPhysicsModule::Get()->GetMochiContext();
  check(MochiContext != nullptr);

  // Create MochiScene
  if (SceneType == EMochiSceneType::Async) {
    MochiSceneObj->MochiAsyncScene =
        MochiContext->CreateAsyncScene(TCHAR_TO_ANSI(*NameStr), mochi::ErrorAssert{});
    MochiSceneObj->MochiAsyncScene->Pause(bStartPaused);
  } else {
    MochiSceneObj->MochiScene = MochiContext->CreateScene(TCHAR_TO_ANSI(*NameStr));
  }
  check(MochiSceneObj->IsValid());

  // Init contact layers, configure settings, and register callbacks
  MochiSceneObj->InitDefaultContactLayers();
  MochiSceneObj->ConfigureSettings(InSettings);
  MochiSceneObj->RegisterDebugDrawAndStatsCallbacks();

  return MochiSceneObj;
}

void UMochiSceneObject::DestroyScene() {
  MOCHI_PROFILE_SCOPE();
  if (!bDestructionOwnedByFriend) {
    mochi::Context* MochiContext = IMochiPhysicsModule::Get()->GetMochiContext();
    check(MochiContext);
    if (MochiAsyncScene) {
      MochiContext->DestroyAsyncScene(MochiAsyncScene);
    }
    if (MochiScene) {
      MochiContext->DestroyScene(MochiScene);
    }
  }
  MochiAsyncScene = nullptr;
  MochiScene = nullptr;
}

void UMochiSceneObject::QueueOrExecuteCommand(std::function<void(mochi::Scene*)> Callback) const {
  if (MochiAsyncScene) {
    MochiAsyncScene->QueueCommand(Callback);
  } else if (MochiScene) {
    Callback(MochiScene);
  }
}

void UMochiSceneObject::QueueOrExecuteActorCommand(
    mochi::ActorHandle Actor,
    std::function<void(mochi::Actor*)> Callback) const {
  if (MochiAsyncScene) {
    MochiAsyncScene->QueueActorCommand(Actor, Callback);
  } else if (MochiScene) {
    mochi::Actor* ActorPtr = MochiScene->GetActor(Actor);
    if (ActorPtr) {
      Callback(ActorPtr);
    } else {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT(
              "UMochiSceneObject::QueueOrExecuteActorCommand: Synchronous scene was passed an invalid ActorHandle! Skipping callback execution."));
    }
  }
}

void UMochiSceneObject::WaitForQueuedCommands() const {
  if (MochiAsyncScene) {
    MochiAsyncScene->WaitForQueuedCommands();
  }
}

mochi::CallbackHandle UMochiSceneObject::RegisterPreStepCallback(
    std::string_view DebugName,
    std::function<void(mochi::StepInfo const&)> Fn,
    int Priority) const {
  if (MochiAsyncScene) {
    return MochiAsyncScene->RegisterPreStepCallback(DebugName, Fn, Priority);
  } else if (MochiScene) {
    return MochiScene->RegisterPreStepCallback(DebugName, Fn, Priority);
  }
  return mochi::CallbackHandle{};
}

mochi::CallbackHandle UMochiSceneObject::RegisterPostStepCallback(
    std::string_view DebugName,
    std::function<void(mochi::StepInfo const&)> Fn,
    int Priority) const {
  if (MochiAsyncScene) {
    return MochiAsyncScene->RegisterPostStepCallback(DebugName, Fn, Priority);
  } else if (MochiScene) {
    return MochiScene->RegisterPostStepCallback(DebugName, Fn, Priority);
  }
  return mochi::CallbackHandle{};
}

void UMochiSceneObject::CancelCallback(mochi::CallbackHandle Handle) const {
  if (MochiAsyncScene) {
    MochiAsyncScene->CancelCallback(Handle);
  } else if (MochiScene) {
    MochiScene->CancelCallback(Handle);
  }
}

void UMochiSceneObject::SetGravity(const FVector& GravityVector) const {
  MOCHI_PROFILE_SCOPE();
  if (bContentsExternallyOwned) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("UMochiSceneObject::SetGravity: ignored; the scene's settings are externally owned."));
    return;
  }
  QueueOrExecuteCommand([this, GravityVector](mochi::Scene* Scene) {
    FUnrealToMochiConverter U2m;
    // NB: Gravity vector is represented as m/s^2 in UE; apply no scale
    Scene->SetGravity(U2m.DirectionToMochi(GravityVector));
  });
}

void UMochiSceneObject::SetContactMatrixEntry(const FMochiContactMatrixEntry& Entry) const {
  MOCHI_PROFILE_SCOPE();
  if (bContentsExternallyOwned) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "UMochiSceneObject::SetContactMatrixEntry: ignored; the scene's settings are externally owned."));
    return;
  }
  QueueOrExecuteCommand([this, Entry](mochi::Scene* Scene) {
    if (Entry.LayerA.IsNone() || Entry.LayerB.IsNone()) {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "UMochiSceneObject::SetContactMatrixEntry: Invalid MochiContactMatrix entry with layer value of None!"));
      return;
    }
    const FString StrA = Entry.LayerA.ToString(); // keep in scope
    const FString StrB = Entry.LayerB.ToString(); // keep in scope
    auto A = StringCast<ANSICHAR>(*StrA);
    auto B = StringCast<ANSICHAR>(*StrB);
    if (Entry.bSymmetric) {
      Scene->EnableLayerContactSymmetric(A.Get(), B.Get(), Entry.bEnabled, mochi::ErrorAssert{});
    } else {
      Scene->EnableLayerContactAsymmetric(A.Get(), B.Get(), Entry.bEnabled, mochi::ErrorAssert{});
    }
  });
}

void UMochiSceneObject::ConfigureSettings(const FMochiSceneSettings& InSettings) {
  MOCHI_PROFILE_SCOPE();
  if ((MochiAsyncScene == nullptr) && (MochiScene == nullptr)) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "UMochiSceneObject::ConfigureSettings: Attempted to configure settings on invalid scene!"));
    return;
  }
  // Cache settings
  Settings = InSettings;
  if (!bContentsExternallyOwned) {
    ConfigureSceneContentSettings(InSettings);
  } else {
    UE_LOG(
        LogMochiPhysics,
        Display,
        TEXT(
            "UMochiSceneObject::ConfigureSettings: scene contents are externally owned; keeping their gravity, solver params and contact layers."));
  }

  // These settings control the simulation time steps on the AsyncScene
  if (MochiAsyncScene) {
    MochiAsyncScene->SetAsyncStepParams(::ToMochi(InSettings.StepParams), mochi::ErrorAssert{});
  }

  ConfigureDebugDraw(InSettings.bEnableDebugDraw, InSettings.DebugDrawFlags);
}

void UMochiSceneObject::ConfigureSceneContentSettings(const FMochiSceneSettings& InSettings) {
  // get solver params
  UMochiSolverParams* SolverParams = InSettings.GetSolverParams();
  if (SolverParams == nullptr) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "UMochiSceneObject::ConfigureSettings: FMochiSceneSettings::GetSolverParams was null. Scene will be configured with default solver params."));
  } else {
    UE_LOG(
        LogMochiPhysics,
        Display,
        TEXT("UMochiSceneObject::ConfigureSettings: Using UMochiSolverParms [%s]"),
        *SolverParams->GetFullName())
  }

  // get contact matrix
  UMochiContactMatrix* const ContactMatrix = InSettings.GetContactMatrix();
  if (ContactMatrix == nullptr) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "UMochiSceneObject::ConfigureSettings: FMochiSceneSettings::GetContactMatrix was null. Scene will be configured with default contact matrix only."));
  } else {
    UE_LOG(
        LogMochiPhysics,
        Display,
        TEXT("UMochiSceneObject::ConfigureSettings: Using UMochiContactMatrix [%s]"),
        *ContactMatrix->GetFullName())
  }

  // Initialize Scene
  QueueOrExecuteCommand([this, InSettings, SolverParams, ContactMatrix](mochi::Scene* Scene) {
    // Add user Contact Matrix entries
    // TODO: remove user defined contact matrix entries that were in prior settings?
    if (ContactMatrix) {
      for (FMochiContactMatrixEntry& Entry : ContactMatrix->Matrix) {
        if (Entry.LayerA.IsNone() || Entry.LayerB.IsNone()) {
          UE_LOG(
              LogMochiPhysics,
              Warning,
              TEXT(
                  "UMochiSceneObject::ConfigureSettings: Skipping MochiContactMatrix entry with layer value of None!"));
          continue;
        }
        const FString StrA = Entry.LayerA.ToString(); // keep in scope
        const FString StrB = Entry.LayerB.ToString(); // keep in scope
        std::string A = TCHAR_TO_ANSI(*StrA);
        std::string B = TCHAR_TO_ANSI(*StrB);
        if (Entry.bSymmetric) {
          Scene->EnableLayerContactSymmetric(A, B, Entry.bEnabled, mochi::ErrorAssert{});
        } else {
          Scene->EnableLayerContactAsymmetric(A, B, Entry.bEnabled, mochi::ErrorAssert{});
        }
      }
    }
    // Set the Scene's solver parameters. These will be used for all islands in the scene.
    if (SolverParams) {
      Scene->SetSolverParams(SolverParams->ToMochi(), mochi::ErrorAssert{});
    }
    // Set gravity vector
    FUnrealToMochiConverter U2m;
    Scene->SetGravity(U2m.DirectionToMochi(InSettings.GravityVector));
  });
  WaitForQueuedCommands();
}

FMochiSceneSettings UMochiSceneObject::GetSettings() const {
  return Settings;
}

void UMochiSceneObject::DebugDrawScene(UWorld* World) {
  // Sync data from Mochi thread
  {
    FReadScopeLock Lock(DebugDrawDataMutex);
    DebugDrawDataRender.bHasChanged = false;
    if (DebugDrawDataPending.bHasChanged) {
      std::swap(DebugDrawDataPending, DebugDrawDataRender);
    }
  }

  const int Depth = Settings.DebugDrawFlags.bDebugDrawForeground ? SDPG_Foreground : 0;
  const FMochiToUnrealConverter M2u;

  UWorld* UnrealWorld = World ? World : GetWorld();
  if (UnrealWorld) {
    // Draw all lines
    const FDebugDrawLineVerticesBuffer& LineVerts = DebugDrawDataRender.LineVerts;
    for (size_t StartIdx = 0; StartIdx < LineVerts.Positions.size(); StartIdx += 2) {
      size_t EndIdx = StartIdx + 1;
      FVector StartPos = M2u.TranslationToUnreal(LineVerts.Positions[StartIdx]);
      FVector EndPos = M2u.TranslationToUnreal(LineVerts.Positions[EndIdx]);
      // Scale the line length based on the multiplier.
      // NOTE: the multiplier affects all lines. So a smaller number may make contact forces look
      // good but will cause the mochi mesh visualization to be ruined.
      EndPos = (EndPos - StartPos) * Settings.DebugDraw_Line_Length_Multiplier + StartPos;
      FColor StartColor = ColorToUnreal(LineVerts.Colors[StartIdx]);
      DrawDebugLine(UnrealWorld, StartPos, EndPos, StartColor, false, -1, Depth);
    }

    // Draw all spheres
    const FDebugDrawSpheresBuffer& Spheres = DebugDrawDataRender.Spheres;
    for (size_t Idx = 0; Idx < Spheres.Positions.size(); ++Idx) {
      FVector Pos = M2u.TranslationToUnreal(Spheres.Positions[Idx]);
      float Radius = static_cast<float>(Spheres.Radii[Idx]);
      FColor Color = ColorToUnreal(Spheres.Colors[Idx]);
      int32 NumSegments = 32;
      DrawDebugSphere(UnrealWorld, Pos, Radius, NumSegments, Color, false, -1, Depth);
    }
  }
}

bool UMochiSceneObject::IsDebugDrawEnabled() const {
  return Settings.bEnableDebugDraw;
}

FMochiDebugDrawFlags UMochiSceneObject::GetDebugDrawFlags() const {
  return Settings.DebugDrawFlags;
}

void UMochiSceneObject::SetPaused(bool bPaused) const {
  if (MochiAsyncScene) {
    return MochiAsyncScene->Pause(bPaused);
  } else if (MochiScene) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "UMochiSceneObject::SetPaused called on Mochi scene that is not Asynchonous! Step managed outside this object."));
  } else {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("UMochiSceneObject::SetPaused called on Mochi scene that is not yet created!"));
  }
}

bool UMochiSceneObject::IsPaused() const {
  if (MochiAsyncScene) {
    return MochiAsyncScene->IsPaused();
  } else if (MochiScene) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "UMochiSceneObject::IsPaused called on Mochi scene that is not Asynchonous! Step managed outside this object."));
  } else {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("UMochiSceneObject::IsPaused called on Mochi scene that is not yet created!"));
  }
  return true;
}

void UMochiSceneObject::StepPause() const {
  if (MochiAsyncScene) {
    return MochiAsyncScene->RequestStepThenPause();
  } else if (MochiScene) {
    MochiScene->Step(Settings.StepParams.FixedTimeStepSeconds);
  } else {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("UMochiSceneObject::StepPause called on Mochi scene that is not yet created!"));
  }
}

void UMochiSceneObject::UpdateStatsGroups() {
  MOCHI_PROFILE_SCOPE();
  FReadScopeLock Lock(StatsLock);

  SET_DWORD_STAT(
      STAT_NumMochiThreads, IMochiPhysicsModule::Get()->GetMochiContext()->GetNumThreads());

  SET_DWORD_STAT(STAT_NumMochiActors, Stats.NumActors);
  SET_DWORD_STAT(STAT_NumMochiLayers, Stats.NumLayers);
  SET_DWORD_STAT(STAT_NumMochiConstraints, Stats.NumConstraints);

  SET_FLOAT_STAT(STAT_TotalStepDuration, Stats.Performance.totalStepDurationSec * 1000);
  SET_FLOAT_STAT(STAT_SimulatedTimeStep, Stats.LastTimeStepSec * 1000);
  SET_FLOAT_STAT(
      STAT_PreStepCallbacksDuration, Stats.Performance.preStepCallbacksDurationSec * 1000);
  SET_FLOAT_STAT(
      STAT_PostStepCallbacksDuration, Stats.Performance.postStepCallbacksDurationSec * 1000);
}

int UMochiSceneObject::GetNumMochiStepsSinceStart() const {
  FReadScopeLock Lock(StatsLock);
  return Stats.numMochiStepsSinceStart;
}

double UMochiSceneObject::GetTotalSimulatedTimeSinceStartSec() const {
  FReadScopeLock Lock(StatsLock);
  return Stats.totalSimulatedTimeSinceStartSec;
}

double UMochiSceneObject::GetTotalStepDuration() const {
  FReadScopeLock Lock(StatsLock);
  return Stats.Performance.totalStepDurationSec;
}

void UMochiSceneObject::BeginDestroy() {
  MOCHI_PROFILE_SCOPE();
  UObject::BeginDestroy();
  DestroyScene();
}

void UMochiSceneObject::InitDefaultContactLayers() {
  // Initialize scene default contact layers
  QueueOrExecuteCommand([this](mochi::Scene* Scene) {
    // left hand shouldn't collide with self
    Scene->EnableLayerContactSymmetric(
        FMochiLayer::LeftHand, FMochiLayer::LeftHand, false, mochi::ErrorAssert{});
    // right hand shouldn't collide with self
    Scene->EnableLayerContactSymmetric(
        FMochiLayer::RightHand, FMochiLayer::RightHand, false, mochi::ErrorAssert{});
    // left and right hands shouldn't collide with each other
    Scene->EnableLayerContactSymmetric(
        FMochiLayer::LeftHand, FMochiLayer::RightHand, false, mochi::ErrorAssert{});

    // Optionally force all actors to simulate in a single island. This may be useful for debugging,
    // but it will significantly reduce CPU performance.
    const UMochiPhysicsSettings* const ProjSettings = GetDefault<UMochiPhysicsSettings>();
    if (ProjSettings && ProjSettings->bForceSingleIsland) {
      Scene->SetForceSingleIsland(true);
      UE_LOG(
          LogMochiPhysics,
          Display,
          TEXT("UMochiSceneObject::CreateAsyncScene - Forcing single simulation island (SLOW)"));
    }
  });
  WaitForQueuedCommands();
}

void UMochiSceneObject::RegisterDebugDrawAndStatsCallbacks() {
  // Register callback to render debug draw elements to Unreal
  [[maybe_unused]] const mochi::CallbackHandle DebugDrawHandle = RegisterPostStepCallback(
      "DebugDraw", [this](mochi::StepInfo const& Info) { // Update debug draw data
        const mochi::DebugDrawData& DrawData = Info.scene->GetDebugDraw().GatherData();

        // Make a copy
        DebugDrawDataMochi.LineVerts.Positions.assign(
            DrawData.lineVertices.positions.begin(), DrawData.lineVertices.positions.end());
        DebugDrawDataMochi.LineVerts.Colors.assign(
            DrawData.lineVertices.colors.begin(), DrawData.lineVertices.colors.end());
        DebugDrawDataMochi.Spheres.Positions.assign(
            DrawData.spheres.positions.begin(), DrawData.spheres.positions.end());
        DebugDrawDataMochi.Spheres.Radii.assign(
            DrawData.spheres.radii.begin(), DrawData.spheres.radii.end());
        DebugDrawDataMochi.Spheres.Colors.assign(
            DrawData.spheres.colors.begin(), DrawData.spheres.colors.end());
        DebugDrawDataMochi.bHasChanged = true;

        // Synchronize
        FWriteScopeLock Lock(DebugDrawDataMutex);
        std::swap(DebugDrawDataPending, DebugDrawDataMochi);
      });

  // Register callback to get stats
  [[maybe_unused]] const mochi::CallbackHandle StatHandle =
      RegisterPostStepCallback("Stats", [this](mochi::StepInfo const& Info) {
        FWriteScopeLock Lock(StatsLock);
        Stats.Performance = Info.scene->GetPerformanceStats();
        Stats.NumActors = Info.scene->GetNumActors();
        Stats.NumLayers = Info.scene->GetNumContactLayers();
        Stats.NumConstraints = Info.scene->GetNumConstraints();
        Stats.numMochiStepsSinceStart++;
        Stats.LastTimeStepSec = Info.scene->GetLastTimeStep();
        Stats.totalSimulatedTimeSinceStartSec += Stats.LastTimeStepSec;
      });
}

void UMochiSceneObject::ConfigureDebugDraw(
    bool bEnableDebugDraw,
    const FMochiDebugDrawFlags& DebugDrawFlags) {
  MOCHI_PROFILE_SCOPE();
  QueueOrExecuteCommand([this, bEnableDebugDraw, DebugDrawFlags](mochi::Scene* Scene) {
    mochi::DebugDraw& MochiDebugDraw = Scene->GetDebugDraw();
    MochiDebugDraw.Enable(bEnableDebugDraw);
    FMochiDebugDrawFlags::SetMochiDebugDrawFlags(MochiDebugDraw, DebugDrawFlags);
    Settings.bEnableDebugDraw = bEnableDebugDraw;
    Settings.DebugDrawFlags = DebugDrawFlags;
    // See MochiDebugDraw.h for explanation of this code.
    if (FMochiDebugDrawFlags::bGenerateAndLogMochiDebugDrawFlagsCode) {
      FMochiDebugDrawFlags::GenerateAndLogMochiDebugDrawFlagsCode(MochiDebugDraw);
    }
  });
}

void UMochiSceneObject::ConfigureDebugDraw(bool bEnableDebugDraw) {
  ConfigureDebugDraw(bEnableDebugDraw, Settings.DebugDrawFlags);
}

void UMochiSceneObject::StartRecording(const FString& FilePath) const {
  if ((MochiAsyncScene) || (MochiScene)) {
    std::string RecordingPath = TCHAR_TO_ANSI(*FilePath);

    if (!std::filesystem::is_directory(RecordingPath)) {
      std::filesystem::create_directory(RecordingPath);
    }
    constexpr char const* RecordingName = "unrealrecording";

    // Generate a date-time string so that alphabetical sorting is also chronological
    std::time_t T = std::time(0); // now
    std::tm* Now = std::localtime(&T);
    std::string TimeStamp = mochi::Format(
        "%.2d%.2d%.2d_%.2d%.2d%.2d",
        Now->tm_year + 1900 - 2000, // 2 digit year (eg 23 for 2023)
        Now->tm_mon,
        Now->tm_mday,
        Now->tm_hour,
        Now->tm_min,
        Now->tm_sec);

    // Put it all together
    std::string FullFileName = mochi::Format(
        "%s/%s_%s.recording.h5", RecordingPath.c_str(), TimeStamp.c_str(), RecordingName);
    QueueOrExecuteCommand([FullFileName](mochi::Scene* Scene) {
      // Use default options + enable contact point recording
      mochi::experimental::RecordingParams RecordingParams;
      // RecordingParams.recordContactPoints = true;

      // Attempt to open the file for writing and start recording.
      mochi::Error RecordingError;
      mochi::experimental::StartRecording(Scene, FullFileName, RecordingParams, RecordingError);
      if (!RecordingError.IsOK()) {
        UE_LOG(
            LogMochiPhysics,
            Error,
            TEXT(
                "UMochiSceneObject::StartRecording: Failed to start recording to file \"%hs\"! Reason: %hs"),
            FullFileName.c_str(),
            RecordingError.ToString().c_str());
      }
    });
  }
};

void UMochiSceneObject::StopRecording() const {
  QueueOrExecuteCommand(
      [](mochi::Scene* Scene) { mochi::experimental::StopRecording(Scene, mochi::ErrorAssert{}); });
};
