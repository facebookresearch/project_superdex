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
  UMochiSceneObject* mochiSceneObj = NewObject<UMochiSceneObject>(Outer, FName(*NameStr));
  check(mochiSceneObj);

  // Get MochiPhysics world
  mochi::Context* mochiContext = IMochiPhysicsModule::Get()->GetMochiContext();
  check(mochiContext != nullptr);

  // Create MochiScene
  if (SceneType == EMochiSceneType::Async) {
    mochiSceneObj->MochiAsyncScene =
        mochiContext->CreateAsyncScene(TCHAR_TO_ANSI(*NameStr), mochi::ErrorAssert{});
    mochiSceneObj->MochiAsyncScene->Pause(bStartPaused);
  } else {
    mochiSceneObj->MochiScene = mochiContext->CreateScene(TCHAR_TO_ANSI(*NameStr));
  }
  check(mochiSceneObj->IsValid());

  // Init contact layers, configure settings, and register callbacks
  mochiSceneObj->InitDefaultContactLayers();
  mochiSceneObj->ConfigureSettings(InSettings);
  mochiSceneObj->RegisterDebugDrawAndStatsCallbacks();

  return mochiSceneObj;
}

void UMochiSceneObject::DestroyScene() {
  MOCHI_PROFILE_SCOPE();
  if (!bDestructionOwnedByFriend) {
    mochi::Context* mochiContext = IMochiPhysicsModule::Get()->GetMochiContext();
    check(mochiContext);
    if (MochiAsyncScene) {
      mochiContext->DestroyAsyncScene(MochiAsyncScene);
    }
    if (MochiScene) {
      mochiContext->DestroyScene(MochiScene);
    }
  }
  MochiAsyncScene = nullptr;
  MochiScene = nullptr;
}

void UMochiSceneObject::QueueOrExecuteCommand(std::function<void(mochi::Scene*)> callback) const {
  if (MochiAsyncScene) {
    MochiAsyncScene->QueueCommand(callback);
  } else if (MochiScene) {
    callback(MochiScene);
  }
}

void UMochiSceneObject::QueueOrExecuteActorCommand(
    mochi::ActorHandle actor,
    std::function<void(mochi::Actor*)> callback) const {
  if (MochiAsyncScene) {
    MochiAsyncScene->QueueActorCommand(actor, callback);
  } else if (MochiScene) {
    mochi::Actor* actorPtr = MochiScene->GetActor(actor);
    if (actorPtr) {
      callback(actorPtr);
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
    std::string_view debugName,
    std::function<void(mochi::StepInfo const&)> fn,
    int priority) const {
  if (MochiAsyncScene) {
    return MochiAsyncScene->RegisterPreStepCallback(debugName, fn, priority);
  } else if (MochiScene) {
    return MochiScene->RegisterPreStepCallback(debugName, fn, priority);
  }
  return mochi::CallbackHandle{};
}

mochi::CallbackHandle UMochiSceneObject::RegisterPostStepCallback(
    std::string_view debugName,
    std::function<void(mochi::StepInfo const&)> fn,
    int priority) const {
  if (MochiAsyncScene) {
    return MochiAsyncScene->RegisterPostStepCallback(debugName, fn, priority);
  } else if (MochiScene) {
    return MochiScene->RegisterPostStepCallback(debugName, fn, priority);
  }
  return mochi::CallbackHandle{};
}

void UMochiSceneObject::CancelCallback(mochi::CallbackHandle handle) const {
  if (MochiAsyncScene) {
    MochiAsyncScene->CancelCallback(handle);
  } else if (MochiScene) {
    MochiScene->CancelCallback(handle);
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
  QueueOrExecuteCommand([this, GravityVector](mochi::Scene* scene) {
    FUnrealToMochiConverter u2m;
    // NB: Gravity vector is represented as m/s^2 in UE; apply no scale
    scene->SetGravity(u2m.DirectionToMochi(GravityVector));
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
  QueueOrExecuteCommand([this, Entry](mochi::Scene* scene) {
    if (Entry.LayerA.IsNone() || Entry.LayerB.IsNone()) {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "UMochiSceneObject::SetContactMatrixEntry: Invalid MochiContactMatrix entry with layer value of None!"));
      return;
    }
    const FString strA = Entry.LayerA.ToString(); // keep in scope
    const FString strB = Entry.LayerB.ToString(); // keep in scope
    auto a = StringCast<ANSICHAR>(*strA);
    auto b = StringCast<ANSICHAR>(*strB);
    if (Entry.bSymmetric) {
      scene->EnableLayerContactSymmetric(a.Get(), b.Get(), Entry.bEnabled, mochi::ErrorAssert{});
    } else {
      scene->EnableLayerContactAsymmetric(a.Get(), b.Get(), Entry.bEnabled, mochi::ErrorAssert{});
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
    auto stepParams = MochiAsyncScene->GetAsyncStepParams();
    stepParams.useFixedTimeStep = InSettings.StepParams.bUseFixedTimeStep;
    stepParams.fixedTimeStepSeconds = InSettings.StepParams.FixedTimeStepSeconds;
    stepParams.dynamicTimeStepMinSeconds = InSettings.StepParams.DynamicTimeStepMinSeconds;
    stepParams.dynamicTimeStepMaxSeconds = InSettings.StepParams.DynamicTimeStepMaxSeconds;
    MochiAsyncScene->SetAsyncStepParams(stepParams, mochi::ErrorAssert{});
  }

  ConfigureDebugDraw(InSettings.bEnableDebugDraw, InSettings.DebugDrawFlags);
}

void UMochiSceneObject::ConfigureSceneContentSettings(const FMochiSceneSettings& InSettings) {
  // get solver params
  auto solverParams = InSettings.GetSolverParams();
  if (solverParams == nullptr) {
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
        *solverParams->GetFullName())
  }

  // get contact matrix
  const auto contactMatrix = InSettings.GetContactMatrix();
  if (contactMatrix == nullptr) {
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
        *contactMatrix->GetFullName())
  }

  // Initialize Scene
  QueueOrExecuteCommand([this, InSettings, solverParams, contactMatrix](mochi::Scene* scene) {
    // Add user Contact Matrix entries
    // TODO: remove user defined contact matrix entries that were in prior settings?
    if (contactMatrix) {
      for (auto& entry : contactMatrix->Matrix) {
        if (entry.LayerA.IsNone() || entry.LayerB.IsNone()) {
          UE_LOG(
              LogMochiPhysics,
              Warning,
              TEXT(
                  "UMochiSceneObject::ConfigureSettings: Skipping MochiContactMatrix entry with layer value of None!"));
          continue;
        }
        const FString strA = entry.LayerA.ToString(); // keep in scope
        const FString strB = entry.LayerB.ToString(); // keep in scope
        std::string a = TCHAR_TO_ANSI(*strA);
        std::string b = TCHAR_TO_ANSI(*strB);
        if (entry.bSymmetric) {
          scene->EnableLayerContactSymmetric(a, b, entry.bEnabled, mochi::ErrorAssert{});
        } else {
          scene->EnableLayerContactAsymmetric(a, b, entry.bEnabled, mochi::ErrorAssert{});
        }
      }
    }
    // Set the Scene's solver parameters. These will be used for all islands in the scene.
    if (solverParams) {
      scene->SetSolverParams(solverParams->ToMochi(), mochi::ErrorAssert{});
    }
    // Set gravity vector
    FUnrealToMochiConverter u2m;
    scene->SetGravity(u2m.DirectionToMochi(InSettings.GravityVector));
  });
  WaitForQueuedCommands();
}

FMochiSceneSettings UMochiSceneObject::GetSettings() const {
  return Settings;
}

void UMochiSceneObject::DebugDrawScene(UWorld* World) {
  // Sync data from Mochi thread
  {
    FReadScopeLock lock(debugDrawDataMutex);
    debugDrawDataRender.hasChanged = false;
    if (debugDrawDataPending.hasChanged) {
      std::swap(debugDrawDataPending, debugDrawDataRender);
    }
  }

  const auto depth = Settings.DebugDrawFlags.bDebugDrawForeground ? SDPG_Foreground : 0;
  const FMochiToUnrealConverter m2u;

  auto* UnrealWorld = World ? World : GetWorld();
  if (UnrealWorld) {
    // Draw all lines
    auto const& lineVerts = debugDrawDataRender.lineVerts;
    for (size_t StartIdx = 0; StartIdx < lineVerts.positions.size(); StartIdx += 2) {
      size_t EndIdx = StartIdx + 1;
      FVector StartPos = m2u.TranslationToUnreal(lineVerts.positions[StartIdx]);
      FVector EndPos = m2u.TranslationToUnreal(lineVerts.positions[EndIdx]);
      // Scale the line length based on the multiplier.
      // NOTE: the multiplier affects all lines. So a smaller number may make contact forces look
      // good but will cause the mochi mesh visualization to be ruined.
      EndPos = (EndPos - StartPos) * Settings.DebugDraw_Line_Length_Multiplier + StartPos;
      FColor StartColor = ColorToUnreal(lineVerts.colors[StartIdx]);
      DrawDebugLine(UnrealWorld, StartPos, EndPos, StartColor, false, -1, depth);
    }

    // Draw all spheres
    auto const& spheres = debugDrawDataRender.spheres;
    for (size_t Idx = 0; Idx < spheres.positions.size(); ++Idx) {
      FVector Pos = m2u.TranslationToUnreal(spheres.positions[Idx]);
      float Radius = static_cast<float>(spheres.radii[Idx]);
      FColor Color = ColorToUnreal(spheres.colors[Idx]);
      int32 NumSegments = 32;
      DrawDebugSphere(UnrealWorld, Pos, Radius, NumSegments, Color, false, -1, depth);
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
  FReadScopeLock lock(StatsLock);

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
  FReadScopeLock lock(StatsLock);
  return Stats.numMochiStepsSinceStart;
}

double UMochiSceneObject::GetTotalSimulatedTimeSinceStartSec() const {
  FReadScopeLock lock(StatsLock);
  return Stats.totalSimulatedTimeSinceStartSec;
}

double UMochiSceneObject::GetTotalStepDuration() const {
  FReadScopeLock lock(StatsLock);
  return Stats.Performance.totalStepDurationSec;
}

void UMochiSceneObject::BeginDestroy() {
  MOCHI_PROFILE_SCOPE();
  UObject::BeginDestroy();
  DestroyScene();
}

void UMochiSceneObject::InitDefaultContactLayers() {
  // Initialize scene default contact layers
  QueueOrExecuteCommand([this](mochi::Scene* scene) {
    // left hand shouldn't collide with self
    scene->EnableLayerContactSymmetric(
        FMochiLayer::LeftHand, FMochiLayer::LeftHand, false, mochi::ErrorAssert{});
    // right hand shouldn't collide with self
    scene->EnableLayerContactSymmetric(
        FMochiLayer::RightHand, FMochiLayer::RightHand, false, mochi::ErrorAssert{});
    // left and right hands shouldn't collide with each other
    scene->EnableLayerContactSymmetric(
        FMochiLayer::LeftHand, FMochiLayer::RightHand, false, mochi::ErrorAssert{});

    // Optionally force all actors to simulate in a single island. This may be useful for debugging,
    // but it will significantly reduce CPU performance.
    const auto projSettings = GetDefault<UMochiPhysicsSettings>();
    if (projSettings && projSettings->bForceSingleIsland) {
      scene->SetForceSingleIsland(true);
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
  [[maybe_unused]] const auto debugDrawHandle = RegisterPostStepCallback(
      "DebugDraw", [this](mochi::StepInfo const& info) { // Update debug draw data
        auto const& drawData = info.scene->GetDebugDraw().GatherData();

        // Make a copy
        debugDrawDataMochi.lineVerts.positions.assign(
            drawData.lineVertices.positions.begin(), drawData.lineVertices.positions.end());
        debugDrawDataMochi.lineVerts.colors.assign(
            drawData.lineVertices.colors.begin(), drawData.lineVertices.colors.end());
        debugDrawDataMochi.spheres.positions.assign(
            drawData.spheres.positions.begin(), drawData.spheres.positions.end());
        debugDrawDataMochi.spheres.radii.assign(
            drawData.spheres.radii.begin(), drawData.spheres.radii.end());
        debugDrawDataMochi.spheres.colors.assign(
            drawData.spheres.colors.begin(), drawData.spheres.colors.end());
        debugDrawDataMochi.hasChanged = true;

        // Synchronize
        FWriteScopeLock lock(debugDrawDataMutex);
        std::swap(debugDrawDataPending, debugDrawDataMochi);
      });

  // Register callback to get stats
  [[maybe_unused]] const auto statHandle =
      RegisterPostStepCallback("Stats", [this](mochi::StepInfo const& info) {
        FWriteScopeLock lock(StatsLock);
        Stats.Performance = info.scene->GetPerformanceStats();
        Stats.NumActors = info.scene->GetNumActors();
        Stats.NumLayers = info.scene->GetNumContactLayers();
        Stats.NumConstraints = info.scene->GetNumConstraints();
        Stats.numMochiStepsSinceStart++;
        Stats.LastTimeStepSec = info.scene->GetLastTimeStep();
        Stats.totalSimulatedTimeSinceStartSec += Stats.LastTimeStepSec;
      });
}

void UMochiSceneObject::ConfigureDebugDraw(
    bool bEnableDebugDraw,
    const FMochiDebugDrawFlags& DebugDrawFlags) {
  MOCHI_PROFILE_SCOPE();
  QueueOrExecuteCommand([this, bEnableDebugDraw, DebugDrawFlags](mochi::Scene* scene) {
    mochi::DebugDraw& MochiDebugDraw = scene->GetDebugDraw();
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

void UMochiSceneObject::StartRecording(const FString& filePath) const {
  if ((MochiAsyncScene) || (MochiScene)) {
    std::string recordingPath = TCHAR_TO_ANSI(*filePath);

    if (!std::filesystem::is_directory(recordingPath)) {
      std::filesystem::create_directory(recordingPath);
    }
    constexpr char const* kRecordingName = "unrealrecording";

    // Generate a date-time string so that alphabetical sorting is also chronological
    std::time_t t = std::time(0); // now
    std::tm* now = std::localtime(&t);
    std::string timeStamp = mochi::Format(
        "%.2d%.2d%.2d_%.2d%.2d%.2d",
        now->tm_year + 1900 - 2000, // 2 digit year (eg 23 for 2023)
        now->tm_mon,
        now->tm_mday,
        now->tm_hour,
        now->tm_min,
        now->tm_sec);

    // Put it all together
    auto fullFileName = mochi::Format(
        "%s/%s_%s.recording.h5", recordingPath.c_str(), timeStamp.c_str(), kRecordingName);
    QueueOrExecuteCommand([fullFileName](mochi::Scene* scene) {
      // Use default options + enable contact point recording
      mochi::experimental::RecordingParams recordingParams;
      // recordingParams.recordContactPoints = true;

      // Attempt to open the file for writing and start recording.
      mochi::Error recordingError;
      mochi::experimental::StartRecording(scene, fullFileName, recordingParams, recordingError);
      if (!recordingError.IsOK()) {
        UE_LOG(
            LogMochiPhysics,
            Error,
            TEXT(
                "UMochiSceneObject::StartRecording: Failed to start recording to file \"%hs\"! Reason: %hs"),
            fullFileName.c_str(),
            recordingError.ToString().c_str());
      }
    });
  }
};

void UMochiSceneObject::StopRecording() const {
  QueueOrExecuteCommand(
      [](mochi::Scene* scene) { mochi::experimental::StopRecording(scene, mochi::ErrorAssert{}); });
};
