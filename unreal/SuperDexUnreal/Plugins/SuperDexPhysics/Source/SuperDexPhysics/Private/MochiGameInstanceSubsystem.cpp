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

#include "MochiGameInstanceSubsystem.h"

#include "MochiPhysicsModule.h"
#include "MochiPhysicsSettings.h"

#include "Engine/Level.h"
#include "Engine/StaticMesh.h"
#include "EngineUtils.h"
#include "Kismet/GameplayStatics.h"

#if WITH_EDITOR
#include "Editor/EditorPerformanceSettings.h"
#endif

#include <mochi_core/utils/profile.h>

#ifdef _MSC_VER
#define RT_DISABLE_OPTIMIZE 0
#if RT_DISABLE_OPTIMIZE
#pragma optimize("", off)
#endif // RT_OPTIMIZE
#endif // _MSC_VER

namespace {
template <typename T>
T* FindAsset(const TCHAR* AssetToFind) {
  auto result = ConstructorHelpers::FObjectFinder<T>(AssetToFind);
  if (result.Succeeded()) {
    UE_LOG(
        LogMochiPhysics,
        Display,
        TEXT("UMochiGameInstanceSubsystem: Found asset [%s]"),
        AssetToFind);
    return result.Object;
  }
  UE_LOG(
      LogMochiPhysics,
      Error,
      TEXT("UMochiGameInstanceSubsystem: Failed to find asset [%s]"),
      AssetToFind);
  return nullptr;
}
} // namespace

UMochiGameInstanceSubsystem::UMochiGameInstanceSubsystem() {
#if !PLATFORM_LINUX
  MOCHI_PROFILE_SCOPE();
  // TODO: ideally this would be in IMochiPhysicsModule, but FObjectFinder must
  // be called from a UObject constructor. Eventually, I imagine we'll have a
  // more sophisticated database mapping static meshes to mochi meshes.
  // find default UE mesh references
  const auto defaultUnrealMesh_Plane =
      FindAsset<UStaticMesh>(TEXT("StaticMesh'/Engine/BasicShapes/Plane.Plane'"));
  const auto defaultUnrealMesh_Cube =
      FindAsset<UStaticMesh>(TEXT("StaticMesh'/Engine/BasicShapes/Cube.Cube'"));
  const auto defaultUnrealMesh_Sphere =
      FindAsset<UStaticMesh>(TEXT("StaticMesh'/Engine/BasicShapes/Sphere.Sphere'"));
  // find default Mochi mesh references
  const auto defaultMochiMesh_Plane =
      FindAsset<UMochiModel>(TEXT("MochiModel'/SuperDexPhysics/Models/MM_Plane.MM_Plane'"));
  const auto defaultMochiMesh_Cube =
      FindAsset<UMochiModel>(TEXT("MochiModel'/SuperDexPhysics/Models/MM_Cube.MM_Cube'"));
  const auto defaultMochiMesh_Sphere =
      FindAsset<UMochiModel>(TEXT("MochiModel'/SuperDexPhysics/Models/MM_Sphere.MM_Sphere'"));
#endif
}

bool UMochiGameInstanceSubsystem::ShouldCreateSubsystem(UObject* Outer) const {
#if PLATFORM_LINUX
  // On Linux eval builds, MochiPhysics is not used at runtime - skip subsystem
  // creation to avoid loading mochi_physics.so and initializing the sim engine.
  return false;
#else
  return Super::ShouldCreateSubsystem(Outer);
#endif
}

UMochiSceneObject* UMochiGameInstanceSubsystem::GetScene() const {
  return Scene;
}

void UMochiGameInstanceSubsystem::EnableDebugDraw() const {
  if (const auto scene = GetScene()) {
    scene->ConfigureDebugDraw(true);
  } else {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("UMochiGameInstanceSubsystem::EnableDebugDraw: Failed to get UMochiSceneObject!"));
  }
}

void UMochiGameInstanceSubsystem::DisableDebugDraw() const {
  if (const auto scene = GetScene()) {
    scene->ConfigureDebugDraw(false);
  } else {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("UMochiGameInstanceSubsystem::EnableDebugDraw: Failed to get UMochiSceneObject!"));
  }
}

bool UMochiGameInstanceSubsystem::IsDebugDrawEnabled() const {
  const auto scene = GetScene();
  if (scene) {
    return scene->IsDebugDrawEnabled();
  }
  UE_LOG(
      LogMochiPhysics,
      Error,
      TEXT("UMochiGameInstanceSubsystem::IsDebugDrawEnabled: Failed to get UMochiSceneObject!"));
  return false;
}

void UMochiGameInstanceSubsystem::Initialize(FSubsystemCollectionBase& Collection) {
  MOCHI_PROFILE_SCOPE();
  UE_LOG(LogMochiPhysics, Display, TEXT("UMochiGameInstanceSubsystem: Initializing"));
  Super::Initialize(Collection);
  StoreAndSetEditorPreferencesOverrides();
  CreateScene();
}

void UMochiGameInstanceSubsystem::Deinitialize() {
  MOCHI_PROFILE_SCOPE();
  UE_LOG(LogMochiPhysics, Display, TEXT("UMochiGameInstanceSubsystem: Deinitializing"));
  DestroyScene();
  UMochiPhysicsSettings::ClearActiveSimulationSpaceOverride();
  RestoreEditorPreferencesOverrides();
  Super::Deinitialize();
}

UWorld* UMochiGameInstanceSubsystem::GetTickableGameObjectWorld() const {
  return GetWorld();
}

ETickableTickType UMochiGameInstanceSubsystem::GetTickableTickType() const {
  return IsTemplate() ? ETickableTickType::Never : FTickableGameObject::GetTickableTickType();
}

bool UMochiGameInstanceSubsystem::IsAllowedToTick() const {
  return !IsTemplate() && Scene != nullptr;
}

void UMochiGameInstanceSubsystem::Tick(float DeltaTime) {
  if (Scene) {
    // unpause the simulation on the first frame
    if (!bSceneRunning) {
      Scene->SetPaused(bStartPaused);
      bSceneRunning = true;
      UE_LOG(
          LogMochiPhysics, Display, TEXT("UMochiGameInstanceSubsystem: Set UMochiScene running."));
    }
    MOCHI_PROFILE_SCOPE();
    // debug draw game instance scene
    if (Scene->IsDebugDrawEnabled()) {
      Scene->DebugDrawScene(GetWorld());
    }
    // update stats group
    Scene->UpdateStatsGroups();
  }
}

TStatId UMochiGameInstanceSubsystem::GetStatId() const {
  RETURN_QUICK_DECLARE_CYCLE_STAT(UMochiGameInstanceSubsystem, STATGROUP_Tickables);
}

void UMochiGameInstanceSubsystem::CreateScene() {
  // destroy existing scene if present
  DestroyScene();

  // Ensure the SimulationSpace override is set before any converters are constructed. Any
  // override from a previous level must not leak into this one (the GameInstance subsystem
  // persists across OpenLevel transitions), so either apply this level's override or clear it.
  // Doing this in one call restarts a listening debugger at most once.
  // GetAllActorsOfClass (used by RetrieveMochiSceneSettings) is unreliable during subsystem
  // initialization because persistent level actors may not be registered yet. However, the
  // actors DO exist in PersistentLevel->Actors at this point (deserialized during level load),
  // so we scan there directly as an early fallback.
  const AMochiSceneSettingsActor* OverrideSettings = nullptr;
  if (UWorld* World = GetWorld()) {
    if (ULevel* Level = World->PersistentLevel) {
      for (AActor* Actor : Level->Actors) {
        auto* Settings = Cast<AMochiSceneSettingsActor>(Actor);
        if (Settings && Settings->SceneSettings.bOverrideSimulationSpace) {
          OverrideSettings = Settings;
          break;
        }
      }
    }
  }
  if (OverrideSettings) {
    UMochiPhysicsSettings::SetActiveSimulationSpaceOverride(
        OverrideSettings->SceneSettings.SimulationSpace);
    UE_LOG(
        LogMochiPhysics,
        Display,
        TEXT(
            "UMochiGameInstanceSubsystem: Early SimulationSpace override from PersistentLevel scan."));
  } else {
    UMochiPhysicsSettings::ClearActiveSimulationSpaceOverride();
  }

  // NB: When CreateScene is called by Initialize, the settings returned here will
  // almost always be the *project* settings since persistent level actors may not
  // have been initialized yet (UE is inconsistent in this regard). However, this
  // does not mean that settings applied through AMochiSceneSettingsActor won't be
  // used since that actor configures the scene settings in its BeginPlay function.
  auto settings = RetrieveMochiSceneSettings();

  // Create game instance scene. Create the scene paused so that all UE actors/components can issue
  // commands in BeginPlay before simulation begins. The scene will be unpaused on the next call to
  // Tick.
  Scene = UMochiSceneObject::CreateScene(
      this, EMochiSceneType::Async, "MochiGameInstaceScene", settings, /*bStartPaused*/ true);
  if (Scene) {
    bSceneRunning = false;
    bStartPaused = settings.bStartPaused;
    Scene->ConfigureDebugDraw(settings.bEnableDebugDraw, settings.DebugDrawFlags);
    UE_LOG(
        LogMochiPhysics,
        Display,
        TEXT("UMochiGameInstanceSubsystem: Created game instance UMochiScene."));
  } else {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("UMochiGameInstanceSubsystem: Failed to create new UMochiScene!"));
  }
}

void UMochiGameInstanceSubsystem::DestroyScene() {
  if (Scene) {
    Scene->DestroyScene();
    Scene->MarkAsGarbage();
    Scene = nullptr;
    UE_LOG(
        LogMochiPhysics,
        Display,
        TEXT("UMochiGameInstanceSubsystem: Destroyed game instance UMochiScene."));
  }
}

void UMochiGameInstanceSubsystem::StoreAndSetEditorPreferencesOverrides() {
#if WITH_EDITOR
  UEditorPerformanceSettings* EditorPerformanceSettings =
      GetMutableDefault<UEditorPerformanceSettings>();
  const UMochiPhysicsSettings* MochiSettings = GetDefault<UMochiPhysicsSettings>();
  if (EditorPerformanceSettings) {
    EditorPreferencesBeforeOverride.bThrottleCPUWhenNotForeground =
        EditorPerformanceSettings->bThrottleCPUWhenNotForeground;

    if (MochiSettings->bTurnOffUseLessCPUInBackgroundDuringPlay) {
      EditorPerformanceSettings->bThrottleCPUWhenNotForeground = false;
    }
  }
#endif
}

void UMochiGameInstanceSubsystem::RestoreEditorPreferencesOverrides() const {
#if WITH_EDITOR
  UEditorPerformanceSettings* EditorPerformanceSettings =
      GetMutableDefault<UEditorPerformanceSettings>();
  if (EditorPerformanceSettings) {
    EditorPerformanceSettings->bThrottleCPUWhenNotForeground =
        EditorPreferencesBeforeOverride.bThrottleCPUWhenNotForeground;
  }
#endif
}

FMochiSceneSettings UMochiGameInstanceSubsystem::RetrieveMochiSceneSettings() const {
  // first try to find settings from level instanced MochiSceneSettingsActor
  // NB: This path is not guaranteed to be executed if this function is called during
  // subsystem initialization, since persistent actors may have not been initialized yet
  // (put a break point here and note that settingsActors.Num is sometimes zero). Subsequent calls
  // in Tick or elsewhere *will* however return the AMochiSceneSettingsActor.
  TArray<AActor*> settingsActors;
  FMochiSceneSettings outSettings;
  UGameplayStatics::GetAllActorsOfClass(
      GetWorld(), AMochiSceneSettingsActor::StaticClass(), settingsActors);
  if (settingsActors.Num() > 0) {
    outSettings = Cast<AMochiSceneSettingsActor>(settingsActors[0])->SceneSettings;
    // throw a warning about multiple instances
    if (settingsActors.Num() > 1) {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "UMochiGameInstanceSubsystem: Found multiple AMochiSceneSettingsActors! Using first instance."))
    }
    // apply simulation space override so all converters pick it up
    if (outSettings.bOverrideSimulationSpace) {
      UMochiPhysicsSettings::SetActiveSimulationSpaceOverride(outSettings.SimulationSpace);
      UE_LOG(
          LogMochiPhysics,
          Display,
          TEXT(
              "UMochiGameInstanceSubsystem: Overriding SimulationSpace from AMochiSceneSettingsActor."))
    }
    UE_LOG(
        LogMochiPhysics,
        Display,
        TEXT("UMochiGameInstanceSubsystem: Obtained scene settings from AMochiSceneSettingsActor."))
    // mark all instances as applied
    for (const auto settingsActor : settingsActors) {
      Cast<AMochiSceneSettingsActor>(settingsActor)->bApplied = true;
    }
  }
  // otherwise fallback to project settings
  else {
    const auto projectSettings = GetDefault<UMochiPhysicsSettings>();
    if (projectSettings == nullptr) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("UMochiGameInstanceSubsystem: Failed to load UMochiPhysicsSettings!"));
      return outSettings;
    }
    outSettings = projectSettings->DefaultSceneSettings;
    UE_LOG(
        LogMochiPhysics,
        Display,
        TEXT("UMochiGameInstanceSubsystem: Obtained scene settings from UMochiPhysicsSettings."))
  }
  return outSettings;
}

#ifdef _MSC_VER
#if RT_DISABLE_OPTIMIZE
#pragma optimize("", on)
#endif
#undef RT_DISABLE_OPTIMIZE
#endif
