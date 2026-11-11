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

#include "SuperDexRoboticsFunctionLibrary.h"

#include "SuperDexBotComponent.h"

#include "MochiConversions.h"
#include "MochiMeshComponentInterface.h"
#include "MochiPhysicsFunctionLibrary.h"
#include "MochiPhysicsModule.h"

#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "EngineUtils.h"

#include <mochi_physics/utils/mochi_prefab.h>
#include <superdex_robotics/utils/archive_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <superdex_robotics/scene/bot_scene.h>

// Editor only: bots are exported from their source .superdex_bot files, and a bot's source path
// (USuperDexBot::GetSourceFilePath, from its import data) is editor-only data.
#if WITH_EDITOR
namespace {

struct FSuperDexSpawnablePrefabExportData {
  FName PrefabName;
  TSubclassOf<AActor> ActorClass;
};

/*
 * Scene session convention. The export can record data that lives in the consuming project, which
 * this plugin cannot link against: which scene slot each roboperator drives, which key names each
 * roboperator blueprint, and the spawnable prefabs a task may spawn. A project provides them on a
 * scene session actor, read by reflection: any actor whose class has a
 * `ResolvedRoboperatorClasses` property (TMap<FString, TSubclassOf<AActor>>: key -> roboperator
 * blueprint). The same actor may also have
 *  - `SessionMode`: an object whose `LeftRoboperatorActor` / `RightRoboperatorActor` actor
 *    properties are the roboperators currently driving each side;
 *  - `LeftSlotName` / `RightSlotName`: FString scene slot names (default "left" / "right");
 *  - `SpawnablePrefabSet`: an object with a `Prefabs` array of structs with `PrefabName` (FName)
 *    and `ActorClass` (class) fields.
 * Task actors (any actor with a `SuperDexBotTaskPath` property) may implement
 * `DestroySpawnedPrefabActors` and `ResetTaskFromSuperDexBotTaskFile` so the default export can
 * remove and restore what the task spawned. Worlds without a scene session actor export no slots
 * and no spawnable prefabs.
 */
const TCHAR* const kResolvedRoboperatorClassesProperty = TEXT("ResolvedRoboperatorClasses");

bool IsSceneSessionClass(const UClass* Class) {
  return Class && Class->FindPropertyByName(kResolvedRoboperatorClassesProperty) != nullptr;
}

AActor* FindSceneSessionActor(UWorld* World) {
  if (!World) {
    return nullptr;
  }
  for (TActorIterator<AActor> It(World); It; ++It) {
    if (IsSceneSessionClass(It->GetClass())) {
      return *It;
    }
  }
  return nullptr;
}

UObject* GetObjectPropertyValue(UObject* Owner, const TCHAR* PropertyName) {
  if (!Owner) {
    return nullptr;
  }
  const FObjectPropertyBase* Property =
      CastField<FObjectPropertyBase>(Owner->GetClass()->FindPropertyByName(PropertyName));
  return Property ? Property->GetObjectPropertyValue_InContainer(Owner) : nullptr;
}

/**
 * Which scene slot each roboperator drives, and the key naming each roboperator blueprint, read
 * from the scene session actor (see the convention above).
 */
struct FRoboperatorExportBindings {
  /** Whether the world has a scene session actor; worlds without one have no roboperator slots. */
  bool bFoundSession = false;
  TMap<const AActor*, FString> SlotByRoboperator;
  TMap<const UClass*, FString> KeyByRoboperatorClass;
};

/** Read an FString UPROPERTY by name, or return Fallback when the owner does not expose one. */
FString GetStringPropertyValue(UObject* Owner, FName PropertyName, const TCHAR* Fallback) {
  const FStrProperty* Property = Owner
      ? CastField<FStrProperty>(Owner->GetClass()->FindPropertyByName(PropertyName))
      : nullptr;
  if (!Property) {
    return FString(Fallback);
  }
  const FString Value = Property->GetPropertyValue_InContainer(Owner);
  return Value.IsEmpty() ? FString(Fallback) : Value;
}

FString DescribeRoboperatorClassKeys(const FRoboperatorExportBindings& Bindings) {
  TArray<FString> Entries;
  Entries.Reserve(Bindings.KeyByRoboperatorClass.Num());
  for (auto const& Pair : Bindings.KeyByRoboperatorClass) {
    Entries.Add(FString::Printf(TEXT("'%s' -> %s"), *Pair.Value, *Pair.Key->GetName()));
  }
  return Entries.IsEmpty() ? FString(TEXT("<none>")) : FString::Join(Entries, TEXT(", "));
}

void CollectRoboperatorSlots(UObject* Session, FRoboperatorExportBindings& OutBindings) {
  // The spawned roboperators live on the session's active mode object, not on the session actor.
  UObject* SessionMode = GetObjectPropertyValue(Session, TEXT("SessionMode"));
  if (!SessionMode) {
    return;
  }
  // The slot a side drives is the level's to name, so export whatever this session calls it
  // rather than baking "left"/"right" into the exporter.
  if (auto* Left =
          Cast<AActor>(GetObjectPropertyValue(SessionMode, TEXT("LeftRoboperatorActor")))) {
    OutBindings.SlotByRoboperator.Add(
        Left, GetStringPropertyValue(Session, TEXT("LeftSlotName"), TEXT("left")));
  }
  if (auto* Right =
          Cast<AActor>(GetObjectPropertyValue(SessionMode, TEXT("RightRoboperatorActor")))) {
    OutBindings.SlotByRoboperator.Add(
        Right, GetStringPropertyValue(Session, TEXT("RightSlotName"), TEXT("right")));
  }
}

void CollectRoboperatorClassKeys(UObject* Session, FRoboperatorExportBindings& OutBindings) {
  const FProperty* Property =
      Session->GetClass()->FindPropertyByName(kResolvedRoboperatorClassesProperty);
  const FMapProperty* MapProperty = CastField<FMapProperty>(Property);
  if (!MapProperty) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "ExportSuperDexBotScene: '%s' exposes no ResolvedRoboperatorClasses map (found %s); exporting no roboperatorClass."),
        *Session->GetClass()->GetName(),
        Property ? *Property->GetClass()->GetName() : TEXT("nothing"));
    return;
  }
  // The declared value type is TSubclassOf<ATeleopRoboperatorActor>, which this module cannot
  // name. Checking the reflected key and value types establishes that the map is laid out as the
  // pointer-compatible TSubclassOf<AActor> spelling below, so it can then be read directly.
  if (!CastField<FStrProperty>(MapProperty->KeyProp) ||
      !CastField<FClassProperty>(MapProperty->ValueProp)) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "ExportSuperDexBotScene: ResolvedRoboperatorClasses is <%s, %s>, not <FString, TSubclassOf<>>; exporting no roboperatorClass."),
        *MapProperty->KeyProp->GetClass()->GetName(),
        *MapProperty->ValueProp->GetClass()->GetName());
    return;
  }
  const auto& Map =
      *MapProperty->ContainerPtrToValuePtr<TMap<FString, TSubclassOf<AActor>>>(Session);
  if (Map.IsEmpty()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "ExportSuperDexBotScene: '%s' resolved no roboperator classes; exporting no roboperatorClass. "
            "Give the session roboperator classes to have the scene name its rigs."),
        *Session->GetName());
    return;
  }

  for (auto const& Pair : Map) {
    const FString& Key = Pair.Key;
    const UClass* Class = Pair.Value.Get();
    if (Key.IsEmpty() || !Class) {
      continue;
    }
    if (const FString* Existing = OutBindings.KeyByRoboperatorClass.Find(Class)) {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "ExportSuperDexBotScene: Roboperator classes name blueprint '%s' as both '%s' and '%s'; exporting '%s'."),
          *Class->GetName(),
          **Existing,
          *Key,
          **Existing);
      continue;
    }
    OutBindings.KeyByRoboperatorClass.Add(Class, Key);
  }
}

FRoboperatorExportBindings CollectRoboperatorBindings(UWorld* World) {
  FRoboperatorExportBindings Bindings;
  UObject* Session = FindSceneSessionActor(World);
  if (!Session) {
    return Bindings;
  }
  Bindings.bFoundSession = true;
  CollectRoboperatorSlots(Session, Bindings);
  CollectRoboperatorClassKeys(Session, Bindings);
  return Bindings;
}

void ForEachTaskActorWithTaskFile(
    UWorld* World,
    TFunctionRef<void(AActor* TaskActor, UFunction* Function)> Callback) {
  if (!World) {
    return;
  }

  for (TActorIterator<AActor> It(World); It; ++It) {
    AActor* Actor = *It;
    if (!Actor->GetClass()->FindPropertyByName(TEXT("SuperDexBotTaskPath"))) {
      continue;
    }

    UFunction* Function = Actor->FindFunction(TEXT("DestroySpawnedPrefabActors"));
    if (!Function) {
      continue;
    }

    Callback(Actor, Function);
  }
}

void DestroyTaskSpawnedActors(UWorld* World) {
  ForEachTaskActorWithTaskFile(World, [](AActor* TaskActor, UFunction* Function) {
    TaskActor->ProcessEvent(Function, nullptr);
  });
}

void ReapplyTaskSpawnedActorsOnNextTick(UWorld* World) {
  TArray<AActor*> TaskActors;
  ForEachTaskActorWithTaskFile(
      World, [&TaskActors](AActor* TaskActor, UFunction*) { TaskActors.Add(TaskActor); });

  for (AActor* TaskActor : TaskActors) {
    if (!IsValid(TaskActor)) {
      continue;
    }

    UFunction* Function = TaskActor->FindFunction(TEXT("ResetTaskFromSuperDexBotTaskFile"));
    if (!Function) {
      continue;
    }
    TaskActor->GetWorldTimerManager().SetTimerForNextTick([TaskActor, Function]() {
      if (IsValid(TaskActor)) {
        TaskActor->ProcessEvent(Function, nullptr);
      }
    });
  }
}

TArray<FSuperDexSpawnablePrefabExportData> CollectSpawnablePrefabs(UWorld* World) {
  TArray<FSuperDexSpawnablePrefabExportData> SpawnablePrefabs;
  if (!World) {
    return SpawnablePrefabs;
  }

  bool bFoundSpawnablePrefabOwner = false;
  for (TActorIterator<AActor> It(World); It; ++It) {
    AActor* Actor = *It;
    if (!IsSceneSessionClass(Actor->GetClass())) {
      continue;
    }

    // The scene session actor references its spawnable prefabs through a `SpawnablePrefabSet`
    // object property; resolve that object and its `Prefabs` array by reflection.
    const FObjectPropertyBase* PrefabSetProp = CastField<FObjectPropertyBase>(
        Actor->GetClass()->FindPropertyByName(TEXT("SpawnablePrefabSet")));
    if (!PrefabSetProp) {
      continue;
    }

    bFoundSpawnablePrefabOwner = true;
    UObject* PrefabSet = PrefabSetProp->GetObjectPropertyValue_InContainer(Actor);
    if (!PrefabSet) {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT("ExportSuperDexBotScene: Actor '%s' has no SpawnablePrefabSet data asset assigned."),
          *Actor->GetName());
      continue;
    }

    const FArrayProperty* SpawnablePrefabsProp =
        CastField<FArrayProperty>(PrefabSet->GetClass()->FindPropertyByName(TEXT("Prefabs")));
    if (!SpawnablePrefabsProp) {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "ExportSuperDexBotScene: SpawnablePrefabSet on actor '%s' has no 'Prefabs' array property."),
          *Actor->GetName());
      continue;
    }
    const FStructProperty* EntryStructProp =
        CastField<FStructProperty>(SpawnablePrefabsProp->Inner);
    if (!EntryStructProp) {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "ExportSuperDexBotScene: Actor '%s' has spawnable prefab property '%s' with non-struct entries."),
          *Actor->GetName(),
          *SpawnablePrefabsProp->GetName());
      continue;
    }

    const FNameProperty* PrefabNameProp =
        CastField<FNameProperty>(EntryStructProp->Struct->FindPropertyByName(TEXT("PrefabName")));
    const FClassProperty* ActorClassProp =
        CastField<FClassProperty>(EntryStructProp->Struct->FindPropertyByName(TEXT("ActorClass")));
    if (!PrefabNameProp || !ActorClassProp) {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "ExportSuperDexBotScene: Actor '%s' has spawnable prefab entries without PrefabName/ActorClass fields."),
          *Actor->GetName());
      continue;
    }

    const void* ArrayPtr = SpawnablePrefabsProp->ContainerPtrToValuePtr<void>(PrefabSet);
    FScriptArrayHelper ArrayHelper(SpawnablePrefabsProp, ArrayPtr);
    UE_LOG(
        LogMochiPhysics,
        Log,
        TEXT("ExportSuperDexBotScene: Found %d spawnable prefab entries on actor '%s'."),
        ArrayHelper.Num(),
        *Actor->GetName());

    for (int32 Index = 0; Index < ArrayHelper.Num(); ++Index) {
      const uint8* EntryPtr = ArrayHelper.GetRawPtr(Index);
      const FName PrefabName = PrefabNameProp->GetPropertyValue_InContainer(EntryPtr);
      auto* ActorClass = Cast<UClass>(ActorClassProp->GetObjectPropertyValue_InContainer(EntryPtr));
      if (PrefabName.IsNone() || !ActorClass) {
        UE_LOG(
            LogMochiPhysics,
            Warning,
            TEXT("ExportSuperDexBotScene: Ignoring empty spawnable prefab entry %d on actor '%s'."),
            Index,
            *Actor->GetName());
        continue;
      }

      FSuperDexSpawnablePrefabExportData SpawnablePrefab;
      SpawnablePrefab.PrefabName = PrefabName;
      SpawnablePrefab.ActorClass = ActorClass;
      SpawnablePrefabs.Add(MoveTemp(SpawnablePrefab));
      UE_LOG(
          LogMochiPhysics,
          Log,
          TEXT("ExportSuperDexBotScene: Collected spawnable prefab '%s' for actor class '%s'."),
          *PrefabName.ToString(),
          *ActorClass->GetName());
    }
  }

  if (!bFoundSpawnablePrefabOwner) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("ExportSuperDexBotScene: No actor with SpawnablePrefabSet was found in the world."));
  }

  return SpawnablePrefabs;
}

} // namespace

bool USuperDexRoboticsFunctionLibrary::ExportSuperDexBotScene(
    UObject* WorldContext,
    const FString& ExportName,
    const FString& OutputDir,
    bool bCaptureLiveScene) {
  const auto subsys = UMochiPhysicsFunctionLibrary::GetMochiGameInstanceSubsystem(WorldContext);
  if (!subsys || !subsys->GetScene()) {
    UE_LOG(LogMochiPhysics, Warning, TEXT("ExportSuperDexBotScene: No active Mochi scene."));
    return false;
  }

  // --- Game thread: collect export data from UE actors ---

  struct BotExportData {
    FString BotName;
    FString SuperDexBotSourcePath;
    TArray<double> InitialJointPositions;
    mochi::ActorHandle ActorHandle;
    TArray<superdex::robotics::ControllerEntry> ControllerEntries;
    /** Scene slot this bot filled, or empty for a bot no roboperator drove. */
    FString SlotName;
    FString RoboperatorClassKey;
  };

  struct SpawnableExportData {
    FString PrefabName;
    FString SuperDexBotSourcePath;
    TArray<mochi::ActorHandle> ActorHandles;
  };

  TArray<BotExportData> CollectedBots;
  TArray<SpawnableExportData> CollectedSpawnables;
  TArray<mochi::ActorHandle> ExcludeHandles;

  FString ControllersDir = FPaths::Combine(OutputDir, TEXT("controllers"));
  IPlatformFile::GetPlatformPhysical().CreateDirectoryTree(*ControllersDir);

  UWorld* World = WorldContext->GetWorld();
  if (!bCaptureLiveScene) {
    DestroyTaskSpawnedActors(World);
  }

  // Collect spawnable prefab *definitions* only. No actors are spawned yet: the base scene must be
  // exported before any spawnable export actors exist in the live Mochi scene, otherwise their
  // meshes leak into the base scene. A live-scene capture skips this entirely: the task and its
  // spawnables are already present in the live scene and are captured directly in the base scene.
  TArray<FSuperDexSpawnablePrefabExportData> SpawnablePrefabs;
  if (!bCaptureLiveScene) {
    SpawnablePrefabs = CollectSpawnablePrefabs(World);
  }

  // A roboperator blueprint owns the bot component of the bot it drives, so the actor holding the
  // component below is the roboperator whose slot and class key the scene should record.
  const FRoboperatorExportBindings RoboperatorBindings = CollectRoboperatorBindings(World);
  if (RoboperatorBindings.bFoundSession && RoboperatorBindings.SlotByRoboperator.IsEmpty()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "ExportSuperDexBotScene: No spawned roboperators found, so the scene will declare no "
            "slots. Start an episode before exporting if the scene should drive teleop."));
  }

  // Collect bots already present in the level. Their articulated actors are excluded from the base
  // scene and exported separately as bot archives.
  for (TActorIterator<AActor> It(World); It; ++It) {
    AActor* Actor = *It;

    auto* BotComp = Actor->FindComponentByClass<USuperDexBotComponent>();
    if (!BotComp || !BotComp->GetBotAsset()) {
      continue;
    }

    BotExportData Data;
    Data.BotName = ANSI_TO_TCHAR(BotComp->GetBotAsset()->GetBotPrefab().name.c_str());
    Data.SuperDexBotSourcePath = BotComp->GetBotAsset()->GetSourceFilePath();
    Data.InitialJointPositions = BotComp->GetInitialPose();

    if (const FString* Slot = RoboperatorBindings.SlotByRoboperator.Find(Actor)) {
      Data.SlotName = *Slot;
      if (const FString* Key = RoboperatorBindings.KeyByRoboperatorClass.Find(Actor->GetClass())) {
        Data.RoboperatorClassKey = *Key;
      } else {
        // An unmapped key would make the exported scene reject the session it was exported from,
        // so leave it empty and let the loading level pick the slot's default blueprint.
        UE_LOG(
            LogMochiPhysics,
            Warning,
            TEXT(
                "ExportSuperDexBotScene: Bot '%s' is driven by roboperator blueprint '%s', which "
                "this level's roboperator classes do not name (they name %s). Exporting no "
                "roboperatorClass; add the blueprint to the session's roboperator classes to name it."),
            *Data.BotName,
            *Actor->GetClass()->GetName(),
            *DescribeRoboperatorClassKeys(RoboperatorBindings));
      }
    }

    if (auto* MochiActor = BotComp->GetMochiArticulatedActor()) {
      Data.ActorHandle = MochiActor->GetArticulatedActorHandle();
      ExcludeHandles.Add(Data.ActorHandle);
    }

    if (auto* Controller = BotComp->GetController()) {
      Data.ControllerEntries = Controller->ExportControllerEntries(ControllersDir);
    }

    CollectedBots.Add(MoveTemp(Data));
  }

  if (!bCaptureLiveScene && CollectedBots.IsEmpty() && SpawnablePrefabs.IsEmpty()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "ExportSuperDexBotScene: No actors with USuperDexBotComponent or configured spawnable prefab classes found."));
    ReapplyTaskSpawnedActorsOnNextTick(World);
    return false;
  }

  bool bWasPaused = UMochiPhysicsFunctionLibrary::IsPaused(WorldContext);
  UMochiPhysicsFunctionLibrary::SetPaused(WorldContext, /*bPaused=*/true);

  // --- Mochi thread (phase 1): export the base scene before any spawnables are created. ---

  bool bBaseSceneExported = false;
  subsys->GetScene()->QueueOrExecuteCommand([ExcludeHandles = MoveTemp(ExcludeHandles),
                                             OutputDir,
                                             &bBaseSceneExported](mochi::Scene* Scene) {
    mochi::Error Error;

    mochi::prefab::ExportParams Params;
    Params.excludeActors = ToMochiSpan(ExcludeHandles);

    mochi::prefab::ExportScene(Scene, "base_scene", TCHAR_TO_UTF8(*OutputDir), Params, Error);
    if (!Error.IsOK()) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("ExportSuperDexBotScene: Failed to export base scene: %hs"),
          Error.ToString().c_str());
      return;
    }
    bBaseSceneExported = true;
  });
  subsys->GetScene()->WaitForQueuedCommands();
  if (!bBaseSceneExported) {
    UMochiPhysicsFunctionLibrary::SetPaused(WorldContext, /*bPaused=*/bWasPaused);
    if (!bCaptureLiveScene) {
      ReapplyTaskSpawnedActorsOnNextTick(World);
    }
    return false;
  }

  // --- Game thread: spawn spawnable prefab actors now that the base scene is captured. ---

  TArray<AActor*> SpawnedExportPrefabActors;
  for (auto const& SpawnablePrefab : SpawnablePrefabs) {
    if (!SpawnablePrefab.ActorClass || SpawnablePrefab.PrefabName.IsNone()) {
      continue;
    }

    AActor* SpawnedActor =
        World->SpawnActor<AActor>(SpawnablePrefab.ActorClass, FTransform::Identity);
    if (!IsValid(SpawnedActor)) {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT("ExportSuperDexBotScene: Failed to spawn export prefab actor for '%s'."),
          *SpawnablePrefab.PrefabName.ToString());
      continue;
    }
    SpawnedExportPrefabActors.Add(SpawnedActor);
    SpawnedActor->Tags.AddUnique(FName(TEXT("SuperDexBotSceneExportSpawnableActor")));

    if (auto* BotComp = SpawnedActor->FindComponentByClass<USuperDexBotComponent>()) {
      if (BotComp->GetBotAsset()) {
        SpawnableExportData Data;
        Data.PrefabName = SpawnablePrefab.PrefabName.ToString();
        Data.SuperDexBotSourcePath = BotComp->GetBotAsset()->GetSourceFilePath();
        CollectedSpawnables.Add(MoveTemp(Data));
        continue;
      }
    }

    TArray<mochi::ActorHandle> SpawnableActorHandles;
    TArray<UActorComponent*> ActorComponents;
    SpawnedActor->GetComponents(ActorComponents);
    for (auto* Component : ActorComponents) {
      auto* MochiMeshComp = Cast<IMochiMeshComponentInterface>(Component);
      if (!MochiMeshComp) {
        continue;
      }
      auto* MochiActor = MochiMeshComp->GetMochiActor();
      if (MochiActor && MochiActor->IsValid()) {
        SpawnableActorHandles.Add(MochiActor->GetMochiActorHandle());
      }
    }

    if (SpawnableActorHandles.Num() > 0) {
      SpawnableExportData Data;
      Data.PrefabName = SpawnablePrefab.PrefabName.ToString();
      Data.ActorHandles = MoveTemp(SpawnableActorHandles);
      CollectedSpawnables.Add(MoveTemp(Data));
    } else {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "ExportSuperDexBotScene: Spawned export prefab actor for '%s' has no valid Mochi mesh actor."),
          *SpawnablePrefab.PrefabName.ToString());
    }
  }

  // --- Mochi thread (phase 2): export spawnables, archive bots, compose .superdex_scene. ---

  bool bSuccess = false;
  subsys->GetScene()->QueueOrExecuteCommand([CollectedBots = MoveTemp(CollectedBots),
                                             CollectedSpawnables = MoveTemp(CollectedSpawnables),
                                             OutputDir,
                                             ExportName,
                                             &bSuccess](mochi::Scene* Scene) {
    mochi::Error Error;

    // 1. Resolve exportable spawnables. The base scene was already exported in phase 1, so no
    // exclusion is needed here.
    TArray<SpawnableExportData> ExportableSpawnables;
    for (auto const& SpawnableData : CollectedSpawnables) {
      if (!SpawnableData.SuperDexBotSourcePath.IsEmpty()) {
        ExportableSpawnables.Add(SpawnableData);
        continue;
      }

      bool bAllActorsFound = true;
      for (auto const& Handle : SpawnableData.ActorHandles) {
        if (!Scene->GetActor(Handle)) {
          bAllActorsFound = false;
          break;
        }
      }
      if (!bAllActorsFound || SpawnableData.ActorHandles.IsEmpty()) {
        UE_LOG(
            LogMochiPhysics,
            Warning,
            TEXT(
                "ExportSuperDexBotScene: Skipping spawnable prefab '%s' because one or more Mochi actors were not found."),
            *SpawnableData.PrefabName);
        continue;
      }

      ExportableSpawnables.Add(SpawnableData);
    }

    // 2. Create asset directories.
    FString BotsDir = FPaths::Combine(OutputDir, TEXT("bots"));
    IPlatformFile::GetPlatformPhysical().CreateDirectoryTree(*BotsDir);
    FString SpawnablesDir = FPaths::Combine(OutputDir, TEXT("spawnables"));
    IPlatformFile::GetPlatformPhysical().CreateDirectoryTree(*SpawnablesDir);

    // 3. Build BotScenePrefab.
    superdex::robotics::BotScenePrefab Prefab;
    Prefab.metadata.name = mochi::DynamicString(TCHAR_TO_UTF8(*ExportName));
    Prefab.metadata.version = mochi::DynamicString("1.0");
    Prefab.metadata.author = mochi::DynamicString("Unreal Export");
    Prefab.scene.baseScene = mochi::DynamicString("//base_scene/base_scene.mochi_scene");

    for (auto const& SpawnableData : ExportableSpawnables) {
      superdex::robotics::PrefabEntry PrefabEntry;
      PrefabEntry.name = mochi::DynamicString(TCHAR_TO_UTF8(*SpawnableData.PrefabName));

      if (!SpawnableData.SuperDexBotSourcePath.IsEmpty()) {
        FString BotArchiveName = SpawnableData.PrefabName + TEXT(".superdex_bot_archive");
        FString BotArchivePath = FPaths::Combine(BotsDir, BotArchiveName);
        if (superdex::robotics::IsBotArchivePath(
                TCHAR_TO_UTF8(*SpawnableData.SuperDexBotSourcePath))) {
          if (!IPlatformFile::GetPlatformPhysical().CopyFile(
                  *BotArchivePath, *SpawnableData.SuperDexBotSourcePath)) {
            UE_LOG(
                LogMochiPhysics,
                Error,
                TEXT("ExportSuperDexBotScene: Failed to copy spawnable bot archive '%s' to '%s'."),
                *SpawnableData.SuperDexBotSourcePath,
                *BotArchivePath);
            return;
          }
        } else {
          superdex::robotics::ArchiveParams ArchiveParams;
          ArchiveParams.src = TCHAR_TO_UTF8(*SpawnableData.SuperDexBotSourcePath);
          ArchiveParams.dst = TCHAR_TO_UTF8(*BotArchivePath);
          superdex::robotics::ArchiveBot(ArchiveParams, Error);
          if (!Error.IsOK()) {
            UE_LOG(
                LogMochiPhysics,
                Error,
                TEXT("ExportSuperDexBotScene: Failed to archive spawnable bot '%s' (%s): %hs"),
                *SpawnableData.PrefabName,
                *SpawnableData.SuperDexBotSourcePath,
                Error.ToString().c_str());
            return;
          }
        }
        PrefabEntry.path = mochi::DynamicString(
            TCHAR_TO_UTF8(*FString::Printf(TEXT("//bots/%s"), *BotArchiveName)));
      } else {
        // Build exclusion list: all scene actors except this spawnable's actors.
        // Skip nested link/soft actors - they are excluded implicitly via their parent.
        mochi::prefab::ExportParams Params;
        Scene->ForEachActor([&](mochi::Actor const* SceneActor) {
          if (SceneActor->IsNestedLinkActor() || SceneActor->IsNestedSoftActor()) {
            return;
          }
          auto Handle = SceneActor->GetHandle();
          bool bKeep = false;
          for (auto const& SpawnableHandle : SpawnableData.ActorHandles) {
            if (Handle == SpawnableHandle) {
              bKeep = true;
              break;
            }
          }
          if (!bKeep) {
            Params.excludeActors.push_back(Handle);
          }
        });
        mochi::prefab::ExportScene(
            Scene,
            TCHAR_TO_UTF8(*SpawnableData.PrefabName),
            TCHAR_TO_UTF8(*SpawnablesDir),
            Params,
            Error);
        if (!Error.IsOK()) {
          UE_LOG(
              LogMochiPhysics,
              Error,
              TEXT("ExportSuperDexBotScene: Failed to export spawnable prefab '%s': %hs"),
              *SpawnableData.PrefabName,
              Error.ToString().c_str());
          return;
        }
        PrefabEntry.path = mochi::DynamicString(TCHAR_TO_UTF8(*FString::Printf(
            TEXT("//spawnables/%s/%s.mochi_scene"),
            *SpawnableData.PrefabName,
            *SpawnableData.PrefabName)));
      }

      Prefab.scene.spawnablePrefabs.push_back(std::move(PrefabEntry));
    }

    for (auto const& BotData : CollectedBots) {
      // Get bot transform from the live Mochi scene.
      auto* Actor = Scene->GetActor(BotData.ActorHandle);

      superdex::robotics::BotEntry BotEntry;
      BotEntry.name = mochi::DynamicString(TCHAR_TO_UTF8(*BotData.BotName));
      if (!BotData.SlotName.IsEmpty()) {
        // One bot per slot: this scene records the arrangement it was exported from, and a later
        // edit is what adds alternatives to a slot.
        superdex::robotics::SlotEntry Slot;
        Slot.name = mochi::DynamicString(TCHAR_TO_UTF8(*BotData.SlotName));
        Slot.controlledBots.push_back(mochi::DynamicString(TCHAR_TO_UTF8(*BotData.BotName)));
        Prefab.slots.push_back(std::move(Slot));
      }
      if (!BotData.RoboperatorClassKey.IsEmpty()) {
        BotEntry.roboperatorClass =
            mochi::DynamicString(TCHAR_TO_UTF8(*BotData.RoboperatorClassKey));
      }

      if (Actor) {
        auto Transform = Actor->GetRootTransform();
        BotEntry.parentFromBot =
            mochi::TransformRT(Transform.GetRotation(), Transform.GetTranslation());
      }

      for (auto const& Pos : BotData.InitialJointPositions) {
        BotEntry.initialPose.push_back(Pos);
      }

      // Place the bot archive into the output directory. If the source bot is already a
      // bot archive (.superdex_bot_archive), just copy it; otherwise archive the raw bot
      // (.superdex_bot).
      FString BotArchiveName = BotData.BotName + TEXT(".superdex_bot_archive");
      FString BotArchivePath = FPaths::Combine(BotsDir, BotArchiveName);
      if (superdex::robotics::IsBotArchivePath(TCHAR_TO_UTF8(*BotData.SuperDexBotSourcePath))) {
        if (!IPlatformFile::GetPlatformPhysical().CopyFile(
                *BotArchivePath, *BotData.SuperDexBotSourcePath)) {
          UE_LOG(
              LogMochiPhysics,
              Error,
              TEXT("ExportSuperDexBotScene: Failed to copy bot archive '%s' to '%s'."),
              *BotData.SuperDexBotSourcePath,
              *BotArchivePath);
          return;
        }
      } else {
        superdex::robotics::ArchiveParams ArchiveParams;
        ArchiveParams.src = TCHAR_TO_UTF8(*BotData.SuperDexBotSourcePath);
        ArchiveParams.dst = TCHAR_TO_UTF8(*BotArchivePath);
        superdex::robotics::ArchiveBot(ArchiveParams, Error);
        if (!Error.IsOK()) {
          UE_LOG(
              LogMochiPhysics,
              Error,
              TEXT("ExportSuperDexBotScene: Failed to archive bot '%s': %hs"),
              *BotData.BotName,
              Error.ToString().c_str());
          return;
        }
      }
      BotEntry.path =
          mochi::DynamicString(TCHAR_TO_UTF8(*FString::Printf(TEXT("//bots/%s"), *BotArchiveName)));

      for (auto const& CtrlEntry : BotData.ControllerEntries) {
        superdex::robotics::ControllerEntry Ctrl;
        Ctrl = CtrlEntry;
        FString ParamsFilename = FPaths::GetCleanFilename(CtrlEntry.params.c_str());
        Ctrl.params = mochi::DynamicString(
            TCHAR_TO_UTF8(*FString::Printf(TEXT("//controllers/%s"), *ParamsFilename)));
        BotEntry.controllers.push_back(std::move(Ctrl));
      }

      Prefab.bots.push_back(std::move(BotEntry));
    }

    // 4. Save .superdex_scene file.
    FString SceneFilePath = FPaths::Combine(OutputDir, ExportName + TEXT(".superdex_scene"));
    superdex::robotics::SaveToFile(Prefab, TCHAR_TO_UTF8(*SceneFilePath), Error);
    if (!Error.IsOK()) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("ExportSuperDexBotScene: Failed to save .superdex_scene: %hs"),
          Error.ToString().c_str());
      return;
    }

    // 5. Drop a .superdex_root marker next to the .superdex_scene file.
    FString RootMarkerPath = FPaths::Combine(OutputDir, TEXT(".superdex_root"));
    {
      TUniquePtr<IFileHandle> MarkerHandle(
          IPlatformFile::GetPlatformPhysical().OpenWrite(*RootMarkerPath));
      if (!MarkerHandle) {
        UE_LOG(
            LogMochiPhysics,
            Error,
            TEXT("ExportSuperDexBotScene: Failed to write .superdex_root marker at %s"),
            *RootMarkerPath);
        return;
      }
    }

    // 6. Archive the entire output directory.
    FString ArchivePath = OutputDir + TEXT(".superdex_scene_archive");
    superdex::robotics::ArchiveParams ArchiveParams;
    ArchiveParams.src = TCHAR_TO_UTF8(*SceneFilePath);
    ArchiveParams.dst = TCHAR_TO_UTF8(*ArchivePath);
    superdex::robotics::ArchiveBotScene(ArchiveParams, Error);
    if (!Error.IsOK()) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("ExportSuperDexBotScene: Failed to create scene archive: %hs"),
          Error.ToString().c_str());
      return;
    }

    UE_LOG(
        LogMochiPhysics,
        Log,
        TEXT("ExportSuperDexBotScene: Successfully exported to %s"),
        *ArchivePath);
    bSuccess = true;
  });
  subsys->GetScene()->WaitForQueuedCommands();
  for (AActor* SpawnedActor : SpawnedExportPrefabActors) {
    if (IsValid(SpawnedActor)) {
      SpawnedActor->Destroy();
    }
  }
  UMochiPhysicsFunctionLibrary::SetPaused(WorldContext, /*bPaused=*/bWasPaused);
  if (!bCaptureLiveScene) {
    ReapplyTaskSpawnedActorsOnNextTick(World);
  }
  return bSuccess;
}
#else
// UnrealHeaderTool does not evaluate WITH_EDITOR here, so it emits this UFUNCTION's reflection
// data regardless. The Blueprint node has to exist; only the body can go.
bool USuperDexRoboticsFunctionLibrary::ExportSuperDexBotScene(
    UObject*,
    const FString&,
    const FString&,
    bool) {
  UE_LOG(LogMochiPhysics, Warning, TEXT("ExportSuperDexBotScene: only editor builds can export."));
  return false;
}
#endif
