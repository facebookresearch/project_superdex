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

#include "SuperDexSceneImporter.h"

#include "AssetRegistry/AssetRegistryModule.h"
#include "Engine/Blueprint.h"
#include "Engine/World.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/SecureHash.h"
#include "SuperDexBot.h"
#include "SuperDexBotComponent.h"
#include "SuperDexImportUtils.h"
#include "SuperDexScene.h"

#include <superdex_robotics/scene/bot_scene.h>
#include <superdex_robotics/utils/archive_utils.h>
#include <superdex_robotics/utils/file_utils.h>

DEFINE_LOG_CATEGORY_STATIC(LogSuperDexSceneImport, Log, All);

namespace {

// Changing what the importer generates should bump this so existing imports are regenerated.
constexpr const TCHAR* kImporterVersion = TEXT("2");

FString ToFString(const mochi::DynamicString& S) {
  return FString(UTF8_TO_TCHAR(S.c_str()));
}

bool IsBotFile(const FString& Path) {
  const std::string PathUtf8(TCHAR_TO_UTF8(*Path));
  return superdex::robotics::IsBotPath(PathUtf8) || superdex::robotics::IsBotArchivePath(PathUtf8);
}

bool LoadBotScene(const FString& SourcePath, superdex::robotics::BotScenePrefab& OutBotScene) {
  mochi::Error Error;
  OutBotScene = superdex::robotics::LoadBotScenePrefabFromFile(TCHAR_TO_UTF8(*SourcePath), Error);
  if (!Error.IsOK()) {
    UE_LOG(
        LogSuperDexSceneImport,
        Error,
        TEXT("Failed to load scene %s: %hs"),
        *SourcePath,
        Error.ToString().c_str());
    return false;
  }
  return true;
}

void WarnUnsupported(const FString& SceneName, const FString& What) {
  UE_LOG(
      LogSuperDexSceneImport,
      Warning,
      TEXT("[%s] %s is not supported by the scene importer and was ignored."),
      *SceneName,
      *What);
}

void AddBot(
    const FString& SceneName,
    const FMochiToUnrealConverter& M2U,
    FSuperDexPrefabImporter::FOutput& Output,
    const superdex::robotics::BotEntry& Entry,
    USuperDexBot* Bot) {
  const FString BotName = ToFString(Entry.name);
  for (const superdex::robotics::ControllerEntry& Controller : Entry.controllers) {
    WarnUnsupported(
        SceneName,
        FString::Printf(
            TEXT("Controller %s (%s) of bot %s"),
            *ToFString(Controller.name),
            *ToFString(Controller.type),
            *BotName));
  }
  if (!Bot) {
    return;
  }
  USuperDexBotComponent* Component = CastChecked<USuperDexBotComponent>(
      Output.AddActor(USuperDexBotComponent::StaticClass(), BotName));
  Component->SetRelativeTransform(M2U.TransformToUnreal(Entry.parentFromBot));
  Component->SetBotAsset(Bot);
  if (Entry.initialPose.empty()) {
    return;
  }
  const int32 NumDofs = Bot->GetBotPrefab()._numDofs;
  const int32 NumPoseDofs = static_cast<int32>(Entry.initialPose.size());
  if (NumPoseDofs != NumDofs) {
    UE_LOG(
        LogSuperDexSceneImport,
        Warning,
        TEXT(
            "[%s] Initial pose of bot %s has %d values but the bot has %d DOFs; using its "
            "default pose."),
        *SceneName,
        *BotName,
        NumPoseDofs,
        NumDofs);
    return;
  }
  TArray<double> Pose;
  Pose.Reserve(NumDofs);
  for (const mochi::real Q : Entry.initialPose) {
    Pose.Add(Q);
  }
  Component->SetInitialPose(Pose);
}

} // namespace

FSuperDexSceneImporter::FSuperDexSceneImporter(FSuperDexImportSession& InSession)
    : PrefabImporter(InSession),
      BotFactory(NewObject<USuperDexBotFactory>()),
      M2U(kSuperDexSpace) {}

FString FSuperDexSceneImporter::GetSceneName(const FString& SourcePath) {
  return FSuperDexPrefabImporter::GetPrefabName(SourcePath);
}

USuperDexScene* FSuperDexSceneImporter::FindSceneAsset(
    const FString& DestFolder,
    const FString& SceneName) {
  const FString AssetName = TEXT("SDS_") + SceneName;
  return LoadObject<USuperDexScene>(
      nullptr,
      *(FPaths::Combine(DestFolder, AssetName) + TEXT(".") + AssetName),
      nullptr,
      LOAD_NoWarn | LOAD_Quiet);
}

USuperDexScene* FSuperDexSceneImporter::CreateSceneAsset(
    const FString& DestFolder,
    const FString& SceneName) {
  const FString AssetName = TEXT("SDS_") + SceneName;
  USuperDexScene* Scene = NewObject<USuperDexScene>(
      CreatePackage(*FPaths::Combine(DestFolder, AssetName)),
      FName(*AssetName),
      RF_Public | RF_Standalone);
  FAssetRegistryModule::AssetCreated(Scene);
  return Scene;
}

FString FSuperDexSceneImporter::ComputeSourceHash(const FString& SourcePath) {
  FSHA1 Sha;
  const FTCHARToUTF8 Version(kImporterVersion);
  Sha.Update(reinterpret_cast<const uint8*>(Version.Get()), Version.Length());
  TArray<uint8> Bytes;
  if (FFileHelper::LoadFileToArray(Bytes, *SourcePath, FILEREAD_Silent)) {
    Sha.Update(Bytes.GetData(), Bytes.Num());
  }
  Sha.Final();
  FSHAHash Hash;
  Sha.GetHash(Hash.Hash);
  return Hash.ToString();
}

bool FSuperDexSceneImporter::IsUpToDate(const USuperDexScene* Scene) {
  const FString SourcePath = Scene->GetSourceFilePath();
  if (SourcePath.IsEmpty() || !Scene->AssetImportData ||
      Scene->AssetImportData->SourceFileHash != ComputeSourceHash(SourcePath)) {
    return false;
  }
  const bool bOutputPresent = Scene->ImportMode == ESuperDexPrefabImportMode::Blueprint
      ? Scene->GeneratedBlueprint.LoadSynchronous() != nullptr
      : Scene->GeneratedMap.LoadSynchronous() != nullptr;
  if (!bOutputPresent) {
    return false;
  }
  // The scene file is unchanged, so the prefabs and bots it references are the ones recorded.
  if (Scene->BaseScene && !FSuperDexPrefabImporter::IsUpToDate(Scene->BaseScene)) {
    return false;
  }
  for (const USuperDexBot* Bot : Scene->Bots) {
    if (!Bot || Bot->IsOutOfSync()) {
      return false;
    }
  }
  for (const FSuperDexSceneSpawnable& Spawnable : Scene->SpawnablePrefabs) {
    const bool bSpawnableUpToDate = Spawnable.Bot
        ? !Spawnable.Bot->IsOutOfSync()
        : Spawnable.Prefab && FSuperDexPrefabImporter::IsUpToDate(Spawnable.Prefab);
    if (!bSpawnableUpToDate) {
      return false;
    }
  }
  const FString Folder = FPackageName::GetLongPackagePath(Scene->GetOutermost()->GetName());
  for (const FString& RelPath : Scene->AssetImportData->GeneratedAssetPaths) {
    const FString ObjectPath = FPaths::Combine(Folder, RelPath);
    if (!LoadObject<UObject>(nullptr, *ObjectPath, nullptr, LOAD_NoWarn | LOAD_Quiet)) {
      UE_LOG(LogSuperDexSceneImport, Display, TEXT("Generated asset missing: %s"), *ObjectPath);
      return false;
    }
  }
  return true;
}

USuperDexBot* FSuperDexSceneImporter::ImportBot(
    const FString& SourcePath,
    const FString& ParentFolder) {
  if (USuperDexBot** Imported = ImportedBots.Find(SourcePath)) {
    return *Imported;
  }
  USuperDexBot* Bot = BotFactory->ImportBot(
      SourcePath,
      SuperDexImportUtils::GetNestedImportFolder(ParentFolder, SourcePath),
      SuperDexImportUtils::SanitizeForPackagePath(FPaths::GetBaseFilename(SourcePath)),
      RF_Public | RF_Standalone);
  ImportedBots.Add(SourcePath, Bot);
  return Bot;
}

bool FSuperDexSceneImporter::Import(
    USuperDexScene* Scene,
    const FString& SourcePath,
    ESuperDexPrefabImportMode Mode) {
  if (!Scene->AssetImportData) {
    UE_LOG(LogSuperDexSceneImport, Error, TEXT("%s has no import data."), *Scene->GetPathName());
    return false;
  }
  const FString NormalizedSourcePath = SuperDexImportUtils::NormalizePath(SourcePath);
  superdex::robotics::BotScenePrefab BotScene;
  if (!LoadBotScene(NormalizedSourcePath, BotScene)) {
    return false;
  }
  const FString SceneName = GetSceneName(NormalizedSourcePath);
  const FString DestFolder = FPackageName::GetLongPackagePath(Scene->GetOutermost()->GetName());
  UE_LOG(
      LogSuperDexSceneImport,
      Display,
      TEXT("Importing SuperDex scene %s into %s"),
      *NormalizedSourcePath,
      *DestFolder);
  const bool bBlueprint = Mode == ESuperDexPrefabImportMode::Blueprint;
  const FString OutputPath =
      FPaths::Combine(DestFolder, (bBlueprint ? TEXT("BP_") : TEXT("L_")) + SceneName);
  if (FSuperDexPrefabImporter::IsOutputTaken(
          OutputPath,
          bBlueprint ? Scene->GeneratedBlueprint.ToSoftObjectPath()
                     : Scene->GeneratedMap.ToSoftObjectPath())) {
    return false;
  }

  USuperDexPrefab* BaseScene = nullptr;
  UBlueprint* BaseBlueprint = nullptr;
  const FString BaseScenePath = BotScene.scene.baseScene.empty()
      ? FString()
      : SuperDexImportUtils::NormalizePath(ToFString(BotScene.scene.baseScene));
  if (!BaseScenePath.IsEmpty()) {
    BaseScene = PrefabImporter.ImportPrefab(
        BaseScenePath, SuperDexImportUtils::GetNestedImportFolder(DestFolder, BaseScenePath));
    BaseBlueprint = BaseScene ? BaseScene->GeneratedBlueprint.LoadSynchronous() : nullptr;
    if (!BaseBlueprint) {
      UE_LOG(
          LogSuperDexSceneImport,
          Error,
          TEXT("[%s] Failed to import base scene %s."),
          *SceneName,
          *BaseScenePath);
      return false;
    }
  }

  TArray<TObjectPtr<USuperDexBot>> Bots;
  for (const superdex::robotics::BotEntry& Entry : BotScene.bots) {
    const FString Path = SuperDexImportUtils::NormalizePath(ToFString(Entry.path));
    USuperDexBot* Bot = ImportBot(Path, DestFolder);
    if (!Bot) {
      UE_LOG(
          LogSuperDexSceneImport,
          Error,
          TEXT("[%s] Failed to import bot %s from %s."),
          *SceneName,
          *ToFString(Entry.name),
          *Path);
    }
    Bots.Add(Bot);
  }

  TArray<FSuperDexSceneSpawnable> Spawnables;
  for (const superdex::robotics::PrefabEntry& Entry : BotScene.scene.spawnablePrefabs) {
    FSuperDexSceneSpawnable& Spawnable = Spawnables.AddDefaulted_GetRef();
    Spawnable.Name = ToFString(Entry.name);
    const FString Path = SuperDexImportUtils::NormalizePath(ToFString(Entry.path));
    if (IsBotFile(Path)) {
      Spawnable.Bot = ImportBot(Path, DestFolder);
    } else {
      Spawnable.Prefab = PrefabImporter.ImportPrefab(
          Path, SuperDexImportUtils::GetNestedImportFolder(DestFolder, Path));
    }
    if (!Spawnable.Bot && !Spawnable.Prefab) {
      UE_LOG(
          LogSuperDexSceneImport,
          Error,
          TEXT("[%s] Failed to import spawnable %s from %s."),
          *SceneName,
          *Spawnable.Name,
          *Path);
    }
  }
  if (!BotScene.sensors.empty()) {
    WarnUnsupported(SceneName, TEXT("Scene sensors"));
  }

  TArray<FString> GeneratedAssetPaths;
  auto Populate = [&](FSuperDexPrefabImporter::FOutput& Output) {
    if (BaseBlueprint) {
      Output.AddNestedPrefab(BaseBlueprint, BaseScene->PrefabName, FTransform::Identity);
      // Blueprints can't hold scene settings; the base scene's own import already warns about it.
      if (Mode == ESuperDexPrefabImportMode::Map) {
        PrefabImporter.AddPrefabSceneSettings(
            BaseScenePath, DestFolder, Output, GeneratedAssetPaths);
      }
    }
    for (int32 i = 0; i < Bots.Num(); ++i) {
      AddBot(SceneName, M2U, Output, BotScene.bots[i], Bots[i]);
    }
  };
  if (bBlueprint) {
    UBlueprint* Blueprint = FSuperDexPrefabImporter::BuildBlueprint(OutputPath, Populate);
    if (!Blueprint) {
      return false;
    }
    FSuperDexPrefabImporter::RecordGeneratedAsset(DestFolder, Blueprint, GeneratedAssetPaths);
    Scene->GeneratedBlueprint = Blueprint;
    Scene->GeneratedMap.Reset();
  } else {
    UWorld* Map = FSuperDexPrefabImporter::BuildMap(OutputPath, Populate);
    if (!Map) {
      return false;
    }
    FSuperDexPrefabImporter::RecordGeneratedAsset(DestFolder, Map, GeneratedAssetPaths);
    Scene->GeneratedMap = Map;
    Scene->GeneratedBlueprint.Reset();
  }

  Scene->SceneName = SceneName;
  Scene->ImportMode = Mode;
  Scene->BaseScene = BaseScene;
  Scene->Bots = MoveTemp(Bots);
  Scene->SpawnablePrefabs = MoveTemp(Spawnables);
  Scene->AssetImportData->Update(NormalizedSourcePath);
  Scene->AssetImportData->SourceFileHash = ComputeSourceHash(NormalizedSourcePath);
  Scene->AssetImportData->GeneratedAssetPaths = MoveTemp(GeneratedAssetPaths);
  Scene->MarkPackageDirty();
  return true;
}
