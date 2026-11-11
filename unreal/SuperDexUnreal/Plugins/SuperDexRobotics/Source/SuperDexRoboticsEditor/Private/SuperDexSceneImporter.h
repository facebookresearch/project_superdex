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
#include "MochiConversions.h"
#include "SuperDexBotFactory.h"
#include "SuperDexPrefab.h"
#include "SuperDexPrefabImporter.h"
#include "UObject/StrongObjectPtr.h"

class FSuperDexImportSession;
class USuperDexBot;
class USuperDexScene;

// Imports .superdex_scene and .superdex_scene_archive files authored in SuperDex Studio into
// USuperDexScene assets and a Blueprint or Map that places the base scene and bots. The base scene
// and spawnable prefabs are imported as USuperDexPrefab Blueprints, and bots as USuperDexBot
// assets, each in a subfolder of the scene's folder.
class FSuperDexSceneImporter {
 public:
  explicit FSuperDexSceneImporter(FSuperDexImportSession& InSession);

  // Import the scene at SourcePath into Scene, generating assets next to it. Returns false if
  // the scene could not be imported.
  bool Import(USuperDexScene* Scene, const FString& SourcePath, ESuperDexPrefabImportMode Mode);

  // Name of the scene at SourcePath, derived from its file name.
  static FString GetSceneName(const FString& SourcePath);
  // Find the USuperDexScene asset of a scene in its destination folder, or nullptr.
  static USuperDexScene* FindSceneAsset(const FString& DestFolder, const FString& SceneName);
  // Create the USuperDexScene asset of a scene in its destination folder.
  static USuperDexScene* CreateSceneAsset(const FString& DestFolder, const FString& SceneName);
  // Hash of the scene file at SourcePath. The prefabs and bots it references are checked by
  // their own assets.
  static FString ComputeSourceHash(const FString& SourcePath);
  // Whether the scene, its prefabs and bots are up to date and its generated assets exist.
  static bool IsUpToDate(const USuperDexScene* Scene);

 private:
  // Import the bot at SourcePath into a subfolder of ParentFolder.
  USuperDexBot* ImportBot(const FString& SourcePath, const FString& ParentFolder);

  FSuperDexPrefabImporter PrefabImporter;
  TStrongObjectPtr<USuperDexBotFactory> BotFactory;
  // SuperDex Studio space (right-handed, X forward, Y left, Z up, meters) to Unreal.
  FMochiToUnrealConverter M2U;
  // Bots imported by this importer, by source path.
  TMap<FString, USuperDexBot*> ImportedBots;
};
