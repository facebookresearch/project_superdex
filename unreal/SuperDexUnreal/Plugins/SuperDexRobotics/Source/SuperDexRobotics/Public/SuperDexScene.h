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
#include "SuperDexPrefab.h"
#include "UObject/Object.h"
#include "UObject/SoftObjectPtr.h"

#include "SuperDexScene.generated.h"

class UBlueprint;
class USuperDexBot;
class UWorld;

// A prefab or bot that a scene can spawn at runtime.
USTRUCT(BlueprintType)
struct SUPERDEXROBOTICS_API FSuperDexSceneSpawnable {
  GENERATED_BODY()

  // Name the scene gives the spawnable.
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Spawnable")
  FString Name;
  // The imported prefab, if the spawnable is a SuperDex prefab or scene.
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Spawnable")
  TObjectPtr<USuperDexPrefab> Prefab;
  // The imported bot, if the spawnable is a bot.
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Spawnable")
  TObjectPtr<USuperDexBot> Bot;
};

// Imported Unreal asset representing a .superdex_scene (or .superdex_scene_archive) authored in
// SuperDex Studio. It records where the scene came from and owns the Blueprint or Map generated
// from it, which are regenerated when the scene is reimported.
UCLASS(BlueprintType, HideCategories = Object, DisplayName = "SuperDex Scene")
class SUPERDEXROBOTICS_API USuperDexScene : public UObject {
  GENERATED_BODY()
 public:
  // Name of the scene, derived from its file name.
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, AssetRegistrySearchable, Category = "Scene")
  FString SceneName;
  // What the scene was imported as.
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Scene")
  ESuperDexPrefabImportMode ImportMode = ESuperDexPrefabImportMode::Map;
  // The generated Blueprint (Blueprint import mode).
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Scene")
  TSoftObjectPtr<UBlueprint> GeneratedBlueprint;
  // The generated level (Map import mode).
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Scene")
  TSoftObjectPtr<UWorld> GeneratedMap;
  // The base scene, imported as a prefab Blueprint, or null if the scene has none.
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Scene")
  TObjectPtr<USuperDexPrefab> BaseScene;
  // The bots, in scene order. Null where a bot failed to import.
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Scene")
  TArray<TObjectPtr<USuperDexBot>> Bots;
  // The prefabs and bots the scene can spawn at runtime. They are imported but not placed.
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Scene")
  TArray<FSuperDexSceneSpawnable> SpawnablePrefabs;
#if WITH_EDITORONLY_DATA
  // Source file and generated asset tracking for reimport.
  UPROPERTY(VisibleAnywhere, Category = "Internal")
  TObjectPtr<USuperDexPrefabImportData> AssetImportData;
#endif

 public:
#if WITH_EDITOR
  virtual void PostInitProperties() override;
  virtual void GetAssetRegistryTags(FAssetRegistryTagsContext Context) const override;
  // Absolute path of the source scene file.
  FString GetSourceFilePath() const;
#endif
};
