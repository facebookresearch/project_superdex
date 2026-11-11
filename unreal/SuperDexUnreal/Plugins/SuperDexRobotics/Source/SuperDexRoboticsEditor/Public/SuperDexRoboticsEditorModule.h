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

#include "AssetTypeActions_Base.h"
#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class FSuperDexClassIconStyle;

/// SuperDexRobotics editor module. Owns everything bot-specific: bot asset actions/factories/
/// thumbnails, and the "Export SuperDex Bot Scene..."/"Reimport Bots" Mochi-menu entries.
class FSuperDexRoboticsEditorModule : public IModuleInterface {
 public:
  static inline FSuperDexRoboticsEditorModule& Get() {
    return FModuleManager::LoadModuleChecked<FSuperDexRoboticsEditorModule>(
        "SuperDexRoboticsEditor");
  }
  static inline bool IsLoaded() {
    return FModuleManager::Get().IsModuleLoaded("SuperDexRoboticsEditor");
  }
  virtual void StartupModule() override;
  virtual void ShutdownModule() override;

  // Registers the "Split Mochi Children..." Content Browser context-menu entry for
  // Blueprints.
  void RegisterAssetContextMenu();

  // Extends the "Mochi" submenu (registered by FMochiPhysicsEditorModule) with the
  // bot-specific menu entries. Registered as its own UToolMenus startup callback, so it
  // does not require FMochiPhysicsEditorModule to know about SuperDexRobotics at all.
  void RegisterSuperDexRoboticsMenu();

  // Absent from an externally built plugin: bot-scene authoring is stripped from the export.
  // Exports a .superdex_scene_archive from the current level (requires PIE)
  static void ExecuteExportSuperDexBotScene();

  // Exports a .superdex_scene_archive of the live scene: captures the current scene
  // (task + spawnables) into the base scene instead of the two-phase split.
  static void ExecuteExportLiveSuperDexBotScene();

  // Shared implementation for the bot-scene export menu entries.
  static void ExecuteExportSuperDexBotSceneInternal(bool bCaptureLiveScene);

  // Scans all SuperDexBot assets and reimports those whose source files have changed
  static void ExecuteReimportOutOfSyncBots();

 private:
  TArray<TSharedRef<IAssetTypeActions>> RegisteredAssetTypeActions;

  // Class icons and thumbnails of the SuperDexRobotics asset types.
  TSharedPtr<FSuperDexClassIconStyle> ClassIconStyle;
};
