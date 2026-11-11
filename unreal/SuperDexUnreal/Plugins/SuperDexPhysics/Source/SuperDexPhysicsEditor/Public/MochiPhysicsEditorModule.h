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
#include "MochiAssetsEditor.h"
#include "MochiDynamicMeshUtil.h"
#include "MochiModel.h"
#include "Modules/ModuleManager.h"

#include "MochiPhysicsEditorModule.generated.h"

class FMochiModelAssetActions;
// Exported (not just extern) so sibling editor modules (e.g. SuperDexRoboticsEditor) that were
// split out of this module can keep logging to the shared "Mochi Physics Editor" category.
SUPERDEXPHYSICSEDITOR_API DECLARE_LOG_CATEGORY_EXTERN(LogMochiPhysicsEditor, Log, All);

class FMochiPhysicsEditorModule : public IModuleInterface {
 public:
  static inline FMochiPhysicsEditorModule& Get() {
    return FModuleManager::LoadModuleChecked<FMochiPhysicsEditorModule>("SuperDexPhysicsEditor");
  }
  static inline bool IsLoaded() {
    return FModuleManager::Get().IsModuleLoaded("SuperDexPhysicsEditor");
  }
  virtual void StartupModule() override;
  virtual void ShutdownModule() override;

  // Registers the Mochi menu in the Level Editor's main menu bar
  void RegisterMochiMenu();

  // Exports the current Mochi scene to a file (opens file dialog)
  static void ExecuteExportMochiScene();

  // Returns true if a PIE session is active (for menu item availability). Exported since
  // the sibling SuperDexRoboticsEditor module
  // (FSuperDexRoboticsEditorModule::RegisterSuperDexRoboticsMenu/ExecuteExportSuperDexBotScene)
  // also gates its bot-scene export on this same PIE check.
  SUPERDEXPHYSICSEDITOR_API static bool CanExecuteExportMochiScene();

  virtual TSharedPtr<class FExtensibilityManager> GetEditorToolbarExtensibilityManager() {
    return ToolbarExtensibilityManager;
  }

 private:
  template <typename TAssetActions>
  void RegisterAssetActions(IAssetTools& AssetTools, EAssetTypeCategories::Type Category) {
    const TSharedRef<IAssetTypeActions> actions = MakeShareable(new TAssetActions(Category));
    AssetTools.RegisterAssetTypeActions(actions);
    RegisteredAssetTypeActions.Add(actions);
  }
  void RegisterAssetContextMenu();

  static void ExecuteExportMochiModelToHDF5(UMochiModel* MochiModel, FName BaseName);

 private:
  TArray<TSharedRef<IAssetTypeActions>> RegisteredAssetTypeActions;
  FMochiPhysicsEditorStyle MochiPhysicsEditorStyle;

  TSharedPtr<FExtensibilityManager> ToolbarExtensibilityManager;
};

// Options for baking SDF into Mochi models
UCLASS(HideCategories = Object, MinimalAPI, DisplayName = "Bake SDF Options")
class UBakeSdfOptions : public UObject {
  GENERATED_BODY()
 public:
  UPROPERTY(EditAnywhere, Category = "SDF")
  FMochiGridSdfParams GridSdfParams;
};
