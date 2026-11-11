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

#include "AssetEditorViewportLayout.h"
#include "CoreMinimal.h"
#include "Misc/NotifyHook.h"
#include "MochiModel.h"
#include "SMochiModelEditorViewport.h"

enum class EMochiModelEditorElementViewMode : int8;
class FMochiModelPreviewScene;

// This is the "entry point" for editing Mochi Models
// All Viewports, Toolbars, are generated here.
class FMochiModelEditorToolkit : public FAssetEditorToolkit, public FNotifyHook {
 private:
  using Super = FAssetEditorToolkit;

 public:
  FMochiModelEditorToolkit();
  virtual ~FMochiModelEditorToolkit();

  // IToolkit Interface
  virtual FName GetToolkitFName() const override {
    return FName("MochiModelEditor");
  }
  virtual FText GetBaseToolkitName() const override {
    return FText::FromString("MochiModel Editor");
  }
  virtual FString GetWorldCentricTabPrefix() const override {
    return TEXT("MochiModel ");
  }
  virtual FLinearColor GetWorldCentricTabColorScale() const override {
    return FLinearColor::White;
  }

  void BindCommands();
  void ExtendToolBars();
  void FocusViewport() const;
  void AddMaterialModifiers();
  void RemoveMaterialModifiers();
  TSharedPtr<FMochiModelPreviewScene> CreatePreviewScene();

  virtual void RegisterTabSpawners(const TSharedRef<class FTabManager>& InTabManager) override;
  virtual void UnregisterTabSpawners(const TSharedRef<class FTabManager>& InTabManager) override;
  const TSharedRef<FTabManager::FLayout> GetDefaultLayout() const;
  virtual void CreateEditorModeManager() override;

  // Hook onto FCoreUObjectDelegates::OnObjectModified to detect when the MochiModel is modified
  void OnObjectModified(UObject* InObject);

  void InitMochiModelEditor(
      const EToolkitMode::Type Mode,
      const TSharedPtr<class IToolkitHost>& InitToolkitHost,
      UMochiModel* ObjectToEdit);

  // Begin FNotifyHook Interface
  virtual void NotifyPostChange(
      const FPropertyChangedEvent& PropertyChangedEvent,
      FProperty* PropertyThatChanged) override;
  // End FNotifyHookInterface

  TObjectPtr<UMochiModel> GetModel();

 public:
  static const FName ViewportTabID;
  static const FName DetailsTabID;

 protected:
  void RegisterToolbar();

  // Viewport Spawner
  TSharedRef<SDockTab> SpawnTab_Viewport(const FSpawnTabArgs& Args);
  // Detail Panel Spawner
  virtual TSharedRef<SDockTab> SpawnTab_Details(const FSpawnTabArgs& Args);

 protected:
  // Property View
  TSharedPtr<IDetailsView> DetailsView;
  // Tracking the active viewports in this editor.
  TSharedPtr<FEditorViewportTabContent> ViewportTabContent;
  // Extender for adding to the default layout for this asset editor
  TSharedPtr<FLayoutExtender> LayoutExtender;
  TSharedPtr<FTabManager::FLayout> StandaloneDefaultLayout;

  // Reference to the MochiModel currently being viewed
  TObjectPtr<UMochiModel> MochiModel;

  // Scene and mesh component for preview rendering (note these are garbage-collected)
  TSharedPtr<FMochiModelPreviewScene> ModelPreviewScene;

  TSharedPtr<SMochiModelEditorViewport> ModelEditorViewport;

  virtual void OnClose() override;

  EMochiModelEditorSelectionMode GetViewportSelectionMode() const;
  void SetViewportSelectionMode(EMochiModelEditorSelectionMode NewSelectionMode) const;

  EMochiModelEditorElementViewMode GetViewportElementViewMode() const;
  void SetViewportElementViewMode(EMochiModelEditorElementViewMode NewViewMode) const;

 public:
  // Called from FMochiModelAssetActions::OpenAssetEditor()
  static TSharedRef<FMochiModelEditorToolkit> CreateEditor(
      const EToolkitMode::Type Mode,
      const TSharedPtr<IToolkitHost>& InitToolkitHost,
      UMochiModel* ObjectToEdit);
};
