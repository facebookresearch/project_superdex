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

/**
 * This is the "entry point" for editing Mochi Models
 * All Viewports, Toolbars, are generated here.
 */
class FMochiModelEditorToolkit : public FAssetEditorToolkit, public FNotifyHook {
 private:
  using Super = FAssetEditorToolkit;

 public:
  /** Initialize the model asset editor toolkit. */
  FMochiModelEditorToolkit();
  /** Release the model asset editor toolkit. */
  virtual ~FMochiModelEditorToolkit();

  /**
   * IToolkit Interface
   *
   * @return The toolkit fname.
   */
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

  /** Bind model-editor actions to commands. */
  void BindCommands();
  /** Add model-editor toolbar extensions. */
  void ExtendToolBars();
  /** Focus the model preview viewport. */
  void FocusViewport() const;
  /** Add material modifiers for the selected model elements. */
  void AddMaterialModifiers();
  /** Remove material modifiers for the selected model elements. */
  void RemoveMaterialModifiers();
  /**
   * Create the model editor's preview scene.
   *
   * @return The object returned by creation.
   */
  TSharedPtr<FMochiModelPreviewScene> CreatePreviewScene();

  virtual void RegisterTabSpawners(const TSharedRef<class FTabManager>& InTabManager) override;
  virtual void UnregisterTabSpawners(const TSharedRef<class FTabManager>& InTabManager) override;
  /**
   * Get the model editor's tab layout.
   *
   * @return The default layout.
   */
  const TSharedRef<FTabManager::FLayout> GetDefaultLayout() const;
  virtual void CreateEditorModeManager() override;

  /**
   * Hook onto FCoreUObjectDelegates::OnObjectModified to detect when the MochiModel is modified
   *
   * @param InObject Object whose modification is being handled.
   */
  void OnObjectModified(UObject* InObject);

  /**
   * Open the model asset in the requested editor mode and host.
   *
   * @param Mode Editor or prefab import mode.
   * @param InitToolkitHost Host for the asset editor.
   * @param ObjectToEdit Asset to open in the editor.
   */
  void InitMochiModelEditor(
      const EToolkitMode::Type Mode,
      const TSharedPtr<class IToolkitHost>& InitToolkitHost,
      UMochiModel* ObjectToEdit);

  // Begin FNotifyHook Interface
  virtual void NotifyPostChange(
      const FPropertyChangedEvent& PropertyChangedEvent,
      FProperty* PropertyThatChanged) override;
  // End FNotifyHookInterface

  /**
   * Get the model asset being edited.
   *
   * @return The model.
   */
  TObjectPtr<UMochiModel> GetModel();

 public:
  static const FName ViewportTabID;
  static const FName DetailsTabID;

 protected:
  void RegisterToolbar();

  /** Viewport Spawner */
  TSharedRef<SDockTab> SpawnTab_Viewport(const FSpawnTabArgs& Args);

  /** Detail Panel Spawner */
  virtual TSharedRef<SDockTab> SpawnTab_Details(const FSpawnTabArgs& Args);

 protected:
  /** Property View */
  TSharedPtr<IDetailsView> DetailsView;

  /** Tracking the active viewports in this editor. */
  TSharedPtr<FEditorViewportTabContent> ViewportTabContent;

  /** Extender for adding to the default layout for this asset editor */
  TSharedPtr<FLayoutExtender> LayoutExtender;
  TSharedPtr<FTabManager::FLayout> StandaloneDefaultLayout;

  /** Reference to the MochiModel currently being viewed */
  TObjectPtr<UMochiModel> MochiModel;

  /** Scene and mesh component for preview rendering (note these are garbage-collected) */
  TSharedPtr<FMochiModelPreviewScene> ModelPreviewScene;

  TSharedPtr<SMochiModelEditorViewport> ModelEditorViewport;

  virtual void OnClose() override;

  EMochiModelEditorSelectionMode GetViewportSelectionMode() const;
  void SetViewportSelectionMode(EMochiModelEditorSelectionMode NewSelectionMode) const;

  EMochiModelEditorElementViewMode GetViewportElementViewMode() const;
  void SetViewportElementViewMode(EMochiModelEditorElementViewMode NewViewMode) const;

 public:
  /**
   * Called from FMochiModelAssetActions::OpenAssetEditor()
   *
   * @param Mode Editor or prefab import mode.
   * @param InitToolkitHost Host for the asset editor.
   * @param ObjectToEdit Asset to open in the editor.
   * @return The newly initialized asset editor.
   */
  static TSharedRef<FMochiModelEditorToolkit> CreateEditor(
      const EToolkitMode::Type Mode,
      const TSharedPtr<IToolkitHost>& InitToolkitHost,
      UMochiModel* ObjectToEdit);
};
