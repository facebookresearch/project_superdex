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
#include "Misc/EngineVersionComparison.h"
#include "MochiSoftMeshComponent.h"
#include "SAssetEditorViewport.h"
#include "SMochiModelEditorViewportToolbar.h"
#include "Widgets/Text/SRichTextBlock.h"

class FMochiModelEditorToolkit;
class FMochiModelPreviewScene;

enum class EMochiModelEditorSelectionMode : int8 {
  ClearFirst,
  Additive,
  Subtractive,
};

class SMochiModelEditorViewport : public SAssetEditorViewport,
                                  public FGCObject,
                                  public ICommonEditorViewportToolbarInfoProvider {
 public:
  SLATE_BEGIN_ARGS(SMochiModelEditorViewport) {}
  SLATE_END_ARGS()

  TSharedPtr<FMochiModelPreviewScene> ModelPreviewScene;

  void Construct(
      const FArguments& InArgs,
      TSharedPtr<FMochiModelEditorToolkit> InEditorToolkit,
      TSharedPtr<FMochiModelPreviewScene> InPreviewScene);
  virtual ~SMochiModelEditorViewport() override;

  virtual void BindCommands() override;
  virtual TSharedRef<SEditorViewport> GetViewportWidget() override;
  virtual TSharedPtr<FExtender> GetExtenders() const override;
  // Called when the Viewport's hamburger menu or "Perspective" menu are clicked
  // Currently a no-op.
  virtual void OnFloatingButtonClicked() override {};
  virtual void OnFocusViewportToSelection() override;
  virtual TSharedRef<FEditorViewportClient> MakeEditorViewportClient() override;
#if UE_VERSION_OLDER_THAN(5, 7, 0)
  virtual TSharedPtr<SWidget> MakeViewportToolbar() override;
#endif
  // Adds more toolbars or other overlays to the inside of the viewport
  virtual void PopulateViewportOverlays(TSharedRef<SOverlay> Overlay) override;
  // Makes the viewport toolbar. SEditorViewport adds it through MakeViewportToolbar() before
  // UE 5.7; from 5.7 that is deprecated and final, so PopulateViewportOverlays() adds it.
  TSharedRef<SWidget> MakeLegacyViewportToolbar();

  // Begin FGCObject Interface
  virtual void AddReferencedObjects(FReferenceCollector& Collector) override;
  virtual FString GetReferencerName() const override {
    return "SMochiModelEditorViewport";
  };
  // End FGCObject Interface

  TSharedPtr<class FMochiModelEditorViewportClient> GetViewportClient() {
    return ViewportClient;
  }

  TSharedPtr<FMochiModelEditorToolkit> EditorToolkit;

  TArray<FVector> ElementBarycenterPositions;
  TArray<FVector> VolumeNodePositions;

  UMochiSoftMeshComponent* GetPreviewMeshComponent() {
    return PreviewMeshComponent;
  }

  bool IsUsingLassoSelection() const;
  bool IsUsingBoxSelection() const;
  void SetSelectLasso(bool bUseLasso);

  // Specifies an array of text items which will be added to the viewport overlay
  void PopulateOverlayText(const TArray<FText>& TextItems);
  EMochiModelEditorSelectionMode GetSelectionMode() const;
  void SetSelectionMode(EMochiModelEditorSelectionMode NewSelectionMode);
  bool IsUsingOneShotSelection() const;
  bool IsUsingAdditiveSelection() const;
  bool IsUsingSubtractiveSelection() const;

  void SetUseOrthographicView(bool bOrthographic);
  bool IsUsingOrthographicView();
  bool IsUsingPerspectiveView();

 private:
  // Pointer to the box into which the overlay text items are added
  TSharedPtr<SRichTextBlock> OverlayText;
  // The viewport widget handles the Slate side; the viewport client handles rendering and
  // interaction.
  TSharedPtr<FMochiModelEditorViewportClient> ViewportClient;

  // This doesn't necessarily be the DynamicMeshComponent, but using it
  // for convenience.
  TObjectPtr<UMochiSoftMeshComponent> PreviewMeshComponent;

  EMochiModelEditorSelectionMode SelectionMode = EMochiModelEditorSelectionMode::ClearFirst;
  bool bUseOrthographic = false;

  // 5.0f is the minimum FOV you can set
  float kOrthographicFOV = 5.0f;
  // this is the same as the default FOV for most UE cameras
  float kPerspectiveFOV = 90.0f;
};
