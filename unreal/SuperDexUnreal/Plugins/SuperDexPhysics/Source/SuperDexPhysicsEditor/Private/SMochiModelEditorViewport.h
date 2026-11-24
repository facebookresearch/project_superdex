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

  /**
   * Construct the model preview widget and its editor connections.
   *
   * @param InArgs Slate widget construction arguments.
   * @param InEditorToolkit Toolkit associated with the viewport widget.
   * @param InPreviewScene Scene containing the preview objects.
   */
  void Construct(
      const FArguments& InArgs,
      TSharedPtr<FMochiModelEditorToolkit> InEditorToolkit,
      TSharedPtr<FMochiModelPreviewScene> InPreviewScene);
  virtual ~SMochiModelEditorViewport() override;

  virtual void BindCommands() override;
  virtual TSharedRef<SEditorViewport> GetViewportWidget() override;
  virtual TSharedPtr<FExtender> GetExtenders() const override;

  /**
   * Called when the Viewport's hamburger menu or "Perspective" menu are clicked
   * Currently a no-op.
   */
  virtual void OnFloatingButtonClicked() override {};
  virtual void OnFocusViewportToSelection() override;
  virtual TSharedRef<FEditorViewportClient> MakeEditorViewportClient() override;
#if UE_VERSION_OLDER_THAN(5, 7, 0)
  virtual TSharedPtr<SWidget> MakeViewportToolbar() override;
#endif
  /**
   * Adds more toolbars or other overlays to the inside of the viewport
   *
   * @param Overlay Viewport overlay container to populate.
   */
  virtual void PopulateViewportOverlays(TSharedRef<SOverlay> Overlay) override;

  /**
   * Makes the viewport toolbar. SEditorViewport adds it through MakeViewportToolbar() before
   * UE 5.7; from 5.7 that is deprecated and final, so PopulateViewportOverlays() adds it.
   *
   * @return The viewport toolbar widget.
   */
  TSharedRef<SWidget> MakeLegacyViewportToolbar();

  // Begin FGCObject Interface
  virtual void AddReferencedObjects(FReferenceCollector& Collector) override;
  virtual FString GetReferencerName() const override {
    return "SMochiModelEditorViewport";
  };
  // End FGCObject Interface

  /**
   * Get the client controlling this preview viewport.
   *
   * @return The viewport client.
   */
  TSharedPtr<class FMochiModelEditorViewportClient> GetViewportClient() {
    return ViewportClient;
  }

  TSharedPtr<FMochiModelEditorToolkit> EditorToolkit;

  TArray<FVector> ElementBarycenterPositions;
  TArray<FVector> VolumeNodePositions;

  /**
   * Get the mesh component displayed in the preview.
   *
   * @return The preview mesh component.
   */
  UMochiSoftMeshComponent* GetPreviewMeshComponent() {
    return PreviewMeshComponent;
  }

  /**
   * Get whether lasso selection is active.
   *
   * @return The is using lasso selection state.
   */
  bool IsUsingLassoSelection() const;

  /**
   * Get whether box selection is active.
   *
   * @return The is using box selection state.
   */
  bool IsUsingBoxSelection() const;

  /**
   * Select lasso or box selection.
   *
   * @param bUseLasso Whether selection uses a lasso rather than a box.
   */
  void SetSelectLasso(bool bUseLasso);

  /**
   * Specifies an array of text items which will be added to the viewport overlay
   *
   * @param TextItems Text entries displayed in the viewport overlay.
   */
  void PopulateOverlayText(const TArray<FText>& TextItems);

  /**
   * Get the element-selection mode.
   *
   * @return The selection mode.
   */
  EMochiModelEditorSelectionMode GetSelectionMode() const;

  /**
   * Set the element-selection mode.
   *
   * @param NewSelectionMode Element selection behavior to use.
   */
  void SetSelectionMode(EMochiModelEditorSelectionMode NewSelectionMode);

  /**
   * Get whether selection replaces the current selection.
   *
   * @return The is using one shot selection state.
   */
  bool IsUsingOneShotSelection() const;

  /**
   * Get whether selection adds to the current selection.
   *
   * @return The is using additive selection state.
   */
  bool IsUsingAdditiveSelection() const;

  /**
   * Get whether selection removes from the current selection.
   *
   * @return The is using subtractive selection state.
   */
  bool IsUsingSubtractiveSelection() const;

  /**
   * Select an orthographic or perspective view.
   *
   * @param bOrthographic Whether to select an orthographic view.
   */
  void SetUseOrthographicView(bool bOrthographic);

  /**
   * Get whether the view is orthographic.
   *
   * @return The is using orthographic view state.
   */
  bool IsUsingOrthographicView();

  /**
   * Get whether the view is perspective.
   *
   * @return The is using perspective view state.
   */
  bool IsUsingPerspectiveView();

 private:
  /** Pointer to the box into which the overlay text items are added */
  TSharedPtr<SRichTextBlock> OverlayText;

  /**
   * The viewport widget handles the Slate side; the viewport client handles rendering and
   * interaction.
   */
  TSharedPtr<FMochiModelEditorViewportClient> ViewportClient;

  /**
   * This doesn't necessarily be the DynamicMeshComponent, but using it
   * for convenience.
   */
  TObjectPtr<UMochiSoftMeshComponent> PreviewMeshComponent;

  EMochiModelEditorSelectionMode SelectionMode = EMochiModelEditorSelectionMode::ClearFirst;
  bool bUseOrthographic = false;

  /** 5.0f is the minimum FOV you can set */
  float OrthographicFOV = 5.0f;

  /** this is the same as the default FOV for most UE cameras */
  float PerspectiveFOV = 90.0f;
};
