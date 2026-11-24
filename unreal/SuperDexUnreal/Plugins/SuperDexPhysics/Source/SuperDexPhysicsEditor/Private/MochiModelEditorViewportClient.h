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

#include "AdvancedPreviewScene.h"
#include "CoreMinimal.h"
#include "MochiModel.h"
#include "MochiModelEditorSelectedElements.h"
#include "MochiSoftMeshComponent.h"
#include "SMochiModelEditorViewport.h"

enum class EMochiModelEditorElementViewMode : int8 {
  None = 0,
  ModifiedYoungsModulus,
  ModifiedPoissonRatio,
  All,
};

class FMochiModelEditorViewportClient : public FEditorViewportClient {
 public:
  /**
   * Create a viewport client for the model preview.
   *
   * @param InEditorViewport Viewport widget associated with this client.
   * @param InPreviewScene Scene containing the preview objects.
   * @param InMochiModel Model asset displayed in the preview.
   * @param InMochiSoftMeshComponent Soft mesh component displayed in the preview.
   */
  FMochiModelEditorViewportClient(
      const TSharedRef<SMochiModelEditorViewport> InEditorViewport,
      const TSharedRef<FAdvancedPreviewScene> InPreviewScene,
      UMochiModel* InMochiModel,
      UMochiSoftMeshComponent* InMochiSoftMeshComponent);

  // Begin FEditorViewportClient Overrides
  // Do not override ShouldOrbitCamera(): drag tools stop working if it is overridden.

  /**
   * NB: The WidgetMode overrides must be implemented, otherwise the camera menu doesn't show up
   *
   * @param NewMode Requested editor viewport widget mode.
   */
  virtual void SetWidgetMode(UE::Widget::EWidgetMode NewMode) override;
  /**
   * returning anything other than WM_None will display the camera controls
   *
   * @return The widget mode.
   */
  virtual UE::Widget::EWidgetMode GetWidgetMode() const override;
  virtual bool CanSetWidgetMode(UE::Widget::EWidgetMode NewMode) const override;
  /**
   * Draw gives us the PDI interface for easy rendering for tools
   * 2D drawing can be done at the Canvas level, see: MochiModelEditorDragTool.cpp
   *
   * @param View Scene view used for editor rendering.
   * @param PDI Primitive drawing interface used for debug geometry.
   */
  virtual void Draw(const FSceneView* View, FPrimitiveDrawInterface* PDI) override;
  virtual TSharedPtr<FDragTool> MakeDragTool(EDragTool::Type DragToolType) override;
  // End FEditorViewportClient Overrides

  // Begin FGCObject Interface
  virtual void AddReferencedObjects(FReferenceCollector& Collector) override;
  virtual FString GetReferencerName() const override {
    return "FMochiModelEditorViewportClient";
  };
  // End FGCObject Interface

  /**
   * Set the model and mesh component shown in the preview.
   *
   * @param InMochiModel Model asset displayed in the preview.
   * @param InPreviewMeshComponent Mesh component displayed in the preview.
   */
  void SetPreviewMesh(UMochiModel* InMochiModel, UMochiSoftMeshComponent* InPreviewMeshComponent);

  /**
   * Show/hide Mochi MeshDescription elements.
   *
   * @param bShowOn Requested visualization enable state.
   */
  void SetShowVolumeVertices(bool bShowOn);
  /** Toggle display of volume-mesh vertices. */
  void ToggleShowVolumeVertices();
  /**
   * Get whether volume-mesh vertices are shown.
   *
   * @return The is show volume vertices checked state.
   */
  bool IsShowVolumeVerticesChecked() const;
  /**
   * Set display of element barycenters.
   *
   * @param bShowOn Requested visualization enable state.
   */
  void SetShowElementBarycenters(bool bShowOn);
  /** Toggle display of element barycenters. */
  void ToggleShowElementBarycenters();
  /**
   * Get whether element barycenters are shown.
   *
   * @return The is show element barycenters checked state.
   */
  bool IsShowElementBarycentersChecked() const;
  /**
   * Set display of selected-element barycenters.
   *
   * @param bShowOn Requested visualization enable state.
   */
  void SetShowSelectedElementBarycenters(bool bShowOn);
  /** Toggle display of selected-element barycenters. */
  void ToggleShowSelectedElementBarycenters();
  /**
   * Get whether selected-element barycenters are shown.
   *
   * @return The is show selected element barycenters checked state.
   */
  bool IsShowSelectedElementBarycentersChecked() const;
  /**
   * Select the element visualization mode.
   *
   * @param NewMode Element visualization mode to select.
   */
  void SetElementViewMode(EMochiModelEditorElementViewMode NewMode);
  /**
   * Get the element visualization mode.
   *
   * @return The element view mode.
   */
  EMochiModelEditorElementViewMode GetElementViewMode() const;

  /**
   * Lasso/Box Selection
   *
   * @return The is using lasso selection state.
   */
  bool IsUsingLassoSelection() {
    return bUseLassoSelection;
  }
  /**
   * Select lasso or box selection.
   *
   * @param bUseLasso Whether selection uses a lasso rather than a box.
   */
  void SetUseLassoSelection(bool bUseLasso) {
    bUseLassoSelection = bUseLasso;
  }

  /**
   * Get the model-element selection used by this viewport.
   *
   * @return The viewport's mutable model-element selection.
   */
  TSet<int>& GetSelectedElements();
  /** Add material modifiers for the current element selection. */
  void AddSelectionToMaterialModifiers();
  /** Remove material modifiers for the current element selection. */
  void RemoveSelectionFromMaterialModifiers();

  /**
   * Get the object storing the model-element selection.
   *
   * @return The selected elements container.
   */
  UMochiModelEditorSelectedElements* GetSelectedElementsContainer() const;

  TWeakPtr<SMochiModelEditorViewport> MochiViewportPtr;

  UE::Widget::EWidgetMode WidgetMode;

  const float MaxYoungsModulusDisplay = 2000.0f;
  const float MaxPoissonRatioDisplay = 0.5f;
  const FLinearColor MaxElementColor = FLinearColor(0.0, 1.0f, 0.0f);
  const FLinearColor MinElementColor = FLinearColor(1.0, 0.0f, 0.0f);
  const float DefaultFOV = 90.0f;
  const float ElementPointSize = 5.0f;
  const float SelectedElementPointSize = 10.0f;

 private:
  FAdvancedPreviewScene* AdvancedPreviewScene;
  UMochiModel* PreviewModel;
  UMochiSoftMeshComponent* PreviewMeshComponent;
  SMochiModelEditorViewport* MochiViewport;

  TObjectPtr<UMochiModelEditorSelectedElements> SelectedElementsContainer;

  bool bDrawVolumeVertices = false;
  bool bDrawElementBarycenters = true;
  bool bDrawSelectedElementBarycenters = true;

  EMochiModelEditorElementViewMode ElementViewMode =
      EMochiModelEditorElementViewMode::ModifiedYoungsModulus;

  bool bUseLassoSelection = true;
};
