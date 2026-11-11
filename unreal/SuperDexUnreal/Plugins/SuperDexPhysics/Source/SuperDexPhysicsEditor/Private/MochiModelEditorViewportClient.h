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
  FMochiModelEditorViewportClient(
      const TSharedRef<SMochiModelEditorViewport> InEditorViewport,
      const TSharedRef<FAdvancedPreviewScene> InPreviewScene,
      UMochiModel* InMochiModel,
      UMochiSoftMeshComponent* InMochiSoftMeshComponent);

  // Begin FEditorViewportClient Overrides
  // Do not override ShouldOrbitCamera(): drag tools stop working if it is overridden.

  // NB: The WidgetMode overrides must be implemented, otherwise the camera menu doesn't show up
  virtual void SetWidgetMode(UE::Widget::EWidgetMode NewMode) override;
  // returning anything other than WM_None will display the camera controls
  virtual UE::Widget::EWidgetMode GetWidgetMode() const override;
  virtual bool CanSetWidgetMode(UE::Widget::EWidgetMode NewMode) const override;
  // Draw gives us the PDI interface for easy rendering for tools
  // 2D drawing can be done at the Canvas level, see: MochiModelEditorDragTool.cpp
  virtual void Draw(const FSceneView* View, FPrimitiveDrawInterface* PDI) override;
  virtual TSharedPtr<FDragTool> MakeDragTool(EDragTool::Type DragToolType) override;
  // End FEditorViewportClient Overrides

  // Begin FGCObject Interface
  virtual void AddReferencedObjects(FReferenceCollector& Collector) override;
  virtual FString GetReferencerName() const override {
    return "FMochiModelEditorViewportClient";
  };
  // End FGCObject Interface

  void SetPreviewMesh(UMochiModel* InMochiModel, UMochiSoftMeshComponent* InPreviewMeshComponent);

  // Show/hide Mochi MeshDescription elements.
  void SetShowVolumeVertices(bool bShowOn);
  void ToggleShowVolumeVertices();
  bool IsShowVolumeVerticesChecked() const;
  void SetShowElementBarycenters(bool bShowOn);
  void ToggleShowElementBarycenters();
  bool IsShowElementBarycentersChecked() const;
  void SetShowSelectedElementBarycenters(bool bShowOn);
  void ToggleShowSelectedElementBarycenters();
  bool IsShowSelectedElementBarycentersChecked() const;
  void SetElementViewMode(EMochiModelEditorElementViewMode NewMode);
  EMochiModelEditorElementViewMode GetElementViewMode() const;

  // Lasso/Box Selection
  bool IsUsingLassoSelection() {
    return bUseLassoSelection;
  }
  void SetUseLassoSelection(bool bUseLasso) {
    bUseLassoSelection = bUseLasso;
  }

  TSet<int>& GetSelectedElements();
  void AddSelectionToMaterialModifiers();
  void RemoveSelectionFromMaterialModifiers();

  UMochiModelEditorSelectedElements* GetSelectedElementsContainer() const;

  TWeakPtr<SMochiModelEditorViewport> MochiViewportPtr;

  UE::Widget::EWidgetMode WidgetMode;

  const float kMaxYoungsModulusDisplay = 2000.0f;
  const float kMaxPoissonRatioDisplay = 0.5f;
  const FLinearColor kMaxElementColor = FLinearColor(0.0, 1.0f, 0.0f);
  const FLinearColor kMinElementColor = FLinearColor(1.0, 0.0f, 0.0f);
  const float kDefaultFOV = 90.0f;
  const float kElementPointSize = 5.0f;
  const float kSelectedElementPointSize = 10.0f;

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
