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

#include "MochiModelEditorViewportClient.h"

#include "CameraController.h"
#include "MochiModelEditorDragTool.h"
#include "MochiPhysicsEditorModule.h"
#include "ScopedTransaction.h"

#define LOCTEXT_NAMESPACE "FMochiPhysicsEditorModule"

FMochiModelEditorViewportClient::FMochiModelEditorViewportClient(
    const TSharedRef<SMochiModelEditorViewport> InEditorViewport,
    const TSharedRef<FAdvancedPreviewScene> InPreviewScene,
    UMochiModel* InMochiModel,
    UMochiSoftMeshComponent* InMochiSoftMeshComponent)
    : FEditorViewportClient(
          nullptr,
          &InPreviewScene.Get(),
          StaticCastSharedRef<SMochiModelEditorViewport>(InEditorViewport)),
      MochiViewportPtr(InEditorViewport) {
  SetRealtime(true);

  // we might want to customize these later, via a button or menu
  DrawHelper.bDrawGrid = false;
  DrawHelper.bDrawPivot = false;
  DrawHelper.AxesLineThickness = 5;
  DrawHelper.PivotSize = 5;

  WidgetMode = UE::Widget::WM_None;

  SetViewLocation(FVector(75.0, 75.0, 75.0));
  SetViewRotation(FVector(-75.0, -75.0, -75.0).Rotation());

  ELevelViewportType NewViewportType = LVT_Perspective;
  FEditorViewportClient::SetViewportType(NewViewportType);

  bDrawVolumeVertices = false;

  AdvancedPreviewScene = static_cast<FAdvancedPreviewScene*>(PreviewScene);

  SetPreviewMesh(InMochiModel, InMochiSoftMeshComponent);

  SetViewModes(VMI_Lit, VMI_Lit);

  SelectedElementsContainer =
      NewObject<UMochiModelEditorSelectedElements>(GetTransientPackage(), NAME_None, RF_Transient);
}

void FMochiModelEditorViewportClient::SetWidgetMode(UE::Widget::EWidgetMode NewMode) {
  FEditorViewportClient::SetWidgetMode(NewMode);
  WidgetMode = NewMode;

  Invalidate();
}

UE::Widget::EWidgetMode FMochiModelEditorViewportClient::GetWidgetMode() const {
  // TODO: Support transform widgets for moving volumes.

  return UE::Widget::WM_Max;
}

bool FMochiModelEditorViewportClient::CanSetWidgetMode(UE::Widget::EWidgetMode NewMode) const {
  // TODO: Implement logic for locking the user from exiting out of a Widget Mode

  return true;
}

void FMochiModelEditorViewportClient::Draw(const FSceneView* View, FPrimitiveDrawInterface* PDI) {
  FEditorViewportClient::Draw(View, PDI);

  UMochiModelMaterialModifiers* ModifierAsset = PreviewModel->MaterialModifiersAsset;

  if (bDrawVolumeVertices /* || bDrawOtherThingCommonToTheIteration */) {
    const TArray<FVector>& Vertices = MochiViewportPtr.Pin()->VolumeNodePositions;
    FMatrix LocalToWorldInverseTransform =
        PreviewMeshComponent->GetComponentTransform().ToMatrixWithScale().InverseFast();

    for (auto i = 0; i < Vertices.Num(); i++) {
      const FVector VertexPos = Vertices[i];
      const FVector WorldPos =
          PreviewMeshComponent->GetComponentTransform().TransformPosition(VertexPos);
      if (bDrawVolumeVertices) {
        PDI->DrawPoint(WorldPos, FLinearColor(0.0f, 1.0f, 0.0f), 5.0f, SDPG_Foreground);
      }
    }
  }

  if (bDrawElementBarycenters || bDrawSelectedElementBarycenters) {
    const TArray<FVector>& Centers = MochiViewportPtr.Pin()->ElementBarycenterPositions;

    for (auto i = 0; i < Centers.Num(); i++) {
      const FVector CenterPos = Centers[i];
      const FVector WorldPos =
          PreviewMeshComponent->GetComponentTransform().TransformPosition(CenterPos);
      // draw selected as larger points first (so it looks like the points are "highlighted")
      if (bDrawSelectedElementBarycenters &&
          SelectedElementsContainer->SelectedElements.Contains(i)) {
        PDI->DrawPoint(
            WorldPos,
            FLinearColor(61.0f / 255.0f, 200.0f / 255.0f, 1.0f),
            kSelectedElementPointSize * (ViewFOV / kDefaultFOV),
            SDPG_Foreground);
        // scaling point size by FOV helps with orthographic view where a 5.0f point
        // is extremely large, whereas it looks reasonable in perspective view
      }

      if (bDrawElementBarycenters) {
        bool bShouldDrawElement = false;
        FLinearColor ElementColor;

        switch (ElementViewMode) {
          case EMochiModelEditorElementViewMode::ModifiedYoungsModulus:
            if (ModifierAsset) {
              if (FMochiMaterialModifier* Modifier = ModifierAsset->MaterialModifiers.Find(i)) {
                ElementColor = FMath::Lerp(
                    kMinElementColor,
                    kMaxElementColor,
                    Modifier->YoungsModulus / kMaxYoungsModulusDisplay);
                bShouldDrawElement = true;
              }
            }
            break;
          case EMochiModelEditorElementViewMode::ModifiedPoissonRatio:
            if (ModifierAsset) {
              if (FMochiMaterialModifier* Modifier = ModifierAsset->MaterialModifiers.Find(i)) {
                ElementColor = FMath::Lerp(
                    kMinElementColor,
                    kMaxElementColor,
                    Modifier->PoissonRatio / kMaxPoissonRatioDisplay);
                bShouldDrawElement = true;
              }
            }
            break;
          case EMochiModelEditorElementViewMode::All:
            ElementColor = FLinearColor::Yellow;
            bShouldDrawElement = true;
            break;
          case EMochiModelEditorElementViewMode::None:
          default:
            break;
        }

        if (bShouldDrawElement) {
          PDI->DrawPoint(
              WorldPos, ElementColor, kElementPointSize * (ViewFOV / kDefaultFOV), SDPG_Foreground);
          // scaling point size by FOV helps with orthographic view where a 5.0f point
          // is extremely large, whereas it looks reasonable in perspective view
        }
      }
    }
  }
}

TSharedPtr<FDragTool> FMochiModelEditorViewportClient::MakeDragTool(EDragTool::Type DragToolType) {
  // Without custom input handling, these drag tools are activated by the level editor's key
  // modifiers.
  // See: FMouseDeltaTracker::ConditionalBeginUsingDragTool
  // FrustumSelect (marquee selection) is CTRL+ALT+LMB, and only works in Perspective
  // Measure only works in isometric views and is just holding MMB
  // BoxSelect only works in isometric and is just holding LMB
  // ViewportChange is not supported.

  TSharedPtr<FDragTool> DragTool;
  switch (DragToolType) {
    case EDragTool::FrustumSelect: {
      // TODO: Clean up references and use TWeakPtr/TSharedPtr instead of raw pointers; this
      // currently follows the engine's drag tool pattern.
      DragTool = MakeShareable(new FMochiModelEditorDragTool(this, MochiViewportPtr.Pin().Get()));

      auto MochiDragTool = StaticCastSharedPtr<FMochiModelEditorDragTool>(DragTool);
      if (MochiDragTool.IsValid()) {
        MochiDragTool.Get()->SetSelectLasso(bUseLassoSelection);
      }
      break;
    }
    default:
      break;
      // case EDragTool::BoxSelect:
      //   DragTool = MakeShareable( new FDragTool_ActorBoxSelect(this) );
      //   break;
      // case EDragTool::Measure:
      //   DragTool = MakeShareable( new FDragTool_Measure(this) );
      //   break;
      // case EDragTool::ViewportChange:
      //   DragTool = MakeShareable( new FDragTool_ViewportChange(this) );
      //   break;
  };

  return DragTool;
}

void FMochiModelEditorViewportClient::AddReferencedObjects(FReferenceCollector& Collector) {
  FEditorViewportClient::AddReferencedObjects(Collector);
  // Keep the selection container from being garbage collected.
  Collector.AddReferencedObject(SelectedElementsContainer);
}

void FMochiModelEditorViewportClient::SetPreviewMesh(
    UMochiModel* InMochiModel,
    UMochiSoftMeshComponent* InPreviewMeshComponent) {
  PreviewModel = InMochiModel;
  PreviewMeshComponent = InPreviewMeshComponent;
}

void FMochiModelEditorViewportClient::SetShowVolumeVertices(bool bShowOn) {
  bDrawVolumeVertices = bShowOn;
  Invalidate();
}

void FMochiModelEditorViewportClient::ToggleShowVolumeVertices() {
  SetShowVolumeVertices(!bDrawVolumeVertices);
}

bool FMochiModelEditorViewportClient::IsShowVolumeVerticesChecked() const {
  return bDrawVolumeVertices;
}

void FMochiModelEditorViewportClient::SetShowElementBarycenters(bool bShowOn) {
  bDrawElementBarycenters = bShowOn;
  Invalidate();
}

void FMochiModelEditorViewportClient::ToggleShowElementBarycenters() {
  SetShowElementBarycenters(!bDrawElementBarycenters);
}

bool FMochiModelEditorViewportClient::IsShowElementBarycentersChecked() const {
  return bDrawElementBarycenters;
}

void FMochiModelEditorViewportClient::SetShowSelectedElementBarycenters(bool bShowOn) {
  bDrawSelectedElementBarycenters = bShowOn;
}

void FMochiModelEditorViewportClient::ToggleShowSelectedElementBarycenters() {
  SetShowSelectedElementBarycenters(!bDrawSelectedElementBarycenters);
}

bool FMochiModelEditorViewportClient::IsShowSelectedElementBarycentersChecked() const {
  return bDrawSelectedElementBarycenters;
}

void FMochiModelEditorViewportClient::SetElementViewMode(EMochiModelEditorElementViewMode NewMode) {
  ElementViewMode = NewMode;
}

EMochiModelEditorElementViewMode FMochiModelEditorViewportClient::GetElementViewMode() const {
  return ElementViewMode;
}

TSet<int>& FMochiModelEditorViewportClient::GetSelectedElements() {
  return SelectedElementsContainer->SelectedElements;
}

void FMochiModelEditorViewportClient::AddSelectionToMaterialModifiers() {
  UMochiModelMaterialModifiers* ModifierAsset = PreviewModel->MaterialModifiersAsset;
  if (!ModifierAsset) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Error,
        TEXT("[%hs] No %s Asset specified!"),
        __FUNCTION__,
        *GET_MEMBER_NAME_CHECKED(UMochiModel, MaterialModifiersAsset).ToString());
    return;
  }

  const FScopedTransaction Transaction(
      LOCTEXT("ApplyModifiers", "Apply Material Modifier Elements"));
  ModifierAsset->Modify();

  TMap<int, FMochiMaterialModifier>& Modifiers = ModifierAsset->MaterialModifiers;
  FMochiMaterialModifier Template = PreviewModel->MaterialModifierTemplate;

  for (auto Element : SelectedElementsContainer->SelectedElements) {
    Modifiers.Emplace(Element, Template);
  }
}

void FMochiModelEditorViewportClient::RemoveSelectionFromMaterialModifiers() {
  UMochiModelMaterialModifiers* ModifierAsset = PreviewModel->MaterialModifiersAsset;
  if (!ModifierAsset) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Error,
        TEXT("[%hs] No %s Asset specified!"),
        __FUNCTION__,
        *GET_MEMBER_NAME_CHECKED(UMochiModel, MaterialModifiersAsset).ToString());
    return;
  }

  const FScopedTransaction Transaction(
      LOCTEXT("RemoveModifiers", "Remove Material Modifier Elements"));
  ModifierAsset->Modify();

  TMap<int, FMochiMaterialModifier>& Modifiers = ModifierAsset->MaterialModifiers;

  for (auto Element : SelectedElementsContainer->SelectedElements) {
    Modifiers.Remove(Element);
  }
}

UMochiModelEditorSelectedElements* FMochiModelEditorViewportClient::GetSelectedElementsContainer()
    const {
  return SelectedElementsContainer;
}

#undef LOCTEXT_NAMESPACE
