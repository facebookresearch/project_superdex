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

#include "SMochiModelEditorViewport.h"

#include "MochiDynamicMeshUtil.h"
#include "MochiModelEditorCommands.h"
#include "MochiModelEditorViewportClient.h"
#include "MochiModelPreviewScene.h"
#include "MochiPhysicsEditorModule.h"
#include "MochiViewportFraming.h"
#include "SMochiModelEditorViewportToolbar.h"

void SMochiModelEditorViewport::Construct(
    const FArguments& InArgs,
    TSharedPtr<FMochiModelEditorToolkit> InEditorToolkit,
    TSharedPtr<FMochiModelPreviewScene> InPreviewScene) {
  EditorToolkit = InEditorToolkit;
  ModelPreviewScene = InPreviewScene;
  // TODO: should we pass InArgs instead? why default struct?

  PreviewMeshComponent =
      NewObject<UMochiSoftMeshComponent>(GetTransientPackage(), NAME_None, RF_Transient);
  PreviewMeshComponent->bUseMochiNodeNormals = false;
  PreviewMeshComponent->bEnableVolumeNodePositionsQuery = true;

  const TObjectPtr<UMochiModel> Model = EditorToolkit->GetModel();

  PreviewMeshComponent->SetMochiModel(Model);
  if (Model->PreviewRenderMaterial) {
    PreviewMeshComponent->SetMaterial(0, Model->PreviewRenderMaterial);
  }

  const FVector Scale = PreviewMeshComponent->GetComponentTransform().GetScale3D();

  ConvertElementBarycenterReferencePositions(Model, Scale, ElementBarycenterPositions);
  ConvertVolumeNodeReferencePositions(Model, Scale, VolumeNodePositions);

  // Putting model in the middle since the PreviewScene's floor is disabled.
  // If we were to add the floor --the floor has to be moved down.
  ModelPreviewScene->AddComponent(PreviewMeshComponent, FTransform::Identity);

  SEditorViewport::Construct(SEditorViewport::FArguments());
}

SMochiModelEditorViewport::~SMochiModelEditorViewport() {
  if (ViewportClient.IsValid()) {
    ViewportClient->Viewport = nullptr;
  }
}

void SMochiModelEditorViewport::AddReferencedObjects(FReferenceCollector& Collector) {
  Collector.AddReferencedObject(PreviewMeshComponent);
}

void SMochiModelEditorViewport::BindCommands() {
  SAssetEditorViewport::BindCommands();

  const FMochiModelEditorCommands& Commands = FMochiModelEditorCommands::Get();

  const TSharedRef<FMochiModelEditorViewportClient> EditorViewportClient =
      ViewportClient.ToSharedRef();

  CommandList->MapAction(
      Commands.SetOrthographicView,
      FExecuteAction::CreateSP(this, &SMochiModelEditorViewport::SetUseOrthographicView, true),
      FCanExecuteAction(),
      FIsActionChecked::CreateSP(this, &SMochiModelEditorViewport::IsUsingOrthographicView));

  CommandList->MapAction(
      Commands.SetPerspectiveView,
      FExecuteAction::CreateSP(this, &SMochiModelEditorViewport::SetUseOrthographicView, false),
      FCanExecuteAction(),
      FIsActionChecked::CreateSP(this, &SMochiModelEditorViewport::IsUsingPerspectiveView));

  CommandList->MapAction(
      Commands.SetShowVolumeVertices,
      FExecuteAction::CreateSP(
          EditorViewportClient, &FMochiModelEditorViewportClient::ToggleShowVolumeVertices),
      FCanExecuteAction(),
      FIsActionChecked::CreateSP(
          EditorViewportClient, &FMochiModelEditorViewportClient::IsShowVolumeVerticesChecked));

  CommandList->MapAction(
      Commands.SetShowElementBarycenters,
      FExecuteAction::CreateSP(
          EditorViewportClient, &FMochiModelEditorViewportClient::ToggleShowElementBarycenters),
      FCanExecuteAction(),
      FIsActionChecked::CreateSP(
          EditorViewportClient, &FMochiModelEditorViewportClient::IsShowElementBarycentersChecked));

  CommandList->MapAction(
      Commands.SetShowSelectedElementBarycenters,
      FExecuteAction::CreateSP(
          EditorViewportClient,
          &FMochiModelEditorViewportClient::ToggleShowSelectedElementBarycenters),
      FCanExecuteAction(),
      FIsActionChecked::CreateSP(
          EditorViewportClient,
          &FMochiModelEditorViewportClient::IsShowSelectedElementBarycentersChecked));

  CommandList->MapAction(
      Commands.LassoSelectionTool,
      FExecuteAction::CreateSP(this, &SMochiModelEditorViewport::SetSelectLasso, true),
      FCanExecuteAction(),
      FIsActionChecked::CreateSP(this, &SMochiModelEditorViewport::IsUsingLassoSelection));

  CommandList->MapAction(
      Commands.BoxSelectionTool,
      FExecuteAction::CreateSP(this, &SMochiModelEditorViewport::SetSelectLasso, false),
      FCanExecuteAction(),
      FIsActionChecked::CreateSP(this, &SMochiModelEditorViewport::IsUsingBoxSelection));

  CommandList->MapAction(
      Commands.OneShotSelectionMode,
      FExecuteAction::CreateSP(
          this,
          &SMochiModelEditorViewport::SetSelectionMode,
          EMochiModelEditorSelectionMode::ClearFirst),
      FCanExecuteAction(),
      FIsActionChecked::CreateSP(this, &SMochiModelEditorViewport::IsUsingOneShotSelection));

  CommandList->MapAction(
      Commands.AdditiveSelectionMode,
      FExecuteAction::CreateSP(
          this,
          &SMochiModelEditorViewport::SetSelectionMode,
          EMochiModelEditorSelectionMode::Additive),
      FCanExecuteAction(),
      FIsActionChecked::CreateSP(this, &SMochiModelEditorViewport::IsUsingAdditiveSelection));

  CommandList->MapAction(
      Commands.SubtractiveSelectionMode,
      FExecuteAction::CreateSP(
          this,
          &SMochiModelEditorViewport::SetSelectionMode,
          EMochiModelEditorSelectionMode::Subtractive),
      FCanExecuteAction(),
      FIsActionChecked::CreateSP(this, &SMochiModelEditorViewport::IsUsingSubtractiveSelection));
}

TSharedRef<SEditorViewport> SMochiModelEditorViewport::GetViewportWidget() {
  return SharedThis(this);
}

TSharedPtr<FExtender> SMochiModelEditorViewport::GetExtenders() const {
  // TODO: implement more tools beyond the "Show" button

  TSharedPtr<FExtender> Result(MakeShareable(new FExtender));
  return Result;
}

void SMochiModelEditorViewport::OnFocusViewportToSelection() {
  SAssetEditorViewport::OnFocusViewportToSelection();

  if (ModelPreviewScene.IsValid() && PreviewMeshComponent) {
    MochiViewportFraming::FocusOnBounds(*ViewportClient, PreviewMeshComponent->Bounds, false);
  }
}

TSharedRef<FEditorViewportClient> SMochiModelEditorViewport::MakeEditorViewportClient() {
  UMochiModel* Model = EditorToolkit->GetModel();

  ViewportClient = MakeShareable(new FMochiModelEditorViewportClient(
      SharedThis(this), ModelPreviewScene.ToSharedRef(), Model, PreviewMeshComponent));
  return ViewportClient.ToSharedRef();
}

TSharedRef<SWidget> SMochiModelEditorViewport::MakeLegacyViewportToolbar() {
  return SNew(SMochiModelEditorViewportToolbar, SharedThis(this));
}

#if UE_VERSION_OLDER_THAN(5, 7, 0)
TSharedPtr<SWidget> SMochiModelEditorViewport::MakeViewportToolbar() {
  return MakeLegacyViewportToolbar();
}
#endif

void SMochiModelEditorViewport::PopulateViewportOverlays(TSharedRef<SOverlay> Overlay) {
  SEditorViewport::PopulateViewportOverlays(Overlay);
#if !UE_VERSION_OLDER_THAN(5, 7, 0)
  // Placed here by SEditorViewport itself before UE 5.7.
  Overlay->AddSlot().VAlign(VAlign_Top)[MakeLegacyViewportToolbar()];
#endif

  Overlay->AddSlot()
      .VAlign(VAlign_Top)
      .HAlign(HAlign_Left)
      .Padding(FMargin(
          6.0f, 36.0f, 6.0f, 6.0f))[SNew(SBorder)
                                        .BorderImage(FAppStyle::Get().GetBrush("FloatingBorder"))
                                        .Padding(4.f)[SAssignNew(OverlayText, SRichTextBlock)]];
}

void SMochiModelEditorViewport::PopulateOverlayText(const TArray<FText>& TextItems) {
  FTextBuilder FinalText;
  for (FText const& item : TextItems) {
    FinalText.AppendLine(item);
  }
  OverlayText->SetText(FinalText.ToText());
}

EMochiModelEditorSelectionMode SMochiModelEditorViewport::GetSelectionMode() const {
  return SelectionMode;
}

void SMochiModelEditorViewport::SetSelectionMode(EMochiModelEditorSelectionMode NewSelectionMode) {
  SelectionMode = NewSelectionMode;
}

bool SMochiModelEditorViewport::IsUsingOneShotSelection() const {
  return SelectionMode == EMochiModelEditorSelectionMode::ClearFirst;
}

bool SMochiModelEditorViewport::IsUsingAdditiveSelection() const {
  return SelectionMode == EMochiModelEditorSelectionMode::Additive;
}

bool SMochiModelEditorViewport::IsUsingSubtractiveSelection() const {
  return SelectionMode == EMochiModelEditorSelectionMode::Subtractive;
}

void SMochiModelEditorViewport::SetUseOrthographicView(bool bOrthographic) {
  bUseOrthographic = bOrthographic;

  if (!ViewportClient.IsValid()) {
    return;
  }

  if (bOrthographic) {
    ViewportClient->ViewFOV = kOrthographicFOV;
    ModelPreviewScene->SetEnvironmentVisibility(false);
  } else {
    ViewportClient->ViewFOV = kPerspectiveFOV;
    ModelPreviewScene->SetEnvironmentVisibility(true);
  }

  if (ModelPreviewScene.IsValid() && PreviewMeshComponent) {
    MochiViewportFraming::FocusOnBounds(*ViewportClient, PreviewMeshComponent->Bounds, true);
  }
}

bool SMochiModelEditorViewport::IsUsingOrthographicView() {
  return bUseOrthographic;
}

bool SMochiModelEditorViewport::IsUsingPerspectiveView() {
  return !bUseOrthographic;
}

bool SMochiModelEditorViewport::IsUsingLassoSelection() const {
  return ViewportClient->IsUsingLassoSelection();
}

bool SMochiModelEditorViewport::IsUsingBoxSelection() const {
  return !ViewportClient->IsUsingLassoSelection();
}

void SMochiModelEditorViewport::SetSelectLasso(bool bUseLasso) {
  ViewportClient->SetUseLassoSelection(bUseLasso);
}
