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

#include "SMochiModelEditorViewportToolbar.h"

#include "PreviewProfileController.h"

#include "MochiModelEditorCommands.h"
#include "SEditorViewportToolBarMenu.h"
#include "SMochiModelEditorViewport.h"
#include "Widgets/Input/SSegmentedControl.h"

#define LOCTEXT_NAMESPACE "SMochiModelEditorViewportToolbar"

void SMochiModelEditorViewportToolbar::Construct(
    const FArguments& InArgs,
    TSharedPtr<ICommonEditorViewportToolbarInfoProvider> InInfoProvider) {
  SCommonEditorViewportToolbarBase::Construct(
      SCommonEditorViewportToolbarBase::FArguments().PreviewProfileController(
          MakeShared<FPreviewProfileController>()),
      InInfoProvider);
}

TSharedRef<SWidget> SMochiModelEditorViewportToolbar::GenerateShowMenu() const {
  GetInfoProvider().OnFloatingButtonClicked();

  TSharedRef<SEditorViewport> ViewportRef = GetInfoProvider().GetViewportWidget();

  const bool bShouldCloseAfterSelection = false;
  FMenuBuilder ShowMenuBuilder(bShouldCloseAfterSelection, ViewportRef->GetCommandList());
  {
    auto Commands = FMochiModelEditorCommands::Get();

    ShowMenuBuilder.BeginSection("MochiModel", LOCTEXT("MochiModel", "Mochi Model"));
    ShowMenuBuilder.AddMenuEntry(Commands.SetShowVolumeVertices);
    ShowMenuBuilder.AddMenuEntry(Commands.SetShowElementBarycenters);
    ShowMenuBuilder.AddMenuEntry(Commands.SetShowSelectedElementBarycenters);
  }

  return ShowMenuBuilder.MakeWidget();
}

void SMochiModelEditorViewportToolbar::ExtendLeftAlignedToolbarSlots(
    TSharedPtr<SHorizontalBox> MainBoxPtr,
    TSharedPtr<SViewportToolBar> ParentToolBarPtr) const {
  const FMargin ToolbarSlotPadding(2.0f, 2.0f);

  if (!MainBoxPtr.IsValid()) {
    return;
  }

  MainBoxPtr->AddSlot().AutoWidth().Padding(ToolbarSlotPadding)
      [SNew(SEditorViewportToolbarMenu)
           .Label(this, &SMochiModelEditorViewportToolbar::GetSelectionToolLabel)
           .OnGetMenuContent(this, &SMochiModelEditorViewportToolbar::GenerateSelectionToolMenu)
           .Cursor(EMouseCursor::Default)
           .ParentToolBar(ParentToolBarPtr)];
  MainBoxPtr->AddSlot().AutoWidth().Padding(ToolbarSlotPadding)
      [SNew(SEditorViewportToolbarMenu)
           .Label(this, &SMochiModelEditorViewportToolbar::GetCameraProjectionLabel)
           .OnGetMenuContent(this, &SMochiModelEditorViewportToolbar::GenerateCameraProjectionMenu)
           .Cursor(EMouseCursor::Default)
           .ParentToolBar(ParentToolBarPtr)];
}

TSharedPtr<FExtender> SMochiModelEditorViewportToolbar::GetViewMenuExtender() const {
  return SCommonEditorViewportToolbarBase::GetViewMenuExtender();
}

TSharedRef<SWidget> SMochiModelEditorViewportToolbar::GenerateSelectionToolMenu() const {
  const FMochiModelEditorCommands& Commands = FMochiModelEditorCommands::Get();

  TSharedRef<SEditorViewport> BaseViewportRef = GetInfoProvider().GetViewportWidget();
  TSharedRef<SMochiModelEditorViewport> ViewportRef =
      StaticCastSharedRef<SMochiModelEditorViewport, SEditorViewport>(BaseViewportRef);

  TSharedPtr<FExtender> MenuExtender = GetInfoProvider().GetExtenders();

  const bool bShouldCloseAfterSelect = true;
  FMenuBuilder InMenuBuilder(bShouldCloseAfterSelect, ViewportRef->GetCommandList(), MenuExtender);

  InMenuBuilder.PushCommandList(ViewportRef->GetCommandList().ToSharedRef());
  if (MenuExtender.IsValid()) {
    InMenuBuilder.PushExtender(MenuExtender.ToSharedRef());
  }

  InMenuBuilder.BeginSection(
      "MochiModelEditorViewportSelectionTool",
      LOCTEXT("SelectionMode_PreviewLabel", "Selection Tool"));
  InMenuBuilder.AddMenuEntry(Commands.LassoSelectionTool);
  InMenuBuilder.AddMenuEntry(Commands.BoxSelectionTool);
  InMenuBuilder.EndSection();

  InMenuBuilder.PopCommandList();
  if (MenuExtender.IsValid()) {
    InMenuBuilder.PopExtender();
  }

  return InMenuBuilder.MakeWidget();
}

TSharedRef<SWidget> SMochiModelEditorViewportToolbar::GenerateCameraProjectionMenu() const {
  const FMochiModelEditorCommands& Commands = FMochiModelEditorCommands::Get();

  TSharedRef<SEditorViewport> BaseViewportRef = GetInfoProvider().GetViewportWidget();
  TSharedRef<SMochiModelEditorViewport> ViewportRef =
      StaticCastSharedRef<SMochiModelEditorViewport, SEditorViewport>(BaseViewportRef);

  TSharedPtr<FExtender> MenuExtender = GetInfoProvider().GetExtenders();

  const bool bShouldCloseAfterSelect = true;
  FMenuBuilder InMenuBuilder(bShouldCloseAfterSelect, ViewportRef->GetCommandList(), MenuExtender);

  InMenuBuilder.PushCommandList(ViewportRef->GetCommandList().ToSharedRef());
  if (MenuExtender.IsValid()) {
    InMenuBuilder.PushExtender(MenuExtender.ToSharedRef());
  }

  InMenuBuilder.BeginSection(
      "MochiModelEditorViewportCameraProjection",
      LOCTEXT("ProjectionMode_PreviewLabel", "Camera Projection"));
  InMenuBuilder.AddMenuEntry(Commands.SetOrthographicView);
  InMenuBuilder.AddMenuEntry(Commands.SetPerspectiveView);
  InMenuBuilder.EndSection();

  InMenuBuilder.PopCommandList();
  if (MenuExtender.IsValid()) {
    InMenuBuilder.PopExtender();
  }

  return InMenuBuilder.MakeWidget();
}

FText SMochiModelEditorViewportToolbar::GetSelectionToolLabel() const {
  FText Label = LOCTEXT("SelectionMode_LassoLabel", "Lasso Selection");

  TSharedRef<SEditorViewport> BaseViewportRef = GetInfoProvider().GetViewportWidget();
  TSharedRef<SMochiModelEditorViewport> ViewportRef =
      StaticCastSharedRef<SMochiModelEditorViewport, SEditorViewport>(BaseViewportRef);
  if (!ViewportRef->IsUsingLassoSelection()) {
    Label = LOCTEXT("SelectionMode_BoxLabel", "Box Selection");
  }

  return Label;
}

FText SMochiModelEditorViewportToolbar::GetCameraProjectionLabel() const {
  FText Label = LOCTEXT("CameraProjection_Perspective", "Perspective View");

  TSharedRef<SEditorViewport> BaseViewportRef = GetInfoProvider().GetViewportWidget();
  TSharedRef<SMochiModelEditorViewport> ViewportRef =
      StaticCastSharedRef<SMochiModelEditorViewport, SEditorViewport>(BaseViewportRef);
  if (ViewportRef->IsUsingOrthographicView()) {
    Label = LOCTEXT("CameraProjection_Orthographic", "Faux Orthographic View");
  }

  return Label;
}

#undef LOCTEXT_NAMESPACE
