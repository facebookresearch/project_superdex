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
#include "SCommonEditorViewportToolbarBase.h"
#include "SEditorViewport.h"
#include "Widgets/SWidget.h"

/**
 *
 */
class SMochiModelEditorViewportToolbar : public SCommonEditorViewportToolbarBase {
 public:
  SLATE_BEGIN_ARGS(SMochiModelEditorViewportToolbar) {}
  SLATE_END_ARGS()

  // Constructs the toolbar
  void Construct(
      const FArguments& InArgs,
      TSharedPtr<class ICommonEditorViewportToolbarInfoProvider> InInfoProvider);

 private:
  // Populates items in the Show menu, otherwise blank
  virtual TSharedRef<SWidget> GenerateShowMenu() const override;

  // Adds more buttons to the toolbar (beyond the Show menu button)
  virtual void ExtendLeftAlignedToolbarSlots(
      TSharedPtr<SHorizontalBox> MainBoxPtr,
      TSharedPtr<SViewportToolBar> ParentToolBarPtr) const override;

  // Hopefully we can "inject" more view modes using this
  virtual TSharedPtr<FExtender> GetViewMenuExtender() const override;

  // Lasso vs Box selection
  TSharedRef<SWidget> GenerateSelectionToolMenu() const;

  // For quickly changing FOV to emulate perspective vs orthographic
  TSharedRef<SWidget> GenerateCameraProjectionMenu() const;

  // Replaces menu button with "Lasso" or "Box" depending on tool
  FText GetSelectionToolLabel() const;

  // Replaces menu button with "Perspective" or "Faux Orthographic"
  FText GetCameraProjectionLabel() const;
};
