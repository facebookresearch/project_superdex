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

#include "Framework/Commands/Commands.h"
#include "Styling/AppStyle.h"
#include "Templates/SharedPointer.h"

class FMochiModelEditorCommands : public TCommands<FMochiModelEditorCommands> {
 public:
  FMochiModelEditorCommands()
      : TCommands<FMochiModelEditorCommands>(
            "MochiModelEditor",
            NSLOCTEXT("Contexts", "MochiModelEditor", "Mochi Model Editor"),
            "EditorViewport",
            FAppStyle::GetAppStyleSetName()) {}

  // Viewport manipulation
  TSharedPtr<FUICommandInfo> FocusViewport;
  TSharedPtr<FUICommandInfo> SetOrthographicView;
  TSharedPtr<FUICommandInfo> SetPerspectiveView;

  // Meshing commands
  TSharedPtr<FUICommandInfo> Remesh;
  TSharedPtr<FUICommandInfo> RemeshWithNewMesh;

  // Toolbar commands ("Show" menu)
  TSharedPtr<FUICommandInfo> SetShowVolumeVertices;
  TSharedPtr<FUICommandInfo> SetShowElementBarycenters;
  TSharedPtr<FUICommandInfo> SetShowSelectedElementBarycenters;

  // Material Modifier commands
  TSharedPtr<FUICommandInfo> ApplyMaterialModifer;
  TSharedPtr<FUICommandInfo> RemoveMaterialModifier;

  // Selection tool commands
  TSharedPtr<FUICommandInfo> LassoSelectionTool;
  TSharedPtr<FUICommandInfo> BoxSelectionTool;
  TSharedPtr<FUICommandInfo> OneShotSelectionMode;
  TSharedPtr<FUICommandInfo> AdditiveSelectionMode;
  TSharedPtr<FUICommandInfo> SubtractiveSelectionMode;

  virtual void RegisterCommands() override;
};
