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

#include "MochiModelEditorCommands.h"
#include "InputCoreTypes.h"

// TODO: Check whether these LOCTEXT entries are needed.
#define LOCTEXT_NAMESPACE "FMochiModelEditorCommands"

void FMochiModelEditorCommands::RegisterCommands() {
  UI_COMMAND(
      FocusViewport,
      "Focus Viewport",
      "Focus Viewport on Mochi Model",
      EUserInterfaceActionType::Button,
      FInputChord(EKeys::F));
  UI_COMMAND(
      SetOrthographicView,
      "Faux Orthographic",
      "Set camera FOV to emulate orthographic projection",
      EUserInterfaceActionType::Button,
      FInputChord());
  UI_COMMAND(
      SetPerspectiveView,
      "Perspective",
      "Set camera FOV to default perspective view",
      EUserInterfaceActionType::Button,
      FInputChord());
  UI_COMMAND(
      Remesh,
      "Remesh",
      "Remesh using last used Static Mesh",
      EUserInterfaceActionType::Button,
      FInputChord());
  UI_COMMAND(
      RemeshWithNewMesh,
      "Remesh With New Static Mesh",
      "Remesh using another Static Mesh",
      EUserInterfaceActionType::Button,
      FInputChord());
  UI_COMMAND(
      SetShowVolumeVertices,
      "Volume Vertices",
      "Show volume vertices as green points",
      EUserInterfaceActionType::ToggleButton,
      FInputChord());
  UI_COMMAND(
      SetShowElementBarycenters,
      "Element Barycenters",
      "Show element barycenters as red points",
      EUserInterfaceActionType::ToggleButton,
      FInputChord());
  UI_COMMAND(
      SetShowSelectedElementBarycenters,
      "Selected Element Barycenters",
      "Show selected element barycenters as blue points",
      EUserInterfaceActionType::ToggleButton,
      FInputChord());
  UI_COMMAND(
      ApplyMaterialModifer,
      "Apply Material Modifiers",
      "Adds selected elements to the Material Modifiers Map",
      EUserInterfaceActionType::Button,
      FInputChord());
  UI_COMMAND(
      RemoveMaterialModifier,
      "Remove Material Modifiers",
      "Removes selected elements the Material Modifiers Map",
      EUserInterfaceActionType::Button,
      FInputChord());
  UI_COMMAND(
      LassoSelectionTool,
      "Lasso Selection",
      "Use Lasso Selection for selecting elements",
      EUserInterfaceActionType::RadioButton,
      FInputChord());
  UI_COMMAND(
      BoxSelectionTool,
      "Box Selection",
      "Use Box Selection for selecting elements",
      EUserInterfaceActionType::RadioButton,
      FInputChord());
  UI_COMMAND(
      OneShotSelectionMode,
      "One Shot Selection",
      "Always clear previously selected elements before selecting new elements.",
      EUserInterfaceActionType::RadioButton,
      FInputChord(EKeys::Zero));
  UI_COMMAND(
      AdditiveSelectionMode,
      "Additive Selection",
      "Add newly selected elements to current selection",
      EUserInterfaceActionType::RadioButton,
      FInputChord(EKeys::Add));
  UI_COMMAND(
      SubtractiveSelectionMode,
      "Subtractive Selection",
      "Subtract newly selected elements from current selection",
      EUserInterfaceActionType::RadioButton,
      FInputChord(EKeys::Subtract));
}

#undef LOCTEXT_NAMESPACE
