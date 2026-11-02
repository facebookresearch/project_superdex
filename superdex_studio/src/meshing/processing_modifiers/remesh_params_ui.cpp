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

#include "meshing/processing_modifiers/remesh_params_ui.h"

#include <imgui.h>

#include <algorithm>
#include <string>
#include <vector>

namespace superdex::studio::processing {

void ShowReferenceCombo(char const* label, int& referenceIndex, ModifierGuiContext const& gui) {
  std::vector<std::string> items;
  std::vector<int> itemRefIndex;
  items.emplace_back("Preceding Source");
  itemRefIndex.push_back(MeshProcessingMethod::kReferencePrecedingSource);
  if (gui.modifierNames != nullptr) {
    for (std::size_t j = 0; j < gui.selfIndex && j < gui.modifierNames->size(); ++j) {
      items.push_back("[" + std::to_string(j) + "] " + (*gui.modifierNames)[j]);
      itemRefIndex.push_back(static_cast<int>(j));
    }
  }
  int current = -1;
  for (std::size_t k = 0; k < itemRefIndex.size(); ++k) {
    if (itemRefIndex[k] == referenceIndex) {
      current = static_cast<int>(k);
      break;
    }
  }
  char const* const preview =
      current < 0 ? "(invalid source)" : items[static_cast<std::size_t>(current)].c_str();
  if (ImGui::BeginCombo(label, preview)) {
    for (std::size_t k = 0; k < items.size(); ++k) {
      bool const selected = static_cast<int>(k) == current;
      if (ImGui::Selectable(items[k].c_str(), selected)) {
        referenceIndex = itemRefIndex[k];
      }
      if (selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
}

void ShowLengthControl(
    char const* name,
    char const* description,
    double& length,
    bool relativeToMeshSize,
    ModifierTooltip const& tooltip) {
  double const displayScale = relativeToMeshSize ? 100.0 : 1000.0;
  double value = length * displayScale;
  std::string const label = std::string(name) + (relativeToMeshSize ? " (%)" : " (mm)");
  if (ImGui::InputDouble(label.c_str(), &value, 0.0, 0.0, "%.4f")) {
    length = std::max(value, 0.0) / displayScale;
  }
  tooltip((std::string(description) +
           (relativeToMeshSize ? " as a percent of the mesh bounding-box average dimension."
                               : " in millimeters."))
              .c_str());
}

void ShowEdgeSizeControl(
    double& edgeSize,
    bool& relativeToMeshSize,
    ModifierTooltip const& tooltip) {
  ShowLengthControl("Edge Size", "Target edge length", edgeSize, relativeToMeshSize, tooltip);
  ImGui::Checkbox("Relative to Mesh Size", &relativeToMeshSize);
  tooltip("Interpret lengths as fractions of the bounding box (on) or absolute mm (off).");
}

void ShowFeatureControls(
    bool& detectFeatures,
    double& sharpFeatureAngle,
    ModifierTooltip const& tooltip) {
  ImGui::Checkbox("Detect Features", &detectFeatures);
  tooltip("Detect sharp feature edges (using the angle below) and preserve them.");
  ImGui::BeginDisabled(!detectFeatures);
  ImGui::InputDouble("Sharp Feature Angle (deg)", &sharpFeatureAngle, 0.0, 0.0, "%.3f");
  tooltip("Angle between adjacent face normals above which an edge counts as a sharp feature.");
  ImGui::EndDisabled();
}

} // namespace superdex::studio::processing
