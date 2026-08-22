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

#include <imguios/imguios.h>

#include <algorithm>

namespace superdex::studio::processing {

void ShowEdgeSizeControl(
    double& edgeSize,
    bool& relativeToMeshSize,
    ModifierTooltip const& tooltip) {
  if (relativeToMeshSize) {
    double percent = edgeSize * 100.0;
    if (ImGui::InputDouble("Edge Size (%)", &percent, 0.0, 0.0, "%.4f")) {
      edgeSize = std::max(percent, 0.0) / 100.0;
    }
    tooltip("Target triangle edge length as a percent of the mesh bounding-box average dimension.");
  } else {
    double mm = edgeSize * 1000.0;
    if (ImGui::InputDouble("Edge Size (mm)", &mm, 0.0, 0.0, "%.4f")) {
      edgeSize = std::max(mm, 0.0) / 1000.0;
    }
    tooltip("Target triangle edge length in millimeters.");
  }
  ImGui::Checkbox("Relative to Mesh Size", &relativeToMeshSize);
  tooltip("Interpret edge size as a fraction of the bounding box (on) or absolute mm (off).");
}

void ShowFeatureControls(
    bool& detectFeatures,
    double& sharpFeatureAngle,
    ModifierTooltip const& tooltip) {
  ImGui::Checkbox("Detect Features", &detectFeatures);
  tooltip("Detect sharp feature edges (using the angle below) and preserve them while remeshing.");
  ImGui::BeginDisabled(!detectFeatures);
  ImGui::InputDouble("Sharp Feature Angle (deg)", &sharpFeatureAngle, 0.0, 0.0, "%.3f");
  tooltip("Dihedral angle threshold above which an edge counts as a sharp feature to preserve.");
  ImGui::EndDisabled();
}

} // namespace superdex::studio::processing
