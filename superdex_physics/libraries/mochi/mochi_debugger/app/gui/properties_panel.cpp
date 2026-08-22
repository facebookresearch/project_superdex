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

#include "properties_panel.h"

#include "gui/gui.h"
#include "ui_helpers.h"
#include "viewport/render_scene.h"

#include <mochi_core/utils/color.h>
#include <mochi_core/utils/defer.h>

#include <imguios/imguios.h>

using namespace mochi;
using namespace mochi::dbg;

static void AddMeshColorProperty(Color& color) {
  UiColorPicker("Color", &color);
}

static void AddRenderingProperties(UiState& state) {
  constexpr float kSliderWidth = 160.0f;

  auto& rendering = state.rendering;
  UiCheckbox("Show Origin", &rendering.showOriginTriAxis, "Show a tri-axis at the scene origin");
  UiCheckbox("Show Debug Draw", &rendering.showDebugDraw, "Show debug draw lines and spheres");
  UiCheckbox("Show Meshes", &rendering.showMeshes, "Show actor surface meshes");

  if (ImGui::TreeNode("Meshes")) {
    MOCHI_DEFER(ImGui::TreePop());
    auto settings = state.client->GetSettings();
    if (UiCheckbox(
            "Visual Mesh",
            &settings.sync.useVisualMesh,
            "Some actors have a high resolution visual mesh, which is skinned to the surface of "
            "the simulation mesh. If disabled, the simulation mesh will be rendered instead. Use "
            "in combination with \"Show Meshes\".")) {
      state.client->SetSettings(settings);
    }
    UiCheckbox(
        "Flat Shading",
        &rendering.useFlatShading,
        "Flat shading helps you to see individual polygons. Use in combination with \"Show Meshes\".");
  }

  if (ImGui::TreeNode("Lights")) {
    MOCHI_DEFER(ImGui::TreePop());

    if (ImGui::TreeNode("Ambient Light")) {
      MOCHI_DEFER(ImGui::TreePop());
      ImGui::SetNextItemWidth(kSliderWidth);
      ImGui::SliderFloat("Intensity", &rendering.ambientLightIntensity, 0.0f, 1.0f);
    }

    if (ImGui::TreeNode("Directional Light")) {
      MOCHI_DEFER(ImGui::TreePop());
      ImGui::SetNextItemWidth(kSliderWidth);
      ImGui::SliderFloat("Intensity", &rendering.directionalLightIntensity, 0.0f, 1.0f);
      ImGui::SetNextItemWidth(kSliderWidth);
      ImGui::SliderFloat("Pitch", &rendering.directionalLightPitchDeg, -90.0f, 90.0f);
      ImGui::SetNextItemWidth(kSliderWidth);
      ImGui::SliderFloat("Yaw", &rendering.directionalLightYawDeg, -180.0f, 180.0f);
    }
  }

  if (ImGui::TreeNode("Materials")) {
    MOCHI_DEFER(ImGui::TreePop());
    ImGui::SetNextItemWidth(kSliderWidth);
    ImGui::SliderFloat(
        "Roughness",
        &rendering.materialRoughness,
        0.0f,
        1.0f,
        "%.2f",
        ImGuiSliderFlags_AlwaysClamp);
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Lower = shinier/tighter highlight; higher = duller (matte rubber).");
    }
    ImGui::SetNextItemWidth(kSliderWidth);
    ImGui::SliderFloat(
        "Metallic", &rendering.materialMetallic, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Plastic = 0 (dielectric). 1 = metal.");
    }
    ImGui::SetNextItemWidth(kSliderWidth);
    ImGui::SliderFloat(
        "Reflectance",
        &rendering.materialReflectance,
        0.0f,
        1.0f,
        "%.2f",
        ImGuiSliderFlags_AlwaysClamp);
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip(
          "Dielectric specular strength (0.5 ~ typical plastic). Only affects metallic=0.");
    }
    if (ImGui::TreeNode("Dynamic Meshes")) {
      MOCHI_DEFER(ImGui::TreePop());
      UiCheckbox("Use Shared Material", &rendering.dynamicMeshesShareMaterial);
      AddMeshColorProperty(rendering.dynamicMeshColor);
    }
    if (ImGui::TreeNode("Static Meshes")) {
      MOCHI_DEFER(ImGui::TreePop());
      UiCheckbox("Use Shared Material", &rendering.staticMeshesShareMaterial);
      AddMeshColorProperty(rendering.staticMeshColor);
    }
  }
}

void dbg::BuildPropertiesPanel(UiState& state) {
  ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
  if (ImGui::TreeNode("Rendering")) {
    MOCHI_DEFER(ImGui::TreePop());
    AddRenderingProperties(state);
  }
}
