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
#include <mochi_debugger/lib/debug_client.h>

#include <imguios/imguios.h>

#include <limits>

using namespace mochi;
using namespace mochi::dbg;

static void AddMeshColorProperty(Color& color) {
  UiColorPicker("Color", &color);
}

static void AddRenderingProperties(UiState& state) {
  constexpr float kSliderWidth = 160.0f;
  auto& rendering = state.rendering;
  bool enableDebugDraw = state.client->IsDebugDrawEnabled();

  UiCheckbox("Show Origin", &rendering.showOriginTriAxis, "Show a tri-axis at the scene origin");
  if (UiCheckbox(
          "Show Debug Draw",
          &enableDebugDraw,
          "Render the selected debug draw features using data from the simulation")) {
    state.client->EnableDebugDraw(enableDebugDraw);
  }
  rendering.showDebugDraw = enableDebugDraw; // Kept in sync

  UiCheckbox("Show Meshes", &rendering.showMeshes, "Show actor surface meshes");

  if (ImGui::TreeNode("Debug Draw")) {
    MOCHI_DEFER(ImGui::TreePop());

    auto const features = state.client->GetDebugDrawFeatures();
    for (int i = 0; i < isize(features); ++i) {
      bool featureEnabled = features[i].enabled;
      if (UiCheckbox(features[i].name.c_str(), &featureEnabled, features[i].description.c_str())) {
        state.client->EnableDebugDrawFeature(features[i].name, featureEnabled);
      }
    }
  }

  if (ImGui::TreeNode("Meshes")) {
    MOCHI_DEFER(ImGui::TreePop());
    auto settings = state.client->GetSettings();
    if (UiCheckbox(
            "Visual Mesh",
            &settings.sync.useVisualMesh,
            "Some actors have a high resolution visual mesh, which is skinned to the surface of "
            "the simulation mesh. If disabled, the simulation mesh or contact skin will be "
            "rendered instead. Use in combination with \"Show Meshes\".")) {
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

static void AddIslandProperties(UiState& state) {
  constexpr float kSliderWidth = 160.0f;
  auto& islands = state.islands;
  auto const current = state.client->GetSleepParams();
  if (!current.has_value()) {
    islands.editing = false; // Do not carry an edit across a disconnect or scene change
  }
  if (!islands.editing) {
    islands.edit = current.value_or(experimental::SleepParams{});
  }

  ImGui::BeginDisabled(!current.has_value());
  MOCHI_DEFER(ImGui::EndDisabled());

  // Send the value once the user releases the last widget.
  auto const sendOnRelease = [&](char const* tooltip) {
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("%s", tooltip);
    }
    if (ImGui::IsItemActivated()) {
      islands.editing = true;
    }
    if (ImGui::IsItemDeactivatedAfterEdit()) {
      state.client->SetSleepParams(islands.edit);
    }
    if (ImGui::IsItemDeactivated()) {
      islands.editing = false;
    }
  };

  if (UiCheckbox("Can Sleep", &islands.edit.canSleep, "Whether islands are allowed to sleep.")) {
    state.client->SetSleepParams(islands.edit);
  }

  // The server rejects 0, and rejected values are not reverted on the client, so keep them valid.
  constexpr float kMinSleepThreshold = 0.001f;
  auto sleepThreshold = static_cast<float>(islands.edit.sleepThreshold);
  ImGui::SetNextItemWidth(kSliderWidth);
  if (ImGui::SliderFloat(
          "Sleep Threshold",
          &sleepThreshold,
          kMinSleepThreshold,
          1.0f,
          "%.3f",
          ImGuiSliderFlags_AlwaysClamp)) {
    islands.edit.sleepThreshold = static_cast<double>(sleepThreshold);
  }
  sendOnRelease(
      "How strictly a step is judged to be at rest. Higher values make islands sleep later and "
      "less often. Lower values save more computation, but may put slowly moving actors to sleep "
      "before they settle. Must be in (0, 1].");

  ImGui::SetNextItemWidth(kSliderWidth);
  ImGui::DragInt(
      "Min Steps Before Sleep",
      &islands.edit.minStepsBeforeSleep,
      1.0f,
      2,
      std::numeric_limits<int>::max(),
      "%d",
      ImGuiSliderFlags_AlwaysClamp);
  sendOnRelease("Number of consecutive rest steps required before an island goes to sleep.");
}

static void AddPerformanceProperties(UiState& state) {
  auto settings = state.client->GetSettings();
  bool changed = UiCheckbox(
      "Force Single Threaded",
      &settings.scene.forceSingleThreaded,
      "Step the selected scene on a single thread. Released when the scene is deselected or the "
      "debugger disconnects.");
  changed |= UiCheckbox(
      "Force Single Island",
      &settings.scene.forceSingleIsland,
      "Merge all islands of the selected scene into one, in addition to the application's own "
      "setting. Released when the scene is deselected or the debugger disconnects.");
  if (changed) {
    state.client->SetSettings(settings);
  }
}

void dbg::BuildPropertiesPanel(UiState& state) {
  if (ImGui::TreeNode("Islands")) {
    MOCHI_DEFER(ImGui::TreePop());
    AddIslandProperties(state);
  }

  if (ImGui::TreeNode("Performance")) {
    MOCHI_DEFER(ImGui::TreePop());
    AddPerformanceProperties(state);
  }

  ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
  if (ImGui::TreeNode("Rendering")) {
    MOCHI_DEFER(ImGui::TreePop());
    AddRenderingProperties(state);
  }
}
