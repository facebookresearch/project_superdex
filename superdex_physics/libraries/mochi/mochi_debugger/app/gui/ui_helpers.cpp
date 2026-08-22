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

#include "ui_helpers.h"

#include <imguios/imguios.h>
#include <mochi_core/utils/defer.h>

#include <cstdint>

namespace mochi::dbg {

bool UiButton(char const* label, char const* tooltip, bool enabled) {
  if (!enabled) {
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
  }
  bool clicked = ImGui::Button(label);
  if (ImGui::IsItemHovered() && tooltip && *tooltip) {
    ImGui::SetTooltip("%s", tooltip);
  }
  if (!enabled) {
    ImGui::PopStyleVar();
    ImGui::PopItemFlag();
  }
  return clicked && enabled;
}

bool UiCheckbox(char const* label, bool* value, char const* tooltip, bool enabled) {
  if (!enabled) {
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
  }
  bool clicked = ImGui::Checkbox(label, value);
  if (ImGui::IsItemHovered() && tooltip && *tooltip) {
    ImGui::SetTooltip("%s", tooltip);
  }
  if (!enabled) {
    ImGui::PopStyleVar();
    ImGui::PopItemFlag();
  }
  return clicked;
}

bool UiColorPicker(char const* label, Color* color) {
  bool changed = false;

  ImGui::PushID(label);
  MOCHI_DEFER(ImGui::PopID());

  // Create input fields for R, G, B, A
  ImGui::PushItemWidth(40);
  MOCHI_DEFER(ImGui::PopItemWidth());

  int colorInts[4] = {
      static_cast<int>((*color)[0]),
      static_cast<int>((*color)[1]),
      static_cast<int>((*color)[2]),
      static_cast<int>((*color)[3])};
  char const* colorLabels[4] = {"##R", "##G", "##B", "##A"};
  char const* colorTips[4] = {"Red", "Green", "Blue", "Alpha"};

  for (int i = 0; i < 4; ++i) {
    changed |= ImGui::DragInt(colorLabels[i], &colorInts[i], 1.0f, 0, 255);
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("%s", colorTips[i]);
    }
    ImGui::SameLine();
  }

  // Display color preview box
  auto const colorF = ToFloat4(*color);
  ImVec4 colorPreview = ImVec4(colorF[0], colorF[1], colorF[2], colorF[3]);
  ImGui::ColorButton(
      "##ColorPreview",
      colorPreview,
      ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop,
      ImVec2(20, 20));
  ImGui::SameLine();

  // Display label
  ImGui::Text("%s", label);

  if (changed) {
    *color = Color{
        static_cast<uint8_t>(colorInts[0]),
        static_cast<uint8_t>(colorInts[1]),
        static_cast<uint8_t>(colorInts[2]),
        static_cast<uint8_t>(colorInts[3])};
  }

  return changed;
}

} // namespace mochi::dbg
