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

#include <imguios/common.h>

#include <imgui_internal.h>

namespace ImGuios {

void StyleColorsDefault(const ImVec4& accentColor) {
  ImVec4 accentColorLight;
  ImGui::ColorConvertRGBtoHSV(
      accentColor.x,
      accentColor.y,
      accentColor.z,
      accentColorLight.x,
      accentColorLight.y,
      accentColorLight.z);
  accentColorLight.z = ImClamp(accentColorLight.z * 1.2f, 0.0f, 1.0f);
  accentColorLight.w = 1.0f;
  ImGui::ColorConvertHSVtoRGB(
      accentColorLight.x,
      accentColorLight.y,
      accentColorLight.z,
      accentColorLight.x,
      accentColorLight.y,
      accentColorLight.z);

  const ImVec4 windowBg(0.19f, 0.18f, 0.20f, 1.00f);
  const ImVec4 frameBg(0.142f, 0.134f, 0.149f, 1.000f);
  const ImVec4 titleBg(0.097f, 0.092f, 0.102f, 1.000f);

  auto& style = ImGui::GetStyle();

  style.WindowPadding = {8, 6};
  style.FramePadding = {8, 6};
  style.CellPadding = {4, 4};
  style.ItemSpacing = {8, 6};
  style.ItemInnerSpacing = {8, 4};
  style.TouchExtraPadding = {0, 0};
  style.IndentSpacing = 20;
  style.ScrollbarSize = 18;
  style.GrabMinSize = 10;
  style.WindowBorderSize = 1;
  style.ChildBorderSize = 1;
  style.PopupBorderSize = 1;
  style.FrameBorderSize = 1;
  style.TabBorderSize = 0;
  style.WindowRounding = 4;
  style.ChildRounding = 4;
  style.FrameRounding = 4;
  style.PopupRounding = 4;
  style.ScrollbarRounding = 2;
  style.GrabRounding = 2;
  style.LogSliderDeadzone = 4;
  style.TabRounding = 4;
  style.WindowMenuButtonPosition = ImGuiDir_Right;
  style.DisabledAlpha = 0.5f;

  ImVec4* colors = style.Colors;

  colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
  colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
  colors[ImGuiCol_WindowBg] = windowBg;
  colors[ImGuiCol_ChildBg] = titleBg;
  colors[ImGuiCol_PopupBg] = titleBg;
  colors[ImGuiCol_Border] = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
  colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.06f);
  colors[ImGuiCol_FrameBg] = frameBg;
  colors[ImGuiCol_FrameBgHovered] = accentColorLight;
  colors[ImGuiCol_FrameBgActive] = accentColor;
  colors[ImGuiCol_TitleBg] = titleBg;
  colors[ImGuiCol_TitleBgActive] = titleBg;
  colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
  colors[ImGuiCol_MenuBarBg] = titleBg;
  colors[ImGuiCol_ScrollbarBg] = titleBg;
  colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
  colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
  colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
  colors[ImGuiCol_CheckMark] = accentColor;
  colors[ImGuiCol_SliderGrab] = accentColor;
  colors[ImGuiCol_SliderGrabActive] = accentColorLight;
  colors[ImGuiCol_Button] = frameBg;
  colors[ImGuiCol_ButtonHovered] = accentColorLight;
  colors[ImGuiCol_ButtonActive] = accentColor;
  colors[ImGuiCol_Header] = titleBg;
  colors[ImGuiCol_HeaderHovered] = accentColorLight;
  colors[ImGuiCol_HeaderActive] = accentColor;
  colors[ImGuiCol_Separator] = titleBg;
  colors[ImGuiCol_SeparatorHovered] = accentColorLight;
  colors[ImGuiCol_SeparatorActive] = accentColor;
  colors[ImGuiCol_ResizeGrip] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
  colors[ImGuiCol_ResizeGripHovered] = accentColorLight;
  colors[ImGuiCol_ResizeGripActive] = accentColor;
  colors[ImGuiCol_Tab] = ImVec4(0.098f, 0.090f, 0.102f, 0.510f);
  colors[ImGuiCol_TabHovered] = accentColor;
  colors[ImGuiCol_TabActive] = ImVec4(0.910f, 0.902f, 1.000f, 0.110f);
  colors[ImGuiCol_TabUnfocused] = titleBg;
  colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.910f, 0.902f, 1.000f, 0.110f);
  colors[ImGuiCol_DockingPreview] = accentColor;
  colors[ImGuiCol_DockingEmptyBg] = titleBg;
  colors[ImGuiCol_PlotLines] = accentColor;
  colors[ImGuiCol_PlotLinesHovered] = accentColor;
  colors[ImGuiCol_PlotHistogram] = accentColor;
  colors[ImGuiCol_PlotHistogramHovered] = accentColor;
  colors[ImGuiCol_TableHeaderBg] = ImVec4(0.19f, 0.19f, 0.20f, 1.00f);
  colors[ImGuiCol_TableBorderStrong] = ImVec4(0.31f, 0.31f, 0.35f, 1.00f);
  colors[ImGuiCol_TableBorderLight] = ImVec4(0.23f, 0.23f, 0.25f, 1.00f);
  colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
  colors[ImGuiCol_TableRowBgAlt] = frameBg;
  colors[ImGuiCol_TextSelectedBg] = accentColor;
  colors[ImGuiCol_DragDropTarget] = accentColor;
  colors[ImGuiCol_NavHighlight] = accentColor;
  colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
  colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
  colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
};

bool ToggleButton(const char* label, bool* toggled, const ImVec2& size) {
  bool pressed = false;
  bool dim = !*toggled;
  if (dim) {
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.25f);
  }
  if (ImGui::Button(label, size)) {
    *toggled = !(*toggled);
    pressed = true;
  }
  if (dim) {
    ImGui::PopStyleVar();
  }
  return pressed;
}

bool ToggleSwitch(const char* str_id, bool* v) {
  bool pressed = false;

  ImVec4* colors = ImGui::GetStyle().Colors;
  ImVec2 p = ImGui::GetCursorScreenPos();
  ImDrawList* draw_list = ImGui::GetWindowDrawList();

  float height = ImGui::GetFrameHeight();
  float width = height * 1.55f;
  float radius = height * 0.50f;

  ImGui::InvisibleButton(str_id, ImVec2(width, height));
  if (ImGui::IsItemClicked()) {
    *v = !*v;
    pressed = true;
  }

  /*
  ImGuiContext& gg = *GImGui;
  float ANIM_SPEED = 0.085f;
  if (gg.LastActiveId == gg.CurrentWindow->GetID(str_id)) // && g.LastActiveIdTimer < ANIM_SPEED)
    float t_anim = ImSaturate(gg.LastActiveIdTimer / ANIM_SPEED);
  */

  if (ImGui::IsItemHovered()) {
    draw_list->AddRectFilled(
        p,
        ImVec2(p.x + width, p.y + height),
        ImGui::GetColorU32(*v ? colors[ImGuiCol_ButtonActive] : ImVec4(0.78f, 0.78f, 0.78f, 1.0f)),
        height * 0.5f);
  } else {
    draw_list->AddRectFilled(
        p,
        ImVec2(p.x + width, p.y + height),
        ImGui::GetColorU32(*v ? colors[ImGuiCol_Button] : ImVec4(0.85f, 0.85f, 0.85f, 1.0f)),
        height * 0.50f);
  }
  draw_list->AddCircleFilled(
      ImVec2(p.x + radius + (*v ? 1 : 0) * (width - radius * 2.0f), p.y + radius),
      radius - 1.5f,
      IM_COL32(255, 255, 255, 255));
  return pressed;
}

bool ButtonColored(const char* label, const ImVec4& color, const ImVec2& size) {
  ImVec4 colorHover = ImLerp({1, 1, 1, color.w}, color, 0.8f);
  ImVec4 colorPress = ImLerp({0, 0, 0, color.w}, color, 0.8f);
  ImGui::PushStyleColor(ImGuiCol_Button, color);
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, colorHover);
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, colorPress);
  bool ret = ImGui::Button(label, size);
  ImGui::PopStyleColor(3);
  return ret;
}

void LabelTextLeft(const char* label, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  LabelTextLeftV(label, fmt, args);
  va_end(args);
}

void LabelTextLeftV(const char* label, const char* fmt, va_list args) {
  ImGuiWindow* window = ImGui::GetCurrentWindow();
  if (window->SkipItems) {
    return;
  }

  ImGuiContext& g = *GImGui;
  const ImGuiStyle& style = g.Style;
  const float w = ImGui::CalcItemWidth();

  const char* value_text_begin = &g.TempBuffer[0];
  const char* value_text_end =
      value_text_begin + ImFormatStringV(g.TempBuffer.begin(), g.TempBuffer.size(), fmt, args);
  const ImVec2 value_size = ImGui::CalcTextSize(value_text_begin, value_text_end, false);
  const ImVec2 label_size = ImGui::CalcTextSize(label, nullptr, true);

  const ImVec2 pos = window->DC.CursorPos;
  const ImRect label_bb(pos, pos + ImVec2(w, label_size.y + style.FramePadding.y * 2));
  const ImRect total_bb(
      pos,
      pos +
          ImVec2(
              w + (label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f),
              ImMax(value_size.y, label_size.y) + style.FramePadding.y * 2));
  ImGui::ItemSize(total_bb, style.FramePadding.y);
  if (!ImGui::ItemAdd(total_bb, 0)) {
    return;
  }

  // Render
  ImGui::RenderTextClipped(
      label_bb.Min + style.FramePadding,
      label_bb.Max,
      label,
      nullptr,
      &label_size,
      ImVec2(0.0f, 0.0f));
  if (value_size.x > 0.0f) {
    ImGui::RenderText(
        ImVec2(label_bb.Max.x + style.ItemInnerSpacing.x, label_bb.Min.y + style.FramePadding.y),
        value_text_begin,
        value_text_end);
  }
}

bool CheckboxAligned(const char* label, bool* value) {
  ImGuiContext& g = *GImGui;
  const ImGuiStyle& style = g.Style;
  ImGui::Dummy(
      ImVec2(ImGui::CalcItemWidth() - ImGui::GetFrameHeight() - style.ItemInnerSpacing.x, 0));
  ImGui::SameLine();
  return ImGui::Checkbox(label, value);
}

} // namespace ImGuios
