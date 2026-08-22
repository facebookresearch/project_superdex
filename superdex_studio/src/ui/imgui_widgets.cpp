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

#include "ui/imgui_widgets.h"
#include "app/app.h"
#include "assets/asset.h"
#include "assets/asset_manager.h"
#include "rendering/render_target.h"
#include "ui/asset_browser.h"

#include <imgui_internal.h>
#include <imguios/fonts/icons_font_awesome5.h>
#include <misc/cpp/imgui_stdlib.h>
#include <mochi_physics/utils/mochi_prefab.h>
#include <mochi_renderer/type_conversions.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/math_utils.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <numbers>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi_renderer;
using namespace ImGui;
using namespace superdex::studio;

ImVec4 superdex::studio::HashStringToColor(std::string_view str) {
  ImU32 const hashId = ImHashStr(str.data(), str.size());
  ImVec4 color = ImGui::ColorConvertU32ToFloat4(hashId);
  color.w = 1.0f;
  return color;
}

filament::math::float3 superdex::studio::WireframeColorForSurface(
    filament::math::float3 surfaceColor) {
  float h = 0.0f, s = 0.0f, v = 0.0f;
  ImGui::ColorConvertRGBtoHSV(surfaceColor.x, surfaceColor.y, surfaceColor.z, h, s, v);
  v = std::clamp(v + (v < 0.5f ? 0.5f : -0.5f), 0.0f, 1.0f);
  float r = 0.0f, g = 0.0f, b = 0.0f;
  ImGui::ColorConvertHSVtoRGB(h, s, v, r, g, b);
  return {r, g, b};
}

std::string superdex::studio::MakeUniqueFileName(
    std::string const& baseName,
    mochi::Path const& dir,
    std::string const& extension,
    std::function<bool(mochi::Path const&)> const& alsoTaken) {
  std::error_code ec;
  auto isTaken = [&](std::string const& candidate) {
    mochi::Path const path = dir / (candidate + extension);
    if (std::filesystem::exists(path.AsFilesystemPath(), ec)) {
      return true;
    }
    ec.clear();
    std::filesystem::directory_iterator it(dir.AsFilesystemPath(), ec);
    std::filesystem::directory_iterator const end;
    for (; !ec && it != end; it.increment(ec)) {
      if (mochi::Path{it->path()} == path) {
        return true;
      }
    }
    return alsoTaken ? alsoTaken(path) : false;
  };
  std::string name = baseName;
  for (int i = 2; isTaken(name); ++i) {
    name = baseName + "_" + std::to_string(i);
  }
  return name;
}

std::string superdex::studio::ComputeBatchRenamedName(
    std::string const& original,
    BatchRenameParams const& params) {
  std::string result = original;
  if (!params.find.empty()) {
    std::string replaced;
    replaced.reserve(result.size());
    size_t pos = 0;
    while (pos < result.size()) {
      auto const found = result.find(params.find, pos);
      if (found == std::string::npos) {
        replaced.append(result, pos, std::string::npos);
        break;
      }
      replaced.append(result, pos, found - pos);
      replaced.append(params.replace);
      pos = found + params.find.size();
    }
    result = std::move(replaced);
  }
  // Trim front/back characters before adding prefix/suffix.
  int const trimFront = std::max(0, params.trimFront);
  int const trimBack = std::max(0, params.trimBack);
  if (static_cast<size_t>(trimFront) + static_cast<size_t>(trimBack) >= result.size()) {
    result.clear();
  } else {
    result = result.substr(trimFront, result.size() - trimFront - trimBack);
  }
  result = params.prefix + result + params.suffix;
  if (params.caseChange == BatchRenameCase::Uppercase) {
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
      return static_cast<char>(std::toupper(c));
    });
  } else if (params.caseChange == BatchRenameCase::Lowercase) {
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
  }
  return result;
}

// Unit format implementation

// Helper function to determine format specifier ('f' for fixed-point, 'e' for scientific)
// based on value magnitude and precision thresholds.
static char GetFormatSpecifier(float value, int precision) {
  // Zero values always use fixed-point notation
  if (value == 0.0f) {
    return 'f';
  }

  // Calculate thresholds based on precision
  // Upper threshold: min(10000, 10^precision)
  float upperThreshold = 10000.0f;
  if (precision < 10) { // Avoid overflow for large precision values
    float precisionThreshold = std::pow(10.0f, static_cast<float>(precision));
    if (precisionThreshold < upperThreshold) {
      upperThreshold = precisionThreshold;
    }
  }

  // Lower threshold: max(0.001, 10^(-precision))
  float lowerThreshold = 0.001f;
  if (precision < 10) { // Avoid underflow for large precision values
    float precisionThreshold = std::pow(10.0f, static_cast<float>(-precision));
    if (precisionThreshold > lowerThreshold) {
      lowerThreshold = precisionThreshold;
    }
  }

  float absValue = std::abs(value);

  // Use scientific notation for values outside the thresholds
  if (absValue >= upperThreshold || absValue < lowerThreshold) {
    return 'e';
  }

  // Use fixed-point notation for medium values
  return 'f';
}

// Helper function to build a format string with units, using adaptive notation
// based on the provided value. Returns a pointer to a static buffer.
static char const* BuildAdaptiveFormatWithUnits(
    float value,
    char const* units,
    int precision,
    char* buf,
    int bufSize) {
  char formatSpecifier = GetFormatSpecifier(value, precision);
  // Use one less digit of precision for scientific notation to keep significant
  // digits consistent
  int displayPrecision = (formatSpecifier == 'e' && precision > 0) ? precision - 1 : precision;
  if (units == nullptr || units[0] == '\0') {
    snprintf(buf, bufSize, "%%.%d%c", displayPrecision, formatSpecifier);
  } else {
    snprintf(buf, bufSize, "%%.%d%c %s", displayPrecision, formatSpecifier, units);
  }
  return buf;
}

char const* superdex::studio::GetUnitFormat(
    UnitFormat unit,
    ArticulatedJointType jointType,
    float value,
    int precision) {
  static char buffer[64];
  char const* unitStr = "";

  switch (unit) {
    case UnitFormat::Stiffness: {
      if (jointType == ArticulatedJointType::Revolute) {
        unitStr = "N\xC2\xB7m/rad"; // N·m/rad
      } else if (jointType == ArticulatedJointType::Prismatic) {
        unitStr = "N/m";
      }
      break;
    }
    case UnitFormat::Damping:
    case UnitFormat::ViscousFriction: {
      if (jointType == ArticulatedJointType::Revolute) {
        unitStr = "N\xC2\xB7m\xC2\xB7s/rad"; // N·m·s/rad
      } else if (jointType == ArticulatedJointType::Prismatic) {
        unitStr = "N\xC2\xB7s/m"; // N·s/m
      }
      break;
    }
    case UnitFormat::Effort:
    case UnitFormat::CoulombFriction: {
      if (jointType == ArticulatedJointType::Revolute) {
        unitStr = "N\xC2\xB7m"; // N·m
      } else if (jointType == ArticulatedJointType::Prismatic) {
        unitStr = "N";
      }
      break;
    }
    case UnitFormat::Inertia: {
      if (jointType == ArticulatedJointType::Revolute) {
        unitStr = "kg\xC2\xB7m\xC2\xB2"; // kg·m²
      } else if (jointType == ArticulatedJointType::Prismatic) {
        unitStr = "kg";
      }
      break;
    }
    case UnitFormat::Position: {
      if (jointType == ArticulatedJointType::Revolute) {
        unitStr = "\xC2\xB0"; // °
      } else if (jointType == ArticulatedJointType::Prismatic) {
        unitStr = "m";
      }
      break;
    }
    case UnitFormat::Degrees: {
      unitStr = "\xC2\xB0"; // °
      break;
    }
    case UnitFormat::Length: {
      unitStr = "m";
      break;
    }
    case UnitFormat::Mass: {
      unitStr = "kg";
      break;
    }
    case UnitFormat::Density: {
      unitStr = "kg/m\xC2\xB3"; // kg/m³
      break;
    }
    case UnitFormat::InertiaTensor: {
      unitStr = ""; // Empty for now - units shown in label instead to avoid crowding
      break;
    }
  }

  return BuildAdaptiveFormatWithUnits(value, unitStr, precision, buffer, sizeof(buffer));
}

char const* superdex::studio::GetUnitFormat(UnitFormat unit, float value, int precision) {
  // Verify this is not a joint-dependent unit
  [[maybe_unused]] bool isJointDependent = false;
  switch (unit) {
    case UnitFormat::Stiffness:
    case UnitFormat::Damping:
    case UnitFormat::ViscousFriction:
    case UnitFormat::CoulombFriction:
    case UnitFormat::Inertia:
    case UnitFormat::Position:
    case UnitFormat::Effort: {
      isJointDependent = true;
      break;
    }
    case UnitFormat::Degrees:
    case UnitFormat::Length:
    case UnitFormat::Mass:
    case UnitFormat::Density:
    case UnitFormat::InertiaTensor: {
      break;
    }
  }

  MOCHI_ASSERT_VERBOSE(
      !isJointDependent,
      "GetUnitFormat(UnitFormat, int) called with joint-dependent unit. "
      "Use the overload with ArticulatedJointType parameter.");

  // Dispatch to joint-dependent version with dummy joint type
  // (standalone units will be handled correctly regardless of joint type)
  return GetUnitFormat(unit, ArticulatedJointType::Revolute, value, precision);
}

struct InputTextCallback_UserData {
  mochi::DynamicString* Str;
  ImGuiInputTextCallback ChainCallback;
  void* ChainCallbackUserData;
};

static int InputTextCallback(ImGuiInputTextCallbackData* data) {
  auto* user_data = (InputTextCallback_UserData*)data->UserData;
  if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
    // Resize string callback
    // If for some reason we refuse the new length (BufTextLen) and/or capacity (BufSize) we need to
    // set them back to what we want.
    mochi::DynamicString* str = user_data->Str;
    IM_ASSERT(data->Buf == str->c_str());
    str->resize(data->BufTextLen);
    data->Buf = (char*)str->c_str();
  } else if (user_data->ChainCallback) {
    // Forward to user callback, if any
    data->UserData = user_data->ChainCallbackUserData;
    return user_data->ChainCallback(data);
  }
  return 0;
}

void ImGui::StyleColorsSuperdex(ImVec4 const& accentColor) {
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

  ImVec4 const windowBg(0.157f, 0.165f, 0.212f, 1.000f);
  ImVec4 const frameBg(0.118f, 0.133f, 0.173f, 1.000f);
  ImVec4 const titleBg(0.098f, 0.102f, 0.129f, 1.000f);

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
  colors[ImGuiCol_TextDisabled] = ImVec4(0.389f, 0.397f, 0.449f, 1.000f);
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
  colors[ImGuiCol_ScrollbarGrab] = ImVec4(1.00f, 1.00f, 1.00f, 0.25f);
  colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(1.00f, 1.00f, 1.00f, 0.50f);
  colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(1.00f, 1.00f, 1.00f, 0.36f);
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
  colors[ImGuiCol_ResizeGrip] = ImVec4(1.00f, 1.00f, 1.00f, 0.25f);
  colors[ImGuiCol_ResizeGripHovered] = accentColorLight;
  colors[ImGuiCol_ResizeGripActive] = accentColor;
  colors[ImGuiCol_Tab] = ImVec4(0.098f, 0.090f, 0.102f, 0.510f);
  colors[ImGuiCol_TabSelected] = ImVec4(0.520f, 0.579f, 0.700f, 0.130f);
  colors[ImGuiCol_TabSelectedOverline] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
  colors[ImGuiCol_TabDimmed] = titleBg;
  colors[ImGuiCol_TabDimmedSelected] = ImVec4(0.520f, 0.579f, 0.700f, 0.130f);
  colors[ImGuiCol_TabDimmedSelectedOverline] = ImVec4(1.00f, 1.00f, 1.00f, 0.25f);
  colors[ImGuiCol_TabHovered] = accentColor;
  colors[ImGuiCol_DockingPreview] = accentColor;
  colors[ImGuiCol_DockingEmptyBg] = titleBg;
  colors[ImGuiCol_PlotLines] = accentColor;
  colors[ImGuiCol_PlotLinesHovered] = accentColor;
  colors[ImGuiCol_PlotHistogram] = accentColor;
  colors[ImGuiCol_PlotHistogramHovered] = accentColor;
  colors[ImGuiCol_TableHeaderBg] = windowBg;
  colors[ImGuiCol_TableBorderStrong] = ImVec4(1.00f, 1.00f, 1.00f, 0.13f);
  colors[ImGuiCol_TableBorderLight] = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
  colors[ImGuiCol_TableRowBg] = titleBg;
  colors[ImGuiCol_TableRowBgAlt] = frameBg;
  colors[ImGuiCol_TextSelectedBg] = accentColor;
  colors[ImGuiCol_DragDropTarget] = accentColor;
  colors[ImGuiCol_NavCursor] = accentColor;
  colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
  colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
  colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
};

void ImGui::Separator(ImVec4 const& color) {
  ImGui::PushStyleColor(ImGuiCol_Separator, color);
  ImGui::Separator();
  ImGui::PopStyleColor();
}

void ImGui::PushFramelessWidgetStyle() {
  ImVec4 const transparent(0.0f, 0.0f, 0.0f, 0.0f);
  ImGui::PushStyleColor(ImGuiCol_FrameBg, transparent);
  ImGui::PushStyleColor(ImGuiCol_Border, transparent);
  ImGui::PushStyleColor(ImGuiCol_BorderShadow, transparent);
  ImGui::PushStyleColor(ImGuiCol_Button, transparent);
}

void ImGui::PopFramelessWidgetStyle() {
  ImGui::PopStyleColor(4);
}

bool ImGui::IconSelectable(char const* label, char const* icon, float text_offset_x) {
  char id[64];
  snprintf(id, sizeof(id), "%s##%s", icon, label);
  bool ret = ImGui::Selectable(id, false, ImGuiSelectableFlags_SpanAvailWidth);
  ImGui::SameLine(text_offset_x);
  ImGui::TextUnformatted(label);
  return ret;
}

bool ImGui::InputText(
    char const* label,
    mochi::DynamicString* str,
    ImGuiInputTextFlags flags,
    ImGuiInputTextCallback callback,
    void* user_data) {
  {
    IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
    flags |= ImGuiInputTextFlags_CallbackResize;

    InputTextCallback_UserData cb_user_data;
    cb_user_data.Str = str;
    cb_user_data.ChainCallback = callback;
    cb_user_data.ChainCallbackUserData = user_data;
    return InputText(
        label, (char*)str->c_str(), str->capacity() + 1, flags, InputTextCallback, &cb_user_data);
  }
}

bool ImGui::NameInputWithCollisionCheck(
    char const* label,
    mochi::DynamicString& name,
    bool collides) {
  if (collides) {
    PushStyleColor(ImGuiCol_FrameBg, kNameConflictColor);
  }
  bool const changed = InputText(label, &name, ImGuiInputTextFlags_CharsNoBlank);
  if (collides) {
    PopStyleColor();
    if (IsItemHovered()) {
      SetTooltip("Another actor already uses this name.");
    }
  }
  return changed;
}

bool ImGui::BatchRenameInputs(superdex::studio::BatchRenameParams& params, int trimMax) {
  using superdex::studio::BatchRenameCase;
  bool changed = false;
  changed |= ImGui::InputText("Find", &params.find);
  changed |= ImGui::InputText("Replace", &params.replace);
  int trim[2] = {params.trimFront, params.trimBack};
  if (ImGui::SliderInt2("Trim Front/Back", trim, 0, trimMax)) {
    params.trimFront = std::max(0, trim[0]);
    params.trimBack = std::max(0, trim[1]);
    changed = true;
  }
  changed |= ImGui::InputText("Prefix", &params.prefix);
  changed |= ImGui::InputText("Suffix", &params.suffix);
  char const* kCaseLabels[] = {"None", "Uppercase", "Lowercase"};
  int caseIdx = static_cast<int>(params.caseChange);
  if (ImGui::Combo("Case Change", &caseIdx, kCaseLabels, IM_ARRAYSIZE(kCaseLabels))) {
    params.caseChange = static_cast<BatchRenameCase>(caseIdx);
    changed = true;
  }
  return changed;
}

void ImGui::BatchRenamePreviewTable(
    std::vector<std::string> const& oldNames,
    std::vector<std::string> const& newNames,
    std::vector<bool> const& rowInvalid,
    float heightPx) {
  if (ImGui::BeginTable(
          "##BatchRenamePreview",
          2,
          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
              ImGuiTableFlags_SizingStretchSame,
          ImVec2(0, heightPx))) {
    ImGui::TableSetupColumn("Old Name");
    ImGui::TableSetupColumn("New Name");
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableHeadersRow();
    for (size_t i = 0; i < oldNames.size(); ++i) {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(oldNames[i].c_str());
      ImGui::TableNextColumn();
      if (rowInvalid[i]) {
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(230, 100, 100, 255));
        ImGui::TextUnformatted(newNames[i].empty() ? "<empty>" : newNames[i].c_str());
        ImGui::PopStyleColor();
      } else {
        ImGui::TextUnformatted(newNames[i].c_str());
      }
    }
    ImGui::EndTable();
  }
}

void ImGui::RenderTargetImage(RenderTarget const* renderTarget, int width, int height) {
  auto alpha = ImGui::GetStyle().Alpha;
  int w, h;
  renderTarget->GetSize(w, h);
  Image(
      (void*)(intptr_t)renderTarget->GetTextureId(),
      ImVec2(width == -1 ? w : width, height == -1 ? h : height),
      kDefaultRenderTargetUV0,
      kDefaultRenderTargetUV1,
      ImVec4(1, 1, 1, alpha));
}

bool ImGui::RenderTargetButton(
    char const* str_id,
    RenderTarget const* renderTarget,
    int width,
    int height) {
  auto alpha = ImGui::GetStyle().Alpha;
  PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
  int w, h;
  renderTarget->GetSize(w, h);
  auto ret = ImageButton(
      str_id,
      (void*)(intptr_t)renderTarget->GetTextureId(),
      ImVec2(width == -1 ? w : width, height == -1 ? h : height),
      kDefaultRenderTargetUV0,
      kDefaultRenderTargetUV1,
      ImVec4(1, 1, 1, alpha));
  PopStyleVar();
  return ret;
}

bool ImGui::DragFloatXYZ(
    char const* label,
    float v[3],
    float v_speed,
    float v_min,
    float v_max,
    char const* format,
    ImGuiSliderFlags flags) {
  bool value_changed = false;

  ImGuiStyle const& style = GetStyle();
  float const full_width = CalcItemWidth();
  float const spacing = style.ItemInnerSpacing.x;
  float const line_height = GetFrameHeight();

  // Each component is a drag input with a colored X/Y/Z badge drawn over its left edge.
  // Layout: [|drag] spacing [|drag] spacing [|drag]  (| = axis swatch)
  float const swatch_width = line_height * 0.25f;
  float const component_width = (full_width - spacing * 2.0f) / 3.0f;

  PushID(label);
  BeginGroup();

  for (int i = 0; i < 3; ++i) {
    PushID(i);
    if (i > 0) {
      SameLine(0, spacing);
    }

    ImVec2 const drag_pos = GetCursorScreenPos();
    SetNextItemWidth(component_width);
    value_changed |= DragFloat("##v", &v[i], v_speed, v_min, v_max, format, flags);

    // Non-interactive RGB axis badge overlaid on the left edge of the drag input.
    ImVec2 const swatch_max(drag_pos.x + swatch_width, drag_pos.y + line_height);
    GetWindowDrawList()->AddRectFilled(
        drag_pos, swatch_max, kAxes[i].normal, style.FrameRounding, ImDrawFlags_RoundCornersLeft);

    PopID();
  }

  EndGroup();

  // Draw the right-aligned label (standard ImGui widget label)
  char const* label_end = FindRenderedTextEnd(label);
  if (label != label_end) {
    SameLine(0, style.ItemInnerSpacing.x);
    TextEx(label, label_end);
  }

  PopID();
  return value_changed;
}

// -- Shared tile helpers --

static ImU32 ApplyAlpha(ImU32 col, float alpha) {
  int a = static_cast<int>(((col >> IM_COL32_A_SHIFT) & 0xFF) * alpha);
  return (col & ~IM_COL32_A_MASK) | (static_cast<ImU32>(a) << IM_COL32_A_SHIFT);
}

static void DrawTileTypeText(
    ImFont const* font,
    float font_size,
    float text_x,
    float text_max_x,
    float y,
    char const* type,
    ImU32 color) {
  ImGuiWindow* window = GImGui->CurrentWindow;
  ImVec2 const type_size = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, type);
  float const type_x = ImMax(text_x, text_max_x - type_size.x);
  window->DrawList->AddText(font, font_size, ImVec2(type_x, y), color, type);
}

static void
DrawTileBorder(ImRect const& bb, float rounding, bool held, bool hovered, bool selected) {
  ImGuiWindow* window = GImGui->CurrentWindow;
  if (held && hovered) {
    window->DrawList->AddRect(
        bb.Min, bb.Max, GetColorU32(ImGuiCol_FrameBgActive), rounding, 0, 2.0f);
  } else if (hovered) {
    window->DrawList->AddRect(
        bb.Min, bb.Max, GetColorU32(ImGuiCol_FrameBgHovered), rounding, 0, 2.0f);
  } else if (selected) {
    window->DrawList->AddRect(
        bb.Min, bb.Max, GetColorU32(ImGuiCol_FrameBgActive), rounding, 0, 2.0f);
  }
}

static bool DrawTileNameOrRename(
    float text_x,
    float text_y,
    float text_width,
    float name_line_height,
    char const* name,
    ImGui::TileRenameState* rename) {
  ImGuiContext& g = *GImGui;
  ImGuiWindow* window = g.CurrentWindow;

  if (rename) {
    ImVec2 const savedCursorPos = window->DC.CursorPos;
    ImVec2 const savedCursorMaxPos = window->DC.CursorMaxPos;
    ImVec2 const savedCursorPosPrevLine = window->DC.CursorPosPrevLine;
    ImVec2 const savedPrevLineSize = window->DC.PrevLineSize;
    ImVec2 const savedCurrLineSize = window->DC.CurrLineSize;
    ImGuiLastItemData const savedLastItemData = g.LastItemData;

    SetCursorScreenPos(ImVec2(text_x, text_y));
    PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 0));
    if (!rename->valid) {
      PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.5f, 0.0f, 0.0f, 1.0f));
    }
    SetNextItemWidth(text_width);

    ImGuiID const renameId = window->GetID("##rename");
    if (g.ActiveId != renameId) {
      SetKeyboardFocusHere();
    }

    char buf[256];
    strncpy(buf, rename->buffer.c_str(), sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    ImGuiInputTextFlags const flags =
        ImGuiInputTextFlags_AutoSelectAll | ImGuiInputTextFlags_EnterReturnsTrue;
    bool const enterPressed = InputText("##rename", buf, sizeof(buf), flags);
    rename->buffer = buf;

    if (!rename->valid) {
      PopStyleColor();
    }

    bool const escapedPressed = IsKeyPressed(ImGuiKey_Escape);
    bool const clickedAway = IsMouseClicked(ImGuiMouseButton_Left) && !IsItemHovered();

    bool const shouldCommit = rename->valid && (enterPressed || clickedAway);
    bool const shouldCancel = escapedPressed || (clickedAway && !rename->valid);

    if (shouldCommit && rename->onFinished) {
      rename->onFinished(rename->buffer);
    } else if (shouldCancel && rename->onCanceled) {
      rename->onCanceled();
    }

    PopStyleVar();

    window->DC.CursorPos = savedCursorPos;
    window->DC.CursorMaxPos = savedCursorMaxPos;
    window->DC.CursorPosPrevLine = savedCursorPosPrevLine;
    window->DC.PrevLineSize = savedPrevLineSize;
    window->DC.CurrLineSize = savedCurrLineSize;
    g.LastItemData = savedLastItemData;

    return true;
  }

  // Normal name text with custom 2-line wrap. Prefer breaking at the last space
  // or underscore in the line-1-fits range; spaces are consumed (no leading
  // space on line 2), underscores are pushed to line 2. Line 2 is ellipsized if
  // the remainder doesn't fit.
  char const* name_end_ptr = name + strlen(name);

  auto measure = [&](char const* s, char const* e) {
    return g.Font->CalcTextSizeA(g.FontSize, FLT_MAX, 0.0f, s, e).x;
  };

  // Draws [s, e) at the given y. If it doesn't fit in text_width, trailing characters are dropped
  // and an ellipsis appended so it fits.
  auto drawEllipsized = [&](char const* s, char const* e, float y) {
    if (measure(s, e) <= text_width) {
      window->DrawList->AddText(
          g.Font, g.FontSize, ImVec2(text_x, y), GetColorU32(ImGuiCol_Text), s, e);
      return;
    }
    char buf[260];
    auto const remain_len = static_cast<size_t>(e - s);
    auto len = ImMin(remain_len, static_cast<size_t>(256));
    while (len > 0) {
      std::copy_n(s, len, buf);
      std::copy_n("...", 4, buf + len);
      if (measure(buf, buf + len + 3) <= text_width) {
        break;
      }
      --len;
    }
    if (len == 0) {
      std::copy_n("...", 4, buf);
    }
    window->DrawList->AddText(
        g.Font, g.FontSize, ImVec2(text_x, y), GetColorU32(ImGuiCol_Text), buf, buf + len + 3);
  };

  if (measure(name, name_end_ptr) <= text_width) {
    // Fits on a single line.
    window->DrawList->AddText(
        g.Font, g.FontSize, ImVec2(text_x, text_y), GetColorU32(ImGuiCol_Text), name, name_end_ptr);
    return false;
  }

  // Find the longest prefix [name, hard_end) that fits on a single line.
  char const* hard_end = name_end_ptr;
  while (hard_end > name && measure(name, hard_end) > text_width) {
    --hard_end;
  }

  // Scan backward from hard_end for the last space/underscore break. Stop before the very first
  // character: a break there (e.g. a leading underscore) would leave line 1
  // empty. If no interior break exists (a single long word), ellipsize on one line rather than
  // hard-breaking mid-word, which would orphan a few trailing characters on line 2.
  char const* line1_end = hard_end;
  char const* line2_start = hard_end;
  bool foundBreak = false;
  for (char const* p = hard_end; p > name + 1; --p) {
    char const c = *(p - 1);
    if (c == ' ' || c == '\t') {
      line1_end = p - 1;
      line2_start = p; // consume the whitespace
      foundBreak = true;
      break;
    }
    if (c == '_') {
      line1_end = p - 1;
      line2_start = p - 1; // keep underscore on line 2
      foundBreak = true;
      break;
    }
  }

  if (!foundBreak) {
    // Single long word with no break point: ellipsize on one line.
    drawEllipsized(name, name_end_ptr, text_y);
    return false;
  }

  // Draw line 1.
  window->DrawList->AddText(
      g.Font, g.FontSize, ImVec2(text_x, text_y), GetColorU32(ImGuiCol_Text), name, line1_end);

  if (line2_start >= name_end_ptr) {
    return false;
  }

  // Draw line 2, ellipsizing if it doesn't fit.
  float const text_y2 = text_y + name_line_height;
  drawEllipsized(line2_start, name_end_ptr, text_y2);
  return false;
}

static void
DrawCheckerBackground(ImDrawList* dl, ImVec2 bb_min, float size, int num_checker, float rounding) {
  ImVec2 const upper_max(bb_min.x + size, bb_min.y + size);
  dl->AddRectFilled(
      bb_min, upper_max, GetColorU32(ImGuiCol_PopupBg), rounding, ImDrawFlags_RoundCornersTop);
  if (num_checker > 1) {
    float const cell_size = size / static_cast<float>(num_checker);
    ImU32 const col_popup = GetColorU32(ImGuiCol_PopupBg);
    ImU32 const col_frame = GetColorU32(ImGuiCol_FrameBg);
    for (int row = 0; row < num_checker; ++row) {
      for (int col = 0; col < num_checker; ++col) {
        if ((row + col) % 2 == 1) {
          continue;
        }
        float const x0 = bb_min.x + static_cast<float>(col) * cell_size;
        float const y0 = bb_min.y + static_cast<float>(row) * cell_size;
        float const x1 = x0 + cell_size;
        float const y1 = y0 + cell_size;
        ImU32 const cell_col = ((row + col) % 2 == 0) ? col_frame : col_popup;
        float cell_rounding = 0.0f;
        ImDrawFlags cell_flags = 0;
        if (row == 0 && col == 0) {
          cell_rounding = rounding;
          cell_flags = ImDrawFlags_RoundCornersTopLeft;
        } else if (row == 0 && col == num_checker - 1) {
          cell_rounding = rounding;
          cell_flags = ImDrawFlags_RoundCornersTopRight;
        }
        dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), cell_col, cell_rounding, cell_flags);
      }
    }
  }
}

static void DrawCenteredIconWithShadow(
    ImDrawList* dl,
    ImFont const* font,
    char const* icon,
    ImVec2 area_min,
    float area_width,
    float area_height,
    float global_font_scale,
    float alpha,
    ImU32 icon_color) {
  float const font_size = font->FontSize * global_font_scale;
  ImVec2 const icon_size = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, icon);
  float const icon_x = area_min.x + (area_width - icon_size.x) * 0.5f;
  float const icon_y = area_min.y + (area_height - icon_size.y) * 0.5f;
  dl->AddText(
      font,
      font_size,
      ImVec2(icon_x + 2, icon_y + 2),
      ApplyAlpha(IM_COL32(0, 0, 0, 128), alpha),
      icon);
  dl->AddText(font, font_size, ImVec2(icon_x, icon_y), icon_color, icon);
}

static ImU32 LightenColor(ImU32 color, int amount) {
  int const cr = ImMin(255, static_cast<int>((color >> 0) & 0xFF) + amount);
  int const cg = ImMin(255, static_cast<int>((color >> 8) & 0xFF) + amount);
  int const cb = ImMin(255, static_cast<int>((color >> 16) & 0xFF) + amount);
  return IM_COL32(cr, cg, cb, 255);
}

static void
DrawDropShadow(ImDrawList* dl, ImRect const& bb, float offset, float alpha, float rounding) {
  ImRect const shadow_bb(
      ImVec2(bb.Min.x + offset, bb.Min.y + offset), ImVec2(bb.Max.x + offset, bb.Max.y + offset));
  dl->AddRectFilled(
      shadow_bb.Min, shadow_bb.Max, ApplyAlpha(IM_COL32(0, 0, 0, 128), alpha), rounding);
}

static void DrawTileBackground(
    ImDrawList* dl,
    ImRect const& bb,
    float rounding,
    bool held,
    bool hovered,
    bool selected) {
  ImU32 const col = GetColorU32(
      (held && hovered) ? ImGuiCol_FrameBgActive
          : hovered     ? ImGuiCol_FrameBgHovered
          : selected    ? ImGuiCol_FrameBgActive
                        : ImGuiCol_FrameBg);
  dl->AddRectFilled(bb.Min, bb.Max, col, rounding);
}

static float ComputeTileBottomHeight(float font_size, float small_font_size, float padding) {
  return padding + font_size * 2.0f + small_font_size + padding;
}

static bool DrawTileBottomText(
    ImRect const& bb,
    float stripe_y,
    float stripe_height,
    float padding,
    float name_line_height,
    ImFont const* small_font,
    float small_font_size,
    char const* name,
    char const* type,
    ImU32 type_color,
    ImGui::TileRenameState* rename) {
  float const text_x = bb.Min.x + padding;
  float const text_max_x = bb.Max.x - padding;
  float const text_width = text_max_x - text_x;
  ImGui::PushClipRect(ImVec2(text_x, stripe_y + stripe_height), ImVec2(text_max_x, bb.Max.y), true);
  if (type) {
    float const type_y = bb.Max.y - small_font_size - padding;
    DrawTileTypeText(small_font, small_font_size, text_x, text_max_x, type_y, type, type_color);
  }
  float const text_y = stripe_y + stripe_height + padding;
  bool const result =
      DrawTileNameOrRename(text_x, text_y, text_width, name_line_height, name, rename);
  ImGui::PopClipRect();
  return result;
}

bool ImGui::AssetThumbnail(
    char const* str_id,
    ImTextureID image,
    float thumb_size,
    ImU32 stripe_color,
    ImVec2 const& uv0,
    ImVec2 const& uv1,
    int num_checker) {
  ImGuiContext& g = *GImGui;
  ImGuiWindow* window = g.CurrentWindow;
  if (window->SkipItems) {
    return false;
  }

  ImGuiStyle const& style = g.Style;
  ImGuiID const id = window->GetID(str_id);
  float const alpha = style.Alpha;
  stripe_color = ApplyAlpha(stripe_color, alpha);

  float const tile_width = thumb_size;
  float const stripe_height = 2.0f;
  float const tile_height = thumb_size + stripe_height;
  float const rounding = style.FrameRounding;

  ImVec2 const pos = window->DC.CursorPos;
  ImRect const bb(pos, ImVec2(pos.x + tile_width, pos.y + tile_height));

  ItemSize(bb);
  if (!ItemAdd(bb, id)) {
    return false;
  }

  bool hovered = false;
  bool held = false;
  bool pressed = ButtonBehavior(bb, id, &hovered, &held);

  // -- Draw checkered or solid background --
  DrawCheckerBackground(window->DrawList, bb.Min, thumb_size, num_checker, rounding);

  // -- Draw image on top of background --
  if (image) {
    ImVec2 const image_max(bb.Min.x + tile_width, bb.Min.y + thumb_size);
    ImU32 const image_tint = ImGui::ColorConvertFloat4ToU32({1.0, 1.0, 1.0, alpha});
    window->DrawList->AddImageRounded(
        image, bb.Min, image_max, uv0, uv1, image_tint, rounding, ImDrawFlags_RoundCornersTop);
  }

  // -- Draw stripe --
  if (!hovered && !held) {
    float const stripe_y = bb.Min.y + thumb_size;
    window->DrawList->AddRectFilled(
        ImVec2(bb.Min.x, stripe_y), ImVec2(bb.Max.x, stripe_y + stripe_height), stripe_color);
  }

  // -- Draw border --
  DrawTileBorder(bb, rounding, held, hovered, false);

  return pressed;
}

bool ImGui::SimpleAssetThumbnail(
    char const* str_id,
    float thumb_size,
    char const* icon,
    ImFont const* icon_font,
    ImU32 stripe_color) {
  ImGuiContext& g = *GImGui;
  ImGuiWindow* window = g.CurrentWindow;
  if (window->SkipItems) {
    return false;
  }

  ImGuiStyle const& style = g.Style;
  ImGuiID const id = window->GetID(str_id);
  float const alpha = style.Alpha;
  stripe_color = ApplyAlpha(stripe_color, alpha);

  float const tile_width = thumb_size;
  float const stripe_height = 2.0f;
  float const tile_height = thumb_size + stripe_height;
  float const rounding = style.FrameRounding;

  ImVec2 const pos = window->DC.CursorPos;
  ImRect const bb(pos, ImVec2(pos.x + tile_width, pos.y + tile_height));

  ItemSize(bb);
  if (!ItemAdd(bb, id)) {
    return false;
  }

  bool hovered = false;
  bool held = false;
  bool pressed = ButtonBehavior(bb, id, &hovered, &held);

  // -- Draw solid background --
  ImVec2 const upper_max(bb.Max.x, bb.Min.y + thumb_size);
  window->DrawList->AddRectFilled(
      bb.Min, upper_max, GetColorU32(ImGuiCol_PopupBg), rounding, ImDrawFlags_RoundCornersTop);

  // -- Draw icon centered --
  DrawCenteredIconWithShadow(
      window->DrawList,
      icon_font,
      icon,
      bb.Min,
      tile_width,
      thumb_size,
      g.IO.FontGlobalScale,
      alpha,
      stripe_color);

  // -- Draw stripe --
  float const stripe_y = bb.Min.y + thumb_size;
  window->DrawList->AddRectFilled(
      ImVec2(bb.Min.x, stripe_y), ImVec2(bb.Max.x, stripe_y + stripe_height), stripe_color);

  // -- Draw border --
  DrawTileBorder(bb, rounding, held, hovered, false);

  return pressed;
}

bool ImGui::FolderThumbnail(
    char const* str_id,
    float thumb_size,
    ImFont const* icon_font,
    ImU32 icon_color) {
  ImGuiContext& g = *GImGui;
  ImGuiWindow* window = g.CurrentWindow;
  if (window->SkipItems) {
    return false;
  }

  ImGuiStyle const& style = g.Style;
  ImGuiID const id = window->GetID(str_id);
  float const alpha = style.Alpha;
  icon_color = ApplyAlpha(icon_color, alpha);

  float const tile_width = thumb_size;
  float const tile_height = thumb_size;

  ImVec2 const pos = window->DC.CursorPos;
  ImRect const bb(pos, ImVec2(pos.x + tile_width, pos.y + tile_height));

  ItemSize(bb);
  if (!ItemAdd(bb, id)) {
    return false;
  }

  bool hovered = false;
  bool held = false;
  bool pressed = ButtonBehavior(bb, id, &hovered, &held);

  // -- Draw folder icon centered (lighten on hover/active) --
  int lighten = 0;
  if (hovered) {
    lighten = held ? 60 : 40;
  }
  ImU32 const draw_color = lighten > 0 ? LightenColor(icon_color, lighten) : icon_color;
  DrawCenteredIconWithShadow(
      window->DrawList,
      icon_font,
      ICON_FA_FOLDER,
      bb.Min,
      tile_width,
      tile_height,
      g.IO.FontGlobalScale,
      alpha,
      ApplyAlpha(draw_color, alpha));

  // -- Draw border --
  float const rounding = style.FrameRounding;
  DrawTileBorder(bb, rounding, held, hovered, false);

  return pressed;
}

bool ImGui::AssetTile(
    char const* str_id,
    ImTextureID image,
    float image_size,
    char const* name,
    char const* type,
    ImFont const* small_font,
    ImU32 stripe_color,
    AssetTileState state,
    bool selected,
    char const* extra_icon,
    ImU32 extra_icon_color,
    ImVec2 const& uv0,
    ImVec2 const& uv1,
    int num_checker,
    TileRenameState* rename) {
  ImGuiContext& g = *GImGui;
  ImGuiWindow* window = g.CurrentWindow;
  if (window->SkipItems) {
    return false;
  }

  ImGuiStyle const& style = g.Style;
  ImGuiID const id = window->GetID(str_id);
  float const alpha = style.Alpha;
  stripe_color = ApplyAlpha(stripe_color, alpha);
  extra_icon_color = ApplyAlpha(extra_icon_color, alpha);

  float const tile_width = image_size;
  float const padding = style.FramePadding.y;
  float const stripe_height = 2.0f;
  float const shadow_offset = 3.0f;

  // Measure text height for the bottom section (name up to 2 lines + type)
  float const name_line_height = g.FontSize;
  float const small_font_size = small_font->FontSize * g.IO.FontGlobalScale;
  float const bottom_height = ComputeTileBottomHeight(name_line_height, small_font_size, padding);
  float const tile_height = image_size + stripe_height + bottom_height;

  ImVec2 const pos = window->DC.CursorPos;
  ImRect const bb(pos, ImVec2(pos.x + tile_width, pos.y + tile_height));

  ItemSize(bb);
  if (!ItemAdd(bb, id)) {
    return false;
  }

  bool hovered = false;
  bool held = false;
  bool pressed = ButtonBehavior(bb, id, &hovered, &held);

  // -- Draw drop shadow --
  float const rounding = style.FrameRounding;
  DrawDropShadow(window->DrawList, bb, shadow_offset, alpha, rounding);

  // -- Draw bottom section background --
  DrawTileBackground(window->DrawList, bb, rounding, held, hovered, selected);

  // -- Draw upper half background (PopupBg, static or checkered) --
  DrawCheckerBackground(window->DrawList, bb.Min, image_size, num_checker, rounding);

  // -- Draw image on top of background --
  if (image) {
    ImVec2 const image_max(bb.Min.x + tile_width, bb.Min.y + image_size);
    auto const image_tint = ImGui::ColorConvertFloat4ToU32({1.0, 1.0, 1.0, alpha});
    window->DrawList->AddImageRounded(
        image, bb.Min, image_max, uv0, uv1, image_tint, rounding, ImDrawFlags_RoundCornersTop);
  }

  // -- Draw stripe --
  float const stripe_y = bb.Min.y + image_size;
  window->DrawList->AddRectFilled(
      ImVec2(bb.Min.x, stripe_y), ImVec2(bb.Max.x, stripe_y + stripe_height), stripe_color);

  // -- Draw bottom text --
  if (DrawTileBottomText(
          bb,
          stripe_y,
          stripe_height,
          padding,
          name_line_height,
          small_font,
          small_font_size,
          name,
          type,
          stripe_color,
          rename)) {
    pressed = false;
  }

  // -- Draw unsaved indicator (asterisk in top right) --
  if (state != AssetTileState_None) {
    char const* icon = state == AssetTileState_Unsaved ? ICON_FA_ASTERISK : ICON_FA_LOCK;
    ImVec2 const icon_size = g.Font->CalcTextSizeA(g.FontSize, FLT_MAX, 0.0f, icon);
    float const icon_x = bb.Max.x - icon_size.x - padding;
    float const icon_y = bb.Min.y + padding;
    window->DrawList->AddText(
        g.Font, g.FontSize, ImVec2(icon_x + 2, icon_y + 2), IM_COL32_BLACK, icon);
    window->DrawList->AddText(
        g.Font, g.FontSize, ImVec2(icon_x, icon_y), GetColorU32(ImGuiCol_Text), icon);
  }

  // -- Draw extra icon --
  if (extra_icon) {
    // ImVec2 const icon_size = g.Font->CalcTextSizeA(g.FontSize, FLT_MAX, 0.0f, extra_icon);
    float const icon_x = bb.Min.x + padding;
    float const icon_y = bb.Min.y + padding;
    window->DrawList->AddText(
        g.Font, g.FontSize, ImVec2(icon_x + 2, icon_y + 2), IM_COL32_BLACK, extra_icon);
    window->DrawList->AddText(
        g.Font, g.FontSize, ImVec2(icon_x, icon_y), extra_icon_color, extra_icon);
  }

  // -- Draw hover/active/selected border around the entire tile --
  DrawTileBorder(bb, rounding, held, hovered, selected);

  return pressed;
}

bool ImGui::SimpleAssetTile(
    char const* str_id,
    float tile_size,
    char const* icon,
    ImFont const* icon_font,
    char const* name,
    char const* type,
    ImFont const* small_font,
    ImU32 stripe_color,
    bool selected,
    TileRenameState* rename) {
  ImGuiContext& g = *GImGui;
  ImGuiWindow* window = g.CurrentWindow;
  if (window->SkipItems) {
    return false;
  }

  ImGuiStyle const& style = g.Style;
  ImGuiID const id = window->GetID(str_id);
  float const alpha = style.Alpha;
  stripe_color = ApplyAlpha(stripe_color, alpha);

  float const tile_width = tile_size;
  float const padding = style.FramePadding.y;
  float const stripe_height = 2.0f;
  float const shadow_offset = 3.0f;

  float const name_line_height = g.FontSize;
  float const small_font_size = small_font->FontSize * g.IO.FontGlobalScale;
  float const bottom_height = ComputeTileBottomHeight(name_line_height, small_font_size, padding);
  float const tile_height = tile_size + stripe_height + bottom_height;

  ImVec2 const pos = window->DC.CursorPos;
  ImRect const bb(pos, ImVec2(pos.x + tile_width, pos.y + tile_height));

  ItemSize(bb);
  if (!ItemAdd(bb, id)) {
    return false;
  }

  bool hovered = false;
  bool held = false;
  bool pressed = ButtonBehavior(bb, id, &hovered, &held);
  float const rounding = style.FrameRounding;

  // -- Drop shadow --
  DrawDropShadow(window->DrawList, bb, shadow_offset, alpha, rounding);

  // -- Bottom section background --
  DrawTileBackground(window->DrawList, bb, rounding, held, hovered, selected);

  // -- Upper half background (solid PopupBg) --
  ImVec2 const upper_max(bb.Max.x, bb.Min.y + tile_size);
  window->DrawList->AddRectFilled(
      bb.Min, upper_max, GetColorU32(ImGuiCol_PopupBg), rounding, ImDrawFlags_RoundCornersTop);

  // -- Draw icon centered in upper area --
  DrawCenteredIconWithShadow(
      window->DrawList,
      icon_font,
      icon,
      bb.Min,
      tile_width,
      tile_size,
      g.IO.FontGlobalScale,
      alpha,
      stripe_color);

  // -- Stripe --
  float const stripe_y = bb.Min.y + tile_size;
  window->DrawList->AddRectFilled(
      ImVec2(bb.Min.x, stripe_y), ImVec2(bb.Max.x, stripe_y + stripe_height), stripe_color);

  // -- Draw bottom text --
  if (DrawTileBottomText(
          bb,
          stripe_y,
          stripe_height,
          padding,
          name_line_height,
          small_font,
          small_font_size,
          name,
          type,
          stripe_color,
          rename)) {
    pressed = false;
  }

  // -- Border --
  DrawTileBorder(bb, rounding, held, hovered, selected);

  return pressed;
}

bool ImGui::FolderTile(
    char const* str_id,
    float tile_size,
    ImFont const* icon_font,
    char const* name,
    ImFont const* small_font,
    ImU32 icon_color,
    bool selected,
    TileRenameState* rename) {
  ImGuiContext& g = *GImGui;
  ImGuiWindow* window = g.CurrentWindow;
  if (window->SkipItems) {
    return false;
  }

  ImGuiStyle const& style = g.Style;
  ImGuiID const id = window->GetID(str_id);
  float const alpha = style.Alpha;
  icon_color = ApplyAlpha(icon_color, alpha);

  float const tile_width = tile_size;
  float const padding = style.FramePadding.y;
  float const stripe_height = 2.0f;

  float const name_line_height = g.FontSize;
  float const small_font_size = small_font->FontSize * g.IO.FontGlobalScale;
  float const bottom_height = ComputeTileBottomHeight(name_line_height, small_font_size, padding);
  float const tile_height = tile_size + stripe_height + bottom_height;

  ImVec2 const pos = window->DC.CursorPos;
  ImRect const bb(pos, ImVec2(pos.x + tile_width, pos.y + tile_height));

  ItemSize(bb);
  if (!ItemAdd(bb, id)) {
    return false;
  }

  bool hovered = false;
  bool held = false;
  bool pressed = ButtonBehavior(bb, id, &hovered, &held);

  // -- Draw icon centered (lighten on hover, more on active/selected) --
  int lighten = 0;
  if (hovered) {
    lighten = held ? 60 : (selected ? 80 : 40);
  } else if (selected) {
    lighten = 60;
  }
  ImU32 const draw_color = lighten > 0 ? LightenColor(icon_color, lighten) : icon_color;
  DrawCenteredIconWithShadow(
      window->DrawList,
      icon_font,
      ICON_FA_FOLDER,
      bb.Min,
      tile_width,
      tile_size,
      g.IO.FontGlobalScale,
      alpha,
      ApplyAlpha(draw_color, alpha));

  // -- Name text --
  float const stripe_y = bb.Min.y + tile_size;
  if (DrawTileBottomText(
          bb,
          stripe_y,
          stripe_height,
          padding,
          name_line_height,
          small_font,
          small_font_size,
          name,
          nullptr,
          0,
          rename)) {
    pressed = false;
  }

  return pressed;
}

bool ImGui::AssetSlot(
    char const* label,
    mochi::DynamicString& assetPath,
    AssetManager const& assetManager,
    SuperDexStudio* studio,
    AssetType assetType,
    bool acceptDragDropPayload,
    bool showClearButton) {
  constexpr float thumbnailSize = 64;
  bool changed = false;
  // Scope all internal widget IDs under the slot label so this widget's icon-only buttons
  // (search/trash) never collide with identical buttons drawn by the caller or by sibling slots.
  ImGui::PushID(label);
  auto asset = assetManager.FindAssetByPath(assetPath);
  if (asset) {
    // Use a constant thumbnail ID (not the asset name) so it never collides with the slot's
    // InputText label — e.g. a prefab named "Prefab" in a slot labeled "Prefab".
    ImGui::AssetThumbnail(
        "##thumbnail", asset->GetThumbnailImage(), thumbnailSize, asset->GetColor());
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
      studio->OpenAssetEditor(asset);
    }
  } else {
    ImGui::AssetThumbnail("##thumbnail", nullptr, thumbnailSize, GetAssetTypeColor(assetType));
  }
  if (acceptDragDropPayload && ImGui::BeginDragDropTarget()) {
    // Peek first to validate without drawing the highlight rect. Accept any single asset of the
    // matching type, whether or not it has been loaded yet: the drag payload carries the type
    // (classified from the filename when the asset is not yet loaded) and the full path.
    if (auto const* peek = ImGui::AcceptDragDropPayload(
            kAssetBrowserDragDropType, ImGuiDragDropFlags_AcceptPeekOnly)) {
      auto const* dragPayload = *static_cast<AssetBrowserDragDropPayload* const*>(peek->Data);
      bool const valid = dragPayload && dragPayload->items.size() == 1 &&
          !dragPayload->items[0].isFolder && dragPayload->items[0].type == assetType;
      if (valid) {
        // Accept for real (draws highlight, delivers on release)
        if (ImGui::AcceptDragDropPayload(kAssetBrowserDragDropType)) {
          auto const& item = dragPayload->items[0];
          // Load the asset if it has not been loaded yet (a freshly-browsed asset has no Asset* and
          // no thumbnail) so the drop populates and renders the slot rather than being rejected.
          studio->GetAssetManager().LoadAsset(item.fullPath);
          assetPath = (item.asset ? item.asset->GetPath() : item.fullPath).ToString();
          changed |= true;
        }
      } else {
        ImGui::SetMouseCursor(ImGuiMouseCursor_NotAllowed);
      }
    }
    ImGui::EndDragDropTarget();
  }
  ImGui::SameLine();
  auto width = ImGui::CalcItemWidth() - ImGui::GetCursorPosX() + ImGui::GetCursorStartPos().x;
  ImGui::BeginGroup();

  ImGuiStyle const& style = ImGui::GetStyle();

  // Asset picker button: shows the asset's filename (name + extension), left-aligned. Hovering
  // reveals the full path; clicking opens a picker popup listing all assets of this slot's type.
  std::string const displayName = asset ? asset->GetPath().GetFilename() : std::string("(None)");
  std::string buttonLabel = displayName;
  buttonLabel += "###"; // stable ID from the slot label; the visible text stays the filename
  buttonLabel += label;
  std::string popupId = "##AssetSlotPicker";
  popupId += label;

  ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));
  bool const openPicker = ImGui::Button(buttonLabel.c_str(), ImVec2(width, 0.0f));
  ImGui::PopStyleVar();
  ImVec2 const buttonMin = ImGui::GetItemRectMin();
  ImVec2 const buttonMax = ImGui::GetItemRectMax();

  // Draw the visible label to the right of the button, mirroring InputText's label placement.
  if (label[0] != '\0' && (label[0] != '#' || label[1] != '#')) {
    ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
    ImGui::TextUnformatted(label);
  }

  if (openPicker) {
    ImGui::OpenPopup(popupId.c_str());
  }
  ImGui::SetNextWindowPos(ImVec2(buttonMin.x, buttonMax.y));
  ImGui::SetNextWindowSize(ImVec2(width, 0.0f));
  if (ImGui::BeginPopup(popupId.c_str())) {
    // Sticky name filter, pinned above the scrolling list.
    static ImGuiTextFilter assetFilter;
    // Candidate list is fetched once when the popup opens (not every frame): GetAllAssetsOfType
    // allocates a fresh vector per call, and only one AssetSlot popup is open at a time.
    static std::vector<Asset*> candidates;
    if (ImGui::IsWindowAppearing()) {
      assetFilter.Clear();
      candidates = assetManager.GetAllAssetsOfType(assetType);
      ImGui::SetKeyboardFocusHere();
    }
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    if (ImGui::InputTextWithHint(
            "##AssetFilter",
            ICON_FA_FILTER " Filter",
            assetFilter.InputBuf,
            IM_ARRAYSIZE(assetFilter.InputBuf))) {
      assetFilter.Build();
    }
    ImGui::Separator();

    // Scrolling list of matching assets; the filter above stays fixed.
    constexpr float kPickerListHeight = 300.0f;
    float const rowThumbSize = thumbnailSize * 0.75f;
    if (ImGui::BeginChild("##AssetList", ImVec2(0.0f, kPickerListHeight))) {
      for (Asset* candidate : candidates) {
        if (assetFilter.IsActive() && !assetFilter.PassFilter(candidate->GetName().c_str())) {
          continue;
        }
        ImGui::PushID(candidate);
        ImVec2 const rowStart = ImGui::GetCursorScreenPos();
        float const rowWidth = ImGui::GetContentRegionAvail().x;
        bool const selected = ImGui::Selectable(
            "##row", candidate == asset, ImGuiSelectableFlags_None, ImVec2(rowWidth, rowThumbSize));
        ImVec2 const afterRow = ImGui::GetCursorScreenPos();
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        // Thumbnail image on the left (no tile background/stripe/border).
        if (void* const thumb = candidate->GetThumbnailImage()) {
          ImVec2 const imgMin(rowStart.x, rowStart.y);
          ImVec2 const imgMax(imgMin.x + rowThumbSize, imgMin.y + rowThumbSize);
          drawList->AddImage(
              thumb, imgMin, imgMax, kDefaultRenderTargetUV0, kDefaultRenderTargetUV1);
        }
        // To the right of the thumbnail: the asset name, then its full path in a small dimmed font.
        ImFont* const smallFont = studio->GetFont("Roboto Regular Small");
        float const pathHeight = smallFont ? smallFont->FontSize : ImGui::GetTextLineHeight();
        float const groupHeight = ImGui::GetTextLineHeight() + pathHeight;
        ImGui::SetCursorScreenPos(ImVec2(
            rowStart.x + rowThumbSize + style.ItemInnerSpacing.x,
            rowStart.y + (rowThumbSize - groupHeight) * 0.5f));
        ImGui::BeginGroup();
        ImGui::TextUnformatted(candidate->GetName().c_str());
        if (smallFont) {
          ImGui::PushFont(smallFont);
        }
        std::string const pathStr = candidate->GetPath().ToString();
        float const pathAvail = rowStart.x + rowWidth - ImGui::GetCursorScreenPos().x;
        if (ImGui::CalcTextSize(pathStr.c_str()).x <= pathAvail) {
          ImGui::TextDisabled("%s", pathStr.c_str());
        } else {
          // Path overflows: keep the trailing (most specific) part visible and prefix it with "..."
          // so the leading, clipped portion is clearly indicated. CalcTextSize of the tail
          // shrinks monotonically as the start offset grows, so binary-search the smallest offset
          // whose tail fits instead of advancing one character at a time.
          float const remaining = pathAvail - ImGui::CalcTextSize("...").x;
          size_t lo = 0;
          size_t hi = pathStr.size();
          while (lo < hi) {
            size_t const mid = lo + (hi - lo) / 2;
            if (ImGui::CalcTextSize(pathStr.c_str() + mid).x > remaining) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }
          ImGui::TextDisabled("...%s", pathStr.c_str() + lo);
        }
        if (smallFont) {
          ImGui::PopFont();
        }
        ImGui::EndGroup();
        ImGui::SetCursorScreenPos(afterRow);
        if (selected) {
          assetPath = candidate->GetPath().ToString();
          changed = true;
          ImGui::CloseCurrentPopup();
        }
        ImGui::PopID();
      }
    }
    ImGui::EndChild();
    ImGui::EndPopup();
  }

  if (ImGui::Button(ICON_FA_SEARCH)) {
    if (asset) {
      studio->GetAssetBrowser().SelectAsset(asset);
    }
  }
  if (showClearButton) {
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_TRASH)) {
      assetPath = "";
      changed |= true;
    }
  }
  ImGuiContext& g = *GImGui;
  ImGuiWindow* window = g.CurrentWindow;
  ImVec2 const buttonRowContPos = window->DC.CursorPosPrevLine;
  ImGui::EndGroup();
  window->DC.PrevLineSize.y = window->DC.CursorPos.y - g.Style.ItemSpacing.y - buttonRowContPos.y;
  window->DC.CursorPosPrevLine = buttonRowContPos;

  ImGui::PopID();
  return changed;
}

bool ImGui::ViewportOrientationGizmo(
    char const* str_id,
    double const* view_matrix,
    ImVec2 center,
    float radius,
    int* out_clicked_axis,
    bool* out_dragging,
    ImFont* font,
    mochi::CoordinateSpaceConverter const& converter) {
  // Reset click state
  constexpr int kNumAxes = 6;
  *out_clicked_axis = -1;
  *out_dragging = false;
  // Constants
  constexpr char const* kLabels[kNumAxes] = {"X", "Y", "Z", "-X", "-Y", "-Z"};
  constexpr float kUnitAxes[kNumAxes][3] = {
      {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}};
  constexpr float kCircleRadius = 9.0f;
  constexpr float kNegCircleRadius = 8.0f;
  constexpr float kNegCircleThickness = 2.0f;
  radius -= kCircleRadius;

  // Build a converter from display space to render space
  // (because the view_matrix is in render space).
  // For each display axis (±X, ±Y, ±Z in display space),
  // convert to render space to get the world direction for projection.
  float kDirWorld[kNumAxes][3];
  for (int i = 0; i < kNumAxes; ++i) {
    mochi::Float3 const worldDir = converter.DirectionToOutput(
        mochi::Float3{kUnitAxes[i][0], kUnitAxes[i][1], kUnitAxes[i][2]});
    kDirWorld[i][0] = worldDir[0];
    kDirWorld[i][1] = worldDir[1];
    kDirWorld[i][2] = worldDir[2];
  }

  // Window draw list.
  ImGuiWindow* window = GetCurrentWindow();
  ImDrawList* dl = GetWindowDrawList();
  // Extract look directions
  auto m = [&](int col, int row) -> float {
    return static_cast<float>(view_matrix[col * 4 + row]);
  };
  float const lookDirX = -m(2, 0);
  float const lookDirY = -m(2, 1);
  float const lookDirZ = -m(2, 2);
  // Build transient state.
  struct Projected {
    ImVec2 tip;
    float depth = 0.0f;
    int index = 0;
  };
  Projected proj[kNumAxes];
  for (int i = 0; i < kNumAxes; ++i) {
    float wx = kDirWorld[i][0];
    float wy = kDirWorld[i][1];
    float wz = kDirWorld[i][2];
    float vx = m(0, 0) * wx + m(0, 1) * wy + m(0, 2) * wz;
    float vy = m(1, 0) * wx + m(1, 1) * wy + m(1, 2) * wz;
    float vz = m(2, 0) * wx + m(2, 1) * wy + m(2, 2) * wz;
    proj[i].tip = ImVec2{center.x + vx * radius, center.y - vy * radius};
    proj[i].depth = vz;
    proj[i].index = i;
  }
  // Register the whole gizmo as a single ImGui button.
  ImGuiID const id = window->GetID(str_id);
  float const hitRadius = radius + kCircleRadius;
  ImRect const bb(
      center.x - hitRadius, center.y - hitRadius, center.x + hitRadius, center.y + hitRadius);
  ImVec2 cursorMaxBak = window->DC.CursorMaxPos;
  ItemAdd(bb, id);
  window->DC.CursorMaxPos = cursorMaxBak;
  bool gizmoHovered = false;
  bool gizmoHeld = false;
  bool gizmoClicked =
      ButtonBehavior(bb, id, &gizmoHovered, &gizmoHeld, ImGuiButtonFlags_AllowOverlap);
  // Determine which axis circle the mouse is closest to.
  constexpr float kAlignThreshold = 0.9999f;
  int closestAxisInteract = -1;
  if (gizmoHovered || gizmoHeld) {
    ImVec2 mousePos = GetIO().MousePos;
    float bestDistSq = FLT_MAX;
    for (auto& axisProj : proj) {
      float dx = mousePos.x - axisProj.tip.x;
      float dy = mousePos.y - axisProj.tip.y;
      float distSq = dx * dx + dy * dy;
      if (distSq < bestDistSq) {
        bestDistSq = distSq;
        closestAxisInteract = axisProj.index;
      }
    }
    // Blender-style toggle: when looking perfectly down an axis, the positive
    // and negative endpoints overlap on screen. If the user is already viewing
    // from that axis direction, redirect the interaction to the opposite axis
    // so that clicking toggles the view (e.g. +X -> −X).
    if (closestAxisInteract >= 0) {
      float alignment = lookDirX * kDirWorld[closestAxisInteract][0] +
          lookDirY * kDirWorld[closestAxisInteract][1] +
          lookDirZ * kDirWorld[closestAxisInteract][2];
      if (alignment < -kAlignThreshold) {
        closestAxisInteract = (closestAxisInteract + 3) % kNumAxes;
      }
    }
    if (gizmoClicked) {
      // Only report click if mouse didn't exceed drag threshold.
      // MouseDragMaxDistanceSqr tracks max distance moved while held.
      ImGuiContext& g = *GImGui;
      float drag_threshold = g.IO.MouseDragThreshold;
      bool was_dragging =
          g.IO.MouseDragMaxDistanceSqr[ImGuiMouseButton_Left] >= drag_threshold * drag_threshold;
      if (!was_dragging) {
        // Remap display axis index → Filament axis index.
        // kDirWorld[displayIdx] contains the Filament direction for display axis displayIdx.
        // Find which Filament unit axis (0-5) this direction matches.
        int filamentAxis = closestAxisInteract;
        for (int filamentIdx = 0; filamentIdx < kNumAxes; ++filamentIdx) {
          float dx = kDirWorld[closestAxisInteract][0] - kUnitAxes[filamentIdx][0];
          float dy = kDirWorld[closestAxisInteract][1] - kUnitAxes[filamentIdx][1];
          float dz = kDirWorld[closestAxisInteract][2] - kUnitAxes[filamentIdx][2];
          if (dx * dx + dy * dy + dz * dz < 0.01f) {
            filamentAxis = filamentIdx;
            break;
          }
        }
        *out_clicked_axis = filamentAxis;
      }
    }
    if (gizmoHeld && IsMouseDragging(ImGuiMouseButton_Left)) {
      // SetMouseCursor(ImGuiMouseCursor_None);
      SetCursorPos(GetMousePos());
      *out_dragging = true;
    }
  }
  // Sort back-to-front: draw axes pointing away from the viewer first so
  // that axes pointing toward the viewer are painted on top.
  for (int pass = 0; pass < kNumAxes - 1; ++pass) {
    for (int i = 0; i < kNumAxes - 1 - pass; ++i) {
      if (proj[i].depth > proj[i + 1].depth) {
        ImSwap(proj[i], proj[i + 1]);
      }
    }
  }
  // Render hover circle.
  if (gizmoHeld || gizmoHovered) {
    dl->AddCircleFilled(center, radius + kCircleRadius, IM_COL32(255, 255, 255, 32));
  }
  // Blends axis color toward gray to simulate fading, keeping full opacity
  // so overlapping geometry appears as a single entity.
  auto blendCol = [](ImU32 c, float strength) {
    static constexpr float kG = 0.25f;
    ImVec4 f = ColorConvertU32ToFloat4(c);
    f.x = kG + (f.x - kG) * strength;
    f.y = kG + (f.y - kG) * strength;
    f.z = kG + (f.z - kG) * strength;
    f.w = 1.0f;
    return ColorConvertFloat4ToU32(f);
  };
  // Render axes.
  for (auto& axisProj : proj) {
    int ai = axisProj.index;
    int colorIndex = ai % 3;
    bool isNegative = ai >= 3;
    ImU32 col = kAxes[colorIndex].normal;
    bool negInFront = isNegative && axisProj.depth > kAlignThreshold;
    bool posInFront = !isNegative && axisProj.depth > kAlignThreshold;
    // Circle/line rendering.
    float t = ImClamp(axisProj.depth * 0.5f + 0.5f, 0.0f, 1.0f);
    if (isNegative) {
      if (negInFront) {
        dl->AddCircleFilled(axisProj.tip, kCircleRadius, col);
      } else {
        float outlineAlpha = 0.25f + t * 0.75f;
        float fillAlpha = 0.0f + t * 0.25f;
        dl->AddCircleFilled(axisProj.tip, kNegCircleRadius, blendCol(col, fillAlpha));
        dl->AddCircle(
            axisProj.tip, kNegCircleRadius, blendCol(col, outlineAlpha), 0, kNegCircleThickness);
      }
    } else {
      float alpha = 0.25f + t * 0.75f;
      ImU32 fadedCol = blendCol(col, alpha);
      dl->AddLine(center, axisProj.tip, fadedCol, 3.0f);
      dl->AddCircleFilled(axisProj.tip, kCircleRadius, fadedCol);
    }
    // Text rendering.
    char const* label = kLabels[ai];
    float const fontScale = GetIO().FontGlobalScale;
    float fontSize = font ? font->FontSize * fontScale : GetFontSize();
    ImVec2 textSize =
        font ? font->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, label) : CalcTextSize(label);
    ImVec2 textPos{axisProj.tip.x - textSize.x * 0.5f, axisProj.tip.y - textSize.y * 0.5f};
    if (isNegative) {
      if (negInFront) {
        int closestAxisRender = (closestAxisInteract + 3) % kNumAxes;
        bool isHovered = closestAxisRender == ai && gizmoHovered && !gizmoHeld;
        ImU32 textCol = (isHovered && !*out_dragging) ? IM_COL32_WHITE : IM_COL32_BLACK;
        dl->AddText(font, fontSize, textPos, textCol, label);
      } else if (closestAxisInteract == ai && (gizmoHovered || gizmoHeld) && !*out_dragging) {
        dl->AddText(font, fontSize, textPos, IM_COL32_WHITE, label);
      }
    } else {
      if (posInFront) {
        int closestAxisRender = (closestAxisInteract + 3) % kNumAxes;
        bool isHovered = closestAxisRender == ai && gizmoHovered && !gizmoHeld;
        ImU32 textCol = (isHovered && !*out_dragging) ? IM_COL32_WHITE : IM_COL32_BLACK;
        dl->AddText(font, fontSize, textPos, textCol, label);
      } else {
        bool isHovered = closestAxisInteract == ai && gizmoHovered && !gizmoHeld;
        ImU32 textCol = (isHovered && !*out_dragging) ? IM_COL32_WHITE : IM_COL32_BLACK;
        dl->AddText(font, fontSize, textPos, textCol, label);
      }
    }
  }
  return *out_clicked_axis != -1 && !*out_dragging;
}

bool ImGui::DragReal(
    char const* label,
    real* value,
    float v_speed,
    float v_min,
    float v_max,
    char const* format,
    float scale) {
  float v = static_cast<float>(*value) * scale;
  bool changed = DragFloat(label, &v, v_speed, v_min, v_max, format);
  if (changed) {
    *value = static_cast<real>(v) / scale;
  }
  return changed;
}

bool ImGui::DragReal3(
    char const* label,
    Real3& value,
    float v_speed,
    float v_min,
    float v_max,
    char const* format,
    float scale) {
  float v[3] = {
      static_cast<float>(value[0]) * scale,
      static_cast<float>(value[1]) * scale,
      static_cast<float>(value[2]) * scale};
  bool changed = DragFloat3(label, v, v_speed, v_min, v_max, format);
  if (changed) {
    value[0] = static_cast<real>(v[0]) / scale;
    value[1] = static_cast<real>(v[1]) / scale;
    value[2] = static_cast<real>(v[2]) / scale;
  }
  return changed;
}

bool ImGui::DragRealXYZ(
    char const* label,
    mochi::Real3& value,
    float v_speed,
    float v_min,
    float v_max,
    char const* format,
    ImGuiSliderFlags flags,
    float scale) {
  float v[3] = {
      static_cast<float>(value[0]) * scale,
      static_cast<float>(value[1]) * scale,
      static_cast<float>(value[2]) * scale};
  bool changed = DragFloatXYZ(label, v, v_speed, v_min, v_max, format, flags);
  if (changed) {
    value[0] = static_cast<real>(v[0]) / scale;
    value[1] = static_cast<real>(v[1]) / scale;
    value[2] = static_cast<real>(v[2]) / scale;
  }
  return changed;
}

bool ImGui::DragInertia(char const* label, Real6& value, float v_speed, float v_min, float v_max) {
  bool changed = false;
  PushID(label);
  float cellWidth = (CalcItemWidth() - 2 * GetStyle().ItemInnerSpacing.x) / 3.0f;

  // Extract values to float array for ImGui
  float v[6] = {
      static_cast<float>(value[0]),
      static_cast<float>(value[1]),
      static_cast<float>(value[2]),
      static_cast<float>(value[3]),
      static_cast<float>(value[4]),
      static_cast<float>(value[5])};

  // Component names for format strings
  constexpr char const* kNames[6] = {"Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"};

  // Build format strings for each component based on its value.
  // Note: GetUnitFormat uses a static buffer, so we must copy the result
  // immediately after each call before it gets overwritten.
  char fmt[6][64];
  char unitFmt[64];
  for (int i = 0; i < 6; ++i) {
    strncpy(unitFmt, ::GetUnitFormat(UnitFormat::InertiaTensor, v[i]), sizeof(unitFmt));
    snprintf(fmt[i], sizeof(fmt[i]), "%s= %s", kNames[i], unitFmt);
  }

  // Inertia tensor matrix layout:
  //   [Ixx] [Ixy] [Ixz]  ← Row 0: all editable (upper triangle)
  //   [Ixy] [Iyy] [Iyz]  ← Row 1: Ixy disabled (mirrored), Iyy/Iyz editable
  //   [Ixz] [Iyz] [Izz]  ← Row 2: Ixz/Iyz disabled (mirrored), Izz editable
  //
  // Real6 storage: [0]=Ixx, [1]=Ixy, [2]=Ixz, [3]=Iyy, [4]=Iyz, [5]=Izz
  constexpr int kMatrix[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
  constexpr bool kEditable[3][3] = {{true, true, true}, {false, true, true}, {false, false, true}};

  // Render 3x3 matrix
  for (int row = 0; row < 3; ++row) {
    PushID(row);
    for (int col = 0; col < 3; ++col) {
      if (col > 0) {
        SameLine(0, GetStyle().ItemInnerSpacing.x);
      }

      int idx = kMatrix[row][col];
      SetNextItemWidth(cellWidth);

      // Special case: the Iyz component at [1][2] uses the label parameter
      // for the ImGui ID instead of the generated "##" ID
      char id[8];
      char const* idPtr = id;
      if (row == 1 && col == 2) {
        idPtr = label;
      } else {
        snprintf(id, sizeof(id), "##%d%d", row, col);
      }

      if (kEditable[row][col]) {
        if (DragFloat(idPtr, &v[idx], v_speed, v_min, v_max, fmt[idx])) {
          value[idx] = static_cast<real>(v[idx]);
          changed = true;
        }
      } else {
        BeginDisabled();
        DragFloat(idPtr, &v[idx], v_speed, v_min, v_max, fmt[idx]);
        EndDisabled();
      }
    }
    PopID();
  }

  PopID();
  return changed;
}

bool ImGui::DragQuaternion(
    char const* label,
    float* x,
    float* y,
    float* z,
    float* w,
    QuaternionMode mode) {
  bool changed = false;
  if (mode == QuaternionMode::RPY) {
    float v_speed = 1.0f;
    // Nudge the clamp limits just below ±180° so a dragged value never lands exactly on the wrap
    // boundary.
    float v_min = -180.0f + 1e-6f;
    float v_max = 180.0f - 1e-6f;
    char const* format = ::GetUnitFormat(UnitFormat::Degrees);
    constexpr float kRadToDeg = 180.0f / std::numbers::pi_v<float>;
    constexpr float kDegToRad = std::numbers::pi_v<float> / 180.0f;
    constexpr float kHalfPi = std::numbers::pi_v<float> / 2.0f;

    // While a drag is in progress we keep the *edited* RPY as the source of truth instead of
    // round-tripping through the quaternion. Euler<->quaternion is many-to-one near pitch = ±90°
    // (gimbal lock), so re-deriving every frame makes the angles flip while dragging through the
    // singularity. The cache is scoped to the active edit (only one widget can be active at a time)
    // and keyed by the live widget id, so it is cleared when the drag ends and can never leak stale
    // angles to a different/recreated object that happens to reuse the same id.
    ImGuiContext& g = *GImGui;
    ImGuiID const groupId = GetID(label);
    static ImGuiID sEditId = 0;
    static float sEditRpyDeg[3] = {0.0f, 0.0f, 0.0f};

    float fv[3];
    if (sEditId == groupId) {
      // Mid-drag: use the angles the user is editing.
      fv[0] = sEditRpyDeg[0];
      fv[1] = sEditRpyDeg[1];
      fv[2] = sEditRpyDeg[2];
    } else {
      // Decompose the quaternion into roll-pitch-yaw (intrinsic Z-Y-X: yaw about Z, then pitch
      // about Y, then roll about X). Euler angles are a many-to-one encoding, so this returns one
      // valid representative.
      //
      // Near pitch = ±90° (gimbal lock) roll and yaw become coupled: the usual atan2 terms both
      // collapse to ~0, so the naive formula returns roll = yaw = 0, which is a *different*
      // rotation than the input. Instead we snap pitch to ±90° (also dodging asin's precision cliff
      // there), pin roll = 0, and fold the coupled rotation into yaw using well-conditioned terms.
      float const qx = *x, qy = *y, qz = *z, qw = *w;
      float sinp = 2.0f * (qw * qy - qz * qx);
      sinp = (sinp > 1.0f) ? 1.0f : ((sinp < -1.0f) ? -1.0f : sinp);
      float const cosp_sq = 1.0f - sinp * sinp; // cos²(pitch)
      float roll = 0.0f;
      float pitch = 0.0f;
      float yaw = 0.0f;
      constexpr float kGimbalEps = 1e-6f;
      if (cosp_sq > kGimbalEps) {
        roll = std::atan2(2.0f * (qw * qx + qy * qz), 1.0f - 2.0f * (qx * qx + qy * qy));
        pitch = std::asin(sinp);
        yaw = std::atan2(2.0f * (qw * qz + qx * qy), 1.0f - 2.0f * (qy * qy + qz * qz));
      } else {
        float const sgn = (sinp >= 0.0f) ? 1.0f : -1.0f;
        roll = 0.0f;
        pitch = sgn * kHalfPi;
        yaw = std::atan2(sgn * 2.0f * (qy * qz - qx * qw), 1.0f - 2.0f * (qx * qx + qz * qz));
      }
      fv[0] = roll * kRadToDeg;
      fv[1] = pitch * kRadToDeg;
      fv[2] = yaw * kRadToDeg;
    }

    // Snap values that round to zero (including IEEE negative zero) to +0 so the field never
    // displays "-0.0000". Below the "%.4f" rounding threshold the change is rotation-negligible.
    for (int i = 0; i < 3; ++i) {
      if (std::abs(fv[i]) < 5e-5f) {
        fv[i] = 0.0f;
      }
    }

    changed = DragFloatXYZ(label, fv, v_speed, v_min, v_max, format);

    // Detect whether this widget is the one being actively dragged. Checked *after* the call so
    // ImGui has set the active id for a freshly-started drag. The component drag ids mirror how
    // DragFloatXYZ builds its id stack: PushID(label) -> PushID(i) -> DragFloat("##v").
    bool active = false;
    PushID(label);
    for (int i = 0; i < 3; ++i) {
      PushID(i);
      active = active || (g.ActiveId == GetID("##v"));
      PopID();
    }
    PopID();

    if (active) {
      sEditId = groupId;
      sEditRpyDeg[0] = fv[0];
      sEditRpyDeg[1] = fv[1];
      sEditRpyDeg[2] = fv[2];
    } else if (sEditId == groupId) {
      sEditId = 0; // Edit finished; revert to deriving from the quaternion.
    }

    if (changed) {
      // Convert the edited RPY back to a quaternion.
      float r = fv[0] * kDegToRad;
      float p = fv[1] * kDegToRad;
      float ya = fv[2] * kDegToRad;
      float cr = std::cos(r * 0.5f), sr = std::sin(r * 0.5f);
      float cp = std::cos(p * 0.5f), sp = std::sin(p * 0.5f);
      float cy = std::cos(ya * 0.5f), sy = std::sin(ya * 0.5f);
      *w = cr * cp * cy + sr * sp * sy;
      *x = sr * cp * cy - cr * sp * sy;
      *y = cr * sp * cy + sr * cp * sy;
      *z = cr * cp * sy - sr * sp * cy;
    }
  } else {
    float v_speed = 0.01f;
    float v_min = -1.0f;
    float v_max = 1.0f;
    char const* format = "%.4f";
    float fv[4] = {*x, *y, *z, *w};
    changed = DragFloat4(label, fv, v_speed, v_min, v_max, format);
    if (changed) {
      *x = fv[0];
      *y = fv[1];
      *z = fv[2];
      *w = fv[3];
    }
  }
  return changed;
}

bool ImGui::DragQuaternion(char const* label, Quaternion& value, QuaternionMode mode) {
  Real4 v = value.ToReal4();
  auto x = static_cast<float>(v[0]);
  auto y = static_cast<float>(v[1]);
  auto z = static_cast<float>(v[2]);
  auto w = static_cast<float>(v[3]);
  bool changed = DragQuaternion(label, &x, &y, &z, &w, mode);
  if (changed) {
    value = Quaternion(
        static_cast<real>(x), static_cast<real>(y), static_cast<real>(z), static_cast<real>(w));
  }
  return changed;
}

bool ImGui::DragTransformRT(
    char const* label,
    mochi::Quaternion& rotation,
    mochi::Real3& translation) {
  PushID(label);
  bool changed = false;
  changed |=
      DragRealXYZ("Translation", translation, 0.01f, 0, 0, ::GetUnitFormat(UnitFormat::Length));
  changed |= DragQuaternion("Rotation", rotation);
  PopID();
  return changed;
}

bool ImGui::DragTransformRT(char const* label, TransformRT& value) {
  PushID(label);
  bool changed = false;
  Quaternion rotation = value.GetRotation();
  Real3 translation = value.GetTranslation();
  if (DragRealXYZ("Translation", translation, 0.01f, 0, 0, ::GetUnitFormat(UnitFormat::Length))) {
    value.SetTranslation(translation);
    changed = true;
  }
  if (DragQuaternion("Rotation", rotation)) {
    value.SetRotation(rotation);
    changed = true;
  }
  PopID();
  return changed;
}

bool ImGui::ComboArticulatedJointType(
    char const* label,
    ArticulatedJointType& type,
    ArticulatedJointTypeFilter filter) {
  static constexpr char const* kAllItems[] = {
      "Free", "Prismatic", "Revolute", "Spherical", "Hard", "Cycle"};
  static constexpr ArticulatedJointType kAllValues[] = {
      ArticulatedJointType::Free,
      ArticulatedJointType::Prismatic,
      ArticulatedJointType::Revolute,
      ArticulatedJointType::Spherical,
      ArticulatedJointType::Hard,
      ArticulatedJointType::Cycle};

  static constexpr char const* kNoFreeItems[] = {"Prismatic", "Revolute", "Spherical", "Hard"};
  static constexpr ArticulatedJointType kNoFreeValues[] = {
      ArticulatedJointType::Prismatic,
      ArticulatedJointType::Revolute,
      ArticulatedJointType::Spherical,
      ArticulatedJointType::Hard};

  static constexpr char const* kHardFreeItems[] = {"Free", "Hard"};
  static constexpr ArticulatedJointType kHardFreeValues[] = {
      ArticulatedJointType::Free, ArticulatedJointType::Hard};

  char const* const* items = kAllItems;
  ArticulatedJointType const* values = kAllValues;
  int count = IM_ARRAYSIZE(kAllItems);
  switch (filter) {
    case ArticulatedJointTypeFilter::All:
      items = kAllItems;
      values = kAllValues;
      count = IM_ARRAYSIZE(kAllItems);
      break;
    case ArticulatedJointTypeFilter::NoFreeCycle:
      items = kNoFreeItems;
      values = kNoFreeValues;
      count = IM_ARRAYSIZE(kNoFreeItems);
      break;
    case ArticulatedJointTypeFilter::HardFreeOnly:
      items = kHardFreeItems;
      values = kHardFreeValues;
      count = IM_ARRAYSIZE(kHardFreeItems);
      break;
  }

  // Select the current type's index, defaulting to 0 if it is not offered by
  // the active filter.
  int currentItem = 0;
  for (int i = 0; i < count; ++i) {
    if (values[i] == type) {
      currentItem = i;
      break;
    }
  }
  bool changed = Combo(label, &currentItem, items, count);
  if (changed) {
    type = values[currentItem];
  }
  return changed;
}

void ImGui::IconInputPrefix(char const* icon, char const* tooltip) {
  float const checkboxWidth = GetFrameHeight();
  float const innerWidth = CalcItemWidth() - checkboxWidth - GetStyle().ItemInnerSpacing.x;
  float const cellStartX = GetCursorPosX();

  // Vertically center the glyph within the row (align to frame padding like a framed widget) and
  // horizontally center it in the checkbox-sized cell so it occupies the same slot a checkbox
  // would.
  AlignTextToFramePadding();
  float const glyphWidth = CalcTextSize(icon).x;
  SetCursorPosX(cellStartX + (checkboxWidth - glyphWidth) * 0.5f);
  TextUnformatted(icon);
  if (tooltip != nullptr && IsItemHovered()) {
    SetTooltip("%s", tooltip);
  }

  // Position the following item's field to match the value column of checkbox-fronted rows.
  SameLine(0.0f, 0.0f);
  SetCursorPosX(cellStartX + checkboxWidth + GetStyle().ItemInnerSpacing.x);
  SetNextItemWidth(innerWidth);
}

bool ImGui::DragOptionalReal(
    char const* label,
    std::optional<real>& value,
    float v_speed,
    float v_min,
    float v_max,
    char const* format,
    float scale) {
  bool changed = false;
  bool hasValue = value.has_value();

  PushID(label);

  float const fullWidth = CalcItemWidth();
  float const checkboxWidth = GetFrameHeight();
  float const innerWidth = fullWidth - checkboxWidth - GetStyle().ItemInnerSpacing.x;

  if (Checkbox("##toggle", &hasValue)) {
    if (hasValue) {
      value = 0_r;
    } else {
      value = std::nullopt;
    }
    changed = true;
  }

  SameLine(0, GetStyle().ItemInnerSpacing.x);
  SetNextItemWidth(innerWidth);
  BeginDisabled(!hasValue);
  real v = value.value_or(0_r);
  if (DragReal(label, &v, v_speed, v_min, v_max, format, scale)) {
    value = v;
    changed = true;
  }
  EndDisabled();

  PopID();
  return changed;
}

bool ImGui::DragOptionalRealXYZ(
    char const* label,
    std::optional<Real3>& value,
    float v_speed,
    float v_min,
    float v_max,
    char const* format,
    ImGuiSliderFlags flags,
    float scale) {
  bool changed = false;
  bool hasValue = value.has_value();

  PushID(label);

  float const fullWidth = CalcItemWidth();
  float const checkboxWidth = GetFrameHeight();
  float const innerWidth = fullWidth - checkboxWidth - GetStyle().ItemInnerSpacing.x;

  if (Checkbox("##toggle", &hasValue)) {
    if (hasValue) {
      value = Real3{0_r, 0_r, 0_r};
    } else {
      value = std::nullopt;
    }
    changed = true;
  }

  SameLine(0, GetStyle().ItemInnerSpacing.x);
  SetNextItemWidth(innerWidth);
  BeginDisabled(!hasValue);
  Real3 v = value.value_or(Real3{0_r, 0_r, 0_r});
  if (DragRealXYZ(label, v, v_speed, v_min, v_max, format, flags, scale)) {
    value = v;
    changed = true;
  }
  EndDisabled();

  PopID();
  return changed;
}

bool ImGui::ColorSwatchButton(char const* label, ImU32 swatch_color, bool grayed_out, float width) {
  ImGuiContext& g = *GImGui;
  ImGuiWindow* window = g.CurrentWindow;
  if (window->SkipItems) {
    return false;
  }

  ImGuiStyle const& style = g.Style;
  float const line_height = GetFrameHeight();
  float const swatch_width = line_height * 0.25f;

  ImVec2 button_size(width > 0.0f ? width : 0.0f, 0.0f);
  ImVec2 button_pos = window->DC.CursorPos;

  bool pressed = Button(label, button_size);

  ImU32 display_color = swatch_color;
  if (grayed_out) {
    display_color = IM_COL32(128, 128, 128, 128);
  }

  ImVec2 swatch_min(button_pos.x, button_pos.y);
  ImVec2 swatch_max(button_pos.x + swatch_width, button_pos.y + line_height);
  window->DrawList->AddRectFilled(
      swatch_min, swatch_max, display_color, style.FrameRounding, ImDrawFlags_RoundCornersLeft);

  return pressed;
}

void ImGui::HoverableSeparatorText(char const* label) {
  ImGui::SeparatorText(label);
  bool const hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup);
  if (hovered) {
    ImVec2 const min = ImGui::GetItemRectMin();
    ImVec2 const max = ImGui::GetItemRectMax();
    ImU32 const col = ImGui::GetColorU32(ImGuiCol_HeaderHovered);
    float const rounding = ImGui::GetStyle().FrameRounding;
    ImGui::GetWindowDrawList()->AddRect(min, max, col, rounding, 0, 2);
  }
}

// SimpleReflection

bool ImGui::BoolCombo(char const* label, bool* value) {
  bool changed = false;
  // A combo renders its preview text top-left with no style hook for alignment, which reads badly
  // next to the drag widgets (ImGui centers their value text). Take over the preview and center it
  // ourselves.
  float const frameWidth = CalcItemWidth();
  if (BeginCombo(label, "", ImGuiComboFlags_CustomPreview)) {
    for (bool const option : {true, false}) {
      bool const selected = (*value == option);
      if (Selectable(option ? "True" : "False", selected) && !selected) {
        *value = option;
        changed = true;
      }
      if (selected) {
        SetItemDefaultFocus();
      }
    }
    EndCombo();
  }
  ImGuiLastItemData const comboItem = GetCurrentContext()->LastItemData;
  if (BeginComboPreview()) {
    char const* const text = *value ? "True" : "False";
    // Center over the whole frame, not the preview rect: the latter stops at the arrow button,
    // which would leave the text visibly left of center. Falls back to left-aligned if the text is
    // ever too wide.
    ImRect const& preview = GetCurrentContext()->ComboPreviewData.PreviewRect;
    float const indent = ImMax((frameWidth - CalcTextSize(text).x) * 0.5f, 0.0f);
    SetCursorScreenPos({preview.Min.x + indent, GetCursorScreenPos().y});
    TextUnformatted(text);
    EndComboPreview();
    // The preview text was submitted as an item, which would leave it, rather than the combo, as
    // the target of the caller's IsItemHovered (tooltips) -- so hovering only the text would count.
    // Put the combo back.
    GetCurrentContext()->LastItemData = comboItem;
  }
  return changed;
}

void ImGui::ItemTooltipWrapped(char const* text) {
  if (!ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
    return;
  }
  ImGui::BeginTooltip();
  // Same wrap width as the model editor's modifier tooltips.
  ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
  ImGui::TextUnformatted(text);
  ImGui::PopTextWrapPos();
  ImGui::EndTooltip();
}

static char const* CamelCaseToDisplayName(char const* name, char* buf, int bufSize) {
  // Serialization names sometimes carry a leading underscore to sort a key first in JSON (e.g.
  // "_comment"); it is not part of the human-readable name.
  while (*name == '_') {
    ++name;
  }
  int j = 0;
  for (int i = 0; name[i] != '\0' && j < bufSize - 1; ++i) {
    char c = name[i];
    if (i == 0) {
      buf[j++] = (c >= 'a' && c <= 'z') ? static_cast<char>(c - 32) : c;
    } else {
      bool isUpper = (c >= 'A' && c <= 'Z');
      bool prevIsLower = (name[i - 1] >= 'a' && name[i - 1] <= 'z');
      bool nextIsLower = (name[i + 1] != '\0' && name[i + 1] >= 'a' && name[i + 1] <= 'z');
      bool prevIsUpper = (name[i - 1] >= 'A' && name[i - 1] <= 'Z');
      if (isUpper && (prevIsLower || (prevIsUpper && nextIsLower))) {
        if (j < bufSize - 2) {
          buf[j++] = ' ';
        }
      }
      buf[j++] = c;
    }
  }
  buf[j] = '\0';
  return buf;
}

// Helper function to build a format string with units using fixed-point notation.
// This is used for types where adaptive formatting doesn't make sense (e.g., Real3
// where components may vary widely).
static char const*
BuildFormatWithUnits(char const* baseFmt, char const* units, char* buf, int bufSize) {
  if (units == nullptr || units[0] == '\0') {
    return baseFmt;
  }
  snprintf(buf, bufSize, "%s %s", baseFmt, units);
  return buf;
}

static char const* GetDisplayLabel(SReflect::FieldTypeInfo const& field, char* buf, int bufSize) {
  auto const* dn = field.GetAttribute<SReflect::Attribute_DisplayName>();
  if (dn != nullptr) {
    return dn->_displayName.c_str();
  }
  return CamelCaseToDisplayName(field._name, buf, bufSize);
}

static bool SimpleReflectionField(
    char const* label,
    SReflect::TypeInfo const& typeInfo,
    void* fieldPtr,
    SReflect::FieldTypeInfo const* fieldInfo);

static bool SimpleReflectionStruct_Internal(
    SReflect::StructTypeInfo const& structInfo,
    void* structPtr,
    SimpleReflectionWidgetOverride overrideFn,
    SimpleReflectionReadOnlyPredicate readOnlyFn = nullptr);

// Default-constructed instance of `typeInfo`, used to render the widget an unset optional would
// get. Built once per type and kept for the process: the unset branch is immediate-mode, so it
// re-runs every frame the field is visible, and the instance is never mutated (the widget is
// disabled). Returns null if the type cannot be constructed.
static void* GetUnsetOptionalPreview(SReflect::TypeInfo const& typeInfo) {
  static std::unordered_map<SReflect::TypeInfo const*, void*> previews;
  auto const [it, inserted] = previews.try_emplace(&typeInfo, nullptr);
  if (inserted) {
    it->second = typeInfo.New();
  }
  return it->second;
}

template <typename T>
static bool
DragScalarField(char const* label, void* fieldPtr, char const* units, float min, float max) {
  T* value = static_cast<T*>(fieldPtr);
  char fmtBuf[64];
  char const* fmt =
      BuildAdaptiveFormatWithUnits(static_cast<float>(*value), units, 4, fmtBuf, sizeof(fmtBuf));
  T const scalarMin = static_cast<T>(min);
  T const scalarMax = static_cast<T>(max);
  ImGuiDataType const dataType = [] {
    if constexpr (std::is_same_v<T, float>) {
      return ImGuiDataType_Float;
    } else {
      static_assert(std::is_same_v<T, double>);
      return ImGuiDataType_Double;
    }
  }();
  return DragScalarN(label, dataType, value, 1, 0.01f, &scalarMin, &scalarMax, fmt);
}

static bool SimpleReflectionField(
    char const* label,
    SReflect::TypeInfo const& typeInfo,
    void* fieldPtr,
    SReflect::FieldTypeInfo const* fieldInfo) {
  bool changed = false;

  char const* units = nullptr;
  float floatMin = 0.0f;
  float floatMax = 0.0f;
  int intMin = 0;
  int intMax = 0;

  if (fieldInfo != nullptr) {
    if (auto const* u = fieldInfo->GetAttribute<SReflect::Attribute_Units>()) {
      units = u->_units.c_str();
    }
    if (auto const* r = fieldInfo->GetAttribute<SReflect::Attribute_FloatRange>()) {
      floatMin = static_cast<float>(r->_min);
      floatMax = static_cast<float>(r->_max);
    }
    if (auto const* r = fieldInfo->GetAttribute<SReflect::Attribute_IntRange>()) {
      intMin = static_cast<int>(r->_min);
      intMax = static_cast<int>(r->_max);
    }
  }

  // Colors are opt-in via the Color attribute rather than inferred from shape: a 3- or 4-element
  // float field is just as likely to be a quaternion, a translation or a pair of extents.
  if (fieldInfo != nullptr && fieldInfo->HasAttribute<SReflect::Attribute_Color>() &&
      typeInfo._coreType == SReflect::CoreType::CT_array) {
    auto const& arrInfo = static_cast<SReflect::ArrayTypeInfo const&>(typeInfo);
    size_t const count = arrInfo.GetNumElements(fieldPtr);
    bool const isDouble = arrInfo._innerTypeInfo->_coreType == SReflect::CoreType::CT_double;
    bool const isFloat = arrInfo._innerTypeInfo->_coreType == SReflect::CoreType::CT_float;
    if ((count == 3 || count == 4) && (isFloat || isDouble)) {
      // Round-trip through a float[4] so this works for both real == float and real == double.
      float rgba[4] = {0.0f, 0.0f, 0.0f, 1.0f};
      for (size_t i = 0; i < count; ++i) {
        void* elem = arrInfo.GetElement(fieldPtr, i);
        rgba[i] =
            isFloat ? *static_cast<float*>(elem) : static_cast<float>(*static_cast<double*>(elem));
      }
      changed = count == 4 ? ColorEdit4(label, rgba, ImGuiColorEditFlags_AlphaPreview)
                           : ColorEdit3(label, rgba);
      if (changed) {
        for (size_t i = 0; i < count; ++i) {
          void* elem = arrInfo.GetElement(fieldPtr, i);
          if (isFloat) {
            *static_cast<float*>(elem) = rgba[i];
          } else {
            *static_cast<double*>(elem) = static_cast<double>(rgba[i]);
          }
        }
      }
      return changed;
    }
  }

  if (typeInfo._typeId == SReflect::GetTypeId<Real3>()) {
    // Real3: use fixed-point formatting since components can vary widely
    // and there's no single representative value
    char fmtBuf[64];
    char const* fmt = BuildFormatWithUnits("%.4f", units, fmtBuf, sizeof(fmtBuf));
    changed = DragRealXYZ(label, *static_cast<Real3*>(fieldPtr), 0.01f, floatMin, floatMax, fmt);
  } else if (typeInfo._typeId == SReflect::GetTypeId<Real6>()) {
    changed = DragInertia(label, *static_cast<Real6*>(fieldPtr), 0.01f, floatMin, floatMax);
  } else if (typeInfo._typeId == SReflect::GetTypeId<Quaternion>()) {
    changed = DragQuaternion(label, *static_cast<Quaternion*>(fieldPtr));
  } else if (typeInfo._typeId == SReflect::GetTypeId<TransformRT>()) {
    changed = DragTransformRT(label, *static_cast<TransformRT*>(fieldPtr));
  } else {
    switch (typeInfo._coreType) {
      case SReflect::CoreType::CT_float: {
        changed = DragScalarField<float>(label, fieldPtr, units, floatMin, floatMax);
        break;
      }
      case SReflect::CoreType::CT_double: {
        changed = DragScalarField<double>(label, fieldPtr, units, floatMin, floatMax);
        break;
      }
      case SReflect::CoreType::CT_bool: {
        changed = BoolCombo(label, static_cast<bool*>(fieldPtr));
        break;
      }
      case SReflect::CoreType::CT_int32: {
        changed = DragInt(label, static_cast<int*>(fieldPtr), 1.0f, intMin, intMax);
        break;
      }
      case SReflect::CoreType::CT_string: {
        auto const& strInfo = static_cast<SReflect::StringTypeInfo const&>(typeInfo);
        std::string_view sv = strInfo.GetString(fieldPtr);
        char buf[256];
        size_t len = sv.size() < sizeof(buf) - 1 ? sv.size() : sizeof(buf) - 1;
        memcpy(buf, sv.data(), len);
        buf[len] = '\0';
        if (InputText(label, buf, sizeof(buf))) {
          strInfo.SetString(fieldPtr, buf);
          changed = true;
        }
        break;
      }
      case SReflect::CoreType::CT_enum: {
        auto const& enumInfo = static_cast<SReflect::EnumTypeInfo const&>(typeInfo);
        changed = SimpleReflectionEnum(label, enumInfo, fieldPtr);
        break;
      }
      case SReflect::CoreType::CT_optional: {
        auto const& optInfo = static_cast<SReflect::OptionalTypeInfo const&>(typeInfo);
        void* innerPtr = optInfo.GetOptionalValue(fieldPtr);
        bool hasValue = (innerPtr != nullptr);
        PushID(label);
        float const fullWidth = CalcItemWidth();
        float const checkboxWidth = GetFrameHeight();
        float const innerWidth = fullWidth - checkboxWidth - GetStyle().ItemInnerSpacing.x;
        bool toggled = Checkbox("##toggle", &hasValue);
        if (toggled) {
          if (hasValue) {
            innerPtr = optInfo.EnsureOptionalValue(fieldPtr);
          } else {
            optInfo.SetOptionalValue(nullptr, fieldPtr);
            innerPtr = nullptr;
          }
          changed = true;
        }
        SameLine(0, GetStyle().ItemInnerSpacing.x);
        SetNextItemWidth(innerWidth);
        if (innerPtr != nullptr) {
          changed |= SimpleReflectionField(label, *optInfo._innerTypeInfo, innerPtr, nullptr);
        } else {
          // Show the widget the value would get, greyed out, so an unset optional still reads as
          // the type it holds. Needs an instance to point at.
          BeginDisabled();
          if (void* preview = GetUnsetOptionalPreview(*optInfo._innerTypeInfo)) {
            SimpleReflectionField(label, *optInfo._innerTypeInfo, preview, nullptr);
          } else {
            TextDisabled("%s (unset)", label);
          }
          EndDisabled();
        }
        PopID();
        break;
      }
      case SReflect::CoreType::CT_struct: {
        auto const& inner = static_cast<SReflect::StructTypeInfo const&>(typeInfo);
        // Framed header rather than a plain tree node so nested structs read as sections.
        // CollapsingHeader implies NoTreePushOnOpen, so indent the body explicitly.
        if (CollapsingHeader(label)) {
          Indent();
          changed = SimpleReflectionStruct_Internal(inner, fieldPtr, nullptr);
          Unindent();
        }
        break;
      }
      case SReflect::CoreType::CT_array: {
        auto const& arrInfo = static_cast<SReflect::ArrayTypeInfo const&>(typeInfo);
        size_t const count = arrInfo.GetNumElements(fieldPtr);
        SReflect::CoreType const elemType = arrInfo._innerTypeInfo->_coreType;

        // Render small scalar arrays (e.g. Int3) inline as a single multi-component drag widget
        // rather than an expandable tree node. NdArray storage is contiguous, so the first element
        // pointer doubles as the base of the component array.
        bool const isScalarElem = elemType == SReflect::CoreType::CT_int32 ||
            elemType == SReflect::CoreType::CT_float || elemType == SReflect::CoreType::CT_double;
        if (isScalarElem && count >= 1 && count <= 4) {
          void* const basePtr = arrInfo.GetElement(fieldPtr, 0);
          int const components = static_cast<int>(count);
          char fmtBuf[64];
          if (elemType == SReflect::CoreType::CT_int32) {
            char const* fmt = BuildFormatWithUnits("%d", units, fmtBuf, sizeof(fmtBuf));
            int lo = intMin, hi = intMax;
            changed =
                DragScalarN(label, ImGuiDataType_S32, basePtr, components, 1.0f, &lo, &hi, fmt);
          } else if (elemType == SReflect::CoreType::CT_float) {
            char const* fmt = BuildFormatWithUnits("%.4f", units, fmtBuf, sizeof(fmtBuf));
            float lo = floatMin, hi = floatMax;
            changed =
                DragScalarN(label, ImGuiDataType_Float, basePtr, components, 0.01f, &lo, &hi, fmt);
          } else {
            char const* fmt = BuildFormatWithUnits("%.4f", units, fmtBuf, sizeof(fmtBuf));
            double lo = floatMin, hi = floatMax;
            changed =
                DragScalarN(label, ImGuiDataType_Double, basePtr, components, 0.01f, &lo, &hi, fmt);
          }
          break;
        }

        if (TreeNode(label)) {
          for (size_t i = 0; i < count; ++i) {
            char elemLabel[64];
            snprintf(elemLabel, sizeof(elemLabel), "[%zu]", i);
            void* elemPtr = arrInfo.GetElement(fieldPtr, i);
            changed |= SimpleReflectionField(elemLabel, *arrInfo._innerTypeInfo, elemPtr, nullptr);
          }
          TreePop();
        }
        break;
      }
      default:
        TextDisabled("%s (unsupported type)", label);
        break;
    }
  }
  return changed;
}

static bool SimpleReflectionStruct_Internal(
    SReflect::StructTypeInfo const& structInfo,
    void* structPtr,
    SimpleReflectionWidgetOverride overrideFn,
    SimpleReflectionReadOnlyPredicate readOnlyFn) {
  bool changed = false;
  for (SReflect::FieldTypeInfo const* field : structInfo._fields) {
    if (field->HasAttribute<SReflect::Attribute_HideFromEditor>()) {
      continue;
    }

    char labelBuf[128];
    char const* label = GetDisplayLabel(*field, labelBuf, sizeof(labelBuf));
    void* fieldPtr = field->GetFieldPtr(structPtr);

    // ImGui derives widget IDs from the label, so two sibling structs with a like-named field (or
    // like-named nested sections) would share state -- dragging one would drive both. Field names
    // are unique within a struct, so scoping by them makes every widget in the tree unique.
    PushID(field->_name);

    bool const isReadOnly = field->HasAttribute<SReflect::Attribute_ReadOnly>() ||
        (readOnlyFn != nullptr && readOnlyFn(*field));
    if (isReadOnly) {
      BeginDisabled();
    }

    bool handled = false;
    if (overrideFn != nullptr) {
      handled = overrideFn(label, *field, fieldPtr);
    }
    if (!handled) {
      changed |= SimpleReflectionField(label, *field->_innerTypeInfo, fieldPtr, field);
    }

    if (isReadOnly) {
      EndDisabled();
    }

    if (field->HasAttribute<SReflect::Attribute_Description>()) {
      ItemTooltipWrapped(
          field->GetAttribute<SReflect::Attribute_Description>()->_description.c_str());
    }

    PopID();
  }
  return changed;
}

bool ImGui::SimpleReflectionEnum(
    char const* label,
    SReflect::EnumTypeInfo const& enumInfo,
    void* enumPtr) {
  bool changed = false;
  uint64_t currentVal = enumInfo.GetValue(enumPtr);
  int currentIdx = 0;
  for (int i = 0; i < static_cast<int>(enumInfo._items.size()); ++i) {
    if (enumInfo._items[i]._value == currentVal) {
      currentIdx = i;
      break;
    }
  }
  if (BeginCombo(label, enumInfo._items[currentIdx]._name)) {
    for (int i = 0; i < static_cast<int>(enumInfo._items.size()); ++i) {
      bool selected = (i == currentIdx);
      if (Selectable(enumInfo._items[i]._name, selected)) {
        enumInfo.SetValue(enumPtr, enumInfo._items[i]._value);
        changed = true;
      }
      if (selected) {
        SetItemDefaultFocus();
      }
    }
    EndCombo();
  }
  return changed;
}

bool ImGui::SimpleReflectionStruct(
    SReflect::StructTypeInfo const& typeInfo,
    void* structPtr,
    SimpleReflectionWidgetOverride overrideFn,
    SimpleReflectionReadOnlyPredicate readOnlyFn) {
  return SimpleReflectionStruct_Internal(typeInfo, structPtr, overrideFn, readOnlyFn);
}

bool ImGui::TextButton(char const* label) {
  ImGuiWindow* window = GetCurrentWindow();
  if (window->SkipItems) {
    return false;
  }
  ImGuiID const id = window->GetID(label);
  ImVec2 const textSize = CalcTextSize(label);
  ImVec2 const pos = window->DC.CursorPos;
  ImRect const bb(pos, ImVec2(pos.x + textSize.x, pos.y + textSize.y));
  ItemSize(textSize);
  if (!ItemAdd(bb, id)) {
    return false;
  }
  bool hovered = false;
  bool held = false;
  bool const pressed = ButtonBehavior(bb, id, &hovered, &held);
  ImU32 col = GetColorU32(ImGuiCol_Text);
  if (held) {
    col = GetColorU32(ImGuiCol_ButtonActive);
  } else if (hovered) {
    col = GetColorU32(ImGuiCol_ButtonHovered);
  }
  window->DrawList->AddText(pos, col, label);
  return pressed;
}

bool ImGui::ButtonColored(char const* label, ImVec4 const& color, ImVec2 const& size) {
  ImVec4 colorHover = ImLerp({1, 1, 1, color.w}, color, 0.8f);
  ImVec4 colorPress = ImLerp({0, 0, 0, color.w}, color, 0.8f);
  ImGui::PushStyleColor(ImGuiCol_Button, color);
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, colorHover);
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, colorPress);
  bool ret = ImGui::Button(label, size);
  ImGui::PopStyleColor(3);
  return ret;
}

bool ImGui::SplitButtonSegment(char const* label, ImVec2 size, ImDrawFlags corners, bool active) {
  ImGuiWindow* window = GetCurrentWindow();
  if (window->SkipItems) {
    return false;
  }
  ImGuiID const id = window->GetID(label);
  ImVec2 const pos = window->DC.CursorPos;
  ImRect const bb(pos, {pos.x + size.x, pos.y + size.y});
  ItemSize(size, GetStyle().FramePadding.y);
  if (!ItemAdd(bb, id)) {
    return false;
  }
  bool hovered = false;
  bool held = false;
  bool const pressed = ButtonBehavior(bb, id, &hovered, &held);
  ImGuiCol const colId = (held && hovered) ? ImGuiCol_ButtonActive
      : hovered                            ? ImGuiCol_ButtonHovered
      : active                             ? ImGuiCol_ButtonActive
                                           : ImGuiCol_Button;
  // Round to half the height so the outer side forms a semicircle, matching the pill toolbar
  // buttons.
  window->DrawList->AddRectFilled(bb.Min, bb.Max, GetColorU32(colId), size.y * 0.5f, corners);
  RenderTextClipped(bb.Min, bb.Max, label, nullptr, nullptr, ImVec2(0.5f, 0.5f));
  return pressed;
}

bool ImGui::InertialProperties(
    std::optional<mochi::real>& density,
    std::optional<mochi::real>& mass,
    std::optional<mochi::Real3>& centerOfMass,
    std::optional<mochi::Real6>& momentOfInertia) {
  bool changed = false;
  ImGui::SeparatorText("Inertial Properties");
  bool const hasDensity = density.has_value();
  bool const hasMass = mass.has_value();
  bool const hasCOM = centerOfMass.has_value();
  bool const hasMOI = momentOfInertia.has_value();
  bool const allNull = !hasDensity && !hasMass && !hasCOM && !hasMOI;
  bool const densityOnly = hasDensity && !hasMass && !hasCOM && !hasMOI;
  bool const massOnly = !hasDensity && hasMass && !hasCOM && !hasMOI;
  bool const explicitMass = !hasDensity && hasMass && hasCOM && hasMOI;
  bool const explicitDensity = hasDensity && !hasMass && hasCOM && hasMOI;
  enum InertialSetup {
    Default = 0,
    DensityOnly = 1,
    MassOnly = 2,
    MassComMoi = 3,
    DensityComMoi = 4,
  };
  constexpr char const* setupStrs[] = {
      "Default",
      "Density Only",
      "Mass Only",
      "Mass, COM, MOI",
      "Density, COM, MOI",
  };
  int setup = 0;
  if (allNull) {
    setup = Default;
  } else if (densityOnly) {
    setup = DensityOnly;
  } else if (massOnly) {
    setup = MassOnly;
  } else if (explicitMass) {
    setup = MassComMoi;
  } else if (explicitDensity) {
    setup = DensityComMoi;
  } else {
    setup = 0;
    density = {};
    mass = {};
    centerOfMass = {};
    momentOfInertia = {};
    changed |= true;
  }
  if (ImGui::Combo("Setup", &setup, setupStrs, IM_ARRAYSIZE(setupStrs))) {
    if (setup == Default) {
      density = {};
      mass = {};
      centerOfMass = {};
      momentOfInertia = {};
    } else if (setup == DensityOnly) {
      if (!hasDensity) {
        density = mochi::kDefaultDensity;
      }
      mass = {};
      centerOfMass = {};
      momentOfInertia = {};
    } else if (setup == MassOnly) {
      if (!hasMass) {
        mass = 1.0;
      }
      density = {};
      centerOfMass = {};
      momentOfInertia = {};
    } else if (setup == MassComMoi) {
      if (!hasMass) {
        mass = 1.0;
      }
      if (!hasCOM) {
        centerOfMass = mochi::Real3{};
      }
      if (!hasMOI) {
        momentOfInertia = mochi::Real6{};
      }
      density = {};
    } else if (setup == DensityComMoi) {
      if (!hasDensity) {
        density = mochi::kDefaultDensity;
      }
      if (!hasCOM) {
        centerOfMass = mochi::Real3{};
      }
      if (!hasMOI) {
        momentOfInertia = mochi::Real6{};
      }
      mass = {};
    }
    changed |= true;
  }
  if (setup == DensityOnly) {
    changed |= ImGui::DragOptionalReal(
        "Density", density, 0.01, 0, 0, GetUnitFormat(UnitFormat::Density, density.value_or(0.0f)));
  } else if (setup == MassOnly) {
    changed |= ImGui::DragOptionalReal(
        "Mass", mass, 0.01, 0, 1000, GetUnitFormat(UnitFormat::Mass, mass.value_or(0.0f)));
  } else if (setup == MassComMoi) {
    changed |= ImGui::DragReal(
        "Mass", &*mass, 0.01, 0, 1000, GetUnitFormat(UnitFormat::Mass, static_cast<float>(*mass)));
    changed |= ImGui::DragRealXYZ(
        "Center of Mass", *centerOfMass, 0.01, 0, 0, GetUnitFormat(UnitFormat::Length));
    changed |= ImGui::DragInertia("Inertia Tensor (kg\xC2\xB7m\xC2\xB2)", *momentOfInertia);
  } else if (setup == DensityComMoi) {
    changed |= ImGui::DragReal(
        "Density",
        &*density,
        0.01,
        0,
        0,
        GetUnitFormat(UnitFormat::Density, static_cast<float>(*density)));
    changed |= ImGui::DragRealXYZ(
        "Center of Mass", *centerOfMass, 0.01, 0, 0, GetUnitFormat(UnitFormat::Length));
    changed |= ImGui::DragInertia("Inertia Tensor (kg\xC2\xB7m\xC2\xB2)", *momentOfInertia);
  }
  return changed;
}

bool ImGui::ModelEditor(
    char const* label,
    SuperDexStudio* studio,
    AssetType assetType,
    mochi::DynamicString& path,
    mochi::Real3& scale,
    mochi::Quaternion& rotation,
    mochi::Real3& translation,
    mochi::Real3* otherScale,
    mochi::Quaternion* otherRotation,
    mochi::Real3* otherTranslation,
    AssetManager const& assetManager,
    bool acceptDragDropPayload,
    bool& modelChanged) {
  bool changed = false;
  ImGui::SeparatorText(label);
  ImGui::PushID(label);
  static bool linkModelTransforms = false;
  if (ImGui::AssetSlot("Model", path, assetManager, studio, assetType, acceptDragDropPayload)) {
    if (path.empty()) {
      scale = {1_r, 1_r, 1_r};
      rotation = {};
      translation = {};
    }
    modelChanged |= true;
  }
  changed |= modelChanged;
  ImGui::SameLine();
  if (ImGui::Button(linkModelTransforms ? ICON_FA_LINK : ICON_FA_UNLINK)) {
    linkModelTransforms = !linkModelTransforms;
  }
  if (ImGui::DragTransformRT("Transform", rotation, translation)) {
    changed |= true;
    if (linkModelTransforms && otherRotation && otherTranslation) {
      *otherRotation = rotation;
      *otherTranslation = translation;
    }
  }
  if (ImGui::DragRealXYZ("Scale", scale, 0.01, 0, 0, "%.4f", 0, 1.0f)) {
    changed |= true;
    if (linkModelTransforms && otherScale) {
      *otherScale = scale;
    }
  }
  ImGui::PopID();
  return changed;
}

bool ImGui::CollisionContact(
    char const* label,
    mochi::ColliderType& colliderType,
    mochi::ContactParams& contact,
    mochi::ActorBoundaryElementType& boundaryElementType,
    std::optional<mochi::BoundarySubsamplingParams>* boundarySubsampling) {
  bool changed = false;
  ImGui::SeparatorText(label);
  changed |= ImGui::SimpleReflectionEnum<mochi::ColliderType>("Collider Type", colliderType);
  changed |= ImGui::SimpleReflectionStruct(contact);
  changed |= ImGui::SimpleReflectionEnum<mochi::ActorBoundaryElementType>(
      "Boundary Element Type", boundaryElementType);

  // Soft actors have no boundary-subsampling field; callers pass null to skip that row.
  if (boundarySubsampling != nullptr) {
    bool hasSubsampling = boundarySubsampling->has_value();
    if (ImGui::Checkbox("Boundary Subsampling", &hasSubsampling)) {
      *boundarySubsampling =
          hasSubsampling ? std::make_optional(mochi::BoundarySubsamplingParams{}) : std::nullopt;
      changed = true;
    }
    if (boundarySubsampling->has_value()) {
      auto subsamplingDensity = static_cast<float>((*boundarySubsampling)->subsamplingDensity);
      if (ImGui::DragFloat("Subsampling Density", &subsamplingDensity, 0.01f, 0.0f, 1.0f, "%.3f")) {
        (*boundarySubsampling)->subsamplingDensity = static_cast<mochi::real>(subsamplingDensity);
        changed = true;
      }
      if (ImGui::SimpleReflectionEnum<mochi::BoundarySubsamplingStrategy>(
              "Subsampling Strategy", (*boundarySubsampling)->strategy)) {
        changed = true;
      }
    }
  }
  return changed;
}

// JSON does not allow non-finite values, so replace infinities with practically-infinite huge
// finite values (float max, not real max) before they reach serialization.
static void ReplaceInfWithHuge(mochi::Real3& v) {
  using namespace mochi;
  real constexpr kHuge = std::numeric_limits<float>::max();
  for (int i = 0; i < 3; ++i) {
    if (!IsFinite(v[i])) {
      v[i] = (v[i] < 0_r) ? -kHuge : kHuge;
    }
  }
}

bool ImGui::ArticulatedJointEditor(mochi::prefab::ArticulatedJointPrefab& joint, bool isRoot) {
  using namespace mochi;
  bool changed = false;
  ImGui::SeparatorText("Info");
  if (ImGui::InputText("Name", &joint.name, ImGuiInputTextFlags_CharsNoBlank)) {
    changed |= true;
  }
  changed |= ImGui::ComboArticulatedJointType(
      "Type",
      joint.type,
      isRoot ? ImGui::ArticulatedJointTypeFilter::HardFreeOnly
             : ImGui::ArticulatedJointTypeFilter::NoFreeCycle);
  ImGui::SeparatorText("Local Transform (Parent Link From Joint)");
  changed |= ImGui::DragTransformRT("Parent Link From Joint", joint.parentLinkFromJoint);
  ImGui::SeparatorText("Axis / Limits");
  bool canEditAxis = joint.type == mochi::ArticulatedJointType::Prismatic ||
      joint.type == mochi::ArticulatedJointType::Revolute ||
      joint.type == mochi::ArticulatedJointType::Spherical;
  ImGui::BeginDisabled(!canEditAxis);
  auto currentAxisNormalized = mochi::Normalize(joint.axis);

  if (ImGui::DragRealXYZ("Axis", joint.axis)) {
    auto const newAxisNormalized = mochi::Normalize(joint.axis);
    if (joint.type != mochi::ArticulatedJointType::Spherical) {
      // reassign current min/max limits to changed axis (extract scalar using old axis)
      if (joint.minLimit && IsFinite(*joint.minLimit)) {
        mochi::real scalarMinLimit =
            superdex::robotics::SafeDot(*joint.minLimit, currentAxisNormalized);
        joint.minLimit = superdex::robotics::SafeScale(newAxisNormalized, scalarMinLimit);
      }
      if (joint.maxLimit && IsFinite(*joint.maxLimit)) {
        mochi::real scalarMaxLimit =
            superdex::robotics::SafeDot(*joint.maxLimit, currentAxisNormalized);
        joint.maxLimit = superdex::robotics::SafeScale(newAxisNormalized, scalarMaxLimit);
      }
    }
    currentAxisNormalized = newAxisNormalized;
    changed |= true;
  }
  char const* unitFmt = GetUnitFormat(UnitFormat::Position, joint.type);
  float const scale =
      joint.type == mochi::ArticulatedJointType::Prismatic ? 1.0f : mochi::kDegreesPerRadian;
  float const speed = joint.type == mochi::ArticulatedJointType::Prismatic ? 0.001f : 1.0f;
  if (joint.type == mochi::ArticulatedJointType::Spherical) {
    if (ImGui::DragOptionalRealXYZ("Min Limit", joint.minLimit, speed, 0, 0, unitFmt, 0, scale)) {
      changed |= true;
    }
    if (ImGui::DragOptionalRealXYZ("Max Limit", joint.maxLimit, speed, 0, 0, unitFmt, 0, scale)) {
      changed |= true;
    }
  } else {
    std::optional<mochi::real> scalarMinLimit = joint.minLimit
        ? superdex::robotics::SafeDot(*joint.minLimit, currentAxisNormalized)
        : std::optional<mochi::real>{};
    std::optional<mochi::real> scalarMaxLimit = joint.maxLimit
        ? superdex::robotics::SafeDot(*joint.maxLimit, currentAxisNormalized)
        : std::optional<mochi::real>{};
    if (ImGui::DragOptionalReal("Min Limit", scalarMinLimit, speed, 0, 0, unitFmt, scale)) {
      changed |= true;
      joint.minLimit = scalarMinLimit
          ? superdex::robotics::SafeScale(currentAxisNormalized, *scalarMinLimit)
          : std::optional<mochi::Real3>{};
    }
    if (ImGui::DragOptionalReal("Max Limit", scalarMaxLimit, speed, 0, 0, unitFmt, scale)) {
      changed |= true;
      joint.maxLimit = scalarMaxLimit
          ? superdex::robotics::SafeScale(currentAxisNormalized, *scalarMaxLimit)
          : std::optional<mochi::Real3>{};
    }
  }
  if (changed) {
    if (joint.minLimit.has_value() && !IsFinite(*joint.minLimit)) {
      ReplaceInfWithHuge(*joint.minLimit);
    }
    if (joint.maxLimit.has_value() && !IsFinite(*joint.maxLimit)) {
      ReplaceInfWithHuge(*joint.maxLimit);
    }
  }

  changed |= ImGui::DragReal(
      "Limit Stiffness",
      &joint.limitStiffness,
      0.01f,
      0.0f,
      std::numeric_limits<float>::max(),
      GetUnitFormat(UnitFormat::Stiffness, joint.type, joint.limitStiffness));
  changed |= ImGui::DragReal(
      "Limit Damping",
      &joint.limitDamping,
      0.01f,
      0.0f,
      std::numeric_limits<float>::max(),
      GetUnitFormat(UnitFormat::Damping, joint.type, joint.limitDamping));
  ImGui::EndDisabled();
  ImGui::SeparatorText("Dynamics");
  // A Hard joint has no degrees of freedom, so joint dynamics are meaningless for it.
  ImGui::BeginDisabled(joint.type == mochi::ArticulatedJointType::Hard);
  changed |= ImGui::DragReal(
      "Viscous Friction",
      &joint.friction.viscous,
      0.01f,
      0.0f,
      std::numeric_limits<float>::max(),
      GetUnitFormat(
          UnitFormat::ViscousFriction, joint.type, static_cast<float>(joint.friction.viscous)));
  changed |= ImGui::DragReal(
      "Coulomb Friction",
      &joint.friction.coulomb,
      0.01f,
      0.0f,
      std::numeric_limits<float>::max(),
      GetUnitFormat(
          UnitFormat::CoulombFriction, joint.type, static_cast<float>(joint.friction.coulomb)));
  changed |= ImGui::DragOptionalReal(
      "Inertia",
      joint.inertia,
      0.01f,
      0.0f,
      std::numeric_limits<float>::max(),
      GetUnitFormat(UnitFormat::Inertia, joint.type, joint.inertia.value_or(0.0f)));
  ImGui::EndDisabled();
  return changed;
}

bool ImGui::ArticulatedLinkEditor(
    mochi::prefab::ArticulatedLinkPrefab& link,
    SuperDexStudio* studio,
    AssetManager const& assetManager,
    bool acceptDragDropPayload,
    bool& modelChanged) {
  bool changed = false;

  ImGui::SeparatorText("Info");
  changed |= ImGui::InputText("Name", &link.name, ImGuiInputTextFlags_CharsNoBlank);
  changed |= ImGui::InputText("Layer", &link.layer, ImGuiInputTextFlags_CharsNoBlank);
  changed |= ImGui::Checkbox("Has Gravity", &link.hasGravity);
  ImGui::SeparatorText("Local Transform (Parent Joint From Link)");
  changed |= ImGui::DragTransformRT("Parent Joint From Link", link.parentJointFromLink);

  changed |=
      ImGui::InertialProperties(link.density, link.mass, link.centerOfMass, link.momentOfInertia);

  changed |= ImGui::ModelEditor(
      "Collision Model",
      studio,
      AssetType::MochiModel,
      link.shapeFile,
      link.shapeScale,
      link.shapeRotation,
      link.shapeTranslation,
      &link.renderModelScale,
      &link.renderModelRotation,
      &link.renderModelTranslation,
      assetManager,
      acceptDragDropPayload,
      modelChanged);
  changed |= ImGui::ModelEditor(
      "Render Model",
      studio,
      AssetType::RenderModel,
      link.renderModelFile,
      link.renderModelScale,
      link.renderModelRotation,
      link.renderModelTranslation,
      &link.shapeScale,
      &link.shapeRotation,
      &link.shapeTranslation,
      assetManager,
      acceptDragDropPayload,
      modelChanged);

  changed |= ImGui::CollisionContact(
      "Collision / Contact",
      link.colliderType,
      link.contact,
      link.boundaryElementType,
      &link.boundarySubsampling);

  return changed;
}

bool ImGui::JointPoseEditor(BotPrefab const& builtPrefab, DynamicArray<real>& pose) {
  using namespace mochi;
  if (builtPrefab._dofIndices.empty()) {
    return false;
  }
  bool changed = false;

  // Ensure the pose array is sized to the bot's DOF count.
  if (pose.size() != builtPrefab._dofIndices.size()) {
    pose.resize(builtPrefab._dofIndices.size(), 0.0f);
  }

  float const sliderWidth = ImGui::CalcItemWidth();
  float const spacing = ImGui::GetStyle().ItemSpacing.x;
  float const btnW = (sliderWidth - 4 * spacing) / 5.0f;
  auto setPose = [&](DynamicArray<real> newPose) {
    if (newPose.size() == pose.size()) {
      pose = std::move(newPose);
      changed = true;
    }
  };
  constexpr std::pair<char const*, MakeBotPoseType> kPoseButtons[] = {
      {"Zero", MakeBotPoseType::Zero},
      {"Min", MakeBotPoseType::Min},
      {"Max", MakeBotPoseType::Max},
      {"Mid", MakeBotPoseType::Mid},
      {"Random", MakeBotPoseType::Random}};
  bool firstPoseButton = true;
  for (auto const& [label, poseType] : kPoseButtons) {
    if (!firstPoseButton) {
      ImGui::SameLine();
    }
    firstPoseButton = false;
    if (ImGui::Button(label, ImVec2(btnW, 0))) {
      setPose(MakeBotPose(builtPrefab, poseType, mochi::ErrorLog{}));
    }
  }

  for (size_t iDof = 0; iDof < builtPrefab._dofIndices.size(); ++iDof) {
    int const iJoint = builtPrefab._dofIndices[iDof];
    auto const& joint = builtPrefab.joints[iJoint];
    bool const isSpherical = joint.type == ArticulatedJointType::Spherical;
    // For spherical joints, _dofIndices contains 3 consecutive entries pointing
    // to the same joint. Determine which sub-DOF (X/Y/Z) this entry represents.
    int sphericalSubDof = 0;
    if (isSpherical) {
      for (size_t j = iDof; j > 0 && builtPrefab._dofIndices[j - 1] == iJoint; --j) {
        ++sphericalSubDof;
      }
    }

    ImGui::PushID(static_cast<int>(iDof));

    char const* unitFmt = isSpherical ? ::GetUnitFormat(UnitFormat::Degrees)
                                      : ::GetUnitFormat(UnitFormat::Position, joint.type);
    char hiddenLabel[32];
    snprintf(hiddenLabel, sizeof(hiddenLabel), "##dof%zu", iDof);

    // Rotational DOFs (revolute and spherical sub-DOFs) display in degrees.
    float scale =
        (joint.type == ArticulatedJointType::Revolute || isSpherical) ? kDegreesPerRadian : 1.0f;
    auto value = static_cast<float>(pose[iDof]) * scale;
    bool edited = false;
    // Compute per-DOF scalar limits. Revolute/Prismatic encode limits as
    // axis * scalar (extract via Dot). Spherical encodes per-axis limits in
    // (X, Y, Z) components — index directly by sphericalSubDof.
    bool hasFiniteLimits = joint.minLimit.has_value() && IsFinite(*joint.minLimit) &&
        joint.maxLimit.has_value() && IsFinite(*joint.maxLimit);
    if (hasFiniteLimits) {
      float minLimit = 0.0f;
      float maxLimit = 0.0f;
      if (isSpherical) {
        minLimit = static_cast<float>((*joint.minLimit)[sphericalSubDof]) * scale;
        maxLimit = static_cast<float>((*joint.maxLimit)[sphericalSubDof]) * scale;
      } else {
        auto const axisNormalized = Normalize(joint.axis);
        minLimit = static_cast<float>(Dot(*joint.minLimit, axisNormalized)) * scale;
        maxLimit = static_cast<float>(Dot(*joint.maxLimit, axisNormalized)) * scale;
      }
      edited = ImGui::SliderFloat(hiddenLabel, &value, minLimit, maxLimit, unitFmt);
    } else {
      // DragFloat for infinite / unbounded position.
      float const speed =
          (joint.type == ArticulatedJointType::Revolute || isSpherical) ? 1.0f : 0.001f;
      edited = ImGui::DragFloat(hiddenLabel, &value, speed, 0, 0, unitFmt);
    }
    if (edited) {
      pose[iDof] = static_cast<real>(value) / scale;
      changed = true;
    }
    ImGui::SameLine();
    ImGui::Text("[%d] %s", iJoint, joint.name.c_str());
    if (isSpherical) {
      ImGui::SameLine(0.0f, 0.0f);
      ImGui::TextDisabled(" (%c)", "XYZ"[sphericalSubDof]);
    }
    ImGui::PopID();
  }

  return changed;
}
