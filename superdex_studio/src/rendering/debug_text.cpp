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

#include "rendering/debug_text.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace superdex::studio {

// Labels sit on a rounded background chip similar to viewport's stats (FPS)
constexpr ImVec2 kLabelPadding{5.0f, 2.0f};

// Vertical gap left between two chips that would otherwise have collided.
constexpr float kLabelGap = 2.0f;

namespace {

struct PlacedLabel {
  ImVec2 textPos;
  ImVec2 backgroundMin;
  ImVec2 backgroundMax;
  ImU32 color = 0;
  char const* begin = nullptr;
  char const* end = nullptr;
  // Clip w, i.e. view-space depth. Nearer labels are placed first and so keep their ideal spot.
  double depth = 0.0;
};

// Whether two chips are closer than @ref kLabelGap vertically while also overlapping horizontally.
bool Overlaps(PlacedLabel const& a, PlacedLabel const& b) {
  return a.backgroundMin.x < b.backgroundMax.x && b.backgroundMin.x < a.backgroundMax.x &&
      a.backgroundMin.y < b.backgroundMax.y + kLabelGap &&
      b.backgroundMin.y < a.backgroundMax.y + kLabelGap;
}

void ShiftVertically(PlacedLabel& label, float dy) {
  label.textPos.y += dy;
  label.backgroundMin.y += dy;
  label.backgroundMax.y += dy;
}

// Whether a chip is worth laying out at all: finite, and not wholly off the viewport.
bool IsLayoutable(PlacedLabel const& label, ImVec2 viewportMin, ImVec2 viewportMax) {
  if (!std::isfinite(label.backgroundMin.x) || !std::isfinite(label.backgroundMin.y) ||
      !std::isfinite(label.backgroundMax.x) || !std::isfinite(label.backgroundMax.y)) {
    return false;
  }
  return label.backgroundMax.x > viewportMin.x && label.backgroundMin.x < viewportMax.x &&
      label.backgroundMax.y > viewportMin.y && label.backgroundMin.y < viewportMax.y;
}

} // namespace

void DebugText::Draw(
    filament::math::float3 worldPosition,
    std::string_view text,
    filament::math::float4 color,
    ImVec2 pixelOffset) {
  _items.push_back({worldPosition, std::string{text}, color, pixelOffset});
}

void DebugText::Show(
    filament::math::mat4 const& viewMatrix,
    filament::math::mat4 const& projMatrix,
    ImVec2 contentOrigin,
    float logicalWidth,
    float logicalHeight,
    float backgroundOpacity) const {
  if (_items.empty()) {
    return;
  }
  filament::math::mat4 const viewProj = projMatrix * viewMatrix;
  ImVec2 const viewportMax{contentOrigin.x + logicalWidth, contentOrigin.y + logicalHeight};

  // Project everything first, then de-collide, then draw
  std::vector<PlacedLabel> placed;
  placed.reserve(_items.size());
  for (Item const& item : _items) {
    filament::math::double4 const clip = viewProj *
        filament::math::double4{
            item.worldPosition.x, item.worldPosition.y, item.worldPosition.z, 1.0};
    if (clip.w <= 0.0) {
      continue; // behind the camera; projecting would mirror it to the wrong side of the screen
    }
    auto const ndcX = static_cast<float>(clip.x / clip.w);
    auto const ndcY = static_cast<float>(clip.y / clip.w);
    PlacedLabel label;
    label.begin = item.text.c_str();
    label.end = label.begin + item.text.size();
    label.depth = clip.w;
    label.color = ImGui::ColorConvertFloat4ToU32(
        ImVec4{item.color.x, item.color.y, item.color.z, item.color.w});
    ImVec2 const textSize = ImGui::CalcTextSize(label.begin, label.end);
    label.textPos = {
        contentOrigin.x + (ndcX * 0.5f + 0.5f) * logicalWidth - textSize.x * 0.5f +
            item.pixelOffset.x,
        contentOrigin.y + (0.5f - ndcY * 0.5f) * logicalHeight - textSize.y * 0.5f +
            item.pixelOffset.y};
    label.backgroundMin = {label.textPos.x - kLabelPadding.x, label.textPos.y - kLabelPadding.y};
    label.backgroundMax = {
        label.textPos.x + textSize.x + kLabelPadding.x,
        label.textPos.y + textSize.y + kLabelPadding.y};
    if (!IsLayoutable(label, contentOrigin, viewportMax)) {
      continue;
    }
    placed.push_back(label);
  }
  if (placed.empty()) {
    return;
  }

  // Nearest first, so when two labels want the same spot the one on the geometry closest to the
  // camera is the one that keeps it and the farther one steps aside.
  std::ranges::stable_sort(
      placed, [](PlacedLabel const& a, PlacedLabel const& b) { return a.depth < b.depth; });

  // Push each label clear of the ones already placed. Displacement is vertical only
  float const viewportMidY = contentOrigin.y + logicalHeight * 0.5f;
  for (std::size_t i = 1; i < placed.size(); ++i) {
    float const direction = placed[i].backgroundMin.y < viewportMidY ? 1.0f : -1.0f;
    for (std::size_t pass = 0; pass <= i; ++pass) {
      bool moved = false;
      for (std::size_t j = 0; j < i; ++j) {
        if (!Overlaps(placed[i], placed[j])) {
          continue;
        }
        float const dy = direction > 0.0f
            ? (placed[j].backgroundMax.y + kLabelGap) - placed[i].backgroundMin.y
            : (placed[j].backgroundMin.y - kLabelGap) - placed[i].backgroundMax.y;
        if (dy * direction <= 0.0f) {
          continue; // no forward progress available; leave this pair
        }
        ShiftVertically(placed[i], dy);
        moved = true;
      }
      if (!moved) {
        break;
      }
    }
  }

  ImDrawList* drawList = ImGui::GetWindowDrawList();
  ImVec4 background = ImGui::GetStyleColorVec4(ImGuiCol_Button);
  background.w *= std::clamp(backgroundOpacity, 0.0f, 1.0f);
  ImU32 const backgroundColor = ImGui::ColorConvertFloat4ToU32(background);
  for (PlacedLabel const& label : placed) {
    // Half the chip's height rounds it into a pill, matching the viewport's toolbar buttons.
    drawList->AddRectFilled(
        label.backgroundMin,
        label.backgroundMax,
        backgroundColor,
        (label.backgroundMax.y - label.backgroundMin.y) * 0.5f);
    drawList->AddText(label.textPos, label.color, label.begin, label.end);
  }
}

void DebugText::Clear() {
  _items.clear();
}

} // namespace superdex::studio
