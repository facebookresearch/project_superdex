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

namespace superdex::studio {

// Offset and opacity of the 1px drop shadow that keeps labels legible over arbitrary scene content
// (same treatment as the icon shadows in imgui_widgets.cpp).
constexpr ImVec2 kShadowOffset{1.0f, 1.0f};
constexpr ImU32 kShadowColor = IM_COL32(0, 0, 0, 160);

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
    float logicalHeight) const {
  if (_items.empty()) {
    return;
  }
  filament::math::mat4 const viewProj = projMatrix * viewMatrix;
  ImDrawList* drawList = ImGui::GetWindowDrawList();
  for (Item const& item : _items) {
    filament::math::double4 const clip = viewProj *
        filament::math::double4{
            item.worldPosition.x, item.worldPosition.y, item.worldPosition.z, 1.0};
    if (clip.w <= 0.0) {
      continue; // behind the camera; projecting would mirror it to the wrong side of the screen
    }
    float const ndcX = static_cast<float>(clip.x / clip.w);
    float const ndcY = static_cast<float>(clip.y / clip.w);
    char const* begin = item.text.c_str();
    char const* end = begin + item.text.size();
    ImVec2 const textSize = ImGui::CalcTextSize(begin, end);
    ImVec2 const pos{
        contentOrigin.x + (ndcX * 0.5f + 0.5f) * logicalWidth - textSize.x * 0.5f +
            item.pixelOffset.x,
        contentOrigin.y + (0.5f - ndcY * 0.5f) * logicalHeight - textSize.y * 0.5f +
            item.pixelOffset.y};
    drawList->AddText(
        ImVec2{pos.x + kShadowOffset.x, pos.y + kShadowOffset.y}, kShadowColor, begin, end);
    drawList->AddText(
        pos,
        ImGui::ColorConvertFloat4ToU32(
            ImVec4{item.color.x, item.color.y, item.color.z, item.color.w}),
        begin,
        end);
  }
}

void DebugText::Clear() {
  _items.clear();
}

} // namespace superdex::studio
