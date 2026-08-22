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

#include "scene_panel.h"

#include "gui/gui.h"

#include <imguios/imguios.h>
#include <mochi_core/utils/basic_utils.h> // Max
#include <mochi_core/utils/defer.h>

#include <algorithm>
#include <cfloat>
#include <tuple>

using namespace mochi;
using namespace mochi::dbg;

// Font Awesome glyph for an actor type.
static char const* IconForActorType(ActorType type) {
  switch (type) {
    case ActorType::Rigid:
      return ICON_FA_CUBE;
    case ActorType::Soft:
      return ICON_FA_VOLLEYBALL_BALL;
    case ActorType::Articulated:
      return ICON_FA_ROBOT;
    case ActorType::Shell:
      return ICON_FA_TSHIRT;
    case ActorType::Rod:
      return ICON_FA_GRIP_LINES;
    case ActorType::None:
    case ActorType::Count:
      return ICON_FA_QUESTION;
  }
  return ICON_FA_QUESTION; // Unreachable; satisfies the non-void return.
}

// Font Awesome glyph for an actor's render visibility. The "###vis" suffix pins the ImGui ID so it
// does not change with the glyph, which would otherwise reset the widget mid-interaction.
static char const* IconForVisibility(bool visible) {
  return visible ? ICON_FA_EYE "###vis" : ICON_FA_EYE_SLASH "###vis";
}

// Fixed column metrics [px] shared by every actor row, so rows align regardless of glyph widths.
// Computed once per frame (font size can change between frames).
namespace {
struct ActorRowLayout {
  float typeIconWidth = 0.0f; // Widest actor-type glyph; names start past it.
  float visIconWidth = 0.0f; // Widest visibility glyph; reserved at the panel's right edge.
  float spacing = 0.0f; // Gap between the actor-type icon and the name.
};
} // namespace

static ActorRowLayout ComputeActorRowLayout() {
  ActorRowLayout layout;
  for (int i = 0; i < static_cast<int>(ActorType::Count); ++i) {
    char const* const icon = IconForActorType(static_cast<ActorType>(i));
    layout.typeIconWidth = Max(layout.typeIconWidth, ImGui::CalcTextSize(icon).x);
  }
  layout.visIconWidth =
      Max(ImGui::CalcTextSize(ICON_FA_EYE).x, ImGui::CalcTextSize(ICON_FA_EYE_SLASH).x);
  layout.spacing = ImGui::GetStyle().ItemInnerSpacing.x;
  return layout;
}

// Draw an actor's icon and name on the current line, aligning the name to a fixed icon column
// (relative to where the icon starts) so names line up regardless of glyph width.
static void DrawActorIconAndName(ActorInfo const& info, ActorRowLayout const& layout) {
  float const iconStartX = ImGui::GetCursorPosX();
  ImGui::TextUnformatted(IconForActorType(info.type));
  ImGui::SameLine();
  ImGui::SetCursorPosX(iconStartX + layout.typeIconWidth + layout.spacing);
  ImGui::TextUnformatted(info.displayName.c_str());
}

// Draw the render visibility toggle on the current line, right-justified.
static void DrawVisibilityToggle(ActorInfo& info, RenderScene& renderScene, float iconWidth) {
  if (!info.hasMesh) {
    return;
  }

  ImGui::SameLine();
  ImVec2 const cursor = ImGui::GetCursorScreenPos();
  ImGui::SetCursorScreenPos({cursor.x + ImGui::GetContentRegionAvail().x - iconWidth, cursor.y});

  // Dim the crossed-out eye so a hidden actor reads as inactive at a glance.
  ImGuiCol const color = info.isVisible ? ImGuiCol_Text : ImGuiCol_TextDisabled;
  ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(color));
  MOCHI_DEFER(ImGui::PopStyleColor());

  if (ImGui::Selectable(
          IconForVisibility(info.isVisible), false, ImGuiSelectableFlags_None, {iconWidth, 0.0f})) {
    info.isVisible = !info.isVisible;
    info.visibilityOverridden = true;
    renderScene.SetMeshVisible(info.meshId, info.isVisible);
  }
}

// Render one actor row in the tree. Its children (empty for a leaf) are rendered indented.
static void DrawActorNode(
    UiState& state,
    ActorHandle handle,
    ActorRowLayout const& layout,
    bool isLeafNode = false) {
  ScenePanelState& sp = state.scene;
  auto const it = sp.actors.find(handle);
  MOCHI_ASSERT_VERBOSE(it != sp.actors.end(), "Actor missing from the actor map");
  ActorInfo& info = it->second;
  bool const isSelected = sp.selectedActor == handle;
  bool const hasChildren = !info.children.empty();

  // Disambiguate rows using the handle. Names may collide, but the lower 32 bits of the handle
  // comes from the entt::entity ID which is unique within the current scene.
  ImGui::PushID(static_cast<int>(handle.value));

  MOCHI_DEFER(ImGui::PopID());

  // AllowOverlap lets the visibility toggle, submitted later on the same line, win the hit test
  // against this node's full-width hit box.
  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnArrow |
      ImGuiTreeNodeFlags_AllowOverlap;
  if (isSelected) {
    flags |= ImGuiTreeNodeFlags_Selected;
  }
  if (!hasChildren) {
    flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
  }

  bool const pushed = ImGui::TreeNodeEx("##node", flags, "") && hasChildren;

  // Clicking the label (not the disclosure arrow) toggles selection.
  if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
    sp.selectedActor = isSelected ? ActorHandle{} : handle;
  }
  ImGui::SameLine(0.0f, 0.0f);
  DrawActorIconAndName(info, layout);
  DrawVisibilityToggle(info, *state.viewport.renderScene, layout.visIconWidth);

  if (pushed) {
    MOCHI_ASSERT_VERBOSE(
        info.children.empty() || !isLeafNode,
        "Illegal for a child actor to have children. Expected just one level of hierarchy.");
    if (!isLeafNode) {
      for (ActorHandle const child : info.children) {
        DrawActorNode(state, child, layout, /*isLeafNode*/ true);
      }
    }
    ImGui::TreePop();
  }
}

// Rebuild the display ordering (`sortedRootActors` + each ActorInfo::children) from `sp.actors`.
static void RebuildActorTree(ScenePanelState& sp) {
  sp.sortedRootActors.clear();
  for (auto& [handle, info] : sp.actors) {
    info.children.clear();
  }

  // An actor whose parent is not itself in the map (e.g. the parent was pruned) is treated as
  // top-level, so it still appears in the tree instead of being silently dropped.
  for (auto& [handle, info] : sp.actors) {
    auto const parentIt = sp.actors.find(info.parent);
    if (parentIt != sp.actors.end()) {
      parentIt->second.children.push_back(handle);
    } else {
      sp.sortedRootActors.push_back(handle);
    }
  }

  // Sort by (displayName, handle). The handle tiebreak makes the ordering total, which matters
  // because unordered_map iteration order is not deterministic.
  auto const byNameThenHandle = [&sp](ActorHandle a, ActorHandle b) {
    return std::tie(sp.actors.at(a).displayName, a.value) <
        std::tie(sp.actors.at(b).displayName, b.value);
  };
  std::ranges::sort(sp.sortedRootActors, byNameThenHandle);
  for (auto& [handle, info] : sp.actors) {
    std::ranges::sort(info.children, byNameThenHandle);
  }
}

void dbg::BuildScenePanel(UiState& state) {
  ScenePanelState& sp = state.scene;
  MOCHI_ASSERT_VERBOSE(sp.selectedSceneIndex < isize(sp.scenes));

  // Scene dropdown. Selecting an entry queues a SelectScene command (applied before the next UI
  // build in OnUpdate) — do not mutate the selection here.
  char const* previewLabel = sp.scenes[sp.selectedSceneIndex].name.c_str();
  ImGui::SetNextItemWidth(-FLT_MIN); // Use remaining width
  if (ImGui::BeginCombo("##scene", previewLabel)) {
    MOCHI_DEFER(ImGui::EndCombo());
    for (size_t i = 0; i < sp.scenes.size(); ++i) {
      SceneInfo const& info = sp.scenes[i];
      bool const isSelected = sp.selectedSceneIndex == static_cast<int>(i);
      ImGui::PushID(static_cast<int>(i)); // Scene names may collide; disambiguate by index.
      MOCHI_DEFER(ImGui::PopID());
      if (ImGui::Selectable(info.name.c_str(), isSelected)) {
        state.uiCommands.push_back(UiCommand{CommandId::SelectScene, info.handle.value});
      }
      if (isSelected) {
        ImGui::SetItemDefaultFocus();
      }
      if (i == 0) {
        ImGui::Separator(); // Divide the "None" entry from the real scenes.
      }
    }
  }

  if (ImGui::BeginTabBar("##sceneTabs")) {
    MOCHI_DEFER(ImGui::EndTabBar());
    if (ImGui::BeginTabItem("Actors")) {
      MOCHI_DEFER(ImGui::EndTabItem());
      if (sp.treeDirty) {
        RebuildActorTree(sp);
        sp.treeDirty = false;
      }
      if (sp.actors.empty()) {
        ImGui::TextDisabled("No actors.");
      } else {
        ActorRowLayout const layout = ComputeActorRowLayout();
        for (ActorHandle const rootHandle : sp.sortedRootActors) {
          DrawActorNode(state, rootHandle, layout);
        }
      }
    }

    // TODO: Populate constraints tab
    // if (ImGui::BeginTabItem("Constraints")) {
    //   MOCHI_DEFER(ImGui::EndTabItem());
    //   ImGui::TextDisabled("No constraints.");
    // }
  }
}
