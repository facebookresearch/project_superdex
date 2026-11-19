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

#include "editors/prefab_hierarchy.h"
#include "rendering/scene_stage.h"
#include "rendering/viewport.h"
#include "ui/imgui_widgets.h"

#include <superdex_robotics/superdex_robotics.h>

#include <imgui.h>

#include <algorithm>
#include <cstring>
#include <unordered_set>

namespace superdex::studio {

namespace {

using Kind = PrefabHierarchyItem::Kind;

constexpr filament::math::float3 kHoverColor{46.f / 255.f, 134.f / 255.f, 233.f / 255.f};

bool IsHidden(SceneStage const& stage, PrefabHierarchyItem const& item) {
  std::vector<int> const indices = StagedIndicesFor(stage, item);
  return !indices.empty() &&
      std::ranges::all_of(indices, [&stage](int i) { return stage.GetActors()[i].hidden; });
}

// Dims a hidden row's name until the returned guard goes out of scope.
struct HiddenTextStyle {
  explicit HiddenTextStyle(bool hidden) : _hidden(hidden) {
    if (_hidden) {
      ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    }
  }
  ~HiddenTextStyle() {
    if (_hidden) {
      ImGui::PopStyleColor();
    }
  }
  HiddenTextStyle(HiddenTextStyle const&) = delete;
  HiddenTextStyle& operator=(HiddenTextStyle const&) = delete;
  HiddenTextStyle(HiddenTextStyle&&) = delete;
  HiddenTextStyle& operator=(HiddenTextStyle&&) = delete;

 private:
  bool _hidden;
};

} // namespace

std::string FallbackName(std::string_view prefix, mochi::DynamicString const& name, int index) {
  return name.empty() ? (std::string(prefix) + std::to_string(index)) : std::string(name);
}

std::vector<int> StagedIndicesFor(SceneStage const& stage, PrefabHierarchyItem const& item) {
  StagedActorSource source = StagedActorSource::Rigid;
  switch (item.kind) {
    case Kind::Rigid:
      source = StagedActorSource::Rigid;
      break;
    case Kind::Soft:
      source = StagedActorSource::Soft;
      break;
    case Kind::Rod:
      source = StagedActorSource::Rod;
      break;
    case Kind::ArticulatedActor:
    case Kind::ArticulatedLink:
      source = StagedActorSource::Articulated;
      break;
    case Kind::NestedPrefab:
      source = StagedActorSource::NestedPrefab;
      break;
    case Kind::None:
      return {};
  }
  bool const oneLink = item.kind == Kind::ArticulatedLink;
  std::vector<int> indices;
  StagedActors const& actors = stage.GetActors();
  for (int i = 0; i < static_cast<int>(actors.size()); ++i) {
    if (actors[i].source == source && actors[i].sourceActorIndex == item.actorIndex &&
        (!oneLink || actors[i].sourceLinkIndex == item.linkIndex)) {
      indices.push_back(i);
    }
  }
  return indices;
}

struct PrefabHierarchy::DrawContext {
  mochi::prefab::ScenePrefab const& prefab;
  SceneStage& stage;
  Viewport& viewport;
  Callbacks const& callbacks;
  // Links of the articulated actor being drawn that open to reveal the link RevealLink named.
  std::unordered_set<int> ancestorsToOpen;

  bool IsSelected(PrefabHierarchyItem const& item) const {
    return callbacks.isSelected && callbacks.isSelected(item);
  }

  bool HasChildRows(PrefabHierarchyItem const& item) const {
    return callbacks.hasChildRows && callbacks.showChildRows && callbacks.hasChildRows(item);
  }

  // The editor's extra rows under `item`, indented like leaf rows, whose labels sit right of the
  // (absent) arrow.
  void ShowChildRows(PrefabHierarchyItem const& item) const {
    float const indent = ImGui::GetTreeNodeToLabelSpacing();
    ImGui::Indent(indent);
    callbacks.showChildRows(item);
    ImGui::Unindent(indent);
  }

  // Hover, click and right-click handling for the row item just drawn.
  void HandleRowItem(int rowIndex, PrefabHierarchyItem const& item, bool clicked) const {
    if (clicked && callbacks.onClick) {
      callbacks.onClick(rowIndex, item);
    }
    if (ImGui::IsItemHovered()) {
      for (int const i : StagedIndicesFor(stage, item)) {
        if (mochi_renderer::SceneObject* const object = stage.GetActors()[i].sceneObject) {
          viewport.HighlightSceneObject(object, kHoverColor);
        }
      }
    }
    if (callbacks.rowMenu &&
        ImGui::BeginPopupContextItem("RowMenu", ImGuiPopupFlags_MouseButtonRight)) {
      callbacks.rowMenu(item);
      ImGui::EndPopup();
    }
  }

  // The row's last cell: its eye.
  void ShowVisibility(PrefabHierarchyItem const& item) const {
    ImGui::TableNextColumn();
    std::vector<int> const indices = StagedIndicesFor(stage, item);
    if (indices.empty()) {
      return;
    }
    bool const hidden = IsHidden(stage, item);
    if (ImGui::VisibilityToggle("##Visibility", hidden)) {
      for (int const i : indices) {
        stage.SetActorHidden(i, !hidden);
      }
    }
  }
};

void PrefabHierarchy::Show(
    mochi::prefab::ScenePrefab const& prefab,
    SceneStage& stage,
    Viewport& viewport,
    Callbacks const& callbacks) {
  _rows.clear();
  _rows.reserve(
      prefab.actors.rigid.size() + prefab.actors.soft.size() + prefab.actors.rod.size() +
      prefab.actors.articulated.size() + prefab.prefabs.size());
  for (int i = 0; i < static_cast<int>(prefab.actors.rigid.size()); ++i) {
    _rows.push_back(
        {FallbackName(kRigidPrefix, prefab.actors.rigid[i].name, i), "Rigid", {Kind::Rigid, i}});
  }
  for (int i = 0; i < static_cast<int>(prefab.actors.soft.size()); ++i) {
    _rows.push_back(
        {FallbackName(kSoftPrefix, prefab.actors.soft[i].name, i), "Soft", {Kind::Soft, i}});
  }
  for (int i = 0; i < static_cast<int>(prefab.actors.rod.size()); ++i) {
    _rows.push_back(
        {FallbackName(kRodPrefix, prefab.actors.rod[i].name, i), "Rod", {Kind::Rod, i}});
  }
  for (int i = 0; i < static_cast<int>(prefab.actors.articulated.size()); ++i) {
    _rows.push_back(
        {FallbackName(kArticulatedPrefix, prefab.actors.articulated[i].name, i),
         "Articulated",
         {Kind::ArticulatedActor, i}});
  }
  for (int i = 0; i < static_cast<int>(prefab.prefabs.size()); ++i) {
    _rows.push_back(
        {FallbackName(kNestedPrefix, prefab.prefabs[i].name, i),
         "Nested Prefab",
         {Kind::NestedPrefab, i}});
  }

  DrawContext context{
      .prefab = prefab, .stage = stage, .viewport = viewport, .callbacks = callbacks};
  // Match the Bot Hierarchy table styling.
  ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive));
  ImGui::BeginChild("PrefabActorsChild");
  auto const tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
      ImGuiTableFlags_Resizable | ImGuiTableFlags_Sortable;
  if (ImGui::BeginTable("##PrefabActors", 3, tableFlags)) {
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_DefaultSort);
    ImGui::TableSetupColumn("Type");
    ImGui::TableSetupVisibilityColumn();
    if (ImGui::TableHeadersRowWithVisibilityToggle(stage.GetNumHiddenActors() > 0)) {
      if (stage.GetNumHiddenActors() > 0) {
        stage.ShowAllActors();
      } else {
        stage.HideAllActors();
      }
    }

    // Sort by the active column and direction; the name breaks ties.
    if (ImGuiTableSortSpecs const* const sortSpecs = ImGui::TableGetSortSpecs();
        sortSpecs != nullptr && sortSpecs->SpecsCount > 0) {
      ImGuiTableColumnSortSpecs const& spec = sortSpecs->Specs[0];
      bool const ascending = spec.SortDirection == ImGuiSortDirection_Ascending;
      std::ranges::stable_sort(_rows, [&spec, ascending](Row const& lhs, Row const& rhs) {
        int cmp = spec.ColumnIndex == 1 ? std::strcmp(lhs.type, rhs.type) : 0;
        if (cmp == 0) {
          cmp = lhs.name.compare(rhs.name);
        }
        return ascending ? cmp < 0 : cmp > 0;
      });
    }

    // With an articulated actor in the table, leaf rows indent their names to line up with the
    // tree nodes' labels, which sit right of the collapse arrow.
    bool const indentLeaves = !prefab.actors.articulated.empty();
    for (int rowIndex = 0; rowIndex < static_cast<int>(_rows.size()); ++rowIndex) {
      PrefabHierarchyItem const& item = _rows[rowIndex].item;
      // Kind, then index, so rows of different kinds sharing an index get different IDs.
      ImGui::PushID(static_cast<int>(item.kind));
      ImGui::PushID(item.actorIndex);
      if (item.kind == Kind::ArticulatedActor) {
        ShowArticulatedRow(context, rowIndex);
      } else {
        ShowLeafRow(context, rowIndex, indentLeaves);
      }
      ImGui::PopID();
      ImGui::PopID();
    }
    ImGui::EndTable();
  }
  // The table scrolls in its own child window, so its empty space is found through ChildWindows
  // rather than BeginPopupContextWindow, which only tests the outer child.
  if (callbacks.emptySpaceMenu && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
      !ImGui::IsAnyItemHovered() && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
    ImGui::OpenPopup("PrefabTableContext");
  }
  if (ImGui::BeginPopup("PrefabTableContext")) {
    callbacks.emptySpaceMenu();
    ImGui::EndPopup();
  }
  ImGui::EndChild();
  ImGui::PopStyleColor(); // ImGuiCol_Header
  // A link to reveal is drawn, and scrolled to, in the frame that asks for it.
  _reveal.reset();
}

void PrefabHierarchy::ShowLeafRow(DrawContext const& context, int rowIndex, bool indent) {
  Row const& row = _rows[rowIndex];
  bool const isSelected = context.IsSelected(row.item);
  bool const hasChildRows = context.HasChildRows(row.item);
  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  float const leafIndent = ImGui::GetTreeNodeToLabelSpacing();
  float const originalIndent = ImGui::GetStyle().IndentSpacing;
  bool clicked = false;
  bool open = false;
  {
    HiddenTextStyle const style(IsHidden(context.stage, row.item));
    // Lets the row's eye take clicks over the row.
    ImGui::SetNextItemAllowOverlap();
    if (hasChildRows) {
      ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow |
          ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanAllColumns;
      if (isSelected) {
        flags |= ImGuiTreeNodeFlags_Selected;
      }
      ImGui::GetStyle().IndentSpacing = originalIndent * 0.4f;
      open = ImGui::TreeNodeEx(row.name.c_str(), flags);
      clicked = ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen();
    } else {
      if (indent) {
        ImGui::Indent(leafIndent);
      }
      clicked =
          ImGui::Selectable(row.name.c_str(), isSelected, ImGuiSelectableFlags_SpanAllColumns);
      if (indent) {
        ImGui::Unindent(leafIndent);
      }
    }
  }
  context.HandleRowItem(rowIndex, row.item, clicked);
  ImGui::TableNextColumn();
  ImGui::TextUnformatted(row.type);
  context.ShowVisibility(row.item);
  if (open) {
    context.ShowChildRows(row.item);
    ImGui::TreePop();
  }
  ImGui::GetStyle().IndentSpacing = originalIndent;
}

void PrefabHierarchy::ShowArticulatedRow(DrawContext& context, int rowIndex) {
  Row const& row = _rows[rowIndex];
  int const actorIndex = row.item.actorIndex;
  auto const& actor = context.prefab.actors.articulated[actorIndex];
  int const numLinks = static_cast<int>(actor.links.size());
  if (static_cast<int>(_linkChildren.size()) < numLinks) {
    _linkChildren.resize(numLinks);
  }
  for (int i = 0; i < numLinks; ++i) {
    _linkChildren[i].clear();
  }
  for (int i = 0; i < numLinks; ++i) {
    if (int const parent = actor.links[i].parentLink; parent >= 0 && parent < numLinks) {
      _linkChildren[parent].push_back(i);
    }
  }
  context.ancestorsToOpen.clear();
  bool const reveal = _reveal.has_value() && _reveal->actorIndex == actorIndex &&
      _reveal->linkIndex >= 0 && _reveal->linkIndex < numLinks;
  if (reveal) {
    for (int link = actor.links[_reveal->linkIndex].parentLink; link >= 0 && link < numLinks;
         link = actor.links[link].parentLink) {
      context.ancestorsToOpen.insert(link);
    }
  }

  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  if (reveal) {
    ImGui::SetNextItemOpen(true, ImGuiCond_Always);
  }
  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick |
      ImGuiTreeNodeFlags_SpanAllColumns;
  if (context.IsSelected(row.item)) {
    flags |= ImGuiTreeNodeFlags_Selected;
  }
  float const originalIndent = ImGui::GetStyle().IndentSpacing;
  ImGui::GetStyle().IndentSpacing = originalIndent * 0.4f;
  bool open = false;
  {
    HiddenTextStyle const style(IsHidden(context.stage, row.item));
    ImGui::SetNextItemAllowOverlap();
    open = ImGui::TreeNodeEx(row.name.c_str(), flags);
  }
  context.HandleRowItem(rowIndex, row.item, ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen());
  ImGui::TableNextColumn();
  ImGui::TextUnformatted(row.type);
  context.ShowVisibility(row.item);
  if (open) {
    for (int i = 0; i < numLinks; ++i) {
      if (actor.links[i].parentLink == superdex::robotics::kIndexNone) {
        ShowLinkTree(context, actorIndex, i);
      }
    }
    ImGui::TreePop();
  }
  ImGui::GetStyle().IndentSpacing = originalIndent;
}

void PrefabHierarchy::ShowLinkTree(DrawContext const& context, int actorIndex, int linkIndex) {
  auto const& link = context.prefab.actors.articulated[actorIndex].links[linkIndex];
  PrefabHierarchyItem const item{Kind::ArticulatedLink, actorIndex, linkIndex};
  bool const hasChildRows = context.HasChildRows(item);
  bool const hasChildren = !_linkChildren[linkIndex].empty() || hasChildRows;

  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  ImGui::PushID(linkIndex);
  if (context.ancestorsToOpen.contains(linkIndex)) {
    ImGui::SetNextItemOpen(true, ImGuiCond_Always);
  }
  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanFullWidth |
      ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnDoubleClick |
      ImGuiTreeNodeFlags_SpanAllColumns;
  if (!hasChildren) {
    flags |=
        ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_Bullet;
  }
  if (context.IsSelected(item)) {
    flags |= ImGuiTreeNodeFlags_Selected;
  }
  bool open = false;
  {
    HiddenTextStyle const style(IsHidden(context.stage, item));
    ImGui::SetNextItemAllowOverlap();
    open = ImGui::TreeNodeEx(FallbackName(kLinkPrefix, link.name, linkIndex).c_str(), flags);
  }
  context.HandleRowItem(-1, item, ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen());
  ImGui::TableNextColumn();
  ImGui::TextDisabled("Articulated Link");
  context.ShowVisibility(item);
  if (_reveal == item) {
    ImGui::SetScrollHereY(0.5f);
  }
  if (hasChildren && open) {
    if (hasChildRows) {
      context.ShowChildRows(item);
    }
    for (int const child : _linkChildren[linkIndex]) {
      ShowLinkTree(context, actorIndex, child);
    }
    ImGui::TreePop();
  }
  ImGui::PopID();
}

} // namespace superdex::studio
