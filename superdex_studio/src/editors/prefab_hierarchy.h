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

#pragma once

#include <mochi_core/utils/dynamic_string.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace superdex::studio {

struct SceneStage;
class Viewport;

// Fallback-name prefixes for unnamed top-level actors and links, so rows, display names and
// collision checks all resolve the same names.
inline constexpr std::string_view kRigidPrefix = "Rigid_";
inline constexpr std::string_view kSoftPrefix = "Soft_";
inline constexpr std::string_view kRodPrefix = "Rod_";
inline constexpr std::string_view kArticulatedPrefix = "Articulated_";
inline constexpr std::string_view kLinkPrefix = "Link_";
inline constexpr std::string_view kNestedPrefix = "Nested_";

// An actor's display name: its own name, or "<prefix><index>" when unnamed.
std::string FallbackName(std::string_view prefix, mochi::DynamicString const& name, int index);

// One row of a prefab hierarchy: a top-level actor, a nested prefab, or an articulated link.
struct PrefabHierarchyItem {
  enum class Kind { None, Rigid, ArticulatedActor, ArticulatedLink, NestedPrefab, Soft, Rod };
  Kind kind = Kind::None;
  int actorIndex = -1;
  int linkIndex = -1; // articulated link index (ArticulatedLink only)

  bool operator==(PrefabHierarchyItem const& other) const = default;
};

// The staged actors a hierarchy row covers: one actor or link, or everything an articulated actor
// (its links and skin) or a nested prefab stages.
std::vector<int> StagedIndicesFor(SceneStage const& stage, PrefabHierarchyItem const& item);

// The actor table of a scene prefab: its rigid, soft, rod and articulated actors (with their link
// trees) and nested prefabs, sortable by name and type. Hovering a row highlights what it covers in
// the viewport, and an eye on each row hides or shows it. Shared by the Prefab Editor, which edits
// the prefab through the callbacks, and the Asset Tuner, which only shows it.
class PrefabHierarchy {
 public:
  // A top-level row: an actor or nested prefab, its display name and type.
  struct Row {
    std::string name;
    char const* type = "";
    PrefabHierarchyItem item;
  };

  // What a click, a selection and the right-click menus mean to the editor. Each may be left unset.
  struct Callbacks {
    // Whether a row draws as selected.
    std::function<bool(PrefabHierarchyItem const&)> isSelected;
    // A click on a row. `rowIndex` is the row's position in GetRows(), or -1 for a link.
    std::function<void(int rowIndex, PrefabHierarchyItem const&)> onClick;
    // Fills a row's right-click menu.
    std::function<void(PrefabHierarchyItem const&)> rowMenu;
    // Fills the right-click menu of the table's empty space.
    std::function<void()> emptySpaceMenu;
    // Extra rows an editor lists under a rigid, soft, rod or nested-prefab row, or under a link
    // before its child links, such as the constraints naming it. `hasChildRows` makes the row a
    // tree node; `showChildRows` draws them (one table row each) while it is open.
    std::function<bool(PrefabHierarchyItem const&)> hasChildRows;
    std::function<void(PrefabHierarchyItem const&)> showChildRows;
  };

  // Draws the table, filling the rest of the current window.
  void Show(
      mochi::prefab::ScenePrefab const& prefab,
      SceneStage& stage,
      Viewport& viewport,
      Callbacks const& callbacks);

  // Opens the tree down to a link and scrolls to it the next time the table is drawn.
  void RevealLink(int actorIndex, int linkIndex) {
    _reveal =
        PrefabHierarchyItem{PrefabHierarchyItem::Kind::ArticulatedLink, actorIndex, linkIndex};
  }

  // The top-level rows in the order the table last drew them.
  std::vector<Row> const& GetRows() const {
    return _rows;
  }

 private:
  struct DrawContext;
  void ShowLeafRow(DrawContext const& context, int rowIndex, bool indent);
  void ShowArticulatedRow(DrawContext& context, int rowIndex);
  void ShowLinkTree(DrawContext const& context, int actorIndex, int linkIndex);

  // Reused each frame rather than reallocated.
  std::vector<Row> _rows;
  // Each link's children, rebuilt per articulated actor while drawing.
  std::vector<std::vector<int>> _linkChildren;
  std::optional<PrefabHierarchyItem> _reveal;
};

} // namespace superdex::studio
