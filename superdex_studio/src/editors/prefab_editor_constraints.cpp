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

#include "editors/prefab_editor.h"

#include "assets/mochi_prefab_asset.h"
#include "rendering/scene_stage.h"
#include "ui/imgui_widgets.h"

#include <string>
#include <string_view>
#include <vector>

// The PrefabEditor members that keep a prefab's constraints in step with its actors.

namespace superdex::studio {

ConstraintActorIndex PrefabEditor::BuildConstraintActorIndex() const {
  std::vector<std::string> nested;
  for (auto const& staged : _stage.GetActors()) {
    if (staged.source == StagedActorSource::NestedPrefab) {
      nested.push_back(staged.name);
    }
  }
  return superdex::studio::BuildConstraintActorIndex(_prefabAsset->GetPrefab(), nested);
}

bool PrefabEditor::RenameActorInConstraints(std::string_view name, bool collides) {
  // Constraints follow the rename once it is committed, not on every keystroke: an intermediate
  // empty or colliding name would detach their references or merge them into another actor's.
  if (ImGui::IsItemActivated()) {
    _renameStartName = std::string(name);
  }
  return ImGui::IsItemDeactivatedAfterEdit() && !name.empty() && !collides &&
      RenameConstraintActor(_prefabAsset->GetPrefab().constraints, _renameStartName, name) > 0;
}

} // namespace superdex::studio
