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

#include "editors/asset_editor.h"

#include "app/app.h"

namespace superdex::studio {

AssetEditor::AssetEditor(SuperDexStudio* studio, Asset* asset) : _studio(studio), _asset(asset) {}

std::string const& AssetEditor::GetTabDisplayName() const {
  return _asset->GetName();
}

ImU32 AssetEditor::GetTabColor() const {
  if (_asset) {
    return _asset->GetColor();
  }
  return IM_COL32_BLACK_TRANS;
}

std::vector<AssetEditor::WindowDeclaration> AssetEditor::GetAuxiliaryWindows() const {
  return {};
}

void AssetEditor::ShowAuxiliaryWindows() {}

void AssetEditor::FocusWindowIfOpen(char const* name) {
  if (_studio->GetWindowVisible(name)) {
    ImGui::SetWindowFocus(name);
  }
}

Asset* AssetEditor::GetAsset() const {
  return _asset;
}

bool AssetEditor::IsReadOnly() const {
  return _asset->IsReadOnly();
}

bool AssetEditor::IsDirty() const {
  return _asset->IsDirty();
}

bool AssetEditor::Save() {
  if (_asset->Save()) {
    _asset->SetDirty(false);
    if (_undoStack.IsInitialized()) {
      _undoStack.SetCurrentAsSaved();
    }
    return true;
  }
  return false;
}

UndoStack& AssetEditor::GetUndoStack() {
  return _undoStack;
}

bool AssetEditor::CanUndoRedo() const {
  return true;
}

void AssetEditor::ApplySceneViewSettings(mochi_renderer::SceneViewSettings const&) {}

void AssetEditor::OnAppSettingsChanged(AppSettings const&) {}

std::string const& AssetEditor::GetReferencerName() const {
  return GetTabDisplayName();
}

void AssetEditor::ForEachReferencedPath(
    std::function<void(mochi::Path const&)> const& callback) const {
  callback(_asset->GetPath());
}

bool AssetEditor::RewriteReferencedPath(
    mochi::Path const& /*oldPath*/,
    mochi::Path const& /*newPath*/) {
  // The asset file path changed; previous undo entries refer to a now-stale path.
  // Reset the stack to avoid restoring inconsistent state.
  if (_undoStack.IsInitialized()) {
    _undoStack.Reset();
  }
  // Return true so the reference manager knows we're still tracking the asset.
  return true;
}

} // namespace superdex::studio
