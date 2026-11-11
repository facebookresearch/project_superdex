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

#include "assets/asset_referencer.h"
#include "assets/asset_types.h"
#include "core/undo_stack.h"
#include "rendering/render_target.h"
#include "ui/imgui_widgets.h"

#include <mochi_renderer/resource.h>
#include <mochi_renderer/scene.h>

#include <array>
#include <string_view>

namespace superdex::studio {

//--------------------------------------------------------------------------------------------------
// FORWARDS / TYPES / HELPERS
//--------------------------------------------------------------------------------------------------

class SuperDexStudio;
class AssetManager;
class AssetBrowser;
class AssetEditor;

constexpr std::array<ImU32, kAssetTypeCount> kAssetTypeColors = {
    IM_COL32(134, 197, 62, 255), // RenderModel (green)
    IM_COL32(0, 128, 255, 255), // MochiModel (blue)
    IM_COL32(128, 128, 255, 255), // MochiPrefab (purple)
    IM_COL32(255, 165, 0, 255), // Bot (orange)
    IM_COL32(255, 96, 96, 255), // BotScene (red)
    IM_COL32(230, 110, 200, 255), // BotTask (pink)
    IM_COL32(0, 206, 209, 255), // CadModel (teal)
    IM_COL32(255, 215, 0, 255), // Tuner (gold)
    IM_COL32(80, 190, 150, 255), // IK (mint)
    IM_COL32(192, 192, 192, 192), // Unknown (gray)
};

inline constexpr ImU32 GetAssetTypeColor(AssetType type) {
  return kAssetTypeColors[static_cast<int>(type)];
}

//--------------------------------------------------------------------------------------------------
// ASSET
//--------------------------------------------------------------------------------------------------

class Asset {
 public:
  virtual ~Asset() = default;

  // Identity
  std::string const& GetName() const;
  mochi::Path const& GetPath() const;
  AssetType GetType() const;
  virtual ImU32 GetColor() const;
  virtual char const* GetTypeLabel() const;

  // Thumbnails
  virtual bool RendersThumbnail() const;
  virtual void StageThumbnailScene(mochi_renderer::Scene& scene);
  void MarkThumbnailDirty();
  void* GetThumbnailImage() const;

  // Asset State
  void SetDirty(bool dirty);
  bool IsDirty() const;
  bool IsReadOnly() const;
  void SetReadOnly(bool readOnly);
  virtual bool IsSavable() const;
  virtual bool Save();
  // Whether SaveAs can write this asset's current state to another file of the same type.
  [[nodiscard]] virtual bool SupportsSaveAs() const;
  // Writes the current state, including unsaved edits, to @p path. Leaves this asset's own file
  // unchanged.
  [[nodiscard]] virtual bool SaveAs(mochi::Path const& path);
  virtual bool ReloadFromDisk();

  // References
  int GetReferenceCount() const;
  std::vector<IAssetReferencer*> GetReferencers() const;

  // Editor
  virtual std::unique_ptr<AssetEditor> CreateEditor(SuperDexStudio*);

  // ImGui
  virtual void ShowAssetTileTooltipItems() const {}

 protected:
  friend class AssetManager;
  Asset(std::string const& name, mochi::Path const& path, AssetType type, AssetManager* manager);
  virtual void OnUnload(mochi_renderer::ResourceManager& resourceManager);
  virtual void OnRewritePath(
      mochi::Path const& oldPath,
      mochi::Path const& newPath,
      mochi_renderer::ResourceManager& resourceManager);
  // Rebuild any derived state after this asset's references were rewritten out-of-band -- currently
  // an asset replace / rename / move, and intended to also cover disk reloads in the future. The
  // default marks the thumbnail dirty so the browser tile refreshes; overrides may additionally
  // rebuild derived data. Open editors are refreshed separately via AssetEditor::Refresh.
  virtual void Refresh();

 protected:
  AssetManager* _manager = nullptr;
  std::string _name;
  mochi::Path _path;
  AssetType _type = AssetType::Unknown;
  bool _readOnly = false;
  bool _dirty = false;
  bool _thumbDirty = true;
  bool _thumbRendered = false;
  std::unique_ptr<RenderTarget> _thumbRenderTarget;
};

} // namespace superdex::studio
