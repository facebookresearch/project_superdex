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

#include "assets/asset.h"
#include "assets/asset_referencer.h"
#include "assets/bot_asset.h"
#if MOCHI_INTERNAL
#include "assets/bot_scene_asset.h"
#endif
#include "assets/cad_model_asset.h"
#include "assets/mochi_model_asset.h"
#include "assets/mochi_prefab_asset.h"
#include "assets/render_model_asset.h"

#include <functional>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace mochi_renderer {
class IBL;
} // namespace mochi_renderer

namespace superdex::studio {

class SuperDexStudio;
class Renderer;

class AssetManager {
 public:
  static std::unique_ptr<AssetManager> Create(SuperDexStudio* studio);

  AssetManager(AssetManager const&) = delete;
  AssetManager& operator=(AssetManager const&) = delete;
  AssetManager(AssetManager&&) = delete;
  AssetManager& operator=(AssetManager&&) = delete;
  ~AssetManager() = default;

  //------------------------------------------------------------------------------------------------
  // Load / Unload Assets
  //------------------------------------------------------------------------------------------------
  Asset* LoadAsset(mochi::Path const& path);
  RenderModelAsset* LoadRenderModelAsset(mochi::Path const& path);
  CadModelAsset* LoadCadModelAsset(mochi::Path const& path);
  MochiModelAsset* LoadMochiModelAsset(mochi::Path const& path);
  MochiPrefabAsset* LoadMochiPrefabAsset(mochi::Path const& path);
  BotAsset* LoadBotAsset(mochi::Path const& path);
#if MOCHI_INTERNAL
  BotSceneAsset* LoadBotSceneAsset(mochi::Path const& path);
#endif
  bool UnloadAssetByPath(mochi::Path const& path);
  bool UnloadAssets(std::set<mochi::Path> const& paths);
  bool UnloadAllAssets();

  //------------------------------------------------------------------------------------------------
  // Find / Iterate Assets
  //------------------------------------------------------------------------------------------------
  Asset* FindAssetByPath(mochi::Path const& path, bool silent = true) const;
  template <typename TAsset>
  TAsset* FindAssetByPath(mochi::Path const& path, bool silent = true) const {
    return dynamic_cast<TAsset*>(FindAssetByPath(path, silent));
  }
  void ForEachAsset(std::function<void(Asset*, mochi::Path const&)> const& callback) const;
  /// All currently loaded assets whose type matches @p type.
  std::vector<Asset*> GetAllAssetsOfType(AssetType type) const;

  //------------------------------------------------------------------------------------------------
  // Thumbnails
  //------------------------------------------------------------------------------------------------
  void SetThumbnailIbl(mochi_renderer::IBL* ibl);
  void RenderAssetThumbnails(Renderer& renderer, int maxThumbnails = 1);

  /// Render @p asset's thumbnail into @p target using the shared thumbnail scene: stages the
  /// asset's geometry, auto-frames it with the ortho thumbnail camera, and renders with a fully
  /// transparent clear color (so the background alpha is 0). The scene is cleared again on exit.
  /// The render fills @p target at whatever size it was created with, so the same framing used for
  /// browser tiles can be reused to render an arbitrary-resolution export.
  void RenderAssetToTarget(Asset& asset, RenderTarget& target, Renderer& renderer);

  //------------------------------------------------------------------------------------------------
  // Asset Path Reference Tracking
  //------------------------------------------------------------------------------------------------

  /// Number of distinct referencers that reference path.
  int GetPathReferenceCount(mochi::Path const& path) const;
  /// Rewrite an asset's path and notify all referencers.
  void RewriteAssetPath(mochi::Path const& oldPath, mochi::Path const& newPath);
  /// Batch rewrite several assets' paths and notify all referencers.
  void RewriteAssetPaths(mochi::Span<std::pair<mochi::Path, mochi::Path>> const& moves);
  /// Repoint external referencers of oldPath to reference newPath instead. Editors that represent
  /// the asset at oldPath are left untouched, so they keep referencing (and blocking deletion of)
  /// that asset rather than being silently redirected onto the replacement.
  void RepointReferences(mochi::Path const& oldPath, mochi::Path const& newPath);
  /// All referencers that reference @p path.
  std::vector<IAssetReferencer*> GetReferencersToPath(mochi::Path const& path) const;
  /// All paths referenced from @p asset.
  std::vector<mochi::Path> GetPathsFromReferencer(IAssetReferencer* asset) const;
  /// Register a referencer for path rewrite notifications.
  void RegisterReferencer(IAssetReferencer* referencer);
  /// Unregister a referencer from path rewrite notifications. No-op if not registered.
  void UnregisterReferencer(IAssetReferencer* referencer);
  /// Re-scan the referencer's references and update the index incrementally.
  void ResyncReferencer(IAssetReferencer* referencer);

  //------------------------------------------------------------------------------------------------
  // Studio accessors
  //------------------------------------------------------------------------------------------------
  SuperDexStudio* GetStudio() const;

 private:
  AssetManager(SuperDexStudio* studio);
  Asset* RegisterAsset(mochi::Path const& path, std::unique_ptr<Asset> asset);
  void RewriteAssetPathInMap(mochi::Path const& oldPath, mochi::Path const& newPath);
  std::vector<IAssetReferencer*> NotifyPathMoved(
      mochi::Path const& oldPath,
      mochi::Path const& newPath,
      std::function<bool(IAssetReferencer*)> const& shouldNotify = {});
  std::vector<IAssetReferencer*> NotifyPathsMoved(
      mochi::Span<std::pair<mochi::Path, mochi::Path>> const& moves);
  void RefreshReferencers(std::vector<IAssetReferencer*> const& referencers);

 private:
  SuperDexStudio* _studio = nullptr;
  std::map<IAssetReferencer*, std::set<mochi::Path>> _forwardRefs;
  std::map<mochi::Path, std::set<IAssetReferencer*>> _reverseRefs;
  std::map<mochi::Path, std::unique_ptr<Asset>> _assets;
  std::unique_ptr<mochi_renderer::Scene> _thumbnailScene;
};

} // namespace superdex::studio
