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

#include <superdex_asset_tuner/tuner_spec.h>

#include <string>

namespace superdex::studio {

class AssetManager;
class SceneAsset;
class MochiPrefabAsset;

// A .superdex_tuner file: a prefab or scene reference plus loads, probes and run settings.
class TunerAsset : public Asset, public IAssetReferencer {
 public:
  ~TunerAsset() override = default;

  // Asset
  char const* GetTypeLabel() const override;
  // The thumbnail is the referenced prefab's.
  bool RendersThumbnail() const override;
  void StageThumbnailScene(mochi_renderer::Scene& scene) override;
  bool IsSavable() const override;
  bool Save() override;
  std::unique_ptr<AssetEditor> CreateEditor(SuperDexStudio* studio) override;

  // IAssetReferencer
  std::string const& GetReferencerName() const override;
  void ForEachReferencedPath(
      std::function<void(mochi::Path const&)> const& callback) const override;
  bool RewriteReferencedPath(mochi::Path const& oldPath, mochi::Path const& newPath) override;

  superdex::tuner::TunerSpec const& GetSpec() const {
    return _spec;
  }
  superdex::tuner::TunerSpec& GetSpec() {
    return _spec;
  }
  // The referenced prefab, looked up by path on every call (asset pointers do not survive an
  // unload/reload). Null if it failed to load.
  MochiPrefabAsset* GetPrefabAsset() const;
  mochi::Path const& GetPrefabPath() const {
    return _prefabPath;
  }
  // The referenced scene, looked up by path on every call. Null without one, or if it failed to
  // load.
  SceneAsset* GetSceneAsset() const;
  // Empty without a scene.
  mochi::Path const& GetScenePath() const {
    return _scenePath;
  }
  // Empty without a task.
  mochi::Path const& GetTaskPath() const {
    return _taskPath;
  }
  // Points the tuner at a prefab or a scene, replacing the one it has, and loads it. Replacing a
  // scene clears its task and the bots to build. The spec stores the path relative to the tuner
  // ("./...") when it is in or below the tuner's folder, else relative to the assets root when it
  // is under it, else absolute.
  void SetPrefabOrScene(mochi::Path const& path);
  // Sets the scene's task, stored like SetPrefabOrScene's path; an empty path clears it.
  void SetTaskPath(mochi::Path const& path);
  // Follows the spec's prefab, scene and task paths after they changed in place (an undo), loading
  // what they reference. Returns true if any changed.
  bool RefreshReferencesFromSpec();

 private:
  friend class AssetManager;
  using Asset::Asset;
  static std::unique_ptr<TunerAsset>
  Create(std::string const& name, mochi::Path const& path, AssetManager* manager);
  // How the spec stores `path`; see SetPrefabOrScene.
  [[nodiscard]] std::string ToSpecPath(mochi::Path const& path) const;

  superdex::tuner::TunerSpec _spec;
  mochi::Path _prefabPath;
  mochi::Path _scenePath;
  mochi::Path _taskPath;
};

} // namespace superdex::studio
