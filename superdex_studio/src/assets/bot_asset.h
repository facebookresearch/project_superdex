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

#include <mochi_renderer/resource.h>
#include <mochi_renderer/scene.h>

#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/bot_utils.h>

namespace superdex::studio {

class AssetManager;

class BotAsset : public Asset, public IAssetReferencer {
 public:
  ~BotAsset() override = default;

  // Asset
  unsigned int GetColor() const override;
  char const* GetTypeLabel() const override;
  bool RendersThumbnail() const override;
  void StageThumbnailScene(mochi_renderer::Scene& scene) override;
  bool IsSavable() const override;
  bool Save() const override;
  void ShowAssetTileTooltipItems() const override;
  std::unique_ptr<AssetEditor> CreateEditor(SuperDexStudio* studio) override;

  // IAssetReferencer
  std::string const& GetReferencerName() const override;
  void ForEachReferencedPath(
      std::function<void(mochi::Path const&)> const& callback) const override;
  bool RewriteReferencedPath(mochi::Path const& oldPath, mochi::Path const& newPath) override;

  superdex::robotics::BotFileType GetBotFileType() const;
  superdex::robotics::BotPrefab const& GetBotPrefab() const;
  superdex::robotics::BotPrefab& GetBotPrefab();
  superdex::robotics::ModBotPrefab const& GetModBotPrefab() const;
  superdex::robotics::ModBotPrefab& GetModBotPrefab();
  mochi::DynamicString const& GetBotName() const;
  mochi::DynamicString& GetBotName();
  mochi::DynamicString const& GetBotHash() const;
  void Rebuild(bool* buildOk = nullptr, bool* validateOk = nullptr);
  void Rebuild(
      superdex::robotics::IBotLoader const& loader,
      bool* buildOk = nullptr,
      bool* validateOk = nullptr);
  superdex::robotics::ValidateResults const& GetValidateResults() const;
  bool IsBuildOk() const;
  bool IsValidateOk() const;

 private:
  friend class AssetManager;
  using Asset::Asset;
  static std::unique_ptr<BotAsset>
  Create(std::string const& name, mochi::Path const& path, AssetManager* manager);

 private:
  superdex::robotics::BotFileType _botType = superdex::robotics::BotFileType::BotPrefab;
  bool _isArchive = false;
  superdex::robotics::BotPrefab _botPrefab;
  superdex::robotics::ModBotPrefab _modBotPrefab;
  superdex::robotics::ValidateResults _lastValidateResults;
  bool _lastBuildOk = true;
  mutable mochi::DynamicString _lastSavedHash;
};

} // namespace superdex::studio
