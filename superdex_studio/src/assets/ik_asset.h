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
#include "assets/ik_config.h"

namespace superdex::studio {

class BotAsset;

class IKAsset : public Asset, public IAssetReferencer {
 public:
  // Callers must mark edits dirty and resync tracking when document references change.
  BotIKConfig& GetConfig();
  BotIKConfig const& GetConfig() const;
  mochi::Path GetOutputBotPath() const;
  mochi::Path GetInputBotPath() const;
  BotAsset* GetInputBotAsset() const;
  // Looks up or loads the current output bot; returns nullptr if it cannot be resolved.
  BotAsset* GetOutputBotAsset() const;
  std::string const& GetDiagnostic() const;

  ImU32 GetColor() const override;
  bool IsSavable() const override;
  bool Save() override;
  bool SupportsSaveAs() const override;
  bool SaveAs(mochi::Path const& path) override;
  // Success replaces edits and clears dirty state; failure preserves the config and dirty state.
  bool ReloadFromDisk() override;
  std::unique_ptr<AssetEditor> CreateEditor(SuperDexStudio* studio) override;

  std::string const& GetReferencerName() const override;
  void ForEachReferencedPath(
      std::function<void(mochi::Path const&)> const& callback) const override;
  bool RewriteReferencedPath(mochi::Path const& oldPath, mochi::Path const& newPath) override;

 protected:
  void OnRewritePath(
      mochi::Path const& oldPath,
      mochi::Path const& newPath,
      mochi_renderer::ResourceManager& resourceManager) override;

 private:
  friend class AssetManager;
  using Asset::Asset;
  static std::unique_ptr<IKAsset>
  Create(std::string const& name, mochi::Path const& path, AssetManager* manager);

  BotIKConfig _config;
  std::string _diagnostic;
};

} // namespace superdex::studio
