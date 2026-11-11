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

#include "assets/ik_asset.h"

#include "app/app.h"
#include "assets/asset_manager.h"
#include "editors/ik_editor.h"

#include <mochi_core/utils/log.h>

#include <utility>

namespace superdex::studio {

std::unique_ptr<IKAsset>
IKAsset::Create(std::string const& name, mochi::Path const& path, AssetManager* manager) {
  BotIKConfig config;
  std::string diagnostic;
  if (!LoadIKConfig(path, config, diagnostic)) {
    MOCHI_LOG_ERROR("Failed to load IK configuration: %s", diagnostic.c_str());
    return nullptr;
  }
  auto asset = std::unique_ptr<IKAsset>(new IKAsset(name, path, AssetType::IK, manager));
  asset->_config = std::move(config);
  asset->_diagnostic = std::move(diagnostic);
  manager->RegisterReferencer(asset.get());
  return asset;
}

BotIKConfig& IKAsset::GetConfig() {
  return _config;
}

BotIKConfig const& IKAsset::GetConfig() const {
  return _config;
}

mochi::Path IKAsset::GetOutputBotPath() const {
  return ResolveIKBotPath(_path, _config.outputBot);
}

BotAsset* IKAsset::GetOutputBotAsset() const {
  mochi::Path const path = GetOutputBotPath();
  if (path.IsEmpty() || ClassifyAssetTypeByPath(path) != AssetType::Bot) {
    return nullptr;
  }
  if (auto* bot = _manager->FindAssetByPath<BotAsset>(path)) {
    return bot;
  }
  return _manager->LoadBotAsset(path);
}

mochi::Path IKAsset::GetInputBotPath() const {
  return ResolveIKBotPath(_path, _config.inputBot);
}

BotAsset* IKAsset::GetInputBotAsset() const {
  auto const path = GetInputBotPath();
  if (path.IsEmpty() || ClassifyAssetTypeByPath(path) != AssetType::Bot) {
    return nullptr;
  }
  if (auto* bot = _manager->FindAssetByPath<BotAsset>(path)) {
    return bot;
  }
  return _manager->LoadBotAsset(path);
}

std::string const& IKAsset::GetDiagnostic() const {
  return _diagnostic;
}

ImU32 IKAsset::GetColor() const {
  return GetAssetTypeColor(AssetType::IK);
}

bool IKAsset::IsSavable() const {
  return !IsReadOnly();
}

bool IKAsset::Save() {
  if (!IsSavable()) {
    MOCHI_LOG_ERROR("Attempting to save read-only IK configuration");
    return false;
  }
  if (!SaveIKConfig(_path, _config, _diagnostic)) {
    MOCHI_LOG_ERROR("Failed to save IK configuration: %s", _diagnostic.c_str());
    return false;
  }
  _manager->ResyncReferencer(this);
  return true;
}

bool IKAsset::SupportsSaveAs() const {
  return true;
}

bool IKAsset::SaveAs(mochi::Path const& path) {
  BotIKConfig config = _config;
  config.outputBot = MakeIKBotReference(path, GetOutputBotPath());
  config.inputBot = MakeIKBotReference(path, GetInputBotPath());
  if (!SaveIKConfig(path, config, _diagnostic)) {
    MOCHI_LOG_ERROR("Failed to save IK configuration: %s", _diagnostic.c_str());
    return false;
  }
  return true;
}

bool IKAsset::ReloadFromDisk() {
  BotIKConfig config;
  std::string diagnostic;
  if (!LoadIKConfig(_path, config, diagnostic)) {
    _diagnostic = std::move(diagnostic);
    MOCHI_LOG_ERROR("Failed to reload IK configuration: %s", _diagnostic.c_str());
    return false;
  }
  _config = std::move(config);
  _diagnostic = std::move(diagnostic);
  SetDirty(false);
  _manager->ResyncReferencer(this);
  if (auto* studio = _manager->GetStudio()) {
    studio->RefreshEditors({this});
  }
  return true;
}

std::unique_ptr<AssetEditor> IKAsset::CreateEditor(SuperDexStudio* studio) {
  return std::make_unique<IKEditor>(studio, this);
}

std::string const& IKAsset::GetReferencerName() const {
  return GetName();
}

void IKAsset::ForEachReferencedPath(std::function<void(mochi::Path const&)> const& callback) const {
  mochi::Path const path = GetOutputBotPath();
  if (!path.IsEmpty()) {
    callback(path);
  }
  auto const inputBotPath = GetInputBotPath();
  if (!inputBotPath.IsEmpty()) {
    callback(inputBotPath);
  }
}

bool IKAsset::RewriteReferencedPath(mochi::Path const& oldPath, mochi::Path const& newPath) {
  bool changed = false;
  if (GetOutputBotPath() == oldPath) {
    _config.outputBot = MakeIKBotReference(_path, newPath);
    changed = true;
  }
  if (GetInputBotPath() == oldPath) {
    _config.inputBot = MakeIKBotReference(_path, newPath);
    changed = true;
  }
  if (changed) {
    SetDirty(true);
  }
  return changed;
}

void IKAsset::OnRewritePath(
    mochi::Path const& oldPath,
    mochi::Path const& newPath,
    mochi_renderer::ResourceManager& /*resourceManager*/) {
  // The manager has already assigned _path, but output-bot-move notifications have not run yet.
  // Preserve the old resolved output bot so a folder move can subsequently rewrite it as well.
  mochi::Path const outputBotPath = ResolveIKBotPath(oldPath, _config.outputBot);
  std::string const reference = MakeIKBotReference(newPath, outputBotPath);
  if (std::string_view(_config.outputBot) != reference) {
    _config.outputBot = reference;
    SetDirty(true);
  }
  auto const inputBotPath = ResolveIKBotPath(oldPath, _config.inputBot);
  auto const inputBotReference = MakeIKBotReference(newPath, inputBotPath);
  if (std::string_view(_config.inputBot) != inputBotReference) {
    _config.inputBot = inputBotReference;
    SetDirty(true);
  }
}

} // namespace superdex::studio
