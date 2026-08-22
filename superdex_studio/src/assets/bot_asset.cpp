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

#include "assets/bot_asset.h"
#include "assets/asset_manager.h"

#include <mochi_renderer/resource_manager.h>

#include <superdex_robotics/core/loader.h>
#include <superdex_robotics/utils/archive_utils.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <variant>

#include "editors/bot_editor.h"

namespace superdex::studio {

std::unique_ptr<BotAsset>
BotAsset::Create(std::string const& name, mochi::Path const& path, AssetManager* manager) {
  mochi::ErrorLog error;
  superdex::robotics::FileBotLoader loader;

  // Path 1: bot archive (.superdex_bot_archive)
  if (superdex::robotics::IsBotArchivePath(path.ToString())) {
    auto extractedDir = superdex::robotics::ExtractBotArchiveToCache(path.ToString(), error);
    auto targetPath = superdex::robotics::GetExtractedBotArchiveTarget(extractedDir, error);
    // first load the bot from the unzipped target path so that it is also a valid asset
    manager->LoadBotAsset(targetPath);
    // now continue to internalize the archived bot as a valid asset too
    auto botPrefab = superdex::robotics::LoadBotPrefab(targetPath, loader, false, error);
    if (!error.IsOK()) {
      MOCHI_LOG_ERROR("Failed to load bot from: %s", path.ToString().c_str());
      return nullptr;
    }
    for (auto& link : botPrefab.links) {
      if (!link.renderModelFile.empty()) {
        if (auto* a = manager->LoadRenderModelAsset(link.renderModelFile)) {
          a->SetReadOnly(true);
        }
      }
      if (!link.shapeFile.empty()) {
        if (auto* a = manager->LoadMochiModelAsset(link.shapeFile)) {
          a->SetReadOnly(true);
        }
      }
    }
    auto asset = std::unique_ptr<BotAsset>(new BotAsset(name, path, AssetType::Bot, manager));
    asset->_botType = superdex::robotics::BotFileType::BotPrefab;
    asset->_botPrefab = std::move(botPrefab);
    asset->_isArchive = true;
    asset->SetReadOnly(true);
    asset->_lastSavedHash = superdex::robotics::HashBotFile(targetPath, mochi::ErrorLog{});
    manager->RegisterReferencer(asset.get());
    return asset;
  }

  // Path 2: Mod Bot (.superdex_bot)
  auto type = loader.GetBotFileType(path.ToString(), error);
  if (type == superdex::robotics::BotFileType::ModBotPrefab) {
    auto modBotPrefab = loader.LoadModBotPrefab(path.ToString(), error);
    if (!error.IsOK()) {
      MOCHI_LOG_ERROR("Failed to load bot from: %s", path.ToString().c_str());
      return nullptr;
    }
    if (!modBotPrefab.base.empty()) {
      manager->LoadBotAsset(modBotPrefab.base);
    }
    for (auto& mod : modBotPrefab.modifications) {
      std::visit(
          [manager](auto&& m) {
            using T = std::decay_t<decltype(m)>;
            if constexpr (
                std::is_same_v<T, superdex::robotics::AttachBot> ||
                std::is_same_v<T, superdex::robotics::ReplaceLinkWithBot>) {
              manager->LoadBotAsset(m.path);
            } else if constexpr (
                std::is_same_v<T, superdex::robotics::AttachLink> ||
                std::is_same_v<T, superdex::robotics::ReplaceLink>) {
              if (!m.link.renderModelFile.empty()) {
                manager->LoadRenderModelAsset(m.link.renderModelFile);
              }
              if (!m.link.shapeFile.empty()) {
                manager->LoadMochiModelAsset(m.link.shapeFile);
              }
            }
          },
          mod);
    }
    auto botPrefab = superdex::robotics::BuildBot(modBotPrefab, loader, false, error);
    if (!error.IsOK()) {
      MOCHI_LOG_ERROR("Failed to build bot from: %s", path.ToString().c_str());
      return nullptr;
    }
    auto asset = std::unique_ptr<BotAsset>(new BotAsset(name, path, AssetType::Bot, manager));
    asset->_botType = superdex::robotics::BotFileType::ModBotPrefab;
    asset->_botPrefab = std::move(botPrefab);
    asset->_modBotPrefab = std::move(modBotPrefab);
    asset->_lastSavedHash = superdex::robotics::HashBotFile(path.ToString(), mochi::ErrorLog{});
    manager->RegisterReferencer(asset.get());
    return asset;
  }

  // Path 3. Regular bot (.superdex_bot)
  auto botPrefab = loader.LoadBotPrefab(path.ToString(), error);
  if (!error.IsOK()) {
    MOCHI_LOG_ERROR("Failed to load bot from: %s", path.ToString().c_str());
    return nullptr;
  }
  for (auto& link : botPrefab.links) {
    if (!link.renderModelFile.empty()) {
      manager->LoadRenderModelAsset(link.renderModelFile);
    }
    if (!link.shapeFile.empty()) {
      manager->LoadMochiModelAsset(link.shapeFile);
    }
  }
  auto asset = std::unique_ptr<BotAsset>(new BotAsset(name, path, AssetType::Bot, manager));
  asset->_botType = superdex::robotics::BotFileType::BotPrefab;
  asset->_botPrefab = std::move(botPrefab);
  asset->_lastSavedHash = superdex::robotics::HashBotFile(path.ToString(), mochi::ErrorLog{});
  manager->RegisterReferencer(asset.get());
  return asset;
}

unsigned int BotAsset::GetColor() const {
  return Asset::GetColor();
}

char const* BotAsset::GetTypeLabel() const {
  return _isArchive                                               ? "Bot Archive"
      : _botType == superdex::robotics::BotFileType::ModBotPrefab ? "Mod Bot"
                                                                  : "Bot";
}

bool BotAsset::RendersThumbnail() const {
  return true;
}

superdex::robotics::BotPrefab const& BotAsset::GetBotPrefab() const {
  return _botPrefab;
}

superdex::robotics::BotPrefab& BotAsset::GetBotPrefab() {
  return _botPrefab;
}

superdex::robotics::ModBotPrefab const& BotAsset::GetModBotPrefab() const {
  return _modBotPrefab;
}

superdex::robotics::ModBotPrefab& BotAsset::GetModBotPrefab() {
  return _modBotPrefab;
}

mochi::DynamicString const& BotAsset::GetBotName() const {
  if (_botType == superdex::robotics::BotFileType::ModBotPrefab) {
    return _modBotPrefab.name;
  }
  return _botPrefab.name;
}

mochi::DynamicString& BotAsset::GetBotName() {
  if (_botType == superdex::robotics::BotFileType::ModBotPrefab) {
    return _modBotPrefab.name;
  }
  return _botPrefab.name;
}

mochi::DynamicString const& BotAsset::GetBotHash() const {
  return _lastSavedHash;
}

void BotAsset::Rebuild(bool* buildOk, bool* validateOk) {
  superdex::robotics::FileBotLoader loader;
  Rebuild(loader, buildOk, validateOk);
}

void BotAsset::Rebuild(
    superdex::robotics::IBotLoader const& loader,
    bool* buildOk,
    bool* validateOk) {
  mochi::ErrorLog e;
  if (_botType == superdex::robotics::BotFileType::ModBotPrefab) {
    // Build into a local so a failing build does not clobber the last good
    // prefab. Only commit on success; keep the retained prefab otherwise.
    superdex::robotics::BotPrefab built =
        superdex::robotics::BuildBot(_modBotPrefab, loader, false, e);
    if (e.IsOK()) {
      _botPrefab = std::move(built);
    }
  } else {
    superdex::robotics::RebuildBotData(_botPrefab, e);
  }
  _lastBuildOk = e.IsOK();
  // Refresh validation of whatever prefab is currently retained. Use a fresh
  // error so a failed build above does not suppress validation of the good prefab.
  // Issues are surfaced in the UI (asset browser badge + editor issue list), so they are not
  // also logged here.
  mochi::Error validateError;
  _lastValidateResults = {};
  _lastValidateResults.suppressWarnings = true;
  superdex::robotics::Validate(_botPrefab, &_lastValidateResults, validateError);
  if (buildOk != nullptr) {
    *buildOk = _lastBuildOk;
  }
  if (validateOk != nullptr) {
    *validateOk = IsValidateOk();
  }
}

superdex::robotics::ValidateResults const& BotAsset::GetValidateResults() const {
  return _lastValidateResults;
}

bool BotAsset::IsValidateOk() const {
  return _lastValidateResults.botIssues.empty();
}

bool BotAsset::IsBuildOk() const {
  return _lastBuildOk;
}

void BotAsset::StageThumbnailScene(mochi_renderer::Scene& scene) {
  SceneStage stage(_manager->GetStudio(), "BotThumbnailStage");
  stage.BindRenderScene(&scene);
  stage.StageBot(GetBotPrefab(), StageType::RenderModelOnly);
}

bool BotAsset::IsSavable() const {
  return !IsReadOnly();
}

bool BotAsset::Save() const {
  mochi::ErrorLog e;
  if (IsReadOnly()) {
    MOCHI_LOG_ERROR("Attempting to save read-only Mochi Bot");
    return false;
  }
  auto pathStr = _path.ToString();
  if (_botType == superdex::robotics::BotFileType::ModBotPrefab) {
    superdex::robotics::SaveToFile(_modBotPrefab, pathStr, e);
  } else {
    superdex::robotics::SaveToFile(_botPrefab, pathStr, e);
  }
  if (e.IsOK()) {
    _lastSavedHash = superdex::robotics::HashBotFile(pathStr, mochi::ErrorLog{});
    return true;
  }
  return false;
}

void BotAsset::ShowAssetTileTooltipItems() const {
  ImGui::Text("Bot Hash: %s", GetBotHash().c_str());
}

std::unique_ptr<AssetEditor> BotAsset::CreateEditor(SuperDexStudio* studio) {
  return std::make_unique<BotEditor>(studio, this);
}

std::string const& BotAsset::GetReferencerName() const {
  return GetName();
}

superdex::robotics::BotFileType BotAsset::GetBotFileType() const {
  return _botType;
}

//--------------------------------------------------------------------------------------------------
// REFERENCE TRACKING
//--------------------------------------------------------------------------------------------------

namespace {

// Visit every referenced path in a BotPrefab.
template <typename Fn>
void ForEachPathInBotPrefab(superdex::robotics::BotPrefab const& prefab, Fn const& fn) {
  for (auto const& link : prefab.links) {
    if (!link.shapeFile.empty()) {
      fn(mochi::Path{link.shapeFile.c_str()});
    }
    if (!link.renderModelFile.empty()) {
      fn(mochi::Path{link.renderModelFile.c_str()});
    }
  }
}

// Visit every referenced path in a ModBotPrefab.
template <typename Fn>
void ForEachPathInModBotPrefab(superdex::robotics::ModBotPrefab const& params, Fn const& fn) {
  if (!params.base.empty()) {
    fn(mochi::Path{params.base.c_str()});
  }
  for (auto const& mod : params.modifications) {
    std::visit(
        [&](auto const& m) {
          using T = std::decay_t<decltype(m)>;
          if constexpr (
              std::is_same_v<T, superdex::robotics::AttachBot> ||
              std::is_same_v<T, superdex::robotics::ReplaceLinkWithBot>) {
            if (!m.path.empty()) {
              fn(mochi::Path{m.path.c_str()});
            }
          } else if constexpr (
              std::is_same_v<T, superdex::robotics::AttachLink> ||
              std::is_same_v<T, superdex::robotics::ReplaceLink>) {
            if (!m.link.shapeFile.empty()) {
              fn(mochi::Path{m.link.shapeFile.c_str()});
            }
            if (!m.link.renderModelFile.empty()) {
              fn(mochi::Path{m.link.renderModelFile.c_str()});
            }
          }
        },
        mod);
  }
}

// Rewrite a single path field if it matches `oldPath` after normalization.
bool MaybeRewrite(
    mochi::DynamicString& field,
    mochi::Path const& oldPath,
    mochi::Path const& newPath) {
  if (field.empty()) {
    return false;
  }
  if (mochi::Path{field.c_str()} == oldPath) {
    field = newPath.ToString();
    return true;
  }
  return false;
}

} // namespace

void BotAsset::ForEachReferencedPath(
    std::function<void(mochi::Path const&)> const& callback) const {
  ForEachPathInBotPrefab(_botPrefab, callback);
  if (_botType == superdex::robotics::BotFileType::ModBotPrefab) {
    ForEachPathInModBotPrefab(_modBotPrefab, callback);
  }
}

bool BotAsset::RewriteReferencedPath(mochi::Path const& oldPath, mochi::Path const& newPath) {
  bool changed = false;
  for (auto& link : _botPrefab.links) {
    changed |= MaybeRewrite(link.shapeFile, oldPath, newPath);
    changed |= MaybeRewrite(link.renderModelFile, oldPath, newPath);
  }
  if (_botType == superdex::robotics::BotFileType::ModBotPrefab) {
    changed |= MaybeRewrite(_modBotPrefab.base, oldPath, newPath);
    for (auto& mod : _modBotPrefab.modifications) {
      std::visit(
          [&](auto& m) {
            using T = std::decay_t<decltype(m)>;
            if constexpr (
                std::is_same_v<T, superdex::robotics::AttachBot> ||
                std::is_same_v<T, superdex::robotics::ReplaceLinkWithBot>) {
              changed |= MaybeRewrite(m.path, oldPath, newPath);
            } else if constexpr (
                std::is_same_v<T, superdex::robotics::AttachLink> ||
                std::is_same_v<T, superdex::robotics::ReplaceLink>) {
              changed |= MaybeRewrite(m.link.shapeFile, oldPath, newPath);
              changed |= MaybeRewrite(m.link.renderModelFile, oldPath, newPath);
            }
          },
          mod);
    }
  }
  if (changed) {
    SetDirty(true);
  }
  return changed;
}

} // namespace superdex::studio
