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

#include "assets/mochi_prefab_asset.h"
#include "app/app.h"
#include "assets/asset_manager.h"
#include "editors/mochi_prefab_editor.h"
#include "rendering/scene_stage.h"

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/path.h>

#include <superdex_robotics/utils/archive_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_physics/cpp_api/mochi_actor.h>
#include <mochi_physics/mochi_physics.h>

#include <filesystem>

namespace superdex::studio {

// Compute the root directory used as `rootForRelativePath` when resolving non-"./" path
// references: the nearest `.superdex_root`, matching BotSceneEditor. Every asset tree holding
// prefabs is expected to carry that marker at its root. There is deliberately no fallback root --
// one would resolve rootless prefabs against an unrelated tree, so a missing marker is reported
// instead.
static std::string ComputeAssetsRoot(mochi::Path const& prefabPath) {
  if (auto botsRoot = superdex::robotics::FindBotsRoot(prefabPath.ToString())) {
    return botsRoot->string();
  }
  MOCHI_LOG_WARNING(
      "No .superdex_root above '%s'; its root-relative asset references will not resolve. Add a "
      ".superdex_root at the root of that asset tree.",
      prefabPath.ToString().c_str());
  return {};
}

// Visit every mutable path field in the prefab's "path field set" (rigid actor
// shape/render model, articulated link shape/render model, skin shape/render model,
// and nested prefab reference paths). Mirrors the const visitor `ForEachPathInPrefab`
// but yields mutable references, so it can be reused by the resolve and relativize
// passes. Does not descend into loaded nested `ScenePrefab`s — those keep their own
// (relative) representation and are resolved transiently at stage/load time.
template <typename Fn>
static void ForEachPathFieldInPrefab(mochi::prefab::ScenePrefab& prefab, Fn const& fn) {
  for (auto& a : prefab.actors.rigid) {
    fn(a.shapeFile);
    fn(a.renderModelFile);
  }
  for (auto& a : prefab.actors.soft) {
    // Only the shape file is staged/loaded for soft actors; renderModel* is deferred.
    fn(a.shapeFile);
  }
  for (auto& art : prefab.actors.articulated) {
    for (auto& link : art.links) {
      fn(link.shapeFile);
      fn(link.renderModelFile);
    }
    if (art.skin.has_value()) {
      fn(art.skin->shapeFile);
      fn(art.skin->renderModelFile);
    }
  }
  for (auto& nested : prefab.prefabs) {
    fn(nested.path);
  }
}

// Resolve every top-level path field to an absolute path, in place, using the exact
// same resolution `PreloadReferencedAssets` uses to register assets. This keeps the
// stored strings matching the `AssetManager` keys so widget lookups hit, and makes
// downstream `GetPrefabFullPath` calls idempotent no-ops.
static void ResolvePrefabPathsToAbsolute(
    mochi::prefab::ScenePrefab& prefab,
    std::string const& assetsRoot) {
  std::string_view const sourceFilePath = prefab.sourceFilePath.has_value()
      ? std::string_view(*prefab.sourceFilePath)
      : std::string_view();
  ForEachPathFieldInPrefab(prefab, [&](mochi::DynamicString& field) {
    if (field.empty()) {
      return;
    }
    field = mochi::prefab::GetPrefabFullPath(field, assetsRoot, sourceFilePath);
  });
}

// Stateless inverse of `GetPrefabFullPath`: turn an absolute field back into the
// most compact on-disk relative spelling, with precedence matching the forward
// function. Returns the field unchanged when it is empty, already relative, or
// cannot be relativized.
static mochi::DynamicString MakePrefabPathRelative(
    mochi::DynamicString const& field,
    std::string const& assetsRoot,
    std::optional<mochi::DynamicString> const& sourceFilePath) {
  if (field.empty() || !mochi::path::IsAbsolutePath(field)) {
    return field;
  }

  auto hasParentTraversal = [](std::filesystem::path const& p) {
    for (auto const& part : p) {
      if (part == "..") {
        return true;
      }
    }
    return false;
  };

  // Case A: prefab-directory relative (the "./" form). Purely lexical so it
  // round-trips exactly against the stored absolute value.
  if (sourceFilePath.has_value() && !sourceFilePath->empty()) {
    auto const prefabDir = std::filesystem::path(sourceFilePath->c_str()).parent_path();
    auto const rel = std::filesystem::path(field.c_str()).lexically_relative(prefabDir);
    if (!rel.empty()) {
      return mochi::DynamicString("./" + rel.generic_string());
    }
  }

  // Case B: assets-root relative (plain).
  if (!assetsRoot.empty()) {
    auto const rel = mochi::path::GetRelativePath(field, assetsRoot);
    if (!rel.empty() && !mochi::path::IsAbsolutePath(rel) &&
        !hasParentTraversal(std::filesystem::path(rel))) {
      return mochi::DynamicString(rel);
    }
  }

  // Case C: leave absolute (prefabs permit absolute paths).
  return field;
}

// Re-relativize every top-level path field in place, the inverse of
// `ResolvePrefabPathsToAbsolute`.
static void RelativizePrefabPaths(
    mochi::prefab::ScenePrefab& prefab,
    std::string const& assetsRoot) {
  ForEachPathFieldInPrefab(prefab, [&](mochi::DynamicString& field) {
    field = MakePrefabPathRelative(field, assetsRoot, prefab.sourceFilePath);
  });
}

// Preload sibling assets for every file referenced by the prefab. Fields are
// absolute in memory (see `ResolvePrefabPathsToAbsolute`), so they are used directly.
static void PreloadReferencedAssets(
    mochi::prefab::ScenePrefab const& prefab,
    AssetManager* manager) {
  auto loadModelAndRenderModel = [&](mochi::DynamicString const& shapeFile,
                                     mochi::DynamicString const& renderModelFile) {
    if (!shapeFile.empty()) {
      manager->LoadMochiModelAsset(shapeFile);
    }
    if (!renderModelFile.empty()) {
      manager->LoadRenderModelAsset(renderModelFile);
    }
  };

  for (auto const& a : prefab.actors.rigid) {
    loadModelAndRenderModel(a.shapeFile, a.renderModelFile);
  }
  for (auto const& a : prefab.actors.soft) {
    // Only the shape file is loaded for soft actors; renderModel* is deferred.
    if (!a.shapeFile.empty()) {
      manager->LoadMochiModelAsset(a.shapeFile);
    }
  }
  if (!prefab.actors.softSkinned.empty()) {
    MOCHI_LOG_WARNING(
        "MochiPrefabAsset: softSkinned actors are not supported by the asset loader.");
  }
  for (auto const& art : prefab.actors.articulated) {
    for (auto const& link : art.links) {
      loadModelAndRenderModel(link.shapeFile, link.renderModelFile);
    }
    if (art.skin.has_value()) {
      loadModelAndRenderModel(art.skin->shapeFile, art.skin->renderModelFile);
    }
  }
  for (auto const& nested : prefab.prefabs) {
    if (!nested.path.empty()) {
      // Nested prefabs can be either .mochi_scene prefabs or bot archives
      if (superdex::robotics::IsBotArchivePath(nested.path) ||
          superdex::robotics::IsBotPath(nested.path)) {
        manager->LoadBotAsset(nested.path);
      } else {
        manager->LoadMochiPrefabAsset(nested.path);
      }
    }
  }
}

// Rewrite a single (absolute) path field if it matches `oldPath`.
static bool MaybeRewriteResolved(
    mochi::DynamicString& field,
    mochi::Path const& oldPath,
    mochi::Path const& newPath) {
  if (field.empty()) {
    return false;
  }
  if (mochi::Path{field} == oldPath) {
    field = newPath.ToString();
    return true;
  }
  return false;
}

template <typename Fn>
static void ForEachPathInPrefab(mochi::prefab::ScenePrefab const& prefab, Fn const& fn) {
  auto visit = [&](mochi::DynamicString const& s) {
    if (!s.empty()) {
      fn(mochi::Path{s});
    }
  };
  for (auto const& a : prefab.actors.rigid) {
    visit(a.shapeFile);
    visit(a.renderModelFile);
  }
  for (auto const& a : prefab.actors.soft) {
    visit(a.shapeFile);
  }
  for (auto const& art : prefab.actors.articulated) {
    for (auto const& link : art.links) {
      visit(link.shapeFile);
      visit(link.renderModelFile);
    }
    if (art.skin.has_value()) {
      visit(art.skin->shapeFile);
      visit(art.skin->renderModelFile);
    }
  }
  for (auto const& nested : prefab.prefabs) {
    visit(nested.path);
  }
}

std::unique_ptr<MochiPrefabAsset>
MochiPrefabAsset::Create(std::string const& name, mochi::Path const& path, AssetManager* manager) {
  mochi::ErrorLog error;
  auto prefab = mochi::prefab::ShallowLoadFromFile(path.ToString(), error);
  if (!error.IsOK()) {
    MOCHI_LOG_ERROR("Failed to load MochiPrefabAsset: %s", path.ToString().c_str());
    return nullptr;
  }

  auto assetsRoot = ComputeAssetsRoot(path);

  mochi::prefab::LoadNestedPrefabs(prefab, assetsRoot, error);
  if (!error.IsOK()) {
    MOCHI_LOG_ERROR(
        "Failed to load nested prefabs for MochiPrefabAsset: %s", path.ToString().c_str());
    return nullptr;
  }

  // Resolve top-level path fields to absolute in memory (relativized again on Save).
  ResolvePrefabPathsToAbsolute(prefab, assetsRoot);

  // Preload sibling assets the prefab references.
  PreloadReferencedAssets(prefab, manager);

  auto asset = std::unique_ptr<MochiPrefabAsset>(
      new MochiPrefabAsset(name, path, AssetType::MochiPrefab, manager));
  asset->_prefab = std::move(prefab);
  asset->_assetsRoot = std::move(assetsRoot);

  manager->RegisterReferencer(asset.get());
  return asset;
}

unsigned int MochiPrefabAsset::GetColor() const {
  return Asset::GetColor();
}

char const* MochiPrefabAsset::GetTypeLabel() const {
  return _prefab.scene.has_value() ? "Scene Prefab" : "Prefab";
}

bool MochiPrefabAsset::RendersThumbnail() const {
  return true;
}

void MochiPrefabAsset::StageThumbnailScene(mochi_renderer::Scene& scene) {
  SceneStage stage(_manager->GetStudio(), "MochiPrefabThumbnailStage");
  stage.BindRenderScene(&scene);
  stage.StagePrefab(GetPrefab(), _assetsRoot, StageType::RenderModelFallbackToMochiModel);
}

bool MochiPrefabAsset::IsSavable() const {
  return !IsReadOnly();
}

bool MochiPrefabAsset::Save() const {
  if (IsReadOnly()) {
    MOCHI_LOG_ERROR("Attempting to save read-only MochiPrefabAsset");
    return false;
  }
  mochi::ErrorLog error;
  // In-memory fields are absolute; write a relativized copy so the on-disk prefab
  // keeps its relative path contract. `_prefab` stays absolute.
  mochi::prefab::ScenePrefab prefabToSave = _prefab;
  RelativizePrefabPaths(prefabToSave, _assetsRoot);
  mochi::prefab::SaveToJsonFile(prefabToSave, _path.ToString(), error);
  return error.IsOK();
}

void MochiPrefabAsset::OnRewritePath(
    mochi::Path const& /*oldPath*/,
    mochi::Path const& newPath,
    mochi_renderer::ResourceManager& /*resourceManager*/) {
  _prefab.sourceFilePath = mochi::DynamicString(newPath.ToString());
  _assetsRoot = ComputeAssetsRoot(newPath);
  SetDirty(true);
}

std::unique_ptr<AssetEditor> MochiPrefabAsset::CreateEditor(SuperDexStudio* studio) {
  return std::make_unique<MochiPrefabEditor>(studio, this);
}

std::string const& MochiPrefabAsset::GetReferencerName() const {
  return GetName();
}

void MochiPrefabAsset::ForEachReferencedPath(
    std::function<void(mochi::Path const&)> const& callback) const {
  ForEachPathInPrefab(_prefab, callback);
}

bool MochiPrefabAsset::RewriteReferencedPath(
    mochi::Path const& oldPath,
    mochi::Path const& newPath) {
  bool changed = false;
  for (auto& a : _prefab.actors.rigid) {
    changed |= MaybeRewriteResolved(a.shapeFile, oldPath, newPath);
    changed |= MaybeRewriteResolved(a.renderModelFile, oldPath, newPath);
  }
  for (auto& a : _prefab.actors.soft) {
    changed |= MaybeRewriteResolved(a.shapeFile, oldPath, newPath);
  }
  for (auto& art : _prefab.actors.articulated) {
    for (auto& link : art.links) {
      changed |= MaybeRewriteResolved(link.shapeFile, oldPath, newPath);
      changed |= MaybeRewriteResolved(link.renderModelFile, oldPath, newPath);
    }
    if (art.skin.has_value()) {
      changed |= MaybeRewriteResolved(art.skin->shapeFile, oldPath, newPath);
      changed |= MaybeRewriteResolved(art.skin->renderModelFile, oldPath, newPath);
    }
  }
  for (auto& nested : _prefab.prefabs) {
    changed |= MaybeRewriteResolved(nested.path, oldPath, newPath);
  }
  if (changed) {
    SetDirty(true);
  }
  return changed;
}

mochi::prefab::ScenePrefab const& MochiPrefabAsset::GetPrefab() const {
  return _prefab;
}

mochi::prefab::ScenePrefab& MochiPrefabAsset::GetPrefab() {
  return _prefab;
}

std::string const& MochiPrefabAsset::GetAssetsRoot() const {
  return _assetsRoot;
}

std::optional<mochi::prefab::SceneParams>& MochiPrefabAsset::GetStashedSceneParams() {
  return _stashedSceneParams;
}

} // namespace superdex::studio
