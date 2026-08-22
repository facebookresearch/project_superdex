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

#include "assets/asset.h"
#include "assets/asset_manager.h"
#include "editors/asset_editor.h"

#include <superdex_robotics/utils/file_utils.h> // FindAssetBaseFolder, k*Subdir

#include <mochi_renderer/resource_manager.h>

#include <mochi_core/utils/log.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace superdex::studio {

//--------------------------------------------------------------------------------------------------
// FORWARDS / TYPES / HELPERS
//--------------------------------------------------------------------------------------------------

static bool HasExtension(std::string_view str, std::string_view suffixLower) {
  if (str.size() < suffixLower.size()) {
    return false;
  }
  auto const tail = str.substr(str.size() - suffixLower.size());
  for (size_t i = 0; i < suffixLower.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(tail[i])) != suffixLower[i]) {
      return false;
    }
  }
  return true;
}

namespace {

#if MOCHI_INTERNAL
constexpr bool kInternalBuild = true;
#else
constexpr bool kInternalBuild = false;
#endif

// Folders swept as a last resort when the ordered search finds nothing, covering assets whose
// models sit under a role that does not match their type (a URDF import leaves the `.stl` it was
// given in `collision/`) and foreign trees that name the render role `visual`. Deliberately
// excludes `intermediates`: generated output must never fill a slot.
constexpr std::array<std::string_view, 5> kCrossRoleSearchSubdirs{
    superdex::robotics::kCadSubdir,
    superdex::robotics::kCollisionSubdir,
    superdex::robotics::kRenderSubdir,
    "visual",
    "visuals"};

// Index into kAssetTypeExtensions of the first entry whose extension matches @p filename, or
// kAssetTypeExtensions.size() when none does. Doubles as the preference rank: entries of one type
// are contiguous and ordered there, so among candidates of a single type the lower index is the
// more preferred format.
std::size_t AssetTypeExtensionIndex(std::string_view filename) {
  for (std::size_t i = 0; i < kAssetTypeExtensions.size(); ++i) {
    auto const& entry = kAssetTypeExtensions[i];
    if (!kInternalBuild && entry.type == AssetType::BotScene) {
      continue;
    }
    if (HasExtension(filename, entry.extension)) {
      return i;
    }
  }
  return kAssetTypeExtensions.size();
}

} // namespace

AssetType ClassifyAssetTypeByFilename(std::string_view filename) {
  std::size_t const index = AssetTypeExtensionIndex(filename);
  return index < kAssetTypeExtensions.size() ? kAssetTypeExtensions[index].type
                                             : AssetType::Unknown;
}

AssetType ClassifyAssetTypeByPath(mochi::Path const& path) {
  return ClassifyAssetTypeByFilename(path.GetFilename());
}

std::string GetAssetNameFromPath(mochi::Path const& path) {
  std::string stem = path.GetStem();
  auto const dot = stem.find_last_of('.');
  if (dot != std::string::npos && HasExtension(std::string_view(stem).substr(dot), ".mochi")) {
    return stem.substr(0, dot);
  }
  return stem;
}

std::string
FindAssetFileByName(std::string const& baseName, AssetType type, mochi::Path const& dir) {
  std::error_code ec;
  if (!std::filesystem::is_directory(dir.AsFilesystemPath(), ec)) {
    return {};
  }
  std::string best;
  std::size_t bestRank = kAssetTypeExtensions.size();
  for (auto const& entry : std::filesystem::directory_iterator(dir.AsFilesystemPath(), ec)) {
    if (!entry.is_regular_file(ec)) {
      continue;
    }
    mochi::Path const candidate(entry.path());
    std::string const filename = candidate.GetFilename();
    // Names are compared case-insensitively, as extensions already are: a `Block.glb` authored on
    // Windows has to pair with `block.step` on Linux too, or one asset tree behaves two ways.
    if (ClassifyAssetTypeByFilename(filename) != type ||
        !superdex::robotics::EqualsCaseInsensitive(GetAssetNameFromPath(candidate), baseName)) {
      continue;
    }
    std::size_t const rank = AssetTypeExtensionIndex(filename);
    std::string path = candidate.ToString();
    if (rank < bestRank || (rank == bestRank && !best.empty() && path < best)) {
      bestRank = rank;
      best = std::move(path);
    }
  }
  return best;
}

std::string_view AssetTypeToToken(AssetType type) {
  switch (type) {
    case AssetType::RenderModel:
      return "RenderModel";
    case AssetType::MochiModel:
      return "MochiModel";
    case AssetType::MochiPrefab:
      return "MochiPrefab";
    case AssetType::Bot:
      return "Bot";
    case AssetType::BotScene:
      return "BotScene";
    case AssetType::CadModel:
      return "CadModel";
    case AssetType::Unknown:
    case AssetType::Count:
      return "Unknown";
  }
  return "Unknown";
}

AssetType AssetTypeFromToken(std::string_view token) {
  for (int i = 0; i < kAssetTypeCount; ++i) {
    auto const type = static_cast<AssetType>(i);
    if (AssetTypeToToken(type) == token) {
      return type;
    }
  }
  return AssetType::Unknown;
}

namespace {

// Breadth-first search of @p root for the best (baseName, type) match, so a role folder that has
// been partitioned into subfolders (`cad/internal/` for CAD that must not be open-sourced) still
// resolves. Shallower wins outright: every match at one depth is considered before descending, and
// within a depth the winner is by extension rank then path, so the result never depends on
// directory-iteration order. Directory symlinks are not followed -- an asset tree may be a symlink
// farm (see NormalizeBotPath), which could otherwise be walked twice or cyclically.
std::string
FindAssetFileInTree(std::string const& baseName, AssetType type, mochi::Path const& root) {
  std::vector<mochi::Path> level{root};
  while (!level.empty()) {
    std::vector<mochi::Path> deeper;
    std::string best;
    std::size_t bestRank = kAssetTypeExtensions.size();
    for (mochi::Path const& dir : level) {
      if (std::string found = FindAssetFileByName(baseName, type, dir); !found.empty()) {
        std::size_t const rank = AssetTypeExtensionIndex(mochi::Path{found}.GetFilename());
        if (best.empty() || rank < bestRank || (rank == bestRank && found < best)) {
          bestRank = rank;
          best = std::move(found);
        }
      }
      std::error_code ec;
      auto const options = std::filesystem::directory_options::skip_permission_denied;
      for (auto const& entry :
           std::filesystem::directory_iterator(dir.AsFilesystemPath(), options, ec)) {
        if (entry.is_directory(ec) && !entry.is_symlink()) {
          deeper.emplace_back(entry.path());
        }
      }
    }
    if (!best.empty()) {
      return best;
    }
    level = std::move(deeper);
  }
  return {};
}

} // namespace

std::string_view AssetRoleFolderForType(AssetType type) {
  switch (type) {
    case AssetType::CadModel:
      return superdex::robotics::kCadSubdir;
    case AssetType::RenderModel:
      return superdex::robotics::kRenderSubdir;
    case AssetType::MochiModel:
      return superdex::robotics::kCollisionSubdir;
    case AssetType::MochiPrefab:
    case AssetType::Bot:
    case AssetType::BotScene:
    case AssetType::Unknown:
    case AssetType::Count:
      break;
  }
  return {};
}

std::string
AssetGeneratedFilePath(mochi::Path const& originPath, bool isCanonical, std::string_view suffix) {
  std::filesystem::path const originDir = originPath.GetParentPath().AsFilesystemPath();
  std::filesystem::path folder = superdex::robotics::AssetRoleFolderForWrite(
      originDir, superdex::robotics::kIntermediatesSubdir);
  std::string stem = GetAssetNameFromPath(originPath);
  if (!isCanonical) {
    // Mirror where the model sits within the asset, and keep its extension in the stem. The mirror
    // separates two shadowed copies of one name in different subfolders; the extension separates a
    // shadowed model from the canonical one when it sits directly in the base folder, where there
    // is no subpath to mirror.
    std::filesystem::path const base = superdex::robotics::FindAssetBaseFolder(originDir);
    std::filesystem::path const relative =
        superdex::robotics::NormalizeBotPath(originDir).lexically_relative(base);
    if (!relative.empty() && relative != "." && *relative.begin() != "..") {
      folder /= relative;
    }
    stem = originPath.GetFilename();
  }
  std::filesystem::path full = folder / (stem + std::string(suffix));
  full.make_preferred();
  return full.string();
}

std::string
FindAssetForSlot(std::string const& baseName, AssetType type, mochi::Path const& originDir) {
  std::string_view const role = AssetRoleFolderForType(type);
  mochi::Path const base{superdex::robotics::FindAssetBaseFolder(originDir.AsFilesystemPath())};

  // Every directory is visited at most once, so a folder already covered by an earlier (higher
  // precedence) step cannot contribute a second candidate to the sweep below.
  std::vector<std::string> searched;
  // Role folders are searched as trees; the base folder and the origin's own folder are not. A
  // base folder can contain whole other assets -- a two-handed bot keeps `left/` and `right/`
  // subfolders, each with role folders of its own -- so descending from it would let one asset's
  // models fill another's slots.
  auto search = [&](mochi::Path const& dir, bool recurse) -> std::string {
    std::string key = dir.ToString();
    if (std::find(searched.begin(), searched.end(), key) != searched.end()) {
      return {};
    }
    searched.push_back(std::move(key));
    return recurse ? FindAssetFileInTree(baseName, type, dir)
                   : FindAssetFileByName(baseName, type, dir);
  };

  // Steps 1-3: ordered and short-circuiting, so the winner is decided by precedence, not by which
  // candidates happen to exist.
  if (!role.empty()) {
    if (std::string found = search(base / role, /*recurse=*/true); !found.empty()) {
      return found;
    }
  }
  if (std::string found = search(base, /*recurse=*/false); !found.empty()) {
    return found;
  }
  if (std::string found = search(originDir, /*recurse=*/false); !found.empty()) {
    return found;
  }

  // Step 4: unique-or-nothing sweep of the remaining role folders and aliases.
  std::vector<std::string> crossRoleHits;
  for (std::string_view const subdir : kCrossRoleSearchSubdirs) {
    if (std::string found = search(base / subdir, /*recurse=*/true); !found.empty()) {
      crossRoleHits.push_back(std::move(found));
    }
  }
  if (crossRoleHits.size() == 1) {
    return crossRoleHits.front();
  }
  if (crossRoleHits.size() > 1) {
    std::string paths;
    for (std::string const& hit : crossRoleHits) {
      paths += "\n  " + hit;
    }
    MOCHI_LOG_WARNING(
        "Asset discovery: '%s' has %d equally-plausible %s files outside their role folder; "
        "leaving the slot empty rather than guessing:%s",
        baseName.c_str(),
        static_cast<int>(crossRoleHits.size()),
        GetAssetTypeLabel(type),
        paths.c_str());
  }
  return {};
}

//--------------------------------------------------------------------------------------------------
// ASSET
//--------------------------------------------------------------------------------------------------

std::string const& Asset::GetName() const {
  return _name;
}

mochi::Path const& Asset::GetPath() const {
  return _path;
}

AssetType Asset::GetType() const {
  return _type;
}

ImU32 Asset::GetColor() const {
  return GetAssetTypeColor(_type);
}

char const* Asset::GetTypeLabel() const {
  return GetAssetTypeLabel(_type);
}

bool Asset::RendersThumbnail() const {
  return false;
}

void Asset::StageThumbnailScene(mochi_renderer::Scene& /*scene*/) {}

void Asset::MarkThumbnailDirty() {
  _thumbDirty = true;
}

void* Asset::GetThumbnailImage() const {
  if (RendersThumbnail()) {
    return _thumbRendered ? _thumbRenderTarget->GetTextureId() : nullptr;
  }
  return nullptr;
}

void Asset::SetDirty(bool dirty) {
  _dirty = dirty;
}

bool Asset::IsDirty() const {
  return _dirty;
}

bool Asset::IsReadOnly() const {
  return _readOnly;
}

void Asset::SetReadOnly(bool readOnly) {
  _readOnly = readOnly;
}

bool Asset::IsSavable() const {
  return false;
}

bool Asset::Save() const {
  return false;
}

bool Asset::ReloadFromDisk() {
  return false;
}

int Asset::GetReferenceCount() const {
  return _manager->GetPathReferenceCount(_path);
}

std::vector<IAssetReferencer*> Asset::GetReferencers() const {
  return _manager->GetReferencersToPath(_path);
}

std::unique_ptr<AssetEditor> Asset::CreateEditor(SuperDexStudio*) {
  return nullptr;
}

Asset::Asset(
    std::string const& name,
    mochi::Path const& path,
    AssetType type,
    AssetManager* manager)
    : _manager(manager), _name(name), _path(path), _type(type) {}

void Asset::OnUnload(mochi_renderer::ResourceManager& /*resourceManager*/) {}

void Asset::OnRewritePath(
    mochi::Path const& /*oldPath*/,
    mochi::Path const& /*newPath*/,
    mochi_renderer::ResourceManager& /*resourceManager*/) {}

void Asset::Refresh() {
  MarkThumbnailDirty();
}

} // namespace superdex::studio
