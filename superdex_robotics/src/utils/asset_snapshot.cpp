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

#include <superdex_robotics/utils/asset_snapshot.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/json_utils.h>
#include <mochi_physics/utils/mochi_path_remap.h>

#include <filesystem>
#include <set>
#include <string>
#include <utility>

using namespace mochi;
using namespace superdex::robotics;

AssetSnapshot superdex::robotics::LoadAssetSnapshot(std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  auto const json = ParseJsonFromFile(path, error);
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(!json.is<picojson::object>(), error, "Snapshot must be a JSON object.");
  MOCHI_ERROR_RETURN(error, {});
  auto const& root = json.get<picojson::object>();
  auto const field = [&](char const* name) -> picojson::value const* {
    auto const it = root.find(name);
    return it == root.end() ? nullptr : &it->second;
  };
  auto const* formatVersion = field("formatVersion");
  MOCHI_ERROR_IF(
      formatVersion == nullptr || !formatVersion->is<int64_t>() ||
          formatVersion->get<int64_t>() != kAssetSnapshotFormatVersion,
      error,
      "Snapshot \"formatVersion\" is missing or not supported by this reader.");
  MOCHI_ERROR_RETURN(error, {});
  auto const* scene = field("scene");
  auto const* createdAt = field("createdAt");
  auto const* sourceRevision = field("sourceRevision");
  auto const* files = field("files");
  if (scene == nullptr || !scene->is<std::string>() || createdAt == nullptr ||
      !createdAt->is<int64_t>() || createdAt->get<int64_t>() < 0 || sourceRevision == nullptr ||
      !sourceRevision->is<std::string>() || files == nullptr || !files->is<picojson::object>()) {
    MOCHI_ERROR_SET(
        error,
        "Snapshot needs a string \"scene\", non-negative integer \"createdAt\", string "
        "\"sourceRevision\", and object \"files\".");
    return {};
  }

  AssetSnapshot snapshot;
  snapshot.scene = DynamicString(scene->get<std::string>());
  snapshot.createdAt = static_cast<uint64_t>(createdAt->get<int64_t>());
  snapshot.sourceRevision = DynamicString(sourceRevision->get<std::string>());
  for (auto const& [file, version] : files->get<picojson::object>()) {
    MOCHI_ERROR_IF(!version.is<std::string>(), error, "Snapshot \"files\" values must be strings.");
    MOCHI_ERROR_RETURN(error, {});
    snapshot.files.push_back(
        {.file = DynamicString(file), .version = DynamicString(version.get<std::string>())});
  }
  return snapshot;
}

void superdex::robotics::SaveAssetSnapshot(
    AssetSnapshot const& snapshot,
    std::string_view path,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  picojson::object files;
  for (auto const& entry : snapshot.files) {
    files.insert_or_assign(std::string(entry.file), picojson::value(std::string(entry.version)));
  }
  picojson::object root;
  root.emplace("formatVersion", picojson::value(kAssetSnapshotFormatVersion));
  root.emplace("scene", picojson::value(std::string(snapshot.scene)));
  root.emplace("createdAt", picojson::value(static_cast<int64_t>(snapshot.createdAt)));
  root.emplace("sourceRevision", picojson::value(std::string(snapshot.sourceRevision)));
  root.emplace("files", picojson::value(files));
  WriteFile(
      std::filesystem::path(path),
      std::string_view(picojson::value(root).serialize(/*prettify=*/true)),
      error);
}

/* @brief Whether @p path is the .superdex_root marker of its directory. */
static bool IsRootMarker(std::filesystem::path const& path) {
  auto const marker = FindRootMarker(path.parent_path());
  return marker.has_value() && marker->filename() == path.filename();
}

DynamicString superdex::robotics::ApplyAssetSnapshot(
    AssetSnapshot const& snapshot,
    std::string_view assetRoot,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  // Resolve against the files as they are, not through a table an earlier snapshot installed.
  ClearPathRemap();
  auto const base = std::filesystem::path(assetRoot) / kRootMarkerFile;
  auto const resolve = [&](DynamicString const& botPath) {
    return ResolveBotPath(
        std::string_view(botPath.c_str(), botPath.size()), base, kBotPathMaxParentDepth, error);
  };
  DynamicArray<PathRemapEntry> entries;
  std::set<std::string> pinnedFiles;
  std::set<std::filesystem::path> strictRoots = {NormalizeBotPath(assetRoot)};
  for (auto const& entry : snapshot.files) {
    auto const file = resolve(entry.file);
    auto const version = resolve(entry.version);
    MOCHI_ERROR_RETURN(error, {});
    if (!std::filesystem::is_regular_file(version)) {
      MOCHI_LOG_ERROR("Snapshot version not found: %s", version.generic_string().c_str());
      MOCHI_ERROR_SET(error, "A snapshot version does not exist under the asset root.");
      return {};
    }
    // A version's own `./` and root-relative references resolve from its directory.
    if (version.parent_path() != file.parent_path()) {
      MOCHI_LOG_ERROR(
          "Snapshot version %s is not beside %s",
          version.generic_string().c_str(),
          file.generic_string().c_str());
      MOCHI_ERROR_SET(error, "A snapshot version must be in the same directory as its file.");
      return {};
    }
    if (IsRootMarker(file)) {
      auto const current = ReadFileString(file, error);
      auto const pinned = ReadFileString(version, error);
      MOCHI_ERROR_RETURN(error, {});
      if (current != pinned) {
        MOCHI_LOG_ERROR(
            "%s no longer matches its pinned version %s",
            file.generic_string().c_str(),
            version.generic_string().c_str());
        MOCHI_ERROR_SET(error, "A pinned .superdex_root changed since the snapshot was taken.");
        return {};
      }
    }
    if (auto root = FindBotsRoot(file)) {
      strictRoots.insert(std::move(*root));
    }
    pinnedFiles.insert(file.generic_string());
    entries.push_back(
        {.from = DynamicString(file.generic_string()),
         .to = DynamicString(version.generic_string())});
  }
  for (auto const& entry : entries) {
    if (pinnedFiles.contains(std::string(entry.to))) {
      MOCHI_LOG_ERROR("Snapshot version %s is also a file the snapshot pins", entry.to.c_str());
      MOCHI_ERROR_SET(error, "A snapshot version must not be a file the snapshot pins.");
      return {};
    }
  }
  auto const scene = resolve(snapshot.scene);
  MOCHI_ERROR_RETURN(error, {});
  DynamicArray<DynamicString> roots;
  for (auto const& root : strictRoots) {
    roots.push_back(DynamicString(root.generic_string()));
  }
  SetPathRemap(MakeConstSpan(entries), MakeConstSpan(roots));
  return DynamicString(scene.generic_string());
}

void superdex::robotics::ClearAssetSnapshot() {
  ClearPathRemap();
}
