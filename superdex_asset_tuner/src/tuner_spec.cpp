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

#include "superdex_asset_tuner/tuner_spec.h"

#include "superdex_asset_tuner/json_path.h"

#include <mochi_core/utils/error.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

namespace superdex::tuner {

namespace {

bool ReadTextFile(std::filesystem::path const& path, std::string& out, std::string& outError) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    outError = "Cannot open '" + path.string() + "'.";
    return false;
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  out = buffer.str();
  return true;
}

// Brings the record settings of files written before sim.rrd replaced sim.h5 up to date:
// `contactPoints` is now `contactForces`, and static actors are always recorded once.
void MigrateRecordSpec(picojson::value& document) {
  if (!document.is<picojson::object>()) {
    return;
  }
  auto& root = document.get<picojson::object>();
  auto const run = root.find("run");
  if (run == root.end() || !run->second.is<picojson::object>()) {
    return;
  }
  auto& runObject = run->second.get<picojson::object>();
  auto const record = runObject.find("record");
  if (record == runObject.end() || !record->second.is<picojson::object>()) {
    return;
  }
  auto& fields = record->second.get<picojson::object>();
  if (auto const contactPoints = fields.find("contactPoints"); contactPoints != fields.end()) {
    fields.emplace("contactForces", contactPoints->second);
    fields.erase(contactPoints);
  }
  fields.erase("staticActors");
}

} // namespace

bool LoadTunerSpecFromJson(
    std::string_view json,
    Span<SpecOverride const> overrides,
    TunerSpec& outSpec,
    std::string& outError) {
  picojson::value document;
  if (!ParseJson(json, document, outError)) {
    return false;
  }
  MigrateRecordSpec(document);
  for (SpecOverride const& entry : overrides) {
    if (!SetJsonPathFromText(document, entry.path, entry.value, outError)) {
      return false;
    }
  }
  outSpec = {};
  if (!SReflect::FromJsonString(
          outSpec, document.serialize(), SReflect::DeserializeFlags::WarnIfExtraneousFields)) {
    outError = "Tuner spec does not match the schema (unknown field or wrong type; see log).";
    return false;
  }
  return true;
}

bool LoadTunerSpecFromFile(
    std::filesystem::path const& path,
    Span<SpecOverride const> overrides,
    TunerSpec& outSpec,
    std::string& outError) {
  std::string json;
  if (!ReadTextFile(path, json, outError) ||
      !LoadTunerSpecFromJson(json, overrides, outSpec, outError)) {
    outError = path.string() + ": " + outError;
    return false;
  }
  return true;
}

std::string SaveTunerSpecToJson(TunerSpec const& spec) {
  return SReflect::ToJsonString(spec, true);
}

bool SaveTunerSpecToFile(
    TunerSpec const& spec,
    std::filesystem::path const& path,
    std::string& outError) {
  std::ofstream file(path, std::ios::binary);
  file << SaveTunerSpecToJson(spec);
  // Buffered write errors (e.g. a full disk) only surface when the stream is flushed.
  file.close();
  if (!file) {
    outError = "Cannot write '" + path.string() + "'.";
    return false;
  }
  return true;
}

std::filesystem::path ResolveAssetsRoot(
    TunerSpec const& spec,
    std::filesystem::path const& tunerFilePath,
    std::filesystem::path const& anchorPath) {
  std::filesystem::path const tunerDir = tunerFilePath.parent_path();
  if (!spec.assetsRoot.empty()) {
    return std::filesystem::absolute(tunerDir / std::string_view(spec.assetsRoot));
  }
  std::filesystem::path const anchor = std::filesystem::absolute(anchorPath);
  std::error_code ec;
  for (std::filesystem::path dir = anchor.parent_path(); !dir.empty(); dir = dir.parent_path()) {
    if (std::filesystem::is_regular_file(dir / kSuperdexRootMarker, ec)) {
      return dir;
    }
    if (dir == dir.root_path()) {
      break;
    }
  }
  if (char const* env = std::getenv("SUPERDEX_ASSETS_PATH"); env != nullptr && *env != '\0') {
    return std::filesystem::absolute(env);
  }
  return anchor.parent_path();
}

std::filesystem::path ResolvePrefabPath(
    TunerSpec const& spec,
    std::filesystem::path const& tunerFilePath) {
  std::string_view const prefab = spec.prefab;
  if (prefab.starts_with("./")) {
    return std::filesystem::absolute(tunerFilePath.parent_path() / prefab.substr(2));
  }
  std::filesystem::path const path(prefab);
  if (path.is_absolute()) {
    return path;
  }
  return ResolveAssetsRoot(spec, tunerFilePath, tunerFilePath) / path;
}

namespace {

// PrefabReference fields: an override of these edits the reference itself, not the nested prefab.
constexpr std::array<std::string_view, 6>
    kPrefabReferenceFields{"_comment", "name", "path", "scale", "rotation", "translation"};

// Matches `prefabs[<selector>].<rest>` where `rest` addresses the referenced prefab's content.
bool SplitNestedOverride(
    std::string_view path,
    std::string_view& outSelector,
    std::string_view& outRest) {
  constexpr std::string_view kPrefix = "prefabs[";
  size_t const close = path.find(']');
  if (!path.starts_with(kPrefix) || close == std::string_view::npos || close + 2 > path.size() ||
      path[close + 1] != '.') {
    return false;
  }
  outSelector = path.substr(kPrefix.size(), close - kPrefix.size());
  outRest = path.substr(close + 2);
  std::string_view const field = outRest.substr(0, outRest.find_first_of(".["));
  return std::find(kPrefabReferenceFields.begin(), kPrefabReferenceFields.end(), field) ==
      kPrefabReferenceFields.end();
}

mochi::prefab::PrefabReference* FindPrefabReference(
    mochi::prefab::ScenePrefab& prefab,
    std::string_view selector,
    std::string& outError) {
  constexpr std::string_view kNameSelector = "name=";
  if (selector.starts_with(kNameSelector)) {
    std::string_view const name = selector.substr(kNameSelector.size());
    for (auto& reference : prefab.prefabs) {
      if (std::string_view(reference.name) == name) {
        return &reference;
      }
    }
  } else {
    size_t index = 0;
    auto const [end, ec] =
        std::from_chars(selector.data(), selector.data() + selector.size(), index);
    if (ec == std::errc{} && end == selector.data() + selector.size() &&
        index < prefab.prefabs.size()) {
      return &prefab.prefabs[index];
    }
  }
  outError = "No nested prefab matches [" + std::string(selector) + "].";
  return nullptr;
}

} // namespace

bool ApplyPrefabOverrides(
    mochi::prefab::ScenePrefab& prefab,
    Span<PrefabOverride const> overrides,
    std::string& outError) {
  // Overrides inside a nested prefab edit that loaded prefab directly: nested prefab content is not
  // part of this prefab's JSON. The nested prefab is copied first, since references can share it.
  DynamicArray<PrefabOverride> local;
  for (PrefabOverride const& entry : overrides) {
    std::string_view selector;
    std::string_view rest;
    if (!SplitNestedOverride(entry.path, selector, rest)) {
      local.push_back(entry);
      continue;
    }
    mochi::prefab::PrefabReference* reference = FindPrefabReference(prefab, selector, outError);
    if (reference == nullptr || !reference->prefab) {
      outError = "Prefab override '" + std::string(std::string_view(entry.path)) +
          "': " + (reference == nullptr ? outError : "the nested prefab is not loaded.");
      return false;
    }
    PrefabOverride nestedEntry;
    nestedEntry.path = DynamicString(rest);
    nestedEntry.value = entry.value;
    auto copy = std::make_shared<mochi::prefab::ScenePrefab>(*reference->prefab);
    if (!ApplyPrefabOverrides(*copy, Span<PrefabOverride const>(&nestedEntry, 1), outError)) {
      return false;
    }
    reference->prefab = std::move(copy);
  }
  if (local.empty()) {
    return true;
  }

  mochi::Error error;
  std::string const json(mochi::prefab::SaveToJsonString(prefab, error));
  picojson::value document;
  if (!error.IsOK() || !ParseJson(json, document, outError)) {
    outError = "Failed to serialize prefab for overrides. " + outError;
    return false;
  }
  for (PrefabOverride const& entry : local) {
    if (!SetJsonPathFromText(document, entry.path, entry.value, outError)) {
      outError = "Prefab override: " + outError;
      return false;
    }
  }
  mochi::prefab::ScenePrefab overridden =
      mochi::prefab::ShallowLoadFromJsonString(document.serialize(), error);
  if (!error.IsOK()) {
    outError = std::string("Prefab override produced an invalid prefab: ") +
        error.GetDescription() + " (see log)";
    return false;
  }
  overridden.sourceFilePath = prefab.sourceFilePath;
  // The JSON round trip drops loaded nested prefabs, and the nested overrides applied above. Put
  // each one back on its reference, matched by name and path, not by position. Overrides can add,
  // remove or reorder references, and two references can point at the same file. A reference with
  // a new path loads from disk later.
  std::vector<bool> taken(prefab.prefabs.size(), false);
  auto const reattach = [&](mochi::prefab::PrefabReference& reference, bool matchName) {
    for (size_t i = 0; i < prefab.prefabs.size(); ++i) {
      mochi::prefab::PrefabReference const& original = prefab.prefabs[i];
      if (!taken[i] && original.prefab && original.path == reference.path &&
          (!matchName || original.name == reference.name)) {
        reference.prefab = original.prefab;
        taken[i] = true;
        return;
      }
    }
  };
  for (mochi::prefab::PrefabReference& reference : overridden.prefabs) {
    reattach(reference, true);
  }
  // A renamed reference: match by path alone, among the prefabs not yet used.
  for (mochi::prefab::PrefabReference& reference : overridden.prefabs) {
    if (!reference.prefab) {
      reattach(reference, false);
    }
  }
  prefab = std::move(overridden);
  return true;
}

bool LoadTunerPrefab(
    TunerSpec const& spec,
    std::filesystem::path const& tunerFilePath,
    mochi::Context* context,
    mochi::prefab::ScenePrefab& outPrefab,
    std::string& outError) {
  std::filesystem::path const prefabPath = ResolvePrefabPath(spec, tunerFilePath);
  // Anchored on the prefab, as Studio's MochiPrefabAsset resolves the prefab's own references.
  std::string const rootPath = ResolveAssetsRoot(spec, tunerFilePath, prefabPath).string();
  mochi::Error error;
  outPrefab = mochi::prefab::ShallowLoadFromFile(prefabPath.string(), error);
  if (!error.IsOK()) {
    outError = "Failed to load prefab '" + prefabPath.string() + "': " + error.GetDescription();
    return false;
  }
  // Nested prefabs must be loaded before overrides so overrides can reach into them.
  mochi::prefab::LoadNestedPrefabs(outPrefab, rootPath, error);
  if (!error.IsOK()) {
    outError =
        "Failed to load nested prefabs of '" + prefabPath.string() + "': " + error.GetDescription();
    return false;
  }
  if (!ApplyPrefabOverrides(outPrefab, spec.prefabOverrides, outError)) {
    return false;
  }
  mochi::prefab::EnsureFullyLoaded(outPrefab, rootPath, context, error);
  if (!error.IsOK()) {
    outError = "Failed to load prefab contents for '" + prefabPath.string() + "' (assets root '" +
        rootPath + "'): " + error.GetDescription();
    return false;
  }
  return true;
}

} // namespace superdex::tuner
