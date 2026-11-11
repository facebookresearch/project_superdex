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

#include "assets/bot_task_cache.h"

#include <superdex_robotics/scene/bot_task.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/utils/reflection.h>

#include <algorithm>

namespace superdex::studio {

BotTaskCache::Entry* BotTaskCache::GetOrLoad(std::string const& path, std::string& error) {
  if (auto* entry = Find(path)) {
    return entry;
  }
  superdex::Error loadError;
  auto const task = superdex::robotics::LoadBotTaskPrefabFromFile(path, loadError);
  if (!loadError.IsOK()) {
    error = loadError.GetDescription();
    return nullptr;
  }
  Entry& entry = _entries[path];
  entry.json = SReflect::ToJsonString(task, true);
  entry.baselineJson = entry.json;
  entry.loadedJson = entry.json;
  return &entry;
}

BotTaskCache::Entry* BotTaskCache::Find(std::string const& path) {
  auto const it = _entries.find(path);
  return it != _entries.end() ? &it->second : nullptr;
}

bool BotTaskCache::AnyModified() const {
  return std::ranges::any_of(_entries, [](auto const& item) { return IsModified(item.second); });
}

bool BotTaskCache::Save(std::string const& path, std::string& error) {
  Entry* const entry = Find(path);
  return entry == nullptr || !IsModified(*entry) || Write(path, *entry, error);
}

bool BotTaskCache::SaveAs(std::string const& from, std::string const& to, std::string& error) {
  if (from == to) {
    return Save(from, error);
  }
  Entry* const source = Find(from);
  if (source == nullptr) {
    error = "Task '" + from + "' is not loaded";
    return false;
  }
  Entry copy;
  copy.json = source->json;
  if (!Write(to, copy, error)) {
    return false;
  }
  source->json = source->baselineJson;
  // A task already loaded from @p to keeps the value undo restores it to.
  auto const [it, inserted] = _entries.try_emplace(to);
  it->second.json = copy.json;
  it->second.baselineJson = copy.json;
  if (inserted) {
    it->second.loadedJson = copy.json;
  }
  return true;
}

bool BotTaskCache::SaveModified(std::string& error) {
  for (auto& [path, entry] : _entries) {
    if (IsModified(entry) && !Write(path, entry, error)) {
      return false;
    }
  }
  return true;
}

std::map<std::string, std::string> BotTaskCache::CaptureValues() const {
  std::map<std::string, std::string> values;
  for (auto const& [path, entry] : _entries) {
    values.emplace(path, entry.json);
  }
  return values;
}

void BotTaskCache::RestoreValues(std::map<std::string, std::string> const& values) {
  for (auto& [path, entry] : _entries) {
    // An entry missing from the snapshot had not been loaded yet.
    auto const it = values.find(path);
    entry.json = it != values.end() ? it->second : entry.loadedJson;
  }
}

bool BotTaskCache::Write(std::string const& path, Entry& entry, std::string& error) {
  superdex::robotics::BotTaskPrefab task;
  if (!SReflect::FromJsonString(task, entry.json)) {
    error = "Failed to save task '" + path + "': its edited value could not be read";
    return false;
  }
  superdex::Error saveError;
  superdex::robotics::ValidateBotTaskPrefab(task, saveError);
  if (saveError.IsOK()) {
    superdex::robotics::SaveParamsToFile(task, path, saveError);
  }
  if (!saveError.IsOK()) {
    error = "Failed to save task '" + path + "': " + saveError.GetDescription();
    return false;
  }
  entry.baselineJson = entry.json;
  return true;
}

} // namespace superdex::studio
