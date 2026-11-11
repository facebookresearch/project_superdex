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

#include <mochi_renderer/path.h>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <unordered_map>

namespace superdex::studio {

using IKRecordingMemory = std::unordered_map<std::string, std::string>;

// Stable local-settings identity: independent of cwd, dot segments, and existing symlink aliases.
inline std::string IKDesignPathKey(mochi::Path const& path) {
  if (path.IsEmpty()) {
    return {};
  }
  std::error_code error;
  auto absolute = std::filesystem::absolute(path.AsFilesystemPath(), error);
  if (error) {
    return {};
  }
  auto canonical = std::filesystem::weakly_canonical(absolute, error);
  std::string key = mochi::Path{error ? absolute.lexically_normal() : canonical}.ToString();
#if MOCHI_PLATFORM_WINDOWS
  std::transform(
      key.begin(), key.end(), key.begin(), [](unsigned char c) { return char(std::tolower(c)); });
#endif
  return key;
}

inline std::string RememberedIKRecording(
    IKRecordingMemory const& settings,
    mochi::Path const& document) {
  auto const found = settings.find(IKDesignPathKey(document));
  return found == settings.end() ? std::string{} : found->second;
}

// Selecting/clearing a recording changes only preferences; callers do not mark the asset dirty.
inline bool RememberIKRecording(
    IKRecordingMemory& settings,
    mochi::Path const& document,
    mochi::Path const& recording) {
  auto const key = IKDesignPathKey(document);
  if (key.empty()) {
    return false;
  }
  if (recording.IsEmpty()) {
    return settings.erase(key) != 0;
  }
  auto const value = IKDesignPathKey(recording);
  if (value.empty()) {
    return false;
  }
  auto const current = settings.find(key);
  if (current != settings.end() && current->second == value) {
    return false;
  }
  settings[key] = value;
  return true;
}

// Batch filesystem rename/move notifications can update local state for unopened documents too.
inline bool RewriteIKDesignPath(
    IKRecordingMemory& settings,
    mochi::Path const& oldPath,
    mochi::Path const& newPath) {
  auto const oldKey = IKDesignPathKey(oldPath);
  auto const newKey = IKDesignPathKey(newPath);
  if (oldKey.empty() || newKey.empty() || oldKey == newKey) {
    return false;
  }
  bool changed = false;
  auto found = settings.find(oldKey);
  if (found != settings.end()) {
    auto value = found->second;
    settings.erase(found);
    settings[newKey] = std::move(value);
    changed = true;
  }
  for (auto& [document, recording] : settings) {
    if (recording == oldKey) {
      recording = newKey;
      changed = true;
    }
  }
  return changed;
}

} // namespace superdex::studio
