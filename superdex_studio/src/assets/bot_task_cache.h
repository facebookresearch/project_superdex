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

#include <map>
#include <string>

namespace superdex::studio {

// .superdex_task files edited in a bot scene's editor, keyed by path. Values are JSON as
// SReflect::ToJsonString writes a BotTaskPrefab, so equal tasks compare equal as strings. Whether
// an entry is modified is derived by comparing it with the value last loaded or saved.
class BotTaskCache {
 public:
  struct Entry {
    std::string json;
    // Value last loaded from or saved to the file.
    std::string baselineJson;
    // Value when first loaded, which undo restores to for snapshots taken before the load.
    std::string loadedJson;
  };

  // Returns the entry for @p path, loading the file on first use, or null with @p error set if it
  // fails to load.
  Entry* GetOrLoad(std::string const& path, std::string& error);
  Entry* Find(std::string const& path);

  [[nodiscard]] static bool IsModified(Entry const& entry) {
    return entry.json != entry.baselineJson;
  }

  [[nodiscard]] bool AnyModified() const;

  // Writes the entry for @p path if it is modified. Refuses tasks that fail validation.
  bool Save(std::string const& path, std::string& error);
  // Writes @p from's current value to @p to and makes it @p to's entry, then reverts @p from to its
  // file's value, so its unsaved edits move to the copy instead of also reaching the original.
  bool SaveAs(std::string const& from, std::string const& to, std::string& error);
  // Writes every modified entry, stopping at the first failure.
  bool SaveModified(std::string& error);

  [[nodiscard]] std::map<std::string, std::string> CaptureValues() const;
  // Restores values from CaptureValues. Entries the capture lacks revert to their loaded value.
  void RestoreValues(std::map<std::string, std::string> const& values);

 private:
  static bool Write(std::string const& path, Entry& entry, std::string& error);

  std::map<std::string, Entry> _entries;
};

} // namespace superdex::studio
