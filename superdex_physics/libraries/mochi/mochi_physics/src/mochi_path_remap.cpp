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

#include <mochi_physics/utils/mochi_path_remap.h>

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/guarded.h>
#include <mochi_core/utils/log.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace mochi;

namespace {

struct PathRemapTable {
  std::unordered_map<std::string, std::string> entries;
  // Normalized targets of the entries, which a strict root never refuses.
  std::unordered_set<std::string> pinned;
  // Normalized and lexical forms of each root (see LexicalKey), each with a trailing '/'.
  DynamicArray<std::string> strictRoots;
};

SharedGuarded<PathRemapTable>& GetTable() {
  static SharedGuarded<PathRemapTable> table;
  return table;
}

// Absolute, with symlinks resolved as far as the path exists, so every spelling of a file agrees.
// When the filesystem can't be queried, falls back to the lexical form and clears @p resolved.
std::string NormalizeKey(std::string_view path, bool* resolved = nullptr) {
  std::error_code ec;
  auto const absolute = std::filesystem::absolute(std::filesystem::path(path), ec);
  auto const fallback = ec ? std::filesystem::path(path) : absolute;
  if (!ec) {
    auto const canonical = std::filesystem::weakly_canonical(absolute, ec);
    if (!ec) {
      return canonical.generic_string();
    }
  }
  if (resolved != nullptr) {
    *resolved = false;
  }
  return fallback.lexically_normal().generic_string();
}

// Absolute and lexically normal, with symlinks left as written. A symlink under a strict root
// that points outside every root is under a root only in this form.
std::string LexicalKey(std::string_view path) {
  std::error_code ec;
  auto const absolute = std::filesystem::absolute(std::filesystem::path(path), ec);
  return (ec ? std::filesystem::path(path) : absolute).lexically_normal().generic_string();
}

bool IsRefused(PathRemapTable const& table, std::string const& key, std::string const& lexicalKey) {
  if (table.pinned.contains(key)) {
    return false;
  }
  auto const isUnderStrictRoot = [&](std::string const& path) {
    return std::ranges::any_of(
        table.strictRoots, [&](std::string const& root) { return path.starts_with(root); });
  };
  if (!isUnderStrictRoot(key) && !isUnderStrictRoot(lexicalKey)) {
    return false;
  }
  std::error_code ec;
  return !std::filesystem::is_directory(key, ec);
}

// Strict roots alone still refuse every file under them, so they count as an installed table.
bool IsInstalled(PathRemapTable const& table) {
  return !table.entries.empty() || !table.strictRoots.empty();
}

} // namespace

void mochi::SetPathRemap(
    Span<PathRemapEntry const> entries,
    Span<DynamicString const> strictRoots) {
  PathRemapTable table;
  for (auto const& entry : entries) {
    table.entries.insert_or_assign(NormalizeKey(entry.from), std::string(entry.to));
    table.pinned.insert(NormalizeKey(entry.to));
  }
  for (auto const& root : strictRoots) {
    for (auto normalizedRoot : {NormalizeKey(root), LexicalKey(root)}) {
      if (!normalizedRoot.ends_with('/')) {
        normalizedRoot += '/';
      }
      if (std::ranges::find(table.strictRoots, normalizedRoot) == table.strictRoots.end()) {
        table.strictRoots.push_back(std::move(normalizedRoot));
      }
    }
  }
  GetTable() = std::move(table);
}

void mochi::ClearPathRemap() {
  GetTable() = PathRemapTable{};
}

bool mochi::IsPathRemapActive() {
  return GetTable().Read([](PathRemapTable const& table) { return IsInstalled(table); });
}

DynamicString mochi::RemapPath(std::string_view path) {
  return GetTable().Read([&](PathRemapTable const& table) -> DynamicString {
    if (!IsInstalled(table)) {
      return DynamicString(path);
    }
    bool resolved = true;
    auto const key = NormalizeKey(path, &resolved);
    auto const it = table.entries.find(key);
    if (it != table.entries.end()) {
      return DynamicString(it->second);
    }
    // A key that couldn't be resolved may not match a strict root's spelling; refuse it rather
    // than risk loading an unpinned file.
    if (!resolved && !table.strictRoots.empty()) {
      MOCHI_LOG_ERROR(
          "Path remap table cannot resolve %s while strict roots are installed.", key.c_str());
      return {};
    }
    if (IsRefused(table, key, LexicalKey(path))) {
      MOCHI_LOG_ERROR(
          "Path remap table does not pin %s, which is under a strict root.", key.c_str());
      return {};
    }
    return DynamicString(path);
  });
}
