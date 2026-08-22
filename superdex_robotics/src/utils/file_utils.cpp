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

#include "urdf_utils.h"

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/overload_visitor.h>
#include <superdex_robotics/utils/archive_utils.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#define XXH_INLINE_ALL
#include "xxhash.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cinttypes>
#include <sstream>
#include <variant>

#if !MOCHI_USE_REFLECTION
#error Reflection is required for the mochi_bots library internals. Please define `MOCHI_USE_REFLECTION=1` in your build system.
#endif

using namespace mochi;
using namespace superdex::robotics;

bool superdex::robotics::IsBotPath(std::string_view path) {
  if (path.ends_with(kBotExtension)) {
    return true;
  }
#if MOCHI_INTERNAL
  if (path.ends_with(kBotExtensionLegacy)) {
    return true;
  }
#endif
  return false;
}

static std::filesystem::path GetDirectoryPath(std::filesystem::path const& path) {
  return path.parent_path();
}

void superdex::robotics::EnsureDirectoriesCreated(std::filesystem::path const& path, Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto parentDir = std::filesystem::path(path).parent_path();
  if (!parentDir.empty() && !std::filesystem::exists(parentDir)) {
    std::error_code ec;
    std::filesystem::create_directories(parentDir, ec);
    MOCHI_ERROR_IF(ec, error, "Failed to create directory path for saving.");
    MOCHI_ERROR_RETURN(error);
  }
}

std::optional<std::filesystem::path> superdex::robotics::FindRootMarker(
    std::filesystem::path const& dir) {
  auto const canonical = dir / kRootMarkerFile;
  if (std::filesystem::exists(canonical)) {
    return canonical;
  }
#if MOCHI_INTERNAL
  auto const legacy = dir / kRootMarkerFileLegacy;
  if (std::filesystem::exists(legacy)) {
    return legacy;
  }
#endif
  return std::nullopt;
}

std::filesystem::path superdex::robotics::NormalizeBotPath(std::filesystem::path const& path) {
  auto normalized = std::filesystem::absolute(path).lexically_normal();
  // "a/b/" normalizes with a trailing separator, which would make an otherwise equal path compare
  // unequal; filename() is empty exactly in that case.
  if (!normalized.has_filename()) {
    normalized = normalized.parent_path();
  }
  return normalized;
}

bool superdex::robotics::IsAssetRoleSubdirName(std::string_view name) {
  return std::any_of(
      kAssetRoleSubdirs.begin(), kAssetRoleSubdirs.end(), [name](std::string_view subdir) {
        return EqualsCaseInsensitive(name, subdir);
      });
}

std::filesystem::path superdex::robotics::FindAssetBaseFolder(
    std::filesystem::path const& originDir) {
  auto dir = NormalizeBotPath(originDir);
  // Ascend to the nearest role-folder ancestor and take its parent, so a file partitioned into a
  // subfolder of one (`cad/internal/`) still anchors on the asset rather than on the subfolder.
  for (auto current = dir;;) {
    auto parent = current.parent_path();
    if (parent.empty() || parent == current) {
      break; // filesystem root: no role folder on the way up
    }
    if (IsAssetRoleSubdirName(current.filename().string())) {
      return parent;
    }
    current = std::move(parent);
  }
  return dir;
}

AssetFolderLayout superdex::robotics::DetectAssetFolderLayout(
    std::filesystem::path const& originDir) {
  auto const dir = NormalizeBotPath(originDir);
  // FindAssetBaseFolder moves off `dir` exactly when it found a role-folder ancestor, so this is
  // the same walk asking "are we inside a role folder", subfolders included.
  if (FindAssetBaseFolder(dir) != dir) {
    return AssetFolderLayout::RoleSubdirs;
  }
  std::error_code ec;
  for (auto const& subdir : kAssetRoleSubdirs) {
    if (std::filesystem::is_directory(dir / std::filesystem::path(subdir), ec)) {
      return AssetFolderLayout::RoleSubdirs;
    }
  }
  return AssetFolderLayout::Flat;
}

std::filesystem::path superdex::robotics::AssetRoleFolderForWrite(
    std::filesystem::path const& originDir,
    std::string_view roleSubdir) {
  auto base = FindAssetBaseFolder(originDir);
  if (roleSubdir.empty()) {
    return base;
  }
  if (EqualsCaseInsensitive(roleSubdir, kIntermediatesSubdir) ||
      DetectAssetFolderLayout(originDir) == AssetFolderLayout::RoleSubdirs) {
    return base / std::filesystem::path(roleSubdir);
  }
  return base;
}

std::optional<std::filesystem::path> superdex::robotics::FindBotsRoot(
    std::filesystem::path const& startPath) {
  auto dir = NormalizeBotPath(startPath);
  if (!std::filesystem::is_directory(dir)) {
    dir = dir.parent_path();
  }
  while (!dir.empty()) {
    if (FindRootMarker(dir).has_value()) {
      return dir;
    }
    auto parent = dir.parent_path();
    if (parent == dir) {
      break;
    }
    dir = parent;
  }
  return std::nullopt;
}

// Returns true if `name` is a valid tag identifier (the part after the leading '@').
// Identifier rules: every character must be a letter, digit, or underscore.
static bool IsValidTagIdentifier(std::string_view name) {
  if (name.empty()) {
    return false;
  }
  auto isAllowed = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; };
  for (char c : name) {
    if (!isAllowed(c)) {
      return false;
    }
  }
  return true;
}

// Returns the number of path components that are exactly "..".
// Backslashes are normalized to forward slashes so behavior is consistent on POSIX
// (where std::filesystem treats only '/' as a separator) and Windows. Leading
// separators are stripped so that prefixes like "//" don't get parsed as a Windows
// UNC root-name (which would hide a following ".." from the iterator).
static int CountParentDirComponents(std::string_view path) {
  std::string normalized(path);
  std::replace(normalized.begin(), normalized.end(), '\\', '/');
  size_t const firstNonSlash = normalized.find_first_not_of('/');
  normalized.erase(0, firstNonSlash == std::string::npos ? normalized.size() : firstNonSlash);
  int count = 0;
  for (auto const& component : std::filesystem::path{normalized}) {
    if (component == "..") {
      ++count;
    }
  }
  return count;
}

// Returns true if any path component is exactly "..".
static bool ContainsParentDirComponent(std::string_view path) {
  return CountParentDirComponents(path) > 0;
}

// Used to determine of a .superdex_root file should be parsed.
static bool IsWhitespaceOnly(std::string_view s) {
  return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c) != 0; });
}

ParsedRootFile superdex::robotics::ParseRootFile(
    std::filesystem::path const& rootFilePath,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  std::string contents = ReadFileString(rootFilePath, error);
  MOCHI_ERROR_RETURN(error, {});

  ParsedRootFile result;
  result.rootDir = rootFilePath.parent_path();

  if (contents.empty() || IsWhitespaceOnly(contents)) {
    return result;
  }

  std::unordered_map<std::string, std::string> rawTags;
  bool const success =
      SReflect::FromJsonString(rawTags, contents, SReflect::DeserializeFlags::Default);
  if (!success) {
    MOCHI_ERROR_SET(
        error, "Failed to deserialize .superdex_root JSON (expected object of @tag : path).");
    return {};
  }

  for (auto const& [key, valueStr] : rawTags) {
    if (key.empty() || key.front() != '@' ||
        !IsValidTagIdentifier(std::string_view(key).substr(1))) {
      MOCHI_LOG_ERROR(
          "Invalid tag key '%s' in .superdex_root: must start with '@' followed by only letters, digits, or underscores.",
          key.c_str());
      MOCHI_ERROR_SET(error, "Invalid tag key in .superdex_root.");
      return {};
    }
    std::filesystem::path tagPath(valueStr);
    if (tagPath.is_relative()) {
      tagPath = result.rootDir / tagPath;
    }
    auto const resolved = NormalizeBotPath(tagPath);
    if (!FindRootMarker(resolved).has_value()) {
      /* Tag target directory has no .superdex_root marker. Log a warning
       * but still store the tag — it may never be referenced at runtime
       * if the environment does not use that tag. */
      MOCHI_LOG_WARNING(
          "Tag '%s' in .superdex_root targets '%s' which has no '%s' marker — "
          "will fail if this tag is actually referenced.",
          key.c_str(),
          resolved.generic_string().c_str(),
          std::string(kRootMarkerFile).c_str());
    }
    result.tags.emplace(key, resolved);
  }

  return result;
}

std::filesystem::path superdex::robotics::ResolveBotPath(
    std::string_view input,
    std::filesystem::path const& basePath,
    int maxParentDepth,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  if (input.empty()) {
    return {};
  }

  // A file-relative path may ascend at most maxParentDepth parent directories (0 = descendant-only,
  // the default for bots). // and @tag remainders remain strictly descendant (checked per-branch).
  if (CountParentDirComponents(input) > maxParentDepth) {
    MOCHI_LOG_ERROR("Error resolving bot path: %s", std::string(input).c_str());
    MOCHI_ERROR_SET(
        error, "Bot path ascends beyond the permitted number of parent ('..') directories.");
    return {};
  }

  if (input.starts_with(kTagPrefix)) {
    // Find the end of the tag (first '/' or '\\').
    size_t const slashPos = input.find_first_of("/\\");
    std::string_view tagName =
        slashPos == std::string_view::npos ? input : input.substr(0, slashPos);
    std::string_view remainder =
        slashPos == std::string_view::npos ? std::string_view{} : input.substr(slashPos + 1);

    auto rootDir = FindBotsRoot(basePath);
    if (!rootDir) {
      MOCHI_LOG_ERROR("Error resolving bot path: %s", std::string(input).c_str());
      MOCHI_ERROR_SET(error, "No .superdex_root marker found for tagged path.");
      return {};
    }
    auto parsed = ParseRootFile(*FindRootMarker(*rootDir), error);
    MOCHI_ERROR_RETURN(error, {});
    auto it = parsed.tags.find(std::string(tagName));
    if (it == parsed.tags.end()) {
      MOCHI_LOG_ERROR("Error resolving bot path: %s", std::string(input).c_str());
      MOCHI_ERROR_SET(error, "Tag is not defined in .superdex_root.");
      return {};
    }
    if (remainder.empty()) {
      return it->second;
    }
    if (std::filesystem::path(remainder).is_absolute()) {
      MOCHI_LOG_ERROR("Error resolving bot path: %s", std::string(input).c_str());
      MOCHI_ERROR_SET(error, "Tagged-path remainder must not be an absolute path.");
      return {};
    }
    // maxParentDepth relaxes only the file-relative form; a @tag reference stays descendant-only.
    if (ContainsParentDirComponent(remainder)) {
      MOCHI_LOG_ERROR("Error resolving bot path: %s", std::string(input).c_str());
      MOCHI_ERROR_SET(error, "Tagged-path remainder must not contain '..' components.");
      return {};
    }
    return NormalizeBotPath(it->second / remainder);
  }

  if (input.starts_with(kRootPrefix)) {
    auto rootDir = FindBotsRoot(basePath);
    if (!rootDir) {
      MOCHI_LOG_ERROR("Error resolving bot path: %s", std::string(input).c_str());
      MOCHI_ERROR_SET(error, "Path uses // prefix but no marker found");
      return {};
    }
    auto remainder = input.substr(kRootPrefix.size());
    if (std::filesystem::path(remainder).is_absolute()) {
      MOCHI_LOG_ERROR("Error resolving bot path: %s", std::string(input).c_str());
      MOCHI_ERROR_SET(error, "// root-prefixed path remainder must not be an absolute path.");
      return {};
    }
    // maxParentDepth relaxes only the file-relative form; a // reference stays descendant-only.
    if (ContainsParentDirComponent(remainder)) {
      MOCHI_LOG_ERROR("Error resolving bot path: %s", std::string(input).c_str());
      MOCHI_ERROR_SET(error, "// root-prefixed path remainder must not contain '..' components.");
      return {};
    }
    return NormalizeBotPath(*rootDir / remainder);
  }

  std::filesystem::path inputAsPath(input);
  if (inputAsPath.is_absolute()) {
    MOCHI_LOG_ERROR("Error resolving bot path: %s", std::string(input).c_str());
    MOCHI_ERROR_SET(
        error, "Absolute bot paths are not allowed; use //, @tag/, or a file-relative path.");
    return {};
  }

  // Plain relative path; join with basePath's directory.
  auto const dir = GetDirectoryPath(basePath);
  return NormalizeBotPath(dir / inputAsPath);
}

DynamicString superdex::robotics::UnresolveBotPath(
    std::filesystem::path const& absPath,
    std::filesystem::path const& basePath,
    int maxParentDepth,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  // Empty input -> empty output.
  if (absPath.empty()) {
    return {};
  }
  // Defensive: an already-relative input has no canonical form to derive; pass through.
  if (!absPath.is_absolute()) {
    return DynamicString{absPath.generic_string()};
  }

  // Normalized, not canonicalized: a bot may name a file this build does not have, and that
  // reference has to anchor to the same text from the source tree as from an extracted archive.
  auto const canonical = NormalizeBotPath(absPath);

  // Helper: compute a relative form against `dir` if it ascends at most `allowedParentDepth`
  // parent directories (0 = descendant-only).
  auto tryRelative = [&canonical](
                         std::filesystem::path const& dir,
                         int allowedParentDepth) -> std::optional<std::string> {
    if (dir.empty()) {
      return std::nullopt;
    }
    /* Lexical, to match: std::filesystem::relative would canonicalize both operands and undo it. */
    auto rel = canonical.lexically_relative(NormalizeBotPath(dir));
    if (rel.empty()) {
      return std::nullopt;
    }
    auto s = rel.generic_string();
    if (CountParentDirComponents(s) > allowedParentDepth) {
      return std::nullopt;
    }
    return s;
  };

  // The // and @tag forms can never encode a parent ('..') hop, so they resolve descendant-only for
  // every caller -- independent of maxParentDepth, which governs only the file-relative form (rule
  // 1) below.
  constexpr int kDescendantOnly = 0;

  // Rule 1: file-relative to basePath's parent directory, ascending at most maxParentDepth levels.
  // Normalized the same way as `canonical` above, so both sides of the comparison agree on what a
  // path means; see NormalizeBotPath for why that normalization does not resolve symlinks.
  auto const baseDir = NormalizeBotPath(GetDirectoryPath(basePath));
  if (auto rel = tryRelative(baseDir, maxParentDepth)) {
    return DynamicString{*rel};
  }

  // Locate the nearest .superdex_root for rules 2 and 3.
  auto const rootDirOpt = FindBotsRoot(basePath);

  // Rule 2: //root-prefixed descendant of the nearest bots root (always descendant-only).
  if (rootDirOpt) {
    if (auto rel = tryRelative(*rootDirOpt, kDescendantOnly)) {
      return DynamicString{std::string(kRootPrefix) + *rel};
    }
  }

  // Rule 3: @tag/... — pick the most specific tag whose directory is an
  // ancestor of the absolute path. Specificity is path-component count of the
  // tag's target directory (semantic depth). std::map iteration is sorted by
  // tag name, so equal-depth ties are broken by lexicographically smallest
  // name (and the strict `>` keeps the first one seen).
  if (rootDirOpt) {
    auto const parsed = ParseRootFile(*FindRootMarker(*rootDirOpt), error);
    MOCHI_ERROR_RETURN(error, {});
    std::string bestTag;
    std::string bestRel;
    size_t bestDepth = 0;
    for (auto const& [tagName, tagDir] : parsed.tags) {
      // A @tag reference is always descendant-only, regardless of maxParentDepth.
      auto rel = tryRelative(tagDir, kDescendantOnly);
      if (!rel) {
        continue;
      }
      auto const depth = static_cast<size_t>(std::distance(tagDir.begin(), tagDir.end()));
      if (bestTag.empty() || depth > bestDepth) {
        bestTag = tagName;
        bestRel = std::move(*rel);
        bestDepth = depth;
      }
    }
    if (!bestTag.empty()) {
      // bestRel may be "." when the path equals the tag directory itself.
      if (bestRel == ".") {
        return DynamicString{bestTag};
      }
      return DynamicString{bestTag + "/" + bestRel};
    }
  }

  // No relative form available. Absolute bot paths are not allowed.
  MOCHI_LOG_ERROR(
      "Cannot express path as a bot-relative reference: %s", canonical.generic_string().c_str());
  MOCHI_ERROR_SET(
      error,
      "Path is not reachable via the base file directory, the //-root, or any @tag; absolute bot paths are not allowed.");
  return {};
}

static void
MakeLinkPathsRelative(BotLinkPrefab& link, std::filesystem::path const& basePath, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MakePathRelative(link.renderModelFile, basePath, kBotPathMaxParentDepth, error);
  MOCHI_ERROR_RETURN(error);
  MakePathRelative(link.shapeFile, basePath, kBotPathMaxParentDepth, error);
  MOCHI_ERROR_RETURN(error);
  for (auto& sensor : link.sensors) {
    MakeParamsPathRelative(sensor.params, basePath, error);
    MOCHI_ERROR_RETURN(error);
  }
}

static void
MakeLinkPathsAbsolute(BotLinkPrefab& link, std::filesystem::path const& basePath, Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (!link.renderModelFile.empty()) {
    MakePathAbsolute(link.renderModelFile, basePath, kBotPathMaxParentDepth, error);
  }
  if (!link.shapeFile.empty()) {
    MakePathAbsolute(link.shapeFile, basePath, kBotPathMaxParentDepth, error);
  }
  for (auto& sensor : link.sensors) {
    MakeParamsPathAbsolute(sensor.params, basePath, error);
    MOCHI_ERROR_RETURN(error);
  }
}

void superdex::robotics::MakePathRelative(
    DynamicString& path,
    std::filesystem::path const& basePath,
    int maxParentDepth,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (path.empty()) {
    return;
  }
  std::string_view const pathView(path.c_str(), path.size());
  if (pathView.starts_with(kRootPrefix) || pathView.starts_with(kTagPrefix)) {
    return;
  }
  std::filesystem::path fsPath(path.c_str());
  if (!fsPath.is_absolute()) {
    return;
  }
  auto const result = UnresolveBotPath(fsPath, basePath, maxParentDepth, error);
  MOCHI_ERROR_RETURN(error);
  path = result;
}

void superdex::robotics::MakePathAbsolute(
    DynamicString& path,
    std::filesystem::path const& basePath,
    int maxParentDepth,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (path.empty()) {
    return;
  }
  std::string_view pathView(path.c_str(), path.size());
  auto resolved = ResolveBotPath(pathView, basePath, maxParentDepth, error);
  MOCHI_ERROR_RETURN(error);
  if (resolved.empty()) {
    return;
  }
  path = DynamicString(resolved.generic_string());
}

void superdex::robotics::MakeParamsPathRelative(
    DynamicString& params,
    std::filesystem::path const& basePath,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  // A `params` field is path-or-inline-JSON; only resolve the path form (inline JSON has no path).
  if (IsInlineJson(std::string_view(params.c_str(), params.size()))) {
    return;
  }
  MakePathRelative(params, basePath, kBotPathMaxParentDepth, error);
}

void superdex::robotics::MakeParamsPathAbsolute(
    DynamicString& params,
    std::filesystem::path const& basePath,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  // A `params` field is path-or-inline-JSON; only resolve the path form (inline JSON has no path).
  if (IsInlineJson(std::string_view(params.c_str(), params.size()))) {
    return;
  }
  MakePathAbsolute(params, basePath, kBotPathMaxParentDepth, error);
}

void superdex::robotics::MakePathsRelative(
    BotPrefab& botPrefab,
    std::filesystem::path const& basePath,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  for (auto& link : botPrefab.links) {
    MakeLinkPathsRelative(link, basePath, error);
  }
}

void superdex::robotics::MakePathsAbsolute(
    BotPrefab& botPrefab,
    std::filesystem::path const& basePath,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  for (auto& link : botPrefab.links) {
    MakeLinkPathsAbsolute(link, basePath, error);
  }
}

void superdex::robotics::MakePathsRelative(
    ModBotPrefab& modBotPrefab,
    std::filesystem::path const& basePath,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MakePathRelative(modBotPrefab.base, basePath, kBotPathMaxParentDepth, error);
  for (auto& mod : modBotPrefab.modifications) {
    std::visit(
        OverloadVisitor{
            [&](AttachBot& m) {
              MakePathRelative(m.path, basePath, kBotPathMaxParentDepth, error);
            },
            [&](ReplaceLinkWithBot& m) {
              MakePathRelative(m.path, basePath, kBotPathMaxParentDepth, error);
            },
            [&](AttachLink& m) { MakeLinkPathsRelative(m.link, basePath, error); },
            [&](ReplaceLink& m) { MakeLinkPathsRelative(m.link, basePath, error); },
        },
        mod);
  }
}

void superdex::robotics::MakePathsAbsolute(
    ModBotPrefab& modBotPrefab,
    std::filesystem::path const& basePath,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MakePathAbsolute(modBotPrefab.base, basePath, kBotPathMaxParentDepth, error);
  MOCHI_ERROR_RETURN(error);
  for (auto& mod : modBotPrefab.modifications) {
    std::visit(
        OverloadVisitor{
            [&](AttachBot& m) {
              MakePathAbsolute(m.path, basePath, kBotPathMaxParentDepth, error);
            },
            [&](ReplaceLinkWithBot& m) {
              MakePathAbsolute(m.path, basePath, kBotPathMaxParentDepth, error);
            },
            [&](AttachLink& m) { MakeLinkPathsAbsolute(m.link, basePath, error); },
            [&](ReplaceLink& m) { MakeLinkPathsAbsolute(m.link, basePath, error); },
        },
        mod);
    MOCHI_ERROR_RETURN(error);
  }
}

BotPrefab superdex::robotics::LoadBotPrefabFromUrdfFile(std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  BotPrefab botPrefab;
  LoadBotPrefabFromUrdfFile(botPrefab, path, nullptr, error);
  MOCHI_ERROR_RETURN(error, {});
  return botPrefab;
}

BotPrefab superdex::robotics::LoadBotPrefabFromUrdfFile(
    std::string_view path,
    UrdfMeshReferences& meshRefs,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  BotPrefab botPrefab;
  LoadBotPrefabFromUrdfFile(botPrefab, path, &meshRefs, error);
  MOCHI_ERROR_RETURN(error, {});
  return botPrefab;
}

void superdex::robotics::SaveToFile(
    BotPrefab const& botPrefab,
    std::string_view path,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (IsBotPath(path)) {
    EnsureDirectoriesCreated(path, error);
    MOCHI_ERROR_RETURN(error);
    auto temp = botPrefab;
    MakePathsRelative(temp, path, error);
    MOCHI_ERROR_RETURN(error);
    bool const success = SReflect::SaveToJsonFile(temp, std::string(path).c_str());
    if (!success) {
      MOCHI_LOG_ERROR("Failed to save BotPrefab to: %s", std::string(path).c_str());
    }
    MOCHI_ERROR_IF(!success, error, "Failed to save BotPrefab to .superdex_bot (JSON) file");
  } else {
    MOCHI_LOG_ERROR("Failed to save BotPrefab to: %s", std::string(path).c_str());
    MOCHI_ERROR_SET(error, "Unsupported file extension; expected .superdex_bot");
  }
}

DynamicString superdex::robotics::WriteToJsonString(BotPrefab const& botPrefab, Error& error) {
  MOCHI_ERROR_RETURN(error, "");
  auto json = SReflect::ToJsonString(botPrefab, true);
  return DynamicString{json.c_str(), json.size()};
}

DynamicString superdex::robotics::WriteToJsonString(
    ModBotPrefab const& modBotPrefab,
    Error& error) {
  MOCHI_ERROR_RETURN(error, "");
  auto json = SReflect::ToJsonString(modBotPrefab, true);
  return DynamicString{json.c_str(), json.size()};
}

bool superdex::robotics::IsInlineJson(std::string_view value) {
  for (char const c : value) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      continue;
    }
    return c == '{' || c == '[';
  }
  return false; // empty or all-whitespace -> treat as (empty) path, not inline JSON
}

namespace {
// Fold each sensor's legacy typeName/paramsFile keys into type/params and clear the legacy fields.
void ApplyLegacyFieldsToSensors(DynamicArray<BotSensorPrefab>& sensors) {
  for (auto& sensor : sensors) {
    if (sensor.type.empty() && !sensor._legacyTypeName.empty()) {
      sensor.type = sensor._legacyTypeName;
    }
    if (sensor.params.empty() && !sensor._legacyParamsFile.empty()) {
      sensor.params = sensor._legacyParamsFile;
    }
    sensor._legacyTypeName = {};
    sensor._legacyParamsFile = {};
  }
}
} // namespace

void superdex::robotics::ApplyLegacyBotSensorFields(BotPrefab& botPrefab) {
  for (auto& link : botPrefab.links) {
    ApplyLegacyFieldsToSensors(link.sensors);
  }
}

void superdex::robotics::ApplyLegacyBotSensorFields(ModBotPrefab& modBotPrefab) {
  // Mod bots carry inline links (with sensors) via AttachLink/ReplaceLink modifications; fold their
  // legacy sensor fields exactly as for a base bot. AttachBot/ReplaceLinkWithBot instead reference
  // external .superdex_bot files, whose sensors are folded when those files are loaded.
  for (auto& mod : modBotPrefab.modifications) {
    if (auto* attach = std::get_if<AttachLink>(&mod)) {
      ApplyLegacyFieldsToSensors(attach->link.sensors);
    } else if (auto* replace = std::get_if<ReplaceLink>(&mod)) {
      ApplyLegacyFieldsToSensors(replace->link.sensors);
    }
  }
}

namespace {
// Restore the pre-migration default for a legacy joint. BotJointPrefab::type used to default to
// ArticulatedJointType::Hard but now defaults to ArticulatedJointType::Invalid, so files authored
// before the change omit "type" and deserialize as Invalid.
void ApplyLegacyJointTypeDefault(BotJointPrefab& joint) {
  if (joint.type == ArticulatedJointType::Invalid) {
    joint.type = ArticulatedJointType::Hard;
  }
}
} // namespace

void superdex::robotics::ApplyLegacyBotJointTypes(BotPrefab& botPrefab) {
  for (auto& joint : botPrefab.joints) {
    ApplyLegacyJointTypeDefault(joint);
  }
}

void superdex::robotics::ApplyLegacyBotJointTypes(ModBotPrefab& modBotPrefab) {
  // Mod bots carry inline connecting joints via AttachLink/AttachBot modifications; default their
  // type exactly as for a base bot's joints. BuildBot pushes these joints into the assembled bot
  // as-is, so without this a legacy mod's connecting joint stays Invalid. ReplaceLink /
  // ReplaceLinkWithBot carry no joint.
  for (auto& mod : modBotPrefab.modifications) {
    if (auto* attach = std::get_if<AttachLink>(&mod)) {
      ApplyLegacyJointTypeDefault(attach->joint);
    } else if (auto* attachBot = std::get_if<AttachBot>(&mod)) {
      ApplyLegacyJointTypeDefault(attachBot->joint);
    }
  }
}

BotPrefab superdex::robotics::ReadFromJsonString(std::string_view json, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  BotPrefab outBotPrefab;
  bool const success = SReflect::FromJsonString(
      outBotPrefab, std::string(json), SReflect::DeserializeFlags::Default);
  MOCHI_ERROR_IF(!success, error, "Failed to deserialize BotPrefab from JSON string");
  MOCHI_ERROR_RETURN(error, {});
  ApplyLegacyBotSensorFields(outBotPrefab);
  RebuildBotData(outBotPrefab, error);
  return outBotPrefab;
}

void superdex::robotics::SaveToFile(
    ModBotPrefab const& modBotPrefab,
    std::string_view path,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (IsBotPath(path)) {
    EnsureDirectoriesCreated(path, error);
    MOCHI_ERROR_RETURN(error);
    auto temp = modBotPrefab;
    MakePathsRelative(temp, path, error);
    MOCHI_ERROR_RETURN(error);
    bool const success = SReflect::SaveToJsonFile(temp, std::string(path).c_str());
    if (!success) {
      MOCHI_LOG_ERROR("Failed to save ModBotPrefab to: %s", std::string(path).c_str());
    }
    MOCHI_ERROR_IF(!success, error, "Failed to save ModBotPrefab to .superdex_bot (JSON) file");
  } else {
    MOCHI_LOG_ERROR("Failed to save ModBotPrefab to: %s", std::string(path).c_str());
    MOCHI_ERROR_SET(error, "Unsupported file extension; expected .superdex_bot");
  }
}

#if MOCHI_INTERNAL
void superdex::robotics::SaveToFile(
    BotScenePrefab const& scenePrefab,
    std::string_view path,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      !path.ends_with(kBotSceneExtension),
      error,
      "Unsupported file format; expected .mochi_bot_scene");
  MOCHI_ERROR_RETURN(error);
  EnsureDirectoriesCreated(std::filesystem::path(std::string(path)), error);
  MOCHI_ERROR_RETURN(error);
  // Convert any absolute in-memory paths back to the on-disk relative form (//, @tag/, or
  // file-relative), mirroring the resolution done by LoadBotScenePrefabFromFile (ResolvePaths).
  // Symmetric with SaveToFile(BotPrefab) / SaveToFile(ModBotPrefab). MakePathRelative is a
  // no-op on paths that are already relative.
  auto temp = scenePrefab;
  std::filesystem::path const basePath{std::string(path)};
  MakePathRelative(temp.scene.baseScene, basePath, kBotPathMaxParentDepth, error);
  for (auto& prefab : temp.scene.spawnablePrefabs) {
    MakePathRelative(prefab.path, basePath, kBotPathMaxParentDepth, error);
  }
  for (auto& bot : temp.bots) {
    MakePathRelative(bot.path, basePath, kBotPathMaxParentDepth, error);
    for (auto& ctrl : bot.controllers) {
      MakeParamsPathRelative(ctrl.params, basePath, error);
    }
  }
  MOCHI_ERROR_RETURN(error);
  // Each controller's initArgs is a JsonString-tagged field, so plain SReflect serialization emits
  // it as a nested "initArgs" object (empty ones omitted via NoSerializeDefaults). No bespoke
  // splicing needed.
  SaveParamsToFile(temp, path, error);
}
#endif // MOCHI_INTERNAL

static void HashFileIntoState(XXH3_state_t* state, std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto const data = mochi::ReadFileBytes(std::filesystem::path(path), error);
  MOCHI_ERROR_RETURN(error);
  XXH3_64bits_update(state, data.data(), data.size());
}

DynamicString superdex::robotics::HashBotFile(std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  XXH3_state_t* state = XXH3_createState();
  MOCHI_DEFER(XXH3_freeState(state));
  XXH3_64bits_reset(state);
  std::string botPath = std::string(path);
  // 1. If this is a .mochi_bot_archive file, we need to modify path to
  // point to the target .mochi_bot file inside the archive.
  if (IsBotArchivePath(path) && std::filesystem::is_regular_file(std::string(path))) {
    auto extractedDir = ExtractBotArchiveToCache(path, error);
    botPath = GetExtractedBotArchiveTarget(extractedDir, error);
  }
  // 2. Load the flattened BotPrefab.
  //
  // For base bots (BotPrefab) this is a no-op (load -> serialize round-trip).
  //
  // For mod bots (ModBotPrefab), LoadBotPrefabFromFile calls BuildBot
  // which resolves the base reference, applies all modifications (AttachBot,
  // AttachLink, ReplaceLink, etc.), and returns a fully flattened BotPrefab.
  // Hashing the flattened result means we only capture the *effective* bot
  // configuration — the one actually used at runtime — rather than recursively
  // hashing every referenced .mochi_bot in the mod bot chain.
  //
  // Trade-off: a mod bot's .mochi_bot file could change (e.g. reordering
  // operations) without changing the hash, as long as the resulting BotPrefab is
  // identical. This is desirable — it avoids unnecessary UE reimports when mod bot
  // operations are reordered but produce the same effective result.
  auto botPrefab = LoadBotPrefabFromFile(botPath, error);
  MOCHI_ERROR_RETURN(error, {});
  // 3. Hash referenced model files for each link (uses absolute paths from LoadBotPrefabFromFile)
  /* A file the tree does not have contributes nothing and warns, rather than failing the hash. A
   * bot may name an asset that is absent -- the open-source export strips some -- and such a tree
   * must still be hashable and archivable, since archiving hashes the bot for its metadata. The
   * resulting hash differs from a tree where the file is present, which is correct: they are
   * different trees. */
  auto const hashFileIfPresent = [&](DynamicString const& file) {
    if (file.empty()) {
      return;
    }
    if (!std::filesystem::exists(std::string(file))) {
      MOCHI_LOG_WARNING(
          "HashBotFile: '%s' does not exist; hashing without its contents", file.c_str());
      return;
    }
    HashFileIntoState(state, file.c_str(), error);
  };
  for (auto const& link : botPrefab.links) {
    hashFileIfPresent(link.renderModelFile);
    MOCHI_ERROR_RETURN(error, {});
    hashFileIfPresent(link.shapeFile);
    MOCHI_ERROR_RETURN(error, {});
  }
  /* 4. Convert link asset paths back to relative for the JSON we hash.
   *
   *    Anchor at the bots root, NOT at botPath: anchoring at botPath would embed the
   *    mod bot's depth in the bots tree (e.g. "../../../arms/fr3/...") into the
   *    hashed JSON, so moving a bot would change its hash even when its content didn't.
   *    Anchoring at the bots root yields short, location-independent paths like
   *    "arms/fr3/collision/fr3_link0.mochi.h5", and matches what ArchiveBot writes into
   *    the zip layout. We pass `botsRoot / kRootMarkerFile` so MakePathsRelative's
   *    internal parent_path() recovers the root directory.
   *
   *    Fail loudly if no marker is reachable: a hash that can't be compared across
   *    processes/hosts is worse than no hash. */
  auto const botsRoot = FindBotsRoot(botPath);
  MOCHI_ERROR_IF_NOT(
      botsRoot.has_value(),
      error,
      "HashBotFile requires a .superdex_root marker reachable from the bot file.");
  MOCHI_ERROR_RETURN(error, {});
  MakePathsRelative(botPrefab, *botsRoot / kRootMarkerFile, error);
  MOCHI_ERROR_RETURN(error, {});
  // 5. Reserialize BotPrefab; hash the JSON string.
  auto json = WriteToJsonString(botPrefab, error);
  MOCHI_ERROR_RETURN(error, {});
  XXH3_64bits_update(state, json.c_str(), json.size());
  XXH64_hash_t const hash = XXH3_64bits_digest(state);
  std::array<char, 17> hex{};
  std::snprintf(hex.data(), hex.size(), "%016" PRIx64, hash);
  return DynamicString{hex.data()};
}

DynamicString superdex::robotics::HashGenericFile(std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  XXH3_state_t* state = XXH3_createState();
  MOCHI_DEFER(XXH3_freeState(state));
  XXH3_64bits_reset(state);
  HashFileIntoState(state, path, error);
  MOCHI_ERROR_RETURN(error, {});
  XXH64_hash_t const hash = XXH3_64bits_digest(state);
  std::array<char, 17> hex{};
  std::snprintf(hex.data(), hex.size(), "%016" PRIx64, hash);
  return DynamicString{hex.data()};
}

void superdex::robotics::SaveToUrdfFile(
    BotPrefab const& botPrefab,
    std::string_view path,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  ExportBotPrefabToUrdfFile(botPrefab, path, error);
}

DynamicString superdex::robotics::SaveToUrdfString(BotPrefab const& botPrefab, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  std::string str = ExportBotPrefabToUrdfXml(botPrefab, error);
  return DynamicString{str};
}

BotPrefab superdex::robotics::LoadBotPrefabFromUrdfString(
    std::string_view xmlString,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  BotPrefab botPrefab;
  LoadBotPrefabFromUrdfXml(botPrefab, xmlString, error);
  MOCHI_ERROR_RETURN(error, {});
  return botPrefab;
}
