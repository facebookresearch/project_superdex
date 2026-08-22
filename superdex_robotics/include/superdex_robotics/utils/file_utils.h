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
#include <superdex_physics.h>
#include <superdex_robotics/superdex_robotics.h>
#if MOCHI_INTERNAL
#include <superdex_robotics/internal/bot_scene.h>
#include <superdex_robotics/internal/internal.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <map>
#include <optional>
#include <string>

namespace superdex::robotics {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constexpr std::string_view kRootPrefix = "//";
constexpr std::string_view kTagPrefix = "@";
constexpr std::string_view kRootMarkerFile = ".superdex_root";
constexpr std::string_view kBotExtension = ".superdex_bot";
constexpr std::string_view kBotSceneExtension = ".mochi_bot_scene";
constexpr std::string_view kBotTaskExtension = ".mochi_bot_task";
constexpr std::string_view kUrdfExtension = ".urdf";
constexpr std::string_view kCollisionSubdir = "collision";
constexpr std::string_view kRenderSubdir = "render";
constexpr std::string_view kCadSubdir = "cad";
constexpr std::string_view kIntermediatesSubdir = "intermediates";

/* The standard role folders, which sit side by side inside an asset's base folder (the folder
 * holding its `.superdex_bot` / prefab file). Fixed order, so a search that sweeps them is
 * deterministic. */
constexpr std::array<std::string_view, 4> kAssetRoleSubdirs = {
    kCadSubdir,
    kCollisionSubdir,
    kRenderSubdir,
    kIntermediatesSubdir};

/* Bot file references are descendant-only: a file-relative path may not ascend into a parent
 * directory. Pass this as the maxParentDepth argument to @ref ResolveBotPath / @ref
 * UnresolveBotPath / @ref MakePathRelative / @ref MakePathAbsolute for all bot (de)serialization.
 * (Callers that store loosely-coupled local clusters, e.g. Studio's .StudioProcessing pipelines,
 * opt into a small positive depth instead.) */
constexpr int kBotPathMaxParentDepth = 0;

// ---------------------------------------------------------------------------
// Asset folder layout
// ---------------------------------------------------------------------------

/**
 * @brief How one asset's files are arranged relative to its base folder.
 */
enum class AssetFolderLayout {
  /// Models live in @ref kAssetRoleSubdirs children of the base folder.
  RoleSubdirs,
  /// Models live directly in the base folder, alongside the bot / prefab file.
  Flat,
};

/**
 * @brief Case-insensitive string equality.
 *
 * The asset layout is compared this way throughout -- role folder names here, asset base names in
 * Studio's discovery -- so that a tree authored on a case-insensitive filesystem resolves the same
 * way on a case-sensitive one.
 */
[[nodiscard]] inline bool EqualsCaseInsensitive(std::string_view a, std::string_view b) {
  return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](unsigned char x, unsigned char y) {
    return std::tolower(x) == std::tolower(y);
  });
}

/**
 * @brief True if @p name is one of @ref kAssetRoleSubdirs.
 *
 * Compared case-insensitively, so a `Render/` authored on Windows is still recognized on Linux.
 */
[[nodiscard]] MOCHI_API bool IsAssetRoleSubdirName(std::string_view name);

/**
 * @brief Resolve the base folder of the asset that @p originDir belongs to.
 *
 * @details Ascends to the nearest role-folder ancestor (@ref IsAssetRoleSubdirName) and resolves to
 * its parent; a path with no role folder above it resolves to itself. Nothing on disk is consulted.
 *
 * This anchors lookups and default save paths on the asset itself rather than on a fixed number of
 * directory levels. Counting levels breaks on a flat asset, where it reaches past the asset into
 * the folder holding its siblings -- a namespace shared with unrelated assets. Ascending rather
 * than checking only the immediate name additionally handles a role folder partitioned into
 * subfolders, e.g. `cad/internal/` holding CAD sources that must not be open-sourced.
 *
 * @warning Being purely lexical, this treats any role-folder name on the way up as the asset's,
 * so an asset nested beneath a folder that happens to be named `cad` / `render` / `collision` /
 * `intermediates` anchors one level too high. No asset in the tree is laid out that way.
 *
 * @param[in] originDir Directory holding the file being worked on.
 *
 * @return The asset's base folder.
 */
[[nodiscard]] MOCHI_API std::filesystem::path FindAssetBaseFolder(
    std::filesystem::path const& originDir);

/**
 * @brief Classify the layout of the asset that @p originDir belongs to.
 *
 * @details @ref AssetFolderLayout::RoleSubdirs when @p originDir sits inside a role folder -- it is
 * one, or a subfolder of one, so the asset demonstrably uses them -- or when the base folder
 * already holds any role folder; @ref AssetFolderLayout::Flat otherwise. Only the second test
 * touches the filesystem.
 *
 * @param[in] originDir Directory holding the file being worked on.
 *
 * @return The layout in use.
 */
[[nodiscard]] MOCHI_API AssetFolderLayout
DetectAssetFolderLayout(std::filesystem::path const& originDir);

/**
 * @brief Directory that a newly written file of role @p roleSubdir belongs in, for a file being
 * worked on in @p originDir.
 *
 * @details `<base>/<roleSubdir>` under @ref AssetFolderLayout::RoleSubdirs, whether or not that
 * folder exists yet: exporting the render and collision models of a STEP opened from `cad/` fills
 * `render/` and `collision/` rather than dropping them beside the STEP. The base folder itself
 * under @ref AssetFolderLayout::Flat, which keeps a flat asset flat. Creating the directory is the
 * caller's job (see @ref EnsureDirectoriesCreated).
 *
 * @note @ref kIntermediatesSubdir is the deliberate exception, and always yields
 * `<base>/intermediates`: generated pipelines and scratch exports are bookkeeping rather than
 * asset content, so they stay out of a flat asset's folder.
 *
 * @param[in] originDir Directory holding the file being worked on.
 * @param[in] roleSubdir One of @ref kAssetRoleSubdirs. Empty yields the base folder.
 *
 * @return The directory to write into.
 */
[[nodiscard]] MOCHI_API std::filesystem::path AssetRoleFolderForWrite(
    std::filesystem::path const& originDir,
    std::string_view roleSubdir);

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/**
 * @brief Recognize a bot file path by extension.
 *
 * Always accepts the canonical @ref kBotExtension (`.superdex_bot`). In internal
 * builds (@c MOCHI_INTERNAL) it additionally accepts the legacy extensions for backward
 * and Meta internal compatibility; public/OSS builds recognize only the canonical extension.
 *
 * @param[in] path The file path to test.
 * @return @c true if @p path ends with a recognized bot extension.
 */
[[nodiscard]] MOCHI_API bool IsBotPath(std::string_view path);

/**
 * @brief Parsed contents of a `.superdex_root` file.
 *
 * Maps tag names (including the leading `@`) to absolute, validated directory
 * paths. Each tag directory is guaranteed to itself contain a
 * `.superdex_root` marker. Stored in a @c std::map so iteration order is
 * deterministic (lexicographic by tag name).
 */
struct ParsedRootFile {
  std::map<std::string, std::filesystem::path> tags;
  std::filesystem::path rootDir;
};

/**
 * @brief Locate the bots-root marker file within a single directory.
 *
 * Prefers the canonical @ref kRootMarkerFile (`.superdex_root`). In internal
 * builds (@c MOCHI_INTERNAL) it additionally falls back to the legacy marker
 * (see internal.h) for backward and Meta internal compatibility; public/OSS
 * builds recognize only the canonical marker.
 *
 * @param[in] dir The directory to inspect (not walked; this checks @p dir only).
 *
 * @return The path to the marker file present in @p dir, or @c std::nullopt if none.
 */
[[nodiscard]] MOCHI_API std::optional<std::filesystem::path> FindRootMarker(
    std::filesystem::path const& dir);

/**
 * @brief Find the bots root by walking up from a starting path.
 *
 * Walks up the directory tree from @p startPath looking for a `.superdex_root` marker file.
 *
 * @param[in] startPath A file or directory path to start searching from.
 *
 * @return The directory containing `.superdex_root`, or @c std::nullopt if not found.
 */
[[nodiscard]] MOCHI_API std::optional<std::filesystem::path> FindBotsRoot(
    std::filesystem::path const& startPath);

/**
 * @brief Parse a `.superdex_root` JSON file into its tag table.
 *
 * The file is a JSON object mapping `"@tagName"` keys to path strings.
 * Empty or whitespace-only files are valid and produce an empty tag table.
 * Relative tag values are resolved against the directory containing the
 * `.superdex_root` file, then @c weakly_canonical-ized. Each resolved tag
 * directory is verified to itself contain a `.superdex_root` marker.
 *
 * Any invalid entry (malformed key, key not matching the @ref kTagPrefix
 * identifier rules, or target directory missing a `.superdex_root` marker)
 * is treated as a hard error: the function logs the offending entry and returns
 * failure with no tags loaded.
 *
 * @param[in] rootFilePath Path to the `.superdex_root` file.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @return Parsed tag table on success, default-constructed on failure.
 */
[[nodiscard]] MOCHI_API ParsedRootFile
ParseRootFile(std::filesystem::path const& rootFilePath, superdex::Error& error);

/**
 * @brief Absolute + lexically normalized, with any trailing separator dropped.
 *
 * @details The single definition of what a bot path means, used wherever one is resolved, compared
 * or collected. Symlinks are deliberately NOT resolved: an asset tree can be a symlink farm
 * (buck-out file groups are), and resolving would place a file that exists in the link's target
 * tree while a file the build does not have -- which a bot may legitimately name -- has no link to
 * follow and stays where it was written. The two then have no relative form between them. Treating
 * paths as written keeps every reference in one tree, at the cost of not recognising two spellings
 * of one file as the same; an archive then carries such a file at both paths, which is what its
 * references require.
 *
 * @param[in] path Path to normalize; may name a file that does not exist.
 *
 * @return Normalized absolute path.
 */
[[nodiscard]] MOCHI_API std::filesystem::path NormalizeBotPath(std::filesystem::path const& path);

/**
 * @brief Resolve a bot path string into an absolute path.
 *
 * The single source of truth for path resolution rules used by all bot
 * file references:
 * - Empty input -> empty result.
 * - A file-relative @p input may ascend at most @p maxParentDepth parent
 *   (`..`) directories; more than that -> hard error. `@tag/...` and
 *   `//...` remainders are always descendant-only (any `..` -> hard error),
 *   regardless of @p maxParentDepth.
 * - Absolute input -> hard error (absolute bot paths are not allowed).
 * - `@tag/...` -> look up `@tag` in the nearest `.superdex_root` tag
 *   table (walking up from @p basePath) and join the remainder.
 * - `//...` -> resolve against the directory of the nearest
 *   `.superdex_root` marker (regardless of its tag table).
 * - Anything else -> join with @p basePath.parent_path() and
 *   @c weakly_canonical.
 *
 * @param[in] input Raw path string (e.g. from a `.superdex_bot` or .mochi_bot_scene field).
 * @param[in] basePath The referencing file (its parent directory is used as
 *                     the search/join origin).
 * @param[in] maxParentDepth Maximum number of parent (`..`) directories a file-relative path may
 *                     ascend. Pass @c 0 for descendant-only, which all bot references use. Callers
 *                     that store loosely-coupled local clusters (e.g. Studio's
 *                     `.StudioProcessing.json`) may pass a small positive value. No default: every
 *                     caller must state its policy explicitly.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @return Absolute resolved path.
 */
[[nodiscard]] MOCHI_API std::filesystem::path ResolveBotPath(
    std::string_view input,
    std::filesystem::path const& basePath,
    int maxParentDepth,
    superdex::Error& error);

/**
 * @brief Convert an absolute path to its canonical bot-path string form for serialization.
 *
 * This is the inverse of @ref ResolveBotPath.
 *
 * The result is always a valid @ref ResolveBotPath input that resolves to the
 * same absolute path. It is **not** guaranteed to be string-identical to the
 * original input, because multiple input forms may refer to the same path and
 * this function deterministically selects the highest-priority form per the
 * rules below. For example, an input `@assets/foo` whose target is also
 * reachable via `//` will round-trip as `//foo`.
 *
 * Selection rules, applied in order:
 *  1. File-relative — if @p absPath is reachable from @c basePath.parent_path()
 *     while ascending at most @p maxParentDepth parent (`..`) directories, emit
 *     that relative form. With the default @p maxParentDepth of @c 0 this is a
 *     plain descendant, as all bot references require.
 *  2. `//root-prefixed` — else, if @p absPath is a descendant of @c FindBotsRoot(basePath),
 *     emit `//+remainder` (always descendant-only).
 *  3. `@tag/...` — else, parse the nearest `.superdex_root` tag table and pick
 *     the tag whose target directory is the deepest (most path components) strict
 *     ancestor of @p absPath; ties are broken by lexicographically smallest tag name;
 *     emit `@tag/+remainder` (always descendant-only).
 *
 * If none of the above produce a valid relative form, the function reports a
 * hard error: absolute bot paths are not permitted on either end of resolution.
 *
 * Empty input -> empty result. Already-relative input is returned unchanged
 * (defensive; should not happen post-load).
 *
 * @param[in] absPath Absolute path to convert.
 * @param[in] basePath The referencing file (its parent directory and bots-root context
 *                     determine the chosen prefix form).
 * @param[in] maxParentDepth Maximum number of parent (`..`) directories the file-relative
 *                     form (rule 1) may ascend before falling back to `//` / `@tag`. Pass @c 0
 *                     for descendant-only, which all bot references use. The `//` and `@tag`
 *                     forms are unaffected and stay descendant-only. No default: every caller must
 *                     state its policy explicitly.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @return Canonical string representation acceptable as @ref ResolveBotPath input.
 */
[[nodiscard]] MOCHI_API DynamicString UnresolveBotPath(
    std::filesystem::path const& absPath,
    std::filesystem::path const& basePath,
    int maxParentDepth,
    superdex::Error& error);

/**
 * @brief Ensure that the parent directory of @p path exists, creating it if necessary.
 *
 * If the parent directory of @p path is empty or already exists, this is a no-op.
 *
 * @param[in] path A file path whose parent directory should be created.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void EnsureDirectoriesCreated(std::filesystem::path const& path, superdex::Error& error);

/**
 * @brief In-place wrapper for @ref UnresolveBotPath.
 *
 * @param[in,out] path The path to convert.
 * @param[in] basePath The referencing file (its parent directory and bots-root
 *                     context determine the chosen prefix form).
 * @param[in] maxParentDepth Forwarded to @ref UnresolveBotPath. Pass @c 0 for descendant-only,
 *                     which all bot references use. No default: every caller must state its policy.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void MakePathRelative(
    DynamicString& path,
    std::filesystem::path const& basePath,
    int maxParentDepth,
    superdex::Error& error);

/**
 * @brief In place wrapper for @ref ResolveBotPath
 *
 * @param[in,out] path The path to convert.
 * @param[in] basePath The base directory (or a file whose parent directory is used).
 * @param[in] maxParentDepth Forwarded to @ref ResolveBotPath. Pass @c 0 for descendant-only,
 *                     which all bot references use. No default: every caller must state its policy.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @note If the path is empty or already absolute, it is left unchanged.
 */
MOCHI_API void MakePathAbsolute(
    DynamicString& path,
    std::filesystem::path const& basePath,
    int maxParentDepth,
    superdex::Error& error);

/**
 * @brief @ref MakePathRelative variant for a controller/sensor `params` field, which may hold
 * either a file path or inline JSON. Inline JSON (per `IsInlineJson`) is left untouched;
 * anything else is treated as a path and made relative. Use this instead of @ref MakePathRelative
 * for `params`.
 */
MOCHI_API void MakeParamsPathRelative(
    DynamicString& params,
    std::filesystem::path const& basePath,
    superdex::Error& error);

/**
 * @brief @ref MakePathAbsolute variant for a controller/sensor `params` field, which may hold
 * either a file path or inline JSON. Inline JSON (per `IsInlineJson`) is left untouched;
 * anything else is treated as a path and made absolute. Use this instead of @ref MakePathAbsolute
 * for `params`.
 */
MOCHI_API void MakeParamsPathAbsolute(
    DynamicString& params,
    std::filesystem::path const& basePath,
    superdex::Error& error);

/**
 * @brief Convert all file paths in a @ref BotPrefab to be relative to a base directory.
 *
 * Each path-bearing field is run through @ref UnresolveBotPath, so the
 * serialized form mirrors the original (`//`, `@tag/`, or file-relative)
 * deterministically based on the `.superdex_root` topology.
 *
 * @param[in,out] botPrefab The bot parameters to modify.
 * @param[in] basePath The base directory (typically the parent directory of the save file).
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @note Affects renderModelFile, shapeFile, and each sensor's params path in each link.
 */
MOCHI_API void MakePathsRelative(
    BotPrefab& botPrefab,
    std::filesystem::path const& basePath,
    superdex::Error& error);

/**
 * @brief Convert all relative file paths in a @ref BotPrefab to absolute, resolving against a base
 * directory.
 *
 * @param[in,out] botPrefab The bot parameters to modify.
 * @param[in] basePath The base directory (typically the parent directory of the loaded file).
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @note Affects renderModelFile, shapeFile, and each sensor's params path in each link.
 */
MOCHI_API void MakePathsAbsolute(
    BotPrefab& botPrefab,
    std::filesystem::path const& basePath,
    superdex::Error& error);

/**
 * @brief LEGACY ONLY: migrate a freshly-deserialized @ref BotPrefab's sensors from the old
 * `typeName`/`paramsFile` keys (captured by BotSensorPrefab::_legacyTypeName/_legacyParamsFile) to
 * the current `type`/`params` fields, then clear the legacy fields so a re-save emits only the new
 * keys. Idempotent. Call right after deserialization and before path resolution. Remove together
 * with the legacy fields on @ref BotSensorPrefab.
 */
MOCHI_API void ApplyLegacyBotSensorFields(BotPrefab& botPrefab);

/**
 * @brief LEGACY ONLY: @ref ModBotPrefab overload of @ref ApplyLegacyBotSensorFields. Folds the
 * legacy `typeName`/`paramsFile` sensor keys on the inline links carried by @ref AttachLink /
 * @ref ReplaceLink modifications (@ref AttachBot / @ref ReplaceLinkWithBot reference external
 * .superdex_bot files, whose sensors are folded when those files load). Call right after
 * deserialization and before path resolution. Idempotent.
 */
MOCHI_API void ApplyLegacyBotSensorFields(ModBotPrefab& modBotPrefab);

/**
 * @brief LEGACY ONLY: default any joint whose type is @ref ArticulatedJointType::Invalid to
 * @ref ArticulatedJointType::Hard. BotJointPrefab::type used to default to Hard but now defaults to
 * Invalid, so joints in files authored before the change deserialize as Invalid. Idempotent. Call
 * right after deserialization. Remove once all bots have been re-saved.
 */
MOCHI_API void ApplyLegacyBotJointTypes(BotPrefab& botPrefab);

/**
 * @brief LEGACY ONLY: @ref ModBotPrefab overload of @ref ApplyLegacyBotJointTypes. Defaults the
 * inline connecting joints carried by @ref AttachLink / @ref AttachBot modifications (@ref
 * ReplaceLink / @ref ReplaceLinkWithBot carry no joint). Idempotent.
 */
MOCHI_API void ApplyLegacyBotJointTypes(ModBotPrefab& modBotPrefab);

/**
 * @brief Convert all file paths in a @ref ModBotPrefab to be relative.
 *
 * Every path-bearing field — @ref ModBotPrefab::base, the @c path of any
 * @ref AttachBot or @ref ReplaceLinkWithBot, and the embedded link asset paths
 * in @ref AttachLink and @ref ReplaceLink — is run through @ref UnresolveBotPath.
 * The resulting prefix form (file-relative, `//`, or `@tag/`) is chosen by
 * @ref UnresolveBotPath's priority rules and mirrors how the path was
 * originally written when round-tripped from a load.
 *
 * @param[in,out] modBotPrefab The mod bot prefab to modify.
 * @param[in] basePath The base directory (typically the parent directory of the save file).
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void MakePathsRelative(
    ModBotPrefab& modBotPrefab,
    std::filesystem::path const& basePath,
    superdex::Error& error);

/**
 * @brief Convert all relative file paths in a @ref ModBotPrefab to absolute, resolving against
 * a base directory.
 *
 * @param[in,out] modBotPrefab The mod bot prefab to modify.
 * @param[in] basePath The base directory (typically the parent directory of the loaded file).
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @note Affects the base path and all paths within modifications.
 */
MOCHI_API void MakePathsAbsolute(
    ModBotPrefab& modBotPrefab,
    std::filesystem::path const& basePath,
    superdex::Error& error);

// ---------------------------------------------------------------------------
// BotPrefab
// ---------------------------------------------------------------------------

/**
 * @brief Save bot parameters to a .superdex_bot JSON file.
 *
 * @param botPrefab The bot parameters to save.
 * @param path The file path to save to (should end with .superdex_bot).
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void
SaveToFile(BotPrefab const& botPrefab, std::string_view path, superdex::Error& error);

/**
 * @brief Serialize @ref BotPrefab to a JSON string.
 *
 * @param botPrefab The bot parameters to serialize.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return JSON string representation.
 */
[[nodiscard]] MOCHI_API DynamicString
WriteToJsonString(BotPrefab const& botPrefab, superdex::Error& error);

/**
 * @brief Serialize @ref ModBotPrefab to a JSON string.
 *
 * @param modBotPrefab The mod bot prefab to serialize.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return JSON string representation.
 */
[[nodiscard]] MOCHI_API DynamicString
WriteToJsonString(ModBotPrefab const& modBotPrefab, superdex::Error& error);

/**
 * @brief Deserialize @ref BotPrefab from a JSON string.
 *
 * @param json The JSON string to deserialize.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return Deserialized @ref BotPrefab, or default-constructed on failure.
 *
 * @note Calls @ref RebuildBotData on the result before returning.
 */
[[nodiscard]] MOCHI_API BotPrefab ReadFromJsonString(std::string_view json, superdex::Error& error);

/**
 * @brief Save @ref ModBotPrefab to a .superdex_bot JSON file.
 *
 * @param modBotPrefab The mod bot prefab to save.
 * @param path The file path to save to (should end with .superdex_bot).
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void
SaveToFile(ModBotPrefab const& modBotPrefab, std::string_view path, superdex::Error& error);

// ---------------------------------------------------------------------------
// BotScenePrefab
// ---------------------------------------------------------------------------

#if MOCHI_INTERNAL
/* @brief Save bot scene prefab to a .mochi_bot_scene JSON file.
 *
 * Creates parent directories if they do not exist.
 *
 * @param[in] prefab The bot scene parameters to save.
 * @param[in] path File path (should end with .mochi_bot_scene).
 * @param[in,out] error Error status. */
MOCHI_API void
SaveToFile(BotScenePrefab const& scenePrefab, std::string_view path, superdex::Error& error);
#endif // MOCHI_INTERNAL

// ---------------------------------------------------------------------------
// URDF
// ---------------------------------------------------------------------------

/**
 * @brief Verbatim URDF mesh references for one bot, parallel to @ref BotPrefab::links.
 *
 * Each link slot records the raw `<mesh filename="...">` string exactly as written in
 * the URDF (e.g. an unresolvable @c package:// URI), independent of whether it resolved
 * to a file on disk. A slot's string is empty when that link has no mesh in that role.
 * Unlike the resolved path fields (`renderModelFile` / `shapeFile`, which
 * are empty when resolution fails), this preserves the original reference so callers can
 * distinguish "referenced but missing" from "absent" and show the original text.
 */
struct UrdfMeshReferences {
  struct LinkRefs {
    DynamicString visual; ///< Raw visual mesh reference, or empty if the link has none.
    DynamicString collision; ///< Raw collision mesh reference, or empty if the link has none.
  };
  DynamicArray<LinkRefs> links; ///< Indexed identically to @ref BotPrefab::links.
};

/**
 * @brief Load bot parameters from a URDF file.
 *
 * @details Parses the URDF and resolves each link's visual and collision mesh references
 * relative to the URDF location (ROS @c package:// URIs are supported), storing the resolved
 * on-disk paths in the returned prefab. Collision meshes may be any format the shape loader
 * accepts (e.g. .stl, .obj, .ply, .off, .mochi.h5); when a collision mesh carries no baked
 * SDF, one is generated on demand at bot creation. A free @c world_joint is injected at index 0.
 *
 * @note Only `<mesh>` geometry is imported. Primitive URDF shapes (`<box>`, `<cylinder>`,
 * `<sphere>`) in `<visual>` / `<collision>` elements are not supported and are silently
 * skipped; the importer uses the first mesh-bearing visual/collision on each link.
 *
 * @param[in] path Path to the .urdf file.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return The loaded or built @ref BotPrefab.
 */
[[nodiscard]] MOCHI_API BotPrefab
LoadBotPrefabFromUrdfFile(std::string_view path, superdex::Error& error);

/**
 * @brief Load a @ref BotPrefab and collect the verbatim per-link URDF mesh references.
 *
 * Same as @ref LoadBotPrefabFromUrdfFile but additionally fills @p meshRefs with the raw
 * `<mesh filename="...">` strings (see @ref UrdfMeshReferences), sized and indexed to
 * match @ref BotPrefab::links. Resolved path fields keep their path-only semantics (empty
 * when the mesh cannot be found on disk).
 *
 * @param[in] path Path to the .urdf file.
 * @param[out] meshRefs Receives the raw per-link mesh references.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return The loaded or built @ref BotPrefab.
 */
[[nodiscard]] MOCHI_API BotPrefab LoadBotPrefabFromUrdfFile(
    std::string_view path,
    UrdfMeshReferences& meshRefs,
    superdex::Error& error);

/**
 * @brief Load bot parameters from a URDF XML string (no file I/O).
 *
 * Same as @ref LoadBotPrefabFromUrdfFile but parses from an in-memory string instead of a file.
 * Mesh paths in the URDF are ignored since there is no base directory to resolve them against.
 *
 * @param[in] xmlString The URDF XML content.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return The loaded @ref BotPrefab.
 */
[[nodiscard]] MOCHI_API BotPrefab
LoadBotPrefabFromUrdfString(std::string_view xmlString, superdex::Error& error);

/**
 * @brief Export a @ref BotPrefab to URDF XML format and write to file.
 *
 * The inverse of @ref LoadBotPrefabFromUrdfFile. Only data that originated from a URDF import is
 * guaranteed to round-trip losslessly. The injected world_joint (index 0) is stripped during
 * export. Does not convert glb/h5 meshes to dae/STL meshes
 *
 * @param[in] botPrefab The bot parameters to export.
 * @param[in] path Output file path (should end with .urdf).
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void
SaveToUrdfFile(BotPrefab const& botPrefab, std::string_view path, superdex::Error& error);

/**
 * @brief Export a @ref BotPrefab to a URDF XML string (no file I/O).
 *
 * Same as @ref SaveToUrdfFile but returns the XML as a string instead of writing to a file.
 * Mesh elements are omitted since there is no output directory to resolve paths against.
 *
 * @param[in] botPrefab The bot parameters to export.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return The URDF XML string.
 */
[[nodiscard]] MOCHI_API DynamicString
SaveToUrdfString(BotPrefab const& botPrefab, superdex::Error& error);

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

/**
 * @brief Compute a composite xxHash (XXH3, 64-bit) of a `.superdex_bot` or
 * `.superdex_bot_archive` file's effective content and all referenced model assets.
 *
 * Hashes two layers of data into a single streaming hash:
 * 1. Serialized JSON of the resolved @ref BotPrefab (for mod bots, this is the fully
 *    flattened result after @ref BuildBot — see implementation comment for trade-offs).
 * 2. Each link's @c renderModelFile (`.glb`) and @c shapeFile (`.mochi.h5`) files, if present.
 *
 * Internally calls @ref LoadBotPrefabFromFile to resolve the bot parameters, so the file must
 * contain valid @ref BotPrefab or @ref ModBotPrefab JSON.
 *
 * @param[in] path Absolute or relative path to the `.superdex_bot` file.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @return 16-character lowercase hex string.
 */
[[nodiscard]] MOCHI_API DynamicString HashBotFile(std::string_view path, superdex::Error& error);

/**
 * @brief Compute an xxHash (XXH3, 64-bit) of a single file's raw contents.
 *
 * Reads the entire file into memory and hashes it as a single block. Do not
 * use this for hashing .superdex_bot files; use @ref HashBotFile instead which
 * recursively hashes all referenced model assets.
 *
 * @param[in] path Absolute or relative path to the file.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @return 16-character lowercase hex string.
 */
[[nodiscard]] MOCHI_API DynamicString
HashGenericFile(std::string_view path, superdex::Error& error);

// ---------------------------------------------------------------------------
// Generic SR Structs
// ---------------------------------------------------------------------------

#if MOCHI_USE_REFLECTION

/**
 * @brief Load an SReflect-annotated struct from a JSON file.
 *
 * By default (strict mode), any issue reported by the deserializer —
 * unrecognized keys, missing keys — is treated as an error. Pass
 * @p allowPartial = true to permit partial loads where missing fields
 * retain their C++ default values.
 *
 * @tparam T The struct type (must have MOCHI_STRUCT_BEGIN/END macros).
 * @param[in] path Path to the JSON file.
 * @param[in,out] error Error status.
 * @param[in] allowPartial If true, permit partial loads (missing fields keep defaults).
 * @return Loaded struct, or default-constructed on failure.
 */
template <typename T>
T LoadParamsFromFile(std::string_view path, superdex::Error& error, bool allowPartial = false) {
  MOCHI_ERROR_RETURN(error, {});
  T params;
  int numIssues = 0;
  std::string pathStr(path);
  bool const success = SReflect::LoadFromJsonFile(
      params, pathStr.c_str(), SReflect::DeserializeFlags::Default, numIssues);
  MOCHI_ERROR_IF(!success, error, "Failed to load params from file");
  MOCHI_ERROR_RETURN(error, {});
  if (!allowPartial) {
    MOCHI_ERROR_IF(numIssues > 0, error, "Params file has issues (unrecognized or missing fields)");
    MOCHI_ERROR_RETURN(error, {});
  }
  return params;
}

/**
 * @brief True if @p value looks like inline JSON -- its first non-whitespace character is '{' or
 * '[' -- rather than a path to a JSON file. Empty / all-whitespace returns false.
 */
[[nodiscard]] MOCHI_API bool IsInlineJson(std::string_view value);

/**
 * @brief Load an SReflect-annotated struct from a value that is EITHER a path to a JSON file OR an
 * inline JSON string. Inline JSON is detected via @ref IsInlineJson; any other non-empty value is
 * treated as a file path (a .json/.superdex_* file). An empty value yields a default-constructed
 * struct. Same strict / @p allowPartial semantics as @ref LoadParamsFromFile.
 *
 * Sensors, controllers, and (future) actuators use this so a scene/bot entry's params or initArgs
 * may be given either as a path or inline JSON. When @p outBaseDir is non-null it receives the
 * directory against which relative paths the struct itself references (e.g. a model/weights file)
 * should be resolved: the params file's parent directory in the path case, or empty in the inline
 * case (inline JSON has no base directory, so any paths it references must be absolute).
 *
 * @tparam T The struct type (must have MOCHI_STRUCT_BEGIN/END macros).
 * @param[in] pathOrJson A file path, an inline JSON string, or empty.
 * @param[in,out] error Error status.
 * @param[in] allowPartial If true, permit partial loads (missing fields keep defaults).
 * @param[out] outBaseDir Optional base directory for resolving relative sub-paths (see above).
 * @return Loaded struct, or default-constructed on failure / empty input.
 */
template <typename T>
T LoadParamsFromPathOrJson(
    std::string_view pathOrJson,
    superdex::Error& error,
    bool allowPartial = false,
    std::string* outBaseDir = nullptr) {
  MOCHI_ERROR_RETURN(error, {});
  if (outBaseDir != nullptr) {
    outBaseDir->clear();
  }
  T params;
  if (pathOrJson.empty()) {
    return params;
  }
  int numIssues = 0;
  if (IsInlineJson(pathOrJson)) {
    // The numIssues overload of FromJsonString returns void; a malformed JSON string surfaces as
    // numIssues > 0 (caught by the strict check below unless allowPartial).
    SReflect::FromJsonString(
        params, std::string(pathOrJson), SReflect::DeserializeFlags::Default, numIssues);
  } else {
    std::string const pathStr(pathOrJson);
    bool const success = SReflect::LoadFromJsonFile(
        params, pathStr.c_str(), SReflect::DeserializeFlags::Default, numIssues);
    MOCHI_ERROR_IF(!success, error, "Failed to load params from file");
    MOCHI_ERROR_RETURN(error, {});
    if (outBaseDir != nullptr) {
      *outBaseDir = std::filesystem::path(pathStr).parent_path().string();
    }
  }
  if (!allowPartial) {
    MOCHI_ERROR_IF(numIssues > 0, error, "Params have issues (unrecognized or missing fields)");
    MOCHI_ERROR_RETURN(error, {});
  }
  return params;
}

/**
 * @brief Save an SReflect-annotated struct to a JSON file.
 * @tparam T The struct type (must have MOCHI_STRUCT_BEGIN/END macros).
 * @param[in] params The struct to serialize.
 * @param[in] path Destination file path.
 * @param[in,out] error Error status.
 */
template <typename T>
void SaveParamsToFile(T const& params, std::string_view path, superdex::Error& error) {
  MOCHI_ERROR_RETURN(error);
  std::string pathStr(path);
  bool const success = SReflect::SaveToJsonFile(params, pathStr.c_str());
  MOCHI_ERROR_IF(!success, error, "Failed to save params to file");
}

#endif // MOCHI_USE_REFLECTION

} // namespace superdex::robotics
