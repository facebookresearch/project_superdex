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
#include <superdex_robotics/utils/file_utils.h>

#include <filesystem>

namespace superdex::robotics {

constexpr std::string_view kArchiveMetadataFile = ".mochi_bot_archive_metadata";
/// Canonical bot archive extension, emitted by all write/creation paths.
constexpr std::string_view kBotArchiveExtension = ".superdex_bot_archive";

/**
 * @brief Recognize a bot archive file path by extension.
 *
 * Always accepts the canonical @ref kBotArchiveExtension (`.superdex_bot_archive`).
 * In internal builds (@c MOCHI_INTERNAL) it additionally accepts the legacy extensions for backward
 * and Meta internal compatibility; public/OSS builds recognize only the canonical extension.
 *
 * @param[in] path The file path to test.
 * @return @c true if @p path ends with a recognized bot archive extension.
 */
[[nodiscard]] MOCHI_API bool IsBotArchivePath(std::string_view path);

/**
 * @brief Parameters for @ref ArchiveBot.
 */
struct ArchiveParams {
  /// Path to the source `.superdex_bot` file.
  DynamicString src;

  /// Path to the destination `.superdex_bot_archive` file. Parent directories are created if they
  /// do not exist.
  DynamicString dst;

  /// Optional comment to include in the archive root metadata. Metadata already includes the
  /// date, source content hash, and the commit hash from which the archive was produced.
  std::optional<DynamicString> comment;
};

/**
 * @brief Metadata stored at the root of every `.superdex_bot_archive` file.
 *
 * Serialized as JSON to a `.mochi_bot_archive_metadata` entry inside the archive zip.
 */
struct BotArchiveMetadata {
  /// The date the archive was created (UTC, YYYY-MM-DD).
  DynamicString date;
  /// The hash of the source `.superdex_bot` file at the time of archiving (matches @ref
  /// HashBotFile of the source).
  DynamicString botHash;
  /// The commit hash of the source code at the time of archiving.
  DynamicString commitHash;
  /// Archive-root-relative path of the `.superdex_bot` file that should be loaded from the
  /// archive. Joined with the extracted directory by @ref GetExtractedBotArchiveTarget.
  DynamicString target;
  /// Optional comment provided by the user when the archive was created.
  std::optional<DynamicString> comment;
  /// Every asset the source tree could not supply when the archive was written: a file the bot
  /// names that was missing, or a sensor params file that could not be parsed. An archive mirrors
  /// the tree it came from, gaps included, so these record what was already absent rather than
  /// anything archiving did wrong. Empty for an archive built from a complete tree.
  DynamicArray<DynamicString> warnings;

  MOCHI_STRUCT_BEGIN(superdex::robotics::BotArchiveMetadata)
  MOCHI_FIELD(date)
  MOCHI_FIELD(botHash)
  MOCHI_FIELD(commitHash)
  MOCHI_FIELD(target)
  MOCHI_FIELD(comment)
  MOCHI_FIELD(warnings)
  MOCHI_STRUCT_END()
};

/**
 * @brief Archive a `.superdex_bot` file and all its transitive dependencies into a self-contained
 * `.superdex_bot_archive` zip file.
 *
 * Recursively collects every referenced file — other `.superdex_bot` files
 * (@ref ModBotPrefab::base, @ref AttachBot::path, @ref ReplaceLinkWithBot::path) and
 * model assets (`shapeFile`, `renderModelFile`) — then
 * compresses them into a single zip file at the destination.
 *
 * The archive mirrors the on-disk multi-root topology rooted at the deepest common ancestor
 * of all involved `.superdex_root` directories: each source root is laid out at its
 * relative offset under the archive top, and one (possibly rewritten) `.superdex_root`
 * file is included per source root. Tags whose target is not in the archive are dropped;
 * surviving tags are rewritten to the relative path between the archived roots so cross-root
 * `@tag` references continue to resolve.
 *
 * Errors out if the involved `.superdex_root` directories share no common ancestor (e.g.
 * different Windows drive letters). The destination path must end with `.superdex_bot_archive`.
 *
 * @param[in] params Archive parameters (source bot, destination `.superdex_bot_archive` path).
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @see LoadBotPrefabFromFile, which can transparently load a bot from a `.superdex_bot_archive`.
 */
MOCHI_API void ArchiveBot(ArchiveParams const& params, superdex::Error& error);

/**
 * @brief Extract a `.superdex_bot_archive` zip file into a deterministic location under the OS
 * temp directory and return the resulting path.
 *
 * The destination directory is keyed on the archive's content hash (via @ref HashGenericFile),
 * so identical archives share an extracted directory across calls and processes regardless of
 * path or modification time, while modified archives are re-extracted into a fresh directory.
 *
 * @param[in] archiveFile Path to a `.superdex_bot_archive` zip file.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @return Path to the extracted directory, or an empty path on failure.
 */
[[nodiscard]] MOCHI_API DynamicString
ExtractBotArchiveToCache(std::string_view archiveFile, superdex::Error& error);

/**
 * @brief Resolve the absolute path of the target `.superdex_bot` file inside an extracted archive.
 *
 * Typically called immediately after @ref ExtractBotArchiveToCache to determine which
 * `.superdex_bot` file inside the extracted directory should be handed to a @ref FileBotLoader.
 *
 * @param[in] extractedDir Directory produced by @ref ExtractBotArchiveToCache (or any directory
 *     containing a `.mochi_bot_archive_metadata` file at its root).
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @return Absolute path to the target `.superdex_bot` file, or an empty path on failure.
 */
[[nodiscard]] MOCHI_API DynamicString
GetExtractedBotArchiveTarget(std::string_view extractedDir, superdex::Error& error);

/**
 * @brief Read the metadata written at the root of an extracted bot archive.
 *
 * @details Companion to @ref GetExtractedBotArchiveTarget, which reads the same file for its target
 * alone. Use this to inspect @ref BotArchiveMetadata::warnings and learn which assets the source
 * tree could not supply, without having to load the bot first.
 *
 * @param[in] extractedDir Directory produced by @ref ExtractBotArchiveToCache.
 * @param[out] error Set if the metadata file is missing or cannot be parsed.
 *
 * @return The archive's metadata; a default-constructed value on error.
 */
[[nodiscard]] MOCHI_API BotArchiveMetadata
ReadBotArchiveMetadata(std::string_view extractedDir, superdex::Error& error);

// ---------------------------------------------------------------------------
// Bot scene archives (.mochi_bot_scene_archive)
// ---------------------------------------------------------------------------

#if MOCHI_INTERNAL
constexpr std::string_view kSceneArchiveMetadataFile = ".mochi_bot_scene_archive_metadata";
constexpr std::string_view kSceneArchiveExtension = ".mochi_bot_scene_archive";

/* @brief Metadata stored at the root of every .mochi_bot_scene_archive file. */
struct BotSceneArchiveMetadata {
  DynamicString date;
  DynamicString sceneHash;
  DynamicString commitHash;
  DynamicString target;
  std::optional<DynamicString> comment;

  MOCHI_STRUCT_BEGIN(superdex::robotics::BotSceneArchiveMetadata)
  MOCHI_FIELD(date)
  MOCHI_FIELD(sceneHash)
  MOCHI_FIELD(commitHash)
  MOCHI_FIELD(target)
  MOCHI_FIELD(comment)
  MOCHI_STRUCT_END()
};

/* @brief Archive a .mochi_bot_scene file and all its transitive dependencies into a
 * self-contained .mochi_bot_scene_archive zip file.
 *
 * Collects all transitive dependencies — the base scene and its nested prefabs/shapes,
 * spawnable prefabs, bot archives, and controller param files — then mirrors the on-disk
 * multi-root topology (identical to the strategy used by @ref ArchiveBot) so that path
 * resolution works identically after extraction.
 *
 * Requirements:
 *  - Every bots[].path must reference a .superdex_bot_archive (raw .superdex_bot is rejected).
 *  - params.src must end with .mochi_bot_scene.
 *  - params.dst must end with .mochi_bot_scene_archive.
 *
 * @param[in] params Archive parameters (source `.mochi_bot_scene`, destination
 *     `.mochi_bot_scene_archive` path).
 * @param[in,out] error Error status. Check @ref Error::IsOK for success. */
MOCHI_API void ArchiveBotScene(ArchiveParams const& params, superdex::Error& error);

/* @brief Extract a .mochi_bot_scene_archive to a content-hash-keyed cache directory.
 *
 * @param[in] archiveFile Path to the .mochi_bot_scene_archive file.
 * @param[in,out] error Error status.
 * @return Path to the extracted directory, or empty on failure. */
[[nodiscard]] MOCHI_API DynamicString
ExtractBotSceneArchiveToCache(std::string_view archiveFile, superdex::Error& error);

/* @brief Get the target .mochi_bot_scene path inside an extracted scene archive.
 *
 * @param[in] extractedDir Directory produced by ExtractBotSceneArchiveToCache.
 * @param[in,out] error Error status.
 * @return Absolute path to the target .mochi_bot_scene file, or empty on failure. */
[[nodiscard]] MOCHI_API DynamicString
GetExtractedBotSceneArchiveTarget(std::string_view extractedDir, superdex::Error& error);
#endif // MOCHI_INTERNAL

} // namespace superdex::robotics
