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

#include <superdex_robotics/superdex_robotics.h>

#include <cstdint>
#include <string_view>

namespace superdex::robotics {

/* @brief File extension of a snapshot file (see @ref AssetSnapshot). */
constexpr std::string_view kAssetSnapshotExtension = ".superdex_scene_snapshot";

/* @brief The `formatVersion` @ref SaveAssetSnapshot writes and the only one @ref LoadAssetSnapshot
 * accepts. Bump it whenever a change to the format would make an older reader apply a snapshot
 * differently, so older readers reject the file instead. */
constexpr int64_t kAssetSnapshotFormatVersion = 1;

/* @brief One file a snapshot pins: the bot path a scene's references resolve to, and the bot path
 * of the version to load in its place. Both are `//`- or `@tag/`-prefixed, relative to the
 * scene's .superdex_root. */
struct AssetSnapshotEntry {
  DynamicString file;
  DynamicString version;
};

/* @brief The contents of a .superdex_scene_snapshot file: every file a scene uses, each mapped to
 * a pinned version. Where the versions come from is up to whoever writes the snapshot. */
struct AssetSnapshot {
  /* The scene the snapshot was taken of. */
  DynamicString scene;
  /* When the snapshot was taken [Unix seconds]. */
  uint64_t createdAt = 0;
  /* The source revision the versions were read at. */
  DynamicString sourceRevision;
  DynamicArray<AssetSnapshotEntry> files;
};

/* @brief Read a .superdex_scene_snapshot file. Fails unless its `formatVersion` is
 * @ref kAssetSnapshotFormatVersion.
 *
 * @param[in] path The snapshot file.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return The snapshot, or a default-constructed one on failure. */
[[nodiscard]] MOCHI_API AssetSnapshot
LoadAssetSnapshot(std::string_view path, superdex::Error& error);

/* @brief Write @p snapshot to @p path as JSON, with `formatVersion`
 * @ref kAssetSnapshotFormatVersion.
 *
 * @param[in] snapshot The snapshot to write.
 * @param[in] path Destination file.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success. */
MOCHI_API void
SaveAssetSnapshot(AssetSnapshot const& snapshot, std::string_view path, superdex::Error& error);

/* @brief Pin every file in @p snapshot for the rest of the process: install a path remap table
 * (see mochi::SetPathRemap) so each file loads from its version instead, replacing any table
 * already installed.
 *
 * Entries are resolved against @p assetRoot, the directory holding the scene's .superdex_root,
 * wherever its asset tree is mounted. Every version must exist, beside the file it pins, and no
 * version may itself be a file the snapshot pins. Then load the scene from the returned path as
 * usual: the scene and every file it references, however deeply, load from their pinned versions.
 *
 * The table is strict: a file under @p assetRoot, or under the .superdex_root directory of any
 * file the snapshot pins, that the snapshot does not pin fails to load instead of loading its tip.
 * Directories are not refused.
 *
 * The snapshot's paths resolve through the tip .superdex_root files, so a snapshot may pin
 * .superdex_root files too: applying it then fails if a pinned .superdex_root no longer matches its
 * version, since its `//` root and tag table may have moved.
 *
 * Loaded assets carry the versions' paths, so don't save assets while a snapshot is applied: the
 * pins would be written into them. Archiving is refused for the same reason. Call
 * @ref ClearAssetSnapshot to stop applying the snapshot.
 *
 * @param[in] snapshot The snapshot to apply.
 * @param[in] assetRoot Directory containing the .superdex_root the snapshot's paths are relative
 *     to.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success. On failure no table is
 *     installed, including any a previous snapshot installed.
 * @return Absolute path of the snapshot's scene, to load as usual. */
[[nodiscard]] MOCHI_API DynamicString ApplyAssetSnapshot(
    AssetSnapshot const& snapshot,
    std::string_view assetRoot,
    superdex::Error& error);

/* @brief Stop applying the snapshot @ref ApplyAssetSnapshot installed, so files load from their
 * tips again. Loads already done are unaffected. */
MOCHI_API void ClearAssetSnapshot();

} // namespace superdex::robotics
