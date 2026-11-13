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

#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/span.h>
#include <mochi_physics/utils/mochi_physics_macros.h>

#include <string_view>

namespace mochi {

/**
 * @brief One entry of a path remap table: a load whose path resolves to @ref from reads @ref to
 * instead.
 */
struct PathRemapEntry {
  /// Absolute path a reference resolves to.
  DynamicString from;
  /// Absolute path to load in its place.
  DynamicString to;
};

/**
 * @brief Install a process-wide path remap table, replacing any previous one.
 *
 * Every asset reference resolved after this call is passed through @ref RemapPath, so a file
 * whose resolved path matches an entry's @ref PathRemapEntry::from loads from its
 * @ref PathRemapEntry::to instead. Prefab references (@ref prefab::GetPrefabFullPath) honor the
 * table. Use it to pin a scene's files to specific versions without editing the files that
 * reference them.
 *
 * Paths are compared as absolute paths with symlinks resolved, so a relative or symlinked spelling
 * of a file matches its entry. Comparison is case-sensitive, even on file systems that are not.
 * Install the table before loading; loads already done are unaffected. A @ref PathRemapEntry::to
 * that is also some entry's @ref PathRemapEntry::from is remapped again wherever a resolved path is
 * passed through @ref RemapPath twice, so keep the two sets disjoint.
 *
 * @param[in] entries The table. Later entries win when two share a @ref PathRemapEntry::from.
 * @param[in] strictRoots Directories the table must cover. A file under one of them, as written or
 *     with its symlinks resolved, that is neither some entry's @ref PathRemapEntry::from nor some
 *     entry's @ref PathRemapEntry::to is refused: @ref RemapPath logs an error and returns an empty
 * string, so its load fails instead of reading a file the table does not pin. Directories are never
 * refused. Empty (the default) refuses nothing.
 */
MOCHI_API void SetPathRemap(
    Span<PathRemapEntry const> entries,
    Span<DynamicString const> strictRoots = {});

/** @brief Remove the process-wide path remap table installed by @ref SetPathRemap. */
MOCHI_API void ClearPathRemap();

/** @brief True while a path remap table with entries or strict roots is installed. */
[[nodiscard]] MOCHI_API bool IsPathRemapActive();

/**
 * @brief The path to load for a resolved absolute path: the matching entry's
 * @ref PathRemapEntry::to, @p path unchanged when the table has no entry for it, or an empty
 * string when @p path is refused (see the strictRoots of @ref SetPathRemap).
 */
[[nodiscard]] MOCHI_API DynamicString RemapPath(std::string_view path);

} // namespace mochi
