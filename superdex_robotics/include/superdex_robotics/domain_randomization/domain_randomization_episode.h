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

/* @brief One DR draw: a stable key and its sampled value as a uniform array.
 *
 * `value` is always a list so it maps to a DynamicArray<real>; a scalar draw
 * rides as a length-1 list flagged by `scalar`. The flag lets a reader recover
 * the exact scalar-vs-vector shape with no registry. */
struct DREpisodeEntry {
  DynamicString key;
  bool scalar = false;
  DynamicArray<real> value;

  MOCHI_STRUCT_BEGIN(superdex::robotics::DREpisodeEntry)
  MOCHI_FIELD(key)
  MOCHI_FIELD(scalar)
  MOCHI_FIELD(value)
  MOCHI_STRUCT_END()
};

/* @brief One episode's sampled DR: seed and the flat keyed values.
 *
 * Per-spawn pose DR is not a separate field — it rides in `values` under
 * `dr/spawn/<name>/<dr_class>/pose` keys, so a reader consumes it through the same
 * generic `{key -> value}` path. `episodeSeed` is a decimal string — a full uint64
 * can't round-trip as a JSON number, so parse it with @ref ParseSeed. */
struct DREpisode {
  int32_t index = 0;
  DynamicString episodeSeed;
  DynamicArray<DREpisodeEntry> values;

  MOCHI_STRUCT_BEGIN(superdex::robotics::DREpisode)
  MOCHI_FIELD(index)
  MOCHI_FIELD_NAME(episodeSeed, "episode_seed")
  MOCHI_FIELD(values)
  MOCHI_STRUCT_END()
};

/* @brief A pre-randomized DR episode list — the neutral cross-language interchange.
 *
 * Read from a `dr_manifest.json` produced by the Python DR pipeline
 * (schema v1). `profile` and `experiment_overrides` from the file are ignored: a
 * C++ applier consumes resolved per-episode values, not the config that produced
 * them, so @ref LoadDREpisodeListFromFile tolerates those unmodeled keys. */
struct DREpisodeList {
  int32_t schemaVersion = 0;
  DynamicString specVersion;
  DynamicString masterSeed;
  DynamicArray<DREpisode> episodes;

  MOCHI_STRUCT_BEGIN(superdex::robotics::DREpisodeList)
  MOCHI_FIELD_NAME(schemaVersion, "schema_version")
  MOCHI_FIELD_NAME(specVersion, "spec_version")
  MOCHI_FIELD_NAME(masterSeed, "master_seed")
  MOCHI_FIELD(episodes)
  MOCHI_STRUCT_END()
};

/* @brief Parse a decimal uint64 seed string (`episode_seed` / `master_seed`).
 *
 * @param[in] seed Decimal seed string.
 * @param[in,out] error Set if `seed` is empty or not a valid uint64.
 * @return The parsed seed, or 0 on failure. */
[[nodiscard]] MOCHI_API uint64_t ParseSeed(std::string_view seed, superdex::Error& error);

/* @brief Load a pre-randomized DR episode list from a JSON manifest file.
 *
 * Deserializes a schema-v1 `dr_manifest.json` and validates it via
 * @ref ValidateDREpisodeList. Unmodeled top-level keys
 * (`experiment_overrides`, `profile`) are ignored.
 *
 * @param[in] path File path to the manifest.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return Loaded @ref DREpisodeList, or default-constructed on failure. */
MOCHI_API DREpisodeList LoadDREpisodeListFromFile(std::string_view path, superdex::Error& error);

/* @brief Save a @ref DREpisodeList to a JSON manifest file.
 *
 * The inverse of @ref LoadDREpisodeListFromFile: validates @p list via
 * @ref ValidateDREpisodeList, then serializes it to a schema-v1
 * `dr_manifest.json`. Validating before writing makes a malformed list a hard
 * error rather than a bad file on disk. Only the modeled fields are emitted;
 * the optional producer keys the loader tolerates (`experiment_overrides`,
 * `profile`) are not part of @ref DREpisodeList, so they are absent.
 *
 * @param[in] list The episode list to save.
 * @param[in] path Destination file path.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success. */
MOCHI_API void
SaveDREpisodeListToFile(DREpisodeList const& list, std::string_view path, superdex::Error& error);

/* @brief Validate a @ref DREpisodeList: supported schema version, parseable seeds,
 * and self-consistent value entries (a `scalar` entry has a length-1 `value`).
 *
 * @param[in] list The episode list to validate.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success. */
MOCHI_API void ValidateDREpisodeList(DREpisodeList const& list, superdex::Error& error);

} // namespace superdex::robotics
