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
#include <optional>
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
 * generic `{key -> value}` path. `episodeSeed` is a decimal string to preserve
 * the full uint64 range across JSON decoders; parse it with @ref ParseSeed.
 * `index` is a nonnegative signed-int64 episode identity, encoded as a JSON integer.
 * It is independent of the full-width uint64 RNG seed. */
struct DREpisode {
  int64_t index = -1;
  DynamicString episodeSeed;
  DynamicArray<DREpisodeEntry> values;
  /* Absent means application coverage was not recorded. An engaged empty list
   * records that no keys were applied; nonempty lists name applied value keys. */
  std::optional<DynamicArray<DynamicString>> appliedKeys;

  MOCHI_STRUCT_BEGIN(superdex::robotics::DREpisode)
  MOCHI_FIELD(index)
  MOCHI_FIELD_NAME(episodeSeed, "episode_seed")
  MOCHI_FIELD(values)
  MOCHI_FIELD_NAME(appliedKeys, "applied_keys") MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/* @brief A pre-randomized DR episode list — the neutral cross-language interchange.
 *
 * A C++ applier consumes resolved per-episode values rather than the experiment
 * configuration that produced them. The public file helpers support schema 2,
 * preserving producer-owned JSON without interpreting its experiment semantics. */
struct DREpisodeList {
  int32_t schemaVersion = 0;
  DynamicString specVersion;
  DynamicString masterSeed;
  DynamicArray<DREpisode> episodes;
  /* String-valued producer provenance as JSON object text; empty means unspecified.
   * Payload meaning belongs to the producer. See adjacent README.md#producer-payloads. */
  DynamicString provenance;
  /* Producer-owned JSON object text, emitted under metadata.
   * Payload meaning belongs to the producer. See adjacent README.md#producer-payloads. */
  DynamicString metadata;

  MOCHI_STRUCT_BEGIN(superdex::robotics::DREpisodeList)
  MOCHI_FIELD_NAME(schemaVersion, "schema_version")
  MOCHI_FIELD_NAME(specVersion, "spec_version")
  MOCHI_FIELD_NAME(masterSeed, "master_seed")
  MOCHI_FIELD(episodes)
  MOCHI_FIELD(provenance) MOCHI_ATTRIBUTE(JsonString) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(metadata) MOCHI_ATTRIBUTE(JsonString) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/* @brief Parse a decimal uint64 seed string (`episode_seed` / `master_seed`).
 *
 * This parser accepts leading zeroes. Schema-2 validation requires canonical seeds.
 *
 * @param[in] seed Decimal seed string.
 * @param[in,out] error Set if `seed` is empty or not a valid uint64.
 * @return The parsed seed, or 0 on failure. */
[[nodiscard]] MOCHI_API uint64_t ParseSeed(std::string_view seed, superdex::Error& error);

/* @brief Load a pre-randomized DR episode list from a JSON manifest file.
 *
 * Uses standard reflection and @ref ValidateDREpisodeList. Episode indices are
 * nonnegative int64 JSON integers; seeds are canonical decimal uint64 strings.
 * Schema 1 requires an explicit internal importer.
 *
 * @param[in] path File path to the manifest.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return Loaded @ref DREpisodeList, or default-constructed on failure. */
MOCHI_API DREpisodeList LoadDREpisodeListFromFile(std::string_view path, superdex::Error& error);

/* @brief Save a @ref DREpisodeList to a JSON manifest file.
 *
 * Validates via @ref ValidateDREpisodeList before opening the destination, then
 * uses standard reflection. Preserves native-real values, producer JSON and
 * absent-versus-empty application coverage.
 *
 * @param[in] list The episode list to save.
 * @param[in] path Destination file path.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success. */
MOCHI_API void
SaveDREpisodeListToFile(DREpisodeList const& list, std::string_view path, superdex::Error& error);

/* @brief Validate a @ref DREpisodeList's schema, seeds and keyed values.
 *
 * Checks schema-specific index/seed ranges, unique episode indices,
 * finite values, scalar/vector shape, application keys and JSON object metadata.
 *
 * @param[in] list The episode list to validate.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success. */
MOCHI_API void ValidateDREpisodeList(DREpisodeList const& list, superdex::Error& error);

} // namespace superdex::robotics
