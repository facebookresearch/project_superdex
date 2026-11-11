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

#include <superdex_robotics/domain_randomization/domain_randomization_episode.h>
#include <superdex_robotics/utils/file_utils.h>

#include <charconv>
#include <string>
#include <system_error>
#include <unordered_set>

#if !MOCHI_USE_REFLECTION
#error Reflection is required for the mochi_bots library internals.
#endif

using namespace mochi;
using namespace superdex::robotics;

namespace superdex::robotics {
namespace {

// The only manifest schema version the C++ reader accepts.
constexpr int32_t kSchemaVersion = 1;

} // namespace

uint64_t ParseSeed(std::string_view seed, Error& error) {
  MOCHI_ERROR_RETURN(error, 0);
  MOCHI_ERROR_IF(seed.empty(), error, "DREpisodeList: empty seed string");
  MOCHI_ERROR_RETURN(error, 0);
  uint64_t value = 0;
  auto const result = std::from_chars(seed.data(), seed.data() + seed.size(), value);
  MOCHI_ERROR_IF(
      result.ec != std::errc() || result.ptr != seed.data() + seed.size(),
      error,
      "DREpisodeList: seed is not a valid uint64");
  MOCHI_ERROR_RETURN(error, 0);
  return value;
}

void ValidateDREpisodeList(DREpisodeList const& list, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      list.schemaVersion != kSchemaVersion,
      error,
      "DREpisodeList: unsupported schema_version (expected 1)");
  MOCHI_ERROR_RETURN(error);

  (void)ParseSeed(list.masterSeed, error); // master_seed must be a valid uint64
  MOCHI_ERROR_RETURN(error);

  for (auto const& episode : list.episodes) {
    (void)ParseSeed(episode.episodeSeed, error); // episode_seed must be a valid uint64
    MOCHI_ERROR_RETURN(error);
    std::unordered_set<std::string> keys;
    for (auto const& entry : episode.values) {
      MOCHI_ERROR_IF(entry.key.empty(), error, "DREpisode: value entry has empty key");
      MOCHI_ERROR_IF(entry.value.empty(), error, "DREpisode: value entry has no elements");
      MOCHI_ERROR_IF(
          entry.scalar && entry.value.size() != 1,
          error,
          "DREpisode: scalar value entry must have exactly one element");
      MOCHI_ERROR_RETURN(error);
      MOCHI_ERROR_IF(
          !keys.insert(std::string(entry.key)).second, error, "DREpisode: duplicate value key");
      MOCHI_ERROR_RETURN(error);
    }
  }
}

DREpisodeList LoadDREpisodeListFromFile(std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  // allowPartial: the manifest carries top-level fields a C++ applier doesn't
  // model (experiment_overrides, profile); tolerate them rather than fail.
  // Structural correctness is enforced by ValidateDREpisodeList.
  auto list = LoadParamsFromFile<DREpisodeList>(path, error, /*allowPartial=*/true);
  MOCHI_ERROR_RETURN(error, {});

  ValidateDREpisodeList(list, error);
  MOCHI_ERROR_RETURN(error, {});

  return list;
}

void SaveDREpisodeListToFile(DREpisodeList const& list, std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error);

  // Validate before writing so a malformed list is a hard error rather than a
  // bad manifest on disk — the write-side mirror of LoadDREpisodeListFromFile.
  ValidateDREpisodeList(list, error);
  MOCHI_ERROR_RETURN(error);

  SaveParamsToFile(list, path, error);
}

} // namespace superdex::robotics
