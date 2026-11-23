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

#include <mochi_core/utils/json_utils.h>

#include <charconv>
#include <cmath>
#include <string>
#include <system_error>
#include <unordered_set>

using namespace mochi;

namespace superdex::robotics {
namespace {

picojson::value ParsePayload(std::string_view text, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  if (text.empty()) {
    return {picojson::object{}};
  }
  return ParseJsonFromString(text, error);
}

uint64_t ParseCanonicalSeed(std::string_view seed, Error& error) {
  MOCHI_ERROR_RETURN(error, 0);
  MOCHI_ERROR_IF(seed.size() > 1 && seed.front() == '0', error, "DREpisodeList: noncanonical seed");
  return ParseSeed(seed, error);
}

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
  MOCHI_ERROR_IF(list.schemaVersion != 2, error, "DREpisodeList: expected schema_version 2");
  MOCHI_ERROR_IF(list.specVersion.empty(), error, "DREpisodeList: empty spec_version");
  (void)ParseCanonicalSeed(list.masterSeed, error);
  auto const provenance = ParsePayload(list.provenance, error);
  MOCHI_ERROR_IF_NOT(
      provenance.is<picojson::object>(), error, "DREpisodeList: provenance must be an object");
  auto const metadata = ParsePayload(list.metadata, error);
  MOCHI_ERROR_IF_NOT(
      metadata.is<picojson::object>(), error, "DREpisodeList: metadata must be an object");
  MOCHI_ERROR_RETURN(error);
  for (auto const& [key, value] : provenance.get<picojson::object>()) {
    MOCHI_ERROR_IF_NOT(
        value.is<std::string>(), error, "DREpisodeList: provenance values must be strings");
  }
  MOCHI_ERROR_RETURN(error);
  std::unordered_set<int64_t> indices;
  for (auto const& episode : list.episodes) {
    (void)ParseCanonicalSeed(episode.episodeSeed, error);
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF(episode.index < 0, error, "DREpisodeList: negative episode index");
    MOCHI_ERROR_IF(
        !indices.insert(episode.index).second, error, "DREpisodeList: duplicate episode index");
    MOCHI_ERROR_RETURN(error);
    std::unordered_set<std::string> keys;
    for (auto const& entry : episode.values) {
      MOCHI_ERROR_IF(entry.key.empty(), error, "DREpisode: value entry has empty key");
      MOCHI_ERROR_IF(entry.value.empty(), error, "DREpisode: value entry has no elements");
      MOCHI_ERROR_IF(
          entry.scalar && entry.value.size() != 1, error, "DREpisode: scalar length must be one");
      MOCHI_ERROR_IF(
          !keys.insert(std::string(entry.key)).second, error, "DREpisode: duplicate value key");
      MOCHI_ERROR_RETURN(error);
      for (auto const value : entry.value) {
        MOCHI_ERROR_IF(!std::isfinite(value), error, "DREpisode: nonfinite value");
      }
      MOCHI_ERROR_RETURN(error);
    }
    if (episode.appliedKeys) {
      std::unordered_set<std::string> applied;
      for (auto const& key : *episode.appliedKeys) {
        MOCHI_ERROR_IF(
            keys.count(std::string(key)) == 0, error, "DREpisode: applied key has no value");
        MOCHI_ERROR_IF(
            !applied.insert(std::string(key)).second, error, "DREpisode: duplicate applied key");
        MOCHI_ERROR_RETURN(error);
      }
    }
  }
}

DREpisodeList LoadDREpisodeListFromFile(std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  auto list = LoadParamsFromFile<DREpisodeList>(path, error);
  MOCHI_ERROR_RETURN(error, {});
  ValidateDREpisodeList(list, error);
  MOCHI_ERROR_RETURN(error, {});
  return list;
}

void SaveDREpisodeListToFile(DREpisodeList const& list, std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error);
  ValidateDREpisodeList(list, error);
  MOCHI_ERROR_RETURN(error);
  SaveParamsToFile(list, path, error);
}

} // namespace superdex::robotics
