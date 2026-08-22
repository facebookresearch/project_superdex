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

#include <mochi_core/utils/error.h>
#include <mochi_core/utils/span.h>
#include <picojson/picojson.h>

#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace mochi {
/**
 * Parses a Json object from the given stream.
 */
template <typename StreamT>
[[nodiscard]] picojson::value ParseJsonFromStream(StreamT& instream, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  picojson::value jsonVal;
  std::string errorStr = picojson::parse(jsonVal, instream);
  if (!errorStr.empty()) {
    // The picojson error string might describe the specific problem. Write it to the log
    // in addition to returning an Error.
    MOCHI_LOG_WARNING("JSON parsing error: %s", errorStr.c_str());
    MOCHI_ERROR_SET(error, "Failed to parse JSON stream.");
    return {};
  }
  return jsonVal;
}

/**
 * Stores the given Json object in the given stream.
 */
template <typename StreamT>
void SerializeJsonToStream(StreamT& outstream, picojson::value const& jsonVal, Error& error) {
  MOCHI_ERROR_RETURN(error);
  outstream << jsonVal.serialize();
  if (!outstream.good()) {
    MOCHI_LOG_WARNING("JSON serialization error");
    MOCHI_ERROR_SET(error, "Failed to serialize JSON to stream.");
  }
}

/**
 * Parses a Json object from the given string.
 */
[[nodiscard]] inline picojson::value ParseJsonFromString(Span<char const> fileData, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  std::istringstream instream(std::string(fileData.data(), fileData.size()));
  return ParseJsonFromStream(instream, error);
}

/**
 * Parses a Json object from the given file.
 */
[[nodiscard]] inline picojson::value ParseJsonFromFile(std::string_view filename, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  std::ifstream infile(std::string{filename});
  if (!infile.is_open()) {
    MOCHI_ERROR_SET(error, "Unable to open file.");
    return {};
  }
  return ParseJsonFromStream(infile, error);
}

/**
 * Interprets/parses the given Json object as the given scalar type.
 */
template <class T>
[[nodiscard]] std::vector<T> ParseJsonScalarArray(picojson::value const& jsonValue, Error& error) {
  MOCHI_ERROR_IF_NOT(
      jsonValue.is<picojson::array>(), error, "Type mismatch. Expected a JSON array.");
  MOCHI_ERROR_RETURN(error, {});
  auto const& jsonArr = jsonValue.get<picojson::array>();
  std::vector<T> arr(jsonArr.size());
  for (size_t i = 0; i < arr.size(); ++i) {
    MOCHI_ERROR_IF_NOT(
        jsonArr[i].is<double>(), error, "Type mismatch. Expected a numeric value in JSON array.");
    MOCHI_ERROR_RETURN(error, {});
    arr[i] = static_cast<T>(jsonArr[i].get<double>());
  }
  return arr;
}

/**
 * Stores the given array as a Json object.
 */
template <class T>
[[nodiscard]] picojson::array SerializeJsonScalarArray(std::vector<T> const& values, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  picojson::array array(values.begin(), values.end());
  return array;
}

} // namespace mochi
