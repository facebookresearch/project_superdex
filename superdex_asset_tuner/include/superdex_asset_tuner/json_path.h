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

#include <picojson/picojson.h>

#include <string>
#include <string_view>
#include <vector>

namespace superdex::tuner {

// Sets `root[path] = value`. Path: dot-separated keys, each optionally followed by `[index]` or
// `[key=value]` (selects the array element whose string field `key` equals `value`). Missing object
// keys are created; array selectors must match an existing element.
bool SetJsonPath(
    picojson::value& root,
    std::string_view path,
    picojson::value value,
    std::string& outError);

// Parses `text` as JSON; text that is not valid JSON becomes a JSON string.
[[nodiscard]] picojson::value ParseJsonOrString(std::string_view text);

// Like SetJsonPath with ParseJsonOrString(text), except that an existing string value stays a
// string (so `name=123` sets "123", not 123).
bool SetJsonPathFromText(
    picojson::value& root,
    std::string_view path,
    std::string_view text,
    std::string& outError);

// Parses `text` as JSON into `out`.
bool ParseJson(std::string_view text, picojson::value& out, std::string& outError);

// Splits a `path=value` assignment (e.g. a CLI `--set`) at the first '=' outside `[...]`, so name
// selectors such as `loads[name=push].profile.magnitude=3` keep their '='. False if there is no
// such '=' or the path is empty.
bool SplitAssignment(std::string_view assignment, std::string& outPath, std::string& outValue);

// One difference found by DiffJson. `before` and `after` are compact JSON, empty where the value is
// absent.
struct JsonChange {
  std::string path;
  std::string before;
  std::string after;
};

// Differences between two JSON documents, as paths in SetJsonPath's syntax. Array elements that are
// objects with unique string `name` fields are matched by name (`loads[name=push].enabled`), other
// arrays of the same length by index; an array whose elements cannot be matched is one change.
[[nodiscard]] std::vector<JsonChange> DiffJson(
    picojson::value const& before,
    picojson::value const& after);

} // namespace superdex::tuner
