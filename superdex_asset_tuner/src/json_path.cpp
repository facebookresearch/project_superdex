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

#include "superdex_asset_tuner/json_path.h"

#include <algorithm>
#include <charconv>
#include <optional>
#include <set>
#include <vector>

namespace superdex::tuner {

namespace {

struct PathSegment {
  std::string key;
  std::optional<std::string> selector;
};

bool SplitPath(std::string_view path, std::vector<PathSegment>& out, std::string& outError) {
  out.clear();
  size_t i = 0;
  while (i <= path.size()) {
    PathSegment segment;
    while (i < path.size() && path[i] != '.' && path[i] != '[') {
      segment.key += path[i++];
    }
    if (i < path.size() && path[i] == '[') {
      size_t const close = path.find(']', i);
      if (close == std::string_view::npos) {
        outError = "Unterminated '[' in path '" + std::string(path) + "'.";
        return false;
      }
      segment.selector = std::string(path.substr(i + 1, close - i - 1));
      i = close + 1;
    }
    if (segment.key.empty()) {
      outError = "Empty key in path '" + std::string(path) + "'.";
      return false;
    }
    out.push_back(std::move(segment));
    if (i == path.size()) {
      break;
    }
    if (path[i] != '.') {
      outError = "Expected '.' after ']' in path '" + std::string(path) + "'.";
      return false;
    }
    ++i;
  }
  return true;
}

picojson::value* SelectArrayElement(
    picojson::value& array,
    std::string const& selector,
    std::string const& key,
    std::string& outError) {
  if (!array.is<picojson::array>()) {
    outError = "'" + key + "' is not an array.";
    return nullptr;
  }
  auto& elements = array.get<picojson::array>();
  size_t const eq = selector.find('=');
  if (eq == std::string::npos) {
    size_t index = 0;
    auto const [end, ec] =
        std::from_chars(selector.data(), selector.data() + selector.size(), index);
    if (ec != std::errc{} || end != selector.data() + selector.size()) {
      outError = "Invalid array selector '[" + selector + "]'.";
      return nullptr;
    }
    if (index >= elements.size()) {
      outError = "Index " + selector + " out of range for '" + key + "' (size " +
          std::to_string(elements.size()) + ").";
      return nullptr;
    }
    return &elements[index];
  }
  std::string const field = selector.substr(0, eq);
  std::string const wanted = selector.substr(eq + 1);
  for (auto& element : elements) {
    if (!element.is<picojson::object>()) {
      continue;
    }
    auto const& object = element.get<picojson::object>();
    auto const it = object.find(field);
    if (it != object.end() && it->second.is<std::string>() &&
        it->second.get<std::string>() == wanted) {
      return &element;
    }
  }
  outError = "No element of '" + key + "' has " + field + " == \"" + wanted + "\".";
  return nullptr;
}

// Returns the value at `path`, creating missing object keys (as null) along the way.
picojson::value*
ResolveJsonPath(picojson::value& root, std::string_view path, std::string& outError) {
  std::vector<PathSegment> segments;
  if (!SplitPath(path, segments, outError)) {
    return nullptr;
  }
  picojson::value* current = &root;
  for (PathSegment const& segment : segments) {
    if (current->is<picojson::null>()) {
      *current = picojson::value(picojson::object{});
    }
    if (!current->is<picojson::object>()) {
      outError = "Cannot index into non-object at '" + segment.key + "' in path '" +
          std::string(path) + "'.";
      return nullptr;
    }
    current = &current->get<picojson::object>()[segment.key];
    if (segment.selector) {
      current = SelectArrayElement(*current, *segment.selector, segment.key, outError);
      if (current == nullptr) {
        outError += " (path '" + std::string(path) + "')";
        return nullptr;
      }
    }
  }
  return current;
}

} // namespace

bool SetJsonPath(
    picojson::value& root,
    std::string_view path,
    picojson::value value,
    std::string& outError) {
  picojson::value* target = ResolveJsonPath(root, path, outError);
  if (target == nullptr) {
    return false;
  }
  *target = std::move(value);
  return true;
}

bool SetJsonPathFromText(
    picojson::value& root,
    std::string_view path,
    std::string_view text,
    std::string& outError) {
  picojson::value* target = ResolveJsonPath(root, path, outError);
  if (target == nullptr) {
    return false;
  }
  *target =
      target->is<std::string>() ? picojson::value(std::string(text)) : ParseJsonOrString(text);
  return true;
}

bool ParseJson(std::string_view text, picojson::value& out, std::string& outError) {
  std::string parseError;
  picojson::parse(out, text.begin(), text.end(), &parseError);
  if (!parseError.empty()) {
    outError = "Invalid JSON: " + parseError;
    return false;
  }
  return true;
}

picojson::value ParseJsonOrString(std::string_view text) {
  picojson::value parsed;
  std::string error;
  if (!ParseJson(text, parsed, error)) {
    return {std::string(text)};
  }
  return parsed;
}

bool SplitAssignment(std::string_view assignment, std::string& outPath, std::string& outValue) {
  int depth = 0;
  for (size_t i = 0; i < assignment.size(); ++i) {
    char const c = assignment[i];
    if (c == '[') {
      ++depth;
    } else if (c == ']') {
      depth = std::max(0, depth - 1);
    } else if (c == '=' && depth == 0) {
      if (i == 0) {
        return false;
      }
      outPath = std::string(assignment.substr(0, i));
      outValue = std::string(assignment.substr(i + 1));
      return true;
    }
  }
  return false;
}

namespace {

// Names of the array's elements, or nullopt unless every element is an object with a unique string
// `name`.
std::optional<std::vector<std::string>> ElementNames(picojson::array const& array) {
  std::vector<std::string> names;
  std::set<std::string, std::less<>> unique;
  for (picojson::value const& element : array) {
    if (!element.is<picojson::object>()) {
      return std::nullopt;
    }
    auto const& object = element.get<picojson::object>();
    auto const name = object.find("name");
    if (name == object.end() || !name->second.is<std::string>() ||
        !unique.insert(name->second.get<std::string>()).second) {
      return std::nullopt;
    }
    names.push_back(name->second.get<std::string>());
  }
  return names;
}

void Diff(
    std::string const& path,
    picojson::value const* before,
    picojson::value const* after,
    std::vector<JsonChange>& out) {
  if (before != nullptr && after != nullptr && *before == *after) {
    return;
  }
  auto const record = [&] {
    out.push_back(
        {path,
         before != nullptr ? before->serialize() : std::string{},
         after != nullptr ? after->serialize() : std::string{}});
  };
  if (before == nullptr || after == nullptr) {
    record();
    return;
  }
  if (before->is<picojson::object>() && after->is<picojson::object>()) {
    auto const& a = before->get<picojson::object>();
    auto const& b = after->get<picojson::object>();
    std::set<std::string> keys;
    for (auto const& [key, value] : a) {
      keys.insert(key);
    }
    for (auto const& [key, value] : b) {
      keys.insert(key);
    }
    for (std::string const& key : keys) {
      auto const ai = a.find(key);
      auto const bi = b.find(key);
      Diff(
          path.empty() ? key : path + "." + key,
          ai != a.end() ? &ai->second : nullptr,
          bi != b.end() ? &bi->second : nullptr,
          out);
    }
    return;
  }
  if (before->is<picojson::array>() && after->is<picojson::array>() && !path.empty()) {
    auto const& a = before->get<picojson::array>();
    auto const& b = after->get<picojson::array>();
    auto const aNames = ElementNames(a);
    auto const bNames = ElementNames(b);
    if (aNames && bNames) {
      auto const find = [](std::vector<std::string> const& names,
                           picojson::array const& array,
                           std::string const& name) -> picojson::value const* {
        auto const it = std::ranges::find(names, name);
        return it != names.end() ? &array[static_cast<size_t>(it - names.begin())] : nullptr;
      };
      for (std::string const& name : *aNames) {
        Diff(path + "[name=" + name + "]", find(*aNames, a, name), find(*bNames, b, name), out);
      }
      for (std::string const& name : *bNames) {
        if (find(*aNames, a, name) == nullptr) {
          Diff(path + "[name=" + name + "]", nullptr, find(*bNames, b, name), out);
        }
      }
      return;
    }
    if (a.size() == b.size()) {
      for (size_t i = 0; i < a.size(); ++i) {
        Diff(path + "[" + std::to_string(i) + "]", &a[i], &b[i], out);
      }
      return;
    }
  }
  record();
}

} // namespace

std::vector<JsonChange> DiffJson(picojson::value const& before, picojson::value const& after) {
  std::vector<JsonChange> changes;
  Diff({}, &before, &after, changes);
  return changes;
}

} // namespace superdex::tuner
