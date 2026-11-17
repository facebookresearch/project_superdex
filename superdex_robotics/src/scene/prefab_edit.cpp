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

#include "scene/prefab_edit.h"

#include <mochi_core/utils/json_utils.h>
#include <mochi_core/utils/reflection_utils.h>
#include <mochi_physics/utils/mochi_prefab.h>
#include <superdex_robotics/scene/bot_scene.h>

#include <simple_reflection/simple_reflection.h>

#include <optional>
#include <string>
#include <vector>

#if !MOCHI_USE_REFLECTION
#error Reflection is required for the mochi_bots library internals.
#endif

using namespace mochi;
using namespace superdex::robotics;

namespace {

constexpr char const* kName = "name";

/* The names of an array's elements when each is an object with a string "name", all different. */
std::optional<std::vector<std::string>> ElementNames(picojson::array const& elements) {
  std::vector<std::string> names;
  for (picojson::value const& element : elements) {
    if (!element.is<picojson::object>()) {
      return std::nullopt;
    }
    picojson::object const& members = element.get<picojson::object>();
    auto const name = members.find(kName);
    if (name == members.end() || !name->second.is<std::string>()) {
      return std::nullopt;
    }
    for (std::string const& seen : names) {
      if (seen == name->second.get<std::string>()) {
        return std::nullopt;
      }
    }
    names.push_back(name->second.get<std::string>());
  }
  return names;
}

/* The members of `edited` that differ from `original`, as DiffPrefab describes them; a member
   `edited` lacks is null. */
picojson::object Diff(picojson::object const& original, picojson::object const& edited) {
  picojson::object diff;
  for (auto const& [key, value] : edited) {
    auto const found = original.find(key);
    if (found != original.end() && found->second == value) {
      continue;
    }
    if (found != original.end() && found->second.is<picojson::object>() &&
        value.is<picojson::object>()) {
      diff[key] = picojson::value(
          Diff(found->second.get<picojson::object>(), value.get<picojson::object>()));
      continue;
    }
    if (found != original.end() && found->second.is<picojson::array>() &&
        value.is<picojson::array>()) {
      picojson::array const& before = found->second.get<picojson::array>();
      picojson::array const& after = value.get<picojson::array>();
      std::optional<std::vector<std::string>> const names = ElementNames(after);
      if (names && ElementNames(before) == names) {
        picojson::object byName;
        for (std::size_t i = 0; i < after.size(); ++i) {
          picojson::object changed =
              Diff(before[i].get<picojson::object>(), after[i].get<picojson::object>());
          if (!changed.empty()) {
            byName[(*names)[i]] = picojson::value(changed);
          }
        }
        diff[key] = picojson::value(byName);
        continue;
      }
    }
    diff[key] = value;
  }
  for (auto const& [key, value] : original) {
    if (edited.find(key) == edited.end()) {
      diff[key] = picojson::value();
    }
  }
  return diff;
}

template <typename Prefab>
DynamicString DiffPrefabs(Prefab const& original, Prefab const& edited) {
  picojson::value before;
  picojson::value after;
  SReflect::ToJsonValue(original, before);
  SReflect::ToJsonValue(edited, after);
  picojson::object const diff = Diff(before.get<picojson::object>(), after.get<picojson::object>());
  if (diff.empty()) {
    return {};
  }
  std::string const json = picojson::value(diff).serialize();
  return DynamicString{json.c_str(), json.size()};
}

/* `edit` in the shape of `current`, the prefab's JSON, so that reading it into the prefab sets only
   the edited fields: each object of named elements becomes an array as long as the prefab's, with
   an empty object for each element the edit leaves alone. */
bool Sparse(
    picojson::object const& edit,
    picojson::object const& current,
    picojson::object& sparse,
    std::string& problem) {
  for (auto const& [key, value] : edit) {
    if (value.is<picojson::null>()) {
      problem = "it resets " + key;
      return false;
    }
    auto const found = current.find(key);
    bool const isObject = value.is<picojson::object>();
    if (found == current.end() || !isObject) {
      sparse[key] = value;
      continue;
    }
    picojson::object const& members = value.get<picojson::object>();
    if (found->second.is<picojson::object>()) {
      picojson::object inner;
      if (!Sparse(members, found->second.get<picojson::object>(), inner, problem)) {
        return false;
      }
      sparse[key] = picojson::value(inner);
      continue;
    }
    if (!found->second.is<picojson::array>()) {
      sparse[key] = value;
      continue;
    }
    picojson::array const& elements = found->second.get<picojson::array>();
    std::optional<std::vector<std::string>> const names = ElementNames(elements);
    if (!names) {
      problem = key + " has no named elements";
      return false;
    }
    picojson::array out(elements.size(), picojson::value(picojson::object{}));
    for (auto const& [name, fields] : members) {
      std::size_t i = 0;
      while (i < names->size() && (*names)[i] != name) {
        ++i;
      }
      if (i == names->size() || !fields.is<picojson::object>()) {
        problem = key + " has no element " + name;
        return false;
      }
      picojson::object element;
      if (!Sparse(
              fields.get<picojson::object>(),
              elements[i].get<picojson::object>(),
              element,
              problem)) {
        return false;
      }
      out[i] = picojson::value(element);
    }
    sparse[key] = picojson::value(out);
  }
  return true;
}

template <typename Prefab>
void ApplyPrefabEdits(std::string_view edit, Prefab& prefab, Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (edit.empty()) {
    return;
  }
  picojson::value const parsed =
      ParseJsonFromString(Span<char const>{edit.data(), edit.size()}, error);
  MOCHI_ERROR_IF(!parsed.is<picojson::object>(), error, "A prefab edit must be a JSON object");
  MOCHI_ERROR_RETURN(error);
  picojson::value current;
  SReflect::ToJsonValue(prefab, current);
  picojson::object sparse;
  std::string problem;
  if (!Sparse(parsed.get<picojson::object>(), current.get<picojson::object>(), sparse, problem)) {
    MOCHI_LOG_ERROR("The prefab edit does not fit its prefab: %s", problem.c_str());
    MOCHI_ERROR_SET(error, "The prefab edit does not fit its prefab");
    return;
  }
  SReflect::IssueList issues;
  SReflect::FromJsonValue(
      prefab, picojson::value(sparse), SReflect::DeserializeFlags::WarnIfExtraneousFields, &issues);
  if (!issues.empty()) {
    LogDeserializationWarnings("The prefab edit does not fit its prefab's fields", issues);
    MOCHI_ERROR_SET(error, "The prefab edit does not fit its prefab's fields");
  }
}

} // namespace

DynamicString superdex::robotics::DiffPrefab(BotPrefab const& original, BotPrefab const& edited) {
  return DiffPrefabs(original, edited);
}

DynamicString superdex::robotics::DiffPrefab(
    prefab::ScenePrefab const& original,
    prefab::ScenePrefab const& edited) {
  return DiffPrefabs(original, edited);
}

void superdex::robotics::ApplyPrefabEdit(std::string_view edit, BotPrefab& prefab, Error& error) {
  ApplyPrefabEdits(edit, prefab, error);
}

void superdex::robotics::ApplyPrefabEdit(
    std::string_view edit,
    prefab::ScenePrefab& prefab,
    Error& error) {
  ApplyPrefabEdits(edit, prefab, error);
}
