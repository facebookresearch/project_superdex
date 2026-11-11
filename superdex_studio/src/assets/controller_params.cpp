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

#include "assets/controller_params.h"

#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/reflection.h>

#include <picojson/picojson.h>

#include <filesystem>
#include <vector>

namespace superdex::studio {

namespace {

bool CreateParentDirectory(std::string const& path, std::string& error) {
  auto const parent = std::filesystem::path(path).parent_path();
  if (parent.empty()) {
    return true;
  }
  std::error_code ec;
  std::filesystem::create_directories(parent, ec);
  if (ec) {
    error = "Failed to create directory '" + parent.generic_string() + "': " + ec.message();
    return false;
  }
  return true;
}

// Parses @p json strictly into @p params. The value must be inline JSON: anything else would be
// treated as a file path by LoadParamsFromPathOrJson.
template <typename TParams>
bool ParseParams(std::string_view json, TParams& params, std::string& error) {
  if (!superdex::robotics::IsInlineJson(json)) {
    error = "Expected a JSON object";
    return false;
  }
  mochi::Error parseError;
  params = superdex::robotics::LoadParamsFromPathOrJson<TParams>(json, parseError);
  if (!parseError.IsOK()) {
    error = parseError.GetDescription();
    return false;
  }
  return true;
}

bool NormalizeCustomParams(std::string_view json, std::string& out, std::string& error) {
  picojson::value value;
  std::string const text{json};
  picojson::parse(value, text.begin(), text.end(), &error);
  if (!error.empty()) {
    return false;
  }
  if (!value.is<picojson::object>()) {
    error = "Expected a JSON object";
    return false;
  }
  out = value.serialize(true);
  return true;
}

} // namespace

bool IsBuiltinControllerType(std::string_view type) {
  return ForEachBuiltinController(
      [&]<typename T>() { return T::TypeName() == CanonicalControllerType(type); });
}

std::string DefaultParamsJson(std::string_view type) {
  std::string json = "{}";
  ForEachBuiltinController([&]<typename T>() {
    if (T::TypeName() != CanonicalControllerType(type)) {
      return false;
    }
    json = SReflect::ToJsonString(typename T::Params{}, true);
    return true;
  });
  return json;
}

bool NormalizeParamsJson(
    std::string_view type,
    std::string_view json,
    std::string& out,
    std::string& error) {
  bool ok = false;
  bool const builtin = ForEachBuiltinController([&]<typename T>() {
    if (T::TypeName() != CanonicalControllerType(type)) {
      return false;
    }
    typename T::Params params;
    ok = ParseParams(json, params, error);
    if (ok) {
      out = SReflect::ToJsonString(params, true);
    }
    return true;
  });
  return builtin ? ok : NormalizeCustomParams(json, out, error);
}

bool ResizeArrayFields(SReflect::StructTypeInfo const& structInfo, void* structPtr, size_t count) {
  bool changed = false;
  for (SReflect::FieldTypeInfo const* field : structInfo._fields) {
    if (field->_innerTypeInfo->_coreType != SReflect::CoreType::CT_array) {
      continue;
    }
    auto const& arrInfo = static_cast<SReflect::ArrayTypeInfo const&>(*field->_innerTypeInfo);
    void* const arrPtr = field->GetFieldPtr(structPtr);
    size_t const oldCount = arrInfo.GetNumElements(arrPtr);
    if (!arrInfo.CanResize() || oldCount == count) {
      continue;
    }
    arrInfo.SetNumElements(arrPtr, count);
    for (size_t i = oldCount; oldCount > 0 && i < count; ++i) {
      arrInfo._innerTypeInfo->Set(
          arrInfo.GetElement(arrPtr, oldCount - 1), arrInfo.GetElement(arrPtr, i));
    }
    changed = true;
  }
  return changed;
}

bool LoadParamsFile(
    std::string_view type,
    std::string const& path,
    std::string& json,
    std::string& error) {
  mochi::Error readError;
  std::string const text = mochi::ReadFileString(path, readError);
  if (!readError.IsOK()) {
    error = "Failed to read '" + path + "': " + readError.GetDescription();
    return false;
  }
  if (!NormalizeParamsJson(type, text, json, error)) {
    error = "Failed to parse '" + path + "': " + error;
    return false;
  }
  return true;
}

bool SaveParamsFile(
    std::string_view type,
    std::string const& path,
    std::string_view json,
    std::string& error) {
  std::string normalized;
  if (!NormalizeParamsJson(type, json, normalized, error) || !CreateParentDirectory(path, error)) {
    return false;
  }
  mochi::Error writeError;
  mochi::WriteFile(path, mochi::Span<char const>(normalized.data(), normalized.size()), writeError);
  if (!writeError.IsOK()) {
    error = "Failed to write '" + path + "': " + writeError.GetDescription();
    return false;
  }
  return true;
}

ControllerParamsCache::Entry& ControllerParamsCache::GetOrLoad(
    std::string const& path,
    std::string_view type) {
  auto [it, inserted] = _entries.try_emplace(path);
  Entry& entry = it->second;
  if (inserted) {
    entry.type = type;
    if (LoadParamsFile(type, path, entry.json, entry.error)) {
      entry.baselineJson = entry.json;
    }
  }
  return entry;
}

ControllerParamsCache::Entry* ControllerParamsCache::Find(std::string const& path) {
  auto const it = _entries.find(path);
  return it != _entries.end() ? &it->second : nullptr;
}

ControllerParamsCache::Entry&
ControllerParamsCache::Adopt(std::string const& path, std::string_view type, std::string json) {
  Entry& entry = _entries[path];
  entry.type = type;
  entry.json = std::move(json);
  entry.baselineJson.clear();
  entry.error.clear();
  return entry;
}

ControllerParamsCache::Entry& ControllerParamsCache::SaveAs(
    std::string const& from,
    std::string const& to) {
  Entry& source = _entries.at(from);
  Entry& target = Adopt(to, source.type, source.json);
  source.json = source.baselineJson;
  return target;
}

bool ControllerParamsCache::SaveModified(std::string& error) {
  for (auto& [path, entry] : _entries) {
    if (!IsModified(entry)) {
      continue;
    }
    if (!SaveParamsFile(entry.type, path, entry.json, error)) {
      return false;
    }
    entry.baselineJson = entry.json;
    entry.error.clear();
  }
  return true;
}

std::map<std::string, std::string> ControllerParamsCache::CaptureValues() const {
  std::map<std::string, std::string> values;
  for (auto const& [path, entry] : _entries) {
    values.emplace(path, entry.json);
  }
  return values;
}

void ControllerParamsCache::RestoreValues(std::map<std::string, std::string> const& values) {
  for (auto& [path, entry] : _entries) {
    // An entry missing from the snapshot had not been loaded or adopted yet, so it still matched
    // its file.
    auto const it = values.find(path);
    entry.json = it != values.end() ? it->second : entry.baselineJson;
  }
}

bool ExternalizeInlineParams(
    superdex::robotics::BotScenePrefab& prefab,
    ControllerParamsCache& cache,
    ParamsPathResolver const& resolvePath,
    std::string& error) {
  struct Pending {
    superdex::robotics::ControllerEntry* controller;
    std::string json;
    // Empty when the params are the type's defaults.
    std::string path;
  };
  std::vector<Pending> pending;
  for (auto& bot : prefab.bots) {
    for (auto& controller : bot.controllers) {
      if (!superdex::robotics::IsInlineJson(controller.params)) {
        continue;
      }
      std::string const type{controller.type};
      std::string json;
      if (!NormalizeParamsJson(type, controller.params, json, error)) {
        std::string message = "Invalid params for controller '";
        message += std::string_view(bot.name);
        message += '/';
        message += std::string_view(controller.name);
        message += "': ";
        message += error;
        error = std::move(message);
        return false;
      }
      std::string path;
      if (json != DefaultParamsJson(type)) {
        path = resolvePath(bot, controller);
        if (path.empty()) {
          error = "Saving controller parameters was canceled";
          return false;
        }
      }
      pending.push_back({&controller, std::move(json), std::move(path)});
    }
  }

  for (auto& [controller, json, path] : pending) {
    if (path.empty()) {
      controller->params.clear();
    } else {
      cache.Adopt(path, std::string(controller->type), std::move(json));
      controller->params = path;
    }
  }
  return true;
}

} // namespace superdex::studio
