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

#include <superdex_robotics/controllers/controller_basic_jsc_pd.h>
#include <superdex_robotics/controllers/controller_basic_osc_pd.h>
#include <superdex_robotics/controllers/controller_mochi_articulated_pose.h>
#include <superdex_robotics/scene/bot_scene.h>
#if MOCHI_INTERNAL
#include <superdex_robotics/controllers/internal/controller_osc_v1.h>
#include <superdex_robotics/controllers/internal/controller_osc_v2.h>
#endif

#include <array>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <utility>

namespace superdex::studio {

// The internal-only OSC controllers (ControllerOscV1, ControllerOscV2) in the tables below.
#if MOCHI_INTERNAL
inline constexpr std::size_t kInternalControllerCount = 2;
#else
inline constexpr std::size_t kInternalControllerCount = 0;
#endif

// Calls fn.template operator()<TController>() for each built-in controller type until one returns
// true, and returns whether any did.
template <typename Fn>
bool ForEachBuiltinController(Fn&& fn) {
  namespace robotics = superdex::robotics;
  bool const found = fn.template operator()<robotics::ControllerBasicJscPd>() ||
      fn.template operator()<robotics::ControllerBasicOscPd>() ||
      fn.template operator()<robotics::ControllerMochiArticulatedPose>();
#if MOCHI_INTERNAL
  return found || fn.template operator()<robotics::ControllerOscV1>() ||
      fn.template operator()<robotics::ControllerOscV2>();
#else
  return found;
#endif
}

constexpr std::array<std::string_view, 3 + kInternalControllerCount> kBuiltinControllerTypes = {
    superdex::robotics::ControllerBasicJscPd::TypeName(),
    superdex::robotics::ControllerBasicOscPd::TypeName(),
    superdex::robotics::ControllerMochiArticulatedPose::TypeName(),
#if MOCHI_INTERNAL
    superdex::robotics::ControllerOscV1::TypeName(),
    superdex::robotics::ControllerOscV2::TypeName(),
#endif
};

// Legacy names old scenes still use for built-in controllers, mirroring
// superdex::robotics::RegisterLegacyControllerTypes.
constexpr std::array<std::pair<std::string_view, std::string_view>, 1 + kInternalControllerCount>
    kLegacyControllerTypes = {{
        {"joint_space_pd_v1", superdex::robotics::ControllerBasicJscPd::TypeName()},
#if MOCHI_INTERNAL
        {"opspace_v1", superdex::robotics::ControllerOscV1::TypeName()},
        {"opspace_v2", superdex::robotics::ControllerOscV2::TypeName()},
#endif
    }};

// The canonical name for a legacy controller type name, or @p type unchanged.
constexpr std::string_view CanonicalControllerType(std::string_view type) {
  for (auto const& [legacy, canonical] : kLegacyControllerTypes) {
    if (type == legacy) {
      return canonical;
    }
  }
  return type;
}

[[nodiscard]] bool IsBuiltinControllerType(std::string_view type);

// Default params for a built-in type, or "{}" for any other type.
[[nodiscard]] std::string DefaultParamsJson(std::string_view type);

// Parses @p json and reserializes it pretty-printed into @p out, so equal params compare equal as
// strings. Built-in types parse strictly into their Params struct, as the runtime does; other types
// only need to be a JSON object.
bool NormalizeParamsJson(
    std::string_view type,
    std::string_view json,
    std::string& out,
    std::string& error);

// Resizes every resizable array field of @p structPtr to @p count, filling new elements with the
// last existing one so a single gain extends to every joint. Returns whether any field changed.
bool ResizeArrayFields(SReflect::StructTypeInfo const& structInfo, void* structPtr, size_t count);

// Loads a params file into normalized JSON (see NormalizeParamsJson).
bool LoadParamsFile(
    std::string_view type,
    std::string const& path,
    std::string& json,
    std::string& error);

// Writes @p json to a params file, creating its parent directory if needed.
bool SaveParamsFile(
    std::string_view type,
    std::string const& path,
    std::string_view json,
    std::string& error);

// Params loaded from or destined for .superdex_controller files, keyed by absolute path. Whether an
// entry is modified is derived by comparing it with the value last loaded or saved.
class ControllerParamsCache {
 public:
  struct Entry {
    std::string type;
    // Current value, normalized (see NormalizeParamsJson).
    std::string json;
    // Value last loaded from or saved to the file; empty if the file has not been written yet.
    std::string baselineJson;
    // Why the file failed to load. Such an entry is never written until its json is replaced.
    std::string error;
  };

  // Returns the entry for @p path, loading the file on first use. An entry keeps the type it was
  // first loaded as.
  Entry& GetOrLoad(std::string const& path, std::string_view type);
  Entry* Find(std::string const& path);
  // Sets @p path to @p json without reading the file; it is written by the next SaveModified.
  Entry& Adopt(std::string const& path, std::string_view type, std::string json);
  // Adopts @p to with @p from's current value and reverts @p from to its file's value, so the edits
  // move to the new file instead of also being written to the old one.
  Entry& SaveAs(std::string const& from, std::string const& to);

  [[nodiscard]] static bool IsModified(Entry const& entry) {
    return entry.json != entry.baselineJson;
  }

  // Writes every modified entry, stopping at the first failure.
  bool SaveModified(std::string& error);

  [[nodiscard]] std::map<std::string, std::string> CaptureValues() const;
  // Restores values from CaptureValues. Entries the capture lacks revert to their file's value.
  void RestoreValues(std::map<std::string, std::string> const& values);

 private:
  std::map<std::string, Entry> _entries;
};

// Returns the file to save a controller's params to, or an empty string to cancel.
using ParamsPathResolver = std::function<std::string(
    superdex::robotics::BotEntry const& bot,
    superdex::robotics::ControllerEntry const& controller)>;

// Moves every controller's inline JSON params out of @p prefab: params equal to the type's defaults
// are cleared, and others are adopted into @p cache at the path @p resolvePath returns (written by
// the cache's next SaveModified). Leaves @p prefab and @p cache unchanged if any params are invalid
// or a path is canceled.
bool ExternalizeInlineParams(
    superdex::robotics::BotScenePrefab& prefab,
    ControllerParamsCache& cache,
    ParamsPathResolver const& resolvePath,
    std::string& error);

} // namespace superdex::studio
