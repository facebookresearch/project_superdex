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

#include "meshing/processing_modifiers/preset_discovery.h"

#include "app/app.h"

#include <picojson/picojson.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace superdex::studio {

std::vector<ProcessingPreset> DiscoverProcessingPresets() {
  std::vector<ProcessingPreset> presets;

  std::filesystem::path dir;
  if (char const* const overrideDir = std::getenv("MOCHI_STUDIO_PRESETS_DIR")) {
    dir = overrideDir;
  } else {
    std::filesystem::path const exeDir = SuperDexStudio::GetExecutableDir();
    if (exeDir.empty()) {
      return presets;
    }
    dir = exeDir / "processing_presets";
  }

  std::error_code ec;
  if (!std::filesystem::is_directory(dir, ec)) {
    return presets;
  }
  std::filesystem::directory_iterator dirIt(dir, ec);
  std::filesystem::directory_iterator const end;
  for (; !ec && dirIt != end; dirIt.increment(ec)) {
    std::error_code entryEc;
    if (!dirIt->is_regular_file(entryEc) || entryEc || dirIt->path().extension() != ".json") {
      continue;
    }
    ProcessingPreset preset;
    preset.name = dirIt->path().stem().string();
    std::replace(preset.name.begin(), preset.name.end(), '_', ' '); // underscores -> spaces
    preset.path = dirIt->path().string();

    std::ifstream file(dirIt->path());
    if (file.is_open()) {
      picojson::value rootVal;
      std::string parseErr = picojson::parse(rootVal, file);
      if (parseErr.empty() && rootVal.is<picojson::object>()) {
        auto const& root = rootVal.get<picojson::object>();
        auto const it = root.find("description");
        if (it != root.end() && it->second.is<std::string>()) {
          preset.description = it->second.get<std::string>();
        }
      }
    }

    presets.push_back(std::move(preset));
  }
  std::sort(
      presets.begin(), presets.end(), [](ProcessingPreset const& a, ProcessingPreset const& b) {
        return a.name < b.name;
      });
  return presets;
}

} // namespace superdex::studio
