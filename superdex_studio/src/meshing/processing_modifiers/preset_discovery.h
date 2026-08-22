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

// Discovery of shipped "preset" pipeline JSONs. Presets live in a `processing_presets/` folder
// bundled next to the studio executable (via the app's BUCK `resources`). Each is a
// StudioProcessing document (same schema as a model's saved pipeline) that the "Populate Default
// Processing" dropdown can load to replace the current stack.

#include <string>
#include <vector>

namespace superdex::studio {

struct ProcessingPreset {
  std::string name; // display name (file stem, underscores shown as spaces)
  std::string path; // absolute path to the .json
  std::string description; // optional tooltip text from the JSON's "description" field
};

// Returns the presets found next to the executable's `processing_presets/` folder (override the
// directory with the MOCHI_STUDIO_PRESETS_DIR environment variable), sorted by name. Empty if the
// folder is absent.
std::vector<ProcessingPreset> DiscoverProcessingPresets();

} // namespace superdex::studio
