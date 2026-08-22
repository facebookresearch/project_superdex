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

#include "meshing/processing_modifiers/processing_export_path.h"

#include "meshing/processing_modifiers/processing_mesh_utils.h" // FileBaseName

#include <superdex_robotics/utils/file_utils.h> // AssetRoleFolderForWrite

#include <filesystem>

namespace superdex::studio::processing {

std::string
DefaultExportPath(std::string const& sourceFilePath, std::string_view roleSubdir, char const* ext) {
  if (sourceFilePath.empty()) {
    return {};
  }
  std::filesystem::path const source(sourceFilePath);
  std::string const base = FileBaseName(sourceFilePath);
  std::filesystem::path const dir =
      superdex::robotics::AssetRoleFolderForWrite(source.parent_path(), roleSubdir);
  std::filesystem::path full = dir / (base + ext);
  full.make_preferred(); // native separators so the OS file dialog honors the initial folder
  return full.string();
}

} // namespace superdex::studio::processing
