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

#include <mochi_core/utils/path.h>
#include <mochi_core/utils/string_utils.h>

using namespace mochi;

DynamicArray<std::filesystem::path> mochi::path::ScanDirectoryForFiles(
    std::filesystem::path const& dirPath,
    std::string_view extFilter,
    bool recursive,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(!std::filesystem::is_directory(dirPath), error, "Not a directory");
  MOCHI_ERROR_RETURN(error, {});

  auto const extFilterList = Split(extFilter, "|");
  auto const dirPathStr = dirPath.string();
  DynamicArray<std::filesystem::path> outFiles;

  // Check the file extension using std::string::ends_with (works for compound extensions too)
  auto hasCorrectExt = [&](std::string const& path) {
    for (auto const& ext : extFilterList) {
      if (path.ends_with(ext)) {
        return true;
      }
    }
    return false; // Not found
  };

  // Called for each filesystem entry
  auto onEach = [&](auto const& entry) {
    if (entry.is_regular_file()) {
      auto const& filePath = entry.path();
      if (extFilterList.empty() || hasCorrectExt(filePath.string())) {
        // We want to return a relative file path, but we don't use std::filesystem::relative
        // because it gets confused by symlinks. Instead, we simply remove the directory prefix.
        MOCHI_ASSERT(
            filePath.string().starts_with(dirPathStr),
            "Expected the filePath to start with the specified dirPath, verbatim.");
        auto prefixLen = (dirPathStr.ends_with('/') || dirPathStr.ends_with('\\'))
            ? dirPathStr.length()
            : dirPathStr.length() + 1;
        auto filePathStr = filePath.string().substr(prefixLen);
        // Standardize slash direction
        for (auto& c : filePathStr) {
          if (c == '\\') {
            c = '/';
          }
        }
        outFiles.emplace_back(filePathStr);
      }
    }
  };

  try {
    if (recursive) {
      for (auto const& entry : std::filesystem::recursive_directory_iterator(dirPath)) {
        onEach(entry);
      }
    } else {
      for (auto const& entry : std::filesystem::directory_iterator(dirPath)) {
        onEach(entry);
      }
    }
  } catch (std::exception const& e) {
    // Log the details of the exception and return an error to the users.
    MOCHI_LOG_ERROR("Failed to scan directory for files. Reason: %s", e.what());
    MOCHI_ERROR_SET(error, "Failed to scan directory for files");
    return {};
  }

  // Sort the paths for consistent ordering
  std::ranges::sort(outFiles);

  return outFiles;
}
