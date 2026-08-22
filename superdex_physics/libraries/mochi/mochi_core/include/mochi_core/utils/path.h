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

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/span.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>

namespace mochi::path {

/**
 Add a trailing slash, if none exists
*/
[[nodiscard]] inline std::string EnsureTrailingSlash(std::string_view path) {
  if (!path.empty()) {
    auto lastChar = path[path.length() - 1];
    if (lastChar != '/' && lastChar != '\\') {
      return std::string{path} + "/";
    }
  }
  return std::string{path};
}

/**
  Walk up the directory hierarchy, starting from the current working directory. Return the first
  parent directory that contains the specified child path (file or subdirectory).
*/
[[nodiscard]] inline std::string FindParentDirectoryWithChild(
    std::string_view relativeChildPath,
    Error& error) {
  MOCHI_ERROR_RETURN(error, "");
  auto dir = std::filesystem::absolute(std::filesystem::current_path());
  auto child = std::filesystem::path(relativeChildPath);
  while (true) {
    if (std::filesystem::exists(dir / child)) {
      return EnsureTrailingSlash(dir.string()); // Found it
    }
    if (dir.has_parent_path() && (dir.parent_path() != dir)) {
      dir = dir.parent_path();
    } else {
      MOCHI_ERROR_SET(error, "Not found");
      return "";
    }
  }
}

/**
  Return the root of the Mercurial repo (e.g. fbsource) by walking up the hierarchy, starting from
  the current working directory. It is identified by the ".hg" subdirectory at that location.
*/
[[nodiscard]] inline std::string FindHgRepoRoot(Error& error) {
  return FindParentDirectoryWithChild(".hg", error);
}

/**
  Attempt locate the "assets" directory by checking these cases, in order:

    Case 1: If environment variable "MOCHI_ASSETS_PATH" is defined, then return it.
            Used for unit tests.

    Case 2: If an "assets" directory exists in the current working directory, then return it.
            Used when running from a packaged build.

    Case 3: If the current working directory is within a Mercurial repo, then assume it is fbsource.
            Format the path relative to the repo root. Used when running directly from "buck-out" or
            when using "buck run" commands.

    Case 4: Walk up the directory hierarchy looking for an "assets" directory.
            Used when running a CMake target from a location like "<mochi_github_root>/build/bin".
*/
[[nodiscard]] inline std::string FindAssetsDirectory(Error& error) {
  MOCHI_ERROR_RETURN(error, "");

  // Case 1
  char const* envPath = std::getenv("MOCHI_ASSETS_PATH");
  if (envPath) {
    return EnsureTrailingSlash(envPath);
  }

  // Case 2
  if (std::filesystem::exists("./assets")) {
    return "./assets/";
  }

  // Case 3
  std::string rootDir = FindHgRepoRoot(error);
  if (error.IsOK()) {
    return rootDir + "arvr/libraries/mochi/assets/";
  }

  // Case 4:
  error = {}; // Clear previous error
  rootDir = FindParentDirectoryWithChild("assets", error);
  if (error.IsOK()) {
    return rootDir + "assets/";
  } else {
    // Not found
    return "";
  }
}

/**
  Return true if the path looks like an absolute path.
*/
[[nodiscard]] inline bool IsAbsolutePath(std::string_view path) {
  if (!path.empty()) {
    // Unix style absolute paths
    if (path[0] == '~' || path[0] == '/') {
      return true;
    }
    // Windows style absolute paths
    if ((path.find(":/") != std::string_view::npos) ||
        (path.find(":\\") != std::string_view::npos) ||
        (path.length() >= 2 && path[0] == '\\' && path[1] == '\\')) {
      return true;
    }
  }
  return false;
}

/**
  If the input path looks like an absolute path, then return it as-is.
  Else join it with the given root path.
*/
[[nodiscard]] inline std::string GetFullPath(
    std::string_view inputPath,
    std::string_view rootForRelativePath) {
  if (IsAbsolutePath(inputPath)) {
    return std::string{inputPath}; // Return as-is
  } else {
    return EnsureTrailingSlash(rootForRelativePath) + std::string{inputPath};
  }
}

/**
  If the input path is already a relative path, then return it as-is.
  Else convert it to a relative path.
 */
[[nodiscard]] inline std::string GetRelativePath(
    std::string_view inputPath,
    std::string_view rootForRelativePath) {
  if (!IsAbsolutePath(inputPath)) {
    return std::string{inputPath}; // Return as-is.
  }
  return std::filesystem::relative({inputPath}, {rootForRelativePath}).generic_string();
}

/**
 * @brief Find files in a directory tree.
 *
 * @param[in] dirPath Directory path
 * @param[in] extFilter An optional '|' separated list of file extensions to include
 * (example: ".txt|.json"). Ignored if empty.
 * @param[in] recursive If true, include files in subdirectories recursively.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return DynamicArray of file paths
 */
DynamicArray<std::filesystem::path> ScanDirectoryForFiles(
    std::filesystem::path const& dirPath,
    std::string_view extFilter,
    bool recursive,
    Error& error);

} // namespace mochi::path
