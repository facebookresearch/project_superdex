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

#include "urdf_utils.h"

#include <mochi_core/utils/log.h>

#include <filesystem>
#include <initializer_list>

using namespace mochi;
using namespace superdex::robotics;

std::string superdex::robotics::FindUrdfPackageRoot(std::string_view urdfPath) {
  namespace fs = std::filesystem;
  fs::path path = fs::path(urdfPath).parent_path();

  while (!path.empty()) {
    fs::path packageXml = path / "package.xml";
    if (fs::exists(packageXml)) {
      return path.string();
    }
    fs::path parent = path.parent_path();
    if (parent == path) {
      break;
    }
    path = parent;
  }
  return {};
}

std::optional<std::string> superdex::robotics::ResolveMeshPath(
    std::string_view meshPath,
    std::string_view urdfPath) {
  namespace fs = std::filesystem;

  if (meshPath.empty()) {
    return std::nullopt;
  }

  // Returns the first candidate that exists on disk (canonicalized), or nullopt.
  auto firstExisting =
      [](std::initializer_list<fs::path> const candidates) -> std::optional<std::string> {
    for (fs::path const& candidate : candidates) {
      if (candidate.empty()) {
        continue;
      }
      fs::path const resolved = fs::weakly_canonical(candidate);
      if (fs::exists(resolved)) {
        return resolved.string();
      }
    }
    return std::nullopt;
  };

  constexpr std::string_view kPackagePrefix = "package://";
  if (meshPath.starts_with(kPackagePrefix)) {
    // `package://<pkg>/<rel>`: <pkg> is the ROS package name, <rel> the path within it.
    std::string_view const remainder = meshPath.substr(kPackagePrefix.size());
    size_t const slashPos = remainder.find('/');
    if (slashPos == std::string_view::npos) {
      return std::nullopt;
    }
    // The leading <pkg> segment (the ROS package name).
    fs::path const packageName{std::string(remainder.substr(0, slashPos))};
    // Path within the package, with the leading <pkg> segment stripped.
    fs::path const relativePath{std::string(remainder.substr(slashPos + 1))};
    // Path including the <pkg> segment (e.g. "<pkg>/<rel>").
    fs::path const packageRelativePath{std::string(remainder)};

    std::string const packageRoot = FindUrdfPackageRoot(urdfPath);
    if (!packageRoot.empty()) {
      fs::path const root{packageRoot};
      // Try, in order: the package root with the <pkg> segment stripped (the common
      // case where the root directory is itself named <pkg>), then the parent of the
      // package root keeping the <pkg> segment (the single-level parent-directory
      // fallback used by the Unreal importer for layouts where the package-name
      // segment is duplicated), then the root keeping the <pkg> segment.
      return firstExisting(
          {root / relativePath,
           root.parent_path() / packageRelativePath,
           root / packageRelativePath});
    }

    // No package.xml found (common for datasets that strip it, e.g. xacro-generated
    // exports). Walk up from the URDF directory and try to resolve the mesh against
    // each ancestor, handling two malformed-but-common conventions:
    //   - `package://<pkg>/<rel>` where an ancestor directory is named <pkg>
    //     (resolve <ancestor>/<rel>), and
    //   - `package://<rel>` where the package-name segment was omitted and <rel>
    //     (e.g. "meshes/...") lives directly under the package root
    //     (resolve <ancestor>/<remainder>, i.e. the full remainder).
    // The closest matching ancestor wins, which in practice is the package root.
    fs::path const urdfDir = fs::path(urdfPath).parent_path();
    for (fs::path dir = urdfDir; !dir.empty();) {
      if (dir.filename() == packageName) {
        if (auto const hit = firstExisting({dir / relativePath})) {
          return hit;
        }
      }
      if (auto const hit = firstExisting({dir / packageRelativePath})) {
        return hit;
      }
      fs::path parent = dir.parent_path();
      if (parent == dir) {
        break;
      }
      dir = std::move(parent);
    }

    // Last resort: resolve relative to the URDF directory itself.
    return firstExisting({urdfDir / relativePath, urdfDir / packageRelativePath});
  }

  fs::path const meshFsPath{std::string(meshPath)};
  if (meshFsPath.is_absolute()) {
    if (fs::exists(meshFsPath)) {
      return fs::canonical(meshFsPath).string();
    }
    MOCHI_LOG_ERROR(
        "URDF mesh path '%s' is absolute but does not exist on disk.",
        std::string(meshPath).c_str());
    return std::nullopt;
  }

  fs::path const urdfDir = fs::path(urdfPath).parent_path();
  return firstExisting({urdfDir / meshFsPath});
}

// Algorithm from Dave Koelle and implementation adapted from alphanum.hpp by Dirk Jagdmann.
int superdex::robotics::AlphaNumCompare(char const* l, char const* r) {
  auto IsDigit = [](char const c) { return c >= '0' && c <= '9'; };
  enum mode_t { STRING, NUMBER } mode = STRING;
  while (*l && *r) {
    if (mode == STRING) {
      char l_char = *l, r_char = *r;
      while (l_char && r_char) {
        bool const l_digit = IsDigit(l_char), r_digit = IsDigit(r_char);
        if (l_digit && r_digit) {
          mode = NUMBER;
          break;
        }
        if (l_digit) {
          return -1;
        }
        if (r_digit) {
          return +1;
        }
        int const diff = l_char - r_char;
        if (diff != 0) {
          return diff;
        }
        ++l;
        ++r;
        l_char = *l;
        r_char = *r;
      }
    } else {
      unsigned long l_int = 0;
      while (*l && IsDigit(*l)) {
        l_int = l_int * 10 + *l - '0';
        ++l;
      }
      unsigned long r_int = 0;
      while (*r && IsDigit(*r)) {
        r_int = r_int * 10 + *r - '0';
        ++r;
      }
      long const diff = l_int - r_int;
      if (diff != 0) {
        return static_cast<int>(diff);
      }
      mode = STRING;
    }
  }
  if (*r) {
    return -1;
  }
  if (*l) {
    return +1;
  }
  return 0;
}
