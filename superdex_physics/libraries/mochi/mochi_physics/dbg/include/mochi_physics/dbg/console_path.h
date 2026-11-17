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
#include <mochi_core/utils/string_utils.h>
#include <simple_reflection/simple_reflection_console.h>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace mochi::dbg {

/// @brief Resolve a debugger path relative to an absolute working directory.
/// @details Normalize separators and unquoted . and .. steps, preserving quoted names.
/// Return false for malformed paths or attempts to step above the root.
inline bool ResolveConsolePath(
    std::string_view workingDirectory,
    std::string_view path,
    std::string& resolved,
    std::string& error,
    size_t& errorOffset) {
  DynamicArray<std::string> names;
  auto const append = [&](std::string_view text) {
    std::vector<SReflect::detail::ConsolePathStep> parts;
    if (!SReflect::detail::SplitConsolePath(text, parts, &error, &errorOffset)) {
      return false;
    }
    for (auto const& part : parts) {
      if (part.isQuoted) {
        names.emplace_back(text.substr(part.offset, part.length));
      } else if (part.name == "..") {
        if (names.empty()) {
          error = "Cannot go above the root";
          errorOffset = part.offset;
          return false;
        }
        names.pop_back();
      } else if (!part.name.empty() && part.name != ".") {
        names.push_back(part.name);
      }
    }
    return true;
  };
  if (!path.starts_with('/')) {
    if (!workingDirectory.starts_with('/')) {
      error = "Working directory must be absolute";
      return false;
    }
    if (!append(workingDirectory)) {
      return false;
    }
  }
  if (!append(path)) {
    return false;
  }
  resolved = "/" + Join(names, "/");
  return true;
}

} // namespace mochi::dbg
