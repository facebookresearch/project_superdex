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

// Minimal standalone replacements for a handful of mochi_core utilities used in the cli app (which
// does not link mochi_core)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <span>
#include <string>

#if defined(_WIN32)
#define MOCHI_MESH_CLI_PLATFORM_WINDOWS 1
#else
#define MOCHI_MESH_CLI_PLATFORM_WINDOWS 0
#endif

namespace mochi::mesh::cli {

using Vector3d = std::array<double, 3>;

// Bare-minimum stand-in for mochi::Error, which the CLI cannot use: accumulates the first
// error message and reports OK/message. The MOCHI_MESH_CLI_ERROR_* macros mirror the signature of
// their mochi_core counterparts, including the file/line arguments, so that swapping back to
// mochi::Error would be mechanical; this stand-in does not record them.
class CliError {
 public:
  bool IsOK() const {
    return _message.empty();
  }
  void SetFirstError(char const* message, char const*, int) {
    if (_message.empty()) {
      _message = message;
    }
  }
  std::string ToString() const {
    return _message;
  }

 private:
  std::string _message;
};

class Bounds3d {
 public:
  Bounds3d() = default;
  Bounds3d(Vector3d min, Vector3d max) : _min(min), _max(max) {}
  Vector3d GetSize() const {
    return {_max[0] - _min[0], _max[1] - _min[1], _max[2] - _min[2]};
  }

 private:
  Vector3d _min{};
  Vector3d _max{};
};

inline Bounds3d CalcAabb(std::span<double const> coordinates) {
  double constexpr kHighest = std::numeric_limits<double>::max();
  double constexpr kLowest = std::numeric_limits<double>::lowest();
  Vector3d min{kHighest, kHighest, kHighest};
  Vector3d max{kLowest, kLowest, kLowest};
  for (size_t index = 0; index + 3 <= coordinates.size(); index += 3) {
    for (int axis = 0; axis < 3; ++axis) {
      min[axis] = std::min(min[axis], coordinates[index + axis]);
      max[axis] = std::max(max[axis], coordinates[index + axis]);
    }
  }
  return {min, max};
}

} // namespace mochi::mesh::cli

#define MOCHI_MESH_CLI_ERROR_SET(error, message) \
  (error).SetFirstError((message), __FILE__, __LINE__)

#define MOCHI_MESH_CLI_ERROR_IF(condition, error, message) \
  do {                                                     \
    if (condition) {                                       \
      MOCHI_MESH_CLI_ERROR_SET((error), (message));        \
    }                                                      \
  } while (false)

#define MOCHI_MESH_CLI_ERROR_RETURN(error, value) \
  do {                                            \
    if (!(error).IsOK()) {                        \
      return value;                               \
    }                                             \
  } while (false)

#define MOCHI_MESH_CLI_LOG_WARNING(...)           \
  do {                                            \
    std::fprintf(stderr, "[superdex_mesh_cli] "); \
    std::fprintf(stderr, __VA_ARGS__);            \
    std::fprintf(stderr, "\n");                   \
  } while (false)
