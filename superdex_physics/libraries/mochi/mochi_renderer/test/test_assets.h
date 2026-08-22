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

#include <mochi_core/utils/error.h>
#include <mochi_core/utils/path.h>

#include <cstdlib>
#include <string>
#include <string_view>

namespace mochi_renderer::test {

// Root of mochi_renderer's checked-in test assets, with a trailing slash.
[[nodiscard]] inline std::string GetTestAssetsDir() {
  char const* const staged = std::getenv("MOCHI_RENDERER_TEST_ASSETS");
  if (staged != nullptr && staged[0] != '\0') {
    return mochi::path::EnsureTrailingSlash(staged);
  }

  mochi::Error error;
  constexpr std::string_view kRepoRelativeDir = "libraries/mochi/mochi_renderer/test/assets";
  std::string const root = mochi::path::FindParentDirectoryWithChild(kRepoRelativeDir, error);
  if (!error.IsOK()) {
    return {};
  }
  return root + std::string{kRepoRelativeDir} + "/";
}

// Path to one of mochi_renderer's test assets, e.g.
// GetTestAssetPath("basic_shapes/Cube.glb"). Returns the relative path unchanged
// if the asset root cannot be located, so callers report a usable path when
// their existence check fails.
[[nodiscard]] inline std::string GetTestAssetPath(std::string_view relativePath) {
  return GetTestAssetsDir() + std::string{relativePath};
}

} // namespace mochi_renderer::test
