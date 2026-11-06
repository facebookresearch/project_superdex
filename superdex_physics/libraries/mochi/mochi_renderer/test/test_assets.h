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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

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

// Assembles a GLB byte buffer from a JSON descriptor and binary chunk, applying
// the spec-required 4-byte chunk padding (spaces for JSON, zeros for BIN).
[[nodiscard]] inline std::vector<uint8_t> AssembleGlb(std::string json, std::vector<uint8_t> bin) {
  json.append((4 - (json.size() % 4)) % 4, ' ');
  bin.insert(bin.end(), (4 - (bin.size() % 4)) % 4, uint8_t{0});

  auto const jsonLen = static_cast<uint32_t>(json.size());
  auto const binLen = static_cast<uint32_t>(bin.size());
  auto const total = static_cast<uint32_t>(12 + 8 + jsonLen + 8 + binLen);

  std::vector<uint8_t> glb;
  glb.reserve(total);
  auto push32 = [&](uint32_t v) {
    uint8_t b[4];
    std::memcpy(b, &v, 4);
    glb.insert(glb.end(), b, b + 4);
  };
  push32(0x46546C67); // "glTF"
  push32(2);
  push32(total);
  push32(jsonLen);
  push32(0x4E4F534A); // "JSON"
  glb.insert(glb.end(), json.begin(), json.end());
  push32(binLen);
  push32(0x004E4942); // "BIN\0"
  glb.insert(glb.end(), bin.begin(), bin.end());
  return glb;
}

} // namespace mochi_renderer::test
