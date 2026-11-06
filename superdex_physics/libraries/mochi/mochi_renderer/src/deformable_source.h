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

#include <mochi_core/utils/span.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mochi_renderer {

// The one primitive of a GLB that a RenderModel's independently deformable instances draw.
// Geometry is in the glTF mesh node's local frame; texcoords (TEXCOORD_0) and colors (COLOR_0, as
// RGBA) are empty when the primitive lacks them.
struct DeformableSourceData {
  // Column-major transform from the mesh node's frame to the glTF scene root's.
  std::array<float, 16> rootFromNode{};
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<float> texcoords;
  std::vector<float> colors;
  std::vector<int> indices;
  // Why the GLB cannot be deformed, as a string literal; null when it can.
  char const* error = nullptr;

  bool IsSupported() const {
    return error == nullptr && !positions.empty() && !indices.empty();
  }
};

// Identifies the bytes of a GLB file, so ReadDeformableGlb can confirm it reads the file that was
// loaded.
size_t HashGlbBytes(mochi::Span<uint8_t const> bytes);

// Reads the deformable primitive of the binary GLB at @p path, provided the file's bytes still
// hash to @p expectedHash. The GLB must embed its buffers, so those bytes are all it reads.
DeformableSourceData ReadDeformableGlb(char const* path, size_t expectedHash);

} // namespace mochi_renderer
