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

#include <mochi_renderer/image_io.h>

// Filament's imageio PNG encoder only supports 1- or 3-channel output (no
// alpha), so it cannot emit the transparent thumbnails this helper is for.
// stb_image_write handles RGBA and ships in the same third-party target that
// already provides the <stb_image.h> used elsewhere in the renderer, so no
// build change is needed. STB_IMAGE_WRITE_STATIC gives the implementation
// internal linkage so it cannot clash with another translation unit that also
// compiles stb_image_write (e.g. imgui_mcp_bridge in the studio app).
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace mochi_renderer {

void WritePng(
    mochi::Path const& path,
    int width,
    int height,
    int channels,
    mochi::Span<uint8_t const> pixels,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);

  MOCHI_ERROR_IF(width <= 0 || height <= 0, error, "PNG dimensions must be positive.");
  MOCHI_ERROR_IF(
      channels != 1 && channels != 3 && channels != 4, error, "PNG channels must be 1, 3 or 4.");
  MOCHI_ERROR_RETURN(error);

  size_t const expected = static_cast<size_t>(width) * height * channels;
  MOCHI_ERROR_IF(
      pixels.size() != expected,
      error,
      "PNG pixel buffer size does not match width*height*channels.");
  MOCHI_ERROR_RETURN(error);

  int const strideBytes = width * channels;
  int const ok =
      stbi_write_png(path.ToString().c_str(), width, height, channels, pixels.data(), strideBytes);
  MOCHI_ERROR_IF(ok == 0, error, "Failed to write PNG file.");
}

} // namespace mochi_renderer
