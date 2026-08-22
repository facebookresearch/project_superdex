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

#include <mochi_renderer/path.h>

#include <mochi_core/utils/error.h>
#include <mochi_core/utils/span.h>

#include <cstdint>

namespace mochi_renderer {

/// @brief Encodes tightly-packed 8-bit pixels to a PNG file on disk.
///
/// @param[in] path Destination file; any existing file is overwritten.
/// @param[in] width Image width [px]; must be > 0.
/// @param[in] height Image height [px]; must be > 0.
/// @param[in] channels Components per pixel: 1 (grayscale), 3 (RGB) or 4 (RGBA).
/// @param[in] pixels Row-major, top-left origin, tightly packed (no row
///   padding); must hold exactly @p width * @p height * @p channels bytes.
/// @param[out] error Set on invalid arguments or write failure.
///
/// @note The alpha channel is preserved when @p channels == 4, so fully
///   transparent backgrounds round-trip to the file.
void WritePng(
    mochi::Path const& path,
    int width,
    int height,
    int channels,
    mochi::Span<uint8_t const> pixels,
    mochi::Error& error);

} // namespace mochi_renderer
