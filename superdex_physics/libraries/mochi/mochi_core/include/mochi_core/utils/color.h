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

// PLEASE DO NOT ADD OTHER INCLUDES HERE. This header is included in the mochi_physics public API.
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/reflection.h>

namespace mochi {

/**************************************************************************************************
  Color RGBA as 4 bytes. Used debug visualization.
*/

/**
 * @brief RGBA color representation using 4 bytes (0-255 per channel) in RGBA order.
 * Used primarily for debug visualization.
 */
using Color = NdArray<uint8_t, 4>;

/**
 * @brief Creates a Color from 4 unsigned bytes in RGBA order. This is a convenience function to
 * relax the requirements of the NdArray constructor (which disallows implicit conversion from int
 * to uint8_t).
 */
[[nodiscard]] constexpr Color MakeColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
  return Color{r, g, b, a};
}

/**
 * @brief Creates a Color from a packed 32-bit RGBA value.
 *
 * @details Unpacks a 32-bit unsigned integer into RGBA components where:
 * - Bits 24-31: Red channel
 * - Bits 16-23: Green channel
 * - Bits 8-15:  Blue channel
 * - Bits 0-7:   Alpha channel
 *
 * @param rgba Packed RGBA value as a 32-bit unsigned integer
 * @return Color object with extracted RGBA components
 */
[[nodiscard]] constexpr Color MakeColor(uint32_t rgba) {
  auto const r = static_cast<uint8_t>(rgba >> 24);
  auto const g = static_cast<uint8_t>(rgba >> 16);
  auto const b = static_cast<uint8_t>(rgba >> 8);
  auto const a = static_cast<uint8_t>(rgba);
  return Color{r, g, b, a};
}

/**
 * @brief Converts a Color to a normalized Float3 vector (RGB only).
 *
 * @details Converts the RGB channels of a Color to a Float3 vector with values normalized to the
 * range [0.0f, 1.0f]. The alpha channel is discarded.
 *
 * @param color Input color to convert
 * @return Float3 vector with normalized RGB values (0.0f - 1.0f)
 */
[[nodiscard]] constexpr Float3 ToFloat3(Color const& color) {
  return Float3{color[0] / 255.0f, color[1] / 255.0f, color[2] / 255.0f};
}

/**
 * @brief Converts a Color to a normalized Float4 vector (RGBA).
 *
 * @details Converts all four channels of a Color to a Float4 vector with values normalized to the
 * range [0.0f, 1.0f].
 *
 * @param color Input color to convert
 * @return Float4 vector with normalized RGBA values (0.0f - 1.0f)
 */
[[nodiscard]] constexpr Float4 ToFloat4(Color const& color) {
  return Float4{color[0] / 255.0f, color[1] / 255.0f, color[2] / 255.0f, color[3] / 255.0f};
}

/**************************************************************************************************/

/**
 * @namespace mochi::colors
 * @brief Predefined color constants for common use cases.
 *
 * @details This namespace contains a collection of commonly used colors in RGBA format. All colors
 * have full opacity (alpha = 255). Use these constants for convenience, but feel free to define
 * custom colors as needed.
 */
namespace colors {
constexpr static Color kBlack = MakeColor(0x000000FF); ///< Pure black color
constexpr static Color kWhite = MakeColor(0xFFFFFFFF); ///< Pure white color
constexpr static Color kRed = MakeColor(0xFF0000FF); ///< Pure red color
constexpr static Color kGreen = MakeColor(0x00FF00FF); ///< Pure green color
constexpr static Color kBlue = MakeColor(0x0000FFFF); ///< Pure blue color
constexpr static Color kYellow = MakeColor(0xFFFF00FF); ///< Yellow color (red + green)
constexpr static Color kCyan = MakeColor(0x00FFFFFF); ///< Cyan color (green + blue)
constexpr static Color kMagenta = MakeColor(0xFF00FFFF); ///< Magenta color (red + blue)
constexpr static Color kSilver = MakeColor(0xC0C0C0FF); ///< Light gray color
constexpr static Color kGray = MakeColor(0x808080FF); ///< Medium gray color (US spelling)
constexpr static Color kGrey = MakeColor(0x808080FF); ///< Medium gray color (UK spelling)
constexpr static Color kMaroon = MakeColor(0x800000FF); ///< Dark red color
constexpr static Color kOlive = MakeColor(0x808000FF); ///< Dark yellow-green color
constexpr static Color kPurple = MakeColor(0x800080FF); ///< Dark magenta color
constexpr static Color kTeal = MakeColor(0x008080FF); ///< Dark cyan color
constexpr static Color kNavy = MakeColor(0x000080FF); ///< Dark blue color
constexpr static Color kOrange = MakeColor(0xFFA500FF); ///< Orange color
} // namespace colors

} // namespace mochi
