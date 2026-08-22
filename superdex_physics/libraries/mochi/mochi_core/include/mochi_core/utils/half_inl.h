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

#include "half.h" // For Intellisense

namespace mochi {

MOCHI_ANY MOCHI_FORCE_INLINE Half::Half(float f) {
#if MOCHI_COMPILER_CUDA
  data = __float2half(f);
#elif defined(__FLT16_MAX__)
  data = static_cast<_Float16>(f);
#elif MOCHI_ARCH_X64_AVX2
  data = static_cast<uint16_t>(
      _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0));
#else
  data = FloatToHalfBits(f);
#endif
}

MOCHI_ANY MOCHI_FORCE_INLINE Half::operator float() const {
#if MOCHI_COMPILER_CUDA
  return __half2float(data);
#elif defined(__FLT16_MAX__)
  return static_cast<float>(data);
#elif MOCHI_ARCH_X64_AVX2
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(data)));
#else
  return HalfBitsToFloat(data);
#endif
}

MOCHI_ANY MOCHI_FORCE_INLINE bool Half::operator==(Half const& rhs) const {
#if MOCHI_COMPILER_MSVC
  // Work-around for MSVC optimizer bug. When it sees the same value on both side, it skips the work
  // and returns true. This is wrong for (NaN == NaN).
  float const lf = static_cast<float>(*this);
  float const rf = static_cast<float>(rhs);
  uint32_t lb MOCHI_NO_INIT;
  uint32_t rb MOCHI_NO_INIT;
  memcpy(&lb, &lf, sizeof(lb));
  memcpy(&rb, &rf, sizeof(rb));
  if ((lb & 0x7FFFFFFFu) > 0x7F800000u || (rb & 0x7FFFFFFFu) > 0x7F800000u) {
    return false; // NaN is never equal
  }
#endif
  return static_cast<float>(*this) == static_cast<float>(rhs);
}

inline float HalfBitsToFloat(uint16_t h) {
  uint32_t const sign = (uint32_t(h) >> 15) & 1;
  uint32_t const exponent = (uint32_t(h) >> 10) & 0x1F;
  uint32_t const mantissa = uint32_t(h) & 0x3FF;

  uint32_t result{};

  if (exponent == 0) {
    if (mantissa == 0) {
      // ±zero
      result = sign << 31;
    } else {
      // Denormal: normalize it
      uint32_t m = mantissa;
      int e = 0;
      while ((m & 0x400) == 0) {
        m <<= 1;
        ++e;
      }
      m &= 0x3FF; // Remove the leading 1 bit
      result = (sign << 31) | (uint32_t(127 - 14 - e) << 23) | (m << 13);
    }
  } else if (exponent == 0x1F) {
    // Inf or NaN — preserve payload bits
    result = (sign << 31) | (0xFFu << 23) | (mantissa << 13);
  } else {
    // Normal
    result = (sign << 31) | (uint32_t(exponent - 15 + 127) << 23) | (mantissa << 13);
  }

  float floatResult;
  memcpy(&floatResult, &result, sizeof(floatResult));
  return floatResult;
}

inline uint16_t FloatToHalfBits(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(bits));

  uint32_t const sign = (bits >> 31) & 1;
  int32_t const exponent = int32_t((bits >> 23) & 0xFF) - 127;
  uint32_t const mantissa = bits & 0x7FFFFF;

  uint16_t result{};

  if (exponent > 15) {
    if (exponent == 128 && mantissa != 0) {
      // NaN — preserve some payload bits
      result = uint16_t((sign << 15) | (0x1F << 10) | (mantissa >> 13));
      if ((result & 0x3FF) == 0) {
        result |= 1; // Ensure NaN payload is nonzero
      }
    } else {
      // Overflow or Inf → ±inf
      result = uint16_t((sign << 15) | (0x1F << 10));
    }
  } else if (exponent < -25) {
    // Too small even for denormal → ±zero
    result = uint16_t(sign << 15);
  } else if (exponent < -14) {
    // Denormal range
    // Add the implicit 1 bit, then shift right to denormal position
    uint32_t const full = mantissa | 0x800000;
    auto const shift = uint32_t(-1 - exponent);
    uint32_t m = full >> shift;
    // Round-to-nearest-even using guard bit (highest truncated) and sticky bits (remaining)
    uint32_t const guard = (full >> (shift - 1)) & 1;
    uint32_t const sticky = full & ((1u << (shift - 1)) - 1);
    if (guard != 0 && (sticky != 0 || (m & 1) != 0)) {
      ++m;
    }
    result = uint16_t((sign << 15) | m);
  } else {
    // Normal range
    auto halfExp = uint32_t(exponent + 15);
    uint32_t halfMantissa = mantissa >> 13;
    // Round-to-nearest-even: check the bits being truncated
    uint32_t const truncated = mantissa & 0x1FFF;
    if (truncated > 0x1000) {
      ++halfMantissa;
    } else if (truncated == 0x1000) {
      // Tie: round to even
      halfMantissa += (halfMantissa & 1);
    }
    // Handle mantissa overflow (carry into exponent)
    if (halfMantissa > 0x3FF) {
      halfMantissa = 0;
      ++halfExp;
      if (halfExp >= 0x1F) {
        // Overflow to infinity
        result = uint16_t((sign << 15) | (0x1F << 10));
        return result;
      }
    }
    result = uint16_t((sign << 15) | (halfExp << 10) | halfMantissa);
  }

  return result;
}

} // namespace mochi
