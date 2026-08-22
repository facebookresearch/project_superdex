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

#include "basic_utils.h"

#include <limits>

namespace mochi {

template <typename T>
MOCHI_FORCE_INLINE constexpr T SignedSqrt(T a) {
  auto sqrt = Sqrt(Abs(a));
  return Select(a >= T{}, sqrt, -sqrt);
}

template <typename T>
MOCHI_FORCE_INLINE constexpr T IntegralSqrt(T a) {
  static_assert(
      std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool>, "Unsupported type");

  // Base case.
  if (a < 2)
    MOCHI_UNLIKELY {
      return a;
    }

  // Find square root using binary search.
  T lo = T(1);
  T hi = T(a / 2 + 1);
  T ans = T(0);

  while (lo <= hi) {
    T md = (lo + hi + 1) / 2;

    if (md <= a / md) {
      if (md * md == a) {
        return md;
      }
      lo = md + 1;
      ans = md;
    } else {
      hi = md - 1;
    }
  }

  return ans;
}

MOCHI_WARNING_PUSH()
MOCHI_WARNING_IGNORE_MSVC(4723) // warning C4723: potential divide by 0

template <typename T>
MOCHI_FORCE_INLINE constexpr T Sinc(T x) {
  T x2 = x * x;
  T x4 = x2 * x2;

  if (x4 < std::numeric_limits<T>::epsilon())
    MOCHI_UNLIKELY {
      return T(1) - x2 / T(6);
    }
  else {
    return Sin(x) / x;
  }
}

MOCHI_WARNING_POP()

} // namespace mochi
