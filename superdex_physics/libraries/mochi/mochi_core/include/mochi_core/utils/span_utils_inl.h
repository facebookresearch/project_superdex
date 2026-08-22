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

// Reverse include for intellisense
#include "span_utils.h"

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/simd.h>

#include <type_traits>

namespace mochi {

template <typename ToScalar, typename FromScalar, typename ToSz, typename FromSz>
void StaticCast(Span<FromScalar const, FromSz> src, Span<ToScalar, ToSz> dst) {
  static_assert(!std::is_const_v<ToScalar>, "Destination type must be non-const");
  MOCHI_ASSERT_VERBOSE(dst.size() == src.size(), "Incompatible sizes");

  // Fast path: types are identical, just copy
  if constexpr (std::is_same_v<std::remove_const_t<FromScalar>, ToScalar>) {
    std::copy(src.begin(), src.end(), dst.begin());
    return;
  }

  constexpr int kSize = Max(Simd<std::remove_const_t<FromScalar>>::kSize, Simd<ToScalar>::kSize);
  using VecSrc = Simd<std::remove_const_t<FromScalar>, kSize>;
  using VecDst = Simd<ToScalar, kSize>;
  std::remove_const_t<FromSz> i = 0;
  if constexpr (VecSrc::kIsSupported && VecDst::kIsSupported) {
    for (; i + kSize <= src.size(); i += kSize) {
      Store(&dst[i], StaticCast<VecDst>(Load<VecSrc>(&src[i])));
    }
  }
  for (; i < src.size(); ++i) {
    dst[i] = static_cast<ToScalar>(src[i]);
  }
}

} // namespace mochi
