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

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/no_copy.h>
#include <mochi_core/utils/span.h>

#include <algorithm>

namespace mochi {

/**
 * @brief Helper class for identifying indices of a src span missing in a dst span.
 *
 * @details @ref GetMissingSamples provides a unified entry point that dispatches based on whether
 * feature arrays are provided. When feature arrays are empty, it matches by sample index only. When
 * feature arrays are provided, it matches by (sample, feature) pairs. All instances of the
 * same sample must appear consecutively in the input arrays.
 */
class ContactCorrespondence : public NoCopy {
 public:
  // Pair of colliding sample (first) and contact index (second)
  using Pair = std::pair<int, int>;

  ContactCorrespondence() = default;

  explicit ContactCorrespondence(int numSamples) : _numSamples(numSamples) {
    // Preallocate memory for arrays used with all collider types.
    _stamp.resize_noinit(_numSamples);
    _missing.reserve(_numSamples);
    std::fill(_stamp.begin(), _stamp.end(), 0);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Span<Pair const> GetMissingSamples(
      Span<int const> srcSamples,
      Span<int const> srcFeatures,
      Span<int const> dstSamples,
      Span<int const> dstFeatures) {
    return (srcFeatures.empty() && dstFeatures.empty())
        ? GetMissingSamplesImpl(srcSamples, dstSamples)
        : GetMissingSamplesImpl(srcSamples, srcFeatures, dstSamples, dstFeatures);
  }

 protected:
  void Clear();
  [[nodiscard]] Span<Pair const> GetMissingSamplesImpl(
      Span<int const> srcSamples,
      Span<int const> dstSamples);
  [[nodiscard]] Span<Pair const> GetMissingSamplesImpl(
      Span<int const> srcSamples,
      Span<int const> srcFeatures,
      Span<int const> dstSamples,
      Span<int const> dstFeatures);

  // Maximum number of samples that can be registered.
  int _numSamples = 0;

  // Array containing, for each colliding sample, the last epoch in which it was registered.
  DynamicArray<int> _stamp;

  // Array containing the missing samples for the last call to GetMissingSamples.
  DynamicArray<Pair> _missing;

  // [Colliders with feature indices only] Array containing, for each colliding sample, the start
  // index in dst arrays.
  DynamicArray<int> _startIdx;

  // [Colliders with feature indices only] Array containing, for each colliding sample, the number
  // of contacts in dst.
  DynamicArray<int> _count;

  // Last epoch executed.
  int _epoch = 0;
};

} // namespace mochi
