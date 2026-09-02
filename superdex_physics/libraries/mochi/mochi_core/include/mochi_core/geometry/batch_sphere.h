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

#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/simd.h>

namespace mochi {

/**
 * @brief Represents a fixed-size collection of spheres for efficient SIMD batch processing.
 *
 * @tparam kBatchSize The number of spheres in a batch. Requires @ref Simd<real,
 * kBatchSize>::kIsSupported.
 */
template <int kBatchSize>
struct BatchSphere {
  static_assert(
      Simd<real, kBatchSize>::kIsSupported,
      "Please use a supported SIMD size for BatchSphere queries");
  static constexpr int kSize = BatchReal<kBatchSize>::kSize; ///< Number of SIMD lanes.
  BatchReal3<kBatchSize> center; ///< Sphere centers [m], stored as X/Y/Z lane vectors.
  BatchReal<kBatchSize> radius; ///< Sphere radii [m], stored per lane.
};

} // namespace mochi
