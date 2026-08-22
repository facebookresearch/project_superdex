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
#include <mochi_core/utils/simd.h>

namespace mochi {

/**
 * @brief Default batch size for batched FEM operations.
 * @note This is an overridable default. Tune the batch size per kernel, e.g., based on register
 * pressure.
 */
// TODO(T264957520):
// - Tune the heuristic, e.g., the current cap underutilizes AVX-512.
// - Consider sub-batching to a narrower width for high-register-pressure codepaths (e.g.,
//   constitutive response with 81-entry tangent, stress dresidual contraction with 144-entry
//   element matrix).
inline constexpr int kDefaultFemBatchSize = Min(2 * Simd<real>::kSize, 8);

} // namespace mochi
