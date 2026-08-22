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

//
// Shared batched material infrastructure: generic gather utilities and explicit instantiation
// macros. Material-model agnostic — no material-specific types live here.
//
// Per-material headers (batched_linear_elastic.h, batched_arap.h, etc.) include this.
// Consumers should include per-material headers or batched_materials.h, not this file directly.
//

#pragma once

#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

#include <type_traits>

namespace mochi::materials::details {

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE bool IsHomogeneous(T const& perElem) {
  MOCHI_ASSERT_VERBOSE(perElem.size() > 0, "Invalid size.");
  return perElem.size() == 1;
}

[[nodiscard]] MOCHI_FORCE_INLINE int GetPerElementParamsIndex(int numParams, int elementIndex) {
  MOCHI_ASSERT_VERBOSE(numParams > 0, "Per-element material params must not be empty.");
  MOCHI_ASSERT_VERBOSE(elementIndex >= 0, "Element index must be non-negative.");
  MOCHI_ASSERT_VERBOSE(
      numParams == 1 || elementIndex < numParams,
      "Element index out of range for per-element material params.");
  return (numParams == 1) ? 0 : elementIndex;
}

template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchReal<kBatchSize> GatherIndexed(
    Span<real const> data,
    NdArray<int, kBatchSize> const& indices) {
  using V = BatchReal<kBatchSize>;
  alignas(alignof(V)) real staging[V::kSize]{};
  for (int b = 0; b < kBatchSize; ++b) {
    staging[b] = data[indices[b]];
  }
  return Load<V>(staging);
}

template <int kBatchSize, typename Fn>
  requires(std::is_invocable_v<Fn const&, int>)
[[nodiscard]] MOCHI_FORCE_INLINE BatchReal<kBatchSize> GatherIndexed(
    Fn const& accessor,
    NdArray<int, kBatchSize> const& indices) {
  using V = BatchReal<kBatchSize>;
  alignas(alignof(V)) real staging[V::kSize]{};
  for (int b = 0; b < kBatchSize; ++b) {
    staging[b] = accessor(indices[b]);
  }
  return Load<V>(staging);
}

} // namespace mochi::materials::details

// Explicit template instantiation macro for batched constitutive response functions.
// Output pointers are optional: pass nullptr to skip strain energy Ψ(F), first Piola-Kirchhoff
// stress ∂Ψ/∂F, or material tangent ∂²Ψ/∂F². The projectPsd flag requests material-specific PSD
// projection of the tangent when supported.
#if MOCHI_HAS_VA_OPT
#define MOCHI_INSTANTIATE_BATCHED_MATERIAL_HELPER(FuncName, ParamsType, BatchSize, ...) \
  template void FuncName<BatchSize>(                                                    \
      ParamsType<BatchSize> const&,                                                     \
      ::mochi::BatchReal3x3<BatchSize> const&,                                          \
      ::mochi::BatchDouble<BatchSize>*,                                                 \
      ::mochi::BatchReal3x3<BatchSize>*,                                                \
      ::mochi::NdArray<::mochi::BatchReal3x3<BatchSize>, 3, 3>*,                        \
      bool __VA_OPT__(, ) __VA_ARGS__)

#define MOCHI_INSTANTIATE_BATCHED_MATERIAL(FuncName, ParamsType, ...)                            \
  MOCHI_INSTANTIATE_BATCHED_MATERIAL_HELPER(FuncName, ParamsType, 1 __VA_OPT__(, ) __VA_ARGS__); \
  MOCHI_INSTANTIATE_BATCHED_MATERIAL_HELPER(FuncName, ParamsType, 4 __VA_OPT__(, ) __VA_ARGS__); \
  MOCHI_INSTANTIATE_BATCHED_MATERIAL_HELPER(FuncName, ParamsType, 8 __VA_OPT__(, ) __VA_ARGS__)
#else
#define MOCHI_INSTANTIATE_BATCHED_MATERIAL_HELPER(FuncName, ParamsType, BatchSize, ...) \
  template void FuncName<BatchSize>(                                                    \
      ParamsType<BatchSize> const&,                                                     \
      ::mochi::BatchReal3x3<BatchSize> const&,                                          \
      ::mochi::BatchDouble<BatchSize>*,                                                 \
      ::mochi::BatchReal3x3<BatchSize>*,                                                \
      ::mochi::NdArray<::mochi::BatchReal3x3<BatchSize>, 3, 3>*,                        \
      bool,                                                                             \
      ##__VA_ARGS__)

#define MOCHI_INSTANTIATE_BATCHED_MATERIAL(FuncName, ParamsType, ...)                \
  MOCHI_INSTANTIATE_BATCHED_MATERIAL_HELPER(FuncName, ParamsType, 1, ##__VA_ARGS__); \
  MOCHI_INSTANTIATE_BATCHED_MATERIAL_HELPER(FuncName, ParamsType, 4, ##__VA_ARGS__); \
  MOCHI_INSTANTIATE_BATCHED_MATERIAL_HELPER(FuncName, ParamsType, 8, ##__VA_ARGS__)
#endif // MOCHI_HAS_VA_OPT
