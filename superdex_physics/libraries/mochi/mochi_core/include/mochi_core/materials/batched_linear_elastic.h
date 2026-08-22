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

#include <mochi_core/materials/batched_lame_params.h>
#include <mochi_core/materials/linear_elastic_params.h>
#include <mochi_core/materials/material_traits.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/batch_types.h>

#include <type_traits>

namespace mochi::materials {

template <>
[[nodiscard]] constexpr bool IsPsdStrategySupported<LinearElasticMaterialParams>(
    MaterialPsdStrategy) {
  return true;
}

template <>
inline constexpr bool kIsLameMaterial<LinearElasticMaterialParams> = true;

/// @brief Batched constitutive response declaration.
template <int kBatchSize>
void BatchedLinearElasticConstitutiveResponse(
    BatchLameParams<kBatchSize> const&,
    BatchReal3x3<kBatchSize> const&,
    BatchDouble<kBatchSize>*,
    BatchReal3x3<kBatchSize>*,
    NdArray<BatchReal3x3<kBatchSize>, 3, 3>*,
    bool);

/// @brief Heterogeneous factory for batched Linear Elastic constitutive response.
///
/// @warning The returned lambda captures @p perElem by reference; @p perElem must outlive the
/// returned lambda.
template <typename ParamsT, int kBatchSize = kDefaultFemBatchSize>
  requires(std::is_same_v<ParamsT, LinearElasticMaterialParams>)
[[nodiscard]] auto MakeBatchedConstitutiveResponse(PerElementLameParams const& perElem) {
  return [&perElem](
             NdArray<int, kBatchSize> const& elementIndices,
             auto const& F,
             auto* e,
             auto* pk1,
             auto* tangent,
             bool psd) MOCHI_FORCE_INLINE_LAMBDA {
    auto const lame = details::GatherLameParams<kBatchSize>(perElem, elementIndices);
    BatchedLinearElasticConstitutiveResponse<kBatchSize>(lame, F, e, pk1, tangent, psd);
  };
}

} // namespace mochi::materials
