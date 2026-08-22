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

#include <mochi_core/materials/arap_params.h>
#include <mochi_core/materials/batched_material_infra.h>
#include <mochi_core/materials/material_traits.h>
#include <mochi_core/materials/material_utils.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/batch_types.h>

#include <type_traits>

/**
 * As-Rigid-As-Possible (ARAP) material model.
 *
 * Reference: Kim and Eberle, "Dynamic Deformables", 2022
 * (https://www.tkim.graphics/DYNAMIC_DEFORMABLES/DynamicDeformables.pdf)
 */
namespace mochi::materials {

template <>
[[nodiscard]] constexpr bool IsPsdStrategySupported<ArapMaterialParams>(MaterialPsdStrategy s) {
  return s == MaterialPsdStrategy::MaterialDefault || s == MaterialPsdStrategy::None ||
      s == MaterialPsdStrategy::Projection || s == MaterialPsdStrategy::AbsEigenProjection;
}

// ---------------------------------------------------------------------------
// Per-element ARAP params
// ---------------------------------------------------------------------------

/// @brief Per-element ARAP material params.
/// @see ArapMaterialParams
struct PerElementArapParams {
  DynamicArray<real> stiffness; ///< ARAP stiffness (μ) [Pa], one per element.
  MaterialPsdStrategy psdStrategy = MaterialPsdStrategy::MaterialDefault;

  [[nodiscard]] auto size() const {
    return stiffness.size();
  }
};

/// @brief Batch ARAP material params.
/// @see ArapMaterialParams
template <int kBatchSize>
struct BatchArapParams {
  using V = BatchReal<kBatchSize>;
  V stiffness{};
  MaterialPsdStrategy psdStrategy = MaterialPsdStrategy::MaterialDefault;
};

/// @brief ArapMaterialParams → PerElementArapParams (homogeneous)
[[nodiscard]] inline PerElementArapParams BuildPerElementParams(ArapMaterialParams const& p) {
  MOCHI_ASSERT_VERBOSE(IsFinite(p.stiffness) && p.stiffness > 0, "Invalid ARAP stiffness.");
  return {.stiffness = {p.stiffness}, .psdStrategy = utils::ResolvePsdStrategy(p)};
}

/// @brief PerElementArapParams → ArapMaterialParams for one element.
template <typename ParamsT>
  requires(std::is_same_v<ParamsT, ArapMaterialParams>)
[[nodiscard]] inline ParamsT GetElementParams(
    PerElementArapParams const& perElementParams,
    int elementIndex) {
  int const i = details::GetPerElementParamsIndex(isize(perElementParams), elementIndex);
  return {.stiffness = perElementParams.stiffness[i], .psdStrategy = perElementParams.psdStrategy};
}

/// @brief ArapMaterialParams → BatchArapParams.
template <int kBatchSize>
[[nodiscard]] BatchArapParams<kBatchSize> BuildBatchParams(ArapMaterialParams const& p) {
  MOCHI_ASSERT_VERBOSE(IsFinite(p.stiffness) && p.stiffness > 0, "Invalid ARAP stiffness.");
  return {.stiffness = p.stiffness, .psdStrategy = utils::ResolvePsdStrategy(p)};
}

// ---------------------------------------------------------------------------
// Batched constitutive response
// ---------------------------------------------------------------------------

/// @brief Batched constitutive response declaration.
///
/// @warning `params.psdStrategy` must be resolved. It must not be @ref
/// MaterialPsdStrategy::MaterialDefault.
template <int kBatchSize>
void BatchedArapConstitutiveResponse(
    BatchArapParams<kBatchSize> const&,
    BatchReal3x3<kBatchSize> const&,
    BatchDouble<kBatchSize>*,
    BatchReal3x3<kBatchSize>*,
    NdArray<BatchReal3x3<kBatchSize>, 3, 3>*,
    bool);

/// @brief Heterogeneous factory for batched ARAP constitutive response.
///
/// @warning The returned lambda captures @p perElem by reference; @p perElem must outlive the
/// returned lambda.
template <typename /*ParamsT*/, int kBatchSize = kDefaultFemBatchSize>
[[nodiscard]] auto MakeBatchedConstitutiveResponse(PerElementArapParams const& perElem) {
  return [&perElem](
             NdArray<int, kBatchSize> const& elementIndices,
             auto const& F,
             auto* e,
             auto* pk1,
             auto* tangent,
             bool psd) MOCHI_FORCE_INLINE_LAMBDA {
    BatchArapParams<kBatchSize> params;
    if (details::IsHomogeneous(perElem)) {
      params = {.stiffness = perElem.stiffness[0], .psdStrategy = perElem.psdStrategy};
    } else {
      params = {
          .stiffness = details::GatherIndexed<kBatchSize>(perElem.stiffness, elementIndices),
          .psdStrategy = perElem.psdStrategy};
    }
    BatchedArapConstitutiveResponse<kBatchSize>(params, F, e, pk1, tangent, psd);
  };
}

} // namespace mochi::materials
