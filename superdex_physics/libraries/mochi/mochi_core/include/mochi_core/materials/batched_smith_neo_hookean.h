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
#include <mochi_core/materials/material_traits.h>
#include <mochi_core/materials/smith_neo_hookean_params.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/batch_types.h>

#include <type_traits>

/**
 * Stable neo-Hookean material model by Smith et al. (2018).
 *
 * Energy formulation:
 *   Ψ(F) = (μ̂/2)(Ic - 3) + (λ̂/2)(J - α)² - (μ̂/2)log(Ic + 1)
 *
 * Where:
 *   - F is the deformation gradient
 *   - Ic = tr(F^T F) is the first invariant of the right Cauchy-Green tensor
 *   - J = det(F) is the determinant of the deformation gradient
 *   - μ, λ are the standard Lamé parameters
 *   - μ̂ = (4/3)μ, λ̂ = λ + (5/6)μ are reparameterized constants to recover linear elasticity
 *     in the small deformation limit (Sec. 3.4 of reference)
 *   - α = 1 + μ̂/λ̂ - μ̂/(4λ̂) is the rest-stability term
 *
 * Note:
 *   - Similar to Kim neo-Hookean but with logarithmic stabilization to prevent singular behavior as
 *     Ic → 0.
 *   - It robustly handles extreme deformations and element inversions.
 *
 * Reference: Smith et al. "Stable Neo-Hookean Flesh Simulation", 2018
 * (https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf)
 */
namespace mochi::materials {

/**
 * @brief Strategy for determining if it is necessary to project the material's output Hessian to
 * the PSD cone.
 */
enum struct MaterialPsdOracle {
  /**
   * @brief None. Always project to PSD if @ref SmithNeoHookeanMaterialParams::psdStrategy requests
   * projection.
   */
  None,

  /**
   * @brief Evaluate the true indefiniteness condition. Slower but always produces the correct
   * answer.
   */
  Correct,

  /**
   * @brief Approximate the condition to evaluate it faster. Conservative: can project more often
   * than necessary, but does not miss required projections.
   */
  Conservative,

  /** @brief Number of unique enum values. */
  Count,

  /** @brief Default oracle. */
  Default = Correct,
};

template <>
[[nodiscard]] constexpr bool IsPsdStrategySupported<SmithNeoHookeanMaterialParams>(
    MaterialPsdStrategy s) {
  return s == MaterialPsdStrategy::MaterialDefault || s == MaterialPsdStrategy::None ||
      s == MaterialPsdStrategy::Projection || s == MaterialPsdStrategy::Fast ||
      s == MaterialPsdStrategy::AbsEigenProjection;
}

template <>
inline constexpr bool kIsLameMaterial<SmithNeoHookeanMaterialParams> = true;

/// @brief Batched constitutive response declaration.
///
/// @warning `params.psdStrategy` must be resolved. It must not be @ref
/// MaterialPsdStrategy::MaterialDefault.
template <int kBatchSize>
void BatchedSmithNeoHookeanConstitutiveResponse(
    BatchLameParams<kBatchSize> const&,
    BatchReal3x3<kBatchSize> const&,
    BatchDouble<kBatchSize>*,
    BatchReal3x3<kBatchSize>*,
    NdArray<BatchReal3x3<kBatchSize>, 3, 3>*,
    bool,
    MaterialPsdOracle = MaterialPsdOracle::Default);

/// @brief Heterogeneous factory for batched Smith Neo-Hookean constitutive response.
///
/// @warning The returned lambda captures @p perElem by reference; @p perElem must outlive the
/// returned lambda.
template <typename ParamsT, int kBatchSize = kDefaultFemBatchSize>
  requires(std::is_same_v<ParamsT, SmithNeoHookeanMaterialParams>)
[[nodiscard]] auto MakeBatchedConstitutiveResponse(
    PerElementLameParams const& perElem,
    MaterialPsdOracle oracle = MaterialPsdOracle::Default) {
  return [&perElem, oracle](
             NdArray<int, kBatchSize> const& elementIndices,
             auto const& F,
             auto* e,
             auto* pk1,
             auto* tangent,
             bool psd) MOCHI_FORCE_INLINE_LAMBDA {
    auto const lame = details::GatherLameParams<kBatchSize>(perElem, elementIndices);
    BatchedSmithNeoHookeanConstitutiveResponse<kBatchSize>(lame, F, e, pk1, tangent, psd, oracle);
  };
}

} // namespace mochi::materials
