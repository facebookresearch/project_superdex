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
#include <mochi_core/materials/kim_neo_hookean_params.h>
#include <mochi_core/materials/material_traits.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/batch_types.h>

#include <type_traits>

/**
 * Stable neo-Hookean material model based on Kim and Eberle (2022).
 *
 * Energy formulation:
 *   Ψ(F) = (μ/2)(‖F‖² - 3) - μ(J - 1) + (λ̂/2)(J - 1)²
 *
 * Where:
 *   - F is the deformation gradient
 *   - ‖F‖² = tr(F^T F) is the first invariant of the right Cauchy-Green tensor
 *   - J = det(F) is the determinant of the deformation gradient
 *   - μ, λ are the standard Lamé parameters
 *   - λ̂ = λ + μ is a modified first Lamé parameter
 *
 * Reparameterization:
 *   The reference formulation uses λ directly in the volumetric term. In the small deformation
 *   limit (F → I), this yields an effective first Lamé parameter of (λ - μ) instead of λ, so
 *   the material does not reduce to linear elasticity. Replacing λ with λ̂ = λ + μ ensures the
 *   tangent at F = I matches the isotropic linear elastic stiffness tensor:
 *
 *     C_ijkl = λ δ_ij δ_kl + μ (δ_ik δ_jl + δ_il δ_jk)
 *
 * Notes:
 *   - Similar to Smith neo-Hookean but without the J = 0 barrier protection.
 *   - This is the preferred stable neo-Hookean material model in Pixar. The barrier protection in
 *     Smith neo-Hookean was to please reviewers (see Footnote 9 in Chapter 6 of Kim and Eberle,
 *     2022).
 *
 * Reference: Eq. (6.9) in Kim and Eberle, "Dynamic Deformables", 2022
 * (https://www.tkim.graphics/DYNAMIC_DEFORMABLES/DynamicDeformables.pdf)
 */
namespace mochi::materials {

template <>
[[nodiscard]] constexpr bool IsPsdStrategySupported<KimNeoHookeanMaterialParams>(
    MaterialPsdStrategy s) {
  return s == MaterialPsdStrategy::MaterialDefault || s == MaterialPsdStrategy::None ||
      s == MaterialPsdStrategy::Projection || s == MaterialPsdStrategy::Fast ||
      s == MaterialPsdStrategy::AbsEigenProjection;
}

template <>
inline constexpr bool kIsLameMaterial<KimNeoHookeanMaterialParams> = true;

/// @brief Batched constitutive response declaration.
///
/// @warning `params.psdStrategy` must be resolved. It must not be @ref
/// MaterialPsdStrategy::MaterialDefault.
template <int kBatchSize>
void BatchedKimNeoHookeanConstitutiveResponse(
    BatchLameParams<kBatchSize> const&,
    BatchReal3x3<kBatchSize> const&,
    BatchDouble<kBatchSize>*,
    BatchReal3x3<kBatchSize>*,
    NdArray<BatchReal3x3<kBatchSize>, 3, 3>*,
    bool);

/// @brief Heterogeneous factory for batched Kim Neo-Hookean constitutive response.
///
/// @warning The returned lambda captures @p perElem by reference; @p perElem must outlive the
/// returned lambda.
template <typename ParamsT, int kBatchSize = kDefaultFemBatchSize>
  requires(std::is_same_v<ParamsT, KimNeoHookeanMaterialParams>)
[[nodiscard]] auto MakeBatchedConstitutiveResponse(PerElementLameParams const& perElem) {
  return [&perElem](
             NdArray<int, kBatchSize> const& elementIndices,
             auto const& F,
             auto* e,
             auto* pk1,
             auto* tangent,
             bool psd) MOCHI_FORCE_INLINE_LAMBDA {
    auto const lame = details::GatherLameParams<kBatchSize>(perElem, elementIndices);
    BatchedKimNeoHookeanConstitutiveResponse<kBatchSize>(lame, F, e, pk1, tangent, psd);
  };
}

} // namespace mochi::materials
