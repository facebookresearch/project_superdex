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

#include <mochi_core/materials/active_aniso_arap_params.h>
#include <mochi_core/materials/batched_material_infra.h>
#include <mochi_core/materials/material_traits.h>
#include <mochi_core/materials/material_utils.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/batch_types.h>

#include <type_traits>

/**
 * Active anisotropic ARAP material model by Kim et al. (2019).
 *
 * Reference: Kim et al., "Anisotropic Elasticity for Inversion-Safety and Element Rehabilitation",
 * 2019 (http://tkim.graphics/ANISOTROPY/AnisotropyAndRehab.pdf)
 */
namespace mochi::materials {

template <>
[[nodiscard]] constexpr bool IsPsdStrategySupported<ActiveAnisoArapMaterialParams>(
    MaterialPsdStrategy s) {
  return s == MaterialPsdStrategy::MaterialDefault || s == MaterialPsdStrategy::None ||
      s == MaterialPsdStrategy::Projection || s == MaterialPsdStrategy::AbsEigenProjection;
}

// ---------------------------------------------------------------------------
// Per-element Active Aniso ARAP params
// ---------------------------------------------------------------------------

/// @brief Per-element Active Anisotropic ARAP params.
/// @see ActiveAnisoArapMaterialParams
struct PerElementActiveAnisoArapParams {
  DynamicArray<real> alpha; ///< Anisotropic stiffness (α) along fiber direction [Pa].
  DynamicArray<real> length; ///< Anisotropic reference length (dimensionless).
  DynamicArray<Real3> anisoDir; ///< Fiber direction as a unit vector.
  MaterialPsdStrategy psdStrategy = MaterialPsdStrategy::MaterialDefault;

  [[nodiscard]] auto size() const {
    MOCHI_ASSERT_VERBOSE(
        alpha.size() == length.size() && alpha.size() == anisoDir.size(), "Inconsistent sizes.");
    return alpha.size();
  }
};

/// @brief Batch Active Aniso ARAP params.
/// @see ActiveAnisoArapMaterialParams
template <int kBatchSize>
struct BatchActiveAnisoArapParams {
  using V = BatchReal<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  V alpha{};
  V length{};
  V3 anisoDir{};
  MaterialPsdStrategy psdStrategy = MaterialPsdStrategy::MaterialDefault;
};

/// @brief ActiveAnisoArapMaterialParams → PerElementActiveAnisoArapParams
[[nodiscard]] inline PerElementActiveAnisoArapParams BuildPerElementParams(
    ActiveAnisoArapMaterialParams const& p) {
  MOCHI_ASSERT_VERBOSE(p.alpha >= 0 && p.length >= 0, "Invalid active aniso ARAP parameters.");
  return {
      .alpha = {p.alpha},
      .length = {p.length},
      .anisoDir = {p.anisoDir},
      .psdStrategy = utils::ResolvePsdStrategy(p)};
}

/// @brief PerElementActiveAnisoArapParams → ActiveAnisoArapMaterialParams for one element.
template <typename ParamsT>
  requires(std::is_same_v<ParamsT, ActiveAnisoArapMaterialParams>)
[[nodiscard]] inline ParamsT GetElementParams(
    PerElementActiveAnisoArapParams const& perElementParams,
    int elementIndex) {
  int const i = details::GetPerElementParamsIndex(isize(perElementParams), elementIndex);
  return {
      .alpha = perElementParams.alpha[i],
      .length = perElementParams.length[i],
      .anisoDir = perElementParams.anisoDir[i],
      .psdStrategy = perElementParams.psdStrategy};
}

/// @brief ActiveAnisoArapMaterialParams → BatchActiveAnisoArapParams.
template <int kBatchSize>
[[nodiscard]] BatchActiveAnisoArapParams<kBatchSize> BuildBatchParams(
    ActiveAnisoArapMaterialParams const& p) {
  MOCHI_ASSERT_VERBOSE(p.alpha >= 0 && p.length >= 0, "Invalid active aniso ARAP parameters.");
  using V = BatchReal<kBatchSize>;
  return {
      .alpha = p.alpha,
      .length = p.length,
      .anisoDir = {V{p.anisoDir[0]}, V{p.anisoDir[1]}, V{p.anisoDir[2]}},
      .psdStrategy = utils::ResolvePsdStrategy(p)};
}

// ---------------------------------------------------------------------------
// Implementation details
// ---------------------------------------------------------------------------

namespace details {

template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchActiveAnisoArapParams<kBatchSize> GatherActiveAnisoParams(
    PerElementActiveAnisoArapParams const& pe,
    NdArray<int, kBatchSize> const& indices) {
  using V = BatchReal<kBatchSize>;
  if (IsHomogeneous(pe)) {
    return {
        .alpha = pe.alpha[0],
        .length = pe.length[0],
        .anisoDir = {V{pe.anisoDir[0][0]}, V{pe.anisoDir[0][1]}, V{pe.anisoDir[0][2]}},
        .psdStrategy = pe.psdStrategy};
  } else {
    return {
        .alpha = details::GatherIndexed<kBatchSize>(pe.alpha, indices),
        .length = details::GatherIndexed<kBatchSize>(pe.length, indices),
        .anisoDir =
            {details::GatherIndexed<kBatchSize>([&](int i) { return pe.anisoDir[i][0]; }, indices),
             details::GatherIndexed<kBatchSize>([&](int i) { return pe.anisoDir[i][1]; }, indices),
             details::GatherIndexed<kBatchSize>([&](int i) { return pe.anisoDir[i][2]; }, indices)},
        .psdStrategy = pe.psdStrategy};
  }
}

} // namespace details

// ---------------------------------------------------------------------------
// Batched constitutive response
// ---------------------------------------------------------------------------

/// @brief Batched constitutive response declaration.
///
/// @warning `params.psdStrategy` must be resolved. It must not be @ref
/// MaterialPsdStrategy::MaterialDefault.
template <int kBatchSize>
void BatchedActiveAnisoArapConstitutiveResponse(
    BatchActiveAnisoArapParams<kBatchSize> const&,
    BatchReal3x3<kBatchSize> const&,
    BatchDouble<kBatchSize>*,
    BatchReal3x3<kBatchSize>*,
    NdArray<BatchReal3x3<kBatchSize>, 3, 3>*,
    bool);

/// @brief Heterogeneous factory for batched Active Aniso ARAP constitutive response.
///
/// @warning The returned lambda captures @p perElem by reference; @p perElem must outlive the
/// returned lambda.
template <typename /*ParamsT*/, int kBatchSize = kDefaultFemBatchSize>
[[nodiscard]] auto MakeBatchedConstitutiveResponse(PerElementActiveAnisoArapParams const& perElem) {
  return [&perElem](
             NdArray<int, kBatchSize> const& elementIndices,
             auto const& F,
             auto* e,
             auto* pk1,
             auto* tangent,
             bool psd) MOCHI_FORCE_INLINE_LAMBDA {
    auto const params = details::GatherActiveAnisoParams<kBatchSize>(perElem, elementIndices);
    BatchedActiveAnisoArapConstitutiveResponse<kBatchSize>(params, F, e, pk1, tangent, psd);
  };
}

} // namespace mochi::materials
