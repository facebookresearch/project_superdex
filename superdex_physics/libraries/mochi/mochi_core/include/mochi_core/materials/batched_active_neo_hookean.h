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

#include <mochi_core/materials/active_neo_hookean_params.h>
#include <mochi_core/materials/batched_active_aniso_arap.h>
#include <mochi_core/materials/batched_lame_params.h>
#include <mochi_core/materials/batched_smith_neo_hookean.h>
#include <mochi_core/materials/material_traits.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/batch_types.h>

#include <type_traits>

namespace mochi::materials {

template <>
[[nodiscard]] constexpr bool IsPsdStrategySupported<ActiveNeoHookeanMaterialParams>(
    MaterialPsdStrategy s) {
  return IsPsdStrategySupported<decltype(ActiveNeoHookeanMaterialParams::passiveIsotropic)>(s) &&
      IsPsdStrategySupported<decltype(ActiveNeoHookeanMaterialParams::activeAnisotropic)>(s);
}

// ---------------------------------------------------------------------------
// Per-element Active Neo-Hookean params
// ---------------------------------------------------------------------------

/// @brief Per-element Active Neo-Hookean params.
/// @see ActiveNeoHookeanMaterialParams
struct PerElementActiveNeoHookeanParams {
  PerElementLameParams lame{};
  PerElementActiveAnisoArapParams aniso{};

  [[nodiscard]] auto size() const {
    MOCHI_ASSERT_VERBOSE(lame.size() == aniso.size(), "Inconsistent sizes.");
    return lame.size();
  }
};

/// @brief Batch Active Neo-Hookean params.
/// @see ActiveNeoHookeanMaterialParams
template <int kBatchSize>
struct BatchActiveNeoHookeanParams {
  BatchLameParams<kBatchSize> lame{};
  BatchActiveAnisoArapParams<kBatchSize> aniso{};
};

/// @brief ActiveNeoHookeanMaterialParams → PerElementActiveNeoHookeanParams (homogeneous).
[[nodiscard]] inline PerElementActiveNeoHookeanParams BuildPerElementParams(
    ActiveNeoHookeanMaterialParams const& p) {
  return {
      .lame = details::BuildPerElementLameParams(p.passiveIsotropic),
      .aniso = BuildPerElementParams(p.activeAnisotropic)};
}

/// @brief PerElementActiveNeoHookeanParams → ActiveNeoHookeanMaterialParams for one element.
template <typename ParamsT>
  requires(std::is_same_v<ParamsT, ActiveNeoHookeanMaterialParams>)
[[nodiscard]] inline ParamsT GetElementParams(
    PerElementActiveNeoHookeanParams const& perElementParams,
    int elementIndex) {
  return {
      .passiveIsotropic = details::GetElementLameParams<SmithNeoHookeanMaterialParams>(
          perElementParams.lame, elementIndex),
      .activeAnisotropic =
          GetElementParams<ActiveAnisoArapMaterialParams>(perElementParams.aniso, elementIndex)};
}

/// @brief ActiveNeoHookeanMaterialParams → BatchActiveNeoHookeanParams.
template <int kBatchSize>
[[nodiscard]] BatchActiveNeoHookeanParams<kBatchSize> BuildBatchParams(
    ActiveNeoHookeanMaterialParams const& p) {
  return {
      .lame = details::BuildBatchLameParams<kBatchSize>(p.passiveIsotropic),
      .aniso = BuildBatchParams<kBatchSize>(p.activeAnisotropic)};
}

// ---------------------------------------------------------------------------
// Implementation details
// ---------------------------------------------------------------------------

namespace details {

template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchActiveNeoHookeanParams<kBatchSize>
GatherActiveNeoHookeanParams(
    PerElementActiveNeoHookeanParams const& pe,
    NdArray<int, kBatchSize> const& indices) {
  return {
      .lame = GatherLameParams<kBatchSize>(pe.lame, indices),
      .aniso = GatherActiveAnisoParams<kBatchSize>(pe.aniso, indices)};
}

} // namespace details

// ---------------------------------------------------------------------------
// Batched constitutive response
// ---------------------------------------------------------------------------

/// @brief Batched constitutive response declaration.
///
/// @warning Nested PSD strategies in `params` must be resolved. They must not be @ref
/// MaterialPsdStrategy::MaterialDefault.
template <int kBatchSize>
void BatchedActiveNeoHookeanConstitutiveResponse(
    BatchActiveNeoHookeanParams<kBatchSize> const&,
    BatchReal3x3<kBatchSize> const&,
    BatchDouble<kBatchSize>*,
    BatchReal3x3<kBatchSize>*,
    NdArray<BatchReal3x3<kBatchSize>, 3, 3>*,
    bool);

/// @brief Heterogeneous factory for batched Active Neo-Hookean constitutive response.
///
/// @warning The returned lambda captures @p perElem by reference; @p perElem must outlive the
/// returned lambda.
template <typename /*ParamsT*/, int kBatchSize = kDefaultFemBatchSize>
[[nodiscard]] auto MakeBatchedConstitutiveResponse(
    PerElementActiveNeoHookeanParams const& perElem) {
  return [&perElem](
             NdArray<int, kBatchSize> const& elementIndices,
             auto const& F,
             auto* e,
             auto* pk1,
             auto* tangent,
             bool psd) MOCHI_FORCE_INLINE_LAMBDA {
    auto const params = details::GatherActiveNeoHookeanParams<kBatchSize>(perElem, elementIndices);
    BatchedActiveNeoHookeanConstitutiveResponse<kBatchSize>(params, F, e, pk1, tangent, psd);
  };
}

} // namespace mochi::materials
