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

#include <mochi_core/materials/active_shape_targeting_arap_params.h>
#include <mochi_core/materials/batched_material_infra.h>
#include <mochi_core/materials/material_traits.h>
#include <mochi_core/materials/material_utils.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/batch_types.h>

#include <type_traits>

/**
 * Active As-Rigid-As-Possible (ARAP) material model with Shape Targeting by Klár et al. (2020).
 *
 * Reference: Klár et al., "Shape Targeting: A Versatile Active Elasticity Constitutive Model", 2020
 * (https://par.nsf.gov/servlets/purl/10230451)
 */
namespace mochi::materials {

template <>
[[nodiscard]] constexpr bool IsPsdStrategySupported<ActiveShapeTargetingArapMaterialParams>(
    MaterialPsdStrategy s) {
  return s == MaterialPsdStrategy::MaterialDefault || s == MaterialPsdStrategy::None ||
      s == MaterialPsdStrategy::Projection || s == MaterialPsdStrategy::PerTermProjection ||
      s == MaterialPsdStrategy::AbsEigenProjection;
}

// ---------------------------------------------------------------------------
// Per-element Active Shape Targeting ARAP params
// ---------------------------------------------------------------------------

/// @brief Per-element Active Shape Targeting ARAP params.
/// @see ActiveShapeTargetingArapMaterialParams
struct PerElementActiveShapeTargetingArapParams {
  DynamicArray<real> stiffness; ///< Shape targeting stiffness [Pa], one per element.

  /// 6 values per element (symmetric upper triangle), flat layout.
  /// Defines the shape target tensor S_t = I + [[s0,s1,s2],[s1,s3,s4],[s2,s4,s5]].
  DynamicArray<real> shapeTargetTensor;

  MaterialPsdStrategy psdStrategy = MaterialPsdStrategy::MaterialDefault;

  [[nodiscard]] auto size() const {
    MOCHI_ASSERT_VERBOSE(shapeTargetTensor.size() == stiffness.size() * 6, "Inconsistent sizes.");
    return stiffness.size();
  }
};

/// @brief Batch Active Shape Targeting ARAP params.
/// @see ActiveShapeTargetingArapMaterialParams
template <int kBatchSize>
struct BatchActiveShapeTargetingArapParams {
  using V = BatchReal<kBatchSize>;
  using V6 = BatchReal6<kBatchSize>;
  V stiffness{};
  V6 shapeTargetTensor{};
  MaterialPsdStrategy psdStrategy = MaterialPsdStrategy::MaterialDefault;
};

/// @brief ActiveShapeTargetingArapMaterialParams → PerElementActiveShapeTargetingArapParams
/// (homogeneous).
[[nodiscard]] inline PerElementActiveShapeTargetingArapParams BuildPerElementParams(
    ActiveShapeTargetingArapMaterialParams const& p) {
  MOCHI_ASSERT_VERBOSE(
      IsFinite(p.stiffness) && p.stiffness > 0, "Invalid shape targeting stiffness.");
  MOCHI_ASSERT_VERBOSE(IsFinite(p.shapeTargetTensor), "Invalid shape target tensor.");
  return {
      .stiffness = {p.stiffness},
      .shapeTargetTensor =
          {p.shapeTargetTensor[0],
           p.shapeTargetTensor[1],
           p.shapeTargetTensor[2],
           p.shapeTargetTensor[3],
           p.shapeTargetTensor[4],
           p.shapeTargetTensor[5]},
      .psdStrategy = utils::ResolvePsdStrategy(p)};
}

/// @brief PerElementActiveShapeTargetingArapParams → ActiveShapeTargetingArapMaterialParams for one
/// element.
template <typename ParamsT>
  requires(std::is_same_v<ParamsT, ActiveShapeTargetingArapMaterialParams>)
[[nodiscard]] inline ParamsT GetElementParams(
    PerElementActiveShapeTargetingArapParams const& perElementParams,
    int elementIndex) {
  int const i = details::GetPerElementParamsIndex(isize(perElementParams), elementIndex);
  return {
      .stiffness = perElementParams.stiffness[i],
      .shapeTargetTensor =
          {perElementParams.shapeTargetTensor[6 * i + 0],
           perElementParams.shapeTargetTensor[6 * i + 1],
           perElementParams.shapeTargetTensor[6 * i + 2],
           perElementParams.shapeTargetTensor[6 * i + 3],
           perElementParams.shapeTargetTensor[6 * i + 4],
           perElementParams.shapeTargetTensor[6 * i + 5]},
      .psdStrategy = perElementParams.psdStrategy};
}

/// @brief ActiveShapeTargetingArapMaterialParams → BatchActiveShapeTargetingArapParams.
template <int kBatchSize>
[[nodiscard]] BatchActiveShapeTargetingArapParams<kBatchSize> BuildBatchParams(
    ActiveShapeTargetingArapMaterialParams const& p) {
  MOCHI_ASSERT_VERBOSE(
      IsFinite(p.stiffness) && p.stiffness > 0, "Invalid shape targeting stiffness.");
  MOCHI_ASSERT_VERBOSE(IsFinite(p.shapeTargetTensor), "Invalid shape target tensor.");
  using V = BatchReal<kBatchSize>;
  auto const& st = p.shapeTargetTensor;
  return {
      .stiffness = p.stiffness,
      .shapeTargetTensor = BroadcastEach<V>(st),
      .psdStrategy = utils::ResolvePsdStrategy(p)};
}

// ---------------------------------------------------------------------------
// Batched constitutive response
// ---------------------------------------------------------------------------

/// @brief Batched constitutive response declaration.
///
/// @warning `params.psdStrategy` must be resolved. It must not be @ref
/// MaterialPsdStrategy::MaterialDefault.
template <int kBatchSize>
void BatchedActiveShapeTargetingArapConstitutiveResponse(
    BatchActiveShapeTargetingArapParams<kBatchSize> const&,
    BatchReal3x3<kBatchSize> const&,
    BatchDouble<kBatchSize>*,
    BatchReal3x3<kBatchSize>*,
    NdArray<BatchReal3x3<kBatchSize>, 3, 3>*,
    bool);

/// @brief Heterogeneous factory for batched Active Shape Targeting ARAP constitutive response.
///
/// @warning The returned lambda captures @p perElem by reference; @p perElem must outlive the
/// returned lambda.
template <typename /*ParamsT*/, int kBatchSize = kDefaultFemBatchSize>
[[nodiscard]] auto MakeBatchedConstitutiveResponse(
    PerElementActiveShapeTargetingArapParams const& perElem) {
  return [&perElem](
             NdArray<int, kBatchSize> const& elementIndices,
             auto const& F,
             auto* e,
             auto* pk1,
             auto* tangent,
             bool psd) MOCHI_FORCE_INLINE_LAMBDA {
    using V = BatchReal<kBatchSize>;
    BatchActiveShapeTargetingArapParams<kBatchSize> params{.psdStrategy = perElem.psdStrategy};
    if (details::IsHomogeneous(perElem)) {
      params.stiffness = perElem.stiffness[0];
      params.shapeTargetTensor = {
          V{perElem.shapeTargetTensor[0]},
          V{perElem.shapeTargetTensor[1]},
          V{perElem.shapeTargetTensor[2]},
          V{perElem.shapeTargetTensor[3]},
          V{perElem.shapeTargetTensor[4]},
          V{perElem.shapeTargetTensor[5]}};
    } else {
      params.stiffness = details::GatherIndexed<kBatchSize>(perElem.stiffness, elementIndices);
      for (int c = 0; c < 6; ++c) {
        params.shapeTargetTensor[c] = details::GatherIndexed<kBatchSize>(
            [&](int i) { return perElem.shapeTargetTensor[i * 6 + c]; }, elementIndices);
      }
    }
    BatchedActiveShapeTargetingArapConstitutiveResponse<kBatchSize>(
        params, F, e, pk1, tangent, psd);
  };
}

} // namespace mochi::materials
