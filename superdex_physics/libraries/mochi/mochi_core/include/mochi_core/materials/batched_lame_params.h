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

#include <mochi_core/materials/batched_material_infra.h>
#include <mochi_core/materials/material_utils.h>

namespace mochi::materials {

/// Trait: is this a Lamé-based material? Specialized per-material.
template <typename T>
inline constexpr bool kIsLameMaterial = false;

// ---------------------------------------------------------------------------
// Per-element Lamé coefficients
// ---------------------------------------------------------------------------

/// @brief Per-element Lamé coefficients. size()==1 → homogeneous, size()==N →
/// heterogeneous.
struct PerElementLameParams {
  DynamicArray<real> lambda; ///< First Lamé parameter (λ) [Pa], one per element.
  DynamicArray<real> mu; ///< Shear modulus (μ) [Pa], one per element.
  MaterialPsdStrategy psdStrategy = MaterialPsdStrategy::MaterialDefault;

  [[nodiscard]] auto size() const {
    MOCHI_ASSERT_VERBOSE(lambda.size() == mu.size(), "Inconsistent sizes.");
    return mu.size();
  }
};

/// @brief Batch Lamé coefficients.
template <int kBatchSize>
struct BatchLameParams {
  using V = BatchReal<kBatchSize>;
  V lambda{};
  V mu{};
  MaterialPsdStrategy psdStrategy = MaterialPsdStrategy::MaterialDefault;
};

/// @brief Build heterogeneous Lamé coefficients from per-element Young's modulus and Poisson ratio.
[[nodiscard]] inline PerElementLameParams BuildPerElementLameParams(
    Span<real const> E,
    Span<real const> nu,
    MaterialPsdStrategy psdStrategy = MaterialPsdStrategy::MaterialDefault) {
  MOCHI_ASSERT_VERBOSE(E.size() == nu.size(), "Size mismatch.");
  int const n = isize(E);
  PerElementLameParams out;
  out.mu.resize(n);
  out.lambda.resize(n);
  for (int i = 0; i < n; ++i) {
    MOCHI_ASSERT_VERBOSE(IsFinite(E[i]) && E[i] > 0, "Invalid Young's modulus.");
    MOCHI_ASSERT_VERBOSE(
        IsFinite(nu[i]) && nu[i] > -1_r && nu[i] < 0.5_r, "Invalid Poisson ratio.");
    auto const [lam, mu] = utils::ComputeLameConstants(E[i], nu[i]);
    out.mu[i] = mu;
    out.lambda[i] = lam;
  }
  out.psdStrategy = psdStrategy;
  return out;
}

// ---------------------------------------------------------------------------
// Implementation details
// ---------------------------------------------------------------------------

namespace details {

template <typename ParamsT>
[[nodiscard]] PerElementLameParams BuildPerElementLameParams(ParamsT const& p) {
  MOCHI_ASSERT_VERBOSE(
      IsFinite(p.youngsModulus) && p.youngsModulus > 0, "Invalid Young's modulus.");
  MOCHI_ASSERT_VERBOSE(
      IsFinite(p.poissonRatio) && p.poissonRatio > -1_r && p.poissonRatio < 0.5_r,
      "Invalid Poisson ratio.");
  auto const [lam, mu] = utils::ComputeLameConstants(p.youngsModulus, p.poissonRatio);
  return {.lambda = {lam}, .mu = {mu}, .psdStrategy = utils::ResolvePsdStrategy(p)};
}

template <typename ParamsT>
[[nodiscard]] ParamsT GetElementLameParams(
    PerElementLameParams const& perElementParams,
    int elementIndex) {
  int const i = GetPerElementParamsIndex(isize(perElementParams), elementIndex);
  real const lambda = perElementParams.lambda[i];
  real const mu = perElementParams.mu[i];
  real const denom = lambda + mu;
  MOCHI_ASSERT_VERBOSE(denom != 0_r, "Invalid Lame constants.");
  real const invDenom = 1_r / denom;

  ParamsT out;
  out.youngsModulus = mu * (3_r * lambda + 2_r * mu) * invDenom;
  out.poissonRatio = lambda * (0.5_r * invDenom);
  if constexpr (requires { out.psdStrategy; }) {
    out.psdStrategy = perElementParams.psdStrategy;
  }
  return out;
}

template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchLameParams<kBatchSize> GatherLameParams(
    PerElementLameParams const& pe,
    NdArray<int, kBatchSize> const& elementIndices) {
  if (IsHomogeneous(pe)) {
    return {.lambda = pe.lambda[0], .mu = pe.mu[0], .psdStrategy = pe.psdStrategy};
  } else {
    return {
        .lambda = details::GatherIndexed<kBatchSize>(pe.lambda, elementIndices),
        .mu = details::GatherIndexed<kBatchSize>(pe.mu, elementIndices),
        .psdStrategy = pe.psdStrategy};
  }
}

template <int kBatchSize, typename ParamsT>
[[nodiscard]] BatchLameParams<kBatchSize> BuildBatchLameParams(ParamsT const& p) {
  MOCHI_ASSERT_VERBOSE(
      IsFinite(p.youngsModulus) && p.youngsModulus > 0, "Invalid Young's modulus.");
  MOCHI_ASSERT_VERBOSE(
      IsFinite(p.poissonRatio) && p.poissonRatio > -1_r && p.poissonRatio < 0.5_r,
      "Invalid Poisson ratio.");
  auto const [lam, mu] = utils::ComputeLameConstants(p.youngsModulus, p.poissonRatio);
  return {.lambda = lam, .mu = mu, .psdStrategy = utils::ResolvePsdStrategy(p)};
}

} // namespace details

/// @brief MaterialParams → PerElementLameParams (homogeneous).
template <typename T>
  requires(kIsLameMaterial<T>)
[[nodiscard]] PerElementLameParams BuildPerElementParams(T const& p) {
  return details::BuildPerElementLameParams(p);
}

/// @brief PerElementLameParams → MaterialParams for one element.
template <typename T>
  requires(kIsLameMaterial<T>)
[[nodiscard]] T GetElementParams(PerElementLameParams const& perElementParams, int elementIndex) {
  return details::GetElementLameParams<T>(perElementParams, elementIndex);
}

/// @brief MaterialParams → BatchLameParams.
template <int kBatchSize, typename T>
  requires(kIsLameMaterial<T>)
[[nodiscard]] BatchLameParams<kBatchSize> BuildBatchParams(T const& p) {
  return details::BuildBatchLameParams<kBatchSize>(p);
}

} // namespace mochi::materials
