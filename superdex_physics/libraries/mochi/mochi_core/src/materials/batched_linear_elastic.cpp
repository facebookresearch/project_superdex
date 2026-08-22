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

#include <mochi_core/materials/batched_linear_elastic.h>

#include <mochi_core/materials/batched_material_infra.h>
#include <mochi_core/materials/material_utils.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/simd.h>

namespace mochi::materials {

template <int kBatchSize>
void BatchedLinearElasticConstitutiveResponse(
    BatchLameParams<kBatchSize> const& params,
    BatchReal3x3<kBatchSize> const& F,
    BatchDouble<kBatchSize>* outEnergy,
    BatchReal3x3<kBatchSize>* outPK1,
    NdArray<BatchReal3x3<kBatchSize>, 3, 3>* outTangent,
    bool /*projectPsd*/) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  using V3x3 = BatchReal3x3<kBatchSize>;
  MOCHI_ASSERT_VERBOSE(
      IsFinite(params.mu) && AllTrue<kBatchSize>(params.mu > V{0_r}) && IsFinite(params.lambda),
      "Invalid Lame parameters.");
  MOCHI_ASSERT_VERBOSE(
      IsResolvedPsdStrategySupported<LinearElasticMaterialParams>(params.psdStrategy),
      "Batched material response requires a supported, resolved PSD strategy.");

  V const lambda = params.lambda;
  V const mu = params.mu;

  // Strain: ε = 0.5 * (F + F^T) - I
  V3x3 strain;
  for (int r = 0; r < 3; ++r) {
    strain[r][r] = F[r][r] - V{1_r};
    for (int c = r + 1; c < 3; ++c) {
      V const s = 0.5_r * (F[r][c] + F[c][r]);
      strain[r][c] = s;
      strain[c][r] = s;
    }
  }

  V const trE = Trace(strain);

  if (outEnergy) {
    // Strain energy: Ψ = μ‖ε‖² + ½λ(tr ε)²
    *outEnergy = StaticCast<Vd>(mu * NormSqr(strain) + 0.5_r * lambda * trE * trE);
  }

  if (outPK1) {
    // First Piola–Kirchhoff stress: PK1 = ∂Ψ/∂F = λ(tr ε)I + 2με
    (*outPK1) = (2_r * mu) * strain;
    for (int r = 0; r < 3; ++r) {
      (*outPK1)[r][r] += lambda * trE;
    }
  }

  if (outTangent) {
    // Tangent (∂²Ψ/∂F²): 𝒞ᵢⱼₖₗ = λ δᵢⱼ δₖₗ + μ (δᵢₖ δⱼₗ + δᵢₗ δⱼₖ), where δ is the Kronecker
    // delta.
    // TODO: The tangent is constant and could be precomputed once per material.
    auto& C = *outTangent;
    V const z = 0_r;
    V const l2m = lambda + 2_r * mu;

    // clang-format off
    C[0][0] = {V3{l2m,    z,   z},  V3{   z, lambda,  z},  V3{    z,  z, lambda}};
    C[0][1] = {V3{  z,   mu,   z},  V3{  mu,      z,  z},  V3{    z,  z,      z}};
    C[0][2] = {V3{  z,    z,  mu},  V3{   z,      z,  z},  V3{   mu,  z,      z}};
    C[1][0] = {V3{  z,   mu,   z},  V3{  mu,      z,  z},  V3{    z,  z,      z}};
    C[1][1] = {V3{lambda, z,   z},  V3{   z,    l2m,  z},  V3{    z,  z, lambda}};
    C[1][2] = {V3{  z,    z,   z},  V3{   z,      z, mu},  V3{    z, mu,      z}};
    C[2][0] = {V3{  z,    z,  mu},  V3{   z,      z,  z},  V3{   mu,  z,      z}};
    C[2][1] = {V3{  z,    z,   z},  V3{   z,      z, mu},  V3{    z, mu,      z}};
    C[2][2] = {V3{lambda, z,   z},  V3{   z, lambda,  z},  V3{    z,  z,    l2m}};
    // clang-format on
  }
}

MOCHI_INSTANTIATE_BATCHED_MATERIAL(BatchedLinearElasticConstitutiveResponse, BatchLameParams);

} // namespace mochi::materials
