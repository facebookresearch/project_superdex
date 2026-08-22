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

#include <mochi_core/materials/batched_arap.h>

#include <mochi_core/materials/material_params_utils.h>
#include <mochi_core/materials/material_utils.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/simd.h>

namespace mochi::materials {

template <int kBatchSize>
void BatchedArapConstitutiveResponse(
    BatchArapParams<kBatchSize> const& params,
    BatchReal3x3<kBatchSize> const& F,
    BatchDouble<kBatchSize>* outEnergy,
    BatchReal3x3<kBatchSize>* outPK1,
    NdArray<BatchReal3x3<kBatchSize>, 3, 3>* outTangent,
    bool projectPsd) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  using V3x3 = BatchReal3x3<kBatchSize>;
  MOCHI_ASSERT_VERBOSE(
      IsFinite(params.stiffness) && AllTrue<kBatchSize>(params.stiffness > V{0_r}),
      "ARAP stiffness must be positive and finite.");
  MOCHI_ASSERT_VERBOSE(
      IsResolvedPsdStrategySupported<ArapMaterialParams>(params.psdStrategy),
      "Batched material response requires a supported, resolved PSD strategy.");
  MaterialPsdStrategy const psdStrategy = params.psdStrategy;

  // Perform SVD decomposition of F = U * Diag(sigma) * V^T.
  // Note: BatchedRotationVariantSvd3x3 returns U, V^T and sigma such that det(U) >= 0, det(V^T) >=
  // 0, and sigma[2] < 0 if and only if det(F) < 0. When only energy is requested, use the vals-only
  // SVD (skipping eigenvectors).
  V3x3 U MOCHI_NO_INIT, VT MOCHI_NO_INIT;
  V3 sigma MOCHI_NO_INIT;
  bool const needSingularVectors = (outPK1 != nullptr) || (outTangent != nullptr);
  if (needSingularVectors) {
    BatchedRotationVariantSvd3x3<kBatchSize>(F, U, sigma, VT);
  } else {
    BatchedRotationVariantSvdVals3x3<kBatchSize>(F, sigma);
  }

  V const mu = params.stiffness;

  if (outEnergy) {
    V const one{1_r};
    V3 const d = {sigma[0] - one, sigma[1] - one, sigma[2] - one};
    *outEnergy = StaticCast<Vd>(V{0.5_r} * mu * NormSqr(d));
  }

  if (outPK1) {
    // PK1 = μ * (F - U * VT)
    (*outPK1) = mu * (F - Dot(U, VT));
  }

  if (outTangent) {
    // Twist modes: T0 = UT[2]⊗VT[1] - UT[1]⊗VT[2]  (mode 0: axes 1,2)
    //              T1 = UT[2]⊗VT[0] - UT[0]⊗VT[2]  (mode 1: axes 0,2)
    //              T2 = UT[1]⊗VT[0] - UT[0]⊗VT[1]  (mode 2: axes 0,1)
    V3x3 const UT = Transpose(U);
    V3x3 T[3] = {
        Outer(UT[2], VT[1]) - Outer(UT[1], VT[2]),
        Outer(UT[2], VT[0]) - Outer(UT[0], VT[2]),
        Outer(UT[1], VT[0]) - Outer(UT[0], VT[1])};

    // Eigenvalues: λₙ = -μ / (σⱼ + σₖ) for twist mode n pairing axes j,k.
    // Mode 0 pairs (1,2), mode 1 pairs (0,2), mode 2 pairs (0,1).
    V3 lambda = {
        -mu / (sigma[1] + sigma[2]), -mu / (sigma[0] + sigma[2]), -mu / (sigma[0] + sigma[1])};

    // Apply PSD projection, if necessary. The tangent eigenvalues along twist modes are
    // eₙ = 2λₙ + μ. Non-twist eigenvalues are μ > 0 and need no projection.
    if (projectPsd && (psdStrategy != MaterialPsdStrategy::None)) {
      // Tangent eigenvalues along twist modes: eₙ = 2λₙ + μ.
      V const e0 = V{2_r} * lambda[0] + mu;
      V const e1 = V{2_r} * lambda[1] + mu;
      V const e2 = V{2_r} * lambda[2] + mu;

      if (psdStrategy == MaterialPsdStrategy::AbsEigenProjection) {
        // Compute eₙ, apply abs-filtering eₙ'  = max(|eₙ|, ε), then recover λₙ' = (eₙ' − μ) / 2.
        V const minEig = V{kMinProjectedEigenvalue};
        lambda[0] = V{0.5_r} * (Max(Abs(e0), minEig) - mu);
        lambda[1] = V{0.5_r} * (Max(Abs(e1), minEig) - mu);
        lambda[2] = V{0.5_r} * (Max(Abs(e2), minEig) - mu);
      } else {
        // Clamp eₙ' = max(eₙ, ε), then recover λₙ' = (eₙ' − μ) / 2.
        V const minLambda = V{0.5_r * (kMinProjectedEigenvalue - params.stiffness)};
        lambda[0] = Max(lambda[0], minLambda);
        lambda[1] = Max(lambda[1], minLambda);
        lambda[2] = Max(lambda[2], minLambda);
      }
    }

    // Tangent assembly: C[i][j][k][l] = μ·δᵢₖδⱼₗ + Σₙ λₙ·Tₙ[i][j]·Tₙ[k][l]
    // Contraction strategy: hoist the 3 temporary products, mirror in a separate pass. Fastest for
    // 3-rank contractions on x86-64 (AVX2); within noise on ARM (NEON).
    auto& C = *outTangent;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        int const ij = i * 3 + j;
        V const s0 = lambda[0] * T[0][i][j];
        V const s1 = lambda[1] * T[1][i][j];
        V const s2 = lambda[2] * T[2][i][j];
        for (int k = 0; k < 3; ++k) {
          for (int l = 0; l < 3; ++l) {
            int const kl = k * 3 + l;
            if (kl >= ij) {
              C[i][j][k][l] = s0 * T[0][k][l] + s1 * T[1][k][l] + s2 * T[2][k][l];
            }
          }
        }
        C[i][j][i][j] += mu;
      }
    }
    utils::BatchMirrorTangentUpperToLower<kBatchSize>(C);
  }
}

MOCHI_INSTANTIATE_BATCHED_MATERIAL(BatchedArapConstitutiveResponse, BatchArapParams);

} // namespace mochi::materials
