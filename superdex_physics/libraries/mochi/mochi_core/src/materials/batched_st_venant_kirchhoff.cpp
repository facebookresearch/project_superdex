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

#include <mochi_core/materials/batched_st_venant_kirchhoff.h>

#include <mochi_core/materials/material_params_utils.h>
#include <mochi_core/materials/material_utils.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/simd.h>

namespace mochi::materials {

template <int kBatchSize>
void BatchedStVenantKirchhoffConstitutiveResponse(
    BatchLameParams<kBatchSize> const& params,
    BatchReal3x3<kBatchSize> const& F,
    BatchDouble<kBatchSize>* outEnergy,
    BatchReal3x3<kBatchSize>* outPK1,
    NdArray<BatchReal3x3<kBatchSize>, 3, 3>* outTangent,
    bool projectPsd) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  using V6 = BatchReal6<kBatchSize>;
  using V9 = BatchReal9<kBatchSize>;
  using V3x3 = BatchReal3x3<kBatchSize>;
  MOCHI_ASSERT_VERBOSE(
      IsFinite(params.mu) && AllTrue<kBatchSize>(params.mu > V{0_r}) && IsFinite(params.lambda),
      "Invalid Lame parameters.");
  MOCHI_ASSERT_VERBOSE(
      IsResolvedPsdStrategySupported<StVenantKirchhoffMaterialParams>(params.psdStrategy),
      "Batched material response requires a supported, resolved PSD strategy.");
  MaterialPsdStrategy const psdStrategy = params.psdStrategy;

  V const lambda = params.lambda;
  V const mu = params.mu;
  V const coef0 = V{2_r} * mu;

  // Green strain G = ½(FᵀF − I). Symmetric, so only 6 unique entries.
  V const half = 0.5_r;
  V const one = 1_r;
  V G00 = half * (F[0][0] * F[0][0] + F[1][0] * F[1][0] + F[2][0] * F[2][0] - one);
  V G11 = half * (F[0][1] * F[0][1] + F[1][1] * F[1][1] + F[2][1] * F[2][1] - one);
  V G22 = half * (F[0][2] * F[0][2] + F[1][2] * F[1][2] + F[2][2] * F[2][2] - one);
  V G01 = half * (F[0][0] * F[0][1] + F[1][0] * F[1][1] + F[2][0] * F[2][1]);
  V G02 = half * (F[0][0] * F[0][2] + F[1][0] * F[1][2] + F[2][0] * F[2][2]);
  V G12 = half * (F[0][1] * F[0][2] + F[1][1] * F[1][2] + F[2][1] * F[2][2]);

  V trG = G00 + G11 + G22;

  if (outEnergy) {
    // ψ = μ|G|² + ½λ tr²(G), where |G|² = sum of all 9 entries squared.
    V const normSqrG =
        G00 * G00 + G11 * G11 + G22 * G22 + V{2_r} * (G01 * G01 + G02 * G02 + G12 * G12);
    *outEnergy = StaticCast<Vd>(mu * normSqrG + half * lambda * trG * trG);
  }

  if (outPK1) {
    // P = F · S, where S = 2μG + λ tr(G) I (symmetric 3×3).
    V const ltrG = lambda * trG;
    V const S00 = coef0 * G00 + ltrG;
    V const S11 = coef0 * G11 + ltrG;
    V const S22 = coef0 * G22 + ltrG;
    V const S01 = coef0 * G01;
    V const S02 = coef0 * G02;
    V const S12 = coef0 * G12;

    auto& P = *outPK1;
    for (int r = 0; r < 3; ++r) {
      P[r][0] = F[r][0] * S00 + F[r][1] * S01 + F[r][2] * S02;
      P[r][1] = F[r][0] * S01 + F[r][1] * S11 + F[r][2] * S12;
      P[r][2] = F[r][0] * S02 + F[r][1] * S12 + F[r][2] * S22;
    }
  }

  if (outTangent) {
    auto& C = *outTangent;

    if (projectPsd &&
        (psdStrategy == MaterialPsdStrategy::Projection ||
         psdStrategy == MaterialPsdStrategy::AbsEigenProjection)) {
      // Implementation based on Appendix F.3 of "Analytic Eigensystems for Isotropic Distortion
      // Energies" (https://www.tkim.graphics/EIGENSYSTEMS/AnalyticEigensystems.pdf)
      //
      // Compute SVD of deformation gradient F = U * Diag(sigma) * V^T.
      // Note: BatchedRotationVariantSvd3x3 returns U, V and sigma such that det(U) >= 0, det(V) >=
      // 0, and sigma[2] < 0 if and only if det(F) < 0.
      V3x3 U MOCHI_NO_INIT, VT MOCHI_NO_INIT;
      V3 sigma MOCHI_NO_INIT;
      BatchedRotationVariantSvd3x3<kBatchSize>(F, U, sigma, VT);

      V3 const sigmaSqr = Sqr(sigma);
      V const I2 = sigmaSqr[0] + sigmaSqr[1] + sigmaSqr[2];
      V const baseValue = -mu + half * lambda * (I2 - V{3_r});

      // 9 eigenvalues of the Hessian in layout [scaling_0..2, twist_0..2, flip_0..2].
      V9 eigLambda MOCHI_NO_INIT;

      // Twist: baseValue + μ(σⱼ²+σₖ² − σⱼσₖ). Mode n pairs (j,k): n=0→(1,2), n=1→(0,2), n=2→(0,1).
      eigLambda[3] = baseValue + mu * (sigmaSqr[1] + sigmaSqr[2] - sigma[1] * sigma[2]);
      eigLambda[4] = baseValue + mu * (sigmaSqr[0] + sigmaSqr[2] - sigma[0] * sigma[2]);
      eigLambda[5] = baseValue + mu * (sigmaSqr[0] + sigmaSqr[1] - sigma[0] * sigma[1]);

      // Flip: baseValue + μ(σⱼ²+σₖ² + σⱼσₖ).
      eigLambda[6] = baseValue + mu * (sigmaSqr[1] + sigmaSqr[2] + sigma[1] * sigma[2]);
      eigLambda[7] = baseValue + mu * (sigmaSqr[0] + sigmaSqr[2] + sigma[0] * sigma[2]);
      eigLambda[8] = baseValue + mu * (sigmaSqr[0] + sigmaSqr[1] + sigma[0] * sigma[1]);

      // Scaling mode matrix A (3×3 symmetric).
      V const diagCoeff = lambda + V{3_r} * mu;
      V6 const Asym = {
          baseValue + diagCoeff * sigmaSqr[0], // A(0,0)
          baseValue + diagCoeff * sigmaSqr[1], // A(1,1)
          baseValue + diagCoeff * sigmaSqr[2], // A(2,2)
          lambda * sigma[0] * sigma[1], // A(0,1)
          lambda * sigma[0] * sigma[2], // A(0,2)
          lambda * sigma[1] * sigma[2]}; // A(1,2)

      V3 scalingEigvals MOCHI_NO_INIT;
      V3x3 AqT MOCHI_NO_INIT;
      BatchedAnalyticalEigendecompSym3x3<kBatchSize>(Asym, scalingEigvals, &AqT);

      eigLambda[0] = scalingEigvals[0];
      eigLambda[1] = scalingEigvals[1];
      eigLambda[2] = scalingEigvals[2];

      // Clamp eigenvalues.
      V const minEig = V{kMinProjectedEigenvalue};
      if (psdStrategy == MaterialPsdStrategy::Projection) {
        for (int i = 0; i < 9; ++i) {
          eigLambda[i] = Max(eigLambda[i], minEig);
        }
      } else {
        for (int i = 0; i < 9; ++i) {
          eigLambda[i] = Max(Abs(eigLambda[i]), minEig);
        }
      }

      utils::BatchedAssembleTangentFromEigensystem<kBatchSize>(eigLambda, U, VT, AqT, C);

    } else {
      // Direct tangent: ∂P/∂F_{ij} = δᵢ·S_row_j + F·[2μ(∂G/∂F_{ij}) + λ(∂tr(G)/∂F_{ij})I]
      // where ∂G_{pq}/∂F_{uv} = ½(δ_{pv}F_{uq} + δ_{qv}F_{up}), ∂tr(G)/∂F_{uv} = F_{uv}.
      if (projectPsd && psdStrategy == MaterialPsdStrategy::Fast) {
        // Fast PSD path - Use filtered G to prevent negative eigenvalues upon compression.
        // Remove shear components and cap stretch components to ℝ⁺.
        V const zero = 0_r;
        G00 = Max(G00, zero);
        G11 = Max(G11, zero);
        G22 = Max(G22, zero);
        G01 = zero;
        G02 = zero;
        G12 = zero;
        trG = G00 + G11 + G22;
      }

      // S = 2μG + λ tr(G) I.
      V const ltrGs = lambda * trG;
      V3x3 const S = {
          V3{coef0 * G00 + ltrGs, coef0 * G01, coef0 * G02},
          V3{coef0 * G01, coef0 * G11 + ltrGs, coef0 * G12},
          V3{coef0 * G02, coef0 * G12, coef0 * G22 + ltrGs}};

      V3x3 const FFT = Dot(F, Transpose(F)); // FFᵀ

      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          V const lFij = lambda * F[i][j]; // λ·F_{ij} — from ∂tr(G)/∂F_{ij} = F_{ij}
          for (int r = 0; r < 3; ++r) {
            V const muFrj = mu * F[r][j]; // μ·F_{rj} — from ∂G/∂F_{ij} contribution
            V const muFFTri = mu * FFT[r][i]; // μ·(FFᵀ)_{ri} — symmetric ∂G/∂F_{ij} term
            // F·[2μ(∂G/∂F_{ij}) + λ(∂tr(G)/∂F_{ij})I]
            for (int c = 0; c < 3; ++c) {
              C[i][j][r][c] = muFrj * F[i][c] + lFij * F[r][c];
            }
            C[i][j][r][j] += muFFTri;
          }
          // δᵢ·S_row_j: the ∂F/∂F_{ij} term contributes S to row i
          for (int c = 0; c < 3; ++c) {
            C[i][j][i][c] += S[j][c];
          }
        }
      }
    }
  }
}

MOCHI_INSTANTIATE_BATCHED_MATERIAL(BatchedStVenantKirchhoffConstitutiveResponse, BatchLameParams);

} // namespace mochi::materials
