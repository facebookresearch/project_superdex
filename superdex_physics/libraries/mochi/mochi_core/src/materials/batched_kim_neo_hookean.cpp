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

#include <mochi_core/materials/batched_kim_neo_hookean.h>

#include <mochi_core/materials/material_params_utils.h>
#include <mochi_core/materials/material_utils.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/simd.h>

namespace mochi::materials {

template <int kBatchSize>
void BatchedKimNeoHookeanConstitutiveResponse(
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
      IsResolvedPsdStrategySupported<KimNeoHookeanMaterialParams>(params.psdStrategy),
      "Batched material response requires a supported, resolved PSD strategy.");
  MaterialPsdStrategy const psdStrategy = params.psdStrategy;

  // λ̂ = λ + μ reparameterization to recover linear elasticity in the small deformation limit.
  V const mu = params.mu;
  V const lambdaHat = params.lambda + params.mu;
  V const one = 1_r;

  V const J = Det(F);
  V const Jm1 = J - one;
  V const coeff0 = lambdaHat * Jm1 - mu; // λ̂(J−1) − μ

  if (outEnergy) {
    *outEnergy = StaticCast<Vd>(
        V{0.5_r} * mu * (NormSqr(F) - V{3_r}) - mu * Jm1 + V{0.5_r} * lambdaHat * Jm1 * Jm1);
  }

  bool const useEigensystemPath = outTangent && projectPsd &&
      (psdStrategy == MaterialPsdStrategy::Projection ||
       psdStrategy == MaterialPsdStrategy::AbsEigenProjection);
  bool const needCofF = outPK1 || (outTangent && !useEigensystemPath);

  V3x3 cofF MOCHI_NO_INIT;
  if (needCofF) {
    cofF = Cofactor(F);
  }

  if (outPK1) {
    // NOTE: *outPK1 = mu * F + coeff0 * cofF is cleaner but may cause register spilling.
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        (*outPK1)[r][c] = mu * F[r][c] + coeff0 * cofF[r][c];
      }
    }
  }

  if (outTangent) {
    auto& C = *outTangent;

    if (useEigensystemPath) {
      // Eigensystem decomposition path: F = U Σ VT.
      // Note: BatchedRotationVariantSvd3x3 returns U, V^T and sigma such that det(U) >= 0, det(V^T)
      // >= 0, and sigma[2] < 0 if and only if det(F) < 0.
      V3x3 U MOCHI_NO_INIT, VT MOCHI_NO_INIT;
      V3 sigma MOCHI_NO_INIT;
      BatchedRotationVariantSvd3x3<kBatchSize>(F, U, sigma, VT);

      // 9 eigenvalues of the Hessian in layout [scaling_0..2, twist_0..2, flip_0..2].
      V9 eigLambda MOCHI_NO_INIT;

      // Twist eigenvalues: μ + coeff0 · σₙ. Mode n pairs the other two axes.
      eigLambda[3] = mu + coeff0 * sigma[0];
      eigLambda[4] = mu + coeff0 * sigma[1];
      eigLambda[5] = mu + coeff0 * sigma[2];

      // Flip eigenvalues: μ − coeff0 · σₙ.
      eigLambda[6] = mu - coeff0 * sigma[0];
      eigLambda[7] = mu - coeff0 * sigma[1];
      eigLambda[8] = mu - coeff0 * sigma[2];

      // Scaling mode matrix A (3×3 symmetric).
      // A_diag[i] = μ + λ̂ σⱼ² σₖ², A_offdiag[01] = (λ̂(2J−1)−μ) σ₂, etc.
      V3 const sigmaSqr = Sqr(sigma);
      V const offDiagCoeff = lambdaHat * (V{2_r} * J - one) - mu;

      V6 const Asym = {
          mu + lambdaHat * sigmaSqr[1] * sigmaSqr[2], // A(0,0)
          mu + lambdaHat * sigmaSqr[0] * sigmaSqr[2], // A(1,1)
          mu + lambdaHat * sigmaSqr[0] * sigmaSqr[1], // A(2,2)
          offDiagCoeff * sigma[2], // A(0,1)
          offDiagCoeff * sigma[1], // A(0,2)
          offDiagCoeff * sigma[0]}; // A(1,2)

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
      // Direct tangent: H = μI + λ̂(cofF ⊗ cofF) + coeff0 · d²J/dF².
      //
      // Fast PSD path: drop the indefinite d²J/dF² term.
      // Remaining tangent H = μI + λ̂(cofF ⊗ cofF) is guaranteed PSD.
      //
      // Exploit symmetry: compute upper triangle, then mirror.
      // Contraction strategy: hoist the temporary product, mirror in a separate pass. Fastest for
      // 1-rank contractions on x86-64 (AVX2) and ARM (NEON).
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          int const ij = i * 3 + j;
          V const lhc = lambdaHat * cofF[i][j];
          for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
              int const kl = k * 3 + l;
              if (kl >= ij) {
                C[i][j][k][l] = lhc * cofF[k][l];
              }
            }
          }
        }
      }

      // μ on diagonal.
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          C[i][j][i][j] += mu;
        }
      }

      // d²J/dF² term (skip for Fast PSD).
      if (!projectPsd || psdStrategy != MaterialPsdStrategy::Fast) {
        utils::BatchAddD2JdF2UpperTriangle<kBatchSize>(coeff0, F, C);
      }

      utils::BatchMirrorTangentUpperToLower<kBatchSize>(C);
    }
  }
}

MOCHI_INSTANTIATE_BATCHED_MATERIAL(BatchedKimNeoHookeanConstitutiveResponse, BatchLameParams);

} // namespace mochi::materials
