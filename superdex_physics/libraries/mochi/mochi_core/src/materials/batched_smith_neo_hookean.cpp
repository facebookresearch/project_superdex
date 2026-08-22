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

#include <mochi_core/materials/batched_smith_neo_hookean.h>

#include <mochi_core/materials/material_params_utils.h>
#include <mochi_core/materials/material_utils.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/simd.h>

namespace mochi::materials {

template <int kBatchSize>
void BatchedSmithNeoHookeanConstitutiveResponse(
    BatchLameParams<kBatchSize> const& params,
    BatchReal3x3<kBatchSize> const& F,
    BatchDouble<kBatchSize>* outEnergy,
    BatchReal3x3<kBatchSize>* outPK1,
    NdArray<BatchReal3x3<kBatchSize>, 3, 3>* outTangent,
    bool projectPsd,
    MaterialPsdOracle oracle) {
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
      IsResolvedPsdStrategySupported<SmithNeoHookeanMaterialParams>(params.psdStrategy),
      "Batched material response requires a supported, resolved PSD strategy.");
  MaterialPsdStrategy const psdStrategy = params.psdStrategy;

  // Reparameterized Lamé constants (Sec. 3.4 of Smith et al. 2018).
  V const one = V{1_r};
  V const muHat = params.mu * V{4_r / 3_r};
  V const lambdaHat = params.lambda + params.mu * V{5_r / 6_r};
  V const alpha = one + V{3_r / 4_r} * muHat / lambdaHat;

  // Invariants.
  V const Ic = NormSqr(F);
  V const IcPlus1 = Ic + one;
  V const IcPlus1Inv = one / IcPlus1;
  V const J = Det(F);
  V const Jma = J - alpha;

  if (outEnergy) {
    *outEnergy = StaticCast<Vd>(
        V{0.5_r} * muHat * (Ic - V{3_r}) + V{0.5_r} * lambdaHat * Jma * Jma -
        V{0.5_r} * muHat * Ln(IcPlus1));
  }

  V3 sigma MOCHI_NO_INIT;
  BatchedRotationVariantSvdNormalEigensystem3x3<kBatchSize> svdNormalEigensystem MOCHI_NO_INIT;
  bool svdValsDone = false;
  bool projectingPsd = projectPsd && (psdStrategy != MaterialPsdStrategy::None);
  V isIndefiniteMask = ~SimdZero<V>(); // All lanes indefinite unless oracle proves otherwise.

  if (outTangent && projectingPsd && oracle != MaterialPsdOracle::None) {
    // PSD oracle: evaluate flip/twist indefiniteness per lane, skipping projection if no lane
    // requires it. The oracle is based on the fact that only twist and flip eigenvalues can make
    // the Smith tangent non-PSD. Their closed-form eigenvalues are:
    //
    //   e_twist_i =  lambdaHat * (J - alpha) * sigma_i + muHatK
    //   e_flip_i  = -lambdaHat * (J - alpha) * sigma_i + muHatK
    //
    // where muHatK = muHat * (1 - 1 / (Ic + 1)), Ic = sum_i sigma_i^2, and
    // J = prod_i sigma_i. Inverted elements (J < 0) are indefinite. For J >= 0, the sign cases for
    // (J - alpha) and sigma_i reduce to checking:
    //
    //   lambdaHat * abs(J - alpha) * abs(sigma_i) > muHatK.
    //
    // Squaring and taking the largest singular value gives the exact condition:
    //
    //   lambdaHat^2 * (J - alpha)^2 * max_i(sigma_i^2) > muHatK^2.
    //
    // Equivalently, Ic > K * C where K = Ic / max_i(sigma_i^2) and
    // C = muHatK^2 / (lambdaHat^2 * (J - alpha)^2). K lies in [1, 3]. Correct computes
    // max_i(sigma_i^2) from the SVD values, which is equivalent to the exact K. Conservative avoids
    // the SVD by using K = 1, i.e. Ic as an upper bound on max_i(sigma_i^2), so it can project more
    // often than necessary but does not miss required projections.
    V const muHatKSqr = Sqr(muHat * (one - IcPlus1Inv));
    V const lhsBase = Sqr(lambdaHat * Jma);

    if (oracle == MaterialPsdOracle::Correct) {
      BatchedRotationVariantSvdVals3x3<kBatchSize>(F, sigma, svdNormalEigensystem);
      svdValsDone = true;
      V const maxSigmaSq = Max(Sqr(sigma[0]), Sqr(sigma[1]), Sqr(sigma[2]));
      isIndefiniteMask = (lhsBase * maxSigmaSq > muHatKSqr) | (J < V{0_r});
    } else {
      MOCHI_ASSERT_VERBOSE(oracle == MaterialPsdOracle::Conservative, "Unexpected PSD oracle.");
      isIndefiniteMask = (lhsBase * Ic > muHatKSqr) | (J < V{0_r});
    }

    projectingPsd = AnyTrue<kBatchSize>(isIndefiniteMask);
  }

  bool const useEigensystemPath = outTangent && projectingPsd &&
      (psdStrategy == MaterialPsdStrategy::Projection ||
       psdStrategy == MaterialPsdStrategy::AbsEigenProjection);
  bool const needCofF = outPK1 || (outTangent && !useEigensystemPath);

  // The cofactor of the deformation gradient.
  V3x3 cofF MOCHI_NO_INIT;
  if (needCofF) {
    cofF = Cofactor(F);
  }

  // The first Piola--Kirchhoff.
  if (outPK1) {
    V const c0 = muHat * (one - IcPlus1Inv);
    V const c1 = lambdaHat * Jma;
    // NOTE: *outPK1 = c0 * F + c1 * cofF is cleaner but may cause register spilling.
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        (*outPK1)[r][c] = c0 * F[r][c] + c1 * cofF[r][c];
      }
    }
  }

  if (outTangent) {
    auto& C = *outTangent;

    if (useEigensystemPath) {
      // Eigensystem decomposition path (Smith et al. 2018).
      // Note: BatchedRotationVariantSvd3x3 returns U, V^T and sigma such that det(U) >= 0, det(V^T)
      // >= 0, and sigma[2] < 0 iff det(F) < 0. Likewise for BatchedRotationVariantSvdVecs3x3.
      V3x3 U MOCHI_NO_INIT, VT MOCHI_NO_INIT;
      if (svdValsDone) {
        BatchedRotationVariantSvdVecs3x3<kBatchSize>(F, svdNormalEigensystem, U, VT);
      } else {
        BatchedRotationVariantSvd3x3<kBatchSize>(F, U, sigma, VT);
      }

      V const coeff0 = muHat - muHat * IcPlus1Inv;
      V const coeff1 = lambdaHat * Jma;

      // 9 eigenvalues of the Hessian in layout [scaling_0..2, twist_0..2, flip_0..2].
      V9 eigLambda MOCHI_NO_INIT;

      // Twist: coeff1·σₙ + coeff0.
      eigLambda[3] = coeff1 * sigma[0] + coeff0;
      eigLambda[4] = coeff1 * sigma[1] + coeff0;
      eigLambda[5] = coeff1 * sigma[2] + coeff0;

      // Flip: −coeff1·σₙ + coeff0.
      eigLambda[6] = -coeff1 * sigma[0] + coeff0;
      eigLambda[7] = -coeff1 * sigma[1] + coeff0;
      eigLambda[8] = -coeff1 * sigma[2] + coeff0;

      // Scaling mode matrix A (3×3 symmetric).
      V const IcPlus1SqrInv = IcPlus1Inv * IcPlus1Inv;
      V const mu2 = V{2_r} * muHat;
      V const J2alphaLambda = (V{2_r} * J - alpha) * lambdaHat;

      V3 const sigmaSqr = Sqr(sigma);
      // clang-format off
      V6 const Asym = {
          (mu2 * sigmaSqr[0] - muHat * IcPlus1) * IcPlus1SqrInv + lambdaHat * sigmaSqr[1] * sigmaSqr[2] + muHat,
          (mu2 * sigmaSqr[1] - muHat * IcPlus1) * IcPlus1SqrInv + lambdaHat * sigmaSqr[0] * sigmaSqr[2] + muHat,
          (mu2 * sigmaSqr[2] - muHat * IcPlus1) * IcPlus1SqrInv + lambdaHat * sigmaSqr[0] * sigmaSqr[1] + muHat,
          J2alphaLambda * sigma[2] + mu2 * sigma[0] * sigma[1] * IcPlus1SqrInv,
          J2alphaLambda * sigma[1] + mu2 * sigma[0] * sigma[2] * IcPlus1SqrInv,
          J2alphaLambda * sigma[0] + mu2 * sigma[1] * sigma[2] * IcPlus1SqrInv};
      // clang-format on

      V3 scalingEigvals MOCHI_NO_INIT;
      V3x3 AqT MOCHI_NO_INIT;
      BatchedAnalyticalEigendecompSym3x3<kBatchSize>(Asym, scalingEigvals, &AqT);

      eigLambda[0] = scalingEigvals[0];
      eigLambda[1] = scalingEigvals[1];
      eigLambda[2] = scalingEigvals[2];

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
      // Direct tangent: C[ij][kl] = c₃·F[ij]·F[kl] + λ̂·cofF[ij]·cofF[kl]
      //                            + λ̂(J−α)·d²J/dF[ij]dF[kl] + c₂·δᵢₖδⱼₗ
      //
      // Coefficients:
      //   c₂ = μ̂(1 − 1/(Ic+1))   (identity term)
      //   c₃ = 2μ̂ / (Ic+1)²      (Ic² second derivative)
      V const c2 = muHat * (one - IcPlus1Inv);
      V const c3 = V{2_r} * muHat * IcPlus1Inv * IcPlus1Inv;
      bool const projectingFast = projectingPsd && (psdStrategy == MaterialPsdStrategy::Fast);

      if (projectingFast && AllTrue<kBatchSize>(isIndefiniteMask)) {
        // Fast PSD path - Drop offending second derivatives in the volume preservation term
        // altogether to prevent negative eigenvalues.
        // C[ij][kl] = c₃·F[ij]·F[kl] + λ̂·cofF[ij]·cofF[kl] + c₂·δᵢₖδⱼₗ
        // Note the omitted λ̂(J−α)·d²J/dF² term.
        // Contraction strategy: hoist both temporary products, mirror in a separate pass. Fastest
        // for 2-rank contractions on x86-64 (AVX2) and ARM (NEON).
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            int const ij = i * 3 + j;
            V const c3Fij = c3 * F[i][j];
            V const lambdaHatCofFij = lambdaHat * cofF[i][j];
            for (int k = 0; k < 3; ++k) {
              for (int l = 0; l < 3; ++l) {
                int const kl = k * 3 + l;
                if (kl >= ij) {
                  C[i][j][k][l] = c3Fij * F[k][l] + lambdaHatCofFij * cofF[k][l];
                }
              }
            }
            C[i][j][i][j] += c2;
          }
        }

      } else {
        V d2JCoeff = lambdaHat * Jma;
        if (projectingFast) {
          d2JCoeff = Select(isIndefiniteMask, V{0_r}, d2JCoeff);
        }

        // Contraction strategy for separable terms: hoist temporary products, mirror in a separate
        // pass.
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            int const ij = i * 3 + j;
            V const c3Fij = c3 * F[i][j];
            V const lambdaHatCofFij = lambdaHat * cofF[i][j];
            for (int k = 0; k < 3; ++k) {
              for (int l = 0; l < 3; ++l) {
                int const kl = k * 3 + l;
                if (kl >= ij) {
                  C[i][j][k][l] = c3Fij * F[k][l] + lambdaHatCofFij * cofF[k][l];
                }
              }
            }
            C[i][j][i][j] += c2;
          }
        }
        utils::BatchAddD2JdF2UpperTriangle<kBatchSize>(d2JCoeff, F, C);
      }
      utils::BatchMirrorTangentUpperToLower<kBatchSize>(C);
    }
  }
}

MOCHI_INSTANTIATE_BATCHED_MATERIAL(
    BatchedSmithNeoHookeanConstitutiveResponse,
    BatchLameParams,
    MaterialPsdOracle);

} // namespace mochi::materials
