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

#include <mochi_core/materials/batched_active_shape_targeting_arap.h>

#include <mochi_core/materials/material_params_utils.h>
#include <mochi_core/materials/material_utils.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/simd.h>

#include <limits>

namespace mochi::materials {

template <int kBatchSize>
void BatchedActiveShapeTargetingArapConstitutiveResponse(
    BatchActiveShapeTargetingArapParams<kBatchSize> const& params,
    BatchReal3x3<kBatchSize> const& F,
    BatchDouble<kBatchSize>* outEnergy,
    BatchReal3x3<kBatchSize>* outPK1,
    NdArray<BatchReal3x3<kBatchSize>, 3, 3>* outTangent,
    bool projectPsd) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  using V6 = BatchReal6<kBatchSize>;
  using V3x3 = BatchReal3x3<kBatchSize>;
  MOCHI_ASSERT_VERBOSE(
      IsFinite(params.stiffness) && AllTrue<kBatchSize>(params.stiffness > V{0_r}),
      "Active shape targeting ARAP stiffness must be positive and finite.");
  MOCHI_ASSERT_VERBOSE(
      IsResolvedPsdStrategySupported<ActiveShapeTargetingArapMaterialParams>(params.psdStrategy),
      "Batched material response requires a supported, resolved PSD strategy.");
  MaterialPsdStrategy const psdStrategy = params.psdStrategy;

  V const mu = params.stiffness;

  // Shape target: S_t = I + symmetric tensor from params.
  // clang-format off
  V3x3 const S_t = {
      V3{V{1_r} + params.shapeTargetTensor[0], params.shapeTargetTensor[1],          params.shapeTargetTensor[2]},
      V3{params.shapeTargetTensor[1],          V{1_r} + params.shapeTargetTensor[3], params.shapeTargetTensor[4]},
      V3{params.shapeTargetTensor[2],          params.shapeTargetTensor[4],          V{1_r} + params.shapeTargetTensor[5]}};
  // clang-format on

  // F_t = F * S_t
  V3x3 const F_t = Dot(F, S_t);

  // Compute the rotation between the deformation gradient and the shape target. Note that
  // BatchedRotationVariantSvd3x3 returns U, V^T and sigma such that det(U) >= 0, det(V^T) >= 0, and
  // sigma[2] < 0 if and only if det(F) < 0.
  V3x3 U_t MOCHI_NO_INIT, VT_t MOCHI_NO_INIT;
  V3 sigma_t MOCHI_NO_INIT;
  BatchedRotationVariantSvd3x3<kBatchSize>(F_t, U_t, sigma_t, VT_t);

  V3x3 const VT_t_S_t = Dot(VT_t, S_t);

  if (outEnergy || outPK1) {
    // R_t * S_t = U_t * (VT_t * S_t)
    V3x3 const R_t_S_t = Dot(U_t, VT_t_S_t);
    V3x3 const F_dev = F - R_t_S_t;

    if (outEnergy) {
      *outEnergy = StaticCast<Vd>(0.5_r * mu * NormSqr(F_dev));
    }

    if (outPK1) {
      *outPK1 = mu * F_dev;
    }
  }

  if (outTangent) {
    // Twist-mode coefficients: coeffs[n] = -μ / (σⱼ + σₖ)
    // Mode 0 pairs (1,2), mode 1 pairs (0,2), mode 2 pairs (0,1).
    V3 const coeffs = {
        -mu / (sigma_t[1] + sigma_t[2]),
        -mu / (sigma_t[0] + sigma_t[2]),
        -mu / (sigma_t[0] + sigma_t[1])};

    // Transpose U_t -> UT_t
    V3x3 const UT_t = Transpose(U_t);

    // Twist modes: N[n] = UT_t_row[j] ⊗ VT_t_S_t_row[k] - UT_t_row[k] ⊗ VT_t_S_t_row[j]
    V3x3 N[3] = {
        Outer(UT_t[2], VT_t_S_t[1]) - Outer(UT_t[1], VT_t_S_t[2]),
        Outer(UT_t[2], VT_t_S_t[0]) - Outer(UT_t[0], VT_t_S_t[2]),
        Outer(UT_t[1], VT_t_S_t[0]) - Outer(UT_t[0], VT_t_S_t[1])};

    auto& C = *outTangent;

    if (projectPsd && psdStrategy != MaterialPsdStrategy::None) {
      // PSD path: QR factorization of [N0, N1, N2] via modified Gram-Schmidt,
      // eigendecomp of R * Diag(coeffs) * R^T, clamp, reconstruct.

      V const min = std::numeric_limits<real>::min();

      // QR step 1: Q0
      V const r00 = Norm(N[0]);
      V const invR00 = V{1_r} / (r00 + min);
      V3x3 const Q0 = invR00 * N[0];

      // QR step 2: Q1
      V const r01 = Colon(Q0, N[1]);
      V3x3 Q1 = N[1] - r01 * Q0;
      V const r11 = Norm(Q1);
      V const invR11 = V{1_r} / (r11 + min);
      Q1 *= invR11;

      // QR step 3: Q2
      V const r02 = Colon(Q0, N[2]);
      V const r12 = Colon(Q1, N[2]);
      V3x3 Q2 = N[2] - r02 * Q0 - r12 * Q1;
      V const r22 = Norm(Q2);
      V const invR22 = V{1_r} / (r22 + min);
      Q2 *= invR22;

      // R_coeffs_RT = R * Diag(coeffs) * R^T, where R is upper triangular:
      //   R = [[r00, r01, r02],
      //        [  0, r11, r12],
      //        [  0,   0, r22]]
      // (R*D*R^T)[i][j] = sum_k R[i][k] * coeffs[k] * R[j][k]
      V const c0 = coeffs[0];
      V const c1 = coeffs[1];
      V const c2 = coeffs[2];
      V6 const sym = {
          r00 * r00 * c0 + r01 * r01 * c1 + r02 * r02 * c2, // (0,0)
          r11 * r11 * c1 + r12 * r12 * c2, // (1,1)
          r22 * r22 * c2, // (2,2)
          r01 * r11 * c1 + r02 * r12 * c2, // (0,1)
          r02 * r22 * c2, // (0,2)
          r12 * r22 * c2}; // (1,2)

      // Eigendecomp of R_coeffs_RT.
      V3 eigvals MOCHI_NO_INIT;
      V3x3 eigvecs MOCHI_NO_INIT;
      BatchedAnalyticalEigendecompSym3x3<kBatchSize>(sym, eigvals, &eigvecs);

      // Clamp eigenvalues so that C = μI + Q * Clamped(R*D*R^T) * Q^T is PSD.
      if (psdStrategy == MaterialPsdStrategy::AbsEigenProjection) {
        for (int k = 0; k < 3; ++k) {
          eigvals[k] = Max(Abs(eigvals[k] + mu), V{kMinProjectedEigenvalue}) - mu;
        }
      } else if (psdStrategy == MaterialPsdStrategy::Projection) {
        V const minEig = V{kMinProjectedEigenvalue - params.stiffness};
        for (int k = 0; k < 3; ++k) {
          eigvals[k] = Max(eigvals[k], minEig);
        }
      } else {
        MOCHI_ASSERT_VERBOSE(
            psdStrategy == MaterialPsdStrategy::PerTermProjection, "Unexpected PSD strategy.");
        V const zero = {};
        for (int k = 0; k < 3; ++k) {
          eigvals[k] = Max(eigvals[k], zero);
        }
      }

      // A = eigvecs^T * Diag(clamped_eigvals) * eigvecs
      V3x3 A MOCHI_NO_INIT;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          A[i][j] = eigvals[0] * eigvecs[0][i] * eigvecs[0][j] +
              eigvals[1] * eigvecs[1][i] * eigvecs[1][j] +
              eigvals[2] * eigvecs[2][i] * eigvecs[2][j];
        }
      }

      // Q_A columns: QA[k][i][j] = Q0[i][j]*A[0][k] + Q1[i][j]*A[1][k] + Q2[i][j]*A[2][k]
      V3x3 const QA[3] = {
          A[0][0] * Q0 + A[1][0] * Q1 + A[2][0] * Q2,
          A[0][1] * Q0 + A[1][1] * Q1 + A[2][1] * Q2,
          A[0][2] * Q0 + A[1][2] * Q1 + A[2][2] * Q2};

      // Tangent: C[i][j][k][l] = sum_n QA[n][i][j] * Q_n[k][l] + μ δᵢₖ δⱼₗ
      // Contraction strategy: hoist the 3 temporary products, mirror in a separate pass. Fastest
      // for 3-rank contractions on x86-64 (AVX2); within noise on ARM (NEON).
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          int const ij = i * 3 + j;
          V const a0 = QA[0][i][j];
          V const a1 = QA[1][i][j];
          V const a2 = QA[2][i][j];
          for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
              int const kl = k * 3 + l;
              if (kl >= ij) {
                C[i][j][k][l] = a0 * Q0[k][l] + a1 * Q1[k][l] + a2 * Q2[k][l];
              }
            }
          }
          C[i][j][i][j] += mu;
        }
      }
      utils::BatchMirrorTangentUpperToLower<kBatchSize>(C);

    } else {
      // Non-PSD path: C[i][j][k][l] = sum_n coeffs[n] * N[n][i][j] * N[n][k][l] + μ δᵢₖ δⱼₗ
      // Contraction strategy: hoist the 3 temporary products, mirror in a separate pass. Fastest
      // for 3-rank contractions on x86-64 (AVX2); within noise on ARM (NEON).
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          int const ij = i * 3 + j;
          V const s0 = coeffs[0] * N[0][i][j];
          V const s1 = coeffs[1] * N[1][i][j];
          V const s2 = coeffs[2] * N[2][i][j];
          for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
              int const kl = k * 3 + l;
              if (kl >= ij) {
                C[i][j][k][l] = s0 * N[0][k][l] + s1 * N[1][k][l] + s2 * N[2][k][l];
              }
            }
          }
          C[i][j][i][j] += mu;
        }
      }
      utils::BatchMirrorTangentUpperToLower<kBatchSize>(C);
    }
  }
}

MOCHI_INSTANTIATE_BATCHED_MATERIAL(
    BatchedActiveShapeTargetingArapConstitutiveResponse,
    BatchActiveShapeTargetingArapParams);

} // namespace mochi::materials
