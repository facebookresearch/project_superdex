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
#include <mochi_core/materials/active_shape_targeting_arap_params.h>
#include <mochi_core/materials/arap_params.h>
#include <mochi_core/materials/linear_elastic_params.h>
#include <mochi_core/materials/material_types.h>
#include <mochi_core/materials/smith_neo_hookean_params.h>
#include <mochi_core/materials/st_venant_kirchhoff_params.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/vmatrix.h>

namespace mochi::materials::utils {

/**
 * @brief Compile-time material-type discriminant for a top-level material-parameter struct.
 *
 * @details Maps each user-selectable material-parameter struct to its @ref SoftMaterialType tag via
 * @c kType, keeping the structs themselves as pure data. Only the material types enumerated in
 * @ref SoftMaterialType are specialized; the primary template is intentionally left undefined.
 */
template <typename ParamsType>
struct MaterialTraits;

template <>
struct MaterialTraits<SmithNeoHookeanMaterialParams> {
  static constexpr SoftMaterialType kType = SoftMaterialType::NeoHookean;
};

template <>
struct MaterialTraits<StVenantKirchhoffMaterialParams> {
  static constexpr SoftMaterialType kType = SoftMaterialType::StVenantKirchhoff;
};

template <>
struct MaterialTraits<LinearElasticMaterialParams> {
  static constexpr SoftMaterialType kType = SoftMaterialType::LinearElastic;
};

template <>
struct MaterialTraits<ActiveNeoHookeanMaterialParams> {
  static constexpr SoftMaterialType kType = SoftMaterialType::ActiveNeoHookean;
};

template <>
struct MaterialTraits<ActiveShapeTargetingArapMaterialParams> {
  static constexpr SoftMaterialType kType = SoftMaterialType::ActiveShapeTargetingArap;
};

template <>
struct MaterialTraits<ArapMaterialParams> {
  static constexpr SoftMaterialType kType = SoftMaterialType::Arap;
};

/** @brief Resolve a user-facing material PSD strategy to a runtime strategy. */
template <typename T>
MOCHI_FORCE_INLINE constexpr MaterialPsdStrategy ResolvePsdStrategy(T const& p) {
  if constexpr (requires { p.psdStrategy; }) {
    // Resolve MaterialDefault to the material's own default: its psdStrategy field initializer.
    // That initializer must itself be a concrete strategy, otherwise MaterialDefault would resolve
    // to MaterialDefault and defeat the resolution.
    static_assert(
        T{}.psdStrategy != MaterialPsdStrategy::MaterialDefault,
        "A material's default psdStrategy must be a concrete strategy, not MaterialDefault.");
    return (p.psdStrategy == MaterialPsdStrategy::MaterialDefault) ? T{}.psdStrategy
                                                                   : p.psdStrategy;
  } else {
    return MaterialPsdStrategy::None;
  }
}

/**
 * @brief Assembles the 9x9 Hessian (aka tangent) from its eigenvalues, the SVD of the deformation
 * gradient, and the eigendecomposition of the so-called scaling mode matrix.
 *
 * @details The Hessian is computed as sum_{i=1 to 9} lambda_i * q_i * q_i^T, where the 9
 * eigenvalues (lambda_i) and eigenvectors (q_i) are formed from three distinct groups:
 *
 * 1.  Scaling Modes (3): Capture infinitesimal stretch/compression changes.
 * 2.  Twist Modes (3): Capture infinitesimal rotational changes.
 * 3.  Flip Modes (3): Capture infinitesimal shear/reflection changes along the principal
 *     stretch axes.
 *
 * @param[in] lambda Eigenvalues of the output Hessian in the following order: [scaling_0,
 * scaling_1, scaling_2, twist_0, twist_1, twist_2, flip_0, flip_1, flip_2].
 * @param[in] U 3x3 matrix U from the SVD of the deformation gradient (F = U * Sigma * V^T).
 * @param[in] VT 3x3 matrix V^T from the SVD of the deformation gradient (F = U * Sigma * V^T).
 * @param[in] AqT 3x3 matrix Aq^T from the eigendecomposition of the (symmetric) scaling mode matrix
 * (A = Aq * Lambda * Aq^T).
 * @param[out] outTangent The resulting 9x9 Hessian (aka tangent).
 *
 * @note The paper by Smith et al. stores matrices and tensors in column-major order. This function
 * stores matrices and tensors in row-major order.
 *
 * @see [Analytic Eigensystems for Isotropic Distortion Energies (Smith et al.,
 * 2018)](https://www.tkim.graphics/EIGENSYSTEMS/AnalyticEigensystems.pdf)
 */
MOCHI_FORCE_INLINE void AssembleTangentFromEigensystem(
    NdArray<real, 9> const& lambda,
    VMatrix3x3r const& U,
    VMatrix3x3r const& VT,
    VMatrix3x3r const& AqT,
    VTensor3x3x3x3r* outTangent) {
  VTensor3x3x3x3r QT; // Eigenvectors of the Hessian.

  // Scaling eigenvectors.
  QT[0][0] = Dot3x3(U, VMatrix3x3r{AqT[0][0] * VT[0], AqT[0][1] * VT[1], AqT[0][2] * VT[2]});
  QT[0][1] = Dot3x3(U, VMatrix3x3r{AqT[1][0] * VT[0], AqT[1][1] * VT[1], AqT[1][2] * VT[2]});
  QT[0][2] = Dot3x3(U, VMatrix3x3r{AqT[2][0] * VT[0], AqT[2][1] * VT[1], AqT[2][2] * VT[2]});

  real constexpr kSqrt2Inv = 1_r / kSqrt2;
  VMatrix3x3r const UT = kSqrt2Inv * Transpose3x3(U); // Bake normalization into U^T to save FLOPs.
  VMatrix3x3r const S12 = Outer3(UT[2], VT[1]);
  VMatrix3x3r const S02 = Outer3(UT[2], VT[0]);
  VMatrix3x3r const S01 = Outer3(UT[1], VT[0]);
  VMatrix3x3r const S21 = Outer3(UT[1], VT[2]);
  VMatrix3x3r const S20 = Outer3(UT[0], VT[2]);
  VMatrix3x3r const S10 = Outer3(UT[0], VT[1]);

  // Twist eigenvectors.
  QT[1][0] = S12 - S21;
  QT[1][1] = S02 - S20;
  QT[1][2] = S01 - S10;

  // Flip eigenvectors.
  QT[2][0] = S12 + S21;
  QT[2][1] = S02 + S20;
  QT[2][2] = S01 + S10;

  // NOTE:
  // - The order of the loops below is empirically fastest (both on x86 and ARM), possibly due to
  //   reduced register spilling.
  // - On AVX2, this tensor contraction is slightly faster if implemented via 9x9 matrix-matrix
  //   product due to better SIMD utilization. This may be even more so on AVX-512.
  *outTangent = {};
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      for (int l = 0; l < 3; ++l) {
        auto const lambda_QT = QT[k][l] * lambda[3 * k + l];
        for (int j = 0; j < 3; ++j) {
          (*outTangent)[i][j] += QT[k][l][i][j] * lambda_QT;
        }
      }
    }
  }
}

/**************************************************************************************************
  Batched Utilities
*/

/// @brief Add coeff · d²J/dF² to the upper triangle of tangent C (entries C[ij][kl] with ij ≤ kl).
///
/// @param[in] coeff  Scalar coefficient multiplying d²J/dF².
/// @param[in] F      Deformation gradient (3x3, row-major).
/// @param[in,out] C  3x3x3x3 tangent tensor. Only the upper triangle (ij ≤ kl in flattened
///                   indexing) is updated.
///
/// @note Only writes the upper triangle. The caller is responsible for mirroring.
/// @see Computed2JdF2 in kim_neo_hookean.cpp for the scalar derivation using Skew3.
template <int kBatchSize>
MOCHI_FORCE_INLINE void BatchAddD2JdF2UpperTriangle(
    BatchReal<kBatchSize> const& coeff,
    BatchReal3x3<kBatchSize> const& F,
    NdArray<BatchReal3x3<kBatchSize>, 3, 3>& C) {
  // Precompute scaled F entries (empirically faster).
  BatchReal3x3<kBatchSize> const s = coeff * F;

  // Block (0,1): -skew(row2), C[0][k][1][l]
  C[0][0][1][1] += s[2][2];
  C[0][0][1][2] -= s[2][1];
  C[0][1][1][0] -= s[2][2];
  C[0][1][1][2] += s[2][0];
  C[0][2][1][0] += s[2][1];
  C[0][2][1][1] -= s[2][0];

  // Block (0,2): +skew(row1), C[0][k][2][l]
  C[0][0][2][1] -= s[1][2];
  C[0][0][2][2] += s[1][1];
  C[0][1][2][0] += s[1][2];
  C[0][1][2][2] -= s[1][0];
  C[0][2][2][0] -= s[1][1];
  C[0][2][2][1] += s[1][0];

  // Block (1,2): -skew(row0), C[1][k][2][l]
  C[1][0][2][1] += s[0][2];
  C[1][0][2][2] -= s[0][1];
  C[1][1][2][0] -= s[0][2];
  C[1][1][2][2] += s[0][0];
  C[1][2][2][0] += s[0][1];
  C[1][2][2][1] -= s[0][0];
}

/**
 * @brief Mirror the upper triangle of a symmetric 3x3x3x3 tangent tensor to the lower triangle.
 *
 * @param[in,out] C  3x3x3x3 tangent tensor with upper triangle filled at input.
 */
template <int kBatchSize>
MOCHI_FORCE_INLINE void BatchMirrorTangentUpperToLower(NdArray<BatchReal3x3<kBatchSize>, 3, 3>& C) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      int const ij = i * 3 + j;
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          int const kl = k * 3 + l;
          if (kl > ij) {
            C[k][l][i][j] = C[i][j][k][l];
          }
        }
      }
    }
  }
}

/**
 * @brief Batched implementation of @ref AssembleTangentFromEigensystem.
 *
 * @see AssembleTangentFromEigensystem
 * @see [Analytic Eigensystems for Isotropic Distortion Energies (Smith et al.,
 * 2018)](https://www.tkim.graphics/EIGENSYSTEMS/AnalyticEigensystems.pdf)
 */
template <int kBatchSize>
MOCHI_FORCE_INLINE void BatchedAssembleTangentFromEigensystem(
    BatchReal9<kBatchSize> const& lambda,
    BatchReal3x3<kBatchSize> const& U,
    BatchReal3x3<kBatchSize> const& VT,
    BatchReal3x3<kBatchSize> const& AqT,
    NdArray<BatchReal3x3<kBatchSize>, 3, 3>& outTangent) {
  using V = BatchReal<kBatchSize>;
  using V9 = BatchReal9<kBatchSize>;
  using V3x3 = BatchReal3x3<kBatchSize>;

  // Scaling eigenvectors.
  V3x3 const Q[3] = {
      Dot(U, V3x3{AqT[0][0] * VT[0], AqT[0][1] * VT[1], AqT[0][2] * VT[2]}),
      Dot(U, V3x3{AqT[1][0] * VT[0], AqT[1][1] * VT[1], AqT[1][2] * VT[2]}),
      Dot(U, V3x3{AqT[2][0] * VT[0], AqT[2][1] * VT[1], AqT[2][2] * VT[2]})};

  // Twist and flip eigenvectors.
  constexpr int kPairs[3][2] = {{1, 2}, {0, 2}, {0, 1}};
  V3x3 T[3] MOCHI_NO_INIT, Fl[3] MOCHI_NO_INIT;
  for (int n = 0; n < 3; ++n) {
    int const j = kPairs[n][0];
    int const k = kPairs[n][1];
    for (int r = 0; r < 3; ++r) {
      V const uj = U[r][j];
      V const uk = U[r][k];
      for (int c = 0; c < 3; ++c) {
        V const a = uj * VT[k][c];
        V const b = uk * VT[j][c];
        T[n][r][c] = a - b;
        Fl[n][r][c] = a + b;
      }
    }
  }

  V const half{0.5_r};
  V9 const sl = {
      lambda[0],
      lambda[1],
      lambda[2],
      half * lambda[3],
      half * lambda[4],
      half * lambda[5],
      half * lambda[6],
      half * lambda[7],
      half * lambda[8]};

  // Contraction strategy: hoist the 9 temporary products, interleave mirror writes inside the inner
  // loop. Fastest for 9-rank contractions on ARM (NEON), within noise on x86-64 (AVX2).
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      int const ij = i * 3 + j;
      V const sq0 = sl[0] * Q[0][i][j];
      V const sq1 = sl[1] * Q[1][i][j];
      V const sq2 = sl[2] * Q[2][i][j];
      V const st0 = sl[3] * T[0][i][j];
      V const st1 = sl[4] * T[1][i][j];
      V const st2 = sl[5] * T[2][i][j];
      V const sf0 = sl[6] * Fl[0][i][j];
      V const sf1 = sl[7] * Fl[1][i][j];
      V const sf2 = sl[8] * Fl[2][i][j];

      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          int const kl = k * 3 + l;
          if (kl >= ij) {
            outTangent[i][j][k][l] = sq0 * Q[0][k][l] + sq1 * Q[1][k][l] + sq2 * Q[2][k][l] +
                st0 * T[0][k][l] + st1 * T[1][k][l] + st2 * T[2][k][l] + sf0 * Fl[0][k][l] +
                sf1 * Fl[1][k][l] + sf2 * Fl[2][k][l];
            if (kl != ij) {
              outTangent[k][l][i][j] = outTangent[i][j][k][l];
            }
          }
        }
      }
    }
  }
}

struct LameConstants {
  real lambda = 0_r; ///< First Lamé parameter [Pa].
  real mu = 0_r; ///< Second Lamé parameter (shear modulus) [Pa].
};

/// @brief Compute Lamé constants (λ, μ) from Young's modulus and Poisson ratio.
///
/// @param[in] youngsModulus Young's modulus [Pa].
/// @param[in] poissonRatio Poisson ratio [-].
MOCHI_FORCE_INLINE constexpr LameConstants ComputeLameConstants(
    real youngsModulus,
    real poissonRatio) {
  MOCHI_ASSERT_VERBOSE(youngsModulus > 0_r, "Invalid Young's modulus.");
  MOCHI_ASSERT_VERBOSE(poissonRatio > -1_r && poissonRatio < 0.5_r, "Invalid Poisson ratio.");
  return {
      .lambda = youngsModulus * poissonRatio / ((1_r + poissonRatio) * (1_r - 2_r * poissonRatio)),
      .mu = youngsModulus / (2_r * (1_r + poissonRatio))};
}

} // namespace mochi::materials::utils
