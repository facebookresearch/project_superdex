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

#include <mochi_core/materials/batched_active_aniso_arap.h>

#include <mochi_core/materials/material_params_utils.h>
#include <mochi_core/materials/material_utils.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/simd.h>

#include <limits>

namespace mochi::materials {

template <int kBatchSize>
void BatchedActiveAnisoArapConstitutiveResponse(
    BatchActiveAnisoArapParams<kBatchSize> const& params,
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
      AllTrue<kBatchSize>(params.alpha >= V{0_r}) && AllTrue<kBatchSize>(params.length >= V{0_r}),
      "Invalid active aniso ARAP parameters.");
  MOCHI_ASSERT_VERBOSE(
      IsResolvedPsdStrategySupported<ActiveAnisoArapMaterialParams>(params.psdStrategy),
      "Batched material response requires a supported, resolved PSD strategy.");
  MaterialPsdStrategy const psdStrategy = params.psdStrategy;

  V const alpha = params.alpha;
  V const length = params.length;
  V3 const aDir = params.anisoDir;

  // Note that BatchedRotationVariantSvdValsAndVT3x3 returns V^T and sigma such that det(V^T) >= 0,
  // and sigma[2] < 0 if and only if det(F) < 0.
  V3x3 VT MOCHI_NO_INIT;
  V3 sigma MOCHI_NO_INIT;
  BatchedRotationVariantSvdValsAndVT3x3<kBatchSize>(F, sigma, VT);

  // Compute I4 = aᵀ VTᵀ Diag(σ) VT a efficiently via v = VT·a.
  V3 const VTa = DotMatVec(VT, aDir);
  V const I4 = Dot(sigma, VTa * VTa);

  // signumI4: per-lane sign of I4.
  V const zero = 0_r;
  V const one = 1_r;
  V const signumI4 = Select(I4 > zero, one, Select(I4 < zero, V{-1_r}, zero));

  // Fa = F * aDir
  V3 const Fa = DotMatVec(F, aDir);

  // I5 = ||Fa||^2
  V const I5 = NormSqr(Fa);

  // Safe denominator to avoid division by zero for degenerate lanes.
  V const safeI5 = Max(I5, V{std::numeric_limits<real>::min()});
  V const sqrtI5 = Sqrt(safeI5);

  // Validity mask: 1.0 for valid lanes, 0.0 for degenerate (I4==0 or I5==0).
  V const validLane = Select(VEqual(I4, zero), zero, one) * Select(I5 <= zero, zero, one);
  V const isValid = (validLane > zero);

  if (outEnergy) {
    V const diff = sqrtI5 - length * signumI4;
    V const energyV = 0.5_r * alpha * diff * diff * validLane;
    *outEnergy = StaticCast<Vd>(energyV);
  }

  if (outPK1 == nullptr && outTangent == nullptr) {
    return;
  }

  // FaaT = Outer(Fa, aDir)
  V3x3 const FaaT = Outer(Fa, aDir);

  // eigs12 = alpha * (1 - length * signumI4 / sqrtI5)
  V const eigs12 = alpha * (one - length * signumI4 / sqrtI5);

  // The first Piola--Kirchhoff.
  if (outPK1) {
    *outPK1 = (eigs12 * validLane) * FaaT;
  }

  // The tangent decomposes as:
  //
  //   H_{ijkl} = c · (Faaᵀ)_{ij}(Faaᵀ)_{kl} + λ₁₂ · δ_{ik}(aaᵀ)_{jl}
  //
  // where Faaᵀ := (Fa)⊗a (outer product), c is a scalar coefficient and λ₁₂ = eigs12.
  // Since ‖Faaᵀ‖² = I₅ (for unit a), the 9 eigenvalues are:
  //
  //   λ₀       = c · I₅ + λ₁₂ = α   (along Fa ⊗ a)
  //   λ₁ = λ₂  = λ₁₂                 (fiber, ⊥ Fa)
  //   λ₃…λ₈   = 0                     (non-fiber)
  //
  // Only λ₁₂ can become negative. PSD projection modifies λ₁₂ while preserving
  // λ₀ = α by recomputing c = (α − λ₁₂') / I₅.
  //
  // Note: The references [Kim 2019, Kim & Eberle 2022] use Ψ = α(√I₅ − ℓ·sgn(I₄))² (no ½
  // factor), so their PK1 and tangent expressions differ from ours by a factor of 2. There also
  // appear to be errors in the eigenvalue formulas of [Kim 2019]. Our expressions were
  // independently derived and verified with unit tests.
  if (outTangent) {
    // c = αℓs / I₅^{3/2} so that c·I₅ + λ₁₂ = αℓs/√I₅ + α(1 − ℓs/√I₅) = α.
    V const stdCoeff = alpha * length * signumI4 / (safeI5 * sqrtI5);
    V const stdEigsVal = eigs12;

    V coeff MOCHI_NO_INIT, eigsVal MOCHI_NO_INIT;
    if (projectPsd && psdStrategy != MaterialPsdStrategy::None) {
      auto const needsProj = eigs12 <= zero;

      if (psdStrategy == MaterialPsdStrategy::AbsEigenProjection) {
        // λ₁₂' = max(|λ₁₂|, ε);  c = (α − λ₁₂') / I₅  so that  c·I₅ + λ₁₂' = α.
        V const absEigs12 = Max(-eigs12, V{kMinProjectedEigenvalue});
        V const projCoeff = (alpha - absEigs12) / safeI5;
        coeff = Select(needsProj, projCoeff, stdCoeff);
        eigsVal = Select(needsProj, absEigs12, stdEigsVal);
      } else {
        MOCHI_ASSERT_VERBOSE(
            psdStrategy == MaterialPsdStrategy::Projection, "Unexpected PSD strategy.");
        // λ₁₂' = 0;  c = α / I₅  so that  c·I₅ = α.
        V const projCoeff = alpha / safeI5;
        coeff = Select(needsProj, projCoeff, stdCoeff);
        eigsVal = Select(needsProj, zero, stdEigsVal);
      }
    } else {
      coeff = stdCoeff;
      eigsVal = stdEigsVal;
    }

    // Zero out for degenerate lanes.
    coeff = Select(isValid, coeff, zero);
    eigsVal = Select(isValid, eigsVal, zero);

    // diagBlock[j][l] = eigsVal * a_j * a_l
    V3x3 const diagBlock = Outer(eigsVal * aDir, aDir);

    // Assemble tangent. Exploit symmetry: compute upper triangle, then mirror.
    // Contraction strategy: hoist the temporary product, mirror in a separate pass. Fastest for
    // 1-rank contractions on x86-64 (AVX2) and ARM (NEON).
    auto& C = *outTangent;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        int const ij = i * 3 + j;
        V const cv = coeff * FaaT[i][j];
        for (int k = 0; k < 3; ++k) {
          for (int l = 0; l < 3; ++l) {
            int const kl = k * 3 + l;
            if (kl >= ij) {
              C[i][j][k][l] = cv * FaaT[k][l];
            }
          }
        }
      }
    }
    utils::BatchMirrorTangentUpperToLower<kBatchSize>(C);

    // Add diagonal block: C[i][j][i][l] += eigsVal * a[j] * a[l]
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int l = 0; l < 3; ++l) {
          C[i][j][i][l] += diagBlock[j][l];
        }
      }
    }
  }
}

MOCHI_INSTANTIATE_BATCHED_MATERIAL(
    BatchedActiveAnisoArapConstitutiveResponse,
    BatchActiveAnisoArapParams);

} // namespace mochi::materials
