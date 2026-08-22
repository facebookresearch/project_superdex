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

#include <mochi_core/element_operations/batched_element_utils.h>
#include <mochi_core/materials/batched_materials.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

namespace mochi::fem {

/// @brief Compute stiffness-damping (viscous) stress work for a batch of tetrahedral elements.
///
/// @details Adds a strain-rate-proportional viscous contribution to a soft (3D FEM) material
/// response, formulated in the total-Lagrangian (second Piola–Kirchhoff / Green–Lagrange)
/// description. The viscous second Piola–Kirchhoff stress is
/// @code
///   S_visc = κ · C₀ : ΔE,   ΔE = E(F) − E(F_stageStart),   E = ½(FᵀF − I)
/// @endcode
/// where `C₀ = ∂S/∂E|₀` is the Lagrangian zero-deformation stiffness and `κ =
/// stiffnessDampingFactor = β/dtStage`. The viscous term is a separate additive contribution with
/// its own residual and tangent, kept distinct from the elastic response because the soft material
/// is nonlinear (the viscous stiffness `C₀` is fixed at zero deformation, not the deformed-state
/// elastic tangent).
///
/// Because `C₀` is constant (deformation-independent), it is precomputed once at material-set time
/// (via @ref materials::ComputeReferenceMaterialStiffnessVoigt) and passed in as a symmetric 6×6
/// Voigt matrix; all per-quadrature contractions then use that symmetric form and the
/// Green–Lagrange strain-displacement matrix `B` (the standard total-Lagrangian operators):
/// @code
///   residual:    Rᶠ      = wᵠ · Bᶠᵀ · S_visc                       (S_visc = κ · C₀ᵥ · ΔEᵥ)
///   material K:  Kᶠᵍ_mat = wᵠ · κ · Bᶠᵀ · C₀ᵥ · Bᵍ
///   geometric K: Kᶠᵍ_geo = wᵠ · (∇Nᶠ · S_visc · ∇Nᵍ) · I₃         (added on the i==j diagonal)
/// @endcode
/// The geometric block is optional (see @p includeGeometricStiffness): the material block is SPD
/// for a stable `C₀`, while the geometric block can be indefinite and is proportional to the
/// per-stage strain increment `ΔE`, so it vanishes at steady state. Gating it off yields a
/// modified-Newton tangent (material block only) that keeps the assembled tangent SPD and leaves
/// the residual and energy unchanged.
/// Voigt ordering is [00, 11, 22, 12, 02, 01]; strains use engineering shear (off-diagonals
/// doubled) and stresses do not, so `S:E = Sᵥ·Eᵥ` and `C₀ᵥ[a][b] = C₀_{i_a j_a i_b j_b}`.
///
/// The operators above are the mathematical definition. `B` is materialized only when it is
/// actually needed for the dense tangent (see @p materialIsIsotropic). Otherwise the implementation
/// evaluates the algebraically equivalent `Rᶠᵢ = wᵠ · Pᵢⱼ · ∂Nᶠ/∂Xⱼ` with `P = F·S_visc`, plus a
/// closed-form isotropic tangent.
///
/// @param[in] elementIndices  Indices into @p elements for each batch lane.
/// @param[in] elements  Element data array (tetrahedral, 4-node).
/// @param[in] disp  Batched current displacement DoF vector.
/// @param[in] stageStartDisp  Batched stage-start displacement DoF vector. Only read when the
///   stage-start strain increment is needed (energy, residual, or geometric stiffness); it is left
///   untouched in a dresidual-only assembly with @p includeGeometricStiffness false, so callers may
///   leave it uninitialized in that mode.
/// @param[in,out] outEnergy  If non-null, accumulates per-element viscous energy.
/// @param[in,out] outRes  If non-null, accumulates per-element viscous residual.
/// @param[in,out] outDRes  If non-null, accumulates per-element viscous stiffness.
/// @param[in] projectPsd  If true, project the symmetric `S_visc` to be positive semi-definite
///   before it enters the geometric stiffness, keeping the assembled tangent positive
///   semi-definite. Has no effect when @p includeGeometricStiffness is false, since `S_visc` is
///   used nowhere else in the tangent.
/// @param[in] includeGeometricStiffness  If true, add the geometric block
///   `Kᶠᵍ_geo = wᵠ · (∇Nᶠ · S_visc · ∇Nᵍ) · I₃` to the tangent, yielding the exact viscous
///   derivative of @p outRes. If false, the geometric block (and its `S_visc` construction + PSD
///   projection) is skipped and @p outDRes holds only the SPD material block `wᵠ · κ · Bᶠᵀ · C₀ᵥ ·
///   Bᵍ`; this is a modified-Newton (quasi-Newton) tangent that is NOT the exact derivative of
///   @p outRes. @p outEnergy and @p outRes are unaffected by this flag.
/// @param[in] stiffnessDampingFactor  κ = β/dtStage [dimensionless]. Must be > 0.
/// @param[in] referenceMaterialStiffnessVoigt  Precomputed reference material stiffness `C₀`
///   (symmetric 6×6 Voigt) for each batch lane. Deformation-independent; built once at material-set
///   time via @ref materials::ComputeReferenceMaterialStiffnessVoigt.
/// @param[in] materialIsIsotropic  If true, `C₀ᵥ` is the isotropic 2-parameter (λ,μ) Voigt form
///   (dense normal-normal block, `μ·I` shear block, zero normal-shear coupling), and the `S_visc`
///   and `C₀ᵥ·B` contractions use the sparser isotropic formulas (read `λ = C₀ᵥ[0][1]`, `μ =
///   C₀ᵥ[3][3]` once per batch) instead of dense 6×6 mat-vecs. Must only be set when every lane's
///   `C₀ᵥ` truly has this structure (see @ref materials::kIsotropicReferenceStiffness); otherwise
///   use false for the exact dense fallback. Both paths are numerically equivalent for an isotropic
///   `C₀ᵥ`. Per-lane `λ,μ` are read from the gathered tensor, so heterogeneous-but-all-isotropic
///   materials are handled correctly.
/// @param[in] perElementExtraWeight  Optional per-element quadrature weight multiplier.
/// @return true if outputs were written.
template <int kBatchSize, class ElementT>
bool StressDampingWork(
    NdArray<int, kBatchSize> const& elementIndices,
    Span<ElementT const> elements,
    BatchElementVector<kBatchSize, ElementT> const& disp,
    BatchElementVector<kBatchSize, ElementT> const& stageStartDisp,
    BatchDouble<kBatchSize>* outEnergy,
    BatchElementVector<kBatchSize, ElementT>* outRes,
    BatchElementMatrix<kBatchSize, ElementT>* outDRes,
    bool projectPsd,
    bool includeGeometricStiffness,
    real stiffnessDampingFactor,
    NdArray<BatchReal<kBatchSize>, 6, 6> const& referenceMaterialStiffnessVoigt,
    bool materialIsIsotropic,
    Span<real const> perElementExtraWeight = {}) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
  using V3x3 = BatchReal3x3<kBatchSize>;
  MOCHI_ASSERT_VERBOSE(!elements.empty(), "Elements span is empty.");
  MOCHI_ASSERT_VERBOSE(
      Min(elementIndices) >= 0 && Max(elementIndices) < isize(elements),
      "Element index out of range.");
  MOCHI_ASSERT_VERBOSE(
      perElementExtraWeight.empty() || (perElementExtraWeight.size() == elements.size()),
      "Inconsistent sizes.");
  MOCHI_ASSERT_VERBOSE(stiffnessDampingFactor > 0_r, "Expected positive stiffnessDampingFactor.");

  constexpr auto kSpaceDim = ElementT::kSpaceDim;
  constexpr auto kNumNodes = ElementT::kNumDofs;
  constexpr auto kNumQuads = ElementT::kNumQuadPoints;
  constexpr auto kNumDofs = kNumNodes * kSpaceDim;
  static_assert(kSpaceDim == 3, "Only 3D elements are supported.");
  static_assert(kNumNodes == 4, "StressDampingWork supports tetrahedral (4-node) elements only.");

  // Voigt index map [00, 11, 22, 12, 02, 01].
  constexpr int kVi[6] = {0, 1, 2, 1, 0, 0};
  constexpr int kVj[6] = {0, 1, 2, 2, 2, 1};

  bool const evalObj = (outEnergy != nullptr);
  bool const evalRes = (outRes != nullptr);
  bool const evalDRes = (outDRes != nullptr);
  MOCHI_ASSERT_VERBOSE(evalObj || evalRes || evalDRes, "Must assemble something.");

  V const kappa{stiffnessDampingFactor};
  auto const extraWeight = ComputeExtraWeights(elementIndices, perElementExtraWeight);

  // Reference (zero-deformation) Lagrangian stiffness as a symmetric 6×6 Voigt matrix,
  // precomputed at material-set time and passed in (see
  // @ref materials::ComputeReferenceMaterialStiffnessVoigt).
  NdArray<V, 6, 6> const& C0v = referenceMaterialStiffnessVoigt;

  // Isotropic fast-path scalars (per-lane), read once from the isotropic C₀ᵥ template: λ =
  // C₀ᵥ[0][1] (a normal off-diagonal), μ = C₀ᵥ[3][3] (a shear diagonal). Only read/used when
  // materialIsIsotropic. κ is folded in so the contractions produce κ·C₀ᵥ·(·) directly.
  V kappaLambda MOCHI_NO_INIT;
  V kappa2Mu MOCHI_NO_INIT;
  V kappaMu MOCHI_NO_INIT;
  if (materialIsIsotropic) {
    kappaLambda = kappa * C0v[0][1];
    kappaMu = kappa * C0v[3][3];
    kappa2Mu = kappaMu + kappaMu;
  }

  // Staging buffer for packing element geometry data.
  alignas(alignof(V)) real staging[V::kSize]{};

  for (int q = 0; q < kNumQuads; ++q) {
    // Pack reference basis gradients ∇N and quad weight.
    NdArray<V, kNumNodes * kSpaceDim> dbasis MOCHI_NO_INIT;
    V quadWeight MOCHI_NO_INIT;
    for (int d = 0; d < kNumNodes * kSpaceDim; ++d) {
      for (int b = 0; b < kBatchSize; ++b) {
        staging[b] = elements[elementIndices[b]].dBasisEvaluated[q][d / kSpaceDim][d % kSpaceDim];
      }
      dbasis[d] = Load<V>(staging);
    }
    for (int b = 0; b < kBatchSize; ++b) {
      staging[b] = elements[elementIndices[b]].quadWeights[q] * extraWeight[b];
    }
    quadWeight = Load<V>(staging);

    // Current deformation gradient F[r][c] = δ_rc + Σ_f disp[f*3+r]·dbasis[f*3+c],
    // used by the direct residual path and by B.
    V3x3 F = Eye<3, V>();
    for (int f = 0; f < kNumNodes; ++f) {
      for (int r = 0; r < kSpaceDim; ++r) {
        V const u_fr = disp[f * kSpaceDim + r];
        for (int c = 0; c < kSpaceDim; ++c) {
          F[r][c] += u_fr * dbasis[f * kSpaceDim + c];
        }
      }
    }

    // Stage-start strain work: the strain increment ΔE and the viscous PK2 S_visc feed the energy,
    // the residual, and the geometric stiffness term, but NOT the SPD material tangent block
    // (Bᵀ·κ·C₀·B, which depends only on C₀ and B). So in a dresidual-only assembly with the
    // geometric term disabled they are all dead: skip them entirely — along with the stage-start
    // displacement they read — in that mode.
    bool const needStrainWork = evalObj || evalRes || includeGeometricStiffness;

    NdArray<V, 6> Sv MOCHI_NO_INIT; // Viscous PK2 in Voigt; only computed/read when needStrainWork.
    if (needStrainWork) {
      // Stage-start right Cauchy–Green tensor Css in Voigt, accumulated one Fss row at a time so
      // the full Fss 3×3 is never materialized. Css[k][l] = Σ_p Fss[p][k]·Fss[p][l].
      NdArray<V, 6> Css{};
      for (int p = 0; p < kSpaceDim; ++p) {
        V fssRow[3] MOCHI_NO_INIT;
        for (int c = 0; c < kSpaceDim; ++c) {
          fssRow[c] = (c == p) ? V{1_r} : V{0_r}; // identity row δ_pc
        }
        for (int f = 0; f < kNumNodes; ++f) {
          V const uss_fp = stageStartDisp[f * kSpaceDim + p];
          for (int c = 0; c < kSpaceDim; ++c) {
            fssRow[c] += uss_fp * dbasis[f * kSpaceDim + c];
          }
        }
        for (int a = 0; a < 6; ++a) {
          Css[a] += fssRow[kVi[a]] * fssRow[kVj[a]];
        }
      }

      // ΔE = E(F) − E(F_ss) = ½(FᵀF − F_ssᵀF_ss), in Voigt with engineering shear (off-diagonals
      // doubled). The identity terms cancel.
      NdArray<V, 6> dEv MOCHI_NO_INIT;
      for (int a = 0; a < 6; ++a) {
        int const k = kVi[a];
        int const l = kVj[a];
        V acc{0_r};
        for (int p = 0; p < 3; ++p) {
          acc += F[p][k] * F[p][l];
        }
        acc -= Css[a];
        dEv[a] = (a < 3) ? (V{0.5_r} * acc) : acc; // off-diagonals: 2·(½·acc) = acc
      }

      // Viscous PK2 in Voigt: S_visc = κ · C₀ᵥ · ΔEᵥ (no shear doubling on stress).
      if (materialIsIsotropic) {
        // Isotropic C₀ᵥ: normal-normal block is (λ + [i==j]·2μ), shear block is μ·I, coupling zero.
        // Reuse the strain trace so the 6×6 mat-vec collapses to a few multiply-adds.
        V const tr = dEv[0] + dEv[1] + dEv[2];
        V const kappaLambdaTr = kappaLambda * tr;
        for (int a = 0; a < 3; ++a) {
          Sv[a] = kappaLambdaTr + kappa2Mu * dEv[a];
        }
        for (int a = 3; a < 6; ++a) {
          Sv[a] = kappaMu * dEv[a];
        }
      } else {
        for (int a = 0; a < 6; ++a) {
          V acc{0_r};
          for (int b = 0; b < 6; ++b) {
            acc += C0v[a][b] * dEv[b];
          }
          Sv[a] = kappa * acc;
        }
      }

      if (evalObj) {
        // Ψ_visc = ½ (ΔE : S_visc) = ½ Σ_a Sᵥ[a]·ΔEᵥ[a].
        V psi{0_r};
        for (int a = 0; a < 6; ++a) {
          psi += Sv[a] * dEv[a];
        }
        *outEnergy += StaticCast<Vd>(V{0.5_r} * psi) * StaticCast<Vd>(quadWeight);
      }
    }

    if (!evalRes && !evalDRes) {
      continue;
    }

    // The isotropic tangent does not require B, so both residual-only and combined isotropic calls
    // can use the direct residual path: Rᶠᵢ = wᵠ · Pᵢⱼ · ∂Nᶠ/∂Xⱼ with P = F·S_visc.
    if (evalRes && (!evalDRes || materialIsIsotropic)) {
      V3x3 P MOCHI_NO_INIT;
      for (int i = 0; i < kSpaceDim; ++i) {
        P[i][0] = F[i][0] * Sv[0] + F[i][1] * Sv[5] + F[i][2] * Sv[4];
        P[i][1] = F[i][0] * Sv[5] + F[i][1] * Sv[1] + F[i][2] * Sv[3];
        P[i][2] = F[i][0] * Sv[4] + F[i][1] * Sv[3] + F[i][2] * Sv[2];
      }
      for (int f = 0; f < kNumNodes; ++f) {
        V const db0 = dbasis[f * kSpaceDim + 0];
        V const db1 = dbasis[f * kSpaceDim + 1];
        V const db2 = dbasis[f * kSpaceDim + 2];
        for (int i = 0; i < kSpaceDim; ++i) {
          V const acc = P[i][0] * db0 + P[i][1] * db1 + P[i][2] * db2;
          (*outRes)[f * kSpaceDim + i] += quadWeight * acc;
        }
      }
    }

    if (!evalDRes) {
      continue;
    }

    NdArray<V, kNumNodes, 6, 3> B MOCHI_NO_INIT;
    if (!materialIsIsotropic) {
      // Green–Lagrange strain-displacement matrix B[f][a][i] = ∂(Voigt strain a)/∂u[f][i], with
      // engineering shear (off-diagonal rows doubled), shared by the residual and dense material
      // tangent.
      for (int f = 0; f < kNumNodes; ++f) {
        V const db0 = dbasis[f * 3 + 0];
        V const db1 = dbasis[f * 3 + 1];
        V const db2 = dbasis[f * 3 + 2];
        for (int i = 0; i < 3; ++i) {
          V const fi0 = F[i][0];
          V const fi1 = F[i][1];
          V const fi2 = F[i][2];
          B[f][0][i] = fi0 * db0;
          B[f][1][i] = fi1 * db1;
          B[f][2][i] = fi2 * db2;
          B[f][3][i] = fi1 * db2 + fi2 * db1;
          B[f][4][i] = fi0 * db2 + fi2 * db0;
          B[f][5][i] = fi0 * db1 + fi1 * db0;
        }
      }

      if (evalRes) {
        // Residual: Rᶠ = wᵠ · Bᶠᵀ · S_visc.
        for (int f = 0; f < kNumNodes; ++f) {
          for (int i = 0; i < kSpaceDim; ++i) {
            V acc{0_r};
            for (int a = 0; a < 6; ++a) {
              acc += B[f][a][i] * Sv[a];
            }
            (*outRes)[f * kSpaceDim + i] += quadWeight * acc;
          }
        }
      }
    }

    // Geometric stiffness uses the (optionally PSD-projected) symmetric PK2. When the geometric
    // term is gated off, this 3x3 materialization and PSD projection are skipped; the material
    // tangent below does not need S.
    V3x3 S MOCHI_NO_INIT;
    if (includeGeometricStiffness) {
      S[0][0] = Sv[0];
      S[1][1] = Sv[1];
      S[2][2] = Sv[2];
      S[1][2] = S[2][1] = Sv[3];
      S[0][2] = S[2][0] = Sv[4];
      S[0][1] = S[1][0] = Sv[5];
      if (projectPsd) {
        BatchedProjectSymPsd<kBatchSize>(S);
      }
    }

    if (!materialIsIsotropic) {
      // CB[g] = κ · C₀ᵥ · Bᵍ (6×3), so the material block is Kᶠᵍ = Bᶠᵀ · CB[g]. The κ factor
      // matches S_visc = κ · C₀ᵥ · ΔEᵥ (the geometric part already carries κ through Sv).
      NdArray<V, kNumNodes, 6, 3> CB MOCHI_NO_INIT;
      for (int g = 0; g < kNumNodes; ++g) {
        for (int a = 0; a < 6; ++a) {
          for (int j = 0; j < 3; ++j) {
            V acc{0_r};
            for (int b = 0; b < 6; ++b) {
              acc += C0v[a][b] * B[g][b][j];
            }
            CB[g][a][j] = kappa * acc;
          }
        }
      }

      // Upper-triangular node blocks (g >= f). The lower triangle is mirrored below.
      for (int f = 0; f < kNumNodes; ++f) {
        for (int g = f; g < kNumNodes; ++g) {
          V geo{0_r};
          if (includeGeometricStiffness) {
            for (int k = 0; k < 3; ++k) {
              V sk{0_r};
              for (int l = 0; l < 3; ++l) {
                sk += S[k][l] * dbasis[g * 3 + l];
              }
              geo += dbasis[f * 3 + k] * sk;
            }
          }
          for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
              V acc{0_r};
              for (int a = 0; a < 6; ++a) {
                acc += B[f][a][i] * CB[g][a][j];
              }
              if (includeGeometricStiffness && i == j) {
                acc += geo;
              }
              (*outDRes)[(f * kSpaceDim + i) * kNumDofs + (g * kSpaceDim + j)] += quadWeight * acc;
            }
          }
        }
      }
    } else {
      V3x3 FdotF MOCHI_NO_INIT;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          V acc{0_r};
          for (int k = 0; k < 3; ++k) {
            acc += F[i][k] * F[j][k];
          }
          FdotF[i][j] = acc;
        }
      }

      NdArray<V, kNumNodes, 3> FdotDbasis MOCHI_NO_INIT;
      for (int f = 0; f < kNumNodes; ++f) {
        V const db0 = dbasis[f * 3 + 0];
        V const db1 = dbasis[f * 3 + 1];
        V const db2 = dbasis[f * 3 + 2];
        for (int i = 0; i < 3; ++i) {
          FdotDbasis[f][i] = F[i][0] * db0 + F[i][1] * db1 + F[i][2] * db2;
        }
      }

      // Upper-triangular node blocks (g >= f). The lower triangle is mirrored below.
      for (int f = 0; f < kNumNodes; ++f) {
        for (int g = f; g < kNumNodes; ++g) {
          V const dbasisDot = dbasis[f * 3 + 0] * dbasis[g * 3 + 0] +
              dbasis[f * 3 + 1] * dbasis[g * 3 + 1] + dbasis[f * 3 + 2] * dbasis[g * 3 + 2];

          V geo{0_r};
          if (includeGeometricStiffness) {
            for (int k = 0; k < 3; ++k) {
              V sk{0_r};
              for (int l = 0; l < 3; ++l) {
                sk += S[k][l] * dbasis[g * 3 + l];
              }
              geo += dbasis[f * 3 + k] * sk;
            }
          }

          for (int i = 0; i < 3; ++i) {
            V const fiNf = FdotDbasis[f][i];
            V const fiNg = FdotDbasis[g][i];
            for (int j = 0; j < 3; ++j) {
              V acc = kappaLambda * fiNf * FdotDbasis[g][j] +
                  kappaMu * (FdotF[i][j] * dbasisDot + fiNg * FdotDbasis[f][j]);
              if (includeGeometricStiffness && i == j) {
                acc += geo;
              }
              (*outDRes)[(f * kSpaceDim + i) * kNumDofs + (g * kSpaceDim + j)] += quadWeight * acc;
            }
          }
        }
      }
    }
  }

  // Mirror the lower-triangular node blocks.
  //
  // Note: This assigns the lower triangle from the upper triangle (rather than accumulating),
  // assuming that all prior content of outDRes was symmetric.
  if (evalDRes) {
    for (int f = 0; f < kNumNodes; ++f) {
      for (int g = f + 1; g < kNumNodes; ++g) {
        for (int i = 0; i < kSpaceDim; ++i) {
          for (int j = 0; j < kSpaceDim; ++j) {
            (*outDRes)[(g * kSpaceDim + j) * kNumDofs + (f * kSpaceDim + i)] =
                (*outDRes)[(f * kSpaceDim + i) * kNumDofs + (g * kSpaceDim + j)];
          }
        }
      }
    }
  }

  return true;
}

} // namespace mochi::fem
