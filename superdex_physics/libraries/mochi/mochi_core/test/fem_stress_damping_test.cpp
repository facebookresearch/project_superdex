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

#include <mochi_core/element_operations/fem_stress.h>
#include <mochi_core/element_operations/fem_stress_damping.h>
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/elements/tetrahedral/simplex_quadrature.h>
#include <mochi_core/materials/batched_lame_params.h>
#include <mochi_core/materials/batched_linear_elastic.h>
#include <mochi_core/materials/batched_smith_neo_hookean.h>
#include <mochi_core/materials/batched_st_venant_kirchhoff.h>
#include <mochi_core/materials/reference_material_stiffness.h>
#include <mochi_core/test/batched_work_test_utils.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/rand_utils.h>

#include <gtest/gtest.h>

using namespace mochi;
using namespace mochi::materials;
using namespace mochi::fem;

// StressDampingWork adds a strain-rate-proportional viscous PK1 (S_visc = κ·C₀:ΔE pushed forward
// by F) to a soft material response. C₀ is the Lagrangian zero-deformation stiffness, obtained from
// the material tangent at F = I with the rest-stress geometric part subtracted.
//
// The kernel is validated against:
//   - a static-state check (F == F_stageStart ⇒ ΔE = 0 ⇒ zero residual / energy),
//   - an independent isotropic-C₀ oracle for energy and residual (with a nonzero synthetic rest
//     stress S₀, this also validates the rest-stress correction: only a correctly stripped C₀
//     reproduces the symmetric oracle response),
//   - finite-difference consistency (res = ∂E/∂u, dRes = ∂res/∂u) across several real materials,
//   - PSD projection (projected geometric stiffness ⇒ symmetric, positive-semi-definite dRes),
//   - symmetry of the assembled dRes for a nonzero rest stress (the corrected C₀ is symmetric),
//   - geometric-term gating (the includeGeometricStiffness flag changes only the tangent's
//     geometric block, leaving energy/residual untouched and the material-only tangent SPD).
//
// Production soft actors use a one-point quadrature rule, but the kernel assembles by summing
// per-quadrature contributions, so every test is additionally run with a four-point rule to
// exercise multi-quadrature accumulation.

using TetElement1 = tetrahedral::Pk3DElement<1, 1>;
using TetElement4 = tetrahedral::Pk3DElement<1, 4>;

template <class Elem>
inline constexpr int kElemDim = Elem::kSpaceDim * Elem::kNumDofs;

// Invokes `fn.template operator()<Elem>()` for each tetrahedral quadrature rule under test.
template <class Fn>
static void RunElementTypes(Fn const& fn) {
  fn.template operator()<TetElement1>();
  fn.template operator()<TetElement4>();
}

static constexpr real kYoungs = 100_r;
static constexpr real kPoisson = 0.3_r;
static constexpr real kKappa = 0.5_r;

// --- Constitutive callbacks ------------------------------------------------------------------

// Real Lamé-based material response. At F = I these have zero rest stress, so they exercise the
// passive (no-op correction) path.
template <typename ParamsT, int kBS, class Fn>
static auto LameResponse(MaterialPsdStrategy psd, Fn const& fn) {
  ParamsT p;
  p.youngsModulus = kYoungs;
  p.poissonRatio = kPoisson;
  if constexpr (requires(ParamsT params) { params.psdStrategy = MaterialPsdStrategy::None; }) {
    p.psdStrategy = psd;
  }
  auto const lame = BuildBatchParams<kBS>(p);
  return [lame, fn](auto const&, auto const& F, auto* e, auto* pk1, auto* tangent, bool project) {
    fn(lame, F, e, pk1, tangent, project);
  };
}

template <int kBS>
static auto StVenantResponse(MaterialPsdStrategy psd = MaterialPsdStrategy::None) {
  return LameResponse<StVenantKirchhoffMaterialParams, kBS>(
      psd, [](auto const& lame, auto const& F, auto* e, auto* pk1, auto* tangent, bool project) {
        BatchedStVenantKirchhoffConstitutiveResponse<kBS>(lame, F, e, pk1, tangent, project);
      });
}

template <int kBS>
static auto SmithNeoHookeanResponse(MaterialPsdStrategy psd = MaterialPsdStrategy::None) {
  return LameResponse<SmithNeoHookeanMaterialParams, kBS>(
      psd, [](auto const& lame, auto const& F, auto* e, auto* pk1, auto* tangent, bool project) {
        BatchedSmithNeoHookeanConstitutiveResponse<kBS>(lame, F, e, pk1, tangent, project);
      });
}

template <int kBS>
static auto LinearResponse() {
  return LameResponse<LinearElasticMaterialParams, kBS>(
      MaterialPsdStrategy::None,
      [](auto const& lame, auto const& F, auto* e, auto* pk1, auto* tangent, bool project) {
        BatchedLinearElasticConstitutiveResponse<kBS>(lame, F, e, pk1, tangent, project);
      });
}

// Synthetic response with a known isotropic Lagrangian stiffness C₀ and an arbitrary symmetric rest
// stress S₀. The tangent returned is the two-point form ∂P/∂F|₀ = C₀ + δ⊗S₀ (the geometric part the
// kernel must strip). The response ignores F, since the kernel only evaluates it at F = I.
struct IsoModulus {
  real lambda;
  real mu;
  Matrix3x3r s0; // symmetric rest stress (zero for passive)
};

template <int kBS>
static auto SyntheticResponse(IsoModulus const& m) {
  return [m](auto const&, auto const& /*F*/, auto* e, auto* pk1, auto* tangent, bool /*project*/) {
    using V = BatchReal<kBS>;
    if (e) {
      *e = BatchDouble<kBS>{0.0};
    }
    if (pk1) {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          (*pk1)[i][j] = V{m.s0[i][j]};
        }
      }
    }
    if (tangent) {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
              real val = m.lambda * (i == j ? 1_r : 0_r) * (r == c ? 1_r : 0_r) +
                  m.mu *
                      ((i == r ? 1_r : 0_r) * (j == c ? 1_r : 0_r) +
                       (i == c ? 1_r : 0_r) * (j == r ? 1_r : 0_r));
              if (i == r) {
                val += m.s0[c][j]; // geometric rest-stress part of the two-point tangent
              }
              (*tangent)[i][j][r][c] = V{val};
            }
          }
        }
      }
    }
  };
}

// Heterogeneous synthetic isotropic response: each lane derives its own (λ, μ) from its element
// index, so a batch that gathers distinct elements sees a distinct-but-isotropic C₀ per lane. This
// is what exercises the fast path's per-lane C₀ᵥ[0][1]/C₀ᵥ[3][3] reads — SyntheticResponse
// broadcasts a single scalar to every lane and never varies λ,μ across the batch. F and rest stress
// are ignored (evaluated at F = I, passive).
template <int kBS>
static auto HeterogeneousIsoResponse() {
  return [](NdArray<int, kBS> const& idx,
            auto const& /*F*/,
            auto* e,
            auto* pk1,
            auto* tangent,
            bool /*project*/) {
    using V = BatchReal<kBS>;
    if (e) {
      *e = BatchDouble<kBS>{0.0};
    }
    alignas(alignof(V)) real lamStg[V::kSize]{};
    alignas(alignof(V)) real muStg[V::kSize]{};
    for (int b = 0; b < kBS; ++b) {
      real const t = static_cast<real>(idx[b]);
      lamStg[b] = 8_r + 2_r * t; // distinct per element index
      muStg[b] = 5_r + t;
    }
    V const lambda = Load<V>(lamStg);
    V const mu = Load<V>(muStg);
    if (pk1) {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          (*pk1)[i][j] = V{0_r};
        }
      }
    }
    if (tangent) {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
              real const cLam = (i == j ? 1_r : 0_r) * (r == c ? 1_r : 0_r);
              real const cMu = (i == r ? 1_r : 0_r) * (j == c ? 1_r : 0_r) +
                  (i == c ? 1_r : 0_r) * (j == r ? 1_r : 0_r);
              (*tangent)[i][j][r][c] = lambda * V{cLam} + mu * V{cMu};
            }
          }
        }
      }
    }
  };
}

// --- Oracle (isotropic C₀, no rest stress) ---------------------------------------------------

template <class Elem>
static Matrix3x3r BuildF(Elem const& el, NdArray<real, kElemDim<Elem>> const& disp, int q) {
  Matrix3x3r F = Eye<3, real>();
  for (int f = 0; f < Elem::kNumDofs; ++f) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        F[r][c] += disp[f * 3 + r] * el.dBasisEvaluated[q][f][c];
      }
    }
  }
  return F;
}

// S_visc = κ·(λ tr(ΔE) I + 2μ ΔE), the intended viscous PK2 for an isotropic C₀.
static Matrix3x3r
ViscousPk2Oracle(Matrix3x3r const& F, Matrix3x3r const& Fss, real lambda, real mu) {
  Matrix3x3r dE{};
  for (int k = 0; k < 3; ++k) {
    for (int l = 0; l < 3; ++l) {
      real acc = 0_r;
      for (int p = 0; p < 3; ++p) {
        acc += F[p][k] * F[p][l] - Fss[p][k] * Fss[p][l];
      }
      dE[k][l] = 0.5_r * acc;
    }
  }
  real const trdE = dE[0][0] + dE[1][1] + dE[2][2];
  Matrix3x3r S{};
  for (int k = 0; k < 3; ++k) {
    for (int l = 0; l < 3; ++l) {
      S[k][l] = kKappa * (lambda * trdE * (k == l ? 1_r : 0_r) + 2_r * mu * dE[k][l]);
    }
  }
  return S;
}

template <class Elem>
static double ViscousEnergyOracle(
    Elem const& el,
    NdArray<real, kElemDim<Elem>> const& disp,
    NdArray<real, kElemDim<Elem>> const& ss,
    real lambda,
    real mu) {
  double energy = 0.0;
  for (int q = 0; q < Elem::kNumQuadPoints; ++q) {
    Matrix3x3r const F = BuildF(el, disp, q);
    Matrix3x3r const Fss = BuildF(el, ss, q);
    Matrix3x3r const S = ViscousPk2Oracle(F, Fss, lambda, mu);
    Matrix3x3r dE{};
    for (int k = 0; k < 3; ++k) {
      for (int l = 0; l < 3; ++l) {
        real acc = 0_r;
        for (int p = 0; p < 3; ++p) {
          acc += F[p][k] * F[p][l] - Fss[p][k] * Fss[p][l];
        }
        dE[k][l] = 0.5_r * acc;
      }
    }
    real psi = 0_r;
    for (int k = 0; k < 3; ++k) {
      for (int l = 0; l < 3; ++l) {
        psi += dE[k][l] * S[k][l];
      }
    }
    psi *= 0.5_r;
    energy += static_cast<double>(psi) * static_cast<double>(el.quadWeights[q]);
  }
  return energy;
}

template <class Elem>
static NdArray<real, kElemDim<Elem>> ViscousResidualOracle(
    Elem const& el,
    NdArray<real, kElemDim<Elem>> const& disp,
    NdArray<real, kElemDim<Elem>> const& ss,
    real lambda,
    real mu) {
  NdArray<real, kElemDim<Elem>> res{};
  for (int q = 0; q < Elem::kNumQuadPoints; ++q) {
    Matrix3x3r const F = BuildF(el, disp, q);
    Matrix3x3r const Fss = BuildF(el, ss, q);
    Matrix3x3r const S = ViscousPk2Oracle(F, Fss, lambda, mu);
    Matrix3x3r P{}; // P = F · S
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        P[r][c] = F[r][0] * S[0][c] + F[r][1] * S[1][c] + F[r][2] * S[2][c];
      }
    }
    for (int f = 0; f < Elem::kNumDofs; ++f) {
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          res[f * 3 + r] += P[r][c] * el.dBasisEvaluated[q][f][c] * el.quadWeights[q];
        }
      }
    }
  }
  return res;
}

// --- Kernel runner ---------------------------------------------------------------------------

template <int kBS, class Elem, class Constitutive>
static void RunDamping(
    Span<Elem const> elemSpan,
    NdArray<int, kBS> const& idx,
    NdArray<NdArray<real, kElemDim<Elem>>, kBS> const& disp,
    NdArray<NdArray<real, kElemDim<Elem>>, kBS> const& stageStart,
    bool wantE,
    bool wantR,
    bool wantD,
    bool projectPsd,
    Constitutive const& constitutive,
    NdArray<double, kBS>& energy,
    NdArray<NdArray<real, kElemDim<Elem>>, kBS>& res,
    NdArray<NdArray<real, kElemDim<Elem> * kElemDim<Elem>>, kBS>& dres,
    real kappa = kKappa,
    bool includeGeo = true,
    bool materialIsIsotropic = false) {
  constexpr int kDim = kElemDim<Elem>;
  using V = BatchReal<kBS>;
  auto pack = [](NdArray<NdArray<real, kDim>, kBS> const& src,
                 fem::BatchElementVector<kBS, Elem>& dst) {
    alignas(alignof(V)) real staging[V::kSize]{};
    for (int k = 0; k < kDim; ++k) {
      for (int b = 0; b < kBS; ++b) {
        staging[b] = src[b][k];
      }
      dst[k] = Load<V>(staging);
    }
  };
  fem::BatchElementVector<kBS, Elem> dispBatch, ssBatch;
  pack(disp, dispBatch);
  pack(stageStart, ssBatch);

  BatchDouble<kBS> e{0.0};
  fem::BatchElementVector<kBS, Elem> r{};
  fem::BatchElementMatrix<kBS, Elem> d{};
  auto const c0v = ComputeReferenceMaterialStiffnessVoigt<kBS>(idx, constitutive);
  StressDampingWork<kBS>(
      idx,
      elemSpan,
      dispBatch,
      ssBatch,
      wantE ? &e : nullptr,
      wantR ? &r : nullptr,
      wantD ? &d : nullptr,
      projectPsd,
      includeGeo,
      kappa,
      c0v,
      materialIsIsotropic);
  for (int b = 0; b < kBS; ++b) {
    if (wantE) {
      energy[b] = e[b];
    }
    if (wantR) {
      for (int k = 0; k < kDim; ++k) {
        res[b][k] = r[k][b];
      }
    }
    if (wantD) {
      for (int k = 0; k < kDim * kDim; ++k) {
        dres[b][k] = d[k][b];
      }
    }
  }
}

// --- Elastic kernel runner (for StVK cross-validation) ---------------------------------------

// Mirrors RunDamping's packing/unpacking, but drives the elastic StressWork kernel. Used to
// cross-check the damping assembly against the StVK elastic assembly (see below).
template <int kBS, class Elem, class Constitutive>
static void RunElastic(
    Span<Elem const> elemSpan,
    NdArray<int, kBS> const& idx,
    NdArray<NdArray<real, kElemDim<Elem>>, kBS> const& disp,
    bool wantE,
    bool wantR,
    bool wantD,
    bool projectPsd,
    Constitutive const& constitutive,
    NdArray<double, kBS>& energy,
    NdArray<NdArray<real, kElemDim<Elem>>, kBS>& res,
    NdArray<NdArray<real, kElemDim<Elem> * kElemDim<Elem>>, kBS>& dres) {
  constexpr int kDim = kElemDim<Elem>;
  using V = BatchReal<kBS>;
  fem::BatchElementVector<kBS, Elem> dispBatch;
  alignas(alignof(V)) real staging[V::kSize]{};
  for (int k = 0; k < kDim; ++k) {
    for (int b = 0; b < kBS; ++b) {
      staging[b] = disp[b][k];
    }
    dispBatch[k] = Load<V>(staging);
  }

  BatchDouble<kBS> e{0.0};
  fem::BatchElementVector<kBS, Elem> r{};
  fem::BatchElementMatrix<kBS, Elem> d{};
  StressWork<kBS>(
      idx,
      elemSpan,
      dispBatch,
      wantE ? &e : nullptr,
      wantR ? &r : nullptr,
      wantD ? &d : nullptr,
      projectPsd,
      constitutive);
  for (int b = 0; b < kBS; ++b) {
    if (wantE) {
      energy[b] = e[b];
    }
    if (wantR) {
      for (int k = 0; k < kDim; ++k) {
        res[b][k] = r[k][b];
      }
    }
    if (wantD) {
      for (int k = 0; k < kDim * kDim; ++k) {
        dres[b][k] = d[k][b];
      }
    }
  }
}

// --- Tests -----------------------------------------------------------------------------------

// Static state: when the current displacement equals the stage-start displacement, ΔE = 0 so the
// viscous residual and energy vanish.
template <int kBS, class Elem>
static void VerifyStaticState() {
  constexpr int kDim = kElemDim<Elem>;
  auto const data = TestTetMeshData<Elem>::CreateMinimalCube();
  auto const elemSpan = MakeConstSpan(data.elements);
  int const numElements = isize(data.elements);
  auto const constitutive = SyntheticResponse<kBS>(IsoModulus{.lambda = 10_r, .mu = 7_r, .s0 = {}});

  NdArray<int, kBS> idx;
  NdArray<NdArray<real, kDim>, kBS> disp;
  for (int b = 0; b < kBS; ++b) {
    idx[b] = b % numElements;
    disp[b] = MakeRandomArray<kDim>(static_cast<unsigned int>(11 + b), -0.05_r, 0.05_r);
  }

  NdArray<double, kBS> energy;
  NdArray<NdArray<real, kDim>, kBS> res;
  NdArray<NdArray<real, kDim * kDim>, kBS> dres;
  RunDamping<kBS, Elem>(
      elemSpan,
      idx,
      disp,
      /*stageStart*/ disp,
      true,
      true,
      false,
      false,
      constitutive,
      energy,
      res,
      dres);
  for (int b = 0; b < kBS; ++b) {
    EXPECT_NEAR(0.0, energy[b], 1e-9);
    for (int k = 0; k < kDim; ++k) {
      EXPECT_NEAR(0_r, res[b][k], kRelTol);
    }
  }
}

// Energy and residual match the independent isotropic-C₀ oracle. A nonzero rest stress S₀ in the
// synthetic tangent exercises the rest-stress correction: only a correctly stripped C₀ reproduces
// the symmetric oracle response.
template <int kBS, class Elem>
static void VerifyEnergyResidualOracle(IsoModulus const& m) {
  constexpr int kDim = kElemDim<Elem>;
  auto const data = TestTetMeshData<Elem>::CreateMinimalCube();
  auto const elemSpan = MakeConstSpan(data.elements);
  int const numElements = isize(data.elements);
  auto const constitutive = SyntheticResponse<kBS>(m);

  real const tol = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 5e-4_r;

  for (int trial = 0; trial < 3; ++trial) {
    NdArray<int, kBS> idx;
    NdArray<NdArray<real, kDim>, kBS> disp, ss;
    for (int b = 0; b < kBS; ++b) {
      idx[b] = (trial + b) % numElements;
      disp[b] =
          MakeRandomArray<kDim>(static_cast<unsigned int>(31 + trial * kBS + b), -0.06_r, 0.06_r);
      ss[b] =
          MakeRandomArray<kDim>(static_cast<unsigned int>(97 + trial * kBS + b), -0.04_r, 0.04_r);
    }

    NdArray<double, kBS> energy;
    NdArray<NdArray<real, kDim>, kBS> res;
    NdArray<NdArray<real, kDim * kDim>, kBS> dres;
    RunDamping<kBS, Elem>(
        elemSpan, idx, disp, ss, true, true, false, false, constitutive, energy, res, dres);

    for (int b = 0; b < kBS; ++b) {
      Elem const& el = data.elements[idx[b]];
      ExpectNearEnergy(ViscousEnergyOracle(el, disp[b], ss[b], m.lambda, m.mu), energy[b], tol);
      ExpectNearL2(ViscousResidualOracle(el, disp[b], ss[b], m.lambda, m.mu), res[b], tol);
    }
  }
}

// Finite-difference consistency: residual = ∂(energy)/∂u and dresidual = ∂(residual)/∂u.
template <int kBS, class Elem, class C>
static void VerifyFd(C const& constitutive) {
  constexpr int kDim = kElemDim<Elem>;
  auto const data = TestTetMeshData<Elem>::CreateMinimalCube();
  auto const elemSpan = MakeConstSpan(data.elements);
  int const numElements = isize(data.elements);

  real const eps = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-3_r;
  real const resTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-4_r : 2e-2_r;
  real const dresTol = MOCHI_USE_DOUBLE_PRECISION ? 2e-3_r : 5e-2_r;

  NdArray<double, kBS> e0, ep, em;
  NdArray<NdArray<real, kDim>, kBS> baseRes, rp, rm, scratchRes;
  NdArray<NdArray<real, kDim * kDim>, kBS> baseDRes, scratchDRes;

  for (int trial = 0; trial < 3; ++trial) {
    NdArray<int, kBS> idx;
    NdArray<NdArray<real, kDim>, kBS> disp, ss;
    for (int b = 0; b < kBS; ++b) {
      idx[b] = (trial + b) % numElements;
      disp[b] =
          MakeRandomArray<kDim>(static_cast<unsigned int>(7 + trial * kBS + b), -0.05_r, 0.05_r);
      ss[b] =
          MakeRandomArray<kDim>(static_cast<unsigned int>(503 + trial * kBS + b), -0.03_r, 0.03_r);
    }

    RunDamping<kBS, Elem>(
        elemSpan, idx, disp, ss, false, true, true, false, constitutive, e0, baseRes, baseDRes);

    // res == d(energy)/d(disp).
    NdArray<NdArray<real, kDim>, kBS> resFd{};
    for (int i = 0; i < kDim; ++i) {
      auto dispP = disp;
      auto dispM = disp;
      for (int b = 0; b < kBS; ++b) {
        dispP[b][i] += eps;
        dispM[b][i] -= eps;
      }
      RunDamping<kBS, Elem>(
          elemSpan,
          idx,
          dispP,
          ss,
          true,
          false,
          false,
          false,
          constitutive,
          ep,
          scratchRes,
          scratchDRes);
      RunDamping<kBS, Elem>(
          elemSpan,
          idx,
          dispM,
          ss,
          true,
          false,
          false,
          false,
          constitutive,
          em,
          scratchRes,
          scratchDRes);
      for (int b = 0; b < kBS; ++b) {
        resFd[b][i] = static_cast<real>((ep[b] - em[b]) / (2.0 * static_cast<double>(eps)));
      }
    }
    for (int b = 0; b < kBS; ++b) {
      ExpectNearL2(baseRes[b], resFd[b], resTol);
    }

    // dRes == d(residual)/d(disp) (raw tangent, projectPsd == false).
    NdArray<NdArray<real, kDim * kDim>, kBS> dresFd{};
    for (int j = 0; j < kDim; ++j) {
      auto dispP = disp;
      auto dispM = disp;
      for (int b = 0; b < kBS; ++b) {
        dispP[b][j] += eps;
        dispM[b][j] -= eps;
      }
      RunDamping<kBS, Elem>(
          elemSpan, idx, dispP, ss, false, true, false, false, constitutive, ep, rp, scratchDRes);
      RunDamping<kBS, Elem>(
          elemSpan, idx, dispM, ss, false, true, false, false, constitutive, em, rm, scratchDRes);
      for (int b = 0; b < kBS; ++b) {
        for (int i = 0; i < kDim; ++i) {
          dresFd[b][i * kDim + j] = (rp[b][i] - rm[b][i]) / (2_r * eps);
        }
      }
    }
    for (int b = 0; b < kBS; ++b) {
      ExpectNearL2(baseDRes[b], dresFd[b], dresTol);
    }
  }
}

// Projected geometric stiffness yields a symmetric, positive-semi-definite assembled dresidual.
template <int kBS, class Elem, class C>
static void VerifyPsd(C const& constitutive) {
  constexpr int kDim = kElemDim<Elem>;
  auto const data = TestTetMeshData<Elem>::CreateMinimalCube();
  auto const elemSpan = MakeConstSpan(data.elements);
  int const numElements = isize(data.elements);

  NdArray<int, kBS> idx;
  NdArray<NdArray<real, kDim>, kBS> disp, ss;
  for (int b = 0; b < kBS; ++b) {
    idx[b] = b % numElements;
    disp[b] = MakeRandomArray<kDim>(static_cast<unsigned int>(4242 + b), -0.08_r, 0.08_r);
    ss[b] = MakeRandomArray<kDim>(static_cast<unsigned int>(8484 + b), -0.04_r, 0.04_r);
  }
  NdArray<double, kBS> energy;
  NdArray<NdArray<real, kDim>, kBS> scratchRes;
  NdArray<NdArray<real, kDim * kDim>, kBS> dres;
  RunDamping<kBS, Elem>(
      elemSpan, idx, disp, ss, false, false, true, true, constitutive, energy, scratchRes, dres);

  real const symTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-9_r : 1e-3_r;
  for (int b = 0; b < kBS; ++b) {
    real diagSum = 0_r;
    for (int i = 0; i < kDim; ++i) {
      diagSum += Abs(dres[b][i * kDim + i]);
    }
    for (int i = 0; i < kDim; ++i) {
      for (int j = 0; j < kDim; ++j) {
        real const scale = 1_r + Abs(dres[b][i * kDim + j]);
        EXPECT_LE(Abs(dres[b][i * kDim + j] - dres[b][j * kDim + i]), symTol * scale);
      }
    }
    for (int t = 0; t < 5; ++t) {
      auto const v = MakeRandomArray<kDim>(static_cast<unsigned int>(900 + t + b), -1_r, 1_r);
      double quad = 0.0;
      for (int i = 0; i < kDim; ++i) {
        for (int j = 0; j < kDim; ++j) {
          quad += static_cast<double>(v[i]) * static_cast<double>(dres[b][i * kDim + j]) *
              static_cast<double>(v[j]);
        }
      }
      EXPECT_GE(quad, -1e-3 * static_cast<double>(diagSum));
    }
  }
}

// The assembled dresidual is symmetric even with a nonzero rest stress (the corrected C₀ is
// symmetric, unlike the raw two-point tangent which would break (f,r)<->(g,c) symmetry).
template <int kBS, class Elem>
static void VerifyDResSymmetryWithRestStress() {
  constexpr int kDim = kElemDim<Elem>;
  Matrix3x3r s0 = {Real3{3_r, 1_r, -2_r}, Real3{1_r, -4_r, 0.5_r}, Real3{-2_r, 0.5_r, 5_r}};
  auto const constitutive = SyntheticResponse<kBS>(IsoModulus{.lambda = 12_r, .mu = 8_r, .s0 = s0});

  auto const data = TestTetMeshData<Elem>::CreateMinimalCube();
  auto const elemSpan = MakeConstSpan(data.elements);
  int const numElements = isize(data.elements);

  NdArray<int, kBS> idx;
  NdArray<NdArray<real, kDim>, kBS> disp, ss;
  for (int b = 0; b < kBS; ++b) {
    idx[b] = b % numElements;
    disp[b] = MakeRandomArray<kDim>(static_cast<unsigned int>(321 + b), -0.06_r, 0.06_r);
    ss[b] = MakeRandomArray<kDim>(static_cast<unsigned int>(654 + b), -0.03_r, 0.03_r);
  }
  NdArray<double, kBS> energy;
  NdArray<NdArray<real, kDim>, kBS> scratchRes;
  NdArray<NdArray<real, kDim * kDim>, kBS> dres;
  RunDamping<kBS, Elem>(
      elemSpan, idx, disp, ss, false, false, true, false, constitutive, energy, scratchRes, dres);

  real const symTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-9_r : 1e-3_r;
  for (int b = 0; b < kBS; ++b) {
    for (int i = 0; i < kDim; ++i) {
      for (int j = 0; j < kDim; ++j) {
        real const scale = 1_r + Abs(dres[b][i * kDim + j]);
        EXPECT_LE(Abs(dres[b][i * kDim + j] - dres[b][j * kDim + i]), symTol * scale);
      }
    }
  }
}

// StVK cross-validation: the viscous damping stress S_visc = κ·C₀:ΔE is, for zero stage-start
// displacement (ΔE = E(F)) and κ = 1, identical to the St. Venant-Kirchhoff elastic PK2
// S_StVK = C₀:E, whose material tangent is exactly the constant C₀. So the damping assembly must
// reproduce the elastic StVK assembly term-by-term (energy, residual, DResidual). The two kernels
// sum in different arithmetic orders (Voigt Bᵀ·C₀·B vs. ∂P/∂F contraction), making this a genuine
// independent recomputation that also exercises the Voigt engineering-shear factor-of-two.
template <int kBS, class Elem>
static void VerifyStVenantEquivalence() {
  constexpr int kDim = kElemDim<Elem>;
  auto const data = TestTetMeshData<Elem>::CreateMinimalCube();
  auto const elemSpan = MakeConstSpan(data.elements);
  int const numElements = isize(data.elements);
  auto const constitutive = StVenantResponse<kBS>(MaterialPsdStrategy::None);

  // Round-off from the two distinct summation orders only. The DResidual accumulates more terms,
  // so it gets a slightly looser dedicated tolerance in single precision.
  real const tol = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 5e-4_r;
  real const dresTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 1e-3_r;

  NdArray<NdArray<real, kDim>, kBS> const zeroSs{};

  for (int trial = 0; trial < 3; ++trial) {
    NdArray<int, kBS> idx;
    NdArray<NdArray<real, kDim>, kBS> disp;
    for (int b = 0; b < kBS; ++b) {
      idx[b] = (trial + b) % numElements;
      disp[b] =
          MakeRandomArray<kDim>(static_cast<unsigned int>(71 + trial * kBS + b), -0.05_r, 0.05_r);
    }

    NdArray<double, kBS> eD, eE;
    NdArray<NdArray<real, kDim>, kBS> rD, rE;
    NdArray<NdArray<real, kDim * kDim>, kBS> dD, dE;
    real const kappa = 1_r;
    RunDamping<kBS, Elem>(
        elemSpan, idx, disp, zeroSs, true, true, true, false, constitutive, eD, rD, dD, kappa);
    RunElastic<kBS, Elem>(elemSpan, idx, disp, true, true, true, false, constitutive, eE, rE, dE);

    for (int b = 0; b < kBS; ++b) {
      ExpectNearEnergy(eE[b], eD[b], tol);
      ExpectNearL2(rE[b], rD[b], tol);
      ExpectNearL2(dE[b], dD[b], dresTol);
    }
  }
}

// Independent oracle for the geometric block of the viscous tangent:
// Kᶠᵍ_geo = wᵠ · (∇Nᶠ · S_visc · ∇Nᵍ) added on the i==j diagonal, with S_visc from the
// isotropic-C₀ oracle. Used to validate that only the geometric term is gated by the flag.
template <class Elem>
static NdArray<real, kElemDim<Elem> * kElemDim<Elem>> ViscousGeometricBlockOracle(
    Elem const& el,
    NdArray<real, kElemDim<Elem>> const& disp,
    NdArray<real, kElemDim<Elem>> const& ss,
    real lambda,
    real mu) {
  constexpr int kDim = kElemDim<Elem>;
  NdArray<real, kDim * kDim> geoBlock{};
  for (int q = 0; q < Elem::kNumQuadPoints; ++q) {
    Matrix3x3r const F = BuildF(el, disp, q);
    Matrix3x3r const Fss = BuildF(el, ss, q);
    Matrix3x3r const S = ViscousPk2Oracle(F, Fss, lambda, mu);
    for (int f = 0; f < Elem::kNumDofs; ++f) {
      for (int g = 0; g < Elem::kNumDofs; ++g) {
        real geo = 0_r;
        for (int k = 0; k < 3; ++k) {
          for (int l = 0; l < 3; ++l) {
            geo += el.dBasisEvaluated[q][f][k] * S[k][l] * el.dBasisEvaluated[q][g][l];
          }
        }
        geo *= el.quadWeights[q];
        for (int d = 0; d < 3; ++d) {
          geoBlock[(f * 3 + d) * kDim + (g * 3 + d)] += geo;
        }
      }
    }
  }
  return geoBlock;
}

// includeGeometricStiffness gates only the geometric block: energy and residual are unchanged, the
// flag-off tangent is symmetric + PSD for an SPD C₀ (material-only modified-Newton tangent), and
// the on/off tangent difference matches the independent geometric-block oracle.
template <int kBS, class Elem>
static void VerifyGeometricTermGating() {
  constexpr int kDim = kElemDim<Elem>;
  auto const data = TestTetMeshData<Elem>::CreateMinimalCube();
  auto const elemSpan = MakeConstSpan(data.elements);
  int const numElements = isize(data.elements);
  constexpr real kLambda = 9_r;
  constexpr real kMu = 6_r;
  auto const constitutive =
      SyntheticResponse<kBS>(IsoModulus{.lambda = kLambda, .mu = kMu, .s0 = {}});

  real const tol = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 5e-4_r;
  real const symTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-9_r : 1e-3_r;

  for (int trial = 0; trial < 3; ++trial) {
    NdArray<int, kBS> idx;
    NdArray<NdArray<real, kDim>, kBS> disp, ss;
    for (int b = 0; b < kBS; ++b) {
      idx[b] = (trial + b) % numElements;
      disp[b] =
          MakeRandomArray<kDim>(static_cast<unsigned int>(202 + trial * kBS + b), -0.06_r, 0.06_r);
      ss[b] =
          MakeRandomArray<kDim>(static_cast<unsigned int>(808 + trial * kBS + b), -0.03_r, 0.03_r);
    }

    NdArray<double, kBS> eOn, eOff;
    NdArray<NdArray<real, kDim>, kBS> rOn, rOff;
    NdArray<NdArray<real, kDim * kDim>, kBS> dOn, dOff;
    RunDamping<kBS, Elem>(
        elemSpan,
        idx,
        disp,
        ss,
        true,
        true,
        true,
        false,
        constitutive,
        eOn,
        rOn,
        dOn,
        kKappa,
        /*includeGeo*/ true);
    RunDamping<kBS, Elem>(
        elemSpan,
        idx,
        disp,
        ss,
        true,
        true,
        true,
        false,
        constitutive,
        eOff,
        rOff,
        dOff,
        kKappa,
        /*includeGeo*/ false);

    for (int b = 0; b < kBS; ++b) {
      // (1) Energy and residual are unchanged by the flag.
      ExpectNearEnergy(eOn[b], eOff[b], tol);
      ExpectNearL2(rOn[b], rOff[b], tol);

      // (2) Flag-off tangent is symmetric and positive-semi-definite (material-only block).
      real diagSum = 0_r;
      for (int i = 0; i < kDim; ++i) {
        diagSum += Abs(dOff[b][i * kDim + i]);
      }
      for (int i = 0; i < kDim; ++i) {
        for (int j = 0; j < kDim; ++j) {
          real const scale = 1_r + Abs(dOff[b][i * kDim + j]);
          EXPECT_LE(Abs(dOff[b][i * kDim + j] - dOff[b][j * kDim + i]), symTol * scale);
        }
      }
      for (int t = 0; t < 5; ++t) {
        auto const v = MakeRandomArray<kDim>(static_cast<unsigned int>(700 + t + b), -1_r, 1_r);
        double quad = 0.0;
        for (int i = 0; i < kDim; ++i) {
          for (int j = 0; j < kDim; ++j) {
            quad += static_cast<double>(v[i]) * static_cast<double>(dOff[b][i * kDim + j]) *
                static_cast<double>(v[j]);
          }
        }
        EXPECT_GE(quad, -1e-3 * static_cast<double>(diagSum));
      }

      // (3) dRes(on) − dRes(off) equals the independent geometric-block oracle.
      Elem const& el = data.elements[idx[b]];
      auto const geoOracle = ViscousGeometricBlockOracle(el, disp[b], ss[b], kLambda, kMu);
      NdArray<real, kDim * kDim> diff{};
      for (int k = 0; k < kDim * kDim; ++k) {
        diff[k] = dOn[b][k] - dOff[b][k];
      }
      ExpectNearL2(geoOracle, diff, tol);
    }
  }
}

// The isotropic fast path (materialIsIsotropic = true) must reproduce the dense 6×6 C₀ᵥ contraction
// (false) up to round-off for any isotropic C₀: identical energy, residual, and tangent (with the
// geometric term both on and off). The two paths differ only in summation order, so this pins the
// optimization to the reference implementation across batch sizes and quadrature rules.
template <int kBS, class Elem, class C>
static void VerifyIsotropicDenseEquivalence(C const& constitutive) {
  constexpr int kDim = kElemDim<Elem>;
  auto const data = TestTetMeshData<Elem>::CreateMinimalCube();
  auto const elemSpan = MakeConstSpan(data.elements);
  int const numElements = isize(data.elements);

  real const tol = MOCHI_USE_DOUBLE_PRECISION ? 1e-9_r : 5e-4_r;

  struct EvalCase {
    bool assembleEnergy;
    bool assembleResidual;
    bool assembleDResidual;
    bool projectPsd;
    bool includeGeo;
  };
  for (OutputConfig const cfg : kAllOutputConfigs) {
    for (bool const projectPsd : {false, true}) {
      for (bool const includeGeo : {false, true}) {
        EvalCase const evalCase{
            .assembleEnergy = cfg.energy,
            .assembleResidual = cfg.residual,
            .assembleDResidual = cfg.dresidual,
            .projectPsd = projectPsd,
            .includeGeo = includeGeo};
        for (int trial = 0; trial < 3; ++trial) {
          NdArray<int, kBS> idx;
          NdArray<NdArray<real, kDim>, kBS> disp, ss;
          for (int b = 0; b < kBS; ++b) {
            idx[b] = (trial + b) % numElements;
            disp[b] = MakeRandomArray<kDim>(
                static_cast<unsigned int>(311 + trial * kBS + b), -0.06_r, 0.06_r);
            ss[b] = MakeRandomArray<kDim>(
                static_cast<unsigned int>(917 + trial * kBS + b), -0.03_r, 0.03_r);
          }

          NdArray<double, kBS> eDense, eIso;
          NdArray<NdArray<real, kDim>, kBS> rDense, rIso;
          NdArray<NdArray<real, kDim * kDim>, kBS> dDense, dIso;
          RunDamping<kBS, Elem>(
              elemSpan,
              idx,
              disp,
              ss,
              evalCase.assembleEnergy,
              evalCase.assembleResidual,
              evalCase.assembleDResidual,
              evalCase.projectPsd,
              constitutive,
              eDense,
              rDense,
              dDense,
              kKappa,
              evalCase.includeGeo,
              /*materialIsIsotropic*/ false);
          RunDamping<kBS, Elem>(
              elemSpan,
              idx,
              disp,
              ss,
              evalCase.assembleEnergy,
              evalCase.assembleResidual,
              evalCase.assembleDResidual,
              evalCase.projectPsd,
              constitutive,
              eIso,
              rIso,
              dIso,
              kKappa,
              evalCase.includeGeo,
              /*materialIsIsotropic*/ true);

          for (int b = 0; b < kBS; ++b) {
            if (evalCase.assembleEnergy) {
              ExpectNearEnergy(eDense[b], eIso[b], tol);
            }
            if (evalCase.assembleResidual) {
              ExpectNearL2(rDense[b], rIso[b], tol);
            }
            if (evalCase.assembleDResidual) {
              ExpectNearL2(dDense[b], dIso[b], tol);
            }
          }
        }
      }
    }
  }
}

// Each legal output subset must match the corresponding outputs from full assembly.
template <int kBS, class Elem>
static void VerifyAssemblySubsetConsistency() {
  constexpr int kDim = kElemDim<Elem>;

  auto const data = TestTetMeshData<Elem>::CreateMinimalCube();
  auto const elemSpan = MakeConstSpan(data.elements);
  int const numElements = isize(data.elements);
  auto const constitutive = SyntheticResponse<kBS>(IsoModulus{.lambda = 9_r, .mu = 6_r, .s0 = {}});

  NdArray<int, kBS> idx;
  NdArray<NdArray<real, kDim>, kBS> disp, ss;
  for (int b = 0; b < kBS; ++b) {
    idx[b] = b % numElements;
    disp[b] = MakeRandomArray<kDim>(static_cast<unsigned int>(311 + b), -0.06_r, 0.06_r);
    ss[b] = MakeRandomArray<kDim>(static_cast<unsigned int>(917 + b), -0.03_r, 0.03_r);
  }

  struct Outputs {
    NdArray<double, kBS> energy;
    NdArray<NdArray<real, kDim>, kBS> residual;
    NdArray<NdArray<real, kDim * kDim>, kBS> dResidual;
  };
  auto run = [&](bool assembleEnergy,
                 bool assembleResidual,
                 bool assembleDResidual,
                 bool projectPsd,
                 bool includeGeo,
                 bool materialIsIsotropic) {
    Outputs outputs{};
    RunDamping<kBS, Elem>(
        elemSpan,
        idx,
        disp,
        ss,
        assembleEnergy,
        assembleResidual,
        assembleDResidual,
        projectPsd,
        constitutive,
        outputs.energy,
        outputs.residual,
        outputs.dResidual,
        kKappa,
        includeGeo,
        materialIsIsotropic);
    return outputs;
  };

  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-9_r : 5e-4_r;
  for (bool const materialIsIsotropic : {false, true}) {
    for (bool const includeGeo : {false, true}) {
      for (bool const projectPsd : {false, true}) {
        Outputs const full = run(
            /*assembleEnergy*/ true,
            /*assembleResidual*/ true,
            /*assembleDResidual*/ true,
            projectPsd,
            includeGeo,
            materialIsIsotropic);
        for (OutputConfig const cfg : kAllOutputConfigs) {
          Outputs const selected = run(
              cfg.energy, cfg.residual, cfg.dresidual, projectPsd, includeGeo, materialIsIsotropic);
          for (int b = 0; b < kBS; ++b) {
            if (cfg.energy) {
              ExpectNearEnergy(full.energy[b], selected.energy[b], kTol);
            }
            if (cfg.residual) {
              ExpectNearL2(full.residual[b], selected.residual[b], kTol);
            }
            if (cfg.dresidual) {
              ExpectNearL2(full.dResidual[b], selected.dResidual[b], kTol);
            }
          }
        }
      }
    }
  }
}

TEST(StressDampingWork, StaticState) {
  RunBatchSizes<1, 4, 8>(
      [&]<int kBS>() { RunElementTypes([&]<class Elem>() { VerifyStaticState<kBS, Elem>(); }); });
}
TEST(StressDampingWork, EnergyResidualOraclePassive) {
  RunBatchSizes<1, 4, 8>([&]<int kBS>() {
    RunElementTypes([&]<class Elem>() {
      VerifyEnergyResidualOracle<kBS, Elem>(IsoModulus{.lambda = 9_r, .mu = 6_r, .s0 = {}});
    });
  });
}

TEST(StressDampingWork, EnergyResidualOracleRestStress) {
  // Nonzero, general symmetric rest stress: validates the rest-stress correction.
  Matrix3x3r const s0 = {Real3{2_r, -1_r, 0.5_r}, Real3{-1_r, 3_r, -2_r}, Real3{0.5_r, -2_r, 1_r}};
  RunBatchSizes<1, 4, 8>([&]<int kBS>() {
    RunElementTypes([&]<class Elem>() {
      VerifyEnergyResidualOracle<kBS, Elem>(IsoModulus{.lambda = 9_r, .mu = 6_r, .s0 = s0});
    });
  });
}

TEST(StressDampingWork, TangentFdSynthetic) {
  Matrix3x3r const s0 = {Real3{2_r, -1_r, 0.5_r}, Real3{-1_r, 3_r, -2_r}, Real3{0.5_r, -2_r, 1_r}};
  RunBatchSizes<1, 4, 8>([&]<int kBS>() {
    RunElementTypes([&]<class Elem>() {
      VerifyFd<kBS, Elem>(SyntheticResponse<kBS>(IsoModulus{.lambda = 9_r, .mu = 6_r, .s0 = {}}));
      VerifyFd<kBS, Elem>(SyntheticResponse<kBS>(IsoModulus{.lambda = 9_r, .mu = 6_r, .s0 = s0}));
    });
  });
}

TEST(StressDampingWork, TangentFdRealMaterials) {
  RunBatchSizes<1, 4, 8>([&]<int kBS>() {
    RunElementTypes([&]<class Elem>() {
      VerifyFd<kBS, Elem>(StVenantResponse<kBS>());
      VerifyFd<kBS, Elem>(SmithNeoHookeanResponse<kBS>());
      VerifyFd<kBS, Elem>(LinearResponse<kBS>());
    });
  });
}

TEST(StressDampingWork, PsdProjection) {
  RunBatchSizes<1, 4, 8>([&]<int kBS>() {
    RunElementTypes([&]<class Elem>() {
      VerifyPsd<kBS, Elem>(SyntheticResponse<kBS>(IsoModulus{.lambda = 9_r, .mu = 6_r, .s0 = {}}));
      VerifyPsd<kBS, Elem>(StVenantResponse<kBS>(MaterialPsdStrategy::Projection));
      VerifyPsd<kBS, Elem>(SmithNeoHookeanResponse<kBS>(MaterialPsdStrategy::Projection));
    });
  });
}

TEST(StressDampingWork, DResSymmetryWithRestStress) {
  RunBatchSizes<1, 4, 8>([&]<int kBS>() {
    RunElementTypes([&]<class Elem>() { VerifyDResSymmetryWithRestStress<kBS, Elem>(); });
  });
}

TEST(StressDampingWork, StVenantKirchhoffEquivalence) {
  RunBatchSizes<1, 4, 8>([&]<int kBS>() {
    RunElementTypes([&]<class Elem>() { VerifyStVenantEquivalence<kBS, Elem>(); });
  });
}

TEST(StressDampingWork, GeometricTermGating) {
  RunBatchSizes<1, 4, 8>([&]<int kBS>() {
    RunElementTypes([&]<class Elem>() { VerifyGeometricTermGating<kBS, Elem>(); });
  });
}

TEST(StressDampingWork, IsotropicDenseEquivalence) {
  RunBatchSizes<1, 4, 8>([&]<int kBS>() {
    RunElementTypes([&]<class Elem>() {
      // Synthetic isotropic C₀ (no rest stress) and the three passive isotropic materials.
      VerifyIsotropicDenseEquivalence<kBS, Elem>(
          SyntheticResponse<kBS>(IsoModulus{.lambda = 9_r, .mu = 6_r, .s0 = {}}));
      VerifyIsotropicDenseEquivalence<kBS, Elem>(StVenantResponse<kBS>());
      VerifyIsotropicDenseEquivalence<kBS, Elem>(SmithNeoHookeanResponse<kBS>());
      VerifyIsotropicDenseEquivalence<kBS, Elem>(LinearResponse<kBS>());
      // Heterogeneous isotropic C₀: per-lane λ,μ differ across the batch, exercising the fast
      // path's per-lane C₀ᵥ scalar reads (the cases above give every lane identical λ,μ).
      VerifyIsotropicDenseEquivalence<kBS, Elem>(HeterogeneousIsoResponse<kBS>());
    });
  });
}

TEST(StressDampingWork, AssemblySubsetConsistency) {
  RunBatchSizes<1, 4, 8>([&]<int kBS>() {
    RunElementTypes([&]<class Elem>() { VerifyAssemblySubsetConsistency<kBS, Elem>(); });
  });
}
