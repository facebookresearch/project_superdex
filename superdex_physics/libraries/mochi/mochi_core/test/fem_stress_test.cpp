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
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/elements/tetrahedral/simplex_quadrature.h>
#include <mochi_core/elements/triangular/finite_element.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/materials/batched_kim_neo_hookean.h>
#include <mochi_core/materials/batched_linear_elastic.h>
#include <mochi_core/materials/batched_smith_neo_hookean.h>
#include <mochi_core/materials/batched_st_venant_kirchhoff.h>
#include <mochi_core/test/batched_work_test_utils.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/rand_utils.h>

#include <gtest/gtest.h>

using namespace mochi;
using namespace mochi::materials;
using namespace mochi::fem;

// The batched stress kernel maps a constitutive model (energy ψ, PK1 = ∂ψ/∂F, tangent = ∂²ψ/∂F²)
// onto an element. It is validated against a constitutive-agnostic, scalar-op-free oracle:
//   - residual is the gradient of the energy   (res ≈ ∂E/∂u, central differences)
//   - dresidual is the gradient of the residual (dRes ≈ ∂res/∂u, central differences, tets only)
//   - rest state (zero displacement) has zero energy
//   - PSD projection yields a symmetric positive-semi-definite dresidual.
// Finite differences are run across several constitutives (Kim-NeoHookean, StVenant-Kirchhoff,
// LinearElastic) so the assembly is exercised against the full PK1/tangent plumbing of each.
// Because FD only checks consistency (it cannot catch a self-consistent magnitude error in the
// assembled F / energy), a closed-form linear-elastic anchor additionally checks the assembled
// energy magnitude for a known affine deformation. Output-mode and per-element-weight coverage uses
// an independent linear-elastic energy oracle plus finite differences. PSD-strategy coverage
// (Projection / Fast / AbsEigenProjection) checks that a projected constitutive tangent yields an
// SPD assembled tangent. The constitutive models themselves are covered independently by the
// materials tests. Each lane uses a distinct element / displacement (mixed-lane).

using TriElement = triangular::Pk2DElement<1, 1>;
using TetElement = tetrahedral::Pk3DElement<1, 1>;

static constexpr real kYoungs = 100_r;
static constexpr real kPoisson = 0.3_r;
static constexpr int kFdTrials = 3;

// Per-material batched constitutive callbacks. The callback signature matches what StressWork
// expects: (elementIndices, F, outEnergy*, outPK1*, outTangent*, projectPsd).
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
static auto KimResponse(MaterialPsdStrategy psd = MaterialPsdStrategy::None) {
  return LameResponse<KimNeoHookeanMaterialParams, kBS>(
      psd, [](auto const& lame, auto const& F, auto* e, auto* pk1, auto* tangent, bool project) {
        BatchedKimNeoHookeanConstitutiveResponse<kBS>(lame, F, e, pk1, tangent, project);
      });
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

static void ExpectNearRel(real a, real b) {
  real const scale = Max(Abs(a), kRelTol);
  EXPECT_LE(Abs(a - b), kRelTol * scale);
}

// Runs the kernel for a per-lane displacement; fills the requested (lane-indexed) outputs.
template <class ElementT, int kBS, class Constitutive>
static void RunStress(
    Span<ElementT const> elemSpan,
    NdArray<int, kBS> const& idx,
    NdArray<NdArray<real, ElementT::kSpaceDim * ElementT::kNumDofs>, kBS> const& disp,
    bool wantE,
    bool wantR,
    bool wantD,
    bool projectPsd,
    Constitutive const& constitutive,
    NdArray<double, kBS>& energy,
    NdArray<NdArray<real, ElementT::kSpaceDim * ElementT::kNumDofs>, kBS>& res,
    NdArray<
        NdArray<
            real,
            (ElementT::kSpaceDim * ElementT::kNumDofs) *
                (ElementT::kSpaceDim * ElementT::kNumDofs)>,
        kBS>& dres,
    Span<real const> perElementExtraWeight = {}) {
  constexpr int kDim = ElementT::kSpaceDim * ElementT::kNumDofs;
  using V = BatchReal<kBS>;
  fem::BatchElementVector<kBS, ElementT> dispBatch;
  alignas(alignof(V)) real staging[V::kSize]{};
  for (int k = 0; k < kDim; ++k) {
    for (int b = 0; b < kBS; ++b) {
      staging[b] = disp[b][k];
    }
    dispBatch[k] = Load<V>(staging);
  }
  BatchDouble<kBS> e{0.0};
  fem::BatchElementVector<kBS, ElementT> r{};
  fem::BatchElementMatrix<kBS, ElementT> d{};
  StressWork<kBS>(
      idx,
      elemSpan,
      dispBatch,
      wantE ? &e : nullptr,
      wantR ? &r : nullptr,
      wantD ? &d : nullptr,
      projectPsd,
      constitutive,
      perElementExtraWeight);
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

static double LinearElasticEnergyOracle(
    TetElement const& element,
    NdArray<real, TetElement::kSpaceDim * TetElement::kNumDofs> const& disp,
    real extraWeight) {
  constexpr int kNumNodes = TetElement::kNumDofs;
  static_assert(TetElement::kSpaceDim == 3);

  real const mu = kYoungs / (2_r * (1_r + kPoisson));
  real const lambda = kYoungs * kPoisson / ((1_r + kPoisson) * (1_r - 2_r * kPoisson));
  double energy = 0.0;

  for (int q = 0; q < TetElement::kNumQuadPoints; ++q) {
    Matrix3x3r F = Eye<3, real>();
    for (int f = 0; f < kNumNodes; ++f) {
      for (int r = 0; r < TetElement::kSpaceDim; ++r) {
        for (int c = 0; c < TetElement::kSpaceDim; ++c) {
          F[r][c] += disp[f * TetElement::kSpaceDim + r] * element.dBasisEvaluated[q][f][c];
        }
      }
    }

    real trEps = 0_r;
    real epsNormSq = 0_r;
    for (int r = 0; r < TetElement::kSpaceDim; ++r) {
      for (int c = 0; c < TetElement::kSpaceDim; ++c) {
        real const eps = 0.5_r * (F[r][c] + F[c][r]) - (r == c ? 1_r : 0_r);
        epsNormSq += eps * eps;
        if (r == c) {
          trEps += eps;
        }
      }
    }
    real const psi = mu * epsNormSq + 0.5_r * lambda * trEps * trEps;
    energy += static_cast<double>(psi) * static_cast<double>(element.quadWeights[q] * extraWeight);
  }
  return energy;
}

static NdArray<real, TetElement::kSpaceDim * TetElement::kNumDofs> LinearElasticResidualOracle(
    TetElement const& element,
    NdArray<real, TetElement::kSpaceDim * TetElement::kNumDofs> const& disp,
    real extraWeight) {
  constexpr int kDim = TetElement::kSpaceDim * TetElement::kNumDofs;
  real const eps = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-3_r;

  NdArray<real, kDim> res{};
  for (int i = 0; i < kDim; ++i) {
    auto dispP = disp;
    auto dispM = disp;
    dispP[i] += eps;
    dispM[i] -= eps;
    res[i] = static_cast<real>(
        (LinearElasticEnergyOracle(element, dispP, extraWeight) -
         LinearElasticEnergyOracle(element, dispM, extraWeight)) /
        (2.0 * static_cast<double>(eps)));
  }
  return res;
}

static auto LinearElasticDResOracle(
    TetElement const& element,
    NdArray<real, TetElement::kSpaceDim * TetElement::kNumDofs> const& disp,
    real extraWeight) {
  constexpr int kDim = TetElement::kSpaceDim * TetElement::kNumDofs;
  real const eps = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-3_r;

  NdArray<real, kDim * kDim> dres{};
  for (int j = 0; j < kDim; ++j) {
    auto dispP = disp;
    auto dispM = disp;
    dispP[j] += eps;
    dispM[j] -= eps;
    auto const rp = LinearElasticResidualOracle(element, dispP, extraWeight);
    auto const rm = LinearElasticResidualOracle(element, dispM, extraWeight);
    for (int i = 0; i < kDim; ++i) {
      dres[i * kDim + j] = (rp[i] - rm[i]) / (2_r * eps);
    }
  }
  return dres;
}

// FD validation (res = ∂E/∂u, dRes = ∂res/∂u for tets) + rest state, for a given constitutive.
template <class ElementT, template <class> class MeshDataT, bool kSupportsDRes, int kBS, class C>
static void VerifyStressFd(C const& constitutive) {
  constexpr int kDim = ElementT::kSpaceDim * ElementT::kNumDofs;

  auto const data = MeshDataT<ElementT>::CreateMinimalCube();
  int const numElements = isize(data.elements);
  auto const elemSpan = MakeConstSpan(data.elements);

  real const eps = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-3_r;
  real const resTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-4_r : 2e-2_r;
  real const dresTol = MOCHI_USE_DOUBLE_PRECISION ? 2e-3_r : 5e-2_r;

  NdArray<double, kBS> e0, ep, em;
  NdArray<NdArray<real, kDim>, kBS> baseRes, rp, rm, scratchRes;
  NdArray<NdArray<real, kDim * kDim>, kBS> baseDRes, scratchDRes;

  for (int trial = 0; trial < kFdTrials; ++trial) {
    NdArray<int, kBS> idx;
    for (int b = 0; b < kBS; ++b) {
      idx[b] = (trial + b) % numElements;
    }
    NdArray<NdArray<real, kDim>, kBS> disp;
    for (int b = 0; b < kBS; ++b) {
      disp[b] = MakeRandomArray<kDim>(static_cast<unsigned int>(trial * kBS + b), -0.05_r, 0.05_r);
    }

    RunStress<ElementT, kBS>(
        elemSpan,
        idx,
        disp,
        false,
        true,
        kSupportsDRes,
        false,
        constitutive,
        e0,
        baseRes,
        baseDRes);

    // Residual == d(energy)/d(disp).
    NdArray<NdArray<real, kDim>, kBS> resFd{};
    for (int i = 0; i < kDim; ++i) {
      auto dispP = disp;
      auto dispM = disp;
      for (int b = 0; b < kBS; ++b) {
        dispP[b][i] += eps;
        dispM[b][i] -= eps;
      }
      RunStress<ElementT, kBS>(
          elemSpan,
          idx,
          dispP,
          true,
          false,
          false,
          false,
          constitutive,
          ep,
          scratchRes,
          scratchDRes);
      RunStress<ElementT, kBS>(
          elemSpan,
          idx,
          dispM,
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

    // Dresidual == d(residual)/d(disp) (tets only; raw tangent, projectPsd == false).
    if constexpr (kSupportsDRes) {
      NdArray<NdArray<real, kDim * kDim>, kBS> dresFd{};
      for (int j = 0; j < kDim; ++j) {
        auto dispP = disp;
        auto dispM = disp;
        for (int b = 0; b < kBS; ++b) {
          dispP[b][j] += eps;
          dispM[b][j] -= eps;
        }
        RunStress<ElementT, kBS>(
            elemSpan, idx, dispP, false, true, false, false, constitutive, ep, rp, scratchDRes);
        RunStress<ElementT, kBS>(
            elemSpan, idx, dispM, false, true, false, false, constitutive, em, rm, scratchDRes);
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

  // Rest state: zero displacement => zero energy.
  {
    NdArray<int, kBS> idx;
    for (int b = 0; b < kBS; ++b) {
      idx[b] = b % numElements;
    }
    NdArray<NdArray<real, kDim>, kBS> zero{};
    RunStress<ElementT, kBS>(
        elemSpan, idx, zero, true, false, false, false, constitutive, e0, scratchRes, scratchDRes);
    for (int b = 0; b < kBS; ++b) {
      EXPECT_NEAR(0.0, e0[b], 1e-9);
    }
  }
}

template <int kBS>
static void VerifyOutputModesAndWeights(OutputConfig cfg) {
  constexpr int kDim = TetElement::kSpaceDim * TetElement::kNumDofs;

  auto const data = TestTetMeshData<TetElement>::CreateMinimalCube();
  int const numElements = isize(data.elements);
  auto const elemSpan = MakeConstSpan(data.elements);
  auto const constitutive = LinearResponse<kBS>();

  DynamicArray<real> extraWeights(numElements);
  for (int i = 0; i < numElements; ++i) {
    extraWeights[i] = 1_r + 0.1_r * static_cast<real>(i);
  }

  real const resTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-4_r : 2e-2_r;
  real const dresTol = MOCHI_USE_DOUBLE_PRECISION ? 2e-3_r : 5e-2_r;

  for (int trial = 0; trial < kFdTrials; ++trial) {
    NdArray<int, kBS> idx;
    NdArray<NdArray<real, kDim>, kBS> disp;
    for (int b = 0; b < kBS; ++b) {
      idx[b] = (trial + b) % numElements;
      disp[b] =
          MakeRandomArray<kDim>(static_cast<unsigned int>(631 + trial * kBS + b), -0.05_r, 0.05_r);
    }

    NdArray<double, kBS> energy;
    NdArray<NdArray<real, kDim>, kBS> res;
    NdArray<NdArray<real, kDim * kDim>, kBS> dres;
    RunStress<TetElement, kBS>(
        elemSpan,
        idx,
        disp,
        cfg.energy,
        cfg.residual,
        cfg.dresidual,
        false,
        constitutive,
        energy,
        res,
        dres,
        MakeConstSpan(extraWeights));

    for (int b = 0; b < kBS; ++b) {
      TetElement const& element = data.elements[idx[b]];
      real const extraWeight = extraWeights[idx[b]];
      if (cfg.energy) {
        ExpectNearEnergy(LinearElasticEnergyOracle(element, disp[b], extraWeight), energy[b]);
      }
      if (cfg.residual) {
        ExpectNearL2(LinearElasticResidualOracle(element, disp[b], extraWeight), res[b], resTol);
      }
      if (cfg.dresidual) {
        ExpectNearL2(LinearElasticDResOracle(element, disp[b], extraWeight), dres[b], dresTol);
      }
    }
  }
}

// Closed-form linear-elastic energy magnitude anchor (tets). For an affine deformation u = (A -
// I)X, the P1 deformation gradient is exactly A on every element, so the assembled energy must
// equal ψ_linear(A) * V_e with ψ_linear(A) = μ‖ε‖² + (λ/2)tr(ε)², ε = sym(A) - I. This validates
// the assembled F and the energy scale (which pure FD, being consistency-only, cannot).
template <int kBS>
static void VerifyEnergyMagnitudeTet() {
  constexpr int kNumNodes = TetElement::kNumDofs;
  constexpr int kDim = TetElement::kSpaceDim * kNumNodes;

  auto const data = TestTetMeshData<TetElement>::CreateMinimalCube();
  int const numElements = isize(data.elements);
  auto const elemSpan = MakeConstSpan(data.elements);
  auto const constitutive = LinearResponse<kBS>();

  Matrix3x3r const A = {
      Real3{1.05_r, 0.03_r, -0.02_r},
      Real3{0.02_r, 0.96_r, 0.03_r},
      Real3{-0.015_r, 0.035_r, 1.04_r}};

  real const mu = kYoungs / (2_r * (1_r + kPoisson));
  real const lambda = kYoungs * kPoisson / ((1_r + kPoisson) * (1_r - 2_r * kPoisson));
  real trEps = 0_r;
  real epsNormSq = 0_r;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      real const eps = 0.5_r * (A[r][c] + A[c][r]) - (r == c ? 1_r : 0_r);
      epsNormSq += eps * eps;
      if (r == c) {
        trEps += eps;
      }
    }
  }
  real const psiLinear = mu * epsNormSq + 0.5_r * lambda * trEps * trEps;

  NdArray<int, kBS> idx;
  NdArray<NdArray<real, kDim>, kBS> disp{};
  NdArray<double, kBS> energyRef;
  for (int b = 0; b < kBS; ++b) {
    int const e = b % numElements;
    idx[b] = e;
    TetElement const& el = data.elements[e];
    for (int f = 0; f < kNumNodes; ++f) {
      for (int d = 0; d < 3; ++d) {
        real u = 0_r;
        for (int c = 0; c < 3; ++c) {
          u += (A[d][c] - (d == c ? 1_r : 0_r)) * el.nodesCrdsPhys[f][c];
        }
        disp[b][f * 3 + d] = u;
      }
    }
    real volume = 0_r;
    for (int q = 0; q < TetElement::kNumQuadPoints; ++q) {
      volume += el.quadWeights[q];
    }
    energyRef[b] = static_cast<double>(psiLinear) * static_cast<double>(volume);
  }

  NdArray<double, kBS> energy;
  NdArray<NdArray<real, kDim>, kBS> scratchRes;
  NdArray<NdArray<real, kDim * kDim>, kBS> scratchDRes;
  RunStress<TetElement, kBS>(
      elemSpan,
      idx,
      disp,
      true,
      false,
      false,
      false,
      constitutive,
      energy,
      scratchRes,
      scratchDRes);

  real const tol = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 1e-4_r;
  for (int b = 0; b < kBS; ++b) {
    ExpectNearEnergy(energyRef[b], energy[b], tol);
  }
}

// A projected constitutive tangent must yield a symmetric, positive-semi-definite assembled
// dresidual (tets). Exercised across several PSD strategies.
template <int kBS, class C>
static void VerifyPsdStrategy(C const& constitutive) {
  constexpr int kDim = TetElement::kSpaceDim * TetElement::kNumDofs;
  auto const data = TestTetMeshData<TetElement>::CreateMinimalCube();
  int const numElements = isize(data.elements);
  auto const elemSpan = MakeConstSpan(data.elements);

  NdArray<int, kBS> idx;
  NdArray<NdArray<real, kDim>, kBS> disp;
  for (int b = 0; b < kBS; ++b) {
    idx[b] = b % numElements;
    disp[b] = MakeRandomArray<kDim>(static_cast<unsigned int>(4242 + b), -0.05_r, 0.05_r);
  }
  NdArray<double, kBS> energy;
  NdArray<NdArray<real, kDim>, kBS> scratchRes;
  NdArray<NdArray<real, kDim * kDim>, kBS> dres;
  RunStress<TetElement, kBS>(
      elemSpan, idx, disp, false, false, true, true, constitutive, energy, scratchRes, dres);

  for (int b = 0; b < kBS; ++b) {
    real diagSum = 0_r;
    for (int i = 0; i < kDim; ++i) {
      diagSum += Abs(dres[b][i * kDim + i]);
    }
    // Symmetry.
    for (int i = 0; i < kDim; ++i) {
      for (int j = 0; j < kDim; ++j) {
        ExpectNearRel(dres[b][i * kDim + j], dres[b][j * kDim + i]);
      }
    }
    // vᵀ K v >= 0 (allowing a small projection/round-off slack relative to the matrix scale).
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

TEST(StressWork, TriFd) {
  RunBatchSizes<1, 4, 8>([&]<int kBS>() {
    VerifyStressFd<TriElement, TestTriMeshData, /*kSupportsDRes*/ false, kBS>(KimResponse<kBS>());
  });
}

TEST(StressWork, TetFd) {
  RunBatchSizes<1, 4, 8>([&]<int kBS>() {
    VerifyStressFd<TetElement, TestTetMeshData, /*kSupportsDRes*/ true, kBS>(KimResponse<kBS>());
    VerifyStressFd<TetElement, TestTetMeshData, /*kSupportsDRes*/ true, kBS>(
        StVenantResponse<kBS>());
    VerifyStressFd<TetElement, TestTetMeshData, /*kSupportsDRes*/ true, kBS>(LinearResponse<kBS>());
  });
}

TEST(StressWork, EnergyMagnitude) {
  RunBatchSizes<1, 4, 8>([&]<int kBS>() { VerifyEnergyMagnitudeTet<kBS>(); });
}

TEST(StressWork, OutputModesAndWeights) {
  for (auto cfg : kAllOutputConfigs) {
    RunBatchSizes<1, 4, 8>([&]<int kBS>() { VerifyOutputModesAndWeights<kBS>(cfg); });
  }
}

TEST(StressWork, PsdStrategies) {
  RunBatchSizes<1, 4, 8>([&]<int kBS>() {
    for (auto psd :
         {MaterialPsdStrategy::Projection,
          MaterialPsdStrategy::Fast,
          MaterialPsdStrategy::AbsEigenProjection}) {
      VerifyPsdStrategy<kBS>(KimResponse<kBS>(psd));
      VerifyPsdStrategy<kBS>(StVenantResponse<kBS>(psd));
      VerifyPsdStrategy<kBS>(SmithNeoHookeanResponse<kBS>(psd));
    }
  });
}
