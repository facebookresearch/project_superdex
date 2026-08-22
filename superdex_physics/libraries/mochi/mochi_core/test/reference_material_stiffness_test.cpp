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

#include <mochi_core/materials/batched_materials.h>
#include <mochi_core/materials/reference_material_stiffness.h>
#include <mochi_core/test/batched_work_test_utils.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

#include <gtest/gtest.h>

using namespace mochi;
using namespace mochi::fem;
using namespace mochi::materials;

// The reference material stiffness store precomputes the Lagrangian zero-deformation stiffness C₀
// (symmetric 6×6 Voigt) from a batched constitutive response. These tests validate the build and
// gather helpers against an independent isotropic C₀ oracle, covering:
//   - homogeneous build (size-1): gather broadcasts to every lane and matches direct compute,
//   - heterogeneous build (size-N): gather returns the correct per-lane tensor,
//   - rest-stress strip: a nonzero synthetic S₀ is removed, leaving a symmetric isotropic C₀.

namespace {

// Isotropic C₀ in Voigt ordering [00, 11, 22, 12, 02, 01], no engineering-shear doubling
// (matches ComputeReferenceMaterialStiffnessVoigt: C₀ᵥ[a][b] = C_{i_a j_a i_b j_b}).
NdArray<real, 6, 6> ExpectedIsoC0v(real lambda, real mu) {
  NdArray<real, 6, 6> c{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = lambda + (i == j ? 2_r * mu : 0_r);
    }
  }
  c[3][3] = mu;
  c[4][4] = mu;
  c[5][5] = mu;
  return c;
}

// Opt-in isotropy trait: the four passive materials (and the NeoHookean alias) declare their
// reference tangent isotropic; the two active materials keep the default false and fall back to the
// dense contraction. Verified at compile time so a mis-declared specialization fails the build.
static_assert(kIsotropicReferenceStiffness<LinearElasticMaterialParams>);
static_assert(kIsotropicReferenceStiffness<StVenantKirchhoffMaterialParams>);
static_assert(kIsotropicReferenceStiffness<SmithNeoHookeanMaterialParams>);
static_assert(kIsotropicReferenceStiffness<ArapMaterialParams>);
static_assert(!kIsotropicReferenceStiffness<ActiveNeoHookeanMaterialParams>);
static_assert(!kIsotropicReferenceStiffness<ActiveShapeTargetingArapMaterialParams>);

// Opt-in correctness: for an opted-in material, form the real C₀ᵥ and assert it matches the
// isotropic template built from the two scalars read out of it (λ = C₀ᵥ[0][1], μ = C₀ᵥ[3][3]). This
// verifies the 24 vanishing normal-shear couplings, equal shear diagonals, and the λ+2μ / λ normal
// block — i.e. that the material really has the 2-parameter isotropic structure the fast path
// assumes. Homogeneous params, so every lane is identical.
template <int kBS, class ParamsT>
void ExpectC0vIsotropic(ParamsT const& params) {
  auto const perElem = materials::BuildPerElementParams(params);
  auto const resp = materials::MakeBatchedConstitutiveResponse<ParamsT, kBS>(perElem);
  NdArray<int, kBS> idx{};
  auto const c0v = ComputeReferenceMaterialStiffnessVoigt<kBS>(idx, resp);

  real const relTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-9_r : 5e-4_r;
  for (int b = 0; b < kBS; ++b) {
    auto const expected = ExpectedIsoC0v(c0v[0][1][b], c0v[3][3][b]);
    real scale = 1_r;
    for (int a = 0; a < 6; ++a) {
      for (int c = 0; c < 6; ++c) {
        scale = Max(scale, Abs(c0v[a][c][b]));
      }
    }
    for (int a = 0; a < 6; ++a) {
      for (int c = 0; c < 6; ++c) {
        EXPECT_LE(Abs(expected[a][c] - c0v[a][c][b]), relTol * scale);
      }
    }
  }
}

// Synthetic isotropic constitutive response with per-element (λ, μ) and a fixed symmetric rest
// stress S₀. Returns the two-point tangent ∂P/∂F|₀ = C₀ + δ⊗S₀ (the geometric part the helper must
// strip) and the rest stress S₀. F is ignored; the helper only evaluates at F = I. When λ/μ hold a
// single entry the response is homogeneous (index 0 for all lanes).
template <int kBS>
auto MakeIsoResponse(Span<real const> lambda, Span<real const> mu, Matrix3x3r s0) {
  return [lambda, mu, s0](
             NdArray<int, kBS> const& idx,
             auto const& /*F*/,
             auto* e,
             auto* pk1,
             auto* tangent,
             bool /*psd*/) {
    using V = BatchReal<kBS>;
    if (e) {
      *e = BatchDouble<kBS>{0.0};
    }
    alignas(alignof(V)) real lamStg[V::kSize]{};
    alignas(alignof(V)) real muStg[V::kSize]{};
    bool const homogeneous = (lambda.size() == 1);
    for (int b = 0; b < kBS; ++b) {
      int const i = homogeneous ? 0 : idx[b];
      lamStg[b] = lambda[i];
      muStg[b] = mu[i];
    }
    V const lam = Load<V>(lamStg);
    V const m = Load<V>(muStg);
    if (pk1) {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          (*pk1)[i][j] = V{s0[i][j]};
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
              V val = lam * V{cLam} + m * V{cMu};
              if (i == r) {
                val += V{s0[c][j]}; // geometric rest-stress part of the two-point tangent
              }
              (*tangent)[i][j][r][c] = val;
            }
          }
        }
      }
    }
  };
}

real Tol() {
  return MOCHI_USE_DOUBLE_PRECISION ? 1e-9_r : 1e-4_r;
}

// Homogeneous build: single stored tensor, broadcast to every lane by gather and equal to a direct
// compute.
template <int kBS>
void VerifyHomogeneousBuildGather() {
  DynamicArray<real> const lambda{11_r};
  DynamicArray<real> const mu{7_r};
  Matrix3x3r const s0{}; // passive

  auto const store = BuildPerElementReferenceMaterialStiffness(
      MakeIsoResponse<1>(MakeConstSpan(lambda), MakeConstSpan(mu), s0),
      /*numEntries*/ 1,
      /*isIsotropic*/ false);
  EXPECT_EQ(isize(store.data), 1);

  auto const expected = ExpectedIsoC0v(lambda[0], mu[0]);

  NdArray<int, kBS> idx;
  for (int b = 0; b < kBS; ++b) {
    idx[b] = (3 * b + 1) % 5; // arbitrary; homogeneous gather ignores indices
  }
  auto const gathered = GatherReferenceMaterialStiffnessVoigt<kBS>(store, idx);
  auto const direct = ComputeReferenceMaterialStiffnessVoigt<kBS>(
      idx, MakeIsoResponse<kBS>(MakeConstSpan(lambda), MakeConstSpan(mu), s0));

  real const tol = Tol();
  for (int b = 0; b < kBS; ++b) {
    for (int a = 0; a < 6; ++a) {
      for (int c = 0; c < 6; ++c) {
        EXPECT_NEAR(expected[a][c], gathered[a][c][b], tol);
        EXPECT_NEAR(direct[a][c][b], gathered[a][c][b], tol);
      }
    }
  }
}

// Heterogeneous build: one tensor per element, gathered per lane by index.
template <int kBS>
void VerifyHeterogeneousBuildGather() {
  constexpr int kNumElems = 5;
  DynamicArray<real> lambda(kNumElems);
  DynamicArray<real> mu(kNumElems);
  for (int i = 0; i < kNumElems; ++i) {
    lambda[i] = 10_r + static_cast<real>(i);
    mu[i] = 5_r + 2_r * static_cast<real>(i);
  }
  Matrix3x3r const s0{};

  auto const store = BuildPerElementReferenceMaterialStiffness(
      MakeIsoResponse<1>(MakeConstSpan(lambda), MakeConstSpan(mu), s0),
      kNumElems,
      /*isIsotropic*/ false);
  EXPECT_EQ(isize(store.data), kNumElems);

  NdArray<int, kBS> idx;
  for (int b = 0; b < kBS; ++b) {
    idx[b] = (2 * b + 1) % kNumElems;
  }
  auto const gathered = GatherReferenceMaterialStiffnessVoigt<kBS>(store, idx);

  real const tol = Tol();
  for (int b = 0; b < kBS; ++b) {
    auto const expected = ExpectedIsoC0v(lambda[idx[b]], mu[idx[b]]);
    for (int a = 0; a < 6; ++a) {
      for (int c = 0; c < 6; ++c) {
        EXPECT_NEAR(expected[a][c], gathered[a][c][b], tol);
      }
    }
  }
}

// A nonzero synthetic rest stress S₀ must be stripped, leaving the symmetric isotropic C₀.
template <int kBS>
void VerifyRestStressStripped() {
  Matrix3x3r const s0 = {Real3{3_r, 1_r, -2_r}, Real3{1_r, -4_r, 0.5_r}, Real3{-2_r, 0.5_r, 5_r}};
  DynamicArray<real> const lambda{12_r};
  DynamicArray<real> const mu{8_r};

  auto const store = BuildPerElementReferenceMaterialStiffness(
      MakeIsoResponse<1>(MakeConstSpan(lambda), MakeConstSpan(mu), s0),
      /*numEntries*/ 1,
      /*isIsotropic*/ false);
  ASSERT_EQ(isize(store.data), 1);

  auto const& c0 = store.data[0];
  auto const expected = ExpectedIsoC0v(lambda[0], mu[0]);

  real const tol = Tol();
  for (int a = 0; a < 6; ++a) {
    for (int b = 0; b < 6; ++b) {
      EXPECT_NEAR(expected[a][b], c0[a][b], tol); // strip recovers the isotropic C₀
      EXPECT_NEAR(c0[a][b], c0[b][a], tol); // symmetric
    }
  }
}

} // namespace

TEST(ReferenceMaterialStiffness, HomogeneousBuildGather) {
  RunBatchSizes<1, 4, 8>([&]<int kBS>() { VerifyHomogeneousBuildGather<kBS>(); });
}

TEST(ReferenceMaterialStiffness, HeterogeneousBuildGather) {
  RunBatchSizes<1, 4, 8>([&]<int kBS>() { VerifyHeterogeneousBuildGather<kBS>(); });
}

TEST(ReferenceMaterialStiffness, RestStressStripped) {
  RunBatchSizes<1, 4, 8>([&]<int kBS>() { VerifyRestStressStripped<kBS>(); });
}

// Each opted-in passive material has a genuinely isotropic reference C₀ᵥ, across representative
// stiffness / Poisson spreads (λ ≫ μ near-incompressible, λ ≈ μ, and λ = 0 at ν = 0).
TEST(ReferenceMaterialStiffness, OptInMaterialsAreIsotropic) {
  RunBatchSizes<1, 4, 8>([&]<int kBS>() {
    for (real const poisson : {0_r, 1_r / 6_r, 0.49_r}) { // λ=0, λ≈μ, λ≫μ
      for (real const youngs : {1e2_r, 1e6_r}) { // soft and stiff
        ExpectC0vIsotropic<kBS>(
            LinearElasticMaterialParams{.youngsModulus = youngs, .poissonRatio = poisson});
        ExpectC0vIsotropic<kBS>(
            StVenantKirchhoffMaterialParams{.youngsModulus = youngs, .poissonRatio = poisson});
        ExpectC0vIsotropic<kBS>(
            SmithNeoHookeanMaterialParams{.youngsModulus = youngs, .poissonRatio = poisson});
      }
    }
    for (real const stiffness : {1e2_r, 1e4_r}) {
      ExpectC0vIsotropic<kBS>(ArapMaterialParams{.stiffness = stiffness});
    }
  });
}

// The build helper caches the isotropy flag onto the store; opted-in materials get true, active
// materials get false.
TEST(ReferenceMaterialStiffness, StoreFlagMatchesTrait) {
  auto const passiveParams = BuildPerElementParams(SmithNeoHookeanMaterialParams{});
  auto const passiveResp =
      MakeBatchedConstitutiveResponse<SmithNeoHookeanMaterialParams, 1>(passiveParams);
  auto const passiveStore = BuildPerElementReferenceMaterialStiffness(
      passiveResp, /*numEntries*/ 1, kIsotropicReferenceStiffness<SmithNeoHookeanMaterialParams>);
  EXPECT_TRUE(passiveStore.isIsotropic);

  auto const activeParams = BuildPerElementParams(ActiveNeoHookeanMaterialParams{});
  auto const activeResp =
      MakeBatchedConstitutiveResponse<ActiveNeoHookeanMaterialParams, 1>(activeParams);
  auto const activeStore = BuildPerElementReferenceMaterialStiffness(
      activeResp, /*numEntries*/ 1, kIsotropicReferenceStiffness<ActiveNeoHookeanMaterialParams>);
  EXPECT_FALSE(activeStore.isIsotropic);
}
