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

#include <mochi_core/element_operations/fem_gravity.h>
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/elements/triangular/finite_element.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/test/batched_work_test_utils.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>

#include <gtest/gtest.h>

using namespace mochi;
using namespace mochi::fem;

// Gravity is linear in the displacement, so it admits a closed-form oracle:
//   - residual is independent of the displacement,
//   - for the P1 elements used here the nodal force is uniform across an element's nodes:
//         r_{i,d} = -density * w_e * (V_e / kNumNodes) * gravity_d,
//     where V_e = sum_q quadWeights (the element volume) and w_e the per-element extra weight,
//   - energy is the quadrature sum of -dot(y_q, density * gravity) * w_q.
// Each batch lane gets a distinct displacement; element indices and extra weights cycle by lane and
// trial, covering mixed and duplicate lanes. "Test passes" implies the kernel is correct, including
// the zero-gravity and per-output-flag edge cases.

using TetGravityElement = tetrahedral::Pk3DElement<1, 1>;
using TriGravityElement = triangular::Pk2DElement<1, 1>;

namespace {

// Packs a per-lane array-of-displacements into a batched displacement vector.
template <class ElementT, int kBS>
fem::BatchElementVector<kBS, ElementT> PackDisp(
    NdArray<NdArray<real, ElementT::kSpaceDim * ElementT::kNumDofs>, kBS> const& disp) {
  using V = BatchReal<kBS>;
  constexpr int kDim = ElementT::kSpaceDim * ElementT::kNumDofs;
  fem::BatchElementVector<kBS, ElementT> out{};
  alignas(alignof(V)) real staging[V::kSize]{};
  for (int f = 0; f < kDim; ++f) {
    for (int b = 0; b < kBS; ++b) {
      staging[b] = disp[b][f];
    }
    out[f] = Load<V>(staging);
  }
  return out;
}

// Relative-tolerance scalar comparison.
void ExpectNearRel(real expected, real actual, real relTol = kRelTol) {
  real const scale = Max(Abs(expected), relTol);
  EXPECT_LE(Abs(expected - actual), relTol * scale);
}

template <class ElementT>
double ExpectedGravityEnergy(
    ElementT const& element,
    NdArray<real, ElementT::kSpaceDim * ElementT::kNumDofs> const& disp,
    Real3 const& gravity,
    real density,
    real extraWeight) {
  constexpr int kNumNodes = ElementT::kNumDofs;
  constexpr int kSpaceDim = ElementT::kSpaceDim;
  double energy = 0.0;
  for (int q = 0; q < ElementT::kNumQuadPoints; ++q) {
    double dotYG = 0.0;
    for (int d = 0; d < kSpaceDim; ++d) {
      real y = element.mapEvaluated[q][d];
      for (int f = 0; f < kNumNodes; ++f) {
        y += element.basisEvaluated[q][f] * disp[f * kSpaceDim + d];
      }
      dotYG += static_cast<double>(y) * static_cast<double>(gravity[d]);
    }
    energy -= static_cast<double>(density * extraWeight * element.quadWeights[q]) * dotYG;
  }
  return energy;
}

template <class ElementT, template <class> class MeshDataT, int kBS>
void VerifyGravityAnalytic() {
  constexpr int kNumNodes = ElementT::kNumDofs;
  constexpr int kSpaceDim = ElementT::kSpaceDim;
  constexpr int kDim = kSpaceDim * kNumNodes;
  static_assert(kSpaceDim == 3);

  auto const data = MeshDataT<ElementT>::CreateMinimalCube();
  int const numElements = isize(data.elements);
  MOCHI_ASSERT(numElements > 0);

  // Non-axis-aligned gravity to catch component/axis bugs.
  Real3 const gravity{0.31_r, -9.81_r, 1.73_r};
  real const density = 1234.5_r;

  DynamicArray<real> extraWeights(numElements);
  for (int i = 0; i < numElements; ++i) {
    extraWeights[i] = 1_r + 0.1_r * static_cast<real>(i);
  }

  for (int trial = 0; trial < kNumTrials; ++trial) {
    // Mixed-lane: distinct displacement per lane; element/weight indices may repeat.
    NdArray<int, kBS> idx;
    for (int b = 0; b < kBS; ++b) {
      idx[b] = (trial * kBS + b) % numElements;
    }
    NdArray<NdArray<real, kDim>, kBS> disp;
    for (int b = 0; b < kBS; ++b) {
      disp[b] = MakeRandomArray<kDim>(static_cast<unsigned int>(trial * kBS + b));
    }
    auto const batchDisp = PackDisp<ElementT, kBS>(disp);

    auto const elemSpan = MakeConstSpan(data.elements);
    auto const weightSpan = MakeConstSpan(extraWeights);

    // Run at the random displacement (energy + residual).
    BatchDouble<kBS> energy{0.0};
    fem::BatchElementVector<kBS, ElementT> res{};
    bool const wrote = GravityWork<kBS, ElementT>(
        idx, elemSpan, batchDisp, &energy, &res, gravity, density, weightSpan);
    EXPECT_TRUE(wrote);

    // Run at zero displacement (residual must be identical; energy gives the rest-state baseline).
    fem::BatchElementVector<kBS, ElementT> const zeroDisp{};
    BatchDouble<kBS> energy0{0.0};
    fem::BatchElementVector<kBS, ElementT> res0{};
    GravityWork<kBS, ElementT>(
        idx, elemSpan, zeroDisp, &energy0, &res0, gravity, density, weightSpan);

    for (int b = 0; b < kBS; ++b) {
      int const e = idx[b];
      real volume = 0_r;
      for (int q = 0; q < ElementT::kNumQuadPoints; ++q) {
        volume += data.elements[e].quadWeights[q];
      }
      real const w = extraWeights[e];

      // Analytic uniform per-node residual.
      for (int f = 0; f < kNumNodes; ++f) {
        for (int d = 0; d < kSpaceDim; ++d) {
          real const expected = -density * w * (volume / static_cast<real>(kNumNodes)) * gravity[d];
          ExpectNearRel(expected, res[f * kSpaceDim + d][b]);
        }
      }

      ExpectNearEnergy(
          ExpectedGravityEnergy(data.elements[e], disp[b], gravity, density, w), energy[b]);

      // Residual is independent of displacement.
      for (int k = 0; k < kDim; ++k) {
        ExpectNearRel(res0[k][b], res[k][b]);
      }

      // Energy-gradient identity: E(u) - E(0) == <r, u>.
      double dotRU = 0.0;
      for (int k = 0; k < kDim; ++k) {
        dotRU += static_cast<double>(res[k][b]) * static_cast<double>(disp[b][k]);
      }
      ExpectNearEnergy(dotRU, energy[b] - energy0[b]);
    }
  }
}

// Verifies the kernel honors the energy-only / residual-only output flags (null pointers).
template <class ElementT, template <class> class MeshDataT, int kBS>
void VerifyGravityOutputFlags() {
  constexpr int kNumNodes = ElementT::kNumDofs;
  constexpr int kSpaceDim = ElementT::kSpaceDim;
  constexpr int kDim = kSpaceDim * kNumNodes;

  auto const data = MeshDataT<ElementT>::CreateMinimalCube();
  int const numElements = isize(data.elements);
  Real3 const gravity{0.31_r, -9.81_r, 1.73_r};
  real const density = 1234.5_r;

  NdArray<int, kBS> idx;
  for (int b = 0; b < kBS; ++b) {
    idx[b] = b % numElements;
  }
  NdArray<NdArray<real, kDim>, kBS> disp;
  for (int b = 0; b < kBS; ++b) {
    disp[b] = MakeRandomArray<kDim>(static_cast<unsigned int>(b));
  }
  auto const batchDisp = PackDisp<ElementT, kBS>(disp);
  auto const elemSpan = MakeConstSpan(data.elements);

  // Reference: both outputs.
  BatchDouble<kBS> energyBoth{0.0};
  fem::BatchElementVector<kBS, ElementT> resBoth{};
  GravityWork<kBS, ElementT>(idx, elemSpan, batchDisp, &energyBoth, &resBoth, gravity, density);

  // Energy only.
  BatchDouble<kBS> energyOnly{0.0};
  GravityWork<kBS, ElementT>(idx, elemSpan, batchDisp, &energyOnly, nullptr, gravity, density);
  // Residual only.
  fem::BatchElementVector<kBS, ElementT> resOnly{};
  GravityWork<kBS, ElementT>(idx, elemSpan, batchDisp, nullptr, &resOnly, gravity, density);

  for (int b = 0; b < kBS; ++b) {
    ExpectNearEnergy(energyBoth[b], energyOnly[b]);
    for (int k = 0; k < kDim; ++k) {
      ExpectNearRel(resBoth[k][b], resOnly[k][b]);
    }
  }
}

// Edge case: zero gravity produces zero residual and zero energy.
template <class ElementT, template <class> class MeshDataT, int kBS>
void VerifyGravityZero() {
  constexpr int kNumNodes = ElementT::kNumDofs;
  constexpr int kSpaceDim = ElementT::kSpaceDim;
  constexpr int kDim = kSpaceDim * kNumNodes;

  auto const data = MeshDataT<ElementT>::CreateMinimalCube();
  int const numElements = isize(data.elements);

  NdArray<int, kBS> idx;
  for (int b = 0; b < kBS; ++b) {
    idx[b] = b % numElements;
  }
  NdArray<NdArray<real, kDim>, kBS> disp;
  for (int b = 0; b < kBS; ++b) {
    disp[b] = MakeRandomArray<kDim>(static_cast<unsigned int>(b));
  }
  auto const batchDisp = PackDisp<ElementT, kBS>(disp);

  BatchDouble<kBS> energy{0.0};
  fem::BatchElementVector<kBS, ElementT> res{};
  GravityWork<kBS, ElementT>(
      idx, MakeConstSpan(data.elements), batchDisp, &energy, &res, Real3{0_r, 0_r, 0_r}, 1234.5_r);

  for (int b = 0; b < kBS; ++b) {
    EXPECT_NEAR(0.0, energy[b], 1e-12);
    for (int k = 0; k < kDim; ++k) {
      EXPECT_EQ(0_r, res[k][b]);
    }
  }
}

// Verifies the no-output sentinel branch used by callers to skip unsupported output modes.
template <class ElementT, template <class> class MeshDataT, int kBS>
void VerifyGravityNoOutputs() {
  constexpr int kNumNodes = ElementT::kNumDofs;
  constexpr int kSpaceDim = ElementT::kSpaceDim;
  constexpr int kDim = kSpaceDim * kNumNodes;

  auto const data = MeshDataT<ElementT>::CreateMinimalCube();
  int const numElements = isize(data.elements);
  Real3 const gravity{0.31_r, -9.81_r, 1.73_r};
  real const density = 1234.5_r;

  NdArray<int, kBS> idx;
  for (int b = 0; b < kBS; ++b) {
    idx[b] = b % numElements;
  }
  NdArray<NdArray<real, kDim>, kBS> disp;
  for (int b = 0; b < kBS; ++b) {
    disp[b] = MakeRandomArray<kDim>(static_cast<unsigned int>(b));
  }
  auto const batchDisp = PackDisp<ElementT, kBS>(disp);

  bool const wrote = GravityWork<kBS, ElementT>(
      idx, MakeConstSpan(data.elements), batchDisp, nullptr, nullptr, gravity, density);
  EXPECT_FALSE(wrote);
}

} // namespace

TEST(Gravity, Analytic) {
  RunAllBatchSizes([&]<int kBS>() {
    VerifyGravityAnalytic<TetGravityElement, TestTetMeshData, kBS>();
    VerifyGravityAnalytic<TriGravityElement, TestTriMeshData, kBS>();
  });
}

TEST(Gravity, OutputFlags) {
  RunAllBatchSizes([&]<int kBS>() {
    VerifyGravityOutputFlags<TetGravityElement, TestTetMeshData, kBS>();
    VerifyGravityOutputFlags<TriGravityElement, TestTriMeshData, kBS>();
  });
}

TEST(Gravity, ZeroGravity) {
  RunAllBatchSizes([&]<int kBS>() {
    VerifyGravityZero<TetGravityElement, TestTetMeshData, kBS>();
    VerifyGravityZero<TriGravityElement, TestTriMeshData, kBS>();
  });
}

TEST(Gravity, NoOutputs) {
  RunAllBatchSizes([&]<int kBS>() {
    VerifyGravityNoOutputs<TetGravityElement, TestTetMeshData, kBS>();
    VerifyGravityNoOutputs<TriGravityElement, TestTriMeshData, kBS>();
  });
}
