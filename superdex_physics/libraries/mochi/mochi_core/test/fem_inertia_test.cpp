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

#include <mochi_core/element_operations/fem_inertia.h>
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/elements/tetrahedral/simplex_quadrature.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/test/batched_work_test_utils.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>

#include <gtest/gtest.h>

using namespace mochi;
using namespace mochi::fem;

// The inertia term is (1/(h²α²))·M·(x − x̃) with M the consistent mass matrix. Using M from the
// (independent, retained) ComputeMassMatrixPerElement, the kernel must satisfy, with a = disp − x̃:
//   res    = dtfi2 · M · a
//   energy = ½ · dtfi2 · aᵀ M a
//   dRes   = M                        (AddMassMatrixToDRes scatters the raw mass matrix)
// Each lane gets distinct displacement and stage-start data, with element indices cycling through
// the mesh. Larger batch sizes also exercise duplicate element lanes while preserving per-lane
// independence (mixed-lane). Passing implies both inertia ops are correct.

using InertiaElement = tetrahedral::Pk3DElement<1, 4>;
namespace {

constexpr int kDim = InertiaElement::kSpaceDim * InertiaElement::kNumDofs;
using MassMatrix = NdArray<real, kDim, kDim>;

template <int kBS>
fem::BatchElementVector<kBS, InertiaElement> PackBatch(
    NdArray<NdArray<real, kDim>, kBS> const& perLane) {
  using V = BatchReal<kBS>;
  fem::BatchElementVector<kBS, InertiaElement> out{};
  alignas(alignof(V)) real staging[V::kSize]{};
  for (int k = 0; k < kDim; ++k) {
    for (int b = 0; b < kBS; ++b) {
      staging[b] = perLane[b][k];
    }
    out[k] = Load<V>(staging);
  }
  return out;
}

template <int kBS>
void VerifyBatchedInertia(OutputConfig cfg) {
  auto const data = TestTetMeshData<InertiaElement>::CreateMinimalCube();
  int const numElements = isize(data.elements);

  real const density = 1000_r;
  real const dtFactor = 0.01_r;
  real const dtfi2 = 1_r / (dtFactor * dtFactor);

  // Independent per-element consistent mass matrices (the oracle).
  DynamicArray<MassMatrix> massMatrices(numElements);
  ComputeMassMatrixPerElement(MakeConstSpan(data.elements), density, MakeSpan(massMatrices));

  DynamicArray<real> extraWeights(numElements);
  for (int i = 0; i < numElements; ++i) {
    extraWeights[i] = 1_r + 0.1_r * static_cast<real>(i);
  }

  for (int trial = 0; trial < kNumTrials; ++trial) {
    NdArray<int, kBS> idx;
    for (int b = 0; b < kBS; ++b) {
      idx[b] = (trial + b) % numElements;
    }
    NdArray<NdArray<real, kDim>, kBS> disp;
    NdArray<NdArray<real, kDim>, kBS> stageStart;
    for (int b = 0; b < kBS; ++b) {
      disp[b] = MakeRandomArray<kDim>(static_cast<unsigned int>(trial * kBS + b));
      stageStart[b] = MakeRandomArray<kDim>(static_cast<unsigned int>(7919 + trial * kBS + b));
    }

    auto const dispBatch = PackBatch<kBS>(disp);
    auto const stageBatch = PackBatch<kBS>(stageStart);
    auto const elemSpan = MakeConstSpan(data.elements);

    BatchDouble<kBS> energy{0.0};
    fem::BatchElementVector<kBS, InertiaElement> res{};
    if (cfg.energy || cfg.residual) {
      InertiaWork<kBS>(
          idx,
          elemSpan,
          dispBatch,
          stageBatch,
          cfg.energy ? &energy : nullptr,
          cfg.residual ? &res : nullptr,
          density,
          dtfi2,
          MakeConstSpan(extraWeights));
    }

    fem::BatchElementMatrix<kBS, InertiaElement> dres{};
    if (cfg.dresidual) {
      AddMassMatrixToDRes<kBS, InertiaElement, kDim>(idx, MakeConstSpan(massMatrices), dres);
    }

    for (int b = 0; b < kBS; ++b) {
      auto const& mm = massMatrices[idx[b]];
      real const extraWeight = extraWeights[idx[b]];
      std::array<real, kDim> a{};
      for (int k = 0; k < kDim; ++k) {
        a[k] = disp[b][k] - stageStart[b][k];
      }

      if (cfg.residual) {
        NdArray<real, kDim> refRes{};
        NdArray<real, kDim> actualRes{};
        for (int i = 0; i < kDim; ++i) {
          real s = 0_r;
          for (int j = 0; j < kDim; ++j) {
            s += mm[i][j] * a[j];
          }
          refRes[i] = dtfi2 * extraWeight * s;
          actualRes[i] = res[i][b];
        }
        ExpectNearL2(refRes, actualRes);
      }

      if (cfg.energy) {
        double refEnergy = 0.0;
        for (int i = 0; i < kDim; ++i) {
          for (int j = 0; j < kDim; ++j) {
            refEnergy += static_cast<double>(a[i]) * static_cast<double>(mm[i][j]) *
                static_cast<double>(a[j]);
          }
        }
        refEnergy *= 0.5 * static_cast<double>(dtfi2 * extraWeight);
        ExpectNearEnergy(refEnergy, energy[b]);
      }

      if (cfg.dresidual) {
        NdArray<real, kDim * kDim> refDRes{};
        NdArray<real, kDim * kDim> actualDRes{};
        for (int i = 0; i < kDim; ++i) {
          for (int j = 0; j < kDim; ++j) {
            refDRes[i * kDim + j] = mm[i][j];
            actualDRes[i * kDim + j] = dres[i * kDim + j][b];
          }
        }
        ExpectNearL2(refDRes, actualDRes);
      }
    }
  }
}

template <int kBS>
void VerifyAddMassMatrixToDResSupportsPaddedDRes() {
  constexpr int kMassDof = 6;
  real constexpr kInitialValue = -1000_r;
  using SmallMassMatrix = NdArray<real, kMassDof, kMassDof>;

  DynamicArray<SmallMassMatrix> massMatrices(kBS);
  for (int e = 0; e < isize(massMatrices); ++e) {
    for (int i = 0; i < kMassDof; ++i) {
      for (int j = 0; j < kMassDof; ++j) {
        massMatrices[e][i][j] = static_cast<real>(100 * e + 10 * i + j + 1);
      }
    }
  }

  NdArray<int, kBS> idx;
  for (int b = 0; b < kBS; ++b) {
    idx[b] = (b + 1) % kBS;
  }
  fem::BatchElementMatrix<kBS, InertiaElement> dres;
  for (auto& entry : dres) {
    entry = kInitialValue;
  }
  bool const added =
      AddMassMatrixToDRes<kBS, InertiaElement, kMassDof>(idx, MakeConstSpan(massMatrices), dres);
  EXPECT_TRUE(added);

  for (int b = 0; b < kBS; ++b) {
    for (int i = 0; i < kDim; ++i) {
      for (int j = 0; j < kDim; ++j) {
        real const expected = (i < kMassDof && j < kMassDof)
            ? kInitialValue + massMatrices[idx[b]][i][j]
            : kInitialValue;
        EXPECT_EQ(expected, dres[i * kDim + j][b]);
      }
    }
  }
}

} // namespace

TEST(Inertia, MatchesReference) {
  for (auto cfg : kAllOutputConfigs) {
    RunAllBatchSizes([&]<int kBS>() { VerifyBatchedInertia<kBS>(cfg); });
  }
}

TEST(Inertia, AddMassMatrixToDResSupportsPaddedDRes) {
  RunAllBatchSizes([&]<int kBS>() { VerifyAddMassMatrixToDResSupportsPaddedDRes<kBS>(); });
}
