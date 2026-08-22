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

#include <mochi_core/element_operations/fem_traction.h>
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/elements/tetrahedral/finite_element_trace.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/test/batched_work_test_utils.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/dynamic_array.h>

#include <gtest/gtest.h>

using namespace mochi;
using namespace mochi::fem;

// The batched traction kernel is position-independent: the callback supplies force / dforce /
// energy per (element, quad point), and the kernel integrates them against the face basis. This
// test validates that integration against a closed-form reference computed directly from the
// trace's basis, quadrature weights, and normals:
//   res[f,d]            = -sum_q normal_q[d] * N_f(q) * w_q * w_e
//   dRes[(f,r),(g,c)]   = -dforce[r][c] * sum_q N_f(q) N_g(q) * w_q * w_e
//   energy              =  cbEnergy * sum_q w_q * w_e
// The callback supplies a general, non-symmetric, per-element-scaled dforce[r][c] so the full
// off-diagonal dResidual assembly (and any r/c transpose) is exercised, not just the diagonal.
// Each lane gets a distinct trace, and the callback masks odd-indexed elements, so per-lane masking
// (active vs inactive) is exercised (mixed-lane). Passing implies correctness, including the mask
// and the kNumFields==4 (extra non-spatial DoF) layout.

using VolumeElement = tetrahedral::Pk3DElement<1, 1>;
template <int kNumQuadPoints>
using TraceElement = tetrahedral::Pk3DElementTrace<VolumeElement, kNumQuadPoints>;
static constexpr int kNumFaces = 4;
// General non-symmetric per-component force gradient dforce[r][c] (diagonally dominant for a
// well-conditioned dResidual), scaled per element in the callback / reference below.
static constexpr real kBaseDForce[3][3] = {
    {3.0_r, 0.2_r, -0.3_r},
    {0.4_r, 2.5_r, 0.5_r},
    {-0.6_r, 0.7_r, 4.0_r}};
static constexpr real kRefTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-9_r : 1e-5_r;

template <int kNumQuadPoints>
static auto const& TraceQuadratureForFace(int face) {
  if constexpr (kNumQuadPoints == 1) {
    return tetrahedral::kTetrahedralTraceQuadrature1[face];
  } else {
    static_assert(kNumQuadPoints == 3);
    return tetrahedral::kTetrahedralTraceQuadrature3[face];
  }
}

namespace {
template <class TraceT>
struct TractionTestData {
  TetrahedralMesh mesh;
  VolumeElement volumeElement;
  DynamicArray<TraceT> traces;

  TractionTestData()
      : mesh(test::CreateMinimalTetMeshSingleTet()),
        volumeElement(
            0,
            mesh.GetNodeCoordinates(),
            mesh.GetElementConnectivity(),
            tetrahedral::kTetrahedralQuadrature1) {
    traces.reserve(kNumFaces);
    for (int f = 0; f < kNumFaces; ++f) {
      traces.emplace_back(volumeElement, f, TraceQuadratureForFace<TraceT::kNumQuadPoints>(f));
    }
  }
};
} // namespace

// Batched callback: face-normal force, non-symmetric per-element-scaled dforce, and a simple
// per-element energy. Odd-indexed elements produce no force (mask coverage).
template <int kBS, class TraceT>
static auto MakeBatchedTraction(DynamicArray<TraceT> const& traces) {
  using V = BatchReal<kBS>;
  using Vd = BatchDouble<kBS>;
  using V3 = BatchReal3<kBS>;
  return [&traces](
             NdArray<int, kBS> const& elementIndices,
             int quadPointIndex,
             Vd* outEnergy,
             V3* outForce,
             NdArray<V3, 3>* outDForce,
             NdArray<bool, kBS>& outHasForce) {
    for (int b = 0; b < kBS; ++b) {
      outHasForce[b] = (elementIndices[b] % 2 == 0);
    }
    if (outEnergy) {
      alignas(alignof(Vd)) double energyStaging[Vd::kSize]{};
      for (int b = 0; b < kBS; ++b) {
        energyStaging[b] = static_cast<double>(elementIndices[b]) + 0.5;
      }
      *outEnergy = Load<Vd>(energyStaging);
    }
    alignas(alignof(V)) real staging[V::kSize]{};
    if (outForce) {
      for (int d = 0; d < 3; ++d) {
        for (int b = 0; b < kBS; ++b) {
          staging[b] = traces[elementIndices[b]].normals[quadPointIndex][d];
        }
        (*outForce)[d] = Load<V>(staging);
      }
    }
    if (outDForce) {
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          for (int b = 0; b < kBS; ++b) {
            staging[b] = kBaseDForce[r][c] * (1_r + 0.1_r * static_cast<real>(elementIndices[b]));
          }
          (*outDForce)[r][c] = Load<V>(staging);
        }
      }
    }
  };
}

// Closed-form reference matching MakeBatchedTraction, computed from the trace geometry directly.
template <class TraceT, int kNumFields>
static void ComputeReference(
    TraceT const& trace,
    int elementIndex,
    real extraWeight,
    NdArray<real, TraceT::kNumDofs * kNumFields>& refRes,
    NdArray<real, (TraceT::kNumDofs * kNumFields) * (TraceT::kNumDofs * kNumFields)>& refDRes,
    double& refEnergy) {
  constexpr int kNumNodes = TraceT::kNumDofs;
  constexpr int kNumDof = kNumNodes * kNumFields;
  refRes = {};
  refDRes = {};
  refEnergy = 0.0;

  bool const active = (elementIndex % 2 == 0);
  if (!active) {
    return;
  }

  real area = 0_r;
  for (int q = 0; q < TraceT::kNumQuadPoints; ++q) {
    area += trace.quadWeights[q] * extraWeight;
  }
  refEnergy = (static_cast<double>(elementIndex) + 0.5) * static_cast<double>(area);

  for (int f = 0; f < kNumNodes; ++f) {
    for (int d = 0; d < 3; ++d) {
      real s = 0_r;
      for (int q = 0; q < TraceT::kNumQuadPoints; ++q) {
        s += trace.normals[q][d] * trace.basisEvaluated[q][f] * trace.quadWeights[q] * extraWeight;
      }
      refRes[f * kNumFields + d] = -s;
    }
  }

  real const dforceScale = 1_r + 0.1_r * static_cast<real>(elementIndex);
  for (int f = 0; f < kNumNodes; ++f) {
    for (int g = 0; g < kNumNodes; ++g) {
      real mass = 0_r;
      for (int q = 0; q < TraceT::kNumQuadPoints; ++q) {
        mass += trace.basisEvaluated[q][f] * trace.basisEvaluated[q][g] * trace.quadWeights[q] *
            extraWeight;
      }
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          refDRes[(f * kNumFields + r) * kNumDof + (g * kNumFields + c)] =
              -kBaseDForce[r][c] * dforceScale * mass;
        }
      }
    }
  }
}

template <int kBS, class TraceT, int kNumFields>
static void VerifyBatchedTraction(OutputConfig cfg) {
  constexpr int kNumNodes = TraceT::kNumDofs;
  constexpr int kDim = kNumNodes * kNumFields;

  TractionTestData<TraceT> data;
  auto const batchedTraction = MakeBatchedTraction<kBS>(data.traces);

  DynamicArray<real> extraWeights(kNumFaces);
  for (int i = 0; i < kNumFaces; ++i) {
    extraWeights[i] = 1_r + 0.1_r * static_cast<real>(i);
  }

  for (int trial = 0; trial < kNumTrials; ++trial) {
    NdArray<int, kBS> elementIndices{};
    for (int b = 0; b < kBS; ++b) {
      elementIndices[b] = (trial + b) % kNumFaces;
    }

    BatchDouble<kBS> energy{0.0};
    fem::BatchElementVector<kBS, TraceT, kNumFields> res{};
    fem::BatchElementMatrix<kBS, TraceT, kNumFields> dres{};
    bool const wrote = TractionWork<kBS, TraceT, kNumFields>(
        elementIndices,
        MakeConstSpan(data.traces),
        cfg.energy ? &energy : nullptr,
        cfg.residual ? &res : nullptr,
        cfg.dresidual ? &dres : nullptr,
        batchedTraction,
        MakeConstSpan(extraWeights));

    bool expectedWrote = false;
    for (int b = 0; b < kBS; ++b) {
      expectedWrote |= (elementIndices[b] % 2 == 0);
    }
    EXPECT_EQ(expectedWrote, wrote);

    for (int b = 0; b < kBS; ++b) {
      int const e = elementIndices[b];
      NdArray<real, kDim> refRes;
      NdArray<real, kDim * kDim> refDRes;
      double refEnergy = 0.0;
      ComputeReference<TraceT, kNumFields>(
          data.traces[e], e, extraWeights[e], refRes, refDRes, refEnergy);

      if (cfg.energy) {
        ExpectNearEnergy(refEnergy, energy[b], kRefTol);
      }
      if (cfg.residual) {
        NdArray<real, kDim> actualRes;
        for (int k = 0; k < kDim; ++k) {
          actualRes[k] = res[k][b];
        }
        ExpectNearL2(refRes, actualRes, kRefTol);
      }
      if (cfg.dresidual) {
        NdArray<real, kDim * kDim> actualDRes;
        for (int k = 0; k < kDim * kDim; ++k) {
          actualDRes[k] = dres[k][b];
        }
        ExpectNearL2(refDRes, actualDRes, kRefTol);
      }
    }
  }
}

TEST(Traction, MatchesReference) {
  for (auto cfg : kAllOutputConfigs) {
    RunAllBatchSizes([&]<int kBS>() { VerifyBatchedTraction<kBS, TraceElement<1>, 3>(cfg); });
    RunAllBatchSizes([&]<int kBS>() { VerifyBatchedTraction<kBS, TraceElement<1>, 4>(cfg); });
    RunAllBatchSizes([&]<int kBS>() { VerifyBatchedTraction<kBS, TraceElement<3>, 3>(cfg); });
    RunAllBatchSizes([&]<int kBS>() { VerifyBatchedTraction<kBS, TraceElement<3>, 4>(cfg); });
  }
}
