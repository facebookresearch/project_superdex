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

#include <mochi_core/element_operations/element_assembler.h>
#include <mochi_core/element_operations/fem_stress.h>
#include <mochi_core/element_operations/fem_traction.h>
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/elements/tetrahedral/finite_element_trace.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/linear_algebra/utils/assembly.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/materials/batched_smith_neo_hookean.h>
#include <mochi_core/materials/material_params.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph_utils.h>
#include <mochi_core/utils/local_to_global_map.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/sparsity_utils.h>
#include <mochi_core/utils/task_scheduler.h>

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

using namespace mochi;

// ================================================================================================
// FEM assembler tests.
//
// The FEM assembler is exercised with a *synthetic* element operation whose per-element
// contribution has a closed form (defined once in ComputeSyntheticContribution). The same closed
// form drives an independent manual element-by-element scatter that is used as the oracle, so these
// tests validate the assembler's gather / scatter / reduction / active-subset / padded-L2G and
// PSD-flag plumbing logic without depending on any FEM element kernel.
// ================================================================================================

namespace {
// 4-node, 3-field shape type for the assembler template. The synthetic op ignores element geometry,
// so only kNumDofs / kSpaceDim are needed.
struct ShapeElement {
  static constexpr int kNumDofs = 4;
  static constexpr int kSpaceDim = 3;
};
} // namespace

static constexpr int kShapeFields = ShapeElement::kSpaceDim;
static constexpr int kShapeEleDofs = ShapeElement::kNumDofs * kShapeFields;
static constexpr real kAssemblerRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-9_r : 1e-3_r;

namespace {
// Single source of truth for the synthetic element contribution:
//  - the residual depends on the gathered displacement -> exercises the gather,
//  - every term depends on the element index           -> exercises per-lane element indexing,
//  - the element matrix is fully populated and non-symmetric -> exercises the full dresidual
//  scatter.
struct SyntheticContribution {
  double energy = 0.0;
  std::array<real, kShapeEleDofs> res{};
  std::array<real, kShapeEleDofs * kShapeEleDofs> dres{};
};
} // namespace

static SyntheticContribution ComputeSyntheticContribution(
    int elementIndex,
    std::array<real, kShapeEleDofs> const& localDisp,
    bool projectPsd) {
  SyntheticContribution c;
  real const ePlus1 = static_cast<real>(elementIndex + 1);
  double energy = ePlus1;
  for (int d = 0; d < kShapeEleDofs; ++d) {
    energy += static_cast<double>(localDisp[d]);
    c.res[d] = localDisp[d] + 0.5_r * ePlus1 + 0.25_r * static_cast<real>(d);
  }
  c.energy = energy;
  for (int i = 0; i < kShapeEleDofs; ++i) {
    for (int j = 0; j < kShapeEleDofs; ++j) {
      real const dresValue = ePlus1 + 0.1_r * static_cast<real>(i) - 0.05_r * static_cast<real>(j);
      c.dres[i * kShapeEleDofs + j] =
          projectPsd ? dresValue + 0.2_r * static_cast<real>(1 + i + j) : dresValue;
    }
  }
  return c;
}

// Honors the FEM assembler's padded-position contract: an element op must produce zero residual and
// dresidual for stencil positions whose DoFs alias node 0 (the padding anchor). Detection matches
// the assembler. No-op for non-padded elements, whose nodes all have distinct global DoFs.
static void MaskPaddedSlots(SyntheticContribution& c, Span<int const> eleIndices) {
  for (int n = 1; n < ShapeElement::kNumDofs; ++n) {
    bool isPadded = true;
    for (int f = 0; f < kShapeFields; ++f) {
      if (eleIndices[n * kShapeFields + f] != eleIndices[f]) {
        isPadded = false;
        break;
      }
    }
    if (!isPadded) {
      continue;
    }
    for (int f = 0; f < kShapeFields; ++f) {
      int const d = n * kShapeFields + f;
      c.res[d] = 0_r;
      for (int j = 0; j < kShapeEleDofs; ++j) {
        c.dres[d * kShapeEleDofs + j] = 0_r;
        c.dres[j * kShapeEleDofs + d] = 0_r;
      }
    }
  }
}

// Batched op evaluating ComputeSyntheticContribution per lane and accumulating into the batch
// outputs.
template <int kBatchSize>
static ElOpFnType<ShapeElement, kShapeFields, kBatchSize> MakeSyntheticOp() {
  return [](NdArray<int, kBatchSize> const& elementIndices,
            Span<int const> indicesFlat,
            fem::BatchElementVector<kBatchSize, ShapeElement> const& disp,
            BatchDouble<kBatchSize>* outEnergy,
            fem::BatchElementVector<kBatchSize, ShapeElement>* outRes,
            fem::BatchElementMatrix<kBatchSize, ShapeElement>* outDRes,
            bool projectPsd) -> bool {
    for (int b = 0; b < kBatchSize; ++b) {
      std::array<real, kShapeEleDofs> localDisp{};
      for (int d = 0; d < kShapeEleDofs; ++d) {
        localDisp[d] = disp[d][b];
      }
      SyntheticContribution c =
          ComputeSyntheticContribution(elementIndices[b], localDisp, projectPsd);
      MaskPaddedSlots(c, indicesFlat.subspan(elementIndices[b] * kShapeEleDofs, kShapeEleDofs));
      if (outEnergy) {
        *outEnergy = Set(*outEnergy, b, (*outEnergy)[b] + c.energy);
      }
      if (outRes) {
        for (int d = 0; d < kShapeEleDofs; ++d) {
          (*outRes)[d] = Set((*outRes)[d], b, (*outRes)[d][b] + c.res[d]);
        }
      }
      if (outDRes) {
        for (int k = 0; k < kShapeEleDofs * kShapeEleDofs; ++k) {
          (*outDRes)[k] = Set((*outDRes)[k], b, (*outDRes)[k][b] + c.dres[k]);
        }
      }
    }
    return true;
  };
}

// Solution-independent batched op: writes a per-element contribution and has no displacement
// argument. Valid input for the no-solution assembler overloads, which do not gather displacement.
template <int kBatchSize>
static NoDispElOpFnType<ShapeElement, kShapeFields, kBatchSize> MakeNoDispSolutionIndependentOp() {
  return [](NdArray<int, kBatchSize> const& elementIndices,
            Span<int const> /*indicesFlat*/,
            BatchDouble<kBatchSize>* outEnergy,
            fem::BatchElementVector<kBatchSize, ShapeElement>* outRes,
            fem::BatchElementMatrix<kBatchSize, ShapeElement>* outDRes,
            bool /*projectPsd*/) -> bool {
    for (int b = 0; b < kBatchSize; ++b) {
      real const ePlus1 = static_cast<real>(elementIndices[b] + 1);
      if (outEnergy) {
        *outEnergy = Set(*outEnergy, b, (*outEnergy)[b] + static_cast<double>(ePlus1));
      }
      if (outRes) {
        for (int d = 0; d < kShapeEleDofs; ++d) {
          (*outRes)[d] =
              Set((*outRes)[d], b, (*outRes)[d][b] + ePlus1 + 0.25_r * static_cast<real>(d));
        }
      }
      if (outDRes) {
        for (int i = 0; i < kShapeEleDofs; ++i) {
          for (int j = 0; j < kShapeEleDofs; ++j) {
            int const k = i * kShapeEleDofs + j;
            real const v = ePlus1 + 0.1_r * static_cast<real>(i) - 0.05_r * static_cast<real>(j);
            (*outDRes)[k] = Set((*outDRes)[k], b, (*outDRes)[k][b] + v);
          }
        }
      }
    }
    return true;
  };
}

template <int kBatchSize>
static ElOpFnType<ShapeElement, kShapeFields, kBatchSize> MakeSolutionIndependentOpWithDisp() {
  auto noDispOp = MakeNoDispSolutionIndependentOp<kBatchSize>();
  return [noDispOp = std::move(noDispOp)](
             NdArray<int, kBatchSize> const& elementIndices,
             Span<int const> indicesFlat,
             fem::BatchElementVector<kBatchSize, ShapeElement> const& /*disp*/,
             BatchDouble<kBatchSize>* outEnergy,
             fem::BatchElementVector<kBatchSize, ShapeElement>* outRes,
             fem::BatchElementMatrix<kBatchSize, ShapeElement>* outDRes,
             bool projectPsd) -> bool {
    return noDispOp(elementIndices, indicesFlat, outEnergy, outRes, outDRes, projectPsd);
  };
}

namespace {
// Independent manual scatter (dense) using the same contribution function.
struct ManualGlobalAssembly {
  double obj = 0.0;
  ColumnVector<real> res;
  RowMatrix<real> dres;
};
} // namespace

static ManualGlobalAssembly ManualAssemble(
    Local2GlobalMap const& l2g,
    int numGlobalDofs,
    ColumnVectorView<real const> sol,
    Span<int const> activeElements, // empty => all elements
    AssemblyParams params) {
  ManualGlobalAssembly g;
  g.res = ColumnVector<real>::Zero(numGlobalDofs);
  g.dres = RowMatrix<real>::Zero(numGlobalDofs, numGlobalDofs);

  auto processElement = [&](int e) {
    Span<int const> const gi = l2g.HasPaddedIndices()
        ? l2g.GetPaddedGlobalIndices().subspan(e * l2g.GetPaddedStride(), l2g.GetPaddedStride())
        : l2g.GetGlobalIndices(e);
    std::array<real, kShapeEleDofs> localDisp{};
    for (int d = 0; d < isize(gi); ++d) {
      localDisp[d] = sol[gi[d]];
    }
    SyntheticContribution c = ComputeSyntheticContribution(e, localDisp, params.psdDRes);
    MaskPaddedSlots(c, gi);
    if (params.assemObj) {
      g.obj += c.energy;
    }
    if (params.assemRes) {
      for (int d = 0; d < isize(gi); ++d) {
        g.res[gi[d]] += c.res[d];
      }
    }
    if (params.assemDRes) {
      for (int i = 0; i < isize(gi); ++i) {
        for (int j = 0; j < isize(gi); ++j) {
          g.dres(gi[i], gi[j]) += c.dres[i * kShapeEleDofs + j];
        }
      }
    }
  };

  if (activeElements.empty()) {
    for (int e = 0; e < l2g.GetNumElements(); ++e) {
      processElement(e);
    }
  } else {
    for (int e : activeElements) {
      processElement(e);
    }
  }
  return g;
}

// Relative-L2 comparison of two vectors (single EXPECT to keep the large sweep fast).
static void ExpectColumnsClose(
    ColumnVectorView<real const> expected,
    ColumnVectorView<real const> actual) {
  real num = 0_r;
  real den = 0_r;
  for (int i = 0; i < expected.Rows(); ++i) {
    num += Sqr(expected[i] - actual[i]);
    den += Sqr(expected[i]);
  }
  EXPECT_LE(Sqrt(num), kAssemblerRelTol * Sqrt(den) + real(1e-12));
}

// Run the FEM assembler with the synthetic op and compare against the manual oracle.
template <int kBatchSize>
static void VerifyOneBatchSize(
    Local2GlobalMap const& l2g,
    NodalBasedStructure const& nbs,
    ColumnVectorView<real const> sol,
    int numGlobalDofs,
    AssemblyActiveSubset const& subset,
    AssemblyParams params,
    ManualGlobalAssembly const& g) {
  double obj = 0.0;
  auto res = ColumnVector<real>::Zero(numGlobalDofs);
  auto dres = ToBlockSparseMatrix<kShapeFields>(MakeSparseMatrix(l2g));
  dres.SetZero();

  AssembleObjResDRes<ShapeElement, kShapeFields, kBatchSize>(
      l2g,
      nbs,
      MakeSyntheticOp<kBatchSize>(),
      sol,
      AssemblyResults<real>{
          .outObj = params.assemObj ? &obj : nullptr,
          .outRes = params.assemRes ? ColumnVectorView<real>{res} : ColumnVectorView<real>{},
          .outDRes = params.assemDRes ? AnyMatrixView<real>{AsView(dres)} : AnyMatrixView<real>{},
          .params = params},
      subset);

  if (params.assemObj) {
    EXPECT_NEAR_RTOL(g.obj, obj, double(kAssemblerRelTol));
  }
  if (params.assemRes) {
    ExpectColumnsClose(AsConstView(g.res), AsConstView(res));
  }
  if (params.assemDRes) {
    auto const assembled = ToMatrix(dres);
    real num = 0_r;
    real den = 0_r;
    for (int i = 0; i < numGlobalDofs; ++i) {
      for (int j = 0; j < numGlobalDofs; ++j) {
        num += Sqr(g.dres(i, j) - assembled(i, j));
        den += Sqr(g.dres(i, j));
      }
    }
    EXPECT_LE(Sqrt(num), kAssemblerRelTol * Sqrt(den) + real(1e-12));
  }
}

// All non-empty (obj, res, dres) request combinations.
static std::vector<AssemblyParams> AllAssemblyModes() {
  std::vector<AssemblyParams> modes;
  for (bool obj : {true, false}) {
    for (bool resd : {true, false}) {
      for (bool dres : {true, false}) {
        if (!obj && !resd && !dres) {
          continue;
        }
        for (bool psd : {false, true}) {
          if (psd && !dres) {
            continue;
          }
          modes.push_back(
              AssemblyParams{.assemObj = obj, .assemRes = resd, .assemDRes = dres, .psdDRes = psd});
        }
      }
    }
  }
  return modes;
}

namespace {
struct SubsetScenario {
  DynamicArray<int> indices;
  DynamicArray<bool> isActive;

  AssemblyActiveSubset ActiveSubset() const {
    return {MakeConstSpan(indices), MakeConstSpan(isActive)};
  }
};
} // namespace

// Active-subset scenarios, including mixed-lane edge cases (single element, partial final batch,
// shuffled order, even/odd partitions). The empty/all-elements case is handled separately.
static std::vector<SubsetScenario> MakeSubsetScenarios(int numElements) {
  std::vector<SubsetScenario> out;
  if (numElements <= 0) {
    return out;
  }

  auto addScenario = [&](DynamicArray<int> indices) {
    if (indices.empty()) {
      return;
    }
    DynamicArray<bool> isActive(numElements, false);
    for (int e : indices) {
      isActive[e] = true;
    }
    out.push_back({std::move(indices), std::move(isActive)});
  };

  // Single element: one real lane, the rest padding.
  DynamicArray<int> singleElement;
  singleElement.push_back(0);
  addScenario(std::move(singleElement));

  // All but the last element: size is typically not a multiple of the batch size.
  if (numElements >= 2) {
    DynamicArray<int> allButLast;
    for (int e = 0; e < numElements - 1; ++e) {
      allButLast.push_back(e);
    }
    addScenario(std::move(allButLast));
  }

  // Even / odd partitions (together these also exercise active-subset additivity).
  DynamicArray<int> even;
  DynamicArray<int> odd;
  for (int e = 0; e < numElements; ++e) {
    (e % 2 == 0 ? even : odd).push_back(e);
  }
  addScenario(std::move(even));
  addScenario(std::move(odd));

  // Shuffled full set: lanes hold non-adjacent elements; exercises order-independence.
  DynamicArray<int> shuffled;
  for (int e = 0; e < numElements; ++e) {
    shuffled.push_back(e);
  }
  auto rng = RandomGenerator(7);
  std::shuffle(shuffled.begin(), shuffled.end(), rng);
  addScenario(std::move(shuffled));

  return out;
}

static void RunOneConfig(
    Local2GlobalMap const& l2g,
    NodalBasedStructure const& nbs,
    ColumnVectorView<real const> sol,
    int numGlobalDofs,
    AssemblyActiveSubset const& subset,
    Span<int const> activeElements,
    AssemblyParams params) {
  // Compute the oracle once and compare every batch width against it.
  ManualGlobalAssembly const g = ManualAssemble(l2g, numGlobalDofs, sol, activeElements, params);
  VerifyOneBatchSize<1>(l2g, nbs, sol, numGlobalDofs, subset, params, g);
  VerifyOneBatchSize<4>(l2g, nbs, sol, numGlobalDofs, subset, params, g);
  VerifyOneBatchSize<8>(l2g, nbs, sol, numGlobalDofs, subset, params, g);
}

// Sweep every assembly mode and thread count over the all-elements case and the mixed-lane
// active-subset scenarios, comparing the assembler against the manual oracle for all batch widths.
static void RunAssemblerSweep(Local2GlobalMap const& l2g, NodalBasedStructure const& nbs) {
  int const numGlobalDofs = l2g.GetGlobalRange().Max() + 1;
  ColumnVector<real> sol(numGlobalDofs);
  sol.SetRandom(123, -0.1_r, 0.1_r);

  auto const scenarios = MakeSubsetScenarios(l2g.GetNumElements());

  for (int numThreads : {0, 1, Min(8, TaskScheduler::GetNumSupportedLogicalProcessors())}) {
    TaskScheduler scheduler(numThreads);
    for (auto const& params : AllAssemblyModes()) {
      // All elements (empty active subset).
      RunOneConfig(l2g, nbs, AsConstView(sol), numGlobalDofs, AssemblyActiveSubset{}, {}, params);

      // Active-subset scenarios (including mixed-lane edge cases).
      for (auto const& scenario : scenarios) {
        RunOneConfig(
            l2g,
            nbs,
            AsConstView(sol),
            numGlobalDofs,
            scenario.ActiveSubset(),
            MakeConstSpan(scenario.indices),
            params);
      }
    }
  }
}

static void RunFemAssemblerTests(TetrahedralMesh& mesh) {
  for (bool usePaddedL2g : {false, true}) {
    Local2GlobalMap l2g;
    l2g.InitializeFromMesh(&mesh, kShapeFields);
    if (usePaddedL2g) {
      l2g.InitializePaddedIndices(kShapeEleDofs);
    }
    NodalBasedStructure nbs(mesh.GetElementConnectivity());
    RunAssemblerSweep(l2g, nbs);
  }
}

// Non-trivial padded stencil: mixed-width elements (2, 3, and 4 real nodes -> 2, 1, and 0 padded
// slots) chained so adjacent elements share nodes. The assembler gathers/scatters real padded slots
// (aliased onto each element's first node) and must honor the padded-position contract; both the
// synthetic op and the oracle zero the padded slots via MaskPaddedSlots. Enough elements are used
// to drive the NBS assembly path or the linear path, depending on the number of threads.
static void RunPaddedStencilAssemblerTests() {
  constexpr int kStencilNodes = ShapeElement::kNumDofs;
  constexpr int kNumElements = 150;

  DynamicArray<DynamicArray<int>> connectivity;
  DynamicArray<DynamicArray<int>> stencil;
  for (int e = 0; e < kNumElements; ++e) {
    int const width = 2 + (e % 3); // 2, 3, or 4 real nodes.
    DynamicArray<int> conn;
    DynamicArray<int> sten;
    for (int k = 0; k < width; ++k) {
      conn.push_back(e + k); // Adjacent elements share nodes.
      sten.push_back(k); // Real nodes in the leading slots; trailing slots are padding.
    }
    connectivity.push_back(std::move(conn));
    stencil.push_back(std::move(sten));
  }
  auto const connectivityGraph = GraphFromRangeOfRanges<int, int>(connectivity);
  auto const stencilGraph = GraphFromRangeOfRanges<int, int>(stencil);

  Local2GlobalMap l2g;
  l2g.InitializeFromElementNodeConnectivity(connectivityGraph, kShapeFields);
  l2g.InitializeStencilIndices(stencilGraph);
  l2g.InitializePaddedIndices(kStencilNodes * kShapeFields);
  NodalBasedStructure const nbs =
      BuildPaddedNodalBasedStructure<kStencilNodes>(connectivityGraph, stencilGraph);

  RunAssemblerSweep(l2g, nbs);
}

TEST(ElementAssembler, SyntheticSingleTet) {
  TetrahedralMesh mesh = test::CreateMinimalTetMeshSingleTet({0.5_r, 1.0_r, 2.5_r});
  RunFemAssemblerTests(mesh);
}

TEST(ElementAssembler, SyntheticCube) {
  TetrahedralMesh mesh = test::CreateMinimalTetMeshUnitCube({0.5_r, 1.0_r, 2.5_r});
  RunFemAssemblerTests(mesh);
}

TEST(ElementAssembler, SyntheticTwoTetsShareNode) {
  TetrahedralMesh mesh = test::CreateMinimalTetMeshTwoShareNode({0.5_r, 1.0_r, 2.5_r});
  RunFemAssemblerTests(mesh);
}

TEST(ElementAssembler, SyntheticTwoTetsShareEdge) {
  TetrahedralMesh mesh = test::CreateMinimalTetMeshTwoShareEdge({0.5_r, 1.0_r, 2.5_r});
  RunFemAssemblerTests(mesh);
}

TEST(ElementAssembler, SyntheticTwoTetsShareFace) {
  TetrahedralMesh mesh = test::CreateMinimalTetMeshTwoShareFace({0.5_r, 1.0_r, 2.5_r});
  RunFemAssemblerTests(mesh);
}

TEST(ElementAssembler, SyntheticGrid) {
  TetrahedralMesh mesh = test::CreateMinimalTetMeshUnitGrid({2.0_r, 2.0_r, 2.0_r}, {2, 2, 2});
  RunFemAssemblerTests(mesh);
}

TEST(ElementAssembler, SyntheticPaddedStencil) {
  RunPaddedStencilAssemblerTests();
}

TEST(ElementAssembler, SyntheticDeterminism) {
  TetrahedralMesh mesh = test::CreateMinimalTetMeshUnitGrid({2.0_r, 2.0_r, 2.0_r}, {2, 2, 2});
  Local2GlobalMap l2g;
  l2g.InitializeFromMesh(&mesh, kShapeFields);
  NodalBasedStructure nbs(mesh.GetElementConnectivity());
  int const numGlobalDofs = l2g.GetGlobalRange().Max() + 1;
  ColumnVector<real> sol(numGlobalDofs);
  sol.SetRandom(5, -0.1_r, 0.1_r);

  AssemblyParams params{.assemObj = false, .assemRes = false, .assemDRes = true, .psdDRes = false};

  // Multi-threaded to stress the parallel reduction's determinism.
  TaskScheduler scheduler(Min(8, TaskScheduler::GetNumSupportedLogicalProcessors()));

  auto assembleDRes = [&]() {
    auto dres = ToBlockSparseMatrix<kShapeFields>(MakeSparseMatrix(l2g));
    dres.SetZero();
    AssembleObjResDRes<ShapeElement, kShapeFields, 8>(
        l2g,
        nbs,
        MakeSyntheticOp<8>(),
        AsConstView(sol),
        AssemblyResults<real>{
            .outObj = nullptr, .outRes = {}, .outDRes = AsView(dres), .params = params},
        AssemblyActiveSubset{});
    return dres;
  };

  auto dres1 = assembleDRes();
  auto dres2 = assembleDRes();

  EXPECT_GT(dres1.Norm(), 0_r);
  auto values1 = dres1.Values();
  auto values2 = MakeConstSpan(dres2.Values());
  ArrayMinusEquals(values1, values2);
  EXPECT_EQ(0_r, AsConstView(values1).Norm());
}

TEST(ElementAssembler, NoSolutionOverloadMatchesZeroSolution) {
  // The no-solution overload (no global solution, no gather) must produce results identical to the
  // gather overload fed an all-zero solution, for a solution-independent op.
  TetrahedralMesh mesh = test::CreateMinimalTetMeshUnitGrid({2.0_r, 2.0_r, 2.0_r}, {2, 2, 2});
  Local2GlobalMap l2g;
  l2g.InitializeFromMesh(&mesh, kShapeFields);
  NodalBasedStructure nbs(mesh.GetElementConnectivity());
  int const numGlobalDofs = l2g.GetGlobalRange().Max() + 1;
  auto const zeroSol = ColumnVector<real>::Zero(numGlobalDofs);

  constexpr int kBatchSize = 8;

  for (int numThreads : {0, 1, Min(8, TaskScheduler::GetNumSupportedLogicalProcessors())}) {
    TaskScheduler scheduler(numThreads);
    for (auto const& params : AllAssemblyModes()) {
      // Reference: gather overload with an explicit zero solution.
      double objRef = 0.0;
      auto resRef = ColumnVector<real>::Zero(numGlobalDofs);
      auto dresRef = ToBlockSparseMatrix<kShapeFields>(MakeSparseMatrix(l2g));
      dresRef.SetZero();
      AssembleObjResDRes<ShapeElement, kShapeFields, kBatchSize>(
          l2g,
          nbs,
          MakeSolutionIndependentOpWithDisp<kBatchSize>(),
          AsConstView(zeroSol),
          AssemblyResults<real>{
              .outObj = params.assemObj ? &objRef : nullptr,
              .outRes = params.assemRes ? ColumnVectorView<real>{resRef} : ColumnVectorView<real>{},
              .outDRes =
                  params.assemDRes ? AnyMatrixView<real>{AsView(dresRef)} : AnyMatrixView<real>{},
              .params = params},
          AssemblyActiveSubset{});

      // No-solution overload (no global solution argument).
      double objNo = 0.0;
      auto resNo = ColumnVector<real>::Zero(numGlobalDofs);
      auto dresNo = ToBlockSparseMatrix<kShapeFields>(MakeSparseMatrix(l2g));
      dresNo.SetZero();
      AssembleObjResDRes<ShapeElement, kShapeFields, kBatchSize>(
          l2g,
          nbs,
          MakeNoDispSolutionIndependentOp<kBatchSize>(),
          AssemblyResults<real>{
              .outObj = params.assemObj ? &objNo : nullptr,
              .outRes = params.assemRes ? ColumnVectorView<real>{resNo} : ColumnVectorView<real>{},
              .outDRes =
                  params.assemDRes ? AnyMatrixView<real>{AsView(dresNo)} : AnyMatrixView<real>{},
              .params = params},
          AssemblyActiveSubset{});

      if (params.assemObj) {
        EXPECT_NE(objRef, 0.0); // Sanity: the assembly is non-trivial.
        EXPECT_EQ(objRef, objNo);
      }
      if (params.assemRes) {
        EXPECT_EQ(0_r, ColumnVector<real>(resRef - resNo).Norm());
      }
      if (params.assemDRes) {
        ArrayMinusEquals(dresRef.Values(), MakeConstSpan(dresNo.Values()));
        EXPECT_EQ(0_r, dresRef.Norm());
      }
    }
  }
}

// ================================================================================================
// Projected (ROM) assembler tests.
//
// Reuses the synthetic op and the manual dense oracle: the projected reduced quantities are
// J^T r and J^T K J of the manually-assembled global residual / dresidual (with the per-element
// initEleDResFn diagonal folded in), since Sum_e J_e^T K_e J_e = J^T (Sum_e scatter(K_e)) J.
// ================================================================================================

// Mirrors the per-element dresidual initialization used by ROM assembly: a scaled diagonal.
static auto MakeInitEleDResFn(bool includeInit) {
  return [includeInit](auto&& eleDRes, int eleIdx) -> bool {
    eleDRes.SetZero();
    if (!includeInit) {
      return false;
    }
    real const scale = 0.01_r * static_cast<real>(1 + eleIdx);
    for (int i = 0; i < kShapeEleDofs; ++i) {
      eleDRes(i, i) = scale;
    }
    return true;
  };
}

// Add the initEleDResFn diagonal contribution to the manual global dense matrix.
static void AddInitToManual(
    ManualGlobalAssembly& g,
    Local2GlobalMap const& l2g,
    Span<int const> activeElements,
    bool includeInit) {
  if (!includeInit) {
    return;
  }
  auto addElement = [&](int e) {
    Span<int const> const gi = l2g.GetGlobalIndices(e);
    real const scale = 0.01_r * static_cast<real>(1 + e);
    for (int i = 0; i < isize(gi); ++i) {
      g.dres(gi[i], gi[i]) += scale;
    }
  };
  if (activeElements.empty()) {
    for (int e = 0; e < l2g.GetNumElements(); ++e) {
      addElement(e);
    }
  } else {
    for (int e : activeElements) {
      addElement(e);
    }
  }
}

template <int kBatchSize>
static void VerifyProjectedOneBatchSize(
    Local2GlobalMap const& l2g,
    ColumnVectorView<real const> sol,
    int numGlobalDofs,
    RowMatrix<real> const& J,
    int reducedDim,
    AssemblyActiveSubset const& subset,
    bool includeInit,
    AssemblyParams params,
    ManualGlobalAssembly const& g) {
  double obj = 0.0;
  auto redRes = ColumnVector<real>::Zero(reducedDim);
  auto redDRes = Matrix<real>::Zero(reducedDim, reducedDim);

  AssembleAndProjectObjResDRes<ShapeElement, kShapeFields, kBatchSize>(
      l2g,
      MakeSyntheticOp<kBatchSize>(),
      subset,
      MakeInitEleDResFn(includeInit),
      sol,
      AsConstView(J),
      obj,
      AsView(redRes),
      AsView(redDRes),
      params);

  if (params.assemObj) {
    EXPECT_NEAR_RTOL(g.obj, obj, double(kAssemblerRelTol));
  }
  if (params.assemRes) {
    // Expected reduced residual: J^T g.res.
    real num = 0_r;
    real den = 0_r;
    for (int k = 0; k < reducedDim; ++k) {
      real s = 0_r;
      for (int i = 0; i < numGlobalDofs; ++i) {
        s += J(i, k) * g.res[i];
      }
      num += Sqr(s - redRes[k]);
      den += Sqr(s);
    }
    EXPECT_LE(Sqrt(num), kAssemblerRelTol * Sqrt(den) + real(1e-12));
  }
  if (params.assemDRes) {
    // Expected reduced dresidual: J^T g.dres J, computed as J^T (g.dres J).
    auto gdJ = RowMatrix<real>::Zero(numGlobalDofs, reducedDim);
    for (int i = 0; i < numGlobalDofs; ++i) {
      for (int l = 0; l < reducedDim; ++l) {
        real s = 0_r;
        for (int j = 0; j < numGlobalDofs; ++j) {
          s += g.dres(i, j) * J(j, l);
        }
        gdJ(i, l) = s;
      }
    }
    real num = 0_r;
    real den = 0_r;
    for (int k = 0; k < reducedDim; ++k) {
      for (int l = 0; l < reducedDim; ++l) {
        real s = 0_r;
        for (int i = 0; i < numGlobalDofs; ++i) {
          s += J(i, k) * gdJ(i, l);
        }
        num += Sqr(s - redDRes(k, l));
        den += Sqr(s);
      }
    }
    EXPECT_LE(Sqrt(num), kAssemblerRelTol * Sqrt(den) + real(1e-12));
  }
}

static void RunProjectedConfig(
    Local2GlobalMap const& l2g,
    ColumnVectorView<real const> sol,
    int numGlobalDofs,
    RowMatrix<real> const& J,
    int reducedDim,
    AssemblyActiveSubset const& subset,
    Span<int const> activeElements,
    bool includeInit,
    AssemblyParams params) {
  ManualGlobalAssembly g = ManualAssemble(l2g, numGlobalDofs, sol, activeElements, params);
  AddInitToManual(g, l2g, activeElements, includeInit && params.assemDRes);
  VerifyProjectedOneBatchSize<1>(
      l2g, sol, numGlobalDofs, J, reducedDim, subset, includeInit, params, g);
  VerifyProjectedOneBatchSize<4>(
      l2g, sol, numGlobalDofs, J, reducedDim, subset, includeInit, params, g);
  VerifyProjectedOneBatchSize<8>(
      l2g, sol, numGlobalDofs, J, reducedDim, subset, includeInit, params, g);
}

static void RunProjectedAssemblerTests(TetrahedralMesh& mesh) {
  Local2GlobalMap l2g;
  l2g.InitializeFromMesh(&mesh, kShapeFields);
  int const numGlobalDofs = l2g.GetGlobalRange().Max() + 1;
  int const numElements = l2g.GetNumElements();

  ColumnVector<real> sol(numGlobalDofs);
  sol.SetRandom(123, -0.1_r, 0.1_r);

  auto const scenarios = MakeSubsetScenarios(numElements);

  for (int reducedDim : {1, 5, 10, 50}) {
    RowMatrix<real> J(numGlobalDofs, reducedDim);
    J.SetRandom(43, -1.0_r, 1.0_r);

    for (int numThreads : {0, 1, Min(8, TaskScheduler::GetNumSupportedLogicalProcessors())}) {
      TaskScheduler scheduler(numThreads);
      for (bool includeInit : {false, true}) {
        for (auto const& params : AllAssemblyModes()) {
          // All elements (empty active subset).
          RunProjectedConfig(
              l2g,
              AsConstView(sol),
              numGlobalDofs,
              J,
              reducedDim,
              AssemblyActiveSubset{},
              {},
              includeInit,
              params);

          // Active-subset scenarios (including mixed-lane edge cases).
          for (auto const& scenario : scenarios) {
            RunProjectedConfig(
                l2g,
                AsConstView(sol),
                numGlobalDofs,
                J,
                reducedDim,
                scenario.ActiveSubset(),
                MakeConstSpan(scenario.indices),
                includeInit,
                params);
          }
        }
      }
    }
  }
}

// Verifies the no-displacement projected overload produces results identical to the gather overload
// fed an all-zero solution, for a solution-independent op and a given active subset.
template <int kBatchSize>
static void VerifyProjectedNoDispMatchesZeroSolution(
    Local2GlobalMap const& l2g,
    RowMatrix<real> const& J,
    int reducedDim,
    ColumnVectorView<real const> zeroSol,
    AssemblyActiveSubset const& subset,
    AssemblyParams params) {
  // Reference: gather overload with an explicit zero solution.
  double objRef = 0.0;
  auto redResRef = ColumnVector<real>::Zero(reducedDim);
  auto redDResRef = Matrix<real>::Zero(reducedDim, reducedDim);
  AssembleAndProjectObjResDRes<ShapeElement, kShapeFields, kBatchSize>(
      l2g,
      MakeSolutionIndependentOpWithDisp<kBatchSize>(),
      subset,
      MakeInitEleDResFn(/*includeInit*/ true),
      zeroSol,
      AsConstView(J),
      objRef,
      AsView(redResRef),
      AsView(redDResRef),
      params);

  // No-solution overload (no global solution argument).
  double objNo = 0.0;
  auto redResNo = ColumnVector<real>::Zero(reducedDim);
  auto redDResNo = Matrix<real>::Zero(reducedDim, reducedDim);
  AssembleAndProjectObjResDRes<ShapeElement, kShapeFields, kBatchSize>(
      l2g,
      MakeNoDispSolutionIndependentOp<kBatchSize>(),
      subset,
      MakeInitEleDResFn(/*includeInit*/ true),
      AsConstView(J),
      objNo,
      AsView(redResNo),
      AsView(redDResNo),
      params);

  if (params.assemObj) {
    EXPECT_NE(objRef, 0.0); // Sanity: the assembly is non-trivial.
    EXPECT_EQ(objRef, objNo);
  }
  if (params.assemRes) {
    EXPECT_EQ(0_r, ColumnVector<real>(redResRef - redResNo).Norm());
  }
  if (params.assemDRes) {
    EXPECT_EQ(0_r, Matrix<real>(redDResRef - redDResNo).Norm());
  }
}

TEST(ProjectedElementAssembler, SyntheticCube) {
  TetrahedralMesh mesh = test::CreateMinimalTetMeshUnitCube({0.5_r, 1.0_r, 2.5_r});
  RunProjectedAssemblerTests(mesh);
}

TEST(ProjectedElementAssembler, SyntheticGrid) {
  TetrahedralMesh mesh = test::CreateMinimalTetMeshUnitGrid({2.0_r, 2.0_r, 2.0_r}, {2, 2, 2});
  RunProjectedAssemblerTests(mesh);
}

TEST(ProjectedElementAssembler, NoSolutionOverloadMatchesZeroSolution) {
  // For a solution-independent op, the no-solution projected overload must produce results
  // identical to the gather overload fed an all-zero solution.
  TetrahedralMesh mesh = test::CreateMinimalTetMeshUnitGrid({2.0_r, 2.0_r, 2.0_r}, {2, 2, 2});
  Local2GlobalMap l2g;
  l2g.InitializeFromMesh(&mesh, kShapeFields);
  int const numGlobalDofs = l2g.GetGlobalRange().Max() + 1;
  int const numElements = l2g.GetNumElements();

  constexpr int kReducedDim = 5;
  RowMatrix<real> J(numGlobalDofs, kReducedDim);
  J.SetRandom(43, -1.0_r, 1.0_r);

  auto const zeroSol = ColumnVector<real>::Zero(numGlobalDofs);

  auto const scenarios = MakeSubsetScenarios(numElements);

  for (int numThreads : {0, 1, Min(8, TaskScheduler::GetNumSupportedLogicalProcessors())}) {
    TaskScheduler scheduler(numThreads);
    for (auto const& params : AllAssemblyModes()) {
      // All elements (empty active subset).
      VerifyProjectedNoDispMatchesZeroSolution<8>(
          l2g, J, kReducedDim, AsConstView(zeroSol), AssemblyActiveSubset{}, params);

      // Active-subset scenarios (including mixed-lane edge cases).
      for (auto const& scenario : scenarios) {
        VerifyProjectedNoDispMatchesZeroSolution<8>(
            l2g, J, kReducedDim, AsConstView(zeroSol), scenario.ActiveSubset(), params);
      }
    }
  }
}

// ================================================================================================
// One physical-operation test: a real Neo-Hookean stress (volume) and an outward-normal traction
// (boundary) assembled through the FEM assembler, validated by implementation-independent
// invariants (force balance, divergence theorem) and by PSD-projection consistency near rest.
// ================================================================================================
TEST(ElementAssembler, PhysicalInvariantsAndPsd) {
  TetrahedralMesh mesh = test::CreateMinimalTetMeshUnitCube({0.5_r, 1.0_r, 2.5_r});
  constexpr int kFields = 3;
  constexpr int kBatchSize = kDefaultFemBatchSize;
  using ElementT = tetrahedral::Pk3DElement<1>;
  using TraceT = tetrahedral::Pk3DElementTrace<ElementT, 3>;
  using V = BatchReal<kBatchSize>;

  int const numElements = mesh.GetNumElements();
  DynamicArray<ElementT> elements;
  elements.reserve(numElements);
  for (int e = 0; e < numElements; ++e) {
    elements.emplace_back(e, mesh.GetNodeCoordinates(), mesh.GetElementConnectivity());
  }

  NeoHookeanMaterialParams materialParams;
  materialParams.youngsModulus = 1000_r;
  materialParams.poissonRatio = 0.3_r;

  Local2GlobalMap l2g;
  l2g.InitializeFromMesh(&mesh, kFields);
  NodalBasedStructure nbs(mesh.GetElementConnectivity());
  int const numGlobalDofs = l2g.GetGlobalRange().Max() + 1;
  int const numNodes = mesh.GetNumNodes();

  // Small, deterministic displacement (near rest, so the stress tangent is PSD).
  ColumnVector<real> sol(numGlobalDofs);
  for (int i = 0; i < numGlobalDofs; ++i) {
    sol[i] = static_cast<real>(i) / static_cast<real>(numGlobalDofs) * 0.1_r;
  }

  TaskScheduler scheduler(Min(8, TaskScheduler::GetNumSupportedLogicalProcessors()));

  auto const lame = materials::BuildBatchParams<kBatchSize>(materialParams);
  auto batchedConstitutive =
      [&](auto const&, auto const& F, auto* energy, auto* pk1, auto* tangent, bool psd) {
        materials::BatchedSmithNeoHookeanConstitutiveResponse<kBatchSize>(
            lame, F, energy, pk1, tangent, psd);
      };
  auto stressOp = [&](NdArray<int, kBatchSize> const& elementIndices,
                      Span<int const> /*indicesFlat*/,
                      fem::BatchElementVector<kBatchSize, ElementT> const& disp,
                      BatchDouble<kBatchSize>* outEnergy,
                      fem::BatchElementVector<kBatchSize, ElementT>* outRes,
                      fem::BatchElementMatrix<kBatchSize, ElementT>* outDRes,
                      bool projectPsd) -> bool {
    return fem::StressWork<kBatchSize>(
        elementIndices,
        MakeConstSpan(elements),
        disp,
        outEnergy,
        outRes,
        outDRes,
        projectPsd,
        batchedConstitutive);
  };

  // --- Volume stress: residual force balance and PSD-projection consistency ---
  auto const dresPattern = ToBlockSparseMatrix<kFields>(MakeSparseMatrix(l2g));

  auto resNoPsd = ColumnVector<real>::Zero(numGlobalDofs);
  auto dresNoPsd = dresPattern;
  dresNoPsd.SetZero();
  AssembleObjResDRes<ElementT, ElementT::kSpaceDim, kBatchSize>(
      l2g,
      nbs,
      stressOp,
      AsConstView(sol),
      AssemblyResults<real>{
          .outObj = nullptr,
          .outRes = AsView(resNoPsd),
          .outDRes = AsView(dresNoPsd),
          .params =
              AssemblyParams{
                  .assemObj = false, .assemRes = true, .assemDRes = true, .psdDRes = false}},
      AssemblyActiveSubset{});

  // Partition of unity: the internal elastic forces sum to zero.
  Real3 sumForce = {};
  for (int n = 0; n < numNodes; ++n) {
    sumForce[0] += resNoPsd[3 * n + 0];
    sumForce[1] += resNoPsd[3 * n + 1];
    sumForce[2] += resNoPsd[3 * n + 2];
  }
  EXPECT_NEAR_EQ(Norm(sumForce) / (resNoPsd.Norm() * numNodes), 0_r);

  // Near rest the stress tangent is PSD, so PSD projection must not change the residual/dresidual.
  auto resPsd = ColumnVector<real>::Zero(numGlobalDofs);
  auto dresPsd = dresPattern;
  dresPsd.SetZero();
  AssembleObjResDRes<ElementT, ElementT::kSpaceDim, kBatchSize>(
      l2g,
      nbs,
      stressOp,
      AsConstView(sol),
      AssemblyResults<real>{
          .outObj = nullptr,
          .outRes = AsView(resPsd),
          .outDRes = AsView(dresPsd),
          .params =
              AssemblyParams{
                  .assemObj = false, .assemRes = true, .assemDRes = true, .psdDRes = true}},
      AssemblyActiveSubset{});

  ExpectColumnsClose(AsConstView(resNoPsd), AsConstView(resPsd));
  {
    auto const noPsdValues = dresNoPsd.Values();
    auto const psdValues = MakeConstSpan(dresPsd.Values());
    real num = 0_r;
    real den = 0_r;
    for (int i = 0; i < isize(noPsdValues); ++i) {
      num += Sqr(noPsdValues[i] - psdValues[i]);
      den += Sqr(noPsdValues[i]);
    }
    EXPECT_LE(Sqrt(num), 1e-3_r * Sqrt(den) + real(1e-12));
  }

  // --- Projected (ROM) assembly of the real stress with PSD projection on. Validated against the
  //     regular PSD-projected global assembly projected by J, i.e. Sum_e J_e^T PSD(K_e) J_e ==
  //     J^T (Sum_e scatter(PSD(K_e))) J. Covers a real, PSD-projected tangent flowing through the
  //     projected path (residual is unaffected by PSD, so reusing resPsd is equivalent). ---
  constexpr int kReducedDim = 5;
  RowMatrix<real> J(numGlobalDofs, kReducedDim);
  J.SetRandom(91, -1.0_r, 1.0_r);

  double objProj = 0.0;
  auto resProj = ColumnVector<real>::Zero(kReducedDim);
  auto dresProj = Matrix<real>::Zero(kReducedDim, kReducedDim);
  AssembleAndProjectObjResDRes<ElementT, ElementT::kSpaceDim, kBatchSize>(
      l2g,
      stressOp,
      AssemblyActiveSubset{},
      MakeInitEleDResFn(false),
      AsConstView(sol),
      AsConstView(J),
      objProj,
      AsView(resProj),
      AsView(dresProj),
      AssemblyParams{.assemObj = false, .assemRes = true, .assemDRes = true, .psdDRes = true});

  // Reduced residual: J^T resPsd.
  {
    real num = 0_r;
    real den = 0_r;
    for (int k = 0; k < kReducedDim; ++k) {
      real s = 0_r;
      for (int i = 0; i < numGlobalDofs; ++i) {
        s += J(i, k) * resPsd[i];
      }
      num += Sqr(s - resProj[k]);
      den += Sqr(s);
    }
    EXPECT_LE(Sqrt(num), kAssemblerRelTol * Sqrt(den) + real(1e-12));
  }

  // Reduced dresidual: J^T dresPsd J, computed as J^T (dresPsd J).
  {
    auto const dresPsdDense = ToMatrix(dresPsd);
    auto gdJ = RowMatrix<real>::Zero(numGlobalDofs, kReducedDim);
    for (int i = 0; i < numGlobalDofs; ++i) {
      for (int l = 0; l < kReducedDim; ++l) {
        real s = 0_r;
        for (int j = 0; j < numGlobalDofs; ++j) {
          s += dresPsdDense(i, j) * J(j, l);
        }
        gdJ(i, l) = s;
      }
    }
    real num = 0_r;
    real den = 0_r;
    for (int k = 0; k < kReducedDim; ++k) {
      for (int l = 0; l < kReducedDim; ++l) {
        real s = 0_r;
        for (int i = 0; i < numGlobalDofs; ++i) {
          s += J(i, k) * gdJ(i, l);
        }
        num += Sqr(s - dresProj(k, l));
        den += Sqr(s);
      }
    }
    EXPECT_LE(Sqrt(num), kAssemblerRelTol * Sqrt(den) + real(1e-12));
  }

  // --- Boundary traction: divergence theorem (integral of the outward normal is zero) ---
  DynamicArray<TraceT> traces;
  traces.reserve(mesh.GetNumBoundaryFaces());
  for (auto const& bdFace : mesh.GetBoundaryFaces()) {
    traces.emplace_back(
        elements[bdFace.element],
        static_cast<int>(bdFace.faceNum),
        tetrahedral::kTetrahedralTraceQuadrature3[bdFace.faceNum]);
  }
  BoundaryAssemblyData bdData(
      MakeConstSpan(traces), mesh.GetElementConnectivity(), nbs.GetNToN(), kFields);

  auto tractionOp = [&](NdArray<int, kBatchSize> const& elementIndices,
                        Span<int const> /*indicesFlat*/,
                        BatchDouble<kBatchSize>* outEnergy,
                        fem::BatchElementVector<kBatchSize, TraceT>* outRes,
                        fem::BatchElementMatrix<kBatchSize, TraceT>* outDRes,
                        bool /*projectPsd*/) -> bool {
    // Traction force == outward face normal at the quad point. Energy is not requested here.
    auto normalForce = [&traces](
                           NdArray<int, kBatchSize> const& eleIndices,
                           int q,
                           BatchDouble<kBatchSize>* /*cbEnergy*/,
                           BatchReal3<kBatchSize>* cbForce,
                           NdArray<BatchReal3<kBatchSize>, 3>* /*cbDForce*/,
                           NdArray<bool, kBatchSize>& hasForce) {
      alignas(alignof(V)) real sx[V::kSize]{};
      alignas(alignof(V)) real sy[V::kSize]{};
      alignas(alignof(V)) real sz[V::kSize]{};
      for (int b = 0; b < isize(eleIndices); ++b) {
        hasForce[b] = true;
        auto const& normal = traces[eleIndices[b]].normals[q];
        sx[b] = normal[0];
        sy[b] = normal[1];
        sz[b] = normal[2];
      }
      if (cbForce) {
        (*cbForce)[0] = Load<V>(sx);
        (*cbForce)[1] = Load<V>(sy);
        (*cbForce)[2] = Load<V>(sz);
      }
    };
    return fem::TractionWork<kBatchSize, TraceT, kFields>(
        elementIndices, MakeConstSpan(traces), outEnergy, outRes, outDRes, normalForce);
  };

  auto resTraction = ColumnVector<real>::Zero(numGlobalDofs);
  AssembleObjResDRes<TraceT, TraceT::kSpaceDim, kBatchSize>(
      bdData.l2g,
      bdData.nbs,
      tractionOp,
      AssemblyResults<real>{
          .outObj = nullptr,
          .outRes = AsView(resTraction),
          .outDRes = {},
          .params =
              AssemblyParams{
                  .assemObj = false, .assemRes = true, .assemDRes = false, .psdDRes = false}},
      AssemblyActiveSubset{});

  auto const bdNodes = mesh.GetBoundaryNodes();
  Real3 sumNormal = {};
  for (int node : bdNodes) {
    sumNormal[0] += resTraction[3 * node + 0];
    sumNormal[1] += resTraction[3 * node + 1];
    sumNormal[2] += resTraction[3 * node + 2];
  }
  EXPECT_NEAR_EQ(Norm(sumNormal) / (resTraction.Norm() * isize(bdNodes)), 0_r);
}
