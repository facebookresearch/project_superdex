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
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/materials/batched_smith_neo_hookean.h>
#include <mochi_core/materials/material_params.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/local_to_global_map.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/sparsity_utils.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace mochi;

TEST(GlobalConsistency, cube) {
  // A solid unit cube with one corner at (0,0,0)
  ///
  //         2 ------- 3
  //       / |       / |
  //      /  |      /  |
  //     6 ------- 7   |
  //     |   0 ----|-- 1
  //     |  /      |  /
  //     | /       | /
  //     4 ------- 5
  //
  TetrahedralMesh mesh = test::CreateMinimalTetMeshUnitCube();

  // The polynomial interpolant
  constexpr int kPolyOrder = 1;

  // Create the material object
  NeoHookeanMaterialParams materialParams;

  // Create the elements and elemental operations
  using ElementT = tetrahedral::Pk3DElement<kPolyOrder>;
  int const num_elements = mesh.GetNumElements();

  // Create the elements
  std::vector<ElementT> elements;
  elements.reserve(num_elements);
  for (int e = 0; e < num_elements; ++e) {
    elements.emplace_back(e, mesh.GetNodeCoordinates(), mesh.GetElementConnectivity());
  }

  // Create the batched element operation: a single Neo-Hookean (Smith) stress term. The assembly
  // loop invokes this functor once per batch of elements.
  constexpr int kBatchSize = kDefaultFemBatchSize;
  auto const lame = materials::BuildBatchParams<kBatchSize>(materialParams);
  auto batchedConstitutive =
      [&](auto const&, auto const& F, auto* energy, auto* pk1, auto* tangent, bool psd) {
        materials::BatchedSmithNeoHookeanConstitutiveResponse<kBatchSize>(
            lame, F, energy, pk1, tangent, psd);
      };

  auto batchedStressOp = [&](NdArray<int, kBatchSize> const& batchElemIndices,
                             Span<int const> /*indicesFlat*/,
                             fem::BatchElementVector<kBatchSize, ElementT> const& batchDispl,
                             BatchDouble<kBatchSize>* outBatchEnergy,
                             fem::BatchElementVector<kBatchSize, ElementT>* outBatchRes,
                             fem::BatchElementMatrix<kBatchSize, ElementT>* outBatchDRes,
                             bool projectPsd) -> bool {
    return fem::StressWork<kBatchSize>(
        batchElemIndices,
        MakeConstSpan(elements),
        batchDispl,
        outBatchEnergy,
        outBatchRes,
        outBatchDRes,
        projectPsd,
        batchedConstitutive);
  };

  real volume = 0_r;
  for (auto& e : elements) {
    for (int q = 0; q < e.kNumQuadPoints; ++q) {
      volume += e.dMapEvaluatedDet[q] * e.quadrature.weights[q];
    }
  }
  EXPECT_TRUE(NearEqual(1_r, volume)); // unit cube

  // Create the local to global map
  constexpr int kFields = 3;
  tetrahedral::BarycentricBasisTetrahedra<1> basis;
  Local2GlobalMap l2g;
  l2g.InitializeFromMeshAndBasis(&mesh, basis, kFields);

  // Total dofs
  int const total_dofs = l2g.GetGlobalRange().Max() + 1;
  EXPECT_EQ(24, total_dofs);

  // Create the nodal based structure.
  NodalBasedStructure nbs(mesh.GetElementConnectivity());

  ColumnVector<real> sol(total_dofs);
  ColumnVector<real> res(total_dofs);
  auto dres = ToBlockSparseMatrix<3>(MakeSparseMatrix(l2g));

  // Set arbitrary(but deterministic) values into the solution vector
  for (int i = 0; i < total_dofs; ++i) {
    // Arbitrary value in range [0,1)
    sol[i] = static_cast<real>(i % total_dofs) / static_cast<real>(total_dofs) * 0.1_r;
  }

  res.SetZero();
  dres.SetZero();
  AssemblyParams tangentParams{
      .assemObj = false, .assemRes = true, .assemDRes = true, .psdDRes = true};
  AssemblyParams residualOnlyParams{
      .assemObj = false, .assemRes = true, .assemDRes = false, .psdDRes = false};

  // Create a task scheduler and bind it to this thread
  TaskScheduler scheduler({});

  // Assemble the system
  AssembleObjResDRes<ElementT, kFields, kBatchSize>(
      l2g,
      nbs,
      batchedStressOp,
      sol,
      AssemblyResults<real>{nullptr, res, AsView(dres), tangentParams});

  // Get the tangent
  RowMatrix<real> const dfuncval = ToMatrix(dres).Transpose();

  // Get the residual
  auto funcval = res.Duplicate();

  // The displacements
  auto argval = sol.Duplicate();

  // real const eps_argval = (*std::max_element(argval.begin(), argval.end()) * 1e-8_r) + 1e-9_r;
  real const eps_argval = 1.e-3_r; // This value works even with 32-bit floats

  RowMatrix<real> _dfuncval(total_dofs, total_dofs);

  for (int i = 0; i < isize(funcval); ++i) {
    res.SetZero();

    // Perturb state vector
    auto _argval = argval.Duplicate();
    _argval[i] += eps_argval;

    // Update state vector
    sol = _argval;

    // Assemble residual
    AssembleObjResDRes<ElementT, kFields, kBatchSize>(
        l2g,
        nbs,
        batchedStressOp,
        sol,
        AssemblyResults<real>{nullptr, res, {}, residualOnlyParams});

    // Copy residual
    auto _funcval = res.Duplicate();

    // Plot difference
    for (int j = 0; j < total_dofs; ++j) {
      _dfuncval(j, i) = (_funcval[j] - funcval[j]) / eps_argval;
    }
  }

  real error = 0_r;
  real norm_dfuncval = 0_r;
  for (int i = 0; i < total_dofs; ++i) {
    for (int j = 0; j < total_dofs; ++j) {
      real const diff = dfuncval(i, j) - _dfuncval(i, j);
      error += (diff * diff);
      norm_dfuncval += (_dfuncval(i, j)) * (_dfuncval(i, j));
    }
  }
  error = std::sqrt(error);
  norm_dfuncval = std::sqrt(norm_dfuncval);

  // We expect this ratio to be very close to zero. If not, that indicates
  // that the two methods produced significantly different results.
  EXPECT_GT(1e-3_r, error / norm_dfuncval);
}
