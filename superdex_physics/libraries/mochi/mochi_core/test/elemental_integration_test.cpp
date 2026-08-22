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

#include <mochi_core/element_operations/elemental_integration.h>

#include <mochi_core/element_operations/fem_stress.h>
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/elements/tetrahedral/simplex_quadrature.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/materials/batched_smith_neo_hookean.h>
#include <mochi_core/test/mochi_test_helpers.h>

#include <gtest/gtest.h>

#include <vector>

using namespace mochi;
using namespace mochi::fem;
using namespace mochi::materials;

TEST(ElementalIntegration, SmithNeoHookeanTetMatchesStressWork) {
  using TetElement = tetrahedral::Pk3DElement<1, 1>;
  constexpr int kBS = 1;
  constexpr int kDim = TetElement::kSpaceDim * TetElement::kNumDofs;

  Matrix<real, 4, 3> const tetBariDiff{
      {-1_r, 1_r, 0_r, 0_r},
      {-1_r, 0_r, 1_r, 0_r},
      {-1_r, 0_r, 0_r, 1_r},
  };
  StridedMatrixView<real const, 4, 3> cuBariDiff{tetBariDiff.data()};
  StridedMatrix<real, 3, 4> nodeData{
      {0.0_r, 0.3_r, -0.7_r},
      {2.0_r, 1.0_r, 0.0_r},
      {1.0_r, 3.0_r, 0.0_r},
      {0.5_r, 0.6_r, 8.0_r},
  };
  StridedMatrix<real, 3, 4> dispData{
      {0.3_r, 0.1_r, 0.2_r},
      {0.12_r, 0.5_r, 0.6_r},
      {0.9_r, 0.3_r, -0.5_r},
      {0.35_r, 0.26_r, -0.18_r},
  };

  real constexpr kE = 1_r;
  real constexpr kNu = 0.4_r;
  SmithNeoHookeanPseudoLame const pseudoLame{kE, kNu};
  StridedMatrix<real, kDim, kDim> integrationStiffness{};
  StridedMatrix<real, 3, 4> integrationForce{};
  StridedMatrix<real, 3, 4> disp = 0.1_r * dispData;
  integrationStiffness.SetZero();
  integrationForce.SetZero();
  real integrationEnergy{0_r};
  Span bdiffs{&cuBariDiff, 1};
  Integrate(
      pseudoLame,
      integrationEnergy,
      integrationForce,
      integrationStiffness,
      disp,
      nodeData,
      bdiffs,
      tetrahedral::kTetrahedralQuadrature1.weights);

  for (int r = 0; r < kDim; ++r) {
    real total = 0_r;
    real absTotal = 0_r;
    for (int c = 0; c < kDim; ++c) {
      total += integrationStiffness(r, c);
      absTotal += Abs(integrationStiffness(r, c));
    }
    EXPECT_LT(Abs(total), absTotal * 1e-6_r);
  }

  auto node = [&nodeData](int i) { return Real3{nodeData(0, i), nodeData(1, i), nodeData(2, i)}; };
  std::vector<Real3> const coordinates{node(0), node(1), node(2), node(3)};
  std::vector<Int4> const connectivity{Int4{0, 1, 2, 3}};
  TetrahedralMesh const mesh{coordinates, connectivity};
  TetElement const element{
      0,
      mesh.GetNodeCoordinates(),
      mesh.GetElementConnectivity(),
      tetrahedral::kTetrahedralQuadrature1};

  SmithNeoHookeanMaterialParams materialParams;
  materialParams.youngsModulus = kE;
  materialParams.poissonRatio = kNu;
  materialParams.psdStrategy = MaterialPsdStrategy::None;
  auto const lame = BuildBatchParams<kBS>(materialParams);
  auto const constitutive =
      [lame](auto const&, auto const& F, auto* e, auto* pk1, auto* tangent, bool project) {
        BatchedSmithNeoHookeanConstitutiveResponse<kBS>(
            lame, F, e, pk1, tangent, project, MaterialPsdOracle::None);
      };

  BatchElementVector<kBS, TetElement> stressDisp{};
  for (int k = 0; k < kDim; ++k) {
    stressDisp[k] = BatchReal<kBS>{disp.Data()[k]};
  }
  BatchDouble<kBS> stressEnergy{0.0};
  BatchElementVector<kBS, TetElement> stressRes{};
  NdArray<int, kBS> idx{0};
  StressWork<kBS>(
      idx,
      MakeSingletonConstSpan(element),
      stressDisp,
      &stressEnergy,
      &stressRes,
      nullptr,
      false,
      constitutive);

  EXPECT_NEAR_EQ(stressEnergy[0], static_cast<double>(integrationEnergy));
  for (int k = 0; k < kDim; ++k) {
    EXPECT_NEAR_EQ(stressRes[k][0], integrationForce.Data()[k]);
  }
}
