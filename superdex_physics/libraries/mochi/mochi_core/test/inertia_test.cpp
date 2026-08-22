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
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

#include <vector>

using namespace mochi;

TEST(FemInertia, MassMatrix) {
  constexpr size_t kDim = 12;
  // clang-format off
  constexpr real kExactMass[kDim][kDim] = {
    {0.016666668_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r  },
    {0.0_r , 0.016666668_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r  },
    {0.0_r , 0.0_r , 0.016666668_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r  },
    {0.008333334_r , 0.0_r , 0.0_r , 0.016666668_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r  },
    {0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.016666668_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r  },
    {0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.016666668_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r  },
    {0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.016666668_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r  },
    {0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.016666668_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r  },
    {0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.016666668_r , 0.0_r , 0.0_r , 0.008333334_r  },
    {0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.016666668_r , 0.0_r , 0.0_r  },
    {0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.016666668_r , 0.0_r  },
    {0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.008333334_r , 0.0_r , 0.0_r , 0.016666668_r  }
    };
  // clang-format on

  constexpr int kPolyOrder = 1;
  constexpr int kQuadDegree = 4;
  constexpr real kDensity = 1_r;

  using ElementT = tetrahedral::Pk3DElement<kPolyOrder, kQuadDegree>;

  TetrahedralMesh mesh = test::CreateMinimalTetMeshSingleTet();

  auto element = ElementT{
      0,
      mesh.GetNodeCoordinates(),
      mesh.GetElementConnectivity(),
      tetrahedral::kTetrahedralQuadrature4};
  auto elements = Span<ElementT>{&element, 1};

  // Compute the mass matrix per element
  using MassMatrix = NdArray<real, kDim, kDim>;
  std::vector<MassMatrix> massMatrixPerElem(elements.size());
  fem::ComputeMassMatrixPerElement(MakeConstSpan(elements), kDensity, MakeSpan(massMatrixPerElem));

  // Compare mass matrix for element 0 vs known values
  for (int i = 0; i < kDim; ++i) {
    for (int j = 0; j < kDim; ++j) {
      EXPECT_TRUE(NearEqual(massMatrixPerElem[0][i][j], kExactMass[i][j]));
    }
  }
}
