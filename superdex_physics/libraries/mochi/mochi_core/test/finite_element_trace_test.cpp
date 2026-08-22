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

#include <mochi_core/elements/tetrahedral/finite_element_trace.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/local_to_global_map.h>

#include <gtest/gtest.h>

#include <vector>

using namespace mochi;

TEST(TetrahedralFiniteElementTrace, QuadraturePointEvaluate) {
  // This test case compares the SIMD vs scalar implementations of the following functions:
  // mochi::tetrahedral::Pk3DElementTrace<>::QuadraturePointEvaluateMap, and
  // mochi::tetrahedral::Pk3DElementTrace<>::QuadraturePointEvaluateWeightNormal

  // TetrahedralMesh
  auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
  auto tetMesh = TetrahedralMesh{coordinates, connectivity};

  // Volume discretizations
  using VolumeElement = tetrahedral::Pk3DElement<1, 1>;
  std::vector<VolumeElement> femVolElements;
  femVolElements.reserve(tetMesh.GetNumElements());
  for (int i = 0; i < tetMesh.GetNumElements(); ++i) {
    femVolElements.emplace_back(
        i,
        tetMesh.GetNodeCoordinates(),
        tetMesh.GetElementConnectivity(),
        tetrahedral::kTetrahedralQuadrature1);
  }

  // Boundary discretization
  using BoundaryElement = tetrahedral::Pk3DElementTrace<tetrahedral::Pk3DElement<1, 1>, 6>;
  std::vector<BoundaryElement> femBoundaryElements;
  std::vector<int> femBoundaryLocalToGlobal;
  femBoundaryElements.reserve(tetMesh.GetNumBoundaryFaces());
  femBoundaryLocalToGlobal.reserve(tetMesh.GetNumBoundaryFaces());
  for (auto const& bdface : tetMesh.GetBoundaryFaces()) {
    // For traction work, collisions, etc.
    femBoundaryLocalToGlobal.push_back(bdface.element);
    femBoundaryElements.emplace_back(
        femVolElements[bdface.element],
        bdface.faceNum,
        tetrahedral::kTetrahedralTraceQuadrature6[bdface.faceNum]);
  }

  // For each boundary element
  for (int e = 0; e < isize(femBoundaryElements); ++e) {
    // Fetch element and indices of the degrees of freedom.
    auto const& element = femBoundaryElements[e];
    for (int q = 0; q < BoundaryElement::kNumQuadPoints; ++q) {
      // Compute normal via scalar math (no displacement)
      Real3 map;
      Matrix3x3r dmap;
      element.QuadraturePointEvaluateMap(q, element.nodesCrdsPhys, map, dmap);
      real det = Det(dmap);
      real unused = {};
      Real3 normal = {};
      element.QuadraturePointEvaluateWeightNormal(q, det, Invert(dmap, det), unused, normal);

      // Compute normal again via SIMD math. This time skip the determinant calculation as well just
      // to prove we can (Invert3x3 would divide by det, and QuadraturePointEvaluateWeightNormal
      // would scale by det).
      Vec4r vmap;
      VMatrix3x3r vdmap;
      VMatrix4x3r vnodeCoords;
      LoadMatrix(vnodeCoords, element.nodesCrdsPhys);
      element.QuadraturePointEvaluateMap(q, vnodeCoords, vmap, vdmap);
      real vunused = {};
      Vec4r vnormal = {};
      element.QuadraturePointEvaluateWeightNormal(q, 1_r, Invert3x3(vdmap, 1_r), vunused, vnormal);

      // The two implementation should match (within tolerance)
      EXPECT_NEAR_EQ(map, ToReal3(vmap));
      EXPECT_NEAR_EQ(dmap, ToNdArray3x3(vdmap));
      EXPECT_NEAR_EQ(normal, ToReal3(vnormal));
    }
  }
}
