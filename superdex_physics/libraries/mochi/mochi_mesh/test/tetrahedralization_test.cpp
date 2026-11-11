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

#include "mesh_test_helpers.h"

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_mesh/tetrahedralization.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <set>

using namespace mochi;
using namespace mochi::mesh;
using namespace mochi::test;

namespace {

Real3 NodePosition(MeshData const& mesh, int node) {
  return {
      mesh.coordinates[3 * node], mesh.coordinates[3 * node + 1], mesh.coordinates[3 * node + 2]};
}

real TetVolume(MeshData const& mesh, int tet) {
  int const* const nodes = &mesh.connectivity[4 * tet];
  Real3 const origin = NodePosition(mesh, nodes[0]);
  return Dot(NodePosition(mesh, nodes[1]) - origin,
             Cross(NodePosition(mesh, nodes[2]) - origin, NodePosition(mesh, nodes[3]) - origin)) /
      6_r;
}

// Checks every node is used and every tetrahedron is positively oriented; returns the total volume.
real ExpectValidTetMesh(MeshData const& mesh) {
  EXPECT_EQ(mesh.nodesPerElement, 4);
  EXPECT_GT(mesh.GetNumElements(), 0);
  DynamicArray<bool> isReferenced(mesh.GetNumNodes(), false);
  for (int const node : mesh.connectivity) {
    isReferenced[node] = true;
  }
  EXPECT_EQ(std::count(isReferenced.begin(), isReferenced.end(), false), 0);
  real totalVolume = 0_r;
  for (int tet = 0; tet < mesh.GetNumElements(); ++tet) {
    real const volume = TetVolume(mesh, tet);
    EXPECT_GT(volume, 0_r);
    totalVolume += volume;
  }
  return totalVolume;
}

MeshData ScaledCube(real scale) {
  MeshData cube = CreateCubeTriMesh();
  for (real& coordinate : cube.coordinates) {
    coordinate *= scale;
  }
  return cube;
}

// Two overlapping unit cubes: each is closed, but their surfaces intersect.
MeshData OverlappingCubes() {
  MeshData mesh = CreateCubeTriMesh();
  MeshData const shifted = CreateCubeTriMesh();
  int const offset = mesh.GetNumNodes();
  for (int node = 0; node < shifted.GetNumNodes(); ++node) {
    for (real const coordinate : NodePosition(shifted, node) + Real3{0.5_r, 0.5_r, 0.5_r}) {
      mesh.coordinates.push_back(coordinate);
    }
  }
  for (int const node : shifted.connectivity) {
    mesh.connectivity.push_back(node + offset);
  }
  return mesh;
}

MeshData TetrahedralizeCube(real edgeSize) {
  TetrahedralizationParams params;
  params.edgeSize = edgeSize;
  return Tetrahedralize(CreateCubeTriMesh(), params, ExpectOK{});
}

// Octahedron with its vertices on the axes, scaled by @p scale. Its faces are convex hull facets,
// which every tetrahedralization of its vertices contains, so it needs no added vertices.
MeshData Octahedron(real scale, bool inward = false) {
  MeshData mesh;
  mesh.nodesPerElement = 3;
  mesh.coordinates = {
      // clang-format off
      1_r, 0_r, 0_r,  -1_r, 0_r, 0_r,
      0_r, 1_r, 0_r,  0_r, -1_r, 0_r,
      0_r, 0_r, 1_r,  0_r, 0_r, -1_r,
      // clang-format on
  };
  for (real& coordinate : mesh.coordinates) {
    coordinate *= scale;
  }
  mesh.connectivity = {
      // clang-format off
      0, 2, 4,  2, 1, 4,  1, 3, 4,  3, 0, 4,
      2, 0, 5,  1, 2, 5,  3, 1, 5,  0, 3, 5,
      // clang-format on
  };
  if (inward) {
    for (auto& face : Unflatten<Int3>(MakeSpan(mesh.connectivity))) {
      std::swap(face[1], face[2]);
    }
  }
  return mesh;
}

real OctahedronVolume(real scale) {
  return 4_r / 3_r * scale * scale * scale;
}

MeshData CombineSurfaces(std::initializer_list<MeshData> surfaces) {
  MeshData result;
  result.nodesPerElement = 3;
  for (MeshData const& surface : surfaces) {
    int const offset = result.GetNumNodes();
    Append(result.coordinates, surface.coordinates);
    for (int const node : surface.connectivity) {
      result.connectivity.push_back(node + offset);
    }
  }
  return result;
}

// Schönhardt's twisted triangular prism: a closed surface that cannot be filled with tetrahedra
// without adding vertices. Each side splits along its reflex diagonal.
MeshData SchonhardtPrism() {
  MeshData mesh;
  mesh.nodesPerElement = 3;
  for (int ring = 0; ring < 2; ++ring) {
    for (int i = 0; i < 3; ++i) {
      double const angle = (120.0 * i + 30.0 * ring) * std::numbers::pi / 180.0;
      mesh.coordinates.push_back(static_cast<real>(std::cos(angle)));
      mesh.coordinates.push_back(static_cast<real>(std::sin(angle)));
      mesh.coordinates.push_back(static_cast<real>(ring));
    }
  }
  // Bottom vertices are 0-2, top vertices 3-5.
  mesh.connectivity = {0, 2, 1, 3, 4, 5};
  for (int i = 0; i < 3; ++i) {
    int const next = (i + 1) % 3;
    for (int const corner : {i, next, 3 + next, i, 3 + next, 3 + i}) {
      mesh.connectivity.push_back(corner);
    }
  }
  return mesh;
}

std::set<std::array<int, 3>> SortedTriangles(Span<Int3 const> triangles) {
  std::set<std::array<int, 3>> sorted;
  for (Int3 const& tri : triangles) {
    std::array<int, 3> nodes{tri[0], tri[1], tri[2]};
    std::ranges::sort(nodes);
    sorted.insert(nodes);
  }
  return sorted;
}

TetrahedralMesh AsTetMesh(MeshData const& tets) {
  return {
      Unflatten<Real3 const>(MakeConstSpan(tets.coordinates)),
      Unflatten<Int4 const>(MakeConstSpan(tets.connectivity))};
}

void ExpectBoundaryUnchanged(MeshData const& surface, MeshData const& tets) {
  EXPECT_EQ(
      SortedTriangles(AsTetMesh(tets).GetBoundaryFacesConnectivity()),
      SortedTriangles(Unflatten<Int3 const>(MakeConstSpan(surface.connectivity))));
}

real TriangleArea(MeshData const& mesh, Int3 const& face) {
  return Norm(Cross(
             NodePosition(mesh, face[1]) - NodePosition(mesh, face[0]),
             NodePosition(mesh, face[2]) - NodePosition(mesh, face[0]))) *
      0.5_r;
}

real SurfaceVolume(MeshData const& surface) {
  real volume = 0_r;
  for (Int3 const& face : Unflatten<Int3 const>(MakeConstSpan(surface.connectivity))) {
    volume += Dot(NodePosition(surface, face[0]),
                  Cross(NodePosition(surface, face[1]), NodePosition(surface, face[2]))) /
        6_r;
  }
  return volume;
}

// Each boundary triangle is contained in an input triangle, and covers the same area collectively.
void ExpectBoundaryShapePreserved(MeshData const& surface, MeshData const& tets) {
  auto const inputFaces = Unflatten<Int3 const>(MakeConstSpan(surface.connectivity));
  DynamicArray<real> coveredArea(inputFaces.size(), 0_r);
  TetrahedralMesh const tetMesh = AsTetMesh(tets);
  for (Int3 const& face : tetMesh.GetBoundaryFacesConnectivity()) {
    bool matched = false;
    for (int tri = 0; tri < inputFaces.size(); ++tri) {
      Int3 const& input = inputFaces[tri];
      VDistanceSignParams sign;
      bool const contained = std::ranges::all_of(face, [&](int node) {
        return VDistancePointTriangleSqr(
                   ToSimd(NodePosition(tets, node)),
                   ToSimd(NodePosition(surface, input[0])),
                   ToSimd(NodePosition(surface, input[1])),
                   ToSimd(NodePosition(surface, input[2])),
                   sign)[0] <= Sqr(1e-5_r);
      });
      if (contained) {
        coveredArea[tri] += TriangleArea(tets, face);
        matched = true;
        break;
      }
    }
    ASSERT_TRUE(matched) << "Boundary triangle leaves the input surface";
  }
  for (int tri = 0; tri < inputFaces.size(); ++tri) {
    EXPECT_NEAR(coveredArea[tri], TriangleArea(surface, inputFaces[tri]), 1e-5_r);
  }
}

// The first nodes of @p tets are exactly the nodes of @p surface, in order.
void ExpectInputVerticesFirst(MeshData const& surface, MeshData const& tets) {
  ASSERT_GE(tets.coordinates.size(), surface.coordinates.size());
  EXPECT_TRUE(
      std::equal(surface.coordinates.begin(), surface.coordinates.end(), tets.coordinates.begin()));
}

MeshData TetrahedralizeConstrainedOctahedron(real edgeSize) {
  ConstrainedTetrahedralizationParams params;
  params.edgeSize = edgeSize;
  return TetrahedralizeConstrained(Octahedron(1_r), params, ExpectOK{});
}

} // namespace

TEST(Tetrahedralization, CubeProducesValidTetMesh) {
  // The unit cube's flat faces and sharp edges are representable exactly.
  EXPECT_NEAR(ExpectValidTetMesh(TetrahedralizeCube(0.5_r)), 1_r, 1e-3_r);
}

TEST(Tetrahedralization, OptimizerAndFeatureOptionsProduceValidMeshes) {
  using Option = void (*)(TetrahedralizationParams&);
  Option const options[] = {
      [](TetrahedralizationParams& p) { p.perturb = false; },
      [](TetrahedralizationParams& p) { p.exude = false; },
      [](TetrahedralizationParams& p) { p.odt = true; },
      [](TetrahedralizationParams& p) { p.detectFeatures = false; },
  };
  for (Option const option : options) {
    TetrahedralizationParams params;
    params.edgeSize = 0.5;
    option(params);
    ExpectValidTetMesh(Tetrahedralize(CreateCubeTriMesh(), params, ExpectOK{}));
  }
}

TEST(Tetrahedralization, SmallerEdgeSizeProducesMoreTets) {
  EXPECT_GT(
      TetrahedralizeCube(0.25_r).GetNumElements(), TetrahedralizeCube(0.5_r).GetNumElements());
}

TEST(Tetrahedralization, ZeroFacetDistanceBoundUsesEdgeSizeFraction) {
  TetrahedralizationParams automatic;
  automatic.edgeSize = 0.5;
  automatic.relativeToMeshSize = false;
  automatic.facetDistanceBound = 0.0;

  TetrahedralizationParams explicitDistanceBound = automatic;
  explicitDistanceBound.facetDistanceBound = automatic.edgeSize / 5.0;

  MeshData const automaticResult = Tetrahedralize(CreateCubeTriMesh(), automatic, ExpectOK{});
  MeshData const explicitResult =
      Tetrahedralize(CreateCubeTriMesh(), explicitDistanceBound, ExpectOK{});
  EXPECT_EQ(automaticResult.coordinates, explicitResult.coordinates);
  EXPECT_EQ(automaticResult.connectivity, explicitResult.connectivity);
}

TEST(Tetrahedralization, RelativeSizesScaleWithTheMesh) {
  int const unitCubeTets = TetrahedralizeCube(0.5_r).GetNumElements();

  TetrahedralizationParams relative;
  relative.edgeSize = 0.5;
  MeshData const relativeResult = Tetrahedralize(ScaledCube(2_r), relative, ExpectOK{});
  EXPECT_NEAR(ExpectValidTetMesh(relativeResult), 8_r, 1e-2_r);
  EXPECT_EQ(relativeResult.GetNumElements(), unitCubeTets);

  TetrahedralizationParams absolute = relative;
  absolute.relativeToMeshSize = false;
  EXPECT_GT(Tetrahedralize(ScaledCube(2_r), absolute, ExpectOK{}).GetNumElements(), unitCubeTets);
}

TEST(Tetrahedralization, RelativeSizesIgnoreUnusedVertices) {
  MeshData const expected = TetrahedralizeCube(0.5_r);

  MeshData cubeWithUnusedVertex = CreateCubeTriMesh();
  cubeWithUnusedVertex.coordinates.push_back(1000_r);
  cubeWithUnusedVertex.coordinates.push_back(0_r);
  cubeWithUnusedVertex.coordinates.push_back(0_r);
  TetrahedralizationParams params;
  params.edgeSize = 0.5;
  MeshData const actual = Tetrahedralize(cubeWithUnusedVertex, params, ExpectOK{});

  EXPECT_EQ(actual.coordinates, expected.coordinates);
  EXPECT_EQ(actual.connectivity, expected.connectivity);
}

TEST(Tetrahedralization, OutputIsDeterministic) {
  MeshData const first = TetrahedralizeCube(0.3_r);
  MeshData const second = TetrahedralizeCube(0.3_r);
  EXPECT_EQ(first.coordinates, second.coordinates);
  EXPECT_EQ(first.connectivity, second.connectivity);
}

TEST(Tetrahedralization, OpenSurfaceIsRejected) {
  (void)Tetrahedralize(CreateCubeTriMeshWithHole(), TetrahedralizationParams{}, ExpectNotOK{});
}

TEST(Tetrahedralization, SelfIntersectingSurfaceIsRejected) {
  (void)Tetrahedralize(OverlappingCubes(), TetrahedralizationParams{}, ExpectNotOK{});
}

// Rejected by the client before it reaches the helper.
TEST(Tetrahedralization, TetInputIsRejected) {
  MeshData const tets = TetrahedralizeCube(0.5_r);
  (void)Tetrahedralize(tets, TetrahedralizationParams{}, ExpectNotOK{});
}

TEST(Tetrahedralization, InvalidParamsAreRejected) {
  using Invalidation = void (*)(TetrahedralizationParams&);
  Invalidation const invalidations[] = {
      [](TetrahedralizationParams& p) { p.edgeSize = 0.0; },
      [](TetrahedralizationParams& p) { p.facetDistanceBound = -1.0; },
      [](TetrahedralizationParams& p) { p.facetAngleBound = 31.0; },
      [](TetrahedralizationParams& p) { p.cellRadiusEdgeRatio = 1.5; },
      [](TetrahedralizationParams& p) { p.sharpFeatureAngle = 0.0; },
  };
  for (Invalidation const invalidate : invalidations) {
    TetrahedralizationParams params;
    invalidate(params);
    (void)Tetrahedralize(CreateCubeTriMesh(), params, ExpectNotOK{});
  }
}

TEST(Tetrahedralization, InvalidScaledLengthsAreRejected) {
  using Invalidation = void (*)(TetrahedralizationParams&);
  struct TestCase {
    real cubeScale;
    Invalidation invalidate;
  };
  TestCase const testCases[] = {
      {.cubeScale = 2_r,
       .invalidate =
           [](TetrahedralizationParams& p) { p.edgeSize = std::numeric_limits<double>::max(); }},
      {.cubeScale = 0.5_r,
       .invalidate =
           [](TetrahedralizationParams& p) {
             p.edgeSize = std::numeric_limits<double>::denorm_min();
           }},
      {.cubeScale = 2_r,
       .invalidate =
           [](TetrahedralizationParams& p) {
             p.facetDistanceBound = std::numeric_limits<double>::max();
           }},
      {.cubeScale = 0.5_r,
       .invalidate =
           [](TetrahedralizationParams& p) {
             p.facetDistanceBound = std::numeric_limits<double>::denorm_min();
           }},
  };
  for (TestCase const& testCase : testCases) {
    TetrahedralizationParams params;
    testCase.invalidate(params);
    Error error;
    (void)Tetrahedralize(ScaledCube(testCase.cubeScale), params, error);
    EXPECT_NOT_OK(error);
    EXPECT_STREQ(
        error.GetDescription(), "Scaled tetrahedralization lengths must be positive and finite.");
  }
}

TEST(ConstrainedTetrahedralization, KeepsInputVerticesFirstAndBoundaryUnchanged) {
  MeshData const octahedron = Octahedron(1_r);
  for (bool const strict : {false, true}) {
    ConstrainedTetrahedralizationParams params;
    params.strict = strict;
    MeshData const tets = TetrahedralizeConstrained(octahedron, params, ExpectOK{});
    EXPECT_NEAR(ExpectValidTetMesh(tets), OctahedronVolume(1_r), 1e-5_r);
    EXPECT_GT(tets.GetNumNodes(), octahedron.GetNumNodes());
    ExpectInputVerticesFirst(octahedron, tets);
    ExpectBoundaryUnchanged(octahedron, tets);
  }
}

TEST(ConstrainedTetrahedralization, PreservesHollowShellAndNestedMaterialIsland) {
  struct TestCase {
    MeshData surface;
    real volume = 0_r;
  };
  TestCase const cases[] = {
      {.surface = CombineSurfaces({Octahedron(2_r), Octahedron(1_r, true)}),
       .volume = OctahedronVolume(2_r) - OctahedronVolume(1_r)},
      {.surface = CombineSurfaces({Octahedron(2_r), Octahedron(1_r, true), Octahedron(0.5_r)}),
       .volume = OctahedronVolume(2_r) - OctahedronVolume(1_r) + OctahedronVolume(0.5_r)},
  };
  for (TestCase const& test : cases) {
    for (bool const strict : {false, true}) {
      ConstrainedTetrahedralizationParams params;
      params.edgeSize = 0.4;
      params.relativeToMeshSize = false;
      params.iterations = 1;
      params.strict = strict;
      MeshData const tets = TetrahedralizeConstrained(test.surface, params, ExpectOK{});
      ASSERT_GT(tets.GetNumElements(), 0);
      EXPECT_NEAR(ExpectValidTetMesh(tets), test.volume, 1e-5_r);
      ExpectInputVerticesFirst(test.surface, tets);
      ExpectBoundaryUnchanged(test.surface, tets);
    }
  }
}

TEST(ConstrainedTetrahedralization, PreservesDisjointSolids) {
  MeshData shifted = Octahedron(2_r);
  for (int node = 0; node < shifted.GetNumNodes(); ++node) {
    shifted.coordinates[3 * node] += 6_r;
  }
  MeshData const surface = CombineSurfaces({Octahedron(2_r), shifted});
  ConstrainedTetrahedralizationParams params;
  params.edgeSize = 0.4;
  params.relativeToMeshSize = false;
  params.iterations = 1;
  MeshData const tets = TetrahedralizeConstrained(surface, params, ExpectOK{});
  ASSERT_GT(tets.GetNumElements(), 0);
  EXPECT_NEAR(ExpectValidTetMesh(tets), 2_r * OctahedronVolume(2_r), 1e-5_r);
  ExpectInputVerticesFirst(surface, tets);
  ExpectBoundaryUnchanged(surface, tets);
}

TEST(ConstrainedTetrahedralization, SmallerEdgeSizeProducesMoreTets) {
  EXPECT_GT(
      TetrahedralizeConstrainedOctahedron(0.1_r).GetNumElements(),
      TetrahedralizeConstrainedOctahedron(0.3_r).GetNumElements());
}

TEST(ConstrainedTetrahedralization, RelativeEdgeSizeIsAFractionOfTheMeshExtent) {
  // The scale 2 octahedron's bounding box is 4 wide along every axis.
  ConstrainedTetrahedralizationParams relative;
  relative.edgeSize = 0.1;
  ConstrainedTetrahedralizationParams absolute;
  absolute.edgeSize = 0.4;
  absolute.relativeToMeshSize = false;

  MeshData const relativeResult = TetrahedralizeConstrained(Octahedron(2_r), relative, ExpectOK{});
  MeshData const absoluteResult = TetrahedralizeConstrained(Octahedron(2_r), absolute, ExpectOK{});
  EXPECT_NEAR(ExpectValidTetMesh(relativeResult), OctahedronVolume(2_r), 1e-4_r);
  EXPECT_EQ(relativeResult.coordinates, absoluteResult.coordinates);
  EXPECT_EQ(relativeResult.connectivity, absoluteResult.connectivity);
}

TEST(ConstrainedTetrahedralization, OutputIsDeterministic) {
  MeshData const first = TetrahedralizeConstrainedOctahedron(0.1_r);
  MeshData const second = TetrahedralizeConstrainedOctahedron(0.1_r);
  EXPECT_EQ(first.coordinates, second.coordinates);
  EXPECT_EQ(first.connectivity, second.connectivity);
}

// The cube's eight vertices are cospherical, so its face diagonals need not be edges of their
// Delaunay tetrahedralization.
TEST(ConstrainedTetrahedralization, StrictModeRejectsSurfacesNeedingAddedVertices) {
  ConstrainedTetrahedralizationParams params;
  params.strict = true;
  for (MeshData const& surface : {SchonhardtPrism(), CreateCubeTriMesh()}) {
    (void)TetrahedralizeConstrained(surface, params, ExpectNotOK{});
  }
}

TEST(ConstrainedTetrahedralization, NonStrictModeAddsVerticesAfterInputVertices) {
  ConstrainedTetrahedralizationParams params;
  ASSERT_FALSE(params.strict);
  // Remeshing cannot split the coarse input edges, so a finer interior would form slivers along
  // them.
  params.edgeSize = 1.0;

  MeshData const prism = SchonhardtPrism();
  MeshData const prismTets = TetrahedralizeConstrained(prism, params, ExpectOK{});
  EXPECT_NEAR(ExpectValidTetMesh(prismTets), SurfaceVolume(prism), 1e-5_r);
  EXPECT_GT(prismTets.GetNumNodes(), prism.GetNumNodes());
  ExpectInputVerticesFirst(prism, prismTets);
  ExpectBoundaryShapePreserved(prism, prismTets);

  MeshData const cube = CreateCubeTriMesh();
  MeshData const cubeTets = TetrahedralizeConstrained(cube, params, ExpectOK{});
  EXPECT_NEAR(ExpectValidTetMesh(cubeTets), 1_r, 1e-5_r);
  ExpectInputVerticesFirst(cube, cubeTets);
  ExpectBoundaryShapePreserved(cube, cubeTets);
}

TEST(ConstrainedTetrahedralization, InvalidSurfacesAreRejected) {
  MeshData cubeWithUnusedVertex = CreateCubeTriMesh();
  for (real const coordinate : {2_r, 0_r, 0_r}) {
    cubeWithUnusedVertex.coordinates.push_back(coordinate);
  }

  // A second cube touching the first at a corner, through a vertex of its own.
  MeshData cornerTouchingCubes = CreateCubeTriMesh();
  MeshData const shifted = CreateCubeTriMesh();
  int const offset = cornerTouchingCubes.GetNumNodes();
  for (int node = 0; node < shifted.GetNumNodes(); ++node) {
    for (real const coordinate : NodePosition(shifted, node) + Real3{1_r, 1_r, 1_r}) {
      cornerTouchingCubes.coordinates.push_back(coordinate);
    }
  }
  for (int const node : shifted.connectivity) {
    cornerTouchingCubes.connectivity.push_back(node + offset);
  }

  MeshData const surfaces[] = {
      CreateCubeTriMeshWithHole(),
      OverlappingCubes(),
      CreateCubeTriMeshWithInconsistentWinding(),
      cubeWithUnusedVertex,
      cornerTouchingCubes,
  };
  for (MeshData const& surface : surfaces) {
    (void)TetrahedralizeConstrained(surface, ConstrainedTetrahedralizationParams{}, ExpectNotOK{});
  }
}

// Rejected by the client before it reaches the helper.
TEST(ConstrainedTetrahedralization, TetInputIsRejected) {
  MeshData const tets = TetrahedralizeCube(0.5_r);
  (void)TetrahedralizeConstrained(tets, ConstrainedTetrahedralizationParams{}, ExpectNotOK{});
}

TEST(ConstrainedTetrahedralization, InvalidParamsAreRejected) {
  using Invalidation = void (*)(ConstrainedTetrahedralizationParams&);
  Invalidation const invalidations[] = {
      [](ConstrainedTetrahedralizationParams& p) { p.edgeSize = 0.0; },
      [](ConstrainedTetrahedralizationParams& p) {
        p.edgeSize = std::numeric_limits<double>::infinity();
      },
      [](ConstrainedTetrahedralizationParams& p) { p.iterations = 0; },
  };
  for (Invalidation const invalidate : invalidations) {
    ConstrainedTetrahedralizationParams params;
    invalidate(params);
    (void)TetrahedralizeConstrained(CreateCubeTriMesh(), params, ExpectNotOK{});
  }
}
