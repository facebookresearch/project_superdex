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

#include <mochi_core/element_operations/fem_rod.h>
#include <mochi_core/geometry/mesh_data_utils.h>
#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/vmatrix.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace mochi;

namespace {

class GenerateDiscreteBishopFrameTest : public ::testing::Test {
 protected:
  static real constexpr kTolerance = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 1e-5_r;

  // Generate a random polyline with the given number of nodes using the provided RNG
  static DynamicArray<Real3> GenerateRandomPolyline(
      int numNodes,
      mochi_default_random_engine& rng,
      real minEdgeLength = 0.1_r,
      real maxEdgeLength = 1.0_r,
      real maxBendAngle = 0.9_r * kPI) {
    MOCHI_ASSERT(numNodes >= 2, "Need at least 2 nodes");

    DynamicArray<Real3> nodes;
    nodes.reserve(numNodes);

    // Start at the origin
    nodes.push_back(Real3{0_r, 0_r, 0_r});

    // Initial direction (along x-axis)
    Real3 direction = {1_r, 0_r, 0_r};

    for (int i = 1; i < numNodes; ++i) {
      // Random edge length
      real const edgeLength = RandomUniformValue(rng, minEdgeLength, maxEdgeLength);

      // Add node in current direction
      nodes.push_back(nodes[i - 1] + direction * edgeLength);

      // Rotate direction for next edge (if not the last node)
      if (i < numNodes - 1) {
        // Generate random rotation axis perpendicular to current direction
        Real3 randomVec;
        SetRandom(rng, -1_r, 1_r, randomVec);
        Real3 rotAxis = Cross(direction, randomVec);

        // Handle case where randomVec is parallel to direction
        if (Norm(rotAxis) < 1e-6_r) {
          rotAxis = {0_r, 1_r, 0_r};
          if (Abs(Dot(direction, rotAxis)) > 0.9_r) {
            rotAxis = {0_r, 0_r, 1_r};
          }
          rotAxis = Cross(direction, rotAxis);
        }
        rotAxis = Normalize(rotAxis);

        // Random bend angle
        real const bendAngle = RandomUniformValue(rng, -maxBendAngle, maxBendAngle);

        // Apply rotation using quaternion multiplication
        Quaternion const rotation = Quaternion::FromAxisAngle(rotAxis, bendAngle);
        direction = rotation * direction;
        direction = Normalize(direction);
      }
    }

    return nodes;
  }

  // Compute the angle between two vectors (in radians)
  static real AngleBetween(Real3 const& a, Real3 const& b) {
    real const cosAngle = std::clamp(Dot(Normalize(a), Normalize(b)), -1_r, 1_r);
    return std::acos(cosAngle);
  }
};

real constexpr kTubeTolerance = MOCHI_USE_DOUBLE_PRECISION ? 1e-10_r : 1e-5_r;

// Nodes on a helical arc with a circular XY projection of radius 1.
DynamicArray<Real3> ArcNodes(int numNodes, real arcAngle) {
  DynamicArray<Real3> nodes;
  for (int i = 0; i < numNodes; ++i) {
    real const angle = arcAngle * static_cast<real>(i) / static_cast<real>(numNodes);
    nodes.push_back(Real3{Cos(angle), Sin(angle), 0.1_r * angle});
  }
  return nodes;
}

// Checks that @p tube surrounds @p nodes as GenerateTubularSkinMesh documents: a ring of
// numSegments nodes at distance radius around each element midpoint (and each end node of an
// open polyline), orthogonal to the ring's element and skinned to it, plus cap nodes at the end
// nodes, and that it is valid as both the contact skin and the visual mesh of the polyline.
void ExpectTubeAroundPolyline(
    MeshData const& tube,
    Span<Real3 const> nodes,
    Span<Real3 const> axes,
    bool isClosedLoop,
    real radius,
    int numSegments) {
  int const numNodes = isize(nodes);
  int const numElements = isClosedLoop ? numNodes : numNodes - 1;
  int const numRings = isClosedLoop ? numElements : numElements + 2;
  int const numCapNodes = isClosedLoop ? 0 : 2;
  int const numStrips = isClosedLoop ? numRings : numRings - 1;
  ASSERT_EQ(tube.nodesPerElement, 3);
  ASSERT_EQ(tube.GetNumNodes(), numSegments * numRings + numCapNodes);
  EXPECT_EQ(tube.GetNumElements(), 2 * numSegments * numStrips + numCapNodes * numSegments);
  ASSERT_TRUE(tube.skinning.has_value());
  ASSERT_EQ(tube.skinning->weightsPerNode, 1);
  EXPECT_TRUE(std::ranges::all_of(tube.skinning->weights, [](real w) { return w == 1_r; }));

  auto const positions = Unflatten<Real3 const>(MakeConstSpan(tube.coordinates));
  for (int r = 0; r < numRings; ++r) {
    int const element = isClosedLoop ? r : std::clamp(r - 1, 0, numElements - 1);
    Real3 const x0 = nodes[element];
    Real3 const x1 = nodes[(element + 1) % numNodes];
    Real3 const center = (!isClosedLoop && r == 0) ? nodes[0]
        : (!isClosedLoop && r == numRings - 1)     ? nodes[numNodes - 1]
                                                   : 0.5_r * (x0 + x1);
    Real3 const tangent = Normalize(x1 - x0);
    EXPECT_NEAR_EQ(positions[r * numSegments], center + radius * axes[element]);
    for (int s = 0; s < numSegments; ++s) {
      int const node = r * numSegments + s;
      Real3 const offset = positions[node] - center;
      EXPECT_NEAR(Norm(offset), radius, kTubeTolerance) << "ring " << r << " node " << s;
      EXPECT_NEAR(Dot(offset, tangent), 0_r, kTubeTolerance) << "ring " << r << " node " << s;
      EXPECT_EQ(tube.skinning->indices[node], element) << "ring " << r << " node " << s;
    }
  }
  if (!isClosedLoop) {
    int const startCap = numSegments * numRings;
    EXPECT_NEAR_EQ(positions[startCap], nodes[0]);
    EXPECT_NEAR_EQ(positions[startCap + 1], nodes[numNodes - 1]);
    EXPECT_EQ(tube.skinning->indices[startCap], 0);
    EXPECT_EQ(tube.skinning->indices[startCap + 1], numElements - 1);
  }

  ModelData model;
  model.mesh.emplace();
  model.mesh->nodesPerElement = 2;
  model.mesh->coordinates = Flatten(nodes);
  model.mesh->connectivity = MakeSequentialPolylineConnectivity(numNodes, isClosedLoop);
  model.elementFrameAxes = Flatten(axes);
  model.contactSkinMesh = tube;
  model.visualMesh = tube;
  model::Validate(model, test::ExpectOK{});
}

} // namespace

TEST_F(GenerateDiscreteBishopFrameTest, RandomPolylinesValidityAndMinimalRotation) {
  // Use a fixed seed for reproducibility
  mochi_default_random_engine rng = RandomGenerator(42);

  // Test various polyline sizes
  DynamicArray<int> nodeCounts = {2, 3, 5, 10, 20, 50};
  int constexpr kNumPolylinesPerSize = 10;

  for (int numNodes : nodeCounts) {
    for (int polylineIdx = 0; polylineIdx < kNumPolylinesPerSize; ++polylineIdx) {
      // Generate a random polyline
      DynamicArray<Real3> nodes = GenerateRandomPolyline(numNodes, rng);
      int const numElements = numNodes - 1;

      // Generate the Bishop frame
      DynamicArray<Real3> axes = model::GenerateDiscreteBishopFrame(nodes, /*isClosedLoop=*/false);

      // Check that we got the expected number of axes
      ASSERT_EQ(isize(axes), numElements)
          << "Wrong number of axes for polyline with " << numNodes << " nodes";

      // Compute tangent vectors for each element
      DynamicArray<Real3> tangents;
      tangents.reserve(numElements);
      for (int i = 0; i < numElements; ++i) {
        tangents.push_back(Normalize(nodes[i + 1] - nodes[i]));
      }

      // Validity check 1: Each axis must be a unit vector
      for (int i = 0; i < numElements; ++i) {
        real const axisNorm = Norm(axes[i]);
        EXPECT_NEAR(1_r, axisNorm, kTolerance)
            << "Axis " << i << " is not a unit vector (norm = " << axisNorm << ")";
      }

      // Validity check 2: Each axis must be orthogonal to its element's tangent
      for (int i = 0; i < numElements; ++i) {
        real const dotProduct = Abs(Dot(axes[i], tangents[i]));
        EXPECT_LT(dotProduct, kTolerance)
            << "Axis " << i << " is not orthogonal to tangent (dot = " << dotProduct << ")";
      }

      // Validity check 3: No 180-degree jumps between consecutive axes
      for (int i = 0; i < numElements - 1; ++i) {
        real const dotProduct = Dot(axes[i], axes[i + 1]);
        EXPECT_GT(dotProduct, kTolerance - 1_r)
            << "Axes " << i << " and " << i + 1 << " have a 180-degree jump";
      }

      // Bishop frame property: angle between consecutive axes <= angle between consecutive
      // tangents, which should always be true for minimal rotation transport.
      for (int i = 0; i < numElements - 1; ++i) {
        real const axisAngle = AngleBetween(axes[i], axes[i + 1]);
        real const tangentAngle = AngleBetween(tangents[i], tangents[i + 1]);

        // Allow small tolerance for numerical errors
        EXPECT_LE(axisAngle, tangentAngle + kTolerance)
            << "Axis rotation (" << axisAngle << " rad) exceeds tangent rotation (" << tangentAngle
            << " rad) between elements " << i << " and " << i + 1;
      }
    }
  }
}

TEST(GenerateDiscreteBishopFrameClosedLoop, NonzeroHolonomy) {
  // Test discrete Bishop frame for a closed-loop curve with nonzero holonomy. The resulting frame
  // axes won't strictly be a Bishop frame (parallel transport), because the holonomy correction
  // deviates from parallel transport. However, the axes should vary continuously around the loop,
  // with adjacent elements having similar orientations that converge under refinement.

  // Helper to generate a torus knot curve
  auto generateTorusKnot = [](int numNodes, real scale) {
    DynamicArray<Real3> nodes;
    nodes.reserve(numNodes);
    for (int i = 0; i < numNodes; ++i) {
      real const theta = 2_r * kPI * static_cast<real>(i) / static_cast<real>(numNodes);
      real const pTheta = 2_r * theta;
      real const qTheta = 3_r * theta;
      real const twoPlusCosPTheta = (2_r + Cos(pTheta));
      nodes.push_back(
          scale *
          Real3{twoPlusCosPTheta * Cos(qTheta), twoPlusCosPTheta * Sin(qTheta), Sin(pTheta)});
    }
    return nodes;
  };

  real constexpr kCurveScale = 0.1_r;
  real constexpr kTolerance = 1e-4_r;

  // Test with multiple resolutions (powers of two) to verify convergence under refinement
  int constexpr kMinPower = 5; // 2^5 = 32 nodes
  int constexpr kMaxPower = 7; // 2^7 = 128 nodes

  real prevMaxAngle = 0_r;
  for (int power = kMinPower; power <= kMaxPower; ++power) {
    int const numNodes = 1 << power;
    DynamicArray<Real3> nodes = generateTorusKnot(numNodes, kCurveScale);
    auto const axes =
        model::GenerateDiscreteBishopFrame(MakeConstSpan(nodes), /*isClosedLoop=*/true);

    EXPECT_EQ(isize(axes), numNodes) << "Wrong number of axes for " << numNodes << " nodes";

    // Basic validity checks: unit length and orthogonality to tangents
    for (int i = 0; i < numNodes; ++i) {
      // Unit length
      EXPECT_NEAR(Norm(axes[i]), 1_r, kTolerance)
          << "Frame axis " << i << " should be unit-length for " << numNodes << " nodes";

      // Perpendicular to element tangent
      Real3 const tangent = Normalize(nodes[(i + 1) % numNodes] - nodes[i]);
      EXPECT_NEAR(Abs(Dot(axes[i], tangent)), 0_r, kTolerance)
          << "Frame axis " << i << " should be perpendicular to tangent for " << numNodes
          << " nodes";
    }

    // Check continuity: adjacent axes should be close (no large jumps)
    // The maximum angle between consecutive axes should decrease with refinement
    real maxAngleBetweenConsecutive = 0_r;
    for (int i = 0; i < numNodes; ++i) {
      int const j = (i + 1) % numNodes;
      real const dot = Clamp(Dot(axes[i], axes[j]), -1_r, 1_r);
      real const angle = ACos(dot);
      maxAngleBetweenConsecutive = Max(maxAngleBetweenConsecutive, angle);

      // No 180-degree flips between consecutive axes
      EXPECT_GT(dot, kTolerance - 1_r)
          << "Axes " << i << " and " << j << " have a 180-degree jump for " << numNodes << " nodes";
    }

    // With holonomy correction, the frame should be continuous across the closing edge.
    // The maximum angle between consecutive elements should scale roughly with 1/numNodes
    // (since the curve is smooth and we're sampling more densely).
    // For this torus knot, the maximum angle should be reasonable even at coarse resolution.
    // Use a generous bound since the torus knot has regions of higher curvature.
    real const expectedMaxAngle = 2_r * kPI / static_cast<real>(numNodes) * 5_r; // generous bound
    EXPECT_LT(maxAngleBetweenConsecutive, expectedMaxAngle)
        << "Max angle between consecutive axes (" << maxAngleBetweenConsecutive
        << " rad) exceeds expected bound for " << numNodes << " nodes";

    // Verify that refinement reduces the maximum angle between consecutive axes
    // (axes of adjacent elements should approach each other under refinement)
    if (power > kMinPower) {
      // The angle should roughly halve when doubling resolution (for smooth curves)
      // We use a generous factor of 0.55 to account for preasymptotic behavior.
      EXPECT_LT(maxAngleBetweenConsecutive, prevMaxAngle * 0.55_r)
          << "Max angle should decrease with refinement: " << prevMaxAngle << " (2^" << (power - 1)
          << " nodes) vs " << maxAngleBetweenConsecutive << " (2^" << power << " nodes)";
    }
    prevMaxAngle = maxAngleBetweenConsecutive;

    // Check continuity across the closing edge: the frame should wrap around smoothly
    // (no discontinuity from last element back to first)
    Vec4r const lastTangent = ToSimd(Normalize(nodes[0] - nodes[numNodes - 1]));
    Vec4r const firstTangent = ToSimd(Normalize(nodes[1] - nodes[0]));
    VMatrix3x3r const P = fem::ParallelTransportOperator(lastTangent, firstTangent);
    Vec4r const transported = DotMatVec3x3(P, ToSimd(axes[numNodes - 1]));
    Real3 const diff = axes[0] - ToReal3(transported);
    // With holonomy correction, the transported last axis should be close to the first axis.
    // The error should decrease with refinement as the discretization improves.
    // Scale threshold by 1/sqrt(numNodes) since error should decrease with refinement.
    real const continuityThreshold = 0.1_r / static_cast<real>(numNodes);
    EXPECT_LT(Norm(diff), continuityThreshold)
        << "Frame should be continuous across closing edge for " << numNodes
        << " nodes (diff norm: " << Norm(diff) << ")";
  }
}

TEST(GenerateTubularSkinMesh, OpenPolyline) {
  real constexpr kRadius = 0.05_r;
  int constexpr kNumSegments = 6;
  DynamicArray<Real3> const nodes = ArcNodes(5, 0.5_r * kPI);
  model::ValidateTubularSkinMeshParameters(
      nodes, /*isClosedLoop=*/false, kRadius, kNumSegments, test::ExpectOK{});
  DynamicArray<Real3> axes = model::GenerateDiscreteBishopFrame(nodes, /*isClosedLoop=*/false);
  for (int i = 0; i < isize(axes); ++i) {
    axes[i] = Cross(Normalize(nodes[i + 1] - nodes[i]), axes[i]);
  }
  model::ValidatePolylineElementFrameAxes(axes, nodes, false, test::ExpectOK{});
  MeshData const tube =
      model::GenerateTubularSkinMesh(nodes, axes, /*isClosedLoop=*/false, kRadius, kNumSegments);
  ExpectTubeAroundPolyline(tube, nodes, axes, /*isClosedLoop=*/false, kRadius, kNumSegments);
}

TEST(GenerateTubularSkinMesh, SingleElement) {
  real constexpr kRadius = 0.05_r;
  int constexpr kNumSegments = 3;
  DynamicArray<Real3> const nodes = {Real3{0_r, 0_r, 0_r}, Real3{0_r, 0_r, 1_r}};
  DynamicArray<Real3> const axes = {Real3{1_r, 0_r, 0_r}};
  MeshData const tube = model::GenerateTubularSkinMesh(
      nodes, axes, /*isClosedLoop=*/false, kRadius, kNumSegments, test::ExpectOK{});
  ExpectTubeAroundPolyline(tube, nodes, axes, /*isClosedLoop=*/false, kRadius, kNumSegments);
}

TEST(GenerateTubularSkinMesh, ClosedLoop) {
  real constexpr kRadius = 0.05_r;
  int constexpr kNumSegments = 8;
  DynamicArray<Real3> nodes = ArcNodes(7, 2_r * kPI);
  for (Real3& node : nodes) {
    node[2] = 0_r; // planar, so the loop closes
  }
  DynamicArray<Real3> const axes(isize(nodes), Real3{0_r, 0_r, -1_r});
  MeshData const tube = model::GenerateTubularSkinMesh(
      nodes, axes, /*isClosedLoop=*/true, kRadius, kNumSegments, test::ExpectOK{});
  ExpectTubeAroundPolyline(tube, nodes, axes, /*isClosedLoop=*/true, kRadius, kNumSegments);
}

TEST(GenerateTubularSkinMesh, InvalidInputsError) {
  DynamicArray<Real3> const nodes = ArcNodes(4, 0.5_r * kPI);
  DynamicArray<Real3> const axes = model::GenerateDiscreteBishopFrame(nodes, false);
  real const nan = std::numeric_limits<real>::quiet_NaN();
  for (real const radius : {0_r, -1_r, std::numeric_limits<real>::infinity(), nan}) {
    EXPECT_TRUE(
        model::GenerateTubularSkinMesh(nodes, axes, false, radius, 8, test::ExpectNotOK{})
            .coordinates.empty())
        << "radius " << radius;
  }
  EXPECT_TRUE(
      model::GenerateTubularSkinMesh(nodes, axes, false, 0.1_r, 2, test::ExpectNotOK{})
          .coordinates.empty());
  DynamicArray<Real3> const coincident = {nodes[0], nodes[0], nodes[1]};
  EXPECT_TRUE(
      model::GenerateTubularSkinMesh(coincident, axes, false, 0.1_r, 8, test::ExpectNotOK{})
          .coordinates.empty());
  DynamicArray<Real3> const twoNodeLoop = {nodes[0], nodes[1]};
  EXPECT_TRUE(
      model::GenerateTubularSkinMesh(twoNodeLoop, axes, true, 0.1_r, 8, test::ExpectNotOK{})
          .coordinates.empty());
  DynamicArray<Real3> nonFiniteNodes = nodes;
  nonFiniteNodes[1][0] = nan;
  EXPECT_TRUE(
      model::GenerateTubularSkinMesh(nonFiniteNodes, axes, false, 0.1_r, 8, test::ExpectNotOK{})
          .coordinates.empty());
  EXPECT_TRUE(
      model::GenerateTubularSkinMesh(nodes, {}, false, 0.1_r, 8, test::ExpectNotOK{})
          .coordinates.empty());
  DynamicArray<Real3> const nonUnitAxes(isize(axes), Real3{0_r, 0_r, 2_r});
  EXPECT_TRUE(
      model::GenerateTubularSkinMesh(nodes, nonUnitAxes, false, 0.1_r, 8, test::ExpectNotOK{})
          .coordinates.empty());
}

TEST(ValidatePolylineGeometry, NonFiniteNodesError) {
  DynamicArray<Real3> const validNodes = {
      Real3{0_r, 0_r, 0_r}, Real3{1_r, 0_r, 0_r}, Real3{1_r, 1_r, 0_r}};
  real const infinity = std::numeric_limits<real>::infinity();
  for (real const value : {std::numeric_limits<real>::quiet_NaN(), infinity, -infinity}) {
    for (bool const isClosedLoop : {false, true}) {
      for (int node = 0; node < isize(validNodes); ++node) {
        DynamicArray<Real3> nodes = validNodes;
        nodes[node][0] = value;
        model::ValidatePolylineGeometry(nodes, isClosedLoop, test::ExpectNotOK{});
      }
    }
  }
}

TEST(ValidatePolylineGeometry, UnrepresentableEdgeLengthsError) {
  real const max = std::numeric_limits<real>::max();
  DynamicArray<Real3> const overflowedEdge = {Real3{-max, 0_r, 0_r}, Real3{max, 0_r, 0_r}};
  model::ValidatePolylineGeometry(overflowedEdge, false, test::ExpectNotOK{});
  DynamicArray<Real3> const overflowedNorm = {
      Real3{0_r, 0_r, 0_r}, Real3{2_r * std::sqrt(max), 0_r, 0_r}};
  model::ValidatePolylineGeometry(overflowedNorm, false, test::ExpectNotOK{});
}

TEST(GenerateTubularSkinMesh, OversizedMeshError) {
  DynamicArray<Real3> const nodes = {Real3{0_r, 0_r, 0_r}, Real3{1_r, 0_r, 0_r}};
  for (bool const isClosedLoop : {false, true}) {
    DynamicArray<Real3> const polyline = isClosedLoop ? ArcNodes(4, 2_r * kPI) : nodes;
    DynamicArray<Real3> const axes = model::GenerateDiscreteBishopFrame(polyline, isClosedLoop);
    EXPECT_TRUE(
        model::GenerateTubularSkinMesh(
            polyline,
            axes,
            isClosedLoop,
            0.1_r,
            std::numeric_limits<int>::max(),
            test::ExpectNotOK{})
            .coordinates.empty());
  }
}

TEST(GenerateTubularSkinMesh, NonFiniteGeneratedCoordinatesError) {
  real const max = std::numeric_limits<real>::max();
  for (real const radius : {0.1_r, max}) {
    real const x = radius == max ? 0.5_r * max : max;
    DynamicArray<Real3> const nodes = {Real3{x, 0_r, 0_r}, Real3{x, 0_r, 1_r}};
    DynamicArray<Real3> const axes = {Real3{1_r, 0_r, 0_r}};
    EXPECT_TRUE(
        model::GenerateTubularSkinMesh(nodes, axes, false, radius, 4, test::ExpectNotOK{})
            .coordinates.empty());
  }
}
