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

#include "rendering/rod_visualization.h"

#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_physics/mochi_physics.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <optional>
#include <tuple>
#include <vector>

using namespace superdex::studio;
using mochi::ActorSegmentElementType;
using mochi::real;
using mochi::operator""_r;
using mochi::Real3;

namespace {

void ExpectNear(Real3 const& actual, Real3 const& expected, float tolerance = 1e-6f) {
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(actual[i], expected[i], tolerance) << "component " << i;
  }
}

TEST(RodVisualizationTest, AnEmptyConnectivityIsTheOpenPolylineThroughTheNodes) {
  std::vector<mochi::Int2> const expected = {{0, 1}, {1, 2}};
  EXPECT_EQ(PolylineElements(3, {}), expected);
}

TEST(RodVisualizationTest, ExplicitConnectivityIsKept) {
  std::vector<int> const connectivity = {0, 1, 1, 2, 2, 0};
  std::vector<mochi::Int2> const expected = {{0, 1}, {1, 2}, {2, 0}};
  EXPECT_EQ(PolylineElements(3, mochi::MakeConstSpan(connectivity)), expected);
}

TEST(RodVisualizationTest, GlyphsStartAtElementMidpointsAndSpanHalfTheMeanElementLength) {
  // Elements of length 1 and 3 (mean 2), so every glyph is 1 long whatever its element's length.
  std::vector<real> const nodes = {0, 0, 0, 1, 0, 0, 4, 0, 0};
  std::vector<real> const axes = {0, 1, 0, 0, 0, 1};

  std::vector<FrameAxisGlyph> const glyphs =
      ComputeFrameAxisGlyphs(mochi::MakeConstSpan(nodes), {}, mochi::MakeConstSpan(axes));

  ASSERT_EQ(glyphs.size(), 2u);
  ExpectNear(glyphs[0].origin, {0.5f, 0.0f, 0.0f});
  ExpectNear(glyphs[0].vector, {0.0f, 1.0f, 0.0f});
  ExpectNear(glyphs[1].origin, {2.5f, 0.0f, 0.0f});
  ExpectNear(glyphs[1].vector, {0.0f, 0.0f, 1.0f});
}

TEST(RodVisualizationTest, AClosedLoopHasAGlyphOnItsWrapElement) {
  std::vector<real> const nodes = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
  std::vector<int> const connectivity = {0, 1, 1, 2, 2, 3, 3, 0};
  std::vector<real> const axes = {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1};

  std::vector<FrameAxisGlyph> const glyphs = ComputeFrameAxisGlyphs(
      mochi::MakeConstSpan(nodes), mochi::MakeConstSpan(connectivity), mochi::MakeConstSpan(axes));

  ASSERT_EQ(glyphs.size(), 4u);
  ExpectNear(glyphs[3].origin, {0.0f, 0.5f, 0.0f});
  ExpectNear(glyphs[3].vector, {0.0f, 0.0f, 0.5f});
}

TEST(RodVisualizationTest, MismatchedFrameAxesGiveNoGlyphs) {
  std::vector<real> const nodes = {0, 0, 0, 1, 0, 0, 2, 0, 0};
  std::vector<real> const oneAxis = {0, 1, 0};

  EXPECT_TRUE(ComputeFrameAxisGlyphs(mochi::MakeConstSpan(nodes), {}, mochi::MakeConstSpan(oneAxis))
                  .empty());
}

TEST(RodVisualizationTest, ANodalColliderSamplesTheNodes) {
  std::vector<real> const nodes = {0, 0, 0, 1, 0, 0, 2, 0, 0};

  std::vector<Real3> const samples =
      ComputeRodColliderSamples(mochi::MakeConstSpan(nodes), {}, std::nullopt);

  std::vector<Real3> const expected = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}};
  EXPECT_EQ(samples, expected);
}

TEST(RodVisualizationTest, AOnePointRuleSamplesElementMidpointsIncludingAClosedLoopsWrap) {
  std::vector<real> const nodes = {0, 0, 0, 1, 0, 0, 1, 1, 0};
  std::vector<int> const connectivity = {0, 1, 1, 2, 2, 0};

  std::vector<Real3> const samples = ComputeRodColliderSamples(
      mochi::MakeConstSpan(nodes),
      mochi::MakeConstSpan(connectivity),
      ActorSegmentElementType::P1Q1);

  ASSERT_EQ(samples.size(), 3u);
  ExpectNear(samples[0], {0.5f, 0.0f, 0.0f});
  ExpectNear(samples[1], {1.0f, 0.5f, 0.0f});
  ExpectNear(samples[2], {0.5f, 0.5f, 0.0f});
}

TEST(RodVisualizationTest, TwoAndThreePointRulesSampleEachElementSymmetrically) {
  std::vector<real> const nodes = {0, 0, 0, 1, 0, 0};

  std::vector<Real3> const two =
      ComputeRodColliderSamples(mochi::MakeConstSpan(nodes), {}, ActorSegmentElementType::P1Q2);
  std::vector<Real3> const three =
      ComputeRodColliderSamples(mochi::MakeConstSpan(nodes), {}, ActorSegmentElementType::P1Q3);

  ASSERT_EQ(two.size(), 2u);
  EXPECT_NEAR(two[0][0] + two[1][0], 1.0f, 1e-6f);
  ASSERT_EQ(three.size(), 3u);
  ExpectNear(three[1], {0.5f, 0.0f, 0.0f});
  EXPECT_NEAR(three[0][0] + three[2][0], 1.0f, 1e-6f);
}

TEST(RodVisualizationTest, TheSphereCloudHasEveryVertexOnItsSphere) {
  std::vector<Real3> const centers = {{0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 3.0f}};
  constexpr real kRadius = 0.25f;

  SphereCloudMesh const mesh = BuildSphereCloud(mochi::MakeConstSpan(centers), kRadius);

  int const verticesPerSphere = static_cast<int>(mesh.positions.size() / 3 / centers.size());
  ASSERT_GT(verticesPerSphere, 0);
  for (int v = 0; v < static_cast<int>(mesh.positions.size() / 3); ++v) {
    Real3 const position{
        mesh.positions[3 * v], mesh.positions[3 * v + 1], mesh.positions[3 * v + 2]};
    EXPECT_NEAR(mochi::Norm(position - centers[v / verticesPerSphere]), kRadius, 1e-5f) << v;
  }
  auto movedCenters = centers;
  for (auto& center : movedCenters) {
    center += Real3{2_r, -1_r, 0.5_r};
  }
  auto const moved = BuildSphereCloudPositions(mochi::MakeConstSpan(movedCenters), 2_r * kRadius);
  ASSERT_EQ(moved.size(), mesh.positions.size());
  for (int v = 0; v < static_cast<int>(moved.size() / 3); ++v) {
    Real3 const position{moved[3 * v], moved[3 * v + 1], moved[3 * v + 2]};
    EXPECT_NEAR(mochi::Norm(position - movedCenters[v / verticesPerSphere]), 2_r * kRadius, 1e-5_r);
  }
  for (int const index : mesh.indices) {
    EXPECT_TRUE(index >= 0 && index < static_cast<int>(mesh.positions.size() / 3)) << index;
  }
}

// Mochi's "Point Cloud Collider" debug draw puts a sphere at each of a rod's collider samples, so
// the helper must place the same samples for every collider rule, on a deformed open or closed rod.
class RodColliderSamplesTest
    : public ::testing::TestWithParam<std::tuple<std::optional<ActorSegmentElementType>, bool>> {};

TEST_P(RodColliderSamplesTest, MatchMochisPointCloudColliderDebugDraw) {
  auto const [elementType, closedLoop] = GetParam();
  mochi::Context* const context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  mochi::Scene* const scene = context->CreateScene("rod_collider_samples_test");
  MOCHI_DEFER(context->DestroyScene(scene));
  mochi::ModelData model;
  model.mesh = mochi::MeshData{};
  model.mesh->nodesPerElement = 2;
  // A square in the xy plane, closed or not; frame axes along z stay orthogonal to every element.
  model.mesh->coordinates = {0, 0, 0, 0.1f, 0, 0, 0.1f, 0.1f, 0, 0, 0.1f, 0};
  if (closedLoop) {
    model.mesh->connectivity = {0, 1, 1, 2, 2, 3, 3, 0};
  }
  model.elementFrameAxes = closedLoop
      ? mochi::DynamicArray<real>{0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1}
      : mochi::DynamicArray<real>{0, 0, 1, 0, 0, 1, 0, 0, 1};
  mochi::ErrorAssert error;
  mochi::RodActorParams params;
  params.shape = context->CreateModelShape(model, error);
  params.hasGravity = false;
  params.material.linearDensity = 0.3f;
  params.material.linearRotationalInertia = 1e-5f;
  params.material.axialStiffness = 3000.0f;
  params.material.torsionalStiffness = 0.05f;
  params.material.flexuralStiffness = {0.08f, 0.08f};
  params.colliderType = mochi::ColliderType::PointCloud;
  params.pointCloudCollider.radius = 0.01f;
  params.pointCloudCollider.colliderSegmentElementType = elementType;
  mochi::Actor* const rod = scene->CreateRodActor(params, error);
  // Deform the rod in its plane, so the frame axes stay orthogonal to it.
  std::vector<real> const displacements = {0, 0, 0, 0.01f, 0.01f, 0, 0, 0.03f, 0, -0.02f, 0, 0};
  rod->SetRodPose(
      mochi::MakeConstSpan(displacements), mochi::MakeConstSpan(*model.elementFrameAxes), error);
  std::vector<real> deformed(model.mesh->coordinates.begin(), model.mesh->coordinates.end());
  for (size_t i = 0; i < deformed.size(); ++i) {
    deformed[i] += displacements[i];
  }

  mochi::DebugDraw& debugDraw = scene->GetDebugDraw();
  debugDraw.Enable(true);
  int const feature = debugDraw.FindFeature("Point Cloud Collider");
  ASSERT_GE(feature, 0);
  debugDraw.EnableFeature(feature, true);
  mochi::DebugDrawData const drawn = debugDraw.GatherData();
  std::vector<Real3> const samples = ComputeRodColliderSamples(
      mochi::MakeConstSpan(deformed), mochi::MakeConstSpan(model.mesh->connectivity), elementType);

  ASSERT_EQ(drawn.spheres.positions.size(), samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    ExpectNear(drawn.spheres.positions[i], samples[i]);
    EXPECT_NEAR(drawn.spheres.radii[i], 0.01f, 1e-7f);
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllColliderRules,
    RodColliderSamplesTest,
    ::testing::Combine(
        ::testing::Values(
            std::nullopt,
            ActorSegmentElementType::P1Q1,
            ActorSegmentElementType::P1Q2,
            ActorSegmentElementType::P1Q3),
        ::testing::Bool()));

TEST(RodVisualizationTest, AGrabClusterIsTheNearestNodeThenItsNeighborsWithinTheRadius) {
  // Ten nodes 0.1 apart: the diagonal is 0.9, so a 0.25 fraction reaches 0.225 from the grab.
  std::vector<Real3> nodes;
  nodes.reserve(10);
  for (int i = 0; i < 10; ++i) {
    nodes.emplace_back(0.1f * static_cast<real>(i), 0.0f, 0.0f);
  }

  std::vector<int> const cluster =
      SelectGrabCluster(mochi::MakeConstSpan(nodes), {0.41f, 0.0f, 0.0f}, 0.25f, 10);

  EXPECT_EQ(cluster, (std::vector<int>{4, 5, 3, 6, 2}));
}

TEST(RodVisualizationTest, AGrabClusterKeepsTheAnchorWhenCappedOrTheRadiusIsTooSmall) {
  std::vector<Real3> const nodes = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}};

  EXPECT_EQ(
      SelectGrabCluster(mochi::MakeConstSpan(nodes), {1.9f, 0.0f, 0.0f}, 0.0f, 10),
      (std::vector<int>{2}));
  EXPECT_EQ(
      SelectGrabCluster(mochi::MakeConstSpan(nodes), {1.1f, 0.0f, 0.0f}, 1.0f, 2),
      (std::vector<int>{1, 2}));
  EXPECT_TRUE(SelectGrabCluster({}, {0.0f, 0.0f, 0.0f}, 1.0f, 10).empty());
}

constexpr mochi::Float2 kViewportSize{200.0f, 200.0f};

ClipSegment ProjectSegment(Real3 const& a, Real3 const& b, bool perspective) {
  // Camera at the origin looking along -z, with near=1, far=20 and unit x/y projection scale.
  auto project = [perspective](Real3 const& p) -> mochi::Float4 {
    if (perspective) {
      return {p[0], p[1], -21.0f / 19.0f * p[2] - 40.0f / 19.0f, -p[2]};
    }
    return {p[0], p[1], -2.0f / 19.0f * p[2] - 21.0f / 19.0f, 1.0f};
  };
  return {project(a), project(b)};
}

ClipSegment SegmentAtPixels(mochi::Float2 a, mochi::Float2 b, float depthA, float depthB) {
  a = a / kViewportSize * 2.0f - 1.0f;
  b = b / kViewportSize * 2.0f - 1.0f;
  return ProjectSegment({a[0], a[1], -depthA}, {b[0], b[1], -depthB}, false);
}

TEST(RodVisualizationTest, AScreenSegmentIsPickedWithinTheRadiusAtTheNearestPoint) {
  std::vector<ClipSegment> const segments = {
      SegmentAtPixels({0.0f, 0.0f}, {100.0f, 0.0f}, 1, 1),
      SegmentAtPixels({0.0f, 50.0f}, {100.0f, 50.0f}, 1, 1)};

  auto const hit =
      PickNearestScreenSegment(mochi::MakeConstSpan(segments), {25.0f, 46.0f}, kViewportSize, 8);

  ASSERT_TRUE(hit.has_value());
  EXPECT_EQ(hit->segment, 1);
  EXPECT_NEAR(hit->t, 0.25f, 1e-6f);
  EXPECT_NEAR(hit->distancePx, 4.0f, 1e-5f);
  EXPECT_FALSE(
      PickNearestScreenSegment(mochi::MakeConstSpan(segments), {25.0f, 25.0f}, kViewportSize, 8)
          .has_value());
  EXPECT_FALSE(
      PickNearestScreenSegment(mochi::MakeConstSpan(segments), {-20.0f, 0.0f}, kViewportSize, 8)
          .has_value());
  EXPECT_FALSE(PickNearestScreenSegment({}, {25.0f, 46.0f}, kViewportSize, 8).has_value());
}

TEST(RodVisualizationTest, PerspectivePickingReturnsTheCorrectWorldPointAndNearestNode) {
  struct Case {
    float nearDepth;
    Real3 point;
  };
  std::array<Case, 2> const cases = {
      Case{1.0f, {3.0f / 23.0f, 0.0f, -50.0f / 23.0f}},
      Case{0.5f, {3.0f / 43.0f, 0.0f, -50.0f / 43.0f}}};
  for (Case const& testCase : cases) {
    for (bool reverse : {false, true}) {
      std::array<Real3, 2> nodes = {
          Real3{0.0f, 0.0f, -testCase.nearDepth}, Real3{1.0f, 0.0f, -10.0f}};
      if (reverse) {
        std::reverse(nodes.begin(), nodes.end());
      }
      std::array<ClipSegment, 1> const segments = {ProjectSegment(nodes[0], nodes[1], true)};

      auto const hit = PickNearestScreenSegment(
          mochi::MakeConstSpan(segments), {106.0f, 100.0f}, kViewportSize, 8);

      ASSERT_TRUE(hit.has_value());
      Real3 const point = nodes[0] + (nodes[1] - nodes[0]) * hit->t;
      ExpectNear(point, testCase.point);
      EXPECT_EQ(
          SelectGrabCluster(mochi::MakeConstSpan(nodes), point, 0.0f, 1),
          (std::vector<int>{reverse ? 1 : 0}));
    }
  }
}

TEST(RodVisualizationTest, PerspectiveCrossingsUseDepthAtTheProjectedHit) {
  std::array<ClipSegment, 2> segments = {
      ProjectSegment({0.0f, 0.0f, -1.0f}, {1.0f, 0.0f, -10.0f}, true),
      ProjectSegment({0.0f, 0.0f, -3.0f}, {0.3f, 0.0f, -3.0f}, true)};
  for (int order = 0; order < 2; ++order) {
    auto const hit = PickNearestScreenSegment(
        mochi::MakeConstSpan(segments), {106.0f, 100.0f}, kViewportSize, 8);

    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->segment, order);
    std::reverse(segments.begin(), segments.end());
  }
}

TEST(RodVisualizationTest, OrthographicOverlapsUseViewDepth) {
  std::array<ClipSegment, 2> segments = {
      ProjectSegment({-0.5f, 0.0f, -5.0f}, {0.5f, 0.0f, -5.0f}, false),
      ProjectSegment({-0.5f, 0.0f, -2.0f}, {0.5f, 0.0f, -2.0f}, false)};
  for (int order = 0; order < 2; ++order) {
    auto const hit = PickNearestScreenSegment(
        mochi::MakeConstSpan(segments), {100.0f, 100.0f}, kViewportSize, 8);

    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->segment, 1 - order);
    std::reverse(segments.begin(), segments.end());
  }
}

TEST(RodVisualizationTest, SegmentsBehindTheNearPlaneAreNotPickable) {
  std::array<Real3, 2> const positions = {Real3{0.0f, 0.0f, 2.0f}, {0.0f, 0.0f, -0.5f}};
  for (bool perspective : {false, true}) {
    for (Real3 const& position : positions) {
      std::array<ClipSegment, 1> const segments = {
          ProjectSegment(position, position + Real3{0.1f, 0.0f, 0.0f}, perspective)};

      EXPECT_FALSE(PickNearestScreenSegment(
                       mochi::MakeConstSpan(segments), {100.0f, 100.0f}, kViewportSize, 8)
                       .has_value())
          << "perspective=" << perspective << ", depth=" << position[2];
    }
  }
}

TEST(RodVisualizationTest, NearPlaneClippingPreservesTheOriginalWorldParameter) {
  for (bool perspective : {false, true}) {
    for (bool reverse : {false, true}) {
      std::array<Real3, 2> nodes = {Real3{0.0f, 0.0f, 2.0f}, Real3{0.0f, 0.0f, -30.0f}};
      if (reverse) {
        std::reverse(nodes.begin(), nodes.end());
      }
      std::array<ClipSegment, 1> const segments = {ProjectSegment(nodes[0], nodes[1], perspective)};

      auto const hit = PickNearestScreenSegment(
          mochi::MakeConstSpan(segments), {100.0f, 100.0f}, kViewportSize, 8);

      ASSERT_TRUE(hit.has_value());
      ExpectNear(nodes[0] + (nodes[1] - nodes[0]) * hit->t, {0.0f, 0.0f, -1.0f}, 4e-6f);
    }
  }
}

TEST(RodVisualizationTest, TheSmallestScreenDistanceWinsRegardlessOfDepthAndOrder) {
  std::vector<ClipSegment> segments;
  for (int i = 0; i < 9; ++i) {
    float const offset = 0.9f * static_cast<float>(i);
    float const depth = 9.0f - static_cast<float>(i);
    segments.push_back(
        SegmentAtPixels({50.0f, 100.0f + offset}, {150.0f, 100.0f + offset}, depth, depth));
  }
  ClipSegment const expected = segments[0];
  for (int order = 0; order < 2; ++order) {
    auto const hit = PickNearestScreenSegment(
        mochi::MakeConstSpan(segments), {100.0f, 100.0f}, kViewportSize, 8);

    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(segments[hit->segment].a, expected.a);
    EXPECT_NEAR(hit->distancePx, 0.0f, 1e-4f);
    std::reverse(segments.begin(), segments.end());
  }
}

TEST(RodVisualizationTest, ThePickAllowanceExtendsPastViewportEdges) {
  std::array<ClipSegment, 1> const segments = {
      SegmentAtPixels({-4.0f, 50.0f}, {-4.0f, 150.0f}, 2, 2)};

  auto const hit =
      PickNearestScreenSegment(mochi::MakeConstSpan(segments), {2.0f, 100.0f}, kViewportSize, 8);

  ASSERT_TRUE(hit.has_value());
  EXPECT_NEAR(hit->distancePx, 6.0f, 1e-5f);
}

} // namespace
