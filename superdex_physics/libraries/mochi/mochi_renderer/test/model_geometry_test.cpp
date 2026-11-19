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

#include <mochi_renderer/model_geometry.h>

#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/coordinate_space_converter.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

struct Geometry {
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;
};

bool Build(
    mochi::ModelData const& model,
    Geometry& out,
    mochi::CoordinateSpaceConverter const* converter = nullptr) {
  return mochi_renderer::BuildMochiModelGeometry(
      model, converter, out.positions, out.normals, out.indices);
}

// A converter that leaves coordinates as they are, so expected positions can be written in Mochi
// space.
mochi::CoordinateSpaceConverter const& Identity() {
  static mochi::CoordinateSpaceConverter const converter(
      mochi::CoordinateSpace::Default(), mochi::CoordinateSpace::Default());
  return converter;
}

mochi::ModelData SurfaceMesh(
    std::vector<mochi::real> const& coordinates,
    std::vector<int> const& connectivity) {
  mochi::ModelData model;
  model.mesh = mochi::MeshData{};
  model.mesh->nodesPerElement = 3;
  model.mesh->coordinates = coordinates;
  model.mesh->connectivity = connectivity;
  return model;
}

// Every index names a vertex, and every normal has unit length.
void ExpectWellFormed(Geometry const& geometry) {
  ASSERT_EQ(geometry.normals.size(), geometry.positions.size());
  auto const vertexCount = static_cast<int>(geometry.positions.size() / 3);
  EXPECT_EQ(geometry.indices.size() % 3, 0u);
  for (int const index : geometry.indices) {
    EXPECT_TRUE(index >= 0 && index < vertexCount) << index;
  }
  for (std::size_t i = 0; i + 2 < geometry.normals.size(); i += 3) {
    float const length =
        std::hypot(geometry.normals[i], geometry.normals[i + 1], geometry.normals[i + 2]);
    EXPECT_NEAR(length, 1.0f, 1e-4f) << "normal " << i / 3;
  }
}

TEST(ModelGeometryTest, SurfaceMeshKeepsItsVerticesTrianglesAndFacing) {
  mochi::ModelData const model = SurfaceMesh({0, 0, 0, 1, 0, 0, 0, 1, 0}, {0, 1, 2});

  Geometry geometry;
  ASSERT_TRUE(Build(model, geometry, &Identity()));

  EXPECT_EQ(geometry.positions, (std::vector<float>{0, 0, 0, 1, 0, 0, 0, 1, 0}));
  EXPECT_EQ(geometry.indices, (std::vector<int>{0, 1, 2}));
  // Counter-clockwise in the x-y plane, so every vertex normal points along +z.
  EXPECT_EQ(geometry.normals, (std::vector<float>{0, 0, 1, 0, 0, 1, 0, 0, 1}));
}

TEST(ModelGeometryTest, DefaultConversionPutsMochiUpOnTheRenderersUpAxis) {
  // Mochi is z-up; the renderer is y-up.
  mochi::ModelData const model = SurfaceMesh({0, 0, 0, 1, 0, 0, 0, 0, 1}, {0, 1, 2});

  Geometry geometry;
  ASSERT_TRUE(Build(model, geometry));

  ASSERT_EQ(geometry.positions.size(), 9u);
  EXPECT_FLOAT_EQ(geometry.positions[6 + 1], 1.0f); // the third vertex, (0, 0, 1) in Mochi
  ExpectWellFormed(geometry);
}

TEST(ModelGeometryTest, TetrahedronGivesItsFourBoundaryTriangles) {
  mochi::ModelData model;
  model.mesh = mochi::MeshData{};
  model.mesh->nodesPerElement = 4;
  model.mesh->coordinates = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  model.mesh->connectivity = {0, 1, 2, 3};

  Geometry geometry;
  ASSERT_TRUE(Build(model, geometry, &Identity()));

  EXPECT_EQ(geometry.indices.size(), 4u * 3u);
  std::vector<float> sorted = geometry.positions;
  std::ranges::sort(sorted);
  // The same four corners (nine zeros and three ones), in whatever order the boundary lists them.
  EXPECT_EQ(sorted, (std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1}));
  ExpectWellFormed(geometry);
}

TEST(ModelGeometryTest, BoxShapeSpansItsHalfExtents) {
  mochi::ModelData model;
  model.box = mochi::Box{mochi::Real3{0.0f, 0.0f, 0.0f}, mochi::Real3{0.1f, 0.2f, 0.3f}};

  Geometry geometry;
  ASSERT_TRUE(Build(model, geometry, &Identity()));

  EXPECT_EQ(geometry.indices.size(), 6u * 2u * 3u);
  for (int axis = 0; axis < 3; ++axis) {
    float highest = 0.0f;
    for (std::size_t i = axis; i < geometry.positions.size(); i += 3) {
      highest = std::max(highest, std::abs(geometry.positions[i]));
    }
    EXPECT_NEAR(highest, 0.1f * static_cast<float>(axis + 1), 1e-6f) << "axis " << axis;
  }
  ExpectWellFormed(geometry);
}

TEST(ModelGeometryTest, PlaneShapeIsTwoTrianglesFacingItsNormal) {
  mochi::ModelData model;
  model.plane = mochi::Plane{};
  model.plane->normal = mochi::Real3{0.0f, 0.0f, 1.0f};
  model.plane->distance = 0.5f;

  Geometry geometry;
  ASSERT_TRUE(Build(model, geometry, &Identity()));

  EXPECT_EQ(geometry.indices.size(), 2u * 3u);
  ASSERT_EQ(geometry.positions.size(), 4u * 3u);
  for (std::size_t i = 0; i < geometry.positions.size(); i += 3) {
    EXPECT_FLOAT_EQ(geometry.positions[i + 2], 0.5f) << "vertex " << i / 3;
    EXPECT_EQ(
        (std::vector<float>{geometry.normals[i], geometry.normals[i + 1], geometry.normals[i + 2]}),
        (std::vector<float>{0, 0, 1}));
  }
  ExpectWellFormed(geometry);
}

TEST(ModelGeometryTest, SphereShapeHasEveryVertexOnItsSurface) {
  mochi::ModelData model;
  model.sphere = mochi::Sphere{};
  model.sphere->center = mochi::Real3{1.0f, 2.0f, 3.0f};
  model.sphere->radius = 0.5f;

  Geometry geometry;
  ASSERT_TRUE(Build(model, geometry, &Identity()));

  ASSERT_FALSE(geometry.positions.empty());
  for (std::size_t i = 0; i < geometry.positions.size(); i += 3) {
    float const distance = std::hypot(
        geometry.positions[i] - 1.0f,
        geometry.positions[i + 1] - 2.0f,
        geometry.positions[i + 2] - 3.0f);
    EXPECT_NEAR(distance, 0.5f, 1e-5f) << "vertex " << i / 3;
  }
  ExpectWellFormed(geometry);
}

TEST(ModelGeometryTest, ModelWithoutMeshOrShapeHasNoGeometry) {
  Geometry geometry;
  EXPECT_FALSE(Build(mochi::ModelData{}, geometry));
}

mochi::MeshData Polyline(
    std::vector<mochi::real> const& coordinates,
    std::vector<int> const& connectivity) {
  mochi::MeshData mesh;
  mesh.nodesPerElement = 2;
  mesh.coordinates = coordinates;
  mesh.connectivity = connectivity;
  return mesh;
}

struct PolylineGeometry {
  std::vector<float> positions;
  std::vector<int> segments;
};

bool BuildPolyline(mochi::MeshData const& mesh, PolylineGeometry& out) {
  return mochi_renderer::BuildMochiPolylineGeometry(mesh, &Identity(), out.positions, out.segments);
}

TEST(ModelGeometryTest, PolylineKeepsItsNodesAndSegments) {
  mochi::MeshData const mesh = Polyline({0, 0, 0, 1, 0, 0, 1, 1, 0}, {0, 1, 1, 2});

  PolylineGeometry geometry;
  ASSERT_TRUE(BuildPolyline(mesh, geometry));

  EXPECT_EQ(geometry.positions, (std::vector<float>{0, 0, 0, 1, 0, 0, 1, 1, 0}));
  EXPECT_EQ(geometry.segments, (std::vector<int>{0, 1, 1, 2}));
}

TEST(ModelGeometryTest, ClosedPolylineKeepsItsWrapSegment) {
  mochi::MeshData const mesh = Polyline({0, 0, 0, 1, 0, 0, 1, 1, 0}, {0, 1, 1, 2, 2, 0});

  PolylineGeometry geometry;
  ASSERT_TRUE(BuildPolyline(mesh, geometry));

  EXPECT_EQ(geometry.segments, (std::vector<int>{0, 1, 1, 2, 2, 0}));
}

TEST(ModelGeometryTest, PolylineWithoutConnectivityIsOpenThroughItsNodesInOrder) {
  mochi::MeshData const mesh = Polyline({0, 0, 0, 1, 0, 0, 2, 0, 0}, {});

  PolylineGeometry geometry;
  ASSERT_TRUE(BuildPolyline(mesh, geometry));

  EXPECT_EQ(geometry.segments, (std::vector<int>{0, 1, 1, 2}));
}

TEST(ModelGeometryTest, MeshesWithoutSegmentsHaveNoPolylineGeometry) {
  PolylineGeometry geometry;
  EXPECT_FALSE(BuildPolyline(*SurfaceMesh({0, 0, 0, 1, 0, 0, 0, 1, 0}, {0, 1, 2}).mesh, geometry));
  EXPECT_FALSE(BuildPolyline(Polyline({0, 0, 0}, {}), geometry));
  EXPECT_FALSE(BuildPolyline(mochi::MeshData{}, geometry));
}

} // namespace
