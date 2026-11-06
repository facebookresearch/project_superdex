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

#include "rendering/soft_render_link.h"

#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_renderer/deformable_mesh.h>
#include <mochi_renderer/render_space.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

using namespace superdex::studio;
using mochi_renderer::DeformableMeshView;

namespace {

struct RenderMesh {
  std::vector<float> positions = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
  std::vector<int> indices = {0, 1, 2, 3, 4, 5, 3, 5, 6};

  [[nodiscard]] DeformableMeshView View() const {
    return {
        .restPositions = mochi::MakeConstSpan(positions),
        .triangleIndices = mochi::MakeConstSpan(indices)};
  }
};

RenderMesh RenderModel() {
  return {};
}

// Maps @p point from the actor frame back into the frame @p placement places it from.
mochi::Real3 Unplace(ModelPlacement const& placement, mochi::Real3 const& point) {
  return placement.transform.TransformPointInverse(point) / placement.scale;
}

// @p renderMesh authored as the Model Editor's embedded visual mesh: converted to Mochi's axes,
// placed by @p render and mapped back through @p shape.
mochi::MeshData Embedded(
    DeformableMeshView renderMesh,
    ModelPlacement const& render = {},
    ModelPlacement const& shape = {}) {
  mochi::CoordinateSpaceConverter const renderToMochi(
      mochi_renderer::RenderSpace(), mochi::CoordinateSpace::Default());
  mochi::MeshData mesh;
  mesh.nodesPerElement = 3;
  mesh.skinning.emplace();
  for (size_t i = 0; i < renderMesh.restPositions.size(); i += 3) {
    mochi::Real3 const point = Unplace(
        shape,
        render.Apply(renderToMochi.TranslationToOutput(
            mochi::Real3{
                renderMesh.restPositions[i],
                renderMesh.restPositions[i + 1],
                renderMesh.restPositions[i + 2]})));
    for (mochi::real const coordinate : point) {
      mesh.coordinates.push_back(coordinate);
    }
  }
  mesh.connectivity.assign(renderMesh.triangleIndices.begin(), renderMesh.triangleIndices.end());
  return mesh;
}

// A non-uniformly scaled, rotated and translated placement.
ModelPlacement Placement() {
  return {
      .scale = {2.0f, 0.5f, 1.5f},
      .transform = mochi::TransformRT(
          mochi::Quaternion::FromAxisAngle(
              mochi::Real3{0.0f, 0.0f, 1.0f}, std::numbers::pi_v<float> / 3.0f),
          mochi::Real3{0.3f, -0.2f, 1.0f})};
}

TEST(SoftRenderLinkTest, TheEmbeddedRenderModelIsLinked) {
  RenderMesh const renderMesh = RenderModel();
  RenderModelLink const link =
      CheckRenderModelLink(renderMesh.View(), Embedded(renderMesh.View()), {}, {});
  EXPECT_TRUE(link.linked) << link.reason;
  EXPECT_EQ(link.reason, "");
}

TEST(SoftRenderLinkTest, MatchingRenderAndShapePlacementsStayLinked) {
  RenderMesh const renderMesh = RenderModel();
  ModelPlacement const placement = Placement();
  RenderModelLink const link =
      CheckRenderModelLink(renderMesh.View(), Embedded(renderMesh.View()), placement, placement);
  EXPECT_TRUE(link.linked) << link.reason;
}

TEST(SoftRenderLinkTest, AnEmbeddingUnderDifferentPlacementsIsLinkedUnderThem) {
  RenderMesh const renderMesh = RenderModel();
  ModelPlacement const render = Placement();
  ModelPlacement const shape{.scale = {1.0f, 3.0f, 1.0f}};
  RenderModelLink const link = CheckRenderModelLink(
      renderMesh.View(), Embedded(renderMesh.View(), render, shape), render, shape);
  EXPECT_TRUE(link.linked) << link.reason;
}

TEST(SoftRenderLinkTest, RoundOffAtAFarPartsDistanceStaysLinked) {
  RenderMesh const renderMesh = RenderModel();
  ModelPlacement const render{
      .scale = {1e-3f, 1e-3f, 1e-3f},
      .transform = mochi::TransformRT(mochi::Real3{100.0f, -50.0f, 20.0f})};
  mochi::MeshData visualMesh = Embedded(renderMesh.View(), render);
  // One float step at the part's distance from the origin, as either file may round it.
  for (mochi::real& coordinate : visualMesh.coordinates) {
    coordinate = std::nextafter(coordinate, std::numeric_limits<mochi::real>::max());
  }
  RenderModelLink const link = CheckRenderModelLink(renderMesh.View(), visualMesh, render, {});
  EXPECT_TRUE(link.linked) << link.reason;
}

TEST(SoftRenderLinkTest, AMovedRenderModelIsNotLinked) {
  RenderMesh const renderMesh = RenderModel();
  ModelPlacement const moved{.transform = mochi::TransformRT(mochi::Real3{0.01f, 0.0f, 0.0f})};
  RenderModelLink const link =
      CheckRenderModelLink(renderMesh.View(), Embedded(renderMesh.View()), moved, {});
  EXPECT_FALSE(link.linked);
  EXPECT_NE(link.reason.find("offset or scale"), std::string::npos) << link.reason;
}

TEST(SoftRenderLinkTest, AnEditedRenderModelIsNotLinked) {
  RenderMesh renderMesh = RenderModel();
  mochi::MeshData const visualMesh = Embedded(renderMesh.View());
  renderMesh.positions[16] += 0.01f;
  EXPECT_FALSE(CheckRenderModelLink(renderMesh.View(), visualMesh, {}, {}).linked);
}

TEST(SoftRenderLinkTest, ADifferentVertexCountIsNotLinked) {
  RenderMesh const renderMesh = RenderModel();
  RenderMesh const fewer{.positions = {0, 0, 0, 1, 0, 0, 0, 1, 0}, .indices = {0, 1, 2}};
  RenderModelLink const link =
      CheckRenderModelLink(fewer.View(), Embedded(renderMesh.View()), {}, {});
  EXPECT_FALSE(link.linked);
  EXPECT_NE(link.reason.find("has 3 vertices, but the visual mesh has 7"), std::string::npos)
      << link.reason;
}

TEST(SoftRenderLinkTest, DifferentTrianglesAreNotLinked) {
  RenderMesh renderMesh = RenderModel();
  mochi::MeshData const visualMesh = Embedded(renderMesh.View());
  renderMesh.indices = {0, 1, 2, 3, 4, 6, 4, 5, 6};
  RenderModelLink const link = CheckRenderModelLink(renderMesh.View(), visualMesh, {}, {});
  EXPECT_FALSE(link.linked);
  EXPECT_NE(link.reason.find("triangles"), std::string::npos) << link.reason;
}

TEST(SoftRenderLinkTest, NoVisualMeshIsNotLinked) {
  RenderMesh const renderMesh = RenderModel();
  RenderModelLink const link =
      CheckRenderModelLink(renderMesh.View(), mochi::MeshDataView{}, {}, {});
  EXPECT_FALSE(link.linked);
  EXPECT_NE(link.reason.find("no visual mesh"), std::string::npos) << link.reason;
}

} // namespace
