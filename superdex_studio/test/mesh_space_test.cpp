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

// Mesh-file space convention (processing_mesh_utils.h). Each format needs something different,
// and getting any of them wrong tips a model a quarter turn about X.
//
// clang-format off
//   .obj / .stl  Mochi-space (Z-up) on disk; converted to RenderSpace() on read
//   .glb         renderer-space already
//   .step        Z-up on disk, but superdex_mesh_cli converts it during tessellation, so it too
//                arrives renderer-space and must NOT be converted again
// clang-format on
//
// The axis_gizmo assets pin this down: the gizmo is the identity in its own file coordinates (the
// red arm runs along the file's +X, green along +Y, blue along +Z, each 0.1 m long), and the .glb
// was baked from the .obj through the very converter under test. So every format has to land on one
// shared orientation -- which is what a missing or doubled conversion breaks.

#include "meshing/processing_modifiers/processing_mesh_utils.h"

#include <mochi_core/geometry/mesh_data_utils.h>
#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/file_utils.h>

#include <mochi_mesh/step_tessellation.h>

#include <mochi_physics/utils/mochi_model_utils.h>

#include <mochi_renderer/utils.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace superdex::studio::processing {
namespace {

using mochi_renderer::MeshSection;

// Axis-aligned bounds as {minX, minY, minZ, maxX, maxY, maxZ}.
using Bounds = std::array<float, 6>;

constexpr float kTolerance = 1e-6f;

// The gizmo's arms in renderer space, i.e. the Mochi-space file extents
// x[-0.0125, 0.1] y[-0.0125, 0.1] z[-0.0125, 0.1] mapped by Default(FLU) -> RenderSpace(FUR),
// which sends (x, y, z) to (x, z, -y): +X stays right, the file's +Z becomes up, the file's +Y
// becomes -Z.
constexpr Bounds kGizmoInRenderSpace{-0.0125f, -0.0125f, -0.1f, 0.1f, 0.1f, 0.0125f};

std::string AssetPath(char const* name) {
  char const* const dir = std::getenv("SUPERDEX_STUDIO_TEST_ASSETS");
  return (std::filesystem::path(dir != nullptr ? dir : ".") / name).string();
}

Bounds SectionBounds(std::vector<MeshSection> const& sections) {
  Bounds bounds{FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
  for (MeshSection const& section : sections) {
    for (std::size_t i = 0; i + 3 <= section.positions.size(); i += 3) {
      for (int axis = 0; axis < 3; ++axis) {
        bounds[axis] = std::min(bounds[axis], section.positions[i + axis]);
        bounds[axis + 3] = std::max(bounds[axis + 3], section.positions[i + axis]);
      }
    }
  }
  return bounds;
}

void ExpectBoundsNear(Bounds const& actual, Bounds const& expected) {
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(actual[i], expected[i], kTolerance) << "bounds component " << i;
  }
}

// --- Reading ------------------------------------------------------------------------------------

TEST(MeshFileSpace, ObjReadsIntoRenderSpace) {
  std::vector<MeshSection> const sections = ReadSectionsInRenderSpace(AssetPath("axis_gizmo.obj"));
  ASSERT_FALSE(sections.empty());
  ExpectBoundsNear(SectionBounds(sections), kGizmoInRenderSpace);
}

TEST(MeshFileSpace, StlReadsIntoRenderSpace) {
  std::vector<MeshSection> const sections = ReadSectionsInRenderSpace(AssetPath("axis_gizmo.stl"));
  ASSERT_FALSE(sections.empty());
  ExpectBoundsNear(SectionBounds(sections), kGizmoInRenderSpace);
}

// The .glb is the reference: it needs no conversion, so it is where the other two must arrive.
TEST(MeshFileSpace, GlbReadsAsRenderSpaceAlready) {
  std::vector<MeshSection> const sections = ReadSectionsInRenderSpace(AssetPath("axis_gizmo.glb"));
  ASSERT_FALSE(sections.empty());
  ExpectBoundsNear(SectionBounds(sections), kGizmoInRenderSpace);
}

// STEP is the exception to the rule the tests above cover: the superdex_mesh_cli helper folds
// OCCT's Z-up -> Y-up rotation (and mm -> m) in as it reads the triangulation, so TessellateStep
// hands back renderer-space geometry that must NOT be converted again. Guarding it here catches a
// conversion added on this side as readily as one dropped from the helper.
//
// Checked by where the three 0.1 m arms point rather than against kGizmoInRenderSpace:
// axis_gizmo.stp is not a match for axis_gizmo.obj (it carries the 25 mm centre feature of the _box
// variant, where the .obj stops at 12.5 mm), so its short extents differ. The arm tips are what
// encode orientation, and they are unambiguous -- a spurious or missing quarter turn about X swaps
// which of +Y / -Z the gizmo reaches 0.1 m along, far outside any tessellation error.
TEST(MeshFileSpace, StepTessellatesIntoRenderSpaceAlready) {
  mochi::ErrorAssert error;
  mochi::mesh::StepTessellationParams params;
  params.linearDeflection = 0.05; // mm, the Model Editor's CAD-preview default
  params.angularDeflection = 0.25; // rad
  mochi::MeshData const mesh =
      mochi::mesh::TessellateStep(AssetPath("axis_gizmo.stp"), params, error);
  ASSERT_GT(mesh.GetNumElements(), 0);

  MeshSection section;
  section.positions.reserve(mesh.coordinates.size());
  for (mochi::real const coordinate : mesh.coordinates) {
    section.positions.push_back(static_cast<float>(coordinate));
  }
  Bounds const bounds = SectionBounds({section});

  constexpr float kArmLength = 0.1f;
  constexpr float kArmTolerance = 1e-3f; // tessellation only approximates the conical tips
  constexpr float kStubLimit = 0.05f; // no arm runs the other way, so these stay near the origin
  EXPECT_NEAR(bounds[3], kArmLength, kArmTolerance) << "red arm should run to renderer +X";
  EXPECT_NEAR(bounds[4], kArmLength, kArmTolerance) << "blue arm should run to renderer +Y";
  EXPECT_NEAR(bounds[2], -kArmLength, kArmTolerance) << "green arm should run to renderer -Z";
  EXPECT_GT(bounds[0], -kStubLimit) << "nothing should run to renderer -X";
  EXPECT_GT(bounds[1], -kStubLimit) << "nothing should run to renderer -Y";
  EXPECT_LT(bounds[5], kStubLimit) << "nothing should run to renderer +Z";
}

// The CAD preview's colored tessellation goes through TessellateStepByColor instead, so it must
// land in the same orientation as TessellateStep AND keep each arm's color on that arm. Colors are
// identified by their dominant channel, which holds for linear and sRGB factors alike.
TEST(MeshFileSpace, StepPreviewSectionsKeepColorsInRenderSpace) {
  mochi::ErrorAssert error;
  mochi::mesh::StepTessellationParams params;
  params.linearDeflection = 0.05; // mm, the Model Editor's CAD-preview default
  params.angularDeflection = 0.25; // rad
  std::vector<MeshSection> const sections =
      TessellateStepSections(AssetPath("axis_gizmo.stp"), params, error);
  ASSERT_GT(sections.size(), 1u) << "expected one section per CAD color";

  auto dominatedBy = [](MeshSection const& section, int channel) {
    auto const& c = section.baseColor;
    return c[channel] > 2.0f * c[(channel + 1) % 3] && c[channel] > 2.0f * c[(channel + 2) % 3];
  };
  auto boundsOfColor = [&](int channel) {
    std::vector<MeshSection> matching;
    for (MeshSection const& section : sections) {
      if (dominatedBy(section, channel)) {
        matching.push_back(section);
      }
    }
    EXPECT_FALSE(matching.empty()) << "no section dominated by channel " << channel;
    return SectionBounds(matching);
  };

  constexpr float kArmLength = 0.1f;
  constexpr float kArmTolerance = 1e-3f;
  EXPECT_NEAR(boundsOfColor(0)[3], kArmLength, kArmTolerance) << "red arm -> renderer +X";
  EXPECT_NEAR(boundsOfColor(2)[4], kArmLength, kArmTolerance) << "blue arm -> renderer +Y";
  EXPECT_NEAR(boundsOfColor(1)[2], -kArmLength, kArmTolerance) << "green arm -> renderer -Z";
}

// --- Materials ----------------------------------------------------------------------------------

// An .obj's colors live in the .mtl its `mtllib` line names, resolved relative to the .obj. When
// that lookup fails there is no error -- every face silently falls back to MeshSection's default
// grey -- so the failure mode is a model that loads perfectly and renders in the wrong color. This
// pins the whole chain: mtllib resolves, Kd becomes baseColor, and one section is emitted per
// usemtl (in first-seen order).
TEST(MeshFileMaterials, ObjResolvesItsMtlColors) {
  std::vector<MeshSection> const sections = ReadSectionsInRenderSpace(AssetPath("axis_gizmo.obj"));
  ASSERT_EQ(sections.size(), 4u) << "expected one section per usemtl in axis_gizmo.mtl";

  // axis_gizmo.mtl's Kd values, in the order the faces first reference them.
  std::array<std::array<float, 4>, 4> const expected{{
      {0.325490f, 0.784314f, 0.015686f, 1.0f}, // mat_1, green
      {1.0f, 1.0f, 1.0f, 1.0f}, // mat_2, white
      {1.0f, 0.149020f, 0.149020f, 1.0f}, // mat_3, red
      {0.117647f, 0.403922f, 0.882353f, 1.0f}, // mat_4, blue
  }};
  for (std::size_t s = 0; s < sections.size(); ++s) {
    for (int c = 0; c < 4; ++c) {
      EXPECT_NEAR(sections[s].baseColor[c], expected[s][c], 1e-5f)
          << "section " << s << " channel " << c;
    }
  }
}

// --- Writing ------------------------------------------------------------------------------------

// An exported .obj must come back out in the space an .obj is read as, or every export would drift
// another quarter turn each time it round-trips through the editor.
TEST(MeshFileSpace, ObjExportRoundTripsThroughRenderSpace) {
  mochi::ErrorAssert error;
  mochi::MeshData const loaded = LoadRenderMesh(AssetPath("axis_gizmo.obj"), error);
  ASSERT_GT(loaded.GetNumElements(), 0);

  mochi::TempDirCleanup const folder = mochi::CreateTempDirectory("mesh_space_test", error);
  std::filesystem::path const out = folder.Path() / "axis_gizmo_roundtrip.obj";
  WriteObjFile(out.string(), loaded, error);

  std::vector<MeshSection> const reread = ReadSectionsInRenderSpace(out.string());
  ASSERT_FALSE(reread.empty());
  ExpectBoundsNear(SectionBounds(reread), kGizmoInRenderSpace);
}

// --- Tetrahedral stage outputs ------------------------------------------------------------------

// A tetrahedron split into four by a node at its centroid, so the boundary excludes one node.
mochi::MeshData CentroidSplitTetrahedron() {
  mochi::MeshData mesh;
  mesh.nodesPerElement = 4;
  for (mochi::real const coordinate :
       {0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.25f,
        0.25f,
        0.25f}) {
    mesh.coordinates.push_back(coordinate);
  }
  for (int const node : {4, 1, 2, 3, 0, 4, 2, 3, 0, 1, 4, 3, 0, 1, 2, 4}) {
    mesh.connectivity.push_back(node);
  }
  return mesh;
}

mochi::MeshData SingleTriangle() {
  mochi::MeshData mesh;
  mesh.nodesPerElement = 3;
  for (mochi::real const coordinate : {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}) {
    mesh.coordinates.push_back(coordinate);
  }
  for (int const node : {0, 1, 2}) {
    mesh.connectivity.push_back(node);
  }
  return mesh;
}

TEST(TetStageOutput, DisplaySurfaceIsTheBoundaryWithoutInteriorNodes) {
  mochi::MeshData const surface = DisplaySurface(CentroidSplitTetrahedron());

  EXPECT_EQ(surface.nodesPerElement, 3);
  EXPECT_EQ(surface.GetNumElements(), 4);
  EXPECT_EQ(surface.GetNumNodes(), 4);
}

TEST(TetStageOutput, DisplaySurfacePassesTrianglesThrough) {
  mochi::MeshData const triangle = SingleTriangle();

  mochi::MeshData const surface = DisplaySurface(triangle);

  EXPECT_EQ(surface.coordinates, triangle.coordinates);
  EXPECT_EQ(surface.connectivity, triangle.connectivity);
}

TEST(TetStageOutput, BufferedStageOutputPreservesTheCompleteTetMesh) {
  mochi::MeshData const tets = CentroidSplitTetrahedron();
  std::vector<mochi_renderer::MeshSection> const sections{
      SectionFromMeshData(DisplaySurface(tets))};

  mochi::MeshData const output = BufferedStageOutput({.mesh = tets}, sections).mesh;

  EXPECT_EQ(output.nodesPerElement, 4);
  EXPECT_EQ(output.GetNumNodes(), 5);
  EXPECT_EQ(output.coordinates, tets.coordinates);
  EXPECT_EQ(output.connectivity, tets.connectivity);
}

TEST(TetStageOutput, BufferedTriangleStagePreservesItsMeshAndEmbeddings) {
  StageMesh stage{.mesh = SingleTriangle()};
  stage.embedded.contactSkinMesh = SingleTriangle();
  stage.embedded.visualMesh = SingleTriangle();
  stage.embedded.visualMesh->coordinates[0] = 2;
  std::vector<mochi_renderer::MeshSection> const sections{
      SectionFromMeshData(*stage.embedded.visualMesh)};

  StageMesh const output = BufferedStageOutput(stage, sections);

  EXPECT_EQ(output.mesh, stage.mesh);
  ASSERT_TRUE(output.embedded.contactSkinMesh);
  ASSERT_TRUE(output.embedded.visualMesh);
  EXPECT_EQ(*output.embedded.contactSkinMesh, *stage.embedded.contactSkinMesh);
  EXPECT_EQ(*output.embedded.visualMesh, *stage.embedded.visualMesh);
}

TEST(TetStageOutput, BufferedStageOutputFallsBackToDisplayedTriangles) {
  mochi::MeshData const triangle = SingleTriangle();
  mochi::MeshData emptyTetMesh;
  emptyTetMesh.nodesPerElement = 4;
  std::vector<mochi_renderer::MeshSection> const sections{SectionFromMeshData(triangle)};

  mochi::MeshData const output = BufferedStageOutput({.mesh = emptyTetMesh}, sections).mesh;

  EXPECT_EQ(output.nodesPerElement, 3);
  EXPECT_EQ(output.coordinates, triangle.coordinates);
  EXPECT_EQ(output.connectivity, triangle.connectivity);
}

// A non-planar polyline in Mochi space: (0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1).
mochi::ModelData PolylineModel() {
  mochi::ModelData model;
  model.mesh.emplace();
  model.mesh->nodesPerElement = 2;
  for (mochi::real const coordinate : {0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1}) {
    model.mesh->coordinates.push_back(coordinate);
  }
  return model;
}

TEST(PolylineStage, DisplaySurfaceIsEmpty) {
  mochi::MeshData polyline = *PolylineModel().mesh;
  polyline.connectivity = mochi::MakeSequentialPolylineConnectivity(4, /*isClosedLoop=*/false);
  EXPECT_EQ(DisplaySurface(polyline).GetNumElements(), 0);
}

TEST(PolylineStage, BufferedStageOutputPreservesThePolylineAndItsFrameAxes) {
  StageMesh full{.mesh = *PolylineModel().mesh};
  full.mesh.connectivity = mochi::MakeSequentialPolylineConnectivity(4, /*isClosedLoop=*/false);
  full.elementFrameAxes = mochi::DynamicArray<mochi::real>{0, 1, 0, 0, 0, 1, 1, 0, 0};

  StageMesh const output = BufferedStageOutput(full, {});

  EXPECT_EQ(output.mesh.coordinates, full.mesh.coordinates);
  EXPECT_EQ(output.mesh.connectivity, full.mesh.connectivity);
  EXPECT_EQ(output.elementFrameAxes, full.elementFrameAxes);
}

// The stage holds the authored frame axes, the polyline's elements stated explicitly, and both in
// renderer space, where Mochi-space (x, y, z) is (x, z, -y). The model's contact skin is left out.
TEST(PolylineStage, FromAMochiModelKeepsItsAuthoredFrameAxes) {
  mochi::ModelData model = PolylineModel();
  model.elementFrameAxes = mochi::DynamicArray<mochi::real>{0, 1, 0, 0, 0, 1, 1, 0, 0};
  model.contactSkinMesh = mochi::model::GenerateTubularSkinMesh(
      mochi::Unflatten<mochi::Real3 const>(mochi::MakeConstSpan(model.mesh->coordinates)),
      mochi::Unflatten<mochi::Real3 const>(mochi::MakeConstSpan(*model.elementFrameAxes)),
      /*isClosedLoop=*/false,
      0.1f,
      4,
      mochi::test::ExpectOK{});

  StageMesh const stage = PolylineStageFromMochiModel(model, mochi::test::ExpectOK{});

  EXPECT_EQ(stage.mesh.nodesPerElement, 2);
  EXPECT_EQ(
      stage.mesh.coordinates,
      (mochi::DynamicArray<mochi::real>{0, 0, 0, 1, 0, 0, 1, 0, -1, 1, 1, -1}));
  EXPECT_EQ(
      stage.mesh.connectivity,
      mochi::MakeSequentialPolylineConnectivity(4, /*isClosedLoop=*/false));
  ASSERT_TRUE(stage.elementFrameAxes.has_value());
  EXPECT_EQ(
      *stage.elementFrameAxes, (mochi::DynamicArray<mochi::real>{0, 0, -1, 0, 1, 0, 1, 0, 0}));
  EXPECT_TRUE(stage.embedded.IsEmpty());
}

TEST(PolylineStage, FromAnUnframedMochiModelGetsItsBishopFrame) {
  mochi::ModelData const model = PolylineModel();
  mochi::DynamicArray<mochi::Real3> const bishop = mochi::model::GenerateDiscreteBishopFrame(
      mochi::Unflatten<mochi::Real3 const>(mochi::MakeConstSpan(model.mesh->coordinates)),
      /*isClosedLoop=*/false);

  StageMesh const stage = PolylineStageFromMochiModel(model, mochi::test::ExpectOK{});

  ASSERT_TRUE(stage.elementFrameAxes.has_value());
  mochi::DynamicArray<mochi::real> expected;
  for (mochi::Real3 const& axis : bishop) {
    expected.append(mochi::MakeConstSpan(mochi::Real3{axis[0], axis[2], -axis[1]}));
  }
  ASSERT_EQ(stage.elementFrameAxes->size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR((*stage.elementFrameAxes)[i], expected[i], 1e-6) << "component " << i;
  }
}

TEST(PolylineStage, LoadMochiStageReadsAPolylineModel) {
  mochi::TempDirCleanup const dir =
      mochi::CreateTempDirectory("polyline_stage_test", mochi::test::ExpectOK{});
  std::string const path = (dir.Path() / "rod.mochi.json").string();
  mochi::model_utils::SaveToFile(
      PolylineModel(), path, mochi::FileFormat::JSON, mochi::test::ExpectOK{});

  StageMesh const stage = LoadMochiStage(path, mochi::test::ExpectOK{});

  EXPECT_EQ(stage.mesh.nodesPerElement, 2);
  EXPECT_EQ(stage.mesh.GetNumElements(), 3);
  EXPECT_TRUE(stage.elementFrameAxes.has_value());
}

// RenderSpace (x, y, z) is Mochi-space (x, -z, y); connectivity is untouched.
TEST(TetStageOutput, ToMochiSpaceKeepsConnectivity) {
  mochi::MeshData const tets = CentroidSplitTetrahedron();

  mochi::MeshData const converted = ToMochiSpace(tets);

  EXPECT_EQ(converted.nodesPerElement, 4);
  EXPECT_EQ(converted.connectivity, tets.connectivity);
  EXPECT_NEAR(converted.coordinates[(3 * 3) + 1], -1.0f, kTolerance); // node 3 = (0, 0, 1)
  EXPECT_NEAR(converted.coordinates[(2 * 3) + 2], 1.0f, kTolerance); // node 2 = (0, 1, 0)
}

// Nodes in Mochi space (see ToMochiSpaceKeepsConnectivity); cells are 0-based VTK_TETRA (type 10).
TEST(TetStageOutput, VtkFileHoldsTheTetsInMochiSpace) {
  mochi::ErrorAssert error;
  mochi::TempDirCleanup const folder = mochi::CreateTempDirectory("mesh_space_test", error);
  std::filesystem::path const path = folder.Path() / "tets.vtk";

  WriteVtkFile(path.string(), CentroidSplitTetrahedron(), error);

  std::ifstream in(path);
  std::string const contents{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  std::string const expected =
      "# vtk DataFile Version 3.0\n"
      "Superdex Studio tetrahedral mesh\n"
      "ASCII\n"
      "DATASET UNSTRUCTURED_GRID\n"
      "POINTS 5 double\n"
      "0 0 0\n"
      "1 0 0\n"
      "0 0 1\n"
      "0 -1 0\n"
      "0.25 -0.25 0.25\n"
      "CELLS 4 20\n"
      "4 4 1 2 3\n"
      "4 0 4 2 3\n"
      "4 0 1 4 3\n"
      "4 0 1 2 4\n"
      "CELL_TYPES 4\n"
      "10\n10\n10\n10\n";
  EXPECT_EQ(contents, expected);
}

TEST(TetStageOutput, VtkFileRejectsTriangles) {
  mochi::MeshData triangle;
  triangle.nodesPerElement = 3;
  WriteVtkFile("unused.vtk", triangle, mochi::test::ExpectNotOK{});
}

#if MOCHI_PLATFORM_LINUX
TEST(TetStageOutput, VtkFileReportsDelayedWriteFailure) {
  WriteVtkFile("/dev/full", CentroidSplitTetrahedron(), mochi::test::ExpectNotOK{});
}
#endif

} // namespace
} // namespace superdex::studio::processing
