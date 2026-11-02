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

// Runs processing pipelines headlessly (app/process_command.h) on a cube written by the test, and
// checks the files they export. The SDF and tetrahedral stages need the superdex_mesh_cli helper,
// which the test target points SUPERDEX_MESH_CLI_PATH at.

#include "app/process_command.h"
#include "meshing/processing_modifiers/processing_mesh_utils.h"

#include <mochi_core/geometry/model_data.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/utils/mochi_model_utils.h>
#include <mochi_renderer/render_space.h>
#include <mochi_renderer/utils.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using namespace superdex::studio;
namespace fs = std::filesystem;

namespace {

constexpr float kHalfSide = 0.05f; // a 10 cm cube centered on the origin
constexpr int kCubeTriangles = 12;
constexpr double kFileSizeEstimateTolerance = 0.1;

// The cube's corners, indexed as bit 0 = x, 1 = y, 2 = z.
constexpr std::array<std::array<float, 3>, 8> kCorner{{
    {-kHalfSide, -kHalfSide, -kHalfSide},
    {kHalfSide, -kHalfSide, -kHalfSide},
    {-kHalfSide, kHalfSide, -kHalfSide},
    {kHalfSide, kHalfSide, -kHalfSide},
    {-kHalfSide, -kHalfSide, kHalfSide},
    {kHalfSide, -kHalfSide, kHalfSide},
    {-kHalfSide, kHalfSide, kHalfSide},
    {kHalfSide, kHalfSide, kHalfSide},
}};

// The cube's sides (-x, +x, -y, +y, -z, +z) as outward-wound corner quads.
constexpr std::array<std::array<int, 4>, 6> kSide{{
    {0, 4, 6, 2},
    {1, 3, 7, 5},
    {0, 1, 5, 4},
    {2, 6, 7, 3},
    {0, 2, 3, 1},
    {4, 5, 7, 6},
}};

// Writes a binary STL of the cube, outward-wound.
void WriteCubeStl(
    fs::path const& path,
    std::array<float, 3> const& center = std::array<float, 3>{}) {
  fs::create_directories(path.parent_path());
  auto corner = kCorner;
  for (auto& position : corner) {
    for (int axis = 0; axis < 3; ++axis) {
      position[axis] += center[axis];
    }
  }
  std::ofstream out(path, std::ios::binary);
  std::array<char, 80> const header{};
  out.write(header.data(), header.size());
  uint32_t const count = kCubeTriangles;
  out.write(reinterpret_cast<char const*>(&count), sizeof(count));
  auto writeTriangle = [&](int a, int b, int c) {
    std::array<float, 3> const normal{};
    out.write(reinterpret_cast<char const*>(normal.data()), sizeof(normal));
    for (int const i : {a, b, c}) {
      out.write(reinterpret_cast<char const*>(corner[i].data()), sizeof(corner[i]));
    }
    uint16_t const attribute = 0;
    out.write(reinterpret_cast<char const*>(&attribute), sizeof(attribute));
  };
  for (auto const& quad : kSide) {
    writeTriangle(quad[0], quad[1], quad[2]);
    writeTriangle(quad[0], quad[2], quad[3]);
  }
}

// Writes the cube as a .glb of two primitives, the x sides and the other four, each with its own
// copy of the corners, as a render model split by material would be. Returns the primitives.
std::vector<mochi_renderer::MeshSection> WriteCubeGlb(fs::path const& path) {
  fs::create_directories(path.parent_path());
  std::vector<mochi_renderer::MeshSection> primitives(2);
  for (mochi_renderer::MeshSection& primitive : primitives) {
    for (auto const& corner : kCorner) {
      primitive.positions.insert(primitive.positions.end(), corner.begin(), corner.end());
    }
  }
  for (std::size_t s = 0; s < kSide.size(); ++s) {
    auto const& quad = kSide[s];
    std::vector<int>& indices = primitives[s < 2 ? 0 : 1].indices;
    indices.insert(indices.end(), {quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]});
  }
  EXPECT_TRUE(mochi_renderer::WriteGlbToFile(path.string().c_str(), primitives));
  return primitives;
}

// A pipeline: a CAD source, a mesh export, and an SDF export with 1 cm voxels. The mesh export
// passes its input on; the SDF export's own output is the SDF's reconstructed surface.
void WritePipeline(fs::path const& path, std::string const& source) {
  fs::create_directories(path.parent_path());
  std::ofstream(path) << R"({
  "version": 1,
  "modifiers": [
    )" << source << R"(,
    {"modifier": "Export Mesh File", "method": "Mesh File", "enabled": true, "collapsed": true,
     "properties": {"color": [0.2, 0.4, 0.6]}},
    {"modifier": "Export Mochi Model", "method": "SDF", "enabled": true, "collapsed": true,
     "properties": {"sdf": {"resolutionMode": "Explicit", "resolutionDelta": [0.01, 0.01, 0.01],
                            "boundaryPaddingDist": 0.01, "minGridResolution": [6, 6, 6]}}}
  ]
})";
}

constexpr char const* kSourceFromModelViewer =
    R"({"modifier": "Source from CAD Model", "method": "From Model Viewer", "enabled": true,
        "collapsed": true, "properties": {}})";

// Coarse, so the cube meshes quickly.
constexpr char const* kTetrahedralize =
    R"({"modifier": "Tetrahedralize", "method": "Delaunay Refinement", "enabled": true,
        "collapsed": true, "properties": {"edgeSize": 0.5}})";

constexpr char const* kExtractSurface =
    R"({"modifier": "Extract Surface", "method": "Boundary", "enabled": true, "collapsed": true,
        "properties": {}})";

constexpr char const* kMakeWatertight =
    R"({"modifier": "Refine Mesh", "method": "Make Watertight", "enabled": true,
        "collapsed": true, "properties": {}})";

// Embed the source surface (the cube) as the contact skin or the visual mesh.
constexpr char const* kEmbedSourceAsContactSkin =
    R"({"modifier": "Embed Contact Skin", "method": "Upstream Surface", "enabled": true,
        "collapsed": true, "properties": {"referenceIndex": 0}})";

constexpr char const* kEmbedSourceAsVisualMesh =
    R"({"modifier": "Embed Visual Mesh", "method": "Upstream Surface", "enabled": true,
        "collapsed": true, "properties": {"referenceIndex": 0}})";

constexpr char const* kRenderSourceFromModelViewer =
    R"({"modifier": "Source from Render Model", "method": "From Model Viewer", "enabled": true,
        "collapsed": true, "properties": {}})";

constexpr char const* kSdfExport =
    R"({"modifier": "Export Mochi Model", "method": "SDF", "enabled": true, "collapsed": true,
        "properties": {"sdf": {"resolutionMode": "Explicit", "resolutionDelta": [0.01, 0.01, 0.01],
                               "boundaryPaddingDist": 0.01, "minGridResolution": [6, 6, 6]}}})";

constexpr char const* kZeroAxisSrt =
    R"({"modifier": "Apply SRT Transformation", "method": "SRT Transform", "enabled": true,
        "collapsed": true, "properties": {"scale": [1, 0, 1]}})";

constexpr char const* kMirroredSrt =
    R"({"modifier": "Apply SRT Transformation", "method": "SRT Transform", "enabled": true,
        "collapsed": true, "properties": {"scale": [-2, 1, 1]}})";

constexpr char const* kTranslatedSrt =
    R"({"modifier": "Apply SRT Transformation", "method": "SRT Transform", "enabled": true,
        "collapsed": true, "properties": {"translation": [0.01, 0.02, 0.03]}})";

// The translation overflows float, so the output coordinates are infinite.
constexpr char const* kNonFiniteSrt =
    R"({"modifier": "Apply SRT Transformation", "method": "SRT Transform", "enabled": true,
        "collapsed": true, "properties": {"translation": [1e39, 0, 0]}})";

constexpr char const* kNoSdfExport =
    R"({"modifier": "Export Mochi Model", "method": "SDF", "enabled": true, "collapsed": true,
        "properties": {"bakeSdf": false}})";

// Writes a pipeline of @p modifiers, each a JSON object, in order.
void WriteModifiers(fs::path const& path, std::vector<std::string> const& modifiers) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path);
  out << R"({"version": 1, "modifiers": [)";
  for (std::size_t i = 0; i < modifiers.size(); ++i) {
    out << (i > 0 ? ",\n" : "\n") << modifiers[i];
  }
  out << "]}";
}

mochi::Real3 NodePosition(mochi::MeshData const& mesh, int node) {
  return {
      mesh.coordinates[3 * node], mesh.coordinates[3 * node + 1], mesh.coordinates[3 * node + 2]};
}

mochi::real TetSignedVolume(mochi::MeshData const& mesh, int tet) {
  int const* const nodes = &mesh.connectivity[4 * tet];
  mochi::Real3 const origin = NodePosition(mesh, nodes[0]);
  return mochi::Dot(
             NodePosition(mesh, nodes[1]) - origin,
             mochi::Cross(
                 NodePosition(mesh, nodes[2]) - origin, NodePosition(mesh, nodes[3]) - origin)) /
      mochi::real{6};
}

class ProcessCommandTest : public ::testing::Test {
 protected:
  static void ExpectCubeSdf(mochi::ModelData const& model, mochi::Int3 const& expectedDims) {
    ASSERT_TRUE(model.sdf.has_value());
    EXPECT_EQ(model.sdf->dims, expectedDims);
    std::size_t const expectedValueCount = static_cast<std::size_t>(expectedDims[0]) *
        static_cast<std::size_t>(expectedDims[1]) * static_cast<std::size_t>(expectedDims[2]);
    ASSERT_EQ(model.sdf->values.size(), expectedValueCount);
    auto const [lowest, highest] =
        std::minmax_element(model.sdf->values.begin(), model.sdf->values.end());
    EXPECT_LT(*lowest, -0.8f * kHalfSide); // deep inside, near the center
    EXPECT_GT(*highest, 0.0f); // the padding outside
  }

  // Checks the .mochi.h5 holds the triangle cube and its grid SDF.
  static void ExpectCubeModel(fs::path const& path, mochi::Int3 const& expectedDims) {
    mochi::Error error;
    mochi::ModelData const model = mochi::model_utils::LoadFromFile(path.string(), error);
    ASSERT_TRUE(error.IsOK()) << error.GetDescription();
    ASSERT_TRUE(model.mesh.has_value());
    EXPECT_EQ(model.mesh->nodesPerElement, 3);
    EXPECT_EQ(model.mesh->GetNumElements(), kCubeTriangles);
    ExpectCubeSdf(model, expectedDims);
  }

  static int GlbTriangles(fs::path const& path) {
    int triangles = 0;
    for (mochi_renderer::MeshSection const& section :
         mochi_renderer::ReadGlbFromFile(path.string().c_str())) {
      triangles += static_cast<int>(section.indices.size() / 3);
    }
    return triangles;
  }

  mochi::TempDirCleanup _rootCleanup =
      mochi::CreateTempDirectory("process_command_test", mochi::test::ExpectOK{});
  fs::path _root = _rootCleanup.Path();
};

} // namespace

TEST_F(ProcessCommandTest, ExportsNextToTheAssetThatOwnsThePipeline) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WritePipeline(pipeline, kSourceFromModelViewer);

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_TRUE(report.succeeded);
  EXPECT_TRUE(fs::equivalent(report.slots.cadPath, _root / "cad" / "cube.stl"));
  ASSERT_EQ(report.stages.size(), 3u);
  EXPECT_EQ(report.stages[0].triangles, kCubeTriangles);
  EXPECT_TRUE(fs::equivalent(report.stages[1].exportedPath, _root / "render" / "cube.glb"));
  ASSERT_TRUE(report.stages[2].sdfGrid.has_value());
  EXPECT_TRUE(fs::equivalent(report.stages[2].exportedPath, _root / "collision" / "cube.mochi.h5"));
  ExpectCubeModel(_root / "collision" / "cube.mochi.h5", *report.stages[2].sdfGrid);
  EXPECT_EQ(GlbTriangles(_root / "render" / "cube.glb"), kCubeTriangles);
}

TEST_F(ProcessCommandTest, NoSdfExportOfATriangleMeshWritesOnlyTheMesh) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WriteModifiers(pipeline, {kSourceFromModelViewer, kNoSdfExport});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_TRUE(report.succeeded);
  fs::path const modelPath = _root / "collision" / "cube.mochi.h5";
  ASSERT_EQ(report.stages.size(), 2u);
  EXPECT_EQ(report.stages[0].triangles, kCubeTriangles);
  EXPECT_EQ(report.stages[1].triangles, kCubeTriangles);
  EXPECT_TRUE(fs::equivalent(report.stages[1].exportedPath, modelPath));
  EXPECT_FALSE(report.stages[1].tetMesh.has_value());
  mochi::ModelData const model = mochi::model_utils::LoadFromFile(modelPath.string(), error);
  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  ASSERT_TRUE(model.mesh.has_value());
  EXPECT_EQ(model.mesh->nodesPerElement, 3);
  EXPECT_EQ(model.mesh->GetNumElements(), kCubeTriangles);
  EXPECT_FALSE(model.sdf.has_value());
}

TEST_F(ProcessCommandTest, ShippedSoftModelPresetProducesASoftActorAndAccurateVtkEstimate) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  char const* const presetsDir = std::getenv("SUPERDEX_STUDIO_PRESETS_PATH");
  ASSERT_NE(presetsDir, nullptr);
  fs::path const preset = fs::path(presetsDir) / "STEP_to_Soft_Mochi_Model.json";
  ASSERT_TRUE(fs::is_regular_file(preset));

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline(
      {.pipelinePath = preset.string(),
       .slots = {.cadPath = (_root / "cad" / "cube.stl").string()}},
      error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_TRUE(report.succeeded);
  ASSERT_EQ(report.stages.size(), 6u);
  for (ProcessStageReport const& stage : report.stages) {
    EXPECT_TRUE(stage.succeeded) << stage.label << ": " << stage.error;
  }
  for (std::size_t stage = 0; stage < 4; ++stage) {
    EXPECT_GT(report.stages[stage].triangles, 0) << report.stages[stage].label;
    EXPECT_FALSE(report.stages[stage].tetMesh.has_value()) << report.stages[stage].label;
  }
  ASSERT_TRUE(report.stages[4].tetMesh.has_value());
  EXPECT_EQ(report.stages[5].tetMesh, report.stages[4].tetMesh);
  fs::path const modelPath = _root / "collision" / "cube.mochi.h5";
  EXPECT_TRUE(fs::equivalent(report.stages[5].exportedPath, modelPath));

  mochi::ModelData const model = mochi::model_utils::LoadFromFile(modelPath.string(), error);
  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  ASSERT_TRUE(model.mesh.has_value());
  EXPECT_EQ(model.mesh->nodesPerElement, 4);
  EXPECT_EQ(model.mesh->GetNumElements(), (*report.stages[4].tetMesh)[1]);
  EXPECT_FALSE(model.sdf.has_value());

  mochi::Context* const context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  mochi::Scene* const scene = context->CreateScene("preset_tetrahedral_export");
  MOCHI_DEFER(context->DestroyScene(scene));
  mochi::SoftActorParams params;
  params.shape = context->LoadShapeFromFile(modelPath.string(), error);
  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  MOCHI_DEFER(context->ReleaseShape(params.shape));
  EXPECT_NE(scene->CreateSoftActor(params, error), nullptr);
  EXPECT_TRUE(error.IsOK()) << error.GetDescription();

  fs::path const vtkPath = _root / "cube.vtk";
  processing::WriteVtkFile(vtkPath.string(), *model.mesh, error);
  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  auto const actualSize = static_cast<double>(fs::file_size(vtkPath));
  auto const estimatedSize = static_cast<double>(processing::EstimateVtkSizeBytes(*model.mesh));
  EXPECT_NEAR(estimatedSize, actualSize, kFileSizeEstimateTolerance * actualSize)
      << "estimated " << estimatedSize << " bytes for a " << actualSize << "-byte file";
}

TEST_F(ProcessCommandTest, ZeroAxisSrtFailsOnlyForTetrahedralMeshes) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const surfacePipeline = _root / "surface" / "cube.StudioProcessing.json";
  WriteModifiers(surfacePipeline, {kSourceFromModelViewer, kZeroAxisSrt});
  fs::path const tetPipeline = _root / "tet" / "cube.StudioProcessing.json";
  WriteModifiers(tetPipeline, {kSourceFromModelViewer, kTetrahedralize, kZeroAxisSrt});
  PipelineModelSlots const slots{.cadPath = (_root / "cad" / "cube.stl").string()};

  mochi::Error surfaceError;
  ProcessReport const surface = RunProcessingPipeline(
      {.pipelinePath = surfacePipeline.string(), .slots = slots}, surfaceError);
  mochi::Error tetError;
  ProcessReport const tet =
      RunProcessingPipeline({.pipelinePath = tetPipeline.string(), .slots = slots}, tetError);

  ASSERT_TRUE(surfaceError.IsOK()) << surfaceError.GetDescription();
  ASSERT_TRUE(tetError.IsOK()) << tetError.GetDescription();
  ASSERT_TRUE(surface.succeeded);
  ASSERT_EQ(surface.stages.size(), 2u);
  EXPECT_TRUE(surface.stages[1].succeeded);
  EXPECT_EQ(surface.stages[1].triangles, kCubeTriangles);
  EXPECT_FALSE(tet.succeeded);
  ASSERT_EQ(tet.stages.size(), 3u);
  EXPECT_TRUE(tet.stages[1].succeeded);
  EXPECT_FALSE(tet.stages[2].succeeded);
  EXPECT_NE(tet.stages[2].error.find("finite, nonzero"), std::string::npos) << tet.stages[2].error;
}

TEST_F(ProcessCommandTest, NonFiniteTetSrtFailsItsStage) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WriteModifiers(pipeline, {kSourceFromModelViewer, kTetrahedralize, kNonFiniteSrt});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_FALSE(report.succeeded);
  ASSERT_EQ(report.stages.size(), 3u);
  EXPECT_TRUE(report.stages[1].succeeded);
  EXPECT_FALSE(report.stages[2].succeeded);
  EXPECT_NE(report.stages[2].error.find("non-finite"), std::string::npos) << report.stages[2].error;
}

TEST_F(ProcessCommandTest, MirroredTetSrtRestoresPositiveOrientation) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WriteModifiers(pipeline, {kSourceFromModelViewer, kTetrahedralize, kMirroredSrt, kNoSdfExport});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  ASSERT_TRUE(report.succeeded);
  ASSERT_EQ(report.stages.size(), 4u);
  ASSERT_TRUE(report.stages[2].tetMesh.has_value());
  mochi::ModelData const model =
      mochi::model_utils::LoadFromFile((_root / "collision" / "cube.mochi.h5").string(), error);
  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  ASSERT_TRUE(model.mesh.has_value());
  ASSERT_EQ(model.mesh->nodesPerElement, 4);
  mochi::real maxAbsX = 0;
  for (int tet = 0; tet < model.mesh->GetNumElements(); ++tet) {
    EXPECT_GT(TetSignedVolume(*model.mesh, tet), mochi::real{0}) << "tetrahedron " << tet;
  }
  for (int node = 0; node < model.mesh->GetNumNodes(); ++node) {
    maxAbsX = std::max(maxAbsX, std::abs(NodePosition(*model.mesh, node)[0]));
  }
  EXPECT_NEAR(maxAbsX, 2 * kHalfSide, 1e-5f);
}

TEST_F(ProcessCommandTest, NoSdfExportEmbedsAContactSkinInTheTetMesh) {
  std::array<float, 3> const center{0.01f, 0.02f, 0.03f};
  WriteCubeStl(_root / "cad" / "cube.stl", center);
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WriteModifiers(
      pipeline, {kSourceFromModelViewer, kTetrahedralize, kEmbedSourceAsContactSkin, kNoSdfExport});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_TRUE(report.succeeded);
  fs::path const modelPath = _root / "collision" / "cube.mochi.h5";
  mochi::ModelData const model = mochi::model_utils::LoadFromFile(modelPath.string(), error);
  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  ASSERT_TRUE(model.mesh.has_value());
  EXPECT_EQ(model.mesh->nodesPerElement, 4);
  ASSERT_TRUE(model.contactSkinMesh.has_value());
  EXPECT_EQ(model.contactSkinMesh->GetNumElements(), kCubeTriangles);
  ASSERT_TRUE(model.contactSkinMesh->skinning.has_value());
  EXPECT_EQ(
      model.contactSkinMesh->skinning->indices.size(),
      4 * static_cast<std::size_t>(model.contactSkinMesh->GetNumNodes()));
  for (int axis = 0; axis < 3; ++axis) {
    mochi::real lowest = std::numeric_limits<mochi::real>::max();
    mochi::real highest = std::numeric_limits<mochi::real>::lowest();
    for (int node = 0; node < model.contactSkinMesh->GetNumNodes(); ++node) {
      mochi::real const coordinate = NodePosition(*model.contactSkinMesh, node)[axis];
      lowest = std::min(lowest, coordinate);
      highest = std::max(highest, coordinate);
    }
    EXPECT_NEAR((lowest + highest) / 2, center[axis], 1e-6f);
  }
  mochi::ModelData meshOnly = model;
  meshOnly.contactSkinMesh.reset();
  EXPECT_GT(
      processing::EstimateMochiH5SizeBytes(model), processing::EstimateMochiH5SizeBytes(meshOnly));

  mochi::Context* const context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  mochi::Scene* const scene = context->CreateScene("contact_skin_export");
  MOCHI_DEFER(context->DestroyScene(scene));
  mochi::SoftActorParams params;
  params.shape = context->LoadShapeFromFile(modelPath.string(), error);
  params.useContactSkin = true;
  MOCHI_DEFER(context->ReleaseShape(params.shape));
  EXPECT_NE(scene->CreateSoftActor(params, error), nullptr);
  EXPECT_TRUE(error.IsOK()) << error.GetDescription();
}

// The usual soft-model pipeline: the wrapped surface, embedded in the tetrahedral mesh that fills
// it, as its contact skin.
TEST_F(ProcessCommandTest, AWrappedSurfaceIsEmbeddedInItsTetrahedralization) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WriteModifiers(
      pipeline,
      {kSourceFromModelViewer,
       R"({"modifier": "Wrap Mesh", "method": "Alpha Wrap", "enabled": true, "collapsed": true,
           "properties": {}})",
       R"({"modifier": "Tetrahedralize", "method": "Delaunay Refinement", "enabled": true,
           "collapsed": true, "properties": {}})",
       R"({"modifier": "Embed Contact Skin", "method": "Upstream Surface", "enabled": true,
           "collapsed": true, "properties": {"referenceIndex": 1}})",
       kNoSdfExport});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  for (ProcessStageReport const& stage : report.stages) {
    EXPECT_TRUE(stage.succeeded) << stage.label << ": " << stage.error;
  }
  mochi::ModelData const model =
      mochi::model_utils::LoadFromFile((_root / "collision" / "cube.mochi.h5").string(), error);
  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_TRUE(model.contactSkinMesh.has_value());
}

TEST_F(ProcessCommandTest, SdfExportEmbedsAContactSkinInTheTetMesh) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WriteModifiers(
      pipeline, {kSourceFromModelViewer, kTetrahedralize, kEmbedSourceAsContactSkin, kSdfExport});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  ASSERT_TRUE(report.succeeded);
  mochi::ModelData const model =
      mochi::model_utils::LoadFromFile((_root / "collision" / "cube.mochi.h5").string(), error);
  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_TRUE(model.sdf.has_value());
  EXPECT_TRUE(model.contactSkinMesh.has_value());
}

TEST_F(ProcessCommandTest, AnSrtBelowAnEmbedIsRejected) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WriteModifiers(
      pipeline,
      {kSourceFromModelViewer,
       kTetrahedralize,
       kEmbedSourceAsContactSkin,
       kTranslatedSrt,
       kNoSdfExport});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_FALSE(report.succeeded);
  ASSERT_EQ(report.stages.size(), 5u);
  EXPECT_TRUE(report.stages[2].succeeded) << report.stages[2].error;
  EXPECT_FALSE(report.stages[3].succeeded);
  EXPECT_NE(report.stages[3].error.find("embedded"), std::string::npos) << report.stages[3].error;
}

TEST_F(ProcessCommandTest, AContactSkinAndAVisualMeshAreEmbeddedTogether) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WriteModifiers(
      pipeline,
      {kSourceFromModelViewer,
       kTetrahedralize,
       kEmbedSourceAsContactSkin,
       kEmbedSourceAsVisualMesh,
       kNoSdfExport});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  ASSERT_TRUE(report.succeeded);
  mochi::ModelData const model =
      mochi::model_utils::LoadFromFile((_root / "collision" / "cube.mochi.h5").string(), error);
  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  ASSERT_TRUE(model.contactSkinMesh.has_value());
  ASSERT_TRUE(model.visualMesh.has_value());
  EXPECT_EQ(model.visualMesh->coordinates, model.contactSkinMesh->coordinates);
  EXPECT_TRUE(model.visualMesh->skinning.has_value());
}

TEST_F(ProcessCommandTest, AMeshIsEmbeddedOnlyOncePerRole) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WriteModifiers(
      pipeline,
      {kSourceFromModelViewer,
       kTetrahedralize,
       kEmbedSourceAsContactSkin,
       kEmbedSourceAsContactSkin,
       kNoSdfExport});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_FALSE(report.succeeded);
  ASSERT_EQ(report.stages.size(), 5u);
  EXPECT_FALSE(report.stages[3].succeeded);
  EXPECT_NE(report.stages[3].error.find("already"), std::string::npos) << report.stages[3].error;
}

TEST_F(ProcessCommandTest, AContactSkinNeedsATetMesh) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WriteModifiers(pipeline, {kSourceFromModelViewer, kEmbedSourceAsContactSkin, kNoSdfExport});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  EXPECT_FALSE(report.succeeded);
  ASSERT_EQ(report.stages.size(), 3u);
  EXPECT_FALSE(report.stages[1].succeeded);
  EXPECT_NE(report.stages[1].error.find("Tetrahedralize"), std::string::npos)
      << report.stages[1].error;
}

// The render model's own source, ahead of the CAD source the tetrahedral mesh is built from,
// provides the surface to embed vertex for vertex in file order.
TEST_F(ProcessCommandTest, AnUpstreamRenderModelIsEmbeddedVertexForVertex) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  std::vector<mochi_renderer::MeshSection> const primitives =
      WriteCubeGlb(_root / "render" / "cube.glb");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WriteModifiers(
      pipeline,
      {kRenderSourceFromModelViewer,
       kSourceFromModelViewer,
       kTetrahedralize,
       kEmbedSourceAsVisualMesh,
       kNoSdfExport});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_TRUE(report.succeeded);
  fs::path const modelPath = _root / "collision" / "cube.mochi.h5";
  mochi::ModelData const model = mochi::model_utils::LoadFromFile(modelPath.string(), error);
  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  ASSERT_TRUE(model.visualMesh.has_value());
  // The render model's vertices, in file order, converted from its Y-up space to Mochi's.
  mochi::CoordinateSpaceConverter const renderToMochi(
      mochi_renderer::RenderSpace(), mochi::CoordinateSpace::Default());
  std::vector<mochi::real> coordinates;
  std::vector<int> connectivity;
  for (mochi_renderer::MeshSection const& primitive : primitives) {
    int const base = static_cast<int>(coordinates.size() / 3);
    for (std::size_t i = 0; i < primitive.positions.size(); i += 3) {
      mochi::Real3 const p = renderToMochi.TranslationToOutput(
          mochi::Real3{
              primitive.positions[i], primitive.positions[i + 1], primitive.positions[i + 2]});
      coordinates.insert(coordinates.end(), p.begin(), p.end());
    }
    for (int const index : primitive.indices) {
      connectivity.push_back(base + index);
    }
  }
  EXPECT_EQ(
      std::vector<int>(
          model.visualMesh->connectivity.begin(), model.visualMesh->connectivity.end()),
      connectivity);
  ASSERT_EQ(model.visualMesh->coordinates.size(), coordinates.size());
  for (std::size_t i = 0; i < coordinates.size(); ++i) {
    EXPECT_NEAR(model.visualMesh->coordinates[i], coordinates[i], 1e-6f) << i;
  }
  ASSERT_TRUE(model.visualMesh->skinning.has_value());
  EXPECT_EQ(model.visualMesh->skinning->indices.size(), coordinates.size() / 3 * 4);

  mochi::Context* const context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  mochi::Scene* const scene = context->CreateScene("visual_mesh_export");
  MOCHI_DEFER(context->DestroyScene(scene));
  mochi::SoftActorParams params;
  params.shape = context->LoadShapeFromFile(modelPath.string(), error);
  MOCHI_DEFER(context->ReleaseShape(params.shape));
  EXPECT_NE(scene->CreateSoftActor(params, error), nullptr);
  EXPECT_TRUE(error.IsOK()) << error.GetDescription();
}

TEST_F(ProcessCommandTest, SdfExportOfATetMeshWritesBoth) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WriteModifiers(pipeline, {kSourceFromModelViewer, kTetrahedralize, kSdfExport});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_TRUE(report.succeeded);
  ASSERT_EQ(report.stages.size(), 3u);
  ASSERT_TRUE(report.stages[1].tetMesh.has_value());
  ASSERT_TRUE(report.stages[2].sdfGrid.has_value());
  mochi::ModelData const model =
      mochi::model_utils::LoadFromFile((_root / "collision" / "cube.mochi.h5").string(), error);
  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  ASSERT_TRUE(model.mesh.has_value());
  EXPECT_EQ(model.mesh->nodesPerElement, 4);
  EXPECT_EQ(model.mesh->GetNumNodes(), (*report.stages[1].tetMesh)[0]);
  EXPECT_EQ(model.mesh->GetNumElements(), (*report.stages[1].tetMesh)[1]);
  ExpectCubeSdf(model, *report.stages[2].sdfGrid);
}

TEST_F(ProcessCommandTest, ASurfaceStepAfterTetrahedralizeFailsItsStage) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WriteModifiers(pipeline, {kSourceFromModelViewer, kTetrahedralize, kMakeWatertight});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  EXPECT_FALSE(report.succeeded);
  ASSERT_EQ(report.stages.size(), 3u);
  EXPECT_TRUE(report.stages[1].succeeded);
  EXPECT_FALSE(report.stages[2].succeeded);
  EXPECT_NE(report.stages[2].error.find("Tetrahedralize"), std::string::npos)
      << report.stages[2].error;
}

TEST_F(ProcessCommandTest, ExtractSurfaceLetsASurfaceStepFollowTetrahedralize) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WriteModifiers(
      pipeline,
      {kSourceFromModelViewer, kTetrahedralize, kExtractSurface, kMakeWatertight, kNoSdfExport});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  ASSERT_EQ(report.stages.size(), 5u);
  for (ProcessStageReport const& stage : report.stages) {
    EXPECT_TRUE(stage.succeeded) << stage.label << ": " << stage.error;
  }
  // The tet stage reports its boundary, which the extracted surface must match.
  ASSERT_TRUE(report.stages[1].tetMesh.has_value());
  EXPECT_FALSE(report.stages[2].tetMesh.has_value());
  EXPECT_EQ(report.stages[2].vertices, report.stages[1].vertices);
  EXPECT_EQ(report.stages[2].triangles, report.stages[1].triangles);
  mochi::ModelData const model =
      mochi::model_utils::LoadFromFile((_root / "collision" / "cube.mochi.h5").string(), error);
  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  ASSERT_TRUE(model.mesh.has_value());
  EXPECT_EQ(model.mesh->nodesPerElement, 3);
  EXPECT_EQ(model.mesh->GetNumElements(), report.stages[3].triangles);
}

TEST_F(ProcessCommandTest, ExtractSurfaceOfATriangleSurfaceFailsItsStage) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WriteModifiers(pipeline, {kSourceFromModelViewer, kExtractSurface});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  EXPECT_FALSE(report.succeeded);
  ASSERT_EQ(report.stages.size(), 2u);
  EXPECT_TRUE(report.stages[0].succeeded);
  EXPECT_FALSE(report.stages[1].succeeded);
  EXPECT_NE(report.stages[1].error.find("Tetrahedralize"), std::string::npos)
      << report.stages[1].error;
}

TEST_F(ProcessCommandTest, OutDirCollectsTheExports) {
  WriteCubeStl(_root / "asset" / "cad" / "cube.stl");
  fs::path const pipeline = _root / "asset" / "intermediates" / "cube.StudioProcessing.json";
  WritePipeline(pipeline, kSourceFromModelViewer);
  fs::path const out = _root / "out";

  mochi::Error error;
  ProcessReport const report =
      RunProcessingPipeline({.pipelinePath = pipeline.string(), .outDir = out.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_TRUE(report.succeeded);
  EXPECT_TRUE(fs::is_regular_file(out / "cube.mochi.h5"));
  EXPECT_TRUE(fs::is_regular_file(out / "cube.glb"));
  ASSERT_EQ(report.stages.size(), 3u);
  EXPECT_EQ(
      report.stages[1].pipelineExportPath, (_root / "asset" / "render" / "cube.glb").string());
  EXPECT_EQ(
      report.stages[2].pipelineExportPath,
      (_root / "asset" / "collision" / "cube.mochi.h5").string());
  EXPECT_FALSE(fs::exists(_root / "asset" / "collision"));
  EXPECT_FALSE(fs::exists(_root / "asset" / "render"));
}

TEST_F(ProcessCommandTest, ExportsThatWouldWriteTheSameFileAreRefused) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  // Two mesh exports with different folders but the same file name.
  fs::create_directories(pipeline.parent_path());
  std::ofstream(pipeline) << R"({
  "version": 1,
  "modifiers": [
    )" << kSourceFromModelViewer
                          << R"(,
    {"modifier": "Export Mesh File", "method": "Mesh File", "enabled": true, "collapsed": true,
     "properties": {"color": [0.2, 0.4, 0.6]}},
    {"modifier": "Export Mesh File", "method": "Mesh File", "enabled": true, "collapsed": true,
     "properties": {"path": "../preview/cube.glb", "color": [0.2, 0.4, 0.6]}}
  ]
})";
  fs::path const out = _root / "out";

  mochi::Error separateError;
  ProcessReport const separate =
      RunProcessingPipeline({.pipelinePath = pipeline.string()}, separateError);
  mochi::Error collectedError;
  ProcessReport const collected = RunProcessingPipeline(
      {.pipelinePath = pipeline.string(), .outDir = out.string()}, collectedError);

  ASSERT_TRUE(separateError.IsOK()) << separateError.GetDescription();
  EXPECT_TRUE(separate.succeeded);
  EXPECT_TRUE(separate.collidingExportPaths.empty());
  EXPECT_TRUE(fs::is_regular_file(_root / "render" / "cube.glb"));
  EXPECT_TRUE(fs::is_regular_file(_root / "preview" / "cube.glb"));
  EXPECT_FALSE(collectedError.IsOK());
  EXPECT_TRUE(collected.stages.empty());
  EXPECT_EQ(collected.collidingExportPaths, std::vector<std::string>{(out / "cube.glb").string()});
  EXPECT_FALSE(fs::exists(out));
}

TEST_F(ProcessCommandTest, OverrideLetsTheLaterOfTwoCollidingExportsWin) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  // Two mesh exports to the same file, the second with a color the first does not use.
  fs::create_directories(pipeline.parent_path());
  std::ofstream(pipeline) << R"({
  "version": 1,
  "modifiers": [
    )" << kSourceFromModelViewer
                          << R"(,
    {"modifier": "Export Mesh File", "method": "Mesh File", "enabled": true, "collapsed": true,
     "properties": {"path": "../render/cube.glb", "color": [0.2, 0.4, 0.6]}},
    {"modifier": "Export Mesh File", "method": "Mesh File", "enabled": true, "collapsed": true,
     "properties": {"path": "../render/cube.glb", "color": [0.9, 0.1, 0.1]}}
  ]
})";

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline(
      {.pipelinePath = pipeline.string(), .allowCollidingExports = true}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_TRUE(report.succeeded);
  ASSERT_EQ(report.collidingExportPaths.size(), 1u);
  EXPECT_TRUE(fs::equivalent(report.collidingExportPaths[0], _root / "render" / "cube.glb"));
  std::vector<mochi_renderer::MeshSection> const sections =
      mochi_renderer::ReadGlbFromFile((_root / "render" / "cube.glb").string().c_str());
  ASSERT_EQ(sections.size(), 1u);
  EXPECT_NEAR(sections[0].baseColor[0], 0.9f, 1e-3f);
}

TEST_F(ProcessCommandTest, CadOptionSuppliesAModelNamedDifferentlyFromThePipeline) {
  WriteCubeStl(_root / "cad" / "part_link.stl");
  fs::path const pipeline = _root / "intermediates" / "link.StudioProcessing.json";
  WritePipeline(pipeline, kSourceFromModelViewer);

  mochi::Error unresolved;
  ProcessReport const withoutCad =
      RunProcessingPipeline({.pipelinePath = pipeline.string()}, unresolved);
  EXPECT_FALSE(unresolved.IsOK());
  EXPECT_TRUE(withoutCad.stages.empty());

  mochi::Error error;
  ProcessReport const withCad = RunProcessingPipeline(
      {.pipelinePath = pipeline.string(),
       .slots = {.cadPath = (_root / "cad" / "part_link.stl").string()}},
      error);
  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_TRUE(withCad.succeeded);
  // Auto exports are named after the model the source read, not after the pipeline.
  EXPECT_TRUE(fs::is_regular_file(_root / "collision" / "part_link.mochi.h5"));
}

TEST_F(ProcessCommandTest, AFailedSourceFailsEveryStageAndWritesNothing) {
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WritePipeline(
      pipeline,
      R"({"modifier": "Source from CAD Model", "method": "From File", "enabled": true,
          "collapsed": true, "properties": {"path": "../cad/missing.stl"}})");

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_FALSE(report.succeeded);
  ASSERT_EQ(report.stages.size(), 3u);
  for (ProcessStageReport const& stage : report.stages) {
    EXPECT_FALSE(stage.succeeded) << stage.label;
    EXPECT_FALSE(stage.error.empty()) << stage.label;
  }
  EXPECT_FALSE(fs::exists(_root / "collision"));
  EXPECT_FALSE(fs::exists(_root / "render"));
}

TEST_F(ProcessCommandTest, AnExportThatCannotBeWrittenFailsOnlyItsStage) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WritePipeline(pipeline, kSourceFromModelViewer);
  std::ofstream(_root / "render") << "a file where the mesh export's folder should be";

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_FALSE(report.succeeded);
  ASSERT_EQ(report.stages.size(), 3u);
  EXPECT_TRUE(report.stages[0].succeeded);
  EXPECT_FALSE(report.stages[1].succeeded);
  EXPECT_TRUE(report.stages[1].exportedPath.empty());
  EXPECT_FALSE(report.stages[1].error.empty());
  EXPECT_NE(report.stages[1].error, "produced no mesh");
  EXPECT_TRUE(report.stages[2].succeeded);
  EXPECT_TRUE(fs::is_regular_file(_root / "collision" / "cube.mochi.h5"));
}

constexpr char const* kMeshExportOnAuto =
    R"({"modifier": "Export Mesh File", "method": "Mesh File", "enabled": true,
        "collapsed": true, "properties": {"color": [0.2, 0.4, 0.6]}})";

TEST_F(ProcessCommandTest, AnUnrecognizedModifierStopsTheRun) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  WriteModifiers(
      pipeline,
      {kSourceFromModelViewer,
       R"({"modifier": "Frobnicate Mesh", "method": "Wobble", "enabled": true,
           "collapsed": true, "properties": {"amount": 3}})",
       kMeshExportOnAuto});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  EXPECT_FALSE(error.IsOK());
  EXPECT_EQ(report.unrecognizedModifiers, std::vector<std::string>{"Frobnicate Mesh / Wobble"});
  EXPECT_TRUE(report.stages.empty());
  EXPECT_FALSE(fs::exists(_root / "render"));
}

TEST_F(ProcessCommandTest, AnExportWithNoFilePathFailsItsStage) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  // Auto exports are named after the first modifier's model; this asset has no mochi model.
  WriteModifiers(
      pipeline,
      {R"({"modifier": "Source from Mochi Model", "method": "From Model Viewer",
           "enabled": false, "collapsed": true, "properties": {}})",
       kSourceFromModelViewer,
       kMeshExportOnAuto});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_FALSE(report.succeeded);
  ASSERT_EQ(report.stages.size(), 2u);
  EXPECT_TRUE(report.stages[0].succeeded);
  EXPECT_FALSE(report.stages[1].succeeded);
  EXPECT_NE(report.stages[1].error.find("no file path"), std::string::npos)
      << report.stages[1].error;
}

TEST_F(ProcessCommandTest, ExportsOutsideTheChainDoNotCollide) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  // The first segment's source is off, so its export never runs.
  WriteModifiers(
      pipeline,
      {R"({"modifier": "Source from CAD Model", "method": "From Model Viewer", "enabled": false,
           "collapsed": true, "properties": {}})",
       kMeshExportOnAuto,
       kSourceFromModelViewer,
       kMeshExportOnAuto});

  mochi::Error error;
  ProcessReport const report = RunProcessingPipeline({.pipelinePath = pipeline.string()}, error);

  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_TRUE(report.succeeded);
  ASSERT_EQ(report.stages.size(), 2u);
  EXPECT_TRUE(fs::equivalent(report.stages[1].exportedPath, _root / "render" / "cube.glb"));
}

TEST_F(ProcessCommandTest, OutRefusesAPipelineWhoseSourceReadsAnEarlierExport) {
  WriteCubeStl(_root / "cad" / "cube.stl");
  fs::path const pipeline = _root / "intermediates" / "cube.StudioProcessing.json";
  // A first run leaves render/cube.glb, which the second segment's source then reads.
  WriteModifiers(pipeline, {kSourceFromModelViewer, kMeshExportOnAuto});
  mochi::Error firstError;
  ASSERT_TRUE(RunProcessingPipeline({.pipelinePath = pipeline.string()}, firstError).succeeded)
      << firstError.GetDescription();
  WriteModifiers(
      pipeline,
      {kSourceFromModelViewer,
       kMeshExportOnAuto,
       R"({"modifier": "Source from Render Model", "method": "From Model Viewer",
           "enabled": true, "collapsed": true, "properties": {}})",
       R"({"modifier": "Export Mesh File", "method": "Mesh File", "enabled": true,
           "collapsed": true, "properties": {"path": "../preview/cube.glb"}})"});
  fs::path const out = _root / "out";

  mochi::Error collectedError;
  ProcessReport const collected = RunProcessingPipeline(
      {.pipelinePath = pipeline.string(), .outDir = out.string()}, collectedError);
  mochi::Error inPlaceError;
  ProcessReport const inPlace =
      RunProcessingPipeline({.pipelinePath = pipeline.string()}, inPlaceError);

  EXPECT_FALSE(collectedError.IsOK());
  EXPECT_TRUE(fs::equivalent(collected.outConflictPath, _root / "render" / "cube.glb"));
  EXPECT_FALSE(fs::exists(out));
  ASSERT_TRUE(inPlaceError.IsOK()) << inPlaceError.GetDescription();
  EXPECT_TRUE(inPlace.succeeded);
  EXPECT_TRUE(fs::is_regular_file(_root / "preview" / "cube.glb"));
}

TEST_F(ProcessCommandTest, AnAssetInAFolderNamedLikeARoleFolderFindsItsOwnModels) {
  // The asset sits in a folder named "cad"; a stray model beside its role folders must not win.
  fs::path const asset = _root / "cad" / "part";
  WriteCubeStl(asset / "cad" / "cube.stl");
  WriteCubeStl(asset / "cube.stl");
  fs::path const pipeline = asset / "intermediates" / "cube.StudioProcessing.json";
  WritePipeline(pipeline, kSourceFromModelViewer);

  PipelineModelSlots const slots = FindPipelineModelSlots(pipeline);

  EXPECT_TRUE(fs::equivalent(slots.cadPath, asset / "cad" / "cube.stl")) << slots.cadPath;
}

TEST_F(ProcessCommandTest, ShadowedModelPipelineFillsOnlyThatModelsSlot) {
  fs::path const model = _root / "render" / "internal" / "part.glb";
  fs::create_directories(model.parent_path());
  std::ofstream(model) << "glb";
  fs::create_directories(_root / "cad");
  std::ofstream(_root / "cad" / "part.step") << "step";

  PipelineModelSlots const slots = FindPipelineModelSlots(
      _root / "intermediates" / "render" / "internal" / "part.glb.StudioProcessing.json");

  EXPECT_TRUE(fs::equivalent(slots.renderPath, model));
  EXPECT_EQ(slots.cadPath, "");
  EXPECT_EQ(slots.mochiPath, "");
}
