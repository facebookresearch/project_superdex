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

// Covers the operations on a whole modifier stack
// (meshing/processing_modifiers/processing_stack.h). The stacks are built from a test method, so no
// mesh operation or file is involved.

#include "meshing/processing_modifiers/processing_stack.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace superdex::studio;

namespace {

using RunFunction = std::function<mochi::MeshData(
    mochi::MeshData const& input,
    ModifierRunContext const& ctx,
    mochi::Error& error)>;

struct ModifierSpec {
  ModifierKind kind = ModifierKind::Transform;
  bool enabled = true;
  int referenceIndex = MeshProcessingMethod::kReferencePrecedingSource;
  bool needsReference = false;
  std::string exportPath;
  bool autoExportPath = false;
  std::string sourcePath; // the file a source reads; the context's CAD model when empty
  RunFunction run; // passes the input through when empty
  std::vector<mochi::MeshData>* saved = nullptr; // receives each SaveToFile input
  bool acceptsTets = false;
  bool acceptsEmbedded = false; // passes the meshes embedded in its input on
  std::optional<mochi::MeshData> contactSkin; // embedded by a stage that acceptsEmbedded
};

// A method that does whatever its spec says. Its Auto export path is the source path plus ".out".
class TestMethod : public MeshProcessingMethod {
 public:
  explicit TestMethod(ModifierSpec spec) : _spec(std::move(spec)) {}

  [[nodiscard]] char const* Name() const override {
    return "Test";
  }
  void ShowParams(ModifierGuiContext const& /*gui*/) override {}
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& ctx,
      mochi::Error& error) const override {
    return _spec.run ? _spec.run(input, ctx, error) : input;
  }
  StageMesh RunStage(StageMesh const& input, ModifierRunContext const& ctx, mochi::Error& error)
      const override {
    if (!_spec.acceptsEmbedded) {
      return MeshProcessingMethod::RunStage(input, ctx, error);
    }
    StageMesh output{.mesh = Run(input.mesh, ctx, error), .embedded = input.embedded};
    if (_spec.contactSkin.has_value()) {
      output.embedded.contactSkinMesh = _spec.contactSkin;
    }
    return output;
  }
  void SerializeProps(picojson::value& /*out*/) const override {}
  int DeserializeProps(picojson::value const& /*in*/) override {
    return 0;
  }
  [[nodiscard]] std::string PropsSignature(ModifierRunContext const& /*ctx*/) const override {
    return {};
  }
  [[nodiscard]] bool NeedsReferenceMesh() const override {
    return _spec.needsReference;
  }
  [[nodiscard]] bool AcceptsTetrahedralInput() const override {
    return _spec.acceptsTets;
  }
  [[nodiscard]] bool AcceptsEmbeddedMeshes() const override {
    return _spec.acceptsEmbedded;
  }
  [[nodiscard]] int ReferenceIndex() const override {
    return _spec.referenceIndex;
  }
  [[nodiscard]] std::string SourceFilePath(ModifierRunContext const& ctx) const override {
    return _spec.sourcePath.empty() ? ctx.cadFilePath : _spec.sourcePath;
  }
  [[nodiscard]] bool ProvidesFileExport() const override {
    return _spec.autoExportPath || !_spec.exportPath.empty();
  }
  [[nodiscard]] std::string ExportPath() const override {
    return _spec.exportPath;
  }
  void SaveToFile(mochi::MeshData const& input, mochi::Error& /*error*/) const override {
    _spec.saved->push_back(input);
  }
  void RefreshAutoExportPath(std::string const& sourceFilePath) override {
    if (_spec.autoExportPath) {
      _spec.exportPath = sourceFilePath + ".out";
    }
  }
  void OverrideExportPath(std::string const& path) override {
    _spec.exportPath = path;
    _spec.autoExportPath = false;
  }

 private:
  ModifierSpec _spec;
};

ModifierStack MakeStack(std::vector<ModifierSpec> const& specs) {
  ModifierStack stack;
  for (ModifierSpec const& spec : specs) {
    std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
    methods.push_back(std::make_unique<TestMethod>(spec));
    auto modifier = std::make_unique<MeshProcessingModifier>("Test", spec.kind, std::move(methods));
    modifier->enabled = spec.enabled;
    stack.push_back(std::move(modifier));
  }
  return stack;
}

// A single triangle whose coordinates all equal @p value, so a mesh's history can be read back
// from its first coordinate.
mochi::MeshData Triangle(mochi::real value) {
  mochi::MeshData mesh;
  mesh.nodesPerElement = 3;
  for (int i = 0; i < 9; ++i) {
    mesh.coordinates.push_back(value);
  }
  for (int i = 0; i < 3; ++i) {
    mesh.connectivity.push_back(i);
  }
  return mesh;
}

mochi::MeshData Tetrahedron() {
  mochi::MeshData mesh;
  mesh.nodesPerElement = 4;
  for (mochi::real const coordinate : {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}) {
    mesh.coordinates.push_back(coordinate);
  }
  for (int const node : {0, 1, 2, 3}) {
    mesh.connectivity.push_back(node);
  }
  return mesh;
}

mochi::MeshData ProduceTetrahedron(
    mochi::MeshData const& /*input*/,
    ModifierRunContext const& /*ctx*/,
    mochi::Error& /*error*/) {
  return Tetrahedron();
}

RunFunction ProduceTriangle(mochi::real value) {
  return [value](
             mochi::MeshData const& /*input*/,
             ModifierRunContext const& /*ctx*/,
             mochi::Error& /*error*/) { return Triangle(value); };
}

mochi::MeshData
AddOne(mochi::MeshData const& input, ModifierRunContext const& /*ctx*/, mochi::Error& /*error*/) {
  mochi::MeshData out = input;
  for (mochi::real& coordinate : out.coordinates) {
    coordinate += 1;
  }
  return out;
}

mochi::MeshData
Fail(mochi::MeshData const& /*input*/, ModifierRunContext const& /*ctx*/, mochi::Error& error) {
  MOCHI_ERROR_SET(error, "test failure");
  return {};
}

mochi::MeshData ReturnReference(
    mochi::MeshData const& /*input*/,
    ModifierRunContext const& ctx,
    mochi::Error& /*error*/) {
  return ctx.referenceMesh;
}

mochi::MeshData NoCurrentOutput(std::size_t /*modifierIndex*/) {
  return {};
}

// StackRun's current outputs from @p mesh, each modifier's mesh, with nothing embedded in them.
std::function<StageMesh(std::size_t)> CurrentMeshes(
    std::function<mochi::MeshData(std::size_t)> mesh) {
  return [mesh = std::move(mesh)](std::size_t index) { return StageMesh{.mesh = mesh(index)}; };
}

mochi::MeshData SameReferenceMesh(mochi::MeshData const& output) {
  return output;
}

constexpr ModifierKind kSource = ModifierKind::Source;
constexpr ModifierKind kTransform = ModifierKind::Transform;
constexpr ModifierKind kExport = ModifierKind::Export;

} // namespace

TEST(ProcessingStackTest, GenerationChainClimbsEnabledModifiersToTheNearestSource) {
  ModifierStack const stack = MakeStack(
      {{.kind = kSource},
       {.kind = kSource},
       {.kind = kTransform, .enabled = false},
       {.kind = kTransform},
       {.kind = kExport}});

  std::vector<std::size_t> const expected{1, 3, 4};
  EXPECT_EQ(BuildGenerationChain(stack, 4), expected);
}

TEST(ProcessingStackTest, GenerationChainIsEmptyWithoutAnUpstreamSource) {
  ModifierStack const stack =
      MakeStack({{.kind = kSource, .enabled = false}, {.kind = kTransform}, {.kind = kExport}});

  EXPECT_TRUE(BuildGenerationChain(stack, 2).empty());
  EXPECT_TRUE(BuildGenerationChain(stack, 7).empty());
}

TEST(ProcessingStackTest, FullGenerationChainSpansEverySegmentFromTheFirstSource) {
  ModifierStack const stack = MakeStack(
      {{.kind = kTransform},
       {.kind = kSource},
       {.kind = kTransform, .enabled = false},
       {.kind = kExport},
       {.kind = kSource},
       {.kind = kExport}});

  std::vector<std::size_t> const expected{1, 3, 4, 5};
  EXPECT_EQ(BuildFullGenerationChain(stack), expected);
}

TEST(ProcessingStackTest, ReferenceIsAnUpstreamModifierOrTheNearestPrecedingSource) {
  ModifierStack const stack = MakeStack(
      {{.kind = kSource},
       {.kind = kTransform},
       {.kind = kSource},
       {.kind = kTransform, .referenceIndex = 1},
       {.kind = kTransform},
       {.kind = kTransform, .referenceIndex = 5}});

  EXPECT_EQ(ReferenceModifierIndex(stack, 3), 1); // explicit, upstream
  EXPECT_EQ(ReferenceModifierIndex(stack, 4), 2); // nearest preceding source
  EXPECT_EQ(ReferenceModifierIndex(stack, 5), -1); // explicit, but not upstream
  EXPECT_EQ(ReferenceModifierIndex(stack, 0), -1); // nothing precedes the first modifier
}

TEST(ProcessingStackTest, SourceFilePathComesFromTheFirstModifier) {
  ModifierRunContext ctx;
  ctx.cadFilePath = "cad/part.step";

  EXPECT_EQ(StackSourceFilePath(MakeStack({}), ctx), "");
  EXPECT_EQ(
      StackSourceFilePath(MakeStack({{.kind = kSource}, {.kind = kExport}}), ctx), ctx.cadFilePath);
  EXPECT_EQ(
      StackSourceFilePath(
          MakeStack(
              {{.kind = kSource, .sourcePath = "first.step"},
               {.kind = kSource, .sourcePath = "second.step"}}),
          ctx),
      "first.step");
}

TEST(ProcessingStackTest, RefreshAutoExportPathsUpdatesOnlyAutoExports) {
  ModifierStack stack = MakeStack(
      {{.kind = kSource},
       {.kind = kExport, .autoExportPath = true},
       {.kind = kExport, .exportPath = "chosen.glb"}});

  RefreshAutoExportPaths(stack, "cad/part.step");

  EXPECT_EQ(stack[1]->ExportPath(), "cad/part.step.out");
  EXPECT_EQ(stack[2]->ExportPath(), "chosen.glb");
}

TEST(ProcessingStackTest, RedirectExportsKeepsFileNamesInTheNewFolder) {
  ModifierStack stack = MakeStack(
      {{.kind = kSource},
       {.kind = kExport, .autoExportPath = true},
       {.kind = kExport, .exportPath = "render/chosen.glb"},
       {.kind = kExport, .enabled = false, .exportPath = "render/off.glb"}});
  RefreshAutoExportPaths(stack, "cad/part.step");

  RedirectExports(stack, "out");

  EXPECT_EQ(stack[0]->ExportPath(), "");
  EXPECT_EQ(stack[1]->ExportPath(), (std::filesystem::path("out") / "part.step.out").string());
  EXPECT_EQ(stack[2]->ExportPath(), (std::filesystem::path("out") / "chosen.glb").string());
  EXPECT_EQ(stack[3]->ExportPath(), "render/off.glb");
  // The redirected Auto export no longer follows the source.
  RefreshAutoExportPaths(stack, "cad/other.step");
  EXPECT_EQ(stack[1]->ExportPath(), (std::filesystem::path("out") / "part.step.out").string());
}

TEST(ProcessingStackTest, CollidingExportPathsCompareNormalizedPathsOfEnabledExports) {
  ModifierStack const stack = MakeStack(
      {{.kind = kSource},
       {.kind = kExport, .exportPath = "render/part.glb"},
       {.kind = kExport, .exportPath = "render/../render/part.glb"},
       {.kind = kExport, .enabled = false, .exportPath = "render/part.glb"},
       {.kind = kExport, .exportPath = "collision/part.mochi.h5"}});

  std::vector<bool> const expected{false, true, true, false, false};
  EXPECT_EQ(FindCollidingExportPaths(stack), expected);
}

TEST(ProcessingStackTest, CollidingExportPathsIgnoreCase) {
  ModifierStack const stack = MakeStack(
      {{.kind = kSource},
       {.kind = kExport, .exportPath = "render/Part.glb"},
       {.kind = kExport, .exportPath = "render/part.glb"}});

  std::vector<bool> const expected{false, true, true};
  EXPECT_EQ(FindCollidingExportPaths(stack), expected);
}

TEST(ProcessingStackTest, StackRunFeedsEachStageThePreviousOutput) {
  std::vector<mochi::MeshData> saved;
  ModifierStack const stack = MakeStack(
      {{.kind = kSource, .run = ProduceTriangle(1)},
       {.kind = kTransform, .run = AddOne},
       {.kind = kExport, .exportPath = "out.glb", .saved = &saved}});
  StackRun run(
      stack, {0, 1, 2}, {}, CurrentMeshes(NoCurrentOutput), NoCurrentOutput, SameReferenceMesh);

  mochi::Error error;
  EXPECT_TRUE(run.Generate(0, error));
  EXPECT_TRUE(run.Generate(1, error));
  EXPECT_TRUE(run.Generate(2, error));
  EXPECT_TRUE(run.Export(2, error));

  EXPECT_EQ(run.Output(2).coordinates[0], 2);
  ASSERT_EQ(saved.size(), 1u);
  EXPECT_EQ(saved[0].coordinates[0], 2);
}

TEST(ProcessingStackTest, StackRunDoesNotExportAfterFailedRegeneration) {
  std::vector<mochi::MeshData> saved;
  ModifierStack const stack = MakeStack(
      {{.kind = kSource, .run = ProduceTriangle(1)},
       {.kind = kExport, .exportPath = "out.glb", .run = Fail, .saved = &saved}});
  auto const currentOutput = [](std::size_t index) {
    return Triangle(static_cast<mochi::real>(10 + index));
  };
  StackRun run(stack, {0, 1}, {}, CurrentMeshes(currentOutput), NoCurrentOutput, SameReferenceMesh);

  mochi::Error sourceError;
  mochi::Error generationError;
  mochi::Error exportError;
  EXPECT_TRUE(run.Generate(0, sourceError));
  EXPECT_FALSE(run.Generate(1, generationError));
  EXPECT_FALSE(run.Export(1, exportError));

  EXPECT_FALSE(generationError.IsOK());
  EXPECT_FALSE(exportError.IsOK());
  EXPECT_EQ(run.Output(1).GetNumElements(), 0);
  EXPECT_TRUE(saved.empty());
}

TEST(ProcessingStackTest, OnlyStagesThatAcceptTetsRunOnATetMesh) {
  ModifierStack const stack = MakeStack(
      {{.kind = kSource, .run = ProduceTetrahedron},
       {.kind = kTransform, .acceptsTets = true},
       {.kind = kTransform}});
  StackRun run(
      stack, {0, 1, 2}, {}, CurrentMeshes(NoCurrentOutput), NoCurrentOutput, SameReferenceMesh);

  mochi::Error acceptingError;
  mochi::Error rejectingError;
  EXPECT_TRUE(run.Generate(0, acceptingError));
  EXPECT_TRUE(run.Generate(1, acceptingError));
  EXPECT_FALSE(run.Generate(2, rejectingError));

  EXPECT_EQ(run.Output(1).nodesPerElement, 4);
  EXPECT_FALSE(rejectingError.IsOK());
  EXPECT_EQ(run.Output(2).GetNumElements(), 0);
}

TEST(ProcessingStackTest, SkippedCachedTetKeepsTetTypeChecksForRegeneratedStages) {
  ModifierStack const stack = MakeStack(
      {{.kind = kSource}, {.kind = kTransform, .acceptsTets = true}, {.kind = kTransform}});
  mochi::MeshData const cachedTet = Tetrahedron();
  StackRun run(
      stack,
      {0, 1, 2},
      {},
      CurrentMeshes(
          [&cachedTet](std::size_t index) { return index == 0 ? cachedTet : mochi::MeshData{}; }),
      NoCurrentOutput,
      SameReferenceMesh);

  mochi::Error acceptingError;
  mochi::Error rejectingError;
  EXPECT_TRUE(run.Generate(1, acceptingError));
  EXPECT_EQ(run.Output(1).nodesPerElement, 4);
  EXPECT_EQ(run.Output(1).connectivity, cachedTet.connectivity);
  EXPECT_FALSE(run.Generate(2, rejectingError));
  EXPECT_FALSE(rejectingError.IsOK());
  EXPECT_EQ(run.Output(2).GetNumElements(), 0);
}

TEST(ProcessingStackTest, StackRunFailureLeavesNothingForTheStagesBelow) {
  ModifierStack const stack = MakeStack(
      {{.kind = kSource, .run = ProduceTriangle(1)},
       {.kind = kTransform, .run = Fail},
       {.kind = kTransform, .run = AddOne}});
  auto const currentOutput = [](std::size_t index) {
    return Triangle(static_cast<mochi::real>(10 * index));
  };
  StackRun run(
      stack, {0, 1, 2}, {}, CurrentMeshes(currentOutput), currentOutput, SameReferenceMesh);

  mochi::Error sourceError;
  mochi::Error failError;
  mochi::Error belowError;
  EXPECT_TRUE(run.Generate(0, sourceError));
  EXPECT_FALSE(run.Generate(1, failError));
  EXPECT_FALSE(run.Generate(2, belowError));

  EXPECT_FALSE(failError.IsOK());
  EXPECT_FALSE(belowError.IsOK());
  EXPECT_EQ(run.Output(1).GetNumElements(), 0);
  EXPECT_EQ(run.Output(2).GetNumElements(), 0);
}

TEST(ProcessingStackTest, StackRunPassesOnTheCurrentOutputOfStagesItSkips) {
  ModifierStack const stack = MakeStack(
      {{.kind = kSource, .run = ProduceTriangle(1)},
       {.kind = kTransform, .run = AddOne},
       {.kind = kTransform, .run = AddOne}});
  auto const currentOutput = [](std::size_t index) {
    return Triangle(static_cast<mochi::real>(10 * index));
  };
  StackRun run(
      stack, {0, 1, 2}, {}, CurrentMeshes(currentOutput), currentOutput, SameReferenceMesh);

  mochi::Error error;
  EXPECT_TRUE(run.Generate(2, error));

  EXPECT_EQ(run.Output(2).coordinates[0], 11);
}

TEST(ProcessingStackTest, StackRunResolvesReferencesInsideAndOutsideTheChain) {
  ModifierStack const stack = MakeStack(
      {{.kind = kSource, .run = ProduceTriangle(1)},
       {.kind = kSource, .run = ProduceTriangle(2)},
       {.kind = kTransform, .referenceIndex = 0, .needsReference = true, .run = ReturnReference},
       {.kind = kTransform, .needsReference = true, .run = ReturnReference}});
  // The chain starts at the nearest source, index 1, so the explicit reference to index 0 is
  // outside it while the preceding-source reference of index 3 is inside it.
  mochi::MeshData const cachedTet = Tetrahedron();
  StackRun run(
      stack,
      BuildGenerationChain(stack, 3),
      {},
      CurrentMeshes(
          [&cachedTet](std::size_t index) { return index == 0 ? cachedTet : mochi::MeshData{}; }),
      [](std::size_t index) { return Triangle(static_cast<mochi::real>(100 + index)); },
      SameReferenceMesh);

  mochi::Error error;
  EXPECT_TRUE(run.Generate(0, error));
  EXPECT_TRUE(run.Generate(1, error));
  EXPECT_TRUE(run.Generate(2, error));

  EXPECT_EQ(run.Output(1).nodesPerElement, 3);
  EXPECT_EQ(run.Output(1).coordinates[0], 100); // modifier 0's current reference surface
  EXPECT_EQ(run.Output(2).coordinates[0], 2); // the chain's own source output
}

TEST(ProcessingStackTest, StackRunBuildsReferenceSurfacesForGeneratedTetStages) {
  ModifierStack const stack = MakeStack(
      {{.kind = kSource, .run = ProduceTetrahedron},
       {.kind = kSource, .run = ProduceTriangle(2)},
       {.kind = kTransform, .referenceIndex = 0, .needsReference = true, .run = ReturnReference}});
  StackRun run(
      stack,
      BuildFullGenerationChain(stack),
      {},
      CurrentMeshes(NoCurrentOutput),
      NoCurrentOutput,
      [](mochi::MeshData const& output) {
        return output.nodesPerElement == 4 ? Triangle(50) : output;
      });

  mochi::Error error;
  EXPECT_TRUE(run.Generate(0, error));
  EXPECT_TRUE(run.Generate(1, error));
  EXPECT_TRUE(run.Generate(2, error));

  EXPECT_EQ(run.Output(2).nodesPerElement, 3);
  EXPECT_EQ(run.Output(2).coordinates[0], 50);
}

TEST(ProcessingStackTest, EmbeddedMeshesReachOnlyStagesThatAcceptThem) {
  ModifierStack const stack = MakeStack(
      {{.kind = kSource, .run = ProduceTetrahedron},
       {.kind = kTransform,
        .acceptsTets = true,
        .acceptsEmbedded = true,
        .contactSkin = Triangle(7)},
       {.kind = kTransform, .acceptsTets = true, .acceptsEmbedded = true},
       {.kind = kTransform, .acceptsTets = true}});
  StackRun run(
      stack, {0, 1, 2, 3}, {}, CurrentMeshes(NoCurrentOutput), NoCurrentOutput, SameReferenceMesh);

  mochi::Error acceptingError;
  mochi::Error rejectingError;
  EXPECT_TRUE(run.Generate(0, acceptingError));
  EXPECT_TRUE(run.Generate(1, acceptingError));
  EXPECT_TRUE(run.Generate(2, acceptingError));
  EXPECT_FALSE(run.Generate(3, rejectingError));

  ASSERT_TRUE(run.Embedded(2).contactSkinMesh.has_value());
  EXPECT_EQ(run.Embedded(2).contactSkinMesh->coordinates[0], 7);
  EXPECT_FALSE(rejectingError.IsOK());
  EXPECT_EQ(run.Output(3).GetNumElements(), 0);
}

TEST(ProcessingStackTest, StackRunPassesOnTheEmbeddedMeshesOfStagesItSkips) {
  ModifierStack const stack = MakeStack(
      {{.kind = kSource, .run = ProduceTetrahedron},
       {.kind = kTransform, .acceptsTets = true, .acceptsEmbedded = true}});
  StackRun run(
      stack,
      {0, 1},
      {},
      [](std::size_t /*index*/) {
        StageMesh output{.mesh = Tetrahedron()};
        output.embedded.contactSkinMesh = Triangle(3);
        return output;
      },
      NoCurrentOutput,
      SameReferenceMesh);

  mochi::Error error;
  EXPECT_TRUE(run.Generate(1, error));

  ASSERT_TRUE(run.Embedded(1).contactSkinMesh.has_value());
  EXPECT_EQ(run.Embedded(1).contactSkinMesh->coordinates[0], 3);
}

TEST(ProcessingStackTest, ASourceRestartsWithoutTheEmbeddedMeshesAboveIt) {
  ModifierStack const stack = MakeStack(
      {{.kind = kSource, .run = ProduceTetrahedron},
       {.kind = kTransform,
        .acceptsTets = true,
        .acceptsEmbedded = true,
        .contactSkin = Triangle(7)},
       {.kind = kSource, .run = ProduceTriangle(2)}});
  StackRun run(
      stack, {0, 1, 2}, {}, CurrentMeshes(NoCurrentOutput), NoCurrentOutput, SameReferenceMesh);

  mochi::Error error;
  EXPECT_TRUE(run.Generate(0, error));
  EXPECT_TRUE(run.Generate(1, error));
  EXPECT_TRUE(run.Generate(2, error));

  EXPECT_TRUE(run.Embedded(1).contactSkinMesh.has_value());
  EXPECT_TRUE(run.Embedded(2).IsEmpty());
}
