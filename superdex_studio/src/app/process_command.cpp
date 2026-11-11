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

#include "app/process_command.h"

#include "assets/asset_types.h"
#include "meshing/processing_modifiers/processing_mesh_utils.h" // DisplaySurface
#include "meshing/processing_modifiers/processing_serialization.h"
#include "meshing/processing_modifiers/processing_stack.h"

#include <superdex_robotics/utils/file_utils.h> // kIntermediatesSubdir

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/path.h>
#include <mochi_mesh/mesh_cli_control.h> // ResolveMeshCliPath
#include <mochi_physics/cpp_api/mochi_context.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace superdex::studio {

namespace {

std::string* SlotForType(PipelineModelSlots& slots, AssetType type) {
  switch (type) {
    case AssetType::CadModel:
      return &slots.cadPath;
    case AssetType::RenderModel:
      return &slots.renderPath;
    case AssetType::MochiModel:
      return &slots.mochiPath;
    case AssetType::MochiPrefab:
    case AssetType::Bot:
    case AssetType::BotScene:
    case AssetType::BotTask:
    case AssetType::Tuner:
    case AssetType::IK:
    case AssetType::Unknown:
    case AssetType::Count:
      break;
  }
  return nullptr;
}

std::string AbsolutePath(std::string const& path) {
  if (path.empty()) {
    return {};
  }
  std::error_code ec;
  std::filesystem::path const absolute = std::filesystem::absolute(path, ec);
  return ec ? path : absolute.lexically_normal().string();
}

} // namespace

PipelineModelSlots FindPipelineModelSlots(std::filesystem::path const& pipelinePath) {
  PipelineModelSlots slots;
  std::string const fileName = pipelinePath.filename().string();
  if (fileName.size() <= kProcessingPipelineSuffix.size() ||
      !fileName.ends_with(kProcessingPipelineSuffix)) {
    return slots;
  }
  std::string const name = fileName.substr(0, fileName.size() - kProcessingPipelineSuffix.size());

  // The model's asset is the folder holding `intermediates/`; without one, the pipeline's own
  // folder stands in for it. Discovery starts from `intermediates/` itself, a role folder, so the
  // asset is found as its parent even when the asset sits in a folder named like a role folder.
  // (Files generated there always carry a suffix, so none of them matches a model's name.)
  std::filesystem::path const pipelineDir = pipelinePath.parent_path();
  std::filesystem::path originDir = pipelineDir;
  std::filesystem::path discoveryDir = pipelineDir;
  std::filesystem::path mirrored; // the pipeline's folder below `intermediates/`
  for (std::filesystem::path dir = pipelineDir; !dir.empty() && dir != dir.root_path();
       dir = dir.parent_path()) {
    if (dir.filename() == superdex::robotics::kIntermediatesSubdir) {
      originDir = dir.parent_path();
      discoveryDir = dir;
      mirrored = pipelineDir.lexically_relative(dir);
      break;
    }
  }

  // A shadowed model's pipeline keeps the model's file name, extension included, under a mirror
  // of its folder (see AssetGeneratedFilePath). It is a set of one.
  AssetType const shadowedType = ClassifyAssetTypeByFilename(name);
  if (std::string* const slot = SlotForType(slots, shadowedType)) {
    std::filesystem::path const origin = (originDir / mirrored / name).lexically_normal();
    std::error_code ec;
    if (std::filesystem::is_regular_file(origin, ec)) {
      *slot = origin.string();
      return slots;
    }
  }

  mochi::Path const origin{discoveryDir};
  slots.cadPath = FindAssetForSlot(name, AssetType::CadModel, origin);
  slots.renderPath = FindAssetForSlot(name, AssetType::RenderModel, origin);
  slots.mochiPath = FindAssetForSlot(name, AssetType::MochiModel, origin);
  return slots;
}

namespace {

ProcessReport RunPipeline(
    ProcessOptions const& options,
    mochi::Error& error,
    std::function<void(ProcessStageReport const&)> const& onStage,
    bool ownContext) {
  ProcessReport report;
  MOCHI_ERROR_RETURN(error, report);

  std::filesystem::path const pipelinePath{AbsolutePath(options.pipelinePath)};
  LoadedPipeline loaded;
  if (!LoadProcessingPipeline(pipelinePath.string(), pipelinePath, loaded, error)) {
    return report;
  }
  ModifierStack stack = std::move(loaded.modifiers);

  report.slots = FindPipelineModelSlots(pipelinePath);
  for (auto [slot, given] :
       {std::pair{&report.slots.cadPath, &options.slots.cadPath},
        std::pair{&report.slots.renderPath, &options.slots.renderPath},
        std::pair{&report.slots.mochiPath, &options.slots.mochiPath}}) {
    if (!given->empty()) {
      *slot = AbsolutePath(*given);
    }
  }

  // The run context the Model Editor would build for this model (see ModelEditor::MakeRunContext).
  ModifierRunContext ctx;
  ctx.cadFilePath = report.slots.cadPath;
  ctx.renderModelPath = report.slots.renderPath;
  ctx.mochiModelPath = report.slots.mochiPath;
  if (loaded.hasEditorState) {
    ctx.cadScale = loaded.editorState.cadScale;
    ctx.cadRotation = loaded.editorState.cadRotation;
    ctx.cadTranslation = loaded.editorState.cadTranslation;
  }

  std::vector<std::size_t> const chain = BuildFullGenerationChain(stack);
  MOCHI_ERROR_IF(chain.empty(), error, "The pipeline has no enabled source modifier.");
  MOCHI_ERROR_RETURN(error, report);
  for (std::size_t const index : chain) {
    if (std::ranges::find(loaded.unrecognized, index) != loaded.unrecognized.end()) {
      report.unrecognizedModifiers.push_back(
          stack.at(index)->DisplayName() + std::string(" / ") +
          stack.at(index)->ActiveMethod().Name());
    }
  }
  MOCHI_ERROR_IF(
      !report.unrecognizedModifiers.empty(),
      error,
      "The pipeline has a modifier this Studio does not know, perhaps saved by a newer one. It "
      "would pass its input on unchanged, so nothing runs.");
  MOCHI_ERROR_RETURN(error, report);

  RefreshAutoExportPaths(stack, StackSourceFilePath(stack, ctx));
  // Where each modifier exports before --out moves it.
  std::unordered_map<MeshProcessingModifier const*, std::string> pipelineExportPaths;
  for (auto const& modifier : stack) {
    pipelineExportPaths.emplace(modifier.get(), modifier->ExportPath());
  }
  // Only the chain's exports run, so only they can write over each other or over a source's model.
  auto const writesFile = [&stack](std::size_t index) {
    return stack.at(index)->ProvidesFileExport() && !stack.at(index)->ExportPath().empty();
  };
  if (!options.outDir.empty()) {
    // A source reading what an earlier export writes would read the file next to the asset rather
    // than the one --out collects.
    for (std::size_t a = 0; a < chain.size(); ++a) {
      for (std::size_t b = a + 1; b < chain.size() && writesFile(chain[a]); ++b) {
        MeshProcessingModifier const& source = *stack.at(chain[b]);
        if (source.Kind() == ModifierKind::Source &&
            SameFilePath(source.SourceFilePath(ctx), stack.at(chain[a])->ExportPath())) {
          report.outConflictPath = stack.at(chain[a])->ExportPath();
        }
      }
    }
    MOCHI_ERROR_IF(
        !report.outConflictPath.empty(),
        error,
        "A source reads a file that an earlier export in the pipeline writes, so --out cannot "
        "collect that export. Run the pipeline without --out.");
    MOCHI_ERROR_RETURN(error, report);
    RedirectExports(stack, AbsolutePath(options.outDir));
  }
  for (std::size_t a = 0; a < chain.size(); ++a) {
    for (std::size_t b = a + 1; b < chain.size(); ++b) {
      std::string const path = stack.at(chain[a])->ExportPath();
      if (writesFile(chain[a]) && writesFile(chain[b]) &&
          SameFilePath(path, stack.at(chain[b])->ExportPath()) &&
          std::ranges::none_of(report.collidingExportPaths, [&path](std::string const& known) {
            return SameFilePath(known, path);
          })) {
        report.collidingExportPaths.push_back(path);
      }
    }
  }
  MOCHI_ERROR_IF(
      !report.collidingExportPaths.empty() && !options.allowCollidingExports,
      error,
      "Two exports would write the same file. Give them different paths in the Model Editor, or "
      "different file names when --out collects them in one folder, or pass --override to let "
      "the later one win, as the Model Editor does.");
  MOCHI_ERROR_RETURN(error, report);
  MOCHI_ERROR_IF_NOT(
      stack.at(chain.front())->CanGenerate(ctx),
      error,
      "The pipeline's source has no model to read: none was found next to the pipeline. Pass it "
      "with --cad, --render or --mochi.");
  MOCHI_ERROR_RETURN(error, report);

  // Same worker setup as the editor's async tasks (see AsyncTaskRunner): the mesh operations use
  // the context's scheduler, and the file cache is on in the app.
  mochi::Context* const context = ownContext ? mochi::CreateContext() : nullptr;
  if (context != nullptr) {
    context->EnableFileCache(true);
    context->BindThisThread();
  }
  MOCHI_DEFER({
    if (context != nullptr) {
      context->UnbindThisThread();
      mochi::DestroyContext(context);
    }
  });

  auto const noCurrentOutput = [](std::size_t /*modifierIndex*/) { return StageMesh{}; };
  auto const noCurrentMesh = [](std::size_t /*modifierIndex*/) { return mochi::MeshData{}; };
  StackRun run(stack, chain, ctx, noCurrentOutput, noCurrentMesh, processing::DisplaySurface);
  report.succeeded = true;
  for (std::size_t p = 0; p < chain.size(); ++p) {
    MeshProcessingModifier const& modifier = *stack.at(chain[p]);
    ProcessStageReport stage;
    stage.label = modifier.HeaderLabel();
    auto const start = std::chrono::steady_clock::now();

    mochi::Error stageError;
    stage.succeeded = run.Generate(p, stageError);
    if (!stage.succeeded) {
      stage.error = stageError.IsOK() ? "produced no mesh" : stageError.GetDescription();
    } else if (modifier.ProvidesFileExport() && modifier.ExportPath().empty()) {
      stage.succeeded = false;
      stage.error =
          "the export has no file path: an Auto export is named after the pipeline's first "
          "modifier, which has no model. Give the export a path in the Model Editor.";
    } else if (modifier.ProvidesFileExport()) {
      stage.pipelineExportPath = pipelineExportPaths.at(&modifier);
      stage.succeeded = run.Export(p, stageError);
      if (stage.succeeded) {
        stage.exportedPath = modifier.ExportPath();
      } else {
        stage.error = stageError.GetDescription();
      }
    }

    mochi::MeshData const& output = run.Output(p);
    if (output.nodesPerElement == 4) {
      // Report a tetrahedral output by its boundary surface, plus its node and tet counts.
      mochi::MeshData const surface = processing::DisplaySurface(output);
      stage.vertices = static_cast<int64_t>(surface.GetNumNodes());
      stage.triangles = static_cast<int64_t>(surface.GetNumElements());
      stage.tetMesh = mochi::Int2{output.GetNumNodes(), output.GetNumElements()};
    } else {
      stage.vertices = static_cast<int64_t>(output.GetNumNodes());
      stage.triangles = static_cast<int64_t>(output.GetNumElements());
    }
    MeshStats stats;
    modifier.AnnotateStats(stats);
    stage.sdfGrid = stats.sdfGrid;
    stage.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    report.succeeded = report.succeeded && stage.succeeded;
    if (onStage) {
      onStage(stage);
    }
    report.stages.push_back(std::move(stage));
  }
  return report;
}

} // namespace

ProcessReport RunProcessingPipeline(
    ProcessOptions const& options,
    mochi::Error& error,
    std::function<void(ProcessStageReport const&)> const& onStage) {
  return RunPipeline(options, error, onStage, /*ownContext=*/true);
}

ProcessReport RunProcessingPipelineInBoundContext(
    ProcessOptions const& options,
    mochi::Error& error,
    std::function<void(ProcessStageReport const&)> const& onStage) {
  return RunPipeline(options, error, onStage, /*ownContext=*/false);
}

int RunProcessCommand(ProcessOptions const& options) {
  std::printf("Processing %s\n", options.pipelinePath.c_str());
  // Flushed before the mesh helper, which shares stdout, prints anything.
  std::fflush(stdout);
  if (mochi::mesh::ResolveMeshCliPath().empty()) {
    std::fprintf(
        stderr,
        "warning: superdex_mesh_cli helper not found, so mesh operations will fail. Install the "
        "superdex-mesh-cli distribution, or set SUPERDEX_MESH_CLI_PATH.\n");
  }

  std::size_t stageIndex = 0;
  mochi::Error error;
  ProcessReport const report =
      RunProcessingPipeline(options, error, [&stageIndex](ProcessStageReport const& stage) {
        std::printf(
            "[%zu] %s: %s, %s%lld vertices, %lld triangles",
            stageIndex++,
            stage.label.c_str(),
            stage.succeeded ? "ok" : "FAILED",
            stage.tetMesh.has_value() ? "boundary " : "",
            static_cast<long long>(stage.vertices),
            static_cast<long long>(stage.triangles));
        if (stage.sdfGrid.has_value()) {
          std::printf(
              ", SDF grid %d x %d x %d",
              (*stage.sdfGrid)[0],
              (*stage.sdfGrid)[1],
              (*stage.sdfGrid)[2]);
        }
        if (stage.tetMesh.has_value()) {
          std::printf(", tet mesh %d nodes, %d tets", (*stage.tetMesh)[0], (*stage.tetMesh)[1]);
        }
        std::printf(" (%.1f s)\n", stage.seconds);
        if (!stage.exportedPath.empty()) {
          std::printf("    wrote %s\n", stage.exportedPath.c_str());
        }
        if (!stage.succeeded) {
          std::fprintf(stderr, "    error: %s\n", stage.error.c_str());
        }
        std::fflush(stdout);
      });

  if (!error.IsOK()) {
    std::fprintf(stderr, "error: %s\n", error.GetDescription());
    for (std::string const& modifier : report.unrecognizedModifiers) {
      std::fprintf(stderr, "    %s\n", modifier.c_str());
    }
    if (!report.outConflictPath.empty()) {
      std::fprintf(stderr, "    %s\n", report.outConflictPath.c_str());
    }
    for (std::string const& path : report.collidingExportPaths) {
      std::fprintf(stderr, "    %s\n", path.c_str());
    }
    return 1;
  }
  for (std::string const& path : report.collidingExportPaths) {
    auto const writes =
        std::ranges::count_if(report.stages, [&path](ProcessStageReport const& stage) {
          return SameFilePath(stage.exportedPath, path);
        });
    if (writes > 1) {
      std::fprintf(
          stderr,
          "warning: more than one export wrote %s; the last one's file is kept\n",
          path.c_str());
    }
  }
  std::printf(
      "Models: CAD '%s', render '%s', mochi '%s'\n",
      report.slots.cadPath.c_str(),
      report.slots.renderPath.c_str(),
      report.slots.mochiPath.c_str());
  std::printf("%s\n", report.succeeded ? "Done." : "Failed.");
  return report.succeeded ? 0 : 1;
}

} // namespace superdex::studio
