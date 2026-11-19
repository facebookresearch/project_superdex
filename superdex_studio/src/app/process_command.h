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

#pragma once

// Runs a Model Editor processing pipeline (a .StudioProcessing.json) without a window: every
// enabled modifier in order, each export writing its file as soon as its stage has run, as the
// editor's Build/Export All does.

#include <mochi_core/utils/error.h>
#include <mochi_core/utils/nd_array.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace superdex::studio {

// The model files a pipeline's "From Model Viewer" sources read.
struct PipelineModelSlots {
  std::string cadPath;
  std::string renderPath;
  std::string mochiPath;
};

struct ProcessOptions {
  std::string pipelinePath;
  // Folder to write every export into, keeping each file name. Empty keeps the pipeline's paths.
  std::string outDir;
  // Overrides for the model slots. An empty slot is found the way the Model Editor finds it.
  PipelineModelSlots slots;
  // Runs even when two exports would write the same file, the later one winning, as in the Model
  // Editor. Otherwise such a pipeline is refused before anything runs.
  bool allowCollidingExports = false;
};

struct ProcessStageReport {
  std::string label; // the modifier's header label
  bool succeeded = false;
  std::string error; // why the stage or its export failed
  int64_t vertices = 0; // of the output surface (for a tetrahedral output, its boundary)
  int64_t triangles = 0;
  std::optional<mochi::Int3> sdfGrid; // set by stages that bake an SDF
  std::optional<mochi::Int2> tetMesh; // nodes and tets, set when the output is tetrahedral
  // Nodes and elements, set when the output is a polyline (which then has no vertices or
  // triangles).
  std::optional<mochi::Int2> polyline;
  std::string exportedPath; // the file an export stage wrote
  // Where the pipeline itself puts the export stage's file: exportedPath, unless --out moved it.
  std::string pipelineExportPath;
  double seconds = 0.0;
};

struct ProcessReport {
  PipelineModelSlots slots; // as resolved
  // Files that more than one enabled export writes.
  std::vector<std::string> collidingExportPaths;
  std::vector<ProcessStageReport> stages;
  bool succeeded = false;
  // "modifier / method" of each unrecognized modifier in the chain, when that refused the run.
  std::vector<std::string> unrecognizedModifiers;
  // The export a later source reads, when that refused --out.
  std::string outConflictPath;
};

// The model slots the Model Editor fills when it opens the model that owns @p pipelinePath. The
// editor keeps a model's pipeline at `<asset>/intermediates/<model name>.StudioProcessing.json`,
// so the name is matched in the asset around that folder. A pipeline kept apart for a shadowed
// model (`intermediates/<subfolder>/<file name>.StudioProcessing.json`) fills only that model's
// slot. A pipeline stored anywhere else is matched by name in its own folder.
PipelineModelSlots FindPipelineModelSlots(std::filesystem::path const& pipelinePath);

// Runs the pipeline at @p options.pipelinePath. Sets @p error, before any stage runs, when the
// pipeline cannot be read or has nothing to build from; a failed stage is reported in its stage
// entry instead, and the stages below it fail for lack of input. @p onStage, if set, is called as
// each stage finishes.
ProcessReport RunProcessingPipeline(
    ProcessOptions const& options,
    mochi::Error& error,
    std::function<void(ProcessStageReport const&)> const& onStage = {});

// Same as RunProcessingPipeline, on the mochi::Context already bound to the calling thread (as on
// an async task's thread) instead of a context of its own.
ProcessReport RunProcessingPipelineInBoundContext(
    ProcessOptions const& options,
    mochi::Error& error,
    std::function<void(ProcessStageReport const&)> const& onStage = {});

// Runs RunProcessingPipeline, printing progress to stdout and failures to stderr. Returns the
// process exit code: 0 when every stage and export succeeded, 1 otherwise.
int RunProcessCommand(ProcessOptions const& options);

} // namespace superdex::studio
