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

// Operations on a whole modifier stack -- the ordered modifiers of one model's processing pipeline.
// The Model Editor and headless tools share them, so both build a stack the same way.

#include "meshing/processing_modifiers/processing_modifier.h"

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/utils/error.h>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace superdex::studio {

using ModifierStack = std::vector<std::unique_ptr<MeshProcessingModifier>>;

// The source-first list of enabled modifier indices feeding @p index (with @p index last), or empty
// when no source is reachable upstream. Stops at the nearest source -- one modifier's Generate
// rebuilds its own segment, not the segments feeding it through files.
[[nodiscard]] std::vector<std::size_t> BuildGenerationChain(
    ModifierStack const& stack,
    std::size_t index);

// The whole enabled stack in order, from its first enabled source -- what Build/Export All builds,
// spanning every source->...->export segment. Empty when no enabled source exists.
[[nodiscard]] std::vector<std::size_t> BuildFullGenerationChain(ModifierStack const& stack);

// The modifier index whose output the stage at @p index references (its explicit upstream target,
// else the nearest preceding source), or -1 if none. See MeshProcessingMethod::NeedsReferenceMesh.
[[nodiscard]] int ReferenceModifierIndex(ModifierStack const& stack, std::size_t index);

// The file the stack's source (first) modifier reads, which every export modifier derives its
// default output path from. Empty when the stack is empty or its source has no file yet.
[[nodiscard]] std::string StackSourceFilePath(
    ModifierStack const& stack,
    ModifierRunContext const& ctx);

// Re-derives every export modifier's path from @p sourceFilePath while its Auto toggle is on (see
// MeshProcessingMethod::RefreshAutoExportPath), so an export writes its file on Build/Export All
// without the user opening Browse, and an Auto path always matches what Browse would offer. Cheap
// enough to call every frame, and safe to: an Auto path is never serialized, so re-deriving it
// cannot dirty the saved snapshot however much it moves.
void RefreshAutoExportPaths(ModifierStack& stack, std::string const& sourceFilePath);

// Moves every enabled file export into @p folder, keeping each file's name (see
// MeshProcessingMethod::OverrideExportPath); disabled modifiers write nothing, so they keep their
// paths. Call after RefreshAutoExportPaths, so Auto exports keep their derived names. Exports that
// shared a file name in different folders now share one path; FindCollidingExportPaths reports
// them, so check it before running the stack.
void RedirectExports(ModifierStack& stack, std::filesystem::path const& folder);

// Whether @p a and @p b name the same file: equal once lexically normalized, ignoring case. Case is
// ignored on every platform because pipelines move between systems, and the default file systems
// of Windows and macOS treat names that differ only in case as one file. An empty path names no
// file.
[[nodiscard]] bool SameFilePath(std::string_view a, std::string_view b);

// Per-modifier flag marking export modifiers that share an output file with another enabled export
// modifier. Two exports aimed at the same path silently overwrite each other; the Auto path is
// deliberately not disambiguated (it has to keep the model's own name to stay associated with it),
// so the fix is to turn Auto off on one of them. Compares resolved paths, so hand-picked duplicates
// are caught as well.
[[nodiscard]] std::vector<bool> FindCollidingExportPaths(ModifierStack const& stack);

// One pass down a generation chain (see BuildGenerationChain), stage by stage: each stage runs on
// the output of the position before it, and an export stage writes that same input to its file.
// Positions that are not regenerated keep the output they started with. The steps must run in chain
// order, one at a time; the stack must outlive this object and stay unchanged while it runs.
class StackRun {
 public:
  // @p currentOutput returns a modifier's complete current output, with the meshes embedded in it,
  // by stack index and seeds every chain position. @p currentReferenceMesh returns cached triangle
  // surfaces for references outside @p chain; @p referenceMeshFromOutput obtains one from an
  // in-chain output on demand.
  StackRun(
      ModifierStack const& stack,
      std::vector<std::size_t> chain,
      ModifierRunContext ctx,
      std::function<StageMesh(std::size_t modifierIndex)> const& currentOutput,
      std::function<mochi::MeshData(std::size_t modifierIndex)> const& currentReferenceMesh,
      std::function<mochi::MeshData(mochi::MeshData const& output)> referenceMeshFromOutput);

  // Regenerates chain position @p p. A failure (an error, or no elements) clears the position's
  // output, so the next stage receives no input and fails as well, and returns false.
  bool Generate(std::size_t p, mochi::Error& error);
  // Writes the input of the export stage at chain position @p p to its file. Refuses a position
  // whose canonical output is empty, and returns false on error.
  bool Export(std::size_t p, mochi::Error& error) const;

  [[nodiscard]] mochi::MeshData const& Output(std::size_t p) const {
    return _outputs[p].mesh;
  }
  // The meshes embedded in the output of chain position @p p.
  [[nodiscard]] EmbeddedMeshes const& Embedded(std::size_t p) const {
    return _outputs[p].embedded;
  }

 private:
  // Where a stage's reference comes from: a position in the chain, else a mesh captured up front
  // because the referenced modifier is not in the chain.
  struct Reference {
    int chainPos = -1;
    mochi::MeshData currentMesh;
  };

  [[nodiscard]] StageMesh const& Input(std::size_t p) const;

  std::vector<MeshProcessingModifier const*> _modifiers; // by chain position
  std::vector<StageMesh> _outputs; // by chain position
  std::vector<Reference> _references; // by chain position
  std::function<mochi::MeshData(mochi::MeshData const& output)> _referenceMeshFromOutput;
  ModifierRunContext _ctx;
};

} // namespace superdex::studio
