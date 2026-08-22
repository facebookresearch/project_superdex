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

// Save / load the modifier stack (+ a little editor state) to a human-readable JSON file via the
// simple-reflection library. The document is:
//   { "version": 1, "editorState": {...}, "modifiers": [ {modifier, method, enabled, collapsed,
//     properties}, ... ] }
// The header + per-entry fields are reflected structs; each entry's "properties" is the reflected
// props of the modifier's active method (a nested object), so the file stays readable/editable.
// Unrecognized modifiers/methods load as lossless passthrough placeholders (see
// unknown_placeholder).

#include "meshing/processing_modifiers/processing_modifier.h"

#include <mochi_core/utils/error.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace superdex::studio {

// Editor-level state persisted alongside the modifier array. Currently the CAD transform.
// Reflected; Quaternion serializes as an xyzw array of 4 reals.
struct ProcessingEditorState {
  mochi::Real3 cadScale = {1.0f, 1.0f, 1.0f};
  mochi::Quaternion cadRotation = {};
  mochi::Real3 cadTranslation = {0.0f, 0.0f, 0.0f};

  MOCHI_STRUCT_BEGIN(superdex::studio::ProcessingEditorState)
  MOCHI_FIELD(cadScale)
  MOCHI_FIELD(cadRotation)
  MOCHI_FIELD(cadTranslation)
  MOCHI_STRUCT_END()
};

// One "Additional Reference Model": a viewer-only reference asset the user added by hand, with its
// own type, transform, and visualization toggles. Persisted per-model so it is restored on reopen.
// NOTE: reference models are visualization context ONLY -- they are deliberately NOT exposed to the
// mesh-processing modifiers. To feed an external mesh into the pipeline, add a Source modifier with
// "From File" instead. (@p type is stored as an AssetTypeToToken string.)
struct ReferenceModelState {
  mochi::DynamicString type; // AssetTypeToToken(...) -- RenderModel / MochiModel / CadModel
  mochi::DynamicString path;
  mochi::Real3 scale = {1.0f, 1.0f, 1.0f};
  mochi::Quaternion rotation = {};
  mochi::Real3 translation = {0.0f, 0.0f, 0.0f};
  bool showSurface = true;
  bool showWireframe = false;
  double opacity = 1.0;
  mochi::Real3 color = {0.75f, 0.55f, 0.85f};
  bool overrideColor = false; // render refs: replace textures with the flat color

  MOCHI_STRUCT_BEGIN(superdex::studio::ReferenceModelState)
  MOCHI_FIELD(type)
  MOCHI_FIELD(path)
  MOCHI_FIELD(scale)
  MOCHI_FIELD(rotation)
  MOCHI_FIELD(translation)
  MOCHI_FIELD(showSurface)
  MOCHI_FIELD(showWireframe)
  MOCHI_FIELD(opacity)
  MOCHI_FIELD(color)
  MOCHI_FIELD(overrideColor)
  MOCHI_STRUCT_END()
};

// Result of loading a pipeline file. @p modifiers are fully built (method selected, props applied,
// enabled/collapsed set) but have no ids assigned yet (the editor assigns them on take-over).
struct LoadedPipeline {
  std::vector<std::unique_ptr<MeshProcessingModifier>> modifiers;
  ProcessingEditorState editorState;
  bool hasEditorState = false; // whether the file actually contained an editorState block
  std::vector<ReferenceModelState> referenceModels;
};

// Serializes @p modifiers + @p editorState + @p referenceModels into the pipeline JSON document
// (pretty-printed). This is the canonical snapshot of the editor's persisted pipeline state:
// written to disk via WriteProcessingPipelineFile, and equality-comparable so a caller can skip
// redundant writes. Also suitable as an in-memory snapshot (e.g. a future undo/redo history).
// Every stored file path (reference-model paths and each modifier method's PathPropKeys) is written
// relative to @p baseFile (the destination .StudioProcessing.json path) via
// superdex::robotics::MakePathRelative, so the file leaks no absolute paths and stays portable.
// Pass the same @p baseFile used when the file is written; LoadProcessingPipeline reverses this
// against the file it reads.
// An export modifier still on Auto omits its path entirely (see SerializeProps on the export
// methods), so a pipeline exporting to the derived location carries no path at all and stays valid
// when copied between bots or machines.
std::string SerializeProcessingPipeline(
    std::vector<std::unique_ptr<MeshProcessingModifier>> const& modifiers,
    ProcessingEditorState const& editorState,
    std::vector<ReferenceModelState> const& referenceModels,
    std::filesystem::path const& baseFile);

// Writes @p text (from SerializeProcessingPipeline) to @p path, creating parent directories.
// Returns false and sets @p error on failure.
bool WriteProcessingPipelineFile(
    std::string const& path,
    std::string const& text,
    mochi::Error& error);

// Reads a pipeline from @p path into @p out. Returns false and sets @p error if the file is
// missing, unparsable, or structurally invalid (callers should then NOT overwrite it). Unknown
// object-shaped modifiers/methods become placeholders (no error).
//
// Every stored path is resolved to absolute via superdex::robotics::MakePathAbsolute against
// @p baseFile, so the rest of the editor works in absolute paths. @p baseFile is the pipeline file
// the paths are to be read as relative to, which is ALWAYS the opened model's own
// .StudioProcessing.json -- not necessarily @p path. The two differ when loading a shipped preset:
// the document is read from processing_presets/, but its paths still mean "relative to the model I
// am being applied to", so the preset's own location must not leak into resolution. Passing
// @p baseFile explicitly also keeps load symmetric with SerializeProcessingPipeline, which
// relativizes against that same file. Empty @p baseFile leaves relative paths untouched.
//
// An export method whose path the document omits loads on Auto and re-derives it (see
// ModelEditor::RefreshAutoExportPaths); one that stores a path keeps it, and stays off Auto.
bool LoadProcessingPipeline(
    std::string const& path,
    std::filesystem::path const& baseFile,
    LoadedPipeline& out,
    mochi::Error& error);

} // namespace superdex::studio
