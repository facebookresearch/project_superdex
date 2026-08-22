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

// Neutral mesh / renderer helpers shared by the processing methods (sources, exports, transform):
// MeshData <-> renderer MeshSection conversion, an in-place affine transform, render/mochi model
// loaders, obj/glb writers, and export-path helpers. Ported from the former file-local helpers in
// mesh_modifier.cpp so multiple modifier files can share them without duplicating code.

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>

#include <mochi_renderer/utils.h> // mochi_renderer::MeshSection

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace superdex::studio::processing {

// --- File-size estimates for the Model Editor stats blocks -------------------------------------
// Rough, deliberately-simple on-disk size estimates for a mesh (and, for the mochi model, its
// SDF). Used to annotate the read-only stats blocks; not meant to be exact (target: within ~10%).
// The per-element sizes and the .mochi.h5 compression factors are calibration constants defined at
// the top of processing_mesh_utils.cpp -- tune them there against real exports.

// Estimated size in bytes of @p mesh written as a binary glTF (.glb): float3 positions + float3
// normals per vertex, indices per triangle, plus a small constant chunk/header/material overhead.
int64_t EstimateGlbSizeBytes(mochi::MeshData const& mesh);

// Estimated size in bytes of @p mesh written as a Wavefront OBJ (.obj): text vertex / normal /
// face lines. Coarser than the GLB estimate (text size depends on coordinate magnitudes).
int64_t EstimateObjSizeBytes(mochi::MeshData const& mesh);

// Estimated size in bytes of a .mochi.h5 holding @p mesh (surface) plus an SDF grid of
// @p sdfValueCount values. Models the writer's gzip (deflate) + shuffle compression via the
// mesh / SDF compression factors in the .cpp; the SDF term normally dominates.
int64_t EstimateMochiH5SizeBytes(mochi::MeshData const& mesh, int64_t sdfValueCount);

// Concatenate renderer sections into a neutral triangle MeshData (positions + indices).
mochi::MeshData MeshDataFromSections(std::vector<mochi_renderer::MeshSection> const& sections);

// Neutral MeshData -> a single renderer MeshSection with computed angle-weighted vertex normals.
mochi_renderer::MeshSection SectionFromMeshData(mochi::MeshData const& mesh);

// Applies scale, then rotation, then translation in place to a mesh's coordinates.
void ApplyTransform(
    mochi::MeshData& mesh,
    mochi::Real3 const& scale,
    mochi::Quaternion const& rotation,
    mochi::Real3 const& translation);

// Case-insensitive suffix test.
bool EndsWithNoCase(std::string const& text, char const* suffix);

// --- The mesh-file space convention ------------------------------------------------------------
// The renderer's scene and every mochi::MeshData flowing through the modifier stack are in
// mochi_renderer::RenderSpace() (Y-up). What a loader has to do depends on the format, and the
// deciding factor is the LOADER, not the file:
//
//   .obj / .stl / .mochi.h5  -- read out in Mochi's space (CoordinateSpace::Default, Z-up);
//                               MUST be converted here, and converted back on write
//   .glb / .gltf / .dae      -- their readers already resolve the up-axis to Y-up
//   .step / .stp             -- Z-up on disk, but mochi::mesh::TessellateStep / MeshStepBody fold
//                               the Z-up -> Y-up rotation in inside the superdex_mesh_cli helper,
//                               so their output is renderer-space ALREADY. Do not convert it.
//
// The two spaces are a quarter turn about X apart, so converting the wrong group -- or converting
// twice -- tips the model over. ReadSectionsInRenderSpace and WriteObjFile below are where this
// happens for mesh files; keep any new mesh-file entry point going through them rather than calling
// the mochi_renderer readers/writers directly. (UrdfImporter applies the same rule when baking
// GLBs; cad_model_asset.cpp is where the STEP exception is handled.)

// Read a mesh file (obj/glb/gltf/stl/dae) into renderer-space sections, applying the convention
// above. Empty for an unsupported extension or a failed/empty read.
std::vector<mochi_renderer::MeshSection> ReadSectionsInRenderSpace(std::string const& path);

// Load a render-model file (obj/glb/stl/dae) as a neutral triangle MeshData in renderer space.
mochi::MeshData LoadRenderMesh(std::string const& path, mochi::Error& error);

// Load a mochi-model file's surface as a neutral triangle MeshData in renderer space.
mochi::MeshData LoadMochiMesh(std::string const& path, mochi::Error& error);

// Write a neutral triangle MeshData to an OBJ file (no material), converting it back out of
// renderer space into the Mochi space an .obj is expected to hold.
void WriteObjFile(std::string const& path, mochi::MeshData const& mesh, mochi::Error& error);

// Write a neutral triangle MeshData to a GLB file, tagging the section's PBR material with
// @p baseColor (written as the glTF baseColorFactor).
void WriteGlbFile(
    std::string const& path,
    mochi::MeshData const& mesh,
    std::array<float, 4> const& baseColor,
    mochi::Error& error);

// Base file name of @p path without its extension (handles the .mochi.h5 / .mochi.json double
// extensions specially).
std::string FileBaseName(std::string const& path);

// Where an export modifier's Browse dialog should start, in order of preference:
//   1. @p currentPath -- while Auto is on this is the derived path, once pinned it is the user's.
//   2. @p suggestedPath -- covers a blank path, and a pinned one whose folder has since been
//      deleted or renamed.
//   3. @p modelFolder as a folder-only default (trailing separator, no file name), so even with no
//      source to derive from the dialog opens beside the opened model.
// A candidate only counts if it is a well-formed path -- one no filesystem could accept (invalid
// characters) is skipped. Pointing at a folder that does not exist yet is fine and is passed
// through: the export creates the folder when it writes there. Empty when none apply, which leaves
// the dialog to pick its own starting point.
std::string ExportDialogStartPath(
    std::string const& currentPath,
    std::string const& suggestedPath,
    std::string const& modelFolder);

// Whether @p a and @p b name the same file. Compares the lexically-normalized generic (forward
// slash) form, so a derived path built with native separators still matches an equivalent one the
// user typed or that round-tripped through the pipeline JSON. Used to spot two export modifiers
// aimed at the same output file.
bool SamePath(std::string_view a, std::string_view b);

} // namespace superdex::studio::processing
