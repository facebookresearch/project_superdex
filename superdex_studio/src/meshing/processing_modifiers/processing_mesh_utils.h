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
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>

#include <mochi_renderer/utils.h> // mochi_renderer::MeshSection

#include <mochi_mesh/mochi_mesh_cli_types.h> // StepTessellationParams

#include <array>
#include <cstdint>
#include <string>
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

// Estimated size in bytes of @p mesh written as an ASCII Legacy VTK tetrahedral mesh (.vtk).
// Deliberately count-based so estimating a large intermediate mesh remains O(1).
int64_t EstimateVtkSizeBytes(mochi::MeshData const& mesh);

// Estimated size in bytes of @p mesh written as a Wavefront OBJ (.obj): text vertex / normal /
// face lines. Coarser than the GLB estimate (text size depends on coordinate magnitudes).
int64_t EstimateObjSizeBytes(mochi::MeshData const& mesh);

// Estimated size in bytes of @p model written as a .mochi.h5: its mesh, contact skin and visual
// mesh (with their skinning) and its SDF grid. Models the writer's gzip (deflate) + shuffle
// compression via the mesh / SDF compression factors in the .cpp; an SDF normally dominates.
int64_t EstimateMochiH5SizeBytes(mochi::ModelData const& model);

// Concatenate renderer sections into a neutral triangle MeshData (positions + indices).
mochi::MeshData MeshDataFromSections(std::vector<mochi_renderer::MeshSection> const& sections);

// Recover a stage's processing output from its display buffer. A non-empty tetrahedral output is
// authoritative because the displayed sections hold only its boundary, or a mesh embedded in it;
// otherwise concatenate the displayed triangle sections.
mochi::MeshData BufferedStageOutput(
    mochi::MeshData const& tetMesh,
    std::vector<mochi_renderer::MeshSection> const& sections);

// Neutral MeshData -> a single renderer MeshSection with computed angle-weighted vertex normals.
mochi_renderer::MeshSection SectionFromMeshData(mochi::MeshData const& mesh);

// Merges vertices at identical positions, joining the triangles of an STL (which repeats every
// corner) or of a mesh split at its seams into connected surfaces.
void WeldIdenticalVertices(mochi::MeshData& mesh);

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
//   .step / .stp             -- Z-up on disk, but mochi::mesh::TessellateStep(ByColor) /
//                               MeshStepBody fold the Z-up -> Y-up rotation in inside the
//                               superdex_mesh_cli helper, so their output is renderer-space
//                               ALREADY. Do not convert it.
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

// Tessellate a STEP file into renderer-space sections, one per CAD color (see
// mochi::mesh::TessellateStepByColor), each carrying that color as its linear baseColor. Uncolored
// faces, if any, form one section with the default baseColor. Blocking; safe off the main thread.
std::vector<mochi_renderer::MeshSection> TessellateStepSections(
    std::string const& path,
    mochi::mesh::StepTessellationParams const& params,
    mochi::Error& error);

// Load a mochi-model file's surface as a neutral triangle MeshData in renderer space.
mochi::MeshData LoadMochiMesh(std::string const& path, mochi::Error& error);

// The triangle surface to show for a stage's output: the mesh itself, or the boundary of a
// tetrahedral mesh (boundary nodes only). Same space as @p mesh.
mochi::MeshData DisplaySurface(mochi::MeshData const& mesh);

// A renderer-space mesh of any element type converted to Mochi space, keeping its connectivity.
// The conversion is a rotation, so tetrahedron orientation is preserved.
mochi::MeshData ToMochiSpace(mochi::MeshData mesh);

// Write a renderer-space tetrahedral MeshData to an ASCII Legacy VTK (version 3.0) unstructured
// grid .vtk file, in Mochi space like the .mochi.h5 it would export to.
void WriteVtkFile(std::string const& path, mochi::MeshData const& mesh, mochi::Error& error);

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

} // namespace superdex::studio::processing
