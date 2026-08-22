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

#include <array>
#include <cstdint>
#include <vector>

namespace mochi {
class CoordinateSpaceConverter;
} // namespace mochi

namespace mochi_renderer {

//--------------------------------------------------------------------------------------------------
// STRUCTS
//--------------------------------------------------------------------------------------------------

// One sub-mesh of a multi-material asset: flat positions/normals/indices plus a
// resolved solid PBR material.
//
// MeshSection is the common intermediate representation that enables a minimum-
// viable conversion of DAE (COLLADA), OBJ, and STL assets into GLB: each reader
// (@ref ReadColladaFromFile, @ref ReadObjFromFile, @ref ReadStlFromFile) emits
// sections, and @ref BuildGlbFromMeshSections turns them into a GLB with one
// glTF primitive (and one material) per section. It captures only what that
// baseline conversion needs (geometry + a solid PBR material); richer glTF
// features such as textures, skinning, and animation are intentionally omitted.
// These and other fields may be added later as needed.
//
// `hasNormals` is the single source of truth for normal presence: it indicates
// whether `normals` is populated, and when true `normals` must hold exactly one
// (x, y, z) normal per vertex (i.e. `normals.size() == positions.size()`). When
// false the recipient is expected to compute normals if necessary.
struct MeshSection {
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;
  bool hasNormals = false;
  std::array<float, 4> baseColor = {0.5f, 0.5f, 0.5f, 1.0f};
  float metallic = 0.0f;
  float roughness = 0.5f;
};

//--------------------------------------------------------------------------------------------------
// GLB BUILDING
//--------------------------------------------------------------------------------------------------

// Builds a minimal GLB (binary glTF) in memory from multiple mesh sections, one
// glTF primitive and one PBR material per section. All sections share a single
// buffer; each contributes its own POSITION/index bufferViews (plus NORMAL when
// the section has normals). Use this for both single- and multi-material assets
// (e.g. COLLADA with per-section materials); pass a single-element vector for a
// single-material mesh.
//
// A NORMAL attribute is emitted for each section whose `hasNormals` is set; such
// a section must supply one normal per vertex (`normals.size() ==
// positions.size()`). Leave `hasNormals` false to skip normals. Triangle indices
// are local to each section's own vertex arrays (zero-based per section). Empty
// sections (no positions or no indices) are skipped.
//
// Every emitted material is marked two-sided: imported geometry often has
// inconsistent triangle winding, so back faces are rendered to avoid sections
// silently disappearing to back-face culling.
std::vector<uint8_t> BuildGlbFromMeshSections(std::vector<MeshSection> const& sections);

//--------------------------------------------------------------------------------------------------
// MESH FILE READING
//--------------------------------------------------------------------------------------------------

// Reads a `.glb` file from disk into one @ref MeshSection per triangle
// primitive, the inverse of @ref BuildGlbFromMeshSections. Each section receives
// flat POSITION data and triangle indices (generated sequentially when the
// primitive is non-indexed); NORMAL is unpacked when present (`hasNormals`). PBR
// base color / metallic / roughness factors are read from the primitive's
// material when it has a metallic-roughness model, otherwise the
// @ref MeshSection defaults apply. Non-triangle primitives are skipped, and node
// transforms are ignored. External `.bin` buffers are resolved relative to
// `path`. Returns an empty vector on read / parse failure or when no triangle
// geometry is found.
std::vector<MeshSection> ReadGlbFromFile(char const* path);

// Reads a `.dae` (COLLADA) file from disk into one @ref MeshSection per geometry
// primitive. COLLADA's solid (non-PBR) materials are resolved to glTF PBR
// factors (diffuse color → baseColor, shininess → roughness, metallic = 0);
// image textures, animations, and skinning are unsupported. Geometry is
// converted into glTF's Y-up convention and node transforms are applied.
// `hasNormals` indicates whether NORMAL data was present. Returns an empty
// vector on read / parse failure or when no geometry is found.
std::vector<MeshSection> ReadColladaFromFile(char const* path);

// Reads a `.obj` (Wavefront) file from disk into one @ref MeshSection per
// material. Faces are grouped by their active `usemtl` material (one section per
// material, first-seen order); faces before any `usemtl` form a default section.
// Referenced `.mtl` libraries (`mtllib`) are resolved relative to `path` and
// their materials mapped to glTF PBR factors (diffuse `Kd` → baseColor, `d`/`Tr`
// → alpha, specular exponent `Ns` → roughness, metallic = 0); textures and
// unknown/unresolved materials fall back to the @ref MeshSection defaults. Faces
// are fan-triangulated with vertices emitted flat per corner. `hasNormals` is
// set only when the file supplies vertex normals (vn); otherwise `normals` is
// left empty for the caller to compute. Returns an empty vector on read failure
// or when no geometry is found.
std::vector<MeshSection> ReadObjFromFile(char const* path);

// Reads a `.stl` file (ASCII or binary) from disk into a single @ref MeshSection
// with deduplicated, indexed geometry. STL stores only per-face normals, so
// `hasNormals` is left false and `normals` empty for the caller to compute. STL
// carries no PBR material, so @ref MeshSection material defaults apply. Multi-
// solid STL files are unsupported and yield an empty vector. Returns an empty
// vector on read failure or when no geometry is found.
std::vector<MeshSection> ReadStlFromFile(char const* path);

//--------------------------------------------------------------------------------------------------
// MESH FILE WRITING
//--------------------------------------------------------------------------------------------------

// Writes @p sections to a `.glb` file: builds the GLB in memory via @ref BuildGlbFromMeshSections
// and writes the bytes to @p path. Returns false on write failure.
bool WriteGlbToFile(char const* path, std::vector<MeshSection> const& sections);

// Writes @p sections to a Wavefront `.obj` file (positions + triangle faces, 1-based indices;
// sections are concatenated with their index ranges offset). Normals and materials are not written.
// Returns false on write failure.
bool WriteObjToFile(char const* path, std::vector<MeshSection> const& sections);

//--------------------------------------------------------------------------------------------------
// SPACE CONVERSION
//--------------------------------------------------------------------------------------------------

// Transforms every section's positions and normals in place from the converter's
// input space to its output space: positions via
// @ref mochi::CoordinateSpaceConverter::TranslationsToOutput (rotation plus unit scaling)
// and normals via @ref mochi::CoordinateSpaceConverter::DirectionsToOutput (rotation only;
// the converter's basis change is orthogonal, so normals stay unit length).
//
// Use this to bring reader output (e.g. @ref ReadStlFromFile / @ref
// ReadObjFromFile geometry authored in some source space) into the space a
// consumer expects before handing sections to @ref BuildGlbFromMeshSections.
//
// Triangle indices are left unchanged. If `converter.FlipsHandedness()` the emitted
// triangles become mirror-wound; callers that care about facing should account for that.
void ConvertMeshSectionsSpace(
    std::vector<MeshSection>& sections,
    mochi::CoordinateSpaceConverter const& converter);

//--------------------------------------------------------------------------------------------------
// GEOMETRY PROCESSING
//--------------------------------------------------------------------------------------------------

// Computes one unit-length face normal per triangle (3 floats each), with
// positions laid out flat as (x, y, z) triples and indices as triangle triples.
// Degenerate triangles produce a zero normal.
void ComputeFaceNormals(
    std::vector<float> const& positions,
    std::vector<int> const& indices,
    std::vector<float>& faceNormalsOut);

// Computes per-vertex normals by angle-weighting adjacent face normals.
// `positions` is flat (x, y, z) per vertex, `faceNormals` is flat (x, y, z)
// per triangle (e.g. the output of @ref ComputeFaceNormals), and `indices`
// holds triangle indices. The output is sized to one normal per vertex.
void ComputeVertexNormalsAngleWeighted(
    std::vector<float> const& positions,
    std::vector<float> const& faceNormals,
    std::vector<int> const& indices,
    std::vector<float>& vertexNormalsOut);

// Computes per-vertex normals by area-weighting adjacent face normals.
// `positions` is flat (x, y, z) per vertex and `indices` holds triangle indices.
// Each triangle accumulates its unnormalized cross product (whose magnitude is
// twice the triangle area) into its three vertices, so larger faces influence a
// shared vertex more; each vertex normal is then normalized to unit length
// (left zero for vertices with no incident triangles). The output is sized to
// one normal per vertex (3 floats each).
//
// Unlike @ref ComputeVertexNormalsAngleWeighted, this variant needs no
// precomputed face normals and is cheaper; prefer the angle-weighted variant
// when incident faces vary greatly in size.
void ComputeVertexNormalsAreaWeighted(
    std::vector<float> const& positions,
    std::vector<int> const& indices,
    std::vector<float>& vertexNormalsOut);

} // namespace mochi_renderer
