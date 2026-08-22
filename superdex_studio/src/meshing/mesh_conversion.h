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

#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/coordinate_space_converter.h>

#include <mochi_renderer/path.h>
#include <mochi_renderer/utils.h>

#include <string_view>
#include <vector>

namespace superdex::studio {

// Output mesh extensions produced by the conversion functions.
constexpr std::string_view kGlbExtension = ".glb";
constexpr std::string_view kMochiH5Extension = ".mochi.h5";

// @brief Convert a render/collision mesh (.dae, .obj, or .stl) to a .glb file.
//
// Parses the input geometry into one glTF primitive per material section via the
// matching mochi_renderer reader (@ref mochi_renderer::ReadColladaFromFile,
// @ref mochi_renderer::ReadObjFromFile, @ref mochi_renderer::ReadStlFromFile),
// mapping each format's solid material colors to glTF PBR factors (diffuse →
// baseColor; shininess/specular exponent → roughness; metallic = 0). Image
// textures are not supported and degrade to a default gray. Missing normals are
// computed.
//
// When `converter` is non-null the parsed geometry is transformed by it (via
// @ref mochi_renderer::ConvertMeshSectionsSpace) before the GLB is written; when
// null the geometry is written in whatever space its reader produced. This
// function makes no per-format assumptions about source space; the caller
// decides which inputs need conversion (e.g. supply a Mochi→renderer converter
// for OBJ/STL authored in Mochi space, and null for COLLADA, whose reader
// already emits renderer-space geometry).
//
// @param[in] inputPath Source mesh file. Must be .dae, .obj, or .stl.
// @param[in] outputPath Destination .glb file.
// @param[in] converter Optional source-space → renderer-space converter; applied
//            to the parsed geometry when non-null, ignored when null.
// @return true on success; false if the input is unsupported or parse/write fails.
bool ConvertToGlb(
    mochi::Path const& inputPath,
    mochi::Path const& outputPath,
    mochi::CoordinateSpaceConverter const* converter = nullptr);

// @brief Merge render mesh sections into a single triangle collision model.
//
// Concatenates every non-empty section's positions into one flat
// `ModelData::mesh` (`nodesPerElement = 3`), offsetting each section's indices by
// the running vertex count so the merged connectivity stays consistent. Per-vertex
// normals and material colors carry no representation in a Mochi collision mesh and
// are dropped. Sections with no positions or no indices are skipped.
//
// When `converter` is non-null, positions are transformed into its output space
// (e.g. a render-space→Mochi converter brings DAE/GLB geometry into Mochi space);
// when null, positions are merged verbatim. Indices are always merged verbatim.
//
// This is a pure data transform; it performs no validation. Callers that persist
// the result (e.g. via @ref mochi::model_utils::SaveToFile) own validation.
//
// @param[in] sections Render mesh sections (e.g. from a COLLADA or GLB reader).
// @param[in] converter Optional source-space → Mochi-space converter applied to
//            positions; null leaves them unchanged.
// @return ModelData whose `mesh` holds the merged triangle geometry. The `mesh`
//         is left unset when no section contributes geometry.
mochi::ModelData MeshSectionsToModel(
    std::vector<mochi_renderer::MeshSection> const& sections,
    mochi::CoordinateSpaceConverter const* converter = nullptr);

// @brief Convert a collision mesh (.stl, .obj, .off, or .ply) or a render mesh
// (.dae, .glb) to a .mochi.h5 file.
//
// Collision formats load directly via CGAL. Render formats are parsed into mesh
// sections (COLLADA via the in-tree reader, GLB via cgltf) and merged into a single
// triangle mesh through @ref MeshSectionsToModel; their material and normal data are
// dropped since a Mochi collision mesh carries none.
//
// When `converter` is non-null it is applied to the parsed render-format (.dae /
// .glb) geometry via @ref MeshSectionsToModel (e.g. a Filament→Mochi converter
// for render-space inputs). It does not apply to the collision formats, which
// CGAL loads directly into Mochi space. This function makes no per-format
// assumptions about source space; the caller decides which inputs need a
// converter.
//
// @param[in] inputPath Source mesh file. Must be .stl, .obj, .off, .ply, .dae, or .glb.
// @param[in] outputPath Destination .mochi.h5 file.
// @param[in] converter Optional source-space → Mochi-space converter applied to
//            parsed .dae / .glb geometry; null leaves geometry in its source space.
// @param[in] remesh When true, remesh the surface via @ref mochi::mesh::RemeshSurface
//            (default parameters) before saving, producing a clean watertight collision
//            mesh. Only applies to triangle meshes; if remeshing fails the original
//            geometry is saved unchanged.
// @param[in] bakeSdf When true, bake a signed distance field via @ref mochi::model_utils::BakeSdf
//            (default parameters) and store it alongside the mesh. Requires a triangle (or
//            tetrahedral) mesh; if baking fails the mesh is saved without an SDF.
// @return true on success; false if the input is unsupported or load/save fails.
bool ConvertToH5(
    mochi::Path const& inputPath,
    mochi::Path const& outputPath,
    mochi::CoordinateSpaceConverter const* converter = nullptr,
    bool remesh = true,
    bool bakeSdf = true);

} // namespace superdex::studio
