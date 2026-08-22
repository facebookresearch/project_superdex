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

#include "mesh_cli_compat.h"

#include <mochi_mesh/mochi_mesh_cli_encoding.h>
#include <mochi_mesh/mochi_mesh_cli_types.h>

#include <span>
#include <string_view>
#include <vector>

// CLI-side geometry domain declarations: the geometry algorithm entry points implemented against
// CGAL/OCCT in this target. The parameter structs and the scalar field that cross the process
// boundary live in mochi_mesh/mochi_mesh_cli_types.h and mochi_mesh/mochi_mesh_cli_encoding.h; both
// sides include those headers, so the on-wire memory image cannot silently drift.

namespace mochi::mesh::cli {

inline int GetNumNodes(MeshData const& mesh) {
  return static_cast<int>(mesh.coordinates.size() / 3);
}

inline int GetNumElements(MeshData const& mesh) {
  return mesh.nodesPerElement > 0
      ? static_cast<int>(mesh.connectivity.size() / mesh.nodesPerElement)
      : 0;
}

// Pull the shared param structs into the cli namespace so the algorithm implementations can
// continue to reference them unqualified.
using mochi::mesh::CadMeshingBackend;
using mochi::mesh::MeshClosureMode;
using mochi::mesh::MeshClosureParams;
using mochi::mesh::MeshDecimateParams;
using mochi::mesh::MeshEdgeSwapParams;
using mochi::mesh::RemeshMethod;
using mochi::mesh::StepMeshBodyParams;
using mochi::mesh::StepTessellationParams;
using mochi::mesh::StepVisualExportParams;
using mochi::mesh::SurfaceRemeshingParams;
using mochi::mesh::VisualExportOutput;
using mochi::mesh::VisualExportStatus;
using mochi::mesh::VisualMeshFormat;

[[nodiscard]] MeshData
RemeshSurface(MeshData const& surfaceMesh, SurfaceRemeshingParams const& params, CliError& error);
[[nodiscard]] MeshData ReconstructSurfaceFromSdf(ScalarField3d const& sdfData, CliError& error);
[[nodiscard]] MeshData TessellateStep(
    std::string_view stepFilePath,
    StepTessellationParams const& params,
    CliError& error);
[[nodiscard]] MeshData
MeshStepBody(std::string_view stepFilePath, StepMeshBodyParams const& params, CliError& error);
/// Loads and tessellates @p stepFilePath once, then writes every entry of @p outputs from that one
/// tessellation. @p params is shared by all outputs; the output format is per entry. On success
/// @p outStatuses has one entry per output, in order; a single output that cannot be written is
/// reported as @ref VisualExportStatus::Failed rather than failing the call. An error that prevents
/// any output from being produced (unreadable STEP, invalid parameters, a failed tessellation) is
/// reported through @p error instead.
[[nodiscard]] bool ExportStepVisual(
    std::string_view stepFilePath,
    std::span<VisualExportOutput const> outputs,
    StepVisualExportParams const& params,
    std::vector<VisualExportStatus>& outStatuses,
    CliError& error);
[[nodiscard]] MeshData CleanupMesh(MeshData const& mesh, CliError& error);
[[nodiscard]] MeshData
CloseMesh(MeshData const& mesh, MeshClosureParams const& params, CliError& error);
[[nodiscard]] MeshData EdgeSwapMesh(
    MeshData const& mesh,
    MeshData const& reference,
    MeshEdgeSwapParams const& params,
    CliError& error);
[[nodiscard]] MeshData
DecimateMesh(MeshData const& mesh, MeshDecimateParams const& params, CliError& error);

} // namespace mochi::mesh::cli
