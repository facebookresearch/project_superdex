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

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/utils/error.h>
#include <mochi_mesh/mochi_mesh_cli_types.h>

// Mesh-level processing stages of the Model Editor's CAD pipeline (Stages 2-5). Each operates on a
// neutral triangle MeshData and returns a new MeshData; the CGAL implementations are isolated in
// the superdex_mesh_cli helper, so these calls marshal across the process boundary. They are unit
// agnostic -- distance thresholds are in the mesh's own units (meters, after Stage 1).

namespace mochi::mesh {

/// @brief Stage 2 -- Cleanup. Welds coincident vertices, removes internal/overlapping faces from
/// flush mating surfaces, and repairs non-manifold topology. No parameters.
///
/// @param[in] mesh Input triangle mesh (typically the Stage 1 raw soup).
/// @param[in,out] error Error status.
/// @return Cleaned triangle mesh; empty on error.
[[nodiscard]] MeshData CleanupMesh(MeshDataView const& mesh, Error& error);

/// @brief Stage 3 -- Closure. Fills holes, shrink-wraps, takes the convex hull, or leaves the
/// surface open.
///
/// @param[in] mesh Input triangle mesh (typically the Stage 2 cleaned mesh).
/// @param[in] params Closure parameters.
/// @param[in,out] error Error status.
/// @return Closed triangle mesh; empty on error.
[[nodiscard]] MeshData
CloseMesh(MeshDataView const& mesh, MeshClosureParams const& params, Error& error);

/// @brief Stage 4 -- Edge swap. Flips the shared edge of a triangle pair to the opposite diagonal
/// when that better follows a reference surface, re-fitting the mesh toward the true STEP-surface /
/// Stage 1 tessellation rather than the mesh's own approximation. Deviation is measured by ray-
/// casting along the triangle-pair average normal. Topology-safe: cannot open holes or create
/// non-manifold edges.
///
/// @param[in] mesh Input triangle mesh to optimize (typically the Stage 3 mesh).
/// @param[in] reference High-resolution reference surface to re-fit toward (typically the Stage 1
/// tessellation); an AABB tree over its triangles is queried by ray.
/// @param[in] params Edge-swap parameters.
/// @param[in,out] error Error status.
/// @return Edge-swapped triangle mesh; empty on error.
[[nodiscard]] MeshData EdgeSwapMesh(
    MeshDataView const& mesh,
    MeshDataView const& reference,
    MeshEdgeSwapParams const& params,
    Error& error);

/// @brief Stage 5 -- Decimate. Collapses short edges (CGAL Surface Mesh Simplification with an
/// edge-length stop predicate). Link-condition safe: will not create non-manifold geometry.
///
/// @param[in] mesh Input triangle mesh (typically the Stage 4 mesh).
/// @param[in] params Decimation parameters.
/// @param[in,out] error Error status.
/// @return Decimated triangle mesh; empty on error.
[[nodiscard]] MeshData
DecimateMesh(MeshDataView const& mesh, MeshDecimateParams const& params, Error& error);

} // namespace mochi::mesh
