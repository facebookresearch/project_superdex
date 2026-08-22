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

// Shape-level tessellation shared by the STEP paths. Only compiled where OpenCascade is available
// (MOCHI_USE_OCCT); elsewhere this header is an empty translation unit.
#if MOCHI_USE_OCCT

#include "occ_isotropic_face_mesher.h" // FaceMeshFallbackStats

#include <mochi_mesh/mochi_mesh_cli_types.h>

#include <IMeshData_Status.hxx>

class Bnd_Box;
class TopoDS_Shape;

namespace mochi::mesh::cli::occ {

/// Per-face tessellation algorithm.
enum class FaceMesher {
  /// OpenCascade's stock BRepMesh. Driven purely by the deflections, which is fast but yields
  /// long slivers across flat regions and widely varying triangle size.
  Delabella,

  /// Uniform edge/curve sampling plus the CGAL constrained-Delaunay per-face mesher, which targets
  /// a uniform 3D edge length and so produces far more regular triangles. Faces the CGAL mesher
  /// cannot handle fall back to @ref Delabella individually.
  Isotropic,
};

/// @brief Parameters for @ref MeshShape.
struct ShapeMeshingParams {
  FaceMesher faceMesher = FaceMesher::Delabella;
  /// Maximum chordal (linear) deviation between a facet and the true surface [mm].
  double linearDeflection = 0.1;
  /// Maximum angular deviation between adjacent facet normals [rad].
  double angularDeflection = 0.5;
  /// Absolute target 3D edge length [mm]; resolve it with @ref ResolveTargetEdgeLength first.
  /// Read only for @ref FaceMesher::Isotropic.
  double targetEdgeLength = 0.0;
  /// Edge sampling strategy. Read only for @ref FaceMesher::Isotropic.
  StepMeshBodyParams::EdgeSampling edgeSampling = StepMeshBodyParams::EdgeSampling::Uniform;
};

/// Target uniform 3D edge length [mm]: @p explicitLength when positive, otherwise @p fraction of
/// @p bbox's diagonal. Floored so a void or degenerate box cannot produce zero.
///
/// Taking a Bnd_Box rather than a shape lets a caller accumulate one box over several shapes and
/// derive a single edge length for a whole assembly.
[[nodiscard]] double
ResolveTargetEdgeLength(Bnd_Box const& bbox, double explicitLength, double fraction);

/// Tessellates @p shape in place, leaving a Poly_Triangulation on each face.
///
/// @param[in,out] fallbackStats Tally of faces the CGAL mesher could not handle; may be null, is
///                              only written for @ref FaceMesher::Isotropic, and must outlive
///                              the call. Share one instance to total several shapes.
/// @return OpenCascade's mesh status flags.
IMeshData_Status MeshShape(
    TopoDS_Shape const& shape,
    ShapeMeshingParams const& params,
    FaceMeshFallbackStats* fallbackStats);

} // namespace mochi::mesh::cli::occ

#endif // MOCHI_USE_OCCT
