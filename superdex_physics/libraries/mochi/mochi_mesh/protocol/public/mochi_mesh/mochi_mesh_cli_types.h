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

// NOTE: Do not include headers from any mochi libraries in this file.
//       It is shared with superdex_mesh_cli, which does not depend on mochi libraries.

// Public wire-protocol types shared between mochi_mesh and the superdex_mesh_cli helper.
//
// Every struct here is serialized field by field; see PayloadWriter/PayloadReader in
// mochi_mesh/mochi_mesh_cli_encoding.h. Adding a field to any of these structs will fail to
// compile there until the matching Write/Read pair is updated.

#include <string>

namespace mochi::mesh {

/// @brief [Experimental] Available surface remeshing methods.
/// @warning This API may change in future releases.
enum class RemeshMethod {
  /// No remeshing — pass the input mesh through as-is. Use with post-processing
  /// parameters to improve triangle quality without changing mesh topology.
  None,

  /// Alpha-wrap the input to produce a watertight mesh, then improve quality
  /// via isotropic remeshing, tangential relaxation, and degenerate repair.
  ///
  /// @note The post-processing pass moves vertices and may break AlphaWrap's
  /// strict offset/containment guarantee. Set smoothingIterations,
  /// tangentialRelaxationIterations, and angleSmoothingIterations all to 0
  /// to preserve the raw alpha-wrapped output.
  AlphaWrap,

  /// Approximated Centroidal Voronoi Diagram remeshing.  Produces near-uniform
  /// triangulations with direct vertex-count control.  Requires Eigen.
  ///
  /// @note Operates directly on the input mesh — no AlphaWrap pre-pass.
  /// Requires a clean, manifold input. For non-manifold or self-intersecting
  /// inputs, run AlphaWrap separately first.
  ACVD,

  /// Surface Delaunay remeshing.  Provides guaranteed
  /// minimum-angle bounds and surface-approximation distance bounds.
  ///
  /// @note Operates directly on the input mesh — no AlphaWrap pre-pass.
  /// Requires a clean, manifold input. For non-manifold or self-intersecting
  /// inputs, run AlphaWrap separately first.
  SurfaceDelaunay,

  /// @internal Not a valid method — used for bounds checking.
  Count,
  Default = AlphaWrap,
};

/// @brief Closure strategy for @ref CloseMesh.
enum class MeshClosureMode {
  None, ///< Leave the surface open (e.g. for a render-only mesh).
  FillHoles, ///< Patch boundary loops (watertight).
  ShrinkWrap, ///< Enclose the surface with a deflating hull.
  ConvexHull, ///< Replace the surface with the convex hull of its vertices.
};

/// @brief [Experimental] Parameters for surface mesh remeshing.
/// @warning This API may change in future releases.
struct SurfaceRemeshingParams {
  // ---- Method selection ----
  RemeshMethod method = RemeshMethod::Default; ///< Remeshing backend to use

  // ---- Common parameters ----
  double edgeSize = 0.025; ///< Target edge length (fraction of avg bounding box dimension, or
                           ///< absolute if relativeToMeshSize=false)
  bool detectFeatures = true; ///< Detect sharp features and boundaries
  bool relativeToMeshSize = true; ///< Length params are fractions of avg bounding box dimension

  // ---- Alpha wrapping parameters ----
  double alphaWrapRelativeAlpha =
      1.0; ///< Alpha value as a multiple of edgeSize. Controls which features appear in the output;
           ///< smaller values preserve finer features but produce more triangles.
  double alphaWrapRelativeOffset = 0.0; ///< Offset as a multiple of edgeSize. Distance from the
                                        ///< input surface. If 0, uses a default of alpha / 30.

  // ---- Post-processing for triangle quality improvement ----
  int smoothingIterations =
      3; ///< Isotropic remeshing iterations for edge/valence regularization (0 to disable)
  int relaxationStepsPerIteration =
      3; ///< Tangential relaxation steps per isotropic remeshing iteration (higher = better vertex
         ///< placement, default was 1 before)
  int tangentialRelaxationIterations =
      3; ///< Standalone tangential relaxation iterations after isotropic remeshing (0 to disable).
         ///< Replaces angle smoothing by default; preserves surface shape without
         ///< self-intersections.
  int angleSmoothingIterations = 0; ///< Legacy angle-based smoothing iterations (0 to disable).
                                    ///< Only used when tangentialRelaxationIterations is 0. Can
                                    ///< create self-intersections on complex geometry.
  double sharpFeatureAngle =
      60.0; ///< Dihedral angle threshold for protecting sharp edges [degrees]

  // ---- Feature-constrained remeshing ----
  bool protectConstraints =
      false; ///< Hard-protect sharp edges during remeshing (no split/collapse)
  bool relaxConstraints = false; ///< Allow feature-edge vertices to slide along their polylines

  // ---- Adaptive sizing (curvature-based) ----
  bool useAdaptiveSizing = false; ///< Use curvature-based adaptive sizing field
  double adaptiveSizingTolerance = 0.01; ///< Error tolerance for adaptive sizing (fraction of avg
                                         ///< bbox, or absolute if relativeToMeshSize=false)
  double minEdgeSizeFactor = 0.25; ///< Min edge size as fraction of edgeSize
  double maxEdgeSizeFactor = 2.0; ///< Max edge size as fraction of edgeSize

  // ---- ACVD parameters ----
  int targetVertexCount = 0; ///< Target number of output vertices for ACVD (0 = auto-estimate from
                             ///< edgeSize and surface area)
  double acvdGradationFactor = 1.5; ///< Curvature-based gradation for ACVD vertex distribution (0 =
                                    ///< uniform, recommended 1.0-2.0 for adaptive)

  // ---- Surface Delaunay parameters ----
  double facetAngleBound = 25.0; ///< Lower bound on facet angles for Surface Delaunay [degrees].
                                 ///< Higher values produce better-shaped triangles but fewer
                                 ///< features are preserved. Maximum is 30°.
  double facetDistanceBound = 0.0; ///< Upper bound on distance from facet center to surface for
                                   ///< Surface Delaunay [coordinate units].
                                   ///< When relativeToMeshSize=true, this value is treated as
                                   ///< a fraction of the average bounding box dimension (consistent
                                   ///< with edgeSize).
                                   ///< If 0, auto-computed as edgeSize / 5.

  // ---- Mesh repair ----
  bool repairMesh = true; ///< Run best-effort mesh repair after quality improvement (degenerate
                          ///< removal, stitching, orientation, self-intersection repair). Never
                          ///< opens a watertight mesh; rolls back if repair would break invariants.
};

/// @brief Parameters controlling STEP tessellation (OpenCascade BRepMesh).
struct StepTessellationParams {
  /// Maximum chordal (linear) deviation between a facet and the true surface, in the STEP file's
  /// own model units. Smaller values yield a finer mesh.
  double linearDeflection = 0.1;
  /// Maximum angular deviation between adjacent facet normals [rad].
  double angularDeflection = 0.5;
};

/// @brief Parameters controlling the parameterized STEP body tessellation (OpenCascade BRepMesh
/// with the isotropic CGAL per-face mesher).
///
/// @note All lengths are in the STEP file's own model units (millimeters after OpenCascade
/// normalization); the returned mesh is converted to meters.
struct StepMeshBodyParams {
  /// Edge sampling strategy fed to the per-face mesher. Both modes place their samples at even
  /// arc-length intervals and differ only in what sets the count: `Uniform` takes it from arc
  /// length alone, while `Adaptive` raises it wherever the deflection tolerance demands it.
  enum class EdgeSampling { Uniform, Adaptive };

  /// Maximum chordal (linear) deviation between a facet and the true surface [mm]. Smaller is
  /// finer.
  double linearDeflection = 0.05;
  /// Maximum angular deviation between adjacent facet normals [rad].
  double angularDeflection = 0.25;
  /// Target uniform 3D edge length [mm]. When <= 0 it is derived from the bounding-box diagonal and
  /// @ref targetEdgeLengthFraction.
  double targetEdgeLength = 0.0;
  /// Fraction of the bounding-box diagonal used to auto-derive the target edge length when
  /// @ref targetEdgeLength <= 0.
  double targetEdgeLengthFraction = 0.02;
  /// Edge sampling strategy (see @ref EdgeSampling). Simulation meshes favour regular triangles
  /// over faithful small features, so this defaults to `Uniform` rather than the `Adaptive` the
  /// render-mesh path (@ref StepVisualExportParams) uses.
  EdgeSampling edgeSampling = EdgeSampling::Uniform;
  /// When true, faces that fail to mesh are skipped and a partial mesh is returned; when false, any
  /// face failure makes the whole call fail.
  bool allowPartialFailure = true;
  /// When true, solids that touch -- sharing a face, or within a small proximity tolerance -- are
  /// fused into one body before meshing, so a model authored as several parts tessellates as a
  /// single watertight surface. Set false to tessellate the solids exactly as authored: fusing is a
  /// boolean operation, and on geometry the kernel handles badly it is better skipped than
  /// attempted.
  bool combineTouchingSolids = true;
};

/// @brief Output file format for @ref VisualExportOutput.
enum class VisualMeshFormat {
  Glb, ///< Binary glTF
  Gltf, ///< Text glTF. Same content as @ref Glb.
  Obj, ///< Wavefront OBJ plus a companion .mtl.
  Stl, ///< Binary STL
  Count,
};

/// @brief One output file requested from @ref ExportStepVisual.
struct VisualExportOutput {
  /// Output file format.
  VisualMeshFormat format = VisualMeshFormat::Glb;
  /// Filesystem path to write (UTF-8). Its extension is not inspected -- @ref format selects the
  /// format.
  std::string path;
};

/// @brief Outcome of a single output of @ref ExportStepVisual.
enum class VisualExportStatus {
  Written, ///< File written; every face tessellated.
  WrittenPartial, ///< File written, but faces that failed to mesh were skipped.
  Failed, ///< File not written.
  Count,
};

/// @brief Face tessellation backend for @ref ExportStepVisual.
enum class CadMeshingBackend {
  /// OpenCascade's stock BRepMesh.
  Delabella,
  /// Uniform edge/curve sampling plus a CGAL constrained-Delaunay per-face triangulation.
  Isotropic,
  Count,
};

/// @brief Parameters controlling the STEP to visual-mesh conversion (@ref ExportStepVisual).
///
/// @note Unlike @ref StepMeshBodyParams, this path preserves the CAD file's per-body colors and
/// the surfaces' analytic normals, so the result is suitable for rendering rather than simulation.
struct StepVisualExportParams {
  /// Face tessellation backend (see @ref CadMeshingBackend)
  CadMeshingBackend backend = CadMeshingBackend::Isotropic;
  /// Maximum chordal (linear) deviation between a facet and the true surface [mm]. Smaller is
  /// finer.
  double linearDeflection = 0.1;
  /// Maximum angular deviation between adjacent facet normals [rad].
  double angularDeflection = 0.5;
  /// Target uniform 3D edge length [mm]. When <= 0 it is derived from the bounding-box diagonal of
  /// the whole file and @ref targetEdgeLengthFraction. Ignored by @ref
  /// CadMeshingBackend::Delabella.
  double targetEdgeLength = 0.0;
  /// Fraction of the bounding-box diagonal used to auto-derive the target edge length when
  /// @ref targetEdgeLength <= 0. Ignored by @ref CadMeshingBackend::Delabella.
  double targetEdgeLengthFraction = 0.02;
  /// Edge sampling strategy fed to the per-face mesher (see @ref StepMeshBodyParams::EdgeSampling).
  /// Adaptive is better suited for visual meshes at the cost of increased triangle count.
  /// Ignored by @ref CadMeshingBackend::Delabella.
  StepMeshBodyParams::EdgeSampling edgeSampling = StepMeshBodyParams::EdgeSampling::Adaptive;
  /// Multiplier applied to the millimeter-normalized STEP geometry. The default converts to
  /// meters, matching the rest of @ref mochi::mesh; use 1.0 for millimeters.
  double scale = 0.001;
  /// When true, faces that fail to tessellate are skipped and a partial file is still written;
  /// when false, any face failure makes the whole call fail.
  bool allowPartialFailure = true;
  /// Rename each glTF material to the `RRGGBBAA` hex of its base color. Ignored for
  /// @ref VisualMeshFormat::Obj and @ref VisualMeshFormat::Stl.
  bool rgbaMaterialNames = false;
};

/// @brief Stage 3 -- Closure parameters.
struct MeshClosureParams {
  MeshClosureMode mode = MeshClosureMode::FillHoles;
  /// Shrink-wrap tightness: low values stay near the convex hull (use MeshClosureMode::ConvexHull
  /// for the exact hull); 1 = deflated onto the input surface (its outer envelope, with concavities
  /// bridged and holes covered).
  double shrinkWrapTightness = 1.0;
  /// Final shrink-wrap pass snaps wrap vertices onto the original surface.
  bool shrinkWrapSnap = true;
  /// Target edge length for shrink-wrap remeshing [mesh units]. 0 = derive from fraction below.
  double shrinkWrapTargetEdgeLength = 0.0;
  /// When shrinkWrapTargetEdgeLength is 0, use this fraction of the bounding-box diagonal.
  double shrinkWrapTargetEdgeLengthFraction = 0.02;
};

/// @brief Stage 4 -- Edge swap parameters.
struct MeshEdgeSwapParams {
  /// Swap an edge only if flipping to the opposite diagonal reduces its midpoint's deviation from
  /// the reference surface by at least this fraction of the local edge length (0.1 = 10%). <= 0
  /// disables the swap.
  double relativeThreshold = 0.1;
  /// Maximum number of optimization passes.
  int maxPasses = 10;
};

/// @brief Stage 5 -- Decimate parameters.
struct MeshDecimateParams {
  /// Edges shorter than this are collapsed [mesh units]. <= 0 disables decimation.
  double collapseDistance = 0.0;
};

} // namespace mochi::mesh
