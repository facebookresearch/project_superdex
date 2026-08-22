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

#include "cgal_mesh_utils.h"
#include "mesh_cli_geometry.h"

#if MOCHI_USE_EIGEN
#include <CGAL/Polygon_mesh_processing/Adaptive_sizing_field.h>
#include <CGAL/Polygon_mesh_processing/approximated_centroidal_Voronoi_diagram_remeshing.h>
#endif
#include <CGAL/Polygon_mesh_processing/angle_and_area_smoothing.h>
#include <CGAL/Polygon_mesh_processing/detect_features.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/repair_degeneracies.h>
#include <CGAL/Polygon_mesh_processing/repair_self_intersections.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/surface_Delaunay_remeshing.h>
#include <CGAL/Polygon_mesh_processing/tangential_relaxation.h>
#include <CGAL/alpha_wrap_3.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <vector>

using namespace mochi::mesh::cli;

using K = mochi::mesh::cli::cgal_utils::K;
using SurfaceMesh = mochi::mesh::cli::cgal_utils::CgalSurfaceMesh;

void cgal_utils::RepairMesh(SurfaceMesh& sm) {
  namespace PMP = CGAL::Polygon_mesh_processing;

  bool const wasClosed = CGAL::is_closed(sm);
  SurfaceMesh entryBackup = sm;

  try {
    // Save state before potentially destructive degenerate removal.
    SurfaceMesh backup = sm;

    // Remove degenerate elements.
    PMP::remove_degenerate_edges(sm);
    PMP::remove_degenerate_faces(sm);
    sm.collect_garbage();

    // Remove near-degenerate triangles (needles with high aspect ratio, caps with obtuse angles).
    PMP::remove_almost_degenerate_faces(sm);
    sm.collect_garbage();

    // If degenerate removal opened a watertight mesh, rollback.
    if (wasClosed && !CGAL::is_closed(sm)) {
      MOCHI_MESH_CLI_LOG_WARNING(
          "Degenerate removal opened a watertight mesh - reverting to pre-repair state.");
      sm = std::move(backup);
      return;
    }

    // Stitch borders on open meshes.
    if (!CGAL::is_closed(sm)) {
      PMP::stitch_borders(sm);
      sm.collect_garbage();
    }

    // Fix non-manifold vertices on all meshes (the polygon soup fallback in
    // MeshDataToSurfaceMesh can produce non-manifold vertices on any mesh).
    PMP::duplicate_non_manifold_vertices(sm);
    sm.collect_garbage();

    // Fix orientation on closed meshes.
    if (CGAL::is_closed(sm) && !PMP::is_outward_oriented(sm)) {
      PMP::reverse_face_orientations(sm);
    }

    // Best-effort self-intersection repair with rollback.
    if (PMP::does_self_intersect(sm)) {
      MOCHI_MESH_CLI_LOG_WARNING("Mesh has self-intersections; attempting repair.");
      SurfaceMesh preRepair = sm;
      bool const repaired = PMP::experimental::remove_self_intersections(sm);
      sm.collect_garbage();

      if (!repaired || PMP::does_self_intersect(sm)) {
        MOCHI_MESH_CLI_LOG_WARNING(
            "Could not fully repair self-intersections - keeping pre-repair mesh.");
        sm = std::move(preRepair);
      } else if (wasClosed && !CGAL::is_closed(sm)) {
        MOCHI_MESH_CLI_LOG_WARNING(
            "Self-intersection repair opened a watertight mesh - reverting.");
        sm = std::move(preRepair);
      } else {
        // Repair succeeded. Fix orientation if needed.
        if (CGAL::is_closed(sm) && !PMP::is_outward_oriented(sm)) {
          PMP::reverse_face_orientations(sm);
        }
      }
    }
  } catch (std::exception const& e) {
    MOCHI_MESH_CLI_LOG_WARNING("Mesh repair failed (%s); keeping pre-repair mesh.", e.what());
    sm = std::move(entryBackup);
  } catch (...) {
    MOCHI_MESH_CLI_LOG_WARNING(
        "Mesh repair failed with unknown exception; keeping pre-repair mesh.");
    sm = std::move(entryBackup);
  }
}

namespace {

// Build a watertight Surface_mesh from a triangle soup via alpha wrapping.
// This works on any input (non-manifold, open, self-intersecting).
bool BuildSurfaceMeshViaAlphaWrap(
    MeshData const& surfaceMesh,
    double alpha,
    double offset,
    SurfaceMesh& sm,
    CliError& error) {
  std::vector<K::Point_3> points;
  std::vector<std::vector<std::size_t>> faces;
  if (!cgal_utils::MeshDataToPolygonSoup(surfaceMesh, points, faces, error)) {
    return false;
  }

  CGAL::alpha_wrap_3(points, faces, alpha, offset, sm);

  MOCHI_MESH_CLI_ERROR_IF(
      sm.number_of_vertices() == 0, error, "Alpha wrapping produced no vertices.");
  MOCHI_MESH_CLI_ERROR_IF(sm.number_of_faces() == 0, error, "Alpha wrapping produced no faces.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, false);

  return true;
}

/// Improve triangle quality of an existing Surface_mesh via isotropic remeshing,
/// tangential relaxation (or legacy angle smoothing), and optional mesh repair.
void ImproveSurfaceMeshQuality(
    SurfaceMesh& sm,
    SurfaceRemeshingParams const& params,
    double edgeSize,
    [[maybe_unused]] double adaptiveTolerance) {
  namespace PMP = CGAL::Polygon_mesh_processing;

  if (CGAL::is_closed(sm) && !PMP::is_outward_oriented(sm)) {
    PMP::reverse_face_orientations(sm);
  }

  auto featureEdgeMap =
      sm.add_property_map<SurfaceMesh::Edge_index, bool>("e:is_feature", false).first;

  if (params.detectFeatures) {
    PMP::detect_sharp_edges(sm, params.sharpFeatureAngle, featureEdgeMap);
  }

  // Step 1: Isotropic remeshing with feature constraints.
  if (params.smoothingIterations > 0) {
    if (params.protectConstraints && params.detectFeatures) {
      std::vector<SurfaceMesh::Edge_index> constrainedEdges;
      for (auto const e : sm.edges()) {
        if (get(featureEdgeMap, e)) {
          constrainedEdges.push_back(e);
        }
      }
      PMP::split_long_edges(constrainedEdges, edgeSize, sm);

      for (auto const e : sm.edges()) {
        put(featureEdgeMap, e, false);
      }
      PMP::detect_sharp_edges(sm, params.sharpFeatureAngle, featureEdgeMap);
    }

#if MOCHI_USE_EIGEN
    if (params.useAdaptiveSizing) {
      double const minEdge = edgeSize * params.minEdgeSizeFactor;
      double const maxEdge = edgeSize * params.maxEdgeSizeFactor;

      PMP::Adaptive_sizing_field<SurfaceMesh> sizing(
          adaptiveTolerance, std::make_pair(minEdge, maxEdge), faces(sm), sm);

      PMP::isotropic_remeshing(
          faces(sm),
          sizing,
          sm,
          PMP::parameters::number_of_iterations(params.smoothingIterations)
              .number_of_relaxation_steps(params.relaxationStepsPerIteration)
              .edge_is_constrained_map(featureEdgeMap)
              .protect_constraints(params.protectConstraints)
              .relax_constraints(params.relaxConstraints));
    } else
#else
    if (params.useAdaptiveSizing) {
      MOCHI_MESH_CLI_LOG_WARNING(
          "Adaptive sizing requested but Eigen is not available (MOCHI_USE_EIGEN=0). "
          "Falling back to uniform edge sizing.");
    }
#endif
    {
      PMP::isotropic_remeshing(
          faces(sm),
          edgeSize,
          sm,
          PMP::parameters::number_of_iterations(params.smoothingIterations)
              .number_of_relaxation_steps(params.relaxationStepsPerIteration)
              .edge_is_constrained_map(featureEdgeMap)
              .protect_constraints(params.protectConstraints)
              .relax_constraints(params.relaxConstraints));
    }

    sm.collect_garbage();

    if (params.detectFeatures) {
      for (auto const e : sm.edges()) {
        put(featureEdgeMap, e, false);
      }
      PMP::detect_sharp_edges(sm, params.sharpFeatureAngle, featureEdgeMap);
    }
  }

  // Step 2: Surface smoothing with feature edge constraints.
  if (params.tangentialRelaxationIterations > 0) {
    PMP::tangential_relaxation(
        sm,
        PMP::parameters::number_of_iterations(params.tangentialRelaxationIterations)
            .edge_is_constrained_map(featureEdgeMap));
  } else if (params.angleSmoothingIterations > 0) {
    // Legacy angle-based smoothing (backward compatibility).
    PMP::angle_and_area_smoothing(
        faces(sm),
        sm,
        PMP::parameters::number_of_iterations(params.angleSmoothingIterations)
            .use_angle_smoothing(true)
            .use_area_smoothing(false)
            .use_Delaunay_flips(true)
            .use_safety_constraints(true)
            .edge_is_constrained_map(featureEdgeMap));
  }

  // Step 3: Optional best-effort mesh repair.
  if (params.repairMesh) {
    cgal_utils::RepairMesh(sm);
  }
}

#if MOCHI_USE_EIGEN
/// Remesh via Approximated Centroidal Voronoi Diagram.
/// Produces near-uniform triangulations with direct vertex-count control.
bool RemeshViaACVD(
    SurfaceMesh& sm,
    double edgeSize,
    int targetVertexCount,
    double gradationFactor,
    CliError& error) {
  namespace PMP = CGAL::Polygon_mesh_processing;

  std::size_t nbVertices = 0;
  if (targetVertexCount > 0) {
    nbVertices = static_cast<std::size_t>(targetVertexCount);
  } else {
    // Estimate from surface area and target edge size, clamped to a sane range.
    // Each vertex in a regular triangulation "owns" ~(e^2)*sqrt(3)/2 of surface area
    // (its dual cell). For a closed manifold, F ~~ 2V (Euler), and triangle area
    // is (e^2)*sqrt(3)/4, so area-per-vertex is twice the triangle area.
    constexpr std::size_t kAcvdMinAutoVertices = 12;
    constexpr std::size_t kAcvdMaxAutoVertices = 1'000'000;
    double const area = PMP::area(sm);
    double const areaPerVertex = edgeSize * edgeSize * std::sqrt(3.0) / 2.0;
    auto const estimated = static_cast<std::size_t>(area / areaPerVertex);
    nbVertices = std::clamp(estimated, kAcvdMinAutoVertices, kAcvdMaxAutoVertices);
    if (nbVertices == kAcvdMaxAutoVertices) {
      MOCHI_MESH_CLI_LOG_WARNING(
          "ACVD auto-vertex-count clamped to maximum (%zu). Consider setting targetVertexCount explicitly.",
          kAcvdMaxAutoVertices);
    }
  }

  PMP::approximated_centroidal_Voronoi_diagram_remeshing(
      sm, nbVertices, PMP::parameters::gradation_factor(gradationFactor));

  MOCHI_MESH_CLI_ERROR_IF(
      sm.number_of_vertices() == 0, error, "ACVD remeshing produced no vertices.");
  MOCHI_MESH_CLI_ERROR_IF(sm.number_of_faces() == 0, error, "ACVD remeshing produced no faces.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, false);

  return true;
}
#endif

/// Remesh via Surface Delaunay remeshing (CGAL Mesh_3).
/// Provides guaranteed minimum-angle bounds and surface-approximation distance bounds.
bool RemeshViaSurfaceDelaunay(
    SurfaceMesh const& inputSm,
    SurfaceMesh& outputSm,
    double edgeSize,
    double facetAngleBound,
    double facetDistance,
    double sharpFeatureAngle,
    bool protectConstraints,
    CliError& error) {
  namespace PMP = CGAL::Polygon_mesh_processing;

  outputSm = PMP::surface_Delaunay_remeshing(
      inputSm,
      PMP::parameters::mesh_edge_size(edgeSize)
          .mesh_facet_size(edgeSize)
          .mesh_facet_angle(facetAngleBound)
          .mesh_facet_distance(facetDistance)
          .features_angle_bound(sharpFeatureAngle)
          .protect_constraints(protectConstraints));

  MOCHI_MESH_CLI_ERROR_IF(
      outputSm.number_of_vertices() == 0,
      error,
      "Surface Delaunay remeshing produced no vertices.");
  MOCHI_MESH_CLI_ERROR_IF(
      outputSm.number_of_faces() == 0, error, "Surface Delaunay remeshing produced no faces.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, false);

  return true;
}

} // namespace

MeshData mochi::mesh::cli::RemeshSurface(
    MeshData const& surfaceMesh,
    SurfaceRemeshingParams const& params,
    CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});

  MOCHI_MESH_CLI_ERROR_IF(
      surfaceMesh.nodesPerElement != 3,
      error,
      "Input mesh must be a triangle mesh (nodesPerElement == 3).");
  MOCHI_MESH_CLI_ERROR_IF(surfaceMesh.coordinates.empty(), error, "Input mesh has no vertices.");
  MOCHI_MESH_CLI_ERROR_IF(surfaceMesh.connectivity.empty(), error, "Input mesh has no faces.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});

  // Validate parameters needed for any actual remeshing or post-processing.
  // Use !(x > 0.0) and std::isfinite to also reject NaN, which IEEE 754 comparisons silently pass.
  MOCHI_MESH_CLI_ERROR_IF(
      !(params.edgeSize > 0.0) || !std::isfinite(params.edgeSize),
      error,
      "edgeSize must be positive and finite.");

  MOCHI_MESH_CLI_ERROR_IF(
      params.smoothingIterations < 0, error, "smoothingIterations must be non-negative.");
  MOCHI_MESH_CLI_ERROR_IF(
      params.relaxationStepsPerIteration < 0,
      error,
      "relaxationStepsPerIteration must be non-negative.");
  MOCHI_MESH_CLI_ERROR_IF(
      params.tangentialRelaxationIterations < 0,
      error,
      "tangentialRelaxationIterations must be non-negative.");
  MOCHI_MESH_CLI_ERROR_IF(
      params.angleSmoothingIterations < 0, error, "angleSmoothingIterations must be non-negative.");
  MOCHI_MESH_CLI_ERROR_IF(
      !(params.sharpFeatureAngle > 0.0) || !std::isfinite(params.sharpFeatureAngle) ||
          params.sharpFeatureAngle > 180.0,
      error,
      "sharpFeatureAngle must be in (0, 180] degrees.");
  MOCHI_MESH_CLI_ERROR_IF(
      params.useAdaptiveSizing &&
          (!(params.adaptiveSizingTolerance > 0.0) ||
           !std::isfinite(params.adaptiveSizingTolerance)),
      error,
      "adaptiveSizingTolerance must be positive and finite when useAdaptiveSizing is true.");
  MOCHI_MESH_CLI_ERROR_IF(
      params.useAdaptiveSizing &&
          (!(params.minEdgeSizeFactor > 0.0) || !std::isfinite(params.minEdgeSizeFactor)),
      error,
      "minEdgeSizeFactor must be positive and finite when useAdaptiveSizing is true.");
  MOCHI_MESH_CLI_ERROR_IF(
      params.useAdaptiveSizing &&
          (!std::isfinite(params.maxEdgeSizeFactor) ||
           params.maxEdgeSizeFactor <= params.minEdgeSizeFactor),
      error,
      "maxEdgeSizeFactor must be finite and greater than minEdgeSizeFactor when useAdaptiveSizing is true.");
  MOCHI_MESH_CLI_ERROR_IF(
      params.method == RemeshMethod::ACVD && params.targetVertexCount < 0,
      error,
      "targetVertexCount must be non-negative for ACVD.");
  MOCHI_MESH_CLI_ERROR_IF(
      params.method == RemeshMethod::ACVD && params.acvdGradationFactor < 0.0,
      error,
      "acvdGradationFactor must be non-negative for ACVD.");
  MOCHI_MESH_CLI_ERROR_IF(
      params.method == RemeshMethod::SurfaceDelaunay &&
          (!(params.facetAngleBound > 0.0) || !std::isfinite(params.facetAngleBound) ||
           params.facetAngleBound > 30.0),
      error,
      "facetAngleBound must be in (0, 30] degrees and finite for Surface Delaunay.");
  MOCHI_MESH_CLI_ERROR_IF(
      params.method == RemeshMethod::SurfaceDelaunay &&
          (!(params.facetDistanceBound >= 0.0) || !std::isfinite(params.facetDistanceBound)),
      error,
      "facetDistanceBound must be non-negative and finite.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});
  try {
    int const numVertices = GetNumNodes(surfaceMesh);

    // Compute bounding box and scale length-dependent parameters.
    double scale = 1.0;
    if (params.relativeToMeshSize && numVertices > 0) {
      Bounds3d const aabb = CalcAabb(surfaceMesh.coordinates);
      Vector3d const size = aabb.GetSize();
      scale = (static_cast<double>(size[0]) + static_cast<double>(size[1]) +
               static_cast<double>(size[2])) /
          3.0;
      MOCHI_MESH_CLI_ERROR_IF(
          scale <= 0.0, error, "Mesh has zero or negative bounding box extent.");
      MOCHI_MESH_CLI_ERROR_RETURN(error, {});
    }

    double const edgeSize = params.edgeSize * scale;
    double const adaptiveTolerance = params.adaptiveSizingTolerance * scale;
    SurfaceMesh sm;

    switch (params.method) {
      case RemeshMethod::None: {
        bool meshNeedsRepair = false;
        sm = cgal_utils::MeshDataToSurfaceMesh(surfaceMesh, error, &meshNeedsRepair);
        MOCHI_MESH_CLI_ERROR_RETURN(error, {});
        if (meshNeedsRepair) {
          cgal_utils::RepairMesh(sm);
        }
        break;
      }

      case RemeshMethod::AlphaWrap: {
        // Validate AlphaWrap-specific parameters. Use !(x > 0.0) and std::isfinite to
        // also reject NaN, which IEEE 754 comparisons silently pass.
        MOCHI_MESH_CLI_ERROR_IF(
            !(params.alphaWrapRelativeAlpha > 0.0) || !std::isfinite(params.alphaWrapRelativeAlpha),
            error,
            "alphaWrapRelativeAlpha must be positive and finite.");
        MOCHI_MESH_CLI_ERROR_IF(
            params.alphaWrapRelativeOffset < 0.0 || !std::isfinite(params.alphaWrapRelativeOffset),
            error,
            "alphaWrapRelativeOffset must be non-negative and finite.");
        MOCHI_MESH_CLI_ERROR_RETURN(error, {});

        double const alphaWrapAlpha = edgeSize * params.alphaWrapRelativeAlpha;
        constexpr double kAlphaWrapDefaultOffsetToAlphaRatio = 1.0 / 30.0; // CGAL default
        double const alphaWrapOffset = params.alphaWrapRelativeOffset > 0.0
            ? edgeSize * params.alphaWrapRelativeOffset
            : alphaWrapAlpha * kAlphaWrapDefaultOffsetToAlphaRatio;

        if (!BuildSurfaceMeshViaAlphaWrap(
                surfaceMesh, alphaWrapAlpha, alphaWrapOffset, sm, error)) {
          return {};
        }
        break;
      }

      case RemeshMethod::ACVD: {
#if MOCHI_USE_EIGEN
        namespace PMP = CGAL::Polygon_mesh_processing;
        bool meshNeedsRepair = false;
        sm = cgal_utils::MeshDataToSurfaceMesh(surfaceMesh, error, &meshNeedsRepair);
        MOCHI_MESH_CLI_ERROR_RETURN(error, {});
        if (meshNeedsRepair) {
          cgal_utils::RepairMesh(sm);
        }

        // ACVD requires a closed, manifold, non-self-intersecting triangle mesh.
        // Without these guarantees, the underlying CGAL implementation may hang
        // or produce invalid results.
        MOCHI_MESH_CLI_ERROR_IF(
            !CGAL::is_closed(sm),
            error,
            "ACVD requires a closed (watertight) mesh. Consider using AlphaWrap first to repair the input.");
        MOCHI_MESH_CLI_ERROR_IF(
            PMP::does_self_intersect(sm),
            error,
            "ACVD requires a non-self-intersecting mesh. Consider using AlphaWrap first to repair the input.");
        MOCHI_MESH_CLI_ERROR_IF(
            !PMP::is_outward_oriented(sm),
            error,
            "ACVD requires an outward-oriented mesh. Consider using AlphaWrap first to repair the input.");
        MOCHI_MESH_CLI_ERROR_RETURN(error, {});

        if (!RemeshViaACVD(
                sm, edgeSize, params.targetVertexCount, params.acvdGradationFactor, error)) {
          return {};
        }
#else
        MOCHI_MESH_CLI_ERROR_SET(
            error,
            "ACVD remeshing requires Eigen (MOCHI_USE_EIGEN=1). "
            "Rebuild with Eigen enabled or use a different method.");
        return {};
#endif
        break;
      }

      case RemeshMethod::SurfaceDelaunay: {
        bool meshNeedsRepair = false;
        sm = cgal_utils::MeshDataToSurfaceMesh(surfaceMesh, error, &meshNeedsRepair);
        MOCHI_MESH_CLI_ERROR_RETURN(error, {});
        if (meshNeedsRepair) {
          cgal_utils::RepairMesh(sm);
        }
        double const facetDistance =
            params.facetDistanceBound > 0.0 ? params.facetDistanceBound * scale : edgeSize / 5.0;
        SurfaceMesh outputSm;
        if (!RemeshViaSurfaceDelaunay(
                sm,
                outputSm,
                edgeSize,
                params.facetAngleBound,
                facetDistance,
                params.sharpFeatureAngle,
                params.protectConstraints,
                error)) {
          return {};
        }
        sm = std::move(outputSm);
        break;
      }

      case RemeshMethod::Count:
        MOCHI_MESH_CLI_ERROR_SET(error, "Invalid RemeshMethod");
        return {};
    }

    // Shared post-processing for all methods (no-op when all iterations are 0 and repair is off).
    ImproveSurfaceMeshQuality(sm, params, edgeSize, adaptiveTolerance);

    // Convert Surface_mesh -> MeshData
    MeshData result = cgal_utils::SurfaceMeshToMeshData(sm);

    MOCHI_MESH_CLI_ERROR_IF(
        result.coordinates.empty(), error, "Surface remeshing produced no vertices.");
    MOCHI_MESH_CLI_ERROR_IF(
        result.connectivity.empty(), error, "Surface remeshing produced no faces.");
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});
    return result;

  } catch (std::exception const& e) {
    MOCHI_MESH_CLI_LOG_WARNING("CGAL surface remeshing failed: %s", e.what());
    MOCHI_MESH_CLI_ERROR_SET(error, "CGAL surface remeshing failed with an exception.");
  } catch (...) {
    MOCHI_MESH_CLI_ERROR_SET(error, "CGAL surface remeshing failed with an unknown exception.");
  }
  return {};
}
