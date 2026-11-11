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

#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#include <CGAL/Random.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/make_mesh_3.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <numbers>
#include <string>
#include <unordered_map>

using namespace mochi::mesh::cli;

namespace {

using K = cgal_utils::K;
using SurfaceMesh = cgal_utils::CgalSurfaceMesh;
using Domain = CGAL::Polyhedral_mesh_domain_with_features_3<K, SurfaceMesh>;
using Tr = CGAL::Mesh_triangulation_3<Domain, CGAL::Default, CGAL::Sequential_tag>::type;
using C3t3 = CGAL::Mesh_complex_3_in_triangulation_3<Tr, Domain::Corner_index, Domain::Curve_index>;
using Criteria = CGAL::Mesh_criteria_3<Tr>;

// Emits only the vertices of cells in the complex: Mesh_3 leaves vertices in the triangulation
// that no output tetrahedron uses, and soft actors reject unreferenced nodes. Nodes are numbered in
// order of first use so the output does not depend on allocation addresses.
MeshData ExtractTetrahedra(C3t3 const& c3t3) {
  Tr const& tr = c3t3.triangulation();
  auto const constructPoint = tr.geom_traits().construct_point_3_object();

  MeshData result;
  result.nodesPerElement = 4;
  result.connectivity.reserve(4 * c3t3.number_of_cells_in_complex());
  std::unordered_map<Tr::Vertex const*, int32_t> nodeIndex;
  for (auto cit = c3t3.cells_in_complex_begin(); cit != c3t3.cells_in_complex_end(); ++cit) {
    for (int i = 0; i < 4; ++i) {
      Tr::Vertex_handle const vertex = cit->vertex(i);
      auto const [it, inserted] =
          nodeIndex.try_emplace(&*vertex, static_cast<int32_t>(nodeIndex.size()));
      if (inserted) {
        K::Point_3 const point = constructPoint(vertex->point());
        result.coordinates.push_back(point.x());
        result.coordinates.push_back(point.y());
        result.coordinates.push_back(point.z());
      }
      result.connectivity.push_back(it->second);
    }
  }
  return result;
}

} // namespace

SurfaceMesh mochi::mesh::cli::cgal_utils::TetrahedralizationInputSurface(
    MeshData const& surfaceMesh,
    bool allowRepair,
    CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});
  MOCHI_MESH_CLI_ERROR_IF(
      surfaceMesh.nodesPerElement != 3,
      error,
      "Input mesh must be a triangle mesh (nodesPerElement == 3).");
  MOCHI_MESH_CLI_ERROR_IF(surfaceMesh.coordinates.empty(), error, "Input mesh has no vertices.");
  MOCHI_MESH_CLI_ERROR_IF(surfaceMesh.connectivity.empty(), error, "Input mesh has no faces.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});

  bool needsRepair = false;
  SurfaceMesh sm = MeshDataToSurfaceMesh(surfaceMesh, error, &needsRepair);
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});
  MOCHI_MESH_CLI_ERROR_IF(
      needsRepair && !allowRepair,
      error,
      "Input surface has non-manifold edges or inconsistently oriented faces. Clean it up first "
      "(e.g. Make Manifold).");
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});
  MOCHI_MESH_CLI_ERROR_IF(
      !CGAL::is_closed(sm),
      error,
      "Input surface must be closed. Close it first (e.g. Make Watertight or Alpha Wrap).");
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});
  MOCHI_MESH_CLI_ERROR_IF(
      CGAL::Polygon_mesh_processing::does_self_intersect(sm),
      error,
      "Input surface self-intersects. Clean it up first (e.g. Alpha Wrap).");
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});
  return sm;
}

double mochi::mesh::cli::cgal_utils::AverageFaceBoundingBoxExtent(
    SurfaceMesh const& sm,
    CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, 0.0);
  CGAL::Bbox_3 bounds;
  for (auto const face : sm.faces()) {
    bounds += CGAL::Polygon_mesh_processing::face_bbox(face, sm);
  }
  double const extent = ((bounds.xmax() - bounds.xmin()) + (bounds.ymax() - bounds.ymin()) +
                         (bounds.zmax() - bounds.zmin())) /
      3.0;
  MOCHI_MESH_CLI_ERROR_IF(
      !(extent > 0.0) || !std::isfinite(extent),
      error,
      "Mesh average bounding box extent must be positive and finite.");
  return extent;
}

MeshData mochi::mesh::cli::Tetrahedralize(
    MeshData const& surfaceMesh,
    TetrahedralizationParams const& params,
    CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});
  MOCHI_MESH_CLI_ERROR_IF(
      !(params.edgeSize > 0.0) || !std::isfinite(params.edgeSize),
      error,
      "edgeSize must be positive and finite.");
  MOCHI_MESH_CLI_ERROR_IF(
      !(params.facetDistanceBound >= 0.0) || !std::isfinite(params.facetDistanceBound),
      error,
      "facetDistanceBound must be non-negative and finite.");
  MOCHI_MESH_CLI_ERROR_IF(
      !(params.facetAngleBound > 0.0) || params.facetAngleBound > 30.0,
      error,
      "facetAngleBound must be in (0, 30] degrees.");
  MOCHI_MESH_CLI_ERROR_IF(
      !(params.cellRadiusEdgeRatio >= 2.0) || !std::isfinite(params.cellRadiusEdgeRatio),
      error,
      "cellRadiusEdgeRatio must be finite and at least 2.");
  MOCHI_MESH_CLI_ERROR_IF(
      !(params.sharpFeatureAngle > 0.0) || params.sharpFeatureAngle > 180.0,
      error,
      "sharpFeatureAngle must be in (0, 180] degrees.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});

  try {
    SurfaceMesh const sm =
        cgal_utils::TetrahedralizationInputSurface(surfaceMesh, /*allowRepair=*/true, error);
    double const scale =
        params.relativeToMeshSize ? cgal_utils::AverageFaceBoundingBoxExtent(sm, error) : 1.0;
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});

    // Mesh_3 bounds circumradii, not edge lengths. Convert the target edge length using the
    // circumradius of the equilateral triangle (L/sqrt(3)) and regular tetrahedron (L*sqrt(6)/4).
    double const edgeSize = params.edgeSize * scale;
    double const facetSize = edgeSize / std::numbers::sqrt3;
    double const facetDistance =
        params.facetDistanceBound > 0.0 ? params.facetDistanceBound * scale : edgeSize / 5.0;
    double const cellSize = edgeSize * (std::sqrt(6.0) / 4.0);
    MOCHI_MESH_CLI_ERROR_IF(
        !(edgeSize > 0.0) || !std::isfinite(edgeSize) || !(facetSize > 0.0) ||
            !std::isfinite(facetSize) || !(facetDistance > 0.0) || !std::isfinite(facetDistance) ||
            !(cellSize > 0.0) || !std::isfinite(cellSize),
        error,
        "Scaled tetrahedralization lengths must be positive and finite.");
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});

    // A fixed seed for the domain's initial point sampling keeps the output deterministic.
    CGAL::Random rng(0);
    Domain domain(sm, &rng);
    if (params.detectFeatures) {
      domain.detect_features(params.sharpFeatureAngle);
    }

    namespace cp = CGAL::parameters;
    Criteria const criteria(
        cp::edge_size(edgeSize)
            .facet_angle(params.facetAngleBound)
            .facet_size(facetSize)
            .facet_distance(facetDistance)
            .cell_radius_edge_ratio(params.cellRadiusEdgeRatio)
            .cell_size(cellSize));

    // Unset perturb/exude time limits default to the refinement's wall-clock time, which makes the
    // output depend on machine load; 0 disables the limit.
    auto options = cp::manifold();
    auto perturbed = params.perturb ? options.perturb(cp::time_limit(0.0)) : options.no_perturb();
    auto exuded = params.exude ? perturbed.exude(cp::time_limit(0.0)) : perturbed.no_exude();
    C3t3 const c3t3 = CGAL::make_mesh_3<C3t3>(
        domain, criteria, params.odt ? exuded.odt(cp::default_values()) : exuded.no_odt());
    MOCHI_MESH_CLI_ERROR_IF(
        c3t3.number_of_cells_in_complex() == 0,
        error,
        "Tetrahedralization produced no tetrahedra. Check that edgeSize is small enough for the "
        "input.");
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});
    return ExtractTetrahedra(c3t3);
  } catch (std::exception const& e) {
    MOCHI_MESH_CLI_ERROR_SET(
        error, (std::string("CGAL tetrahedralization failed: ") + e.what()).c_str());
  } catch (...) {
    MOCHI_MESH_CLI_ERROR_SET(error, "CGAL tetrahedralization failed with an unknown exception.");
  }
  return {};
}
