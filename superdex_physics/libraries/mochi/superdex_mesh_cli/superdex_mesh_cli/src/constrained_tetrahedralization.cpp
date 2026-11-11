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

#include <CGAL/Conforming_constrained_Delaunay_triangulation_3.h>
#include <CGAL/Conforming_constrained_Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Conforming_constrained_Delaunay_triangulation_vertex_base_3.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Tetrahedral_remeshing/Remeshing_cell_base_3.h>
#include <CGAL/Tetrahedral_remeshing/Remeshing_vertex_base_3.h>
#include <CGAL/Triangulation_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/centroid.h>
#include <CGAL/make_conforming_constrained_Delaunay_triangulation_3.h>
#include <CGAL/tetrahedral_remeshing.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mochi::mesh::cli;

namespace {

using K = cgal_utils::K;
using SurfaceMesh = cgal_utils::CgalSurfaceMesh;
// The remeshing bases are innermost: Tetrahedral_remeshing requires its vertex base to build on a
// simplicial mesh vertex base, while the constrained triangulation's bases accept any triangulation
// base.
using Vb = CGAL::Conforming_constrained_Delaunay_triangulation_vertex_base_3<
    K,
    CGAL::Tetrahedral_remeshing::Remeshing_vertex_base_3<K>>;
using Cb = CGAL::Conforming_constrained_Delaunay_triangulation_cell_base_3<
    K,
    CGAL::Tetrahedral_remeshing::Remeshing_cell_base_3<K>>;
using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb>;
using Tr = CGAL::Triangulation_3<K, Tds>;
using Ccdt = CGAL::Conforming_constrained_Delaunay_triangulation_3<K, Tr>;

using Coordinates = std::array<double, 3>;

// Maps each input vertex's coordinates to its index, rejecting unreferenced and coincident
// vertices: both would break the guarantee that the input vertices are the first output nodes.
// Connectivity indices must already be bounds-checked, as TetrahedralizationInputSurface does.
std::map<Coordinates, int32_t> IndexInputVertices(MeshData const& surfaceMesh, CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});
  int const numVertices = GetNumNodes(surfaceMesh);
  std::vector<bool> isReferenced(numVertices, false);
  for (int32_t const vertex : surfaceMesh.connectivity) {
    isReferenced[vertex] = true;
  }
  MOCHI_MESH_CLI_ERROR_IF(
      std::find(isReferenced.begin(), isReferenced.end(), false) != isReferenced.end(),
      error,
      "Input surface has vertices that no face uses. Remove them first.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});

  std::map<Coordinates, int32_t> indices;
  for (int32_t vertex = 0; vertex < numVertices; ++vertex) {
    double const* const p = &surfaceMesh.coordinates[3 * static_cast<std::size_t>(vertex)];
    bool const inserted = indices.try_emplace(Coordinates{p[0], p[1], p[2]}, vertex).second;
    MOCHI_MESH_CLI_ERROR_IF(
        !inserted, error, "Input surface has coincident vertices. Weld them first.");
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});
  }
  return indices;
}

// Numbers the input vertices by their input index and every other vertex after them, in order of
// first use so the output does not depend on allocation addresses. Remeshing never moves boundary
// vertices, so each input vertex is found at its exact input coordinates.
MeshData ExtractTetrahedra(
    Tr const& tr,
    MeshData const& surfaceMesh,
    std::map<Coordinates, int32_t> const& inputIndices,
    CliError& error) {
  int32_t const numInputVertices = GetNumNodes(surfaceMesh);
  MeshData result;
  result.nodesPerElement = 4;
  result.coordinates.reserve(3 * tr.number_of_vertices());
  result.coordinates.insert(
      result.coordinates.end(), surfaceMesh.coordinates.begin(), surfaceMesh.coordinates.end());
  result.connectivity.reserve(4 * tr.number_of_finite_cells());
  std::vector<bool> isInputVertexUsed(numInputVertices, false);
  std::unordered_map<Tr::Vertex const*, int32_t> nodeIndex;
  nodeIndex.reserve(tr.number_of_vertices());
  for (auto const cell : tr.finite_cell_handles()) {
    if (cell->subdomain_index() == 0) {
      continue;
    }
    for (int i = 0; i < 4; ++i) {
      Tr::Vertex const* const vertex = &*cell->vertex(i);
      auto [it, inserted] = nodeIndex.try_emplace(vertex, -1);
      if (inserted) {
        K::Point_3 const& point = vertex->point();
        Coordinates const coordinates{point.x(), point.y(), point.z()};
        auto const input = inputIndices.find(coordinates);
        if (input != inputIndices.end()) {
          it->second = input->second;
          isInputVertexUsed[input->second] = true;
        } else {
          it->second = GetNumNodes(result);
          result.coordinates.insert(
              result.coordinates.end(), coordinates.begin(), coordinates.end());
        }
      }
      result.connectivity.push_back(it->second);
    }
  }
  MOCHI_MESH_CLI_ERROR_IF(
      result.connectivity.empty(), error, "Tetrahedralization produced no tetrahedra.");
  MOCHI_MESH_CLI_ERROR_IF(
      std::find(isInputVertexUsed.begin(), isInputVertexUsed.end(), false) !=
          isInputVertexUsed.end(),
      error,
      "Internal error: tetrahedralization lost input vertices.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});
  return result;
}

} // namespace

MeshData mochi::mesh::cli::TetrahedralizeConstrained(
    MeshData const& surfaceMesh,
    ConstrainedTetrahedralizationParams const& params,
    CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});
  MOCHI_MESH_CLI_ERROR_IF(
      !(params.edgeSize > 0.0) || !std::isfinite(params.edgeSize),
      error,
      "edgeSize must be positive and finite.");
  MOCHI_MESH_CLI_ERROR_IF(params.iterations < 1, error, "iterations must be at least 1.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});

  try {
    SurfaceMesh const sm =
        cgal_utils::TetrahedralizationInputSurface(surfaceMesh, /*allowRepair=*/false, error);
    std::map<Coordinates, int32_t> const inputIndices = IndexInputVertices(surfaceMesh, error);
    double const scale =
        params.relativeToMeshSize ? cgal_utils::AverageFaceBoundingBoxExtent(sm, error) : 1.0;
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});
    double const edgeSize = params.edgeSize * scale;
    MOCHI_MESH_CLI_ERROR_IF(
        !(edgeSize > 0.0) || !std::isfinite(edgeSize),
        error,
        "Scaled edgeSize must be positive and finite.");
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});

    Ccdt ccdt = CGAL::make_conforming_constrained_Delaunay_triangulation_3<Ccdt>(sm);
    // Every vertex beyond the input's was inserted to recover a constrained edge or face.
    auto const numBoundaryVerticesAdded =
        ccdt.triangulation().number_of_vertices() - sm.number_of_vertices();
    if (numBoundaryVerticesAdded > 0) {
      std::string const message = std::to_string(numBoundaryVerticesAdded) +
          " vertices had to be added to the input surface for constrained Delaunay construction";
      MOCHI_MESH_CLI_ERROR_IF(
          params.strict,
          error,
          (message + ". Disable strict mode to allow it, or remesh the input surface first.")
              .c_str());
      MOCHI_MESH_CLI_ERROR_RETURN(error, {});
      MOCHI_MESH_CLI_LOG_WARNING("%s.", message.c_str());
    }

    Tr tr = CGAL::Tetrahedral_remeshing::get_remeshing_triangulation(std::move(ccdt));
    MOCHI_MESH_CLI_ERROR_IF(
        tr.dimension() != 3, error, "Tetrahedralization produced no three-dimensional cells.");
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});
    // A conforming cell's interior lies in one material region, identified by its centroid.
    CGAL::Side_of_triangle_mesh<SurfaceMesh, K> const inside(sm);
    for (auto const cell : tr.all_cell_handles()) {
      bool material = false;
      if (!tr.is_infinite(cell)) {
        K::Point_3 const center = CGAL::centroid(
            cell->vertex(0)->point(),
            cell->vertex(1)->point(),
            cell->vertex(2)->point(),
            cell->vertex(3)->point());
        material = inside(center) != CGAL::ON_UNBOUNDED_SIDE;
      }
      cell->set_subdomain_index(material ? 1 : 0);
    }
    CGAL::tetrahedral_isotropic_remeshing(
        tr,
        edgeSize,
        CGAL::parameters::remesh_boundaries(false).number_of_iterations(
            static_cast<unsigned int>(params.iterations)));
    return ExtractTetrahedra(tr, surfaceMesh, inputIndices, error);
  } catch (std::exception const& e) {
    MOCHI_MESH_CLI_ERROR_SET(
        error, (std::string("CGAL constrained tetrahedralization failed: ") + e.what()).c_str());
  } catch (...) {
    MOCHI_MESH_CLI_ERROR_SET(
        error, "CGAL constrained tetrahedralization failed with an unknown exception.");
  }
  return {};
}
