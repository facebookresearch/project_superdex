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

// Internal CGAL conversion utilities for the superdex_mesh_cli.
// NOT part of the public API - depends on CGAL headers.

#include "mesh_cli_geometry.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Surface_mesh.h>

#include <map>
#include <vector>

namespace mochi::mesh::cli::cgal_utils {

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using CgalSurfaceMesh = CGAL::Surface_mesh<K::Point_3>;

/// Convert a MeshData to a CGAL polygon soup (points + triangle indices).
[[nodiscard]] inline bool MeshDataToPolygonSoup(
    MeshData const& mesh,
    std::vector<K::Point_3>& points,
    std::vector<std::vector<std::size_t>>& faces,
    CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, false);
  int const numVertices = GetNumNodes(mesh);
  int const numFaces = GetNumElements(mesh);

  points.clear();
  points.reserve(numVertices);
  for (int i = 0; i < numVertices; ++i) {
    points.emplace_back(
        static_cast<double>(mesh.coordinates[3 * i + 0]),
        static_cast<double>(mesh.coordinates[3 * i + 1]),
        static_cast<double>(mesh.coordinates[3 * i + 2]));
  }

  faces.clear();
  faces.reserve(numFaces);
  for (int i = 0; i < numFaces; ++i) {
    int const v0 = mesh.connectivity[3 * i + 0];
    int const v1 = mesh.connectivity[3 * i + 1];
    int const v2 = mesh.connectivity[3 * i + 2];
    MOCHI_MESH_CLI_ERROR_IF(
        v0 < 0 || v0 >= numVertices || v1 < 0 || v1 >= numVertices || v2 < 0 || v2 >= numVertices,
        error,
        "Connectivity index out of bounds.");
    MOCHI_MESH_CLI_ERROR_RETURN(error, false);
    faces.push_back(
        {static_cast<std::size_t>(v0), static_cast<std::size_t>(v1), static_cast<std::size_t>(v2)});
  }

  return true;
}

/// Convert a MeshData to a CGAL Surface_mesh.
/// Falls back to polygon soup reconstruction if direct face insertion fails
/// (e.g., due to non-manifold edges or inconsistent face orientations).
/// When @p needsRepairOut is non-null, sets it to true if the polygon soup
/// fallback was used, signalling that the caller should run RepairMesh.
[[nodiscard]] inline CgalSurfaceMesh
MeshDataToSurfaceMesh(MeshData const& mesh, CliError& error, bool* needsRepairOut = nullptr) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});

  if (needsRepairOut != nullptr) {
    *needsRepairOut = false;
  }

  int const numVertices = GetNumNodes(mesh);
  int const numFaces = GetNumElements(mesh);

  CgalSurfaceMesh cgalMesh;
  std::vector<CgalSurfaceMesh::Vertex_index> vertexIndices;
  vertexIndices.reserve(numVertices);

  for (int i = 0; i < numVertices; ++i) {
    K::Point_3 point(
        static_cast<double>(mesh.coordinates[3 * i + 0]),
        static_cast<double>(mesh.coordinates[3 * i + 1]),
        static_cast<double>(mesh.coordinates[3 * i + 2]));
    vertexIndices.push_back(cgalMesh.add_vertex(point));
  }

  bool needsRepair = false;
  for (int i = 0; i < numFaces; ++i) {
    int const v0 = mesh.connectivity[3 * i + 0];
    int const v1 = mesh.connectivity[3 * i + 1];
    int const v2 = mesh.connectivity[3 * i + 2];
    MOCHI_MESH_CLI_ERROR_IF(
        v0 < 0 || v0 >= numVertices || v1 < 0 || v1 >= numVertices || v2 < 0 || v2 >= numVertices,
        error,
        "Connectivity index out of bounds.");
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});
    auto const face = cgalMesh.add_face(vertexIndices[v0], vertexIndices[v1], vertexIndices[v2]);
    if (face == CgalSurfaceMesh::null_face()) {
      needsRepair = true;
      break;
    }
  }

  if (needsRepair) {
    MOCHI_MESH_CLI_LOG_WARNING(
        "Direct face insertion failed; falling back to polygon soup reconstruction "
        "(input likely has non-manifold edges or inconsistent orientations).");
    std::vector<K::Point_3> points;
    std::vector<std::vector<std::size_t>> faces;
    if (!MeshDataToPolygonSoup(mesh, points, faces, error)) {
      return {};
    }
    CGAL::Polygon_mesh_processing::orient_polygon_soup(points, faces);
    cgalMesh.clear();
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, faces, cgalMesh);
    cgalMesh.collect_garbage();
    MOCHI_MESH_CLI_ERROR_IF(
        cgalMesh.number_of_vertices() == 0,
        error,
        "Polygon soup reconstruction produced no vertices.");
    MOCHI_MESH_CLI_ERROR_IF(
        cgalMesh.number_of_faces() == 0, error, "Polygon soup reconstruction produced no faces.");
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});

    if (needsRepairOut != nullptr) {
      *needsRepairOut = true;
    }
  }

  return cgalMesh;
}

/// Best-effort mesh repair. Never fails - rolls back destructive operations
/// if they would break mesh invariants (e.g., opening a watertight mesh).
/// Defined in surface_remeshing.cpp.
void RepairMesh(CgalSurfaceMesh& sm);

/// Convert a CGAL Surface_mesh to MeshData.
[[nodiscard]] inline MeshData SurfaceMeshToMeshData(CgalSurfaceMesh const& sm) {
  MeshData result;
  result.nodesPerElement = 3;

  result.coordinates.reserve(sm.number_of_vertices() * 3);
  std::map<CgalSurfaceMesh::Vertex_index, int> vertexMap;
  int idx = 0;
  for (auto const vi : sm.vertices()) {
    vertexMap[vi] = idx++;
    K::Point_3 const& p = sm.point(vi);
    result.coordinates.push_back(static_cast<double>(p.x()));
    result.coordinates.push_back(static_cast<double>(p.y()));
    result.coordinates.push_back(static_cast<double>(p.z()));
  }

  result.connectivity.reserve(sm.number_of_faces() * 3);
  for (auto const fi : sm.faces()) {
    auto const h = sm.halfedge(fi);
    auto const h1 = sm.next(h);
    auto const h2 = sm.next(h1);
    result.connectivity.push_back(vertexMap[sm.target(h)]);
    result.connectivity.push_back(vertexMap[sm.target(h1)]);
    result.connectivity.push_back(vertexMap[sm.target(h2)]);
  }

  return result;
}

} // namespace mochi::mesh::cli::cgal_utils
