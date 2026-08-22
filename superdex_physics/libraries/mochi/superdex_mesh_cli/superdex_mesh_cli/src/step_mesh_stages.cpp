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

// Helper-side implementation of the Model Editor's mesh-level CAD processing stages: CleanupMesh,
// CloseMesh, and DecimateMesh. (EdgeSwapMesh -- Stage 5 -- lives in mesh_edge_swap.cpp.) These are
// clean reimplementations of cad_mesher's CGAL post-processing, operating on neutral
// MeshData. CGAL is confined to this
// helper. No OpenCascade is used here, so these compile on every platform (unlike the STEP
// tessellation stages, which are OCCT-gated).

#include "cgal_mesh_utils.h"
#include "mesh_cli_geometry.h"

#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_length_cost.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_length_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Midpoint_placement.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/boost/graph/Euler_operations.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/boost/graph/iterator.h>
#include <CGAL/convex_hull_3.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using namespace mochi::mesh::cli;
using mochi::mesh::cli::cgal_utils::CgalSurfaceMesh;
using mochi::mesh::cli::cgal_utils::K;

using Point3 = K::Point_3;
using Vector3 = K::Vector_3;

// Spatial-hash welder that merges near-coincident points within an epsilon. Used to weld the
// unwelded per-face triangle soup that Stage 1 produces, and to re-weld during overlap removal.
class Welder {
 public:
  explicit Welder(double epsilon) : _epsilon(epsilon), _invEpsilon(1.0 / epsilon) {}

  std::size_t FindOrInsert(double x, double y, double z) {
    auto const cx = static_cast<long long>(std::floor(x * _invEpsilon));
    auto const cy = static_cast<long long>(std::floor(y * _invEpsilon));
    auto const cz = static_cast<long long>(std::floor(z * _invEpsilon));
    double const sqEps = _epsilon * _epsilon;

    for (long long dx = -1; dx <= 1; ++dx) {
      for (long long dy = -1; dy <= 1; ++dy) {
        for (long long dz = -1; dz <= 1; ++dz) {
          auto const it = _grid.find(GridKey{cx + dx, cy + dy, cz + dz});
          if (it != _grid.end()) {
            for (std::size_t const idx : it->second) {
              Point3 const& p = _points[idx];
              double const ddx = p.x() - x;
              double const ddy = p.y() - y;
              double const ddz = p.z() - z;
              if (ddx * ddx + ddy * ddy + ddz * ddz < sqEps) {
                return idx;
              }
            }
          }
        }
      }
    }
    std::size_t const newIdx = _points.size();
    _points.emplace_back(x, y, z);
    _grid[GridKey{cx, cy, cz}].push_back(newIdx);
    return newIdx;
  }

  std::vector<Point3> const& Points() const {
    return _points;
  }

 private:
  struct GridKey {
    long long ix{}, iy{}, iz{};
    bool operator==(GridKey const& o) const {
      return ix == o.ix && iy == o.iy && iz == o.iz;
    }
  };
  struct GridKeyHash {
    std::size_t operator()(GridKey const& k) const {
      std::size_t h = 14695981039346656037ULL;
      h ^= static_cast<std::size_t>(k.ix);
      h *= 1099511628211ULL;
      h ^= static_cast<std::size_t>(k.iy);
      h *= 1099511628211ULL;
      h ^= static_cast<std::size_t>(k.iz);
      h *= 1099511628211ULL;
      return h;
    }
  };

  double _epsilon;
  double _invEpsilon;
  std::vector<Point3> _points;
  std::unordered_map<GridKey, std::vector<std::size_t>, GridKeyHash> _grid;
};

// When two parts share a flush mating surface, each contributes its own tessellation of it; after
// welding these become pairs of triangles with the same indices but opposite winding. Removing both
// triangles in each pair eliminates the internal double layer. Returns the number removed.
std::size_t RemoveInternalFaces(std::vector<std::vector<std::size_t>>& triangles) {
  struct TriKey {
    std::size_t v0{}, v1{}, v2{};
    bool operator==(TriKey const& o) const {
      return v0 == o.v0 && v1 == o.v1 && v2 == o.v2;
    }
  };
  struct TriKeyHash {
    std::size_t operator()(TriKey const& k) const {
      std::size_t h = 14695981039346656037ULL;
      h ^= k.v0;
      h *= 1099511628211ULL;
      h ^= k.v1;
      h *= 1099511628211ULL;
      h ^= k.v2;
      h *= 1099511628211ULL;
      return h;
    }
  };
  struct TriInfo {
    TriKey key{};
    bool forward = false;
  };

  std::vector<TriInfo> info(triangles.size());
  std::unordered_map<TriKey, std::pair<std::size_t, std::size_t>, TriKeyHash> counts;

  for (std::size_t i = 0; i < triangles.size(); ++i) {
    auto const& tri = triangles[i];
    if (tri.size() != 3) {
      continue;
    }
    std::size_t minIdx = 0;
    if (tri[1] < tri[minIdx]) {
      minIdx = 1;
    }
    if (tri[2] < tri[minIdx]) {
      minIdx = 2;
    }
    std::size_t const a = tri[minIdx];
    std::size_t const b = tri[(minIdx + 1) % 3];
    std::size_t const c = tri[(minIdx + 2) % 3];

    TriKey const key{a, std::min(b, c), std::max(b, c)};
    bool const forward = (b < c);
    info[i] = {key, forward};
    auto& cnt = counts[key];
    if (forward) {
      ++cnt.first;
    } else {
      ++cnt.second;
    }
  }

  std::unordered_map<TriKey, std::pair<std::size_t, std::size_t>, TriKeyHash> toRemove;
  for (auto const& [key, cnt] : counts) {
    std::size_t const pairs = std::min(cnt.first, cnt.second);
    if (pairs > 0) {
      toRemove[key] = {pairs, pairs};
    }
  }
  if (toRemove.empty()) {
    return 0;
  }

  std::vector<std::vector<std::size_t>> filtered;
  filtered.reserve(triangles.size());
  std::size_t removed = 0;
  for (std::size_t i = 0; i < triangles.size(); ++i) {
    auto const& [key, forward] = info[i];
    auto it = toRemove.find(key);
    if (it != toRemove.end()) {
      auto& [fwdRemain, revRemain] = it->second;
      if (forward && fwdRemain > 0) {
        --fwdRemain;
        ++removed;
        continue;
      }
      if (!forward && revRemain > 0) {
        --revRemain;
        ++removed;
        continue;
      }
    }
    filtered.push_back(std::move(triangles[i]));
  }
  triangles = std::move(filtered);
  return removed;
}

// Drops triangles with a duplicate vertex index or near-zero 3D area.
void RemoveDegenerateTriangles(
    std::vector<Point3> const& points,
    std::vector<std::vector<std::size_t>>& triangles) {
  triangles.erase(
      std::remove_if(
          triangles.begin(),
          triangles.end(),
          [&points](std::vector<std::size_t> const& tri) {
            if (tri.size() != 3) {
              return false;
            }
            if (tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2]) {
              return true;
            }
            Point3 const& p0 = points[tri[0]];
            Point3 const& p1 = points[tri[1]];
            Point3 const& p2 = points[tri[2]];
            double const ax = p1.x() - p0.x(), ay = p1.y() - p0.y(), az = p1.z() - p0.z();
            double const bx = p2.x() - p0.x(), by = p2.y() - p0.y(), bz = p2.z() - p0.z();
            double const cx = ay * bz - az * by;
            double const cy = az * bx - ax * bz;
            double const cz = ax * by - ay * bx;
            return (cx * cx + cy * cy + cz * cz) < 1e-20;
          }),
      triangles.end());
}

// Builds a clean Surface_mesh from a welded polygon soup: repair, orient, build, stitch coincident
// borders, fix non-manifold vertices, and ensure outward orientation.
CgalSurfaceMesh BuildSurfaceMeshFromSoup(
    std::vector<Point3> points,
    std::vector<std::vector<std::size_t>> triangles) {
  namespace PMP = CGAL::Polygon_mesh_processing;
  CgalSurfaceMesh mesh;
  PMP::repair_polygon_soup(points, triangles);
  PMP::orient_polygon_soup(points, triangles);
  PMP::polygon_soup_to_polygon_mesh(points, triangles, mesh);
  PMP::stitch_borders(mesh);
  PMP::duplicate_non_manifold_vertices(mesh);
  if (CGAL::is_closed(mesh) && !PMP::is_outward_oriented(mesh)) {
    PMP::reverse_face_orientations(mesh);
  }
  mesh.collect_garbage();
  return mesh;
}

// Re-welds the mesh's triangles and removes degenerate / internal-double faces, rebuilding the mesh
// in place. Mirrors cad_mesher's removeSelfOverlaps.
void RemoveSelfOverlaps(CgalSurfaceMesh& mesh) {
  constexpr double kMergeEps = 1e-6;
  Welder welder(kMergeEps);
  std::vector<std::vector<std::size_t>> triangles;
  triangles.reserve(mesh.number_of_faces());

  for (auto const f : mesh.faces()) {
    auto h = mesh.halfedge(f);
    Point3 const& p0 = mesh.point(mesh.target(h));
    h = mesh.next(h);
    Point3 const& p1 = mesh.point(mesh.target(h));
    h = mesh.next(h);
    Point3 const& p2 = mesh.point(mesh.target(h));
    std::size_t const i0 = welder.FindOrInsert(p0.x(), p0.y(), p0.z());
    std::size_t const i1 = welder.FindOrInsert(p1.x(), p1.y(), p1.z());
    std::size_t const i2 = welder.FindOrInsert(p2.x(), p2.y(), p2.z());
    if (i0 == i1 || i1 == i2 || i0 == i2) {
      continue;
    }
    triangles.push_back({i0, i1, i2});
  }

  std::size_t const internalRemoved = RemoveInternalFaces(triangles);
  if (internalRemoved == 0) {
    return;
  }
  mesh = BuildSurfaceMeshFromSoup(welder.Points(), std::move(triangles));
}

// Patches small/medium boundary loops. Very large loops (likely whole missing regions) are skipped.
void FillHoles(CgalSurfaceMesh& mesh) {
  namespace PMP = CGAL::Polygon_mesh_processing;
  constexpr int kMaxHoleEdges = 200;

  std::unordered_set<CgalSurfaceMesh::Halfedge_index, CGAL::Handle_hash_function> visited;
  std::vector<CgalSurfaceMesh::Halfedge_index> holeBorders;
  for (auto const he : mesh.halfedges()) {
    if (mesh.is_border(he) && visited.find(he) == visited.end()) {
      holeBorders.push_back(he);
      auto cur = he;
      do {
        visited.insert(cur);
        cur = mesh.next(cur);
      } while (cur != he);
    }
  }

  for (auto const borderHe : holeBorders) {
    int edgeCount = 0;
    auto cur = borderHe;
    do {
      ++edgeCount;
      cur = mesh.next(cur);
    } while (cur != borderHe);
    if (edgeCount > kMaxHoleEdges) {
      continue;
    }
    std::vector<CgalSurfaceMesh::Face_index> patchFacets;
    std::vector<CgalSurfaceMesh::Vertex_index> patchVertices;
    PMP::triangulate_and_refine_hole(
        mesh,
        borderHe,
        CGAL::parameters::face_output_iterator(std::back_inserter(patchFacets))
            .vertex_output_iterator(std::back_inserter(patchVertices))
            .density_control_factor(1.0));
  }
}

// Computes the bounding-box diagonal of a surface mesh.
double BboxDiagonal(CgalSurfaceMesh const& mesh) {
  double minX = 1e300, minY = 1e300, minZ = 1e300;
  double maxX = -1e300, maxY = -1e300, maxZ = -1e300;
  for (auto const v : mesh.vertices()) {
    Point3 const& p = mesh.point(v);
    minX = std::min(minX, p.x());
    minY = std::min(minY, p.y());
    minZ = std::min(minZ, p.z());
    maxX = std::max(maxX, p.x());
    maxY = std::max(maxY, p.y());
    maxZ = std::max(maxZ, p.z());
  }
  double const dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Estimates a target edge length for shrink-wrap remeshing from the mesh's bounding-box diagonal.
double EstimateTargetEdgeLength(CgalSurfaceMesh const& mesh) {
  constexpr double kFraction = 0.02;
  constexpr double kFloor = 1e-6;
  return std::max(BboxDiagonal(mesh) * kFraction, kFloor);
}

// Replaces the mesh with the remeshed convex hull of its vertices.
bool ConvexHullClosure(CgalSurfaceMesh& mesh, double targetEdgeLen) {
  namespace PMP = CGAL::Polygon_mesh_processing;
  std::vector<Point3> points;
  points.reserve(mesh.number_of_vertices());
  for (auto const v : mesh.vertices()) {
    points.push_back(mesh.point(v));
  }
  if (points.size() < 4) {
    return false;
  }
  CgalSurfaceMesh hull;
  CGAL::convex_hull_3(points.begin(), points.end(), hull);
  PMP::isotropic_remeshing(
      faces(hull), targetEdgeLen, hull, CGAL::parameters::number_of_iterations(3));
  mesh = std::move(hull);
  if (CGAL::is_closed(mesh) && !PMP::is_outward_oriented(mesh)) {
    PMP::reverse_face_orientations(mesh);
  }
  return true;
}

// Encloses the surface with a convex hull that is deflated toward the original surface. Replaces
// the mesh wholesale. Mirrors cad_mesher's shrinkWrap.
bool ShrinkWrap(CgalSurfaceMesh& mesh, double tightness, double targetEdgeLen, bool snapToSurface) {
  namespace PMP = CGAL::Polygon_mesh_processing;
  using Primitive = CGAL::AABB_face_graph_triangle_primitive<CgalSurfaceMesh>;
  using AabbTraits = CGAL::AABB_traits_3<K, Primitive>;
  using Tree = CGAL::AABB_tree<AabbTraits>;

  tightness = std::clamp(tightness, 0.0, 1.0);

  std::vector<Point3> allPoints;
  allPoints.reserve(mesh.number_of_vertices());
  for (auto const v : mesh.vertices()) {
    allPoints.push_back(mesh.point(v));
  }
  if (allPoints.size() < 4) {
    return false;
  }

  CgalSurfaceMesh hull;
  CGAL::convex_hull_3(allPoints.begin(), allPoints.end(), hull);
  PMP::isotropic_remeshing(
      faces(hull), targetEdgeLen, hull, CGAL::parameters::number_of_iterations(3));

  CgalSurfaceMesh originalMesh = mesh;
  Tree tree(faces(originalMesh).first, faces(originalMesh).second, originalMesh);
  tree.accelerate_distance_queries();

  CgalSurfaceMesh cleanHull = hull;

  // Pass 1: deflate the hull toward the original surface, with Laplacian smoothing.
  int const pass1Iterations = std::max(1, static_cast<int>(tightness * 50));
  double const pass1Proj = 0.45 * tightness;
  double const pass1Smooth = 0.2;
  for (int iter = 0; iter < pass1Iterations; ++iter) {
    for (auto const v : hull.vertices()) {
      Point3 const pos = hull.point(v);
      Point3 const nearest = tree.closest_point(pos);
      hull.point(v) = Point3(
          pos.x() + (nearest.x() - pos.x()) * pass1Proj,
          pos.y() + (nearest.y() - pos.y()) * pass1Proj,
          pos.z() + (nearest.z() - pos.z()) * pass1Proj);
    }
    for (auto const v : hull.vertices()) {
      if (hull.is_border(v)) {
        continue;
      }
      double sx = 0, sy = 0, sz = 0;
      int count = 0;
      for (auto const he : CGAL::halfedges_around_target(hull.halfedge(v), hull)) {
        Point3 const& np = hull.point(hull.source(he));
        sx += np.x();
        sy += np.y();
        sz += np.z();
        ++count;
      }
      if (count > 0) {
        Point3 const& pos = hull.point(v);
        hull.point(v) = Point3(
            pos.x() * (1.0 - pass1Smooth) + (sx / count) * pass1Smooth,
            pos.y() * (1.0 - pass1Smooth) + (sy / count) * pass1Smooth,
            pos.z() * (1.0 - pass1Smooth) + (sz / count) * pass1Smooth);
      }
    }
  }

  // Pass 2: project the clean hull toward the deflated result, capping per-vertex displacement.
  CgalSurfaceMesh deflatedMesh = std::move(hull);
  Tree targetTree(faces(deflatedMesh).first, faces(deflatedMesh).second, deflatedMesh);
  targetTree.accelerate_distance_queries();

  CgalSurfaceMesh result = std::move(cleanHull);
  std::vector<Point3> originalPos;
  originalPos.reserve(result.number_of_vertices());
  for (auto const v : result.vertices()) {
    originalPos.push_back(result.point(v));
  }

  double const maxDisplacement = targetEdgeLen * 100.0;
  double const maxDisplacementSq = maxDisplacement * maxDisplacement;
  int const pass2Iterations = 100;
  double const pass2Proj = 1.0;
  double const pass2Smooth = 0.1;
  for (int iter = 0; iter < pass2Iterations; ++iter) {
    int vidx = 0;
    for (auto const v : result.vertices()) {
      if (vidx >= static_cast<int>(originalPos.size())) {
        break;
      }
      Point3 const pos = result.point(v);
      Point3 const nearest = targetTree.closest_point(pos);
      Point3 newPos(
          pos.x() + (nearest.x() - pos.x()) * pass2Proj,
          pos.y() + (nearest.y() - pos.y()) * pass2Proj,
          pos.z() + (nearest.z() - pos.z()) * pass2Proj);
      double const dispX = newPos.x() - originalPos[vidx].x();
      double const dispY = newPos.y() - originalPos[vidx].y();
      double const dispZ = newPos.z() - originalPos[vidx].z();
      double const totalDispSq = dispX * dispX + dispY * dispY + dispZ * dispZ;
      if (totalDispSq > maxDisplacementSq) {
        double const scale = maxDisplacement / std::sqrt(totalDispSq);
        newPos = Point3(
            originalPos[vidx].x() + dispX * scale,
            originalPos[vidx].y() + dispY * scale,
            originalPos[vidx].z() + dispZ * scale);
      }
      result.point(v) = newPos;
      ++vidx;
    }
    for (auto const v : result.vertices()) {
      if (result.is_border(v)) {
        continue;
      }
      double sx = 0, sy = 0, sz = 0;
      int count = 0;
      for (auto const he : CGAL::halfedges_around_target(result.halfedge(v), result)) {
        Point3 const& np = result.point(result.source(he));
        sx += np.x();
        sy += np.y();
        sz += np.z();
        ++count;
      }
      if (count > 0) {
        Point3 const& pos = result.point(v);
        result.point(v) = Point3(
            pos.x() * (1.0 - pass2Smooth) + (sx / count) * pass2Smooth,
            pos.y() * (1.0 - pass2Smooth) + (sy / count) * pass2Smooth,
            pos.z() * (1.0 - pass2Smooth) + (sz / count) * pass2Smooth);
      }
    }
  }

  PMP::isotropic_remeshing(
      faces(result), targetEdgeLen, result, CGAL::parameters::number_of_iterations(5));

  // Pass 3: optionally snap vertices onto the original surface.
  if (snapToSurface) {
    double const snapThreshold = targetEdgeLen * 2.0;
    double const snapStrength = 0.8;
    int const snapIterations = 20;
    for (int iter = 0; iter < snapIterations; ++iter) {
      for (auto const v : result.vertices()) {
        Point3 const pos = result.point(v);
        Point3 const nearest = tree.closest_point(pos);
        double const dx = nearest.x() - pos.x();
        double const dy = nearest.y() - pos.y();
        double const dz = nearest.z() - pos.z();
        double const dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < snapThreshold) {
          double const blend = snapStrength * (1.0 - dist / snapThreshold);
          result.point(v) =
              Point3(pos.x() + dx * blend, pos.y() + dy * blend, pos.z() + dz * blend);
        }
      }
    }
  }

  mesh = std::move(result);
  if (CGAL::is_closed(mesh) && !PMP::is_outward_oriented(mesh)) {
    PMP::reverse_face_orientations(mesh);
  }
  PMP::isotropic_remeshing(
      faces(mesh), targetEdgeLen, mesh, CGAL::parameters::number_of_iterations(5));
  if (CGAL::is_closed(mesh) && !PMP::is_outward_oriented(mesh)) {
    PMP::reverse_face_orientations(mesh);
  }
  return true;
}

} // namespace

MeshData mochi::mesh::cli::CleanupMesh(MeshData const& mesh, CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});
  MOCHI_MESH_CLI_ERROR_IF(GetNumElements(mesh) == 0, error, "CleanupMesh: input mesh is empty.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});

  // Weld the unwelded per-face soup, drop degenerate and internal-double faces, and build a clean
  // surface mesh; then remove any remaining self-overlaps.
  constexpr double kMergeEps = 1e-6;
  Welder welder(kMergeEps);
  std::vector<std::vector<std::size_t>> triangles;
  int const numTris = GetNumElements(mesh);
  triangles.reserve(numTris);
  for (int i = 0; i < numTris; ++i) {
    int const v0 = mesh.connectivity[3 * i + 0];
    int const v1 = mesh.connectivity[3 * i + 1];
    int const v2 = mesh.connectivity[3 * i + 2];
    std::size_t const i0 = welder.FindOrInsert(
        static_cast<double>(mesh.coordinates[3 * v0 + 0]),
        static_cast<double>(mesh.coordinates[3 * v0 + 1]),
        static_cast<double>(mesh.coordinates[3 * v0 + 2]));
    std::size_t const i1 = welder.FindOrInsert(
        static_cast<double>(mesh.coordinates[3 * v1 + 0]),
        static_cast<double>(mesh.coordinates[3 * v1 + 1]),
        static_cast<double>(mesh.coordinates[3 * v1 + 2]));
    std::size_t const i2 = welder.FindOrInsert(
        static_cast<double>(mesh.coordinates[3 * v2 + 0]),
        static_cast<double>(mesh.coordinates[3 * v2 + 1]),
        static_cast<double>(mesh.coordinates[3 * v2 + 2]));
    triangles.push_back({i0, i1, i2});
  }

  std::vector<Point3> points = welder.Points();
  RemoveDegenerateTriangles(points, triangles);
  RemoveInternalFaces(triangles);

  try {
    CgalSurfaceMesh result = BuildSurfaceMeshFromSoup(std::move(points), std::move(triangles));
    RemoveSelfOverlaps(result);
    cgal_utils::RepairMesh(result);
    MOCHI_MESH_CLI_ERROR_IF(
        result.number_of_faces() == 0, error, "CleanupMesh: cleanup produced an empty mesh.");
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});
    return cgal_utils::SurfaceMeshToMeshData(result);
  } catch (std::exception const&) {
    MOCHI_MESH_CLI_ERROR_SET(error, "CleanupMesh: CGAL exception during cleanup.");
    return {};
  } catch (...) {
    MOCHI_MESH_CLI_ERROR_SET(error, "CleanupMesh: unknown exception during cleanup.");
    return {};
  }
}

MeshData mochi::mesh::cli::CloseMesh(
    MeshData const& mesh,
    MeshClosureParams const& params,
    CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});
  MOCHI_MESH_CLI_ERROR_IF(GetNumElements(mesh) == 0, error, "CloseMesh: input mesh is empty.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});

  try {
    CgalSurfaceMesh sm = cgal_utils::MeshDataToSurfaceMesh(mesh, error);
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});

    switch (params.mode) {
      case MeshClosureMode::None:
        break;
      case MeshClosureMode::FillHoles:
        FillHoles(sm);
        break;
      case MeshClosureMode::ShrinkWrap: {
        double const targetEdgeLen = (params.shrinkWrapTargetEdgeLength > 0.0)
            ? params.shrinkWrapTargetEdgeLength
            : std::max(BboxDiagonal(sm) * params.shrinkWrapTargetEdgeLengthFraction, 1e-6);
        if (!ShrinkWrap(sm, params.shrinkWrapTightness, targetEdgeLen, params.shrinkWrapSnap)) {
          MOCHI_MESH_CLI_ERROR_SET(error, "CloseMesh: shrink-wrap failed (too few vertices).");
          return {};
        }
        break;
      }
      case MeshClosureMode::ConvexHull:
        if (!ConvexHullClosure(sm, EstimateTargetEdgeLength(sm))) {
          MOCHI_MESH_CLI_ERROR_SET(error, "CloseMesh: convex hull failed (too few vertices).");
          return {};
        }
        break;
    }

    MOCHI_MESH_CLI_ERROR_IF(
        sm.number_of_faces() == 0, error, "CloseMesh: closure produced an empty mesh.");
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});
    return cgal_utils::SurfaceMeshToMeshData(sm);
  } catch (std::exception const&) {
    MOCHI_MESH_CLI_ERROR_SET(error, "CloseMesh: CGAL exception during closure.");
    return {};
  } catch (...) {
    MOCHI_MESH_CLI_ERROR_SET(error, "CloseMesh: unknown exception during closure.");
    return {};
  }
}

MeshData mochi::mesh::cli::DecimateMesh(
    MeshData const& mesh,
    MeshDecimateParams const& params,
    CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});
  MOCHI_MESH_CLI_ERROR_IF(GetNumElements(mesh) == 0, error, "DecimateMesh: input mesh is empty.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});

  // A non-positive collapse distance disables decimation; return the input unchanged.
  if (!(params.collapseDistance > 0.0)) {
    return mesh;
  }

  namespace SMS = CGAL::Surface_mesh_simplification;
  try {
    CgalSurfaceMesh sm = cgal_utils::MeshDataToSurfaceMesh(mesh, error);
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});
    SMS::Edge_length_stop_predicate<double> const stop(params.collapseDistance);
    SMS::Edge_length_cost<CgalSurfaceMesh> const cost;
    SMS::Midpoint_placement<CgalSurfaceMesh> const placement;
    SMS::edge_collapse(sm, stop, CGAL::parameters::get_cost(cost).get_placement(placement));
    sm.collect_garbage();
    MOCHI_MESH_CLI_ERROR_IF(
        sm.number_of_faces() == 0, error, "DecimateMesh: decimation produced an empty mesh.");
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});
    return cgal_utils::SurfaceMeshToMeshData(sm);
  } catch (std::exception const&) {
    MOCHI_MESH_CLI_ERROR_SET(error, "DecimateMesh: CGAL exception during decimation.");
    return {};
  } catch (...) {
    MOCHI_MESH_CLI_ERROR_SET(error, "DecimateMesh: unknown exception during decimation.");
    return {};
  }
}
