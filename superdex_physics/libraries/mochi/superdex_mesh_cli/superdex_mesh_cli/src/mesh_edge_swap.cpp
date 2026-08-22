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

// Helper-side implementation of mochi::mesh::cli::EdgeSwapMesh (Stage 5). Broken out of
// step_mesh_stages.cpp to keep that file focused. CGAL is confined to this helper; no OpenCascade
// is used here, so this compiles on every platform.
//
// Goal: for each interior edge shared by two triangles (a quad with shared-edge endpoints A,B and
// apexes C,D), flip the shared edge from A-B to C-D when the alternate diagonal follows the Stage-1
// reference surface better. This repairs edges that Stage 3-4 remeshing oriented poorly.
//
// Deviation metric: the deviation of an edge midpoint from the reference surface, measured by
// casting a ray along the triangle-pair average normal (both directions) and taking the nearest
// hit. This is robust where a nearest-point query would latch onto an unrelated feature for a large
// triangle.
//
// Criterion: swap iff err(midpoint A-B) - err(midpoint C-D) >= t * L, where L = (|A-B| + |C-D|) / 2
// is the local edge scale and t (relativeThreshold) defaults to 0.1. Being scale-relative, a tiny
// absolute gain on a large triangle is ignored. Since err(C-D) >= 0, a swap can never qualify when
// err(A-B) < t*L -- an exact early-out that skips the second (C-D) ray for already-fit edges. The
// 0.5 from L is folded into the threshold (halfThreshold) so no per-edge division is needed.
//
// Performance: one AABB tree (BVH) over the reference, built once; coplanar pairs and already-fit
// edges are pruned before any ray work; candidate evaluation is read-only and run in parallel, then
// swaps are applied serially under a touched-face guard. No swap ever happens concurrently, so two
// triangle pairs sharing a face can never both flip and corrupt the mesh.

#include "cgal_mesh_utils.h"
#include "mesh_cli_geometry.h"

#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/Handle_hash_function.h>
#include <CGAL/boost/graph/Euler_operations.h>
#include <CGAL/boost/graph/iterator.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

using namespace mochi::mesh::cli;
using mochi::mesh::cli::cgal_utils::CgalSurfaceMesh;
using mochi::mesh::cli::cgal_utils::K;

using Point3 = K::Point_3;
using Vector3 = K::Vector_3;
using Ray3 = K::Ray_3;
using EdgeIndex = CgalSurfaceMesh::Edge_index;
using FaceIndex = CgalSurfaceMesh::Face_index;
using VertexIndex = CgalSurfaceMesh::Vertex_index;

using AabbPrimitive = CGAL::AABB_face_graph_triangle_primitive<CgalSurfaceMesh>;
using AabbTraits = CGAL::AABB_traits_3<K, AabbPrimitive>;
using AabbTree = CGAL::AABB_tree<AabbTraits>;

// Normalized face-normal dot above which a triangle pair counts as flat: both diagonals then lie in
// the same plane and fit the surface equally, so there is nothing to gain by swapping.
constexpr double kCoplanarDot = 0.999;
// Below this squared length a face normal / projection direction is treated as degenerate.
constexpr double kDegenerateSq = 1e-30;

Vector3 TriangleNormal(Point3 const& p0, Point3 const& p1, Point3 const& p2) {
  return CGAL::cross_product(p1 - p0, p2 - p0);
}

double Distance(Point3 const& a, Point3 const& b) {
  double const dx = a.x() - b.x();
  double const dy = a.y() - b.y();
  double const dz = a.z() - b.z();
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

Point3 Midpoint(Point3 const& a, Point3 const& b) {
  return {(a.x() + b.x()) * 0.5, (a.y() + b.y()) * 0.5, (a.z() + b.z()) * 0.5};
}

// Distance from `origin` to the nearest reference-surface hit along +/- `dir` (a unit vector);
// negative if both rays miss. Uses first_intersected_primitive (returns just the hit face) plus a
// manual ray-plane distance, which is independent of the CGAL version's intersection-variant type.
double RayDistanceToSurface(
    AabbTree const& tree,
    CgalSurfaceMesh const& referenceMesh,
    Point3 const& origin,
    Vector3 const& dir) {
  double best = -1.0;
  for (double const sign : {1.0, -1.0}) {
    Vector3 const rayDir = dir * sign; // unit, since dir is unit
    auto const hit = tree.first_intersected_primitive(Ray3(origin, rayDir));
    if (!hit) {
      continue;
    }
    auto const face = *hit;
    auto const h = referenceMesh.halfedge(face);
    Point3 const& p0 = referenceMesh.point(referenceMesh.source(h));
    Point3 const& p1 = referenceMesh.point(referenceMesh.target(h));
    Point3 const& p2 = referenceMesh.point(referenceMesh.target(referenceMesh.next(h)));
    Vector3 const faceNormal = CGAL::cross_product(p1 - p0, p2 - p0);
    double const denom = rayDir * faceNormal;
    if (std::abs(denom) < kDegenerateSq) {
      continue; // ray parallel to the hit triangle's plane (grazing) -- ignore this direction
    }
    double const d =
        std::abs(((p0 - origin) * faceNormal) / denom); // rayDir is unit -> param == distance
    if (best < 0.0 || d < best) {
      best = d;
    }
  }
  return best;
}

// The two triangles around an interior edge, as the quad (a,b | opp0,opp1) with their face normals
// (TriangleNormal of opp1 swaps a,b so both normals share the quad's outward sense).
struct Quad {
  VertexIndex a, b, opp0, opp1;
  Vector3 n0, n1;
};

bool BuildQuad(CgalSurfaceMesh const& mesh, EdgeIndex e, Quad& q) {
  if (mesh.is_removed(e)) {
    return false;
  }
  auto const h = mesh.halfedge(e);
  auto const ho = mesh.opposite(h);
  if (mesh.is_border(h) || mesh.is_border(ho)) {
    return false;
  }
  q.a = mesh.source(h);
  q.b = mesh.target(h);
  q.opp0 = mesh.target(mesh.next(h));
  q.opp1 = mesh.target(mesh.next(ho));
  q.n0 = TriangleNormal(mesh.point(q.a), mesh.point(q.b), mesh.point(q.opp0));
  q.n1 = TriangleNormal(mesh.point(q.b), mesh.point(q.a), mesh.point(q.opp1));
  return true;
}

// Topology + non-fold safety for flipping edge `e` to the opposite diagonal. Re-checked at apply
// time because earlier swaps in the same pass can change a neighbor's degree or create the C-D
// edge.
bool FlipIsSafe(CgalSurfaceMesh const& mesh, EdgeIndex e) {
  Quad q;
  if (!BuildQuad(mesh, e, q)) {
    return false;
  }
  // A degree-3 endpoint would drop to degree 2 after the flip; that configuration always also has
  // the C-D edge present already, so the duplicate-edge check below catches it -- guard cheaply
  // too.
  if (mesh.degree(q.a) < 3 || mesh.degree(q.b) < 3) {
    return false;
  }
  // The new diagonal must not already exist, or the flip creates a non-manifold edge.
  if (mesh.halfedge(q.opp0, q.opp1) != CgalSurfaceMesh::null_halfedge()) {
    return false;
  }
  // Reject folds: both new triangles must keep the quad's average outward orientation.
  Point3 const& pA = mesh.point(q.a);
  Point3 const& pB = mesh.point(q.b);
  Point3 const& pC = mesh.point(q.opp0);
  Point3 const& pD = mesh.point(q.opp1);
  Vector3 const avgNormal = q.n0 + q.n1;
  Vector3 const newN0 = TriangleNormal(pC, pA, pD);
  Vector3 const newN1 = TriangleNormal(pD, pB, pC);
  return avgNormal * newN0 > 0.0 && avgNormal * newN1 > 0.0;
}

struct Candidate {
  EdgeIndex edge;
  double improvement = 0.0; // err(A-B) - err(C-D), positive when worth swapping
  bool swap = false;
};

// Read-only evaluation of one edge. `halfThreshold` == 0.5 * relativeThreshold, so it multiplies
// the raw (|A-B| + |C-D|) sum directly (the /2 of L is baked in).
Candidate EvaluateEdge(
    CgalSurfaceMesh const& mesh,
    AabbTree const& tree,
    CgalSurfaceMesh const& referenceMesh,
    EdgeIndex e,
    double halfThreshold) {
  Candidate result{e};
  Quad q;
  if (!BuildQuad(mesh, e, q)) {
    return result;
  }
  double const n0Sq = q.n0.squared_length();
  double const n1Sq = q.n1.squared_length();
  if (n0Sq < kDegenerateSq || n1Sq < kDegenerateSq) {
    return result; // degenerate triangle
  }
  double const dot = q.n0 * q.n1;
  if (dot > 0.0 && (dot * dot) > (kCoplanarDot * kCoplanarDot) * (n0Sq * n1Sq)) {
    return result; // coplanar: both diagonals fit the surface equally
  }
  Vector3 const avgNormal = q.n0 + q.n1;
  double const avgSq = avgNormal.squared_length();
  if (avgSq < kDegenerateSq) {
    return result; // opposite-facing pair -- no meaningful projection direction
  }
  Vector3 const dir = avgNormal / std::sqrt(avgSq);

  Point3 const& pA = mesh.point(q.a);
  Point3 const& pB = mesh.point(q.b);
  Point3 const& pC = mesh.point(q.opp0);
  Point3 const& pD = mesh.point(q.opp1);
  double const scaledThreshold = halfThreshold * (Distance(pA, pB) + Distance(pC, pD));

  double const errAB = RayDistanceToSurface(tree, referenceMesh, Midpoint(pA, pB), dir);
  if (errAB < 0.0) {
    return result; // both rays missed -- cannot assess this edge
  }
  if (errAB < scaledThreshold) {
    return result; // already well fit; err(C-D) >= 0 means a swap can never qualify (exact
                   // early-out)
  }
  double const errCD = RayDistanceToSurface(tree, referenceMesh, Midpoint(pC, pD), dir);
  if (errCD < 0.0) {
    return result; // alternate ray missed -- treat as no improvement
  }
  if ((errAB - errCD) < scaledThreshold) {
    return result;
  }
  if (!FlipIsSafe(mesh, e)) {
    return result;
  }
  result.improvement = errAB - errCD;
  result.swap = true;
  return result;
}

// Runs fn(i) for i in [0,n) across hardware threads. fn must only write to per-index storage so the
// invocations stay data-race free.
template <typename Fn>
void ParallelFor(std::size_t n, Fn const& fn) {
  if (n == 0) {
    return;
  }
  unsigned const hardware = std::max(1u, std::thread::hardware_concurrency());
  std::size_t const numThreads = std::min<std::size_t>(hardware, n);
  if (numThreads <= 1) {
    for (std::size_t i = 0; i < n; ++i) {
      fn(i);
    }
    return;
  }
  std::size_t const chunk = (n + numThreads - 1) / numThreads;
  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  for (std::size_t t = 0; t < numThreads; ++t) {
    std::size_t const begin = t * chunk;
    std::size_t const end = std::min(n, begin + chunk);
    if (begin >= end) {
      break;
    }
    threads.emplace_back([&fn, begin, end]() {
      for (std::size_t i = begin; i < end; ++i) {
        fn(i);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

// Iteratively flips edges toward the reference surface until a pass makes no swaps or maxPasses is
// reached. Each pass evaluates all interior edges in parallel (read-only), then applies the winning
// swaps serially -- highest improvement first, each triangle touched at most once per pass, every
// flip re-validated against the live mesh. The serial apply is what keeps the mesh manifold: no two
// swaps sharing a triangle (or that would duplicate an edge) are ever applied together.
void EdgeSwapTowardSurface(
    CgalSurfaceMesh& mesh,
    AabbTree const& tree,
    CgalSurfaceMesh const& referenceMesh,
    double relativeThreshold,
    int maxPasses) {
  double const halfThreshold = 0.5 * relativeThreshold;

  for (int pass = 0; pass < maxPasses; ++pass) {
    std::vector<EdgeIndex> edges;
    edges.reserve(mesh.number_of_edges());
    for (auto const e : mesh.edges()) {
      edges.push_back(e);
    }

    std::vector<Candidate> candidates(edges.size());
    ParallelFor(edges.size(), [&](std::size_t i) {
      candidates[i] = EvaluateEdge(mesh, tree, referenceMesh, edges[i], halfThreshold);
    });

    std::vector<Candidate> winners;
    for (auto const& candidate : candidates) {
      if (candidate.swap) {
        winners.push_back(candidate);
      }
    }
    if (winners.empty()) {
      break;
    }
    std::sort(winners.begin(), winners.end(), [](Candidate const& x, Candidate const& y) {
      return x.improvement > y.improvement;
    });

    std::unordered_set<FaceIndex, CGAL::Handle_hash_function> touched;
    int passSwaps = 0;
    for (auto const& candidate : winners) {
      auto const h = mesh.halfedge(candidate.edge);
      if (mesh.is_removed(candidate.edge) || mesh.is_border(h) ||
          mesh.is_border(mesh.opposite(h))) {
        continue;
      }
      FaceIndex const f0 = mesh.face(h);
      FaceIndex const f1 = mesh.face(mesh.opposite(h));
      if (touched.count(f0) != 0 || touched.count(f1) != 0) {
        continue; // a neighbor already swapped this pass -- this quad's geometry is now stale
      }
      if (!FlipIsSafe(mesh, candidate.edge)) {
        continue; // degree / duplicate-edge / fold may have changed since evaluation
      }
      CGAL::Euler::flip_edge(h, mesh);
      touched.insert(f0);
      touched.insert(f1);
      ++passSwaps;
    }

    if (passSwaps == 0) {
      break;
    }
  }
}

} // namespace

MeshData mochi::mesh::cli::EdgeSwapMesh(
    MeshData const& mesh,
    MeshData const& reference,
    MeshEdgeSwapParams const& params,
    CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});
  MOCHI_MESH_CLI_ERROR_IF(GetNumElements(mesh) == 0, error, "EdgeSwapMesh: input mesh is empty.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});

  // A non-positive threshold disables the swap; return the input unchanged.
  if (!(params.relativeThreshold > 0.0)) {
    return mesh;
  }
  MOCHI_MESH_CLI_ERROR_IF(
      GetNumElements(reference) == 0, error, "EdgeSwapMesh: reference mesh is empty.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});

  try {
    CgalSurfaceMesh sm = cgal_utils::MeshDataToSurfaceMesh(mesh, error);
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});
    // Build a triangle surface from the reference (the high-res Stage 1 tessellation) and an AABB
    // tree over it for ray queries. The reference is a raw unwelded soup -> a set of disconnected
    // triangles, which is fine for ray intersection. referenceSm must outlive the tree.
    CgalSurfaceMesh referenceSm = cgal_utils::MeshDataToSurfaceMesh(reference, error);
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});
    AabbTree tree(faces(referenceSm).first, faces(referenceSm).second, referenceSm);
    tree.build(); // force the lazy build single-threaded before the parallel const queries

    EdgeSwapTowardSurface(
        sm, tree, referenceSm, params.relativeThreshold, std::max(1, params.maxPasses));
    sm.collect_garbage();
    return cgal_utils::SurfaceMeshToMeshData(sm);
  } catch (std::exception const&) {
    MOCHI_MESH_CLI_ERROR_SET(error, "EdgeSwapMesh: CGAL exception during edge swap.");
    return {};
  } catch (...) {
    MOCHI_MESH_CLI_ERROR_SET(error, "EdgeSwapMesh: unknown exception during edge swap.");
    return {};
  }
}
