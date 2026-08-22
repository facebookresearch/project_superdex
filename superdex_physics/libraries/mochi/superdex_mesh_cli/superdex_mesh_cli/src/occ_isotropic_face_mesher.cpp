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

// CGAL must be included before OCCT to avoid the Handle macro conflict (see occ_cgal_compat.h).
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_data_structure_2.h>
#include <CGAL/Triangulation_vertex_base_2.h>

#include "occ_cgal_compat.h"

#include "occ_isotropic_face_mesher.h"

#if MOCHI_USE_OCCT

#include <BRepAdaptor_Surface.hxx>
#include <BRepMesh_ShapeTool.hxx>
#include <BRep_Tool.hxx>
#include <IMeshData_Curve.hxx>
#include <IMeshData_Edge.hxx>
#include <IMeshData_Face.hxx>
#include <IMeshData_PCurve.hxx>
#include <IMeshData_Status.hxx>
#include <IMeshData_Wire.hxx>
#include <IMeshTools_Parameters.hxx>
#include <Poly_Triangulation.hxx>
#include <TopLoc_Location.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Vec.hxx>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace mochi::mesh::cli::occ {

namespace {

// CGAL types for 2D constrained Delaunay triangulation.
using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Vb = CGAL::Triangulation_vertex_base_2<K>;
using Fb = CGAL::Delaunay_mesh_face_base_2<K>;
using Tds = CGAL::Triangulation_data_structure_2<Vb, Fb>;
// Exact_intersections_tag: when boundary constraints cross (self-intersecting wires in parametric
// space), split them at intersection points instead of throwing. This handles complex faces that
// fail with the default tag.
using CDT = CGAL::Constrained_Delaunay_triangulation_2<K, Tds, CGAL::Exact_intersections_tag>;
using Point2 = CDT::Point;
using VertexHandle = CDT::Vertex_handle;

constexpr double kSqConfusion = 1e-14;
constexpr int kMetricSampleDivisor = 10;
constexpr double kMinMetricValue = 1e-20;

// How fine the graded interior may get relative to the target, bounding the cost of a face whose
// boundary is locally very dense.
constexpr double kMinSpacingFraction = 0.125;

/// Determine which pcurve orientation to use for a seam edge. Seam edges have two pcurves on the
/// same face -- if they coincide geometrically, mark as INTERNAL to avoid duplicating constraints.
TopAbs_Orientation FixSeamEdgeOrientation(
    IMeshData::IEdgePtr const& dEdge,
    IMeshData::IPCurveHandle const& pcurve,
    IMeshData::IFacePtr const& dFace) {
  for (Standard_Integer i = 0; i < dEdge->PCurvesNb(); ++i) {
    IMeshData::IPCurveHandle const& other = dEdge->GetPCurve(i);
    if (other->GetFace() == dFace && other != pcurve) {
      gp_Pnt2d const& a1 = pcurve->GetPoint(0);
      gp_Pnt2d const& a2 = pcurve->GetPoint(pcurve->ParametersNb() - 1);
      gp_Pnt2d const& b1 = other->GetPoint(0);
      gp_Pnt2d const& b2 = other->GetPoint(other->ParametersNb() - 1);

      double const d1 = std::min(a1.SquareDistance(b1), a1.SquareDistance(b2));
      double const d2 = std::min(a2.SquareDistance(b1), a2.SquareDistance(b2));

      if (d1 < kSqConfusion && d2 < kSqConfusion) {
        return TopAbs_INTERNAL;
      }
    }
  }
  return pcurve->GetOrientation();
}

/// Ray-casting point-in-polygon test in scaled parametric space.
bool PointInPolygon(
    double px,
    double py,
    IsotropicFaceMesher::Wire const& poly,
    double scaleU,
    double scaleV) {
  bool inside = false;
  size_t const n = poly.size();
  for (size_t i = 0, j = n - 1; i < n; j = i++) {
    double const xi = poly[i].uv.X() * scaleU;
    double const yi = poly[i].uv.Y() * scaleV;
    double const xj = poly[j].uv.X() * scaleU;
    double const yj = poly[j].uv.Y() * scaleV;
    if (((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

/// Test whether a point is inside the outer wire but not in any hole.
bool IsInsideDomain(
    double px,
    double py,
    IsotropicFaceMesher::Wires const& wires,
    double scaleU,
    double scaleV) {
  if (!PointInPolygon(px, py, wires[0], scaleU, scaleV)) {
    return false;
  }
  for (size_t w = 1; w < wires.size(); ++w) {
    if (PointInPolygon(px, py, wires[w], scaleU, scaleV)) {
      return false;
    }
  }
  return true;
}

/// Flood-fill from a seed face, marking in-domain faces across non-constrained edges. Returns true
/// if a valid seed was found and flood-fill succeeded.
bool FloodFillDomain(
    CDT& cdt,
    IsotropicFaceMesher::Wires const& wires,
    double scaleU,
    double scaleV) {
  for (auto fit = cdt.all_faces_begin(); fit != cdt.all_faces_end(); ++fit) {
    fit->set_in_domain(false);
  }

  CDT::Face_handle seedFace;
  bool foundSeed = false;

  for (auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) {
    double const cx =
        (fit->vertex(0)->point().x() + fit->vertex(1)->point().x() + fit->vertex(2)->point().x()) /
        3.0;
    double const cy =
        (fit->vertex(0)->point().y() + fit->vertex(1)->point().y() + fit->vertex(2)->point().y()) /
        3.0;

    if (IsInsideDomain(cx, cy, wires, scaleU, scaleV)) {
      seedFace = fit;
      foundSeed = true;
      break;
    }
  }

  if (!foundSeed) {
    // Fallback: locate face nearest to outer wire centroid.
    double sumU = 0.0;
    double sumV = 0.0;
    for (auto const& bp : wires[0]) {
      sumU += bp.uv.X() * scaleU;
      sumV += bp.uv.Y() * scaleV;
    }
    Point2 const centroid(sumU / wires[0].size(), sumV / wires[0].size());
    seedFace = cdt.locate(centroid);
    if (seedFace == nullptr || cdt.is_infinite(seedFace)) {
      return false;
    }
  }

  // Flood fill: mark in-domain, stop at constrained edges.
  seedFace->set_in_domain(true);
  std::vector<CDT::Face_handle> stack;
  stack.push_back(seedFace);

  while (!stack.empty()) {
    auto face = stack.back();
    stack.pop_back();

    for (int i = 0; i < 3; ++i) {
      auto neighbor = face->neighbor(i);
      if (cdt.is_infinite(neighbor) || neighbor->is_in_domain()) {
        continue;
      }
      if (cdt.is_constrained(CDT::Edge(face, i))) {
        continue;
      }
      neighbor->set_in_domain(true);
      stack.push_back(neighbor);
    }
  }

  return true;
}

// Flood-fills the CDT domain, then corrects for the fill potentially landing on the wrong side.
bool MarkDomain(CDT& cdt, IsotropicFaceMesher::Wires const& wires, double scaleU, double scaleV) {
  if (!FloodFillDomain(cdt, wires, scaleU, scaleV)) {
    return false;
  }

  // The flood seed can land on the wrong side when zero-area boundary triangles pass the
  // "centroid inside" test. Detect this by comparing areas: whichever of {marked, unmarked}
  // is closer to the known polygon area is the true interior. (Face count doesn't work
  // because zero-area triangles inflate count without contributing area.)
  auto wireArea = [scaleU, scaleV](IsotropicFaceMesher::Wire const& wire) {
    double sum = 0.0;
    for (size_t i = 0; i < wire.size(); ++i) {
      size_t const j = (i + 1) % wire.size();
      sum += (wire[i].uv.X() * scaleU) * (wire[j].uv.Y() * scaleV) -
          (wire[j].uv.X() * scaleU) * (wire[i].uv.Y() * scaleV);
    }
    return std::abs(0.5 * sum);
  };

  // Inner wires are holes, so they do not count towards the interior's area.
  double polygonArea = wireArea(wires[0]);
  for (size_t w = 1; w < wires.size(); ++w) {
    polygonArea -= wireArea(wires[w]);
  }

  double markedArea = 0.0;
  double totalArea = 0.0;
  for (auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) {
    double const x0 = fit->vertex(0)->point().x();
    double const y0 = fit->vertex(0)->point().y();
    double const area = 0.5 *
        std::abs((fit->vertex(1)->point().x() - x0) * (fit->vertex(2)->point().y() - y0) -
                 (fit->vertex(2)->point().x() - x0) * (fit->vertex(1)->point().y() - y0));
    totalArea += area;
    if (fit->is_in_domain()) {
      markedArea += area;
    }
  }

  // Flip if the complement is a better match for the polygon area.
  if (std::abs(markedArea - polygonArea) > std::abs((totalArea - markedArea) - polygonArea)) {
    for (auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) {
      fit->set_in_domain(!fit->is_in_domain());
    }
  }
  return true;
}

} // namespace

IsotropicFaceMesher::IsotropicFaceMesher(
    double targetEdgeLength,
    bool gradedInterior,
    Handle(IMeshTools_MeshAlgo) fallbackAlgo,
    FaceMeshFallbackStats* stats)
    : _targetEdgeLength(targetEdgeLength),
      _gradedInterior(gradedInterior),
      _fallbackAlgo(std::move(fallbackAlgo)),
      _stats(stats) {}

std::vector<std::pair<double, double>> IsotropicFaceMesher::GenerateInteriorPoints(
    IMeshData::IFaceHandle const& dFace,
    IMeshTools_Parameters const& parameters,
    Wires const& wires,
    double scaleU,
    double scaleV,
    double uMin,
    double uMax,
    double vMin,
    double vMax) const {
  // How fast the size field relaxes from a fine boundary back to the target.
  // Too low (~0.15) doubles element count; too high (~1.0) causes fans.
  constexpr double kGradeRate = 0.8;
  // Points closer than this fraction of local size to the boundary make slivers.
  constexpr double kBoundaryClearance = 0.35;

  double const minSpacing = _targetEdgeLength * kMinSpacingFraction;

  // Narrow faces (fillets) get curvature-based sizing so their cross-section is resolved.
  // Only applied when the face's narrow extent < 4x target length -broad cylinders and
  // hole walls span many target lengths and are left at the base spacing.
  constexpr double kNarrowFaceThreshold = 4.0;
  double baseSpacing = _targetEdgeLength;
  double const narrow = std::min(uMax - uMin, vMax - vMin);
  if (narrow < kNarrowFaceThreshold * _targetEdgeLength) {
    Handle(BRepAdaptor_Surface) const& surface = dFace->GetSurface();
    constexpr int kSamples = 8;
    double maxCurvature = 0.0;
    for (int iu = 0; iu <= kSamples; ++iu) {
      for (int iv = 0; iv <= kSamples; ++iv) {
        double const u = (uMin + (uMax - uMin) * iu / kSamples) / scaleU;
        double const v = (vMin + (vMax - vMin) * iv / kSamples) / scaleV;
        gp_Pnt p;
        gp_Vec dU;
        gp_Vec dV;
        gp_Vec dUU;
        gp_Vec dVV;
        gp_Vec dUV;
        try {
          surface->D2(u, v, p, dU, dV, dUU, dVV, dUV);
        } catch (Standard_Failure const&) {
          continue;
        }
        gp_Vec const normal = dU.Crossed(dV);
        double const normalMagnitude = normal.Magnitude();
        if (normalMagnitude < Precision::Confusion()) {
          continue; // A pole; neighbouring samples still constrain the spacing.
        }
        gp_Vec const unitNormal = normal / normalMagnitude;

        // Normal curvature: second fundamental form / first (L/E along u, N/G along v).
        double const firstU = dU.SquareMagnitude();
        double const firstV = dV.SquareMagnitude();
        if (firstU > Precision::Confusion()) {
          maxCurvature = std::max(maxCurvature, std::abs(dUU.Dot(unitNormal)) / firstU);
        }
        if (firstV > Precision::Confusion()) {
          maxCurvature = std::max(maxCurvature, std::abs(dVV.Dot(unitNormal)) / firstV);
        }
      }
    }
    if (maxCurvature > 0.0) {
      // Chord length for the angular tolerance, matches the adaptive boundary sampling.
      double const radius = 1.0 / maxCurvature;
      baseSpacing = std::clamp(radius * parameters.Angle, minSpacing, _targetEdgeLength);
    }
  }

  // Local spacing at each boundary vertex: mean of its two incident edge lengths.
  std::vector<BoundarySize> boundary;
  for (auto const& wire : wires) {
    size_t const n = wire.size();
    for (size_t i = 0; i < n; ++i) {
      auto scaled = [scaleU, scaleV](BoundaryPoint const& bp) {
        return std::pair<double, double>{bp.uv.X() * scaleU, bp.uv.Y() * scaleV};
      };
      auto const [cu, cv] = scaled(wire[i]);
      auto const [pu, pv] = scaled(wire[(i + n - 1) % n]);
      auto const [nu, nv] = scaled(wire[(i + 1) % n]);
      double const prev = std::hypot(cu - pu, cv - pv);
      double const next = std::hypot(nu - cu, nv - cv);
      boundary.push_back({cu, cv, std::max(0.5 * (prev + next), minSpacing)});
    }
  }

  // Size field: starts at baseSpacing, pulled down near fine boundary samples,
  // relaxing back at kGradeRate. Early-outs skip samples that can't lower the bound.
  auto sizeAt = [&](double u, double v) {
    double best = baseSpacing;
    for (BoundarySize const& b : boundary) {
      if (b.spacing >= best) {
        continue; // Cannot lower the bound from any distance.
      }
      double const candidate = b.spacing + kGradeRate * std::hypot(u - b.u, v - b.v);
      if (candidate < best) {
        best = candidate;
        if (best <= minSpacing) {
          break;
        }
      }
    }
    return std::max(best, minSpacing);
  };

  auto distanceToBoundarySq = [&](double u, double v) {
    double best = std::numeric_limits<double>::max();
    for (auto const& wire : wires) {
      for (size_t i = 0; i < wire.size(); ++i) {
        size_t const j = (i + 1) % wire.size();
        double const ax = wire[i].uv.X() * scaleU;
        double const ay = wire[i].uv.Y() * scaleV;
        double const dx = wire[j].uv.X() * scaleU - ax;
        double const dy = wire[j].uv.Y() * scaleV - ay;
        double const lenSq = dx * dx + dy * dy;
        double const t =
            (lenSq > 0.0) ? std::clamp(((u - ax) * dx + (v - ay) * dy) / lenSq, 0.0, 1.0) : 0.0;
        double const px = ax + t * dx - u;
        double const py = ay + t * dy - v;
        best = std::min(best, px * px + py * py);
      }
    }
    return best;
  };

  std::vector<std::pair<double, double>> points;

  // Adaptive quad-tree: subdivide until cell size <= local size field, then emit centre.
  // Most of the face stays at base resolution; only the band near fine boundaries fills in.
  auto emit = [&](double u, double v, double cellSize) {
    if (!IsInsideDomain(u, v, wires, scaleU, scaleV)) {
      return;
    }
    double const clearance = cellSize * kBoundaryClearance;
    if (distanceToBoundarySq(u, v) < clearance * clearance) {
      return;
    }
    points.emplace_back(u, v);
  };

  auto subdivide = [&](auto&& self, double u, double v, double cellSize) -> void {
    if (cellSize <= sizeAt(u, v) || cellSize <= minSpacing) {
      emit(u, v, cellSize);
      return;
    }
    double const quarter = cellSize * 0.25;
    double const child = cellSize * 0.5;
    for (double du : {-quarter, quarter}) {
      for (double dv : {-quarter, quarter}) {
        self(self, u + du, v + dv, child);
      }
    }
  };

  for (double u = uMin + baseSpacing * 0.5; u < uMax; u += baseSpacing) {
    for (double v = vMin + baseSpacing * 0.5; v < vMax; v += baseSpacing) {
      subdivide(subdivide, u, v, baseSpacing);
    }
  }
  return points;
}

void IsotropicFaceMesher::FallBack(
    IMeshData::IFaceHandle const& dFace,
    IMeshTools_Parameters const& parameters,
    Message_ProgressRange const& range) {
  int triangleCount = 0;
  if (!_fallbackAlgo.IsNull()) {
    _fallbackAlgo->Perform(dFace, parameters, range);
    TopLoc_Location location;
    Handle(Poly_Triangulation) const triangulation =
        BRep_Tool::Triangulation(dFace->GetFace(), location);
    if (!triangulation.IsNull()) {
      triangleCount = triangulation->NbTriangles();
    }
  }

  if (triangleCount > 0) {
    // The fallback may have flagged the face before recovering; the triangulation is the truth.
    dFace->UnsetStatus(IMeshData_Failure);
  } else {
    dFace->SetStatus(IMeshData_Failure);
  }

  if (_stats != nullptr) {
    _stats->Record(triangleCount > 0);
  }
}

IsotropicFaceMesher::Wires IsotropicFaceMesher::ExtractBoundaryWires(
    IMeshData::IFaceHandle const& dFace) {
  Wires wires;

  for (Standard_Integer wireIdx = 0; wireIdx < dFace->WiresNb(); ++wireIdx) {
    IMeshData::IWireHandle const& wire = dFace->GetWire(wireIdx);
    // NOTE: We intentionally do NOT skip wires flagged IMeshData_SelfIntersectingWire. OCCT sets
    // that flag from the current (possibly coarse) edge discretization, which can produce false
    // positives. Our CDT constraint insertion handles actual intersections gracefully.

    std::vector<BoundaryPoint> wirePoints;

    for (Standard_Integer edgeIdx = 0; edgeIdx < wire->EdgesNb(); ++edgeIdx) {
      IMeshData::IEdgePtr const& edge = wire->GetEdge(edgeIdx);
      TopAbs_Orientation const edgeOri = wire->GetEdgeOrientation(edgeIdx);

      IMeshData::IPCurveHandle const& pcurve = edge->GetPCurve(dFace.get(), edgeOri);
      IMeshData::ICurveHandle const& curve3d = edge->GetCurve();

      TopAbs_Orientation const ori = FixSeamEdgeOrientation(edge, pcurve, dFace.get());

      // Skip internal (seam) edges -- handled by the other pcurve.
      if (ori == TopAbs_INTERNAL) {
        continue;
      }

      Standard_Integer const nPts = pcurve->ParametersNb();

      // Add points in the correct order based on orientation.
      if (ori == TopAbs_REVERSED) {
        for (Standard_Integer i = nPts - 1; i >= 0; --i) {
          wirePoints.push_back({pcurve->GetPoint(i), curve3d->GetPoint(i)});
        }
      } else {
        for (Standard_Integer i = 0; i < nPts; ++i) {
          wirePoints.push_back({pcurve->GetPoint(i), curve3d->GetPoint(i)});
        }
      }
    }

    // Remove duplicate consecutive points (from shared edge vertices).
    if (wirePoints.size() >= 2) {
      std::vector<BoundaryPoint> deduped;
      deduped.push_back(wirePoints[0]);
      for (size_t i = 1; i < wirePoints.size(); ++i) {
        if (wirePoints[i].uv.SquareDistance(deduped.back().uv) > kSqConfusion) {
          deduped.push_back(wirePoints[i]);
        }
      }
      // Also check last vs first for closure.
      if (deduped.size() >= 2 &&
          deduped.back().uv.SquareDistance(deduped.front().uv) < kSqConfusion) {
        deduped.pop_back();
      }
      wirePoints = std::move(deduped);
    }

    if (wirePoints.size() >= 3) {
      wires.push_back(std::move(wirePoints));
    }
  }

  return wires;
}

void IsotropicFaceMesher::ComputeMetricScaling(
    IMeshData::IFaceHandle const& dFace,
    Wires const& wires,
    double& scaleU,
    double& scaleV) {
  Handle(BRepAdaptor_Surface) const& surface = dFace->GetSurface();
  // Sample the first fundamental form at boundary vertices to estimate the average metric tensor.
  // Scale (u,v) by sqrt(E) and sqrt(G) so distances in scaled parametric space approximate 3D
  // distances. Critical for anisotropic surfaces like cylinders where angular scale != axial.
  double sumE = 0.0;
  double sumG = 0.0;
  int nSamples = 0;

  for (auto const& wire : wires) {
    size_t const step = std::max<size_t>(1, wire.size() / kMetricSampleDivisor);
    for (size_t i = 0; i < wire.size(); i += step) {
      gp_Pnt p;
      gp_Vec dSdu;
      gp_Vec dSdv;
      try {
        surface->D1(wire[i].uv.X(), wire[i].uv.Y(), p, dSdu, dSdv);
      } catch (...) {
        continue;
      }
      double const e = dSdu.SquareMagnitude();
      double const g = dSdv.SquareMagnitude();
      if (e > kMinMetricValue && g > kMinMetricValue && std::isfinite(e) && std::isfinite(g)) {
        sumE += e;
        sumG += g;
        ++nSamples;
      }
    }
  }

  scaleU = 1.0;
  scaleV = 1.0;
  if (nSamples > 0) {
    scaleU = std::sqrt(sumE / nSamples);
    scaleV = std::sqrt(sumG / nSamples);
  }
}

void IsotropicFaceMesher::Perform(
    IMeshData::IFaceHandle const& dFace,
    IMeshTools_Parameters const& parameters,
    Message_ProgressRange const& range) {
  Handle(BRepAdaptor_Surface) const& surface = dFace->GetSurface();

  // --- Step 1: Extract boundary wires ---
  Wires const wires = ExtractBoundaryWires(dFace);

  if (wires.empty()) {
    FallBack(dFace, parameters, range);
    return;
  }

  // --- Step 2: Compute metric-based pre-scaling ---
  double scaleU = 1.0;
  double scaleV = 1.0;
  ComputeMetricScaling(dFace, wires, scaleU, scaleV);

  // --- Step 3: Build CGAL CDT2 with scaled boundary constraints ---
  CDT cdt;

  // Map boundary CDT vertices to their original 3D positions from edge discretization. This is
  // critical: surface->Value(u,v) on two different faces for the same physical boundary point
  // produces slightly different coordinates, causing non-manifold edges when merging per-face
  // meshes. Using curve3d->GetPoint() ensures identical 3D positions across faces.
  std::map<VertexHandle, gp_Pnt> boundaryPt3d;
  std::map<VertexHandle, gp_Pnt2d> boundaryUV;

  for (auto const& wire : wires) {
    std::vector<VertexHandle> vhs;
    vhs.reserve(wire.size());

    for (auto const& bp : wire) {
      Point2 const scaledPt(bp.uv.X() * scaleU, bp.uv.Y() * scaleV);
      VertexHandle const vh = cdt.insert(scaledPt);
      vhs.push_back(vh);
      boundaryPt3d[vh] = bp.p3d;
      boundaryUV[vh] = bp.uv;
    }

    // Insert constrained edges forming the closed polygon.
    for (size_t i = 0; i < vhs.size(); ++i) {
      size_t const next = (i + 1) % vhs.size();
      if (vhs[i] != vhs[next]) {
        try {
          cdt.insert_constraint(vhs[i], vhs[next]);
        } catch (...) {
          // Should not happen with Exact_intersections_tag, but guard just in case.
          std::cerr << "[IsotropicFaceMesher] WARNING: unexpected constraint failure, edge " << i
                    << "->" << next << "\n";
        }
      }
    }
  }

  // --- Step 4: Mark in-domain faces via flood fill ---
  // We must NOT use refine_Delaunay_mesh_2 because it splits encroached constrained edges, adding
  // Steiner points on the boundary that won't exist on adjacent faces, causing non-manifold edges
  // in the merged mesh.
  if (!MarkDomain(cdt, wires, scaleU, scaleV)) {
    FallBack(dFace, parameters, range);
    return;
  }

  // --- Step 4b: Refine interior with Steiner points ---
  // These are non-constrained points, so boundary edges are never split, preserving cross-face
  // vertex matching. Spacing is uniform at the target length, or graded towards the boundary's own
  // density when that is finer.
  {
    double uMin = std::numeric_limits<double>::max();
    double uMax = std::numeric_limits<double>::lowest();
    double vMin = std::numeric_limits<double>::max();
    double vMax = std::numeric_limits<double>::lowest();
    for (auto const& wire : wires) {
      for (auto const& bp : wire) {
        double const su = bp.uv.X() * scaleU;
        double const sv = bp.uv.Y() * scaleV;
        uMin = std::min(uMin, su);
        uMax = std::max(uMax, su);
        vMin = std::min(vMin, sv);
        vMax = std::max(vMax, sv);
      }
    }

    // Skip if the domain is too small to fit interior points. Graded refinement uses
    // minSpacing (not target length) so it can still fill narrow fillets.
    double const minExtent =
        _gradedInterior ? _targetEdgeLength * kMinSpacingFraction : _targetEdgeLength;
    if ((uMax - uMin) > minExtent && (vMax - vMin) > minExtent) {
      std::vector<std::pair<double, double>> interior;
      if (_gradedInterior) {
        interior = GenerateInteriorPoints(
            dFace, parameters, wires, scaleU, scaleV, uMin, uMax, vMin, vMax);
      } else {
        double const spacing = _targetEdgeLength;
        // Grid points closer than this to any boundary edge create thin triangles at face seams.
        double const minBoundaryDist = spacing * 0.35;
        double const minBoundaryDistSq = minBoundaryDist * minBoundaryDist;

        // Offset by half spacing to avoid points on the boundary.
        for (double u = uMin + spacing * 0.5; u < uMax; u += spacing) {
          for (double v = vMin + spacing * 0.5; v < vMax; v += spacing) {
            if (!IsInsideDomain(u, v, wires, scaleU, scaleV)) {
              continue;
            }

            bool tooClose = false;
            for (auto const& wire : wires) {
              for (size_t i = 0; i < wire.size() && !tooClose; ++i) {
                size_t const j = (i + 1) % wire.size();
                double const ax = wire[i].uv.X() * scaleU;
                double const ay = wire[i].uv.Y() * scaleV;
                double const bx = wire[j].uv.X() * scaleU;
                double const by = wire[j].uv.Y() * scaleV;

                // Point-to-segment distance squared.
                double const dx = bx - ax;
                double const dy = by - ay;
                double const lenSq = dx * dx + dy * dy;
                double const t = (lenSq > 0.0)
                    ? std::clamp(((u - ax) * dx + (v - ay) * dy) / lenSq, 0.0, 1.0)
                    : 0.0;
                double const px = ax + t * dx - u;
                double const py = ay + t * dy - v;
                if (px * px + py * py < minBoundaryDistSq) {
                  tooClose = true;
                }
              }
              if (tooClose) {
                break;
              }
            }

            if (!tooClose) {
              interior.emplace_back(u, v);
            }
          }
        }
      }

      if (!interior.empty()) {
        for (auto const& p : interior) {
          cdt.insert(Point2(p.first, p.second));
        }
        // Re-mark the domain now that the interior points are in.
        if (!MarkDomain(cdt, wires, scaleU, scaleV)) {
          FallBack(dFace, parameters, range);
          return;
        }
      }
    }
  }

  // --- Step 5: Collect in-domain faces, filter degenerates, build Poly_Triangulation ---

  // Sliver filter: area / longestEdge^2 is ~0.433 for equilateral, -> 0 for slivers.
  // Scale-invariant so small well-formed faces (fillet ends, chamfer corners) aren't rejected.
  // Empirically derived with many STEP files. Could probably be fine-tuned further.
  constexpr double kMinShapeQuality = 0.433 * 0.01;

  // Transform CDT vertices back from scaled space to original (u,v) before evaluating the surface.
  double const invScaleU = 1.0 / scaleU;
  double const invScaleV = 1.0 / scaleV;

  std::map<VertexHandle, Standard_Integer> vertexIndex;
  std::vector<gp_Pnt> nodes3d;
  std::vector<gp_Pnt2d> nodesUV;

  // First pass: collect all vertices from in-domain faces.
  for (auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) {
    if (!fit->is_in_domain()) {
      continue;
    }
    for (int i = 0; i < 3; ++i) {
      VertexHandle const vh = fit->vertex(i);
      if (vertexIndex.find(vh) == vertexIndex.end()) {
        Standard_Integer const idx = static_cast<Standard_Integer>(nodes3d.size()) + 1; // 1-based
        vertexIndex[vh] = idx;

        gp_Pnt p3d;
        gp_Pnt2d uv;

        // For boundary vertices use the original 3D position from edge discretization to ensure an
        // exact match across adjacent faces. For interior Steiner vertices evaluate the surface.
        auto bndIt = boundaryPt3d.find(vh);
        if (bndIt != boundaryPt3d.end()) {
          p3d = bndIt->second;
          uv = boundaryUV[vh];
        } else {
          double const u = vh->point().x() * invScaleU;
          double const v = vh->point().y() * invScaleV;
          uv = gp_Pnt2d(u, v);
          p3d = surface->Value(u, v);
        }

        nodes3d.push_back(p3d);
        nodesUV.push_back(uv);
      }
    }
  }

  // Second pass: collect triangles, filtering near-zero-area ones.
  struct TriIndices {
    Standard_Integer n1{};
    Standard_Integer n2{};
    Standard_Integer n3{};
  };
  std::vector<TriIndices> validTriangles;
  validTriangles.reserve(cdt.number_of_faces());

  for (auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) {
    if (!fit->is_in_domain()) {
      continue;
    }

    Standard_Integer const n1 = vertexIndex[fit->vertex(0)];
    Standard_Integer const n2 = vertexIndex[fit->vertex(1)];
    Standard_Integer const n3 = vertexIndex[fit->vertex(2)];

    // Skip truly degenerate triangles (duplicate vertices).
    if (n1 == n2 || n2 == n3 || n1 == n3) {
      continue;
    }

    // UV-space sliver test. Collinear boundary samples (straight parametric edges) produce
    // zero-area UV triangles that look fine in 3D on curved surfaces, reject them here.
    gp_Pnt2d const& uv1 = nodesUV[n1 - 1];
    gp_Pnt2d const& uv2 = nodesUV[n2 - 1];
    gp_Pnt2d const& uv3 = nodesUV[n3 - 1];
    double const su1 = uv2.X() * scaleU - uv1.X() * scaleU;
    double const sv1 = uv2.Y() * scaleV - uv1.Y() * scaleV;
    double const su2 = uv3.X() * scaleU - uv1.X() * scaleU;
    double const sv2 = uv3.Y() * scaleV - uv1.Y() * scaleV;
    double const uvArea = 0.5 * std::abs(su1 * sv2 - su2 * sv1);
    double const uvLongestEdgeSq = std::max(
        {su1 * su1 + sv1 * sv1,
         su2 * su2 + sv2 * sv2,
         (su2 - su1) * (su2 - su1) + (sv2 - sv1) * (sv2 - sv1)});
    if (uvArea < kMinShapeQuality * uvLongestEdgeSq) {
      continue;
    }

    // 3D triangle area via cross product.
    gp_Pnt const& p1 = nodes3d[n1 - 1];
    gp_Pnt const& p2 = nodes3d[n2 - 1];
    gp_Pnt const& p3 = nodes3d[n3 - 1];
    double const ax = p2.X() - p1.X();
    double const ay = p2.Y() - p1.Y();
    double const az = p2.Z() - p1.Z();
    double const bx = p3.X() - p1.X();
    double const by = p3.Y() - p1.Y();
    double const bz = p3.Z() - p1.Z();
    double const cx = ay * bz - az * by;
    double const cy = az * bx - ax * bz;
    double const cz = ax * by - ay * bx;
    double const area = 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
    double const longestEdgeSq = std::max(
        {ax * ax + ay * ay + az * az,
         bx * bx + by * by + bz * bz,
         (bx - ax) * (bx - ax) + (by - ay) * (by - ay) + (bz - az) * (bz - az)});

    if (area >= kMinShapeQuality * longestEdgeSq) {
      validTriangles.push_back({n1, n2, n3});
    }
  }

  Standard_Integer const numTriangles = static_cast<Standard_Integer>(validTriangles.size());
  if (numTriangles == 0) {
    // All slivers - degenerate face (e.g. hairline boolean artifact). Fall back.
    FallBack(dFace, parameters, range);
    return;
  }

  Handle(Poly_Triangulation) triangulation = new Poly_Triangulation();
  Standard_Integer const numNodes = static_cast<Standard_Integer>(nodes3d.size());
  triangulation->ResizeNodes(numNodes, false);
  triangulation->AddUVNodes();
  triangulation->ResizeTriangles(numTriangles, false);

  for (Standard_Integer i = 0; i < numNodes; ++i) {
    triangulation->SetNode(i + 1, nodes3d[i]);
    triangulation->SetUVNode(i + 1, nodesUV[i]);
  }

  for (Standard_Integer i = 0; i < numTriangles; ++i) {
    triangulation->SetTriangle(
        i + 1, Poly_Triangle(validTriangles[i].n1, validTriangles[i].n2, validTriangles[i].n3));
  }

  // --- Step 6: Commit to OCCT face ---
  BRepMesh_ShapeTool::AddInFace(dFace->GetFace(), triangulation);
}

} // namespace mochi::mesh::cli::occ

#endif // MOCHI_USE_OCCT
