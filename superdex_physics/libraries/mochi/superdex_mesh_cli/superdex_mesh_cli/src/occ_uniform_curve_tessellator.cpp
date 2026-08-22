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

#include "occ_uniform_curve_tessellator.h"

#if MOCHI_USE_OCCT

#include <BRep_Tool.hxx>
#include <GCPnts_AbscissaPoint.hxx>
#include <GCPnts_TangentialDeflection.hxx>
#include <GCPnts_UniformAbscissa.hxx>
#include <IMeshData_Edge.hxx>
#include <IMeshData_Face.hxx>
#include <Precision.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <gp_Pnt.hxx>

#include <algorithm>
#include <cmath>
#include <set>

IMPLEMENT_STANDARD_RTTIEXT(
    mochi::mesh::cli::occ::UniformCurveTessellator,
    IMeshTools_CurveTessellator)

namespace mochi::mesh::cli::occ {

UniformCurveTessellator::UniformCurveTessellator(
    IMeshData::IEdgeHandle const& edge,
    IMeshTools_Parameters const& parameters,
    double maxEdgeLength,
    CurveSampling sampling,
    Standard_Integer minPointsNb)
    : _dEdge(edge),
      _edge(edge->GetEdge()),
      _curve(_edge),
      _maxEdgeLength(maxEdgeLength),
      _sampling(sampling),
      _parameters(parameters),
      _minPointsNb(minPointsNb) {
  Init();
}

UniformCurveTessellator::UniformCurveTessellator(
    IMeshData::IEdgeHandle const& edge,
    TopAbs_Orientation orientation,
    IMeshData::IFaceHandle const& face,
    IMeshTools_Parameters const& parameters,
    double maxEdgeLength,
    CurveSampling sampling,
    Standard_Integer minPointsNb)
    : _dEdge(edge),
      _edge(TopoDS::Edge(edge->GetEdge().Oriented(orientation))),
      _curve(_edge, face->GetFace()),
      _maxEdgeLength(maxEdgeLength),
      _sampling(sampling),
      _parameters(parameters),
      _minPointsNb(minPointsNb) {
  Init();
}

void UniformCurveTessellator::Init() {
  TopExp::Vertices(_edge, _firstVertex, _lastVertex);

  std::set<Standard_Real> paramSet;

  // Minimum points a curve of this type needs to be described at all. A circular edge sampled at
  // only two points collapses to a chord, and a wire built from two such edges (the usual
  // split-circle boundary of a hole) degenerates to a line with fewer than three distinct points --
  // which no per-face mesher can triangulate, leaving a hole in the output. This floor applies to
  // both sampling modes.
  Standard_Integer minPntThreshold = 2;
  switch (_curve.GetType()) {
    case GeomAbs_Circle:
    case GeomAbs_Ellipse:
    case GeomAbs_Parabola:
    case GeomAbs_Hyperbola:
      minPntThreshold = 4;
      break;
    default:
      break;
  }
  Standard_Integer const minPntNb = Max(_minPointsNb, minPntThreshold);

  // Adaptive takes only a point count from OCCT's curvature analysis; the uniform pass below then
  // redistributes that count evenly. It asks for the tolerances the caller actually requested --
  // BRepMesh_CurveTessellator's convention of halving them applies to points that land directly in
  // the mesh, which these do not.
  Standard_Integer deflectionSegments = 0;

  if (_sampling == CurveSampling::Adaptive) {
    Standard_Real const preciseAngDef = Max(_dEdge->GetAngularDeflection(), Precision::Angular());
    Standard_Real const preciseLinDef = Max(_dEdge->GetDeflection(), Precision::Confusion());

    Standard_Real minSize = _parameters.MinSize;
    if (_parameters.AdjustMinSize) {
      minSize =
          Min(minSize,
              _parameters.RelMinSize() *
                  GCPnts_AbscissaPoint::Length(
                      _curve, _curve.FirstParameter(), _curve.LastParameter(), preciseLinDef));
    }

    GCPnts_TangentialDeflection deflectionTool;
    deflectionTool.Initialize(
        _curve,
        _curve.FirstParameter(),
        _curve.LastParameter(),
        preciseAngDef,
        preciseLinDef,
        minPntNb,
        Precision::PConfusion(),
        minSize);

    // Add internal vertices (same as BRepMesh_CurveTessellator).
    for (TopExp_Explorer vertexIt(_edge, TopAbs_VERTEX); vertexIt.More(); vertexIt.Next()) {
      TopoDS_Vertex const& vertex = TopoDS::Vertex(vertexIt.Current());
      if (vertex.Orientation() == TopAbs_INTERNAL) {
        deflectionTool.AddPoint(
            BRep_Tool::Pnt(vertex), BRep_Tool::Parameter(vertex, _edge), Standard_True);
      }
    }

    deflectionSegments = Max(deflectionTool.NbPoints() - 1, 1);
  }

  // Seed only the endpoints and any internal vertices; the uniform pass below fills the interior.
  paramSet.insert(_curve.FirstParameter());
  paramSet.insert(_curve.LastParameter());

  for (TopExp_Explorer vertexIt(_edge, TopAbs_VERTEX); vertexIt.More(); vertexIt.Next()) {
    TopoDS_Vertex const& vertex = TopoDS::Vertex(vertexIt.Current());
    if (vertex.Orientation() == TopAbs_INTERNAL) {
      paramSet.insert(BRep_Tool::Parameter(vertex, _edge));
    }
  }

  // Uniform arc-length sampling via GCPnts_UniformAbscissa. round() picks the segment count rather
  // than relying on the tool's internal truncation, so edges of similar length get the same point
  // count -- this avoids the "half-step phase" problem where adjacent N vs N+1 edges create
  // degenerate triangles.
  if (_maxEdgeLength > 0.0) {
    double const curveLength =
        GCPnts_AbscissaPoint::Length(_curve, _curve.FirstParameter(), _curve.LastParameter());
    Standard_Integer numSegments =
        std::max(1, static_cast<Standard_Integer>(std::round(curveLength / _maxEdgeLength)));
    // Adaptive: the target edge length is only a ceiling. Raising the count to whatever the
    // deflection needs is what lets a small hole resolve itself, and because only the count moves
    // the samples stay evenly spaced.
    //
    // The count is uniform over the whole edge, so a long edge with one tight bend carries that
    // bend's density along its full length. Measured on real assemblies this costs little, because
    // the criterion is angular and therefore scale-invariant: it bites on small curved features,
    // which are cheap, and is already satisfied on large ones.
    numSegments = Max(numSegments, deflectionSegments);
    // Never sample a curved edge coarser than its type requires (see minPntNb above).
    Standard_Integer const numPoints = std::max(numSegments + 1, minPntNb);
    GCPnts_UniformAbscissa uniformTool(_curve, numPoints);
    if (uniformTool.IsDone()) {
      for (Standard_Integer i = 1; i <= uniformTool.NbPoints(); ++i) {
        paramSet.insert(uniformTool.Parameter(i));
      }
    }
  }

  _points.reserve(paramSet.size());
  for (Standard_Real const param : paramSet) {
    gp_Pnt p;
    _curve.D0(param, p);
    _points.push_back({param, p});
  }
}

Standard_Integer UniformCurveTessellator::PointsNb() const {
  return static_cast<Standard_Integer>(_points.size());
}

Standard_Boolean UniformCurveTessellator::Value(
    Standard_Integer index,
    gp_Pnt& point,
    Standard_Real& parameter) const {
  if (index < 1 || index > PointsNb()) {
    return Standard_False;
  }
  TessPoint const& pt = _points[index - 1]; // 1-based index
  point = pt.point;
  parameter = pt.param;
  return Standard_True;
}

} // namespace mochi::mesh::cli::occ

#endif // MOCHI_USE_OCCT
