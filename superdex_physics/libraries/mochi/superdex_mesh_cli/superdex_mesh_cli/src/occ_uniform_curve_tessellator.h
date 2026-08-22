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

// Part of the OCCT BRepMesh isotropic tessellation pipeline. Only compiled where OpenCascade
// is available (MOCHI_USE_OCCT); elsewhere this header is an empty translation unit.
#if MOCHI_USE_OCCT

#include <BRepAdaptor_Curve.hxx>
#include <IMeshData_Types.hxx>
#include <IMeshTools_CurveTessellator.hxx>
#include <IMeshTools_Parameters.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Vertex.hxx>

#include <vector>

namespace mochi::mesh::cli::occ {

/// How an edge's sample count is chosen. Both modes place their samples at even arc-length
/// intervals; they differ only in what sets the count.
enum class CurveSampling {
  /// Count taken from arc length alone. Perfectly even, but blind to curvature: a curve short
  /// relative to the target edge length collapses to a chord.
  Uniform,

  /// Count raised to whatever the deflection tolerance requires. The target edge length becomes a
  /// ceiling rather than the sole input, so a small hole refines itself while the samples stay
  /// evenly spaced.
  Adaptive,
};

/// Curve tessellator that places uniform arc-length samples (GCPnts_UniformAbscissa), optionally
/// raising their count to whatever OCCT's curvature analysis (GCPnts_TangentialDeflection) asks
/// for, so edge segments never exceed a maximum length while still resolving curvature. Feeds the
/// per-face CGAL mesher with boundary constraint points dense enough for uniform interior
/// triangulation.
class UniformCurveTessellator : public IMeshTools_CurveTessellator {
 public:
  /// Constructor for free edges (not on any face).
  UniformCurveTessellator(
      IMeshData::IEdgeHandle const& edge,
      IMeshTools_Parameters const& parameters,
      double maxEdgeLength,
      CurveSampling sampling,
      Standard_Integer minPointsNb = 2);

  /// Constructor for edges on a face (non-sameparam case).
  UniformCurveTessellator(
      IMeshData::IEdgeHandle const& edge,
      TopAbs_Orientation orientation,
      IMeshData::IFaceHandle const& face,
      IMeshTools_Parameters const& parameters,
      double maxEdgeLength,
      CurveSampling sampling,
      Standard_Integer minPointsNb = 2);

  ~UniformCurveTessellator() override = default;

  Standard_Integer PointsNb() const override;

  Standard_Boolean Value(Standard_Integer index, gp_Pnt& point, Standard_Real& parameter)
      const override;

  DEFINE_STANDARD_RTTIEXT(UniformCurveTessellator, IMeshTools_CurveTessellator)

 private:
  void Init();

  struct TessPoint {
    Standard_Real param{};
    gp_Pnt point;
  };

  IMeshData::IEdgeHandle _dEdge;
  TopoDS_Edge _edge;
  BRepAdaptor_Curve _curve;
  double _maxEdgeLength{};
  CurveSampling _sampling{};
  IMeshTools_Parameters _parameters;
  Standard_Integer _minPointsNb{};
  TopoDS_Vertex _firstVertex;
  TopoDS_Vertex _lastVertex;

  std::vector<TessPoint> _points;
};

} // namespace mochi::mesh::cli::occ

#endif // MOCHI_USE_OCCT
