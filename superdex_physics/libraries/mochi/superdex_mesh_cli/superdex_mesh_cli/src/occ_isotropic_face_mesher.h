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

#include <IMeshTools_MeshAlgo.hxx>

#include <BRepAdaptor_Surface.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>

#include <atomic>
#include <utility>
#include <vector>

namespace mochi::mesh::cli::occ {

/// Tally of faces the CGAL mesher could not triangulate and how many the fallback rescued. Shared
/// by every face mesher in a run; BRepMesh_FaceDiscret meshes faces in parallel.
class FaceMeshFallbackStats {
 public:
  void Record(bool rescued) {
    ++_failed;
    if (rescued) {
      ++_rescued;
    }
  }

  int FailedCount() const {
    return _failed;
  }

  int RescuedCount() const {
    return _rescued;
  }

 private:
  std::atomic<int> _failed{0};
  std::atomic<int> _rescued{0};
};

/// Per-face meshing algorithm using CGAL's constrained Delaunay triangulation. Operates in 2D
/// parametric space with metric-aware sizing, so spacing stays isotropic in 3D across a face
/// however skewed its parametrisation. Plugs into OCCT's BRepMesh pipeline (via
/// BRepMesh_FaceDiscret) as a replacement for the default Watson/Delabella algorithms.
///
/// Degenerate CAD occasionally defeats the CGAL path on individual faces. Rather than leaving a
/// hole in the model, such a face is retried with @p fallbackAlgo (OCCT's own mesher).
class IsotropicFaceMesher : public IMeshTools_MeshAlgo {
 public:
  /// @param targetEdgeLength Target edge length in 3D model units.
  /// @param gradedInterior Grade interior sampling towards the boundary's own density instead of
  ///        holding it at @p targetEdgeLength everywhere. Without it, a boundary that curvature
  ///        has already refined -- a fillet's cross-section, say -- has nothing to attach to but
  ///        one distant interior point, producing a fan.
  /// @param fallbackAlgo Mesher to retry a failed face with; may be null to disable the retry.
  /// @param stats Fallback tally; may be null. Must outlive this mesher.
  IsotropicFaceMesher(
      double targetEdgeLength,
      bool gradedInterior,
      Handle(IMeshTools_MeshAlgo) fallbackAlgo,
      FaceMeshFallbackStats* stats);

  ~IsotropicFaceMesher() override = default;

  /// Triangulates a single face using CGAL CDT2 + grid refinement.
  void Perform(
      IMeshData::IFaceHandle const& dFace,
      IMeshTools_Parameters const& parameters,
      Message_ProgressRange const& range) override;

  struct BoundaryPoint {
    gp_Pnt2d uv;
    gp_Pnt p3d;
  };

  using Wire = std::vector<BoundaryPoint>;
  using Wires = std::vector<Wire>;

 private:
  /// Extract boundary wires from OCCT face data in 2D parametric space.
  Wires ExtractBoundaryWires(IMeshData::IFaceHandle const& dFace);

  /// Sample the first fundamental form along boundary wires to compute average metric-based scaling
  /// factors mapping (u,v) -> approximate 3D distances.
  void ComputeMetricScaling(
      IMeshData::IFaceHandle const& dFace,
      Wires const& wires,
      double& scaleU,
      double& scaleV);

  /// A boundary sample paired with the edge spacing local to it, in metric-scaled parametric
  /// units. Drives the interior size field.
  struct BoundarySize {
    double u{};
    double v{};
    double spacing{};
  };

  /// Interior sample positions in metric-scaled parametric space, graded from the boundary.
  std::vector<std::pair<double, double>> GenerateInteriorPoints(
      IMeshData::IFaceHandle const& dFace,
      IMeshTools_Parameters const& parameters,
      Wires const& wires,
      double scaleU,
      double scaleV,
      double uMin,
      double uMax,
      double vMin,
      double vMax) const;

  /// Retries @p dFace with the fallback mesher. Marks the face IMeshData_Failure only if the
  /// fallback also produced nothing.
  void FallBack(
      IMeshData::IFaceHandle const& dFace,
      IMeshTools_Parameters const& parameters,
      Message_ProgressRange const& range);

  double _targetEdgeLength{};
  bool _gradedInterior{};
  Handle(IMeshTools_MeshAlgo) _fallbackAlgo;
  FaceMeshFallbackStats* _stats{};
};

} // namespace mochi::mesh::cli::occ

#endif // MOCHI_USE_OCCT
