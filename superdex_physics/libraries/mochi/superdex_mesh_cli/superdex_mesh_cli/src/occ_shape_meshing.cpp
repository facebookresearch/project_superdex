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

#include "occ_shape_meshing.h"

#if MOCHI_USE_OCCT

#include "occ_isotropic_face_mesher_factory.h"
#include "occ_uniform_edge_discret.h"

#include <BRepMesh_Context.hxx>
#include <BRepMesh_FaceDiscret.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <Bnd_Box.hxx>
#include <IMeshTools_MeshAlgoType.hxx>
#include <IMeshTools_Parameters.hxx>
#include <OSD_Parallel.hxx>
#include <TopoDS_Shape.hxx>

#include <algorithm>
#include <cmath>

namespace {

// Floor for the auto-derived target edge length [mm], guarding against a zero/degenerate diagonal.
constexpr double kMinTargetEdgeLength = 1e-3;

} // namespace

double mochi::mesh::cli::occ::ResolveTargetEdgeLength(
    Bnd_Box const& bbox,
    double explicitLength,
    double fraction) {
  if (explicitLength > 0.0) {
    return explicitLength;
  }
  if (bbox.IsVoid()) {
    return kMinTargetEdgeLength;
  }
  double xmin = 0.0;
  double ymin = 0.0;
  double zmin = 0.0;
  double xmax = 0.0;
  double ymax = 0.0;
  double zmax = 0.0;
  bbox.Get(xmin, ymin, zmin, xmax, ymax, zmax);
  double const diagonal = std::sqrt(
      (xmax - xmin) * (xmax - xmin) + (ymax - ymin) * (ymax - ymin) +
      (zmax - zmin) * (zmax - zmin));
  return std::max(diagonal * fraction, kMinTargetEdgeLength);
}

IMeshData_Status mochi::mesh::cli::occ::MeshShape(
    TopoDS_Shape const& shape,
    ShapeMeshingParams const& params,
    FaceMeshFallbackStats* fallbackStats) {
  IMeshTools_Parameters meshParams;
  meshParams.Deflection = params.linearDeflection;
  meshParams.Angle = params.angularDeflection;
  meshParams.Relative = false;
  meshParams.InParallel = true;

  // Delabella is OpenCascade's newer Delaunay mesher and is NOT the default (the historic default
  // is Watson), so it must be selected via the meshing context. It is also what an unhandled face
  // falls back to under FaceMesher::Isotropic.
  Handle(BRepMesh_Context) const context = new BRepMesh_Context(IMeshTools_MeshAlgoType_Delabella);

  if (params.faceMesher == FaceMesher::Isotropic) {
    // Only Adaptive grades the interior, because only Adaptive lets curvature refine the edges.
    // Pairing them keeps boundary and interior densities consistent; Uniform samples the interior
    // uniformly, matching its uniform boundary.
    bool const adaptive = params.edgeSampling == StepMeshBodyParams::EdgeSampling::Adaptive;
    context->SetFaceDiscret(new BRepMesh_FaceDiscret(
        new IsotropicFaceMesherFactory(params.targetEdgeLength, adaptive, fallbackStats)));
    context->SetEdgeDiscret(new UniformEdgeDiscretizer(
        params.targetEdgeLength, adaptive ? CurveSampling::Adaptive : CurveSampling::Uniform));
  }

  // These mutate global OCCT state; safe here because the helper is a single-shot process.
  OSD_Parallel::SetUseOcctThreads(false);
  BRepMesh_IncrementalMesh::SetParallelDefault(true);

  BRepMesh_IncrementalMesh mesher;
  mesher.SetShape(shape);
  mesher.ChangeParameters() = meshParams;
  mesher.Perform(context);

  return static_cast<IMeshData_Status>(mesher.GetStatusFlags());
}

#endif // MOCHI_USE_OCCT
