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

#include "occ_uniform_edge_discret.h"

#if MOCHI_USE_OCCT

#include "occ_uniform_curve_tessellator.h"

#include <BRepMesh_Deflection.hxx>
#include <BRepMesh_EdgeDiscret.hxx>
#include <BRepMesh_ShapeTool.hxx>
#include <BRep_Tool.hxx>
#include <IMeshData_Edge.hxx>
#include <IMeshData_Face.hxx>
#include <IMeshData_Model.hxx>
#include <IMeshData_PCurve.hxx>
#include <OSD_Parallel.hxx>
#include <Standard_Failure.hxx>
#include <TopExp.hxx>

IMPLEMENT_STANDARD_RTTIEXT(mochi::mesh::cli::occ::UniformEdgeDiscretizer, IMeshTools_ModelAlgo)

namespace mochi::mesh::cli::occ {

UniformEdgeDiscretizer::UniformEdgeDiscretizer(double maxEdgeLength, CurveSampling sampling)
    : _maxEdgeLength(maxEdgeLength), _sampling(sampling) {}

Standard_Boolean UniformEdgeDiscretizer::performInternal(
    Handle(IMeshData_Model) const& model,
    IMeshTools_Parameters const& parameters,
    Message_ProgressRange const& /*range*/) {
  _model = model;
  _parameters = parameters;

  if (_model.IsNull()) {
    return Standard_False;
  }

  OSD_Parallel::For(0, _model->EdgesNb(), *this, !_parameters.InParallel);

  _model.Nullify();
  return Standard_True;
}

void UniformEdgeDiscretizer::Process(Standard_Integer edgeIndex) const {
  IMeshData::IEdgeHandle const& dEdge = _model->GetEdge(edgeIndex);
  try {
    OCC_CATCH_SIGNALS

    BRepMesh_Deflection::ComputeDeflection(dEdge, _model->GetMaxSize(), _parameters);

    Handle(IMeshTools_CurveTessellator) edgeTessellator;

    if (!dEdge->IsFree()) {
      // For edges on faces, always freshly tessellate with our uniform tessellator. We skip the
      // polygon-reuse optimization because our strategy produces more points than any cache.
      IMeshData::IPCurveHandle const& pCurve = dEdge->GetPCurve(0);
      IMeshData::IFaceHandle const dFace = pCurve->GetFace();

      BRepMesh_ShapeTool::CheckAndUpdateFlags(dEdge, pCurve);
      for (Standard_Integer i = 1; i < dEdge->PCurvesNb(); ++i) {
        BRepMesh_ShapeTool::CheckAndUpdateFlags(dEdge, dEdge->GetPCurve(i));
      }

      if (dEdge->GetSameParam()) {
        edgeTessellator =
            new UniformCurveTessellator(dEdge, _parameters, _maxEdgeLength, _sampling);
      } else {
        edgeTessellator = new UniformCurveTessellator(
            dEdge, pCurve->GetOrientation(), dFace, _parameters, _maxEdgeLength, _sampling);
      }
    } else {
      // Free edges (not on any face).
      edgeTessellator = new UniformCurveTessellator(dEdge, _parameters, _maxEdgeLength, _sampling);
    }

    BRepMesh_EdgeDiscret::Tessellate3d(dEdge, edgeTessellator, Standard_True);

    if (!dEdge->IsFree()) {
      BRepMesh_EdgeDiscret::Tessellate2d(dEdge, Standard_True);
    }
  } catch (Standard_Failure const&) {
    dEdge->SetStatus(IMeshData_Failure);
  }
}

} // namespace mochi::mesh::cli::occ

#endif // MOCHI_USE_OCCT
