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

#include "occ_uniform_curve_tessellator.h"

#include <IMeshData_Types.hxx>
#include <IMeshTools_ModelAlgo.hxx>
#include <IMeshTools_Parameters.hxx>

namespace mochi::mesh::cli::occ {

/// Model-level edge discretizer that caps edge segment length. Places evenly spaced arc-length
/// samples (see UniformCurveTessellator), optionally raising their count to resolve curvature, so
/// long straight edges get interior points suitable for the per-face CGAL mesher. Installed via
/// IMeshTools_Context::SetEdgeDiscret.
class UniformEdgeDiscretizer : public IMeshTools_ModelAlgo {
 public:
  UniformEdgeDiscretizer(double maxEdgeLength, CurveSampling sampling);
  ~UniformEdgeDiscretizer() override = default;

  /// Functor API for OSD_Parallel::For.
  void operator()(Standard_Integer edgeIndex) const {
    Process(edgeIndex);
  }

  DEFINE_STANDARD_RTTIEXT(UniformEdgeDiscretizer, IMeshTools_ModelAlgo)

 protected:
  Standard_Boolean performInternal(
      Handle(IMeshData_Model) const& model,
      IMeshTools_Parameters const& parameters,
      Message_ProgressRange const& range) override;

 private:
  void Process(Standard_Integer edgeIndex) const;

  double _maxEdgeLength{};
  CurveSampling _sampling{};
  Handle(IMeshData_Model) _model;
  IMeshTools_Parameters _parameters;
};

} // namespace mochi::mesh::cli::occ

#endif // MOCHI_USE_OCCT
