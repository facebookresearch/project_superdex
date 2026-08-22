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

#include "occ_isotropic_face_mesher.h"

#include <BRepMesh_DelabellaMeshAlgoFactory.hxx>
#include <IMeshTools_MeshAlgoFactory.hxx>

namespace mochi::mesh::cli::occ {

/// Factory returning IsotropicFaceMesher instances for all surface types. Plugs into OCCT's
/// BRepMesh pipeline via BRepMesh_FaceDiscret. Each mesher is handed a stock Delabella algorithm to
/// retry with, so a face the CGAL path cannot handle still gets triangulated.
class IsotropicFaceMesherFactory : public IMeshTools_MeshAlgoFactory {
 public:
  /// @param targetEdgeLength Target 3D edge length for uniform meshing.
  /// @param gradedInterior Grade interior sampling towards the boundary's density; see
  ///        @ref IsotropicFaceMesher.
  /// @param stats Fallback tally shared by every face; may be null. Must outlive the factory.
  IsotropicFaceMesherFactory(
      double targetEdgeLength,
      bool gradedInterior,
      FaceMeshFallbackStats* stats);

  ~IsotropicFaceMesherFactory() override = default;

  Handle(IMeshTools_MeshAlgo)
      GetAlgo(GeomAbs_SurfaceType surfaceType, IMeshTools_Parameters const& parameters)
          const override;

 private:
  double _targetEdgeLength{};
  bool _gradedInterior{};
  FaceMeshFallbackStats* _stats{};
  // Stateless and const-correct, so it is safe to share across BRepMesh_FaceDiscret's workers.
  Handle(BRepMesh_DelabellaMeshAlgoFactory) _fallbackFactory;
};

} // namespace mochi::mesh::cli::occ

#endif // MOCHI_USE_OCCT
