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

// Helper-side implementation of mochi::mesh::cli::TessellateStep. This is the only place
// OpenCascade (OCCT) is used; it is confined to the superdex_mesh_cli helper executable so the
// shipping libraries never link OCCT. The OCCT path is compiled only on platforms where OCCT is
// available (MOCHI_USE_OCCT); elsewhere the function returns a clean "unsupported"
// error.

#include "mesh_cli_geometry.h"

#if MOCHI_USE_OCCT

#include <BRepMesh_Context.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRep_Tool.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <IMeshTools_MeshAlgoType.hxx>
#include <IMeshTools_Parameters.hxx>
#include <Interface_Static.hxx>
#include <Poly_Triangle.hxx>
#include <Poly_Triangulation.hxx>
#include <STEPControl_Controller.hxx>
#include <STEPControl_Reader.hxx>
#include <Standard_Failure.hxx>
#include <Standard_Integer.hxx>
#include <TopAbs.hxx>
#include <TopExp.hxx>
#include <TopLoc_Location.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Pnt.hxx>
#include <gp_Trsf.hxx>

#include <cstdio>
#include <string>
#include <utility>

namespace {

using namespace mochi::mesh::cli;

// Reads a STEP file into a single combined OCCT shape. Returns false (and sets @p error) on any
// read or transfer failure.
bool LoadStepShape(std::string const& path, TopoDS_Shape& outShape, CliError& error) {
  // Pin OCCT's output unit to millimeters (its default), which normalizes the imported geometry to
  // mm regardless of the unit the STEP file declares. TessellateStep then converts mm -> meters
  // explicitly (see kMillimetersToMeters) so the result matches Mochi's meter convention and the
  // .obj/.glb meshes generated from the same CAD parts. (The xstep.cascade.unit="M" target-unit
  // path does not actually rescale here, so we do the conversion ourselves.)
  // Statics are unregistered, and silently ignored, until the controller is initialized.
  STEPControl_Controller::Init();
  Interface_Static::SetCVal("xstep.cascade.unit", "MM");
  // Fix for SolidWorks' single-part export with a non-identity coordinate transform. This flag is
  // benign in most cases.
  Interface_Static::SetCVal("read.step.root.transformation", "OFF");

  STEPControl_Reader reader;
  IFSelect_ReturnStatus status = IFSelect_RetFail;
  try {
    status = reader.ReadFile(path.c_str());
  } catch (...) {
    MOCHI_MESH_CLI_ERROR_SET(error, "OpenCascade threw while reading the STEP file.");
    return false;
  }
  MOCHI_MESH_CLI_ERROR_IF(status != IFSelect_RetDone, error, "Failed to read STEP file.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, false);

  if (reader.TransferRoots() == 0) {
    MOCHI_MESH_CLI_ERROR_SET(error, "STEP file contains no transferable roots.");
    return false;
  }
  outShape = reader.OneShape();
  MOCHI_MESH_CLI_ERROR_IF(outShape.IsNull(), error, "STEP file produced an empty shape.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, false);
  return true;
}

} // namespace

MeshData mochi::mesh::cli::TessellateStep(
    std::string_view stepFilePath,
    StepTessellationParams const& params,
    CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});

  MOCHI_MESH_CLI_ERROR_IF(stepFilePath.empty(), error, "STEP file path is empty.");
  MOCHI_MESH_CLI_ERROR_IF(
      !(params.linearDeflection > 0.0) || !(params.angularDeflection > 0.0),
      error,
      "STEP tessellation deflections must be positive.");
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});

  TopoDS_Shape shape;
  if (!LoadStepShape(std::string(stepFilePath), shape, error)) {
    return {};
  }

  // OCCT emits the geometry in millimeters (see LoadStepShape); Mochi works in meters.
  constexpr double kMillimetersToMeters = 0.001;

  // OpenCascade reports failures by raising Standard_Failure; catch it so a meshing failure becomes
  // a clean error rather than crashing the helper (which would corrupt the framed response).
  try {
    // Tessellate every face of the shape in place. This is the visual-preview tessellation; request
    // OpenCascade's Delabella triangulation algorithm explicitly. Delabella is OCCT's newer
    // Delaunay mesher and is NOT the default (the historic default is Watson), so it must be
    // selected via the meshing context.
    IMeshTools_Parameters meshParams;
    meshParams.Deflection = params.linearDeflection;
    meshParams.Angle = params.angularDeflection;
    meshParams.Relative = false;
    meshParams.InParallel = true;
    Handle(BRepMesh_Context) const context =
        new BRepMesh_Context(IMeshTools_MeshAlgoType_Delabella);
    BRepMesh_IncrementalMesh mesher;
    mesher.SetShape(shape);
    mesher.ChangeParameters() = meshParams;
    mesher.Perform(context);

    // Collect unique faces (a compound may reference the same face more than once).
    TopTools_IndexedMapOfShape faceMap;
    TopExp::MapShapes(shape, TopAbs_FACE, faceMap);

    MeshData mesh;
    mesh.nodesPerElement = 3;
    for (Standard_Integer fi = 1; fi <= faceMap.Extent(); ++fi) {
      TopoDS_Face const face = TopoDS::Face(faceMap(fi));
      TopLoc_Location location;
      Handle(Poly_Triangulation) const triangulation = BRep_Tool::Triangulation(face, location);
      if (triangulation.IsNull() || triangulation->NbTriangles() == 0) {
        continue;
      }
      gp_Trsf const transform = location.Transformation();
      // A REVERSED face's natural normal is flipped relative to the solid's outward normal, so its
      // triangle winding must be reversed to keep a consistent outward orientation.
      bool const reversed = (face.Orientation() == TopAbs_REVERSED);

      // Each face's triangulation has its own node set; append this face's nodes (in parent space)
      // and rebase its 1-based triangle indices into the combined vertex list. Vertices on shared
      // edges are intentionally left unwelded, which yields crisp face boundaries for CAD geometry.
      int const baseIndex = GetNumNodes(mesh);
      for (Standard_Integer i = 1; i <= triangulation->NbNodes(); ++i) {
        gp_Pnt const p = triangulation->Node(i).Transformed(transform);
        // Convert OCCT's native Z-up frame to the Y-up frame the glTF/render pipeline expects
        // (matching cad_conversion's RWGltf Zup->Yup): (x, y, z) -> (x, z, -y), i.e. -90 deg about
        // X. This is a proper rotation, so it does not affect the triangle winding handled below.
        mesh.coordinates.push_back(static_cast<double>(p.X() * kMillimetersToMeters));
        mesh.coordinates.push_back(static_cast<double>(p.Z() * kMillimetersToMeters));
        mesh.coordinates.push_back(static_cast<double>(-p.Y() * kMillimetersToMeters));
      }
      for (Standard_Integer i = 1; i <= triangulation->NbTriangles(); ++i) {
        Standard_Integer n1 = 0;
        Standard_Integer n2 = 0;
        Standard_Integer n3 = 0;
        triangulation->Triangle(i).Get(n1, n2, n3);
        int i1 = baseIndex + (n1 - 1);
        int i2 = baseIndex + (n2 - 1);
        int i3 = baseIndex + (n3 - 1);
        if (reversed) {
          std::swap(i1, i3);
        }
        mesh.connectivity.push_back(i1);
        mesh.connectivity.push_back(i2);
        mesh.connectivity.push_back(i3);
      }
    }

    MOCHI_MESH_CLI_ERROR_IF(
        GetNumElements(mesh) == 0, error, "STEP tessellation produced no triangles.");
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});
    return mesh;
  } catch (Standard_Failure const& failure) {
    char const* const message = failure.GetMessageString();
    std::fprintf(
        stderr, "[superdex_mesh_cli] OpenCascade failure: %s\n", message ? message : "(none)");
    MOCHI_MESH_CLI_ERROR_SET(error, "OpenCascade failed during STEP tessellation.");
    return {};
  } catch (...) {
    MOCHI_MESH_CLI_ERROR_SET(error, "OpenCascade threw during STEP tessellation.");
    return {};
  }
}

#else // !MOCHI_USE_OCCT

mochi::mesh::cli::MeshData mochi::mesh::cli::TessellateStep(
    std::string_view /*stepFilePath*/,
    StepTessellationParams const& /*params*/,
    CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});
  MOCHI_MESH_CLI_ERROR_SET(
      error, "STEP tessellation is not supported on this platform (OpenCascade unavailable).");
  return {};
}

#endif // MOCHI_USE_OCCT
