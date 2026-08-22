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

// Helper-side implementation of mochi::mesh::cli::MeshStepBody -- the Stage-1 parameterized STEP
// tessellation. It loads a STEP file, optionally combines touching solids, normalizes the topology
// via a STEP round-trip, and tessellates it with the isotropic CGAL per-face mesher.
// OpenCascade and CGAL are confined to this helper; the OCCT path is compiled only where OCCT is
// available (MOCHI_USE_OCCT), otherwise the function returns a clean
// error.

#include "mesh_cli_geometry.h"

#if MOCHI_USE_OCCT

#include "occ_shape_meshing.h"

#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBndLib.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepGProp.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <GProp_GProps.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <IMeshData_Status.hxx>
#include <Interface_Static.hxx>
#include <Poly_Triangle.hxx>
#include <Poly_Triangulation.hxx>
#include <STEPControl_Controller.hxx>
#include <STEPControl_Reader.hxx>
#include <STEPControl_StepModelType.hxx>
#include <STEPControl_Writer.hxx>
#include <ShapeFix_Shape.hxx>
#include <Standard_Failure.hxx>
#include <Standard_Integer.hxx>
#include <TopAbs.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Pnt.hxx>
#include <gp_Trsf.hxx>

#if defined(_WIN32)
#include <process.h> // _getpid
#else
#include <unistd.h> // getpid
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using namespace mochi::mesh::cli;

// OpenCascade emits geometry in millimeters (see LoadStepShape); Mochi works in meters.
constexpr double kMillimetersToMeters = 0.001;
// Solids whose closest surface distance is below this [mm] are treated as touching and combined.
constexpr double kCombineProximityTolerance = 1e-3;

// Union-Find with path compression and union by rank.
class DisjointSet {
 public:
  explicit DisjointSet(int n) : _parent(n), _rank(n, 0) {
    std::iota(_parent.begin(), _parent.end(), 0);
  }

  int Find(int x) {
    if (_parent[x] != x) {
      _parent[x] = Find(_parent[x]);
    }
    return _parent[x];
  }

  void Unite(int x, int y) {
    int rx = Find(x);
    int ry = Find(y);
    if (rx == ry) {
      return;
    }
    if (_rank[rx] < _rank[ry]) {
      std::swap(rx, ry);
    }
    _parent[ry] = rx;
    if (_rank[rx] == _rank[ry]) {
      ++_rank[rx];
    }
  }

 private:
  std::vector<int> _parent;
  std::vector<int> _rank;
};

// Repair degenerate geometry after boolean operations.
TopoDS_Shape HealShape(TopoDS_Shape const& shape) {
  ShapeFix_Shape fixer(shape);
  fixer.SetPrecision(1e-6);
  fixer.SetMaxTolerance(1e-3);
  fixer.Perform();
  return fixer.Shape();
}

int CountSubShapes(TopoDS_Shape const& shape, TopAbs_ShapeEnum type) {
  TopTools_IndexedMapOfShape map;
  TopExp::MapShapes(shape, type, map);
  return map.Extent();
}

// Enclosed volume [mm^3]. A fuse that loses volume has swallowed geometry rather than merged it.
double ShapeVolume(TopoDS_Shape const& shape) {
  GProp_GProps props;
  BRepGProp::VolumeProperties(shape, props);
  return props.Mass();
}

// Number of enclosed cavities in a shape: every solid has one outer shell, and one extra shell per
// internal void.
int CountVoids(TopoDS_Shape const& shape) {
  return CountSubShapes(shape, TopAbs_SHELL) - CountSubShapes(shape, TopAbs_SOLID);
}

// Fuses a group of touching solids in a single boolean pass.
//
// Two things matter here. First, one multi-argument fuse instead of a chain of pairwise ones: OCCT
// resolves a whole argument set at once, whereas chaining re-approximates the accumulated result at
// every step and drifts. Second, the fuzzy value -- these bodies meet at faces that are
// geometrically coincident but not topologically shared, and at OCCT's default 1e-7 confusion the
// kernel keeps both copies of a contact face. That leaves sliver faces and trapped voids. Fusing at
// the same tolerance used to decide the solids touch lets the kernel merge those faces.
bool FuseSolids(
    TopoDS_Shape const& a,
    TopoDS_Shape const& b,
    double fuzzyValue,
    TopoDS_Shape& outFused) {
  TopTools_ListOfShape arguments;
  TopTools_ListOfShape tools;
  arguments.Append(a);
  tools.Append(b);

  BRepAlgoAPI_Fuse fuseOp;
  fuseOp.SetArguments(arguments);
  fuseOp.SetTools(tools);
  fuseOp.SetFuzzyValue(fuzzyValue);
  // The originals are reused verbatim when the fuse is rejected, so it must not edit them in place.
  fuseOp.SetNonDestructive(Standard_True);
  fuseOp.Build();

  if (!fuseOp.IsDone() || fuseOp.HasErrors()) {
    return false;
  }
  outFused = fuseOp.Shape();
  return true;
}

// Volume the union of @p a and @p b must have: |A| + |B| - |A and B|.
//
// Computing the intersection explicitly is what makes the volume check meaningful. Bounding the
// union by [max(|A|,|B|), |A|+|B|] instead is far too loose: a fuse can quietly drop a large chunk
// of one operand and still land inside that range, which is exactly how a whole body goes missing.
bool ExpectedUnionVolume(
    TopoDS_Shape const& a,
    TopoDS_Shape const& b,
    double fuzzyValue,
    double& outVolume) {
  TopTools_ListOfShape arguments;
  TopTools_ListOfShape tools;
  arguments.Append(a);
  tools.Append(b);

  BRepAlgoAPI_Common commonOp;
  commonOp.SetArguments(arguments);
  commonOp.SetTools(tools);
  commonOp.SetFuzzyValue(fuzzyValue);
  commonOp.SetNonDestructive(Standard_True);
  commonOp.Build();

  if (!commonOp.IsDone() || commonOp.HasErrors()) {
    return false;
  }
  outVolume = ShapeVolume(a) + ShapeVolume(b) - ShapeVolume(commonOp.Shape());
  return true;
}

// A fuse can report success and still hand back geometry the per-face mesher cannot handle. Three
// signatures matter: a cavity the inputs never had, which means the kernel failed to weld a contact
// interface and left a gap bounded by near-coincident sliver faces; an invalid shape; and a volume
// short of the union the operands must produce, which means geometry was dropped outright.
bool IsFuseAcceptable(TopoDS_Shape const& fused, int inputVoids, double expectedVolume) {
  if (CountSubShapes(fused, TopAbs_SOLID) < 1) {
    MOCHI_MESH_CLI_LOG_WARNING("combine: rejecting fuse -- result has no solid");
    return false;
  }
  int const voids = CountVoids(fused);
  if (voids > inputVoids) {
    MOCHI_MESH_CLI_LOG_WARNING(
        "combine: rejecting fuse -- invented %d enclosed void(s) (inputs had %d)",
        voids - inputVoids,
        inputVoids);
    return false;
  }
  if (!BRepCheck_Analyzer(fused).IsValid()) {
    MOCHI_MESH_CLI_LOG_WARNING("combine: rejecting fuse -- result is not a valid shape");
    return false;
  }
  double constexpr kVolumeTolerance = 1e-3;
  double const volume = ShapeVolume(fused);
  if (std::abs(volume - expectedVolume) > expectedVolume * kVolumeTolerance) {
    MOCHI_MESH_CLI_LOG_WARNING(
        "combine: rejecting fuse -- volume %.6g, expected %.6g (lost %.4g%%)",
        volume,
        expectedVolume,
        100.0 * (expectedVolume - volume) / expectedVolume);
    return false;
  }
  return true;
}

// Welds @p b into @p a, accepting the result only if it is usable.
bool TryFuse(
    TopoDS_Shape const& a,
    TopoDS_Shape const& b,
    double fuzzyValue,
    TopoDS_Shape& outFused) {
  double expectedVolume = 0.0;
  if (!ExpectedUnionVolume(a, b, fuzzyValue, expectedVolume)) {
    MOCHI_MESH_CLI_LOG_WARNING(
        "combine: rejecting fuse -- cannot compute the operand intersection");
    return false;
  }
  return FuseSolids(a, b, fuzzyValue, outFused) &&
      IsFuseAcceptable(outFused, CountVoids(a) + CountVoids(b), expectedVolume);
}

// Fuses as much of a group as the kernel can manage, returning the resulting bodies.
//
// Solids are welded into an accumulator one at a time, and every step is checked against the volume
// its operands must produce. A solid whose addition is rejected is set aside and retried against
// the larger accumulator on a later pass, since a weld that fails early can succeed once
// neighbouring geometry is present. Whatever never fuses is returned as its own body -- one
// uncooperative solid does not force the whole group apart, and no step can silently destroy
// geometry.
std::vector<TopoDS_Shape> FuseGroup(std::vector<TopoDS_Shape> const& solids, double fuzzyValue) {
  TopoDS_Shape accumulator = solids.front();
  std::vector<TopoDS_Shape> pending(solids.begin() + 1, solids.end());
  bool fusedAny = false;

  for (bool progressed = true; progressed && !pending.empty();) {
    progressed = false;
    std::vector<TopoDS_Shape> stillPending;
    for (TopoDS_Shape const& candidate : pending) {
      TopoDS_Shape fused;
      if (TryFuse(accumulator, candidate, fuzzyValue, fused)) {
        accumulator = fused;
        fusedAny = true;
        progressed = true;
      } else {
        stillPending.push_back(candidate);
      }
    }
    pending = std::move(stillPending);
  }

  // Only worth reporting when the group did not weld cleanly: a leftover means the model still
  // contains separate bodies where one was intended.
  if (!pending.empty()) {
    MOCHI_MESH_CLI_LOG_WARNING(
        "combine: welded %d of %d touching solids, %d kept separate",
        static_cast<int>(solids.size() - pending.size()),
        static_cast<int>(solids.size()),
        static_cast<int>(pending.size()));
  }

  std::vector<TopoDS_Shape> bodies;
  bodies.reserve(pending.size() + 1);
  bodies.push_back(fusedAny ? HealShape(accumulator) : accumulator);
  bodies.insert(bodies.end(), pending.begin(), pending.end());
  return bodies;
}

// Reads a STEP file into a single combined OCCT shape. Pins OpenCascade's output unit to
// millimeters (its default) so the imported geometry is normalized to mm regardless of the unit the
// STEP file declares; MeshStepBody converts mm -> meters at extraction.
bool LoadStepShape(std::string const& path, TopoDS_Shape& outShape, CliError& error) {
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

// Fuses groups of touching solids into single bodies. Adjacency is detected first by shared faces
// (topologically coincident), then by geometric proximity for solids that touch without sharing
// topology. A group that cannot be fused as a whole is fused as far as it can be, and the solids
// that resist are kept as separate bodies.
TopoDS_Shape CombineSolids(TopoDS_Shape const& shape, double proximityTolerance) {
  TopTools_IndexedMapOfShape solidMap;
  TopExp::MapShapes(shape, TopAbs_SOLID, solidMap);

  int const numSolids = solidMap.Extent();
  if (numSolids <= 1) {
    return shape;
  }

  DisjointSet dset(numSolids);

  // Primary adjacency: faces shared between multiple solids.
  TopTools_IndexedDataMapOfShapeListOfShape faceToSolids;
  TopExp::MapShapesAndAncestors(shape, TopAbs_FACE, TopAbs_SOLID, faceToSolids);

  for (int i = 1; i <= faceToSolids.Extent(); ++i) {
    TopTools_ListOfShape const& solids = faceToSolids(i);
    if (solids.Extent() < 2) {
      continue;
    }
    std::vector<int> solidIndices;
    for (auto it = solids.cbegin(); it != solids.cend(); ++it) {
      int const idx = solidMap.FindIndex(*it);
      if (idx > 0) {
        solidIndices.push_back(idx - 1); // 0-based
      }
    }
    for (size_t j = 1; j < solidIndices.size(); ++j) {
      dset.Unite(solidIndices[0], solidIndices[j]);
    }
  }

  // Fallback adjacency: geometric proximity for solid pairs not already grouped.
  for (int i = 0; i < numSolids; ++i) {
    for (int j = i + 1; j < numSolids; ++j) {
      if (dset.Find(i) == dset.Find(j)) {
        continue;
      }
      BRepExtrema_DistShapeShape distCalc(solidMap(i + 1), solidMap(j + 1)); // 1-based
      if (distCalc.IsDone() && distCalc.Value() < proximityTolerance) {
        dset.Unite(i, j);
      }
    }
  }

  std::unordered_map<int, std::vector<int>> groups;
  for (int i = 0; i < numSolids; ++i) {
    groups[dset.Find(i)].push_back(i);
  }

  BRep_Builder builder;
  TopoDS_Compound result;
  builder.MakeCompound(result);

  for (auto const& [root, members] : groups) {
    if (members.size() == 1) {
      builder.Add(result, solidMap(members[0] + 1));
      continue;
    }

    std::vector<TopoDS_Shape> groupSolids;
    groupSolids.reserve(members.size());
    for (int const idx : members) {
      groupSolids.push_back(solidMap(idx + 1));
    }

    for (TopoDS_Shape const& body : FuseGroup(groupSolids, proximityTolerance)) {
      builder.Add(result, body);
    }
  }

  // Strip residual triangulation left by the boolean kernel so meshing starts fresh; without this
  // BRepMesh may skip re-meshing faces that carry a stale triangulation.
  BRepTools::Clean(result);
  return result;
}

// Normalizes topology via a STEP write/read round-trip. The reader rebuilds all pcurves by
// projecting 3D curves onto surfaces, recomputes tolerances, and fixes wire orientations -- which
// the per-face mesher relies on. On any failure the original shape is returned unchanged.
TopoDS_Shape NormalizeShape(TopoDS_Shape const& shape) {
#if defined(_WIN32)
  int const pid = _getpid();
#else
  int const pid = getpid();
#endif
  std::filesystem::path const tmpPath = std::filesystem::temp_directory_path() /
      ("superdex_normalize_" + std::to_string(pid) + ".stp");
  std::string const tmp = tmpPath.string();

  STEPControl_Writer writer;
  writer.Transfer(shape, STEPControl_AsIs);
  if (writer.Write(tmp.c_str()) != IFSelect_RetDone) {
    std::cerr << "[step_mesh_body] Warning: normalize STEP write failed; using original shape\n";
    return shape;
  }

  STEPControl_Reader reader;
  if (reader.ReadFile(tmp.c_str()) != IFSelect_RetDone) {
    std::cerr << "[step_mesh_body] Warning: normalize STEP read failed; using original shape\n";
    std::error_code ec;
    std::filesystem::remove(tmpPath, ec);
    return shape;
  }
  reader.TransferRoots();
  TopoDS_Shape result = reader.OneShape();

  std::error_code ec;
  std::filesystem::remove(tmpPath, ec);

  if (result.IsNull()) {
    std::cerr << "[step_mesh_body] Warning: normalize round-trip produced a null shape\n";
    return shape;
  }

  BRepTools::Clean(result);
  return result;
}

// Target uniform 3D edge length [mm] for the shape, from an explicit value or its bounding box.
double ComputeTargetEdgeLength(TopoDS_Shape const& shape, StepMeshBodyParams const& params) {
  Bnd_Box bbox;
  BRepBndLib::Add(shape, bbox);
  return occ::ResolveTargetEdgeLength(
      bbox, params.targetEdgeLength, params.targetEdgeLengthFraction);
}

// Runs the isotropic meshing pipeline in place on the shape: the CGAL per-face mesher and the
// uniform edge discretizer. Faces the CGAL mesher cannot handle are retried with OCCT's Delabella
// mesher and recorded in @p fallbackStats. Returns the OCCT mesh status flags.
IMeshData_Status RunIsotropicMeshing(
    TopoDS_Shape const& shape,
    StepMeshBodyParams const& params,
    double targetEdgeLength,
    occ::FaceMeshFallbackStats& fallbackStats) {
  occ::ShapeMeshingParams meshingParams;
  meshingParams.faceMesher = occ::FaceMesher::Isotropic;
  meshingParams.linearDeflection = params.linearDeflection;
  meshingParams.angularDeflection = params.angularDeflection;
  meshingParams.targetEdgeLength = targetEdgeLength;
  meshingParams.edgeSampling = params.edgeSampling;
  return occ::MeshShape(shape, meshingParams, &fallbackStats);
}

// Extracts every face's triangulation into a single unwelded triangle soup in meters. Shared-edge
// vertices are intentionally left unwelded (crisp CAD boundaries); REVERSED faces get their winding
// flipped so the whole mesh is consistently outward-facing.
MeshData ExtractMesh(TopoDS_Shape const& shape) {
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
    bool const reversed = (face.Orientation() == TopAbs_REVERSED);

    int const baseIndex = GetNumNodes(mesh);
    for (Standard_Integer i = 1; i <= triangulation->NbNodes(); ++i) {
      gp_Pnt const p = triangulation->Node(i).Transformed(transform);
      // Convert OCCT's native Z-up frame to the Y-up frame the glTF/render pipeline expects
      // (matching cad_conversion's RWGltf Zup->Yup): (x, y, z) -> (x, z, -y), i.e. -90 deg about X.
      // This is a proper rotation, so it does not affect the triangle winding handled below.
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
  return mesh;
}

} // namespace

MeshData mochi::mesh::cli::MeshStepBody(
    std::string_view stepFilePath,
    StepMeshBodyParams const& params,
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

  // OpenCascade reports failures by raising Standard_Failure; catch it so a meshing failure becomes
  // a clean error rather than crashing the helper (which would corrupt the framed response).
  try {
    if (params.combineTouchingSolids) {
      shape = CombineSolids(shape, kCombineProximityTolerance);
    }
    shape = NormalizeShape(shape);

    double const targetEdgeLength = ComputeTargetEdgeLength(shape, params);
    occ::FaceMeshFallbackStats fallbackStats;
    IMeshData_Status const status =
        RunIsotropicMeshing(shape, params, targetEdgeLength, fallbackStats);

    if (fallbackStats.FailedCount() > 0) {
      MOCHI_MESH_CLI_LOG_WARNING(
          "%d face(s) failed to mesh and %d were recovered with the fallback mesher",
          fallbackStats.FailedCount(),
          fallbackStats.RescuedCount());
    }

    if ((status & IMeshData_Failure) && !params.allowPartialFailure) {
      MOCHI_MESH_CLI_ERROR_SET(
          error,
          "STEP tessellation failed on one or more faces (enable partial failure to keep a partial mesh).");
      return {};
    }

    MeshData mesh = ExtractMesh(shape);
    MOCHI_MESH_CLI_ERROR_IF(
        GetNumElements(mesh) == 0, error, "STEP tessellation produced no triangles.");
    MOCHI_MESH_CLI_ERROR_RETURN(error, {});
    return mesh;
  } catch (Standard_Failure const& failure) {
    char const* const message = failure.GetMessageString();
    std::fprintf(
        stderr, "[step_mesh_body] OpenCascade failure: %s\n", message ? message : "(none)");
    MOCHI_MESH_CLI_ERROR_SET(error, "OpenCascade failed during STEP body tessellation.");
    return {};
  } catch (...) {
    MOCHI_MESH_CLI_ERROR_SET(error, "OpenCascade threw during STEP body tessellation.");
    return {};
  }
}

#else // !MOCHI_USE_OCCT

mochi::mesh::cli::MeshData mochi::mesh::cli::MeshStepBody(
    std::string_view /*stepFilePath*/,
    StepMeshBodyParams const& /*params*/,
    CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, {});
  MOCHI_MESH_CLI_ERROR_SET(
      error, "STEP tessellation is not supported on this platform (OpenCascade unavailable).");
  return {};
}

#endif // MOCHI_USE_OCCT
