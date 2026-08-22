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

#include "cgal_mesh_utils.h"
#include "mesh_cli_geometry.h"

#include <mochi_mesh/mochi_mesh_cli_encoding.h>

#include <CGAL/Polygon_mesh_processing/distance.h>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#if MOCHI_MESH_CLI_PLATFORM_WINDOWS
#include <fcntl.h>
#include <io.h>
#endif

using namespace mochi::mesh::cli;

namespace {

std::vector<char> ReadAllStdin() {
  std::vector<char> input;
  char buffer[4096];
  size_t bytesRead = 0;
  while ((bytesRead = std::fread(buffer, 1, sizeof(buffer), stdin)) > 0) {
    input.insert(input.end(), buffer, buffer + bytesRead);
  }
  return input;
}

std::vector<char> MakeErrorResponse(uint32_t status, std::string const& message) {
  return EncodeResponseFrame(status, message);
}

std::vector<char> MakeErrorResponse(CliError const& error) {
  return EncodeResponseFrame(2, error.ToString());
}

std::vector<char> MakeMeshResponse(MeshData const& mesh) {
  PayloadWriter writer;
  writer.WriteMeshData(mesh);
  return EncodeResponseFrame(0, writer.Bytes());
}

std::vector<char> DispatchRemeshSurface(std::vector<char> const& payload) {
  PayloadReader reader(payload);
  SurfaceRemeshingParams params;
  MeshData inputMesh;
  if (!reader.ReadSurfaceRemeshingParams(params) || !reader.ReadMeshData(inputMesh) ||
      !reader.AtEnd()) {
    return MakeErrorResponse(1, "superdex_mesh_cli: malformed RemeshSurface request");
  }

  CliError error;
  MeshData const result = RemeshSurface(inputMesh, params, error);
  if (!error.IsOK()) {
    return MakeErrorResponse(error);
  }
  return MakeMeshResponse(result);
}

std::vector<char> DispatchReconstructSurfaceFromSdf(std::vector<char> const& payload) {
  PayloadReader reader(payload);
  ScalarField3d sdf;
  if (!reader.ReadScalarField3d(sdf) || !reader.AtEnd()) {
    return MakeErrorResponse(1, "superdex_mesh_cli: malformed ReconstructSurfaceFromSdf request");
  }

  CliError error;
  MeshData const result = ReconstructSurfaceFromSdf(sdf, error);
  if (!error.IsOK()) {
    return MakeErrorResponse(error);
  }
  return MakeMeshResponse(result);
}

double
ApproximateHausdorffDistance(MeshData const& mesh, MeshData const& referenceMesh, CliError& error) {
  MOCHI_MESH_CLI_ERROR_RETURN(error, -1.0);
  using SurfaceMesh = cgal_utils::CgalSurfaceMesh;
  try {
    SurfaceMesh cgalMesh = cgal_utils::MeshDataToSurfaceMesh(mesh, error);
    MOCHI_MESH_CLI_ERROR_RETURN(error, -1.0);
    SurfaceMesh cgalReferenceMesh = cgal_utils::MeshDataToSurfaceMesh(referenceMesh, error);
    MOCHI_MESH_CLI_ERROR_RETURN(error, -1.0);
    // Preserve the exact original call: same argument order/direction and Sequential_tag.
    return CGAL::Polygon_mesh_processing::approximate_Hausdorff_distance<CGAL::Sequential_tag>(
        cgalMesh, cgalReferenceMesh);
  } catch (...) {
    MOCHI_MESH_CLI_ERROR_SET(
        error, "CGAL Hausdorff distance computation failed with an exception.");
    return -1.0;
  }
}

std::vector<char> DispatchApproximateHausdorffDistance(std::vector<char> const& payload) {
  PayloadReader reader(payload);
  MeshData mesh;
  MeshData referenceMesh;
  if (!reader.ReadMeshData(mesh) || !reader.ReadMeshData(referenceMesh) || !reader.AtEnd()) {
    return MakeErrorResponse(
        1, "superdex_mesh_cli: malformed ApproximateHausdorffDistance request");
  }

  CliError error;
  double const distance = ApproximateHausdorffDistance(mesh, referenceMesh, error);
  if (!error.IsOK()) {
    return MakeErrorResponse(error);
  }
  PayloadWriter writer;
  writer.WriteDouble(distance);
  return EncodeResponseFrame(0, writer.Bytes());
}

std::vector<char> DispatchTessellateStep(std::vector<char> const& payload) {
  PayloadReader reader(payload);
  std::vector<char> pathBytes;
  StepTessellationParams params;
  if (!reader.ReadByteArray(pathBytes) || !reader.ReadStepTessellationParams(params) ||
      !reader.AtEnd()) {
    return MakeErrorResponse(1, "superdex_mesh_cli: malformed TessellateStep request");
  }

  CliError error;
  MeshData const result =
      TessellateStep(std::string_view(pathBytes.data(), pathBytes.size()), params, error);
  if (!error.IsOK()) {
    return MakeErrorResponse(error);
  }
  return MakeMeshResponse(result);
}

std::vector<char> DispatchMeshStepBody(std::vector<char> const& payload) {
  PayloadReader reader(payload);
  std::vector<char> pathBytes;
  StepMeshBodyParams params;
  if (!reader.ReadByteArray(pathBytes) || !reader.ReadStepMeshBodyParams(params) ||
      !reader.AtEnd()) {
    return MakeErrorResponse(1, "superdex_mesh_cli: malformed MeshStepBody request");
  }

  CliError error;
  MeshData const result =
      MeshStepBody(std::string_view(pathBytes.data(), pathBytes.size()), params, error);
  if (!error.IsOK()) {
    return MakeErrorResponse(error);
  }
  return MakeMeshResponse(result);
}

std::vector<char> DispatchCleanupMesh(std::vector<char> const& payload) {
  PayloadReader reader(payload);
  MeshData inputMesh;
  if (!reader.ReadMeshData(inputMesh) || !reader.AtEnd()) {
    return MakeErrorResponse(1, "superdex_mesh_cli: malformed CleanupMesh request");
  }
  CliError error;
  MeshData const result = CleanupMesh(inputMesh, error);
  if (!error.IsOK()) {
    return MakeErrorResponse(error);
  }
  return MakeMeshResponse(result);
}

std::vector<char> DispatchCloseMesh(std::vector<char> const& payload) {
  PayloadReader reader(payload);
  MeshData inputMesh;
  MeshClosureParams params;
  if (!reader.ReadMeshData(inputMesh) || !reader.ReadMeshClosureParams(params) || !reader.AtEnd()) {
    return MakeErrorResponse(1, "superdex_mesh_cli: malformed CloseMesh request");
  }

  CliError error;
  MeshData const result = CloseMesh(inputMesh, params, error);
  if (!error.IsOK()) {
    return MakeErrorResponse(error);
  }
  return MakeMeshResponse(result);
}

std::vector<char> DispatchEdgeSwapMesh(std::vector<char> const& payload) {
  PayloadReader reader(payload);
  MeshData inputMesh;
  MeshData referenceMesh;
  MeshEdgeSwapParams params;
  if (!reader.ReadMeshData(inputMesh) || !reader.ReadMeshData(referenceMesh) ||
      !reader.ReadMeshEdgeSwapParams(params) || !reader.AtEnd()) {
    return MakeErrorResponse(1, "superdex_mesh_cli: malformed EdgeSwapMesh request");
  }

  CliError error;
  MeshData const result = EdgeSwapMesh(inputMesh, referenceMesh, params, error);
  if (!error.IsOK()) {
    return MakeErrorResponse(error);
  }
  return MakeMeshResponse(result);
}

std::vector<char> DispatchDecimateMesh(std::vector<char> const& payload) {
  PayloadReader reader(payload);
  MeshData inputMesh;
  MeshDecimateParams params;
  if (!reader.ReadMeshData(inputMesh) || !reader.ReadMeshDecimateParams(params) ||
      !reader.AtEnd()) {
    return MakeErrorResponse(1, "superdex_mesh_cli: malformed DecimateMesh request");
  }

  CliError error;
  MeshData const result = DecimateMesh(inputMesh, params, error);
  if (!error.IsOK()) {
    return MakeErrorResponse(error);
  }
  return MakeMeshResponse(result);
}

std::vector<char> DispatchExportStepVisual(std::vector<char> const& payload) {
  PayloadReader reader(payload);
  std::vector<char> stepPathBytes;
  StepVisualExportParams params;
  std::vector<VisualExportOutput> outputs;
  if (!reader.ReadByteArray(stepPathBytes) || !reader.ReadStepVisualExportParams(params) ||
      !reader.ReadVisualExportOutputs(outputs) || !reader.AtEnd()) {
    return MakeErrorResponse(1, "superdex_mesh_cli: malformed ExportStepVisual request");
  }

  CliError error;
  std::vector<VisualExportStatus> statuses;
  if (!ExportStepVisual(
          std::string_view(stepPathBytes.data(), stepPathBytes.size()),
          outputs,
          params,
          statuses,
          error)) {
    return MakeErrorResponse(error);
  }
  PayloadWriter writer;
  writer.WriteVisualExportStatuses(statuses);
  return EncodeResponseFrame(0, writer.Bytes());
}

std::vector<char> Dispatch(GeometryOp op, std::vector<char> const& payload) {
  switch (op) {
    case GeometryOp::Ping:
      return EncodeResponseFrame(0, payload);
    case GeometryOp::RemeshSurface:
      return DispatchRemeshSurface(payload);
    case GeometryOp::ReconstructSurfaceFromSdf:
      return DispatchReconstructSurfaceFromSdf(payload);
    case GeometryOp::ApproximateHausdorffDistance:
      return DispatchApproximateHausdorffDistance(payload);
    case GeometryOp::TessellateStep:
      return DispatchTessellateStep(payload);
    case GeometryOp::MeshStepBody:
      return DispatchMeshStepBody(payload);
    case GeometryOp::CleanupMesh:
      return DispatchCleanupMesh(payload);
    case GeometryOp::CloseMesh:
      return DispatchCloseMesh(payload);
    case GeometryOp::EdgeSwapMesh:
      return DispatchEdgeSwapMesh(payload);
    case GeometryOp::DecimateMesh:
      return DispatchDecimateMesh(payload);
    case GeometryOp::ExportStepVisual:
      return DispatchExportStepVisual(payload);
  }
  return MakeErrorResponse(1, "superdex_mesh_cli: unknown operation");
}

} // namespace

int main() {
  // stdout is the binary response channel. Libraries we call (notably OpenCascade's STEP reader)
  // write diagnostics to std::cout, which would corrupt the framed response. Redirect the C++
  // std::cout stream to stderr; the response is written via C stdio (std::fwrite on stdout), which
  // is unaffected by this redirect.
  std::cout.rdbuf(std::cerr.rdbuf());

#if MOCHI_MESH_CLI_PLATFORM_WINDOWS
  // Use binary mode so the framed protocol is not mangled by CR/LF translation.
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
#endif

  std::vector<char> const input = ReadAllStdin();

  GeometryOp op = GeometryOp::Ping;
  std::vector<char> payload;
  std::vector<char> response;
  if (!DecodeRequestFrame(input, op, payload)) {
    response = MakeErrorResponse(1, "superdex_mesh_cli: malformed request frame");
  } else {
    response = Dispatch(op, payload);
  }

  if (!response.empty()) {
    std::fwrite(response.data(), 1, response.size(), stdout);
  }
  std::fflush(stdout);
  return 0;
}
