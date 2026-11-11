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

#include <mochi_mesh/tetrahedralization.h>
#include "mesh_cli_adapter.h"
#include "mesh_cli_client.h"

// Parameter and surface validity checks live in the superdex_mesh_cli helper, so an invalid request
// round-trips a clean error rather than failing here.
using namespace mochi;
using namespace mochi::mesh;

MeshData mochi::mesh::Tetrahedralize(
    MeshDataView const& surfaceMesh,
    TetrahedralizationParams const& params,
    Error& error) {
  MOCHI_ERROR_IF(
      surfaceMesh.nodesPerElement != 3,
      error,
      "Input mesh must be a triangle mesh (nodesPerElement == 3).");
  MOCHI_ERROR_RETURN(error, {});

  cli::PayloadWriter writer;
  writer.WriteTetrahedralizationParams(params);
  cli::MeshData const cliMesh = cli_adapter::ToCliMeshData(surfaceMesh, error);
  MOCHI_ERROR_RETURN(error, {});
  writer.WriteMeshData(cliMesh);
  MeshData result = InvokeMeshOp(cli::GeometryOp::Tetrahedralize, writer, error);
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(
      result.nodesPerElement != 4, error, "Tetrahedralization did not return a tetrahedral mesh.");
  MOCHI_ERROR_RETURN(error, {});
  return result;
}

MeshData mochi::mesh::TetrahedralizeConstrained(
    MeshDataView const& surfaceMesh,
    ConstrainedTetrahedralizationParams const& params,
    Error& error) {
  MOCHI_ERROR_IF(
      surfaceMesh.nodesPerElement != 3,
      error,
      "Input mesh must be a triangle mesh (nodesPerElement == 3).");
  MOCHI_ERROR_RETURN(error, {});

  cli::PayloadWriter writer;
  writer.WriteConstrainedTetrahedralizationParams(params);
  cli::MeshData const cliMesh = cli_adapter::ToCliMeshData(surfaceMesh, error);
  MOCHI_ERROR_RETURN(error, {});
  writer.WriteMeshData(cliMesh);
  MeshData result = InvokeMeshOp(cli::GeometryOp::TetrahedralizeConstrained, writer, error);
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(
      result.nodesPerElement != 4, error, "Tetrahedralization did not return a tetrahedral mesh.");
  MOCHI_ERROR_RETURN(error, {});
  return result;
}
