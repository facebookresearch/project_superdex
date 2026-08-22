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

#include <mochi_mesh/step_mesh_stages.h>

#include "mesh_cli_adapter.h"
#include "mesh_cli_client.h"

// Client side of the Stage 2-5 mesh ops: each marshals its request and routes it to the
// superdex_mesh_cli helper, which runs the CGAL implementation.

using namespace mochi;
using namespace mochi::mesh;

namespace {

// Converts a mesh to the neutral wire type and serializes it into @p writer.
void WriteMesh(MeshDataView const& mesh, cli::PayloadWriter& writer, Error& error) {
  cli::MeshData const cliMesh = cli_adapter::ToCliMeshData(mesh, error);
  MOCHI_ERROR_RETURN(error);
  writer.WriteMeshData(cliMesh);
}

} // namespace

MeshData mochi::mesh::CleanupMesh(MeshDataView const& mesh, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  cli::PayloadWriter writer;
  WriteMesh(mesh, writer, error);
  return InvokeMeshOp(cli::GeometryOp::CleanupMesh, writer, error);
}

MeshData
mochi::mesh::CloseMesh(MeshDataView const& mesh, MeshClosureParams const& params, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  cli::PayloadWriter writer;
  WriteMesh(mesh, writer, error);
  writer.WriteMeshClosureParams(params);
  return InvokeMeshOp(cli::GeometryOp::CloseMesh, writer, error);
}

MeshData mochi::mesh::EdgeSwapMesh(
    MeshDataView const& mesh,
    MeshDataView const& reference,
    MeshEdgeSwapParams const& params,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  cli::PayloadWriter writer;
  WriteMesh(mesh, writer, error);
  WriteMesh(reference, writer, error);
  writer.WriteMeshEdgeSwapParams(params);
  return InvokeMeshOp(cli::GeometryOp::EdgeSwapMesh, writer, error);
}

MeshData mochi::mesh::DecimateMesh(
    MeshDataView const& mesh,
    MeshDecimateParams const& params,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  cli::PayloadWriter writer;
  WriteMesh(mesh, writer, error);
  writer.WriteMeshDecimateParams(params);
  return InvokeMeshOp(cli::GeometryOp::DecimateMesh, writer, error);
}
