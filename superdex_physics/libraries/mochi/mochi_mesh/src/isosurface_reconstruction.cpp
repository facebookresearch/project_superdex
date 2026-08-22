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

#include <mochi_mesh/isosurface_reconstruction.h>

#include "mesh_cli_adapter.h"
#include "mesh_cli_client.h"

// ReconstructSurfaceFromSdf converts the SDF grid to the neutral wire type, serializes it, and
// routes it to the superdex_mesh_cli helper, which runs the isosurface reconstruction. Grid
// conversion and validation live in the adapter; the wire (de)serialization lives in the encoding
// files.

using namespace mochi;
using namespace mochi::mesh;

MeshData mochi::mesh::ReconstructSurfaceFromSdf(GridSdfDataView const& sdfData, Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  cli::ScalarField3d const field = cli_adapter::ToCliScalarField(sdfData, error);
  MOCHI_ERROR_RETURN(error, {});

  cli::PayloadWriter writer;
  writer.WriteScalarField3d(field);
  return InvokeMeshOp(cli::GeometryOp::ReconstructSurfaceFromSdf, writer, error);
}
