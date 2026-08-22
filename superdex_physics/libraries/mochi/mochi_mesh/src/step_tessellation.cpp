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

#include <mochi_mesh/step_tessellation.h>

#include "mesh_cli_client.h"

#include <string_view>

// TessellateStep marshals the request and routes it to the superdex_mesh_cli helper process, which
// runs the OpenCascade tessellation. Input validation lives in the helper, so an invalid request
// round-trips a clean error rather than failing here.

using namespace mochi;
using namespace mochi::mesh;

MeshData mochi::mesh::TessellateStep(
    std::string_view stepFilePath,
    StepTessellationParams const& params,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  cli::PayloadWriter writer;
  writer.WriteByteArray(stepFilePath);
  writer.WriteStepTessellationParams(params);
  return InvokeMeshOp(cli::GeometryOp::TessellateStep, writer, error);
}
