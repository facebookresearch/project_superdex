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

#include <mochi_mesh/step_visual_export.h>

#include "mesh_cli_client.h"

#include <mochi_mesh/mochi_mesh_cli_encoding.h>

#include <span>
#include <string_view>
#include <vector>

using namespace mochi;
using namespace mochi::mesh;

bool mochi::mesh::ExportStepVisual(
    std::string_view stepFilePath,
    Span<VisualExportOutput const> outputs,
    StepVisualExportParams const& params,
    std::vector<VisualExportStatus>& outStatuses,
    Error& error) {
  outStatuses.clear();
  MOCHI_ERROR_RETURN(error, false);
  MOCHI_ERROR_IF(outputs.empty(), error, "No output files were requested.");
  MOCHI_ERROR_RETURN(error, false);

  cli::PayloadWriter writer;
  writer.WriteByteArray(stepFilePath);
  writer.WriteStepVisualExportParams(params);
  writer.WriteVisualExportOutputs({outputs.data(), outputs.size()});

  // The helper writes the files and answers with their statuses rather than a mesh, so this goes
  // through the raw invoke instead of InvokeMeshOp.
  std::vector<char> const response =
      InvokeMeshCli(cli::GeometryOp::ExportStepVisual, writer.Bytes(), error);
  MOCHI_ERROR_RETURN(error, false);

  cli::PayloadReader reader(response);
  bool const decoded = reader.ReadVisualExportStatuses(outStatuses) && reader.AtEnd() &&
      outStatuses.size() == outputs.size();
  if (!decoded) {
    outStatuses.clear();
  }
  MOCHI_ERROR_IF(!decoded, error, "Malformed ExportStepVisual response from superdex_mesh_cli.");
  MOCHI_ERROR_RETURN(error, false);
  return true;
}
