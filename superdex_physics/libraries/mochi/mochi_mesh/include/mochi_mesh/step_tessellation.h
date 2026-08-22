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

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/utils/error.h>
#include <mochi_mesh/mochi_mesh_cli_types.h>

#include <string_view>

namespace mochi::mesh {

/// @brief Load a STEP CAD file (.step/.stp) and tessellate it into a triangle mesh.
///
/// Reads the file with OpenCascade and meshes every face, returning a single triangle mesh.
/// OpenCascade is isolated in the superdex_mesh_cli helper, so this call marshals the request
/// across the process boundary.
///
/// @param[in] stepFilePath Filesystem path to the STEP file (UTF-8).
/// @param[in] params Tessellation quality parameters.
/// @param[in,out] error Error status.
/// @return Triangle mesh in the renderer's Y-up frame, in metres (nodesPerElement == 3); empty on
/// error.
[[nodiscard]] MeshData
TessellateStep(std::string_view stepFilePath, StepTessellationParams const& params, Error& error);

} // namespace mochi::mesh
