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
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/error.h>
#include <mochi_mesh/mochi_mesh_cli_encoding.h>

#include <vector>

namespace mochi::mesh::cli_adapter {

// Boundary between Mochi mesh types and the neutral superdex_mesh_cli protocol. This adapter only
// *converts* (validation + `real` <-> `double`) so the helper never depends on Mochi types; the
// wire (de)serialization lives alongside each struct in the shared encoding files.

[[nodiscard]] cli::MeshData ToCliMeshData(MeshDataView const& mesh, Error& error);
[[nodiscard]] MeshData FromCliMeshData(cli::MeshData const& mesh, Error& error);

[[nodiscard]] cli::ScalarField3d ToCliScalarField(GridSdfDataView const& grid, Error& error);
[[nodiscard]] GridSdfData FromCliScalarField(cli::ScalarField3d const& field, Error& error);

} // namespace mochi::mesh::cli_adapter
