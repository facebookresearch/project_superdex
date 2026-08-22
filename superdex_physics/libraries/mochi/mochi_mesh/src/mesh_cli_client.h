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

#include <mochi_mesh/mochi_mesh_cli_encoding.h>

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/span.h>

#include <string>
#include <vector>

// Cross-platform client for the superdex_mesh_cli helper executable. mochi_mesh routes its heavier
// mesh operations through this client (spawn-per-request over pipes). This header is internal to
// mochi_mesh (not part of the public API).

namespace mochi::mesh {

// Resolves the path to the superdex_mesh_cli helper executable, trying, in order:
//   1. the SUPERDEX_MESH_CLI_PATH environment variable (used by packaging, test configuration, and
//      local overrides),
//   2. the helper sitting next to the current executable (the production / Buck-resource case),
//   3. the sibling `superdex_mesh_cli` distribution's payload directory (the wheel case, where the
//      GPL helper ships separately and lands in a different site-packages directory).
// Returns an empty string if the helper cannot be found.
[[nodiscard]] std::string FindMeshCliPath();

// Spawns the helper, writes a single framed request, reads the framed response to EOF, and returns
// the response payload bytes. Sets @p error (and returns an empty vector) if the helper is missing,
// cannot be spawned, returns a non-zero status, or produces a malformed response. Never throws or
// crashes on a missing/unspawnable helper.
[[nodiscard]] std::vector<char>
InvokeMeshCli(cli::GeometryOp op, Span<char const> requestPayload, Error& error);

// Resolves the helper normally and forwards a non-empty @p cliExtraArg as argv[1]. Use this when a
// helper needs one setup argument before it begins reading the framed request from stdin.
[[nodiscard]] std::vector<char> InvokeMeshCli(
    std::string const& cliExtraArg,
    cli::GeometryOp op,
    Span<char const> requestPayload,
    Error& error);

// Sends one framed request and decodes the mesh it returns. Sets @p error (and returns an empty
// MeshData) if the call fails, or if the response payload is malformed or has trailing bytes.
// This is the whole client-side tail of every mesh-returning op.
[[nodiscard]] MeshData
InvokeMeshOp(cli::GeometryOp op, cli::PayloadWriter const& writer, Error& error);

} // namespace mochi::mesh
