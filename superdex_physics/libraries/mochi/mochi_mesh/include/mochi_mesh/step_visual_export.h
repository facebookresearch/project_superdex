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

#include <mochi_core/utils/error.h>
#include <mochi_core/utils/span.h>
#include <mochi_mesh/mochi_mesh_cli_types.h>

#include <string_view>
#include <vector>

namespace mochi::mesh {

/// @brief Convert a STEP CAD file into one or more render-ready mesh files, preserving the CAD
/// colors and the surfaces' analytic normals.
///
/// This is the visual counterpart to @ref MeshStepBody and @ref TessellateStep, which return an
/// unwelded triangle soup suitable for simulation but carry no materials and no smooth normals.
/// OpenCascade is isolated in the superdex_mesh_cli helper, so this call marshals the request
/// across the process boundary; the helper writes the files itself.
///
/// @param[in] stepFilePath Filesystem path to the STEP file (UTF-8).
/// @param[in] outputs Files to write, each with its own format. Must not be empty.
/// @param[in] params Export parameters, shared by every output.
/// @param[out] outStatuses Outcome of each output, in the order given. One format can fail to write
///                         while the others succeed, so check every entry. Cleared on failure.
/// @param[in,out] error Error status.
/// @return True when the STEP loaded and tessellated, i.e. every output was attempted and
///         @p outStatuses is populated. False means no file was written; see @p error.
[[nodiscard]] bool ExportStepVisual(
    std::string_view stepFilePath,
    Span<VisualExportOutput const> outputs,
    StepVisualExportParams const& params,
    std::vector<VisualExportStatus>& outStatuses,
    Error& error);

} // namespace mochi::mesh
