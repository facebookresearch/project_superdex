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

// Where an export modifier's output belongs, derived from the model it came from. Kept apart from
// processing_mesh_utils so that resolving an asset's role folders -- which needs superdex_robotics
// -- does not pull that dependency into the mesh helpers, whose translation unit mesh_space_test
// compiles on its own against mochi alone.

#include <string>
#include <string_view>

namespace superdex::studio::processing {

// Default export path derived from the source file: a same-basenamed file with @p ext, in the
// @p roleSubdir folder of the source's asset (superdex::robotics::AssetRoleFolderForWrite). An
// asset that uses role folders exports into `<base>/<roleSubdir>` whether or not that folder exists
// yet, so a STEP opened from `cad/` exports its render and collision models to `render/` and
// `collision/`; a flat asset exports beside the source. Empty when there is no source file path.
// The folder itself is not created -- callers offering the path to a file dialog must create it
// first (superdex::robotics::EnsureDirectoriesCreated), since the dialog ignores a default whose
// parent is missing.
std::string
DefaultExportPath(std::string const& sourceFilePath, std::string_view roleSubdir, char const* ext);

} // namespace superdex::studio::processing
