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

// "Export Mochi Model" modifier: writes its input mesh (a triangle surface or a tetrahedral
// volume), with the meshes embedded in it (see EmbeddedMeshes), to a .mochi.h5 file on Export, by
// default with a baked grid SDF whose reconstructed surface is the in-stack output (preview). With
// Bake SDF off it writes the meshes alone and passes its input mesh through. This header exposes
// only the registry entry.

#include "meshing/processing_modifiers/processing_modifier.h"

namespace superdex::studio {

ModifierRegistryEntry MakeMochiModelExportEntry();

} // namespace superdex::studio
