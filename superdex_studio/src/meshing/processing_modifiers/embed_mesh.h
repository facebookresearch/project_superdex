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

// "Embed Contact Skin" and "Embed Visual Mesh" pass their input mesh on unchanged with an embedded
// triangle mesh: a soft or rod actor's contact skin or visual mesh, which Export Mochi Model writes
// alongside it (see EmbeddedMeshes). Each method checks the kind of mesh it embeds in. The editor
// shows the embedded mesh as the stage's output.

#include "meshing/processing_modifiers/processing_modifier.h"

#include <mochi_core/geometry/mesh_data.h>

namespace superdex::studio {

ModifierRegistryEntry MakeEmbedContactSkinEntry();
ModifierRegistryEntry MakeEmbedVisualMeshEntry();

// The stats of @p mesh, embedded as @p role in @p host, for the stage that embedded it.
EmbeddedMeshStats MeasureEmbeddedMesh(
    EmbeddedMeshRole role,
    mochi::MeshData const& mesh,
    mochi::MeshData const& host);

} // namespace superdex::studio
