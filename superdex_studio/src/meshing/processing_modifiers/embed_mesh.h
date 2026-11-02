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

// "Embed Contact Skin" and "Embed Visual Mesh" modifiers: each passes its input mesh on unchanged
// with one more triangle mesh embedded in it, the soft actor's contact skin or visual mesh, which
// Export Mochi Model writes alongside it (see EmbeddedMeshes). Their methods are the ways of making
// that mesh, and each checks that it supports the kind of mesh it embeds in. The editor shows the
// embedded mesh as the stage's output.

#include "meshing/processing_modifiers/processing_modifier.h"

#include <mochi_core/geometry/mesh_data.h>

namespace superdex::studio {

ModifierRegistryEntry MakeEmbedContactSkinEntry();
ModifierRegistryEntry MakeEmbedVisualMeshEntry();

// The stats of @p mesh, embedded as @p role, for the stage that embedded it.
EmbeddedMeshStats MeasureEmbeddedMesh(EmbeddedMeshRole role, mochi::MeshData const& mesh);

} // namespace superdex::studio
