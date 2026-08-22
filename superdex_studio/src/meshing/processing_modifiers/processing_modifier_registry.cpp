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

#include "meshing/processing_modifiers/processing_modifier.h"

#include "meshing/processing_modifiers/export_mesh_file.h"
#include "meshing/processing_modifiers/export_mochi_model.h"
#include "meshing/processing_modifiers/refine_mesh.h"
#include "meshing/processing_modifiers/remesh.h"
#include "meshing/processing_modifiers/source_cad_model.h"
#include "meshing/processing_modifiers/source_mochi_model.h"
#include "meshing/processing_modifiers/source_render_model.h"
#include "meshing/processing_modifiers/transform.h"
#include "meshing/processing_modifiers/wrap_mesh.h"

#include <memory>
#include <string_view>
#include <vector>

namespace superdex::studio {

std::vector<ModifierRegistryEntry> const& ProcessingModifierRegistry() {
  // Built once on first use. Order = the Add-menu order within each kind (sources, then transforms,
  // then exports). The name is both the registry key and the JSON "modifier" value.
  static std::vector<ModifierRegistryEntry> const registry = [] {
    std::vector<ModifierRegistryEntry> entries;
    // Sources
    entries.push_back(MakeCadModelSourceEntry());
    entries.push_back(MakeRenderModelSourceEntry());
    entries.push_back(MakeMochiModelSourceEntry());
    // Transforms
    entries.push_back(MakeRefineMeshEntry());
    entries.push_back(MakeWrapMeshEntry());
    entries.push_back(MakeRemeshEntry());
    entries.push_back(MakeTransformEntry());
    // Exports
    entries.push_back(MakeMochiModelExportEntry());
    entries.push_back(MakeExportMeshFileEntry());
    return entries;
  }();
  return registry;
}

std::unique_ptr<MeshProcessingModifier> MakeProcessingModifier(std::string_view name) {
  for (ModifierRegistryEntry const& entry : ProcessingModifierRegistry()) {
    if (name == entry.name) {
      return std::make_unique<MeshProcessingModifier>(entry.name, entry.kind, entry.makeMethods());
    }
  }
  return nullptr;
}

} // namespace superdex::studio
