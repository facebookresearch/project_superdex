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

// The mesh-processing modifier stack for the Model Editor. A modifier is a named group of one or
// more methods (see processing_method.h) plus the shared per-instance state the editor manages
// (enabled / collapsed / output buffer / generation ids). The modifier delegates all processing to
// its currently-selected method. Modifiers are created by name from a registry, which also drives
// the "Add Modifier" menu and decodes the JSON "modifier" field on load.

#include "meshing/processing_modifiers/processing_method.h"

#include <mochi_renderer/utils.h> // MeshSection

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mochi_renderer {
class SceneObject;
} // namespace mochi_renderer

namespace superdex::studio {

// Surface/wireframe/opacity toggles for one modifier's output buffer (runtime only; not
// serialized).
struct StageVisualization {
  bool showSurface = false;
  bool showWireframe = false;
  bool showStats = false; // show the read-only mesh statistics block
  float opacity = 1.0f;
};

// A modifier's output geometry plus the scene objects built from it and its viz toggles. Owned by
// the modifier so the buffer travels with the element when the stack is reordered.
struct StageBuffer {
  std::vector<mochi_renderer::MeshSection> sections;
  mochi_renderer::SceneObject* surfaceMesh = nullptr;
  mochi_renderer::SceneObject* wireframeMesh = nullptr;
  StageVisualization viz;
  // Mesh statistics for `sections` (base summary + composited annotations), recomputed by the
  // editor when the geometry changes and shown under the viz controls when viz.showStats is on
  // (runtime only; not serialized).
  MeshStats stats;
};

// Whether a modifier produces geometry (source), transforms its input, or writes it to a file
// (export). Export behaves like a transform for ordering (not pinned to the front).
enum class ModifierKind { Source, Transform, Export };

// A modifier: a named group of >=1 methods plus shared per-instance state. Delegates ShowParams /
// Run / serialization / capabilities to the active method.
class MeshProcessingModifier {
 public:
  MeshProcessingModifier(
      std::string displayName,
      ModifierKind kind,
      std::vector<std::unique_ptr<MeshProcessingMethod>> methods);
  virtual ~MeshProcessingModifier() = default;

  // Also the registry key and the JSON "modifier" value (kept identical so it round-trips).
  char const* DisplayName() const {
    return _displayName.c_str();
  }
  ModifierKind Kind() const {
    return _kind;
  }

  // --- method selection ---
  std::size_t MethodCount() const {
    return _methods.size();
  }
  int ActiveMethodIndex() const {
    return _activeMethod;
  }
  MeshProcessingMethod& ActiveMethod() {
    return *_methods[static_cast<std::size_t>(_activeMethod)];
  }
  MeshProcessingMethod const& ActiveMethod() const {
    return *_methods[static_cast<std::size_t>(_activeMethod)];
  }
  char const* ActiveMethodName() const {
    return ActiveMethod().Name();
  }
  std::vector<std::unique_ptr<MeshProcessingMethod>> const& Methods() const {
    return _methods;
  }
  // Selects a method by combo index (clamped) or by name (returns false if not found).
  void SelectMethod(int index);
  bool SelectMethodByName(std::string_view name);

  // "method [modifier]" when the modifier has >1 method, else just the modifier name.
  virtual std::string HeaderLabel() const;

  // --- delegation to the active method (used throughout the editor) ---
  void ShowParams(ModifierGuiContext const& gui) {
    ActiveMethod().ShowParams(gui);
  }
  mochi::MeshData
  Run(mochi::MeshData const& input, ModifierRunContext const& ctx, mochi::Error& error) const {
    // Non-source modifiers consume their input mesh. If an upstream modifier failed and produced no
    // output, fail fast here with a clear error instead of running an op on empty geometry -- this
    // propagates the original failure cleanly down the chain (each later stage in turn gets no
    // input and stops). Sources ignore the input (they generate geometry from a file/slot), so they
    // are exempt.
    if (_kind != ModifierKind::Source && input.GetNumElements() == 0) {
      MOCHI_ERROR_SET(
          error, "Modifier received no input mesh (an upstream modifier produced no output).");
      return {};
    }
    return ActiveMethod().Run(input, ctx, error);
  }
  bool NeedsReferenceMesh() const {
    return ActiveMethod().NeedsReferenceMesh();
  }
  int ReferenceIndex() const {
    return ActiveMethod().ReferenceIndex();
  }
  // Remaps stored upstream references after a reorder/insert/remove. Applied to every method so a
  // reference set on a non-active method survives method switches.
  void RemapReferences(std::vector<int> const& oldToNew);
  bool CanGenerate(ModifierRunContext const& ctx) const {
    return ActiveMethod().CanGenerate(ctx);
  }
  std::string SourceFilePath(ModifierRunContext const& ctx) const {
    return ActiveMethod().SourceFilePath(ctx);
  }
  bool ProvidesFileExport() const {
    return ActiveMethod().ProvidesFileExport();
  }
  std::string ExportPath() const {
    return ActiveMethod().ExportPath();
  }
  void SaveToFile(mochi::MeshData const& input, mochi::Error& error) const {
    ActiveMethod().SaveToFile(input, error);
  }
  void RefreshAutoExportPath(std::string const& sourceFilePath) {
    ActiveMethod().RefreshAutoExportPath(sourceFilePath);
  }
  std::optional<mochi::Real3> PreferredDisplayColor() const {
    return ActiveMethod().PreferredDisplayColor();
  }
  void AnnotateStats(MeshStats& stats) const {
    ActiveMethod().AnnotateStats(stats);
  }
  // Active method name + its props signature, so switching method or editing params invalidates.
  std::string PropsSignature(ModifierRunContext const& ctx) const;

  // --- shared per-instance editor state ---
  bool enabled = true;
  bool collapsed = true; // new modifiers start collapsed so the stack stays compact
  StageBuffer output;
  // Optional copy of this modifier's INPUT mesh, with its own scene objects and viz toggles, shown
  // as an extra visualization section when the active method sets ShowsInputVisualization() (Export
  // Mochi Model shows its source surface next to the reconstructed SDF). Captured at generation so
  // it survives reorders; empty and unused for other modifiers.
  StageBuffer inputView;
  uint64_t id = 0; // stable across reorders (ImGui IDs + drag-drop); runtime only
  // Change-detection state maintained by the editor's generation cascade (runtime only).
  int outputGenId = 0;
  int inputGenIdAtLastGen = 0;
  int referenceGenIdAtLastGen = 0;
  std::string lastGenSignature;

 protected:
  std::string _displayName;
  ModifierKind _kind;
  std::vector<std::unique_ptr<MeshProcessingMethod>> _methods;
  int _activeMethod = 0;
};

// --- registry
// ------------------------------------------------------------------------------------- One
// registrable modifier type: its name (== DisplayName == JSON "modifier"), kind, and a factory that
// builds its ordered method instances.
struct ModifierRegistryEntry {
  std::string name;
  ModifierKind kind;
  std::function<std::vector<std::unique_ptr<MeshProcessingMethod>>()> makeMethods;
};

// The ordered set of modifier types (sources, then transforms, then exports). Order is the Add-menu
// order within each kind.
std::vector<ModifierRegistryEntry> const& ProcessingModifierRegistry();

// Builds a modifier by registry name (its methods created via the entry factory, active method 0),
// or nullptr if the name is not registered.
std::unique_ptr<MeshProcessingModifier> MakeProcessingModifier(std::string_view name);

} // namespace superdex::studio
