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

// A mesh-processing MODIFIER (see processing_modifier.h) is a named group of one or more METHODS.
// A method is a single concrete operation (e.g. "Make Watertight", "Alpha Wrap") that owns its
// parameters, its GUI, the neutral op it runs, and its JSON (de)serialization. Modifiers hold one
// instance of each of their methods and delegate to the active one. Methods are never shared
// between modifiers.
//
// Most methods store their parameters in a reflected struct and derive from ReflectedMethod<Props>,
// which implements the JSON + signature plumbing from that struct via the simple-reflection library
// (enums serialize as their string names, keeping the files human-readable/editable).

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/reflection.h> // MOCHI_STRUCT_BEGIN etc. (props structs reflect via these)

#include <simple_reflection/simple_reflection.h> // SReflect::ToJsonValue / FromJsonValue / ToJsonString

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace picojson {
class value;
} // namespace picojson

namespace superdex::studio {

// What a mesh embedded in another is for: the field of mochi::ModelData it is exported as.
enum class EmbeddedMeshRole { ContactSkin };

// Triangle meshes embedded in a stage's mesh, each by skinning that gives its nodes as weighted
// combinations of that mesh's nodes. They are valid only for the exact mesh they were embedded in,
// so they pass down the stack from the Embed modifier that made them only through stages that
// accept them (see MeshProcessingMethod::AcceptsEmbeddedMeshes). Named after, and exported as, the
// mochi::ModelData fields of the same names.
struct EmbeddedMeshes {
  std::optional<mochi::MeshData> contactSkinMesh;

  // The field holding the mesh embedded in @p role.
  [[nodiscard]] static std::optional<mochi::MeshData> EmbeddedMeshes::* Field(
      EmbeddedMeshRole role) {
    switch (role) {
      case EmbeddedMeshRole::ContactSkin:
        return &EmbeddedMeshes::contactSkinMesh;
    }
    return nullptr;
  }
  [[nodiscard]] std::optional<mochi::MeshData>& Mesh(EmbeddedMeshRole role) {
    return this->*Field(role);
  }
  [[nodiscard]] std::optional<mochi::MeshData> const& Mesh(EmbeddedMeshRole role) const {
    return this->*Field(role);
  }
  [[nodiscard]] bool IsEmpty() const {
    return !contactSkinMesh.has_value();
  }
};

// What one stage hands the next: its mesh, plus the meshes embedded in it.
struct StageMesh {
  mochi::MeshData mesh;
  EmbeddedMeshes embedded;
};

// Everything a method's Run may need beyond its own params and input, supplied by the editor.
struct ModifierRunContext {
  // Resolved upstream reference mesh, for methods that NeedsReferenceMesh (see ReferenceIndex).
  mochi::MeshData referenceMesh;
  std::string cadFilePath; // CAD (STEP) file for the CAD source
  std::string renderModelPath; // editor's current Render Model slot (From Model Viewer)
  std::string mochiModelPath; // editor's current Mochi Model slot (From Model Viewer)
  // CAD transform baked into the CAD source's output so it flows downstream; ignored by transforms.
  mochi::Real3 cadScale = {1.0f, 1.0f, 1.0f};
  mochi::Quaternion cadRotation = {};
  mochi::Real3 cadTranslation = {};
};

// Hover-tooltip helper injected by the editor so a method can annotate its widgets consistently.
using ModifierTooltip = std::function<void(char const*)>;

// The model types a source method's own file slot can hold. AnyModel holds a CAD, render, or Mochi
// model.
enum class ModelSlotType { CadModel, RenderModel, MochiModel, AnyModel };

// The app's widgets and services, for the methods whose parameters need more than Dear ImGui. The
// editor implements it; without the app ModifierGuiContext::host is null and methods leave those
// widgets out.
class ModifierHost {
 public:
  virtual ~ModifierHost() = default;

  // Asset slot showing and accepting a model file of @p type, bound to @p path.
  virtual void ModelSlot(char const* id, mochi::DynamicString& path, ModelSlotType type) = 0;
  // Native save dialog opened at @p defaultPath. Returns the chosen path, or empty on cancel.
  virtual std::string SaveFileDialog(
      char const* title,
      std::span<char const* const> filters,
      char const* filterDescription,
      std::string const& defaultPath) = 0;
  // Editor for every field of the reflected struct at @p value. Returns true when a field changed.
  virtual bool ReflectedStruct(SReflect::StructTypeInfo const& type, void* value) = 0;
  // Per-axis drag for @p value, moving @p speed per pixel and shown with the printf @p format, and
  // a rotation + translation editor. Return true on change.
  virtual bool DragXYZ(char const* label, mochi::Real3& value, float speed, char const* format) = 0;
  virtual bool
  DragTransformRT(char const* label, mochi::Quaternion& rotation, mochi::Real3& translation) = 0;
  // The stable color the app derives from a file path, so a default matches the models' colors.
  virtual mochi::Real3 PathColor(std::string_view path) = 0;
  // Runs @p work on a worker thread behind the app's progress modal, then @p onDone on the main
  // thread. Returns false when no task could start.
  virtual bool RunBackgroundTask(
      char const* title,
      char const* label,
      std::function<bool()> work,
      std::function<void()> onDone) = 0;
  // True while a background task is running; only one runs at a time.
  virtual bool IsBackgroundTaskRunning() const = 0;
};

// What the editor hands each method's ShowParams. Methods draw with Dear ImGui and reach the app
// only through @ref host, so they never depend on it.
struct ModifierGuiContext {
  ModifierTooltip tooltip;
  ModifierHost* host = nullptr;
  // File path of the stack's first (source) modifier, used to pre-populate export default paths.
  std::string sourceFilePath;
  // Folder of the model this editor was opened for. Used only as the Browse dialog's starting point
  // when there is no sourceFilePath to derive a suggestion from, so the user lands beside the asset
  // instead of an arbitrary folder. Never used to build a stored path.
  std::string modelFolder;
  // Editor's current CAD Model slot path. The CAD "From Model Viewer" source uses its extension to
  // show STEP tessellation options only when the slotted file is a STEP (an STL needs none).
  std::string cadFilePath;
  // --- assigned per modifier just before its ShowParams call, not at construction ---------------
  // This modifier's index in the stack, and the display names of all modifiers by array index. The
  // edge-flip reference dropdown uses these to list the elements above this one.
  std::size_t selfIndex = 0;
  std::vector<std::string> const* modifierNames = nullptr;
  // Set when another enabled export modifier writes to this one's export path, so the method can
  // flag it inline. Two exports sharing a file silently overwrite each other, and the Auto path is
  // deliberately not disambiguated (it has to keep the model's own name to stay associated with
  // it), so resolving this means turning Auto off on one of them.
  bool exportPathCollides = false;
};

// Size of a triangle mesh embedded in a tetrahedral mesh, and how far its nodes reach outside the
// tetrahedra (mochi::model::MaxAuxiliaryMeshExtrapolation, in element heights).
struct EmbeddedMeshStats {
  EmbeddedMeshRole role = EmbeddedMeshRole::ContactSkin;
  int nodes = 0;
  int triangles = 0;
  double maxExtrapolation = 0.0;
};

// Statistics shown in a mesh's viz "Stats" block. `base` is the geometry-derived summary (verts /
// edges / tris / AABB / edge-length / angles), recomputed only when the mesh changes. Annotations
// known separately -- an SDF grid (from the producing modifier at generation) or a Hausdorff
// distance (from the async CLI) -- are composited around `base` into `display` without recomputing
// it. Extend by adding a field here plus a line in ComposeStats (model_editor.cpp).
struct MeshStats {
  std::string base; // FormatMeshStats output; the geometry summary
  std::optional<mochi::Int3> sdfGrid; // prepended "SDF grid" line when set
  std::optional<mochi::Int2> tetMesh; // prepended "Tet mesh" line (nodes, tets) when set
  std::optional<EmbeddedMeshStats> embeddedMesh; // prepended line named after its role when set
  // File-size line shown first, when fileSizeBytes is set. `fileSizeLabel` is its caption ("File
  // Size" for an on-disk Model Viewer slot, "Est. File Size" for a method's GLB estimate, "Est.
  // Export Size" for an export modifier). See ComposeStats (model_editor.cpp).
  std::optional<int64_t> fileSizeBytes;
  std::string fileSizeLabel;
  double hausdorff = -1.0; // appended "Hausdorff" line when >= 0
  std::string display; // composited text actually shown in the UI
};

// One concrete operation within a modifier.
class MeshProcessingMethod {
 public:
  // Sentinel ReferenceIndex meaning "use the nearest preceding source modifier" (the default).
  static constexpr int kReferencePrecedingSource = -1;

  virtual ~MeshProcessingMethod() = default;

  // Stable, human-readable identifier used both in the method combo and as the JSON "method" value.
  // Renaming it changes the JSON key, so keep it stable.
  [[nodiscard]] virtual char const* Name() const = 0;

  // One-line explanation of what this method does, shown as a hover tooltip on the modifier's
  // Method combo (on both the collapsed selector and each option). Only surfaced when the modifier
  // has more than one method -- single-method modifiers hide the combo. Empty default = no tooltip.
  [[nodiscard]] virtual char const* Description() const {
    return "";
  }

  // Renders the method's parameter widgets only (the modifier wraps the chrome and disables these
  // when the modifier is disabled).
  virtual void ShowParams(ModifierGuiContext const& gui) = 0;

  // Produces this method's output mesh. Source methods ignore @p input.
  virtual mochi::MeshData
  Run(mochi::MeshData const& input, ModifierRunContext const& ctx, mochi::Error& error) const = 0;
  // Produces this method's output with the meshes embedded in it, which is what the stack runs. The
  // default runs Run and embeds nothing; methods that AcceptsEmbeddedMeshes override it to read and
  // pass on the meshes embedded in @p input.
  virtual StageMesh
  RunStage(StageMesh const& input, ModifierRunContext const& ctx, mochi::Error& error) const {
    return {.mesh = Run(input.mesh, ctx, error)};
  }

  // JSON (de)serialization of this method's parameters. ReflectedMethod implements these from a
  // reflected Props struct; the unknown-placeholder method implements them by hand to round-trip
  // raw captured data. DeserializeProps returns how many fields and values of @p in it could not
  // read (unknown fields and enum values, values of the wrong type), which keep their defaults.
  virtual void SerializeProps(picojson::value& out) const = 0;
  virtual int DeserializeProps(picojson::value const& in) = 0;
  // A compact string encoding every parameter that affects this method's output, used by the
  // editor's generation cascade to detect changes. @p ctx lets source methods fold in their
  // external inputs (slotted file path / CAD transform).
  [[nodiscard]] virtual std::string PropsSignature(ModifierRunContext const& ctx) const = 0;

  // JSON keys of this method's props that hold a file path. The pipeline serializer rewrites the
  // value at each key between the absolute path kept in memory and the bot-relative form written to
  // disk (via superdex::robotics::MakePathRelative / MakePathAbsolute), so the saved JSON never
  // embeds an absolute path. "list[].field" names a field of every element of an array prop. Empty
  // for methods with no path parameter (the default).
  [[nodiscard]] virtual std::vector<std::string_view> PathPropKeys() const {
    return {};
  }

  // --- optional capabilities (defaults suit most methods) ---------------------------------------
  // True for methods that read an upstream reference mesh resolved into ctx.referenceMesh.
  [[nodiscard]] virtual bool NeedsReferenceMesh() const {
    return false;
  }
  [[nodiscard]] virtual int ReferenceIndex() const {
    return kReferencePrecedingSource;
  }
  virtual void RemapReferences(std::vector<int> const& /*oldToNew*/) {}
  // Whether Run accepts a tetrahedral input mesh (e.g. downstream of Tetrahedralize). Most methods
  // operate on triangle surfaces only.
  [[nodiscard]] virtual bool AcceptsTetrahedralInput() const {
    return false;
  }
  // Whether RunStage reads the meshes embedded in its input: the Embed modifiers, which add to
  // them, and Export Mochi Model, which writes them. The stack refuses to run any other method on
  // an input carrying embedded meshes, since its output mesh would no longer be the one they are
  // embedded in.
  [[nodiscard]] virtual bool AcceptsEmbeddedMeshes() const {
    return false;
  }
  // The role of the mesh this method embeds (the Embed modifiers), which the editor shows as its
  // output in place of the unchanged mesh it passes on. nullopt for every other method.
  [[nodiscard]] virtual std::optional<EmbeddedMeshRole> EmbeddedRole() const {
    return std::nullopt;
  }
  // Whether this method can currently produce output given @p ctx (sources check their file /
  // slot).
  [[nodiscard]] virtual bool CanGenerate(ModifierRunContext const& /*ctx*/) const {
    return true;
  }
  // A source method's input file path (used to derive export default paths); empty otherwise.
  [[nodiscard]] virtual std::string SourceFilePath(ModifierRunContext const& /*ctx*/) const {
    return {};
  }
  // Export methods (Export Mochi Model, Export Mesh File) show a separate Export button and write a
  // file from their current input mesh via SaveToFile.
  [[nodiscard]] virtual bool ProvidesFileExport() const {
    return false;
  }
  // The file SaveToFile would write, normalized exactly as it will be written. Empty when this
  // method has no export path or none is set yet -- which is what the editor tests to decide
  // whether Build/Export All should include this modifier, and what it compares across modifiers to
  // detect two exports writing to the same file.
  [[nodiscard]] virtual std::string ExportPath() const {
    return {};
  }
  virtual void SaveToFile(mochi::MeshData const& /*input*/, mochi::Error& /*error*/) const {}
  // Re-derives this method's export path from @p sourceFilePath (the file the stack's source
  // modifier reads) while its Auto toggle is on, and does nothing once the user has taken the path
  // over. An auto path therefore always equals what Browse would offer, and an export modifier
  // writes its file on Build/Export All without the user opening Browse at all.
  //
  // Called by the editor for the whole stack every frame. That is deliberate: the suggestion
  // depends on the front modifier, its active method, that method's slot, the editor's own model
  // slots, and on whether the sibling output folder exists -- too many inputs to hook individually,
  // and the last is not an event at all. It is safe to recompute freely because an auto path is
  // never serialized (see SerializeProps on the export methods), so it cannot dirty the saved
  // snapshot or a future undo/redo state no matter how often it moves.
  virtual void RefreshAutoExportPath(std::string const& /*sourceFilePath*/) {}
  // Points the export at @p path instead of this method's own setting, turning Auto off. Used to
  // collect a headless run's exports in one folder; not serialized unless the pipeline is saved.
  virtual void OverrideExportPath(std::string const& /*path*/) {}
  // Optional override for the color the editor shows this method's output surface with. Most
  // methods return nullopt (the editor derives a hashed stage color); the Export Mesh File method
  // returns its configured material color so its viewport preview matches the color it will write.
  [[nodiscard]] virtual std::optional<mochi::Real3> PreferredDisplayColor() const {
    return std::nullopt;
  }
  // Lets a method contribute extra stats for its output after generation (e.g. Export Mochi Model
  // sets the baked SDF's grid dimensions). Default no-op; the editor composes the display string.
  virtual void AnnotateStats(MeshStats& /*stats*/) const {}

  // When true, the editor shows this modifier's INPUT mesh as an extra visualization section (a
  // copy captured at generation, so it survives reorders) alongside the output, each under its own
  // SeparatorText heading. Used by Export Mochi Model so the source surface can be compared with
  // the SDF surface it reconstructs. Most methods show only the single output block.
  [[nodiscard]] virtual bool ShowsInputVisualization() const {
    return false;
  }
  // Section headings for the input-mesh and output-mesh visualization blocks (only used when
  // ShowsInputVisualization() is true).
  [[nodiscard]] virtual char const* InputVisualizationLabel() const {
    return "Input Visualization";
  }
  [[nodiscard]] virtual char const* OutputVisualizationLabel() const {
    return "Output Visualization";
  }

 protected:
  // RemapReferences for a method storing one reference index: follows the referenced element to its
  // new index, and falls back to the preceding source if it was removed.
  static void RemapReferenceIndex(int& referenceIndex, std::vector<int> const& oldToNew) {
    if (referenceIndex < 0 || referenceIndex >= static_cast<int>(oldToNew.size())) {
      return; // preceding-source sentinel, or (defensively) out of range: leave unchanged
    }
    referenceIndex = oldToNew[static_cast<std::size_t>(referenceIndex)];
  }
};

// Helper base for methods whose parameters live in a reflected struct `Props`. Provides the JSON +
// signature implementations so concrete methods only declare their Props fields and their behavior.
// Source methods that also depend on ctx (slotted paths / CAD transform) override PropsSignature to
// append those.
template <typename Props>
class ReflectedMethod : public MeshProcessingMethod {
 public:
  void SerializeProps(picojson::value& out) const override {
    SReflect::ToJsonValue(_props, out);
  }
  int DeserializeProps(picojson::value const& in) override {
    int numIssues = 0;
    SReflect::FromJsonValue(_props, in, SReflect::DeserializeFlags::Default, numIssues);
    return numIssues;
  }
  [[nodiscard]] std::string PropsSignature(ModifierRunContext const& /*ctx*/) const override {
    return SReflect::ToJsonString(_props, /*pretty=*/false);
  }

 protected:
  Props _props;
};

// Shared reflected props for methods that take no parameters (Convex Hull, Make Watertight, the
// From-Model-Viewer sources, etc.).
struct EmptyMethodProps {
  MOCHI_STRUCT_BEGIN(superdex::studio::EmptyMethodProps)
  MOCHI_STRUCT_END()
};

} // namespace superdex::studio
