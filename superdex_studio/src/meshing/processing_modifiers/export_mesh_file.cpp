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

#include "meshing/processing_modifiers/export_mesh_file.h"

#include "app/app.h" // SuperDexStudio::GetFileDialogPath
#include "meshing/processing_modifiers/processing_export_path.h" // DefaultExportPath
#include "meshing/processing_modifiers/processing_mesh_utils.h" // WriteObjFile, WriteGlbFile, EndsWithNoCase
#include "ui/imgui_widgets.h" // ImGui::InputText

#include <picojson/picojson.h>

#include <superdex_robotics/utils/file_utils.h> // kRenderSubdir

#include <imguios/imguios.h>

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/nd_array.h>

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace superdex::studio {

namespace {

// Reflected props: the output path and the base (RGB) material color written into a GLB.
struct ExportMeshFileProps {
  mochi::DynamicString path;
  mochi::Real3 color = {0.5f, 0.5f, 0.5f};

  MOCHI_STRUCT_BEGIN(superdex::studio::ExportMeshFileProps)
  MOCHI_FIELD(path)
  MOCHI_FIELD(color)
  MOCHI_STRUCT_END()
};

// The file a given path setting actually writes: the extension picks the format, and anything
// without a recognized one is exported as GLB with the suffix appended.
std::string NormalizeMeshExportPath(std::string path) {
  if (!path.empty() && !processing::EndsWithNoCase(path, ".glb") &&
      !processing::EndsWithNoCase(path, ".obj")) {
    path += ".glb";
  }
  return path;
}

class MeshFileExportMethod : public ReflectedMethod<ExportMeshFileProps> {
 public:
  char const* Name() const override {
    return "Mesh File";
  }
  char const* Description() const override {
    return "Export the input mesh to a mesh file (.glb / .obj) in a chosen material color.";
  }
  bool ProvidesFileExport() const override {
    return true;
  }
  std::string ExportPath() const override {
    return NormalizeMeshExportPath(std::string{_props.path.data(), _props.path.size()});
  }
  // The Export Mesh File stage previews in its own configured material color, so the viewport
  // matches what it will write, rather than a hashed stage color.
  std::optional<mochi::Real3> PreferredDisplayColor() const override {
    return _props.color;
  }

  std::vector<std::string_view> PathPropKeys() const override {
    return {"path"};
  }
  void RefreshAutoExportPath(std::string const& sourceFilePath) override {
    if (_autoExportPath) {
      _props.path = SuggestedExportPath(sourceFilePath);
    }
  }

  // An auto path is reproduced on load, so it is left out of the document entirely; the presence of
  // the key is exactly what marks a path as the user's own (see DeserializeProps).
  void SerializeProps(picojson::value& out) const override {
    ReflectedMethod::SerializeProps(out);
    if (_autoExportPath && out.is<picojson::object>()) {
      out.get<picojson::object>().erase("path");
    }
  }

  void DeserializeProps(picojson::value const& in) override {
    ReflectedMethod::DeserializeProps(in);
    _colorInitialized = true; // a loaded modifier's saved color is authoritative
    // A stored path means the user chose it, even if it happens to equal the current suggestion.
    _autoExportPath = _props.path.empty();
  }

  void ShowParams(ModifierGuiContext const& gui) override {
    ModifierTooltip const& tooltip = gui.tooltip;

    // Default the material color once, from the hash of the output path Browse will offer (the same
    // path-hash scheme the models use), unless it was loaded or the user has already picked a
    // color.
    if (!_colorInitialized) {
      std::string const currentPath{_props.path.data(), _props.path.size()};
      std::string const defaultPath =
          !currentPath.empty() ? currentPath : SuggestedExportPath(gui.sourceFilePath);
      if (!defaultPath.empty()) {
        ImVec4 const c = HashStringToColor(defaultPath);
        _props.color = {c.x, c.y, c.z};
        _colorInitialized = true;
      }
    }

    ImGui::TextUnformatted("Auto");
    ImGui::SameLine();
    if (ImGui::Checkbox("##autopath", &_autoExportPath)) {
      RefreshAutoExportPath(gui.sourceFilePath); // takes effect immediately when switched back on
    }
    tooltip(
        "Keep the output path derived from the source model, and leave it out of the saved "
        "pipeline. Turn this off (or press Browse) to choose the path yourself -- including to "
        "export a .obj, since the derived path is always a .glb.");
    ImGui::SameLine();
    ImGui::BeginDisabled(_autoExportPath);
    ImGui::InputText("Path", &_props.path);
    tooltip(
        "Output path; the extension selects the format (.obj or .glb). No/unknown suffix defaults to "
        ".glb.");
    ImGui::EndDisabled();
    // Browse stays live while Auto is on: choosing a file IS taking the path over, so it turns Auto
    // off. Cancelling leaves everything as it was -- Browse only ever turns Auto off, never on.
    if (ImGui::Button("Browse##meshexport")) {
      char const* filters[] = {"*.obj", "*.glb"};
      std::string const defaultPath = processing::ExportDialogStartPath(
          std::string{_props.path.data(), _props.path.size()},
          SuggestedExportPath(gui.sourceFilePath),
          gui.modelFolder);
      // The suggested folder may not exist yet (the asset uses role folders but has no `render/`);
      // create it up front, because the native dialog silently falls back to its last-used
      // directory when a default's parent is missing, which would defeat suggesting it at all.
      superdex::robotics::EnsureDirectoriesCreated(
          defaultPath, mochi::ErrorLog{mochi::LogChannel::Warning});
      mochi::Path const chosen = SuperDexStudio::GetFileDialogPath(
          "Export Mesh",
          filters,
          2,
          "Mesh (*.obj, *.glb)",
          /*isSaveDialog=*/true,
          mochi::Path{defaultPath});
      if (!chosen.IsEmpty()) {
        _props.path = chosen.ToString();
        _autoExportPath = false;
      }
    }
    tooltip("Choose the output file (the extension picks .obj or .glb).");
    if (gui.exportPathCollides) {
      ImGui::TextColored(
          ImVec4(1.0f, 0.4f, 0.3f, 1.0f), "Another export writes to this same file.");
      tooltip(
          "Two enabled export modifiers share an output path, so one silently overwrites the "
          "other. Turn Auto off on one of them and give it a path of its own.");
    }

    // ColorEdit needs a float[3]; copy in/out (color storage is `real`, which may not be float).
    float rgb[3] = {
        static_cast<float>(_props.color[0]),
        static_cast<float>(_props.color[1]),
        static_cast<float>(_props.color[2])};
    if (ImGui::ColorEdit3("Material Color", rgb, ImGuiColorEditFlags_NoInputs)) {
      _props.color = {rgb[0], rgb[1], rgb[2]};
      _colorInitialized = true; // user picked a color; stop auto-defaulting from the path
    }
    tooltip(
        "Base color written into the exported mesh's PBR material (the glTF baseColorFactor). Only "
        "GLB stores materials; OBJ export ignores this.");
  }

  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& /*error*/) const override {
    _lastInput = input; // cache for the export-size estimate (AnnotateStats)
    return {input}; // passthrough; the file is written by SaveToFile
  }

  void SaveToFile(mochi::MeshData const& input, mochi::Error& error) const override {
    std::string const path = ExportPath();
    if (path.empty()) {
      MOCHI_ERROR_SET(error, "Export Mesh File: set an output path first.");
      return;
    }
    if (input.GetNumElements() == 0) {
      MOCHI_ERROR_SET(error, "Export Mesh File: input mesh is empty.");
      return;
    }
    // The target folder may not exist yet -- a derived render/ folder in a fresh bot, or one the
    // user typed. Create it rather than failing the write.
    superdex::robotics::EnsureDirectoriesCreated(path, error);
    MOCHI_ERROR_RETURN(error);
    std::array<float, 4> const baseColor{
        static_cast<float>(_props.color[0]),
        static_cast<float>(_props.color[1]),
        static_cast<float>(_props.color[2]),
        1.0f};
    // ExportPath has already resolved the extension, so only the two real formats remain.
    if (processing::EndsWithNoCase(path, ".obj")) {
      processing::WriteObjFile(path, input, error); // OBJ stores no material; color is not applied
    } else {
      processing::WriteGlbFile(path, input, baseColor, error);
    }
  }

  // Estimated size of the file this stage will export, keyed to the configured extension (OBJ
  // text vs GLB binary; an unknown/empty suffix exports GLB). Uses the cached input mesh from the
  // last Run. The editor composites this into the display string.
  void AnnotateStats(MeshStats& stats) const override {
    std::string const path = _props.path.c_str();
    stats.fileSizeBytes = processing::EndsWithNoCase(path, ".obj")
        ? processing::EstimateObjSizeBytes(_lastInput)
        : processing::EstimateGlbSizeBytes(_lastInput);
    stats.fileSizeLabel = "Est. Export Size";
  }

 private:
  // The path Browse offers, and what RefreshAutoExportPath keeps the field at while Auto is on.
  static std::string SuggestedExportPath(std::string const& sourceFilePath) {
    return processing::DefaultExportPath(sourceFilePath, superdex::robotics::kRenderSubdir, ".glb");
  }

  // Whether the export path is still derived from the source model rather than chosen by the user.
  // Not serialized: the presence of a stored path is what encodes it (see De/SerializeProps), so a
  // pipeline exporting to the derived location carries no path at all and stays valid when copied
  // into processing_presets/ or onto a machine whose assets live elsewhere. Runtime only.
  bool _autoExportPath = true;
  // Whether the material color has been set from the path hash (or loaded / user-picked). Until
  // then ShowParams derives it once from the projected default output path. Runtime only.
  bool _colorInitialized = false;
  // Input mesh from the last Run, cached so AnnotateStats can size the export without re-running.
  // Runtime only.
  mutable mochi::MeshData _lastInput;
};

} // namespace

ModifierRegistryEntry MakeExportMeshFileEntry() {
  return ModifierRegistryEntry{"Export Mesh File", ModifierKind::Export, []() {
                                 std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
                                 methods.push_back(std::make_unique<MeshFileExportMethod>());
                                 return methods;
                               }};
}

} // namespace superdex::studio
