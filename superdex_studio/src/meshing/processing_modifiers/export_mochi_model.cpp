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

#include "meshing/processing_modifiers/export_mochi_model.h"

#include "meshing/mesh_conversion.h" // MeshSectionsToModel (render->mochi space conversion for .mochi.h5)
#include "meshing/processing_modifiers/processing_export_path.h" // DefaultExportPath
#include "meshing/processing_modifiers/processing_mesh_utils.h" // SectionFromMeshData, WeldIdenticalVertices, ToMochiSpace, EndsWithNoCase

#include <picojson/picojson.h>

#include <superdex_robotics/utils/file_utils.h> // kCollisionSubdir

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h> // ImGui::InputText(std::string*)

#include <mochi_mesh/isosurface_reconstruction.h> // ReconstructSurfaceFromSdf

#include <mochi_renderer/render_space.h>
#include <mochi_renderer/utils.h> // MeshSection, ConvertMeshSectionsSpace

#include <mochi_core/geometry/grid_sdf_params.h>
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_physics/utils/mochi_model_utils.h> // BakeSdf, Validate, SaveToFile, FileFormat

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace superdex::studio {

namespace {

using mochi_renderer::MeshSection;

// Reflected props: whether to bake an SDF, its parameters (a reflected mochi type, nested), and the
// export path.
struct MochiModelExportProps {
  bool bakeSdf = true;
  mochi::GridSdfParams sdf;
  mochi::DynamicString exportPath;

  MOCHI_STRUCT_BEGIN(superdex::studio::MochiModelExportProps)
  MOCHI_FIELD(bakeSdf)
  MOCHI_FIELD(sdf)
  MOCHI_FIELD(exportPath)
  MOCHI_STRUCT_END()
};

std::string NormalizeMochiModelExportPath(std::string path) {
  if (!path.empty() && !processing::EndsWithNoCase(path, ".mochi.h5")) {
    if (processing::EndsWithNoCase(path, ".h5")) {
      path.resize(path.size() - std::strlen(".h5"));
    }
    path += ".mochi.h5";
  }
  return path;
}

// The input mesh, and the meshes embedded in it, converted from the pipeline's renderer (Y-up)
// space to Mochi (Z-up) space, which is what a .mochi.h5 stores. Triangle surfaces keep their
// established section-based conversion; tetrahedral meshes, and meshes carrying embedded meshes,
// are converted directly so their nodes and connectivity are kept.
mochi::ModelData MochiSpaceModel(mochi::MeshData const& input, EmbeddedMeshes const& embedded) {
  if (input.nodesPerElement == 4 || !embedded.IsEmpty()) {
    mochi::ModelData model;
    model.mesh = processing::ToMochiSpace(input);
    if (embedded.contactSkinMesh.has_value()) {
      model.contactSkinMesh = processing::ToMochiSpace(*embedded.contactSkinMesh);
    }
    return model;
  }
  std::vector<MeshSection> const sections{processing::SectionFromMeshData(input)};
  mochi::CoordinateSpaceConverter const renderToMochi(
      mochi_renderer::RenderSpace(), mochi::CoordinateSpace::Default());
  return MeshSectionsToModel(sections, &renderToMochi);
}

class MochiModelExportMethod : public ReflectedMethod<MochiModelExportProps> {
 public:
  [[nodiscard]] char const* Name() const override {
    return "SDF";
  }
  [[nodiscard]] char const* Description() const override {
    return "Bake a grid SDF from the input mesh and export the mesh plus SDF as a .mochi.h5 (or "
           "the mesh alone, with Bake SDF off).";
  }
  [[nodiscard]] bool ProvidesFileExport() const override {
    return true;
  }
  [[nodiscard]] std::string ExportPath() const override {
    return NormalizeMochiModelExportPath(
        std::string{_props.exportPath.data(), _props.exportPath.size()});
  }
  [[nodiscard]] std::vector<std::string_view> PathPropKeys() const override {
    return {"exportPath"};
  }
  void RefreshAutoExportPath(std::string const& sourceFilePath) override {
    if (_autoExportPath) {
      _props.exportPath = SuggestedExportPath(sourceFilePath);
    }
  }
  void OverrideExportPath(std::string const& path) override {
    _props.exportPath = NormalizeMochiModelExportPath(path);
    _autoExportPath = false;
  }

  // An auto path is reproduced on load, so it is left out of the document entirely; the presence of
  // the key is exactly what marks a path as the user's own (see DeserializeProps).
  void SerializeProps(picojson::value& out) const override {
    ReflectedMethod::SerializeProps(out);
    if (_autoExportPath && out.is<picojson::object>()) {
      out.get<picojson::object>().erase("exportPath");
    }
  }

  int DeserializeProps(picojson::value const& in) override {
    int const issues = ReflectedMethod::DeserializeProps(in);
    // A stored path means the user chose it, even if it happens to equal the current suggestion.
    _autoExportPath = _props.exportPath.empty();
    return issues;
  }

  [[nodiscard]] bool AcceptsTetrahedralInput() const override {
    return true;
  }

  // Writes the meshes embedded in its input (see EmbeddedMeshes) into the model.
  [[nodiscard]] bool AcceptsEmbeddedMeshes() const override {
    return true;
  }

  // Show the input surface (the "Mochi Model") next to the reconstructed SDF surface, matching the
  // Model Viewer's Mochi Model section so the two reads are consistent. The input block is a copy
  // the editor captures at generation (survives reorders); the output block is this method's SDF
  // surface.
  [[nodiscard]] bool ShowsInputVisualization() const override {
    return true;
  }
  [[nodiscard]] char const* InputVisualizationLabel() const override {
    return "Mochi Model Visualization";
  }
  [[nodiscard]] char const* OutputVisualizationLabel() const override {
    return "SDF Visualization";
  }

  void ShowParams(ModifierGuiContext const& gui) override {
    ModifierTooltip const& tooltip = gui.tooltip;
    ImGui::Checkbox("Bake SDF", &_props.bakeSdf);
    tooltip(
        "Bake a signed distance field into the exported model. Off exports the mesh alone; Mochi "
        "then bakes an SDF each time it loads the model.");
    ImGui::BeginDisabled(!_props.bakeSdf);
    if (gui.host) {
      (void)gui.host->ReflectedStruct(
          static_cast<SReflect::StructTypeInfo const&>(
              SReflect::GetTypeInfo<mochi::GridSdfParams>()),
          &_props.sdf);
    }
    ImGui::EndDisabled();
    if (_props.bakeSdf) {
      ImGui::TextUnformatted("Bakes a grid SDF from the input mesh; the output buffer is its");
      ImGui::TextUnformatted("reconstructed surface (preview only).");
    } else {
      ImGui::TextUnformatted("Exports the input mesh alone; the output buffer is the input.");
    }

    ImGui::SeparatorText("Mochi Model Export");
    ImGui::TextUnformatted("Auto");
    ImGui::SameLine();
    if (ImGui::Checkbox("##autoexportpath", &_autoExportPath)) {
      RefreshAutoExportPath(gui.sourceFilePath); // takes effect immediately when switched back on
    }
    tooltip(
        "Keep the output path derived from the source model, and leave it out of the saved "
        "pipeline. Turn this off (or press Browse) to choose the path yourself.");
    ImGui::SameLine();
    ImGui::BeginDisabled(_autoExportPath);
    std::string exportPath{_props.exportPath.data(), _props.exportPath.size()};
    if (ImGui::InputText("Path (.mochi.h5)", &exportPath)) {
      _props.exportPath = exportPath;
    }
    if (ImGui::IsItemDeactivatedAfterEdit()) {
      _props.exportPath = NormalizeMochiModelExportPath(
          std::string{_props.exportPath.data(), _props.exportPath.size()});
    }
    tooltip("Export writes the generated Mochi model to a .mochi.h5 file.");
    ImGui::EndDisabled();
    // Browse stays live while Auto is on: choosing a file IS taking the path over, so it turns Auto
    // off. Cancelling leaves everything as it was -- Browse only ever turns Auto off, never on.
    if (gui.host && ImGui::Button("Browse##sdfexport")) {
      // Filter on "*.h5" (not "*.mochi.h5"): the native save dialog treats the extension as the
      // text after the last dot, so a "*.mochi.h5" filter makes it re-append ".mochi.h5" to a name
      // that already ends in it (doubling the suffix). "*.h5" matches .mochi.h5 without
      // re-appending.
      constexpr std::array<char const*, 1> filters{"*.h5"};
      std::string const defaultPath = processing::ExportDialogStartPath(
          std::string{_props.exportPath.data(), _props.exportPath.size()},
          SuggestedExportPath(gui.sourceFilePath),
          gui.modelFolder);
      // The suggested folder may not exist yet (the asset uses role folders but has no
      // `collision/`); create it up front, because the native dialog silently falls back to its
      // last-used directory when a default's parent is missing, which would defeat suggesting it.
      superdex::robotics::EnsureDirectoriesCreated(
          defaultPath, mochi::ErrorLog{mochi::LogChannel::Warning});
      std::string const chosen = gui.host->SaveFileDialog(
          "Export Mochi Model", filters, "Mochi Model (*.mochi.h5)", defaultPath);
      if (!chosen.empty()) {
        _props.exportPath = NormalizeMochiModelExportPath(chosen);
        _autoExportPath = false;
      }
    }
    tooltip("Choose the .mochi.h5 output path.");
    if (gui.exportPathCollides) {
      ImGui::TextColored(
          ImVec4(1.0f, 0.4f, 0.3f, 1.0f), "Another export writes to this same file.");
      tooltip(
          "Two enabled export modifiers share an output path, so one silently overwrites the "
          "other. Turn Auto off on one of them and give it a path of its own.");
    }
  }

  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    return Bake(input, {}, error);
  }
  StageMesh RunStage(StageMesh const& input, ModifierRunContext const& /*ctx*/, mochi::Error& error)
      const override {
    return {.mesh = Bake(input.mesh, input.embedded, error)};
  }

  // Writes the cached, already-baked Mochi-space model (mesh + SDF from the last Run). Does NOT
  // re-bake; @p input is unused (the cache is authoritative). Export always cascades Generate first
  // (stale-aware), so the cache is current.
  void SaveToFile(mochi::MeshData const& /*input*/, mochi::Error& error) const override {
    std::string const path = ExportPath();
    if (path.empty()) {
      MOCHI_ERROR_SET(error, "Export Mochi Model: set a .mochi.h5 export path first.");
      return;
    }
    if (!_bakedModel.mesh.has_value() || (_props.bakeSdf && !_bakedModel.sdf.has_value())) {
      MOCHI_ERROR_SET(error, "Export Mochi Model: generate the model before exporting.");
      return;
    }
    // The target folder may not exist yet -- a derived collision/ folder in a fresh bot, or one the
    // user typed. Create it rather than failing the write.
    superdex::robotics::EnsureDirectoriesCreated(path, error);
    MOCHI_ERROR_RETURN(error);
    mochi::model_utils::SaveToFile(_bakedModel, path, mochi::FileFormat::H5, error);
  }

  // The in-stack output is the reconstructed surface of the SDF baked by the last Run; report that
  // SDF's grid dimensions in the stats, plus the estimated .mochi.h5 export size (meshes + SDF
  // grid). The editor composites them into the display string.
  void AnnotateStats(MeshStats& stats) const override {
    if (_bakedModel.sdf.has_value()) {
      stats.sdfGrid = _bakedModel.sdf->dims;
    }
    if (_bakedModel.mesh.has_value()) {
      stats.fileSizeBytes = processing::EstimateMochiH5SizeBytes(_bakedModel);
      stats.fileSizeLabel = "Est. Export Size";
    }
  }

 private:
  // Bakes the SDF once, in Mochi (Z-up) space -- what a .mochi.h5 stores -- and caches the baked
  // model, with the meshes embedded in @p input, so Export reuses it without re-baking. The preview
  // surface returned to the stack is converted back to render space.
  mochi::MeshData
  Bake(mochi::MeshData const& input, EmbeddedMeshes const& embedded, mochi::Error& error) const {
    _bakedModel = {};
    mochi::ModelData md = MochiSpaceModel(input, embedded);
    if (!md.mesh.has_value()) {
      MOCHI_ERROR_SET(error, "Export Mochi Model: no input geometry.");
      return {};
    }
    // Render and CAD sources leave triangles unjoined at seams (an STL entirely), which a Mochi
    // model must not be. Embedded meshes refer to the input's nodes as they are, so a mesh carrying
    // them is not welded.
    if (embedded.IsEmpty()) {
      processing::WeldIdenticalVertices(*md.mesh);
    }
    mochi::model_utils::Validate(md, error);
    MOCHI_ERROR_RETURN(error, {});
    if (!_props.bakeSdf) {
      _bakedModel = std::move(md);
      return input;
    }
    mochi::model_utils::BakeSdf(md, _props.sdf, error);
    if (!error.IsOK() || !md.sdf.has_value()) {
      return {};
    }

    mochi::MeshData const surface = mochi::mesh::ReconstructSurfaceFromSdf(*md.sdf, error);
    if (!error.IsOK()) {
      return {};
    }
    // The reconstructed surface is in Mochi space; convert back to render space for the in-stack
    // preview.
    std::vector<MeshSection> previewSections{processing::SectionFromMeshData(surface)};
    mochi::CoordinateSpaceConverter const mochiToRender(
        mochi::CoordinateSpace::Default(), mochi_renderer::RenderSpace());
    mochi_renderer::ConvertMeshSectionsSpace(previewSections, mochiToRender);
    _bakedModel = std::move(md);
    return processing::MeshDataFromSections(previewSections);
  }

  // The path Browse offers, and what RefreshAutoExportPath keeps the field at while Auto is on.
  static std::string SuggestedExportPath(std::string const& sourceFilePath) {
    return processing::DefaultExportPath(
        sourceFilePath, superdex::robotics::kCollisionSubdir, ".mochi.h5");
  }

  // Whether the export path is still derived from the source model rather than chosen by the user.
  // Not serialized: the presence of a stored path is what encodes it (see De/SerializeProps), so a
  // pipeline exporting to the derived location carries no path at all and stays valid when copied
  // into processing_presets/ or onto a machine whose assets live elsewhere. Runtime only.
  bool _autoExportPath = true;
  // Cached Mochi-space mesh + baked SDF produced by the last Run, reused by SaveToFile (runtime
  // only; not serialized).
  mutable mochi::ModelData _bakedModel;
};

} // namespace

ModifierRegistryEntry MakeMochiModelExportEntry() {
  return ModifierRegistryEntry{"Export Mochi Model", ModifierKind::Export, []() {
                                 std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
                                 methods.push_back(std::make_unique<MochiModelExportMethod>());
                                 return methods;
                               }};
}

} // namespace superdex::studio
