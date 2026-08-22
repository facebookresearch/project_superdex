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

#include "meshing/processing_modifiers/source_cad_model.h"

#include "app/app.h" // AssetType, AssetManager, SuperDexStudio
#include "meshing/processing_modifiers/processing_mesh_utils.h" // ApplyTransform, LoadRenderMesh, EndsWithNoCase
#include "ui/imgui_widgets.h" // ImGui::AssetSlot

#include <imguios/imguios.h>

#include <mochi_core/utils/dynamic_string.h>

#include <mochi_mesh/step_mesh_body.h> // MeshStepBody (CAD Mesher path)
#include <mochi_mesh/step_tessellation.h> // TessellateStep (Delabella path)

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace superdex::studio {

// Tessellation backend + edge-sampling mode, reflected as string names for readable JSON.
enum class CadTessellationBackend { Isotropic, Delabella };
enum class CadEdgeSampling { Uniform, Adaptive };

} // namespace superdex::studio

// Enum reflection specializes the global SReflectTypeTraits template, so it must be declared at
// global scope (the fully-qualified names refer back into superdex::studio).
MOCHI_ENUM_BEGIN(superdex::studio::CadTessellationBackend)
MOCHI_ENUM_ITEM(Isotropic)
MOCHI_ENUM_ITEM(Delabella)
MOCHI_ENUM_END()

MOCHI_ENUM_BEGIN(superdex::studio::CadEdgeSampling)
MOCHI_ENUM_ITEM(Uniform)
MOCHI_ENUM_ITEM(Adaptive)
MOCHI_ENUM_END()

namespace superdex::studio {

namespace {

// STEP tessellation parameters. Only consulted when the slotted CAD file is a STEP (.step/.stp); an
// STL is already a triangle mesh and is read directly, ignoring these.
struct CadStepToMeshProps {
  CadTessellationBackend backend = CadTessellationBackend::Isotropic;
  double linearDeflection = 0.05; // mm
  double angularDeflection = 0.25; // rad
  bool combineTouchingSolids = true;
  CadEdgeSampling edgeSampling = CadEdgeSampling::Uniform;
  double targetEdgeLength = 0.0; // mm, 0 = auto
  double targetEdgeLengthFraction = 0.02; // fraction of bbox diagonal
  bool allowPartialFailure = true;

  MOCHI_STRUCT_BEGIN(superdex::studio::CadStepToMeshProps)
  MOCHI_FIELD(backend)
  MOCHI_FIELD(linearDeflection)
  MOCHI_FIELD(angularDeflection)
  MOCHI_FIELD(combineTouchingSolids)
  MOCHI_FIELD(edgeSampling)
  MOCHI_FIELD(targetEdgeLength)
  MOCHI_FIELD(targetEdgeLengthFraction)
  MOCHI_FIELD(allowPartialFailure)
  MOCHI_STRUCT_END()
};

// "From File" parameters: the modifier's own CAD-model slot (independent of the editor's slots)
// plus the STEP tessellation params applied when that slot holds a STEP.
struct CadFileSourceProps {
  CadStepToMeshProps step;
  mochi::DynamicString path;

  MOCHI_STRUCT_BEGIN(superdex::studio::CadFileSourceProps)
  MOCHI_FIELD(step)
  MOCHI_FIELD(path)
  MOCHI_STRUCT_END()
};

bool IsStlPath(std::string const& path) {
  return processing::EndsWithNoCase(path, ".stl");
}

// Produces the source mesh from a CAD file: an STL is read directly as a triangle mesh; a STEP is
// tessellated with @p props (CGAL per-face remesher or OpenCascade's BRepMesh).
//
// Only the STL path converts to renderer space (inside LoadRenderMesh); the STEP calls below need
// no conversion because the superdex_mesh_cli helper already returns renderer-space geometry.
mochi::MeshData
CadSourceMesh(std::string const& path, CadStepToMeshProps const& props, mochi::Error& error) {
  if (IsStlPath(path)) {
    return processing::LoadRenderMesh(path, error);
  }
  if (props.backend == CadTessellationBackend::Delabella) {
    // OpenCascade's standard BRepMesh (Delabella) -- meshes the whole file; deflections only.
    mochi::mesh::StepTessellationParams params;
    params.linearDeflection = props.linearDeflection;
    params.angularDeflection = props.angularDeflection;
    return mochi::mesh::TessellateStep(path, params, error);
  }
  mochi::mesh::StepMeshBodyParams params;
  params.linearDeflection = props.linearDeflection;
  params.angularDeflection = props.angularDeflection;
  params.targetEdgeLength = props.targetEdgeLength;
  params.targetEdgeLengthFraction = props.targetEdgeLengthFraction;
  params.edgeSampling = static_cast<mochi::mesh::StepMeshBodyParams::EdgeSampling>(
      static_cast<int>(props.edgeSampling));
  params.allowPartialFailure = props.allowPartialFailure;
  params.combineTouchingSolids = props.combineTouchingSolids;
  return mochi::mesh::MeshStepBody(path, params, error);
}

// Draws the STEP tessellation controls. Only shown when the slotted file is a STEP; an STL has no
// options.
void ShowStepTessellationParams(CadStepToMeshProps& props, ModifierTooltip const& tooltip) {
  int backend = static_cast<int>(props.backend);
  char const* backendItems[] = {"Isotropic (CGAL)", "Delabella (OCCT)"};
  if (ImGui::Combo("Backend", &backend, backendItems, IM_ARRAYSIZE(backendItems))) {
    props.backend = static_cast<CadTessellationBackend>(backend);
  }
  tooltip(
      "Tessellation backend:\n\n"
      "Isotropic = Uses Constrained Delaunay Triangulation based on uniformly "
      "sampled points along CAD faces. Provides highest quality for simple CAD geometry. Prioritizes "
      "uniform edge length. \n\n"
      "Delabella = OpenCascade's fast BRepMesh which aims for visual fidelity. Remeshing is highly "
      "recommended for physics simulation.");

  if (ImGui::InputDouble("Linear Deflection (mm)", &props.linearDeflection, 0.0, 0.0, "%.3f")) {
    props.linearDeflection = std::clamp(props.linearDeflection, 0.001, 10.0);
  }
  tooltip("Maximum chord deviation between the mesh and the true surface (mm). Smaller = finer.");

  if (ImGui::InputDouble("Angular Deflection (rad)", &props.angularDeflection, 0.0, 0.0, "%.3f")) {
    props.angularDeflection = std::clamp(props.angularDeflection, 0.01, 1.57);
  }
  tooltip("Maximum angle between adjacent facets (rad). Smaller = finer on curved surfaces.");

  // The remaining controls only affect the Isotropic (CGAL) path; Delabella meshes the whole file
  // using only the deflections above.
  if (props.backend != CadTessellationBackend::Isotropic) {
    return;
  }

  ImGui::Checkbox("Combine Touching Solids", &props.combineTouchingSolids);
  tooltip(
      "Attempts to fuse solid bodies that touch into one body before meshing. "
      "Not recommended for complex assemblies with high part counts.");

  int edgeSampling = static_cast<int>(props.edgeSampling);
  char const* edgeSamplingItems[] = {"Uniform", "Adaptive"};
  if (ImGui::Combo(
          "Edge Sampling", &edgeSampling, edgeSamplingItems, IM_ARRAYSIZE(edgeSamplingItems))) {
    props.edgeSampling = static_cast<CadEdgeSampling>(edgeSampling);
  }
  tooltip(
      "How face boundary curves are sampled before triangulation. Both space their samples evenly; "
      "Uniform takes the count from the target edge length alone, Adaptive respects angular deflection "
      "at the cost of shorter edge lengths along curves.\n\n"
      "Use Adaptive to preserve round holes or fillets.");

  if (ImGui::InputDouble(
          "Target Edge Length mm (0=auto)", &props.targetEdgeLength, 0.0, 0.0, "%.4f")) {
    props.targetEdgeLength = std::max(props.targetEdgeLength, 0.0);
  }
  tooltip("Desired triangle edge length (mm). 0 = auto from the bounding-box fraction below.");

  ImGui::BeginDisabled(props.targetEdgeLength != 0.0);
  if (ImGui::InputDouble(
          "Edge Length Fraction", &props.targetEdgeLengthFraction, 0.0, 0.0, "%.4f")) {
    props.targetEdgeLengthFraction = std::clamp(props.targetEdgeLengthFraction, 0.001, 0.5);
  }
  tooltip(
      "Edge length is determined from the fraction of diagonal length of bounding-box of "
      "the input geometry.");
  ImGui::EndDisabled();

  ImGui::Checkbox("Allow Partial Failure", &props.allowPartialFailure);
  tooltip("Output a mesh even if some faces fail to tessellate, instead of aborting.");
}

// ---- From Model Viewer --------------------------------------------------------------------------
class CadFromViewerMethod : public ReflectedMethod<CadStepToMeshProps> {
 public:
  char const* Name() const override {
    return "From Model Viewer";
  }
  char const* Description() const override {
    return "Uses the editor's current CAD Model (its Model Viewer slot) as the source. A STEP file "
           "is tessellated with the options below; an STL is read directly.";
  }
  bool CanGenerate(ModifierRunContext const& ctx) const override {
    return !ctx.cadFilePath.empty();
  }
  std::string SourceFilePath(ModifierRunContext const& ctx) const override {
    return ctx.cadFilePath;
  }
  // Fold the slotted file + baked CAD transform into the signature so changing either re-generates.
  std::string PropsSignature(ModifierRunContext const& ctx) const override {
    mochi::Real4 const r = ctx.cadRotation.ToReal4();
    std::string sig = SReflect::ToJsonString(_props, /*pretty=*/false);
    sig += '|';
    sig += ctx.cadFilePath;
    sig += '|';
    for (int i = 0; i < 3; ++i) {
      sig += std::to_string(ctx.cadScale[i]) + ',';
    }
    for (int i = 0; i < 4; ++i) {
      sig += std::to_string(r[i]) + ',';
    }
    for (int i = 0; i < 3; ++i) {
      sig += std::to_string(ctx.cadTranslation[i]) + ',';
    }
    return sig;
  }

  void ShowParams(ModifierGuiContext const& gui) override {
    ImGui::TextUnformatted("Uses the editor's current CAD Model as the source.");
    if (gui.cadFilePath.empty()) {
      ImGui::TextDisabled("Slot a CAD Model (STEP or STL) in the Model Viewer.");
      return;
    }
    if (IsStlPath(gui.cadFilePath)) {
      ImGui::TextDisabled("STL: used directly; no tessellation options.");
      return;
    }
    ShowStepTessellationParams(_props, gui.tooltip);
  }

  mochi::MeshData Run(
      mochi::MeshData const& /*input*/,
      ModifierRunContext const& ctx,
      mochi::Error& error) const override {
    MOCHI_ERROR_IF(
        ctx.cadFilePath.empty(), error, "Source from CAD Model: no CAD file is slotted.");
    MOCHI_ERROR_RETURN(error, {});
    mochi::MeshData mesh = CadSourceMesh(ctx.cadFilePath, _props, error);
    if (!error.IsOK()) {
      return {};
    }
    processing::ApplyTransform(mesh, ctx.cadScale, ctx.cadRotation, ctx.cadTranslation);
    return mesh;
  }
};

// ---- From File ----------------------------------------------------------------------------------
class CadFromFileMethod : public ReflectedMethod<CadFileSourceProps> {
 public:
  char const* Name() const override {
    return "From File";
  }
  char const* Description() const override {
    return "Uses a CAD Model from this modifier's own slot, independent of the editor's Model "
           "Viewer. A STEP file is tessellated with the options below; an STL is read directly.";
  }
  bool CanGenerate(ModifierRunContext const& /*ctx*/) const override {
    return !_props.path.empty();
  }
  std::string SourceFilePath(ModifierRunContext const& /*ctx*/) const override {
    return std::string{_props.path.data(), _props.path.size()};
  }
  std::vector<std::string_view> PathPropKeys() const override {
    return {"path"};
  }
  void ShowParams(ModifierGuiContext const& gui) override {
    if (gui.studio == nullptr || gui.assetManager == nullptr) {
      return;
    }
    ImGui::AssetSlot(
        "##cadsocket",
        _props.path,
        *gui.assetManager,
        gui.studio,
        AssetType::CadModel,
        /*acceptDragDropPayload=*/true);
    gui.tooltip(
        "Drop a CAD Model here (a STEP/STL in the 'cad' folder). This slot is independent of the "
        "editor's own Model Viewer slots.");
    std::string const path{_props.path.data(), _props.path.size()};
    if (path.empty()) {
      return;
    }
    if (IsStlPath(path)) {
      ImGui::TextDisabled("STL: used directly; no tessellation options.");
      return;
    }
    ShowStepTessellationParams(_props.step, gui.tooltip);
  }
  mochi::MeshData Run(
      mochi::MeshData const& /*input*/,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    std::string const path{_props.path.data(), _props.path.size()};
    MOCHI_ERROR_IF(path.empty(), error, "Source from CAD Model: no CAD file is set.");
    MOCHI_ERROR_RETURN(error, {});
    return CadSourceMesh(path, _props.step, error);
  }
};

} // namespace

ModifierRegistryEntry MakeCadModelSourceEntry() {
  return ModifierRegistryEntry{"Source from CAD Model", ModifierKind::Source, []() {
                                 std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
                                 methods.push_back(std::make_unique<CadFromViewerMethod>());
                                 methods.push_back(std::make_unique<CadFromFileMethod>());
                                 return methods;
                               }};
}

} // namespace superdex::studio
