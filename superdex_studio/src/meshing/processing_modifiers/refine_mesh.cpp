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

#include "meshing/processing_modifiers/refine_mesh.h"

#include "meshing/processing_modifiers/remesh_params_ui.h" // ShowReferenceCombo

#include <imgui.h>

#include <mochi_mesh/step_mesh_stages.h> // CleanupMesh / CloseMesh / EdgeSwapMesh / DecimateMesh
#include <mochi_mesh/surface_remeshing.h> // RemeshSurface / SurfaceRemeshingParams

#include <algorithm>
#include <memory>
#include <vector>

namespace superdex::studio {

namespace {

// GUI distance thresholds are entered in millimeters; processed meshes are in meters.
constexpr double kMillimetersToMeters = 0.001;

// --- reflected per-method parameter structs (only the user-facing fields; kept minimal so the JSON
// is small + human-editable) ----------------------------------------------------------------------

struct RefineMakeManifoldProps {
  bool removeAlmostDegenerateFaces = true;
  bool thenMakeWatertight = false;

  MOCHI_STRUCT_BEGIN(superdex::studio::RefineMakeManifoldProps)
  MOCHI_FIELD(removeAlmostDegenerateFaces)
  MOCHI_FIELD(thenMakeWatertight)
  MOCHI_STRUCT_END()
};

struct RefineEdgeFlipProps {
  int referenceIndex =
      MeshProcessingMethod::kReferencePrecedingSource; // -1 = nearest preceding source, else index
  double relativeThreshold = 0.1;

  MOCHI_STRUCT_BEGIN(superdex::studio::RefineEdgeFlipProps)
  MOCHI_FIELD(referenceIndex)
  MOCHI_FIELD(relativeThreshold)
  MOCHI_STRUCT_END()
};

struct RefineEdgeCollapseProps {
  double collapseDistanceMm = 0.0;

  MOCHI_STRUCT_BEGIN(superdex::studio::RefineEdgeCollapseProps)
  MOCHI_FIELD(collapseDistanceMm)
  MOCHI_STRUCT_END()
};

struct RefineTangentialProps {
  int tangentialRelaxationIterations = 3; // matches SurfaceRemeshingParams' default
  int angleSmoothingIterations = 0;

  MOCHI_STRUCT_BEGIN(superdex::studio::RefineTangentialProps)
  MOCHI_FIELD(tangentialRelaxationIterations)
  MOCHI_FIELD(angleSmoothingIterations)
  MOCHI_STRUCT_END()
};

// ---- Make Manifold ------------------------------------------------------------------------------
class MakeManifoldMethod : public ReflectedMethod<RefineMakeManifoldProps> {
 public:
  [[nodiscard]] char const* Name() const override {
    return "Make Manifold";
  }
  [[nodiscard]] char const* Description() const override {
    return "Weld coincident vertices, remove internal/overlapping faces, and repair non-manifold "
           "topology (optionally filling holes afterward).";
  }
  void ShowParams(ModifierGuiContext const& gui) override {
    ImGui::TextUnformatted("Weld coincident vertices, remove internal/overlapping faces,");
    ImGui::TextUnformatted("and repair non-manifold topology.");
    ImGui::Checkbox("Remove Almost-Degenerate Faces", &_props.removeAlmostDegenerateFaces);
    gui.tooltip(
        "Collapse needle triangles (longest edge over 4x the shortest) and flip caps (an angle over "
        "160 degrees). Tidies thin slivers, but merges vertices regardless of how far apart they "
        "are; turn it off to keep the input's vertices in place.");
    ImGui::Checkbox("Then Make Watertight", &_props.thenMakeWatertight);
    gui.tooltip(
        "Also fill any remaining holes so the result is a closed manifold, not a potentially open "
        "one.");
  }
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    namespace mm = mochi::mesh;
    mm::MeshCleanupParams params;
    params.removeAlmostDegenerateFaces = _props.removeAlmostDegenerateFaces;
    mochi::MeshData manifold = mm::CleanupMesh(input, params, error);
    if (!error.IsOK() || !_props.thenMakeWatertight) {
      return manifold;
    }
    // Convenience: also fill holes so an input with boundaries becomes a closed manifold.
    mm::MeshClosureParams cp;
    cp.mode = mm::MeshClosureMode::FillHoles;
    return mm::CloseMesh(manifold, cp, error);
  }
};

// ---- Make Watertight ----------------------------------------------------------------------------
class MakeWatertightMethod : public ReflectedMethod<EmptyMethodProps> {
 public:
  [[nodiscard]] char const* Name() const override {
    return "Make Watertight";
  }
  [[nodiscard]] char const* Description() const override {
    return "Fill boundary holes so the surface is closed (watertight).";
  }
  void ShowParams(ModifierGuiContext const& /*gui*/) override {
    ImGui::TextUnformatted("Fill boundary holes so the surface is closed (watertight).");
    ImGui::TextUnformatted("No parameters.");
  }
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    namespace mm = mochi::mesh;
    mm::MeshClosureParams cp;
    cp.mode = mm::MeshClosureMode::FillHoles;
    return mm::CloseMesh(input, cp, error);
  }
};

// ---- Edge Flip Optimization ---------------------------------------------------------------------
class EdgeFlipMethod : public ReflectedMethod<RefineEdgeFlipProps> {
 public:
  [[nodiscard]] char const* Name() const override {
    return "Edge Flip Optimization";
  }
  [[nodiscard]] char const* Description() const override {
    return "Retriangulate by flipping edges to better fit an upstream reference surface, leaving "
           "vertex positions unchanged.";
  }
  [[nodiscard]] bool NeedsReferenceMesh() const override {
    return true;
  }
  [[nodiscard]] int ReferenceIndex() const override {
    return _props.referenceIndex;
  }
  void RemapReferences(std::vector<int> const& oldToNew) override {
    RemapReferenceIndex(_props.referenceIndex, oldToNew);
  }

  void ShowParams(ModifierGuiContext const& gui) override {
    ModifierTooltip const& tooltip = gui.tooltip;

    // While the reference is invalid the edge flip fails generation (no output), which propagates
    // downstream (see Run + ReferenceModifierIndex).
    processing::ShowReferenceCombo("Reference", _props.referenceIndex, gui);
    tooltip(
        "The upstream mesh this edge flip fits toward. 'Preceding Source' uses the nearest source "
        "above; otherwise pick a specific element. '(invalid source)' means this modifier was moved "
        "above its reference -- pick a valid one (it produces no output until then).");

    if (ImGui::InputDouble(
            "Edge Flip Threshold (rel.)", &_props.relativeThreshold, 0.0, 0.0, "%.3f")) {
      _props.relativeThreshold = std::clamp(_props.relativeThreshold, 0.0, 1.0);
    }
    tooltip(
        "Flip an edge only when the alternate diagonal reduces the surface-fit error against the "
        "reference mesh by at least this fraction of the local edge length (0.1 = 10%). Higher = "
        "fewer, only clearly-beneficial flips.");
  }

  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& ctx,
      mochi::Error& error) const override {
    // No reference mesh means the reference is invalid (this modifier was moved above it) or
    // missing (no preceding source). Fail with no output so the failure propagates down the chain;
    // the editor shows "(invalid source)" in the dropdown for this state.
    if (ctx.referenceMesh.GetNumElements() == 0) {
      MOCHI_ERROR_SET(
          error,
          "Edge Flip Optimization: no valid reference mesh (its reference is invalid or missing).");
      return {};
    }
    mochi::mesh::MeshEdgeSwapParams params;
    params.relativeThreshold = _props.relativeThreshold;
    params.maxPasses = 10;
    return mochi::mesh::EdgeSwapMesh(input, ctx.referenceMesh, params, error);
  }
};

// ---- Edge Collapse (Simplification) -------------------------------------------------------------
class EdgeCollapseMethod : public ReflectedMethod<RefineEdgeCollapseProps> {
 public:
  [[nodiscard]] char const* Name() const override {
    return "Edge Collapse (Simplification)";
  }
  [[nodiscard]] char const* Description() const override {
    return "Decimate the mesh by collapsing edges shorter than a distance threshold.";
  }
  void ShowParams(ModifierGuiContext const& gui) override {
    if (ImGui::InputDouble(
            "Collapse Distance mm (0=off)", &_props.collapseDistanceMm, 0.0, 0.0, "%.5f")) {
      _props.collapseDistanceMm = std::max(_props.collapseDistanceMm, 0.0);
    }
    gui.tooltip("Collapse edges shorter than this distance (mm) to decimate the mesh. 0 = off.");
  }
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    mochi::mesh::MeshDecimateParams dp;
    dp.collapseDistance = _props.collapseDistanceMm * kMillimetersToMeters;
    return mochi::mesh::DecimateMesh(input, dp, error);
  }
};

// ---- Tangential Relaxation (Smoothing) ----------------------------------------------------------
class TangentialRelaxationMethod : public ReflectedMethod<RefineTangentialProps> {
 public:
  [[nodiscard]] char const* Name() const override {
    return "Tangential Relaxation (Smoothing)";
  }
  [[nodiscard]] char const* Description() const override {
    return "Slide vertices along the surface to even out triangle quality without changing "
           "connectivity.";
  }
  void ShowParams(ModifierGuiContext const& gui) override {
    ModifierTooltip const& tooltip = gui.tooltip;
    ImGui::InputInt("Tangential Relaxation Iters", &_props.tangentialRelaxationIterations, 0, 0);
    tooltip(
        "Slides vertices along the surface to even out triangle quality without changing "
        "connectivity. Feature-constrained. 0 = pass-through.");
    ImGui::InputInt("Angle Smoothing Iters (legacy)", &_props.angleSmoothingIterations, 0, 0);
    tooltip("Legacy angle-based smoothing; only used when Tangential Relaxation Iters is 0.");
  }
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    mochi::mesh::SurfaceRemeshingParams p;
    p.method = mochi::mesh::RemeshMethod::None; // relaxation-only post-processing
    p.smoothingIterations = 0;
    p.tangentialRelaxationIterations = _props.tangentialRelaxationIterations;
    p.angleSmoothingIterations = _props.angleSmoothingIterations;
    p.repairMesh = false;
    return mochi::mesh::RemeshSurface(input, p, error);
  }
};

} // namespace

ModifierRegistryEntry MakeRefineMeshEntry() {
  return ModifierRegistryEntry{"Refine Mesh", ModifierKind::Transform, []() {
                                 std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
                                 methods.push_back(std::make_unique<MakeManifoldMethod>());
                                 methods.push_back(std::make_unique<MakeWatertightMethod>());
                                 methods.push_back(std::make_unique<EdgeFlipMethod>());
                                 methods.push_back(std::make_unique<EdgeCollapseMethod>());
                                 methods.push_back(std::make_unique<TangentialRelaxationMethod>());
                                 return methods;
                               }};
}

} // namespace superdex::studio
