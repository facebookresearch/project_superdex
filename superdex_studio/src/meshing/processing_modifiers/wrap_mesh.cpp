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

#include "meshing/processing_modifiers/wrap_mesh.h"

#include "meshing/processing_modifiers/remesh_params_ui.h"

#include <imguios/imguios.h>

#include <mochi_mesh/step_mesh_stages.h> // CloseMesh / MeshClosureParams / MeshClosureMode
#include <mochi_mesh/surface_remeshing.h> // RemeshSurface / SurfaceRemeshingParams

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace superdex::studio {

namespace {

// --- reflected per-method parameter structs (defaults mirror SurfaceRemeshingParams) -------------
struct WrapShrinkWrapProps {
  double tightness = 1.0;
  bool snap = true;
  double targetEdgeLength = 0.0; // mm, 0 = auto
  double targetEdgeLengthFraction = 0.02; // fraction of bbox diagonal

  MOCHI_STRUCT_BEGIN(superdex::studio::WrapShrinkWrapProps)
  MOCHI_FIELD(tightness)
  MOCHI_FIELD(snap)
  MOCHI_FIELD(targetEdgeLength)
  MOCHI_FIELD(targetEdgeLengthFraction)
  MOCHI_STRUCT_END()
};

struct WrapAlphaWrapProps {
  double relativeAlpha = 1.0; // SurfaceRemeshingParams::alphaWrapRelativeAlpha
  double relativeOffset = 0.0; // SurfaceRemeshingParams::alphaWrapRelativeOffset
  double edgeSize = 0.025; // SurfaceRemeshingParams::edgeSize
  bool relativeToMeshSize = true; // SurfaceRemeshingParams::relativeToMeshSize

  MOCHI_STRUCT_BEGIN(superdex::studio::WrapAlphaWrapProps)
  MOCHI_FIELD(relativeAlpha)
  MOCHI_FIELD(relativeOffset)
  MOCHI_FIELD(edgeSize)
  MOCHI_FIELD(relativeToMeshSize)
  MOCHI_STRUCT_END()
};

// ---- Shrink Wrap --------------------------------------------------------------------------------
class ShrinkWrapMethod : public ReflectedMethod<WrapShrinkWrapProps> {
 public:
  char const* Name() const override {
    return "Shrink Wrap";
  }
  char const* Description() const override {
    return "Build a fresh watertight envelope by shrinking a hull onto the surface, bridging "
           "concavities and covering holes.";
  }
  void ShowParams(ModifierGuiContext const& gui) override {
    ModifierTooltip const& tooltip = gui.tooltip;
    auto tightness = static_cast<float>(_props.tightness);
    if (ImGui::SliderFloat("Tightness", &tightness, 0.0f, 1.0f, "%.2f")) {
      _props.tightness = tightness;
    }
    tooltip(
        "Shrink-wrap tightness: low = near the convex hull; 1 = deflated onto the surface's outer "
        "envelope (concavities bridged, holes covered).");
    ImGui::Checkbox("Snap to Surface", &_props.snap);
    tooltip("Final pass snaps wrap vertices onto the original surface for a closer fit.");
    if (ImGui::InputDouble(
            "Target Edge Length mm (0=auto)", &_props.targetEdgeLength, 0.0, 0.0, "%.4f")) {
      _props.targetEdgeLength = std::max(_props.targetEdgeLength, 0.0);
    }
    tooltip("Desired triangle edge length (mm). 0 = auto from the bounding-box fraction below.");
    ImGui::BeginDisabled(_props.targetEdgeLength != 0.0);
    if (ImGui::InputDouble(
            "Edge Length Fraction", &_props.targetEdgeLengthFraction, 0.0, 0.0, "%.4f")) {
      _props.targetEdgeLengthFraction = std::clamp(_props.targetEdgeLengthFraction, 0.001, 0.5);
    }
    tooltip("When target edge length is 0, the target is this fraction of the bbox diagonal.");
    ImGui::EndDisabled();
  }
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    namespace mm = mochi::mesh;
    mm::MeshClosureParams cp;
    cp.mode = mm::MeshClosureMode::ShrinkWrap;
    cp.shrinkWrapTightness = _props.tightness;
    cp.shrinkWrapSnap = _props.snap;
    cp.shrinkWrapTargetEdgeLength = _props.targetEdgeLength;
    cp.shrinkWrapTargetEdgeLengthFraction = _props.targetEdgeLengthFraction;
    return mm::CloseMesh(input, cp, error);
  }
};

// ---- Convex Hull --------------------------------------------------------------------------------
class ConvexHullMethod : public ReflectedMethod<EmptyMethodProps> {
 public:
  char const* Name() const override {
    return "Convex Hull";
  }
  char const* Description() const override {
    return "Replace the surface with its convex hull (remeshed).";
  }
  void ShowParams(ModifierGuiContext const& /*gui*/) override {
    ImGui::TextUnformatted("Replace the surface with its convex hull (remeshed).");
    ImGui::TextUnformatted("No parameters.");
  }
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    namespace mm = mochi::mesh;
    mm::MeshClosureParams cp;
    cp.mode = mm::MeshClosureMode::ConvexHull;
    return mm::CloseMesh(input, cp, error);
  }
};

// ---- Alpha Wrap ---------------------------------------------------------------------------------
class AlphaWrapMethod : public ReflectedMethod<WrapAlphaWrapProps> {
 public:
  char const* Name() const override {
    return "Alpha Wrap";
  }
  char const* Description() const override {
    return "Build a fresh watertight envelope by alpha-wrapping the surface; alpha sets the finest "
           "captured detail.";
  }
  void ShowParams(ModifierGuiContext const& gui) override {
    ModifierTooltip const& tooltip = gui.tooltip;
    ImGui::InputDouble("Alpha (x Edge Size)", &_props.relativeAlpha, 0.0, 0.0, "%.4f");
    tooltip(
        "Alpha as a multiple of edge size. Smaller alpha captures finer detail (more triangles); it "
        "is the detail floor for everything downstream.");
    ImGui::InputDouble("Offset (x Edge Size, 0=auto)", &_props.relativeOffset, 0.0, 0.0, "%.4f");
    tooltip("Offset distance from the surface as a multiple of edge size. 0 = automatic.");
    processing::ShowEdgeSizeControl(_props.edgeSize, _props.relativeToMeshSize, tooltip);
  }
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    namespace mm = mochi::mesh;
    mm::SurfaceRemeshingParams p; // fresh defaults; override only the alpha-wrap fields
    p.method = mm::RemeshMethod::AlphaWrap;
    p.alphaWrapRelativeAlpha = _props.relativeAlpha;
    p.alphaWrapRelativeOffset = _props.relativeOffset;
    p.edgeSize = _props.edgeSize;
    p.relativeToMeshSize = _props.relativeToMeshSize;
    // "Primary only": no isotropic post-passes (matches the former Closure/Resample behavior).
    p.smoothingIterations = 0;
    p.tangentialRelaxationIterations = 0;
    p.angleSmoothingIterations = 0;
    p.repairMesh = true;
    return mm::RemeshSurface(input, p, error);
  }
};

} // namespace

ModifierRegistryEntry MakeWrapMeshEntry() {
  return ModifierRegistryEntry{"Wrap Mesh", ModifierKind::Transform, []() {
                                 std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
                                 methods.push_back(std::make_unique<ShrinkWrapMethod>());
                                 methods.push_back(std::make_unique<ConvexHullMethod>());
                                 methods.push_back(std::make_unique<AlphaWrapMethod>());
                                 return methods;
                               }};
}

} // namespace superdex::studio
