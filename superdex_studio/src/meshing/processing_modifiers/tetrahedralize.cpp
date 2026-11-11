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

#include "meshing/processing_modifiers/tetrahedralize.h"

#include "meshing/processing_modifiers/remesh_params_ui.h" // ShowEdgeSizeControl, ShowLengthControl, ShowFeatureControls

#include <imgui.h>

#include <mochi_mesh/tetrahedralization.h> // Tetrahedralize / TetrahedralizeConstrained + params

#include <memory>
#include <vector>

namespace superdex::studio {

namespace {

constexpr mochi::mesh::TetrahedralizationParams kTetDefaults{};

// Reflected mirror of mochi::mesh::TetrahedralizationParams (which is not reflected). Defaults come
// from the neutral struct so the two cannot drift.
struct TetrahedralizeProps {
  double edgeSize = kTetDefaults.edgeSize;
  bool relativeToMeshSize = kTetDefaults.relativeToMeshSize;
  double facetDistanceBound = kTetDefaults.facetDistanceBound;
  double facetAngleBound = kTetDefaults.facetAngleBound;
  double cellRadiusEdgeRatio = kTetDefaults.cellRadiusEdgeRatio;
  bool detectFeatures = kTetDefaults.detectFeatures;
  double sharpFeatureAngle = kTetDefaults.sharpFeatureAngle;
  bool perturb = kTetDefaults.perturb;
  bool exude = kTetDefaults.exude;
  bool odt = kTetDefaults.odt;

  MOCHI_STRUCT_BEGIN(superdex::studio::TetrahedralizeProps)
  MOCHI_FIELD(edgeSize)
  MOCHI_FIELD(relativeToMeshSize)
  MOCHI_FIELD(facetDistanceBound)
  MOCHI_FIELD(facetAngleBound)
  MOCHI_FIELD(cellRadiusEdgeRatio)
  MOCHI_FIELD(detectFeatures)
  MOCHI_FIELD(sharpFeatureAngle)
  MOCHI_FIELD(perturb)
  MOCHI_FIELD(exude)
  MOCHI_FIELD(odt)
  MOCHI_STRUCT_END()
};

class DelaunayRefinementMethod : public ReflectedMethod<TetrahedralizeProps> {
 public:
  [[nodiscard]] char const* Name() const override {
    return "Delaunay Refinement";
  }
  [[nodiscard]] char const* Description() const override {
    return "Fill the closed input surface with tetrahedra by Delaunay refinement (CGAL Mesh_3).";
  }

  void ShowParams(ModifierGuiContext const& gui) override {
    ModifierTooltip const& tooltip = gui.tooltip;
    ImGui::TextUnformatted("The input must be closed and free of self-intersections (e.g. run");
    ImGui::TextUnformatted("Wrap Mesh first). The output is a tetrahedral mesh.");
    processing::ShowEdgeSizeControl(_props.edgeSize, _props.relativeToMeshSize, tooltip);
    processing::ShowLengthControl(
        "Facet Distance (0=auto)",
        "Facet-to-surface distance bound (0 automatically uses Edge Size / 5), expressed",
        _props.facetDistanceBound,
        _props.relativeToMeshSize,
        tooltip);
    ImGui::InputDouble("Min Facet Angle (deg)", &_props.facetAngleBound, 0.0, 0.0, "%.4f");
    tooltip(
        "Lower bound on boundary triangle angles (max 30). Higher = better boundary triangles.");
    ImGui::InputDouble("Max Radius-Edge Ratio", &_props.cellRadiusEdgeRatio, 0.0, 0.0, "%.4f");
    tooltip(
        "Upper bound on each tetrahedron's circumradius-to-shortest-edge ratio (at least 2). Lower "
        "= better-shaped tetrahedra, but more of them.");
    processing::ShowFeatureControls(_props.detectFeatures, _props.sharpFeatureAngle, tooltip);
    ImGui::Checkbox("Perturb Slivers", &_props.perturb);
    tooltip("Remove flat (sliver) tetrahedra by perturbing vertices.");
    ImGui::Checkbox("Exude Slivers", &_props.exude);
    tooltip("Remove flat (sliver) tetrahedra by sliver exudation.");
    ImGui::Checkbox("ODT Smoothing", &_props.odt);
    tooltip("Smooth the mesh by Optimal Delaunay Triangulation. Slower.");
  }

  // Meshes in the pipeline's renderer space; the conversion to Mochi space at export is a rotation,
  // so sizes and tetrahedron orientation carry over.
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    mochi::mesh::TetrahedralizationParams p;
    p.edgeSize = _props.edgeSize;
    p.relativeToMeshSize = _props.relativeToMeshSize;
    p.facetDistanceBound = _props.facetDistanceBound;
    p.facetAngleBound = _props.facetAngleBound;
    p.cellRadiusEdgeRatio = _props.cellRadiusEdgeRatio;
    p.detectFeatures = _props.detectFeatures;
    p.sharpFeatureAngle = _props.sharpFeatureAngle;
    p.perturb = _props.perturb;
    p.exude = _props.exude;
    p.odt = _props.odt;
    return mochi::mesh::Tetrahedralize(input, p, error);
  }
};

constexpr mochi::mesh::ConstrainedTetrahedralizationParams kConstrainedTetDefaults{};

struct ConstrainedTetrahedralizeProps {
  double edgeSize = kConstrainedTetDefaults.edgeSize;
  bool relativeToMeshSize = kConstrainedTetDefaults.relativeToMeshSize;
  int iterations = kConstrainedTetDefaults.iterations;
  bool strict = kConstrainedTetDefaults.strict;

  MOCHI_STRUCT_BEGIN(superdex::studio::ConstrainedTetrahedralizeProps)
  MOCHI_FIELD(edgeSize)
  MOCHI_FIELD(relativeToMeshSize)
  MOCHI_FIELD(iterations)
  MOCHI_FIELD(strict)
  MOCHI_STRUCT_END()
};

class ConstrainedDelaunayMethod : public ReflectedMethod<ConstrainedTetrahedralizeProps> {
 public:
  [[nodiscard]] char const* Name() const override {
    return "Constrained Delaunay";
  }
  [[nodiscard]] char const* Description() const override {
    return "Fill the closed input surface with tetrahedra, keeping it as the mesh boundary (CGAL "
           "3D constrained Delaunay triangulation and tetrahedral remeshing).";
  }

  void ShowParams(ModifierGuiContext const& gui) override {
    ModifierTooltip const& tooltip = gui.tooltip;
    ImGui::TextUnformatted("The input must be closed and free of self-intersections, and already");
    ImGui::TextUnformatted("meshed at the resolution wanted on the boundary, which is kept.");
    processing::ShowEdgeSizeControl(_props.edgeSize, _props.relativeToMeshSize, tooltip);
    ImGui::InputInt("Iterations", &_props.iterations);
    tooltip(
        "Passes refining the interior (at least 1). More passes may improve tetrahedron shape, "
        "but take longer.");
    ImGui::Checkbox("Strict Boundary", &_props.strict);
    tooltip(
        "Fail unless the boundary triangles stay exactly the input triangles. Off allows splitting "
        "input triangles where constrained Delaunay construction requires extra boundary vertices.");
  }

  // Meshes in the pipeline's renderer space, as DelaunayRefinementMethod does.
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    mochi::mesh::ConstrainedTetrahedralizationParams p;
    p.edgeSize = _props.edgeSize;
    p.relativeToMeshSize = _props.relativeToMeshSize;
    p.iterations = _props.iterations;
    p.strict = _props.strict;
    return mochi::mesh::TetrahedralizeConstrained(input, p, error);
  }
};

} // namespace

ModifierRegistryEntry MakeTetrahedralizeEntry() {
  return ModifierRegistryEntry{"Tetrahedralize", ModifierKind::Transform, []() {
                                 std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
                                 methods.push_back(std::make_unique<DelaunayRefinementMethod>());
                                 methods.push_back(std::make_unique<ConstrainedDelaunayMethod>());
                                 return methods;
                               }};
}

} // namespace superdex::studio
