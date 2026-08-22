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

#include "meshing/processing_modifiers/remesh.h"

#include "meshing/processing_modifiers/remesh_params_ui.h"

#include <imguios/imguios.h>

#include <mochi_mesh/step_mesh_stages.h> // CloseMesh / MeshClosureParams / MeshClosureMode
#include <mochi_mesh/surface_remeshing.h> // RemeshSurface / SurfaceRemeshingParams

#include <memory>
#include <string>
#include <vector>

namespace superdex::studio {

namespace {

// --- reflected per-method parameter structs (defaults mirror SurfaceRemeshingParams) -------------
struct RemeshAcvdProps {
  int targetVertexCount = 0;
  double acvdGradationFactor = 1.5;
  double edgeSize = 0.025;
  bool relativeToMeshSize = true;

  MOCHI_STRUCT_BEGIN(superdex::studio::RemeshAcvdProps)
  MOCHI_FIELD(targetVertexCount)
  MOCHI_FIELD(acvdGradationFactor)
  MOCHI_FIELD(edgeSize)
  MOCHI_FIELD(relativeToMeshSize)
  MOCHI_STRUCT_END()
};

struct RemeshSurfaceDelaunayProps {
  double facetAngleBound = 25.0;
  double facetDistanceBound = 0.0;
  double edgeSize = 0.025;
  bool relativeToMeshSize = true;
  bool detectFeatures = true;
  double sharpFeatureAngle = 60.0;

  MOCHI_STRUCT_BEGIN(superdex::studio::RemeshSurfaceDelaunayProps)
  MOCHI_FIELD(facetAngleBound)
  MOCHI_FIELD(facetDistanceBound)
  MOCHI_FIELD(edgeSize)
  MOCHI_FIELD(relativeToMeshSize)
  MOCHI_FIELD(detectFeatures)
  MOCHI_FIELD(sharpFeatureAngle)
  MOCHI_STRUCT_END()
};

struct RemeshIsotropicProps {
  double edgeSize = 0.025;
  bool relativeToMeshSize = true;
  bool detectFeatures = true;
  double sharpFeatureAngle = 60.0;
  int smoothingIterations = 3;
  int relaxationStepsPerIteration = 3;
  bool protectConstraints = false;
  bool relaxConstraints = false;
  bool useAdaptiveSizing = false;
  double adaptiveSizingTolerance = 0.01;
  double minEdgeSizeFactor = 0.25;
  double maxEdgeSizeFactor = 2.0;

  MOCHI_STRUCT_BEGIN(superdex::studio::RemeshIsotropicProps)
  MOCHI_FIELD(edgeSize)
  MOCHI_FIELD(relativeToMeshSize)
  MOCHI_FIELD(detectFeatures)
  MOCHI_FIELD(sharpFeatureAngle)
  MOCHI_FIELD(smoothingIterations)
  MOCHI_FIELD(relaxationStepsPerIteration)
  MOCHI_FIELD(protectConstraints)
  MOCHI_FIELD(relaxConstraints)
  MOCHI_FIELD(useAdaptiveSizing)
  MOCHI_FIELD(adaptiveSizingTolerance)
  MOCHI_FIELD(minEdgeSizeFactor)
  MOCHI_FIELD(maxEdgeSizeFactor)
  MOCHI_STRUCT_END()
};

// Fill holes so ACVD / Surface Delaunay get a clean, closed input (they require it).
mochi::MeshData FillHolesFirst(mochi::MeshData const& input, mochi::Error& error) {
  mochi::mesh::MeshClosureParams cp;
  cp.mode = mochi::mesh::MeshClosureMode::FillHoles;
  return mochi::mesh::CloseMesh(input, cp, error);
}

// ---- Approx Centroidal Voronoi Diagram (ACVD) ---------------------------------------------------
class AcvdMethod : public ReflectedMethod<RemeshAcvdProps> {
 public:
  char const* Name() const override {
    return "Approx Centroidal Voronoi Diagram (ACVD)";
  }
  char const* Description() const override {
    return "Resample the surface toward a target vertex count via Approximate Centroidal Voronoi "
           "Diagram clustering (fills holes first).";
  }
  void ShowParams(ModifierGuiContext const& gui) override {
    ModifierTooltip const& tooltip = gui.tooltip;
    ImGui::InputInt("Target Vertex Count (0=auto)", &_props.targetVertexCount, 0, 0);
    tooltip("Target output vertex count. 0 = auto-estimate from edge size and surface area.");
    ImGui::InputDouble("Gradation Factor", &_props.acvdGradationFactor, 0.0, 0.0, "%.4f");
    tooltip("Curvature-based vertex-density variation. 0 = uniform; 1-2 = adaptive.");
    processing::ShowEdgeSizeControl(_props.edgeSize, _props.relativeToMeshSize, tooltip);
  }
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    namespace mm = mochi::mesh;
    mochi::MeshData const filled = FillHolesFirst(input, error);
    if (!error.IsOK()) {
      return {};
    }
    mm::SurfaceRemeshingParams p;
    p.method = mm::RemeshMethod::ACVD;
    p.targetVertexCount = _props.targetVertexCount;
    p.acvdGradationFactor = _props.acvdGradationFactor;
    p.edgeSize = _props.edgeSize;
    p.relativeToMeshSize = _props.relativeToMeshSize;
    p.smoothingIterations = 0;
    p.tangentialRelaxationIterations = 0;
    p.angleSmoothingIterations = 0;
    p.repairMesh = true;
    return mm::RemeshSurface(filled, p, error);
  }
};

// ---- Surface Delaunay Triangulation -------------------------------------------------------------
class SurfaceDelaunayMethod : public ReflectedMethod<RemeshSurfaceDelaunayProps> {
 public:
  char const* Name() const override {
    return "Surface Delaunay Triangulation";
  }
  char const* Description() const override {
    return "Resample the surface with Delaunay refinement bounded by facet angle and distance "
           "(fills holes first).";
  }
  void ShowParams(ModifierGuiContext const& gui) override {
    ModifierTooltip const& tooltip = gui.tooltip;
    ImGui::InputDouble("Min Facet Angle (deg)", &_props.facetAngleBound, 0.0, 0.0, "%.4f");
    tooltip("Lower bound on output triangle angles (max 30). Higher = better triangles.");
    ImGui::InputDouble("Facet Distance (0=auto)", &_props.facetDistanceBound, 0.0, 0.0, "%.6f");
    tooltip(
        "Max distance from a facet center to the surface (fraction of bbox when 'Relative to Mesh "
        "Size' is on, else absolute meters). 0 = auto (edge size / 5).");
    processing::ShowEdgeSizeControl(_props.edgeSize, _props.relativeToMeshSize, tooltip);
    processing::ShowFeatureControls(_props.detectFeatures, _props.sharpFeatureAngle, tooltip);
  }
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    namespace mm = mochi::mesh;
    mochi::MeshData const filled = FillHolesFirst(input, error);
    if (!error.IsOK()) {
      return {};
    }
    mm::SurfaceRemeshingParams p;
    p.method = mm::RemeshMethod::SurfaceDelaunay;
    p.facetAngleBound = _props.facetAngleBound;
    p.facetDistanceBound = _props.facetDistanceBound;
    p.edgeSize = _props.edgeSize;
    p.relativeToMeshSize = _props.relativeToMeshSize;
    p.detectFeatures = _props.detectFeatures;
    p.sharpFeatureAngle = _props.sharpFeatureAngle;
    p.smoothingIterations = 0;
    p.tangentialRelaxationIterations = 0;
    p.angleSmoothingIterations = 0;
    p.repairMesh = true;
    return mm::RemeshSurface(filled, p, error);
  }
};

// ---- Incremental Isotropic Remesh ---------------------------------------------------------------
class IsotropicRemeshMethod : public ReflectedMethod<RemeshIsotropicProps> {
 public:
  char const* Name() const override {
    return "Incremental Isotropic Remesh";
  }
  char const* Description() const override {
    return "Iteratively split, collapse, flip, and relax edges toward a uniform target edge "
           "length.";
  }
  void ShowParams(ModifierGuiContext const& gui) override {
    ModifierTooltip const& tooltip = gui.tooltip;
    processing::ShowEdgeSizeControl(_props.edgeSize, _props.relativeToMeshSize, tooltip);
    processing::ShowFeatureControls(_props.detectFeatures, _props.sharpFeatureAngle, tooltip);
    ImGui::InputInt("Smoothing Iterations", &_props.smoothingIterations, 0, 0);
    tooltip(
        "Isotropic remeshing iterations: each splits long edges, collapses short edges, flips "
        "toward valence 6, and relaxes + reprojects vertices toward uniform edge length.");
    ImGui::InputInt("Relaxation Steps/Iter", &_props.relaxationStepsPerIteration, 0, 0);
    tooltip("Tangential relaxation sub-steps per isotropic iteration (better vertex placement).");
    ImGui::Checkbox("Protect Constraints", &_props.protectConstraints);
    tooltip("Hard-protect detected feature edges (no split/collapse across them).");
    ImGui::Checkbox("Relax Constraints", &_props.relaxConstraints);
    tooltip("Allow feature-edge vertices to slide along their feature polylines.");
    ImGui::Checkbox("Use Adaptive Sizing", &_props.useAdaptiveSizing);
    tooltip(
        "Curvature-based sizing: smaller triangles in high-curvature regions (requires Eigen).");
    if (_props.useAdaptiveSizing) {
      ImGui::InputDouble("Tolerance", &_props.adaptiveSizingTolerance, 0.0, 0.0, "%.6f");
      tooltip("Error tolerance for the adaptive sizing field (fraction of bbox, or absolute).");
      ImGui::InputDouble("Min Edge Factor", &_props.minEdgeSizeFactor, 0.0, 0.0, "%.4f");
      tooltip("Minimum edge length as a fraction of edge size.");
      ImGui::InputDouble("Max Edge Factor", &_props.maxEdgeSizeFactor, 0.0, 0.0, "%.4f");
      tooltip("Maximum edge length as a fraction of edge size.");
    }
  }
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    namespace mm = mochi::mesh;
    mm::SurfaceRemeshingParams p;
    p.method = mm::RemeshMethod::None; // isotropic-only post-processing
    p.edgeSize = _props.edgeSize;
    p.relativeToMeshSize = _props.relativeToMeshSize;
    p.detectFeatures = _props.detectFeatures;
    p.sharpFeatureAngle = _props.sharpFeatureAngle;
    p.smoothingIterations = _props.smoothingIterations;
    p.relaxationStepsPerIteration = _props.relaxationStepsPerIteration;
    p.protectConstraints = _props.protectConstraints;
    p.relaxConstraints = _props.relaxConstraints;
    p.useAdaptiveSizing = _props.useAdaptiveSizing;
    p.adaptiveSizingTolerance = _props.adaptiveSizingTolerance;
    p.minEdgeSizeFactor = _props.minEdgeSizeFactor;
    p.maxEdgeSizeFactor = _props.maxEdgeSizeFactor;
    p.tangentialRelaxationIterations = 0;
    p.angleSmoothingIterations = 0;
    p.repairMesh = true;
    return mm::RemeshSurface(input, p, error);
  }
};

} // namespace

ModifierRegistryEntry MakeRemeshEntry() {
  return ModifierRegistryEntry{"Remesh", ModifierKind::Transform, []() {
                                 std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
                                 methods.push_back(std::make_unique<AcvdMethod>());
                                 methods.push_back(std::make_unique<SurfaceDelaunayMethod>());
                                 methods.push_back(std::make_unique<IsotropicRemeshMethod>());
                                 return methods;
                               }};
}

} // namespace superdex::studio
