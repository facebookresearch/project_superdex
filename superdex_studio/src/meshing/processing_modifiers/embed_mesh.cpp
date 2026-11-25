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

#include "meshing/processing_modifiers/embed_mesh.h"

#include "meshing/processing_modifiers/remesh_params_ui.h" // ShowReferenceCombo

#include <imgui.h>

#include <mochi_core/geometry/mesh_data_utils.h> // IsPolylineClosedLoop
#include <mochi_core/geometry/model_utils.h> // ComputePolylineSkinning, GenerateTubularSkinMesh, MaxAuxiliaryMeshExtrapolation, ValidateSkinning
#include <mochi_core/utils/array_utils.h>
#include <mochi_physics/utils/mochi_model_utils.h> // EmbedInTetrahedralMesh

#include <algorithm>
#include <climits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace superdex::studio {

namespace {

// Base of the Embed modifiers' methods: passes the input on with one more mesh embedded in it, in
// the role its modifier embeds.
template <typename Props>
class EmbedMethod : public ReflectedMethod<Props> {
 public:
  explicit EmbedMethod(EmbeddedMeshRole role) : _role(role) {}

  [[nodiscard]] bool AcceptsTetrahedralInput() const override {
    return true;
  }
  [[nodiscard]] bool AcceptsPolylineInput() const override {
    return true;
  }
  [[nodiscard]] bool AcceptsEmbeddedMeshes() const override {
    return true;
  }
  [[nodiscard]] std::optional<EmbeddedMeshRole> EmbeddedRole() const override {
    return _role;
  }

  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& ctx,
      mochi::Error& error) const override {
    return RunStage({.mesh = input}, ctx, error).mesh;
  }
  StageMesh RunStage(StageMesh const& input, ModifierRunContext const& ctx, mochi::Error& error)
      const override {
    MOCHI_ERROR_IF(
        input.embedded.Mesh(_role).has_value(),
        error,
        "Embed: an Embed modifier above has already embedded a mesh in this role. Remove one of "
        "the two.");
    MOCHI_ERROR_RETURN(error, {});
    mochi::MeshData embedded = Embed(input, ctx, error);
    MOCHI_ERROR_RETURN(error, {});
    // Checked here so an invalid embedding fails the stage that made it, not the export.
    if (!embedded.skinning.has_value()) {
      MOCHI_ERROR_SET(error, "Embed: the embedded mesh has no skinning.");
      return {};
    }
    // A polyline's embeddings skin to its elements, any other mesh's to its nodes.
    int const numSkinningSources =
        input.mesh.nodesPerElement == 2 ? input.mesh.GetNumElements() : input.mesh.GetNumNodes();
    mochi::model::ValidateSkinning(
        *embedded.skinning, embedded.GetNumNodes(), numSkinningSources, error);
    MOCHI_ERROR_RETURN(error, {});
    StageMesh output = input;
    output.embedded.Mesh(_role) = std::move(embedded);
    return output;
  }

 protected:
  // The mesh to embed in @p input, with the skinning that embeds it there.
  virtual mochi::MeshData
  Embed(StageMesh const& input, ModifierRunContext const& ctx, mochi::Error& error) const = 0;

  // What the embedded mesh is for, shown above each method's parameters.
  [[nodiscard]] char const* RoleSummary() const {
    switch (_role) {
      case EmbeddedMeshRole::ContactSkin:
        return "A soft or rod actor with Use Contact Skin on takes its contact samples from its "
               "contact skin instead of its tetrahedral mesh's boundary or its centerline.";
      case EmbeddedMeshRole::VisualMesh:
        return "Renderers draw a soft or rod actor by its visual mesh, which deforms with the "
               "simulation.";
    }
    return "";
  }

 private:
  EmbeddedMeshRole _role;
};

// The node positions of the polyline @p mesh.
mochi::Span<mochi::Real3 const> PolylineNodes(mochi::MeshData const& mesh, mochi::Error& error) {
  MOCHI_ERROR_IF(
      mesh.coordinates.size() % 3 != 0,
      error,
      "Coordinates array size must be a multiple of three.");
  MOCHI_ERROR_RETURN(error, {});
  return mochi::Unflatten<mochi::Real3 const>(mochi::MakeConstSpan(mesh.coordinates));
}

struct UpstreamSurfaceProps {
  // Upstream stage whose surface is embedded (see MeshProcessingMethod::ReferenceIndex).
  int referenceIndex = MeshProcessingMethod::kReferencePrecedingSource;
  // Blend widths when embedding in a polyline (see mochi::model::ComputePolylineSkinning).
  int nominalWeightsPerNode = 2;
  int maxWeightsPerNode = 4;

  MOCHI_STRUCT_BEGIN(superdex::studio::UpstreamSurfaceProps)
  MOCHI_FIELD(referenceIndex)
  MOCHI_FIELD(nominalWeightsPerNode)
  MOCHI_FIELD(maxWeightsPerNode)
  MOCHI_STRUCT_END()
};

class UpstreamSurfaceMethod : public EmbedMethod<UpstreamSurfaceProps> {
 public:
  using EmbedMethod::EmbedMethod;

  [[nodiscard]] char const* Name() const override {
    return "Upstream Surface";
  }
  [[nodiscard]] char const* Description() const override {
    return "Embed the triangle surface of a stage above in the tetrahedral mesh or polyline, e.g. "
           "the Wrap Mesh output that Tetrahedralize filled, or a render model.";
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
    ImGui::TextWrapped("%s", RoleSummary());
    processing::ShowReferenceCombo("Surface", _props.referenceIndex, gui);
    gui.tooltip(
        "The upstream surface to embed, e.g. the Wrap Mesh output that Tetrahedralize filled, or a "
        "Source from Render Model stage. '(invalid source)' means this modifier was moved above it "
        "-- pick a valid one.");

    ImGui::SeparatorText("Polyline Blend");
    bool blendEdited = ImGui::InputInt("Nominal Elements", &_props.nominalWeightsPerNode);
    gui.tooltip(
        "Embedding in a polyline only: the most elements a node binds to where the polyline is "
        "straight or gently curved. With 1, two elements still blend around a joint. 2 blends "
        "neighboring elements; more blends more smoothly, with more volume loss at bends.");
    blendEdited |= ImGui::InputInt("Max Elements", &_props.maxWeightsPerNode);
    gui.tooltip(
        "Embedding in a polyline only: the most elements any node binds to, which bounds the cost "
        "of deforming the mesh. Nodes near a bend's center of curvature blend more widely than "
        "nominal, up to this many.");
    if (blendEdited) {
      _props.nominalWeightsPerNode = std::clamp(_props.nominalWeightsPerNode, 1, INT_MAX - 1);
      _props.maxWeightsPerNode =
          std::max(_props.maxWeightsPerNode, _props.nominalWeightsPerNode + 1);
    }
  }

 protected:
  mochi::MeshData Embed(StageMesh const& input, ModifierRunContext const& ctx, mochi::Error& error)
      const override {
    mochi::MeshData const& mesh = input.mesh;
    MOCHI_ERROR_IF(
        mesh.nodesPerElement != 4 && mesh.nodesPerElement != 2,
        error,
        "Embed: Upstream Surface embeds in a tetrahedral mesh or a polyline. Place this modifier "
        "after Tetrahedralize or below a polyline source.");
    MOCHI_ERROR_IF(
        ctx.referenceMesh.nodesPerElement != 3,
        error,
        "Embed: the selected surface is invalid, missing, or not a triangle surface.");
    MOCHI_ERROR_RETURN(error, {});
    mochi::MeshData surface = ctx.referenceMesh;
    if (mesh.nodesPerElement == 4) {
      mochi::model_utils::EmbedInTetrahedralMesh(mesh, surface, error);
      MOCHI_ERROR_RETURN(error, {});
      return surface;
    }
    mochi::Span<mochi::Real3 const> const nodes = PolylineNodes(mesh, error);
    MOCHI_ERROR_RETURN(error, {});
    mochi::model::ComputePolylineSkinning(
        nodes,
        mochi::IsPolylineClosedLoop(mesh),
        _props.nominalWeightsPerNode,
        _props.maxWeightsPerNode,
        surface,
        error);
    MOCHI_ERROR_RETURN(error, {});
    return surface;
  }
};

struct TubeProps {
  double radius = 0.005; // [m]
  int numCrossSectionSegments = 8;

  MOCHI_STRUCT_BEGIN(superdex::studio::TubeProps)
  MOCHI_FIELD(radius)
  MOCHI_FIELD(numCrossSectionSegments)
  MOCHI_STRUCT_END()
};

class TubeMethod : public EmbedMethod<TubeProps> {
 public:
  using EmbedMethod::EmbedMethod;

  [[nodiscard]] char const* Name() const override {
    return "Tube";
  }
  [[nodiscard]] char const* Description() const override {
    return "Embed a tube of circular cross-section around a polyline, skinned to its elements.";
  }

  void ShowParams(ModifierGuiContext const& gui) override {
    ImGui::TextWrapped("%s", RoleSummary());
    if (ImGui::InputDouble("Radius (m)", &_props.radius, 0.0, 0.0, "%.4f")) {
      _props.radius = std::max(_props.radius, 0.0);
    }
    gui.tooltip("The tube's radius, in meters.");
    if (ImGui::InputInt("Segments", &_props.numCrossSectionSegments)) {
      _props.numCrossSectionSegments = std::max(_props.numCrossSectionSegments, 3);
    }
    gui.tooltip(
        "The number of nodes around each of the tube's rings. A ring sits around each element's "
        "midpoint (and each end of an open polyline, which is capped), skinned to that element.");
  }

 protected:
  mochi::MeshData Embed(
      StageMesh const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    MOCHI_ERROR_IF(
        input.mesh.nodesPerElement != 2,
        error,
        "Embed: Tube embeds in a polyline. Place this modifier below a polyline source.");
    MOCHI_ERROR_RETURN(error, {});
    mochi::Span<mochi::Real3 const> const nodes = PolylineNodes(input.mesh, error);
    MOCHI_ERROR_IF(
        !input.elementFrameAxes.has_value(), error, "Embed: Tube requires polyline frame axes.");
    MOCHI_ERROR_RETURN(error, {});
    MOCHI_ERROR_IF(
        input.elementFrameAxes->size() % 3 != 0,
        error,
        "Element frame axes array size must be a multiple of three.");
    MOCHI_ERROR_RETURN(error, {});
    return mochi::model::GenerateTubularSkinMesh(
        nodes,
        mochi::Unflatten<mochi::Real3 const>(mochi::MakeConstSpan(*input.elementFrameAxes)),
        mochi::IsPolylineClosedLoop(input.mesh),
        static_cast<mochi::real>(_props.radius),
        _props.numCrossSectionSegments,
        error);
  }
};

ModifierRegistryEntry MakeEmbedEntry(char const* name, EmbeddedMeshRole role) {
  return ModifierRegistryEntry{name, ModifierKind::Transform, [role]() {
                                 std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
                                 methods.push_back(std::make_unique<UpstreamSurfaceMethod>(role));
                                 methods.push_back(std::make_unique<TubeMethod>(role));
                                 return methods;
                               }};
}

} // namespace

ModifierRegistryEntry MakeEmbedContactSkinEntry() {
  return MakeEmbedEntry("Embed Contact Skin", EmbeddedMeshRole::ContactSkin);
}

ModifierRegistryEntry MakeEmbedVisualMeshEntry() {
  return MakeEmbedEntry("Embed Visual Mesh", EmbeddedMeshRole::VisualMesh);
}

EmbeddedMeshStats MeasureEmbeddedMesh(
    EmbeddedMeshRole role,
    mochi::MeshData const& mesh,
    mochi::MeshData const& host) {
  EmbeddedMeshStats stats{
      .role = role, .nodes = mesh.GetNumNodes(), .triangles = mesh.GetNumElements()};
  if (host.nodesPerElement == 4 && mesh.skinning.has_value()) {
    stats.maxExtrapolation = mochi::model::MaxAuxiliaryMeshExtrapolation(*mesh.skinning);
  }
  return stats;
}

} // namespace superdex::studio
