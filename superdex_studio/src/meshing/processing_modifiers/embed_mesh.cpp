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

#include <mochi_core/geometry/model_utils.h> // MaxAuxiliaryMeshExtrapolation, ValidateSkinning
#include <mochi_physics/utils/mochi_model_utils.h> // EmbedInTetrahedralMesh

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace superdex::studio {

namespace {

// Base of the Embed modifiers' methods: passes the input on with one more mesh embedded in it, in
// the role its modifier embeds. Embedding is invariant under the rotation from the stack's renderer
// space to Mochi space, so it is done in the stack's space and Export Mochi Model converts both.
template <typename Props>
class EmbedMethod : public ReflectedMethod<Props> {
 public:
  explicit EmbedMethod(EmbeddedMeshRole role) : _role(role) {}

  [[nodiscard]] bool AcceptsTetrahedralInput() const override {
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
    mochi::MeshData embedded = Embed(input.mesh, ctx, error);
    MOCHI_ERROR_RETURN(error, {});
    // Checked here so an invalid embedding fails the stage that made it, not the export.
    if (!embedded.skinning.has_value()) {
      MOCHI_ERROR_SET(error, "Embed: the embedded mesh has no skinning.");
      return {};
    }
    mochi::model::ValidateSkinning(
        *embedded.skinning, embedded.GetNumNodes(), input.mesh.GetNumNodes(), error);
    MOCHI_ERROR_RETURN(error, {});
    StageMesh output = input;
    output.embedded.Mesh(_role) = std::move(embedded);
    return output;
  }

 protected:
  // The mesh to embed in @p mesh, with the skinning that embeds it there.
  virtual mochi::MeshData
  Embed(mochi::MeshData const& mesh, ModifierRunContext const& ctx, mochi::Error& error) const = 0;

  // What the embedded mesh is for, shown above each method's parameters.
  [[nodiscard]] char const* RoleSummary() const {
    return "A soft actor with Use Contact Skin on takes its contact samples from its contact skin "
           "instead of its tetrahedral mesh's boundary.";
  }

 private:
  EmbeddedMeshRole _role;
};

struct UpstreamSurfaceProps {
  // Upstream stage whose surface is embedded (see MeshProcessingMethod::ReferenceIndex).
  int referenceIndex = MeshProcessingMethod::kReferencePrecedingSource;

  MOCHI_STRUCT_BEGIN(superdex::studio::UpstreamSurfaceProps)
  MOCHI_FIELD(referenceIndex)
  MOCHI_STRUCT_END()
};

class UpstreamSurfaceMethod : public EmbedMethod<UpstreamSurfaceProps> {
 public:
  using EmbedMethod::EmbedMethod;

  [[nodiscard]] char const* Name() const override {
    return "Upstream Surface";
  }
  [[nodiscard]] char const* Description() const override {
    return "Embed the triangle surface of a stage above in the tetrahedral mesh, e.g. the Wrap Mesh "
           "output that Tetrahedralize filled.";
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
        "The upstream surface to embed in the tetrahedral mesh, e.g. the Wrap Mesh output that "
        "Tetrahedralize filled. '(invalid source)' means this modifier was moved above it -- pick "
        "a valid one.");
  }

 protected:
  mochi::MeshData Embed(
      mochi::MeshData const& mesh,
      ModifierRunContext const& ctx,
      mochi::Error& error) const override {
    MOCHI_ERROR_IF(
        mesh.nodesPerElement != 4,
        error,
        "Embed: Upstream Surface embeds in a tetrahedral mesh. Place this modifier after "
        "Tetrahedralize.");
    MOCHI_ERROR_IF(
        ctx.referenceMesh.nodesPerElement != 3,
        error,
        "Embed: the selected surface is invalid, missing, or not a triangle surface.");
    MOCHI_ERROR_RETURN(error, {});
    mochi::MeshData surface = ctx.referenceMesh;
    mochi::model_utils::EmbedInTetrahedralMesh(mesh, surface, error);
    MOCHI_ERROR_RETURN(error, {});
    return surface;
  }
};

} // namespace

ModifierRegistryEntry MakeEmbedContactSkinEntry() {
  return ModifierRegistryEntry{
      "Embed Contact Skin", ModifierKind::Transform, []() {
        std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
        methods.push_back(std::make_unique<UpstreamSurfaceMethod>(EmbeddedMeshRole::ContactSkin));
        return methods;
      }};
}

EmbeddedMeshStats MeasureEmbeddedMesh(EmbeddedMeshRole role, mochi::MeshData const& mesh) {
  return {
      .role = role,
      .nodes = mesh.GetNumNodes(),
      .triangles = mesh.GetNumElements(),
      .maxExtrapolation = mesh.skinning.has_value()
          ? mochi::model::MaxAuxiliaryMeshExtrapolation(*mesh.skinning)
          : 0.0};
}

} // namespace superdex::studio
