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

#include "meshing/processing_modifiers/source_render_model.h"

#include "app/app.h" // AssetType, AssetManager, SuperDexStudio
#include "meshing/processing_modifiers/processing_mesh_utils.h" // LoadRenderMesh
#include "ui/imgui_widgets.h" // ImGui::AssetSlot

#include <imguios/imguios.h>

#include <mochi_core/utils/dynamic_string.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace superdex::studio {

namespace {

// "From File" parameters: the modifier's own render-model slot (independent of the editor's slots).
struct RenderFileSourceProps {
  mochi::DynamicString path;

  MOCHI_STRUCT_BEGIN(superdex::studio::RenderFileSourceProps)
  MOCHI_FIELD(path)
  MOCHI_STRUCT_END()
};

// ---- From Model Viewer --------------------------------------------------------------------------
class RenderFromViewerMethod : public ReflectedMethod<EmptyMethodProps> {
 public:
  char const* Name() const override {
    return "From Model Viewer";
  }
  char const* Description() const override {
    return "Uses the editor's current Render Model surface (its Model Viewer slot) as the source.";
  }
  bool CanGenerate(ModifierRunContext const& ctx) const override {
    return !ctx.renderModelPath.empty();
  }
  std::string SourceFilePath(ModifierRunContext const& ctx) const override {
    return ctx.renderModelPath;
  }
  std::string PropsSignature(ModifierRunContext const& ctx) const override {
    return "|" + ctx.renderModelPath; // input is the editor's render slot (external to props)
  }
  void ShowParams(ModifierGuiContext const& /*gui*/) override {
    ImGui::TextUnformatted("Uses the editor's current Render Model surface as the source.");
  }
  mochi::MeshData Run(
      mochi::MeshData const& /*input*/,
      ModifierRunContext const& ctx,
      mochi::Error& error) const override {
    MOCHI_ERROR_IF(
        ctx.renderModelPath.empty(), error, "Source from Render Model: no Render Model is set.");
    MOCHI_ERROR_RETURN(error, {});
    return processing::LoadRenderMesh(ctx.renderModelPath, error);
  }
};

// ---- From File ----------------------------------------------------------------------------------
class RenderFromFileMethod : public ReflectedMethod<RenderFileSourceProps> {
 public:
  char const* Name() const override {
    return "From File";
  }
  char const* Description() const override {
    return "Uses a Render Model from this modifier's own slot, independent of the editor's Model "
           "Viewer.";
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
        "##rendersocket",
        _props.path,
        *gui.assetManager,
        gui.studio,
        AssetType::RenderModel,
        /*acceptDragDropPayload=*/true);
    gui.tooltip(
        "Drop a Render Model here (e.g. an .obj/.glb in the _render folder). This slot is "
        "independent of the editor's own Model Viewer slots.");
  }
  mochi::MeshData Run(
      mochi::MeshData const& /*input*/,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    std::string const path{_props.path.data(), _props.path.size()};
    MOCHI_ERROR_IF(path.empty(), error, "Source from Render Model: no Render Model is set.");
    MOCHI_ERROR_RETURN(error, {});
    return processing::LoadRenderMesh(path, error);
  }
};

} // namespace

ModifierRegistryEntry MakeRenderModelSourceEntry() {
  return ModifierRegistryEntry{"Source from Render Model", ModifierKind::Source, []() {
                                 std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
                                 methods.push_back(std::make_unique<RenderFromViewerMethod>());
                                 methods.push_back(std::make_unique<RenderFromFileMethod>());
                                 return methods;
                               }};
}

} // namespace superdex::studio
