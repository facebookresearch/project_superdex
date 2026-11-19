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

#include "meshing/processing_modifiers/source_mochi_model.h"

#include "meshing/processing_modifiers/processing_mesh_utils.h" // LoadMochiStage

#include <imgui.h>

#include <mochi_core/utils/dynamic_string.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace superdex::studio {

namespace {

// "From File" parameters: the modifier's own mochi-model slot (independent of the editor's slots).
struct MochiFileSourceProps {
  mochi::DynamicString path;

  MOCHI_STRUCT_BEGIN(superdex::studio::MochiFileSourceProps)
  MOCHI_FIELD(path)
  MOCHI_STRUCT_END()
};

// ---- From Model Viewer --------------------------------------------------------------------------
class MochiFromViewerMethod : public ReflectedMethod<EmptyMethodProps> {
 public:
  [[nodiscard]] char const* Name() const override {
    return "From Model Viewer";
  }
  [[nodiscard]] char const* Description() const override {
    return "Uses the editor's current Mochi Model (its Model Viewer slot) as the source: its "
           "surface, or for a polyline model its bare polyline.";
  }
  [[nodiscard]] bool CanGenerate(ModifierRunContext const& ctx) const override {
    return !ctx.mochiModelPath.empty();
  }
  [[nodiscard]] std::string SourceFilePath(ModifierRunContext const& ctx) const override {
    return ctx.mochiModelPath;
  }
  [[nodiscard]] std::string PropsSignature(ModifierRunContext const& ctx) const override {
    return "|" + ctx.mochiModelPath; // input is the editor's mochi slot (external to props)
  }
  void ShowParams(ModifierGuiContext const& /*gui*/) override {
    ImGui::TextUnformatted("Uses the editor's current Mochi Model as the source.");
  }
  mochi::MeshData Run(
      mochi::MeshData const& /*input*/,
      ModifierRunContext const& ctx,
      mochi::Error& error) const override {
    return RunStage({}, ctx, error).mesh;
  }
  StageMesh RunStage(StageMesh const& /*input*/, ModifierRunContext const& ctx, mochi::Error& error)
      const override {
    MOCHI_ERROR_IF(
        ctx.mochiModelPath.empty(), error, "Source from Mochi Model: no Mochi Model is set.");
    MOCHI_ERROR_RETURN(error, {});
    return processing::LoadMochiStage(ctx.mochiModelPath, error);
  }
};

// ---- From File ----------------------------------------------------------------------------------
class MochiFromFileMethod : public ReflectedMethod<MochiFileSourceProps> {
 public:
  [[nodiscard]] char const* Name() const override {
    return "From File";
  }
  [[nodiscard]] char const* Description() const override {
    return "Uses a Mochi Model from this modifier's own slot, independent of the editor's Model "
           "Viewer: its surface, or for a polyline model its bare polyline.";
  }
  [[nodiscard]] bool CanGenerate(ModifierRunContext const& /*ctx*/) const override {
    return !_props.path.empty();
  }
  [[nodiscard]] std::string SourceFilePath(ModifierRunContext const& /*ctx*/) const override {
    return std::string{_props.path.data(), _props.path.size()};
  }
  [[nodiscard]] std::vector<std::string_view> PathPropKeys() const override {
    return {"path"};
  }
  void ShowParams(ModifierGuiContext const& gui) override {
    if (!gui.host) {
      return;
    }
    gui.host->ModelSlot("##mochisocket", _props.path, ModelSlotType::MochiModel);
    gui.tooltip(
        "Drop a Mochi Model here (e.g. a .mochi.h5 in the _mochi folder). This slot is independent "
        "of the editor's own Model Viewer slots.");
  }
  mochi::MeshData Run(
      mochi::MeshData const& /*input*/,
      ModifierRunContext const& ctx,
      mochi::Error& error) const override {
    return RunStage({}, ctx, error).mesh;
  }
  StageMesh RunStage(
      StageMesh const& /*input*/,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    std::string const path{_props.path.data(), _props.path.size()};
    MOCHI_ERROR_IF(path.empty(), error, "Source from Mochi Model: no Mochi Model is set.");
    MOCHI_ERROR_RETURN(error, {});
    return processing::LoadMochiStage(path, error);
  }
};

} // namespace

ModifierRegistryEntry MakeMochiModelSourceEntry() {
  return ModifierRegistryEntry{"Source from Mochi Model", ModifierKind::Source, []() {
                                 std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
                                 methods.push_back(std::make_unique<MochiFromViewerMethod>());
                                 methods.push_back(std::make_unique<MochiFromFileMethod>());
                                 return methods;
                               }};
}

} // namespace superdex::studio
