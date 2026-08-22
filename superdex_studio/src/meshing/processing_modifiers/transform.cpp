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

#include "meshing/processing_modifiers/transform.h"

#include "meshing/processing_modifiers/processing_mesh_utils.h" // ApplyTransform
#include "ui/imgui_widgets.h" // ImGui::DragRealXYZ / DragTransformRT

#include <imguios/imguios.h>

#include <memory>
#include <vector>

namespace superdex::studio {

namespace {

// Reflected props. Quaternion reflects as an array of 4 reals, so the serialized form is a plain
// xyzw array.
struct TransformSrtProps {
  mochi::Real3 scale = {1.0f, 1.0f, 1.0f};
  mochi::Quaternion rotation = {};
  mochi::Real3 translation = {0.0f, 0.0f, 0.0f};

  MOCHI_STRUCT_BEGIN(superdex::studio::TransformSrtProps)
  MOCHI_FIELD(scale)
  MOCHI_FIELD(rotation)
  MOCHI_FIELD(translation)
  MOCHI_STRUCT_END()
};

class SrtTransformMethod : public ReflectedMethod<TransformSrtProps> {
 public:
  char const* Name() const override {
    return "SRT Transform";
  }
  char const* Description() const override {
    return "Apply a scale, then rotation, then translation to the mesh.";
  }
  void ShowParams(ModifierGuiContext const& gui) override {
    ModifierTooltip const& tooltip = gui.tooltip;
    ImGui::DragRealXYZ("Scale", _props.scale, 0.01f, 0, 0, "%.4f", 0, 1.0f);
    tooltip("Per-axis scale applied to the input mesh first.");
    mochi::Quaternion rotation = _props.rotation;
    ImGui::DragTransformRT("Transform", rotation, _props.translation);
    _props.rotation = rotation;
    tooltip("Rotation then translation applied after scaling.");
  }
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& /*error*/) const override {
    mochi::MeshData out(input);
    processing::ApplyTransform(out, _props.scale, _props.rotation, _props.translation);
    return out;
  }
};

} // namespace

ModifierRegistryEntry MakeTransformEntry() {
  return ModifierRegistryEntry{"Apply SRT Transformation", ModifierKind::Transform, []() {
                                 std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
                                 methods.push_back(std::make_unique<SrtTransformMethod>());
                                 return methods;
                               }};
}

} // namespace superdex::studio
