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

#include <mochi_core/geometry/model_data.h>
#include <mochi_physics/utils/mochi_model_utils.h> // BakeTransform

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
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
  [[nodiscard]] char const* Name() const override {
    return "SRT Transform";
  }
  [[nodiscard]] char const* Description() const override {
    return "Apply a scale, then rotation, then translation to the mesh.";
  }
  [[nodiscard]] bool AcceptsTetrahedralInput() const override {
    return true;
  }
  [[nodiscard]] bool AcceptsPolylineInput() const override {
    return true;
  }
  void ShowParams(ModifierGuiContext const& gui) override {
    ModifierTooltip const& tooltip = gui.tooltip;
    if (!gui.host) {
      return;
    }
    gui.host->DragXYZ("Scale", _props.scale, 0.01f, "%.4f");
    tooltip("Per-axis scale applied to the input mesh first.");
    mochi::Quaternion rotation = _props.rotation;
    gui.host->DragTransformRT("Transform", rotation, _props.translation);
    _props.rotation = rotation;
    tooltip("Rotation then translation applied after scaling.");
  }
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    if (input.nodesPerElement == 4) {
      auto const invalidScale = [](mochi::real value) {
        return !std::isfinite(value) || value == mochi::real{0};
      };
      MOCHI_ERROR_IF(
          invalidScale(_props.scale[0]) || invalidScale(_props.scale[1]) ||
              invalidScale(_props.scale[2]),
          error,
          "SRT Transform: tetrahedral meshes require finite, nonzero scale components.");
      MOCHI_ERROR_RETURN(error, {});
    }
    mochi::MeshData out(input);
    processing::ApplyTransform(out, _props.scale, _props.rotation, _props.translation);
    // A mirroring scale inverts every tetrahedron; swap two nodes of each to restore orientation.
    int const negativeScaleAxes = static_cast<int>(_props.scale[0] < mochi::real{0}) +
        static_cast<int>(_props.scale[1] < mochi::real{0}) +
        static_cast<int>(_props.scale[2] < mochi::real{0});
    bool const mirrored = negativeScaleAxes % 2 != 0;
    if (out.nodesPerElement == 4 && mirrored) {
      for (int tet = 0; tet < out.GetNumElements(); ++tet) {
        std::size_t const offset = 4 * static_cast<std::size_t>(tet);
        std::swap(out.connectivity[offset + 2], out.connectivity[offset + 3]);
      }
    }
    return out;
  }
  StageMesh RunStage(StageMesh const& input, ModifierRunContext const& ctx, mochi::Error& error)
      const override {
    if (input.mesh.nodesPerElement != 2) {
      return {.mesh = Run(input.mesh, ctx, error)};
    }
    // A polyline's frame axes transform with it, exactly as Mochi bakes a transform into a polyline
    // model.
    mochi::ModelData polyline;
    polyline.mesh = input.mesh;
    polyline.elementFrameAxes = input.elementFrameAxes;
    mochi::model_utils::BakeTransform(
        polyline, _props.scale, _props.rotation, _props.translation, error);
    MOCHI_ERROR_RETURN(error, {});
    return {
        .mesh = std::move(*polyline.mesh),
        .elementFrameAxes = std::move(polyline.elementFrameAxes)};
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
