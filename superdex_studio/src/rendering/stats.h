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

#pragma once

#include <superdex_robotics/superdex_robotics.h>

#include <mochi_core/geometry/aabb.h>
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Counts and text behind the viewport Stats overlay (Show > Stats). Each editor picks the rows that
// apply to it.

namespace mochi_renderer {
class SceneObject;
} // namespace mochi_renderer

namespace superdex::studio {

class AssetManager;
struct SceneStage;

// Counts of one render or collision model, cached per asset (RenderModelAsset::GetStats,
// MochiModelAsset::GetStats).
struct ModelStats {
  // A collision model's triangles are its surface (a tet mesh's boundary).
  int64_t triangles = 0;
  int64_t tetrahedra = 0;
  // mochi::mesh::MeshStatistics::isClosed of the collision surface, as the Model Editor reports
  // it. A model without a mesh (a primitive or an SDF alone) is closed.
  bool watertight = true;
  bool hasSdf = false;
};

[[nodiscard]] ModelStats ComputeCollisionModelStats(mochi::ModelData const& model);

// The render or the collision models of a set of actors, summed.
struct GeometryStats {
  int models = 0;
  int64_t triangles = 0;
  int64_t tetrahedra = 0;
  int watertightModels = 0;
  int modelsWithSdf = 0;
  std::optional<mochi::Aabb> bounds; // editor space [m]

  void Add(ModelStats const& model, std::optional<mochi::Aabb> const& modelBounds);
};

// A box's size, e.g. "0.320 x 0.583 x 0.148 m", or "None".
[[nodiscard]] std::string FormatSize(std::optional<mochi::Aabb> const& bounds);

// "total (n1 Label1, n2 Label2)" over the nonzero parts, or "total" when every part is zero.
[[nodiscard]] std::string FormatBreakdown(
    int64_t total,
    std::initializer_list<std::pair<int64_t, std::string>> parts);

// A bot's DOFs and how many are actuated (nonzero BotJointPrefab::effortLimit), e.g. "20 (16
// Actuated)".
[[nodiscard]] std::string FormatBotDofs(superdex::robotics::BotPrefab const& bot);

// "label  value" rows, labels padded so the values line up in a monospace font.
class StatsText {
 public:
  struct Line {
    std::string text;
    bool error = false; // drawn in the error color
  };

  StatsText& Add(std::string_view label, std::string value, bool error = false);
  StatsText& Add(std::string_view label, int64_t count);
  // Actors by type, articulated links and DOFs, constraints, nested prefabs and controllers of the
  // prefabs together, nested prefabs included.
  StatsText& AddPrefabs(std::span<mochi::prefab::ScenePrefab const* const> prefabs);
  StatsText& AddPrefab(mochi::prefab::ScenePrefab const& prefab);
  // Links, joints by type, DOFs, cycles, linear transmissions, spatial tendons, contact overrides,
  // sensors and skin.
  StatsText& AddBot(superdex::robotics::BotPrefab const& bot);
  StatsText& AddRender(GeometryStats const& render);
  // Watertight is an error row when any model is open.
  StatsText& AddCollision(GeometryStats const& collision);
  [[nodiscard]] std::vector<Line> Lines() const;

 private:
  struct Row {
    std::string label;
    std::string value;
    bool error = false;
  };
  std::vector<Row> _rows;
};

// The render and collision models of every staged actor, shown or hidden.
struct StageStats {
  GeometryStats render;
  GeometryStats collision;
};

// Sums the stage's models from their assets' cached stats. Bounds are the staged objects' world
// boxes as currently posed: editors take this right after staging, while every actor is at its rest
// pose, and keep it for the overlay.
[[nodiscard]] StageStats ComputeStageStats(
    SceneStage const& stage,
    AssetManager const& assets,
    mochi::CoordinateSpaceConverter const& rendererToEditor);

// @p object's world box in editor space, or nullopt when it is null or has no geometry.
[[nodiscard]] std::optional<mochi::Aabb> EditorSpaceBounds(
    mochi_renderer::SceneObject const* object,
    mochi::CoordinateSpaceConverter const& rendererToEditor);

} // namespace superdex::studio
