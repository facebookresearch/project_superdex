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

#include "rendering/stats.h"
#include "assets/asset_manager.h"
#include "assets/mochi_model_asset.h"
#include "assets/prefab_constraints.h"
#include "assets/render_model_asset.h"
#include "rendering/scene_stage.h"

#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_mesh/mesh_statistics.h>
#include <mochi_renderer/scene_object.h>
#include <mochi_renderer/type_conversions.h>
#include <simple_reflection/simple_reflection.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <initializer_list>
#include <utility>

namespace superdex::studio {

namespace {

bool IsClosed(mochi::MeshDataView const& surface) {
  mochi::ErrorLog error;
  return surface.GetNumElements() > 0 &&
      mochi::mesh::ComputeMeshStatistics(surface, nullptr, error).isClosed && error.IsOK();
}

// Free joints count none: like bot poses, these leave out the floating base.
int JointDofs(mochi::ArticulatedJointType type) {
  using mochi::ArticulatedJointType;
  return type == ArticulatedJointType::Spherical                                          ? 3
      : type == ArticulatedJointType::Revolute || type == ArticulatedJointType::Prismatic ? 1
                                                                                          : 0;
}

std::string FormatCount(int64_t count) {
  std::string digits = std::to_string(count);
  for (int i = mochi::isize(digits) - 3; i > (count < 0 ? 1 : 0); i -= 3) {
    digits.insert(static_cast<size_t>(i), ",");
  }
  return digits;
}

struct PrefabCounts {
  int rigid = 0;
  int staticRigid = 0;
  int articulated = 0;
  int links = 0;
  int jointDofs = 0;
  int soft = 0;
  int nestedPrefabs = 0;
  int constraints = 0;
  int controllers = 0;
};

void CountPrefab(mochi::prefab::ScenePrefab const& prefab, PrefabCounts& counts) {
  auto const& actors = prefab.actors;
  counts.rigid += mochi::isize(actors.rigid);
  counts.staticRigid += static_cast<int>(std::ranges::count_if(
      actors.rigid, [](mochi::prefab::RigidActorPrefab const& rigid) { return rigid.isStatic; }));
  counts.articulated += mochi::isize(actors.articulated);
  for (auto const& articulated : actors.articulated) {
    counts.links += mochi::isize(articulated.links);
    for (auto const& joint : articulated.joints) {
      counts.jointDofs += JointDofs(joint.type);
    }
  }
  counts.soft += mochi::isize(actors.soft) + mochi::isize(actors.softSkinned);
  ForEachConstraintArray(prefab.constraints, [&](mochi::ConstraintType, auto const& array) {
    counts.constraints += mochi::isize(array);
  });
  counts.controllers += mochi::isize(prefab.controllers);
  counts.nestedPrefabs += mochi::isize(prefab.prefabs);
  for (auto const& nested : prefab.prefabs) {
    if (nested.prefab) {
      CountPrefab(*nested.prefab, counts);
    }
  }
}

} // namespace

std::string FormatSize(std::optional<mochi::Aabb> const& bounds) {
  if (!bounds) {
    return "None";
  }
  mochi::Real3 const size = bounds->GetSize();
  char buffer[96];
  std::snprintf(
      buffer,
      sizeof(buffer),
      "%.3f x %.3f x %.3f m",
      double(size[0]),
      double(size[1]),
      double(size[2]));
  return buffer;
}

std::string FormatBreakdown(
    int64_t total,
    std::initializer_list<std::pair<int64_t, std::string>> parts) {
  std::string list;
  for (auto const& [n, label] : parts) {
    if (n != 0) {
      list += (list.empty() ? "" : ", ") + FormatCount(n) + " " + label;
    }
  }
  return list.empty() ? FormatCount(total) : FormatCount(total) + " (" + list + ")";
}

std::string FormatBotDofs(superdex::robotics::BotPrefab const& bot) {
  int64_t dofs = 0;
  int64_t actuated = 0;
  for (auto const& joint : bot.joints) {
    dofs += JointDofs(joint.type);
    actuated += joint.effortLimit != 0 ? JointDofs(joint.type) : 0;
  }
  return FormatBreakdown(dofs, {{actuated, "Actuated"}});
}

ModelStats ComputeCollisionModelStats(mochi::ModelData const& model) {
  ModelStats stats{.hasSdf = model.sdf.has_value()};
  if (!model.mesh) {
    return stats;
  }
  mochi::MeshData const& mesh = *model.mesh;
  if (mesh.nodesPerElement == 3) {
    stats.triangles = mesh.GetNumElements();
    stats.watertight = IsClosed(mesh);
  } else if (mesh.nodesPerElement == 4) {
    stats.tetrahedra = mesh.GetNumElements();
    mochi::TetrahedralMesh const tetMesh(
        mochi::Unflatten<mochi::Real3 const>(mochi::MakeConstSpan(mesh.coordinates)),
        mochi::Unflatten<mochi::Int4 const>(mochi::MakeConstSpan(mesh.connectivity)));
    mochi::TriangularMesh const& boundary = *tetMesh.GetBoundaryMesh();
    mochi::MeshDataView surface;
    surface.nodesPerElement = 3;
    surface.coordinates = mochi::Flatten(boundary.GetNodeCoordinates());
    surface.connectivity = mochi::Flatten(boundary.GetElementConnectivity());
    stats.triangles = surface.GetNumElements();
    stats.watertight = IsClosed(surface);
  }
  return stats;
}

void GeometryStats::Add(ModelStats const& model, std::optional<mochi::Aabb> const& modelBounds) {
  ++models;
  triangles += model.triangles;
  tetrahedra += model.tetrahedra;
  watertightModels += model.watertight ? 1 : 0;
  modelsWithSdf += model.hasSdf ? 1 : 0;
  if (modelBounds) {
    bounds = !bounds ? *modelBounds
                     : mochi::Aabb(
                           mochi::Min(bounds->VGetMin(), modelBounds->VGetMin()),
                           mochi::Max(bounds->VGetMax(), modelBounds->VGetMax()));
  }
}

StatsText& StatsText::Add(std::string_view label, std::string value, bool error) {
  _rows.push_back({.label = std::string(label), .value = std::move(value), .error = error});
  return *this;
}

StatsText& StatsText::Add(std::string_view label, int64_t count) {
  return Add(label, FormatCount(count));
}

StatsText& StatsText::AddPrefabs(std::span<mochi::prefab::ScenePrefab const* const> prefabs) {
  PrefabCounts c;
  for (auto const* prefab : prefabs) {
    CountPrefab(*prefab, c);
  }
  Add("Actors",
      FormatBreakdown(
          c.rigid + c.articulated + c.soft,
          {{c.rigid - c.staticRigid, "Dynamic"},
           {c.staticRigid, "Static"},
           {c.articulated, "Articulated"},
           {c.soft, "Soft"}}));
  Add("Links", c.links);
  Add("DOFs", c.jointDofs);
  Add("Constraints", c.constraints);
  Add("Nested Prefabs", c.nestedPrefabs);
  return Add("Controllers", c.controllers);
}

StatsText& StatsText::AddPrefab(mochi::prefab::ScenePrefab const& prefab) {
  std::array<mochi::prefab::ScenePrefab const*, 1> const prefabs{&prefab};
  return AddPrefabs(prefabs);
}

StatsText& StatsText::AddBot(superdex::robotics::BotPrefab const& bot) {
  using mochi::ArticulatedJointType;
  auto const joints = [&](ArticulatedJointType type) {
    return std::pair<int64_t, std::string>{
        std::ranges::count(bot.joints, type, &superdex::robotics::BotJointPrefab::type),
        SReflect::EnumToString(type)};
  };
  int64_t sensors = 0;
  for (auto const& link : bot.links) {
    sensors += mochi::isize(link.sensors);
  }
  Add("Links", mochi::isize(bot.links));
  Add("Joints",
      FormatBreakdown(
          mochi::isize(bot.joints),
          {joints(ArticulatedJointType::Free),
           joints(ArticulatedJointType::Prismatic),
           joints(ArticulatedJointType::Revolute),
           joints(ArticulatedJointType::Spherical),
           joints(ArticulatedJointType::Hard),
           joints(ArticulatedJointType::Cycle)}));
  Add("DOFs", FormatBotDofs(bot));
  Add("Cycles", mochi::isize(bot.cycles));
  Add("Linear Transmissions", mochi::isize(bot.linearTransmissions));
  Add("Spatial Tendons", mochi::isize(bot.spatialTendons));
  Add("Contact Overrides", mochi::isize(bot.contactOverrides));
  Add("Sensors", sensors);
  return Add("Skin", bot.skin ? "Yes" : "No");
}

StatsText& StatsText::AddRender(GeometryStats const& render) {
  Add("Render Tris", render.triangles);
  return Add("Render AABB", FormatSize(render.bounds));
}

StatsText& StatsText::AddCollision(GeometryStats const& collision) {
  Add("Collision Tris", collision.triangles);
  Add("Collision Tets", collision.tetrahedra);
  Add("Collision AABB", FormatSize(collision.bounds));
  if (collision.models == 0) {
    Add("Watertight", "N/A");
    return Add("SDF", "N/A");
  }
  std::string const total = "/" + FormatCount(collision.models);
  int const open = collision.models - collision.watertightModels;
  Add("Watertight",
      open == 0 ? "Yes" : "No (" + FormatCount(open) + total + " Open)",
      /*error=*/open > 0);
  return Add("SDF", FormatCount(collision.modelsWithSdf) + total);
}

std::vector<StatsText::Line> StatsText::Lines() const {
  size_t width = 0;
  for (Row const& row : _rows) {
    width = std::max(width, row.label.size());
  }
  std::vector<Line> lines;
  for (Row const& row : _rows) {
    lines.push_back(
        {.text = row.label + std::string(width - row.label.size() + 2, ' ') + row.value,
         .error = row.error});
  }
  return lines;
}

StageStats ComputeStageStats(
    SceneStage const& stage,
    AssetManager const& assets,
    mochi::CoordinateSpaceConverter const& rendererToEditor) {
  StageStats stats;
  for (StagedActor const& actor : stage.GetActors()) {
    if (auto const* asset = assets.FindAssetByPath<RenderModelAsset>(actor.render.modelFile)) {
      stats.render.Add(
          asset->GetStats(), EditorSpaceBounds(actor.render.sceneObject, rendererToEditor));
    }
    if (auto const* asset = assets.FindAssetByPath<MochiModelAsset>(actor.shape.modelFile)) {
      stats.collision.Add(
          asset->GetStats(), EditorSpaceBounds(actor.shape.sceneObject, rendererToEditor));
    }
  }
  return stats;
}

std::optional<mochi::Aabb> EditorSpaceBounds(
    mochi_renderer::SceneObject const* object,
    mochi::CoordinateSpaceConverter const& rendererToEditor) {
  if (object == nullptr) {
    return std::nullopt;
  }
  filament::Box const box = object->GetAABB();
  if (box.isEmpty()) {
    return std::nullopt;
  }
  // The spaces differ by an axis permutation and sign flips, so the converted corners still bound
  // the box once each axis is re-sorted.
  auto const corner = [&](filament::math::float3 const& p) {
    return mochi::ToSimd(rendererToEditor.TranslationToOutput(
        mochi::StaticCast<mochi::Real3>(mochi_renderer::ToMochi(p))));
  };
  mochi::Vec4r const a = corner(box.getMin());
  mochi::Vec4r const b = corner(box.getMax());
  return mochi::Aabb(mochi::Min(a, b), mochi::Max(a, b));
}

} // namespace superdex::studio
