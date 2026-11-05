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

#include "rendering/measure_tool.h"

#include "app/app.h"
#include "meshing/processing_modifiers/processing_mesh_utils.h"
#include "rendering/debug_text.h"
#include "rendering/scene_stage.h"
#include "rendering/viewport.h"
#include "ui/imgui_widgets.h"

#include <mochi_renderer/debug.h>
#include <mochi_renderer/scene.h>
#include <mochi_renderer/scene_object.h>
#include <mochi_renderer/type_conversions.h>

#include <mochi_core/geometry/aabb.h>
#include <mochi_core/geometry/ray.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/math_utils.h>

#include <imgui.h>
#include <imguios/fonts/icons_font_awesome5.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <utility>

namespace superdex::studio {

namespace {

using mochi::real;
using mochi::Real3;

filament::math::float4 const kSelectionColor{kViewportSelectionColor, 1.0f};
filament::math::float4 const kSelectionFillColor{kViewportSelectionColor, 0.35f};
constexpr filament::math::float4 kHoverColor{0.6f, 0.8f, 1.0f, 1.0f};
constexpr filament::math::float4 kHoverFillColor{0.6f, 0.8f, 1.0f, 0.2f};
// The measurement span. Magenta: it has to read against the axis-colored legs, the orange selection
// markers and arbitrary scene geometry, and nothing else in the viewport uses it.
constexpr filament::math::float4 kSpanColor{1.0f, 0.15f, 0.9f, 1.0f};

filament::math::float4 AxisColor(int axis) {
  ImVec4 const color = ImGui::ColorConvertU32ToFloat4(kAxes[static_cast<std::size_t>(axis)].normal);
  return {color.x, color.y, color.z, 1.0f};
}

constexpr float kMarkerScreenFraction = 0.006f;
constexpr float kMarkerMinRadius = 0.0002f;
constexpr float kMarkerMaxRadius = 0.05f;
constexpr int kMarkerStacks = 8;
constexpr int kMarkerSlices = 12;

constexpr float kLineScreenFraction = 0.0015f;
constexpr float kLineMinHalfWidth = 0.00005f;
constexpr float kLineMaxHalfWidth = 0.01f;

float ScreenScaledSize(
    filament::math::float3 point,
    filament::math::float3 cameraPosition,
    float fraction,
    float minSize,
    float maxSize) {
  float const distance = length(point - cameraPosition);
  return std::clamp(distance * fraction, minSize, maxSize);
}

// Draws a segment as a camera-facing quad in the overlay (translucent) bucket.
void DrawOverlayLine(
    mochi_renderer::DebugDraw* debugDraw,
    filament::math::float3 from,
    filament::math::float3 to,
    filament::math::float3 cameraPosition,
    filament::math::float4 color) {
  filament::math::float3 const along = to - from;
  float const alongLength = length(along);
  if (alongLength <= 0.0f) {
    return;
  }
  filament::math::float3 const direction = along / alongLength;
  filament::math::float3 const midpoint = (from + to) * 0.5f;
  filament::math::float3 toCamera = cameraPosition - midpoint;
  float const toCameraLength = length(toCamera);
  if (toCameraLength <= 0.0f) {
    return;
  }
  toCamera /= toCameraLength;

  filament::math::float3 side = cross(direction, toCamera);
  float sideLength = length(side);
  if (sideLength <= 1e-6f) {
    // The segment points almost straight at the camera, so it has no screen-space width to expand
    // into. Any perpendicular will do; the quad is nearly edge-on either way.
    side = cross(direction, filament::math::float3{0.0f, 0.0f, 1.0f});
    sideLength = length(side);
    if (sideLength <= 1e-6f) {
      side = cross(direction, filament::math::float3{1.0f, 0.0f, 0.0f});
      sideLength = length(side);
      if (sideLength <= 1e-6f) {
        return;
      }
    }
  }
  float const halfWidth = ScreenScaledSize(
      midpoint, cameraPosition, kLineScreenFraction, kLineMinHalfWidth, kLineMaxHalfWidth);
  filament::math::float3 const offset = (side / sideLength) * halfWidth;

  debugDraw->DrawSolidTriangle(from - offset, from + offset, to + offset, color, /*overlay=*/true);
  debugDraw->DrawSolidTriangle(from - offset, to + offset, to - offset, color, /*overlay=*/true);
}

// Collision models are `.mochi.h5`; anything else is treated as a render mesh.
bool IsMochiModelPath(mochi::Path const& path) {
  return path.AsLowercaseString().ends_with(".mochi.h5");
}

// Change stamp for a set of in-memory sections: their element counts plus their first and last
// positions. Catches a regeneration (the counts change) and a transform edit (the positions move).
uint64_t SectionsChangeStamp(std::vector<mochi_renderer::MeshSection> const& sections) {
  uint64_t stamp = 1469598103934665603ull;
  auto mix = [&stamp](uint64_t value) { stamp = (stamp ^ value) * 1099511628211ull; };
  for (mochi_renderer::MeshSection const& section : sections) {
    mix(section.positions.size());
    mix(section.indices.size());
    if (!section.positions.empty()) {
      for (std::size_t i = 0; i < 3 && i < section.positions.size(); ++i) {
        mix(std::bit_cast<uint32_t>(section.positions[i]));
        mix(std::bit_cast<uint32_t>(section.positions[section.positions.size() - 1 - i]));
      }
    }
  }
  return stamp;
}

// True for an actor whose rendered surface deforms per-vertex while simulating: a soft actor (its
// dynamic solid/wireframe meshes) or an articulated skin (its GPU-skinned surface, plus the
// simulation-driven one that stands in for it during a run).
bool DeformsDuringSimulation(StagedActor const& actor) {
  return actor.dynamicSolidMesh != nullptr || actor.dynamicWireframeMesh != nullptr ||
      actor.shapeSkinnedSurface != nullptr || actor.shapeSimSurface != nullptr;
}

// Change stamp for the staged pose: the actor count plus every actor's world transform
uint64_t StagedPoseStamp(SceneStage const& stage) {
  uint64_t stamp = 1469598103934665603ull;
  auto mix = [&stamp](uint64_t value) { stamp = (stamp ^ value) * 1099511628211ull; };
  // Widened to double first, so the stamp does not depend on whether mochi::real is float.
  auto mixReal = [&mix](real value) { mix(std::bit_cast<uint64_t>(static_cast<double>(value))); };
  auto const& actors = stage.GetActors();
  mix(actors.size());
  for (StagedActor const& actor : actors) {
    Real3 const translation = actor.worldTransform.GetTranslation();
    mochi::Quaternion const rotation = actor.worldTransform.GetRotation();
    for (int i = 0; i < 3; ++i) {
      mixReal(translation[i]);
    }
    for (int i = 0; i < 4; ++i) {
      mixReal(rotation.data[i]);
    }
  }
  return stamp;
}

bool IsSamePick(MeasureEntity const& a, MeasureEntity const& b) {
  return a.kind == b.kind && a.targetKind == b.targetKind && a.targetLabel == b.targetLabel &&
      a.points == b.points;
}

} // namespace

//--------------------------------------------------------------------------------------------------
// GEOMETRY CACHE
//--------------------------------------------------------------------------------------------------

std::shared_ptr<MeasureMesh const> MeasureGeometryCache::GetForModelFile(mochi::Path const& path) {
  if (auto const it = _byFile.find(path); it != _byFile.end()) {
    return it->second;
  }

  std::shared_ptr<MeasureMesh const> mesh;
  if (IsMochiModelPath(path)) {
    // Both loaders return renderer-space geometry, matching the SceneObject staged from the same
    // file. See the space convention in processing_mesh_utils.h.
    mochi::ErrorLog error;
    mochi::MeshData const meshData = processing::LoadMochiMesh(path.ToString(), error);
    mesh = BuildMeasureMesh(mochi::MeshDataView(meshData), _params);
  } else {
    std::vector<mochi_renderer::MeshSection> const sections =
        processing::ReadSectionsInRenderSpace(path.ToString());
    std::vector<MeasureMeshSection> views;
    views.reserve(sections.size());
    for (mochi_renderer::MeshSection const& section : sections) {
      views.push_back(
          {mochi::MakeConstSpan(section.positions), mochi::MakeConstSpan(section.indices)});
    }
    mesh = BuildMeasureMesh(mochi::MakeConstSpan(views), _params);
  }
  // Cache the empty result too: a model with no measurable triangles (an SDF-only collision model,
  // say) must not be re-read on every hover.
  _byFile.emplace(path, mesh);
  return mesh;
}

std::shared_ptr<MeasureMesh const> MeasureGeometryCache::GetForSections(
    void const* owner,
    std::vector<mochi_renderer::MeshSection> const& sections) {
  uint64_t const stamp = SectionsChangeStamp(sections);
  SectionsEntry& entry = _bySections[owner];
  if (entry.mesh && entry.stamp == stamp) {
    return entry.mesh;
  }
  std::vector<MeasureMeshSection> views;
  views.reserve(sections.size());
  for (mochi_renderer::MeshSection const& section : sections) {
    views.push_back(
        {mochi::MakeConstSpan(section.positions), mochi::MakeConstSpan(section.indices)});
  }
  entry.stamp = stamp;
  entry.mesh = BuildMeasureMesh(mochi::MakeConstSpan(views), _params);
  return entry.mesh;
}

void MeasureGeometryCache::SetParams(MeasureMeshParams params) {
  _params = params;
  Clear();
}

void MeasureGeometryCache::Clear() {
  _byFile.clear();
  _bySections.clear();
}

void CollectSceneStageMeasureTargets(
    SceneStage const& stage,
    MeasureGeometryCache& cache,
    bool excludeDeformable,
    std::vector<MeasureTarget>& out) {
  for (StagedActor const& actor : stage.GetActors()) {
    if (excludeDeformable && DeformsDuringSimulation(actor)) {
      continue;
    }
    auto add = [&](StagedActor::Instance const& instance, MeasureTargetKind kind) {
      if (instance.sceneObject == nullptr || instance.modelFile.IsEmpty()) {
        return;
      }
      mochi::Path const file = instance.modelFile;
      out.push_back({actor.name, kind, instance.sceneObject, [&cache, file] {
                       return cache.GetForModelFile(file);
                     }});
    };
    add(actor.render, MeasureTargetKind::Render);
    add(actor.shape, MeasureTargetKind::Collision);
  }
}

void BindSceneStageMeasureTargets(
    Viewport& viewport,
    SceneStage const& stage,
    std::function<bool()> const& isSimulating,
    std::function<bool()> const& isPaused) {
  MeasureTool* const tool = viewport.GetMeasureTool();
  tool->SetTargetProvider([tool, &stage, isSimulating](std::vector<MeasureTarget>& out) {
    bool const simulating = isSimulating && isSimulating();
    CollectSceneStageMeasureTargets(stage, tool->GetGeometryCache(), simulating, out);
  });
  tool->SetPoseStampProvider([&stage] { return StagedPoseStamp(stage); });
  if (isSimulating) {
    tool->SetAvailablePredicate([isSimulating, isPaused] {
      // A running simulation moves the staged transforms out from under any measurement taken
      // against them; paused, they hold still, so the numbers describe what is on screen.
      return !isSimulating() || (isPaused && isPaused());
    });
  }
}

//--------------------------------------------------------------------------------------------------
// TOOL
//--------------------------------------------------------------------------------------------------

char const* MeasureTool::GetWindowName() {
  return "Measure";
}

MeasureTool::MeasureTool(SuperDexStudio* studio, Viewport* viewport)
    : _studio(studio), _viewport(viewport), _cache(MeasureMeshParams{}) {}

MeasureTool::~MeasureTool() = default;

void MeasureTool::SetTargetProvider(TargetProvider provider) {
  _targetProvider = std::move(provider);
}

void MeasureTool::SetAvailablePredicate(std::function<bool()> predicate) {
  _availablePredicate = std::move(predicate);
}

bool MeasureTool::IsAvailable() const {
  return !_availablePredicate || _availablePredicate();
}

void MeasureTool::SetPoseStampProvider(std::function<uint64_t()> provider) {
  _poseStampProvider = std::move(provider);
}

uint64_t MeasureTool::CurrentPoseStamp() const {
  return _poseStampProvider ? _poseStampProvider() : 0;
}

void MeasureTool::DiscardSelectionIfPoseChanged() {
  if (!_poseStampProvider) {
    return;
  }
  uint64_t const stamp = _poseStampProvider();
  if (stamp == _selectionPoseStamp) {
    return;
  }
  _selectionPoseStamp = stamp;
  ClearHover();
  if (!_selection.empty()) {
    ClearSelection();
  }
}

void MeasureTool::RevealWindow() {
  _studio->GetWindowVisible(GetWindowName()) = true;
  _pendingWindowFocus = true;
}

void MeasureTool::ToggleFromShortcut() {
  if (_active && !_studio->GetWindowVisible(GetWindowName())) {
    RevealWindow();
    return;
  }
  SetActive(!_active);
}

void MeasureTool::SetActive(bool active) {
  if (active == _active) {
    return;
  }
  if (active && !IsAvailable()) {
    return;
  }
  _active = active;
  if (_active) {
    // The tool is useless without its readout, so switching it on brings the panel up.
    RevealWindow();
    // Take over left-click: a measure click must not also re-select a link or start a force-drag.
    // The previous values are restored.
    _savedEnableViewportPicking = _viewport->enableViewportPicking;
    _savedEnableSceneObjectDrag = _viewport->enableSceneObjectDrag;
    _viewport->enableViewportPicking = false;
    _viewport->enableSceneObjectDrag = false;
  } else {
    _viewport->enableViewportPicking = _savedEnableViewportPicking;
    _viewport->enableSceneObjectDrag = _savedEnableSceneObjectDrag;
    ClearSelection();
    _cache.Clear();
  }
  ClearHover();
}

void MeasureTool::ClearSelection() {
  _selection.clear();
  _report = {};
  _activeRecord = -1;
}

void MeasureTool::ClearHover() {
  _hover.clear();
  _hoverKey = {};
}

void MeasureTool::RecomputeReport() {
  _report = ComputeMeasurements(_selection, _reportParams);
}

void MeasureTool::ApplyMeshParams() {
  _cache.SetParams(_meshParams);
}

bool MeasureTool::IsTargetPickable(MeasureTarget const& target) const {
  if (target.object == nullptr || !target.getMesh) {
    return false;
  }
  // Only what the user can see
  if (!target.object->GetVisible()) {
    return false;
  }
  switch (target.kind) {
    case MeasureTargetKind::Render:
      return _pickRender;
    case MeasureTargetKind::Collision:
      return _pickCollision;
  }
  return false;
}

void MeasureTool::ProcessCursorRay(
    filament::math::float3 rayOrigin,
    filament::math::float3 rayDirection,
    bool commit,
    bool additive) {
  if (!_active) {
    ClearHover();
    return;
  }
  _targets.clear();
  if (_targetProvider) {
    _targetProvider(_targets);
  }

  mochi::Ray const worldRay(
      mochi_renderer::ToMochi<real>(rayOrigin), mochi_renderer::ToMochi<real>(rayDirection));
  MeasureTarget const* bestTarget = nullptr;
  std::shared_ptr<MeasureMesh const> bestMesh;
  MeasureRayHit bestHit;
  filament::math::mat4f bestWorldFromLocal;
  float bestWorldDistance = 0.0f;
  bool haveHit = false;

  for (MeasureTarget const& target : _targets) {
    if (!IsTargetPickable(target)) {
      continue;
    }
    // Reject against the object's world bounds before asking for its geometry
    filament::Box const bounds = target.object->GetAABB();
    filament::math::float3 const boundsMin = bounds.center - bounds.halfExtent;
    filament::math::float3 const boundsMax = bounds.center + bounds.halfExtent;
    if (!mochi::RayCast(
            worldRay,
            mochi::Aabb(
                mochi_renderer::ToMochi<real>(boundsMin),
                mochi_renderer::ToMochi<real>(boundsMax)))) {
      continue;
    }
    std::shared_ptr<MeasureMesh const> mesh = target.getMesh();
    if (!mesh || mesh->IsEmpty()) {
      continue;
    }
    filament::math::mat4f const worldFromLocal = target.object->GetWorldTransform();
    filament::math::mat4f const localFromWorld = inverse(worldFromLocal);
    filament::math::float3 const localOrigin =
        (localFromWorld * filament::math::float4{rayOrigin, 1.0f}).xyz;
    filament::math::float3 const localDirection =
        (localFromWorld * filament::math::float4{rayDirection, 0.0f}).xyz;
    if (length(localDirection) <= 0.0f) {
      continue;
    }
    std::optional<MeasureRayHit> const hit = RayCastMeasureMesh(
        *mesh,
        mochi_renderer::ToMochi<real>(localOrigin),
        mochi_renderer::ToMochi<real>(localDirection));
    if (!hit) {
      continue;
    }
    filament::math::float3 const worldPoint =
        (worldFromLocal *
         filament::math::float4{mochi_renderer::ToFilament<float>(hit->point), 1.0f})
            .xyz;
    float const worldDistance = length(worldPoint - rayOrigin);
    if (!haveHit || worldDistance < bestWorldDistance) {
      haveHit = true;
      bestWorldDistance = worldDistance;
      bestTarget = &target;
      bestMesh = std::move(mesh);
      bestHit = *hit;
      bestWorldFromLocal = worldFromLocal;
    }
  }

  if (!haveHit) {
    ClearHover();
    if (commit && !additive) {
      ClearSelection();
    }
    return;
  }

  HoverKey const key{
      bestTarget->object, bestHit.triangle, bestHit.facet, bestHit.nearestVertex, _pickMode};
  if (!commit && key == _hoverKey && !_hover.empty()) {
    return; // same thing under the cursor as last frame; keep the entity we already built
  }

  // Local (render space) -> world (render space) -> editor space
  auto const& rendererToEditor = _studio->GetRendererToEditorSpaceConverter();
  auto localToMeasure = [&bestWorldFromLocal, &rendererToEditor](Real3 const& localPoint) {
    filament::math::float3 const world =
        (bestWorldFromLocal *
         filament::math::float4{mochi_renderer::ToFilament<float>(localPoint), 1.0f})
            .xyz;
    return rendererToEditor.TranslationToOutput(mochi_renderer::ToMochi<real>(world));
  };
  MeasureEntity entity = MakeMeasureEntity(
      *bestMesh, bestHit, _pickMode, bestTarget->label, bestTarget->kind, localToMeasure);

  if (!commit) {
    _hover.assign(1, entity);
    _hoverKey = key;
    return;
  }
  ClearHover();
  if (!additive) {
    ClearSelection();
  } else if (auto const existing = std::ranges::find_if(
                 _selection,
                 [&entity](MeasureEntity const& picked) { return IsSamePick(picked, entity); });
             existing != _selection.end()) {
    _selection.erase(existing);
    _activeRecord = -1;
    RecomputeReport();
    return;
  }
  _selection.push_back(std::move(entity));
  _selectionPoseStamp = CurrentPoseStamp();
  _activeRecord = -1;
  RecomputeReport();
}

void MeasureTool::DrawDebug(mochi_renderer::DebugDraw* debugDraw, DebugText* debugText) const {
  if (!_active || debugDraw == nullptr) {
    return;
  }
  auto const& editorToRenderer = _studio->GetEditorToRendererSpaceConverter();
  filament::math::double3 const cameraPos = _viewport->GetRenderScene()->GetCameraPosition();
  filament::math::float3 const camera{
      static_cast<float>(cameraPos.x),
      static_cast<float>(cameraPos.y),
      static_cast<float>(cameraPos.z)};

  auto toRender = [&editorToRenderer](Real3 const& measurePoint) {
    return mochi_renderer::ToFilament<float>(editorToRenderer.TranslationToOutput(measurePoint));
  };

  auto drawEntities = [&](std::vector<MeasureEntity> const& entities,
                          filament::math::float4 lineColor,
                          filament::math::float4 fillColor) {
    for (MeasureEntity const& entity : entities) {
      if (!entity.IsFace()) {
        filament::math::float3 const point = toRender(entity.points.front());
        float const radius = ScreenScaledSize(
            point, camera, kMarkerScreenFraction, kMarkerMinRadius, kMarkerMaxRadius);
        // tessellated overload
        debugDraw->DrawSolidSphere(
            point, radius, lineColor, kMarkerStacks, kMarkerSlices, /*overlay=*/true);
        continue;
      }
      for (std::size_t i = 0; i + 2 < entity.triangles.size(); i += 3) {
        debugDraw->DrawSolidTriangle(
            toRender(entity.points[static_cast<std::size_t>(entity.triangles[i + 0])]),
            toRender(entity.points[static_cast<std::size_t>(entity.triangles[i + 1])]),
            toRender(entity.points[static_cast<std::size_t>(entity.triangles[i + 2])]),
            fillColor,
            /*overlay=*/true);
      }
      for (std::size_t i = 0; i + 1 < entity.boundaryEdges.size(); i += 2) {
        DrawOverlayLine(
            debugDraw,
            toRender(entity.points[static_cast<std::size_t>(entity.boundaryEdges[i + 0])]),
            toRender(entity.points[static_cast<std::size_t>(entity.boundaryEdges[i + 1])]),
            camera,
            lineColor);
      }
    }
  };

  drawEntities(_hover, kHoverColor, kHoverFillColor);
  drawEntities(_selection, kSelectionColor, kSelectionFillColor);

  for (MeasureAnnotation const& annotation : _report.annotations) {
    bool const isAxisLeg = annotation.axis >= 0;
    filament::math::float3 const from = toRender(annotation.from);
    filament::math::float3 const to = toRender(annotation.to);
    filament::math::float4 const color = isAxisLeg ? AxisColor(annotation.axis) : kSpanColor;
    DrawOverlayLine(debugDraw, from, to, camera, color);
    bool const showLabel = !isAxisLeg || _showAxisLabels;
    if (debugText != nullptr && showLabel && !annotation.label.empty()) {
      debugText->Draw((from + to) * 0.5f, annotation.label, color);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// PANEL
//--------------------------------------------------------------------------------------------------

void MeasureTool::ShowOptions() {
  bool const available = IsAvailable();
  ImGui::BeginDisabled(!available);
  bool active = _active;
  if (ImGui::Checkbox("Enable Measure Tool", &active)) {
    SetActive(active);
  }
  ImGui::EndDisabled();
  if (!available) {
    ImGui::TextDisabled("Unavailable while the simulation is running. Pause to measure.");
  } else if (!_active) {
    ImGui::TextDisabled("Enable (Ctrl + M), then click geometry in the viewport.");
  }

  if (!ImGui::CollapsingHeader("Configuration", ImGuiTreeNodeFlags_DefaultOpen)) {
    return;
  }

  ImGui::HoverableSeparatorText("Pick");
  int pickMode = static_cast<int>(_pickMode);
  bool pickModeChanged =
      ImGui::RadioButton("Vertex", &pickMode, static_cast<int>(MeasurePickMode::Vertex));
  ImGui::SameLine();
  pickModeChanged |=
      ImGui::RadioButton("Triangle", &pickMode, static_cast<int>(MeasurePickMode::Triangle));
  ImGui::SameLine();
  pickModeChanged |=
      ImGui::RadioButton("Facet", &pickMode, static_cast<int>(MeasurePickMode::Facet));
  ImGui::ItemTooltipWrapped(
      "A facet is the coplanar group of triangles around the one under the cursor; Triangle picks "
      "just that one triangle.");
  if (pickModeChanged) {
    _pickMode = static_cast<MeasurePickMode>(pickMode);
    ClearHover();
  }

  ImGui::HoverableSeparatorText("Target");
  ImGui::Checkbox("Render", &_pickRender);
  ImGui::SameLine();
  ImGui::Checkbox("Collision", &_pickCollision);
  ImGui::ItemTooltipWrapped(
      "Only representations currently visible in the viewport can be picked.");

  ImGui::HoverableSeparatorText("Viewport");
  ImGui::Checkbox("Axis labels", &_showAxisLabels);
  ImGui::ItemTooltipWrapped("Show the X/Y/Z values on each axis.");

  ImGui::HoverableSeparatorText("Report");
  char const* const kUnitNames[] = {"m", "cm", "mm"};
  int unit = static_cast<int>(_reportParams.unit);
  if (ImGui::Combo("Units", &unit, kUnitNames, IM_ARRAYSIZE(kUnitNames))) {
    _reportParams.unit = static_cast<MeasureUnit>(unit);
    RecomputeReport();
  }

  auto coplanarTolerance = static_cast<float>(_meshParams.facetAngleToleranceDeg);
  if (ImGui::SliderFloat("Coplanar tol.", &coplanarTolerance, 0.05f, 45.0f, "%.2f deg")) {
    _meshParams.facetAngleToleranceDeg = static_cast<real>(coplanarTolerance);
    ApplyMeshParams();
  }
  ImGui::ItemTooltipWrapped(
      "How far two neighbouring triangles' normals may differ and still belong to the same facet. "
      "Raise it to treat a gently curved surface as one face.");

  auto parallelTolerance = static_cast<float>(_reportParams.parallelAngleToleranceDeg);
  if (ImGui::SliderFloat("Parallel tol.", &parallelTolerance, 0.01f, 10.0f, "%.2f deg")) {
    _reportParams.parallelAngleToleranceDeg = static_cast<real>(parallelTolerance);
    RecomputeReport();
  }
  ImGui::ItemTooltipWrapped(
      "How close to parallel two faces must be to report a plane-to-plane gap.");
}

void MeasureTool::ShowSelectionList() {
  std::string const header =
      "Selection (" + std::to_string(_selection.size()) + ")###MeasureSelection";
  if (!ImGui::CollapsingHeader(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
    return;
  }
  ImGui::BeginDisabled(_selection.empty());
  if (ImGui::Button("Clear")) {
    ClearSelection();
  }
  ImGui::EndDisabled();

  if (_selection.empty()) {
    ImGui::TextDisabled("Click geometry in the viewport. Ctrl/Shift-click to add or remove.");
    return;
  }
  int removeIndex = -1;
  ImGui::PushID("Selection");
  for (std::size_t i = 0; i < _selection.size(); ++i) {
    MeasureEntity const& entity = _selection[i];
    ImGui::PushID(static_cast<int>(i));
    // A framed Button, not SmallButton: SmallButton zeroes the vertical frame padding, which makes
    // the row exactly one font cell tall and clips the top and bottom of the taller trash glyph.
    if (ImGui::Button(ICON_FA_TRASH)) {
      removeIndex = static_cast<int>(i);
    }
    ImGui::SetItemTooltip("Remove from the selection");
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::Text(
        "[%d] %s  %s (%s)",
        static_cast<int>(i + 1),
        entity.GetDescription().c_str(),
        entity.targetLabel.c_str(),
        GetMeasureTargetKindLabel(entity.targetKind));
    ImGui::PopID();
  }
  ImGui::PopID();
  if (removeIndex >= 0) {
    _selection.erase(_selection.begin() + removeIndex);
    _activeRecord = -1;
    RecomputeReport();
  }
}

void MeasureTool::RecordSelection() {
  MeasureRecord record;
  std::string value;
  for (MeasureAnnotation const& annotation : _report.annotations) {
    if (annotation.style == MeasureAnnotation::Style::Span && !annotation.label.empty()) {
      value = annotation.label;
      break;
    }
  }
  record.label = std::to_string(_nextRecordNumber++) + ". ";
  record.label += value.empty() ? "measurement" : value;
  record.label += "  " + _selection.front().targetLabel;
  if (_selection.back().targetLabel != _selection.front().targetLabel) {
    record.label += " -> " + _selection.back().targetLabel;
  }
  record.report = _report;
  record.entities = _selection;
  _records.push_back(std::move(record));
  _activeRecord = static_cast<int>(_records.size()) - 1;
}

void MeasureTool::ShowRecords() {
  ImGui::BeginDisabled(_selection.size() < 2);
  if (ImGui::Button("Record")) {
    RecordSelection();
  }
  ImGui::EndDisabled();
  ImGui::ItemTooltipWrapped("Store this measurement so you can click back to it later.");
  ImGui::SameLine();
  ImGui::BeginDisabled(_records.empty());
  if (ImGui::Button("Clear All##Records")) {
    _records.clear();
    _activeRecord = -1;
  }
  ImGui::EndDisabled();

  if (_records.empty()) {
    ImGui::TextDisabled("No recorded measurements.");
    return;
  }
  ImGui::TextDisabled(
      "Recorded highlights stay where the geometry was when recorded, and are lost on exit.");

  int removeIndex = -1;
  ImGui::PushID("Records");
  for (std::size_t i = 0; i < _records.size(); ++i) {
    MeasureRecord const& record = _records[i];
    ImGui::PushID(static_cast<int>(i));
    // See the selection list: a framed Button so the trash glyph is not clipped.
    if (ImGui::Button(ICON_FA_TRASH)) {
      removeIndex = static_cast<int>(i);
    }
    ImGui::SetItemTooltip("Delete this recorded measurement");
    ImGui::SameLine();
    // Match the button's height so the whole row highlights, and center the label in it (Selectable
    // otherwise top-aligns its text once given an explicit height).
    ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2(0.0f, 0.5f));
    bool const restore = ImGui::Selectable(
        record.label.c_str(),
        _activeRecord == static_cast<int>(i),
        ImGuiSelectableFlags_None,
        {0.0f, ImGui::GetFrameHeight()});
    ImGui::PopStyleVar();
    if (restore) {
      // Restoring re-highlights the recorded entities and re-draws their measurement
      _selection = record.entities;
      _report = record.report;
      _selectionPoseStamp = CurrentPoseStamp();
      _activeRecord = static_cast<int>(i);
    }
    if (ImGui::BeginPopupContextItem("##RecordMenu")) {
      if (ImGui::Selectable("Copy")) {
        ImGui::SetClipboardText(record.report.text.c_str());
      }
      if (ImGui::Selectable("Delete")) {
        removeIndex = static_cast<int>(i);
      }
      ImGui::EndPopup();
    }
    ImGui::ItemTooltipWrapped(record.report.text.c_str());
    ImGui::PopID();
  }
  ImGui::PopID();
  if (removeIndex >= 0) {
    _records.erase(_records.begin() + removeIndex);
    _activeRecord = -1;
  }
}

void MeasureTool::ShowWindow(char const* name, bool* open) {
  if (_pendingWindowFocus) {
    ImGui::SetNextWindowFocus();
    _pendingWindowFocus = false;
  }
  ImGui::Begin(name, open);
  ShowOptions();
  ShowSelectionList();

  if (ImGui::CollapsingHeader("Result", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::BeginDisabled(_report.text.empty());
    if (ImGui::Button("Copy")) {
      ImGui::SetClipboardText(_report.text.c_str());
    }
    ImGui::EndDisabled();
    ImGui::ReadOnlyTextBlock(
        "##MeasureResult",
        _report.text.empty() ? std::string("(nothing measured)") : _report.text,
        _studio->GetFont("Roboto Mono"),
        /*maxLines=*/16);
  }

  ImGui::Separator();
  ShowRecords();
  ImGui::End();
}

} // namespace superdex::studio
