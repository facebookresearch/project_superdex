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

#include "rendering/viewport.h"
#include "app/app.h"
#include "rendering/scene_stage.h"
#include "ui/imgui_widgets.h"

#include <mochi_core/mochi_platform.h>

#include <mochi_renderer/material.h>
#include <mochi_renderer/type_conversions.h>

#include <filament/Camera.h>
#include <filament/IndexBuffer.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/TextureSampler.h>
#include <filament/VertexBuffer.h>
#include <filament/View.h>
#include <filament/Viewport.h>
#include <utils/EntityManager.h>

#include <imgui_internal.h>
#include <imguios/fonts/icons_font_awesome5.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

using namespace mochi_renderer;

namespace superdex::studio {

constexpr int kDefaultViewportWidth = 512;
constexpr int kDefaultViewportHeight = 512;

// Highlight appearance, shared by every highlight source (selection + hovers).
constexpr filament::math::float3 kSelectionColor = kViewportSelectionColor;

// Camera world position and normalized forward axis, extracted from Filament's camera model matrix
// (column 3 is the translation, the negated column 2 is the forward axis).
struct CameraFrame {
  filament::math::float3 position;
  filament::math::float3 forward;
};

static CameraFrame GetCameraFrame(mochi_renderer::Scene const& scene) {
  filament::math::mat4 const model = scene.GetCameraModelMatrix();
  return {
      filament::math::float3{
          static_cast<float>(model[3].x),
          static_cast<float>(model[3].y),
          static_cast<float>(model[3].z)},
      normalize(
          filament::math::float3{
              -static_cast<float>(model[2].x),
              -static_cast<float>(model[2].y),
              -static_cast<float>(model[2].z)})};
}

// Draws the ground grid on the render-space XZ plane at `height`, as thin depth-tested cylinders.
static void DrawGroundGrid(
    mochi_renderer::DebugDraw* debugDraw,
    GroundGridSettings const& settings,
    float height) {
  // Negated comparisons so a NaN from a hand-edited settings file also bails out.
  if (!debugDraw || !(settings.gridSpacing > 0.0f) || !(settings.gridExtents > 0.0f)) {
    return;
  }
  constexpr float kGridLineRadius = 0.0005f; // 0.5mm radius for grid lines
  // The reflected settings are range-clamped, but a hand-edited file is not, and the loop below
  // submits geometry every frame for every viewport. Beyond this the grid is unreadable anyway.
  constexpr int kMaxGridLines = 1024;
  float const halfSize = settings.gridExtents * 0.5f;
  // Computed in double and clamped before the cast: the ratio can overflow int.
  int const numLines = static_cast<int>(std::min(
      std::round(static_cast<double>(settings.gridExtents) / settings.gridSpacing) + 1.0,
      static_cast<double>(kMaxGridLines)));
  filament::math::float4 const gridColor = ToFilament(settings.gridColor);
  for (int i = 0; i < numLines; ++i) {
    float const offset = -halfSize + i * settings.gridSpacing;
    // Line parallel to X axis (constant Z)
    filament::math::float3 const startX{-halfSize, height, offset};
    filament::math::float3 const dirX{settings.gridExtents, 0.0f, 0.0f};
    debugDraw->DrawSolidCylinder(startX, dirX, kGridLineRadius, gridColor, 4, true, false);
    // Line parallel to Z axis (constant X)
    filament::math::float3 const startZ{offset, height, -halfSize};
    filament::math::float3 const dirZ{0.0f, 0.0f, settings.gridExtents};
    debugDraw->DrawSolidCylinder(startZ, dirZ, kGridLineRadius, gridColor, 4, true, false);
  }
}

// Single source of truth for the square button size (width == height) shared by every viewport
// toolbar, so all toolbar buttons are identical. Sized to the regular control height, but at least
// wide enough to fit the widest icon used by any toolbar so glyphs stay centered.
static float ToolbarButtonSize() {
  ImGuiStyle const& style = ImGui::GetStyle();
  char const* const kToolbarIcons[] = {
      ICON_FA_ARROWS_ALT,
      ICON_FA_SYNC_ALT,
      ICON_FA_EXPAND_ARROWS_ALT,
      ICON_FA_CUBE,
      ICON_FA_GLOBE,
      ICON_FA_HOME,
      ICON_FA_VIDEO,
      ICON_FA_CUBES,
  };
  float maxIconWidth = 0.0f;
  for (char const* icon : kToolbarIcons) {
    maxIconWidth = std::max(maxIconWidth, ImGui::CalcTextSize(icon).x);
  }
  return std::max(ImGui::GetFrameHeight(), maxIconWidth + style.FramePadding.x * 2.0f);
}

// Draws a small camera-with-speed-lines glyph filling the given square cell, used to label the
// viewport fly-speed field. Hand-drawn because no FontAwesome glyph combines a camera with motion
// lines. All geometry is expressed as fractions of the cell so it scales with the toolbar.
static void DrawCameraSpeedIcon(ImVec2 const cellMin, float const cellSize, ImU32 const color) {
  ImDrawList* const drawList = ImGui::GetWindowDrawList();
  float const thickness = std::max(1.0f, cellSize * 0.07f);
  auto at = [cellMin, cellSize](float fx, float fy) {
    return ImVec2{cellMin.x + fx * cellSize, cellMin.y + fy * cellSize};
  };
  float const rounding = cellSize * 0.08f;
  // Camera body occupies the right portion of the cell.
  drawList->AddRect(
      at(0.40f, 0.30f), at(0.95f, 0.72f), color, rounding, ImDrawFlags_RoundCornersAll, thickness);
  // Viewfinder bump on top of the body and the lens.
  drawList->AddRectFilled(at(0.50f, 0.22f), at(0.66f, 0.31f), color, rounding * 0.5f);
  drawList->AddCircle(at(0.69f, 0.51f), cellSize * 0.11f, color, 0, thickness);
  // Speed/motion lines trailing to the left of the body, staggered to suggest movement.
  drawList->AddLine(at(0.02f, 0.40f), at(0.36f, 0.40f), color, thickness);
  drawList->AddLine(at(0.10f, 0.51f), at(0.36f, 0.51f), color, thickness);
  drawList->AddLine(at(0.04f, 0.62f), at(0.36f, 0.62f), color, thickness);
}

Viewport::Viewport(SuperDexStudio* studio, mochi_renderer::SceneViewSettings const& viewSettings)
    : _studio(studio) {
  auto engine = _studio->GetEngine();
  _renderScene = mochi_renderer::Scene::Create(engine, viewSettings);
  // we give the user access to the scene, so we need to know when they
  // destroy objects so we can clean selection state.
  _renderScene->onDestroySceneObject = [this](SceneObject* object) {
    auto it = std::find(_selectedObjects.begin(), _selectedObjects.end(), object);
    if (it != _selectedObjects.end()) {
      _selectedObjects.erase(it);
      if (onSceneSelectionChanged) {
        onSceneSelectionChanged(_selectedObjects);
      }
    }
    // Highlight clones are owned by the SceneStage and destroyed alongside their bases on
    // restage/clear (see SceneStage::Clear), so there is nothing to clean up here.
  };
  _renderScene->SetViewport(kDefaultViewportWidth, kDefaultViewportHeight);
  _renderScene->CreateSkybox();
  _renderScene->CreateSunlight();
  _renderScene->CreateIndirectLight();
  _renderScene->CreateGroundPlane();
  _renderScene->SetIbl(_studio->GetCurrentIbl());
  _renderScene->SetSkyboxVisible(false);
  _renderScene->CreateDebugDraw();
  auto const& converter = _studio->GetEditorToRendererSpaceConverter();
  auto const from = mochi_renderer::ToFilament<double>(
      converter.TranslationToOutput(mochi::Real3{1_r, 1_r, 0.5_r}));
  auto const to = mochi_renderer::ToFilament<double>(
      converter.TranslationToOutput(mochi::Real3{0_r, 0_r, 0_r}));
  auto const up =
      mochi_renderer::ToFilament<double>(converter.DirectionToOutput(mochi::Real3{0_r, 0_r, 1_r}));
  _renderScene->CameraLookAt(from, to, up);
  _renderTarget = RenderTarget::Create(engine, kDefaultViewportWidth, kDefaultViewportHeight);
  _cameraController = CameraController::Create(_renderScene.get());
  // Start at the last-used fly speed so newly opened viewports keep the previous feel.
  _cameraController->SetMoveSpeed(_studio->GetAppSettings().viewport.camera.flySpeed);
  // Restore the persisted highlight overlay opacity.
  _highlightOverlayAlpha =
      static_cast<float>(_studio->GetAppSettings().viewport.selection.highlightOverlayOpacity);
  // Every viewport offers the ground grid; its visibility is per-viewport (and so per-editor) and
  // seeded here, while its size/spacing/color are app-wide (AppSettings::viewport::groundGrid).
  _showGroundGrid = _studio->GetAppSettings().viewport.groundGrid.showByDefault;
  RegisterShowCommand(
      {.name = "Grid",
       .onToggle = [this] { _showGroundGrid = !_showGroundGrid; },
       .getState = [this] { return _showGroundGrid; },
       .shortcut = ImGuiKey_G});
}

Viewport::~Viewport() {
  DestroyHighlightOverlay();
}

std::unique_ptr<Viewport> Viewport::Create(
    SuperDexStudio* studio,
    mochi_renderer::SceneViewSettings const& viewSettings) {
  return std::unique_ptr<Viewport>(new Viewport(studio, viewSettings));
}

mochi_renderer::Scene* Viewport::GetRenderScene() const {
  return _renderScene.get();
}

float Viewport::UpdateGroundPlane() {
  if (!_renderScene) {
    return 0.0f;
  }
  float const height = _renderScene->ComputeGroundPlaneHeight();
  _renderScene->SetGroundPlaneHeight(height);
  // The grid is drawn at the same height, so it lines up with the drop-shadow plane and with the
  // studio physics ground plane (which editors position from this same value).
  _groundPlaneHeight = height;
  return height;
}

void Viewport::RenderScene(Renderer const* renderer) {
  if (!_renderScene) {
    return;
  }
  DrawDebug();
  // Pass the camera position so the "on top" debug overlays (joint limits, transform axes, etc.)
  // are sorted back-to-front and nearer ones paint over farther ones instead of by submission
  // order.
  filament::math::double3 const camPos = _renderScene->GetCameraPosition();
  _renderScene->GetDebugDraw()->Commit(
      filament::math::float3{
          static_cast<float>(camPos.x),
          static_cast<float>(camPos.y),
          static_cast<float>(camPos.z)});

  // Selection feeds the same highlight path as hovers, re-applied every frame. Hover requests are
  // declared earlier during ImGui, so (RequestHighlight being first-wins per frame) an already-
  // requested hovered link keeps its hover color; otherwise the selected link gets the selection
  // color. The stage then reconciles the per-link clones and hides/shows base meshes.
  if (_stage) {
    for (SceneObject* selected : _selectedObjects) {
      _stage->RequestHighlight(_stage->GetSceneObjectIndex(selected), kSelectionColor);
    }
    _stage->UpdateHighlights();
  }

  // When anything is highlighted, drive the see-through overlay: an isolated re-render of the
  // highlight clones composited over the main image (see EnsureHighlightOverlay / HighlightPass).
  HighlightPass highlight;
  HighlightPass const* highlightPtr = nullptr;
  if (_stage && _stage->HasActiveHighlights()) {
    int width = 0, height = 0;
    _renderTarget->GetSize(width, height);
    EnsureHighlightOverlay(width, height);
    // Push the current (view-settings-adjustable) opacity to the composite material each frame.
    if (_compositeMaterial) {
      _compositeMaterial->Get()->setParameter("alpha", _highlightOverlayAlpha);
    }
    highlight.overlayView = _overlayView;
    highlight.overlayTarget = _overlayTarget.get();
    highlight.compositeView = _compositeView;
    highlightPtr = &highlight;
  }

  renderer->Render(_renderScene.get(), _renderTarget.get(), true, highlightPtr);
  _renderScene->GetDebugDraw()->Clear();
}

void Viewport::EnsureHighlightOverlay(int width, int height) {
  width = std::max(width, 1);
  height = std::max(height, 1);
  auto* engine = _studio->GetEngine();
  auto& em = utils::EntityManager::get();

  if (!_overlayView) {
    // Overlay pass: a second view over the main scene + camera, filtered to the highlight-overlay
    // layer, rendering into its own target. Its isolated (cleared) depth means only the nearest
    // surface shows and scene geometry can't occlude it.
    _overlayTarget = RenderTarget::Create(engine, width, height);
    _overlayView = engine->createView();
    _overlayView->setScene(_renderScene->GetFilamentScene());
    _overlayView->setCamera(_renderScene->GetCamera());
    // Replace the whole layer mask (select 0xFF) with only the overlay bit, so the pass renders the
    // highlight clones alone -- not the default-0x01 scene geometry. (setVisibleLayers only touches
    // the selected bits, so selecting just the overlay bit would leave 0x01 enabled and draw the
    // entire scene.)
    _overlayView->setVisibleLayers(0xFF, kHighlightOverlayLayer);
    _overlayView->setPostProcessingEnabled(false);
    _overlayView->setShadowingEnabled(false);
    // Clear the overlay's own depth each frame (orthogonal to Renderer clear options, which only
    // clear color). Without this the clones depth-test against stale depth and lose coverage where
    // the scene would occlude them -- defeating the through-occluder isolation.
    _overlayView->setChannelDepthClearEnabled(0, true);
    // The overlay is a separate render, so give it its own MSAA to keep the composited silhouette
    // edge from looking jagged against the main image.
    filament::View::MultiSampleAntiAliasingOptions msaaOptions;
    msaaOptions.enabled = true;
    msaaOptions.sampleCount = 4;
    _overlayView->setMultiSampleAntiAliasingOptions(msaaOptions);
    _overlayView->setViewport({0, 0, static_cast<uint32_t>(width), static_cast<uint32_t>(height)});

    // Composite pass: a fullscreen triangle that samples the overlay target and blends it over the
    // main image (a TRANSLUCENT view). Geometry mirrors the distortion post-process: an oversized
    // triangle covering the viewport after clipping.
    struct Vertex {
      filament::math::float2 position;
      filament::math::float2 uv;
    };
#if MOCHI_PLATFORM_MACOS
    // Metal: texture origin is top-left, matches clip-space Y — no UV flip needed.
    static Vertex const kVertices[] = {
        {.position = {-1.0f, -1.0f}, .uv = {0.0f, 0.0f}},
        {.position = {3.0f, -1.0f}, .uv = {2.0f, 0.0f}},
        {.position = {-1.0f, 3.0f}, .uv = {0.0f, 2.0f}},
    };
#else
    // OpenGL: texture origin is bottom-left, flip V to sample top-to-bottom.
    static Vertex const kVertices[] = {
        {.position = {-1.0f, -1.0f}, .uv = {0.0f, 1.0f}},
        {.position = {3.0f, -1.0f}, .uv = {2.0f, 1.0f}},
        {.position = {-1.0f, 3.0f}, .uv = {0.0f, -1.0f}},
    };
#endif
    static uint16_t const kIndices[] = {0, 1, 2};
    _compositeVB = filament::VertexBuffer::Builder()
                       .vertexCount(3)
                       .bufferCount(1)
                       .attribute(
                           filament::VertexAttribute::POSITION,
                           0,
                           filament::backend::ElementType::FLOAT2,
                           0,
                           sizeof(Vertex))
                       .attribute(
                           filament::VertexAttribute::UV0,
                           0,
                           filament::backend::ElementType::FLOAT2,
                           sizeof(filament::math::float2),
                           sizeof(Vertex))
                       .build(*engine);
    _compositeVB->setBufferAt(
        *engine, 0, filament::VertexBuffer::BufferDescriptor(kVertices, sizeof(kVertices)));
    _compositeIB = filament::IndexBuffer::Builder()
                       .indexCount(3)
                       .bufferType(filament::IndexBuffer::IndexType::USHORT)
                       .build(*engine);
    _compositeIB->setBuffer(
        *engine, filament::IndexBuffer::BufferDescriptor(kIndices, sizeof(kIndices)));

    _compositeMaterial = _studio->GetResourceManager().CreateHighlightCompositeMaterial(
        _overlayTarget->GetColorTexture(), _highlightOverlayAlpha);

    _compositeQuadEntity = em.create();
    filament::RenderableManager::Builder(1)
        .geometry(
            0,
            filament::RenderableManager::PrimitiveType::TRIANGLES,
            _compositeVB,
            _compositeIB,
            0,
            3)
        .material(0, _compositeMaterial->Get())
        .culling(false)
        .receiveShadows(false)
        .castShadows(false)
        .boundingBox({.center = {-1, -1, -1}, .halfExtent = {1, 1, 1}})
        .build(*engine, _compositeQuadEntity);

    _compositeScene = engine->createScene();
    _compositeScene->addEntity(_compositeQuadEntity);

    _compositeCameraEntity = em.create();
    _compositeCamera = engine->createCamera(_compositeCameraEntity);
    _compositeCamera->setProjection(filament::Camera::Projection::ORTHO, -1, 1, -1, 1, 0, 1);

    _compositeView = engine->createView();
    _compositeView->setScene(_compositeScene);
    _compositeView->setCamera(_compositeCamera);
    _compositeView->setBlendMode(filament::View::BlendMode::TRANSLUCENT);
    _compositeView->setPostProcessingEnabled(false);
    _compositeView->setShadowingEnabled(false);
    _compositeView->setViewport(
        {0, 0, static_cast<uint32_t>(width), static_cast<uint32_t>(height)});
    return;
  }

  int ow = 0, oh = 0;
  _overlayTarget->GetSize(ow, oh);
  if (ow != width || oh != height) {
    _overlayTarget->Resize(width, height);
    filament::Viewport const vp{0, 0, static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    _overlayView->setViewport(vp);
    _compositeView->setViewport(vp);
    // Resize recreates the overlay color texture, so rebind it on the composite material.
    filament::TextureSampler const sampler(
        filament::TextureSampler::MinFilter::LINEAR, filament::TextureSampler::MagFilter::LINEAR);
    _compositeMaterial->Get()->setParameter(
        "sourceTexture", _overlayTarget->GetColorTexture(), sampler);
  }
}

void Viewport::DestroyHighlightOverlay() {
  if (_studio == nullptr) {
    return;
  }
  auto* engine = _studio->GetEngine();
  auto& em = utils::EntityManager::get();
  // Destroy the renderable before the material instance it references.
  if (_compositeView) {
    engine->destroy(_compositeView);
    _compositeView = nullptr;
  }
  if (_compositeCamera) {
    engine->destroyCameraComponent(_compositeCameraEntity);
    em.destroy(_compositeCameraEntity);
    _compositeCamera = nullptr;
  }
  if (_compositeScene) {
    engine->destroy(_compositeScene);
    _compositeScene = nullptr;
  }
  if (_compositeQuadEntity) {
    engine->destroy(_compositeQuadEntity);
    em.destroy(_compositeQuadEntity);
    _compositeQuadEntity = {};
  }
  if (_compositeIB) {
    engine->destroy(_compositeIB);
    _compositeIB = nullptr;
  }
  if (_compositeVB) {
    engine->destroy(_compositeVB);
    _compositeVB = nullptr;
  }
  _compositeMaterial.reset();
  if (_overlayView) {
    engine->destroy(_overlayView);
    _overlayView = nullptr;
  }
  _overlayTarget.reset();
}

void Viewport::SetSelectedSceneObjects(std::vector<SceneObject*> objects, bool invokeCallback) {
  // Drop nulls and duplicates while preserving order (back() stays the primary/active object).
  std::vector<SceneObject*> deduped;
  deduped.reserve(objects.size());
  for (SceneObject* object : objects) {
    if (object != nullptr && std::find(deduped.begin(), deduped.end(), object) == deduped.end()) {
      deduped.push_back(object);
    }
  }
  if (deduped == _selectedObjects) {
    return;
  }
  _selectedObjects = std::move(deduped);
  if (invokeCallback && onSceneSelectionChanged) {
    onSceneSelectionChanged(_selectedObjects);
  }
}

void Viewport::ToggleSceneObjectSelection(SceneObject* object, bool invokeCallback) {
  if (object == nullptr) {
    return;
  }
  auto it = std::find(_selectedObjects.begin(), _selectedObjects.end(), object);
  if (it != _selectedObjects.end()) {
    _selectedObjects.erase(it);
  } else {
    _selectedObjects.push_back(object);
  }
  if (invokeCallback && onSceneSelectionChanged) {
    onSceneSelectionChanged(_selectedObjects);
  }
}

void Viewport::SelectNeighborAndDestroy(SceneObject* object) {
  auto& objects = _renderScene->GetSceneObjects();
  if (std::find(_selectedObjects.begin(), _selectedObjects.end(), object) !=
      _selectedObjects.end()) {
    int idx = -1;
    for (int i = 0; i < (int)objects.size(); ++i) {
      if (objects[i] == object) {
        idx = i;
        break;
      }
    }
    if (idx != -1) {
      if ((int)objects.size() <= 1) {
        SetSelectedSceneObjects({});
      } else if (idx < (int)objects.size() - 1) {
        SetSelectedSceneObjects({objects[idx + 1]});
      } else {
        SetSelectedSceneObjects({objects[idx - 1]});
      }
    }
  }
  _renderScene->DestroySceneObject(object);
}

std::vector<SceneObject*> const& Viewport::GetSelectedSceneObjects() const {
  return _selectedObjects;
}

bool Viewport::HasSelection() const {
  return !_selectedObjects.empty();
}

void Viewport::HighlightSceneObject(SceneObject* object, filament::math::float3 color) {
  // Per-frame highlight request routed to the stage by the link it belongs to (either
  // representation resolves to the same link). The stage builds/hides the clones in
  // UpdateHighlights().
  if (_stage && object) {
    _stage->RequestHighlight(_stage->GetSceneObjectIndex(object), color);
  }
}

void Viewport::ApplyCameraFocus(
    filament::math::double3 from,
    filament::math::double3 to,
    float orthoHeight,
    std::optional<mochi::Real3> dir) const {
  if (dir.has_value()) {
    // View the focal point from the given editor-space direction. Convert it into renderer space
    // (the same way the constructor sets up the initial camera) and derive the orbit yaw/pitch from
    // it so the result is independent of the renderer's axis convention.
    auto const& converter = _studio->GetEditorToRendererSpaceConverter();
    filament::math::double3 const d =
        normalize(mochi_renderer::ToFilament<double>(converter.DirectionToOutput(*dir)));
    constexpr double kRad2Deg = 180.0 / std::numbers::pi;
    double const yawDeg = std::atan2(d.x, d.z) * kRad2Deg;
    double const pitchDeg = -std::asin(std::clamp(d.y, -1.0, 1.0)) * kRad2Deg;
    _cameraController->LerpOrbitTo(to, length(from - to), yawDeg, pitchDeg, 0.2f, orthoHeight);
  } else {
    _cameraController->SetOrbitPosition(to);
    // Always pass the orthographic height (for both modes) so that when switching
    // from perspective to ortho, the height is already at the target value
    _cameraController->LerpPositionTo(from, 0.2f, orthoHeight);
  }
}

void Viewport::FocusCameraOnScene(std::optional<mochi::Real3> dir) const {
  filament::math::double3 from = {1.0f, 0.5f, 1.0f};
  filament::math::double3 to = {0.0f, 0.0f, 0.0f};
  float orthoHeight = 10.0f;
  _renderScene->GetCameraFocusOnAllSceneObjects(from, to, orthoHeight);
  ApplyCameraFocus(from, to, orthoHeight, dir);
}

void Viewport::FocusCameraOnSelectedSceneObject(std::optional<mochi::Real3> dir) const {
  filament::math::double3 from = {1.0f, 0.5f, 1.0f};
  filament::math::double3 to = {0.0f, 0.0f, 0.0f};
  float orthoHeight = 10.0f;
  bool focused = false;
  if (_selectedObjects.size() == 1) {
    focused =
        _renderScene->GetCameraFocusOnSceneObject(_selectedObjects.back(), from, to, orthoHeight);
  } else if (_selectedObjects.size() > 1) {
    // Frame the combined (union) world bounds of the whole selection via its bounding sphere.
    filament::math::float3 boundsMin{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    filament::math::float3 boundsMax{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};
    for (SceneObject* selected : _selectedObjects) {
      filament::Box const box = selected->GetAABB();
      filament::math::float3 const lo = box.center - box.halfExtent;
      filament::math::float3 const hi = box.center + box.halfExtent;
      for (int i = 0; i < 3; ++i) {
        boundsMin[i] = std::min(boundsMin[i], lo[i]);
        boundsMax[i] = std::max(boundsMax[i], hi[i]);
      }
    }
    if (boundsMin.x <= boundsMax.x) {
      filament::math::float3 const center = (boundsMin + boundsMax) * 0.5f;
      float const radius = length(boundsMax - boundsMin) * 0.5f;
      focused = _renderScene->GetCameraFocusOnSphere(center, radius, from, to);
    }
  }
  if (!focused) {
    _renderScene->GetCameraFocusOnAllSceneObjects(from, to, orthoHeight);
  }
  ApplyCameraFocus(from, to, orthoHeight, dir);
}

void Viewport::ShowViewportContents(bool showCameraOrientationGizmo) {
  ImVec2 contentSize = ImGui::GetContentRegionAvail();
  int logicalWidth = std::max<int>(static_cast<int>(contentSize.x), 64);
  int logicalHeight = std::max<int>(static_cast<int>(contentSize.y), 64);
  // Use framebuffer scale for render target sizing and pick coordinates.
  // On macOS this is 2.0 (logical points -> physical pixels).
  // On Windows this is 1.0 (ImGui coords are already physical pixels).
  float const fbScale = ImGui::GetIO().DisplayFramebufferScale.x;
  int const renderWidth = static_cast<int>(logicalWidth * fbScale);
  int const renderHeight = static_cast<int>(logicalHeight * fbScale);
  int prevW = 0, prevH = 0;
  _renderTarget->GetSize(prevW, prevH);
  if (renderWidth != prevW || renderHeight != prevH) {
    _renderTarget->Resize(renderWidth, renderHeight);
    _renderScene->SetViewport(renderWidth, renderHeight);
  }
  ImVec2 contentOrigin = ImGui::GetCursorScreenPos();

  // InvisibleButton fills the viewport with AllowOverlap so the orientation gizmo
  // can properly steal focus via ImGui's overlap priority system.
  int vpFlags = static_cast<int>(ImGuiButtonFlags_AllowOverlap) |
      static_cast<int>(ImGuiButtonFlags_MouseButtonLeft) |
      static_cast<int>(ImGuiButtonFlags_MouseButtonRight) |
      static_cast<int>(ImGuiButtonFlags_MouseButtonMiddle);
  ImGui::InvisibleButton("ViewportInput", ImGui::GetContentRegionAvail(), vpFlags);
  bool vpActive = ImGui::IsItemActive();
  ImGui::SetCursorScreenPos(contentOrigin);

  // Draw the rendered scene image first, then overlay the gizmos: they share the window draw
  // list, so the gizmos must be submitted after the image to appear on top.
  ImGui::RenderTargetImage(_renderTarget.get(), logicalWidth, logicalHeight);

  // Labels composite over the 3D image; drawn before the gizmos so the gizmos stay on top.
  ShowDebugText(contentOrigin, static_cast<float>(logicalWidth), static_cast<float>(logicalHeight));

  bool orientationGizmoActive = false;
  if (showCameraOrientationGizmo) {
    orientationGizmoActive = ShowCameraOrientationGizmo();
    if (ShowCameraOrientationToolbar()) {
      orientationGizmoActive = true;
    }
  }

  // Process the transform gizmo before picking/camera so we can tell whether it captured the
  // mouse *this* frame. ImGuizmo only refreshes its hover/use state inside Manipulate, so we
  // must read IsOver()/IsUsing() via ShowTransformGizmo's return value (which is false when the
  // gizmo isn't drawn this frame). Reading the global state directly would linger as stale
  // "true" whenever no object is selected and wedge the viewport.
  bool transformGizmoActive = false;
  bool transformGizmoToolbarClicked = false;
  if (showTransformGizmoTarget && showTransformGizmoTarget()) {
    transformGizmoActive = ShowTransformGizmo(
        contentOrigin.x,
        contentOrigin.y,
        static_cast<float>(logicalWidth),
        static_cast<float>(logicalHeight));
    transformGizmoToolbarClicked = ShowTransformGizmoToolbar();
  }
  if (transformGizmoActive || orientationGizmoActive || transformGizmoToolbarClicked) {
    vpActive = false;
  }

  // Handle inputs if ImGui doesn't want key capture (e.g. text input)
  if (!ImGui::GetIO().WantCaptureKeyboard) {
    // F Key / Keypad Period = Focus Viewport on Selected Scene Object
    if (ImGui::IsKeyChordPressed(ImGuiKey_F) || ImGui::IsKeyChordPressed(ImGuiKey_KeypadDecimal)) {
      FocusCameraOnSelectedSceneObject();
    }
    // Home = Frame all scene objects from home vantage point
    if (ImGui::IsKeyChordPressed(ImGuiKey_Home)) {
      FocusCameraOnScene(mochi::Real3{1.0f, 1.0f, 0.5f});
    }
    // Keypad 5 = Toggle Orthographic/Perspective Camera Mode
    if (ImGui::IsKeyChordPressed(ImGuiKey_Keypad5)) {
      auto currentMode = _renderScene->GetCameraMode();
      auto newMode = (currentMode == CameraMode::Perspective) ? CameraMode::Orthographic
                                                              : CameraMode::Perspective;
      _renderScene->SetCameraMode(newMode);
    }
    // Blender-style orthographic view shortcuts.
    struct ViewShortcut {
      ImGuiKeyChord chord;
      mochi::Real3 dir;
    };
    ViewShortcut const kViewShortcuts[] = {
        {ImGuiKey_Keypad1, {0.0f, -1.0f, 0.0f}}, // Front
        {ImGuiKey_Keypad1 | ImGuiMod_Ctrl, {0.0f, 1.0f, 0.0f}}, // Back
        {ImGuiKey_Keypad3, {1.0f, 0.0f, 0.0f}}, // Right
        {ImGuiKey_Keypad3 | ImGuiMod_Ctrl, {-1.0f, 0.0f, 0.0f}}, // Left
        {ImGuiKey_Keypad7, {0.0f, 0.0f, 1.0f}}, // Top
        {ImGuiKey_Keypad7 | ImGuiMod_Ctrl, {0.0f, 0.0f, -1.0f}}, // Bottom
    };
    for (auto const& shortcut : kViewShortcuts) {
      if (ImGui::IsKeyChordPressed(shortcut.chord)) {
        FocusCameraOnSelectedSceneObject(shortcut.dir);
      }
    }
    // Keypad 9 = Flip to the opposite of the current view direction.
    if (ImGui::IsKeyChordPressed(ImGuiKey_Keypad9)) {
      filament::math::double3 const forwardRender =
          normalize(_renderScene->GetCameraRotation() * filament::math::double3{0.0, 0.0, -1.0});
      mochi::Float3 const dirEditor =
          _studio->GetRendererToEditorSpaceConverter().DirectionToOutput(
              mochi::Float3{
                  static_cast<float>(forwardRender.x),
                  static_cast<float>(forwardRender.y),
                  static_cast<float>(forwardRender.z)});
      FocusCameraOnSelectedSceneObject(mochi::Real3{dirEditor[0], dirEditor[1], dirEditor[2]});
    }
    HandleShowCommandShortcuts();
  }

  // Scene-object drag (left-drag on an object). The drag is issued on press but only commits once
  // the pointer moves past a threshold, so a plain click still falls through to selection. Alt+left
  // is excluded so it drives the camera's orbit instead.
  bool const canObjectDrag = enableSceneObjectDrag && onSceneObjectDragStart &&
      !orientationGizmoActive && !transformGizmoActive && !transformGizmoToolbarClicked &&
      !ImGui::GetIO().KeyAlt;
  constexpr float kObjectDragThresholdPx = 3.0f;

  // 1) Press over the image: issue a pick to find the grab point. Use vpActive (not
  // IsWindowHovered, which is false on the press frame since the active item captures the mouse).
  if (canObjectDrag && vpActive && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
    ImVec2 const mousePos = ImGui::GetMousePos();
    ImVec2 const windowPos = ImGui::GetWindowPos();
    float const pickX = (mousePos.x - windowPos.x) * fbScale;
    float const localY = (mousePos.y - windowPos.y) * fbScale;
    float const pickY = static_cast<float>(renderHeight) - 1.0f - localY;
    _grabArmed = false;
    _grabActive = false;
    _grabObject = nullptr;
    // Picks resolve asynchronously, so a result can land after this press was released or
    // superseded. Tag the request and drop results that no longer match, otherwise a stale result
    // would re-arm a released drag on an object the user is no longer holding.
    uint64_t const pickGeneration = ++_grabPickGeneration;
    _renderScene->PickSceneObjectWithPosition(
        pickX, pickY, [this, pickGeneration](mochi_renderer::PickResult const& result) {
          if (pickGeneration != _grabPickGeneration) {
            return;
          }
          if (result.hit && result.object != nullptr) {
            _grabArmed = true;
            _grabObject = result.object;
            _grabPlanePoint = result.worldPosition;
          }
        });
  }

  // 2) Commit once the pointer has moved far enough: ask the consumer to start the drag. Re-check
  // canObjectDrag so a modifier pressed (or a gizmo activated) between press and commit cancels the
  // pending grab instead of stealing the drag from the camera or the gizmo.
  if (canObjectDrag && _grabArmed && !_grabActive && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
    ImVec2 const drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
    if (drag.x * drag.x + drag.y * drag.y >= kObjectDragThresholdPx * kObjectDragThresholdPx) {
      if (onSceneObjectDragStart && onSceneObjectDragStart(_grabObject, _grabPlanePoint)) {
        _grabActive = true;
        _suppressNextReleasePick = true;
        // Record the grab depth (distance along the view to the grabbed point) so the object keeps
        // that distance from the camera as the camera moves.
        CameraFrame const camera = GetCameraFrame(*_renderScene);
        _grabPlaneDistance = dot(_grabPlanePoint - camera.position, camera.forward);
        _grabTargetSmoothed = _grabPlanePoint; // seed the low-pass filter on the grab point
      } else {
        // Consumer declined (e.g. not a draggable object): fall back to normal behavior.
        _grabArmed = false;
        _grabObject = nullptr;
      }
    }
  }

  // 3) While held, cast the cursor onto a screen-parallel plane held a fixed distance in front of
  // the CURRENT camera, and push the target there. Rebuilding the plane from the live camera each
  // frame means moving the camera (RMB free-look / MMB pan / WASD) carries the object with it,
  // keeping it a constant distance away, and the plane never goes edge-on (which would fling it).
  if (_grabActive && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
    // Scroll while grabbing pushes the object nearer/farther along the view (wheel up = farther).
    // Consume the wheel so the camera controller doesn't also act on it.
    float const scroll = ImGui::GetIO().MouseWheel;
    if (scroll != 0.0f) {
      constexpr float kGrabScrollZoomPerNotch = 1.1f;
      constexpr float kMinGrabDistance = 0.05f;
      constexpr float kMaxGrabDistance = 1000.0f;
      _grabPlaneDistance *= std::pow(kGrabScrollZoomPerNotch, scroll);
      _grabPlaneDistance =
          std::max(kMinGrabDistance, std::min(_grabPlaneDistance, kMaxGrabDistance));
      ImGui::GetIO().MouseWheel = 0.0f;
    }
    ImVec2 const mousePos = ImGui::GetMousePos();
    ImVec2 const windowPos = ImGui::GetWindowPos();
    float const pickX = (mousePos.x - windowPos.x) * fbScale;
    float const localY = (mousePos.y - windowPos.y) * fbScale;
    float const pickY = static_cast<float>(renderHeight) - 1.0f - localY;
    CameraFrame const camera = GetCameraFrame(*_renderScene);
    filament::math::float3 const planePoint = camera.position + camera.forward * _grabPlaneDistance;
    mochi_renderer::WorldRay const ray = _renderScene->ScreenPixelToWorldRay(pickX, pickY);
    float const denom = dot(ray.direction, camera.forward);
    constexpr float kParallelEps = 1e-5f;
    if (std::abs(denom) > kParallelEps) {
      float const t = dot(planePoint - ray.origin, camera.forward) / denom;
      if (t > 0.0f) {
        filament::math::float3 const target = ray.origin + ray.direction * t;
        // Low-pass the target so discrete input (scroll notches, WASD / free-look steps) eases in
        // instead of lurching; smooth cursor motion passes through nearly untouched.
        float const dt = ImGui::GetIO().DeltaTime;
        constexpr float kTargetSmoothingTau = 0.05f; // [s]
        float const alpha = (dt > 0.0f) ? (1.0f - std::exp(-dt / kTargetSmoothingTau)) : 1.0f;
        _grabTargetSmoothed = _grabTargetSmoothed + (target - _grabTargetSmoothed) * alpha;
        if (onSceneObjectDragUpdate) {
          onSceneObjectDragUpdate(_grabTargetSmoothed);
        }
      }
    }
  }

  // 4) Release ends the drag. Capture (and always consume) the suppression flag so the selection
  // pick below is skipped for the release that ended a drag, regardless of picking state.
  bool suppressReleasePick = false;
  if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
    if (_grabActive && onSceneObjectDragEnd) {
      onSceneObjectDragEnd();
    }
    suppressReleasePick = _suppressNextReleasePick;
    _grabActive = false;
    _grabArmed = false;
    _grabObject = nullptr;
    _suppressNextReleasePick = false;
    // Invalidate any pick still in flight from this press so its result can't re-arm the drag.
    ++_grabPickGeneration;
  }

  // Pick on left-click release (with drag threshold to avoid picking during drag)
  bool canPick = !orientationGizmoActive && !transformGizmoActive &&
      !transformGizmoToolbarClicked && enableViewportPicking && !suppressReleasePick;
  if (canPick && ImGui::IsWindowHovered() && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
    ImVec2 drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
    float dragDist = std::sqrt(drag.x * drag.x + drag.y * drag.y);
    constexpr float dragThresh = 3.0f;
    if (dragDist < dragThresh) {
      ImVec2 mousePos = ImGui::GetMousePos();
      ImVec2 windowPos = ImGui::GetWindowPos();
      float localX = (mousePos.x - windowPos.x) * fbScale;
      float localY = (mousePos.y - windowPos.y) * fbScale;
      float pickX = localX;
      float pickY = static_cast<float>(renderHeight) - 1.0f - localY;
      _renderScene->PickSceneObject(pickX, pickY, [this](SceneObject* object) {
        bool const multiSelect =
            allowMultiSelect && (ImGui::GetIO().KeyCtrl || ImGui::GetIO().KeyShift);
        if (multiSelect) {
          // Ctrl/Shift-click in the 3D viewport adds/removes the picked object (range-select has
          // no meaning for 3D picks). Clicking empty space is a no-op so the group isn't lost.
          ToggleSceneObjectSelection(object);
        } else if (object == nullptr) {
          // Empty click clears the selection. Notify even when the viewport already had no
          // selection so table-only selections (e.g. articulated actors) still clear.
          bool const wasEmpty = _selectedObjects.empty();
          SetSelectedSceneObjects({});
          if (wasEmpty && onSceneSelectionChanged) {
            onSceneSelectionChanged(_selectedObjects);
          }
        } else if (_selectedObjects.size() == 1 && _selectedObjects.back() == object) {
          // Clicking the sole selected object toggles it off.
          SetSelectedSceneObjects({});
        } else {
          // Plain click replaces the selection with the picked object.
          SetSelectedSceneObjects({object});
        }
      });
    }
  }

  // update camera controller. The camera is intentionally left enabled during an object drag so the
  // user can RMB free-look / MMB pan / WASD to reposition the view (and the object) mid-drag; plain
  // left-drag doesn't drive the camera, so there's nothing to suppress.
  double const moveSpeedBefore = _cameraController->GetMoveSpeed();
  _cameraController->Update(ImGui::GetIO(), vpActive);
  if (_cameraController->GetMoveSpeed() != moveSpeedBefore) {
    // Scroll-wheel adjusted the fly speed while flying; remember it (persisted on shutdown).
    _studio->GetAppSettings().viewport.camera.flySpeed = _cameraController->GetMoveSpeed();
  }

  // Draw the top-left "Show" dropdown last so it overlays the scene image and gizmos.
  ShowShowMenu();
}

void Viewport::ShowStatsOverlay(std::optional<float> mochiSPS, std::optional<float> simTimeSec)
    const {
  ImGuiStyle const& style = ImGui::GetStyle();
  ImVec2 const windowPos = ImGui::GetWindowPos();
  ImVec2 const windowSize = ImGui::GetWindowSize();

  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 12.0f); // ImGui will clamp this

  // Anchor the overlay to the bottom-left corner of the viewport (pivot at its bottom-left).
  constexpr float kPad = 10.0f;
  ImVec2 const overlayPos = {windowPos.x + kPad, windowPos.y + windowSize.y - kPad};
  ImGui::SetNextWindowPos(overlayPos, ImGuiCond_Always, {0.0f, 1.0f});

  constexpr ImGuiWindowFlags kOverlayFlags = ImGuiWindowFlags_NoDecoration |
      ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize |
      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
      ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoInputs;

  // Tighten the vertical gap between rows and match the toolbar buttons' background color.
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {style.ItemSpacing.x, 2.0f});
  ImGui::PushStyleColor(ImGuiCol_WindowBg, style.Colors[ImGuiCol_Button]);
  if (ImGui::Begin("##ViewportStatsOverlay", nullptr, kOverlayFlags)) {
    // Align the value column past the widest label so the numbers line up vertically.
    bool const showPhysics = mochiSPS.has_value();
    bool const showSimTime = simTimeSec.has_value();
    float valueColX = ImGui::CalcTextSize("Render: ").x;
    if (showPhysics) {
      valueColX = std::max(valueColX, ImGui::CalcTextSize("Physics: ").x);
    }
    if (showSimTime) {
      valueColX = std::max(valueColX, ImGui::CalcTextSize("Sim time: ").x);
    }
    valueColX += style.ItemSpacing.x;

    // Sim time is rendered first so it sits at the top of the overlay, above the perf rows.
    if (showSimTime) {
      ImGui::TextUnformatted("Sim time: ");
      ImGui::SameLine(valueColX);
      ImGui::Text("%.3f s", simTimeSec.value());
    }

    ImGui::TextUnformatted("Render: ");
    ImGui::SameLine(valueColX);
    ImGui::Text("%.0f FPS", ImGui::GetIO().Framerate);

    if (showPhysics) {
      // A physics value of 0 (e.g. the simulation hasn't produced a sample yet) is shown as "N/A".
      float const sps = mochiSPS.value();
      ImGui::TextUnformatted("Physics: ");
      ImGui::SameLine(valueColX);
      if (sps > 0.0f) {
        ImGui::Text("%.0f SPS", sps);
      } else {
        ImGui::TextUnformatted("N/A");
      }
    }
  }
  ImGui::End();
  ImGui::PopStyleColor(); // ImGuiCol_WindowBg
  ImGui::PopStyleVar(2); // ImGuiStyleVar_WindowRounding, ImGuiStyleVar_ItemSpacing
}

void Viewport::ShowSceneHierarchyWindow(char const* name, bool* open) {
  ImGui::Begin(name, open);
  auto const& objects = GetRenderScene()->GetSceneObjects();
  for (int i = 0; i < (int)objects.size(); ++i) {
    auto* object = objects[i];
    if (object->_internal) {
      MOCHI_LOG_ERROR("Encountered hidden SceneObject %s", object->GetName().c_str());
      continue;
    }
    bool const isSelected = std::find(_selectedObjects.begin(), _selectedObjects.end(), object) !=
        _selectedObjects.end();
    ImGui::PushID(i);
    if (ImGui::Selectable(object->GetName().c_str(), isSelected)) {
      if (allowMultiSelect && ImGui::GetIO().KeyCtrl) {
        ToggleSceneObjectSelection(object);
      } else if (allowMultiSelect && ImGui::GetIO().KeyShift && !_selectedObjects.empty()) {
        // Shift-click selects the contiguous range (in scene-object order) from the primary object
        // to the clicked one.
        int anchor = -1;
        for (int j = 0; j < (int)objects.size(); ++j) {
          if (objects[j] == _selectedObjects.back()) {
            anchor = j;
            break;
          }
        }
        if (anchor < 0) {
          SetSelectedSceneObjects({object});
        } else {
          int const lo = std::min(anchor, i);
          int const hi = std::max(anchor, i);
          std::vector<SceneObject*> range;
          for (int j = lo; j <= hi; ++j) {
            if (!objects[j]->_internal) {
              range.push_back(objects[j]);
            }
          }
          SetSelectedSceneObjects(std::move(range));
        }
      } else {
        SetSelectedSceneObjects({object});
      }
    }
    ImGui::PopID();
  }
  ImGui::End();
}

void Viewport::ShowSelectedObjectDetailsWindow(char const* name, bool* open) const {
  ImGui::Begin(name, open);
  if (_selectedObjects.size() > 1) {
    ImGui::Text("Multiple Objects Selected (%d)", static_cast<int>(_selectedObjects.size()));
    ImGui::End();
    return;
  }
  SceneObject* selected = _selectedObjects.empty() ? nullptr : _selectedObjects.back();
  if (selected) {
    ImGui::SeparatorText("Transform");
    ImGui::BeginDisabled(true);
    ImGui::DragFloatXYZ("Translation", selected->_translation.v, 0.001f, 0, 0, "%.4f m");
    auto& x = selected->_rotation.x;
    auto& y = selected->_rotation.y;
    auto& z = selected->_rotation.z;
    auto& w = selected->_rotation.w;
    ImGui::DragQuaternion("Rotation", &x, &y, &z, &w, ImGui::QuaternionMode::RPY);
    ImGui::DragFloatXYZ("Scale", selected->_scale.v, 0.01f, 0.001f, 100.0f, "%.4f", 0);
    ImGui::EndDisabled();
    ImGui::SeparatorText("Show");
    ImGui::Checkbox("Axis Aligned Bounding Box", &selected->_showAABB);
    ImGui::SeparatorText("Render Flags");
    bool castShadows = selected->GetCastShadows();
    bool receiveShadows = selected->GetReceiveShadows();
    if (ImGui::Checkbox("Cast Shadows", &castShadows)) {
      selected->SetShadows(castShadows, receiveShadows);
    }
    if (ImGui::Checkbox("Receive Shadows", &receiveShadows)) {
      selected->SetShadows(castShadows, receiveShadows);
    }
    bool visible = selected->GetVisible();
    if (ImGui::Checkbox("Visible", &visible)) {
      selected->SetVisible(visible);
    }
    bool culling = selected->GetCulling();
    if (ImGui::Checkbox("Frustum Culling", &culling)) {
      selected->SetCulling(culling);
    }
    bool sscs = selected->GetScreenSpaceContactShadows();
    if (ImGui::Checkbox("Screen Space Contact Shadows", &sscs)) {
      selected->SetScreenSpaceContactShadows(sscs);
    }
    int sortPriority = selected->GetSortPriority();
    if (ImGui::SliderInt("Sort Priority", &sortPriority, 0, 7)) {
      selected->SetSortPriority(static_cast<uint8_t>(sortPriority));
    }
  }
  ImGui::End();
}

constexpr float kButtonPadding = 10.0f;
constexpr float kGizmoRadius = 50.0f;
constexpr float kGizmoMargin = 20.0f;

bool Viewport::ShowCameraOrientationGizmo() const {
  filament::math::mat4 camMatrix = _renderScene->GetCameraModelMatrix();
  ImVec2 childPos = ImGui::GetWindowPos();
  ImVec2 childSize = ImGui::GetWindowSize();
  ImVec2 gizmoCenter{
      childPos.x + childSize.x - kGizmoRadius - kGizmoMargin,
      childPos.y + kGizmoRadius + kGizmoMargin};

  int clickedAxis = -1;
  bool dragging = false;
  bool gizmoClicked = ImGui::ViewportOrientationGizmo(
      "OrientationGizmo",
      &camMatrix[0][0],
      gizmoCenter,
      kGizmoRadius,
      &clickedAxis,
      &dragging,
      _studio->GetFont("Roboto Bold Small"),
      _studio->GetEditorToRendererSpaceConverter());

  // Return true if gizmo was clicked or is being dragged (to prevent picking)
  if (gizmoClicked && clickedAxis != -1) {
    // clickedAxis is a renderer-space unit axis (0..5 = +X,+Y,+Z,-X,-Y,-Z). Convert it into an
    // editor-space view direction so the focus helpers frame the scene from that side.
    constexpr float kRenderAxes[6][3] = {
        {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}};
    filament::math::float3 const renderDir = {
        kRenderAxes[clickedAxis][0], kRenderAxes[clickedAxis][1], kRenderAxes[clickedAxis][2]};
    mochi::Float3 const editorDir = _studio->GetRendererToEditorSpaceConverter().DirectionToOutput(
        mochi_renderer::ToMochi(renderDir));
    FocusCameraOnSelectedSceneObject(mochi::Real3{editorDir[0], editorDir[1], editorDir[2]});
  } else if (dragging) {
    _cameraController->DoOrbitControl(ImGui::GetIO().MouseDelta, 0.75);
  }
  return gizmoClicked || dragging;
}

bool Viewport::ShowCameraOrientationToolbar() const {
  ImGuiStyle const& style = ImGui::GetStyle();
  ImVec2 const childPos = ImGui::GetWindowPos();
  ImVec2 const childSize = ImGui::GetWindowSize();

  ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 100.0f); // ImGui will clamp this
  ImGui::PushFont(_studio->GetFont("Roboto Regular Small"));

  // Shared button size so these match the transform toolbar exactly.
  float const buttonSize = ToolbarButtonSize();

  // Right-justify the column of buttons with kButtonPadding
  float const buttonsRight = childPos.x + childSize.x - kButtonPadding;
  float const startX = buttonsRight - buttonSize;
  float const startY = childPos.y + 2.0f * kGizmoRadius + 2.0f * kGizmoMargin;
  float const rowStride = buttonSize + style.ItemSpacing.y;

  bool clicked = false;

  // Fly-speed field: a numeric box (labeled with a hand-drawn camera/speed-lines icon) showing the
  // current free-fly speed, where the user can click and type a new value. It mirrors the
  // scroll-wheel speed adjustment, so it updates live while flying. The icon + field group is
  // right-justified to the same edge as the buttons, sitting one row above Home View.
  float const fieldWidth =
      std::max(buttonSize, ImGui::CalcTextSize("0000.00").x + style.FramePadding.x * 2.0f);
  float const fieldX = buttonsRight - fieldWidth;
  float const iconX = fieldX - style.ItemSpacing.x - buttonSize;
  DrawCameraSpeedIcon({iconX, startY}, buttonSize, ImGui::GetColorU32(ImGuiCol_Text));
  double speed = _cameraController->GetMoveSpeed();
  ImGui::SetCursorScreenPos({fieldX, startY + (buttonSize - ImGui::GetFrameHeight()) * 0.5f});
  ImGui::SetNextItemWidth(fieldWidth);
  bool const speedCommitted = ImGui::InputDouble("##CameraFlySpeed", &speed, 0.0, 0.0, "%.2f");
  if (speedCommitted || ImGui::IsItemDeactivatedAfterEdit()) {
    _cameraController->SetMoveSpeed(speed);
    // Remember the typed speed for future viewports; persist immediately for durability.
    _studio->GetAppSettings().viewport.camera.flySpeed = _cameraController->GetMoveSpeed();
    _studio->SaveSettings();
    clicked = true;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Camera fly speed (scroll while flying to adjust)");
  }

  // Frame the whole scene from the default vantage.
  ImGui::SetCursorScreenPos({startX, startY + rowStride});
  if (ImGui::Button(ICON_FA_HOME "##FrameScene", {buttonSize, buttonSize})) {
    FocusCameraOnScene(mochi::Real3{1.0f, 1.0f, 0.5f});
    clicked = true;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Home View (Home)");
  }

  // Toggle between perspective and orthographic projection (same as the Numpad 5 shortcut).
  bool const isPerspective = (_renderScene->GetCameraMode() == CameraMode::Perspective);
  ImGui::SetCursorScreenPos({startX, startY + 2.0f * rowStride});
  if (ImGui::Button(
          isPerspective ? ICON_FA_VIDEO "##Projection" : ICON_FA_CUBES "##Projection",
          {buttonSize, buttonSize})) {
    _renderScene->SetCameraMode(isPerspective ? CameraMode::Orthographic : CameraMode::Perspective);
    clicked = true;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(isPerspective ? "Perspective (Numpad 5)" : "Orthographic (Numpad 5)");
  }

  ImGui::PopFont();
  ImGui::PopStyleVar(); // ImGuiStyleVar_FrameRounding
  return clicked;
}

bool Viewport::ShowTransformGizmo(float x, float y, float width, float height) const {
  std::optional<std::pair<mochi::TransformRT, mochi::Real3>> gizmoTarget;
  if (getTransformGizmoTarget) {
    gizmoTarget = getTransformGizmoTarget();
  }
  if (!gizmoTarget.has_value()) {
    return false;
  }

  // Build the view matrix in editor (Mochi) space: viewEditor = viewRender * editorToRender, where
  // editorToRender is the 4x4 change-of-basis (3x3 basis * uniform scale, zero translation). The
  // projection is unchanged. Both Mochi and render space are right-handed, so no handedness flip
  // is needed.
  auto const& editorToRender = _studio->GetEditorToRendererSpaceConverter();
  filament::math::mat4f const editorToRender4 =
      mochi_renderer::ToFilament<float>(editorToRender.GetTransformMatrix());
  filament::math::mat4 const viewMatD = _renderScene->GetCameraViewMatrix();
  filament::math::mat4 const projMatD = _renderScene->GetCameraProjectionMatrix();
  filament::math::mat4f const viewEditor = filament::math::mat4f(viewMatD) * editorToRender4;
  float viewMat[16], projMat[16];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      viewMat[i * 4 + j] = viewEditor[i][j];
      projMat[i * 4 + j] = static_cast<float>(projMatD[i][j]);
    }
  }

  // Build the model matrix in editor space (T * R * S) from the target transform and scale.
  mochi::TransformRT const& targetTransform = gizmoTarget->first;
  mochi::Real3 const& targetScale = gizmoTarget->second;
  mochi::Real3 const t = targetTransform.GetTranslation();
  mochi::Quaternion const r = targetTransform.GetRotation();
  filament::math::float3 const translation{
      static_cast<float>(t[0]), static_cast<float>(t[1]), static_cast<float>(t[2])};
  filament::math::quatf const rotation{
      static_cast<float>(r.data[3]),
      static_cast<float>(r.data[0]),
      static_cast<float>(r.data[1]),
      static_cast<float>(r.data[2])};
  filament::math::float3 const scale{
      static_cast<float>(targetScale[0]),
      static_cast<float>(targetScale[1]),
      static_cast<float>(targetScale[2])};
  filament::math::mat4f const objectTransform = filament::math::mat4f::translation(translation) *
      filament::math::mat4f(rotation) * filament::math::mat4f::scaling(scale);
  float objectMat[16];
  std::copy_n(&objectTransform[0][0], 16, objectMat);

  // Configure ImGuizmo for this viewport
  ImGuizmo::SetDrawlist();
  ImGuizmo::SetRect(x, y, width, height);
  bool const isOrthographic = (_renderScene->GetCameraMode() == CameraMode::Orthographic);
  ImGuizmo::SetOrthographic(isOrthographic);

  TransformGizmoSettings const& snapSettings = _studio->GetAppSettings().viewport.transformGizmo;
  bool const snapActive = snapSettings.enabled != ImGui::GetIO().KeyCtrl;
  float snap[3] = {0.0f, 0.0f, 0.0f};
  if (snapActive) {
    float snapValue = 0.0f;
    if (_gizmoOperation == ImGuizmo::TRANSLATE) {
      snapValue = static_cast<float>(snapSettings.translate);
    } else if (_gizmoOperation == ImGuizmo::ROTATE) {
      snapValue = static_cast<float>(snapSettings.rotateDeg);
    } else if (_gizmoOperation == ImGuizmo::SCALE) {
      snapValue = static_cast<float>(snapSettings.scale);
    }
    snap[0] = snap[1] = snap[2] = snapValue;
  }

  // Manipulate
  bool changed = false;
  mochi::TransformRT newTransform;
  mochi::Real3 outScale{};
  if (ImGuizmo::Manipulate(
          viewMat,
          projMat,
          _gizmoOperation,
          _gizmoMode,
          objectMat,
          /*deltaMatrix=*/nullptr,
          snapActive ? snap : nullptr)) {
    // The result is already in editor space; decompose into translation, rotation, and scale.
    filament::math::mat4f result;
    std::copy_n(objectMat, 16, &result[0][0]);

    filament::math::float3 newTranslation{result[3].x, result[3].y, result[3].z};
    filament::math::float3 col0{result[0].x, result[0].y, result[0].z};
    filament::math::float3 col1{result[1].x, result[1].y, result[1].z};
    filament::math::float3 col2{result[2].x, result[2].y, result[2].z};
    filament::math::float3 newScale{length(col0), length(col1), length(col2)};

    // Extract rotation (normalize columns → mat4f → quaternion)
    if (newScale.x > 1e-6f && newScale.y > 1e-6f && newScale.z > 1e-6f) {
      filament::math::float3 rc0 = col0 / newScale.x;
      filament::math::float3 rc1 = col1 / newScale.y;
      filament::math::float3 rc2 = col2 / newScale.z;
      filament::math::mat4f rotMat4(
          filament::math::float4{rc0, 0},
          filament::math::float4{rc1, 0},
          filament::math::float4{rc2, 0},
          filament::math::float4{0, 0, 0, 1});
      filament::math::quatf rotQ = normalize(rotMat4.toQuaternion());

      newTransform = mochi::TransformRT(
          mochi::Quaternion(
              static_cast<mochi::real>(rotQ.x),
              static_cast<mochi::real>(rotQ.y),
              static_cast<mochi::real>(rotQ.z),
              static_cast<mochi::real>(rotQ.w)),
          mochi::Real3{
              static_cast<mochi::real>(newTranslation.x),
              static_cast<mochi::real>(newTranslation.y),
              static_cast<mochi::real>(newTranslation.z)});
      outScale = mochi::Real3{
          static_cast<mochi::real>(newScale.x),
          static_cast<mochi::real>(newScale.y),
          static_cast<mochi::real>(newScale.z)};
      changed = true;
    }
  }

  // On the rising edge of IsUsing(), decide when this drag's start fires. The client supplies a
  // pixel threshold via transformGizmoStartThresholdPx (the viewport itself has no notion of which
  // modifier, if any, is involved). A zero threshold starts immediately; a positive threshold
  // defers the start — and withholds this drag's delta so the target isn't touched — until the
  // pointer moves that far, so an accidental click doesn't trigger it.
  bool const usingNow = ImGuizmo::IsUsing();
  if (usingNow && !_gizmoWasUsing) {
    // Seed the per-frame scale tracker with the gizmo target's scale: ImGuizmo reports scale
    // cumulatively from the drag start, so the first frame's multiplier is measured against the
    // target's starting scale (the first selected item's scale, which is also the gizmo pivot for a
    // multi-selection).
    _gizmoPrevScale = targetScale;
    _gizmoStartThresholdPx =
        transformGizmoStartThresholdPx ? transformGizmoStartThresholdPx() : 0.0f;
    if (_gizmoStartThresholdPx > 0.0f) {
      _gizmoStartPending = true;
    } else if (onTransformGizmoStarted) {
      onTransformGizmoStarted();
    }
  } else if (!usingNow) {
    // Drag ended (or the gizmo isn't in use); drop a pending start that never cleared its
    // threshold, so a bare click leaves the target untouched.
    _gizmoStartPending = false;
  }
  _gizmoWasUsing = usingNow;

  // Once a pending start clears its threshold, fire it before this frame's delta is applied, so a
  // duplicate-on-drag handler can retarget the gizmo (e.g. onto fresh copies) and have the
  // accumulated movement land on the new target.
  if (_gizmoStartPending) {
    ImVec2 const drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
    if (drag.x * drag.x + drag.y * drag.y >= _gizmoStartThresholdPx * _gizmoStartThresholdPx) {
      _gizmoStartPending = false;
      if (onTransformGizmoStarted) {
        onTransformGizmoStarted();
      }
    }
  }

  // Apply an incremental delta about the gizmo target to every selected item. This is uniform for
  // one or many selections: ImGuizmo's absolute result gives the new target placement, from which
  // we derive the per-frame transform delta and per-axis scale multiplier. With a single selection
  // the target is the item itself, so `delta * itemTransform` reconstructs its new absolute
  // transform. While a threshold-gated start is still pending, deltas are held back so the target
  // stays put; once the threshold is crossed the accumulated movement lands on the (possibly
  // retargeted) selection.
  if (changed && !_gizmoStartPending && onTransformGizmoDelta) {
    mochi::TransformRT const deltaTransform = newTransform * mochi::Invert(targetTransform);
    mochi::Real3 const deltaScale{
        outScale[0] / _gizmoPrevScale[0],
        outScale[1] / _gizmoPrevScale[1],
        outScale[2] / _gizmoPrevScale[2]};
    _gizmoPrevScale = outScale;
    onTransformGizmoDelta(deltaTransform, deltaScale);
  }

  // Report whether the gizmo captured the mouse this frame so the caller can suppress picking
  // and camera control. This is fresh state: Manipulate was just called above.
  return ImGuizmo::IsOver() || ImGuizmo::IsUsing();
}

bool Viewport::ShowTransformGizmoToolbar() {
  ImVec2 childPos = ImGui::GetWindowPos();
  ImVec2 childSize = ImGui::GetWindowSize();
  ImGuiStyle const& style = ImGui::GetStyle();
  ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 100.0f); // ImGui will clamp this
  ImGui::PushFont(_studio->GetFont("Roboto Regular Small"));

  // Match the simulation toolbar height and right-justify against the camera
  // orientation gizmo at the top-right.

  float const spacing = style.ItemSpacing.x;
  // Shared button size so all toolbar buttons are identical. All buttons share width and height.
  float const buttonSize = ToolbarButtonSize();
  // The snap dropdown caret is a narrow button joined directly to the snap toggle (split-button).
  float const caretSize = buttonSize * 0.6f;
  // translate, rotate, scale, space, snap-toggle (5 full buttons) plus the connected caret. The
  // caret joins the snap toggle with no spacing, so there are only 4 inter-button gaps.
  float const totalWidth = buttonSize * 5.0f + caretSize + spacing * 4.0f;

  // Left edge of the orientation gizmo circle.
  float const gizmoLeft = childSize.x - 2.0f * kGizmoRadius - kGizmoMargin;
  float const startX = childPos.x + gizmoLeft - kGizmoMargin - totalWidth;
  float const startY = childPos.y + kButtonPadding;
  ImGui::SetCursorScreenPos({startX, startY});

  bool toolbarClicked = false;
  auto activeButton = [buttonSize](char const* label, bool isActive) {
    if (isActive) {
      ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    }
    bool clicked = ImGui::Button(label, {buttonSize, buttonSize});
    if (isActive) {
      ImGui::PopStyleColor();
    }
    return clicked;
  };

  if (activeButton(ICON_FA_ARROWS_ALT "##GizmoTranslate", _gizmoOperation == ImGuizmo::TRANSLATE)) {
    _gizmoOperation = ImGuizmo::TRANSLATE;
    toolbarClicked = true;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Translate (W)");
  }

  ImGui::SameLine();
  if (activeButton(ICON_FA_SYNC_ALT "##GizmoRotate", _gizmoOperation == ImGuizmo::ROTATE)) {
    _gizmoOperation = ImGuizmo::ROTATE;
    toolbarClicked = true;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Rotate (E)");
  }

  ImGui::SameLine();
  if (activeButton(ICON_FA_EXPAND_ARROWS_ALT "##GizmoScale", _gizmoOperation == ImGuizmo::SCALE)) {
    _gizmoOperation = ImGuizmo::SCALE;
    toolbarClicked = true;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Scale (R)");
  }

  ImGui::SameLine();

  bool const isSpaceLocal = (_gizmoMode == ImGuizmo::LOCAL);
  if (ImGui::Button(
          isSpaceLocal ? ICON_FA_CUBE "##GizmoSpace" : ICON_FA_GLOBE "##GizmoSpace",
          {buttonSize, buttonSize})) {
    _gizmoMode = isSpaceLocal ? ImGuizmo::WORLD : ImGuizmo::LOCAL;
    toolbarClicked = true;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(isSpaceLocal ? "Local Space" : "World Space");
  }

  ImGui::SameLine();

  TransformGizmoSettings& snapSettings = _studio->GetAppSettings().viewport.transformGizmo;
  if (ImGui::SplitButtonSegment(
          ICON_FA_TH "##GizmoSnap",
          {buttonSize, buttonSize},
          ImDrawFlags_RoundCornersLeft,
          snapSettings.enabled)) {
    snapSettings.enabled = !snapSettings.enabled;
    _studio->SaveSettings();
    toolbarClicked = true;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Snap (hold Ctrl to toggle)");
  }

  ImGui::SameLine(0.0f, 0.0f);
  if (ImGui::SplitButtonSegment(
          ICON_FA_CARET_DOWN "##GizmoSnapSettings",
          {caretSize, buttonSize},
          ImDrawFlags_RoundCornersRight,
          false)) {
    ImGui::OpenPopup("##SnapSettings");
    toolbarClicked = true;
  }
  ImVec2 const snapPopupAnchor = ImGui::GetItemRectMax();

  // Keyboard shortcuts (W/E/R) — only when viewport is focused, suppress during text input
  if (ImGui::IsWindowFocused() && !ImGui::GetIO().WantCaptureKeyboard &&
      !ImGui::IsMouseDown(ImGuiMouseButton_Right) && !ImGuizmo::IsUsing()) {
    if (ImGui::IsKeyChordPressed(ImGuiKey_W)) {
      _gizmoOperation = ImGuizmo::TRANSLATE;
    }
    if (ImGui::IsKeyChordPressed(ImGuiKey_E)) {
      _gizmoOperation = ImGuizmo::ROTATE;
    }
    if (ImGui::IsKeyChordPressed(ImGuiKey_R)) {
      _gizmoOperation = ImGuizmo::SCALE;
    }
  }

  ImGui::PopFont();
  ImGui::PopStyleVar(); // ImGuiStyleVar_FrameRounding

  ImGui::PushFont(_studio->GetFont("Roboto Regular Small"));
  ImGui::SetNextWindowPos(snapPopupAnchor, ImGuiCond_Always, {1.0f, 0.0f});
  if (ImGui::BeginPopup("##SnapSettings")) {
    ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
    constexpr float kPresetWidth = 44.0f;
    constexpr float kCustomWidth = 96.0f;
    if (ImGui::BeginTable("##SnapGrid", 9, ImGuiTableFlags_SizingFixedFit)) {
      auto snapRow = [this](
                         char const* label, double* value, std::initializer_list<double> presets) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        for (double const preset : presets) {
          ImGui::TableNextColumn();
          bool const isActive = (*value == preset);
          if (isActive) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
          }
          char buttonLabel[32];
          std::snprintf(buttonLabel, sizeof(buttonLabel), "%g##%s", preset, label);
          if (ImGui::Button(buttonLabel, {kPresetWidth, 0.0f})) {
            *value = preset;
            _studio->SaveSettings();
          }
          if (isActive) {
            ImGui::PopStyleColor();
          }
        }
        ImGui::TableSetColumnIndex(8);
        char inputLabel[32];
        std::snprintf(inputLabel, sizeof(inputLabel), "##custom%s", label);
        ImGui::SetNextItemWidth(kCustomWidth);
        if (ImGui::InputDouble(inputLabel, value, 0.0, 0.0, "%g")) {
          _studio->SaveSettings();
        }
      };
      snapRow("Translate", &snapSettings.translate, {0.001, 0.01, 0.1, 1.0, 10.0});
      snapRow("Rotate", &snapSettings.rotateDeg, {1.0, 5.0, 10.0, 15.0, 30.0, 45.0, 90.0});
      snapRow("Scale", &snapSettings.scale, {0.1, 0.25, 0.5, 1.0});
      ImGui::EndTable();
    }
    ImGui::PopItemFlag();
    ImGui::EndPopup();
  }
  ImGui::PopFont();

  return toolbarClicked;
}

void Viewport::DrawDebug() {
  auto* dd = GetRenderScene()->GetDebugDraw();
  if (_showGroundGrid) {
    DrawGroundGrid(dd, _studio->GetAppSettings().viewport.groundGrid, _groundPlaneHeight);
  }
  auto const& objects = GetRenderScene()->GetSceneObjects();
  for (auto const& object : objects) {
    if (object->_showAABB) {
      dd->DrawBox(object->GetAABB(), filament::math::float4{1.0f, 0.0f, 0.0f, 1.0f});
    }
  }
}

void Viewport::ShowDebugText(ImVec2 contentOrigin, float logicalWidth, float logicalHeight) {
  _debugText.Show(
      _renderScene->GetCameraViewMatrix(),
      _renderScene->GetCameraProjectionMatrix(),
      contentOrigin,
      logicalWidth,
      logicalHeight);
  _debugText.Clear();
}

void Viewport::RegisterShowCommand(ShowCommand command) {
  _showCommands.push_back(std::move(command));
}

void Viewport::HandleShowCommandShortcuts() {
  for (auto const& cmd : _showCommands) {
    if (cmd.shortcut == ImGuiKey_None) {
      continue;
    }
    if (cmd.isEnabled && !cmd.isEnabled()) {
      continue;
    }
    if (ImGui::IsKeyChordPressed(cmd.shortcut)) {
      cmd.onToggle();
    }
  }
}

void Viewport::ShowShowMenu() {
  if (_showCommands.empty()) {
    return;
  }

  ImVec2 const childPos = ImGui::GetWindowPos();
  ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 100.0f); // ImGui will clamp this

  ImGui::SetCursorScreenPos({childPos.x + kButtonPadding, childPos.y + kButtonPadding});
  if (ImGui::Button("Show")) {
    ImGui::OpenPopup("##ShowMenu");
  }
  ImGui::SetNextWindowPos({ImGui::GetItemRectMin().x, ImGui::GetItemRectMax().y});
  if (ImGui::BeginPopup("##ShowMenu")) {
    // Keep the menu open after a click so multiple toggles can be flipped; it closes only when the
    // user clicks away.
    ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
    for (auto const& cmd : _showCommands) {
      bool const enabled = !cmd.isEnabled || cmd.isEnabled();
      bool state = cmd.getState();
      char const* shortcutText =
          (cmd.shortcut != ImGuiKey_None) ? ImGui::GetKeyChordName(cmd.shortcut) : nullptr;
      if (ImGui::MenuItem(cmd.name.c_str(), shortcutText, &state, enabled)) {
        cmd.onToggle();
      }
    }
    ImGui::PopItemFlag();
    ImGui::EndPopup();
  }

  ImGui::PopStyleVar(); // ImGuiStyleVar_FrameRounding
}

} // namespace superdex::studio
