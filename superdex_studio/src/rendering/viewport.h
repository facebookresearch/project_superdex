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

#include "rendering/camera_controller.h"
#include "rendering/debug_text.h"
#include "rendering/render_target.h"

#include <mochi_renderer/resource_manager.h>
#include <mochi_renderer/scene.h>

#include <ImGuizmo.h>
#include <imgui.h>

#include <math/vec3.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace filament {
class VertexBuffer;
class IndexBuffer;
} // namespace filament

namespace superdex::studio {

class SuperDexStudio;
class Renderer;
struct SceneStage;

// Orange used for viewport selection/hover highlights; shared so overlays (e.g. the force-drag
// debug gizmo) can match it exactly.
constexpr filament::math::float3 kViewportSelectionColor{1.0f, 0.5f, 0.0f};

class Viewport {
 public:
  static std::unique_ptr<Viewport> Create(
      SuperDexStudio* studio,
      mochi_renderer::SceneViewSettings const& viewSettings);

  // A toggleable "Show" command surfaced both as a keyboard shortcut and a checkable item in the
  // viewport's top-left "Show" dropdown. Editors register these in their Initialize().
  struct ShowCommand {
    std::string name; // Menu label.
    std::function<void()> onToggle; // Invoked on click or shortcut.
    std::function<bool()> getState; // Current on/off state (drives the checkmark).
    std::function<bool()> isEnabled; // Optional; empty == always enabled.
    ImGuiKeyChord shortcut = ImGuiKey_None; // Optional binding (e.g. ImGuiKey_G).
  };

 public:
  // Invoked whenever the set of selected scene objects changes, with the full (ordered) selection.
  // The back() element is the primary/active object. Empty when nothing is selected.
  std::function<void(std::vector<mochi_renderer::SceneObject*> const&)> onSceneSelectionChanged;
  std::function<bool()> showTransformGizmoTarget;
  // Target transform and scale in Mochi/editor space, or nullopt when there is no valid target.
  // For a single selection this is the object's own transform; for a multi-selection it is the
  // pivot the group is manipulated about.
  std::function<std::optional<std::pair<mochi::TransformRT, mochi::Real3>>()>
      getTransformGizmoTarget;
  // Invoked each manipulation frame with the incremental transform delta (rotation + translation
  // about the gizmo target) and the per-axis scale multiplier to apply to every selected item.
  // Applying `delta * itemTransform` (and `itemScale * scaleMul`) to each selected item works
  // uniformly for one or many: with a single selection the target is the item itself, so the delta
  // reconstructs its new absolute transform.
  std::function<void(mochi::TransformRT const&, mochi::Real3 const&)> onTransformGizmoDelta;
  // Invoked once per drag when the manipulation is considered started, before that drag's first
  // onTransformGizmoDelta. "Started" is deferred until the pointer has moved past the threshold
  // returned by transformGizmoStartThresholdPx (0 ⇒ fires on the rising edge). Until it fires,
  // deltas are withheld so the target isn't touched. Lets editors implement e.g. duplicate-on-drag
  // (duplicate here, retarget the gizmo onto the copies, and this drag continues on them).
  std::function<void()> onTransformGizmoStarted;
  // Screen-space distance (pixels) the pointer must travel before onTransformGizmoStarted fires for
  // the current drag; consulted once on the rising edge. Returning 0 (or leaving this unset) starts
  // immediately. Editors use this to gate a start behind a deliberate drag — e.g. a small threshold
  // when Alt is held so a stray alt+click doesn't trigger duplicate-on-drag.
  std::function<float()> transformGizmoStartThresholdPx;
  // When true, viewport picking supports additive/toggle multi-selection via Ctrl/Shift-click.
  // False ⇒ picking always replaces the selection (single). Editors opt in by setting this.
  bool allowMultiSelect = false;
  // When false, left-click picking in the viewport is disabled (no scene-object selection). Editors
  // that don't need selection (e.g. the model editor) can turn this off.
  bool enableViewportPicking = true;

  // --- Scene-object drag (left-drag on a scene object) -----------------------------------------
  // Generic click-and-drag for scene objects: the Viewport handles input (pick, drag-plane math,
  // threshold gating so a click still selects) and routes the lifecycle through these hooks.
  // Consumers decide what a drag means (e.g. a physics force-drag). All points are world-space.
  //
  // Fired once when the drag crosses the movement threshold, with the picked object and hit point;
  // returns true if the consumer started a drag (false falls back to normal selection).
  std::function<bool(mochi_renderer::SceneObject*, filament::math::float3)> onSceneObjectDragStart;
  // Fired each frame while held, with the new target on the screen-parallel drag plane.
  std::function<void(filament::math::float3)> onSceneObjectDragUpdate;
  // Fired once when the drag is released.
  std::function<void()> onSceneObjectDragEnd;
  // Gate mirroring @ref enableViewportPicking; consumers that support scene-object drag leave it
  // on.
  bool enableSceneObjectDrag = true;

 public:
  Viewport(Viewport const&) = delete;
  Viewport& operator=(Viewport const&) = delete;
  Viewport(Viewport&&) = delete;
  Viewport& operator=(Viewport&&) = delete;
  ~Viewport();
  // Render Scene
  mochi_renderer::Scene* GetRenderScene() const;
  void RenderScene(Renderer const* renderer);
  // World-anchored debug labels for this viewport. Submit alongside the debug geometry in
  // OnRender; the viewport draws and clears them in its ImGui pass (see @ref DebugText).
  DebugText* GetDebugText() {
    return &_debugText;
  }
  // The Filament-backed color/depth target this viewport renders into. Its size tracks the ImGui
  // content region (times DisplayFramebufferScale) each frame, so it holds the viewport exactly as
  // shown at native pixel resolution -- e.g. to read the pixels back for a screenshot. Never null.
  RenderTarget* GetRenderTarget() const {
    return _renderTarget.get();
  }
  // Positions the drop-shadow ground plane at the lowest point of the scene's meshes (from their
  // AABBs, including hidden ones, so toggling visibility doesn't move it) and returns that height.
  // Call after geometry changes; do NOT call while a physics sim is running -- the plane should
  // stay fixed as bodies move.
  float UpdateGroundPlane();
  // Binds the SceneStage backing this viewport (non-owning). The stage owns the per-link highlight
  // clones and their show/hide; the viewport routes selection/hover highlight requests to it and
  // drives the see-through overlay pass from it. Null for viewports without a stage (no highlight).
  void SetSceneStage(SceneStage* stage) {
    _stage = stage;
  }
  // Scene Object Selection / Highlighting
  // Replaces the whole selection with `objects` (nulls and duplicates are dropped; order preserved,
  // back() = primary/active). Pass {} to clear the selection.
  void SetSelectedSceneObjects(
      std::vector<mochi_renderer::SceneObject*> objects,
      bool invokeCallback = true);
  // Adds `object` to the selection if absent, or removes it if already selected (Ctrl-click). No-op
  // for null.
  void ToggleSceneObjectSelection(mochi_renderer::SceneObject* object, bool invokeCallback = true);
  void SelectNeighborAndDestroy(mochi_renderer::SceneObject* object);
  // The full ordered selection (back() = primary/active).
  std::vector<mochi_renderer::SceneObject*> const& GetSelectedSceneObjects() const;
  bool HasSelection() const;
  // Requests that the link owning `object` be highlighted this frame with `color`. Call while
  // hovering; selection is applied automatically. Re-declared every frame (see RenderScene). Routed
  // to the bound SceneStage, which highlights the link's currently-visible representation(s).
  void HighlightSceneObject(mochi_renderer::SceneObject* object, filament::math::float3 color);
  // Global opacity of the see-through highlight overlay (0..1). Bound by the View Settings slider.
  float& HighlightOverlayAlpha() {
    return _highlightOverlayAlpha;
  }
  void FocusCameraOnScene(std::optional<mochi::Real3> dir = std::nullopt) const;
  void FocusCameraOnSelectedSceneObject(std::optional<mochi::Real3> dir = std::nullopt) const;
  // ImGui
  void ShowViewportContents(bool showCameraOrientationGizmo);
  void ShowStatsOverlay(
      std::optional<float> mochiSPS = std::nullopt,
      std::optional<float> simTimeSec = std::nullopt) const;
  void ShowSceneHierarchyWindow(char const* name, bool* open);
  void ShowSelectedObjectDetailsWindow(char const* name, bool* open) const;
  // Register a toggleable "Show" command. The viewport owns its keyboard shortcut and lists it in
  // the top-left "Show" dropdown.
  void RegisterShowCommand(ShowCommand command);

 private:
  Viewport(SuperDexStudio* studio, mochi_renderer::SceneViewSettings const& viewSettings);
  void ApplyCameraFocus(
      filament::math::double3 from,
      filament::math::double3 to,
      float orthoHeight,
      std::optional<mochi::Real3> dir) const;
  bool ShowCameraOrientationGizmo() const;
  bool ShowCameraOrientationToolbar() const;
  bool ShowTransformGizmo(float x, float y, float width, float height) const;
  bool ShowTransformGizmoToolbar();
  // Projects and draws the queued debug labels over the viewport image, then clears them.
  void ShowDebugText(ImVec2 contentOrigin, float logicalWidth, float logicalHeight);
  void ShowShowMenu();
  void HandleShowCommandShortcuts();
  void DrawDebug();
  // Lazily creates, and resizes to (width, height), the highlight overlay target + view and the
  // fullscreen composite view. Called each frame a highlight is present.
  void EnsureHighlightOverlay(int width, int height);
  // Destroys the engine-owned highlight overlay/composite Filament objects (called from dtor).
  void DestroyHighlightOverlay();

 private:
  SuperDexStudio* _studio = nullptr;
  // Rendering (per-tab, owned)
  std::unique_ptr<mochi_renderer::Scene> _renderScene;
  std::unique_ptr<RenderTarget> _renderTarget;
  std::unique_ptr<CameraController> _cameraController;
  // World-anchored debug labels submitted this frame (see GetDebugText).
  DebugText _debugText;
  // Selected objects (non-owning), ordered; back() is the primary/active object (drives
  // gizmo-anchor fallback and camera focus). Selection identity is owned by the editor; the
  // viewport tracks the selected SceneObjects to drive the gizmo, click-to-deselect, and to declare
  // their links' selection highlights to the stage each frame (see RenderScene).
  std::vector<mochi_renderer::SceneObject*> _selectedObjects;
  // SceneStage backing this viewport (non-owning; set via SetSceneStage). Owns the per-link
  // highlight clones and their base-mesh show/hide; the viewport only routes highlight requests to
  // it and drives the overlay pass from HasActiveHighlights(). Null when there is no stage.
  SceneStage* _stage = nullptr;
  // Highlight overlay compositing (see renderer.h HighlightPass). Built lazily when a highlight
  // first appears and resized to match _renderTarget. The overlay view re-renders the highlighted
  // clones (tagged with mochi_renderer::kHighlightOverlayLayer) into _overlayTarget with its own
  // cleared color+depth (isolated: nearest surface only, no scene occlusion); the composite view is
  // a fullscreen TRANSLUCENT pass that blends that overlay over the main image. All raw Filament
  // objects here are engine-owned and released in DestroyHighlightOverlay.
  std::unique_ptr<RenderTarget> _overlayTarget;
  filament::View* _overlayView = nullptr;
  filament::Scene* _compositeScene = nullptr;
  utils::Entity _compositeCameraEntity;
  filament::Camera* _compositeCamera = nullptr;
  filament::View* _compositeView = nullptr;
  filament::VertexBuffer* _compositeVB = nullptr;
  filament::IndexBuffer* _compositeIB = nullptr;
  utils::Entity _compositeQuadEntity;
  std::shared_ptr<mochi_renderer::MaterialInstance> _compositeMaterial;
  // Global opacity of the composited see-through highlight overlay (0 = invisible, 1 = opaque).
  // Exposed for the View Settings slider via HighlightOverlayAlpha().
  float _highlightOverlayAlpha = 0.35f;
  // Transform Gizmo
  ImGuizmo::OPERATION _gizmoOperation = ImGuizmo::TRANSLATE;
  ImGuizmo::MODE _gizmoMode = ImGuizmo::WORLD;
  // Tracks ImGuizmo::IsUsing() across frames to detect the start of a manipulation (rising edge).
  mutable bool _gizmoWasUsing = false;
  // The current drag's start hasn't fired yet: onTransformGizmoStarted is deferred (and this drag's
  // deltas withheld) until the pointer moves past _gizmoStartThresholdPx. Only set when that
  // threshold is > 0; a zero threshold starts on the rising edge with no pending window.
  mutable bool _gizmoStartPending = false;
  // Pixel threshold captured on the rising edge (from transformGizmoStartThresholdPx) that the
  // pointer must clear before the deferred start fires for the current drag.
  mutable float _gizmoStartThresholdPx = 0.0f;
  // Previous frame's absolute gizmo scale during a manipulation, used to derive the per-frame
  // incremental scale multiplier (ImGuizmo reports scale cumulatively from the drag start). Reset
  // to the gizmo target's scale on the rising edge of a manipulation.
  mutable mochi::Real3 _gizmoPrevScale{1.0f, 1.0f, 1.0f};
  // Registered "Show" toggle commands (keyboard bindings + top-left dropdown).
  std::vector<ShowCommand> _showCommands;
  // Ground grid: per-viewport visibility (Show menu / G), seeded from the app settings at creation
  // and drawn at the scene's lowest point. Its appearance is app-wide -- see
  // AppSettings::viewport::groundGrid.
  bool _showGroundGrid = true;
  float _groundPlaneHeight = 0.0f;

  // --- Scene-object drag state (left-drag) ---
  // Bumped on every press that issues a grab pick. Picks resolve asynchronously, so the callback
  // compares against this to discard results from a press that has since been released or
  // superseded.
  uint64_t _grabPickGeneration = 0;
  // The pick returned a draggable object; the drag is armed and will commit once the pointer moves
  // past the threshold (so a plain Ctrl-click still selects instead of dragging).
  bool _grabArmed = false;
  // A drag is committed and held (onSceneObjectDragStart returned true).
  bool _grabActive = false;
  // Set when a drag committed on this press so the release doesn't also run a selection pick.
  bool _suppressNextReleasePick = false;
  // The picked object awaiting commit (only valid while _grabArmed and not yet active).
  mochi_renderer::SceneObject* _grabObject = nullptr;
  // World-space hit point on the grabbed object, captured at pick time; passed to the drag consumer
  // and used to seed _grabPlaneDistance.
  filament::math::float3 _grabPlanePoint{};
  // Perpendicular distance from the camera to the grab point at commit. Each frame the drag plane
  // is held at this depth in front of the live camera, so moving the camera carries the object
  // along.
  float _grabPlaneDistance = 0.0f;
  // Low-pass-filtered target sent to the drag consumer, so discrete input (scroll/WASD/free-look)
  // eases in rather than lurching. Seeded to the grab point on commit.
  filament::math::float3 _grabTargetSmoothed{};
};

} // namespace superdex::studio
