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

#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/reflection.h>

#include <mochi_physics/mochi_physics.h>

#include <mochi_renderer/types.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace superdex::studio {

using mochi::operator""_r; // real-valued literals (0_r, -9.81_r, ...)

struct AssetBrowserSettings {
  std::vector<std::string> rootPaths;
  std::string lastPath;
  bool showDirectoryTree = true;
  bool showFilters = true;
  bool showImportableFiles = true;
  bool showUnknownFiles = true;
  bool sortByType = true;
  // Max filesystem entries (files + folders) the Asset Browser scans under a root, and the ceiling
  // above which a folder is refused as a root. Bounds the synchronous scan so a huge tree (e.g. the
  // repo root) can't hang the UI. A properly-scoped asset folder is well under this (~2500 in
  // practice); the default leaves headroom while keeping a worst-case scan to a few seconds.
  int fileFolderLimit = 5000;

  MOCHI_STRUCT_BEGIN(superdex::studio::AssetBrowserSettings)
  // rootPaths / lastPath are the open workspace, not a preference: hidden from the Settings window.
  MOCHI_FIELD(rootPaths) MOCHI_ATTRIBUTE(HideFromEditor);
  MOCHI_FIELD(lastPath) MOCHI_ATTRIBUTE(HideFromEditor);
  MOCHI_FIELD(showDirectoryTree)
  MOCHI_ATTRIBUTE(Description("Show the directory tree sidebar next to the asset tiles."));
  MOCHI_FIELD(showFilters)
  MOCHI_ATTRIBUTE(Description("Show the search box and asset-type filter swatches."));
  MOCHI_FIELD(showImportableFiles)
  MOCHI_ATTRIBUTE(
      Description("List files that are not native assets but can be imported, such as URDF."));
  MOCHI_FIELD(showUnknownFiles)
  MOCHI_ATTRIBUTE(Description("List files that SuperDex Studio can neither open nor import."));
  MOCHI_FIELD(sortByType)
  MOCHI_ATTRIBUTE(
      Description("Group entries by asset type (folders first) instead of sorting them by path."));
  MOCHI_FIELD(fileFolderLimit)
  MOCHI_ATTRIBUTE(DisplayName("File/Folder Limit"));
  MOCHI_ATTRIBUTE(Description(
      "Maximum number of files and folders scanned under a root, and the ceiling above which a "
      "folder is refused as a root. Bounds the scan so a huge tree cannot hang the UI."));
  MOCHI_ATTRIBUTE(IntRange(1, 1000000));
  MOCHI_STRUCT_END()
};

struct RecentEntriesSettings {
  std::vector<std::string> files;
  std::vector<std::string> folders;

  MOCHI_STRUCT_BEGIN(superdex::studio::RecentEntriesSettings)
  MOCHI_FIELD_NAME(files, "recentFiles")
  MOCHI_FIELD_NAME(folders, "recentFolders")
  MOCHI_STRUCT_END()
};

struct BotEditorSettings {
  bool showlinkTreeHierarchical = true;

  MOCHI_STRUCT_BEGIN(superdex::studio::BotEditorSettings)
  MOCHI_FIELD(showlinkTreeHierarchical)
  MOCHI_ATTRIBUTE(DisplayName("Hierarchical Link Tree"));
  MOCHI_ATTRIBUTE(Description(
      "List a bot's links in parent-child order rather than in the order they are stored."));
  MOCHI_STRUCT_END()
};

// Linear RGBA in [0, 1]. A plain 4-float array so the settings header stays free of renderer
// types; converted to the renderer's color type at the draw call sites.
using Color4 = mochi::NdArray<float, 4>;

// Appearance of the bot debug visualizations. Visibility is per-editor (toggled from the viewport
// Show menu); everything here is app-wide.
struct LinkVisualizationSettings {
  bool scaleToLinkSize = true;
  float linkVisualizationScale = 1.0f;
  float axisLength = 0.05f;
  float comRadius = 0.01f;
  Color4 inertiaColor{0.2f, 0.8f, 0.2f, 0.4f};
  Color4 comColor{1.0f, 0.0f, 0.0f, 0.8f};
  Color4 xAxisColor{233.f / 255.f, 55.f / 255.f, 81.f / 255.f, 1.0f};
  Color4 yAxisColor{131.f / 255.f, 204.f / 255.f, 15.f / 255.f, 1.0f};
  Color4 zAxisColor{46.f / 255.f, 134.f / 255.f, 233.f / 255.f, 1.0f};

  MOCHI_STRUCT_BEGIN(superdex::studio::LinkVisualizationSettings)
  MOCHI_FIELD(scaleToLinkSize)
  MOCHI_ATTRIBUTE(Description(
      "Size the link visualizations from each link's extents instead of a fixed size."));
  MOCHI_FIELD(linkVisualizationScale)
  MOCHI_ATTRIBUTE(DisplayName("Visualization Scale"));
  MOCHI_ATTRIBUTE(Description("Multiplier applied on top of the link visualization size."));
  MOCHI_ATTRIBUTE(FloatRange(0.01, 100.0));
  MOCHI_FIELD(axisLength)
  MOCHI_ATTRIBUTE(Description("Length of the local-transform axes when not scaling to link size."));
  MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(comRadius)
  MOCHI_ATTRIBUTE(DisplayName("Center of Mass Radius"));
  MOCHI_ATTRIBUTE(
      Description("Radius of the center-of-mass marker when not scaling to link size."));
  MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(inertiaColor)
  MOCHI_ATTRIBUTE(Color);
  MOCHI_ATTRIBUTE(Description("Color of the equivalent-inertia box."));
  MOCHI_FIELD(comColor)
  MOCHI_ATTRIBUTE(Color);
  MOCHI_ATTRIBUTE(DisplayName("Center of Mass Color"));
  MOCHI_FIELD(xAxisColor) MOCHI_ATTRIBUTE(Color);
  MOCHI_ATTRIBUTE(DisplayName("X Axis Color"));
  MOCHI_FIELD(yAxisColor) MOCHI_ATTRIBUTE(Color);
  MOCHI_ATTRIBUTE(DisplayName("Y Axis Color"));
  MOCHI_FIELD(zAxisColor) MOCHI_ATTRIBUTE(Color);
  MOCHI_ATTRIBUTE(DisplayName("Z Axis Color"));
  MOCHI_STRUCT_END()
};

struct JointVisualizationSettings {
  bool scaleToLinkSize = true;
  float jointVisualizationScale = 1.0f;
  Color4 jointColor{0.6f, 0.2f, 0.9f, 0.7f};
  Color4 jointFanColor{0.6f, 0.2f, 0.9f, 0.4f};

  MOCHI_STRUCT_BEGIN(superdex::studio::JointVisualizationSettings)
  MOCHI_FIELD(scaleToLinkSize)
  MOCHI_ATTRIBUTE(Description(
      "Size the joint visualizations from the owning link's extents instead of a fixed size."));
  MOCHI_FIELD(jointVisualizationScale)
  MOCHI_ATTRIBUTE(DisplayName("Visualization Scale"));
  MOCHI_ATTRIBUTE(Description("Multiplier applied on top of the joint visualization size."));
  MOCHI_ATTRIBUTE(FloatRange(0.01, 100.0));
  MOCHI_FIELD(jointColor)
  MOCHI_ATTRIBUTE(Color);
  MOCHI_ATTRIBUTE(Description("Color of joint axes, limit caps and cycle pivots."));
  MOCHI_FIELD(jointFanColor)
  MOCHI_ATTRIBUTE(Color);
  MOCHI_ATTRIBUTE(Description("Color of the swept area showing a joint's range of motion."));
  MOCHI_STRUCT_END()
};

struct TransmissionVisualizationSettings {
  Color4 linearTransmissionColor{1.0f, 0.5f, 0.0f, 0.8f};
  Color4 linearTransmissionCompressedColor{0.25f, 0.12f, 0.0f, 0.8f};
  Color4 spatialTendonColor{0.6f, 0.2f, 0.9f, 0.8f};
  Color4 spatialTendonCompressedColor{0.18f, 0.06f, 0.27f, 0.8f};

  MOCHI_STRUCT_BEGIN(superdex::studio::TransmissionVisualizationSettings)
  MOCHI_FIELD(linearTransmissionColor)
  MOCHI_ATTRIBUTE(Color);
  MOCHI_ATTRIBUTE(Description("Color of a linear transmission that is stretched."));
  MOCHI_FIELD(linearTransmissionCompressedColor)
  MOCHI_ATTRIBUTE(Color);
  MOCHI_ATTRIBUTE(DisplayName("Linear Transmission Compressed Color"));
  MOCHI_ATTRIBUTE(Description("Color of a linear transmission that is slack or compressed."));
  MOCHI_FIELD(spatialTendonColor)
  MOCHI_ATTRIBUTE(Color);
  MOCHI_ATTRIBUTE(Description("Color of a spatial tendon that is stretched."));
  MOCHI_FIELD(spatialTendonCompressedColor)
  MOCHI_ATTRIBUTE(Color);
  MOCHI_ATTRIBUTE(DisplayName("Spatial Tendon Compressed Color"));
  MOCHI_ATTRIBUTE(Description("Color of a spatial tendon that is slack or compressed."));
  MOCHI_STRUCT_END()
};

// Appearance of the viewport ground grid. Every viewport draws the same grid, at the height of its
// scene's lowest point; whether it is shown is per-editor (toggled from the viewport Show menu),
// seeded from `showByDefault`.
struct GroundGridSettings {
  bool showByDefault = true;
  float gridExtents = 1.0f;
  float gridSpacing = 0.1f;
  Color4 gridColor{0.3f, 0.3f, 0.3f, 0.25f};

  MOCHI_STRUCT_BEGIN(superdex::studio::GroundGridSettings)
  MOCHI_FIELD(showByDefault)
  MOCHI_ATTRIBUTE(DisplayName("Show by Default"));
  MOCHI_ATTRIBUTE(Description(
      "Whether a newly opened editor starts with the grid visible. Each editor can then show or "
      "hide it from the viewport's Show menu (G) for that session."));
  MOCHI_FIELD(gridExtents)
  MOCHI_ATTRIBUTE(DisplayName("Grid Extents"));
  MOCHI_ATTRIBUTE(Description("Width of the ground grid."));
  MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_ATTRIBUTE(FloatRange(0.1, 100.0));
  MOCHI_FIELD(gridSpacing)
  MOCHI_ATTRIBUTE(Description("Distance between adjacent grid lines."));
  MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_ATTRIBUTE(FloatRange(0.01, 10.0));
  MOCHI_FIELD(gridColor) MOCHI_ATTRIBUTE(Color);
  MOCHI_STRUCT_END()
};

struct BotVisualizationSettings {
  LinkVisualizationSettings link;
  JointVisualizationSettings joint;
  TransmissionVisualizationSettings transmission;

  MOCHI_STRUCT_BEGIN(superdex::studio::BotVisualizationSettings)
  MOCHI_FIELD(link) MOCHI_ATTRIBUTE(DisplayName("Links"));
  MOCHI_FIELD(joint) MOCHI_ATTRIBUTE(DisplayName("Joints"));
  MOCHI_FIELD(transmission) MOCHI_ATTRIBUTE(DisplayName("Transmissions"));
  MOCHI_STRUCT_END()
};

// Renderer options shared by every viewport, plus the viewport background color used when the
// skybox is off.
struct GraphicsSettings {
  mochi_renderer::SceneViewSettings view;
  Color4 clearColor{0.056f, 0.058f, 0.074f, 1.0f};
  std::string iblPath{"studio_small_08_1k.hdr"};

  MOCHI_STRUCT_BEGIN(superdex::studio::GraphicsSettings)
  MOCHI_FIELD(view)
  MOCHI_FIELD(clearColor) MOCHI_ATTRIBUTE(Color);
  MOCHI_FIELD(iblPath) MOCHI_ATTRIBUTE(HideFromEditor);
  MOCHI_STRUCT_END()
};

struct ViewportCameraSettings {
  double flySpeed = 1.0;

  MOCHI_STRUCT_BEGIN(superdex::studio::ViewportCameraSettings)
  MOCHI_FIELD(flySpeed)
  MOCHI_ATTRIBUTE(Description(
      "Last-used free-fly camera speed, restored so newly opened viewports keep the same feel."));
  MOCHI_ATTRIBUTE(Units("m/s"));
  MOCHI_STRUCT_END()
};

struct ViewportSelectionSettings {
  double highlightOverlayOpacity = 0.35;

  MOCHI_STRUCT_BEGIN(superdex::studio::ViewportSelectionSettings)
  MOCHI_FIELD(highlightOverlayOpacity)
  MOCHI_ATTRIBUTE(
      Description("Opacity of the see-through selection highlight (0 = invisible, 1 = opaque)."));
  MOCHI_ATTRIBUTE(FloatRange(0.0, 1.0));
  MOCHI_STRUCT_END()
};

struct TransformGizmoSettings {
  bool enabled = false;
  double translate = 0.01;
  double rotateDeg = 15.0;
  double scale = 0.25;

  MOCHI_STRUCT_BEGIN(superdex::studio::TransformGizmoSettings)
  MOCHI_FIELD(enabled)
  MOCHI_ATTRIBUTE(DisplayName("Snapping Enabled"));
  MOCHI_ATTRIBUTE(
      Description("Snap transform-gizmo drags to fixed increments. Holding Ctrl inverts this."));
  MOCHI_FIELD(translate)
  MOCHI_ATTRIBUTE(DisplayName("Translate Increment"));
  MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(rotateDeg)
  MOCHI_ATTRIBUTE(DisplayName("Rotate Increment"));
  MOCHI_ATTRIBUTE(Units("deg"));
  MOCHI_FIELD(scale) MOCHI_ATTRIBUTE(DisplayName("Scale Increment"));
  MOCHI_STRUCT_END()
};

// Everything that applies to every viewport, grouped into the sections the Settings window shows.
struct ViewportSettings {
  ViewportCameraSettings camera;
  ViewportSelectionSettings selection;
  TransformGizmoSettings transformGizmo;
  GroundGridSettings groundGrid;

  MOCHI_STRUCT_BEGIN(superdex::studio::ViewportSettings)
  MOCHI_FIELD(camera)
  MOCHI_FIELD(selection)
  MOCHI_FIELD(transformGizmo) MOCHI_ATTRIBUTE(DisplayName("Transform Gizmo"));
  MOCHI_FIELD(groundGrid) MOCHI_ATTRIBUTE(DisplayName("Ground Grid"));
  MOCHI_STRUCT_END()
};

struct LogConsoleSettings {
  bool focusOnError = true;
  bool showVerbose = false;
  bool showInfo = true;
  bool showWarning = true;
  bool showError = true;

  MOCHI_STRUCT_BEGIN(superdex::studio::LogConsoleSettings)
  MOCHI_FIELD(focusOnError)
  MOCHI_ATTRIBUTE(DisplayName("Focus on Error"));
  MOCHI_ATTRIBUTE(Description("Bring the Log Console forward whenever an error is logged."));
  MOCHI_FIELD(showVerbose) MOCHI_ATTRIBUTE(DisplayName("Show Verbose"));
  MOCHI_FIELD(showInfo) MOCHI_ATTRIBUTE(DisplayName("Show Info"));
  MOCHI_FIELD(showWarning) MOCHI_ATTRIBUTE(DisplayName("Show Warnings"));
  MOCHI_FIELD(showError) MOCHI_ATTRIBUTE(DisplayName("Show Errors"));
  MOCHI_STRUCT_END()
};

// Tunables for the interactive mouse force-drag tool (PhysicsDragController). Snapshotted per
// simulation session when the controller is created.
struct PhysicsDragSettings {
  // Position spring, frequency-based: stiffness = m*w^2, damping = 2*zeta*m*w, w = 2*pi*f. Gains
  // scale with the grabbed mass so the feel is consistent and heavy bodies still move.
  float responseFrequencyHz = 6.0f;
  float dampingRatio = 1.0f;
  float maxAcceleration = 30.0f;
  float maxTargetSpeed = 10.0f;
  float maxStretch = 0.25f;
  bool enableRotationConstraint = true;
  float rotResponseFrequencyHz = 4.0f;
  float rotDampingRatio = 1.0f;
  float maxAngularAcceleration = 30.0f;
  float softGrabRadiusFraction = 0.15f;
  int softGrabMaxNodes = 32;
  float debugSphereRadius = 0.005f;

  MOCHI_STRUCT_BEGIN(superdex::studio::PhysicsDragSettings)
  MOCHI_FIELD(responseFrequencyHz)
  MOCHI_ATTRIBUTE(DisplayName("Response Frequency"));
  MOCHI_ATTRIBUTE(Description("How quickly a grabbed body chases the mouse. Higher is stiffer."));
  MOCHI_ATTRIBUTE(Units("Hz"));
  MOCHI_FIELD(dampingRatio)
  MOCHI_ATTRIBUTE(
      Description("Damping of the grab spring. 1 is critically damped; below 1 overshoots."));
  MOCHI_FIELD(maxAcceleration)
  MOCHI_ATTRIBUTE(Description(
      "Caps the grab force so a drag yields to contacts instead of pushing through them."));
  MOCHI_ATTRIBUTE(Units("m/s^2"));
  MOCHI_FIELD(maxTargetSpeed)
  MOCHI_ATTRIBUTE(
      Description("Rate limit on the spring target, so fast mouse motion cannot spike forces."));
  MOCHI_ATTRIBUTE(Units("m/s"));
  MOCHI_FIELD(maxStretch)
  MOCHI_ATTRIBUTE(
      Description("How far the spring target may drift from the point that was grabbed."));
  MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(enableRotationConstraint)
  MOCHI_ATTRIBUTE(
      Description("Softly pin the grab-time orientation so a single-point grab does not dangle."));
  MOCHI_FIELD(rotResponseFrequencyHz)
  MOCHI_ATTRIBUTE(DisplayName("Rotation Response Frequency"));
  MOCHI_ATTRIBUTE(Description("Stiffness of the optional rotation constraint."));
  MOCHI_ATTRIBUTE(Units("Hz"));
  MOCHI_FIELD(rotDampingRatio)
  MOCHI_ATTRIBUTE(DisplayName("Rotation Damping Ratio"));
  MOCHI_ATTRIBUTE(Description("Damping of the optional rotation constraint."));
  MOCHI_FIELD(maxAngularAcceleration)
  MOCHI_ATTRIBUTE(Description("Caps the torque applied by the rotation constraint."));
  MOCHI_ATTRIBUTE(Units("rad/s^2"));
  MOCHI_FIELD(softGrabRadiusFraction)
  MOCHI_ATTRIBUTE(Description(
      "Soft-body grab radius, as a fraction of the body's bounding diagonal. A larger value "
      "moves a wider patch instead of pinching a single node."));
  MOCHI_ATTRIBUTE(FloatRange(0.0, 1.0));
  MOCHI_FIELD(softGrabMaxNodes)
  MOCHI_ATTRIBUTE(
      Description("Upper bound on the nodes a soft-body grab springs, closest to the pick first."));
  MOCHI_ATTRIBUTE(IntRange(1, 1024));
  MOCHI_FIELD(debugSphereRadius)
  MOCHI_ATTRIBUTE(Description("Radius of the debug gizmo drawn at the grab point."));
  MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_STRUCT_END()
};

// Mochi's debug draw. `features` is keyed by mochi's runtime feature names (see
// mochi::DebugDraw::GetFeatureName) rather than an index, so it survives features being added or
// reordered. It has no reflected widget -- it is edited from the per-editor Physics Settings
// window, which is where the live feature list is known.
struct PhysicsDebugSettings {
  bool enabled = false;
  std::unordered_map<std::string, bool> features{{"Actor Mesh", true}};

  MOCHI_STRUCT_BEGIN(superdex::studio::PhysicsDebugSettings)
  MOCHI_FIELD(enabled)
  MOCHI_ATTRIBUTE(DisplayName("Enable Debug Draw"));
  MOCHI_ATTRIBUTE(Description("Draw mochi's physics debug visualization while simulating."));
  MOCHI_FIELD(features) MOCHI_ATTRIBUTE(HideFromEditor);
  MOCHI_STRUCT_END()
};

// Studio-only additions to a simulation session: things the studio puts in the scene for
// convenience that are never part of the asset.
struct StudioPhysicsSettings {
  bool groundPlane = true;

  MOCHI_STRUCT_BEGIN(superdex::studio::StudioPhysicsSettings)
  MOCHI_FIELD(groundPlane)
  MOCHI_ATTRIBUTE(Description(
      "Add a static ground plane at the lowest point of the staged asset when a simulation "
      "starts. Takes effect the next time you press play."));
  MOCHI_STRUCT_END()
};

// The scene-level half of the simulation settings. Solver params sit alongside rather than inside,
// so every place that edits these shows Scene and Solver as sibling sections.
struct PhysicsSceneSettings {
  mochi::Real3 gravity{0_r, 0_r, -9.81_r};
  bool useFixedTimeStep = false;
  double fixedTimeStepSeconds = 0.01;
  bool throttleToRealTime = false;

  MOCHI_STRUCT_BEGIN(superdex::studio::PhysicsSceneSettings)
  MOCHI_FIELD(gravity) MOCHI_ATTRIBUTE(Units("m/s^2"));
  MOCHI_FIELD(useFixedTimeStep)
  MOCHI_ATTRIBUTE(DisplayName("Use Fixed Time Step"));
  MOCHI_ATTRIBUTE(Description(
      "Step by a fixed amount instead of following the wall clock. Makes a session repeatable, "
      "but the simulation no longer tracks real time."));
  MOCHI_FIELD(fixedTimeStepSeconds)
  MOCHI_ATTRIBUTE(DisplayName("Fixed Time Step"));
  MOCHI_ATTRIBUTE(Description("Time step used when Use Fixed Time Step is on."));
  MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_ATTRIBUTE(FloatRange(0.00001, 1.0));
  MOCHI_FIELD(throttleToRealTime)
  MOCHI_ATTRIBUTE(DisplayName("Throttle to Real Time"));
  MOCHI_ATTRIBUTE(Description(
      "Sleep when a fixed step finishes early so simulation time does not outrun the wall "
      "clock. Only applies with a fixed time step."));
  MOCHI_STRUCT_END()
};

// Defaults every simulation session starts from. Each editor seeds a copy at initialization and may
// override it for the session from its Physics Settings window; settings the asset itself defines
// (a prefab's gravity/solver) are applied afterwards and win.
struct PhysicsSettings {
  PhysicsSceneSettings scene;
  mochi::SolverParams solver;
  PhysicsDebugSettings debug;
  StudioPhysicsSettings studio;

  MOCHI_STRUCT_BEGIN(superdex::studio::PhysicsSettings)
  MOCHI_FIELD(scene)
  MOCHI_FIELD(solver)
  MOCHI_FIELD(debug) MOCHI_ATTRIBUTE(DisplayName("Debug Draw"));
  MOCHI_FIELD(studio)
  MOCHI_STRUCT_END()
};

// Ordered to match the Settings window's category list. `physicsDrag` has no category of its own
// (it is drawn under Physics); `recentEntries` and `windowVisibility` are app state rather than
// preferences, so they sit at the end.
struct AppSettings {
  PhysicsSettings physics;
  PhysicsDragSettings physicsDrag;
  GraphicsSettings graphics;
  ViewportSettings viewport;
  AssetBrowserSettings assetBrowser;
  LogConsoleSettings logConsole;
  BotEditorSettings botEditor;
  BotVisualizationSettings botVisualization;
  RecentEntriesSettings recentEntries;
  std::unordered_map<std::string, bool> windowVisibility;

  MOCHI_STRUCT_BEGIN(superdex::studio::AppSettings)
  MOCHI_FIELD(physics)
  MOCHI_FIELD(physicsDrag)
  MOCHI_FIELD(graphics)
  MOCHI_FIELD(viewport)
  MOCHI_FIELD(assetBrowser)
  MOCHI_FIELD(logConsole)
  MOCHI_FIELD(botEditor)
  MOCHI_FIELD(botVisualization)
  MOCHI_FIELD(recentEntries) MOCHI_ATTRIBUTE(HideFromEditor);
  MOCHI_FIELD(windowVisibility) MOCHI_ATTRIBUTE(HideFromEditor);
  MOCHI_STRUCT_END()
};

constexpr int kMaxRecentFiles = 10;

} // namespace superdex::studio
