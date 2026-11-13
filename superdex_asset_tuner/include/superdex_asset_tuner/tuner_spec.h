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

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/span.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace superdex::tuner {

enum class LoadType {
  // Force at a point on a dynamic rigid body, or on an articulated link ("actor/link", mapped to
  // joint forces through the link Jacobian).
  Force,
  // Torque on a dynamic rigid body or an articulated link.
  Torque,
  // Generalized force on an articulated joint's DoFs ("actor/joint").
  JointEffort,
  // Pulls a point on a rigid body or link (and the body's orientation) along timed waypoints with a
  // spring-damper, like SuperDex Studio's Physics Drag.
  PosePath,
};

enum class LoadFrame {
  // Direction fixed in the world; the application point rides on the body.
  World,
  // Direction follows the body and turns with it in its local frame.
  Local,
  // Direction and application point (`pointWorld`) both fixed in the world, where the body
  // can move away/independent of the point (but can still be affected by it).
  WorldPoint,
};

enum class ProfileKind {
  Constant,
  // Trapezoid: rises linearly from 0 at `start` to `magnitude` over `riseTime`, holds, then falls
  // linearly back to 0 over `fallTime`, reaching 0 at `end`. Without an `end`, it holds forever.
  // Overlapping rise and fall give a triangle that peaks below `magnitude`.
  Envelope,
  // offset + magnitude * sin(2 pi freqHz (t - start) + phase).
  Sine,
  // Curve through `keys`, holding the first value before them and the last after.
  Keyframes,
};

// How a waypoint or key moves on to the next one.
enum class Interpolation {
  // Holds this value until the next keyframe/waypoint's time.
  Step,
  // Straight line (constant-speed slerp for orientations).
  Linear,
  // Smooth curve through the neighbors (Catmull-Rom); the path's first and last points are passed
  // at rest.
  Smooth,
};

// What a PosePath drive does from a waypoint until the next one.
enum class PathDriveMode {
  // Pulls the point onto the path and, if PathDrive::rotation, turns the body to its orientation.
  On,
  // Pulls only the point; the body turns freely about it.
  PositionOnly,
  // Releases the body.
  Off,
};

enum class ProbeKind {
  BodyPose,
  BodyVelocity,
  JointState,
  ContactForce,
  SolverStats,
  // Wall-clock timings: varies between runs
  Performance,
  // Approximate volume by which the target overlaps other actors.
  Penetration,
};

enum class StopKind {
  // The target body's origin leaves an axis-aligned world box.
  LeavesBox,
  // The target body's origin enters an axis-aligned world box.
  EntersBox,
  // A channel goes above or below a threshold.
  ChannelThreshold,
};

enum class StopComparison {
  Above,
  Below,
};

} // namespace superdex::tuner

MOCHI_ENUM_BEGIN(superdex::tuner::LoadType)
MOCHI_ENUM_ITEM(Force)
MOCHI_ENUM_ITEM(Torque)
MOCHI_ENUM_ITEM(JointEffort)
MOCHI_ENUM_ITEM(PosePath)
MOCHI_ENUM_END()

MOCHI_ENUM_BEGIN(superdex::tuner::LoadFrame)
MOCHI_ENUM_ITEM(World)
MOCHI_ENUM_ITEM(Local)
MOCHI_ENUM_ITEM(WorldPoint)
MOCHI_ENUM_END()

MOCHI_ENUM_BEGIN(superdex::tuner::ProfileKind)
MOCHI_ENUM_ITEM(Constant)
MOCHI_ENUM_ITEM(Envelope)
MOCHI_ENUM_ITEM(Sine)
MOCHI_ENUM_ITEM(Keyframes)
MOCHI_ENUM_END()

MOCHI_ENUM_BEGIN(superdex::tuner::Interpolation)
MOCHI_ENUM_ITEM(Step)
MOCHI_ENUM_ITEM(Linear)
MOCHI_ENUM_ITEM(Smooth)
MOCHI_ENUM_END()

MOCHI_ENUM_BEGIN(superdex::tuner::PathDriveMode)
MOCHI_ENUM_ITEM(On)
MOCHI_ENUM_ITEM(PositionOnly)
MOCHI_ENUM_ITEM(Off)
MOCHI_ENUM_END()

MOCHI_ENUM_BEGIN(superdex::tuner::ProbeKind)
MOCHI_ENUM_ITEM(BodyPose)
MOCHI_ENUM_ITEM(BodyVelocity)
MOCHI_ENUM_ITEM(JointState)
MOCHI_ENUM_ITEM(ContactForce)
MOCHI_ENUM_ITEM(SolverStats)
MOCHI_ENUM_ITEM(Performance)
MOCHI_ENUM_ITEM(Penetration)
MOCHI_ENUM_END()

MOCHI_ENUM_BEGIN(superdex::tuner::StopKind)
MOCHI_ENUM_ITEM(LeavesBox)
MOCHI_ENUM_ITEM(EntersBox)
MOCHI_ENUM_ITEM(ChannelThreshold)
MOCHI_ENUM_END()

MOCHI_ENUM_BEGIN(superdex::tuner::StopComparison)
MOCHI_ENUM_ITEM(Above)
MOCHI_ENUM_ITEM(Below)
MOCHI_ENUM_END()

namespace superdex::tuner {

using mochi::DynamicArray;
using mochi::DynamicString;
using mochi::real;
using mochi::Real3;
using mochi::Span;

inline constexpr std::string_view kTunerFileExtension = ".superdex_tuner";
inline constexpr std::string_view kSuperdexRootMarker = ".superdex_root";

struct ProfileKey {
  double time = 0.0;
  real value = 0;
  // Toward the next key.
  Interpolation interpolation = Interpolation::Linear;
  // Used when the profile's animateDirection / animatePoint is on, in the same frame as the load's
  // own `direction` and point (pointLocal, or pointWorld for WorldPoint).
  Real3 direction{1.0f, 0.0f, 0.0f};
  Real3 point{0.0f, 0.0f, 0.0f};

  MOCHI_STRUCT_BEGIN(superdex::tuner::ProfileKey)
  MOCHI_FIELD(time) MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_FIELD(value)
  MOCHI_FIELD(interpolation)
  MOCHI_FIELD(direction)
  MOCHI_FIELD(point) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_STRUCT_END()
};

struct LoadProfile {
  ProfileKind kind = ProfileKind::Constant;
  real magnitude = 1;
  real offset = 0;
  double start = 0.0;
  // Load is inactive from `end` on. Values <= start mean the load never ends.
  double end = 0.0;
  // Envelope only.
  double riseTime = 1.0;
  double fallTime = 0.0;
  real freqHz = 1;
  real phase = 0;
  // Keyframes only, sorted by time. `start` and `end` do not apply.
  DynamicArray<ProfileKey> keys;
  // Keyframes only: the load's direction (Force, Torque, or JointEffort DoF weights) and
  // application point (Force) come from the keys instead of the load's single values.
  bool animateDirection = false;
  bool animatePoint = false;

  MOCHI_STRUCT_BEGIN(superdex::tuner::LoadProfile)
  MOCHI_FIELD(kind)
  MOCHI_FIELD(magnitude)
  MOCHI_FIELD(offset)
  MOCHI_FIELD(start) MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_FIELD(end) MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_FIELD(riseTime) MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_ATTRIBUTE(PreviouslyKnownAs("rampTime"));
  MOCHI_FIELD(fallTime) MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_FIELD(freqHz) MOCHI_ATTRIBUTE(Units("Hz"));
  MOCHI_FIELD(phase) MOCHI_ATTRIBUTE(Units("rad"));
  MOCHI_FIELD(keys)
  MOCHI_FIELD(animateDirection)
  MOCHI_FIELD(animatePoint)
  MOCHI_STRUCT_END()
};

// PosePath targets, where the load's point should be, and how the body should be oriented, at
// `time`.
struct PoseWaypoint {
  double time = 0.0;
  // World position of the load's point.
  Real3 position{0.0f, 0.0f, 0.0f};
  // World orientation of the body, as a rotation vector.
  Real3 rotation{0.0f, 0.0f, 0.0f};
  // Toward the next waypoint.
  Interpolation interpolation = Interpolation::Smooth;
  // From this waypoint until the next.
  PathDriveMode drive = PathDriveMode::On;

  MOCHI_STRUCT_BEGIN(superdex::tuner::PoseWaypoint)
  MOCHI_FIELD(time) MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_FIELD(position) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(rotation) MOCHI_ATTRIBUTE(Units("rad"));
  MOCHI_FIELD(interpolation)
  MOCHI_FIELD(drive)
  MOCHI_STRUCT_END()
};

// Spring-damper pulling a PosePath load's point onto the path, and optionally the body's
// orientation. Gains scale with the body's mass and inertia (for a link, the whole articulation's).
// Similar to SuperDex Studio's Physics Drag in the viewport
struct PathDrive {
  real responseHz = 6;
  real dampingRatio = 1;
  // Caps the spring's pull at mass * maxAcceleration, 0: no cap.
  real maxAcceleration = 30;
  bool rotation = true;
  real rotResponseHz = 4;
  real rotDampingRatio = 1;
  real maxAngularAcceleration = 30;

  MOCHI_STRUCT_BEGIN(superdex::tuner::PathDrive)
  MOCHI_FIELD(responseHz) MOCHI_ATTRIBUTE(Units("Hz"));
  MOCHI_FIELD(dampingRatio)
  MOCHI_FIELD(maxAcceleration) MOCHI_ATTRIBUTE(Units("m/s^2"));
  MOCHI_FIELD(rotation)
  MOCHI_FIELD(rotResponseHz) MOCHI_ATTRIBUTE(Units("Hz"));
  MOCHI_FIELD(rotDampingRatio)
  MOCHI_FIELD(maxAngularAcceleration) MOCHI_ATTRIBUTE(Units("rad/s^2"));
  MOCHI_STRUCT_END()
};

struct LoadSpec {
  DynamicString name;
  bool enabled = true;
  LoadType type = LoadType::Force;
  DynamicString target;
  LoadFrame frame = LoadFrame::World;
  // Force/torque direction; normalized on use. For JointEffort, per-DoF weights.
  Real3 direction{1.0f, 0.0f, 0.0f};
  // Application point in the body (or link) frame, for forces in World and Local, and the point a
  // PosePath drives.
  Real3 pointLocal{0.0f, 0.0f, 0.0f};
  // Application point in world space, for WorldPoint forces.
  Real3 pointWorld{0.0f, 0.0f, 0.0f};
  // Magnitude over time. Not used by PosePath.
  LoadProfile profile;
  // PosePath only, sorted by time.
  DynamicArray<PoseWaypoint> waypoints;
  PathDrive drive;

  MOCHI_STRUCT_BEGIN(superdex::tuner::LoadSpec)
  MOCHI_FIELD(name)
  MOCHI_FIELD(enabled)
  MOCHI_FIELD(type)
  MOCHI_FIELD(target)
  MOCHI_FIELD(frame)
  MOCHI_FIELD(direction)
  MOCHI_FIELD(pointLocal) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(pointWorld) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(profile)
  MOCHI_FIELD(waypoints)
  MOCHI_FIELD(drive)
  MOCHI_STRUCT_END()
};

struct ProbeSpec {
  ProbeKind kind = ProbeKind::BodyPose;
  // Rigid actor name, or "actor/joint" for JointState. Ignored for SolverStats and Performance.
  DynamicString target;
  // Penetration: leave out overlap with static unbounded actors (ground planes).
  bool ignoreGround = false;

  MOCHI_STRUCT_BEGIN(superdex::tuner::ProbeSpec)
  MOCHI_FIELD(kind)
  MOCHI_FIELD(target)
  MOCHI_FIELD(ignoreGround)
  MOCHI_STRUCT_END()
};

// Ends a run early when triggered: the CLI stops with status "stopped", Studio pauses.
struct StopCondition {
  DynamicString name;
  bool enabled = true;
  StopKind kind = StopKind::LeavesBox;
  // Checked from this simulation time on, e.g. to let a scene settle first.
  double start = 0.0;
  // Triggers only once the condition has held continuously for this long; 0 triggers at once.
  double holdTime = 0.0;
  // LeavesBox / EntersBox: rigid body or articulated link ("actor/link"), and the world-space box
  // its origin must stay inside (LeavesBox) or outside (EntersBox).
  DynamicString target;
  Real3 boxMin{-1.0f, -1.0f, -1.0f};
  Real3 boxMax{1.0f, 1.0f, 1.0f};
  // ChannelThreshold: a channel produced by the tuner's probes or loads, e.g. "block.pos.z".
  DynamicString channel;
  StopComparison comparison = StopComparison::Below;
  real threshold = 0;

  MOCHI_STRUCT_BEGIN(superdex::tuner::StopCondition)
  MOCHI_FIELD(name)
  MOCHI_FIELD(enabled)
  MOCHI_FIELD(kind)
  MOCHI_FIELD(start) MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_FIELD(holdTime) MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_FIELD(target)
  MOCHI_FIELD(boxMin) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(boxMax) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(channel)
  MOCHI_FIELD(comparison)
  MOCHI_FIELD(threshold)
  MOCHI_STRUCT_END()
};

// One plot area in SuperDex Studio's Tuner Plots window (layout only; ignored by the CLI).
struct PlotSpec {
  // Channel names as in signals.csv, e.g. "load.push.mag" or "block.pos.x".
  DynamicArray<DynamicString> channels;
  // Plot height in (unscaled) pixels.
  float height = 200.0f;

  MOCHI_STRUCT_BEGIN(superdex::tuner::PlotSpec)
  MOCHI_FIELD(channels)
  MOCHI_FIELD(height)
  MOCHI_STRUCT_END()
};

// Sets the JSON value at `path` in the prefab before it is simulated.
// Path syntax: dot-separated keys, with `[index]` or `[name=value]` array selectors, e.g.:
//
// `actors.rigid[name=cube].contact.coulombFrictionCoefficient`.
// A `prefabs[...]` prefix reaches into a nested prefab's content (for that reference only), e.g.
//
// `prefabs[name=Nested1].actors.rigid[name=Rigid1].isStatic`.
//
// `value` is JSON text; text that is not valid JSON is used as a string.
struct PrefabOverride {
  DynamicString path;
  DynamicString value;

  MOCHI_STRUCT_BEGIN(superdex::tuner::PrefabOverride)
  MOCHI_FIELD(path)
  MOCHI_FIELD(value)
  MOCHI_STRUCT_END()
};

struct RecordSpec {
  bool enabled = true;
  bool contactForces = false;

  MOCHI_STRUCT_BEGIN(superdex::tuner::RecordSpec)
  MOCHI_FIELD(enabled)
  MOCHI_ATTRIBUTE(DisplayName("Record sim.rrd"));
  MOCHI_ATTRIBUTE(Description(
      "In Studio enables recording to sim.rrd when the Record button is armed. "
      "On CLI runs, records the run's simulation to sim.rrd in the run folder."));
  MOCHI_FIELD(contactForces)
  MOCHI_ATTRIBUTE(DisplayName("Record Contact Forces"));
  MOCHI_ATTRIBUTE(Description(
      "Include each step's contact points and forces in sim.rrd (larger files). Scenes with "
      "actors other than rigid and articulated actors cannot record them."));
  MOCHI_STRUCT_END()
};

// Matches SuperDex Studio (Z-up) so CLI and Studio runs agree.
inline Real3 const kDefaultGravity{0.0f, 0.0f, -9.81f};

struct RunSpec {
  double dt = 0.01;
  double duration = 2.0;
  // Run physics on one thread, in the CLI and in Studio.
  bool singleThreaded = true;
  // Applied after the prefab, so it overrides prefab scene gravity when set.
  std::optional<Real3> gravity = std::nullopt;
  bool groundPlane = true;
  // Override for the ground plane height. Unset: the lowest point of the prefab (as in the prefab
  // editor), so bodies resting at their authored pose sit on the ground.
  std::optional<real> groundHeight = std::nullopt;
  // Contact parameters of the ground plane. Unset: SuperDex's defaults.
  std::optional<mochi::ContactParams> groundContact = std::nullopt;
  RecordSpec record;

  MOCHI_STRUCT_BEGIN(superdex::tuner::RunSpec)
  MOCHI_FIELD(dt) MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_ATTRIBUTE(DisplayName("Time Step"));
  MOCHI_ATTRIBUTE(Description("Fixed simulation time step, used by both Studio and the CLI."));
  MOCHI_ATTRIBUTE(FloatRange(0.00001, 1.0));
  MOCHI_FIELD(duration) MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_ATTRIBUTE(Description(
      "Simulated time per run. Studio pauses here when Stop at Duration is on; the CLI stops."));
  MOCHI_FIELD(singleThreaded)
  MOCHI_ATTRIBUTE(DisplayName("Single-Threaded"));
  MOCHI_ATTRIBUTE(Description(
      "Run physics on one thread, in the CLI and in Studio. Repeated CLI runs are then "
      "bit-identical on the same build and machine. Large scenes run slower."));
  MOCHI_FIELD(gravity) MOCHI_ATTRIBUTE(Units("m/s^2"));
  MOCHI_ATTRIBUTE(Description(
      "Overrides the prefab's scene gravity when set. Unset: the prefab's gravity, else "
      "(0, 0, -9.81)."));
  MOCHI_FIELD(groundPlane)
  MOCHI_ATTRIBUTE(Description("Add a static ground plane (normal +Z) at Ground Height."));
  MOCHI_FIELD(groundHeight) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_ATTRIBUTE(
      Description("Override for the ground plane height. Unset: the lowest point of the prefab."));
  MOCHI_FIELD(groundContact)
  MOCHI_ATTRIBUTE(Description(
      "Contact parameters of the ground plane. Unset: SuperDex's defaults. A body touching the ground "
      "uses the ground's smoothing distance and thresholds, its own penalty coefficient and "
      "friction falloff velocity, and the geometric mean of both friction and damping "
      "coefficients."));
  MOCHI_FIELD(record)
  MOCHI_STRUCT_END()
};

struct TunerSpec {
  std::optional<DynamicString> comment = std::nullopt;
  // "./"-prefixed paths are relative to the tuner file, others to the assets root.
  DynamicString prefab;
  // Root for asset-relative paths, relative to the tuner file. Empty: the directory of the nearest
  // `.superdex_root` marker (as in SuperDex Studio), else $SUPERDEX_ASSETS_PATH, else the directory
  // of the referencing file.
  DynamicString assetsRoot;
  DynamicArray<PrefabOverride> prefabOverrides;
  DynamicArray<LoadSpec> loads;
  DynamicArray<ProbeSpec> probes;
  RunSpec run;
  DynamicArray<StopCondition> stopConditions;
  // Studio plot layout. Empty: one plot showing every load channel.
  DynamicArray<PlotSpec> plots;

  MOCHI_STRUCT_BEGIN(superdex::tuner::TunerSpec)
  MOCHI_FIELD(comment)
  MOCHI_FIELD(prefab)
  MOCHI_FIELD(assetsRoot)
  MOCHI_FIELD(prefabOverrides)
  MOCHI_FIELD(loads)
  MOCHI_FIELD(probes)
  MOCHI_FIELD(run)
  MOCHI_FIELD(stopConditions)
  MOCHI_FIELD(plots)
  MOCHI_STRUCT_END()
};

// A `--set path=value` override applied to the tuner spec JSON (same syntax as PrefabOverride).
struct SpecOverride {
  std::string path;
  std::string value;
};

// Functions returning bool fill `outError` and return false on failure.

bool LoadTunerSpecFromJson(
    std::string_view json,
    Span<SpecOverride const> overrides,
    TunerSpec& outSpec,
    std::string& outError);

bool LoadTunerSpecFromFile(
    std::filesystem::path const& path,
    Span<SpecOverride const> overrides,
    TunerSpec& outSpec,
    std::string& outError);

[[nodiscard]] std::string SaveTunerSpecToJson(TunerSpec const& spec);

bool SaveTunerSpecToFile(
    TunerSpec const& spec,
    std::filesystem::path const& path,
    std::string& outError);

// Root for asset-relative paths referenced from `anchorPath` (the tuner file, or the prefab for the
// prefab's own references). See TunerSpec::assetsRoot.
[[nodiscard]] std::filesystem::path ResolveAssetsRoot(
    TunerSpec const& spec,
    std::filesystem::path const& tunerFilePath,
    std::filesystem::path const& anchorPath);

[[nodiscard]] std::filesystem::path ResolvePrefabPath(
    TunerSpec const& spec,
    std::filesystem::path const& tunerFilePath);

// Loads the prefab (applying `spec.prefabOverrides`) with all nested prefabs and shapes, ready for
// mochi::prefab::AddToScene.
bool LoadTunerPrefab(
    TunerSpec const& spec,
    std::filesystem::path const& tunerFilePath,
    mochi::Context* context,
    mochi::prefab::ScenePrefab& outPrefab,
    std::string& outError);

// Applies `overrides` to an already-loaded prefab (round-trips through JSON). Nested prefabs and
// shapes must be (re)loaded afterwards, e.g. with mochi::prefab::EnsureFullyLoaded.
bool ApplyPrefabOverrides(
    mochi::prefab::ScenePrefab& prefab,
    Span<PrefabOverride const> overrides,
    std::string& outError);

} // namespace superdex::tuner
