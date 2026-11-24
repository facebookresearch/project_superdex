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

#include "superdex_asset_tuner/load_math.h"

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/quaternion_utils.h>

#include <algorithm>
#include <cmath>
#include <numbers>

namespace superdex::tuner {

namespace {

// Index i of the segment [keys[i], keys[i + 1]] holding `t`, for `t` strictly inside the keys'
// span. keys[i + 1].time > t, so the segment is never empty.
template <typename Key>
size_t SegmentIndex(Span<Key const> keys, double t) {
  auto const after = std::upper_bound(
      keys.begin(), keys.end(), t, [](double time, Key const& key) { return time < key.time; });
  return static_cast<size_t>(after - keys.begin()) - 1;
}

// Cubic Hermite from p1 (tangent m1) to p2 (tangent m2) over a segment lasting h, at fraction s.
template <typename T>
T Hermite(T const& p1, T const& p2, T const& m1, T const& m2, double h, double s) {
  double const s2 = s * s;
  double const s3 = s2 * s;
  auto const weight = [](double w) { return static_cast<real>(w); };
  return p1 * weight(2 * s3 - 3 * s2 + 1) + m1 * weight((s3 - 2 * s2 + s) * h) +
      p2 * weight(3 * s2 - 2 * s3) + m2 * weight((s3 - s2) * h);
}

// Fraction `s` of a segment eased in and out: zero slope at both ends.
double Smoothstep(double s) {
  return s * s * (3 - 2 * s);
}

// Catmull-Rom tangent at keys[k] from its neighbors, or zero at the first and last key so a smooth
// curve starts and ends at rest.
template <typename Key, typename ValueOf>
auto CatmullRomTangent(Span<Key const> keys, size_t k, ValueOf valueOf) {
  auto const zero = valueOf(keys[k]) * real{0};
  if (k == 0 || k + 1 >= keys.size()) {
    return zero;
  }
  double const span = keys[k + 1].time - keys[k - 1].time;
  return span > 0 ? (valueOf(keys[k + 1]) - valueOf(keys[k - 1])) * static_cast<real>(1 / span)
                  : zero;
}

// Value at `t` of the curve through `keys` (non-empty, sorted by time), each segment interpolated
// as its first key says.
template <typename Key, typename ValueOf>
auto InterpolateKeys(Span<Key const> keys, double t, ValueOf valueOf) {
  if (t <= keys.front().time) {
    return valueOf(keys.front());
  }
  if (t >= keys.back().time) {
    return valueOf(keys.back());
  }
  size_t const i = SegmentIndex(keys, t);
  Key const& a = keys[i];
  Key const& b = keys[i + 1];
  double const h = b.time - a.time;
  if (!(h > 0)) {
    return valueOf(b); // unsorted keys; callers sort them first
  }
  double const s = (t - a.time) / h;
  switch (a.interpolation) {
    case Interpolation::Step:
      return valueOf(a);
    case Interpolation::Linear:
      return valueOf(a) + (valueOf(b) - valueOf(a)) * static_cast<real>(s);
    case Interpolation::Ease:
      return valueOf(a) + (valueOf(b) - valueOf(a)) * static_cast<real>(Smoothstep(s));
    case Interpolation::Smooth:
      break;
  }
  return Hermite(
      valueOf(a),
      valueOf(b),
      CatmullRomTangent(keys, i, valueOf),
      CatmullRomTangent(keys, i + 1, valueOf),
      h,
      s);
}

mochi::Quaternion WaypointRotation(PoseWaypoint const& waypoint) {
  return mochi::Quaternion::FromRotationVector(waypoint.rotation);
}

// Orientation at `t` through `keys` (non-empty, sorted by time), each segment interpolated as its
// first key says: Linear slerps at constant angular speed, Ease slerps from rest to rest, Smooth is
// Catmull-Rom on the rotation.
template <typename Key, typename RotationOf>
mochi::Quaternion InterpolateRotation(Span<Key const> keys, double t, RotationOf rotationOf) {
  if (t <= keys.front().time) {
    return rotationOf(keys.front());
  }
  if (t >= keys.back().time) {
    return rotationOf(keys.back());
  }
  size_t const i = SegmentIndex(keys, t);
  Key const& a = keys[i];
  Key const& b = keys[i + 1];
  double const h = b.time - a.time;
  if (!(h > 0)) {
    return rotationOf(b); // unsorted keys; callers sort them first
  }
  double const s = (t - a.time) / h;
  mochi::Quaternion const start = rotationOf(a);
  switch (a.interpolation) {
    case Interpolation::Step:
      return start;
    case Interpolation::Linear:
      return mochi::Normalize(mochi::Slerp(start, rotationOf(b), static_cast<real>(s)));
    case Interpolation::Ease:
      return mochi::Normalize(mochi::Slerp(start, rotationOf(b), static_cast<real>(Smoothstep(s))));
    case Interpolation::Smooth:
      break;
  }
  // Catmull-Rom on rotation vectors relative to the segment's start, so both keys are met exactly
  // and the neighbors shape the curve between them.
  mochi::Quaternion const toStart = start.GetConjugate();
  auto const relative = [&](Key const& key) {
    return (toStart * rotationOf(key)).ToRotationVector();
  };
  Real3 const r = Hermite(
      Real3{0.0f, 0.0f, 0.0f},
      relative(b),
      CatmullRomTangent(keys, i, relative),
      CatmullRomTangent(keys, i + 1, relative),
      h,
      s);
  return start * mochi::Quaternion::FromRotationVector(r);
}

} // namespace

mochi::Quaternion ShortestRotation(Real3 const& from, Real3 const& to) {
  real const fromNorm = mochi::Norm(from);
  real const toNorm = mochi::Norm(to);
  if (!(fromNorm > 0) || !(toNorm > 0)) {
    return mochi::Quaternion::Identity();
  }
  Real3 const a = from * (1 / fromNorm);
  Real3 const b = to * (1 / toNorm);
  real const cosAngle = mochi::Dot(a, b);
  constexpr real kParallel = 1.0f - 1e-6f;
  if (cosAngle > kParallel) {
    return mochi::Quaternion::Identity();
  }
  if (cosAngle < -kParallel) {
    // Half a turn about any axis perpendicular to `from`.
    Real3 const helper = std::abs(a[0]) < 0.9f ? Real3{1.0f, 0.0f, 0.0f} : Real3{0.0f, 1.0f, 0.0f};
    return mochi::Quaternion::FromAxisAngle(
        mochi::Normalize(mochi::Cross(a, helper)), std::numbers::pi_v<real>);
  }
  return mochi::Quaternion::FromAxisAngle(
      mochi::Normalize(mochi::Cross(a, b)), std::acos(mochi::Clamp(cosAngle, real{-1}, real{1})));
}

bool AnimatesDirection(LoadSpec const& load) {
  bool const takesDirection = load.type == LoadType::Force || load.type == LoadType::Torque ||
      load.type == LoadType::JointEffort;
  return takesDirection && load.profile.kind == ProfileKind::Keyframes &&
      load.profile.animateDirection && !load.profile.keys.empty();
}

bool AnimatesPoint(LoadSpec const& load) {
  return load.type == LoadType::Force && load.profile.kind == ProfileKind::Keyframes &&
      load.profile.animatePoint && !load.profile.keys.empty();
}

Real3 EvaluateLoadDirection(LoadSpec const& load, double t) {
  if (!AnimatesDirection(load)) {
    return load.direction;
  }
  Span<ProfileKey const> const keys = mochi::MakeConstSpan(load.profile.keys);
  if (load.type == LoadType::JointEffort) {
    // Per-DoF weights, not a direction: blended as plain values.
    return InterpolateKeys(keys, t, [](ProfileKey const& key) { return key.direction; });
  }
  // Each key as the rotation taking +X onto its direction, chained by the smallest turn from the
  // previous key, so consecutive keys are interpolated along the shortest arc (never through zero,
  // even between opposite directions). A zero direction keeps the previous one.
  Real3 const x{1.0f, 0.0f, 0.0f};
  DynamicArray<mochi::Quaternion> rotations;
  rotations.reserve(keys.size());
  Real3 previous = x;
  mochi::Quaternion rotation = mochi::Quaternion::Identity();
  for (ProfileKey const& key : keys) {
    if (mochi::Norm(key.direction) > 0) {
      rotation = ShortestRotation(previous, key.direction) * rotation;
      previous = key.direction;
    }
    rotations.push_back(rotation);
  }
  auto const rotationOf = [&](ProfileKey const& key) {
    return rotations[static_cast<size_t>(&key - keys.data())];
  };
  return InterpolateRotation(keys, t, rotationOf) * x;
}

Real3 EvaluateLoadPoint(LoadSpec const& load, double t) {
  if (!AnimatesPoint(load)) {
    return load.frame == LoadFrame::WorldPoint ? load.pointWorld : load.pointLocal;
  }
  return InterpolateKeys(
      mochi::MakeConstSpan(load.profile.keys), t, [](ProfileKey const& key) { return key.point; });
}

real EvaluateProfile(LoadProfile const& profile, double t) {
  if (profile.kind == ProfileKind::Keyframes) {
    Span<ProfileKey const> const keys = mochi::MakeConstSpan(profile.keys);
    return keys.empty() ? real{0}
                        : InterpolateKeys(keys, t, [](ProfileKey const& key) { return key.value; });
  }
  bool const ended = profile.end > profile.start && t >= profile.end;
  if (t < profile.start || ended) {
    return 0;
  }
  double const elapsed = t - profile.start;
  switch (profile.kind) {
    case ProfileKind::Constant:
      return profile.magnitude;
    case ProfileKind::Envelope: {
      double const rise = profile.riseTime > 0.0 ? elapsed / profile.riseTime : 1.0;
      bool const hasEnd = profile.end > profile.start;
      double const fall =
          hasEnd && profile.fallTime > 0.0 ? (profile.end - t) / profile.fallTime : 1.0;
      return static_cast<real>(profile.magnitude * mochi::Clamp(mochi::Min(rise, fall), 0.0, 1.0));
    }
    case ProfileKind::Sine:
      return static_cast<real>(
          profile.offset +
          profile.magnitude *
              std::sin(2.0 * std::numbers::pi * profile.freqHz * elapsed + profile.phase));
    case ProfileKind::Keyframes:
      break;
  }
  return 0;
}

mochi::TransformRT EvaluatePosePath(Span<PoseWaypoint const> waypoints, double t) {
  MOCHI_ASSERT(!waypoints.empty(), "A pose path needs at least one waypoint.");
  Real3 const position =
      InterpolateKeys(waypoints, t, [](PoseWaypoint const& waypoint) { return waypoint.position; });
  return mochi::TransformRT(InterpolateRotation(waypoints, t, WaypointRotation), position);
}

void EvaluateInputKeys(Span<InputKey const> keys, double t, Span<real> out) {
  MOCHI_ASSERT(!keys.empty(), "Input keys must not be empty.");
  for (size_t j = 0; j < out.size(); ++j) {
    out[j] = InterpolateKeys(keys, t, [j](InputKey const& key) { return key.values[j]; });
  }
}

PathDriveMode EvaluatePathDrive(Span<PoseWaypoint const> waypoints, double t) {
  MOCHI_ASSERT(!waypoints.empty(), "A pose path needs at least one waypoint.");
  auto const after = std::upper_bound(
      waypoints.begin(), waypoints.end(), t, [](double time, PoseWaypoint const& waypoint) {
        return time < waypoint.time;
      });
  return after == waypoints.begin() ? waypoints.front().drive : (after - 1)->drive;
}

real MeanDiagonalInertia(mochi::Real6 const& inertia) {
  return (inertia[0] + inertia[3] + inertia[5]) / 3;
}

DriveScale MakeDriveScale(real ownMass, real ownInertia, real articulationMass) {
  DriveScale scale;
  scale.mass = articulationMass > 0 ? articulationMass : ownMass > 0 ? ownMass : real{1};
  scale.inertia = ownInertia * (ownMass > 0 ? scale.mass / ownMass : real{1});
  return scale;
}

std::array<real, 3> SpringGains(real mass, real hz, real dampingRatio, real maxAcceleration) {
  real const omega = 2 * std::numbers::pi_v<real> * hz;
  if (!(omega > 0) || !(mass > 0)) {
    return {0, 0, -1};
  }
  real const omegaSq = omega * omega;
  return {
      mass * omegaSq,
      2 * std::max(dampingRatio, real{0}) * mass * omega,
      maxAcceleration > 0 ? maxAcceleration / omegaSq : real{-1}};
}

DrivePullEstimate EstimateDrivePull(LoadSpec const& load, DriveScale const& scale, double dt) {
  Span<PoseWaypoint const> const waypoints = mochi::MakeConstSpan(load.waypoints);
  PathDrive const& drive = load.drive;
  auto const drivesPoint = [](PathDriveMode mode) { return mode != PathDriveMode::Off; };
  auto const drivesOrientation = [&](PathDriveMode mode) {
    return mode == PathDriveMode::On && drive.rotation;
  };
  // The drive sees the path move in steps. Velocity is the change in target over at least one step.
  real peakVelocity = 0;
  real peakAngularVelocity = 0;
  auto const addMotion = [&](mochi::TransformRT const& from,
                             mochi::TransformRT const& to,
                             double interval,
                             bool point,
                             bool orientation) {
    real const inverse = static_cast<real>(1 / interval);
    if (point) {
      peakVelocity = std::max(
          peakVelocity, mochi::Norm(to.GetTranslation() - from.GetTranslation()) * inverse);
    }
    if (orientation) {
      real const angle =
          mochi::Norm((to.GetRotation() * from.GetRotation().GetConjugate()).ToRotationVector());
      peakAngularVelocity = std::max(peakAngularVelocity, angle * inverse);
    }
  };
  // Samples per Linear, Smooth or Ease segment. Enough to find a Smooth segment's top velocity
  // within 1%.
  constexpr int kSamplesPerSegment = 32;
  for (size_t i = 0; dt > 0 && i + 1 < waypoints.size(); ++i) {
    PoseWaypoint const& a = waypoints[i];
    PoseWaypoint const& b = waypoints[i + 1];
    double const span = b.time - a.time;
    if (!(span > 0)) {
      continue;
    }
    if (a.interpolation == Interpolation::Step) {
      // The jump to `b` happens within one step. That step may use the drive mode of `a` or `b`.
      addMotion(
          EvaluatePosePath(waypoints, a.time),
          EvaluatePosePath(waypoints, b.time),
          dt,
          drivesPoint(a.drive) || drivesPoint(b.drive),
          drivesOrientation(a.drive) || drivesOrientation(b.drive));
      continue;
    }
    int const samples = std::clamp(static_cast<int>(span / dt), 1, kSamplesPerSegment);
    double const interval = span / samples;
    mochi::TransformRT previous = EvaluatePosePath(waypoints, a.time);
    for (int j = 1; j <= samples; ++j) {
      mochi::TransformRT const next = EvaluatePosePath(waypoints, a.time + j * interval);
      // A segment shorter than a step still takes a whole step.
      addMotion(
          previous, next, std::max(interval, dt), drivesPoint(a.drive), drivesOrientation(a.drive));
      previous = next;
    }
  }

  DrivePullEstimate estimate;
  std::array<real, 3> const position =
      SpringGains(scale.mass, drive.responseHz, drive.dampingRatio, drive.maxAcceleration);
  estimate.capLag = position[2];
  estimate.springForce = position[2] > 0 ? position[0] * position[2] : real{-1};
  estimate.peakVelocity = peakVelocity;
  estimate.damperForce = position[1] * peakVelocity;
  if (drive.rotation) {
    std::array<real, 3> const rotation = SpringGains(
        scale.inertia, drive.rotResponseHz, drive.rotDampingRatio, drive.maxAngularAcceleration);
    estimate.capAngle = rotation[2];
    estimate.springTorque = rotation[2] > 0 ? rotation[0] * rotation[2] : real{-1};
    estimate.peakAngularVelocity = peakAngularVelocity;
    estimate.damperTorque = rotation[1] * peakAngularVelocity;
  } else {
    estimate.springTorque = 0;
  }
  return estimate;
}

void SortByTime(LoadSpec& load) {
  auto const byTime = [](auto const& a, auto const& b) { return a.time < b.time; };
  std::stable_sort(load.waypoints.begin(), load.waypoints.end(), byTime);
  std::stable_sort(load.profile.keys.begin(), load.profile.keys.end(), byTime);
  std::stable_sort(load.inputKeys.begin(), load.inputKeys.end(), byTime);
}

std::array<real, 6>
RigidPointForceToDofs(Real3 const& force, Real3 const& pointWorld, Real3 const& comWorld) {
  Real3 const torque = mochi::Cross(pointWorld - comWorld, force);
  return {force[0], force[1], force[2], torque[0], torque[1], torque[2]};
}

namespace {

// Adds J^T wrench to `inOutGeneralized`; see AddLinkForceToGeneralized for the Jacobian layout.
void AddWrenchToGeneralized(
    Span<real const> jacobian,
    std::array<real, 6> const& wrench,
    Span<real> inOutGeneralized) {
  auto const numDofs = inOutGeneralized.size();
  MOCHI_ASSERT(jacobian.size() == 6 * numDofs, "Jacobian must be 6 x numDofs.");
  for (size_t row = 0; row < 6; ++row) {
    real const w = wrench[row];
    if (w == 0) {
      continue;
    }
    for (size_t dof = 0; dof < numDofs; ++dof) {
      inOutGeneralized[dof] += jacobian[row * numDofs + dof] * w;
    }
  }
}

} // namespace

void AddLinkForceToGeneralized(
    Span<real const> jacobian,
    Real3 const& force,
    Real3 const& pointWorld,
    Real3 const& jacobianRefWorld,
    Span<real> inOutGeneralized) {
  Real3 const torque = mochi::Cross(pointWorld - jacobianRefWorld, force);
  AddWrenchToGeneralized(
      jacobian, {force[0], force[1], force[2], torque[0], torque[1], torque[2]}, inOutGeneralized);
}

void AddLinkTorqueToGeneralized(
    Span<real const> jacobian,
    Real3 const& torque,
    Span<real> inOutGeneralized) {
  AddWrenchToGeneralized(jacobian, {0, 0, 0, torque[0], torque[1], torque[2]}, inOutGeneralized);
}

} // namespace superdex::tuner
