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

#include "superdex_asset_tuner/json_path.h"
#include "superdex_asset_tuner/load_math.h"

#include <gtest/gtest.h>

#include <mochi_core/utils/math_utils.h>

#include <array>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

namespace superdex::tuner {
namespace {

LoadProfile MakeProfile(ProfileKind kind, real magnitude, double start, double end) {
  LoadProfile profile;
  profile.kind = kind;
  profile.magnitude = magnitude;
  profile.start = start;
  profile.end = end;
  return profile;
}

TEST(LoadMath, ConstantIsActiveOnlyInsideItsWindow) {
  LoadProfile const profile = MakeProfile(ProfileKind::Constant, 3, 1.0, 2.0);
  EXPECT_EQ(EvaluateProfile(profile, 0.99), 0);
  EXPECT_EQ(EvaluateProfile(profile, 1.0), 3);
  EXPECT_EQ(EvaluateProfile(profile, 1.99), 3);
  EXPECT_EQ(EvaluateProfile(profile, 2.0), 0);
}

TEST(LoadMath, EndBeforeStartMeansOpenEnded) {
  LoadProfile const profile = MakeProfile(ProfileKind::Constant, 3, 1.0, 0.0);
  EXPECT_EQ(EvaluateProfile(profile, 1000.0), 3);
}

TEST(LoadMath, EnvelopeWithoutEndRisesLinearlyThenHolds) {
  LoadProfile profile = MakeProfile(ProfileKind::Envelope, 10, 1.0, 0.0);
  profile.riseTime = 2.0;
  profile.fallTime = 1.0; // ignored without an end
  EXPECT_FLOAT_EQ(EvaluateProfile(profile, 1.0), 0);
  EXPECT_FLOAT_EQ(EvaluateProfile(profile, 2.0), 5);
  EXPECT_FLOAT_EQ(EvaluateProfile(profile, 5.0), 10);
}

TEST(LoadMath, EnvelopeRisesHoldsAndFallsToZeroAtEnd) {
  LoadProfile profile = MakeProfile(ProfileKind::Envelope, 10, 1.0, 6.0);
  profile.riseTime = 2.0;
  profile.fallTime = 2.0;
  EXPECT_FLOAT_EQ(EvaluateProfile(profile, 2.0), 5); // rising
  EXPECT_FLOAT_EQ(EvaluateProfile(profile, 3.5), 10); // holding
  EXPECT_FLOAT_EQ(EvaluateProfile(profile, 5.0), 5); // falling
  EXPECT_FLOAT_EQ(EvaluateProfile(profile, 6.0), 0); // ended
}

TEST(LoadMath, EnvelopeWithOverlappingRiseAndFallIsATriangle) {
  LoadProfile profile = MakeProfile(ProfileKind::Envelope, 10, 0.0, 2.0);
  profile.riseTime = 2.0;
  profile.fallTime = 2.0;
  EXPECT_FLOAT_EQ(EvaluateProfile(profile, 1.0), 5);
}

TEST(LoadMath, SineStartsAtPhaseFromStartTime) {
  LoadProfile profile = MakeProfile(ProfileKind::Sine, 2, 1.0, 0.0);
  profile.offset = 1;
  profile.freqHz = 0.25;
  EXPECT_NEAR(EvaluateProfile(profile, 1.0), 1.0, 1e-5);
  EXPECT_NEAR(EvaluateProfile(profile, 2.0), 3.0, 1e-5);
}

LoadProfile MakeKeyframes(std::vector<ProfileKey> keys) {
  LoadProfile profile;
  profile.kind = ProfileKind::Keyframes;
  profile.keys.assign(keys.begin(), keys.end());
  return profile;
}

TEST(LoadMath, KeyframesInterpolateBetweenKeysAndHoldOutside) {
  LoadProfile const profile = MakeKeyframes({
      {1.0, 2, Interpolation::Linear},
      {3.0, 6, Interpolation::Step},
      {4.0, 0, Interpolation::Linear},
  });
  EXPECT_FLOAT_EQ(EvaluateProfile(profile, 0.0), 2); // before the first key
  EXPECT_FLOAT_EQ(EvaluateProfile(profile, 2.0), 4); // linear
  EXPECT_FLOAT_EQ(EvaluateProfile(profile, 3.9), 6); // step holds until the next key
  EXPECT_FLOAT_EQ(EvaluateProfile(profile, 9.0), 0); // after the last key
  EXPECT_EQ(EvaluateProfile(MakeKeyframes({}), 1.0), 0);
}

TEST(LoadMath, SmoothKeyframesStartAndEndAtRest) {
  // Two smooth keys: a smoothstep from 0 to 4, 3s^2 - 2s^3.
  LoadProfile const profile =
      MakeKeyframes({{0.0, 0, Interpolation::Smooth}, {2.0, 4, Interpolation::Smooth}});
  EXPECT_NEAR(EvaluateProfile(profile, 0.5), 4 * 0.15625, 1e-5);
  EXPECT_NEAR(EvaluateProfile(profile, 1.0), 2, 1e-5);
  EXPECT_NEAR(EvaluateProfile(profile, 1.999), 4, 1e-4);
}

ProfileKey MakeDirectionKey(double time, Real3 direction, Interpolation interpolation) {
  return {time, 1, interpolation, direction, Real3{0.0f, 0.0f, 0.0f}};
}

LoadSpec MakeAnimatedLoad(LoadType type, std::vector<ProfileKey> keys) {
  LoadSpec load;
  load.type = type;
  load.direction = {0.0f, 0.0f, 1.0f};
  load.pointLocal = {0.5f, 0.5f, 0.5f};
  load.profile = MakeKeyframes(std::move(keys));
  load.profile.animateDirection = true;
  load.profile.animatePoint = true;
  return load;
}

TEST(LoadMath, UnanimatedLoadsUseTheirOwnDirectionAndPoint) {
  LoadSpec load = MakeAnimatedLoad(
      LoadType::Force, {MakeDirectionKey(0.0, {1.0f, 0.0f, 0.0f}, Interpolation::Linear)});
  load.profile.animateDirection = false;
  load.profile.animatePoint = false;
  EXPECT_EQ(EvaluateLoadDirection(load, 0.5), load.direction);
  EXPECT_EQ(EvaluateLoadPoint(load, 0.5), load.pointLocal);

  // A profile other than Keyframes ignores the switches.
  load.profile.animateDirection = true;
  load.profile.kind = ProfileKind::Constant;
  EXPECT_EQ(EvaluateLoadDirection(load, 0.5), load.direction);
  // A WorldPoint force's own point is pointWorld.
  load.frame = LoadFrame::WorldPoint;
  load.pointWorld = {1.0f, 2.0f, 3.0f};
  EXPECT_EQ(EvaluateLoadPoint(load, 0.5), load.pointWorld);
}

TEST(LoadMath, LinearDirectionKeysTurnAtConstantSpeed) {
  LoadSpec const load = MakeAnimatedLoad(
      LoadType::Force,
      {MakeDirectionKey(0.0, {2.0f, 0.0f, 0.0f}, Interpolation::Linear),
       MakeDirectionKey(1.0, {0.0f, 3.0f, 0.0f}, Interpolation::Linear)});
  real const inv = 1 / std::numbers::sqrt2_v<real>;
  EXPECT_NEAR(mochi::Norm(EvaluateLoadDirection(load, 0.5) - Real3{inv, inv, 0.0f}), 0, 1e-5);
  // A third of the way: a third of the quarter turn.
  Real3 const third = EvaluateLoadDirection(load, 1.0 / 3.0);
  EXPECT_NEAR(std::atan2(third[1], third[0]), std::numbers::pi_v<real> / 6, 1e-5);
  EXPECT_NEAR(mochi::Norm(third), 1, 1e-5);
}

TEST(LoadMath, OppositeDirectionKeysTurnWithoutPassingThroughZero) {
  LoadSpec const load = MakeAnimatedLoad(
      LoadType::Torque,
      {MakeDirectionKey(0.0, {1.0f, 0.0f, 0.0f}, Interpolation::Linear),
       MakeDirectionKey(1.0, {-1.0f, 0.0f, 0.0f}, Interpolation::Linear)});
  for (double t : {0.0, 0.25, 0.5, 0.75, 1.0}) {
    EXPECT_NEAR(mochi::Norm(EvaluateLoadDirection(load, t)), 1, 1e-5) << "t = " << t;
  }
  EXPECT_NEAR(EvaluateLoadDirection(load, 1.0)[0], -1, 1e-5);
}

TEST(LoadMath, StepAndSmoothDirectionKeysPassThroughEveryKey) {
  std::vector<Real3> const directions{{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
  for (Interpolation const interpolation : {Interpolation::Step, Interpolation::Smooth}) {
    std::vector<ProfileKey> keys;
    for (size_t k = 0; k < directions.size(); ++k) {
      keys.push_back(MakeDirectionKey(static_cast<double>(k), directions[k], interpolation));
    }
    LoadSpec const load = MakeAnimatedLoad(LoadType::Force, keys);
    for (size_t k = 0; k < directions.size(); ++k) {
      Real3 const atKey = EvaluateLoadDirection(load, static_cast<double>(k));
      EXPECT_NEAR(mochi::Norm(atKey - directions[k]), 0, 1e-5);
    }
  }
  LoadSpec const step = MakeAnimatedLoad(
      LoadType::Force,
      {MakeDirectionKey(0.0, directions[0], Interpolation::Step),
       MakeDirectionKey(1.0, directions[1], Interpolation::Step)});
  EXPECT_EQ(EvaluateLoadDirection(step, 0.9), directions[0]);
}

TEST(LoadMath, PointKeysAndJointEffortWeightsBlendAsValues) {
  std::vector<ProfileKey> keys{
      {0.0, 1, Interpolation::Linear, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
      {1.0, 1, Interpolation::Linear, {3.0f, 0.0f, 0.0f}, {2.0f, 4.0f, 0.0f}},
  };
  LoadSpec const force = MakeAnimatedLoad(LoadType::Force, keys);
  EXPECT_EQ(EvaluateLoadPoint(force, 0.5), (Real3{1.0f, 2.0f, 0.0f}));

  // Weights are not directions: blended and left unnormalized.
  LoadSpec const effort = MakeAnimatedLoad(LoadType::JointEffort, keys);
  EXPECT_EQ(EvaluateLoadDirection(effort, 0.5), (Real3{2.0f, 0.0f, 0.0f}));
  // Only forces have an application point.
  LoadSpec const torque = MakeAnimatedLoad(LoadType::Torque, keys);
  EXPECT_EQ(EvaluateLoadPoint(torque, 0.5), torque.pointLocal);
}

TEST(LoadMath, ShortestRotationMapsFromOntoTo) {
  std::vector<std::pair<Real3, Real3>> const pairs{
      {{1.0f, 0.0f, 0.0f}, {0.0f, 2.0f, 0.0f}},
      {{1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, -1.0f}},
      {{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 5.0f}}, // parallel
      {{1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}}, // opposite
      {{0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}}, // opposite, other helper axis
  };
  for (auto const& [from, to] : pairs) {
    Real3 const mapped = ShortestRotation(from, to) * mochi::Normalize(from);
    EXPECT_NEAR(mochi::Norm(mapped - mochi::Normalize(to)), 0, 1e-5);
  }
}

PoseWaypoint MakeWaypoint(double time, Real3 position, real angleZ, Interpolation interpolation) {
  return {time, position, Real3{0.0f, 0.0f, angleZ}, interpolation};
}

real AngleZ(mochi::TransformRT const& pose) {
  return pose.GetRotation().ToRotationVector()[2];
}

TEST(LoadMath, PosePathPassesThroughWaypointsAndHoldsOutside) {
  real const quarterTurn = std::numbers::pi_v<real> / 2;
  for (Interpolation const interpolation :
       {Interpolation::Step, Interpolation::Linear, Interpolation::Smooth, Interpolation::Ease}) {
    std::vector<PoseWaypoint> const path{
        MakeWaypoint(1.0, {0.0f, 0.0f, 0.0f}, 0, interpolation),
        MakeWaypoint(2.0, {1.0f, 0.0f, 0.0f}, quarterTurn, interpolation),
        MakeWaypoint(3.0, {1.0f, 2.0f, 0.0f}, 0, interpolation),
    };
    for (PoseWaypoint const& waypoint : path) {
      mochi::TransformRT const pose = EvaluatePosePath(path, waypoint.time);
      EXPECT_NEAR(mochi::Norm(pose.GetTranslation() - waypoint.position), 0, 1e-5);
      EXPECT_NEAR(AngleZ(pose), waypoint.rotation[2], 1e-5);
    }
    EXPECT_NEAR(mochi::Norm(EvaluatePosePath(path, 0.0).GetTranslation()), 0, 1e-6);
    EXPECT_NEAR(
        mochi::Norm(EvaluatePosePath(path, 5.0).GetTranslation() - path.back().position), 0, 1e-6);
  }
}

TEST(LoadMath, PosePathSegmentsFollowTheirInterpolation) {
  real const quarterTurn = std::numbers::pi_v<real> / 2;
  auto const midpoint = [&](Interpolation interpolation) {
    std::vector<PoseWaypoint> const path{
        MakeWaypoint(0.0, {0.0f, 0.0f, 0.0f}, 0, interpolation),
        MakeWaypoint(1.0, {2.0f, 0.0f, 0.0f}, quarterTurn, interpolation),
    };
    return EvaluatePosePath(path, 0.25);
  };
  mochi::TransformRT const step = midpoint(Interpolation::Step);
  EXPECT_NEAR(step.GetTranslation()[0], 0, 1e-6);
  EXPECT_NEAR(AngleZ(step), 0, 1e-6);
  mochi::TransformRT const linear = midpoint(Interpolation::Linear);
  EXPECT_NEAR(linear.GetTranslation()[0], 0.5, 1e-5);
  EXPECT_NEAR(AngleZ(linear), 0.25f * quarterTurn, 1e-5);
  // Smooth with only two waypoints eases in and out (smoothstep: 0.15625 at a quarter).
  mochi::TransformRT const smooth = midpoint(Interpolation::Smooth);
  EXPECT_NEAR(smooth.GetTranslation()[0], 2 * 0.15625, 1e-5);
  EXPECT_NEAR(AngleZ(smooth), 0.15625f * quarterTurn, 1e-5);
  mochi::TransformRT const ease = midpoint(Interpolation::Ease);
  EXPECT_NEAR(ease.GetTranslation()[0], 2 * 0.15625, 1e-5);
  EXPECT_NEAR(AngleZ(ease), 0.15625f * quarterTurn, 1e-5);
}

TEST(LoadMath, EaseStopsAtEveryWaypointWhileSmoothPassesThrough) {
  auto const speedAtMiddle = [](Interpolation interpolation) {
    std::vector<PoseWaypoint> const path{
        MakeWaypoint(0.0, {0.0f, 0.0f, 0.0f}, 0, interpolation),
        MakeWaypoint(1.0, {1.0f, 0.0f, 0.0f}, 0, interpolation),
        MakeWaypoint(2.0, {2.0f, 0.0f, 0.0f}, 0, interpolation),
    };
    constexpr double kDt = 1e-3;
    return (EvaluatePosePath(path, 1.0 + kDt).GetTranslation()[0] -
            EvaluatePosePath(path, 1.0 - kDt).GetTranslation()[0]) /
        (2 * kDt);
  };
  EXPECT_NEAR(speedAtMiddle(Interpolation::Ease), 0, 1e-2);
  EXPECT_NEAR(speedAtMiddle(Interpolation::Smooth), 1, 1e-2);
}

TEST(LoadMath, InputKeysInterpolateEveryValue) {
  std::vector<InputKey> const keys{
      {0.0, {0.0f, 10.0f}, Interpolation::Linear},
      {1.0, {2.0f, 20.0f}, Interpolation::Step},
      {2.0, {4.0f, 0.0f}, Interpolation::Linear},
  };
  std::array<real, 2> values{};

  EvaluateInputKeys(keys, 0.5, values);
  EXPECT_EQ(values, (std::array<real, 2>{1.0f, 15.0f}));
  EvaluateInputKeys(keys, 1.5, values);
  EXPECT_EQ(values, (std::array<real, 2>{2.0f, 20.0f}));
  EvaluateInputKeys(keys, 3.0, values);
  EXPECT_EQ(values, (std::array<real, 2>{4.0f, 0.0f}));
}

TEST(LoadMath, PathDriveHoldsFromEachWaypointUntilTheNext) {
  std::vector<PoseWaypoint> path{
      MakeWaypoint(1.0, {0.0f, 0.0f, 0.0f}, 0, Interpolation::Linear),
      MakeWaypoint(2.0, {1.0f, 0.0f, 0.0f}, 0, Interpolation::Linear),
      MakeWaypoint(3.0, {2.0f, 0.0f, 0.0f}, 0, Interpolation::Linear),
  };
  path[0].drive = PathDriveMode::PositionOnly;
  path[1].drive = PathDriveMode::Off;
  path[2].drive = PathDriveMode::On;
  EXPECT_EQ(EvaluatePathDrive(path, 0.0), PathDriveMode::PositionOnly); // before the first
  EXPECT_EQ(EvaluatePathDrive(path, 1.5), PathDriveMode::PositionOnly);
  EXPECT_EQ(EvaluatePathDrive(path, 2.0), PathDriveMode::Off); // from the waypoint's own time
  EXPECT_EQ(EvaluatePathDrive(path, 2.9), PathDriveMode::Off);
  EXPECT_EQ(EvaluatePathDrive(path, 9.0), PathDriveMode::On); // after the last
}

TEST(LoadMath, DriveScaleUsesTheArticulationsMassForALink) {
  DriveScale const rigid = MakeDriveScale(2, 0.1f, 0);
  DriveScale const link = MakeDriveScale(2, 0.1f, 8);
  DriveScale const massless = MakeDriveScale(0, 0, 0);

  EXPECT_FLOAT_EQ(rigid.mass, 2);
  EXPECT_FLOAT_EQ(rigid.inertia, 0.1f);
  EXPECT_FLOAT_EQ(link.mass, 8);
  EXPECT_FLOAT_EQ(link.inertia, 0.4f);
  EXPECT_FLOAT_EQ(massless.mass, 1);
}

// A pose path along `waypoints` with fixed drive gains.
LoadSpec MakeDriveLoad(std::vector<PoseWaypoint> waypoints) {
  LoadSpec load;
  load.type = LoadType::PosePath;
  load.waypoints.assign(waypoints.begin(), waypoints.end());
  load.drive.responseHz = 10;
  load.drive.dampingRatio = 1;
  load.drive.maxAcceleration = 30;
  load.drive.rotResponseHz = 4;
  load.drive.rotDampingRatio = 1;
  load.drive.maxAngularAcceleration = 20;
  return load;
}

TEST(LoadMath, DrivePullIsTheSpringCapPlusTheDamperAtTheFastestDrivenPathSpeed) {
  // With the drive on: 0.1 m and an eighth turn in 0.5 s. Then with the drive off: 1 m in 0.1 s,
  // which should not count.
  real const eighthTurn = std::numbers::pi_v<real> / 4;
  std::vector<PoseWaypoint> path{
      MakeWaypoint(0.0, {0.0f, 0.0f, 0.0f}, 0, Interpolation::Linear),
      MakeWaypoint(0.5, {0.1f, 0.0f, 0.0f}, eighthTurn, Interpolation::Linear),
      MakeWaypoint(0.6, {1.1f, 0.0f, 0.0f}, eighthTurn, Interpolation::Linear),
  };
  path[1].drive = PathDriveMode::Off;
  path[2].drive = PathDriveMode::On;
  LoadSpec load = MakeDriveLoad(path);
  DriveScale const scale{.mass = 2, .inertia = 0.01f};
  real const omega = 2 * std::numbers::pi_v<real> * 10;
  real const rotOmega = 2 * std::numbers::pi_v<real> * 4;

  DrivePullEstimate const pull = EstimateDrivePull(load, scale, 0.01);

  EXPECT_FLOAT_EQ(pull.springForce, 2 * 30);
  EXPECT_FLOAT_EQ(pull.capLag, 30 / (omega * omega));
  EXPECT_NEAR(pull.peakVelocity, 0.2f, 1e-4f);
  EXPECT_NEAR(pull.damperForce, 2 * 2 * omega * 0.2f, 1e-2f);
  EXPECT_FLOAT_EQ(pull.springTorque, 0.01f * 20);
  EXPECT_FLOAT_EQ(pull.capAngle, 20 / (rotOmega * rotOmega));
  EXPECT_NEAR(pull.peakAngularVelocity, eighthTurn / 0.5f, 1e-4f);
  EXPECT_NEAR(pull.damperTorque, 2 * 0.01f * rotOmega * (eighthTurn / 0.5f), 1e-4f);

  load.drive.rotation = false;
  load.drive.maxAcceleration = 0;
  DrivePullEstimate const uncapped = EstimateDrivePull(load, scale, 0.01);

  EXPECT_LT(uncapped.springForce, 0);
  EXPECT_LT(uncapped.capLag, 0);
  EXPECT_EQ(uncapped.springTorque, 0);
  EXPECT_EQ(uncapped.damperTorque, 0);
}

TEST(LoadMath, DrivePullSeesAStepJumpWithinOneStepAndASmoothSegmentsPeak) {
  DriveScale const scale{.mass = 1, .inertia = 0.01f};
  LoadSpec const step = MakeDriveLoad({
      MakeWaypoint(0.0, {0.0f, 0.0f, 0.0f}, 0, Interpolation::Step),
      MakeWaypoint(1.0, {0.05f, 0.0f, 0.0f}, 0, Interpolation::Step),
  });
  // Smoothstep over 1 m in 1 s peaks at 1.5 m/s halfway.
  LoadSpec const smooth = MakeDriveLoad({
      MakeWaypoint(0.0, {0.0f, 0.0f, 0.0f}, 0, Interpolation::Smooth),
      MakeWaypoint(1.0, {1.0f, 0.0f, 0.0f}, 0, Interpolation::Smooth),
  });

  EXPECT_NEAR(EstimateDrivePull(step, scale, 0.01).peakVelocity, 5.0f, 1e-3f);
  EXPECT_NEAR(EstimateDrivePull(smooth, scale, 0.01).peakVelocity, 1.5f, 0.015f);
}

TEST(LoadMath, SortByTimeOrdersWaypointsAndKeys) {
  LoadSpec load;
  load.waypoints.push_back(MakeWaypoint(2.0, {2.0f, 0.0f, 0.0f}, 0, Interpolation::Linear));
  load.waypoints.push_back(MakeWaypoint(1.0, {1.0f, 0.0f, 0.0f}, 0, Interpolation::Linear));
  load.profile.keys.push_back({3.0, 3, Interpolation::Linear});
  load.profile.keys.push_back({0.0, 0, Interpolation::Linear});

  SortByTime(load);

  EXPECT_EQ(load.waypoints[0].time, 1.0);
  EXPECT_EQ(load.profile.keys[0].time, 0.0);
}

TEST(LoadMath, PointForceAddsTorqueAboutCenterOfMass) {
  auto const dofs =
      RigidPointForceToDofs({2.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f});
  std::array<real, 6> const expected{2, 0, 0, 0, 0, -2};
  EXPECT_EQ(dofs, expected);
}

TEST(LoadMath, LinkForceIsJacobianTransposeTimesWrench) {
  // Two DoFs: DoF 0 translates along x, DoF 1 rotates about z.
  std::vector<real> const jacobian{
      1,
      0, // vx
      0,
      0, // vy
      0,
      0, // vz
      0,
      0, // wx
      0,
      0, // wy
      0,
      1, // wz
  };
  std::vector<real> generalized{0.5f, 0};
  // Force +y at 1 m along +x from the reference point: torque +z of 3 N m.
  AddLinkForceToGeneralized(
      jacobian,
      {0.0f, 3.0f, 0.0f},
      {1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 0.0f},
      mochi::MakeSpan(generalized));
  std::vector<real> const expected{0.5f, 3};
  EXPECT_EQ(generalized, expected);
}

TEST(LoadMath, LinkTorqueUsesOnlyTheJacobiansRotationRows) {
  // Two DoFs: DoF 0 translates along x, DoF 1 rotates about z.
  std::vector<real> const jacobian{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  std::vector<real> generalized{0.5f, 0};

  AddLinkTorqueToGeneralized(jacobian, {0.0f, 0.0f, 2.0f}, mochi::MakeSpan(generalized));

  std::vector<real> const expected{0.5f, 2};
  EXPECT_EQ(generalized, expected);
}

TEST(JsonPath, SetsNestedKeysAndCreatesMissingObjects) {
  picojson::value root = ParseJsonOrString(R"({"a": {"b": 1}})");
  std::string error;
  ASSERT_TRUE(SetJsonPath(root, "a.b", ParseJsonOrString("2"), error)) << error;
  ASSERT_TRUE(SetJsonPath(root, "x.y", ParseJsonOrString("true"), error)) << error;
  EXPECT_EQ(root.serialize(), R"({"a":{"b":2},"x":{"y":true}})");
}

TEST(JsonPath, SelectsArrayElementsByIndexAndName) {
  picojson::value root =
      ParseJsonOrString(R"({"list": [{"name": "a", "v": 0}, {"name": "b.c", "v": 0}]})");
  std::string error;
  ASSERT_TRUE(SetJsonPath(root, "list[0].v", ParseJsonOrString("1"), error)) << error;
  ASSERT_TRUE(SetJsonPath(root, "list[name=b.c].v", ParseJsonOrString("2"), error)) << error;
  EXPECT_EQ(root.serialize(), R"({"list":[{"name":"a","v":1},{"name":"b.c","v":2}]})");
}

TEST(JsonPath, ReportsMissingArrayElements) {
  picojson::value root = ParseJsonOrString(R"({"list": [{"name": "a"}]})");
  std::string error;
  EXPECT_FALSE(SetJsonPath(root, "list[name=zzz].v", ParseJsonOrString("1"), error));
  EXPECT_NE(error.find("zzz"), std::string::npos);
  EXPECT_FALSE(SetJsonPath(root, "list[3].v", ParseJsonOrString("1"), error));
}

TEST(JsonPath, NonJsonValuesBecomeStrings) {
  EXPECT_EQ(ParseJsonOrString("foo/bar.mochi_scene").serialize(), R"("foo/bar.mochi_scene")");
  EXPECT_EQ(ParseJsonOrString("[1,2]").serialize(), "[1,2]");
}

TEST(JsonPath, TextOverridesKeepExistingStringsAsStrings) {
  picojson::value root = ParseJsonOrString(R"({"s": "a", "n": 1})");
  std::string error;
  ASSERT_TRUE(SetJsonPathFromText(root, "s", "0.8", error)) << error;
  ASSERT_TRUE(SetJsonPathFromText(root, "n", "0.5", error)) << error;
  EXPECT_EQ(root.serialize(), R"({"n":0.5,"s":"0.8"})");
}

TEST(JsonPath, AssignmentsSplitAtTheFirstEqualsOutsideSelectors) {
  struct Case {
    std::string_view assignment;
    std::string path;
    std::string value;
  };
  std::vector<Case> const cases{
      {"run.dt=0.005", "run.dt", "0.005"},
      {"loads[name=push].profile.magnitude=3", "loads[name=push].profile.magnitude", "3"},
      {"prefabOverrides[0].value=a=b", "prefabOverrides[0].value", "a=b"},
  };
  for (Case const& c : cases) {
    std::string path;
    std::string value;
    ASSERT_TRUE(SplitAssignment(c.assignment, path, value)) << c.assignment;
    EXPECT_EQ(path, c.path);
    EXPECT_EQ(value, c.value);
  }
  std::string path;
  std::string value;
  EXPECT_FALSE(SplitAssignment("loads[name=push]", path, value)) << "no '=' outside brackets";
  EXPECT_FALSE(SplitAssignment("=3", path, value)) << "empty path";
}

} // namespace
} // namespace superdex::tuner
