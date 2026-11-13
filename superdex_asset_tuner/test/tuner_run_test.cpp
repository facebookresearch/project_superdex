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

#include "tuner_test_fixture.h"

#include "superdex_asset_tuner/sim_recording.h"
#include "superdex_asset_tuner/tuner_spec.h"

#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/math_utils.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numbers>
#include <sstream>

namespace superdex::tuner::test {
namespace {

std::string ReadFile(std::filesystem::path const& path) {
  std::ifstream file(path);
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

real RigidMass(SceneTargets const& targets, std::string_view name) {
  for (RigidTarget const& rigid : targets.rigid) {
    if (std::string_view(rigid.name) == name) {
      return rigid.mass;
    }
  }
  ADD_FAILURE() << "No rigid actor '" << name << "'";
  return 0;
}

TEST_F(TunerTest, ConstantForceAcceleratesRigidBodyAtForceOverMass) {
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  real const force = 2;
  // The cube spans [0, 0.1]^3 locally, so its COM is at 0.05 on each axis.
  LoadSpec load = MakeLoad("push", LoadType::Force, "cube", {1.0f, 0.0f, 0.0f}, force);
  load.pointLocal = {0.05f, 0.05f, 0.05f};
  spec.loads.push_back(load);
  spec.probes.push_back(MakeProbe(ProbeKind::BodyPose, "cube"));
  real const mass = RigidMass(Describe(spec), "cube");
  ASSERT_GT(mass, 0);

  RunCapture const run = Run(spec);

  ASSERT_EQ(run.summary.status, "ok");
  ASSERT_EQ(run.samples.size(), 100u);
  double const expected = ConstantAccelerationDisplacement(force / mass, spec.run.duration);
  EXPECT_NEAR(run.Final("cube.pos.x"), expected, 0.03 * expected);
  EXPECT_NEAR(run.Final("cube.rotvec.z"), 0, 1e-2) << "A force at the COM must not rotate the body";
}

TEST_F(TunerTest, ForceAboveCenterOfMassRotatesByRightHandRule) {
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  // The cube spans [0, 0.1]^3 locally; push +x at the top face center (COM is at 0.05).
  LoadSpec load = MakeLoad("push", LoadType::Force, "cube", {1.0f, 0.0f, 0.0f}, 1);
  load.pointLocal = {0.05f, 0.05f, 0.1f};
  spec.loads.push_back(load);
  spec.probes.push_back(MakeProbe(ProbeKind::BodyPose, "cube"));
  spec.run.duration = 0.2;

  RunCapture const run = Run(spec);

  // r = +z, F = +x  =>  torque = r x F = +y.
  real const rotationY = run.Final("cube.rotvec.y");
  EXPECT_GT(rotationY, 1e-3);
  EXPECT_LT(std::abs(run.Final("cube.rotvec.x")), 0.01 * rotationY);
  EXPECT_LT(std::abs(run.Final("cube.rotvec.z")), 0.01 * rotationY);
}

// Final linear and angular velocity of a free single-link articulation and of a rigid body with
// identical geometry under the same load.
struct BodyResponse {
  Real3 rigidLinear;
  Real3 linkLinear;
  Real3 rigidAngular;
  Real3 linkAngular;
};

class TunerLinkVsRigidTest : public TunerTest {
 protected:
  BodyResponse Respond(LoadSpec load) {
    auto const runBody = [&](char const* prefab, char const* target) {
      load.target = target;
      TunerSpec spec = MakeSpec(prefab);
      spec.loads.push_back(load);
      spec.probes.push_back(MakeProbe(ProbeKind::BodyVelocity, target));
      spec.run.duration = 0.05;
      return Run(spec);
    };
    RunCapture const rigid = runBody(kRigidCubePrefab, "cube");
    RunCapture const link = runBody(kFreeLinkPrefab, "arm/body");
    auto const velocity = [](RunCapture const& run, std::string const& prefix, char const* kind) {
      return Real3{
          run.Final(prefix + kind + ".x"),
          run.Final(prefix + kind + ".y"),
          run.Final(prefix + kind + ".z")};
    };
    return {
        .rigidLinear = velocity(rigid, "cube.", "vel"),
        .linkLinear = velocity(link, "arm/body.", "vel"),
        .rigidAngular = velocity(rigid, "cube.", "angvel"),
        .linkAngular = velocity(link, "arm/body.", "angvel")};
  }

  // Same rotation axis; magnitudes loosely, since rigid and link inertias differ by ~1%.
  static void ExpectSameAngular(BodyResponse const& r) {
    ASSERT_GT(mochi::Norm(r.rigidAngular), 0.5f) << "Test load must produce rotation";
    real const cosine = mochi::Dot(r.linkAngular, r.rigidAngular) /
        (mochi::Norm(r.linkAngular) * mochi::Norm(r.rigidAngular));
    EXPECT_GT(cosine, 0.9999f);
    EXPECT_NEAR(
        mochi::Norm(r.linkAngular),
        mochi::Norm(r.rigidAngular),
        0.03f * mochi::Norm(r.rigidAngular));
  }
};

TEST_F(TunerLinkVsRigidTest, ForceOnALinkMatchesTheSameForceOnARigidBody) {
  // An off-COM point force. A wrong Jacobian reference point would change the torque direction.
  LoadSpec load = MakeLoad("push", LoadType::Force, "", {1.0f, 0.0f, 1.0f}, 1);
  load.pointLocal = {0.0f, 0.1f, 0.03f};

  BodyResponse const r = Respond(load);

  EXPECT_NEAR(mochi::Norm(r.linkLinear - r.rigidLinear), 0, 1e-3 * mochi::Norm(r.rigidLinear));
  ExpectSameAngular(r);
}

TEST_F(TunerLinkVsRigidTest, TorqueOnALinkMatchesTheSameTorqueOnARigidBody) {
  BodyResponse const r = Respond(MakeLoad("spin", LoadType::Torque, "", {1.0f, 0.0f, 1.0f}, 0.05f));

  EXPECT_NEAR(mochi::Norm(r.linkLinear - r.rigidLinear), 0, 1e-3f);
  ExpectSameAngular(r);
}

TEST_F(TunerTest, JointEffortOnPrismaticJointAcceleratesCarriage) {
  TunerSpec spec = MakeSpec(kSliderPrefab);
  real const force = 0.5f;
  spec.loads.push_back(
      MakeLoad("drive", LoadType::JointEffort, "slider/slide", {1.0f, 0.0f, 0.0f}, force));
  spec.probes.push_back(MakeProbe(ProbeKind::JointState, "slider/slide"));
  SceneTargets const targets = Describe(spec);
  ASSERT_EQ(targets.articulated.size(), 1u);
  ASSERT_EQ(targets.articulated[0].numDofs, 1);

  RunCapture const run = Run(spec);

  // Carriage mass: 0.1^3 m^3 at 1000 kg/m^3.
  double const expected = ConstantAccelerationDisplacement(force / 1.0, spec.run.duration);
  EXPECT_NEAR(run.Final("slider/slide.q0"), expected, 0.03 * expected);
}

// A load of `magnitude` on the free 1 kg cube's center of mass for 1 s, keyed Linear from one
// direction to another.
LoadSpec MakeTurningLoad(LoadType type, real magnitude, Real3 const& from, Real3 const& to) {
  LoadSpec load = MakeLoad("turn", type, "cube", from, magnitude);
  load.pointLocal = {0.05f, 0.05f, 0.05f};
  load.profile.kind = ProfileKind::Keyframes;
  load.profile.animateDirection = true;
  load.profile.keys.push_back({0.0, magnitude, Interpolation::Linear, from, {0.0f, 0.0f, 0.0f}});
  load.profile.keys.push_back({1.0, magnitude, Interpolation::Linear, to, {0.0f, 0.0f, 0.0f}});
  return load;
}

TEST_F(TunerTest, AnimatedForceDirectionTurnsTheImpulse) {
  // Turning +X to +Y at constant speed over 1 s delivers (2/pi, 2/pi, 0) N s to a 1 kg cube.
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  spec.loads.push_back(MakeTurningLoad(LoadType::Force, 1, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}));
  spec.probes.push_back(MakeProbe(ProbeKind::BodyVelocity, "cube"));

  RunCapture const run = Run(spec);

  real const expected = 2 / std::numbers::pi_v<real>;
  EXPECT_NEAR(run.Final("cube.vel.x"), expected, 0.03f * expected);
  EXPECT_NEAR(run.Final("cube.vel.y"), expected, 0.03f * expected);
  EXPECT_NEAR(run.Final("cube.vel.z"), 0, 1e-4f);
  EXPECT_NEAR(run.Final("cube.angvel.z"), 0, 1e-3f);
}

TEST_F(TunerTest, AnimatedTorqueDirectionTurnsTheSpinAxis) {
  // Turning +Z to +X spins the (isotropic) cube equally about z and x, and not about y. Small, as
  // the 0.1 m cube's inertia is only ~1.7e-3 kg m^2.
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  spec.loads.push_back(
      MakeTurningLoad(LoadType::Torque, 1e-3f, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}));
  spec.probes.push_back(MakeProbe(ProbeKind::BodyVelocity, "cube"));

  RunCapture const run = Run(spec);

  real const spinX = run.Final("cube.angvel.x");
  ASSERT_GT(spinX, 0);
  EXPECT_NEAR(run.Final("cube.angvel.z"), spinX, 0.03f * spinX);
  EXPECT_NEAR(run.Final("cube.angvel.y"), 0, 0.01f * spinX);
}

TEST_F(TunerTest, AnimatedWorldPointMovesTheLineOfAction) {
  // A small +X force whose world point slides from 5 cm above the center of mass (in y) to 5 cm
  // below: the spin about z builds one way, then the other, and is back near zero at the end.
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  real const push = 0.01f;
  LoadSpec load = MakeLoad("slide", LoadType::Force, "cube", {1.0f, 0.0f, 0.0f}, push);
  load.frame = LoadFrame::WorldPoint;
  load.profile.kind = ProfileKind::Keyframes;
  load.profile.animatePoint = true;
  load.profile.keys.push_back(
      {0.0, push, Interpolation::Linear, {1.0f, 0.0f, 0.0f}, {0.05f, 0.1f, 0.05f}});
  load.profile.keys.push_back(
      {1.0, push, Interpolation::Linear, {1.0f, 0.0f, 0.0f}, {0.05f, 0.0f, 0.05f}});
  spec.loads.push_back(load);
  spec.probes.push_back(MakeProbe(ProbeKind::BodyVelocity, "cube"));

  RunCapture const run = Run(spec);

  // Sample i ends at (i + 1) dt.
  real const midway = run.Value("cube.angvel.z", 49);
  ASSERT_LT(midway, 0);
  EXPECT_NEAR(run.Final("cube.angvel.z"), 0, 0.1f * std::abs(midway));
}

TEST_F(TunerTest, GroundContactSetsTheGroundPlanesFriction) {
  // A 1 N push on a 1 kg cube resting on the ground: default friction (mu = sqrt(0.5 * 0.5)) holds
  // it, while a frictionless ground (the geometric mean is then 0) lets it slide at 1 m/s^2.
  auto const slide = [&](std::optional<mochi::ContactParams> const& groundContact) {
    TunerSpec spec = MakeSpec(kRigidCubePrefab);
    spec.run.gravity = kDefaultGravity;
    spec.run.groundPlane = true;
    spec.run.groundContact = groundContact;
    spec.run.duration = 0.5;
    LoadSpec push = MakeLoad("push", LoadType::Force, "cube", {1.0f, 0.0f, 0.0f}, 1);
    push.pointLocal = {0.05f, 0.05f, 0.05f};
    spec.loads.push_back(push);
    spec.probes.push_back(MakeProbe(ProbeKind::BodyPose, "cube"));
    RunCapture const run = Run(spec);
    EXPECT_EQ(run.summary.status, "ok");
    return run.Final("cube.pos.x") - run.Value("cube.pos.x", 0);
  };
  mochi::ContactParams frictionless;
  frictionless.coulombFrictionCoefficient = 0;

  double const expected = ConstantAccelerationDisplacement(1.0, 0.5);
  EXPECT_LT(slide(std::nullopt), 0.05 * expected);
  EXPECT_NEAR(slide(frictionless), expected, 0.05 * expected);
}

TEST_F(TunerTest, JointStateReportsEveryDofOfFreeAndHardJoints) {
  // A Free joint has 6 DoFs (translation first); a Hard joint has none, so it adds no channels.
  TunerSpec spec = MakeSpec(kFreeLinkPrefab);
  LoadSpec push = MakeLoad("push", LoadType::Force, "arm/body", {1.0f, 0.0f, 0.0f}, 1);
  push.pointLocal = {0.05f, 0.05f, 0.05f};
  spec.loads.push_back(push);
  spec.probes.push_back(MakeProbe(ProbeKind::JointState, "arm/root"));
  spec.run.duration = 0.2;

  RunCapture const run = Run(spec);

  ASSERT_EQ(run.summary.status, "ok");
  EXPECT_GT(run.Final("arm/root.q0"), 0);
  EXPECT_GT(run.Final("arm/root.qd0"), 0);
  // Pushed through (about) the center of mass, so it barely spins.
  EXPECT_LT(std::abs(run.Final("arm/root.qd5")), 0.01f * run.Final("arm/root.qd0"));

  TunerSpec welded = MakeSpec(kSliderPrefab);
  welded.probes.push_back(MakeProbe(ProbeKind::JointState, "slider/weld"));
  RunCapture const weldRun = Run(welded);
  ASSERT_EQ(weldRun.summary.status, "ok");
  EXPECT_TRUE(std::none_of(weldRun.channels.begin(), weldRun.channels.end(), [](auto const& name) {
    return name.starts_with("slider/weld.");
  }));
}

TEST_F(TunerTest, LoadsAndProbesTargetActorsInNestedPrefabs) {
  // Mochi names nested prefab actors "<reference name>/<actor>".
  std::ofstream(Dir() / "inner.mochi_scene") << kRigidCubePrefab;
  TunerSpec spec = MakeSpec(R"({"prefabs": [{"name": "inner", "path": "./inner.mochi_scene"}]})");
  LoadSpec load = MakeLoad("push", LoadType::Force, "inner/cube", {1.0f, 0.0f, 0.0f}, 1);
  load.pointLocal = {0.05f, 0.05f, 0.05f};
  spec.loads.push_back(load);
  spec.probes.push_back(MakeProbe(ProbeKind::BodyPose, "inner/cube"));
  spec.run.duration = 0.2;

  RunCapture const run = Run(spec);

  EXPECT_GT(run.Final("inner/cube.pos.x"), 0.01f);
}

TEST_F(TunerTest, PrefabOverridesReachIntoOneNestedReference) {
  std::ofstream(Dir() / "inner.mochi_scene") << kRigidCubePrefab;
  TunerSpec spec = MakeSpec(R"({"prefabs": [
      {"name": "a", "path": "./inner.mochi_scene"},
      {"name": "b", "path": "./inner.mochi_scene", "translation": [1, 0, 0]}]})");
  PrefabOverride makeStatic;
  makeStatic.path = "prefabs[name=a].actors.rigid[name=cube].isStatic";
  makeStatic.value = "true";
  spec.prefabOverrides.push_back(makeStatic);
  // A disabled load on the now-static body must not stop the tuner from binding.
  LoadSpec disabled = MakeLoad("off", LoadType::Force, "a/cube", {1.0f, 0.0f, 0.0f}, 1);
  disabled.enabled = false;
  spec.loads.push_back(disabled);

  SceneTargets const targets = Describe(spec);
  RunSummary summary;
  std::string error;
  EXPECT_TRUE(RunTuner(spec, TunerPath(), {}, summary, error)) << error;

  auto const isStatic = [&](std::string_view name) {
    for (RigidTarget const& rigid : targets.rigid) {
      if (std::string_view(rigid.name) == name) {
        return rigid.isStatic;
      }
    }
    ADD_FAILURE() << "No rigid actor '" << name << "'";
    return false;
  };
  EXPECT_TRUE(isStatic("a/cube"));
  EXPECT_FALSE(isStatic("b/cube")) << "Overrides must not leak into other references";
}

TEST_F(TunerTest, AutoGroundHeightCountsArticulatedLinksAndTheActorsAfterThem) {
  // A single-link articulation with its bottom at z = 0.2, then a rigid cube at z = 0.5. The
  // articulated actor has no box of its own (only its link does), which must neither put the
  // ground at z = 0 nor hide the cube created after it.
  TunerSpec spec = MakeSpec(R"({
    "actors": {
      "articulated": [ {
        "name": "arm", "translation": [0, 0, 0.2],
        "joints": [ { "name": "root", "type": "Free" } ],
        "links": [ { "name": "body", "parentLink": -1, "shape": "cube/cube_fine_mesh.mochi.h5",
                     "shapeScale": [0.1, 0.1, 0.1], "colliderType": "Box", "density": 1000 } ] } ],
      "rigid": [ {
        "name": "cube", "shape": "cube/cube_fine_mesh.mochi.h5", "scale": [0.1, 0.1, 0.1],
        "translation": [1, 0, 0.5], "colliderType": "Box", "density": 1000 } ] }
  })");
  spec.run.groundPlane = true;
  EXPECT_NEAR(Describe(spec).groundHeight.value_or(-1), 0.2f, 1e-3f);

  // Without the articulation, the lowest point is the cube's.
  TunerSpec cubeOnly = MakeSpec(R"({
    "actors": { "rigid": [ {
      "name": "cube", "shape": "cube/cube_fine_mesh.mochi.h5", "scale": [0.1, 0.1, 0.1],
      "translation": [1, 0, 0.5], "colliderType": "Box", "density": 1000 } ] }
  })");
  cubeOnly.run.groundPlane = true;
  EXPECT_NEAR(Describe(cubeOnly).groundHeight.value_or(-1), 0.5f, 1e-3f);
}

TEST_F(TunerTest, GroundPlaneDefaultsToPrefabLowestPointAndCanBeOverridden) {
  // A 0.1 m cube authored with its bottom at z = 0.5, under gravity.
  constexpr char kRaisedCube[] = R"({
    "actors": { "rigid": [ {
      "name": "cube", "shape": "cube/cube_fine_mesh.mochi.h5", "scale": [0.1, 0.1, 0.1],
      "translation": [0, 0, 0.5], "colliderType": "Box", "density": 1000 } ] }
  })";
  auto const settledHeight = [&](std::optional<real> groundHeight) {
    TunerSpec spec = MakeSpec(kRaisedCube);
    spec.run.gravity = kDefaultGravity;
    spec.run.groundPlane = true;
    spec.run.groundHeight = groundHeight;
    spec.run.duration = 1.0;
    spec.probes.push_back(MakeProbe(ProbeKind::BodyPose, "cube"));
    return Run(spec).Final("cube.pos.z");
  };

  // Auto: the ground is at the cube's bottom, so it rests where it was authored.
  EXPECT_NEAR(settledHeight(std::nullopt), 0.5f, 0.005f);
  EXPECT_NEAR(Describe(MakeSpec(kRaisedCube)).groundHeight.value_or(-1), -1, 1e-6)
      << "MakeSpec disables the ground plane";
  TunerSpec withGround = MakeSpec(kRaisedCube);
  withGround.run.groundPlane = true;
  EXPECT_NEAR(Describe(withGround).groundHeight.value_or(-1), 0.5f, 1e-3f);
  // Override: it falls onto a plane at z = 0.2.
  EXPECT_NEAR(settledHeight(0.2f), 0.2f, 0.005f);
}

TEST_F(TunerTest, StopConditionsEndTheRunWhenTheyTrigger) {
  // A free-falling cube (no ground) under gravity; after 1 s it has dropped ~4.9 m.
  auto const fallingCube = [&] {
    TunerSpec spec = MakeSpec(kRigidCubePrefab);
    spec.run.gravity = kDefaultGravity;
    spec.run.duration = 1.0;
    spec.probes.push_back(MakeProbe(ProbeKind::BodyPose, "cube"));
    return spec;
  };
  StopCondition box;
  box.name = "escaped";
  box.kind = StopKind::LeavesBox;
  box.target = "cube";
  box.boxMin = {-1.0f, -1.0f, -0.5f};
  box.boxMax = {1.0f, 1.0f, 1.0f};
  StopCondition fell;
  fell.name = "fell";
  fell.kind = StopKind::ChannelThreshold;
  fell.channel = "cube.pos.z";
  fell.comparison = StopComparison::Below;
  fell.threshold = -0.2f;

  // Leaving the box: z = -0.5 is reached after sqrt(2 * 0.5 / 9.81) ~ 0.32 s.
  TunerSpec leaves = fallingCube();
  leaves.stopConditions.push_back(box);
  RunCapture const leftBox = Run(leaves);
  EXPECT_EQ(leftBox.summary.status, "stopped");
  EXPECT_EQ(leftBox.summary.stopCondition, "escaped");
  EXPECT_NEAR(leftBox.summary.simTime, 0.32, 0.02);

  // Crossing a threshold, checked in spec order: "fell" (z < -0.2 at ~0.2 s) triggers first.
  TunerSpec threshold = fallingCube();
  threshold.stopConditions.push_back(box);
  threshold.stopConditions.push_back(fell);
  RunCapture const crossed = Run(threshold);
  EXPECT_EQ(crossed.summary.stopCondition, "fell");
  EXPECT_NEAR(crossed.summary.simTime, 0.2, 0.02);
  EXPECT_LT(crossed.Final("cube.pos.z"), -0.2f);

  // Not checked before its start time, and ignored when disabled.
  TunerSpec delayed = fallingCube();
  fell.start = 0.5;
  delayed.stopConditions.push_back(fell);
  EXPECT_NEAR(Run(delayed).summary.simTime, 0.5, 0.02);
  delayed.stopConditions[0].enabled = false;
  EXPECT_EQ(Run(delayed).summary.status, "ok");
}

TEST_F(TunerTest, EntersBoxStopsWhenTheBodyReachesTheBox) {
  // A free-falling cube (no ground) reaches a box spanning z in [-0.5, -0.3] after
  // sqrt(2 * 0.3 / 9.81) ~ 0.25 s; it starts outside, so it does not trigger at once.
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  spec.run.gravity = kDefaultGravity;
  spec.run.duration = 1.0;
  StopCondition stop;
  stop.name = "landed";
  stop.kind = StopKind::EntersBox;
  stop.target = "cube";
  stop.boxMin = {-1.0f, -1.0f, -0.5f};
  stop.boxMax = {1.0f, 1.0f, -0.3f};
  spec.stopConditions.push_back(stop);

  RunCapture const run = Run(spec);

  EXPECT_EQ(run.summary.status, "stopped");
  EXPECT_EQ(run.summary.stopCondition, "landed");
  EXPECT_NEAR(run.summary.simTime, 0.25, 0.02);
  EXPECT_NE(std::string_view(run.summary.message).find("entered"), std::string_view::npos);
}

TEST_F(TunerTest, ThresholdOnAMissingChannelFailsWithTheAvailableNames) {
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  StopCondition stop;
  stop.name = "fell";
  stop.kind = StopKind::ChannelThreshold;
  stop.channel = "cube.pos.z"; // no BodyPose probe on the cube
  spec.stopConditions.push_back(stop);

  RunSummary summary;
  std::string error;
  EXPECT_FALSE(RunTuner(spec, TunerPath(), {}, summary, error));
  EXPECT_NE(error.find("Stop condition 'fell'"), std::string::npos) << error;
  EXPECT_NE(error.find("Add a probe"), std::string::npos) << error;
}

TEST_F(TunerTest, UnknownTargetListsAvailableNames) {
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  spec.loads.push_back(MakeLoad("push", LoadType::Force, "cub", {1.0f, 0.0f, 0.0f}, 1));
  RunSummary summary;
  std::string error;
  EXPECT_FALSE(RunTuner(spec, TunerPath(), {}, summary, error));
  EXPECT_NE(error.find("'cub'"), std::string::npos) << error;
  EXPECT_NE(error.find("cube"), std::string::npos) << error;
}

TEST_F(TunerTest, ContactProbesOnABodyWithoutContactsFail) {
  // A link without a shape has no contact samples, so Mochi cannot report its contacts.
  constexpr char kShapelessLinkPrefab[] = R"({
  "actors": { "articulated": [ {
    "name": "slider",
    "joints": [
      { "name": "weld", "type": "Hard" },
      { "name": "slide", "type": "Prismatic", "axis": [1, 0, 0] } ],
    "links": [
      { "name": "base", "parentLink": -1, "shape": "cube/cube_fine_mesh.mochi.h5",
        "shapeScale": [0.1, 0.1, 0.1], "density": 1000 },
      { "name": "carriage", "parentLink": 0 } ] } ] }
})";
  for (ProbeKind const kind : {ProbeKind::ContactForce, ProbeKind::Penetration}) {
    TunerSpec spec = MakeSpec(kShapelessLinkPrefab);
    spec.probes.push_back(MakeProbe(kind, "slider/carriage"));
    RunSummary summary;
    std::string error;

    EXPECT_FALSE(RunTuner(spec, TunerPath(), {}, summary, error));
    EXPECT_NE(error.find("'slider/carriage' does not report contact"), std::string::npos) << error;
  }
}

TEST_F(TunerTest, SpecRoundTripsAndAppliesOverrides) {
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  spec.loads.push_back(MakeLoad("push", LoadType::Force, "cube", {1.0f, 0.0f, 0.0f}, 1));
  PlotSpec plot;
  plot.channels.push_back("load.push.mag");
  plot.channels.push_back("cube.pos.x");
  plot.height = 150.0f;
  spec.plots.push_back(plot);
  std::string error;
  ASSERT_TRUE(SaveTunerSpecToFile(spec, TunerPath(), error)) << error;
  std::vector<SpecOverride> const overrides{
      {"run.dt", "0.005"}, {"loads[name=push].profile.magnitude", "7"}};
  TunerSpec loaded;
  ASSERT_TRUE(LoadTunerSpecFromFile(TunerPath(), overrides, loaded, error)) << error;
  EXPECT_EQ(loaded.prefab, spec.prefab);
  EXPECT_EQ(loaded.loads.size(), 1u);
  EXPECT_EQ(loaded.loads[0].type, LoadType::Force);
  EXPECT_EQ(loaded.run.dt, 0.005);
  EXPECT_EQ(loaded.loads[0].profile.magnitude, 7);
  ASSERT_EQ(loaded.plots.size(), 1u);
  EXPECT_EQ(loaded.plots[0].channels, plot.channels);
  EXPECT_EQ(loaded.plots[0].height, 150.0f);
}

TEST_F(TunerTest, RecordSettingsOfSimH5FilesStillLoad) {
  TunerSpec spec;
  std::string error;

  bool const loaded = LoadTunerSpecFromJson(
      R"({"run": {"record": {"enabled": false, "contactPoints": true, "staticActors": true}}})",
      {},
      spec,
      error);

  ASSERT_TRUE(loaded) << error;
  EXPECT_FALSE(spec.run.record.enabled);
  EXPECT_TRUE(spec.run.record.contactForces);
}

TEST_F(TunerTest, PrefabOverrideChangesSimulatedParameters) {
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  PrefabOverride density;
  density.path = "actors.rigid[name=cube].density";
  density.value = "2000";
  spec.prefabOverrides.push_back(density);
  EXPECT_NEAR(RigidMass(Describe(spec), "cube"), 2.0f, 0.01f);
}

TEST_F(TunerTest, RunDirectoryContainsResolvedSpecSignalsAndSummary) {
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  spec.loads.push_back(MakeLoad("push", LoadType::Force, "cube", {1.0f, 0.0f, 0.0f}, 1));
  spec.probes.push_back(MakeProbe(ProbeKind::BodyPose, "cube"));
  spec.run.duration = 0.05;

  RunCapture const run = Run(spec, {.outDir = Dir() / "runs", .tag = "t"});

  std::filesystem::path const runDir(std::string_view(run.summary.runDir));
  ASSERT_TRUE(std::filesystem::is_directory(runDir));
  EXPECT_TRUE(std::filesystem::exists(runDir / "tuner.resolved.json"));
  EXPECT_TRUE(std::filesystem::exists(runDir / "summary.json"));
  EXPECT_EQ(run.summary.simRecording, SimRecording::IsSupported());
  EXPECT_EQ(std::filesystem::exists(runDir / "sim.rrd"), SimRecording::IsSupported());
  std::istringstream csv(ReadFile(runDir / "signals.csv"));
  std::string header;
  std::getline(csv, header);
  EXPECT_EQ(header.rfind("t,load.push.mag,cube.pos.x", 0), 0u) << header;
  int rows = 0;
  for (std::string line; std::getline(csv, line);) {
    ++rows;
  }
  EXPECT_EQ(rows, 5);
}

TEST_F(TunerTest, SingleThreadedRunsAreBitIdentical) {
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  LoadSpec load = MakeLoad("push", LoadType::Force, "cube", {1.0f, 0.3f, 0.0f}, 1);
  load.pointLocal = {0.0f, 0.1f, 0.1f};
  load.profile.kind = ProfileKind::Sine;
  spec.loads.push_back(load);
  spec.probes.push_back(MakeProbe(ProbeKind::BodyPose, "cube"));
  spec.run.duration = 0.3;

  RunCapture const a = Run(spec);
  RunCapture const b = Run(spec);

  ASSERT_EQ(a.samples.size(), b.samples.size());
  for (size_t i = 0; i < a.samples.size(); ++i) {
    ASSERT_EQ(a.samples[i].values, b.samples[i].values) << "step " << i;
  }
}

TEST_F(TunerTest, PenetrationVolumeGrowsWithPressAndIgnoresGroundOnRequest) {
  // A cube resting on the ground, pressed down through its COM; contact overlap grows with load.
  auto const settledVolume = [&](real press, bool ignoreGround) {
    TunerSpec spec = MakeSpec(kRigidCubePrefab);
    spec.run.gravity = kDefaultGravity;
    spec.run.groundPlane = true;
    spec.run.duration = 0.5;
    LoadSpec load = MakeLoad("press", LoadType::Force, "cube", {0.0f, 0.0f, -1.0f}, press);
    load.pointLocal = {0.05f, 0.05f, 0.05f};
    spec.loads.push_back(load);
    ProbeSpec probe = MakeProbe(ProbeKind::Penetration, "cube");
    probe.ignoreGround = ignoreGround;
    spec.probes.push_back(probe);
    return Run(spec).Final("cube.penetration.volume");
  };

  real const resting = settledVolume(0, false);
  real const pressed = settledVolume(500, false);

  EXPECT_GE(resting, 0);
  EXPECT_GT(pressed, resting);
  EXPECT_EQ(settledVolume(500, true), 0);
}

TEST_F(TunerTest, HoldTimeDelaysTheStopUntilTheConditionPersists) {
  // A free-falling cube passes z = -0.2 at ~0.2 s and stays below it.
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  spec.run.gravity = kDefaultGravity;
  spec.run.duration = 1.0;
  spec.probes.push_back(MakeProbe(ProbeKind::BodyPose, "cube"));
  StopCondition fell;
  fell.name = "fell";
  fell.kind = StopKind::ChannelThreshold;
  fell.channel = "cube.pos.z";
  fell.comparison = StopComparison::Below;
  fell.threshold = -0.2f;
  fell.holdTime = 0.3;
  spec.stopConditions.push_back(fell);

  RunCapture const run = Run(spec);

  EXPECT_EQ(run.summary.stopCondition, "fell");
  EXPECT_NEAR(run.summary.simTime, 0.5, 0.02);
  EXPECT_NE(std::string_view(run.summary.message).find("for 0.3 s"), std::string_view::npos)
      << run.summary.message;
}

TEST_F(TunerTest, WorldPointLoadsActAtAFixedWorldPointWhileWorldLoadsRideTheBody) {
  // A free cube pushed along +x through its COM, which starts at (0.05, 0.05, 0.05).
  Real3 const comStart{0.05f, 0.05f, 0.05f};
  auto const finalPoint = [&](LoadFrame frame) {
    TunerSpec spec = MakeSpec(kRigidCubePrefab);
    LoadSpec load = MakeLoad("push", LoadType::Force, "cube", {1.0f, 0.0f, 0.0f}, 2);
    load.frame = frame;
    load.pointLocal = comStart;
    load.pointWorld = comStart;
    spec.loads.push_back(load);
    spec.probes.push_back(MakeProbe(ProbeKind::BodyPose, "cube"));
    spec.run.duration = 0.5;
    RunCapture const run = Run(spec);
    EXPECT_GT(run.Final("cube.pos.x"), 0.1f) << "the push must move the cube";
    return run.samples.back().loads[0].pointWorld;
  };

  Real3 const riding = finalPoint(LoadFrame::World);
  Real3 const fixed = finalPoint(LoadFrame::WorldPoint);

  EXPECT_GT(riding[0], comStart[0] + 0.1f);
  EXPECT_EQ(fixed, comStart);
}

TEST_F(TunerTest, PosePathDrivesTheBodyAlongItsWaypoints) {
  // Drive a free cube's COM 0.5 m along +x while turning it a quarter turn about z, over 1 s.
  real const quarterTurn = std::numbers::pi_v<real> / 2;
  Real3 const com{0.05f, 0.05f, 0.05f};
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  LoadSpec load;
  load.name = "drive";
  load.type = LoadType::PosePath;
  load.target = "cube";
  load.pointLocal = com;
  load.waypoints.push_back({0.0, com, {0.0f, 0.0f, 0.0f}, Interpolation::Linear});
  load.waypoints.push_back(
      {1.0, com + Real3{0.5f, 0.0f, 0.0f}, {0.0f, 0.0f, quarterTurn}, Interpolation::Linear});
  spec.loads.push_back(load);
  spec.probes.push_back(MakeProbe(ProbeKind::BodyPose, "cube"));
  spec.run.duration = 1.5;

  RunCapture const run = Run(spec);

  ASSERT_EQ(run.summary.status, "ok");
  // The drive pushes the cube up to the path's speed on the first sample. Halfway along, it tracks
  // the moving target closely, not just once the path has stopped, and needs no force.
  EXPECT_GT(run.Value("load.drive.mag", 0), 0);
  size_t const halfway = run.samples.size() / 3;
  EXPECT_LT(run.Value("load.drive.error", halfway), 0.05f);
  EXPECT_LT(run.Final("load.drive.error"), 0.005f);
  EXPECT_LT(run.Final("load.drive.angleError"), 0.01f);
  EXPECT_NEAR(run.Final("cube.rotvec.z"), quarterTurn, 0.01f);
  // The corner origin, a quarter turn about the moved COM: (0.55, 0.05) + (0.05, -0.05).
  EXPECT_NEAR(run.Final("cube.pos.x"), 0.6f, 0.01f);
  EXPECT_NEAR(run.Final("cube.pos.y"), 0.0f, 0.01f);
}

TEST_F(TunerTest, PosePathDriveFollowsEachWaypointsMode) {
  // Carry a free cube along +x at 1 m/s, release it from 0.4 s, drive only its position from 0.8 s,
  // then hold at the last waypoint.
  Real3 const com{0.05f, 0.05f, 0.05f};
  Real3 const noRotation{0.0f, 0.0f, 0.0f};
  auto const along = [&](real x) { return com + Real3{x, 0.0f, 0.0f}; };
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  LoadSpec load;
  load.name = "drive";
  load.type = LoadType::PosePath;
  load.target = "cube";
  load.pointLocal = com;
  load.waypoints.push_back(
      {0.0, along(0.0f), noRotation, Interpolation::Linear, PathDriveMode::On});
  load.waypoints.push_back(
      {0.4, along(0.4f), noRotation, Interpolation::Linear, PathDriveMode::Off});
  load.waypoints.push_back(
      {0.8, along(0.8f), noRotation, Interpolation::Linear, PathDriveMode::PositionOnly});
  load.waypoints.push_back(
      {1.2, along(1.2f), noRotation, Interpolation::Linear, PathDriveMode::PositionOnly});
  spec.loads.push_back(load);
  spec.probes.push_back(MakeProbe(ProbeKind::BodyVelocity, "cube"));
  spec.run.duration = 1.6;

  RunCapture const run = Run(spec);

  ASSERT_EQ(run.summary.status, "ok");
  // Sample i ends at (i + 1) dt.
  auto const sampleAt = [&](double t) {
    return static_cast<size_t>(std::lround(t / spec.run.dt)) - 1;
  };
  EXPECT_EQ(run.Value("load.drive.drive", sampleAt(0.2)), 2);
  // Released: nothing acts on it in zero gravity, so it coasts at the speed it was carried.
  size_t const released = sampleAt(0.5);
  size_t const coasting = sampleAt(0.75);
  EXPECT_EQ(run.Value("load.drive.drive", released), 0);
  EXPECT_EQ(run.Value("load.drive.mag", released), 0);
  EXPECT_EQ(run.Value("load.drive.error", released), 0);
  real const releaseSpeed = run.Value("cube.vel.x", released);
  EXPECT_GT(releaseSpeed, 0.5f);
  EXPECT_NEAR(run.Value("cube.vel.x", coasting), releaseSpeed, 0.01f * releaseSpeed);
  // Position only: the point is pulled onto the path while the orientation is left free.
  EXPECT_EQ(run.Value("load.drive.drive", sampleAt(1.0)), 1);
  EXPECT_EQ(run.Value("load.drive.angleError", sampleAt(1.0)), 0);
  EXPECT_LT(run.Final("load.drive.error"), 0.005f);
}

TEST_F(TunerTest, DescribeReportsCentersOfMassOfBodiesAndLinks) {
  // Both cubes span [0, 0.1]^3 in their own frame, so their COM is at 0.05 on each axis.
  Real3 const expected{0.05f, 0.05f, 0.05f};
  std::optional<Real3> const rigid =
      FindCenterOfMassLocal(Describe(MakeSpec(kRigidCubePrefab)), "cube");
  std::optional<Real3> const link =
      FindCenterOfMassLocal(Describe(MakeSpec(kFreeLinkPrefab)), "arm/body");

  ASSERT_TRUE(rigid.has_value());
  ASSERT_TRUE(link.has_value());
  EXPECT_NEAR(mochi::Norm(*rigid - expected), 0, 1e-4f);
  EXPECT_NEAR(mochi::Norm(*link - expected), 0, 1e-4f);
  EXPECT_FALSE(FindCenterOfMassLocal(Describe(MakeSpec(kRigidCubePrefab)), "nope").has_value());
}

TEST_F(TunerTest, DescribeReportsTheDriveScaleABodyOrLinkIsDrivenWith) {
  // A 1 kg, 0.1 m cube: inertia m (0.1^2 + 0.1^2) / 12 about each axis.
  real const cubeInertia = 0.02f / 12;
  std::optional<DriveScale> const rigid =
      FindDriveScale(Describe(MakeSpec(kRigidCubePrefab)), "cube");
  // The carriage uses the mass of its whole articulation: itself plus the 1 kg base.
  std::optional<DriveScale> const link =
      FindDriveScale(Describe(MakeSpec(kSliderPrefab)), "slider/carriage");

  ASSERT_TRUE(rigid.has_value());
  ASSERT_TRUE(link.has_value());
  EXPECT_NEAR(rigid->mass, 1, 1e-3f);
  EXPECT_NEAR(rigid->inertia, cubeInertia, 1e-3f * cubeInertia);
  EXPECT_NEAR(link->mass, 2, 2e-3f);
  EXPECT_NEAR(link->inertia, 2 * cubeInertia, 2e-3f * cubeInertia);
  EXPECT_FALSE(FindDriveScale(Describe(MakeSpec(kRigidCubePrefab)), "nope").has_value());
}

TEST_F(TunerTest, SolverStatsReportConvergenceAndPerformanceReportsSolveTime) {
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  spec.probes.push_back(MakeProbe(ProbeKind::SolverStats, ""));
  spec.probes.push_back(MakeProbe(ProbeKind::Performance, ""));
  spec.run.duration = 0.05;

  RunCapture const run = Run(spec);

  EXPECT_EQ(
      run.Final("solver.convergence"), static_cast<real>(mochi::ConvergenceStatus::Converged));
  EXPECT_GT(run.Final("perf.solveMs"), 0);
}

TEST_F(TunerTest, UnwrappedRotationVectorStaysContinuousThroughFullTurns) {
  TunerSpec spec = MakeSpec(kRigidCubePrefab);
  // Spins a free-floating cube about world z through more than a full turn, well under pi per step.
  spec.loads.push_back(MakeLoad("spin", LoadType::Torque, "cube", {0.0f, 0.0f, 1.0f}, 0.05f));
  spec.probes.push_back(MakeProbe(ProbeKind::BodyPose, "cube"));
  spec.run.gravity = Real3{0.0f, 0.0f, 0.0f};
  spec.run.groundPlane = false;
  spec.run.duration = 1.0;

  RunCapture const run = Run(spec);

  real maxWrappedJump = 0;
  real maxUnwrappedJump = 0;
  for (size_t i = 1; i < run.samples.size(); ++i) {
    maxWrappedJump = std::max(
        maxWrappedJump,
        std::abs(run.Value("cube.rotvec.z", i) - run.Value("cube.rotvec.z", i - 1)));
    maxUnwrappedJump = std::max(
        maxUnwrappedJump,
        std::abs(
            run.Value("cube.rotvecUnwrapped.z", i) - run.Value("cube.rotvecUnwrapped.z", i - 1)));
  }
  EXPECT_GT(maxWrappedJump, mochi::kPI);
  EXPECT_LT(maxUnwrappedJump, 0.5f);
  EXPECT_GT(run.Final("cube.rotvecUnwrapped.z"), 2 * mochi::kPI);
}

} // namespace
} // namespace superdex::tuner::test
