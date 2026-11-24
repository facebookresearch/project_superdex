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

#include "superdex_asset_tuner/load_math.h"
#include "superdex_asset_tuner/tuner_spec.h"

#include <mochi_physics/mochi_physics.h>

namespace superdex::robotics {
class BotScene;
} // namespace superdex::robotics

namespace superdex::tuner {

struct RigidTarget {
  DynamicString name;
  bool isStatic = false;
  real mass = 0;
  // Average of ixx, iyy and izz (see MeanDiagonalInertia). Zero for static bodies.
  real inertia = 0;
  // In the body's local frame; zero for static bodies.
  Real3 centerOfMassLocal{0.0f, 0.0f, 0.0f};

  MOCHI_STRUCT_BEGIN(superdex::tuner::RigidTarget)
  MOCHI_FIELD(name)
  MOCHI_FIELD(isStatic)
  MOCHI_FIELD(mass) MOCHI_ATTRIBUTE(Units("kg"));
  MOCHI_FIELD(inertia) MOCHI_ATTRIBUTE(Units("kg*m^2"));
  MOCHI_FIELD(centerOfMassLocal) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_STRUCT_END()
};

struct JointTarget {
  DynamicString name;
  int dofOffset = 0;
  int dofCount = 0;

  MOCHI_STRUCT_BEGIN(superdex::tuner::JointTarget)
  MOCHI_FIELD(name)
  MOCHI_FIELD(dofOffset)
  MOCHI_FIELD(dofCount)
  MOCHI_STRUCT_END()
};

struct ArticulatedTarget {
  DynamicString name;
  int numDofs = 0;
  DynamicArray<DynamicString> links;
  // Each link's center of mass in its own frame, in the order of `links`.
  DynamicArray<Real3> linkCentersOfMass;
  // Each link's mass and average inertia, in the order of `links`.
  DynamicArray<real> linkMasses;
  DynamicArray<real> linkInertias;
  DynamicArray<JointTarget> joints;

  MOCHI_STRUCT_BEGIN(superdex::tuner::ArticulatedTarget)
  MOCHI_FIELD(name)
  MOCHI_FIELD(numDofs)
  MOCHI_FIELD(links)
  MOCHI_FIELD(linkCentersOfMass) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(linkMasses) MOCHI_ATTRIBUTE(Units("kg"));
  MOCHI_FIELD(linkInertias) MOCHI_ATTRIBUTE(Units("kg*m^2"));
  MOCHI_FIELD(joints)
  MOCHI_STRUCT_END()
};

// The joint an entry of a joint-position input sets, and its limits about the joint's axis (equal
// when it has none).
struct InputJoint {
  DynamicString name;
  mochi::ArticulatedJointType type = mochi::ArticulatedJointType::Revolute;
  real min = 0;
  real max = 0;

  MOCHI_STRUCT_BEGIN(superdex::tuner::InputJoint)
  MOCHI_FIELD(name)
  MOCHI_FIELD(type)
  MOCHI_FIELD(min)
  MOCHI_FIELD(max)
  MOCHI_STRUCT_END()
};

// A bot controller a ControllerInput load can drive, with its input at the start of a run.
struct ControllerTarget {
  // "bot/controller"
  DynamicString name;
  DynamicString type;
  // "JointPositions", "Pose" (x y z, qx qy qz qw in the bot's root frame) or "Custom".
  DynamicString inputKind;
  DynamicArray<real> input;
  // JointPositions: one per input entry.
  DynamicArray<InputJoint> inputJoints;
  // The bot's root at the start of a run, the frame of a Pose input: world position and rotation
  // vector.
  Real3 rootPosition{0.0f, 0.0f, 0.0f};
  Real3 rootRotation{0.0f, 0.0f, 0.0f};

  MOCHI_STRUCT_BEGIN(superdex::tuner::ControllerTarget)
  MOCHI_FIELD(name)
  MOCHI_FIELD(type)
  MOCHI_FIELD(inputKind)
  MOCHI_FIELD(input)
  MOCHI_FIELD(inputJoints) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(rootPosition) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(rootRotation) MOCHI_ATTRIBUTE(Units("rad"));
  MOCHI_STRUCT_END()
};

// Everything a LoadSpec or ProbeSpec can target in a scene.
struct SceneTargets {
  DynamicArray<RigidTarget> rigid;
  DynamicArray<ArticulatedTarget> articulated;
  // Height of the tuner ground plane (the override or the automatic height); unset without one.
  std::optional<real> groundHeight = std::nullopt;
  DynamicArray<ControllerTarget> controllers;

  MOCHI_STRUCT_BEGIN(superdex::tuner::SceneTargets)
  MOCHI_FIELD(rigid)
  MOCHI_FIELD(articulated)
  MOCHI_FIELD(groundHeight) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(controllers) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

[[nodiscard]] SceneTargets CollectSceneTargets(mochi::Scene& scene);

// The built bots' controllers that take a float input. Fails when one cannot report its input.
bool CollectControllerTargets(
    robotics::BotScene& bots,
    DynamicArray<ControllerTarget>& outTargets,
    std::string& outError);

// Center of mass, in its own frame, of a dynamic rigid body or articulated link ("actor/link"), or
// nullopt if `targets` has no such body.
[[nodiscard]] std::optional<Real3> FindCenterOfMassLocal(
    SceneTargets const& targets,
    std::string_view target);

// The mass and inertia a PosePath drive uses for a dynamic rigid body or articulated link
// ("actor/link"). Matches what a tuner run uses. Nullopt if `targets` has no such body.
[[nodiscard]] std::optional<DriveScale> FindDriveScale(
    SceneTargets const& targets,
    std::string_view target);

// Loads the tuner's scene into a headless scene (same setup as RunTuner) and collects targets.
bool DescribeTuner(
    TunerSpec const& spec,
    std::filesystem::path const& tunerFilePath,
    SceneTargets& outTargets,
    std::string& outError);

// As above, in a temporary scene of `context`. Use this inside an app that already has a context: a
// second context binds its task scheduler to the calling thread, and destroying it leaves the
// thread unbound from the app's scheduler, so the app hangs when its own context shuts down.
bool DescribeTuner(
    TunerSpec const& spec,
    std::filesystem::path const& tunerFilePath,
    mochi::Context* context,
    SceneTargets& outTargets,
    std::string& outError);

} // namespace superdex::tuner
