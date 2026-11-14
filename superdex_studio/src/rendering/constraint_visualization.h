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

#include "assets/prefab_constraints.h"

#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <math/vec3.h>

#include <optional>

namespace mochi_renderer {
class DebugDraw;
}

namespace superdex::studio {

class DebugText;
struct SceneStage;

/// How prominently a constraint is drawn.
enum class ConstraintEmphasis {
  Normal, // Shown with all the others (Show > Constraints): thin, no labels.
  Hovered, // Hovered in the hierarchy: bold, no labels.
  Selected, // The selected constraint: bold, with labels.
};

/// What the constraint drawing reads: the prefab, the staged actors (rest pose while editing, the
/// simulated pose while playing), and where to draw.
struct ConstraintDrawContext {
  mochi::prefab::ScenePrefab const& prefab;
  SceneStage const& stage;
  mochi_renderer::DebugDraw& debugDraw;
  DebugText* debugText = nullptr;
  mochi::CoordinateSpaceConverter const& editorToRenderer;
  // Camera position in renderer space; markers are sized by their distance to it so they keep a
  // constant size on screen.
  filament::math::float3 cameraPosition;
};

/// Draws constraint `ref` of `ctx.prefab` with debug geometry: joint ranges as arcs or bars, joint
/// targets as a marker with a spring to the current value, anchors and the lines tying them for
/// actor-pair and pivot constraints. Draws nothing when the actors it references are not staged.
void DrawConstraint(
    ConstraintDrawContext const& ctx,
    ConstraintRef ref,
    ConstraintEmphasis emphasis);

/// The two frames of articulated joint `jointIndex` of `actor`, in editor space: `parent` is fixed
/// to the parent link (parentLinkFromJoint), `child` moves with the child link. They coincide at
/// the joint's zero position. `axis` is the joint axis in joint coordinates (normalized; zero for
/// joints without one). Distances include the actor's scale.
struct ArticulatedJointFrames {
  mochi::TransformRT parent;
  mochi::TransformRT child;
  mochi::Real3 axis;
  mochi::ArticulatedJointType type = mochi::ArticulatedJointType::Invalid;
  mochi::real scale = 1;
};
std::optional<ArticulatedJointFrames> ComputeArticulatedJointFrames(
    mochi::prefab::ArticulatedActorPrefab const& actor,
    int jointIndex,
    SceneStage const& stage);

/// The value of single DoF `dofIndex` of `frames`: an angle [rad] for a revolute joint, a distance
/// [m] (scaled, as simulated) for a prismatic joint or a free joint's translations.
mochi::real MeasureJointDof(ArticulatedJointFrames const& frames, int dofIndex);

} // namespace superdex::studio
