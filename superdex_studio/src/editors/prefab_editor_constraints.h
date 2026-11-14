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

#include <mochi_physics/utils/mochi_prefab.h>

#include <string>
#include <vector>

// Constraint editing widgets for the Prefab Editor. The editor's own constraint methods (hierarchy
// rows, creation, drawing) are in prefab_editor_constraints.cpp too.

namespace superdex::studio {

struct SceneStage;

/// What the constraint widgets read besides the constraint itself.
struct ConstraintEditContext {
  mochi::prefab::ScenePrefab const& prefab;
  ConstraintActorIndex const& actors;
  // Staged actors (rest pose while editing), for the buttons that place anchors and targets.
  SceneStage const& stage;
};

/// Draws every field of constraint `ref` in `lists` (the constraints of `ctx.prefab`): actor and
/// joint pickers, values in the units of what they constrain, stiffness, damping and saturation.
/// Returns true if anything changed.
bool ConstraintEditor(
    ConstraintEditContext const& ctx,
    mochi::prefab::ConstraintLists& lists,
    ConstraintRef ref);

/// The actors a new constraint attaches to, in order of preference.
struct ConstraintTargets {
  std::string articulatedActor; // For joint constraints.
  int joint = -1; // Preferred joint of `articulatedActor`, or -1 for the first one that fits.
  std::vector<std::string> rigid; // For pair (first two) and pivot (first) constraints.
  std::vector<std::string> deformable; // For node constraints.
};

/// Attaches constraint `ref` (freshly appended) to `targets` and sets it up so it starts satisfied
/// where it can: joint ranges copy the joint's limits, anchors and targets sit where the actors
/// rest.
void AttachConstraint(
    ConstraintEditContext const& ctx,
    mochi::prefab::ConstraintLists& lists,
    ConstraintRef ref,
    ConstraintTargets const& targets);

} // namespace superdex::studio
