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

#include "mochi_contact.h"
#include "mochi_contact_anchors.h"
#include "mochi_ecs_utils.h"
#include "mochi_island.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"

#include <mochi_core/utils/dynamic_array.h>

namespace mochi::diffsim {
// Forward declarations: this header only aliases these types as ECS components.
struct BackPropagationSolverParams;
struct BackPropagationSceneStats;
} // namespace mochi::diffsim

namespace mochi {

/**************************************************************************
  ECS components for differentiable scenes, islands and actors
*/

// Tag to denote a differentiable scene.
struct TagDifferentiableScene {};

// Tag set by PrepareBackPropagate and cleared by BackPropagate.
struct TagBackPropagationPrepared {};

// Stores the state pair restored by PrepareBackPropagate.
struct CStatePair {
  StateHandle stateNew;
  StateHandle stateOld;
};

// Tag to denote a constraint with differentiable input (e.g. pose controller).
struct TagConstraintWithDifferentiableInput {};

// Component to store the differentiability solver parameters.
using CBackPropagationSolverParams = diffsim::BackPropagationSolverParams;

// Component to store the performance metrics of the last back-propagation step.
using CBackPropagationSceneStats = diffsim::BackPropagationSceneStats;

// Stores information on the number of DoFs of the derived state of an island, necessary for sizing
// derived state vectors.
struct CIslandDerivedStateInfo : NoCopy {
  int dofsSize = 0;
};

// Stores information on the number of DoFs of the derived state of an actor, necessary for indexing
// derived state vectors.
struct CActorDerivedStateInfo : NoCopy {
  int dofsSize = 0;
};

// Stores the DoF offset of the derived state of an actor within an island, necessary for indexing
// derived state vectors.
struct CDerivedStateOffset : NoCopy {
  int dofsOffset = 0;
};

// Stores information on the size of the differentiable input of an island, necessary for sizing
// input vectors.
struct CIslandDiffInputInfo : NoCopy {
  int size = 0;
};

// Stores information on the statistics of an island backprop solve execution.
struct CIslandBackPropSolverStats : NoCopy {
  StageSolverStats stats;
  // True if every finite-difference Hvp validation check this back-prop
  // performed (across every Hvp evaluation in the outer solve) passed its
  // tolerance. Only meaningful when BackPropagationSolverParams::validateFiniteDiff
  // is set; otherwise stays at its default of true. Aggregated into
  // BackPropagationSceneStats::finiteDiffValid via logical AND across islands.
  bool finiteDiffValid = true;
};

// Stores information on the size of the differentiable input of an actor, necessary for indexing
// input vectors.
struct CActorDiffInputInfo : NoCopy {
  int dofsSize = 0;
};

// Stores the offset of the differentiable input of an actor within an island, necessary for
// indexing input vectors.
struct CDiffInputOffset : NoCopy {
  int dofsOffset = 0;
};

// Components to store adjoints. CDiffDerivedStepGrad is only used internally across calls to
// BackPropagate. CDiffStateGrad OTOH receives external gradients in GetFooBackward() functions, and
// exposes gradients for SetFooBackward() functions.
struct CDiffStateGrad {
  ColumnVector<real> value;

  CDiffStateGrad() = default;
  explicit CDiffStateGrad(int size) : value(size) {}

  MOCHI_STRUCT_BEGIN(mochi::CDiffStateGrad);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_ATTRIBUTE(HasAdjoint);
  MOCHI_FIELD(value);
  MOCHI_STRUCT_END();
};
struct CDiffDerivedStepGrad {
  ColumnVector<real> value;

  CDiffDerivedStepGrad() = default;
  explicit CDiffDerivedStepGrad(int size) : value(size) {}

  MOCHI_STRUCT_BEGIN(mochi::CDiffDerivedStepGrad);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_ATTRIBUTE(HasAdjoint);
  MOCHI_FIELD(value);
  MOCHI_STRUCT_END();
};
// Component to store contact-force adjoints, templatized by GradTarget.
template <GradTarget kGradTarget>
struct CDiffContactGrad : public ColumnVector<real> {
  using ColumnVector<real>::ColumnVector;
};
// Components to store temporary gradient data during the back-propagation step.
struct CDiffContainerState : public ColumnVector<real> {
  using ColumnVector<real>::ColumnVector;
};
struct CDiffContainerDerivedState : public ColumnVector<real> {
  using ColumnVector<real>::ColumnVector;
};
// Component to store gradients wrt pose targets computed during the back-propagation step.
struct CDiffTargetPoseGrad : NoCopy {
  ColumnVector<real> current; // Gradient wrt the current target pose
  ColumnVector<real> previous; // Gradient wrt the previous target pose
  ColumnVector<real> propagated; // Gradient accumulated from previous steps

  CDiffTargetPoseGrad() = default;
  explicit CDiffTargetPoseGrad(int size)
      : current(ColumnVector<real>::Zero(size)),
        previous(ColumnVector<real>::Zero(size)),
        propagated(ColumnVector<real>::Zero(size)) {}

  MOCHI_STRUCT_BEGIN(mochi::CDiffTargetPoseGrad);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_ATTRIBUTE(HasAdjoint);
  MOCHI_FIELD(propagated);
  // `current` and `previous` are not captured.
  MOCHI_STRUCT_END();
};
// Component to store the gradient wrt external forces computed during the back-propagation step. On
// articulated links, it is set only in islands whose step read friction anchors, whose adjoints
// read it.
struct CDiffForceGrad : public ColumnVector<real> {
  using ColumnVector<real>::ColumnVector;
};
// Anchor-offset adjoints parallel to the corresponding Active/Pending anchor set. Pending carries
// incoming and handed-off state adjoints; Active accumulates the step-used-anchor adjoints.
// BackPropagate reverses drops from Active into the old Pending layout. Empty Pending means zero.
template <AnchorSet kSet>
struct CDiffFrictionAnchorGrad {
  DynamicArray<Real3> grads;

  MOCHI_TEMPLATE_BEGIN(mochi::CDiffFrictionAnchorGrad, kSet);
  MOCHI_ATTRIBUTE_IF(kSet == AnchorSet::Pending, CaptureState);
  MOCHI_ATTRIBUTE(HasAdjoint);
  MOCHI_FIELD(grads);
  MOCHI_TEMPLATE_END();
};

/**************************************************************************
  Utility functions
*/

// Function to obtain TimeStep from GradTarget. Should not be called with GradTarget::PreviousDelta.
template <GradTarget kGradTarget>
TimeStep constexpr GetTimeStep() {
  if constexpr (kGradTarget == GradTarget::Current || kGradTarget == GradTarget::CurrentInput) {
    return TimeStep::Current;
  } else {
    static_assert(
        kGradTarget == GradTarget::Previous || kGradTarget == GradTarget::PreviousInput,
        "Unexpected grad target");
    return TimeStep::StageStart;
  }
}

// Get the colliding position at the appropriate time step. Only supported for TimeStep::Current and
// TimeStep::StageStart.
template <TimeStep kTimeStep>
Real3 const& GetCollidingPosition(ContactDetectionResult const& data, int contact) {
  static_assert(
      kTimeStep == TimeStep::Current || kTimeStep == TimeStep::StageStart,
      "Only TimeStep::Current and TimeStep::StageStart store colliding positions");
  return (kTimeStep == TimeStep::Current) ? data.posColliding[contact]
                                          : data.posCollidingStageStart[contact];
}

/**************************************************************************
  Back-propagation operations
*/

void PrepareBackPropagation(entt::registry& reg);

void BackPropagationSolve(entt::registry& reg);

/**************************************************************************
  Per-actor systems
*/

// System to emplace differentiability components on actors
void EmplaceDifferentiabilityComponents(
    int numDerivedStateDofs,
    entt::registry& reg,
    entt::entity e,
    CActorDofInfo const& dofInfo);

// System to emplace components for differentiable contact forces
void EmplaceDifferentiableContactComponents(
    entt::registry& reg,
    entt::entity e,
    CActorDofInfo const& dofInfo);

// Emplaces the anchor adjoint component on the differentiable actors with friction anchors, or
// removes it from all actors, to follow ExperimentalEvalParams::elastoPlasticFriction in a scene
// that is already differentiable. Call after contact::EnableFrictionAnchorComponents.
void EnableDifferentiableFrictionAnchorComponents(entt::registry& reg, bool enable);

// System to emplace differentiability components on constraints
void EmplaceConstraintDifferentiabilityComponents(entt::registry& reg, entt::entity e);

// System to reset back-propagation components between runs.
void ResetBackPropagationContainers(
    CDiffStateGrad& outGradState,
    CDiffDerivedStepGrad& outGradDerivedStep,
    CDiffTargetPoseGrad* outTargetPoseGrad);

// System to reset the friction anchor adjoints between runs. Separate from
// ResetBackPropagationContainers, because articulated links have anchor adjoints but no state
// adjoint.
void ResetFrictionAnchorAdjoints(
    CDiffFrictionAnchorGrad<AnchorSet::Active>& active,
    CDiffFrictionAnchorGrad<AnchorSet::Pending>& pending);

// System to reset contact force adjoints before running backward contact queries.
void PrepareContactForceAdjoints(
    CQueryActorContactForces const& queryActorContactForces,
    CRequiresFarSdfEvaluation const* farSdfEval,
    CActiveCollisions</*kIsSync*/ false, TimeStep::Current>& outActiveCollisionsAsync,
    CActiveCollisions</*kIsSync*/ true, TimeStep::Current>& outActiveCollisionsSync,
    CCollJacs<CollRole::Collider>* outColliderJacs);

// System to accumulate contact force adjoints to actor level. With elasto-plastic friction, it also
// adds the direct dependence of the queried contact forces on the anchor offsets that the step
// read.
void AccumulateContactForceAdjoints(entt::registry& reg);

// After restoring the old state except Active anchors and adjoints, reverse the forward drop:
// scatter the Active gradient into the old Pending layout, leaving dropped-anchor entries zero.
void ReverseFrictionAnchorDropAdjoints(
    CDiffFrictionAnchorGrad<AnchorSet::Active>& activeGrad,
    CDiffFrictionAnchorGrad<AnchorSet::Pending>& pendingGrad,
    CFrictionAnchors<AnchorSet::Active> const& activeAnchors,
    CFrictionAnchors<AnchorSet::Pending> const& pendingAnchors);

// Back-propagates through the friction anchor advance of the step (AdvanceFrictionAnchorOffsets),
// before the back-propagation solve: adds the contributions of the anchor-offset adjoint to
// CDiffContactGrad and its direct contribution to the offsets that the step read. Call after
// AccumulateContactForceAdjoints.
void AccumulateFrictionAnchorAdjoints(entt::registry& reg);

// Adds the dependence of the solve on the anchor offsets that the step read, using the solve's
// adjoint (-CDiffForceGrad), and stores their adjoint for the step before it. Call after
// BackPropagationSolve, before restoring the old state.
void PropagateFrictionAnchorAdjoints(entt::registry& reg);

namespace differentiable {
void InitializeOnce(entt::registry& reg);
}

} // namespace mochi
