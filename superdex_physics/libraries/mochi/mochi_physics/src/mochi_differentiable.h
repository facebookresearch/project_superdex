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
#include "mochi_ecs_utils.h"
#include "mochi_island.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"

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

// Store the DoF offset of the actor in the scene
struct CSceneStateOffset : NoCopy {
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
// Component to store the gradient wrt external forces computed during the back-propagation step.
struct CDiffForceGrad : public ColumnVector<real> {
  using ColumnVector<real>::ColumnVector;
};

struct CForwardPropContainerDerivedStateJac {
  Matrix<real> data;
  // The following two fields are used to store the island this actor belongs to,
  // including other actors in the island as well as the total number of degrees of freedom.
  //
  // We store these information so that it is kept when we restore a new state, which overrides the
  // island information. Note that GetStepJacobian correlates three states: q_k,q_k-1,q_k-2.
  // In order to compute dq_k/dq_k-1, we need to restore to state q_k, where we record the island
  // information. Next, in order to compute dq_k/dq_k-2, we need to restore to state q_k-1, but we
  // need to use island information at q_k, which is stored here.
  DynamicArray<entt::entity> actors;
  int numIslandDofs = 0;
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
  Fore/Back propagation operations
*/

void PrepareBackPropagation(entt::registry& reg);

void BackPropagationSolve(entt::registry& reg);

void ComputeHqx(
    int numIslandDofs,
    Span<entt::entity const> actors,
    CActorSnle const& actorSnle,
    CDofOffset const& dofOffset,
    CActorDerivedStateInfo const& derivedStateInfo,
    CForwardPropContainerDerivedStateJac& outDerivedState);

void ComputeDqDDerivedState(
    int numIslandDofs,
    LU<real> const& invDRes,
    Span<entt::entity const> actors,
    CActorSnle const& actorSnle,
    CDofOffset const& dofOffset,
    CActorDerivedStateInfo const& derivedStateInfo,
    CForwardPropContainerDerivedStateJac& outDerivedState);

void StepJacobianSolve(entt::registry& reg, MatrixView<real> jacCurr);

void StepJacobianShiftAndProject(
    entt::registry& reg,
    MatrixView<real> jacCurr,
    MatrixView<real> jacOld);

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

// System to emplace differentiability components on constraints
void EmplaceConstraintDifferentiabilityComponents(entt::registry& reg, entt::entity e);

// System to reset back-propagation components between runs.
void ResetBackPropagationContainers(
    CDiffStateGrad& outGradState,
    CDiffDerivedStepGrad& outGradDerivedStep,
    CDiffTargetPoseGrad* outTargetPoseGrad);

// System to reset contact force adjoints before running backward contact queries.
void PrepareContactForceAdjoints(
    CQueryActorContactForces const& queryActorContactForces,
    CRequiresFarSdfEvaluation const* farSdfEval,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& outActiveCollisionsAsync,
    CActiveCollisions<ContactType::Sync, TimeStep::Current>& outActiveCollisionsSync,
    CCollJacs<CollRole::Collider>* outColliderJacs);

// System to accumulate contact force adjoints to actor level.
void AccumulateContactForceAdjoints(entt::registry& reg);

namespace differentiable {
void InitializeOnce(entt::registry& reg);
}

} // namespace mochi
