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

#include "mochi_attributes.h"
#include "mochi_ecs.h"

#include <mochi_core/solvers/linear_solver_params.h>
#include <mochi_core/utils/no_copy.h>

#include <cstdint>
#include <optional>

namespace mochi {

struct CIslandDescendants;
struct CIslandSolverStats;
struct KrylovSolverParams;

// Mode of IslandActorPreconditionerPolicy.
enum class ActorPreconditionerPolicyMode {
  BlockJacobi,
  AmgTrial,
  Amg,
  BlockJacobiProbe,
  BlockJacobiAfterAmgDivergence,
};

} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::ActorPreconditionerPolicyMode);
MOCHI_ENUM_ITEM(BlockJacobi);
MOCHI_ENUM_ITEM(AmgTrial);
MOCHI_ENUM_ITEM(Amg);
MOCHI_ENUM_ITEM(BlockJacobiProbe);
MOCHI_ENUM_ITEM(BlockJacobiAfterAmgDivergence);
MOCHI_ENUM_END();

namespace mochi {

// Selects one preconditioner for all eligible actors in an island (see
// IsEligibleForAdaptivePreconditioner). Island solve statistics cannot attribute performance or
// failure to individual actors, including ineligible ones, so a uniform choice avoids the
// combinatorial trial space of mixed assignments but may miss a faster mixed assignment. The
// selection compares the mean cost per linear solve of both preconditioners (see AmgCost) and
// switches only to one that costs less by a margin, so that near ties don't switch:
// - kBjHighIterationStepsBeforeAmgTrial consecutive Block Jacobi observations with linear solves
//   that each stop or average at least AmgCost::bjIterationsForAmgTrial iterations per linear
//   solve trigger an AMG trial.
// - The trial commits AMG if AMG costs less than Block Jacobi did before the trial; a stopped AMG
//   solve ends it at once, since a stopped AMG window never costs less. Otherwise, a cooldown
//   follows, in which only observations a fixed percentage costlier than before the trial count
//   toward a new trial, and none do if Block Jacobi stopped before the trial.
// - Committed AMG is periodically checked by a Block Jacobi probe, which leaves AMG if Block Jacobi
//   costs less than AMG did in its last two observations. Each failed probe doubles the interval to
//   the next one, up to a maximum, so that persistently hard scenes pay for fewer failed probes.
//   Other entries into AMG restore the minimum interval. A stopped AMG solve brings the next probe
//   forward, unless it is already due sooner, to compare the two AMG observations after the stop,
//   but no sooner than kMinAmgStepsBetweenBjProbes observations, in any mode, after the previous
//   probe started.
// Divergence switches to the other preconditioner, except that after AMG diverges, Block Jacobi is
// retained while it diverges too, and then a cooldown follows in which no observation counts toward
// an AMG trial.
// While adaptive selection stays enabled, changes to solver parameters other than the linear solver
// type do not reset the policy. Observations spanning such changes may produce suboptimal
// preconditioner selections.
struct IslandActorPreconditionerPolicy {
  // Costs in units of 1 / kCostUnitsPerBlockJacobiIteration of a Block Jacobi iteration of the
  // island, so that comparisons are exact.
  static constexpr int64_t kCostUnitsPerBlockJacobiIteration = 10;

  // Cost of AMG relative to Block Jacobi in an island, and when to try AMG. A linear solve that
  // takes n iterations costs n * kCostUnitsPerBlockJacobiIteration with Block Jacobi, and
  // updateCost + n * iterationCost with AMG.
  struct AmgCost {
    // One AMG iteration.
    int64_t iterationCost = 0;

    // Updating AMG for a linear solve, beyond updating Block Jacobi.
    int64_t updateCost = 0;

    // Block Jacobi iterations per linear solve from which Block Jacobi observations count toward an
    // AMG trial.
    int64_t bjIterationsForAmgTrial = 0;

    bool operator==(AmgCost const&) const = default;

    MOCHI_STRUCT_BEGIN(mochi::IslandActorPreconditionerPolicy::AmgCost);
    MOCHI_FIELD(iterationCost);
    MOCHI_FIELD(updateCost);
    MOCHI_FIELD(bjIterationsForAmgTrial);
    MOCHI_STRUCT_END();
  };

  struct StepObservation {
    // Sum of linear solver iterations, excluding retried solves, whatever their convergence. The
    // selection reads it only if no linear solve stopped or diverged.
    int64_t linearIterationSum = 0;

    // Number of Newton iterations with at least one linear solver attempt.
    int64_t numLinearSolves = 0;

    // Whether a linear solve stopped or diverged. The selection reads hasStoppedLinearSolve only if
    // no linear solve diverged.
    bool hasStoppedLinearSolve = false;
    bool hasDivergedLinearSolve = false;

    StepObservation& operator+=(StepObservation const& other) {
      linearIterationSum += other.linearIterationSum;
      numLinearSolves += other.numLinearSolves;
      hasStoppedLinearSolve |= other.hasStoppedLinearSolve;
      hasDivergedLinearSolve |= other.hasDivergedLinearSolve;
      return *this;
    }

    MOCHI_STRUCT_BEGIN(mochi::IslandActorPreconditionerPolicy::StepObservation);
    MOCHI_FIELD(linearIterationSum);
    MOCHI_FIELD(numLinearSolves);
    MOCHI_FIELD(hasStoppedLinearSolve);
    MOCHI_FIELD(hasDivergedLinearSolve);
    MOCHI_STRUCT_END();
  };

  struct Configuration {
    // HashUnordered of the eligible actors: the selection depends on which actors are eligible, not
    // on their order.
    uint64_t eligibleActorsHash = 0;
    LinearSolverType solverType = LinearSolverType::Default;
    AmgCost amgCost;

    bool operator==(Configuration const&) const = default;

    MOCHI_STRUCT_BEGIN(mochi::IslandActorPreconditionerPolicy::Configuration);
    MOCHI_FIELD(eligibleActorsHash);
    MOCHI_FIELD(solverType);
    MOCHI_FIELD(amgCost);
    MOCHI_STRUCT_END();
  };

  // Returns to the initial state, in which adaptive selection is disabled until the next
  // SetConfiguration.
  void Reset();

  // Enables adaptive selection. Restarts from Block Jacobi if adaptive selection was disabled or
  // the configuration changed.
  void SetConfiguration(Configuration const& configuration);

  // Updates the selection from a step that solved with it. Does nothing while adaptive selection is
  // disabled.
  void UpdateFromStep(StepObservation const& observation);

  // Preconditioner type for the eligible actors, or std::nullopt if adaptive selection is disabled.
  [[nodiscard]] std::optional<PreconditionerType> GetSelectedType() const;

 private:
  using Mode = ActorPreconditionerPolicyMode;

  static constexpr int kBjHighIterationStepsBeforeAmgTrial = 2;
  static constexpr int kAmgTrialSteps = 2;
  static constexpr int kBjProbeStepsToLeaveAmg = 2;
  static constexpr int kMinAmgStepsBetweenBjProbes = 32;
  static constexpr int kMaxAmgStepsBetweenBjProbes = 128;
  static constexpr int kAmgStepsBeforeBjProbeAfterStop = 3;
  static constexpr int kAmgTrialCooldownSteps = 50;

  void ResetObservationWindow();
  void MeasureBlockJacobi(StepObservation const& observation);
  void MeasureAmg(StepObservation const& observation);
  void EnterBlockJacobi();
  void EnterAmgTrial();
  void EnterBlockJacobiCooldown(int64_t bjIterationsForAmgRetry);
  void EnterAmg(int amgStepsBetweenBjProbes = kMinAmgStepsBetweenBjProbes);
  void EnterAmgAfterFailedBjProbe();
  void EnterBlockJacobiAfterAmgDivergence();

  bool _enabled = false;
  Configuration _configuration;
  Mode _mode = Mode::BlockJacobi;
  int _numObservedSolveSteps = 0;
  int _amgStepsBetweenBjProbes = 0;
  int _amgStepsUntilBjProbe = 0;
  StepObservation _blockJacobiWindow;

  // Observations with linear solves, in any mode, since committed AMG last entered a Block Jacobi
  // probe, saturating at kMinAmgStepsBetweenBjProbes + 1.
  int _numObservationsSinceBjProbe = kMinAmgStepsBetweenBjProbes + 1;

  // AMG's last two observations, cleared when a trial starts.
  StepObservation _previousAmgObservation;
  StepObservation _lastAmgObservation;

  int _amgTrialCooldownStepsLeft = 0;

  // Block Jacobi iterations per linear solve from which observations count toward an AMG trial
  // during the cooldown, or 0 if none do.
  int64_t _bjIterationsForAmgRetry = 0;

 public:
  MOCHI_STRUCT_BEGIN(mochi::IslandActorPreconditionerPolicy);
  MOCHI_FIELD(_enabled);
  MOCHI_FIELD(_configuration);
  MOCHI_FIELD(_mode);
  MOCHI_FIELD(_numObservedSolveSteps);
  MOCHI_FIELD(_amgStepsBetweenBjProbes);
  MOCHI_FIELD(_amgStepsUntilBjProbe);
  MOCHI_FIELD(_blockJacobiWindow);
  MOCHI_FIELD(_numObservationsSinceBjProbe);
  MOCHI_FIELD(_previousAmgObservation);
  MOCHI_FIELD(_lastAmgObservation);
  MOCHI_FIELD(_amgTrialCooldownStepsLeft);
  MOCHI_FIELD(_bjIterationsForAmgRetry);
  MOCHI_STRUCT_END();
};

// Adaptive preconditioner selection of an island, which each of its solves configures from the
// copies in its actors.
struct CIslandActorPreconditionerPolicy final : IslandActorPreconditionerPolicy, NoCopy {};

// Copy of the adaptive preconditioner selection of the island that last solved the actor. Island
// state is not captured, so capturing this copy makes the selection replay-exact.
// TODO[T229060237] - Islands that sleep are not replay-exact: sleeping isn't captured, and
// restoring wakes every island.
struct CActorPreconditionerPolicyCopy : NoCopy {
  // island::HashNonCompoundActors of the island that last solved the actor.
  uint64_t islandActorsHash = 0;
  IslandActorPreconditionerPolicy policy;

  MOCHI_STRUCT_BEGIN(mochi::CActorPreconditionerPolicyCopy);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_FIELD(islandActorsHash);
  MOCHI_FIELD(policy);
  MOCHI_STRUCT_END();
};

// Whether the preconditioner type of an actor is selected adaptively when adaptive selection is
// enabled in its island.
[[nodiscard]] bool IsEligibleForAdaptivePreconditioner(
    entt::registry const& reg,
    entt::entity actor);

// Configures the adaptive selection for the island's solve. Adaptive selection requires
// PreconditionerType::AdaptivePerActor, a linear solver that uses IslandOperators, and at least one
// eligible actor. It continues from the copies in the island's actors if all of them were made with
// the island's current actors, and otherwise starts over: the selection compares the island's
// linear-solver iteration counts across steps, and these depend on all island actors, not only
// eligible ones. Islands aren't captured but the copies are, so the selection is replay-exact.
void ConfigureIslandActorPreconditionerPolicy(
    IslandActorPreconditionerPolicy& outPolicy,
    entt::registry const& reg,
    CIslandDescendants const& descendants,
    KrylovSolverParams const& linearParams);

// Updates the adaptive selection from the island's solve, and its copies in the
// CActorPreconditionerPolicyCopy of the island's soft and shell actors.
void UpdateIslandActorPreconditionerPolicyAndCopies(
    IslandActorPreconditionerPolicy& policy,
    entt::registry& reg,
    CIslandDescendants const& descendants,
    CIslandSolverStats const& stats);

} // namespace mochi
