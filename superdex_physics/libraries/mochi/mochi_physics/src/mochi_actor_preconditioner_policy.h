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

#include "mochi_ecs.h"

#include <mochi_core/solvers/linear_solver_params.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/no_copy.h>

#include <cstdint>
#include <optional>

namespace mochi {

struct CIslandDescendants;
struct KrylovSolverParams;

// Selects one preconditioner for all eligible actors in an island (see
// IsEligibleForAdaptivePreconditioner). Island solve statistics cannot attribute performance or
// failure to individual actors, including ineligible ones, so a uniform choice avoids the
// combinatorial trial space of mixed assignments but may miss a faster mixed assignment. The
// selection compares the mean cost per linear solve of both preconditioners (see AmgCost) and
// switches only to one that costs less by a margin, so that near ties don't switch:
// - kBjHighIterationStepsBeforeAmgTrial consecutive Block Jacobi observations with linear solves
//   that each stop or average at least AmgCost::bjIterationsForAmgTrial iterations per linear
//   solve trigger an AMG trial.
// - The trial commits AMG if AMG costs less than Block Jacobi did before the trial. Otherwise, a
//   cooldown follows, in which only observations a fixed percentage costlier than before the trial
//   count toward a new trial, and none do if Block Jacobi stopped before the trial.
// - Committed AMG is periodically checked by a Block Jacobi probe, which leaves AMG if Block Jacobi
//   costs less than AMG did in its last two observations. Each failed probe doubles the interval to
//   the next one, up to a maximum, so that persistently hard scenes pay for fewer failed probes.
//   Other entries into AMG restore the minimum interval, and a stopped AMG solve brings the next
//   probe within the minimum interval.
// Divergence switches to the other preconditioner, except that after AMG diverges, Block Jacobi is
// retained while it diverges too, and then a cooldown follows in which no observation counts toward
// an AMG trial.
// While adaptive selection stays enabled, changes to solver parameters other than the linear solver
// type do not reset the policy. Observations spanning such changes may produce suboptimal
// preconditioner selections.
struct CIslandActorPreconditionerPolicy : public NoCopy {
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
  };

  // Returns to the initial state, in which adaptive selection is disabled until the next
  // SetConfiguration.
  void Reset();

  // Sets the eligible actors and solver type, enabling adaptive selection. If either changed,
  // restarts from Block Jacobi and returns true, so that the caller does not use the statistics of
  // the previous step. The actors must come in a stable order, as the island's descendants do:
  // their order changes only with the island's composition, which resets the policy.
  [[nodiscard]] bool SetConfiguration(
      DynamicArray<entt::entity> const& actors,
      LinearSolverType solverType);

  void UpdateFromPreviousStep(StepObservation const& observation, AmgCost const& amgCost);

  // Preconditioner type for the eligible actors, or std::nullopt if adaptive selection is disabled.
  [[nodiscard]] std::optional<PreconditionerType> GetSelectedType() const;

 private:
  enum class Mode {
    BlockJacobi,
    AmgTrial,
    Amg,
    BlockJacobiProbe,
    BlockJacobiAfterAmgDivergence,
  };

  static constexpr int kBjHighIterationStepsBeforeAmgTrial = 2;
  static constexpr int kAmgTrialSteps = 2;
  static constexpr int kBjProbeStepsToLeaveAmg = 2;
  static constexpr int kMinAmgStepsBetweenBjProbes = 32;
  static constexpr int kMaxAmgStepsBetweenBjProbes = 128;
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

  Mode _mode = Mode::BlockJacobi;
  int _numObservedSolveSteps = 0;
  int _amgStepsBetweenBjProbes = 0;
  int _amgStepsUntilBjProbe = 0;
  StepObservation _blockJacobiWindow;

  // AMG's last two observations, cleared when a trial starts.
  StepObservation _previousAmgObservation;
  StepObservation _lastAmgObservation;

  int _amgTrialCooldownStepsLeft = 0;

  // Block Jacobi iterations per linear solve from which observations count toward an AMG trial
  // during the cooldown, or 0 if none do.
  int64_t _bjIterationsForAmgRetry = 0;

  DynamicArray<entt::entity> _eligibleActors;

  // std::nullopt while adaptive selection is disabled.
  std::optional<LinearSolverType> _solverType;
};

// Whether the preconditioner type of an actor is selected adaptively when adaptive selection is
// enabled in its island.
[[nodiscard]] bool IsEligibleForAdaptivePreconditioner(
    entt::registry const& reg,
    entt::entity actor);

// Updates the adaptive selection from the island's previous step. Adaptive selection requires
// PreconditionerType::AdaptivePerActor, a linear solver that uses IslandOperators, and at least one
// eligible actor.
void UpdateIslandActorPreconditionerPolicy(
    entt::registry& reg,
    entt::entity island,
    CIslandDescendants const& descendants,
    KrylovSolverParams const& linearParams);

} // namespace mochi
