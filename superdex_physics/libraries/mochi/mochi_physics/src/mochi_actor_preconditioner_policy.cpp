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

#include "mochi_actor_preconditioner_policy.h"

#include "mochi_common_components.h"
#include "mochi_island.h"
#include "mochi_snle.h"

#include <mochi_core/solvers/krylov_solver.h>
#include <mochi_core/solvers/linear_solver.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>

#include <compare>

namespace mochi {

namespace {
using StepObservation = CIslandActorPreconditionerPolicy::StepObservation;

struct EligibleActors {
  DynamicArray<entt::entity> entities;
  bool hasSoftActor = false;
};

// Cost of a linear solve that takes n iterations: fixedCost + n * iterationCost.
struct LinearSolveCost {
  int64_t fixedCost;
  int64_t iterationCost;
};

// Mean cost per linear solve of the window's linear solves, times scalePercent, kept exact as
// integerPart + remainder / numLinearSolves. Comparisons are thus exact and deterministic, and no
// product can overflow, whatever the iteration counts.
struct MeanCost {
  MeanCost(LinearSolveCost const& cost, StepObservation const& window, int64_t scalePercent)
      : numLinearSolves(window.numLinearSolves) {
    MOCHI_ASSERT_VERBOSE(numLinearSolves > 0, "A mean cost needs at least one linear solve.");
    int64_t const fixedCost = scalePercent * cost.fixedCost;
    int64_t const iterationCost = scalePercent * cost.iterationCost;
    int64_t const iterationSum = window.linearIterationSum;
    int64_t const fractionalCost = iterationCost * (iterationSum % numLinearSolves);
    integerPart = fixedCost + iterationCost * (iterationSum / numLinearSolves) +
        fractionalCost / numLinearSolves;
    remainder = fractionalCost % numLinearSolves;
  }

  friend std::strong_ordering operator<=>(MeanCost const& a, MeanCost const& b) {
    return a.integerPart != b.integerPart
        ? a.integerPart <=> b.integerPart
        : a.remainder * b.numLinearSolves <=> b.remainder * a.numLinearSolves;
  }

  int64_t integerPart = 0;
  int64_t remainder = 0;
  int64_t numLinearSolves = 1;
};
} // namespace

// iterationCost and updateCost are those of the rotation-aware per-actor AMG: medians of the costs
// measured on soft and shell samples with ParallelCG at 8 workers, where AMG costs relatively more
// than with CG. bjIterationsForAmgTrial is the calibrated trigger, above AMG's typical cost per
// linear solve in steps where Block Jacobi reaches it. The values are deliberately the same for
// every actor size and topology, so that the selection depends only on iteration counts.
static constexpr CIslandActorPreconditionerPolicy::AmgCost kSoftAmgCost{
    .iterationCost = 37,
    .updateCost = 320,
    .bjIterationsForAmgTrial = 85,
};
static constexpr CIslandActorPreconditionerPolicy::AmgCost kShellAmgCost{
    .iterationCost = 30,
    .updateCost = 210,
    .bjIterationsForAmgTrial = 70,
};

static constexpr LinearSolveCost kBlockJacobiCost{
    .fixedCost = 0,
    .iterationCost = CIslandActorPreconditionerPolicy::kCostUnitsPerBlockJacobiIteration,
};

// A preconditioner replaces the other only if it costs at most 100 - kSwitchMarginPercent percent
// as much. Comparisons scale the two costs by 100 and 100 - kSwitchMarginPercent, so they stay
// exact.
static constexpr int64_t kSwitchMarginPercent = 10;

// Compile-time test of the cost, trigger, and margin constants, used only by the static_asserts
// below. Failure means modeled AMG cost at one iteration per linear solve cannot meet the switch
// margin against Block Jacobi at the iteration threshold for starting an AMG trial.
[[nodiscard]] static constexpr bool CanKeepAmgTrialsStartedAtTrigger(
    CIslandActorPreconditionerPolicy::AmgCost const& amgCost) {
  return 100 * (amgCost.updateCost + amgCost.iterationCost) <= (100 - kSwitchMarginPercent) *
      amgCost.bjIterationsForAmgTrial *
      CIslandActorPreconditionerPolicy::kCostUnitsPerBlockJacobiIteration;
}
static_assert(CanKeepAmgTrialsStartedAtTrigger(kSoftAmgCost));
static_assert(CanKeepAmgTrialsStartedAtTrigger(kShellAmgCost));

// During the cooldown after a rejected trial, a Block Jacobi observation counts toward a new trial
// only if it stops or averages at least the iterations per linear solve of Block Jacobi before the
// trial plus kBjIncreaseForAmgRetryPercent percent, rounded down to whole iterations. None counts
// if Block Jacobi stopped before the trial. This bounds the time spent on a Block Jacobi that gets
// much slower after the rejection.
static constexpr int64_t kBjIncreaseForAmgRetryPercent = 30;

[[nodiscard]] static EligibleActors CollectEligibleActors(
    entt::registry const& reg,
    CIslandDescendants const& descendants) {
  EligibleActors actors;
  actors.entities.reserve(descendants.softActors.size() + descendants.shellActors.size());

  auto const visit = [&](entt::entity actor, bool isSoft) {
    if (!IsEligibleForAdaptivePreconditioner(reg, actor)) {
      return;
    }
    actors.entities.push_back(actor);
    actors.hasSoftActor |= isSoft;
  };
  for (auto actor : descendants.softActors) {
    visit(actor, /*isSoft=*/true);
  }
  for (auto actor : descendants.shellActors) {
    visit(actor, /*isSoft=*/false);
  }
  return actors;
}

// Whether the candidate preconditioner costs less than the incumbent by the switch margin,
// comparing the mean costs per linear solve of their windows. A stopped linear solve costs
// infinitely much: a stopped candidate never costs less, and a candidate that didn't stop always
// costs less than a stopped incumbent.
[[nodiscard]] static bool CostsLessByMargin(
    LinearSolveCost const& candidateCost,
    StepObservation const& candidate,
    LinearSolveCost const& incumbentCost,
    StepObservation const& incumbent) {
  if (candidate.hasStoppedLinearSolve || incumbent.hasStoppedLinearSolve) {
    return !candidate.hasStoppedLinearSolve;
  }
  return MeanCost(candidateCost, candidate, 100) <=
      MeanCost(incumbentCost, incumbent, 100 - kSwitchMarginPercent);
}

[[nodiscard]] static StepObservation SummarizePreviousStep(CIslandSolverStats const& islandStats) {
  static_assert(
      static_cast<int>(LinearSolverConvergenceStatus::Count) == 4,
      "Please update this function if LinearSolverConvergenceStatus changes.");
  StepObservation observation;
  for (auto const& stage : islandStats.stages) {
    MOCHI_ASSERT(
        stage.numIterDone == 0 ||
            stage.worstLinearConvergence != LinearSolverConvergenceStatus::None,
        "Invalid linear solver convergence status.");
    MOCHI_ASSERT_VERBOSE(stage.numLinearIterDone >= 0, "Iteration counts are nonnegative.");

    observation.linearIterationSum += stage.numLinearIterDone;
    observation.numLinearSolves += stage.numIterDone;
    observation.hasStoppedLinearSolve |=
        stage.worstLinearConvergence == LinearSolverConvergenceStatus::Stopped;
    observation.hasDivergedLinearSolve |=
        stage.worstLinearConvergence == LinearSolverConvergenceStatus::Diverged;
  }
  return observation;
}

void CIslandActorPreconditionerPolicy::ResetObservationWindow() {
  _numObservedSolveSteps = 0;
  _blockJacobiWindow = {};
}

void CIslandActorPreconditionerPolicy::MeasureBlockJacobi(StepObservation const& observation) {
  _blockJacobiWindow += observation;
}

void CIslandActorPreconditionerPolicy::MeasureAmg(StepObservation const& observation) {
  // Trials compare their own two observations, and probes the last two before them, so that neither
  // compares an empty or stale window.
  static_assert(kAmgTrialSteps == 2 && kAmgStepsBetweenBjProbes >= 2);
  _previousAmgObservation = _lastAmgObservation;
  _lastAmgObservation = observation;
}

void CIslandActorPreconditionerPolicy::Reset() {
  *this = CIslandActorPreconditionerPolicy{};
}

bool CIslandActorPreconditionerPolicy::SetConfiguration(
    DynamicArray<entt::entity> const& actors,
    LinearSolverType solverType) {
  if (_eligibleActors == actors && _solverType == solverType) {
    return false;
  }
  Reset();
  _eligibleActors = actors;
  _solverType = solverType;
  return true;
}

void CIslandActorPreconditionerPolicy::UpdateFromPreviousStep(
    StepObservation const& observation,
    AmgCost const& amgCost) {
  // An observation is empty if the previous step performed no linear solve, e.g., because every
  // integration stage terminated at the nonlinear solver's initial stopping check. The AMG trial
  // cooldown is measured in island steps, so it also advances on empty observations. The
  // observation windows measure solver behavior, so they only advance on observations with linear
  // solves.
  bool const amgTrialCooldownActive = _mode == Mode::BlockJacobi && _amgTrialCooldownStepsLeft > 0;
  if (amgTrialCooldownActive) {
    --_amgTrialCooldownStepsLeft;
  }
  if (observation.numLinearSolves == 0) {
    return;
  }

  if (observation.hasDivergedLinearSolve) {
    switch (_mode) {
      case Mode::BlockJacobi:
      case Mode::BlockJacobiProbe:
        EnterAmg();
        return;
      case Mode::AmgTrial:
      case Mode::Amg:
        EnterBlockJacobiAfterAmgDivergence();
        return;
      case Mode::BlockJacobiAfterAmgDivergence:
        return;
    }
  }

  LinearSolveCost const amgSolveCost{
      .fixedCost = amgCost.updateCost,
      .iterationCost = amgCost.iterationCost,
  };

  switch (_mode) {
    case Mode::BlockJacobi: {
      int64_t const bjIterationsForAmgTrial =
          amgTrialCooldownActive ? _bjIterationsForAmgRetry : amgCost.bjIterationsForAmgTrial;
      if (bjIterationsForAmgTrial == 0) {
        return;
      }
      if (observation.hasStoppedLinearSolve ||
          observation.linearIterationSum >= bjIterationsForAmgTrial * observation.numLinearSolves) {
        ++_numObservedSolveSteps;
        MeasureBlockJacobi(observation);
        if (_numObservedSolveSteps >= kBjHighIterationStepsBeforeAmgTrial) {
          EnterAmgTrial();
        }
      } else {
        ResetObservationWindow();
      }
      return;
    }

    case Mode::AmgTrial: {
      ++_numObservedSolveSteps;
      MeasureAmg(observation);
      if (_numObservedSolveSteps < kAmgTrialSteps) {
        return;
      }

      // Divergence does not enter this comparison: Block Jacobi divergence before the trial commits
      // AMG directly, while AMG-trial divergence selects Block Jacobi recovery.
      StepObservation amgWindow = _previousAmgObservation;
      amgWindow += _lastAmgObservation;
      if (CostsLessByMargin(amgSolveCost, amgWindow, kBlockJacobiCost, _blockJacobiWindow)) {
        EnterAmg();
      } else {
        int64_t const bjIterationsForAmgRetry = _blockJacobiWindow.hasStoppedLinearSolve
            ? 0
            : (100 + kBjIncreaseForAmgRetryPercent) * _blockJacobiWindow.linearIterationSum /
                (100 * _blockJacobiWindow.numLinearSolves);
        EnterBlockJacobiCooldown(bjIterationsForAmgRetry);
      }
      return;
    }

    case Mode::Amg:
      MeasureAmg(observation);
      // TODO: Consider keeping the AMG hierarchy alive during Block Jacobi probes.
      if (--_amgStepsUntilBjProbe == 0) {
        _mode = Mode::BlockJacobiProbe;
        ResetObservationWindow();
      }
      return;

    case Mode::BlockJacobiProbe: {
      // TODO: Consider capping the linear-solver iterations of probe steps to bound the cost of
      // failed probes.

      // The test covers all of the probe's observations so far, so that a failing first
      // observation ends the probe.
      ++_numObservedSolveSteps;
      MeasureBlockJacobi(observation);
      StepObservation amgWindow = _previousAmgObservation;
      amgWindow += _lastAmgObservation;
      if (!CostsLessByMargin(kBlockJacobiCost, _blockJacobiWindow, amgSolveCost, amgWindow)) {
        EnterAmg();
        return;
      }
      if (_numObservedSolveSteps >= kBjProbeStepsToLeaveAmg) {
        EnterBlockJacobi();
      }
      return;
    }

    case Mode::BlockJacobiAfterAmgDivergence:
      // AMG diverged, so the cooldown allows no new AMG trial.
      EnterBlockJacobiCooldown(0);
      return;
  }
}

void CIslandActorPreconditionerPolicy::EnterAmgTrial() {
  MOCHI_ASSERT_VERBOSE(_mode == Mode::BlockJacobi, "AMG trial can only start from Block Jacobi.");
  _mode = Mode::AmgTrial;
  _numObservedSolveSteps = 0;
  _previousAmgObservation = {};
  _lastAmgObservation = {};
}

std::optional<PreconditionerType> CIslandActorPreconditionerPolicy::GetSelectedType() const {
  if (!_solverType) {
    return std::nullopt;
  }
  return _mode == Mode::Amg || _mode == Mode::AmgTrial ? PreconditionerType::AMG
                                                       : PreconditionerType::BlockJacobi;
}

void CIslandActorPreconditionerPolicy::EnterBlockJacobi() {
  _mode = Mode::BlockJacobi;
  ResetObservationWindow();
  _amgStepsUntilBjProbe = 0;
  _amgTrialCooldownStepsLeft = 0;
  _bjIterationsForAmgRetry = 0;
}

void CIslandActorPreconditionerPolicy::EnterBlockJacobiCooldown(int64_t bjIterationsForAmgRetry) {
  EnterBlockJacobi();
  _amgTrialCooldownStepsLeft = kAmgTrialCooldownSteps;
  _bjIterationsForAmgRetry = bjIterationsForAmgRetry;
}

void CIslandActorPreconditionerPolicy::EnterAmg() {
  _mode = Mode::Amg;
  ResetObservationWindow();
  _amgStepsUntilBjProbe = kAmgStepsBetweenBjProbes;
  _amgTrialCooldownStepsLeft = 0;
}

void CIslandActorPreconditionerPolicy::EnterBlockJacobiAfterAmgDivergence() {
  EnterBlockJacobi();
  _mode = Mode::BlockJacobiAfterAmgDivergence;
}

bool IsEligibleForAdaptivePreconditioner(entt::registry const& reg, entt::entity actor) {
  auto const* actorSnle = reg.try_get<CActorSnle const>(actor);
  return actorSnle && actorSnle->useInSolver && !actorSnle->UseReduced() &&
      !actorSnle->GetActivePreconditionerTypeOverride() &&
      reg.any_of<TagSoftActor, TagShellActor>(actor);
}

void UpdateIslandActorPreconditionerPolicy(
    entt::registry& reg,
    entt::entity island,
    CIslandDescendants const& descendants,
    KrylovSolverParams const& linearParams) {
  auto& policy = reg.get<CIslandActorPreconditionerPolicy>(island);

  bool const enabled = linearParams.preconditionerType == PreconditionerType::AdaptivePerActor &&
      details::UsesPerActorPreconditioners(linearParams);
  if (!enabled) {
    policy.Reset();
    return;
  }
  MOCHI_ASSERT(
      linearParams.preconditionerLifespan == 1,
      "Adaptive island preconditioner selection requires preconditionerLifespan = 1.");

  EligibleActors const actors = CollectEligibleActors(reg, descendants);
  if (actors.entities.empty()) {
    policy.Reset();
    return;
  }
  if (!policy.SetConfiguration(actors.entities, linearParams.solverType)) {
    // Mixed islands use the soft actors' AMG cost, the higher one, conservatively biasing toward
    // Block Jacobi.
    static_assert(
        kSoftAmgCost.updateCost > kShellAmgCost.updateCost &&
            kSoftAmgCost.iterationCost > kShellAmgCost.iterationCost &&
            kSoftAmgCost.bjIterationsForAmgTrial > kShellAmgCost.bjIterationsForAmgTrial,
        "The soft actors' AMG cost must be the higher one.");
    policy.UpdateFromPreviousStep(
        SummarizePreviousStep(reg.get<CIslandSolverStats const>(island)),
        actors.hasSoftActor ? kSoftAmgCost : kShellAmgCost);
  }
}

} // namespace mochi
