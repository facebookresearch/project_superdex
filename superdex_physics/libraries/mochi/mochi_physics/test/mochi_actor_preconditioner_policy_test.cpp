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

#include <mochi_physics/src/mochi_actor_preconditioner_policy.h>

#include "mochi_physics_test_fixture.h"

#include <mochi_core/linear_algebra/any_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/solvers/krylov_solver.h>
#include <mochi_core/solvers/newton_solver_status.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_physics/src/mochi_common_components.h>
#include <mochi_physics/src/mochi_island.h>
#include <mochi_physics/src/mochi_snle.h>
#include <mochi_physics/src/mochi_solve.h>

#include <gtest/gtest.h>

#include <initializer_list>
#include <optional>
#include <utility>
#include <vector>

namespace mochi::test::actor_preconditioner_policy {

static constexpr PreconditionerType kBlockJacobi = PreconditionerType::BlockJacobi;
static constexpr PreconditionerType kAmg = PreconditionerType::AMG;
static constexpr int kExpectedSoftBjIterationsForAmgTrial = 85;
static constexpr int kExpectedShellBjIterationsForAmgTrial = 70;
static constexpr int kExpectedMinAmgStepsBetweenBjProbes = 32;
static constexpr int kExpectedMaxAmgStepsBetweenBjProbes = 128;
static constexpr int kExpectedAmgTrialCooldownSteps = 50;

static LinearSolverStatus Converged(int iterations) {
  return {
      .numIterDone = iterations,
      .convergence = LinearSolverConvergenceStatus::Converged,
  };
}

// One iteration, as with maxIter = 1, so that the stop, not the iteration count, decides.
static LinearSolverStatus Stopped() {
  return {
      .numIterDone = 1,
      .convergence = LinearSolverConvergenceStatus::Stopped,
  };
}

static LinearSolverStatus Diverged() {
  return {
      .numIterDone = 1,
      .convergence = LinearSolverConvergenceStatus::Diverged,
  };
}

using StageStatuses = DynamicArray<LinearSolverStatus>;
using StepStatuses = std::initializer_list<StageStatuses>;

using OptionalType = std::optional<PreconditionerType>;

class PolicyScenario {
 public:
  explicit PolicyScenario(int numSoftActors = 1, int numShellActors = 0) {
    _island = _reg.create();
    _reg.emplace<CIslandActorPreconditionerPolicy>(_island);
    _reg.emplace<CIslandSolverStats>(_island);
    _linearParams.preconditionerType = PreconditionerType::AdaptivePerActor;
    _linearParams.solverType = LinearSolverType::CG;
    _linearParams.preconditionerLifespan = 1;

    for (int i = 0; i < numSoftActors; ++i) {
      AddSoftActor();
    }
    for (int i = 0; i < numShellActors; ++i) {
      AddShellActor();
    }

    Update();
  }

  entt::entity AddSoftActor(OptionalType typeOverride = std::nullopt) {
    return AddActor<TagSoftActor>(
        _descendants.softActors, typeOverride, /*useInSolver=*/true, /*useReduced=*/false);
  }

  entt::entity AddShellActor() {
    return AddActor<TagShellActor>(
        _descendants.shellActors, std::nullopt, /*useInSolver=*/true, /*useReduced=*/false);
  }

  entt::entity AddSoftActorNotInSolver() {
    return AddActor<TagSoftActor>(
        _descendants.softActors, std::nullopt, /*useInSolver=*/false, /*useReduced=*/false);
  }

  entt::entity AddReducedSoftActor() {
    return AddActor<TagSoftActor>(
        _descendants.softActors, std::nullopt, /*useInSolver=*/true, /*useReduced=*/true);
  }

  // An actor that is neither soft nor shell, e.g. a rod.
  entt::entity AddOtherActor() {
    auto const actor = _reg.create();
    _reg.emplace<CActorSnle>(actor, AnyMatrix<real>{Matrix<real>(1, 1)});
    return actor;
  }

  void ReplaceWithSoftActor() {
    _descendants.softActors.clear();
    _descendants.shellActors.clear();
    AddSoftActor();
  }

  void SetTypeOverride(entt::entity actor, PreconditionerType type) {
    _reg.get<CActorSnle>(actor).SetActivePreconditionerTypeOverride(type);
  }

  void SetUseReduced(entt::entity actor, bool useReduced) {
    auto& actorSnle = _reg.get<CActorSnle>(actor);
    if (useReduced) {
      actorSnle.EnableReduced(AnyMatrix<real>{Matrix<real>(1, 1)});
    } else {
      actorSnle.DisableReduced();
    }
  }

  void SetSolverType(LinearSolverType type) {
    _linearParams.solverType = type;
  }

  void SetPreconditionerType(PreconditionerType type) {
    _linearParams.preconditionerType = type;
  }

  entt::entity GetFirstSoftActor() const {
    return _descendants.softActors.at(0);
  }

  void Step(LinearSolverStatus status, OptionalType expectedType) {
    Step(StepStatuses{{status}}, expectedType);
  }

  void Step(StepStatuses stages, OptionalType expectedType) {
    auto& stats = _reg.get<CIslandSolverStats>(_island);
    stats.stages.clear();
    for (auto const& statuses : stages) {
      NewtonSolverStatus<real> newtonStatus;
      for (auto const& status : statuses) {
        ++newtonStatus.numIterDone;
        newtonStatus.totalNumLinearIterDone += status.numIterDone;
        newtonStatus.worstLinearConvergence =
            Max(newtonStatus.worstLinearConvergence, status.convergence);
      }
      stats.stages.push_back(StageSolverStats::FromNewtonSolverStatus(newtonStatus));
    }

    Update();
    ExpectType(expectedType);
  }

  void NoSolve(OptionalType expectedType) {
    Step(StepStatuses{StageStatuses{}}, expectedType);
  }

  void Repeat(LinearSolverStatus status, int count, OptionalType expectedType) {
    for (int i = 0; i < count; ++i) {
      Step(status, expectedType);
    }
  }

  void RepeatNoSolve(int count, OptionalType expectedType) {
    for (int i = 0; i < count; ++i) {
      NoSolve(expectedType);
    }
  }

  void EnterAmgTrial() {
    Step(Converged(200), kBlockJacobi);
    Step(Converged(200), kAmg);
  }

  // In a soft island, a 20-iteration AMG costs as much as 32 + 3.7 * 20 = 106 Block Jacobi
  // iterations, less than 90% of the trial's 200-iteration baseline.
  void EnterCommittedAmg() {
    EnterAmgTrial();
    Step(Converged(20), kAmg);
    Step(Converged(20), kAmg);
  }

  void EnterBlockJacobiProbe() {
    EnterCommittedAmg();
    ProbeAfter(kExpectedMinAmgStepsBetweenBjProbes);
  }

  // Expects the given number of committed AMG observations to start a Block Jacobi probe.
  void ProbeAfter(int amgSteps, int amgIterations = 20) {
    Repeat(Converged(amgIterations), amgSteps - 1, kAmg);
    Step(Converged(amgIterations), kBlockJacobi);
  }

  void ExpectActorType(entt::entity actor, OptionalType expectedType) const {
    EXPECT_EQ(expectedType, GetActorPreconditionerType(_reg, _island, actor));
  }

 private:
  template <typename TagActorFamily>
  entt::entity AddActor(
      std::vector<entt::entity>& family,
      OptionalType typeOverride,
      bool useInSolver,
      bool useReduced) {
    auto const actor = _reg.create();
    if (useReduced) {
      _reg.emplace<CActorSnle>(
          actor, AnyMatrix<real>{Matrix<real>(1, 1)}, /*fullSize*/ 1, typeOverride);
    } else {
      _reg.emplace<CActorSnle>(actor, AnyMatrix<real>{Matrix<real>(1, 1)}, typeOverride);
    }
    _reg.get<CActorSnle>(actor).useInSolver = useInSolver;
    _reg.emplace<TagActorFamily>(actor);
    family.push_back(actor);
    return actor;
  }

  void Update() {
    UpdateIslandActorPreconditionerPolicy(_reg, _island, _descendants, _linearParams);
  }

  void ExpectType(OptionalType expectedType) const {
    EXPECT_EQ(
        expectedType, _reg.get<CIslandActorPreconditionerPolicy const>(_island).GetSelectedType());

    auto const expectEligibleActors = [&](std::vector<entt::entity> const& actors) {
      for (auto const actor : actors) {
        if (IsEligibleForAdaptivePreconditioner(_reg, actor)) {
          ExpectActorType(actor, expectedType);
        }
      }
    };
    expectEligibleActors(_descendants.softActors);
    expectEligibleActors(_descendants.shellActors);
  }

  entt::registry _reg;
  entt::entity _island = entt::null;
  CIslandDescendants _descendants;
  KrylovSolverParams _linearParams;
};

TEST(ActorPreconditionerPolicy, EntryAndAmgTrialTransitions) {
  // In a soft island, observations that stop or average at least
  // kExpectedSoftBjIterationsForAmgTrial Block Jacobi iterations per linear solve count toward an
  // AMG trial, and other observations with linear solves discard them. The trial then compares
  // with a finite baseline, against which a 100-iteration AMG, which costs as much as
  // 32 + 3.7 * 100 = 402 Block Jacobi iterations, loses.
  {
    PolicyScenario scenario;
    scenario.Step({{Stopped(), Converged(330)}}, kBlockJacobi);
    scenario.Step(
        {{Converged(kExpectedSoftBjIterationsForAmgTrial),
          Converged(kExpectedSoftBjIterationsForAmgTrial - 1)}},
        kBlockJacobi);
    scenario.Step(
        {{Converged(kExpectedSoftBjIterationsForAmgTrial - 1),
          Converged(kExpectedSoftBjIterationsForAmgTrial + 1)}},
        kBlockJacobi);
    scenario.Step(Converged(kExpectedSoftBjIterationsForAmgTrial), kAmg);
    scenario.Step(Converged(100), kAmg);
    scenario.Step(Converged(100), kBlockJacobi);
  }

  // Compare solve-weighted mean costs, pooled across integration stages, whose solve counts and
  // step means differ, and commit AMG if it costs at most 90% of Block Jacobi: Block Jacobi
  // averages 600 / 3 = 200 iterations per linear solve, and a trial averaging 120 / 3 = 40 AMG
  // iterations costs as much as 32 + 3.7 * 40 = 180.
  {
    PolicyScenario scenario;
    scenario.Step(Converged(180), kBlockJacobi);
    scenario.Step({{Converged(210)}, {Converged(210)}}, kAmg);
    scenario.Step(Converged(60), kAmg);
    scenario.Step({{Converged(20), Converged(40)}}, kAmg);
  }

  // Reject an AMG trial that costs just more than 90% of Block Jacobi: 261 / 11 AMG iterations cost
  // as much as 32 + 3.7 * 261 / 11 = 1317.7 / 11 Block Jacobi iterations, and 90% of 1331 / 10 is
  // 1317.69 / 11. Only the remainders of the exact mean costs differ. Steps without linear solves
  // don't count toward the trial. In the cooldown that follows, observations count toward a new
  // trial from 130% of 1331 / 10, rounded down to 173 iterations.
  {
    PolicyScenario scenario;
    scenario.Step(
        {{Converged(133), Converged(133)}, {Converged(133), Converged(133), Converged(133)}},
        kBlockJacobi);
    scenario.Step(
        {{Converged(133), Converged(133), Converged(133), Converged(133), Converged(134)}}, kAmg);
    scenario.Step(
        {{Converged(26), Converged(26), Converged(26), Converged(26), Converged(26)}}, kAmg);
    scenario.NoSolve(kAmg);
    scenario.Step(
        {{Converged(22),
          Converged(22),
          Converged(22),
          Converged(22),
          Converged(22),
          Converged(21)}},
        kBlockJacobi);
    scenario.Step(Converged(172), kBlockJacobi);
    scenario.Step(Converged(173), kBlockJacobi);
    scenario.Step(Converged(173), kAmg);
  }

  // Any finite AMG trial succeeds against an infinite stopped Block Jacobi baseline, which later
  // finite stages and observations don't change.
  {
    PolicyScenario scenario;
    scenario.Step({{Stopped()}, {Converged(200)}}, kBlockJacobi);
    scenario.NoSolve(kBlockJacobi);
    scenario.Step(Converged(200), kAmg);
    scenario.Step(Converged(500), kAmg);
    scenario.Step(Converged(500), kAmg);
  }

  // A stopped AMG trial loses at once, even against a stopped Block Jacobi baseline: when both cost
  // infinitely much, the incumbent stays. In the cooldown that follows, stopped observations start
  // a new trial after a finite baseline, but nothing does after a stopped one.
  for (auto const& [firstBaselineStatus, retryType] :
       {std::pair{Converged(200), kAmg}, std::pair{Stopped(), kBlockJacobi}}) {
    for (int const convergedTrialObservations : {0, 1}) {
      PolicyScenario scenario;
      scenario.Step(firstBaselineStatus, kBlockJacobi);
      scenario.Step(Converged(200), kAmg);
      scenario.Repeat(Converged(1), convergedTrialObservations, kAmg);
      scenario.Step(Stopped(), kBlockJacobi);
      scenario.Repeat(Converged(200), 2, kBlockJacobi);
      scenario.Step(Stopped(), kBlockJacobi);
      scenario.Step(Stopped(), retryType);
    }
  }
}

TEST(ActorPreconditionerPolicy, CommittedAmgAndBlockJacobiProbeTransitions) {
  // A probe leaves AMG if Block Jacobi costs at most 90% of AMG in its last two observations, over
  // all of the probe's observations. After AMG observations averaging (30 + 50) / 2 = 40
  // iterations, which cost as much as 32 + 3.7 * 40 = 180 Block Jacobi iterations, a probe
  // averaging (160 + 164) / 2 = 162 iterations leaves AMG, and one averaging 162.5 doesn't. AMG's
  // earlier, cheaper observations don't count, and observations without linear solves count neither
  // toward the probe interval nor the probe.
  for (auto const& [secondProbeIterations, expectedType] :
       {std::pair{164, kBlockJacobi}, std::pair{165, kAmg}}) {
    PolicyScenario scenario;
    scenario.EnterCommittedAmg();
    scenario.Repeat(Converged(20), kExpectedMinAmgStepsBetweenBjProbes - 2, kAmg);
    scenario.NoSolve(kAmg);
    scenario.Step(Converged(30), kAmg);
    scenario.Step(Converged(50), kBlockJacobi);
    scenario.Step(Converged(160), kBlockJacobi);
    scenario.NoSolve(kBlockJacobi);
    scenario.Step(Converged(secondProbeIterations), expectedType);
  }

  // A probe returns to AMG at its first observation that stops, diverges, or makes Block Jacobi
  // cost more than 90% of AMG, and doubles the interval to the next probe, up to the maximum: after
  // 20-iteration AMG observations, Block Jacobi must average at most 0.9 * 106 = 95.4 iterations,
  // which 96 and (95 + 96) / 2 don't. A stopped AMG observation brings the next probe forward to
  // compare the two AMG observations after it, but never to start fewer than
  // kExpectedMinAmgStepsBetweenBjProbes observations with linear solves, each counted once, after
  // the previous one, and leaves the interval to later probes unchanged.
  for (auto const& failingStatus : {Stopped(), Converged(96), Diverged()}) {
    for (int const passingObservations : {0, 1}) {
      PolicyScenario scenario;
      scenario.EnterBlockJacobiProbe();
      scenario.Repeat(Converged(95), passingObservations, kBlockJacobi);
      scenario.Step(failingStatus, kAmg);
      scenario.ProbeAfter(2 * kExpectedMinAmgStepsBetweenBjProbes);
      scenario.Step(failingStatus, kAmg);
      scenario.ProbeAfter(4 * kExpectedMinAmgStepsBetweenBjProbes);
      scenario.Step(failingStatus, kAmg);
      scenario.ProbeAfter(kExpectedMaxAmgStepsBetweenBjProbes);
      scenario.Step(failingStatus, kAmg);
      scenario.NoSolve(kAmg);
      scenario.Step({{Stopped(), Converged(20)}}, kAmg);
      scenario.ProbeAfter(kExpectedMinAmgStepsBetweenBjProbes - 2);
      scenario.Step(failingStatus, kAmg);
      scenario.Repeat(Converged(20), kExpectedMinAmgStepsBetweenBjProbes, kAmg);
      scenario.Step(Stopped(), kAmg);
      scenario.ProbeAfter(2);
      scenario.Step(failingStatus, kAmg);
      scenario.ProbeAfter(kExpectedMaxAmgStepsBetweenBjProbes);
    }
  }

  // Stopped AMG observations in the last two before a probe keep it on schedule and make AMG cost
  // infinitely much in it: converged probe observations leave AMG however costly they are, while a
  // stopped one keeps AMG, the incumbent.
  for (auto const& [probeStatus, expectedType] :
       {std::pair{Converged(400), kBlockJacobi}, std::pair{Stopped(), kAmg}}) {
    PolicyScenario scenario;
    scenario.EnterCommittedAmg();
    scenario.Repeat(Converged(20), kExpectedMinAmgStepsBetweenBjProbes - 2, kAmg);
    scenario.Step(Stopped(), kAmg);
    scenario.Step(Stopped(), kBlockJacobi);
    scenario.Step(probeStatus, expectedType);
    scenario.Step(Converged(400), expectedType);
  }

  // After a failed probe, a probe that leaves AMG doesn't start an AMG trial cooldown, and
  // committing AMG after the trial restores the minimum interval to the next probe. A stopped AMG
  // observation brings the next probe forward only to kExpectedMinAmgStepsBetweenBjProbes
  // observations after the previous one, whatever happened in between: here a trial that commits
  // AMG, a stopped trial followed by Block Jacobi divergence, or a probe that leaves AMG followed
  // by Block Jacobi divergence.
  for (auto const& [rejectTrial, stopAmg] :
       {std::pair{false, false}, std::pair{false, true}, std::pair{true, true}}) {
    PolicyScenario scenario;
    scenario.EnterBlockJacobiProbe();
    scenario.Step(Converged(96), kAmg);
    scenario.ProbeAfter(2 * kExpectedMinAmgStepsBetweenBjProbes);
    scenario.Step(Converged(95), kBlockJacobi);
    scenario.Step(Converged(95), kBlockJacobi);
    scenario.Step(Converged(kExpectedSoftBjIterationsForAmgTrial), kBlockJacobi);
    scenario.Step(Converged(kExpectedSoftBjIterationsForAmgTrial), kAmg);
    if (rejectTrial) {
      scenario.Step(Stopped(), kBlockJacobi);
      scenario.Step(Diverged(), kAmg);
    } else {
      scenario.Step(Converged(10), kAmg);
      scenario.Step(Converged(10), kAmg);
    }
    if (stopAmg) {
      scenario.Step(Stopped(), kAmg);
      scenario.ProbeAfter(kExpectedMinAmgStepsBetweenBjProbes - 7);
    } else {
      scenario.ProbeAfter(kExpectedMinAmgStepsBetweenBjProbes);
    }
    scenario.Repeat(Converged(95), 2, kBlockJacobi);
    scenario.Step(Diverged(), kAmg);
    scenario.Step(Stopped(), kAmg);
    scenario.ProbeAfter(kExpectedMinAmgStepsBetweenBjProbes - 4);
  }
}

TEST(ActorPreconditionerPolicy, DivergenceAndStickyRecoveryTransitions) {
  // Divergence switches to the other preconditioner, except that after AMG diverges, here on the
  // committed AMG observation that would start a probe, two after the first stopped one, sticky
  // recovery retains Block Jacobi even if it also diverges.
  {
    PolicyScenario scenario;
    scenario.Step({{Diverged(), Converged(1), Diverged()}, {Converged(1), Stopped()}}, kAmg);
    scenario.Step(Stopped(), kAmg);
    scenario.Step(Stopped(), kAmg);
    scenario.Step(Diverged(), kBlockJacobi);
    scenario.NoSolve(kBlockJacobi);
    scenario.Step(Diverged(), kBlockJacobi);
    scenario.Step(Stopped(), kBlockJacobi);
    scenario.Step(Diverged(), kAmg);
  }

  // A non-diverged recovery solve starts an AMG trial cooldown in which no observation, not even a
  // stopped one, counts toward a trial. Block Jacobi divergence then enters AMG with the minimum
  // interval to the next probe, even after a failed probe doubled it. A stopped AMG observation
  // there brings the next probe forward only to kExpectedMinAmgStepsBetweenBjProbes observations
  // after the previous one, counting those in recovery and the cooldown.
  for (bool const stopAmg : {false, true}) {
    PolicyScenario scenario;
    scenario.EnterBlockJacobiProbe();
    scenario.Step(Converged(96), kAmg);
    scenario.Step(Diverged(), kBlockJacobi);
    scenario.Step(Stopped(), kBlockJacobi);
    scenario.Step(Stopped(), kBlockJacobi);
    scenario.Step(Stopped(), kBlockJacobi);
    scenario.Step(Diverged(), kAmg);
    if (stopAmg) {
      scenario.Step(Stopped(), kAmg);
      scenario.ProbeAfter(kExpectedMinAmgStepsBetweenBjProbes - 7);
    } else {
      scenario.ProbeAfter(kExpectedMinAmgStepsBetweenBjProbes);
    }
  }

  // Divergence during an AMG trial enters Block Jacobi recovery.
  {
    PolicyScenario scenario;
    scenario.EnterAmgTrial();
    scenario.Step({{Diverged()}, {Stopped()}}, kBlockJacobi);
    scenario.Step(Diverged(), kBlockJacobi);
    scenario.Step(Converged(1), kBlockJacobi);
    scenario.Step(Diverged(), kAmg);
  }
}

TEST(ActorPreconditionerPolicy, AmgTrialCooldownUsesIslandStepClock) {
  // 41 AMG iterations cost as much as 32 + 3.7 * 41 = 183.7 Block Jacobi iterations, more than 90%
  // of the 200-iteration baseline, so the rejected trial starts a cooldown in which observations
  // count toward a new trial from 260 iterations per linear solve, 30% above the baseline. Updates
  // 1-49 age the cooldown; update 50 reaches zero but still applies the 260-iteration threshold.
  {
    PolicyScenario scenario;
    scenario.EnterAmgTrial();
    scenario.Step(Converged(41), kAmg);
    scenario.Step(Converged(41), kBlockJacobi);
    scenario.RepeatNoSolve(kExpectedAmgTrialCooldownSteps - 1, kBlockJacobi);
    scenario.Step(Converged(200), kBlockJacobi);
    scenario.EnterAmgTrial();
  }
}

TEST(ActorPreconditionerPolicy, ActorFamilyCostsAndEligibility) {
  // In a shell island, observations count toward an AMG trial from
  // kExpectedShellBjIterationsForAmgTrial Block Jacobi iterations per linear solve. After
  // 10-iteration AMG observations, which cost as much as 21 + 3.0 * 10 = 51 Block Jacobi
  // iterations, a probe leaves AMG if Block Jacobi averages at most 0.9 * 51 = 45.9 iterations.
  {
    PolicyScenario scenario(/*numSoftActors*/ 0, /*numShellActors*/ 2);
    scenario.Step(Converged(kExpectedShellBjIterationsForAmgTrial - 1), kBlockJacobi);
    scenario.Step(Converged(kExpectedShellBjIterationsForAmgTrial), kBlockJacobi);
    scenario.Step(Converged(200), kAmg);
    scenario.Step(Converged(10), kAmg);
    scenario.Step(Converged(10), kAmg);
    scenario.ProbeAfter(kExpectedMinAmgStepsBetweenBjProbes, /*amgIterations=*/10);
    scenario.Step(Converged(46), kAmg);
    scenario.ProbeAfter(2 * kExpectedMinAmgStepsBetweenBjProbes, /*amgIterations=*/10);
    scenario.Step(Converged(45), kBlockJacobi);
    scenario.Step(Converged(46), kBlockJacobi);
  }

  // A mixed island takes the soft actors' AMG cost: observations just below
  // kExpectedSoftBjIterationsForAmgTrial don't count toward an AMG trial, and after 10-iteration
  // AMG observations, which cost as much as 32 + 3.7 * 10 = 69 Block Jacobi iterations, a probe
  // leaves AMG if Block Jacobi averages at most 0.9 * 69 = 62.1 iterations.
  {
    PolicyScenario scenario(/*numSoftActors*/ 2, /*numShellActors*/ 2);
    scenario.Step(Converged(kExpectedSoftBjIterationsForAmgTrial - 1), kBlockJacobi);
    scenario.Step(Converged(kExpectedSoftBjIterationsForAmgTrial - 1), kBlockJacobi);
    scenario.Step(Converged(kExpectedSoftBjIterationsForAmgTrial), kBlockJacobi);
    scenario.Step(Converged(200), kAmg);
    scenario.Step(Converged(10), kAmg);
    scenario.Step(Converged(10), kAmg);
    scenario.ProbeAfter(kExpectedMinAmgStepsBetweenBjProbes, /*amgIterations=*/10);
    scenario.Step(Converged(62), kBlockJacobi);
    scenario.Step(Converged(62), kBlockJacobi);
  }

  // Ineligible actors neither affect the AMG cost nor use the adaptive selection.
  {
    PolicyScenario scenario(/*numSoftActors*/ 0, /*numShellActors*/ 2);
    auto const disabledSoftActor = scenario.AddSoftActorNotInSolver();
    auto const reducedSoftActor = scenario.AddReducedSoftActor();
    auto const overriddenSoftActor = scenario.AddSoftActor(PreconditionerType::Jacobi);
    auto const otherActor = scenario.AddOtherActor();
    scenario.Step(Converged(kExpectedShellBjIterationsForAmgTrial), kBlockJacobi);
    scenario.Step(Converged(kExpectedShellBjIterationsForAmgTrial), kAmg);
    scenario.ExpectActorType(disabledSoftActor, std::nullopt);
    scenario.ExpectActorType(reducedSoftActor, std::nullopt);
    scenario.ExpectActorType(overriddenSoftActor, PreconditionerType::Jacobi);
    scenario.ExpectActorType(otherActor, std::nullopt);
  }
}

TEST(ActorPreconditionerPolicy, PolicyResetBehavior) {
  // An empty eligible set disables adaptive selection.
  {
    PolicyScenario scenario(/*numSoftActors*/ 0, /*numShellActors*/ 0);
    scenario.AddSoftActorNotInSolver();
    scenario.AddReducedSoftActor();
    scenario.Repeat(Converged(500), 3, std::nullopt);
  }

  // An eligible-set change resets the policy and discards stale solver evidence, including the last
  // probe, which no longer delays the probe after a stopped AMG observation.
  {
    PolicyScenario scenario;
    scenario.EnterBlockJacobiProbe();
    scenario.ReplaceWithSoftActor();
    scenario.Step(Diverged(), kBlockJacobi);
    scenario.Step(Diverged(), kAmg);
    scenario.Step(Stopped(), kAmg);
    scenario.ProbeAfter(2);
  }

  // Overriding the type of an eligible actor removes it from the eligible set.
  {
    PolicyScenario scenario(/*numSoftActors*/ 2);
    scenario.EnterCommittedAmg();
    auto const overriddenSoftActor = scenario.GetFirstSoftActor();
    scenario.SetTypeOverride(overriddenSoftActor, PreconditionerType::Jacobi);
    scenario.ExpectActorType(overriddenSoftActor, PreconditionerType::Jacobi);
    scenario.Step(Diverged(), kBlockJacobi);
    scenario.Step(Diverged(), kAmg);
  }

  // Switching the only eligible actor to its reduced representation disables adaptive selection,
  // and switching it back restarts it.
  {
    PolicyScenario scenario;
    scenario.EnterCommittedAmg();
    auto const softActor = scenario.GetFirstSoftActor();
    scenario.SetUseReduced(softActor, true);
    scenario.Step(Diverged(), std::nullopt);
    scenario.SetUseReduced(softActor, false);
    scenario.Step(Diverged(), kBlockJacobi);
    scenario.Step(Diverged(), kAmg);
  }

  // Changing the linear solver type resets the policy.
  {
    PolicyScenario scenario;
    scenario.EnterCommittedAmg();
    scenario.SetSolverType(LinearSolverType::GMRES);
    scenario.Step(Diverged(), kBlockJacobi);
    scenario.Step(Diverged(), kAmg);
  }

  // Direct and CUDA solvers disable adaptation.
  for (auto const solverType : {LinearSolverType::LDLT, LinearSolverType::CudaCG}) {
    PolicyScenario scenario;
    scenario.EnterCommittedAmg();
    scenario.SetSolverType(solverType);
    scenario.Step(Diverged(), std::nullopt);
  }

  // Adaptation requires the adaptive per-actor preconditioner.
  {
    PolicyScenario scenario;
    scenario.EnterCommittedAmg();
    scenario.SetPreconditionerType(PreconditionerType::PerActor);
    scenario.Step(Diverged(), std::nullopt);
    scenario.SetPreconditionerType(PreconditionerType::AdaptivePerActor);
    scenario.Step(Diverged(), kBlockJacobi);
    scenario.Step(Diverged(), kAmg);
  }
}

class ActorPreconditionerPolicyScene : public MochiSceneTestBase {
 protected:
  // With a single CG iteration, stopped Block Jacobi solves start an AMG trial on the third step.
  void CreateAdaptiveSoftActorScene() {
    auto solverParams = _scene->GetSolverParams();
    solverParams.linearSolver.solverType = LinearSolverType::CG;
    solverParams.linearSolver.preconditionerType = PreconditionerType::AdaptivePerActor;
    solverParams.linearSolver.maxIter = 1;
    _scene->SetSolverParams(solverParams, ExpectOK{});

    SoftActorParams actorParams;
    actorParams.shape = CreateUnitCubeTetMeshShape(_mochiContext);
    _actor = _scene->CreateSoftActor(actorParams, ExpectOK{});
  }

  void SetPreconditionerType(PreconditionerType type) {
    auto solverParams = _scene->GetSolverParams();
    solverParams.linearSolver.preconditionerType = type;
    _scene->SetSolverParams(solverParams, ExpectOK{});
  }

  // Steps the scene and expects the type of the actor preconditioner used in the step.
  void StepAndExpectType(PreconditionerType expectedType) {
    _scene->Step(0.01_r);
    auto const& actorPreconditioner =
        GetRegistry().get<CActorSnle>(GetEntity(_actor)).GetActivePreconditioner();
    ASSERT_NE(nullptr, actorPreconditioner);
    EXPECT_EQ(expectedType, actorPreconditioner->GetType());
  }

  // Selects Block Jacobi, Block Jacobi, and AMG in consecutive steps, which exercises the
  // production wiring: Newton linear-solver statuses reach the policy, and the island solve uses
  // the selected type.
  void StepIntoAmgTrial() {
    StepAndExpectType(kBlockJacobi);
    StepAndExpectType(kBlockJacobi);
    StepAndExpectType(kAmg);
  }

  Actor* _actor = nullptr;
};

TEST_F(ActorPreconditionerPolicyScene, SwitchingToPerActorDropsAdaptiveSelection) {
  ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
  CreateAdaptiveSoftActorScene();
  StepIntoAmgTrial();
  SetPreconditionerType(PreconditionerType::PerActor);
  StepAndExpectType(kBlockJacobi);
  SetPreconditionerType(PreconditionerType::AdaptivePerActor);
  StepIntoAmgTrial();
}

// The island's linear-solver iteration counts depend on all island members, so adding an ineligible
// actor to the island also restarts the adaptive selection.
TEST_F(ActorPreconditionerPolicyScene, IslandCompositionChangeResetsAdaptiveSelection) {
  ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
  _scene->SetForceSingleIsland(true);
  CreateAdaptiveSoftActorScene();
  StepAndExpectType(kBlockJacobi);
  StepAndExpectType(kBlockJacobi);

  RigidActorParams rigidParams;
  rigidParams.shape = CreateUnitCubeTetMeshShape(_mochiContext);
  rigidParams.colliderType = ColliderType::None;
  _scene->CreateRigidActor(rigidParams, ExpectOK{});
  // Without the reset, an AMG trial would start.
  StepAndExpectType(kBlockJacobi);
}

// Island state is not captured, so restoring must not resume the adaptive selection from before
// the restore. Refreshing a query that requires assembly restores the state internally, which must
// neither reset nor advance the selection.
TEST_F(ActorPreconditionerPolicyScene, RestoreResetsAdaptiveSelection) {
  ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
  CreateAdaptiveSoftActorScene();
  _actor->RegisterQuery(QueryType::ContactPoints, ExpectOK{});
  StateHandle const initial = _scene->CaptureState(ExpectOK{});
  StepAndExpectType(kBlockJacobi);
  StepAndExpectType(kBlockJacobi);
  _scene->UpdateQueries(ExpectOK{});
  StepAndExpectType(kAmg);
  StateHandle const first = _scene->CaptureState(ExpectOK{});
  StepAndExpectType(kBlockJacobi);

  _scene->RestoreState(initial, /*releaseImmediately=*/false, ExpectOK{});
  StepIntoAmgTrial();
  EXPECT_TRUE(_scene->IsEqualState(first, _scene->CaptureState(ExpectOK{})));
}

} // namespace mochi::test::actor_preconditioner_policy
