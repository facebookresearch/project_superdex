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

#include <mochi_core/linear_algebra/krylov/parallel_pcg.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/solvers/actor_preconditioner.h>
#include <mochi_core/solvers/linear_solver.h>
#include <mochi_core/solvers/per_actor_preconditioner.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/task_scheduler.h>

#include <gtest/gtest.h>

#include <atomic>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

using namespace mochi;

constexpr int kParallelSize = 400;
constexpr real kRelativeTolerance = 100_r * std::numeric_limits<real>::epsilon();
constexpr real kDivergenceTolerance = 1e10_r;

class ScaledDot {
 public:
  explicit constexpr ScaledDot(real scale = 100_r) : _scale(scale) {}

  template <typename Left, typename Right>
  [[nodiscard]] auto operator()(Left const& left, Right const& right) const {
    return _scale * left.Dot(right);
  }

  template <typename Vector>
  [[nodiscard]] auto NormSqr(Vector const& value) const {
    return (*this)(value, value);
  }

  [[nodiscard]] ScaledDot MiddleRows([[maybe_unused]] int rowStart, [[maybe_unused]] int numRows)
      const {
    return *this;
  }

 private:
  real const _scale;
};

class TrackingDiagonalPreconditioner final : public Preconditioner<real> {
 public:
  explicit TrackingDiagonalPreconditioner(DynamicArray<real> inverseDiagonal)
      : _inverseDiagonal(std::move(inverseDiagonal)) {}

  void Solve(ColumnVectorView<real const> input, ColumnVectorView<real> output) const override {
    ++_solveCalls;
    Apply(input, output, 0, input.Rows());
  }

  void PrepareConcurrentSolve(Span<int const> workerRowRanges) const override {
    _workerRowRanges.assign(workerRowRanges.begin(), workerRowRanges.end());
    _prepareCalls.fetch_add(1, std::memory_order_relaxed);
  }

  void ConcurrentSolve(
      ColumnVectorView<real const> input,
      ColumnVectorView<real> output,
      ParallelWorkerInfo const& worker) const override {
    bool const valid = _prepareCalls.load(std::memory_order_relaxed) > 0 &&
        worker.numWorkers >= 1 && worker.workerId >= 0 && worker.workerId < worker.numWorkers &&
        isize(_workerRowRanges) == worker.numWorkers + 1 &&
        _workerRowRanges[worker.workerId] == worker.rBegin &&
        _workerRowRanges[worker.workerId + 1] == worker.rEnd;
    if (!valid) {
      _invalidConcurrentSolve = true;
      return;
    }
    Apply(input, output, worker.rBegin, worker.rEnd);
    ++_concurrentSolveCalls;
  }

  [[nodiscard]] constexpr PreconditionerType GetType() const override {
    return PreconditionerType::None;
  }

  [[nodiscard]] int NumSolveCalls() const {
    return _solveCalls;
  }
  [[nodiscard]] int NumPrepareCalls() const {
    return _prepareCalls;
  }
  [[nodiscard]] int NumConcurrentSolveCalls() const {
    return _concurrentSolveCalls;
  }
  [[nodiscard]] bool HadInvalidConcurrentSolve() const {
    return _invalidConcurrentSolve;
  }
  [[nodiscard]] DynamicArray<int> const& WorkerRowRanges() const {
    return _workerRowRanges;
  }

 private:
  void Apply(
      ColumnVectorView<real const> input,
      ColumnVectorView<real> output,
      int rowBegin,
      int rowEnd) const {
    for (int row = rowBegin; row < rowEnd; ++row) {
      output[row] = _inverseDiagonal[row] * input[row];
    }
  }

  DynamicArray<real> _inverseDiagonal;
  mutable DynamicArray<int> _workerRowRanges;
  mutable std::atomic<int> _solveCalls{0};
  mutable std::atomic<int> _prepareCalls{0};
  mutable std::atomic<int> _concurrentSolveCalls{0};
  mutable std::atomic<bool> _invalidConcurrentSolve{false};
};

class ParallelPcgTest : public testing::Test {
 protected:
  void SetUp() override {
    if (TaskScheduler::GetNumSupportedLogicalProcessors() < 3) {
      GTEST_SKIP() << "At least three logical processors are required.";
    }
    _scheduler.emplace(2);
  }

 private:
  std::optional<TaskScheduler> _scheduler;
};

struct PcgProblem {
  Matrix<real> A;
  ColumnVector<real> b;
  ColumnVector<real> initialGuess;
  ColumnVector<real> expectedSolution;
  DynamicArray<real> inversePrecDiagonal;
  DynamicArray<real> residualWeights;
};

static PcgProblem MakeProblem(bool identity = false) {
  Matrix<real> A = Matrix<real>::Zero(kParallelSize, kParallelSize);
  ColumnVector<real> expectedSolution(kParallelSize);
  DynamicArray<real> inversePrecDiagonal(kParallelSize);
  DynamicArray<real> residualWeights(kParallelSize);
  for (int row = 0; row < kParallelSize; ++row) {
    A(row, row) = identity ? 1_r : static_cast<real>(1 << (row % 4));
    inversePrecDiagonal[row] = 1_r / A(row, row);
    if (!identity && row + 1 < kParallelSize) {
      A(row, row + 1) = -0.1_r;
      A(row + 1, row) = -0.1_r;
    }
    expectedSolution[row] = 0.25_r + 0.1_r * static_cast<real>(row % 7);
    residualWeights[row] = 1_r + static_cast<real>(row % 3);
  }
  ColumnVector<real> b = A * expectedSolution;
  return {
      .A = std::move(A),
      .b = std::move(b),
      .initialGuess = ColumnVector<real>::Zero(kParallelSize),
      .expectedSolution = std::move(expectedSolution),
      .inversePrecDiagonal = std::move(inversePrecDiagonal),
      .residualWeights = std::move(residualWeights)};
}

struct SolveOptions {
  int maxIter = 20;
  bool abortIfNotSpd = false;
  bool usePolakRibiere = true;
  InitialGuessHint initialGuessHint = InitialGuessHint::Zero;
};

struct SolveResult {
  LinearSolverStatus status;
  ColumnVector<real> x;
  DynamicArray<int> workerRowRanges;
  int serialSolveCalls;
  int prepareCalls;
  int concurrentSolveCalls;
  bool invalidConcurrentSolve;
};

template <template <typename, typename> class Criterion, typename Dot = krylov::UsualDot>
[[nodiscard]] static auto MakeCriterion(real relativeTolerance = kRelativeTolerance) {
  return Criterion<Dot, real>{relativeTolerance, 0_r, kDivergenceTolerance};
}

[[nodiscard]] static auto MakeWeightedResidualCriterion(PcgProblem const& problem) {
  return krylov::StatusResidualL2<krylov::WeightedDot<real>, real>{
      kRelativeTolerance,
      0_r,
      kDivergenceTolerance,
      /*retainedDirections*/ 0,
      {MakeConstSpan(problem.residualWeights)}};
}

template <typename StopCriterion, typename Dot = krylov::UsualDot>
[[nodiscard]] static SolveResult RunParallelPcg(
    PcgProblem const& problem,
    StopCriterion&& stopCriterion,
    SolveOptions const& options = {},
    Dot dot = {}) {
  TrackingDiagonalPreconditioner prec(problem.inversePrecDiagonal);
  auto x = problem.initialGuess;
  auto xView = AsView(x);
  auto const status = krylov::ParallelPCG(
      problem.A,
      AsConstView(problem.b),
      xView,
      details::PrecApplyer<real>{prec},
      options.maxIter,
      stopCriterion,
      options.abortIfNotSpd,
      VerbosityLevel::Silent,
      options.usePolakRibiere,
      options.initialGuessHint,
      dot,
      krylov::MatrixFactoryType<ColumnVector<real>>{});
  return {
      .status = status,
      .x = std::move(x),
      .workerRowRanges = prec.WorkerRowRanges(),
      .serialSolveCalls = prec.NumSolveCalls(),
      .prepareCalls = prec.NumPrepareCalls(),
      .concurrentSolveCalls = prec.NumConcurrentSolveCalls(),
      .invalidConcurrentSolve = prec.HadInvalidConcurrentSolve()};
}

static void ExpectValidExecution(SolveResult const& result, int numRows) {
  EXPECT_FALSE(result.invalidConcurrentSolve);
  if (result.prepareCalls == 0) {
    EXPECT_TRUE(result.workerRowRanges.empty());
    EXPECT_EQ(0, result.concurrentSolveCalls);
    return;
  }
  EXPECT_EQ(0, result.serialSolveCalls);
  EXPECT_EQ(1, result.prepareCalls);
  EXPECT_GE(result.workerRowRanges.size(), 3);
  if (result.workerRowRanges.size() >= 3) {
    EXPECT_EQ(0, result.workerRowRanges.front());
    EXPECT_EQ(numRows, result.workerRowRanges.back());
  }
  EXPECT_GT(result.concurrentSolveCalls, 0);
}

enum class CriterionKind { ResidualL2, PreconditionedResidualL2, PreconditionerInduced };
enum class DotKind { Usual, Scaled, Weighted };
using ConvergenceParam = std::tuple<CriterionKind, DotKind, InitialGuessHint, bool>;

[[nodiscard]] static real ExpectedRelativeResidualNorm(
    CriterionKind criterion,
    PcgProblem const& problem,
    ColumnVector<real> const& residual,
    DotKind dot = DotKind::Usual) {
  if (dot == DotKind::Weighted) {
    krylov::WeightedDot<real> const weightedDot{MakeConstSpan(problem.residualWeights)};
    return Sqrt(weightedDot.NormSqr(residual) / weightedDot.NormSqr(problem.b));
  }
  if (criterion == CriterionKind::ResidualL2) {
    return residual.Norm() / problem.b.Norm();
  }

  auto preconditionedResidual = residual;
  auto preconditionedRhs = problem.b;
  for (int row = 0; row < residual.Rows(); ++row) {
    preconditionedResidual[row] *= problem.inversePrecDiagonal[row];
    preconditionedRhs[row] *= problem.inversePrecDiagonal[row];
  }
  if (criterion == CriterionKind::PreconditionedResidualL2) {
    return preconditionedResidual.Norm() / preconditionedRhs.Norm();
  }

  MOCHI_ASSERT(criterion == CriterionKind::PreconditionerInduced);
  return Sqrt(residual.Dot(preconditionedResidual) / problem.b.Dot(preconditionedRhs));
}

template <typename Dot>
[[nodiscard]] static SolveResult RunCriterion(
    CriterionKind criterion,
    PcgProblem const& problem,
    SolveOptions const& options,
    Dot dot = {}) {
  switch (criterion) {
    case CriterionKind::ResidualL2:
      return RunParallelPcg(problem, MakeCriterion<krylov::StatusResidualL2, Dot>(), options, dot);
    case CriterionKind::PreconditionedResidualL2:
      return RunParallelPcg(
          problem, MakeCriterion<krylov::StatusPreconditionedResidualL2, Dot>(), options, dot);
    case CriterionKind::PreconditionerInduced:
      return RunParallelPcg(
          problem, MakeCriterion<krylov::StatusResidualPreconditionerInduced, Dot>(), options, dot);
  }
  MOCHI_ASSERT(false, "Invalid stopping criterion.");
  return {};
}

[[nodiscard]] static SolveResult RunConfiguration(
    PcgProblem const& problem,
    ConvergenceParam const& param) {
  auto const [criterion, dot, initialGuessHint, usePolakRibiere] = param;
  SolveOptions const options{
      .usePolakRibiere = usePolakRibiere, .initialGuessHint = initialGuessHint};
  switch (dot) {
    case DotKind::Usual:
      return RunCriterion<krylov::UsualDot>(criterion, problem, options);
    case DotKind::Scaled:
      // The criterion owns a default-constructed ScaledDot with scale 100. Give the solver's
      // distinct ScaledDot scale 1 so the test detects incorrect reuse of a dot product across the
      // two instances.
      return RunCriterion<ScaledDot>(criterion, problem, options, ScaledDot{1_r});
    case DotKind::Weighted:
      // Only the residual criterion supports a dot product other than the solver's.
      MOCHI_ASSERT(criterion == CriterionKind::ResidualL2, "Invalid stopping criterion.");
      return RunParallelPcg(problem, MakeWeightedResidualCriterion(problem), options);
  }
  MOCHI_ASSERT(false, "Invalid dot type.");
  return {};
}

[[nodiscard]] static std::string GetConvergenceParamName(
    testing::TestParamInfo<ConvergenceParam> const& info) {
  static constexpr char const* kCriterionNames[]{
      "ResidualL2", "PreconditionedResidualL2", "PreconditionerInduced"};
  static constexpr char const* kDotNames[]{"UsualDot", "ScaledDot", "WeightedDot"};
  auto const [criterion, dot, initialGuessHint, usePolakRibiere] = info.param;
  return std::string{kCriterionNames[static_cast<int>(criterion)]} + "_" +
      kDotNames[static_cast<int>(dot)] + "_" +
      (initialGuessHint == InitialGuessHint::Zero ? "Zero" : "Unknown") + "_" +
      (usePolakRibiere ? "PolakRibiere" : "FletcherReeves");
}

class ParallelPcgConvergenceTest : public ParallelPcgTest,
                                   public testing::WithParamInterface<ConvergenceParam> {};

TEST_P(ParallelPcgConvergenceTest, SolvesKnownSystem) {
  auto problem = MakeProblem();
  ASSERT_GE(krylov::parallel_pcg::GetNumParallelWorkers(problem.A), 3);
  if (std::get<2>(GetParam()) == InitialGuessHint::Unknown) {
    problem.initialGuess = 0.25_r * problem.expectedSolution;
  }
  auto const result = RunConfiguration(problem, GetParam());

  EXPECT_EQ(LinearSolverConvergenceStatus::Converged, result.status.convergence);
  EXPECT_GT(result.status.numIterDone, 2);
  ColumnVector<real> const residual = problem.b - problem.A * result.x;
  real const relativeResidual = residual.Norm() / problem.b.Norm();
  EXPECT_LE(relativeResidual, 6_r * kRelativeTolerance);
  EXPECT_NEAR(
      ExpectedRelativeResidualNorm(
          std::get<0>(GetParam()), problem, residual, std::get<1>(GetParam())),
      result.status.relativeResidualNorm,
      5_r * kRelativeTolerance);
  ExpectValidExecution(result, problem.A.Rows());
  if (result.prepareCalls != 0) {
    bool const needsPrec = std::get<0>(GetParam()) != CriterionKind::ResidualL2;
    int const numPrecApplications = result.status.numIterDone +
        (needsPrec ? 1 + static_cast<int>(std::get<2>(GetParam()) == InitialGuessHint::Unknown)
                   : 0);
    EXPECT_EQ(
        numPrecApplications * (isize(result.workerRowRanges) - 1), result.concurrentSolveCalls);
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllOptions,
    ParallelPcgConvergenceTest,
    testing::Combine(
        testing::Values(
            CriterionKind::ResidualL2,
            CriterionKind::PreconditionedResidualL2,
            CriterionKind::PreconditionerInduced),
        testing::Values(DotKind::Usual, DotKind::Scaled),
        testing::Values(InitialGuessHint::Zero, InitialGuessHint::Unknown),
        testing::Bool()),
    GetConvergenceParamName);

INSTANTIATE_TEST_SUITE_P(
    WeightedResidual,
    ParallelPcgConvergenceTest,
    testing::Combine(
        testing::Values(CriterionKind::ResidualL2),
        testing::Values(DotKind::Weighted),
        testing::Values(InitialGuessHint::Zero, InitialGuessHint::Unknown),
        testing::Bool()),
    GetConvergenceParamName);

TEST_F(ParallelPcgTest, KeepsConfiguredDotInstancesDistinct) {
  auto const problem = MakeProblem(/*identity*/ true);
  ASSERT_GE(krylov::parallel_pcg::GetNumParallelWorkers(problem.A), 3);
  auto const result = RunParallelPcg(
      problem,
      MakeCriterion<krylov::StatusPreconditionedResidualL2, ScaledDot>(0.2_r),
      {},
      ScaledDot{1_r});

  EXPECT_EQ(LinearSolverConvergenceStatus::Converged, result.status.convergence);
  EXPECT_EQ(1, result.status.numIterDone);
  EXPECT_TRUE(mochi::test::NearEqualMatrices(result.x, problem.expectedSolution));
  ExpectValidExecution(result, problem.A.Rows());
}

TEST_F(ParallelPcgTest, ConvergesImmediatelyFromExactInitialGuess) {
  auto problem = MakeProblem();
  problem.initialGuess = problem.expectedSolution;
  ASSERT_GE(krylov::parallel_pcg::GetNumParallelWorkers(problem.A), 3);
  auto const result = RunParallelPcg(
      problem,
      MakeCriterion<krylov::StatusPreconditionedResidualL2>(),
      {.initialGuessHint = InitialGuessHint::Unknown});

  EXPECT_EQ(LinearSolverConvergenceStatus::Converged, result.status.convergence);
  EXPECT_EQ(0, result.status.numIterDone);
  EXPECT_TRUE(mochi::test::NearEqualMatrices(result.x, problem.expectedSolution));
  ExpectValidExecution(result, problem.A.Rows());
}

TEST_F(ParallelPcgTest, ConvergesImmediatelyForZeroSystemWithPreconditionedResidual) {
  auto problem = MakeProblem();
  problem.b.SetZero();
  problem.expectedSolution.SetZero();
  ASSERT_GE(krylov::parallel_pcg::GetNumParallelWorkers(problem.A), 3);
  auto const result =
      RunParallelPcg(problem, MakeCriterion<krylov::StatusPreconditionedResidualL2>());

  EXPECT_EQ(LinearSolverConvergenceStatus::Converged, result.status.convergence);
  EXPECT_EQ(0, result.status.numIterDone);
  EXPECT_TRUE(mochi::test::NearEqualMatrices(result.x, problem.expectedSolution));
  ExpectValidExecution(result, problem.A.Rows());
  if (result.prepareCalls != 0) {
    EXPECT_EQ(isize(result.workerRowRanges) - 1, result.concurrentSolveCalls);
  }
}

using NonSpdParam = std::tuple<bool, real>;

class ParallelPcgNonSpdTest : public ParallelPcgTest,
                              public testing::WithParamInterface<NonSpdParam> {};

TEST_P(ParallelPcgNonSpdTest, HonorsAbortPolicy) {
  auto const [abortIfNotSpd, diagonal] = GetParam();
  auto problem = MakeProblem(/*identity*/ true);
  for (int row = 0; row < problem.A.Rows(); ++row) {
    problem.A(row, row) = diagonal;
    problem.expectedSolution[row] = problem.b[row] / diagonal;
  }
  ASSERT_GE(krylov::parallel_pcg::GetNumParallelWorkers(problem.A), 3);
  auto const result = RunParallelPcg(
      problem, MakeCriterion<krylov::StatusResidualL2>(), {.abortIfNotSpd = abortIfNotSpd});

  EXPECT_EQ(
      abortIfNotSpd ? LinearSolverConvergenceStatus::Diverged
                    : LinearSolverConvergenceStatus::Converged,
      result.status.convergence);
  EXPECT_EQ(1, result.status.numIterDone);
  EXPECT_TRUE(
      mochi::test::NearEqualMatrices(
          result.x, abortIfNotSpd ? problem.initialGuess : problem.expectedSolution));
  ExpectValidExecution(result, problem.A.Rows());
}

static std::string GetAbortPolicyName(testing::TestParamInfo<NonSpdParam> const& info) {
  auto const [abortIfNotSpd, diagonal] = info.param;
  return std::string{abortIfNotSpd ? "Abort" : "Continue"} + (IsFinite(diagonal) ? "" : "OnNan");
}

INSTANTIATE_TEST_SUITE_P(
    AbortPolicy,
    ParallelPcgNonSpdTest,
    testing::Values(
        NonSpdParam{false, -1_r},
        NonSpdParam{true, -1_r},
        NonSpdParam{true, std::numeric_limits<real>::quiet_NaN()}),
    GetAbortPolicyName);

TEST_F(ParallelPcgTest, ReportsSingularPreconditionerBreakdown) {
  auto problem = MakeProblem(/*identity*/ true);
  problem.b.SetZero();
  problem.b[0] = 1_r;
  problem.b[1] = 1_r;
  problem.expectedSolution = problem.b;
  problem.inversePrecDiagonal = DynamicArray<real>(kParallelSize, 0_r);
  problem.inversePrecDiagonal[0] = 1_r;
  ASSERT_GE(krylov::parallel_pcg::GetNumParallelWorkers(problem.A), 3);
  auto const result = RunParallelPcg(problem, MakeCriterion<krylov::StatusResidualL2>());

  EXPECT_EQ(LinearSolverConvergenceStatus::Diverged, result.status.convergence);
  EXPECT_EQ(1, result.status.numIterDone);
  EXPECT_NEAR(1_r, result.x[0], kRelativeTolerance);
  EXPECT_NEAR(0_r, result.x[1], kRelativeTolerance);
  ExpectValidExecution(result, problem.A.Rows());
}

// The weighted residual criterion stops at its second iteration, whose status check is fused with
// the r^T z reduction. Both runs apply the preconditioner three times.
TEST_F(ParallelPcgTest, StopsAtIterationLimit) {
  auto problem = MakeProblem();
  problem.initialGuess = 0.25_r * problem.expectedSolution;
  ASSERT_GE(krylov::parallel_pcg::GetNumParallelWorkers(problem.A), 3);
  auto expectStopsAtLimit = [&](auto criterion,
                                auto solverDot,
                                int maxIter,
                                CriterionKind criterionKind,
                                DotKind dotKind) {
    auto const result = RunParallelPcg(
        problem,
        criterion,
        {.maxIter = maxIter, .initialGuessHint = InitialGuessHint::Unknown},
        solverDot);

    EXPECT_EQ(LinearSolverConvergenceStatus::Stopped, result.status.convergence);
    EXPECT_EQ(maxIter, result.status.numIterDone);
    EXPECT_FALSE(mochi::test::NearEqualMatrices(result.x, problem.initialGuess));
    auto const residual = problem.b - problem.A * result.x;
    EXPECT_NEAR(
        ExpectedRelativeResidualNorm(criterionKind, problem, residual, dotKind),
        result.status.relativeResidualNorm,
        kRelativeTolerance);
    EXPECT_EQ(criterion.GetLatestRelativeResidualNorm(), result.status.relativeResidualNorm);
    ExpectValidExecution(result, problem.A.Rows());
    if (result.prepareCalls != 0) {
      EXPECT_EQ(3 * (isize(result.workerRowRanges) - 1), result.concurrentSolveCalls);
    }
  };
  expectStopsAtLimit(
      MakeCriterion<krylov::StatusPreconditionedResidualL2, ScaledDot>(),
      ScaledDot{1_r},
      /*maxIter*/ 1,
      CriterionKind::PreconditionedResidualL2,
      DotKind::Usual);
  expectStopsAtLimit(
      MakeWeightedResidualCriterion(problem),
      krylov::UsualDot{},
      /*maxIter*/ 2,
      CriterionKind::ResidualL2,
      DotKind::Weighted);
}

TEST(ParallelPcg, FallsBackWhenWorkIsInsufficient) {
  constexpr int kSize = 4;
  TaskScheduler scheduler(1);
  Matrix<real> A(kSize, kSize);
  A.SetIdentity();
  ColumnVector<real> expectedSolution(kSize);
  expectedSolution.SetRandom(23);
  PcgProblem const problem{
      .A = std::move(A),
      .b = expectedSolution,
      .initialGuess = ColumnVector<real>::Zero(kSize),
      .expectedSolution = expectedSolution,
      .inversePrecDiagonal = DynamicArray<real>(kSize, 1_r)};
  ASSERT_LE(krylov::parallel_pcg::GetNumParallelWorkers(problem.A), 1);
  auto const result =
      RunParallelPcg(problem, MakeCriterion<krylov::StatusPreconditionedResidualL2>());

  EXPECT_EQ(LinearSolverConvergenceStatus::Converged, result.status.convergence);
  EXPECT_TRUE(mochi::test::NearEqualMatrices(result.x, problem.expectedSolution));
  EXPECT_EQ(2, result.serialSolveCalls);
  EXPECT_EQ(0, result.prepareCalls);
  EXPECT_EQ(0, result.concurrentSolveCalls);
}

TEST_F(ParallelPcgTest, SolvesWithMixedModePerActorPreconditioner) {
  constexpr int kBlockSize = 3;
  // Wider than any of the three equal worker ranges, so the single-worker actor writes rows that
  // other workers own.
  constexpr int kSingleWorkerActorSize = 720;
  // Costly enough that the planner gives the synchronized actor more than one worker.
  constexpr int kSynchronizedActorSize = 1200;
  constexpr int kActorSize = 120;
  constexpr int kNumRows = kSingleWorkerActorSize + kSynchronizedActorSize + kActorSize;

  Matrix<real> A = Matrix<real>::Zero(kNumRows, kNumRows);
  auto makeActorMatrix = [&A](int offset, int size) {
    auto actorMatrix = Matrix<real>::Zero(size, size);
    for (int row = 0; row < size; ++row) {
      actorMatrix(row, row) = 4_r + static_cast<real>(row % 5);
      if (row + kBlockSize < size) {
        actorMatrix(row, row + kBlockSize) = -1_r;
        actorMatrix(row + kBlockSize, row) = -1_r;
      }
    }
    for (int row = 0; row < size; ++row) {
      for (int col = 0; col < size; ++col) {
        A(offset + row, offset + col) = actorMatrix(row, col);
      }
    }
    return ToBlockSparseMatrix<kBlockSize>(actorMatrix, true);
  };
  // Last, so no worker boundary splits the independent-row actor off its row blocks.
  constexpr int kSynchronizedOffset = kSingleWorkerActorSize;
  constexpr int kIndependentOffset = kSingleWorkerActorSize + kSynchronizedActorSize;
  auto const singleWorkerMatrix = makeActorMatrix(0, kSingleWorkerActorSize);
  auto const independentMatrix = makeActorMatrix(kIndependentOffset, kActorSize);
  auto const synchronizedMatrix = makeActorMatrix(kSynchronizedOffset, kSynchronizedActorSize);
  ILU0ActorPrec<real, kBlockSize> singleWorkerPrec{
      {0, AnyMatrixView<real const>{AsConstView(singleWorkerMatrix)}, {}}};
  BlockJacobiActorPrec<real, kBlockSize> independentPrec{
      {kIndependentOffset, AnyMatrixView<real const>{AsConstView(independentMatrix)}, {}}};
  // Random node positions: collinear ones would decouple the AMG's rotations.
  ColumnVector<real> reference(kSynchronizedActorSize);
  reference.SetRandom(25);
  auto const displacement = ColumnVector<real>::Zero(kSynchronizedActorSize);
  RigidBodyModesAMGActorPrec<real> synchronizedPrec{
      {.offset = kSynchronizedOffset,
       .actorMatrix = AnyMatrixView<real const>{AsConstView(synchronizedMatrix)},
       .interactionMatrices = {},
       .nodePositions = {
           .reference = reference.GetConstSpan(), .displacement = displacement.GetConstSpan()}}};
  ASSERT_EQ(
      ActorPreconditionerParallelMode::SingleWorker,
      singleWorkerPrec.GetConcurrentSolveRequirements().mode);
  ASSERT_EQ(
      ActorPreconditionerParallelMode::IndependentRows,
      independentPrec.GetConcurrentSolveRequirements().mode);
  ASSERT_EQ(
      ActorPreconditionerParallelMode::SynchronizedTeam,
      synchronizedPrec.GetConcurrentSolveRequirements().mode);
  ASSERT_EQ(3, krylov::parallel_pcg::GetNumParallelWorkers(A));
  PerActorPrec<real> prec(
      {
          {0, kSingleWorkerActorSize, singleWorkerPrec},
          {kSynchronizedOffset, kSynchronizedActorSize, synchronizedPrec},
          {kIndependentOffset, kActorSize, independentPrec},
      },
      PreconditionerType::PerActor);

  ColumnVector<real> b(kNumRows);
  b.SetRandom(24);
  auto x = ColumnVector<real>::Zero(kNumRows);
  auto xView = AsView(x);
  auto criterion = MakeCriterion<krylov::StatusResidualL2>();
  auto const status = krylov::ParallelPCG(
      A, b, xView, details::PrecApplyer<real>{prec}, /*maxIter*/ 100, criterion);

  EXPECT_EQ(LinearSolverConvergenceStatus::Converged, status.convergence);
  ColumnVector<real> const residual = b - A * x;
  EXPECT_LE(residual.Norm() / b.Norm(), 6_r * kRelativeTolerance);
}
