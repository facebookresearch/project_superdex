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
#include <mochi_core/solvers/linear_solver.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
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

  void ConcurrentSolve(
      ColumnVectorView<real const> input,
      ColumnVectorView<real> output,
      ParallelWorkerInfo const& worker) const override {
    bool const valid = worker.numWorkers >= 1 && worker.workerId >= 0 &&
        worker.workerId < worker.numWorkers && worker.rBegin >= 0 && worker.rBegin <= worker.rEnd &&
        worker.rEnd <= input.Rows();
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
  [[nodiscard]] int NumConcurrentSolveCalls() const {
    return _concurrentSolveCalls;
  }
  [[nodiscard]] bool HadInvalidConcurrentSolve() const {
    return _invalidConcurrentSolve;
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
  mutable std::atomic<int> _solveCalls{0};
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
};

static PcgProblem MakeProblem(bool identity = false) {
  Matrix<real> A = Matrix<real>::Zero(kParallelSize, kParallelSize);
  ColumnVector<real> expectedSolution(kParallelSize);
  DynamicArray<real> inversePrecDiagonal(kParallelSize);
  for (int row = 0; row < kParallelSize; ++row) {
    A(row, row) = identity ? 1_r : static_cast<real>(1 << (row % 4));
    inversePrecDiagonal[row] = 1_r / A(row, row);
    if (!identity && row + 1 < kParallelSize) {
      A(row, row + 1) = -0.1_r;
      A(row + 1, row) = -0.1_r;
    }
    expectedSolution[row] = 0.25_r + 0.1_r * static_cast<real>(row % 7);
  }
  ColumnVector<real> b = A * expectedSolution;
  return {
      .A = std::move(A),
      .b = std::move(b),
      .initialGuess = ColumnVector<real>::Zero(kParallelSize),
      .expectedSolution = std::move(expectedSolution),
      .inversePrecDiagonal = std::move(inversePrecDiagonal)};
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
  int serialSolveCalls;
  int concurrentSolveCalls;
  bool invalidConcurrentSolve;
};

template <template <typename, typename> class Criterion, typename Dot = krylov::UsualDot>
[[nodiscard]] static auto MakeCriterion(real relativeTolerance = kRelativeTolerance) {
  return Criterion<Dot, real>{relativeTolerance, 0_r, kDivergenceTolerance};
}

template <typename StopCriterion, typename Dot = krylov::UsualDot>
[[nodiscard]] static SolveResult RunParallelPcg(
    PcgProblem const& problem,
    StopCriterion stopCriterion,
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
      .serialSolveCalls = prec.NumSolveCalls(),
      .concurrentSolveCalls = prec.NumConcurrentSolveCalls(),
      .invalidConcurrentSolve = prec.HadInvalidConcurrentSolve()};
}

enum class CriterionKind { ResidualL2, PreconditionedResidualL2, PreconditionerInduced };
enum class DotKind { Usual, Scaled };
using ConvergenceParam = std::tuple<CriterionKind, DotKind, InitialGuessHint, bool>;

[[nodiscard]] static real ExpectedRelativeResidualNorm(
    CriterionKind criterion,
    PcgProblem const& problem,
    ColumnVector<real> const& residual) {
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
[[nodiscard]] static SolveResult
RunCriterion(CriterionKind criterion, PcgProblem const& problem, SolveOptions const& options) {
  switch (criterion) {
    case CriterionKind::ResidualL2:
      return RunParallelPcg(
          problem, MakeCriterion<krylov::StatusResidualL2, Dot>(), options, Dot{});
    case CriterionKind::PreconditionedResidualL2:
      return RunParallelPcg(
          problem, MakeCriterion<krylov::StatusPreconditionedResidualL2, Dot>(), options, Dot{});
    case CriterionKind::PreconditionerInduced:
      return RunParallelPcg(
          problem,
          MakeCriterion<krylov::StatusResidualPreconditionerInduced, Dot>(),
          options,
          Dot{});
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
      return RunCriterion<ScaledDot>(criterion, problem, options);
  }
  MOCHI_ASSERT(false, "Invalid dot type.");
  return {};
}

[[nodiscard]] static std::string GetConvergenceParamName(
    testing::TestParamInfo<ConvergenceParam> const& info) {
  static constexpr char const* kCriterionNames[]{
      "ResidualL2", "PreconditionedResidualL2", "PreconditionerInduced"};
  static constexpr char const* kDotNames[]{"UsualDot", "ScaledDot"};
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

  EXPECT_EQ(result.status.convergence, LinearSolverConvergenceStatus::Converged);
  EXPECT_GT(result.status.numIterDone, 2);
  ColumnVector<real> const residual = problem.b - problem.A * result.x;
  real const relativeResidual = residual.Norm() / problem.b.Norm();
  EXPECT_LE(relativeResidual, 6_r * kRelativeTolerance);
  EXPECT_NEAR(
      result.status.relativeResidualNorm,
      ExpectedRelativeResidualNorm(std::get<0>(GetParam()), problem, residual),
      5_r * kRelativeTolerance);
  EXPECT_FALSE(result.invalidConcurrentSolve);
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

TEST_F(ParallelPcgTest, KeepsConfiguredDotInstancesDistinct) {
  auto const problem = MakeProblem(/*identity*/ true);
  ASSERT_GE(krylov::parallel_pcg::GetNumParallelWorkers(problem.A), 3);
  auto const result = RunParallelPcg(
      problem,
      MakeCriterion<krylov::StatusPreconditionedResidualL2, ScaledDot>(0.2_r),
      {},
      ScaledDot{1_r});

  EXPECT_EQ(result.status.convergence, LinearSolverConvergenceStatus::Converged);
  EXPECT_EQ(result.status.numIterDone, 1);
  EXPECT_TRUE(mochi::test::NearEqualMatrices(result.x, problem.expectedSolution));
  EXPECT_FALSE(result.invalidConcurrentSolve);
}

TEST_F(ParallelPcgTest, ConvergesImmediatelyFromExactInitialGuess) {
  auto problem = MakeProblem();
  problem.initialGuess = problem.expectedSolution;
  ASSERT_GE(krylov::parallel_pcg::GetNumParallelWorkers(problem.A), 3);
  auto const result = RunParallelPcg(
      problem,
      MakeCriterion<krylov::StatusPreconditionedResidualL2>(),
      {.initialGuessHint = InitialGuessHint::Unknown});

  EXPECT_EQ(result.status.convergence, LinearSolverConvergenceStatus::Converged);
  EXPECT_EQ(result.status.numIterDone, 0);
  EXPECT_TRUE(mochi::test::NearEqualMatrices(result.x, problem.expectedSolution));
  EXPECT_FALSE(result.invalidConcurrentSolve);
}

TEST_F(ParallelPcgTest, ConvergesImmediatelyForZeroSystemWithPreconditionedResidual) {
  auto problem = MakeProblem();
  problem.b.SetZero();
  problem.expectedSolution.SetZero();
  ASSERT_GE(krylov::parallel_pcg::GetNumParallelWorkers(problem.A), 3);
  auto const result =
      RunParallelPcg(problem, MakeCriterion<krylov::StatusPreconditionedResidualL2>());

  EXPECT_EQ(result.status.convergence, LinearSolverConvergenceStatus::Converged);
  EXPECT_EQ(result.status.numIterDone, 0);
  EXPECT_TRUE(mochi::test::NearEqualMatrices(result.x, problem.expectedSolution));
  EXPECT_FALSE(result.invalidConcurrentSolve);
}

class ParallelPcgNonSpdTest : public ParallelPcgTest, public testing::WithParamInterface<bool> {};

TEST_P(ParallelPcgNonSpdTest, HonorsAbortPolicy) {
  auto problem = MakeProblem(/*identity*/ true);
  for (int row = 0; row < problem.A.Rows(); ++row) {
    problem.A(row, row) = -1_r;
    problem.expectedSolution[row] = -problem.b[row];
  }
  ASSERT_GE(krylov::parallel_pcg::GetNumParallelWorkers(problem.A), 3);
  bool const abortIfNotSpd = GetParam();
  auto const result = RunParallelPcg(
      problem, MakeCriterion<krylov::StatusResidualL2>(), {.abortIfNotSpd = abortIfNotSpd});

  EXPECT_EQ(
      result.status.convergence,
      abortIfNotSpd ? LinearSolverConvergenceStatus::Diverged
                    : LinearSolverConvergenceStatus::Converged);
  EXPECT_EQ(result.status.numIterDone, 1);
  EXPECT_TRUE(
      mochi::test::NearEqualMatrices(
          result.x, abortIfNotSpd ? problem.initialGuess : problem.expectedSolution));
  EXPECT_FALSE(result.invalidConcurrentSolve);
}

static std::string GetAbortPolicyName(testing::TestParamInfo<bool> const& info) {
  return info.param ? "Abort" : "Continue";
}

INSTANTIATE_TEST_SUITE_P(AbortPolicy, ParallelPcgNonSpdTest, testing::Bool(), GetAbortPolicyName);

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

  EXPECT_EQ(result.status.convergence, LinearSolverConvergenceStatus::Diverged);
  EXPECT_EQ(result.status.numIterDone, 1);
  EXPECT_NEAR(result.x[0], 1_r, kRelativeTolerance);
  EXPECT_NEAR(result.x[1], 0_r, kRelativeTolerance);
  EXPECT_FALSE(result.invalidConcurrentSolve);
}

TEST_F(ParallelPcgTest, StopsAtIterationLimit) {
  auto const problem = MakeProblem();
  ASSERT_GE(krylov::parallel_pcg::GetNumParallelWorkers(problem.A), 3);
  auto const result =
      RunParallelPcg(problem, MakeCriterion<krylov::StatusResidualL2>(), {.maxIter = 1});

  EXPECT_EQ(result.status.convergence, LinearSolverConvergenceStatus::Stopped);
  EXPECT_EQ(result.status.numIterDone, 1);
  EXPECT_FALSE(mochi::test::NearEqualMatrices(result.x, problem.initialGuess));
  EXPECT_FALSE(result.invalidConcurrentSolve);
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
  auto const result = RunParallelPcg(problem, MakeCriterion<krylov::StatusResidualL2>());

  EXPECT_EQ(result.status.convergence, LinearSolverConvergenceStatus::Converged);
  EXPECT_TRUE(mochi::test::NearEqualMatrices(result.x, problem.expectedSolution));
  EXPECT_GT(result.serialSolveCalls, 0);
  EXPECT_EQ(result.concurrentSolveCalls, 0);
}
